import os, sys
import imageio
import numpy as np
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F

from tqdm import trange

from src.dataset.load_pinf import load_pinf_frame_data

from src.network.hybrid_model import create_model

from src.renderer.occupancy_grid import init_occ_grid, update_occ_grid, update_static_occ_grid
from src.renderer.render_ray import render, render_path, prepare_rays

from src.utils.args import config_parser
from src.utils.training_utils import set_rand_seed, save_log
from src.utils.coord_utils import BBox_Tool, Voxel_Tool, jacobian3D
from src.utils.loss_utils import get_rendering_loss, get_velocity_loss, fade_in_weight, to8b
from src.utils.visualize_utils import den_scalar2rgb, vel2hsv, vel_uv2hsv
from src.renderer.occupancy_grid import OccupancyGrid, OccupancyGridDynamic

from visualize import VoxelTrans
from siren import load_SIREN_model, save_SIREN_model, SIREN
from ingp import INGP

import json

from test import visualize_all

from tqdm import tqdm
import random

import taichi as ti
from taichi_encoders.mgpcg import MGPCG_3

ti.init(arch=ti.cuda, device_memory_GB=12.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count

def train(args):
    boundary_types = ti.Matrix([[1, 1], [2, 2], [1, 1]], ti.i32)  # boundaries: 1 means Dirichlet, 2 means Neumann
    project_solver = MGPCG_3(boundary_types=boundary_types, N=[args.proj_size, args.proj_size, args.proj_size], base_level=3)
    logdir, writer, targetdir = save_log(args)
    
    with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
        # read render settings
        meta = json.load(fp)

        # read scene data
        voxel_tran = np.float32(meta['voxel_matrix'])
        voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1) # swap_zx
        voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
        
        ## apply manual scaling
        scene_scale = args.scene_scale
        voxel_scale = voxel_scale.copy() * scene_scale
        voxel_tran[:3,3] *= scene_scale
        train_video = meta['train_videos'][0]
        delta_t = 1.0/train_video['frame_num']
        t_info = np.float32([0.0, 1.0, 0.5, delta_t])
        
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    voxel_tran_inv = torch.Tensor(voxel_tran_inv)
    voxel_tran = torch.Tensor(voxel_tran)
    voxel_scale = torch.Tensor(voxel_scale)
    p_w_p_s = torch.eye(4)
    p_w_p_s[:3, :3] = voxel_tran[:3, :3] * voxel_scale.view(1, 3)
    p_w_p_s *= 0.5
    p_s_p_w = torch.inverse(p_w_p_s)
    p_w_p_s = p_w_p_s.to(device)
    p_s_p_w = p_s_p_w.to(device)
    my_voxel_tool = VoxelTrans(voxel_tran, voxel_tran_inv, voxel_scale, device)

    # Create Bbox model from smoke perspective
    bbox_model = None

    # this bbox in in the smoke simulation coordinate
    in_min = [float(_) for _ in args.bbox_min.split(",")]
    in_max = [float(_) for _ in args.bbox_max.split(",")]
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale, in_min, in_max)

    # Create model
    dens_model, optimizer, start = create_model(
        args=args, bbox_model=bbox_model, device=device
    )
    
    global_step = start

    # Prepare Voxel Sampling Tools for Image Summary (voxel_writer), Physical Priors (training_voxel), Data Priors Represented by D2V (den_p_all)
    # voxel_writer: to sample low resolution data for for image summary
    resX = args.vol_output_W
    resY = int(args.vol_output_W * float(voxel_scale[1]) / voxel_scale[0] + 0.5)
    resZ = int(args.vol_output_W * float(voxel_scale[2]) / voxel_scale[0] + 0.5)
    voxel_writer = Voxel_Tool(
        voxel_tran,
        voxel_tran_inv,
        voxel_scale,
        resZ,
        resY,
        resX,
        middleView="mid3",
        hybrid_neus="hybrid_neus" in args.net_model,
    )

    dens_model.voxel_writer = voxel_writer

    dens_model.iter_step = global_step
    dens_model.update_model(2, global_step)
    dens_model.eval()
    
    xs, ys, zs = torch.meshgrid(
        [
            torch.linspace(-1, 1, args.Nx + 1),
            torch.linspace(-1, 1, args.Ny + 1),
            torch.linspace(-1, 1, args.Nz + 1),
        ],
        indexing="ij",
    )
    center_location = torch.stack([xs, ys, zs], dim=-1).to(device)
    center_location = center_location[:-1, :-1, :-1, :]
    bias = torch.tensor(
        [1.0 / args.Nx, 1.0 / args.Ny, 1.0 / args.Nz],
        dtype=torch.float32,
        device=device,
    )
    center_location = center_location + bias.view(1, 1, 1, 3)
    center_location = center_location.view(-1, 3)

    t_list = np.arange(t_info[0], t_info[1], t_info[-1])
    frame_num = t_list.shape[0]

    batchsize = args.batchsize

    if args.load_path == "None":
        model = INGP(args, device)
    else:
        model = INGP.load(args.load_path, args, device)
    # optim = torch.optim.Adam(lr=args.lrate, params=model.parameters())
    gamma = 1e-3 ** (1.0 / args.num_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma, verbose=True)
    
    if args.siren_model_path != "None":
        siren_model_path = args.siren_model_path
        siren_model= load_SIREN_model(siren_model_path).to(device)
        siren_model.eval()

    train_loss = AverageMeter()
    adv_loss, regular_loss, div_loss, nse_loss, ingp_loss, proj_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    
    # dataset = CoordinateDataset(args.Nx, args.Ny, args.Nz, frame_num)
    # print(1)
    # dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    # print(2)
    
    # t_coords = torch.linspace(-1.0 + 1.0 / frame_num, 1.0 - 1.0 / frame_num, frame_num)
    # d_coords = torch.linspace(-1.0 + 1.0 / args.Nz, 1.0 - 1.0 / args.Nz, args.Nz)
    # w_coords = torch.linspace(-1.0 + 1.0 / args.Nx, 1.0 - 1.0 / args.Nx, args.Nx)
    # h_coords = torch.linspace(-1.0 + 1.0 / args.Ny, 1.0 - 1.0 / args.Ny, args.Ny)
    sx = args.sampleX
    sy = args.sampleY
    sz = args.sampleZ
    d_coords = torch.linspace(-1.0 + 1.0 / sz, 1.0 - 1.0 / sz, sz)
    w_coords = torch.linspace(-1.0 + 1.0 / sx, 1.0 - 1.0 / sx, sx)
    h_coords = torch.linspace(-1.0 + 1.0 / sy, 1.0 - 1.0 / sy, sy)
    
    # tot_points = frame_num * args.Nx * args.Ny * args.Nz
    tot_points = sx * sy * sz
    
    global_step = 1
    
    for iter in tqdm(range(1, args.num_epochs + 1)):
        shuffled_indices_np = np.random.permutation(tot_points)
        cnt_batch = 1
        for b_idx in tqdm(range(0, tot_points, batchsize)):
            batch_indices_np = shuffled_indices_np[b_idx: b_idx+batchsize]
            batch_indices = torch.tensor(batch_indices_np)
            w_batch_coords = batch_indices // (sy * sz)
            h_batch_coords = (batch_indices % (sy * sz)) // sz
            d_batch_coords = batch_indices % sz
            selected_coords = torch.stack((w_coords[w_batch_coords], h_coords[h_batch_coords], d_coords[d_batch_coords]), dim=-1)
            perturbations = torch.zeros_like(selected_coords)
            perturbations[:, 0] = torch.rand(selected_coords.shape[0]) * 2 / sx - 1 / sx  # [-1/Nx, 1/Nx]
            perturbations[:, 1] = torch.rand(selected_coords.shape[0]) * 2 / sy - 1 / sy  # [-1/Ny, 1/Ny]
            perturbations[:, 2] = torch.rand(selected_coords.shape[0]) * 2 / sz - 1 / sz  # [-1/Nz, 1/Nz]
            # perturbations[:, 3] = torch.rand(selected_coords.shape[0]) * 2 / frame_num - 1 / frame_num  # [-1/frame_num, 1/frame_num]
            selected_coords = selected_coords + perturbations
            
            # selected_coords = torch.cat((selected_coords, torch.full((selected_coords.shape[0], 1), torch.empty(1).uniform_(-1, 1).item())), dim=1)
            selected_coords = torch.cat((selected_coords, torch.rand(selected_coords.shape[0], 1) * 2 - 1), dim=1)
            b_coords = selected_coords.clone().to(device)
            with torch.no_grad():
                siren_vel = siren_model(b_coords)
            pts = my_voxel_tool.smoke2world(b_coords)
            pts.requires_grad_()
            
            ingp_vel = model(pts)
            ingp_vel = ingp_vel.reshape(-1, 3)
            
            jacobian = []

            for i in range(ingp_vel.shape[1]):
                grad_i = torch.autograd.grad(
                    outputs=ingp_vel[:, i],
                    inputs=pts,
                    grad_outputs=torch.ones_like(ingp_vel[:, i], device=ingp_vel.device),
                    retain_graph=True,
                    create_graph=True
                )[0]
                jacobian.append(grad_i.unsqueeze(1))

            ingp_vel_jac = torch.cat(jacobian, dim=1)
            
            D_ingp_vel_D_t_x = ingp_vel_jac[..., 0, 3] + siren_vel[..., 0] * ingp_vel_jac[..., 0, 0] + siren_vel[..., 1] * ingp_vel_jac[..., 0, 1] + siren_vel[..., 2] * ingp_vel_jac[..., 0, 2]
            D_ingp_vel_D_t_y = ingp_vel_jac[..., 1, 3] + siren_vel[..., 0] * ingp_vel_jac[..., 1, 0] + siren_vel[..., 1] * ingp_vel_jac[..., 1, 1] + siren_vel[..., 2] * ingp_vel_jac[..., 1, 2]
            D_ingp_vel_D_t_z = ingp_vel_jac[..., 2, 3] + siren_vel[..., 0] * ingp_vel_jac[..., 2, 0] + siren_vel[..., 1] * ingp_vel_jac[..., 2, 1] + siren_vel[..., 2] * ingp_vel_jac[..., 2, 2]
            D_ingp_vel_D_t = torch.stack((D_ingp_vel_D_t_x, D_ingp_vel_D_t_y, D_ingp_vel_D_t_z), dim=-1)
            loss_adv_ingp_vel = torch.mean(torch.square(D_ingp_vel_D_t))
            # loss_adv_ingp_vel = torch.tensor(0.0, device=device)
            
            cur_vel = ingp_vel + siren_vel.detach()

            b_den_raw = dens_model.dynamic_model_siren.density(pts)
            with torch.no_grad():
                b_grad_den = torch.autograd.grad(
                    b_den_raw,
                    pts,
                    torch.ones_like(b_den_raw, device=b_den_raw.get_device()),
                    retain_graph=True,
                    create_graph=True,
                )[0]
                b_grad_den = b_grad_den.detach()

            D_dens_D_t = (
                b_grad_den[..., 3]
                + cur_vel[..., 0] * b_grad_den[..., 0]
                + cur_vel[..., 1] * b_grad_den[..., 1]
                + cur_vel[..., 2] * b_grad_den[..., 2]
            )
            loss_adv = torch.mean(torch.square(D_dens_D_t))

            # loss_regular = torch.mean(torch.square(cur_vel))
            loss_regular = torch.tensor(0.0, device=device)
            
            loss_nse = torch.tensor(0.0, device=device)
            loss_div = torch.tensor(0.0, device=device)
            
            if cnt_batch % args.proj_interval == 0:
                indice_t = random.random() * 2.0 - 1.0
                sample_coords = torch.cat(
                    (
                        center_location,
                        torch.full(
                            (center_location.size(0), 1),
                            indice_t,
                            device=center_location.device,
                        ),
                    ),
                    dim=1,
                )
                sample_pts = my_voxel_tool.smoke2world(sample_coords)
                sample_vel = model(sample_pts)
                sample_vel = sample_vel.view(args.Nx, args.Ny, args.Nz, 3)
                max_x = args.Nx - args.proj_size
                max_y = args.Ny - args.proj_size
                max_z = args.Nz - args.proj_size

                start_x = torch.randint(0, max_x + 1, (1,)).item()
                start_y = torch.randint(0, max_y + 1, (1,)).item()
                start_z = torch.randint(0, max_z + 1, (1,)).item()
                slice_vel = sample_vel[start_x:start_x + args.proj_size, start_y:start_y + args.proj_size, start_z:start_z + args.proj_size, :]
                
                vel_world_supervised = slice_vel.detach().clone()
                vel_world_supervised[..., 2] *= -1
                vel_world_supervised = project_solver.Poisson(vel_world_supervised)
                vel_world_supervised[..., 2] *= -1

                loss_proj = torch.mean((vel_world_supervised - slice_vel) ** 2)
                proj_loss.update(loss_proj.item())
            else:
                loss_proj = torch.tensor(0.0, device=device)
            # loss_proj = torch.tensor(0.0, device=device)
            cnt_batch += 1

            loss = (
                loss_adv
                + args.lambda_ingp * loss_adv_ingp_vel
                + args.lambda_regular * loss_regular
                + args.lambda_div * loss_div
                + args.lambda_nse * loss_nse
                + args.lambda_proj * loss_proj
            )

            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            model.optimizer.step()

            train_loss.update(loss.item())
            adv_loss.update(loss_adv.item())
            ingp_loss.update(loss_adv_ingp_vel.item())
            regular_loss.update(loss_regular.item())
            div_loss.update(loss_div.item())
            nse_loss.update(loss_nse.item())
            
            if global_step % args.i_print == 0:
                print(
                    f"[{iter}/{args.num_epochs}=={global_step}]: traing loss = {train_loss.avg:08f}, adv loss = {adv_loss.avg:08f}, regular loss = {regular_loss.avg:08f}, div loss = {div_loss.avg:08f}, nse loss = {nse_loss.avg:08f}, ingp loss = {ingp_loss.avg:08f}, proj loss = {proj_loss.avg:08f}"
                )

                writer.add_scalar("training loss", train_loss.avg, global_step)
                writer.add_scalar("adv loss", adv_loss.avg, global_step)
                writer.add_scalar("ingp loss", ingp_loss.avg, global_step)
                writer.add_scalar("regular loss", regular_loss.avg, global_step)
                writer.add_scalar("div loss", div_loss.avg, global_step)
                writer.add_scalar("nse loss", nse_loss.avg, global_step)
                writer.add_scalar("proj loss", proj_loss.avg, global_step)

                train_loss.reset()
                adv_loss.reset()
                ingp_loss.reset()
                regular_loss.reset()
                div_loss.reset()
                nse_loss.reset()
                proj_loss.reset()
            global_step += 1

        scheduler.step()

        point_num = args.Nx * args.Ny * args.Nz
        draw_indices = torch.randperm(point_num, device=device)
        
        # ============= draw velocity ============= #
        print("start draw velocity")
        imgs = []
        im_estim = torch.zeros((point_num, 3), device=device)
        for frame in tqdm(range(frame_num)):
            indice_t = (frame / frame_num) * 2.0 - 1.0

            for b_idx in range(0, point_num, batchsize):
                b_indices = draw_indices[b_idx : min(point_num, b_idx + batchsize)]
                b_coords = center_location[b_indices, ...]
                # extend to Nx4
                b_coords = torch.cat(
                    (
                        b_coords,
                        torch.full(
                            (b_coords.size(0), 1),
                            indice_t,
                            device=b_coords.device,
                        ),
                    ),
                    dim=1,
                )
                b_pts = my_voxel_tool.smoke2world(b_coords)
                with torch.no_grad():
                    # pixelvalues = model(b_pts).squeeze() + siren_model(b_coords).squeeze()
                    pixelvalues = model(b_pts).squeeze()

                im_estim[b_indices, :] = pixelvalues

            st_scale = [resX/frame_num, resY/frame_num, resZ/frame_num]
            im_estim = my_voxel_tool.vel_world2smoke(im_estim, st_scale)
            im_estim_np = im_estim.cpu().detach().numpy()
            im_estim_np = np.reshape(im_estim_np, (args.Nx, args.Ny, args.Nz, 3))
            im_estim_np = np.swapaxes(im_estim_np, 0, 2)
            estim_image = vel_uv2hsv(im_estim_np, scale=1000, is3D=True, logv=False)
            
            if frame == frame_num // 2:
                imageio.imwrite(f"{targetdir}/image/vel_image_{iter:06d}.png", estim_image)
            imgs.append(estim_image)

        video = np.stack(imgs, axis=0)
        imageio.mimwrite(
            f"{targetdir}/video/vel_video_{iter:06d}.mp4",
            video,
            fps=30,
            quality=8,
        )

        model.save(f"{targetdir}/ckpt/ingp_ckpt_{iter:06d}.pth")
        
        sys.stdout.flush()

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')    
    
    parser = config_parser()
    args = parser.parse_args()
    set_rand_seed(args.fix_seed)

    bkg_flag = args.white_bkgd
    args.white_bkgd = np.ones([3], dtype=np.float32) if bkg_flag else None

    import pdb
    try:
        train(args) # call train in run_nerf
    except Exception as e:
        print(e)
        pdb.post_mortem()
    