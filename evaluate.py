import os, sys
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import trange
import taichi as ti
from taichi_encoders.mgpcg import MGPCG_3

from src.dataset.load_pinf import load_pinf_frame_data

from src.network.hybrid_model import create_model

from src.renderer.occupancy_grid import init_occ_grid, update_occ_grid, update_static_occ_grid
from src.renderer.render_ray import render_path, render_eval, render_2d_trajectory, render

from src.utils.args import config_parser
from src.utils.training_utils import set_rand_seed, save_log
from src.utils.coord_utils import BBox_Tool, Voxel_Tool, jacobian3D, get_voxel_pts
from src.renderer.occupancy_grid import OccupancyGrid, OccupancyGridDynamic
from src.utils.loss_utils import (
    get_rendering_loss,
    get_velocity_loss,
    fade_in_weight,
    to8b,
)
from src.utils.visualize_utils import (
    draw_mapping,
    draw_mapping_3d,
    draw_mapping_3d_animation,
    vel_uv2hsv,
    den_scalar2rgb,
)
from siren import SIREN, load_SIREN_model, save_SIREN_model
from ingp import INGP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import json

ti.init(arch=ti.cuda, device_memory_GB=12.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def init(args):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    savedir = os.path.join("log", f"{args.expname}_{current_time}")
    os.makedirs(savedir, exist_ok=True)
    os.mkdir(os.path.join(savedir, "images"))
    os.mkdir(os.path.join(savedir, "images/vel"))
    os.mkdir(os.path.join(savedir, "images/vor"))
    os.mkdir(os.path.join(savedir, "images/trans"))
    os.mkdir(os.path.join(savedir, "video"))
    os.mkdir(os.path.join(savedir, "video/vel"))
    os.mkdir(os.path.join(savedir, "video/vor"))
    os.mkdir(os.path.join(savedir, "video/trans"))
    os.mkdir(os.path.join(savedir, "ckpt"))
    shutil.copy(args.config, savedir)
    return savedir


class VoxelTrans:
    def __init__(self, voxel_tran, voxel_tran_inv, voxel_scale, device):
        self.s2w = voxel_tran
        self.w2s = voxel_tran_inv
        self.scale = voxel_scale
        self.p_w_p_s = torch.eye(4)
        self.p_w_p_s[:3, :3] = voxel_tran[:3, :3] * voxel_scale.view(1, 3)
        self.p_w_p_s *= 0.5
        self.p_s_p_w = torch.inverse(self.p_w_p_s)
        self.p_s_p_w = self.p_s_p_w.to(device)
        self.p_w_p_s = self.p_w_p_s.to(device)
        self.device = device

    def smoke2world(self, coords):
        pts = (coords.clone() + 1) / 2
        pts = pts.view(-1, 4)
        Psmoke = pts[:, :3]
        pos_scale = Psmoke * (self.scale)  # 2.simulation to 3.target
        pos_rot = torch.sum(
            pos_scale[..., None, :] * (self.s2w[:3, :3]), -1
        )  # 3.target to 4.world
        pos_off = (self.s2w[:3, -1]).expand(pos_rot.shape)  # 3.target to 4.world
        pts[:, :3] = pos_rot + pos_off
        return pts

    def world2smoke(self, pts):
        pts_clone = pts.clone()
        Pworld = pts_clone[..., :3]
        pos_rot = torch.sum(
            Pworld[..., None, :] * self.w2s[:3, :3], dim=-1
        )
        pos_off = self.w2s[:3, -1].expand_as(pos_rot)
        new_pose = pos_rot + pos_off
        pos_scale = new_pose / self.scale
        pts_clone[..., :3] = pos_scale
        coords = pts_clone * 2.0 - 1.0
        return coords
    
    def vel_world2smoke(self, Vworld, st_factor):
        Vworld_clone = Vworld.clone()
        _st_factor = torch.Tensor(st_factor).expand((3, )).to(device)
        vel_rot = Vworld_clone[..., None, :] * (self.w2s[:3,:3])
        vel_rot = torch.sum(vel_rot, -1) # 4.world to 3.target 
        vel_scale = vel_rot / (self.scale) * _st_factor # 3.target to 2.simulation
        return vel_scale
    
    def vel_smoke2world(self, Vsmoke, st_factor):
        Vsmoke_clone = Vsmoke.clone()
        _st_factor = torch.Tensor(st_factor).expand((3, ))
        vel_scale = Vsmoke_clone * (self.scale) / _st_factor # 2.simulation to 3.target
        vel_rot = torch.sum(vel_scale[..., None, :] * (self.s2w[:3,:3]), -1) # 3.target to 4.world
        return vel_rot


def div_np(x):
    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]
    
    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)
    
    div = dudx + dvdy + dwdz
    return div


def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = np.stack([u,v,w], axis=-1)
    
    return j, c


def vis_vel_dual(args):
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
        fps = train_video['frame_rate']
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
    
    resX = args.vol_output_W
    resY = int(args.vol_output_W * float(voxel_scale[1]) / voxel_scale[0] + 0.5)
    resZ = int(args.vol_output_W * float(voxel_scale[2]) / voxel_scale[0] + 0.5)
    
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
    
    if args.load_path == "None":
        model = None
    else:
        model = INGP.load(args.load_path, args, device)
        model.eval()
    
    if args.siren_model_path == "None":
        siren_model = None
    else:
        siren_model = load_SIREN_model(args.siren_model_path).to(device)
        siren_model.eval()
    
    if siren_model is None:
        print("[Error]: siren_model can't be None!")
        exit(1)
    
    point_num = args.Nx * args.Ny * args.Nz
    draw_indices = torch.randperm(point_num, device=device)
    
    t_list = np.arange(t_info[0], t_info[1], t_info[-1])
    frame_num = t_list.shape[0]

    batchsize = args.batchsize
    
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
    
    # ============= draw velocity ============= #
    imgs = []
    vor_imgs = []
    gt_imgs = []
    gt_vor_imgs = []
    im_estim = torch.zeros((point_num, 3), device=device)
    ingp_vel = torch.zeros((point_num, 3), device=device)
    siren_vel = torch.zeros((point_num, 3), device=device)
    dens_value = torch.zeros((point_num, 1), device=device)
    mses = []
    divs = []
    vor_mses = []
    have_gt = (args.gt_prefix is not None) and (args.gt_ext is not None)
    have_dens_gt = (args.gt_dens_prefix is not None) and (args.gt_dens_ext is not None)
    
    writer = SummaryWriter(log_dir=savedir)
    
    for frame in tqdm(range(frame_num)):
        # frame = frame_num // 2
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
                pixelvalues_siren = siren_model(b_coords).squeeze()
                if model is not None:
                    pixelvalues_ingp = model(b_pts).squeeze()
                pixel_dens = dens_model.dynamic_model_siren.density(b_pts)

            with torch.no_grad():
                siren_vel[b_indices, :] = pixelvalues_siren
                if model is not None:
                    ingp_vel[b_indices, :] = pixelvalues_ingp
                dens_value[b_indices, :] = pixel_dens
                # ingp_vel[b_indices, :] = pixelvalues_ingp

        if args.mask:
            if have_dens_gt:
                gt_dens_path = f"{args.gt_dens_prefix}{frame:04d}{args.gt_dens_ext}"
                dens_gt_np = np.load(gt_dens_path)
                dens_gt_np = dens_gt_np["arr_0"]
                dens_gt_np = dens_gt_np[:args.Nz, :args.Ny, :args.Nx]
                if dens_gt_np.shape[0] == 1:
                    dens_gt_np = dens_gt_np.squeeze(axis=0)
                dens_mask = dens_gt_np > 0
                dens_mask = dens_mask.ravel()
            else:
                dens_mask = dens_value.squeeze(-1) > args.dens_thresh
                siren_vel[~dens_mask] *= 0
                ingp_vel[~dens_mask] *= 0
        st_scale = [resX/frame_num, resY/frame_num, resZ/frame_num]
        if model is not None:
            im_ingp = my_voxel_tool.vel_world2smoke(ingp_vel, st_scale)
            im_ingp_np = im_ingp.cpu().detach().numpy()
        # im_ingp_np = np.reshape(im_ingp_np, (args.Nx, args.Ny, args.Nz, 3))
        # im_ingp_np = freq_decompose(im_ingp_np)
        
        im_siren = my_voxel_tool.vel_world2smoke(siren_vel, st_scale)
        im_siren_np = im_siren.cpu().detach().numpy()
        # im_siren_np = np.reshape(im_siren_np, (args.Nx, args.Ny, args.Nz, 3))
        
        im_estim_np = im_siren_np
        if model is not None:
            norms = np.linalg.norm(im_estim_np, axis=1)
            mean_norm = np.mean(norms)
            threshold = np.minimum((norms / (mean_norm * 2)) ** 5, 1.0)
            im_estim_np += im_ingp_np * threshold[:, np.newaxis]
        im_estim_np = np.reshape(im_estim_np, (args.Nx, args.Ny, args.Nz, 3))
        
        # im_estim_np = im_ingp_np + im_siren_np
        im_estim_np = np.swapaxes(im_estim_np, 0, 2)
        
        div = div_np(im_estim_np)
        cur_div = np.mean(np.square(div))
        if args.mask and have_dens_gt:
            cur_div = np.mean(np.square(div.ravel()[dens_mask]))
        divs.append(cur_div)
        print(f"[{frame}/{frame_num} div]: {cur_div}")
        
        estim_image = vel_uv2hsv(im_estim_np, scale=args.vel_color, is3D=True, logv=False)
        
        imageio.imwrite(f"{savedir}/images/vel/vel_image_{frame:06d}.png", estim_image)
        imgs.append(estim_image)
        
        _, NETw = jacobian3D_np(im_estim_np)
        estim_image = vel_uv2hsv(
            NETw[0], scale=args.vor_color, is3D=True, logv=False
        )
        imageio.imwrite(f"{savedir}/images/vor/vor_image_{frame:06d}.png", estim_image)
        vor_imgs.append(estim_image)
        
        if have_gt:
            gt_path = f"{args.gt_prefix}{frame:04d}{args.gt_ext}"
            im_gt_np = np.load(gt_path)
            im_gt_np = im_gt_np["arr_0"]
            if im_gt_np.shape[0] == 1:
                im_gt_np = im_gt_np.squeeze(axis=0)
            im_gt_np = im_gt_np[:args.Nz, :args.Ny, :args.Nx]
            gt_image = vel_uv2hsv(im_gt_np, scale=args.vel_color, is3D=True, logv=False)
            imageio.imwrite(f"{savedir}/images/vel/gt_image_{frame:06d}.png", gt_image)
            gt_imgs.append(gt_image)
            
            _, NETw_gt = jacobian3D_np(im_gt_np)
            gt_vor_image = vel_uv2hsv(
                NETw_gt[0], scale=args.vor_color, is3D=True, logv=False
            )
            imageio.imwrite(f"{savedir}/images/vor/gt_image_{frame:06d}.png", gt_vor_image)
            gt_vor_imgs.append(gt_vor_image)
            
            vel_error = im_estim_np / args.render_interval - im_gt_np
            cur_mse = np.mean(np.square(vel_error))
            if args.mask and have_dens_gt:
                cur_mse = np.mean(np.square(vel_error.reshape(-1, 3)[dens_mask]))
            mses.append(cur_mse)
            
            vor_error = NETw[0] / args.render_interval - NETw_gt[0]
            cur_vor_mse = np.mean(np.square(vor_error))
            if args.mask and have_dens_gt:
                cur_vor_mse = np.mean(np.square(vor_error.reshape(-1, 3)[dens_mask]))
            vor_mses.append(cur_vor_mse)
            
            print(f"[{frame}/{frame_num}]: velocity MSE: {cur_mse}, vorticity MSE: {cur_vor_mse}")
            writer.add_scalar("MSE", cur_mse, frame)
            writer.add_scalar("Vor_MSE", cur_vor_mse, frame)
        
    video = np.stack(imgs, axis=0)
    imageio.mimwrite(
        f"{savedir}/video/vel/vel_video.mp4",
        video,
        fps=fps,
        quality=8,
    )
    
    video = np.stack(vor_imgs, axis=0)
    imageio.mimwrite(
        f"{savedir}/video/vor/vor_video.mp4",
        video,
        fps=fps,
        quality=8,
    )
    
    if have_gt:
        mean_mse = sum(mses) / len(mses)
        print("mean mse: ", mean_mse)
        mean_vor_mse = sum(vor_mses) / len(vor_mses)
        print("mean vor mse: ", mean_vor_mse)
        with open(os.path.join(savedir, "mse.txt"), 'w') as fp:
            fp.write(str(mean_mse))
            fp.write(str(mean_vor_mse))
            fp.close()
        
        video = np.stack(gt_imgs, axis=0)
        imageio.mimwrite(
            f"{savedir}/video/vel/gt_vel_video.mp4",
            video,
            fps=fps,
            quality=8,
        )
        
        video = np.stack(gt_vor_imgs, axis=0)
        imageio.mimwrite(
            f"{savedir}/video/vor/gt_vor_video.mp4",
            video,
            fps=fps,
            quality=8,
        )
    
    mean_div = sum(divs) / len(divs)
    print("mean div: ", mean_div)
    with open(os.path.join(savedir, "div.txt"), 'w') as fp:
        fp.write(str(mean_div))
        fp.close()


def advect_SL(dens_grid, vel_world, cur_pts, my_voxel_tool, dt, args):
    # cur_pts = my_voxel_tool.smoke2world(cur_coord)
    vel_world = vel_world.view(-1, 3)
    backtrace_pts = cur_pts - torch.cat((vel_world, torch.full((vel_world.size(0), 1), 1, device=device)), dim=-1) * dt
    backtrace_coord = my_voxel_tool.world2smoke(backtrace_pts) # [..., 4]
    
    dens_field = dens_grid.view(args.Nx, args.Ny, args.Nz)
    dens_field = F.pad(dens_field, (1, 1, 1, 1, 1, 1))
    dens_field = dens_field.view(1, 1, args.Nx+2, args.Ny+2, args.Nz+2)
    grid = backtrace_coord[..., :3]
    grid[..., 0] *= args.Nx / (args.Nx+1)
    grid[..., 1] *= args.Ny / (args.Ny+1)
    grid[..., 2] *= args.Nz / (args.Nz+1)
    grid[:, [0, 2]] = grid[:, [2, 0]]
    N = grid.shape[0]
    grid = grid.view(1, N, 1, 1, 3)
    output = F.grid_sample(dens_field, grid, mode='bilinear', padding_mode='border', align_corners=True)
    cur_dens = output.view(args.Nx, args.Ny, args.Nz, 1)
    
    return cur_dens


def run_advect_den_dual(args):
    with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
        meta = json.load(fp)
        train_video = meta['train_videos'][0]
        fps = train_video['frame_rate']
        
    images, masks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, data_extras = load_pinf_frame_data(args, args.datadir, args.half_res, args.testskip, args.trainskip)
    Ks = [
        [
        [hwf[-1], 0, 0.5*hwf[1]],
        [0, hwf[-1], 0.5*hwf[0]],
        [0, 0, 1]
        ] for hwf in hwfs
    ]
    
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    print('Loaded pinf frame data', images.shape, render_poses.shape, hwfs[0], args.datadir)
    print('Loaded voxel matrix', voxel_tran, 'voxel scale',  voxel_scale)

    args.time_size = len(list(np.arange(t_info[0],t_info[1],t_info[-1])))

    voxel_tran_inv = torch.Tensor(voxel_tran_inv)
    voxel_tran = torch.Tensor(voxel_tran)
    voxel_scale = torch.Tensor(voxel_scale)

    i_train, i_val, i_test = i_split
    if bkg_color is not None:
        args.white_bkgd = torch.Tensor(bkg_color).to(device)
        print('Scene has background color', bkg_color, args.white_bkgd)
    test_bkg_color = bkg_color
    
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
    
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if (use_batching) or (N_rand is None):
        print('Not supported!')
        return

    # Prepare Loss Tools (VGG, Den2Vel)
    ###############################################
    # vggTool = VGGlossTool(device)

    # Move to GPU, except images
    poses = torch.Tensor(poses).to(device)
    time_steps = torch.Tensor(time_steps).to(device)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    init_occ_grid(args, dens_model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None)

    trainImg = False

    dens_model.train()

    training_stage = 0
    
    if (args.occ_dynamic_path is not None) and (args.occ_static_path is not None):
        occ_dynamic_path = args.occ_dynamic_path
        occ_static_path = args.occ_static_path
        first_update_occ_grid = False
        dens_model.occupancy_grid_dynamic = OccupancyGridDynamic.load(occ_dynamic_path, device=device)
        dens_model.occupancy_grid_static = OccupancyGrid.load(occ_static_path, device=device)
        print("load occ grid")
    else:
        first_update_occ_grid = True
    
    training_stage = 2
    trainImg = True
        
    dens_model.iter_step = global_step
    dens_model.update_model(training_stage, global_step) # progressive training for siren smoke
    
    if trainImg and global_step > args.uniform_sample_step and args.cuda_ray:
        if first_update_occ_grid:

            for i in range(16):
                update_occ_grid(args, dens_model, global_step, update_interval = 1, update_interval_static = 1, neus_early_terminated = False)
            if not dens_model.single_scene:
                update_static_occ_grid(args, dens_model, times=30)
            
            first_update_occ_grid = False
        else:
            update_occ_grid(args, dens_model, global_step, update_interval = 1000, neus_early_terminated = training_stage is not 1 and args.neus_early_terminated)
    
    dens_model.eval()
    
    if args.load_path == "None":
        model = None
    else:
        model = INGP.load(args.load_path, args, device)
        model.eval()
    
    if args.siren_model_path == "None":
        siren_model = None
    else:
        siren_model = load_SIREN_model(args.siren_model_path).to(device)
        siren_model.eval()
    
    if siren_model is None:
        print("[Error]: siren_model can't be None!")
        exit(1)
    
    t_list = np.arange(t_info[0], t_info[1], t_info[-1])
    frame_num = t_list.shape[0]
    
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
    
    init_time = torch.ones_like(center_location[..., :1]) * (-1)
    init_coord = torch.cat((center_location, init_time), dim=1)
    init_pts = my_voxel_tool.smoke2world(init_coord)
    
    chunk_size = 4096 * 4
    init_dens = torch.zeros((args.Nx * args.Ny * args.Nz, 1))
    with torch.no_grad():
        for i in range(0, init_pts.shape[0], chunk_size):
            i_right = min(i+chunk_size, init_pts.shape[0])
            init_dens[i:i_right] = dens_model.dynamic_model_siren.density(init_pts[i:i_right])
    cur_dens = init_dens.view(args.Nx, args.Ny, args.Nz, 1)
    
    imgs = []
    delta_t = 1.0 / frame_num
    
    source_height = -1 + 0.25 * 2.0
    mask_to_sim = center_location[..., 1] > source_height
        
    hwf = hwfs[i_test[0]]
    hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
    H, W, focal = hwf
    K = Ks[i_test[0]]
    render_poses = torch.Tensor(poses[i_test]).to(device)
    time_steps = torch.Tensor(time_steps).to(device)
    render_steps = time_steps[i_test]
    gt_imgs=images[i_test]
    
    mean_psnr = 0.0
    cnt = 0
    
    rgbs = []
    disps = []
    
    for i in tqdm(range(frame_num)):
        cur_time = torch.ones_like(center_location[..., :1]) * (2.0 * i / frame_num - 1.0)
        cur_coord = torch.cat((center_location, cur_time), dim=1)
        cur_pts = my_voxel_tool.smoke2world(cur_coord)
        
        ingp_vel = torch.zeros_like(center_location)
        cur_vel = torch.zeros_like(center_location)
        with torch.no_grad():
            for j in range(0, cur_pts.shape[0], chunk_size):
                j_right = min(j+chunk_size, cur_pts.shape[0])
                cur_vel[j:j_right] = siren_model(cur_coord[j:j_right])
                if model is not None:
                    ingp_vel[j:j_right] = model(cur_pts[j:j_right])
        
        # ingp_vel_np = ingp_vel.detach().cpu().numpy()
        # ingp_vel_np = ingp_vel_np.reshape(args.Nx, args.Ny, args.Nz, 3)
        # # ingp_vel_np = freq_decompose(ingp_vel_np)
        # ingp_vel_np = ingp_vel_np.reshape(-1, 3)
        # cur_vel += torch.from_numpy(ingp_vel_np).to(device)
        if model is not None:
            norms = torch.norm(cur_vel, dim=1)
            mean_norm = torch.mean(norms)
            threshold = torch.min((norms / (mean_norm * 2)) ** 5, torch.tensor(1.0))
            cur_vel += ingp_vel * threshold.unsqueeze(1)
            # cur_vel += ingp_vel
                
        nxt_dens = advect_SL(cur_dens, cur_vel, cur_pts, my_voxel_tool, delta_t, args)
        back_dens = advect_SL(nxt_dens, cur_vel, cur_pts, my_voxel_tool, -delta_t, args)
        advect_dens = nxt_dens + (cur_dens - back_dens) / 2
        cur_dens = advect_dens
        
        cur_time = torch.ones_like(center_location[..., :1]) * (-1 + 2.0 * i / frame_num)
        cur_coord = torch.cat((center_location, cur_time), dim=1)
        cur_pts = my_voxel_tool.smoke2world(cur_coord)
        cur_dens = cur_dens.view(-1, 1)
        with torch.no_grad():
            indices = torch.nonzero(~mask_to_sim, as_tuple=False).squeeze()
            for j in range(0, len(indices), chunk_size):
                j_right = min(j+chunk_size, len(indices))
                # cur_dens[~mask_to_sim][j:i_right] = dens_model.dynamic_model_siren.density(cur_pts[~mask_to_sim][j:i_right])
                batch_indices = indices[j:j_right]
                # import pdb; pdb.set_trace()
                cur_dens[batch_indices] = dens_model.dynamic_model_siren.density(cur_pts[batch_indices])
            # import pdb; pdb.set_trace()
            # cur_dens[~mask_to_sim] = dens_model.dynamic_model_siren.density(cur_pts[~mask_to_sim])
        cur_dens = cur_dens.view(args.Nx, args.Ny, args.Nz, 1)
        # with torch.no_grad():
        #     cur_dens[~mask_to_sim] = dens_model.dynamic_model_siren.density(cur_pts[~mask_to_sim])
        # cur_dens = cur_dens.view(args.Nx, args.Ny, args.Nz, 1)
        
        # drawing density
        dens_np = cur_dens.cpu().detach().numpy()
        dens_np = np.reshape(dens_np, (args.Nx, args.Ny, args.Nz, 1))
        dens_np = np.swapaxes(dens_np, 0, 2)
        dens_image = den_scalar2rgb(dens_np, scale=args.dens_color, is3D=True, logv=False, mix=True)
        imageio.imwrite(f"{savedir}/images/trans/dens_{i:04d}.png", dens_image)
        # return
        imgs.append(dens_image)
        
        c2w = render_poses[i]
        rgb, disp, acc, extras = render(H, W, K, dens_model, chunk=args.test_chunk, c2w=c2w[:3,:4], netchunk=args.netchunk, time_step=render_steps[i], bkgd_color=test_bkg_color, near = near, far = far, cuda_ray = global_step > args.uniform_sample_step and args.cuda_ray, perturb=0, my_voxel_tool = my_voxel_tool, use_grid=True, den_grid=cur_dens)
        rgbs.append(rgb.detach().cpu().numpy())
        disps.append(disp.detach().cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print("PSNR: ", p)
            mean_psnr += p
            cnt += 1

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, 'images', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            other_rgbs = []
            if gt_imgs is not None:
                other_rgbs.append(gt_imgs[i])
            for rgb_i in ['rgbh1','rgbh2','rgb0']: 
                if rgb_i in extras:
                    _data = extras[rgb_i].detach().cpu().numpy()
                    other_rgbs.append(_data)
            if len(other_rgbs) >= 1:
                other_rgb8 = np.concatenate(other_rgbs, axis=1)
                other_rgb8 = to8b(other_rgb8)
                filename = os.path.join(savedir, 'images', '_{:03d}.png'.format(i))
                imageio.imwrite(filename, other_rgb8)

            filename = os.path.join(savedir, 'images', 'disp_{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(disp.squeeze(-1).detach().cpu().numpy()))

            ## acc map
            filename = os.path.join(savedir, 'images', 'acc_{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(acc.squeeze(-1).detach().cpu().numpy()))
        
        del rgb, disp, acc, extras        
        del cur_vel, ingp_vel
        torch.cuda.empty_cache()
    
    video = np.stack(imgs, axis=0)
    imageio.mimwrite(f"{savedir}/run_advect_dens.mp4", video, fps=fps, quality=8)
    
    if cnt != 0:
        mean_psnr = mean_psnr / cnt
        print("mean psnr: ", mean_psnr)
        psnr_file_path = os.path.join(savedir, "psnr.txt")
        with open(psnr_file_path, 'w') as fp:
            fp.write(str(mean_psnr))
            fp.close()
    
    imageio.mimwrite(f"{savedir}/resim_rgb.mp4", to8b(rgbs), fps=fps, quality=8)


def run_future_pred_dual(args):
    boundary_types = ti.Matrix([[1, 1], [2, 1], [1, 1]], ti.i32)
    project_solver = MGPCG_3(boundary_types=boundary_types, N=[args.Nx, args.Nx, args.Nx], base_level=3)
    
    start_frame = args.frame_start
    predict_duration = args.frame_duration
    
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
        fps = train_video['frame_rate']
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
    
    if args.load_path == "None":
        model = None
    else:
        model = INGP.load(args.load_path, args, device)
        model.eval()
    
    if args.siren_model_path == "None":
        siren_model = None
    else:
        siren_model = load_SIREN_model(args.siren_model_path).to(device)
        siren_model.eval()
    
    if siren_model is None:
        print("[Error]: siren_model can't be None!")
        exit(1)
    
    t_list = np.arange(t_info[0], t_info[1], t_info[-1])
    frame_num = t_list.shape[0]
    
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
    
    init_time = torch.ones_like(center_location[..., :1]) * (2.0 * start_frame / frame_num - 1.0)
    init_coord = torch.cat((center_location, init_time), dim=1)
    init_pts = my_voxel_tool.smoke2world(init_coord)
    # with torch.no_grad():
    #     static_sdf = dens_model.static_model.sdf(init_pts[..., :3])
    #     mask = static_sdf < 0
        # import pdb; pdb.set_trace()
    
    chunk_size = 4096 * 4
    init_dens = torch.zeros((args.Nx * args.Ny * args.Nz, 1))
    init_vel = torch.zeros((args.Nx * args.Ny * args.Nz, 3))
    ingp_vel = torch.zeros((args.Nx * args.Ny * args.Nz, 3))
    with torch.no_grad():
        for i in range(0, init_pts.shape[0], chunk_size):
            i_right = min(i+chunk_size, init_pts.shape[0])
            init_dens[i:i_right] = dens_model.dynamic_model_siren.density(init_pts[i:i_right])
            init_vel[i: i_right] = siren_model(init_coord[i: i_right])
            if model is not None:
                ingp_vel[i: i_right] = model(init_pts[i:i_right])
                
    cur_dens = init_dens.view(args.Nx, args.Ny, args.Nz, 1)
    cur_vel = init_vel
    if model is not None:
        norms = torch.norm(cur_vel, dim=1)
        mean_norm = torch.mean(norms)
        threshold = torch.min((norms / (mean_norm * 2)) ** 5, torch.tensor(1.0))
        cur_vel += ingp_vel * threshold.unsqueeze(1)
    
    imgs = []
    vel_imgs = []
    delta_t = 1.0 / frame_num
    
    y_start = args.y_start
    y_proj = args.y_proj
    source_height = -1 + y_start / args.Ny * 2.0
    mask_to_sim = center_location[..., 1] > source_height
    
    for i in tqdm(range(start_frame, start_frame+predict_duration)):
        cur_time = torch.ones_like(center_location[..., :1]) * (2.0 * i / frame_num - 1.0)
        cur_coord = torch.cat((center_location, cur_time), dim=1)
        
        cur_dens = advect_SL(cur_dens, cur_vel, cur_coord, my_voxel_tool, delta_t, args)
        origin_vel_x = cur_vel[..., 0].unsqueeze(-1)
        origin_vel_y = cur_vel[..., 1].unsqueeze(-1)
        origin_vel_z = cur_vel[..., 2].unsqueeze(-1)
        cur_vel_x = advect_SL(origin_vel_x, cur_vel, cur_coord, my_voxel_tool, delta_t, args)
        cur_vel_y = advect_SL(origin_vel_y, cur_vel, cur_coord, my_voxel_tool, delta_t, args)
        cur_vel_z = advect_SL(origin_vel_z, cur_vel, cur_coord, my_voxel_tool, delta_t, args)
        cur_vel = torch.cat([cur_vel_x, cur_vel_y, cur_vel_z], dim=-1)
        
        cur_vel = cur_vel.view(args.Nx, args.Ny, args.Nz, 3)
        cur_vel[..., 2] *= -1
        # cur_vel = project_solver.Poisson(cur_vel)
        cur_vel[:, y_start:y_start + y_proj, ...] = project_solver.Poisson(cur_vel[:, y_start:y_start + y_proj, ...])
        cur_vel[..., 2] *= -1
        cur_vel = cur_vel.view(-1, 3)
        
        cur_time = torch.ones_like(center_location[..., :1]) * (-1 + 2.0 * i / frame_num)
        cur_coord = torch.cat((center_location, cur_time), dim=1)
        cur_pts = my_voxel_tool.smoke2world(cur_coord)
        cur_dens = cur_dens.view(-1, 1)
        
        with torch.no_grad():
            indices = torch.nonzero(~mask_to_sim, as_tuple=False).squeeze()
            for j in range(0, len(indices), chunk_size):
                j_right = min(j+chunk_size, len(indices))
                batch_indices = indices[j:j_right]
                cur_dens[batch_indices] = dens_model.dynamic_model_siren.density(cur_pts[batch_indices])
                cur_vel[batch_indices] = siren_model(cur_coord[batch_indices])
        
        # with torch.no_grad():
        #     cur_dens[~mask_to_sim] = dens_model.dynamic_model_siren.density(cur_pts[~mask_to_sim])
        #     cur_vel[~mask_to_sim] = model(cur_coord[~mask_to_sim])
        cur_dens = cur_dens.view(args.Nx, args.Ny, args.Nz, 1)

        # drawing density
        dens_np = cur_dens.cpu().detach().numpy()
        dens_np = np.reshape(dens_np, (args.Nx, args.Ny, args.Nz, 1))
        dens_np = np.swapaxes(dens_np, 0, 2)
        dens_image = den_scalar2rgb(dens_np, scale=args.dens_color, is3D=True, logv=False, mix=True)
        imgs.append(dens_image)
        
        # drawing velocity
        st_scale = [resX/frame_num, resY/frame_num, resZ/frame_num]
        cur_vel_smoke = my_voxel_tool.vel_world2smoke(cur_vel, st_scale)
        vel_np = cur_vel_smoke.cpu().detach().numpy()
        vel_np = np.reshape(vel_np, (args.Nx, args.Ny, args.Nz, 3))
        vel_np = np.swapaxes(vel_np, 0, 2)
        vel_image = vel_uv2hsv(vel_np, scale=args.vel_color, is3D=True, logv=False)
        vel_imgs.append(vel_image)
        
    video = np.stack(imgs, axis=0)
    imageio.mimwrite(f"{savedir}/run_future_pred.mp4", video, fps=fps, quality=8)
    video = np.stack(vel_imgs, axis=0)
    imageio.mimwrite(f"{savedir}/run_future_pred_velocity.mp4", video, fps=fps, quality=8)


def run_future_pred_manta(args):
    start_frame = args.frame_start
    predict_duration = args.frame_duration
    
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
        fps = train_video['frame_rate']
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
    
    if args.load_path == "None":
        model = None
    else:
        model = INGP.load(args.load_path, args, device)
        model.eval()
    
    if args.siren_model_path == "None":
        siren_model = None
    else:
        siren_model = load_SIREN_model(args.siren_model_path).to(device)
        siren_model.eval()
    
    if siren_model is None:
        print("[Error]: siren_model can't be None!")
        exit(1)
    
    t_list = np.arange(t_info[0], t_info[1], t_info[-1])
    frame_num = t_list.shape[0]
    
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
    
    dens_value = torch.zeros((args.Nx * args.Ny * args.Nz, 1))
    siren_vel = torch.zeros((args.Nx * args.Ny * args.Nz, 3))
    ingp_vel = torch.zeros((args.Nx * args.Ny * args.Nz, 3))
    
    dens_imgs = []
    vel_imgs = []
    for i in tqdm(range(start_frame, start_frame+predict_duration)):
        cur_time = torch.ones_like(center_location[..., :1]) * (2.0 * i / frame_num - 1.0)
        cur_coord = torch.cat((center_location, cur_time), dim=1)
        cur_pts = my_voxel_tool.smoke2world(cur_coord)
        
        chunk_size = 4096 * 4
        with torch.no_grad():
            for j in range(0, cur_pts.shape[0], chunk_size):
                j_right = min(j+chunk_size, cur_pts.shape[0])
                dens_value[j:j_right] = dens_model.dynamic_model_siren.density(cur_pts[j:j_right])
                siren_vel[j: j_right] = siren_model(cur_coord[j: j_right])
                if model is not None:
                    ingp_vel[j: j_right] = model(cur_pts[j:j_right])
                    
        if model is not None:
            norms = torch.norm(siren_vel, dim=1)
            mean_norm = torch.mean(norms)
            threshold = torch.min((norms / (mean_norm * 2)) ** 5, torch.tensor(1.0))
            siren_vel += ingp_vel * threshold.unsqueeze(1)
        
        dens_np = dens_value.cpu().detach().numpy()
        dens_np = dens_np.reshape(args.Nx, args.Ny, args.Nz, 1)
        st_scale = [resX/frame_num, resY/frame_num, resZ/frame_num]
        cur_vel = my_voxel_tool.vel_world2smoke(siren_vel, st_scale)
        vel_np = cur_vel.cpu().detach().numpy()
        vel_np = vel_np.reshape(args.Nx, args.Ny, args.Nz, 3)
        
        np.savez_compressed(f"{savedir}/ckpt/vel_{i:04d}.npz", vel_np)
        np.savez_compressed(f"{savedir}/ckpt/dens_{i:04d}.npz", dens_np)
        
        dens_np = np.swapaxes(dens_np, 0, 2)
        dens_image = den_scalar2rgb(dens_np, scale=args.dens_color, is3D=True, logv=False, mix=True)
        imageio.imwrite(f"{savedir}/images/trans/dens_{i:04d}.png", dens_image)
        dens_imgs.append(dens_image)
        vel_np = np.swapaxes(vel_np, 0, 2)
        vel_image = vel_uv2hsv(vel_np, scale=args.vel_color, is3D=True, logv=False)
        imageio.imwrite(f"{savedir}/images/vel/vel_{i:04d}.png", vel_image)
        vel_imgs.append(vel_image)
    
    video = np.stack(dens_imgs, axis=0)
    imageio.mimwrite(f"{savedir}/vis_density.mp4", video, fps=fps, quality=8)
    video = np.stack(vel_imgs, axis=0)
    imageio.mimwrite(f"{savedir}/vis_velocity.mp4", video, fps=fps, quality=8)


def render_test(args):
    images, masks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, data_extras = load_pinf_frame_data(args, args.datadir, args.half_res, args.testskip, args.trainskip)
    Ks = [
        [
        [hwf[-1], 0, 0.5*hwf[1]],
        [0, hwf[-1], 0.5*hwf[0]],
        [0, 0, 1]
        ] for hwf in hwfs
    ]
    
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    print('Loaded pinf frame data', images.shape, render_poses.shape, hwfs[0], args.datadir)
    print('Loaded voxel matrix', voxel_tran, 'voxel scale',  voxel_scale)

    args.time_size = len(list(np.arange(t_info[0],t_info[1],t_info[-1])))

    voxel_tran_inv = torch.Tensor(voxel_tran_inv)
    voxel_tran = torch.Tensor(voxel_tran)
    voxel_scale = torch.Tensor(voxel_scale)

    i_train, i_val, i_test = i_split
    if bkg_color is not None:
        args.white_bkgd = torch.Tensor(bkg_color).to(device)
        print('Scene has background color', bkg_color, args.white_bkgd)


    # Create Bbox model from smoke perspective
    bbox_model = None

    # this bbox in in the smoke simulation coordinate
    in_min = [float(_) for _ in args.bbox_min.split(",")]
    in_max = [float(_) for _ in args.bbox_max.split(",")]
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale, in_min, in_max)


    # Create model
    model, optimizer, start = create_model(args = args, bbox_model = bbox_model, device=device)

    global_step = start

    
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_timesteps = torch.Tensor(render_timesteps).to(device)

    test_bkg_color = bkg_color
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if (use_batching) or (N_rand is None):
        print('Not supported!')
        return

    # Prepare Loss Tools (VGG, Den2Vel)
    ###############################################
    # vggTool = VGGlossTool(device)

    # Move to GPU, except images
    poses = torch.Tensor(poses).to(device)
    time_steps = torch.Tensor(time_steps).to(device)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    init_occ_grid(args, model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None)

    trainImg = False

    model.train()

    training_stage = 0
    
    if (args.occ_dynamic_path is not None) and (args.occ_static_path is not None):
        occ_dynamic_path = args.occ_dynamic_path
        occ_static_path = args.occ_static_path
        first_update_occ_grid = False
        model.occupancy_grid_dynamic = OccupancyGridDynamic.load(occ_dynamic_path, device=device)
        model.occupancy_grid_static = OccupancyGrid.load(occ_static_path, device=device)
        print("load occ grid")
    else:
        first_update_occ_grid = True
    
    training_stage = 1 
    trainImg = True
        
    model.iter_step = global_step
    model.update_model(training_stage, global_step) # progressive training for siren smoke

    if trainImg and global_step > args.uniform_sample_step and args.cuda_ray:
        if first_update_occ_grid:

            for i in range(16):
                update_occ_grid(args, model, global_step, update_interval = 1, update_interval_static = 1, neus_early_terminated = False)
            if not model.single_scene:
                update_static_occ_grid(args, model, times=30)
            
            first_update_occ_grid = False
        else:
            update_occ_grid(args, model, global_step, update_interval = 1000, neus_early_terminated = training_stage is not 1 and args.neus_early_terminated)
    
    # for visualize rendering effect on testset
    model.eval()
    testsavedir = os.path.join(savedir, 'testset_{:06d}'.format(global_step))
    os.makedirs(testsavedir, exist_ok=True)
    print('test poses shape', poses[i_test].shape)
    hwf = hwfs[i_test[0]]
    hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
    # render_path(model, torch.Tensor(poses[i_test]).to(device), hwf, Ks[i_test[0]], args.test_chunk, near, far, netchunk = args.netchunk, cuda_ray = trainImg and global_step >= args.uniform_sample_step, gt_imgs=images[i_test], savedir=testsavedir, render_steps=time_steps[i_test], bkgd_color=test_bkg_color)
    render_path(model, torch.Tensor(poses[i_test]).to(device), hwf, Ks[i_test[0]], args.test_chunk, near, far, netchunk = args.netchunk, cuda_ray = trainImg and global_step > args.uniform_sample_step and args.cuda_ray, gt_imgs=images[i_test], savedir=testsavedir, render_steps=time_steps[i_test], bkgd_color=test_bkg_color)
    print('Saved test set')


def render_pred(args):
    with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
        meta = json.load(fp)
        train_video = meta['train_videos'][0]
        fps = train_video['frame_rate']
        
    start_frame = args.frame_start
    predict_duration = args.frame_duration
    
    images, masks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, data_extras = load_pinf_frame_data(args, args.datadir, args.half_res, args.testskip, args.trainskip)
    Ks = [
        [
        [hwf[-1], 0, 0.5*hwf[1]],
        [0, hwf[-1], 0.5*hwf[0]],
        [0, 0, 1]
        ] for hwf in hwfs
    ]
    
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    print('Loaded pinf frame data', images.shape, render_poses.shape, hwfs[0], args.datadir)
    print('Loaded voxel matrix', voxel_tran, 'voxel scale',  voxel_scale)

    args.time_size = len(list(np.arange(t_info[0],t_info[1],t_info[-1])))

    voxel_tran_inv = torch.Tensor(voxel_tran_inv)
    voxel_tran = torch.Tensor(voxel_tran)
    voxel_scale = torch.Tensor(voxel_scale)

    i_train, i_val, i_test = i_split
    if bkg_color is not None:
        args.white_bkgd = torch.Tensor(bkg_color).to(device)
        print('Scene has background color', bkg_color, args.white_bkgd)
    
    my_voxel_tool = VoxelTrans(voxel_tran, voxel_tran_inv, voxel_scale, device)


    # Create Bbox model from smoke perspective
    bbox_model = None

    # this bbox in in the smoke simulation coordinate
    in_min = [float(_) for _ in args.bbox_min.split(",")]
    in_max = [float(_) for _ in args.bbox_max.split(",")]
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale, in_min, in_max)


    # Create model
    model, optimizer, start = create_model(args = args, bbox_model = bbox_model, device=device)

    global_step = start

    
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_timesteps = torch.Tensor(render_timesteps).to(device)

    test_bkg_color = bkg_color
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if (use_batching) or (N_rand is None):
        print('Not supported!')
        return

    # Prepare Loss Tools (VGG, Den2Vel)
    ###############################################
    # vggTool = VGGlossTool(device)

    # Move to GPU, except images
    poses = torch.Tensor(poses).to(device)
    time_steps = torch.Tensor(time_steps).to(device)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    init_occ_grid(args, model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None)

    trainImg = False

    model.train()

    training_stage = 0
    
    if (args.occ_dynamic_path is not None) and (args.occ_static_path is not None):
        occ_dynamic_path = args.occ_dynamic_path
        occ_static_path = args.occ_static_path
        first_update_occ_grid = False
        model.occupancy_grid_dynamic = OccupancyGridDynamic.load(occ_dynamic_path, device=device)
        model.occupancy_grid_static = OccupancyGrid.load(occ_static_path, device=device)
        print("load occ grid")
    else:
        first_update_occ_grid = True
    
    training_stage = 1 
    trainImg = True
        
    model.iter_step = global_step
    model.update_model(training_stage, global_step) # progressive training for siren smoke

    if trainImg and global_step > args.uniform_sample_step and args.cuda_ray:
        if first_update_occ_grid:

            for i in range(16):
                update_occ_grid(args, model, global_step, update_interval = 1, update_interval_static = 1, neus_early_terminated = False)
            if not model.single_scene:
                update_static_occ_grid(args, model, times=30)
            
            first_update_occ_grid = False
        else:
            update_occ_grid(args, model, global_step, update_interval = 1000, neus_early_terminated = training_stage is not 1 and args.neus_early_terminated)
    
    # for visualize rendering effect on testset
    model.eval()
    
    hwf = hwfs[i_test[0]]
    hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
    H, W, focal = hwf
    K = Ks[i_test[0]]
    render_poses = torch.Tensor(poses[i_test]).to(device)
    time_steps = torch.Tensor(time_steps).to(device)
    render_steps = time_steps[i_test]
    gt_imgs=images[i_test]
    
    mean_psnr = 0.0
    cnt = 0
    
    rgbs = []
    disps = []
    
    pred_path = args.pred_path
    
    for i in tqdm(range(start_frame, start_frame + predict_duration)):
        dens_path = f"{pred_path}/dens_{i:04d}.npz"
        dens_np = np.load(dens_path)["arr_0"]
        cur_dens = torch.from_numpy(dens_np).to(device)
        
        c2w = render_poses[i]
        rgb, disp, acc, extras = render(H, W, K, model, chunk=args.test_chunk, c2w=c2w[:3,:4], netchunk=args.netchunk, time_step=render_steps[i], bkgd_color=test_bkg_color, near = near, far = far, cuda_ray = global_step > args.uniform_sample_step and args.cuda_ray, perturb=0, my_voxel_tool = my_voxel_tool, use_grid=True, den_grid=cur_dens)
        rgbs.append(rgb.detach().cpu().numpy())
        disps.append(disp.detach().cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print("PSNR: ", p)
            mean_psnr += p
            cnt += 1

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, 'images', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            other_rgbs = []
            if gt_imgs is not None:
                other_rgbs.append(gt_imgs[i])
            for rgb_i in ['rgbh1','rgbh2','rgb0']: 
                if rgb_i in extras:
                    _data = extras[rgb_i].detach().cpu().numpy()
                    other_rgbs.append(_data)
            if len(other_rgbs) >= 1:
                other_rgb8 = np.concatenate(other_rgbs, axis=1)
                other_rgb8 = to8b(other_rgb8)
                filename = os.path.join(savedir, 'images', '_{:03d}.png'.format(i))
                imageio.imwrite(filename, other_rgb8)

            filename = os.path.join(savedir, 'images', 'disp_{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(disp.squeeze(-1).detach().cpu().numpy()))

            ## acc map
            filename = os.path.join(savedir, 'images', 'acc_{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(acc.squeeze(-1).detach().cpu().numpy()))
        
        del rgb, disp, acc, extras
        torch.cuda.empty_cache()

    if cnt != 0:
        mean_psnr = mean_psnr / cnt
        print("mean psnr: ", mean_psnr)
        psnr_file_path = os.path.join(savedir, "psnr.txt")
        with open(psnr_file_path, 'w') as fp:
            fp.write(str(mean_psnr))
            fp.close()
    
    imageio.mimwrite(f"{savedir}/pred_rgb.mp4", to8b(rgbs), fps=fps, quality=8)


def add_parser(parser):
    parser.add_argument("--y_start", type=int, default=48)
    parser.add_argument("--y_proj", type=int, default=128)
    
    parser.add_argument("--frame_start", type=int, default=89)
    parser.add_argument("--frame_duration", type=int, default=30)
    
    parser.add_argument('--mask', action = 'store_true')
    parser.add_argument('--dens_thresh', type=float, default=0)
    
    parser.add_argument('--inflow_dir', type=str, default=None)
    
    parser.add_argument('--render_pred', action = 'store_true')
    parser.add_argument('--pred_path', type=str, default=None)
    
    parser.add_argument('--render_interval', type=float, default=1.0)
    
    parser.add_argument('--sim_steps', type=int, default=1)
    
    parser.add_argument('--gt_dens_prefix', type=str, default=None)
    parser.add_argument('--gt_dens_ext', type=str, default=None)
    return parser


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    parser = config_parser()
    parser = add_parser(parser)
    args = parser.parse_args()
    set_rand_seed(args.fix_seed)
    savedir = init(args)

    bkg_flag = args.white_bkgd
    args.white_bkgd = np.ones([3], dtype=np.float32) if bkg_flag else None
    args.test_mode = True

    import pdb
    try:
        if args.render_test:
            render_test(args)
        if args.vis_vel:
            vis_vel_dual(args)
        if args.resim:
            run_advect_den_dual(args)
        if args.pred:
            # run_future_pred_dual(args)
            '''
            To utilize mantaflow(http://mantaflow.com/index.html) for prediction, 
            run run_future_pred_manta first, 
            then running 'manta prediction.py --config configs/recons-scalar/eval.txt',
            remember to set inflow_dir
            '''
            run_future_pred_manta(args)
        if args.render_pred:
            render_pred(args)
    except Exception as e:
        print(e)
        pdb.post_mortem()