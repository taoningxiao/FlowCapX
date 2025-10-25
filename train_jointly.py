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

from test import visualize_all

from tqdm import tqdm

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
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    logdir, writer, targetdir = save_log(args)

    # Load data
    cam_info_others = None
 
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
    
    p_w_p_s = torch.eye(4)
    p_w_p_s[:3, :3] = voxel_tran[:3, :3] * voxel_scale.view(1, 3)
    p_w_p_s *= 0.5
    p_s_p_w = torch.inverse(p_w_p_s)
    p_w_p_s = p_w_p_s.to(device)
    p_s_p_w = p_s_p_w.to(device)
    my_voxel_tool = VoxelTrans(voxel_tran, voxel_tran_inv, voxel_scale, device)

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

    N_iters = args.N_iter

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Prepare Voxel Sampling Tools for Image Summary (voxel_writer), Physical Priors (training_voxel), Data Priors Represented by D2V (den_p_all)
    # voxel_writer: to sample low resolution data for for image summary 
    resX = args.vol_output_W # complexity O(N^3)
    resY = int(resX*float(voxel_scale[1])/voxel_scale[0]+0.5)
    resZ = int(resX*float(voxel_scale[2])/voxel_scale[0]+0.5)
    voxel_writer = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,resZ,resY,resX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)

  
    # start = start + 1

    testimgdir = os.path.join(targetdir, "imgs_"+logdir)
    os.makedirs(testimgdir, exist_ok=True)
    # some loss terms 
    

    init_occ_grid(args, model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None)

  

    trainVGG = False
    trainVel = False
    trainVel_using_rendering_samples = False
    trainImg = False

    local_step = 0
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

    t_list = np.arange(t_info[0], t_info[1], t_info[-1])
    frame_num = t_list.shape[0]
    
    if args.load_path == "None":
        vel_model = SIREN(
            args.in_dim, args.out_dim, args.hidden_layers, args.hidden_dim
        ).to(device)
        vel_model.init_weight()
    else:
        vel_model = load_SIREN_model(args.load_path).to(device)
    vel_optim = torch.optim.Adam(lr=args.lrate_vel, params=vel_model.parameters())
    gamma = 1e-3 ** (1.0 / args.num_epochs)
    vel_scheduler = torch.optim.lr_scheduler.ExponentialLR(vel_optim, gamma, verbose=True)
    
    adv_loss, regular_loss, div_loss, nse_loss, boundary_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    my_vel_loss = AverageMeter()
    
    batchsize = args.batchsize
    sampleX = args.sampleX
    sampleY = args.sampleY
    sampleZ = args.sampleZ
    d_coords = torch.linspace(-1.0 + 1.0 / sampleZ, 1.0 - 1.0 / sampleZ, sampleZ)
    w_coords = torch.linspace(-1.0 + 1.0 / sampleX, 1.0 - 1.0 / sampleX, sampleX)
    h_coords = torch.linspace(-1.0 + 1.0 / sampleY, 1.0 - 1.0 / sampleY, sampleY)
    tot_points = sampleX * sampleY * sampleZ
    # shuffled_indices = torch.randperm(tot_points)
    shuffled_indices_np = np.random.permutation(tot_points)
    b_idx = 0
    cur_epoch = 0
    
    long_term_len = args.long_term_len
    beta = args.long_term_beta
    criterion = torch.nn.MSELoss()
    
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
    
    min_ratio = float(64+4*2)/min(voxel_scale[0],voxel_scale[1],voxel_scale[2])
    minX = int(min_ratio*voxel_scale[0]+0.5)
    trainX = max(args.vol_output_W,minX) # a minimal resolution of 64^3
    trainY = int(trainX*float(voxel_scale[1])/voxel_scale[0]+0.5)
    trainZ = int(trainX*float(voxel_scale[2])/voxel_scale[0]+0.5)
    training_voxel = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,trainZ,trainY,trainX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)
    training_pts = torch.reshape(training_voxel.pts, (-1,3)) 
    
    decay_rate = 0.1
    decay_steps = args.lrate_decay * 1000
    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    # new_lrate_vel = args.lrate_vel * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    
    for local_step in trange(1, N_iters + 1):
        global_step = local_step + start
        training_stage = 2
        trainImg = True
        trainVel = True
        Vel2Dens = (global_step % args.nse_loss_interval == 0)
            
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
              
 
        loss = 0
        rendering_loss_dict = None
        vel_loss_dict = None
        

        # Random from one frame
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        
  
        target_mask = None

        pose = poses[img_i, :3,:4]
        time_locate = time_steps[img_i].to(device) 
        
        if trainImg:


            # time1 = time.time()
            # Cast intrinsics to right types
            H, W, focal = hwfs[img_i]
            H, W = int(H), int(W)
            focal = float(focal)
            hwf = [H, W, focal]

            _cam_info = None

    
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

            # batch_rays: training rays
            # target_s: target image
            # dw: get a cropped img (dw,dw) to train vgg

            batch_rays, target_s, dw, target_mask, bg_color, select_coords = prepare_rays(args, H, W, K, pose, target, trainVGG, global_step, start, N_rand, target_mask, _cam_info)

            bg_color = bg_color + args.white_bkgd

            rgb, disp, acc, extras = render(H, W, K, model, N_samples = args.N_samples, chunk=args.training_ray_chunk, rays=batch_rays, netchunk=args.netchunk,
                                        time_step=time_locate,
                                        near = near,
                                        far = far,
                                        bkgd_color=bg_color,
                                        # cuda_ray = trainImg and global_step >= args.uniform_sample_step,
                                        cuda_ray = global_step > args.uniform_sample_step and args.cuda_ray,
                                        perturb = args.perturb
                                        )
            


            if "num_points" in extras and extras["num_points"] == 0:
                print(f"no points in the ray, skip iteration {global_step}")
                torch.cuda.empty_cache()
                # local_step += 1
                continue
                

            rendering_loss, rendering_loss_dict = get_rendering_loss(args, model, rgb, acc, target_s, bg_color, extras, time_locate, global_step, target_mask)
            loss += rendering_loss
            
            smoke_samples_xyz = extras['samples_xyz_dynamic'] # has bugs for uniform sample

            if Vel2Dens:
                train_random = np.random.choice(trainZ*trainY*trainX, args.train_vel_grid_size**3)
                training_samples = training_pts[train_random]

                training_samples = training_samples.view(-1,3)
                training_t = torch.ones([training_samples.shape[0], 1])*time_locate
                training_samples = torch.cat([training_samples,training_t], dim=-1)
                
                b_coords = my_voxel_tool.world2smoke(training_samples)
                delta_t = 1.0 / frame_num
                pts = my_voxel_tool.smoke2world(b_coords)
                ori_dens = model.dynamic_model_siren.density(pts)

                nxt_pts = pts
                nxt_b_coords = b_coords
                
                min_step = 3
                max_step = 7
                steps = torch.randint(min_step, max_step, (training_samples.shape[0],), device=device)
                
                for _ in range(max_step):
                    active_mask = steps > 0
                    if not active_mask.any():
                        break
                    
                    adv_vel = vel_model(nxt_b_coords[active_mask]).detach()
                    mid_pts = (nxt_pts[active_mask] + 0.5 * torch.cat((adv_vel,torch.full((adv_vel.size(0), 1), 1, device=device)), dim=-1)* delta_t)
                    mid_b_coords = my_voxel_tool.world2smoke(mid_pts)
                    mask_mid = (mid_b_coords >= -1) & (mid_b_coords <= 1)
                    mask_mid = mask_mid.all(dim=1)
                    mid_b_coords = mid_b_coords[mask_mid]
                    mask1 = active_mask.clone()
                    active_mask[mask1] = mask_mid
                                        
                    mid_adv_vel = vel_model(mid_b_coords).detach()
                    final_pts = (nxt_pts[active_mask] + torch.cat((mid_adv_vel, torch.full((mid_adv_vel.size(0), 1), 1, device=device)), dim=-1) * delta_t)
                    final_coords = my_voxel_tool.world2smoke(final_pts)
                    mask_final = (final_coords >= -1) & (final_coords <= 1)
                    mask_final = mask_final.all(dim=1)
                    mask1 = active_mask.clone()
                    active_mask[mask1] = mask_final
                    nxt_pts[active_mask] = final_pts[mask_final]
                    nxt_b_coords[active_mask] = final_coords[mask_final]
                    
                    steps[active_mask] -= 1
                    steps[~active_mask] = 0
                    
                nxt_dens = model.dynamic_model_siren.density(nxt_pts)
                loss_v2d = criterion(nxt_dens, ori_dens)
                
                loss += args.lambda_v2d * loss_v2d

            optimizer.zero_grad()
            loss.backward()
            ## grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
        if trainVel:
            model.eval()
            batch_indices_np = shuffled_indices_np[b_idx: b_idx+batchsize]
            # batch_indices = shuffled_indices[b_idx: min(b_idx+batchsize, tot_points)]
            batch_indices = torch.from_numpy(batch_indices_np)
            w_batch_coords = batch_indices // (sampleY * sampleZ)
            h_batch_coords = (batch_indices % (sampleY * sampleZ)) // sampleZ
            d_batch_coords = batch_indices % sampleZ
            selected_coords = torch.stack((
                w_coords[w_batch_coords],
                h_coords[h_batch_coords],
                d_coords[d_batch_coords],
            ), dim=-1)
            
            perturbations = torch.zeros_like(selected_coords)
            perturbations[:, 0] = torch.rand(selected_coords.shape[0]) * 2 / sampleX - 1 / sampleX
            perturbations[:, 1] = torch.rand(selected_coords.shape[0]) * 2 / sampleY - 1 / sampleY
            perturbations[:, 2] = torch.rand(selected_coords.shape[0]) * 2 / sampleZ - 1 / sampleZ
            selected_coords = selected_coords + perturbations
            
            selected_coords = torch.cat((selected_coords, torch.rand(selected_coords.shape[0], 1) * 2 - 1), dim=1)
            
            b_coords = selected_coords.clone().to(device)
            b_coords.requires_grad_()
            cur_vel = vel_model(b_coords)
            
            if args.long_term:
                delta_t = 1.0 / frame_num

                pts = my_voxel_tool.smoke2world(b_coords)
                # b_raw = network_fn(pts)
                # ori_dens = F.relu(b_raw[..., -1:])
                ori_dens = model.dynamic_model_siren.density(pts)

                nxt_pts = pts
                nxt_b_coords = b_coords

                loss_adv = torch.tensor(0.0, device=device)

                for step in range(long_term_len):
                    adv_vel = vel_model(nxt_b_coords)
                    mid_pts = (
                        nxt_pts
                        + 0.5
                        * torch.cat(
                            (
                                adv_vel,
                                torch.full((adv_vel.size(0), 1), 1, device=device),
                            ),
                            dim=-1,
                        )
                        * delta_t
                    )
                    mid_b_coords = my_voxel_tool.world2smoke(mid_pts)
                    mask_mid = (mid_b_coords >= -1) & (mid_b_coords <= 1)
                    mask_mid = mask_mid.all(dim=1)
                    mid_b_coords = mid_b_coords[mask_mid]
                    nxt_b_coords = nxt_b_coords[mask_mid]
                    nxt_pts = nxt_pts[mask_mid]
                    ori_dens = ori_dens[mask_mid]

                    mid_adv_vel = vel_model(mid_b_coords)
                    nxt_pts = (
                        nxt_pts
                        + torch.cat(
                            (
                                mid_adv_vel,
                                torch.full((mid_adv_vel.size(0), 1), 1, device=device),
                            ),
                            dim=-1,
                        )
                        * delta_t
                    )
                    nxt_b_coords = my_voxel_tool.world2smoke(nxt_pts)
                    mask = (nxt_b_coords >= -1) & (nxt_b_coords <= 1)
                    mask = mask.all(dim=1)
                    nxt_b_coords = nxt_b_coords[mask]
                    nxt_pts = nxt_pts[mask]
                    ori_dens = ori_dens[mask]
                    # nxt_raw = network_fn(nxt_pts)
                    # nxt_dens = F.relu(nxt_raw[..., -1:])
                    nxt_dens = model.dynamic_model_siren.density(nxt_pts)
                    if nxt_dens.numel() == 0:
                        break
                    dens_diff = criterion(nxt_dens, ori_dens)
                    # p_dens_p_t = (nxt_dens - ori_dens) / delta_t
                    # dens_diff = torch.mean(torch.square(p_dens_p_t))
                    loss_adv = loss_adv + dens_diff * beta**step
            else:
                pts = my_voxel_tool.smoke2world(b_coords)
                pts.requires_grad_()

                b_den_raw = model.dynamic_model_siren.density(pts)
                # b_den_raw = F.relu(b_raw[..., -1:])
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
            
            loss_regular = torch.mean(torch.square(cur_vel))
            
            cur_vel = cur_vel.reshape(-1, 3)
            jacobian = []
            
            for i in range(cur_vel.shape[1]):
                grad_i = torch.autograd.grad(
                    outputs=cur_vel[:, i],
                    inputs=b_coords,
                    grad_outputs=torch.ones_like(cur_vel[:, i], device=cur_vel.device),
                    retain_graph=True,
                    create_graph=True 
                )[0]
                jacobian.append(grad_i.unsqueeze(1))
            
            grad_vel_sim = torch.cat(jacobian, dim=1)
            grad_vel_world = torch.matmul(grad_vel_sim, p_s_p_w.unsqueeze(0))
            div = (
                grad_vel_world[..., 0, 0]
                + grad_vel_world[..., 1, 1]
                + grad_vel_world[..., 2, 2]
            )
            loss_div = torch.mean(torch.square(div))


            if args.nsew:
                N = b_coords.shape[0]
                hes_vel = torch.zeros(N, 3, 4, 4)

                for i in range(3):
                    grad_u = torch.autograd.grad(cur_vel[:, i].sum(), b_coords, create_graph=True)[0]
                    for j in range(4):
                        hes_vel[:, i, j, :] = torch.autograd.grad(grad_u[:, j].sum(), b_coords, retain_graph=True, create_graph=True)[0]

                p_s_p_w_square = torch.matmul(p_s_p_w, p_s_p_w)
                hes_vel_world = torch.matmul(hes_vel, p_s_p_w_square.unsqueeze(0))

                omega_x = grad_vel_world[..., 2, 1] - grad_vel_world[..., 1, 2]
                omega_y = grad_vel_world[..., 0, 2] - grad_vel_world[..., 2, 0]
                omega_z = grad_vel_world[..., 1, 0] - grad_vel_world[..., 0, 1]

                L_w_L_t_x = (
                    (hes_vel_world[..., 2, 1, 3] - hes_vel_world[..., 1, 2, 3])
                    + cur_vel[..., 0]
                    * (hes_vel_world[..., 2, 0, 1] - hes_vel_world[..., 1, 0, 2])
                    + cur_vel[..., 1]
                    * (hes_vel_world[..., 2, 1, 1] + hes_vel_world[..., 1, 1, 2])
                    + cur_vel[..., 2]
                    * (hes_vel_world[..., 2, 2, 1] + hes_vel_world[..., 1, 2, 2])
                    - omega_x * grad_vel_world[..., 0, 0]
                    - omega_y * grad_vel_world[..., 0, 1]
                    - omega_z * grad_vel_world[..., 0, 2]
                )

                L_w_L_t_y = (
                    (hes_vel_world[..., 0, 2, 3] - hes_vel_world[..., 2, 0, 3])
                    + cur_vel[..., 0]
                    * (hes_vel_world[..., 0, 0, 2] - hes_vel_world[..., 2, 0, 0])
                    + cur_vel[..., 1]
                    * (hes_vel_world[..., 0, 1, 2] - hes_vel_world[..., 2, 1, 0])
                    + cur_vel[..., 2]
                    * (hes_vel_world[..., 0, 2, 2] - hes_vel_world[..., 2, 2, 0])
                    - omega_x * grad_vel_world[..., 1, 0]
                    - omega_y * grad_vel_world[..., 1, 1]
                    - omega_z * grad_vel_world[..., 1, 2]
                )

                L_w_L_t_z = (
                    (hes_vel_world[..., 1, 0, 3] - hes_vel_world[..., 0, 1, 3])
                    + cur_vel[..., 0]
                    * (hes_vel_world[..., 1, 0, 0] - hes_vel_world[..., 0, 0, 1])
                    + cur_vel[..., 1]
                    * (hes_vel_world[..., 1, 1, 0] - hes_vel_world[..., 0, 1, 1])
                    + cur_vel[..., 2]
                    * (hes_vel_world[..., 1, 2, 0] - hes_vel_world[..., 0, 2, 1])
                    - omega_x * grad_vel_world[..., 2, 0]
                    - omega_y * grad_vel_world[..., 2, 1]
                    - omega_z * grad_vel_world[..., 2, 2]
                )

                L_w_L_t = torch.stack((L_w_L_t_x, L_w_L_t_y, L_w_L_t_z), dim=-1)
                loss_nse = torch.mean(torch.square(L_w_L_t))
            else:
                D_vel_D_t_x = (
                    grad_vel_world[..., 0, 3]
                    + cur_vel[..., 0] * grad_vel_world[..., 0, 0]
                    + cur_vel[..., 1] * grad_vel_world[..., 0, 1]
                    + cur_vel[..., 2] * grad_vel_world[..., 0, 2]
                )
                D_vel_D_t_y = (
                    grad_vel_world[..., 1, 3]
                    + cur_vel[..., 0] * grad_vel_world[..., 1, 0]
                    + cur_vel[..., 1] * grad_vel_world[..., 1, 1]
                    + cur_vel[..., 2] * grad_vel_world[..., 1, 2]
                )
                D_vel_D_t_z = (
                    grad_vel_world[..., 2, 3]
                    + cur_vel[..., 0] * grad_vel_world[..., 2, 0]
                    + cur_vel[..., 1] * grad_vel_world[..., 2, 1]
                    + cur_vel[..., 2] * grad_vel_world[..., 2, 2]
                )
                D_vel_D_t = torch.stack((D_vel_D_t_x, D_vel_D_t_y, D_vel_D_t_z), dim=-1)
                loss_nse = torch.mean(torch.square(D_vel_D_t))
            
            if args.net_model == "hybrid_neus":
                pts = my_voxel_tool.smoke2world(b_coords)
                sdf = model.static_model.sdf(pts[..., :3])
                mask = sdf.squeeze(-1) <= 0
                loss_boundary = torch.mean(torch.square(cur_vel[mask]))
            else:
                loss_boundary = torch.tensor(0.0, device=device)
            
            loss_vel = (
                loss_adv
                + args.lambda_regular * loss_regular
                + args.lambda_div * loss_div
                + args.lambda_nse * loss_nse
                + args.lambda_boundary * loss_boundary
            )
            
            my_vel_loss.update(loss_vel.item())
            adv_loss.update(loss_adv.item())
            regular_loss.update(loss_regular.item())
            div_loss.update(loss_div.item())
            nse_loss.update(loss_nse.item())
            boundary_loss.update(loss_boundary.item())
        
            vel_optim.zero_grad()
            loss_vel.backward()
            torch.nn.utils.clip_grad_norm_(vel_model.parameters(), 1)
            vel_optim.step()
            
            b_idx += batchsize
            if b_idx >= tot_points:
                print("start draw velocity")
                point_num = args.Nx * args.Ny * args.Nz
                draw_indices = torch.randperm(point_num, device=device)
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
                        pixelvalues = vel_model(b_coords).squeeze()

                        with torch.no_grad():
                            im_estim[b_indices, :] = pixelvalues

                    # im_estim = torch.matmul(
                    #     im_estim, p_w_p_s[:3, :3].unsqueeze(0)
                    # ).squeeze()
                    # resX = float(args.vol_output_W)
                    st_scale = [resX/frame_num, resY/frame_num, resZ/frame_num]
                    im_estim = my_voxel_tool.vel_world2smoke(im_estim, st_scale)
                    im_estim_np = im_estim.cpu().detach().numpy()
                    im_estim_np = np.reshape(im_estim_np, (args.Nx, args.Ny, args.Nz, 3))
                    im_estim_np = np.swapaxes(im_estim_np, 0, 2)
                    estim_image = vel_uv2hsv(im_estim_np, scale=1000, is3D=True, logv=False)
                    
                    if frame == frame_num // 2:
                        imageio.imwrite(f"{targetdir}/image/vel_image_{cur_epoch:06d}.png", estim_image)
                    imgs.append(estim_image)

                video = np.stack(imgs, axis=0)
                imageio.mimwrite(
                    f"{targetdir}/video/vel_video_{cur_epoch:06d}.mp4",
                    video,
                    fps=25,
                    quality=8,
                )
                
                save_SIREN_model(vel_model, f"{targetdir}/ckpt/siren_ckpt_{cur_epoch:06d}.pth")
                
                b_idx = 0
                # shuffled_indices = torch.randperm(tot_points)
                shuffled_indices_np = np.random.permutation(tot_points)
                vel_scheduler.step()
                cur_epoch += 1
                print("finish one iteration")
            model.train()
        
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if args.adaptive_num_rays and args.cuda_ray == True and global_step > args.uniform_sample_step:
            samples_per_ray = extras["num_points"] / (extras["num_rays"] + 1e-6)
            num_rays = extras["num_rays"]
            cur_batch_size = num_rays * samples_per_ray
            N_rand = int(round((args.target_batch_size / cur_batch_size) * N_rand))

        # Rest is logging
        if global_step%args.i_weights==0:
            path = os.path.join(targetdir, '{:06d}.tar'.format(global_step))
            save_dic = {
                'global_step': global_step,
                'static_model_state_dict': model.static_model.state_dict() if not model.single_scene else None,
                'dynamic_model_lagrangian_state_dict': model.dynamic_model_lagrangian.state_dict(),
                'dynamic_model_siren_state_dict': model.dynamic_model_siren.state_dict() if model.use_two_level_density else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }
     
            torch.save(save_dic, path)
            print('Saved checkpoints at', path)
            
            if global_step > args.uniform_sample_step:
                occ_dynamic_path = os.path.join(targetdir, 'occ_dynamic_{:06d}.tar'.format(global_step))
                model.occupancy_grid_dynamic.save(occ_dynamic_path)
                occ_static_path = os.path.join(targetdir, 'occ_static_{:06d}.tar'.format(global_step))
                model.occupancy_grid_static.save(occ_static_path)
                print('Saved occ dynamic at', occ_dynamic_path, "occ static at", occ_static_path)
            
        if global_step%args.i_print==0:
            
            
            print(f"[TRAIN] Training stage: {training_stage} Iter: {global_step} Loss: {loss.item()}")
            print(f"CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0} GB\n")
            print(f"[TRAIN] lr: {new_lrate}")
            writer.add_scalar('Loss/loss', loss.item(), global_step)
            
            if rendering_loss_dict is not None:
                
             
                img_loss = rendering_loss_dict['img_loss']
                psnr = rendering_loss_dict['psnr']
                eikonal_loss = rendering_loss_dict['eikonal_loss']
                curvature_loss = rendering_loss_dict['curvature_loss']
                smoke_inside_sdf_loss = rendering_loss_dict['smoke_inside_sdf_loss']
                ghost_loss = rendering_loss_dict['ghost_loss']
                color_divergence_loss = rendering_loss_dict['color_divergence_loss'] if 'color_divergence_loss' in rendering_loss_dict.keys() else None
                
                print("img_loss: ", img_loss.item())
                writer.add_scalar('Loss/img_loss', img_loss.item(), global_step)
        
                print("PSNR: ", psnr.item())
                writer.add_scalar('Statistics/PSNR', psnr.item(), global_step)
                    
                if smoke_inside_sdf_loss is not None:
                    print("smoke_inside_sdf_loss: ", smoke_inside_sdf_loss.item())
                    writer.add_scalar('Loss/smoke_inside_sdf_loss', smoke_inside_sdf_loss.item(), global_step)
            
                if eikonal_loss is not None:
                    print("eikonal_loss: ", eikonal_loss.item())
                    writer.add_scalar('Loss/eikonal_loss', eikonal_loss.item(), global_step)

                if curvature_loss is not None:
                    print("curvature_loss: ", curvature_loss.item())
                    writer.add_scalar('Loss/curvature_loss', curvature_loss.item(), global_step)
                    
                if ghost_loss is not None:
                    print("ghost_loss: ", ghost_loss.item())
                    writer.add_scalar('Loss/ghost_loss', ghost_loss.item(), global_step)
                    
                if color_divergence_loss is not None:
                    print("color_divergence_loss: ", color_divergence_loss.item())
                    writer.add_scalar('Loss/color_divergence_loss', color_divergence_loss.item(), global_step)

                if "num_points" in extras:
                    samples_per_ray = extras["num_points"] / (extras["num_rays"] + 1e-6)
                    print("samples_per_ray: ", samples_per_ray)
                    writer.add_scalar('Statistics/samples_per_ray', samples_per_ray, global_step)

                if "num_points_static" in extras:
                    samples_per_ray_static = extras["num_points_static"] / (extras["num_rays"] + 1e-6)
                    print("samples_per_ray_static: ", samples_per_ray_static)
                    writer.add_scalar('Statistics/samples_per_ray_static', samples_per_ray_static, global_step)

                if "num_points_dynamic" in extras:
                    num_points_dynamic = extras["num_points_dynamic"] / (extras["num_rays"] + 1e-6)
                    print("num_points_dynamic: ", num_points_dynamic)
                    writer.add_scalar('Statistics/num_points_dynamic', num_points_dynamic, global_step)

                if args.adaptive_num_rays:
                    writer.add_scalar('Statistics/cur_batch_size', cur_batch_size, global_step)
                    writer.add_scalar('Statistics/batch_size_ratio', cur_batch_size/args.target_batch_size, global_step)
                    writer.add_scalar('Statistics/num_rays', N_rand, global_step)


                if not model.single_scene:
                    with torch.no_grad():
                        inv_s = model.get_deviation()         # Single parameter
                        print("s_val: ",  1.0 / inv_s.item())
                        writer.add_scalar('Statistics/s_val', 1.0 / inv_s.item(), global_step)

            if trainVel:
                print("my_vel_loss: ", my_vel_loss.avg)
                writer.add_scalar('Loss/my_vel_loss', my_vel_loss.avg, global_step)
                my_vel_loss.reset()
                
                print("adv_loss: ", adv_loss.avg)
                writer.add_scalar("Loss/adv_loss", adv_loss.avg, global_step)
                adv_loss.reset()
                
                print("regular_loss: ", regular_loss.avg)
                writer.add_scalar("Loss/regular_loss", regular_loss.avg, global_step)
                regular_loss.reset()
                
                print("div_loss: ", div_loss.avg)
                writer.add_scalar("Loss/div_loss", div_loss.avg, global_step)
                div_loss.reset()
                
                print("nse_loss", nse_loss.avg)
                writer.add_scalar("Loss/nse_loss", nse_loss.avg, global_step)
                nse_loss.reset()
                
                print("boundary_loss", boundary_loss.avg)
                writer.add_scalar("Loss/boundary_loss", boundary_loss.avg, global_step)
                boundary_loss.reset()


        if (global_step) % args.i_img==0:
            model.eval()
            voxel_den_list = voxel_writer.get_voxel_density_list(model, 0.5, args.chunk, 
                middle_slice=False)[::-1]
            
            voxel_img = []
            for voxel in voxel_den_list:
                voxel = voxel.detach().cpu().numpy()
                if voxel.shape[-1] == 1:
                    voxel_img.append(den_scalar2rgb(voxel, scale=None, is3D=True, logv=False, mix=True))
                else:
                    voxel_img.append(vel_uv2hsv(voxel, scale=300, is3D=True, logv=False))
            voxel_img = np.concatenate(voxel_img, axis=0) # 128,64*3,3
            imageio.imwrite( os.path.join(testimgdir, 'vox_{:06d}.png'.format(global_step)), voxel_img)
            model.train()
            
            if trainVel:
                point_num = args.Nx * args.Ny * args.Nz
                draw_indices = torch.randperm(point_num, device=device)
                im_estim = torch.zeros((point_num, 3), device=device)
                for img_b_idx in range(0, point_num, batchsize):
                    b_indices = draw_indices[img_b_idx : min(point_num, img_b_idx + batchsize)]
                    b_coords = center_location[b_indices, ...]
                    # extend to Nx4
                    b_coords = torch.cat(
                        (
                            b_coords,
                            torch.full(
                                (b_coords.size(0), 1),
                                0,
                                device=b_coords.device,
                            ),
                        ),
                        dim=1,
                    )
                    pixelvalues = vel_model(b_coords).squeeze()

                    with torch.no_grad():
                        im_estim[b_indices, :] = pixelvalues
                
                resX = float(args.vol_output_W)
                st_scale = [resX/frame_num, resY/frame_num, resZ/frame_num]
                im_estim = my_voxel_tool.vel_world2smoke(im_estim, st_scale)
                im_estim_np = im_estim.cpu().detach().numpy()
                im_estim_np = np.reshape(im_estim_np, (args.Nx, args.Ny, args.Nz, 3))
                im_estim_np = np.swapaxes(im_estim_np, 0, 2)
                estim_image = vel_uv2hsv(im_estim_np, scale=1000, is3D=True, logv=False)
                imageio.imwrite(os.path.join(testimgdir, 'vel_{:06d}.png'.format(global_step)), estim_image)

        if global_step % args.i_visualize==0:
            # resX = args.vol_output_W
            # resY = int(args.vol_output_W*float(voxel_scale[1])/voxel_scale[0]+0.5)
            # resZ = int(args.vol_output_W*float(voxel_scale[2])/voxel_scale[0]+0.5)
            
            # voxel_writer = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,resZ,resY,resX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)
            visualize_all(args, model, voxel_writer, t_info, global_step, targetdir)
                
        sys.stdout.flush()
        # torch.cuda.empty_cache()
        # local_step += 1


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
    