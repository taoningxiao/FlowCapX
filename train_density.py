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
    resX = 64 # complexity O(N^3)
    resY = int(resX*float(voxel_scale[1])/voxel_scale[0]+0.5)
    resZ = int(resX*float(voxel_scale[2])/voxel_scale[0]+0.5)
    voxel_writer = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,resZ,resY,resX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)

    # training_voxel: to sample data for for velocity NSE training
    # training_voxel should have a larger resolution than voxel_writer
    # note that training voxel is also used for visualization in testing
    min_ratio = float(64+4*2)/min(voxel_scale[0],voxel_scale[1],voxel_scale[2])
    minX = int(min_ratio*voxel_scale[0]+0.5)
    trainX = max(args.vol_output_W,minX) # a minimal resolution of 64^3
    trainY = int(trainX*float(voxel_scale[1])/voxel_scale[0]+0.5)
    trainZ = int(trainX*float(voxel_scale[2])/voxel_scale[0]+0.5)
    training_voxel = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,trainZ,trainY,trainX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)
    training_pts = torch.reshape(training_voxel.pts, (-1,3)) 

    ## spatial alignment from wolrd coord to simulation coord
    train_reso_scale = torch.Tensor([256*t_info[-1],256*t_info[-1],256*t_info[-1]])

  
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
    
    adv_loss, regular_loss, div_loss, nse_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    my_vel_loss = AverageMeter()
    
    decay_rate = 0.1
    decay_steps = args.lrate_decay * 1000
    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    # new_lrate_vel = args.lrate_vel * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    
    for global_step in trange(start + 1, N_iters + 1):
        training_stage = 1 
        trainImg = True
        trainVel = False
            
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
                local_step += 1
                continue
                

            rendering_loss, rendering_loss_dict = get_rendering_loss(args, model, rgb, acc, target_s, bg_color, extras, time_locate, global_step, target_mask)
            loss += rendering_loss
            
            smoke_samples_xyz = extras['samples_xyz_dynamic'] # has bugs for uniform sample
                            
            optimizer.zero_grad()
            loss.backward()
            ## grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        new_lrate_vel = args.lrate_vel * (decay_rate ** (global_step / decay_steps))
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

        if global_step % args.i_visualize==0:
            resX = args.vol_output_W
            resY = int(args.vol_output_W*float(voxel_scale[1])/voxel_scale[0]+0.5)
            resZ = int(args.vol_output_W*float(voxel_scale[2])/voxel_scale[0]+0.5)
            
            voxel_writer = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,resZ,resY,resX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)
            visualize_all(args, model, voxel_writer, t_info, global_step, targetdir)
                
        sys.stdout.flush()
        torch.cuda.empty_cache()
        local_step += 1


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
    