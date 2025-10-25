import torch

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
        # pts_clone = pts.clone()
        # Pworld = pts_clone[:, :3]
        # pos_rot = torch.sum(
        #     Pworld[..., None, :] * (self.w2s[:3, :3]), -1
        # )  # 4.world to 3.target
        # pos_off = (self.w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
        # new_pose = pos_rot + pos_off
        # pos_scale = new_pose / (self.scale)  # 3.target to 2.simulation
        # pts_clone[:, :3] = pos_scale
        # coords = pts_clone * 2.0 - 1.0
        # return coords
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
        _st_factor = torch.Tensor(st_factor).expand((3, )).to(self.device)
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