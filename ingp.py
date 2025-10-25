import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
# from torch.func import jacrev, vmap
import taichi as ti
from src.utils.args import config_parser


class NeRFSmallPotential(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=16,
                 input_ch=3,
                 use_f=False
                 ):
        super(NeRFSmallPotential, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = hidden_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = nn.ModuleList(sigma_net)
        self.out = nn.Linear(hidden_dim, 3, bias=True)
        self.use_f = use_f
        if use_f:
            self.out_f = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.out_f2 = nn.Linear(hidden_dim, 3, bias=True)


    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = F.relu(h, True)

        v = self.out(h)
        if self.use_f:
            f = self.out_f(h)
            f = F.relu(f, True)
            f = self.out_f2(f)
        else:
            f = v * 0
        return v, f


class INGP(nn.Module):
    def __init__(self, args, device):
        super(INGP, self).__init__()
        from taichi_encoders.hash4 import Hash4Encoder
        self.max_res = np.array([args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v_t])
        self.min_res = np.array([args.base_resolution_v, args.base_resolution_v, args.base_resolution_v, args.base_resolution_v_t])

        self.embed_fn = Hash4Encoder(max_res=self.max_res, min_res=self.min_res, num_scales=args.num_levels,
                                max_params=2 ** args.log2_hashmap_size)
        self.input_ch = self.embed_fn.num_scales * 2  # default 2 params per scale
        self.embedding_params = list(self.embed_fn.parameters())


        self.model = NeRFSmallPotential(num_layers=args.vel_num_layers,
                                hidden_dim=64,
                                geo_feat_dim=15,
                                num_layers_color=2,
                                hidden_dim_color=16,
                                input_ch=self.input_ch,
                                use_f=args.use_f).to(device)
        self.grad_vars = list(self.model.parameters())
        print(self.model)
        print('Total number of trainable parameters in model: {}'.format(
            sum([p.numel() for p in self.model.parameters() if p.requires_grad])))
        print('Total number of parameters in embedding: {}'.format(
            sum([p.numel() for p in self.embedding_params if p.requires_grad])))

        self.optimizer = torch.optim.RAdam([
            {'params': self.grad_vars, 'weight_decay': 1e-6},
            {'params': self.embedding_params, 'eps': 1e-15}
        ], lr=args.lrate_ingp, betas=(0.9, 0.99))


    def forward(self, x):
        with torch.enable_grad():
            h = self.embed_fn(x)
            v, f = self.model(h)
            return v
    
    # def compute_with_jacobian(self, x):
    #     with torch.enable_grad():
    #         h = self.embed_fn(x)
    #         vel_output, f_output = self.model(h)

    #         def g(x):
    #             return self.model(x)[0]

    #         jac = vmap(jacrev(g))(h)
    #         # print('jac', jac.shape)
    #         jac_x = [] #_get_minibatch_jacobian(h, pts)
    #         for j in range(h.shape[1]):
    #             dy_j_dx = torch.autograd.grad(
    #                 h[:, j],
    #                 x,
    #                 torch.ones_like(h[:, j], device=h.get_device()),
    #                 retain_graph=True,
    #                 create_graph=True,
    #             )[0].view(x.shape[0], -1)
    #             jac_x.append(dy_j_dx.unsqueeze(1))
    #         jac_x = torch.cat(jac_x, dim=1)
    #         # print(jac_x.shape)
    #         jac = jac @ jac_x
    #         return vel_output, jac
        
    def save(self, path):
        torch.save({
            'vel_network_fn_state_dict': self.model.state_dict(),
            'vel_embed_fn_state_dict': self.embed_fn.state_dict(),
            'vel_optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, args, device='cuda'):
        ckpt = torch.load(path, map_location=device)
        obj = cls(args, device)
        model_dict = obj.model.state_dict()
        pretrained_dict = ckpt['vel_network_fn_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        obj.model.load_state_dict(model_dict)
        # obj.model.to(device)
        print("Updated parameters:{}/{}".format(len(pretrained_dict), len(model_dict)))
        obj.embed_fn.load_state_dict(ckpt['vel_embed_fn_state_dict'])
        obj.optimizer.load_state_dict(ckpt['vel_optimizer_state_dict'])
        return obj


def add_parser(parser):
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_duration", type=int, default=120)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batchsize", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.71)
    parser.add_argument("--i_draw", type=int, default=1)
    parser.add_argument("--i_save", type=int, default=1)
    parser.add_argument("--draw_v", action="store_true")
    parser.add_argument("--draw_w", action="store_true")
    parser.add_argument("--draw_t", action="store_true")
    parser.add_argument("-l", type=bool)
    parser.add_argument("--vis_vel_only", action="store_true")
    parser.add_argument("--grad_scale", type=float, default=0.0)
    
    parser.add_argument("--finest_resolution_v", type=int, default=128)
    parser.add_argument("--finest_resolution_v_t", type=int, default=128)
    parser.add_argument("--base_resolution_v", type=int, default=16)
    parser.add_argument("--base_resolution_v_t", type=int, default=16)
    parser.add_argument("--num_levels", type=int, default=16)
    parser.add_argument("--log2_hashmap_size", type=int, default=19)
    parser.add_argument("--vel_num_layers", type=int, default=2)
    parser.add_argument("--use_f", action='store_true', default=False)
    return parser


if __name__ == "__main__":
    ti.init(arch=ti.cuda, device_memory_GB=12.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    parser = config_parser()
    parser = add_parser(parser)
    args = parser.parse_args()

    import torch.nn as nn
    import torch.nn.functional as F

    print(torch.__version__)
    
    model = INGP(args, device)
    
    pts = torch.rand(100, 4)
    pts.requires_grad = True
    
    vel, jac = model.compute_with_jacobian(pts)
    
    # network_vel = NeRFSmallPotential(input_ch=32).to(device)
    # from taichi_encoders.hash4 import Hash4Encoder
    # embed_vel = Hash4Encoder()

    # pts = torch.rand(100, 4)
    # pts.requires_grad = True
    # with torch.enable_grad():
    #     h = embed_vel(pts)
    #     vel_output, f_output = network_vel(h)

    #     print('vel_output', vel_output.shape)
    #     print('h', h.shape)
    #     def g(x):
    #         return network_vel(x)[0]

    #     jac = vmap(jacrev(g))(h)
    #     print('jac', jac.shape)
    #     jac_x = [] #_get_minibatch_jacobian(h, pts)
    #     for j in range(h.shape[1]):
    #         dy_j_dx = torch.autograd.grad(
    #             h[:, j],
    #             pts,
    #             torch.ones_like(h[:, j], device=device),
    #             retain_graph=True,
    #             create_graph=True,
    #         )[0].view(pts.shape[0], -1)
    #         jac_x.append(dy_j_dx.unsqueeze(1))
    #     jac_x = torch.cat(jac_x, dim=1)
    #     print(jac_x.shape)
    #     jac = jac @ jac_x
    #     assert jac.shape == (pts.shape[0], 3, 4)
    #     _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)
    
    print("vel.shape: ", vel.shape)
    print("jac.shape: ", jac.shape)