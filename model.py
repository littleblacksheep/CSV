import torch
import torch.nn as nn

def get_t(first, frame_interval, s_h, n_frames, mode):
    '''get relevant time indices'''
    t = []
    if mode == "all_frames":  # only decode frames where labels are given
        for n in range(1, n_frames + 1):
            if n % frame_interval == 0 or n == n_frames:
                if len(t) < 1000:
                    t.append(int(s_h * (n - first)))
                else:
                    raise ValueError()
    elif mode == "last_frame":  # only decode last frame where label is given
        t.append(int(s_h * (n_frames - first)))
    elif mode == "inference":  # decode all frames, regardless of labels
        for n in range(1, n_frames + 1):
            t.append(int(s_h * (n - first)))
    return t

def concat_time(x, info, n_frames_nonzero_max, frame_interval):
    if info["mode"] == "last_frame":
        n_frames = (info["z0"][:, 0].float().cuda() / n_frames_nonzero_max).unsqueeze(-1).unsqueeze(-1)  # [b,1,1]
        x = torch.cat((x, n_frames), dim=2).unsqueeze(0) # [1,b,1,n_x+1]
    elif info["mode"] == "all_frames":
        s = x.shape
        x_combine = torch.zeros((0, s[2]+1), device=x.device)  # [T,n_x+1]
        for b in range(s[0]):
            n_frames = int(info["z0"][b, 0].item())
            n_frames_nonzero = int(info["z0"][b, 1].item())
            for n in range(1, n_frames + 1):
                if n % frame_interval == 0 or n == n_frames:
                    x_combine_b = torch.cat((x[b,0,:], n * torch.ones_like(x[b,0,:1]) / n_frames_nonzero_max))  # [1,n_x+1]
                    x_combine = torch.cat((x_combine, x_combine_b.unsqueeze(0)), dim=0)
        x = x_combine.unsqueeze(0).unsqueeze(2)  # [1,T,1,n_x+1]
    return x

def t_end(info, s_h):
    n_frames = info["z0"][:, 0]
    n_steps_model = s_h * n_frames
    n_steps_max = int(n_steps_model.max().item())  # number of steps to take for minibatch
    return n_steps_max, n_steps_model

def x2y(x, n_z_agg):
    """x_setup --> y_agg"""
    x = x.squeeze(1)  # [b,n_x]
    y = torch.zeros((x.shape[0], n_z_agg), device=x.device)
    y[:,0] = x[:,1] / (1. + x[:,2])  # largest aggregate = target mass = total mass / (1 + gamma)
    y[:,1] = x[:,5]  # core mass fraction target
    y[:,2] = x[:,6]  # shell mass fraction target
    y[:,3] = (x[:,2] * x[:,1]) / (1. + x[:,2])  # 2nd largest aggregate = projectile mass = gamma * total mass / (1 + gamma)
    y[:,4] = x[:,3]  # core mass fraction projectile
    y[:,5] = x[:,4]  # shell mass fraction projectile
    y[:,6] = 0.  # rest
    y[:,7] = 0.  # core mass fraction rest
    y[:,8] = 0.  # shell mass fraction rest
    y[:,9:9+6] = x[:,19:19+6]  # target
    y[:,15:15+6] = x[:,13:13+6]  # projectile
    y[:,21:21+6] = 0.  # rest
    return y

def stack(y_seq, first, frame_interval, s_h, info):
    '''postprocess into same shape as labels'''
    y = torch.zeros((0, y_seq.shape[2]), device=y_seq.device)  # [T,n_z]
    for b in range(y_seq.shape[0]):
        t = get_t(first, frame_interval, s_h, info["z0"][b,0].long(), info["mode"])
        y = torch.cat((y, y_seq[b, t, :]), dim=0)
    return y.unsqueeze(0).unsqueeze(2) # [1,T,1,n_z]

def rotate(cfg, x):
    rot_info = None
    if cfg["ML"]["dataset"]["winter"]["augmentation"]:
        v = x[:, 0, 19 + 3:19 + 6] - x[:, 0, 16:16 + 3]
        d = x[:, 0, 19:19 + 3] - x[:, 0, 13:13 + 3]
        e0 = v  # [b,3]
        e2 = torch.cross(e0, d, dim=1)  # [b,3]
        e1 = torch.cross(e0, e2, dim=1)
        e0 /= torch.norm(e0, dim=1, keepdim=True)
        e1 /= torch.norm(e1, dim=1, keepdim=True)
        e2 /= torch.norm(e2, dim=1, keepdim=True)
        basis_old = torch.cat((e0.unsqueeze(1), e1.unsqueeze(1), e2.unsqueeze(1)), dim=1)  # orthonormal basis vectors before rotation, [b,3,3]
        """
        basis_new = torch.diag(torch.ones(3, device=x.device)).unsqueeze(0).repeat(x.shape[0], 1, 1)  # orthonormal basis vectors after rotation, [b,3,3]
        R = torch.zeros((x.shape[0], 3, 3), device=x.device)
        for i in range(3):
            for j in range(3):
                R[:, i, j] = torch.einsum('bs,bs->b', basis_new[:, i, :], basis_old[:, j, :])
        """
        R = torch.inverse(-basis_old)  # Rotation matrix: basis_old --> basis_new
        rot_info = R
        # apply rotation:
        if cfg["ML"]["dataset"]["winter"]["inputs"] == "setup":
            cols_x = [7, 10, 13, 16, 19, 22]
        else:
            raise NotImplementedError()
        for c in cols_x:
            x[:,0,c:c+3] = torch.einsum('bsv,bs->bv', R, x[:,0,c:c+3])
    return x, rot_info

def derotate(cfg, x, rot_info):
    if cfg["ML"]["dataset"]["winter"]["augmentation"]:
        R = rot_info
        R_inv = R.permute(0, 2, 1)  # inverse is transposed
        # apply rotation (all frames):
        if cfg["ML"]["dataset"]["winter"]["targets"] == "agg":
            cols_z = [9, 12, 15, 18, 21, 24]
        else:
            raise NotImplementedError()
        for c in self.cols_z:
            x[:,:,c:c+3] = torch.einsum('bsv,bts->btv', R_inv, x[:,:,c:c+3])
    return x

@torch.no_grad()
def selu_weights_init(l, n_inp):
    nn.init.normal_(l.weight, 0., (1. / n_inp)**0.5)
    try:
        nn.init.constant_(l.bias, 0.)
    except:
        pass

class block(nn.Module):
    """neural network block"""
    def __init__(self, n_inp, n_h, n_out, n_l, gain=1.):
        super().__init__()
        self.n_l = n_l
        if self.n_l == 1:
            n_h = n_out
        l = []
        for i in range(self.n_l):
            if i == 0:
                a, b = n_inp, n_h
            elif i == self.n_l - 1:
                a, b = n_h, n_out
            else:
                a, b = n_h, n_h
            l.append(nn.Linear(a, b))
        self.l = nn.ModuleList(l)
        self.act = nn.SELU()
        self.reset_params(gain)
        print(f"INFO: init block: n_inp: {n_inp}, n_h: {n_h}, n_out: {n_out}, n_l: {n_l}, gain: {gain}")

    def reset_params(self, gain):
        for i in range(self.n_l):
            selu_weights_init(self.l[i], self.l[i].weight.shape[1])
        self.l[self.n_l - 1].weight.data *= gain
        self.l[self.n_l - 1].bias.data *= gain

    def forward(self, x):
        for i in range(self.n_l):
            if i < self.n_l - 1:
                x = self.act(self.l[i](x))
            else:
                x = self.l[i](x)  # no activation
        return x

class pm:
    '''perfect merging'''
    def __init__(self, cfg, info=False):
        self.n_z = cfg["ML"]["dataset"]["winter"]["n_z_agg"]

    def train(self): return
    def eval(self): return

    def forward(self, x, _):
        x = x.squeeze(1)  # [b,1,n_x] --> [b,n_x]
        s_t = x[:, 19:19 + 6]  # pos and vel target
        s_p = x[:, 13:13 + 6]  # pos and vel projectile
        m_tot = x[:, 1]
        gamma = x[:, 2]
        m_t = m_tot / (gamma + 1.)
        m_p = gamma * m_t

        # inelastic perfect merging with momentum conservation:
        s = ((m_t.unsqueeze(-1) * s_t[:, :] + m_p.unsqueeze(-1) * s_p[:, :]) / m_tot.unsqueeze(-1)).squeeze(-1)

        m_mat0 = m_t * x[:, 5] + m_p * x[:, 3]  # total mass of material 0 (core)
        m_mat2 = m_t * x[:, 6] + m_p * x[:, 4]  # total mass of material 2 (shell)
        m_mat1 = m_t * (1. - x[:, 5] - x[:, 6]) + m_p * (1. - x[:, 3] - x[:, 4])  # total mass of material 1 (mantle)

        y = torch.zeros((x.shape[0], self.n_z), device=x.device)
        y[:, 0] = m_tot
        y[:, 1] = m_mat0 / m_tot
        y[:, 2] = m_mat2 / m_tot
        y[:, 3] = 0.
        y[:, 4] = 0.
        y[:, 5] = 0.
        y[:, 6] = 0.
        y[:, 7] = 0.
        y[:, 8] = 0.
        y[:, 9:9 + 6] = s[:]
        y[:, 15:15 + 6] = 0.
        y[:, 21:21 + 6] = 0.
        y_lbl = y.unsqueeze(0).unsqueeze(2)  # [1,b,1,n_z]
        out = {"y_lbl": y_lbl, "L_reg": torch.zeros(1), "w" : None}
        return out

class ffn(nn.Module):
    '''feed-forward net or linear model'''
    def __init__(self, cfg, info=False, linear=False):
        super().__init__()
        self.cfg = cfg
        self.frame_interval = cfg["SPH"]["sim"]["random_setup"]["frame_interval"]
        self.n_frames_nonzero_max = cfg["ML"]["dataset"]["winter"]["n_frames_nonzero_max"]
        n_h = cfg["ML"]["model"]["ffn"]["n_h"]
        n_x = cfg["ML"]["dataset"]["winter"]["n_x_setup"] + 1
        n_z = cfg["ML"]["dataset"]["winter"]["n_z_agg"]
        n_l = cfg["ML"]["model"]["ffn"]["n_l"]
        if linear: n_l = 1
        self.block = block(n_x, n_h, n_z, n_l)

    def forward(self, x, info):
        x, rot_info = rotate(self.cfg, x)
        x = concat_time(x, info, self.n_frames_nonzero_max, self.frame_interval)  # [1,b,1,n_x+1]
        x = self.block(x)  # [1,b,1,n_z]
        x = x.squeeze(0)
        x = derotate(self.cfg, x, rot_info)
        x = x.unsqueeze(0)
        out = {"y_lbl": x, "L_reg": torch.zeros(1, device=x.device), "w" : None}
        return out

class res(nn.Module):
    '''weight-tied residual net'''
    def __init__(self, cfg, info=False):
        super().__init__()
        self.cfg = cfg
        self.n_x = cfg["ML"]["dataset"]["winter"]["n_x_setup"]
        self.n_z = cfg["ML"]["dataset"]["winter"]["n_z_agg"]
        self.n_h = cfg["ML"]["model"]["res"]["n_h"]
        self.n_l = cfg["ML"]["model"]["res"]["n_l"]
        self.reg_drift = cfg["ML"]["model"]["res"]["reg_drift"]
        self.reg_drift_lim = cfg["ML"]["model"]["res"]["reg_drift_lim"]
        self.first = cfg["ML"]["model"]["res"]["first"]
        self.s_h = cfg["ML"]["model"]["res"]["s_h"]
        self.frame_interval = cfg["SPH"]["sim"]["random_setup"]["frame_interval"]
        self.res = block(2 * self.n_h, self.n_h, self.n_h, self.n_l, gain=0.1/self.s_h)
        self.enc = block(self.n_z+6, max(self.n_z+6, 2*self.n_h), 2*self.n_h, self.n_l, gain=0.1)
        self.dec = block(self.n_h, max(self.n_h, self.n_z+6), self.n_z+6, self.n_l, gain=0.1/self.s_h)

    def mse(self, a, b):
        return ((a - b) ** 2).mean()

    def forward(self, x, info):
        x, rot_info = rotate(self.cfg, x)
        y = x2y(x, self.n_z)
        y = torch.cat((y, x[:,0,7:7+6]), dim=1)  # concat spin
        hci = self.enc(y)
        h, inp = hci[:, :self.n_h], hci[:, self.n_h:]
        bs = h.shape[0]

        h_seq = torch.zeros((bs, 0, self.n_h), device=h.device)
        y_seq = torch.zeros((bs, 0, self.n_z+6), device=y.device)
        n_steps_max, n_steps_res = t_end(info, self.s_h)
        for t in range(n_steps_max):

            h = h + self.res(torch.cat((h, inp), dim=1))
            y = y + self.dec(h)

            hci = self.enc(y)
            inp = hci[:, self.n_h:]

            h_seq = torch.cat((h_seq, h.unsqueeze(1)), dim=1)
            y_seq = torch.cat((y_seq, y.unsqueeze(1)), dim=1)

        # regularization to avoid diverging sequences:
        L_reg = torch.zeros(1, device=h.device)
        if self.reg_drift:
            h_seq_ = h_seq.view(-1)
            h_neg = h_seq_[h_seq_ < -self.reg_drift_lim]
            h_pos = h_seq_[h_seq_ > self.reg_drift_lim]
            if h_neg.shape[0] > 0: L_reg += self.mse(h_neg, -self.reg_drift_lim * torch.ones_like(h_neg))
            if h_pos.shape[0] > 0: L_reg += self.mse(h_pos, self.reg_drift_lim * torch.ones_like(h_pos))

        y_seq = y_seq[:,:,:self.n_z]  # remove dummy spin units
        y_seq = derotate(self.cfg, y_seq, rot_info)  # [b,T,n_z]
        y_lbl = stack(y_seq, self.first, self.frame_interval, self.s_h, info)  # [1,T,1,n_z]
        out = {"y_lbl" : y_lbl, "L_reg" : L_reg, "y_seq" : y_seq, "h_seq" : h_seq, "n_steps_res" : n_steps_res}
        return out