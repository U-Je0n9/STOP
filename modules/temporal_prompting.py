import torch
import torch.nn as nn
from collections import deque

TP_DEBUG_BUFFER = deque(maxlen=100)
TP_DEBUG_TRIGGERED = False

def _norm_stats(prefix, x):
    with torch.no_grad():
        x = x.detach().float()
        n = x.norm(dim=-1)

        return {
            f"{prefix}_norm_mean": n.mean().item(),
            f"{prefix}_norm_max": n.max().item(),
            f"{prefix}_mean": x.mean().item(),
            f"{prefix}_std": x.std().item(),
        }

def _tp_summarize_tensor(x, print_rows=False):
    info = {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "nan": torch.isnan(x).any().item(),
        "inf": torch.isinf(x).any().item(),
    }

    x_float = x.detach().float()
    finite_mask = torch.isfinite(x_float)

    if finite_mask.any():
        finite_vals = x_float[finite_mask]
        info["finite_min"] = finite_vals.min().item()
        info["finite_max"] = finite_vals.max().item()
        info["finite_mean"] = finite_vals.mean().item()
    else:
        info["finite_min"] = None
        info["finite_max"] = None
        info["finite_mean"] = None

    if print_rows and x.dim() >= 2:
        try:
            bad_rows = torch.where(~torch.isfinite(x_float.view(x.shape[0], -1)).all(dim=1))[0]
            info["bad_rows"] = bad_rows.detach().cpu().tolist()
        except Exception:
            info["bad_rows"] = "unavailable"

    return info

def _tp_flush_debug_buffer(header=None):
    if header is not None:
        print(header)
    print("[TP_DEBUG] ===== buffered tensor trace start =====")
    for line in TP_DEBUG_BUFFER:
        print(line)
    print("[TP_DEBUG] ===== buffered tensor trace end =====")

def _debug_tensor(name, x, print_rows=False):
    global TP_DEBUG_TRIGGERED

    if x is None:
        msg = f"[TP_DEBUG] {name}: None"
        TP_DEBUG_BUFFER.append(msg)
        if TP_DEBUG_TRIGGERED:
            print(msg)
        return False

    with torch.no_grad():
        info = _tp_summarize_tensor(x, print_rows=print_rows)
        msg = (
            f"[TP_DEBUG] {name}: "
            f"shape={info['shape']}, dtype={info['dtype']}, device={info['device']}, "
            f"nan={info['nan']}, inf={info['inf']}, "
            f"finite_min={info['finite_min']}, finite_max={info['finite_max']}, finite_mean={info['finite_mean']}"
        )
        if "bad_rows" in info:
            msg += f", bad_rows={info['bad_rows']}"

        TP_DEBUG_BUFFER.append(msg)

        bad = info["nan"] or info["inf"]

        if bad and not TP_DEBUG_TRIGGERED:
            TP_DEBUG_TRIGGERED = True
            _tp_flush_debug_buffer(header=f"[TP_DEBUG] NaN/Inf first detected at {name}")

        if TP_DEBUG_TRIGGERED:
            print(msg)

        return bad

def get_TemporalPrompt(args):
    if args.temporal_prompt in ['group2-2']:
        return TemporalPrompt_3(args=args)

class TrajectoryPrompter(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=768, dropout=0.0, scale=0.05):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.last_debug_stats = {}
        self.scale = scale

    def forward(self, delta_h):
        """
        delta_h: [B, T-1, D]
        return: [B, 1, D]
        """

        C = torch.cumsum(delta_h, dim=1)   # [B, T-1, D]

        C_first = C[:, 0, :]               # [B, D]
        C_global = C.mean(dim=1)           # [B, D]
        C_last = C[:, -1, :]               # [B, D]

        traj_feat = torch.cat(
            [C_first, C_global, C_last],
            dim=-1
        )                                  # [B, 3D]

        p_traj = self.mlp(traj_feat)       # [B, D]
        p_traj = p_traj * self.scale
        self.last_debug_stats = {}

        self.last_debug_stats.update(
            _norm_stats("traj_delta_h", delta_h)
        )

        self.last_debug_stats.update(
            _norm_stats("traj_p", p_traj)
        )
        return p_traj.unsqueeze(1)         # [B, 1, D]

class TemporalPrompt_3(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        kernel_spatial = 11
        kernel_temporal = 3
        kernel = (kernel_temporal, kernel_spatial, kernel_spatial)
        padding = (int((kernel_temporal-1)/2), int((kernel_spatial-1)/2), int((kernel_spatial-1)/2))
        hid_dim_1 = 9
        hid_dim_2 = 9
        hid_dim_3 = 16
        hid_dim_l1 = 16
        
        self.Conv = nn.Sequential(
            nn.Conv3d(3, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.Conv3d(hid_dim_1, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_1, hid_dim_2, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_2, hid_dim_3, kernel_size=kernel, stride=1, padding=padding),
        )
        self.MLP = nn.Sequential(
            nn.Linear(hid_dim_3, hid_dim_l1),
            nn.PReLU(),
            nn.Linear(hid_dim_l1, 3),
        )
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        
        padding_tmp = (int((5-1)/2), int((11-1)/2), int((11-1)/2))
        self.Cal_Net = nn.Conv3d(3, 1, kernel_size=(5, 11, 11), stride=1, padding=padding_tmp)
        self.eta = 6
        
        self.InterFramePrompt = self.init_InterFramePrompt(args)
        self.use_traj_prompt = getattr(args, "use_traj_prompt", 0)

        if self.use_traj_prompt:
            self.TrajectoryPrompter = TrajectoryPrompter(
                embed_dim=768,
                hidden_dim=768,
                dropout=getattr(args, "traj_prompt_dropout", 0.0),
                scale=getattr(args, "traj_prompt_scale", 0.05),
            )
        
    def forward(self, x):
        B, T, C, W, H = x.shape
        with torch.amp.autocast("cuda", enabled=False): # 추가
            prompt = x.permute(0, 2, 1, 3, 4)
            _debug_tensor("TP3.forward.input_prompt", prompt, print_rows=True)
            prompt = self.Conv(prompt)
            _debug_tensor("TP3.forward.after_conv", prompt, print_rows=True)
            prompt = self.dropout1(prompt)
            _debug_tensor("TP3.forward.after_dropout1", prompt, print_rows=True)
            prompt = prompt.permute(0, 2, 3, 4, 1)
            _debug_tensor("TP3.forward.before_mlp", prompt, print_rows=True)
            prompt = self.MLP(prompt)
            _debug_tensor("TP3.forward.after_mlp", prompt, print_rows=True)
            prompt = self.dropout2(prompt)
            prompt = prompt.permute(0, 1, 4, 2, 3)
            prompt = self.dropout2(prompt)
        mask = self.get_mask(x)
        _debug_tensor("TP3.forward.mask", mask, print_rows=True)
        out= x + prompt*0.05 + prompt*mask*0.05
        _debug_tensor("TP3.forward.output", out, print_rows=True)
        return out
    
    def get_mask(self, x):
        B, T, C, W, H = x.shape
        self.B, self.T = B, T
        x = x.permute(0, 2, 1, 3, 4)
        x = self.Cal_Net(x)
        x = x.squeeze(1)
        x = x.reshape(B, T, 32, 7, 32, 7)
        x = x.mean(dim=(2, 4))
        bar = x.reshape(B, T, -1)
        bar = bar.sort(dim=2, descending=True)[0]
        bar = bar[:, :, self.eta]
        x = x > bar.unsqueeze(2).unsqueeze(3)
        self.mask = x
        x = x.unsqueeze(2).unsqueeze(4)
        x = x.repeat(1, 1, 32, 1, 32, 1)
        x = x.reshape(B, T, 224, 224)
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, 3, 1, 1)
        return x
    
    def init_InterFramePrompt(self, args):
        self.Attention = nn.MultiheadAttention(embed_dim = 768, num_heads = 12)
        kernel_temporal = 3 
        kernel_token = 15
        kernel_hid = 25
        kernel = (kernel_temporal, kernel_token, kernel_hid)
        padding = (int((kernel_temporal-1)/2), int((kernel_token-1)/2), int((kernel_hid-1)/2))
        hid_dim_1 = 9
        hid_dim_l1 = 16
        
        self.InterConv = nn.Sequential(
            nn.Conv3d(1, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_1, 1, kernel_size=kernel, stride=1, padding=padding),
        )
        self.InterMLP = nn.Sequential(
            nn.Linear(49, 16),
            nn.PReLU(),
            nn.Linear(16, 4),
        )
        padding_tmp = (int((5-1)/2), int((11-1)/2), int((11-1)/2))
        self.Cal_Net_Inter = nn.Conv3d(1, 1, kernel_size=(5, 11, 11), stride=1, padding=padding_tmp)
        
    def get_mask_Inter(self, x):
        x = self.Cal_Net_Inter(x)
        x = x.squeeze(1)
        mask_tmp = self.mask.reshape(self.B, self.T, -1)
        mask_tmp = mask_tmp.unsqueeze(3)
        x = x + x*mask_tmp
        x = x.mean(dim=(2, 3))
        denom = x.max() - x.min()
        x = (x - x.min()) / (denom + 1e-6)
        # x = (x-x.min())/(x.max()-x.min())
        return x

    
    def get_inter_frame_prompt(self, x):
        BT, L, D = x.shape
        x = x.reshape(self.B, self.T, L, D)
        x = x[:,:,1:,:]
        x = x.reshape(self.B, -1, D)
        x = x.permute(1, 0, 2)
        x = self.Attention(x, x, x)[0]
        x = x.permute(1, 0, 2)
        x = x.reshape(self.B, self.T, -1, D)
        x = x.unsqueeze(1)
        mask = self.get_mask_Inter(x)
        x = self.InterConv(x)
        x = x.squeeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.InterMLP(x)
        x = x.permute(0, 1, 3, 2)
        mask = mask.unsqueeze(2).unsqueeze(3)
        x = x+x*mask
        x = x.reshape(self.B, -1, 768)
        x = x.unsqueeze(0)
        x = x.repeat(12, 1, 1, 1)
        return x*0.05

    def get_trajectory_prompt(self, x):
        """
        x: [B*T, L, D]
        ln_pre 이후의 visual token.
        L = 1 + patch token 수.
        return:
        [B, 1, D]
        """

        BT, L, D = x.shape

        if not hasattr(self, "B") or not hasattr(self, "T"):
            raise RuntimeError(
                "self.B and self.T are not set. "
                "TemporalPrompt.forward() or get_mask() must be called before get_trajectory_prompt()."
            )

        x = x.reshape(self.B, self.T, L, D)

        # 방법 1: CLS token 기반 trajectory
        # frame_feat = x[:, :, 0, :]             # [B, T, D]

        # # 방법 2는 나중에 ablation용:
        frame_feat = x[:, :, 1:, :].mean(dim=2)  # [B, T, D]

        delta_h = frame_feat[:, 1:, :] - frame_feat[:, :-1, :]  # [B, T-1, D]

        p_traj = self.TrajectoryPrompter(delta_h)               # [B, 1, D]

        _debug_tensor("TP3.get_trajectory_prompt.delta_h", delta_h, print_rows=True)
        _debug_tensor("TP3.get_trajectory_prompt.p_traj", p_traj, print_rows=True)
        self.last_traj_stats = {}

        self.last_traj_stats.update(
            _norm_stats("frame_feat", frame_feat)
        )

        self.last_traj_stats.update(
            self.TrajectoryPrompter.last_debug_stats
        )
        return p_traj
        
        