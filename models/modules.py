import torch
import torch.nn as nn
import torch.nn.functional as F
class DABlock(nn.Module):
    def __init__(self, D, use_residual=True, init_a=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.y = nn.Parameter(torch.ones(1, 1, D))
        self.b = nn.Parameter(torch.zeros(1, 1, D))
        self.norm = nn.LayerNorm(D)
        self.use_residual = use_residual

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.a * x
        x = torch.tanh(x)
        x = x * self.y + self.b
        return x + residual if self.use_residual else x


class FAA(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()

        self.low_linear_real = nn.Linear(in_channels, in_channels)
        self.low_linear_imag = nn.Linear(in_channels, in_channels)
        self.high_linear_real = nn.Linear(in_channels, in_channels)
        self.high_linear_imag = nn.Linear(in_channels, in_channels)

        self.high_DAB = DABlock(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)

    def upsample_to(self, x, target_len):

        x_trans = x.transpose(1, 2)
        x_up = F.interpolate(x_trans, size=target_len, mode='linear', align_corners=True)
        return x_up.transpose(1, 2)

    def forward(self, x):

        B, N, C = x.shape


        x_fft = torch.fft.fft(x, dim=1)

        cutoff = N // 4
        f_low_raw = x_fft[:, :cutoff, :]
        f_high_raw = x_fft[:, -cutoff:, :]


        f_low_fused = self.low_linear_real(f_low_raw.real) + self.low_linear_imag(f_low_raw.imag)
        f_high_fused = self.high_linear_real(f_high_raw.real) + self.high_linear_imag(f_high_raw.imag)


        q = torch.sigmoid(f_low_fused)
        k = self.high_DAB(f_high_fused)
        v = x

        q = self.upsample_to(q, N)
        k = self.upsample_to(k, N)


        freq, attn = self.attn(q, k, v)

        return freq, attn