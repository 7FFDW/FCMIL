import torch
import torch.nn as nn
import torch.nn.functional as F

class DABlock(nn.Module):
    def __init__(self, D, use_residual=True, init_a=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.y = nn.Parameter(torch.ones(1, D))
        self.b = nn.Parameter(torch.zeros(1, D))
        self.norm = nn.LayerNorm(D)
        self.use_residual = use_residual

        self.act = torch.tanh


    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.a * x
        x = self.act(x)
        x = x * self.y + self.b
        if self.use_residual:
            x = x + residual
        return x

class FAA(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3)
        self.low_linear = nn.Linear(in_channels, in_channels)

        self.high_linear = nn.Linear(in_channels, in_channels)
        self.high_DAB = DABlock(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=1, batch_first=True)
    def upsample_to(self,x, target_len):
        # x: [L, C]
        x = x.unsqueeze(0).permute(0, 2, 1)  # [1, C, L]
        x_up = F.interpolate(x, size=target_len, mode='linear', align_corners=True)
        x_up = x_up.permute(0, 2, 1).squeeze(0)  # [target_len, C]
        return x_up

    def forward(self, x):

        N, C = x.shape
        x_fft = torch.fft.fft(x, dim=0)


        cutoff = N // 4
        f_low_raw = x_fft[:cutoff, :]
        f_high_raw = x_fft[-cutoff:, :]


        f_low_fused = self.low_linear(f_low_raw.real) + self.low_linear(f_low_raw.imag)
        f_high_fused = self.high_linear(f_high_raw.real) + self.high_linear(f_high_raw.imag)


        q = torch.sigmoid(f_low_fused)
        k = self.high_DAB(f_high_fused)
        v = x


        q = self.upsample_to(q, N)
        k = self.upsample_to(k, N)

        freq,attn = self.attn(q, k, v)


        # attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        # attn = torch.softmax(attn, dim=-1)
        return freq, attn


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class FCMIL(nn.Module):
    def __init__(self, n_classes=2):
        super(FCMIL, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1


        self.feature = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.spatial_attn_layer = nn.MultiheadAttention(embed_dim=512, num_heads=1, batch_first=True)

        self.freq_layer = FAA(in_channels=512)

        self.classifier = nn.Linear(self.L, n_classes)

    def forward(self, x):

        h = self.feature(x.squeeze(0))

        h_spatial, attn_weights = self.spatial_attn_layer(h, h, h)
        h_freq,attn = self.freq_layer(h.squeeze(0))

        z_total = h_spatial + h_freq

        z_total = torch.mean(z_total, dim=0, keepdim=True)
        logits = self.classifier(z_total)

        return logits, attn.mean(dim=0)


