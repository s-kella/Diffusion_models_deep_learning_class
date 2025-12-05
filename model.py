import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels, cond_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

        # initialize final conv to zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ConditionalGroupNorm(nn.Module):
    def __init__(self, num_channels: int, cond_channels: int, num_groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.linear = nn.Linear(cond_channels, num_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond shape: (batch_size, cond_channels)
        params = self.linear(cond)
        scale, bias = params.chunk(2, dim=1)
        # reshape to (batch_size, num_channels, 1, 1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)

        normalized = self.norm(x)
        return scale * normalized + bias


class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.norm1 = ConditionalGroupNorm(nb_channels, cond_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = ConditionalGroupNorm(nb_channels, cond_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)

        # initialize second conv to zero
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x, cond)))
        y = self.conv2(F.silu(self.norm2(y, cond)))
        return x + y
