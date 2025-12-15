import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization where affine parameters are predicted from conditioning."""
    def __init__(self, nb_channels: int, cond_channels: int) -> None:
        super().__init__()
        # BatchNorm without learnable affine parameters
        self.bn = nn.BatchNorm2d(nb_channels, affine=False)
        # Linear layer to predict gamma and beta from conditioning
        self.linear = nn.Linear(cond_channels, nb_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Normalize the input
        normalized = self.bn(x)

        # Predict affine parameters from conditioning
        affine_params = self.linear(cond)  # Shape: (batch_size, nb_channels * 2)
        gamma, beta = affine_params.chunk(2, dim=1)  # Split into gamma and beta

        # Reshape for broadcasting: (batch_size, nb_channels, 1, 1)
        gamma = gamma.view(-1, gamma.size(1), 1, 1)
        beta = beta.view(-1, beta.size(1), 1, 1)

        # Apply affine transformation: gamma * normalized + beta
        return gamma * normalized + beta


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

        # Initialize conv_out to zero for better training
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


class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.norm1 = ConditionalBatchNorm2d(nb_channels, cond_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = ConditionalBatchNorm2d(nb_channels, cond_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)

        # Initialize conv2 to zero for better residual training
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x, cond)))
        y = self.conv2(F.relu(self.norm2(y, cond)))
        return x + y
