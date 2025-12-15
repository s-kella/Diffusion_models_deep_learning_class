import torch


def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80):
    """Sample noise levels from log-normal distribution."""
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)


def c_in(sigma, sigma_data):
    """Scale network input for unit variance."""
    return 1 / torch.sqrt(sigma_data**2 + sigma**2)


def c_out(sigma, sigma_data):
    """Scale network output for unit variance."""
    return (sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2)


def c_skip(sigma, sigma_data):
    """Control skip connection."""
    return sigma_data**2 / (sigma_data**2 + sigma**2)


def c_noise(sigma):
    """Transform noise level before giving to network."""
    return torch.log(sigma) / 4


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    """Build a schedule of decreasing noise levels."""
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
