import torch


def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80):
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)


def c_in(sigma, sigma_data):
    return 1 / torch.sqrt(sigma_data**2 + sigma**2)


def c_out(sigma, sigma_data):
    return (sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2)


def c_skip(sigma, sigma_data):
    return sigma_data**2 / (sigma_data**2 + sigma**2)


def c_noise(sigma):
    return torch.log(sigma) / 4


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
