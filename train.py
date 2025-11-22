import torch


def c_in(sigma, sigma_data):
    return 1 / (sigma_data**2 + sigma**2).sqrt()


def c_out(sigma, sigma_data):
    return sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()


def c_skip(sigma, sigma_data):
    return sigma_data**2 / (sigma_data**2 + sigma**2)


def c_noise(sigma):
    return sigma.log() / 4


def sample_sigma(n, device, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80):
    return (torch.randn(n, device=device) * scale + loc).exp().clip(sigma_min, sigma_max)
