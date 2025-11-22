import torch


def c_in(sigma, sigma_data):
    return 1 / (sigma_data**2 + sigma**2).sqrt()


def c_out(sigma, sigma_data):
    return sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()


def c_skip(sigma, sigma_data):
    return sigma_data**2 / (sigma_data**2 + sigma**2)


def c_noise(sigma):
    return sigma.log() / 4
