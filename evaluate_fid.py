import os
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from data import load_dataset_and_make_dataloaders
from model import Model
from sample import build_sigma_schedule, c_in, c_out, c_skip, c_noise


def calculate_fid(real_features, fake_features):
    """
    Calculate Fréchet Inception Distance between real and fake features.

    FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real*sigma_fake))
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_fake

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    if not np.isfinite(covmean).all():
        print("FID calculation produces singular product; adding epsilon to diagonal of cov estimates")
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * tr_covmean
    return fid


def extract_features(images, inception_model, device):
    """Extract features from images using Inception v3."""
    inception_model.eval()

    with torch.no_grad():
        # Resize images to 299x299 for Inception v3
        if images.size(-1) != 299:
            images = torch.nn.functional.interpolate(
                images,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )

        # Convert grayscale to RGB if needed
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)

        # Normalize to [-1, 1] range expected by Inception
        images = (images - images.min()) / (images.max() - images.min()) * 2 - 1

        features = inception_model(images)

    return features.cpu().numpy()


def generate_samples(model, device, sigma_data, num_samples, image_channels, image_size, batch_size=64):
    """Generate samples from the diffusion model."""
    model.eval()
    all_samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        # Create sigma schedule
        sigmas = build_sigma_schedule(steps=50, rho=7).to(device)

        # Initialize with noise
        x = torch.randn(current_batch_size, image_channels, image_size, image_size, device=device) * sigmas[0]

        # Denoising loop
        for i, sigma in enumerate(sigmas):
            with torch.no_grad():
                sigma_batch = sigma.expand(current_batch_size)

                cin_val = c_in(sigma_batch, sigma_data)
                cout_val = c_out(sigma_batch, sigma_data)
                cskip_val = c_skip(sigma_batch, sigma_data)
                cnoise_val = c_noise(sigma_batch)

                network_input = cin_val.view(-1, 1, 1, 1) * x
                network_output = model(network_input, cnoise_val)

                x_denoised = cskip_val.view(-1, 1, 1, 1) * x + cout_val.view(-1, 1, 1, 1) * network_output

            sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else torch.tensor(0.0, device=device)
            d = (x - x_denoised) / sigma
            x = x + d * (sigma_next - sigma)

        # Clamp to [-1, 1]
        x = x.clamp(-1, 1)
        all_samples.append(x)

        print(f'Generated batch {batch_idx + 1}/{num_batches}')

    return torch.cat(all_samples, dim=0)


def main():
    # Check if in Colab
    try:
        import google.colab
        if os.path.exists('/content/drive/MyDrive'):
            checkpoint_dir = '/content/drive/MyDrive/diffusion_ckpts'
        else:
            checkpoint_dir = 'checkpoints'
    except:
        checkpoint_dir = 'checkpoints'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name='FashionMNIST',
        root_dir='data',
        batch_size=64,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f'Dataset info: {info}')

    # Load model
    model = Model(
        image_channels=info.image_channels,
        nb_channels=64,
        num_blocks=4,
        cond_channels=128
    ).to(device)

    model_path = os.path.join(checkpoint_dir, 'diffusion_model_final.pth')
    if not os.path.exists(model_path):
        model_path = 'diffusion_model.pth'

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Loaded model from {model_path}')

    # Load Inception v3
    print('Loading Inception v3...')
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity()  # Remove final classification layer
    inception_model = inception_model.to(device)
    inception_model.eval()

    # Extract features from real images
    print('\nExtracting features from real images...')
    real_features = []
    num_real_samples = 10000

    for batch_idx, (images, _) in enumerate(dl.valid):
        if len(real_features) * images.size(0) >= num_real_samples:
            break

        images = images.to(device)
        features = extract_features(images, inception_model, device)
        real_features.append(features)

        if (batch_idx + 1) % 10 == 0:
            print(f'Processed {len(real_features) * images.size(0)} real images')

    real_features = np.concatenate(real_features, axis=0)[:num_real_samples]
    print(f'Extracted features from {len(real_features)} real images')

    # Generate fake images
    print('\nGenerating fake images...')
    fake_images = generate_samples(
        model,
        device,
        info.sigma_data,
        num_samples=num_real_samples,
        image_channels=info.image_channels,
        image_size=info.image_size,
        batch_size=64
    )

    # Extract features from fake images
    print('\nExtracting features from generated images...')
    fake_features = []

    for i in range(0, len(fake_images), 64):
        batch = fake_images[i:i+64].to(device)
        features = extract_features(batch, inception_model, device)
        fake_features.append(features)

        if (i // 64 + 1) % 10 == 0:
            print(f'Processed {i + len(batch)} fake images')

    fake_features = np.concatenate(fake_features, axis=0)
    print(f'Extracted features from {len(fake_features)} generated images')

    # Calculate FID
    print('\nCalculating FID score...')
    fid_score = calculate_fid(real_features, fake_features)

    print(f'\n{"="*50}')
    print(f'FID Score: {fid_score:.2f}')
    print(f'{"="*50}')

    # Save result
    result_path = os.path.join(checkpoint_dir, 'fid_score.txt')
    with open(result_path, 'w') as f:
        f.write(f'FID Score: {fid_score:.2f}\n')
    print(f'\nFID score saved to {result_path}')


if __name__ == '__main__':
    main()
