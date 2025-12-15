import os
import torch
import torch.nn as nn
from model import Model
from data import load_dataset_and_make_dataloaders
from PIL import Image
from torchvision.utils import make_grid


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    """Build a schedule of decreasing noise levels."""
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


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


def sample_images(model, device, sigma_data, num_images=8, steps=50, image_channels=1, image_size=32):
    """Sample images using the trained diffusion model."""
    model.eval()

    # Step 1: Create a sequence of decreasing sigmas using build_sigma_schedule
    sigmas = build_sigma_schedule(steps=steps, rho=7).to(device)
    print(f'Created sigma schedule with {len(sigmas)} steps')
    print(f'Sigma range: [{sigmas.min():.4f}, {sigmas.max():.4f}]')

    # Step 2: Sample initial gaussian noise of standard deviation sigmas[0]
    x = torch.randn(num_images, image_channels, image_size, image_size, device=device) * sigmas[0]
    print(f'Initial noise shape: {x.shape}')

    # Step 3: Apply iteratively the denoising network, following Euler's method
    print('Starting denoising process...')
    for i, sigma in enumerate(sigmas):
        with torch.no_grad():
            # Compute denoiser D(x, sigma) = cskip(sigma) * x + cout(sigma) * F(cin(sigma) * x, cnoise(sigma))
            sigma_batch = sigma.expand(num_images)

            # Compute scaling functions
            cin_val = c_in(sigma_batch, sigma_data)
            cout_val = c_out(sigma_batch, sigma_data)
            cskip_val = c_skip(sigma_batch, sigma_data)
            cnoise_val = c_noise(sigma_batch)

            # Forward pass through network
            network_input = cin_val.view(-1, 1, 1, 1) * x
            network_output = model(network_input, cnoise_val)

            # Compute denoised image
            x_denoised = cskip_val.view(-1, 1, 1, 1) * x + cout_val.view(-1, 1, 1, 1) * network_output

        # Euler's method update
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else torch.tensor(0.0, device=device)
        d = (x - x_denoised) / sigma
        x = x + d * (sigma_next - sigma)

        # Print progress every 10 steps
        if (i + 1) % 10 == 0 or i == len(sigmas) - 1:
            print(f'Step [{i+1}/{len(sigmas)}], sigma: {sigma:.4f}')

    print('Denoising completed!')
    return x


def main():
    # Check if running in Google Colab and if drive is already mounted
    try:
        import google.colab
        IN_COLAB = True
        print('Running in Google Colab')

        # Check if drive is already mounted
        if os.path.exists('/content/drive/MyDrive'):
            print('Google Drive already mounted')
            checkpoint_dir = '/content/drive/MyDrive/diffusion_ckpts'
        else:
            # Try to mount drive, but handle if running from script
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                checkpoint_dir = '/content/drive/MyDrive/diffusion_ckpts'
            except:
                print('Warning: Could not mount Google Drive. Using local directory.')
                print('Please mount drive manually in a notebook cell before running script.')
                checkpoint_dir = 'checkpoints'
    except:
        IN_COLAB = False
        print('Running locally')
        checkpoint_dir = 'checkpoints'

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset info to get sigma_data and image dimensions
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name='FashionMNIST',
        root_dir='data',
        batch_size=32,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f'Dataset info: {info}')
    print(f'sigma_data: {info.sigma_data}')

    # Initialize model
    model = Model(
        image_channels=info.image_channels,
        nb_channels=64,
        num_blocks=4,
        cond_channels=128
    ).to(device)

    # Load trained weights
    model_path = os.path.join(checkpoint_dir, 'diffusion_model_final.pth')
    if not os.path.exists(model_path):
        # Try loading from local path as fallback
        model_path = 'diffusion_model.pth'

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Loaded model weights from {model_path}')

    # Sample images
    images = sample_images(
        model,
        device,
        info.sigma_data,
        num_images=8,
        steps=50,
        image_channels=info.image_channels,
        image_size=info.image_size
    )

    if images is not None:
        # Convert to image format and save
        images = images.clamp(-1, 1).add(1).div(2).mul(255).byte()
        grid = make_grid(images, nrow=4)
        img = Image.fromarray(grid.permute(1, 2, 0).cpu().numpy())
        img.save('generated_images.png')
        print('Generated images saved to generated_images.png')


if __name__ == '__main__':
    main()
