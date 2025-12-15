import os
import torch
import torch.nn as nn
from model import Model
from data import load_dataset_and_make_dataloaders
from PIL import Image
from torchvision.utils import make_grid
from utils import build_sigma_schedule, c_in, c_out, c_skip, c_noise


def sample_images(model, device, sigma_data, num_images=8, steps=50, image_channels=1, image_size=32):
    model.eval()

    # Create a sequence of decreasing sigmas using build_sigma_schedule
    sigmas = build_sigma_schedule(steps=steps, rho=7).to(device)
    print(f'Created sigma schedule with {len(sigmas)} steps')
    print(f'Sigma range: [{sigmas.min():.4f}, {sigmas.max():.4f}]')

    # Sample initial gaussian noise of standard deviation sigmas[0]
    x = torch.randn(num_images, image_channels, image_size, image_size, device=device) * sigmas[0]

    #Apply the denoising network, following Euler's method
    print('Starting denoising process...')
    for i, sigma in enumerate(sigmas):
        with torch.no_grad():
            #D(x, sigma) = cskip(sigma) * x + cout(sigma) * F(cin(sigma) * x, cnoise(sigma))
            sigma_batch = sigma.expand(num_images)

            cin_val = c_in(sigma_batch, sigma_data)
            cout_val = c_out(sigma_batch, sigma_data)
            cskip_val = c_skip(sigma_batch, sigma_data)
            cnoise_val = c_noise(sigma_batch)

            # Forward pass
            network_input = cin_val.view(-1, 1, 1, 1) * x
            network_output = model(network_input, cnoise_val)

            x_denoised = cskip_val.view(-1, 1, 1, 1) * x + cout_val.view(-1, 1, 1, 1) * network_output

        # Euler's method update
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else torch.tensor(0.0, device=device)
        d = (x - x_denoised) / sigma
        x = x + d * (sigma_next - sigma)

        if (i + 1) % 10 == 0 or i == len(sigmas) - 1:
            print(f'Step [{i+1}/{len(sigmas)}], sigma: {sigma:.4f}')

    return x


def main():
    try:
        import google.colab

        if os.path.exists('/content/drive/MyDrive'):
            print('Google Drive already mounted')
            checkpoint_dir = '/content/drive/MyDrive/diffusion_ckpts'
        else:
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                checkpoint_dir = '/content/drive/MyDrive/diffusion_ckpts'
            except:
                checkpoint_dir = 'checkpoints'
    except:
        checkpoint_dir = 'checkpoints'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dl, info = load_dataset_and_make_dataloaders(
        dataset_name='FashionMNIST',
        root_dir='data',
        batch_size=32,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f'Dataset info: {info}')
    print(f'sigma_data: {info.sigma_data}')

    model = Model(
        image_channels=info.image_channels,
        nb_channels=64,
        num_blocks=4,
        cond_channels=128
    ).to(device)

    # Load trained weights
    model_path = os.path.join(checkpoint_dir, 'diffusion_model_final.pth')
    if not os.path.exists(model_path):
        model_path = 'diffusion_model.pth'

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Loaded model weights from {model_path}')

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
        images = images.clamp(-1, 1).add(1).div(2).mul(255).byte()
        grid = make_grid(images, nrow=4)
        img = Image.fromarray(grid.permute(1, 2, 0).cpu().numpy())

        save_path = os.path.join(checkpoint_dir, 'generated_images.png')
        img.save(save_path)
        print(f'Generated images saved to {save_path}')


if __name__ == '__main__':
    main()
