import torch
from PIL import Image
from torchvision.utils import make_grid
import argparse

from model import Model
from train import c_in, c_out, c_skip, c_noise


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def denoise(model, x, sigma, sigma_data):
    # D_theta(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x, c_noise(sigma))
    batch_size = sigma.shape[0]
    sigma = sigma.view(-1, 1, 1, 1)

    cin = c_in(sigma, sigma_data)
    cout = c_out(sigma, sigma_data)
    cskip = c_skip(sigma, sigma_data)
    cnoise = c_noise(sigma).view(batch_size)

    network_input = cin * x
    network_output = model(network_input, cnoise)

    return cskip * x + cout * network_output


@torch.no_grad()
def sample_images(model, num_images, image_shape, sigma_data, device, steps=50):
    model.eval()
    sigma_data = sigma_data.to(device)

    sigmas = build_sigma_schedule(steps=steps).to(device)

    # initialize with pure gaussian noise
    x = torch.randn(num_images, *image_shape, device=device) * sigmas[0]

    for i, sigma in enumerate(sigmas):
        sigma_batch = sigma.repeat(num_images)
        x_denoised = denoise(model, x, sigma_batch, sigma_data)

        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - x_denoised) / sigma

        x = x + d * (sigma_next - sigma)

    return x


def save_image_grid(images, filename):
    # [-1, 1] -> [0, 1] -> [0, 255]
    images = images.clamp(-1, 1).add(1).div(2).mul(255).byte()
    grid = make_grid(images, nrow=8)
    arr = grid.cpu()

    if arr.shape[0] == 1:
        # grayscale image
        arr = arr.squeeze(0).numpy()
        img = Image.fromarray(arr, mode='L')
    else:
        # rgb image
        arr = arr.permute(1, 2, 0).numpy()
        img = Image.fromarray(arr)

    img.save(filename)
    print(f'Saved image to {filename}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--num_images', type=int, default=64, help='number of images to generate')
    parser.add_argument('--steps', type=int, default=50, help='number of sampling steps')
    parser.add_argument('--output', type=str, default='generated.png', help='output filename')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    info = checkpoint['info']

    # create model
    model = Model(
        image_channels=info.image_channels,
        nb_channels=128,
        num_blocks=4,
        cond_channels=128
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"] + 1}')

    # generate images
    print(f'Generating {args.num_images} images with {args.steps} steps...')
    images = sample_images(
        model=model,
        num_images=args.num_images,
        image_shape=(info.image_channels, info.image_size, info.image_size),
        sigma_data=checkpoint['sigma_data'],
        device=device,
        steps=args.steps
    )

    # save images
    save_image_grid(images, args.output)


if __name__ == '__main__':
    main()
