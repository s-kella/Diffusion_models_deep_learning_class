import torch
import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from model import Model
from sample import sample_images


def save_images_to_folder(images, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)

    # [-1, 1] -> [0, 1] -> [0, 255]
    images = images.clamp(-1, 1).add(1).div(2).mul(255).byte()

    for i, img in enumerate(images):
        arr = img.cpu()
        if arr.shape[0] == 1:
            arr = arr.squeeze(0).numpy()
            img_pil = Image.fromarray(arr, mode='L')
        else:
            arr = arr.permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(arr)
        img_pil.save(folder / f'{i:05d}.png')


def extract_real_images(dataloader, num_images, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)

    count = 0
    for images, _ in tqdm(dataloader, desc='Extracting real images'):
        # [-1, 1] -> [0, 1] -> [0, 255]
        images = images.clamp(-1, 1).add(1).div(2).mul(255).byte()

        for img in images:
            if count >= num_images:
                return

            arr = img.cpu()
            if arr.shape[0] == 1:
                arr = arr.squeeze(0).numpy()
                img_pil = Image.fromarray(arr, mode='L')
            else:
                arr = arr.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(arr)
            img_pil.save(folder / f'{count:05d}.png')
            count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--num_images', type=int, default=10000, help='number of images for FID')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for generation')
    parser.add_argument('--steps', type=int, default=50, help='number of sampling steps')
    parser.add_argument('--real_folder', type=str, default='fid_real', help='folder for real images')
    parser.add_argument('--fake_folder', type=str, default='fid_fake', help='folder for generated images')
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

    # use ema model if available, otherwise use regular model
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print(f'Loaded EMA model from epoch {checkpoint["epoch"] + 1}')
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from epoch {checkpoint["epoch"] + 1}')

    # extract real images if needed
    if not os.path.exists(args.real_folder) or len(list(Path(args.real_folder).glob('*.png'))) == 0:
        print(f'Extracting {args.num_images} real images...')
        from data import load_dataset_and_make_dataloaders
        dl, _ = load_dataset_and_make_dataloaders(
            dataset_name='FashionMNIST',
            root_dir='data',
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False
        )
        extract_real_images(dl.valid, args.num_images, args.real_folder)
        print(f'Saved real images to {args.real_folder}')

    # generate fake images
    print(f'Generating {args.num_images} fake images...')
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    all_images = []

    for i in tqdm(range(num_batches), desc='Generating batches'):
        batch_size = min(args.batch_size, args.num_images - i * args.batch_size)
        images = sample_images(
            model=model,
            num_images=batch_size,
            image_shape=(info.image_channels, info.image_size, info.image_size),
            sigma_data=checkpoint['sigma_data'],
            device=device,
            steps=args.steps
        )
        all_images.append(images)

    all_images = torch.cat(all_images, dim=0)
    save_images_to_folder(all_images, args.fake_folder)
    print(f'Saved generated images to {args.fake_folder}')

    # compute FID using pytorch-fid
    print('Computing FID score...')
    try:
        from pytorch_fid import fid_score
        fid_value = fid_score.calculate_fid_given_paths(
            [args.real_folder, args.fake_folder],
            batch_size=50,
            device=device,
            dims=2048
        )
        print(f'FID score: {fid_value:.2f}')
    except ImportError:
        print('pytorch-fid not found. Install with: pip install pytorch-fid')
        print(f'Or run manually: python -m pytorch_fid {args.real_folder} {args.fake_folder}')


if __name__ == '__main__':
    main()
