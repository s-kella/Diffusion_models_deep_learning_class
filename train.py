import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import load_dataset_and_make_dataloaders
from model import Model


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


def train_epoch(model, dataloader, optimizer, sigma_data, device):
    model.train()
    total_loss = 0

    for images, _ in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        batch_size = images.shape[0]

        # sample noise level
        sigma = sample_sigma(batch_size, device).view(-1, 1, 1, 1)

        # add noise to images
        noise = torch.randn_like(images)
        noisy_images = images + sigma * noise

        # compute scaling coefficients
        cin = c_in(sigma, sigma_data)
        cout = c_out(sigma, sigma_data)
        cskip = c_skip(sigma, sigma_data)
        cnoise = c_noise(sigma).squeeze()

        # forward pass
        network_input = cin * noisy_images
        network_output = model(network_input, cnoise)

        # compute target
        target = (images - cskip * noisy_images) / cout

        # compute loss
        loss = nn.functional.mse_loss(network_output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load data
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name='FashionMNIST',
        root_dir='data',
        batch_size=128,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f'Dataset info: {info}')

    # create model
    model = Model(
        image_channels=info.image_channels,
        nb_channels=128,
        num_blocks=4,
        cond_channels=128
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    # training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = train_epoch(model, dl.train, optimizer, info.sigma_data, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}')

        # save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sigma_data': info.sigma_data,
                'info': info
            }, f'checkpoint_epoch_{epoch+1}.pt')
            print(f'Saved checkpoint at epoch {epoch+1}')


if __name__ == '__main__':
    main()
