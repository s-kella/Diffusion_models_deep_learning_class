import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from copy import deepcopy

from data import load_dataset_and_make_dataloaders
from model import Model


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.shadow.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)


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


def train_epoch(model, dataloader, optimizer, sigma_data, device, ema=None):
    model.train()
    total_loss = 0
    sigma_data = sigma_data.to(device)

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
        cnoise = c_noise(sigma).view(batch_size)

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

        # update ema
        if ema is not None:
            ema.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # setup checkpoint directory (google drive if available)
    if os.path.exists('/content/drive'):
        save_dir = '/content/drive/MyDrive/diffusion_ckpts_v2'
    else:
        save_dir = './checkpoints_v2'
    os.makedirs(save_dir, exist_ok=True)
    print(f'Saving checkpoints to: {save_dir}')

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

    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    ema = EMA(model, decay=0.9999)

    # resume from checkpoint if exists
    start_epoch = 0
    best_loss = float('inf')
    ckpt_path = os.path.join(save_dir, 'checkpoint_last.pt')

    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema.shadow.load_state_dict(ckpt['ema_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}')

    # training loop
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        loss = train_epoch(model, dl.train, optimizer, info.sigma_data, device, ema=ema)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}')

        # save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.shadow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'sigma_data': info.sigma_data,
            'info': info,
            'best_loss': best_loss,
            'loss': loss
        }

        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_last.pt'))

        # save best checkpoint
        if loss < best_loss:
            best_loss = loss
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pt'))
            print(f'Saved best checkpoint with loss: {loss:.6f}')


if __name__ == '__main__':
    main()
