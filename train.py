import os
import torch
import torch.nn as nn
from data import load_dataset_and_make_dataloaders
from model import Model


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


def train():
    # if running in Google Colab and if drive is already mounted
    try:
        import google.colab
        print('Running in Google Colab')

        if os.path.exists('/content/drive/MyDrive'):
            checkpoint_dir = '/content/drive/MyDrive/diffusion_ckpts'
        else:
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                checkpoint_dir = '/content/drive/MyDrive/diffusion_ckpts'
            except:
                checkpoint_dir = 'checkpoints'
    except:
        print('Running locally')
        checkpoint_dir = 'checkpoints'

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'Checkpoint directory: {checkpoint_dir}')

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    start_epoch = 0
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')])

    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f'Found checkpoint: {latest_checkpoint}')
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Resuming training from epoch {start_epoch}')
        print(f'Previous loss: {checkpoint["loss"]:.4f}')

    # Training loop
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (y, labels) in enumerate(dl.train):
            y = y.to(device)
            batch_size = y.size(0)

            # Sample sigma from p_noise
            sigma = sample_sigma(batch_size).to(device)

            #Create a noisy image
            epsilon = torch.randn_like(y)
            x = y + sigma.view(-1, 1, 1, 1) * epsilon

            cin = c_in(sigma, info.sigma_data)
            cout = c_out(sigma, info.sigma_data)
            cskip = c_skip(sigma, info.sigma_data)
            cnoise = c_noise(sigma)

            # Forward pass - takes cin * x and cnoise as inputs
            network_input = cin.view(-1, 1, 1, 1) * x
            network_output = model(network_input, cnoise)

            target = (y - cskip.view(-1, 1, 1, 1) * x) / cout.view(-1, 1, 1, 1)
            loss = criterion(network_output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {avg_loss:.4f}')

        avg_epoch_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}')

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

    print('\nTraining completed!')

    final_model_path = os.path.join(checkpoint_dir, 'diffusion_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')


if __name__ == '__main__':
    train()
