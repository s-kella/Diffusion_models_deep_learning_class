import torch
import torch.nn as nn
from data import load_dataset_and_make_dataloaders
from model import Model


def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80):
    """Sample noise levels from log-normal distribution."""
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)


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


def train():
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

    # Initialize model
    model = Model(
        image_channels=info.image_channels,
        nb_channels=64,
        num_blocks=4,
        cond_channels=128
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f'\nStarting training...')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (y, labels) in enumerate(dl.train):
            # Step 1: Sample an image y from the dataset (already done by dataloader)
            y = y.to(device)
            batch_size = y.size(0)

            # Step 2: Sample sigma from p_noise
            sigma = sample_sigma(batch_size).to(device)

            # Step 3: Create a noisy image x by adding gaussian noise of std sigma to the clean image y
            epsilon = torch.randn_like(y)
            x = y + sigma.view(-1, 1, 1, 1) * epsilon

            # Step 4: Compute cin, cout, cskip and cnoise
            cin = c_in(sigma, info.sigma_data)
            cout = c_out(sigma, info.sigma_data)
            cskip = c_skip(sigma, info.sigma_data)
            cnoise = c_noise(sigma)

            # Step 5: Forward pass: network takes cin * x and cnoise as inputs
            network_input = cin.view(-1, 1, 1, 1) * x
            network_output = model(network_input, cnoise)

            # Step 6: Compute MSE loss with target = (y - cskip * x) / cout
            target = (y - cskip.view(-1, 1, 1, 1) * x) / cout.view(-1, 1, 1, 1)
            loss = criterion(network_output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {avg_loss:.4f}')

        # Print epoch summary
        avg_epoch_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}')

    print('\nTraining completed!')

    # Save the model
    torch.save(model.state_dict(), 'diffusion_model.pth')
    print('Model saved to diffusion_model.pth')


if __name__ == '__main__':
    train()
