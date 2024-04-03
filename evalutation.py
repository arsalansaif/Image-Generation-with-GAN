import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pickle as pkl

# Define the device for the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Global variables
z_size = 100

# Generator class
class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = torch.tanh(self.fc4(x))
        return out

# Create a Generator.
netG = Generator(z_size, 32, 784).to(device)

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = self.fc4(x)
        return out

# Create a Discriminator.
netD = Discriminator(784, 32, 1).to(device)

# Load MNIST dataset
def load_data(batch_size):
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader

# Train the GAN model
def train_GAN(train_loader, num_epochs=100, batch_size=64, lr=0.002):
    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.9, 0.999))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Lists to keep track of progress
    losses = []

    # Train the model
    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            optimizerD.zero_grad()
            
            # Real images
            d_real = netD(real_images)
            d_real_loss = criterion(d_real, torch.ones_like(d_real))
            
            # Fake images
            z = torch.randn(batch_size, z_size, device=device)
            fake_images = netG(z)
            d_fake = netD(fake_images)
            d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
            
            # Total loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizerD.step()

            # Train Generator
            optimizerG.zero_grad()
            z = torch.randn(batch_size, z_size, device=device)
            fake_images = netG(z)
            d_fake = netD(fake_images)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            optimizerG.step()

            # Print progress
            if batch_i % 400 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch {batch_i}: D_loss={d_loss.item()}, G_loss={g_loss.item()}")

        # Save losses
        losses.append((d_loss.item(), g_loss.item()))

    # Save trained models
    torch.save(netG, 'generator.pt')
    # torch.save(netD, 'discriminator.pt')
    torch.save(netG.state_dict(), 'generator_weights.pt')


    return losses

# Generate images using the trained generator
def generate_images(model, num_images):
    images = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_images):
            z = torch.randn(1, z_size, device=device)
            generated_image = model(z).cpu().numpy().reshape(28, 28)
            images.append(generated_image)
    return images

# Plot images
def plot_images(images, grid_size):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10), tight_layout=True)
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j].imshow(images[i * grid_size + j], cmap='gray')
            axes[i, j].axis("off")
    plt.show()

if __name__ == "__main__":
    # Load data
    batch_size = 64
    train_loader = load_data(batch_size)

    # Train GAN
    losses = train_GAN(train_loader)

    # Plot losses
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.show()

    # Generate images
    netG = torch.load('generator.pt').to(device)
    generated_images = generate_images(netG, 25)

    # Plot generated images
    plot_images(generated_images, 5)
