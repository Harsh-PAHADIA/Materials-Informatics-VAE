import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

# 1. IHI Project VAE Model
class MicroVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(MicroVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(self.decoder_input(z).view(-1, 64, 16, 16)), mu, logvar

# 2. Training execution
def run_training():
    device = torch.device("cpu")
    model = MicroVAE(latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    transform = transforms.Compose([transforms.ToTensor()])
    
    class PatchDataset(Dataset):
        def __init__(self, path):
            # 5000 samples for fast training tonight
            self.files = [os.path.join(path, f) for f in os.listdir(path)[:5000]]
        def __len__(self): return len(self.files)
        def __getitem__(self, i): return transform(Image.open(self.files[i]).convert('L'))

    dataset = PatchDataset('data/processed_data')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("🚀 Training start ho rahi hai... (Wait for 5 epochs)")
    for epoch in range(5):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = nn.functional.mse_loss(recon, batch, reduction='sum') + \
                   -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} complete! Loss: {total_loss/len(dataset):.2f}")

    torch.save(model.state_dict(), 'model.pth')
    print("✅ Model saved as 'model.pth'")

if __name__ == "__main__":
    run_training()
