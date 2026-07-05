import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAE_model import VAE, vae_loss

device = "mps"
batch_size = 128
epochs = 20
lr = 1e-3
latent_dim = 20

# [0,1] here since decoder ends in sigmoid + BCE and here pixels is reconstructed as probabilites
transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    root = './data', train=True, download=True, transform=transform
), batch_size=batch_size, shuffle=True)

model = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total = 0
    for i, (data, _) in enumerate(train_loader):
        x = data.view(data.size(0), -1).to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total += loss.item()
        if i % 100 == 0:
            print(f"epoch {epoch} batch {i}/{len(train_loader)} loss {loss.item()/x.size(0):.2f}")

    print(f"epoch {epoch} avg loss {total/len(train_loader.dataset):.2f}")
    torch.save(model.state_dict(), "vae.pth")

model.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim).to(device)
    samples = model.decode(z).view(64, 1, 28, 28)
    save_image(samples, "vae_sample.png", nrow=8)
    print("saved vae_samples.png")