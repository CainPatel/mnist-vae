import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim = 400, latent_dim = 20):
        super().__init__()

        # encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        #decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + std * eps

    def decode(self, x):
        h = F.relu(self.dec_fc1(x))
        return torch.sigmoid(self.dec_fc2(h)) # [0,1] for probabilities 
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar #loss needs mu and logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl
    

    