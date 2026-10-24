"""RGB64 image adaptation; inherits the repaired full-prior VAE loss and sampler."""
from torch import nn
from cnn_vae_module_mnist import VAE

class RGBVAE(VAE):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder=nn.Sequential(
            nn.Conv2d(3,32,4,2,1),nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),nn.ReLU(),
            nn.Conv2d(128,256,4,2,1),nn.ReLU(),nn.Flatten())
        self.fc1=nn.Linear(4096,12)
        self.fc2=nn.Linear(4096,12)
        self.fc3=nn.Linear(12,4096)
        self.decoder=nn.Sequential(nn.Unflatten(1,(256,4,4)),
            nn.ConvTranspose2d(256,128,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),nn.Sigmoid())
