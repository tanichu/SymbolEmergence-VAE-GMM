from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tool import visualize_ls, sample, get_param

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def gaussian_kl(en_mu, en_logvar, prior_mu, prior_cov):
    """KL(diagonal recognition Gaussian || full GMM Gaussian), per image."""
    mu, logvar = en_mu.double(), en_logvar.double()
    pm = torch.as_tensor(prior_mu, device=mu.device, dtype=mu.dtype)
    cov = torch.as_tensor(prior_cov, device=mu.device, dtype=mu.dtype)
    if pm.shape != mu.shape or cov.shape != (*mu.shape, mu.shape[-1]):
        raise ValueError("Prior arrays must match each image's latent dimensions")
    chol = torch.linalg.cholesky(cov)
    precision = torch.cholesky_inverse(chol)
    delta = mu - pm
    trace = (precision.diagonal(dim1=-2, dim2=-1) * logvar.exp()).sum(-1)
    quadratic = torch.einsum('bi,bij,bj->b', delta, precision, delta)
    logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return 0.5 * (trace + quadratic - mu.shape[-1] + logdet - logvar.sum(-1))

x_dim = 12
h_dim = 1024
image_channels=1
image_size = 28

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) # [10,1024]
class UnFlatten(nn.Module):
    def forward(self, input, size=h_dim):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=image_channels, h_dim=h_dim, x_dim=x_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, x_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.fc3 = nn.Linear(x_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=image_channels, kernel_size=4, stride=2, padding=1),            
            #nn.ReLU(),
            #nn.ConvTranspose2d(in_channels=16, out_channels=image_channels, kernel_size=2, stride=2),
            #nn.ReLU(),
            #nn.ConvTranspose2d(in_channels=8, out_channels=image_channels, kernel_size=2, stride=2),            
            #nn.ReLU(),
            #nn.ConvTranspose2d(in_channels=16, out_channels=image_channels, kernel_size=3, stride=2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(h_dim, x_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.fc3 = nn.Linear(x_dim, h_dim)
        self.softplus= nn.Softplus()
        # prior (0,I)
        self.prior_var = nn.Parameter(torch.Tensor(1, x_dim).float().fill_(1.0))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, o_d):
        h = self.encoder(o_d)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, o_d):
        z, mu, logvar = self.encode(o_d)
        return self.decode(z), mu, logvar, z
     
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_o, o_d, en_mu, en_logvar, gmm_mu, gmm_var, iteration):
        BCE = F.binary_cross_entropy(recon_o, o_d, reduction='sum')
        beta = 1.0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114 
        if iteration != 0: 
            KLD = gaussian_kl(en_mu, en_logvar, gmm_mu, gmm_var).sum()
        else:
            KLD = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())
        return BCE + KLD

def train(iteration, gmm_mu, gmm_var, epoch, train_loader, batch_size, all_loader, model_dir, agent, cache_prior=False, gpu_images=False, epoch_sync=False, model_factory=None):
    if not isinstance(train_loader.sampler, torch.utils.data.SequentialSampler) or train_loader.drop_last:
        raise ValueError("Feedback priors require sequential, complete image order")
    model = (model_factory or VAE)().to(device)
    print(f"VAE_Agent {agent} Training Start({iteration}): Epoch:{epoch}, Batch_size:{batch_size}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    resumed = None
    if iteration:
        resumed = Path(model_dir) / 'pth' / f'train{agent}_{iteration-1}.pth'
        state = torch.load(resumed, map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if gmm_mu.shape != (len(train_loader.dataset), x_dim) or gmm_var.shape != (len(train_loader.dataset), x_dim, x_dim):
            raise ValueError("Expected full word-conditioned prior for every image")
        np.savez(Path(model_dir) / 'npy' / f'used_prior{agent}_{iteration}.npz',
                 mean=gmm_mu.cpu().numpy(), covariance=gmm_var.cpu().numpy())
    prepared_prior = None
    if iteration and cache_prior:
        from runtime_cache import prepare_gaussian_prior
        prepared_prior = prepare_gaussian_prior(gmm_mu, gmm_var, device)
    loss_list = np.zeros((epoch))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    setup_start = time.perf_counter()
    batches = train_loader
    if gpu_images:
        if device.type != 'cuda':
            raise ValueError('GPU residency requested without CUDA')
        from runtime_cache import GPUResidentBatches
        batches = GPUResidentBatches(train_loader, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    resident_setup_s = time.perf_counter() - setup_start
    training_start = time.perf_counter()
    for i in range(epoch):
        model.train()
        train_loss = 0
        epoch_losses = []
        for batch_idx, (data, _) in enumerate(batches):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, x_d = model(data)
            if iteration==0: 
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu=None, gmm_var=None, iteration=iteration)
            elif prepared_prior is not None:
                from runtime_cache import cached_gaussian_kl
                sl = slice(batch_idx*batch_size, (batch_idx+1)*batch_size)
                loss = F.binary_cross_entropy(recon_batch, data, reduction='sum') + cached_gaussian_kl(
                    mu, logvar, tuple(x[sl] for x in prepared_prior)).sum()
            else:
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu[batch_idx*batch_size:(batch_idx+1)*batch_size], gmm_var[batch_idx*batch_size:(batch_idx+1)*batch_size], iteration=iteration)
            loss = loss.mean()
            if not epoch_sync and not torch.isfinite(loss):
                raise FloatingPointError("Non-finite VAE objective")
            loss.backward()
            if epoch_sync:
                epoch_losses.append(loss.detach())
            else:
                train_loss += loss.item()
            optimizer.step()
        if epoch_sync:
            from runtime_cache import finish_epoch_losses
            train_loss = finish_epoch_losses(epoch_losses)
        if i == 0 or (i+1) % 25 == 0 or i == (epoch-1):
            print('====> Epoch: {} Average loss: {:.4f}'.format(i+1, train_loss / len(train_loader.dataset)))
            
        loss_list[i] = -(train_loss / len(train_loader.dataset))
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_loop_s = time.perf_counter() - training_start
    # plot
    plt.figure()
    plt.plot(range(0,epoch), loss_list, color="blue", label="ELBO")
    if iteration!=0: 
        loss_0 = np.load(model_dir+'/npy/loss'+agent+'_0.npy')
        plt.plot(range(0,epoch), loss_0, color="red", label="ELBO_I0")
    plt.xlabel('epoch'); plt.ylabel('ELBO'); plt.legend(loc='lower right')
    plt.savefig(model_dir+'/graph'+agent+'/vae_loss_'+str(iteration)+'.png')
    plt.close()
    
    np.save(model_dir+'/npy/loss'+agent+'_'+str(iteration)+'.npy', np.array(loss_list))
    torch.save(model.state_dict(), model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth")
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
               Path(model_dir) / 'pth' / f'train{agent}_{iteration}.pth')
    (Path(model_dir) / 'log' / f'vae{agent}_{iteration}.json').write_text(json.dumps({
        'iteration': iteration, 'agent': agent, 'resumed_from': str(resumed) if resumed else None,
        'prior': 'full-gmm' if iteration else 'standard-normal',
        'epochs': epoch, 'final_negative_objective_per_image': float(loss_list[-1]),
        'training_loop_s': training_loop_s, 'resident_setup_s': resident_setup_s,
        'gpu_images': gpu_images, 'epoch_sync': epoch_sync,
    }, indent=2), encoding='utf-8')
    
    x_d, label = send_all_x(iteration=iteration, all_loader=all_loader, model_dir=model_dir, agent=agent, model_factory=model_factory)

    # Send latent variable x_d to GMM
    return x_d, label, loss_list
    

def decode(iteration, decode_k, sample_num, sample_d, manual, model_dir, agent):
    print(f"Reconstruct image on Agent: {agent}, category: {decode_k}")
    model = VAE().to("cpu")
    
    model.load_state_dict(torch.load(str(model_dir)+"/pth/vae"+agent+"_"+str(iteration)+".pth",map_location=torch.device('cpu'))); model.eval()
    mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k = get_param(iteration, model_dir=model_dir, agent=agent)
    sample_d = torch.from_numpy(sample_d.astype(np.float32)).clone()
    with torch.no_grad():
        sample_d = sample_d.to("cpu")
        sample_d = model.decode(sample_d).cpu()
        save_image(sample_d.view(sample_num, image_channels, image_size, image_size),model_dir+'/recon'+agent+'/random_'+str(decode_k)+'.png') if manual != True else save_image(sample_d.view(sample_num, image_channels, image_size, image_size),model_dir+'/recon'+agent+'/manual_'+str(decode_k)+'.png')
        
    

def plot_latent(iteration, all_loader, model_dir, agent): 
    print(f"Plot latent space on Agent: {agent}")
    model = VAE().to("cpu")
    model.load_state_dict(torch.load(model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth", map_location=torch.device('cpu')))
    model.eval()
    for batch_idx, (data, label) in enumerate(all_loader):
        data = data.to("cpu")
        recon_batch, mu, logvar, x_d = model(data)
        x_d = x_d.cpu()
        visualize_ls(iteration, x_d.detach().numpy(), label, model_dir,agent=agent)
        break

def send_all_x(iteration, all_loader, model_dir, agent, model_factory=None):
    if not isinstance(all_loader.sampler, torch.utils.data.SequentialSampler) or all_loader.drop_last:
        raise ValueError("Latent extraction requires sequential, complete image order")
    model = (model_factory or VAE)().to(device)
    model.load_state_dict(torch.load(model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth"))
    model.eval()
    latent_batches, label_batches = [], []
    with torch.no_grad():
        for data, label in all_loader:
            x_d, _, _ = model.encode(data.to(device))
            latent_batches.append(x_d.cpu().numpy())
            label_batches.append(label.cpu().numpy())
    return np.concatenate(latent_batches), np.concatenate(label_batches)
