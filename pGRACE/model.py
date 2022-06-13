from random import random
from typing import Optional
import matplotlib
from torch._C import device
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
import scipy.stats as stats
from torch.distributions.beta import Beta

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            # self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            self.conv = [base_model(in_channels, 2 * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]

class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, epoch):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = self.sim(z1, z1)
        between_sim = self.sim(z1, z2)
        refl_sim = f(refl_sim)
        between_sim = f(between_sim)
        return -torch.log(between_sim.diag() / (between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))

    def semi_loss_bmm(self, z1: torch.Tensor, z2: torch.Tensor, epoch, args, bmm_model, fit = False):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = self.sim(z1, z1)
        between_sim = self.sim(z1, z2)
        N = between_sim.size(0)
        mask = torch.ones((N,N),dtype=bool).to(z1.device)
        mask[np.eye(N,dtype=bool)] = False 
        if epoch == args.epoch_start and fit:
            global B
            N_sel = 100
            index_fit = np.random.randint(0, N, N_sel)          
            sim_fit = between_sim[:,index_fit]
            sim_fit = (sim_fit + 1) / 2   # Min-Max Normalization
            bmm_model.fit(sim_fit.flatten())
            between_sim_norm = between_sim.masked_select(mask).view(N, -1)
            between_sim_norm = (between_sim_norm + 1) / 2
            print('Computing positive probility,wait...')
            B = bmm_model.posterior(between_sim_norm,0) * between_sim_norm.detach() 
            print('Over!') 
        if args.mode == 'weight':
            refl_sim = f(refl_sim)
            between_sim = f(between_sim)
            ng_bet = (between_sim.masked_select(mask).view(N,-1) * B).sum(1) / B.mean(1)
            ng_refl = (refl_sim.masked_select(mask).view(N,-1) * B).sum(1) / B.mean(1)
            return -torch.log(between_sim.diag()/(between_sim.diag() + ng_bet + ng_refl)) 
        elif args.mode == 'mix':  
            eps = 1e-12
            sorted, indices = torch.sort(B, descending=True)
            N_sel = torch.gather(between_sim[mask].view(N,-1), -1, indices)[:,:args.sel_num]
            random_index = np.random.permutation(np.arange(args.sel_num))
            N_random = N_sel[:,random_index]
            M = sorted[:,:args.sel_num]
            M_random = M[:,random_index]
            M = (N_sel * M + N_random * M_random) / (M + M_random + eps)
            refl_sim = f(refl_sim)
            between_sim = f(between_sim)
            M = f(M)
            return -torch.log(between_sim.diag()/(M.sum(1) + between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))  
        else:
            print('Mode Error!')

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int, epoch):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = np.arange(0, num_nodes)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = self.sim(z1[mask], z1)  # [B, N]
            between_sim = self.sim(z1[mask], z2)  # [B, N]
            refl_sim = f(refl_sim)
            between_sim = f(refl_sim)
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def batched_semi_loss_bmm(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int, epoch, args, bmm_model, fit):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        global B
        B = []
        for i in range(num_batches):
            index = indices[i * batch_size:(i + 1) * batch_size]
            neg_mask = torch.ones((batch_size, num_nodes),dtype=bool).to(device)
            pos_index = np.transpose(np.column_stack((np.arange(0,batch_size,1),np.arange(i*batch_size, (i + 1) * batch_size,1))))
            neg_mask[pos_index] = False
            refl_sim = self.sim(z1[index], z1)
            between_sim = self.sim(z1[index], z2)
            if epoch == args.epoch_start and fit:
                N_sel = 100
                index_fit = np.random.randint(0, num_nodes, N_sel)          
                sim_fit = between_sim[:,index_fit]
                sim_fit = (sim_fit - sim_fit.min()) / (sim_fit.max() - sim_fit.min())
                bmm_model.fit(sim_fit.flatten())
                between_sim_norm = between_sim.masked_select(neg_mask).view(batch_size,-1)
                between_sim_norm = (between_sim_norm - between_sim_norm.min()) / (between_sim_norm.max() - between_sim_norm.min())
                print('Computing positive probility,wait...')
                B.append(bmm_model.posterior(between_sim_norm,0) * between_sim_norm.detach())
                print('Over!')
            if args.mode == 'weight':
                refl_sim = f(refl_sim)
                between_sim = f(between_sim)
                ng_bet = (between_sim.masked_select(neg_mask).view(neg_mask.size(0),-1) * B[i]).sum(1) / B[i].mean(1)
                ng_refl = (refl_sim.masked_select(neg_mask).view(neg_mask.size(0),-1) * B[i]).sum(1) / B[i].mean(1)
                losses.append(-torch.log(between_sim.diag()/(between_sim.diag() + ng_bet + ng_refl)))
                return torch.cat(losses)
            elif args.mode == 'mix':
                eps = 1e-12
                B_sel, indices = torch.sort(B[i],descending=True)
                N_sel = torch.gather(between_sim, -1, indices)
                random_index = np.random.permutation(np.arange(N_sel.size(1)))
                N_sel_random = N_sel[:,random_index]
                B_sel_random = B_sel[:,random_index]
                M = (B_sel * N_sel + B_sel_random * N_sel_random) / (B_sel + B_sel_random + eps)
                refl_sim = f(refl_sim)
                between_sim = f(between_sim)
                M = f(M)
                losses.append(-torch.log(between_sim.diag()/(M.sum(1) + between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag())))  
                return torch.cat(losses)          
            else:
                print('Mode Error!')      
        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, epoch, args, bmm_model,mean: bool = True, batch_size: Optional[int] = None):
        
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        if epoch < args.epoch_start:
            if batch_size is None:
                l1 = self.semi_loss(h1, h2, epoch)
                l2 = self.semi_loss(h2, h1, epoch)
            else:
                l1 = self.batched_semi_loss(h1, h2, batch_size, epoch)
                l2 = self.batched_semi_loss(h2, h1, batch_size, epoch)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        else:
            if batch_size is None:
                l1 = self.semi_loss_bmm(h1, h2, epoch, args, bmm_model, fit = True)
                l2 = self.semi_loss_bmm(h2, h1, epoch, args, bmm_model)
            else:
                l1 = self.batched_semi_loss_bmm(h1, h2, batch_size, epoch, args, bmm_model, fit = True)
                l2 = self.batched_semi_loss_bmm(h2, h1, batch_size, epoch, args, bmm_model)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        return ret

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def weighted_mean(x, w):
    return torch.sum(w * x) / torch.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters,
                 alphas_init,
                 betas_init,
                 weights_init):
        self.alphas = alphas_init
        self.betas = betas_init
        self.weight = weights_init
        self.max_iters = max_iters
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        x_cpu = x.cpu().detach().numpy()
        alpha_cpu = self.alphas.cpu().detach().numpy()
        beta_cpu = self.betas.cpu().detach().numpy()
        return torch.from_numpy(stats.beta.pdf(x_cpu, alpha_cpu[y], beta_cpu[y])).to(x.device)

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return self.weighted_likelihood(x, 0) + self.weighted_likelihood(x, 1)

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = torch.cat((self.weighted_likelihood(x, 0).view(1,-1),self.weighted_likelihood(x, 1).view(1,-1)),0)
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(0)
        return r

    def fit(self, x):
        eps = 1e-12
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            if self.betas[1] < 1:
                self.betas[1] = 1.01
            self.weight = r.sum(1)
            self.weight /= self.weight.sum()
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
