import torch
from torch import nn
from torch.distributions import Normal
from torch.autograd import Variable
import numpy as np
import scipy.stats as stats

class SCVAE(nn.Module):
    def __init__(self, x_dim, label_dim, h_dim, z_dim, input_dim, device=None, is_prior=False):
        super(SCVAE, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.is_prior = is_prior

        self.device = torch.device("cpu") if device is None else device

        # Feature-extracting transformations
        self.embedding = nn.Sequential(nn.Linear(input_dim, 1), nn.ReLU())
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.phi_y = nn.Sequential(nn.Linear(label_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

        # Encoder
        self.enc = nn.Sequential(nn.Linear(h_dim * 3, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.enc_mean = nn.Sequential(nn.Linear(h_dim, z_dim), nn.LeakyReLU(0.1))
        self.enc_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # Prior
        self.prior = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim), nn.LeakyReLU(0.1))
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # Decoder
        self.dec = nn.Sequential(nn.Linear(h_dim * 3, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.dec_mean = nn.Sequential(nn.Linear(h_dim, h_dim), nn.LeakyReLU(0.1),
                                      nn.Linear(h_dim, input_dim * label_dim), nn.LeakyReLU(0.1))
        self.dec_std = nn.Sequential(nn.Linear(h_dim, input_dim * label_dim), nn.Softplus())

        self.rnn = nn.GRUCell(h_dim * 2, h_dim)

    def forward(self, X, Y):
        # Iuput X shape : (seq_len, Batch_size, feature_dim, input_dim)
        X_emb = torch.squeeze(self.embedding(X), dim=-1)
        Y_emb = torch.squeeze(self.embedding(Y), dim=-1)

        self._reset_variables()
        h = torch.zeros(X.shape[1], self.h_dim).to(self.device)

        for t in range(X.shape[0]):
            x_t = X_emb[t]
            y_t = Y_emb[t]
            h = self.recurrence(x_t, y_t, h)

        return self.calc_loss(Y)

    def recurrence(self, x_t, y_t, h):
        phi_x_t = self.phi_x(x_t)
        phi_y_t = self.phi_y(y_t)

        enc_t = self.enc(torch.cat([phi_x_t, phi_y_t, h], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)

        prior_t = self.prior(torch.cat([phi_x_t, h], 1))
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
        phi_z_t = self.phi_z(z_t)

        dec_t = self.dec(torch.cat([phi_x_t, phi_z_t, h], 1))
        dec_mean_t = self.dec_mean(dec_t)
        dec_std_t = self.dec_std(dec_t)

        self.Z_mean.append(enc_mean_t)
        self.Z_std.append(enc_std_t)
        self.pZ_mean.append(prior_mean_t)
        self.pZ_std.append(prior_std_t)
        self.Xr_mean.append(dec_mean_t)
        self.Xr_std.append(dec_std_t)
        self.h_chain.append(h)

        h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), h)
        return h

    def _reset_variables(self):
        self.Z_mean, self.Z_std = [], []
        self.pZ_mean, self.pZ_std = [], []
        self.Xr_mean, self.Xr_std = [], []
        self.h_chain = []

    def calc_loss(self, X):
        X = X.view(X.shape[0], X.shape[1], -1)
        kld_loss = 0
        nll_loss = 0

        for t in range(len(self.h_chain)):
            normal_t = Normal(self.Xr_mean[t], self.Xr_std[t])
            # Avoid nan in KLD
            kld_loss += self._kld_gauss(self.Z_mean[t], self.Z_std[t] + 1e-8, self.pZ_mean[t], self.pZ_std[t] + 1e-8)
            nll_loss -= normal_t.log_prob(X[t] + 1e-8).sum()

        return kld_loss, nll_loss

    def _reparameterized_sample(self, mean, std):
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return Variable(eps).mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)
