import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from distribution import FactorizedGaussian
import utils


class BayesianModule(nn.Module):
    def __init__(self):
        super().__init__()

    def kl_div(self):
        prior = FactorizedGaussian(self.prior_mu, self.prior_logvar)
        posterior = FactorizedGaussian(self.weight_mu, self.weight_logvar)
        return posterior.kl_div(prior)


class VariationalDropoutModule(BayesianModule):
    def __init__(self):
        super().__init__()

    def kl_div(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        alpha = self.log_alpha.exp()
        kl_div = -torch.sum(0.5*self.log_alpha + c1*alpha + c2*alpha**2 + c3*alpha**3)
        return kl_div


class Rank1Module(BayesianModule):
    def __init__(self):
        super().__init__()

    def kl_div(self):
        r_prior = FactorizedGaussian(self.r_prior_mu, self.r_prior_logvar)
        r_posterior = FactorizedGaussian(self.r_mu, self.r_logvar)
        s_prior = FactorizedGaussian(self.s_prior_mu, self.s_prior_logvar)
        s_posterior = FactorizedGaussian(self.s_mu, self.s_logvar)
        return r_posterior.kl_div(r_prior) + s_posterior.kl_div(s_prior)


class DensityModule(BayesianModule):
    def __init__(self):
        super().__init__()


class BayesianLinear(BayesianModule):
    def __init__(self, in_features, out_features, bias=True, prior_std=0.1, posterior_std_init=1e-3):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features))
        
        std = math.sqrt(2) / math.sqrt(in_features)
        self.weight_mu.data.normal_(0, std)
        self.weight_logvar.data.normal_(2.0 * np.log(posterior_std_init), std=0.01)

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('prior_mu', torch.zeros(out_features, in_features))
        self.register_buffer('prior_logvar', prior_logvar * torch.ones(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        posterior = FactorizedGaussian(self.weight_mu, self.weight_logvar)
        weight = posterior.sample()
        return F.linear(x, weight, self.bias)
    

class Rank1Linear(Rank1Module):
    def __init__(self, in_features, out_features, bias=True, prior_std=0.1, posterior_std_init=1e-3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight)
        self.r_mu = nn.Parameter(torch.ones(1, in_features))
        self.r_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, in_features))

        self.s_mu = nn.Parameter(torch.ones(1, out_features))
        self.s_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_features))

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('r_prior_mu', torch.ones(1, in_features))
        self.register_buffer('r_prior_logvar', prior_logvar * torch.ones(1, in_features))
        self.register_buffer('s_prior_mu', torch.ones(1, out_features))
        self.register_buffer('s_prior_logvar', prior_logvar * torch.ones(1, out_features))

    def forward(self, x):
        r = self.r_mu + torch.exp(0.5*self.r_logvar) * torch.randn_like(x)
        out = self.linear(r * x)
        s = self.s_mu + torch.exp(0.5*self.s_logvar) * torch.randn_like(out)
        return s * out
    

class VariationalDropoutLinear(VariationalDropoutModule):
    def __init__(self, in_features, out_features, bias=True, alpha_init=0.1, train_alpha=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight)

        self.log_alpha = nn.Parameter(np.log(alpha_init) * torch.ones(1, out_features), requires_grad=train_alpha)        

    def forward(self, x):
        alpha = torch.clip(self.log_alpha.exp(), 0, 1)
        out = self.linear(x)
        eps = 1 + alpha.sqrt() * torch.randn_like(out)
        return eps * out


class MCDropoutLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight)
        self.deterministic = False

    def forward(self, x):
        out = self.linear(F.dropout(x, p=self.dropout, training=not self.deterministic))
        return out


class BayesianConv2d(BayesianModule):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True,
                 prior_std=0.1, posterior_std_init=1e-3):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_logvar = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        
        std = math.sqrt(2) / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight_mu.data.normal_(0, std)
        self.weight_logvar.data.normal_(2.0 * np.log(posterior_std_init), std=0.01)

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('prior_mu', torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer('prior_logvar', prior_logvar * torch.ones(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        posterior = FactorizedGaussian(self.weight_mu, self.weight_logvar)
        weight = posterior.sample()
        return F.conv2d(x, weight, self.bias, padding=self.padding, stride=self.stride)


class VariationalDropoutConv2d(VariationalDropoutModule):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True, alpha_init=0.1, train_alpha=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)

        self.log_alpha = nn.Parameter(np.log(2.0 * alpha_init) * torch.ones(1, out_channels, 1, 1), requires_grad=train_alpha)        

    def forward(self, x):
        alpha = torch.clip(self.log_alpha.exp(), 1e-16, 1)
        out = self.conv(x)
        eps = 1 + alpha.sqrt() * torch.randn_like(out)
        return eps * out


class MCDropoutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        self.deterministic = False

    def forward(self, x):
        out = self.conv(F.dropout(x, p=self.dropout, training=not self.deterministic))
        return out


class Rank1Conv2d(Rank1Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True,
                 prior_std=0.1, posterior_std_init=1e-3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)

        self.r_mu = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.r_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, in_channels, 1, 1))

        self.s_mu = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.s_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_channels, 1, 1))

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('r_prior_mu', torch.ones(1, in_channels, 1, 1))
        self.register_buffer('r_prior_logvar', prior_logvar * torch.ones(1, in_channels, 1, 1))
        self.register_buffer('s_prior_mu', torch.ones(1, out_channels, 1, 1))
        self.register_buffer('s_prior_logvar', prior_logvar * torch.ones(1, out_channels, 1, 1))        

    def forward(self, x):
        r = self.r_mu + torch.exp(0.5*self.r_logvar) * torch.randn_like(x)
        out = self.conv(r * x)
        s = self.s_mu + torch.exp(0.5*self.s_logvar) * torch.randn_like(out)
        return s * out


class DensityLinear(DensityModule):
    def __init__(self, in_features, out_features, bias=True, prior_std=0.1, posterior_std_init=1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight)
        
        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('s_prior_mu', torch.zeros(1, out_features))
        self.register_buffer('s_prior_logvar', prior_logvar * torch.ones(1, out_features))
        self.register_buffer('b_prior_mu', torch.zeros(1, out_features))
        self.register_buffer('b_prior_logvar', prior_logvar * torch.ones(1, out_features))

        self.s_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_features))
        self.b_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_features))

        # Initialize activation covariance estimates
        self.L = nn.Parameter(torch.zeros(in_features, in_features))
        self.register_buffer('I', torch.eye(in_features))
        # Diagonal log variance
        self.logvar = nn.Parameter(torch.zeros(1, in_features))       

    def kl_div(self):
        s_prior = FactorizedGaussian(self.s_prior_mu, self.s_prior_logvar)
        s_posterior = FactorizedGaussian(self.s_prior_mu, self.s_logvar)
        b_prior = FactorizedGaussian(self.b_prior_mu, self.b_prior_logvar)
        b_posterior = FactorizedGaussian(self.b_prior_mu, self.b_logvar)
        return s_posterior.kl_div(s_prior) + b_posterior.kl_div(b_prior)    

    def forward(self, x):
        B, D = x.shape
       
        L = self.L.tril(diagonal=-1) + self.I
        z = x.detach() @ L
        Ex = torch.sum(z**2/self.logvar.exp(), 1, keepdim=True) / 2
        self.loglikelihood = -0.5 * (D * np.log(2*np.pi) + self.logvar.sum()) - Ex.mean(dim=1)

        # Energy can fluctuate wildly during training so apply clipping
        Ex = Ex.clip(0, D)
        # Energy will be D/2 on average. Scale the noise bias term to match their scales
        noise_var = self.s_logvar.exp() * Ex.detach() + self.b_logvar.exp() * D / 2
        noise_std = torch.sqrt(noise_var + 1e-16)

        a = self.linear(x)
        a = a + noise_std * torch.rand_like(a)
        return a


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=False, padding_mode='replicate')
        self.conv.weight.data.zero_()
        mask = utils.weight_mask(in_channels, kernel_size)
        self.register_buffer('mask', mask)
        
    def forward(self, x, detach=False):
        self.conv.weight.data *= self.mask
        if detach:
            return F.conv2d(x, self.conv.weight.detach(), padding=self.padding, stride=self.stride, padding_mode='replicate')
        else:
            return self.conv(x)


class DensityConv2d(DensityModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding=0, bias=True, prior_std=0.1, posterior_std_init=1e-3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('s_prior_mu', torch.zeros(1, out_channels, 1, 1))
        self.register_buffer('s_prior_logvar', prior_logvar * torch.ones(1, out_channels, 1, 1))
        self.register_buffer('b_prior_mu', torch.zeros(1, out_channels, 1, 1))
        self.register_buffer('b_prior_logvar', torch.ones(1, out_channels, 1, 1))

        self.s_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_channels, 1, 1))
        self.b_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_channels, 1, 1))
        
        # Generative 
        self.masked_conv = MaskedConv2d(in_channels, kernel_size, padding=padding)
        self.logvar = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def kl_div(self):
        s_prior = FactorizedGaussian(self.s_prior_mu, self.s_prior_logvar)
        s_posterior = FactorizedGaussian(self.s_prior_mu, self.s_logvar)
        b_prior = FactorizedGaussian(self.b_prior_mu, self.b_prior_logvar)
        b_posterior = FactorizedGaussian(self.b_prior_mu, self.b_logvar)
        return s_posterior.kl_div(s_prior) + b_posterior.kl_div(b_prior)    
        
    def forward(self, x):
        # x: [B, D, H, W]
        B, D, H, W = x.shape
        
        z = x.detach() + self.masked_conv(x.detach())
        Ex = torch.sum(z**2/self.logvar.exp(), dim=1, keepdim=True) / 2
        self.loglikelihood = -0.5 * (D * np.log(2*np.pi) + self.logvar.sum()) - Ex.mean(dim=[1, 2, 3])
        # Average pool the energy in the local convolutional window
        Ex_pool = self.pool(Ex)

        a = self.conv(x)
       
        Ex_pool = Ex_pool.clip(0, D)
        noise_var = self.s_logvar.exp() * Ex_pool.detach() + self.b_logvar.exp() * D / 2
        noise_std = torch.sqrt(noise_var + 1e-16)
            
        a = a + noise_std * torch.rand_like(a)
        return a


class Rank1Gaussian(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(1, dim))
        self.log_diagonals = nn.Parameter(torch.zeros(1, dim))

    def loglikelihood(self, x):
        # x: [batch_size, dim]
        B, D = x.shape
        x = x.detach()
        s = torch.exp(0.5 * self.log_diagonals)
        energy = 0.5 * ((x @ self.v.T)**2 + torch.sum((s * x)**2, 1, keepdim=True))  # [batch_size, 1]
        loglikelihood = -0.5 * (D * np.log(2*np.pi) 
                                - self.log_diagonals.sum()  
                                - torch.log(1 + torch.sum(self.v**2 / (s**2), 1))) - energy.mean(dim=1)   # [batch_size]
        return loglikelihood

    def energy(self, x):
        x = x.detach()
        s = torch.exp(0.5 * self.log_diagonals)
        energy = 0.5 * ((x @ self.v.T)**2 + torch.sum((s * x)**2, 1, keepdim=True))  # [batch_size, 1]
        return energy
        

class Rank1GaussianMixture(nn.Module):
    def __init__(self, K, dim):
        super().__init__()
        self.mixture_logits = nn.Parameter(torch.zeros(K))
        self.v = nn.Parameter(torch.zeros(K, dim))
        self.log_diagonals = nn.Parameter(torch.zeros(K, dim))

    def forward(self, x):
        B, D = x.shape
        x = x.detach()
        d_inv = torch.exp(-self.log_diagonals)   # [K, dim]
        u = self.v * d_inv
        vT_D_inv_v = torch.sum((self.v**2) * d_inv, dim=1) # [K]
        energy = 0.5 * ((x**2) @ d_inv.T - ((x @ u.T)**2)/(1 + vT_D_inv_v))
        logbias = -0.5 * (D * np.log(2*np.pi) 
                          + self.log_diagonals.sum(1)
                          + torch.log(1 + vT_D_inv_v))
        loglikelihood = logbias.unsqueeze(0) - energy   # [batch_size, K]
        loglikelihood += F.log_softmax(self.mixture_logits, dim=0).unsqueeze(0) # [batch_size, K]
        loglikelihood = torch.logsumexp(loglikelihood, dim=1)

        logbias = torch.logsumexp(logbias + F.log_softmax(self.mixture_logits, dim=0), dim=0)
        normalized_energy =  -loglikelihood + logbias
        return loglikelihood, normalized_energy


class ConvolutionalRank1GaussianMixuture(nn.Module):
    def __init__(self, K, in_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.mixture_logits = nn.Parameter(torch.zeros(K))
        self.v = nn.Parameter(torch.zeros(K, in_channels, kernel_size, kernel_size))
        nn.init.orthogonal_(self.v.data)
        self.log_diagonals = nn.Parameter(torch.zeros(K, in_channels, kernel_size, kernel_size))
        self.D = in_channels * kernel_size * kernel_size

    def forward(self, x):
        x = x.detach()
        d_inv = torch.exp(-self.log_diagonals)
        u = self.v * d_inv
        vT_D_inv_v = torch.sum(self.v * u, dim=[1, 2, 3]) # [K]
        energy = 0.5 * (F.conv2d((x**2), d_inv, stride=self.stride, padding=self.padding) 
                        - (F.conv2d(x, u, stride=self.stride, padding=self.padding)**2)/(1 + vT_D_inv_v).unsqueeze(-1).unsqueeze(-1))   # [batch_size, K, height, width]
        logbias = -0.5 * (self.D * np.log(2*np.pi) 
                         - self.log_diagonals.sum(dim=[1,2,3])
                         - torch.log(1 + vT_D_inv_v)) # [K]
        loglikelihood = logbias.unsqueeze(-1).unsqueeze(-1) - energy   # [batch_size, K, height, width]
        loglikelihood += F.log_softmax(self.mixture_logits, dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        loglikelihood = torch.logsumexp(loglikelihood, dim=1)   #[batch_size, height, width]

        logbias = torch.logsumexp(logbias + F.log_softmax(self.mixture_logits, dim=0), dim=0)
        normalized_energy = -loglikelihood + logbias
        
        return loglikelihood, normalized_energy
    

class Rank1DensityLinear(DensityModule):
    def __init__(self, in_features, out_features, bias=True, prior_std=0.1, posterior_std_init=1e-3, n_mixture_p=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight)
        
        self.s_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_features))
        self.b_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_features))
        
        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('s_prior_mu', torch.zeros(1, out_features))
        self.register_buffer('s_prior_logvar', prior_logvar * torch.ones(1, out_features))

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('b_prior_mu', torch.zeros(1, out_features))
        self.register_buffer('b_prior_logvar', prior_logvar * torch.ones(1, out_features))

        n_mixture = max(1, int(n_mixture_p * in_features))

        self.energy_model = Rank1GaussianMixture(n_mixture, in_features)

    def kl_div(self):
        s_prior = FactorizedGaussian(self.s_prior_mu, self.s_prior_logvar)
        s_posterior = FactorizedGaussian(self.s_prior_mu, self.s_logvar)
        b_prior = FactorizedGaussian(self.b_prior_mu, self.b_prior_logvar)
        b_posterior = FactorizedGaussian(self.b_prior_mu, self.b_logvar)
        return s_posterior.kl_div(s_prior) + b_posterior.kl_div(b_prior)   

    def forward(self, x):
        B, D = x.shape

        loglikelihood, energy = self.energy_model(x)
        energy = energy.unsqueeze(1)
        self.loglikelihood = loglikelihood

        noise_var = self.s_logvar.exp() * energy.detach() + self.b_logvar.exp() * D / 2
        noise_std = torch.sqrt(noise_var + 1e-16)

        # Local reparametrization
        a = self.linear(x)
        a = a + noise_std * torch.rand_like(a)
        return a
        

class Rank1DensityConv2d(DensityModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding=0, bias=True, prior_std=0.1, posterior_std_init=1e-3, n_mixture_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)

        self.s_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_channels, 1, 1))
        self.b_logvar = nn.Parameter(2.0 * np.log(posterior_std_init) * torch.ones(1, out_channels, 1, 1))
        
        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('s_prior_mu', torch.zeros(1, out_channels, 1, 1))
        self.register_buffer('s_prior_logvar', prior_logvar * torch.ones(1, out_channels, 1, 1))

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer('b_prior_mu', torch.zeros(1, out_channels, 1, 1))
        self.register_buffer('b_prior_logvar', torch.ones(1, out_channels, 1, 1))

        if in_channels == 3:
            # For the first conv layer just use all in_channels 
            n_mixture = in_channels
        else:
            n_mixture = max(int(n_mixture_p * in_channels), 1)

        self.energy_model = ConvolutionalRank1GaussianMixuture(n_mixture, in_channels, kernel_size, stride=stride, padding=padding)

    def kl_div(self):
        s_prior = FactorizedGaussian(self.s_prior_mu, self.s_prior_logvar)
        s_posterior = FactorizedGaussian(self.s_prior_mu, self.s_logvar)
        b_prior = FactorizedGaussian(self.b_prior_mu, self.b_prior_logvar)
        b_posterior = FactorizedGaussian(self.b_prior_mu, self.b_logvar)
        return s_posterior.kl_div(s_prior) + b_posterior.kl_div(b_prior) 
        
    def forward(self, x):
        # x: [B, D, H, W]
        B, D, H, W = x.shape
        
        # loglikelihood, energy: [B, H, W]
        loglikelihood, energy = self.energy_model(x)
        Ex_pool = energy.unsqueeze(1)
        self.loglikelihood = loglikelihood.mean(dim=[1, 2])

        a = self.conv(x)
       
        noise_var = self.s_logvar.exp() * Ex_pool.detach() + self.b_logvar.exp() * D / 2
        noise_std = torch.sqrt(noise_var + 1e-16)
        
        a = a + noise_std * torch.rand_like(a)
        return a