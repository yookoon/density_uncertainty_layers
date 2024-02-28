import numpy as np
import torch

class FactorizedGaussian():
    def __init__(self, mu, logvar):
        # mu: [D] or [D1, D2] or [D1, D2, D3, D4]
        # logvar: [D] or [D1, D2] or [D1, D2, D3, D4]
        self.mu = mu
        self.logvar = logvar

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(self.mu)
        return self.mu + std * eps
    
    def logpdf(self, x):
        # x: [D] or [D1, D2] or [D1, D2, D3, D4]
        return -0.5 * torch.sum(np.log(2.0*np.pi) + self.logvar + ((x - self.mu))**2 / torch.exp(self.logvar))

    def kl_div(self, q):
        assert isinstance(q, FactorizedGaussian)
        return 0.5 * torch.sum(q.logvar - self.logvar - 1.0 + (torch.exp(self.logvar) + (self.mu - q.mu)**2)/(torch.exp(q.logvar)))