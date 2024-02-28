from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

from layers import *

class BayesianModel(nn.Module):
    def __init__(self):
        super().__init__()

    def kl_div(self):
        kl_divs = []
        for name, module in self.named_modules():
            if isinstance(module, BayesianModule):
                kl_divs.append(module.kl_div())

        return sum(kl_divs)


class DensityModel(BayesianModel):
    def __init__(self):
        super().__init__()

    def kl_div(self):
        kl_divs = []
        for name, module in self.named_modules():
            if isinstance(module, DensityModule):
                kl_divs.append(module.kl_div())

        return sum(kl_divs)

    def loglikelihood(self):
        loglikelihoods = []
        for name, module in self.named_modules():
            if isinstance(module, DensityModule):
                loglikelihoods.append(module.loglikelihood.mean())

        return sum(loglikelihoods)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv_layer, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes,
                           kernel_size=1, stride=stride, padding=0, bias=True),
            )

    def forward(self, x):
        # Preactivation ResNet
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, conv_layer, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(planes, self.expansion * planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes,
                           kernel_size=1, stride=stride, padding=0, bias=True),
            )

    def forward(self, x):
        # Preactivation ResNet
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=nn.Conv2d, linear_layer=nn.Linear, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv_layer(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(conv_layer, block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(conv_layer, block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(conv_layer, block, 64, num_blocks[2], stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = linear_layer(64*block.expansion, num_classes)

    def _make_layer(self, conv_layer, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(conv_layer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class WideResNet(ResNet):
    def __init__(self, num_blocks=[4, 4, 4], block=BasicBlock, conv_layer=nn.Conv2d, linear_layer=nn.Linear, num_classes=10):
        super().__init__()
        self.in_planes = 16
        k = 10

        self.conv1 = conv_layer(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(conv_layer, block, 16 * k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(conv_layer, block, 32 * k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(conv_layer, block, 64 * k, num_blocks[2], stride=2)
        self.bn1 = nn.BatchNorm2d(64 * k)
        self.linear = linear_layer(64 * k, num_classes)

    def _make_layer(self, conv_layer, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(conv_layer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class BayesianResNet(ResNet, BayesianModel):
    ...
    
class DensityResNet(ResNet, DensityModel):
    ...

def resnet(args, num_classes):
    return ResNet(num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=nn.Conv2d, linear_layer=nn.Linear, num_classes=num_classes)

def bayesian_resnet(args, num_classes):
    conv_layer = partial(BayesianConv2d, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    linear_layer = partial(BayesianLinear, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    return BayesianResNet(num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def mcdropout_resnet(args, num_classes):
    conv_layer = partial(MCDropoutConv2d, dropout=args.dropout)
    linear_layer = partial(MCDropoutLinear, dropout=args.dropout)
    return ResNet(num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def vdropout_resnet(args, num_classes):
    conv_layer = partial(VariationalDropoutConv2d, alpha_init=args.alpha_init, train_alpha=False)
    linear_layer = partial(VariationalDropoutLinear, alpha_init=args.alpha_init, train_alpha=False)
    return BayesianResNet(num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)
 
def rank1_resnet(args, num_classes):
    conv_layer = partial(Rank1Conv2d, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    linear_layer = partial(Rank1Linear, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    return BayesianResNet(num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def density_resnet(args, num_classes):
    conv_layer = partial(DensityConv2d, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init)
    linear_layer = partial(DensityLinear, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init)
    return DensityResNet(num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def rank1_density_resnet(args, num_classes):
    conv_layer = partial(Rank1DensityConv2d, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init, n_mixture_p=args.n_mixture_p)
    linear_layer = partial(Rank1DensityLinear, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init, n_mixture_p=args.n_mixture_p)
    return DensityResNet(num_blocks=[3, 3, 3], block=BasicBlock, conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

class BayesianWideResNet28(WideResNet, BayesianModel):
    ...
    
class DensityWideResNet28(WideResNet, DensityModel):
    ...

def wrn28(args, num_classes):
    return WideResNet(conv_layer=nn.Conv2d, linear_layer=nn.Linear, num_classes=num_classes)

def bayesian_wrn28(args, num_classes):
    conv_layer = partial(BayesianConv2d, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    linear_layer = partial(BayesianLinear, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    return BayesianWideResNet28(conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def mcdropout_wrn28(args, num_classes):
    conv_layer = partial(MCDropoutConv2d, mc_dropout=args.mc_dropout)
    linear_layer = partial(MCDropoutLinear, mc_dropout=args.mc_dropout)
    return WideResNet(conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def vdropout_wrn28(args, num_classes):
    conv_layer = partial(VariationalDropoutConv2d, alpha_init=0.1, train_alpha=False)
    linear_layer = partial(VariationalDropoutLinear, alpha_init=0.1, train_alpha=False)
    return BayesianWideResNet28(conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def rank1_wrn28(args, num_classes):
    conv_layer = partial(Rank1Conv2d, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    linear_layer = partial(Rank1Linear, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    return BayesianWideResNet28(conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def density_wrn28(args, num_classes):
    conv_layer = partial(DensityConv2d, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init)
    linear_layer = partial(DensityLinear, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init)
    return DensityWideResNet28(conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)

def rank1_density_wrn28(args, num_classes):
    conv_layer = partial(Rank1DensityConv2d, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init, n_mixture_p=args.n_mixture_p)
    linear_layer = partial(Rank1DensityLinear, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init, n_mixture_p=args.n_mixture_p)
    return DensityWideResNet28(conv_layer=conv_layer, linear_layer=linear_layer, num_classes=num_classes)


# MLP for UCI benchmarks
class MLP(nn.Module):
    def __init__(self, layer, input_dim, hidden_dim=50):
        super().__init__()
        self.input_proj = layer(input_dim, hidden_dim)
        self.hidden1 = layer(hidden_dim, hidden_dim)
        self.hidden2 = layer(hidden_dim, hidden_dim)
        self.linear = layer(hidden_dim, 1)
        self.logvar = nn.Parameter(torch.zeros(1))
        # self.logvar = nn.Parameter(-np.log(0.1) * torch.ones(1), requires_grad=False)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        h = F.relu(self.hidden1(h))
        h = F.relu(self.hidden2(h))
        out = self.linear(h)
        return out.squeeze()
    
class BayesianMLP(MLP, BayesianModel):
    ...
    
class DensityMLP(MLP, DensityModel):
    ...


def mlp(args, input_dim, hidden_dim):
    layer = nn.Linear
    return MLP(layer, input_dim, hidden_dim)

def bayesian_mlp(args, input_dim, hidden_dim):
    layer = partial(BayesianLinear, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    return BayesianMLP(layer, input_dim, hidden_dim)

def mcdropout_mlp(args, input_dim, hidden_dim):
    layer = partial(MCDropoutLinear, dropout=args.dropout)
    return MLP(layer, input_dim, hidden_dim)

def vdropout_mlp(args, input_dim, hidden_dim):
    layer = partial(VariationalDropoutLinear, alpha_init=args.alpha_init, train_alpha=False)
    return BayesianMLP(layer, input_dim, hidden_dim)

def rank1_mlp(args, input_dim, hidden_dim):
    layer = partial(Rank1Linear, prior_std=args.prior_std, posterior_std_init=args.posterior_std_init)
    return BayesianMLP(layer, input_dim, hidden_dim)

def density_mlp(args, input_dim, hidden_dim):
    layer = partial(DensityLinear, prior_std=args.density_prior_std, posterior_std_init=args.posterior_std_init)
    return DensityMLP(layer, input_dim, hidden_dim)