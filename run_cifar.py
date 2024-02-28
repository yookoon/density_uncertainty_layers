import math
import os
import argparse
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torcheval.metrics as metrics

import network
import layers
from utils import ExpectedCalibrationError, CorruptedCIFAR10, CorruptedCIFAR100, OODDataset

# Reference: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

# For debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--save_dir', type=str, default='exps')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--eval_every', type=int, default=25)
parser.add_argument('--n_ensemble', type=int, default=25)
parser.add_argument('--log_every', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--debug', action='store_true')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--model', type=str, default='rank1_density_wrn28')
parser.add_argument('--num_workers', type=int, default=0)

# Bayesian models
parser.add_argument('--kl_annealing_epochs', type=int, default=0)
parser.add_argument('--kl_beta', type=float, default=1.0)
parser.add_argument('--prior_std', type=float, default=1e-1)
parser.add_argument('--posterior_std_init', type=float, default=1e-4)

# For MC Dropout
parser.add_argument('--dropout', type=float, default=0.1)

# Variational Dropout
parser.add_argument('--alpha_init', type=float, default=1e-3)

# Density Uncertainty
parser.add_argument('--pretrain_epochs', type=int, default=1)
parser.add_argument('--ll_scale', type=float, default=0.01)
parser.add_argument('--gen_wd', type=float, default=1e-5)
parser.add_argument('--density_prior_std', type=float, default=1.0)
parser.add_argument('--n_mixture_p', type=float, default=0.0125)


def run(args):
    if args.model == 'resnet':
        args.n_ensemble = 1
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'cifar10':
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        num_classes = 100
        # Use CIFAR-10 statistics for simplicity
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    args.n_train = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    net = getattr(network, args.model)(args, num_classes=num_classes)
    net = net.to(device)

    torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        if isinstance(net, network.DensityModel):
            generative_params = []
            params = []
            for name, param in net.named_parameters():
                if 'masked_conv' in name or 'L' in name or 'log_diagonals' in name or 'logvar' in name:
                    generative_params.append(param)
                else:
                    params.append(param)
            optimizer = SGD([
                {'params': params},
                {'params': generative_params, 'weight_decay': args.gen_wd}
            ], lr=args.lr, momentum=0.9, weight_decay=args.wd)
        else:
            optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    
    def linear_warmup_cosine_annealing(step):
        steps_per_epoch = len(trainloader)
        warmup_steps = args.warmup_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch

        if warmup_steps > 0 and step <= warmup_steps:
            lr = step / warmup_steps
        else:
            lr = (1 + math.cos(math.pi * (step-warmup_steps) / (total_steps-warmup_steps))) / 2
        return lr
    scheduler = LambdaLR(optimizer, linear_warmup_cosine_annealing) 

    writer = SummaryWriter(log_dir=args.log_dir)
    args_string = ' --'.join([f"{k}={v}" for k, v in vars(args).items()])
    writer.add_text('args', args_string)

    # Pretraining the generative energy model: not strictly necessary but helps stabilize training
    if isinstance(net, network.DensityModel) and args.pretrain_epochs > 0:
        generative_params = [param for name, param in net.named_parameters() \
                             if 'masked_conv' in name or 'log_diagonals' in name or 'L' in name in name or 'logvar' in name]
        preoptimizer = SGD(generative_params, lr=args.lr, momentum=0.9, weight_decay=args.gen_wd)
        for epoch in range(1, args.pretrain_epochs + 1):
            train_ll = 0
            n_train = 0
            net.train()
            start = time.time()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = len(inputs)

                outputs = net(inputs)
                loglikelihood = net.loglikelihood()
                loss = -loglikelihood * args.ll_scale
                train_ll += loglikelihood.item() * batch_size

                preoptimizer.zero_grad()
                loss.backward()
                preoptimizer.step()

                n_train += batch_size
                if args.debug:
                    break

            train_ll = train_ll / n_train
            train_time = time.time() - start
            print_str = f'Petrain Epoch: {epoch}/{args.pretrain_epochs}, Time: {train_time:.2f}, LL: {train_ll:.3f}'
            print(print_str)

            writer.add_scalar('pretrain/loglikelihood', train_ll, epoch)

    if args.kl_annealing_epochs > 0:
        kl_mult = 0.0
        step = 0
        kl_annealing_steps = args.kl_annealing_epochs * len(trainloader)
    else:
        kl_mult = 1.0
        kl_annealing_steps = 1

    for epoch in range(1, args.epochs + 1):
        train_loss = 0
        train_kl = 0
        train_ll = 0
        correct = 0
        n_train = 0
        net.train()
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = len(inputs)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item() * batch_size
            if isinstance(net, network.BayesianModel):
                kl_div = net.kl_div() / args.n_train
                loss += args.kl_beta * kl_mult * kl_div
                train_kl += kl_div.item() * batch_size
                kl_mult = min(1.0, kl_mult + 1/kl_annealing_steps)

            if isinstance(net, network.DensityModel):
                loglikelihood = net.loglikelihood()
                loss -= loglikelihood * args.ll_scale
                train_ll += loglikelihood.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(targets).sum().item()
            n_train += batch_size
            if args.debug:
                break

        train_loss = train_loss / n_train
        train_kl = train_kl / n_train
        train_ll = train_ll / n_train
        train_accuracy = 100. * correct / n_train
        train_time = time.time() - start
        
        print_str = f'Epoch: {epoch}/{args.epochs}, Time: {train_time:.2f}, Loss: {train_loss:.3f}, Acc: {train_accuracy:.3f}'
        if isinstance(net, network.BayesianModel):
            print_str += f' KL: {train_kl:.3f}'
        if isinstance(net, network.DensityModel):
            print_str += f' LL: {train_ll:.3f}'
        print(print_str)

        if epoch % args.log_every == 0 or args.debug:
            with torch.no_grad():
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('train/accuracy', train_accuracy, epoch)
                writer.add_scalar('train/time', train_time, epoch)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

                if isinstance(net, network.BayesianModel):
                    writer.add_scalar('train/elbo', train_loss + train_kl, epoch)
                    writer.add_scalar('train/kl', train_kl, epoch)
                    writer.add_scalar('train/kl_mult', kl_mult, epoch)

                if isinstance(net, network.DensityModel):
                    writer.add_scalar('train/loglikelihood', train_ll, epoch)

            writer.flush()

        if epoch % args.eval_every == 0 or epoch == args.epochs or args.debug:
            test(net, testloader, epoch, device, writer, prefix='test', n_ensemble=1, debug=args.debug)

        if args.debug:
            break

    # Ensemble evaluation
    test_loss, test_accuracy, ece = test(net, testloader, args.epochs, device, writer, prefix='ensemble', n_ensemble=args.n_ensemble, debug=args.debug)

    writer.flush()
    torch.save({'args': args, 'state_dict': net.state_dict()}, os.path.join(args.log_dir, './ckpt.pth'))            
    return 


def test(net, testloader, epoch, device, writer=None, prefix='test', n_ensemble=1, debug=False):
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    ece = ExpectedCalibrationError(device=device)

    n_test = 0
    test_loss = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = len(inputs)
            probs = 0.0
            losses = []
            outputs_ensemble = []
            for _ in range(n_ensemble):
                outputs = net(inputs)
                outputs_ensemble.append(outputs)
                loss = criterion(outputs, targets)
                losses.append(loss)
                probs += F.softmax(outputs, dim=1) / n_ensemble

            loss = criterion(probs.log(), targets).mean()
            test_loss += loss.item() * batch_size
            _, predicted = probs.max(dim=1)
            correct += predicted.eq(targets).sum().item()
            ce = ece.update(probs, targets)
            n_test += batch_size

            if debug:
                break

    test_loss = test_loss / n_test
    test_accuracy = 100. * correct / n_test
    test_time = time.time() - start
    ce = ce.item()
    print_str = f'{prefix}: using {n_ensemble} samples, Test time: {test_time:.2f}, ' \
                f'Test acc: {test_accuracy:.3f} ECE: {ce:.6f}, Test loss: {test_loss:.3f}, '
    print(print_str)
    
    if writer is not None:
        with torch.no_grad():
            writer.add_scalar(f'{prefix}/loss', test_loss, epoch)
            writer.add_scalar(f'{prefix}/accuracy', test_accuracy, epoch)
            writer.add_scalar(f'{prefix}/ece', ce, epoch)
            writer.add_scalar(f'{prefix}/time', test_time, epoch)

    return test_loss, test_accuracy, ce


if __name__ == "__main__":
    args = parser.parse_args()
    args.timestamp = datetime.now().strftime("%Y-%m-%d-%H'%M'%S")
    if args.debug:
        args.log_dir = os.path.join(args.save_dir, args.dataset, 'debug')
    else:
        args.log_dir = os.path.join(args.save_dir, args.dataset, args.model, ' '.join([args.timestamp, args.exp_name]).rstrip())
    os.makedirs(args.log_dir, exist_ok=True)
    run(args)