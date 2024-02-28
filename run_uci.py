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
import uci_datasets


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--save_dir', type=str, default='exps/uci_regression')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--n_ensemble', type=int, default=10)
parser.add_argument('--log_every', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--debug', action='store_true')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--model', type=str, default='mlp')

# Bayesian models
parser.add_argument('--kl_annealing_epochs', type=int, default=0)
parser.add_argument('--kl_beta', type=float, default=1.0)
parser.add_argument('--prior_std', type=float, default=1e-1)
parser.add_argument('--posterior_std_init', type=float, default=0.1)

# For MC Dropout
parser.add_argument('--dropout', type=float, default=0.1)

# Variational Dropout
parser.add_argument('--alpha_init', type=float, default=0.1)

# Density Uncertainty
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--ll_scale', type=float, default=0.01)
parser.add_argument('--density_prior_std', type=float, default=1.0)


def run(args):
    print(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark=True

    if args.dataset is not None:
        run_exp(args, args.dataset)
    else:
        for dataset_name in uci_datasets.UCI_DATASETS:
            run_exp(args, dataset_name)

def run_exp(args, dataset_name):
    writer = SummaryWriter(log_dir=args.log_dir)
    args_string = ' --'.join([f"{k}={v}" for k, v in vars(args).items()])
    writer.add_text('args', args_string)

    dataset = getattr(uci_datasets, dataset_name)
    test_rmse_splits = []
    test_loss_splits = []
    for split_id in range(dataset.n_split):
        test_rmse, test_loss = run_split(args, dataset, split_id, writer if split_id==0 else None, prefix=f'{dataset_name}')
        test_rmse_splits.append(test_rmse)
        test_loss_splits.append(test_loss)

    # Log the average and the standard deviation of test RMSE and test NLL
    writer.add_scalar(f'Average/{dataset_name}_rmse', np.mean(test_rmse_splits))
    writer.add_scalar(f'Average/{dataset_name}_rmse_std', np.std(test_rmse_splits, ddof=1))
    writer.add_scalar(f'Average/{dataset_name}_loss', np.mean(test_loss_splits))
    writer.add_scalar(f'Average/{dataset_name}_loss_std', np.std(test_loss_splits, ddof=1))

    print(f'{dataset_name} Average RMSE: {np.mean(test_rmse_splits):.3f}+-{np.std(test_rmse_splits, ddof=1):.3f} '\
          f'Average NLL: {np.mean(test_loss_splits):.3f}+-{np.std(test_loss_splits, ddof=1):.3f}')
    
    writer.flush()

def run_split(args, dataset, split_id, writer, prefix=None):
    trainset = dataset(root='./data', split_id=split_id, train=True)
    testset = dataset(root='./data', split_id=split_id, train=False)
    args.n_train = len(trainset)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    input_dim = trainset.data.shape[-1]
    if isinstance(dataset, uci_datasets.YearPredictionMSD) or isinstance(dataset, uci_datasets.ProteinStructure):
        hidden_dim = 100
    else:
        hidden_dim = 50

    net = getattr(network, args.model)(args, input_dim, hidden_dim)
    net = net.to(args.device)

    if args.optimizer == 'sgd':
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

    # Generative pretraining: not strictly necessary but helps improve the performance slightly
    if isinstance(net, network.DensityModel) and args.pretrain_epochs > 0:
        generative_params = [param for name, param in net.named_parameters() if 'L' in name or 'logvar' in name]
        preoptimizer = SGD(generative_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        for epoch in range(1, args.pretrain_epochs + 1):
            train_ll = 0
            net.train()
            start = time.time()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)

                outputs = net(inputs)
                loglikelihood = net.loglikelihood()
                loss = -loglikelihood * args.ll_scale
                train_ll += loglikelihood.item()

                preoptimizer.zero_grad()
                loss.backward()
                preoptimizer.step()
                if args.debug:
                    break

            train_ll = train_ll / args.n_train
            train_time = time.time() - start
            print_str = f'{prefix}, Petrain Epoch: {epoch}/{args.pretrain_epochs}, Time: {train_time:.2f}, LL: {train_ll:.3f}'
            # print(print_str)

            if writer is not None:
                writer.add_scalar(f'{prefix}/pretrain/loglikelihood', train_ll, epoch)

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
        n_train = 0
        net.train()
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            batch_size = len(inputs)

            outputs = net(inputs)
            loss = torch.mean(0.5 * ((outputs - targets)**2/net.logvar.exp() + np.log(2.0*np.pi) + net.logvar))
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

            n_train += batch_size
            if args.debug:
                break

        train_loss = train_loss / n_train
        train_kl = train_kl / n_train
        train_ll = train_ll / n_train
        train_time = time.time() - start
        if writer is not None:
            print_str = f'{prefix}, Epoch: {epoch}/{args.epochs}, Loss: {train_loss:.3f}'
            if isinstance(net, network.BayesianModel):
                print_str += f' KL: {train_kl:.3f}'
            if isinstance(net, network.DensityModel):
                print_str += f' LL: {train_ll:.3f}'
            print(print_str)

        if epoch % args.log_every == 0 or args.debug:
            if writer is not None:
                with torch.no_grad():
                    writer.add_scalar(f'{prefix}/train/loss', train_loss, epoch)

                    if isinstance(net, network.BayesianModel):
                        writer.add_scalar(f'{prefix}/train/elbo', train_loss + train_kl, epoch)
                        writer.add_scalar(f'{prefix}/train/kl', train_kl, epoch)
                        writer.add_scalar(f'{prefix}/train/kl_mult', kl_mult, epoch)
                        
                    if isinstance(net, network.DensityModel):
                        writer.add_scalar(f'{prefix}/train/loglikelihood', train_ll, epoch)

                writer.flush()

        if epoch % args.eval_every == 0 or args.debug:
            if writer is not None:
                test_rmse, test_loss = test(net, testloader, epoch, args.device, writer, prefix=prefix+'/test10', n_ensemble=10, debug=args.debug)

        if args.debug:
            break

    test_rmse, test_loss = test(net, testloader, args.epochs, args.device, writer, prefix=prefix+f'/ensemble', n_ensemble=args.n_ensemble, debug=args.debug)

    return test_rmse, test_loss


def test(net, testloader, epoch, device, writer=None, prefix='test', n_ensemble=1, debug=False):
    net.eval()
    n_test = 0
    test_loss = 0
    test_rmse = 0
    start = time.time()
    target_std = testloader.dataset.target_std
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = len(inputs)
            preds = 0.0
            losses = []
            for _ in range(n_ensemble):
                outputs = net(inputs)
                preds += outputs / n_ensemble
                loss = 0.5 * ((outputs - targets)**2/net.logvar.exp() + np.log(2.0*np.pi) + net.logvar + 2.0 * np.log(target_std))
                losses.append(loss)

            rmse = torch.mean((preds - targets)**2)
            test_rmse += rmse.item() * batch_size
            losses = torch.stack(losses, dim=0)
            loss = -torch.mean((-losses).logsumexp(dim=0)) + np.log(n_ensemble)
            test_loss += loss.item() * batch_size
            n_test += batch_size

            if debug:
                break

    test_rmse = np.sqrt(test_rmse / n_test) * target_std
    test_loss = test_loss / n_test
    test_time = time.time() - start
    print_str = f'{prefix}, Test RMSE: {test_rmse:.3f}, Test loss: {test_loss:.3f}'
    print(print_str)
    
    if writer is not None:
        with torch.no_grad():
            writer.add_scalar(f'{prefix}/rmse', test_rmse, epoch)
            writer.add_scalar(f'{prefix}/loss', test_loss, epoch)
        
        writer.flush()

    return test_rmse, test_loss


if __name__ == "__main__":
    args = parser.parse_args()
    args.timestamp = datetime.now().strftime("%Y-%m-%d-%H'%M'%S")
    if args.debug:
        args.log_dir = os.path.join(args.save_dir, 'debug')
    else:
        args.log_dir = os.path.join(args.save_dir, args.model, ' '.join([args.timestamp, args.exp_name]).rstrip())
    os.makedirs(args.log_dir, exist_ok=True)
    run(args)