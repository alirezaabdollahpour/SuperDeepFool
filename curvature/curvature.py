"Estimate model curvature using the power method"

'''
In this code we use source code of this paper -> "https://github.com/kylematoba/lcnn"
'''

import os
import logging
from typing import Tuple
from collections import OrderedDict
from typing import Tuple
from distutils.version import LooseVersion
from typing import Union
import argparse
import socket
import datetime as dt
import copy
import random
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from utils.utils import *
import torch.nn.functional as F
from robustbench import load_cifar10, load_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
torch.cuda.empty_cache()
device = torch.device("cuda")
print(device)




def curvature_hessian_estimator(model: torch.nn.Module,
                        image: torch.Tensor,
                        target: torch.Tensor,
                        num_power_iter: int=20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    model.eval()
    u = torch.randn_like(image)
    u /= torch.norm(u, p=2, dim=(1, 2, 3), keepdim=True)

    with torch.enable_grad():
        image = image.requires_grad_()
        out = model(image)
        y = F.log_softmax(out, 1)
        output = F.nll_loss(y, target, reduction='none')
        model.zero_grad()
        # Gradients w.r.t. input
        gradients = torch.autograd.grad(outputs=output.sum(),
                                        inputs=image, create_graph=True)[0]
        gnorm = torch.norm(gradients, p=2, dim=(1, 2, 3))
        assert not gradients.isnan().any()

        # Power method to find singular value of Hessian
        for _ in range(num_power_iter):
            grad_vector_prod = (gradients * u.detach_()).sum()
            hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image, retain_graph=True)[0]
            assert not hessian_vector_prod.isnan().any()

            hvp_norm = torch.norm(hessian_vector_prod, p=2, dim=(1, 2, 3), keepdim=True)
            u = hessian_vector_prod.div(hvp_norm + 1e-6) #1e-6 for numerical stability

        grad_vector_prod = (gradients * u.detach_()).sum()
        hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image)[0]
        hessian_singular_value = (hessian_vector_prod * u.detach_()).sum((1, 2, 3))
    
    # curvature = hessian_singular_value / (grad_norm + epsilon) by definition
    curvatures = hessian_singular_value.abs().div(gnorm + 1e-6)
    hess = hessian_singular_value.abs()
    grad = gnorm
    
    return curvatures, hess, grad


def measure_curvature(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      data_fraction: float=0.1,
                      batch_size: int=64,
                      num_power_iter: int=20,
                      device: torch.device='cpu') -> Tuple[tuple, tuple, tuple]:

    """
    Compute curvature, hessian norm and gradient norm of a subset of the data given by the dataloader.
    These values are computed using the power method, which requires setting the number of power iterations (num_power_iter).
    """
    
    model.eval()
    datasize = int(data_fraction * len(dataloader.dataset))
    max_batches = int(datasize / batch_size)
    curvature_agg = torch.zeros(size=(datasize,))
    grad_agg = torch.zeros(size=(datasize,))
    hess_agg = torch.zeros(size=(datasize,))

    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device).requires_grad_(), target.to(device)
        with torch.no_grad():
            
            curvatures, hess, grad = curvature_hessian_estimator(model, data, target, num_power_iter=num_power_iter)
        curvature_agg[idx * batch_size:(idx + 1) * batch_size] = curvatures.detach()
        hess_agg[idx * batch_size:(idx + 1) * batch_size] = hess.detach()
        grad_agg[idx * batch_size:(idx + 1) * batch_size] = grad.detach()

        avg_curvature, std_curvature = curvature_agg.mean().item(), curvature_agg.std().item()
        avg_hessian, std_hessian = hess_agg.mean().item(), hess_agg.std().item()
        avg_grad, std_grad = grad_agg.mean().item(), grad_agg.std().item()

        if idx == (max_batches - 1):
            
            print('Average Curvature: {:.6f} +/- {:.2f} '.format(avg_curvature, std_curvature))
            print('Average Hessian Spectral Norm: {:.6f} +/- {:.2f} '.format(avg_hessian, std_hessian))
            print('Average Gradient Norm: {:.6f} +/- {:.2f}'.format(avg_grad, std_grad))
            return


def main():

    parser = argparse.ArgumentParser(description='Experiment arguments')

    parser.add_argument('--model-arch',
                        default="resnet18", 
                        help='What architecture to use?')

    parser.add_argument('--model-filename',
                        type=str, 
                        help='Full filename with path')
    
    parser.add_argument('--dataset', 
                        choices=['cifar10', 'cifar100', "svhn"],
                        default='cifar100',
                        help='Which dataset to use?')

    parser.add_argument("--data-fraction", 
                        type=float, 
                        default=0.1,
                        help="Fraction of data to use for curvature estimation")
    
    parser.add_argument("--batch-size", 
                        type=int, 
                        default=64)

    parser.add_argument('--num-power-iter',
                        type=int,
                        default=10,
                        help="# power iterations for power method")

    parser.add_argument("--prng_seed", 
                        type=int, 
                        default=1729)

    args = parser.parse_args()

    # Show user some information about current job
    logger.info(f"UTC time {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Host: {socket.gethostname()}")

    logger.info("\n----------------------------")
    logger.info("    Argparse arguments")
    logger.info("----------------------------")
    # print all argparse'd args
    for arg in vars(args):
        logger.info(f"{arg} \t {getattr(args, arg)}")
    
    logger.info("----------------------------\n")

    return args



if __name__ == "__main__":
    args = main()
    batch_size = 64
    os.makedirs(os.path.join('data', 'torchvision'), exist_ok=True)
    os.makedirs(os.path.join('results', 'cifar10'), exist_ok=True)

    n_examples = 1000
    images, labels = load_cifar10(n_examples=n_examples, data_dir='data/torchvision')
    print(f"images shape is : {images.shape}")

    # create test_loader with images & labels
    test_dataset = torch.utils.data.TensorDataset(images, labels)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    ##### test data
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # for using robustbenchmarks
    model = load_model(model_name='Standard', dataset='cifar10')
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    requires_grad_(model, False)


    logger.info("\nEstimating curvature on test data...")
    measure_curvature(model, test_loader, 
                        data_fraction=0.2, 
                        batch_size=64, 
                        num_power_iter=10,
                        device=device)