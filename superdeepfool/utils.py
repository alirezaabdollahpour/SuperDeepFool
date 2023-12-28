from collections import OrderedDict
from typing import Tuple
import torch
from torch import nn, Tensor
import numpy as np
import copy
import matplotlib.pyplot as plt
import math


class ForwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0 # it should be 0, but we want to start at 1

    def __call__(self, module, input) -> None:
        self.num_samples_called += len(input[0])


class BackwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0

    def __call__(self, module, grad_input, grad_output) -> None:
        self.num_samples_called += len(grad_output[0])

def predict_inputs(model: nn.Module, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    logits = model(inputs)
    probabilities = torch.softmax(logits, 1)
    predictions = logits.argmax(1)
    return logits, probabilities, predictions


class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)

def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


def requires_grad_(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()




def clip_image_values(x, minv, maxv):

    x = torch.max(x, minv)
    x = torch.min(x, maxv)
    return x


def valid_bounds(img, delta=255):

    im = copy.deepcopy(np.asarray(img))
    im = im.astype(np.int)

    # General valid bounds [0, 255]
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)

    # Compute the bounds
    lb = im - delta
    ub = im + delta

    # Validate that the bounds are in [0, 255]
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))

    # Change types to uint8
    lb = lb.astype(np.uint8)
    ub = ub.astype(np.uint8)

    return lb, ub


def inv_tf(x, mean, std):

    for i in range(len(mean)):

        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x


def inv_tf_pert(r):

    pert = np.sum(np.absolute(r), axis=0)
    pert[pert != 0] = 1

    return pert


def get_label(x):
    s = x.split(' ')
    label = ''
    for l in range(1, len(s)):
        label += s[l] + ' '

    return label


def nnz_pixels(arr):
    return np.count_nonzero(np.sum(np.absolute(arr), axis=0))

from typing import Tuple

from torch import Tensor


def robust_accuracy_curve(distances: Tensor,
                          successes: Tensor,
                          worst_distance: float = float('inf')) -> Tuple[Tensor, Tensor]:
    worst_case_distances = distances.clone()
    worst_case_distances[~successes] = worst_distance
    unique_distances = worst_case_distances.unique()
    robust_accuracies = (worst_case_distances.unsqueeze(0) > unique_distances.unsqueeze(1)).float().mean(1)
    return unique_distances, robust_accuracies


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()

def lp_distances(x1: Tensor, x2: Tensor, p = float('inf'), dim: int = 1) -> Tensor:
    return (x1 - x2).flatten(dim).norm(p=p, dim=dim)

def requires_grad_(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


