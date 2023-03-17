import argparse
import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from robustbench import load_cifar10, load_model
from robustbench.data import load_cifar100
from attacks.SuperDeepFool import SuperDeepFool
from utils.utils import *

def main(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(8)

    batch_size = 1
    os.makedirs(os.path.join('data', 'torchvision'), exist_ok=True)
    os.makedirs(os.path.join('results', 'cifar10'), exist_ok=True)

    n_examples = args.n_examples
    images, labels = load_cifar10(n_examples=n_examples, data_dir=args.data_dir)
    test_dataset = torch.utils.data.TensorDataset(images, labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name='Standard', dataset='cifar10')
    torch.cuda.empty_cache()
    accuracy_orig = clean_accuracy(model.to(device), images.to(device), labels.to(device),batch_size=batch_size, device=device)
    logger.info(f"Accuracy of original model: {accuracy_orig}")
    requires_grad_(model, False)

    attacks = SuperDeepFool(model, steps=args.steps, overshoot=args.overshoot, search_iter=args.search_iter, number_of_samples=n_examples, l_norm='L2')

    ls = [attacks]
    for attack in ls:
        perturbation_norm_list = []
        for j, data in tqdm(enumerate(test_loader)):
            im, target = data
            im = im.to(device)
            target = target.to(device)
            x = attack(im, target)
            r_tot = torch.abs(x - im)
            perturbation_norm_list.append(l2_norm(r_tot).detach().cpu().numpy())

        mean_r_l2 = np.mean(perturbation_norm_list)
        median_r_l2 = np.median(perturbation_norm_list)
        logger.info(f"mean_r_l2 is:{mean_r_l2}")
        logger.info(f"median_r_l2 is:{median_r_l2}")
        logger.info(f"lenght of perturb is : {len(perturbation_norm_list)}")

if __name__ == '__main__':
    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('results/cifar10/superdeepfool.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    parser = argparse.ArgumentParser(description='Run SuperDeepFool attack on CIFAR-10 dataset')
    parser.add_argument('--n-examples', type=int, default=2, metavar='N',
                        help='number of examples to load for the dataset (default: 2)')
    parser.add_argument('--steps', type=int, default=100, metavar='N',
                        help='maximum number of iterations for SuperDeepFool (default: 100)')
    parser.add_argument('--overshoot', type=float, default=0.02, metavar='F',
                        help='parameter for SuperDeepFool (default: 0.02)')
    parser.add_argument('--search-iter', type=int, default=10, metavar='N',
                        help='number of iterations for the line search of SuperDeepFool (default: 10)')
    parser.add_argument('--data-dir', type=str, default='data/torchvision', metavar='PATH',help='path to the dataset (default: data/torchvision)')
    parser.add_argument('--model-dir', type=str, default='models', metavar='PATH',help='path to the model (default: models)')

    args = parser.parse_args()
    main(args, logger)
