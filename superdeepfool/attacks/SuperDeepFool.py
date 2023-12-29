import numpy as np
import torch
import torch.nn as nn

from superdeepfool.attacks.attack import Attack
from superdeepfool.attacks.DeepFool import DeepFool


class SuperDeepFool(Attack):
    def __init__(
        self,
        model,
        steps: int = 100,
        overshoot: float = 0.02,
        search_iter: int = 0,
        number_of_samples=None,
        l_norm: str = "L2",
    ):
        super().__init__("SuperDeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.deepfool = DeepFool(
            model, steps=steps, overshoot=overshoot, search_iter=10
        )
        self._supported_mode = ["default"]
        self.search_iter = search_iter
        self.number_of_samples = number_of_samples
        self.fool_checker = 0
        self.l_norm = l_norm
        self.target_label = None

    def forward(self, images, labels, verbose: bool = True):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        curr_steps = 0
        r_tot = torch.zeros_like(images)
        adv_images = [
            images[i :: torch.cuda.device_count()].clone().detach().to(device=i)
            for i in range(torch.cuda.device_count())
        ]
        if batch_size % torch.cuda.device_count() != 0:
            adv_images[-1] = (
                images[
                    torch.cuda.device_count()
                    * (batch_size // torch.cuda.device_count()) :
                ]
                .clone()
                .detach()
                .to(self.device)
            )

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                image = images[idx : idx + 1]
                label = labels[idx : idx + 1]
                r_ = r_tot[idx : idx + 1]
                adv_image = adv_images[idx]

                fs = self.model(adv_image)[0]
                _, pre = torch.max(fs, dim=0)
                if pre != label:
                    correct[idx] = False
                    continue

                adv_image_Deepfool, target_label = self.deepfool(
                    adv_image, label, return_target_labels=True
                )
                r_i = adv_image_Deepfool - image
                adv_image_Deepfool.requires_grad = True
                fs = self.model(adv_image_Deepfool)[0]
                _, pre = torch.max(fs, dim=0)

                if pre == label:
                    pre = target_label
                cost = fs[pre] - fs[label]

                last_grad = torch.autograd.grad(
                    cost, adv_image_Deepfool, retain_graph=False, create_graph=False
                )[0]

                if self.l_norm == "L2":
                    last_grad = last_grad / last_grad.norm()
                    r_ = (
                        r_
                        + (last_grad * (r_i)).sum()
                        * last_grad
                        / (
                            np.linalg.norm(
                                last_grad.detach().cpu().numpy().flatten(), ord=2
                            )
                        )
                        ** 2
                    )

                adv_image = image + r_
                adv_images[idx] = adv_image.detach()
                r_tot[idx] = r_.detach()
                self.target_label = target_label.detach()

            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()
        if self.search_iter > 0:
            if verbose:
                print(f"search iteration for SuperDeepfool -> {self.search_iter}")
            dx = adv_images - images
            dx_l_low, dx_l_high = torch.zeros_like(dx), torch.ones_like(dx)
            for i in range(self.search_iter):
                dx_l = (dx_l_low + dx_l_high) / 2.0
                dx_x = images + dx_l * dx
                dx_y = self.model(dx_x).argmax(-1)
                label_stay = dx_y == labels
                label_change = dx_y != labels
                dx_l_low[label_stay] = dx_l[label_stay]
                dx_l_high[label_change] = dx_l[label_change]
            adv_images = images + dx_l_high * dx
        return adv_images
