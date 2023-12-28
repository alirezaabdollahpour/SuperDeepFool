import torch
from joblib import Parallel, delayed


class Search:
    def __init__(self, model, search_iter=10, num_jobs=15):
        self.model = model
        self.search_iter = search_iter
        self.num_jobs = num_jobs

    def generate(self, images, adv_images, labels):
        images = images.clone()
        adv_images = adv_images.clone()
        dx = adv_images - images
        dx_l_low, dx_l_high = torch.zeros_like(dx), torch.ones_like(dx)

        def loop_job(idx):
            dx_l = (dx_l_low + dx_l_high) / 2.
            dx_x = images + dx_l * dx
            dx_y = self.model(dx_x).argmax(-1)
            label_stay = dx_y == labels
            label_change = dx_y != labels
            dx_l_low[label_stay] = dx_l[label_stay]
            dx_l_high[label_change] = dx_l[label_change]
            return dx_l_low, dx_l_high
        
        for i in range(self.search_iter):
            loop_inputs = range(dx.shape[0])
            loop_outputs = Parallel(n_jobs=self.num_jobs)(
                delayed(loop_job)(idx) for idx in loop_inputs)
            dx_l_low, dx_l_high = zip(*loop_outputs)
            dx_l_low, dx_l_high = torch.stack(dx_l_low), torch.stack(dx_l_high)

        adv_images = images + dx_l_high * dx
        return adv_images
