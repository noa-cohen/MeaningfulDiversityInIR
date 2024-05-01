import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps

import torchvision.utils as tvu

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                # ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.args.checkpoints_folder, "celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.checkpoints_folder, '%dx%d_diffusion.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt'
                             % (self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.checkpoints_folder, '256x256_diffusion_uncond.pt')
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/'
                             '256x256_diffusion_uncond.pt', ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.checkpoints_folder, '%dx%d_classifier.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt'
                             % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F

                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config

        # get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)
        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        def set_seed(seed):
            seed = seed % 2**32
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,  # True,  # TODO: shuffles indices
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # = get degradation matrix = #
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif deg[:3] == 'inp':
            from functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device
                                                                                                          ).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'deno':
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from functions.svd_replacement import SRConv

            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i - np.floor(factor*4/2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            H_funcs = SRConv(kernel / kernel.sum(),
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]
                                   ).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]
                                   ).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'color':
            from functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        else:
            print("ERROR: degradation type not supported")
            quit()
        args.sigma_0 = 2 * args.sigma_0  # to account for scaling to [-1,1]
        sigma_0 = args.sigma_0

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_lr_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)

        if config.data.dataset == "CelebA_HQ":
            classes_dict = dataset.find_classes(args.path_y)[0]
        elif config.data.dataset == "ImageNet":
            classes_dict = dataset.find_classes(args.path_y)[0]

        for x_orig, classes in pbar:
            assert torch.equal(classes, classes[0]*torch.ones_like(classes)), \
                "not all images in batch from the same identity"
            class_idx = classes_dict[classes[0].item()]
            os.makedirs(os.path.join(self.args.image_folder, class_idx, "images"), exist_ok=True)
            set_seed(int(class_idx.split('_')[-1]))
            if args.parallel_batch > 1 and config.sampling.batch_size > 1:
                raise NotImplementedError("Parallel batches and normal batches are implemented seperately currently.")

            if args.parallel_batch > 1:
                x_orig = x_orig.expand(args.parallel_batch, *x_orig.shape[1:])
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y_0 = H_funcs.H(x_orig)
            # y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
            noise = torch.randn_like(y_0[0]).repeat([y_0.shape[0], 1])  # we want the same noise for all images
            y_0 = y_0 + sigma_0 * noise

            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size)
            if deg[:6] == 'deblur':
                pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)
            elif deg == 'color':
                pinv_y_0 = y_0.view(y_0.shape[0], 1, self.config.data.image_size, self.config.data.image_size
                                    ).repeat(1, 3, 1, 1)
            elif deg[:3] == 'inp':
                pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

            # for i in range(len(pinv_y_0)):
            for i in range(len(classes)):
                tvu.save_image(
                    inverse_data_transform(config, pinv_y_0[args.parallel_batch * i]),
                    os.path.join(self.args.image_folder, classes_dict[classes[i]], "images",
                                 f"{classes_dict[classes[i]]}_masked.png"))
                                 #  f"y0_{idx_so_far + i}.png"))
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, class_idx, f"orig_{idx_so_far + i}.png"))

            # Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():
                ret_vals = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False,
                                             cls_fn=cls_fn, classes=classes)
                x, x0_preds, x_adj, x0_adj = ret_vals

            # # debug: check mean of each channel:
            # ctemp = [x0_preds[0][i].mean(dim=[1,2]) for i in range(x0_preds[0].shape[0])]
            # print(ctemp)

            # degrade the generations (x[-1]) to check consistency
            gen_y0 = H_funcs.H(x[-1].to(self.device))
            # gen_y0 = gen_y0 + sigma_0 * noise  # Update: noise should not be added
            pinv_gen_y0 = H_funcs.H_pinv(gen_y0).view(gen_y0.shape[0], config.data.channels,
                                                      self.config.data.image_size, self.config.data.image_size)
            if deg[:6] == 'deblur':
                pinv_gen_y0 = gen_y0.view(gen_y0.shape[0], config.data.channels, self.config.data.image_size,
                                          self.config.data.image_size)
            elif deg == 'color':
                pinv_gen_y0 = gen_y0.view(gen_y0.shape[0], 1, self.config.data.image_size, self.config.data.image_size
                                          ).repeat(1, 3, 1, 1)
            elif deg[:3] == 'inp':
                pinv_gen_y0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_gen_y0))).reshape(*pinv_gen_y0.shape) - 1
            for i, pinv_gen_y0_i in enumerate(pinv_gen_y0):
                tvu.save_image(inverse_data_transform(config, pinv_gen_y0_i),
                               os.path.join(self.args.image_folder, class_idx, f"y0_gen_{idx_so_far+i}.png"))
            # pinv_gen_y0 and pinv_y_0[-1] are both in [-(1+eps),1]
            # and inverse_data_transform(config, *) transforms them to [0,1]

            x = [inverse_data_transform(config, y) for y in x]
            if args.save_intermediate:
                os.makedirs(os.path.join(self.args.image_folder, class_idx, "diffusion_steps"))
                x0_preds = [inverse_data_transform(config, y) for y in x0_preds]
                x_adj = [inverse_data_transform(config, y) for y in x_adj]
                x0_adj = [inverse_data_transform(config, y) for y in x0_adj]

                for i, x_i in enumerate(x):
                    tvu.save_image(x_i,
                                   os.path.join(self.args.image_folder, class_idx, "diffusion_steps", f"x_{i}.png"))
                for i, x0_i in enumerate(x0_preds):
                    tvu.save_image(x0_i,
                                   os.path.join(self.args.image_folder, class_idx, "diffusion_steps", f"x0_{i}_0.png"))
                for i, xadj_i in enumerate(x_adj):
                    tvu.save_image(xadj_i,
                                   os.path.join(self.args.image_folder, class_idx, "diffusion_steps", f"x_{i}_adj.png"))
                for i, x0_adj_i in enumerate(x0_adj):
                    tvu.save_image(x0_adj_i,
                                   os.path.join(self.args.image_folder, class_idx, "diffusion_steps", f"x0_{i}_adj.png"))
            curr_avg_lr_psnr = 0.0
            # for i in [-1]:  # range(len(x)):
            i = -1 
            for j in range(x[i].size(0)):  # For image in batch
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder,
                                          classes_dict[classes[j // args.parallel_batch]], "images",
                                          f"{j % args.parallel_batch}_"
                                          f"{classes_dict[classes[j // args.parallel_batch]]}.png")
                                        #   f"{class_idx}_{j}.png")
                )

                if i == len(x)-1 or i == -1:
                    orig = inverse_data_transform(config, x_orig[j])
                    mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                    psnr = 10 * torch.log10(1 / mse)
                    avg_psnr += psnr
                    # psnr of lr for consistency
                    # pinv_gen_y0 and pinv_y_0[-1] ar both in [-(1+eps),1]
                    # and inverse_data_transform(config, *) transforms them to [0,1]
                    orig_inp = inverse_data_transform(config, pinv_y_0[j])
                    lr_gen = inverse_data_transform(config, pinv_gen_y0[j])
                    mse = torch.mean((lr_gen - orig_inp) ** 2)
                    psnr = 10 * torch.log10(1 / mse)
                    avg_lr_psnr += psnr
                    curr_avg_lr_psnr += psnr

            idx_so_far += y_0.shape[0]
            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        avg_lr_psnr = avg_lr_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR on LR: %.2f" % avg_lr_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

        with open(os.path.join(self.args.image_folder, 'psnr.txt'), 'w') as f:
            f.write(str({'avg_psnr': avg_psnr.item(), 'avg_lr_psnr': avg_lr_psnr.item(),
                         'num_samples': idx_so_far - idx_init}))

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0,
                                        etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta,
                                        cls_fn=cls_fn, classes=classes, guidance_eta=self.args.guidance_eta,
                                        g_dist=self.args.guidance_dist, num_timesteps=self.num_timesteps)
        if last:
            x = x[0][-1]
        return x
