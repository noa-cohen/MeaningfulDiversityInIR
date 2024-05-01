import os

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from PIL import Image

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

def split_batch(batch, fpath, fname):
    for i, image in enumerate(batch):
        filename = f"{fname}_{i}.png"
        tvu.save_image(image.unsqueeze(0), os.path.join(fpath, filename))


def set_seed(seed):
    seed = seed % 2**32
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gaussian_noisy_img(img, noise_level, parallel_batch=1):
    if parallel_batch > 1:
        return img + torch.randn_like(img[0]).expand(parallel_batch, *img.shape[1:]).cuda() * noise_level
    else:
        return img + torch.randn_like(img).cuda() * noise_level


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    out = out.view(n, c, scale*h, scale*w)
    return out


def color2gray(x):
    coef = 1/3
    x = x[:, 0, :, :] * coef + x[:, 1, :, :]*coef + x[:, 2, :, :]*coef
    return x.repeat(1, 3, 1, 1)


def gray2color(x):
    x = x[:, 0, :, :]
    coef = 1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)


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
    elif beta_schedule == "jsd":
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
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, simplified):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

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
                ckpt = os.path.join(self.args.checkpoints_folder, 'celeba_hq.ckpt')
                # ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
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
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion.pt' % (
                            self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.checkpoints_folder, '256x256_diffusion_uncond.pt')
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.checkpoints_folder, '%dx%d_classifier.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        f'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/'
                        f'{image_size}x{image_size}_classifier.pt',
                        ckpt)
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

        if simplified:
            print('Run Simplified DDNM, without SVD.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                  )
            self.simplified_ddnm_plus(model, cls_fn)
        else:
            print('Run SVD-based DDNM.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                  )
            self.svd_based_ddnm_plus(model, cls_fn)

    def simplified_ddnm_plus(self, model, cls_fn):
        raise NotImplementedError("Simplified")

    def svd_based_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting

            masks = {}
            for mask_path in os.listdir(args.path_masks):
                mask = np.array(Image.open(os.path.join(args.path_masks, mask_path)))
                mask = mask[:, :, 0]
                mask = (mask > 127) * 1
                mask = torch.from_numpy(mask).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
                missing_g = missing_r + 1
                missing_b = missing_g + 1
                missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
                masks[mask_path.split('.')[0]] = missing
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv

            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(),
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_operators import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y  # to account for scaling to [-1, 1]
        sigma_y = args.sigma_y

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        # classes_dict = dataset.find_classes(os.path.join(args.exp, "datasets", args.path_y))[0]
        classes_dict = dataset.find_classes(args.path_y)[0]

        if config.model.class_cond:
            with open(os.path.join(args.exp, "imagenet_classes.txt")) as fp:
                fnames_to_classes = {}
                for line in fp.readlines():
                    fnames_to_classes[line.split()[0]] = int(line.split()[1])

        for x_orig, classes in pbar:
            for i in classes:
                os.makedirs(os.path.join(self.args.image_folder, classes_dict[i], "images"),
                            exist_ok=True)

            if args.parallel_batch > 1:
                class_idx = classes_dict[classes[0]]
                if '_' in class_idx:
                    class_idx = class_idx.split('_')[-1]
                set_seed(int(class_idx))

            if deg == 'inpainting':
                if config.sampling.batch_size > 1:
                    raise NotImplementedError("Inpainting currently for B=1")
                A_funcs = Inpainting(config.data.channels, config.data.image_size,
                                     masks[classes_dict[i]], self.device)

            if args.parallel_batch > 1 and config.sampling.batch_size > 1:
                raise NotImplementedError("Parallel batches and normal batches are implemented seperately currently.")

            if args.parallel_batch > 1:
                x_orig = x_orig.expand(args.parallel_batch, *x_orig.shape[1:])
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)

            b, hwc = y.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))

            if self.args.add_noise:  # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y, args.parallel_batch)

            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                         self.config.data.image_size)

            if deg[:6] == 'deblur':
                Apy = y.view(y.shape[0], config.data.channels, self.config.data.image_size,
                             self.config.data.image_size)
            elif deg == 'colorization':
                Apy = y.view(y.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1, 3, 1, 1)
            elif deg == 'inpainting':
                Apy += A_funcs.A_pinv(A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(classes)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[args.parallel_batch * i]),
                    os.path.join(self.args.image_folder, classes_dict[classes[i]], "images",
                                 f"{classes_dict[classes[i]]}_masked.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, Apy[args.parallel_batch * i]),
                    os.path.join(self.args.image_folder,
                                 f"Apy/Apy_{idx_so_far // args.parallel_batch + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[args.parallel_batch * i]),
                    os.path.join(self.args.image_folder,
                                 f"Apy/orig_{idx_so_far // args.parallel_batch + i}.png")
                )

            # translate classes
                if config.model.class_cond:
                    classes_trans = []
                    for i in range(len(classes)):
                        classes_trans.append(fnames_to_classes[classes_dict[classes[i]].split('_')[0]])
                    classes_trans = torch.tensor(classes_trans).to(classes.device)
                else:
                    classes_trans = classes

            # Start DDIM
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                if sigma_y == 0.:  # noise-free case, turn to ddnm
                    x, _, x0_preds, x0_orig_preds, xs = ddnm_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, cls_fn=cls_fn,
                                          classes=classes_trans, config=config,
                                          guidance_eta=args.guidance_eta,
                                          g_dist=args.guidance_dist)
                else:  # noisy case, turn to ddnm+
                    x, _ = ddnm_plus_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, sigma_y,
                                               cls_fn=cls_fn, classes=classes_trans, config=config,
                                               guidance_eta=args.guidance_eta,
                                               g_dist=args.guidance_dist)

            x0_preds = [inverse_data_transform(config, xi) for xi in x0_preds]
            for j, x0_p in enumerate(x0_preds):
                split_batch(x0_p, os.path.join(self.args.image_folder,
                                               classes_dict[classes[0]], "images"),
                                               f"{classes_dict[classes[0]]}_x0_pred_{j}")

            x0_orig_preds = [inverse_data_transform(config, xi) for xi in x0_orig_preds]
            for j, x0_p in enumerate(x0_orig_preds):
                split_batch(x0_p, os.path.join(self.args.image_folder,
                                               classes_dict[classes[0]], "images"),
                                               f"{classes_dict[classes[0]]}_x0_orig_pred_{j}")

            xs = [inverse_data_transform(config, xi) for xi in xs]
            for j, x_p in enumerate(xs):
                split_batch(x_p, os.path.join(self.args.image_folder,
                                               classes_dict[classes[0]], "images"),
                                               f"{classes_dict[classes[0]]}_x_{j}")
            
            x = [inverse_data_transform(config, xi) for xi in x]

            for j in range(x[0].size(0)):
                tvu.save_image(
                    x[0][j], os.path.join(self.args.image_folder,
                                          classes_dict[classes[j // args.parallel_batch]], "images",
                                          f"{j % args.parallel_batch}_"
                                          f"{classes_dict[classes[j // args.parallel_batch]]}.png")
                )

                orig = inverse_data_transform(config, x_orig[j])
                mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))


# Code form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts


def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
