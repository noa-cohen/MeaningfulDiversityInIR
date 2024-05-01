from functools import partial
import os
import argparse
import yaml
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def set_seed(seed):
    seed = seed % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=1)
    # for guidance
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--parallel_batch', type=int, required=False, default=5)
    parser.add_argument('--guidance_eta', type=float, required=False, default=0)
    parser.add_argument('--guidance_dist', type=float, required=False, default=None)
    args = parser.parse_args()

    g_dict = {'g_eta': args.guidance_eta, 'g_dist': args.guidance_dist}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn, g_dict=g_dict)

    # Working directory
    out_path = args.save_dir
    # out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    # for img_dir in ['input', 'recon', 'progress', 'label']:
    #     os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=args.batch_size, num_workers=2, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image batch {i}")
        # fname = str(i).zfill(5) + '.png'
        ref_img, fnames = ref_img
        set_seed(int(fnames[0].split('_')[-1]))

        if args.parallel_batch > 1 and args.batch_size > 1:
            raise NotImplementedError("Parallel batches and normal batches are implemented seperately currently.")

        if args.parallel_batch > 1:
            ref_img = ref_img.expand(args.parallel_batch, *ref_img.shape[1:])
            fnames = fnames * args.parallel_batch
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img, fnames)
            if measure_config['mask_opt']['mask_type'] != 'path':
                mask = mask[:, 0, :, :].unsqueeze(dim=0)
            assert (mask > 0).sum() + (mask < 1).sum() == mask.numel(), "mask is not binary"
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
            g_dict['is_inp'] = True
            g_dict['mask_size'] = (mask[0] == 0.).sum()
            g_dict['mask'] = mask[0]
        else:
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
            g_dict['is_inp'] = False
            g_dict['mask_size'] = ref_img[0].numel()
            g_dict['mask'] = None

        masked_im = operator.transpose(y_n)

        for i in range(len(fnames)):
            os.makedirs(os.path.join(out_path, fnames[i], 'images'), exist_ok=True)
            plt.imsave(os.path.join(out_path, fnames[i], 'images', f'{fnames[i].split("_")[0]}_masked.png'),
                       clear_color(masked_im[i]))
            if args.parallel_batch > 1:  # Only 1 gt in parallel batch
                break

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        for k in range(len(sample)):  # run over batch
            Image.fromarray((clear_color(sample[k]) * 255).astype(np.uint8)).save(
                os.path.join(out_path, fnames[k], 'images', f'{k}_{fnames[k]}.png'))


if __name__ == '__main__':
    main()
