# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import numpy as np
import random
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except Exception:
    pass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
th.use_deterministic_algorithms(True)
th.backends.cudnn.benchmark = False


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def set_seed(seed):
    seed = seed % 2**32
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def main(conf: conf_mgt.Default_Conf, g_eta, g_dist, parallel_batch, save_intermediate=False, save_suppl=False,
         s_shift=0):

    print("Start", conf['name'], f"with g_eta={g_eta}, g_dist={g_dist}")

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf, g_eta=g_eta, 
        g_dist=g_dist, save_intermediate=save_intermediate
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...", end='\t')
    dset = 'eval'
    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    print(f"{len(dl)} batches to go...")
    for b_i, batch in enumerate(iter(dl)):
        if parallel_batch > 1 and batch['GT'].shape[0] > 1:
            raise NotImplementedError("Parallel batches and normal batches are implemented seperately currently.")

        if not b_i % 5:
            print(f"[{b_i}/{len(dl)}]")
        im_names = [imgn.split('_')[-1].split('.')[0] for imgn in batch['GT_name']]

        if parallel_batch > 1 and len(set(im_names)) > 1:
            raise NotImplementedError('not all images in batch share the same name')
        set_seed(int(im_names[0]) + s_shift)
        diffusion.curr_im_name = im_names[0]
        diffusion.reverse_steps_count = 0

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}
        gt_keep_mask = batch.get('gt_keep_mask')

        if parallel_batch > 1:
            batch['GT'] = batch['GT'].expand(parallel_batch, *batch['GT'].shape[1:])
            batch['GT_name'] = batch['GT_name'] * parallel_batch
            gt_keep_mask = gt_keep_mask.expand(parallel_batch, *gt_keep_mask.shape[1:])

        model_kwargs["gt"] = batch['GT']

        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop  # defults to p_sample_loop
        )

        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        srs = toU8(result['sample'])
        save_names = batch['GT_name']
        if parallel_batch > 1:
            save_names = [f'{i}_{save_names[0]}' for i in range(len(srs))]
        if save_suppl:
            gts = toU8(result['gt'])
            lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                       th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

            gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

            conf.eval_imswrite(
                srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
                img_names=save_names, dset=dset, name=eval_name, verify_same=False)
        else:
            conf.eval_imswrite(
                srs=srs,
                img_names=save_names, dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--guidance_eta', type=float, required=False, default=None)
    parser.add_argument('--guidance_dist', type=float, required=False, default=None)
    parser.add_argument('--save_intermediate', action='store_true')
    parser.add_argument('--save_suppl', action='store_true')
    parser.add_argument('--s_shift', type=int, required=False, default=0)
    parser.add_argument('--parallel_batch', type=int, required=False, default=5)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    g_eta = args.get('guidance_eta')
    g_dist = args.get('guidance_dist')

    main(conf_arg, g_eta, g_dist, args.get('parallel_batch'),
         args.get('save_intermediate'), args.get('save_suppl'), args.get('s_shift'))
