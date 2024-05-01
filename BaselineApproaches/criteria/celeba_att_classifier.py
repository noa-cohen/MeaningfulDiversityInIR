import torch
from torchvision import models

"""
Code is adopted from: AnyCostGAN (https://github.com/mit-han-lab/anycost-gan)
Example of original use in:
https://github.com/mit-han-lab/anycost-gan/blob/5be666daf0eed6189e792a3381c285c749bb4b1e/metrics/attribute_consistency.py
"""

URL_TEMPLATE = 'https://hanlab.mit.edu/projects/anycost-gan/files/{}_{}.pt'

def safe_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False,
                                  file_name=None):
    return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)


def load_state_dict_from_url(url, key=None):
    if url.startswith('http'):
        sd = safe_load_state_dict_from_url(url, map_location='cpu', progress=True)
    else:
        sd = torch.load(url, map_location='cpu')
    if key is not None:
        return sd[key]
    return sd


def get_pretrained(model):
    if model == 'attribute-predictor':
        predictor = models.resnet50()
        predictor.fc = torch.nn.Linear(predictor.fc.in_features, 40 * 2)
        predictor.load_state_dict(load_state_dict_from_url('../checkpoints/attribute_predictor.pt', 'state_dict'))
        return predictor
    else:
        raise NotImplementedError


class AnycostPredictor:
    def __init__(self, pp=False):
        if torch.cuda.is_available():
            self.estimator = get_pretrained('attribute-predictor').cuda()
        else:
            self.estimator = get_pretrained('attribute-predictor')
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.pp = pp

    def pre_process(self, img):
        # preprocess image to [-1,1]
        img = img * 2 - 1
        return img

    def get_attr(self, img):
        # get attribute scores for the generated image
        if self.pp:
            img = self.pre_process(img)
        img = self.face_pool(img)
        logits = self.estimator(img).view((-1, 40, 2))[0]
        attr_preds = torch.nn.functional.softmax(logits, dim=1).cpu().detach()
        return attr_preds
