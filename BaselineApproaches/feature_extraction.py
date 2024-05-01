import os.path
from tqdm import tqdm
import argparse
import torch
from torchvision import transforms as tt
from PIL import Image
from typing import Optional, Callable, Dict

from data.datasets import get_dataset, FolderDataset
from utils.utils import create_path


def get_celeba_bbox(landmarks):
    width_factor = 2
    height_factor = 3
    xs, ys = landmarks[::2], landmarks[1::2]  # xs for horizontal dim and ys for vertical dim
    nose_x, nose_y = landmarks[4], landmarks[5]
    assert xs.shape == ys.shape
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    width = (max_x - min_x) * width_factor
    height = (max_y - min_y) * height_factor
    y = nose_y - torch.div(height, 2, rounding_mode='floor')
    x = nose_x - 2 * torch.div(width, 3, rounding_mode='floor')
    return torch.hstack([y, x, height, width])


def get_feature_extractor(name: str, model: torch.nn.Module, ret_nodes: Dict[str, str],
                          verbose_path: Optional[str] = None, device: Optional[str] = None) -> torch.nn.Module:
    # ret_nodes as {"eval_node_name": "my_name"}
    # feature extraction model
    from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
    if ret_nodes is None:
        train_nodes, eval_nodes = get_graph_node_names(model)
        print(train_nodes)
        print(eval_nodes)
        raise RuntimeError("Choose return nodes")
    fe_model = create_feature_extractor(model, return_nodes=ret_nodes)
    if verbose_path is not None:
        with open(os.path.join(verbose_path, name + '_fe_net.txt'), 'w') as f:
            print(fe_model, file=f)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fe_model = fe_model.to(device)
    return fe_model


def reduce_to_k_dims_pca(data: torch.Tensor, k: int) -> torch.Tensor:
    q = min(100, data.shape[-1], data.shape[-2])
    _, _, V = torch.pca_lowrank(data, q=q, center=True)
    first_k_components = torch.matmul(data, V[:, :k])
    return first_k_components


def extract_features(name: str,
                     fe_model: Callable[[torch.Tensor], torch.Tensor],
                     dataset: FolderDataset,
                     save_path: str,
                     device: str,
                     post_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                     pass_labels_to_model: bool = False,
                     save_labels: bool = True,
                     max_dim: Optional[int] = None,
                     pp: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
    print(f"Extracting features: {name}")
    # data
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    # extract
    if hasattr(fe_model, 'eval'):
        fe_model.eval()
    images_features = []
    for data in tqdm(data_loader):
        images, labels = data
        if pp is not None:
            images = pp(images)
        images = images.to(device)
        if torch.is_tensor(labels):
            labels = labels.to(device)
        with torch.no_grad():
            if pass_labels_to_model:
                curr_images_features = fe_model(images, labels)
            else:
                curr_images_features = fe_model(images)
            images_features.append(curr_images_features)
    if torch.is_tensor(images_features[0]):
        images_features = torch.vstack(images_features)
        images_features = {'features': images_features}
    elif type(images_features[0]) is dict:
        if len(images_features) > 1:
            raise NotImplementedError("Batch size smaller than data len when using nodes")
        images_features = images_features[0]
    # save features, labels and classes
    features_dict = {'len': len(labels)}
    if save_labels:
        labels = labels.detach().cpu().numpy()
        features_dict['labels'] = labels
    if hasattr(dataset, "classes"):
        features_dict['classes'] = dataset.classes
    feature_shapes = []
    for key, value in images_features.items():
        value = value.reshape(value.shape[0], -1)
        if max_dim is not None:
            value = reduce_to_k_dims_pca(value, max_dim)
        value = value.detach().cpu().numpy()
        if post_function is not None:
            value = post_function(value)
        features_dict[key] = value
        feature_shapes.append(value.shape)
    full_path = save_path + '/' + name + '_features.pt'
    print(f"Dumping dict with {len(images_features)} features of {len(labels)} images at:\n\t{full_path}")
    print(f"\twith shapes {feature_shapes}")
    torch.save(features_dict, full_path)
    print("Done extracting")


def get_mask_pixels_extractor(mask: torch.Tensor, grayscale: bool = False, keep_black_pixs: bool = False
                              ) -> Callable[[torch.Tensor], torch.Tensor]:
    def get_pixels_in_stat_mask(images: torch.Tensor):
        if grayscale:
            images = tt.Grayscale()(images)
        b, c, _, _ = images.shape
        masks = mask.repeat(b, c, 1, 1)
        if keep_black_pixs:
            pixels = images[masks == 0].reshape(b, -1)
        else:
            pixels = images[masks > 0].reshape(b, -1)
        return pixels
    return get_pixels_in_stat_mask


# -------------- Post Functions
def sum_of_squares(v):
    return (v ** 2).sum(axis=1, keepdims=True)


def get_first_row(features: torch.Tensor) -> torch.Tensor:
    return features.reshape((-1, 40, 2))[:, :, 0]


# -------------- Pre process functions
def attclass_pp(imgs: torch.Tensor) -> torch.Tensor:
    imgs = imgs * 2 - 1
    return imgs


def arcface_pp(imgs: torch.Tensor) -> torch.Tensor:
    pp = tt.Compose([tt.Resize((256, 256)),
                     tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return pp(imgs)


def vgg_pp(imgs: Image) -> torch.Tensor:
    pp = tt.Compose([
                    tt.Resize(256),
                    tt.CenterCrop(224),
                    tt.ToTensor(),
                    tt.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                    )])
    return pp(imgs)


def multiple_extractions(imgs_folder: str,
                         im_name: str,
                         device: str,
                         is_face: bool = False,
                         is_inp: bool = False,
                         mask_p: Optional[str] = None,
                         output_path: str = './Outputs/Features') -> None:

    # Patch
    if is_inp:
        dataset = get_dataset(imgs_folder, im_name)
        mask_pil = Image.open(mask_p)
        thresh = 170

        def fn(x): 255 if x > thresh else 0
        mask_pil = mask_pil.convert('L').point(fn, mode='1')
        mask = tt.ToTensor()(mask_pil)

        model_name = "patch"
        features_name = f"{model_name}_{im_name}"
        features_path = os.path.join(output_path, model_name)
        create_path(features_path)
        feature_extractor = get_mask_pixels_extractor(mask, keep_black_pixs=True)
        extract_features(features_name, feature_extractor, dataset, features_path, device)

        # Patch PCA 25
        model_name = "patch_pca25"
        features_name = f"{model_name}_{im_name}"
        features_path = os.path.join(output_path, model_name)
        create_path(features_path)
        feature_extractor = get_mask_pixels_extractor(mask, keep_black_pixs=True)
        extract_features(features_name, feature_extractor, dataset, features_path, device, max_dim=25)

    if is_face:  # ~~~~~~~~~~~~~~~~~ FOR FACES ~~~~~~~~~~~~~~~~~ #
        dataset = get_dataset(imgs_folder, im_name)
        from criteria.celeba_att_classifier import AnycostPredictor
        ap = AnycostPredictor()
        model_name = "attclassifier_pp"
        features_name = f"{model_name}_{im_name}"
        features_path = os.path.join(output_path, model_name)
        create_path(features_path)
        nodes = {'layer4.1.relu': 'l4_1_c1_relu', 'layer4.2.relu_1': 'l4_2_c2_relu'}
        feature_extractor = get_feature_extractor('att_classifier', ap.estimator, nodes, features_path)
        extract_features(features_name, feature_extractor, dataset, features_path, device, pp=attclass_pp)
        model_name = "attclassifier_pp_logits"
        features_name = f"{model_name}_{im_name}"
        features_path = os.path.join(output_path, model_name)
        create_path(features_path)
        nodes = {'fc': 'fc'}
        feature_extractor = get_feature_extractor('att_classifier', ap.estimator, nodes, features_path)
        extract_features(features_name, feature_extractor, dataset, features_path, device, get_first_row,
                         pp=attclass_pp)

        # PCA 25
        model_name = "attclassifier_pp_pca25"
        features_name = f"{model_name}_{im_name}"
        features_path = os.path.join(output_path, model_name)
        create_path(features_path)
        nodes = {'layer4.1.relu': 'l4_1_c1_relu', 'layer4.2.relu_1': 'l4_2_c2_relu'}
        feature_extractor = get_feature_extractor('att_classifier', ap.estimator, nodes, features_path)
        extract_features(features_name, feature_extractor, dataset, features_path, device, max_dim=25, pp=attclass_pp)
    else:  # ~~~~~~~~~~~~~~~~~ FOR ImageNet ~~~~~~~~~~~~~~~~~ #
        dataset = get_dataset(imgs_folder, im_name, transform=vgg_pp)
        # VGG
        from criteria.vgg import VGG16
        model = VGG16()
        model_name = "vgg16"
        features_name = f"{model_name}_{im_name}"
        features_path = os.path.join(output_path, model_name)
        create_path(features_path)
        nodes = {'features.30': 'features.30.MaxP'}
        feature_extractor = get_feature_extractor(model_name, model.model, nodes, features_path, device)
        extract_features(features_name, feature_extractor, dataset, features_path, device)
        # VGG PCA 25
        model_name = "vgg16_pca25"
        features_name = f"{model_name}_{im_name}"
        features_path = os.path.join(output_path, model_name)
        create_path(features_path)
        feature_extractor = get_feature_extractor(model_name, model.model, nodes, features_path, device)
        extract_features(features_name, feature_extractor, dataset, features_path, device, max_dim=25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use')
    parser.add_argument('--im_dir', type=str, required=True,
                        help='Path to a directory containing and "images" folder with images.'
                             'The name of the directory should be the name of the image.')
    parser.add_argument('--domain', type=str, required=True, choices=['faces', 'inet'],
                        help='Image domain. Either "faces" or "inet".')
    parser.add_argument('--inpainting', action='store_true', help='Inpainting task. Requires --mask_path to be set')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask image')

    args = parser.parse_args()

    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if args.im_dir.endswith('/'):
        args.im_dir = args.im_dir[:-1]

    imgs_folder = os.path.dirname(args.im_dir)
    im_name = os.path.basename(args.im_dir)

    if args.inpainting and args.mask_path is None:
        raise ValueError("Inpainting task requires --mask_path to be set")

    multiple_extractions(imgs_folder, im_name, device,
                         is_face=args.domain == 'faces',
                         is_inp=args.inpainting,
                         mask_p=args.mask_path)
    print("Done.")
