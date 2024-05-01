import torch
import os
from glob import glob
from PIL import Image
from torchvision import transforms
import re
from typing import Callable, Optional, Tuple

# from data.datasets import get_folder_dataset


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, img_transform: Callable, glob_str: str,
                 max_len: Optional[int] = None, must_include: Tuple = ()):
        self.directory = directory
        self.img_transform = img_transform
        self.dataset_filepaths = []
        self.preprocess(glob_str, max_len, must_include)
        self.num_images = len(self.dataset_filepaths)

    def preprocess(self, glob_str: str, max_len: int, must_include: Tuple) -> None:
        imgs_names = [os.path.basename(im_path) for im_path in glob(f"{self.directory}/{glob_str}")
                      if os.path.isfile(im_path) and 'masked' not in im_path]
        imgs_names.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        for i, img_name in enumerate(imgs_names):
            img_path = os.path.join(self.directory, img_name)
            self.dataset_filepaths.append(img_path)
            if max_len is not None:
                # TODO: make sure it does not already exist, and if it does add others
                if i == max_len - len(must_include) - 1:
                    for im_n in must_include:
                        img_path = os.path.join(self.directory, im_n)
                        self.dataset_filepaths.append(img_path)
                    break
        print(f'Finished preprocessing the files in {os.path.basename(self.directory)}...'
              f'Found {len(self.dataset_filepaths)} images.')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.dataset_filepaths[index]
        image = Image.open(img_path)
        return self.img_transform(image), torch.zeros(1)  # TODO: adapt extract_features to work with unlabeled data

    def __len__(self) -> int:
        """Return the number of images."""
        return self.num_images


def get_folder_dataset(folder: str, max_len: Optional[int] = None,
                       must_include: Tuple = (), glob_str: str = '*_output.jpg',
                       img_transform: Callable = transforms.ToTensor()) -> FolderDataset:
    dataset = FolderDataset(folder, img_transform, glob_str, max_len, must_include)
    assert len(dataset) > 0, f"Dataset is empty: {folder}"
    return dataset


def get_dataset(folder: str, im_name: str, max_len: int = 100,
                transform: Callable = transforms.ToTensor()) -> FolderDataset:  # transforms.Compose([])
    images_path = os.path.join(folder, im_name, 'images')
    dataset = get_folder_dataset(images_path, glob_str=f'*_{im_name}.png', max_len=max_len, img_transform=transform)
    print(f"Using {len(dataset)} images")
    assert len(dataset) > 0, "Empty dataset!"
    return dataset
