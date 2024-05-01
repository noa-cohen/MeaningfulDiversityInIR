import os
import numpy as np
import torch
from tqdm import tqdm
import lpips
from PIL import Image
from torchvision import transforms


def get_dists_lpips(im_folder, device='cpu', max_num_ims=100):
    imgs_names = os.listdir(im_folder)
    imgs_names = [x for x in imgs_names if 'masked' not in x and '.pth' not in x]
    imgs_prefix = "_".join(imgs_names[0].split('_')[1:])
    imgs_paths = [os.path.join(im_folder, f'{i}_{imgs_prefix}') for i in range(len(imgs_names))]
    ims = []
    for im_path in imgs_paths:
        if len(ims) >= max_num_ims:
            break
        im_pil = Image.open(im_path)
        trans = transforms.ToTensor()
        ims.append((trans(im_pil) * 2 - 1).to(device=device))

    ims = torch.stack(ims).to(device=device)
    with torch.no_grad():
        model = lpips.LPIPS(net='vgg').to(device)
        dists = []
        print(len(ims))
        for im_row in tqdm(ims):
            row_dists = []
            for im in ims:
                row_dists.append(model(im_row.unsqueeze(0).to(device), im.unsqueeze(0).to(device)).detach().cpu())
            dists.append(torch.concat(row_dists))
        dists = torch.stack(dists)
    return dists.squeeze()


def get_distance_matrix(data, metric, sqrt):
    d_len = len(data)
    D, I, _ = get_ordered_distance_index_matrices(data, k=d_len, metric=metric, sqrt=sqrt)
    _, y = np.meshgrid(np.arange(d_len), np.arange(d_len))
    distances = np.zeros_like(D)
    distances[y, I] = D
    assert np.allclose(distances, distances.T, rtol=1e-04, atol=1e-04), \
        f'Distance matrix is not symmetric, max abs diff ={abs(distances - distances.T).max()}'
    return distances


def get_ordered_distance_index_matrices(given_data, k, query_data=None, metric='l2', sqrt=True, verbose=False):
    import faiss
    d = given_data.shape[-1]
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()  # use a single GPU
    if metric == 'l2':
        index_flat = faiss.IndexFlatL2(d)
    elif metric == 'cos_similarity':
        faiss.normalize_L2(given_data)
        if query_data is not None:
            faiss.normalize_L2(query_data)
        index_flat = faiss.IndexFlatIP(d)
    else:
        raise AttributeError("Unsupported metric")
    if torch.cuda.is_available():
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.add(given_data)
    if verbose:
        print(index_flat.ntotal)
    if query_data is None:
        D, I = index_flat.search(given_data, k)
    else:
        D, I = index_flat.search(query_data, k)
    if metric == 'cos_similarity':
        if query_data is None:
            D[:, 0] = 1
        else:
            D[D > 1] = 1
        D = 2 - 2*D
    assert np.all(D >= 0), f'negative distances in D{np.unravel_index(D.argmin(), D.shape)}={D.min()}?'
    if sqrt:
        D = np.sqrt(D)
    return D, I, index_flat


def dist_from_kth_by_matrix(distances, k):
    kth_d = np.partition(distances, k-1, axis=0)[k-1]
    return kth_d