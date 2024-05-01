import os
import math
import re
import numpy as np
import torch
import json
import argparse

from clustering import get_distance_matrix, get_ordered_distance_index_matrices, dist_from_kth_by_matrix
from utils.utils import load_pt, set_seed, create_path


def weights_prop_to_knn_de(distances, d, k, thresholdpercentage=0, normalize=True):
    r = dist_from_kth_by_matrix(distances, k)
    sampling_probs = pow(r, d)  # sampling probabilities before normalizing by total sum (without the constant)
    if thresholdpercentage != 0:  # We want to change probabilities that are too high to zero
        num_of_indices_to_zeroise = int(len(sampling_probs) * thresholdpercentage)
        indices = np.argsort(sampling_probs)
        if num_of_indices_to_zeroise > 0:
            weights_to_zeroise = indices[-num_of_indices_to_zeroise:]
            sampling_probs[weights_to_zeroise] = 0
        else:
            weights_to_zeroise = ()
    if normalize:
        sampling_probs = sampling_probs / sampling_probs.sum()
    return sampling_probs


def furthest_point_sampling(sample_size, distances, init_method='random', L=None):
    assert L is None or L > sample_size, f"L={L} smaller that sample_size={sample_size}"
    if L is not None:
        distances = distances[:L][:, :L]
    d_len = len(distances)
    # Represent the points by their indices in points
    points_left = np.arange(d_len)  # [P]
    # Initialise an array for the sampled indices
    sample_inds = np.zeros(sample_size, dtype='int')  # [S]
    if init_method == 'random':
        selected = np.random.randint(d_len)
    elif init_method == 'furthest':
        selected = np.unravel_index(np.argmax(distances), distances.shape)[0]
    else:
        raise AttributeError("Unknown initialization method")
    sample_inds[0] = points_left[selected]
    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]
    # Initialise distances to inf
    dists_to_selected = np.ones(d_len) * float('inf')  # [P]
    # Iteratively select points for a maximum of n_samples
    for i in range(1, sample_size):
        # Find the distance to the last added point in selected and all the others
        last_added = sample_inds[i-1]
        dist_to_last_added_point = distances[points_left, last_added]
        # If closer, updated distances
        dists_to_selected[points_left] = np.minimum(dist_to_last_added_point,
                                                    dists_to_selected[points_left])  # [P - i]
        # We want the point that has the largest nearest neighbour distance to the sampled points
        selected = np.argmax(dists_to_selected[points_left])
        sample_inds[i] = points_left[selected]
        # Update points_left
        points_left = np.delete(points_left, selected)
    return sample_inds


def calc_k(n, d, verbose=False):
    k = math.pow(n, 2/(d+2)) * math.pow(math.log(n), d/(d+2))
    if verbose:
        print(f"k={k}, round(k)={round(k)}")
    return round(k)


def recc_add_nearest_neighbor(distances, point, subset, max_dist):
    local_dists = distances.copy()
    local_dists[point, point] = np.inf  # not to choose image itself as neighbor
    neighbor = np.argmin(local_dists[point])
    add_to_subset = []
    if neighbor not in subset and local_dists[point, neighbor] < max_dist:
        add_to_subset.append(neighbor)
        toadd = recc_add_nearest_neighbor(local_dists, neighbor, subset+add_to_subset, max_dist)
        add_to_subset.extend(toadd)
    return add_to_subset


def add_1_nearest_neighbor(distances, point, subset, max_dist):
    local_dists = distances.copy()
    local_dists[point, point] = np.inf  # not to choose image itself as neighbor
    neighbor = np.argmin(local_dists[point])
    add_to_subset = []
    if neighbor not in subset and local_dists[point, neighbor] < max_dist:
        add_to_subset.append(neighbor)
    return add_to_subset


def get_descendants(distances, curr_p, other_ps, ps_in_branch, soft, rec_neighbors, max_dist, seen):
    # get all points closer to current point than to other points
    descendants = np.argwhere(distances[curr_p] < distances[np.delete(other_ps, np.where(other_ps == curr_p))
                                                            ].min(axis=0)).squeeze(1)
    if ps_in_branch is not None:
        # take the intersection of these global points with the points in this branch
        descendants = np.intersect1d(descendants, ps_in_branch)
    added_neighbors = []
    if soft:
        descendants_list = descendants.tolist()
        for i in range(len(descendants_list)):
            if rec_neighbors:
                c_added_neighbors = recc_add_nearest_neighbor(distances, descendants_list[i],
                                                              descendants_list, max_dist)
            else:
                c_added_neighbors = add_1_nearest_neighbor(distances, descendants_list[i],
                                                           descendants_list, max_dist)
            added_neighbors.extend(c_added_neighbors)
        descendants_list.extend(added_neighbors)
        descendants = np.asarray(descendants_list)
    # remove current point as it should not appear in next hierarchies
    descendants = np.delete(descendants, np.where(descendants == curr_p))
    if seen is not None:
        descendants = np.delete(descendants, [i for i in range(len(descendants)) if descendants[i] in seen])
    return descendants, np.asarray(added_neighbors)


def gen_tree_description(features, distances, ssize, sampling_func, metrics, s_probs=None, soft_factor=float('inf'),
                         rec_neighbors=False, max_dist=float('inf')):
    set_seed(125)
    tree = ""
    l_tree = []
    # Initial sample
    seeds = sampling_func(features[0], ssize, metrics[0], distances[0], s_probs[0])
    # Create ssize branches, as string for ete3 and as list for user study
    for i, global_chosen in enumerate(seeds):
        descendants, not2sample = get_descendants(distances[0], global_chosen, seeds, None,
                                                  len(distances[0]) > soft_factor * ssize,
                                                  rec_neighbors, max_dist, seeds)
        curr_desc, l_curr_desc = find_leafs(global_chosen, descendants, distances, features, metrics, sampling_func,
                                            str(i), s_probs, ssize, soft_factor, rec_neighbors,
                                            max_dist, not2sample, seeds)
        l_tree.extend(l_curr_desc)
        tree += curr_desc
        if i != ssize -1 :
            tree += ','
    len_tree_l, len_tree_s = len(l_tree), len(re.findall(r"[\w']+", tree))
    assert len_tree_l == len_tree_s, f"tree list and string with different lengths, {len_tree_l} != {len_tree_s}"
    return '(' + tree + ');', l_tree


def find_leafs(splitter, descendants, distances, features, metrics, sampling_func, parent_c, s_probs, ssize,
               soft_factor, rec_neighbors, max_dist, not2sample, seen):
    if len(descendants) <= ssize:  # stopping criteria: leafs or empty set
        l_leafs = []
        leafs = ""
        if len(descendants) > 0:  # if the splitter has children
            for i, n in enumerate(descendants):  # append all leafs
                l_leafs.append([parent_c+str(i), str(n)])
                leafs += str(n)
                if i != len(descendants) - 1:
                    leafs += ','
            l_leafs.insert(0, [parent_c, str(splitter)])
            return "(" + leafs + ")" + str(splitter), l_leafs
        else:  # if splitter has no children, return it as leaf
            return str(splitter), [[parent_c, str(splitter)]]
    else:
        indices2sample_from = np.delete(descendants, [i for i in range(len(descendants))
                                                      if descendants[i] in not2sample])
        if len(indices2sample_from) < ssize:
            indices2sample_from = descendants.copy()

        sub_features = features[0][indices2sample_from] if features[0] is not None else None
        sub_distances = distances[0][indices2sample_from][:, indices2sample_from]
        sub_s_probs = s_probs[0][indices2sample_from] if s_probs[0] is not None else None
        sub_indices = sampling_func(sub_features, ssize, metrics[0], sub_distances, sub_s_probs)
        distances4des = distances[0]
        global_indices = descendants[sub_indices]
        seen = np.hstack([seen, global_indices]) if seen is not None else global_indices
        childs = ""
        l_childs = []
        for i, c in enumerate(global_indices):
            c_descendants, c_not2sample = get_descendants(distances4des, c, global_indices, descendants,
                                                          len(descendants) > soft_factor * ssize, rec_neighbors,
                                                          max_dist, seen)
            desc, l_desc = find_leafs(c, c_descendants, distances, features, metrics, sampling_func, parent_c+str(i),
                                      s_probs, ssize, soft_factor, max_dist, rec_neighbors, c_not2sample, seen)
            childs += desc
            l_childs.extend(l_desc)
            if i != len(global_indices) - 1:
                childs += ','
        l_childs.insert(0, [parent_c, str(splitter)])
        return "(" + childs + ")" + str(splitter), l_childs


def k_means_sampling(data, k, metric, distances, s_probs):
    from sklearn.cluster import KMeans
    c_data = data.copy()
    if metric == 'cos_similarity':
        from sklearn import preprocessing
        c_data = preprocessing.normalize(c_data)
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(c_data)
    seeds_coordinates = kmeans.cluster_centers_
    _, I, _ = get_ordered_distance_index_matrices(data, 1, query_data=seeds_coordinates, metric=metric)
    return I.squeeze(1)


def random_sampling(data, n, metric, distances, s_probs):
    if len(data) <= n:
        return np.arange(len(data))
    else:
        indices = np.random.choice(np.arange(len(data)), n, replace=False)
        return indices


def fps_sampling(data, n, metric, distances, s_probs):
    return furthest_point_sampling(n, distances)


def weighted_sampling(data, ssize, metric, distances, s_probs):
    s_probs = s_probs / s_probs.sum()
    seeds = np.random.choice(np.arange(len(distances)), ssize, replace=False, p=s_probs)
    return seeds


def calc_max_dist(distances, percentile, verbose=True):
    unique_ds = distances[np.triu_indices_from(distances, k=1)]
    percentile_val = np.percentile(unique_ds, percentile)
    if verbose:
        print(f"percentile{percentile}={percentile_val}, min={unique_ds.min()}, max={unique_ds.max()},"
              f"mean={unique_ds.mean()}, median={np.median(unique_ds)}")
    return percentile_val


def print_iterative_trees(run_name, folder, n_samples, feature_options, s_type, output_path,
                          n_imgs_to_use=None, soft_f=0,
                          limit_n_dist=False, rec_n=False, dists_mat_p=None,
                          is_inet=False, pt_path=None, distances_path=None):
    for model_name, metric, feature_str in feature_options:
        print(f"Generating tree string for {folder}/{run_name}: {model_name} {metric} {feature_str}")
        # get distances
        if is_inet:
            distances = torch.load(dists_mat_p)['dists'].numpy()
            if n_imgs_to_use is not None:
                distances = distances[:n_imgs_to_use][:, :n_imgs_to_use]
                print(distances.shape)
            distances_s = [distances]
            features_s = [None]
            metric_s = [None]
        else:
            # Load features
            features = load_pt(pt_path, feature_str)
            if n_imgs_to_use is not None:
                features = features[:n_imgs_to_use]
            distances = get_distance_matrix(features, metric, True)
            metric_s = [metric]
            features_s = [features]
            distances_s = [distances]
        if distances_path is not None:
            dist_dict = {'distances': distances}
            torch.save(dist_dict, os.path.join(distances_path,
                                               f"{run_name}-{model_name}-{metric}-{feature_str}_{len(distances)}ps.pt"))
            continue
        max_dist = calc_max_dist(distances, 10) if limit_n_dist else float('inf')
        f_name = f"{folder}-{run_name}-{model_name}-{metric}-{feature_str}-{n_samples}from{len(distances)}" \
            f"-sf{soft_f}-rn{rec_n}-maxd{limit_n_dist}"
        out_path = os.path.join(output_path,
                                f_name)
        create_path(out_path)
        if 'attclassifier_pp' in model_name:
            if feature_str == 'l4_1_c1_relu':
                model_name = model_name + '_c1'
            elif feature_str == 'l4_2_c2_relu':
                model_name = model_name + '_c2'
        # Create trees
        trees = {}
        if s_type in ['uniformization', 'all']:
            if is_inet:
                d = 5
                k = calc_k(len(distances), d)
                features = None
            else:
                k = calc_k(*features.shape)
                d = features.shape[1]
            power = 1 if d > 25 else None
            weights, p = weights_prop_to_knn_de(distances, d, k, power=power, normalize=False)
            assert weights.min() > 0, f"weights.min()={weights.min()}"
            assert weights.max() < float('inf'), "weights have infs"
            weights_s = [weights]

            tree_str, tree_l = gen_tree_description(features_s, distances_s, n_samples, weighted_sampling, metric_s,
                                                    weights_s, soft_f, rec_n, max_dist)
            trees['uniformization'] = tree_str
            trees['uniformization_power'] = p
            with open(os.path.join(out_path, f'tree-uni-{model_name}.json'), "w") as f:
                json.dump(tree_l, f)
        if s_type in ['fps', 'all']:
            tree_str, tree_l = gen_tree_description(features_s, distances_s, n_samples, fps_sampling, metric_s,
                                                    [None, None], soft_f, rec_n, max_dist)
            trees['fps'] = tree_str
            with open(os.path.join(out_path, f'tree-fps-{model_name}.json'), "w") as f:
                json.dump(tree_l, f)
        if s_type in ['kmeans', 'all'] and not is_inet:
            tree_str, tree_l = gen_tree_description(features_s, distances_s, n_samples, k_means_sampling, metric_s,
                                                    [None, None], soft_f, rec_n, max_dist)
            trees['kmeans'] = tree_str
            with open(os.path.join(out_path, f'tree-kmeans-{model_name}.json'), "w") as f:
                json.dump(tree_l, f)
        elif s_type not in ['uniformization', 'fps', 'kmeans', 'all']:
            raise AttributeError(f"Unknown sampling type {s_type}")
        # Write tree strings to a file
        print("tree sizes: {}".format([(key, len(re.findall(r"[\w']+", val))) for key, val in trees.items()
                                       if type(val) == str]))
        output_fname = f'{output_path}/{f_name}/{f_name}.json'
        with open(output_fname, "w") as f:
            json.dump(trees, f)


def sample_by_fps(folder, im_name, model_name, feature_str=None, metric='l2', n_samples=5, n_imgs_to_use=None, load_dists=False, l=100):
    if load_dists:
        # get distances
        dists_mat_p = os.path.join(folder, f'{im_name}-lpip_dists.pth')
        distances = torch.load(dists_mat_p)['dists'].numpy()
        if n_imgs_to_use is not None:
            distances = distances[:n_imgs_to_use][:, :n_imgs_to_use]
        features = None
        d = 5
    else:
        # Load features
        pt_path = os.path.join(folder, model_name, f'{model_name}_{im_name}_features.pt')
        features = load_pt(pt_path, feature_str)
        if n_imgs_to_use is not None:
            features = features[:n_imgs_to_use]
        distances = get_distance_matrix(features, metric, True)

    set_seed(125)
    sample_inds = furthest_point_sampling(n_samples, distances, init_method='random', L=l)
    return sample_inds


def sample_by_uniformization(folder, im_name, model_name, feature_str=None, metric='l2', n_samples=5, n_imgs_to_use=None, load_dists=False, k=6, t=0):
    # t stands for the percent of the images not to use, in [0,1]
    if load_dists:
        # get distances
        dists_mat_p = os.path.join(folder, f'{im_name}-lpip_dists.pth')
        distances = torch.load(dists_mat_p)['dists'].numpy()
        if n_imgs_to_use is not None:
            distances = distances[:n_imgs_to_use][:, :n_imgs_to_use]
        features = None
        d = 5
    else:
        # Load features
        pt_path = os.path.join(folder, model_name, f'{model_name}_{im_name}_features.pt')
        features = load_pt(pt_path, feature_str)
        if n_imgs_to_use is not None:
            features = features[:n_imgs_to_use]
        distances = get_distance_matrix(features, metric, True)
        d = features.shape[1]

    set_seed(125)
    weights = weights_prop_to_knn_de(distances, d, k, t)
    assert weights.max() < float('inf'), "weights have infs"
    assert (weights == 0).sum() == int(len(weights)*t), f"{(weights == 0).sum()} weights are zero but {len(weights)}*{t} = {int(len(weights)*t)}"
    sample_inds = np.random.choice(np.arange(len(distances)), n_samples, replace=False, p=weights)
    return sample_inds


def sample_by_kmeans(folder, im_name, model_name, feature_str, metric='l2', n_samples=5, n_imgs_to_use=None, **kwargs):
    pt_path = os.path.join(folder, model_name, f'{model_name}_{im_name}_features.pt')
    features = load_pt(pt_path, feature_str)
    if n_imgs_to_use is not None:
        features = features[:n_imgs_to_use]

    set_seed(125)
    ids = k_means_sampling(features, n_samples, metric, None, None)
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', required=True, type=str, choices=['kmeans', 'unif', 'fps'])
    parser.add_argument('--im_name', type=str, required=True,
                        help='The name of the image to calculate sub-sample using some baseline approach.')
    parser.add_argument('--domain', type=str, required=True, choices=['faces', 'inet'],
                        help='Image domain. Either "faces" or "inet".')
    parser.add_argument('--feature_type', required=True,
                        choices=['patch', 'deep_features'],
                        help='The name of the semantic feature to use.')
    parser.add_argument('--inpainting', action='store_true', help='Inpainting task.')
    parser.add_argument('--features_dir', type=str, default='./Outputs')

    args = parser.parse_args()

    if args.feature_type == 'patch' and not args.inpainting:
        raise ValueError("Patch PCA features were only meant to work on the inpainting task.")

    if args.approach == 'kmeans':
        approach_sampling = sample_by_kmeans
    elif args.approach == 'unif':
        approach_sampling = sample_by_uniformization
    elif args.approach == 'fps':
        approach_sampling = sample_by_fps
    
    load_dists = False
    if args.feature_type == 'patch':
        feature_name = 'patch_pca25'
    elif args.feature_type == 'deep_features':
        if args.domain == 'faces':
            feature_name = 'attclassifier_pp_pca25'
        # ImageNet kmeans
        elif args.approach == 'kmeans':
            feature_name = 'vgg16_pca25'
        else: # ImageNet unif or fps
            feature_name = 'lpips'
            load_dists = True

    path = os.path.join(args.features_dir, 'Distances' if load_dists else 'Features')

    feature_str = {'vgg16_pca25': 'features.30.MaxP',
                   'patch_pca25' : 'features',
                   'lpips': None,
                   'attclassifier_pp_pca25': 'l4_1_c1_relu'}

    print("Sampling...")
    ids = approach_sampling(path, args.im_name, feature_name, feature_str[feature_name], load_dists=load_dists)
    print(ids)
    print("Done.")

