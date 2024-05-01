import os
import torch
import argparse
from clustering import get_dists_lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use')
    parser.add_argument('--im_dir', type=str, required=True,
                        help='Path to a directory containing and "images" folder with images, of the ImageNet domain.'
                             'The name of the directory should be the name of the image.')
    args = parser.parse_args()

    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    if args.im_dir.endswith('/'):
        args.im_dir = args.im_dir[:-1]

    distances = get_dists_lpips(os.path.join(args.im_dir, 'images'), device)
    im_name = os.path.basename(args.im_dir)

    os.makedirs(os.path.join('Outputs', 'Distances'), exist_ok=True)
    torch.save({'dists': distances}, os.path.join('Outputs', 'Distances', f'{im_name}-lpip_dists.pth'))

    print("Done")
