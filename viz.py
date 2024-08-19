import argparse
from pathlib import Path
import random
from random import sample

import torch

from utils.stylegan import create_image, load_generator, save_images
from utils.attack_config_parser import AttackConfigParser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    config = AttackConfigParser(args.config)

    args.num_classes = config._config['target_model']['num_classes']

    if args.skip:
        model_name = f"{config._config['target_model']['architecture']}_{args.skip}"
    else:
        model_name = f"{config._config['target_model']['architecture']}_FULL"

    gpu_devices = [i for i in range(torch.cuda.device_count())]
    G = load_generator(config.stylegan_model)

    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = G.num_ws

    w = torch.load(args.weight)
    imgs = create_image(w, synthesis, crop_size=config.attack_center_crop, resize=config.attack_resize)

    img_batches = torch.split(imgs,8)

    random.seed(config._config['seed'])
    if args.num_images:
        random_IDs = sample(range(args.num_classes), args.num_images)
    else:
        random_IDs = range(args.num_classes)

    for i in random_IDs:
        images = img_batches[i]

        folder = f"viz/{model_name}/ID_{i}"
        Path(folder).mkdir(parents=True, exist_ok=True)

        save_images(
            imgs=images,
            folder=folder,
            filename=f"img",
            center_crop=224)
        

def create_parser():
    parser = argparse.ArgumentParser(description='Visualising model inversion attack')
    parser.add_argument('-c', '--config', default=None, type=str, dest="config", help='Config .json file path (default: None)')
    parser.add_argument('-n', '--num_images', default=None, type=int, dest="num_images", help='Config number of images (default: None)')
    parser.add_argument('-s', '--skip', default=None, type=str, dest="skip", help='Specify Skip configuration (default: None)')
    parser.add_argument('-w', '--weight', default=None, type=str, dest="weight", help='Config .pth file path (default: None)')
    return parser


if __name__ == '__main__':
    main()