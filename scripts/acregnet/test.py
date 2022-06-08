import argparse
import os
import time
import numpy as np
import torch

from acregnet.models import ACRegNet
from acregnet.modules import SpatialTransformer
from acregnet.datasets import ImagePairsDataset
from acregnet.utils.io import save_image
from acregnet.utils.tensor import rescale_intensity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_images_file', type=str)
    parser.add_argument('--test_labels_file', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--save_images', action='store_true')
    args = parser.parse_args()

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    os.makedirs(args.results_dir, exist_ok=True)

    if args.save_images:
        output_dir = os.path.join(args.results_dir, 'output')

    test_dataset = ImagePairsDataset(args.test_images_file, args.test_labels_file, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    model = ACRegNet(test_dataset.input_size, test_dataset.num_labels, args, mode='test')
    model.load(args.model_file)
    transformer = SpatialTransformer(test_dataset.input_size, mode='nearest').to(args.device)

    num_pairs = len(test_loader.dataset)

    print('\nTesting:\n')
    with torch.no_grad():

        t_start = time.time()
        for i, (images, labels) in enumerate(test_loader):
            print(f'\t{i+1:>3}/{num_pairs}', end=' ', flush=True)

            moving, fixed = images
            moving, fixed = moving.to(args.device), fixed.to(args.device)

            moving_label, fixed_label = labels
            moving_label, fixed_label = moving_label.to(args.device), fixed_label.to(args.device)

            # Run registration
            output, flow = model.register(moving, fixed)

            # Transform moving label
            output_label = transformer(moving_label, flow)

            if args.save_images:
                cur_out_dir = os.path.join(output_dir, f'{i+1:0>3}')
                os.makedirs(cur_out_dir, exist_ok=True)

                save_image(moving * 255., os.path.join(cur_out_dir, 'im_00_mov.png'))
                save_image(fixed * 255., os.path.join(cur_out_dir, 'im_01_fix.png'))
                save_image(output * 255., os.path.join(cur_out_dir, 'im_02_out.png'))

                save_image(rescale_intensity(moving_label), os.path.join(cur_out_dir, 'lb_00_mov.png'))
                save_image(rescale_intensity(fixed_label), os.path.join(cur_out_dir, 'lb_01_fix.png'))
                save_image(rescale_intensity(output_label), os.path.join(cur_out_dir, 'lb_02_out.png'))

                np.save(os.path.join(cur_out_dir, 'flow.npy'), flow.squeeze().permute(1, 2, 0).cpu().numpy())

            print(f'({(time.time() - t_start):.2f} sec)')
            t_start = time.time()

    with open(os.path.join(args.results_dir, 'run_args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
