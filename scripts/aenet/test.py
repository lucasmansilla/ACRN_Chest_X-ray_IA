import argparse
import os
import time
import torch

from acregnet.models import AENet
from acregnet.datasets import LabelsDataset
from acregnet.utils.io import save_image
from acregnet.utils.tensor import to_labels, rescale_intensity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    test_dataset = LabelsDataset(args.test_labels_file, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    model = AENet(test_dataset.input_size, test_dataset.num_labels, args, mode='test')
    model.load(args.model_file)

    num_images = len(test_loader.dataset)

    print('\nTesting:\n')
    with torch.no_grad():

        t_start = time.time()
        for i, inputs in enumerate(test_loader):
            print(f'\t{i+1:>3}/{num_images}', end=' ', flush=True)

            inputs = inputs.to(args.device)
            output = model.predict(inputs)

            if args.save_images:
                cur_out_dir = os.path.join(output_dir, f'{i+1:0>3}')
                os.makedirs(cur_out_dir, exist_ok=True)

                inputs = rescale_intensity(to_labels(inputs.squeeze(0)))
                output = rescale_intensity(to_labels(output.squeeze(0)))

                save_image(inputs, os.path.join(cur_out_dir, 'lb_00_in.png'))
                save_image(output, os.path.join(cur_out_dir, 'lb_01_out.png'))

            print(f'({(time.time() - t_start):.2f} sec)')
            t_start = time.time()

    with open(os.path.join(args.results_dir, 'run_args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
