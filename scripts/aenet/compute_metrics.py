import argparse
import os
import glob
import time
import pandas as pd

from acregnet.utils.io import read_image
from acregnet.utils.metric import dc, hd, assd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    os.makedirs(args.output_dir, exist_ok=True)

    test_images = sorted(glob.glob(os.path.join(args.results_dir, 'output/*')))
    num_images = len(test_images)

    results = {'dsc': [], 'hd': [], 'assd': []}

    print('\nComputing metrics and saving results:\n')
    for i, image_dir in enumerate(test_images):
        print(f'\t{i+1:>3}/{num_images}', end=' ', flush=True)
        t_start = time.time()

        lb_in = read_image(os.path.join(image_dir, 'lb_00_in.png'))
        lb_out = read_image(os.path.join(image_dir, 'lb_01_out.png'))

        # Compute metrics
        results['dsc'].append(dc(lb_out, lb_in))
        results['hd'].append(hd(lb_out, lb_in))
        results['assd'].append(assd(lb_out, lb_in))

        print(f'({(time.time() - t_start):.2f} sec)')

    # Saving results to csv files
    data = pd.DataFrame.from_dict(results)
    data.to_csv(os.path.join(args.output_dir, f'metrics.csv'))

    print('\nDone.\n')
