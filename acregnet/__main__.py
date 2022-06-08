import os
import argparse
import time
import numpy as np

from acregnet.models import ACRegNet
from acregnet.utils.io import read_image, save_image
from acregnet.utils.tensor import to_tensor


parser = argparse.ArgumentParser(description='Image registration with AC-RegNet')
parser.add_argument('-m', '--mov', type=str, help='Moving image')
parser.add_argument('-f', '--fix', type=str, help='Fixed image')
parser.add_argument('--model', type=str, help='Model file')
parser.add_argument('--device', type=str, default='cuda', help='Device')
parser.add_argument('--dst', type=str, help='Output directory')
args = parser.parse_args()


def main():

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    os.makedirs(args.dst, exist_ok=True)

    print('\nReading images', end=' ')
    t_start = time.time()
    moving = to_tensor(read_image(args.mov), add_batch_dim=True) / 255.
    fixed = to_tensor(read_image(args.fix), add_batch_dim=True) / 255.
    print(f'({time.time() - t_start:.2f} sec)')

    print('\nInitializing model', end=' ')
    t_start = time.time()
    model = ACRegNet(moving.shape[2:], None, args, mode='test')
    model.load(args.model)
    print(f'({time.time() - t_start:.2f} sec)')

    print('\nRegistering images', end=' ')
    t_start = time.time()
    output, flow = model.register(moving.to(args.device), fixed.to(args.device))
    print(f'({time.time() - t_start:.2f} sec)')

    print('\nSaving output', end=' ')
    t_start = time.time()
    save_image(output * 255., os.path.join(args.dst, 'result.png'))
    np.save(os.path.join(args.dst, 'flow.npy'), flow.squeeze().permute(1, 2, 0).cpu().numpy())
    print(f'({time.time() - t_start:.2f} sec)')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
