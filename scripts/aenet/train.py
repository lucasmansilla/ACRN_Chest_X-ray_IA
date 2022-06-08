import argparse
import os
import random
import time
import numpy as np
import torch

from acregnet.models import AENet
from acregnet.datasets import LabelsDataset
from acregnet.dataloader import InfiniteDataLoader
from acregnet.utils.io import save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_labels_file', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.results_dir, exist_ok=True)

    train_dataset = LabelsDataset(args.train_labels_file)
    train_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = AENet(train_dataset.input_size, train_dataset.num_labels, args)

    train_results = {
        'train_loss': [],
        'train_time': []
    }

    print('\nTraining:\n')
    for epoch in range(args.epochs):

        train_loss = 0.0
        train_time = 0.0

        t_start = time.time()
        for step in range(args.steps_per_epoch):

            inputs, target = next(train_loader)
            inputs = inputs.to(args.device)
            target = target.to(args.device)

            # Training
            loss = model.train(inputs, target)

            train_loss += loss
            train_time += (time.time() - t_start)
            t_start = time.time()

        train_loss = train_loss / args.steps_per_epoch
        train_results['train_loss'].append(train_loss)

        train_time = train_time / args.steps_per_epoch
        train_results['train_time'].append(train_time)

        if args.save_model:
            model.save(os.path.join(args.results_dir, 'model.pt'))

        print(f'\tEpoch {epoch+1:>3}/{args.epochs} '
              f'train_loss: {train_loss:.4f} '
              f'step_time: {train_time:.2f} sec')

    if args.save_model:
        model.save(os.path.join(args.results_dir, 'model.pt'))

    save_dict(train_results, os.path.join(args.results_dir, 'train_results.pkl'))

    with open(os.path.join(args.results_dir, 'run_args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
