#!/bin/bash

DATASET='JSRT'

python scripts/aenet/train.py \
    --train_labels_file=data/$DATASET/split/train_labels_256.txt \
    --epochs=100 \
    --steps_per_epoch=100 \
    --batch_size=16 \
    --lr=1e-3 \
    --results_dir=results/$DATASET/AENet/train \
    --save_model

python scripts/aenet/test.py \
    --test_labels_file=data/$DATASET/split/test_labels_256.txt \
    --model_file=results/$DATASET/AENet/train/model.pt \
    --results_dir=results/$DATASET/AENet/test \
    --save_images

python scripts/aenet/compute_metrics.py \
    --results_dir=results/$DATASET/AENet/test \
    --output_dir=results/$DATASET/AENet/metrics