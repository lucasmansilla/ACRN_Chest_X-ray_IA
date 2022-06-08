#!/bin/bash

DATASET='JSRT' 

python scripts/acregnet/train.py \
    --train_images_file=data/$DATASET/split/train_images_256.txt \
    --train_labels_file=data/$DATASET/split/train_labels_256.txt \
    --autoencoder_file=results/$DATASET/AENet/train/model.pt \
    --epochs=10 \
    --steps_per_epoch=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --flow_weight=5e-5 \
    --label_weight=1.0 \
    --shape_weight=1e-1 \
    --results_dir=results/$DATASET/ACRegNet/train \
    --save_model

python scripts/acregnet/test.py \
    --test_images_file=data/$DATASET/split/test_images_256.txt \
    --test_labels_file=data/$DATASET/split/test_labels_256.txt \
    --model_file=results/$DATASET/ACRegNet/train/model.pt \
    --results_dir=results/$DATASET/ACRegNet/test \
    --save_images

python scripts/acregnet/compute_metrics.py \
    --results_dir=results/$DATASET/ACRegNet/test \
    --output_dir=results/$DATASET/ACRegNet/metrics