#!/bin/bash
DATA_PATH="/ImageNet/val"
BS=128
checkpoint='/model_weights/mambavision_tiny_1k.pth.tar'

python validate_tf.py --model mamba_vision_T --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch_size $BS --input_size 224 224
