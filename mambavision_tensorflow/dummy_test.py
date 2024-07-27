import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', metavar='NAME', default='mamba_vision_T', help='model architecture (default: mamba_vision_T)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--use_pip', action='store_true', default=False, help='to use pip package')
args = parser.parse_args()

# Define mamba_vision_T model with 224 x 224 resolution
if args.use_pip:
    from mambavision import create_model
    model = create_model(args.model, pretrained=True, model_path="/tmp/mambavision_tiny_1k.h5")
else:
    from models.mamba_vision import *
    model = create_model(args.model)

    if args.checkpoint:
        model.load_weights(args.checkpoint)

print('{} model successfully created!'.format(args.model))

# Create a dummy input image with the same dimensions as in the PyTorch example
image = tf.random.uniform((1, 754, 234, 3))

# Perform inference on the dummy input
output = model(image)

print('Inference successfully completed on dummy input!')

