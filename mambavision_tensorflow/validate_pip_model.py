""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical TensorFlow, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman), translated to TensorFlow.
"""

import argparse
import csv
import json
import logging
import os
import time
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Layer

from models.mamba_vision import create_model, load_checkpoint, list_models, resolve_data_config
from utils import accuracy, AverageMeter, setup_default_logging, set_fast_norm

_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='TensorFlow ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='gpu', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use TensorFlow preprocessing pipeline')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--model-kwargs', nargs='*', default={}, action='store')


parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')

def apply_test_time_pool(model, data_config, pool_type='avg'):
    """Apply test time pooling to the model.

    Args:
        model (tf.keras.Model): The model to modify.
        data_config (dict): Data configuration.
        pool_type (str): Type of pooling ('avg' or 'max').

    Returns:
        tf.keras.Model: Modified model with test-time pooling.
    """
    input_shape = data_config['input_size']
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.Input(shape=input_shape))

    for layer in model.layers:
        if isinstance(layer, GlobalAveragePooling2D) or isinstance(layer, GlobalMaxPooling2D):
            if pool_type == 'avg':
                new_model.add(GlobalAveragePooling2D(name='test_time_avg_pool'))
            elif pool_type == 'max':
                new_model.add(GlobalMaxPooling2D(name='test_time_max_pool'))
        else:
            new_model.add(layer)

    new_model.build((None,) + input_shape)
    new_model.set_weights(model.get_weights())
    return new_model

def validate(args):
    setup_default_logging()

    device = '/GPU:0' if args.device == 'gpu' else '/CPU:0'
    if args.amp:
        policy = mixed_precision.Policy(args.amp_dtype)
        mixed_precision.set_global_policy(policy)

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        **args.model_kwargs,
    )

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, use_ema=args.use_ema)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    if args.test_pool:
        model = apply_test_time_pool(model, data_config, pool_type=args.gp or 'avg')

    # Setup dataset
    input_size = data_config['input_size']
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='int',
        batch_size=args.batch_size,
        image_size=input_size[1:3],
        shuffle=False
    )

    # Validation loop
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    @tf.function
    def compute_loss_and_accuracy(input, target):
        preds = model(input, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(target, preds)
        acc1 = accuracy(preds, target, topk=(1,))
        acc5 = accuracy(preds, target, topk=(5,))
        return loss, acc1, acc5

    end = time.time()
    for batch_idx, (input, target) in enumerate(dataset):
        with tf.device(device):
            loss, acc1, acc5 = compute_loss_and_accuracy(input, target)

        reduced_loss = tf.reduce_mean(loss).numpy()
        top1_acc = tf.reduce_mean(acc1).numpy()
        top5_acc = tf.reduce_mean(acc5).numpy()

        losses.update(reduced_loss, input.shape[0])
        top1.update(top1_acc, input.shape[0])
        top5.update(top5_acc, input.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_freq == 0:
            _logger.info(
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(dataset), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

    _logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

    results = OrderedDict(top1=round(top1.avg, 4), top5=round(top5.avg, 4), loss=round(losses.avg, 4))
    return results

def main():
    setup_default_logging()
    args = parser.parse_args()

    results = validate(args)

    if args.results_file:
        ext = os.path.splitext(args.results_file)[-1].lower()
        if not ext:
            args.results_file += '.' + args.results_format
            ext = os.path.splitext(args.results_file)[-1].lower()

        with open(args.results_file, mode='w') as cf:
            if ext == '.csv':
                cf.write('model,top1,top5,loss\n')
                cf.write('%s,%s,%s,%s\n' % (args.model, results['top1'], results['top5'], results['loss']))
            elif ext == '.json':
                json.dump(results, cf, indent=4)

if __name__ == '__main__':
    main()
