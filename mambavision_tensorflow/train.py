import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from models.registry import create_model

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import shutil
from mambavision_tensorflow.tensorboard import program as tb_program
import utils
from utils.datasets import imagenet_lmdb_dataset

try:
    import wandb
    from wandb.keras import WandbCallback
    has_wandb = True
except ImportError:
    has_wandb = False

# Enable mixed precision
mixed_precision = tf.keras.mixed_precision.experimental
mixed_precision.set_policy('mixed_float16')

# Configure logging
_logger = logging.getLogger('train')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='TensorFlow ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--tag', default='exp', type=str, metavar='TAG')
# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='gc_vit_tiny', type=str, metavar='MODEL',
                    help='Name of model to train (default: "gc_vit_tiny"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--loadcheckpoint', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
scripting_group = group.add_mutually_exclusive_group()
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
group.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8, use opt default)')
group.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
group.add_argument('--clip-grad', type=float, default=5.0, metavar='NORM',
                    help='Clip gradient norm (default: 5.0, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr-ep', action='store_true', default=False,
                        help='using the epoch-based scheduler')
group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
group.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
group.add_argument('--epochs', type=int, default=310, metavar='N',
                    help='number of epochs to train (default: 310)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
group = parser.add_argument_group('Augmentation parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5, metavar='PCT',
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.0, metavar='PCT',
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
group.add_argument('--aug-repeats', type=int, default=0, metavar='N',
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0, metavar='N',
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='const', metavar='NAME',
                    help='Random erase mode (default: "const")')
group.add_argument('--recount', type=int, default=1, metavar='COUNT',
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0, metavar='ALPHA',
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0, metavar='ALPHA',
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, metavar='PAIR',
                    help='cutmix min/max ratio, overrides alpha (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0, metavar='PCT',
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5, metavar='PCT',
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch', metavar='MODE',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
group.add_argument('--head-dropout', type=float, default=None, metavar='PCT',
                    help='Dropout rate of head (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (default: None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (default: None)')
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='', help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model EMA parameters
group = parser.add_argument_group('Model Exponential Moving Average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
group.add_argument('--log-dir', default='', type=str, metavar='LOG_DIR',
                    help='path to log directory (default: none, no logging)')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument('--local_rank', default=0, type=int, metavar='N',
                    help='local rank of current process')
group.add_argument('--use-amp', action='store_true', default=False,
                    help='use NVIDIA AMP automatic mixed precision')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
group.add_argument('--device', default='cuda', type=str,
                    help='device to use for training/testing (default: "cuda")')
group.add_argument('--bench-labels', action='store_true', default=False,
                    help='evaluate as bench/labels model output')

# Distribution training parameters
group = parser.add_argument_group('Distribution training parameters')
group.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
group.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
group.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
group.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
group.add_argument('--rank', default=0, type=int,
                    help='rank of distributed processes')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='use pre-trained model')
group.add_argument('--evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')
group.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
group.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
group.add_argument('--deterministic', action='store_true', default=False,
                    help='make the training deterministic')
group.add_argument('--suffix', default='', type=str, help='suffix for checkpoint and log files')
group.add_argument('--start-eval', default=0, type=int, help='start eval epoch')
group.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')


# Define knowledge distillation loss function
def kdloss(y, teacher_scores):
    T = 3.0
    p = tf.nn.log_softmax(y / T)
    q = tf.nn.softmax(teacher_scores / T)
    l_kl = 50.0 * tf.reduce_sum(q * (tf.math.log(q) - p), axis=-1)
    return l_kl

# Argument parsing
def _parse_args():
    parser = argparse.ArgumentParser(description='TensorFlow Training')
    parser.add_argument('--config', default='', type=str, help='Path to the config file')
    # Add other arguments as needed

    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = _parse_args()
    
    if args.log_wandb:
        import wandb
        wandb.init(project=args.experiment, config=args)
    
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = '/gpu:0'
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.device = f'/gpu:{args.local_rank}'
        tf.config.experimental.set_visible_devices(
            tf.config.experimental.list_physical_devices('GPU')[args.local_rank], 'GPU')
        # TensorFlow's strategy for distributed training can be set here
    
    # Model and optimizer creation, mixed-precision setup
    mixed_precision_policy = None
    if args.amp:
        mixed_precision_policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
    
    # Random seed setup
    tf.random.set_seed(args.seed)
    
    # Model creation
    model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
    
    if args.bfloat:
        dtype = tf.bfloat16
    else:
        dtype = tf.float16

    if args.num_classes is None:
        args.num_classes = model.output_shape[-1]
    
    def resolve_data_config(args, model=None):
        input_size = (args['img_size'], args['img_size'], 3) if 'img_size' in args else (224, 224, 3)
        return {
            'input_size': input_size,
            'interpolation': 'bilinear',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }

    def create_dataset(dataset_name, root, split, is_training=True):
        def parse_image(filename):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            return image
        
        pattern = f"{root}/{split}/*/*.jpg"
        dataset = tf.data.Dataset.list_files(pattern)
        dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    
    def create_loader(dataset, input_size, batch_size, is_training=True):
        def preprocess_image(image):
            image = tf.image.resize(image, input_size[:2])
            image = image / 255.0  # Normalize to [0, 1]
            return image
        
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset


    # Data configuration and loaders
    data_config = resolve_data_config(vars(args), model=model)
    
    # Data loaders (replace with actual data loader functions)
    dataset_train = create_dataset(args.dataset, root=args.data_dir, split=args.train_split, is_training=True)
    dataset_eval = create_dataset(args.dataset, root=args.data_dir, split=args.val_split, is_training=False)
    
    # Data augmentation and loaders
    # (Replace with actual data loader and augmentation functions)
    loader_train = create_loader(dataset_train, input_size=data_config['input_size'], batch_size=args.batch_size)
    loader_eval = create_loader(dataset_eval, input_size=data_config['input_size'], batch_size=args.validation_batch_size)
    
    
    train_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    validate_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Optimizer setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    if mixed_precision_policy:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Checkpointing and logging
    checkpoint_dir = utils.get_outdir(args.output if args.output else f'../output/train/{args.tag}/', args.experiment)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    
    if args.rank == 0:
        log_dir = os.path.join(args.log_dir, args.tag)
        summary_writer = tf.summary.create_file_writer(log_dir)
    else:
        summary_writer = None
    
    # Training loop
    best_metric = None
    best_epoch = None
    for epoch in range(args.start_epoch, args.num_epochs):
        train_one_epoch(epoch, model, loader_train, optimizer, train_loss_fn, args)
        
        if args.distributed:
            # Synchronize batch norm statistics
            pass  # Implement batch norm synchronization if needed
        
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args)
        
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar('test_accuracy', eval_metrics['accuracy'], step=epoch)
                tf.summary.scalar('test_loss', eval_metrics['loss'], step=epoch)
        
        # Save checkpoints
        if checkpoint_manager:
            checkpoint_manager.save()
        
        if eval_metrics['accuracy'] > best_metric:
            best_metric = eval_metrics['accuracy']
            best_epoch = epoch

    if best_metric:
        print(f'Best metric: {best_metric} (epoch {best_epoch})')

def train_one_epoch(
        epoch, model, dataset, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, mixed_precision_policy=None,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and dataset.mixup_enabled:
            dataset.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    batch_time_m = tf.keras.metrics.Mean()
    data_time_m = tf.keras.metrics.Mean()
    losses_m = tf.keras.metrics.Mean()

    model.train()

    end = time.time()
    num_updates = epoch * len(dataset)
    display_first = True

    if args.ampere_sparsity:
        model.enforce_mask()

    for batch_idx, (input, target) in enumerate(dataset):

        if lr_scheduler is not None and not args.lr_ep:
            lr_scheduler.step_update(num_updates=(epoch * len(dataset)) + batch_idx + 1)

        if (batch_idx == 0) or (batch_idx % 50 == 0):
            lrl = [param_group['lr'] for param_group in optimizer.get_weights()]
            lr = sum(lrl) / len(lrl)

        last_batch = batch_idx == len(dataset) - 1
        data_time_m.update(time.time() - end)
        
        if not args.prefetcher:
            input, target = input, target
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = tf.transpose(input, perm=[0, 3, 1, 2])

        with tf.keras.mixed_precision.experimental.Policy(mixed_precision_policy):
            with tf.GradientTape() as tape:
                output = model(input, training=True)
                loss = loss_fn(target, output)

                if args.mesa > 0.0:
                    if epoch / args.epochs > args.mesa_start_ratio:
                        ema_output = model_ema(input, training=False)
                        kd = kdloss(output, ema_output)
                        loss += args.mesa * kd

        if not args.distributed:
            losses_m.update(loss.numpy(), input.shape[0])

        gradients = tape.gradient(loss, model.trainable_variables)
        if loss_scaler is not None:
            gradients = loss_scaler.scale(gradients)
            gradients = loss_scaler.unscale(gradients)
            gradients = [tf.clip_by_value(g, -args.clip_grad, args.clip_grad) for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if model_ema is not None:
            model_ema.update(model)

        tf.keras.backend.clear_session()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:

            lrl = [param_group['lr'] for param_group in optimizer.get_weights()]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = tf.distribute.MirroredStrategy(loss, args.world_size)

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(dataset),
                        100. * batch_idx / (len(dataset) - 1),
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.shape[0] * args.world_size / batch_time_m.result(),
                        rate_avg=input.shape[0] * args.world_size / batch_time_m.result(),
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    tf.keras.preprocessing.image.save_img(
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        input)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None and args.lr_ep:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.result())

        end = time.time()

    return OrderedDict([('loss', losses_m.result().numpy())])

def validate(model, dataset, loss_fn, args, mixed_precision_policy=None, log_suffix=''):
    batch_time_m = tf.keras.metrics.Mean()
    losses_m = tf.keras.metrics.Mean()
    top1_m = tf.keras.metrics.Mean()
    top5_m = tf.keras.metrics.Mean()

    model.evaluate()

    if args.ampere_sparsity:
        model.enforce_mask()

    end = time.time()
    last_idx = len(dataset) - 1
    for batch_idx, (input, target) in enumerate(dataset):
        last_batch = batch_idx == last_idx
        if not args.prefetcher:
            input, target = input, target
        if args.channels_last:
            input = tf.transpose(input, perm=[0, 3, 1, 2])

    with tf.keras.mixed_precision.experimental.Policy(mixed_precision_policy):
        output = model(input, training=False)
    
    # augmentation reduction
    reduce_factor = args.tta
    if reduce_factor > 1:
        output = tf.reduce_mean(tf.reshape(output, [-1, reduce_factor]), axis=1)
        target = target[0:target.shape[0]:reduce_factor]

    def accuracy(output, target, topk=(1,5)):
        """
        Computes the accuracy over the k top predictions for the specified values of k.

        Args:
        - output: Model predictions (logits).
        - target: Ground truth labels.
        - topk: Tuple of integers specifying the top-k accuracies to compute.

        Returns:
        - List of accuracies for each k in topk.
        """
        maxk = max(topk)
        batch_size = target.shape[0]

        # Get the indices of the top-k predictions
        _, pred = tf.math.top_k(output, k=maxk, sorted=True)
        pred = tf.transpose(pred, perm=[1, 0])
        target = tf.reshape(target, [1, -1])
        target = tf.tile(target, [maxk, 1])

        correct = tf.equal(pred, target)
        res = []
        for k in topk:
            correct_k = tf.reduce_sum(tf.cast(correct[:k], dtype=tf.float32), axis=0)
            res.append(tf.reduce_sum(correct_k) * (100.0 / batch_size))
        return res
    
    loss = loss_fn(target, output)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    if args.distributed:
        reduced_loss = tf.distribute.MirroredStrategy(loss, args.world_size)
        acc1 = tf.distribute.MirroredStrategy(acc1, args.world_size)
        acc5 = tf.distribute.MirroredStrategy(acc5, args.world_size)
    else:
        reduced_loss = loss

    tf.keras.backend.clear_session()

    losses_m.update(reduced_loss.numpy(), input.shape[0])
    top1_m.update(acc1.numpy(), output.shape[0])
    top5_m.update(acc5.numpy(), output.shape[0])

    batch_time_m.update(time.time() - end)
    end = time.time()
    if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
        log_name = 'Test' + log_suffix
        _logger.info(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
            'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m,
                loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.result().numpy()), ('top1', top1_m.result().numpy()), ('top5', top5_m.result().numpy())])
    return metrics

if __name__ == '__main__':
    main()
