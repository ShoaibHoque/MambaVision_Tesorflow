import os
import io
import lmdb
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds

# Custom dataset splitting functions
def my_random_split(dataset, lengths, seed=0):
    if sum(lengths) != dataset.cardinality().numpy():
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    
    np.random.seed(seed)
    indices = np.random.permutation(sum(lengths))
    
    def _split_index_fn(index):
        offset = 0
        for length in lengths:
            if index < offset + length:
                return index - offset
            offset += length
        raise IndexError
    
    return [dataset.enumerate().filter(lambda i, data: tf.less(i, length)) for length in lengths]

def my_random_split_perc(dataset, percent_train, seed=0):
    num_train = dataset.cardinality().numpy()
    print('Found %d samples' % (num_train))
    sub_num_train = int(np.floor(percent_train * num_train))
    sub_num_valid = num_train - sub_num_train
    dataset_train, dataset_validation = my_random_split(dataset, [sub_num_train, sub_num_valid], seed=seed)
    print('Train: Split into %d samples' % (sub_num_train))
    print('Valid: Split into %d samples' % (sub_num_valid))

    return dataset_train, dataset_validation

# LMDB data loading functions
def lmdb_loader(path, lmdb_env):
    with lmdb_env.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return np.array(img.convert('RGB'))

def imagenet_lmdb_dataset(root, transform=None, target_transform=None):
    if root.endswith('/'):
        root = root[:-1]
    lmdb_path = os.path.join(root + '_faster_imagefolder.lmdb')
    lmdb_env = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)

    pt_path = os.path.join(root + '_faster_imagefolder.lmdb.pt')
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"{pt_path} not found. Make sure the dataset is prepared.")

    dataset = torch.load(pt_path)
    
    images = [lmdb_loader(path, lmdb_env) for path, _ in dataset.imgs]
    labels = [label for _, label in dataset.imgs]
    
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if transform:
        ds = ds.map(lambda x, y: (transform(x), y))
    if target_transform:
        ds = ds.map(lambda x, y: (x, target_transform(y)))

    return ds

# Sample transformations
def normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - normalize['mean']) / normalize['std']
    return image

def val_transform(image, label):
    image = tf.image.resize(image, [args.resize // downscale, args.resize // downscale])
    image = tf.image.central_crop(image, args.resolution // downscale / image.shape[0])
    image = normalize_image(image)
    return image, label

def train_transform(image, label):
    image = tf.image.random_crop(image, [args.resolution // downscale, args.resolution // downscale, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.4)
    image = tf.image.random_contrast(image, 0.4, 0.4)
    image = tf.image.random_saturation(image, 0.4, 0.4)
    image = tf.image.random_hue(image, 0.2)
    image = normalize_image(image)
    return image, label

# Main function
def get_imagenet_loader(args, mode='eval', testdir=""):
    """Get train/val for ImageNet."""
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    print("verify testing path")
    if len(testdir) < 2:
        testdir = os.path.join("../ImageNetV2/", 'test')
    
    normalize = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    downscale = 1

    if 'lmdb' in args.data:
        train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
        valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)
    else:
        train_data = tfds.load('imagenet2012', split='train', as_supervised=True).map(train_transform)
        valid_data = tfds.load('imagenet2012', split='validation', as_supervised=True).map(val_transform)

    if mode == 'eval':
        train_queue = train_data.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        valid_queue = valid_data.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        if args.distributed:
            strategy = tf.distribute.MirroredStrategy()
            train_queue = strategy.experimental_distribute_dataset(train_queue)
            valid_queue = strategy.experimental_distribute_dataset(valid_queue)

    elif mode == 'search':
        train_queue, valid_queue = my_random_split_perc(train_data, args.train_portion, seed=args.seed)

        train_queue = train_queue.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        valid_queue = valid_queue.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        if args.distributed:
            strategy = tf.distribute.MirroredStrategy()
            train_queue = strategy.experimental_distribute_dataset(train_queue)
            valid_queue = strategy.experimental_distribute_dataset(valid_queue)

    elif mode in ['timm', 'timm2', 'timm3']:
        if mode == 'timm2':
            valid_data = tfds.load('imagenet_v2', split='test', as_supervised=True).map(val_transform)
        
        train_queue = train_data.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        valid_queue = valid_data.batch(args.batch_size * 4 if mode == 'timm3' else args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        if args.distributed:
            strategy = tf.distribute.MirroredStrategy()
            train_queue = strategy.experimental_distribute_dataset(train_queue)
            valid_queue = strategy.experimental_distribute_dataset(valid_queue)

    return train_queue, valid_queue, 1000

# Helper functions for CIFAR datasets
def _data_transforms_cifar10(args):
    def train_transform(image, label):
        image = tf.image.resize(image, [args.image_size, args.image_size])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, [args.image_size, args.image_size, 3])
        image = tf.image.per_image_standardization(image)
        return image, label

    def valid_transform(image, label):
        image = tf.image.resize(image, [args.image_size, args.image_size])
        image = tf.image.per_image_standardization(image)
        return image, label

    return train_transform, valid_transform

def _data_transforms_cifar100(args):
    def train_transform(image, label):
        image = tf.image.resize(image, [args.image_size, args.image_size])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, [args.image_size, args.image_size, 3])
        image = tf.image.per_image_standardization(image)
        return image, label

    def valid_transform(image, label):
        image = tf.image.resize(image, [args.image_size, args.image_size])
        image = tf.image.per_image_standardization(image)
        return image, label

    return train_transform, valid_transform

# Data loaders for CIFAR datasets
def get_loaders_eval(dataset, args):
    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        (train_data, valid_data), info = tfds.load('cifar10', split=['train', 'test'], with_info=True, as_supervised=True)
    elif dataset == 'cifar100':
        num_classes = 100
        train_transform, valid_transform = _data_transforms_cifar100(args)
        (train_data, valid_data), info = tfds.load('cifar100', split=['train', 'test'], with_info=True, as_supervised=True)

    train_data = train_data.map(train_transform).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    valid_data = valid_data.map(valid_transform).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.distributed:
        strategy = tf.distribute.MirroredStrategy()
        train_data = strategy.experimental_distribute_dataset(train_data)
        valid_data = strategy.experimental_distribute_dataset(valid_data)

    return train_data, valid_data, num_classes

def get_loaders_search(args):
    if args.dataset == 'cifar10':
        num_classes = 10
        train_transform, _ = _data_transforms_cifar10(args)
        (train_data, _), info = tfds.load('cifar10', split=['train', 'test'], with_info=True, as_supervised=True)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_transform, _ = _data_transforms_cifar100(args)
        (train_data, _), info = tfds.load('cifar100', split=['train', 'test'], with_info=True, as_supervised=True)

    num_train = info.splits['train'].num_examples
    sub_num_train = int(np.floor(args.train_portion * num_train))
    sub_num_valid = num_train - sub_num_train

    train_indices = np.random.choice(num_train, sub_num_train, replace=False)
    valid_indices = np.setdiff1d(np.arange(num_train), train_indices)

    sub_train_data = SubsetDataset(train_data, train_indices).map(train_transform).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    sub_valid_data = SubsetDataset(train_data, valid_indices).map(train_transform).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.distributed:
        strategy = tf.distribute.MirroredStrategy()
        sub_train_data = strategy.experimental_distribute_dataset(sub_train_data)
        sub_valid_data = strategy.experimental_distribute_dataset(sub_valid_data)

    return sub_train_data, sub_valid_data, num_classes

# Main function to get loaders based on dataset type
def get_loaders(args, mode='eval', dataset=None):
    if dataset is None:
        dataset = args.dataset
    if dataset == 'imagenet':
        return get_imagenet_loader(args, mode)
    else:
        if mode == 'search':
            return get_loaders_search(args)
        elif mode == 'eval':
            return get_loaders_eval(dataset, args)

# SubsetDataset class for indexing dataset
class SubsetDataset(tf.data.Dataset):
    def __new__(cls, dataset, indices):
        subset = dataset.enumerate().filter(lambda i, _: tf.reduce_any(i == indices)).map(lambda i, data: data)
        return subset

# Main execution block
if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser('Cell search')
    args = parser.parse_args()
    args.data = '/data/datasets/imagenet_lmdb/'
    args.train_portion = 0.9
    args.batch_size = 48
    args.seed = 1
    args.local_rank = 0

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6020'
    
    tf.random.set_seed(args.seed)
    
    q1, q2, _ = get_imagenet_loader(args, mode='search')

    iterator = iter(q1)
    input_search, target_search = next(iterator)

    print(len(q1), len(q2))
    ind = 0
    for batch, target in q1:
        if ind % 100 == 0:
            print(ind)
        ind += 1

    t1, t2, _ = get_imagenet_loader(args, mode='eval')
    print(len(t1), len(t2))
    for batch, target in t1:
        img = batch[0].numpy().transpose(1, 2, 0)[:, :, 0]
        plt.imshow(img)
        plt.show()
        plt.pause(1.)
        break
