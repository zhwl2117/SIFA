import tensorflow as tf
import json
from typing import Dict, Any, Tuple

with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
MR_MIN, MR_MAX, CT_MIN, CT_MAX = -1.8, 4.4, -2.8, 3.2


decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 3]
        'label_vol': tf.io.FixedLenFeature([], tf.string)}

def _decode(serialized_example: tf.string) -> Dict[str, tf.Tensor]:
    return tf.io.parse_single_example(serialized_example,  decomp_feature)

def _parse(data: Dict[str, Any], data_min, data_max) -> Tuple[tf.Tensor, tf.Tensor]:
    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1] # the label has size [256,256,3] in the preprocessed data, but only the middle slice is used

    data_vol = tf.io.decode_raw(data['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.io.decode_raw(data['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)

    batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)
    batch_x = tf.expand_dims(data_vol[:, :, 1], axis=2)
    batch_x = tf.subtract(tf.multiply(tf.divide(tf.subtract(batch_x, data_min), tf.subtract(data_max, data_min)), 2.0), 1)

    return batch_x, batch_y

# def _decode_samples(image_list, shuffle=False):
#     decomp_feature = {
#         # image size, dimensions of 3 consecutive slices
#         'dsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
#         'dsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
#         'dsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
#         # label size, dimension of the middle slice
#         'lsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
#         'lsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
#         'lsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 1
#         # image slices of size [256, 256, 3]
#         'data_vol': tf.io.FixedLenFeature([], tf.string),
#         # label slice of size [256, 256, 3]
#         'label_vol': tf.io.FixedLenFeature([], tf.string)}

#     raw_size = [256, 256, 3]
#     volume_size = [256, 256, 3]
#     label_size = [256, 256, 1] # the label has size [256,256,3] in the preprocessed data, but only the middle slice is used

#     data_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)
#     reader = tf.TFRecordReader()
#     fid, serialized_example = reader.read(data_queue)
#     parser = tf.parse_single_example(serialized_example, features=decomp_feature)

#     data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
#     data_vol = tf.reshape(data_vol, raw_size)
#     data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

#     label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
#     label_vol = tf.reshape(label_vol, raw_size)
#     label_vol = tf.slice(label_vol, [0, 0, 1], label_size)

#     batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)

#     return tf.expand_dims(data_vol[:, :, 1], axis=2), batch_y


def _load_samples(source_pth, target_pth, source_minmax, target_minmax):

    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
    imagea_list = [row[:-1] for row in rows]

    with open(target_pth, 'r') as fp:
        rows = fp.readlines()
    imageb_list = [row[:-1] for row in rows]

    source_dataset = tf.data.TFRecordDataset(imagea_list)
    source_dataset = source_dataset.map(_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    source_dataset = source_dataset.map(lambda data: _parse(data, source_minmax[0], source_minmax[1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    target_dataset = tf.data.TFRecordDataset(imageb_list)
    target_dataset = target_dataset.map(_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    target_dataset = target_dataset.map(lambda data: _parse(data, target_minmax[0], target_minmax[1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return source_dataset, target_dataset


def load_data(source_pth, target_pth, do_shuffle=True):

    if 'mr' in source_pth:
        source_minmax = [MR_MIN, MR_MAX]
        target_minmax = [CT_MIN, CT_MAX]
    else:
        source_minmax = [CT_MIN, CT_MAX]
        target_minmax = [MR_MIN, MR_MAX]

    source_dataset, target_dataset = _load_samples(source_pth, target_pth, source_minmax, target_minmax)

    # For converting the value range to be [-1 1] using the equation 2*[(x-x_min)/(x_max-x_min)]-1.
    # The values {-1.8, 4.4, -2.8, 3.2} need to be changed according to the statistics of specific datasets
    # if 'mr' in source_pth:
    #     image_i = tf.subtract(tf.multiply(tf.divide(tf.subtract(image_i, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
    # elif 'ct' in source_pth:
    #     image_i = tf.subtract(tf.multiply(tf.divide(tf.subtract(image_i, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)

    # if 'ct' in target_pth:
    #     image_j = tf.subtract(tf.multiply(tf.divide(tf.subtract(image_j, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
    # elif 'mr' in target_pth:
    #     image_j = tf.subtract(tf.multiply(tf.divide(tf.subtract(image_j, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)


    # Batch
    if do_shuffle is True:
        # images_i, images_j, gt_i, gt_j = tf.train.shuffle_batch([image_i, image_j, gt_i, gt_j], BATCH_SIZE, 500, 100)
        source_dataset = source_dataset.repeat().shuffle(100).batch(BATCH_SIZE)
        target_dataset = target_dataset.repeat().shuffle(100).batch(BATCH_SIZE)
    else:
        # images_i, images_j, gt_i, gt_j = tf.train.batch([image_i, image_j, gt_i, gt_j], batch_size=BATCH_SIZE, num_threads=1, capacity=500)
        source_dataset = source_dataset.repeat().batch(BATCH_SIZE)
        target_dataset = target_dataset.repeat().batch(BATCH_SIZE)

    return source_dataset, target_dataset


if __name__ == '__main__':
    source_dataset, target_dataset = load_data(r'./data/datalist/training_mr.txt', r'./data/datalist/training_ct.txt')
    iter_data = iter(source_dataset)
    image, label = next(iter_data)
    image, label = next(iter_data)
    print(tf.shape(image), tf.shape(label))
