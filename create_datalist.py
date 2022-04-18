import os
import numpy as np


def make_datalist(data_fd, data_list):
    filename_all = os.listdir(data_fd)
    filename_all = [data_fd+'/'+filename+'\n' for filename in filename_all if filename.endswith('.tfrecords')]

    np.random.shuffle(filename_all)
    np.random.shuffle(filename_all)
    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)


if __name__ == '__main__':
    data_fd = './data/validation_mr'
    data_list = './data/datalist/validation_mr.txt'

    make_datalist(data_fd, data_list)
