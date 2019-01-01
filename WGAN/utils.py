import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from PIL import Image
from functools import reduce
import os, cv2

'''
    Be careful about everything you write here because every model use the code like:
    from utils import *
    Give variable names carefully and strictly to avoid namespace conflict
'''

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

# arr should be 4 dimensional
def save_image(arr, name, idx, scale=True, path='./generated/'):
    if scale:
        arr = arr * (255.0 / np.max(arr))
    try:
        for i in range(arr.shape[0]):
            img_to_save = arr[i, :, :, :].astype(np.uint8)
            cv2.imwrite(path + str(idx) + '_' + str(i) + '_' + name, img_to_save)
    except:
        print('Error encountered when saving generated images!')
        return
    print('SAVING GENERATED IMAGES TO: ' + path + name)

def show_all_variables():
    all_variables = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)

def get_inception_score(images, sess):
    inps = list(map(lambda img: np.expand_dims(img.astype(np.float32), axis=0), images))
    bs = 100
    preds = list()
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)

# The anime portrait dataset
class animeReader(object):
    def __init__(self, path, low=0, high=5000):
        self.low, self.high = low, high
        fs = os.listdir(path)
        self.files = list(map(lambda x: path + x, fs))
        arrs = list(map(lambda x: cv2.imread(x).astype(np.float32), self.files[low: high]))
        arrs = list(map(lambda x: x.flatten()[None, :], arrs))
        self.data = np.concatenate(arrs, axis=0)
        print('Data successfully loaded --sample number: %d' % (high - low))
        self.counter = 0

    # Randomized output
    def next_batch(self, num):
        index = np.random.randint(low=self.low, high=self.high, size=(1, num))
        x_batch = self.data[index]
        return x_batch[0]

    # Ordinal output
    def ordinal_batch(self, num):
        if self.counter + num < len(self.files):
            end_idx = self.counter + num
            return self.data[self.counter: end_idx, :]
        else:
            start_idx = (self.counter + num) % len(self.files)
            sub1 = self.data[: start_idx, :]
            sub2 = self.data[self.counter:, :]
            return np.concatenate([sub1, sub2], axis=0)


class cifar10Reader(object):
    dataset_dir = 'E:\\datasets\\cifar-10-batches-py\\'

    def __init__(self):
        import pickle
        file_names = ['data_batch_%d' % i for i in range(1, 6)]
        file_names = list(map(lambda x: self.dataset_dir + x, file_names))
        self.dictionaries = list(map(lambda x: pickle.load(open(x, 'rb'), encoding='bytes'), file_names))
        for item in self.dictionaries:
            # The valid value is supposed to be between 0 and 1!!!
            item[b'labels'] = np.array(item[b'labels'])
            item[b'data'] = item[b'data'].astype(np.float32) / 255.0

    def test(self):
        x, y = self.next_batch(64)
        res = x.reshape((64,32,32,3))
        cv2.imwrite('test.png', res[0, :, :, :]*255.0)

    def next_batch(self, num):
        batch_idx = np.random.randint(0, 5)
        image_idx = np.random.randint(0, 10000, size=(num))
        samples = self.dictionaries[batch_idx][b'data'][image_idx]
        labels = self.dictionaries[batch_idx][b'labels'][image_idx]
        # Fucking bloody cifar-10 orders
        for i in range(num):
            res = samples[i, :].reshape((3,32,32)).transpose([1,2,0])
            samples[i, :] = res.reshape((1, 3072))
        return samples, labels


if __name__ == '__main__':
    # Just for tests

    reader = cifar10Reader()
    reader.test()
    # x, y = reader.next_batch(1)
    # print(x, y)
    # print(x.max(), x.min(), x.shape, x.dtype)