import glob
import os
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
INPUT_DIM = (64, 64, 3)

linux_delimiter = '/'
window_delimiter = '\\'
if os.name == 'nt':
    delimiter = window_delimiter
else:
    delimiter = linux_delimiter

class Data(object):

    datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-6,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.3,
                zoom_range=0.1,
                channel_shift_range=30,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=True,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None)
    dataSetFmt = ['./*/*.jpg', './*/*.jpeg']
  
    def __init__(self, conf):
        self.config = conf
        self.train_valid_ratio = conf['train_valid_ratio']

    def getFileList(self, datadir=DataDir):
        file_list = []
        for i in self.dataSetFmt:
            lis = glob.glob(os.path.join(datadir, i))
            file_list += lis

        self.num_images = len(file_list)
        img_list = []
        label_list_tmp = []

        label_cate_path = glob.glob(DataDir + '/*')
        label_cate = []
        colors = {}
        cloths = {}
        for i in label_cate_path:
            color, cloth = i.split(delimiter)[-1].split('_')
            colors[color] = 1
            cloths[cloth] = 1
        for i in colors.keys():
            label_cate.append(i)
        for i in cloths.keys():
            label_cate.append(i)
        label_idx = {}
        for i, j in enumerate(label_cate):
            label_idx[j] = i
        self.num_class = len(label_cate)
        label_list = np.zeros((self.num_images, self.num_class))
        i = 0
        for f in file_list:
            img_list.append(f)
            labels = f.split(delimiter)[-2]
            cr, cl = labels.split('_')
            label_list[i,label_idx[cr]] = 1
            label_list[i,label_idx[cl]] = 1
            i += 1
        self.label_list = label_list
        self.img_list = img_list
        self.label_cate = label_cate
        self.label_idx = label_idx

    def getTrainBatch(self):
        return DataBatch(self)

    def getValidBatch(self):
        pass

    def preprocessData(img):
        return img

    def augment_data(img_batch, y_batch, batch_size):
        x_batch, y_batch = Data.datagen.flow((img_batch, y_batch), batch_size=batch_size).next()
        return x_batch, y_batch

class DataBatch(Sequence):

    def __init__(self, dataconfig):
        self.config = dataconfig.config
        self.label = dataconfig.label_list
        self.img_list = dataconfig.img_list
        self.num_images = dataconfig.num_images
        self.num_class = dataconfig.num_class
        self.label_cate = dataconfig.label_cate
        self.label_idx = dataconfig.label_idx
        self.generate_list = list(range(len(self)))
        self.augment_data = dataconfig.config['do_augment']
        self.shuffle = dataconfig.config['shuffle']
        print("[INFO] total %d training images" % self.num_images)
        print("[INFO] total %d classes" % self.num_class)
        print("[INFO] ", self.label_cate)
        np.random.shuffle(self.generate_list)


    def __len__(self):
        return int((self.num_images / self.config['batch_size']) + 0.5)

    def size(self):
        return self.num_images

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.generate_list)

    def __getitem__(self, idx):
        x_batch = np.empty((self.config['batch_size'],
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        y_batch = np.empty((self.config['batch_size'],
                            self.num_class))
        idx = self.generate_list[idx]
        idx_base = idx * self.config['batch_size']
        idx_top = idx_base + self.config['batch_size']
        if idx_top > self.size():
            idx_top = self.size()
            size = idx_top - idx_base
            x_batch = x_batch[:size, ...]
            y_batch = y_batch[:size, ...]
        i = 0
        for j in range(idx_base, idx_top):
            img_path = self.img_list[j]
            img = cv2.imread(img_path)[:,:,::-1]
            img = cv2.resize(img, (self.config['img_w'], self.config['img_h']))
            x_batch[i, ...] = img
            y_batch[i, ...] = self.label[j, ...]
            i += 1
        if self.augment_data:
            x_batch, y_batch = Data.augment_data(x_batch, y_batch, batch_size=self.config['batch_size'])
        x_batch = Data.preprocessData(x_batch)
        return x_batch, y_batch

