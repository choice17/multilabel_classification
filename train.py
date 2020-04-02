import numpy as np
import cv2
import glob
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from model import MyModel
from data import Data

INPUT_DIM = (64, 64, 3)
activation = 'sigmoid'
DataDir = './dataset'
EPOCHS = 20
modelName = 'mymodel'
log_path = './log'

config = {
    'train_valid_ratio': 0.7,
    'batch_size':16,
    'img_h':INPUT_DIM[0],
    'img_w':INPUT_DIM[1],
    'img_c':INPUT_DIM[2],
    'activation':'sigmoid',
    'epochs': EPOCHS,
    'lr': 1e-2,
    'decay': 1 / EPOCHS,
    'shuffle': 1,
    'do_augment': 1,
    'dropout':1,
    'output_func':'sigmoid',
    'loss':'binary_crossentropy',
    'model':modelName + '.h5'
}

class Train(object):

    def getData(self, conf):
        data = Data(conf)
        data.getFileList(datadir=DataDir)
        self.data = data

    def getModel(self):
        model = MyModel()
        self.model = model.set(input_dim=INPUT_DIM,
                   num_class=self.data.num_class,
                   activation=self.config['activation'])
        self.model.summary()

    def buildGraph(self):
        optimizer = SGD(lr=self.config['lr'], decay=self.config['decay'], momentum=0.9)

        self.model.compile(loss=self.config['loss'], #'binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        

    def fit(self):
        path_name = './tmp'
        file = path_name + '/' + modelName + '-best.h5'
        if os.path.exists(file):
            self.model.load_weights(file)
        #early_stop = EarlyStopping(#monitor='train_loss', 
        #                   min_delta=0.0001, 
        #                   patience=20, 
        #                   mode='min', 
        #                   verbose=1)

        checkpoint = ModelCheckpoint(path_name + '/' + modelName + '-best.h5', 
                                    #monitor='train_loss', 
                                    verbose=1, 
                                    save_best_only=True,
                                    period=1)
        tb_counter  = 1
        tensorboard = TensorBoard(log_dir=log_path + '/' + modelName + '_' + str(tb_counter), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False)
        
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
        #                      patience=2, min_lr=0.000001, verbose=1,
        #                             cooldown=1)

        train_batch =self.data.getTrainBatch()
        self.history = self.model.fit_generator(
                    generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = self.config['epochs'], 
                    verbose          = 1,
                    validation_data  = None,
                    validation_steps = None,
                    callbacks        = [checkpoint, tensorboard], 
                    max_queue_size   = 3)

    def run(self):
        self.config = config
        self.getData(config)
        self.getModel()
        self.buildGraph()
        self.fit()