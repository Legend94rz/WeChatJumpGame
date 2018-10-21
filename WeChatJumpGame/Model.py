import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config = config)
K.set_session(session)

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D, Reshape, Flatten
import numpy as np
import cv2
import pandas as pd
from keras.utils import plot_model



dataDir = './Data'
def initFileList():
    fileList = []
    for i in range(3, 10):
        dir = os.path.join(dataDir, 'exp_%02d' % i)
        this_name = os.listdir(dir)
        this_name = [os.path.join(dir, name) for name in this_name]
        fileList = fileList + this_name
    fileList = list(filter(lambda name: 'res' in name, fileList))
    return fileList
fileList = initFileList()
#def genFineImg

#def genCoarseImg

class CoarseModel(object):
    def __init__(self, **kwargs):
        self.batchSize = 16
        self.valList = fileList[:200]
        self.trainList = fileList[200:]
        #==============================
        self.m = Sequential()
        self.m.add(Conv2D(16,(3,3),input_shape = (640,720,3),strides = 2, padding = 'same', activation='relu',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))

        self.m.add(Conv2D(32,(3,3),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))

        self.m.add(Conv2D(64,(5,5),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))

        self.m.add(Conv2D(128,(7,7),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))

        self.m.add(Conv2D(256,(9,9),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))

        self.m.add(Flatten())
        self.m.add(Dense( 256 ))
        self.m.add(Dense(2))

        self.m.compile(optimizer='adam',loss = 'mse',metrics = ['accuracy'])
        plot_model(self.m,'CoarseModel.png',show_shapes = True,show_layer_names = True)

    def nextBatch(self, fileList):
        while True:
            batch_name = np.random.choice(fileList, self.batchSize)
            batch = {'img':np.array([]).reshape((0,640,720,3)),'label':np.array([]).reshape((0,2))}
            for idx, name in enumerate(batch_name):
                posi = name.index('_res')
                img_name = name[:posi] + '.png'
                x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
                x, y = int(x), int(y)
                img = cv2.imread(img_name)
                img = img[320: -320, :, :]
                label = np.array([x, y], dtype=np.float32)
                mask1 = (img[:, :, 0] == 245)
                mask2 = (img[:, :, 1] == 245)
                mask3 = (img[:, :, 2] == 245)
                mask = mask1 * mask2 * mask3
                img[mask] = img[x - 320 + 10, y + 14, :]
                batch['img'] = np.concatenate((batch['img'], img[np.newaxis, :, :, :]), axis=0)
                batch['label'] = np.concatenate((batch['label'], label.reshape([1, label.shape[0]])), axis=0)
            yield (batch['img'],batch['label'])
    
    def train(self):
        self.m.load_weights('CoarseModelWeights.h5')
        #self.m.fit_generator(self.nextBatch(self.trainList), epochs = 100000, steps_per_epoch = 1, verbose = 2)
        #self.m.save_weights('CoarseModelWeights.h5')
    
    def predict(self,X):
        return self.m.predict(X)

class FineModel(object):
    def __init__(self, **kwargs):
        self.batchSize = 16
        self.valList = fileList[:200]
        self.trainList = fileList[200:]
        #=========
        self.m = Sequential()
        self.m.add(Conv2D(16,(3,3),input_shape = (320,320,3),strides = 2, padding = 'same',activation = 'relu',bias_initializer = 'constant', kernel_initializer = 'truncated_normal'))

        self.m.add(Conv2D(64,(3,3),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same'))

        self.m.add(Conv2D(128,(5,5),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same'))

        self.m.add(Conv2D(256,(7,7),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same'))

        self.m.add(Conv2D(512,(9,9),padding = 'same',bias_initializer = 'constant',kernel_initializer = 'truncated_normal'))
        self.m.add(BatchNormalization())
        self.m.add(Activation('relu'))
        self.m.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same'))

        self.m.add(Flatten())
        self.m.add(Dense(512))
        self.m.add(Dense(2))

        self.m.compile(optimizer = 'adam',loss = 'mse',metrics = ['accuracy'])
        plot_model(self.m,'FineModel.png',show_shapes = True,show_layer_names = True)

    def nextBatch(self,fileList):
        while True:
            batch_name = np.random.choice(fileList, self.batchSize)
            batch = {'img':np.array([]).reshape((0,320,320,3)),'label':np.array([]).reshape((0,2))}
            for idx, name in enumerate(batch_name):
                posi = name.index('_res')
                img_name = name[:posi] + '.png'
                x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
                x, y = int(x), int(y)
                img = cv2.imread(img_name)
                mask1 = (img[:, :, 0] == 245)
                mask2 = (img[:, :, 1] == 245)
                mask3 = (img[:, :, 2] == 245)
                mask = mask1 * mask2 * mask3
                img[mask] = img[x + 10, y + 14, :]
                x_a = np.random.randint(-50, 50)
                y_a = np.random.randint(-50, 50)

                x1 = x - 160 + x_a
                x2 = x + 160 + x_a
                y1 = y - 160 + y_a
                y2 = y + 160 + y_a
                x = 160 - x_a
                y = 160 - y_a
                if y1 < 0:
                    y = 160 - y_a + y1
                    y1 = 0
                    y2 = 320
                if y2 > img.shape[1]:
                    y = 160 - y_a + y2 - img.shape[1]
                    y2 = img.shape[1]
                    y1 = y2 - 320
                img = img[x1: x2, y1: y2, :]
                label = np.array([x, y], dtype=np.float32)

                batch['img'] = np.concatenate((batch['img'], img[np.newaxis, :, :, :]), axis=0)
                batch['label'] = np.concatenate((batch['label'], label.reshape((1, label.shape[0]))), axis=0)
            yield (batch['img'],batch['label'])

    def predict(self,X):
        return self.m.predict(X)

    def train(self):
        self.m.load_weights('FineModelWeights.h5')
        #self.m.fit_generator(self.nextBatch(self.trainList),epochs = 100000, steps_per_epoch = 1, verbose = 2)
        #self.m.save_weights('FineModelWeights.h5')


if __name__=="__main__":
    m1 = CoarseModel()
    m2 = FineModel()
