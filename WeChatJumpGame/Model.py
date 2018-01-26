from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D, Reshape, Flatten
#from keras.utils import plot_model
import numpy as np
import os
import cv2
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config = config)
KTF.set_session(session)

dataDir = './Data'

def initFileList():
    fileList = []
    for i in range(3, 10):
        dir = os.path.join(dataDir, 'exp_%02d' % i)
        this_name = os.listdir(dir)
        this_name = [os.path.join(dir, name) for name in this_name]
        fileList = fileList + this_name
    name_list_raw = fileList
    def _name_checker(name):
        posi = name.index('_res')
        img_name = name[:posi] + '.png'
        if img_name in name_list_raw:
            return True
        else:
            return False

    fileList = list(filter(lambda name: 'res' in name, fileList))
    fileList = list(filter(_name_checker, fileList))
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
        #plot_model(self.m,'CoarseModel.png',show_shapes = True,show_layer_names = True)

    def nextBatch(self, fileList):
        while True:
            batch_name = np.random.choice(fileList, self.batchSize)
            batch = {}
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
                if idx == 0:
                    batch['img'] = img[np.newaxis, :, :, :]
                    batch['label'] = label.reshape([1, label.shape[0]])
                else:
                    img_tmp = img[np.newaxis, :, :, :]
                    label_tmp = label.reshape((1, label.shape[0]))
                    batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
                    batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)
            yield (batch['img'],batch['label'])
    
    def train(self):
        #self.m.load_weights('CoarseModelWeights.h5')
        self.m.fit_generator(self.nextBatch(self.trainList), epochs = 100000, steps_per_epoch = 1, verbose = 2)
        self.m.save_weights('CoarseModelWeights.h5')

    def evaluate(self):
        print(self.m.metrics_names)
        print( self.m.evaluate_generator(self.nextBatch(self.valList), steps = 1000) )
    
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
        #plot_model(self.m,'FineModel.png',show_shapes = True,show_layer_names = True)

    def nextBatch(self,fileList):
        while True:
            batch_name = np.random.choice(fileList, self.batchSize)
            batch = {}
            for idx, name in enumerate(batch_name):
                posi = name.index('_res')
                img_name = name[:posi] + '.png'
                x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
                x, y = int(x), int(y)
                img = cv2.imread(img_name)
                # img = img[320: -320, :, :]
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

                if idx == 0:
                    batch['img'] = img[np.newaxis, :, :, :]
                    batch['label'] = label.reshape([1, label.shape[0]])
                else:
                    img_tmp = img[np.newaxis, :, :, :]
                    label_tmp = label.reshape((1, label.shape[0]))
                    batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
                    batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)
            yield (batch['img'],batch['label'])

    def predict(self,X):
        return self.m.predict(X)

    def train(self):
        #self.m.load_weights('FineModelWeights.h5')
        self.m.fit_generator(self.nextBatch(self.trainList),epochs = 100000, steps_per_epoch = 1, verbose = 2)
        self.m.save_weights('FineModelWeights.h5')

    def evaluate(self):
        print(self.m.metrics_names)
        print(self.m.evaluate_generator(self.nextBatch(self.valList), steps = 1000))


if __name__=="__main__":
    m1 = CoarseModel()
    m2 = FineModel()
    try:
        m1.train()
        #m1.evaluate()
    except KeyboardInterrupt:
        m1.m.save('CurrputCoarse.h5')

    try:
        m2.train()
        #m2.evaluate()
    except KeyboardInterrupt:
        m2.m.save('CurrputFine.h5')
