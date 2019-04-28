from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf 
from keras import backend as K

import time
import sys

_mel_scale = 96
_time_len = 1366
_channels = 1


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.epoch_time_end = time.time()
        self.times.append(self.epoch_time_end - self.epoch_time_start)
        logs['Timer'] = self.epoch_time_end - self.epoch_time_start
        
class AUC_callback_TF(Callback):
    def __init__(self, y_true, SDG, batchSize):
        self.y_true = y_true
        self.SDG = SDG
        self.step = (len(y_true) // batchSize)
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        
        Skip_Tag = []
        print("[AUC] Start predict..." ,end = '  ')
        y_pred = self.model.predict_generator(generator = self.SDG,
                                         steps = self.step,
                                        workers = 12,
                                        use_multiprocessing=True,) # (#data x 50()) 
        y_pred_out = y_pred.copy()
        print("[OK]")
    #print(y_pred.shape,self.y.shape,len(self.SDG.text_filenames))
        print("[AUC] Start calcute auc...", end = '  ')
        for i in range(len(y_pred)):
            for j in range(len(y_pred[i])):
                if y_pred[i][j] >= 0.5 : 
                    y_pred[i][j] = 1
                else:
                    y_pred[i][j] = 0
        try:
            auc, update_op = tf.metrics.auc(self.y_true ,K.round(y_pred))
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                sess.run(tf.local_variables_initializer())
                AUC_this_epoch = sess.run([auc, update_op])[0]
        except Exception as e:
            AUC_this_epoch = 0
            print(e)
        print("[Total OK!]")
        
        
        logs['AUC_test'] = AUC_this_epoch
        #logs['AUC_List'] = AUC_List
        print('AUC_test: %s '% (str(round(AUC_this_epoch,5))))
        
        
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
        
class AUC_Evalu(Callback):
    def __init__(self, y_true, SDG, batchsize):
        self.SDG = SDG
        self.step = (len(y_true) // batchsize)
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        
        test_loss, test_auc = self.model.evaluate_generator(self.SDG, self.step, workers=12, use_multiprocessing=True)
        
        
        logs['AUC_test'] = test_auc
        logs['test_loss']= test_loss
        #logs['AUC_List'] = AUC_List
        print('test_loss: %s - AUC_test: %s  '% (str(round(test_loss,5)),str(round(test_auc,5))))
        
        
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


    
from keras.utils import Sequence
import json
import codecs
class AudioDataGenerator(Sequence):
    #batch size can only be 1
    def __init__(self, text_filenames, labels, batch_size, A_Collection):
        self.text_filenames, self.labels = text_filenames, labels
        self.batch_size = batch_size
        self.step = len(text_filenames)//batch_size
        self.audio_collection = A_Collection

    def __len__(self):
        return int(np.ceil(len(self.text_filenames)/ (self.batch_size)))

    def __getitem__(self, idx):
        #Here, you have to imprement what the data looks like in each epcho
        filename_List =  self.text_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        #Text_batch_x = [] 
        Audio_batch_x = []
        for filename in filename_List:  
            Audio_x = np.array(self.audio_collection.find({"$text": {"$search":filename}})[0] \
                               ['Spectrogram']).reshape((_mel_scale,_time_len,_channels))
            Audio_batch_x.append(Audio_x)
        y_List = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return np.array(Audio_batch_x), np.array(y_List)
    def getitem(self, idx):
        #Here, you have to imprement what the data looks like in each epcho
        
        return self.__getitem__(idx)


class LyricDataGenerator(Sequence):
    #batch size can only be 1
    def __init__(self, text_filenames, labels, batch_size, L_Collection):
        self.text_filenames, self.labels = text_filenames, labels
        self.batch_size = batch_size
        self.step = len(text_filenames)//batch_size
        self.lyric_collection = L_Collection
        
    def __len__(self):
        return int(np.ceil(len(self.text_filenames)/ (self.batch_size)))

    def __getitem__(self, idx):
        #Here, you have to imprement what the data looks like in each epcho
        filename_List =  self.text_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        #print(filename_List)
        Text_batch_x = [] 
        for filename in filename_List:  
            Text_x = np.array(self.lyric_collection.find({"$text": {"$search":filename}})[0]['LineMatrix'])
            #Text_x = Text_x.reshape(_num_Lines,_num_LineLen, _num_WEDim,1) #(num_sentence, 20 word ,100dim WE, 1for cnn)
            Text_batch_x.append(Text_x)
            
        y_List = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return np.array(Text_batch_x),np.array(y_List)
    def getitem(self, idx):
        #Here, you have to imprement what the data looks like in each epcho
        
        return self.__getitem__(idx)


class SongDataGenerator(Sequence):
    #batch size can only be 1
    def __init__(self, text_filenames, labels, batch_size, A_Collection, L_Collection):
        self.text_filenames, self.labels = text_filenames, labels
        self.batch_size = batch_size
        self.step = len(text_filenames)//batch_size
        self.audio_collection = A_Collection
        self.lyric_colleciont = L_Collection

    def __len__(self):
        return int(np.ceil(len(self.text_filenames)/ (self.batch_size)))

    def __getitem__(self, idx):
        #Here, you have to imprement what the data looks like in each epcho
        filename_List =  self.text_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        #Text_batch_x = [] 
        Audio_batch_x = []
        for filename in filename_List:  
            Audio_x = np.array(self.audio_collection.find({"$text": {"$search":filename}})[0]\
                               ['Spectrogram']).reshape((_mel_scale,_time_len,_channels))
            Audio_batch_x.append(Audio_x)
        y_List = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        Text_batch_x = [] 
        for filename in filename_List:  
            Text_x = np.array(self.lyric_colleciont.find({"$text": {"$search":filename}})[0]['LineMatrix'])
            #Text_x = Text_x.reshape(_num_Lines,_num_LineLen, _num_WEDim,1) #(num_sentence, 20 word ,100dim WE, 1for cnn)
            Text_batch_x.append(Text_x)
            
        
        return [np.array(Audio_batch_x),np.array(Text_batch_x)], np.array(y_List)
    def getitem(self, idx):
        #Here, you have to imprement what the data looks like in each epcho
        
        return self.__getitem__(idx)

class BestAUC_callback_TF(Callback):
    def __init__(self, y_true, SDG, batchSize):
        self.y_true = y_true
        self.SDG = SDG
        self.step = (len(y_true) // batchSize)
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        start = time. time()
    ##predict part
        Skip_Tag = []
        print("[AUC] Start predict..." ,end = '  ')
        y_pred = self.model.predict_generator(generator = self.SDG,
                                         steps = self.step,
                                        workers = 12,
                                        use_multiprocessing=True,) # (#data x 50()) 
        print("[OK]")
    ## AUC_test, threshold = 0.5
        y_pred_out = y_pred.copy()
        for i in range(len(y_pred_out)):
            for j in range(len(y_pred_out[i])):
                if y_pred_out[i][j] >= 0.5 : 
                    y_pred_out[i][j] = 1
                else:
                    y_pred_out[i][j] = 0
        try:
            Graph_AUC = tf.Graph()
            with Graph_AUC.as_default():
                y_true_holder = tf.placeholder(dtype=tf.float32)
                y_pred_holder = tf.placeholder(dtype=tf.float32)
                auc, update_op = tf.metrics.auc(labels=y_true_holder, predictions=y_pred_holder)
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)).as_default() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    sess.run(update_op, feed_dict={y_true_holder: self.y_true, y_pred_holder: y_pred_out})
                    AUC_this_epoch = sess.run(auc, feed_dict={y_true_holder: self.y_true, y_pred_holder: y_pred_out}).item()
            Graph_AUC.finalize()
            del Graph_AUC
        except Exception as e:
            AUC_this_epoch = 0
            print(e)
        logs['AUC_test'] = AUC_this_epoch
        print("[Total OK!]")
        print('AUC_test: %s '% (str(round(AUC_this_epoch,5))))
        ## Best_AUC
        AUC_List = [0 for i in range(50)]
        Threshold_List = [0 for i in range(50)]
        y_true_T = self.y_true.T
        print("[AUC] Best AUC calcute..." )
        for j in range(5,0,-1):
            threshold = round(j*0.1,2)
            #print("threshold now : {}".format(threshold))
            
            y_pred_out = y_pred.copy()
            y_pred_T = y_pred_out.T
            for i in range(len(y_pred_T)):
                for j in range(len(y_pred_T[i])):
                    if y_pred_T[i][j] >= threshold : 
                        y_pred_T[i][j] = 1
                    else:
                        y_pred_T[i][j] = 0
            for i in range(50):
                try:
                    Graph_AUC = tf.Graph()
                    with Graph_AUC.as_default():
                        y_true_holder = tf.placeholder(dtype=tf.float32)
                        y_pred_holder = tf.placeholder(dtype=tf.float32)
                        auc, update_op = tf.metrics.auc(labels=y_true_holder, predictions=y_pred_holder)
                        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)).as_default() as sess:
                            sess.run(tf.global_variables_initializer())
                            sess.run(tf.local_variables_initializer())
                            sess.run(update_op, {y_true_holder: y_true_T[i], y_pred_holder: y_pred_T[i]})
                            AUC_this = sess.run(auc, {y_true_holder: y_true_T[i], y_pred_holder: y_pred_T[i]}).item()
                            if AUC_this >= AUC_List[i]:
                                AUC_List[i] = AUC_this
                                Threshold_List[i] = threshold
                    Graph_AUC.finalize()
                    del Graph_AUC
                except Exception as e:
                    AUC_List[i]=0
                    print(e)
        logs['AUC_List'] = AUC_List
        logs['AUC_Threshold'] = Threshold_List
        print("[Tag OK!]")   
        AUC_Best = np.mean(AUC_List)
        logs['AUC_Best'] = AUC_Best
        #print('AUC_test: %s '% (str(round(AUC_this_epoch,5))))
        print('AUC_Best: %s '% (str(round(AUC_Best,5))))
        end =  time. time()
        print('Best AUC Timer : %s '% (str(round(end-start,5))))

        
        
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    


    
Data_Generator = {
    'Audio': AudioDataGenerator,
    'Lyric': LyricDataGenerator,
    'Both': SongDataGenerator,
}