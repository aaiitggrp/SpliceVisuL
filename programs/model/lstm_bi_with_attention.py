import warnings
warnings.simplefilter(action='ignore')

import keras
import numpy as np 
import scipy
import h5py
from keras.layers import Embedding, recurrent, Activation, Dense, Flatten
import os
from keras import backend as K
from keras import metrics
from keras.callbacks import *
from keras.preprocessing import sequence
from keras.models import Sequential,load_model
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout
from keras.backend.tensorflow_backend import set_session
from util import *
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tqdm import tqdm


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

import datetime
import shutil
import argparse
import time
#IMPORT DATASET


parser = argparse.ArgumentParser()
parser.add_argument('upstream', type=int, const=40, nargs='?', default=40)

args = parser.parse_args()
up_down_stream = args.upstream

version = '20'
date = datetime.datetime.now()
month = str((date.month))
day = str(date.day)
hour=str(date.hour)

work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)
print("Select the directory containing training and validation data  for training the model")
time.sleep(1)
root=tk.Tk()
root.withdraw()
data_dir= filedialog.askdirectory()

print('Select the training file (.h5 format)')
time.sleep(1)
root=tk.Tk()
root.withdraw()
file_train = filedialog.askopenfilename()

print('Select the validation file (.h5 format)')
time.sleep(1)
root=tk.Tk()
root.withdraw()
file_val = filedialog.askopenfilename()

print("Select the directory to save the trained model")
time.sleep(1)
root=tk.Tk()
root.withdraw()
save_dir= filedialog.askdirectory()
file_path = '../../checkpoints/v%s/lstm_bi_with_attention/%sstream/' % (version,up_down_stream)
log_path = os.path.join(file_path, 'logs')
#print(up_down_stream)

#######################################################
for path in [save_dir, file_path, log_path]:
    if not os.path.exists(path):
      #shutil.rmtree(path)
      os.makedirs(path)

    #os.mkdir(path)

########################################################################################################################################
def one_hot(_input):
	yy = np.zeros((_input.shape[0], _input.shape[1]+1), dtype=np.uint8)
	for idx in range(_input.shape[0]):
		yy[idx][_input[idx][0]]=1
	return yy

#######################################################################################################################################
check_pt = ModelCheckpoint(file_path+'%s_%s_chkpts.{epoch:02d}-{val_loss:.2f}.hdf5' %(day,hour), monitor='val_acc', verbose=0, save_weights_only=False, save_best_only=True, mode='max', period=1)
early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=2, verbose=0, mode='auto')
t_board = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False)

call_backs = [ early_stop, t_board, check_pt]
#######################################################################################################################################

os.chdir(data_dir)

print("\t>>> Loading Train data")

with h5py.File(file_train, 'r') as F:
  X_train = np.concatenate((F['true'],F['false']), axis=0)
  Y_train = np.concatenate((F['Y_true'],F['Y_false']), axis=0).reshape(-1,1)

print("\t>>> Train data Loaded !!!")

print("\t>>> Loading Validation data")

with  h5py.File(file_val, 'r') as F:
  X_val = np.concatenate((F['true'],F['false']), axis=0)
  Y_val = np.concatenate((F['Y_true'],F['Y_false']), axis=0).reshape(-1,1)

print("\t>>> Validation data Loaded !!!")

#######################################################################################################################################
os.chdir(work_dir)
#print(os.getcwd())
# X_train = np.concatenate((X_true,X_false, X_pos, X_neg), axis= 0)
# Y_train = np.concatenate((Y_true,Y_false, Y_pos, Y_neg), axis=0)

np.random.seed(None)

# get the initial state of the RNG
state = np.random.get_state()

np.random.shuffle(X_train)

np.random.set_state(state)

np.random.shuffle(Y_train)
#######################################################################################################################################
batch_size = 128
max_features = X_train.shape[1]
#######################################################################################################################################
model = Sequential()
model.add(Embedding(max_features, 4, input_shape=[max_features]))
model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dropout(0.5))
model.add(Attention(max_features))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# try using different optimizers and different optimizer configs
print(model.summary())
#######################################################################################################################################

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',f_1])
  

print('\n'+'Train...')
model.fit(X_train, one_hot(Y_train),batch_size=128,epochs=50,
          validation_data=[X_val, one_hot(Y_val)],
          callbacks = call_backs,verbose=1)

os.chdir(save_dir)
print ('\n>>> SAVING MODEL AND WEIGHTS')  
model.save('%s_%s_lstm_bi_with_attention_%d.h5' % (day,hour,up_down_stream))
print('\nmodel is saved as {}_{}_lstm_bi_with_attention_{}.h5 in the directory specified >>{}\n '.format(day,hour,up_down_stream,save_dir))
os.chdir(work_dir)
####################################################


