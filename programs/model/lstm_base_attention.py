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

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

import datetime
import argparse
#IMPORT DATASET


parser = argparse.ArgumentParser()
parser.add_argument('--upstream', type=int, const=40, nargs='?', default=40)
args = parser.parse_args()
up_down_stream = args.upstream

version = '20'
date = datetime.datetime.now()
month = str.lower(date.strftime('%b'))
day = str(date.day)

work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

data_dir= '../../data/version_20_26_intronic_type2'
save_dir = './save_lstm_base_with_attention_intronic_type2_v%s/' % version
file_path = '../../checkpoints/v%s/lstm_base_with_attention/' % version
log_path = os.path.join(file_path, 'logs')
print(up_down_stream)
file_train = 'train_v%s_%d.h5' % (version, up_down_stream)
file_val = 'val_v%s_%d.h5' % (version, up_down_stream)

for path in [save_dir, file_path, log_path]:
    if not os.path.exists(path):
        os.mkdir(path)

######################################################################################################################################## Written By Athul R Sept 2017
def one_hot(_input):
	yy = np.zeros((_input.shape[0], _input.shape[1]+1), dtype=np.uint8)
	for idx in xrange(_input.shape[0]):
		yy[idx][_input[idx][0]]=1
	return yy

#######################################################################################################################################
check_pt = ModelCheckpoint(file_path+'chkpts.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=0, save_weights_only=True, mode='auto', period=1)
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
print(os.getcwd())
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
model.add(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(Attention(max_features))
# model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# try using different optimizers and different optimizer configs
print(model.summary())
#######################################################################################################################################
# model.load_weights('model_wghts_lstm_1.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f_1])
  

print('Train...')
model.fit(X_train, one_hot(Y_train),
          batch_size=batch_size,
          epochs=10,
          validation_data=[X_val, one_hot(Y_val)],
          callbacks = call_backs)

os.chdir(save_dir)
print ('\n>>> SAVING MODEL AND WEIGHTS')  
model.save('lstm_base_2layer_with_attention_intronic_type2_%s_%s_%d_final.h5' % (month, day, up_down_stream))
os.chdir(work_dir)
