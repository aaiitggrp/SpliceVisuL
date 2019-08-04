
# coding: utf-8
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate
from keras.models import Sequential
from keras.utils import to_categorical
from util import *
import os
from seq_logo_util import *

from collections import Counter
import h5py
import numpy as np
import shutil
import tkinter as tk
from tkinter import filedialog
from os.path import exists, join
from os import mkdir

import copy
import random
import statistics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from modest_image import ModestImage, imshow

import sys
from PIL import Image
import seaborn as sns
import time
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

version = 20
upstream = 40 #length of the upstream and downstream flanking region
max_features = (4*upstream) + 4 #total length of a sequence
K_TOP = 10
OCCLUSION_LOCAL_WINDOW_SIZE = 3
NUM_SAMPLES = 2    #7 for non-can, 15 for can (change based on canonical or non-canonical sequences being executed for visualization)
SEQ_PER_SAMPLE = 5    #10 for non-can, 5 for can (change based on canonical or non-canonical sequences being executed for visualization)
M_TOP = 5   # number of non-can sequeneces whose patterns will be shown
K_TOP_VISU = 20 # number of indices to be shown for frequency occurrence graph
SAMPLE_SIZE_VAL = 50 # number of sequences taken from both positive and negative set for generating each sample 

random.seed(0)
import glob
import os

print('Do you want to use:\n 1. Your own checkpoint file \n 2. System selected best checkpoint file')
choice=int(input())
if(choice==1):
    time.sleep(1)
    root=tk.Tk()
    root.withdraw()
    latest_file=str(filedialog.askopenfilename())
elif(choice==2):
     list_of_files = glob.glob('../../checkpoints/v20/lstm_bi_with_attention/40stream/*.hdf5') # * means all if need specific format then *.csv
     latest_file = max(list_of_files, key=os.path.getctime)
     latest_file = './lstm_bi_attention.hdf5'
model =keras.models.load_model(latest_file, custom_objects={'f_1': f_1, 'Attention': Attention})

# testing on version 26 on which it was not trained
print('Select the train data file which is used to train the model')
root=tk.Tk()
root.withdraw()
train = h5py.File(str(filedialog.askopenfilename()))
time.sleep(1)
print('Select the test data file which is not used for training')
time.sleep(1)
root=tk.Tk()
root.withdraw()
test = h5py.File(str(filedialog.askopenfilename()))

X = np.concatenate([test['true'], test['false']])
y= np.concatenate([test['Y_true'], test['Y_false']])

X_train = np.concatenate([train['true'], train['false']])
y_train = np.concatenate([train['Y_true'], train['Y_false']])

y = y.reshape(-1)
y_train = y_train.reshape(-1)

# Separate Canonical & non-Canonical

canonical_indices = [i for i in range(len(X)) if (X[i][upstream] == 3 and X[i][upstream+1] == 2 
                                                  and X[i][-upstream-2] == 1 and X[i][-upstream-1] == 3)]
canonical_indices_train = [i for i in range(len(X_train)) if (X_train[i][upstream] == 3 and X_train[i][upstream+1] == 2 
                                                  and X_train[i][-upstream-2] == 1 and X_train[i][-upstream-1] == 3)]

canonical_indices_set = set(canonical_indices)
canonical_indices_set_train = set(canonical_indices_train)

non_canonical_indices = [i for i in range(len(X)) if i not in canonical_indices_set]
non_canonical_indices_train = [i for i in range(len(X_train)) if i not in canonical_indices_set_train]

X_can, y_can = X[canonical_indices], y[canonical_indices]
X_can_train, y_can_train = X_train[canonical_indices_train], y_train[canonical_indices_train]

X_non_can, y_non_can = X[non_canonical_indices], y[non_canonical_indices]
X_non_can_train, y_non_can_train = X_train[non_canonical_indices_train], y_train[non_canonical_indices_train]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# to plot splicing motifs

def plot_average_value_per_position(X_, values, sequence,image_name):
    values = abs(values)
    sum_per_position = np.sum(values, axis = 0)
    average_per_position = sum_per_position/values.shape[0]
    average_per_position = average_per_position.tolist()
    donor_average_per_position = average_per_position[:82]
    acceptor_average_per_position = average_per_position[82:]
    donor_xticks = range(-40,42)
    acceptor_xticks = range(-42,40)
    font = {'weight' : 'bold', 'size' : 14} 

    image_name_donor = image_name[:-4] + "_avg_per_pos_donor_%s.eps" %(sequence)
    plt.plot(donor_xticks,donor_average_per_position,'b-')
    plt.xlabel('Donor junction indices', **font)
    plt.ylabel('Average deviation value', **font)
    plt.ylim((np.min(donor_average_per_position), np.max(donor_average_per_position)))
    plt.yticks((np.min(donor_average_per_position), np.max(donor_average_per_position)))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(image_name_donor, format='eps', dpi=1000)
    plt.close()

    image_name_acceptor = image_name[:-4] + "_avg_per_pos_acceptor_%s.eps" %(sequence)
    plt.plot(acceptor_xticks,acceptor_average_per_position,'b-')
    plt.xlabel('Acceptor junction indices', **font)
    plt.ylabel('Average deviation value', **font)
    plt.ylim((np.min(acceptor_average_per_position), np.max(acceptor_average_per_position)))
    plt.yticks((np.min(acceptor_average_per_position), np.max(acceptor_average_per_position)))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(image_name_acceptor, format='eps', dpi=1000)
    plt.close()

def plot_average_window_per_position_per_nucleotide(X_, values, sequences_or_window_sizes, sequence, image_name):
    print(values)
    print(sequences_or_window_sizes)
    print(sequences_or_window_sizes.shape)
    window_lengths = [1,3,5,7,9,11]
    positions = range(len(window_lengths))
    sum_per_position_per_window_size = np.zeros((values.shape[1],len(window_lengths)), dtype='float64')
    values= abs(values)

    for i,item in enumerate(window_lengths):
        for col in range(values.shape[1]):
            for row in range(values.shape[0]):
                if (sequences_or_window_sizes[row][col] == item):
                    sum_per_position_per_window_size[col][i] = sum_per_position_per_window_size[col][i] + 1

    for junction in ['acceptor', 'donor']:
        if junction == 'donor':
            sum_per_position_per_window_size_logo = sum_per_position_per_window_size[37:46,:]
            x_low = -3
            x_high = 6
        elif junction == 'acceptor':
            sum_per_position_per_window_size_logo = sum_per_position_per_window_size[110:125,:]
            x_low = -14
            x_high = 1

        fig, ax = plt.subplots(figsize=(10, len(window_lengths)))  #corresponds to the number of window sizes respectively
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        x = x_low
        maxi = 0
        for positions in sum_per_position_per_window_size_logo:
            y = 0
            for value, window_length in sorted(zip(positions, window_lengths)):
                numberAt(str(window_length), x,y, value, ax)
                y += value

            x += 1
            maxi = max(maxi, y)

        image_name_1 = image_name[:-4] + "_per_pos_per_window_%s_%s.eps" %(junction,sequence)
        font = {'weight' : 'bold', 'size' : 14} 
        plt.xlabel('%s junction indices' %(junction), **font)
        plt.ylabel('Frequency', **font)
        plt.xticks(range(x_low,x))
        plt.xlim((x_low-1,x)) 
        plt.ylim((0,maxi))
        ax.set_yticklabels(np.arange(0,9,1))
        plt.tight_layout()
        plt.savefig(image_name_1, format='eps', dpi=1000)
        plt.close(fig)

def plot_average_value_per_position_per_nucleotide(X_, values,sequence,image_name):
    bases = ['A','T','G','C']
    nucleotides = [1,2,3,4]
    positions = range(len(nucleotides))
    sum_per_position_per_nucleotide = np.zeros((values.shape[1],len(nucleotides)), dtype='float64')
    values= abs(values)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if (X_[row][col] != 5):
                sum_per_position_per_nucleotide[col][X_[row][col]-1] = sum_per_position_per_nucleotide[col][X_[row][col]-1] + values[row][col]

    sum_per_position_per_nucleotide /= values.shape[0]

    for junction in ['acceptor', 'donor']:
        if junction == 'donor':
            sum_per_position_per_nucleotide_logo = sum_per_position_per_nucleotide[37:46,:]
            x_low = -3
            x_high = 6
        elif junction == 'acceptor':
            sum_per_position_per_nucleotide_logo = sum_per_position_per_nucleotide[110:125,:]
            x_low = -14
            x_high = 1

        fig, ax = plt.subplots(figsize=(10, len(nucleotides)))  #corresponds to the number of nucleotides respectively
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
        x = x_low
        maxi = 0
        for positions in sum_per_position_per_nucleotide_logo:
            y = 0
            for value, base in sorted(zip(positions, bases)):
                alphabetAt(base, x,y, value, ax)
                y += value

            x += 1
            maxi = max(maxi, y)

        image_name_1 = image_name[:-4] + "_avg_per_pos_per_nt_%s_%s.eps" %(junction,sequence)
        font = {'weight' : 'bold', 'size' : 18} 
        plt.xlabel('%s junction indices' %(junction), **font)
        plt.ylabel('Avg. deviation value', **font)
        plt.xticks(range(x_low,x))
        plt.xticks(rotation=90, fontsize=22)
        plt.xlim((x_low-1,x)) 
        plt.ylim((0,maxi))
        ax.set_yticklabels(np.arange(0,9,1), fontsize=20)
        plt.tight_layout()
        plt.savefig(image_name_1, format='eps', dpi=1000)
        plt.close(fig)

def scan_pattern_across_sequence(X_,values,sequence,image_name,patterns):

    def compare(a, b):
        flag = 0
        for x, y in zip(a, b):
            if y == 'N':
                if (x == 'A' or x == 'T' or x == 'G' or x == 'C'):
                    flag = 1
            elif y == 'Y':
                if (x == 'C' or x == 'T'):
                    flag = 1
            elif y == 'R':
                if (x == 'A' or x == 'G'):
                    flag = 1
            elif x == y:
                flag = 1
            else:
                flag = 0
                return flag
        return flag

    for i,pattern in enumerate(patterns):
        pattern_length = len(pattern)
        pattern_value_per_column = np.zeros((values.shape[1]), dtype='float64')
        matches_count_per_column = np.zeros((values.shape[1]), dtype='int')
        for col in range(values.shape[1]-pattern_length+1):
            matches_per_column = list()
            for row in range(values.shape[0]):
                substring = ""
                if compare(substring.join(X_[row][col:col+pattern_length].tolist()), pattern):
                    matches_per_column.append(row)
                    for match in range(col,col+pattern_length):
                        pattern_value_per_column[col] += values[row][match]
            matches_count_per_column[col] = len(matches_per_column)
            if (len(matches_per_column) != 0):
                pattern_value_per_column[col] = pattern_value_per_column[col]/len(matches_per_column)

        pattern_value_per_column = pattern_value_per_column.tolist()
        donor_pattern_value_per_column = pattern_value_per_column[:82]
        acceptor_pattern_value_per_column = pattern_value_per_column[84:109]
        donor_xticks = range(-40,42)
        acceptor_xticks = range(-40,-15)
        font = {'weight' : 'bold', 'size' : 14} 

        image_name_donor = image_name[:-4] + "pattern_num_%d_donor_%s.eps" %(i,sequence)
        plt.plot(donor_xticks,donor_pattern_value_per_column,'b-')
        plt.xlabel('Donor junction indices', **font)
        plt.ylabel('Average deviation value', **font)
        plt.ylim((np.min(donor_pattern_value_per_column), np.max(donor_pattern_value_per_column)))
        plt.yticks((np.min(donor_pattern_value_per_column), np.max(donor_pattern_value_per_column)))
        plt.xticks(fontsize=14)
    	plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(image_name_donor, format='eps', dpi=1000)
        plt.close()

        image_name_acceptor = image_name[:-4] + "pattern_num_%d_acceptor_%s.eps" %(i,sequence)
        plt.plot(acceptor_xticks,acceptor_pattern_value_per_column,'b-')
        plt.xlabel('Acceptor junction indices', **font)
        plt.ylabel('Average deviation value', **font)
        plt.ylim((np.min(acceptor_pattern_value_per_column), np.max(acceptor_pattern_value_per_column)))
        plt.yticks((np.min(acceptor_pattern_value_per_column), np.max(acceptor_pattern_value_per_column)))
        plt.xticks(fontsize=14)
    	plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(image_name_acceptor, format='eps', dpi=1000)
        plt.close()

def plot_top_k(X_, y, values, sequence, image_name, file_name, top_k_image_name, nucleotide_in_each_sample, labels_in_each_sample, 
    values_in_each_sample, K_top=K_TOP, type='normal', sequences_or_window_sizes=None, mode='normal', pattern=[]):
    
    number_to_char_map = dict()
    number_to_char_map[1] = 'A'
    number_to_char_map[2] = 'T'
    number_to_char_map[3] = 'G'
    number_to_char_map[4] = 'C'

    plot_average_value_per_position(X_, values, sequence, image_name)
    plot_average_value_per_position_per_nucleotide(X_, values, sequence, image_name)
    if type == 'occlusion_global':
        plot_average_window_per_position_per_nucleotide(X_, values, sequences_or_window_sizes, sequence, image_name)
    
    X_ = np.vectorize(number_to_char_map.get)(X_)

    if (len(pattern) != 0):
        scan_pattern_across_sequence(X_,values,sequence,image_name,pattern)       

# Attention

def visualize_attention(X, y, model, root, seq_type='canonical'):
    
    root = os.path.join(root, 'attention')

    if not os.path.exists(root):
        mkdir_p(root)
        
    def visualize_per_label(label):
        print('Label = {}'.format(label))
        indices = np.where(y == label)[0]
        X_ = copy.copy(X[indices])
        
        if(len(X_) == 0):
            
            print('No sequences of label {}. Returning!'.format(label))
            return

        inp = model.input
        out = [model.layers[1].output]

        print('Predicting...')
        function = K.function([inp] + [K.learning_phase()], out)
        lstm_outs = function([X_, 1.0])
        print('Done.')

        lstm_outs = lstm_outs[0]
        lstm_outs = K.variable(lstm_outs)

        attention_weights = model.layers[2].get_weights()
        V_a = K.variable(attention_weights[0])
        W_a = K.variable(attention_weights[1])
        b_a = K.variable(attention_weights[2])

        print('Calculating attention values...')
        ej = K.squeeze(activations.tanh(K.dot(lstm_outs, W_a) + b_a), axis=-1) * V_a
        at = K.expand_dims(K.exp(ej))
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, max_features)
        at /= at_sum_repeated
        at = K.squeeze(at, axis=-1)

        at = K.eval(at)
        print('Done.')
        
        if label == 0:
            image_name = os.path.join(root, str(upstream)+'stream_attention_negative.eps')
            file_name = os.path.join(root, str(upstream)+'stream_attention_negative.txt')
            top_k_image_name = os.path.join(root, str(upstream)+'stream_attention_negative_top_k.eps')
            nucleotide_in_each_sample = os.path.join(root, 'attention_negative_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'attention_negative_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'attention_negative_values_in_each_sample')

        else:
            image_name = os.path.join(root, str(upstream)+'stream_attention_positive.eps')
            file_name = os.path.join(root, str(upstream)+'stream_attention_positive.txt')
            top_k_image_name = os.path.join(root, str(upstream)+'stream_attention_positive_top_k.eps')
            nucleotide_in_each_sample = os.path.join(root, 'attention_positive_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'attention_positive_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'attention_positive_values_in_each_sample')

        
        print('Plotting individual images...')
        plot_top_k(X_, y, at, seq_type, image_name, file_name, top_k_image_name, nucleotide_in_each_sample, labels_in_each_sample, values_in_each_sample,pattern=["CTRAY","CTNA","CAGGTAAG","GT","AG","TRA"])
        
        del at, X_, lstm_outs

        print('Done.')
        
    visualize_per_label(0)
    visualize_per_label(1)

# T-sne of last dense layer

def visualize_tsne(X, y, model, root, seq_type='canonical'):

    if not os.path.exists(root):
        mkdir_p(root)
    
    from sklearn.manifold import TSNE
    
    dense_out =  [model.layers[3].output]
    inp = model.input
    
    dense_function = K.function([inp] + [K.learning_phase()], dense_out)
    indices = list(range(len(X)))
    
    dense_outs = dense_function([X, 1.0])
    
    dense_outs = dense_outs[0]
    dense_out_transform = TSNE(n_components=2).fit_transform(dense_outs)
    
    plt.scatter(dense_out_transform[:, 0], dense_out_transform[:, 1], c=y)
    plt.xticks([])
    plt.yticks([])
    plt.title('')
    plt.savefig(os.path.join(root, 'last_layer_t-sne_projection.eps'), format='eps', dpi=1000)
    plt.close(plt.gcf())

# consensus for training sequences (canonical)

def print_dataset_consensus(X_, file_name, consensus_indices, sequence,indices_label):

    number_to_char_map = dict()
    number_to_char_map[1] = 'A'
    number_to_char_map[2] = 'T'
    number_to_char_map[3] = 'G'
    number_to_char_map[4] = 'C'

    X_ = np.vectorize(number_to_char_map.get)(X_)
    sequences_or_window_sizes = X_[:, consensus_indices]

    concensus_file_name = file_name[:-4] + '_' + sequence + '_seq_logo.txt'
    f = open(concensus_file_name,'w')
    for row in  range(sequences_or_window_sizes.shape[0]):
    	for col in range(sequences_or_window_sizes.shape[1]):
    		f.write(sequences_or_window_sizes[row][col])
    	f.write('\n')

    xticks_labels = consensus_indices+1
    mapped_xticks = mapped_indices(xticks_labels)
    nucleotides_or_window_lengths = np.array(['A','C','G','T'])

    all_scores = np.zeros((len(consensus_indices), len(nucleotides_or_window_lengths)), dtype=float)
    for i, nucleotide_or_window_length in enumerate(nucleotides_or_window_lengths):
        all_scores[:, i] = np.count_nonzero(sequences_or_window_sizes == nucleotide_or_window_length, axis=0)
    all_scores_sum = np.sum(all_scores, axis=1)[..., np.newaxis]
    all_scores /= all_scores_sum
    all_scores /= X_.shape[0]

    fig, ax = plt.subplots(figsize=(20, len(nucleotides_or_window_lengths)))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    x = 1
    maxi = 0
    for scores in all_scores:
        y = 0
        for score, base in sorted(zip(scores, nucleotides_or_window_lengths)):
            alphabetAt(base, x,y, score, ax)
            y += score

        x += 1
        maxi = max(maxi, y)
    image_name = file_name[:-4] + '_' + sequence + '_seq_logo.eps'
    font = {'weight' : 'bold'}

    plt.xlabel('Consensus indices', **font)
    plt.ylabel('Frequency', **font)
    plt.xticks(range(1,x),indices_label)
    plt.xlim((0,x))
    plt.ylim((0,maxi))
    plt.tight_layout()
    plt.savefig(image_name, format='eps', dpi=1000)
    plt.close(fig)

def visualize_consensus(model, sequence, name='lstm_bi_attention'):

    seq_type = sequence[0]
    X, y = sequence[1]

    consensus_dir = '../../visualizations/{}/{}/'.format(seq_type, name)
    root = os.path.join(consensus_dir, 'train_consensus')
        
    if not os.path.exists(root):
        mkdir_p(root)

    print('Visualizing consensus ({})'.format(seq_type))
    print('-----------------------')

    if(seq_type=='canonical'):
        consensus_indices_donor = [37,38,39,40,41,42,43,44,45]
        donor_x_low = -3
        donor_x_high = 6
        donor_indices = range(donor_x_low,donor_x_high)
        consensus_indices_acceptor = [110,111,112,113,114,115,116,117,118,119,120,121,122,123,124]
        acceptor_x_low = -14
        acceptor_x_high = 1
        acceptor_indices = range(acceptor_x_low,acceptor_x_high)

    if(seq_type=='non_canonical'):
        consensus_indices_donor = [37,38,39,40,41,42,43,44,45]
        donor_indices = consensus_indices_donor
        consensus_indices_acceptor = [110,111,112,113,114,115,116,117,118,119,120,121,122,123,124]
        acceptor_indices = consensus_indices_acceptor

    consensus_indices_donor = np.array(list(consensus_indices_donor))
    consensus_indices_acceptor = np.array(list(consensus_indices_acceptor))
    consensus_indices_donor_all = range(1,82)
    consensus_indices_acceptor_all = range(83,164)
    consensus_indices_donor_all = np.array(list(consensus_indices_donor_all))
    consensus_indices_acceptor_all = np.array(list(consensus_indices_acceptor_all))

    def visualize_per_label(label):
        indices = np.where(y == label)[0]
        X_ = copy.copy(X[indices])
        
        if(len(X_) == 0):
            print('No sequences of label {}. Returning!'.format(label))
            return

        print('Plotting...')

        for junction in ['donor', 'acceptor']:

            if label == 0:
                file_name = os.path.join(root, 'consensus_%s_negative.txt' %(junction))
                file_name_all = os.path.join(root, 'consensus_%s_negative_all.txt' %(junction))

            else:
                file_name = os.path.join(root, 'consensus_%s_positive.txt' %(junction))
                file_name_all = os.path.join(root, 'consensus_%s_positive_all.txt' %(junction))

            if junction=='donor':

                print_dataset_consensus(X_, file_name, consensus_indices_donor, seq_type,donor_indices)
                print_dataset_consensus(X_, file_name_all, consensus_indices_donor_all, seq_type,consensus_indices_donor_all)

            else:

                print_dataset_consensus(X_, file_name, consensus_indices_acceptor, seq_type,acceptor_indices)
                print_dataset_consensus(X_, file_name_all, consensus_indices_acceptor_all, seq_type,consensus_indices_donor_all)

    visualize_per_label(0)
    visualize_per_label(1)

# Occlusion

def visualize_occlusion(X, y, model, root, seq_type='canonical'):
    
    root = os.path.join(root, 'occlusion')
    if not os.path.exists(root):
        mkdir_p(root)
    
    def visualize_per_window(X_, nucleotide_or_window_length, label, type='local', all_window_predictions=None):

        if (type == 'local'):

            print('Plotting...')

            if label == 0:
                image_name = os.path.join(root, 'local_occlusion_negative_%s.eps' %(nucleotide_or_window_length))
                file_name = os.path.join(root, 'local_occlusion_negative_%s.txt' %(nucleotide_or_window_length))
                top_k_image_name = os.path.join(root, 'local_occlusion_negative_top_k_%s.eps' %(nucleotide_or_window_length))
                nucleotide_in_each_sample = os.path.join(root, 'occlusion_local_negative_nucleotide_in_each_sample')
                labels_in_each_sample = os.path.join(root, 'occlusion_local_negative_labels_in_each_sample')
                values_in_each_sample = os.path.join(root, 'occlusion_local_negative_values_in_each_sample_%s' %(nucleotide_or_window_length))
                prediction_file_name = seq_type + '_'+str(nucleotide_or_window_length)+ '_local_occlusion_predictions_negative.npy'
                absolute_prediction_file_name = seq_type + '_'+str(nucleotide_or_window_length)+ '_local_occlusion_absolute_predictions_negative.npy'
                data_file_name = seq_type + '_local_occlusion_data_negative.npy'


            else:
                image_name = os.path.join(root, 'local_occlusion_positive_%s.eps' %(nucleotide_or_window_length))
                file_name = os.path.join(root, 'local_occlusion_positive_%s.txt' %(nucleotide_or_window_length))
                top_k_image_name = os.path.join(root, 'local_occlusion_positive_top_k_%s.eps' %(nucleotide_or_window_length))
                nucleotide_in_each_sample = os.path.join(root, 'occlusion_local_positive_nucleotide_in_each_sample')
                labels_in_each_sample = os.path.join(root, 'occlusion_local_positive_labels_in_each_sample')
                values_in_each_sample = os.path.join(root, 'occlusion_local_positive_values_in_each_sample_%s' %(nucleotide_or_window_length))
                prediction_file_name = seq_type + '_'+str(nucleotide_or_window_length)+ '_local_occlusion_predictions_positive.npy'
                absolute_prediction_file_name = seq_type + '_'+str(nucleotide_or_window_length)+ '_local_occlusion_absolute_predictions_positive.npy'
                data_file_name = seq_type + '_local_occlusion_data_positive.npy'

            if os.path.exists(prediction_file_name):
                predictions = np.load(prediction_file_name)
                predictions_absolute = np.load(absolute_prediction_file_name)
                X_ = np.load(data_file_name)

            else:
            
                shift = (nucleotide_or_window_length - 1) / 2

                X_pred = np.zeros((X_.shape[0] * X_.shape[1], X_.shape[1]))
                original_predictions = model.predict(X_)
                original_predictions = original_predictions[:, label]
                original_predictions = original_predictions[..., np.newaxis]

                #forming a batch of sequences by masking each window position
                for i, x in enumerate(X_):

                    for j in range(0, X_.shape[1]):
                        
                        if j >= shift and j < (X_.shape[1]-shift):
                            cache = copy.copy(x[j-shift: j+shift+1])
                            x[j-shift: j+shift+1] = 5
                            X_pred[i * X_.shape[1] + j] = copy.copy(x)
                            x[j-shift: j+shift+1] = cache
                            
                        elif j < shift:
                            cache = copy.copy(x[: j+shift+1])
                            x[: j+shift+1] = 5
                            X_pred[i * X_.shape[1] + j] = copy.copy(x)
                            x[: j+shift+1] = cache
                        
                        else:
                            cache = copy.copy(x[j-shift:])
                            x[j-shift:] = 5
                            X_pred[i * X_.shape[1] + j] = copy.copy(x)
                            x[j-shift:] = cache                

                print('Predicting...')
                _predictions = model.predict(X_pred)
                del X_pred
                _predictions = _predictions[:, label]
                _predictions = _predictions.reshape(X_.shape[0], X_.shape[1])

                predictions = copy.copy(_predictions)
                predictions = predictions-original_predictions
                predictions_absolute = np.abs(predictions-original_predictions)

                np.save(prediction_file_name, predictions)
                np.save(absolute_prediction_file_name, predictions_absolute)
                np.save(data_file_name, X_)

                del _predictions
                print('Done')

            plot_top_k(X_, y, predictions, seq_type, image_name,  file_name, top_k_image_name, nucleotide_in_each_sample, labels_in_each_sample,
             values_in_each_sample, type='occlusion_local',pattern=["CTRAY","CTNA","CAGGTAAG","GT","AG","TRA"])

            plot_top_k(X_, y, predictions_absolute, seq_type, image_name[:-4] + "_absolute.eps",  file_name[:-4] + "_absolute.txt", top_k_image_name[:-4] + "_absolute.eps", nucleotide_in_each_sample, labels_in_each_sample,
             values_in_each_sample, type='occlusion_local',pattern=["CTRAY","CTNA","CAGGTAAG","GT","AG","TRA"]) #gives sequence logos of equal heights at all positions

        else:

            shift = (nucleotide_or_window_length - 1) / 2

            X_pred = np.zeros((X_.shape[0] * X_.shape[1], X_.shape[1]))
            original_predictions = model.predict(X_)
            original_predictions = original_predictions[:, label]
            original_predictions = original_predictions[..., np.newaxis]

            #forming a batch of sequences by masking each window position
            for i, x in enumerate(X_):

                for j in range(0, X_.shape[1]):
                    
                    if j >= shift and j < (X_.shape[1]-shift):
                        cache = copy.copy(x[j-shift: j+shift+1])
                        x[j-shift: j+shift+1] = 5
                        X_pred[i * X_.shape[1] + j] = copy.copy(x)
                        x[j-shift: j+shift+1] = cache
                        
                    elif j < shift:
                        cache = copy.copy(x[: j+shift+1])
                        x[: j+shift+1] = 5
                        X_pred[i * X_.shape[1] + j] = copy.copy(x)
                        x[: j+shift+1] = cache
                    
                    else:
                        cache = copy.copy(x[j-shift:])
                        x[j-shift:] = 5
                        X_pred[i * X_.shape[1] + j] = copy.copy(x)
                        x[j-shift:] = cache                

            print('Predicting...')
            _predictions = model.predict(X_pred)
            del X_pred
            _predictions = _predictions[:, label]
            _predictions = _predictions.reshape(X_.shape[0], X_.shape[1])

            predictions = copy.copy(_predictions)
            predictions = predictions-original_predictions
            predictions_absolute = np.abs(predictions-original_predictions)

            del _predictions
            print('Done')
            
            predictions = predictions[..., np.newaxis]
            
            if all_window_predictions is not None:
                
                all_window_predictions = np.concatenate([all_window_predictions, predictions], axis=-1)
                
            else:
                
                all_window_predictions = predictions
            
            return all_window_predictions
            
        del predictions
    
    
    def visualize_per_label(label):
            
        indices = np.where(y == label)[0]
        X_ = copy.copy(X[indices])
        
        if(len(X_) == 0):
            print('No sequences of label {}. Returning!'.format(label))
            return
        
        # global analysis            
        if label == 0:
            image_name = os.path.join(root, 'global_occlusion_negative.eps')
            file_name = os.path.join(root, 'global_occlusion_negative.txt')
            top_k_image_name = os.path.join(root, 'global_occlusion_negative_top_k.eps')
            prediction_file_name = seq_type + '_global_occlusion_best_predictions_negative.npy'
            window_sizes_file_name = seq_type + '_global_occlusion_best_window_sizes_negative.npy'
            data_file_name = seq_type + '_global_occlusion_data_negative.npy'
            nucleotide_in_each_sample = os.path.join(root, 'occlusion_global_negative_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'occlusion_global_negative_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'occlusion_global_negative_values_in_each_sample')


        else:
            image_name = os.path.join(root, 'global_occlusion_positive.eps')
            file_name = os.path.join(root, 'global_occlusion_positive.txt')
            top_k_image_name = os.path.join(root, 'global_occlusion_positive_top_k.eps')
            prediction_file_name = seq_type + '_global_occlusion_best_predictions_positive.npy'                
            window_sizes_file_name = seq_type + '_global_occlusion_best_window_sizes_positive.npy'      
            data_file_name = seq_type + '_global_occlusion_data_positive.npy'
            nucleotide_in_each_sample = os.path.join(root, 'occlusion_global_positive_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'occlusion_global_positive_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'occlusion_global_positive_values_in_each_sample')


        if os.path.exists(prediction_file_name):
            best_predictions = np.load(prediction_file_name)
            sequences_or_window_sizes = np.load(window_sizes_file_name)
            X_ = np.load(data_file_name)
            
        else:
            all_window_predictions = None
            nucleotides_or_window_lengths = [1, 3, 5, 7, 9, 11]
            for nucleotide_or_window_length in nucleotides_or_window_lengths:

                print('Evaluating window size: {}'.format(nucleotide_or_window_length))
                all_window_predictions = visualize_per_window(X_, nucleotide_or_window_length, label, 'global', all_window_predictions)

            best_predictions = np.max(all_window_predictions, axis=-1)  #axis=-1 means the last axis
            all_possible_window_sizes = np.ones(all_window_predictions.shape)
            all_possible_window_sizes *= np.array(nucleotides_or_window_lengths) # shape = (X_.shape[0], X_.shape[1], len(nucleotides_or_window_lengths))
            sequences_or_window_sizes = np.take(all_possible_window_sizes, np.argmax(all_window_predictions, axis=-1))

            np.save(prediction_file_name, best_predictions)
            np.save(window_sizes_file_name, sequences_or_window_sizes)
            np.save(data_file_name, X_)
            
        plot_top_k(X_, y, best_predictions, seq_type, image_name, file_name, top_k_image_name, nucleotide_in_each_sample, labels_in_each_sample, 
            values_in_each_sample, type='occlusion_global', sequences_or_window_sizes=sequences_or_window_sizes)
        
        # local analysis
        X_ = copy.copy(X[indices])
        visualize_per_window(X_, 1, label)
        visualize_per_window(X_, OCCLUSION_LOCAL_WINDOW_SIZE, label)
        
        print('Done.')
    
    visualize_per_label(0)
    visualize_per_label(1)

# Omission score

def visualize_omission_score(X, y, model, root, seq_type='canonical'):
    
    root = os.path.join(root, 'omission')
    if not os.path.exists(root):
        mkdir_p(root)
        
    if os.path.exists(root):
        shutil.rmtree(root)

    mkdir_p(root)

    def norm(x):
        return K.sqrt(K.sum(K.square(x), axis=-1))
    
    def cosine_distance(s, s_minus_x):
        unnorm_cosine = K.batch_dot(s, s_minus_x, axes=[1, 2])
        norm_s = norm(s)
        norm_s = K.repeat_elements(K.expand_dims(norm_s), 
                                   unnorm_cosine.get_shape().as_list()[1] , axis=1)
        norm_s_minus_x = norm(s_minus_x)
        
        normalization = norm_s * norm_s_minus_x
        cosine_distance = unnorm_cosine / normalization
        return cosine_distance
    
    def omission_score(s, s_minus_x):
        return 1 - cosine_distance(s, s_minus_x)
    
    def visualize_per_label(label):

    	indices = np.where(y == label)[0]
        X_ = copy.copy(X[indices])
        
        if(len(X_) == 0):
            print('No sequences of label {}. Returning!'.format(label))
            return

        if label == 0:
            image_name = os.path.join(root, 'omission_negative.eps')
            file_name = os.path.join(root, 'omission_negative.txt')
            top_k_image_name = os.path.join(root, 'omission_negative_top_k.eps')
            nucleotide_in_each_sample = os.path.join(root, 'omission_negative_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'omission_negative_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'omission_negative_values_in_each_sample')
            omission_scores_file_name = seq_type + '_omission_scores_negative.npy'

        else:
            image_name = os.path.join(root, 'omission_positive.eps')
            file_name = os.path.join(root, 'omission_positive.txt')
            top_k_image_name = os.path.join(root, 'omission_positive_top_k.eps')
            nucleotide_in_each_sample = os.path.join(root, 'omission_positive_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'omission_positive_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'omission_positive_values_in_each_sample')
            omission_scores_file_name = seq_type + '_omission_scores_positive.npy'

        if os.path.exists(omission_scores_file_name):
            omission_scores = np.load(omission_scores_file_name)

        else:
            predictions = np.zeros(X_.shape)
            
            inp = model.input
            out = [model.layers[-3].output]

            print('Predicting original...')
            function = K.function([inp] + [K.learning_phase()], out)
            s = function([X_, 1.0])
            s = K.variable(s[0])
            s_minus_x = None
                        
            print('Predicting removed...')
            for i, x in enumerate(X_):
                X_pred = np.zeros((X_.shape[1], X_.shape[1]))
                
                for j in range(0, X_.shape[1]):

                    cache = copy.copy(x[j])
                    x[j] = 5
                    X_pred[j] = copy.copy(x)
                    x[j] = cache
                    
                prediction = function([X_pred, 1.0])
                prediction = prediction[0]
                prediction = K.reshape(prediction, (1, X_.shape[1], -1))
                
                if s_minus_x is not None:
                    
                    s_minus_x = Concatenate(axis=0)([s_minus_x, prediction])
                    
                else:
                    
                    s_minus_x = prediction
            
            omission_scores = omission_score(s, s_minus_x)
            omission_scores = K.eval(omission_scores)
            np.save(omission_scores_file_name, omission_scores)            

            print('Done')

            
        plot_top_k(X_, y, omission_scores, seq_type, image_name, file_name, top_k_image_name, nucleotide_in_each_sample, labels_in_each_sample, 
            values_in_each_sample, K_top=10,pattern=["CTRAY","CTNA","CAGGTAAG","GT","AG","TRA"])
            

        print('Done.')
    
    visualize_per_label(0)
    visualize_per_label(1)

# Smooth Gradient

def visualizeImageGrayscale(image_3d, percentile=99):
    if len(image_3d.shape)==3:
        image_2d = np.sum(np.abs(image_3d), axis=0)
    else:
        image_2d = image_3d
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def plot_image(X, image, sequence, nucleotide_in_each_sample, values_in_each_sample, name):
    name = name[:-4] + '_' + sequence + '.png'
    m, n = X.shape
    image = visualizeImageGrayscale(image)
    
    vocab = {1:'A', 2:'T', 3:'G', 4:'C', 5:'N'}
    X_ = np.vectorize(vocab.get)(X)
    axis_labels = 1 + np.array(list(range(n)))
    
    np.save(nucleotide_in_each_sample, X_)
    np.save(values_in_each_sample, image)

    
def getGrad(X_in, label, model):
    input_ = model.input
    emb = model.layers[0].output
    out = model.layers[-1].output
    y = out[:,label]
    
    func = K.function([input_, K.learning_phase()], [out])
    soft_score = func([X_in, 1.0])[0] # (m, 2)
    score = soft_score[:, label]
    
    func = K.function([input_, K.learning_phase()], [emb])
    X_embed = func([X_in, 1.0])[0]
    
    gradient = K.gradients(y, emb)
    fn = K.function([y, emb, K.learning_phase()], gradient)
    
    grad = fn([score, X_embed, 1.0])[0] #(m, 164, 4)
    grad = grad*grad
    grad = np.transpose(grad, (0, 2, 1))
    
    return grad #(m, 4, 164)  

def getSmoothGrad(X_in, label, model, nsamples=50, std_dev=.15):
    m, n = X_in.shape
    input_ = model.input
    emb = model.layers[0].output
    out = model.layers[-1].output
    y = out[:,label]
    
    func = K.function([input_, K.learning_phase()], [out])
    soft_score = func([X_in, 1.0])[0] # (m, 2)
    score = soft_score[:, label]
    
    func = K.function([input_, K.learning_phase()], [emb])
    gradients = K.gradients(y, emb)
    fn = K.function([y, emb, K.learning_phase()], gradients)
    
    total_gradients = np.zeros(shape=(m, 164, 4), dtype='float32')
    
    weights_ = model.layers[0].get_weights()[0]
    weights = copy.copy(weights_)
    w = copy.copy(weights_[1:5,:])
    std_dev = std_dev * (np.max(w) - np.min(w))
    
    for i in range(nsamples):
        noise = np.random.normal(0, std_dev, w.shape)
        w_noise = w + noise
        weights[1:5,:] = w_noise
        model.layers[0].set_weights([weights])
        
        X_embed_noise = func([X_in, 1.0])[0] #(m, 164, 4)
        
        grad = fn([score, X_embed_noise, 1.0])[0] #(m, 164, 4)
        
        total_gradients += (grad*grad)
    
    total_gradients = total_gradients / nsamples
    total_gradients = np.transpose(total_gradients, (0, 2, 1))
    model.layers[0].set_weights([weights_])
    return total_gradients # (m, 4, 164)

def visualize_smoothgrad(X, y, model, root, nsamples=50, std_dev=0.15, seq_type='canonical'):
    
    root = os.path.join(root, 'smoothgrad')
    if os.path.exists(root):
        shutil.rmtree(root)

    os.mkdir(root)
    
    def visualize_per_label(label):

    	indices = np.where(y == label)[0]
        X_ = copy.copy(X[indices])
        
        if(len(X_) == 0):
            print('No sequences of label {}. Returning!'.format(label))
            return
        
        if label == 0:
            image_name_grad = os.path.join(root, 'gradient_negative.eps')
            image_name_smoothgrad = os.path.join(root, 'smoothgrad_negative.eps')
            file_name = os.path.join(root, 'smoothgrad_negative.txt')
            top_k_image_name = os.path.join(root, 'smoothgrad_negative_top_k.eps')
            nucleotide_in_each_sample = os.path.join(root, 'smoothgrad_negative_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'smoothgrad_negative_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'smoothgrad_negative_values_in_each_sample')
            grad_file_name = seq_type + '_grad_negative.npy'
            smoothgrad_file_name = seq_type + '_smoothgrad_negative.npy'

        else:
            image_name_grad = os.path.join(root, 'gradient_positive.eps')
            image_name_smoothgrad = os.path.join(root, 'smoothgrad_positive.eps')
            file_name = os.path.join(root, 'smoothgrad_positive.txt')
            top_k_image_name = os.path.join(root, 'smoothgrad_positive_top_k.eps')
            nucleotide_in_each_sample = os.path.join(root, 'smoothgrad_positive_nucleotide_in_each_sample')
            labels_in_each_sample = os.path.join(root, 'smoothgrad_positive_labels_in_each_sample')
            values_in_each_sample = os.path.join(root, 'smoothgrad_positive_values_in_each_sample')
            grad_file_name = seq_type + '_grad_positive.npy'
            smoothgrad_file_name = seq_type + '_smoothgrad_positive.npy'

        if os.path.exists(grad_file_name):
            grad_reduced = np.load(grad_file_name)
            smoothgrad_reduced = np.load(smoothgrad_file_name)

        else:   
            m, n = X_.shape
            
            print('Calculating...')
            
            grad = getGrad(X_, label, model) # (m, 4, 164)
            smoothgrad = getSmoothGrad(X_, label, model, nsamples=nsamples, std_dev=0.15) # (m, 4, 164)
            
            grad_reduced = np.amax(grad, axis=1) #(m, 164)
            smoothgrad_reduced = np.amax(smoothgrad, axis=1) #(m, 164)

            np.save(grad_file_name, grad_reduced)
            np.save(smoothgrad_file_name, smoothgrad_reduced)
            
            print('Done.')
 
        print('Plotting...')
        
        plot_image(X_, grad_reduced, seq_type, nucleotide_in_each_sample, 
            values_in_each_sample, name=image_name_grad)
        plot_image(X_, smoothgrad_reduced, seq_type, nucleotide_in_each_sample, 
            values_in_each_sample, name=image_name_smoothgrad)
        plot_top_k(X_, y, smoothgrad_reduced, seq_type, image_name_smoothgrad, file_name, top_k_image_name, nucleotide_in_each_sample, labels_in_each_sample, 
    values_in_each_sample, mode='smoothgrad',pattern=["CTRAY","CTNA","CAGGTAAG","GT","AG","TRA"])
        
        print('Done.')
    
    visualize_per_label(0)
    visualize_per_label(1) 

def getIntegratedGrads(X_in, label, model, steps=50):
    m, n = X_in.shape
    input_ = model.input
    emb = model.layers[0].output
    out = model.layers[-1].output
    y = out[:,label]
    
    func = K.function([input_, K.learning_phase()], [out])
    soft_score = func([X_in, 1.0])[0] # (m, 2)
    score = soft_score[:, label]
    
    func = K.function([input_, K.learning_phase()], [emb])
    gradients = K.gradients(y, emb)
    fn = K.function([y, emb, K.learning_phase()], gradients)
    
    total_gradients = np.zeros(shape=(m, 164, 4), dtype='float32')
    
    weights_ = model.layers[0].get_weights()[0]
    weights = copy.copy(weights_)
    w = copy.copy(weights_[1:5,:])
    m_w, n_w = w.shape
    baseline = np.zeros(shape=(m_w, n_w), dtype='float32')
    
    for i in range(steps):
    	print(i)
    	scaled_inputs = baseline + (float(i)/steps)*(w-baseline)
        weights[1:5,:] = scaled_inputs
        model.layers[0].set_weights([weights])
        
        X_scaled_input = func([X_in, 1.0])[0] #(m, 164, 4)
        
        grad = fn([score, X_scaled_input, 1.0])[0] #(m, 164, 4)
        
        total_gradients += (grad*grad)
    
    total_gradients = total_gradients / steps
    total_gradients = (X_scaled_input)*total_gradients  # shape: (m, 164, 4)
    total_gradients = np.transpose(total_gradients, (0, 2, 1))
    total_gradients = np.amax(total_gradients, axis=1) #(m, 164)
    model.layers[0].set_weights([weights_])
  	
    return total_gradients # (m, 164) 

def visualize_integrated_grads(X, y, model, root, steps=50, seq_type='canonical'):

    root = os.path.join(root, 'integrated_gradients')
    if os.path.exists(root):
        shutil.rmtree(root)

    os.mkdir(root)

    def visualize_per_label(label):

	    indices = np.where(y == label)[0]
	    X_ = copy.copy(X[indices])
	    
	    if(len(X_) == 0):
	        print('No sequences of label {}. Returning!'.format(label))
	        return
	    
	    if label == 0:
	        image_name_intgrad = os.path.join(root, 'intgrad_negative.eps')
	        file_name = os.path.join(root, 'intgrad_negative.txt')
	        top_k_image_name = os.path.join(root, 'intgrad_negative_top_k.eps')
	        nucleotide_in_each_sample = os.path.join(root, 'intgrad_negative_nucleotide_in_each_sample')
	        labels_in_each_sample = os.path.join(root, 'intgrad_negative_labels_in_each_sample')
	        values_in_each_sample = os.path.join(root, 'intgrad_negative_values_in_each_sample')
	        intgrad_file_name = seq_type + '_intgrad_negative.npy'

	    else:
	        image_name_intgrad = os.path.join(root, 'intgrad_positive.eps')
	        file_name = os.path.join(root, 'intgrad_positive.txt')
	        top_k_image_name = os.path.join(root, 'intgrad_positive_top_k.eps')
	        nucleotide_in_each_sample = os.path.join(root, 'intgrad_positive_nucleotide_in_each_sample')
	        labels_in_each_sample = os.path.join(root, 'intgrad_positive_labels_in_each_sample')
	        values_in_each_sample = os.path.join(root, 'intgrad_positive_values_in_each_sample')
	        intgrad_file_name = seq_type + '_intgrad_positive.npy'

	    if os.path.exists(intgrad_file_name):
	        intgrad = np.load(intgrad_file_name)

	    else:   
	        print('Calculating...')

	        intgrad = getIntegratedGrads(X_, label, model, steps=steps) # (m, 164)  

	        print(intgrad)
	        np.save(intgrad_file_name, intgrad)
	        
	        print('Done.')

	    print('Plotting...')

	    plot_top_k(X_, y, intgrad, seq_type, image_name_intgrad, file_name, top_k_image_name, nucleotide_in_each_sample, labels_in_each_sample, 
	values_in_each_sample, mode='integrated_gradients',pattern=["CTRAY","CTNA","CAGGTAAG","GT","AG","TRA"])
	    
	    print('Done.')
    
    visualize_per_label(0)
    visualize_per_label(1)   

# various visualizations

def visualize(model, sequence, num_samples, name, attention=True):
    seq_type = sequence[0]
    X, y = sequence[1]
    
    model_root = '../../visualizations/{}/{}/'.format(seq_type, name)

    if not os.path.exists(model_root):
        mkdir_p(model_root)

    if not os.path.exists(model_root + 'attention'):
        mkdir(model_root + 'attention')
        mkdir(model_root + 'occlusion')
        mkdir(model_root + 'omission')
        mkdir(model_root + 'smoothgrad')

    print(X.shape)
    print(y.shape)

    X_sampled = X
    y_sampled = y

    if attention:

        print('{}. Visualizing attention weights ({})'.format(i, seq_type))
        print('--------------------------------')
        visualize_attention(X_sampled, y_sampled, model=model, root=model_root, seq_type=seq_type)

    # # print('{}. Visualizing t-sne ({})'.format(i, seq_type))
    # # print('--------------------')
    # # visualize_tsne(X_sampled, y_sampled, model=model, root=model_root,
    # #                seq_type=seq_type)

    print('{}. Visualizing occlusion ({})'.format(i, seq_type))
    print('-----------------------')
    visualize_occlusion(X_sampled, y_sampled, model=model, root=model_root,
                          seq_type=seq_type)
    
    print('{}. Visualizing omission scores ({})'.format(i, seq_type))
    print('-----------------------')
    visualize_omission_score(X_sampled, y_sampled, model=model, root=model_root,
                           seq_type=seq_type)

    print('{}. Visualizing SmoothGrad ({})'.format(i, seq_type))
    print('-----------------------')
    visualize_smoothgrad(X_sampled, y_sampled, model=model, root=model_root,
                            seq_type=seq_type)

    print('{}. Visualizing Integrated Gradients ({})'.format(i, seq_type))
    print('-----------------------')
    visualize_integrated_grads(X_sampled, y_sampled, model=model, root=model_root,
                            seq_type=seq_type)



sequences = [                                       # execute one sequence type at a time due to resource constraints
    ('canonical', (X_can, y_can)),        
    ('non_canonical', (X_non_can, y_non_can))
]

sequences_train = [
    ('canonical', (X_can_train, y_can_train)),
    ('non_canonical', (X_non_can_train, y_non_can_train))
]

for sequence in sequences:
    visualize(model, sequence, num_samples=NUM_SAMPLES, name='lstm_bi_attention')

# for sequence in sequences_train:
#     visualize_consensus(model, sequence, name='lstm_bi_attention') # to generate the consensus for canonical sequences



