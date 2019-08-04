import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import keras
from keras.preprocessing import sequence
import h5py
from sklearn.model_selection import train_test_split
import argparse
import time
import datetime
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('upstream', type=int, const=40, nargs='?', default=40)

args = parser.parse_args()
up_down_stream = args.upstream

version = 20
chop_at = 2*up_down_stream + 2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #warings suppressor used

work_dir= os.getcwd()

print("Select the directory for saving data\n")
time.sleep(1.5)
root=tk.Tk()
root.withdraw()
data_dir= filedialog.askdirectory()
os.chdir(data_dir)

import tkinter as tk
from tkinter import filedialog
root=tk.Tk()
root.withdraw()
print("Select train positive file.txt\n")
time.sleep(2)
trpospath=filedialog.askopenfilename()
time.sleep(1)
print("Select train negative file.txt\n")
time.sleep(2)
trnegpath=filedialog.askopenfilename()
#####################################################

false_train_file = trnegpath
true_train_file = trpospath

false = open(false_train_file)
false_introns = false.readlines()

false_introns = [false_intron.strip() for false_intron in false_introns]

false_introns = [ intron[:chop_at] + intron[-chop_at:] for intron in (false_introns)]

mapped_introns = []
dn ={}
dn['A'] = 1
dn['T'] = 2
dn['G'] = 3
dn['C'] = 4
dn['N'] = 5
print('--------Processing Train Positive Data-----------\n')

# tqdm used below for progress bar display
for intron in tqdm(false_introns):   
    arr = np.zeros((len(intron)), dtype='uint8');
    for i,ch in enumerate(intron) :
        arr[i] = dn[ch];
    mapped_introns.append(arr)
#####################################################

maxlen = max(intron.shape[0] for intron in mapped_introns)
#print("Maxlen = %d" % maxlen)

mapped_introns = np.asarray(mapped_introns)
false_train, false_val = train_test_split(mapped_introns, test_size=0.1, random_state=0)

y_false_train = np.zeros((false_train.shape[0],1), dtype= np.uint8)
y_false_val = np.zeros((false_val.shape[0],1), dtype= np.uint8)

train = h5py.File('train_v{}_{}_{}.h5'.format(version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")), 'w')
train.create_dataset('false', data=false_train)
train.create_dataset('Y_false', data=y_false_train)
train.close()

val = h5py.File('val_v{}_{}_{}.h5'.format(version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")), 'w')
val.create_dataset('false', data=false_val)
val.create_dataset('Y_false', data=y_false_val)
val.close()
##################################################

del false, false_val, false_train, y_false_train, y_false_val

true_train = open(true_train_file, 'r')
true_train_introns = true_train.readlines()

true_train_introns = [true_train_intron.strip() for true_train_intron in true_train_introns]
true_train_introns = [intron[:chop_at] + intron[-chop_at:] for intron in (true_train_introns)]

mapped_introns = []
dn ={}
dn['A'] = 1
dn['T'] = 2
dn['G'] = 3
dn['C'] = 4
dn['N'] = 5
print('--------Processing Train Negative Data-----------\n')

# tqdm used for progress bar display
for intron in tqdm(true_train_introns): 
    arr = np.zeros((len(intron)), dtype='uint8');
    for i,ch in enumerate(intron) :
        arr[i] = dn[ch];
    mapped_introns.append(arr)
####################################################

maxlen = max(intron.shape[0] for intron in mapped_introns)

mapped_introns = np.asarray(mapped_introns)

true_train, true_val = train_test_split(mapped_introns, test_size=0.1, random_state=0)

y_true_train = np.ones((true_train.shape[0],1), dtype= np.uint8)
y_true_val = np.ones((true_val.shape[0],1), dtype= np.uint8)


print('\n'+'\n'+'File save format :::: name_version_up/downstream_month_day_hour'+'\n')
train = h5py.File('train_v{}_{}_{}.h5'.format(version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")), 'a')
print('\ntrain File saved as train_v{}_{}_{}.h5'.format(version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")))
train.create_dataset('true', data=true_train)
train.create_dataset('Y_true', data=y_true_train)

train.close()

val = h5py.File('val_v{}_{}_{}.h5'.format(version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")), 'a')
print('\nVal File saved as val_v{}_{}_{}.h5'.format(version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")))
val.create_dataset('true', data=true_val)
val.create_dataset('Y_true', data=y_true_val)
val.close()
