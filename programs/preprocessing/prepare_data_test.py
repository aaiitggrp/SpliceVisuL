import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import scipy 
import os
from scipy.misc import *
import tkinter as tk
from tkinter import filedialog
import keras
from keras.preprocessing import sequence
import h5py
import argparse
import time
import datetime
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('upstream', type=int, const=40, nargs='?', default=40)
args = parser.parse_args()
up_down_stream = args.upstream
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #tensorflow warnings suppressor used

work_dir = os.getcwd()
test_version = 26


print("\nSelect the directory to save the test file\n")
time.sleep(1)
root=tk.Tk()
root.withdraw()
data_dir= filedialog.askdirectory()
os.chdir(data_dir)
test = h5py.File('test_v{}_{}_{}.h5'.format(test_version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")) , 'w')
#####################################################

chop_at = 2 * up_down_stream + 2

def convert_data(file_name):
    with open(file_name) as  f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [ con[:chop_at] + con[-chop_at:] for con in (content)]
    
    values = []
    dictionary={}
    dictionary['A'] = 1
    dictionary['T'] = 2
    dictionary['G'] = 3
    dictionary['C'] = 4
    dictionary['N'] = 5
    for introns in tqdm(content): #tqdm is used for progres bar
        mapped_values = np.zeros((len(introns)), dtype='uint8');
        for i,ch in enumerate(introns):
            mapped_values[i] = dictionary[ch];
        values.append(mapped_values)
    
    del content[:]
    maxlen = max(intron.shape[0] for intron in values)
    #print("Maxlen = %d" % maxlen)
    
    values= np.asarray(values)
    np.random.RandomState(0)
    np.random.shuffle(values)
    return values

print("\nSelect the test positive file to prepare")
time.sleep(1)
test_true = convert_data(file_name=filedialog.askopenfilename())
print("\nSelect the test negative file to prepare\n")
time.sleep(1)
test_false = convert_data(file_name=filedialog.askopenfilename())

y_false = np.zeros((test_false.shape[0], 1), dtype=np.uint8)
y_true = np.ones((test_true.shape[0], 1), dtype=np.uint8)

print('\n'+'\n'+'File save format :::: name_version_up/downstream_month_day_hour'+'\n')
test.create_dataset('true', data=test_true)
test.create_dataset('false', data=test_false)
test.create_dataset('Y_true', data=y_true)
test.create_dataset('Y_false', data=y_false)
test.close()
print('\nTest file saved as test_v{}_{}_{}.h5 '.format(test_version,up_down_stream,datetime.datetime.now().strftime("%m_%d_%H")))
