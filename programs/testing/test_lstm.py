import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

from keras.models import load_model
from keras.utils import to_categorical
from util import *
import h5py
import numpy as np
import argparse
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import glob
import os
import tkinter as tk
from tkinter import filedialog
import time
from tqdm import tqdm
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


print("Select the trained MODEL file")
time.sleep(1)
root=tk.Tk()
root.withdraw()
ft=open("performance.txt",'a')
ft.write('\n'+"########################################"+'\n')
ft.write("{}".format(str(datetime.date.today()))+'\n')

model = load_model(str(filedialog.askopenfilename()), custom_objects={'f_1': f_1, 'Attention': Attention})


# testing on version on which it was not trained
print("Select the test file (in .h5 format) for predictions")
time.sleep(2)
root=tk.Tk()
root.withdraw()
test = h5py.File(str(filedialog.askopenfilename())) # put appropriate test file name here
####################################################

X = np.concatenate([test['true'], test['false']])
y= np.concatenate([test['Y_true'], test['Y_false']])
print("\t\t\n\n_____PREDICTING_______\n")
y_pred = model.predict(X,verbose=1)
y_pred = np.argmax(y_pred, axis=1)
y = y.reshape(-1)

mean = np.mean(y == y_pred)
ft.write('Accuracy:'+ str(sk.metrics.accuracy_score(y, y_pred))+'\n')
ft.write('Precision:'+ str(sk.metrics.precision_score(y, y_pred))+'\n')
ft.write('Recall:' + str(sk.metrics.recall_score(y, y_pred))+'\n')
ft.write('f1_score:' + str(sk.metrics.f1_score(y, y_pred))+'\n')
ft.write('confusion_matrix:'+'\n')
ft.write(str(sk.metrics.confusion_matrix(y, y_pred))+'\n')
fpr, tpr, thresholds = sk.metrics.roc_curve(y, y_pred)
ft.write(str(sk.metrics.roc_auc_score(y, y_pred))+'\n')
ft.close

print('Accuracy:'+ str(sk.metrics.accuracy_score(y, y_pred)) + '\n')
print('Precision:'+str(sk.metrics.precision_score(y, y_pred))+'\n')
print('Recall:' +str(sk.metrics.recall_score(y, y_pred))+'\n')
print('f1_score:'+str(sk.metrics.f1_score(y, y_pred))+'\n')
print('\n\n----Results are stored in Performance.txt----\n\n')

