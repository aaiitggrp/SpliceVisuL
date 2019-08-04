import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

import os
import time
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
work_dir= os.getcwd()
print(work_dir)
def exe():
     print("\n\n\t\t\t====ENTER YOUR CHOICE====\t\t\n")
     print("\t\t1. -> Do you want to preprocess your data ?\t\t")
     print("\t\t2. -> Do you want to train a preprocessed data ?\t\t")
     print("\t\t3. -> Do you want to prepare a test data (.h5 format)  ?\t\t")
     print("\t\t4. -> Do you want to check performance of trained model on test data ? \t\t")
     print("\t\t5. -> Do you want to visualize your data ?\t\t")
     print("\t\t6. -> Do you want to test your data ?\t\t")
     print("\t\t7. -> Do you want to exit ?\t\t\n\t\t")

exe()
ch=int(input())

if (ch==1):
     print('\t------PREPARING TRAINING AND VALIDATION DATA---------\t\n\n\n')
     time.sleep(1)
     exec(open('programs/preprocessing/prepare_data_train_val.py').read())  
if (ch==2):
     os.chdir('programs/model')
     exec(open('lstm_bi_with_attention.py').read())
if(ch==4):
     os.chdir('programs/testing')
     exec(open('test_lstm.py').read())
if(ch==3):
     os.chdir('programs/preprocessing')
     exec(open('prepare_data_test.py').read())
if(ch==5):
     os.chdir('programs/visualization')
     exec(open('seq_logo_util.py').read())
     exec(open('visualizations.py').read())
if(ch==6):
     choice=int(input("1. Test on single sequence \n2. Test on a file containing sequences\n"))
     if(choice==1):
          
          import numpy as np
          sample=input("Enter your sample\n")
          sample=list(sample)
          mapping = {'A':1,'T':2,'G':3,'C':4,'N':5}
          encsample=[mapping[c] for c in sample]
          encsample=np.array(encsample).reshape(1,-1)
          print('\n\t\tSelect the model to predict your sample')
          exec(open('util.py').read())
          root=tk.Tk()
          root.withdraw()
          model=load_model(str(filedialog.askopenfilename()), custom_objects={'Attention': Attention,'f_1':f_1})
          import warnings
          warnings.filterwarnings("ignore",category=DeprecationWarning)
          os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

          print('\n\t\t------------PREDICTING------------')
          time.sleep(1.5)
          pre=model.predict(encsample)
          pre=np.round(pre)
          if(pre[0][0]==1):
               print('\n'+'FALSE')
               time.sleep(1)
          else:
               print('\n'+'TRUE')
               time.sleep(1)

     elif(choice==2):
          def load_doc(filename):
               file = open(filename, 'r')
               text = file.read()
               file.close()
               return text
          print('Select the file containing sequences')
          root=tk.Tk()
          root.withdraw()
          in_filename = str(filedialog.askopenfilename())
          raw_text = load_doc(in_filename)
          lines = raw_text.split('\n')
          lines.pop()
          mapping = {'A':1,'T':2,'G':3,'C':4,'N':5}
          import numpy as np
          sequences = list()
          for line in lines:
               encoded_seq = [mapping[char] for char in line]
               sequences.append(encoded_seq)
          encsample=np.array(sequences)
          time.sleep(1)
          print('\n\t\tSelect the model to predict your sequence')
          exec(open('util.py').read())
          time.sleep(1.5)
          root=tk.Tk()
          root.withdraw()
          model=load_model(str(filedialog.askopenfilename()), custom_objects={'Attention': Attention,'f_1':f_1})
          import warnings
          warnings.filterwarnings("ignore",category=DeprecationWarning)
          os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

          print('\n\t\t------------PREDICTING------------')
          time.sleep(1.5)
          pre=model.predict(encsample)
          pre=np.round(pre)
          for i in range(len(lines)):
               if(pre[i][0]==1):
                    lines[i]=lines[i]+' '+'\"FALSE\"'+'\n'
               else:
                    lines[i]=lines[i]+' '+'\"TRUE\"'+'\n'

          fresult=open('Results.txt','w')
          fresult.write('##########################################'+'\n')
          for i in range(len(lines)):
               fresult.write(lines[i])
          fresult.close()
          print('\n===RESULTS ARE STORED IN \"Results.txt\"=====')
          print(os.getcwd())

     

if(ch!=6 and ch!=7):
     os.chdir("../..")
     exec(open('main.py').read())
elif(ch==6):
     exec(open('main.py').read())
elif(ch==7):
     exit()

###############################################################
          
       
     