# SpliceVisuL: Visualization of Bidirectional Long Short-term Memory Networks for Splice Junction Prediction
Authors: Aparajita Dutta, Aman Dalmia, Athul R, Kusum Kumari Singh, Ashish Anand

This is the SpliceVisuL tool with code and data for visualization of bidirectional long short-term memory networks.

# Requirements

The code requires the following libraries to run:
1. tensorflow and all its dependencies (with GPU support). Please check https://www.tensorflow.org/install/install_linux
   to check all the source dependencies.
2. python 2.7 and libraries keras, numpy, h5py, seaborn, matplotlib

# Organization

The folders are organized into the following subdirectories:

- `data` - Contains true and false data.
- `checkpoints` - Checkpoints to store the model weights
- `programs` - Contains code for all the operations like - data preprocessing, training & testing various RNN models, visualization of splicing features.

	1. In directory programs/preprocessing:
	- For creating training and validation dataset: python prepare_data_train_val.py
	- For creating test dataset: python prepare_data_test.py

	2. In directory programs/model:
	- Run the model: python lstm_bi_with_attention.py (Various other RNN models are also provided)

	3. In directory programs/testing:
	- For testing any model: python test_lstm.py (change appropriate model path in the file)

	4. In directory programs/visualization:
	- For visualizing splicing motifs: python visualizations.py (Move the model weights file from 'checkpoints' folder. A sample file 'lstm_bi_attention.hdf5' provided.)
	
- `visualizations` - All files related to visualization saved here.

There are readme files within the subdirectories with further explanations of the file organization.


# Execution steps

1. Our dataset can be downloaded from: http://www.iitg.ac.in/anand.ashish/resources.html by clicking on the link 'Download splice junction dataset' provided.
Extract the downloaded data 'Splice_junction_data.tgz' in the root directory

2. Run the script main.py and choose from the given options to perform various tasks.




