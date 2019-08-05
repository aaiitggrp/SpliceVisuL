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
		- For testing any model: python test_lstm.py 

	4. In directory programs/visualization:
		- For visualizing splicing motifs: python visualizations.py (A sample model weights file 'lstm_bi_attention.hdf5' is provided.)
	
- `visualizations` - All files related to visualization saved here.

There are readme files within the subdirectories with further explanations of the file organization.


# Execution steps

1. Our dataset can be downloaded from: http://www.iitg.ac.in/anand.ashish/resources.html by clicking on the link 'Download splice junction dataset' provided.
Extract the downloaded data 'Splice_junction_data.tgz' in the root directory

2. Run the script main.py and choose from the given options to perform various tasks.
- Options:
		- Preprocess the training data by mapping nucleotides into integers and partitioning the training data into 90% train and 10% validation data. Train and validation data saved in hierarchial data format.
		- Train the proposed model with data generated in step 1.
		- Preprocess the test data by mapping nucleotides into integers and save in hierarchial data format.
		- Test the performance of the model trained in step 2 on test data generated in step 3. Results saved in performance.txt
		- Produces various visualization results on the test data. User can choose from a set of visualization techniques provided. User can also choose any model-weights file from 'checkpoint' directory. System selected best checkpoint file will choose the last model-weights file.
		
	




