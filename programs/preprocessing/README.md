# Preprocessing

- `prepare_data_train_val.py` - Preprocess the training data by mapping nucleotides into integers and partitioning the training data into 90% train and 10% validation data. Train and validation data saved in hierarchial data format.

- `prepare_data_test.py` - Preprocess the test data by mapping nucleotides into integers and save in hierarchial data format.

# Running the code

To run the scripts for different values of `up_down_stream`, simply pass the value as the argument.
For running the experiment with `up_down_stream=30` (default is `40`):

$ python *.py 30



