# Model

Code for the BLSTM and various other RNN architectures.

## Relevant file descriptions

- `lstm_base.py` - The vanilla LSTM network.
- `lstm_base_attention.py` - The vanilla LSTM network with attention applied on the hidden layer outputs.
- `lstm_bi.py` - The Bidirectional LSTM network.
- `lstm_bi_with_attention.py` - Bidirectional LSTM with attention applied on the hidden layer outputs. 

## Running the code

To run a particular model for different values of `up_down_stream`, simply pass the value as an argument.
For running the experiment with `up_down_stream=30` (default is `40`):

$ python *.py 30
