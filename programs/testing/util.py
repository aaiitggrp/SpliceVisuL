from keras import backend as K
from keras.engine.topology import Layer
from keras import constraints, regularizers, initializers, activations


class Attention(Layer):

    def __init__(self, 
                 timesteps,
                 return_probabilities=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        
        self.timesteps = timesteps
        self.return_probabilities = return_probabilities
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        self.V_a = self.add_weight(shape=(input_shape[-2],),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.W_a = self.add_weight(shape=(input_shape[-1], 1),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        self.b_a = self.add_weight(shape=(1,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        super(Attention, self).build(input_shape)  


    def call(self, x):
        ej = K.squeeze(activations.tanh(K.dot(x, self.W_a) + self.b_a), axis=-1) * self.V_a
        at = K.expand_dims(K.exp(ej))
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated

        if(self.return_probabilities):
            return at
        
        out = K.batch_dot(at, x, axes=1)
        return out
    
    
    def compute_output_shape(self, input_shape):
        if self.return_probabilities:
            return (input_shape[0], input_shape[1])
        
        return (input_shape[0], input_shape[-1])


    def get_config(self):
        config = {
            'timesteps': self.timesteps,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def f_1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))