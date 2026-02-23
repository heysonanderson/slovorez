import keras 
from keras import ops
from keras.src.backend import KerasTensor


class CRF(keras.layers.Layer):

    def __init__(self, units=None, regularizer=None, chain_initializer="orthogonal",name="crf", **kwargs):
        super().__init__(name=name, **kwargs)
        self.chain_initializer = keras.initializers.get(chain_initializer)
        self.regularizer = regularizer
        self.transitions = None
        self.supports_masking = True
        self.mask = None
        self.accuracy_fn = keras.metrics.Accuracy()
        self.units = units
        if units is not None:
            self.dense = keras.layers.Dense(units)

    def build(self, input_shape):
        assert len(input_shape) == 3
        if self.units:
            units = self.units
        else:
            units = input_shape[-1]
        self.transitions = self.add_weight(
            name="transition",
            shape=[units, units],
            initializer=self.chain_initializer,
            regularizer=self.regularizer
        )
    
    def call(self, inputs, mask=None, training=False):
        if mask is None:
            input_shape = ops.slice(ops.shape(inputs), [0], [2])
            mask = ops.ones(input_shape)
        sequence_lengths = ops.sum(ops.cast(mask, 'int32'), axis=-1)

        if self.units:
            inputs = self.dense(inputs)


    def _crf_decode(self, inputs, seqence_lengths):
        pass