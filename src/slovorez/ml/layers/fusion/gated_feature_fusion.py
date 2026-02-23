import keras
import keras.ops as ops
from keras.src import layers as KL
from keras.src.backend import KerasTensor

@keras.saving.register_keras_serializable()
class GatedFeatureFusion(KL.Layer):
    def __init__(self,
                 gate_activation='sigmoid', 
                 name="gated_feature_fusion", 
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.gate_activation = keras.activations.get(gate_activation)
        self.gate_dense = None
        self.transform_dense = None

    def build(self, inputs_shape):
        vector_shape, sequence_shape = inputs_shape
        seq_dim = sequence_shape[-1]

        self.gate_dense = KL.Dense(seq_dim, activation=self.gate_activation)
        self.gate_dense.build(vector_shape)

        self.transform_dense = KL.Dense(seq_dim,activation='tanh')
        self.transform_dense.build(vector_shape)

        super().build(inputs_shape)


    def call(self, inputs):
        vector, sequence = inputs

        gate = self.gate_dense(vector)
        gate = ops.expand_dims(gate, axis=1)

        transformation = self.transform_dense(vector)
        transformation = ops.expand_dims(transformation, axis=1)

        return sequence + gate * transformation
    
    def compute_output_spec(self, inputs):
        vector, sequence = inputs
        return KerasTensor(sequence.shape, dtype=self.compute_dtype)
        