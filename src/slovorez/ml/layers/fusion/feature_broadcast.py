import keras
import keras.ops as ops
from keras.src import layers as l
from keras.src.backend import KerasTensor

@keras.saving.register_keras_serializable()
class FeatureBroadcast(l.Layer):
    def __init__(self, name="repeat_vector_broadcast", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        vector, sequence = inputs
        return sequence * ops.reshape(vector, [-1, 1, vector.shape[-1]])
    
    def compute_output_spec(self, inputs):
        vector, sequence = inputs

        def safe_dim(dim):
            return dim if dim is not None else None
        
        batch_size = safe_dim(vector.shape[0])
        seq_len = safe_dim(sequence.shape[1])
        vector_dim = safe_dim(vector.shape[-1])
        
        return KerasTensor((batch_size, seq_len, vector_dim), dtype=self.compute_dtype)
        