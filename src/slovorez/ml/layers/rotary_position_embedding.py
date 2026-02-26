import keras
import keras.ops as ops
from keras.src.backend import KerasTensor

@keras.saving.register_keras_serializable()
class RotaryPositionEmbedding(keras.layers.Layer):
    def __init__(self, dim, max_seq_len=48, name="rotary_position_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        
    def build(self, input_shape):
        dtype = self.compute_dtype
        position = ops.arange(0, self.max_seq_len, dtype=dtype)
        position = ops.reshape(position, [-1, 1])
        
        dim_range = ops.arange(0, self.dim, 2, dtype=dtype)
        dim_range = ops.reshape(dim_range, [1, -1])
        
        angle_rates = 1.0 / (10000 ** (dim_range / self.dim))
        angle_rads = position * angle_rates

        self.sin_cached = ops.sin(angle_rads)  # shape: [max_seq_len, dim//2]
        self.cos_cached = ops.cos(angle_rads)  # shape: [max_seq_len, dim//2]
        
        self.built = True
        
    def call(self, x):
        seq_len = ops.shape(x)[1]

        begin = ops.array([0, 0])
        size = ops.array([seq_len, self.dim // 2])

        sin = ops.slice(self.sin_cached, begin, size)
        cos = ops.slice(self.cos_cached, begin, size)

        x1, x2 = ops.split(x, 2, axis=-1)
        
        sin = ops.expand_dims(sin, 0)  # [1, seq_len, dim//2]
        cos = ops.expand_dims(cos, 0)  # [1, seq_len, dim//2]
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        return ops.concatenate([rotated_x1, rotated_x2], axis=-1)
    
    def compute_output_spec(self, inputs):
        return KerasTensor(inputs.shape, dtype=self.compute_dtype)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_seq_len": self.max_seq_len,
        })
        return config
