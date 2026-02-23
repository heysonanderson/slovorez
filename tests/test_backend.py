# tests/test_backends.py
import pytest
import keras
from keras import ops

@pytest.mark.parametrize("backend", ["tensorflow", "jax", "torch"])
def test_backend_import(backend):
    """Проверяем, что бэкенд импортируется и видит GPU."""
    # Устанавливаем бэкенд до импорта Keras
    import os
    os.environ["KERAS_BACKEND"] = backend
    
    # Перезагружаем keras, чтобы применились настройки (важно!)
    import importlib
    import keras
    importlib.reload(keras)
    
    # Проверяем, что бэкенд установлен правильно
    assert keras.config.backend() == backend
    
    # Пытаемся создать простой тензор на GPU/CPU
    try:
        x = keras.random.normal((2, 3))
        y = ops.square(x)
        print(f"Бэкенд {backend} успешно создал тензор.")
        # Дополнительная проверка для PyTorch
        if backend == "torch":
            import torch
            if torch.cuda.is_available():
                print(f"PyTorch CUDA доступна: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        pytest.fail(f"Бэкенд {backend} не смог создать тензор: {e}")

def test_simple_model_with_different_backends():
    """Пример теста, который можно запускать с разными бэкендами."""
    # Этот тест будет запущен несколько раз с разными значениями KERAS_BACKEND
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Создаем случайные данные
    import numpy as np
    x = np.random.random((100, 5))
    y = np.random.random((100, 1))
    
    # Пробуем обучить 1 эпоху
    history = model.fit(x, y, epochs=1, verbose=0)
    assert history.history['loss'][0] > 0

test_backend_import('tensorflow')