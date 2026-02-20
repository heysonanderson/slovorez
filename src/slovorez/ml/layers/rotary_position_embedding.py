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
        
        # Кэшируем sin и cos для МАКСИМАЛЬНОЙ длины
        self.sin_cached = ops.sin(angle_rads)  # shape: [max_seq_len, dim//2]
        self.cos_cached = ops.cos(angle_rads)  # shape: [max_seq_len, dim//2]
        
        self.built = True
        
    def call(self, x):
        seq_len = ops.shape(x)[1]
        
        # ВАЖНО: Используем статический срез через tf.slice для XLA совместимости
        # Создаем индексы для среза
        begin = ops.array([0, 0])
        size = ops.array([seq_len, self.dim // 2])
        
        # XLA-совместимый способ получения среза
        sin = ops.slice(self.sin_cached, begin, size)
        cos = ops.slice(self.cos_cached, begin, size)
        
        # Применяем поворотное кодирование
        x1, x2 = ops.split(x, 2, axis=-1)
        
        # Убедимся, что размеры совпадают
        # sin/cos: [seq_len, dim//2] -> нужно расширить до [batch, seq_len, dim//2]
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


# def demonstrate_rotation_math():
#     """Демонстрация математики поворота"""
#     print("=== МАТЕМАТИКА ПОВОРОТА ===")
    
#     # Простой пример поворота 2D вектора
#     angle = ops.array(45.0 * 3.14159 / 180.0)  # 45 градусов в радианах
#     cos_a = ops.cos(angle)
#     sin_a = ops.sin(angle)
    
#     # Исходный вектор
#     vector = ops.array([1.0, 0.0])
#     print(f"Исходный вектор: {vector}")
    
#     # Поворот по формуле RoPE
#     x1, x2 = vector[0], vector[1]
#     rotated_x1 = x1 * cos_a - x2 * sin_a
#     rotated_x2 = x1 * sin_a + x2 * cos_a
#     rotated = ops.array([rotated_x1, rotated_x2])
    
#     print(f"Угол поворота: 45°")
#     print(f"cos(45°): {cos_a:.3f}, sin(45°): {sin_a:.3f}")
#     print(f"Повернутый вектор: {rotated}")
#     print(f"Норма сохранилась: {ops.norm(vector):.3f} == {ops.norm(rotated):.3f}")
#     print()


# def test_basic_rotations():
#     """Базовые примеры поворотов"""
#     print("=== БАЗОВЫЕ ПОВОРОТЫ ===")
    
#     rope = RotaryPositionEmbedding(dim=4, max_seq_len=3)
    
#     # Пример 1: Единичные векторы
#     print("Пример 1: Единичные векторы вдоль осей")
#     x1 = ops.array([[
#         [1.0, 0.0, 1.0, 0.0],  # позиция 0: (1,0) и (1,0)
#         [1.0, 0.0, 1.0, 0.0],  # позиция 1: те же векторы
#         [1.0, 0.0, 1.0, 0.0],  # позиция 2: те же векторы
#     ]])
    
#     output1 = rope(x1)
#     print("Вход (все позиции одинаковые):")
#     print(x1[0])
#     print("Выход (разные позиции):")
#     print(output1[0])
#     print()
    
#     # Пример 2: Вращение на 90 градусов
#     print("Пример 2: Векторы под 45 градусов")
#     x2 = ops.array([[
#         [0.707, 0.707, 0.0, 1.0],   # позиция 0
#         [0.707, 0.707, 0.0, 1.0],   # позиция 1  
#         [0.707, 0.707, 0.0, 1.0],   # позиция 2
#     ]])
    
#     output2 = rope(x2)
#     print("Вход:")
#     print(x2[0])
#     print("Выход:")
#     print(output2[0])
#     print()


# def test_rotation_patterns():
#     """Различные паттерны поворотов"""
#     print("=== РАЗЛИЧНЫЕ ПАТТЕРНЫ ПОВОРОТОВ ===")
    
#     rope = RotaryPositionEmbedding(dim=6, max_seq_len=4)
    
#     # Паттерн 1: Чередующиеся координаты
#     print("Паттерн 1: Чередующиеся координаты")
#     pattern1 = ops.array([[
#         [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # (1,0), (0,1), (1,0)
#         [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
#         [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
#         [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
#     ]])
    
#     out1 = rope(pattern1)
#     print("Вход:")
#     for i in range(4):
#         print(f"Поз {i}: {[f'{x:.3f}' for x in pattern1[0, i]]}")
#     print("Выход:")
#     for i in range(4):
#         print(f"Поз {i}: {[f'{x:.3f}' for x in out1[0, i]]}")
#     print()
    
#     # Паттерн 2: Единичный круг
#     print("Паттерн 2: Точки на единичной окружности")
#     angles = [0, 45, 90, 135]  # градусы
#     pattern2 = []
#     for angle in angles:
#         rad = angle * 3.14159 / 180.0
#         # Три пары координат на окружности
#         point = [
#             ops.cos(rad).numpy(), ops.sin(rad).numpy(),  # первая пара
#             0.0, 1.0,                                   # вторая пара  
#             -ops.sin(rad).numpy(), ops.cos(rad).numpy()  # третья пара
#         ]
#         pattern2.append(point)
    
#     pattern2 = ops.array([pattern2])
#     out2 = rope(pattern2)
    
#     print("Вход (точки на окружности):")
#     for i, angle in enumerate(angles):
#         print(f"Поз {i} ({angle}°): {[f'{x:.3f}' for x in pattern2[0, i]]}")
#     print("Выход:")
#     for i, angle in enumerate(angles):
#         print(f"Поз {i} ({angle}°): {[f'{x:.3f}' for x in out2[0, i]]}")
#     print()


# def test_norm_preservation():
#     """Тест сохранения нормы"""
#     print("=== СОХРАНЕНИЕ НОРМЫ ===")
    
#     rope = RotaryPositionEmbedding(dim=8, max_seq_len=5)
    
#     # Случайные векторы
#     x = keras.random.normal((2, 5, 8))
#     output = rope(x)
    
#     # Вычисляем нормы
#     norm_before = ops.sqrt(ops.sum(x * x, axis=-1))
#     norm_after = ops.sqrt(ops.sum(output * output, axis=-1))
    
#     print("Нормы до RoPE (первые 3 последовательности):")
#     print(norm_before[0, :3])
#     print("Нормы после RoPE (первые 3 последовательности):")
#     print(norm_after[0, :3])
#     print(f"Нормы сохранились: {ops.isclose(norm_before, norm_after, atol=1e-6)}")
#     print()


# def test_relative_positions():
#     """Тест относительных позиций"""
#     print("=== ОТНОСИТЕЛЬНЫЕ ПОЗИЦИИ ===")
    
#     rope = RotaryPositionEmbedding(dim=4, max_seq_len=5)
    
#     # Один и тот же вектор в разных позициях
#     x = ops.array([[
#         [1.0, 0.0, 0.0, 1.0],  # позиция 0
#         [1.0, 0.0, 0.0, 1.0],  # позиция 1
#         [1.0, 0.0, 0.0, 1.0],  # позиция 2
#         [1.0, 0.0, 0.0, 1.0],  # позиция 3
#         [1.0, 0.0, 0.0, 1.0],  # позиция 4
#     ]])
    
#     output = rope(x)
    
#     print("Разности между соседними позициями:")
#     for i in range(4):
#         diff = output[0, i+1] - output[0, i]
#         diff_norm = ops.norm(diff)
#         print(f"Поз {i+1} - Поз {i}: норма разности = {diff_norm:.4f}")
    
#     print("Это показывает постоянство относительных позиций")
#     print()


# def visualize_rotation_2d():
#     """Визуализация поворота в 2D"""
#     print("=== 2D ВИЗУАЛИЗАЦИЯ ПОВОРОТА ===")
    
#     # Создаем слой для 2D векторов (одна пара)
#     rope_2d = RotaryPositionEmbedding(dim=2, max_seq_len=4)
    
#     # Вектор, который будем вращать
#     vector = ops.array([[
#         [1.0, 0.0],  # исходное положение
#         [1.0, 0.0],  # позиция 1
#         [1.0, 0.0],  # позиция 2  
#         [1.0, 0.0],  # позиция 3
#     ]])
    
#     rotated = rope_2d(vector)
    
#     print("Вращение вектора [1, 0] по позициям:")
#     for i in range(4):
#         x, y = rotated[0, i, 0], rotated[0, i, 1]
#         angle_deg = ops.arctan2(y, x) * 180 / 3.14159
#         print(f"Позиция {i}: [{x:.3f}, {y:.3f}] ~ {angle_deg:.1f}°")
    
#     print("\nВидно как вектор поворачивается на разные углы в зависимости от позиции!")
#     print()


# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_rotation_2d_plot():
#     """Визуализация поворота в 2D с графиком"""
#     print("=== 2D ВИЗУАЛИЗАЦИЯ ПОВОРОТА С ГРАФИКОМ ===")
    
#     # Создаем слой для 2D векторов (одна пара)
#     rope_2d = RotaryPositionEmbedding(dim=2, max_seq_len=12)
    
#     # Вектор, который будем вращать
#     vector = ops.array([[
#         [1.0, 0.0],  # позиция 0
#         [1.0, 0.0],  # позиция 1
#         [1.0, 0.0],  # позиция 2  
#         [1.0, 0.0],  # позиция 3
#         [1.0, 0.0],  # позиция 4
#         [1.0, 0.0],  # позиция 5
#         [1.0, 0.0],  # позиция 6
#         [1.0, 0.0],  # позиция 7
#     ]])
    
#     rotated = rope_2d(vector)
    
#     # Создаем график
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # График 1: Вращение единичного вектора
#     colors = plt.cm.viridis(np.linspace(0, 1, 8))
    
#     for i in range(8):
#         x, y = rotated[0, i, 0].numpy(), rotated[0, i, 1].numpy()
#         angle_deg = np.arctan2(y, x) * 180 / np.pi
        
#         # Рисуем вектор
#         ax1.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, 
#                  fc=colors[i], ec=colors[i], length_includes_head=True, 
#                  label=f'Поз {i} ({angle_deg:.1f}°)')
        
#         # Подписываем позиции
#         ax1.text(x * 1.1, y * 1.1, f'{i}', fontsize=10, color=colors[i])
    
#     ax1.set_xlim(-1.5, 1.5)
#     ax1.set_ylim(-1.5, 1.5)
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')
#     ax1.set_title('Вращение вектора [1, 0] по позициям')
#     ax1.grid(True, alpha=0.3)
#     ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
#     ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
#     ax1.set_aspect('equal')
#     ax1.legend()
    
#     # График 2: Углы поворота по позициям
#     positions = list(range(8))
#     angles = []
    
#     for i in range(8):
#         x, y = rotated[0, i, 0].numpy(), rotated[0, i, 1].numpy()
#         angle_deg = np.arctan2(y, x) * 180 / np.pi
#         angles.append(angle_deg)
    
#     ax2.plot(positions, angles, 'o-', linewidth=2, markersize=8, 
#             color='red', alpha=0.7)
#     ax2.set_xlabel('Позиция')
#     ax2.set_ylabel('Угол поворота (градусы)')
#     ax2.set_title('Зависимость угла от позиции')
#     ax2.grid(True, alpha=0.3)
    
#     # Добавляем значения на точки
#     for i, (pos, angle) in enumerate(zip(positions, angles)):
#         ax2.annotate(f'{angle:.1f}°', (pos, angle), 
#                     textcoords="offset points", xytext=(0,10), 
#                     ha='center', fontsize=9)
    
#     plt.tight_layout()
#     plt.savefig("vis_rot.png")
    
#     print("Визуализация завершена!")
#     print()


# def visualize_multiple_vectors():
#     """Визуализация нескольких разных векторов"""
#     print("=== ВИЗУАЛИЗАЦИЯ НЕСКОЛЬКИХ ВЕКТОРОВ ===")
    
#     rope_2d = RotaryPositionEmbedding(dim=2, max_seq_len=12)
    
#     # Несколько разных начальных векторов
#     initial_vectors = [
#         [1.0, 0.0],    # Вдоль оси X
#         [0.0, 1.0],    # Вдоль оси Y
#         [0.707, 0.707], # Под 45 градусов
#         [-0.5, 0.866],  # Под 120 градусов
#     ]
    
#     vector_names = ['[1, 0]', '[0, 1]', '[0.7, 0.7]', '[-0.5, 0.87]']
#     colors = ['red', 'blue', 'green', 'purple']
    
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     axes = axes.flatten()
    
#     for idx, (initial_vec, name, color) in enumerate(zip(initial_vectors, vector_names, colors)):
#         # Создаем тензор с одним вектором во всех позициях
#         vectors = ops.array([[initial_vec] * 6])  # 6 позиций
        
#         # Применяем RoPE
#         rotated_vectors = rope_2d(vectors)
        
#         ax = axes[idx]
        
#         # Рисуем начальный вектор
#         ax.arrow(0, 0, initial_vec[0], initial_vec[1], 
#                 head_width=0.05, head_length=0.1, 
#                 fc='black', ec='black', length_includes_head=True,
#                 linestyle='--', alpha=0.5, label='Начальный')
        
#         # Рисуем повернутые векторы
#         for i in range(6):
#             x, y = rotated_vectors[0, i, 0].numpy(), rotated_vectors[0, i, 1].numpy()
#             ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.06, 
#                     fc=color, ec=color, length_includes_head=True, alpha=0.7)
#             ax.text(x * 1.1, y * 1.1, f'{i}', fontsize=8, color=color)
        
#         ax.set_xlim(-1.2, 1.2)
#         ax.set_ylim(-1.2, 1.2)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_title(f'Вектор {name}')
#         ax.grid(True, alpha=0.3)
#         ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
#         ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
#         ax.set_aspect('equal')
#         ax.legend()
    
#     plt.tight_layout()
#     plt.savefig("vis_mul_vecs.png")
    
#     print("Визуализация нескольких векторов завершена!")
#     print()


# def visualize_rotation_animation_preview():
#     """Предпросмотр анимации вращения (статичные кадры)"""
#     print("=== ПРЕДПРОСМОТР АНИМАЦИИ ВРАЩЕНИЯ ===")
    
#     rope_2d = RotaryPositionEmbedding(dim=2, max_seq_len=12)
    
#     # Создаем несколько кадров для анимации
#     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
#     axes = axes.flatten()
    
#     initial_vector = [1.0, 0.0]
#     vectors = ops.array([[initial_vector] * 12])
#     rotated_vectors = rope_2d(vectors)
    
#     for frame_idx in range(12):
#         ax = axes[frame_idx]
        
#         # Рисуем начальный вектор
#         ax.arrow(0, 0, initial_vector[0], initial_vector[1], 
#                 head_width=0.05, head_length=0.1, 
#                 fc='gray', ec='gray', length_includes_head=True,
#                 linestyle='--', alpha=0.3)
        
#         # Рисуем все векторы до текущей позиции
#         for i in range(frame_idx + 1):
#             x, y = rotated_vectors[0, i, 0].numpy(), rotated_vectors[0, i, 1].numpy()
#             color = plt.cm.plasma(i / 12)
#             ax.arrow(0, 0, x, y, head_width=0.04, head_length=0.08, 
#                     fc=color, ec=color, length_includes_head=True, alpha=0.8)
            
#             if i == frame_idx:  # Выделяем текущую позицию
#                 ax.text(x * 1.15, y * 1.15, f'Поз {i}', fontsize=9, 
#                        color=color, weight='bold')
        
#         ax.set_xlim(-1.3, 1.3)
#         ax.set_ylim(-1.3, 1.3)
#         ax.set_title(f'Кадр {frame_idx + 1}/12')
#         ax.grid(True, alpha=0.3)
#         ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
#         ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
#         ax.set_aspect('equal')
    
#     plt.tight_layout()
#     plt.savefig("vis_rot_anim_prev.png")
    
#     print("Предпросмотр анимации завершен!")
#     print("Каждый кадр показывает накопление повернутых векторов.")
#     print()


# def demonstrate_rope_properties():
#     """Демонстрация ключевых свойств RoPE"""
#     print("=== КЛЮЧЕВЫЕ СВОЙСТВА ROPE ===")
    
#     rope = RotaryPositionEmbedding(dim=4, max_seq_len=5)
    
#     # Тестовые данные
#     x = ops.array([[
#         [1.0, 0.0, 0.0, 1.0],
#         [1.0, 0.0, 0.0, 1.0],
#         [1.0, 0.0, 0.0, 1.0],
#     ]])
    
#     output = rope(x)
    
#     # Свойство 1: Сохранение нормы
#     norms_before = ops.sqrt(ops.sum(x * x, axis=-1))
#     norms_after = ops.sqrt(ops.sum(output * output, axis=-1))
    
#     print("1. Сохранение нормы:")
#     print(f"   Нормы до: {[f'{n:.3f}' for n in norms_before[0].numpy()]}")
#     print(f"   Нормы после: {[f'{n:.3f}' for n in norms_after[0].numpy()]}")
#     print(f"   ✓ Нормы сохраняются: {ops.isclose(norms_before, norms_after)}")
    
#     # Свойство 2: Разные позиции - разные кодирования
#     different_positions = not ops.all(output[0, 0] == output[0, 1])
#     print(f"\n2. Разные позиции: {different_positions}")
#     print("   ✓ Каждая позиция получает уникальное кодирование")
    
#     # Свойство 3: Относительные позиции
#     rel_diff_01 = ops.norm(output[0, 1] - output[0, 0])
#     rel_diff_12 = ops.norm(output[0, 2] - output[0, 1])
#     print(f"\n3. Относительные позиции:")
#     print(f"   Разница поз1-поз0: {rel_diff_01:.4f}")
#     print(f"   Разница поз2-поз1: {rel_diff_12:.4f}")
#     print("   ✓ Относительные различия постоянны")
    
#     print()


# # Обновленная функция запуска всех демонстраций
# def run_comprehensive_demo_with_plots():
#     print("🎯 ПОЛНАЯ ДЕМОНСТРАЦИЯ ROTARY POSITION EMBEDDING\n")
    
#     demonstrate_rotation_math()
#     test_basic_rotations()
#     test_rotation_patterns()
#     test_norm_preservation()
#     test_relative_positions()
    
#     # Визуализации с графиками
#     visualize_rotation_2d_plot()
#     visualize_multiple_vectors()
#     visualize_rotation_animation_preview()
#     demonstrate_rope_properties()
    
#     print("🎉 Все демонстрации завершены!")
#     print("\n📝 КЛЮЧЕВЫЕ СВОЙСТВА ROPE:")
#     print("   • Сохраняет нормы векторов")
#     print("   • Уникальное кодирование для каждой позиции") 
#     print("   • Постоянные относительные различия")
#     print("   • Работает с последовательностями переменной длины")
#     print("   • Эффективно для attention механизмов")


# if __name__ == "__main__":
#     run_comprehensive_demo_with_plots()




# def demonstrate_isometry():
#     """Демонстрация что поворот сохраняет все отношения"""
#     rope = RotaryPositionEmbedding(dim=4, max_seq_len=3)
    
#     # Два семантически близких вектора
#     vec1 = ops.array([1.0, 0.5, 0.3, 0.2])
#     vec2 = ops.array([1.1, 0.6, 0.4, 0.3])  # близкий к vec1
    
#     # Два семантически далеких вектора  
#     vec3 = ops.array([-1.0, -0.5, -0.3, -0.2])  # противоположный
    
#     # Применяем RoPE ко всем позициям
#     input_vectors = ops.array([[vec1, vec1, vec1],
#                               [vec2, vec2, vec2], 
#                               [vec3, vec3, vec3]])
    
#     output_vectors = rope(input_vectors)
    
#     # Проверяем сохранение отношений
#     original_distance = ops.norm(vec1 - vec2)  # маленькое расстояние
#     rotated_distance = ops.norm(output_vectors[0,0] - output_vectors[1,0])  # остается маленьким
    
#     print(f"Исходное расстояние между близкими векторами: {original_distance:.4f}")
#     print(f"Расстояние после поворота: {rotated_distance:.4f}")
#     print(f"Расстояние сохранилось: {ops.isclose(original_distance, rotated_distance)}")


# demonstrate_isometry()


# def demonstrate_semantics_vs_position():
#     """Разделение семантики и позиционной информации"""
    
#     # Эмбеддинг слова "cat" в разных позициях
#     cat_embedding = ops.array([0.8, 0.2, 0.1, 0.9])
    
#     # Эмбеддинг слова "dog" в разных позициях  
#     dog_embedding = ops.array([0.7, 0.3, 0.2, 0.8])
    
#     rope = RotaryPositionEmbedding(dim=4, max_seq_len=3)
    
#     # Семантическое сходство должно сохраняться
#     original_similarity = ops.dot(cat_embedding, dog_embedding)
    
#     # После RoPE сходство должно остаться тем же для одинаковых позиций
#     cat_rotated = rope(ops.array([[cat_embedding] * 3]))
#     dog_rotated = rope(ops.array([[dog_embedding] * 3]))
    
#     for pos in range(3):
#         rotated_similarity = ops.dot(cat_rotated[0, pos], dog_rotated[0, pos])
#         print(f"Позиция {pos}: сходство до={original_similarity:.3f}, после={rotated_similarity:.3f}")


# demonstrate_semantics_vs_position()