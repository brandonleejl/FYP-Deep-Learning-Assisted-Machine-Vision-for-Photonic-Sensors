import numpy as np
import tensorflow as tf
print(f"Current NumPy: {np.__version__}")
print(f"GPUs Found: {tf.config.list_physical_devices('GPU')}")