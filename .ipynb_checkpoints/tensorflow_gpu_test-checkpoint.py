# import tensorflow
import tensorflow as tf
# Test if GPU is available
gpu_available = tf.test.is_gpu_available()
print(f"Is GPU available? {gpu_available}")

# Test GPU name
gpu_name = tf.test.gpu_device_name()
print(f"GPU Name: {gpu_name}")

# Test GPU device
gpu_device = tf.test.gpu_device_name()
print(f"GPU Device: {gpu_device}")

# Test CUDA version
cuda_built = tf.test.is_built_with_cuda()
print(f"Is TensorFlow built with CUDA? {cuda_built}")

# List physical GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Physical GPU Devices: {physical_devices}")
