# TensorFlow Distributed Training Strategies

## Overview

TensorFlow's `tf.distribute.Strategy` API empowers developers to scale training across multiple GPUs, machines, or TPUs with minimal code adjustments. This flexibility enhances both efficiency and scalability in model training. 🎯

### Key Objectives of `tf.distribute.Strategy`

- **Ease of Use**: Designed to support a wide range of users, including researchers and machine learning engineers. 🧑‍💻
- **Performance**: Optimized to deliver high performance out of the box. ⚡
- **Flexibility**: Facilitates easy switching between different distribution strategies. 🔄

This API integrates seamlessly with high-level TensorFlow APIs, such as Keras `Model.fit`, as well as custom training loops. 🔗

## Types of Distribution Strategies

### 1. Default Strategy (`tf.distribute.get_strategy()`)

- **Description**: The default no-op strategy that operates as if no distribution strategy is in place, utilizing only the available device (CPU or single GPU). 🖥️
- **Use Case**: Suitable for simple setups or debugging on a single device. 🛠️
- **Implementation**:
  ```python
  strategy = tf.distribute.get_strategy()
  ```

### 2. `OneDeviceStrategy`

- **Description**: Allocates all computations to a single specified device (e.g., CPU or GPU). 🎯
- **Use Case**: Ideal for debugging or testing on a specific device. 🧪
- **Implementation**:
  ```python
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
  ```

### 3. `MirroredStrategy`

- **Description**: Replicates the model across multiple GPUs (or CPUs) and synchronizes updates in each step. 🔄
- **Use Case**: Best for synchronous training on **multiple GPUs within a single machine**. 🖥️🖥️
- **Implementation**:
  ```python
  strategy = tf.distribute.MirroredStrategy()
  ```

### 4. `MultiWorkerMirroredStrategy`

- **Description**: Extends `MirroredStrategy` to multiple workers (machines), synchronizing updates across all devices on all workers. 🌐
- **Use Case**: Suitable for synchronous training across **multiple machines with GPUs**. 🖥️🖥️🖥️
- **Implementation**:
  ```python
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  ```

### 5. `TPUStrategy`

- **Description**: Designed for training on Tensor Processing Units (TPUs), optimized for large-scale training. 🚀
- **Use Case**: Ideal for training extensive models on **Google Cloud TPU clusters**. ☁️
- **Implementation**:
  ```python
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='your_tpu_address')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.TPUStrategy(resolver)
  ```

### 6. `ParameterServerStrategy`

- **Description**: Distributes the model across parameter servers and workers, with parameters stored on the servers and computations handled by workers. 🗄️💻
- **Use Case**: Applicable for large-scale asynchronous training across **multiple nodes**. 🌐
- **Implementation**:
  ```python
  strategy = tf.distribute.ParameterServerStrategy()
  ```

### 7. `CentralStorageStrategy`

- **Description**: Places variables on a single device (CPU or a specific GPU) and mirrors computation across multiple devices. 🖥️🔄
- **Use Case**: Useful when GPU memory is limited, necessitating variable storage on the CPU or a single GPU. 💾
- **Implementation**:
  ```python
  strategy = tf.distribute.experimental.CentralStorageStrategy()
  ```

## General Usage Pattern

To utilize a distribution strategy, follow these steps:

1. **Define the Strategy**: Select and instantiate the appropriate strategy. 📝
2. **Scope the Model Creation and Compilation**: Place the model creation and compilation within the strategy's scope. 🔍
3. **Train the Model**: Proceed with model training as usual. 🏋️‍♂️

### Example Workflow

```python
# Step 1: Define the strategy
strategy = tf.distribute.MirroredStrategy()

# Step 2: Scope model creation and compilation
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# Step 3: Train the model
model.fit(dataset, epochs=10)
```

## Practical Examples

### Using `MirroredStrategy`

`MirroredStrategy` facilitates synchronous training on multiple GPUs within a single machine. 🖥️🖥️

```python
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,), 
                              kernel_regularizer=tf.keras.regularizers.L2(1e-4))
    ])
    model.compile(loss='mse', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)
```

### Using `TPUStrategy`

`TPUStrategy` is tailored for training on TPUs, offering optimized performance for large-scale models. 🚀

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='your_tpu_address')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,), 
                              kernel_regularizer=tf.keras.regularizers.L2(1e-4))
    ])
    model.compile(loss='mse', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)
```

### Using `MultiWorkerMirroredStrategy`

`MultiWorkerMirroredStrategy` enables synchronous training across multiple machines, each with multiple GPUs. 🌐🖥️🖥️🖥️

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,), 
                              kernel_regularizer=tf.keras.regularizers.L2(1e-4))
    ])
    model.compile(loss='mse', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)
```

## Conclusion

Distributed training in TensorFlow allows you to scale your models across multiple devices and machines, enabling faster training times 

## Credits

This project utilizes the following resources:

- **TensorFlow**: An open-source machine learning framework developed by the TensorFlow team. For more information, visit the [TensorFlow website](https://www.tensorflow.org/).

- **Kaggle**: A platform offering datasets and a cloud computational environment that enables reproducible and collaborative analysis. For more information, visit the [Kaggle website](https://www.kaggle.com/).

- **Fake News Classification Dataset**: Provided by Kaggle user Aadya Singh. The dataset can be accessed [here](https://www.kaggle.com/datasets/aadyasingh55/fake-news-classification).

I extend my gratitude to the developers and contributors of these resources for their invaluable tools and datasets.<br>
Feel free to **fork**, **clone**, and **experiment** with the code. Your **feedback** and **suggestions** are always welcome. If you find it helpful, don't forget to **star** the repository! Your support keeps this project alive. 
