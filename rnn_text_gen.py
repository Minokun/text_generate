import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import re

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
def Set_GPU_Memory_Growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置 GPU 显存占用为按需分配
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # 异常处理
            print(e)
    else:
        print('No GPU')

# 放在建立模型实例之前
Set_GPU_Memory_Growth()

# 步骤
# 1. 生成词表
# 2. 建立词id映射关系
# 3. id词映射
# 4. 生成词

# ************************************* 读取数据文件 *****************************************
content = ''
n = 0
for root, dirs, files in os.walk('data3'):
    for name in files:
        file_path = os.path.join(root, name)
        with open(file_path, 'r', encoding='utf-8') as fp:
            content += fp.read()
        n += 1

# 特殊字符处理
content_deal = re.sub('\t', '\n', content)
print('总字数：', len(content_deal))
vocab = sorted(set(content_deal))
# 字符与id的映射
char2idx = {char:idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in content_deal])

# 输入输出函数
def split_input_target(id_text):
    return id_text[0:-1], id_text[1:]

# 创建输入数据集
char_dataset = tf.data.Dataset.from_tensor_slices(
    text_as_int
)
seq_length = 100
seq_dataset = char_dataset.batch(seq_length + 1, drop_remainder=True)

seq_dataset = seq_dataset.map(split_input_target)

batch_size = 32
buffer_size = 100

seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)


# 类别数
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# 构建模型
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        # keras.layers.SimpleRNN(units=rnn_units, return_sequences=True),
        keras.layers.LSTM(units=rnn_units, return_sequences=True),
        keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
#
for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_pred = model(input_example_batch)

# 随机采样
# sample_indices = tf.random.categorical(logits=example_batch_pred[0],
#                       num_samples=1)
# sample_indices = tf.squeeze(sample_indices, axis=-1)
# print(sample_indices)
# print("Input: ", repr("".join(idx2char[input_example_batch[0]])))
# print()
# print("Output: ", repr("".join(idx2char[target_example_batch[0]])))
# print()
# print("Predict: ", repr("".join(idx2char[sample_indices])))
# 损失函数
def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
# example_lost = loss(target_example_batch, example_batch_pred)
output_dir = './text_generation_checkpoints'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)
epochs = 100
history = model.fit(seq_dataset, epochs=epochs, callbacks=[checkpoint_callback])