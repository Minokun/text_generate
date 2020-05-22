import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
import re
import numpy as np

content = ''
n = 0
for root, dirs, files in os.walk('data3/励志作文'):
    for name in files:
        file_path = os.path.join(root, name)
        with open(file_path, 'r', encoding='utf-8') as fp:
            content += fp.read()
        n += 1

# 特殊字符处理
content_deal = re.sub('\t', '\n', content)
vocab = sorted(set(content_deal))
# 字符与id的映射
char2idx = {char:idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# 类别数
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# 构建模型
def build_model(vocab_size, embedding_dim, rnn_units, batch_size=1):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(units=rnn_units, return_sequences=True),
        keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(vocab_size, embedding_dim, rnn_units)
output_dir = './text_generation_checkpoints'
model.load_weights(tf.train.latest_checkpoint(output_dir))

# 文本生成
def generate_text(model, start_string, num_generate=600):
    input_eval = [char2idx[ch] for ch in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictied_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        text_generated.append(idx2char[predictied_id])
        input_eval = tf.expand_dims([predictied_id], 0)
    return start_string + ''.join(text_generated)

new_text = generate_text(model, '失败是成功之母')
print(new_text)