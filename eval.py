import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from model_train import VDCNN_classification
import os
import util

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

strategy = tf.distribute.MirroredStrategy(devices=["GPU:1"])

file = open('spam1.csv', 'r', encoding='utf-8-sig')

rdr = csv.reader(file)

X, y = util.load_data(rdr)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

vocab = set()
for i in X_train:
    vocab.update(i)

vocab_index = {}

for i in vocab:
    vocab_index[i] = len(vocab_index) + 1

vocab_index['OOV'] = len(vocab_index) + 1

train_X, val_X, test_X = util.word_to_index(vocab_index, X_train), util.word_to_index(vocab_index, X_val),\
                         util.word_to_index(vocab_index, X_test)

max_len = max([len(i) for i in train_X])
vocab_size = len(vocab_index)

X_train = pad_sequences(train_X, padding='post', maxlen=max_len)
X_val = pad_sequences(val_X, padding='post', maxlen=max_len)
X_test = pad_sequences(test_X, padding='post', maxlen=max_len)

y_train = to_categorical(y_train, dtype='int64')
y_val = to_categorical(y_val, dtype='int64')
y_test = to_categorical(y_test, dtype='int64')

batch_size = 256

train_data = util.tensor_transform(X_train, y_train, batch_size)
val_data = util.tensor_transform(X_val, y_val, batch_size)
test_data = util.tensor_transform(X_test, y_test, batch_size)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

train_data = train_data.with_options(options)
val_data = val_data.with_options(options)
test_data = test_data.with_options(options)

model_name = input('Model Input : [VDCNN9, VDCNN17, VDCNN29, VDCNN49]  ')

with strategy.scope():
    vdcnn = VDCNN_classification(model_name=model_name, max_len=max_len, vocab_size=vocab_size, batch_size=batch_size,
                                 train_data=train_data, val_data=val_data, test_data=test_data)
    vdcnn.train()
