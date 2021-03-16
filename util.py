import numpy as np
import tensorflow as tf


def load_data(rdr):
    sentence = []
    label = []
    for idx, line in enumerate(rdr):
        if idx == 0:
            continue
        if len(line[2]) == 0:
            continue
        etc = []
        for i in line[2]:
            etc.append(i.lower())
        sentence.append(etc)
        label.append(line[1].strip())
    return sentence, label


def tensor_transform(X, y, batch_size):
    X, y = np.array(X), np.array(y)
    data = tf.data.Dataset.from_tensor_slices((X, y))
    data = data.batch(batch_size)

    return data


def word_to_index(vocab_index, data):
    X = []
    for i in data:
        etc = []
        for j in i:
            if j in vocab_index.keys():
                etc.append(vocab_index[j])
            else:
                etc.append(vocab_index['OOV'])
        X.append(etc)

    return X
