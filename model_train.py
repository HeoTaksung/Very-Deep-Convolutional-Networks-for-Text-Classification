import models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


class VDCNN_classification(object):
    def __init__(self, model_name='VDCNN29', max_len=1024, vocab_size=10000, embedding_size=16,
                kernel_size=3, k=8, class_num=2, opt=tf.keras.optimizers.SGD(), batch_size=128,
                epochs=100, loss='binary_crossentropy', train_data=None, val_data=None, test_data=None):
        self.model_name = model_name
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.k = k
        self.class_num = class_num
        self.opt = opt
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
    def train(self):
        if self.model_name == 'VDCNN9':
            model = models.VDCNN(self, depth=9)
        elif self.model_name == 'VDCNN17':
            model = models.VDCNN(self, depth=17)
        elif self.model_name == 'VDCNN29':
            model = models.VDCNN(self, depth=29)
        elif self.model_name == 'VDCNN49':
            model = models.VDCNN(self, depth=49)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4, restore_best_weights=True)
        
        model.fit(self.train_data, epochs=self.epochs, batch_size=self.batch_size,
                  validation_data=self.val_data, callbacks=[es])
    
        test_loss, test_acc = model.evaluate(self.test_data)
        
        print("TEST Loss : {:.6f}".format(test_loss))
        print("TEST ACC : {:.6f}".format(test_acc))
