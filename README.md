# Very-Deep-Convolutional-Networks-for-Text-Classification

  * Text Classification
  
  * Conneau, A., Schwenk, H., Barrault, L., & Lecun, Y. (2017, April). Very Deep Convolutional Networks for Text Classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers (pp. 1107-1116). [[Paper]](https://www.aclweb.org/anthology/E17-1104.pdf)
   
  * Model Structure
    
    * Temp Conv
    
    * Convolutional Block, kernel_size=3, the number of filters=64 (shortcut + pooling)
    
    * Convolutional Block, kernel_size=3, the number of filters=128 (shortcut + pooling)
    
    * Convolutional Block, kernel_size=3, the number of filters=256 (shortcut + pooling)
    
    * Convolutional Block, kernel_size=3, the number of filters=512 (shortcut + pooling)

    * K-Max Pooling (k=8)

    * Fully-Connected Layer * 2

    * Classfication layer

  * Dataset used in this model
    * Spam dataset : https://www.kaggle.com/uciml/sms-spam-collection-dataset

  * Result
  
    | Depth | Accuracy |   Loss  |
    | :---: | :------: | :-----: |
    |   9   | 0.974855 | 0.085224|
    |   17  | 0.976789 | 0.089545|
    |   29  | 0.980658 | 0.072725|
    |   49  | 0.847195 | 0.450622|
