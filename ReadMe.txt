The data set is due to "SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction"
This pyhton package trains a convolutional neural network (MobileNetV2) to rate the images from the data set by a value between [1,5]. 
Transfer-learning with pretrained weights from ImageNet is used. We train only the classificator, leaving the convolutional base frozen.
With the loss MSE we achieved ~0.18 on validation at its best. Early stopping is required, otherwise the model will overfit after too many epochs.

