# DeXpression-PyTorch
A PyTorch implementation of DeXpression for Expression Recognition

This project is an implementation of the DeXpression deep convolutional neural network based on [this](https://arxiv.org/abs/1509.05371) paper.

The project consists of 4 parts:

- image_extractor.py is a Python3 pre-processing file that was used to sort the Cohn Kanade Extended dataset into its respective classes
- Model.py is the Python3 file containing the network architecture
- train.py is the Python3 file used to train the network. It contains explicit CONSTANT declarations for easy tweaking of hyper-parameters and file paths
- test.py is a Python3 file for validation of the trained network
