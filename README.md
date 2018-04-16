# Caffe2 GPU/CPU Inference & training by memory data

Try to implement C++ API for caffe2 to do inferencing and parallel training by memory data instead of database.

Inference: (see the sample program)
	1. load network
	2. set device (CPU/GPU)
	3. initialize
	4. set input dimension and input
	5. run net

Training: (Assume that predictDef and initDef are created by python API) (see the sample program)
	1. load network (RNN is not supported)
	2. indicate the input & output blob to Caffe2TrainNet
	3. set gpu list in order to parallelize the net
	4. initialize
	Before each run, set the training data into the blobs (i.g. gpu_{GPU#}/{DATA BLOB NAME})
	The training data can be either calculated by programs or read from the file.
	testTrain function shows how to train according to softmax loss. However, I can't provide the training data to demo.
	Since I have only tested the network I'm working on (consist of FC, conv, bn... ), if it doesn't work on other operators please inform me to improve that.

usage of sample program:
cmake . (-DUSE_MKL=True if caffe2 is compiled with MKL lib)
make
./bin/caffe2_inference_test

The forwording network and data used in the program follows https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Loading_Pretrained_Models.ipynb
