# Caffe2 GPU/CPU Inference 

program for GPU/CPU inference on caffe2, this could be set when loading predict_net

usage:

cmake . (-DUSE_MKL=True if caffe2 is compiled with MKL lib)

make

./bin/caffe2_inference_test

The network and data used in the program follows https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Loading_Pretrained_Models.ipynb
