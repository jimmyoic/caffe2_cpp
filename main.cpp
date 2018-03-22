#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/tensor.h"
#include "Caffe2Handler.hpp"
#include <iostream>
#include <fstream>
using namespace std;

void testForward();
//void testSaveNetwork();

int main(int argc, char** argv) {
	testForward();
	return 0;
}
/*
void testSaveNetwork()
{
	fstream f("img.bin",ios::in|ios::binary);
	float imgBinary[1 * 3 * 227 * 227];
	f.read((char*)&imgBinary, sizeof(float)*1 * 3 * 227 * 227);
	
	Caffe2Handler caffe2Handler;
	caffe2Handler.loadNetworkProto("init_net.pb", "predict_net.pb");
	caffe2Handler.initializeNetwork();
	caffe2Handler.setInputDimension("data",vector<TIndex>({1,3,227,227}));
	caffe2Handler.setDataInput(imgBinary, "data");
	caffe2Handler.forward();
	std::vector<float> result = caffe2Handler.getBlobContentByName<float>("softmaxout");
	caffe2Handler.saveNetwork("test.pb");
	
	Caffe2Handler caffe2Handler__;
	caffe2Handler__.loadNetworkProto("test.pb", "predict_net.pb");
	caffe2Handler__.initializeNetwork();
	caffe2Handler.setInputDimension("data",vector<TIndex>({1,3,227,227}));
	caffe2Handler__.setDataInput(imgBinary, "data");
	caffe2Handler__.forward();
	
	std::vector<float> result__ = caffe2Handler.getBlobContentByName<float>("softmaxout");
	for(int i=0;i<result.size();++i){
		if(result[i] != result__[i]){
			std::cout << "answer inconsistent!!" <<endl;
			break;
		}
	}
	
	cout << "Pass" <<endl;
	
	
	
}
*/
void testForward()
{
  Caffe2Handler caffe2Handler;

  // false for CPU mode, True for GPU mode (default in GPU mode)
  caffe2Handler.loadNetworkProto("init_net.pb", "predict_net.pb");
  
  // set the device for running the net. if run on CUDA, specify if we want to assign deviceOption according to prototxt
  caffe2Handler.setRunDevice(CUDA, false);
  caffe2Handler.initializeNetwork();
  
  caffe2Handler.setInputDimension("data",vector<TIndex>({1,3,227,227}));
  
  // img.bin is made from "images/flower.jpg" (caffe2 respository) 
  fstream f("img.bin",ios::in|ios::binary);
  float imgBinary[1 * 3 * 227 * 227];

  //for(int i=0;i<1*3*227*227; ++i) { imgBinary[i] = 0;}
  f.read((char*)&imgBinary, sizeof(float)*1 * 3 * 227 * 227);
  
  caffe2Handler.setDataInput(imgBinary, "data");
  caffe2Handler.forward();
  
  
  // output blob of squeezenet
  std::vector<float> result = caffe2Handler.getBlobContentByName<float>("softmaxout");
  
  // check the result
  vector< pair<int, float> > outputPair;
  for(int i=0;i<1000; ++i){
	  outputPair.push_back(pair<int,float>(i, result[i]));
	  //cout << fixed << setprecision(4) << result[i] << " ";
  }
  sort( outputPair.begin(), outputPair.end(), 
    [](const pair<int, float> & a, pair<int, float>  & b)
  { 
    return a.second > b.second; 
  });

  cout << "Raw top 3 results:" << endl;
  for(int i=0;i<3;++i)
  cout << outputPair[i].first << ": " << outputPair[i].second << endl;

	
}
