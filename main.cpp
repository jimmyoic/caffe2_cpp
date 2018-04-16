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
#include "Caffe2TrainNet.hpp"
#include <iostream>
#include <fstream>

#define BATCH_SIZE 32
using namespace std;

void testForward();
void testTrain();
//void testSaveNetwork();

int main(int argc, char** argv) {
	testForward();
	return 0;
}

void testTrain()
{
  Caffe2TrainNet caffe2TrainNet;
  caffe2TrainNet.loadNetworkProto("initDef.txt", "trainDef.txt");
  vector<LossLayerData> lossLayerDatas({LossLayerData({"_sl_softmax_loss_1"}, {"sl_softmax_loss_1"})});
 
  vector<int> vGpus({0,1,2,3});
  caffe2TrainNet.parallelTrainingModel(vGpus ,lossLayerDatas);
  caffe2TrainNet.initializeNetwork();
 
  for(int m=0;m<vGpus.size(); ++m){
		string data_blob_name = "gpu_" + to_string(vGpus[m]) + "/data";
		string label_blob_name = "gpu_" + to_string(vGpus[m]) + "/label_sl";
		caffe2TrainNet.setInputDimension(data_blob_name, vector<TIndex>({BATCH_SIZE,51,19,19}));
		caffe2TrainNet.setInputDimension(label_blob_name,vector<TIndex>({BATCH_SIZE}));
  }

  clock_t start = clock();
  
  float *featureData = new float[BATCH_SIZE * 51 * 361];
  float *featureLabel = new float[BATCH_SIZE * 1 ];
  for(int i=0;i< BATCH_SIZE * 51 * 361 ; ++i)  featureData[i] = 0;
  for(int i=0;i< BATCH_SIZE; ++i)  featureLabel[i] = 0;
  
  
  int a=0;
  for(int i=0;i<100000; ++i){
  	//std::cout << "Iter : " << i << std::endl;
  	for (int j = 1; j <= 10; j++) {
  		std::ifstream fs;
  		string featureName = "feature" + std::to_string(j);
  		fs.open(featureName.c_str(), std::ios::binary);
  		for (int k = 0; k < 100000 / BATCH_SIZE / vGpus.size(); k++) {
  			for(int m=0;m<vGpus.size(); ++m){
  				int intLabel[BATCH_SIZE * 1];
  				for (int n = 0; n < BATCH_SIZE; n++) {
  					fs.read(reinterpret_cast<char*>(featureData + n * (51 * 361)), sizeof(float) * (51 * 361));
  					fs.read(reinterpret_cast<char*>(featureLabel + n), sizeof(float));
  				}
  				for (int n = 0; n < BATCH_SIZE; n++) {
  					intLabel[n] = (int)featureLabel[n];
  					//std::cout << intLabel[n] << std::endl;
  				}
  				string data_blob_name = "gpu_" + to_string(vGpus[m]) + "/data";
  				string label_blob_name = "gpu_" + to_string(vGpus[m]) + "/label_sl";
  				caffe2TrainNet.setDataInput<float>(featureData, data_blob_name);
  				caffe2TrainNet.setDataInput<int>(intLabel, label_blob_name);
  			}
  			
  			caffe2TrainNet.forward();
			
  			if ((a+1) % 10 == 0) {
  				float loss=0, acc=0;
  				for(int m = 0; m < vGpus.size(); ++m){
  					TensorCPU lossTensor = caffe2TrainNet.getTensorByName("gpu_" + to_string(vGpus[m]) + "/_sl_softmax_loss_1");
  					TensorCPU accTensor = caffe2TrainNet.getTensorByName("gpu_" + to_string(vGpus[m]) + "/top_1_01");
  					loss+= lossTensor.data<float>()[0];
  					acc+= accTensor.data<float>()[0];
  				}
  				printf("[Iteration %d] Loss: %.5f , acc: %.5f,  Time cost = %.5f\n",a+1, (loss/vGpus.size() ) , (acc/vGpus.size() ), double(clock() - start) / CLOCKS_PER_SEC);
				start = clock();
  			}
			a++;
  		}
  	}
  	std::cout << std::endl;
  
  }
  return;
}

void testForward()
{

  // set the device for running the net. if run on CUDA, specify if we want to assign deviceOption according to prototxt
  // default is CPU
  
  Caffe2Handler caffe2Handler;
  caffe2Handler.loadNetworkProto("init_net.pb", "predict_net.pb");
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
