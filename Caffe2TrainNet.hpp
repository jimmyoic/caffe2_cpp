#ifndef CAFFE2_TRAIN_NET
#define CAFFE2_TRAIN_NET


#include "Caffe2Handler.hpp"
#include "IR.hpp"

struct LossLayerData{
	vector<string> inputs;
	vector<string> outputs;
	LossLayerData(vector<string> inputs, vector<string> outputs)
	{
			this->inputs = inputs;
			this->outputs = outputs;
	}
	 
};

class Caffe2TrainNet : public Caffe2Handler
{
	public:
		Caffe2TrainNet():m_bIsAddGradientOperator(false), m_fLearningRate(0.005), m_fWeightDecay(0.0001), m_fMomentum(0.9){}
		void prepareTrainingDef();
		void parallelTrainingModel(const vector<int>& gpuList, const vector<LossLayerData>& vLossLayerData);
		void addGradientOperator(const vector<LossLayerData>& vLossLayerData);
		void addOperator(const OperatorDef& op);
	
	private:
		void recordNetworkBlobParam();
		void inferBlobDevice();
		void broadcastFromMasterGPU(const string& param);
		bool isParamGPU(string param);
		void allReduceBlobs(const vector<string>& blobNames);
		void sumBetweenDevice(const string& param, unordered_map<int, string>& mGpuBlobsGroup, const vector<vector<bool>>& vP2PAccess, const vector<int>& deviceIndices);
		
		// set training parameter
		inline void setWeightDecay(float weightDecay){ m_fWeightDecay = weightDecay;}
		inline void setLearningRate(float learningRate){ m_fLearningRate = learningRate;}
		inline void setMomentum(float momentum){ m_fMomentum = momentum;}
		
		
		// should be modified flexibly according to need (TODO)
		vector<string> paramUpdateBuilder();
		void addWeightDecay();
		void basicSettingBuilder();
		
		unordered_map<string, string> getParamToGrad();
		unordered_map<string,  map<int , string>> updateDeviceGrouped(const vector<int>& vDevices, const vector<string>& vParams);
	
	
	
		float m_fLearningRate;
		float m_fWeightDecay;
		float m_fMomentum;
		
	
	
	// for parallelizing model
	
		bool m_bIsAddGradientOperator;
		unordered_map<string, DeviceOption> m_mBlobToDevice;
		unordered_map<string,  map<int , string>> m_mDeviceGrouped; // usually we use unorder_map. However, the order (int(deviceID)) is matter here
		unordered_map<string, string> m_mGradMap;
		vector<string> m_vBlobParam;
		vector<int> m_vGpuList;






};

#endif	
