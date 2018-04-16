#ifndef CAFFE2_NETWORK_HANDLER
#define CAFFE2_NETWORK_HANDLER

#include <vector>
#include <map>
#include <set>

#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/operator_gradient.h"

using namespace caffe2;
using namespace std;



class Caffe2Handler 
{
public:
	Caffe2Handler(void)
	{
		//default setting
		m_bUseDefaultGPUDeviceOption=false;
		m_bIsParallel=false;
		m_DeviceType=CPU;
	}
	~Caffe2Handler(void)
	{ 
	}	

	/* need some way to fix this arch (repeated switch/case in functions) */
	template <typename T = float, class ContextT = CPUContext, class TensorT = TensorCPU>
	bool setDataInput(T* input, string sBlobName, bool isControlByDevice=true)
	{
		if(isControlByDevice){
			switch(m_DeviceType){
				case CPU:  return setDataInputForDevice<T, CPUContext, TensorCPU>(input, sBlobName);
				case CUDA: return setDataInputForDevice<T, CUDAContext, TensorCUDA>(input, sBlobName); 
				default:
					;
					return setDataInputForDevice<T, CPUContext, TensorCPU>(input, sBlobName);
			}
		}
		else{
			return setDataInputForDevice<T, ContextT, TensorT>(input, sBlobName);
		}
		return false;
	}
	
	template <class TensorT = TensorCPU>
	bool setInputDimension(string sBlobName, vector<TIndex> vInputDims, bool isControlByDevice=true)
	{
		if(isControlByDevice){
			switch(m_DeviceType){
				case CPU:  return setInputDimensionForDevice<TensorCPU>(sBlobName, vInputDims);
				case CUDA: return setInputDimensionForDevice<TensorCUDA>(sBlobName, vInputDims); 
				default:
					return setInputDimensionForDevice<TensorCPU>(sBlobName, vInputDims);
					//not implemented
			}
		}
		else{
			return setInputDimensionForDevice<TensorT>(sBlobName, vInputDims);
		}
		return false;
		
	}
	
	template <class TensorT = TensorCPU>
	inline TensorCPU getTensorByName(std::string sBlobName, bool isControlByDevice=true)
	{
		if(isControlByDevice){
			switch(m_DeviceType){
				case CPU:  return getTensorFromDeviceByName<TensorCPU>(sBlobName);
				case CUDA: return getTensorFromDeviceByName<TensorCUDA>(sBlobName); 
				default:
				std::cout << "default data, somewhere error" << std::endl;
					;
					CPU:  return getTensorFromDeviceByName<TensorCPU>(sBlobName);
			}
		}
		else{
			return getTensorFromDeviceByName<TensorT>(sBlobName);
			// implemented when needed by giving template
		}
	}
	
	void loadNetworkProto( string sInitNetName, string sPredictNetName);
	void initializeNetwork();    
	void saveNetwork(std::string outputName, bool isSnapShot=false); 
	void releaseNetwork(); // TODO
	bool enableCUDA();
	void reloadNetworkParameter(); // TODO
	bool enableCPU();
	void forward();
	void setRunDevice(DeviceType deviceType, bool isFollowPrototxt=true);
	OperatorDef genOp(string type, const vector<string>& inputs, const vector<string>& outputs);
	OperatorDef genOp(string type, const vector<string>& inputs, const vector<string>& outputs, int deviceType, int gpuID);
	
	inline void printNetPredictDef() { cout << m_Predict_net.DebugString() << endl;}
	inline void printNetInitDef() { cout << m_Init_net.DebugString() << endl;}
	inline bool hasBlob(std::string sBlobName) { return m_Workspace.HasBlob(sBlobName); }
	inline int getGPUId(){ return 0;}
	inline int getBatchSize(){ return 0;}
	inline int getInputChannelSize(){ return 0;}
	inline int getDataWidthSize(){ return 0;}
	inline int getDataHeightSize(){ return 0;}
	
	
	
	template <typename T>
	const inline std::vector<T> getBlobContentByName(std::string name){  
		TensorCPU cpudata = getTensorByName(name); // need a instance to save
		const T *data = cpudata.data<T>();
		const int size = cpudata.size();
		return std::vector<T>(data, data+size);
	}
	// This return a "copied" data from device
	
protected:
	template <class TensorT>
	inline TensorT *getTensorPtrFromDeviceByName(std::string sBlobName)
	{
		return m_Workspace.GetBlob(sBlobName)->Get<TensorT>();
	}
	
	template <class TensorT>
	inline TensorCPU getTensorFromDeviceByName(std::string sBlobName)
	{
		return TensorCPU(m_Workspace.GetBlob(sBlobName)->Get<TensorT>());
	}
	
	template <typename T = float, class ContextT, class TensorT>
	bool setDataInputForDevice(T* input, std::string sBlobName)
	{
		ContextT t;
		if(!m_Workspace.HasBlob(sBlobName)) return false;
		auto tensor = m_Workspace.GetBlob(sBlobName)->template GetMutable<TensorT>();
		ContextT context(m_BlobOptionMap[sBlobName]);
		context.SwitchToDevice();
		context. template CopyBytes< CPUContext , ContextT>(tensor->size() * sizeof(T), static_cast<void*>(input), tensor->raw_mutable_data());
		context.FinishDeviceComputation();
		return true;
	}
	
	template <class TensorT>
	bool setInputDimensionForDevice(string sBlobName, vector<TIndex>& vInputDims)
	{
		TensorT *tensor= m_Workspace.GetBlob(sBlobName)->template GetMutable<TensorT>();
		tensor->Resize(vInputDims); 		
		return true;
		
	}
	
	void collectNetParams(); // record all the blob param
	
	// Problem: Since Python API have record some information like tags (COMPUTED, WEIGHT, BIAS ...), we don't have those info if we load exist NetDef.
	// Hack, Roughly find out those type by some postfix _w, _b... , if needed. (Only Test for conv, fc, bn ...). There should be some problems, but anyway do it first.
	vector<string> m_sParams;
	vector<string> m_sParamsWeights;
	vector<string> m_sParamsBias;
	vector<string> m_sParamsComputed;
	
	Workspace m_Workspace;
	NetDef m_Init_net, m_Predict_net;
	map<string, DeviceOption> m_BlobOptionMap;
	DeviceType m_DeviceType;
	bool m_bUseDefaultGPUDeviceOption;
	bool m_bIsParallel;
	
	

};

#endif	
