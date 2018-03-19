#ifndef CAFFE2_NETWORK_HANDLER
#define CAFFE2_NETWORK_HANDLER

#include <vector>
#include <map>

#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/tensor.h"

using namespace caffe2;
using namespace std;

class Caffe2Handler 
{
public:
	Caffe2Handler(void)
	{
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
	
	void loadNetworkProto( string sInitNetName, string sPredictNetName, bool isCUDA, int gpuID=0);
	void initializeNetwork();    
	void saveNetwork(std::string outputName, bool isSnapShot=false); 
	void releaseNetwork(); // TODO
	bool enableCUDA(bool isDeviceFollowProto=true);
	void reloadNetworkParameter(); // TODO
	bool enableCPU();
	void forward();
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
	
private:
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
	
	Workspace m_Workspace;
	NetDef m_Init_net, m_Predict_net, m_Deploy_net;
	map<string, DeviceOption> m_BlobOptionMap;
	DeviceType m_DeviceType;
	
	

};

#endif	