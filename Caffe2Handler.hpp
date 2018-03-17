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
	template <class ContextT = CPUContext, class TensorT = TensorCPU>
	bool setDataInput(float* input, string sBlobName, bool isControlByDevice=true)
	{
		if(isControlByDevice){
			switch(m_DeviceType){
				case CPU:  return setDataInputForDevice<CPUContext, TensorCPU>(input, sBlobName);
				case CUDA: return setDataInputForDevice<CUDAContext, TensorCUDA>(input, sBlobName); 
				default:
					;
					return setDataInputForDevice<CPUContext, TensorCPU>(input, sBlobName);
			}
		}
		else{
			return setDataInputForDevice<ContextT, TensorT>(input, sBlobName);
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
	inline int getGPUId(){ return m_iGPU_ID;}
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
		
		// cannot return TensorCPU(m_Workspace.GetBlob(name)->Get<TensorCUDA>()).data<float>(); 
		// need copy memory in this way (sharePtr?)
		// maybe more efficient if get the content directly outside the function by a for loop
		// for( auto value : TensorCPU(m_Workspace.GetBlob(name)->Get<TensorCUDA>()).data<float>() ) ....
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
	
	template <class ContextT, class TensorT>
	bool setDataInputForDevice(float* input, std::string sBlobName)
	{
		ContextT t;
		if(!m_Workspace.HasBlob(sBlobName)) return false;
		auto tensor = m_Workspace.GetBlob(sBlobName)->template GetMutable<TensorT>();
		ContextT cudaContext(m_BlobOptionMap[sBlobName]);
		cudaContext.SwitchToDevice();
		cudaContext. template CopyBytes< CPUContext , ContextT>(tensor->size() * sizeof(float), static_cast<void*>(input), tensor->raw_mutable_data());
		cudaContext.FinishDeviceComputation();
		
		// TODO: abstract layer for TensorCUDA/CPU?
	
		
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
	
	
	uint m_iGPU_ID;
	map<string, DeviceOption> m_BlobOptionMap;
	DeviceType m_DeviceType;
	
	


};

#endif	