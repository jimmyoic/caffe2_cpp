#include "Caffe2Handler.hpp"

#include <iostream>

void Caffe2Handler::loadNetworkProto( string sInitNetName, string sPredictNetName)
{
	CAFFE_ENFORCE(ReadProtoFromFile(sInitNetName, &m_Init_net));
	CAFFE_ENFORCE(ReadProtoFromFile(sPredictNetName, &m_Predict_net));
	
}

void Caffe2Handler::setRunDevice(DeviceType deviceType, bool bIsFollowPrototxt)
{
	m_DeviceType = deviceType;
	m_bUseDefaultGPUDeviceOption= !bIsFollowPrototxt;
}


void Caffe2Handler::saveNetwork(string outputName, bool isSnapShot)
{
	NetDef init_net_; 
	for ( auto inputName : m_Deploy_net.external_input()) {
		TensorCPU tensor = TensorCPU(m_Workspace.GetBlob(inputName)->Get<TensorCUDA>());
		OperatorDef *op = init_net_.add_op();
		op->set_type("GivenTensorFill");
		// shape
		op->add_arg()->CopyFrom(MakeArgument("shape", tensor.dims()));	
		// insert weights
		const float *weights = tensor.data<float>();
		std::vector<float> weightsVector( weights, weights + tensor.size()); 
		op->add_arg()->CopyFrom(MakeArgument("values", weightsVector));
		// add output
		op->add_output(inputName);
	}
	WriteProtoToBinaryFile(init_net_, outputName);
	if(isSnapShot) m_Init_net = init_net_;
}

void Caffe2Handler::releaseNetwork()
{
	m_Workspace.DeleteNet(m_Predict_net.name());
}

bool Caffe2Handler::enableCUDA()
{
	if(!HasCudaGPU()){
		std::cerr << "No GPU found, this will be launched in CPU mode" << std::endl;
		m_DeviceType=CPU;
		return false;
	}
	m_Predict_net.mutable_device_option()->set_device_type(CUDA);
	m_Init_net.mutable_device_option()->set_device_type(CUDA);

	// set input data with put data to tensor according to deviceType
	
	// TODO: check is gpu ID available
	if(m_bUseDefaultGPUDeviceOption){ // default runAllOnGPU on gpu 0
		m_Predict_net.mutable_device_option()->set_cuda_gpu_id(0);
		m_Init_net.mutable_device_option()->set_cuda_gpu_id(0);
	}
	
	// record the deviceOption for data blob, in order to know which GPU or TENSOR type it needs
	for(int i=0; i<m_Predict_net.op_size(); ++i){
		if(m_bUseDefaultGPUDeviceOption){
			m_Predict_net.mutable_op(i)->mutable_device_option()->set_cuda_gpu_id(0);
			m_Predict_net.mutable_op(i)->mutable_device_option()->set_device_type(CUDA);
		}
		for(int j=0;j< m_Predict_net.op(i).input_size(); ++j){
			const string &inputBlobName = m_Predict_net.op(i).input(j);
			if(m_BlobOptionMap.find(inputBlobName) == m_BlobOptionMap.end()){
				m_BlobOptionMap[inputBlobName] = m_Predict_net.op(i).device_option();
			}
		}
	}
	return true; 
}
bool Caffe2Handler::enableCPU()
{
	// no need to specify deviceID
	m_Predict_net.mutable_device_option()->set_device_type(CPU);
	m_Init_net.mutable_device_option()->set_device_type(CPU);
	for(int i=0; i<m_Predict_net.op_size(); ++i){
		m_Predict_net.mutable_op(i)->mutable_device_option()->set_device_type(CPU);
			
	}
	for(int i=0; i<m_Init_net.op_size(); ++i){
		m_Init_net.mutable_op(i)->mutable_device_option()->set_device_type(CPU);	
	}

}


void Caffe2Handler::initializeNetwork()
{
	if(m_DeviceType == CUDA){ enableCUDA(); }
	else if(m_DeviceType == CPU) { enableCPU(); } 
	
	//cout << m_Predict_net.DebugString() << endl;
	CAFFE_ENFORCE(m_Workspace.RunNetOnce(m_Init_net));
	CAFFE_ENFORCE(m_Workspace.CreateNet(m_Predict_net));
}

void Caffe2Handler::forward()
{
	m_Workspace.RunNet(m_Predict_net.name());
}

