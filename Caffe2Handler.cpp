#include "Caffe2Handler.hpp"

#include <iostream>

void Caffe2Handler::loadNetworkProto( string sInitNetName, string sPredictNetName, bool isCUDA, int gpuID)
{
	
	
	
	
	
	CAFFE_ENFORCE(ReadProtoFromFile(sInitNetName, &m_Init_net));
	CAFFE_ENFORCE(ReadProtoFromFile(sPredictNetName, &m_Predict_net));
	if(isCUDA){ enableCUDA(gpuID); }
	else { enableCPU(); }
	/*
	for(int i=0;i < m_Predict_net.op_size(); ++i){
		if ("Conv" == m_Predict_net.op(i).type()){
		m_Predict_net.mutable_op(i)->set_engine("NNPACK");
		//std::cout << "Conv engine used:" << predict_net.op(i).has_engine() << std::endl;
		//std::cout << "Conv engine set:" << predict_net.op(i).engine() << std::endl;
		}
	}
	*/
	cout << m_Predict_net.DebugString() << endl;
	
	

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

bool Caffe2Handler::enableCUDA(uint gpuID)
{
	if(!HasCudaGPU()){
		std::cerr << "No GPU found, this will be launched in CPU mode" << std::endl;
		return false;
		// MKLDNN?
	}
	// TODO: check is gpu ID available
	m_Predict_net.mutable_device_option()->set_device_type(CUDA);
	m_Init_net.mutable_device_option()->set_device_type(CUDA);
	m_Predict_net.mutable_device_option()->set_cuda_gpu_id(gpuID);
	m_Init_net.mutable_device_option()->set_cuda_gpu_id(gpuID);
	m_DeviceType = CUDA;
	return true; 
}
bool Caffe2Handler::enableCPU()
{
	m_Predict_net.mutable_device_option()->set_device_type(CPU);
	m_Init_net.mutable_device_option()->set_device_type(CPU);
	m_DeviceType = CPU;
}

void Caffe2Handler::initializeNetwork()
{
	CAFFE_ENFORCE(m_Workspace.RunNetOnce(m_Init_net));
	CAFFE_ENFORCE(m_Workspace.CreateNet(m_Predict_net));
}

void Caffe2Handler::forward()
{
	m_Workspace.RunNet(m_Predict_net.name());
}

