#include "Caffe2TrainNet.hpp"


void Caffe2TrainNet::recordNetworkBlobParam()
{
	for(OperatorDef op : m_Predict_net.op()){
		for( string input : op.input()){
			m_vBlobParam.push_back(input);		
		}
	}
}

void Caffe2TrainNet::inferBlobDevice()
{
	// TODO: for init_net
	unordered_map<string, DeviceOption> mapping;
	
	for(OperatorDef op : m_Predict_net.op()){
		//cout << "OP: " << op.type() << " :" << endl;
		DeviceOption deviceOption = op.device_option();
		if(op.type() == "Iter"){
			deviceOption.set_device_type(CPU);
		}
		for(string blob : op.input()){
			if(mapping.find(blob) == mapping.end()){
				mapping[blob] = deviceOption;
				//cout << blob << "<-- " << deviceOption.device_type() << " " << deviceOption.cuda_gpu_id() << endl;;
			}			
		}
		for(string blob : op.output()){
			if(mapping.find(blob) == mapping.end()){
				mapping[blob] = deviceOption;
				//cout << blob << "--> " << deviceOption.device_type() << " " << deviceOption.cuda_gpu_id() << endl;;
			}			
		}
		
	}
	m_mBlobToDevice = mapping;
}

bool Caffe2TrainNet::isParamGPU(string param)
{
	// not consider CPU just now, (TODO)
	/*
	if(m_mBlobToDevice.find(param) != m_mBlobToDevice.end()){
		return m_mBlobToDevice[param].device_type() == CUDA;
	}
	else{
		param = "gpu_" + m_vGpuList[0] + "/" + param;
		if(m_mBlobToDevice.find(param) == m_mBlobToDevice.end()){
			return m_mBlobToDevice[param].device_type() == CUDA;
		}
	}
	*/
	
	return true;
}

void Caffe2TrainNet::broadcastFromMasterGPU(const string& param)
{
	int masterDevice = m_vGpuList[0];
	// TODO: if nccl
	
	for(int devIndex=1; devIndex< m_vGpuList.size(); ++devIndex){
		DeviceOption deviceOption;
		if(isParamGPU(param)){
			deviceOption.set_device_type(CUDA);
			deviceOption.set_cuda_gpu_id(devIndex);
		}
		else{
			deviceOption.set_device_type(CPU);
		}
		OperatorDef copyOperator = genOp("Copy", {m_mDeviceGrouped[param][masterDevice]}, {m_mDeviceGrouped[param][m_vGpuList[devIndex]]}, deviceOption.device_type(), deviceOption.cuda_gpu_id());
		m_Predict_net.add_op()->CopyFrom(copyOperator);
			
	}
}

unordered_map<string, string> Caffe2TrainNet::getParamToGrad()
{
	unordered_map<string, string> paramToGrad;
	if(!m_bIsAddGradientOperator){
		cerr << "Not added gradient operators yet. " <<endl;
		exit(1);
	}

	for(string param : m_sParams){
		if(m_mGradMap.find(param) != m_mGradMap.end()){
			paramToGrad[param] = m_mGradMap[param];
		}
	}
	return paramToGrad;
	
}
void Caffe2TrainNet::addGradientOperator(const vector<LossLayerData>& vLossLayerData)
{
	if(m_bIsAddGradientOperator){
		cerr << " cannot add Gradient operators twice" << endl;
		return ;
	}
	recordNetworkBlobParam();
	unordered_map<string, string> ys;
	// maybe check if the lossBlob in vector is valid while parsing in for loop, or just let IR give error.
	
	// I think it is unnecessary to add this into netDef, follow the original first
	for(int i=0;i<vLossLayerData.size();++i){
		for(int j=0;j<m_vGpuList.size(); ++j){
			OperatorDef lossGradOpdef = genOp("ConstantFill", {"gpu_" + to_string(m_vGpuList[j]) + "/" + vLossLayerData[i].outputs[0]}, {"gpu_" + to_string(m_vGpuList[j]) + "/" + vLossLayerData[i].outputs[0] + "_grad"}, CUDA, m_vGpuList[j]);
			AddArgument<float>("value", 1, &lossGradOpdef);
			m_Predict_net.add_op()->CopyFrom(lossGradOpdef);
			ys["gpu_" + to_string(m_vGpuList[j]) + "/" + vLossLayerData[i].outputs[0]] = "gpu_" + to_string(m_vGpuList[j]) + "/" + vLossLayerData[i].outputs[0] + "_grad";
		}
	}
	IR ir(m_Predict_net.op());
	vector<OperatorDef> vGradOps = ir.GetBackwardPass(ys);
	for(int i=0;i<vGradOps.size(); ++i){
		m_Predict_net.add_op()->CopyFrom(vGradOps[i]);
	}
	m_mGradMap = ir.getAllInputToGrad();
	
	// if call addGradientOperator, follow the user's definition
	m_bUseDefaultGPUDeviceOption=false;  
	m_bIsAddGradientOperator=true;
}

unordered_map<string, map<int , string>> Caffe2TrainNet::updateDeviceGrouped(const vector<int>& vDevices, const vector<string>& vParams)
{
	unordered_map<string,  map<int , string>> deviceGrouped;
	// not consider nonDataPtr and GradientSlice now, assume the format of a blob "must" be {}/{}, where the first part indicates device param and the second part is operator name
	for(string param : vParams){
		// get deviceParam. i.g. get the first part of a param  {}/{}
		string deviceParam = param.substr(0, param.find_first_of("/"));
		string operatorParam = param.substr(param.find_first_of("/")+1);
		// extract GPU ID from device Param, i.g. gpu_1 -> get 1
		int gpuID = stoi(deviceParam.substr(deviceParam.find_first_of("_")+1));

		m_mDeviceGrouped[operatorParam][gpuID] = param;
		deviceGrouped[operatorParam][gpuID] = param;
	}
	return deviceGrouped;
}


void Caffe2TrainNet::sumBetweenDevice(const string& param, unordered_map<int, string>& mGpuBlobsGroup, const vector<vector<bool>>& vP2PAccess, const vector<int>& deviceIndices)
{
	// only consider devicetype = CUDA now, actually I don't care CPU now
	vector<string> blobs;
	vector<int> devices;
	
	for(int device : deviceIndices){
		blobs.push_back(mGpuBlobsGroup[device]);
		devices.push_back(device);
	}
	for(int i=1;i<devices.size(); ++i){
		if(vP2PAccess.size() != 0 && !vP2PAccess[devices[0]][devices[i]]){
			OperatorDef copyOperator = genOp("Copy", {blobs[i]}, {"gpu_" + to_string(devices[0]) + "/" + param + "_gpu" + to_string(devices[i]) + "_copy"});
			m_Predict_net.add_op()->CopyFrom(copyOperator);
		}	
	}

	OperatorDef sumOperator = genOp("Sum", {blobs}, {blobs[0]}, CUDA, devices[0]);
	sumOperator.set_name("dpm");

	m_Predict_net.add_op()->CopyFrom(sumOperator);
}

void Caffe2TrainNet::allReduceBlobs(const vector<string>& blobNames)
{
	if(m_vGpuList.size() == 1) return;
	int masterDev = m_vGpuList[0];
	cout << "MasterDev: " << m_vGpuList[0] << endl;
	
	for(string blobName : blobNames){
		// not implement gradientSlice version and nccl
		unordered_map<int, string> mGpuBlobsGroup;
		for(auto& kv : m_mDeviceGrouped[blobName]){
			string deviceScope = kv.second.substr(0, kv.second.find_first_of("/"));
			int gpuID = stoi(deviceScope.substr(deviceScope.find_first_of("_")+1));
			mGpuBlobsGroup[gpuID] = kv.second;
		}
		if(mGpuBlobsGroup.size() == 1) continue;
		if(isParamGPU(blobName)){
			vector<vector<bool>> pattern;
			// allreduce
			GetCudaPeerAccessPattern(&pattern);
			
			if(m_vGpuList.size() == 16){
				for(int i=0; i<8; ++i){
					sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({i*2, i*2+1}));
				}
				for(int i=0; i<4; ++i){
					sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({i*4, i*4+2}));
				}
				for(int i=0; i<2; ++i){
					sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({i*8, i*8+4}));
				}
			}
			else if(m_vGpuList.size() == 8){
				for(int i=0; i<4; ++i){
					sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({i*2, i*2+1}));
				}
				for(int i=0; i<2; ++i){
					sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({i*4, i*4+2}));
				}
			}
			else if(m_vGpuList.size() == 4){
				sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({0,1}));
				sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({2,3}));
				sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, vector<int>({0,2}));
			}
			else{
				sumBetweenDevice(blobName, mGpuBlobsGroup, pattern, m_vGpuList);
			}
		}
		broadcastFromMasterGPU(blobName);
	}
}

void Caffe2TrainNet::parallelTrainingModel(const vector<int>& gpuList, const vector<LossLayerData>& vLossLayerData)
{
	
	NetDef pNet, pNetInit;
	pNet.set_type("dag");
	pNet.set_name(m_Predict_net.name());
	pNetInit.set_name(m_Init_net.name());
	
	// for non-paralleling part, copy the original predict and init net
	
	for(int j=0;j< gpuList.size();++j){
		for(int i=0;i<m_Predict_net.op_size() ; ++i){
				OperatorDef copy_op(m_Predict_net.op(i));
				for(int k=0;k< copy_op.input_size(); ++k){
					copy_op.set_input(k, "gpu_" + to_string(gpuList[j]) + "/" + m_Predict_net.op(i).input(k));
				}
				for(int k=0;k< copy_op.output_size(); ++k){
					copy_op.set_output(k, "gpu_" + to_string(gpuList[j]) + "/" + m_Predict_net.op(i).output(k));
				}
				copy_op.mutable_device_option()->set_cuda_gpu_id(gpuList[j]);
				copy_op.mutable_device_option()->set_device_type(CUDA);
				pNet.add_op()->CopyFrom(copy_op);
		}
		for(string externalInput: m_Predict_net.external_input()){
			pNet.add_external_input("gpu_" + to_string(gpuList[j]) + "/" +externalInput);
		}
	}
	for(int j=0;j< gpuList.size();++j){
		for(int i=0;i<m_Init_net.op_size() ; ++i){
				OperatorDef copy_op(m_Init_net.op(i));
				for(int k=0;k< copy_op.input_size(); ++k){
					copy_op.set_input(k, "gpu_" + to_string(gpuList[j]) + "/" + m_Init_net.op(i).input(k));
				}
				for(int k=0;k< copy_op.output_size(); ++k){
					copy_op.set_output(k, "gpu_" + to_string(gpuList[j]) + "/" + m_Init_net.op(i).output(k));
				}
				copy_op.mutable_device_option()->set_cuda_gpu_id(gpuList[j]);
				copy_op.mutable_device_option()->set_device_type(CUDA);
				pNetInit.add_op()->CopyFrom(copy_op);
		}
	}
	
	m_Init_net.CopyFrom(pNetInit);
	m_Predict_net.CopyFrom(pNet);
	m_bIsParallel=true;
	m_vGpuList = gpuList;
	
	// if call parallelize, must be under CUDA mode
	m_DeviceType = CUDA; // for later initialize
	collectNetParams();
	
	addGradientOperator(vLossLayerData);
	unordered_map<string, string> paramToGrad = getParamToGrad();
	vector<string> orderedGrad;
	for(string param : m_sParams){
		if(paramToGrad.find(param) != paramToGrad.end() && paramToGrad[param] != ""){
			orderedGrad.push_back(paramToGrad[param]);
		}
	}
	unordered_map<string, map<int , string>> gradsOrderedGroup = updateDeviceGrouped(gpuList, orderedGrad);
	unordered_map<string, map<int , string>> deviceGroupedBlobs  = updateDeviceGrouped(gpuList, m_sParams);
	unordered_map<string, map<int , string>> computedParamsGrouped  = updateDeviceGrouped(gpuList, m_sParamsComputed);
	
	vector<string> vParamsName, vComputedParamsName, vGradNames;
	for(auto& kv : deviceGroupedBlobs) vParamsName.push_back(kv.first);
	for(auto& kv : computedParamsGrouped) vComputedParamsName.push_back(kv.first);
	for(auto& kv : gradsOrderedGroup) vGradNames.push_back(kv.first);
	
	inferBlobDevice();
	// ==================== reduce parameter =====================

	for(string param : vComputedParamsName){
		broadcastFromMasterGPU(param);
	}
	
	 if(vGradNames.size() > 0){
		vector<string> vReverseGradNames = vGradNames;
		reverse(vReverseGradNames.begin(), vReverseGradNames.end()); 
		allReduceBlobs(vReverseGradNames);
	 }
	if(m_fWeightDecay > 0.0){ addWeightDecay(); }
	basicSettingBuilder();
	vector<string> vUpdateParams = paramUpdateBuilder();
	
	//====================== post processing ======================
	// sync param between devices for init_net 
	collectNetParams();
	unordered_map<string, map<int , string>> mUpdateParamsGrouped  = updateDeviceGrouped(gpuList, vUpdateParams);
	for(string param : vUpdateParams){
		m_Predict_net.add_external_input(param);
	}
	
	if(m_vGpuList.size() > 1){
		vector<string> paramForSync;
		for(auto& kv : deviceGroupedBlobs){
			paramForSync.push_back(kv.first);	
		}
		for(auto& kv : mUpdateParamsGrouped){
			paramForSync.push_back(kv.first);	
		}
		for(int i=1;i<m_vGpuList.size(); ++i){
			for(string param : paramForSync){
				OperatorDef syncDef = genOp("Copy", {"gpu_" + to_string(m_vGpuList[0]) + "/" + param}, {"gpu_" + to_string(m_vGpuList[i]) + "/" + param}, CUDA, m_vGpuList[i]);;
				m_Init_net.add_op()->CopyFrom(syncDef);
			}
		}
	}
	m_Predict_net.set_num_workers(m_vGpuList.size()*4);
	m_Predict_net.add_arg()->CopyFrom(MakeArgument<int>("first_iter_only_one_worker", 1));
}

void Caffe2TrainNet::addWeightDecay()
{
	unordered_map<string, string> paramToGrad = getParamToGrad();
	for(string param : m_sParamsWeights){
		// dirty way to ignore bias term
		if(param.substr(param.find_last_of("_")) == "_b") continue;
		string grad = paramToGrad[param];
		vector<string> inputs({grad, "gpu_" + to_string(m_mBlobToDevice[param].cuda_gpu_id()) + "/ONE", param, "gpu_" + to_string(m_mBlobToDevice[param].cuda_gpu_id()) + "/wd"});
		OperatorDef weightDecayDef = genOp("WeightedSum", inputs, {grad}, m_mBlobToDevice[param].device_type(), m_mBlobToDevice[param].cuda_gpu_id());
	
		m_Predict_net.add_op()->CopyFrom(weightDecayDef);
	}
	
	for(int i=0;i<m_vGpuList.size(); ++i){
		OperatorDef weightDecayInitDef = genOp("ConstantFill", {}, {"gpu_" + to_string(m_vGpuList[i]) + "/wd"}, CUDA, m_vGpuList[i]);
		AddArgument<vector<int>>("shape", vector<int>({1}) , &weightDecayInitDef);
		AddArgument<float>("value", m_fWeightDecay, &weightDecayInitDef);
	
		OperatorDef ONEDef = genOp("ConstantFill", {}, {"gpu_" + to_string(m_vGpuList[i]) + "/ONE"}, CUDA, m_vGpuList[i]);
		AddArgument<vector<int>>("shape", vector<int>({1}) , &ONEDef);
		AddArgument<float>("value", 1.0, &ONEDef);
	
		m_Init_net.add_op()->CopyFrom(weightDecayInitDef);
		m_Init_net.add_op()->CopyFrom(ONEDef);
		m_Predict_net.add_external_input("gpu_" + to_string(m_vGpuList[i]) + "/wd");
		m_Predict_net.add_external_input("gpu_" + to_string(m_vGpuList[i]) + "/ONE");
	}
}

void Caffe2TrainNet::basicSettingBuilder()
{
	for(int i=0;i<m_vGpuList.size(); ++i){
		OperatorDef iterDef = genOp("Iter", {"gpu_" + to_string(m_vGpuList[i]) + "/iter"}, {"gpu_" + to_string(m_vGpuList[i]) + "/iter"}, CPU, m_vGpuList[i]);
		AddArgument<int>("use_cudnn", 1, &iterDef);
		AddArgument<string>("order", "NCHW", &iterDef);
		AddArgument<int>("cudnn_exhaustive_search", 0, &iterDef);
		
		OperatorDef iterInitDef = genOp("ConstantFill", {}, {"gpu_" + to_string(m_vGpuList[i]) + "/iter"}, CPU, m_vGpuList[i]);
		AddArgument<int>("dtype", 10, &iterInitDef);
		AddArgument<int>("cudnn_exhaustive_search", 0, &iterInitDef);
		AddArgument<int>("value", 0, &iterInitDef);
		AddArgument<int>("use_cudnn", 1, &iterInitDef);
		AddArgument<vector<int>>("shape", vector<int>({1}) , &iterInitDef);
		AddArgument<string>("order", "NCHW", &iterInitDef);
		 	
		OperatorDef lrDef = genOp("LearningRate", {"gpu_" + to_string(m_vGpuList[i]) + "/iter"}, {"gpu_" + to_string(m_vGpuList[i]) + "/lr"}, CUDA, m_vGpuList[i]);
		AddArgument<string>("policy", "step", &lrDef);
		AddArgument<int>("stepsize", 1000000 , &lrDef);
		AddArgument<float>("base_lr", m_fLearningRate, &lrDef);
		AddArgument<float>("gamma", 0.1, &lrDef);
			
		m_Predict_net.add_external_input("gpu_" + to_string(m_vGpuList[i]) + "/iter");
		m_Predict_net.add_op()->CopyFrom(lrDef);
		m_Predict_net.add_op()->CopyFrom(iterDef);
		m_Init_net.add_op()->CopyFrom(iterInitDef);
	}
}

vector<string> Caffe2TrainNet::paramUpdateBuilder()
{
	vector<string> vInputParams;
	vector<string> vBuildParameters;
	vBuildParameters.insert(vBuildParameters.end(), m_sParamsWeights.begin(), m_sParamsWeights.end()); // update weight
	vBuildParameters.insert(vBuildParameters.end(), m_sParamsBias.begin(), m_sParamsBias.end()); // update bias
	unordered_map<string, string> paramToGrad = getParamToGrad();

	// update weight
	for(string param : vBuildParameters){
		string grad = paramToGrad[param];
		
		vector<string> inputs({grad, param + "_momentum", "gpu_" + to_string(m_mBlobToDevice[param].cuda_gpu_id()) + "/lr", param });
		vector<string> outputs({grad, param + "_momentum", param });
		OperatorDef momentumSGDUpdateDef = genOp("MomentumSGDUpdate", inputs, outputs, m_mBlobToDevice[param].device_type(), m_mBlobToDevice[param].cuda_gpu_id());
		AddArgument<int>("nesterov" , 1, &momentumSGDUpdateDef);
		AddArgument<float>("momentum", m_fMomentum, &momentumSGDUpdateDef);
		vInputParams.push_back(param + "_momentum");  // for external input
		
		OperatorDef momentumInitDef = genOp("ConstantFill", {param}, {param + "_momentum"}, m_mBlobToDevice[param].device_type(), m_mBlobToDevice[param].cuda_gpu_id());
		AddArgument<float>("value", 0.0, &momentumInitDef);
		
		m_Predict_net.add_external_input(param + "_momentum");
		m_Predict_net.add_op()->CopyFrom(momentumSGDUpdateDef);
		m_Init_net.add_op()->CopyFrom(momentumInitDef);	
	}
	return vInputParams;
}	

void Caffe2TrainNet::prepareTrainingDef()
{
	// copy from testing model
	m_Predict_net.CopyFrom(m_Predict_net);
}

void Caffe2TrainNet::addOperator(const OperatorDef& op)
{
	m_Predict_net.add_op()->CopyFrom(op);
}