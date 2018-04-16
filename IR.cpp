#include "IR.hpp"

void IR::play(OperatorDef& op)
{
	
	unordered_map<string, int> in_versions;
	for( string s : op.input()){
		in_versions[s] = m_mFrontier[s];
		m_mInput_usages[s][m_mFrontier[s]].push_back(m_vSSA.size());
		m_nIn_version_history[s].push_back(pair<OperatorDef, int>(op, m_mFrontier[s]));
	}
	
	unordered_map<string, int> out_versions;
	for( string s : op.output()){
		if(m_mFrontier.find(s) != m_mFrontier.end()){
			m_mFrontier[s]+=1;
		}
	
		out_versions[s] = m_mFrontier[s];
		m_nOut_version_history[s].push_back(OpHistory(op, m_mFrontier[s]));
	}
	m_vSSA.push_back(OpSSA(op, in_versions, out_versions));
	
	
}

void IR::sanityCheck(const google::protobuf::RepeatedPtrField<caffe2::OperatorDef>& vOps)
{
	for(OperatorDef op: vOps){
		if(op.type() == "StopGradient"){
			if( m_mInput_usages.find(op.output(0)) == m_mInput_usages.end()){
				cerr << "StopGradient's output " << op.output(0) << " is orphan. You typically want to specify same input and output for StopGradient. Op:\n\n" << op.type() << endl;	
				exit(1);
			}		
		}
	}	
}

OperatorDef IR::genAutoGradOp(string ysKey)
{
	OperatorDef op;
	op.set_type("ConstantFill");
	op.add_input(ysKey);
	op.add_output(ysKey + "_grad");
	op.set_engine("CUDNN");
	op.mutable_device_option()->set_device_type(CUDA);
	op.mutable_device_option()->set_cuda_gpu_id(0);
	AddArgument<int>("value", 1, &op);
	return op;
	
}



void IR::GetInitGradient(unordered_map<string, string>& ys)
{
	vector<OperatorDef> gradient_ops;
	for(unordered_map<string, string>::iterator it= ys.begin() ; it != ys.end(); it++){
		OperatorDef op;
		if(it->second == ""){
			//TODO, handle if grad part is not defined
			;
		}
		// just follow the python code, and not consider GradientSlice case now
		m_mAllInputToGrad[it->first] = it->second;
		
		// dont consider autograd gen now, should be empty now

	}

	m_vAllGradientOps = gradient_ops;	
}

GradientWrapper IR::from_untyped(string &blobName)
{
	GradientWrapper w;
	if(blobName == "") return w;
	// Dont consider GradientSlice case now (which, for our net, is the case.)
	// try ...
	w.dense_ = blobName;
	return w;

}

GradientOpsMeta IR::GetGradientMetaForOp(const OperatorDef& op, vector<string> &gradOutputBlob)
{
	
	vector<GradientWrapper> gWrapOutput;
	for( string &blobName : gradOutputBlob){
		gWrapOutput.push_back(from_untyped(blobName));
	}
	return GetGradientForOp(op, gWrapOutput);
}

void IR::DoGradientAccumulation(int forward_op_index)
{
	vector<OperatorDef> additional_sum_ops;
	OperatorDef& forward_op = m_vSSA[forward_op_index].op;
	unordered_map<string, int> &in_versions = m_vSSA[forward_op_index].in_versions;
	unordered_map<string, string> grad_map;
	vector<OperatorDef> additionalSumOps;
	set<string> opInputsSet;
	for(string inputName : forward_op.input()){
		opInputsSet.insert(inputName);
	}
	for(set<string>::iterator it=opInputsSet.begin(); it!=opInputsSet.end(); ++it){
		int inputVersion = in_versions[*it];
		vector<int> inputUsage = m_mInput_usages[*it][inputVersion];
		if(inputUsage.size() <= 1 || forward_op_index != inputUsage[0]){
			continue;
		}
		
		vector<GradGenMeta> vGradGenMeta = m_GradientGenerator[*it][inputVersion];
		
		if(!VerifyGradientGenerators(vGradGenMeta)){
			continue;
		}
		// dont want to throw exception now, if needed, will add here, for now, exit in where it thrown an exception
		
			
		pair<OperatorDef, string> sumOpOutNamePair = MakeSumOps(*it, inputVersion);
		additionalSumOps.push_back(sumOpOutNamePair.first);
		grad_map[*it] = sumOpOutNamePair.second;
	}

	
	for(unordered_map<string, string>::iterator it=grad_map.begin(); it!=grad_map.end(); ++it){
		m_mAllInputToGrad[it->first] = it->second;
	}

	for(int i=0;i<additionalSumOps.size(); ++i){
		m_vAllGradientOps.push_back(additionalSumOps[i]);
	}
}

pair<OperatorDef, string> IR::MakeSumOps(const string inputName, const int inputVersion)
{
	// skip sparse gradGenMeta (TODO)
	vector<GradGenMeta> &vGradGenMeta = m_GradientGenerator[inputName][inputVersion];
	string outBaseName = GetSumOpOutputName(vGradGenMeta, inputName);
	pair<OperatorDef, string> sumOpOutNamePair = MakeDenseSumOps(vGradGenMeta, outBaseName);
	SetSumOpsDeviceOption(sumOpOutNamePair.first, vGradGenMeta);
	return sumOpOutNamePair;
	
}
string IR::DisambiguateGradOpOutput(OperatorDef *gradOp, int index, int count)
{
	string newOutputName = "_" + gradOp->output(index) + "_autosplit_" + to_string(count);
	gradOp->set_output(index, newOutputName);
	return newOutputName;
	
}
void IR::CheckSumOpsConflict(string outBaseName, string gName)
{
	if(outBaseName == gName){
		cerr << "The gradient output of empty gradient op can not be the same as the normal name of the current input gradient." << endl;
		exit(1);
	}
}

void IR::SetSumOpsDeviceOption(OperatorDef &sumOp, vector<GradGenMeta>& vGradGenMeta)
{
	for(GradGenMeta gMeta : vGradGenMeta){
		OperatorDef *gradOp = gMeta.grad_op;
		if(gMeta.grad_op != nullptr){
			if(gradOp->has_device_option()){
				sumOp.mutable_device_option()->CopyFrom(gradOp->device_option());
			}	
		}
	}
}
pair<OperatorDef, string> IR::MakeDenseSumOps(vector<GradGenMeta>& vGradGenMeta, string outBaseName)
{
	vector<string> vSumOpInputs;
	int count=0;
	// assert(len(generators) > 1 )
	bool firstGradOp = true;
	for(GradGenMeta gMeta : vGradGenMeta){
		OperatorDef *gradOp = gMeta.grad_op;
		int idx = gMeta.idx;
		string gradName = gMeta.gradient;
        //assert(this is not GradSlice)
		string outName;
		
		if(gMeta.grad_op != nullptr){
			if(firstGradOp){
				firstGradOp = false;
				outName = gradOp->output(idx);
			}
			else{
				outName = DisambiguateGradOpOutput(gradOp, idx, count);
				count++;
			}
			vSumOpInputs.push_back(outName);
		}
		else{
			CheckSumOpsConflict(outBaseName, gradName);
			vSumOpInputs.push_back(gradName);
		}
	}
	int outBaseNameIndex = GetIndexFromGradientList(vSumOpInputs, outBaseName);
	if(outBaseNameIndex != -1){
		 // Sum inplace mode works only for the first input
         // So we do a swap
		swap(vSumOpInputs[0],vSumOpInputs[outBaseNameIndex]);
		// why need to swap? check
	}
	OperatorDef sumOp;
	sumOp.set_type("Sum");
	sumOp.set_name("");
	for(string inputName : vSumOpInputs){
		sumOp.add_input(inputName);
	}
	sumOp.add_output(outBaseName);
	return pair<OperatorDef, string>(sumOp, outBaseName);
}

string IR::GetSumOpOutputName(vector<GradGenMeta>& vGradGenMeta, string inputName)
{
	// skip sparse gradGenMeta and suffix (TODO)
	for( GradGenMeta g : vGradGenMeta){
		if(g.grad_op != nullptr){
			return g.grad_op->output(g.idx);
		}
	}
	return inputName + "_grad";
}


bool IR::deviceOptionEqual(const DeviceOption& opt1, const DeviceOption& opt2)
{
	//if not opt1 or not opt2: ignore here first, check later
    //   return opt1 == opt2
	
	// ignore "ignore_node_name... " since we dont consider when add training op
	
	// at least one option is CPU, check if both are for CPU
   if(!opt1.device_type() || !opt2.device_type()){
	   
	   return (!opt1.device_type() && !opt2.device_type());
   }
 
    return opt1.cuda_gpu_id() == opt2.cuda_gpu_id();
}

bool IR::VerifyGradientGenerators(const vector<GradGenMeta> &vGradGenMeta)
{
	// TODO sparse and dense gradient**
	
	if(vGradGenMeta.size() < 2) return false;
	vector<string> all_gradient_names;
	vector<DeviceOption> all_device_options;
	
	for( GradGenMeta gradGenMeta : vGradGenMeta){
		if(gradGenMeta.grad_op != nullptr){
			all_gradient_names.push_back(gradGenMeta.gradient);
			all_device_options.push_back(gradGenMeta.grad_op->device_option());
			if(!deviceOptionEqual(all_device_options.back(), all_device_options[0])){
				cerr << "Unexpected behavior: not all grad ops have the same device option." << endl;
				exit(1);
			}
		}
	}
	return true;
}


void IR::GenerateGradientsForForwardOp(int forwardOpIndex)
{
	unordered_map<string, string> new_input_to_grad;
	const OperatorDef& forward_op = m_vSSA[forwardOpIndex].op;
	vector<string> g_output;
	bool isAllGradNone = true;

	for(string blobName : forward_op.output()){
		if(m_mAllInputToGrad.find(blobName) == m_mAllInputToGrad.end()){ // not need to do grad
			g_output.push_back(""); // empty string for "None", check if this is Okay
			
		}
		else{
			isAllGradNone = false;
			g_output.push_back(m_mAllInputToGrad[blobName]);
		}
	}
	
	GradientOpsMeta opsMeta;
	if(!isAllGradNone || (forward_op.type() == "ZeroGradient")){		
		//_GetGradientForOpCC, skip Exception here
		// Since we directly use C++ API for finding GradientOpsMeta, we replace the data processing part in python by simple one line, not that g_input_ is GradientWrapper
		opsMeta = GetGradientMetaForOp(forward_op, g_output);
		
		vector<OperatorDef*> tmp;
		for(int i=0;i<opsMeta.ops_.size(); ++i){
			m_vAllGradientOps.push_back(opsMeta.ops_[i]);
			tmp.push_back(&(m_vAllGradientOps.back()));
		}
	
		BuildGradientGenerators(forwardOpIndex, tmp, g_output, opsMeta.g_input_);
		// transform to vector from google protobud repeated field temperarily, do it better latter
		vector<string> vForwardInputBlobs;
		vector<string> vForwardOutputBlobs; 
			
		for( string blobName : forward_op.input()){
			vForwardInputBlobs.push_back(blobName);
			
		}
		for( string blobName : forward_op.output()){
			vForwardOutputBlobs.push_back(blobName);
		}	
		// will this always be the same size? at least in dense grad.	
		for(int i=0;i< vForwardInputBlobs.size(); ++i){
			string name = vForwardInputBlobs[i];
			string grad = opsMeta.g_input_[i].dense_;
			if(grad != "" 
					|| m_mAllInputToGrad.find(name) == m_mAllInputToGrad.end() 
					|| GetIndexFromGradientList(vForwardOutputBlobs, name) != -1){
				new_input_to_grad[name] = grad;
			}
		}	 
	}
	// need to move m_vGenerator's pointer into global GradientOps since it currently points to opsMeta; but that memory may soon be erased before the next function call
	// update to global inputToGrad
	for(unordered_map<string, string>::iterator it=new_input_to_grad.begin(); it!=new_input_to_grad.end(); ++it){
		m_mAllInputToGrad[it->first] = it->second;
	}
}

int IR::GetIndexFromGradientList(const vector<string>& vGradOutput, string grad_op_input)
{
	for(int i=0; i<vGradOutput.size(); ++i){
		if(vGradOutput[i] == grad_op_input) return i;
	}
	return -1;
}

string IR::versionMismatchInfoOut(string name)
{
	string s;
	s=  "DEBUG HELP: \n" 
     +	string("Maybe you use same output blob twice for different ops?\n") 
	 +  string("== Version history of blob [")
	 +  name 
	 +  string("]\n") ;
	 vector<OpHistory> &vOpHistory = m_nOut_version_history[name];
	 for(int i=0;i<vOpHistory.size(); ++i){
		 s += "Version (out) " + vOpHistory[i].first.type() + " <-- " + to_string(vOpHistory[i].second) + "\n";
	 }
	 
	return s;
}
string IR::versionMismatchInfoIn(string name)
{
	string s;
	s=  "DEBUG HELP: \n" 
     +	string("Maybe you use same output blob twice for different ops?\n") 
	 +  string("== Version history of blob [")
	 +  name 
	 +  string("]\n") ;
	 vector<OpHistory> &vOpHistory = m_nOut_version_history[name];
	 for(int i=0;i<vOpHistory.size(); ++i){
		 s += "Version (out) " + vOpHistory[i].first.type() + " <-- " + to_string(vOpHistory[i].second) + "\n";
	 }
	 
	return s;
}

void IR::CheckGradientOperatorInput(string gradInputBlob, const vector<string>& gradOutput, int forwardOpIndex, vector<string> vLocallyGeneratedBlobs)
{
	OpSSA &opssa = m_vSSA[forwardOpIndex];
	int original_index = GetIndexFromGradientList(gradOutput, gradInputBlob);

	if(original_index != -1){
		string original_name = opssa.op.output(original_index);
		if (opssa.out_versions[original_name] != m_mGradientFrontier[original_name]){
			cerr << "Gradient name " << gradInputBlob << " is expected to correspond to version " << opssa.out_versions[original_name] 
			<< " of " << "original_name, but currently we have version " << m_mGradientFrontier[original_name] << ".\n\n'";
			cerr << versionMismatchInfoIn(original_name);
			exit(1);
		}
	}
	else if(opssa.out_versions.find(gradInputBlob) != opssa.out_versions.end()){
		if( m_mFrontier[gradInputBlob] != opssa.out_versions[gradInputBlob]){
			cerr << 
				"Gradient operator needs output " << gradInputBlob << 
				" at version " << opssa.out_versions[gradInputBlob] << ", but currently we have version " 
				<< m_mFrontier[gradInputBlob] << ".\n\n";
			cerr << versionMismatchInfoOut(gradInputBlob);
			exit(1);
		}
	}
        // If it is an input name, the current version should match the
        // version when the operator was run.
	else if(opssa.in_versions.find(gradInputBlob) != opssa.in_versions.end()){
		if (m_mFrontier[gradInputBlob] != opssa.in_versions[gradInputBlob]){
				cerr << "Gradient operator needs input " << gradInputBlob << " at version " << opssa.in_versions[gradInputBlob]
				<< ", but currently we have version " << m_mFrontier[gradInputBlob] << ".\n\n";
				cerr << versionMismatchInfoIn(gradInputBlob);
		}
	}			
	else{
		if(GetIndexFromGradientList(vLocallyGeneratedBlobs, gradInputBlob) == -1){
				cerr << "Blob name " << gradInputBlob << " not in the scope of operator: " << opssa.op.name() 
				<< "\n and is not generated by any of the local gradient operator. ";
		}
	}
}

void IR::BuildGradientGenerators(int forwardOpIndex, vector<OperatorDef*> &vGradientOps, vector<string> gradOutput, vector<GradientWrapper> &vGradWrapperInputs)
{
	OpSSA &opssa = m_vSSA[forwardOpIndex];
	OperatorDef &forward_op = opssa.op;
	unordered_map<string, int> &in_versions = opssa.in_versions;
	vector<string> vLocallyGeneratedBlobs;
	
	// get wrapper's string
	vector<string> vGradBlobInputs;
	for(int i=0;i<vGradWrapperInputs.size(); ++i){
		vGradBlobInputs.push_back(vGradWrapperInputs[i].dense_);
	}
	for(OperatorDef *grad_op : vGradientOps){
		// check inputs are valid
		for(string inputBlob : grad_op->input()){
			//cout << "input " << inputBlob << endl;
			CheckGradientOperatorInput(inputBlob, gradOutput, forwardOpIndex, vLocallyGeneratedBlobs);
		}
		
		for(string outputBlob : grad_op->output()){
			//cout << "output " << outputBlob << endl;
			vLocallyGeneratedBlobs.push_back(outputBlob);
		}	
		int grad_op_order=0;
		for(string outputBlob : grad_op->output()){
			int input_index = GetIndexFromGradientList(vGradBlobInputs, outputBlob);
			if(input_index != -1){
				string input_name = forward_op.input(input_index);
				int in_version = in_versions[input_name];
				string g = vGradBlobInputs[input_index];
				// skip GradientSlice part, go to else
				m_GradientGenerator[input_name][in_version].push_back(GradGenMeta(grad_op, grad_op_order, g));
			}
			grad_op_order++;
		}	
	}
	// remember if handle gradientSlice type, we need to add sparseGenerator here
	int g_input_order=0;
	for(string gradInputName : vGradBlobInputs){
		string input_name = forward_op.input(g_input_order);
		int input_version = in_versions[input_name];
		if(gradInputName == ""){ g_input_order++; continue;}  // not g
		
		// again, skip GradientSlice part
		int idx = GetIndexFromGradientList(vLocallyGeneratedBlobs, gradInputName);
		if(idx == -1){
			m_GradientGenerator[input_name][input_version].push_back(GradGenMeta(nullptr, 0, gradInputName));
		}
		g_input_order++;
	}
	
	g_input_order=0;
	for(string gradInputName : vGradBlobInputs){
		if(gradInputName != ""){
			string input_name = forward_op.input(g_input_order);
			int input_version = in_versions[input_name];
			m_mGradientFrontier[input_name] = input_version;
		}
		g_input_order++;
	}
}


vector<OperatorDef> IR::GetBackwardPass(unordered_map<string, string>& ys)
{
	for(unordered_map<string, string>::iterator it= ys.begin() ; it != ys.end(); it++){
		m_mGradientFrontier[it->first] = m_mFrontier[it->first];
		m_mInput_usages[it->first][m_mFrontier[it->first]].push_back(m_vSSA.size());
	}
	
	//construct the gradMap (m_allInputToGrad) and gradient ops (m_vAllGradientOps)
	GetInitGradient(ys);
	for(int i=m_vSSA.size()-1;i>=0;--i){
		GenerateGradientsForForwardOp(i) ;
        DoGradientAccumulation(i);
	}

	return m_vAllGradientOps;
}