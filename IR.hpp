
#include <vector>
#include <unordered_map>


#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
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
#include "caffe2/core/operator.h"

using namespace std;
using namespace caffe2;

struct OpSSA{
	OperatorDef op;
	unordered_map<string, int> in_versions;
	unordered_map<string, int> out_versions;
	OpSSA(OperatorDef op, unordered_map<string, int> in_versions, unordered_map<string, int> out_versions){
		this->op = op;
		this->in_versions = in_versions;
		this->out_versions = out_versions;
	}
	
};

struct GradientSlice{
	string indices;
	string values;
	GradientSlice(string indices, string values){
		this->indices = indices;
		this->values = values;
	}
	
};

struct GradGenMeta{
	OperatorDef *grad_op;
	int idx;
	string gradient;

	GradGenMeta(OperatorDef *grad_op, int idx, string gradient)
	{
		this->grad_op = grad_op;
		this->gradient = gradient;
		this->idx = idx;
	}
};


typedef pair<unordered_map<string, string>, vector<OperatorDef> > GradientOpsPair;
typedef pair<OperatorDef, int> OpHistory;
typedef unordered_map<int, vector<int>> UsageMapping;
typedef unordered_map<int, vector<GradGenMeta>> GradientGeneratorMapping;

class IR{
	
	public:
		IR(const google::protobuf::RepeatedPtrField<caffe2::OperatorDef>& vOps){
			for( auto op : vOps ){
				play(op);
			}
			sanityCheck(vOps);
			m_vAllGradientOps.resize(10000); // will fix this latter
		}
		vector<OperatorDef> GetBackwardPass(unordered_map<string, string>& ys);
		inline unordered_map<string, string> getAllInputToGrad(){ return m_mAllInputToGrad; }
		
		
	private:
		void play(OperatorDef& op);
		void sanityCheck(const google::protobuf::RepeatedPtrField<caffe2::OperatorDef>& vOps);
		void GenerateGradientsForForwardOp(int forwardOpIndex);
		void DoGradientAccumulation(int forward_op_index);
		
		bool VerifyGradientGenerators(const vector<GradGenMeta> &vGradGenMeta);
		bool deviceOptionEqual(const DeviceOption& opt1, const DeviceOption& opt2);
		pair<OperatorDef, string> MakeSumOps(const string inputName, const int inputVersion);
		string GetSumOpOutputName(vector<GradGenMeta>& vGradGenMeta, string inputName);
		pair<OperatorDef, string> MakeDenseSumOps(vector<GradGenMeta>& vGradGenMeta, string outBaseName);
		string DisambiguateGradOpOutput(OperatorDef *gradOp, int index, int count);
		void CheckSumOpsConflict(string outBaseName, string gName);
		void SetSumOpsDeviceOption(OperatorDef &sumOp, vector<GradGenMeta>& vGradGenMeta);
		
		
		OperatorDef genAutoGradOp(string ysKey);
		GradientWrapper from_untyped(string &blobName);
		GradientOpsMeta GetGradientMetaForOp(const OperatorDef& op, vector<string> &gradOutputBlob);
		void BuildGradientGenerators(int forwardOpIndex, vector<OperatorDef*> &vGradientOps, vector<string> gradOutput, vector<GradientWrapper> &vGradWrapperInputs);
		
		
		// CheckGradientOperatorInput
		void CheckGradientOperatorInput(string gradInputBlob, const vector<string>& gradOutput, int forwardOpIndex, vector<string> vLocallyGeneratedBlobName);
		int  GetIndexFromGradientList(const vector<string>& vGradOutput, string grad_op_input);
		string versionMismatchInfoOut(string name);
		string versionMismatchInfoIn(string name);
		
		// TODO 
		void GetInitGradient(unordered_map<string, string>& ys);
		
		//
		
		
		// **assume #gradient_ops < 10000, due to some pointer memory issues. TODO: find someway to fix 
		vector<OperatorDef> m_vAllGradientOps;
		unordered_map<string, string> m_mAllInputToGrad;
		
		
		
		
		unordered_map<string, GradientGeneratorMapping> m_GradientGenerator;
		unordered_map<string, int> m_mGradientFrontier;
		unordered_map<string, int> m_mFrontier;
		unordered_map<string, UsageMapping> m_mInput_usages;
		vector<OpSSA> m_vSSA;
		unordered_map<string, vector<OpHistory>> m_nIn_version_history;
		unordered_map<string, vector<OpHistory>> m_nOut_version_history;
};