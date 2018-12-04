from __future__ import division
import pandas as pd 
import subprocess
import platform,os
import sklearn
import numpy as np
qa_path="data/nlpcc-iccpol-2016.dbqa.testing-data"


def percisionAT1_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	rr=candidates[candidates["flag"]==1].index.min()+1
	if rr!=rr:
		return 0
	return 1.0 if rr==1 else 0.0
def mrr_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	rr=candidates[candidates["flag"]==1].index.min()+1
	if rr!=rr:
		return 0
	return 1.0/rr

def map_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	ap=0
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]==1]
	if len(correct_candidates)==0:
		return 0
	for i,index in enumerate(correct_candidates.index):
		ap+=1.0* (i+1) /(index+1)
	#print( ap/len(correct_candidates))
	return ap/len(correct_candidates)


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def evaluation_plus(modelfile, groundtruth=qa_path):
	answers=pd.read_csv(groundtruth,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	answers["score"]=pd.read_csv(modelfile,header=None,sep="\t",names=["score"],quoting =3)
	print( answers.groupby("question").apply(mrr_metric).mean())
	print( answers.groupby("question").apply(map_metric).mean())

def eval(predicted,groundtruth=qa_path, file_flag=False):
	if  'Windows' in platform.system() and file_flag ==False:
		modelfile=write2file(predicted)
		evaluationbyFile(modelfile)
		return 

	if type(groundtruth)!= str :
		answers=groundtruth
	else:
		answers=pd.read_csv(groundtruth,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	answers["score"]=predicted
	mrr= answers.groupby("question").apply(mrr_metric).mean()
	map= answers.groupby("question").apply(map_metric).mean()
	return map,mrr
def evaluate(predicted,groundtruth):
	filename=write2file(predicted)
	evaluationbyFile(filename,groundtruth=groundtruth)
def write2file(datas,filename="train.QApair.TJU_IR_QA.score"):
	with open(filename,"w") as f:
		for data in datas:
			f.write(("%.10f" %data )+"\n")
	return filename

def accurancy(df,predicted):
    label =  predicted>0.5
    return sum(label==df["flag"]) * 1.0 / len(df)

def evaluationbyFile(modelfile,resultfile="result.text",groundtruth=qa_path):
	cmd="test.exe " + " ".join([groundtruth,modelfile,resultfile])
	print( modelfile[19:-6]+":") # )
	subprocess.call(cmd, shell=True)
def evaluationBypandas(df,predicted,acc=False):
    df["score"]=predicted
    mrr= df.groupby("question").apply(mrr_metric).mean()
    map= df.groupby("question").apply(map_metric).mean()
    percsisionAT1= df.groupby("question").apply(percisionAT1_metric).mean()
    if acc:
        return map,mrr,percsisionAT1, accurancy(df,predicted)
    else:
        return map,mrr,percsisionAT1
def precision_per(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	rr=candidates[candidates["flag"]==1].index.min()
	if rr==0:
		return 1
	return 0
def precision(df,predicted):
	df["score"]=predicted
	precision = df.groupby("question").apply(precision_per).mean()
	return precision

def briany_test_file(df_test,  predicted=None,mode = 'test'):
	N = len(df_test)

	nnet_outdir = 'tmp/' + mode
	if not os.path.exists(nnet_outdir):
		os.makedirs(nnet_outdir)
	question2id=dict()
	for index,quesion in enumerate( df_test["question"].unique()):
		question2id[quesion]=index

	df_submission = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
	df_submission['qid'] =df_test.apply(lambda row: question2id[row['question']],axis=1)
	df_submission['iter'] = 0
	df_submission['docno'] = np.arange(N)
	df_submission['rank'] = 0
	if  predicted is None:
		df_submission['sim'] = df_test['score']
	else:
		df_submission['sim'] = predicted
	df_submission['run_id'] = 'nnet'
	df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), header=False, index=False, sep=' ')

	df_gold = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
	df_gold['qid'] = df_test.apply(lambda row: question2id[row['question']],axis=1)
	df_gold['iter'] = 0
	df_gold['docno'] = np.arange(N)
	df_gold['rel'] = df_test['flag']
	df_gold.to_csv(os.path.join(nnet_outdir, 'gold.txt'), header=False, index=False, sep=' ')

if __name__ =="__main__":
	data_dir="data/QA/"+"wiki"
	train_file=os.path.join(data_dir,"train.txt")
	test_file=os.path.join(data_dir,"test.txt")

	train=pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	train["score"]=np.random.randn(len(train))
	print(evaluationBypandas(train,train["score"]))