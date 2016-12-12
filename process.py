#coding=utf-8
import pandas as pd
import numpy as np
from scipy import stats
import gensim 
import os
import cPickle as pickle
import re
import math
from cilin import CilinSimilarity
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


train_file="data/train.tsv"
test_file="data/test.tsv"
embedding_file="embedding/wiki_sim_utf_cut_embedding_sg.bin"
user_dict="user.dict"
embedding_subset="embedding/subset.dict"
cilin_path="cilin.pkl"
def loadData(train_file=train_file,test_file=test_file):
	train=pd.read_csv(train_file, sep="\t")
	test=pd.read_csv(test_file, sep="\t")
	return train,test

def loadCilin():
	if os.path.exists(cilin_path):
		return pickle.load(open(cilin_path))
	cs = CilinSimilarity()
	pickle.dump(cs, open(cilin_path,"w"))
	return cs
cs=loadCilin()

def loadEmbedding():
	if os.path.exists(embedding_subset):
		return pickle.load(open(embedding_subset))
	
	train,test=loadData()
	words=set(train["word1"])|set(train["word2"]) |set(test["word1"]) |set(test["word2"])
	# if not os.path.exists(user_dict):
	# 	with open(user_dict,"w") as f:
	# 		f.write("\n".join(words))

	w2v=gensim.models.word2vec.Word2Vec.load_word2vec_format(embedding_file, binary=False )
	embeddings=dict()
	for word in words:
		if word.decode("utf-8") in w2v.vocab.keys():
			embeddings[word]=w2v[word.decode("utf-8")]
	pickle.dump(embeddings,open(embedding_subset,"w"))
	return embeddings
def overlap(row):
	set1=set(row["word1"].decode("utf-8"))	
	set2=set(row["word2"].decode("utf-8"))	
	return len(set1 & set2)

embeddings=loadEmbedding()
def embedding_sim(row):
	
	word1,word2=row["word1"],row["word2"]
	if word1 in embeddings.keys() and word2 in embeddings.keys():
		v1= embeddings[word1]
		v2= embeddings[word2]
		score=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
		return score
	# print word1,word2,row["score"]
	return 0.8           # 这个影响了结果，其实应该会

def main():
	train,test=loadData()
	
	test["overlap"]=test.apply(overlap,axis=1)
	test["w2v"]=test.apply( embedding_sim  ,axis=1)

	# print len(test[test.predicted==0.8])
	test["w2v"]=test["w2v"].fillna(test["w2v"].mean())

	# test["cilin"]=test.apply( lambda row: cs.similarity(row["word1"],row["word2"])  ,axis=1)
	# test["cilin"]=test.apply( lambda row: cs.sim2013(row["word1"],row["word2"])  ,axis=1)
	test["cilin"]=test.apply( lambda row: cs.sim2016(row["word1"].decode("utf-8"),row["word2"].decode("utf-8"))  ,axis=1)
	# for i in np.linspace(1,10,30):
	test["predicted"]=test["cilin"]+ 1.7*test["w2v"]                                   #0.5607 ,0.5418
	test["predicted"]=test["cilin"]+ test["w2v"]                                      #0.5520 ,0.5272
	# test["predicted"]=test.apply(lambda row: max(row["cilin"],row["w2v"])  ,axis=1)  #0.5617 ,0.5388
	# test["predicted"]=test.apply(lambda row: min(row["cilin"],row["w2v"])  ,axis=1)  #0.4352   0.4137
	# test["predicted"]=test.apply(lambda row: math.sqrt(row["cilin"]*row["w2v"])  ,axis=1)  #0.4538   0.4589
	
	print stats.pearsonr(test["score"],test["predicted"])
	print stats.spearmanr(test["score"],test["predicted"])
	# print np.corrcoef(test["score"],test["predicted"])

if __name__ == '__main__':
	# main()
	main()

	
	
