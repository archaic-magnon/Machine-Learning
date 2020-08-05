from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xclib.data import data_utils
import numpy as np
import scipy.sparse as spp
from scipy.sparse import csr_matrix
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm
from graphviz import Digraph
import time
import joblib, pickle
from joblib import Parallel,delayed
import random
import sys, os
from sklearn.model_selection import GridSearchCV
import itertools as it
import matplotlib.pyplot as plt



base_path = 'data/'

train_x_path = base_path + 'train_x.txt'
train_y_path = base_path + 'train_y.txt'

test_x_path = base_path + 'test_x.txt'
test_y_path = base_path + 'test_y.txt'

valid_x_path = base_path + 'valid_x.txt'
valid_y_path = base_path + 'valid_y.txt'


if len(sys.argv) == 1:
	pass

elif len(sys.argv) == 7:
	train_x_path = sys.argv[1] 
	train_y_path = sys.argv[2] 

	test_x_path = sys.argv[3]
	test_y_path = sys.argv[4]

	valid_x_path = sys.argv[5] 
	valid_y_path = sys.argv[6]
else:
	print("USAGE:\n")
	print("python3 Q_3_4.py")
	print("OR")
	print("python3 Q_3_4.py [train_x_path] [train_y_path] [test_x_path] [test_y_path] [valid_x_path] [valid_y_path]")

	raise("Either 0 or 6 Argument Expected")





###################################################################
#						RANDOM FOREST    					      #                         
###################################################################


def getAccuracy(original, predicted):
    original = np.array(original).reshape(-1,)
    predicted = np.array(predicted).reshape(-1,)

    if original.shape[0]!=predicted.shape[0]:
        raise("Unequal length")

    total = original.shape[0]
    correct = np.count_nonzero(original==predicted)

    return correct/total

def readFileSparse(path):
    return data_utils.read_sparse_file(path, header=True)

def readFile(x_path, y_path):
    x_data = np.array(csr_matrix.todense(readFileSparse(x_path)))
    y_data = np.array(pd.read_csv(y_path, header=None))
    return x_data, y_data



train_x, train_y = readFile(train_x_path, train_y_path)
test_x, test_y = readFile(test_x_path, test_y_path)
val_x, val_y = readFile(valid_x_path, valid_y_path)



class RandomForestParameter:
    def __init__(self, params):
        self.params = params
        self.params_comb = self.getCombination()
        self.best_params = None
        self.best_params_index = None
        self.best_score = None
        self.accuracy = []
        self.rfc = []
        self.time_to_fit = 0
        
    def getCombination(self):
        params = self.params
        comb_params = list(it.product(params['n_estimators'], params['max_features'], params['min_samples_split']))
        return comb_params

    def fit(self, train_x, train_y):
        s_time = time.time()
        for i, el in enumerate(self.params_comb):
            n_estimator, max_feature, min_sample_split = el
            print(str(i)+" iteration: n_estimators="+str(n_estimator)+" max_features="+str(max_feature)+" min_samples_split="+str(min_sample_split))
            clf = RandomForestClassifier(n_estimators=n_estimator, max_features=max_feature, min_samples_split=min_sample_split, oob_score=True, n_jobs=-1, verbose=0)
            clf.fit(train_x, train_y)
            self.rfc.append(clf)
            
            score = clf.score(train_x, train_y)
            self.accuracy.append(score)
        
        total_time = time.time()-s_time

        self.time_to_fit = total_time
        best_score_index = self.accuracy.index(max(self.accuracy))
        self.best_params_index = best_score_index
        self.best_score = self.accuracy[best_score_index]
        self.best_params = self.params_comb[best_score_index]




def Q3_scratch():

	params = {
	    'n_estimators': [50, 150, 250, 350, 450],
	    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
	    'min_samples_split': [2, 4, 6, 8, 10] ,
	}

	comb_params = list(it.product(params['n_estimators'], params['max_features'], params['min_samples_split']))

	rfp = RandomForestParameter(params)
	rfp.fit(train_x, train_y.reshape(-1,))

	train_acc_list = []
	test_acc_list = []
	val_acc_list = []
	oob_acc_list = []
	comb_list = rfp.params_comb

	for i, el in enumerate(rfp.params_comb):
	    n_estimator, max_feature, min_sample_split = el
	    print(str(i)+" iteration: n_estimators="+str(n_estimator)+" max_features="+str(max_feature)+" min_samples_split="+str(min_sample_split))
	    rfc_clf = rfp.rfc[i]

	    oob_acc = rfc_clf.oob_score_
	    oob_acc_list.append(oob_acc)

	    train_acc = rfc_clf.score(train_x, train_y)
	    train_acc_list.append(train_acc)

	    test_acc = rfc_clf.score(test_x, test_y)
	    test_acc_list.append(test_acc)

	    val_acc = rfc_clf.score(val_x, val_y)
	    val_acc_list.append(val_acc)

	result  = {
	    "accuracy": rfp.accuracy,
	    "best_params": rfp.best_params,
	    "best_params_index": rfp.best_params_index,
	    "best_score": rfp.best_score,
	    "params": rfp.params,
	    "params_comb": rfp.params_comb,
	    "time_to_fit": rfp.time_to_fit,
	    "train_acc_list": train_acc_list,
	    "test_acc_list": test_acc_list,
	    "val_acc_list": val_acc_list,
	    "oob_acc_list": oob_acc_list,
	}

	joblib.dump(result, "complete_result_1")

	return result




def Q3():
	result = None
	if os.path.isfile("./complete_result.dms"):
		result = joblib.load("complete_result.dms")
	else:
		result = Q3_scratch()


	oob_acc_list = result["oob_acc_list"]
	val_acc_list = result["val_acc_list"]
	test_acc_list = result["test_acc_list"]
	train_acc_list = result["train_acc_list"]
	time_to_fit = result["time_to_fit"]
	params_comb = result["params_comb"]
	params = result["params"]

	best_oob_parameter_index = oob_acc_list.index(max(oob_acc_list))

	best_oob_paramters = result["params_comb"][best_oob_parameter_index]
	print("Best oob parameter: ")
	print(best_oob_paramters)
	best_n_estimator, best_max_features, best_min_samples_split = best_oob_paramters
	print("best_n_estimator",best_n_estimator)
	print("best_max_features",best_max_features)
	print("best_min_samples_split",best_min_samples_split)


	oob_acc_oob = oob_acc_list[best_oob_parameter_index]
	train_acc_oob = train_acc_list[best_oob_parameter_index]
	test_acc_oob = test_acc_list[best_oob_parameter_index]
	val_acc_oob = val_acc_list[best_oob_parameter_index]

	print("Oob accuracy:", oob_acc_oob)
	print("Train accuracy:", train_acc_oob)
	print("Test accuracy:", test_acc_oob)
	print("Val accuracy:", val_acc_oob)




###################################################################
#				PARAMETER  SENSITIVITY ANALYSIS    				  #                         
###################################################################



def Q4():
	result = None
	if os.path.isfile("./complete_result.dms"):
		result = joblib.load("complete_result.dms")
	else:
		result = Q3_scratch()

	oob_acc_list = result["oob_acc_list"]
	val_acc_list = result["val_acc_list"]
	test_acc_list = result["test_acc_list"]
	train_acc_list = result["train_acc_list"]
	time_to_fit = result["time_to_fit"]
	params_comb = result["params_comb"]
	params = result["params"]

	best_oob_parameter_index = oob_acc_list.index(max(oob_acc_list))

	best_oob_paramters = result["params_comb"][best_oob_parameter_index]

	best_n_estimator, best_max_features, best_min_samples_split = best_oob_paramters
	print("best_n_estimator",best_n_estimator)
	print("best_max_features",best_max_features)
	print("best_min_samples_split",best_min_samples_split)


	# ### Fixing 2 parameters at a time

	# ##### Fix n_estimator & max_features
	n_estimator_max_feature_index = [i for i, el in enumerate(params_comb) if el[0]==best_n_estimator and el[1]==best_max_features]
	n_estimator_max_feature_comb = [params_comb[i] for i in n_estimator_max_feature_index]
	
	n_estimator_max_feature_x_array = [i[2] for i in n_estimator_max_feature_comb]

	n_estimator_max_feature_train_acc = np.array([train_acc_list[i] for i in n_estimator_max_feature_index])
	n_estimator_max_feature_test_acc =  np.array([test_acc_list[i] for i in n_estimator_max_feature_index])
	n_estimator_max_feature_val_acc =  np.array([val_acc_list[i] for i in n_estimator_max_feature_index])


	plt.plot(n_estimator_max_feature_x_array, n_estimator_max_feature_test_acc*100, label="Test Accuracy")
	plt.plot(n_estimator_max_feature_x_array, n_estimator_max_feature_val_acc*100, label="Validation Accuracy")
	plt.xlabel("Varying min_sample_split")
	plt.ylabel("Accuracy")
	plt.title(f"Fixed n_estimators = {best_n_estimator} & max_features = {best_max_features}")
	plt.legend()
	plt.show()

	# ##### Fix n_estimator & min_samples_split
	n_estimator_min_samples_split_index = [i for i, el in enumerate(params_comb) if el[0]==best_n_estimator and el[2]==best_min_samples_split]
	n_estimator_min_samples_split_comb = [params_comb[i] for i in n_estimator_min_samples_split_index]

	n_estimator_min_samples_split_x_array = [i[1] for i in n_estimator_min_samples_split_comb]

	n_estimator_min_samples_split_train_acc = np.array([train_acc_list[i] for i in n_estimator_min_samples_split_index])
	n_estimator_min_samples_split_test_acc = np.array([test_acc_list[i] for i in n_estimator_min_samples_split_index])
	n_estimator_min_samples_split_val_acc = np.array([val_acc_list[i] for i in n_estimator_min_samples_split_index])

	plt.plot(n_estimator_min_samples_split_x_array, n_estimator_min_samples_split_test_acc*100, label="Test Accuracy")
	plt.plot(n_estimator_min_samples_split_x_array, n_estimator_min_samples_split_val_acc*100, label="Validation Accuracy")
	plt.xlabel("Varying max_features")
	plt.ylabel("Accuracy")
	plt.title(f"Fixed n_estimators = {best_n_estimator} & min_samples_split = {best_min_samples_split}")
	plt.legend()
	plt.savefig(base_path+"fixed_n_estimator_min_samples_split")
	plt.show()

	# ##### Fix max_features & min_samples_split
	max_features_min_samples_split_index = [i for i, el in enumerate(params_comb) if el[1]==best_max_features and el[2]==best_min_samples_split]
	max_features_min_samples_split_comb = [params_comb[i] for i in max_features_min_samples_split_index]

	max_features_min_samples_split_x_array = [i[0] for i in max_features_min_samples_split_comb]

	max_features_min_samples_split_train_acc = np.array([train_acc_list[i] for i in max_features_min_samples_split_index])
	max_features_min_samples_split_test_acc = np.array([test_acc_list[i] for i in max_features_min_samples_split_index])
	max_features_min_samples_split_val_acc = np.array([val_acc_list[i] for i in max_features_min_samples_split_index])

	plt.plot(max_features_min_samples_split_x_array, max_features_min_samples_split_test_acc*100, label="Test Accuracy")
	plt.plot(max_features_min_samples_split_x_array, max_features_min_samples_split_val_acc*100, label="Validation Accuracy")
	plt.xlabel("Varying n_estimators")
	plt.ylabel("Accuracy")
	plt.title(f"Fixed max_features = {best_max_features} & min_samples_split = {best_min_samples_split}")
	plt.legend()
	plt.show()





if __name__ == "__main__":
	print("Q3 is running")
	Q3()
	print("Q4 is running")
	Q4()
