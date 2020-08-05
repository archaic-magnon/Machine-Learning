#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxopt
import pandas as pd
import numpy as np
from cvxopt import solvers, matrix
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt  
import math
from scipy.spatial.distance import cdist
import time
from tqdm import tqdm
from sklearn.svm import SVC
from scipy.spatial.distance import pdist, squareform
import scipy
import joblib
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[2]:


import multiprocessing 
print(multiprocessing.cpu_count())


# In[3]:


gama = 0.05


# In[4]:


train_path = "data/train.csv"
test_path = "data/test.csv"
val_path = "data/val.csv"


# In[5]:


def readData(path, c1, c2):
	data = pd.read_csv(path, header=None)

	last_column = data.shape[1]-1
	data = filterData(c1, c2, data)

	# y_data = setY_1(data[last_column], c2, c1)
	x_data = scaleData(data.drop(last_column,1))
	y_data = data[last_column]

	return np.array(x_data), np.array(y_data).reshape(-1,1)


# In[6]:


def readDataUnfiltered(path):
    data = pd.read_csv(path, header=None)

    last_column = data.shape[1]-1
    # data = filterData(c1, c2, data)

    x_data = scaleData(data.drop(last_column,1))
    y_data = data[last_column]


    return np.array(x_data), np.array(y_data).reshape(-1,1)


# In[7]:


def filterData(c1, c2, data):
	last_column = data.shape[1]-1

	r_data = data[(data[last_column]==c1) | (data[last_column]==c2)]
	return r_data 


# In[8]:


def scaleData(data):
    return data / 255


# In[9]:



def calculateAccuracy(actual, predicted):
    count = 0
    for y1, y2 in zip(actual, predicted):
        if y1==y2:
            count+=1
    return count/len(predicted)


# In[10]:


def plot_cm(actual, predicted,title="Confusion Matrix" ,labels=None):
    cm = confusion_matrix(actual, predicted, labels=labels)
    df_cm = pd.DataFrame(cm, columns=np.unique(actual), index = np.unique(actual))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (11,9))
    sns.set(font_scale=1.4)#for label size
    plt.title(title)

    ax = sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='d')# font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


# # Train Multi

# In[22]:


sk_data_list = []

total_time = time.time()
for i in tqdm(range(0,9)):
    for j in tqdm(range(i+1, 10)):
        model_name = str(i)+"_"+str(j)
        
        clf_gauss = SVC(gamma=gama)
        
        train_x, train_y = readData(train_path, i, j)
        val_x, val_y = readData(val_path, i, j)
        test_x, test_y = readData(test_path, i, j)
        
        train_time = time.time()
        clf_gauss.fit(train_x, train_y.reshape(-1,))
        train_time = time.time() - train_time
        
        train_pred = clf_gauss.predict(train_x)
        val_pred = clf_gauss.predict(val_x)
        test_pred = clf_gauss.predict(test_x)
        
        a1 = calculateAccuracy(train_y, train_pred)
        a2 = calculateAccuracy(val_y, val_pred)
        a3 = calculateAccuracy(test_y, test_pred)
        
        
        model_data = {
            "name": model_name,
            "train_acc": a1,
            "val_acc": a2,
            "test_acc": a3,
            "train_time": train_time,
            "train_pred": train_pred,
            "val_pred": val_pred,
            "test_pred": test_pred,  
        }
        sk_data_list.append(model_data)
#         joblib.dump(clf_gauss, model_name)
        
        print("train_pred", a1)
        print("val_pred", a2)
        print("test_pred", a3)
        
        
print(f"Total time", time.time() - total_time)


# In[23]:


sk_data_list


# In[24]:


joblib.dump(sk_data_list, "sk_data_list")


# In[58]:


test_x_f , test_y_f = readDataUnfiltered(test_path)


# In[63]:


test_count = test_x_f.shape[0]


# In[65]:


test = test_x_f


# In[66]:


test_count = test.shape[0]


# In[67]:


m_vote = np.zeros((test_count,10))


# In[71]:


m_conf = np.zeros((test_count,10))


# In[72]:


m_conf.shape


# In[ ]:





# In[ ]:





# In[11]:


from tqdm import tqdm_notebook


# # Predict Multi

# In[166]:


def predictMulti(test):
    test_count = test.shape[0]
    m_vote = np.zeros((test_count,10))
    m_conf = np.zeros((test_count,10))

    for i in tqdm_notebook(range(0,9)):
        for j in (range(i+1, 10)):
            model_name = str(i)+"_"+str(j)
            model = joblib.load(model_name)
            for test_no, test_example in tqdm_notebook(enumerate(test), total=test_count):

               

                pred =  int(model.predict(test_example.reshape(1,-1))[0])
                conf = model.decision_function(test_example.reshape(1,-1))[0]

                m_vote[test_no][pred] = m_vote[test_no][pred] + 1
                m_conf[test_no][pred] = m_conf[test_no][pred] + abs(conf)
    
    p = np.argmax(m_vote, axis=1) 
    for test_no in range(test_count):
        n_vote = m_vote[test_no][p[test_no]]
        max_conf = m_conf[test_no][p[test_no]]

        for i in range(10):
            if n_vote == m_vote[test_no][i]:
                if(m_conf[test_no][i] > max_conf):
                    p[test_no] = i

    return p, m_vote, m_conf


# In[167]:


test_x, test_y = readDataUnfiltered(test_path)
val_x, val_y = readDataUnfiltered(val_path)


# In[168]:


t = time.time()
p_test, v,c = predictMulti(test_x)
print("time:", time.time()-t)


# In[173]:


acc_test = calculateAccuracy(test_y, p_test)


# In[178]:


f1_score_test = f1_score(test_y, p_test, average="weighted")


# In[180]:


print("Test Accuracy:", acc_test)
print("F1 Score:", f1_score_test)


# In[202]:


plot_cm(test_y.astype(int), p_test.astype(int), title="Scikit-Learn Confusion Matrix: Test")


# In[203]:


t = time.time()
p_val, vv,cv = predictMulti(val_x)
print("time:", time.time()-t)


# In[204]:


acc_test = calculateAccuracy(val_y, p_val)
f1_score_test = f1_score(val_y, p_val, average="weighted")
print("Test Accuracy:", acc_test)
print("F1 Score:", f1_score_test)


# In[205]:


plot_cm(val_y.astype(int), p_val.astype(int), title="Scikit-Learn Confusion Matrix: Validation")


# In[ ]:





# In[12]:


train_x, train_y = readDataUnfiltered(train_path)
test_x, test_y = readDataUnfiltered(test_path)
val_x, val_y = readDataUnfiltered(val_path)


# In[13]:


# K Fold 


# In[14]:


# grid search


# In[7]:


C_set = [1e-5, 1e-3, 1, 5, 10]


# In[21]:


# Test set


# In[22]:


# train_x, train_y
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(train_x, train_y)

parameters = { 'C':C_set, 'gamma':[0.05]}
# for train_index, test_index in skf.split(train_x, train_y):
#     X_train, X_test = train_x[train_index], train_x[test_index]
#     y_train, y_test = train_y[train_index], train_y[test_index]
#     svc = svm.SVC()
#     clf = GridSearchCV(svc, parameters)
#     clf.fit(iris.data, iris.target)


    
    
    
    


# # Grid Search for C

# In[23]:


svc = svm.SVC()


# In[32]:


clf = GridSearchCV(svc, parameters, cv=5, verbose=15, scoring='accuracy', n_jobs=-1, )


# In[33]:


st = time.time()
clf.fit(train_x, train_y.reshape(-1))
print("Time taken:", time.time() - st)


# In[34]:


clf.best_params_


# In[36]:


samplesjoblib.dump(clf, "k_fold_clf")


# In[14]:


clf = joblib.load("k_fold_clf")


# In[54]:


# test acc


# In[55]:


st = time.time()
pred_test = clf.predict(test_x)
print("Time taken:", time.time() - st)


# In[63]:


pred_test_acc = calculateAccuracy(test_y, pred_test)
f1_test_score = f1_score(test_y, pred_test, average="weighted")
print("Accuracy on test data:",pred_test_acc)
print("F1 score on test data:", f1_test_score)


# In[ ]:


# val acc


# In[66]:


st = time.time()
pred_val = clf.predict(val_x)
print("Time taken:", time.time() - st)


# In[67]:


pred_val_acc = calculateAccuracy(val_y, pred_val)
f1_val_score = f1_score(val_y, pred_val, average="weighted")
print("Accuracy on val data:",pred_val_acc)
print("F1 score on val data:", f1_val_score)


# In[17]:


gama


# # Plot of C vs Accuracy

# In[15]:


test_acc_list = []
val_acc_list = []


# In[18]:


C_set = [1e-5, 11e-3, 1, 5, 10]
model_list = []


# In[ ]:


for c in tqdm(C_set):
    print(c)
    clf_gauss = SVC(gamma=gama, C=c)
    clf_gauss.fit(train_x, train_y.reshape(-1,))
    model_list.append(clf)
    
    
    pred_test_list = clf.predict(test_x)
    pred_val_list = clf.predic(val_x)
    
    test_acc = calculateAccuracy(test_y, pred_test_list)
    val_acc = calculateAccuracy(val_y, pred_val_list)
    
    test_acc_list.append(test_acc)
    val_acc_list.append(val_acc)


# In[19]:


def pl_C(c):
    clf_gauss = SVC(gamma=gama, C=c)
    clf_gauss.fit(train_x, train_y.reshape(-1,))
    return clf_gauss


# In[21]:


out = Parallel(n_jobs=-1)(delayed(pl_C)(c) for c in tqdm(C_set))


# In[ ]:


# time taken = 14 vmin 


# In[22]:


joblib.dump(out, "SVM_C_Model")


# In[23]:


joblib.dump(out, "SVM_C_Model")


# In[24]:


model_out = joblib.load("SVM_C_Model")


# In[ ]:


test_acc_list = []
val_acc_list = []
for m in model_out:
    pred_test_list = m.predict(test_x)
    pred_val_list = m.predic(val_x)
    
    test_acc = calculateAccuracy(test_y, pred_test_list)
    val_acc = calculateAccuracy(val_y, pred_val_list)
    
    test_acc_list.append(test_acc)
    val_acc_list.append(val_acc)


# In[29]:


def pl_predict(m):
    pred_test_list = m.predict(test_x)
    pred_val_list = m.predict(val_x)
    
    test_acc = calculateAccuracy(test_y, pred_test_list)
    val_acc = calculateAccuracy(val_y, pred_val_list)
    
    return test_acc, val_acc
    


# In[30]:


out = Parallel(n_jobs=-1)(delayed(pl_predict)(m) for m in tqdm(model_out))


# In[ ]:


# time take = 3 min 


# In[71]:


out


# In[72]:


joblib.dump(out, "SVM_C_ACC_TEST_VAL")


# In[2]:


out = joblib.load("SVM_C_ACC_TEST_VAL")


# In[3]:


a,b = zip(*out)


# In[ ]:





# In[4]:


b


# In[8]:


C_set_log = np.log10(C_set)


# In[9]:


C_set_log


# In[10]:


test_acc_list = list(a)


# In[11]:


val_acc_list = list(b)


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.title("C vs Accuracy")
plt.xlabel("C in log10")
plt.ylabel("Accuracy")
plt.plot(C_set_log, test_acc_list, color="b", label="Test Acccuracy", marker="o")
plt.plot(C_set_log, val_acc_list, color="r", label="Validation Accuracy", marker="x")
plt.legend(loc="upper left")


# In[ ]:




