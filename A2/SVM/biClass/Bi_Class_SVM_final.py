#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing 
print(multiprocessing.cpu_count())


# In[2]:


import numpy as np


# In[3]:


# arr = np.ones(10**9, dtype='float128')


# In[4]:


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


# In[5]:


gama = 0.05


# In[6]:


# train_path = "/content/drive/My Drive/ML/fashion_mnist/train.csv"
# test_path = "/content/drive/My Drive/ML/fashion_mnist/test.csv"
# val_path = "/content/drive/My Drive/ML/fashion_mnist/val.csv"


# In[8]:


train_path = "data/train.csv"
test_path = "data/test.csv"
val_path = "data/val.csv"


# In[ ]:





# In[9]:


def readData(path, c1, c2):
	data = pd.read_csv(path, header=None)

	last_column = data.shape[1]-1
	data = filterData(c1, c2, data)

	# y_data = setY_1(data[last_column], c2, c1)
	x_data = scaleData(data.drop(last_column,1))
	y_data = data[last_column]

	return np.array(x_data), np.array(y_data).reshape(-1,1)


# In[10]:


def readDataUnfiltered(path):
    data = pd.read_csv(path, header=None)

    last_column = data.shape[1]-1
    # data = filterData(c1, c2, data)

    x_data = scaleData(data.drop(last_column,1))
    y_data = data[last_column]


    return np.array(x_data), np.array(y_data).reshape(-1,1)


# In[11]:


def filterData(c1, c2, data):
	last_column = data.shape[1]-1

	r_data = data[(data[last_column]==c1) | (data[last_column]==c2)]
	return r_data 


# In[12]:


def setY_1(y, pos_class, neg_class):
    y[y==pos_class] = 1
    y[y==neg_class] = -1

    return y


# In[13]:


def scaleData(data):
    return data / 255


# In[14]:


def getP(y_train, x_train, kernel):
    m = y_train.shape[0]
    KernelMatrix = None
    if kernel == linear_kernel:
        KernelMatrix = (x_train@x_train.T)
        P = (y_train@y_train.T) * KernelMatrix
        return matrix(P), KernelMatrix
    if kernel == gaussian_kernel:
        print("Calculating P")
        # pairwise_sq_dists = squareform(pdist(x_train, 'sqeuclidean'))
        # KernelMatrix = np.exp(-pairwise_sq_dists * gama)
        KernelMatrix = np.exp( -gama*(cdist(x_train, x_train,'sqeuclidean')))
        P = (y_train@y_train.T) * KernelMatrix
        # print("y")
        # print(y_train)
        # print("yy.t")
        # print(y_train@y_train.T)
        # print("k")
        # print(KernelMatrix)

        return matrix(P), KernelMatrix


# In[15]:



def getq(m):
    q = -np.ones((m,1))
    return matrix(q)


# In[16]:


def getGh(m, C):
    G1 = np.identity(m)
    G = np.vstack((-G1, G1))

    H1 = np.zeros((m,1))
    H2 = C * np.ones((m,1))
    h = np.vstack((H1, H2))

    return matrix(G), matrix(h)


# In[17]:


def getAb(y_train):
    A = y_train.T
    b = np.zeros(1)
    return matrix(A), matrix(b)


# In[18]:


def linear_kernel(x, z):
    return (x.T)@z


# In[19]:


def gaussian_kernel(x, z):
    return np.exp((-np.linalg.norm(x-z)**2) * gama)


# In[20]:


def getWtX(alpha, Y, X, x, kernel):
    m = Y.shape[0]
    ans = 0
    for i in range(m):
        if(alpha[i]==0):
            continue
        ans+= alpha[i] * Y[i] * kernel(X[i].reshape(-1,1), x.reshape(-1,1))
    return ans


# In[21]:


def getb(alpha, Y, X, kernel):
    print("calculating b")
    ans = 0
    m = Y.shape[0]
    max1 = -float('inf')
    min1 = float('inf')
    for i in tqdm(range(m)):
        wtx = getWtX(alpha, Y, X, X[i], kernel)
        if Y[i] == -1:
            max1 = max(max1, wtx)
        elif Y[i] == 1:
            min1 = min(min1, wtx)

    b = -1*(max1 + min1)/2
    print("b", b)
    return b


# In[22]:


def getGaussb(alpha,alpha_f, X, X_f, Y, Y_f, kernalMatrix):
  print("alpha_f",alpha_f.shape)
  print("Y_f",Y_f.shape)
  print("X_f",X_f.shape)
  print("X",X.shape)
  wtx = (   np.exp(-1*cdist(X, X_f, "sqeuclidean")*gama) @ (alpha_f*Y_f) )
  print("wtx",wtx.shape)
  t1 = np.max(wtx[np.where(Y==-1)])
  t2 = np.min(wtx[np.where(Y==1)])
  b = -1 * (t1 + t2) /2
  print("gb", b)
  return b



# In[23]:


def getbb(alpha, X,Y, kernalMatrix):
  temp = alpha * Y * kernalMatrix
  print(temp.shape)
  # filter temp where y = -1 and 1
  m,n = X.shape
  Yy = np.repeat(Y,m,axis=1)
  temp1 = temp[np.where(Yy ==1)].reshape(-1, m)
  temp2 = temp[np.where(Yy ==-1)].reshape(-1, m)
  print(Yy.shape)
  print(temp1.shape)
  print(temp2.shape)

  b1 = np.max(np.sum(temp1, axis=0))
  b2 = np.min(np.sum(temp2, axis=0))
  b = -1*(b1+b2)/2
  print("bb", b)

  return b


# In[24]:



def calculateAccuracy(actual, predicted):
    count = 0
    for y1, y2 in zip(actual, predicted):
        if y1==y2:
            count+=1
    return count/len(predicted)


# In[46]:


def plot_cm(actual, predicted, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(actual, predicted, labels=labels)
    df_cm = pd.DataFrame(cm, columns=np.unique(actual), index = np.unique(actual))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    # plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size

    ax = sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='d')# font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)
    plt.show()


# In[26]:


class BiSVM:

    def replaceLabelsBefore(self, y):
      y = np.where(y==self.neg_class, -1, y)
      y = np.where(y==self.pos_class, 1, y)
      return y

    def replaceLabelsAfter(self, y):
      # y = np.where(y==100, self.pos_class, y)
      # print(self.pos_class)
      y = np.where(y==1, self.pos_class, y)
      # print(y)
      # print(y.max())
      # print(y.min())
      y = np.where(y==-1, self.neg_class, y)
      # print(y)
      # print(y.max())
      # print(y.min())
      return y

    def predictGauss(self, X_test):
        kernel = self.kernel
        alpha = self.alpha
        Y = self.Y
        X = self.X
        b = self.b
        wtx =  np.exp(-1*cdist(X_test, X, "sqeuclidean")*gama) @ (alpha*Y) 
        return self.replaceLabelsAfter(np.sign(wtx+b))



    def predict(self, X_test):
        kernel = self.kernel
        if kernel ==gaussian_kernel:
          return self.predictGauss(X_test)
        alpha = self.alpha
        Y = self.Y
        X = self.X
        b = self.b
        p =  np.array([ 1  if getWtX(alpha, Y, X, el, kernel) + b > 0 else -1  for el in tqdm(X_test)])
        p = self.replaceLabelsAfter(p)
        return np.array(p).astype(int)

    def getParameters(self):
        parameter = {}
        parameter["alpha"]  = self.alpha
        parameter["b"] = self.b
        parameter["pos_class"] = self.pos_class
        parameter["neg_class"] = self.neg_class
        parameter["kernel"] = self.kernel
        parameter["b"] = self.b
        parameter["w"] = self.w
        parameter["count_sv"] = self.count_sv
        return parameter

    def fit(self, train_x, train_y, pos_class, neg_class, kernel):
        self.pos_class = pos_class
        self.neg_class = neg_class
        self.kernel = kernel

        train_y = self.replaceLabelsBefore(train_y)

        cut_off = 1e-5
        C = 1
        m,n = train_x.shape
        P, KernelMatrix = getP(train_y, train_x, kernel)
        q = getq(m)
        G, h = getGh(m, C)
        A, b = getAb(train_y)

        self.KernelMatrix = KernelMatrix

        sol = solvers.qp(P,q,G,h,A,b)
        # print(sol['x'])

        alpha = np.array(sol['x']).reshape(-1,1)

        count_sv = len(alpha[alpha>cut_off])
        print(f"No. of support vectors:",  count_sv)

        alpha_index = alpha>cut_off
        alpha[alpha<cut_off] = 0

        alpha_index1 = np.where(alpha_index)[0]

        train_x_f = np.take(train_x, alpha_index1, axis=0)

        alpha_f = (alpha*alpha_index)
        alpha_f = (alpha_f[alpha_f!=0]).reshape(-1,1)

        train_y_f = (train_y*alpha_index)
        train_y_f = (train_y_f[train_y_f!=0]).reshape(-1,1)
        if kernel == gaussian_kernel:
          b = getGaussb(alpha,alpha_f, train_x, train_x_f, train_y, train_y_f, KernelMatrix)
        else:
          b = getb(alpha_f, train_y_f, train_x_f, kernel)
        # bb = getbb(alpha, train_x,train_y, KernelMatrix)
        

        # self.bb = bb
        self.count_sv = count_sv
        self.alpha_f = alpha_f
        self.train_x_f = train_x_f
        self.train_y_f = train_y_f
        self.b = b
        self.w = None

        self.alpha = alpha
        self.X = train_x
        self.Y = train_y

        

        if kernel == linear_kernel:
            w = np.sum(train_y_f * alpha_f * train_x_f, axis=0).reshape(-1,1)
            self.w = w

        return True
        
       


# In[27]:


# def plot_cm(actual, predicted):
#     cm = confusion_matrix(actual, predicted)
#     df_cm = pd.DataFrame(cm, columns=np.unique(actual), index = np.unique(actual))
#     df_cm.index.name = 'Actual'
#     df_cm.columns.name = 'Predicted'
#     # plt.figure(figsize = (10,7))
#     sns.set(font_scale=1.4)#for label size

#     ax = sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='d')# font size
#     bottom, top = ax.get_ylim()
#     ax.set_ylim(bottom + 0.5, top - 0.5)
#     plt.show()


# In[ ]:





# In[ ]:





# # Linear SVM

# In[ ]:





# In[28]:


d = 5
c1 = d
c2 = (d+1)%10

train_x, train_y = readData(train_path, c1, c2)
test_x, test_y = readData(test_path, c1, c2)
val_x, val_y = readData(val_path, c1, c2)


# In[ ]:





# In[29]:


# train_x, train_y = readDataUnfiltered(train_path)
# test_x, test_y = readDataUnfiltered(test_path)
# val_x, val_y = readDataUnfiltered(val_path)


# In[30]:


train_x.shape


# In[31]:


# yy = np.put(train_y, [-1, 1], [5, 6])


# In[32]:


# yy


# In[33]:


BiSVM_model = BiSVM()


# In[34]:


start_time = time.time()
BiSVM_model.fit(train_x, train_y, c2, c1, linear_kernel)
print(f"Time taken = {time.time() - start_time} sec")


# In[35]:


BiSVM_model.count_sv


# In[36]:


BiSVM_model.w


# In[37]:


bisvm_parameter = BiSVM_model.getParameters()


# In[38]:


start_time = time.time()
predicted_train =  BiSVM_model.predict(train_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[39]:


train_acc = calculateAccuracy(predicted_train, train_y)


# In[40]:


train_f1 = f1_score(train_y, predicted_train, pos_label=c2)


# In[41]:


print(f"Train Accuracy:", train_acc)
print(f"Train F1 score:", train_f1)


# In[59]:


plot_cm(train_y.astype(int), predicted_train.astype(int), labels=[6,5], title="5 vs 6 on Train")


# In[ ]:


#validation


# In[52]:


start_time = time.time()
predicted_val = BiSVM_model.predict(val_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[53]:


val_acc = calculateAccuracy(predicted_val, val_y)


# In[54]:


val_f1 = f1_score(val_y, predicted_val, pos_label=c1)


# In[55]:


print(f"Val Accuracy:", val_acc)
print(f"Val F1 score:", val_f1)


# In[60]:


plot_cm(val_y.astype(int), predicted_val.astype(int), labels=[c2,c1],  title="5 vs 6 on Val")


# In[ ]:


# test


# In[ ]:





# In[61]:


start_time = time.time()
predicted_test = BiSVM_model.predict(test_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[62]:


test_acc = calculateAccuracy(predicted_test, test_y)


# In[63]:


test_f1 = f1_score(test_y, predicted_test, pos_label=c1)


# In[120]:


print(f"Test Accuracy:", val_acc)
print(f"Test F1 score:", val_f1)


# In[64]:


plot_cm(test_y.astype(int), predicted_test.astype(int), labels=[c2,c1],  title="5 vs 6 on Test")


# In[ ]:





# In[ ]:





# ## Linear SVM SKLearn

# In[ ]:





# In[65]:


# SKLearn


# In[66]:


clf_linear_model = SVC(kernel="linear")


# In[67]:


start_time = time.time()
clf_linear_model.fit(train_x, train_y)
print(f"Time taken = {time.time() - start_time} sec")


# In[68]:


clf_linear_model.intercept_


# In[69]:


print("SKLearn b", clf_linear_model.intercept_)


# In[128]:


clf_linear_model.coef_.shape


# In[131]:


w_diff = np.linalg.norm(clf_linear_model.coef_ - (BiSVM_model.w).reshape(1,-1))


# In[132]:


print("Norm distance between w", w_diff)


# In[70]:


print("SKLearn w", clf_linear_model.coef_)


# In[ ]:





# In[71]:


clf_linear_model.coef_.shape


# In[72]:


clf_linear_model.n_support_


# In[73]:


print(f"SKLearn NSV", clf_linear_model.n_support_)


# In[74]:


# train


# In[75]:


start_time = time.time()
clf_linear_pred = clf_linear_model.predict(train_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[76]:


train_sk_l_acc = calculateAccuracy(train_y,clf_linear_pred)


# In[77]:


train_sk_l_f1 = f1_score(train_y,clf_linear_pred, pos_label=c1)


# In[78]:


print(f"Train Accuracy:", train_sk_l_acc)
print(f"Train F1 score:", train_sk_l_f1)


# In[79]:


#Val


# In[80]:


start_time = time.time()
clf_linear_pred_val = clf_linear_model.predict(val_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[81]:


val_sk_l_acc = calculateAccuracy(val_y,clf_linear_pred_val)
val_sk_l_f1 = f1_score(val_y,clf_linear_pred_val, pos_label=c1)


# In[82]:


print(f"Val Accuracy:", val_sk_l_acc)
print(f"Val F1 score:", val_sk_l_f1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[83]:


#test


# In[84]:


start_time = time.time()
clf_linear_pred_test = clf_linear_model.predict(test_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[85]:


test_sk_l_acc = calculateAccuracy(test_y,clf_linear_pred_test)
test_sk_l_f1 = f1_score(test_y,clf_linear_pred_test, pos_label=c1)


# In[86]:


print(f"Test Accuracy:", test_sk_l_acc)
print(f"Test F1 score:", test_sk_l_f1)


# In[87]:


# Gaussian


# # Gaussian

# In[88]:


## Gaussian Manual


# ## Gaussian Manual

# In[89]:


train_x.shape


# In[90]:


BiSVM_model_gauss = BiSVM()


# In[91]:


start_time = time.time()
BiSVM_model_gauss.fit(train_x, train_y, c1, c2, gaussian_kernel)
print(f"Time taken = {time.time() - start_time} sec")


# In[92]:


print(BiSVM_model_gauss.b)


# In[93]:


# train


# In[94]:


start_time = time.time()
g_train_pred = BiSVM_model_gauss.predictGauss(train_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[121]:


g_train_acc = calculateAccuracy(g_train_pred, train_y)
g_train_f1 = f1_score(train_y, g_train_pred, pos_label=c1)
print(f"Val Accuracy:", g_train_acc)
print(f"Val F1 score:", g_train_f1)
plot_cm(train_y.astype(int), g_train_pred.astype(int), labels=[c2,c1], title ="5 vs 6 on Train")


# In[96]:


# val


# In[97]:


start_time = time.time()
g_val_pred = BiSVM_model_gauss.predictGauss(val_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[122]:


g_val_acc = calculateAccuracy(g_val_pred, val_y)
g_val_f1 = f1_score(val_y, g_val_pred, pos_label=c1)
print(f"Val Accuracy:", g_val_acc)
print(f"Val F1 score:", g_val_f1)
plot_cm(val_y.astype(int), g_val_pred.astype(int), labels=[c2,c1],  title ="5 vs 6 on Validation")


# In[99]:


# test


# In[100]:


start_time = time.time()
g_test_pred = BiSVM_model_gauss.predictGauss(test_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[123]:


g_test_acc = calculateAccuracy(g_test_pred, test_y)
g_test_f1 = f1_score(test_y, g_test_pred, pos_label=c1)
print(f"Val Accuracy:", g_test_acc)
print(f"Val F1 score:", g_test_f1)
plot_cm(test_y.astype(int), g_test_pred.astype(int), labels=[c2,c1], title="5 vs 6 on Test")


# In[102]:


# joblib.dump(BiSVM_model_gauss, "/content/drive/My Drive/ML/SVM/MultiClass/BiSVM_model_gauss_j")


# In[ ]:





# ## Gaussian SKLearn

# In[103]:


# sklearn Gaussian


# In[104]:


clf_gauss = SVC(gamma=gama)


# In[105]:


start_time = time.time()
clf_gauss.fit(train_x, train_y)
print(f"Time taken = {time.time() - start_time} sec")


# In[106]:


print("b:",clf_gauss.intercept_)
print("nsv",clf_gauss.n_support_ )


# In[107]:


#train predict


# In[108]:


start_time = time.time()
g_predicted_train  = clf_gauss.predict(train_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[109]:


g_acc_train = calculateAccuracy(g_predicted_train, train_y)
g_f1_train = f1_score(train_y, g_predicted_train, pos_label=c2)


# In[110]:


print(f"Train Accuracy:", g_acc_train)
print(f"Train F1 score:", g_f1_train)


# In[111]:


#validation


# In[112]:


start_time = time.time()
g_predicted_val  = clf_gauss.predict(val_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[113]:


g_acc_val = calculateAccuracy(g_predicted_val, val_y)
g_f1_val = f1_score(val_y, g_predicted_val, pos_label=c2)


# In[114]:


print(f"Val Accuracy:", g_acc_val)
print(f"Val F1 score:", g_f1_val)


# In[115]:


# test


# In[116]:


start_time = time.time()
g_predicted_test  = clf_gauss.predict(test_x)
print(f"Time taken = {time.time() - start_time} sec")


# In[117]:


g_acc_test = calculateAccuracy(g_predicted_test, test_y)
g_f1_test = f1_score(test_y, g_predicted_test, pos_label=c2)


# In[118]:


print(f"Test Accuracy:", g_acc_test)
print(f"Test F1 score:", g_f1_test)


# In[ ]:





# In[119]:


# Multi Class


# In[ ]:





# In[ ]:





# In[ ]:




