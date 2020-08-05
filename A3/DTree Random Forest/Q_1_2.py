from xclib.data import data_utils
import numpy as np
import scipy.sparse as spp
from scipy.sparse import csr_matrix, csc_matrix
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm, tqdm_notebook
from graphviz import Digraph
import time
import joblib
from joblib import Parallel,delayed
import sklearn
import sklearn.utils.sparsefuncs as sparsefuncs
import matplotlib.pyplot as plt
from collections import deque
import sys, os

base_path = 'data/'

train_x_path = base_path + 'train_x.txt'
train_y_path = base_path + 'train_y.txt'

test_x_path = base_path + 'test_x.txt'
test_y_path = base_path + 'test_y.txt'

val_x_path = base_path + 'valid_x.txt'
val_y_path = base_path + 'valid_y.txt'




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
	print("python3 Q_1_2.py")
	print("OR")
	print("python3 Q_1_2.py [train_x_path] [train_y_path] [test_x_path] [test_y_path] [valid_x_path] [valid_y_path]")

	raise("Either 0 or 6 Argument Expected")






###################################################################
#						DECISION  TREE    					      #                         
###################################################################


def visualizeTree(tree):
    def getlabel(node):
        str_label = ""
        if node.isLeaf:
            str_label ='''label={},
    total={}
    pos={} | neg={}
            '''.format(node.label, node.total, node.pos_count, node.neg_count)

        else:
            str_label = '''
    total={}
    pos={} | neg={}
    median={}
    gain={}
            '''.format(node.total, node.pos_count, node.neg_count, node.median, node.entropy_gain)
        return str_label
    def add_node_edges(tree, dot=None):
        
        if dot is None:
            dot = Digraph()
            dot.node(name=str(tree), label=getlabel(tree))
        if tree.depth <= 1:
            if tree.left:
                dot.node(name=str(tree.left), label=getlabel(tree.left))
                dot.edge(str(tree), str(tree.left), label=str(tree.decisionAttribute))
                dot = add_node_edges(tree.left, dot=dot)

            if tree.right:
                dot.node(name=str(tree.right) ,label=getlabel(tree.right))
                dot.edge(str(tree), str(tree.right), label=str(tree.decisionAttribute))
                dot = add_node_edges(tree.right, dot=dot)

        
        return dot
    
    dot = add_node_edges(tree)
    display(dot)
    dot.view()
    return dot


def getAccuracy(original, predicted):
    original = np.array(original).reshape(-1,)
    predicted = np.array(predicted).reshape(-1,)

    if original.shape[0]!=predicted.shape[0]:
        raise("Unequal length")

    total = original.shape[0]
    correct = np.count_nonzero(original==predicted)

    return correct/total

def predict(root, test_ex):
    def predict_single(root, x):
        if root.isLeaf:
            return root.label
        else:
            split_attr = root.decisionAttribute
            if x[split_attr] <= root.median:
                return predict_single(root.left, x)
            else:
                return predict_single(root.right, x)

    return np.array([predict_single(root, ex) for ex in (test_ex)])

def getMode(y):
    return mode(y.reshape(-1,))[0][0]

def readFileSparse(path):
    return data_utils.read_sparse_file(path, header=True)

def readFile(x_path, y_path):
    x_data = np.array(csr_matrix.todense(readFileSparse(x_path)))
    y_data = np.array(pd.read_csv(y_path, header=None))
    return x_data, y_data


train_x, train_y = readFile(train_x_path, train_y_path)
test_x, test_y = readFile(test_x_path, test_y_path)
val_x, val_y = readFile(val_x_path, val_y_path)


def getEntropy(y):
    m = y.shape[0]
    if m==0:
        return 0
    positive = np.count_nonzero(y)
    negative = m -  positive
    
    pos_prob = positive/m
    neg_prob = negative/m
    
    if pos_prob == 0:
        pos_prob = 1
    if neg_prob == 0:
        neg_prob = 1
    
    entropy = -(pos_prob * np.log2(pos_prob)) - (neg_prob * np.log2(neg_prob))
    
    return entropy

def getSplitIndex(x, y, median_value, attribute_index):
    attr_values_arr = x[:, attribute_index]
    x1_cond = np.where(attr_values_arr<=median_value)[0]
    x2_cond = np.where(attr_values_arr>median_value)[0]

    return x1_cond, x2_cond




def getInformationGain(x, y, median_value, attribute_index):
    m = y.shape[0]
    h = getEntropy(y)
    
    split_index1, split_index2 = getSplitIndex(x, y, median_value, attribute_index)
    y1 = y[split_index1]
    y2 = y[split_index2]
    
    m1 = y1.shape[0]
    m2 = y2.shape[0]
    
    h1 = getEntropy(y1)
    h2 = getEntropy(y2)
    
    gain = h - (1/m)*(m1*h1 + m2*h2)
    return gain, median_value, attribute_index, split_index1, split_index2



def getBestAttribute(x, y, attr_list):
    median_list = np.median(x, axis=0)
    
    best_attr = -1
    max_gain = -1
    median = -1
    
    best_split1, best_split2 = None, None
    best_x1, best_y1, best_x2, best_y2 = None, None, None, None
    
    gain, median, best_attr, best_split1, best_split2 = max([getInformationGain(x, y, median_list[i], attr_list[i]) for i in range(len(attr_list))])
    best_x1, best_y1, best_x2, best_y2 = x[best_split1], y[best_split1], x[best_split2], y[best_split2]
    
    return best_attr, gain, median, best_x1, best_y1, best_x2, best_y2


class Node:
    def __init__(self, total, pos_count):
        self.left = None
        self.right = None

        self.total = total
        self.pos_count = pos_count
        self.neg_count = total - pos_count

        self.isLeaf = True
        self.label = 1 if self.pos_count > self.neg_count else 0
        self.decisionAttribute = None
        self.median = None
        self.depth = None
        self.entropy_gain = None
        
        self.val_pos = 0
        self.val_neg = 0
        self.val_total = 0


def getDepth(root):
    if root.isLeaf:
        return 0
    return 1 + max(getDepth(root.left), getDepth(root.right))

def getSubtreeNodeCount(root):
    if root.isLeaf:
        return 1
    left_count = getSubtreeNodeCount(root.left)
    right_count = getSubtreeNodeCount(root.right)
    return left_count + right_count + 1


def leafCount(root):
    if root.isLeaf:
        return 1
    return leafCount(root.left) + leafCount(root.right)
  

class DecisionTree:

    def __init__(self):
        self.root = None
        self.node_count = 0
        self.leaf_count = 0
        self.time_to_fit = 0
        self.hasRoot = False
        
        self.train_acc_list = []
        self.test_acc_list = []
        self.val_acc_list = []
        self.node_count_list = []
        self.tree_depth = 0
        

    def bfsId3(self, x, y, attr_list, depth=0):
        
        start_time = time.time()
        
        root = Node(y.shape[0], np.count_nonzero(y))
        root.depth = 0
        root.x = x
        root.y = y
        bfsQ = deque()
        bfsQ.append(root)
        self.root = root
        self.hasRoot = True
        self.node_count+=1
        
        itr_count = 0
        
        self.node_count_list.append(self.node_count)
        self.train_acc_list.append(getAccuracy(train_y, self.predict(train_x)))
        self.test_acc_list.append(getAccuracy(test_y, self.predict(test_x)))
        self.val_acc_list.append(getAccuracy(val_y, self.predict(val_x)))

        while bfsQ:
            itr_count+=1
           
            curr_node = bfsQ.popleft()
            curr_x, curr_y, curr_depth = curr_node.x, curr_node.y, curr_node.depth
            
            # message log after 1000 iteration
            if itr_count%1000 ==0:
                print("depth:"+str(curr_depth)+" Node:"+str(self.node_count)+" Leaf:"+str(self.leaf_count) )
            
            # accuracy after every 100 nodes
            if self.node_count % 101 == 0:
                self.node_count_list.append(self.node_count)
                self.train_acc_list.append(getAccuracy(train_y, self.predict(train_x)))
                self.test_acc_list.append(getAccuracy(test_y, self.predict(test_x)))
                self.val_acc_list.append(getAccuracy(val_y, self.predict(val_x)))
            
            curr_node.isLeaf = False

            
            # check for leaf condition
            if np.all(curr_y == curr_y[0][0]):
                curr_node.isLeaf = True
                self.leaf_count+=1
                self.tree_depth = curr_depth
                continue

            
            best_attr, gain, median_value, x1, y1, x2, y2 = getBestAttribute(curr_x, curr_y, attr_list)
            
             # If no gain then make it leaf
            if gain < 1e-10:
                curr_node.isLeaf = True
#                 curr_node.label = getMode(y)
                self.leaf_count+=1
                self.tree_depth = curr_depth
                continue
            
            # The decision attribute for node <- A 
            curr_node.decisionAttribute = best_attr
            curr_node.median = median_value
            curr_node.entropy_gain = gain
            
            left_node = Node(y1.shape[0], np.count_nonzero(y1))
            left_node.x = x1
            left_node.y = y1
            left_node.depth = curr_depth+1
            
            right_node = Node(y2.shape[0], np.count_nonzero(y2))
            right_node.x = x2
            right_node.y = y2
            right_node.depth = curr_depth+1
            
            curr_node.left = left_node
            curr_node.right = right_node
            
            bfsQ.append(left_node)
            bfsQ.append(right_node)
            
#             destroy current node data
            curr_node.x = None
            curr_node.y = None

            self.node_count+=2
        
        self.time_to_fit = time.time() - start_time
        return root
            
            
    def id3(self, x, y, attr_list, depth=0):
        # create root node
        root = Node(y.shape[0], np.count_nonzero(y))
        root.depth = depth
        
        # log meassage about progress
        if self.node_count % 1000 == 0:
            print("node processed:",self.node_count)
            print("leaf count:",self.leaf_count)
        
            
        
        self.node_count+=1
        
        if not self.hasRoot:
            self.root = root
            self.hasRoot = True
        
        # calculate accuray
#         root.isLeaf = True
#         root.label = getMode(y)
#         self.train_acc_list.append(getAccuracy(train_y, self.predict(train_x)))
#         self.test_acc_list.append(getAccuracy(test_y, self.predict(test_x)))
#         self.val_acc_list.append(getAccuracy(val_y, self.predict(val_x)))
#         root.isLeaf = False
#         root.label = None

        
        # if maximum depth reached
        
        if depth >= self.max_depth >= 0:
            root.isLeaf = True
            root.label = getMode(y)
            self.leaf_count+=1
            return root
            
        
        # If all Examples are positive, Return the single-node tree Root, with label = +
        if np.all(y==1):
            root.isLeaf = True
            root.label = 1
            self.leaf_count+=1
            return root

        # If all Examples are negative, Return the single-node tree Root, with label = -
        if np.all(y==0):
            root.isLeaf = True
            root.label = 0
            self.leaf_count+=1
            return root

        # If Attributes is empty, Return the single-node tree Root, with label = most common value of 
        # Targetattribute in Examples 
        if attr_list.shape[0] == 0:
            root.isLeaf = True
            root.label = getMode(y)
            self.leaf_count+=1
            return root

        # A <- the attribute from Attributes that best* classifies Examples
        best_attr, gain, median_value, x1, y1, x2, y2 = getBestAttribute(x, y, attr_list)
        

        # If no gain then make it leaf
        if gain < 1e-10:
            root.isLeaf = True
            root.label = getMode(y)
            self.leaf_count+=1
            return root
       
        
        
        

        # The decision attribute for Root <- A 
        root.decisionAttribute = best_attr
        root.median = median_value
        root.entropy_gain = gain

        if y1.shape[0] == 0:
            # never happen
            left_node = Node(0, 0)
            left_node.isLeaf  = True
            left_node.label = getMode(y)
            root.left = left_node
            self.leaf_count+=1
            self.node_count+=1
        else:
            root.left = self.id3(x1, y1, attr_list, depth+1)

        if y2.shape[0] == 0:
            # never happen
            right_node = Node(0, 0)
            right_node.isLeaf  = True
            right_node.label = getMode(y)
            root.right = right_node
            self.leaf_count+=1
            self.node_count+=1
        else:
            root.right = self.id3(x2, y2, attr_list, depth+1)

        return root
    
    def fit(self, x, y):
        n_feature = x.shape[1]
        start_time = time.time()
        self.root = self.bfsId3(x, y, np.arange(0, n_feature))
#         self.root = self.id3(x, y, np.arange(0, n_feature))
        self.time_to_fit = time.time() - start_time
    
    def predict(self,test_ex):
        root = self.root
        def predict_single(root, x):
            if root.isLeaf:
                return root.label
            else:
                split_attr = root.decisionAttribute
                if x[split_attr] <= root.median:
                    return predict_single(root.left, x)
                else:
                    return predict_single(root.right, x)

        return np.array([predict_single(root, ex) for ex in (test_ex)])




def plotQ1(clf):
#     plt.figure(figsize=(10,5))
    plt.title("No. of nodes vs Accuracies")
    plt.xlabel("No. of nodes")
    plt.ylabel("Accuracies")
    plt.plot(clf.node_count_list, clf.train_acc_list, label="Train Accuracy")
    plt.plot(clf.node_count_list, clf.test_acc_list, label="Test Accuracy")
    plt.plot(clf.node_count_list, clf.val_acc_list, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("fig1")
    plt.show()





def Q1_scratch():
	clf = DecisionTree()
	clf.fit(train_x, train_y)
	joblib.dump(clf, "clfQ1_1")
	return clf


def Q1():
	clf = None
	if os.path.isfile("./clfQ1"):
		clf = joblib.load("clfQ1")
	else:
		clf = Q1_scratch()


	print("Node count:", clf.node_count)
	print("Leaf count:", clf.leaf_count)
	print("Tree depth:", clf.tree_depth)
	print("Time to built:", clf.time_to_fit)
	print("Train Accuracy", getAccuracy(train_y, clf.predict(train_x)))
	print("Test Accuracy", getAccuracy(test_y, clf.predict(test_x)))
	print("Validation Accuracy", getAccuracy(val_y, clf.predict(val_x)))

	plotQ1(clf)

###################################################################
#							POST PRUNING    					  #                         
###################################################################
def plotQ2(prune_clf, name):
#     plt.figure(figsize=(15,10))
    plt.title("No. of nodes vs Accuracies after pruning")
    plt.xlabel("No. of nodes as we prue")
    plt.ylabel("Accuracies")
    plt.plot(prune_clf.node_count_list, prune_clf.train_acc_list, label="Train Accuracy")
    plt.plot(prune_clf.node_count_list, prune_clf.test_acc_list, label="Test Accuracy")
    plt.plot(prune_clf.node_count_list, prune_clf.val_acc_list, label="Validation Accuracy")
    plt.legend(loc="upper right")
    plt.gca().invert_xaxis()

    plt.savefig(name)
    plt.show()


def getValAccuracy():
    return getAccuracy(val_y, clf.predict(val_x))


####
# APPROACH 1
####


class PruneNew:
    def __init__(self, root):
        self.root = root
#         self.node_count = clf.node_count
#         self.leaf_count = 0
        self.time_to_fit = 0
        self.train_acc_list = []
        self.test_acc_list = []
        self.val_acc_list = []
        self.node_count_list = []
        self.start_time = 0
#         self.tree_depth = 0

    def prunePostOrder(self, root):
        if root.left:
            self.prunePostOrder(root.left)
        if root.right:
            self.prunePostOrder(root.right)
        
        if not root.isLeaf:
            current_accuracy = getValAccuracy()
            root.isLeaf = True
            prune_accuracy = getValAccuracy()
            
            if prune_accuracy > current_accuracy :
                node_count = getSubtreeNodeCount(self.root)
                if not self.node_count_list or self.node_count_list[-1]//100 != node_count//100:
                    self.node_count_list.append(node_count)
                    self.train_acc_list.append(getAccuracy(train_y, clf.predict(train_x)))
                    self.test_acc_list.append(getAccuracy(test_y, clf.predict(test_x)))
                    self.val_acc_list.append(getAccuracy(val_y, clf.predict(val_x)))
                print("Time:", time.time() - self.start_time)
                print("Node:", node_count)

            else:
                root.isLeaf = False
    
    def fit(self):
        self.start_time = time.time()
        self.prunePostOrder(self.root)
        self.time_to_fit = time.time() - self.start_time
        


def Q2_1_scratch():
	clf = joblib.load("clfQ1")
	prune_clf_new = PruneNew(clf.root)
	prune_clf_new.fit()
	return prune_clf_new


def Q2_1():
	prune_clf_new = None
	if os.path.isfile("./prune_clf_new"):
		prune_clf_new = joblib.load("prune_clf_new")
	else:
		prune_clf_new = Q2_1_scratch()

	print("Node count:", getSubtreeNodeCount(prune_clf_new.root))
	print("Leaf count:", leafCount(prune_clf_new.root))
	print("Tree depth:", getDepth(prune_clf_new.root))
	print("Train Accuracy", getAccuracy(train_y, predict(prune_clf_new.root,train_x)) )
	print("Test Accuracy", getAccuracy(test_y, predict(prune_clf_new.root,test_x)) )
	print("Validation Accuracy", getAccuracy(val_y, predict(prune_clf_new.root,val_x)) )

	plotQ2(prune_clf_new, "fig2_1")



####
# APPROACH 2
####

class Prune:
    def __init__(self, root):
        self.root = root
        # self.node_count = clf.node_count
        # self.leaf_count = 0
        self.time_to_fit = 0
        self.train_acc_list = []
        self.test_acc_list = []
        self.val_acc_list = []
        self.node_count_list = []
        self.start_time = 0
        # self.tree_depth = 0
    
    def classifyValidationSet(self):
        root = self.root
        
        def classifySingle(root, x, y):
            root.val_total+=1
            if y==1:
                root.val_pos+=1
            else:
                root.val_neg+=1
                    
            if not root.isLeaf:
                split_attr = root.decisionAttribute
                if x[split_attr] <= root.median:
                    classifySingle(root.left, x, y)
                else:
                    classifySingle(root.right, x, y)
        [classifySingle(root, val_x[i], val_y[i][0])  for i in range(val_y.shape[0])]

        
        
    def prune(self):
        self.pruneTreeError(self.root)
        

    def pruneTreeError(self, root):
        if root.isLeaf:
            if root.label == 1:
                return root.val_neg
            else:
                return root.val_pos
        else:
            error = self.pruneTreeError(root.left) + self.pruneTreeError(root.right)
            if error < min(root.val_pos, root.val_neg):
                return error
            else:
                # prune
                root.isLeaf = True
                node_count = getSubtreeNodeCount(self.root)
                if not self.node_count_list or self.node_count_list[-1]//100 != node_count//100:
                    self.node_count_list.append(node_count)
                    self.train_acc_list.append(getAccuracy(train_y, clf.predict(train_x)))
                    self.test_acc_list.append(getAccuracy(test_y, clf.predict(test_x)))
                    self.val_acc_list.append(getAccuracy(val_y, clf.predict(val_x)))

                
                    print("Time:",time.time() - self.start_time)
                    print("Prune=  "+"Node:"+str(node_count))

                if root.val_pos > root.val_neg:
                    root.label = 1
                    return root.val_neg 
                else:
                    root.label = 0
                    return root.val_pos
                    
                
    
    def fit(self):
        self.start_time = time.time()
        self.classifyValidationSet()
        print("Classifying done")
        self.prune()
        self.time_to_fit = time.time() - self.start_time
        
        


def Q2_2_scratch():
	clf = joblib.load("clfQ1")
	prune_clf = Prune(clf.root)
	prune_clf.fit()
	return prune_clf


def Q2_2():
	prune_clf = None
	if os.path.isfile("./prune_clf"):
		prune_clf = joblib.load("prune_clf")
	else:
		prune_clf = Q2_1_scratch()

	print("Node count:", getSubtreeNodeCount(prune_clf.root))
	print("Leaf count:", leafCount(prune_clf.root))
	print("Tree depth:", getDepth(prune_clf.root))
	print("Train Accuracy", getAccuracy(train_y, predict(prune_clf.root,train_x)) )
	print("Test Accuracy", getAccuracy(test_y, predict(prune_clf.root,test_x)) )
	print("Validation Accuracy", getAccuracy(val_y, predict(prune_clf.root,val_x)) )

	plotQ2(prune_clf, "fig2_1")

if __name__ == "__main__":
	print("Q1 is running")
	Q1()
	print("Q2 is running")
	print("APPROACH 1")
	Q2_1()
	print("APPROACH 2")
	Q2_2()
