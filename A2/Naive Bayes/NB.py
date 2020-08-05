import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from collections import Counter 
import sys
import math
import re
import random
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt  
import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2
import pickle
import joblib


def saveModel(model, file_name):
    joblib.dump(model, filename)
    return True

def readModel(file_name):
    return joblib.load(file_name)



tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

split_regex = "\.|,| "
split_regex1= "\.|,|\?|\)|\(|!|#| "



def splitTweets(sentence):
    word_tokens = re.split(split_regex1, sentence)
    return word_tokens

def readData(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None, encoding="latin-1")
    test_data = pd.read_csv(test_path, header=None, encoding="latin-1")

    train_data = train_data[train_data[0]!=2]
    test_data = test_data[test_data[0]!=2]
    return (train_data[[0,5]],test_data[[0,5]])


def calculatePhi(data):
    m = data.shape[0]
    positive_count = data[data[0]!=4].shape[0]
    phi = positive_count/m
    return phi


def getVocab(data):
    vocab = {}
    data_list = list(data[5])
    y_list = list(data[0])

    data_list =  [re.split(split_regex, el) for el in tqdm(data_list) if type(el)==str]

    for el in tqdm(data_list):
        for w in el:
            if w not in vocab:
                vocab[w] = 1
                continue
            vocab[w]+=1
    return vocab


def plot_cm(actual, predicted, title):
    cm = confusion_matrix(actual, predicted)
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


def calculateAccuracy(actual, predicted):
    count = 0
    for y1, y2 in zip(actual, predicted):
        if y1==y2:
            count+=1
    return count/len(predicted)

def getBasicData(train_data):
    vocab = getVocab(train_data)

    unique_word = len(vocab)
    total_word = sum(list(vocab.values()))


    p_train_data = train_data[train_data[0]==4]
    n_train_data = train_data[train_data[0]==0]

    p_vocab = getVocab(p_train_data)
    n_vocab = getVocab(n_train_data)

    del p_vocab[""]
    del n_vocab[""]

    prob_w_y_1 = {}
    prob_w_y_0 = {}

    p_total_word = sum(p_vocab.values())
    n_total_word = sum(n_vocab.values())

    for w in tqdm(vocab):
        try:
            prob_w_y_1[w] = (p_vocab[w] + 1) / (p_total_word + unique_word)
        except:
            prob_w_y_1[w] = 1 /  (p_total_word + unique_word)
        
        try:
            prob_w_y_0[w] = (n_vocab[w] + 1) / (n_total_word + unique_word)
        except:
            prob_w_y_0[w] = 1 /  (n_total_word + unique_word)
    
    return (p_train_data, n_train_data, unique_word, p_total_word, n_total_word,  prob_w_y_0, prob_w_y_1)


def q1TestData(train_data, test_data, phi, prob_w_y_0, prob_w_y_1, p_total_word, n_total_word, unique_word):
    test_tweets = list(test_data[5])
    test_tweets = [re.split(split_regex, el) if type(el)==str else " " for el in test_tweets]
    y_class_test_pred = []
    phi_y = phi

    p_prob = []
    n_prob = []

    for tweet in tqdm(test_tweets):
        p_p = math.log(phi_y)
        n_p = math.log(1-phi_y)
        pred_y = 0
        for w in tweet:
            try:
                p_p+= math.log(prob_w_y_1[w])
                n_p+= math.log(prob_w_y_0[w])
            except:
                p_p+= 1 / (p_total_word + unique_word)
                n_p+= 1 / (n_total_word + unique_word)
        pred_y = 4 if math.exp(p_p) >= math.exp(n_p) else 0
        y_class_test_pred.append(pred_y)
        p_prob.append(math.exp(p_p))
        n_prob.append(math.exp(n_p))


    return y_class_test_pred, p_prob, n_prob


def cleanTweet(sentence):
    word_tokens = tknzr.tokenize(sentence)
    filtered_sentence = [stemmer.stem(w) for w in word_tokens if (w and (not (w in stop_words)))]
    return filtered_sentence

def readCleanFile():
    data = pd.read_csv("cleaned.csv", header=None, encoding="latin-1")
    r_data = pd.DataFrame()
    r_data[0] = data[0]
    r_data[5] = data[1]

    r_data[5].replace(np.nan, "zero")
    r_data[5] = r_data[5].astype(str)
    return r_data


def readBigramFile():
    data = pd.read_csv("train_bi.csv", header=None, encoding="latin-1")
    r_data = pd.DataFrame()
    r_data[0] = data[0]
    r_data[5] = data[1]

    r_data[5].replace(np.nan, "zero")
    r_data[5] = r_data[5].astype(str)
    return r_data


def readPosFile():
    data = pd.read_csv("train_pos.csv", header=None, encoding="latin-1")
    r_data = pd.DataFrame()
    r_data[0] = data[0]
    r_data[5] = data[1]

    r_data[5].replace(np.nan, "zero")
    r_data[5] = r_data[5].astype(str)
    return r_data


def plot_roc(y, p_prob, n_prob, title):
    p_fpr, p_tpr, p_thresholds = roc_curve(y, p_prob, pos_label=4)
    n_fpr, n_tpr, n_thresholds = roc_curve(y, n_prob, pos_label=4)
    p_auc = auc(p_fpr, p_tpr)
    n_auc = auc(n_fpr, n_tpr)


    plt.plot(p_fpr, p_tpr, color='blue', label="Positive, area="+str(round(p_auc, 2)))
    plt.plot([0,1], linestyle='--', color='g', label="Baseline")

    # plt.plot(n_fpr, n_tpr, color='red', label="Negative, area="+str(round(n_auc, 2)))
    plt.plot(1,1, )
    plt.title("ROC Curve for " + title)
    plt.xlabel("False Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate", fontweight='bold')
    plt.legend()
    plt.show()


def q1(train_data, test_data):
    phi = calculatePhi(train_data)

    p_train_data, n_train_data, unique_word, p_total_word, n_total_word,  prob_w_y_0, prob_w_y_1= getBasicData(train_data)

    # accuracy on train data
    y_class_train_pred, p_prob, n_prob = q1TestData(train_data, train_data, phi, prob_w_y_0, prob_w_y_1, p_total_word, n_total_word, unique_word)
    y_class_train_orig = list(train_data[0])
    acc1 = calculateAccuracy(y_class_train_orig, y_class_train_pred)
    print("Accuracy on train data :", acc1)
    plot_cm(y_class_train_orig, y_class_train_pred, "Confusion Matrix: Train Data")

    plot_roc(y_class_train_orig, p_prob, n_prob, "training data")


    # accuracy on test data
    y_class_test_pred, p_prob, n_prob = q1TestData(train_data, test_data, phi, prob_w_y_0, prob_w_y_1, p_total_word, n_total_word, unique_word)
    y_class_test_orig = list(test_data[0])
    acc2 = calculateAccuracy(y_class_test_orig, y_class_test_pred)
    print("Accuracy on test data :", acc2)
    plot_cm(y_class_test_orig, y_class_test_pred, "Confusion Matrix: Test Data")
    plot_roc(y_class_test_orig, p_prob, n_prob, "test data")


    # accuracy on majority class
    majority_class = 0
    if len(p_train_data) > len(n_train_data):
        majority_class = 4
    y_class_test_maj = [majority_class] * len(test_data)
    acc3 = calculateAccuracy(y_class_test_orig, y_class_test_maj)
    print("Accuracy on majority prediction data :", acc3)
    plot_cm(y_class_test_orig, y_class_test_maj, "Confusion Matrix: Majority prediction")
    # plot_roc(y_class_test_orig, p_prob, n_prob, "majority prediction")

    # accuracy on random class
    y_class_rand = []
    for tweet in range(len(test_data)):
        y_class_rand.append(random.choice([0,4]))
    acc4 = calculateAccuracy(y_class_test_orig, y_class_rand)
    print("Accuracy on random prediction data :", acc4)
    plot_cm(y_class_test_orig, y_class_rand, "Confusion Matrix: Random prediction")


def clean(test_data):
    test_tweets = list(test_data[5])

    split_tweets = [splitTweets(tweet) for tweet in tqdm(test_tweets)]
    split_tweets_strs = [" ".join(tweet) for tweet in split_tweets]
    cleaned_tweet = [cleanTweet(tweet) for tweet in tqdm(split_tweets_strs)]
    cleaned_tweet_strs = [" ".join(tweet) for tweet in cleaned_tweet]
    
    pd_cleaned_tweet  = pd.DataFrame(cleaned_tweet_strs)
    pd_cleaned_tweet.replace(np.nan, "zero")
    pd_cleaned_tweet = pd_cleaned_tweet.astype(str)

    test_data[5] = pd_cleaned_tweet
    return test_data


def q2(train_data, test_data):
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    print(test_data)

    # one time task
    # tweets = list(train_data[5])
    # split_tweets = [splitTweets(tweet) for tweet in tqdm(tweets)]
    # split_tweets_strs = [" ".join(tweet) for tweet in split_tweets]


    # cleaned_tweet = [cleanTweet(tweet) for tweet in tqdm(split_tweets_strs)]
    # cleaned_tweet_strs = [" ".join(tweet) for tweet in cleaned_tweet]
    # # save to file
    # y_tweets = list(train_data[0])
    # df_x = pd.DataFrame(cleaned_tweet_strs)
    # df_y = pd.DataFrame(y_tweets)
    # df_y[5] = df_x
    # cleaned_df = df_y
    # cleaned_df.to_csv("cleaned.csv", header=False, index=False, encoding="latin-1")

    #read clean file
    # todo add if/else if file not exist

    
    test_data = clean(test_data)


    # test_tweets = list(test_data[5])

    # split_tweets = [splitTweets(tweet) for tweet in tqdm(test_tweets)]
    # split_tweets_strs = [" ".join(tweet) for tweet in split_tweets]
    # cleaned_tweet = [cleanTweet(tweet) for tweet in tqdm(split_tweets_strs)]
    # cleaned_tweet_strs = [" ".join(tweet) for tweet in cleaned_tweet]

    # pd_cleaned_tweet  = pd.DataFrame(cleaned_tweet_strs)
    # pd_cleaned_tweet.replace(np.nan, "zero")
    # pd_cleaned_tweet = pd_cleaned_tweet.astype(str)


    # test_data[5] = pd_cleaned_tweet
    # print(test_data)

    # test_data.to_csv("test_data_1", header=False, index=False, encoding="latin-1")







    cleaned_train_data = readCleanFile()
    q1(cleaned_train_data, test_data)





    
vectorizer = TfidfVectorizer(min_df=0.00065)


def q3(train_data, test_data):
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    test_data = clean(test_data)

    cleaned_train_data = readCleanFile()
    X = cleaned_train_data[5]
    Y = cleaned_train_data[0]

    # vectorizer = TfidfVectorizer(min_df=0.00065)

    # X_vect = vectorizer.fit_transform(X)

    # print("fitting")
    # clf = nb_partialFit(X_vect, Y)
    # joblib.dump((clf, vectorizer), "TFIDF_clf")
    # print("fitted")

    clf, vectorizer = joblib.load("TFIDF_clf")

    # test
    test_x = test_data[5]
    test_y = test_data[0]
    print("vectorizer transform")
    X_test_vect = vectorizer.transform(test_x)
    print("vectorizer transform done")

    p_test = clf.predict(X_test_vect.todense())
    a_test = calculateAccuracy(test_y, p_test)
    print(a_test)
    plot_cm(test_y, p_test, "TF-IDF: Test Data")

    prob = clf.predict_proba(X_test_vect.todense())
    n_prob = prob[:,0]
    p_prob = prob[:,1]
    
    plot_roc(test_y, p_prob, n_prob, "ROC TF-IDF: Test Data ")

    return True




def nb_partialFit(X, Y):
    m = X.shape[0]
    clf = GaussianNB()

    chunk_size = 10000
    n_chunk = int(m / chunk_size)-1
    for i in tqdm(range(n_chunk)):
        x_partial = X[i*chunk_size:(i+1)*chunk_size]
        y_partial = Y[i*chunk_size:(i+1)*chunk_size]
        clf.partial_fit(x_partial.todense(), y_partial, classes=[0,4])

    return clf

        


def nb_SelectPercentile(train_data, test_data):
    cleaned_train_data = readCleanFile()

    X = cleaned_train_data[5]
    Y = cleaned_train_data[0]

    percentile = 10

    vectorizer = TfidfVectorizer(min_df=0.00065)
    sp = SelectPercentile(chi2, percentile=percentile)
    X_vect = vectorizer.fit_transform(X)
    X_new = sp.fit_transform(X_vect, Y)
    print(f"New shape after SelectPercentile {X_new.shape}")
    clf = nb_partialFit(X_new, Y)
    # joblib.dump((clf, vectorizer, sp), "TFIDF_SelectPercentile")

    # clf, vectorizer, sp = joblib.load("TFIDF_SelectPercentile")

    # test
    x_test = test_data[5]
    y_test = test_data[0]

    x_test_vect = vectorizer.transform(x_test)
    x_test_new = sp.transform(x_test_vect)

    y_pred = clf.predict(x_test_new.todense())

    acc1 = calculateAccuracy(y_test, y_pred)
    print(f"Accuracy on test data after 2 percentile feature selection:", acc1)

    plot_cm(y_test, y_pred, "Confusion Matrix: SelectPercentile - 10 Percentile on Test Data")


    prob = clf.predict_proba(x_test_new.todense())
    n_prob = prob[:,0]
    p_prob = prob[:,1]
    
    plot_roc(y_test, p_prob, n_prob, "ROC TF-IDF: SelectPercentile - 10 Percentile on Test Data ")

    return clf





def q4(train_data, test_data):
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    test_data = clean(test_data)

    clf = nb_SelectPercentile(train_data, test_data)

def pd_bigram(s):
    # print(s)
    if(s):
        try:
            extras = bi_count_vectorizer.fit_transform([s])
            features = bi_count_vectorizer.get_feature_names()

            jf = " ".join(["".join(el.split()) for el in features])

            return (s + " " + jf)
        except ValueError:
            # print("error")
            return s
    
    return " "

from pandas import Panel
from tqdm import tqdm_pandas
tqdm_pandas(tqdm())
bi_count_vectorizer = CountVectorizer(ngram_range = (2,2))
def applyBigrams(x):

    # print("applying bigrams")
    mod_x = x.apply(pd_bigram)
    return mod_x





def q5(train_data, test_data):
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    # test_data = clean(test_data)

    # generate bigram
    # cleaned_train_data = readCleanFile()
    # bg_data = applyBigrams(cleaned_train_data[5])
    # cleaned_train_data[5] = bg_data
    # cleaned_train_data.to_csv("train_bi.csv", header=False, index=False, encoding="latin-1")


    # read bigram file
    bigram_file = readBigramFile()
    bi_clean_test = applyBigrams(test_data[5])
    test_data[5] = bi_clean_test
    
    q1(bigram_file, test_data)


from nltk import pos_tag
# nltk.download("averaged_perceptron_tagger")
igonre_pos = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WP', 'TO', ","}


def pd_pos(s):
    # print(s)
    try:
        sentence = nltk.word_tokenize(s)
        sentence = pos_tag(sentence)
        final = " ".join([el[0] for el in sentence if el[1] not in igonre_pos])
        return final
    except:
        return s

def applyPos(x):
    mod_x = x.apply(pd_pos)
    return mod_x

def  q6(train_data, test_data):
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    # generate pos and filter
    # pos_train_data = applyPos(train_data[5])
    # train_data[5] = pos_train_data
    # train_data.to_csv("train_pos.csv", header=False, index=False, encoding="latin-1")

    # posFile = clean(readPosFile())
    # posFile.to_csv("train_pos.csv", header=False, index=False, encoding="latin-1")

    train_pos =  readPosFile()
    pos_test = applyPos(test_data[5])
    test_data[5] = clean(pos_test)
    
    q1(train_pos, test_data)






train_path = './data/train_data.csv'
test_path = './data/test_data.csv'
def main(train_path, test_path):
    train_data, test_data = readData(train_path, test_path)
    # print(test_data.shape)

    # parta, parb, partc
    # 
    q1(train_data, test_data)

    # partd
    q2(train_data, test_data)

    # partf
    q3(train_data, test_data)

    # part f 2nd part
    q4(train_data, test_data)

    # part e feature engineer: bigram
    q5(train_data, test_data)

    
    # part e feature engineer: POS
    q6(train_data, test_data)
    


main(train_path, test_path)






