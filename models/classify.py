import gensim
import gensim.downloader as api
from gensim.utils import simple_preprocess
import numpy as np
import pickle


def model_load():
    global wv
    global Classifier
    wv= api.load('word2vec-google-news-300')
    Classifier = pickle.load(open('/Users/akshatvaja/Documents/Tata_Project/models/rfc.pkl','rb'))

def avg_word2vec(doc): 
    valid_words = [wv[word] for word in doc if word in wv]
    if not valid_words:
        return np.zeros(wv.vector_size)
    return np.mean(valid_words, axis=0)


def pred_function(sent):

    sample = simple_preprocess(sent)
    sample_vector = avg_word2vec(sample)
    X_test = sample_vector.reshape(1, -1)
    y_pred = Classifier.predict(X_test)
    return y_pred[0]