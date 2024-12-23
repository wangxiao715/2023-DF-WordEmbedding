import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.models import  Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import  Conv1D
from keras.layers import  MaxPooling1D
import keras
from keras.models import load_model

import csv
import codecs
import nltk
from nltk.tokenize import word_tokenize
#nltk.download("stopwords")
#nltk.download("wordnet")
from tempfile import mkstemp
import matplotlib

lst_dics = []
with codecs.open('D:机器学习/train.csv', encoding='utf-8-sig') as csv_file:
   for dic in csv.DictReader(csv_file, skipinitialspace = True):
       lst_dics.append(dic)
csv_file.close

## create dtf
dtf = pd.DataFrame(lst_dics)
dtf = dtf[dtf["label"].isin(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']) ][["node_id","label","text"]]


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
   ## clean
   text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
           
   ## Tokenize
   lst_text = text.split()    ## remove Stopwords
   if lst_stopwords is not None:
      lst_text = [word for word in lst_text if word not in
                  lst_stopwords]
               
   ## Stemming 
   if flg_stemm == True:
       ps = nltk.stem.porter.PorterStemmer()
       lst_text = [ps.stem(word) for word in lst_text]
               
   ## Lemmatisation
   if flg_lemm == True:
       lem = nltk.stem.wordnet.WordNetLemmatizer()
       lst_text = [lem.lemmatize(word) for word in lst_text]
           
   ## back to string from list
   text = " ".join(lst_text)
   return text
##'discription' is useless
lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords.append("description")

dtf["text_clean"] = dtf["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm = True, lst_stopwords = lst_stopwords))
dtf_train = dtf


def create_tokenizer(lines):
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(lines)
	return  tokenizer
tokenizer=create_tokenizer(dtf_train["text_clean"])
dic_vocabulary = tokenizer.word_index
max_length=max([len(string.split()) for string in dtf_train["text_clean"]])
#print('最长词语 句子：',max_length)


def encode_docs(tokenizer,max_length,docs):
	encoded=tokenizer.texts_to_sequences(docs)
	padded=pad_sequences(encoded,maxlen=max_length,padding='post')
	return padded

X_train=encode_docs(tokenizer,max_length,dtf_train["text_clean"])
y_train=dtf_train["label"]
y_train=np.array(y_train)
Y_train=keras.utils.to_categorical(y_train,24)

i = 0
len_txt = len(dtf_train["text_clean"].iloc[i].split())
print("from: ", dtf_train["text_clean"].iloc[i], "| len:", len_txt)
len_tokens = len(X_train[i])
print("to: ", X_train[i], "| len:", len(X_train[i]))
print("check: ", dtf_train["text_clean"].iloc[i].split()[0],  " -- idx in vocabulary -->",  dic_vocabulary[dtf_train["text_clean"].iloc[i].split()[0]])
print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")

#定义神经网络模型
#使用Embedding层作为第一层，需要指定词汇表大小，实值向量空间的额大小以及输入文档的最大长度。词汇表大小是我们词汇表中的单词总数，加上一个未知单词
vocad_size=len(tokenizer.word_index)+1
print('词汇表大小：',vocad_size)

#定义神经网络模型
def define_model(vocad_size,max_length):
	model=Sequential()
	model.add(Embedding(vocad_size,130,input_length=max_length))
	model.add(Conv1D(filters=96,kernel_size=5,activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(48,activation='relu'))
	model.add(Dense(24,activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	return model

print("model init")
def model_train():
    model = define_model(vocad_size, max_length)
    model.fit(X_train, Y_train, epochs=10, verbose=2)
    model.save('model.h5')
    #model_train()#
model_train()
print("model trained")

from sklearn.metrics import accuracy_score#分类器评估
model=load_model('model.h5')
predict_y=model.predict(X_train)
pred_y=np.argmax(predict_y,axis=1)
test_y=np.array(dtf_train['label'])
test_y=np.int64(test_y)
correct_prediction = np.equal(pred_y,test_y)
print(np.mean(correct_prediction))

lst_dics=[]
with codecs.open('D:机器学习/test.csv', encoding='utf-8-sig') as csv_file:
   for dic in csv.DictReader(csv_file, skipinitialspace = True):
       lst_dics.append(dic)
csv_file.close

dtf = pd.DataFrame(lst_dics)
dtf = dtf[["node_id","text"]]

dtf["text_clean"] = dtf["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm = True, lst_stopwords = lst_stopwords))
dtf_test = dtf

X_test=encode_docs(tokenizer,max_length,dtf_test["text_clean"])

predict_y=model.predict(X_test)
pred_y=np.argmax(predict_y,axis=1)

data = {'node_id':np.array(dtf_test['node_id'],), 'label':pred_y}
df = pd.DataFrame(data)
filename='D:机器学习/try0.csv'
df.to_csv(filename, index = False)

