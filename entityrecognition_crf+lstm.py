# -*- coding: utf-8 -*-
"""EntityRecognition-CRF+LSTM.ipynb

Used a google Colab notebook to work this out. 

Storing this here for future use. 

- CYRUS DSOUZA
"""

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import pandas as pd
import numpy as np 
import io
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# from google.colab import files
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import io
# upload = files.upload()


#USED THE NER DATASET FROM A CURATED WEBSITE TO PROCESS THIS. 
df = pd.read_csv(io.BytesIO(upload['ner_dataset.csv']), encoding = 'latin-1')

df2.fillna(method = 'ffill', inplace = True)

df2.head(50
        )

# words = list(set(df['Word'].values))
# words.append('ENDPAD'
#             )
# n_words = len(words)
# print(n_words)
# words

words = list(set(df2['Word'].values))
words.append('ENDPAD')
n_words = len(words)
print(n_words)

tags  = list(set(df2['Tag'].values))
n_tags = len(tags)
len(tags)

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        
        agg_func = lambda x : [(w,p,t) for w,p,t in zip(x['Word'].values.tolist(),
                                                       x['POS'].values.tolist(),
                                                       x['Tag'].values.tolist())]
        
        self.grouped = self.data.groupby("SentenceNumber").apply(agg_func)
        self.sentences = [s for s in self.grouped]

getter = SentenceGetter(df2)
sentences = getter.sentences
plt.style.use('ggplot')
plt.hist([len(s) for s in sentences], bins = 50)
plt.show()

word2idx =  {w:i for i,w in enumerate(words)}
tag2idx =  {t:i for i,t in enumerate(tags)}


maxlen = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
y = [[tag2idx[w[2]] for w in s ] for s in sentences]

X = pad_sequences(maxlen = maxlen, sequences = X, value = n_words-1, padding = 'post')
y = pad_sequences(maxlen = maxlen, sequences = y, value = tag2idx['O'], padding = 'post')

y = [to_categorical(i, num_classes = n_tags) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# !pip install git+https://www.github.com/keras-team/keras-contrib.git

from keras_contrib.layers import CRF

input = Input(shape=(maxlen,))
model = Embedding(input_dim=n_words + 1, output_dim=20, \
                  input_length=maxlen, mask_zero=True)(input)# 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = Dropout(0.1)(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

# input.shape
model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

print(model.summary())

# LOG_DIR = './log'
# get_ipython().system_raw(
#     'tensorboard --logdir {} --host 0.0.0.0 --port 6016 &'
#     .format(LOG_DIR))

# get_ipython().system_raw('./ngrok http 6016 &')

# ! curl -s http://localhost:4040/api/tunnels | python3 -c \
#     "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

# from keras.callbacks import TensorBoard
# tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
#                          write_graph=True,
#                          write_grads=True,
#                          batch_size=32,
#                          write_images=True)

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5,
                    validation_split=0.1, verbose=1)#, callbacks = [tbCallBack])

hist = pd.DataFrame(history.history)
hist
plt.style.use('ggplot')
plt.figure(figsize = (12,12))
plt.plot(hist['crf_viterbi_accuracy'])
plt.plot(hist['val_crf_viterbi_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# pip install seqeval

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

test_pred = model.predict(X_test, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
test_labels = pred2label(y_test)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

print(classification_report(test_labels, pred_labels))

i = 90
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis= -1)
true = np.argmax(y_test[i], -1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_test[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))

test_sentences = "restated master commercial agreement was approved between amd and dxc".lower().split()

x_test_sentence = pad_sequences(maxlen = maxlen, sequences = [[word2idx.get(w,0) for w in test_sentences]], padding = 'post', value= 0)

x_test_sentence

p = model.predict(x_test_sentence)
p = np.argmax(p, axis = -1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentences, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))

