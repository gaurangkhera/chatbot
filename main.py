import nltk 
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn as tfl
import tensorflow as tf
import random as rand 
import json

stemmer = LancasterStemmer()

with open('intents_data.json') as f:
    data = json.load(f)
    
words = []
labels = []
docs_x = []
docs_y = []
intents = data['intents']

for i in intents:
    for j in i['patterns']:
        wl = nltk.word_tokenize(j)
        words.extend(wl)
        docs_x.append(j)
        docs_y.append(i['tag'])
    if i['tag'] not in labels:
        labels.append(i['tag'])
        
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
# print(labels)
labels = sorted(labels)

training = []
out = []

out_e = [0 for _ in range(len(labels))]

for i, doc in enumerate(docs_x):
    bag = []
    wl = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wl:
            bag.append(1)
        else:
            bag.append(0)
    out_r = out_e[:]
    out_r[labels.index(docs_y[i])] = 1
    training.append(bag)
    out.append(out_r)
    
training = np.array(training)
out = np.array(out)

# tf.reset_default_graph()
tf.compat.v1.reset_default_graph() 
net = tfl.input_data(shape=[None, len(training[0])])
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, len(out[0]), activation='softmax')
net = tfl.regression(net)

#shesh

model = tfl.DNN(net)
model.fit(training, out, n_epoch=10000, batch_size=8, show_metric=True)
model.save('model.tflearn')