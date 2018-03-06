import pandas as pd
import numpy as np
import keras
import re
import pickle

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

feedback = input('Write down your feedback \n')

feedback = feedback.lower()
feedback = re.sub('[^a-zA-Z0-9\s]','', feedback)

feedback = np.array(feedback.split(' '))
print(feedback)

#data = pd.read_csv('twitter_sentiment.csv', error_bad_lines=False, usecols=[1,3], names=['score', 'text'], skiprows = 1)

#data.text = data.text.apply(lambda x: x.lower())
#data.text = data.text.apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',x)))



#x = tokenizer.texts_to_sequences(data.text.values)
#print(x[128])
#print(data.text.iloc[128])


feedback = tokenizer.texts_to_sequences(feedback)
feedback = [item for sublist in feedback for item in sublist]
feedback = np.array(feedback)
feedback = feedback.reshape(-1, len(feedback))
print(feedback.shape)
feedback = sequence.pad_sequences(feedback, maxlen = 84)
print(feedback)

max_features = 2000
sequence_length = 84

model = Sequential()
model.add(Embedding(input_dim = max_features, output_dim = 200, input_length=sequence_length))
model.add(LSTM(200))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.load_weights('sentiment_weights_02.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(('Feedback was %.3f' % (model.predict(feedback)*100)) + '% positive' )  

