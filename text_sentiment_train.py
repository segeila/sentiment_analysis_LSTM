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
from keras.callbacks import EarlyStopping


#Read dataset from csv file
data = pd.read_csv('samples/twitter_sentiment.csv', error_bad_lines=False, usecols=[1,3], names=['score', 'text'], skiprows = 1)

#Check first 5 entries
print(data.head())

#Transform text
data.text = data.text.apply(lambda x: x.lower())
data.text = data.text.apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',x)))

#Create and fit tokenizer
max_features = 2000 #Number of words to consider
tokenizer = Tokenizer(num_words = max_features, split = ' ')
tokenizer.fit_on_texts(data.text.values)
x = tokenizer.texts_to_sequences(data.text.values)

#Saving tokenizer to use for inference
with open('tokenizer/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Check how single tweet looks like after transformation
print(x[np.random.randint(len(x))])

#Create training data and training labels
X_train = sequence.pad_sequences(x)
y_train = data.score

#Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='auto')

#Create model
model = Sequential()
model.add(Embedding(input_dim = max_features, output_dim = 200, input_length=X_train.shape[1]))
model.add(LSTM(200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Fit model
model.fit(X_train, y_train, epochs = 100, batch_size = 1024, validation_split = 0.05, verbose=1, callbacks=[early_stopping])

#Save weights
model.save_weights('models/sentiment_weights.h5')