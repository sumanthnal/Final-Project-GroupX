import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import tensorflow as tf

from transformers import BertTokenizer, TFBertForSequenceClassification
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping

target_encoder = LabelEncoder()

train = pd.read_csv("train_set.csv", index_col=0)
x_train = train.iloc[:,:-1]
y_train = target_encoder.fit_transform(train['type'])

# Load the test data
test = pd.read_csv("test_set.csv", index_col=0)
x_test = test.iloc[:,:-1]
y_test = target_encoder.fit_transform(test['type'])

# Load the val data
val = pd.read_csv("val_set.csv", index_col=0)
x_val = val.iloc[:,:-1]
y_val = target_encoder.fit_transform(val['type'])


models_accuracy={}


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)





model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Fine-tune the model
optimizer = Adam(learning_rate=2e-5)
loss = BinaryCrossentropy(from_logits=True)
metric = BinaryAccuracy()
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
es_callback = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=32, callbacks=[es_callback])

loss, acc = model.evaluate(x_test, y_test)
print(acc)

print(models_accuracy)