import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Embedding, Bidirectional
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# import tensorflow as tf

from transformers import BertTokenizer, TFBertForSequenceClassification
# from keras.optimizers import Adam
# from keras.losses import BinaryCrossentropy
# from keras.metrics import BinaryAccuracy
# from keras.callbacks import EarlyStopping


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# nltk.download('wordnet')
# nltk.download('stopwords')
#nltk.download('punkt')

target_encoder = LabelEncoder()

# Load the train data
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

# Logistic Regression

model_log = LogisticRegression(max_iter=3000,C=0.5,n_jobs=-1)
model_log.fit(x_train, y_train)
print("Checking performance of Logistic Regression")
train_report = classification_report(y_train, model_log.predict(x_train), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)

macro_precision = train_report['weighted avg']['precision']
macro_recall = train_report['weighted avg']['recall']
macro_f1 = train_report['weighted avg']['f1-score']

print("Train Precision " + str(macro_precision))
print("Train Recall " + str(macro_recall))
print("Train F1 " + str(macro_f1))

test_report = classification_report(y_test, model_log.predict(x_test), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)
#
macro_precision =  test_report['weighted avg']['precision']
macro_recall = test_report['weighted avg']['recall']
macro_f1 = test_report['weighted avg']['f1-score']

print("Test Precision " + str(macro_precision))
print("Test Recall " + str(macro_recall))
print("Test F1 " + str(macro_f1))



val_report = classification_report(y_val, model_log.predict(x_val), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)
#
macro_precision =  val_report['weighted avg']['precision']
macro_recall = val_report['weighted avg']['recall']
macro_f1 = val_report['weighted avg']['f1-score']

print("Validation Precision " + str(macro_precision))
print("Validation Recall " + str(macro_recall))
print("Validation F1 " + str(macro_f1))

models_accuracy['logistic regression']=accuracy_score(y_val,model_log.predict(x_val))


# Multinomial Naive Bayes

model_multinomial_nb=MultinomialNB()
model_multinomial_nb.fit(x_train, y_train)

train_report = classification_report(y_train, model_multinomial_nb.predict(x_train), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)
test_report = classification_report(y_test, model_multinomial_nb.predict(x_test), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)
val_report = classification_report(y_val, model_multinomial_nb.predict(x_val), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)

macro_precision = train_report['weighted avg']['precision']
macro_recall = train_report['weighted avg']['recall']
macro_f1 = train_report['weighted avg']['f1-score']

print("Train Precision " + str(macro_precision))
print("Train Recall " + str(macro_recall))
print("Train F1 " + str(macro_f1))

macro_precision =  test_report['weighted avg']['precision']
macro_recall = test_report['weighted avg']['recall']
macro_f1 = test_report['weighted avg']['f1-score']

print("Test Precision " + str(macro_precision))
print("Test Recall " + str(macro_recall))
print("Test F1 " + str(macro_f1))

macro_precision =  val_report['weighted avg']['precision']
macro_recall = val_report['weighted avg']['recall']
macro_f1 = val_report['weighted avg']['f1-score']

print("Validation Precision " + str(macro_precision))
print("Validation Recall " + str(macro_recall))
print("Validation F1 " + str(macro_f1))

models_accuracy['multinomial bayes']=accuracy_score(y_val,model_multinomial_nb.predict(x_val))


# XGBoost

model_xgb=XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=5,n_estimators=50,learning_rate=0.1)
model_xgb.fit(x_train,y_train)

train_report = classification_report(y_train, model_xgb.predict(x_train), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)
test_report = classification_report(y_test, model_xgb.predict(x_test), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)
val_report = classification_report(y_val, model_xgb.predict(x_val), target_names=target_encoder.inverse_transform([i for i in range(16)]), output_dict=True)

macro_precision = train_report['weighted avg']['precision']
macro_recall = train_report['weighted avg']['recall']
macro_f1 = train_report['weighted avg']['f1-score']

print("Train Precision " + str(macro_precision))
print("Train Recall " + str(macro_recall))
print("Train F1 " + str(macro_f1))

macro_precision =  test_report['weighted avg']['precision']
macro_recall = test_report['weighted avg']['recall']
macro_f1 = test_report['weighted avg']['f1-score']

print("Test Precision " + str(macro_precision))
print("Test Recall " + str(macro_recall))
print("Test F1 " + str(macro_f1))

macro_precision =  val_report['weighted avg']['precision']
macro_recall = val_report['weighted avg']['recall']
macro_f1 = val_report['weighted avg']['f1-score']

print("Validation Precision " + str(macro_precision))
print("Validation Recall " + str(macro_recall))
print("Validation F1 " + str(macro_f1))

models_accuracy['XGBoost']=accuracy_score(y_val,model_xgb.predict(x_val))

# # Now, time for LSTM
# num_words = 10000
#
# # Define the model architecture
# model = Sequential()
# model.add(Embedding(num_words, 32))
# model.add(Bidirectional(LSTM(32)))
# model.add(Dense(1, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#
# # Train the model
# history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# Load the pre-trained BERT model

# Time for BERT?
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
#
# # Fine-tune the model
# optimizer = Adam(learning_rate=2e-5)
# loss = BinaryCrossentropy(from_logits=True)
# metric = BinaryAccuracy()
# model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# es_callback = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
# history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=32, callbacks=[es_callback])
#
# loss, acc = model.evaluate(x_test, y_test)
# print(acc)
#
# print(models_accuracy)

