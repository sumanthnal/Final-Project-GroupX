import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
# nltk.download('stopwords')
#nltk.download('punkt')

data = pd.read_csv("mbti_1.csv")
print("length of data: "  + str(len(data)))

# Remove URL's
data['posts'] = data['posts'].apply(lambda s: ' '.join(re.sub(r'http\S+', '', s).split()))
# # Remove HTML tags
data['posts'] = data['posts'].apply(lambda s: ' '.join(re.sub(r'<[^>]+>', '', s).split()))

# # Remove punctuations and convert text to lowercase
data['posts'] = data['posts'].apply(lambda s: s.translate(str.maketrans('', '', string.punctuation)).lower())

# # Remove digits
data['posts'] = data['posts'].apply(lambda s: ' '.join(re.sub(r'\d+', '', s).split()))

# # Remove special characters and symbols
data['posts'] = data['posts'].apply(lambda s: ' '.join(re.sub(r'[^\w\s]', '', s).split()))

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if not token in stop_words]
    text = ' '.join(tokens)
    return text


data['posts'] = data['posts'].apply(remove_stop_words)

nas = pd.isnull(data['posts'])
data['na'] = nas
no_nas = data[data.na != True]


data = no_nas.drop(['na'], axis=1)

print("length of data after preprocessing: "  + str(len(data)))

train_data, test_data = train_test_split(data,test_size=0.3,random_state=20, stratify=data.type)
test_data, val_data = train_test_split(test_data,test_size=0.2, random_state=20, stratify=test_data.type)
print("train data length: " + str(len(train_data)))

print("test data length: " + str(len(test_data)))

print("val data length: " + str(len(val_data)))


# Tokenize the texts
class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>2]

#print(train_data.info())
vectorizer=TfidfVectorizer(max_features=5000,stop_words='english',tokenizer=Lemmatizer())
vectorizer.fit(train_data.posts)


x_train = vectorizer.transform(train_data.posts).toarray()
print("X train length")
print(len(x_train))
x_test = vectorizer.transform(test_data.posts).toarray()
print("X test length")
print(len(x_test))
x_val = vectorizer.transform(val_data.posts).toarray()
print("X val length")
print(len(x_val))







print("Create training CSV with vectorized X and encoded Y")
x_train_buffer = pd.DataFrame(x_train)
y_train_buffer = pd.DataFrame(train_data['type'].values)

x_train_buffer["type"] = y_train_buffer
train_set = x_train_buffer
#train_set = pd.concat([x_train_buffer, y_train_buffer], axis=1, ignore_index=True)
#train_set = train_set.reset_index(drop=True)
# train_set = x_train_buffer.join(y_train_buffer)
print("train dataset length")
print(len(train_set))
print(train_set)
train_set.to_csv("train_set.csv")

print("Create testing CSV with vectorized X and encoded Y")
x_test_buffer = pd.DataFrame(x_test)
y_test_buffer = pd.DataFrame(test_data['type'].values)

x_test_buffer["type"] = y_test_buffer
test_set = x_test_buffer
# test_set = pd.concat([x_test_buffer, y_test_buffer], axis=1, ignore_index=True)
# test_set = test_set.reset_index(drop=True)
# test_set = x_test_buffer.join(y_test_buffer)
print("test dataset length")
print(len(test_set))
print(test_set)
test_set.to_csv("test_set.csv")

print("Create validation CSV with vectorized X and encoded Y")
x_val_buffer = pd.DataFrame(x_val)
y_val_buffer = pd.DataFrame(val_data['type'].values)
x_val_buffer["type"] = y_val_buffer
val_set = x_val_buffer
# val_set = pd.concat([x_val_buffer, y_val_buffer], axis=1, ignore_index=True)
# val_set = val_set.reset_index(drop=True)
# val_set = x_val_buffer.join(y_val_buffer)
print("validation dataset length")
print(len(val_set))
print(val_set)
val_set.to_csv("val_set.csv")

# data.to_csv("processed_data.csv", index=False)