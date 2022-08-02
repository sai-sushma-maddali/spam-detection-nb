import pandas as pd
import numpy as np

#natural language tool kit
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB

import string
import pickle

msg_data = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

############## Data Pre Processing & Feature Engineering ######################
###############################################################################


# Encoding the target column
encoder =LabelEncoder()
msg_data['label']=encoder.fit_transform(msg_data['label'])
# Dropping duplicates
msg_data = msg_data.drop_duplicates()
# Num of characters
msg_data['num_characters']=msg_data['message'].apply(len)
# Num of words
msg_data['num_words']=msg_data['message'].apply(lambda x:len(nltk.word_tokenize(x)))
# Num of sentences
msg_data['num_sentences']=msg_data['message'].apply(lambda x: len(nltk.sent_tokenize(x)))

#### Processing the text message
################################

def text_transform(message):
    message = message.lower()  # change to lowercase
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

        y.clear()

    # for checking punctuations and stopwords
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    message = y[:]
    y.clear()

    # now stemming function

    ps = PorterStemmer()

    for i in message:
        y.append(ps.stem(i))

    # return y  --> returns as list
    return " ".join(y)

    # Removing stop words and punctuations

    stopwords.words('english')
    len(stopwords.words('english'))

    # now for punctuation



# On the message column
msg_data['transformed_msg']=msg_data['message'].apply(text_transform)


# Vectorizing

tfidf= TfidfVectorizer(max_features=3000)
X=tfidf.fit_transform(msg_data['transformed_msg']).toarray()
y=msg_data['label'].values

# Train Test Split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Building Model
bnb = BernoulliNB()

bnb.fit(X_train,y_train)

# Saving model,text transform and tfidf vectorizer to pickle

pickle.dump(bnb, open('model.pkl','wb'))
pickle.dump(text_transform,open('text_transform.pkl','wb'))
pickle.dump(tfidf,open('tfidf.pkl','wb'))



