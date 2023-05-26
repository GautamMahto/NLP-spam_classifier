import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
import scikitplot as skplt
import matplotlib.pyplot  as plt
from PIL import Image
from wordcloud import WordCloud
 
# text data-NLP problem
# Stemming and  Lamitization
# Text Vectorization /TFIDF
# train test
# model building
# check accuracy


class NLP:
    def __init__(self,data) -> None:
        self.data=data

    # Stemming Part
    def stemming(self,column_name):
        try:
            corpus=[]
            stemming=PorterStemmer()

            for i in range(len(self.data)):
                tweet=re.sub('[^a-zA-Z]'," ",self.data[column_name][i])
                tweet=re.sub('http',"",tweet)
                tweet=tweet.lower()
                tweet=tweet.split()
                tweet=[stemming.stem(word) for word in tweet if word not in set(stopwords.words('english'))]
                tweet=" ".join(tweet)
                corpus.append(tweet)

        except Exception as e:
            print("Stemming Error",e)
        
        else:
            return corpus
    
    # Lamitization Part
    def lemmitizing(self,column_name):
        "Stopwords are being used"
        try:
            corpus=[]
            stemming=WordNetLemmatizer()

            for i in range(len(self.data)):
                tweet=re.sub('[^a-zA-Z]'," ",self.data[column_name][i])
                tweet=re.sub('http',"",tweet)
                tweet=tweet.lower()
                tweet=tweet.split()
                tweet=[stemming.lemmatize(word) for word in tweet if word not in set(stopwords.words('english'))]
                tweet=" ".join(tweet)
                corpus.append(tweet)

        except Exception as e:
            print("Stemming Error",e)

        else:
            print("Cleaning was Completed")
            return corpus
    
    # Count Vectorizer/TFIDF
    def count_vectorizer(self,corpus,max_features=3000,ngram_range=(1,2)):
        # bag of words
        try:
            cv=CountVectorizer(max_features=max_features,ngram_range=ngram_range)
            X=cv.fit_transform(corpus).toarray()
        except Exception as e:
            print("Exception occured at count_vectorization and exception is : ",e)
        else:
            return X
    
    
    def tfidf(self,corpus,max_features=3000,ngram_range=(1,2)):
        # bag of words
        try:
            tfidf=TfidfVectorizer(max_features=max_features,ngram_range=ngram_range)
            X=tfidf.fit_transform(corpus).toarray()
        except Exception as e:
            print("Exception occured at TFIDF and exception is : ",e)
        else:
            return X

    # Encoding Target Column
    def y_encoding(self,target_label):
        try:
            y=pd.get_dummies(self.data[target_label],drop_first=True)
        except Exception as e:
            print("y_endoning target variable has not been done proper",e)
        else:
            return y
    
    def split(self,X,y,test_size=0.2,random_state=12):
        try:
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
        except Exception as e:
            print("Train_test_split has not been done properly so, excepton has been occured:",e)
        else:
            print("Successfully printing our data")
            return X_train,X_test,y_train,y_test
    
    def navie_model(self,X_train,X_test,y_train,y_test):
        try:
            navie=MultinomialNB()
            navie.fit(X_train,y_train)
            y_pred=navie.predict(X_test)
        except Exception as e:
            print("Navie's Bayes algorithm model error",e)
        else:
            return y_pred
    
    def cm_accuracy(self,y_test,y_pred):
        try:
            skplt.metrics.plot_confusion_matrix(y_test,y_pred,figsize=(8,7))
            plt.savefig('confusion_matrix.jpg')
            img_cm=Image.open('confusion_matrix.jpg')
            accuracy=accuracy_score(y_test,y_pred)
        except Exception as e:
            print("Confusion matrix error has been occured",e)
        else:
            return accuracy,img_cm
        
    def word_cloud(self,corpus):
        try:
            wordcloud=WordCloud(background_color='white',width=750,height=500).generate(" ".join(corpus))
            plt.imshow(wordcloud,interpolation="bilinear")
            plt.axis("off")
            plt.savefig('wordcloud.jpg')
            img_wc=Image.open('wordcloud.jpg')
        except Exception as e:
            print("Eorld Cloud Error",e)
        else:
            return img_wc
    
    # Sentimental Analysis
    def sentimental_analysis_clean(self,text):
        try:
            text=re.sub('http',"",text)
            text=re.sub('co',"",text)
            text=re.sub('amp',"",text)
            text=re.sub('new',"",text)
            text=re.sub('one',"",text)
            text=re.sub('@[a-zA-Z0-9]+','',text)
            text=re.sub('#','',text)
            text=re.sub('RT[\s]+','',text)
            text=re.sub('https?:\/\/\S+','',text)

            return text
        except Exception as e:
            print("sentimental analysis has been occured:",e)
