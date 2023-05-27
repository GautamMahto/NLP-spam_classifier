import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import seaborn as sns
from nlp import NLP

#Create space between two context
def space():
    st.markdown("<br>",unsafe_allow_html=True)


# Heading 
st.markdown("<h1 style='text-align':center;color:#3f3f44>NLP Automation-NAive bayes",unsafe_allow_html=True)
space()
#Sub-Heading
st.markdown("<strong><p style='color:#434874'>1) This project uses Navie Bayes Algorithm</p></strong>",unsafe_allow_html=True)
st.markdown("<strong><p style='color:#434874'>2) you can choose different cleaning process</p></strong>",unsafe_allow_html=True)
st.markdown("<strong><p style='color:#434874'>3) Different type of metrics formation(Count Vectorizing,TF-IDF)</p></strong>",unsafe_allow_html=True)
st.markdown("<strong><p style='color:#43fe74'>4) Plotting SentimentalAnalysis</p></strong>",unsafe_allow_html=True) 


# Data Preprocessing
def preprocess():
    try:
        preprocessing_option=['Stemming','lemmitizing']
        preprocesser=st.selectbox('Select Preprocessing techinque',preprocessing_option)
        return preprocesser
    except Exception as e:
        print("error occured in preprocessed data",e)

# Hyperparameter tuning
def hyperparameter():
    try:
        features=['2500','3000','3500','4000']
        max_features=st.selectbox('Maximum Features you want to restrict',features)
        space()
        space()
        ranges=["1,1","1,2","1,3"]
        ngram_range=st.selectbox('Combination of Words',ranges).split(',')
        return max_features,ngram_range
    except Exception as e:
        print("hYPERPARAMETER TUNING eRROR",e)

# Count Vectorizer(bag of Words) / TF-IDF
def bow():
    try:
        metrics=['Count Vectorizer','TFIDF']#*************************************************************************************************************************************************
        bag_words=st.selectbox('Bag of words techinique',metrics)
        return bag_words
    except Exception as e:
        print('Bow Error:-',e)

# Converting Target column
def y_label():
    try:
        target_columns=["Yes","No"]
        y_option=st.selectbox('Do you want to encode your target column',target_columns)
        return y_option
    except Exception as e:
        print("y_label Error has been occured",e)

# main Function
def app():
    try:
        df=st.file_uploader('Upload your Dataset',type=["csv","txt"])#
        space()

        if df is not None:
            data=pd.read_csv(df,encoding="ISO-8859-1")#encoding="ISO-8859-1"
            st.dataframe(data.head())
            space()

            text=st.selectbox("Select text Column",data.columns)
            space()

            target=st.selectbox("Select Tatget Column",data.columns)

            # Reassigning Features to Dataframe
            data=data[[text,target]]

            # Drop Nan Values
            data=data.dropna()

            # 
            nlp_model=NLP(data)

            st.markdown("<h4 style='color:#438a5e'>Final Dataset</h4>",unsafe_allow_html=True)

            st.dataframe(data.head())
            space()

            # callif Functions for Preprocessing , Bag of Words, Target Variables
            preprocessor=preprocess()
            space()
            space()
            max_features,ngram_range=hyperparameter()
            space()
            space()
            bag_words=bow()
            space()
            space()
            y_option=y_label()
            space()
            space()

            # Define Function
            def matrix(corpus,bag_words,max_features,ngram_range):
                try:
                    if bag_words=='Count Vectorizer':
                        X=nlp_model.count_vectorizer(corpus,int(max_features),(int(ngram_range[0]),int(ngram_range[1])))
                        return X
                    elif bag_words=='TFIDF':
                        X=nlp_model.tfidf(corpus,int(max_features),(int(ngram_range[0]),int(ngram_range[1])))
                        return X
                except Exception as e:
                    print("Exception Occured in defining the matrix",e)
            
            def target_series(y_option,target):
                try:
                    if y_option=='Yes':
                        y=nlp_model.y_encoding(target)
                        return y
                    elif y_option=="No":
                        y=data[target]
                        return y
                    
                except Exception as e:
                    print("Error Occured at the target column",e)
            
            def plot_wordcloud(corpus,y_test,y_pred):
                st.success('Word Cloud')
                wordcloud=nlp_model.word_cloud(corpus)
                st.image(wordcloud)
                accuracy,cm=nlp_model.cm_accuracy(y_test,y_pred)
                st.success(f"Accuracy: {round(accuracy*100,2)}%")
                st.image(cm)
            
            # Sentiment
            def sentimental(text):
                data['sentiments']=data[text].apply(nlp_model.sentimental_analysis_clean)

                def getSubjectivity(text):
                    return TextBlob(text).sentiment.subjectivity

                def getPolarity(text):
                    return TextBlob(text).sentiment.polarity
                
                def getAnalysis(score):
                    if score<0:
                        return "Negative"
                    elif score==0:
                        return "Neutral"
                    else:
                        return "Positive"
                
                data['Subjectivity']=data['sentiments'].apply(getSubjectivity)
                data['Popularity']=data['sentiments'].apply(getPolarity)
                data['Analysis']=data['Popularity'].apply(getAnalysis)

                st.success("Sentiments")

                sns.countplot(x=data['Analysis'],data=data)
                st.pyplot(use_container_width=True)
            

            # Model Creation
            if st.button("Submit"):
                space()
                if preprocessor=='Stemming':
                    corpus=nlp_model.stemming(text)
                    X=matrix(corpus,bag_words,max_features,ngram_range)
                    y=target_series(y_option,target)
                    X_train,X_test,y_train,y_test=nlp_model.split_data(X,y)
                    y_pred=nlp_model.navie_model(X_train,X_test,y_train,y_test)
                    sentimental(text)
                    plot_wordcloud(corpus,y_test,y_pred)
                elif preprocess=='lemmitizing':
                    corpus=nlp_model.lemmitizing(text)
                    X=matrix(corpus,bag_words,ngram_range,max_features)
                    y=target_series(y_option,target)
                    X_train,X_test,y_train,y_test=nlp_model.split_data(X,y)
                    y_pred=nlp_model.navie_model(X_train,X_test,y_train,y_test)
                    sentimental(text)
                    plot_wordcloud(corpus,y_test,y_pred)

                


            

              
    except Exception as e:
        print('Main Function Error has been Occured',e)

if __name__=='__main__':
    app()
          