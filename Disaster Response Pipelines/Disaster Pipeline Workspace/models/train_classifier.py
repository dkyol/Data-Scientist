import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine 
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from nltk import pos_tag 

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
import pickle 


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    #conn = sqlite3.connect(database_filepath)
    #cur = conn.cursor()
    #df = pd.read_sql('table', conn)
    df = pd.read_sql_table('book', engine)
    
    #df = cur.execute("SELECT * FROM table")
    #print(df.head())
    
    X = df.message
    
    y = df.drop(['id', 'message', 'original', 'genre', 'categories'], axis = 1)
    
    cat_names = y.columns 
    
    return X, y, cat_names


def tokenize(text):
    stop = stopwords.words('english')
    # remove punctutation, covert to lowercase, strip spaces, lemmatize, remove common words 
    # reduce case
    words = text.lower()
    # remove puntuatuion 
    words = re.sub('[^a-z0-9]', ' ', words)
    #split words into list 
    word_list = word_tokenize(words)
   
    #lemmatize
    Lemma = WordNetLemmatizer()
    
    token_list = []
    
    for x in word_list:
        
        if x not in stop:
          
            token = Lemma.lemmatize(x, 'v').strip()

            token_list.append(token)
  
    return token_list

def build_model():
    
    pipeline = Pipeline([
    
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    
        ('rfc', MultiOutputClassifier(RandomForestClassifier())), 
   
    ]) 
    
        
    parameters = {
    #'tfidf__ngram_range': ((1,1),(1,2),(2,1),(2,2)),
    'tfidf__max_df': (1.0, 3.0, 10.0),
    'tfidf__max_features': (None,1000, 2500)      
             
    }

    clf = GridSearchCV(pipeline, param_grid = parameters, n_jobs = 4)
    
    return clf


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    print("Accuracy", (y_pred == Y_test).mean())


def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath,'wb')) 
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()