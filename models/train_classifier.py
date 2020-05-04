import sys
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.utils import parallel_backend
import time
import pickle


nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

def load_data(database_filepath='DisasterResponse.db'):
    '''
    Load data from a database
    Input : database_filepath - String
              file path to database that generated from ETL pipeline
    Output : X - Numpy Array
    		   feature variables for ML pipeline 
             Y - Numpy Array
               target variable for ML pipeline
             category_names - Numpy Array
               column names of target variable
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.values

    return X, Y, category_names

def tokenize(text):
    '''
    Normalize, tokenize, remove stop words and stem input text
    Input : text - String
              text to be tokenized
    Output : stemmed - list
    		   list of tokenized text
    '''
    # Normalization : Lower case and punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Word tokenization
    words = word_tokenize(text)
    
    # Remove Stop Words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in words]

    return stemmed    


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
	Customized function that tests if a sentence started with a verb
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if (pos_tags):   
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0

    def fit(self, X, y=None):
        return self
    def test(self, text):
        print(test)
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
	Grid Search a set of parameters for a designed pipeline
	Output : cv - sklearn GridSearchCV object
	           scikit learn GridSearchCV object used for latter model fitting
    '''
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('starting_verb', StartingVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier( SVC() ))
        ])
    parameters = {'features__text_pipeline__vect__max_df' : [1.0],
                  'features__text_pipeline__tfidf__use_idf' : [True],
                  'clf__estimator__C' : [0.5, 1.0], 
                  'clf__estimator__kernel' : ['linear', 'rbf'],
                  'features__transformer_weights': (
                      {'text_pipeline': 1, 'starting_verb': 0.5},
                      {'text_pipeline': 0.5, 'starting_verb': 1},                      
                  )
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs = -1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):    
    '''
	Evaluate model performance through test set
	Input : model - sklearn GridSearchCV object
			  trained model used for performance evaluation
	        X_test - Numpy array
	          splitted test feature variables for ML pipeline 
	        Y_test - Numpy array
	          splitted test target variables for ML pipeline 
	Output : metric_df - Pandas DataFrame
			   model perforance DataFrame including accuracy, precision, recall and f1 scores
    '''
    metric_names = ['ACCURACY', 'PRECISION', 'RECALL', 'F1']
    metric_df = pd.DataFrame(columns = metric_names)
    Y_pred = model.predict(X_test)
    
    all_accuracy, all_precision, all_recall, all_f1 = [], [], [], []
    for i, col in enumerate(category_names):
        y_true = Y_test[i]
        y_pred = Y_pred[i]
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        metric_df = metric_df.append(pd.Series([acc, prec, recall, f1], index = metric_names, name = col)) 
    return metric_df


def save_model(model, model_filepath):
    '''
	Save model object to a model file
	Input : model - sklearn GridSearchCV object
			  trained model to be saved
	        model_filepath - String
	          file path to be saved for a trained model
    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    '''
    Main function of ML pipeline
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()
            
            print('Training model...')
            model.fit(X_train, Y_train)
            print('Best model parameters : \n', model.best_params_)
            
            print('Evaluating model...')
            metrics = evaluate_model(model, X_test, Y_test, category_names)
            print('Best model evaluation : \n', metrics.describe())

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