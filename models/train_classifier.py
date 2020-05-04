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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.values

    return X, Y, category_names

def tokenize(text):
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
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('starting_verb', StartingVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier( LinearSVC() ))
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
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
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