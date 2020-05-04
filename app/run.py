import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import sys
sys.path.append("../models")
from train_classifier import StartingVerbExtractor

app = Flask(__name__)

def tokenize(text):
    '''
    pipeline to tokenize text
    Input : text - String
              text to be tokenized
    Output : clean_tokens - list
               list of tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Trained model overview visualization. Two bar charts of overview of training dataset are visualized.
    One is aid related message counts classified by different genre, and the other is medical help message counts
    '''
    # extract data needed for visuals    
    genre_aid_related = df[df['aid_related']==1].groupby('genre').count()['message']
    genre_not_aid_related = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_aid_related_names = list(genre_aid_related.index)
    

    genre_medical_help = df[df['medical_help']==1].groupby('genre').count()['message']
    genre_not_medical_help = df[df['medical_help']==0].groupby('genre').count()['message']
    genre_medical_help_names = list(genre_medical_help.index)

    # create visuals
    graphs =  [
        {
            'data': [
                Bar(
                    x=genre_aid_related_names,
                    y=genre_aid_related,
                    name = 'Aid Related'
                ),

                Bar(
                    x=genre_aid_related_names,
                    y=genre_not_aid_related,
                    name = 'Not Aid Related'
                )
            ],

            'layout': {
                'title': 'Distribution of Aid Related Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=genre_medical_help_names,
                    y=genre_medical_help,
                    name = 'Medical Help'
                ),

                Bar(
                    x=genre_medical_help_names,
                    y=genre_not_medical_help,
                    name = 'Not Medical Help'
                )
            ],

            'layout': {
                'title': 'Distribution of Medical Help Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
    function used for handling user query and displaying model results
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    '''
    Main function of Flask Web App
    '''
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()