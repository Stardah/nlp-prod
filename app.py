import sys

from flask import Flask, request, flash
from flask import render_template

from nlp import NLP_model

app = Flask(__name__)
app.secret_key = 'secret key'

model = NLP_model()

@app.route('/')
def hello_world():
    return render_template('base.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    sentence = request.form['sentence']
    prediction = model.predict([str(sentence)])
    if prediction:
        answer = 'Позитивное предложение :)'
        flash(answer, 'flash')
    else:
        answer = 'Нагативное предложение :('
        flash(answer, 'error')
        #.flash { margin: 1em 0; padding: 1em; background: #cae6f6; border: 1px solid #377ba8; }


    return render_template('base.html', words = sentence)

if __name__ == '__main__':
    #sess.init_app(app)
    app.run()
