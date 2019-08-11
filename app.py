import json
import sys

from flask import Flask, request, flash
from flask import render_template

from ml.nlp import NLP_model

app = Flask(__name__)
app.secret_key = 'secret key'

model = NLP_model()

@app.route('/')
def hello_world():
    return render_template('base.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    #print(request.data, file=sys.stdout)
    sentence = request.form['content']
    prediction = model.predict([str(sentence)])
    if prediction == 1:
        answer = 'Позитивное высказывание :)'
        flash(answer, 'flash')
    elif prediction == 2:
        answer = 'Нагативное высказывание :('
        flash(answer, 'error')
    else:
        answer = 'Нейтральное высказывание ¯\_(ツ)_/¯'
        flash(answer, 'norm')
    return render_template('base.html', words = sentence)

@app.route('/get_message', methods=['GET', 'POST'])
def get_message():
    sentence = request.form['content']
    prediction = model.predict([str(sentence)])
    return json.dumps({'prediction':int(prediction[0])})

@app.route('/tolmachev_best')
def jeez():
    print(request.values, file=sys.stdout)

    number = request.form['number']
    result = 1
    for digit in str(number):
        result *= int(digit)

    return json.dumps({'tolmachev_best_result':result})

if __name__ == '__main__':
    app.run()
