from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle


with open('data/train_qa.txt','rb') as f:
  train_data = pickle.load(f)

with open('data/test_qa.txt','rb') as f:
  test_data = pickle.load(f)

app = Flask(__name__)
model = load_model('chatbot.h5')


all_data  = train_data + test_data

vocab = set()
for story,question,answer in all_data:
  vocab = vocab.union(set(story))
  vocab = vocab.union(set(question))
vocab.add('yes')
vocab.add('no')

tokenizer = Tokenizer(filters='!"#$%&()*+-/:;<=>@[\\]^_`{|}~')
tokenizer.fit_on_texts(vocab)
maxlen_stories = 156
maxlen_question = 6

def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=maxlen_stories,max_question_len=maxlen_question):
  
    # X = STORIES
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []

    for story, query, answer in data:

        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        X.append(x)
        Xq.append(xq)
        if answer.lower() == 'yes':
            Y.append([1, 0]) 
        else:
            Y.append([0, 1])  

    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_story = request.form['story']
        user_story = user_story.split()
        user_question = request.form['question']
        user_question = user_question.split()
  
        processed_data = [(user_story,user_question,'yes')]
        s,q,_ = vectorize_stories(processed_data)

        prediction = model.predict(([ s, q]))
        
        val_max = np.argmax(prediction[0])
        for key, val in tokenizer.word_index.items():
            if val == val_max:
                predicted_answer = key
                probability = prediction[0][val_max]

        return render_template('result.html', prediction=predicted_answer,probability=probability)  

if __name__ == '__main__':
    app.run(debug=False)
