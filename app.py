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

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)
# print(tokenizer.word_index)
maxlen_stories = 156
maxlen_question = 6

def vectorize_stories(data, max_story_len=maxlen_stories, max_question_len=maxlen_question):
    X = []
    Xq = []
    
    for story, query in data:
        # Tokenize and  stories to sequences
        x = tokenizer.texts_to_sequences([story])[0]
        xq = tokenizer.texts_to_sequences([query])[0]

        # Pad sequences
        x = pad_sequences([x], maxlen=max_story_len)[0]
        xq = pad_sequences([xq], maxlen=max_question_len)[0]

        X.append(x)
        Xq.append(xq)

    return np.array(X), np.array(Xq)



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

        processed_data = [(user_story, user_question)]
        s, q = vectorize_stories(processed_data)

        prediction = model.predict([s, q])
        print(s,q)

        val_max = np.argmax(prediction[0])
        for key, val in tokenizer.word_index.items():
            if val == val_max:
                k = key
        probability = prediction[0][val_max]

        return render_template('result.html', prediction=k, probability=probability)

  

if __name__ == '__main__':
    app.run(debug=False)
