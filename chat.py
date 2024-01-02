import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding,Input, Activation, Dense, Permute, Dropout,add, dot, concatenate,LSTM
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer


with open('data\train_qa.txt','rb') as f:
  train_data = pickle.load(f)

with open('data\test_qa.txt','rb') as f:
  test_data = pickle.load(f)

all_data  = train_data + test_data

' '.join(train_data[0][0])

' '.join(train_data[0][0])

' '.join(train_data[0][2])

df  = pd.DataFrame(all_data,columns=['Story','Question','Answer'])
print(df)

vocab = set()
for story,question,answer in all_data:
  vocab = vocab.union(set(story))
  vocab = vocab.union(set(question))

vocab.add('yes')
vocab.add('no')
vocab

vocab_len = len(vocab) + 1

# for i in range(len(all_data)):
#   for data in all_data[i]:
#     print(' '.join(data))

stories_len = [len(data[0]) for data in all_data]
question_len = [len(data[1]) for data in all_data]

maxlen_stories = max(stories_len)
maxlen_stories

maxlen_question = max(question_len)
maxlen_question

tokenizer = Tokenizer(filters = [])

tokenizer.fit_on_texts(vocab)

tokenizer.word_index['yes']

tokenizer.word_index['no']

tokenizer.word_index

train_story_text = []
train_question_text = []
train_answer_text = []
for s,q,a in train_data:
  train_story_text.append(s)
  train_question_text.append(q)
  train_answer_text.append(q)

# train_story_text

train_story_seq = tokenizer.texts_to_sequences(train_story_text)
# train_story_seq = np.array(train_story_seq)
# train_story_seq

def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=maxlen_stories,max_question_len=maxlen_question):
    '''
    OUTPUT:
    Vectorizes the stories,questions, and answers into padded sequences. We first loop for every story, query , and
    answer in the data. Then we convert the raw words to an word index value. Then we append each set to their appropriate
    output list. Then once we have converted the words to numbers, we pad the sequences so they are all of equal length.
    Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)
    '''
    # X = STORIES
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []


    for story, query, answer in data:

        #word index for every word in story
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)

input_sequence = Input((maxlen_stories,))
question_sequence = Input((maxlen_question,))
question_sequence

"""### input encoder M"""

# Input gets embedded to a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len,output_dim=64))
input_encoder_m.add(Dropout(0.5))

# This encoder will output:
# (samples, story_maxlen, embedding_dim)

"""### input encoder C"""

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len,output_dim=maxlen_question))
input_encoder_c.add(Dropout(0.5))
# output: (samples, story_maxlen, query_maxlen)

"""### question encoder

"""

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_len,
                               output_dim=64,
                               input_length=maxlen_question))
question_encoder.add(Dropout(0.5))
# output: (samples, query_maxlen, embedding_dim)

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question_sequence)

match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])
answer

# Reduce with RNN (LSTM)
answer = LSTM(32)(answer)  # (samples, 32)
# Regularization with Dropout
answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)  # (samples, vocab_size)
#probability distribution over the vocabulary
answer = Activation('softmax')(answer)

model = Model([input_sequence, question_sequence], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit([inputs_train, queries_train], answers_train,batch_size=32,epochs=100,validation_data=([inputs_test, queries_test], answers_test))

filename = 'chatbot.h5'
model.save(filename)

model.load_weights(filename)
pred_results = model.predict(([inputs_test, queries_test]))

story =' '.join(word for word in test_data[1][0])
print(story)

query = ' '.join(word for word in test_data[1][1])
print(query)

print("Answer is:",test_data[1][2])

#prediction from model
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])


my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()

my_question = "Is the football in the garden ?"
my_question.split()

mydata = [(my_story.split(),my_question.split(),'yes')]

my_story,my_ques,my_ans = vectorize_stories(mydata)

pred_results = model.predict(([ my_story, my_ques]))
pred_results

#Generate prediction from model
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])
