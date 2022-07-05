
from flask import Flask, jsonify
from flask import render_template
import nltk
import nltk.stem as stemmer
from nltk.stem import *
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
import numpy as np
import tflearn
import random
import json

stemmer = LancasterStemmer()
# creates a Flask application, named app
from nltk.corpus import words

app = Flask(__name__)


with open("intents.json") as json_data:
    intents = json.load(json_data)

# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
    return render_template('index.html')


context = {}  # Create a dictionary to hold user's context

ERROR_THRESHOLD = 0.25

# Empty lists for appending the data after processing NLP
import nltk

words = []
documents = []
classes = []

# This list will be used for ignoring all unwanted punctuation marks.
ignore = ["?"]



# Starting a loop through each intent in intents["patterns"]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # tokenizing each and every word in the sentence by using word tokenizer and storing in w
        w = nltk.word_tokenize(pattern)
        # print(w)

        # Adding tokenized words to words empty list that we created
        words.extend(w)
        # print(words)

        # Adding words to documents with tag given in intents file
        documents.append((w, intent["tag"]))
        # print(documents)

        # Adding only tag to our classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])  # If tag is not present in classes[] then it will append into it.
            # print(classes)

#Performing Stemming by using stemmer.stem() nd lower each word
#Running loop in words[] and ignoring punctuation marks present in ignore[]

stemmer = PorterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))  #Removing Duplicates in words[]

#Removing Duplicate Classes
classes = sorted(list(set(classes)))

#Printing length of lists we formed
print(len(documents),"Documents \n")
print(len(classes),"Classes \n")
print(len(words), "Stemmed Words ")

def clean_up_sentence(sentence):
    # Tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)  # Again tokenizing the sentence

    # Stemming each word from the user's input
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    return sentence_words


def bow(sentence, words, show_details=False):
    # Tokenizing the user input
    sentence_words = clean_up_sentence(sentence)

    # Generating bag of words from the sentence that user entered
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return (np.array(bag))

def classify(sentence):
    net = tflearn.input_data(shape=[None, 247])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 55, activation="softmax")
    net = tflearn.regression(net)

    # Defining Model and setting up tensorboard
    model = tflearn.DNN(net, tensorboard_dir="tflearn_logs")
    model.load("./model.tflearn")
    # Generating probabilities from the model
    print(bow(sentence, words))
    results = model.predict([bow(sentence, words)])[0]

    print(results)
    # Filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]


    # Sorting by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))

    # return tuple of intent and probability
    return return_list


@app.route("/response/<sentence>", methods=['GET'])
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # If we have a classification then find the matching intent tag
    data = "not found"
    if results:
        with open("intents.json") as json_data:
            intents = json.load(json_data)  # Loading our json_data
        # Loop as long as there are matches to process
        while results:
            for i in intents['intents']:

                # Find a tag matching the first result
                if i['tag'] == results[0][0]:

                    # Set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print('tag:', i['tag'])

                        # A random response from the intent
                        return random.choice(i['responses'])

            results.pop(0)

    return data




# run the application
if __name__ == "__main__":

    app.run()
