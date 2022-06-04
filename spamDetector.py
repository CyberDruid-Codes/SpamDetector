#!/usr/bin/env python3.7

##
#Import Section
import random
import email
import sys
import json
import pickle
import numpy as np 
import re
import nltk 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model 
##

##
#Initialises needed objects to process the input and opens the intents/model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = load_model('cyberbotmodel.h5')
##

####
#DNN processing and initialisation

#Cleans up the input and lemmatizes the sentence 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Creates a bag of words ( An array with a 1 if the word is found and a 0 if not ) in place of the words in the sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 
                
    return np.array(bag)

#Uses the model and takes the highest probable result from the different results of the neuralnet based on the probability
def predict_class(sentence):
    #create bag of words
    bow = bag_of_words(sentence)
    #predicting the results
    res = model.predict(np.array([bow]))[0]
    #threshold of the probability capped at 25%
    ERROR_THRESHOLD = 0.25
    #remove the uncenrtanty by cappping with the threshold 
    results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    #sort by probability in order ( highest probability first) 
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probablility': str(r[1])})
    return return_list

##
#Responses
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result 
##

##
#Opens the email file and reads its contents
content = sys.argv
"""
msg = email.message_from_file(open(content[1]))
attachments=msg.get_payload()
for attachment in attachments:
    fnam=attachment.get_filename()
    scantext=open(fnam, 'wb',encoding="cp437").write(attachment.get_payload(decode=True,))
    scantext.close()
"""
scantext = open(content[1], "r",encoding="cp437") 
content = scantext.read()
scantext.close()
#print(content)
##
"""
content = sys.argv
if content[-4:] == ".eml":
    pass
else:
    content = content + ".eml"
"""
##

##
#removes characters such as { } [ ] to be able to split and process the string 
replaced_content = content.replace('[', '')
replaced_content = replaced_content.replace(']','')
replaced_content = replaced_content.replace('{','')
replaced_content = replaced_content.replace('}','')
scan_content = re.split(' ',replaced_content)
##

##
#initialising variables needed
message = " "
curse = 0 
commonS = 0
commonP = 0
commonSy = 0
vulgarW = 0 
##


##Creating a while loop that iterates and feeds the deep learning algorithm the results in cut strings    
def dnnProcessing():
    for x in scan_content:
        global message
        #if 'cpe' in x: 
        message = x
        ints = predict_class(message)
        res = get_response(ints,intents)
        #print(res)
        interpret(res)
        message = " "
##

##
def interpret(dnnResults):
    global curse, commonS, commonP, commonSy, vulgarW
    if dnnResults == "Cursewords":
        curse+=1
    elif dnnResults == "CommonSpammy":
        commonS+=1
    elif dnnResults == "SpamPhrases":
        commonP+=1
    elif dnnResults == "SpammySymbols":
        commonSy+=1
    elif dnnResults == "VulgarWords":
        vulgarW+=1
    else:
        pass

##
def decision():
    global curse, commonS, commonP, commonSy, vulgarW
    dnnProcessing() 

    fres = curse + commonS + commonP + commonSy + vulgarW
    spammy = "This is Spam. We have found "+str(curse)+" cursewords and "+ str(commonS)+" Common Spam phrases." 
    notSpammy = "It is not a Spam" 

    if fres >= 5:
        print(spammy)
    else: 
        print(notSpammy)

decision()


