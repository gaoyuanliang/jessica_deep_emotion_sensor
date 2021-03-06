########jessica_emotion_tagger.py########

import os
import keras
import numpy
import random
import tensorflow

from jessica_deep_emotion_sensor import *
from jessica_emotion_tagger_train import *

tagger_models = {
"fear": keras.models.load_model('fear_tagger.h5'),
"anger": keras.models.load_model('anger_tagger.h5'),
"sadness": keras.models.load_model('sadness_tagger.h5'),
"joy": keras.models.load_model('joy_tagger.h5'),
}

def emotion_tagging(text):
	x = texts_to_input([text])
	output_tags = []
	for emotion in tagger_models:
		emotion_tagger = tagger_models[emotion]
		scores = emotion_tagger.predict(x)
		prediction = numpy.argmax(scores)
		confidence = numpy.max(scores)
		if prediction == 1:
			output_tags.append({'tag':emotion, "confidence":confidence})
	return output_tags

'''
from jessica_emotion_tagger import emotion_tagging

emotion_tagging(u"All I want to do is watch some netflix but I am stuck here in class. #depressing")
#[{'tag': 'sadness', 'confidence': 0.99995244}]

emotion_tagging(u"Watching It Follows.  This is a super freaky movie.  #scary")
#[{'tag': 'fear', 'confidence': 0.9999993}]

emotion_tagging(u"Ready for that nice, breezy, calm, sunshine weather.🍂🍁 #Autumn")
#[{'tag': 'joy', 'confidence': 0.9989052}]

emotion_tagging(u"@leesyatt you are a cruel, cruel man. #therewillbeblood #revenge")
#[{'tag': 'anger', 'confidence': 1.0}]

'''
########jessica_emotion_tagger.py########