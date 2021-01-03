########jessica_emotion_tagger.py########
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.utils import *

from jessica_deep_emotion_sensor import texts_to_input

fear_tagger = keras.models.load_model('fear_tagger.h5')
sadness_tagger = keras.models.load_model('sadness_tagger.h5')
joy_tagger = keras.models.load_model('joy_tagger.h5')
anger_tagger = keras.models.load_model('anger_tagger.h5')

tagger_models = {"fear": fear_tagger,
"sadness": sadness_tagger,
"joy": joy_tagger,
"anger": anger_tagger}

def emotion_tagging(text):
	x = texts_to_input([text])
	output_tags = []
	for emotion in tagger_models:
		emotion_tagger = tagger_models[emotion]
		scores = emotion_tagger.predict(x)
		prediction = np.argmax(scores)
		confidence = np.max(scores)
		if prediction == 1:
			output_tags.append({'tag':emotion, "confidence":confidence})
	return output_tags

########jessica_emotion_tagger.py########
