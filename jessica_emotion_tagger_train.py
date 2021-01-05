##############################
from jessica_deep_emotion_sensor import *
from jessica_emoint_data_conversion import *

print('training the emotion taggers - fear')

fear_texts, fear_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "fear",
	data_file = "*.txt")
fear_tagger = train_tagger(
	texts = fear_texts,
	tags = fear_tags,
	tagger_model_path = 'fear_tagger.h5',
	epochs = 4,
	validation_split = 0.1,
	)

'''
(3960, 100) (3960, 2)
[2703. 1257.]
Train on 3564 samples, validate on 396 samples

Epoch 20/20
3564/3564 [==============================] - 9s 2ms/sample - loss: 0.0093 - acc: 0.9938 - val_loss: 1.2104 - val_acc: 0.8182
'''

print('training the emotion taggers - anger')

anger_texts, anger_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "anger",
	data_file = "*.txt")
anger_tagger = train_tagger(
	texts = anger_texts,
	tags = anger_tags,
	tagger_model_path = 'anger_tagger.h5',
	epochs = 4,
	validation_split = 0.1,
	)
'''
(3960, 100) (3960, 2)
[3019.  941.]
Train on 3564 samples, validate on 396 samples

Epoch 20/20
3564/3564 [==============================] - 29s 8ms/sample - loss: 0.0111 - acc: 0.9905 - val_loss: 0.8090 - val_acc: 0.8662
'''

print('training the emotion taggers - sadness')

sadness_texts, sadness_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "sadness",
	data_file = "*.txt")
sadness_tagger = train_tagger(texts = sadness_texts,
	tags = sadness_tags,
	tagger_model_path = 'sadness_tagger.h5',
	epochs = 4,
	validation_split = 0.1,
	)
'''
(3960, 100) (3960, 2)
[3100.  860.]
Train on 3564 samples, validate on 396 samples

Epoch 20/20
3564/3564 [==============================] - 21s 6ms/sample - loss: 0.0174 - acc: 0.9874 - val_loss: 0.6581 - val_acc: 0.9015
'''

print('training the emotion taggers - joy')

joy_texts, joy_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "joy",
	data_file = "*.txt")
joy_tagger = train_tagger(texts = joy_texts,
	tags = joy_tags,
	tagger_model_path = 'joy_tagger.h5',
	epochs = 4,
	validation_split = 0.1,
	)
'''
(3960, 100) (3960, 2)
[3058.  902.]
Train on 3564 samples, validate on 396 samples

Epoch 20/20
3564/3564 [==============================] - 17s 5ms/sample - loss: 9.8552e-04 - acc: 0.9992 - val_loss: 1.5256 - val_acc: 0.8384
'''

'''
scp liangyu@192.168.1.108:/Users/liangyu*.h5 ./
'''

##############################