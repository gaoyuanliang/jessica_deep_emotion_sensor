# jessica_deep_emotion_sensor

```python
>>> from jessica_emotion_tagger import emotion_tagging
>>> 
>>> emotion_tagging("Watching It Follows.  This is a super freaky movie.  #scary")
[{'tag': 'fear', 'confidence': 0.9999993}]
>>> 
>>> emotion_tagging(u"Ready for that nice, breezy, calm, sunshine weather. #Autumn")
[{'tag': 'joy', 'confidence': 0.9989052}]
>>> 
>>> emotion_tagging(u"@leesyatt you are a cruel, cruel man. #therewillbeblood #revenge")
[{'tag': 'anger', 'confidence': 1.0}]
>>> 
>>> emotion_tagging(u"All I want to do is watch some netflix but I am stuck here in class. #depressing")
[{'tag': 'sadness', 'confidence': 0.99995244}]
```
