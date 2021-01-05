# jessica_deep_emotion_sensor

```bash
docker pull gaoyuanliang/jessica_deep_emotion_sensor:1.0.1

docker run -it \
-v /Users/liangyu/Downloads/:/Downloads/ \
gaoyuanliang/jessica_deep_emotion_sensor:1.0.1
```

```python
>>> from jessica_emotion_tagger import emotion_tagging
>>> 
>>> emotion_tagging(u"Watching It Follows.  This is a super freaky movie.  #scary")
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
