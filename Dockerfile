##################Dockerfile##################
FROM openjdk:8

RUN apt-get update
RUN apt-get install -y bzip2 
RUN apt-get install -y wget
RUN apt-get install -y gcc 
RUN apt-get install -y git 
RUN apt-get install -y curl

RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip

RUN pip3 install numpy==1.19.4
RUN pip3 install pyspark==3.0.1
RUN pip3 install h5py==2.10.0
RUN pip3 install gdown==3.12.2

ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/anger-ratings-0to1.train.txt
RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/fear-ratings-0to1.train.txt
RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/joy-ratings-0to1.train.txt
RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/sadness-ratings-0to1.train.txt

RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/anger-ratings-0to1.dev.gold.txt
RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/fear-ratings-0to1.dev.gold.txt
RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/joy-ratings-0to1.dev.gold.txt
RUN wget http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/sadness-ratings-0to1.dev.gold.txt

RUN pip3 install keras==2.2.5
RUN pip3 install tensorflow==1.13.1

RUN echo "5916165851071"

RUN git clone https://github.com/gaoyuanliang/jessica_deep_emotion_sensor.git
RUN mv jessica_deep_emotion_sensor/* ./

##################Dockerfile##################
