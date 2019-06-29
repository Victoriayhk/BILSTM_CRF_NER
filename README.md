Chinese Named Entity Recognition Modeling
=========================================

Intro
-----

this is code for training a Chinese NER model with Tensorflow(version<1.0), 
written in 2017, basic LSTM-CRF idea was from [tagger](https://github.com/glample/tagger).

A whole modeling needs: traing data, validation data, test data, Corpus to modeling 
word2vec, modeling idea, coding; this is only very small part of whole work.

At that time, I had no access to the best corpus to train my word-embedding vectors, 
I used free Wikipedia Corpus as replacement. Meaningwhile, A extra Chinese charactor 
level word-embedding was used too. And, I used [jieba](https://github.com/glample/tagger) 
to do word setmentation.

[this paper](https://www.itm-conferences.org/articles/itmconf/pdf/2017/04/itmconf_ita2017_04002.pdf) 
explains details of my work.

How to Run
----------

go check the code(train.py) -->