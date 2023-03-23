 ---
layout: post
title:  "[2014]Sequence to Sequence Learning with Neural Networks"
date:   2023-03-23 16:49:33 +0900
categories: study
---


{% highlight ruby %}
짧은 요약(Abstract) :    
* DNN powerful but seq2seq 'X'  
** multi layer LSTM으로 seq2seq 'O'  
** input->deepLSTM enocder->vector->deepLSTM decoder->target 
** eng2french -data: WMT14 dataset   
** result: BLEU 34.8 for test  
** OOV 때문에 BLEU 저하  
** long sent 잘 다룸  
** 대조군 : phrase 기반 SMT BLEU33.3  
** LSTM rerank 1000hypothesis after SMT: BLEU 36.5 - SOTA근접  
** word order에 민감, 상대적으로 active%passive voice에 invarient  
** 역순 입력으로 LSTM 성능 많이 올림(short term dependency 줌-> 소스와 타겟 사이->optimal easier)  


   
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1aCKj7Q_oCMt4V51PrkqUbX2XBBblW5UC?usp=sharing)  


[~~Lecture link~~]()  


# 단어정리  
* invarient: 변함없는, 변치 않는  
* quadratic: 이차의(이차방정식할 때)  
* pose a challenge: 도전하다  
* time lag: 시간상의 차이  
* non-monotonic: 비단조  
* negligible: 무시해도 될 정도의, 무시해도 좋은   

   

# 1 Introduction  
* .  







