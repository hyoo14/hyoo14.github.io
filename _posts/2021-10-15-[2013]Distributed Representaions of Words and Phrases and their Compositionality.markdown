---
layout: post
title:  "[2013]Distributed Representaions of Words and Phrases and their Compositionality"
date:   2021-10-15 16:00:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

스킵그램이 이전 모델들 보다 syntatic & semantic 단어 연관성 잘 잡아줌

Hierarchical Softmax 대체하는 Negative Sampling 으로 학습속도와 단어임베딩의 퀄리티를 높였음

관용구 통쨔ㅐ로 학습시켜서 기존 관용구 약점 좀 보완함

{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1DZn7lY8wZPNmCAf2GcGmBFGy_ls7o7NK?usp=sharing)


-그냥 소프트맥스는 계산이 너무 많음 (W) 

-그래서 하이어라키컬 소프트 맥스 사용 (logW)

-하이어라키컬 소프트 맥스를 NCE 생각에 기반한 Negative Sampling(NEG)로 대체

여기서 NCE 생각은 "노이즈와 data를 잘 분리하려는 것" 으로 확률 분포를 알아야함
이 때 NCE, NEG ahen noise 분포로 unigram distribution 3/4power를 사용하면 가장 좋은 성능을 보였음(실험적으로)

Negative Sampling은 주변 단어들이 아닌 단어의 잡합을 만들어 negative 레이블링 하고 주변 단어들의 집합을 만들어 positive 레이블링하여 학습시키는 것으로
단어임베딩 학습을 (연산적으로 좀 더 효율적이게) 이진 분류 문제 학습으로 변환한 것
결과적으로 성능적으로도 더 좋았음

