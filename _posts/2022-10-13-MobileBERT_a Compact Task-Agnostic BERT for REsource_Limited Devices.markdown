---
layout: post
title:  "MobileBERT_a Compact Task-Agnostic BERT for REsource_Limited Devices"
date:   2022-10-13 23:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

pre-train model이 nlp에서 매우 성공  
그러나 헤비한 모델 사이즈와 높은 latency는 아쉬운 점  
->해결책으로 모바일 버트 제안  
-버트라지 thin버전  
-보틀넥 구조 사용  
-지식 전이 기법 이용한 티처 구조 만들어서 사용  
-버트베이스보다 4.3배 작아지고 5.5배 빨라짐  
-GLUE 77.7점으로 버트베이스보다 0.6만 낮음  
-SQuAD F1 90/79.2로 버트보다 1.5/1.2 높음  

{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1zC8qtkySWQq_PVwUR7q-0bxgIVkXAMyn?usp=sharing)


# 단어정리  
*consecutive: 연속적인(연이은), quantifies: 양화(증량화), auxiliary: 보조자, amenable: 받을 수 있는,  
disentangle: 풀다  


# 1 Introduction  
*프리트레인 self supervised러닝 성능 폭발적  