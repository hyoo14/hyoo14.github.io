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


[링크](https://drive.google.com/drive/folders/1ztQKESAp8HbBBCRgYTgs_7VbM21V-m6W?usp=sharing)


# 단어정리  
*consecutive: 연속적인(연이은), quantifies: 양화(증량화), auxiliary: 보조자, amenable: 받을 수 있는,  
disentangle: 풀다  


# 1 Introduction  
*프리트레인 self supervised러닝 성능 폭발적  
**천만 파라미터 가짐  
**헤비한 모델 & 높은 레이턴시  
**리소스 제한된 모바일 기기서 못 씀  
*해결책으로 distillBERT 나옴  
**compact한 버트 만드는 것  
**task-agnostic에서 쓰기 어렵  
*모바일 버트 실험  
**largeBERT 파인튠해서 teacher모델 만들고 distill하는 것  
*컴팩트 버트 쉬워보이지만 그렇지 않음  
**narrow, shallow한 버트 만들면 끝일 것 같지만  
**convex combination에 수렵하게 하고 prediction loss & convex combination 해주면  
***정작 정확도 많이 떨어짐  
***얕은 네트워크는 표현이 불충분함  
***좁은 네트워크는 학습이 힘듬  


*제안하는 모바일버트는  
**narrow but bottleneck구조로 self어텐션과 FFNN 벨런싱 해줌  
**깊고 얕은 모델 위해 티처 학습(특별 제작한)  
**지식전이 기법 사용  


*결과  
**4.3배 축소, 5.5배 속도 증가  
**NLP벤치마크 중 GLUE의 경우 약간 미흡하나 성능 비슷, SQuAD는 성능 오히려 향상시킴  