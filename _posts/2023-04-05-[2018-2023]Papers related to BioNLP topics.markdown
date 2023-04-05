---
layout: post
title:  "[2018-2023]Papers related to BioNLP topics"
date:   2023-04-05 04:13:23 +0900
categories: study
---



[Paper with my notes]()  

<br/>

# Protein BERT + Antibiotic Resistence Genes  
* 시행착오   
** 파인튜닝이면 만사 ok라고 생각  
*** 코드문제가 아니었던 것 같음  
*** 데이터가 너무 편향적이고 카테고리는 14개로 많은편이며, 데이터 개수가 적음  
*** 계속 같은 예측값만 뱉어내는 것을 보아 overfiting 됨  
** 데이터 균형 잡히게 하는 방안 고민  
*** 언더샘플링 / 오버샘플링 생각  
*** 언더샘플링은 효과 없었음  
** 그냥 프로틴버트에서 임베딩 뽑아내서 이 데이터로 ml기법들(dl포함) 사용해야하나 생각이 듦  
*** 예전에 했던 광고 카테고리 분류처럼 여러 테스트 해봐야하나? 일단 자야겠음 근데..
** 임베딩 벡터를 잘 뽑아와서 ML기법 적용해보기로 함  
*** 임베딩 평균으로 센텐스임베딩 처럼 사용하여 랜덤포레스트 모델에서 학습 -> acc0.5  
**** 근데 이 정확도마저 재현이 안됨.....  
*** 임베딩 컨캣해서 길게 이어붙임, 이것으로 랜덤포레스트 모델 학습 -> acc 0.3
*** 바로 위의 임베딩 중 길이 제한 300, 100으로 줌 이걸로 랜덤포레스트 -> acc 0.3  
*** 각종 앙상블 모델들 다 30~40 언저리... mlp도 그 언저리..  

<br/>

# [2021]Knowledge-based XAI through CBR  


{% highlight ruby %}
짧은 요약 :  

*KB x AI  
**데이터 중심 에이전트  
***diverse x data  
*도메인지식 중요  
**분류 성능 증대  

{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  

<br/>





   


