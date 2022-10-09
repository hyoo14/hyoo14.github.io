---
layout: post
title:  "RoBERTa_ A Robustly Optimized BERT Pretraining Approach"
date:   2022-10-09 23:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

PLM 성능이 높아짐 특히 BERT  

이 BERT의 replication study 통해 parameter 조정을 함  

그래서 GLUE, RACE, SQuAD에서 SOTA 달성  


{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1zC8qtkySWQq_PVwUR7q-0bxgIVkXAMyn?usp=sharing)


# 단어정리  
*consecutive: 연속적인(연이은), quantifies: 양화(증량화), auxiliary: 보조자, amenable: 받을 수 있는,  
disentangle: 풀다  


# 1 Introduction  
*Self-train model의 어떤 hyperparameter가 성능 영향 끼치는지 모름  
**계산 비싸고 튜닝 제한, 사적 data 확장 제한, 성능 측정 어려움 있기 때문  
*BERT 복제 실험으로 RoBERTa 제안  
**버트 대비 성능 향상됨, 차이점은 아래와 같음    
**1. 모델 길게, 배치 크게, 데이터 양 더 많이  
**2. NSP 제외  
**3. 긴 시퀀스로 학습  
**4. 학습데이터 마스킹 때 다이나믹하게 해줌  
**또한 CC-NEWS 큰코퍼스 수집  
*학습데이터 각기 다른 엔드테스크에서도 본 논문 제안이 SOTA 찍음  
**glue 88.5(이전 88.4)  
**MNLI, QNLI, RTE, STS-B서 SOTA(GLUE 중)  
**SQuAD, RACE서 SOTA  
**버트를 재성립시킴(공헌점이자 논문 제안 포인트)  
*요약하면  
**버트 디자인 엡디어트(학습 전략 등)  
**CC-NEWS 새 데이터셋 사용  
**올바른 디자인 사용 pre-training이 성능을 향상  