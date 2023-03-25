---
layout: post
title:  "[2021]A Gentle Introduction to Conformal Prediction and Distrubition-Free Uncertainty Quantification"
date:   2023-03-15 19:57:03 +0900
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :    
*ML 자체가 블랙박스여서 불확실성에 대한 정량화가 필요
**특히 의료분야에서 그냥 쓰기에 위험이 큼  
**conformal prediction은 예측모델에 통계적으로 엄밀한 불확실성 집합을 제공  
**특히 분포를 모를 때 사용 가능  
**사전 학습된 뉴럴네트워크에서 ground truth을 포함하는 것으로 보장된 집을 만들어줌  
***이 때, 유저가 특정화한 확률 기반으로 함 90%와 같은  
**이해가 쉽고 사용이 쉽고 일반화, 여러 분야에 적용이 쉬움, 예를 들어 cv, nlp, deep 강화학습 등  
*본 논문의 목적 및 공헌  
**conformal prediction과 관련 분포에서 자유로운 불확실성 정량화 기술 소개  
**관련 예제 및 이론 서술-머신러닝 태스크 포함, 타임시리즈, 분포 전환, 아웃라이어 포함  
**코드도 제공  





{% endhighlight %}  

<br/>


[Paper with my notes](https://drive.google.com/drive/folders/1VcrHqSlLZKguhClBB0X8MOU4_-NSGJ7E?usp=sharing)  


[Lecture link](https://www.youtube.com/watch?v=usaHyuu2TzY)  

<br/>

# 단어정리  
* rigorious: 엄밀한  
* quantification: 정량화  
* conformal prediction: confidence level 측정, uncertainty 측정  
* quantile: 분위수, 전체 분포를 특정 개수로 나눌 때 기준이 되는 수  
* remark: 비고  
* conformalizing: 일치하는, 부합하는 -> provide valid intervals of prediction  
* deviation: 편차, 통계학에서 deviation은 관측값에서 평균 또는 중앙값을 뺀 것으로, 자료값들이 특정값으로부터 떨어진 정도를 나타내는 수치  
* reliable: 신뢰할 수 있는, 믿을만한, 신뢰성  
* magnitude: 크기의 정도  
* residual: 추정된 식과 관측값의 차이, 예측값과 실제 관측값의 차이, 잔차  
* perturbations: 변화를 나타내는 작은 변화, 근사값 추정    
* sake: 목적, 이익, 원인, 혜택이나 이득  
* multiplicative: 곱셈의, 급격하게 증가하는  
* pragmatic: 실용적인  
* intersperse: 배치하다  
* stratified: 층을 이루게 하다, 계층화하다  
* discretize: 연속적인 공간이나 물건을 나누다  
* benign: 상냥한, 유순한, 자비로운, 친절한, 양성의(악성이 아닌)    
* fluctuation: 파동, 이상한 변화, 변동    
* stratify: 층화, 데이터를 분리  
* progenitors: 특정한 현상이 발생하기 전의 원인을 찾아내는 것  
* progeny: 어떤 현상이 발생한 후의 결과  
* precocious: 조숙한, 빠른 학습, 빠르고 높은 성능을 보임  

   

# 1 Introduction  
* .   