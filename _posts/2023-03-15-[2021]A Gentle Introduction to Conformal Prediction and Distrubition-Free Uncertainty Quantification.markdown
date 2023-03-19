---
layout: post
title:  "[2021]A Gentle Introduction to Conformal Prediction and Distrubition-Free Uncertainty Quantification"
date:   2023-03-15 19:57:59 +0900
categories: study
---


# 기타 정보(수집된)  
* gentle introduction to conformal prediction은 conformal prediction이라는 통계적 기법을 쉽게 설명한 문서입니다.  
conformal prediction은 입력에 대해 예측 구간이나 클래스 집합을 추정하는데, 이들은 실제 값을 높은 확률로 포함한다는 것이 보장됩니다.  
이 문서는 통계학자가 아니더라도 분포-자유 불확실성 정량화를 실제로 구현하는데 관심이 있는 독자를 위한 것입니다.  
파이썬과 파이토치를 사용한 많은 그림, 예시, 코드 샘플이 포함되어 있습니다.  
<br/>
* conformal prediction은 예측 알고리즘(보통 머신러닝 시스템)에 대한 가정 없이 데이터의 교환가능성만 가정하여 예측 집합을 생성하는 통계적 기법입니다.  
예측 집합은 사용자가 지정한 확률(예: 90%)로 실제 값을 포함한다는 것이 보장됩니다.    
conformal prediction은 이전에 레이블된 데이터에 비일치 측정값(점수 함수라고도 함)을 계산하고, 이를 사용하여 새로운 (레이블되지 않은) 테스트 데이터 포인트에 대한 예측 집합을 만듭니다.  
conformal prediction은 컴퓨터 비전, 자연어 처리, 강화 학습 등의 분야에서 발생하는 문제에 자연스럽게 적용되는 이해하기 쉽고 사용하기 쉬운 일반적인 기법입니다.  
<br/>
1장에서는 분포-자유 불확실성 정량화의 필요성과 동기를 설명합니다.  
2장에서는 conformal prediction의 기본 개념과 원리를 소개하고, 이미지 분류와 회귀 문제에 적용하는 예시를 보여줍니다.    
3장에서는 conformal prediction의 변형과 확장 방법들을 소개하고, 각각의 장단점을 비교합니다.    
4장에서는 conformal prediction과 관련된 다른 분포-자유 불확실성 정량화 방법들을 소개하고, 각각의 특징과 한계를 설명합니다.  
5장에서는 conformal prediction과 관련된 연구 동향과 응용 분야를 소개하고, 앞으로의 연구 방향을 제안합니다.  
<br/>
* 1장에서는 분포-자유 불확실성 정량화의 필요성과 동기를 설명합니다. 
먼저, 기계 학습 모델의 불확실성이란 무엇인지와 왜 중요한지에 대해 간단히 소개합니다.  
그 다음, 기존의 불확실성 정량화 방법들이 가지는 한계와 문제점들을 살펴봅니다.  
예를 들어, 베이지안 방법은 계산 비용이 많이 들고, 데이터의 분포에 대한 사전 지식을 요구하며, 모델의 가정과 일치하지 않을 수 있습니다. 
마지막으로, 분포-자유 불확실성 정량화라는 새로운 패러다임을 제시하고, 이것이 어떻게 기존의 방법들보다 우수하고 유연한지를 설명합니다. 
분포-자유 불확실성 정량화는 데이터의 분포에 대한 가정 없이 작동하며, 임의의 모델에 적용할 수 있으며, 통계적으로 엄격한 보장을 제공합니다.  



{% highlight ruby %}
짧은 요약 :    
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


[Paper with my notes](https://drive.google.com/drive/folders/1VcrHqSlLZKguhClBB0X8MOU4_-NSGJ7E?usp=sharing)  


[Lecture link](https://www.youtube.com/watch?v=usaHyuu2TzY)  


# 단어정리  
* rigorious: 엄밀한  
* quantification: 정량화  
* conformal prediction: confidence level 측정, uncertainty 측정  
* quantile: 분위수, 전체 분포를 특정 개수로 나눌 때 기준이 되는 수  
* remark: 비고  

   

# 1 Conformal Prediction    
* Conformal Prediction은 prediction set을 만드는 직관적인 방법  
* 크게 두 단계로 구성  
** 1. fitted 모델(학습된) f_hat으로 시작  
** 2.f_hat 이용해서 prediction set(가능한 레이블들의 집합) 만듬  
*** 이 때, 적은 양의 추가적인 calibration(교정용) 데이터 사용  
* 보다 상세한 절차 또는 개념  
** k개 클래스가 있다고 가정  
** 각 클래스마다 분류기의 결과값 확률 추정(소프트맥스 점수, [0.1]값)  
** 숫자 하나(N) 정해서 iid(independent.identical.distribution)하게 데이터인풋과 클래스 쌍 N개 뽑음->교정용 데이터 셋(calibration dataset)    
*** 이 때, 이 데이터/레이블 쌍은 학습 때 사용되지 않은 거여야 함  
*** 그리고 이는 (1)번 식을 만족함  
**** 1-alpha <= P(Ytest (= C(Xtest)) <= 1-alpha+ 1/(n+1)  
**** (Xtest, Ytest) 는 같은 분포에서 온 사용되지 안하은 데이터/레이블 쌍  
**** alpha는 유저에게 입력받은 에러율  
**** prediction set이 올바른 레이블을 가질 확률은 거의 1-alpha 임  
**** 우리는 이 프로퍼티를 marginal coverage라고 함  
**** 이는 교정셋과 테스트셋의 어떤 쌍의 확률이 랜덤성보다 평균적으로 얼마나 나은지 나타냄  
** f_hat에서 가능한 클래스 레이블 집합(prediction set) 만들 때, 간단한 교정 스텝 수행
*** 코드 수는 단 몇줄에 불과함  
*** 상세한 교정 스텝 설명  
**** 1. conformal score 정의 si = 1 - f_hat(Xi)Yi  (1-softmax 실제 클래스일 결과)  
**** 점수는 소프트맥스 실제 클래스 결과가 낮을 경우 높아짐, 모델이 아주 성능이 안 좋을 때  
**** 2. 스코어 s1,...,sn을 보고 경험적으로 정한 분위수(quantile) q_hat 을 반올림( (n+1)(1-alpha) ) / n 로 정의   
**** (원래 q_hat은 1-alpha quantile이지만 맞추는 경우가 적었(small correction)음)  
**** 3. 새로운 데이터 포인트(Xtest는 알지만 Ytest는 모르는)를 위해 prediction set을 만듬  
**** prediction set C(Xtest) = {y:f_hat(Xtest)y >= 1-q_hat} 는 softmax값이 충분히 큰 클래스를 모두 가짐   
*** 이 알고리즘은 (1)을 만족하는 것을 보장하는 prediction set을 줌  
*** 어떤 모델이 사용되건 어떤 알지못하는 분포의 데이터를 사용하던  


### Remark  
.  














