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


   
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1VcrHqSlLZKguhClBB0X8MOU4_-NSGJ7E?usp=sharing)  


[Lecture link](https://www.youtube.com/watch?v=usaHyuu2TzY)  


# 단어정리  
* .

   

# 1 Introduction  
* .

