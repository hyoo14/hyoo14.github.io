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
* prediction set C에 대한 해석  
** set-valued(대응, 매핑함수)로 데이터 인풋과 매핑되는 클래스들의 셋(집합)을 갖고 있음   
*** 모델의 소프트맥스 아웃풋을 이용하여 셋 생성함  
** 입력에 따라 다른 결과값 생성(당연한 거 아닌가)   
** 모델이 불확실하거나 인풋이 어려우면 집합의 크기가 커짐  
*** 집합의 크기는 모델의 확실성 지표로 볼 수도 있음  
** C(Xtest)는 Xtest에 할당될 수 있는 괜찮은 클래스들로 해석할 수도 있음   
** C는 valid하고 (1)식 만족시킴  
*** 이는 회귀같은 기계학습 문제로 볼 수 있음    

<br/>
* 본 제시 모델 보다 일반화   
** 우선 소프트맥스 결과값으로 각 클래스별 조건 확률 측정  
*** 예를 들어 j번재 결과는 P(Y=j | X=x)로 측정-인풋x일 때 클래스j일 조건부 확률    
** 그런데 소프트맥스가 모든 케이스에서 좋으리라는 보장은 없음  
*** 오버핏되거나 신뢰할 수 없기도 함  
** 그래서 소프트맥스 대신에 holdout set을 사용하여 부족한 부분을 매꾸었음  

<br/>
* holdout set이란  
** n = 500 언저리의 신선한(학습 때 사용되지 않은) 데이터로 성능 판단용  
** 데이터셋 분할은 conformal score 계산을 수반함, 모델이 불확실하면 커짐  
** 하지만 커진다고 valid한 고간을 갖는 것은 또 아님  
** 본 논문의 경우 conformal score는 1-실제 클래스의 소프트맥스 결과 였음  
** 일반적으로 그러나 어떤 x, y관련 함수나 스코어 펑션이 될 수 있음  
** q_hat은 1-alpha 분위수(나누는 수)로 사용  
*** 만약 alpha를 0.1로 보면, 적어도 90%의 정답분포 소프트맥스 결과가 1-q_hat 수준으로 보장됨(Appendix D에서 증명됨)    
** 위 사실 기반으로 소프트맥스 결과값으로 새로운 데이터에서 테스트했을 때 1-q_hat기반 prediction set이 C(Xtest)임  
** 결과인 Ytest 클래스는 1-q_hat 분위수 기준 90% 보장되고 식(1) 만족함  
<br/>

## 1.1 Instructions for Conformal Prediction  
* conformal prediction은 소프트맥스 결과값이나 분류 문제에 국한된 것은 아님  
** 불확실성 들어간 어느 휴리스틱 개념이나 모델에 사용 가능하고 더 엄밀하게 만들어줌  
** 예측이 이산형인지 연속형인지, 분류인지 회귀인지 가리지 않음  
<br/>
* 일반적 입력 x와 출력 y인 상황 인스트럭션  
** 1. 사전학습된 모델을  사용하여 불확실성에대한 휴리스틱 개념 식별  
** 2. 스코어 함수 s(x, y) 정의(큰 점수는 x와 y 사이의 불일치 뜻함)  
** 3. 반올림( (n+1)(1-alpha) )/n로 교정 점수 s1=s(X1, Y1), ..., sn=s(Xn, Yn)의 분위수 q_hat 계산  
** 4. 이 분위수 사용하여 prediction sets 만듬  
*** C(Xtest) = {y: s(Xtest, y) <= q_hat}  
** 이 셋은 어떤 스코어함수에 어떠한 분포에서도 (1)식 만족  
<br/>
### Theorem 1  
* Conformal coverae 보장  
** (Xi, yi) i=1,..,n이 있는데 i.i.d(independent, identical distribution)이라 가정    
** q_hat은 위 스텝 3처럼, C(Xtest)는 위 스텝 4처럼 정의  
** 그럴 경우 아래 식 만족  
P(Xtest (= C(Xtest)) >= 1-alpha  
** Appendix D의 증명을 볼 것, 이 성립식은 (1)의 상위 바운드임  
** 이 성립 이론은 특수한 경우에만 성립되며 split conformal prediction이라 부름  
** 이것이 가장 널리 쓰이는 conformal prediction임  
** 본 논문은 여기에 초점  
** 섹션6과 섹션7에서 더 다룰 예정  
<br/>  
### Choice of score function  
* 통계적으로 유효한 prediction set을 심지어 불확실성 휴리스틱 개념의 나쁜 모델로 구축이 가능하다고? Appendix D의 증명에 대한 직관적 설명    
** 만약 스코어 si가 올바르게 입력들을 낮은 에러에서 높은 에러로 랭크매겼다고 가정  
** 결과 셋은 쉬운 인풋일 경우 작고, 어려운 인풋일 경우 커짐  
** 만약 스코어가 나쁘다면, 랭크를 잘 어림잡지 못하고 셋은 쓸모 없어짐  
*** 예를 들어, 스코어가 랜덤 노이즈라면, 셋들은 랜덤샘플 레이블을 포함할 것, 셋의 크기 커짐  
*** 이것은 conformal prediction의 중요 사실 직시함  
**** 비록 보장될지라도 prediction sets의 유용성은 스코어 펑션에 의해 결정됨  
** 스코어 함수(펑션)이 모든 정보, 문제와 데이터, 모델 다 사용하므로 놀랄일은 아님  
** 예를 들어, conformal prediction 적용에 있어서 분류와 회귀의 큰 차이점은 스코어 선택에 있음  
** 하나의 모델이어도 많은 가능한 스코어 함수들이 있고 각기 다른 프로퍼티(속성 값)들을 가짐  
** 그러므로 올바른 스코어 함수를 만드는 것은 공학적 선택에서 매우 중요  
** 다음 파트에서 좋은 스코어 함수의 예들 나열할 것  
 


















