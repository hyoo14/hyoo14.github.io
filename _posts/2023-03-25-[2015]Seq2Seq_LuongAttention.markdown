---
layout: post
title:  "[2015]Effective Approaches to Attention-based Neural Machine Translation"
date:   2023-03-25 19:59:30 +0900
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :    
* 어텐션이 NMT 향상  
* 근데 아직 연구/접근법 적음  
* 간단, 효과적 2가지 방법 소개  
	* global: 전체 소스 문장 단어 다 봄  
	* local: subset만 봄  
* WMT 영/독 번역으로 실험  
	* 5.0 BLEU 향상 with dropout  
* 앙상블 모델로 SOTA 달성  
	* 25.9 BLEU로 기존 n-gram reranker에 1.0BLEU 앞섬    




{% endhighlight %}  

<br/>


[Paper with my notes](https://drive.google.com/drive/folders/1pGtzNyK5IkgwWkjayZVsoc6ryOpl986-?usp=sharing)  


[Lecture link](https://vimeo.com/162101582)  

<br/>

# 단어정리  
* monotonic: 단조로운  
* alignment: 정렬, 가지런한  
* monotonic alignment: 시퀀스의 단조성과 거리를 동시에 고려하는 기법  
* monotonicity of sequence: 시퀀스의 원소가 증가하거나 감소하는 경향  
* monolingual: 하나의 언어  
* by-product: 부산물, 부작용, 이차 물품?  
   

   

# 1 Introduction  
* MNT 활약  
** 영프, 영독 번역서 SOTA  
** sent 전부 다 봄(어텐션이)  
** 전부 다 볼 필요 근데 없음   
** 모델 자체 mem도 작기 때문에..  
** MNT 디코더는 간단(기준 MT보다)  
* NN에서 어텐션 인기 증가  
** 다른 모달간의 alignment 학습  
*** 이미지 객체 & 행동  
*** speech frame & text in speech recognition  
*** picture & text description  
** NMT  
*** Bahdanau 성공적  
* 새 attention 설계(2가지, 심플 효과적임)  
** global:   
*** all source 반영  
*** Bahdanau와 유사하지만 더 심플  
** local:  
*** subset 반영  
*** hard/soft attention mix   
*** computation cost global 보다 작음    
*** hard attention보다 구현, 학습 더 쉬움    
** 다양한 alignment function 실험들 논문에서 소개  
* 실험으로 입증된 것   
** 영/독 양방향 번역서 효과적  
** non-어텐션 대비 5.0BLEU 향상(drop-out사용)  
** WMT14, 15 영/독 번역서 SOTA  
** NMT + n-gram LM 대비 최소 1BLEU 초과 달성  
* 기타 분석  
** 학습, 긴문장 처리 능력, 어텐션 선택, alignment 퀄리티, 번역 결과 소개  
<br/>  


# 2 Neural Machine Translation  
* NMT  
** NN모델: 조건부 확률 P(y|X) 가능성으로 번역  
*** source x1,...,xn to target y1,...,yn  
** 구성  
*** 인코더: 소스문장 representation 계산  
*** 디코더: 한번에 one target word generation  
**** 조건부 확률 분해  
* 디코더 모델  
** RNN 선택  
** 연구들마다 각기 다른 RNN 사용  
*** Kalchbrenner&Blunsom: RNN with 표준 hidden unit for decoder and CNN encoder  
*** Sutskever, Luong: RNN 여러층 with LSTM hidden unit for enc/dec  
*** Cho, Bahadanau, Jean: 특이 RNN, GRU(LSTM서 영감) for enc/dec  
** 본 논문 더 자세히->확률 변수화, 각 단어 yj: P(yj|y<j,x)=softmax(g(hj))  
*** hj는 RNN 히든  
*** hj=f(hj-1,s)서 f는 현 hidden 구함, RNN/LSTM/GRU가 될 수 있음  











