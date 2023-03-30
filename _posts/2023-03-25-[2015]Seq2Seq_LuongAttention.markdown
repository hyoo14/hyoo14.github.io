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
** 소스 present s는 decoder hidden 초기화 때 한번만 사용  
** 반면 본 모델과 Bahdanau, Jean 에서 s는 hidden state 집합, 번역서 계속 이용  
*** attention 기법  
* 본 논문  
** LSTM 쌓아서 NMT   
** Zaremba LSTM 씀  
** 목적함수: Jt = Sigma(x,y) in D -log p(y|x)    
** D는 병렬 학습 코퍼스  
<br/>
# 3 Attention-based Models  
* 어텐션 기반 모델  
** 어텐션 global, local 둘로 나뉨  
*** 소스 전체냐 일부냐를 보는 거에 따라 나뉨  
* 둘 다 공통:  
** t 시점에서 ht(소스)->LSTM->Ct(Context from 소스)->yt(예측)  
** 위를 통해 attentional hidden state 얻음(concat해서)  
*** ht_tilda = tanh(Wc[ct;ht])  
** 위 ht_tilda가 softmax로 가서 예측  
** 이제 각 모델이 ct 어떻게 계산하는지 설명할 예정  


## 3.1 Gloabal Attention  
*global attention  
** context vector ct구할 때 모든 hidden state 고려  
** alignment at는 소스의 타임스텝과 길이 같고 현재 target hidden state ht와 소스 hidden state h_bar s로 계산  
** score는 attention 기반 펑션으로 일컬어짐, 세가지 안이 있음  
*** dot(내적), general(weighted dot), concat(weighted concat ht, t_bar_s)  
(ht 는 현 target state, hs_tilda는 소스 state 전체.)  
(ht, hs_tilda로 alignment vector at 예측)  
(ct는 at 가중으로 계산)  
* 위치 기반 함수 at 만듬 = softmax(Waht)  
** ct는 모든 소스 hidden 가중 평균으로 구함  
** global과 Bahdanau 비슷한데 차이점:  
*** 단순화, 일반화  
*** Top LSTM hidden state 사용-인/디코더 모두에서  
(바다나우는 bidirectional encoder/1개 층 uni directional decoder )  
*** 계산 심플 ht->at->ct->ht_tilda & Eq5, 6,Figure2로 예측  
(바다나우는 ht-1->at->ct->ht, deep output&maxout layer 이후 예측)    
(Local attention model:)  
(1. 현 단어에 대해 single align 위치 Pt 예측)  
(2. Pt 주변 context ct 계산(가중평균으로), at는 현 target state ht와 소스들 hs_bar로 추출)  
*** alignment concat 말고도 test 및 더 나은 대안 찾음  


## 3.2 Local Attention  
* Local 어텐션  
** global 단점:  
*** 비쌈  
*** 긴 seq가 번역에 실용적이지 않음  
** local 제안  
*** target word 주변 subset만 고려  
** soft/hard 어텐션 tradeoff서 영감(image caption 모델의)  
*** soft: global all patch  
*** hard: one patch at a time  
** hard는 싸지만(추출 시간), 미분 불가& 분산감소나 가오하학습과 같은 복잡한 기술이 train에 필요  
** 위에서 small window 떠올림  
*** 계산 싸고 미분 가능  
*** 학습 hard 보다 쉬움
** train 구체적  
*** 타겟에 대해 시간 t일때 aligned 위치 pt 생성  
** ct는 [pt-D, pt+D] 범위(윈도우)서 가중평균으로 구함  
** monotonic alignment(local-m)는 pt=t 설정, 소스와 타겟 seq가 monotonic하게.  
** 예측 alignment(local-p)는 monotonic 가정x position 예측   
*** pt = S sigmoid(vpT tanh(Wpht)),  
*** Wp, vp는 모델 파라미터 for predict  
*** S: 소스 sent 길이, pt in [0, s]  
** 가우시안 분포 pt 중심으로 사용  
** Eq(7) function 사용  
** 경험적으로 sigma = D/2 deviation  
** pt는 실수, s는 정수, pt는 윈도우의 중심  
** Gregor와 비교  
*** 선택적 어텐션(본 모델 local과 유사, image generation용임 Gregor는)  
*** 모델이 image patch 선택하게함  
*** 본 모델은 대신 zoom 씀, 모든 target position에 formula 단순화되고 성능도 굿  


## 3.3 Input-feeding Approach  
* 어텐션 독립적이지만 과거 반영하게 "커버효과"  
** 이전 align 정보 사용  
** 효과  
*** (a) 이전 alignm fully 앎  
*** (b) DN 구축, 넓고 길게    
* Bahdanau와 비교  
** context vector 사용(본 논문 ct와 같은), "coverage효과" 얻음    
*** 효과 입증은 안 됨  
*** 본 논문이 더 일반적 in 쌓기 RNN(어텐션 없을 때도)  
* Xu는 doubly attentional 접근  
** 추가 제약 사용(학습 때)  
*** 모든 부분에 동이랗게 어텐션(집중)  
** NMT에 유용  
*** 본 논문은 input-feeding사용, 이는 제약에 더 유연  
<br/>
# 4 Experiments  
* 실험   
** WMT 영/독 테스트  
** newstest2013 학습  
** 번역 성능 case-sensitive BLEU on newstest2014, 2015  
** BLEU 타입  
*** (a) tokenized  BLEU(기존 NMT와 비교)  
*** (b) NIST BLEU WMT 비교  


## 4.1 Training Details  
* 디테일  
** WMT'14 학습(4.5M sent, 116M 영단, 110M 독일어 단어)  
** vocap top 50K로 제한(빈도)  
** 적은거-><unk>  
* 트레이닝 세팅  
** 50 초과시 sent 제외  
** LSTM 4layer 각 1000cell, 1000차원 임베딩  
** (a) 파라미터 균일 초기화 [-0.1, 0.1]  
** (b) 학습 10에폭 SGD   
** (c) l.r 1 ~5에폭까지, 이후 반감기 각 에폭마다  
** (d) 미니배치 128사이즈  
** (e) gradient normalize(norm 5 초과시)  
** drop out 사용, drop out 학습 12에폭 동안, 이후 l.r 반감기  
** MATLAB으로 구현  
** 테슬라 K40 GPU, 속도: 1K target words / s, 7-10일 동안 학습  


## 4.2 English-German Results  
* 영/독 결과  
** WMT14 승자와 비교  
*** phrase 기반 시스템 LM은 큰 monolingual text로 학습  
*** Common Crawl corpus 사용   
*** NMT best는 Jean이었고 이것도 비교  
* 향상  
(a) 소스 거꾸로 +1.3 BLEU  
(b) dropout 사용 +1.4 BLEU  
(c) global attention 사용 +2.8 BLEU  
(d) input feeding +1.3 BLEU  
(e) local attention +0.9 BLEU(global 대비)  
** PP와 번역의 연관관계  
** 총 5.0 BLEU 향상(non 어텐션 베이스라인 대비)  
** 이미 알려진 향상법(거꾸로 입력, dropout 포함)  
** 안 알려진 대체 기술 +1.9 BLEU  
** 본 어텐션 모델 유용한 alignment 학습  
** 8개 다른 모델 앙상블(다양한 어텐션 / dropout 유,무에 따라)  
*** 23.0 sota BLEU 달성, 기존 sota대비 1.4 BLEU 향상  
** WMT15 결과: WMT14로 학습이지만 일반화 잘 됨 확인  


## 4.3 German-English Results  
* 영/독 결과2  
** WMT15   
** 어텐션으로 +2.2BLEU  
** boost로 +1.0BLEU  
** data alignment function + dropout +2.7BLEU  
** 모르는 단어 대체->2.7BLEU  
<br/>  
# 5 Analysis    
* 분석    
** 나은 이해 위해  
*** long sent  
*** 어텐션 선택 중요  
*** 영/독 newstest 2014  테스트  


## 5.1 Learning curves  
* 러닝 커브  
** non attention과 attention 비교  
** local 어텐션 cost작고  
** non attention +dropout 학습 느림 non-droput 대비  
*** 대신 더 robust하게 만듬(error 낮춤)  


## 5.2 Effect of Translating Long Sentences  
* 긴 문장 번역 효과  
** 본 논문이 압도  


## 5.3 Choices of Attentional Architectures  
* 어텐션 선택  
** 어텐션(global, local-m, local-p)  
** alignment function(location, dot, general, concat)  
** TB4보면 위치기반 함수가 좋은 alignment 못 배움  
** 다른 alignment 함수 대비 global 어텐션으로 성능 약간만 향상  
** content 기반 function 차원서 conat은 성능 별로, 이유분석 더 필요  


## 5.4 Alignment Quality  
* alignment 퀄리티  
** 단어 배치는 어텐션 모델 산물이지만 평가척도가 없었음  
** 평가척도 Alignment Error Rate(AER) 산정  
** RWTH 508 영/독 문장의 황금 alignment 주어질 때,  
*** 본 어텐션 모델이 강제로이 번역 output 하게 force  
** one-to-one alignment 추출, source word 추출로(높은 배치 weight로)  
** AER 성취! one-to-many alignment 대비 나은 성능  
** 논문 어텐션 만든 alignment 은 global 보다 AER낮음 발견  
** AER 앙상블 결과는 좋지만 local-m 보단 별로  
** AER과 번역 점수 상관관계 없음 알려짐  
** space 제약 땜에 visual 결과는 아카이브에 수록  


## 5.5 Sample Translations  
* 번역 예  
** 이름 잘 번역  
*** non-attention은 소스와 커넥션 부족하여 잘 번역 못 함  
** 두번째 예시 흥미로움  
*** 이중 부정 잘 번역  
*** 우위 확인  
<br/>
# 6 Conclusion  
* 2가지 간단 효과적 어텐션기법 제시 for NMT  
** global: 모든 소스 위치 다 봄  
** local: subset만 봄  
** WMT 영/독서 효과 입증, local로 5.0BLEU향상(non-attention대비)  
** 앙상블로 SOTA in WMT14, 15   
* 다양 배치 함수 비교  
** 어떤 어텐션에 어떤 모델이 BEST인지  
* 분석  
** 어텐션>논어텐션  
*** 특히 이름과 긴 문장에서 어텐션이 강력  










