---
layout: post
title:  "Exploring the Limits of Transfer Learning with a Unified Text_to_Text Transformer"
date:   2022-12-23 15:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

transfer learning -> nlp 적용으로 좋은 성능 보임  
본 논문에서는 통합된 text based 문제를 text-to-text 포맷으로 만들어주는 프레임워크를 소개하면서 transfer learning 기술들을 탐색  
다음들을 비교: 프리트레인 목적함수, 구조, 레이블링되지 않은 데이터셋, transfer 접근법, 12가지 언어이해 테스크에서의 다른 요소
이러한 탐구에서의 인사이트들을 종합한 스케일된 것과 "Colossal Clean Crawled Corpus" 를 통해 SOTA를 달성 in benchmarks(요약, QA, 텍스트분류,  등)   
NLP transfer learning의 퓨처워크 활용을 위해 데이터셋, 프리트리엔된 모델들, 코드 공개함  

    
{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1j7ewpojOBNdOpyQgVkzIxnj4GW9BfK40?usp=sharing)


# 단어정리  
*auxiliary: ㅇ  
*burgeoning(burgeoning field): o  
*rigorous: o  
*sinusoidal: o

# 1. Introduction  
*transfer learning to perform nlp에는 downstream 러닝을 위한 text처리가 필요   
**이것은 텍스트를 이해하는 일반적 목적의 지식을 개발하는 것으로 볼 수 있음  
**이 지식들은 low-level(스펠링 또는 단어의 의미)에서 high-level(투바는 백팩에 담기에는 너무 크다)까지의 범위를 커버함  
**모던 머신러닝에서 이러한 지식은 외적으론 드뭄, 대신, 부분으로써 학습됨(단어임베딩)     
*최근 전체 모델을 데이터가 풍부한 테스크에서 프리트레인하는 것이 매우 증가함  
**이는 이상적으로는 일반적인 능력을 향상시켜주고 지식이 다운스트림 테스크에 적용(전이)되게 함  
**컴퓨터비전에서의 이런 활용은 이미지넷과 같은 라벨이 잘 된 큰 데이터를 활용하여 프리트레이닝함  
**반면, 전이학습 nlp는 비지도학습을 프리트레인하는데 레이블링 안 된 데이터 사용  
**이것은 sota 찍었고 다양한 데이터가 많고 크로울링도 할 수 있다는 장점임  
*이러한 시너지는 nlp 전이학습 발전에 큰 활약, 그리고 빠른 속도로 발전중       
*더 엄격한 이해가 필요함에 모티브를 받아 통합된 방법을 사용하여 체계적으로 다른 접근법들 연구하고 현재의 한계에 푸쉬함  
*기본 아이디어는 텍스트 처리 문제를 text-to-text 문제로 치환  
**QA, LM, span추출과 같은   
**장점은 모든 테스크가 같은 모델, 목적함수, 학습 과정, 디코딩 프로세스를 갖게 할 수 있음  
**영어 기반 NLP 문제(QA, 요약, 감정분류 등)에 적용 가능   
**이러한 통합적 접근을 통해 다른 목적함수들, 언레이블드 데이터셋, 다른 요소들 비교해볼 수 있음  
*본 논문의 목적은 새로운 방법을 제시하는 것이 아니라 이 분야의 이해와 안목을 주는 것  
**서베이, 탐구, 경험적 비교 수행  
**한계를 살피고 sota 결과 얻어냄  
**실험을 위해 Colossal Clean Craweld Corpus (C4) 소개  
***수백기가 데이터로 구성된 정제된 영어 텍스트, 웹에서 수집됨  
***모델 비롯 데이터들, 코드들 모두 공개함  
*본 논문은 베이스 모델에 대해 논의하고 그것의 구현, 우리의 텍스트 처리 치환(텍스트2텍스트로), 특정 태스크에 적용의 흐름을 따름  
**실험(섹션3)과 종합을 통한 통찰로 sota 얻은 결과를 보임(섹션3.7)  
**마지막으로 요약하고 퓨처워크 소개(섹션4)  


 # 2. Setup  
라지 스케일 스터디 전에 결과에 대한 이해를 위해 백그라운드 토픽을 리뷰하였음  
트랜스포머 모델 아키텍처와 다운스트림 태스크들의 평가 등.  
또한 모든 문제를 text-to-text로 풀어낸 접근방법 소개 Colossal Clean Crawled Corpus(C4) 설명함  
본 논문의 모델과 프레임워크는 Text-to-Text Transfer Transformer(T5)로 명명  


# 2.1. Model  
*초기 전이 학습 for NLP는 RNN이용했으나 최근에는 트렌스포머 구조 사용이 일반적  
**MT때 처음 등장한 트렌스포머는 NLP전반에 넓게 퍼짐  
**본 논문서 연구한 모든 모델들은 트랜스포머 기반  
*트랜스포머의 주요 빌딩 블록은 셀프 어텐션임  
**셀프 오텐션은 어텐션의 변형형으로 시퀀스가 대체됨 각 요소에 그 요소를 제외한 가중 평균으로..
**트랜스포머 원본은 인코더-디코더 아키텍처이고 seq2seq 태스크를 위한 설계임  
**최근엔 싱글아키텍처 접근 LM용으로 쓰이고 우리는 이 아키텍처 사용  
*인코더-디코더 트랜스포머 구현은 오리지널 형태에 가깝게 구현  
**input->map->sequence embedding -->pass-->encoder  
**encoder(blocks로 구성, self attention layer+FFNN)  
**+layer normalization(simple버전으로 activation rescaled, additive bias 없음)  
**+residual skip connection(sub component의 in/out에)  
**+dropout(FFNN과 함께, skip connection에서, 어텐션 웨이트에서, in/out전반에서)  
**디코더도 비슷하지만 스탠다드 어텐션을 포함함(autoregressive 또는 causal self-atteition형태)  
***모델이 과거 결과에 접근하게 해줌  
**디코더 out은 dense softmax layer로 감  
**모든 트랜스포머는 독립적 head들로 나눠지고 output전에 합쳐짐     
*셀프어텐션은 순서의존적  
**기존 트랜스포머는 sinusoidal 위치 또는 학습된 위치 임베딩 사용, 최근에는 상대적 거리 임베딩 사용이 일반적  
**단순화된 위치 임베딩 사용  
**종합하여 거의 오리지널과 같으나, 다른 점은 layer norm bias 제외, layer norm을 residual path바깥에 배치, 다른 위치 임베딩 스킴 사용임  
**실험의 일환으로 규모성 테스트함, 모델 컴비네이션과 데이터 병렬화, tpu pod으로 나눠서 학습 1,024tpu v3, mesh tensorflow library사용  


# 2.2. The Colossal Clean Crawled Corpus  
*nlp경향: 언레이블 데이터셋 비지도학습용으로 사용  
**본 논문에서는 언레이블드 데이터의 퀄리티, 특성, 사이즈에 따른 효과 측정에 흥미  
**니즈에 맞는 데이터 생성을 위해 Common Crawl로 web으로부터 수집(원래 n-gram lm용, commonsense reasoning 사용, MT 병렬연구서 사용, 테스팅 최적화 등지에서도 사용)     


















