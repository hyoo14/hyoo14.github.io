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
*gibberish: o  
*lorem ipsum: o  
*word sense: o    
*disambiguation: o  
*pronoun: o  
*resolution: o  


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
*Common Crawl은 web extrated text를 통해 제공됨  
**문제는 자연어만으로 이루어지지 않았다는 것   
**문제 해결을 위해 아래와 같은 휴리스틱 따름  
***마침표/느낌표/물음표/인용기호로 종결된 라인들만 보유  
***5문장보다 적은 페이지는 버리고 3 단어 이상 가진 문장만 보유  
***부적절하거나 나쁜 단어를 포함한 페이지는 버림  
***javascript 포함 페이지 버림  
***"lorem ipsum" 있는 text 버림  
***중괄호( "{" ) 있는 경우는 코드인 경우가 많으므로 버림  
***중복된 경우 모두 버림( 세 문장당 하나는 버림)  
**추가적으로, 다운스트림 태스크는 영어 텍스트에 집중, langdetect로 영어 아닌 페이지 식별(가능성 0.99이하)  
**이전 크롤러에서 영감받아서 만듬(라인레벨, 하지만 데이터 새로 만듬-필터에 너무 피팅되서, 새로운 range, 병렬처리 트레이닝 데이터에 초점을 맞춤)  
*데이터 합칠 때 19년 4월에서부터부터 누적, 다운로드  
**텍스트 고르고 필터 적용  
**C4라 명명, 텐서플로우 데이터셋 통해 배포  


# 2.3. Downstream Tasks  
*본 논문의 목표는 일반적인 언어 학습 능력을 측정하는 것  
**MT, QA, 요약, 텍스트 분류 포하  
***GLUE, SuperGLUE text분류, CNN/CM 요약, SQuAD QA, WMT 영독/프/루마니아 번역  
****모두 텐서플로우 데이터셋에 존재  
*데이터셋들 아래와 같음  
**문장 허용여부 판단(CoLA)  
**감정분석(SST-2)  
**파라프레이징/문장 유사도(MRPC, STS-B, QQP)  
**NLI(MNLI, QNLI, RTE, CB)  
**Coreference resolution(WNLI, WSC)  
**문장 완결성(COPA)  
**Word sense disambiguation(WIC)  
**QA(MultiRC, ReCoRD, BoolQ)  
*GLUE, SuperGLUE에 의한 데이터셋 분산 사용  
**단순함을 위해, 파인튜닝시 GLUE벤치마크 테스크들 모두 하나의 태스크로 다룸(컨캣해서)  
**Definite Pronoun Resolution도 포함시킴  
*CNN/DM 데이터셋은 QA로 소개되었지만 텍스트 요약으로 적용됨  
**비익명버전을 사용함  
**SQuAD는 일반적 QA 벤치마크, 실험과정서 모델은 질문과 맥락으로부터 답변을 토큰 하나하나씩 생성  
**WMT 영/독을 위해 본 논문은 newstest2013과 같은 학습데이터을 검증셋으로 사용  
**영/프를 위해 standard dataset2015와 newstest2014를 검증셋으로 사용  
**영/루마니아 위해 WMT2016 검증셋 사용  
**프리트레인은 영어 데이터이고 새 언어를 위한 텍스트가 필요함  


# 2.4. Input and Output Format  
*다양한 태스크들의 단일모델 학습을 위해 모든 태스크들을 text-to-text 포맷으로 캐스트해줌  
**일관적인 트레이닝 목적함수를 프리트레이닝과 파인튜닝에서 모두 갖게 해줌, maximum likelihood 목적함수+티처포싱 사용  
**태스크 특화된 prefix사용하여 태스크들 구분지어줌  
*예를 들어, "That is good"을 영어에서 독일어로 번역한다고 하면, 모델은 translate English to German: That is good"이라고 입력받고 "Das ist gut"이라고 출력함  
**텍스트 분류의 경우 모델이 간단하게 단일 단어와 일치하는 타겟 레이블을 예측해줌  
***예를 들어, MNLI에서 목적은 전제가 내포하는지를 예측해주는 것(수반, 반박, 중립 과 같은)  
****전처리를 통해 입력 시퀀스는 mnli premise: I hate pigeons. hyphothesis: My feelings towrards pigeons are filled with animosity" 타겟은 entailment.  
*****아무 레이블도 일치하는 것이 없을 경우 이슈가 발생함. 이 경우 모델 아웃풋을 wrong이라 카운트. 하지만 한번도 이런 적은 없었음  
*****텍스트 prefix 선택은 하이퍼파라미터임, 우리는 prefix 특정단어 바꾸는 것이 제한해주는 효과를 주는 것을 발견함, figure1다이어그램이 몇몇 input/output을 보여줌  
*본 논문의 텍스트 투 텍스트 프레임워크는 다양한 nlp 태스크들의 공통 포맷을 따름  
**대신 따로 파인튜닝하지 않고 동시에 해결(제로샷 러닝이나 스팬추출 등 대신에 전이학습에 집중)  
*본 논문은 직접적으로  모든 태스크를 텍스트-투-텍스트로 캐스트함(STS-B 제외)    
**STS-B는 유사도 점수를 1-5까지 중 하나로 예측하는 목표(점수는 반올림함)    
***회귀문제를 21개 클래스 분류 문제로 효과적으로 바꿔줌  
*각각 WNLI(from GLUE), DPR(from SuperGLUE) 태스크들도 간단한 포멧으로 텍스트투텍스트로 바꿔줌  
**예를 들어, WNLI Winograd task들에서는 어구 포함을 언급?  
***예: "The city councilmen refused the demonstrators a permit because they feared violence." 포함여부 "they" -> "city councilmen" or "demonstarators"  
****모호한 대명사를 강조하는 것으로(언급하는 것이 무엇인지 알아내는 것으로) 치환  
*WSC를 위해, 모호한 대명사, 정답지 명사, 참/거짓이 잘 맞는지를 모두 학습(참/거짓이 뭔지 모르는 채로)  
*WNLI위한 것은 WSC와 유사  
**검증 예 부족을 피하기 위해 WNLI를 학습시키지 않고 WNLI 검정셋의 결과를 리포트하지 않음   
**WNLI 검증 제외는 스탠더드임  
***그러므로 WNLI가 avg GLUE에 포함되지 않음  



# 3. Experiments  
*최근 NLP의 전이학습의 진보는 다양한 개발들의 도래를 불러움, 새로운 사전학습 목적함수들, 모델 구조들, 언레이블드 데이터셋 등  
**이 섹션에서 이 기술들을 각각 경험적 서베이하고 공헌과 중요점 봄  
***인사이트들을 통합하고 sota 얻어냄  
*본 논문에서는 체계적으로 이러한 공헌들을 











