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
*auxiliary: 보조의, 예비의    
*burgeoning(burgeoning field): 자라나는, 신흥의, 급증하는    
*rigorous: 철저한, 엄격한, 엄밀한    
*sinusoidal: 사인 곡선적    
*gibberish: 횡설수설    
*lorem ipsum: 로렘 입숨(내용보다 디자인 요소를 강조하는 텍스트, 공간 채움을 위한 의미없는 글)    
*word sense: 단어 의미? 참 뜻?      
*disambiguation: 중의성 해소(word sense disambiguation: 단어 중의성 해소)      
*pronoun: 대명사  
*resolution: 결의한, 해결, 결단, 해답, 결심, 해상도  
*fortuitous: 우연한, 행운의, 뜻밖의    
*forgoes:  포기하다, 그만두다, 버리다  
*ostensibly:  표면상, 구체적으로 나타내는, 표면적으로    
*suboptional: 차선책, 선택적??  
*amortize: 분할하다  
*opted: 택하다, 결정을 내리다    



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
*본 논문에서는 체계적으로 이러한 공헌들을 합리적인 베이스라인과 특성들을 바꿔보면서 연구함  
**비지도 목적함수 달리하며 수행(근데 비용 매우 비쌈)  
**더 풍부하고 철저한 조합들로 연구하는 것이 미래 목표  
*다양한 테스트들에 따른 다양한 다른 접근법들을 비교하는 것이 목표(가능한한 특성들 고정하며)  
**완전 똑같이 기존의 접근법 복제하진 않음  
*베이스라인 셋업 이후, 아키텍처들 비교(3.2), 목적함수들 비교(3.3), 프리트레인 데이터셋들 비교(3.4), 전이학습 접근론 비교(3.5), 스케일링(3.6), sota(3.7)  


# 3.1. Baseline  
*베이스라인 타입, 모던 예제에 적용 목적  
*스탠다드 트랜스포머 프리트레인, 간단 디노이징 목적함수, 각 다운스트림 태스크들로 파인튜닝  


# 3.1.1. MODEL  
*스탠다드 인코더-이도커 트랜스포머 사용(single stack구성 또는 분류/길이 예측)  
**생성/분류 태스크들에 괜찮음 확인    
*버트 베이스 스택 유사한 사이즈, 상세 세팅(12블럭의 인코더/디코더, 블럭은 셀프어텐션, 옵셔널 인코더-디코더 어텐션으로 구성하고 FFNN가짐, FFNN의 차원은 3072, ReLU이어짐, 어텐션헤드12개)  
*나머지 레이어나 임베딩은 모두 768차원이고 토털 220million 파라미터(이는 버트베이스의 두배), 드롭아웃 0.1  


# 3.1.2. TRAINING    
*모든 태스크들은 text-to-text tasks 따르도록 함  
**이는 standard maximum likelihood 학습 가능케 함(티쳐 포스, 크로스엔트로피 로스)  
**최적화 위해 AdaFactor 사용, 테스트 때는 greedy decoding 사용  
*프리트레인 2^19 steps on C4, max length 512, batch size 128 seq  
*멀티플 seq 압축 batch에, 배치는 총 2^16 토큰 포함, 토털 2^35(34B) 토큰  
**이건 버트 137B 토큰 보다 적음(RoBERTa 2.2T 토큰)  
*프리-트레이닝 과정에서 inverse square root 사용 1/ root( max(n, k) ), n은 현재 트레이닝 반복 회수, k는 warm up step(10^4), l.r0.01, exponentially decay..근데 이거보다 더 generic한 버전 만들어 사용  
*파인튜닝 2^18steps-trade off로 선택된 회수(하이 리소스 태스크들 사이의), 배치사이즈128len, 512seq, l.r0.001, save 5000steps마다  


# 3.1.3. VOCABULARY  
*SentencePiece 사용(text-> word token), 32,000 wordpieces사용, 영독/프/루마니아 번역 파인튜닝 때 영어 외에도 커버해야함(그래서 C4 독어, 불어, 루미나아어 만듬, 센텐스피스 또한 다 섞어서 학습)      
**이렇게 생성된 어휘들이 공유되고 사용됨 in/out으로  


# 3.1.4. UNSUPERVISED OBJECTIVE  
*프리트레인을 위한 언레이블드 데이터 사용이므로 레이블 관련된 목적함수는 필요없지만 다운스트림테스크에서 도움이 되긴함  
**디노이징이 최근에 좋은 성능 보임  
**MLM과 단어 드랍아웃 제한 기술에 영감받아서 랜덤 샘플과 드랍아웃 15% 토큰 착안  
***좀 더 복잡 조합 드랍아웃, 마스킹 사용  


# 3.1.5. BASELINE PERFORMANCE  
*다운스트림 태스크에 따른 베이스라인 결과 다룸(코스트는 큼, 드라마틱한 변화 기대X   
*메인 텍스트에서 결과 보기위해 스코어 평균 사용(GLUE, SGLUE), BLEU SCORE도 봄  
*정확한 매칭 스코어와 F1스코어는 상관관계가 매우 큼(그래서 정확 매칭만 따짐)  
*모든 스코어, 모든 태스크에 대한 결과는 태이블 16(어펜딕스E)에 있음  
*기존 버트베이스 80.8SQuAD, 84.4MNLI, 본 실험 80.88, 84.24, 직접 비교에 무리가 있는 점은 본 구현은 인코더/디코더 모델이고, 스텝수가 1/4이기 때문  
*전반적으로 직접 비교가 어렵  


# 3.2. Architectures  
초기 트랜스포머 구조는 인코더/디코더이지만 최근 자연어처리 전이학습용은 다른 구조들 사용  
이 구조들을 리뷰해봄  



# 3.2.1. MODEL STRUCTURES  
*주요 차이점은 다른 마스킹 방식을 가진 어텐션 메커니즘 사용  
*트랜스포머의 셀프-어텐션 연산은 문자열 입력과 새로운 동일 길이 문자열 출력을 수반  
**가중치 계산하여 반영  
*첫 고려 모델은 인코더-디코더 트랜스포머  
**2layer, 2stack, fully visible attention mask(출력 때 모든 입력 다 반영(봄), prefix는 나중에 예측, 버트는 fully visible사용  )  
*트랜스포머 디코더의 셀프어텐션은 causal(인과적?) 마스킹 패턴 사용  
**문자열 중 예측 이전 것들만 반영, 이후 것들은 마스킹  
*인코더-디코더 트랜스포머의 디코더는 autoregressive하게 출력열 생성  
*LM일반적으로 압축 또는 문자열 생성에 사용    
**그러나 text-to-text framework로 쉽게 이용(input과 target 이어붙여서) 가능  
**약점은 causal masking이 input 기준 이전에 지나치게 의존적으로 만듬  
**트랜스포머 기반 LM의 새로운 masking 패턴으로 해결됨(prefix LM에서 fully visible masking으로)    


# 3.2.2. COMPARING DIFFERENT MODEL STRUCTURES  
*유의미한 비교를 위해 다양한 상세 세팅 고려해줌(인코더-디코더 모델에서)  
**레이어 개수, 파라미터 개수 버트베이스 개준 L, P, FLOPS는 M(L+L 레이어 인코더-디코더 위한)  
*상세세팅(비교)들은 아래와 같음  
**인코더-디코더 모델 L 레이어 in 인코더, L 레이어 in 디코더, 2P 파라미터, M FLOPs 연산코스트  
**위와 동일 모델이지만, 인코더와 디코더 사이의 P 파라미터가 공유됨  
**인코더-디코더 모델, 각각 인코더/디코더에 L/2 레이어 갖고 P 파라미터 가지며 M/2 FLOP cost  
**디코더only LM, L 레이어, P 파라미터, M FLOP cost  
**디코더only 로 위와 갖은 구조이되 prefix LM(레이어나 파라미터는 갖고 대신 fully visible 셀프어텐션)  


# 3.2.3. OBJECTIVES  
비지도학습 목적함수로 기본LM 목적함수, 베이스라인 디노이징 목적함수 고려  


# 3.2.4. RESULTS  
*디노이징 목적함수 사용한 인코더-디코더 구조가 가장 좋은 성능 보임  
*인코더/디코더 파라미터 공유 유/무는 성능에 크게 연관 없음  
*레이어수 줄이면 성능 떨어짐  


# 3.3 Unsupervised Objectives  
*비지도학습 목적함수 정하는 것 매우 중요  
*기존 방법 그대로 사용하지 않고 text-to-text에 맞게 고쳐서 사용  


# 3.3.1. DISPARATE HIGH-LEVEL APPROACEHS  
*첫번째 방법, prefix LM 목적함수 사용 - text span 두 부분으로 나누고 각각 input/target 으로 줌     
*두번째 방법, MLM from BERT(단, 버트는 encoder only 이므로 본모델 적용 위해 uncorrupted를 target으로)  
*세번째 방법, basic deshuffling(디노이징 seq 오토인코더 방법으로써)  
*버트 스타일인 두번째 방법이 성능 좋음  


# 3.3.2. SIMPLIFYING THE BERT OBJECTIVE  
*디노이징 목적함수 개조 실험  
**기존 15% 마스킹, 약간 변조는 MASS(by Song)  
*그냥 마스킹이 아닌 유니크 마스킹 실험  
**마스킹될 길이 조절 등  
*결과는 성능 다 비슷  


# 3.3.3. VARYING THE CORRUPTION RATE  
*오염 10, 15, 25, 50% 실험  
**50%만 현저히 성능 덜어지고 나머지는 비슷, 15%로 결정  


# 3.3.4. CORRUPTING SPANS  
*랜덤, 연속적 길이 선택 고려  
**3개가 성능 좋았음  


# 3.3.5. DISCUSSION  
*디노이징 목적함수가 좋고 그 이외에 변주는 크게 성능을 좌지우지하진 않음  
**대신 트레이닝 스피드에는 영향 있음  
*결론은 언레이블 데이터 탐색이 더 성능향상의 여지가 있다는 것  


# 3.4. Pre-training Data set  
*본 논문이 나오기까지 새 데이터셋 만드는 것 적극적이지 않았음  
**본 논문에서 C4 제안, 공개  


# 3.4.1. UNLABELED DATA SETS  
*C4의 다양 변주들 만들어보고 실험함  
**C4 -베이스라인(휴리스틱 필터 사용)  
**Unfiltered C4 -langdetect 사용하여 영어 추출(휴리스틱 필터 없이)  
**RealNews-like -C4에 추가적 필터링, RealNews 데이터셋 사용, non-news 뺌  
**WebText-like -C4에서 URL유례 빼고 다 제외하니 2GB만 남음.. 그래서 Common Crawl data  12달 돌림  
**Wikipedia -텐서플로우 위키 텍스트 사용 markup/ref 제외  
**Wikipedia + Toronto Books Corpus -위키피디아의 중의성 제외 약점 보완위해 토론토북 코퍼스 합침  
*C4에 휴리스틱 필터 제거시 성능 저하  
*Wiki+TBC는 베이스라인보다 SGLUE성능 좋음  
*위키 사용이 SQuAD 성능 끌어올림  
*도메인 적응과 평행을 이루는 결과  


# 3.4.2. PRE-TRAINING DATA SET SIZE  
*C4는 큰데이터셋 만즈는 것 가능케 함  
**데이터셋 사이즈 달리하여 테스트(2^29, 2^27, 2^25, 2^23 토큰)  
**결과는 큰 것이 성능 제일 좋음  


# 3.5. Training Strategy  
*다양한 파인튜닝 트레이닝 모델 실험  


# 3.5.1. FINE-TUNING METHODS  
*모든 모델 파라미터 파인튜닝은 서브옵셔널한 결과 나옴(오버핏?)  
**대신 두 대안 방법 적용-인코더-디코더모델의 서브셋만 파인튜닝  
*첫번째로, 적응 레이어 방법, 추가적 댄스-ReLU-댄스 블락을 두어 이것만 파인튜닝  
*두번째로, gradual unfreezing, 점진적으로 더 많은 모델 파라미터 파인튜닝하는 방식  
*첫 방법이 더 성능 좋았음  


# 3.5.2. MULTI-TASK LEARNING  
*단일 비지도학습으로만 진행했었는데 대안으로 멀티테스크 학습있음   
**다양 테스크 한번에 학습시키는 것  
*다양 전략 테스트  
**Examples-proportional mixing -길이가 오버피팅 좌우해서 길이 인위적으로 제한, 샘플 확률 m번째 task는  min(em,K)/SIGMA min(en,K) 이 때, K는 인위적 길이 제한
**Temperature-scaled mixing -다언어 버트용으로 리소스 적은 언어를 위함, 온도 T라면 mixing rate rm은 1/T이고 sum=1되게 조정  
**Equal mixing -동일하게 샘플링 uniform하게  
*결과는 멀티태스크 러닝이 성능저하 가져옴(오버피팅)  


# 3.5.3. COMBINING MULTI-TASK LEARNING WITH FINE-TUNING  
멀티테스크 러닝이 sota이니 변주 적용해봄  
*먼저 프리트레인 examples-proportional mixture with limit K=2^19, 파인튜닝-프리트레인 동시 leave-one-out MT(다른 분야 비지도학습 없는 거 보고 영감)  
*결과는 멀티테스크 프리트레인 후 파인튜닝이 좋음  


# 3.6. Scaling  
*머신러닝의 쓴 교훈은 짱짱한 컴퓨팅 파워가 승리한다는 것  
**이것은 nlp 전이학습에도 해당됨  
**하지만 스케일하는 데에는 여러 방법이 있음  
**베이스라인모델 220M 파라미터, 프리트레인/파인튜닝 2^19/2^18 스텝, 인코더/디코더 사이즈는 버트베이스와 같음  
**확장 실험위해 버트라지 따름, dff=4096, dmodel=1024, dkv=64, 16head어텐션  
***16/32레이어 2가지로 실험(인코더 디코더 각각 16or32)  
**파라미터 2배, 4배 실험, 스텝수 2배 4배 실험(배치4배는 결과가 달라짐)  
*결과는 자연스럽게 키운 것이 좋음  
**다만 큰 모델 만들 시 파인튜닝이 비쌈(트레이드오프)  


# 3.7. Putting It All Together  
앞서 얻은 인사이트들 모두 반영 합침  
*목적함수 -디노이징 mean span length 3, corrupt 15%  
*더 긴 학습 -pretrain 1M steps(batch size 2^12, seq length 512, pre-train token 1 trillion)  
*모델사이즈   
**베이스 -220m 파라미터, 스몰 - dmodel=512,dff=2048,8head attention, 6layer 각각 인/디코더, 60m 파라미터  
**라지 -dmodel=1024,dff=4096,dkv=64,16head attention, 24layer각각 인/디코더, 770m 파라미터  
**3B and 11B - 왕창 크게 라지대비 3배, 11배 만듬  
*멀티테스크 프리트레이닝 -적용  
*파인튜닝 각각 GLUE, SGLUE -벤치마크 다 컨캣해서 학습  
*빔서치 - 그리디 디코딩, 빔위드 4, 랭스패널티 alpha=0.6  
*테스트셋 -WMT tasks 번역셋, GLUE,SGLUE official, SQuAD 사용  
*24개 중 18개에서 SOTA(11B가 해냄)  
*WMT는 SOTA 못 찍음-영어만 언레이블드 데이터 때문일 것(백트랜스레이션 없음)  


# 4. Reflection  
중요 발견에 대한 복습  


# 4.1. Takeaways  
*Text-to-text : 요약, 분류, 생성 명료하게 통합  
*Architectures : 인코더-디코더 구조 사용, 텍스트-투-텍스트에 제일 적합, 인코더(또는 디코더)만 보다 두배 성능  
*Unsupervised objectives : 디노이징 목적함수가 성능좋음 for 비지도 프리트레이닝  
*Data sets : C4 휴리스틱 클린 텍스트 제공  
*Training strategies : 적은 파라미터만 파인튜닝하는 것이 좋음  
*Scaling : 큰 것이 짱  
*Pushing the limits : 엄청 키워서 SOTA 찍음  


# 4.2. Outlook  
*The inconvenience of large models : 커야 성능 좋았음    
*More efficient knowledge extraction : 효율적 프리트레인(일반적 지식학습) 방법은 아니었음(제안모델이)  
*Formalizing the similarity between tasks : 태스크 간의 유사도 활용하면 결과 향상에 도움될 것  
*Language-agnostic models : 일반 언어 모델에 적용하면 좋을 듯  












   












