---
layout: post
title:  "Leveraging Dependency Grammar for Fine-Grained Offensive Language Detection using Graph Convolutional Networks"
date:   2022-09-30 17:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

공격적인 언어 탐지에서 올바른 탐지가 중요함  


올바른 탐지  
-false positive 체크  
-차별적 편견 없음  


제안하는 SyLSTM 통해 SOTA 성능 얻음  
GCN+LSTM 구조  

{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1gfg2VPZjYvfNgm5g7YUFfsYg5Z6WBcxg?usp=sharing)


# 단어정리  
*propagation: 전파, innocuous: 무해한, pejorative: 경멸적인, profanity: 욕설, disparage: 얕보다,   
unprecedented: 전례없는, lexial: 단어나 어휘 관한 것, fall prey to: 먹이감이 되다, subsuming: 포함하다,  
compatible: 호환되는, pave the way: 길을 마련하다, plethora: 과다, ethinicity: 민족성, caste: 카스트,  
morphology: 형태, syntax: 통사론, nodal: 노드의, amod: 상태, posit: 단정짓다, lexicon: 사전,  
congruent: 거의 일치, copies: 사본  
 

# 1 Introduction  
*공격적언어  
**욕설, 얕보기(인종, 피부색, 민족, 성별, 국적, 종교 등)  
**소셜미디어에서 폭증중이어서 필터 시급  
*기존의 탐지  
**어법적, 룰적 ngram, bow 사용  
**욕설은 잘 탐지하나 혐오발언 탐지에 약함  
**dnn의 경우 스스로 편견 생김(예를 들어, 아프리칸 아메리칸의 영어를 혐오로 인식)  


*구조적 특징(통사적, 문장에서 단어의 역할을 따지는)이 잠재 공격 탐지에 핵심 역할을 함  
**특히 대상없는 공격 또는 냉소에  
**성급한 일반화 방지에도 역할함(욕설, 인종적 언어와 같은)  
**특정 단어에 대한 bias 극복  


*의존파스트리 중요  
**언어에서 형태학적으로 부유하고 단어순서가 상대적으로 자유로움  
**트위터 어휘로부터 영감 받아서 구조적 특징을 의존문법에서 부여하고 통합함  
**딥러닝에서 사용할 경우 좋을 것  


*SyLSTM제안   
**신텍스기반+LSTM  
**딥러닝+선택적의존성(딥러닝에 의한 시스템 bias 경감시켜줌)  
**의존문법 사용 위해 GCN 사용  
**의존파스트리의 input graph로 호환시킴  
**BiLSTM기반 의미인코더+ GCN기반 구조인코더로 구성  
**softmax 헤드로 분류  
**기존 SOTA버트가 재학습 110M 파라미터인데 반해 제안보델은 9.5M 파라미터로 효율적  
**성능도 SOTA  
**테스트 data  
***hate speech and offensive long detection  
***계층 분류 시스템  
***타입, 타겟, 공격 대상 구분  
**공헌  
***의미피처(임베딩같은)와 구조피처 같이 사용  
***욕설공격탐지 성능 향상, 공격어 인식, 타입, target 3가지 모두 성능 향상  
(논문구성: 2-연관, 3-디자인SyLSTM, 4-구체화(데이터셋+실험), 5-결과, 6-결론)  


# 2 Related Work  
*혐오탐지(20년 넘게 이어져옴)  
**Smokey   
***시작격으로 디시전 트리 분류  
***47신텍스와 이멘틱스 사용  
***3가지 클래스(flame, okay, maybe)  
***전통적 머신러닝 사용의 토대가 됨  
***과다 필터가 약점  


*소셜네트워크 폭증  
**소셜미디어 도메인 도입 계기  
**기본언어학적 피처 제안  
***욕탐지, 혐오발언 내적 탐지  
***욕설탐지 훈련  
****내적 편견에 빠지고 비효율적  


*구조적 피처 중요  
**공격탐지, 타겟 선별  
**SNS서 발견  
***혐오->특정대상  
**언어특성-딥러닝 접목  
***인스턴스 지목에 집중  
**2가지 문제점  
***내재된 편견 생김  
***모델의 도메인 shift  


*구조 피처 사용시 유용함 증명됨  
**non-Euclidean syntactic 특성 사용(의존트리 같은)하므로 cost 절약  


*GNN으로 딥러닝이 그래프까지 확장되고 GCN사용하여(CNN일반화버전) non-Euclidean data까지 확장  
**스케일 확장 커짐  
***Chebyshev 폴리노미얼 기반  
**SNS, NLP, 자연과학서 성공적  


*GCN서 NLP 우수성 입증  
**시맨틱롤 레이블링 분석  
**NLP 피처 추출용 GCN의 기반  
**임베딩용으로 쓰임  
**그러나 약점이 있음  
***멀티릴레이션 그래프서 사용 못 함  
***즉, 의존파스트리 사용 어려움  


*의존파스트리  
**본 논문에서 GCN서 쓰게 처음으로 변환함(공헌점)  
**모델  
**BiLSTM시맨틱인코더-의미피처추출(긴범위)  
**GCN신텍틱인코더-의존파스트리(문장의) 피처 추출  
***결과 향상  



# 3 Methodology  
*어법은 형태론 + 신텍스  
**형태론은 단어의 구조  
**신텍스는 문장서 단어가 어떤 역할을 하는지(명사, 동사 등..)  
*문장  
**신텍스와 단어나열에 의존적  
**명사, 동사 출연 기록으로 볼 수 있음  
***공격 감지에 도움  
***syntactic 구조 공격 post, 타겟, 강도 아는 데 도움  
**본 논문에서는 두 측면 모두 고려(의미+의존문법)  


*전처리 모듈 구현  
**트위터에 반복이 많고 노이즈 있어서 그현함  
***다양한 유저 이름, URL등과 같은  


*SyLSTM  
**6개의 컴포넌트로 구성  
1. 인풋토큰-전처리후 단어 토크나이징 후 인풋  
2. 임베딩 레이어 - 각 단어 작은 차원 피처벡터 맵핑  
3. BiLSTM 레이어 - 단어임베딩->하이레벨 피처 추출  
4. GCN 레이어 - 어법의존 통해 웨이트 만들고 곱해줌  
5. FFN - 차원 축소  
6. 아웃풋 - 3번과 5번 결과 concat하고 디택팅 결과 나열  


*단어임베딩  
**용어  
***S:문장, T:문장단어개수, Xi:i번째 단어, e_i:i번째 벡터(임베딩), |v|:vocab개수, d(w): 차원수(임베딩의)  
**두 방식 사용  
***랜덤 초기화 후 학습  
***사전학습  
****글로브-트위터 임베딩  
****트위터 특수해서 사용  
****27B 토큰 파싱(트위터 코퍼스부터)  
****프리트레인 임베딩이 랜덤이니셜라이즈보다는 좋지만 글로브+트위터보다는 성능 떨어짐  


*BiLSTM  
**기존 GCN은 비방향 노드 표현에 집중  
***싱글릴레이션이기에 적합  
***멀티릴레이션서 gap은 표현의 제약의 문제  
***그래서 BiLSTM 사용  
**적합한 gating 기법으로 LSTM서 중요 피처 추출(이전부터 현재까지)   
**혐오탐지 컨텍스트서 효과적(혐오표현 위치는 랜덤하기 때문)  
**과거, 현재 모두 반영 BiLSTM은 인식률 높음  
**구조  
***input은 문장임베딩  
***2 layer BiLSTM  
***32hidden units  
***dropout 0.4, sequential vector 추출, final hidden state 추출 -> 마지막 state에 concat후 쓰임  
***sequential vector 배치노멀라이제이션 이후 GCN으로(모멘텀0.6)  


*GCN  
**의존파스트리 특별 속성 있음  
***다중관계 선분들  
***선분 타입 다양  
***의미론 선분과 다름  
**GCN input dependecny 핸들 못 함  
***각각 연결 노드로 취급  
***구성 단어 중요성 주기 위해 self-loop 줌  
**이를 통해,  
***의존 파스트리->graph됨  
****G = (V, E)  
****vertices(단어), edges(의존, 반대에 강조, self-loop)  
****비방향그래프 + self loop로 나옴  
****의존구조 확장-하이라이트됨  
****중요단어 위치 + graph->GCN(인접행렬)  
****더해서 dependency -> weight  
****GCN이 conv해서 graph G됨  
**인접행렬 ~A = A(비방향그래프) + I_n(self-loop)  
***~D_ij - sigma A_ij  and W(l)  
***ceta : 활성함수, w(l) : weight  
**학습  
***local graph 구조(의존), nodal피처(컨텍스트서 단어 중요)  
***의미인코더가 구조인코더 보강  
****긴 기간일 경우 GCN에 배니싱 그라디언트 생김  
****배치노멀라이션 모멘텀0.6, 드랍아웃0.5, 자비에분포 이니셜라이징, 아웃풋 32  


*FFN  
**GCN 결과 단층 FFNN으로가고 하이레벨피처 의조구조 의존 학습, 활성함수 ReLU  


*아웃풋  
**FFNN 결과는 LSTM 마지막 히든레이어와 concat되어 softmax 지나 확률 계산해줌  



# 4 Experimental Setup  
*NN + syntactic dependency  
**공격탐지, 2dataset 사용  


*공격어 탐지 데이터셋  
**SmEvalChallenge 2019  
**계층적  
***1. 공격적 여부, 타겟 여부, 타겟이 개인인지 그룹인지 그 외인지 여부  
**레입르링 실험 통해 검증 됨  
**14,- 영문트윗기반  


*혐오발언, 공격어 데이터셋  
**중심문제와 주변문제 관련성 모티브 받음  
**HATE, OFF, NONE으로 구성  
**혐오어 사전 이용하여 Hate-base 구축  
**사전으로 혐오트윗 추론, 25k 트윗 기반  


*베이스라인  
**리니어SVM  
***텍스트분류서 SOTA(?)  
***유니그램으로 학습  
***그리드서치로 bert 파라미터 선택  


**투채널 BiLSTM  
***임베딩스페이스 랜덤이니셜라이제이션  
***버트  
***BiLSTM은 32 hidden units  
***최종에는 forward, backword 합쳐서 MLP softmax로 분류  


**파인튠 버트  
**SOTA 버트  
***SemEval2019우승 버전사용  
***최고 성능 보임  
***SyLSTM  
****이 세팅 참고 (loss function, initializer, learining rate 기반)  



*학습  
**스텐다드 크로스엔트로피  
**AdamW optimizer  
**cosine anealing learnign rate 스케쥴러  
**vocab 30,- in corpus  
**initial learning rate : 0.001  
**regularization 0.1  


**평가척도  
**가중 F1 측정  
**Precision & recall  



# 5 Results  
*결과  
**랜덤이니셜라이징 임베딩 or 글로브+트위터 임베딩 사용  
**student t-test with 가중 F1 측정   
**베이스라인 압도  


*공격어탐지  성능  
**본 모델과 선택적 baseline이 성능 좋음  
*공격어 탐지  
**NN이 LinearSVM 보다 나음  
**본 모델이 압도  
*공격어 카테고리  
**본 모델이 버트대비 4% 성능향상  
*공격어 타겟 인식  
**베이스라인 다 같음  
**본모델이 압도, 버트보다 5.7% 향상  
*혐오&공격어 성능  
**본모델이 압도  


*논의  
**BiLSTM과 BERT 비교 논의  
**근데 의존 tree 변환 시 비방향 그래프(단일관계)로 LSTM서 멀티릴레이션 케어됨  
**BERT+GCN  
***파라미터 과대화  
***sparsity 증가  
***이미 버트는 의존성 학습  
***의존문법 추가가 불필요한 반복임  



# 6 Conclusion  
*SyLSTM 제안  
**Semantic + Syntactic 인코더  
**성능압도(tweet기반으로)  
**의존문법 사용  
**효율적 & 확장성 좋음  
**언어학적 피처 추가에 좋음(DNN과 함께)   














































