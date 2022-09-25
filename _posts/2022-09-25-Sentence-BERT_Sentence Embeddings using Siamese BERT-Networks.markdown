---
layout: post
title:  "Sentence-BERT_Sentence Embeddings using Siamese BERT-Networks"
date:   2022-09-25 17:45:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

BERT, RoBERTa SOTA 성능 보였지만, 속도는 극악  
특히 문장 비교에서  


그래서 SBERT제안  
-cosine 유사도 같은 느낌 적용 가능케함  
-속도 5초로 버트 사용보다 훨씬 빠름  
-siamese와 triplet networks 사용  


SBERT, SRoBERTa로 STS task서 SOTA 찍음  



{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1TfN3ngXQxHX-0UzQgcOTxz3nRjDwlr4c?usp=sharing)


# 단어정리  
*argument: 논쟁, facet: 한 면, infeasible: 실행 가능하지 않은, polarity: 극성, newswire: 보도자료배포통신사,  
excerpt: 발췌  
 

# 1 Introduction  
*SBERT는 BERT + siamese and triplet network  
**버트 이용하지 않음  
**라지 스케일 의미대로 군집, 검색에 사용(의미 검색)  


*버트는 SOTA임  
**sentence 분류의 경우 pair 갖고 회귀분석함  
***크로스 인코더와 같음  
**근데 pair 하나하나 다 bert에 넣고 유사도 구한 후, 유사도들을 다 체크해서, 가장 높은 유사도 찾아줘야함  
***시간이 매우 오래 걸림 (n C 2) 만큼 걸림( O(N^2) )  


*군집 and 의미 검색에서 주된 방법은 sentence를 vector space에 매핑시키는 것  
**이 때, BERT의 output의 평균을 사용하거나  
**CLS 토큰 사용  
**근데 이것들의 성능은 GLoVE 사용하여 평균 낸 것 보다 떨어짐  


*그래서 SBERT 제안  
**siamese 네트워크 사용  
**fixed size input 가능  
**의미적 sim 거리 사용  
**유사도 계산 시간 단축  


*SBERT는 NLI로 파인튜닝하고 SOTA 찍음  
**Infersent, Universial Sentence Encoder대비 STS서 11.5, 5.5 points 향상  


*SBERT는 특정 작업(task) 적용 가능  
**similarity detect, triplet dataset으로 다른 섹션 문장 구분과 같은 것에 적용 가능하고 SOTA임  


*본 논문의 구성은 아래와 같음  
**섹션3 - SBERT 소개  
**섹션4 - SBERT 평가(STS, AFS)  
**섹션5 - SBERT 평가(SentEval)  
**섹션6 - 요소 제거 실험  
**섹션7 - 컴퓨팅 효율성 비교  


# 2 Related Work  
*버트와 SOTA 문장임베딩들 소개  


*버트는 PLM으로 트랜스포머네트워크  
**NLP task에서 우수(QA, 문장분류, 문장 pair 회귀)  
**두 문장 입력([SEP])로 구분  
**멀티헤드어텐션 12-24레이어로 구성  
**output은 fin레이블  
**STS서 SOTA  
**RoBERTa 특정 task서 약간 성능 향상  
**XLNet은 성능 그닥 특출나진 않았음  


*버트 약점에 대해  
**독립적 문장 임베딩 없음  
***버트서부터 역산하여 평균내거나 CLS와 같은 특수토큰 사용  
***평가 지표가 없어서 우수성 증명 어려움  


*문장 임베딩은 매우 활발한 분야  
**Skip-Thought  
***인코더-디코더 구조로 학습  
***주변 문장 예측  
**InferSent  
***SNLI와 MNLI레이블 데이터 이용  
***siamese BiLSTM  + 맥주풀링 이용  
***Skip-Thought 압도  
**Universial Sentence Encoder  
***트랜스포머 학습  
***비지도 증폭(SNLI 사용)  
***학습 질이 성능 좌지우지  
**이전의 연구들로부터 인스이트  
***SNLI 학습용으로 좋음  
***Reddit 사용 siamese DAN, siamese 트랜스포머 STS서 성능 좋음  


*cross-encoder BERT  
**런타임 오버헤드가 너무 큼  
**성능은 좋음  
**시간 복잡도 O(N^2)  


*본 논문은  
**이전 뉴런 문장 임베딩의 경우 랜덤 이니셜라이즈로 학습 시작  
**여기선 pre-train BERT, RoBERTa 사용하여 파인튜닝  
**학습 속도 단축됨(20분 미만으로)  
**성능도 잘 나옴  


# 3 Model  
*SBERT는 버트에서 아웃풋 받아서 풀링 거치게해줌  
**아웃풋으로 버트 활용 방안 3가지 중 하나 사용  
***CLS토큰  
***Mean(평균)->디폴트  
***맥스  


*파인튜닝 위해 siamese와 triplet network 사용하여 weight 업데이트  


*네트워크구조는 training data에 따라 다르게 함  


*분류의 경우  
문장임베딩 u, v, |u-v| concat하고 weight multipl해서 사용  
**n은 센텐스 임베딩 차원  
**k는 라벨 수  
**cross entropy로 최적화  


*회귀의 경우  
**코사인 similarity(두 문장 u, v의)  
**MSE로 최적화  


*트리플렛네트워크의 경우  
**앵커문장 a  
**앵커와 긍정 p, 앵커와 부정 n  
**a와 p는 가깝게, a와 n은 멀어지게 학습  
**마진 eta 사용  
***a문장과 p문장 사이의 길이가 최소 eta만큼 n 문장 사이 거리보다 가깝게 함  
**유클리디안 거리 사용  


*SNLI와 MNLI로 학습  
**SNLI 칼로케이션 570,000문장(contradiction, entail, neutral포함)    
**MNLI 430,000 장르(발화와 글로 구성)  
**3way softmax를 분류의 목적함수로 사용  
**batch size 16, Adam optimizer(learning rate는 2e-5, linear learning rates는 10%)     


# 4 Evaluation - Semantic Testual Similarity   
*평가 STS task  
**회귀 매핑은 계산 너무 많음  
***코사인 유사도만 사용   
***네거티브 맨하탄, 네거티브 유클리디안도 테스트 실시하였고 결과는 비슷함  


*비지도 STS  
**STS 2012-2016, sick 관련도(레이블 0-5 의미적 관련도) 사용  
**피어슨 상관도 안 맞음  
***스피어맨 랭크(코사인유사도와 레이블)가 잘 맞음  
**다른 문장 임베딩도 테스트  


*결과  
**그냥 버트의 경우 성능 별로 안 좋음  
**평균 버트 correlation 29.19  
**GloVE 보다 성능 안 좋음  


*본 논문에서 제안하는 SBERT(siamese 파인튜닝)  
**correlation 증가   
**InferSent, Universial Sentence Encoder 능가  
**Sick 관련도는 SBERT가 성능에서 밀림  
***Universial Sentence Encoder의 경우 트레이닝 데이터가 뉴스, QA, 포럼으로 다양해서 relatedness 서 이김  
***SBERT는 wiki만 사용(bert 거쳐서)  
**RoBERTA는 성능 크게 차이 안 남  


*지도 STS  
**8,628 문장(캡션, 뉴스포럼, 카테고리)으로 학습  
**train 5,748, dev 1,500, test 1,379로 구성  
**버트가 SOTA로 이 때는 리그레션 사용  


*SBERT는 train 때 regression이고 test 때는 cosim 사용  
**10 랜덤 씨드로 학습  


*결과  
**STSb로만 학습(비교용)  
**NLI + STSb로만 학습한 경우 STSb만 학습한 것보다 1-2점 향상  
***버트서 성능 크게 향상됨  
***RoBERTa는 크게 성능에 영향 안 끼침  


*AFS(Argument Facet Similarity)로 SBERT평가  
**6,000문장으로 구성  
***3개의 민감한 토픽으로 나누어짐  
***0-5민감도로 레이블링  
***STS와 다른 구성  
**AFS는 발췌된 데이터  
***주관이 다를 수 있음  
***비슷한 추론 가능  
***lexical gap이 훨씬 큼  
***비지도 STS SOTA로 실험할 경우 결과는 처참  
**SBERT의 경우  
***10fold cross validation 적용  
***약점은 불명확함, 다른 토픽간 일반화가 어려운 점으로 예상  
***cross-topic 남은 토픽으로 평가  
**이거 3토픽으로 반복하고 평가메김  
**SBERT회귀 목적함수로 학습, 코사인 유사도로 평가  
***피어슨 상관도는 STS비교 보다 약함  
**비지도 tf-idf, 평균 GloVe, InferSent 성능이 SBERT 보다 못 함  
***10fold cross validation SBERT는 거의 BERT 성능에 근접  
**cross-topic평가에서 SBERT의 성능이 떨어짐  
***BERT는 어텐션으로 직접 두 문장 비교하는데 SBERT는 매핑되어버려서 이것이 원인  
***버트만큼 성능 나오려면 2토픽만 테스트해야함  


*위키피디아 데이터의 경우 섹션 구분하여 레이블링  
**위키 사용하여 세분화 트레이닝  
***섹션별 긍/부정 나눔  
***같은 섹션의 경우 긍정, 다른 섹션의 경우 부정으로 레이블링  


*Dor 데이터셋 사용  
**트리블렛 오브젝티브 사용  
**accuracy를 척도로 사용  


*SBERT가 BiLSTM(fine tune) 압도    


# 5 Evaluation - SentEval  
*SentEval toolkit으로 평가  
**SBERT의 목적은 전이학습이 아니지만 전이학습 평가용 SentEval은 퀄리티 측정에 좋음  


*7개의 SentEval task는 아래와 같음  
**MR - 영화리뷰 바탕 감정 예측  
**CR - 상품리뷰 바탕 감정 예측  
**SUBJ - 주어 예측(영화리뷰와 줄거리 요약 바탕으로)  
**MPQA - 구단위 의견 분류(뉴스 바탕)   
**SST - 스텐포드 감정 트리(이진 라벨)  
**TREC - TREC의 질문 분류  
**MRPC - MS리서치 파라프레이징 코퍼스(병렬 뉴스 바탕)  


*SBERT의 경우 7개 중 5개에서 성능 좋았음  
**평균적으로 2점 상승  
**비교군 -  InferSent, Universial Sentence Encoder  
**버트가 성능은 제일 좋았음  
**대체로는 SBERT 성능 우위  
**TREC서 Universial Setence Encoder가 더 좋았음  
***QA 학습 했기 때문  
**평균 버트, CLS 토큰 버트 STS에서 평균 GloVe 보다 떨어졌지만 SentEvall에서는 반대  
***세팅 때문임  
****STS는 cosine 유사도, SentEval은 회귀 -> 문장별로 다른 차원에 따른 다른 가중치가 가능하기 때문  
**버트(평균, cls)는 코사인 유사도, 멘하탄 / 유클리디안 거리 쓰기 용이  
***전이학습서 성능은 떨어지지만, 파인튜닝시 SentEval에서 성능 더 올라감  


# 6 Ablation Study  
*SBERT이해 위해 Ablation study 진행  


*풀링 달리함  
**concat 도 달리 함  


*목적함수-레이블데이터 의존적으로 함  
**NLI, MNLI 학습  
**STS 평가  


*NLI의 경우  
**풀링의 영향은 별로 안 크고 concat은 중요  
**|u-v|, u*v, u, v 사용 따라 성능 많이 달라짐  
**u*v 성능 떨어짐  
**|u-v| 영향 제일 큼  
***소프트맥스에 영향  
***거리여서 similarity 잘 반영되는 것  


*regression 학습시 풀링의 영향이 큼  
**MAX가 성능 떨어짐  
**BiLSTM의 경우 MAX가 성능이 제일 좋은 것과 대조적  



# 7 Computational Efficiency  
*문장 수가 많아지는 경우가 많기 때문에 속도 중요  


*STS로 test  
**GloVe - for loop로 계산  
**InferSent - Pytorch  
**Universial Sentence Encoder - tensorflow  
**SBERT - Pytorch  
***padding token서 계산 줄어짐  


*CPU서 InferSent가 SBERT 보다 65% 성능 좋음  


*GPU서 SBERT가 9% InferSent보다 성능 좋고, Universial Sentence Encoder보다 55% 향상  
**스마트배칭의 경우 CPU서 59%, GPU서 48% 성능 향상  
**GloVe가 fasttext 보다 빠름  


# 8 Conclusion  
*버트  
**일반 문장유사 측정에 적합하지 않음  
**STS서 글로브보다 성능 떨어짐  


*SBERT  
**버트 파인튜닝 siamese, triplet network 사용  
**SOTA  
**RoBERTa는 크게 성능에 양향 없음  
**계산력 좋음  
***GPU서 각각 InferSent와 Universial Sentence Encoder 보다 9%, 55% 빠름  
**버트가 용이하지 않은 task에서도 성능 좋음  
***10,000문장 계층 클러스터링 65시간 걸리는 버트에 비해 SBERT는 5초 걸림  
