---
layout: post
title:  "[2020]Dense Passage Retrieval for Open-Domain QA"
date:   2023-03-04 13:11:42 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

*Open domain QA는 효율적 검색 내포된 task  
**기존 tf-idf, BM25 쓰임(sparse vector)   
**dense representation만 사용으로 더 효과적 성능 가능   
**실험상 성능 sota 찍음, qa서 루신-bm25보다 9~19% 향상  
   
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1HvR4KPgOxNDmgoDysz9t3EE1z1Wl-o6Q?usp=sharing)  


[Lecture link](https://slideslive.com/38939151/dense-passage-retrieval-for-opendomain-question-answering)  


# 단어정리  
* defacto: 사실상  
* factoid: 사실  
* inverted: 거꾸로  
* inverted index: 역 인덱스, 책의 맨 뒤에 있는 주요 키워드에 대한 내용이 몇 페이지에 있는 "찾아보기 페이지"에 비유할 수 있음  
* inverse: 역의, 반대의  
* surrogate: 대리의, 대용의  
* fraction: 부분, 일부, 분수  
* schemes: 계획, 제도, 책략, 기획, 설계, 기구  
* conjecture: 추측, 억측, 어림짐작  
* lexical: 어휘의  
* lexicon: 어휘, 어휘목록, 사전  
* block: 블록  
* middle block: 중간 블록  
* non-iid:(non-independent and identically distributed) 데이터가 비독립적이고 동일하지 않게 분산. 즉 데이터는 종속적일 수 있고 동일한 확률분포를 갖고있지 않음    


   

# 1 Introduction  
* 오픈도메인 QA는 대용량 호부군 중에서 정답을 찾는 것  
**초기 매우 복잡한 컴포넌트가 사용  
**시간이 자나며 2단계로 단순화  
***검색(추림)  
***읽기(정답 찾기)  
**머신리더 없을 시 성능 하향 등 개선이 필요해짐  
* 검색 일반적으로 tf-idf/bm25 같은 keyword 위주이지만 dense representation 사용해서 동의어/파라프이징 잘 찾아줄 수 있음  
**dense vector는 학습 가능하고 엔드테스크 튜닝도 가능  
* dense vector학습 데이터가 많이 필요하지만 tf-idf/bm25 보다 성능 안 좋다 여겨졌었음  
**하지만 ORQA가 인식 깨고 sota성능 보임  
**masked sentence 추가 P-T 이용(ORQA)  
**F-T로 QA쌍 학습    
* QA데이터셋 두가지 문제점   
** 1)P-T  
***가격이 비싸고 일반 문장이 도움되는지 검증되지 않음  
** 2) context 인코더  
***F-T아니어서(QA) suboptimal 결과임(optimal이 아님)  
* 더 나은 dense embedding  
**추가 P-T없이  
**버트같은 plm 사용  
**듀얼 인코더 구조  
**QA pair 이용  
**내적 maximize로 최적화  
**BM25를 성능적으로 큰 격차를 보이며 압도  
**QA서도 ORQA 압도 in OpenNeuralData에서  
* 공헌  
**간단 QA데이터 F-T로 MB25 압도  
**QA데이터셋 실험서 SOTA 달성  


# 2 Background  
*배경  
**다양 토픽, 큰 코퍼스에서 답변해야할 필요성 증대  
**추출 QA상 답변이 1,2 문단으로 제한적  
**반면 코퍼스 풀은 수백만에서 수십억 다큐로 구생  


# 3 Dense Passage Retriever(DPR)  
* 검색 향상 목적으로 dense한 벡터 이용 모델 제안  


## 3.1 Overview  
**덴스인코더 Ep(.) 사용  
***text passage to vector & index build  
*Dense Passage Retriever은 다른 인코더 Eq(.)에 적용  
**input->d차원 벡터&k개의 passage 검색  
**dot product 이용  
**유사도 함수 분해, 연산 되야해서 채택  
**대부분 유사도는 유클리드 거리(L2), 마하노비스(L2), 이너프로덕트 사용  


### Encoders  
*인코더  
**DRP 뉴럴넷 구조지만 두 버트 네트워크 사용 & [cls] 토큰 아웃풋으로 사용, d=768  


### Inference  
*추론  
**passage 인코더 Ep 사용 & FAISS로 인덱싱  
**FAISS 효율적 오픈소스로 유사도 검색&분사벡터 클러스터링에 사용  
**q(질문) 들어올 때 임베딩 Vq=EQ(q)로 top k passage 찾음(Vq는 유사도 의미)  


## 3.2 Training  
*학습  
**인코더 학습: 검색 잘 되게 metric학습  
***연관 pair의 벡터간 거리 가깝게  
**D가 트레이닝 데이터 의미, m은 인스턴스(q:질문, Pi+:연관문단, Pi-:n개의 연관성 없는 문단)  
**negative log likelihood로 최적화  


### Positive and negative passages  
*연관 없는 passage pick 어렵고 중요하지만 간과됨  
**(1)랜덤 (2)BM25 낮은 score 가진 것들 (3)다른 GOLD  
**(3)이 가장 성능 좋음  


### In-batch negatives  
*배치 내부에서 negative해서 사용  
**Q:질문, P:문단, S=QPt matrix 구성, 사이즈는 B라 가정  
**B**2의 q/p mat는 효과적으로 재사용 가능  
**B는 배치 안에서 트레이닝의 인스턴스가 됨  
**이러한 minibatch는 좋은 전략으로 일컬어짐  


# 4 Experimental Setup  
*데이터와 세팅  


## 4.1 Wikipedia Data Pre-processing  
*위키데이터 전처리  
**위키 dump 20.12.20 사용  
**DrQA 전처리 코드 사용  


## 4.2 Question Answering Datasets  
*QA 데이터  
**5개 QA dataset 같은 split으로 나눔  
### Natural Questions(NQ)  
**Natural Questions(NQ)  
***구글 서치 쿼리와 정답  
***위키서 추출  
### TriviaQA  
**TriviaQA  
***사소한 QA  
***웹서 추출   
### WebQuestions(WQ)  
**WebQuestions(WQ)  
***구글 제안 API & Freebase answers  
### CuratedTREC(TREC)  
**CuratedTREC(TREC)  
***웹 소스의 코퍼스 for 오픈도메인QA  
### SQuADv1.1  
**SQuADv1.1  
***RC 벤치마크  
***위키 문단  
***주석자가 레이블   
***오픈도메인 QA용이지만 적합성은 떨어지는 것으로 평가됨(context부족하여)  
****비교 위해 사용   
### Selection of positive passages  
**긍정 문단 선택  
***TREC서 QA만 주어짐(WQ, TriviaQA도)  
***BM25로 passage 검색  
***정담이 top100 안에 없으면 Q 삭제  
***위키 버전 달라서 일치하지 않을 시 삭제  


# 5 Experiments: Passage Retrieval  
*실험: 문단 찾기  
**검색 성능 평가  
**분석(전통검색론, 학습스킴(방법)에 따른 효과, 런타임 효율)  
*DPR(DensePassageRetrieval) 모델  
**in-batch negative 세팅(batch size128, BM25 negative passage) 사용  
**q&p 인코더 학습  
***40epoch for 큰data(NQ, TQA, SQuAD)  
***100epoch for 작은 data(TREC, WQ)   
***l.r:10e-5, Adam optimizer, linear schedule warm up 사용 & drop out rate: 0.1  
*multi dataset 인코더 학습  
**학습데이터 SQuAD 빼고 합침  
**BM25 결과 사용  
**리니어 콤비네이션 of score for new rank  
**위 바탕으로 top-2000 passage들 얻음  
**합침 : BM25(q,p) + rambda * sim(q,p) 랭킹함수, rambda=1.1  


## 5.1 Main Results  
**SQuAD제외 모든 데이터셋에서 제안 모델이 BM25 앞섬  
**gap도 큼(특히 k 값 작을 수록)  
**트레이닝 데이터셋 여러개일수록 데이터 적은 TREC서 성능 높아짐  
**대표적으로 NQ, WQ 향상폭 더디고 TQA는 감소됨  
**DPR+BM25가 약간 더 성능 향상  
**SQuAD 성능 낮은 이유:  
***1)레이블링한 사람이 질문작성 후 passage를 보기 때문에 높은 lexical 겹침 생기고 BM25가 이러한 경우에 강하므로  
***2)데이터가 500개 조금 넘는 위키피디아 문서로붜 추출되었기 때문에 분포가 편향적임  


## 5.2 Ablation Study on Model Training  
*경감스터디  


### Sample efficiency  
**샘플 효율성  
***얼마나 많은 호부예가 필요한지 테스트  
***1000개 dense 학습으로 이미 BM25 능가  
***PLM에 적은 수 학습으로 성능 높일 수 있음  


### In-batch negative training  
**배치 내부 negative 학습  
***rand/BM25/gold 전략 중 k>=20이면 성능 비슷함  
***batch 안에서 진행하므로 메모리효율적이고 학습쉬움  
****성능도 향상되고 batch size 커질수록 성능 더 향상  
***hard negative 추가시 1개 추가일 때는 성능 오르나 2개 이상부터는 안 오름  


### Impact of gold passages  
**gold passage 포함 여부 아주 중요치 않음  


### Similarity and loss  
**유사도와 loss  
***cosine, 유클리디안(L2) 테스트  
***L2가 dot product만큼 성능 보임, cosine은 성능 떨어짐  
***triplet loss 사용 효과 적음  


### Cross-dataset generalization  
*데이터 교차 일반화  
**다른 데이터셋으로 추가 F-T없이 성능 나는지 테스트  
**성능 남. 일봔화 잘 된 것이고 BM25보다 뛰어남  


## 5.3 Qualitative Analysis  
*질적 분석  
**BM25는 키워드 위주이고 DPR은 의미관계 위주여서 조금 다름  
**어펜딕스 참고  


### 5.4 Run-time Efficiency  
*런타임 효율성  
**open domain 호부군 줄이기가 목적  
**검색 속도 DPR 매우 빠름  
***995 questions / second  
***100 passages / second  
***BM25는 23.7 questions / second  
**index 빌딩은 DPR이 오래걸림  
***21M 임베딩 처리해야하는데 8.8시간 on 8GPU, FAISS는 21M 벡터처리, 8.5시간  
***BM25는 36분  


# 6 Experiments: Question Answering  
*다른 passage 검색 실험  


## 6.1 End-to-end QA System  
**End-to-End QA 구현 및 테스트 with 다른 검색시스템 사용  
***길이를 달리하거나 시작 부분을 범위의 시작, 끝, 선택 중 어디로 할 지를 달리하여 실험  

## 6.2 Results  
**DPR이 잘 맞음  
**DPR이 Sota  
**일부 DPR+BM25가 성능 잘 나오고, 멀티 트레이닝이 성능 잘 나옴  


# 7 Related Work  
**문단 찾기는 오픈QA의 중요 부분  
**TF-IDF & BM25 널리 사용되어옴  
**dense벡터 사용 역시 역사는 오래되었지만 그간 sparse벡터 보다 성능 떨어짐  
**최근 denseQA시도되고 성능 보임  
***추가 P-T 사용  
**DPR 연구 널리 성행, 검색능력 상승중(BART나 T5와 융합 통해)   


# 8 Conclusion  
*dense 검색 굿  
**성능 압도 보임(전통 대체 잠재력 충분)  
**간단 듀얼 인코더로 성과 굿  
**실험&경감 스터디  
***더 어렵고 복잡한 구조나 유사도 검색 필요하지 않음을 보여줌  
**SOTA 달성  

















