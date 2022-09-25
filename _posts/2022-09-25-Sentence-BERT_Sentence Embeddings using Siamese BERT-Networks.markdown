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


링크 (https://drive.google.com/drive/folders/1TfN3ngXQxHX-0UzQgcOTxz3nRjDwlr4c?usp=sharing)


# 단어정리  
*argument: 논쟁, facet: 한 면, infeasible: 실행 가능하지 않은, polarity: 극성, newswire: 보도자료배포통신사  
 

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


