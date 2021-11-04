---
layout: post
title:  "November week1"
date:   2021-11-01 16:20:10 +0900
categories: study
---




{% highlight ruby %}

11월 첫째 주:  
논문  
-Attentionn is All You Need  

모델링  
-BERT Finetuning   

CODE  

REC-SYS  

기타  
-eng, app



{% endhighlight %}

2021년 11월 01일 월요일  
논문  
-읽기 시작  
모델링  
-데이터확보_서울시음식점  
-cp949라서 좀 읽는 방식이 다름.. encoding='cp949' 추가해줘야함.  
-좀 달라져서 파인튜닝은 내일 해봐야할 듯..  
CODE  
-징검다리 바이너리서치 문제 커밋  
REC_SYS  
-정량적 평가 척도 RMSE(Root Mean Square Error)-실제값과 예측값의 차이를 제곱하여 평균하여 root해준 것(근데 scale dependent함,  
rmse 낮다고 반드시 좋은 것은 아님-왜냐하면 민값 가깝기만 하게 추천하면 이상)  
-랭킹문제에서 랭킹 순서가 잘 나왔는지를 평가하는 NDCG(Normalized Discounted Cumulative Gain)-topN 랭킹리스트 만들고, 더 관심있거나 관련성 높은 아이템 포함 여부 평가  
(CG-연관성 값(예를 들어 0또는 1)합해준 것, DCG-랭킹에 따라,, 멀어질수록 디스카운트 해준 것, IDCG-정답셋, NDCG-DCG/IDCG )  
-그 외  
(topK- 연관된 것 개수/전체개수, MAP(Mean Average Precision)- 각각 topK(Precision) 구하고 평균,,아래참고)  
-그 외2  
(Precision(topK) = TP / (TP + FP) = 잘맞춘값 / 전체맞춰본값, Recall = TP / (TP + FN) = 잘맞춘값 / 맞춰야할값)  
(Average Precision = Precision 값들의 평균, MAP=AP를 전체 대상으로 mean한 값)  

기타  
-eng  

2021년 11월 02일 화요일 & 2021년 11월 03일 수요일  
논문 -읽는중  
모델링 -전처리중  
CODE -정리 및 한문제 품  
REC_SYS -정리는 내일..  
기타 -eng  

2021년 11월 04일 목요일  
논문 -거의 다 읽음  
모델링 -파인튜닝 여러번 돌려보는 중(성능이 안 좋음.. score 0.1)  
CODE
REC-SYS -콘텐츠기반 추천(고전)은 코사인 유사도, kNN(유클리디언 거리 기준), 나이브베이시안, tfidf(피처 만듬) 등을 사용함.  
기타 -ENG  











