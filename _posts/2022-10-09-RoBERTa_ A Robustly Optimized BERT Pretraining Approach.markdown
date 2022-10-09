---
layout: post
title:  "RoBERTa_ A Robustly Optimized BERT Pretraining Approach"
date:   2022-10-09 23:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

PLM 성능이 높아짐 특히 BERT  

이 BERT의 replication study 통해 parameter 조정을 함  

그래서 GLUE, RACE, SQuAD에서 SOTA 달성  


{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1zC8qtkySWQq_PVwUR7q-0bxgIVkXAMyn?usp=sharing)


# 단어정리  
*consecutive: 연속적인(연이은), quantifies: 양화(증량화), auxiliary: 보조자, amenable: 받을 수 있는,  
disentangle: 풀다  


# 1 Introduction  
*Self-train model의 어떤 hyperparameter가 성능 영향 끼치는지 모름  
**계산 비싸고 튜닝 제한, 사적 data 확장 제한, 성능 측정 어려움 있기 때문  
*BERT 복제 실험으로 RoBERTa 제안  
**버트 대비 성능 향상됨, 차이점은 아래와 같음    
**1. 모델 길게, 배치 크게, 데이터 양 더 많이  
**2. NSP 제외  
**3. 긴 시퀀스로 학습  
**4. 학습데이터 마스킹 때 다이나믹하게 해줌  
**또한 CC-NEWS 큰코퍼스 수집  
*학습데이터 각기 다른 엔드테스크에서도 본 논문 제안이 SOTA 찍음  
**glue 88.5(이전 88.4)  
**MNLI, QNLI, RTE, STS-B서 SOTA(GLUE 중)  
**SQuAD, RACE서 SOTA  
**버트를 재성립시킴(공헌점이자 논문 제안 포인트)  
*요약하면  
**버트 디자인 엡디어트(학습 전략 등)  
**CC-NEWS 새 데이터셋 사용  
**올바른 디자인 사용 pre-training이 성능을 향상  


# 2 Background  
*버트 오버뷰  
**사전학습 접근법  
**학습 때 선택(하이퍼파라미터)  


# 2.1 Setup  
*셋업(버트)  
**두 문장 concat & input with 스페셜 토큰 [CLS], [SEP], [EOS]  
**문장 길이 각 M,N 이고 M+N < T, T는 주어진 맥스 랭스  
**큰 언레이블 데이터로 프리트레인, 이 후 파인튜닝(엔드task 레이블 데이터로)  


# 2.2 Architecture  
*아키텍처  
**트랜스포머 아키텍처  
***L (레이어 개수), A (어텐션 헤드 개수), H (히든 디멘션 차수)   


# 2.3 Training Objectives  
*학습 목적 함수  
**MLM & NSP  


*MLM  
**랜덤 마스킹  
**크로스엔트로피 loss를 목적함수로 사용  
**15% 문장 중에서 택해서 이 중 80%는 mask, 10%는 그대로, 10%는 다른 랜덤 단어로  
**랜덤마스킹 대체 경우  
***처음에, 학습 중 저장, 실전서는 데이터 복제되서 mask 달라짐  


*NSP   
**binary classification loss로 두 문장 연속되나 체크  
**p: 연속 문장 from 코퍼스, n : 다른 docu에서 각각 가져온 두 문장, plm : 같은 비율로 샘플링  
**nsp는 원래 NLI 성능 향상 용임  


# 2.4 Optimization  
*최적화  
**Adam 씀  
**beta1 = 0.9, beta = 0.999, eta = 1e-6, L2 weight = 0.01, l.r 은 10,000스텝마다 피크 1e-4 마다 linear decay  
**dropout = 0.1, GELU 활성함수 씀  
**S=1,000,000 업데이트로 프리트레인  
**미니배치 B = 256 seq서 max length T = 512 token  


# 2.5 Data  
BOOK CORPUS + ENG WIKI 총 16GB text  


# 3 Experimental Setup  
*버트 실험 셋업  


# 3.1 Implementation  
*구현  
**FAIRSEQ의 BERT 재구현  
**section2 대로 최적화  
**beta2 = 0.98 안정성 향상  
**T = 512 tokens  
**작은 랜덤 픽 sent x  
**seq 길이 줄여서 학습 x(full sent로 학습)  
**DGX-1 머신 8x32GB Nvidia V100 GPU  


# 3.2 Data  
*BERT data 늘어날 경우 성능 올라감  
**단, data 질 괜찮아야 함  
**data 모음 160GB 사용  


**BOOK 코퍼스 & Eng 위키-오리지널 BERT용 160Gb    
**CC-NEWS - 일반 크롤링 데이터 76Gb    
**open web text - 오픈소스 생성 redit url 사용 38Gb  
**stories - story style winogrod 스키마 사용 31Gb  


# 3.3 Evaluation  
*3개 벤치마크 평가 진행  


*GLUE(General Language Understanding Evaluation)  
**9개 셋 for NLI  
**단일 문장분류 or 문장 쌍 분류로 구성  
**학습, 개발 data 분리됨  
**section 4 에서 개발 data 결과와 파인튜닝(bert 따름)  결과 보여줌  
**section 5에서 public leader board 결과 보여줌  


*SQuAD(Standford Question Answering dataset)  
**문단(먼택스트의) 질문으로 구성  
**문제에 대해 답하는, 추론하는 task  
**v1.1은 답 있고 버트는 답 길이 따름  
**v2.0은 일부 답이 없고, 추가 이진분류로 답 가능한지부터 체크, 같이 train  


*RACE(ReAding Comprehension from Examination)  
**큰 RC dataset(28,000 문단, 100,000 문장, 중국의 영어시험(중/고생용))  
**사지선답  
**문장, 문맥 다 김  


# 4 Training Procedure Analysis  
*어떤 statement 선택이 중한지 실험  


# 4.1 Static vs. Dynamic Masking  
*Static Masking  
**10배 복제, 10개 다른 마스킹 140 epoch마다(기존보다 중복 마스킹 줄어듦)  
*dynamic  
**매번 다르게 masking  
*결과  
**dynamic masking 우위  


# 4.2 Model Input Format and Next Sentence Prediction  
*보조자 NSP test  
**필요성에 대한 찬반 있음  


*SEG-PAIR + NSP  
**오리지널 BERT 구조  
*SENT-PAIR + NSP  
**batch size 증가  
*full sent  
**no nsp  
*doc sent  
**input sent similarity  
**no nsp   
**batch size up  
*결과  
**single sent 성능 떨어짐  
**NSP 성능 떨어짐  
**full sent 성능 올라감  
**full doc 성능 올라감  



# 4.3 Training with large batches  
*큰 batch train  
**큰 batch optimize  
***speed and optimization 증대  
***성능 증대  
**배치사이즈 증가는 pp향상과 성능 증대  
**너무 클 경우는 실험 남겨둠  


# 4.4 Text Encoding  
*버트 텍스트 인코딩  
**기존 캐릭터단위 vocab 30K  
**여기선 trainable로 subword 50K  
**초기 실험서 성능 떨어진다고 알려져 있음  
**그러나 여기선 사용  
**관련 실험은 남겨둠  


# 5 RoBERTa   
앞서 언급한 것들 다 반영한 모델  
**dynamic mask  
**FULL sent  
**NSP x   
**large mini batch  
**large byte BPE  
**pretrain data up  
**train up  
**비교 위해 기존 버트도 test  
*결과 SOTA  


# 5.1 GLUE Results  
*GLUE에서 다음의 경우들 고려하여 파인튜닝 세팅 후 실험  
**batch size 16, 32  
**learning rate 1e-5, 2e-5, 3e-5  
**warm up 6& step  
**10epoch  
**early stoping  


*아래 또 고려하여 테스트  
**leader board 비교  
**RTE, STS, MRPC style task train 더 나음  


*end task 맞춰 modification  
**QNLI - pairwise 랭킹 사용  
***train set서 answer mining, compare pair -> classification  
**WNLI  
***SuperGLUE format에 맞춤  
***margin ranking loss로 파인튜닝  
***spaCy로 candidate noun 추출  
*결과  
**버트, XLNet 대비 9개 GLUE서 SOTA  
**GLUE리더보드 중 4개서 SOTA  


# 5.2 SQuAD Results  
*SQuAD 실험  
**버트, XLNet 들은 추가 데이터 사용  
**RoBERTa는 SQuAD 데이터만 사용  
**러닝레이트는 XLNet과 같게 함  
*결과  
**v1.1서 SOTA인 XLNet과 같은 결과  
**v2.0서 new SOTA  
**data 중복 없는 것들 중 리더보드 최상 결과  


# 5.3 RACE Results  
*RACE 결과  
**CLS 토큰 classify로 사용  
**SOTA   


# 6 Related Work   
*관련연구  
**목적(학습) 다름  
**LM, MT, MLM  
**파인튜닝 트렌드 따름  
**그러나 최근 multitask 파인튜닝, entity embeding 사용, span 예측, auto regressive 프리트레인 등으로 성능 올림  
**본 논문은 BERT replication, simplify로 성능 향상 목적(구조 이해 바탕으로)   


# 7 Conclusion  
*디자인 중요(버트)  
**모델 longer, batch bigger, NSP no, long sequence training, dynamic masking -> RoBERTa  
**SOTA in GLUE, RACE, SQuAD without 멀티테스크 파인튜닝 모델들 중  
**디자인 구조 중요성 알려줌(버트 프리트레인에서)  




