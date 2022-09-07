---
layout: post
title:  "SimCSE_Simple Contrastive Learning of Sentence Embeddings"
date:   2022-09-07 17:01:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

SimpleCSE - 간단한 contrast learning으로 SOTA 문장 임베딩 만듬  

비지도학습 - input:같은문장, output:positive or negative  
    -dropout 사용해서 성능 SOTA  

지도학습 - NLI(자연어추론) 데이터 사용  
    -entailment->positive, contradiction->hard negative  

평가 - STS(Semantic Textual Similarity)  
    -4.2%, 3.3%향상  

anisotopic space서 더 align 함(더 uniform함)  




{% endhighlight %}


링크(https://drive.google.com/drive/folders/1iPARz7CTetOHOmxm79xaOmeoyn7ri62e?usp=sharing)


# 단어정리  
*on par with: 같은 수준의, anisotropic: 이방성(비등방성), flatten: 단조롭게 하다(평평하게), alignment: 정렬? 가지런하게 있는? 멀리 있는? 안 붙은??, asymptotics: 점근적인, ablation: 제거, ablation study: feature 제거하며 영향/성능 분석, back-translation: 기존 훈련 반대방향으로 번역기 이용하여 번역(양방향 훈련용)  


# 1. Introduction  
*범용 문장 임베딩 학습 매력적  
*대조학습 with PLM(BERT or RoBERTa) 효과적  
*비지도 SimCSE  
**같은 문장 with 다른 dropout 페어로 input(positive)  
**다른 문장과 pair로 negative  
*NSP, discrete 데이터 증식보다 성능 뛰어남  
**dropout이 데이터 증식 역할 했기 때문  


*SimCSE  
**NLI 씀  
**기존 3분류 대신 entailment, contradiction 사용(neutral 제외)   
**성능 이전보다 향상  


*alignment(의미 연관 두 pair 사이) and uniformity(학습임베딩들 동일한지) 분석(성능척도)  
**비지도 SimCSE는 uniformity 향상  
***드롭아웃 노이즈 덕분에 alignment 비생산 방지  
**NLI 지도학습은 alignment 향상  
***positive pair 사이에서.. 더 잘 임베딩 만듦  


*PLM임베딩 anistropy에 빠짐  
**이유? 스펙트럼 보니..
**본 모델 CL 통해 단일 분포에 flatten해서 성능 더 좋음  
*척도2: STS and 7가지 transfer task  
**비지도, 지도 각각 4.2%, 2.2% 향상  
*문학서 비일치는 future work  


# 2 Background: Contrastive Learning  
*Contrastive learning  
**의미 유사 가까이, 먼 경우는 멀리 배치  
**BERT나 RoBERTa PLM 임베딩 사용  
**CL로 파인튜닝  
**Xi, Xi+ pair 만들기 -> 간단 dropout 적용, 성능 좋음  


*유사 CL 목적함수 dual 인코더임  
**Xi와 Xi+ 다르므로  


*CL 주요속성  
**alignment - 두 임베딩 사이 기대거리   
**uniformity - 균등 분포하는지  



# 3 Unsupervised SimCSE  

*같은 문장 + dropout으로 학습  
*dropout noise 추가가 성능 굿  
*next setence objective 보다 성능 좋음  
*fixed dropout 많아질수록 성능 떨어짐  
*modify 모델들 uniformity 좋고 alignment 낮아짐  
*단어 삭제의 경우 alignment 늘어남  
*dropout 보다 안 좋음  


# 4 Supervised SimCSE  
*dropout noise추가가 효과적이고 -> good alignment 만들어주기 때문임  
*NLI datset으로 성능(alignment) 향상 해본 것  
*라벨 데이터 선정 후보는 아래와 같음  
**QQP - Quora Question Pairs  
**Pickr30K - 이미지/사람이 쓴 5개의 관련 문장  
**ParaNMT - 파라프레이징  
**NLI - SNLI + MNLI  
***성능 제일 좋았는데, 고품질이고 크라우드 소싱이기 때문  
***또한 오버랩도 39%로 가장 적엇음(두 문장 사이)  
*NLI의 contradiction 추가 사용 시 성능 올라감  
**반대셋 사람이 만든 것인데 괜춘하기 때문  
*ANLI 또는 unsupervised SimCSE 같이 사용하는 것은 도움이 되진 않았음  


# 5 Connection to Anisotropy  
*단어임베딩 비등방성 문제 있음  
*해결책?  
**주된 컴포넌트 삭제  
**등방성 분포와 매핑  
**제약추가(학습 시)  


*비등방성 문제는 uniformity와 연관  
*CL은 uniformity 오르고 비등방성은 낮아짐  
*점근적 CL 목적함수(네거티브->무한대일때)의 범위가 성립됨  
**?  
*결론은 CL이 degeneration prob 경감시켜주고 문장임베딩의 uniformity 증가시킴  


# 6 Experiment  
# 6.1 Evaluation Setup  
*7STS task로 평가(비지도학습)(STS학습셋을 사용하진 않음)    
*지도용 7 transfer 학습  


*STS 디테일  
**STS12-16  
**SICK 연관도  
**추가 회귀분류 여부  
**스피어맨 vs 피어슨 코릴레이션  
**결과 합성 여부  
**Appendix 참조  


*학습디테일  
**버트 or 로버타  
**[cls] 토큰 -> 센텐스 임베딩  
**영어 위키피디아 랜덤샘플 문장을 비지도 학습으로 ->SimCSE  
**지도학습으로 MNLI and SNLI  



# 6.2 Main Results  
*비지도 STS 평가  
**Glove 임베딩  
**BERT and RoBERTa  
*후처리  
**BERT-flow  
**BERT-whitening  
*CL  
**IS-BERT - 글로벌/로컬 피처 agreement max  
**DECLUTR - 길이 조절(같은 다큐먼트의) pair  
*CT  
**다른 인코더(같은 문장)(후처리 포함)  
*지도  
**InferSent  
**UniversalSentenceEncoder  
**SBERT/SRoBERTa  
*SimCSE가 SOTA in 7STS  
*Trnasfer task도 SimCSE SOTA  


# 6.3 Ablation Studies  
*풀링메서드 and 네거티브셋 영향 and ablation study(피처 제거해보며 탐구)  
*PLM 평균임베딩 성능이 [cls]토큰 사용보다 좋았음  
*SimCSE 에선 [cls]와 MLP durint training이 성능 좋았음  


*Hard negative 추가시 성능 좋을 것으로 예상됨  
*가중치 주고 실험한 경우 alpha가 1일 때 성능 좋음  
*중립은 의미 없었음  


# 7 Analysis  
*SimCSE 내부 이해  
*uniformity and aligment 오르면 성능 오름  
*PL 임베딩  
**alignment 좋고 uniformity 안 좋음  
*후처리(버트플로우, 와이트닝)  
**uniformity 좋고 alignment 낮음  
*비지도 SimCSE  
**uniformity 높고 alignment 유지  
*지도 data  
**alignment 높여줌  
*SimCSE  
**similarity 높여줌  


*검색실험  
**SBERT and SimCSE BERT  
***SimCSE 가 SOTA  


# 8 Related Work  
*word2vec- 초기 센텐스 임베딩 균등 분할 가설(주변-문장 예측) 이용  
*요즘 CL(다양 관점에서)  
**데이터 augementation  
**다른 모델 카피해서(같은 다큐 센텐스 사용하되)  


*지도 문장 임베딩이 비지도 보다 강력(NLI서)  
*이중언어 and 백번역에서 성능 입증됨  
**대표성 상실 문제 경감시키고 PLM보다 성능 올림    



# 9 Conclusion  
*SimCSE CL 이용, PLM SOTA in STS(NLI 비지도 dropout noise 이용)  
*alignment and uniformity로 SOTA 이유 분석  
*비지도 용용성이 높을 것으로 예상됨  
**data 증폭  
**연속 LM, 임베딩 사용 예상  