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



