---
layout: post
title:  "[2018-2022]Papers related to BioNLP topics"
date:   2023-04-05 04:13:23 +0900
categories: study
---



[Paper with my notes]()  

<br/>

# Protein BERT + Antibiotic Resistence Genes  
* 시행착오   
** 파인튜닝이면 만사 ok라고 생각  
*** 코드문제가 아니었던 것 같음  
*** 데이터가 너무 편향적이고 카테고리는 14개로 많은편이며, 데이터 개수가 적음  
*** 계속 같은 예측값만 뱉어내는 것을 보아 overfiting 됨  
** 데이터 균형 잡히게 하는 방안 고민  
*** 언더샘플링 / 오버샘플링 생각  
*** 언더샘플링은 효과 없었음  
** 그냥 프로틴버트에서 임베딩 뽑아내서 이 데이터로 ml기법들(dl포함) 사용해야하나 생각이 듦  
*** 예전에 했던 광고 카테고리 분류처럼 여러 테스트 해봐야하나? 일단 자야겠음 근데..
** 임베딩 벡터를 잘 뽑아와서 ML기법 적용해보기로 함  
*** 임베딩 평균으로 센텐스임베딩 처럼 사용하여 랜덤포레스트 모델에서 학습 -> acc0.5  
**** 근데 이 정확도마저 재현이 안됨.....  
*** 임베딩 컨캣해서 길게 이어붙임, 이것으로 랜덤포레스트 모델 학습 -> acc 0.3
*** 바로 위의 임베딩 중 길이 제한 300, 100으로 줌 이걸로 랜덤포레스트 -> acc 0.3  
*** 각종 앙상블 모델들 다 30~40 언저리... mlp도 그 언저리..  
** official한 github에서 제공하는 코드 사용  
*** 제공하는 feature extraction을 통해 얻은 임베딩으로 위 과정 반복해보니 0.5~0.6  
** 파인튜닝 시도  
*** 멀티태스크러닝으로?  
*** 제공 멀티태스크 코드 분석해서 단일 테스크로 파인튜닝  
*** epoch3시 55%, epoch20까지 늘려서 57% 정확도  
*** weighted precision/weightedd recall도 비슷한 수준, confusion matrix visual추가  

<br/>

# [2022]Fine-tuning of BERT Model to Accurately Predict Drug-Target Interactions  


{% highlight ruby %}
짧은 요약 :  

* 최적 약 후보 식별 중요  
** 여기에 ML도입(DTI에)  
** Protein 모델 도입(ChemBERT화학물질, ProtBERT담백질 염기서열)  
*** 사전학습된 모델로 데이터: BIOSNAP, DAVIS&BINDING DB  
** ChemBERT + ProtBERT + DBs(fine-tuning)으로 BEST 성능  
*** BERT들의 [CLS] 토큰을 concat하여 classifying한다는 점이 특이  

{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  

<br/>

# [2022]Structure-Aware Antibiotic Resistance Classification Using Graph Neural Network  

{% highlight ruby %}
짧은 요약 :  

* 이전: 기존 seq와 비교  
* 단점: 늘 seq가 유사한 것만은 아님  
* DL기반 분류(ARG)  
** 2steps:  
(1) AlphaFold Model로 3D구조 predict  
(2) seq transformer 기반 protein LM 사용 & graph NN으로 graph 추출  
** SOTA 달성  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  

<br/>


# [2022]ProteinBERT: a universal deep-learning model of protein sequence and function  


{% highlight ruby %}
짧은 요약 :  

* self-supervised deep LM 대성공 in NLP  
** bio seq에 적용: ProtBERT with Gene Ontology Annotation Prediction  
** local/global 다 커버  
** SOTA 근접 또는 효과를 보임  
(Protein structure, post ranslation modification, biophysical attribute)  



{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  

<br/>


   
# [2018]DeepARG: a deep learning approach for predicting antibiotic resistance genes from metagenomic data  


{% highlight ruby %}
짧은 요약 :  

* AR 증가  
** 모니터링 기법 진보, 새 ARG 식별 필요  
** hit 기반은 false negative가 많음  
** DL 기법 제안  
** 결과 precision >0.97 / recall > 0.90  
** SOTA  k 

{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  

<br/>

