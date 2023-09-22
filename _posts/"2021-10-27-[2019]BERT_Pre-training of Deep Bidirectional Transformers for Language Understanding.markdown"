---
layout: post
title:  "[2019]BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
date:   2021-10-27 15:10:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

유명한 버트 소개.
진정한 bidirectional representations pretraining
fine tuning으로 SOTA 찍음
(QA, LI 등)

{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1F202cyZu4fuCeUOIGzoz7J-sFCN3zLN0?usp=sharing)


-pretrained LM은 2가지가 있음
1.feature-based (ELMO 처럼 추가 특징 학습)
2.fine-tuning (GPT처럼 모든 파라미터 미세조정)

-근데 다 한방향임 LR. ELMO가 양방향이나 LR, RL 따로 학습 후 concat한 것에 불과.
-MLM(Masked Language Model) 통해 BERT는 양방향성 획득.
-NSP(Next Sentence Prediction) 통해 문장 간의 관계도 학습(QA와 NLI에 효과적)
-연속된 문장 한 묶음으로 input에 넣음
-[cls] 토큰으로 classification 용으로 사용
-[SEP] 토큰으로 두 문장 구분
-임베딩 레이어 추가하여 두 문장 달리 봄. (문장 1은 AAA 2는 BBB 이런 식)
-[MASK]를 전체의 15% 골라서 이 중 80%에 씌워서 MLM 학습하는 것. 비율 조정은 [MASK] 땜에 pretraining과 fine tuning 사이 괴리 생기는 거 조금 줄여보려고 한 것.
-corss entropy loss로 학습
-fine tuning은 기존의 encode 후 bidirectional cross attention 과정을 한방으로 합친 셈으로, encode with self-attention.
-BERT가 GLUE(General Language Understanding Evaluation), SQuAD(Stanford Question Answering Dataset), SWAG(Situations With Adversarial Generations) 등에서 SOTA임

