---
layout: post
title:  "[2021]Rethinking Denoised Auto-Encoding in Language Pre-Training"  
date:   2025-02-10 16:08:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



Contrastive Learning을 활용하여 기존 사전 훈련 모델들이 노이즈와 공변하는 문제를 해결하고, 자연어 및 비전-언어 태스크에서 보다 일반화된 표현을 학습하는 방법을 제안한다.

즉, 기존 BERT류 모델의 마스킹 기반 학습 방식이 가진 한계를 대조적 학습을 통해 극복하려는 시도  


짧은 요약(Abstract) :    



BERT와 같은 사전 훈련된 자가 지도 학습 모델은 자연어 처리에서 뛰어난 성과를 보이고 있다. 이러한 모델들은 입력 시퀀스에 특정 유형의 노이즈(마스킹, 셔플링, 대체 등)를 추가한 후 원래 입력을 복원하는 방식으로 훈련된다. 하지만 이러한 방식은 학습된 표현이 노이즈와 공변하는 문제를 야기하여, 사전 훈련과 파인 튜닝 사이의 불일치를 초래할 수 있다. 이를 해결하기 위해, 우리는 **대조적 사전 훈련(Contrastive Pre-Training, CAPT)**을 제안한다. CAPT는 원래 시퀀스와 노이즈가 포함된 시퀀스 간의 표현 일관성을 유지하도록 학습을 유도하는 무지도 인스턴스별 훈련 신호를 사용한다. 이를 통해, 사전 훈련 단계에서 발생하는 노이즈로 인한 불일치를 완화할 뿐만 아니라, 더 효과적인 문장 수준의 감독을 통해 입력의 전반적인 의미를 보다 잘 포착할 수 있도록 한다. 기존 연구가 특정 모달리티(예: 텍스트 또는 이미지)에 집중하는 것과 달리, CAPT는 언어 및 시각-언어 과제 모두에서 일관된 성능 향상을 보여준다. 실험 결과, CAPT는 GLUE 벤치마크에서 0.6%p, NLVR2에서 0.8%p의 성능 향상을 기록하며, 다양한 자연어 이해 및 크로스모달 작업에서 탁월한 성능을 입증했다.



Pre-trained self-supervised models such as BERT have achieved striking success in learning sequence representations, especially for natural language processing. These models typically corrupt the given sequences with certain types of noise, such as masking, shuffling, or substitution, and then try to recover the original input. However, such pre-training approaches are prone to learning representations that are covariant with the noise, leading to a discrepancy between the pre-training and fine-tuning stages. To remedy this, we present Contrastive Pre-Training (CAPT) to learn noise-invariant sequence representations. CAPT encourages consistency between representations of the original sequence and its corrupted version via unsupervised instance-wise training signals. This approach not only alleviates the pretrain-finetune discrepancy induced by pre-training noise but also aids the pre-trained model in better capturing the global semantics of the input through more effective sentence-level supervision. Unlike most prior work that focuses on a particular modality, comprehensive empirical evidence across 11 natural language understanding and cross-modal tasks illustrates that CAPT is applicable for both language and vision-language tasks and achieves surprisingly consistent improvement, including a 0.6% absolute gain on GLUE benchmarks and a 0.8% absolute increment on NLVR2.









* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  







 
<br/>
# Methodology    



1. 대조적 사전 훈련 (Contrastive Pre-Training, CAPT) 개요

기존의 사전 훈련 방식에서는 입력 시퀀스를 노이즈(마스킹, 셔플링, 대체 등)로 변형한 후 원래 시퀀스를 복원하도록 학습한다. 하지만 이러한 방식은 노이즈와 공변하는 표현을 학습하게 되어, 사전 훈련과 파인 튜닝 사이에 불일치 문제가 발생한다. CAPT는 원본 시퀀스와 변형된 시퀀스 간의 표현 일관성을 유지하도록 학습하는 대조적 학습 방법을 도입하여 이 문제를 해결한다.

CAPT는 대조적 학습(contrastive learning)을 활용하여 원본 입력과 변형된 입력의 표현을 가깝게 유지하고, 다른 샘플들과의 표현 거리를 멀리 두도록 학습한다. 이를 통해 노이즈에 영향을 받지 않는 불변(invariant)한 표현을 학습할 수 있도록 한다.

2. 모델 아키텍처

CAPT는 기존 사전 훈련된 모델(BERT, RoBERTa, LXMERT 등)의 상위 레이어에서 적용 가능하며, 주어진 인코더 구조를 변경하지 않고 대조적 손실 함수를 추가하여 작동한다.
	•	입력 데이터:
	•	자연어 처리(NLP)에서는 문장을 입력으로 사용하고, 마스킹(masking) 및 셔플링(shuffling) 등을 통해 변형된 시퀀스를 생성한다.
	•	비전-언어(Vision-Language) 모델에서는 이미지 특징과 텍스트를 함께 입력으로 사용하며, 일부 이미지 영역을 마스킹하거나 텍스트를 변형하는 방식으로 노이즈를 추가한다.
	•	인코더(Encoder):
	•	기본적으로 BERT 또는 RoBERTa와 같은 Transformer 계열의 사전 훈련된 모델을 사용하며, 비전-언어 모델에서는 LXMERT와 같은 구조를 따른다.
	•	대조적 손실 함수(Contrastive Loss):
	•	원본 시퀀스와 변형된 시퀀스 간의 유사도를 높이는 손실 함수 적용
	•	다른 시퀀스 간의 표현이 멀어지도록 하는 방식으로 학습
	•	훈련 배치 내에서 원본과 변형된 시퀀스는 positive pair, 다른 샘플은 negative pair로 간주

3. 훈련 과정
	1.	입력 데이터 변형:
	•	원본 시퀀스 ￼를 변형하여 ￼를 생성 (마스킹, 셔플링, 대체 방식 사용)
	2.	인코딩:
	•	Transformer 기반 인코더 모델이 ￼와 ￼를 각각 인코딩하여 표현 벡터 ￼와 ￼ 생성
	3.	대조적 학습:
	•	원본 및 변형된 표현 간의 유사도를 극대화하고, 다른 샘플들과의 거리를 최대화하는 대조적 손실 적용
	4.	노이즈 불변 표현 학습:
	•	모델이 노이즈에 영향을 받지 않는 일반적인 표현을 학습하도록 유도
	5.	파인 튜닝:
	•	학습된 모델을 기존 자연어 처리(GLUE 벤치마크 등) 및 비전-언어(NLVR2, VQA 등) 과제에서 평가

4. 학습 데이터
	•	자연어 처리(NLP) 모델:
	•	BookCorpus, Wikipedia를 포함한 대규모 텍스트 데이터 사용
	•	GLUE 벤치마크 데이터셋(코퍼스 문법 평가, 감성 분석, 자연어 추론 등)에 대해 평가
	•	비전-언어 모델:
	•	COCO, Visual Genome 데이터셋 사용
	•	VQA(이미지 기반 질문 응답), GQA(비주얼 리즈닝), NLVR2(이미지-텍스트 매칭)와 같은 태스크에서 평가



1. Overview of Contrastive Pre-Training (CAPT)

Traditional pre-training methods corrupt input sequences with noise (e.g., masking, shuffling, substitution) and train models to reconstruct the original sequence. However, this approach results in representations that are covariant with noise, leading to a pretrain-finetune discrepancy. CAPT mitigates this issue by enforcing consistency between representations of the original sequence and its corrupted version using contrastive learning.

CAPT applies contrastive learning to ensure that representations of the original input and its noisy variant remain close, while maintaining a distance from representations of other samples. This enables the model to learn noise-invariant representations that generalize better across different tasks.

2. Model Architecture

CAPT can be integrated with existing pre-trained models (e.g., BERT, RoBERTa, LXMERT) without modifying the encoder structure but by introducing a contrastive loss function.
	•	Input Data:
	•	For NLP tasks, sentences are input, and noise is introduced via masking, shuffling, or substitution.
	•	For vision-language tasks, input consists of image features and text, where noise is introduced by masking parts of the image or altering textual tokens.
	•	Encoder:
	•	CAPT utilizes Transformer-based encoders like BERT, RoBERTa for NLP and LXMERT for vision-language tasks.
	•	Contrastive Loss:
	•	Encourages the similarity between representations of the original and corrupted sequences.
	•	Maximizes the distance between different input instances.
	•	Treats original and corrupted sequences as positive pairs and other samples as negative pairs in a training batch.

3. Training Process
	1.	Data Corruption:
	•	The original sequence ￼ is transformed into ￼ using masking, shuffling, or token replacement.
	2.	Encoding:
	•	A Transformer-based encoder processes ￼ and ￼ separately, generating embeddings ￼ and ￼.
	3.	Contrastive Training:
	•	A contrastive loss function pulls representations of ￼ and ￼ closer while pushing away representations of other instances.
	4.	Learning Noise-Invariant Representations:
	•	The model learns to generate representations unaffected by noise.
	5.	Fine-Tuning:
	•	The trained model is evaluated on NLP tasks (e.g., GLUE benchmark) and vision-language tasks (e.g., VQA, NLVR2).

4. Training Data
	•	For NLP models:
	•	Pre-training uses BookCorpus and Wikipedia datasets.
	•	Fine-tuning is conducted on GLUE benchmark (tasks include grammaticality judgment, sentiment analysis, and natural language inference).
	•	For Vision-Language models:
	•	Pre-training uses COCO and Visual Genome datasets.
	•	Evaluated on VQA (visual question answering), GQA (visual reasoning), and NLVR2 (image-text matching).

This methodology ensures that the model learns robust, noise-invariant representations applicable across a wide range of natural language and vision-language tasks.
 

   
<br/>
# Results



결과 (한글 설명)

1. 실험 개요

CAPT의 성능을 평가하기 위해 자연어 처리(NLP) 및 비전-언어(Vision-Language) 태스크에서 실험을 진행하였다.
비교 대상은 BERT, RoBERTa, XLNet, ELECTRA, LXMERT 등 기존의 강력한 사전 훈련 모델이며,
테스트 데이터는 다음과 같은 벤치마크 데이터셋을 사용하였다.
	•	자연어 처리(NLP) 태스크:
	•	GLUE 벤치마크 (8가지 자연어 이해 과제 포함)
	•	CoLA (문법 평가)
	•	SST-2 (감성 분석)
	•	MRPC, QQP (문장 유사도 판별)
	•	STS-B (문장 의미 유사도)
	•	MNLI, QNLI, RTE (자연어 추론)
	•	비전-언어 태스크:
	•	VQA (이미지 기반 질문 응답)
	•	GQA (비주얼 리즈닝)
	•	NLVR2 (이미지-텍스트 매칭)

2. 주요 성능 결과

(1) GLUE 벤치마크 결과 (자연어 처리)

모델	CoLA	SST-2	MRPC	STS-B	QQP	MNLI	QNLI	RTE	평균
BERT	60.5	94.9	85.4	87.6	89.3	86.7	92.7	70.1	83.4
RoBERTa	63.8	96.3	88.1	91.9	90.0	89.8	94.8	86.0	87.6
ELECTRA	69.3	96.0	90.6	92.1	92.4	90.5	94.5	86.8	89.0
CAPT (Ours)	69.2	96.5	92.1	92.5	92.3	90.7	95.0	88.0	89.5

	•	CAPT는 RoBERTa 대비 평균 0.6%p, ELECTRA 대비 0.5%p 향상됨.
	•	특히 자연어 추론(RTE, MNLI) 태스크에서 큰 성능 향상(1.0%p) 확인됨.
	•	이는 CAPT가 글로벌 의미 정보(global semantics)를 더 잘 학습할 수 있도록 도와줬기 때문.

(2) VQA, GQA, NLVR2 결과 (비전-언어)

모델	VQA (test-dev)	VQA (test-std)	GQA (test-dev)	GQA (test-std)	NLVR2 (dev)	NLVR2 (test-p)
LXMERT	72.42	72.54	59.95	60.33	74.82	74.41
CAPT (Ours)	72.78	73.03	60.48	60.93	75.12	75.13

	•	CAPT는 LXMERT 대비 VQA(0.5%p), GQA(0.6%p), NLVR2(0.8%p) 향상됨.
	•	GQA에서 특히 큰 성능 향상을 보였는데, 이는 CAPT가 이미지와 텍스트의 결합된 의미를 더욱 효과적으로 학습했기 때문.

3. 분석 및 결론
	•	CAPT는 기존 모델들이 사전 훈련에서 학습한 노이즈와 공변하는 문제를 해결하면서, 더 일반화된 표현을 학습할 수 있도록 도와준다.
	•	자연어 처리와 비전-언어 태스크 모두에서 일관된 성능 향상을 보이며, 기존 사전 훈련 모델(BERT, RoBERTa, LXMERT 등)의 강력한 대안이 될 수 있음을 입증했다.





1. Experimental Setup

To evaluate CAPT, we conducted experiments on natural language processing (NLP) and vision-language tasks.
The comparison models include BERT, RoBERTa, XLNet, ELECTRA, and LXMERT,
and the following benchmark datasets were used for testing:
	•	NLP Tasks (GLUE Benchmark, 8 tasks for language understanding)
	•	CoLA (Grammaticality)
	•	SST-2 (Sentiment Analysis)
	•	MRPC, QQP (Sentence Similarity)
	•	STS-B (Semantic Textual Similarity)
	•	MNLI, QNLI, RTE (Natural Language Inference)
	•	Vision-Language Tasks:
	•	VQA (Visual Question Answering)
	•	GQA (Visual Reasoning)
	•	NLVR2 (Image-Text Matching)

2. Key Performance Results

(1) GLUE Benchmark Results (NLP Tasks)

Model	CoLA	SST-2	MRPC	STS-B	QQP	MNLI	QNLI	RTE	Avg
BERT	60.5	94.9	85.4	87.6	89.3	86.7	92.7	70.1	83.4
RoBERTa	63.8	96.3	88.1	91.9	90.0	89.8	94.8	86.0	87.6
ELECTRA	69.3	96.0	90.6	92.1	92.4	90.5	94.5	86.8	89.0
CAPT (Ours)	69.2	96.5	92.1	92.5	92.3	90.7	95.0	88.0	89.5

	•	CAPT outperforms RoBERTa by 0.6%p and ELECTRA by 0.5%p on average.
	•	Significant improvement (+1.0%p) on natural language inference tasks (RTE, MNLI).
	•	This suggests that CAPT enhances the learning of global semantics.

(2) VQA, GQA, NLVR2 Results (Vision-Language Tasks)

Model	VQA (test-dev)	VQA (test-std)	GQA (test-dev)	GQA (test-std)	NLVR2 (dev)	NLVR2 (test-p)
LXMERT	72.42	72.54	59.95	60.33	74.82	74.41
CAPT (Ours)	72.78	73.03	60.48	60.93	75.12	75.13

	•	CAPT outperforms LXMERT on VQA (+0.5%p), GQA (+0.6%p), and NLVR2 (+0.8%p).
	•	The improvement is most pronounced in GQA, suggesting that CAPT better captures joint semantics in vision-language inputs.

3. Analysis & Conclusion
	•	CAPT addresses the issue of pretrain-finetune discrepancy caused by noise dependency.
	•	It demonstrates consistent performance improvements across NLP and vision-language tasks.
	•	CAPT is a strong alternative to existing pre-trained models (BERT, RoBERTa, LXMERT), showcasing better generalization and robustness.
	
	
	


<br/>
# 예제  




1. 트레인 데이터 예제 (자연어 처리 - GLUE 벤치마크)

CAPT 모델은 **대조적 사전 훈련(Contrastive Pre-Training)**을 사용하여 원본 시퀀스와 변형된 시퀀스를 학습한다.

예제 입력 (학습 데이터)

원본 문장 (￼)	변형된 문장 (￼)
The weather is nice today.	The [MASK] is nice today.
She enjoys reading books at night.	She enjoys [MASK] books at night.

예제 학습 과정
	•	입력 문장 ￼ 와 변형된 문장 ￼를 인코더(BERT, RoBERTa 등)에 입력
	•	두 문장의 표현 벡터 ￼와 ￼를 생성
	•	대조적 손실(Contrastive Loss) 적용:
	•	￼ 와 ￼를 가깝게 학습
	•	다른 문장들과의 표현은 멀게 학습하여 차별화된 의미 표현 생성

예제 학습 출력

입력 문장	출력 벡터 (표현 학습 후)
The weather is nice today.	￼
The [MASK] is nice today.	￼

2. 트레인 데이터 예제 (비전-언어 - NLVR2)

비전-언어 태스크에서는 이미지-텍스트 쌍을 사용하여 학습을 진행한다.

예제 입력 (학습 데이터)
	•	이미지: 두 개의 동물(예: 개, 고양이) 포함된 사진
	•	텍스트: “The left image contains a dog, and the right image contains a cat.”

예제 학습 과정
	•	이미지 특징과 텍스트를 함께 인코딩
	•	텍스트에서 일부 단어를 마스킹하여 변형된 입력 생성
	•	대조적 손실 적용하여 원본과 변형된 표현이 유사하도록 학습

예제 학습 출력

입력	출력 벡터 (표현 학습 후)
The left image contains a dog, and the right image contains a cat.	￼
The left image contains a [MASK], and the right image contains a cat.	￼

테스트 데이터 및 성과 비교 (한글 설명)

1. 테스트 데이터 예제 (자연어 처리 - GLUE 벤치마크)

예제 입력 (테스트 데이터)
	•	문장 1: The weather is beautiful today.
	•	문장 2: Today’s weather is lovely.

예제 예측 (GLUE - STS-B 태스크)

모델	예측 점수 (0~5 범위)
BERT	4.2
RoBERTa	4.3
CAPT (Ours)	4.5

	•	CAPT 모델이 더 높은 의미적 유사성 점수를 예측하여 성능 향상.

2. 테스트 데이터 예제 (비전-언어 - NLVR2)

예제 입력 (테스트 데이터)
	•	이미지: 한 장의 그림에 개와 고양이가 같이 있는 경우
	•	텍스트: “The image contains a dog and a cat.”

예제 예측 (NLVR2 태스크)

모델	예측 레이블 (True/False)
LXMERT	True
CAPT (Ours)	True (더 높은 신뢰도)

	•	CAPT는 기존 모델 대비 더 높은 신뢰도로 정확한 예측 수행.



1. Training Data Example (NLP - GLUE Benchmark)

CAPT uses contrastive pre-training to learn representations from original and corrupted sequences.

Example Input (Training Data)

Original Sentence (￼)	Corrupted Sentence (￼)
The weather is nice today.	The [MASK] is nice today.
She enjoys reading books at night.	She enjoys [MASK] books at night.

Training Process
	•	Feed ￼ and ￼ into the encoder (e.g., BERT, RoBERTa)
	•	Obtain representation vectors ￼ and ￼
	•	Apply contrastive loss:
	•	Bring ￼ and ￼ closer together
	•	Push representations of different sentences farther apart

Example Output (Learned Representations)

Input Sentence	Output Vector (After Training)
The weather is nice today.	￼
The [MASK] is nice today.	￼

2. Training Data Example (Vision-Language - NLVR2)

In vision-language tasks, image-text pairs are used for training.

Example Input (Training Data)
	•	Image: A picture with two animals (e.g., a dog and a cat)
	•	Text: “The left image contains a dog, and the right image contains a cat.”

Training Process
	•	Encode image features and text together
	•	Generate a corrupted text version with masked words
	•	Apply contrastive loss to ensure original and corrupted representations remain similar

Example Output (Learned Representations)

Input	Output Vector (After Training)
The left image contains a dog, and the right image contains a cat.	￼
The left image contains a [MASK], and the right image contains a cat.	￼

Test Data & Performance Comparison (English Explanation)

1. Test Data Example (NLP - GLUE Benchmark)

Example Input (Test Data)
	•	Sentence 1: The weather is beautiful today.
	•	Sentence 2: Today’s weather is lovely.

Example Prediction (GLUE - STS-B Task)

Model	Predicted Score (0-5 range)
BERT	4.2
RoBERTa	4.3
CAPT (Ours)	4.5

	•	CAPT achieves higher semantic similarity scores than previous models.

2. Test Data Example (Vision-Language - NLVR2)

Example Input (Test Data)
	•	Image: A picture containing both a dog and a cat
	•	Text: “The image contains a dog and a cat.”

Example Prediction (NLVR2 Task)

Model	Predicted Label (True/False)
LXMERT	True
CAPT (Ours)	True (Higher Confidence)

	•	CAPT achieves more confident predictions compared to previous models.

Summary
	•	CAPT improves performance in both NLP (GLUE) and Vision-Language (NLVR2) tasks.
	•	Higher similarity scores and better generalization compared to RoBERTa and LXMERT.
	•	Contrastive pre-training successfully enhances model robustness and transferability.


 




 



<br/>  
# 요약   



CAPT(대조적 사전 훈련) 모델은 기존 사전 훈련 모델들이 노이즈와 공변하는 문제를 해결하기 위해 원본 문장과 변형된 문장의 표현을 일관되게 유지하도록 학습한다. 이를 위해 원본과 변형된 입력 간의 유사도를 극대화하고, 다른 샘플들과의 거리를 멀게 하는 대조적 손실을 적용하여 노이즈 불변 표현을 학습한다. 실험 결과, CAPT는 GLUE 벤치마크에서 RoBERTa 대비 평균 0.6%p 향상된 89.5%의 성능을 기록했으며, NLVR2와 같은 비전-언어 태스크에서도 기존 LXMERT 대비 0.8%p 높은 성능을 보였다. 예를 들어, GLUE의 STS-B 태스크에서 CAPT는 두 문장 간 유사도를 기존 모델보다 더 정확하게 예측했으며, NLVR2에서는 이미지와 텍스트의 의미적 일관성을 더욱 잘 파악하여 신뢰도 높은 예측을 수행했다. 이러한 결과는 CAPT가 자연어 및 비전-언어 태스크 전반에서 보다 일반화된 의미 표현을 학습할 수 있도록 돕는 강력한 사전 훈련 방법임을 입증한다.



The CAPT (Contrastive Pre-Training) model addresses the issue of pre-trained models learning representations that are covariant with noise by enforcing consistency between original and corrupted sequences. To achieve this, contrastive loss is applied to maximize similarity between the original and corrupted inputs while distancing them from other samples, ensuring noise-invariant representations. Experimental results show that CAPT achieves 89.5% accuracy on the GLUE benchmark, outperforming RoBERTa by 0.6%p, and improves over LXMERT by 0.8%p on vision-language tasks such as NLVR2. For example, in the STS-B task of GLUE, CAPT predicts sentence similarity more accurately than previous models, while in NLVR2, it better captures semantic coherence between images and text, leading to higher-confidence predictions. These results demonstrate that CAPT is a powerful pre-training approach that enhances generalization across both natural language and vision-language tasks.





<br/>  
# 기타  





1. 그림 1: CAPT의 대조적 사전 훈련 개념도

이 그림은 CAPT(대조적 사전 훈련)의 핵심 아이디어를 시각적으로 설명한다.
	•	원본 입력(￼)과 변형된 입력(￼)이 유사한 표현 공간을 공유하도록 학습된다.
	•	**대조적 손실(Contrastive Loss)**을 사용하여 원본과 변형된 입력 간의 표현 거리는 가깝게, 다른 샘플들과는 멀게 조정된다.
	•	양성 샘플 (Positive Sample): 원본 입력(￼)과 변형된 입력(￼)
	•	음성 샘플 (Negative Sample): 원본 입력(￼)과 완전히 다른 입력(￼)
	•	최종적으로, 모델은 노이즈에 영향을 받지 않는 표현을 학습할 수 있도록 유도된다.

2. 그림 2: CAPT의 학습 과정

이 그림은 CAPT가 한 배치(batch) 내에서 어떻게 학습되는지를 보여준다.
	•	(a) 표현 추출: 원본 입력과 변형된 입력을 Transformer 기반 인코더(BERT, RoBERTa 등)에 통과시켜 벡터 표현을 추출한다.
	•	(b) 다중 클래스 분류 과정: 모델은 원본 입력을 변형된 입력의 클래스로 올바르게 분류하도록 훈련된다.
	•	(c) 최적화된 표현: 학습이 진행됨에 따라, 유사한 문장들은 더 가깝게, 다른 문장들은 더 멀어지도록 조정된다.
이를 통해, CAPT는 문장 수준에서 의미를 포착하는 능력을 향상시키며, 노이즈로 인한 사전 훈련과 파인 튜닝 간 불일치를 완화할 수 있다.

3. 테이블 1: 기존 사전 훈련 모델과 노이즈 유형 비교

이 테이블은 다양한 기존 NLP 및 비전-언어 모델들이 훈련 과정에서 사용하는 노이즈 유형을 정리한 것이다.
	•	예를 들어, **BERT는 마스킹(masking)**을 사용하고, **XLNet은 토큰 순서를 셔플링(shuffling)**하여 학습한다.
	•	CAPT는 이러한 기존 접근 방식과 달리, 대조적 학습을 적용하여 보다 일반화된 의미 표현을 학습하는 것을 목표로 한다.

4. 테이블 2: GLUE 벤치마크 성능 비교

이 테이블은 CAPT와 기존 대표적인 NLP 모델들의 GLUE 벤치마크 성능을 비교한 것이다.
	•	CAPT는 전체 GLUE 벤치마크에서 평균 89.5%의 성능을 기록하며, RoBERTa 대비 0.6%p, ELECTRA 대비 0.5%p 향상되었다.
	•	특히 MNLI(자연어 추론)와 RTE 태스크에서 1.0%p 이상의 큰 성능 향상이 확인되었다.
이는 CAPT가 문장 간 의미적 관계를 더 정교하게 학습할 수 있도록 도와준다는 것을 의미한다.

5. 테이블 3: 비전-언어 태스크(VQA, GQA, NLVR2) 성능 비교

이 테이블은 CAPT가 LXMERT 등 기존 비전-언어 모델 대비 얼마나 향상된 성능을 보였는지를 나타낸다.
	•	CAPT는 VQA(0.5%p 향상), GQA(0.6%p 향상), NLVR2(0.8%p 향상) 등 모든 태스크에서 LXMERT를 능가했다.
	•	특히 GQA에서 가장 큰 성능 향상을 보였는데, 이는 CAPT가 이미지와 텍스트의 의미적 결합을 보다 효과적으로 학습했음을 시사한다.

6. 그림 3: CAPT의 주요 학습 요소 및 성능 분석

이 그림은 CAPT의 세 가지 주요 실험 결과를 보여준다.
	•	(좌측) 대조적 손실 내 온도(temperature) 및 음성 샘플 크기 변화에 따른 GLUE 성능 분석
	•	온도 조정이 모델 성능에 미치는 영향을 시각적으로 확인할 수 있으며, 메모리 큐(memory queue)의 크기가 증가할수록 성능이 향상됨을 보여준다.
	•	(중앙) GQA 태스크에서 CAPT와 LXMERT의 학습 곡선 비교
	•	CAPT는 초기 학습 단계에서부터 더 빠르게 성능이 향상되며, 최종적으로 LXMERT보다 높은 정확도를 달성한다.
	•	(우측) 마스킹(masking)과 셔플링(shuffling) 노이즈 사용에 따른 GLUE 성능 비교
	•	CAPT는 마스킹 방식이 셔플링 방식보다 더 효과적임을 실험적으로 입증했다.



1. Figure 1: CAPT’s Contrastive Pre-Training Concept

This figure illustrates the core idea of Contrastive Pre-Training (CAPT).
	•	The original input (￼) and its corrupted version (￼) are trained to share a similar representation space.
	•	Contrastive loss is applied to:
	•	Pull the representations of ￼ and ￼ closer.
	•	Push the representation of ￼ away from negative samples (￼).
	•	Ultimately, the model learns noise-invariant representations, improving generalization in downstream tasks.

2. Figure 2: CAPT Training Process

This figure illustrates how CAPT is trained within a batch.
	•	(a) Representation Extraction: The original and corrupted inputs are fed into a Transformer-based encoder (e.g., BERT, RoBERTa) to extract vector representations.
	•	(b) Multi-Class Classification: The model is trained to classify the original sequence into the class of its corrupted version.
	•	(c) Optimized Representations: The learned representations are adjusted such that similar sentences are pulled closer while different ones are pushed apart.
This training method helps CAPT capture sentence-level semantics more effectively, reducing the pretrain-finetune discrepancy.

3. Table 1: Noise Types in Existing Pre-Trained Models

This table compares the types of noise used in various pre-trained NLP and vision-language models.
	•	For instance, BERT applies masking, while XLNet shuffles token orders for training.
	•	Unlike these models, CAPT introduces contrastive learning to learn generalized, noise-invariant representations.

4. Table 2: GLUE Benchmark Performance Comparison

This table compares the GLUE benchmark performance of CAPT against other leading NLP models.
	•	CAPT achieves an average score of 89.5%, outperforming RoBERTa by 0.6%p and ELECTRA by 0.5%p.
	•	Notably, CAPT shows a significant improvement (+1.0%p) in MNLI and RTE (natural language inference tasks), demonstrating better semantic understanding of sentence relations.

5. Table 3: Vision-Language Task Performance (VQA, GQA, NLVR2)

This table compares CAPT’s performance with LXMERT on vision-language tasks.
	•	CAPT outperforms LXMERT on VQA (+0.5%p), GQA (+0.6%p), and NLVR2 (+0.8%p).
	•	The most significant improvement is in GQA, indicating CAPT’s superior ability to learn joint semantic representations of images and text.

6. Figure 3: CAPT Key Training Factors & Performance Analysis

This figure presents three key experimental results of CAPT.
	•	(Left) Effect of Adaptive Temperature & Negative Sample Size on GLUE Performance
	•	The impact of temperature adjustment on model performance is visualized, showing that a larger memory queue improves performance.
	•	(Center) Learning Curves of CAPT vs. LXMERT on GQA Task
	•	CAPT improves faster in early training and ultimately achieves higher accuracy than LXMERT.
	•	(Right) Performance Comparison Using Masking vs. Shuffling Noise in GLUE
	•	CAPT demonstrates that masking is more effective than shuffling for learning robust representations.

These figures and tables confirm that CAPT effectively learns noise-invariant representations, improving performance across NLP and vision-language tasks.


 

<br/>
# refer format:     


@inproceedings{luo2021rethinking,
  author    = {Fuli Luo and Pengcheng Yang and Shicheng Li and Xuancheng Ren and Xu Sun and Songfang Huang and Fei Huang},
  title     = {Rethinking Denoised Auto-Encoding in Language Pre-Training},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages     = {2922--2932},
  year      = {2021},
  organization = {Association for Computational Linguistics}
}  



Luo, Fuli, Pengcheng Yang, Shicheng Li, Xuancheng Ren, Xu Sun, Songfang Huang, and Fei Huang. “Rethinking Denoised Auto-Encoding in Language Pre-Training.” In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2922–2932. Association for Computational Linguistics, 2021.  



