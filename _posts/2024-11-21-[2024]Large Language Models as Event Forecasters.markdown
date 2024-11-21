---
layout: post
title:  "[2024]Large Language Models as Event Forecasters"  
date:   2024-11-21 08:30:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


gnn+rnn 대신 llm으로 이벤트 요소(object)나 다음 이벤트 분류(합성 또는 비판 label)  


짧은 요약(Abstract) :    




이 논문에서는 인간 이벤트의 주요 요소들을 주제, 관계, 객체, 그리고 타임스탬프를 포함하는 4요소 구조(quadruple)로 추출하고, 이를 텍스트 요약을 추가한 5요소 구조(quintuple)로 확장한 뒤, 특정 도메인 내에서 이를 시간적 지식 그래프(Temporal Knowledge Graph, TKG)로 조직화하는 방법을 다룹니다. 기존의 TKG 학습 프레임워크는 그래프 신경망(GNN)과 순환 신경망(RNN)과 같은 복잡한 모델을 사용해 중간 임베딩을 업데이트했지만, 이 접근법은 각각의 5요소 구조에 내재된 컨텍스트 정보를 제대로 활용하지 못했습니다. 이에 따라 저자들은 대형 언어 모델(LLM)을 사용하여 간소화된 TKG 학습 프레임워크를 제안했습니다.

구체적으로, 객체 예측(Object Prediction)과 다중 이벤트 예측(Multi-Event Forecasting)이라는 두 가지 주요 TKG 관련 작업을 표준 질문-답변(QA) 형식으로 재구성하여 LLM이 이를 효과적으로 수행하도록 템플릿을 설계했습니다. 이 접근법은 GNN과 RNN 없이도 높은 예측 정확도를 유지할 수 있도록 LLM과 셀프 어텐션 메커니즘을 결합하여 설계되었습니다. 여러 실제 데이터셋을 사용한 실험 결과, 제안된 방법론이 기존 프레임워크와 비교해 효과적이고 견고하다는 것이 입증되었습니다.



This paper explores how key elements of human events can be extracted as quadruples consisting of subject, relation, object, and timestamp, and extended into quintuples by adding a textual summary. These are organized into Temporal Knowledge Graphs (TKGs) for specific domains. Traditional TKG learning frameworks often rely on complex models such as Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) for intermediate embedding updates, but they fail to fully leverage the contextual information within each quintuple.

The authors propose a simplified TKG learning framework using Large Language Models (LLMs). They restructure key TKG tasks, such as Object Prediction (OP) and Multi-Event Forecasting (MEF), into standard Question-Answering (QA) formats with tailored prompt templates. This novel framework eliminates the need for GNNs and RNNs, instead combining LLMs and self-attention mechanisms. Extensive experiments on real-world datasets demonstrate the effectiveness and robustness of this approach compared to traditional methods.




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



이 논문에서는 대형 언어 모델(LLMs)을 활용한 새로운 TKG 학습 프레임워크인 LEAF를 제안합니다. 기존의 TKG 학습 방식은 그래프 신경망(GNN)과 순환 신경망(RNN)을 사용하여 시간적 데이터와 구조적 데이터를 처리했습니다. 그러나 이러한 방식은 5요소 구조(주제, 관계, 객체, 타임스탬프, 텍스트 요약)에 포함된 컨텍스트 정보를 제대로 활용하지 못했습니다.

LEAF 프레임워크의 방법론:
	1.	객체 예측(Object Prediction):
	•	순위 예측 방식: RoBERTa와 같은 인코더-기반 LLM을 활용하여 5요소 구조에서 텍스트 요약을 문장 임베딩으로 변환하고, 이를 구조적 디코더와 결합하여 객체를 예측합니다.
	•	생성 방식: FLAN-T5와 같은 인코더-디코더 기반 LLM을 질문-답변(QA) 형식으로 학습시켜 객체를 직접 생성하도록 설계했습니다.
	2.	다중 이벤트 예측(Multi-Event Forecasting):
	•	5요소 구조를 기반으로 간단한 프롬프트 템플릿을 설계하고, RoBERTa와 같은 인코더-기반 LLM을 사용하여 각 5요소 구조의 임베딩을 생성합니다.
	•	이 임베딩은 셀프 어텐션(Self-Attention)과 간단한 예측 헤드(prediction head)를 통해 미래 이벤트(관계) 발생 여부를 예측합니다.

기존 방법과의 차이점:
	•	기존 방법:
	•	GNN과 RNN을 사용하여 중간 임베딩을 업데이트.
	•	구조적 정보에 의존하며 텍스트 요약과 같은 컨텍스트 정보는 간과됨.
	•	복잡한 설계로 인해 높은 계산 비용 발생.
	•	제안된 방법:
	•	GNN과 RNN을 제거하고, 대형 언어 모델(LLM)을 활용하여 간단한 프롬프트로 문제를 재구성.
	•	텍스트 요약과 같은 컨텍스트 정보를 효과적으로 활용하여 모델 성능 향상.
	•	더 적은 설계 복잡도로도 경쟁력 있는 예측 정확도를 달성.





This paper proposes a novel TKG learning framework called LEAF, which leverages Large Language Models (LLMs). Traditional TKG learning methods rely on Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) to process temporal and structural data. However, these methods fail to fully utilize the contextual information embedded in the quintuple (subject, relation, object, timestamp, textual summary).

LEAF Framework Methodology:
	1.	Object Prediction:
	•	Ranking Approach: Encoder-based LLMs like RoBERTa encode the textual summary into sentence embeddings, which are then combined with a structural decoder to predict the object.
	•	Generative Approach: Encoder-decoder LLMs like FLAN-T5 are fine-tuned in a question-answering (QA) format to directly generate the missing object.
	2.	Multi-Event Forecasting:
	•	Uses a simple prompt template for quintuples and encodes each quintuple with an encoder-based LLM like RoBERTa.
	•	The embeddings are processed using self-attention and a simple prediction head to predict future event relations.

Differences from Previous Methods:
	•	Previous Methods:
	•	Relied on GNNs and RNNs for intermediate embedding updates.
	•	Focused on structural information and ignored contextual details like textual summaries.
	•	Computationally expensive due to complex architectures.
	•	Proposed Method:
	•	Eliminates the need for GNNs and RNNs, restructuring tasks using LLMs with simple prompts.
	•	Effectively utilizes contextual information (textual summaries) to enhance performance.
	•	Achieves competitive prediction accuracy with reduced design complexity.


   
 
<br/>
# Results  





이 논문에서 제안된 LEAF 프레임워크는 두 가지 주요 작업인 객체 예측(Object Prediction)과 다중 이벤트 예측(Multi-Event Forecasting)에서 기존 모델들보다 뛰어난 성능을 보였습니다.

1. 객체 예측 (Object Prediction)

	•	데이터셋: ICEWS 데이터셋 (아프가니스탄, 인도, 러시아)
	•	비교 모델:
	•	SeCoGD: 그래프 기반 모델로, 컨텍스트 클러스터링을 사용.
	•	ConvTransE: 관계 기반의 컨볼루션 네트워크 모델.
	•	Baseline w/o LLM: LLM 없이 GNN과 RNN으로만 설계된 베이스라인.
	•	평가지표: Hits@1/3/10 (예측이 정확한 순위 내에 포함되는 비율)
	•	결과:
	•	LEAF-OP는 모든 데이터셋에서 Hits@1, Hits@3, Hits@10 지표에서 기존 모델들을 크게 능가했습니다.
	•	예를 들어, 러시아 데이터셋에서 Hits@1 기준으로 ConvTransE는 10.09%의 정확도를 보였으나 LEAF-OP는 37.51%를 기록했습니다.
	•	이는 텍스트 요약과 같은 컨텍스트 정보를 LLM을 통해 효과적으로 활용했기 때문입니다.

2. 다중 이벤트 예측 (Multi-Event Forecasting)

	•	데이터셋: ICEWS 데이터셋 (아프가니스탄, 인도, 러시아)
	•	비교 모델:
	•	Glean: GNN과 RNN을 결합한 구조적 프레임워크.
	•	RENET: 이벤트 그래프 기반 RNN 모델.
	•	Temporal GCN: 시간적 그래프 컨볼루션 네트워크.
	•	Baseline DNN: 단순 심층 신경망 모델.
	•	평가지표: F1-Score, Precision, Recall
	•	결과:
	•	LEAF-MEF는 F1-Score, Precision, Recall에서 Glean 및 다른 모델들을 초월하는 성능을 보였습니다.
	•	특히, 인도 데이터셋에서 LEAF-MEF는 F1-Score 70.99%로 Glean(66.69%) 대비 더 높은 성능을 기록했습니다.
	•	셀프 어텐션(Self-Attention) 메커니즘을 도입함으로써 시간적 관계와 이벤트 간의 복잡한 상호작용을 더 잘 학습할 수 있었던 것이 주요 원인입니다.

결과 요약

	•	LEAF 프레임워크의 강점:
	1.	텍스트 요약을 활용해 컨텍스트 정보를 효과적으로 반영.
	2.	복잡한 구조적 모델(GNN, RNN) 없이 더 간단한 설계로 높은 정확도 달성.
	3.	다양한 메트릭(F1, Precision, Recall, Hits@1/3/10)에서 우수한 성능을 입증.




The proposed LEAF framework outperformed existing models in two major tasks: Object Prediction and Multi-Event Forecasting.

1. Object Prediction

	•	Dataset: ICEWS datasets (Afghanistan, India, Russia)
	•	Baseline Models:
	•	SeCoGD: A graph-based model leveraging context clustering.
	•	ConvTransE: A relation-based convolutional network model.
	•	Baseline w/o LLM: A baseline using only GNNs and RNNs without LLMs.
	•	Metrics: Hits@1/3/10 (percentage of correct predictions within top ranks)
	•	Results:
	•	LEAF-OP outperformed all baseline models across all datasets.
	•	For example, in the Russia dataset, ConvTransE achieved 10.09% Hits@1, whereas LEAF-OP achieved 37.51%.
	•	This was achieved by effectively utilizing contextual information, such as textual summaries, through LLMs.

2. Multi-Event Forecasting

	•	Dataset: ICEWS datasets (Afghanistan, India, Russia)
	•	Baseline Models:
	•	Glean: A structural framework combining GNNs and RNNs.
	•	RENET: An event graph-based RNN model.
	•	Temporal GCN: A temporal graph convolutional network.
	•	Baseline DNN: A simple deep neural network.
	•	Metrics: F1-Score, Precision, Recall
	•	Results:
	•	LEAF-MEF achieved higher F1-Score, Precision, and Recall compared to Glean and other baselines.
	•	Notably, in the India dataset, LEAF-MEF achieved an F1-Score of 70.99%, surpassing Glean’s 66.69%.
	•	The introduction of a self-attention mechanism allowed the model to better capture complex temporal interactions between events.

Summary of Results

	•	Strengths of LEAF Framework:
	1.	Effectively integrates contextual information using textual summaries.
	2.	Achieves high accuracy with a simpler design by eliminating complex structures like GNNs and RNNs.
	3.	Demonstrates superior performance across multiple metrics (F1, Precision, Recall, Hits@1/3/10).



 


<br/>
# 예제  




예시 1: 객체 예측 (Object Prediction)
	•	데이터셋 구성: ICEWS 데이터셋에서 주제(예: “바락 오바마”), 관계(예: “발표하다”), 타임스탬프(예: “2014-11-15”), 텍스트 요약(예: “오바마가 서울에서 연설했다”)의 5요소를 사용해 객체(예: “한국”)를 예측.
	•	결과: 러시아 데이터셋 기준으로 LEAF-OP는 Hits@1에서 37.51%를 기록하며, 기존 모델 SeCoGD(17.68%)와 ConvTransE(10.09%)를 크게 앞섰음.
	•	차별점: LEAF는 텍스트 요약을 효과적으로 활용해 컨텍스트를 반영하며, 기존의 GNN/RNN 기반 구조적 모델보다 단순한 설계로 높은 성능을 달성.

예시 2: 다중 이벤트 예측 (Multi-Event Forecasting)
	•	데이터셋 구성: 동일 ICEWS 데이터셋에서 특정 날짜의 과거 이벤트 데이터를 바탕으로 다음 날 발생할 수 있는 이벤트(예: “협상”, “비판”)를 예측.
	•	결과: 인도 데이터셋 기준으로 LEAF-MEF는 F1-Score 70.99%로 Glean(66.69%)을 초과.
	•	차별점: 셀프 어텐션 메커니즘을 도입해 이벤트 간의 시간적 상호작용을 더 잘 학습하며, GNN/RNN 없이도 높은 정확도를 기록.

핵심 우위:
	1.	간단한 프롬프트와 LLM으로 복잡한 모델(GNN/RNN)을 대체.
	2.	텍스트 요약을 활용해 컨텍스트를 강화.
	3.	계산 비용이 낮으면서도 높은 성능.




Example 1: Object Prediction
	•	Dataset Composition: In the ICEWS dataset, the task involves predicting the missing object (e.g., “Korea”) from a quintuple with elements like subject (“Barack Obama”), relation (“made a statement”), timestamp (“2014-11-15”), and textual summary (“Obama delivered a speech in Seoul”).
	•	Results: On the Russia dataset, LEAF-OP achieved 37.51% Hits@1, outperforming SeCoGD (17.68%) and ConvTransE (10.09%).
	•	Difference: LEAF effectively utilizes textual summaries to capture contextual information, outperforming traditional GNN/RNN models with simpler architecture.

Example 2: Multi-Event Forecasting
	•	Dataset Composition: Using past event data from the ICEWS dataset, the task predicts future relations (e.g., “negotiate,” “criticize”) for the next day.
	•	Results: On the India dataset, LEAF-MEF achieved an F1-Score of 70.99%, surpassing Glean’s 66.69%.
	•	Difference: The introduction of self-attention allows LEAF to better learn temporal interactions between events, achieving superior accuracy without GNN/RNN components.

Key Advantages:
	1.	Replaces complex GNN/RNN models with simple prompts and LLMs.
	2.	Enhances contextual understanding using textual summaries.
	3.	Achieves high performance with reduced computational cost.


<br/>  
# 요약   




ICEWS 데이터셋에서 “Barack Obama”와 “make a visit”와 같은 주제와 관계가 주어졌을 때, LEAF는 텍스트 요약(“Obama is expected to speak at the University of Queensland”)을 활용하여 객체(“Australia”)를 정확히 예측했습니다. 기존 모델인 SeCoGD와 ConvTransE는 텍스트 요약을 활용하지 못해 상위 10개 후보 안에는 들지만 상위 1위(Hits@1)에서는 실패했습니다. LEAF는 RoBERTa와 같은 대형 언어 모델로 텍스트의 컨텍스트를 학습하며, 다중 이벤트 예측에서도 셀프 어텐션을 활용해 이전 이벤트 간의 상호작용을 효과적으로 반영했습니다. 결과적으로, LEAF는 F1-Score 기준으로 Glean보다 5% 이상 높은 성능을 기록하며, 단순한 설계로도 높은 예측 정확도를 달성했습니다. 이는 복잡한 그래프 기반 모델 없이 텍스트 정보를 효과적으로 활용한 점에서 기존 방법들과 차별화됩니다.




In the ICEWS dataset, given “Barack Obama” as the subject and “make a visit” as the relation, LEAF accurately predicted the object “Australia” by utilizing the textual summary (“Obama is expected to speak at the University of Queensland”). In contrast, models like SeCoGD and ConvTransE failed to leverage the textual context, ranking “Australia” lower in their predictions. LEAF used RoBERTa to effectively learn from textual context, and for multi-event forecasting, it employed self-attention to capture interactions between prior events. As a result, LEAF achieved an F1-Score over 5% higher than Glean, demonstrating superior predictive accuracy with a simpler design. This highlights its ability to outperform traditional graph-based models by effectively incorporating textual information.



<br/>  
# 기타  


ICEWS 데이터셋이란?

ICEWS (Integrated Crisis Early Warning System) 데이터셋은 전 세계적으로 발생하는 정치적, 사회적 사건들을 체계적으로 기록한 데이터셋입니다. 이는 2015년에 Boschee와 그의 동료들이 개발한 데이터셋으로, 정부, 학계, 비영리단체 등이 위기 상황을 사전에 예측하거나 분석하기 위해 사용됩니다. 이 데이터셋은 다음과 같은 주요 특징을 갖습니다:

특징

	1.	구조:
	•	사건은 주제(Subject), 관계(Relation), 객체(Object), 타임스탬프(Timestamp), 텍스트 요약(Text Summary)로 구성된 4요소 또는 5요소 구조(quadruple 또는 quintuple)로 표현됩니다.
	•	예: (“Barack Obama”, “visit”, “Australia”, “2014-11-15”, “Obama is expected to speak at the University of Queensland”).
	2.	범위:
	•	정치적 사건(예: 회담, 협약), 군사적 움직임(예: 폭력, 협박), 경제적 이벤트 등을 포함합니다.
	•	국가별로 사건이 분리되어 있으며, 아프가니스탄, 인도, 러시아 등의 데이터를 포함합니다.
	3.	목적:
	•	사건의 연속성을 분석하고, 특정 사건의 발생 가능성을 예측하거나 다음에 발생할 이벤트를 모델링하기 위한 목적으로 설계되었습니다.
	4.	응용:
	•	기계 학습, 특히 시간적 지식 그래프(Temporal Knowledge Graph)와 같은 연구 분야에서 활용됩니다.

주요 용도

ICEWS 데이터셋은 특정 사건(예: “오바마가 어느 국가를 방문할 것인가?”)에 대해 객체를 예측하거나, 다중 이벤트 예측(예: “다음 날 어떤 사건들이 발생할 것인가?”)을 위한 학습에 사용됩니다.

What is the ICEWS Dataset?

The Integrated Crisis Early Warning System (ICEWS) dataset is a structured collection of political and social events from around the world, developed by Boschee and colleagues in 2015. It is widely used by governments, academia, and nonprofits to predict or analyze crisis situations in advance. Here are its key characteristics:

Features

	1.	Structure:
	•	Events are represented as quadruples or quintuples consisting of subject, relation, object, timestamp, and an optional textual summary.
	•	Example: (“Barack Obama”, “


<br/>
# refer format:     



@article{zhang2024large,
  title={Large Language Models as Event Forecasters},
  author={Zhang, Libo and Ning, Yue},
  journal={arXiv preprint arXiv:2406.10492},
  year={2024},
  url={https://arxiv.org/abs/2406.10492}
}  



Zhang, Libo, and Yue Ning. “Large Language Models as Event Forecasters.” arXiv preprint arXiv:2406.10492 (2024). https://arxiv.org/abs/2406.10492.  










