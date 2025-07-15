---
layout: post
title:  "[2025]Best of Both Worlds: Advantages of Hybrid Graph Sequence Models"  
date:   2025-07-15 00:25:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 


Graph Sequence Model과, 계층적 클러스터링 기반 토큰화 및 하이브리드 시퀀스 인코더를 결합한 GSM++ 모델을 제안   



짧은 요약:  




최근에는 Transformer나 순환 신경망(RNN) 같은 시퀀스 모델이 효율성과 표현력, 장거리 의존성 처리 능력 덕분에 그래프 데이터에도 적용되고 있습니다. 하지만 기존의 메시지 전달 신경망(MPNN)과 비교하여 어떤 시퀀스 모델이 그래프 학습에 적합한지에 대한 이론적 기반은 부족한 실정입니다. 이에 본 논문에서는 다양한 시퀀스 모델을 그래프에 적용할 수 있도록 통합된 이론적 틀인 **Graph Sequence Model (GSM)**을 제안합니다. 이 프레임워크를 통해 시퀀스 모델의 그래프 학습 능력을 체계적으로 비교하고 이해할 수 있습니다.

또한, 저자들은 **GSM++**라는 새로운 하이브리드 모델을 제안합니다. 이 모델은 **계층적 유사도 클러스터링(HAC)**을 이용해 그래프를 계층적으로 토큰화한 후, 순환 모델과 Transformer를 결합한 하이브리드 구조를 적용합니다. 이론적 분석과 실험 결과는 GSM++가 다양한 그래프 태스크에서 뛰어난 성능을 보인다는 것을 입증합니다.




Modern sequence models (e.g., Transformers and linear RNNs) have become dominant in deep learning due to their efficiency, expressive power, and ability to capture long-range dependencies. Recently, these models have also been adapted for graph-structured data as alternatives to traditional Message Passing Neural Networks (MPNNs). However, a foundational understanding of what makes a good graph sequence model—and how different sequence models behave on graphs—is lacking.

To address this, the authors introduce the Graph Sequence Model (GSM) framework, which unifies various sequence modeling approaches for graph data and enables a systematic evaluation of their strengths and weaknesses. Building on this, the paper proposes GSM++, a fast hybrid model that uses Hierarchical Affinity Clustering (HAC) to tokenize graphs hierarchically, followed by a hybrid encoder combining recurrent and transformer layers. Theoretical insights and experimental results validate the effectiveness of GSM++, showing strong performance across various graph learning tasks.


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




이 논문에서는 그래프 데이터를 시퀀스로 변환하여 학습하는 통합 프레임워크인 \*\*Graph Sequence Model (GSM)\*\*을 제안합니다. 이 모델은 세 단계로 구성됩니다:

1. **Tokenization (토큰화)**
   그래프를 시퀀스로 변환하는 과정입니다. 기존의 노드/엣지 단위 토큰화 외에도, 저자들은 **서브그래프 기반 토큰화**를 사용하거나 이를 결합하는 방식도 제안합니다. 특히, 유사한 노드들이 가까이 위치하도록 **계층적 유사도 클러스터링(Hierarchical Affinity Clustering, HAC)** 기반의 **HAC Tokenization**을 도입합니다.

2. **Local Encoding (지역 인코딩)**
   토큰화된 각 노드 또는 서브그래프를 벡터로 임베딩합니다. 주로 MPNN(메시지 전달 그래프 신경망)을 이용해 각 토큰의 로컬 구조를 학습합니다.

3. **Global Encoding (전역 인코딩)**
   시퀀스 모델을 통해 각 토큰 간의 장거리 의존성과 전체 그래프 구조를 학습합니다. 이 단계에서 저자들은 다음 세 가지 구조를 실험합니다:

   * Transformer 기반 모델
   * Mamba와 같은 State Space Model(SSM)
   * **하이브리드 모델 (GSM++)**: 두 개의 SSM 레이어 + 하나의 Transformer 블록으로 구성된 새로운 구조로, 순차성과 전역 표현력을 동시에 활용합니다.

추가적으로, 토큰화 전략을 혼합하여 사용하는 \*\*Mixture of Tokenization (MoT)\*\*도 제안되며, 이는 각 노드마다 최적의 토큰화 방식을 동적으로 선택하는 방식입니다.

---



The authors propose a unified framework called **Graph Sequence Model (GSM)** for applying sequence models to graph-structured data. It is composed of the following three stages:

1. **Tokenization**
   Graphs are converted into sequences. Beyond basic node or edge tokenization, the authors introduce **Hierarchical Affinity Clustering (HAC)** to create **HAC Tokenization**, which organizes similar nodes close together in a sequence using a hierarchical tree structure.

2. **Local Encoding**
   Each token (node, edge, or subgraph) is embedded using local encoders such as message-passing GNNs, capturing local graph structure and inductive biases.

3. **Global Encoding**
   Sequence models are applied to learn dependencies across tokens. The paper compares three architectures:

   * Transformer-based models
   * State Space Models (SSMs), such as Mamba
   * **Hybrid architecture (GSM++)**, which consists of two SSM layers followed by a Transformer block, combining the benefits of recurrence and global attention.

Additionally, they propose a **Mixture of Tokenization (MoT)** technique, which selects the best tokenization strategy for each node based on its local graph context, thereby improving flexibility and performance.




   
 
<br/>
# Results  





1. **그래프 태스크에서의 성능**
   논문은 로컬(Local) 태스크(예: 노드 차수 예측, 삼각형 개수 세기)와 글로벌(Global) 태스크(예: 그래프 연결성 판별, 색상 개수 세기, 최단 경로 예측) 모두에 대해 평가했습니다.

   * 로컬 태스크에서는 기존 MPNN 계열 모델이 여전히 강한 성능을 보였고, \*\*서브그래프 기반 토큰화(k-hop, HAC)\*\*가 노드 기반보다 우수함을 보였습니다.
   * 글로벌 태스크에서는 Transformer 기반 모델이 전반적으로 강력했으며, **노드 기반 토큰화**가 더 적합했습니다.

2. **GSM++의 우수성**
   GSM++ 모델은 다양한 조합 실험에서 일관되게 높은 성능을 보였습니다. 특히, \*\*하이브리드 구조(Mamba + Transformer)\*\*와 **HAC 토큰화**, \*\*계층적 위치 인코딩(PE)\*\*의 조합은 전반적으로 기존 모델보다 성능이 뛰어났습니다.

3. **벤치마크 데이터셋 성능**
   GSM++는 CIFAR10, PATTERN, Peptides-Func, MalNet-Tiny, COCO-SP 등 다양한 공개 그래프 데이터셋에서 최고 또는 두 번째 성능을 달성했습니다. 특히 \*\*GSM++ (MoT)\*\*는 가장 많은 태스크에서 최상위 성능을 보였습니다.

4. **어블레이션 실험 결과**
   하이브리드 구조나 HAC 토큰화, MoT 전략 중 하나라도 제거할 경우 성능이 떨어졌으며, **GSM++의 구성요소들이 모두 성능에 기여**함을 입증했습니다.

---



1. **Performance on Graph Tasks**
   The paper evaluates performance on both **local tasks** (e.g., node degree prediction, triangle counting) and **global tasks** (e.g., graph connectivity, color counting, shortest path estimation).

   * For local tasks, MPNNs perform strongly, and **subgraph tokenizers** like k-hop and HAC outperform node-based tokenization.
   * For global tasks, Transformer-based models generally perform better, and **node tokenization** proves more suitable.

2. **Superiority of GSM++**
   The proposed **GSM++** model consistently shows strong performance across different tasks. Its **hybrid architecture (Mamba + Transformer)**, **HAC tokenization**, and **hierarchical positional encoding** contribute significantly to its success.

3. **Benchmark Performance**
   GSM++ achieves top or second-best results on major benchmarks including **CIFAR10, PATTERN, Peptides-Func, MalNet-Tiny, and COCO-SP**. The **GSM++ (MoT)** variant delivers the highest accuracy in most cases.

4. **Ablation Study**
   Removing any component (hybrid, HAC, or positional encoding) from GSM++ led to performance drops, confirming that **each design choice contributes positively** to the final model's effectiveness.




<br/>
# 예제  


---

####  예제 1: 노드 차수 예측 (Node Degree Prediction)

* **입력**: 노드로 구성된 그래프, 각 노드의 연결 정보
* **출력**: 각 노드의 차수 (연결된 엣지의 수)
* **목표**: 모델이 노드의 주변 연결 정보를 바탕으로 정확한 차수를 예측할 수 있는지 평가
* **학습 데이터**: 노드 특성과 인접 정보가 주어진 그래프들

---

####  예제 2: 그래프 연결성 예측 (Graph Connectivity)

* **입력**: 엣지 리스트 기반으로 토큰화된 그래프 시퀀스
* **출력**: 해당 그래프가 연결 그래프(connected)인지 아닌지에 대한 이진 분류 (1 또는 0)
* **목표**: 전체 그래프 구조를 이해할 수 있는 전역 표현 학습
* **학습 데이터**: 서로 다른 노드 수와 구조를 갖는 연결/비연결 그래프 샘플

---

####  예제 3: 색상 개수 세기 (Color Counting)

* **입력**: 노드에 색상이 부여된 그래프
* **출력**: 각 색상의 노드 수 (예: 빨강: 3, 파랑: 5)
* **목표**: 전체 노드를 모두 고려한 정합적인 수량 예측
* **학습 데이터**: 다양한 색상 분포를 갖는 노드-컬러 그래프

---

이러한 태스크는 각각 **로컬(Local) 혹은 글로벌(Global)** 정보 처리 능력을 요구하며, 모델의 구조적 특성과 토큰화 전략이 성능에 큰 영향을 미쳤습니다.

---

###  Example Tasks (English)

The paper evaluates the model on the following **concrete graph-based tasks**, highlighting the input/output structure and learning setup:

---

#### Example 1: Node Degree Prediction

* **Input**: A graph with nodes and their connections
* **Output**: The degree of each node (i.e., the number of connected edges)
* **Objective**: Assess whether the model can infer local structural information
* **Training Data**: Graphs with known node degrees and neighborhood structure

---

####  Example 2: Graph Connectivity Classification

* **Input**: A graph represented as a sequence of edges (tokenized)
* **Output**: A binary label (1 if the graph is connected, 0 otherwise)
* **Objective**: Test the model’s ability to understand global graph structure
* **Training Data**: Samples of both connected and disconnected graphs with varying sizes and structures

---

####  Example 3: Color Counting

* **Input**: A graph with nodes labeled by colors
* **Output**: A count of how many nodes belong to each color class (e.g., red: 3, blue: 5)
* **Objective**: Require the model to consider all nodes to produce an accurate global count
* **Training Data**: Graphs with varied color distributions and node sets





<br/>  
# 요약   



이 논문은 그래프 데이터를 시퀀스로 처리하기 위한 통합 프레임워크인 **GSM(Graph Sequence Model)**과, 계층적 클러스터링 기반 토큰화 및 하이브리드 시퀀스 인코더를 결합한 GSM++ 모델을 제안합니다. 다양한 그래프 태스크(노드 차수 예측, 그래프 연결성 판별, 색상 개수 세기 등)에서 GSM++는 기존 MPNN이나 Transformer보다 더 높은 정확도와 일반화 성능을 보여주었습니다. 특히 HAC 기반 토큰화, 계층적 위치 인코딩, Mamba+Transformer 하이브리드 구조의 조합이 모델 성능 향상에 핵심 역할을 했습니다.


This paper introduces the Graph Sequence Model (GSM) framework and proposes GSM++, a model combining hierarchical clustering-based tokenization with a hybrid sequence encoder. Across a range of graph tasks—such as node degree prediction, connectivity classification, and color counting—GSM++ outperforms standard MPNNs and Transformer-based models in both accuracy and generalization. Key contributors to this performance include the HAC-based tokenization, hierarchical positional encoding, and the hybrid architecture of Mamba and Transformer.

<br/>  
# 기타    




####  Figure 1: GSM++ 구조 개요

* **구성**:

  * (1) HAC 기반 토큰화
  * (2) 계층적 위치 인코딩 (Hierarchical PE)
  * (3) 하이브리드 시퀀스 인코더 (SSM 2층 + Transformer)
* **인사이트**: 시퀀스로 변환된 그래프에서 유사한 노드를 가깝게 배치하고, 지역과 전역 정보를 동시에 처리하여 더 강력한 그래프 표현을 학습할 수 있음을 시각화로 보여줌.

---

####  Table 1 & 2: 로컬/글로벌 태스크 성능 비교

* **내용**:

  * Table 1은 노드 차수, 삼각형 개수 등의 **로컬 태스크**
  * Table 2는 그래프 연결성, 색상 개수 등의 **글로벌 태스크** 결과 제시
* **결론**:

  * 로컬 태스크에서는 MPNN, k-hop, HAC 기반 서브그래프 토큰화가 강세
  * 글로벌 태스크에서는 Transformer 기반 구조 + 노드 토큰화가 효과적
  * **하이브리드 구조(GSM++)는 대부분의 태스크에서 전체적으로 가장 높은 성능**을 보임

---

####  Table 3, 5, 6: 벤치마크 데이터셋 결과

* **내용**:

  * CIFAR10, PATTERN, Peptides-Func 등에서 GSM++가 최고 또는 차상위 성능 기록
  * 특히 GSM++ (MoT)는 대부분 태스크에서 기존 모델보다 우위
* **인사이트**:

  * GSM++는 다양한 그래프 구조와 문제 유형에 강건하며, **단일 모델로도 넓은 범위의 그래프 문제에 대응 가능**함을 시사함

---

####  Table 4: 어블레이션 스터디

* **내용**: HAC, 하이브리드, 위치 인코딩 각각 제거했을 때 성능 하락 분석
* **결론**: 각 구성요소가 성능 향상에 필수적이며, **GSM++의 성능은 각 기술의 조합에서 발생**함을 확인

---

####  Figure 2: 모델 조합별 순위 시각화

* **내용**: 서로 다른 시퀀스 모델과 토크나이저 조합 54개를 7개 데이터셋에서 실험한 결과의 평균 순위
* **인사이트**: 단일 모델이 모든 경우에서 압도적으로 우세하지 않으며, **태스크나 데이터셋에 따라 최적 모델이 달라짐** → "No Free Lunch" 성질 확인

---


####  Figure 1: GSM++ Architecture Overview

* **Includes**:

  * (1) HAC-based tokenization
  * (2) Hierarchical Positional Encoding
  * (3) A hybrid sequence model with two SSM layers followed by a Transformer
* **Insight**: Visually demonstrates how GSM++ combines locality-aware structure and global reasoning, placing similar nodes close in sequence and improving representational capacity.

---

####  Table 1 & 2: Local and Global Task Performance

* **Content**:

  * Table 1: Local tasks (node degree, triangle counting)
  * Table 2: Global tasks (connectivity, color counting)
* **Conclusion**:

  * MPNNs and subgraph tokenization (like k-hop, HAC) excel in local tasks
  * Transformers and node tokenization perform better on global tasks
  * **GSM++ achieves the best or second-best performance across nearly all tasks**, validating its hybrid design

---

####  Table 3, 5, 6: Benchmark Dataset Results

* **Content**: Performance on CIFAR10, PATTERN, Peptides-Func, MalNet-Tiny, etc.
* **Insight**:

  * GSM++ (especially with MoT) outperforms existing models in 8/10 cases
  * Indicates strong generalization and robustness across diverse graph settings

---

####  Table 4: Ablation Studies

* **Content**: Performance drops when HAC, hybrid model, or PE are removed
* **Conclusion**: Confirms that each component (HAC, hybrid encoder, PE) is critical to GSM++’s performance; their **combination leads to synergy**.

---

####  Figure 2: Model Combination Rankings

* **Content**: Normalized performance ranks of 54 combinations (9 sequence models × 6 tokenizations) on 7 datasets
* **Insight**: No single model dominates universally, supporting the **“No Free Lunch” theorem**—model choice must be tailored to the task and dataset.




<br/>
# refer format:     



@inproceedings{behrouz2025gsmpp,
  title     = {Best of Both Worlds: Advantages of Hybrid Graph Sequence Models},
  author    = {Ali Behrouz and Ali Parviz and Mahdi Karami and Clayton Sanford and Bryan Perozzi and Vahab Mirrokni},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  publisher = {PMLR},
  volume    = {267},
  address   = {Vancouver, Canada},
  url       = {https://proceedings.mlr.press/v267/behrouz25a.html}
}




Behrouz, Ali, Ali Parviz, Mahdi Karami, Clayton Sanford, Bryan Perozzi, and Vahab Mirrokni. “Best of Both Worlds: Advantages of Hybrid Graph Sequence Models.” In Proceedings of the 42nd International Conference on Machine Learning, vol. 267, Vancouver, Canada: PMLR, 2025.






