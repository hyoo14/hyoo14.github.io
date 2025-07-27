---
layout: post
title:  "[2025]Layer by Layer Uncovering Hidden Representations in Language Models"  
date:   2025-07-27 02:36:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 


대형 언어 모델의 중간 레이어 표현이 마지막 레이어보다 더 유용할 수 있다는 가설을 검증




---


기존에는 대형 언어 모델(LLMs)의 마지막 레이어가 다운스트림 작업에 가장 유용한 표현을 제공한다고 여겨졌습니다. 하지만 이 논문은 **중간 레이어**들이 오히려 더 풍부하고 유용한 표현을 담고 있다는 사실을 밝혔습니다. 저자들은 정보 이론, 기하 구조, 입력 변화에 대한 불변성(invariance)이라는 세 가지 관점에서 표현 품질을 측정하는 통합 프레임워크를 제안합니다. 이 프레임워크는 각 레이어가 정보 압축과 신호 보존 사이의 균형을 어떻게 유지하는지를 보여줍니다. 실험 결과, 언어와 비전 모델 모두에서 중간 레이어 표현이 마지막 레이어보다 더 높은 정확도를 보여주며, 기존의 통념을 뒤엎는 결과를 보였습니다. 이 논문은 중간 레이어 표현을 보다 강력하고 정확한 특징으로 활용하는 새로운 방향성을 제시합니다.

---


> From extracting features to generating text, the outputs of large language models (LLMs) typically rely on the final layers, following the conventional wisdom that earlier layers capture only low-level cues. However, our analysis shows that intermediate layers can encode even richer representations, often improving performance on a range of downstream tasks. To explain and quantify these hidden-layer properties, we propose a unified framework of representation quality metrics based on information theory, geometry, and invariance to input perturbations. Our framework highlights how each layer balances information compression and signal preservation, revealing why mid-depth embeddings can exceed the last layer’s performance. Through extensive experiments on 32 text-embedding tasks across various architectures (transformers, state-space models) and domains (language, vision), we demonstrate that intermediate layers consistently provide stronger features, challenging the standard view on final-layer embeddings and opening new directions on using mid-layer representations for more robust and accurate representations.

---





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





이 논문에서는 다양한 대형 언어 모델(LLMs)의 **중간 레이어 표현**이 마지막 레이어보다 더 나은 특성을 가질 수 있다는 가설을 검증하기 위해, 다음과 같은 메서드를 사용합니다.

1. **모델 아키텍처**
   Transformer 기반 모델(Pythia, LLaMA3, BERT), 상태공간모델(State Space Model, Mamba), 양방향 인코더 기반 모델(LLM2Vec) 등 다양한 구조의 LLM을 사용했습니다.
   모델 크기는 수천만에서 10억 파라미터 이상까지 포함됩니다.

2. **평가 데이터**
   Massive Text Embedding Benchmark (MTEB)의 32개 다운스트림 태스크(분류, 군집화, 재정렬 등)에 대해 레이어별 임베딩을 사용해 성능을 측정합니다.

3. **핵심 분석 프레임워크**
   중간 레이어 표현을 측정하기 위해 **정보 이론적 지표(Entropy, Effective Rank)**, **기하학적 지표(Curvature)**, \*\*불변성 지표(InfoNCE, LiDAR, DiME)\*\*를 통합한 **표현 품질 평가 프레임워크**를 제안합니다.
   특히, **matrix-based entropy**를 중심으로 다양한 표현 품질 척도를 통일된 수학적 틀로 분석합니다.

4. **비지도 레이어 선택**
   학습된 LLM의 모든 레이어에서 임베딩을 추출한 후, **레이블 없이도 성능이 높은 레이어를 식별**할 수 있도록 하는 지표(DiME 등)를 사용합니다.

---



To investigate whether intermediate layers in large language models (LLMs) yield better representations than the final layer, the authors employ the following methodology:

1. **Model Architectures**
   They analyze diverse architectures, including transformer-based models (Pythia, LLaMA3, BERT), state-space models (Mamba), and bidirectional encoders (LLM2Vec), ranging from tens of millions to over a billion parameters.

2. **Evaluation Data**
   Representations from each layer are evaluated on 32 downstream tasks from the Massive Text Embedding Benchmark (MTEB), including classification, clustering, and reranking.

3. **Unified Evaluation Framework**
   The authors introduce a unified framework of representation quality metrics based on three perspectives:
   (a) **Information-theoretic** (e.g., prompt entropy, effective rank),
   (b) **Geometric** (e.g., curvature), and
   (c) **Invariance-based** (e.g., InfoNCE, LiDAR, DiME).
   These are unified through **matrix-based entropy** to provide a consistent theoretical foundation.

4. **Unsupervised Layer Selection**
   Without using task-specific labels, they select high-performing intermediate layers by ranking them according to quality metrics such as DiME, enabling more efficient downstream usage.




   
 
<br/>
# Results  






1. **중간 레이어의 성능 우위**
   다양한 모델(Pythia, BERT, Mamba)을 MTEB 벤치마크의 32개 과제에서 평가한 결과, **거의 모든 모델에서 중간 레이어가 마지막 레이어보다 높은 정확도**를 기록했습니다. 중간 레이어는 최종 레이어보다 평균적으로 **최대 16% 더 높은 성능**을 보였습니다.

2. **모델 구조별 차이**

   * \*\*Autoregressive 모델(Pythia)\*\*은 **중간층에서 정보 압축이 가장 심해지는 ‘entropy valley’** 현상이 나타나고, 이 구간이 성능이 가장 좋음.
   * **BERT와 같은 양방향 인코더**는 전체적으로 고른 entropy 분포를 보이며, 특정 레이어가 도드라지지 않음.
   * \*\*State-space 모델(Mamba)\*\*은 완만한 압축 경향을 보이며, transformer와는 다른 layer-wise 특성을 보임.

3. **평가 지표와의 상관관계**
   제안한 표현 품질 지표들(DiME, InfoNCE, Curvature 등)이 실제 다운스트림 성능과 \*\*높은 상관관계(distance correlation 0.75\~0.9)\*\*를 보였습니다. 특히 DiME 기반으로 최적의 중간 레이어를 선택하면, **비지도 학습만으로도 평균 3% 성능 향상**이 가능했습니다.

4. **시각 모델로의 일반화**
   비전 모델(AIM, BEiT, DINOv2 등)에 대해서도 실험을 확장한 결과, **Autoregressive 방식의 AIM에서는 중간 레이어에서 성능이 정점에 이르는 패턴**이 다시 나타났습니다. 이는 autoregressive 학습 목표 자체가 중간 레이어에 정보를 응축시키는 구조임을 시사합니다.

---



1. **Intermediate Layers Outperform Final Layers**
   Across 32 tasks from the MTEB benchmark, intermediate layers consistently outperform final-layer embeddings. In some cases, the improvement reaches up to **16% higher accuracy**.

2. **Architectural Differences**

   * **Autoregressive models** (e.g., Pythia) show a pronounced "entropy valley" at mid-depth layers, which aligns with the **peak in downstream task performance**.
   * **Bidirectional encoder models** (e.g., BERT) maintain **more uniform representation quality** across layers, without a clear peak.
   * **State-space models** (e.g., Mamba) exhibit milder compression and different layer-wise dynamics compared to transformers.

3. **Correlation with Evaluation Metrics**
   Metrics like **DiME, InfoNCE, and curvature** strongly correlate with downstream accuracy (distance correlations of 0.75–0.9).
   Using DiME to select layers **without labels** improves performance by **an average of 3%** on MTEB.

4. **Generalization to Vision Models**
   Applying the analysis to vision models (e.g., AIM, BEiT, DINOv2) reveals that **autoregressive models like AIM** also exhibit a mid-layer performance peak. This suggests that **the training objective (autoregressive vs. masked) drives the information bottleneck**, rather than the data modality.




<br/>
# 예제  




이 논문은 표현의 품질을 측정하기 위해 \*\*Massive Text Embedding Benchmark (MTEB)\*\*라는 **대규모 벤치마크**를 활용합니다. 이 벤치마크에는 다음과 같은 다양한 다운스트림 태스크들이 포함되어 있습니다.

####  입력 (Input)

* 모델의 각 레이어에서 추출한 **문장 또는 문단 수준의 임베딩 벡터**
* 예: "The quick brown fox jumps over the lazy dog." → 임베딩 벡터 (예: 768차원)

####  출력 (Output)

* 태스크에 따라 다르며, 다음 중 하나:

  * 분류 결과 (예: 긍정/부정, 카테고리 라벨 등)
  * 클러스터 할당 (비지도 클러스터링)
  * 문장 유사도 점수 또는 랭킹 (재정렬 태스크)

####  구체적인 태스크 예시

1. **문장 분류 (Sentence Classification)**

   * 예: 질문이 정보성인지 여부 분류
   * 입력: 문장 텍스트
   * 출력: "informational" 또는 "not informational"

2. **문장 유사도 평가 (STS: Semantic Textual Similarity)**

   * 입력: 문장 쌍 (예: 문장 A와 B)
   * 출력: 두 문장의 의미적 유사도 점수 (예: 0.0 \~ 5.0)

3. **군집화 (Clustering)**

   * 입력: 뉴스 기사 제목들
   * 출력: 각 제목이 속하는 클러스터 (예: 정치, 스포츠, 과학 등)

4. **재정렬 (Reranking)**

   * 입력: 쿼리 문장과 후보 응답 문장들
   * 출력: 후보 응답들의 랭킹

####  태스크 구성

* 총 32개의 태스크가 포함됨 (분류, 클러스터링, 재정렬 등)
* 문장 수준에서 레이어별 임베딩을 활용하여 예측

---



The authors evaluate representation quality using **32 tasks from the Massive Text Embedding Benchmark (MTEB)**, which covers a wide range of downstream settings.

#### Input

* **Sentence or paragraph-level embeddings** extracted from each layer of the language model
* Example: "The quick brown fox jumps over the lazy dog." → 768-dimensional embedding vector

####  Output

Depending on the task, outputs vary:

* **Classification label** (e.g., positive/negative, category ID)
* **Cluster assignment** (unsupervised grouping)
* **Semantic similarity score** (for sentence pairs)
* **Ranking** (for reranking tasks)

####  Specific Task Examples

1. **Sentence Classification**

   * Task: Classify whether a sentence is informational
   * Input: Sentence text
   * Output: "informational" or "not informational"

2. **Semantic Textual Similarity (STS)**

   * Input: Pair of sentences
   * Output: A similarity score (e.g., from 0.0 to 5.0)

3. **Clustering**

   * Input: News headlines
   * Output: Cluster assignments such as politics, sports, science

4. **Reranking**

   * Input: A query sentence and a list of candidate answers
   * Output: Ranked list of candidates

####  Task Format

* A total of **32 text embedding tasks** are used, including classification, clustering, and reranking
* Representations are evaluated layer-wise using sentence embeddings for each task





<br/>  
# 요약   





이 논문은 대형 언어 모델의 중간 레이어 표현이 마지막 레이어보다 더 유용할 수 있다는 가설을 검증하기 위해, 정보 이론, 기하학, 불변성 기반의 통합 표현 품질 프레임워크를 제안하였다. MTEB의 32개 다운스트림 과제에서 실험한 결과, 거의 모든 모델에서 중간 레이어가 최대 16% 더 높은 성능을 보이며 마지막 레이어를 능가하였다. 문장 분류, 의미 유사도, 클러스터링 등 다양한 입력-출력 과제에서 중간 레이어 임베딩이 더욱 강건하고 일반화 가능한 특성을 가짐을 실증하였다.

---


This paper proposes a unified evaluation framework based on information theory, geometry, and invariance to assess whether intermediate representations in large language models outperform final-layer embeddings. Experiments on 32 downstream tasks from MTEB show that intermediate layers consistently yield up to 16% higher accuracy than final layers across various architectures. In tasks like sentence classification, semantic similarity, and clustering, intermediate-layer embeddings demonstrate more robust and generalizable representations.

---


<br/>  
# 기타  




####  Figure 1: 중간 레이어가 최종 레이어보다 성능이 높음

* 세 모델(Pythia, Mamba, BERT)에 대해 모든 레이어 임베딩을 MTEB에서 평가한 결과를 시각화
* **중간 레이어에서 평균 성능이 최고조에 달함**, 마지막 레이어는 오히려 성능이 감소함

####  Figure 2\~3: 표현 품질 메트릭 (Entropy, Curvature, LiDAR)

* 중간 레이어에서 정보 압축(entropy 감소), 기하 구조 안정화(curvature 감소), 불변성(LiDAR 향상)이 뚜렷하게 나타남
* 특히 Pythia는 "entropy valley" 현상이 심하게 나타나며, 이 구간에서 다운스트림 성능이 최고로 향상됨

####  Figure 6: 입력 조건 변화에 따른 레이어 반응

* 반복된 토큰 입력 → 중간 레이어에서 entropy 급감 (중복 정보 압축)
* 무작위 입력 → 초기 레이어에서 entropy 상승 (잡음에 민감), 후반부는 상대적으로 안정적
* 입력 길이 증가 → 전체 entropy 증가하나 중간층은 일정 패턴 유지

####  Figure 5, 14: CoT 파인튜닝 및 비전 모델 확장 실험

* 체인오브쏘트(CoT) 파인튜닝 후 중간 레이어가 \*\*더 풍부한 문맥 정보(높은 entropy)\*\*를 유지함
* autoregressive 비전 모델(AIM)에서도 언어 모델과 동일하게 **중간층 성능 피크 및 entropy valley**가 관측됨

####  Appendix E: 비지도 방식으로 레이어 선택

* 표현 품질 지표(DiME, InfoNCE 등)를 사용해 **레이블 없이 고성능 중간 레이어를 선택**할 수 있음을 보임
* 이 방법으로 **최종 레이어보다 평균 3% 향상된 성능 확보**

---


####  Figure 1: Intermediate Layers Outperform Final Layers

* Visualization of layer-wise MTEB scores across three models (Pythia, Mamba, BERT)
* **Intermediate layers achieve peak performance**, while final layers often decline

####  Figures 2–3: Representation Metrics (Entropy, Curvature, LiDAR)

* Entropy dips, curvature flattens, and LiDAR scores improve at mid-depth layers
* Pythia exhibits a pronounced **“entropy valley”**, correlating with downstream performance peaks

####  Figure 6: Layer Responses to Extreme Input Conditions

* Token repetition reduces entropy in intermediate layers (compression of redundancy)
* Random token inputs elevate entropy in early layers (sensitive to noise), while deeper layers remain robust
* Longer inputs lead to overall entropy growth but consistent mid-layer behavior

####  Figures 5, 14: Chain-of-Thought and Vision Model Results

* Chain-of-thought (CoT) fine-tuning leads to **higher and more stable entropy** in intermediate layers, preserving contextual richness
* Autoregressive vision models (e.g., AIM) mirror language models with **intermediate performance peaks and entropy valleys**

####  Appendix E: Unsupervised Layer Selection

* Quality metrics like DiME and InfoNCE are used to **select high-performing layers without any labels**
* This unsupervised method achieves **an average 3% performance boost** over final-layer representations

---




<br/>
# refer format:     



@inproceedings{skean2025layer,
  title     = {Layer by Layer: Uncovering Hidden Representations in Language Models},
  author    = {Oscar Skean and Md Rifat Arefin and Dan Zhao and Niket Patel and Jalal Naghiyev and Yann LeCun and Ravid Shwartz-Ziv},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  volume    = {267},
  publisher = {PMLR},
  address   = {Vancouver, Canada},
  url       = {https://github.com/OFSkean/information_flow}
}




Skean, Oscar, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. 2025. “Layer by Layer: Uncovering Hidden Representations in Language Models.” In Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR Vol. 267, Vancouver, Canada. https://github.com/OFSkean/information_flow.




