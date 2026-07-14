---
layout: post
title:  "[2026]The Digital Dunning-Kruger Effect: Decoupling Hallucinations via Geometric Hidden-state Observation for Semantic Truthfulness"
date:   2026-07-14 00:32:31 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 GHOST라는 새로운 프레임워크를 제안하여 대형 언어 모델의 환각을 탐지하는 방법을 소개합니다.


짧은 요약(Abstract) :


이 논문에서는 대형 언어 모델(LLM)이 종종 과도한 자신감으로 사실과 다른 환각을 생성하는 문제를 다룹니다. 현재의 환각 탐지 방법들은 계산 비용이 높은 블랙박스 방법의 높은 정확도와 화이트박스 방법의 한계를 겪고 있습니다. 이를 해결하기 위해, 저자들은 GHOST(Geometric Hidden-state Observation for Semantic Truthfulness)라는 효율적인 화이트박스 프레임워크를 제안합니다. GHOST는 내부 추론의 불안정성으로 표시된 혼란스러운 환각을 주로 목표로 하며, 조기 수렴으로 특징지어지는 고집스러운 환각을 보완 신호로 포착합니다. 내부 기하학적 동역학과 출력 확률 분포를 통합하여 GHOST는 비선형 진실성 분류를 위한 고차원 특성 공간을 구성합니다. FinanceBench, RAGTruth, HaluEval, PopQA에 대한 광범위한 평가 결과, GHOST는 화이트박스 기준선을 초과하고 블랙박스 성능과 경쟁하며 90% 이상의 계산 오버헤드를 줄여 실시간 탐지를 위한 강력한 솔루션을 제공합니다.



This paper addresses the issue of large language models (LLMs) occasionally generating factually incorrect hallucinations with excessive confidence. Current detection methods face a trade-off between the high accuracy of computationally expensive black-box methods and the limitations of white-box methods. To bridge this gap, the authors propose GHOST (Geometric Hidden-state Observation for Semantic Truthfulness), an efficient white-box framework for hallucination detection. GHOST primarily targets confused hallucinations marked by internal reasoning instability while also capturing stubborn hallucinations characterized by premature convergence as a complementary signal. By integrating internal geometric dynamics with output probability distributions, GHOST constructs a high-dimensional feature space for non-linear truthfulness classification. Extensive evaluations on FinanceBench, RAGTruth, HaluEval, and PopQA show that GHOST outperforms white-box baselines and achieves competitive black-box performance while reducing computational overhead by over 90%, offering a robust solution for real-time detection.


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



**GHOST(Geometric Hidden-state Observation for Semantic Truthfulness) 프레임워크**

GHOST는 대형 언어 모델(LLM)의 환각(hallucination) 탐지를 위한 효율적인 화이트박스(white-box) 프레임워크입니다. 이 프레임워크는 두 가지 주요 메커니즘인 '혼란스러운 환각(Confused Hallucinations)'과 '고집스러운 환각(Stubborn Hallucinations)'을 구분하여 탐지합니다. 혼란스러운 환각은 내부 추론의 불안정성으로 나타나며, 고집스러운 환각은 충분한 사실적 근거 없이 조기 수렴을 특징으로 합니다.

#### 1. 모델 아키텍처
GHOST는 Transformer 아키텍처를 기반으로 하며, 내부 상태의 기하학적 동역학을 활용하여 응답의 진실성을 평가합니다. 이 프레임워크는 각 레이어의 숨겨진 상태(hidden state)에서 기하학적 특성을 추출하여, 모델의 추론 과정에서 발생하는 동적 경로를 분석합니다.

#### 2. 특징 추출
GHOST는 다음과 같은 주요 특징을 추출합니다:
- **표현의 불안정성(Representation Turbulence)**: 모델이 내부 정보의 모순을 해결하는 데 어려움을 겪을 때 발생하는 기하학적 변화입니다. 이는 인접 레이어 간의 숨겨진 상태의 코사인 유사도를 기반으로 측정됩니다.
- **고집스러움(Stubbornness)**: 모델이 잘못된 결론에 조기 수렴하는 경향을 나타내며, 이는 중간 레이어 상태와 최종 레이어 표현 간의 유사도를 통해 평가됩니다.
- **예측 엔트로피(Predictive Entropy)**: 출력 확률 분포의 불확실성을 측정합니다.
- **의미적 분산(Semantic Divergence)**: 후보 토큰 간의 기하학적 분산을 측정하여, 의미적으로 혼란스러운 예측을 구별합니다.

#### 3. 훈련 및 평가
GHOST는 다양한 데이터셋(예: FinanceBench, RAGTruth, HaluEval, PopQA)에서 훈련 및 평가됩니다. 각 데이터셋은 고유한 도메인 특성을 가지고 있으며, GHOST는 이들 데이터셋에서의 성능을 비교하여 환각 탐지의 효율성을 입증합니다. GHOST는 기존의 화이트박스 방법보다 우수한 성능을 보이며, 블랙박스 방법과 경쟁할 수 있는 성능을 유지하면서도 90% 이상의 계산 오버헤드를 줄입니다.

#### 4. 효율성
GHOST는 단일 전방 패스를 통해 내부 메트릭을 계산하는 벡터화된 추출 메커니즘을 사용하여, 실시간 배포에 적합한 효율성을 제공합니다. 이는 기존의 샘플링 기반 방법보다 훨씬 낮은 지연 시간을 자랑합니다.




**GHOST (Geometric Hidden-state Observation for Semantic Truthfulness) Framework**

GHOST is an efficient white-box framework for hallucination detection in large language models (LLMs). This framework distinguishes between two main mechanisms: 'Confused Hallucinations' and 'Stubborn Hallucinations'. Confused hallucinations manifest as internal reasoning instability, while stubborn hallucinations are characterized by premature convergence without sufficient factual grounding.

#### 1. Model Architecture
GHOST is based on the Transformer architecture and utilizes the geometric dynamics of internal states to assess the truthfulness of responses. The framework analyzes the dynamic trajectories that occur during the model's inference process by extracting geometric features from the hidden states of each layer.

#### 2. Feature Extraction
GHOST extracts the following key features:
- **Representation Turbulence**: This represents the geometric changes that occur when the model struggles to resolve contradictions in internal information. It is measured based on the cosine similarity between hidden states across adjacent layers.
- **Stubbornness**: This indicates the model's tendency to prematurely converge to incorrect conclusions, evaluated through the similarity between intermediate layer states and the final layer representation.
- **Predictive Entropy**: This measures the uncertainty of the output probability distribution.
- **Semantic Divergence**: This measures the geometric dispersion among candidate tokens to distinguish semantically confused predictions.

#### 3. Training and Evaluation
GHOST is trained and evaluated on various datasets (e.g., FinanceBench, RAGTruth, HaluEval, PopQA). Each dataset has unique domain characteristics, and GHOST demonstrates its effectiveness in hallucination detection by comparing performance across these datasets. GHOST outperforms existing white-box methods and maintains competitive performance with black-box methods while reducing computational overhead by over 90%.

#### 4. Efficiency
GHOST employs a fully vectorized extraction mechanism integrated into the model's single forward pass, providing efficiency suitable for real-time deployment. This results in significantly lower latency compared to existing sampling-based methods.


<br/>
# Results



이 논문에서는 GHOST(Geometric Hidden-state Observation for Semantic Truthfulness)라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLM)의 환각(hallucination) 탐지 문제를 해결하고자 하였습니다. GHOST는 내부 기하학적 동역학을 활용하여 모델의 응답이 사실적으로 정확한지를 평가하는 데 중점을 두고 있습니다. 

#### 실험 결과 요약

1. **경쟁 모델**: GHOST는 여러 기존 모델과 비교되었습니다. 여기에는 Predictive Entropy, INSIDE, LI, SelfCheckGPT와 같은 다양한 (black-box) 및 (white-box) 방법이 포함되었습니다.

2. **테스트 데이터**: GHOST는 FinanceBench, RAGTruth, HaluEval, PopQA의 네 가지 주요 벤치마크에서 평가되었습니다. 각 데이터셋은 서로 다른 환각의 특성을 평가하기 위해 설계되었습니다.
   - **FinanceBench**: 금융 관련 질문을 통해 모델의 정확성을 평가합니다.
   - **RAGTruth**: 외부 문서와의 통합에서 발생하는 환각을 평가합니다.
   - **HaluEval**: 일반 지식에 대한 질문을 포함하여 모델의 기본 사실 정렬을 평가합니다.
   - **PopQA**: 저명하지 않은 엔티티에 대한 질문을 통해 모델의 기억 능력을 평가합니다.

3. **메트릭**: GHOST의 성능은 AUPRC(Precision-Recall Curve의 면적)와 F1-score로 측정되었습니다. AUPRC는 클래스 불균형이 있는 데이터셋에서 더 민감하게 반응하는 지표로, 환각 탐지의 효과성을 평가하는 데 적합합니다.

4. **비교 결과**: GHOST는 모든 평가된 LLM 아키텍처와 벤치마크에서 최첨단 성능을 달성했습니다. 예를 들어, Qwen2.5-1.5B 모델에서 GHOST는 0.9801의 AUPRC를 기록하며, 기존의 화이트박스 및 블랙박스 방법들을 초월하는 성능을 보였습니다. DeepSeek-R1-7B 모델에서도 GHOST는 평균 AUPRC 0.9819를 기록하여 SelfCheckGPT를 초과하는 성능을 나타냈습니다.

5. **효율성**: GHOST는 기존의 방법들에 비해 90% 이상의 계산 오버헤드를 줄이면서도 경쟁력 있는 성능을 유지했습니다. 이는 GHOST가 실시간 배포에 적합하다는 것을 의미합니다.

6. **결론**: GHOST는 LLM의 환각 탐지에서 효과적이며, 기하학적 동역학을 활용하여 모델의 행동을 정밀하게 분석할 수 있는 방법을 제시합니다. 이 연구는 LLM의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.

---




This paper proposes a new framework called GHOST (Geometric Hidden-state Observation for Semantic Truthfulness) to address the hallucination detection problem in large language models (LLMs). GHOST focuses on evaluating the factual accuracy of model responses by leveraging internal geometric dynamics.

#### Summary of Experimental Results

1. **Competing Models**: GHOST was compared with various existing models, including Predictive Entropy, INSIDE, LI, and SelfCheckGPT, encompassing both black-box and white-box methods.

2. **Test Data**: GHOST was evaluated on four major benchmarks: FinanceBench, RAGTruth, HaluEval, and PopQA. Each dataset was designed to assess different aspects of hallucination.
   - **FinanceBench**: Evaluates the model's accuracy on financial-related questions.
   - **RAGTruth**: Assesses hallucinations occurring in the context of external document integration.
   - **HaluEval**: Contains general knowledge questions to evaluate the model's fundamental factual alignment.
   - **PopQA**: Challenges the model's ability to recall less prominent entities.

3. **Metrics**: The performance of GHOST was measured using AUPRC (Area Under the Precision-Recall Curve) and F1-score. AUPRC is particularly sensitive to class imbalances, making it suitable for evaluating the effectiveness of hallucination detection.

4. **Comparison Results**: GHOST achieved state-of-the-art performance across all evaluated LLM architectures and benchmarks. For instance, on the Qwen2.5-1.5B model, GHOST recorded an AUPRC of 0.9801, surpassing existing white-box and black-box methods. On the DeepSeek-R1-7B model, GHOST also achieved an average AUPRC of 0.9819, exceeding the performance of SelfCheckGPT.

5. **Efficiency**: GHOST reduced computational overhead by over 90% compared to existing methods while maintaining competitive performance. This indicates that GHOST is suitable for real-time deployment.

6. **Conclusion**: GHOST is effective in hallucination detection for LLMs and presents a method to analyze model behaviors with precision by utilizing geometric dynamics. This research is expected to contribute to enhancing the reliability of LLMs.


<br/>
# 예제


이 논문에서는 GHOST(Geometric Hidden-state Observation for Semantic Truthfulness)라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLM)의 환각(hallucination)을 탐지하는 방법을 설명합니다. GHOST는 내부 기하학적 동역학을 활용하여 모델의 응답이 사실적으로 정확한지를 평가합니다. 이 프레임워크는 두 가지 주요 메트릭인 'Representation Turbulence'와 'Stubbornness'를 사용하여 환각을 구분합니다.

#### 데이터셋 및 실험 설정

1. **HaluEval**: 일반 지식에 대한 질문을 포함하는 데이터셋으로, 총 10,000개의 샘플이 있습니다. 이 데이터셋은 모델의 기본적인 사실성 정렬을 평가하는 데 사용됩니다.
   - **입력 예시**: "지구의 가장 큰 대륙은 무엇인가요?"
   - **출력 예시**: "아시아입니다." (정답) / "남극입니다." (환각)

2. **PopQA**: 저명하지 않은 엔티티에 대한 질문을 포함하는 데이터셋으로, 총 1,400개의 샘플이 있습니다. 이 데이터셋은 모델이 내부 지식과 외부 메모리를 구분하는 능력을 평가합니다.
   - **입력 예시**: "가장 인기 없는 영화는 무엇인가요?"
   - **출력 예시**: "영화 A입니다." (정답) / "영화 B입니다." (환각)

3. **FinanceBench**: 금융 관련 질문을 포함하는 데이터셋으로, 총 1,200개의 샘플이 있습니다. 이 데이터셋은 모델의 정확한 수치 및 개념적 정확성을 평가합니다.
   - **입력 예시**: "2023년의 미국 GDP는 얼마인가요?"
   - **출력 예시**: "21조 달러입니다." (정답) / "15조 달러입니다." (환각)

4. **RAGTruth**: 외부 문서와 관련된 질문을 포함하는 데이터셋으로, 총 2,500개의 샘플이 있습니다. 이 데이터셋은 모델이 제공된 문서에 따라 환각을 탐지하는 능력을 평가합니다.
   - **입력 예시**: "이 문서에서 언급된 주요 사건은 무엇인가요?"
   - **출력 예시**: "사건 X입니다." (정답) / "사건 Y입니다." (환각)

각 데이터셋은 80%의 훈련 데이터와 20%의 테스트 데이터로 나누어져 있으며, GHOST는 이 데이터를 사용하여 모델의 환각 탐지 성능을 평가합니다.




This paper introduces a new framework called GHOST (Geometric Hidden-state Observation for Semantic Truthfulness) to detect hallucinations in large language models (LLMs). GHOST utilizes internal geometric dynamics to assess whether the model's responses are factually accurate. The framework employs two main metrics: 'Representation Turbulence' and 'Stubbornness' to differentiate between types of hallucinations.

#### Datasets and Experimental Setup

1. **HaluEval**: A dataset containing questions about general knowledge, with a total of 10,000 samples. This dataset is used to evaluate the model's fundamental factual alignment.
   - **Input Example**: "What is the largest continent on Earth?"
   - **Output Example**: "Asia." (Correct) / "Antarctica." (Hallucination)

2. **PopQA**: A dataset focused on questions about less well-known entities, containing a total of 1,400 samples. This dataset assesses the model's ability to distinguish between internal knowledge and external memory.
   - **Input Example**: "What is the least popular movie?"
   - **Output Example**: "Movie A." (Correct) / "Movie B." (Hallucination)

3. **FinanceBench**: A dataset containing financial questions, with a total of 1,200 samples. This dataset evaluates the model's accuracy in numerical and conceptual reasoning.
   - **Input Example**: "What is the GDP of the United States in 2023?"
   - **Output Example**: "21 trillion dollars." (Correct) / "15 trillion dollars." (Hallucination)

4. **RAGTruth**: A dataset that includes questions related to external documents, with a total of 2,500 samples. This dataset assesses the model's ability to detect hallucinations based on the provided context.
   - **Input Example**: "What is the main event mentioned in this document?"
   - **Output Example**: "Event X." (Correct) / "Event Y." (Hallucination)

Each dataset is split into 80% training data and 20% testing data, and GHOST uses this data to evaluate the model's hallucination detection performance.

<br/>
# 요약


이 논문에서는 GHOST라는 새로운 프레임워크를 제안하여 대형 언어 모델의 환각을 탐지하는 방법을 소개합니다. GHOST는 내부 기하학적 동역학과 출력 확률 분포를 통합하여 고차원 특징 공간을 구성하고, 이를 통해 혼란스러운 환각과 고집스러운 환각을 효과적으로 구분합니다. 실험 결과, GHOST는 기존의 화이트박스 방법보다 우수한 성능을 보이며, 90% 이상의 계산 오버헤드를 줄이면서 실시간 탐지가 가능함을 입증했습니다.

---

This paper introduces a novel framework called GHOST for detecting hallucinations in large language models. GHOST integrates internal geometric dynamics with output probability distributions to construct a high-dimensional feature space, effectively distinguishing between confused and stubborn hallucinations. Experimental results demonstrate that GHOST outperforms existing white-box methods while reducing computational overhead by over 90%, enabling real-time detection.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1**: Manifold Deviation Analysis는 디지털 던닝-크루거 효과를 시각화하여 모델의 내부 상태가 어떻게 진화하는지를 보여줍니다. 이 그림은 혼란스러운 환각과 고집스러운 환각의 기하학적 차이를 강조합니다.
- **Figure 2**: GHOST 프레임워크의 구조를 설명하며, 내부 기하학적 동역학과 출력 확률 분포를 통합하여 혼란스러운 환각과 고집스러운 환각을 식별하는 방법을 보여줍니다.

#### 2. 테이블
- **Table 1**: 다양한 분류기 성능 비교. Random Forest가 모든 데이터셋에서 가장 우수한 성능을 보이며, GHOST 프레임워크의 효과를 입증합니다.
- **Table 2**: GHOST의 성능을 여러 LLM 아키텍처와 벤치마크에서 비교한 결과, GHOST가 기존의 화이트박스 및 블랙박스 방법보다 우수한 성능을 보임을 나타냅니다.
- **Table 3**: GHOST의 각 기능 구성 요소의 효과를 검증하기 위한 ablation study 결과. Representation Turbulence가 가장 중요한 요소로 나타났습니다.
- **Table 4**: GHOST의 일반화 성능을 평가한 결과, 다양한 데이터셋에서 안정적인 성능을 보였습니다.
- **Table 5**: GHOST의 효율성을 보여주는 데이터로, 추가 지연 시간이 매우 적어 실시간 배포에 적합함을 나타냅니다.

#### 3. 어펜딕스
- **A.1**: 데이터셋 분석 및 특성. HaluEval, PopQA, FinanceBench, RAGTruth의 통계적 분포와 특성을 설명합니다.
- **A.2**: 실험에 사용된 컴퓨팅 자원 및 환경에 대한 설명.
- **A.3**: 모델 구성 및 레이어 선택 전략. GHOST가 가장 대표적인 내부 추론 동역학을 포착하기 위해 동적 레이어 선택 전략을 적용했음을 설명합니다.
- **A.4**: 훈련 프로토콜 및 분류기 최적화 방법. 클래스 불균형 문제를 해결하기 위한 방법과 하이퍼파라미터 최적화 방법을 설명합니다.

### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Figure 1**: The Manifold Deviation Analysis visualizes the Digital Dunning-Kruger Effect, illustrating how the internal states of the model evolve. This figure emphasizes the geometric differences between confused hallucinations and stubborn hallucinations.
- **Figure 2**: It describes the architecture of the GHOST framework, showing how it integrates internal geometric dynamics with output probability distributions to identify confused and stubborn hallucinations.

#### 2. Tables
- **Table 1**: Comparison of classifier performance. Random Forest consistently outperforms other classifiers across all datasets, demonstrating the effectiveness of the GHOST framework.
- **Table 2**: Performance comparison of GHOST across various LLM architectures and benchmarks, indicating that GHOST outperforms existing white-box and black-box methods.
- **Table 3**: Results of the ablation study verifying the effectiveness of each feature component in GHOST, with Representation Turbulence being the most critical.
- **Table 4**: Evaluation of GHOST's generalization performance, showing stable performance across diverse datasets.
- **Table 5**: Efficiency analysis of GHOST, indicating minimal additional latency, making it suitable for real-time deployment.

#### 3. Appendix
- **A.1**: Detailed analysis and characteristics of the datasets used, including statistical distributions for HaluEval, PopQA, FinanceBench, and RAGTruth.
- **A.2**: Description of the computational resources and environment used for experiments.
- **A.3**: Model configuration and layer selection strategy, explaining the dynamic layer selection approach to capture representative internal reasoning dynamics.
- **A.4**: Training protocol and classifier optimization methods, detailing how class imbalance was addressed and hyperparameter optimization was conducted.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Mao2026,
  author    = {Yueheng Mao and Min Yu and Gengwang Li and Jianguo Jiang and Gang Li and Meng Zhang and Zhen Xu and Weiqing Huang and Ming Liu},
  title     = {The Digital Dunning-Kruger Effect: Decoupling Hallucinations via Geometric Hidden-state Observation for Semantic Truthfulness},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {21786--21800},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},
}
```

### 시카고 스타일

Yueheng Mao, Min Yu, Gengwang Li, Jianguo Jiang, Gang Li, Meng Zhang, Zhen Xu, Weiqing Huang, and Ming Liu. "The Digital Dunning-Kruger Effect: Decoupling Hallucinations via Geometric Hidden-state Observation for Semantic Truthfulness." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 21786–21800. July 2026. Association for Computational Linguistics.
    