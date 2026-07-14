---
layout: post
title:  "[2026]Geoparsing: Diagram Parsing for Plane and Solid Geometry with a Unified Formal Language"
date:   2026-07-14 00:57:55 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 평면 기하학과 입체 기하학을 통합한 통일된 형식 언어를 설계하고, 이를 기반으로 한 GDP-29K 데이터셋을 구축하여 기하 도형 파싱의 성능을 향상시켰습니다.


짧은 요약(Abstract) :


이 논문에서는 다중 모달 대형 언어 모델(MLLMs)이 기하학적 추론에서 겪는 어려움을 해결하기 위해 통합된 형식 언어를 설계하고, 평면 기하학과 입체 기하학을 아우르는 새로운 접근 방식을 제안합니다. 연구진은 GDP-29K라는 대규모 데이터셋을 구축하여 20,000개의 평면 기하학 샘플과 9,000개의 입체 기하학 샘플을 포함하고 있으며, 각 샘플은 정답 형식 설명과 쌍을 이루고 있습니다. 이 데이터셋은 다양한 실제 소스에서 수집되었으며, 기하학적 구조와 의미적 관계를 포괄적으로 다룹니다. 연구진은 감독된 미세 조정(Supervised Fine-Tuning)과 검증 가능한 보상을 통한 강화 학습(Reinforcement Learning via Verifiable Rewards)을 결합한 훈련 패러다임을 제안하여 구문적 정확성과 기하학적 일관성을 보장합니다. 실험 결과, 이 접근 방식은 최첨단 구문 분석 성능을 달성하며, 파싱된 형식 설명이 MLLMs의 기하학적 추론 능력을 크게 향상시키는 데 기여함을 보여줍니다.



This paper addresses the challenges faced by Multimodal Large Language Models (MLLMs) in geometric reasoning by designing a unified formal language that integrates both plane and solid geometry. The authors construct a large-scale dataset called GDP-29K, which comprises 20,000 plane geometry samples and 9,000 solid geometry samples, each paired with its ground-truth formal description. This dataset is collected from diverse real-world sources and comprehensively covers geometric structures and semantic relations. The authors propose a training paradigm that combines Supervised Fine-Tuning with Reinforcement Learning via Verifiable Rewards to ensure syntactic correctness and geometric consistency. Experimental results demonstrate that this approach achieves state-of-the-art parsing performance and that the parsed formal descriptions significantly enhance the capabilities of MLLMs for geometric reasoning tasks.


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


이 논문에서는 기하학적 도형을 파싱하기 위한 새로운 방법론을 제안합니다. 이 방법론은 두 가지 주요 구성 요소로 이루어져 있습니다: 통합된 형식 언어와 대규모 데이터셋인 GDP-29K입니다. 이 두 가지 요소는 기하학적 문제 해결을 위한 인공지능 모델의 성능을 향상시키기 위해 설계되었습니다.

1. **통합된 형식 언어**: 이 언어는 평면 기하학과 입체 기하학을 아우르는 구조를 가지고 있습니다. 기존의 평면 기하학 형식 언어를 확장하여 입체 기하학의 복잡한 구조를 설명할 수 있도록 설계되었습니다. 이 언어는 점, 선, 면, 그리고 고차원 구조와 같은 기하학적 원소를 포함하며, 이들 간의 관계를 명확하게 정의합니다. 이를 통해 모델이 기하학적 도형을 보다 정확하게 이해하고 해석할 수 있도록 돕습니다.

2. **GDP-29K 데이터셋**: 이 데이터셋은 20,000개의 평면 기하학 샘플과 9,000개의 입체 기하학 샘플로 구성되어 있습니다. 각 샘플은 실제 세계에서 수집된 다양한 도형 이미지로, 해당 도형의 형식적 설명과 쌍을 이루고 있습니다. 데이터셋은 손으로 그린 도형과 인쇄된 도형을 모두 포함하여 데이터의 다양성을 높였습니다. 이 데이터셋은 기하학적 도형 파싱을 위한 첫 번째 대규모 데이터셋으로, 입체 기하학 분야의 데이터 부족 문제를 해결하는 데 기여합니다.

3. **훈련 방법론**: 모델은 두 단계의 훈련 과정을 거칩니다. 첫 번째 단계는 감독된 미세 조정(Supervised Fine-Tuning, SFT)으로, 모델이 형식 언어의 기본 구문을 학습하도록 합니다. 두 번째 단계는 검증 가능한 보상을 통한 강화 학습(Reinforcement Learning via Verifiable Rewards, RLVR)으로, 생성된 형식 설명의 구문적 정확성과 기하학적 일관성을 보장합니다. 이 두 단계의 훈련 방법론은 모델이 기하학적 도형을 보다 정확하게 파싱할 수 있도록 돕습니다.

4. **실험 결과**: 제안된 방법론은 기존의 최첨단 모델들보다 우수한 성능을 보였으며, 기하학적 추론 작업에서 모델의 능력을 크게 향상시켰습니다. 특히, 파싱된 형식 설명은 후속 기하학적 추론 작업에서 중요한 인지적 지주 역할을 하여, 다양한 기하학적 문제를 해결하는 데 기여합니다.

이러한 방법론은 기하학적 문제 해결을 위한 인공지능 모델의 성능을 크게 향상시키며, 향후 연구에 중요한 기초 자료를 제공합니다.

---



This paper proposes a novel methodology for parsing geometric figures, consisting of two main components: a unified formal language and a large-scale dataset called GDP-29K. These two elements are designed to enhance the performance of artificial intelligence models in solving geometric problems.

1. **Unified Formal Language**: This language encompasses structures for both plane and solid geometry. It extends existing formal languages for plane geometry to describe the complex structures of solid geometry. The language includes geometric primitives such as points, lines, planes, and higher-order structures, clearly defining the relationships between them. This helps models better understand and interpret geometric figures.

2. **GDP-29K Dataset**: This dataset comprises 20,000 plane geometry samples and 9,000 solid geometry samples. Each sample consists of various geometric images collected from real-world sources, paired with their formal descriptions. The dataset includes both hand-drawn and printed figures, enhancing data diversity. It is the first large-scale dataset for geometric diagram parsing, addressing the data scarcity issue in the field of solid geometry.

3. **Training Methodology**: The model undergoes a two-stage training process. The first stage is Supervised Fine-Tuning (SFT), where the model learns the basic syntax of the formal language. The second stage is Reinforcement Learning via Verifiable Rewards (RLVR), which ensures the syntactic correctness and geometric consistency of the generated formal descriptions. This two-stage training methodology aids the model in accurately parsing geometric figures.

4. **Experimental Results**: The proposed methodology demonstrates superior performance compared to existing state-of-the-art models, significantly enhancing the model's capabilities in geometric reasoning tasks. The parsed formal descriptions serve as a critical cognitive scaffold, contributing to solving various geometric problems.

This methodology greatly improves the performance of AI models in geometric problem-solving and provides essential foundational resources for future research.


<br/>
# Results


이 논문에서는 기하학적 도형을 파싱하는 새로운 방법론을 제안하고, 이를 통해 생성된 GDP-29K 데이터셋을 활용하여 다양한 모델의 성능을 평가했습니다. 이 데이터셋은 평면 기하학과 입체 기하학을 포함하여 총 28,977개의 샘플로 구성되어 있으며, 각 샘플은 정밀한 형식적 설명과 함께 제공됩니다.

#### 경쟁 모델
논문에서는 여러 최신 모델과의 성능 비교를 수행했습니다. 특히, Qwen3-VL 시리즈와 GPT-5.2, Gemini-3-Flash와 같은 대규모 멀티모달 언어 모델(MLLMs)과의 비교가 이루어졌습니다. 이들 모델은 기하학적 문제 해결에서 뛰어난 성능을 보였지만, 여전히 기하학적 인식에서 한계를 드러냈습니다.

#### 테스트 데이터
테스트 데이터는 GDP-29K 데이터셋의 일부로, PGDP-2K(평면 기하학)와 SGDP-1K(입체 기하학)로 나뉘어 있습니다. PGDP-2K는 2,000개의 평면 기하학 샘플로 구성되어 있으며, SGDP-1K는 1,000개의 입체 기하학 샘플로 구성되어 있습니다.

#### 메트릭
모델의 성능은 Precision(정밀도), Recall(재현율), F1-score(조화 평균)와 같은 메트릭을 사용하여 평가되었습니다. 특히, F1-score는 각 기하학적 원소의 예측 정확도를 종합적으로 나타내는 지표로 사용되었습니다.

#### 성능 비교
실험 결과, 제안된 GDP-4B-RL 모델은 PGDP-2K에서 96.4의 F1-score를 기록하며, 기존의 모든 모델을 초월하는 성능을 보였습니다. 특히, 기본 원소(Points)와 같은 기초적인 인식에서는 높은 정확도를 보였지만, Lines와 Semantics와 같은 복잡한 구조에서는 상대적으로 낮은 성능을 보였습니다. 반면, SGDP-1K에서는 94.9의 F1-score를 기록하며, 입체 기하학에서의 인식 능력을 크게 향상시켰습니다.

이러한 결과는 제안된 형식적 언어와 훈련 방법론이 기하학적 인식의 정확성을 크게 향상시켰음을 보여줍니다. 특히, RLVR(Reinforcement Learning via Verifiable Rewards) 방법론이 높은 구조적 정확성을 유지하는 데 기여했음을 확인할 수 있었습니다.






This paper proposes a novel approach for parsing geometric figures and evaluates the performance of various models using the newly generated GDP-29K dataset. This dataset consists of a total of 28,977 samples, encompassing both plane and solid geometry, with each sample paired with precise formal descriptions.

#### Competing Models
The paper conducts performance comparisons with several state-of-the-art models, particularly large multimodal language models (MLLMs) such as the Qwen3-VL series, GPT-5.2, and Gemini-3-Flash. While these models demonstrated impressive capabilities in solving geometric problems, they still exhibited limitations in geometric perception.

#### Test Data
The test data is a subset of the GDP-29K dataset, divided into PGDP-2K (plane geometry) and SGDP-1K (solid geometry). PGDP-2K consists of 2,000 plane geometry samples, while SGDP-1K comprises 1,000 solid geometry samples.

#### Metrics
The performance of the models was evaluated using metrics such as Precision, Recall, and F1-score. In particular, the F1-score was used as a comprehensive indicator of the prediction accuracy for each geometric primitive.

#### Performance Comparison
The experimental results showed that the proposed GDP-4B-RL model achieved an F1-score of 96.4 on PGDP-2K, surpassing all existing models. Notably, while it demonstrated high accuracy on basic primitives (Points), it exhibited relatively lower performance on more complex structures like Lines and Semantics. In contrast, it recorded an F1-score of 94.9 on SGDP-1K, significantly enhancing its recognition capabilities in solid geometry.

These results indicate that the proposed formal language and training methodology greatly improved the accuracy of geometric perception. Specifically, the Reinforcement Learning via Verifiable Rewards (RLVR) methodology contributed to maintaining high structural accuracy.


<br/>
# 예제


이 논문에서는 기하학 다이어그램 파싱을 위한 GDP-29K 데이터셋을 구축하고, 이를 통해 기하학적 문제 해결을 위한 새로운 접근 방식을 제안합니다. 이 데이터셋은 평면 기하학과 입체 기하학을 포함하며, 각 다이어그램은 정답으로 제공되는 형식적 설명과 쌍을 이루고 있습니다. 

#### 트레이닝 데이터 예시
트레이닝 데이터는 다음과 같은 형식으로 구성됩니다:

- **입력 (Input)**: 기하학 다이어그램 이미지와 해당 다이어그램에 대한 질문
- **출력 (Output)**: 다이어그램의 기하학적 요소를 설명하는 형식적 언어

예를 들어, 다음과 같은 다이어그램이 있을 수 있습니다:

- **다이어그램**: ABCD는 정사각형이며, AC의 길이는 18입니다. 
- **질문**: x의 값을 구하시오. (여기서 m∠ABC = 5x)

이 경우, 모델은 다음과 같은 형식적 설명을 생성해야 합니다:

- **출력**:
  - "points": ["A", "B", "C", "D"]
  - "lines": ["line A B", "line A D", "line D C", "line B C", "line A C"]
  - "semantics": ["AC = 18", "m ∠ABC = 5x"]

#### 테스트 데이터 예시
테스트 데이터는 모델의 성능을 평가하기 위해 사용됩니다. 예를 들어:

- **입력**: HIJK는 원에 내접하는 사각형입니다. x의 값을 구하시오.
- **출력**: 
  - "points": ["H", "I", "J", "K"]
  - "lines": ["line H S I", "line I T J", "line K U J", "line H R K"]
  - "semantics": ["IS = KR = 5", "HR = JU = 13", "JT = x"]

이와 같은 방식으로, 모델은 기하학적 관계를 이해하고, 주어진 질문에 대한 답을 도출할 수 있습니다.




This paper proposes a new approach to geometric problem-solving by constructing the GDP-29K dataset for geometry diagram parsing. This dataset includes both plane and solid geometry, with each diagram paired with its ground-truth formal description.

#### Training Data Example
The training data is structured as follows:

- **Input**: A geometric diagram image and a question related to that diagram.
- **Output**: A formal language description of the geometric elements in the diagram.

For example, consider the following diagram:

- **Diagram**: ABCD is a square, and the length of AC is 18.
- **Question**: Find the value of x. (where m∠ABC = 5x)

In this case, the model should generate the following formal description:

- **Output**:
  - "points": ["A", "B", "C", "D"]
  - "lines": ["line A B", "line A D", "line D C", "line B C", "line A C"]
  - "semantics": ["AC = 18", "m ∠ABC = 5x"]

#### Test Data Example
The test data is used to evaluate the model's performance. For instance:

- **Input**: HIJK is a quadrilateral circumscribed about a circle. Find the value of x.
- **Output**:
  - "points": ["H", "I", "J", "K"]
  - "lines": ["line H S I", "line I T J", "line K U J", "line H R K"]
  - "semantics": ["IS = KR = 5", "HR = JU = 13", "JT = x"]

In this way, the model can understand geometric relationships and derive answers to the given questions.

<br/>
# 요약


이 논문에서는 평면 기하학과 입체 기하학을 통합한 통일된 형식 언어를 설계하고, 이를 기반으로 한 GDP-29K 데이터셋을 구축하여 기하 도형 파싱의 성능을 향상시켰습니다. 실험 결과, 제안된 방법이 기존의 최첨단 모델보다 우수한 성능을 보였으며, 파싱된 형식 설명이 기하학적 추론 작업에서 중요한 인지적 지지대 역할을 한다는 것을 입증했습니다. 이 연구는 기하학 문제 해결에서의 인식 병목 현상을 해결하는 데 기여하고 있습니다.

---

This paper designs a unified formal language that integrates plane and solid geometry, constructing the GDP-29K dataset to enhance the performance of geometric diagram parsing. Experimental results demonstrate that the proposed method outperforms existing state-of-the-art models, showing that the parsed formal descriptions serve as a critical cognitive scaffold in geometry reasoning tasks. This research contributes to addressing the perception bottleneck in geometric problem-solving.

<br/>
# 기타




#### 다이어그램
논문에서는 다양한 기하학적 다이어그램을 사용하여 평면 기하학과 입체 기하학의 구조를 설명합니다. 각 다이어그램은 기하학적 요소(점, 선, 면 등)와 그들 간의 관계를 명확하게 나타내며, 이를 통해 모델이 기하학적 문제를 이해하고 해결하는 데 필요한 정보를 제공합니다. 다이어그램은 GDP-29K 데이터셋의 예시로 사용되며, 각 다이어그램은 해당하는 형식적 설명과 함께 제공됩니다.

#### 피규어
피규어는 기하학적 구조의 시각적 표현을 제공합니다. 예를 들어, 피규어 1에서는 최신 MLLM 모델들이 기하학적 다이어그램을 파싱하는 데 어려움을 겪는 모습을 보여줍니다. 이러한 피규어는 모델의 성능을 시각적으로 비교하고, 기하학적 문제 해결의 복잡성을 강조하는 데 중요한 역할을 합니다.

#### 테이블
테이블은 다양한 모델의 성능을 비교하는 데 사용됩니다. 예를 들어, 테이블 3과 4에서는 PGDP-2K 및 SGDP-1K 테스트 벤치마크에서의 모델 성능을 정량적으로 나타냅니다. 이러한 데이터는 각 모델의 강점과 약점을 파악하는 데 유용하며, 연구 결과의 신뢰성을 높이는 데 기여합니다.

#### 어펜딕스
어펜딕스에서는 GDP-29K 데이터셋의 수집 과정, 형식적 언어의 구문, 통계 분석 등을 자세히 설명합니다. 이 부분은 연구의 방법론과 데이터의 질을 보장하는 데 중요한 정보를 제공합니다. 예를 들어, 데이터 수집 과정에서의 필터링 및 주석 작업의 세부 사항은 데이터셋의 신뢰성을 높이는 데 기여합니다.

---




#### Diagrams
The paper utilizes various geometric diagrams to illustrate the structures of plane and solid geometry. Each diagram clearly represents geometric elements (points, lines, planes, etc.) and their relationships, providing the necessary information for the model to understand and solve geometric problems. The diagrams serve as examples from the GDP-29K dataset, with each diagram paired with its corresponding formal description.

#### Figures
Figures provide visual representations of geometric structures. For instance, Figure 1 shows the difficulties faced by state-of-the-art MLLM models in parsing geometric diagrams. These figures play a crucial role in visually comparing model performance and emphasizing the complexity of geometric problem-solving.

#### Tables
Tables are used to compare the performance of various models. For example, Tables 3 and 4 quantitatively present the performance of models on the PGDP-2K and SGDP-1K test benchmarks. This data is useful for identifying the strengths and weaknesses of each model and contributes to the reliability of the research findings.

#### Appendix
The appendix provides detailed explanations of the GDP-29K dataset's collection process, the syntax of the formal language, and statistical analyses. This section offers important information for ensuring the methodology and quality of the data. For example, details on the filtering and annotation processes during data collection contribute to the dataset's reliability.

<br/>
# refer format:




### BibTeX 
```bibtex
@inproceedings{Wang2026,
  author    = {Peijie Wang and Ming-Liang Zhang and Jun Cao and Chao Deng and Dekang Ran and Hongda Sun and Pi Bu and Xuan Zhang and Yingyao Wang and Jun Song and Bo Zheng and Fei Yin and Cheng-Lin Liu},
  title     = {Geoparsing: Diagram Parsing for Plane and Solid Geometry with a Unified Formal Language},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {29876--29903},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  address   = {July 2-7, 2026}
}
```

### 시카고 스타일 
Wang, Peijie, Ming-Liang Zhang, Jun Cao, Chao Deng, Dekang Ran, Hongda Sun, Pi Bu, Xuan Zhang, Yingyao Wang, Jun Song, Bo Zheng, Fei Yin, and Cheng-Lin Liu. "Geoparsing: Diagram Parsing for Plane and Solid Geometry with a Unified Formal Language." In *Findings of the Association for Computational Linguistics: ACL 2026*, 29876–29903. Association for Computational Linguistics, July 2-7, 2026.
    