---
layout: post
title:  "[2026]TIGER: Text-Informed Generalized Enzyme-Reaction Retrieval"
date:   2026-02-05 04:05:37 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: TIGER는 텍스트 기반의 일반화된 효소-반응 검색 프레임워크로, 효소 시퀀스에서 텍스트 의미 지식을 추출하여 효소와 생화학 반응 간의 관계를 효과적으로 매핑한다.


짧은 요약(Abstract) :


이 논문에서는 효소-반응 검색(enzyme-reaction retrieval) 문제를 다루고 있습니다. 이 문제는 효소의 특성화, 반응 메커니즘 설명, 대사 경로 및 생물 촉매의 합리적 설계를 위한 기초적인 문제입니다. 기존의 접근 방식들은 작업과 분포 간의 일반화가 부족하고, 데이터셋 분할에 민감하며, 검색 방향 간의 비대칭성이 존재하는 등의 문제를 겪고 있습니다. 이를 해결하기 위해, 우리는 TIGER라는 텍스트 기반의 일반화된 효소-반응 검색 프레임워크를 제안합니다. TIGER는 효소 서열에서 텍스트 의미 지식을 추출하여 효소와 생화학 반응 간의 다리 역할을 하는 일반화된 표현을 제공합니다. 또한, 동적 게이팅 네트워크(Dynamic Gating Network)를 설계하여 텍스트에서 파생된 지식을 시퀀스 특징과 적응적으로 융합하여 보다 일관되고 유익한 효소 표현을 가능하게 합니다. TIGER는 다양한 분포에서 기존의 최첨단 방법들을 능가하며, 강력한 견고성과 전이 가능성을 보여줍니다.




This paper addresses the problem of enzyme-reaction retrieval, which is fundamental for enzyme characterization, reaction mechanism elucidation, and the rational design of metabolic pathways and biocatalysts. Existing approaches suffer from poor generalization across tasks and distributions, high sensitivity to dataset splits, and substantial asymmetry between retrieval directions. To tackle these challenges, we present TIGER, a Text-Informed Generalized Enzyme-Reaction Retrieval framework that leverages textual semantic knowledge distilled from enzyme sequences to provide a generalized representation that bridges enzymes and biochemical reactions. Additionally, we design a Dynamic Gating Network to adaptively fuse text-derived knowledge with sequence features, enabling more consistent and informative enzyme representations. Extensive experiments demonstrate that TIGER significantly outperforms state-of-the-art baselines across diverse distributions and exhibits strong robustness and transferability across tasks.


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


TIGER(텍스트 기반 일반화 효소-반응 검색 프레임워크)는 효소와 그들이 촉매하는 반응 간의 관계를 효과적으로 모델링하기 위해 설계된 혁신적인 시스템입니다. 이 프레임워크는 두 가지 주요 구성 요소로 나뉘어 있습니다: 효소 표현 학습과 반응 표현 학습입니다.

1. **효소 표현 학습**:
   - TIGER는 효소의 아미노산 서열을 입력으로 받아, 이를 기반으로 생성된 텍스트 지식을 활용하여 효소의 표현을 강화합니다. 아미노산 서열은 사전 훈련된 단백질 언어 모델(예: ESM2)을 통해 인코딩됩니다. 이 모델은 효소의 구조적 및 진화적 정보를 포착하는 데 효과적입니다.
   - 생성된 텍스트는 PubMedBERT와 같은 도메인 특화 언어 모델을 통해 임베딩됩니다. 이 텍스트는 효소의 기능, 기질 상호작용 및 기타 중요한 세부 정보를 포함하는 설명을 제공합니다.
   - 두 가지 표현(서열 임베딩과 텍스트 임베딩)은 동적 게이팅 네트워크(Dynamic Gating Network, DGN)를 통해 결합됩니다. DGN은 각 임베딩의 신뢰성을 평가하고, 신뢰할 수 있는 정보를 강조하여 노이즈를 억제합니다. 이를 통해 효소의 표현이 더욱 견고하고 신뢰할 수 있게 됩니다.

2. **반응 표현 학습**:
   - 반응은 UniMol-3D와 같은 3D 분자 인코더를 사용하여 인코딩됩니다. 이 인코더는 반응의 기질과 생성물의 구조적 정보를 포착하여 화학적으로 의미 있는 표현을 생성합니다. 각 반응은 기질과 생성물의 임베딩을 평균하여 반응 수준의 표현으로 집계됩니다.
   - 효소와 반응의 표현은 구조 공유 기능 프로젝터(Structure-Shared Feature Projector)를 통해 통합된 잠재 공간으로 매핑됩니다. 이 구조는 대칭적인 정렬을 보장하고, 대조 학습을 통해 강력한 일반화를 지원합니다.

3. **대조 학습**:
   - TIGER는 대칭 대조 학습 목표를 사용하여 효소와 반응 표현을 공유 잠재 공간에서 정렬합니다. 주어진 배치의 효소-반응 쌍에 대해 코사인 유사도를 계산하고, 이를 통해 효소와 반응 간의 관계를 학습합니다. 이 과정에서 두 방향(효소-반응 및 반응-효소)에 대한 손실을 결합하여 최적화합니다.

TIGER는 ReactZyme 데이터셋을 사용하여 훈련되며, 다양한 평가 분할에서 기존의 최첨단 방법들을 초월하는 성능을 보여줍니다. 이 프레임워크는 효소-반응 검색의 방향 비대칭성과 데이터 분포의 민감성을 해결하는 데 중점을 두고 있으며, 텍스트 기반의 지식을 활용하여 효소와 반응 간의 관계를 보다 명확하게 모델링합니다.

---




TIGER (Text-Informed Generalized Enzyme-Reaction Retrieval) is an innovative system designed to effectively model the relationships between enzymes and the reactions they catalyze. This framework is divided into two main components: enzyme representation learning and reaction representation learning.

1. **Enzyme Representation Learning**:
   - TIGER takes the amino acid sequence of an enzyme as input and enhances its representation by leveraging generated textual knowledge. The amino acid sequence is encoded using a pre-trained protein language model (e.g., ESM2), which effectively captures structural and evolutionary information about the enzyme.
   - The generated text is embedded using a domain-specific language model such as PubMedBERT. This text provides descriptions that incorporate the enzyme's function, substrate interactions, and other key details.
   - The two representations (sequence embedding and text embedding) are combined through a Dynamic Gating Network (DGN). The DGN evaluates the reliability of each embedding and emphasizes trustworthy information while suppressing noise. This results in a more robust and reliable representation of the enzyme.

2. **Reaction Representation Learning**:
   - Reactions are encoded using a 3D molecular encoder like UniMol-3D. This encoder captures structural information about the substrates and products of the reaction, generating chemically meaningful representations. Each reaction's representation is aggregated by averaging the embeddings of its substrates and products.
   - The representations of enzymes and reactions are mapped into a unified latent space through a Structure-Shared Feature Projector. This structure ensures symmetric alignment and supports robust generalization through contrastive learning.

3. **Contrastive Learning**:
   - TIGER employs a symmetric contrastive learning objective to align enzyme and reaction representations in a shared latent space. For a given batch of enzyme-reaction pairs, cosine similarity is calculated, allowing the model to learn the relationships between enzymes and reactions. During this process, losses for both directions (enzyme-to-reaction and reaction-to-enzyme) are combined for optimization.

TIGER is trained using the ReactZyme dataset and demonstrates performance that surpasses existing state-of-the-art methods across various evaluation splits. This framework focuses on addressing the challenges of directional asymmetry in enzyme-reaction retrieval and sensitivity to data distribution, utilizing text-based knowledge to model the relationships between enzymes and reactions more clearly.


<br/>
# Results


TIGER는 효소-반응 검색을 위한 새로운 프레임워크로, 기존의 경쟁 모델들과 비교하여 여러 평가 지표에서 우수한 성능을 보였습니다. 이 연구에서는 ReactZyme 데이터셋을 사용하여 TIGER의 성능을 평가하였으며, 이 데이터셋은 178,000개 이상의 효소-반응 쌍을 포함하고 있습니다. 평가 지표로는 Hit@K, 평균 역순(MRR), 정밀도(Precision) 등이 사용되었습니다.

#### 1. 경쟁 모델과의 비교
TIGER는 여러 기존 모델들과 비교되었습니다. 주요 경쟁 모델로는 Bi-RNN, CLIPZyme, GNN 등이 있으며, 이들 모델은 각각 다른 방식으로 효소와 반응을 인코딩합니다. 예를 들어, Bi-RNN은 양방향 순환 신경망을 사용하여 시퀀스의 시간적 의존성을 모델링하고, CLIPZyme은 효소와 반응 간의 관계를 강화하기 위해 그래프 기반 접근 방식을 사용합니다.

#### 2. 테스트 데이터
ReactZyme 데이터셋은 세 가지 평가 분할 방식으로 구성되어 있습니다:
- **시간 기반 분할**: 훈련 데이터가 시간적으로 이전에 수집된 반응을 포함하고, 테스트 데이터는 이후에 수집된 반응으로 구성됩니다.
- **효소 유사성 기반 분할**: 테스트 효소는 훈련 효소와 시퀀스 유사성이 낮은 경우로 설정됩니다.
- **반응 유사성 기반 분할**: 테스트 반응은 훈련 데이터에 포함되지 않은 새로운 반응으로 설정됩니다.

#### 3. 성능 결과
TIGER는 모든 평가 분할에서 우수한 성능을 보였습니다. 예를 들어, 시간 기반 분할에서 TIGER는 효소-반응 검색에서 Hit@1 점수 0.581을 기록하여 Bi-RNN의 0.391보다 48% 향상된 성능을 보였습니다. 반응 유사성 기반 분할에서도 TIGER는 Hit@1 점수 0.4155를 기록하여 기존 모델들보다 월등한 성능을 나타냈습니다.

#### 4. 메트릭
- **Hit@K**: K개의 상위 결과 중에서 정답이 포함되어 있는지를 평가합니다.
- **MRR (Mean Reciprocal Rank)**: 정답이 얼마나 빨리 상위에 위치하는지를 평가합니다.
- **Precision@K**: K개의 상위 결과 중에서 얼마나 많은 정답이 포함되어 있는지를 평가합니다.

TIGER는 모든 메트릭에서 기존 모델들보다 높은 점수를 기록하며, 특히 효소-반응 검색의 양방향 일관성을 유지하는 데 강점을 보였습니다. 이러한 결과는 TIGER가 다양한 데이터 분포에서 강력한 일반화 능력을 가지고 있음을 보여줍니다.

---



TIGER is a novel framework for enzyme-reaction retrieval that demonstrates superior performance compared to existing competitive models across various evaluation metrics. This study evaluated TIGER's performance using the ReactZyme dataset, which contains over 178,000 enzyme-reaction pairs. The evaluation metrics included Hit@K, Mean Reciprocal Rank (MRR), and Precision.

#### 1. Comparison with Competitive Models
TIGER was compared with several existing models, including Bi-RNN, CLIPZyme, and GNN, each employing different methods for encoding enzymes and reactions. For instance, Bi-RNN uses a bidirectional recurrent neural network to model temporal dependencies in sequences, while CLIPZyme employs a graph-based approach to enhance the relationship between enzymes and reactions.

#### 2. Test Data
The ReactZyme dataset is structured into three evaluation splits:
- **Time-based Split**: Training data consists of reactions collected before a certain time, while test data includes reactions collected afterward.
- **Enzyme Similarity-based Split**: Test enzymes are set to have low sequence similarity with training enzymes.
- **Reaction Similarity-based Split**: Test reactions are new and not included in the training data.

#### 3. Performance Results
TIGER exhibited excellent performance across all evaluation splits. For example, in the time-based split, TIGER achieved a Hit@1 score of 0.581 in enzyme-reaction retrieval, representing a 48% improvement over Bi-RNN's score of 0.391. In the reaction similarity-based split, TIGER recorded a Hit@1 score of 0.4155, significantly outperforming existing models.

#### 4. Metrics
- **Hit@K**: Measures whether the ground truth is present in the top K retrieved results.
- **MRR (Mean Reciprocal Rank)**: Evaluates how quickly the correct item appears in the ranked list.
- **Precision@K**: Assesses how many of the top K retrieved items are correct.

TIGER consistently achieved higher scores across all metrics compared to existing models, particularly maintaining strong bidirectional consistency in enzyme-reaction retrieval. These results highlight TIGER's robust generalization ability across diverse data distributions.


<br/>
# 예제



TIGER(텍스트 기반 일반화 효소-반응 검색 프레임워크)는 효소와 그들이 촉매하는 반응 간의 관계를 학습하는 데 중점을 둡니다. 이 프레임워크는 효소와 반응 간의 쌍을 매칭하는 두 가지 방향의 검색 작업을 수행합니다: 효소에서 반응으로(E2R)와 반응에서 효소로(R2E). 

#### 예시

1. **트레이닝 데이터**:
   - **효소**: 효소 A (예: "Alcohol dehydrogenase")
   - **반응**: 반응 1 (예: "Ethanol + NAD+ → Acetaldehyde + NADH")
   - **텍스트 설명**: "Alcohol dehydrogenase는 에탄올을 아세트알데하이드로 산화시키는 효소입니다."

2. **테스트 데이터**:
   - **효소**: 효소 B (예: "Lactate dehydrogenase")
   - **반응**: 반응 2 (예: "Pyruvate + NADH → Lactate + NAD+")
   - **텍스트 설명**: "Lactate dehydrogenase는 피루브산을 젖산으로 환원시키는 효소입니다."

#### 작업 설명
- **E2R 작업**: 효소 A를 입력으로 주면, TIGER는 이 효소가 촉매하는 반응을 검색하여 반응 1을 출력합니다.
- **R2E 작업**: 반응 2를 입력으로 주면, TIGER는 이 반응을 촉매하는 효소를 검색하여 효소 B를 출력합니다.

이러한 방식으로 TIGER는 효소와 반응 간의 관계를 학습하고, 텍스트 설명을 통해 추가적인 정보를 활용하여 검색의 정확성을 높입니다. 

### English Version

TIGER (Text-Informed Generalized Enzyme-Reaction Retrieval) focuses on learning the relationships between enzymes and the reactions they catalyze. This framework performs a bidirectional retrieval task that matches pairs of enzymes and reactions: from enzyme to reaction (E2R) and from reaction to enzyme (R2E).

#### Example

1. **Training Data**:
   - **Enzyme**: Enzyme A (e.g., "Alcohol dehydrogenase")
   - **Reaction**: Reaction 1 (e.g., "Ethanol + NAD+ → Acetaldehyde + NADH")
   - **Text Description**: "Alcohol dehydrogenase is an enzyme that oxidizes ethanol to acetaldehyde."

2. **Test Data**:
   - **Enzyme**: Enzyme B (e.g., "Lactate dehydrogenase")
   - **Reaction**: Reaction 2 (e.g., "Pyruvate + NADH → Lactate + NAD+")
   - **Text Description**: "Lactate dehydrogenase is an enzyme that reduces pyruvate to lactate."

#### Task Description
- **E2R Task**: Given Enzyme A as input, TIGER retrieves the reaction it catalyzes, outputting Reaction 1.
- **R2E Task**: Given Reaction 2 as input, TIGER retrieves the enzyme that catalyzes this reaction, outputting Enzyme B.

In this way, TIGER learns the relationships between enzymes and reactions, utilizing text descriptions to enhance the accuracy of the retrieval process.

<br/>
# 요약
TIGER는 텍스트 기반의 일반화된 효소-반응 검색 프레임워크로, 효소 시퀀스에서 텍스트 의미 지식을 추출하여 효소와 생화학 반응 간의 관계를 효과적으로 매핑한다. 실험 결과, TIGER는 다양한 평가 기준에서 기존 최첨단 방법들을 초과하는 성능을 보였으며, 특히 반응 유사성 기반 분할에서 0.4155의 Hit@1을 기록하여 이전 방법들보다 거의 4배 향상된 결과를 나타냈다. 이 프레임워크는 효소-반응 검색의 효율성과 신뢰성을 높이는 데 기여하며, 실제 생화학 발견 시나리오에서의 활용 가능성을 보여준다.

---

TIGER is a text-informed generalized enzyme-reaction retrieval framework that effectively maps relationships between enzymes and biochemical reactions by extracting textual semantic knowledge from enzyme sequences. Experimental results show that TIGER outperforms existing state-of-the-art methods across various evaluation metrics, achieving a Hit@1 of 0.4155 in the reaction similarity-based split, nearly quadrupling the performance of previous methods. This framework contributes to enhancing the efficiency and reliability of enzyme-reaction retrieval, demonstrating its potential for real-world biochemical discovery scenarios.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Figure 1**: TIGER와 기존 방법들의 성능 비교를 보여줍니다. 다양한 평가 분할(시간 기반, 효소 유사성 기반, 반응 유사성 기반)에서 TIGER가 모든 경우에서 우수한 성능을 보임을 나타냅니다. 이는 TIGER의 강력한 일반화 능력을 강조합니다.

2. **Figure 2**: TIGER 프레임워크의 구조를 설명합니다. 효소 시퀀스와 생성된 텍스트 지식이 동적 게이팅 네트워크를 통해 융합되고, 반응은 3D 분자 인코더를 통해 표현됩니다. 이 두 가지 모달리티는 구조 공유 기능 프로젝터를 통해 통합된 임베딩 공간으로 매핑됩니다.

3. **Figure 3**: AI 생성 텍스트와 SwissProt(인간 검토) 텍스트 간의 코사인 유사성 분포를 보여줍니다. AI 생성 텍스트의 일부가 인간 검토 텍스트와의 유사성이 낮은 경우가 많아, 텍스트 품질 관리의 필요성을 강조합니다.

#### 테이블
1. **Table 1**: ReactZyme 데이터셋에서 TIGER와 기존 방법들의 성능 비교를 보여줍니다. TIGER는 모든 평가 분할에서 Hit@1과 MRR에서 우수한 성능을 보이며, 특히 반응 유사성 기반 분할에서 기존 방법들보다 4배 이상 높은 성능을 기록했습니다.

2. **Table 2**: 동적 게이팅 네트워크(DGN)의 유무에 따른 성능 비교를 보여줍니다. DGN이 있는 경우 모든 메트릭에서 성능이 향상되며, 이는 DGN이 노이즈를 억제하고 유용한 신호를 강조하는 데 효과적임을 나타냅니다.

3. **Table 3-14**: 다양한 평가 분할에서 TIGER의 성능을 Hit@k, Precision@k, MRR, Mean Rank 등 여러 메트릭으로 분석합니다. TIGER는 모든 메트릭에서 기존 방법들보다 우수한 성능을 보이며, 특히 반응 유사성 기반 분할에서의 성능이 두드러집니다.

#### 어펜딕스
- **Appendix D**: AI 생성 텍스트와 인간 검토 텍스트의 품질 비교를 통해 AI 생성 텍스트가 기능적 역할을 잘 포착하지만, 세부적인 생화학적 정보는 부족하다는 점을 강조합니다. 이는 AI 생성 텍스트가 대규모 학습에 유용하지만, 고위험 응용에서는 인간 검토 텍스트가 필요하다는 것을 시사합니다.

---

### English Version

#### Diagrams and Figures
1. **Figure 1**: This figure compares the performance of TIGER with existing methods. It shows that TIGER outperforms all methods across various evaluation splits (time-based, enzyme similarity-based, and reaction similarity-based), highlighting its strong generalization capability.

2. **Figure 2**: This figure illustrates the structure of the TIGER framework. It shows how enzyme sequences and generated textual knowledge are fused through a Dynamic Gating Network, while reactions are represented through a 3D molecular encoder. Both modalities are mapped into a unified embedding space via a Structure-Shared Feature Projector.

3. **Figure 3**: This figure presents the cosine similarity distribution between AI-generated text and human-reviewed SwissProt text. It emphasizes the need for text quality control, as a significant portion of AI-generated text shows low similarity to human-reviewed text.

#### Tables
1. **Table 1**: This table compares the performance of TIGER and existing methods on the ReactZyme dataset. TIGER demonstrates superior performance in Hit@1 and MRR across all evaluation splits, particularly achieving over four times higher performance in the reaction similarity-based split compared to existing methods.

2. **Table 2**: This table shows the performance comparison with and without the Dynamic Gating Network (DGN). The presence of DGN consistently improves performance across all metrics, indicating its effectiveness in suppressing noise and emphasizing useful signals.

3. **Tables 3-14**: These tables analyze the performance of TIGER across various evaluation splits using multiple metrics such as Hit@k, Precision@k, MRR, and Mean Rank. TIGER consistently outperforms existing methods across all metrics, with particularly notable performance in the reaction similarity-based split.

#### Appendix
- **Appendix D**: This section compares the quality of AI-generated text with human-reviewed text, highlighting that while AI-generated descriptions capture functional roles well, they often lack detailed biochemical information. This suggests that while AI-generated text is useful for large-scale learning, human-reviewed text is necessary for high-stakes applications.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Anonymous2026,
  title={TIGER: Text-Informed Generalized Enzyme-Reaction Retrieval},
  author={Anonymous authors},
  booktitle={Under review as a conference paper at ICLR 2026},
  year={2026},
  note={Paper under double-blind review}
}
```

### 시카고 스타일

Anonymous authors. 2026. "TIGER: Text-Informed Generalized Enzyme-Reaction Retrieval." Under review as a conference paper at ICLR 2026. Paper under double-blind review.




<br/>  
질문 및 포인트들..  


일단 다시 요약

본 논문은 양방향 효소–반응 검색(bidirectional enzyme–reaction retrieval)을 위한 텍스트 기반 정보 활용 프레임워크 TIGER를 제안한다. TIGER는 단백질 서열 임베딩에 단백질→텍스트 생성 설명을 추가하고, 텍스트 노이즈를 완화하기 위해 **동적 게이팅 네트워크(Dynamic Gating Network, DGN)**를 통해 이를 융합한다. 또한 **구조 공유 특징 프로젝터(Structure-Shared Feature Projector)**를 사용해 효소와 반응 표현을 정렬하고 대조 학습(contrastive training)을 수행한다.
ReactZyme 벤치마크에서 세 가지 분할(time-based, enzyme similarity-based, reaction similarity-based)에 대해 평가한 결과, TIGER는 기존 방법 대비 큰 성능 향상을 보이며 특히 방향성 비대칭(directional asymmetry)을 유의미하게 감소시킨다. 절제 실험(ablation)은 텍스트 지식과 DGN의 기여를 뒷받침한다.

->본 논문은 효소 표현 학습을 위한 텍스트 기반 보강 프레임워크를 제안한다. 아미노산 서열로부터 텍스트 설명을 생성하고, 이를 서열 기반 표현과 결합하여 다운스트림 성능을 향상시키는 방식이다. 저자들은 생성된 자연어 지식을 통합함으로써 단백질 표현이 강화된다고 주장하며, 게이팅된 멀티모달 아키텍처를 사용한 검색(retrieval) 중심 과제에서 서열만 사용하는 기준 모델 대비 성능 향상을 보였다고 보고한다.


강점을 보면

단백질 표현 학습과 자연어 모델링의 교차 지점에서 흥미로운 연구 방향을 탐구하고 있다.

서열 기반 표현을 보완하기 위해 텍스트 정보를 통합하는 접근은 시의적절하고 관련성이 높다.

자동 생성된 텍스트로부터 발생할 수 있는 노이즈를 완화하기 위한 게이팅 메커니즘은 합리적인 설계 선택으로 보인다.

실험 결과는 평가된 검색 과제에서 서열 전용 기준 모델 대비 일관된 성능 향상을 보여준다.


약점을 보면

정보 누수 및 공정성 문제: 단백질→텍스트 생성기가 SwissProt 기반 주석으로 학습되었을 가능성이 있어, 테스트 시점에 정답 반응 의미에 암묵적으로 접근할 위험이 있다. 논문은 주요 결과에서 AI 생성 텍스트 사용을 주장하지만, 누수를 방지하기 위한 구체적 안전장치가 충분히 문서화되어 있지 않다.

생성된 텍스트 지식에 대한 직접적인 검증 부족 문제: 본 논문의 핵심 주장은 생성된 텍스트 지식이 효소 표현을 강화한다는 것이다. 그러나 생성된 텍스트의 사실적 정확성, 생물학적 타당성, 혹은 기능적 신뢰성에 대한 직접적인 평가는 제시되지 않았다. 텍스트 품질 평가는 SwissProt 설명과의 의미적 유사도나 다운스트림 검색 성능과 같은 간접적인 지표에만 의존하고 있으며, 이는 모두 대리(proxy) 지표에 해당한다. 그 결과, 관측된 성능 향상이 실제로 의미 있는 생물학적 지식 전달에 기인한 것인지, 아니면 단순한 상관관계나 텍스트 추가로 인한 보조적 감독 효과(supervision effect)에 따른 것인지 명확하지 않다.

**“ESM2Text”**가 명확히 정의되어 있지 않다(학습 데이터, 모델, 프롬프트, 버전, 접근 방식 등). 이는 재현성과 정보 누수 평가를 어렵게 한다.

구조 공유 특징 프로젝터의 설명이 모호하다. 모달리티 간에 가중치가 실제로 공유되는지, 아니면 아키텍처 템플릿만 공유되는지가 명확하지 않으며, 이에 따른 대칭성 및 모달리티 불일치의 영향이 불분명하다.

주요 기준 모델(Bi-RNN, CLIPZyme, GNN 기반 모델)은 최근의 단백질 파운데이션 모델이나 단백질–텍스트 멀티모달 접근법에 비해 다소 오래된 것으로 보인다. 텍스트 지식 통합을 논문의 핵심으로 강조하고 있음에도 불구하고, 더 강력한 최신 단백질 인코더나 LLM 기반 보조(assisted) 기준 모델과의 비교가 포함되지 않은 점은 결과의 설득력을 약화시킨다.



