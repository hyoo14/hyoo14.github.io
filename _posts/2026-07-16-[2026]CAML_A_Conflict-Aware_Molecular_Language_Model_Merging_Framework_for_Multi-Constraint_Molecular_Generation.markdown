---
layout: post
title:  "[2026]CAML: A Conflict-Aware Molecular Language Model Merging Framework for Multi-Constraint Molecular Generation"
date:   2026-07-16 21:33:09 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: CAML은 다중 제약 분자의 생성을 위한 새로운 모델 병합 프레임워크로, 전문가 모델 간의 갈등을 최소화하기 위해 협력적 게임 이론을 활용한다.


짧은 요약(Abstract) :


이 논문에서는 다중 속성 제약을 만족하는 분자 생성을 위한 새로운 프레임워크인 CAML(Conflict-Aware Molecular Language Model Merging Framework)을 제안합니다. 기존의 전이 학습 방식은 단일 속성 제약에 효과적이지만, 실제 약물 발견에서는 여러 속성 제약을 동시에 만족해야 합니다. 그러나 기존의 방법들은 치명적인 망각이나 그래디언트 충돌로 인해 이 문제를 해결하는 데 어려움을 겪고 있습니다. CAML은 속성별로 세분화된 모델(전문가 모델) 간의 협력적 게임으로 다중 제약 분자를 생성합니다. 이를 위해 안정성 인식 공분산 행렬 적응 진화 전략(SACMA-ES)을 수립하여 융합 전략을 동적으로 최적화합니다. 이 알고리즘은 각 속성의 중요성과 전문가 모델의 상대적 융합 가중치를 탐색하여 속성 간의 충돌을 최소화하는 내쉬 균형과 유사한 솔루션을 찾습니다. 실험 결과, CAML은 복잡한 다중 제약 시나리오에서 최첨단 성능을 달성하며, 이 훈련 없는 패러다임이 새로운 분자 설계에서 내재된 속성 충돌을 해결하는 강력하고 효율적인 솔루션을 제공함을 입증합니다.



This paper proposes a novel framework called CAML (Conflict-Aware Molecular Language Model Merging Framework) for generating molecules that satisfy multiple property constraints. Existing transfer learning approaches have shown efficacy in single-property constraint molecular generation, but real-world drug discovery requires molecules to meet multiple property constraints simultaneously. However, existing paradigms often struggle with this challenge due to catastrophic forgetting or gradient conflicts. CAML generates multi-constraint molecules as a cooperative game among property-specific fine-tuned models (expert models). Specifically, we formulate a Stability-Aware Covariance Matrix Adaptation Evolution Strategy (SACMA-ES) to dynamically optimize the fusion strategy. This algorithm searches for a Nash-equilibrium-like solution that minimizes conflicts among properties by exploring the optimal combination of the importance of the task parameter (intrinsic scale) and relative fusion weights of each expert (fusion coefficient). Extensive experiments demonstrate that CAML achieves state-of-the-art performance in complex multi-constraint scenarios. Our results validate that this training-free paradigm offers a robust and efficient solution for resolving intrinsic property conflicts in de novo molecular design.


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



CAML(Conflict-Aware Molecular Language Model Merging Framework)은 다중 제약 조건을 만족하는 분자 생성을 위한 새로운 모델 병합 방법론입니다. 이 방법론은 여러 개의 속성 전문 모델(전문가 모델) 간의 협력적 게임으로 분자 속성 전문가 모델 병합을 정의합니다. CAML의 핵심은 안정성 인식 공분산 행렬 적응 진화 전략(SACMA-ES)을 통해 최적의 병합 전략을 동적으로 최적화하는 것입니다.

#### 1. 모델 아키텍처
CAML은 기본적으로 ChemGPT 아키텍처를 기반으로 하며, 이는 Transformer 기반의 디코더 모델입니다. 이 모델은 대규모의 일반 분자 데이터셋에서 사전 훈련을 통해 일반적인 화학 구문을 학습합니다. 이후, 특정 속성에 대한 데이터셋을 사용하여 세부 조정(fine-tuning)을 통해 속성 전문 모델을 생성합니다.

#### 2. 데이터 준비
CAML은 PubChem 데이터베이스와 같은 대규모 데이터셋을 사용하여 훈련 데이터를 준비합니다. 이 데이터셋은 다양한 화학 속성을 가진 분자들로 구성되어 있으며, 각 속성에 대해 고품질의 하위 집합을 필터링하여 속성 전문 모델을 훈련합니다. 그러나 다중 속성을 동시에 만족하는 분자는 매우 드물기 때문에, CAML은 여러 개의 전문 모델을 병합하여 이 문제를 해결합니다.

#### 3. 병합 전략
CAML의 병합 과정은 협력적 게임 이론에 기반하여, 각 전문가 모델이 자신의 최적 파라미터 상태와의 거리를 최소화하는 방향으로 작동합니다. 이를 통해 각 속성의 중요성을 반영한 가중치를 동적으로 조정하여, 속성 간의 충돌을 최소화합니다. SACMA-ES는 이러한 최적화 문제를 해결하기 위해 사용되며, 경량화된 진화 알고리즘을 통해 파라미터 공간을 탐색합니다.

#### 4. 훈련 및 최적화
CAML은 훈련이 필요 없는 패러다임을 제공하여, 최적의 파라미터가 식별된 후에는 전문가 모델을 동적으로 병합하여 고품질의 분자를 생성할 수 있습니다. 이 과정에서 SACMA-ES는 각 속성의 성능을 극대화하고, 속성 간의 충돌과 성능 저하를 방지하는 정규화 항을 포함한 적합도 함수를 사용합니다.

이러한 방식으로 CAML은 다중 제약 조건을 만족하는 분자 생성을 위한 강력하고 효율적인 솔루션을 제공합니다.

---




CAML (Conflict-Aware Molecular Language Model Merging Framework) is a novel model merging methodology designed for generating molecules that satisfy multiple constraints. The core of this approach defines the merging of molecular property expert models as a cooperative game among several property-specific fine-tuned models. The key innovation of CAML is the Stability-Aware Covariance Matrix Adaptation Evolution Strategy (SACMA-ES), which dynamically optimizes the fusion strategy.

#### 1. Model Architecture
CAML is fundamentally based on the ChemGPT architecture, which is a Transformer-based decoder model. This model is pre-trained on large-scale general molecular datasets to learn the general chemical syntax. Subsequently, it fine-tunes on property-specific datasets to create specialized expert models.

#### 2. Data Preparation
CAML utilizes large datasets such as the PubChem database to prepare training data. This dataset consists of molecules with various chemical properties, and high-quality subsets are filtered for each property to train the expert models. However, finding molecules that satisfy multiple properties simultaneously is exceedingly rare, which is why CAML merges multiple expert models to address this challenge.

#### 3. Merging Strategy
The merging process in CAML is based on cooperative game theory, where each expert model operates to minimize the distance to its optimal parameter state. This allows for dynamic adjustment of weights that reflect the importance of each property, thereby minimizing conflicts among properties. SACMA-ES is employed to solve this optimization problem, utilizing a lightweight evolutionary algorithm to explore the parameter space.

#### 4. Training and Optimization
CAML offers a training-free paradigm, allowing for the dynamic merging of expert models to generate high-quality molecules once the optimal parameters are identified. In this process, SACMA-ES employs a fitness function that includes penalties for property conflicts and degradation, ensuring that the performance of each property is maximized while preventing trade-offs that could harm other properties.

Through this approach, CAML provides a robust and efficient solution for generating molecules that meet multiple constraints.


<br/>
# Results



이 논문에서는 CAML(Conflict-Aware Molecular Language Model Merging Framework)이라는 새로운 프레임워크를 제안하고, 이를 통해 다중 제약 조건을 만족하는 분자의 생성을 수행합니다. CAML은 여러 속성 전문가 모델을 통합하여 협력적 게임으로 모델 병합을 수행하며, 이를 통해 속성 간의 충돌을 최소화합니다. 실험 결과, CAML은 다양한 복잡한 제약 조건을 가진 시나리오에서 최첨단 성능을 달성했습니다.

#### 실험 결과 요약

1. **경쟁 모델**: CAML은 여러 기존 모델과 비교되었습니다. 여기에는 전통적인 전이 학습 모델인 ChemGPT, 강화 학습 기반의 ChemGPT-PPO, RationaleRL, 그리고 최신 모델 병합 기법인 TIES, PCB, Iso-Merging 등이 포함됩니다.

2. **테스트 데이터**: 실험은 10,000개의 생성된 분자를 대상으로 진행되었으며, 각 모델의 성능을 평가하기 위해 다양한 속성 조합을 사용했습니다. 예를 들어, QED(Quantitative Estimation of Drug-likeness)와 GSK3β 억제제 생성, 합성 용이성(SA) 및 독성(Tox) 제약 조건을 포함한 복합적인 시나리오가 포함되었습니다.

3. **메트릭**: 성능 평가는 세 가지 주요 메트릭을 사용하여 이루어졌습니다:
   - **Nash Convergence Distance (NCD)**: 생성된 분자의 평균 속성과 전문가 모델의 이상적인 점 간의 유클리드 거리를 측정합니다. 낮은 NCD 값은 충돌이 최소화된 상태를 나타냅니다.
   - **Weighted Average Utility (WAU)**: 생성된 분자의 종합적인 품질을 평가하는 가중치가 부여된 스칼라 점수입니다. 여러 속성의 균형을 반영합니다.
   - **Comprehensive Discovery Score (CDS)**: 품질과 탐색의 균형을 평가하는 복합 메트릭으로, 성공률(SR), 다양성(Div), 참신성(Nov)을 포함합니다.

4. **비교 결과**: CAML은 모든 테스트 시나리오에서 기존 모델들보다 우수한 성능을 보였습니다. 특히, Task 4(QED + SA + Tox + GSK3β)에서는 CAML이 NCD 0.239, WAU 0.54, CDS 0.50을 기록하며, TIES(0.300, 0.51, 0.45) 및 ChemGPT-PPO(0.657, 0.34, 0.40)와 비교하여 현저한 성과를 나타냈습니다. 이는 CAML이 속성 간의 충돌을 효과적으로 해결하고, 고품질의 분자를 생성할 수 있음을 보여줍니다.



This paper introduces CAML (Conflict-Aware Molecular Language Model Merging Framework), a novel framework designed to generate molecules that satisfy multiple constraints. CAML integrates several property-specific expert models and formulates the merging process as a cooperative game, minimizing conflicts among properties. The experimental results demonstrate that CAML achieves state-of-the-art performance in various complex multi-constraint scenarios.

#### Summary of Experimental Results

1. **Competing Models**: CAML was compared against several existing models, including the traditional transfer learning model ChemGPT, reinforcement learning-based ChemGPT-PPO, RationaleRL, and recent model merging techniques such as TIES, PCB, and Iso-Merging.

2. **Test Data**: The experiments were conducted on a dataset of 10,000 generated molecules, evaluating the performance of each model across various combinations of properties. For instance, scenarios included generating QED (Quantitative Estimation of Drug-likeness) and GSK3β inhibitors, as well as incorporating constraints for synthetic accessibility (SA) and toxicity (Tox).

3. **Metrics**: Performance evaluation was conducted using three key metrics:
   - **Nash Convergence Distance (NCD)**: Measures the Euclidean distance between the average property score of the generated batch and the ideal point formed by the expert models. A lower NCD indicates a state of minimal conflict.
   - **Weighted Average Utility (WAU)**: A weighted scalar score that assesses the overall quality of the generated molecules, reflecting the balance among multiple conflicting objectives.
   - **Comprehensive Discovery Score (CDS)**: A composite metric designed to evaluate the trade-off between quality and exploration, incorporating success rate (SR), diversity (Div), and novelty (Nov).

4. **Comparison Results**: CAML outperformed all competing models across all test scenarios. Notably, in Task 4 (QED + SA + Tox + GSK3β), CAML achieved an NCD of 0.239, a WAU of 0.54, and a CDS of 0.50, significantly surpassing TIES (0.300, 0.51, 0.45) and ChemGPT-PPO (0.657, 0.34, 0.40). This demonstrates CAML's effectiveness in resolving property conflicts and generating high-quality molecules.


<br/>
# 예제



이 논문에서는 CAML(Conflict-Aware Molecular Language Model Merging Framework)을 제안하며, 이는 다중 속성 제약을 만족하는 분자 생성을 위한 새로운 접근 방식을 제공합니다. CAML은 여러 속성에 대한 전문가 모델을 통합하여 협력적 게임으로 분자 생성을 수행합니다. 이 과정에서 각 전문가 모델은 특정 속성에 대한 전문성을 가지고 있으며, 이들을 효과적으로 결합하여 최적의 분자를 생성하는 것이 목표입니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **데이터셋**: PubChem 데이터베이스에서 수집된 분자 데이터.
   - **속성**: 각 분자는 여러 속성(예: QED, GSK3β 억제제, 합성 용이성(SA), 독성 등)에 대한 점수를 가집니다.
   - **구체적인 인풋**: 
     - QED 점수가 0.9 이상인 분자 10,000개.
     - GSK3β 억제제 점수가 0.5 이상인 분자 5,000개.
     - SA 점수가 2 이하인 분자 4,000개.
     - 독성 점수가 0.2 이하인 분자 1,500개.
   - **아웃풋**: 각 속성에 대해 훈련된 전문가 모델(예: QED 전문가, GSK3β 전문가 등).

2. **테스트 데이터**:
   - **테스트 시나리오**: QED + GSK3β + SA + 독성 제약을 동시에 만족하는 분자 생성.
   - **구체적인 인풋**: 
     - QED 점수가 0.9 이상, GSK3β 억제제 점수가 0.5 이상, SA 점수가 2 이하, 독성 점수가 0.2 이하인 분자를 생성하기 위한 요청.
   - **아웃풋**: CAML을 통해 생성된 분자, 이 분자는 위의 모든 제약을 만족해야 하며, 각 속성의 점수는 다음과 같을 수 있습니다:
     - QED: 0.92
     - GSK3β 억제제: 0.55
     - SA: 1.8
     - 독성: 0.1

이러한 방식으로 CAML은 다중 속성 제약을 만족하는 분자를 효과적으로 생성할 수 있습니다.

---




This paper proposes CAML (Conflict-Aware Molecular Language Model Merging Framework), which offers a novel approach for generating molecules that satisfy multiple property constraints. CAML performs molecular generation as a cooperative game among property-specific expert models. Each expert model specializes in a specific property, and the goal is to effectively combine them to generate optimal molecules.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Dataset**: Molecular data collected from the PubChem database.
   - **Properties**: Each molecule has scores for various properties (e.g., QED, GSK3β inhibition, Synthetic Accessibility (SA), toxicity, etc.).
   - **Specific Inputs**: 
     - 10,000 molecules with QED scores above 0.9.
     - 5,000 molecules with GSK3β inhibition scores above 0.5.
     - 4,000 molecules with SA scores below 2.
     - 1,500 molecules with toxicity scores below 0.2.
   - **Outputs**: Trained expert models for each property (e.g., QED expert, GSK3β expert, etc.).

2. **Test Data**:
   - **Test Scenario**: Generating molecules that simultaneously satisfy QED + GSK3β + SA + toxicity constraints.
   - **Specific Inputs**: 
     - A request to generate a molecule with QED score above 0.9, GSK3β inhibition score above 0.5, SA score below 2, and toxicity score below 0.2.
   - **Outputs**: Molecules generated by CAML that meet all the above constraints, with property scores such as:
     - QED: 0.92
     - GSK3β inhibition: 0.55
     - SA: 1.8
     - Toxicity: 0.1

In this way, CAML can effectively generate molecules that satisfy multiple property constraints.

<br/>
# 요약


CAML은 다중 제약 분자의 생성을 위한 새로운 모델 병합 프레임워크로, 전문가 모델 간의 갈등을 최소화하기 위해 협력적 게임 이론을 활용한다. 실험 결과, CAML은 복잡한 다중 제약 시나리오에서 기존 방법들보다 우수한 성능을 보였으며, 특히 QED + SA + Tox + GSK3β 조합에서 가장 낮은 NCD 값을 기록했다. 이 방법은 훈련 없이도 다양한 속성을 통합할 수 있는 유연한 솔루션을 제공한다.

---

CAML is a novel model merging framework for multi-constraint molecular generation that utilizes cooperative game theory to minimize conflicts among expert models. Experimental results show that CAML outperforms existing methods in complex multi-constraint scenarios, achieving the lowest NCD value, particularly in the QED + SA + Tox + GSK3β combination. This approach offers a flexible solution for integrating diverse properties without the need for retraining.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 이 피규어는 데이터 희소성과 속성 간의 충돌을 시각적으로 나타냅니다. 데이터 희소성은 다중 속성 요구 사항을 충족하는 분자의 부족을 강조하며, 속성 간의 충돌은 서로 상충하는 물리화학적 속성(예: 생물활성 대 독성) 간의 관계를 보여줍니다. 이 시각적 표현은 CAML의 필요성을 강조합니다.
   - **Figure 2**: CAML의 전체 프레임워크를 보여주는 다이어그램으로, 다양한 속성 전문가 모델이 어떻게 통합되는지를 설명합니다. 이 다이어그램은 각 전문가 모델의 기여를 시각적으로 나타내어, 협력적 게임 이론을 기반으로 한 모델 병합 과정을 이해하는 데 도움을 줍니다.
   - **Figure 3**: Ablation Study 결과를 보여주는 그래프로, CAML의 각 구성 요소가 NCD에 미치는 영향을 시각적으로 나타냅니다. 이 그래프는 동적 파라미터화의 중요성을 강조하며, 각 구성 요소가 모델의 성능에 미치는 영향을 명확히 보여줍니다.

2. **테이블**
   - **Table 1**: NCD(Nash Convergence Distance) 결과를 비교한 표로, CAML이 기존 방법들에 비해 낮은 NCD 값을 기록하여 속성 간의 충돌을 효과적으로 최소화했음을 보여줍니다. 이는 CAML이 다중 속성 요구 사항을 충족하는 데 있어 우수한 성능을 발휘함을 나타냅니다.
   - **Table 2**: WAU(Weighted Average Utility) 결과를 비교한 표로, CAML이 다양한 속성의 균형을 잘 맞추어 높은 유틸리티 점수를 기록했음을 보여줍니다. 이는 CAML이 다중 속성 최적화에서 효과적임을 나타냅니다.
   - **Table 3**: CDS(Comprehensive Discovery Score) 결과를 비교한 표로, CAML이 높은 품질과 탐색의 균형을 잘 맞추어 가장 높은 CDS 점수를 기록했음을 보여줍니다. 이는 CAML이 고품질 분자를 생성하면서도 화학적 다양성을 유지하는 데 성공했음을 나타냅니다.

3. **어펜딕스**
   - **A.1 데이터 준비 및 분석**: 데이터의 품질과 분포가 전문가 모델 훈련에 중요하다는 점을 강조하며, PubChem 데이터셋을 사용하여 단일 속성 전문가 모델을 훈련하는 방법을 설명합니다. 데이터 희소성 문제를 해결하기 위해 다중 속성 요구 사항을 충족하는 분자의 부족을 강조합니다.
   - **A.4 이론적 배경: Nash Equilibrium**: Nash Equilibrium의 개념을 모델 병합 문제에 적용하여, 각 전문가 모델이 어떻게 협력하여 최적의 파라미터를 찾는지를 설명합니다. 이는 CAML의 이론적 기초를 제공하며, 모델 병합 과정에서의 충돌 최소화의 중요성을 강조합니다.

---

### Insights and Results from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: This figure visually represents data scarcity and conflicts between properties. Data scarcity highlights the lack of molecules that meet multiple property requirements, while property conflicts illustrate the relationships between conflicting physicochemical properties (e.g., bioactivity vs. toxicity). This visual representation underscores the necessity of CAML.
   - **Figure 2**: A diagram showing the overall framework of CAML, explaining how various property expert models are integrated. This diagram visually represents the contributions of each expert model, aiding in understanding the model merging process based on cooperative game theory.
   - **Figure 3**: A graph showing the results of the ablation study, visually representing the impact of each component of CAML on NCD. This graph emphasizes the importance of dynamic parameterization and clearly shows the effects of each component on model performance.

2. **Tables**
   - **Table 1**: A comparison of NCD (Nash Convergence Distance) results, showing that CAML achieved lower NCD values compared to existing methods, effectively minimizing conflicts between properties. This indicates CAML's superior performance in meeting multiple property requirements.
   - **Table 2**: A comparison of WAU (Weighted Average Utility) results, demonstrating that CAML achieved high utility scores by effectively balancing various properties. This indicates CAML's effectiveness in multi-property optimization.
   - **Table 3**: A comparison of CDS (Comprehensive Discovery Score) results, showing that CAML recorded the highest CDS score, successfully balancing quality and exploration. This indicates that CAML generates high-quality molecules while maintaining chemical diversity.

3. **Appendices**
   - **A.1 Data Preparation and Analysis**: Emphasizes the importance of data quality and distribution for training expert models, using the PubChem dataset to train single-property expert models. It highlights the issue of data scarcity, emphasizing the lack of molecules that meet multiple property requirements.
   - **A.4 Theoretical Background: Nash Equilibrium**: Applies the concept of Nash Equilibrium to the model merging problem, explaining how each expert model collaborates to find optimal parameters. This provides a theoretical foundation for CAML and emphasizes the importance of minimizing conflicts in the model merging process.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Ren2026,
  author    = {Xuanbai Ren and Luoda Tan and Pei Liu and Tengfei Ma and Xiangzheng Fu and Longyue Wang and Yiping Liu and Xiangxiang Zeng},
  title     = {CAML: A Conflict-Aware Molecular Language Model Merging Framework for Multi-Constraint Molecular Generation},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {19578--19594},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  address   = {Vienna, Austria}
}
```

### Chicago Style Citation

Ren, Xuanbai, Luoda Tan, Pei Liu, Tengfei Ma, Xiangzheng Fu, Longyue Wang, Yiping Liu, and Xiangxiang Zeng. "CAML: A Conflict-Aware Molecular Language Model Merging Framework for Multi-Constraint Molecular Generation." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 19578–19594. Vienna, Austria: Association for Computational Linguistics, 2026.
