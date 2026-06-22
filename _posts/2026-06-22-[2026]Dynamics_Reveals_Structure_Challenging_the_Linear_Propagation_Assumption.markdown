---
layout: post
title:  "[2026]Dynamics Reveals Structure: Challenging the Linear Propagation Assumption"
date:   2026-06-22 08:40:18 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 선형 전파 가정(Linear Propagation Assumption, LPA)의 기하학적 한계를 조사하고, 지식 편집 및 다단계 추론에서의 실패가 LPA의 구조적 제약에서 기인할 수 있음을 보여준다.  


짧은 요약(Abstract) :



이 논문에서는 신경망이 1차 매개변수 업데이트를 통해 적응하는 과정에서 이러한 업데이트가 논리적 일관성을 유지하는지에 대한 불확실성을 다룹니다. 저자들은 "선형 전파 가정(Linear Propagation Assumption, LPA)"의 기하학적 한계를 조사하며, 이는 지역 업데이트가 논리적 결과로 일관되게 전파된다는 전제입니다. 이를 형식화하기 위해 관계 대수(relational algebra)를 채택하고, 관계에 대한 세 가지 핵심 연산인 부정(negation), 대칭(converse), 그리고 합성(composition)을 연구합니다. 부정과 대칭의 경우, 방향에 구애받지 않는 1차 전파를 보장하기 위해서는 개체 쌍의 맥락과 관계 내용을 분리하는 텐서 분해가 필요하다는 것을 증명합니다. 그러나 합성의 경우, 근본적인 장애물을 확인합니다. 합성이 논리적 합(conjunction)으로 축소되며, 선형 특성에 대해 잘 정의된 모든 논리적 합은 이변량(bilinear) 구조를 가져야 한다고 증명합니다. 이 결과는 지식 편집 실패, 역전 저주(reversal curse), 다단계 추론(multi-hop reasoning)에서의 문제들이 LPA에 내재된 공통 구조적 한계에서 비롯될 수 있음을 시사합니다.




This paper addresses the uncertainty regarding whether first-order parameter updates in neural networks preserve logical coherence during adaptation. The authors investigate the geometric limits of the Linear Propagation Assumption (LPA), which posits that local updates coherently propagate to logical consequences. To formalize this, they adopt relation algebra and study three core operations on relations: negation, converse, and composition. For negation and converse, they prove that guaranteeing direction-agnostic first-order propagation necessitates a tensor factorization that separates entity-pair context from relation content. However, for composition, they identify a fundamental obstruction. They show that composition reduces to conjunction, and that any conjunction well-defined on linear features must be bilinear. These results suggest that failures in knowledge editing, the reversal curse, and multi-hop reasoning may stem from common structural limitations inherent to the LPA.


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



이 논문에서는 "Linear Propagation Assumption (LPA)"의 기하학적 한계를 탐구하고, 이를 통해 신경망의 업데이트가 논리적 일관성을 유지할 수 있는지에 대한 질문을 다룹니다. 연구의 주요 방법론은 다음과 같습니다.

1. **관계 대수(Relation Algebra) 사용**: 연구자들은 관계 대수를 통해 논리적 지식을 형식화합니다. 이 대수는 세 가지 기본 연산(부정, 대칭, 합성)을 사용하여 관계를 정의하고, 이들 연산이 신경망의 업데이트에 미치는 영향을 분석합니다.

2. **기하학적 구조 분석**: 연구자들은 신경망의 파라미터 업데이트가 논리적 관계를 어떻게 유지하는지를 이해하기 위해 기하학적 구조를 분석합니다. 이를 위해, 그라디언트 벡터의 정렬과 같은 기하학적 요구 사항을 도출하고, 이러한 요구 사항이 LPA의 유효성을 어떻게 제한하는지를 보여줍니다.

3. **시스템적 선형 전파(Systematic Linear Propagation, SLP)**: 연구자들은 SLP라는 개념을 도입하여, 신경망의 업데이트가 논리적 변환을 자동으로 추적할 수 있는 조건을 정의합니다. SLP는 부정과 대칭 연산이 신경망의 기하학적 구조에 어떻게 통합될 수 있는지를 설명합니다.

4. **텐서 분해(Tensor Factorization)**: 연구자들은 부정과 대칭 연산을 지원하기 위해, 신경망의 피처 공간이 텐서 곱 구조로 분해되어야 한다고 주장합니다. 이 구조는 엔티티 쌍의 맥락 정보와 관계 정보를 분리하여, 논리적 일관성을 유지하는 데 필요한 기하학적 요구 사항을 충족합니다.

5. **실험적 검증**: 연구자들은 다양한 대형 언어 모델(LLM)을 사용하여 이론적 주장을 실험적으로 검증합니다. 그들은 그라디언트 정렬 실험을 통해 LPA가 실제로 논리적 일관성을 유지하지 못하는 경우를 보여주고, 이러한 실패가 신경망의 기하학적 구조와 관련이 있음을 입증합니다.

이러한 방법론을 통해 연구자들은 LPA의 한계를 명확히 하고, 신경망의 업데이트가 논리적 일관성을 유지하기 위해 필요한 기하학적 구조를 제안합니다. 이 연구는 지식 편집, 다단계 추론 및 지속적 학습과 같은 다양한 기계 학습 응용 프로그램에 중요한 시사점을 제공합니다.

---




This paper explores the geometric limits of the "Linear Propagation Assumption (LPA)" and addresses the question of whether updates in neural networks can maintain logical coherence. The main methodologies of the study are as follows:

1. **Use of Relation Algebra**: The researchers formalize logical knowledge using relation algebra, which employs three core operations (negation, converse, and composition) to define relations and analyze the impact of these operations on neural network updates.

2. **Geometric Structure Analysis**: The researchers analyze the geometric structure of neural network parameter updates to understand how they maintain logical relationships. They derive geometric requirements, such as the alignment of gradient vectors, and demonstrate how these requirements limit the validity of LPA.

3. **Systematic Linear Propagation (SLP)**: The concept of SLP is introduced to define the conditions under which neural network updates can automatically track logical transformations. SLP explains how negation and converse operations can be integrated into the geometric structure of the neural network.

4. **Tensor Factorization**: The researchers argue that to support negation and converse operations, the feature space of the neural network must be decomposed into a tensor product structure. This structure separates entity-pair context information from relation content, fulfilling the geometric requirements necessary for maintaining logical coherence.

5. **Experimental Validation**: The researchers empirically validate their theoretical claims using various large language models (LLMs). They conduct gradient alignment experiments to show that LPA fails to maintain logical coherence in practice, demonstrating that these failures are related to the geometric structure of the neural network.

Through these methodologies, the researchers clarify the limitations of LPA and propose the geometric structures required for neural network updates to maintain logical coherence. This research has significant implications for various machine learning applications, including knowledge editing, multi-hop reasoning, and continual learning.


<br/>
# Results



이 논문에서는 Linear Propagation Assumption (LPA)의 기하학적 한계를 분석하고, 이를 통해 지식 편집, 역전의 저주, 다중 홉 추론에서의 실패가 LPA에 내재된 구조적 한계에서 기인할 수 있음을 제시합니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구에서는 Qwen3-4B, Qwen3-30B, OLMo-3-7B와 같은 다양한 대형 언어 모델(LLM)을 사용하여 실험을 진행했습니다. 이 모델들은 서로 다른 아키텍처와 크기를 가지고 있어, 결과의 일반성을 확보하는 데 기여했습니다.

2. **테스트 데이터**: TREX 서브셋의 NEGATEDLAMA 벤치마크를 사용하여 모델의 성능을 평가했습니다. 이 데이터셋은 사실 쿼리와 그 부정 형태의 쌍을 제공하여, 모델이 부정 일관성을 얼마나 잘 유지하는지를 측정할 수 있게 합니다.

3. **메트릭**: 모델의 성능을 평가하기 위해 코사인 유사도를 사용했습니다. 이는 사실 쿼리와 그 부정 쿼리의 그래디언트 벡터 간의 정렬 정도를 측정하는 데 사용되었습니다. 연구 결과, Qwen3-4B 모델의 경우 사실 쿼리와 그 부정 쿼리 간의 그래디언트가 강하게 양의 정렬을 보였으며, 평균 유사도는 0.85로 나타났습니다. 이는 LPA의 이론적 요구 사항과 상충하는 결과입니다.

4. **비교**: Qwen3-30B와 OLMo-3-7B 모델에서도 유사한 결과가 나타났습니다. Qwen3-30B의 경우 평균 유사도는 0.86, OLMo-3-7B는 0.62로, 모든 모델에서 부정 일관성이 유지되지 않는다는 점이 확인되었습니다. 이러한 결과는 LPA가 모델의 업데이트 메커니즘과 일관성을 보장하지 못한다는 것을 시사합니다.

5. **결론**: 이 연구는 LPA의 기하학적 구조가 논리적 일관성을 유지하는 데 필요한 조건을 충족하지 못함을 보여주며, 이는 지식 편집 및 다중 홉 추론에서의 실패와 관련이 있음을 강조합니다. 따라서, LPA에 기반한 접근 방식은 이러한 구조적 한계를 극복하기 위한 추가적인 메커니즘이나 비선형 업데이트가 필요하다는 결론에 도달했습니다.

---



This paper analyzes the geometric limits of the Linear Propagation Assumption (LPA) and suggests that failures in knowledge editing, the reversal curse, and multi-hop reasoning may stem from inherent structural limitations of the LPA. The main results of the study are as follows:

1. **Competing Models**: The study utilized various large language models (LLMs) such as Qwen3-4B, Qwen3-30B, and OLMo-3-7B for experiments. These models have different architectures and sizes, contributing to the generalizability of the results.

2. **Test Data**: The TREX subset of the NEGATEDLAMA benchmark was used to evaluate the performance of the models. This dataset provides pairs of factual queries and their negated forms, allowing for the measurement of how well the models maintain negation consistency.

3. **Metrics**: Cosine similarity was employed to assess the performance of the models. This metric measures the degree of alignment between the gradient vectors of factual queries and their negated counterparts. The results showed that the Qwen3-4B model exhibited a strong positive alignment between the gradients of factual and negated queries, with an average similarity of 0.85, which contradicts the theoretical requirements of the LPA.

4. **Comparison**: Similar results were observed in the Qwen3-30B and OLMo-3-7B models. The average similarity for Qwen3-30B was 0.86, while OLMo-3-7B showed a slightly lower average of 0.62, confirming that negation consistency does not hold across all models. These findings suggest that the LPA does not reliably guarantee logical consistency in the model's update mechanisms.

5. **Conclusion**: The study demonstrates that the geometric structure imposed by the LPA fails to meet the necessary conditions for maintaining logical coherence, which is linked to failures in knowledge editing and multi-hop reasoning. Therefore, approaches based on LPA may require additional mechanisms or nonlinear updates to overcome these structural limitations.


<br/>
# 예제



이 논문에서는 "Linear Propagation Assumption (LPA)"라는 가정을 검토하고, 이 가정이 신경망의 파라미터 업데이트가 논리적 일관성을 유지하는 데 어떤 한계를 가지는지를 분석합니다. 특히, 논문은 세 가지 기본적인 관계 연산인 부정(negation), 대칭(converse), 그리고 조합(composition)에 대해 다룹니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: "T-Rex는 육식동물이다." (fact)
   - **출력**: "T-Rex는 육식동물이다."의 확률 점수 (예: 0.9)
   - **입력**: "T-Rex는 초식동물이 아니다." (negation)
   - **출력**: "T-Rex는 초식동물이 아니다."의 확률 점수 (예: 0.1)

   이 데이터는 모델이 T-Rex에 대한 사실을 학습하도록 돕습니다. 모델은 "T-Rex는 육식동물이다."라는 사실을 학습하고, 그에 따라 "T-Rex는 초식동물이 아니다."라는 부정된 사실도 학습하게 됩니다.

2. **테스트 데이터**:
   - **입력**: "T-Rex는 육식동물이다." (fact)
   - **출력**: 모델이 이 사실을 기반으로 예측한 확률 점수 (예: 0.85)
   - **입력**: "T-Rex는 초식동물이 아니다." (negation)
   - **출력**: 모델이 이 부정된 사실에 대해 예측한 확률 점수 (예: 0.15)

   테스트 데이터는 모델이 학습한 내용을 바탕으로 새로운 입력에 대해 얼마나 잘 예측하는지를 평가합니다. 이 경우, 모델이 "T-Rex는 육식동물이다."라는 사실을 강화하는 업데이트를 수행할 때, "T-Rex는 초식동물이 아니다."라는 부정된 사실의 점수도 자동으로 감소해야 합니다. 그러나 논문에서는 이러한 일관성이 보장되지 않는다고 주장합니다.

#### 구체적인 테스크
- **테스크**: 모델이 주어진 사실에 대해 부정된 사실을 자동으로 업데이트하고 예측하는 능력 평가
- **목표**: 모델이 부정된 사실을 정확하게 예측할 수 있도록 하는 것, 즉 "T-Rex는 육식동물이다."라는 사실이 업데이트되면 "T-Rex는 초식동물이 아니다."라는 사실의 점수도 자동으로 감소해야 함.

이러한 예시는 모델이 어떻게 학습하고 업데이트하는지를 보여주며, LPA의 한계가 실제로 어떻게 나타나는지를 설명합니다.

---




This paper examines the "Linear Propagation Assumption (LPA)" and analyzes the limitations of this assumption in maintaining logical coherence during parameter updates in neural networks. Specifically, the paper addresses three fundamental relational operations: negation, converse, and composition.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Input**: "T-Rex is a carnivore." (fact)
   - **Output**: Probability score for "T-Rex is a carnivore." (e.g., 0.9)
   - **Input**: "T-Rex is not a herbivore." (negation)
   - **Output**: Probability score for "T-Rex is not a herbivore." (e.g., 0.1)

   This data helps the model learn facts about T-Rex. The model learns that "T-Rex is a carnivore," and consequently, it also learns the negated fact "T-Rex is not a herbivore."

2. **Test Data**:
   - **Input**: "T-Rex is a carnivore." (fact)
   - **Output**: Probability score predicted by the model based on this fact (e.g., 0.85)
   - **Input**: "T-Rex is not a herbivore." (negation)
   - **Output**: Probability score predicted by the model for this negated fact (e.g., 0.15)

   The test data evaluates how well the model predicts new inputs based on what it has learned. In this case, when the model performs an update reinforcing the fact "T-Rex is a carnivore," the score for the negated fact "T-Rex is not a herbivore" should also automatically decrease. However, the paper argues that such consistency is not guaranteed.

#### Specific Task
- **Task**: Evaluate the model's ability to automatically update and predict negated facts based on given facts.
- **Goal**: Ensure that when the model strengthens the fact "T-Rex is a carnivore," the score for the negated fact "T-Rex is not a herbivore" also decreases automatically.

These examples illustrate how the model learns and updates, explaining how the limitations of LPA manifest in practice.

<br/>
# 요약


이 논문에서는 선형 전파 가정(Linear Propagation Assumption, LPA)의 기하학적 한계를 조사하고, 지식 편집 및 다단계 추론에서의 실패가 LPA의 구조적 제약에서 기인할 수 있음을 보여준다. 연구 결과, 부정(negation)과 대칭(antisymmetry) 연산은 텐서 곱 분해를 요구하며, 결합(conjunction)은 부정과의 호환성 문제로 인해 선형적으로 구현할 수 없음을 입증하였다. 이러한 결과는 현재의 대형 언어 모델이 논리적 일관성을 유지하는 데 있어 기하학적 제약을 받는다는 것을 시사한다.

---

This paper investigates the geometric limits of the Linear Propagation Assumption (LPA) and demonstrates that failures in knowledge editing and multi-hop reasoning may stem from structural constraints inherent to the LPA. The findings reveal that negation and converse operations necessitate tensor product decomposition, while conjunction is fundamentally incompatible with negation under linearity. These results suggest that current large language models face geometric constraints that hinder their ability to maintain logical coherence.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1**: 이 도표는 논리적 동등성의 기하학적 해석을 보여줍니다. 쿼리 \( q \)의 점수 변화가 쿼리의 부정 \( \neg q \)에 어떻게 영향을 미치는지를 설명합니다. LPA(선형 전파 가정) 하에서, 쿼리의 점수를 증가시키는 업데이트는 그 부정의 점수를 감소시켜야 하며, 이는 기하학적으로 두 기울기 벡터가 반대 방향으로 정렬되어야 함을 의미합니다. 이 결과는 LPA의 유효성을 검증하는 데 중요한 기초를 제공합니다.

- **Figure 2**: 이 피규어는 LLM(대형 언어 모델)에서 사실과 그 부정 간의 기울기 정렬을 보여줍니다. 실험 결과, 기울기 벡터가 강하게 양의 정렬을 보이며, 이는 LPA의 기본 가정이 위배되고 있음을 나타냅니다. 이 결과는 LLM이 부정 일관성을 유지하지 못하는 구조적 문제를 시사합니다.

- **Figure 3**: 이 도표는 논리적 동등성과 LPA의 충돌을 시각적으로 설명합니다. 논리적 규칙과 기하학적 규칙 간의 불일치를 보여주며, 이는 LPA 하에서의 논리적 연산이 어떻게 실패하는지를 설명합니다.

- **Figure 4**: 이 피규어는 다양한 모델에서의 기울기 정렬을 보여줍니다. Qwen3-30B 및 OLMo-3-7B 모델에서의 기울기 정렬 결과는 LPA의 유효성을 의심하게 만듭니다. 이 결과는 LLM의 구조적 한계를 강조합니다.

#### 2. 테이블
- **Table 1**: 이 표는 선형 업데이트 근사치의 유효성을 보여줍니다. 작은 단계 크기에서 실제 업데이트가 첫 번째 차수 예측과 잘 일치하는 반면, 큰 업데이트에서는 예측이 실패하는 경향을 보입니다. 이는 LLM의 업데이트 메커니즘이 선형 근사에 의존하고 있음을 나타내며, 이는 모델의 동적 특성을 이해하는 데 중요한 통찰을 제공합니다.

#### 3. 어펜딕스
- **A. Experimental Details and Additional Results**: 이 섹션에서는 기울기 정렬 실험의 세부 사항과 결과를 설명합니다. 다양한 모델에서의 기울기 정렬을 통해 LPA의 유효성을 검증하고, LLM의 부정 일관성 문제를 강조합니다.

- **B. Primer on Finite Group Representations**: 이 섹션은 유한 그룹 이론의 기본 개념을 설명합니다. 이는 논문에서 사용된 수학적 기초를 제공하며, 논리적 구조와 기하학적 표현 간의 관계를 이해하는 데 도움을 줍니다.

- **C. Proof of Theorem 1**: 이 섹션에서는 정리 1의 증명을 제공합니다. 이는 LPA의 구조적 요구 사항을 수학적으로 정립하며, LPA가 요구하는 기하학적 구조를 명확히 합니다.

- **D. Proof of Theorem 2**: 이 섹션은 정리 2의 증명을 다룹니다. 이는 대칭-비대칭 정렬의 필요성을 설명하며, LPA의 요구 사항을 충족하기 위한 기하학적 구조를 제시합니다.

- **E. Proof of Lemma 2**: 이 섹션은 레마 2의 증명을 제공합니다. 이는 LPA 하에서의 결합 연산의 특성을 설명하며, 기하학적 구조와 논리적 연산 간의 관계를 명확히 합니다.

- **F. Robustness to Approximate Negation**: 이 섹션은 근사 부정에 대한 저항성을 다룹니다. 부정의 근사 표현이 여전히 결합 연산과의 호환성을 유지하지 못하는 구조적 문제를 강조합니다.

---

### Summary of Results and Insights from Figures, Tables, and Appendices

#### 1. Diagrams and Figures
- **Figure 1**: This diagram illustrates the geometric interpretation of logical equivalence. It explains how the score change of a query \( q \) affects its negation \( \neg q \). Under the Linear Propagation Assumption (LPA), an increase in the score of a query should lead to a decrease in the score of its negation, indicating that the gradient vectors must be anti-aligned. This result provides a foundational basis for validating the LPA.

- **Figure 2**: This figure shows the alignment of gradients between factual queries and their negated counterparts in LLMs (Large Language Models). The experimental results reveal a strong positive alignment, contradicting the theoretical requirement for systematic propagation. This finding suggests a structural issue in LLMs regarding the maintenance of negation consistency.

- **Figure 3**: This diagram visually explains the conflict between logical equivalence and LPA. It illustrates the discrepancies between logical rules and geometric rules, highlighting how LPA fails to uphold logical operations.

- **Figure 4**: This figure presents gradient alignment across different models. The results from Qwen3-30B and OLMo-3-7B indicate that the positive alignment phenomenon persists, raising doubts about the validity of the LPA. This finding emphasizes the structural limitations of LLMs.

#### 2. Tables
- **Table 1**: This table demonstrates the validity of the linearized update approximation. It shows that actual updates closely track the first-order predictions for small step sizes, while the approximation breaks down for larger updates. This indicates that the update mechanism of LLMs relies on linear approximations, providing crucial insights into the dynamics of the model.

#### 3. Appendices
- **A. Experimental Details and Additional Results**: This section details the gradient alignment experiments and results. It validates the LPA through gradient alignment across various models, emphasizing the issue of negation consistency in LLMs.

- **B. Primer on Finite Group Representations**: This section explains the basic concepts of finite group theory. It provides the mathematical foundation used in the paper, aiding in understanding the relationship between logical structure and geometric representation.

- **C. Proof of Theorem 1**: This section provides the proof of Theorem 1, establishing the structural requirements of the LPA mathematically and clarifying the geometric structure required by the LPA.

- **D. Proof of Theorem 2**: This section addresses the proof of Theorem 2, explaining the necessity of symmetric-antisymmetric alignment and presenting the geometric structure required to satisfy the LPA.

- **E. Proof of Lemma 2**: This section provides the proof of Lemma 2, explaining the properties of conjunction under the LPA and clarifying the relationship between geometric structure and logical operations.

- **F. Robustness to Approximate Negation**: This section discusses the robustness of the results to approximate representations of negation. It highlights the structural issues that arise when approximate negation is considered, emphasizing the incompatibility with bilinear conjunction.

This summary encapsulates the key findings and insights from the figures, tables, and appendices of the paper, providing a comprehensive overview of the research's implications for understanding the limitations of LLMs under the Linear Propagation Assumption.

<br/>
# refer format:

### BibTeX 

```bibtex
@inproceedings{chang2026dynamics,
  author = {Hoyeon Chang and Bálint Mucsányi and Seong Joon Oh},
  title = {Dynamics Reveals Structure: Challenging the Linear Propagation Assumption},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026},
  address = {Seoul, South Korea},
  publisher = {PMLR},
  volume = {306},
  pages = {1--12},
  url = {https://arxiv.org/abs/2601.21601}
}
```

### Chicago


Chang, Hoyeon, Bálint Mucsányi, and Seong Joon Oh. 2026. "Dynamics Reveals Structure: Challenging the Linear Propagation Assumption." In *Proceedings of the 43rd International Conference on Machine Learning*, 306:1-12. Seoul, South Korea: PMLR. https://arxiv.org/abs/2601.21601.
