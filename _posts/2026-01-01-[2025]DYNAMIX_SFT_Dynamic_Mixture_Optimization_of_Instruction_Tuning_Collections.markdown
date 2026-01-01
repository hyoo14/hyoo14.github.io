---
layout: post
title:  "[2025]DYNAMIX SFT: Dynamic Mixture Optimization of Instruction Tuning Collections"
date:   2026-01-01 15:40:00 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: DYNAMIX SFT는 다수의 인스트럭션 튜닝 데이터셋의 혼합을 최적화하기 위해 multi armed bendit 문제로 설정하고, 원래 데이터셋 비율에 부드럽게 고정된 Prior-scaled Boltzmann Exploration을 도입하여 샘플링 분포를 업데이트합니다.


짧은 요약(Abstract) :

DYNAMIX SFT는 다양한 인스트럭션 튜닝 데이터셋의 혼합을 동적으로 최적화하는 방법을 제안합니다. 이 방법은 다중 무장 밴딧 문제로 문제를 공식화하고, 업데이트된 샘플링 분포를 원래 데이터셋 비율에 부드럽게 고정하는 Prior-scaled Boltzmann Exploration을 도입하여 데이터셋의 고유한 다양성과 범위를 유지합니다. 샘플링 확률은 모델의 현재 상태에서 성능 향상에 기여하는 정도를 반영하는 경량의 1-Step Look-ahead Reward를 사용하여 업데이트됩니다. 16개의 지침 조정 데이터셋으로 구성된 TULU-v2 혼합 컬렉션에 적용했을 때, DYNAMIX SFT는 10개의 벤치마크에서 최대 2.2%의 성능 향상을 달성했습니다. 또한, 우리는 방법의 적응적 동역학에 대한 깊은 통찰을 제공하기 위해 포괄적인 분석과 시각화를 제공합니다.



DYNAMIX SFT proposes a method for dynamically optimizing the mixture of diverse instruction-tuning datasets. It formulates the problem as a multi-armed bandit setup and introduces a Prior-scaled Boltzmann Exploration that softly anchors the updated sampling distribution to the original dataset proportions, thereby preserving the inherent diversity and coverage of the collection. Sampling probabilities are updated using a lightweight 1-Step Look-ahead Reward that reflects how much the dataset contributes to improving the model's performance at its current state. When applied to the TULU-v2 mixture collection comprising 16 instruction-tuning datasets, DYNAMIX SFT achieves up to a 2.2% performance improvement across 10 benchmarks. Furthermore, we provide a comprehensive analysis and visualizations to offer deeper insights into the adaptive dynamics of our method.


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


DYNAMIX SFT는 대규모 언어 모델(LLM)의 후속 훈련 단계에서 동적 데이터셋 혼합 최적화를 위한 방법론입니다. 이 방법은 여러 개의 이질적인 데이터셋을 효과적으로 조합하여 모델의 성능을 극대화하는 것을 목표로 합니다. DYNAMIX SFT는 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. **다중 무장 밴딧(Multi-Armed Bandit) 문제 설정**: DYNAMIX SFT는 각 데이터셋을 '팔'로 간주하고, 최적의 샘플링 비율을 찾기 위해 다중 무장 밴딧 문제로 문제를 정의합니다. 이 설정은 각 데이터셋의 기여도를 평가하고, 이를 기반으로 샘플링 확률을 동적으로 조정할 수 있게 합니다.

2. **Prior-scaled Boltzmann Exploration**: 이 방법은 샘플링 분포를 원래 데이터셋 비율에 부드럽게 고정시키는 전략을 도입합니다. 이를 통해 데이터셋의 다양성과 범위를 유지하면서도, 모델의 학습 진행 상황에 따라 샘플링 비율을 조정할 수 있습니다. 이 과정에서 각 데이터셋의 특성을 반영하여 샘플링 비율을 업데이트합니다.

3. **1-Step Look-ahead Reward**: DYNAMIX SFT는 각 데이터셋의 기여도를 평가하기 위해 1단계 미리보기 보상을 사용합니다. 이는 각 데이터셋에 대해 가상의 한 번의 그래디언트 업데이트를 수행하고, 그 결과로 모델 손실이 얼마나 감소하는지를 측정하여 보상을 계산합니다. 이 보상은 데이터셋의 유용성을 즉각적으로 반영하며, 별도의 검증 세트 없이도 동적으로 샘플링 비율을 조정할 수 있게 합니다.

4. **경량화된 업데이트**: DYNAMIX SFT는 추가적인 파라미터나 복잡한 아키텍처 없이도 동적으로 데이터셋 혼합을 최적화할 수 있도록 설계되었습니다. 이는 훈련 과정에서의 오버헤드를 최소화하고, 효율적인 데이터 할당을 가능하게 합니다.

DYNAMIX SFT는 TULU-v2 혼합 컬렉션에 적용되어 10개의 벤치마크에서 성능을 평가하였으며, 기존의 정적 혼합 방식에 비해 최대 2.2%의 성능 향상을 달성하였습니다. 이 방법은 대규모 데이터셋 컬렉션을 보다 원칙적이고 자동화된 방식으로 활용할 수 있게 해줍니다.




DYNAMIX SFT is a method for dynamic dataset mixture optimization in the post-training phase of large language models (LLMs). This approach aims to effectively combine multiple heterogeneous datasets to maximize model performance. DYNAMIX SFT consists of the following key components:

1. **Multi-Armed Bandit Problem Setup**: DYNAMIX SFT formulates the problem of finding optimal sampling ratios as a multi-armed bandit problem, treating each dataset as an "arm." This setup allows for the evaluation of each dataset's contribution and dynamically adjusts sampling probabilities based on this evaluation.

2. **Prior-scaled Boltzmann Exploration**: This method introduces a strategy that softly anchors the sampling distribution to the original dataset proportions. This allows for the preservation of diversity and coverage of the datasets while adapting the sampling ratios according to the model's learning progress. The sampling probabilities are updated to reflect the inherent characteristics of each dataset.

3. **1-Step Look-ahead Reward**: DYNAMIX SFT employs a 1-step look-ahead reward to assess the contribution of each dataset. This involves temporarily performing a single gradient update on a mini-batch for each dataset and measuring how much the model's loss decreases as a result. This reward provides an immediate signal of each dataset's utility without requiring a separate validation set, enabling dynamic adjustments to the sampling ratios.

4. **Lightweight Updates**: DYNAMIX SFT is designed to optimize dataset mixtures dynamically without the need for additional parameters or complex architectures. This minimizes overhead during training and allows for efficient data allocation.

DYNAMIX SFT was applied to the TULU-v2 mixture collection and evaluated across 10 benchmarks, achieving up to a 2.2% performance improvement compared to static mixture methods. This approach enables the more principled and automated utilization of large-scale dataset collections.


<br/>
# Results



DYNAMIX SFT는 TULU-v2 혼합 컬렉션을 사용하여 16개의 다양한 instruction-tuning 데이터셋을 기반으로 한 실험을 수행하였습니다. 이 실험에서는 LLaMA3.2 1B 모델과 Qwen2.5 3B 모델을 사용하여 10개의 벤치마크에서 성능을 평가하였습니다. 평가된 벤치마크는 지식, 추론, 코딩, 수학, 그리고 일반적인 지시 수행을 포함합니다.

#### 경쟁 모델
DYNAMIX SFT는 두 가지 기본 그룹과 비교되었습니다:
1. **Heuristic Methods**: 
   - **Full Coverage**: 비율 샘플링을 사용하여 모든 데이터셋을 균등하게 포함.
   - **Uniform Sampling**: 모든 데이터셋을 동일한 비율로 샘플링.
   
2. **Dynamic Methods**:
   - **MultiDDS**: 데이터셋 간의 그래디언트 코사인 유사성을 기반으로 혼합을 조정.
   - **MultiUAT**: 모델 불확실성을 기반으로 혼합을 조정.

#### 테스트 데이터 및 메트릭
DYNAMIX SFT는 TULU-v2 혼합 컬렉션에서 약 300K 예제를 사용하였으며, 10개의 벤치마크에서 성능을 평가하였습니다. 각 벤치마크는 다음과 같은 메트릭을 사용하여 평가되었습니다:
- **MMLU**: 일반 지식 평가.
- **TruthfulQA**: 모델의 진실성 평가.
- **PopQA**: 대중적인 질문에 대한 응답 평가.
- **BigBench-Hard**: 복잡한 추론 문제 평가.
- **DROP**: 독해력 평가.
- **HumanEval**: 코드 생성 평가.
- **GSM8K**: 수학 문제 해결 평가.
- **MATH**: 수학 문제 해결 평가.
- **IFEval**: 지시 수행 평가.

#### 비교 결과
DYNAMIX SFT는 다음과 같은 성과를 보였습니다:
- **Full Coverage**와 비교했을 때, DYNAMIX SFT는 1B 모델에서 평균 성능이 +2.2% 향상되었고, 3B 모델에서는 +1.5% 향상되었습니다.
- DYNAMIX SFT는 MultiDDS 및 MultiUAT와 같은 기존의 동적 데이터셋 혼합 방법보다도 우수한 성능을 보였습니다.

이러한 결과는 DYNAMIX SFT의 동적 혼합 최적화 방법이 기존의 정적 혼합 방법보다 더 효과적임을 보여줍니다. DYNAMIX SFT는 각 데이터셋의 기여도를 실시간으로 평가하고, 이를 기반으로 샘플링 비율을 조정하여 모델의 성능을 극대화하는 데 기여합니다.

---



DYNAMIX SFT conducted experiments using the TULU-v2 mixture collection, which consists of 16 diverse instruction-tuning datasets. The experiments utilized the LLaMA3.2 1B model and the Qwen2.5 3B model to evaluate performance across 10 benchmarks. The evaluated benchmarks include knowledge, reasoning, coding, mathematics, and general instruction following.

#### Competing Models
DYNAMIX SFT was compared against two groups of baselines:
1. **Heuristic Methods**: 
   - **Full Coverage**: Utilizes proportional sampling to ensure all datasets are included uniformly.
   - **Uniform Sampling**: Samples all datasets at the same rate.

2. **Dynamic Methods**:
   - **MultiDDS**: Adjusts the mixture based on gradient cosine similarity between datasets.
   - **MultiUAT**: Adjusts the mixture based on model uncertainty.

#### Test Data and Metrics
DYNAMIX SFT utilized approximately 300K examples from the TULU-v2 mixture collection and evaluated performance across 10 benchmarks. Each benchmark was assessed using the following metrics:
- **MMLU**: General knowledge evaluation.
- **TruthfulQA**: Evaluation of the model's truthfulness.
- **PopQA**: Evaluation of responses to popular questions.
- **BigBench-Hard**: Evaluation of complex reasoning problems.
- **DROP**: Reading comprehension evaluation.
- **HumanEval**: Code generation evaluation.
- **GSM8K**: Evaluation of mathematical problem-solving.
- **MATH**: Evaluation of mathematical problem-solving.
- **IFEval**: Evaluation of instruction following.

#### Comparison Results
DYNAMIX SFT achieved the following results:
- Compared to **Full Coverage**, DYNAMIX SFT showed a +2.2% improvement in average performance for the 1B model and a +1.5% improvement for the 3B model.
- DYNAMIX SFT also outperformed existing dynamic dataset mixture methods such as MultiDDS and MultiUAT.

These results demonstrate that the dynamic mixture optimization method of DYNAMIX SFT is more effective than traditional static mixture methods. DYNAMIX SFT evaluates the contribution of each dataset in real-time and adjusts the sampling ratios accordingly, thereby maximizing the model's performance.


<br/>
# 예제



DYNAMIX SFT는 다양한 지침 조정 데이터셋의 혼합을 최적화하기 위한 동적 방법론입니다. 이 방법은 다중 무장 밴딧(Multi-Armed Bandit, MAB) 문제로 데이터셋 혼합 문제를 공식화하고, 각 데이터셋을 하나의 팔(arm)로 간주하여 샘플링 확률을 동적으로 조정합니다. 

#### 예시

1. **트레이닝 데이터**: DYNAMIX SFT는 TULU-v2 혼합 컬렉션을 사용하여 16개의 지침 조정 데이터셋으로부터 약 300,000개의 예제를 포함합니다. 이 데이터셋은 FLAN, Open Assistant, ShareGPT, GPT4-Alpaca 등 다양한 출처에서 수집된 데이터로 구성되어 있습니다.

2. **테스트 데이터**: 모델의 성능은 10개의 벤치마크(예: MMLU, TruthfulQA, HumanEval 등)를 통해 평가됩니다. 각 벤치마크는 특정한 과제를 수행하기 위한 질문과 답변 형식으로 구성되어 있습니다.

3. **구체적인 인풋과 아웃풋**:
   - **인풋**: 예를 들어, MMLU 벤치마크에서는 "다음 중 가장 큰 숫자는 무엇인가요?"라는 질문이 주어질 수 있습니다. 이 질문은 모델이 다양한 지식 기반에서 답변을 생성하도록 요구합니다.
   - **아웃풋**: 모델은 "1000"과 같은 답변을 생성할 수 있으며, 이는 주어진 질문에 대한 올바른 답변입니다.

4. **구체적인 테스크**: DYNAMIX SFT는 각 데이터셋의 기여도를 평가하기 위해 1-Step Look-ahead Reward를 사용합니다. 이 방법은 각 데이터셋에서 샘플을 추출한 후, 모델의 손실이 얼마나 감소하는지를 측정하여 해당 데이터셋의 유용성을 평가합니다. 예를 들어, 특정 데이터셋에서 100개의 샘플을 사용하여 모델을 업데이트한 후, 손실이 0.5에서 0.3으로 감소했다면, 이 데이터셋은 모델 성능 향상에 기여한 것으로 평가됩니다.

이러한 방식으로 DYNAMIX SFT는 데이터셋 혼합을 동적으로 조정하여 모델의 성능을 최적화합니다.

---




DYNAMIX SFT is a dynamic methodology for optimizing the mixture of diverse instruction-tuning datasets. This approach formulates the dataset mixture problem as a Multi-Armed Bandit (MAB) problem, treating each dataset as an arm and dynamically adjusting the sampling probabilities.

#### Example

1. **Training Data**: DYNAMIX SFT utilizes the TULU-v2 mixture collection, which includes approximately 300,000 examples from 16 instruction-tuning datasets. These datasets are composed of data collected from various sources, including FLAN, Open Assistant, ShareGPT, and GPT4-Alpaca.

2. **Test Data**: The model's performance is evaluated across 10 benchmarks (e.g., MMLU, TruthfulQA, HumanEval, etc.). Each benchmark consists of questions and answers designed to assess specific tasks.

3. **Specific Inputs and Outputs**:
   - **Input**: For instance, in the MMLU benchmark, a question might be posed as "What is the largest number among the following?" This question requires the model to generate answers based on a wide range of knowledge.
   - **Output**: The model might generate an answer like "1000," which is the correct response to the given question.

4. **Specific Task**: DYNAMIX SFT employs a 1-Step Look-ahead Reward to evaluate the contribution of each dataset. This method involves extracting samples from each dataset and measuring how much the model's loss decreases as a result. For example, if the model's loss decreases from 0.5 to 0.3 after updating with 100 samples from a specific dataset, that dataset is considered beneficial for improving model performance.

In this way, DYNAMIX SFT dynamically adjusts the dataset mixture to optimize the model's performance.

<br/>
# 요약


DYNAMIX SFT는 다수의 인스트럭션 튜닝 데이터셋의 혼합을 최적화하기 위해 다중 무장 밴딧 문제로 설정하고, 원래 데이터셋 비율에 부드럽게 고정된 Prior-scaled Boltzmann Exploration을 도입하여 샘플링 분포를 업데이트합니다. 실험 결과, DYNAMIX SFT는 TULU-v2 혼합 컬렉션에서 최대 2.2%의 성능 향상을 달성하며, 10개의 벤치마크에서 우수한 성능을 보였습니다. 이 방법은 데이터셋의 다양성과 범위를 유지하면서도 모델의 학습 동향에 적응할 수 있는 효과적인 혼합 최적화 방법을 제공합니다.

---

DYNAMIX SFT formulates the optimization of multiple instruction-tuning dataset mixtures as a multi-armed bandit problem and introduces a Prior-scaled Boltzmann Exploration to softly anchor the sampling distribution to the original dataset proportions. Experimental results show that DYNAMIX SFT achieves up to a 2.2% performance improvement on the TULU-v2 mixture collection, demonstrating superior performance across 10 benchmarks. This method provides an effective mixture optimization approach that adapts to the model's learning dynamics while preserving the diversity and coverage of the datasets.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Mixture Proportions over Time**: DYNAMIX SFT는 훈련 과정에서 데이터셋의 샘플링 비율을 동적으로 조정하여, 각 데이터셋의 특성을 반영한 균형 잡힌 인스턴스 커버리지를 달성합니다. 반면, Full Coverage와 Uniform Sampling은 정적 비율을 유지하여 데이터셋의 자연스러운 특성을 왜곡할 수 있습니다.
   - **Update Interval**: 업데이트 간격이 50 스텝일 때 성능이 가장 좋았으며, 짧은 간격은 불안정성을 초래하고 긴 간격은 적응 속도를 저하시킵니다. 이는 적절한 업데이트 빈도가 반응성과 안정성의 균형을 이룬다는 것을 보여줍니다.
   - **Exploitation vs. Exploration**: β와 γ의 조합에 따라 성능이 달라지며, 강한 탐색 설정에서는 낮은 β 값에서 성능이 최고조에 달하는 반면, 약한 탐색 설정에서는 높은 β 값에서 더 나은 성능을 보입니다. 이는 탐색과 활용의 균형이 중요함을 나타냅니다.

2. **테이블**
   - **Performance Comparison**: DYNAMIX SFT는 Full Coverage와 Uniform Sampling에 비해 평균 성능에서 각각 +2.2% 및 +1.5%의 상대적 개선을 보였습니다. 이는 DYNAMIX SFT의 동적 샘플링 전략이 정적 샘플링 전략보다 효과적임을 나타냅니다.
   - **Impact of Prior-scaled Boltzmann Exploration**: Prior-scaled Boltzmann Exploration을 제거했을 때, DYNAMIX SFT는 초기 균일 분포에서 시작하여 적응력이 제한된 결과를 보였습니다. 이는 초기 분포의 중요성을 강조합니다.

3. **어펜딕스**
   - **Implementation Details**: DYNAMIX SFT의 하이퍼파라미터 설정(γ=0.3, α=0.95, β=4 등)은 성능에 큰 영향을 미치며, 이러한 설정은 실험에서 최적의 결과를 도출하는 데 기여했습니다.
   - **Details of Dataset Collection**: TULU-v2-mixture 컬렉션은 16개의 다양한 instruction-tuning 데이터셋으로 구성되어 있으며, 각 데이터셋의 특성을 반영한 샘플링이 중요합니다.

---




1. **Diagrams and Figures**
   - **Mixture Proportions over Time**: DYNAMIX SFT dynamically adjusts the sampling proportions of datasets during training, achieving balanced instance coverage that reflects the characteristics of each dataset. In contrast, Full Coverage and Uniform Sampling maintain static ratios, which can distort the natural characteristics of the datasets.
   - **Update Interval**: The best performance was observed with an update interval of 50 steps, where shorter intervals introduced instability and longer intervals slowed adaptation. This indicates that an appropriate update frequency strikes a balance between responsiveness and stability.
   - **Exploitation vs. Exploration**: The performance varies with the combination of β and γ, where strong exploration settings peak at lower β values, while weaker exploration settings perform better at higher β values. This highlights the importance of balancing exploration and exploitation.

2. **Tables**
   - **Performance Comparison**: DYNAMIX SFT achieved a relative improvement of +2.2% and +1.5% in average performance compared to Full Coverage and Uniform Sampling, respectively. This indicates that the dynamic sampling strategy of DYNAMIX SFT is more effective than static sampling strategies.
   - **Impact of Prior-scaled Boltzmann Exploration**: When the Prior-scaled Boltzmann Exploration was removed, DYNAMIX SFT started from a uniform distribution and showed limited adaptability. This emphasizes the importance of the initial distribution.

3. **Appendices**
   - **Implementation Details**: The hyperparameter settings for DYNAMIX SFT (γ=0.3, α=0.95, β=4, etc.) significantly impact performance, and these settings contributed to achieving optimal results in the experiments.
   - **Details of Dataset Collection**: The TULU-v2-mixture collection consists of 16 diverse instruction-tuning datasets, and sampling that reflects the characteristics of each dataset is crucial for effective mixture optimization.

<br/>
# refer format:


### BibTeX 

```bibtex
@article{Shin2025,
  author = {Haebin Shin and Lei Ji and Xiao Liu and Zhiwei Yu and Qi Chen and Yeyun Gong},
  title = {DYNAMIX SFT: Dynamic Mixture Optimization of Instruction Tuning Collections},
  journal = {arXiv preprint arXiv:2508.12116},
  year = {2025},
  url = {https://arxiv.org/abs/2508.12116}
}
```

### 시카고 스타일

Haebin Shin, Lei Ji, Xiao Liu, Zhiwei Yu, Qi Chen, and Yeyun Gong. 2025. "DYNAMIX SFT: Dynamic Mixture Optimization of Instruction Tuning Collections." arXiv preprint arXiv:2508.12116. https://arxiv.org/abs/2508.12116.
