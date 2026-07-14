---
layout: post
title:  "[2026]CoopQ: Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs"
date:   2026-07-14 00:31:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 CoopQ라는 새로운 혼합 정밀도 양자화 방법을 제안하며, 이를 통해 레이어 간 상호작용을 고려하여 모델 성능을 최적화한다.


짧은 요약(Abstract) :



대형 언어 모델(LLM)은 인상적인 능력을 제공하지만, 수십억 개의 매개변수로 인해 장치에서의 배포가 어렵습니다. 혼합 정밀도 양자화는 이러한 문제를 해결할 수 있는 유망한 방법이지만, 기존 방법들은 평균 정밀도가 4비트 이하로 떨어질 때 성능 저하를 겪습니다. 이는 개별 레이어의 메트릭에 의존하여 레이어 간의 상호작용을 간과하기 때문입니다. 이러한 한계를 극복하기 위해, 우리는 혼합 정밀도 양자화 문제를 레이어 간의 협력 게임으로 설정하고, 레이어의 민감도와 상호작용을 효율적으로 추정하기 위해 Shapley 기반의 점진적 양자화 추정(SPQE)을 도입합니다. SPQE 추정치를 활용하여, CoopQ라는 협력 게임에서 영감을 받은 혼합 정밀도 양자화 방법을 제안하며, 이는 레이어에 2비트 또는 4비트 정밀도를 할당하는 이진 이차 최적화 문제로 변환됩니다. Llama-3, Gemma-2, Qwen-3 모델에 대한 포괄적인 실험을 통해 CoopQ의 확장성과 기존 방법들에 비해 일관되게 우수한 성능을 입증하였습니다. 평균 정밀도가 2비트에서 4비트에 이르는 범위에서 CoopQ는 최상의 기준선에 비해 20%에서 80%까지 Perplexity를 줄였습니다.




Large Language Models (LLMs) promise impressive capabilities, yet their multi-billion-parameter scale makes on-device or low-resource deployment prohibitive. Mixed-precision quantization offers a compelling solution, but existing methods struggle when the average precision drops below four bits, as they rely on isolated, layer-specific metrics that overlook critical inter-layer interactions affecting overall performance. To address these limitations, we first frame the mixed-precision quantization problem as a cooperative game among layers and introduce Shapley-based Progressive Quantization Estimation (SPQE) to efficiently obtain accurate Shapley estimates of layer sensitivities and inter-layer interactions. Leveraging the SPQE estimates, we propose Cooperative Game Inspired Mixed-Precision Quantization (CoopQ), which translates these Shapley estimates into a binary quadratic optimization formulation, assigning either 2 or 4-bit precision to layers under strict memory constraints. Comprehensive experiments conducted on Llama-3, Gemma-2, and Qwen-3 models demonstrate CoopQ’s scalability and consistently superior performance compared to methods relying solely on isolated metrics. Across average precisions spanning 4 bits down to 2 bits, CoopQ cuts Perplexity by 20% to 80% relative to the best baseline, with the margin growing as the bit-width tightens.


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


이 논문에서는 대형 언어 모델(LLM)의 혼합 정밀도 양자화 문제를 해결하기 위해 협력 게임 이론을 기반으로 한 새로운 접근 방식을 제안합니다. 이 방법은 두 가지 주요 구성 요소로 이루어져 있습니다: **Shapley 기반의 점진적 양자화 추정(SPQE)**와 **협력 게임 기반 혼합 정밀도 양자화(CoopQ)**입니다.

1. **Shapley 기반의 점진적 양자화 추정(SPQE)**:
   - SPQE는 각 Transformer 레이어의 중요성을 평가하기 위해 Shapley 값 이론을 활용합니다. Shapley 값은 협력 게임 이론에서 각 플레이어(여기서는 레이어)가 팀의 성과에 기여하는 정도를 정량화하는 방법입니다.
   - SPQE는 모든 레이어를 먼저 고정된 높은 정밀도로 양자화한 후, 각 레이어의 정밀도를 점진적으로 낮추면서 모델의 성능 변화를 관찰합니다. 이를 통해 각 레이어의 기여도를 정확하게 추정할 수 있습니다.
   - Monte Carlo 샘플링을 사용하여 레이어의 조합에 따른 성능 변화를 측정하고, 이를 통해 레이어의 민감도와 상호작용을 평가합니다. 이 과정은 레이어의 양자화가 모델 성능에 미치는 영향을 보다 안정적으로 추정할 수 있게 합니다.

2. **협력 게임 기반 혼합 정밀도 양자화(CoopQ)**:
   - CoopQ는 SPQE에서 얻은 Shapley 값을 기반으로 각 레이어에 2비트 또는 4비트의 정밀도를 할당하는 최적화 문제를 설정합니다. 이 과정은 레이어 간의 상호작용을 고려하여 양자화 오류가 모델 성능에 미치는 영향을 최소화하는 방향으로 진행됩니다.
   - 이 최적화 문제는 이진 정수 선형 프로그래밍(MILP)으로 변환되어, 각 레이어에 대한 최적의 비트 할당을 찾습니다. 이 과정에서 메모리 제약을 고려하여 레이어의 정밀도를 조정합니다.
   - CoopQ는 기존의 고립된 메트릭에 의존하는 방법들보다 우수한 성능을 보이며, 다양한 모델과 PTQ 백엔드에서 일관되게 더 낮은 Perplexity 값을 달성합니다.

이러한 방법론은 LLM의 양자화 과정에서 레이어 간의 상호작용을 모델링하는 것이 중요하다는 점을 강조하며, 기존의 독립적인 레이어 평가 방식에서 벗어나 레이어 간의 상호작용을 고려하는 새로운 접근 방식을 제시합니다.

---




This paper proposes a novel approach to address the mixed-precision quantization problem of large language models (LLMs) based on cooperative game theory. The method consists of two main components: **Shapley-based Progressive Quantization Estimation (SPQE)** and **Cooperative Game Inspired Mixed-Precision Quantization (CoopQ)**.

1. **Shapley-based Progressive Quantization Estimation (SPQE)**:
   - SPQE leverages Shapley value theory to evaluate the importance of each Transformer layer. The Shapley value quantifies the contribution of each player (in this case, layers) to the overall performance of the team.
   - SPQE first uniformly quantizes all layers to a fixed high precision and then progressively reduces the precision of each layer while observing the model's performance changes. This allows for accurate estimation of each layer's contribution.
   - Monte Carlo sampling is employed to measure performance changes based on different combinations of layers, enabling the assessment of layer sensitivities and interactions. This process allows for a more stable estimation of the impact of layer quantization on model performance.

2. **Cooperative Game Inspired Mixed-Precision Quantization (CoopQ)**:
   - CoopQ formulates an optimization problem to assign either 2-bit or 4-bit precision to each layer based on the Shapley values obtained from SPQE. This process minimizes the impact of quantization errors on model performance by considering inter-layer interactions.
   - The optimization problem is transformed into a Mixed-Integer Linear Programming (MILP) problem, allowing for the determination of optimal bit assignments for each layer while respecting memory constraints.
   - CoopQ consistently outperforms methods relying solely on isolated metrics, achieving lower Perplexity values across various models and PTQ backends.

These methodologies emphasize the importance of modeling inter-layer dependencies in the quantization process of LLMs, presenting a shift from viewing layers as independent entities to understanding them as interconnected components whose collective behavior dictates the final performance of the quantized model.


<br/>
# Results



이 논문에서는 CoopQ라는 새로운 혼합 정밀도 양자화 방법을 제안하고, 이를 통해 대형 언어 모델(LLM)의 성능을 향상시키는 방법을 다룹니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: CoopQ는 세 가지 모델인 Gemma-2, Llama-3, Qwen-3을 대상으로 평가되었습니다. 각 모델은 서로 다른 크기(2B, 9B, 3.2B, 8B 등)로 구성되어 있습니다.

2. **테스트 데이터**: 성능 평가는 WikiText-2 데이터셋을 사용하여 수행되었습니다. 이 데이터셋은 언어 모델의 성능을 평가하는 데 널리 사용되는 벤치마크입니다.

3. **메트릭**: 주요 성능 지표로는 Perplexity가 사용되었습니다. Perplexity는 모델의 예측 능력을 측정하는 지표로, 값이 낮을수록 모델의 성능이 우수함을 나타냅니다.

4. **비교**: CoopQ는 세 가지 기존의 포스트 트레이닝 양자화(PTQ) 방법인 Quanto, HQQ, GPTQ와 비교되었습니다. 실험 결과, CoopQ는 모든 모델에서 기존 방법들보다 우수한 성능을 보였습니다. 특히, CoopQ는 2비트에서 4비트 사이의 평균 정밀도에서 Perplexity를 20%에서 80%까지 감소시켰습니다.

5. **결과 분석**: 
   - Gemma-2-2B 모델에서 CoopQ는 Perplexity 48.52를 기록하여 Sensitivity 방법의 189.55와 LIM 방법의 214.03에 비해 각각 약 74%와 77% 감소했습니다.
   - Llama-3.2-3B 모델에서는 CoopQ가 Perplexity 73.11을 달성하여 Sensitivity의 343.64에 비해 79% 개선되었습니다.
   - Qwen3-4B 모델에서도 CoopQ는 Perplexity 697.28을 기록하여 Sensitivity의 1.56×103에 비해 55% 개선된 성능을 보였습니다.

이러한 결과는 CoopQ가 레이어 간 상호작용을 효과적으로 모델링하여 양자화 오류를 줄이고, 다양한 모델과 PTQ 백엔드에서 일관되게 우수한 성능을 발휘함을 보여줍니다.

---




This paper proposes a new mixed-precision quantization method called CoopQ, which aims to enhance the performance of large language models (LLMs). The main findings of the study are as follows:

1. **Competing Models**: CoopQ was evaluated on three models: Gemma-2, Llama-3, and Qwen-3. Each model consists of different sizes (2B, 9B, 3.2B, 8B, etc.).

2. **Test Data**: The performance evaluation was conducted using the WikiText-2 dataset, which is a widely used benchmark for assessing language model performance.

3. **Metrics**: The primary performance metric used was Perplexity. Perplexity measures the predictive capability of the model, with lower values indicating better performance.

4. **Comparison**: CoopQ was compared against three existing post-training quantization (PTQ) methods: Quanto, HQQ, and GPTQ. The experimental results showed that CoopQ consistently outperformed these baseline methods across all models. Notably, CoopQ reduced Perplexity by 20% to 80% across average precisions ranging from 2 bits to 4 bits.

5. **Result Analysis**:
   - For the Gemma-2-2B model, CoopQ achieved a Perplexity of 48.52, which is approximately 74% and 77% lower than the Sensitivity method (189.55) and the LIM method (214.03), respectively.
   - In the Llama-3.2-3B model, CoopQ reached a Perplexity of 73.11, representing a 79% improvement over Sensitivity's 343.64.
   - For the Qwen3-4B model, CoopQ recorded a Perplexity of 697.28, outperforming Sensitivity's 1.56×103 by 55%.

These results demonstrate that CoopQ effectively models inter-layer interactions, significantly reducing quantization errors and consistently achieving superior performance across various models and PTQ backends.


<br/>
# 예제



이 논문에서는 CoopQ라는 새로운 혼합 정밀도 양자화 방법을 제안하고 있습니다. 이 방법은 대형 언어 모델(LLM)의 성능을 유지하면서 메모리 사용량을 줄이기 위해 레이어 간의 상호작용을 고려합니다. 연구의 주요 목표는 양자화 과정에서 레이어의 중요성을 평가하고, 이를 통해 최적의 비트 할당을 결정하는 것입니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**: 
   - **데이터셋**: C4 데이터셋 (Colossal Clean Crawled Corpus)
   - **입력**: 문장 또는 문서의 텍스트 조각
   - **출력**: 다음 단어의 확률 분포 (예: "나는 오늘 아침에"라는 입력에 대해 "커피"라는 단어의 확률)

2. **테스트 데이터**:
   - **데이터셋**: WikiText-2
   - **입력**: 문장 또는 문서의 텍스트 조각
   - **출력**: 모델이 생성한 다음 단어의 확률 분포와 실제 다음 단어의 비교를 통해 계산된 Perplexity 값

#### 구체적인 작업(Task)
- **작업**: LLM의 양자화
- **목표**: 모델의 메모리 사용량을 줄이면서 성능 저하를 최소화하는 것
- **방법론**:
  - SPQE(Shapley-based Progressive Quantization Estimation)를 사용하여 각 레이어의 중요성을 평가
  - CoopQ를 통해 최적의 비트 할당을 결정
  - 다양한 비트 폭(2비트, 4비트)에서 모델의 Perplexity를 측정하여 성능을 평가

이러한 방식으로, 연구자들은 CoopQ가 기존의 방법들보다 더 나은 성능을 발휘한다는 것을 입증하였습니다. 특히, 2비트에서 4비트 사이의 평균 정밀도에서 Perplexity를 20%에서 80%까지 줄일 수 있었습니다.

---




This paper proposes a new mixed-precision quantization method called CoopQ, which considers inter-layer interactions to maintain the performance of large language models (LLMs) while reducing memory usage. The main goal of the research is to evaluate the importance of layers during the quantization process and determine optimal bit allocation based on this evaluation.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Dataset**: C4 dataset (Colossal Clean Crawled Corpus)
   - **Input**: Text snippets of sentences or documents
   - **Output**: Probability distribution of the next word (e.g., given the input "I had this morning," the output might be the probability of the word "coffee")

2. **Test Data**:
   - **Dataset**: WikiText-2
   - **Input**: Text snippets of sentences or documents
   - **Output**: The probability distribution of the next word generated by the model compared to the actual next word, resulting in a calculated Perplexity value

#### Specific Task
- **Task**: Quantization of LLMs
- **Objective**: To minimize performance degradation while reducing the model's memory usage
- **Methodology**:
  - Use SPQE (Shapley-based Progressive Quantization Estimation) to evaluate the importance of each layer
  - Determine optimal bit allocation through CoopQ
  - Measure the model's Perplexity across various bit widths (2-bit, 4-bit) to evaluate performance

Through this approach, the researchers demonstrated that CoopQ outperforms existing methods, achieving a reduction in Perplexity by 20% to 80% across average precisions ranging from 2 to 4 bits.

<br/>
# 요약


이 논문에서는 CoopQ라는 새로운 혼합 정밀도 양자화 방법을 제안하며, 이를 통해 레이어 간 상호작용을 고려하여 모델 성능을 최적화한다. Shapley 기반의 점진적 양자화 추정(SPQE)을 사용하여 각 레이어의 민감도를 평가하고, 이를 바탕으로 이진 이차 최적화 문제를 해결하여 최적의 비트 할당을 수행한다. 실험 결과, CoopQ는 다양한 모델에서 기존 방법들보다 20%에서 80%까지 더 낮은 Perplexity를 달성하며, 특히 비트 폭이 제한될수록 그 효과가 두드러진다.

---

This paper proposes a new mixed-precision quantization method called CoopQ, which optimizes model performance by considering inter-layer interactions. It employs Shapley-based Progressive Quantization Estimation (SPQE) to assess the sensitivity of each layer and formulates a binary quadratic optimization problem to determine optimal bit allocation. Experimental results show that CoopQ achieves 20% to 80% lower Perplexity compared to existing methods across various models, with the effect being particularly pronounced under tighter bit constraints.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Perplexity 비교 (Figure 1)**: 다양한 양자화 방법(Activation, Sensitivity, LIM, Z-Score, CoopQ)에 대한 Perplexity를 비교한 결과, CoopQ가 모든 모델에서 가장 낮은 Perplexity 값을 기록했습니다. 특히, bit-width가 낮아질수록 CoopQ의 성능이 더욱 두드러졌습니다. 예를 들어, Gemma-2-2B 모델에서 CoopQ는 79% 이상의 Perplexity 감소를 달성했습니다.

2. **Wikitext-2 Perplexity 비교 (Figure 3, Figure 4)**: HQQ 및 Quanto 양자화 방법에 대한 Perplexity 비교 결과, CoopQ는 두 가지 백엔드 모두에서 다른 방법들보다 우수한 성능을 보였습니다. 특히, Qwen3-8B 모델에서 CoopQ는 HQQ 양자화 시 59%의 성능 향상을 보였습니다.

#### 테이블
1. **Perplexity 비교 테이블 (Table 1)**: 다양한 모델과 양자화 방법에 대한 Perplexity 값을 정리한 테이블로, CoopQ가 모든 bit-range에서 다른 방법들보다 우수한 성능을 보였습니다. 특히, 2.01-2.5 비트 범위에서 CoopQ는 79% 이상의 성능 향상을 기록했습니다.

2. **Ablation Study (Table 2)**: SPQE 샘플 수에 따른 Perplexity 변화를 보여주는 테이블로, 샘플 수가 증가할수록 Perplexity가 개선되는 경향을 보였습니다. 50개의 샘플에서 최대 개선이 나타났으며, 그 이후에는 개선폭이 줄어드는 경향을 보였습니다.

#### 어펜딕스
- **메모리 제약 조건 설명**: 각 Transformer 레이어가 2비트 또는 4비트로 양자화될 때의 메모리 예산을 정의하는 방법을 설명합니다. 이 예산은 모든 레이어가 2비트로 설정된 경우와 4비트로 설정된 경우의 메모리 발자국을 기반으로 선형 보간을 통해 계산됩니다.




#### Diagrams and Figures
1. **Perplexity Comparison (Figure 1)**: The comparison of Perplexity across various quantization methods (Activation, Sensitivity, LIM, Z-Score, CoopQ) shows that CoopQ consistently achieved the lowest Perplexity values across all models. Notably, as the bit-width decreased, the performance of CoopQ became even more pronounced. For instance, in the Gemma-2-2B model, CoopQ achieved over a 79% reduction in Perplexity.

2. **Wikitext-2 Perplexity Comparison (Figure 3, Figure 4)**: The comparison of Perplexity for HQQ and Quanto quantization methods indicates that CoopQ outperformed other methods across both backends. Specifically, in the Qwen3-8B model, CoopQ showed a 59% performance improvement when using HQQ quantization.

#### Tables
1. **Perplexity Comparison Table (Table 1)**: This table summarizes the Perplexity values for various models and quantization methods, demonstrating that CoopQ consistently outperformed other methods across all bit ranges. Particularly, in the 2.01-2.5 bit range, CoopQ recorded over a 79% performance improvement.

2. **Ablation Study (Table 2)**: This table illustrates the changes in Perplexity based on the number of SPQE samples, showing a trend of improvement as the sample count increased. The maximum improvement was observed at 50 samples, after which the gains diminished.

#### Appendix
- **Memory Constraint Formulation Details**: This section explains how to define the memory budget when quantizing each Transformer layer to either 2 bits or 4 bits. The budget is calculated based on the memory footprint when all layers are set to 2 bits and when all are set to 4 bits, using linear interpolation to determine the budget for a specific average bit-width.

<br/>
# refer format:

### BibTeX Citation

```bibtex
@inproceedings{zhao2026coopq,
  author = {Junchen Zhao and Ali Derakhshan and Jayden Hyman and Junhao Dong and Sangeetha Abdu Jyothi and Ian Harris},
  title = {CoopQ: Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages = {7566--7578},
  year = {2026},
  month = {July},
  publisher = {Association for Computational Linguistics},
  address = {UCIrvine}
}
```

### Chicago Style Citation

Zhao, Junchen, Ali Derakhshan, Jayden Hyman, Junhao Dong, Sangeetha Abdu Jyothi, and Ian Harris. 2026. "CoopQ: Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs." In *Findings of the Association for Computational Linguistics: ACL 2026*, 7566–7578. UCIrvine: Association for Computational Linguistics.
    