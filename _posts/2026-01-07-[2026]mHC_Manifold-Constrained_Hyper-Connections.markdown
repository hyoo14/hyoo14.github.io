---
layout: post
title:  "[2026]mHC: Manifold-Constrained Hyper-Connections"
date:   2026-01-07 17:09:58 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

Manifold-Constrained Hyper-Connections (mHC)라는 일반적인 프레임워크를 제안합니다. mHC는 HC의 잔여 연결(Residual Connection) 공간을 특정 매니폴드에 투영하여 정체성 매핑 속성을 복원하고, 효율성을 보장하기 위한 엄격한 인프라 최적화를 포함  

즉, 이전에는 레지두얼 커넥션에서 직전 층의 정보만을 더해주는 방식이었다면, mHC는 모든 출력 신호에 직접 접근하는 방식  
단, 모든 층 접근시 신호폭주 및 연산 복잡도 문제가 있으니 매니폴드 개념 도입(각 층 다양 정보들이 막 섞이지 않도록 공간의 기하학적 구조를 유지하도록 강제하는것)  



짧은 요약(Abstract) :





최근 연구들은 Hyper-Connections (HC)와 같은 방법을 통해 잔여 연결(residual connection) 패러다임을 확장하여 성능을 크게 향상시켰습니다. 그러나 이러한 다양화는 잔여 연결의 고유한 정체성 매핑 속성을 손상시켜 훈련의 불안정성과 확장성 제한을 초래하며, 메모리 접근 오버헤드도 증가시킵니다. 이러한 문제를 해결하기 위해, 우리는 Manifold-Constrained Hyper-Connections (mHC)라는 일반적인 프레임워크를 제안합니다. mHC는 HC의 잔여 연결 공간을 특정 매니폴드에 투영하여 정체성 매핑 속성을 복원하고, 효율성을 보장하기 위한 엄격한 인프라 최적화를 포함합니다. 실험 결과, mHC는 대규모 훈련에서 효과적이며, 성능 개선과 우수한 확장성을 제공합니다. 우리는 mHC가 HC의 유연하고 실용적인 확장으로서, 토폴로지 아키텍처 설계에 대한 더 깊은 이해를 제공하고, 기초 모델의 발전을 위한 유망한 방향을 제시할 것이라고 기대합니다.




Recent studies exemplified by Hyper-Connections (HC) have extended the ubiquitous residual connection paradigm established over the past decade by expanding the residual stream width and diversifying connectivity patterns. While yielding substantial performance gains, this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, which causes severe training instability and restricted scalability, and additionally incurs notable memory access overhead. To address these challenges, we propose Manifold-Constrained Hyper-Connections (mHC), a general framework that projects the residual connection space of HC onto a specific manifold to restore the identity mapping property, while incorporating rigorous infrastructure optimization to ensure efficiency. Empirical experiments demonstrate that mHC is effective for training at scale, offering tangible performance improvements and superior scalability. We anticipate that mHC, as a flexible and practical extension of HC, will contribute to a deeper understanding of topological architecture design and suggest promising directions for the evolution of foundational models.


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


**방법론 (Method)**

이 논문에서는 Manifold-Constrained Hyper-Connections (mHC)라는 새로운 아키텍처를 제안합니다. mHC는 기존의 Hyper-Connections (HC) 아키텍처의 한계를 극복하기 위해 설계되었습니다. HC는 잔차 연결(residual connection)의 폭을 확장하고 연결 패턴을 다양화하여 성능을 향상시키지만, 이러한 다양화는 본질적으로 잔차 연결의 정체성 매핑(identity mapping) 속성을 손상시켜 훈련의 불안정성과 확장성의 제한을 초래합니다. mHC는 이러한 문제를 해결하기 위해 잔차 연결 공간을 특정 매니폴드(manifold)로 투영하여 정체성 매핑 속성을 복원합니다.

mHC의 핵심 아이디어는 잔차 매핑 \( H_{res} \)를 이중 확률 행렬(doubly stochastic matrix)로 제한하는 것입니다. 이중 확률 행렬은 모든 행과 열의 합이 1인 비음수 행렬로, 이는 신호 전파의 안정성을 보장합니다. mHC는 Sinkhorn-Knopp 알고리즘을 사용하여 \( H_{res} \)를 이중 확률 행렬로 변환합니다. 이 과정은 신호의 평균을 보존하고 신호의 노름을 엄격하게 정규화하여 신호의 소실(vanishing) 또는 폭발(exploding) 문제를 완화합니다.

mHC는 또한 효율성을 보장하기 위해 여러 가지 인프라 최적화를 포함합니다. 예를 들어, 커널 융합(kernel fusion) 기법을 사용하여 메모리 대역폭을 최적화하고, 선택적 재계산(selective recomputing)을 통해 메모리 사용량을 줄입니다. 이러한 최적화는 mHC가 대규모 훈련에서 안정성과 확장성을 유지하면서도 성능을 향상시킬 수 있도록 합니다.

mHC는 다양한 실험을 통해 그 효과를 입증하였으며, 대규모 훈련에서 6.7%의 추가 시간 오버헤드만으로도 훈련이 가능하다는 것을 보여주었습니다. 이로 인해 mHC는 대규모 언어 모델 훈련에 적합한 아키텍처로 자리 잡을 것으로 기대됩니다.



In this paper, we propose a new architecture called Manifold-Constrained Hyper-Connections (mHC). mHC is designed to overcome the limitations of the existing Hyper-Connections (HC) architecture. While HC enhances performance by expanding the width of the residual connection and diversifying connectivity patterns, this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, leading to training instability and restricted scalability. mHC addresses these challenges by projecting the residual connection space onto a specific manifold to restore the identity mapping property.

The core idea of mHC is to constrain the residual mapping \( H_{res} \) to be a doubly stochastic matrix. A doubly stochastic matrix is a non-negative matrix where the sums of all rows and columns equal 1, which ensures the stability of signal propagation. mHC employs the Sinkhorn-Knopp algorithm to transform \( H_{res} \) into a doubly stochastic matrix. This process effectively preserves the mean of the signals and strictly regularizes the signal norm, mitigating the risks of vanishing or exploding signals.

mHC also incorporates various infrastructure optimizations to ensure efficiency. For instance, it utilizes kernel fusion techniques to optimize memory bandwidth and employs selective recomputing to reduce memory usage. These optimizations allow mHC to maintain stability and scalability during large-scale training while improving performance.

Extensive experiments demonstrate the effectiveness of mHC, showing that it can support training at scale with only a 6.7% additional time overhead. As a result, mHC is expected to establish itself as a suitable architecture for large-scale language model training.


<br/>
# Results



이 논문에서는 Manifold-Constrained Hyper-Connections(mHC) 모델의 성능을 기존의 Baseline 및 Hyper-Connections(HC) 모델과 비교하여 평가하였습니다. 실험은 다양한 다운스트림 벤치마크에서 수행되었으며, 각 모델의 성능은 여러 메트릭을 통해 측정되었습니다.

#### 1. 경쟁 모델
- **Baseline**: 기존의 Residual Connection 구조를 사용하는 모델.
- **HC**: Hyper-Connections 구조를 도입하여 잔여 연결의 폭을 확장하고 연결 복잡성을 증가시킨 모델.
- **mHC**: 본 논문에서 제안한 모델로, HC의 잔여 연결 공간을 특정 매니폴드에 투영하여 안정성을 높이고 성능을 개선한 모델.

#### 2. 테스트 데이터
모델은 다양한 다운스트림 작업을 위한 데이터셋에서 평가되었습니다. 이 데이터셋은 BBH, DROP, GSM8K, HellaSwag, MATH, MMLU, PIQA, TriviaQA 등으로 구성되어 있으며, 각 작업에 대해 서로 다른 샷 수(예: 3-shot, 8-shot 등)로 평가되었습니다.

#### 3. 메트릭
모델의 성능은 다음과 같은 메트릭을 통해 측정되었습니다:
- **EM (Exact Match)**: 정답과 모델의 예측이 완전히 일치하는 비율.
- **F1 Score**: 정답과 모델의 예측 간의 조화 평균.
- **Acc (Accuracy)**: 전체 예측 중 정답의 비율.

#### 4. 비교 결과
실험 결과는 다음과 같습니다:

| 벤치마크 | Baseline | HC | mHC |
|----------|----------|----|-----|
| BBH      | 43.8     | 48.9 | 51.0 |
| DROP     | 47.0     | 51.6 | 53.9 |
| GSM8K    | 46.7     | 53.2 | 53.8 |
| HellaSwag | 73.7    | 74.3 | 74.7 |
| MATH     | 22.0     | 26.4 | 26.0 |
| MMLU     | 59.0     | 63.0 | 63.4 |
| PIQA     | 78.5     | 79.9 | 80.5 |
| TriviaQA | 54.3     | 56.3 | 57.6 |

mHC는 대부분의 벤치마크에서 Baseline 및 HC 모델보다 우수한 성능을 보였습니다. 특히 BBH와 DROP에서 각각 2.1% 및 2.3%의 성능 향상을 기록하며, mHC의 효과적인 안정성과 확장성을 입증하였습니다.

#### 5. 결론
mHC는 기존의 HC 모델이 가진 불안정성을 극복하고, 대규모 훈련에서의 성능을 향상시키는 데 성공하였습니다. 이 모델은 잔여 연결의 매니폴드 제약을 통해 신호 전파의 안정성을 높이고, 다양한 다운스트림 작업에서 우수한 성능을 발휘함으로써, 차세대 기초 모델의 발전 방향을 제시하고 있습니다.

---



In this paper, the performance of the Manifold-Constrained Hyper-Connections (mHC) model was evaluated against the Baseline and Hyper-Connections (HC) models. The experiments were conducted on various downstream benchmarks, and the performance of each model was measured using several metrics.

#### 1. Competing Models
- **Baseline**: A model using the traditional Residual Connection structure.
- **HC**: A model that introduces Hyper-Connections to expand the width of the residual connections and increase connection complexity.
- **mHC**: The proposed model in this paper, which projects the residual connection space of HC onto a specific manifold to enhance stability and improve performance.

#### 2. Test Data
The models were evaluated on datasets for various downstream tasks. These datasets included BBH, DROP, GSM8K, HellaSwag, MATH, MMLU, PIQA, and TriviaQA, with different shot counts (e.g., 3-shot, 8-shot) for evaluation.

#### 3. Metrics
The performance of the models was measured using the following metrics:
- **EM (Exact Match)**: The percentage of predictions that exactly match the ground truth.
- **F1 Score**: The harmonic mean of precision and recall between the ground truth and model predictions.
- **Acc (Accuracy)**: The ratio of correct predictions to the total predictions.

#### 4. Comparison Results
The experimental results are as follows:

| Benchmark | Baseline | HC | mHC |
|-----------|----------|----|-----|
| BBH       | 43.8     | 48.9 | 51.0 |
| DROP      | 47.0     | 51.6 | 53.9 |
| GSM8K     | 46.7     | 53.2 | 53.8 |
| HellaSwag | 73.7     | 74.3 | 74.7 |
| MATH      | 22.0     | 26.4 | 26.0 |
| MMLU      | 59.0     | 63.0 | 63.4 |
| PIQA      | 78.5     | 79.9 | 80.5 |
| TriviaQA  | 54.3     | 56.3 | 57.6 |

mHC consistently outperformed both the Baseline and HC models across most benchmarks. Notably, it achieved performance improvements of 2.1% on BBH and 2.3% on DROP, demonstrating the effective stability and scalability of mHC.

#### 5. Conclusion
mHC successfully overcomes the instability associated with the existing HC model and enhances performance in large-scale training. By imposing manifold constraints on the residual connections, mHC improves the stability of signal propagation and demonstrates superior performance across various downstream tasks, suggesting promising directions for the evolution of next-generation foundational models.


<br/>
# 예제


이 논문에서는 Manifold-Constrained Hyper-Connections(mHC)라는 새로운 아키텍처를 제안하고, 이를 통해 대규모 트레이닝에서의 안정성과 성능 향상을 목표로 하고 있습니다. mHC는 Hyper-Connections(HC)의 한계를 극복하기 위해 설계되었습니다. HC는 잔여 연결(residual connection)의 폭을 확장하고 연결 패턴을 다양화하여 성능을 향상시키지만, 이로 인해 신호의 안정성이 저하되고 메모리 접근 비용이 증가하는 문제를 야기합니다.

#### 트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋

1. **트레이닝 데이터**:
   - **인풋**: 27B 모델을 위한 트레이닝 데이터는 약 262B 토큰으로 구성된 대규모 텍스트 데이터셋입니다. 이 데이터셋은 다양한 자연어 처리(NLP) 작업을 포함하고 있으며, 예를 들어 질문 응답, 텍스트 생성, 문서 요약 등이 있습니다.
   - **아웃풋**: 모델의 출력은 주어진 입력에 대한 예측 결과로, 예를 들어 질문에 대한 답변, 주어진 문장의 다음 단어, 또는 주어진 텍스트의 요약 등이 될 수 있습니다.

2. **테스트 데이터**:
   - **인풋**: 테스트 데이터는 3B, 9B, 27B 모델 각각에 대해 설정된 다양한 벤치마크 데이터셋으로 구성됩니다. 예를 들어, BBH(Benchmark for Big Bench), DROP(Discrete Reasoning Over Paragraphs), MATH(수학 문제 해결) 등의 데이터셋이 포함됩니다.
   - **아웃풋**: 각 테스트 데이터셋에 대한 모델의 성능은 정확도(Accuracy), 정밀도(Precision), F1 점수 등으로 평가됩니다. 예를 들어, BBH 데이터셋에서는 모델이 주어진 질문에 대해 얼마나 정확하게 답변하는지를 평가합니다.

#### 구체적인 테스크 예시
- **질문 응답**: 주어진 질문에 대해 모델이 적절한 답변을 생성하는 작업입니다. 예를 들어, "지구의 가장 큰 대륙은 무엇인가요?"라는 질문에 대해 "아시아"라는 답변을 생성하는 것입니다.
- **수학 문제 해결**: 주어진 수학 문제를 해결하는 작업으로, 예를 들어 "5 + 7은 얼마인가요?"라는 질문에 대해 "12"라는 답변을 생성하는 것입니다.
- **텍스트 요약**: 긴 문서를 주어진 길이로 요약하는 작업으로, 예를 들어 "이 문서는 인공지능의 발전에 대한 내용을 다루고 있습니다."라는 문장을 "인공지능 발전"으로 요약하는 것입니다.




This paper proposes a new architecture called Manifold-Constrained Hyper-Connections (mHC), aiming to enhance stability and performance in large-scale training. mHC is designed to overcome the limitations of Hyper-Connections (HC). While HC improves performance by expanding the width of residual connections and diversifying connectivity patterns, it introduces issues such as reduced signal stability and increased memory access costs.

#### Specific Inputs and Outputs of Training and Testing Data

1. **Training Data**:
   - **Input**: The training data for the 27B model consists of a large-scale text dataset containing approximately 262 billion tokens. This dataset includes various natural language processing (NLP) tasks, such as question answering, text generation, and document summarization.
   - **Output**: The model's output is the predicted result for the given input, which could be an answer to a question, the next word in a given sentence, or a summary of the provided text.

2. **Testing Data**:
   - **Input**: The testing data consists of various benchmark datasets set for the 3B, 9B, and 27B models. For example, datasets like BBH (Benchmark for Big Bench), DROP (Discrete Reasoning Over Paragraphs), and MATH (Mathematical Problem Solving) are included.
   - **Output**: The model's performance on each testing dataset is evaluated using metrics such as accuracy, precision, and F1 score. For instance, in the BBH dataset, the evaluation measures how accurately the model answers the given questions.

#### Specific Task Examples
- **Question Answering**: A task where the model generates an appropriate answer to a given question. For example, for the question "What is the largest continent on Earth?", the model should generate the answer "Asia."
- **Mathematical Problem Solving**: A task where the model solves a given math problem, such as answering "What is 5 + 7?" with "12."
- **Text Summarization**: A task where the model summarizes a long document into a specified length, for example, summarizing "This document discusses the advancements in artificial intelligence." into "Advancements in AI."

<br/>
# 요약


본 논문에서는 Manifold-Constrained Hyper-Connections(mHC)라는 새로운 프레임워크를 제안하여 Hyper-Connections(HC)의 신호 전파 안정성을 개선하고, 대규모 훈련에서의 성능을 향상시킨다. 실험 결과, mHC는 기존 HC보다 더 나은 안정성과 확장성을 보여주며, 다양한 벤치마크에서 성능 향상을 입증하였다. 특히, mHC는 훈련 과정에서의 손실 감소와 그래디언트 안정성을 크게 개선하였다.



This paper introduces a new framework called Manifold-Constrained Hyper-Connections (mHC) to enhance the signal propagation stability of Hyper-Connections (HC) and improve performance in large-scale training. Experimental results demonstrate that mHC exhibits better stability and scalability compared to the existing HC, achieving performance improvements across various benchmarks. Notably, mHC significantly enhances loss reduction and gradient stability during the training process.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: Residual Connection Paradigms
     - 이 그림은 표준 Residual Connection, Hyper-Connections (HC), 그리고 제안된 Manifold-Constrained Hyper-Connections (mHC)의 구조적 설계를 비교합니다. mHC는 HC의 잔여 연결 공간을 특정 매니폴드에 투영하여 안정성을 보장하는 데 중점을 둡니다.
   
   - **Figure 2**: Training Instability of Hyper-Connections (HC)
     - HC의 훈련 불안정성을 보여주는 그래프입니다. mHC와 비교했을 때 HC는 훈련 중 손실이 급증하는 경향을 보이며, 이는 신호의 폭발 또는 소실 문제를 나타냅니다.

   - **Figure 5**: Training Stability of Manifold-Constrained Hyper-Connections (mHC)
     - mHC가 HC와 비교하여 훈련 안정성을 어떻게 개선했는지를 보여줍니다. mHC는 손실 감소와 더 안정적인 그래디언트 노름을 달성하여 훈련의 안정성을 높입니다.

   - **Figure 7**: Propagation Stability of mHC
     - mHC의 신호 전파 안정성을 보여주는 그래프입니다. mHC는 HC에 비해 신호의 전파 안정성을 크게 향상시킵니다.

2. **테이블**
   - **Table 4**: System-level Benchmark Results for 27B Models
     - 다양한 다운스트림 벤치마크에서 Baseline, HC, mHC의 성능을 비교합니다. mHC는 대부분의 벤치마크에서 Baseline과 HC를 초과하는 성능을 보여줍니다. 이는 mHC가 모델의 추론 능력을 향상시키는 데 기여함을 나타냅니다.

   - **Table 3**: Stored and Recomputed Intermediate Activations
     - mHC의 훈련 중 메모리 사용량을 줄이기 위해 어떤 활성화가 저장되고 재계산되는지를 보여줍니다. 이 표는 메모리 효율성을 높이기 위한 전략을 강조합니다.

3. **어펜딕스**
   - **Appendix A.1**: Detailed Model Specifications and Hyper-parameters
     - 3B, 9B, 27B 모델의 아키텍처 구성과 하이퍼파라미터를 상세히 설명합니다. mHC와 HC의 확장 비율, Sinkhorn-Knopp 설정 등을 포함하여 실험에 사용된 최적화 및 훈련 프로토콜을 제공합니다.

### Insights and Results from Figures, Tables, and Appendix

1. **Diagrams and Figures**
   - **Figure 1**: Residual Connection Paradigms
     - This figure compares the structural designs of standard Residual Connections, Hyper-Connections (HC), and the proposed Manifold-Constrained Hyper-Connections (mHC). mHC focuses on projecting the residual connection space onto a specific manifold to ensure stability.

   - **Figure 2**: Training Instability of Hyper-Connections (HC)
     - This graph illustrates the training instability of HC compared to mHC. HC shows a tendency for loss surges during training, indicating issues with signal explosion or vanishing.

   - **Figure 5**: Training Stability of Manifold-Constrained Hyper-Connections (mHC)
     - This figure demonstrates how mHC improves training stability compared to HC, achieving a reduction in loss and maintaining a more stable gradient norm.

   - **Figure 7**: Propagation Stability of mHC
     - This graph shows the signal propagation stability of mHC, significantly enhancing stability compared to HC.

2. **Tables**
   - **Table 4**: System-level Benchmark Results for 27B Models
     - This table compares the performance of Baseline, HC, and mHC across various downstream benchmarks. mHC consistently outperforms both Baseline and HC, indicating its effectiveness in enhancing model reasoning capabilities.

   - **Table 3**: Stored and Recomputed Intermediate Activations
     - This table outlines which activations are stored and recomputed during training, highlighting strategies to improve memory efficiency.

3. **Appendix**
   - **Appendix A.1**: Detailed Model Specifications and Hyper-parameters
     - This section provides detailed architectural configurations and hyper-parameters for the 3B, 9B, and 27B models, including the expansion rate for mHC and HC, as well as the settings for the Sinkhorn-Knopp algorithm used in the experiments.

<br/>
# refer format:
### BibTeX 


```bibtex
@article{xie2026mHC,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Zhenda Xie and Yixuan Wei and Huanqi Cao and Chenggang Zhao and Chengqi Deng and Jiashi Li and Damai Dai and Huazuo Gao and Jiang Chang and Kuai Yu and Liang Zhao and Shangyan Zhou and Zhean Xu and Zhengyan Zhang and Wangding Zeng and Shengding Hu and Yuqing Wang and Jingyang Yuan and Lean Wang and Wenfeng Liang},
  journal={arXiv preprint arXiv:2512.24880},
  year={2026},
  url={https://arxiv.org/abs/2512.24880}
}
```

### 시카고 스타일

Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, and Wenfeng Liang. "mHC: Manifold-Constrained Hyper-Connections." arXiv preprint arXiv:2512.24880 (2026). https://arxiv.org/abs/2512.24880.
