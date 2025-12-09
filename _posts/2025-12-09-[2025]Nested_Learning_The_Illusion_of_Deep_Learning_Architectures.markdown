---
layout: post
title:  "[2025]Nested Learning: The Illusion of Deep Learning Architectures"
date:   2025-12-09 19:59:45 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Nested Learning(NL)이라는 새로운 학습 패러다임을 제안하여, 기계 학습 모델을 다층적이고 병렬적인 최적화 문제의 집합으로 표현합니다.   

기존 딥러닝은한 단계짜리(1-level) 최적화 문제처럼 생각, 
논문이 말하는 **Nested Learning(NL)**은:   

모델 + 옵티마이저 + 메모리까지 전부
서로 다른 시간 스케일로 업데이트되는 여러 개의 최적화 문제들이 중첩된 시스템  

각 컴포넌트(파라미터나 메모리)는 자기만의 컨텍스트 흐름(context flow) 를 보고 
그걸 자기 파라미터 안에 압축해서 저장하는 일종의 연상 기억(associative memory)로 해석 



그래서 바깥에 큰 학습 루프가 있고
그 안에 또 다른 학습 루프(optimizer나 fast weight, attention 메모리 등) 가 들어있는 구조로 보면 됨!   


짧은 요약(Abstract) :


이 논문의 초록에서는 최근 몇 년간 신경망 아키텍처의 발전과 이를 효과적으로 훈련시키기 위한 최적화 알고리즘 설계가 기계 학습 모델의 능력을 향상시키기 위한 핵심 연구 노력의 일환으로 이루어졌음을 언급하고 있습니다. 특히 언어 모델(LLM)의 발전에도 불구하고, 이러한 모델들이 지속적으로 학습하고 기억하며 스스로 개선하고 "효과적인 해결책"을 찾는 데 있어 근본적인 도전과 해결되지 않은 질문들이 존재한다고 지적합니다. 이 논문에서는 "Nested Learning (NL)"이라는 새로운 학습 패러다임을 제안하며, 이는 다층적이고 병렬적인 최적화 문제 집합으로 모델을 일관되게 표현합니다. NL은 기존의 딥러닝 방법이 데이터로부터 자신의 맥락 흐름을 압축하여 학습한다는 것을 드러내고, 대형 모델에서 어떻게 인-컨텍스트 학습이 발생하는지를 설명합니다. NL은 더 많은 "레벨"을 가진 더 표현력이 풍부한 학습 알고리즘을 설계할 수 있는 경로를 제시하며, 세 가지 주요 기여를 통해 그 중요성을 주장합니다: (1) 딥 최적화기, (2) 자기 수정 타이탄, (3) 연속 메모리 시스템. 마지막으로, HOPE라는 학습 모듈을 제시하며, 이는 언어 모델링, 지속적 학습 및 긴 맥락 추론 작업에서 유망한 결과를 보여줍니다.



The abstract of this paper discusses that the development of neural architectures and the design of optimization algorithms to effectively train them have been core research efforts to enhance the capabilities of machine learning models in recent years. Despite advancements, particularly in language models (LLMs), there are fundamental challenges and unanswered questions regarding how such models can continually learn, memorize, self-improve, and find "effective solutions." The paper presents a new learning paradigm called "Nested Learning (NL)," which coherently represents a model as a set of nested, multi-level, and/or parallel optimization problems. NL reveals that existing deep learning methods learn from data by compressing their own context flow and explains how in-context learning emerges in large models. NL suggests a path to design more expressive learning algorithms with more "levels," resulting in higher-order in-context learning abilities. The authors advocate for its importance by presenting three core contributions: (1) Deep Optimizers, (2) Self-Modifying Titans, and (3) Continuum Memory System. Finally, they introduce a learning module called HOPE, which shows promising results in language modeling, continual learning, and long-context reasoning tasks.


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



이 논문에서 제안하는 방법론은 "Nested Learning" (NL)이라는 새로운 학습 패러다임을 기반으로 하고 있습니다. NL은 모델을 중첩된 다중 수준 및/또는 병렬 최적화 문제의 집합으로 표현하며, 각 문제는 고유한 "컨텍스트 흐름"을 가집니다. 이 접근법은 기존의 딥러닝 방법들이 데이터를 통해 자신의 컨텍스트 흐름을 압축하는 방식으로 학습한다는 점을 강조합니다. NL은 더 많은 "레벨"을 가진 표현력이 풍부한 학습 알고리즘을 설계할 수 있는 경로를 제시하며, 이는 고차원적인 인-컨텍스트 학습 능력을 향상시킵니다.

NL의 세 가지 주요 기여는 다음과 같습니다:

1. **Deep Optimizers**: NL에 기반하여, 잘 알려진 경량 기반 최적화 알고리즘(예: Adam, SGD with Momentum 등)이 사실상 기울기를 압축하는 연상 기억 모듈이라는 것을 보여줍니다. 이 통찰을 바탕으로, 더 표현력이 풍부한 최적화 알고리즘을 제시합니다.

2. **Self-Modifying Titans**: NL의 통찰을 활용하여, 자신의 업데이트 알고리즘을 학습함으로써 스스로를 수정하는 새로운 시퀀스 모델을 제안합니다.

3. **Continuum Memory System**: 전통적인 "장기/단기 기억" 관점을 일반화하는 새로운 메모리 시스템 공식을 제시합니다. 이 시스템은 자가 수정 시퀀스 모델과 결합되어 HOPE라는 학습 모듈을 형성하며, 언어 모델링, 지속적 학습 및 긴 컨텍스트 추론 작업에서 유망한 결과를 보여줍니다.

이러한 방법론은 뇌의 신경 과학적 관점에서 영감을 받아, 인간의 지속적 학습 능력을 모방하고자 하며, 이는 뇌의 신경 가소성과 관련이 있습니다. NL은 각 구성 요소가 고유한 업데이트 주기를 가지며, 이를 통해 더 나은 컨텍스트 관리와 지속적 학습을 가능하게 합니다.





The methodology proposed in this paper is based on a new learning paradigm called "Nested Learning" (NL). NL coherently represents a model as a set of nested, multi-level, and/or parallel optimization problems, each with its own "context flow." This approach emphasizes that existing deep learning methods learn from data by compressing their own context flow. NL suggests a path to design more expressive learning algorithms with more "levels," resulting in enhanced higher-order in-context learning abilities.

The three core contributions of NL are as follows:

1. **Deep Optimizers**: Based on NL, we show that well-known gradient-based optimizers (e.g., Adam, SGD with Momentum, etc.) are, in fact, associative memory modules that aim to compress gradients. Building on this insight, we present a set of more expressive optimizers.

2. **Self-Modifying Titans**: Taking advantage of NL's insights on learning algorithms, we propose a novel sequence model that learns how to modify itself by learning its own update algorithm.

3. **Continuum Memory System**: We present a new formulation for a memory system that generalizes the traditional viewpoint of "long-term/short-term memory." This system, combined with our self-modifying sequence model, forms a learning module called HOPE, which shows promising results in language modeling, continual learning, and long-context reasoning tasks.

These methodologies are inspired by neurophysiological perspectives, aiming to mimic the human brain's capacity for continual learning, which is related to neuroplasticity. NL allows for better context management and continual learning by enabling each component to have its own update frequency.


<br/>
# Results




이 논문에서는 HOPE(High-Order Parameterized Learning)라는 새로운 학습 모듈을 제안하고, 이를 기존의 여러 모델과 비교하여 성능을 평가하였다. 실험은 언어 모델링과 상식 추론 과제를 포함하며, 다양한 크기의 모델(340M, 760M, 1.3B 파라미터)을 사용하여 진행되었다.

1. **경쟁 모델**: HOPE는 여러 최신 모델과 비교되었다. 여기에는 Transformer++, RetNet, DeltaNet, Samba, Titans 등이 포함된다. 각 모델은 언어 모델링과 상식 추론 과제에서 성능을 평가받았다.

2. **테스트 데이터**: 실험에 사용된 데이터셋은 Wiki, LMB, PIQA, HellaSwag, ARC-e, ARC-c, SIQA, BoolQ 등으로, 다양한 자연어 처리(NLP) 과제를 포함하고 있다. 각 데이터셋은 모델의 일반화 능력과 상식 추론 능력을 평가하기 위해 선택되었다.

3. **메트릭**: 성능 평가는 주로 두 가지 메트릭을 사용하였다. 첫째, 언어 모델링에서는 Perplexity(ppl)로 모델의 예측 능력을 측정하였다. 둘째, 상식 추론 과제에서는 정확도(acc)를 사용하여 모델의 추론 능력을 평가하였다.

4. **비교 결과**: HOPE는 모든 테스트에서 경쟁 모델들보다 우수한 성능을 보였다. 예를 들어, 1.3B 파라미터 모델의 경우, HOPE는 Wiki 데이터셋에서 15.11의 perplexity를 기록하며, 이는 Transformer++의 18.53보다 낮은 수치이다. 또한, 상식 추론 과제에서도 HOPE는 높은 정확도를 기록하여, 기존 모델들보다 더 나은 성능을 입증하였다.

이러한 결과는 HOPE가 기존의 Transformer 기반 모델보다 더 효과적으로 언어 모델링과 상식 추론을 수행할 수 있음을 보여준다. HOPE는 더 깊고 표현력이 뛰어난 학습 알고리즘을 통해, 다양한 과제에서 우수한 성능을 발휘할 수 있는 가능성을 제시한다.

---




In this paper, a new learning module called HOPE (High-Order Parameterized Learning) is proposed, and its performance is evaluated against various existing models. The experiments include language modeling and commonsense reasoning tasks, utilizing models of different sizes (340M, 760M, 1.3B parameters).

1. **Competing Models**: HOPE was compared with several state-of-the-art models, including Transformer++, RetNet, DeltaNet, Samba, and Titans. Each model was evaluated on language modeling and commonsense reasoning tasks.

2. **Test Data**: The datasets used in the experiments include Wiki, LMB, PIQA, HellaSwag, ARC-e, ARC-c, SIQA, and BoolQ, covering a range of natural language processing (NLP) tasks. These datasets were selected to assess the models' generalization capabilities and commonsense reasoning abilities.

3. **Metrics**: Performance evaluation primarily utilized two metrics. First, for language modeling, Perplexity (ppl) was used to measure the model's predictive capability. Second, for commonsense reasoning tasks, accuracy (acc) was employed to evaluate the model's reasoning ability.

4. **Comparison Results**: HOPE demonstrated superior performance across all tests compared to competing models. For instance, the 1.3B parameter model of HOPE achieved a perplexity of 15.11 on the Wiki dataset, which is lower than Transformer++'s 18.53. Additionally, in commonsense reasoning tasks, HOPE recorded high accuracy, proving to be more effective than existing models.

These results indicate that HOPE can perform language modeling and commonsense reasoning tasks more effectively than traditional Transformer-based models. HOPE presents the potential for superior performance across various tasks through deeper and more expressive learning algorithms.


<br/>
# 예제



이 논문에서는 Nested Learning(NL)이라는 새로운 학습 패러다임을 제안하고, 이를 통해 기계 학습 모델의 훈련 및 최적화 과정을 다층적이고 중첩된 최적화 문제로 표현합니다. NL의 핵심 아이디어는 각 최적화 문제가 고유한 "컨텍스트 흐름"을 가지고 있으며, 이를 통해 모델이 데이터를 학습하는 방식을 설명합니다.

#### 예시: MLP 훈련

1. **훈련 데이터**: 
   - 입력 데이터 \(D_{\text{train}} = \{x_1, x_2, \ldots, x_n\}\)는 각 \(x_i\)가 \(d\) 차원의 벡터로 구성됩니다.
   - 예를 들어, \(x_1 = [0.1, 0.2, 0.3]\), \(x_2 = [0.4, 0.5, 0.6]\)와 같은 형태입니다.

2. **목표**: 
   - 모델의 목표는 주어진 입력 \(x_i\)에 대해 출력 \(y_i\)를 예측하는 것입니다. 여기서 \(y_i\)는 실제 레이블입니다.

3. **훈련 과정**:
   - 모델의 파라미터 \(W\)를 최적화하기 위해 손실 함수 \(L(W; D_{\text{train}})\)를 최소화합니다.
   - 경량화된 MLP(다층 퍼셉트론)를 사용하여 훈련을 진행합니다. 
   - 경량화된 MLP의 업데이트 규칙은 다음과 같습니다:
     \[
     W^{t+1} = W^t - \eta_t \nabla_W L(W^t; x_{t+1})
     \]
   - 여기서 \(\eta_t\)는 학습률이며, \(\nabla_W L(W^t; x_{t+1})\)는 손실 함수의 기울기입니다.

4. **테스트 데이터**:
   - 테스트 데이터 \(D_{\text{test}} = \{x_{test1}, x_{test2}, \ldots, x_{testm}\}\)는 모델이 훈련되지 않은 새로운 데이터입니다.
   - 예를 들어, \(x_{test1} = [0.7, 0.8, 0.9]\)와 같은 형태입니다.

5. **출력**:
   - 모델은 테스트 데이터에 대해 예측된 출력 \(y_{test}\)를 생성합니다. 예를 들어, \(y_{test1} = W \cdot x_{test1}\)와 같이 계산됩니다.

이러한 방식으로 NL은 모델이 데이터를 처리하고 학습하는 방식을 다층적이고 중첩된 최적화 문제로 표현하여, 더 나은 성능을 이끌어낼 수 있는 가능성을 제시합니다.

---



This paper proposes a new learning paradigm called Nested Learning (NL), which represents the training and optimization processes of machine learning models as a set of nested, multi-level optimization problems. The core idea of NL is that each optimization problem has its own unique "context flow," which explains how the model learns from data.

#### Example: MLP Training

1. **Training Data**: 
   - The input data \(D_{\text{train}} = \{x_1, x_2, \ldots, x_n\}\) consists of vectors \(x_i\) in \(d\) dimensions.
   - For example, \(x_1 = [0.1, 0.2, 0.3]\), \(x_2 = [0.4, 0.5, 0.6]\), etc.

2. **Objective**: 
   - The goal of the model is to predict the output \(y_i\) for the given input \(x_i\). Here, \(y_i\) is the actual label.

3. **Training Process**:
   - To optimize the model's parameters \(W\), the loss function \(L(W; D_{\text{train}})\) is minimized.
   - A lightweight MLP (Multi-Layer Perceptron) is used for training.
   - The update rule for the lightweight MLP is as follows:
     \[
     W^{t+1} = W^t - \eta_t \nabla_W L(W^t; x_{t+1})
     \]
   - Here, \(\eta_t\) is the learning rate, and \(\nabla_W L(W^t; x_{t+1})\) is the gradient of the loss function.

4. **Test Data**:
   - The test data \(D_{\text{test}} = \{x_{\text{test1}}, x_{\text{test2}}, \ldots, x_{\text{testm}}\}\) consists of new data that the model has not been trained on.
   - For example, \(x_{\text{test1}} = [0.7, 0.8, 0.9]\).

5. **Output**:
   - The model generates predicted outputs \(y_{\text{test}}\) for the test data. For instance, \(y_{\text{test1}} = W \cdot x_{\text{test1}}\).

In this way, NL presents the model's data processing and learning as a set of nested and multi-level optimization problems, suggesting the potential for improved performance.

<br/>
# 요약


이 논문에서는 Nested Learning(NL)이라는 새로운 학습 패러다임을 제안하여, 기계 학습 모델을 다층적이고 병렬적인 최적화 문제의 집합으로 표현합니다. NL을 기반으로 한 Deep Optimizers, Self-Modifying Titans, Continuum Memory System을 통해 더 표현력이 뛰어난 학습 알고리즘을 설계하고, 이를 통해 언어 모델링, 지속적 학습 및 긴 맥락 추론 작업에서 유망한 결과를 보여줍니다. 이 연구는 기존의 딥러닝 아키텍처의 한계를 극복하고, 더 나은 성능을 위한 새로운 방향을 제시합니다.

---

This paper proposes a new learning paradigm called Nested Learning (NL), which represents machine learning models as a set of multi-level and parallel optimization problems. By leveraging NL, the authors introduce Deep Optimizers, Self-Modifying Titans, and a Continuum Memory System, resulting in more expressive learning algorithms that demonstrate promising results in language modeling, continual learning, and long-context reasoning tasks. This research aims to overcome the limitations of existing deep learning architectures and suggests a new direction for improved performance.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 뇌의 구조와 업데이트 메커니즘을 설명하며, Nested Learning(NL)의 개념을 시각적으로 나타냅니다. 이 그림은 NL이 뇌의 다중 시간 척도 업데이트를 가능하게 하며, 기존의 Transformer 아키텍처가 사실상 다양한 주파수 업데이트를 가진 선형 계층으로 구성되어 있음을 보여줍니다.
   - **Figure 2**: NL 패러다임을 통해 기계 학습 모델과 훈련 절차를 중첩된 최적화 문제의 집합으로 표현합니다. 이는 기존의 딥러닝 관점에서 벗어나 모델의 내부 그래디언트 흐름을 투명하게 나타냅니다.

2. **테이블**
   - **Table 1**: HOPE 모델과 여러 기준 모델의 성능을 비교합니다. HOPE는 언어 모델링 및 상식 추론 작업에서 다른 모델들보다 우수한 성능을 보이며, 특히 낮은 perplexity와 높은 정확도를 기록했습니다. 이는 HOPE의 구조가 더 깊은 메모리 모듈과 동적으로 변화하는 키, 값, 쿼리 프로젝션을 기반으로 하고 있음을 시사합니다.

3. **어펜딕스**
   - 어펜딕스에서는 NL의 이론적 배경, 추가 실험 결과, 그리고 다양한 최적화 알고리즘에 대한 논의가 포함되어 있습니다. 특히, NL이 기존의 딥러닝 방법론을 어떻게 개선할 수 있는지에 대한 구체적인 사례와 실험 결과가 제시되어 있습니다. 이 부분은 연구의 깊이를 더하고, NL의 적용 가능성을 넓히는 데 기여합니다.



1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the structure of the brain and its update mechanisms, visually representing the concept of Nested Learning (NL). This figure shows that NL enables multi-time-scale updates in the brain and indicates that existing architectures like Transformers are essentially linear layers with different frequency updates.
   - **Figure 2**: Represents the NL paradigm by depicting a machine learning model and its training procedure as a set of nested optimization problems. This shifts away from traditional deep learning perspectives, transparently illustrating the inner gradient flows of the model.

2. **Tables**
   - **Table 1**: Compares the performance of the HOPE model against various baseline models. HOPE demonstrates superior performance in language modeling and commonsense reasoning tasks, particularly achieving lower perplexity and higher accuracy. This suggests that HOPE's architecture, which incorporates deeper memory modules and dynamically changing key, value, and query projections, is more effective.

3. **Appendices**
   - The appendices contain theoretical backgrounds on NL, additional experimental results, and discussions on various optimization algorithms. Notably, they provide specific examples of how NL can enhance existing deep learning methodologies. This section adds depth to the research and broadens the applicability of NL in machine learning contexts.

<br/>
# refer format:


### BibTeX 



```bibtex
@inproceedings{Behrouz2025,
  author = {Ali Behrouz and Meisam Razaviyayn and Peiling Zhong and Vahab Mirrokni},
  title = {Nested Learning: The Illusion of Deep Learning Architectures},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year = {2025},
  publisher = {NeurIPS},
  url = {https://arxiv.org/abs/2501.00663}
}
```

### 시카고 스타일

Behrouz, Ali, Meisam Razaviyayn, Peiling Zhong, and Vahab Mirrokni. 2025. "Nested Learning: The Illusion of Deep Learning Architectures." In *Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)*. NeurIPS. https://arxiv.org/abs/2501.00663.
