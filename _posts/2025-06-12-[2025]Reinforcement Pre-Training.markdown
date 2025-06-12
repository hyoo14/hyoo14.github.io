---
layout: post
title:  "[2025]Reinforcement Pre-Training"  
date:   2025-06-12 02:05:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


기존에는 사전학습 이후에 강화학습을 도입했었는데 본 논문은 사전학습 과정에서 강화학습을 도입, 즉 일종의 추론력을 학습초기부터 도입하는 셈(암기보다 앞서서)   



짧은 요약(Abstract) :    




이 논문은 대규모 언어 모델(LLMs)과 강화 학습(RL)을 위한 새로운 스케일링 패러다임인 \*\*Reinforcement Pre-Training (RPT)\*\*를 제안합니다. 기존의 다음 토큰 예측(next-token prediction)을 단순한 예측 문제가 아니라 **추론(reasoning) 과제로 재정의**하고, 모델이 실제 정답 토큰과 일치하는지 여부에 따라 \*\*검증 가능한 보상(verifiable reward)\*\*을 제공하는 강화 학습 방법을 사용합니다.

이 접근 방식의 핵심은 방대한 일반 텍스트 데이터를 도메인 특화 주석 없이도 **범용적인 강화 학습 데이터로 활용**할 수 있다는 점입니다. 이를 통해 다음 토큰 추론 능력이 향상되고, 후속 강화 학습도 더 강력한 기반 위에서 수행할 수 있습니다. 실험 결과, **계산량이 증가할수록 성능이 꾸준히 향상**되는 스케일링 특성을 보였으며, 기존 방법보다 더 나은 정확도와 범용성을 입증했습니다.

---


> In this work, we introduce **Reinforcement Pre-Training (RPT)** as a new scaling paradigm for large language models and reinforcement learning (RL). Specifically, we reframe next-token prediction as a reasoning task trained using RL, where it receives **verifiable rewards** for correctly predicting the next token for a given context. RPT offers a scalable method to leverage vast amounts of text data for general-purpose RL, rather than relying on domain-specific annotated answers. By incentivizing the capability of next-token reasoning, RPT **significantly improves the language modeling accuracy** of predicting the next tokens. Moreover, RPT provides a strong pre-trained foundation for further reinforcement fine-tuning. The scaling curves show that increased training compute consistently improves the next-token prediction accuracy. The results position RPT as an effective and promising scaling paradigm to advance language model pre-training.





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



#### 1. 주요 아이디어: 다음 토큰 예측을 "추론 과제"로 변환

기존 언어 모델은 단순히 다음 토큰을 예측하지만, RPT는 이를 **추론 과정으로 전환**합니다. 모델은 각 문맥에서 단순히 다음 단어를 예측하는 것이 아니라, "왜 이 단어가 적절한지"에 대한 **추론을 먼저 수행**한 뒤, 예측 결과를 생성합니다.

#### 2. 강화 학습 적용: 검증 가능한 보상

모델이 생성한 다음 토큰이 실제 정답과 일치하면 **보상 1**, 그렇지 않으면 **보상 0**을 부여합니다. 이 보상은 문서 자체에서 얻어진 \*\*정답 토큰(ground-truth token)\*\*을 기반으로 하며, 외부 어노테이션이 필요 없습니다.

수식으로는 다음과 같이 정의됩니다:

* 기본 언어 모델 목표 (NTP):

  $$
  J_{\text{NTP}}(\theta) = \sum_{t=1}^{T} \log P(x_t \mid x_0, x_1, ..., x_{t-1}; \theta)
  $$

* RPT의 보상 기반 목표:

  $$
  J_{\text{RPT}}(\theta) = \mathbb{E}_{x_{<t}, x_{\ge t} \sim D, \{o^i_t\} \sim \pi_\theta} [r^i_t]
  $$

  여기서 $r^i_t = 1$ if $y^i_t$ matches the ground-truth prefix; otherwise 0.

#### 3. 데이터셋: OmniMATH

* 4,428개의 **수학 문제**와 해설로 구성된 고난이도 데이터셋
* Deepseek-R1-Distill-Qwen-1.5B를 활용해 **엔트로피 필터링**을 수행하여 **어려운 토큰 중심으로 학습**

#### 4. 아키텍처 및 학습 설정

* 기본 모델: **Deepseek-R1-Distill-Qwen-14B**
* 학습 프레임워크: `verl` 및 `vllm` 사용
* 주요 하이퍼파라미터:

  * 학습률: 1e-6
  * 배치 사이즈: 256
  * 응답 샘플 수 $G = 8$
  * 온정책 강화학습 알고리즘: GRPO 사용
  * 보상 기준: **prefix matching reward**

---


#### 1. Core Idea: Next-token prediction as reasoning

RPT transforms the conventional next-token prediction task into a **reasoning task**. Given a prefix $x_{<t}$, the model is trained to first generate a **chain-of-thought reasoning sequence** $c_t$, then output a predicted next token $y_t$. The full output is $o_t = (c_t, y_t)$.

#### 2. Reinforcement Learning Objective

The model is rewarded **only if** its predicted token exactly matches a prefix of the ground-truth continuation (with correct token boundaries).

* Standard Language Modeling Objective:

  $$
  J_{\text{NTP}}(\theta) = \sum_{t=1}^{T} \log P(x_t \mid x_0, ..., x_{t-1}; \theta)
  $$

* Reinforcement Pre-Training Objective:

  $$
  J_{\text{RPT}}(\theta) = \mathbb{E}_{x_{<t}, x_{\ge t} \sim D, \{o^i_t\} \sim \pi_\theta} [r^i_t]
  $$

  where the reward is defined as:

  $$
  r^i_t =
  \begin{cases}
  1 & \text{if } y^i_t = x_{\ge t}[1:l] \text{ and } l \in L_{\text{gt}} \\
  0 & \text{otherwise}
  \end{cases}
  $$

#### 3. Dataset: OmniMATH

* Contains **4,428 competition-level mathematical problems and solutions**
* Token-level **entropy filtering** is applied using a proxy model (Qwen-1.5B) to focus on challenging predictions.

#### 4. Architecture and Training Configuration

* Base model: **Deepseek-R1-Distill-Qwen-14B**
* Frameworks: **`verl`** for RL training, **`vllm`** for inference
* Training setup:

  * Learning rate: $1 \times 10^{-6}$
  * Batch size: 256 questions
  * Number of trajectories per prompt: $G = 8$
  * Algorithm: **GRPO**
  * Reward: **Prefix matching reward**





   
 
<br/>
# Results  




### 1. 경쟁 아키텍처

* **Baseline 1:** R1-Distill-Qwen-14B
* **Baseline 2:** R1-Distill-Qwen-32B
* **Proposed:** RPT-14B (기존 14B 모델에 RPT 방식 적용)

---

### 2. 테스트 데이터 및 테스크

1. **언어 모델링 성능 평가**

   * 데이터: OmniMATH (수학 문제/해설)
   * 테스트 방식: 엔트로피 기반으로 난이도를 구분 (Easy / Medium / Hard)
   * 평가: 다음 토큰 예측 정확도 (next-token prediction accuracy)

2. **강화학습 파인튜닝 성능 평가**

   * 데이터: Skywork-OR1 (검증 가능한 정답 포함)
   * 평가: 보상 기반 정확도 향상률

3. **제로샷 성능 평가 (Zero-Shot Generalization)**

   * Benchmarks:

     * **SuperGPQA** (285개 대학원 수준 추론 과제)
     * **MMLU-Pro** (범용 다중분야 평가셋)
   * 평가 방식: Reasoning 모드로 multiple-choice 정답 선택

---

### 3. 주요 메트릭 및 결과

####  언어 모델링 정확도 (Next-token prediction accuracy)

| 난이도 | Qwen2.5-14B | R1-14B | RPT-14B   |
| --- | ----------- | ------ | --------- |
| 쉬움  | 41.90       | 41.60  | **45.11** |
| 중간  | 30.03       | 29.46  | **33.56** |
| 어려움 | 20.65       | 20.43  | **23.75** |

→ RPT는 모든 난이도에서 기존 모델보다 더 높은 정확도 기록.
→ 14B 모델임에도 불구하고, 32B 모델 수준까지 도달함.

---

####  강화 학습 파인튜닝 (Reinforcement Fine-tuning)

| 모델                  | 강화학습 전   | 후        |
| ------------------- | -------- | -------- |
| R1-Distill-Qwen-14B | 51.2     | 52.7     |
| + NTP 추가 훈련         | 10.7     | 13.0     |
| RPT-14B             | **56.3** | **58.3** |

→ RPT 모델은 훈련 전부터 더 높은 성능을 보이며, 후속 강화학습에도 더 잘 적응함.

---

####  제로샷 성능 (Zero-shot on end tasks)

| 모델                  | SuperGPQA | MMLU-Pro |
| ------------------- | --------- | -------- |
| R1-14B (NTP)        | 32.0      | 48.4     |
| R1-32B (NTP)        | 37.2      | 56.5     |
| R1-14B (reasoning)  | 36.1      | 68.9     |
| RPT-14B (reasoning) | **39.0**  | **71.1** |

→ RPT-14B는 32B 모델보다도 우수한 성능을 보이며, 특히 Reasoning 설정에서 강력함.

---


### 1. Competing Architectures

* **Baselines:**

  * R1-Distill-Qwen-14B
  * R1-Distill-Qwen-32B
* **Proposed model:**

  * RPT-14B (14B with Reinforcement Pre-Training)

---

### 2. Evaluation Tasks & Datasets

1. **Language Modeling on OmniMATH**

   * Evaluation on next-token prediction accuracy
   * Data filtered into Easy / Medium / Hard based on entropy

2. **Reinforcement Fine-Tuning**

   * Dataset: Skywork-OR1
   * Evaluation: reward-based improvement before/after RLVR

3. **Zero-shot End Task Evaluation**

   * Benchmarks:

     * **SuperGPQA**: graduate-level reasoning across 285 subjects
     * **MMLU-Pro**: general multi-domain benchmark
   * Evaluation: multiple-choice accuracy under reasoning mode

---

### 3. Key Metrics & Results

####  Next-token Prediction Accuracy

| Difficulty | Qwen2.5-14B | R1-14B | RPT-14B   |
| ---------- | ----------- | ------ | --------- |
| Easy       | 41.90       | 41.60  | **45.11** |
| Medium     | 30.03       | 29.46  | **33.56** |
| Hard       | 20.65       | 20.43  | **23.75** |

→ RPT achieves consistently higher accuracy, rivaling the 32B model despite being 14B.

---

####  Reinforcement Fine-tuning (Skywork-OR1)

| Model                    | Before RL | After RL |
| ------------------------ | --------- | -------- |
| R1-Distill-Qwen-14B      | 51.2      | 52.7     |
| + Continual NTP training | 10.7      | 13.0     |
| RPT-14B                  | **56.3**  | **58.3** |

→ RPT provides a stronger initialization and better post-RL adaptation.

---

####  Zero-Shot Performance

| Model                   | SuperGPQA | MMLU-Pro |
| ----------------------- | --------- | -------- |
| R1-14B (standard)       | 32.0      | 48.4     |
| R1-32B (standard)       | 37.2      | 56.5     |
| R1-14B (reasoning mode) | 36.1      | 68.9     |
| **RPT-14B**             | **39.0**  | **71.1** |

→ RPT-14B outperforms both the 14B and 32B baselines in reasoning settings.





<br/>
# 예제  



###  예시 1: 물리 개념 문장의 다음 토큰 예측

**문맥 (Context)**

> *"Electric force grows with charge..."*

**모델 추론 과정 요약**

* 모델은 다음에 올 단어가 무엇인지 고민하면서, 다음과 같은 단계를 거칩니다:

  1. 물리 법칙(Coulomb 법칙)에 따라 “거리” 관련 내용이 나올 것으로 예상
  2. “decreases with distance” 또는 “squared” 등이 가능성 있는 후보로 떠오름
  3. “size”라는 단어가 가능성이 높다고 판단함

**최종 예측 결과**

> \boxed{ size }

→ 이 예시는 단순한 언어 모델링이 아니라, 모델이 **물리적 의미와 문맥 흐름을 논리적으로 추론**하여 예측한다는 점을 보여줍니다.

---

###  예시 2: 수학 문제 풀이 추론

**문맥**

> *"Using the integral test, for a set..."*

**모델 추론**

* 수렴성 증명을 요구하는 문장으로 판단
* 통상적으로 integral test는 “when $p > 1$”일 때 수렴함
* 다음 단어로는 수식 기호 `$`가 올 가능성이 높다고 판단

**최종 예측**

> \boxed{\$}

→ 수학적 문법과 논리적 전개 흐름을 바탕으로 **구체적 수식 형식까지 고려한 추론**을 수행함

---

###  예시 3: 도형 문제의 다음 토큰 예측

**문맥**

> *"Scale the small circle by a factor of..."*

**모델 추론**

* 원의 중심, 반지름, 비율 등을 바탕으로 다음 숫자가 나올 가능성이 높다고 판단
* 실험적으로 30/13 같은 비율 또는 공백(space) 뒤에 숫자 올 것으로 예측

**최종 예측**

> \boxed{ }

→ 정답 토큰이 실제로 공백(스페이스)일 수도 있다는 점을 **정확히 추론**한 사례

---


###  Example 1: Physics sentence continuation

**Context**

> *"Electric force grows with charge..."*

**Model Thought Process**

* Recognizes it's a physics sentence related to Coulomb’s law
* Anticipates the next concept to be about distance, e.g., "decreases with distance"
* Considers multiple candidates like “squared” or “size”
* Selects "size" as most likely next token

**Prediction**

> \boxed{ size }

→ Demonstrates the model’s **scientific reasoning** rather than shallow pattern matching.

---

###  Example 2: Mathematical proof continuation

**Context**

> *"Using the integral test, for a set..."*

**Model Thought Process**

* Infers the sentence relates to proving convergence using integral test
* Standard answer format includes a condition like “when $p > 1$”
* Anticipates a math symbol (e.g., `$`) to begin LaTeX-style math

**Prediction**

> \boxed{\$}

→ Shows the model’s ability to **contextually reason about mathematical notation and structure**.

---

###  Example 3: Geometry reasoning

**Context**

> *"Scale the small circle by a factor of..."*

**Model Thought Process**

* Calculates or estimates a geometric ratio (e.g., 30/13)
* Considers that the original input may include a space or number
* Chooses space as the next most probable token

**Prediction**

> \boxed{ }

→ Captures the **low-level formatting behavior** of texts, suggesting deep alignment with token-level semantics.




<br/>  
# 요약   




Reinforcement Pre-Training (RPT)은 다음 토큰 예측을 추론 과제로 재정의하고, 올바른 예측에 대해 강화 학습 보상을 제공하는 새로운 사전학습 방법이다.
이 방식은 언어 모델의 정확도를 향상시키고, 기존 14B 모델이 32B 모델 수준의 성능을 달성하도록 한다.
예를 들어, 물리 법칙이나 수학 문제에서 모델이 추론을 통해 정확한 다음 토큰(예: `size`, `$`)을 예측함으로써 추론 기반 언어 이해 능력을 보여준다.

---


Reinforcement Pre-Training (RPT) is a novel pretraining paradigm that reframes next-token prediction as a reasoning task, guided by reinforcement learning with correctness-based rewards.
This approach significantly boosts language modeling accuracy, enabling a 14B model to match or surpass the performance of larger 32B models.
For instance, in physics and math contexts, the model correctly predicts tokens like `size` or `$` by reasoning through the semantic and structural cues.




<br/>  
# 기타  



###  Figure 1: RPT 개념도

* **내용**: 기존의 next-token prediction을 강화학습 기반의 추론 태스크로 바꾸는 구조 시각화
* **인사이트**: RPT는 RL을 웹 텍스트 전체로 확장시켜 사전학습 자체를 reasoning 기반으로 만든다는 점에서 기존 scaling paradigm과 구별됨

---

###  Figure 2: 기존 예측 vs RPT 방식

* **내용**: 일반적인 next-token 예측과 RPT의 추론 과정을 비교
* **인사이트**: RPT는 단순 예측이 아닌 생각의 흐름("Let's think...")을 먼저 생성한 뒤 예측함으로써 더 깊은 이해를 도모함

---

###  Table 1: 예측 정확도 비교 (Easy/Medium/Hard)

* **내용**: 난이도별로 RPT가 기존 14B 및 32B 모델보다 더 높은 정확도를 달성
* **인사이트**: reasoning 기반 pretraining이 특히 어려운 입력에서 효과적임

---

###  Figure 4 & 5: 스케일링 곡선

* **내용**: 학습 연산량이 증가함에 따라 예측 정확도가 선형적으로 향상됨 (R² > 0.98)
* **인사이트**: RPT는 계산량 증가에 따라 안정적으로 성능이 향상되는 스케일링 가능성이 높은 방법임

---

###  Table 2: 강화학습 파인튜닝 성능

* **내용**: RPT 기반 모델은 기존 NTP보다 높은 성능 및 더 빠른 적응을 보임
* **인사이트**: RPT는 강화학습 기반 후속 훈련에도 강한 초기화를 제공함

---

###  Table 3: 제로샷 성능 (SuperGPQA, MMLU-Pro)

* **내용**: RPT-14B는 reasoning 모드에서 32B 모델을 능가
* **인사이트**: reasoning 기반 학습이 실제 응용에서도 성능 우위로 이어짐

---

###  Figure 6: reasoning 패턴 분석

* **내용**: RPT는 hypothesis, deduction 패턴을 더 자주 사용하며, 기존 모델보다 더 다양하고 깊은 사고 흐름을 보여줌
* **인사이트**: 단순한 문제 해결이 아닌 **다양한 추론 방식**이 학습됨

---

###  Table 4 & 11: 구체적인 reasoning 예시

* **내용**: 모델이 문제를 풀거나 다음 토큰을 추론하기 위해 여러 가능성을 생각하고 비교하는 사고 흐름을 보여줌
* **인사이트**: 모델이 단순 통계적 연산이 아닌 **의식적인 추론**을 수행하고 있음

---

###  Appendix A\~F:

* A: 다양한 보상 설계 실험 → prefix matching이 가장 효과적
* B: 하이퍼파라미터 표 → 학습 환경 재현 가능
* C: 분야별 zero-shot 상세 결과 → MMLU-Pro, SuperGPQA 성능 상세 비교
* D: 다양한 프롬프트 템플릿 → 초기 추론 성능에 영향 있음
* E: reasoning keyword 목록 → 패턴 분석 기준 제공
* F: 추가 reasoning 사례 → 사고 흐름이 실제로 학습됨을 보여줌

---


###  Figure 1: Concept of RPT

* **Content**: Visualizes how RPT reformulates next-token prediction into a reinforcement learning-driven reasoning task.
* **Insight**: RPT enables RL to scale to web-text corpora, distinguishing itself from conventional self-supervised training.

---

###  Figure 2: Standard vs Reasoning-Based Prediction

* **Content**: Compares traditional next-token prediction with RPT’s “think-then-predict” flow.
* **Insight**: RPT emphasizes thoughtful reasoning, leading to better understanding and generalization.

---

###  Table 1: Accuracy Across Token Difficulty

* **Content**: RPT-14B outperforms R1-14B and even matches R1-32B across Easy, Medium, and Hard token splits.
* **Insight**: RPT excels especially on challenging inputs, suggesting its reasoning structure helps tackle complexity.

---

###  Figures 4 & 5: Scaling Curves

* **Content**: Shows accuracy consistently improves with more training compute (R² ≈ 0.99).
* **Insight**: RPT follows a clean scaling law, promising for larger-scale pretraining.

---

###  Table 2: RL Fine-Tuning

* **Content**: RPT-14B has stronger pre-RL accuracy and adapts better post-RL than standard NTP-trained models.
* **Insight**: RPT provides better initialization and faster improvement in reinforcement fine-tuning.

---

###  Table 3: Zero-Shot Evaluation (SuperGPQA, MMLU-Pro)

* **Content**: RPT-14B in reasoning mode outperforms 32B baselines on both benchmarks.
* **Insight**: Reasoning-based training translates to superior generalization.

---

###  Figure 6: Reasoning Pattern Statistics

* **Content**: RPT uses hypothesis and deduction patterns more than baseline models.
* **Insight**: Indicates learning of diverse and deeper cognitive patterns.

---

###  Table 4 & 11: Qualitative Examples

* **Content**: Cases where the model evaluates multiple hypotheses before selecting the correct next token.
* **Insight**: Demonstrates deliberate, step-by-step reasoning rather than memorization.

---

###  Appendices A–F:

* **A**: Alternative reward strategies tested — prefix matching performs best.
* **B**: Hyperparameters — reproducibility details.
* **C**: Detailed zero-shot results per domain.
* **D**: Prompt template variants — affect initial reasoning performance.
* **E**: Reasoning keywords used in pattern analysis.
* **F**: Additional case studies — confirms reasoning is genuinely learned.




<br/>
# refer format:     




@article{dong2025rpt,
  title={Reinforcement Pre-Training},
  author={Dong, Qingxiu and Dong, Li and Tang, Yao and Ye, Tianzhu and Sun, Yutao and Sui, Zhifang and Wei, Furu},
  journal={arXiv preprint arXiv:2506.08007},
  year={2025},
  url={https://arxiv.org/abs/2506.08007}
}
   



Dong, Qingxiu, Li Dong, Yao Tang, Tianzhu Ye, Yutao Sun, Zhifang Sui, and Furu Wei. “Reinforcement Pre-Training.” arXiv preprint arXiv:2506.08007 (2025). https://arxiv.org/abs/2506.08007.   



