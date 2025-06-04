---
layout: post
title:  "[2022]Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"  
date:   2022-06-03 15:11:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    


이 논문은 **Few-shot In-Context Learning (ICL)**과 파라미터 효율적인 파인튜닝(PEFT) 기법을 비교합니다. ICL은 사전학습된 언어 모델에 소수의 예시를 입력에 포함시켜 새로운 작업을 수행하게 하지만, 매번 예시를 처리해야 하므로 계산량과 메모리 사용량이 매우 큽니다. 반면, PEFT는 모델의 극히 일부 파라미터만 학습시켜 새로운 작업을 수행할 수 있게 하며, 훨씬 저렴하고 효율적입니다.

논문에서는 새로운 PEFT 방법인 (IA)³를 제안하는데, 이는 학습된 벡터를 통해 모델의 활성값을 조절합니다. (IA)³는 적은 수의 파라미터만으로도 뛰어난 성능을 보이며, 새로운 작업에 대한 정확도가 ICL보다 높고 계산 비용도 적습니다.

또한, 저자들은 T-Few라는 레시피를 제안합니다. 이 방법은 T0 모델 기반으로 어떤 작업에도 특별한 조정 없이 적용할 수 있으며, 실제 RAFT 벤치마크에서 인간 성능을 뛰어넘는 결과를 최초로 달성했습니다. 실험에 사용된 코드는 공개되어 있습니다.


Few-shot in-context learning (ICL) enables pre-trained language models to perform a previously-unseen task without any gradient-based training by feeding a small number of training examples as part of the input. ICL incurs substantial computational, memory, and storage costs because it involves processing all of the training examples every time a prediction is made. Parameter-efficient fine-tuning (PEFT) (e.g. adapter modules, prompt tuning, sparse update methods, etc.) offers an alternative paradigm where a small set of parameters are trained to enable a model to perform the new task. In this paper, we rigorously compare few-shot ICL and PEFT and demonstrate that the latter offers better accuracy as well as dramatically lower computational costs. Along the way, we introduce a new PEFT method called (IA)³ that scales activations by learned vectors, attaining stronger performance while only introducing a relatively tiny amount of new parameters. We also propose a simple recipe based on the T0 model called T-Few that can be applied to new tasks without task-specific tuning or modifications. We validate the effectiveness of T-Few on completely unseen tasks by applying it to the RAFT benchmark, attaining super-human performance for the first time and outperforming the state-of-the-art by 6% absolute. All of the code used in our experiments is publicly available.


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


---

### 1. 모델 및 데이터셋

#### 🇰🇷 설명:

T-Few는 T5 기반의 사전학습 모델인 T0-3B를 사용하며, 학습에 사용되지 않았던 태스크로 일반화 성능을 평가합니다.
RAFT와 같은 실제 태스크도 포함되어 있어 현실적인 few-shot 상황을 반영합니다.

#### 🇺🇸 Explanation:

T-Few builds on the T5-based model T0-3B. It evaluates generalization performance on unseen tasks such as ANLI, COPA, and real-world datasets like RAFT.

---

### 2. 손실 함수 (Loss Functions)

#### 🇰🇷 설명:

기존 언어 모델 손실 외에 두 가지 추가 손실을 도입합니다:
(1) Unlikelihood Loss는 틀린 정답에 높은 확률이 할당되지 않도록 억제하며,
(2) Length-Normalized Loss는 짧은 답변에 과도하게 유리한 기존 언어 모델의 경향을 보정합니다.

#### 🇺🇸 Explanation:

Two new loss terms are added to the standard LM loss:
(1) **Unlikelihood Loss** discourages high probabilities for incorrect candidates,
(2) **Length-Normalized Loss** corrects the model’s bias toward shorter answers.

---

#### 수식 (Formulas)

* 기본 LM 손실 (Language Modeling Loss):

```latex
$$
\mathcal{L}_{\text{LM}} = -\frac{1}{T} \sum_{t} \log p(y_t \mid x, y_{<t})
$$
```

* 잘못된 후보 억제 (Unlikelihood Loss):

```latex
$$
\mathcal{L}_{\text{UL}} = - \sum_{n=1}^{N} \sum_{t=1}^{T^{(n)}} \log \left( 1 - p\left(\hat{y}^{(n)}_t \mid x, \hat{y}^{(n)}_{<t} \right) \right)
$$
```

* 정답 길이 보정 (Length-Normalized Loss):

```latex
$$
\beta(x, y) = \frac{1}{T} \sum_{t} \log p(y_t \mid x, y_{<t})
$$

$$
\mathcal{L}_{\text{LN}} = - \log \left( \frac{\exp(\beta(x, y))}{\exp(\beta(x, y)) + \sum_{n=1}^{N} \exp(\beta(x, \hat{y}^{(n)}))} \right)
$$
```

* 최종 손실 결합 (Final Loss):

```latex
$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{UL}} + \mathcal{L}_{\text{LN}}
$$
```

---

### 3. (IA)³: 새로운 PEFT 기법

#### 🇰🇷 설명:

(IA)³는 파라미터 효율적 튜닝 방법으로, 각 레이어의 attention과 feed-forward 중간값에 대해 작은 스케일 벡터를 곱하여 조절합니다.
이 방식은 기존 파라미터를 그대로 유지하면서도 각 태스크에 맞는 조절이 가능합니다.

#### 🇺🇸 Explanation:

(IA)³ is a Parameter-Efficient Fine-Tuning method. It applies small learned vectors to rescale intermediate values in self-attention and feed-forward layers.
This allows task-specific adaptation with minimal new parameters.

---

#### 수식 (Formulas)

* Self-Attention 수정:

```latex
$$
\text{Attention} = \text{softmax}\left( \frac{Q (l_k \odot K^\top)}{\sqrt{d_k}} \right) (l_v \odot V)
$$
```

* Feed-forward 수정:

```latex
$$
(l_{ff} \odot \gamma(W_1 x)) W_2
$$
```

* 설명: \$\odot\$는 element-wise 곱, \$l\_k\$, \$l\_v\$, \$l\_{ff}\$는 각 위치에 대한 학습 가능한 벡터입니다.

---

### 4. (IA)³ 사전학습 전략

#### 🇰🇷 설명:

(IA)³ 벡터는 T0가 학습된 멀티태스크 mixture를 사용해 미리 사전학습됩니다.
이후에 few-shot fine-tuning 시 바로 사용할 수 있어 성능이 빠르게 향상됩니다.

#### 🇺🇸 Explanation:

The (IA)³ vectors are pretrained on the same multi-task mixture used to train T0.
This enables efficient reuse during fine-tuning and boosts few-shot performance.

---

### 5. 전체 T-Few 레시피 요약

#### 🇰🇷 설명:

T-Few는 T0 백본 모델 위에 (IA)³를 추가하고, 새로운 손실 함수를 도입해 적은 샘플로도 빠르게 적응할 수 있게 설계되었습니다.

#### 🇺🇸 Explanation:

T-Few adds (IA)³ and new losses to the T0 backbone, enabling strong few-shot adaptation with minimal data and compute.

---

####  학습 설정 (Training Setup)

| 항목                  | 설정값                  |
| ------------------- | -------------------- |
| 학습 스텝 (Steps)       | 1,000                |
| 배치 크기 (Batch Size)  | 8                    |
| 옵티마이저 (Optimizer)   | Adafactor            |
| 학습률 (Learning Rate) | \$3 \times 10^{-3}\$ |

---





##  T-Few Method Summary (with Explanations – Markdown + LaTeX)

---

### 1. Model and Dataset

**Model**: T0-3B (based on T5, fine-tuned on multi-task prompted data)
**Evaluation Datasets**: 9 tasks held out from T0 training (e.g., ANLI, COPA, WiC) and the RAFT benchmark

** Explanation**:
T-Few leverages T0-3B, a large-scale model built on T5, and evaluates its ability to generalize to unseen tasks. RAFT tasks are also included, making the evaluation reflect realistic few-shot scenarios.

---

### 2. Loss Functions

#### Components:

* **Language Modeling Loss**: Standard token-level loss used during autoregressive generation
* **Unlikelihood Loss**: Penalizes high probabilities assigned to incorrect candidates
* **Length-Normalized Loss**: Reduces bias toward short answers, which usually receive higher log-probs

---

####  Formulas

* **Language Modeling Loss**:

```latex
$$
\mathcal{L}_{\text{LM}} = -\frac{1}{T} \sum_{t} \log p(y_t \mid x, y_{<t})
$$
```

* **Unlikelihood Loss**:

```latex
$$
\mathcal{L}_{\text{UL}} = - \sum_{n=1}^{N} \sum_{t=1}^{T^{(n)}} \log \left( 1 - p\left(\hat{y}^{(n)}_t \mid x, \hat{y}^{(n)}_{<t} \right) \right)
$$
```

* **Length-Normalized Loss**:

```latex
$$
\beta(x, y) = \frac{1}{T} \sum_{t} \log p(y_t \mid x, y_{<t})
$$

$$
\mathcal{L}_{\text{LN}} = - \log \left( \frac{\exp(\beta(x, y))}{\exp(\beta(x, y)) + \sum_{n=1}^{N} \exp(\beta(x, \hat{y}^{(n)}))} \right)
$$
```

* **Final Combined Loss**:

```latex
$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{UL}} + \mathcal{L}_{\text{LN}}
$$
```

** Explanation**:
The final loss improves model robustness and answer quality by:

* Learning from correct outputs (`LM`)
* Penalizing wrong answers (`UL`)
* Adjusting for output length (`LN`)

---

### 3. (IA)³: New PEFT (Parameter-Efficient Fine-Tuning) Method

**What it does**:
Applies learned scaling vectors to attention and FFN intermediate values with minimal additional parameters.

---

####  Formulas

* **Modified Self-Attention**:

```latex
$$
\text{Attention} = \text{softmax}\left( \frac{Q (l_k \odot K^\top)}{\sqrt{d_k}} \right) (l_v \odot V)
$$
```

* **Modified Feed-Forward Layer**:

```latex
$$
(l_{ff} \odot \gamma(W_1 x)) W_2
$$
```

** Explanation**:
Instead of updating all weights, (IA)³ learns small task-specific vectors—\$l\_k\$, \$l\_v\$, and \$l\_{ff}\$—which scale intermediate results.
This saves memory and improves few-shot performance.

---

### 4. (IA)³ Pretraining

**Pretraining Setup**:
(IA)³ vectors are pretrained on the same multitask dataset used for T0.

** Explanation**:
These pretrained adapters make it easier to fine-tune on new tasks with few samples, as the model already has a task-general adjustment mechanism.

---

### 5. T-Few Training Recipe

####  Settings:

| Parameter      | Value                |
| -------------- | -------------------- |
| Training Steps | 1,000                |
| Batch Size     | 8                    |
| Optimizer      | Adafactor            |
| Learning Rate  | \$3 \times 10^{-3}\$ |

---

####  Summary Explanation:

T-Few combines a strong pretrained base model (T0), parameter-efficient tuning (IA³), and a carefully designed loss function to adapt to new tasks quickly and efficiently—especially when data is scarce.





   
 
<br/>
# Results  



### 1.  **벤치마크 및 테스트 데이터셋**

* **T0 학습 제외 태스크** (9개): ANLI, CB, RTE, WiC, WSC, Winogrande, COPA, H-SWAG, Story Cloze
* **RAFT benchmark**: 실제 응용 기반 few-shot 태스크 11개, 각 50개 학습 샘플만 존재

---

### 2.  **비교 모델 (경쟁 아키텍처들)**

| 방법                          | 특징                                                   |
| --------------------------- | ---------------------------------------------------- |
| **T-Few**                   | 제안한 방법: T0 + (IA)³ + L<sub>UL</sub> + L<sub>LN</sub> |
| **T0**                      | zero-shot T5 기반                                      |
| **T5+LM**                   | few-shot in-context 학습                               |
| **GPT-3 6.7B / 13B / 175B** | OpenAI GPT-3, few-shot in-context 방식                 |
| **PET, SetFit**             | RAFT에 사용된 기존 SOTA 방법                                 |

---

### 3.  **테스트 태스크 유형**

* **분류 / 다지선다** (classification / multiple-choice)
* 평가 방식: **Rank Classification** (정답 후보들의 log-prob을 비교하여 최고 확률 선택)

---

### 4.  **성능 비교 결과**

#### (1) T0 테스트셋 (9개 held-out task)

| 모델               | Accuracy  | FLOPs (추론) | 파라미터 업데이트 |
| ---------------- | --------- | ---------- | --------- |
| **T-Few (ours)** | **72.4%** | **1.1e12** | 약 0.01%   |
| T0 (zero-shot)   | 66.9%     | 1.1e12     | 0         |
| T5+LM (ICL)      | 49.6%     | 4.5e13     | 0         |
| GPT-3 6.7B       | 57.2%     | 5.4e13     | 0         |
| GPT-3 13B        | 60.3%     | 1.0e14     | 0         |
| GPT-3 175B       | 66.6%     | 1.4e15     | 0         |

→ **T-Few가 GPT-3 175B보다 16배 작으면서도 정확도는 더 높고 계산량은 1,000배 적음**

---

#### (2) RAFT 벤치마크 (11개 real-world task)

| 모델               | Accuracy    |
| ---------------- | ----------- |
| **T-Few (ours)** | **75.8%**  |
| Human baseline   | 73.5%       |
| PET              | 69.6%       |
| SetFit           | 66.9%       |
| GPT-3 175B       | 62.7%       |

→ **T-Few가 최초로 인간 성능을 초과함**
→ 기존 최고 방법보다 **+6%** 정확도 개선

---

##  English Version: Results Summary

### 1.  **Benchmarks and Evaluation Datasets**

* **Held-out tasks from T0**: ANLI, CB, RTE, WiC, WSC, Winogrande, COPA, H-SWAG, Story Cloze
* **RAFT Benchmark**: 11 real-world few-shot tasks, 50 training examples per task

---

### 2.  **Baselines and Competitor Models**

| Method                  | Description                            |
| ----------------------- | -------------------------------------- |
| **T-Few (Ours)**        | T0 + (IA)³ + new loss terms            |
| **T0 (zero-shot)**      | Multitask fine-tuned T5, no tuning     |
| **T5+LM (ICL)**         | T5 used with in-context examples       |
| **GPT-3 6.7B/13B/175B** | OpenAI GPT-3 models using few-shot ICL |
| **PET / SetFit**        | Prior best methods on RAFT             |

---

### 3.  **Tasks and Metrics**

* Task Types: Classification, Multiple-choice
* **Evaluation Metric**: **Rank classification** (choose highest probability candidate)

---

### 4.  **Key Performance Results**

#### (1) Held-out Tasks from T0

| Model          | Accuracy  | Inference FLOPs | Updated Params |
| -------------- | --------- | --------------- | -------------- |
| **T-Few**      | **72.4%** | **1.1e12**      | \~0.01%        |
| T0 (zero-shot) | 66.9%     | 1.1e12          | 0              |
| T5+LM (ICL)    | 49.6%     | 4.5e13          | 0              |
| GPT-3 6.7B     | 57.2%     | 5.4e13          | 0              |
| GPT-3 13B      | 60.3%     | 1.0e14          | 0              |
| GPT-3 175B     | 66.6%     | 1.4e15          | 0              |

→ T-Few outperforms all, including GPT-3 175B (16× larger model) with 1,000× fewer FLOPs

---

#### (2) RAFT Real-world Tasks

| Model            | Accuracy    |
| ---------------- | ----------- |
| **T-Few (Ours)** | **75.8%**  |
| Human baseline   | 73.5%       |
| PET              | 69.6%       |
| SetFit           | 66.9%       |
| GPT-3 175B       | 62.7%       |

→ **T-Few is the first method to outperform human performance on RAFT**
→ Beats previous SOTA by **+6% absolute accuracy**






<br/>
# 예제  




### 1.  In-Context Learning 예시

ICL에서 사용하는 대표 예시는 다음과 같은 **철자 바꾸기(task: cycled letter unscrambling)** 문제입니다:

####  입력 예시 (4-shot ICL):

```
Please unscramble the letters into a word, and write that word:
asinoc = casino, 
yfrogg = froggy, 
plesim = simple, 
iggestb = biggest, 
astedro =
```

####  모델이 생성해야 할 정답:

```
roasted
```

→ 이런 식으로 few-shot 예시들을 함께 넣고 마지막에 정답을 유도하는 방식이 ICL입니다.

---

### 2.  (IA)³ 적용 위치 예시

제안된 PEFT 방법인 **(IA)³**에서는 Transformer 블록 내 **특정 위치의 활성값**에 대해 학습된 벡터를 곱합니다:

* Self-Attention: key (`K`)와 value (`V`) 벡터에 곱하기

  $$
  \text{softmax}\left( \frac{Q \cdot (l_k \odot K^T)}{\sqrt{d_k}} \right) \cdot (l_v \odot V)
  $$

* Feed-forward network:

  $$
  (l_{ff} \odot \gamma(W_1 x)) W_2
  $$

→ 즉, 이 학습된 벡터들 $l_k, l_v, l_{ff}$는 태스크별로 학습되며, 원래 모델 구조를 거의 변경하지 않고 미세 조정이 가능해집니다.

---

### 3.  RAFT 벤치마크 예시

**RAFT 벤치마크**는 실제 애플리케이션에서 유용한 11개의 텍스트 분류 태스크로 구성되어 있으며, 그 중 일부 예시는 다음과 같습니다:

* **Banking77**: 고객의 질문 분류 ("How do I reset my pin?" → "Card issues")
* **TweetEval-hate**: 트윗의 혐오 발언 감지 ("We don't need those people here." → hate)
* **CivilComments**: 댓글의 공격성 분류

→ 각 태스크는 단 **50개의 학습 샘플**만 제공되고, 정답이 없는 테스트셋으로 평가됩니다.

---

##  English Version: Concrete Examples

### 1.  In-Context Learning Example

A core example for ICL is the **cycled letter unscrambling task**:

####  Input (4-shot ICL prompt):

```
Please unscramble the letters into a word, and write that word:
asinoc = casino, 
yfrogg = froggy, 
plesim = simple, 
iggestb = biggest, 
astedro =
```

####  Expected output:

```
roasted
```

→ This exemplifies how few-shot ICL provides a few labeled examples in the input to condition the model.

---

### 2.  (IA)³ Injection Example

The proposed PEFT method **(IA)³** modifies the model by rescaling intermediate activations. The learnable vectors are injected into the following locations:

* Self-Attention:

  $$
  \text{softmax}\left( \frac{Q \cdot (l_k \odot K^T)}{\sqrt{d_k}} \right) \cdot (l_v \odot V)
  $$

* Feed-Forward Layer:

  $$
  (l_{ff} \odot \gamma(W_1 x)) W_2
  $$

→ These vectors ($l_k$, $l_v$, $l_{ff}$) are task-specific but small, and allow efficient fine-tuning without modifying the entire model.

---

### 3.  RAFT Dataset Examples

**RAFT** contains 11 real-world classification tasks, each with only 50 training examples. Examples include:

* **Banking77**: Classify customer intent
  e.g., *"How do I reset my pin?"* → `"Card issues"`

* **TweetEval-hate**: Hate speech detection on tweets
  e.g., *"We don't need those people here."* → `"hate"`

* **CivilComments**: Toxic comment classification

→ No validation set is given. The test labels are hidden, making this a realistic few-shot setting.




<br/>  
# 요약   




T-Few는 T0 모델에 새로운 PEFT 기법인 (IA)³를 적용하고, 추가 손실 함수(LUL, LLN)를 결합해 효율적으로 파인튜닝하는 방법입니다.
이 방법은 GPT-3보다 훨씬 적은 계산량으로도 더 높은 정확도를 달성했으며, RAFT 벤치마크에서는 인간 성능을 초과했습니다.
예시로, 4-shot 문제나 고객 질문 분류와 같은 실제 태스크에 소수 샘플만으로도 강력한 성능을 보여주었습니다.

---


T-Few fine-tunes the T0 model using a novel PEFT method called (IA)³, combined with unlikelihood and length-normalized losses for better efficiency.
It outperforms GPT-3 while requiring over 1,000× fewer FLOPs, and is the first method to surpass human performance on the RAFT benchmark.
In real-world tasks like few-shot word unscrambling or intent classification, T-Few shows strong results with only a handful of training examples.


<br/>  
# 기타  




###  Figure 2: 다양한 PEFT 방법 비교

* 내용: 여러 PEFT 방법의 정확도와 업데이트된 파라미터 비율 비교
* 결과: (IA)³는 전체 모델 파인튜닝보다 높은 정확도를 달성한 유일한 방법
* 인사이트: 업데이트 파라미터 수가 적으면서도 성능이 높아 효율성이 탁월함

---

###  Figure 3: 다양한 방법들의 FLOPs vs 정확도

* 내용: 추론 계산량(FLOPs) 대비 정확도 비교
* 결과: T-Few는 GPT-3 175B보다 정확도는 더 높고 계산량은 1,000배 적음
* 인사이트: 파인튜닝 비용이 매우 낮으면서도 성능은 최고 수준

---

###  Table 1: Held-out 태스크에 대한 정확도 및 비용 요약

| 모델         | 정확도       | 추론 FLOPs | 학습 FLOPs | 저장 공간 |
| ---------- | --------- | -------- | -------- | ----- |
| T-Few      | **72.4%** | 1.1e12   | 2.7e16   | 4.2MB |
| GPT-3 175B | 66.6%     | 1.4e15   | 0        | 16KB  |

* 인사이트: T-Few는 정확도와 자원 효율성 모두에서 최고 성능

---

###  Table 2: RAFT 벤치마크 결과 (상위 5개 방법)

* T-Few: 75.8% (최고 성능, 인간 성능 73.5% 초과)
* 인사이트: 실제 태스크에 대해 일반화 성능이 뛰어남

---

###  Appendix F: Ablation 실험

* Pre-training 제거 → 정확도 1.6% 감소
* LUL 및 LLN 제거 → 정확도 4.1% 감소
* 둘 다 제거 → 정확도 2.5% 감소
* 인사이트: 각 구성요소가 모두 성능 향상에 기여함

---


###  Figure 2: Comparison of PEFT Methods

* Content: Accuracy vs % of parameters updated
* Finding: **(IA)³ is the only method to outperform full-model fine-tuning**
* Insight: Very few parameters can be updated for excellent performance — high efficiency

---

###  Figure 3: Accuracy vs Inference FLOPs

* Content: Performance vs computational cost
* Finding: **T-Few outperforms GPT-3 175B with 1,000× fewer FLOPs**
* Insight: High performance with very low computational overhead

---

###  Table 1: Held-out Task Performance & Cost Summary

| Model      | Accuracy  | Inference FLOPs | Training FLOPs | Disk Storage |
| ---------- | --------- | --------------- | -------------- | ------------ |
| **T-Few**  | **72.4%** | 1.1e12          | 2.7e16         | 4.2MB        |
| GPT-3 175B | 66.6%     | 1.4e15          | 0              | 16KB         |

* Insight: T-Few dominates both in accuracy and cost-effectiveness

---

###  Table 2: RAFT Benchmark (Top 5 Results)

* **T-Few**: 75.8% (First to outperform human baseline at 73.5%)
* Insight: Outstanding real-world generalization in few-shot settings

---

###  Appendix F: Ablation Study

* Removing pre-training → –1.6% accuracy
* Removing LUL + LLN losses → –4.1%
* Removing both → –2.5%
* Insight: All components (losses, pretraining) contribute meaningfully to performance




<br/>
# refer format:     



@inproceedings{liu2022fewshot,
  title     = {Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning},
  author    = {Haokun Liu and Derek Tam and Mohammed Muqeeth and Jay Mohta and Tenghao Huang and Mohit Bansal and Colin Raffel},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  url       = {https://github.com/r-three/t-few}
}




Liu, Haokun, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. “Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper than In-Context Learning.” Advances in Neural Information Processing Systems (NeurIPS), 2022.  



