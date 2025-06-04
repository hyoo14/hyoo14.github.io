---
layout: post
title:  "[2022]LoRA: Low-Rank Adaptation of Large Language Models"  
date:   2025-06-04 16:34:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

추가 매트릭스(Rank: 매트릭스의 일종의 row크기) 주입해서 이 부분(주입된 부분)만 파인튜닝하는 기법  



짧은 요약(Abstract) :    



대규모 언어 모델을 다양한 작업에 맞게 조정(fine-tuning)하는 것은 매우 높은 비용이 듭니다. 예를 들어 GPT-3 (175B)의 경우 모든 파라미터를 새로 학습하는 방식은 저장 공간, 메모리, 연산 측면에서 매우 비효율적입니다. 이를 해결하기 위해 **LoRA (Low-Rank Adaptation)** 라는 방법을 제안합니다. LoRA는 기존 모델의 파라미터는 고정시키고, 각 Transformer 레이어에 소수의 학습 가능한 저랭크 행렬(분해 행렬 A, B)만 삽입하여 파라미터 수를 대폭 줄입니다.

이 방식은 다음과 같은 장점이 있습니다:

* 학습해야 할 파라미터 수를 **최대 10,000배 줄일 수 있음**
* GPU 메모리 사용량을 **3배까지 절감**
* **추론 속도 저하 없음** (어댑터와 달리)
* RoBERTa, DeBERTa, GPT-2, GPT-3 등의 다양한 모델에 적용 가능하며, 성능도 기존 full fine-tuning과 **동등하거나 더 우수함**

또한, 이 논문은 언어 모델 적응 과정에서 **랭크 결핍(rank-deficiency)** 현상이 있다는 실증적 분석을 제공하며, 이를 통해 LoRA 방식의 효과를 이론적으로도 뒷받침합니다.



A key paradigm in NLP involves large-scale pretraining followed by task-specific adaptation. However, full fine-tuning of large models like GPT-3 (175B parameters) is expensive and inefficient. This paper proposes **LoRA (Low-Rank Adaptation)**, which **freezes pre-trained weights** and **injects trainable low-rank decomposition matrices** into each Transformer layer. This significantly **reduces trainable parameters and GPU memory usage** for downstream tasks.

Compared to full fine-tuning of GPT-3 with Adam, LoRA:

* Reduces trainable parameters by **10,000x**
* Reduces memory requirement by **3x**
* Maintains or exceeds model quality on various models (RoBERTa, DeBERTa, GPT-2, GPT-3)
* Does not introduce **any additional inference latency**

LoRA is further supported by empirical analysis of **rank-deficiency** in model adaptation and is available with PyTorch support and implementations at [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA).

---




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



### 1. 핵심 아이디어

LoRA는 Transformer 언어 모델을 다운스트림 태스크에 맞춰 적응(fine-tuning)할 때, 기존의 전체 파라미터 업데이트 방식이 아닌 **저랭크 행렬 업데이트**를 통해 파라미터 효율성을 극대화합니다.

---

### 2. LoRA 수식: 저랭크 업데이트

Transformer의 Dense layer는 일반적으로 다음과 같이 작동합니다:

* 기존 선형층 가중치: $W_0 \in \mathbb{R}^{d \times k}$
* 입력 벡터: $x \in \mathbb{R}^{k}$
* 출력 벡터: $h = W_0 x$

LoRA는 여기에 \*\*저랭크 행렬 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$\*\*를 추가하여 다음과 같이 재구성합니다:

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

* $W_0$: 기존 학습된 가중치 (freeze)
* $\Delta W = BA$: 학습 가능한 저랭크 행렬 곱
* $r \ll \min(d, k)$: 매우 작은 rank 설정 (예: 1\~4)

즉, 모델은 기존 파라미터는 그대로 고정하고, 작은 추가 행렬 $B$와 $A$만 학습합니다.

---

### 3. 적용 위치 및 설정

* Transformer의 self-attention 모듈의 **query ($W_q$)와 value ($W_v$) projection matrix**에만 LoRA를 적용 (실험에서는 이 둘만 업데이트).
* **MLP 모듈은 학습 제외**하여 파라미터 수를 더 줄임.
* 초기화:

  * $A \sim \mathcal{N}(0, \sigma^2)$
  * $B = 0$ (초기에는 영향 없음)
* 스케일링: $\frac{\alpha}{r}$ (기본값으로 고정하여 별도 튜닝 없음)

---

### 4. 아키텍처적 장점

* **추론 속도 증가 없음**: $W = W_0 + BA$ 형태로 병합 가능
* 여러 태스크에서 **빠르게 전환 가능**: LoRA 모듈(A, B)만 바꾸면 됨
* GPU 메모리 절감 및 속도 향상 (예: GPT-3에서는 VRAM 1.2TB → 350GB)

---



### 1. Key Idea

LoRA enables parameter-efficient fine-tuning of large language models by freezing pretrained weights and introducing **trainable low-rank matrices** into Transformer layers.

---

### 2. Formula: Low-Rank Update

For a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in a linear layer, the standard output is:

$$
h = W_0 x
$$

LoRA modifies this by introducing a low-rank adaptation:

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

Where:

* $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$
* $r \ll \min(d, k)$: rank is small (e.g., 1–4)
* $W_0$ is **frozen**, only $A, B$ are trainable.

---

### 3. Where LoRA is Applied

* Applied to **self-attention projection matrices** (e.g., $W_q, W_v$) in Transformers
* **MLP and LayerNorm are kept frozen** for simplicity and efficiency
* Initialization:

  * $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$
* Optional scaling factor: $\frac{\alpha}{r}$ applied to $BA$

---

### 4. Architectural Benefits

* **No inference latency**: Matrices can be merged as $W = W_0 + BA$
* **Efficient task switching**: Only LoRA weights $A, B$ need to be changed
* Up to **3× less GPU memory**, **10,000× fewer trainable parameters**
* Example: GPT-3 (175B) memory reduced from 1.2 TB → 350 GB




   
 
<br/>
# Results  




### 1.  평가 대상 아키텍처

LoRA는 다음과 같은 주요 사전학습 모델들에 적용되어 평가되었습니다:

* **RoBERTa-base (125M)** / **RoBERTa-large (355M)**
* **DeBERTa-XXL (1.5B)**
* **GPT-2 medium (355M)** / **GPT-2 large (774M)**
* **GPT-3 (175B)**

각각의 모델에 대해 기존 full fine-tuning, adapter 계열, prefix-tuning 등 다양한 경쟁 기법들과 비교 실험이 진행되었습니다.

---

### 2.  테스트 데이터 및 태스크

| 모델                | 태스크 종류       | 사용 데이터셋                                                             |
| ----------------- | ------------ | ------------------------------------------------------------------- |
| RoBERTa / DeBERTa | 자연어 이해 (NLU) | **GLUE Benchmark** (MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B) |
| GPT-2             | 자연어 생성 (NLG) | **E2E NLG Challenge**                                               |
| GPT-3             | 다양한 생성 및 분류  | **WikiSQL**, **MNLI**, **SAMSum** (summarization)                   |

---

### 3.  주요 메트릭

| 태스크 유형                | 사용 메트릭                                                            |
| --------------------- | ----------------------------------------------------------------- |
| GLUE (NLU)            | Accuracy, F1, Matthew’s Correlation, Pearson/Spearman Correlation |
| Text Generation (NLG) | BLEU, NIST, METEOR, ROUGE-L, CIDEr                                |
| SQL Generation        | Logical form accuracy                                             |
| Summarization         | ROUGE-1, ROUGE-2, ROUGE-L                                         |

---

### 4.  핵심 결과 요약

* **RoBERTa & DeBERTa**:

  * LoRA는 full fine-tuning과 거의 동등한 성능 또는 더 높은 성능을 달성함
  * 예: RoBERTa-base 기준으로 GLUE 평균 점수 87.2 (LoRA) vs. 86.4 (Full FT)

* **GPT-2 (E2E NLG)**:

  * LoRA는 Adapter나 Prefix 계열보다 BLEU, ROUGE 등에서 높은 점수를 보임
  * 예: GPT-2 Large에서 BLEU 70.4 (LoRA), 68.5 (Full FT)

* **GPT-3 (175B)**:

  * 175B 파라미터 모델에서도 LoRA는 full fine-tuning 대비 동등 또는 더 높은 정확도
  * 4.7M 파라미터만으로도 **MNLI: 91.7**, **SAMSum ROUGE-L: 45.9** 등 높은 성능

---



### 1.  Tested Architectures

LoRA was evaluated on the following pretrained language models:

* **RoBERTa-base (125M)** and **RoBERTa-large (355M)**
* **DeBERTa-XXL (1.5B)**
* **GPT-2 medium (355M)** and **large (774M)**
* **GPT-3 (175B)**

LoRA was compared against **full fine-tuning**, **adapter methods**, **prefix-tuning**, **BitFit**, and others.

---

### 2.  Test Datasets and Tasks

| Model            | Task Type                        | Dataset                                                             |
| ---------------- | -------------------------------- | ------------------------------------------------------------------- |
| RoBERTa, DeBERTa | NLU                              | **GLUE benchmark** (MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B) |
| GPT-2            | Text Generation (NLG)            | **E2E NLG Challenge**                                               |
| GPT-3            | Text Generation & Classification | **WikiSQL**, **MNLI**, **SAMSum** (summarization)                   |

---

### 3.  Evaluation Metrics

| Task Type       | Metrics                                      |
| --------------- | -------------------------------------------- |
| GLUE (NLU)      | Accuracy, F1, Matthew’s Corr., Pearson Corr. |
| NLG (Text Gen.) | BLEU, NIST, METEOR, ROUGE-L, CIDEr           |
| SQL             | Logical form accuracy                        |
| Summarization   | ROUGE-1, ROUGE-2, ROUGE-L                    |

---

### 4.  Key Results

* **RoBERTa / DeBERTa**:

  * LoRA matches or slightly outperforms full fine-tuning on GLUE tasks.
  * Example: RoBERTa-base achieves 87.2 (LoRA) vs. 86.4 (FT) on GLUE avg.

* **GPT-2 (E2E NLG)**:

  * LoRA outperforms Adapter and Prefix methods on BLEU, ROUGE.
  * Example: GPT-2 Large BLEU = 70.4 (LoRA) vs. 68.5 (FT)

* **GPT-3 (175B)**:

  * LoRA achieves similar or better performance with **0.01–0.1% trainable parameters**
  * Example: With 4.7M params, achieves **MNLI: 91.7**, **ROUGE-L: 45.9** on SAMSum

---





<br/>
# 예제  





### 예시 1: GLUE 벤치마크 (자연어 이해)

* **모델**: RoBERTa-base (125M 파라미터)
* **적용 방법**: LoRA 적용 시 학습 파라미터 수 0.3M (약 0.24%)
* **태스크**: MNLI, SST-2, MRPC, CoLA 등 8개 NLU 과제
* **결과 예시** (LoRA vs Full Fine-tuning):

  * MNLI: 87.5 vs 87.6
  * SST-2: 95.1 vs 94.8
  * MRPC: 89.7 vs 90.2
  * 평균 점수: **87.2 (LoRA)** vs **86.4 (FT)**

 아주 적은 파라미터만 학습해도 full fine-tuning과 거의 유사한 성능 달성

---

### 예시 2: GPT-2 Medium + E2E NLG Challenge (자연어 생성)

* **모델**: GPT-2 Medium (355M 파라미터)
* **태스크**: 레스토랑 정보 → 자연어 문장 생성 (예: "The Eagle is a pub near the city center.")
* **데이터셋**: [E2E NLG Challenge Dataset](https://github.com/tuetschek/e2e-dataset)
* **평가 지표**:

  * BLEU, ROUGE-L, CIDEr 등
* **결과 예시 (LoRA vs FT)**:

  * BLEU: **70.4 (LoRA)** vs 68.2 (Full FT)
  * ROUGE-L: **71.8 (LoRA)** vs 71.0
  * CIDEr: **2.53 (LoRA)** vs 2.47

LoRA는 파라미터 수가 100배 적어도 생성 품질이 더 우수함

---

### 예시 3: GPT-3 175B + WikiSQL / MNLI / SAMSum

* **모델**: GPT-3 (175B)
* **LoRA 파라미터 수**: 4.7M (0.0027%)
* **데이터 및 결과**:

  * **WikiSQL** (NL → SQL 변환):

    * Acc: **73.4% (LoRA)** vs 73.8% (Full FT)
  * **MNLI-matched**:

    * Acc: **91.7% (LoRA)** vs 89.5% (FT)
  * **SAMSum (요약)**:

    * ROUGE-L: **45.9 (LoRA)** vs 44.5 (FT)



### Example 1: GLUE Benchmark (NLU)

* **Model**: RoBERTa-base (125M parameters)
* **LoRA Params**: 0.3M trainable (0.24% of full model)
* **Tasks**: MNLI, SST-2, MRPC, CoLA, etc.
* **Results (LoRA vs Full Fine-Tuning)**:

  * MNLI: 87.5 vs 87.6
  * SST-2: 95.1 vs 94.8
  * MRPC: 89.7 vs 90.2
  * **GLUE Avg**: **87.2 (LoRA)** vs **86.4 (FT)**

 LoRA achieves nearly the same performance with **\~99.76% fewer parameters**

---

### Example 2: GPT-2 Medium + E2E NLG Challenge (Text Generation)

* **Model**: GPT-2 Medium (355M)
* **Task**: Structured data → sentence generation (e.g., restaurant descriptions)
* **Dataset**: [E2E NLG Challenge](https://github.com/tuetschek/e2e-dataset)
* **Metrics**: BLEU, ROUGE-L, METEOR, CIDEr
* **Results**:

  * BLEU: **70.4 (LoRA)** vs 68.2 (Full FT)
  * ROUGE-L: **71.8 (LoRA)** vs 71.0
  * CIDEr: **2.53 (LoRA)** vs 2.47

 LoRA produces **higher-quality output** with only 0.1% of parameters

---

### Example 3: GPT-3 175B + WikiSQL, MNLI, SAMSum

* **Model**: GPT-3 (175B parameters)
* **LoRA Params**: 4.7M (only 0.0027% of full model)
* **Tasks and Results**:

  * **WikiSQL**: 73.4% (LoRA) vs 73.8% (FT)
  * **MNLI-matched**: **91.7% (LoRA)** vs 89.5% (FT)
  * **SAMSum summarization**:

    * ROUGE-L: **45.9 (LoRA)** vs 44.5 (FT)

 Even on GPT-3 scale, LoRA achieves **comparable or better** performance





<br/>  
# 요약   




LoRA는 대규모 언어 모델의 적응 과정에서 전체 파라미터를 업데이트하지 않고, 저랭크 행렬 $A$, $B$만을 삽입해 학습 효율을 극대화하는 방법입니다.
GLUE, E2E NLG, WikiSQL 등 다양한 태스크에서 LoRA는 전체 fine-tuning 대비 유사하거나 더 나은 성능을 보이면서도 수천 배 적은 파라미터만 학습합니다.
예를 들어 GPT-3에서는 0.003%의 파라미터만 학습해도 MNLI 정확도 91.7%를 달성하며 full fine-tuning보다 높은 성능을 보였습니다.

---



LoRA improves adaptation efficiency by freezing pretrained weights and learning only low-rank matrices $A$ and $B$ during fine-tuning.
Across tasks like GLUE, E2E NLG, and WikiSQL, LoRA achieves comparable or better performance than full fine-tuning while updating thousands of times fewer parameters.
For instance, on GPT-3, LoRA achieves 91.7% accuracy on MNLI using just 0.003% of the model's parameters, outperforming full fine-tuning.



<br/>  
# 기타  





###  Figure 1 – LoRA 구조 도식

* **설명**: Transformer의 기존 weight $W$에 저랭크 행렬 $B, A$를 추가하는 구조 시각화
* **인사이트**:

  * $\Delta W = BA$ 형태의 업데이트만 학습되며, 기존 $W$는 고정
  * 기존 weight와 병합 가능 → 추론 지연 없음

---

###  Table 1 – GPT-2 추론 속도 비교

| Method   | Latency (ms) at Batch=1 |
| -------- | ----------------------- |
| Full FT  | 19.8                    |
| AdapterL | 23.9 (**+20.7%**)       |
| AdapterH | 25.8 (**+30.3%**)       |
| LoRA     | **19.8 (0% overhead)**  |

* **인사이트**:

  * LoRA는 **추론 지연 없음**
  * 어댑터는 구조 추가로 인해 짧은 시퀀스일수록 지연 큼

---

###  Table 2 – GLUE 결과 (RoBERTa, DeBERTa)

* **LoRA**는 RoBERTa-base/large 및 DeBERTa-XXL에 대해 full fine-tuning 수준 또는 더 나은 성능 달성

* **예시**: RoBERTa-large 평균 점수

  * Full FT: 88.9
  * LoRA: **89.0**

* **인사이트**:

  * 적은 파라미터로도 높은 정확도 유지 가능
  * 특히 STS-B, RTE와 같이 민감한 태스크에서 더 높은 성능 보임

---

###  Table 3 – GPT-2 + E2E NLG 결과

* **평가 지표**: BLEU, ROUGE-L, CIDEr 등

* **결과 예시**:

  * GPT-2 Medium + LoRA: BLEU = 70.4, ROUGE-L = 71.8, CIDEr = 2.53
  * GPT-2 Full FT: BLEU = 68.2, ROUGE-L = 71.0, CIDEr = 2.47

* **인사이트**:

  * NLG에서도 LoRA가 생성 품질 측면에서 우수
  * 파라미터 수는 Adapter보다 적고 성능은 더 좋음

---

###  Table 4 – GPT-3 성능 (WikiSQL, MNLI, SAMSum)

* **LoRA (4.7M params)**:

  * WikiSQL: 73.4% (vs 73.8 FT)
  * MNLI: **91.7%** (vs 89.5 FT)
  * SAMSum ROUGE-L: **45.9** (vs 44.5 FT)

* **인사이트**:

  * GPT-3에서도 효과적
  * 특히 MNLI에서 full fine-tuning보다 높은 성능

---

###  Appendix 주요 인사이트

* **Appendix C**: Adapter는 짧은 시퀀스에서 지연이 큼 (추론 속도 실험)
* **Appendix F**: 각 실험의 하이퍼파라미터 정리 (GPU, 학습률, 배치 크기 등)
* **Appendix I.2**: 파라미터 수가 증가할수록 LoRA는 성능이 **완만하게 증가** → **스케일 확장성 우수**
* **Appendix I.3**: Low-data regime (데이터 적을 때)에서도 LoRA가 강건함

---



###  Figure 1 – LoRA Architecture Diagram

* **Description**: Visualizes the low-rank adaptation $\Delta W = BA$ added to a frozen weight $W_0$
* **Insight**:

  * Only $A, B$ are trained, no updates to original weights
  * The model can merge $W_0 + BA$ → **no inference latency**

---

###  Table 1 – Inference Latency (GPT-2)

| Method      | Batch=1 Latency      |
| ----------- | -------------------- |
| Fine-tuning | 19.8 ms              |
| AdapterL    | 23.9 ms (**+20.7%**) |
| AdapterH    | 25.8 ms (**+30.3%**) |
| LoRA        | **19.8 ms (0%)**     |

* **Insight**:

  * **LoRA adds no latency** at inference time
  * Adapters slow down short-sequence inference due to structural depth

---

###  Table 2 – GLUE Results (RoBERTa / DeBERTa)

* **RoBERTa-large Avg**:

  * Full FT: 88.9
  * LoRA: **89.0**

* **Insight**:

  * LoRA maintains high performance with significantly fewer parameters
  * Strong performance even on difficult tasks like STS-B and RTE

---

###  Table 3 – GPT-2 + E2E NLG Results

* **LoRA** (0.35M parameters):

  * BLEU: **70.4**
  * ROUGE-L: **71.8**
  * CIDEr: **2.53**

* **Full FT**:

  * BLEU: 68.2
  * ROUGE-L: 71.0
  * CIDEr: 2.47

* **Insight**:

  * LoRA outperforms adapters in both quality and parameter efficiency for NLG

---

###  Table 4 – GPT-3 Results (WikiSQL, MNLI, SAMSum)

* **LoRA (4.7M params)**:

  * WikiSQL: 73.4% (vs 73.8 FT)
  * MNLI: **91.7%** (vs 89.5 FT)
  * ROUGE-L (SAMSum): **45.9** (vs 44.5 FT)

* **Insight**:

  * LoRA performs **as well or better** than full fine-tuning, even at GPT-3 scale

---

###  Key Appendix Insights

* **Appendix C**: Adapters add latency especially at small batch sizes
* **Appendix F**: Detailed hyperparameters for reproducibility
* **Appendix I.2**: LoRA performance scales **smoothly with rank/params**
* **Appendix I.3**: LoRA remains effective in **low-data scenarios**

---




<br/>
# refer format:     


@inproceedings{hu2022lora,
  title={{LoRA: Low-Rank Adaptation of Large Language Models}},
  author={Hu, Edward and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  url={https://arxiv.org/abs/2106.09685}
}







Hu, Edward, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. “LoRA: Low-Rank Adaptation of Large Language Models.” International Conference on Learning Representations (ICLR), 2022. https://arxiv.org/abs/2106.09685.

