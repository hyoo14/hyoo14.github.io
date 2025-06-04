---
layout: post
title:  "[2023]QLORA: Efficient Finetuning of Quantized LLMs"  
date:   2025-06-04 16:24:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

Quantization + LoRA   



짧은 요약(Abstract) :    



우리는 QLoRA라는 효율적인 파인튜닝 기법을 제안합니다. 이 방법은 65억 개 파라미터를 가진 모델도 단일 48GB GPU에서 파인튜닝할 수 있을 만큼 메모리 사용을 줄이면서, 기존의 16비트 파인튜닝 성능을 그대로 유지합니다. QLoRA는 4비트로 양자화된 고정된 언어 모델을 기반으로, LoRA(Low-Rank Adapter)를 통해 그래디언트를 역전파합니다. 우리가 훈련한 Guanaco 모델 계열은 Vicuna 벤치마크에서 모든 공개 모델 중 최고 성능을 기록했으며, 단 24시간의 파인튜닝으로 ChatGPT 성능의 99.3% 수준에 도달했습니다.

QLoRA는 성능을 유지하면서도 메모리를 절약하기 위해 다음의 기술들을 도입합니다:

4-bit NormalFloat (NF4): 정규 분포 가중치에 대해 정보 이론적으로 최적인 새로운 데이터 타입.

Double Quantization: 양자화 상수를 한 번 더 양자화하여 평균 메모리 사용량을 줄임.

Paged Optimizers: 메모리 스파이크를 완화하기 위해 페이징 기법을 활용.

우리는 QLoRA로 1,000개 이상의 모델을 파인튜닝하였으며, 다양한 모델 아키텍처와 크기에서의 챗봇 및 인스트럭션 성능을 분석했습니다. 그 결과, 고품질 소규모 데이터셋이 대용량 데이터셋보다 성능에 더 중요한 영향을 미치며, GPT-4 기반 평가가 인간 평가와 유사한 순위를 매기는 것을 확인했습니다. 또한 현존하는 챗봇 벤치마크의 신뢰성 문제도 지적하며, Guanaco가 ChatGPT보다 성능이 낮은 경우도 분석했습니다. 모든 모델과 코드, 4비트 훈련용 CUDA 커널을 공개합니다.


We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).

Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU.

QLoRA introduces a number of innovations to save memory without sacrificing performance:
(a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights
(b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants
(c) Paged Optimizers to manage memory spikes.

We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models).

Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. We release all of our models and code, including CUDA kernels for 4-bit training.


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



**1. 핵심 아이디어:**
QLoRA는 **4비트로 양자화된 언어 모델**을 고정(freeze)한 채로, 그 위에 \*\*Low-Rank Adapter (LoRA)\*\*만을 훈련시키는 방식입니다. 이 방식은 **전체 모델을 16비트로 파인튜닝하는 성능**을 유지하면서도, **메모리 사용량을 획기적으로 줄여줍니다**.

**2. 주요 구성 요소:**

* **4-bit NormalFloat (NF4)**

  * 정규분포를 따르는 파라미터를 위한 정보 이론적으로 최적화된 데이터 타입
  * 각 양자화 bin에 동일한 수의 값이 들어가도록 설계됨

* **Double Quantization (이중 양자화)**

  * 양자화 상수 자체를 또다시 양자화하여 평균 파라미터당 약 0.373비트 절감

* **Paged Optimizers**

  * GPU 메모리가 부족할 때 자동으로 CPU로 데이터를 스와핑하는 기능을 통해 **메모리 스파이크 방지**

**3. 수식 정리:**

* LoRA 적용 후의 projection:

  $$
  Y = XW + sXL_1L_2
  $$

  여기서:

  * $X \in \mathbb{R}^{b \times h}$
  * $W \in \mathbb{R}^{h \times o}$: 원래의 고정된 가중치
  * $L_1 \in \mathbb{R}^{h \times r}$, $L_2 \in \mathbb{R}^{r \times o}$: 학습되는 low-rank 어댑터
  * $s$: 스케일링 상수

* QLoRA의 전체 연산:

  $$
  Y_{\text{BF16}} = X_{\text{BF16}} \cdot \text{doubleDequant}(c_1^{\text{FP32}}, c_2^{\text{NF4}}, W^{\text{NF4}}) + X_{\text{BF16}} \cdot L_1^{\text{BF16}} \cdot L_2^{\text{BF16}}
  $$

  $$
  \text{doubleDequant}(c_1, c_2, W) = \text{dequant}(\text{dequant}(c_1, c_2), W)
  $$

**4. 훈련 전략:**

* 저장 형식은 4비트 NF4를 사용하고, 계산 시에는 BFloat16으로 디퀀타이즈
* **LoRA 파라미터만 업데이트**, 원래 모델의 4비트 가중치는 고정
* Paged Optimizer 덕분에 48GB GPU로 65B 모델도 훈련 가능

---



**1. Core Idea:**
QLoRA introduces a memory-efficient finetuning strategy by freezing a **4-bit quantized base model** and training only the **Low-Rank Adapter (LoRA)** layers. It achieves **full 16-bit finetuning performance** while drastically reducing memory consumption.

**2. Key Components:**

* **4-bit NormalFloat (NF4):**

  * An information-theoretically optimal data type for normally distributed weights.
  * Ensures equal value assignment per quantization bin.

* **Double Quantization:**

  * Compresses the quantization constants themselves, saving approximately 0.373 bits per parameter.

* **Paged Optimizers:**

  * Uses NVIDIA’s unified memory to handle out-of-memory spikes by offloading data to CPU when needed.

**3. Core Equations:**

* LoRA projection:

  $$
  Y = XW + sXL_1L_2
  $$

  where:

  * $X \in \mathbb{R}^{b \times h}$
  * $W \in \mathbb{R}^{h \times o}$ is the frozen base weight
  * $L_1 \in \mathbb{R}^{h \times r}$, $L_2 \in \mathbb{R}^{r \times o}$: trainable low-rank adapters
  * $s$: scaling factor

* Full QLoRA forward computation:

  $$
  Y_{\text{BF16}} = X_{\text{BF16}} \cdot \text{doubleDequant}(c_1^{\text{FP32}}, c_2^{\text{NF4}}, W^{\text{NF4}}) + X_{\text{BF16}} \cdot L_1^{\text{BF16}} \cdot L_2^{\text{BF16}}
  $$

  $$
  \text{doubleDequant}(c_1, c_2, W) = \text{dequant}(\text{dequant}(c_1, c_2), W)
  $$

**4. Training Strategy:**

* Store weights in 4-bit NF4 format, compute in BFloat16.
* **Only LoRA parameters are updated** during training.
* Paged optimizers enable training even 65B models on 48GB GPUs by offloading memory dynamically.




   
 
<br/>
# Results  




**1. 비교 대상 아키텍처**

* **16-bit full finetuning**
* **16-bit LoRA finetuning**
* **4-bit LoRA with Float4**
* **QLoRA (4-bit NormalFloat + Double Quantization)**

**2. 테스트 테스크 및 벤치마크**

* **MMLU (Massive Multitask Language Understanding)**
  → 다양한 과목 기반 문제를 풀도록 설계된 일반적인 지능 테스트
  → 메트릭: 5-shot accuracy

* **Vicuna Benchmark**
  → GPT-4 기반 채점으로 다수 챗봇 응답을 비교하는 자연어 대화 평가
  → 메트릭: GPT-4 기반 Elo 점수, ChatGPT 대비 상대 점수

**3. 주요 결과 요약**

* **QLoRA는 16-bit full finetuning과 동등한 성능을 달성**

  * MMLU에서 7B\~65B 모델까지 4-bit NormalFloat + DQ가 BFloat16 성능에 근접
  * 예: 65B 모델에서 FLAN v2로 파인튜닝 시 5-shot MMLU 정확도는 63.9%

* **Vicuna 벤치마크에서 Guanaco (QLoRA로 훈련된 모델)가 최고 오픈소스 성능 기록**

  * Guanaco-65B는 ChatGPT 성능의 99.3% 달성
  * Guanaco-33B는 Vicuna-13B보다 낮은 메모리(21GB)로 더 높은 점수 기록
  * Guanaco-7B도 5GB 메모리로 Alpaca-13B보다 20%포인트 높은 점수 달성

**4. 평가 방식**

* **Elo rating** (GPT-4 또는 인간 평가자 기반)
* **MMLU 5-shot accuracy**
* **Perplexity (언어 모델링 평가 지표)**

---



**1. Baseline Architectures Compared**

* **16-bit Full Finetuning**
* **16-bit LoRA Finetuning**
* **4-bit LoRA with Float4**
* **QLoRA (4-bit NormalFloat + Double Quantization)**

**2. Evaluation Tasks & Benchmarks**

* **MMLU (Massive Multitask Language Understanding)**
  → A benchmark testing general knowledge across 57 tasks
  → Metric: 5-shot accuracy

* **Vicuna Benchmark**
  → A chatbot evaluation benchmark using GPT-4 judgments to compare response quality
  → Metrics: Elo score (GPT-4 and human raters), relative percentage vs. ChatGPT

**3. Key Results**

* **QLoRA matches 16-bit full finetuning performance**

  * 4-bit NormalFloat with Double Quantization achieves near-identical accuracy
  * Example: LLaMA-65B model finetuned on FLAN v2 scored 63.9% on MMLU (5-shot)

* **Guanaco (QLoRA-trained models) achieves state-of-the-art open-source performance**

  * **Guanaco-65B** reached **99.3%** of ChatGPT performance on Vicuna benchmark
  * **Guanaco-33B** outperforms **Vicuna-13B**, using **less memory (21GB vs 26GB)**
  * **Guanaco-7B** (5GB) beats **Alpaca-13B** by over **20 percentage points**

**4. Evaluation Metrics**

* **Elo rating** (via GPT-4 or human judges)
* **MMLU 5-shot accuracy**
* **Language modeling perplexity**

---




<br/>
# 예제  




**1. 데이터셋 예시**

* **OASST1**: OpenAssistant의 대화형 지시문 데이터셋 (작지만 고품질)
* **FLAN v2**: 구글에서 만든 대규모 인스트럭션 튜닝용 데이터셋
* **Alpaca**: Stanford에서 만든 LLaMA 기반 instruction-following 데이터셋
* **HH-RLHF, Chip2, Longform 등** 다양한 스타일과 품질의 지시문 포함

> 예시 비교:
>
> * OASST1 (약 9천개 샘플) → 높은 챗봇 성능
> * FLAN v2 (약 45만개 샘플) → 낮은 챗봇 성능 (Vicuna 기준)

**2. 모델 사이즈 & 메모리 예시**

| 모델 이름       | 파라미터 수 | 포맷    | 메모리 사용 | Vicuna 점수 (ChatGPT 대비) |
| ----------- | ------ | ----- | ------ | ---------------------- |
| Guanaco-65B | 65B    | 4-bit | 41GB   | 99.3%                  |
| Guanaco-33B | 33B    | 4-bit | 21GB   | 97.8%                  |
| Guanaco-13B | 13B    | 4-bit | 10GB   | 90.4%                  |
| Guanaco-7B  | 7B     | 4-bit | 5GB    | 87.0%                  |

**3. 메트릭 및 결과 예시**

* **MMLU 5-shot accuracy (LLaMA + QLoRA)**

  * 65B 모델 + FLAN v2 → **63.9%**
  * 33B 모델 + OASST1 → **62.2%**

**4. GPU 한 대에서 가능한 예시**

* 65B 모델: 48GB GPU에서 하루 내 훈련 가능
* 33B 모델: 24GB GPU에서 12시간 이내 훈련 가능
* 7B 모델: 5GB 메모리, 스마트폰에서 추론 및 훈련 가능

---



**1. Dataset Examples**

* **OASST1**: Small but high-quality instruction-following dataset from OpenAssistant
* **FLAN v2**: Large-scale instruction-tuning dataset from Google
* **Alpaca**: Stanford-developed instruction-tuned LLaMA variant
* **Others**: HH-RLHF, Chip2, Longform, UnnaturalInstructions

> Example comparison:
>
> * **OASST1 (\~9K samples)** → best chatbot performance
> * **FLAN v2 (\~450K samples)** → poor chatbot performance (in Vicuna benchmark)

**2. Model Size & Memory Usage**

| Model       | Params | Format | Memory Usage | Vicuna Score (vs. ChatGPT) |
| ----------- | ------ | ------ | ------------ | -------------------------- |
| Guanaco-65B | 65B    | 4-bit  | 41 GB        | 99.3%                      |
| Guanaco-33B | 33B    | 4-bit  | 21 GB        | 97.8%                      |
| Guanaco-13B | 13B    | 4-bit  | 10 GB        | 90.4%                      |
| Guanaco-7B  | 7B     | 4-bit  | 5 GB         | 87.0%                      |

**3. Metrics & Results**

* **MMLU 5-shot accuracy (LLaMA + QLoRA)**

  * 65B model + FLAN v2 → **63.9%**
  * 33B model + OASST1 → **62.2%**

**4. Hardware Feasibility**

* **65B**: Trained in 24 hours on a **single 48GB GPU**
* **33B**: Trained in under 12 hours on a **24GB consumer GPU**
* **7B**: Can run on smartphones with only **5GB of memory**




<br/>  
# 요약   



QLoRA는 4비트로 양자화된 LLM 위에 LoRA 어댑터를 덧붙여 파인튜닝하는 방식으로, 메모리 사용을 줄이면서도 16비트 성능을 유지합니다.
이 방법은 65B 모델도 단일 48GB GPU에서 훈련 가능하게 만들며, Vicuna 벤치마크에서 ChatGPT의 99.3% 성능을 달성했습니다.
특히 Guanaco-7B는 5GB 메모리만으로도 Alpaca-13B보다 20%포인트 더 높은 성능을 보이며 실용적인 경량 챗봇 구현이 가능합니다.



QLoRA fine-tunes 4-bit quantized LLMs by adding Low-Rank Adapters (LoRA), enabling efficient training with minimal memory overhead while preserving 16-bit performance.
This approach allows even 65B models to be fine-tuned on a single 48GB GPU, achieving 99.3% of ChatGPT’s performance on the Vicuna benchmark.
Notably, the 7B Guanaco model runs with just 5GB of memory and outperforms the 13B Alpaca model by over 20 percentage points.





<br/>  
# 기타  



#### Table 1: Elo Ratings (GPT-4 평가 기반)

* GPT-4 기준으로 Guanaco-65B, Guanaco-33B는 Vicuna-13B 및 ChatGPT-3.5보다 더 높은 점수를 받음.
* Guanaco 모델은 대부분의 상용 모델과 유사하거나 더 나은 품질을 달성함.

#### Table 3: MMLU 5-shot 정확도 (NF4, FP4 비교)

* NFloat4 + Double Quantization (DQ)는 BFloat16 성능과 거의 동일함.
* 반면 FP4는 일관되게 약 1%포인트 낮은 정확도를 보임.

#### Table 4: Vicuna 벤치마크 성능 비교 (GPT-4 기준)

* Guanaco-65B: ChatGPT 대비 99.3% 성능.
* Guanaco-7B는 5GB 메모리로도 Alpaca-13B보다 훨씬 우수한 성능을 달성.

#### Table 5: 다양한 데이터셋에서의 MMLU 성능

* FLAN v2가 MMLU 성능에서 가장 좋았지만, Vicuna 벤치마크에서는 낮은 성능.
* → "좋은 벤치마크 성능 ≠ 좋은 챗봇 성능"이라는 점을 강조.

#### Table 6: Elo 점수 비교 (Human/GPT-4 평가자 모두 포함)

* Guanaco-65B는 Vicuna, OA 두 벤치마크에서 모두 상위권.
* GPT-4와 인간 평가자의 순위는 대체로 일치하지만, Kendall’s tau는 0.43으로 중간 정도 일치.

#### Figure 1: 메모리 요구량 비교

* QLoRA는 기존 LoRA보다 훨씬 낮은 메모리 요구량을 달성함 (예: 65B 모델이 48GB에서 학습 가능).

#### Figure 2: NormalFloat이 정확도 측면에서 FP4보다 우수함

* 4-bit 양자화에서도 정확도를 유지하려면 FP4보다 NF4가 훨씬 유리함.

#### Appendix H: NF4 데이터 타입 구성 방법

* NormalFloat을 만들기 위해 정규분포의 이론적 분위수를 사용해 최적의 4-bit 표현을 구성하는 방법을 시각적으로 설명.

#### Appendix F: Guanaco의 실패 사례 및 정성 분석

* Guanaco가 실패하는 경우와 그 원인을 분석해, 향후 개선 방향을 제시.

---


#### Table 1: Elo Ratings (GPT-4 Evaluated)

* Guanaco-65B and Guanaco-33B outperform Vicuna-13B and ChatGPT-3.5 on GPT-4 judged Elo rankings.
* This shows that QLoRA-tuned models rival or surpass proprietary systems.

#### Table 3: MMLU 5-shot Accuracy (NF4 vs. FP4)

* NFloat4 + DQ matches BFloat16 in accuracy.
* FP4 consistently lags behind by around 1 percentage point.

#### Table 4: Vicuna Benchmark Results (GPT-4 Evaluated)

* Guanaco-65B achieves 99.3% of ChatGPT’s score.
* Even Guanaco-7B (5GB) significantly outperforms Alpaca-13B (10GB).

#### Table 5: MMLU Accuracy across Datasets

* FLAN v2 achieves the best MMLU results but ranks poorly on Vicuna.
* This illustrates that high benchmark scores do not necessarily equate to good chatbot performance.

#### Table 6: Elo Ratings by Human and GPT-4 Judges

* Guanaco-65B ranks among the best in both Vicuna and OpenAssistant benchmarks.
* There is moderate agreement between GPT-4 and human judges (Kendall τ = 0.43).

#### Figure 1: Memory Requirements Across Methods

* QLoRA significantly reduces memory usage compared to standard LoRA.
* It enables 65B models to be fine-tuned on a single 48GB GPU.

#### Figure 2: NormalFloat vs. FP4 Accuracy

* NormalFloat achieves better bit-level accuracy than FP4 across tasks.
* Provides strong empirical support for using NF4.

#### Appendix H: How to Construct NF4

* Explains how to create the NF4 data type using theoretical quantiles from a standard normal distribution.

#### Appendix F: Guanaco Failure Analysis

* Provides qualitative examples where Guanaco fails compared to ChatGPT.
* Useful for identifying limitations and guiding future improvements.




<br/>
# refer format:     


@inproceedings{dettmers2023qlora,
  title     = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle = {Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  url       = {https://github.com/artidoro/qlora}
}





Dettmers, Tim, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. "QLoRA: Efficient Finetuning of Quantized LLMs." In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS), 2023. https://github.com/artidoro/qlora.

