---
layout: post
title:  "[2025]Distillation Scaling Laws"  
date:   2025-07-13 21:49:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 


교사가 이미 존재하고 토큰이 제한된 경우 증류가 지도학습보다 효율적**이며, 반대로 교사도 훈련해야 할 경우 지도학습이 더 낫다.


짧은 요약(Abstract) :    





---



이 논문은 주어진 계산 자원(compute budget) 하에서 \*\*교사 모델과 학생 모델 사이의 계산 분배에 따라 증류(distillation)된 모델의 성능을 예측할 수 있는 스케일링 법칙(scaling law)\*\*을 제안합니다. 이를 통해 대규모 증류의 리스크를 줄이고, 교사와 학생 모두에 대해 **계산 최적화(compute-optimal)된 자원 분배 전략**을 설계할 수 있게 합니다.

논문은 두 가지 주요 상황에 대한 최적화 레시피를 제시합니다:

1. 이미 교사 모델이 존재하는 경우
2. 교사 모델을 새로 훈련시켜야 하는 경우

결과적으로, 여러 학생 모델을 증류하거나 기존 교사가 있는 경우에는 **증류가 지도 학습보다 효율적**일 수 있습니다. 반면, 교사도 새로 훈련하고 단 하나의 학생만 증류한다면 **지도 학습이 더 나은 선택**이 됩니다. 대규모 실험을 통해 이러한 증류 과정에 대한 깊은 이해를 제공하고, 향후 실험 설계를 도울 수 있는 통찰을 제공합니다.

---



This paper proposes a **distillation scaling law** that estimates the performance of a distilled model based on a given compute budget and how it is allocated between the teacher and the student. The findings mitigate the risks of large-scale distillation by enabling **compute-optimal allocation strategies** to maximize student performance.

The authors provide optimal distillation recipes for two main scenarios:

1. When a teacher already exists.
2. When a teacher must be trained.

They find that **distillation outperforms supervised learning** when many students are involved or when a teacher is already available. In contrast, if only a single student is to be distilled and the teacher also needs training, **supervised learning is typically more effective**. This large-scale study improves the understanding of the distillation process and informs experimental design.





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





이 논문에서는 \*\*Transformer 기반의 언어 모델(학생과 교사 모델)\*\*을 사용하여 \*\*증류(distillation)\*\*의 스케일링 법칙을 분석합니다. 다음은 주요 구성 요소입니다:

* **모델 아키텍처**:

  * 모두 Transformer 기반이며, **Multi-Head Attention (MHA)**, **Pre-Normalization**, **RMSNorm**, \*\*Rotary Position Embedding (RoPE)\*\*을 사용합니다.
  * 모델 크기는 **143M부터 12.6B 파라미터**까지 다양하게 실험하였으며, 교사 모델이 학생보다 작을 수도, 클 수도 있습니다.
  * 학습 안정성을 위해 **µP(Simple)** 스케일링 전략을 적용하여 학습률 등 하이퍼파라미터가 모델 크기와 상관없이 전이 가능하도록 했습니다.

* **트레이닝 데이터**:

  * **C4 영어 데이터셋**의 하위셋만 사용.
  * 교사와 학생은 \*\*서로 다른 데이터 분할(split)\*\*에서 학습함으로써 데이터 누출을 방지.

* **훈련 설정**:

  * **시퀀스 길이**는 4096.
  * 모든 모델은 **Chinchilla-optimal token-to-parameter ratio (M ≈ 20)** 기준을 따름.
  * 대부분의 실험에서 **순수 증류(pure distillation)** 설정을 사용하였고, λ=1, τ=1 (즉, 오직 교사 출력을 사용하고 NTP, Z-loss는 제외함).
  * 학습 손실 함수는 다음과 같이 구성됨:

    $$
    L_S = (1 - \lambda) \cdot L_{\text{NTP}} + \lambda \cdot L_{\text{KD}} + \lambda_Z \cdot L_Z
    $$

    여기서 distillation temperature $\tau = 1$, λ = 1로 설정하여 pure distillation만 고려.

* **스케일링 실험 프로토콜**:

  * **IsoFLOP 프로파일**: 동일한 계산량 내에서 모델 크기와 토큰 수를 달리하여 성능 비교.
  * 다양한 학생–교사 조합 실험을 통해 **학생 크기, 증류 토큰 수, 교사 크기 및 성능**이 학생의 성능(크로스 엔트로피)에 미치는 영향을 정량화하고 모델화.

---


The paper investigates distillation scaling laws using **Transformer-based language models** for both student and teacher models. Key methodological details include:

* **Model Architecture**:

  * All models use standard Transformer components including **Multi-Head Attention (MHA)**, **Pre-Normalization**, **RMSNorm**, and **Rotary Position Embedding (RoPE)**.
  * Model sizes range from **143M to 12.6B parameters**, with both smaller and larger teachers relative to students.
  * Training follows **µP(Simple)** scaling, which enables hyperparameter transfer (e.g., learning rate) across different model sizes.

* **Training Data**:

  * Uses only the **English subset of the C4 dataset**.
  * The teacher and student are trained on **different data splits** to avoid data leakage.

* **Training Setup**:

  * All models are trained with a **sequence length of 4096**.
  * Models follow the **Chinchilla-optimal token-to-parameter ratio** (M ≈ 20).
  * Most experiments are conducted under **pure distillation**, with **λ = 1** and **τ = 1**, meaning only the distillation loss is used.
  * The total student loss is:

    $$
    L_S = (1 - \lambda) \cdot L_{\text{NTP}} + \lambda \cdot L_{\text{KD}} + \lambda_Z \cdot L_Z
    $$

* **Scaling Experiment Protocols**:

  * Uses **IsoFLOP profiles** to vary model size and token count under fixed compute budgets.
  * A wide range of student-teacher configurations are tested to analyze how **student size, distillation token count, and teacher characteristics** influence student cross-entropy, enabling the formulation of a predictive **distillation scaling law**.




   
 
<br/>
# Results  





이 논문은 다양한 **학생-교사 모델 조합**을 실험하여 \*\*증류 스케일링 법칙(Distillation Scaling Law)\*\*을 도출하고 검증합니다. 주요 결과는 다음과 같습니다:

1. **경쟁 설정**:

   * 모델 크기 범위: **143M \~ 12.6B 파라미터**
   * 사용된 교사 모델 수: 총 10개 (198M \~ 7.75B), 학생 모델 수: 5개 이상
   * 비교 대상:

     * **지도 학습(Supervised Pretraining)**
     * **증류 학습(Distillation Pretraining)**
     * Chinchilla 최적화 조건(M ≈ 20) 하에서 다양한 FLOPs 예산으로 실험됨

2. **주요 결과**:

   * \*\*증류는 특정 조건(교사 존재 + 제한된 토큰 수)\*\*에서 **지도 학습보다 우수**한 성능을 보임.
   * 그러나 **교사도 훈련해야 하고 학생 수가 적은 경우**, 지도 학습이 일반적으로 더 효율적임.
   * **Capacity Gap 현상** 확인: 너무 성능 좋은 교사는 오히려 학생 성능을 떨어뜨릴 수 있음.
   * 학생 크기와 토큰 수가 증가하면 **지도 학습이 항상 상한 성능에 도달**함 → 증류가 효율적인 조건은 제한됨.

3. **스케일링 법칙 정확도**:

   * 논문에서 제안한 distillation scaling law는 **예측 오차 1% 이내**의 정밀도를 달성함.
   * 교사 크기 자체보다는 \*\*교사의 Cross-Entropy (LT)\*\*가 학생 성능에 더 중요한 영향을 미침.

4. **Compute 예산 시나리오 분석**:

   * 총 계산량이 충분할 경우 → **지도 학습이 항상 증류보다 같거나 우수**
   * 단, **여러 학생에게 증류하거나 교사 재사용 가능**할 때는 증류가 **계산적으로 유리**

---



This paper evaluates a wide range of **student-teacher pairs** to establish and validate a **Distillation Scaling Law**. Key results include:

1. **Baseline Comparison**:

   * Model sizes range from **143M to 12.6B parameters**
   * Teacher models: 10 sizes (198M to 7.75B), Student models: ≥ 5 sizes
   * Evaluated under both **supervised pretraining** and **distillation pretraining**
   * Experiments conducted under **Chinchilla-optimal** token-to-parameter ratio (M ≈ 20) across varying FLOPs budgets

2. **Main Findings**:

   * **Distillation outperforms supervised learning** when a teacher already exists and token budget is limited.
   * If the **teacher must also be trained**, and only one student is needed, **supervised learning is generally superior**.
   * Identified the **capacity gap**: overly capable teachers may degrade student performance.
   * As student size and token count increase, **supervised training consistently reaches optimal performance**, limiting the effectiveness of distillation.

3. **Scaling Law Accuracy**:

   * The proposed distillation scaling law achieves **≤ 1% relative prediction error**.
   * It reveals that **teacher cross-entropy (LT)**, not teacher size, is the dominant factor influencing student performance.

4. **Compute Budget Scenarios**:

   * With sufficient compute, **supervised training always matches or exceeds distillation**.
   * However, **distillation becomes more compute-efficient** when reusing a teacher across multiple students or in server-based settings.





<br/>
# 예제  




이 논문은 \*\*언어 모델(Transformer LM)\*\*의 **증류 사전학습(distillation pretraining)** 상황을 정량적으로 분석하기 위해, **다양한 학생 모델이 다양한 교사 모델로부터 증류될 때의 성능을 비교**하는 실험을 진행합니다.

####  테스크 예시 (Task Example)

* **다루는 과제**:

  * **다음 토큰 예측 (Next Token Prediction)**
  * 즉, 주어진 문맥 $x(<i)$에 대해 다음 단어 $x(i)$를 예측하는 확률 분포를 학습함.

####  테스트 데이터 예시 (Dataset Example)

* **사용 데이터셋**:

  * \*\*C4 데이터셋 (Colossal Clean Crawled Corpus)\*\*의 **영어 부분만 사용**
  * 교사와 학생은 **서로 다른 데이터 split**에서 훈련됨 (데이터 누출 방지)

* 예:

  * 문장 일부 `“The capital city of France is”`
  * 목표 토큰: `"Paris"`

* 교사는 이 시퀀스에 대해 다음 단어 분포 $p_T(x(i) | x(<i))$를 출력하고, 학생은 이를 모방하는 $q_S(x(i) | x(<i))$을 학습.

####  입력/출력 형식 (Input/Output Example)

* **입력**:

  * 토큰 시퀀스: `"The capital city of France is"` → 각 단어는 정수 인덱스로 토큰화됨
  * 입력 형태: `[2013, 768, 92, 101, 411, 34]` (예시 인덱스)

* **출력**:

  * 교사: 해당 문맥에 대한 확률 분포 (로짓) 출력, 예: `[logits_1, logits_2, ..., logits_V]`
  * 학생: 동일 문맥에 대해 자신의 로짓 출력, 이후 KL Divergence 손실 계산
  * 최종 목표: 학생이 교사의 soft target 분포를 정확히 재현하도록 학습

---



This paper provides detailed empirical examples from **language modeling tasks using distillation pretraining**.

####  Task Example

* **Task**:

  * **Next Token Prediction**
  * Given a context $x(<i)$, the goal is to predict the next token $x(i)$ with maximum likelihood.

####  Dataset Example

* **Dataset**:

  * The English-only subset of the **C4 dataset (Colossal Clean Crawled Corpus)**
  * Teacher and student models are trained on **different data splits** to avoid data leakage.

* Example:

  * Text input: `"The capital city of France is"`
  * Ground truth next token: `"Paris"`

* The teacher outputs a next-token probability distribution $p_T(x(i)|x(<i))$, and the student learns to match it with $q_S(x(i)|x(<i))$.

####  Input/Output Format

* **Input**:

  * A tokenized sequence of text: e.g., `"The capital city of France is"` → `[2013, 768, 92, 101, 411, 34]`
  * These are integer indices representing vocabulary tokens.

* **Output**:

  * **Teacher**: outputs logits for each token in the vocabulary (soft targets)
  * **Student**: produces its own logits; learning objective is to minimize the **KL divergence** from the teacher distribution
  * Final goal: train the student to mimic the teacher’s probabilistic predictions for each next-token step






<br/>  
# 요약   


이 논문은 다양한 크기의 Transformer 기반 교사·학생 모델을 사용하여, 계산량과 교사 성능에 따라 학생 성능을 예측할 수 있는 **증류 스케일링 법칙**을 제안한다. 실험 결과, **교사가 이미 존재하고 토큰이 제한된 경우 증류가 지도학습보다 효율적**이며, 반대로 교사도 훈련해야 할 경우 지도학습이 더 낫다. 예시로, C4 데이터셋의 문장 `"The capital city of France is"`에 대해 교사는 다음 토큰 `"Paris"`를 예측하며, 학생은 이 분포를 모방하는 방식으로 학습된다.

---


This paper introduces a **distillation scaling law** that predicts student model performance based on compute allocation and teacher quality, using Transformer-based models of varying sizes. Results show that **distillation is more efficient than supervised learning** when a teacher already exists and token budget is limited, but supervised training is preferable when the teacher must also be trained. For example, given the input sentence `"The capital city of France is"` from the C4 dataset, the teacher predicts the next token `"Paris"`, and the student learns to mimic this distribution through distillation.



<br/>  
# 기타  


<br/>
# refer format:     



@inproceedings{busbridge2025distillation,
  title = {Distillation Scaling Laws},
  author = {Busbridge, Dan and Shidani, Amitis and Weers, Floris and Ramapuram, Jason and Littwin, Etai and Webb, Russ},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year = {2025},
  organization = {PMLR},
  volume = {267},
  address = {Vancouver, Canada},
  url = {https://arxiv.org/abs/2502.08606}
}



Busbridge, Dan, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russ Webb. “Distillation Scaling Laws.” In Proceedings of the 42nd International Conference on Machine Learning (ICML), vol. 267. Vancouver, Canada: PMLR, 2025. https://arxiv.org/abs/2502.08606.




