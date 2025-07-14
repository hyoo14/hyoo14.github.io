---
layout: post
title:  "[2025]DISTILLM-2: A Contrastive Approach Boosts the Distillation of LLMs"  
date:   2025-07-13 21:48:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 


DISTILLM-2는 기존의 단일 손실 함수 대신, 교사 모델 출력(Teacher Generated Output, TGO)에는 **skewed KL(SKL)** 손실을, 학생 모델 출력(Student Generated Output, SGO)에는 **skewed reverse KL(SRKL)** 손실을 적용  
->SKL은 교사 응답에 맞춰 학생 분포를 "끌어올리는(pulling-up)" 역할을 하며, SRKL은 품질이 낮은 학생 응답을 억제하여 "밀어내는(pushing-down)" 효과. -> 이처럼 응답 종류에 따라 서로 다른 손실 함수를 적용해 대조 학습 방식으로 정렬 효과를 극대화

데이터 구성 전략 (Data Curation Strategy)으로 학습 데이터는 오프라인으로 수집된 교사 응답(TGO)과 학생 응답(SGO)을 모두 사용하며, 각 미니배치마다 이 두 종류의 응답을 한 쌍으로 구성  
-> 특히 speculative decoding을 통해 생성된 고품질 응답이 항상 최적이 아니며, DISTILLM-2에서는 교사 응답을 그대로 사용하는 것이 SKL 측에서는 가장 효과적이고, 학생 응답을 그대로 사용하는 것이 SRKL 측에서는 효과적임을 실험적으로 확인  


리큘럼 기반 학습 (Curriculum-Based Adaptive Learning)**  
->   학습이 진행됨에 따라 샘플 난이도에 따라 α 값(손실 함수에서 교사와 학생 분포를 혼합하는 비율)을 동적으로 조절  
->쉬운 샘플에는 α를 작게 (교사보다 학생에 더 가깝게), 어려운 샘플에는 α를 크게 (교사 분포에 더 가까이) 설정,   
또한 SRKL 손실 항의 비중 β는 학습 진행에 따라 점진적으로 증가시켜, 학생의 품질이 낮은 응답을 억제하는 효과를 점점 강화.  




짧은 요약(Abstract) :    


이 논문에서는 대형 언어 모델(LLM)의 지식 증류(distillation) 과정에서 일반적으로 교사 모델과 학생 모델에 동일한 손실 함수를 적용하는 기존 방식의 한계를 지적합니다. 이러한 방식은 손실 함수와 데이터 종류 간의 시너지를 고려하지 않아 학생 모델의 성능 향상이 제한됩니다. 이를 해결하기 위해, 저자들은 DISTILLM-2라는 대조적(contrastive) 접근 방식을 제안합니다. 이 방법은 교사 모델의 출력을 강화하고 동시에 학생 모델의 출력을 약화시키는 대조적 손실 함수를 적용하여 두 모델 간의 정렬을 극대화합니다. 다양한 실험을 통해, DISTILLM-2는 명령어 수행(instruction following), 코드 생성 등 여러 과제에서 뛰어난 성능을 보였으며, 선호도 정렬(preference alignment)과 멀티모달 모델 확장 등 다양한 응용에도 효과적임을 보여주었습니다. 이 결과는 대조 학습 접근이 LLM 증류의 효율성과 효과를 향상시킬 수 있음을 시사합니다.


Despite the success of distillation in large language models (LLMs), most prior work applies identical loss functions to both teacher- and student-generated data. These strategies overlook the synergy between loss formulations and data types, leading to suboptimal performance in student models. To address this, we propose DISTILLM-2, a contrastive approach that simultaneously increases the likelihood of teacher responses and decreases that of student responses by leveraging this synergy. Our extensive experiments show that DISTILLM-2 not only builds high-performing student models across a wide range of tasks, including instruction-following and code generation, but also supports diverse applications such as preference alignment and vision-language extensions. These findings highlight the potential of a contrastive approach to enhance the efficacy of LLM distillation by effectively aligning teacher and student models across varied data types.



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





1. **대조 손실 함수 (Contrastive Loss Function)**
   DISTILLM-2는 기존의 단일 손실 함수 대신, 교사 모델 출력(Teacher Generated Output, TGO)에는 **skewed KL(SKL)** 손실을, 학생 모델 출력(Student Generated Output, SGO)에는 **skewed reverse KL(SRKL)** 손실을 적용합니다.

   * SKL은 교사 응답에 맞춰 학생 분포를 "끌어올리는(pulling-up)" 역할을 하며,
   * SRKL은 품질이 낮은 학생 응답을 억제하여 "밀어내는(pushing-down)" 효과를 줍니다.
     이처럼 응답 종류에 따라 서로 다른 손실 함수를 적용해 대조 학습 방식으로 정렬 효과를 극대화합니다.

2. **데이터 구성 전략 (Data Curation Strategy)**
   학습 데이터는 오프라인으로 수집된 교사 응답(TGO)과 학생 응답(SGO)을 모두 사용하며, 각 미니배치마다 이 두 종류의 응답을 한 쌍으로 구성합니다.
   특히 speculative decoding을 통해 생성된 고품질 응답이 항상 최적이 아니며, DISTILLM-2에서는 교사 응답을 그대로 사용하는 것이 SKL 측에서는 가장 효과적이고, 학생 응답을 그대로 사용하는 것이 SRKL 측에서는 효과적임을 실험적으로 확인했습니다.

3. **커리큘럼 기반 학습 (Curriculum-Based Adaptive Learning)**
   학습이 진행됨에 따라 샘플 난이도에 따라 α 값(손실 함수에서 교사와 학생 분포를 혼합하는 비율)을 동적으로 조절합니다.

   * 쉬운 샘플에는 α를 작게 (교사보다 학생에 더 가깝게)
   * 어려운 샘플에는 α를 크게 (교사 분포에 더 가까이) 설정합니다.
     또한 SRKL 손실 항의 비중 β는 학습 진행에 따라 점진적으로 증가시켜, 학생의 품질이 낮은 응답을 억제하는 효과를 점점 강화합니다.

---



1. **Contrastive Loss Function**
   DISTILLM-2 introduces a contrastive loss that applies different loss functions to teacher- and student-generated outputs. Specifically:

   * **SKL (Skewed KL divergence)** is applied to teacher outputs (TGOs), enhancing alignment by "pulling up" the student’s likelihood where the teacher’s probability is high.
   * **SRKL (Skewed Reverse KL)** is applied to student outputs (SGOs), pushing down the likelihood where the teacher assigns low probability.
     This contrastive combination—called **CALD (Contrastive Approach for LLM Distillation)**—effectively aligns the distributions.

2. **Data Curation Strategy**
   Training data consists of paired outputs from both the teacher and student models for each prompt. Instead of relying on speculative decoding or higher-quality outputs from other models, DISTILLM-2 found that directly using the teacher outputs for SKL and student outputs for SRKL yields optimal performance. This respects the "pull-up" and "push-down" nature of the loss terms.

3. **Curriculum-Based Adaptive Learning**
   The α coefficient, which controls the mixing between teacher and student distributions, is dynamically updated based on sample difficulty:

   * Smaller α for easy samples (closer p and q),
   * Larger α for hard samples (larger gap between p and q).
     Additionally, the β coefficient for SRKL is gradually increased over training epochs, emphasizing suppression of undesirable student outputs over time.



   
 
<br/>
# Results  



 DISTILLM-2의 결과는 다양한 테스크(명령어 수행, 수학 추론, 코드 생성 등)와 다양한 학생-교사 모델 쌍에 대해 폭넓게 평가되었고, 기존의 경쟁 지식 증류(KD) 방법들과 비교해 **우수한 성능**을 보였습니다.  


#### 1. **평가 모델 및 테스크**

* **교사(Teacher)** 모델: Qwen2-7B, Mistral-7B, Gemma-2-9B 등
* **학생(Student)** 모델: Qwen2-1.5B, Danube2-1.8B, Gemma-2-2B 등
* **테스트 데이터셋 및 과제(Task)**:

  * **Instruction following**: AlpacaEval, Evol-Instruct, UltraFeedback
  * **수학 추론 (Mathematical reasoning)**: GSM8K, MATH
  * **코드 생성 (Code generation)**: HumanEval, MBPP
  * **비전-언어 확장 (Vision-Language)**: OK-VQA, TextVQA
  * **선호도 정렬 (Preference alignment)**: DPO 사전 모델로 사용

#### 2. **비교 방법 (경쟁 기법)**

* GKD (on-policy KD), DistiLLM (이전 버전), MiniLLM, ImitKD, Speculative KD 등 다양한 최신 증류 방법들과 비교.

#### 3. **성능 요약**

* 전반적으로 **DISTILLM-2는 모든 테스크에서 기존 기법보다 높은 성능**을 보임.
* 예를 들어, **명령어 수행 테스크**에서:

  * Qwen2-1.5B 학생 기준, 기존 DistiLLM 대비 +2.34% 향상.
  * Gemma-2-2B 학생 기준, GKD 대비 +4.53% 향상.
* \*\*수학 테스크(GSM8K, MATH)\*\*에서도 평균 정확도 향상.
* **코드 생성**에서도 HumanEval 및 MBPP에서 높은 pass\@1 점수 달성.
* \*\*VQA(OK-VQA, TextVQA)\*\*와 같은 **멀티모달 과제**에서도 우수한 결과.

#### 4. **추가적 성능 향상 요소**

* 대조 손실 적용, 커리큘럼 α 스케줄링, SRKL 가중치 증가(β) 적용 시 성능이 점진적으로 향상됨 (Table 5).

---


#### 1. **Evaluation Setup**

* **Teacher models**: Qwen2-7B, Mistral-7B, Gemma-2-9B
* **Student models**: Qwen2-1.5B, Danube2-1.8B, Gemma-2-2B
* **Tasks and datasets**:

  * **Instruction Following**: AlpacaEval, Evol-Instruct, UltraFeedback
  * **Mathematical Reasoning**: GSM8K, MATH
  * **Code Generation**: HumanEval, MBPP
  * **Vision-Language Extension**: OK-VQA, TextVQA
  * **Preference Alignment**: As initialization for DPO-style fine-tuning

#### 2. **Baselines for Comparison**

* GKD, DistiLLM (v1), MiniLLM, ImitKD, Speculative KD, among others.

#### 3. **Performance Summary**

* DISTILLM-2 consistently outperformed all baselines across tasks.
* For **instruction-following**:

  * Qwen2-1.5B student achieved +2.34% higher win rate than DistiLLM.
  * Gemma-2-2B improved by +4.53% over GKD.
* For **mathematical reasoning (GSM8K, MATH)**, DISTILLM-2 improved pass\@1 scores.
* For **code generation**, it achieved the highest performance on HumanEval and MBPP.
* For **vision-language tasks**, DISTILLM-2 outperformed baselines on OK-VQA and TextVQA.

#### 4. **Component Ablation**

* Table 5 shows that applying contrastive loss, curriculum-based α updates, and increasing SRKL weight (β) leads to progressive performance improvements.





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

#### Dataset Example

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



DISTILLM-2는 교사 응답에는 SKL, 학생 응답에는 SRKL 손실을 적용하는 대조 학습 방식을 통해 LLM 증류 성능을 높였습니다. 명령어 수행, 수학 추론, 코드 생성, VQA 등 다양한 테스크에서 기존 증류 기법(DistiLLM, GKD 등)보다 일관되게 더 높은 정확도와 수렴 속도를 달성했습니다. 예를 들어 "문자열이 palindrome인지 확인하는 함수 작성" 프롬프트에 대해, 간단한 역순 비교 응답을 따르는 교사 출력을 학습 기준으로 삼고, 복잡하거나 부정확한 학생 출력을 억제합니다.

---


DISTILLM-2 improves LLM distillation by applying a contrastive learning scheme—using SKL loss for teacher outputs and SRKL loss for student outputs. It consistently outperforms previous distillation methods (e.g., DistiLLM, GKD) across tasks like instruction following, math reasoning, code generation, and vision-language QA. For instance, when prompted to write a palindrome-checking function, it aligns with concise teacher outputs (e.g., `return s == s[::-1]`) while suppressing less optimal student variants.



<br/>  
# 기타  




#### 1. **Figure 1: 손실 함수별 효과 시각화**

* (a) Toy 데이터에서는 KL은 중심부(고확률 영역)를 끌어올리는 효과, RKL은 꼬리(저확률 영역)를 억제하는 효과를 가짐.
* (b) 실제 모델 실험에서는 CALD(SKL+SRKL)의 손실 조합이 KL/RKL 단독보다 빠른 수렴과 낮은 NLL을 보임.
* **인사이트**: 서로 다른 손실을 응답 유형에 맞게 적용하면 정렬 효과가 극대화됨.

#### 2. **Table 2–4: 다양한 테스크에서의 성능 비교**

* DISTILLM-2는 Instruction-following, MATH, HumanEval, MBPP 등 모든 테스크에서 기존 증류 방법(DistiLLM, GKD 등)보다 높은 정확도(pass\@1, WR 등)를 보임.
* 특히 Gemma-2-2B 기준 평균 +4.53% 향상됨.
* **결론**: 제안된 대조 손실 및 커리큘럼 기반 학습 방식이 범용적인 성능 향상으로 이어짐.

#### 3. **Table 5: 구성 요소별 성능 향상 분석 (Ablation)**

* 대조 손실 도입, β 증가, 커리큘럼 α 조절을 하나씩 적용할수록 성능이 단계적으로 상승.
* **인사이트**: DISTILLM-2의 각 설계 요소가 실제로 기여함을 실증함.

#### 4. **Figure 2: 데이터 구성 전략 실험**

* Speculative decoding이나 더 강력한 LLM의 응답을 사용할 때 성능이 반드시 향상되는 것은 아님.
* **결론**: 손실 함수의 설계에 따라 오히려 단순한 교사/학생 응답을 쓰는 것이 가장 효과적일 수 있음.

#### 5. **Table 6–10: 확장성 평가 (멀티모달, 선호 정렬, 양자화, 추론속도 등)**

* DISTILLM-2는 VQA, 양자화 모델 성능 회복, speculative decoding 속도 향상 등 다양한 응용에서 기존보다 우수한 성능을 보임.
* **결론**: 이 접근법은 단순한 텍스트 모델 증류에 그치지 않고 다양한 도메인에 일반화 가능함.

---



#### 1. **Figure 1: Visualization of Loss Dynamics**

* (a) On toy data, KL boosts high-probability regions ("pull-up"), while RKL suppresses low-probability regions ("push-down").
* (b) In real model training, CALD (SKL+SRKL) converges faster and achieves lower NLL than standalone KL/RKL.
* **Insight**: Using asymmetric losses tailored to response types improves model alignment.

#### 2. **Tables 2–4: Task-wise Performance Comparison**

* DISTILLM-2 outperforms all prior distillation methods across instruction-following, math, and code generation benchmarks.
* For instance, Gemma-2-2B shows an average gain of +4.53% over GKD.
* **Conclusion**: The contrastive and curriculum-based techniques contribute to broad performance improvements.

#### 3. **Table 5: Component-wise Ablation Study**

* Each added component (contrastive loss, increasing β, curriculum α) leads to step-wise performance gains.
* **Insight**: The improvement is cumulative and attributable to each design choice in DISTILLM-2.

#### 4. **Figure 2: Data Curation Trade-offs**

* Replacing teacher/student outputs with speculative decoding or higher-quality responses doesn't always improve performance.
* **Conclusion**: Loss design matters more than raw response quality—teacher/student outputs are optimally paired with SKL/SRKL.

#### 5. **Tables 6–10: Broader Application Scenarios**

* DISTILLM-2 enhances performance not only in LLM distillation but also in VQA, quantized models, and speculative decoding speed.
* **Conclusion**: The framework generalizes well across modalities and deployment settings.




<br/>
# refer format:     



@article{ko2025distillm2,
  title={DISTILLM-2: A Contrastive Approach Boosts the Distillation of LLMs},
  author={Ko, Jongwoo and Chen, Tianyi and Kim, Sungnyun and Ding, Tianyu and Liang, Luming and Zharkov, Ilya and Yun, Se-Young},
  journal={arXiv preprint arXiv:2503.07067},
  year={2025},
  url={https://arxiv.org/abs/2503.07067v2}
}



Ko, Jongwoo, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and Se-Young Yun. "DISTILLM-2: A Contrastive Approach Boosts the Distillation of LLMs." arXiv preprint arXiv:2503.07067 (2025). https://arxiv.org/abs/2503.07067v2.






