---
layout: post
title:  "[2024]DISTILLM Towards Streamlined Distillation for Large Language Models"  
date:   2025-07-13 21:47:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 

DISTILLM은 α라는 파라미터를 활용해 교사 모델 분포와 학생 모델 분포를 혼합한 새로운 분포로부터 KLD를 계산하는 **Skew KLD** 또는 **Skew Reverse KLD**를 사용, 학생 모델이 생성한 출력(Student-Generated Output, SGO)을 활용하되, **모든 반복에서 생성하는 것이 아니라 확률적으로 활용
-> 이 접근은 **더 안정적인 그래디언트**를 제공하고, **빠른 수렴과 일반화 성능 향상   



짧은 요약(Abstract) :    






이 논문은 대규모 언어 모델(LLM)의 압축을 위한 **효율적이고 효과적인 지식 증류(Knowledge Distillation, KD)** 프레임워크인 **DISTILLM**을 제안합니다. 기존 KD 방법은 표준화된 목적 함수가 없고, 학생 모델이 생성한 출력을 사용하는 과정에서 높은 계산 비용이 드는 문제가 있습니다. 이를 해결하기 위해 DISTILLM은 두 가지 주요 구성 요소를 포함합니다:

1. \*\*Skew Kullback-Leibler Divergence (Skew KLD)\*\*라는 새로운 손실 함수로, 이론적 특성을 분석하고 안정적인 학습을 가능하게 합니다.
2. **Adaptive Off-Policy 방법**으로, 학생 모델이 생성한 출력을 효율적으로 활용하면서도 계산 비용을 줄이는 전략입니다.

실험 결과, DISTILLM은 기존 최신 증류 방법에 비해 **최대 4.3배 더 빠른 속도로 학습되면서도 높은 성능**의 학생 모델을 생성함을 보여줍니다.

---



This paper presents **DISTILLM**, an efficient and effective knowledge distillation (KD) framework for compressing large auto-regressive language models. Existing KD methods for LLMs often lack a standardized objective and incur high computational costs when using student-generated outputs. DISTILLM addresses these challenges through two components:

1. A novel **skew Kullback-Leibler divergence (Skew KLD)** loss with favorable theoretical properties for stable optimization, and
2. An **adaptive off-policy approach** that efficiently leverages student-generated outputs.

Extensive experiments across instruction-following tasks demonstrate that DISTILLM produces high-performing student models with **up to 4.3× speedup** over recent KD baselines.



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




DISTILLM은 \*\*대규모 언어 모델(LLM)\*\*을 더 작고 효율적인 **학생 모델**로 압축하는 새로운 **지식 증류(Knowledge Distillation, KD)** 프레임워크로, 두 가지 핵심 구성 요소를 도입합니다:

1. **Skew KLD 손실 함수 (Skewed Kullback-Leibler Divergence)**

   * 기존의 KLD 손실 함수는 비대칭성 때문에 학생 모델이 교사 모델의 분포를 과도하게 평균화하거나, 특정 모드에 집중해 성능이 저하되는 문제가 있습니다.
   * DISTILLM은 α라는 파라미터를 활용해 교사 모델 분포와 학생 모델 분포를 혼합한 새로운 분포로부터 KLD를 계산하는 **Skew KLD** 또는 **Skew Reverse KLD**를 사용합니다.
   * 이 접근은 **더 안정적인 그래디언트**를 제공하고, **빠른 수렴과 일반화 성능 향상**을 이끌어냅니다.

2. **Adaptive Off-Policy 학습 전략**

   * 학생 모델이 생성한 출력(Student-Generated Output, SGO)을 활용하되, **모든 반복에서 생성하는 것이 아니라 확률적으로 활용**합니다.
   * 학습 초기에 SGOs는 적게 사용하고, 검증 성능에 따라 점차 비율을 증가시키는 **adaptive scheduler**를 도입합니다.
   * SGOs는 **replay buffer**에 저장되며, 이로부터 샘플을 오프폴리시 방식으로 뽑아 학습합니다.
   * 이를 통해 **SGO의 활용 효율을 높이고**, 계산 비용을 줄이면서도 성능을 유지합니다.

 **아키텍처 및 트레이닝**

* 교사 모델: 예시로 GPT-2 XL (1.5B), OPT-2.7B, OpenLLaMA2-7B 등
* 학생 모델: GPT-2 (0.1B), OPT-1.3B, OpenLLaMA2-3B 등
* 학습 데이터: Databricks-Dolly-15K, OpenWebText (보조 LM 학습 목적)
* 손실 함수: SKL, SRKL (α = 0.1로 최적화됨)
* 추가 세부 사항: Adaptive scheduler의 초기 확률은 0으로 시작하며, 검증 손실 기준으로 조절됨

---



DISTILLM is a novel knowledge distillation (KD) framework for compressing large auto-regressive language models into smaller student models. It introduces two key components:

1. **Skew KLD Loss (Skewed Kullback-Leibler Divergence):**

   * Traditional KLD suffers from asymmetry, often causing the student to either over-smooth or collapse to limited modes.
   * DISTILLM proposes **Skew KLD (SKL)** and **Skew Reverse KLD (SRKL)**, which compute the divergence between the teacher and a mixture of teacher and student distributions (using a mixing parameter α).
   * This leads to **more stable gradients**, **faster convergence**, and **better generalization** in student models.

2. **Adaptive Off-Policy Strategy:**

   * Instead of generating student outputs (SGOs) at every iteration (which is computationally expensive), DISTILLM uses them **probabilistically**, guided by an adaptive scheduler based on validation loss.
   * SGOs are stored in a **replay buffer**, and samples are drawn from this buffer during training (off-policy), which improves **sample efficiency** and reduces overhead.

 **Architecture & Training Details:**

* Teacher models: GPT-2 XL (1.5B), OPT-2.7B, OpenLLaMA2-7B
* Student models: GPT-2 (0.1B), OPT-1.3B, OpenLLaMA2-3B
* Datasets: Databricks-Dolly-15K (instruction tuning), OpenWebText (language modeling loss)
* Loss functions: SKL / SRKL with optimal α = 0.1
* Training: Initial probability of using SGO is set to 0, adjusted over time based on validation performance.



   
 
<br/>
# Results  





DISTILLM은 다양한 생성 작업에서 **경쟁 증류 기법보다 더 나은 성능과 효율성**을 보여주었습니다.

####  실험 환경 및 비교 모델

* **교사 모델(Teacher)**: GPT-2 XL (1.5B), OPT-2.7B, OpenLLaMA2-7B, T5-XL (3B), mT5-XL
* **학생 모델(Student)**: GPT-2 (0.1B), OPT-1.3B, OpenLLaMA2-3B, T5-Base (0.2B), T5-Small (0.06B), mT5-Base/Small
* **데이터셋**:

  * Instruction following: Databricks-Dolly-15K
  * Text summarization: SAMSum
  * Machine translation: IWSLT 2017 En-De
* **평가 지표**: ROUGE-L (요약 및 지시 따르기), BLEU (번역), GPT-4 feedback 점수

####  주요 결과

* **Instruction-following (지시 따르기)**:

  * DISTILLM은 ROUGE-L 점수와 GPT-4 평가 모두에서 기존 증류법(KLD, RKLD, JSD, ImitKD, GKD, MiniLLM)보다 **일관되게 높은 성능**을 기록했습니다.
  * 특히 Adaptive Off-policy 전략과 Skew 손실 함수를 결합했을 때 가장 뛰어났습니다.
* **Training Time**:

  * 기존 최신 기법보다 **2.5\~4.3배 빠른 학습 속도**를 보였습니다.
  * 예: OpenLLaMA2-3B distillation 기준, DISTILLM은 1.6× 시간만 소요, 다른 기법은 3\~7× 필요.
* **Text Summarization (요약)**:

  * SAMSum 데이터셋에서 DISTILLM은 T5-Base 및 T5-Small에 대해 ROUGE-L 점수 기준 **최고 성능 달성**.
* **Machine Translation (번역)**:

  * IWSLT 2017 En-De 기준, BLEU 점수에서도 DISTILLM이 GKD, ImitKD 등 기존 방법보다 더 나은 결과.

####  추가 분석

* DISTILLM은 **사전 파인튜닝 없이도** 좋은 성능을 유지함.
* 기존 방법(GKD, ImitKD 등)은 오프폴리시 방식 적용 시 성능 하락, 반면 DISTILLM은 **오프폴리시에서도 성능 유지**.

---



DISTILLM consistently outperforms existing distillation methods in both **performance and training efficiency** across various generative tasks.

####  Experimental Setup & Baselines

* **Teacher Models**: GPT-2 XL (1.5B), OPT-2.7B, OpenLLaMA2-7B, T5-XL (3B), mT5-XL
* **Student Models**: GPT-2 (0.1B), OPT-1.3B, OpenLLaMA2-3B, T5-Base (0.2B), T5-Small (0.06B), mT5-Base/Small
* **Datasets**:

  * Instruction following: Databricks-Dolly-15K
  * Text summarization: SAMSum
  * Machine translation: IWSLT 2017 En-De
* **Metrics**: ROUGE-L, BLEU, GPT-4 Feedback

####  Key Results

* **Instruction-Following**:

  * DISTILLM significantly surpasses baselines (KLD, RKLD, JSD, ImitKD, GKD, MiniLLM) in ROUGE-L and GPT-4 evaluation.
  * Best performance achieved with the combination of Skew loss and adaptive off-policy generation.
* **Training Efficiency**:

  * DISTILLM achieves **2.5–4.3× faster training speed** than recent SGO-based distillation methods.
  * For example, distilling OpenLLaMA2-3B takes only **1.6× time** with DISTILLM compared to **3–7×** with others.
* **Text Summarization**:

  * On SAMSum, DISTILLM yields the highest ROUGE-L scores for both T5-Base and T5-Small students.
* **Machine Translation**:

  * DISTILLM achieves higher BLEU scores on IWSLT 2017 En-De compared to GKD and ImitKD.

####  Additional Insights

* DISTILLM performs well even **without pre-fine-tuning** student models.
* While baselines degrade in performance when using off-policy, DISTILLM maintains **robust performance with superior efficiency**.




<br/>
# 예제  




1. **Instruction-Following (지시 따르기)**

   * **데이터셋**: `Databricks-Dolly-15K`
   * **형식 예시**:

     * 입력 (Input):

       ```
       Who wrote Picture of Dorian Grey in 1891?
       ```
     * 학생 모델 출력 (Student Output):

       ```
       Christopher Columbus.
       ```
     * 정답이 아닌 잘못된 출력이 생성될 수 있음 (SGO의 한계 사례).

2. **Text Summarization (요약)**

   * **데이터셋**: `SAMSum`
   * **형식 예시**:

     * 입력 (Input):

       ```
       A: Hey, are you coming to the party tonight?  
       B: Not sure yet, got work to finish.  
       A: It'll be fun!  
       ```
     * 정답 요약 (Reference Summary):

       ```
       A invited B to a party, but B may not come due to work.
       ```

3. **Machine Translation (기계 번역)**

   * **데이터셋**: `IWSLT 2017 En-De` (영어 → 독일어)
   * **형식 예시**:

     * 입력 문장 (English Input):

       ```
       The weather is nice today.
       ```
     * 번역 정답 (German Reference):

       ```
       Das Wetter ist heute schön.
       ```

####  SGO (Student-Generated Output) 관련 사례:

* 학생 모델이 학습 중 생성한 출력이 교사 모델의 분포와 다를 경우, 교사가 올바른 피드백을 주지 못해 학습을 저해할 수 있음.
* 예:

  * 입력: "What is the Cassandra database?"
  * SGO: "Cassandra is a distributed system developed and maintained by software engineers."
  * 이는 사실과 일부 맞지만, 교사 모델 기준으로는 낮은 확률을 부여할 수 있어 **노이즈 피드백** 발생.

---

####  Example Tasks and Datasets:

1. **Instruction-Following Task**

   * **Dataset**: `Databricks-Dolly-15K`
   * **Example**:

     * Input Prompt:

       ```
       Who wrote Picture of Dorian Grey in 1891?
       ```
     * Student Output (SGO):

       ```
       Christopher Columbus.
       ```
     * This is a failure case showing the limitations of student-generated outputs.

2. **Text Summarization**

   * **Dataset**: `SAMSum`
   * **Example**:

     * Input Conversation:

       ```
       A: Hey, are you coming to the party tonight?  
       B: Not sure yet, got work to finish.  
       A: It'll be fun!  
       ```
     * Reference Summary:

       ```
       A invited B to a party, but B may not come due to work.
       ```

3. **Machine Translation**

   * **Dataset**: `IWSLT 2017 En-De` (English to German)
   * **Example**:

     * Input Sentence:

       ```
       The weather is nice today.
       ```
     * German Reference Translation:

       ```
       Das Wetter ist heute schön.
       ```

####  SGO Case:

* A student model might generate a slightly inaccurate or over-generalized output during training:

  * Input:

    ```
    What is the Cassandra database?
    ```
  * Student Output:

    ```
    Cassandra is a distributed system developed and maintained by software engineers.
    ```
  * Although partially correct, the teacher might assign poor loss due to **distribution mismatch**, leading to **noisy feedback** during distillation.


<br/>  
# 요약   




DISTILLM은 Skew KLD 손실 함수와 적응형 오프폴리시 학습 전략을 통해 대규모 언어 모델의 효과적인 증류를 가능하게 한다.
이 방법은 기존 증류 기법들보다 최대 4.3배 빠른 학습 속도와 더 높은 성능을 다양한 생성 태스크에서 달성하였다.
예를 들어, Dolly-15K에서 "Who wrote Picture of Dorian Grey?"라는 질문에 대해 잘못된 학생 출력("Christopher Columbus")이 발생할 수 있으며, 이를 효과적으로 처리하기 위한 학습 전략이 포함되어 있다.

---


DISTILLM enables effective distillation of large language models by introducing a Skew KLD loss and an adaptive off-policy learning strategy.
It achieves up to 4.3× faster training and superior performance compared to prior distillation methods across various generative tasks.
For instance, on Dolly-15K, a prompt like "Who wrote Picture of Dorian Grey?" may yield incorrect student outputs ("Christopher Columbus"), which DISTILLM handles with improved training dynamics.



<br/>  
# 기타  




####  주요 테이블 및 인사이트:

1. **Table 1** – 다양한 손실 함수(SKLD, SRKLD, KLD, JSD 등)의 성능 비교

   * DISTILLM에서 제안한 **SRKL 손실 함수가 모든 태스크에서 가장 높은 ROUGE-L 점수**를 기록함.
   * → **Skew 손실이 기존 손실보다 일반화에 효과적**임을 실증.

2. **Table 2** – 오프폴리시 전략의 효과

   * DISTILLM의 **Adaptive + Off-policy** 전략이 다른 접근(On-policy, Mixed)보다 **더 높은 성능과 일관성**을 보임.
   * → 학습 효율을 극대화하면서도 성능 저하 없이 SGO를 활용할 수 있음.

3. **Table 4 & 5** – 다른 기존 증류 기법에 오프폴리시를 적용한 경우 성능 저하 발생 vs. DISTILLM은 안정 유지

   * → DISTILLM은 **자체 구성 요소 간의 시너지**가 있음을 입증.

####  주요 피규어:

1. **Figure 1** – 잘못된 학생 출력(SGO)에 대해 교사 모델이 부정확한 피드백을 주는 사례

   * → SGOs만 활용하는 on-policy 방식의 한계를 보여줌.

2. **Figure 2 & 7** – SGOs의 생성 길이/빈도에 따른 계산 비용

   * → **DISTILLM의 adaptive 방식이 최대 3\~4배 속도 향상**을 달성함을 수치로 표현.

3. **Figure 3 & 8** – SKL 손실의 그래디언트 안정성과 skew 계수 α의 선택

   * → α = 0.1일 때 **최적의 안정성과 일반화 성능**을 보여줌.

####  어펜딕스:

* **Appendix B.1, B.2** – SKL 손실 함수의 수학적 안정성과 L2 norm 경계에 대한 이론적 증명 제공
* **Appendix E.1–E.4** – 다양한 세부 ablation 실험: α 값 변화, SGO 사용 비율, 미세 조정 여부 등

  * → DISTILLM은 **세밀한 튜닝 없이도 강건한 성능 유지** 가능

####  결론적 인사이트:

* DISTILLM은 단순한 구성 두 가지(SKL, Adaptive off-policy)를 도입했지만,
  이들의 **결합 효과가 기존 모든 증류 기법을 능가하는 성능과 효율을 만들어낸다**.
* 이 프레임워크는 **사전 튜닝 없이도 적용 가능하고**, SGO의 잡음을 효과적으로 제어하며
  **실제 LLM 증류의 표준이 될 수 있는 가능성**을 보여준다.

---

####  Key Tables & Insights:

1. **Table 1** – Comparison of loss functions (KLD, JSD, SKL, SRKL)

   * DISTILLM's **SRKL loss consistently achieves the highest ROUGE-L scores** across all tasks.
   * → Shows **superior generalization** of skewed losses compared to traditional objectives.

2. **Table 2** – Effectiveness of off-policy strategy

   * **Adaptive + Off-policy (DISTILLM)** significantly outperforms on-policy and mixed approaches in performance and stability.
   * → Demonstrates **optimal utilization of student-generated outputs (SGO)** without added training cost.

3. **Table 4 & 5** – Applying off-policy to other distillation methods reduces their performance, unlike DISTILLM

   * → Proves the **strong synergy of DISTILLM’s two components**.

####  Key Figures:

1. **Figure 1** – Example of incorrect SGO misguiding the teacher

   * → Highlights limitations of relying solely on SGOs in on-policy methods.

2. **Figures 2 & 7** – Training time analysis with respect to SGO frequency and response length

   * → DISTILLM achieves **up to 3–4× training speedup** with better compute efficiency.

3. **Figures 3 & 8** – Gradient stability and skew α value tuning for SKL

   * → α = 0.1 yields the **best trade-off between convergence stability and generalization**.

####  Appendix Insights:

* **Appendix B.1, B.2** – Theoretical proofs on **gradient stability and L2 error bounds** of SKL
* **Appendix E.1–E.4** – Comprehensive ablation studies: skew α tuning, SGO ratio, student initialization

  * → DISTILLM remains **robust without extensive hyperparameter tuning**.

####  Overall Takeaways:

* Although DISTILLM only introduces **two simple components (SKL, adaptive off-policy)**,
  their **combination yields superior performance and efficiency** over all existing KD methods.
* It works **without prior fine-tuning**, effectively controls noisy SGOs, and
  demonstrates **a strong candidate for standard KD in LLM compression**.




<br/>
# refer format:     



@inproceedings{ko2024distillm,
  title     = {DISTILLM: Towards Streamlined Distillation for Large Language Models},
  author    = {Jongwoo Ko and Sungnyun Kim and Tianyi Chen and Se-Young Yun},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  volume    = {235},
  publisher = {PMLR},
  url       = {https://arxiv.org/abs/2402.03898},
  note      = {arXiv preprint arXiv:2402.03898},
}



Ko, Jongwoo, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. 2024. “DISTILLM: Towards Streamlined Distillation for Large Language Models.” Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR Vol. 235. https://arxiv.org/abs/2402.03898.




