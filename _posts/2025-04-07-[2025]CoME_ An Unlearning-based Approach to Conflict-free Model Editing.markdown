---
layout: post
title:  "[2025]CoME: An Unlearning-based Approach to Conflict-free Model Editing"  
date:   2025-04-07 00:02:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

LLM에서 언러닝..  
Locate-then-Edit 방식으로 특정 지식을 저장한 파라미터를 찾아낸 뒤 수정(모델 전체를 건드리는 게 아니라 오래된 지식과 관련된 상위 p%의 민감한 파라미터만 수정)  


* 근데 공통부분(언어처리부분)을 최대한 안 지우게 반영하게함    
z_i =δ_i −α(δ′_i −δ″_i)

 δᵢ: 업데이트벡터, 새지식 새부분  δ′ᵢ: 언러닝벡터, 제거해야할 부분, δ″ᵢ: 공통부분  
즉, 제거해야할 부분에서 공통부분(필요한 언어처리부분)을 빼고 제거한다는뜻  
일종의... 새로운 벡터에서빼주는 느낌?  (새로운 벡터(δᵢ)에서 예전 지식의 핵심만(δ′ᵢ - δ″ᵢ) 빼주는 구조)   



짧은 요약(Abstract) :    



대규모 언어 모델(LLMs)은 종종 사전 학습 과정에서 얻은 오래되거나 잘못된 정보를 유지하는데, 이는 모델의 신뢰성을 떨어뜨립니다. 이를 해결하기 위해 모델을 재학습하지 않고 수정할 수 있는 모델 편집 기법이 연구되어 왔지만, 새로운 지식과 기존 지식 간의 **지식 충돌** 문제가 발생할 수 있습니다.

이 논문에서는 이러한 문제를 해결하기 위해 **CoME (Conflict-free Model Editing)**라는 새로운 프레임워크를 제안합니다. CoME는 **‘언러닝(unlearning)’ 기법을 활용하여 오래된 지식을 선택적으로 제거**하고, 이를 통해 새로운 지식을 모델에 정확하게 반영할 수 있도록 합니다. 이 과정에서 기존 언어 처리 기능은 유지하면서도 불필요한 정보 간섭을 줄입니다.

GPT-J와 LLaMA-3 모델을 대상으로 한 Counterfact 및 ZsRE 데이터셋 실험을 통해, CoME가 기존 편집 방법들에 비해 **편집 정확도와 모델 신뢰성**을 모두 향상시킨다는 결과를 보여줍니다. 또한, **오래된 지식을 타겟으로 제거**하는 것이 모델 편집의 효과성을 높이는 데 중요함을 실험적으로 입증합니다.

---



> Large language models (LLMs) often retain outdated or incorrect information from pre-training, which undermines their reliability. While model editing methods have been developed to address such errors without full retraining, they frequently suffer from knowledge conflicts, where outdated information interferes with new knowledge. In this work, we propose Conflict-free Model Editing (CoME), a novel framework that enhances the accuracy of knowledge updates in LLMs by selectively removing outdated knowledge. CoME leverages unlearning to mitigate knowledge interference, allowing new information to be integrated without compromising relevant linguistic features. Through experiments on GPT-J and LLaMA-3 using Counterfact and ZsRE datasets, we demonstrate that CoME improves both editing accuracy and model reliability when applied to existing editing methods. Our results highlight that the targeted removal of outdated knowledge is crucial for enhancing model editing effectiveness and maintaining the model’s generative performance. Our code is available at https://github.com/ekgus9/COME.





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




**1. 백본 모델 (Backbone Models)**  
CoME는 두 가지 대표적인 LLM 모델을 실험에 사용했어:

- **GPT-J (6B)**: 60억 개의 파라미터를 가진 오픈소스 언어 모델.
- **LLaMA-3 (8B)**: 메타에서 발표한 최신 80억 파라미터 모델.

**2. 모델 구조 및 적용 방식**  
CoME는 기존의 모델 편집 방식인 MEMIT, PMET 위에 적용되는 프레임워크야. 핵심 구조는 다음과 같아:

- **Locate-then-Edit 방식 기반**: 특정 지식을 저장한 파라미터를 찾아낸 뒤 수정함.
- **업데이트 벡터(δi)와 언러닝 벡터(δ′i)의 결합**:  
  새로운 지식을 반영하면서 동시에 오래된 지식(δ′i)을 제거함. 이때 언어적 특징을 보존하기 위해 δi와 δ′i의 공통된 성분(δ′′i)을 추출하고, 차이를 뺀 후에 파라미터 업데이트를 수행함.
- **Top-p% 파라미터만 수정**: 모델 전체를 건드리는 게 아니라 오래된 지식과 관련된 상위 p%의 민감한 파라미터만 수정하여 불필요한 변경을 방지함.

**3. 학습 데이터 (Training Data)**  
CoME는 두 가지 공개 데이터셋으로 실험함:

- **Counterfact (10,000 샘플)**: 사실과 반대되는 정보를 제공하여 모델이 올바른 정보로 편집되었는지 평가.
- **ZsRE (Zero-shot Relation Extraction, 10,000 샘플)**: 문맥 없는 질의응답 형식으로, 편집된 지식이 정확히 반영되는지 확인 가능.

**4. 학습 세부 설정**  
- GPT-J는 3~8번 레이어를, LLaMA-3는 4~8번 레이어를 편집 대상으로 지정.
- **CoME 적용 시**, FFN 레이어 또는 MLP 가중치만 업데이트함 (PMET의 경우 FFN 위주).

---


**1. Backbone Models**  
The CoME framework is applied to two popular large language models:

- **GPT-J (6B)**: A 6-billion parameter open-source autoregressive model.
- **LLaMA-3 (8B)**: Meta's latest 8-billion parameter language model.

**2. Model Structure and Editing Mechanism**  
CoME is designed to enhance existing editing methods like MEMIT and PMET with an unlearning mechanism:

- **Based on Locate-then-Edit paradigm**: Locates knowledge-relevant parameters and updates them.
- **Update and Unlearning Vectors**:
  - Introduces δi (update for new knowledge) and δ′i (update for outdated knowledge).
  - Extracts common linguistic features as δ′′i to preserve language capacity.
  - Performs `zi - α(δ′i - δ′′i)` to remove outdated information while retaining essential features.
- **Top-p% Parameter Editing**:
  - To prevent excessive changes, only the top-p% most affected parameters (empirically set to 20%) are updated during unlearning.

**3. Training Data**  
Experiments were conducted using two datasets:

- **Counterfact (10,000 samples)**: Contains counterfactual statements to test factual knowledge editing.
- **ZsRE (10,000 samples)**: A zero-shot question-answering dataset used to evaluate the integration of new relational knowledge.

**4. Training and Implementation Details**  
- **GPT-J** edits layers {3, 4, 5, 6, 7, 8}, and **LLaMA-3** edits layers {4, 5, 6, 7, 8}.
- In PMET, only the FFN component is updated, and CoME is applied specifically to the FFN residuals.




   
 
<br/>
# Results  




**1. 테스트 데이터 (Test Datasets)**  
CoME의 성능은 다음 두 가지 공개 데이터셋을 사용해 평가되었어:

- **Counterfact (10,000개 샘플)**: 반사실적(틀린) 지식을 기반으로 편집 정확도와 충돌 여부를 평가.
- **ZsRE (10,000개 샘플)**: 문맥 없이 관계 추출을 수행하는 QA 형식의 데이터셋으로, 편집된 지식이 새로운 표현에서도 잘 작동하는지 확인.

---

**2. 평가 메트릭 (Evaluation Metrics)**  
다섯 가지 주요 메트릭을 사용했어:

- **Efficacy**: 편집된 지식이 정확히 반영되었는지 평가 (정확도 기준).
- **Generality**: 질문을 패러프레이즈했을 때에도 올바른 정보를 응답하는지 평가.
- **Locality**: 편집하지 않은 지식은 그대로 보존되는지 확인.
- **Fluency**: 생성된 문장이 유창한지를 평가.
- **Consistency**: 생성된 문장이 위키피디아 문서와 얼마나 일관된지를 평가.

※ **Score**는 Efficacy, Generality, Locality의 조화 평균으로 계산돼.

---

**3. 비교 모델 (Baselines)**  
CoME는 다음과 같은 기존 모델 편집 기법들과 비교되었어:

- **GPT-J, LLaMA-3 (Original)**: 편집을 적용하지 않은 원본 모델.
- **FT-W**: 전통적인 파인튜닝 방법 (가중치 감소 포함).
- **FT**: 전체 파라미터 파인튜닝.
- **F-Learning**: 먼저 기존 지식을 잊고 새로운 지식을 학습하는 방식.
- **MEND**: 하이퍼네트워크를 사용하는 모델 편집 방법.
- **ROME**: 단일 편집에 최적화된 방법.
- **MEMIT**: 대량 편집을 지원하는 최신 모델 편집 기법.
- **PMET**: MHSA와 FFN을 함께 최적화하는 편집 방법.

---

**4. CoME 성능 요약**  
- **Counterfact 결과**  
  - GPT-J 기준: CoME-MEMIT와 CoME-PMET 모두 기존보다 더 높은 **Efficacy (~99%)**, **Generality (~95%)**, 그리고 높은 종합 점수 **Score (86.4)** 달성.
  - LLaMA-3에서도 **CoME-PMET**이 가장 높은 점수를 기록.

- **ZsRE 결과**  
  - GPT-J와 LLaMA-3 모두에서 CoME-PMET이 가장 높은 Efficacy와 Generality 기록.  
  - Locality도 높게 유지되어, 새로운 지식을 추가하면서도 기존 지식을 잘 보존하는 걸 확인함.

---


**1. Test Datasets**  
Two standard datasets were used to evaluate CoME:

- **Counterfact (10,000 samples)**: Contains counterfactual (incorrect) knowledge for evaluating the precision and conflict resolution of model edits.
- **ZsRE (10,000 samples)**: A zero-shot question-answering dataset to test relational knowledge editing and generalization.

---

**2. Evaluation Metrics**  
The model performance was assessed using five main metrics:

- **Efficacy**: Measures how well the updated knowledge is reflected in responses.
- **Generality**: Checks if paraphrased queries also return the correct new knowledge.
- **Locality**: Ensures that unedited knowledge remains unaffected.
- **Fluency**: Evaluates the naturalness of generated text.
- **Consistency**: Assesses alignment between generated responses and reference Wikipedia content.

※ **Score** is the harmonic mean of Efficacy, Generality, and Locality.

---

**3. Baseline Methods for Comparison**  
CoME is compared against various model editing methods:

- **Original GPT-J / LLaMA-3**: Unedited versions of the models.
- **FT-W**: Full fine-tuning with weight decay.
- **FT**: Basic full-parameter fine-tuning.
- **F-Learning**: Fine-tuning with a forgetting step prior to learning new knowledge.
- **MEND**: Uses a hypernetwork for knowledge editing.
- **ROME**: Focused on single-knowledge edits.
- **MEMIT**: Supports batch editing with minimal interference.
- **PMET**: Optimizes both MHSA and FFN components in the model.

---

**4. Performance of CoME**  
- **On Counterfact**:  
  - With GPT-J, CoME-MEMIT and CoME-PMET achieve **Efficacy (~99%)**, **Generality (~95%)**, and high **Score (86.4)**.
  - On LLaMA-3, CoME-PMET also outperforms other methods.

- **On ZsRE**:  
  - CoME-PMET outperforms all baselines in **Efficacy** and **Generality** on both models.
  - It also preserves **Locality**, indicating minimal side effects on unrelated knowledge.





<br/>
# 예제  



**1. 트레이닝/편집 대상 예제 (Counterfact 기반)**  
CoME는 모델이 보유한 잘못된 지식을 고치기 위해 `(주어, 관계, 객체)` 형태의 지식을 편집하는 구조를 사용해. 예를 들어:

- **오래된/잘못된 지식**:  
  `("Motion", "manufactured by", "Microsoft")`  
- **정정된 지식**:  
  `("Motion", "manufactured by", "Apple")`

이 경우, 프롬프트는  
**“Motion, a product manufactured by”**  
처럼 주어와 관계를 포함하는 문장이 되고, 모델이 Apple을 정답으로 출력해야 해.

---

**2. 테스트 예제 (ZsRE 기반 - QA 형식)**  
ZsRE는 문맥 없이 **질문-정답 쌍**으로 구성돼 있어.

예시:

- 질문: **"What company manufactures Motion?"**  
- 정답 (편집 후 기대 응답): **"Apple"**  
- 편집 전 응답: **"Microsoft" (오답)**

---

**3. 생성 예제 (Counterfact 기반 생성)**  
생성 평가에서는 편집된 정보를 바탕으로 자연어 응답을 생성하게 해.

예시:  
- **주제(subject)**: El Correo (신문 이름)  
- **편집 전 잘못된 지식**: 언어가 Spanish  
- **편집 후 새로운 지식**: 언어가 English  
- **프롬프트**: *“The language used by people in El Correo is”*  
- **모델 응답 (CoME 적용 후)**:  
  → “The language used by people in El Correo is English. El Correo, its editors, and its subscribers understand English...”  
→ 편집 내용이 자연스럽게 반영된 응답 생성.

---



**1. Training/Edit Example (from Counterfact Dataset)**  
CoME edits model knowledge represented as a triple:  
`(subject, relation, object)`

For instance:

- **Outdated Knowledge**:  
  `("Motion", "manufactured by", "Microsoft")`
- **Corrected Knowledge**:  
  `("Motion", "manufactured by", "Apple")`

The corresponding **prompt** would be:  
→ *“Motion, a product manufactured by”*  
The expected output should be **“Apple”**, not the outdated “Microsoft.”

---

**2. Test Example (from ZsRE Dataset - QA Style)**  
ZsRE examples are in question-answer format with no additional context.

Example:

- **Question**: *“What company manufactures Motion?”*  
- **Expected Answer (after edit)**: *“Apple”*  
- **Model Response (before edit)**: *“Microsoft” (incorrect)*

---

**3. Generation Example (Counterfact - Generative Task)**  
For generation, the model is prompted to generate a full sentence incorporating the edited knowledge.

Example:

- **Subject**: El Correo (a Spanish newspaper)
- **Before Edit**: Language = Spanish  
- **After Edit**: Language = English  
- **Prompt**: *“The language used by people in El Correo is”*

→ **CoME Output**:  
“The language used by people in El Correo is English. El Correo, its editors, and its subscribers understand English...”  
This confirms that the model has successfully integrated and expressed the updated information.


<br/>  
# 요약   



CoME는 오래된 지식을 선택적으로 제거하면서 새로운 지식을 정확히 반영하는 언러닝 기반 모델 편집 프레임워크다.  
Counterfact와 ZsRE 데이터셋에서 GPT-J 및 LLaMA-3를 대상으로 실험한 결과, 기존 모델 편집 기법들보다 높은 정확도와 일관성을 보였다.  
예를 들어 El Correo의 언어를 Spanish에서 English로 수정한 후, CoME는 그 내용을 자연스럽게 반영한 응답을 생성했다.

---



CoME is an unlearning-based model editing framework that selectively removes outdated knowledge to accurately incorporate new information.  
Experiments on Counterfact and ZsRE datasets using GPT-J and LLaMA-3 show that CoME outperforms existing editing methods in accuracy and consistency.  
For example, after updating the language of "El Correo" from Spanish to English, CoME successfully generated outputs reflecting the new fact naturally.

---




<br/>  
# 기타  



**1. 표 (Tables)**  
- **Table 1 & 2**: Counterfact와 ZsRE 데이터셋에서 CoME, MEMIT, PMET 등 다양한 모델 편집 방법의 성능을 수치로 비교한 표야.  
  → 주요 평가 항목은 **Efficacy, Generality, Locality, Fluency, Consistency**이며, CoME가 대부분의 지표에서 높은 성능을 보였어.  
- **Table 3 (Ablation Study)**: CoME 구성 요소(δ′, δ′′, 제한 여부)를 하나씩 제거했을 때 성능 변화 비교.  
  → 각 구성 요소가 결과에 얼마나 기여하는지 정량적으로 보여줘.

**2. 그림 (Figures)**  
- **Figure 1**: CoME의 구조적 개요. 기존 편집 방법은 오래된 지식과 새로운 지식이 충돌하지만, CoME는 이를 언러닝으로 해결함을 시각화.
- **Figure 2**: 편집 수가 증가함에 따른 성능 변화 (scaling curve). CoME는 많은 편집 수에도 안정적인 성능을 유지함.
- **Figure 3**: 언러닝 강도(α 값)에 따른 성능 변화. α가 클수록 Efficacy와 Generality는 증가하지만, Locality는 감소함.

**3. 어펜딕스 (Appendix)**  
- **Appendix A**: Counterfact와 ZsRE에서 사용된 평가 메트릭의 정의와 수식 제공.  
- **Appendix B**: 실제 사례 기반의 **생성 결과 비교 (Case Study)**가 포함됨.  
  → 예: “El Correo” 사례에서, CoME가 편집된 내용을 자연스럽게 반영하는 예시 문장을 생성함.

---


**1. Tables**  
- **Table 1 & 2**: Present detailed comparisons of model editing performance (Efficacy, Generality, Locality, etc.) across Counterfact and ZsRE datasets.  
  → CoME consistently achieves higher scores than other baselines like MEMIT and PMET.  
- **Table 3 (Ablation Study)**: Shows the effect of removing each component (δ′, δ′′, or restriction) on performance.  
  → Demonstrates the importance of each module in achieving reliable edits.

**2. Figures**  
- **Figure 1**: Visualizes the overall architecture of CoME, contrasting it with traditional editing approaches that suffer from knowledge conflict.  
- **Figure 2**: Shows how performance scales as the number of edits increases. CoME maintains robust results even with large-scale edits.  
- **Figure 3**: Illustrates how varying the unlearning weight α affects Efficacy, Generality, and Locality.

**3. Appendix**  
- **Appendix A**: Provides mathematical definitions and formulas for evaluation metrics used in Counterfact and ZsRE.  
- **Appendix B**: Includes a **case study** showing how CoME generates more accurate and context-aware responses.  
  → Example: The “El Correo” case shows that CoME reflects the updated language (English) fluently in generated output.

---




<br/>
# refer format:     



@article{jung2025come,  
  title     = {CoME: An Unlearning-based Approach to Conflict-free Model Editing},  
  author    = {Dahyun Jung and Jaehyung Seo and Jaewook Lee and Chanjun Park and Heuiseok Lim},  
  journal   = {arXiv preprint arXiv:2502.15826},  
  year      = {2025},  
  url       = {https://arxiv.org/abs/2502.15826}  
}  




Jung, Dahyun, Jaehyung Seo, Jaewook Lee, Chanjun Park, and Heuiseok Lim. "CoME: An Unlearning-based Approach to Conflict-free Model Editing." arXiv preprint arXiv:2502.15826 (2025). https://arxiv.org/abs/2502.15826.   




