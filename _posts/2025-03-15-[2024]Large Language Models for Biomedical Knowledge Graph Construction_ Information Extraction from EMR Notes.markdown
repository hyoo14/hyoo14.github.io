---
layout: post
title:  "[2024]Large Language Models for Biomedical Knowledge Graph Construction: Information Extraction from EMR Notes"  
date:   2025-03-15 10:26:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


EMR to KG(Knowledge Graph) with LLM(LLM으로 채우나보네 KG를, 그리고 수동어노테이션과 비교평가)    


짧은 요약(Abstract) :    



지식 그래프(KG) 자동 구축은 의학 분야에서 중요한 연구 주제로, 신약 개발과 임상 시험 설계와 같은 다양한 응용 사례에서 활용된다. 이러한 응용은 의학 및 생물학적 개체 간의 상호작용을 정확하게 식별하는 데 의존한다. 본 연구에서는 대형 언어 모델(LLM)을 활용하여 전자 의료 기록(EMR) 노트에서 지식 그래프를 구축하는 엔드 투 엔드 머신러닝 솔루션을 제안한다. 이 지식 그래프에는 질병, 요인, 치료법 및 해당 질병을 경험하는 동안 환자에게 나타나는 증상이 포함된다. 의료 응용에서 높은 성능이 필수적인 만큼, 우리는 다양한 아키텍처를 가진 12개의 LLM을 종합적으로 평가하고 성능 및 안전 속성을 분석했다. 이를 위해 Macula and Retina Institute에서 제공한 데이터셋을 수동으로 주석하여 정밀도와 재현율을 평가했으며, 구조화된 출력을 생성하는 능력이나 환각 현상 발생 가능성과 같은 질적 성능도 함께 분석했다. 연구 결과, 인코더-디코더 모델과 비교하여 디코더 전용 LLM이 추가적인 연구가 필요하다는 점이 확인되었으며, 이에 따라 이러한 모델을 효과적으로 활용할 수 있도록 안내형 프롬프트 설계를 제공하였다. 본 방법론의 적용 사례로 노화 관련 황반변성(AMD)을 대상으로 한 실험을 수행했다.


The automatic construction of knowledge graphs (KGs) is an important research area in medicine, with far-reaching applications spanning drug discovery and clinical trial design. These applications hinge on the accurate identification of interactions among medical and biological entities. In this study, we propose an end-to-end machine learning solution based on large language models (LLMs) that utilize electronic medical record notes to construct KGs. The entities used in the KG construction process are diseases, factors, treatments, as well as manifestations that coexist with the patient while experiencing the disease. Given the critical need for high-quality performance in medical applications, we embark on a comprehensive assessment of 12 LLMs of various architectures, evaluating their performance and safety attributes. To gauge the quantitative efficacy of our approach by assessing both precision and recall, we manually annotate a dataset provided by the Macula and Retina Institute. We also assess the qualitative performance of LLMs, such as the ability to generate structured outputs or the tendency to hallucinate. The results illustrate that in contrast to encoder-only and encoder-decoder, decoder-only LLMs require further investigation. Additionally, we provide guided prompt design to utilize such LLMs. The application of the proposed methodology is demonstrated on age-related macular degeneration.



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




본 연구에서는 대형 언어 모델(LLM)을 활용하여 전자 의료 기록(EMR) 노트에서 지식 그래프(KG)를 구축하는 엔드 투 엔드 방법을 제안한다. 제안된 방법론은 크게 데이터 처리, 모델 선택, 관계 추출 및 후처리 단계를 포함한다.

#### **1. 데이터 처리 및 전처리**
- **데이터셋:** 연구에서 사용된 데이터는 *Macula and Retina Institute*에서 제공한 전자 의료 기록(EMR)이며, 약 10,000명의 망막 관련 질환 환자의 데이터를 포함한다. 해당 데이터는 2008년부터 2023년까지 수집된 임상 노트 약 36만 개를 포함한다.
- **전처리 과정:** 임상 노트에는 반복적이고 불필요한 정보가 포함될 가능성이 높기 때문에, 문장 임베딩 기법을 활용하여 유사한 노트를 제거하였다. Sentence T5 XXL 모델을 활용해 문장 임베딩을 생성한 후, 코사인 유사도를 계산하여 일정 임계값을 초과하는 중복 노트를 제거하였다. 또한, 5단어 미만의 짧은 노트는 분석에서 제외하였다.

#### **2. 모델 선택 및 평가**
본 연구에서는 다양한 아키텍처를 갖춘 12개의 LLM을 실험 대상으로 선정하였다. 모델의 아키텍처는 다음 세 가지로 분류된다:
1. **인코더 전용 모델 (Encoder-only)**
   - BioBERT-SQuAD-v2 (110M parameters)
   - BERT-SQuAD-v2 (110M parameters)
   - RoBERTa-SQuAD-v2 (125M parameters)
   
2. **디코더 전용 모델 (Decoder-only)**
   - BioGPT (349M parameters)
   - OPT-30B (180B parameters)
   - OPT-IML-MAX (30B parameters)
   - Llama 2 (70B parameters)
   - Vicuna-33B (33B parameters)
   - BLOOM (176B parameters)
   - WizardLM (70B parameters)

3. **인코더-디코더 모델 (Encoder-decoder)**
   - FLAN-T5-XXL (11B parameters)
   - FLAN-UL2 (20B parameters)

- **모델 평가 지표:** 모델의 성능 평가는 정밀도(Precision)와 재현율(Recall)을 기준으로 수행되었으며, 질병과 관련된 관계(치료법, 원인, 공존하는 질환 등)를 올바르게 추출하는 능력을 측정하였다.

#### **3. 관계 추출 (Relation Extraction)**
- **LLM 기반 관계 추출:** LLM을 활용하여 질병(disease), 치료법(treatment), 요인(factor), 공존 질환(coexists_with)의 관계를 추출하였다. 이를 위해 Open-Book QA와 In-Context Learning 방식을 활용하였다.
- **프롬프트 설계 (Prompt Design):** 질병과 관련된 정보를 추출하기 위해, 세 가지 주요 질문 유형을 정의하였다.
  1. 치료법 관련 질문 (예: "*What treats %s?*")
  2. 원인 및 요인 관련 질문 (예: "*What causes %s?*")
  3. 공존 질환 관련 질문 (예: "*What coexists with %s?*")
  
- **후처리 (Postprocessing):** 모델이 생성한 답변 중 신뢰도가 낮은 항목을 제거하고, 동일한 의미를 가진 여러 표현을 그룹화하여 최종적인 관계 집합을 구성하였다.

#### **4. 실험 및 결과**
- **질병 예시:** 연구에서는 *노화 관련 황반변성(AMD, Age-related Macular Degeneration)*을 대상으로 KG를 구축하였다.
- **LLM 성능 비교:** FLAN-UL2 모델이 정밀도와 재현율에서 가장 우수한 성능을 보였으며, 특히 안내형(prompt-guided) 기법을 적용했을 때 정확도가 향상되었다.
- **디코더 전용 모델의 문제점:** 일부 디코더 전용 모델(BioGPT, OPT 등)은 구조화된 출력을 생성하지 못하거나, 환각(hallucination) 현상이 발생하여 정확한 관계 추출이 어려웠다.

---


This study proposes an end-to-end approach that leverages large language models (LLMs) to construct knowledge graphs (KGs) from electronic medical record (EMR) notes. The methodology consists of data preprocessing, model selection, relation extraction, and postprocessing.

#### **1. Data Processing and Preprocessing**
- **Dataset:** The study utilizes EMR data provided by the *Macula and Retina Institute*, comprising records of approximately 10,000 patients with retina-related diseases. The dataset includes around 360,000 clinical notes collected between 2008 and 2023.
- **Preprocessing:** Since clinical notes often contain redundant information, we employed sentence embeddings generated using the Sentence T5 XXL model to identify and remove duplicate notes based on cosine similarity. Notes with fewer than five words were excluded from the analysis.

#### **2. Model Selection and Evaluation**
A total of 12 LLMs with different architectures were evaluated:

1. **Encoder-only Models**
   - BioBERT-SQuAD-v2 (110M parameters)
   - BERT-SQuAD-v2 (110M parameters)
   - RoBERTa-SQuAD-v2 (125M parameters)
   
2. **Decoder-only Models**
   - BioGPT (349M parameters)
   - OPT-30B (180B parameters)
   - OPT-IML-MAX (30B parameters)
   - Llama 2 (70B parameters)
   - Vicuna-33B (33B parameters)
   - BLOOM (176B parameters)
   - WizardLM (70B parameters)

3. **Encoder-Decoder Models**
   - FLAN-T5-XXL (11B parameters)
   - FLAN-UL2 (20B parameters)

- **Evaluation Metrics:** Model performance was assessed based on precision and recall, focusing on the accurate extraction of relationships involving diseases, treatments, factors, and coexisting conditions.

#### **3. Relation Extraction**
- **LLM-Based Extraction:** Relations between diseases, treatments, factors, and coexisting conditions were extracted using LLMs, employing Open-Book QA and In-Context Learning approaches.
- **Prompt Design:** Three primary question categories were used to extract medical relationships:
  1. Treatment-related (e.g., "*What treats %s?*")
  2. Factor-related (e.g., "*What causes %s?*")
  3. Coexisting conditions (e.g., "*What coexists with %s?*")
  
- **Postprocessing:** Low-confidence outputs were filtered out, and synonymous responses were grouped to construct the final knowledge graph.

#### **4. Experiments and Results**
- **Case Study:** The methodology was demonstrated using *Age-related Macular Degeneration (AMD)* as an example.
- **LLM Performance Comparison:** The FLAN-UL2 model achieved the best precision and recall, especially when guided prompting techniques were applied.
- **Challenges with Decoder-only Models:** Some decoder-only models (e.g., BioGPT, OPT) struggled to generate structured outputs or exhibited hallucination, making them less reliable for structured relation extraction.





   
 
<br/>
# Results  



본 연구에서는 대형 언어 모델(LLM)을 활용하여 전자 의료 기록(EMR) 노트에서 지식 그래프(KG)를 구축하는 방법을 평가하였다. 이를 위해 다양한 아키텍처의 모델을 실험하고, 특정 질병을 대상으로 한 관계 추출 성능을 비교하였다.

---

### **1. 실험 환경 및 테스트 데이터**  
- **데이터셋:** 연구에 사용된 데이터는 *Macula and Retina Institute*에서 제공한 전자 의료 기록(EMR)이며, 약 10,000명의 망막 관련 질환 환자의 데이터를 포함한다. 총 360,000개의 임상 노트가 분석되었으며, 이 중 **노화 관련 황반변성(AMD, Age-related Macular Degeneration)** 관련 노트 320개가 관계 추출 실험에 사용되었다.  
- **데이터 구성:**  
  - 임상 노트에서 **질병(Disease), 치료법(Treatment), 요인(Factor), 공존 질환(Coexists_with)** 정보를 추출  
  - 해당 데이터셋을 기반으로 **수작업 주석(Annotation) 수행**하여 정답(ground truth) 데이터 구축  
  - 의료 전문가(망막 전문의 및 임상 연구 코디네이터)와 협업하여 주석 정확성 검증  

---

### **2. 비교 모델 (Baseline and Competitor Models)**  
본 연구에서는 다양한 대형 언어 모델(LLM)의 성능을 비교하기 위해 **12개의 모델**을 실험하였다.  

#### **(1) 인코더 전용 모델 (Encoder-only)**
- **BioBERT-SQuAD-v2** (110M parameters)  
- **BERT-SQuAD-v2** (110M parameters)  
- **RoBERTa-SQuAD-v2** (125M parameters)  

#### **(2) 디코더 전용 모델 (Decoder-only)**
- **BioGPT** (349M parameters)  
- **OPT-30B** (180B parameters)  
- **OPT-IML-MAX** (30B parameters)  
- **Llama 2** (70B parameters)  
- **Vicuna-33B** (33B parameters)  
- **BLOOM** (176B parameters)  
- **WizardLM** (70B parameters)  

#### **(3) 인코더-디코더 모델 (Encoder-decoder)**
- **FLAN-T5-XXL** (11B parameters)  
- **FLAN-UL2** (20B parameters)  

---

### **3. 평가 메트릭 (Evaluation Metrics)**
모델 성능은 다음과 같은 주요 평가 메트릭을 기반으로 분석되었다.  
1. **정밀도 (Precision):** 모델이 예측한 결과 중 실제로 올바른 관계를 추출한 비율  
2. **재현율 (Recall):** 실제 존재하는 관계 중 모델이 올바르게 추출한 비율  
3. **F1-score:** Precision과 Recall의 조화 평균  
4. **구조화된 출력 생성 여부:** 모델이 일관된 형식으로 관계를 추출할 수 있는지 평가  

---

### **4. 실험 결과 및 분석**  
모든 모델을 질병(AMD)을 중심으로 한 **치료법(Treatment), 요인(Factor), 공존 질환(Coexists_with)** 관계 추출 작업에서 평가하였다.  

#### **(1) 정량적 성능 평가 (Quantitative Performance)**
- **FLAN-UL2 (Guided Prompting)** 모델이 전반적으로 가장 우수한 성능을 보였으며, 정밀도(Precision) 및 재현율(Recall)에서 높은 점수를 기록하였다.  
- **디코더 전용 모델(Decoder-only LLMs)**은 구조화된 출력을 생성하는 데 어려움을 겪었으며, 일부 모델(BioGPT, OPT 등)은 환각(hallucination) 문제를 보였다.  

| **모델**                | **정밀도 (Precision)** | **재현율 (Recall)** |
|------------------------|---------------------|------------------|
| **FLAN-UL2 (Guided)**  | **0.98**            | **1.00**         |
| **FLAN-T5-XXL (Guided)** | **0.88**            | **1.00**         |
| **WizardLM-70B**       | 0.78                | 1.00             |
| **Llama-2-70B**        | 0.65                | 1.00             |
| **Vicuna-33B**         | 0.63                | 1.00             |
| **BioBERT-SQuAD-v2**   | 0.13                | 0.90             |
| **RoBERTa-SQuAD-v2**   | 0.25                | 0.54             |

#### **(2) 디코더 전용 모델의 한계 (Limitations of Decoder-only Models)**
- 일부 모델(BioGPT, OPT)은 **구조화된 출력 생성에 실패**하여 정확한 관계 추출이 어려웠다.  
- 특히 **BLOOM, OPT 모델은 환각(hallucination) 현상**을 보이며, 존재하지 않는 관계를 생성하는 경우가 많았다.  

#### **(3) 임상 적용 가능성**
- FLAN-UL2와 FLAN-T5-XXL 모델이 **의료 도메인 관계 추출에 적합**한 것으로 나타났다.  
- 모델이 생성한 KG는 기존 SemMedDB 기반 KG보다 더 많은 요인 및 치료법을 포함하며, 임상적 의사결정 지원 시스템(CDSS)에서 유용하게 활용될 수 있음이 시사되었다.  

---


This study evaluates the performance of large language models (LLMs) in constructing knowledge graphs (KGs) from electronic medical record (EMR) notes, comparing different architectures and extraction methods.

---

### **1. Experimental Setup & Test Data**
- **Dataset:** The study utilized EMR data from the *Macula and Retina Institute*, containing approximately **10,000 patients' records** with retina-related diseases.  
- **Data Composition:**  
  - Extracted **Disease, Treatment, Factor, and Coexists_with** relations from clinical notes.  
  - **Manual annotation by experts (retina specialist and clinical research coordinator)** to create ground truth data.  
  - **320 clinical notes** related to **Age-related Macular Degeneration (AMD)** were used for model evaluation.  

---

### **2. Baseline and Competitor Models**  
The study experimented with **12 LLMs** spanning different architectures.

#### **(1) Encoder-only Models**
- BioBERT-SQuAD-v2 (110M)  
- BERT-SQuAD-v2 (110M)  
- RoBERTa-SQuAD-v2 (125M)  

#### **(2) Decoder-only Models**
- BioGPT (349M)  
- OPT-30B (180B)  
- OPT-IML-MAX (30B)  
- Llama 2 (70B)  
- Vicuna-33B (33B)  
- BLOOM (176B)  
- WizardLM (70B)  

#### **(3) Encoder-Decoder Models**
- FLAN-T5-XXL (11B)  
- FLAN-UL2 (20B)  

---

### **3. Evaluation Metrics**
Models were assessed based on:  
1. **Precision:** Correct extractions out of total extractions.  
2. **Recall:** Correct extractions out of all actual relations.  
3. **F1-score:** Harmonic mean of Precision and Recall.  
4. **Structured output generation:** Whether the model produced consistent structured outputs.  

---

### **4. Experimental Results**
| **Model**                | **Precision** | **Recall** |
|------------------------|-------------|---------|
| **FLAN-UL2 (Guided)**  | **0.98**    | **1.00** |
| **FLAN-T5-XXL (Guided)** | **0.88**    | **1.00** |
| **WizardLM-70B**       | 0.78        | 1.00     |

- **FLAN-UL2 outperformed other models** with the highest precision and recall.  
- **Decoder-only models like BioGPT and OPT failed to generate structured outputs**, with hallucination issues.  

The results suggest that **FLAN-UL2 and FLAN-T5-XXL are most suitable** for medical knowledge extraction, offering potential applications in clinical decision support systems (CDSS).



<br/>
# 예제  




본 연구에서는 대형 언어 모델(LLM)을 활용하여 **전자 의료 기록(EMR) 노트에서 지식 그래프(KG)를 자동 구축**하는 방법을 제안한다. 이를 위해 다양한 관계 추출 작업을 수행하며, 실험에서 사용된 **트레이닝 데이터 및 테스트 데이터**, 그리고 **입출력 예제**를 포함한 테스크(task) 구성 방식을 설명한다.  

---

### **1. 트레이닝 데이터 (Training Data)**  
- **출처:** 연구에 사용된 데이터는 *Macula and Retina Institute*에서 제공한 전자 의료 기록(EMR) 데이터셋이다.  
- **규모:** 총 **360,000개의 임상 노트** 중, 약 **320개의 노트**를 수작업으로 주석하여 훈련 데이터 및 테스트 데이터로 사용하였다.  
- **데이터 포맷:** 각 노트는 환자의 상태, 진단 결과, 치료 내용 등을 포함한 **비정형 자연어 데이터** 형태로 제공됨.  
- **전처리 과정:** 문장 임베딩(Sentence T5 XXL)을 활용해 **중복 문서 제거** 후, 5단어 미만의 짧은 노트를 제외함.  

#### **훈련 데이터 예시 (Training Data Example)**  
```plaintext
Patient has been diagnosed with age-related macular degeneration. They are currently undergoing anti-VEGF therapy.
```
- **질병 (Disease):** Age-related macular degeneration (AMD)  
- **치료법 (Treatment):** Anti-VEGF therapy  

---

### **2. 테스트 데이터 (Test Data)**  
- **평가 목적:** 훈련된 모델이 새로운 임상 노트에서 **정확하게 관계를 추출할 수 있는지 검증**  
- **구성:** 훈련 데이터와는 다른 새로운 환자 기록을 포함하는 **80개의 임상 노트**를 평가에 사용  
- **메트릭:** 정밀도(Precision), 재현율(Recall), F1-score  

#### **테스트 데이터 예시 (Test Data Example)**  
```plaintext
The patient reports experiencing blurry vision in the left eye. OCT scan suggests geographic atrophy.
```
- **질병 (Disease):** Geographic atrophy  
- **증상 (Symptom):** Blurry vision  

---

### **3. 테스크 정의 (Task Definition: Relation Extraction)**  
본 연구에서는 **관계 추출(Relation Extraction)** 문제를 해결하는 모델을 개발하였다.  
모델이 수행해야 할 주요 관계 유형(Relation Types)은 다음과 같다.  

| 관계 유형 (Relation Type) | 예제 문장 (Example Sentence) | 추출된 관계 (Extracted Relation) |
|-----------------|--------------------------------|------------------|
| **질병-치료법** (Disease-Treatment) | The patient was prescribed **steroid therapy** for **uveitis**. | (Uveitis, Treated with, Steroid therapy) |
| **질병-원인** (Disease-Cause) | **Diabetes** is a major risk factor for **diabetic retinopathy**. | (Diabetic retinopathy, Caused by, Diabetes) |
| **질병-공존 질환** (Disease-Coexists_with) | Patients with **AMD** often also have **hypertension**. | (AMD, Coexists with, Hypertension) |

---

### **4. 모델의 인풋/아웃풋 예제 (Model Input/Output Examples)**  

#### **(1) 모델 입력 (Input Example)**
```plaintext
Patient diagnosed with neovascular AMD, started on monthly anti-VEGF injections.
```
- 질병: **Neovascular AMD**
- 예상 관계: **(Neovascular AMD, Treated with, Anti-VEGF injections)**  

#### **(2) 모델 출력 (Model Output Example)**
```json
{
  "Disease": "Neovascular AMD",
  "Treatment": "Anti-VEGF injections",
  "Relation": "Treated with"
}
```
- 모델이 추출한 결과는 **구조화된 JSON 형식**으로 출력됨.  

---

### **5. 실패 사례 및 개선점 (Failure Cases & Improvements)**  
모델이 완벽하지 않기 때문에 몇 가지 오류 사례가 발생함.  

#### **(1) 관계 누락 (Relation Missing)**  
- 입력 문장:  
  ```plaintext
  The patient suffers from diabetic retinopathy and hypertension.
  ```
- 모델 출력:  
  ```json
  {
    "Disease": "Diabetic retinopathy",
    "Coexists_with": "None"
  }
  ```
- 오류: **Hypertension과의 관계를 누락함** → 개선 필요  

#### **(2) 환각 오류 (Hallucination Issue)**  
- 입력 문장:  
  ```plaintext
  Patient diagnosed with AMD and follows a Mediterranean diet.
  ```
- 모델 출력:  
  ```json
  {
    "Disease": "AMD",
    "Treatment": "Mediterranean diet",
    "Relation": "Treated with"
  }
  ```
- 오류: **지중해식 식단은 공식적인 치료법이 아님** → 모델이 잘못된 관계를 생성함.  

---


This study proposes an approach for **automatically constructing knowledge graphs (KGs) from electronic medical record (EMR) notes using large language models (LLMs)**. Below, we describe the **training data, test data, task definitions, and input/output examples**.

---

### **1. Training Data**  
- **Source:** The dataset was provided by the *Macula and Retina Institute*.  
- **Size:** **360,000 clinical notes**, with **320 manually annotated notes** used for training.  
- **Format:** Free-text EMR notes containing diagnoses, treatments, and symptoms.  
- **Preprocessing:** Removed duplicate documents using **Sentence T5 XXL embeddings** and filtered out notes with fewer than five words.  

#### **Training Data Example**  
```plaintext
Patient has been diagnosed with age-related macular degeneration. They are currently undergoing anti-VEGF therapy.
```
- **Disease:** Age-related macular degeneration (AMD)  
- **Treatment:** Anti-VEGF therapy  

---

### **2. Test Data**  
- **Purpose:** Evaluate the model's ability to **extract relationships from unseen clinical notes**.  
- **Composition:** **80 clinical notes** separate from the training set.  
- **Metrics:** Precision, Recall, F1-score.  

#### **Test Data Example**  
```plaintext
The patient reports experiencing blurry vision in the left eye. OCT scan suggests geographic atrophy.
```
- **Disease:** Geographic atrophy  
- **Symptom:** Blurry vision  

---

### **3. Task Definition: Relation Extraction**  
The model is designed for **relation extraction**, identifying interactions between medical entities.  

| **Relation Type** | **Example Sentence** | **Extracted Relation** |
|-----------------|----------------------|------------------|
| **Disease-Treatment** | The patient was prescribed **steroid therapy** for **uveitis**. | (Uveitis, Treated with, Steroid therapy) |
| **Disease-Cause** | **Diabetes** is a major risk factor for **diabetic retinopathy**. | (Diabetic retinopathy, Caused by, Diabetes) |
| **Disease-Coexists_with** | Patients with **AMD** often also have **hypertension**. | (AMD, Coexists with, Hypertension) |

---

### **4. Model Input/Output Examples**  

#### **(1) Input Example**
```plaintext
Patient diagnosed with neovascular AMD, started on monthly anti-VEGF injections.
```
- **Expected Output:** **(Neovascular AMD, Treated with, Anti-VEGF injections)**  

#### **(2) Model Output Example**
```json
{
  "Disease": "Neovascular AMD",
  "Treatment": "Anti-VEGF injections",
  "Relation": "Treated with"
}
```
- The model outputs **structured JSON data**.  

---

### **5. Failure Cases & Improvements**  
Some common errors in model predictions:  

#### **(1) Missing Relations**
- **Input:**  
  ```plaintext
  The patient suffers from diabetic retinopathy and hypertension.
  ```
- **Output (Incorrect):**  
  ```json
  {
    "Disease": "Diabetic retinopathy",
    "Coexists_with": "None"
  }
  ```
- **Issue:** Failed to recognize **hypertension** as a coexisting condition.  

#### **(2) Hallucination Errors**
- **Input:**  
  ```plaintext
  Patient diagnosed with AMD and follows a Mediterranean diet.
  ```
- **Incorrect Output:**  
  ```json
  {
    "Disease": "AMD",
    "Treatment": "Mediterranean diet",
    "Relation": "Treated with"
  }
  ```
- **Issue:** The Mediterranean diet is **not a formal treatment**.



<br/>  
# 요약   



본 연구에서는 전자 의료 기록(EMR)에서 질병, 치료법, 요인 등의 관계를 자동으로 추출하는 지식 그래프(KG) 구축 모델을 제안하였다. 실험 결과, FLAN-UL2 모델이 가장 높은 정밀도와 재현율을 보였으며, 디코더 전용 모델은 구조화된 출력 생성에 어려움을 겪었다. 테스트 데이터에서 모델은 질병-치료법, 질병-원인, 질병-공존 질환 관계를 정확히 추출했지만, 일부 환각 오류(hallucination)가 발생하는 한계를 보였다.  

---


This study proposes a knowledge graph (KG) construction model that automatically extracts relationships such as diseases, treatments, and causes from electronic medical records (EMRs). Experimental results show that the FLAN-UL2 model achieved the highest precision and recall, while decoder-only models struggled with structured output generation. On test data, the model accurately extracted disease-treatment, disease-cause, and disease-coexisting condition relations, though some hallucination errors were observed.


<br/>  
# 기타  





본 연구에서는 실험 결과를 효과적으로 전달하기 위해 다양한 **피규어(Figure), 다이어그램(Diagram), 테이블(Table)**을 포함하였으며, 추가적인 정보를 제공하기 위해 **부록(Appendix)**을 활용하였다.  

---

### **1. 피규어 및 다이어그램 (Figures and Diagrams)**  
본 논문에서는 모델 아키텍처, 관계 추출 과정, 성능 비교 결과를 시각적으로 표현하기 위해 여러 개의 **피규어(Figure) 및 다이어그램**을 포함하였다.  

- **Figure 1: 연구 개요 다이어그램**  
  - 전자 의료 기록(EMR)에서 **지식 그래프(KG) 자동 구축 프로세스**를 단계별로 도식화  
  - 입력 데이터(임상 노트) → LLM 기반 관계 추출 → 구조화된 지식 그래프 생성 흐름을 나타냄  
- **Figure 2: 모델 아키텍처 비교**  
  - 인코더 전용(Encoder-only), 디코더 전용(Decoder-only), 인코더-디코더(Encoder-Decoder) 모델의 구조 차이를 시각적으로 설명  
- **Figure 3: 프롬프트 설계 및 응답 예시**  
  - 사용된 프롬프트 유형 및 LLM의 실제 응답을 비교  

---

### **2. 테이블 (Tables)**  
연구에서는 **모델 성능 비교 및 실험 결과를 정리하기 위해 여러 개의 테이블(Table)**을 사용하였다.  

- **Table 1: 실험에 사용된 데이터셋 개요**  
  - 훈련 데이터, 테스트 데이터의 개수 및 주요 통계를 포함  
- **Table 2: 모델별 정밀도(Precision) 및 재현율(Recall) 비교**  
  - FLAN-UL2, FLAN-T5-XXL이 가장 높은 성능을 보였으며, 디코더 전용 모델(OPT, BioGPT)은 구조화된 출력 생성에 실패하는 경우가 많았음  
- **Table 3: 관계 유형별 성능 비교**  
  - 질병-치료법, 질병-원인, 질병-공존 질환 관계를 정확하게 예측하는 모델을 비교 분석  

---

### **3. 부록 (Appendix)**  
논문의 본문에 포함하기 어려운 **추가적인 실험 결과 및 세부 구현 사항**을 제공하기 위해 **부록(Appendix)**을 구성하였다.  

- **Appendix A: 모델별 하이퍼파라미터 설정**  
  - 각 LLM의 훈련 시 사용한 하이퍼파라미터(학습률, 배치 크기, 최대 토큰 수 등)를 포함  
- **Appendix B: 추가적인 실패 사례(Failure Cases)**  
  - 모델이 발생시킨 환각(hallucination) 오류 및 잘못된 관계 추출 사례를 정리  
- **Appendix C: 데이터셋 주석 가이드라인**  
  - 임상 데이터 주석(Annotation) 시 사용된 기준 및 의료 전문가의 검토 과정 설명  

---



This study includes various **figures, diagrams, tables, and an appendix** to effectively present the experimental results and provide additional information.  

---

### **1. Figures and Diagrams**  
Several **figures and diagrams** were included to visually illustrate the **model architecture, relation extraction process, and performance comparison**.  

- **Figure 1: Study Overview Diagram**  
  - Depicts the step-by-step process of **automatically constructing a knowledge graph (KG) from EMRs**  
  - Shows the flow from input data (clinical notes) → LLM-based relation extraction → structured KG generation  
- **Figure 2: Model Architecture Comparison**  
  - Visualizes differences between **encoder-only, decoder-only, and encoder-decoder architectures**  
- **Figure 3: Prompt Design and Example Responses**  
  - Displays different prompt formats and actual LLM responses  

---

### **2. Tables**  
Several **tables** were included to summarize **model performance and experimental results**.  

- **Table 1: Dataset Overview**  
  - Summarizes the number of training and test samples along with key statistics  
- **Table 2: Model Precision and Recall Comparison**  
  - Shows that **FLAN-UL2 and FLAN-T5-XXL achieved the highest performance**, while **decoder-only models (OPT, BioGPT) often failed to generate structured outputs**  
- **Table 3: Performance by Relation Type**  
  - Compares models in extracting **disease-treatment, disease-cause, and disease-coexists_with relationships**  

---

### **3. Appendix**  
An **appendix** was included to provide **additional experimental results and implementation details** that were not covered in the main text.  

- **Appendix A: Hyperparameter Settings**  
  - Lists the hyperparameters used for training each LLM (learning rate, batch size, max tokens, etc.)  
- **Appendix B: Additional Failure Cases**  
  - Documents **hallucination errors and incorrect relation extractions**  
- **Appendix C: Dataset Annotation Guidelines**  
  - Describes the **annotation criteria and expert review process** for clinical data


<br/>
# refer format:     


@inproceedings{Arsenyan2024,
  author    = {Vahan Arsenyan and Spartak Bughdaryan and Fadi Shaya and Kent Small and Davit Shahnazaryan},
  title     = {Large Language Models for Biomedical Knowledge Graph Construction: Information Extraction from EMR Notes},
  booktitle = {Proceedings of the 23rd Workshop on Biomedical Language Processing},
  pages     = {295--317},
  year      = {2024},
  organization = {Association for Computational Linguistics},
  month     = {August},
  address   = {Online},
  publisher = {Association for Computational Linguistics}
}


Arsenyan, Vahan, Spartak Bughdaryan, Fadi Shaya, Kent Small, and Davit Shahnazaryan. 2024. “Large Language Models for Biomedical Knowledge Graph Construction: Information Extraction from EMR Notes.” In Proceedings of the 23rd Workshop on Biomedical Language Processing, 295–317. Association for Computational Linguistics, August.   




