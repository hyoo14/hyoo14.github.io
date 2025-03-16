---
layout: post
title:  "[2024]Evaluating Large Language Models for Predicting Protein Behavior under Radiation Exposure and Disease Conditions"  
date:   2025-03-15 11:05:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


LLM으로 질병과 방사선량 관계를 규명... 데이터(텍스트포함)주고 이진분류 시킴   


짧은 요약(Abstract) :    




이 연구는 이온화 방사선 노출이 단백질의 구조와 기능에 미치는 영향을 분석하고, 질병과의 관계를 이해하기 위해 대형 언어 모델(LLMs)을 활용하는 것을 목표로 한다. 높은 방사선량이 즉각적인 세포 손상을 일으켜 암을 유발할 수 있다는 것은 널리 알려져 있지만, 저선량 방사선(LDR)이 생물학적 시스템에 미치는 영향은 아직 명확히 규명되지 않았다. 이에 따라, 본 연구에서는 Mistral, Llama 2, Llama 3와 같은 오픈소스 LLM을 활용하여 방사선에 의해 변화된 단백질과 특정 질병(신경퇴행성 질환, 대사 질환, 암)과의 단백질-단백질 상호작용(PPI) 패턴을 예측하는 데 초점을 맞추었다. 

모델을 미세 조정한 결과, 기존의 방법보다 우수한 성능을 보이며, 특히 신경퇴행성 질환 및 대사 질환과 관련된 단백질 상호작용 예측에서 뛰어난 정확도를 기록했다. 연구 결과는 방사선 노출이 질병 메커니즘과 어떻게 연관되는지를 분석하는 데 기여하며, 현재의 컴퓨터 모델이 가지는 한계와 가능성을 보여준다. 또한, 연구에서 사용된 코드와 데이터는 깃허브를 통해 공개되어 있어, 후속 연구를 위한 기반을 제공한다.

---


The primary concern with exposure to ionizing radiation is the risk of developing diseases. While high doses of radiation can cause immediate damage leading to cancer, the effects of low-dose radiation (LDR) are less clear and more controversial. To further investigate this, it necessitates focusing on the underlying biological structures affected by radiation. Recent work has shown that Large Language Models (LLMs) can effectively predict protein structures and other biological properties. 

The aim of this research is to utilize open-source LLMs, such as Mistral, Llama 2, and Llama 3, to predict both radiation-induced alterations in proteins and the dynamics of protein-protein interactions (PPIs) within the presence of specific diseases. We show that fine-tuning these models yields state-of-the-art performance for predicting protein interactions in the context of neurodegenerative diseases, metabolic disorders, and cancer. 

Our findings contribute to the ongoing efforts to understand the complex relationships between radiation exposure and disease mechanisms, illustrating the nuanced capabilities and limitations of current computational models. The code and data are available at: [GitHub Repository](https://github.com/Rengel2001/SURP_2024).



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





이 연구에서는 **Mistral (7B), Llama 2 (7B), Llama 3 (8B)** 등의 오픈소스 대형 언어 모델(LLMs)을 활용하여 **저선량 방사선(LDR) 노출과 질병 상태에서 단백질의 변화 및 단백질-단백질 상호작용(PPI)을 예측하는 모델**을 개발했다. 이 모델들은 자연어 처리(NLP)에서 최첨단 성능을 보이며, 개방형(Open-source) 모델로서 투명한 연구 및 협업이 가능하다는 장점이 있다.

---

#### **1. 사용된 데이터셋**
연구에서는 **6개의 주요 데이터셋**을 사용했으며, 이를 **13개의 하위 데이터셋**으로 세분화하여 실험을 진행했다. 데이터셋은 다음 두 가지 범주로 나뉜다.

- **LDR 데이터셋 (방사선이 단백질에 미치는 영향 연구)**
  - 방사선이 단백질 발현과 상호작용에 미치는 영향을 조사하는 데이터셋
  - 3개의 핵심 데이터셋이 존재하며, 각각 10개의 세부 데이터셋으로 나누어 실험 진행

- **PPI 데이터셋 (질병 내 단백질-단백질 상호작용 연구)**
  - 특정 질병(신경퇴행성 질환, 대사 질환, 암) 내에서 PPI 네트워크 분석
  - 3개의 핵심 데이터셋으로 구성됨

PPI 데이터는 **BioGRID, STRING, HIPPIE, KEGG** 등의 공공 데이터베이스를 통해 수집되었으며, 특히 LDR 데이터셋과 PPI 데이터셋 간의 **단백질 중복(overlap)** 을 분석하여 방사선에 의해 조절되는 단백질을 파악했다.

---

#### **2. 모델 아키텍처 및 학습 방법**
##### **(1) 모델 구조**
- LLM들은 **이진 분류(Binary Classification) 태스크**를 수행하도록 설계되었으며, 각 프롬프트에 대해 "Yes" 또는 "No" 응답을 생성하도록 훈련됨.
- 모델이 자연어로 작성된 생물학적 정보를 바탕으로 단백질 변화를 예측할 수 있도록 설정.

##### **(2) 프롬프트 설계**
- LDR 태스크에서는 **단일 단백질 이름**을 프롬프트에 포함하여 방사선 노출로 인한 변화 여부 예측
- PPI 태스크에서는 **두 개의 단백질 이름**을 포함하여 상호작용 여부를 예측
- 데이터셋별로 **최적화된 프롬프트 엔지니어링 전략**을 적용하여 모델 성능을 향상

##### **(3) 미세 조정 (Fine-tuning)**
- **Parameter Efficient Fine-Tuning (PEFT)** 기법을 활용하여 LLM의 일부 가중치만 업데이트
- **LoRA (Low-Rank Adaptation)** 및 **QLoRA (Quantized LoRA)** 적용하여 GPU 메모리 사용량 절감 및 성능 향상
- **Hugging Face의 `Trainer` 및 `SFTTrainer`** 를 활용하여 모델을 미세 조정

##### **(4) 하드웨어 및 환경**
- **4× NVIDIA A100 80GB GPU**에서 모델을 학습
- Hugging Face Transformers 라이브러리를 사용하여 LLM 배포

---

#### **3. 데이터 분할 및 평가 방법**
##### **(1) 데이터 분할**
- LDR 데이터셋: **80/10/10 (훈련/검증/테스트) 분할**
- PPI 데이터셋: **5-Fold 교차 검증** 적용하여 보다 견고한 평가 수행

##### **(2) 성능 평가 지표**
- 정확도(Accuracy), 매튜 상관 계수(MCC), 특이도(Specificity), 정밀도(Precision), F1 점수(F1 Score) 등 다양한 지표를 활용하여 모델 성능 평가

---

### **한글 요약**
이 연구에서는 Mistral, Llama 2, Llama 3 등의 LLM을 활용하여 방사선 노출과 질병 환경에서 단백질 변화 및 PPI를 예측하는 모델을 개발했다. 총 6개의 데이터셋(13개의 하위 데이터셋)이 사용되었으며, PPI 데이터는 공개 데이터베이스에서 수집되었다. 모델은 이진 분류 태스크로 훈련되었으며, 최적화된 프롬프트 엔지니어링을 적용했다. 또한, LoRA 및 QLoRA 기법을 활용한 미세 조정을 수행하고, 5-Fold 교차 검증을 통해 성능을 평가했다. 모델은 Hugging Face 라이브러리를 사용하여 4× NVIDIA A100 80GB GPU에서 학습되었으며, 정확도, MCC, F1 Score 등을 기준으로 성능을 분석했다.

---



This study employs **open-source Large Language Models (LLMs)**, including **Mistral (7B), Llama 2 (7B), and Llama 3 (8B)**, to predict **radiation-induced protein alterations and protein-protein interactions (PPIs) under disease conditions**. These models were selected for their **state-of-the-art NLP performance and open-source availability**, facilitating transparent and collaborative research.

---

#### **1. Datasets**
The study utilizes **six core datasets**, further divided into **13 subsets**, categorized as follows:

- **LDR datasets (Radiation-induced protein deregulation)**
  - Focused on the impact of radiation on protein expression and interactions.
  - Three key datasets, split into ten subsets based on different experimental conditions.

- **PPI datasets (Protein-protein interactions in disease conditions)**
  - Investigating PPI networks in neurodegenerative diseases, metabolic disorders, and cancer.
  - Three core datasets.

PPI data was collected from **public databases** such as **BioGRID, STRING, HIPPIE, and KEGG**, and the study **cross-referenced proteins between LDR and PPI datasets** to identify radiation-sensitive proteins.

---

#### **2. Model Architecture & Training**
##### **(1) Model Design**
- LLMs were designed to perform a **binary classification task**, predicting "Yes" or "No" responses to prompts.
- The models leverage **natural language-based biological knowledge** to predict protein alterations.

##### **(2) Prompt Engineering**
- **LDR Task**: Single protein name in the prompt to predict radiation-induced deregulation.
- **PPI Task**: Two protein names in the prompt to predict interaction likelihood.
- Task-specific **prompt optimization strategies** were applied to enhance model performance.

##### **(3) Fine-Tuning**
- **Parameter Efficient Fine-Tuning (PEFT)** was used to update only a subset of model weights.
- **Low-Rank Adaptation (LoRA)** and **Quantized LoRA (QLoRA)** were employed to optimize GPU memory usage and training efficiency.
- **Hugging Face `Trainer` and `SFTTrainer`** were used for model fine-tuning.

##### **(4) Hardware & Training Setup**
- Training was conducted on **4× NVIDIA A100 80GB GPUs**.
- Hugging Face Transformers library was used for model deployment.

---

#### **3. Data Splitting & Evaluation Metrics**
##### **(1) Data Splitting**
- **LDR datasets**: 80/10/10 split for training, validation, and testing.
- **PPI datasets**: **5-fold cross-validation** was implemented for robust evaluation.

##### **(2) Performance Metrics**
- Accuracy, Matthews Correlation Coefficient (MCC), Specificity, Precision, and F1 Score were used to assess model performance.

---

### **English Summary**
This study develops an LLM-based approach for predicting protein behavior under radiation exposure and disease conditions. It utilizes **six datasets (13 subsets)**, with PPI data sourced from public databases. The models were trained as **binary classifiers** with optimized prompt engineering strategies. **LoRA and QLoRA fine-tuning** were applied, and **5-fold cross-validation** was used for evaluation. Training was conducted using Hugging Face libraries on **4× NVIDIA A100 80GB GPUs**, with performance assessed using Accuracy, MCC, and F1 Score.






   
 
<br/>
# Results  




이 연구에서는 **Mistral (7B), Llama 2 (7B), Llama 3 (8B)** 등의 대형 언어 모델(LLMs)을 활용하여 **방사선 노출과 질병 상태에서 단백질 변화 및 단백질-단백질 상호작용(PPI)**을 예측하는 성능을 평가했다. 특히, 기존의 **경쟁 모델들과 비교**하여 LLM 기반 모델이 가지는 장점과 한계를 분석했다.

---

### **1. 경쟁 모델 및 비교**
이 연구에서 평가된 모델들은 다음과 같다:

- **Baseline (기본 모델)**
  - 사전 훈련된 **Mistral (3-shot)**, **Llama 2 (3-shot)**, **Llama 3 (3-shot)**
  - 사전 훈련된 상태에서 3개의 예제(샷)를 제공한 후 테스트 진행

- **미세 조정된 모델**
  - **Mistral (LoRA)**
  - **Llama 2 (LoRA)**
  - **Llama 3 (LoRA)**
  - **Llama 3 (QLoRA)** (GPU 메모리 최적화를 위한 Quantized LoRA 적용)

- **기존 경쟁 모델**
  - **SymLMF (Pei et al., 2021)**: 대칭 로지스틱 행렬 분해(Symmetric Logistic Matrix Factorization)를 사용한 PPI 예측 모델
  - **NECARE (Qiu et al., 2021)**: 암과 관련된 PPI를 예측하는 딥러닝 모델

---

### **2. 테스트 데이터 및 실험 환경**
- **테스트 데이터**
  - **LDR 데이터셋**: 방사선이 단백질 변화에 미치는 영향 평가 (80/10/10 훈련/검증/테스트 분할)
  - **PPI 데이터셋**: 신경퇴행성 질환, 대사 질환, 암과 관련된 단백질 상호작용 예측 (5-Fold 교차 검증)

- **하드웨어 및 환경**
  - 4× NVIDIA A100 80GB GPU 사용
  - Hugging Face `Trainer` 및 `SFTTrainer`를 사용한 모델 학습 및 평가

---

### **3. 평가 메트릭 및 성능 비교**
모델 성능은 **다양한 평가 지표**를 활용하여 비교되었다:

- **정확도 (Accuracy)**
- **매튜 상관 계수 (Matthews Correlation Coefficient, MCC)**
- **특이도 (Specificity)**
- **정밀도 (Precision)**
- **F1 점수 (F1 Score)**

#### **(1) PPI 예측 성능 비교**
| 모델 | 정확도 (%) | MCC (%) | 특이도 (%) | 정밀도 (%) | F1 점수 (%) |
|------|----------|---------|---------|---------|---------|
| **Mistral (3-shot)** | 38.44 ± 0.46 | -31.79 ± 0.54 | 4.15 ± 0.16 | 28.16 ± 0.28 | 30.23 ± 0.25 |
| **Llama 2 (3-shot)** | 55.14 ± 0.16 | 13.18 ± 0.29 | 23.79 ± 0.57 | 58.47 ± 0.23 | 50.23 ± 0.17 |
| **Llama 3 (3-shot)** | 50.38 ± 0.36 | 6.17 ± 0.27 | 1.0 ± 0.0 | 75.10 ± 0.18 | 34.18 ± 0.17 |
| **Mistral (LoRA)** | 62.34 ± 6.88 | 25.53 ± 14.37 | 97.89 ± 1.17 | 48.49 ± 12.84 | 51.97 ± 10.36 |
| **Llama 2 (LoRA)** | 87.28 ± 0.41 | 76.63 ± 1.03 | 88.59 ± 0.95 | 87.33 ± 0.22 | 87.28 ± 0.31 |
| **Llama 3 (QLoRA)** | **88.27 ± 1.08** | **76.92 ± 2.12** | **92.81 ± 1.06** | **88.58 ± 1.08** | **88.26 ± 1.07** |
| **SymLMF (기존 모델, 보고된 값)** | 86.11 ± 1.05 | 74.29 ± 2.07 | N/A | 83.24 ± 1.28 | N/A |

#### **(2) 신경퇴행성 질환 PPI 예측**
- Llama 3 (QLoRA)의 정확도가 **88.27%**로 가장 높으며, 기존의 SymLMF 모델보다 우수한 성능을 보임

#### **(3) 대사 질환 PPI 예측**
| 모델 | 정확도 (%) | MCC (%) | 특이도 (%) | 정밀도 (%) | F1 점수 (%) |
|------|----------|---------|---------|---------|---------|
| **Llama 3 (QLoRA)** | **91.28 ± 0.87** | **82.57 ± 1.73** | 90.41 ± 1.08 | 91.29 ± .86 | 91.28 ± 0.87 |
| **SymLMF (기존 모델)** | 81.37 ± 1.04 | 63.31 ± 2.07 | N/A | 77.70 ± 1.07 | N/A |

#### **(4) 암 PPI 예측**
| 모델 | 정확도 (%) | 정밀도 (%) |
|------|----------|---------|
| **Llama 3 (LoRA)** | **93.94%** | **94.09%** |
| **NECARE (기존 모델)** | 90.0% | 94.0% |

---

### **4. 결과 해석 및 한계점**
- **미세 조정된 Llama 3 (QLoRA) 모델이 모든 PPI 예측 태스크에서 최고 성능**을 기록했으며, 기존의 SymLMF 및 NECARE 모델을 능가함
- **LoRA 기반 미세 조정**을 통해 **약 50% 성능 향상**
- **LDR 태스크에서는 기존 모델 대비 상대적으로 개선이 미미** → 이는 단백질 간 상호작용 정보를 포함하지 않기 때문으로 분석됨
- **향후 연구 방향**:
  - **LDR 데이터셋 확장**
  - **프롬프트 엔지니어링 최적화**
  - **PPI 및 LDR 정보를 통합한 복합 모델 개발**

---



This study evaluated **Mistral (7B), Llama 2 (7B), and Llama 3 (8B)** for predicting **radiation-induced protein alterations and protein-protein interactions (PPIs) in disease conditions**. The performance of these models was compared against **state-of-the-art competitive models**.

---

### **1. Competitive Models**
- **Baseline Models**: Mistral (3-shot), Llama 2 (3-shot), Llama 3 (3-shot)
- **Fine-tuned Models**: Mistral (LoRA), Llama 2 (LoRA), Llama 3 (LoRA), Llama 3 (QLoRA)
- **Existing Competitive Models**:
  - **SymLMF (Pei et al., 2021)**
  - **NECARE (Qiu et al., 2021)**

---

### **2. Test Data & Evaluation Metrics**
- **Datasets**: LDR (80/10/10 split), PPI (5-Fold Cross-Validation)
- **Metrics**: Accuracy, MCC, Specificity, Precision, F1 Score

---

### **3. Performance Summary**
- **Best model: Llama 3 (QLoRA)** outperformed all other models in **PPI prediction tasks**.
- **50% accuracy improvement** from fine-tuning.
- **LDR prediction showed limited improvement**, suggesting the need for additional relational data.

### **4. Future Directions**
- Expanding **LDR datasets**.
- Optimizing **prompt engineering**.
- Developing **integrated PPI-LDR models**.



<br/>
# 예제  




이 연구에서 사용된 데이터는 **방사선 노출이 단백질에 미치는 영향 (LDR 데이터)** 및 **질병 상태에서 단백질-단백질 상호작용(PPI 데이터)** 를 예측하는 두 가지 주요 태스크로 나뉜다. 각각의 태스크에서 훈련 및 테스트 데이터를 사용하여 모델이 특정 입력에 대해 적절한 출력을 생성할 수 있도록 학습되었다.

---

## **1. 태스크 정의 및 데이터 예제**
### **(1) 방사선 노출이 단백질에 미치는 영향 (LDR 태스크)**
#### **목표:**  
주어진 단백질이 저선량 방사선(LDR) 노출에 의해 조절되는지 여부를 예측하는 이진 분류(Binary Classification) 태스크.

#### **입력 데이터 예제 (훈련 데이터)**:
| 단백질 이름 | 방사선 노출 후 조절 여부 (정답) |
|------------|----------------------------|
| TP53       | Yes                         |
| BRCA1      | Yes                         |
| MYC        | No                          |
| AKT1       | Yes                         |

#### **모델 인풋 (프롬프트 예제)**
```
Does exposure to low-dose radiation regulate the protein TP53?
```

#### **모델 아웃풋 (예측값 예제)**
```
Yes
```

#### **평가 기준:**
- 정확도(Accuracy)
- 매튜 상관 계수(MCC)
- 정밀도(Precision)
- F1 점수(F1 Score)

---

### **(2) 단백질-단백질 상호작용 예측 (PPI 태스크)**
#### **목표:**  
특정 질병 환경에서 주어진 두 개의 단백질이 서로 상호작용하는지 여부를 예측하는 태스크.

#### **입력 데이터 예제 (훈련 데이터)**:
| 단백질 A | 단백질 B | 질병 | 상호작용 여부 (정답) |
|----------|----------|------|----------------|
| TP53     | MDM2     | 암   | Yes           |
| APP      | PSEN1    | 알츠하이머 | Yes           |
| BRCA1    | AKT1     | 대사질환 | No            |

#### **모델 인풋 (프롬프트 예제)**
```
Do the proteins TP53 and MDM2 interact in cancer?
```

#### **모델 아웃풋 (예측값 예제)**
```
Yes
```

#### **평가 기준:**
- 정확도(Accuracy)
- 매튜 상관 계수(MCC)
- 특이도(Specificity)
- 정밀도(Precision)
- F1 점수(F1 Score)

---

## **2. 훈련 및 테스트 데이터 구조**
### **(1) 데이터 분할**
- **LDR 태스크**: 80% (훈련) / 10% (검증) / 10% (테스트)로 분할
- **PPI 태스크**: 5-Fold 교차 검증 수행

### **(2) 훈련 및 테스트 데이터 예제**
#### **LDR 태스크 (훈련 데이터 예제)**
| 단백질 이름 | 방사선 노출 후 조절 여부 |
|------------|----------------|
| TP53       | Yes           |
| MYC        | No            |

#### **PPI 태스크 (훈련 데이터 예제)**
| 단백질 A | 단백질 B | 질병 | 상호작용 여부 |
|----------|----------|------|------------|
| APP      | PSEN1    | 알츠하이머 | Yes        |
| BRCA1    | AKT1     | 대사질환 | No         |

---

## **3. 모델 훈련 과정**
- **Llama 3 (QLoRA)** 모델을 사용하여 최적화
- **LoRA 및 QLoRA 기법 적용**하여 효율적인 GPU 메모리 사용
- **프롬프트 엔지니어링을 최적화**하여 더 높은 예측 정확도 확보

---


이 연구에서는 **LDR 태스크 (방사선 노출 단백질 조절 여부 예측)** 및 **PPI 태스크 (질병 상태에서 단백질 간 상호작용 예측)** 를 수행하였다. 모델은 **Llama 3 (QLoRA)** 를 사용하여 학습되었으며, **LoRA 및 QLoRA 기법을 적용하여 GPU 메모리 효율성을 높였다**. 데이터를 **80/10/10 (LDR 태스크)** 및 **5-Fold 교차 검증 (PPI 태스크)** 로 분할하여 평가하였으며, **프롬프트 엔지니어링을 최적화** 하여 정확도를 향상시켰다.

---



This study includes two key tasks: **predicting radiation-induced protein deregulation (LDR Task)** and **predicting protein-protein interactions (PPI Task) in disease conditions**. Training and test datasets were carefully constructed to optimize model learning.

---

## **1. Task Definition & Example Data**
### **(1) Radiation-Induced Protein Deregulation (LDR Task)**
#### **Objective:**  
Binary classification task predicting whether a given protein is regulated under low-dose radiation (LDR) exposure.

#### **Training Data Example**:
| Protein Name | Regulated after Radiation Exposure (Label) |
|-------------|----------------------------------|
| TP53        | Yes                              |
| BRCA1       | Yes                              |
| MYC         | No                               |
| AKT1        | Yes                              |

#### **Model Input (Prompt Example)**
```
Does exposure to low-dose radiation regulate the protein TP53?
```

#### **Model Output (Prediction Example)**
```
Yes
```

#### **Evaluation Metrics:**
- Accuracy
- Matthews Correlation Coefficient (MCC)
- Precision
- F1 Score

---

### **(2) Protein-Protein Interaction Prediction (PPI Task)**
#### **Objective:**  
Predict whether two proteins interact within the context of a given disease.

#### **Training Data Example**:
| Protein A | Protein B | Disease | Interaction (Label) |
|----------|----------|---------|------------------|
| TP53     | MDM2     | Cancer  | Yes              |
| APP      | PSEN1    | Alzheimer's | Yes          |
| BRCA1    | AKT1     | Metabolic Disorder | No |

#### **Model Input (Prompt Example)**
```
Do the proteins TP53 and MDM2 interact in cancer?
```

#### **Model Output (Prediction Example)**
```
Yes
```

#### **Evaluation Metrics:**
- Accuracy
- Matthews Correlation Coefficient (MCC)
- Specificity
- Precision
- F1 Score

---

## **2. Training & Test Data Structure**
### **(1) Data Splitting**
- **LDR Task**: 80% Training / 10% Validation / 10% Test Split
- **PPI Task**: 5-Fold Cross-Validation

### **(2) Training & Test Data Example**
#### **LDR Task (Training Data Example)**
| Protein Name | Regulated after Radiation Exposure |
|-------------|----------------------------------|
| TP53        | Yes                              |
| MYC         | No                               |

#### **PPI Task (Training Data Example)**
| Protein A | Protein B | Disease | Interaction |
|----------|----------|---------|-------------|
| APP      | PSEN1    | Alzheimer's | Yes        |
| BRCA1    | AKT1     | Metabolic Disorder | No |

---

## **3. Model Training Process**
- **Llama 3 (QLoRA) model** was used for optimization.
- **LoRA and QLoRA techniques** were applied for efficient GPU memory usage.
- **Prompt engineering optimization** improved prediction accuracy.

---


This study involves **two tasks: LDR Task (predicting protein regulation under radiation) and PPI Task (predicting protein interactions in disease conditions)**. The model was trained using **Llama 3 (QLoRA)**, employing **LoRA and QLoRA techniques to enhance GPU efficiency**. The data was split into **80/10/10 (LDR Task) and 5-Fold Cross-Validation (PPI Task)**, with **prompt engineering optimizations** improving accuracy.


<br/>  
# 요약   




이 연구에서는 **Mistral, Llama 2, Llama 3** 모델을 활용하여 방사선 노출과 질병 환경에서 단백질 변화 및 단백질-단백질 상호작용(PPI)을 예측하는 모델을 개발하고, LoRA 및 QLoRA 기법을 적용하여 최적화하였다. 실험 결과, **미세 조정된 Llama 3 (QLoRA)** 모델이 모든 PPI 예측 태스크에서 **기존 경쟁 모델을 능가하는 최고 성능(정확도 88~93%)** 을 기록했다. 훈련 데이터는 **방사선 조절 단백질(LDR) 및 질병별 PPI 데이터셋** 으로 구성되었으며, 모델은 **프롬프트 기반 이진 분류 태스크** 를 수행하였다.  

---


This study developed models using **Mistral, Llama 2, and Llama 3** to predict protein alterations and protein-protein interactions (PPIs) under radiation exposure and disease conditions, optimized with LoRA and QLoRA techniques. Experimental results showed that the **fine-tuned Llama 3 (QLoRA) model** outperformed existing competitive models, achieving **88–93% accuracy in PPI prediction tasks**. The training data consisted of **radiation-regulated protein (LDR) and disease-specific PPI datasets**, and the model was trained as a **prompt-based binary classification task**.


<br/>  
# 기타  





이 연구에서는 결과를 시각적으로 이해하기 위해 다양한 **피규어(Figure), 다이어그램(Diagram), 테이블(Table)** 을 포함하고 있으며, 추가적인 실험 결과와 분석을 담은 **부록(Appendix)** 이 제공된다.  

---

### **1. 주요 피규어 및 다이어그램**  
#### **(1) 모델 아키텍처 다이어그램**  
- Llama 3 (QLoRA) 모델을 활용한 **PPI 예측 구조 및 LoRA 적용 방식** 을 설명하는 다이어그램이 포함됨.  
- **프롬프트 기반 분류 방식** 과 **LoRA 파라미터 조정 과정** 을 시각적으로 표현함.  

#### **(2) 데이터 분포 및 전처리 과정**  
- LDR 및 PPI 데이터셋의 **단백질 개수, 상호작용 개수 및 질병별 샘플 수** 를 보여주는 막대 그래프 포함.  
- **5-Fold 교차 검증 데이터 분할 과정** 을 설명하는 흐름도(flowchart) 제공.  

#### **(3) 모델 성능 비교 그래프**  
- Llama 3 (QLoRA) 모델과 기존 모델(SymLMF, NECARE 등)의 **정확도(Accuracy), F1 점수(F1 Score), MCC** 를 비교하는 **막대 그래프 및 선 그래프** 포함.  

---

### **2. 주요 테이블**  
#### **(1) 모델 성능 비교 테이블**  
- PPI 및 LDR 태스크에서의 **정확도, 정밀도, 특이도, MCC 점수** 비교.  
- **LoRA, QLoRA 적용 여부에 따른 성능 차이 분석** 포함.  

#### **(2) 프롬프트 예제 및 모델 출력 테이블**  
- LDR 및 PPI 태스크에서 사용된 **프롬프트 예제 및 모델 예측값(출력)** 정리.  
- 미세 조정된 모델과 사전 훈련된 모델 간의 **출력 차이** 설명.  

---

### **3. 어펜딕스(부록) 내용**  
#### **(1) 추가적인 모델 평가 및 오류 분석**  
- 특정 단백질 상호작용 예측에서 발생한 **False Positive 및 False Negative 케이스 분석**.  
- **잘못 예측된 사례를 정리한 테이블** 포함.  

#### **(2) 데이터셋 세부 정보**  
- 각 데이터셋의 출처(BioGRID, STRING, KEGG 등) 및 **단백질과 질병 간 매핑 방식** 설명.  
- 데이터 전처리 과정 및 **샘플링 전략** 포함.  

#### **(3) 하이퍼파라미터 튜닝 및 학습 세부 설정**  
- **LoRA 적용 방식 및 학습률(Learning Rate) 조정 과정** 설명.  
- GPU 메모리 사용량 비교 및 **훈련 시간 분석** 포함.  

---


이 논문에는 **모델 아키텍처 다이어그램, 데이터 분포 그래프, 모델 성능 비교 그래프 및 다양한 테이블** 이 포함되어 있으며, 추가적인 실험 및 오류 분석 결과를 담은 **부록(Appendix)** 이 제공된다. 부록에는 **잘못된 예측 분석, 데이터셋 상세 정보, 하이퍼파라미터 설정** 등이 포함되며, 연구의 재현성을 높이기 위한 추가 자료가 제공된다.  

---


This study includes **figures, diagrams, tables**, and a **detailed appendix** to provide a comprehensive understanding of the results and methodology.  

---

### **1. Key Figures & Diagrams**  
#### **(1) Model Architecture Diagram**  
- A visual representation of **PPI prediction using Llama 3 (QLoRA) and LoRA parameter adjustments**.  
- **Prompt-based classification framework** and fine-tuning strategy illustrated.  

#### **(2) Data Distribution & Preprocessing**  
- **Bar charts** illustrating the number of proteins, interactions, and disease-specific samples in LDR and PPI datasets.  
- A **flowchart** describing the **5-Fold cross-validation data split** process.  

#### **(3) Model Performance Comparison Graphs**  
- **Bar and line charts** comparing **accuracy, F1 Score, and MCC** of Llama 3 (QLoRA) against competitive models (SymLMF, NECARE).  

---

### **2. Key Tables**  
#### **(1) Model Performance Comparison Table**  
- Accuracy, precision, specificity, and MCC scores across **PPI and LDR tasks**.  
- **Performance differences between LoRA and QLoRA models** analyzed.  

#### **(2) Prompt Examples & Model Output Table**  
- **Example prompts and predictions** for LDR and PPI tasks.  
- Comparison of outputs between **fine-tuned and pre-trained models**.  

---

### **3. Appendix Contents**  
#### **(1) Additional Model Evaluation & Error Analysis**  
- **False Positive and False Negative case analysis** in PPI predictions.  
- **Tables summarizing incorrect predictions**.  

#### **(2) Dataset Details**  
- Sources of data (BioGRID, STRING, KEGG) and **protein-disease mapping strategies**.  
- **Data preprocessing techniques** and sampling strategies included.  

#### **(3) Hyperparameter Tuning & Training Setup**  
- **LoRA fine-tuning details and learning rate adjustments**.  
- **GPU memory usage comparison and training time analysis**.  

---

The paper includes **figures, data distribution graphs, model performance comparison charts, and various tables**, along with a **detailed appendix** containing **additional experiments and error analyses**. The appendix covers **incorrect prediction analysis, dataset details, and hyperparameter configurations**, providing additional resources to ensure reproducibility.


<br/>
# refer format:     



@inproceedings{Engel2024,
  author    = {Ryan Engel and Gilchan Park},
  title     = {Evaluating Large Language Models for Predicting Protein Behavior under Radiation Exposure and Disease Conditions},
  booktitle = {Proceedings of the 23rd Workshop on Biomedical Language Processing},
  pages     = {427--439},
  year      = {2024},
  month     = {August},
  organization = {Association for Computational Linguistics},
  url       = {https://github.com/Rengel2001/SURP_2024}
}



Engel, Ryan, and Gilchan Park. "Evaluating Large Language Models for Predicting Protein Behavior under Radiation Exposure and Disease Conditions." Proceedings of the 23rd Workshop on Biomedical Language Processing, August 16, 2024, 427–439. Association for Computational Linguistics. 




