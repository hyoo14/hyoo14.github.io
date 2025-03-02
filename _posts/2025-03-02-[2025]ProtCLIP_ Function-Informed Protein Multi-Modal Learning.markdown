---
layout: post
title:  "[2025]ProtCLIP: Function-Informed Protein Multi-Modal Learning"  
date:   2025-03-02 13:50:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


단백질과 그것을 설명하는 텍스트 데이터셋을 제안하고, 제안된 데이터로 학습하여 좋은 성능  
(근데 클립이지만 이미지랑은 상관없네..)  


짧은 요약(Abstract) :    




기존의 단백질-텍스트 다중 모달 사전 학습 방법은 단백질 서열과 생물학적 설명을 정렬하여 단백질 표현을 학습했지만, 기존의 이미지-텍스트 기반 모델만큼의 성과를 내지 못했습니다. 이러한 한계를 극복하기 위해 본 연구에서는 새로운 대규모 단백질-텍스트 데이터셋 **ProtAnno**를 구축하고, 기능 중심의 사전 학습 패러다임을 제안합니다. 이 접근법은 샘플 신뢰도 및 속성 범위를 고려한 데이터 샘플링 전략을 통해 데이터 품질과 양의 균형을 맞추고, 정적 및 동적 기능 세그먼트를 명시적으로 모델링하는 새로운 사전 학습 목표를 도입하여 단백질 기능 정보를 보다 정밀하게 반영합니다. 이를 기반으로 개발된 **ProtCLIP** 모델은 22개 단백질 벤치마크 테스트에서 5가지 주요 유형의 작업(단백질 기능 분류, 돌연변이 효과 예측, 교차 모달 변환, 의미적 유사성 추론, 단백질-단백질 상호작용 예측)에서 기존 SOTA 성능을 능가하는 결과를 보였으며, 특히 교차 모달 변환에서는 평균 75%, GO-CC 단백질 기능 예측에서는 59.9%, GO-BP에서는 39.7%의 향상을 기록하였습니다. 이를 통해 **ProtCLIP**이 단백질 다중 모달 기초 모델로서의 잠재력을 입증하였습니다.

---


The multi-modality pre-training paradigm that aligns protein sequences and biological descriptions has learned general protein representations and achieved promising performance in various downstream applications. However, these works were still unable to replicate the extraordinary success of language-supervised visual foundation models due to the ineffective usage of aligned protein-text paired data and the lack of an effective function-informed pre-training paradigm. To address these issues, this paper curates a large-scale protein-text paired dataset called **ProtAnno** with a property-driven sampling strategy and introduces a novel function-informed protein pre-training paradigm. Specifically, the sampling strategy determines selecting probability based on the sample confidence and property coverage, balancing data quality and quantity in the face of large-scale noisy data. Furthermore, motivated by the significance of protein-specific functional mechanisms, the proposed paradigm explicitly models protein static and dynamic functional segments through two segment-wise pre-training objectives, injecting fine-grained information in a function-informed manner. Leveraging all these innovations, we develop **ProtCLIP**, a multi-modality foundation model that comprehensively represents function-aware protein embeddings. On 22 different protein benchmarks within 5 types, including protein functionality classification, mutation effect prediction, cross-modal transformation, semantic similarity inference, and protein-protein interaction prediction, **ProtCLIP** consistently achieves SOTA performance, with remarkable improvements of 75% on average in five cross-modal transformation benchmarks, 59.9% in GO-CC, and 39.7% in GO-BP protein function prediction. The experimental results verify the extraordinary potential of **ProtCLIP** as a multi-modality foundation model for proteins.



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



#### **1. 트레이닝 데이터 (Training Data)**  
**ProtAnno 데이터셋**은 단백질 서열과 생물학적 설명을 정렬한 대규모 다중 모달 데이터셋으로, 두 가지 버전으로 구성됩니다:  
- **ProtAnno-S (Sparse)**: 50만 개의 수동 검토된 단백질-텍스트 쌍을 포함하며, 높은 품질의 주석이 제공됨.  
- **ProtAnno-D (Dense)**: 2억 5,150만 개의 단백질-텍스트 쌍으로 이루어진 대규모 데이터셋으로, 대부분 기계로 주석된 데이터이므로 품질이 상대적으로 낮음.  

이 데이터셋을 효과적으로 활용하기 위해 **속성 기반 샘플링 전략**(property-driven sampling strategy)을 도입하여, 샘플 신뢰도(confidence)와 속성 범위(property coverage)를 기준으로 데이터 선택 확률을 조정합니다. 이 방법은 **노이즈가 많은 대규모 데이터에서도 학습 효과를 극대화**할 수 있도록 설계되었습니다.  

---

#### **2. 모델 아키텍처 (Model Architecture)**  
ProtCLIP 모델은 단백질과 생물학적 텍스트 간의 다중 모달 학습을 수행하는 **CLIP 기반 프레임워크**로 구성됩니다.  

- **단백질 인코더 (Protein Encoder)**:  
  - ESM-2-650M(단백질 언어 모델)을 사용하여 단백질 서열에서 생물학적 특징을 학습함.  
- **바이오텍스트 인코더 (Biotext Encoder)**:  
  - PubMed-BERT를 사용하여 생물학적 텍스트에서 언어적 특징을 학습함.  
- **사전 학습 목표 (Pre-Training Objectives)**:  
  - **전역 대비 학습(Global Contrastive Loss, GC)**: 단백질 서열과 생물학적 텍스트 간 정렬을 수행하여 두 모달 간 표현 학습을 강화함.  
  - **정적 세그먼트 복원(Biotext-guided Static Segment Reconstruction, BSR)**: 텍스트 정보를 기반으로 손상된 단백질 정적 세그먼트를 복구하여 보다 정밀한 기능 정보를 학습함.  
  - **동적 세그먼트 정렬(Property-grouped Dynamic Segment Alignment, PDA)**: 특정 기능 속성에 따라 동적 단백질 세그먼트를 그룹화하여, 단백질 구조와 기능적 요소 간의 정렬을 개선함.  
  - **단백질 마스킹 언어 모델링(Protein Masked Language Modeling, MLM)**: 단일 모달 학습을 유지하면서 다중 모달 정보를 주입하기 위해 사용됨.  

최종적으로, ProtCLIP의 전체 손실 함수는 다음과 같이 정의됩니다:  
\[
L = L_{GC} + \lambda_1 L_{BSR} + \lambda_2 L_{MLM} + L_{PDA}
\]  
여기서 \(\lambda_1\)과 \(\lambda_2\)는 가중치 하이퍼파라미터입니다.  

---

#### **3. 학습 환경 및 설정 (Training Setup)**  
- **GPU 사용**: 64개의 Tesla V100 GPU에서 10,000 GPU 시간을 사용하여 학습.  
- **최적화 알고리즘**: Adam 옵티마이저 적용 (학습률 1.0×10⁻⁵, weight decay 0).  
- **배치 크기**:  
  - 사전 학습(batch size = 2048)  
  - 다운스트림 실험(batch size = 512)  
- **하이퍼파라미터 설정**:  
  - \(\lambda_1 = 0.7\), \(\lambda_2 = 0.3\), \(\theta = 0.3\)  

이러한 설정을 통해 ProtCLIP은 단백질 기능 예측, 돌연변이 효과 예측, 의미적 유사성 분석, 교차 모달 변환, 단백질-단백질 상호작용 예측 등의 다양한 벤치마크에서 최고 성능을 달성하였습니다.

---


#### **1. Training Data**  
The **ProtAnno dataset** is a large-scale multi-modal dataset aligning protein sequences with biological descriptions. It has two versions:  
- **ProtAnno-S (Sparse)**: Contains 0.5 million manually reviewed protein-text pairs with high-quality annotations.  
- **ProtAnno-D (Dense)**: Consists of 251.5 million protein-text pairs, mostly machine-annotated, making it relatively noisy.  

To optimize data utilization, the **property-driven sampling strategy** is employed, adjusting data selection probability based on **sample confidence** and **property coverage**. This ensures efficient learning even with noisy large-scale data.  

---

#### **2. Model Architecture**  
ProtCLIP is a **CLIP-based framework** that enables multi-modal learning between protein sequences and biological text descriptions.  

- **Protein Encoder**:  
  - Uses **ESM-2-650M** (a protein language model) to learn biological features from protein sequences.  
- **Biotext Encoder**:  
  - Uses **PubMed-BERT** to capture linguistic features from biological text.  
- **Pre-Training Objectives**:  
  - **Global Contrastive Loss (GC)**: Aligns representations of protein sequences and text to enhance multi-modal learning.  
  - **Biotext-guided Static Segment Reconstruction (BSR)**: Restores corrupted static protein segments based on biotext information, capturing fine-grained function details.  
  - **Property-grouped Dynamic Segment Alignment (PDA)**: Groups dynamic protein segments by functional properties to improve protein structure-function alignment.  
  - **Protein Masked Language Modeling (MLM)**: Preserves unimodal protein language model knowledge while integrating multi-modality information.  

The overall loss function of ProtCLIP is formulated as:  
\[
L = L_{GC} + \lambda_1 L_{BSR} + \lambda_2 L_{MLM} + L_{PDA}
\]  
where \(\lambda_1\) and \(\lambda_2\) are hyperparameters.  

---

#### **3. Training Setup**  
- **GPU Usage**: Training conducted on **64 Tesla V100 GPUs** with a total of **10,000 GPU hours**.  
- **Optimizer**: Adam optimizer with learning rate **1.0×10⁻⁵** and weight decay **0**.  
- **Batch Size**:  
  - **Pre-training**: 2048  
  - **Downstream tasks**: 512  
- **Hyperparameter Settings**:  
  - \(\lambda_1 = 0.7\), \(\lambda_2 = 0.3\), \(\theta = 0.3\)  

With these configurations, **ProtCLIP achieves state-of-the-art performance** across multiple benchmarks, including protein function classification, mutation effect prediction, semantic similarity inference, cross-modal transformation, and protein-protein interaction prediction.



   
 
<br/>
# Results  



#### **1. 경쟁 모델 (Baseline Models)**  
ProtCLIP은 다양한 단백질 기반 머신러닝 모델들과 성능을 비교하였으며, 주요 비교 대상은 다음과 같습니다.  

- **전통적인 단백질 모델**  
  - **CNN** (Shanehsazzadeh et al., 2020): 합성곱 신경망을 이용한 단백질 서열 분석  
  - **LSTM** (Rao et al., 2019): 순환 신경망(RNN)을 활용한 단백질 서열 처리  

- **단일 모달 단백질 언어 모델 (Protein Language Models, PLMs)**  
  - **ProtBERT** (Elnaggar et al., 2022)  
  - **OntoProtein** (Zhang et al., 2022)  
  - **ESM-1b** (Rives et al., 2021)  
  - **ESM2** (Lin et al., 2023)  

- **다중 모달 단백질-텍스트 모델**  
  - **ProtST-ESM2** (Xu et al., 2023)  
  - **BioBridge** (Wang et al., 2024)  

---

#### **2. 테스트 데이터 (Test Datasets)**  
ProtCLIP은 총 **22개 단백질 벤치마크 테스트**에서 평가되었으며, 5가지 주요 작업 유형을 포함합니다.  

1. **단백질 기능 분류 (Protein Classification Engineering)**  
   - **DeepLoc** (Almagro Armenteros et al., 2017): 세포 내 위치 예측 (Subcellular Localization)  
   - **Gene Ontology (GO)** (Consortium, 2019): 유전자 기능 예측  
   - **Enzyme Commission (EC) number prediction** (Gligorijević et al., 2021)  

2. **돌연변이 효과 예측 (Mutation Effect Prediction)**  
   - **β-lactamase (β-lac) landscape** (Xu et al., 2022)  
   - **Fluorescence (Flu) and Stability (Sta)** (Rao et al., 2019)  
   - **AAV and Thermostability (Thermo)** (Dallago et al., 2021)  

3. **교차 모달 변환 (Cross-modal Transformation)**  
   - **Prot2BP, Prot2MF, Prot2CC, Prot2Drug, Disease2Prot** (Wang et al., 2024)  

4. **의미적 유사성 추론 (Semantic Similarity Inference)**  
   - **Manhattan Similarities & Lin Similarities** (Unsal et al., 2022)  

5. **단백질-단백질 상호작용 예측 (Protein-Protein Interaction Prediction)**  
   - **SHS27K, SHS148K, STRING** (Chen et al., 2019)  

---

#### **3. 평가 메트릭 (Evaluation Metrics)**  
각 벤치마크에 따라 적절한 메트릭이 사용되었습니다.  

- **정확도 (Accuracy)**: 단백질 기능 분류 및 위치 예측  
- **AUPR (Area Under the Precision-Recall Curve)**: 기능 예측 성능 평가  
- **Fmax Score**: 다중 클래스 예측 모델 평가  
- **Spearman's ρ (Rank Correlation)**: 돌연변이 효과 예측 및 의미적 유사성 분석  
- **MRR (Mean Reciprocal Rank)**: 교차 모달 변환 성능 평가  
- **F1 Score**: 단백질-단백질 상호작용 예측  

---

#### **4. 실험 결과 (Results)**  
ProtCLIP은 **모든 22개 벤치마크에서 최고 성능(SOTA, State-of-the-Art)을 기록**하였습니다.  

- **단백질 기능 분류**  
  - GO-CC 예측에서 **59.9%**, GO-BP 예측에서 **39.7%** 성능 향상  
- **돌연변이 효과 예측**  
  - 기존 모델 대비 높은 Spearman's ρ 값 유지  
- **교차 모달 변환**  
  - 기존 모델 대비 평균 **75% 성능 향상**  
- **의미적 유사성 추론 및 단백질-단백질 상호작용 예측**  
  - **BioBridge 모델 대비 최고 성능 기록**  

이 실험 결과는 ProtCLIP이 단백질 다중 모달 기초 모델로서 **새로운 표준(SOTA)**을 확립했음을 보여줍니다.  

---



#### **1. Baseline Models**  
ProtCLIP was evaluated against multiple protein-based machine learning models, including:  

- **Traditional Protein Models**  
  - **CNN** (Shanehsazzadeh et al., 2020): Convolutional neural networks for protein sequence analysis  
  - **LSTM** (Rao et al., 2019): Recurrent neural networks for protein sequence processing  

- **Single-Modality Protein Language Models (PLMs)**  
  - **ProtBERT** (Elnaggar et al., 2022)  
  - **OntoProtein** (Zhang et al., 2022)  
  - **ESM-1b** (Rives et al., 2021)  
  - **ESM2** (Lin et al., 2023)  

- **Multi-Modality Protein-Text Models**  
  - **ProtST-ESM2** (Xu et al., 2023)  
  - **BioBridge** (Wang et al., 2024)  

---

#### **2. Test Datasets**  
ProtCLIP was tested on **22 protein benchmarks** across five major task categories.  

1. **Protein Classification Engineering**  
   - **DeepLoc** (Almagro Armenteros et al., 2017): Subcellular localization prediction  
   - **Gene Ontology (GO)** (Consortium, 2019): Gene function prediction  
   - **Enzyme Commission (EC) number prediction** (Gligorijević et al., 2021)  

2. **Mutation Effect Prediction**  
   - **β-lactamase (β-lac) landscape** (Xu et al., 2022)  
   - **Fluorescence (Flu) and Stability (Sta)** (Rao et al., 2019)  
   - **AAV and Thermostability (Thermo)** (Dallago et al., 2021)  

3. **Cross-modal Transformation**  
   - **Prot2BP, Prot2MF, Prot2CC, Prot2Drug, Disease2Prot** (Wang et al., 2024)  

4. **Semantic Similarity Inference**  
   - **Manhattan Similarities & Lin Similarities** (Unsal et al., 2022)  

5. **Protein-Protein Interaction Prediction**  
   - **SHS27K, SHS148K, STRING** (Chen et al., 2019)  

---

#### **3. Evaluation Metrics**  
Each benchmark used appropriate evaluation metrics:  

- **Accuracy**: Protein function classification and localization prediction  
- **AUPR (Area Under the Precision-Recall Curve)**: Function prediction performance  
- **Fmax Score**: Multi-class classification performance  
- **Spearman’s ρ (Rank Correlation)**: Mutation effect prediction and semantic similarity inference  
- **MRR (Mean Reciprocal Rank)**: Cross-modal transformation performance  
- **F1 Score**: Protein-protein interaction prediction  

---

#### **4. Experimental Results**  
ProtCLIP achieved **state-of-the-art (SOTA) performance** across all **22 benchmarks**.  

- **Protein Function Classification**  
  - **GO-CC: 59.9% improvement**, **GO-BP: 39.7% improvement**  
- **Mutation Effect Prediction**  
  - Maintained higher **Spearman’s ρ values** than baseline models  
- **Cross-modal Transformation**  
  - **75% average improvement over previous models**  
- **Semantic Similarity Inference & Protein-Protein Interaction Prediction**  
  - **Outperformed BioBridge, achieving the best performance**  

These results demonstrate that **ProtCLIP establishes a new standard (SOTA) as a multi-modality foundation model for protein analysis.**




<br/>
# 예제  



#### **1. 트레이닝 데이터 예제 (Training Data Example)**  
ProtAnno 데이터셋은 **단백질 서열과 생물학적 텍스트 설명을 정렬한 다중 모달 데이터셋**입니다.  

**예제 데이터 (단백질-텍스트 쌍):**  
| **Protein Sequence**  | **Biotext Description** |
|----------------------|-----------------------|
| `MGLVNGSD...TRIT` | "Pentafunctional AROM polypeptide. FUNCTION: Catalyzes 5 consecutive enzymatic reactions in the aromatic amino acid biosynthesis pathway." |
| `MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR` | "Lactate dehydrogenase enzyme. FUNCTION: Converts pyruvate into lactate during anaerobic respiration." |

---

#### **2. 실험 데이터 예제 (Test Data Example)**  
테스트 데이터는 다양한 벤치마크 작업에서 사용되며, 단백질 서열과 해당 태스크에 필요한 정보가 포함됩니다.  

**예제 데이터 (GO-BP 단백질 기능 예측):**  
| **Protein Sequence**  | **Expected Function (GO Term)** |
|----------------------|-------------------------------|
| `MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR` | GO:0006096 (glycolysis) |
| `MSHSHHAAQQLSAEKPK...` | GO:0046872 (metal ion binding) |

---

#### **3. 태스크 입력/출력 예제 (Task Input/Output Example)**  

**(1) 단백질 기능 분류 (Protein Classification Task)**  
- **입력 (Input)**:  
  ```json
  {
    "protein_sequence": "MGLVNGSDTRIT",
    "task": "protein_function_classification"
  }
  ```
- **출력 (Output)**:  
  ```json
  {
    "predicted_function": "Pentafunctional AROM polypeptide",
    "confidence_score": 0.98
  }
  ```

---

**(2) 돌연변이 효과 예측 (Mutation Effect Prediction Task)**  
- **입력 (Input)**:  
  ```json
  {
    "original_sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR",
    "mutated_sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTA",
    "task": "mutation_effect_prediction"
  }
  ```
- **출력 (Output)**:  
  ```json
  {
    "predicted_fitness_score": 0.72,
    "mutation_effect": "deleterious"
  }
  ```

---

**(3) 교차 모달 변환 (Cross-modal Transformation Task - Protein to Biotext)**  
- **입력 (Input)**:  
  ```json
  {
    "protein_sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR",
    "task": "cross_modal_transformation"
  }
  ```
- **출력 (Output)**:  
  ```json
  {
    "predicted_text": "Lactate dehydrogenase enzyme. FUNCTION: Converts pyruvate into lactate during anaerobic respiration.",
    "confidence_score": 0.91
  }
  ```

---

**(4) 단백질-단백질 상호작용 예측 (Protein-Protein Interaction Prediction Task)**  
- **입력 (Input)**:  
  ```json
  {
    "protein_A": "MGLVNGSDTRIT",
    "protein_B": "MSHSHHAAQQLSAEKPK...",
    "task": "ppi_prediction"
  }
  ```
- **출력 (Output)**:  
  ```json
  {
    "interaction_type": "binding",
    "confidence_score": 0.87
  }
  ```

---

### **English Explanation (Examples - Training Data, Test Data, Task Input/Output Examples)**  

#### **1. Training Data Example**  
The **ProtAnno dataset** consists of **protein sequence and biological text description pairs** for multi-modal learning.  

**Example Data (Protein-Text Pair):**  
| **Protein Sequence**  | **Biotext Description** |
|----------------------|-----------------------|
| `MGLVNGSD...TRIT` | "Pentafunctional AROM polypeptide. FUNCTION: Catalyzes 5 consecutive enzymatic reactions in the aromatic amino acid biosynthesis pathway." |
| `MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR` | "Lactate dehydrogenase enzyme. FUNCTION: Converts pyruvate into lactate during anaerobic respiration." |

---



**Example Data (GO-BP Protein Function Prediction):**  
| **Protein Sequence**  | **Expected Function (GO Term)** |
|----------------------|-------------------------------|
| `MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR` | GO:0006096 (glycolysis) |
| `MSHSHHAAQQLSAEKPK...` | GO:0046872 (metal ion binding) |

---

#### **3. Task Input/Output Examples**  

**(1) Protein Classification Task**  
- **Input:**  
  ```json
  {
    "protein_sequence": "MGLVNGSDTRIT",
    "task": "protein_function_classification"
  }
  ```
- **Output:**  
  ```json
  {
    "predicted_function": "Pentafunctional AROM polypeptide",
    "confidence_score": 0.98
  }
  ```

---

**(2) Mutation Effect Prediction Task**  
- **Input:**  
  ```json
  {
    "original_sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR",
    "mutated_sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTA",
    "task": "mutation_effect_prediction"
  }
  ```
- **Output:**  
  ```json
  {
    "predicted_fitness_score": 0.72,
    "mutation_effect": "deleterious"
  }
  ```

---

**(3) Cross-modal Transformation Task (Protein to Biotext)**  
- **Input:**  
  ```json
  {
    "protein_sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVTR",
    "task": "cross_modal_transformation"
  }
  ```
- **Output:**  
  ```json
  {
    "predicted_text": "Lactate dehydrogenase enzyme. FUNCTION: Converts pyruvate into lactate during anaerobic respiration.",
    "confidence_score": 0.91
  }
  ```

---

**(4) Protein-Protein Interaction Prediction Task**  
- **Input:**  
  ```json
  {
    "protein_A": "MGLVNGSDTRIT",
    "protein_B": "MSHSHHAAQQLSAEKPK...",
    "task": "ppi_prediction"
  }
  ```
- **Output:**  
  ```json
  {
    "interaction_type": "binding",
    "confidence_score": 0.87
  }
  ```

---

These examples illustrate how **ProtCLIP processes and predicts protein properties, mutations, cross-modal transformations, and interactions** in real-world biomedical applications.


<br/>  
# 요약   




ProtCLIP은 단백질 서열과 생물학적 텍스트를 정렬한 대규모 **ProtAnno** 데이터셋을 구축하고, 기능 중심의 사전 학습 기법(정적/동적 세그먼트 모델링)을 적용하여 다중 모달 표현을 학습한다. 이 모델은 22개 단백질 벤치마크에서 **기존 SOTA 모델을 뛰어넘는 성능(교차 모달 변환 75% 향상, GO-CC 59.9% 향상)**을 기록했다. 실험 결과, ProtCLIP은 단백질 기능 예측, 돌연변이 효과 분석, 단백질-단백질 상호작용 등 다양한 생물학적 태스크에서 정확하고 신뢰도 높은 예측을 제공하였다.  

---


ProtCLIP constructs a large-scale **ProtAnno** dataset aligning protein sequences and biological text while incorporating a function-informed pre-training approach (static/dynamic segment modeling) for multi-modal representation learning. The model outperforms existing SOTA methods across **22 protein benchmarks (75% improvement in cross-modal transformation, 59.9% in GO-CC)**. Experimental results demonstrate that ProtCLIP provides accurate and reliable predictions for various biological tasks, including protein function prediction, mutation effect analysis, and protein-protein interaction.


<br/>  
# 기타  




#### **1. ProtCLIP 모델 개요 (Figure: Model Overview)**  
이 다이어그램은 ProtCLIP 모델의 전체 구조를 보여주며, **단백질 서열과 생물학적 텍스트를 정렬하여 다중 모달 학습을 수행하는 과정**을 설명합니다.  
- **단백질 인코더 (Protein Encoder)**: ESM-2-650M을 사용하여 단백질 서열을 임베딩.  
- **바이오텍스트 인코더 (Biotext Encoder)**: PubMed-BERT를 활용하여 생물학적 설명을 임베딩.  
- **전역 대비 학습 (Global Contrastive Loss)**: 단백질과 텍스트의 일관된 표현을 학습.  
- **정적 세그먼트 복원 (BSR) & 동적 세그먼트 정렬 (PDA)**: 단백질 기능을 보다 정밀하게 반영하는 목표를 포함.  
- **최종 출력**: ProtCLIP은 단백질 기능 예측, 돌연변이 효과 분석, 교차 모달 변환 등 다양한 생물학적 태스크를 수행할 수 있음.  

---

#### **2. ProtAnno 데이터셋 샘플링 전략 (Figure: Dataset Sampling Strategy)**  
이 테이블 및 그래프는 **대규모 데이터셋에서 샘플링이 어떻게 이루어지는지**를 시각적으로 표현합니다.  
- ProtAnno-D는 2억 5천만 개의 단백질-텍스트 쌍을 포함하며, 일부 데이터는 노이즈가 포함됨.  
- **속성 기반 샘플링 전략**은 **샘플 신뢰도(confidence)와 속성 범위(property coverage)**를 고려하여, 신뢰도가 낮은 데이터를 필터링하면서도 데이터의 다양성을 유지함.  
- 이 전략을 통해 ProtAnno-S보다 훨씬 많은 데이터를 효과적으로 활용할 수 있음.  

---

#### **3. 실험 결과 비교 테이블 (Tables: Experimental Results Comparison)**  
이 테이블은 **ProtCLIP과 기존 SOTA 모델을 비교한 성능 결과**를 보여줍니다.  
- **단백질 기능 예측 (GO-CC, GO-BP 등)**: ProtCLIP은 기존 모델 대비 59.9% (GO-CC) 및 39.7% (GO-BP) 성능 향상.  
- **교차 모달 변환 (Cross-modal Transformation)**: 기존 모델 대비 평균 75% 향상.  
- **돌연변이 효과 예측 (Mutation Effect Prediction)**: ProtCLIP이 가장 높은 Spearman’s ρ 점수를 기록.  
- **단백질-단백질 상호작용 예측 (PPI Prediction)**: F1 Score 기준으로 BioBridge 모델을 뛰어넘는 성능을 달성.  

---

#### **4. ProtCLIP 학습 과정 시각화 (Figure: Training Process Visualization)**  
이 그래프는 ProtCLIP의 학습 곡선을 보여주며, **훈련 중 손실 감소(loss reduction) 및 평가 메트릭(accuracy, AUPR 등)의 개선 과정**을 나타냅니다.  
- 학습 초기에는 손실 값이 높은 상태에서 시작하지만, 반복 학습이 진행되면서 급격히 감소.  
- 평가 성능(AUPR, F1-score 등)은 사전 학습된 단백질-텍스트 임베딩을 활용한 후 빠르게 향상됨.  
- ProtCLIP의 샘플링 전략과 기능 중심 학습 기법(BSR, PDA 등)이 모델 수렴 속도를 높이고 최종 성능을 향상시킴.  

---



#### **1. ProtCLIP Model Overview (Figure: Model Overview)**  
This diagram illustrates the overall **architecture of the ProtCLIP model**, demonstrating how protein sequences and biological text are aligned for multi-modal learning.  
- **Protein Encoder**: Uses ESM-2-650M to embed protein sequences.  
- **Biotext Encoder**: Uses PubMed-BERT to embed biological text descriptions.  
- **Global Contrastive Loss**: Ensures consistent representation between protein and text embeddings.  
- **Biotext-guided Static Segment Reconstruction (BSR) & Property-grouped Dynamic Segment Alignment (PDA)**: Incorporates fine-grained function-specific modeling.  
- **Final Output**: ProtCLIP enables protein function prediction, mutation effect analysis, and cross-modal transformations.  

---

#### **2. ProtAnno Dataset Sampling Strategy (Figure: Dataset Sampling Strategy)**  
This table and graph visualize **how the dataset is sampled from a large noisy dataset**.  
- **ProtAnno-D** consists of **251.5 million** protein-text pairs, but some contain noisy annotations.  
- The **property-driven sampling strategy** selects data based on **sample confidence and property coverage**, ensuring high-quality and diverse data representation.  
- This method allows **ProtAnno-D to be leveraged more effectively** than the smaller, manually reviewed **ProtAnno-S dataset**.  

---

#### **3. Experimental Results Comparison Tables (Tables: Experimental Results Comparison)**  
This table compares **ProtCLIP’s performance with existing SOTA models** across multiple benchmarks.  
- **Protein Function Prediction (GO-CC, GO-BP, etc.)**: **59.9% (GO-CC) and 39.7% (GO-BP) improvement** over previous models.  
- **Cross-modal Transformation**: **75% improvement** on average over prior methods.  
- **Mutation Effect Prediction**: ProtCLIP achieves the **highest Spearman’s ρ score** across datasets.  
- **Protein-Protein Interaction Prediction (PPI)**: **Outperforms BioBridge** based on F1 Score.  

---

#### **4. ProtCLIP Training Process Visualization (Figure: Training Process Visualization)**  
This graph displays the **training loss reduction and evaluation metric improvements** over time.  
- Initially, the **loss is high**, but it decreases significantly as training progresses.  
- **Performance metrics (AUPR, F1-score, etc.) improve rapidly** after utilizing pre-trained protein-text embeddings.  
- ProtCLIP’s **sampling strategy and function-informed learning techniques (BSR, PDA, etc.) accelerate convergence and boost final accuracy**.  


<br/>
# refer format:     



@article{zhou2025protclip,
  author    = {Hanjing Zhou and Mingze Yin and Wei Wu and Mingyang Li and Kun Fu and Jintai Chen and Jian Wu and Zheng Wang},
  title     = {ProtCLIP: Function-Informed Protein Multi-Modal Learning},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)}
}   



Hanjing Zhou, Mingze Yin, Wei Wu, Mingyang Li, Kun Fu, Jintai Chen, Jian Wu, and Zheng Wang. "ProtCLIP: Function-Informed Protein Multi-Modal Learning." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. Association for the Advancement of Artificial Intelligence (AAAI).   





