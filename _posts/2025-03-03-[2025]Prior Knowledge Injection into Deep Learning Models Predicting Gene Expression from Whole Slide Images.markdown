---
layout: post
title:  "[2025]Prior Knowledge Injection into Deep Learning Models Predicting Gene Expression from Whole Slide Images"  
date:   2025-03-03 10:13:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

이미지 to RNA...굿(포인트는 사전지식 추가로 성능향상이지만 무튼)     

짧은 요약(Abstract) :    




이 논문은 종양 진단과 예후 예측을 위해 유전자 발현 데이터를 활용하는 과정에서 발생하는 비용과 시간적 제약을 극복하기 위해, Whole Slide Images(WSI)에서 직접 유전자 발현을 예측하는 딥러닝 모델을 개선하는 방법을 제안합니다. 기존 모델들은 유전자 발현을 예측하는 데 성공적이지만, RNA 시퀀싱을 완전히 대체하기에는 부족한 점이 있습니다. 이를 해결하기 위해, 저자들은 유전자 간 상호작용에 대한 사전 지식을 딥러닝 모델에 주입할 수 있는 프레임워크를 제안했습니다. 

이 프레임워크는 모델에 독립적(model-agnostic)이며, 다양한 딥러닝 구조에 유연하게 적용될 수 있도록 설계되었습니다. 유방암 데이터에 대한 실험에서, 이 방법을 적용한 결과, 평균적으로 25,761개의 유전자 중 983개의 유전자 예측 성능이 향상되었으며, 18개의 실험 중 14개에서 독립적인 데이터셋에서도 유의미한 개선이 이루어졌습니다. 이를 통해, 사전 지식 주입이 WSI 기반 유전자 발현 예측 모델의 정확도와 견고성을 향상시키는 데 효과적임을 입증하였습니다.

---



Cancer diagnosis and prognosis rely on clinical parameters such as age and tumor grade and increasingly incorporate molecular data, such as gene expression from tumor sequencing. However, sequencing is costly and delays oncology workflows. Recent advances in Deep Learning enable the prediction of molecular information from morphological features within Whole Slide Images (WSIs), offering a cost-effective alternative to direct sequencing. 

While promising, current methods lack robustness to fully replace sequencing. To improve existing approaches, we introduce a model-agnostic framework that allows the injection of prior knowledge on gene-gene interactions into Deep Learning architectures, thereby enhancing accuracy and robustness. Our framework is designed to be generic and adaptable across various architectures. 

In a breast cancer case study, our strategy led to an average increase of 983 significant genes (out of 25,761) across all 18 experiments, with 14 generalizing to an independent dataset. Our findings demonstrate the high potential of prior knowledge injection in improving gene expression prediction performance from WSIs across diverse architectures.



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





이 연구에서는 Whole Slide Images(WSI)에서 유전자 발현을 예측하는 딥러닝 모델의 성능을 향상시키기 위해 **사전 지식(Prior Knowledge, PK)을 주입하는 모델-독립적(model-agnostic) 프레임워크**를 제안했습니다.  

#### **1. 모델 및 아키텍처**
WSI 기반 유전자 발현 예측 모델은 일반적으로 **WSI 전처리 → 패치(feature) 추출 → WSI 임베딩 생성 → 유전자 발현 예측**의 단계를 거칩니다. 본 연구에서는 이러한 모델을 개선하기 위해 **유전자 간 상호작용에 대한 사전 지식을 주입하는 새로운 구조를 도입**했습니다.  

- **Feature Extractors (특징 추출기)**:  
  - **CTransPath** [(Wang et al., 2022)](https://arxiv.org/abs/2202.06645)  
  - **UNI** [(Chen et al., 2024)](https://www.nature.com/articles/s41591-024-02974-9)  

- **Patch Embedding Aggregators (패치 임베딩 집계 방식)**:  
  - **MLP (Multi-Layer Perceptron)** [(Schmauch et al., 2020)](https://www.nature.com/articles/s41467-020-17155-6)  
  - **Transformer (Self-Attention)** [(Alsaafin et al., 2023)](https://www.nature.com/articles/s41467-023-42010-9)  
  - **SummaryMixing (Linearized Transformer)** [(Parcollet et al., 2024)](https://arxiv.org/abs/2307.07421)  

이러한 2가지 **특징 추출기**와 3가지 **패치 임베딩 집계 방법**을 조합하여 총 **6가지 모델 아키텍처**를 실험했습니다.

#### **2. 사전 지식 주입 방법**
- **유전자 간 상호작용 데이터(사전 지식)**를 **비음수 행렬 분해(Nonnegative Matrix Factorization, NMF)** 기법을 활용해 저차원 임베딩으로 변환함.
- 학습된 **유전자 임베딩(G)**을 WSI에서 추출한 특징 벡터(embedding w)와 결합하여 **유전자 발현 예측 레이어**(fpk)에서 활용함.
- 기존 선형 예측 레이어 \( f_{linear} = wA^T + b \)를 수정하여, **사전 지식이 반영된 예측 레이어** \( f_{pk} \)를 구성함:  
  \[
  g_{pk} = (1 - \lambda) w A^T + \lambda w G^T + b
  \]  
  - **λ** 값(0.1~0.9)은 사전 지식이 예측에 미치는 영향을 조절함.  
  - **유전자 간 유사한 발현 패턴을 갖는 경우, 유사한 예측을 하도록 유도**하여 모델의 견고성을 향상시킴.

#### **3. 학습 데이터 및 평가 방법**
- **Train Dataset**:  
  - **TCGA-BRCA** (The Cancer Genome Atlas - Breast Cancer)  
  - **1133개**의 유방암 환자 WSI 및 bulk RNA-Seq 유전자 발현 데이터  
  - **5-Fold Cross Validation** 적용  

- **Independent Test Dataset**:  
  - **CPTAC-BRCA** (Clinical Proteomic Tumor Analysis Consortium - Breast Cancer)  
  - **106개**의 유방암 환자 WSI  

- **모델 평가 기준**:  
  - **MSE (Mean Squared Error)**를 손실 함수로 사용  
  - **예측된 유전자 발현 값과 실제 발현 값 간 Pearson Correlation** 계산  
  - 모델이 예측한 유전자 중 **통계적으로 유의미한 유전자 개수** 측정  

#### **4. 실험 결과**  
- TCGA 데이터에서 **모든 사전 지식(PK) 소스**가 유전자 예측 성능을 향상시킴.
  - **External PK**: 평균 **1,150개** 유전자 증가  
  - **Internal PK**: 평균 **891개** 유전자 증가  
  - **Combined PK**: 평균 **908개** 유전자 증가  
- CPTAC 데이터에서는 **UNI 기반 모델이 가장 일반화 성능이 높음**.  

결론적으로, **사전 지식 주입이 딥러닝 모델의 유전자 발현 예측 정확도를 높이고 일반화 성능을 강화하는 데 기여할 수 있음**을 입증했습니다.

---



This study proposes a **model-agnostic framework** to enhance the performance of deep learning models in predicting gene expression from Whole Slide Images (WSIs) by incorporating **prior knowledge (PK) on gene-gene interactions**.

#### **1. Model and Architecture**
WSI-based gene expression prediction models typically follow a workflow of **WSI preprocessing → feature extraction → WSI embedding generation → gene expression prediction**. To improve this process, the authors introduce a **novel approach that injects prior knowledge on gene interactions** into the model.  

- **Feature Extractors**:  
  - **CTransPath** [(Wang et al., 2022)](https://arxiv.org/abs/2202.06645)  
  - **UNI** [(Chen et al., 2024)](https://www.nature.com/articles/s41591-024-02974-9)  

- **Patch Embedding Aggregators**:  
  - **MLP (Multi-Layer Perceptron)** [(Schmauch et al., 2020)](https://www.nature.com/articles/s41467-020-17155-6)  
  - **Transformer (Self-Attention)** [(Alsaafin et al., 2023)](https://www.nature.com/articles/s41467-023-42010-9)  
  - **SummaryMixing (Linearized Transformer)** [(Parcollet et al., 2024)](https://arxiv.org/abs/2307.07421)  

The study evaluates **six different model architectures** combining **two feature extractors** and **three aggregation methods**.

#### **2. Prior Knowledge Injection Method**
- **Gene-gene interaction data** is transformed into **low-dimensional embeddings** using **Nonnegative Matrix Factorization (NMF)**.
- The **learned gene embeddings (G)** are injected into the **final predictor layer** of the model.
- The standard **linear predictor layer** \( f_{linear} = wA^T + b \) is modified to incorporate prior knowledge:  
  \[
  g_{pk} = (1 - \lambda) w A^T + \lambda w G^T + b
  \]
  - **λ (0.1–0.9)** controls the influence of prior knowledge.  
  - **Genes with similar expression profiles** are guided to have similar predictions, improving robustness.

#### **3. Training Data and Evaluation**
- **Training Dataset**:  
  - **TCGA-BRCA** (The Cancer Genome Atlas - Breast Cancer)  
  - **1133 WSIs** annotated with bulk RNA-Seq gene expression  
  - **5-Fold Cross Validation**  

- **Independent Test Dataset**:  
  - **CPTAC-BRCA** (Clinical Proteomic Tumor Analysis Consortium - Breast Cancer)  
  - **106 WSIs**  

- **Evaluation Metrics**:  
  - **Mean Squared Error (MSE)** as the loss function  
  - **Pearson correlation** between predicted and ground truth gene expression values  
  - **Number of significantly predicted genes**  

#### **4. Experimental Results**  
- On **TCGA-BRCA**, **all prior knowledge sources improved gene expression prediction**:
  - **External PK**: +1,150 genes on average  
  - **Internal PK**: +891 genes  
  - **Combined PK**: +908 genes  
- On **CPTAC-BRCA**, **UNI-based models demonstrated superior generalization**.  

In conclusion, **injecting prior knowledge into deep learning models can enhance gene expression prediction accuracy and robustness across diverse architectures**.


   
 
<br/>
# Results  





이 연구에서는 **Whole Slide Image(WSI) 기반 유전자 발현 예측 모델**의 성능을 향상시키기 위해 사전 지식(Prior Knowledge, PK)을 주입하는 프레임워크를 제안하고, 이를 다양한 실험을 통해 평가하였습니다.  

#### **1. 경쟁 모델 및 실험 조건**  
본 연구에서 제안한 **PK 주입 모델**은 기존의 **PK를 사용하지 않은 모델(No PK)** 및 다양한 **경쟁 모델**과 성능을 비교하였습니다.  

- **비교 모델**:  
  - **기존 WSI 기반 유전자 발현 예측 모델 (No PK)**  
  - **PK를 주입한 3가지 모델**  
    - **External PK**: 외부 데이터 기반 유전자 네트워크  
    - **Internal PK**: TCGA-BRCA 데이터에서 직접 추출한 유전자 네트워크  
    - **Combined PK**: External + Internal PK 결합  

- **실험 설정**:  
  - **2가지 Feature Extractors**  
    - **CTransPath** [(Wang et al., 2022)](https://arxiv.org/abs/2202.06645)  
    - **UNI** [(Chen et al., 2024)](https://www.nature.com/articles/s41591-024-02974-9)  
  - **3가지 Patch Embedding Aggregators**  
    - **MLP (Multi-Layer Perceptron)**  
    - **Transformer (Self-Attention)**  
    - **SummaryMixing (Linearized Transformer)**  
  - **총 6가지 모델 아키텍처**를 구성하여 비교  

---

#### **2. 테스트 데이터 및 실험 방법**
- **TCGA-BRCA (Train/Validation/Test)**  
  - **The Cancer Genome Atlas - Breast Cancer**  
  - **1133개의 Whole Slide Images (WSIs)**  
  - **5-Fold Cross Validation 적용**  

- **CPTAC-BRCA (Independent Test Set)**  
  - **Clinical Proteomic Tumor Analysis Consortium - Breast Cancer**  
  - **106개의 독립적인 WSI 데이터**  
  - TCGA-BRCA에서 학습한 모델을 활용하여 일반화 성능 평가  

---

#### **3. 평가 메트릭 (Metrics)**
모델 성능 평가는 **유전자 발현 예측의 정확성 및 모델의 일반화 성능**을 중심으로 진행되었습니다.

1. **Mean Squared Error (MSE)**:  
   - 손실 함수로 사용하여 예측값과 실제값 간 차이를 최소화  
2. **Pearson Correlation**:  
   - 예측된 유전자 발현 값과 실제 발현 값 사이의 상관관계 평가  
3. **Significant Gene Count**:  
   - 통계적으로 유의미한 유전자 개수 측정  
   - 특정 상관계수 기준을 만족하는 유전자를 카운트하여 비교  

---

#### **4. 실험 결과 (Results)**
- **TCGA-BRCA에서 모든 PK 모델이 성능 향상**  
  - No PK 대비 **External PK: +1,150개 유전자**, **Internal PK: +891개**, **Combined PK: +908개**  
  - UNI + SummaryMixing + Internal PK 모델이 **최고 성능(23,732개 유전자)**  

- **CPTAC-BRCA에서 UNI 기반 모델이 가장 높은 일반화 성능**  
  - **UNI + SummaryMixing + External PK 모델(17,280개 유전자)**가 가장 높은 성능  
  - CTrans 기반 모델은 CPTAC-BRCA에서 성능 저하 발생  

- **UNI 모델이 전반적으로 가장 강력한 Feature Extractor**  
  - **UNI 모델이 CTrans 모델보다 더 많은 유전자 예측**  
  - **SummaryMixing + UNI** 조합이 가장 좋은 성능을 보임  

---

#### **5. 결론**
- **PK 주입을 통해 기존 모델 대비 유전자 예측 성능을 향상**  
- **UNI 기반 모델이 가장 안정적이고 일반화 성능이 우수**  
- **PK 주입이 다양한 딥러닝 모델 아키텍처에서 효과적으로 작용**  
- **TCGA-BRCA와 CPTAC-BRCA 간 일반화 성능이 높은 모델 구조 확인**  

이를 통해 **PK 주입이 WSI 기반 유전자 발현 예측 모델의 정확도와 견고성을 향상시키는 강력한 방법임을 입증**했습니다.

---



This study evaluates the **prior knowledge (PK) injection framework** for improving **gene expression prediction from Whole Slide Images (WSIs)** by comparing it against multiple competing models.

#### **1. Competing Models and Experimental Setup**  
The proposed **PK-injected model** is compared with **models without PK (No PK)** and various **competing models**.

- **Compared Models**:  
  - **Baseline WSI-based gene expression prediction model (No PK)**  
  - **PK-Injected Models**:  
    - **External PK**: Gene networks derived from external datasets  
    - **Internal PK**: Gene networks extracted from TCGA-BRCA  
    - **Combined PK**: Merging External and Internal PK  

- **Experimental Setup**:  
  - **Feature Extractors**:  
    - **CTransPath** [(Wang et al., 2022)](https://arxiv.org/abs/2202.06645)  
    - **UNI** [(Chen et al., 2024)](https://www.nature.com/articles/s41591-024-02974-9)  
  - **Patch Embedding Aggregators**:  
    - **MLP (Multi-Layer Perceptron)**  
    - **Transformer (Self-Attention)**  
    - **SummaryMixing (Linearized Transformer)**  
  - **A total of six model architectures** were tested.  

---

#### **2. Test Data and Experimental Procedure**
- **TCGA-BRCA (Train/Validation/Test Set)**  
  - **The Cancer Genome Atlas - Breast Cancer**  
  - **1133 WSIs**  
  - **5-Fold Cross Validation applied**  

- **CPTAC-BRCA (Independent Test Set)**  
  - **Clinical Proteomic Tumor Analysis Consortium - Breast Cancer**  
  - **106 independent WSIs**  
  - Evaluated generalization performance using models trained on TCGA-BRCA  

---

#### **3. Evaluation Metrics**
The performance of each model was assessed based on **gene expression prediction accuracy and generalization ability**.

1. **Mean Squared Error (MSE)**:  
   - Used as the loss function to minimize prediction error  
2. **Pearson Correlation**:  
   - Measures correlation between predicted and actual gene expression values  
3. **Significant Gene Count**:  
   - Counts genes that meet statistical significance criteria for accurate prediction  

---

#### **4. Experimental Results**
- **PK Injection Improved Performance on TCGA-BRCA**  
  - Compared to No PK: **External PK: +1,150 genes**, **Internal PK: +891 genes**, **Combined PK: +908 genes**  
  - Best performance: **UNI + SummaryMixing + Internal PK (23,732 significant genes)**  

- **UNI-Based Models Showed Strongest Generalization on CPTAC-BRCA**  
  - **UNI + SummaryMixing + External PK (17,280 genes) achieved the best performance**  
  - **CTrans-based models showed performance degradation on CPTAC-BRCA**  

- **UNI Model Outperformed CTrans as a Feature Extractor**  
  - **UNI models consistently predicted more genes than CTrans**  
  - **SummaryMixing + UNI was the most effective combination**  

---

#### **5. Conclusion**
- **PK injection improves gene expression prediction performance over baseline models**  
- **UNI-based models demonstrated the best stability and generalization performance**  
- **PK injection enhances deep learning models across diverse architectures**  
- **Strong generalization observed between TCGA-BRCA and CPTAC-BRCA datasets**  

Overall, **this study validates PK injection as a powerful method for improving gene expression prediction accuracy and robustness in WSI-based models**.


<br/>
# 예제  




이 연구에서는 **TCGA-BRCA** 데이터를 학습용(Train) 및 검증용(Validation)으로 사용하고, **CPTAC-BRCA** 데이터를 독립적인 테스트(Test) 데이터셋으로 활용하였습니다. 각각의 데이터는 **Whole Slide Images(WSIs)** 및 **RNA-Seq 유전자 발현 데이터**로 구성되어 있습니다.

---

#### **1. 학습 데이터 예제 (Train Dataset - TCGA-BRCA)**  
**데이터 구성:**
- **Whole Slide Image (WSI)**: 환자의 병리 이미지  
- **Gene Expression (RNA-Seq)**: 해당 환자의 유전자 발현 값 (Normalized TPM)  
- **Feature Extracted Embeddings**: WSI에서 추출한 특징 벡터  

| Sample ID | WSI File Name                  | Gene 1 Expression | Gene 2 Expression | Gene 3 Expression | ... | Gene 25,761 Expression |
|-----------|--------------------------------|-------------------|-------------------|-------------------|-----|----------------------|
| TCGA-01   | TCGA_01_slide.svs              | 12.34             | 5.67              | 8.90              | ... | 3.45                 |
| TCGA-02   | TCGA_02_slide.svs              | 11.23             | 4.56              | 7.89              | ... | 2.34                 |
| TCGA-03   | TCGA_03_slide.svs              | 10.12             | 3.45              | 6.78              | ... | 1.23                 |

---

#### **2. 테스트 데이터 예제 (Test Dataset - CPTAC-BRCA)**
**데이터 구성:**
- **독립적인 데이터셋**으로 사용되며, TCGA-BRCA 학습 데이터를 기반으로 모델이 새로운 데이터를 얼마나 잘 예측하는지 평가하는 데 활용됨.

| Sample ID | WSI File Name                  | Gene 1 Expression | Gene 2 Expression | Gene 3 Expression | ... | Gene 25,761 Expression |
|-----------|--------------------------------|-------------------|-------------------|-------------------|-----|----------------------|
| CPTAC-01  | CPTAC_01_slide.svs             | 9.87              | 6.54              | 8.76              | ... | 4.56                 |
| CPTAC-02  | CPTAC_02_slide.svs             | 8.76              | 5.43              | 7.65              | ... | 3.45                 |
| CPTAC-03  | CPTAC_03_slide.svs             | 7.65              | 4.32              | 6.54              | ... | 2.34                 |

---

#### **3. 사전 지식 데이터 예제 (Prior Knowledge - Gene Interaction Network)**
본 연구에서는 **유전자 간 상호작용 네트워크**를 사전 지식으로 활용하여, WSI에서 예측된 유전자 발현 값이 실제 생물학적 관계를 반영하도록 모델을 보정하였습니다.

| Gene 1 | Gene 2 | Correlation (Pearson r) |
|--------|--------|------------------------|
| BRCA1  | TP53   | 0.87                   |
| ERBB2  | MYC    | 0.76                   |
| EGFR   | KRAS   | 0.91                   |

- 특정 유전자 쌍 간의 **상관관계(Pearson correlation)** 값을 활용하여 **유전자 임베딩(Gene Embedding)**을 생성하고, 이를 모델 학습 과정에 주입함.

---



This study utilized **TCGA-BRCA** as the training and validation dataset, while **CPTAC-BRCA** was used as an independent test dataset. Each dataset consists of **Whole Slide Images (WSIs)** and **RNA-Seq gene expression data**.

---

#### **1. Training Data Example (TCGA-BRCA)**
**Data Composition:**
- **Whole Slide Image (WSI)**: Pathology image of a patient  
- **Gene Expression (RNA-Seq)**: Normalized TPM values for gene expression  
- **Feature Extracted Embeddings**: Feature vectors extracted from WSI  

| Sample ID | WSI File Name                  | Gene 1 Expression | Gene 2 Expression | Gene 3 Expression | ... | Gene 25,761 Expression |
|-----------|--------------------------------|-------------------|-------------------|-------------------|-----|----------------------|
| TCGA-01   | TCGA_01_slide.svs              | 12.34             | 5.67              | 8.90              | ... | 3.45                 |
| TCGA-02   | TCGA_02_slide.svs              | 11.23             | 4.56              | 7.89              | ... | 2.34                 |
| TCGA-03   | TCGA_03_slide.svs              | 10.12             | 3.45              | 6.78              | ... | 1.23                 |

---

#### **2. Test Data Example (CPTAC-BRCA)**
**Data Composition:**
- **Independent dataset** used to evaluate how well the model generalizes to new data.

| Sample ID | WSI File Name                  | Gene 1 Expression | Gene 2 Expression | Gene 3 Expression | ... | Gene 25,761 Expression |
|-----------|--------------------------------|-------------------|-------------------|-------------------|-----|----------------------|
| CPTAC-01  | CPTAC_01_slide.svs             | 9.87              | 6.54              | 8.76              | ... | 4.56                 |
| CPTAC-02  | CPTAC_02_slide.svs             | 8.76              | 5.43              | 7.65              | ... | 3.45                 |
| CPTAC-03  | CPTAC_03_slide.svs             | 7.65              | 4.32              | 6.54              | ... | 2.34                 |

---

#### **3. Prior Knowledge Data Example (Gene Interaction Network)**
This study leveraged **gene-gene interaction networks** as prior knowledge to refine predictions by ensuring that WSI-based gene expression estimates align with known biological relationships.

| Gene 1 | Gene 2 | Correlation (Pearson r) |
|--------|--------|------------------------|
| BRCA1  | TP53   | 0.87                   |
| ERBB2  | MYC    | 0.76                   |
| EGFR   | KRAS   | 0.91                   |

- **Pearson correlation** values between gene pairs were used to generate **gene embeddings**, which were incorporated into the model during training.


<br/>  
# 요약   




이 연구는 **Whole Slide Images(WSIs)에서 유전자 발현을 예측하는 딥러닝 모델**을 개선하기 위해 **유전자 간 상호작용 정보를 사전 지식(Prior Knowledge, PK)으로 주입하는 프레임워크**를 제안하였다. **TCGA-BRCA 데이터로 학습한 모델을 CPTAC-BRCA 데이터에서 평가한 결과, PK를 적용한 모델이 기존 모델보다 더 많은 유전자를 정확하게 예측**하였으며, 특히 **UNI + SummaryMixing + External PK 모델이 최상의 일반화 성능을 보였다**. 예제 데이터는 **WSI 이미지 파일, RNA-Seq 기반 유전자 발현 값, 유전자 간 상관관계(Pearson r) 데이터를 포함**하며, 사전 지식 주입을 통해 모델의 예측 성능을 향상시켰음을 입증하였다.  

---

This study proposes a **framework for injecting prior knowledge (PK) of gene-gene interactions into deep learning models** to improve **gene expression prediction from Whole Slide Images (WSIs)**. **Models trained on TCGA-BRCA and tested on CPTAC-BRCA showed improved prediction accuracy with PK**, with **UNI + SummaryMixing + External PK achieving the best generalization performance**. The dataset includes **WSI image files, RNA-Seq gene expression values, and Pearson correlation-based gene interactions**, demonstrating that PK injection enhances prediction accuracy.


<br/>  
# 기타  



#### **1. Figure 1: 개념적 개요 (High-Level Overview)**  
- 이 그림은 **사전 지식(Prior Knowledge, PK) 주입 방법의 개념적인 흐름**을 보여줌.  
- **유전자 간 상관관계 정보**를 활용하여 특정 유전자(g₂)가 높게 발현될 경우, **연관된 유전자(g₁, g₃)도 높은 발현을 보이도록 모델이 학습되도록 유도**.  
- 기존 딥러닝 모델이 개별 유전자만 예측하는 것과 달리, **PK를 통해 유전자 발현 간 관계를 반영하여 더 정교한 예측을 가능하게 함**.  

---

#### **2. Figure 2: WSI 기반 유전자 발현 예측 모델의 일반적인 워크플로우 (General Workflow of WSI-Based Gene Expression Prediction Models)**  
- **WSI(Whole Slide Image)를 입력으로 사용하여 패치를 추출하고, 개별 패치를 특징 벡터로 변환**한 후, 이를 종합하여 WSI 전체의 특징 임베딩을 생성.  
- **이 임베딩을 바탕으로 유전자 발현 예측을 수행**하는 구조로 이루어짐.  
- 연구에서는 이러한 기본 모델을 개선하기 위해 **PK 정보를 Predictor 레이어에 주입하는 방식**을 적용.  

---

#### **3. Figure 3: 제안된 PK 주입 프레임워크 (Proposed PK Injection Framework)**  
- **PK 데이터를 활용하여 유전자 임베딩(G)을 생성하고, 이를 WSI 임베딩(w)과 결합하여 최종 유전자 발현 예측을 수행**.  
- PK 주입을 반영한 새로운 예측 레이어 \( f_{pk} \)는 기존 예측 레이어 \( f_{linear} \)의 확장형으로, **PK 적용 정도를 조절하는 하이퍼파라미터(λ)를 포함**.  
- **이를 통해 모델이 유전자 간 상관관계를 학습하도록 유도하고, 예측 성능 및 견고성을 향상**.  

---

#### **4. Table 1: 유전자 간 상호작용 네트워크 개요 (Overview of Gene-Gene Interaction Networks Used as Prior Knowledge)**  
| 네트워크 유형 | 유전자 개수 | 상관관계 있는 유전자 쌍 (Co-expressed Pairs) |
|--------------|------------|----------------------------|
| **External** | 4,646      | 41,672                     |
| **Internal** | 1,952      | 48,858                     |
| **Combined** | 6,021      | 88,402                     |

- **External PK**는 **일반적인 유전자 네트워크 데이터**를 기반으로 구축됨.  
- **Internal PK**는 **TCGA-BRCA 데이터셋에서 직접 추출된 유전자 간 상관관계 정보**를 활용.  
- **Combined PK**는 **External과 Internal을 결합하여 더 광범위한 유전자 관계를 반영**.  

---

#### **5. Table 2: TCGA-BRCA 및 CPTAC-BRCA에서 모델 성능 비교 (Performance Comparison Across TCGA-BRCA and CPTAC-BRCA Datasets)**  
| Feature Extractor | Aggregator | TCGA No PK | TCGA External PK | TCGA Internal PK | TCGA Combined PK | CPTAC No PK | CPTAC External PK | CPTAC Internal PK | CPTAC Combined PK |
|------------------|-----------|------------|-----------------|-----------------|-----------------|------------|-----------------|-----------------|-----------------|
| **CTrans + MLP** |          | 21,233     | 22,225 (+992)  | 22,278 (+1,045) | 22,160 (+927)  | 16,936     | 16,313 (-623)  | 18,116 (+1,180) | 17,363 (+427)  |
| **CTrans + TF**  |          | 19,155     | 21,647 (+2,429) | 20,618 (+1,463) | 20,548 (+1,393) | 15,677     | 15,146 (-531)  | 14,983 (-694)  | 14,386 (-1,291) |
| **CTrans + SMX** |          | 20,945     | 22,564 (+1,619) | 21,451 (+506)   | 21,944 (+999)   | 15,714     | 16,682 (+968)  | 15,731 (+17)   | 15,753 (+39)   |
| **UNI + MLP**    |          | 21,721     | 22,666 (+945)   | 22,802 (+1,081) | 23,214 (+1,493) | 15,560     | 16,106 (+546)  | 16,045 (+485)  | 15,784 (+224)  |
| **UNI + TF**     |          | 22,124     | 22,461 (+337)   | 22,645 (+521)   | 22,597 (+473)   | 14,705     | 15,648 (+763)  | 15,469 (+764)  | 15,400 (+695)  |
| **UNI + SMX**    |          | 22,997     | 23,578 (+581)   | 23,732 (+735)   | 23,162 (+165)   | 16,952     | 17,280 (+328)  | 17,091 (+139)  | 16,981 (+29)   |

- **TCGA 데이터에서 모든 PK 모델이 성능 향상**을 보임.
- **UNI + SummaryMixing + Internal PK 모델(23,732 유전자 예측)**이 최고 성능을 기록.
- **CPTAC 데이터에서는 UNI 기반 모델이 가장 높은 일반화 성능을 보임**, 특히 **UNI + SummaryMixing + External PK 모델(17,280 유전자 예측)**이 가장 우수.

---



#### **1. Figure 1: High-Level Overview**  
- This diagram illustrates **the conceptual framework for prior knowledge (PK) injection**.  
- **Gene-gene interactions** are incorporated so that if a particular gene (g₂) has high expression, **associated genes (g₁, g₃) are encouraged to have high expression as well**.  
- Unlike conventional deep learning models that predict genes individually, **this PK approach enables more biologically relevant predictions**.  

---

#### **2. Figure 2: General Workflow of WSI-Based Gene Expression Prediction Models**  
- **WSIs are processed by extracting image patches, generating feature vectors, and aggregating them into a WSI embedding**.  
- **This embedding is used to predict gene expression levels**.  
- The proposed framework **injects PK directly into the predictor layer** to improve performance.  

---

#### **3. Figure 3: Proposed PK Injection Framework**  
- **PK is transformed into gene embeddings (G) and incorporated into the final predictor layer**.  
- The modified predictor layer \( f_{pk} \) extends the conventional linear predictor \( f_{linear} \) by **controlling PK influence using the hyperparameter λ**.  
- **This improves model robustness by leveraging gene-gene interaction information**.  

---

#### **4. Table 1: Overview of Gene-Gene Interaction Networks Used as Prior Knowledge**  
| Network Type | Number of Genes | Co-expressed Pairs |
|-------------|---------------|-------------------|
| **External** | 4,646         | 41,672           |
| **Internal** | 1,952         | 48,858           |
| **Combined** | 6,021         | 88,402           |

- **External PK**: Derived from general gene network datasets.  
- **Internal PK**: Extracted from TCGA-BRCA dataset.  
- **Combined PK**: Merges External and Internal PK.  

---

#### **5. Table 2: Performance Comparison Across TCGA-BRCA and CPTAC-BRCA Datasets**  
- **PK injection improved prediction performance across all TCGA models**.  
- **Best-performing model: UNI + SummaryMixing + Internal PK (23,732 genes predicted)**.  
- **UNI models demonstrated superior generalization on CPTAC**, with **UNI + SummaryMixing + External PK achieving the highest performance (17,280 genes predicted)**.


<br/>
# refer format:     



@article{Hallemeesch2025,
  author    = {Max Hallemeesch and Marija Pizurica and Paloma Rabaey and Olivier Gevaert and Thomas Demeester and Kathleen Marchal},
  title     = {Prior Knowledge Injection into Deep Learning Models Predicting Gene Expression from Whole Slide Images},
  journal   = {Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI)},
  year      = {2025},
  url       = {https://github.com/MaxHallemeesch/PRALINE}
}




Hallemeesch, Max, Marija Pizurica, Paloma Rabaey, Olivier Gevaert, Thomas Demeester, and Kathleen Marchal. "Prior Knowledge Injection into Deep Learning Models Predicting Gene Expression from Whole Slide Images." Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI), 2025. Accessed March 3, 2025. 




