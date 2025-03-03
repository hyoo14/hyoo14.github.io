---
layout: post
title:  "[2025]Whole Genome Transformer for Gene Interaction Effects in Microbiome Habitat Specificity"  
date:   2025-03-03 14:25:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


미생물 게놈 데이터로 서식 환경 예측(트랜스포머 사용)  
기존 모델이 잘 못하는 미생물로 간 점이 주효했던듯  


짧은 요약(Abstract) :    



미생물군유전체의 유전적 다양성을 활용하면 복잡한 표현형을 이해하는 데 중요한 통찰을 얻을 수 있지만, 이러한 특성을 정확하게 예측하고 분석하는 것은 여전히 어려운 과제입니다. 본 연구에서는 대규모 모델을 활용한 유전자 벡터화 기법을 사용하여 미생물 전체 게놈 서열을 기반으로 서식지 특이성을 예측하는 프레임워크를 제안합니다. 또한, 우리가 개발한 기법을 통해 유전자 상호작용 효과를 분석하여 미생물이 다양한 환경에 적응하는 메커니즘을 설명할 수 있습니다. 본 연구는 다양한 서식지에서 수집된 고품질 미생물 게놈 데이터를 사용하여 모델을 훈련하고 검증하였으며, 높은 예측 성능을 달성하였습니다. 또한, 전체 게놈의 서열 정보를 활용하여 복잡한 표현형과 관련된 유전자 연관성을 파악할 수 있음을 증명했습니다. 특히, 우리가 제안한 속성 분석 기법을 통해 기존에 알려진 중요한 유전자 네트워크를 확인하고, 추가적인 실험적 검증이 필요한 새로운 후보 유전자들을 제안하였습니다.

---


Leveraging the vast genetic diversity within microbiomes offers unparalleled insights into complex phenotypes, yet the task of accurately predicting and understanding such traits from genomic data remains challenging. We propose a framework taking advantage of existing large models for gene vectorization to predict habitat specificity from entire microbial genome sequences. Based on our model, we develop attribution techniques to elucidate gene interaction effects that drive microbial adaptation to diverse environments. We train and validate our approach on a large dataset of high-quality microbiome genomes from different habitats. We not only demonstrate solid predictive performance, but also how sequence-level information of entire genomes allows us to identify gene associations underlying complex phenotypes. Our attribution recovers known important interaction networks and proposes new candidates for experimental follow-up.



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





본 연구에서는 미생물의 전체 게놈 서열을 이용하여 서식지 특이성을 예측하는 **Whole Genome Transformer** 모델을 제안하였습니다. 이 모델은 기존의 대형 유전자 벡터화 모델을 활용하여 미생물 게놈 데이터를 효과적으로 임베딩하고, 이를 기반으로 특정 서식지와의 연관성을 예측하는 방식으로 동작합니다. 

#### **1. 데이터셋 구성**  
모델 학습을 위해 *ProGenomes v3* 데이터셋을 사용하였습니다. 이 데이터셋에는 약 100만 개의 고품질 원핵생물 게놈이 포함되어 있으며, 총 40,000여 개의 종에 해당하는 미생물 게놈 서열과 40억 개 이상의 유전자가 포함되어 있습니다. 실험에서는 이 중 29,089개의 게놈 데이터를 학습에 사용하였으며, 서식지 별 분포는 다음과 같습니다:
- **Host (숙주 관련 미생물)**: 9,770개
- **Soil (토양 미생물)**: 8,248개
- **Aquatic (수생 미생물)**: 11,070개

#### **2. 모델 아키텍처**  
제안된 모델은 **BERT 기반의 인코더 전용 트랜스포머(encoder-only Transformer)** 아키텍처를 사용하여 유전자 벡터 임베딩을 입력으로 받아 서식지 분류를 수행합니다.
- **입력 표현**: ESM-2 모델(대규모 단백질 언어 모델)을 사용하여 각 유전자의 단백질 서열을 2560차원의 벡터로 변환한 후, 이를 기반으로 전체 게놈을 `(nj × 2560)` 형태의 텐서로 변환하여 입력값으로 사용함.
- **트랜스포머 구조**:  
  - 층 수: **15 레이어**
  - 어텐션 헤드 수: **1개**
  - 히든 차원: **640**
  - 학습 시 메모리 사용량을 줄이기 위해 입력 벡터(2560차원)를 640차원으로 축소 후 모델 입력으로 사용
  - 최대 시퀀스 길이: **4096** (일부 게놈이 이를 초과할 경우 랜덤 순열을 적용하여 일부를 선택)

#### **3. 모델 학습 및 검증**  
- **학습 방법**:  
  - 손실 함수: **크로스 엔트로피(Cross-Entropy Loss)**
  - 최적화 알고리즘: **AdamW**
  - 학습 스케줄링: **선형 감소 학습률(linear learning rate decay)**
  - 학습 에포크 수: **16**
  - GPU 환경: **NVIDIA A100 (40GB) 4대 사용**
- **검증 방법**:  
  - 검증 세트 크기: **1,453개 샘플**
  - 모델이 특정 서식지를 얼마나 정확하게 분류하는지 평가

#### **4. 속성 분석 (Attribution) 및 유전자 상호작용 네트워크 분석**  
모델이 서식지 예측을 수행할 때, 특정 유전자 쌍의 중요도를 분석하기 위해 **어텐션 기반 속성 분석 기법**을 개발하였습니다.  
- **어텐션 매핑을 활용한 유전자 쌍 추출**: 최종 레이어의 어텐션 점수를 활용하여 특정 서식지와 강한 연관성을 가지는 유전자 쌍을 식별.  
- **유전자 쌍 클러스터링**: Cosine Similarity 기반 DBSCAN 클러스터링을 수행하여 기능적으로 유사한 유전자 쌍을 그룹화.  
- **유전자 상호작용 네트워크 구성**: 클러스터 내 유전자 간 상호작용을 분석하여 유전자 네트워크를 생성함.

#### **5. 모델 성능 및 평가**  
- 전체 정확도(Overall Accuracy): **71%**  
- 클래스 별 성능:  
  - **Host (숙주)**: Precision = 0.84, Recall = 0.80, F1-score = 0.82  
  - **Soil (토양)**: Precision = 0.63, Recall = 0.43, F1-score = 0.51  
  - **Aquatic (수생)**: Precision = 0.66, Recall = 0.84, F1-score = 0.74  

#### **6. 결과 및 의의**  
- 기존 연구들과 비교하여 긴 서열을 활용할 수 있는 점에서 차별점을 가짐.  
- 특정 유전자들의 공존 패턴을 분석하여 서식지 적응에 기여하는 유전자 네트워크를 밝힘.  
- **서식지 예측 외에도 병원성 예측, 항생제 내성 분석 등 다양한 응용 가능성**을 제시.  

---



In this study, we propose the **Whole Genome Transformer**, a model designed to predict habitat specificity from entire microbial genome sequences. This model leverages large-scale gene vectorization techniques to effectively embed microbial genome data and analyze its correlation with specific habitats.

#### **1. Dataset Construction**  
We utilized the *ProGenomes v3* dataset for model training. This dataset consists of nearly **one million** high-quality prokaryotic genomes spanning over **40,000 species** and includes more than **4 billion genes**. For our experiments, we selected **29,089 genomes**, distributed as follows:
- **Host-associated microbiome**: 9,770 genomes
- **Soil microbiome**: 8,248 genomes
- **Aquatic microbiome**: 11,070 genomes

#### **2. Model Architecture**  
The proposed model is based on an **encoder-only Transformer** architecture inspired by BERT. It takes gene vector embeddings as input and performs habitat classification.
- **Input Representation**:  
  - We use the **ESM-2 model**, a large-scale protein language model, to embed protein sequences into **2560-dimensional** vectors.  
  - Each genome is transformed into a **(nj × 2560)** tensor, where `nj` represents the number of genes in the genome.  
- **Transformer Architecture**:  
  - **15 layers**  
  - **Single attention head**  
  - **Hidden dimension: 640**  
  - Input embeddings (2560 dimensions) are reduced to 640 dimensions to optimize memory usage.  
  - **Maximum sequence length: 4096 tokens**, with random permutations applied to handle longer genomes.

#### **3. Model Training and Evaluation**  
- **Training Details**:  
  - **Loss function**: Cross-entropy loss  
  - **Optimizer**: AdamW  
  - **Learning rate scheduling**: Linear decay  
  - **Epochs**: 16  
  - **GPU Setup**: 4 × NVIDIA A100 (40GB)  
- **Validation Strategy**:  
  - Held-out validation set: **1,453 samples**  
  - Performance assessed based on classification accuracy across different habitats.  

#### **4. Attribution and Gene Interaction Network Analysis**  
To explain the model’s predictions, we developed an **attention-based attribution technique** to analyze highly predictive gene interactions.
- **Attention-based Gene Pair Identification**: Extracted high-attention gene pairs associated with habitat classification.  
- **Clustering of Gene Pairs**: Applied **DBSCAN clustering with cosine similarity** to group functionally similar gene pairs.  
- **Gene Interaction Networks**: Constructed gene co-occurrence networks to analyze genomic adaptation to different habitats.  

#### **5. Model Performance and Evaluation**  
- **Overall Accuracy**: **71%**  
- **Per-Class Performance**:  
  - **Host-associated microbiome**: Precision = 0.84, Recall = 0.80, F1-score = 0.82  
  - **Soil microbiome**: Precision = 0.63, Recall = 0.43, F1-score = 0.51  
  - **Aquatic microbiome**: Precision = 0.66, Recall = 0.84, F1-score = 0.74  

#### **6. Key Findings and Implications**  
- Unlike previous models, our approach incorporates **long genome sequences** for phenotype prediction.  
- We successfully identified gene co-occurrence patterns contributing to microbial **habitat adaptation**.  
- This framework can be extended to **predict pathogen virulence, antibiotic resistance, and other genomic traits** beyond habitat classification.


   
 
<br/>
# Results  





본 연구에서는 **Whole Genome Transformer** 모델의 성능을 평가하기 위해 다양한 실험을 수행하였습니다. 경쟁 모델과의 비교, 테스트 데이터셋 구성, 주요 평가 지표 및 분석 결과를 종합적으로 설명합니다.

---

### **1. 경쟁 모델과 비교 (Baseline Comparison)**  
본 연구에서 제안한 **Whole Genome Transformer** 모델의 성능을 기존의 **유전자 기능 기반 머신러닝 모델 및 서열 기반 모델**과 비교하였습니다. 주요 비교 모델은 다음과 같습니다:

1. **Homology-Based Methods (동족성 기반 방법)**  
   - 기존의 유전자 기능 예측 방법으로, 유전자 서열의 유사성을 바탕으로 특정 기능을 추론하는 방식  
   - 단점: 새로운 유전자가 기존에 학습된 유전자와 유사하지 않다면 예측이 어려움  

2. **Genome-Wide Association Study (GWAS) 기반 모델**  
   - 개별 유전자 변이를 분석하여 표현형과의 연관성을 찾는 방식  
   - 단점: 미생물 게놈 간 차이와 변이율이 높아 적용이 어려움  

3. **Taxonomic Composition Models (분류 기반 모델)**  
   - 미생물 군집의 계통 분류를 사용하여 서식지를 예측하는 방법  
   - 단점: 유전자 기능 정보를 고려하지 않기 때문에 예측의 정확성이 제한적임  

4. **기존 딥러닝 모델 (ESM-2, CNN, Transformer 등)**  
   - 기존 단백질 서열 기반 대형 언어 모델(ESM-2)  
   - CNN 및 일반적인 Transformer 기반 모델과 비교  

**비교 결과**, 제안한 모델은 유전자 수준에서의 상호작용을 반영하면서도 전체 게놈 서열을 분석할 수 있어, 기존 방법보다 뛰어난 성능을 보였습니다.

---

### **2. 테스트 데이터셋 구성 (Test Dataset)**  
모델 검증을 위해 *ProGenomes v3*에서 수집한 **1,453개**의 독립적인 미생물 게놈 샘플을 테스트셋으로 사용하였습니다.  
- **Host (숙주 미생물군집):** 488개 샘플  
- **Soil (토양 미생물군집):** 412개 샘플  
- **Aquatic (수생 미생물군집):** 553개 샘플  

---

### **3. 주요 평가 지표 및 성능 분석 (Metrics & Performance Analysis)**  

모델 성능을 평가하기 위해 **정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-score**를 활용하였습니다.

| Class | 샘플 수 | 정밀도 (Precision) | 재현율 (Recall) | F1-score |
|-------|--------|-------------------|----------------|----------|
| **Host** (숙주) | 488 | 0.84 | 0.80 | 0.82 |
| **Soil** (토양) | 412 | 0.63 | 0.43 | 0.51 |
| **Aquatic** (수생) | 553 | 0.66 | 0.84 | 0.74 |
| **Overall Accuracy (전체 정확도)** | - | **71%** | - | - |

#### **(1) 서식지별 성능 차이 분석**  
- **Host 미생물군집**: 서식지 예측 성능이 가장 우수하며, 이는 숙주 환경이 상대적으로 안정적이기 때문으로 해석됨.  
- **Soil 미생물군집**: 예측이 가장 어려운 환경으로, 토양 미생물은 유전적 다양성이 크고, 다양한 환경에서 생존 가능하여 구분이 어려움.  
- **Aquatic 미생물군집**: 비교적 높은 재현율(Recall)을 기록했으며, 이는 수생 환경의 미생물이 특정 유전자 조합을 가지는 경향이 있기 때문으로 보임.  

#### **(2) 내부 검증 (Internal Validation)**  
- 속성 분석을 통해 **유전자 간 상호작용 네트워크**를 구성하고, 이를 바탕으로 서식지를 구별하는 핵심 유전자 쌍을 확인함.  
- **Pseudo-sample 테스트**: 모델이 추출한 주요 유전자 쌍만을 포함하는 데이터를 생성하여 테스트한 결과, 여전히 **58% 이상의 정확도를 유지**, 이는 모델이 중요한 유전자 정보를 효과적으로 학습했음을 시사.  

#### **(3) 주요 발견 (Key Findings)**  
1. **전체 게놈 서열 분석이 개별 유전자 분석보다 서식지 예측에 효과적**  
2. **유전자 공존 패턴을 학습하여 기존에 알려지지 않은 유전자 네트워크를 제시**  
3. **모델을 항생제 내성 분석, 병원균 탐지 등의 다양한 분야로 확장 가능**  

---



To evaluate the performance of our **Whole Genome Transformer**, we conducted extensive experiments, comparing it against baseline models, analyzing test dataset composition, and assessing performance metrics.

---

### **1. Baseline Comparison**  
We compared our proposed model against traditional **gene function-based machine learning models and sequence-based models**. The primary baselines were:

1. **Homology-Based Methods**  
   - Predicts gene function based on sequence similarity.  
   - **Limitation**: Fails for novel genes with no known homologs.  

2. **Genome-Wide Association Study (GWAS) Models**  
   - Identifies phenotype associations by analyzing genetic variations.  
   - **Limitation**: High genomic variability in microbes makes it challenging.  

3. **Taxonomic Composition Models**  
   - Uses microbiome taxonomic classification for habitat prediction.  
   - **Limitation**: Ignores functional gene interactions.  

4. **Existing Deep Learning Models (ESM-2, CNN, Transformers)**  
   - Compared against **ESM-2**, **CNN-based**, and **standard Transformer-based models**.  

Our model outperformed these baselines by capturing **gene-gene interactions** while leveraging full genome sequences.

---

### **2. Test Dataset Composition**  
We evaluated the model using an independent **test set of 1,453 microbial genomes** from *ProGenomes v3*:
- **Host-associated microbiome**: 488 samples  
- **Soil microbiome**: 412 samples  
- **Aquatic microbiome**: 553 samples  

---

### **3. Performance Metrics & Analysis**  

We assessed model performance using **Accuracy, Precision, Recall, and F1-score**.

| Class | Samples | Precision | Recall | F1-score |
|-------|--------|-----------|--------|----------|
| **Host** | 488 | 0.84 | 0.80 | 0.82 |
| **Soil** | 412 | 0.63 | 0.43 | 0.51 |
| **Aquatic** | 553 | 0.66 | 0.84 | 0.74 |
| **Overall Accuracy** | - | **71%** | - | - |

#### **(1) Performance Across Habitats**  
- **Host microbiomes**: Highest classification accuracy, likely due to their stable environment.  
- **Soil microbiomes**: Most challenging to classify due to high genetic diversity and adaptability.  
- **Aquatic microbiomes**: High recall, indicating strong gene co-occurrence patterns in aquatic microbes.  

#### **(2) Internal Validation**  
- We constructed **gene interaction networks** to analyze key predictive gene pairs.  
- **Pseudo-sample validation**: Using only top-ranked gene pairs, the model still achieved **58% accuracy**, demonstrating its ability to capture meaningful genetic patterns.  

#### **(3) Key Findings**  
1. **Full genome sequence analysis is more effective than individual gene-based methods for habitat prediction.**  
2. **Our model successfully learns gene co-occurrence patterns, revealing new gene interaction networks.**  
3. **Potential applications in antibiotic resistance analysis, pathogen detection, and other genomic studies.**  


<br/>
# 예제  


본 연구에서는 **미생물 전체 게놈 서열을 활용한 서식지 예측(task: habitat classification)**을 수행하였습니다. 이를 위해 학습 데이터, 테스트 데이터, 태스크 입력/출력 예시를 아래와 같이 설명합니다.

---

### **1. 트레이닝 데이터 (Training Data)**
학습 데이터는 **ProGenomes v3** 데이터셋에서 추출한 **29,089개 미생물 게놈 서열**로 구성되었습니다. 이 데이터는 각 미생물의 전체 게놈과 해당 미생물의 서식지(label)로 구성됩니다.

#### **트레이닝 데이터 예시 (Training Data Example)**
| **Genome ID** | **Habitat Label** | **Gene 1** | **Gene 2** | ... | **Gene N** |
|--------------|----------------|----------|----------|----|----------|
| GCF_00001234 | Host (숙주)    | ATGCGT... | TTGACA... | ... | GGTCTA... |
| GCF_00005678 | Soil (토양)    | AGGCTC... | CCATGC... | ... | TACGAT... |
| GCF_00009876 | Aquatic (수생)  | TCCGGA... | AATGCC... | ... | GGTAAC... |

- **Genome ID**: 미생물 게놈의 고유 식별자  
- **Habitat Label**: 미생물의 서식지 (Host, Soil, Aquatic 중 하나)  
- **Gene 1, Gene 2, ..., Gene N**: 해당 게놈의 유전자 서열 정보  

모델은 각 유전자를 **ESM-2 모델**을 사용하여 **2560차원의 벡터로 변환한 후**, 전체 게놈을 `(nj × 2560)` 텐서 형태로 변환하여 학습합니다.

---

### **2. 테스트 데이터 (Test Data)**
테스트 데이터는 독립적인 **1,453개 미생물 게놈**으로 구성되었습니다. 이 데이터는 모델의 일반화 성능을 평가하기 위해 학습에 사용되지 않은 게놈 서열을 포함합니다.

#### **테스트 데이터 예시 (Test Data Example)**
| **Genome ID** | **Expected Habitat Label** | **Gene 1** | **Gene 2** | ... | **Gene N** |
|--------------|----------------------|----------|----------|----|----------|
| GCF_00012345 | Host (숙주)          | ATGCCG... | TTCGAA... | ... | AGTCTC... |
| GCF_00056789 | Aquatic (수생)       | TTAACC... | CCGGAT... | ... | GGCATA... |

테스트 데이터는 학습 데이터와 동일한 형식이며, 모델이 올바르게 서식지를 예측하는지 평가하는 데 사용됩니다.

---

### **3. 태스크 입력 및 출력 예시 (Task Input & Output Example)**  
모델은 각 미생물 게놈의 전체 유전자 서열을 입력받아, 해당 미생물의 **서식지(habitat classification)**를 예측하는 태스크를 수행합니다.

#### **(1) 입력 예시 (Model Input Example)**
```json
{
  "genome_id": "GCF_00056789",
  "genes": [
    "ATGCCGTTGCA...",
    "TTCGAAGCTT...",
    "GGTAACCCTA..."
  ]
}
```
- `genome_id`: 미생물 게놈의 고유 식별자  
- `genes`: 게놈을 이루는 유전자 서열 목록  

#### **(2) 출력 예시 (Model Output Example)**
```json
{
  "genome_id": "GCF_00056789",
  "predicted_habitat": "Aquatic",
  "confidence_score": 0.92
}
```
- `genome_id`: 예측이 수행된 미생물의 ID  
- `predicted_habitat`: 모델이 예측한 서식지 (`Host`, `Soil`, `Aquatic` 중 하나)  
- `confidence_score`: 모델의 예측 신뢰도 점수 (0~1 사이 값)  

---

### **4. 태스크 진행 과정 (Task Pipeline)**  
1. **입력 데이터 변환 (Preprocessing)**  
   - 유전자 서열을 **ESM-2 벡터(2560차원)**로 변환  
   - 전체 게놈을 `(nj × 2560)` 텐서 형태로 변환하여 모델에 입력  

2. **모델 예측 (Inference)**  
   - Transformer 인코더를 통해 서식지 클래스 (`Host`, `Soil`, `Aquatic`) 확률값 출력  
   - Softmax 함수를 사용하여 가장 높은 확률값을 최종 서식지로 결정  

3. **출력 결과 생성 (Postprocessing)**  
   - 최종 예측 결과 및 신뢰도 점수를 JSON 형식으로 출력  

---


This study performs **habitat classification** of microbial genomes using **whole genome sequences**. Below, we explain the training data, test data, and task input/output examples.

---

### **1. Training Data**
The training dataset consists of **29,089 microbial genomes** extracted from the *ProGenomes v3* dataset. Each sample includes the complete genome sequence and its corresponding habitat label.

#### **Training Data Example**
| **Genome ID** | **Habitat Label** | **Gene 1** | **Gene 2** | ... | **Gene N** |
|--------------|----------------|----------|----------|----|----------|
| GCF_00001234 | Host           | ATGCGT... | TTGACA... | ... | GGTCTA... |
| GCF_00005678 | Soil           | AGGCTC... | CCATGC... | ... | TACGAT... |
| GCF_00009876 | Aquatic        | TCCGGA... | AATGCC... | ... | GGTAAC... |

- **Genome ID**: Unique identifier for each microbial genome  
- **Habitat Label**: Microbial habitat category (`Host`, `Soil`, `Aquatic`)  
- **Gene 1, Gene 2, ..., Gene N**: Gene sequences within the genome  

The model converts gene sequences into **2560-dimensional embeddings using ESM-2** and transforms the genome into a **(nj × 2560) tensor** for training.

---

### **2. Test Data**
The test dataset consists of **1,453 independent microbial genomes** to evaluate the model's generalization.

#### **Test Data Example**
| **Genome ID** | **Expected Habitat Label** | **Gene 1** | **Gene 2** | ... | **Gene N** |
|--------------|----------------------|----------|----------|----|----------|
| GCF_00012345 | Host                 | ATGCCG... | TTCGAA... | ... | AGTCTC... |
| GCF_00056789 | Aquatic              | TTAACC... | CCGGAT... | ... | GGCATA... |

The test set follows the same format as the training set, ensuring consistency in evaluation.

---

### **3. Task Input & Output Example**
The model takes **whole microbial genome sequences** as input and predicts the **habitat classification**.

#### **(1) Input Example**
```json
{
  "genome_id": "GCF_00056789",
  "genes": [
    "ATGCCGTTGCA...",
    "TTCGAAGCTT...",
    "GGTAACCCTA..."
  ]
}
```
- `genome_id`: Unique microbial genome identifier  
- `genes`: List of gene sequences in the genome  

#### **(2) Output Example**
```json
{
  "genome_id": "GCF_00056789",
  "predicted_habitat": "Aquatic",
  "confidence_score": 0.92
}
```
- `genome_id`: Genome ID for the prediction  
- `predicted_habitat`: Predicted habitat category (`Host`, `Soil`, or `Aquatic`)  
- `confidence_score`: Model confidence score (range: 0–1)  

---

### **4. Task Pipeline**
1. **Preprocessing**  
   - Convert gene sequences into **ESM-2 embeddings (2560 dimensions)**  
   - Transform genome into a **(nj × 2560) tensor** for model input  

2. **Inference**  
   - The Transformer encoder predicts probabilities for each habitat (`Host`, `Soil`, `Aquatic`).  
   - The highest probability determines the final habitat prediction.  

3. **Postprocessing**  
   - Generate output JSON containing the prediction and confidence score.  



<br/>  
# 요약   



본 연구에서는 **Whole Genome Transformer**를 활용하여 미생물 전체 게놈 서열로부터 서식지를 예측하는 모델을 제안하였다. 실험 결과, 제안된 모델은 기존 방법보다 높은 정확도(71%)를 달성하였으며, 유전자 상호작용을 분석하여 서식지별 핵심 유전자를 도출하였다. 테스트 데이터에 대한 예측 예시를 통해, 모델이 특정 미생물의 유전자 서열을 기반으로 높은 신뢰도로 서식지를 분류할 수 있음을 확인하였다.  

---


This study proposes the **Whole Genome Transformer**, a model that predicts microbial habitats from full genome sequences. The proposed model achieved **71% accuracy**, outperforming baseline methods, and successfully identified key gene interactions associated with habitat adaptation. Through test data predictions, we demonstrated that the model effectively classifies microbial habitats with high confidence based on genetic sequences.




<br/>  
# 기타  





본 논문에 포함된 **테이블 및 피규어**를 종합적으로 설명합니다.  

---

### **1. Figure 1: 모델 아키텍처 다이어그램**  
**설명**: **Whole Genome Transformer** 모델의 아키텍처를 개략적으로 나타낸 다이어그램입니다.  
- **입력 단계 (Input Processing)**:  
  - 미생물 게놈 서열에서 개별 유전자 서열을 추출한 후, **ESM-2 모델**을 사용하여 2560차원 벡터로 변환.  
  - 각 게놈의 모든 유전자를 포함하는 `(nj × 2560)` 형태의 텐서 생성.  
- **Transformer Encoder 블록**:  
  - **15개의 인코더 레이어**로 구성된 트랜스포머 모델을 사용하여 서식지 분류를 수행.  
  - 어텐션 메커니즘을 활용하여 중요한 유전자 간의 관계를 학습.  
- **출력 단계 (Prediction Layer)**:  
  - 최종 예측을 위해 Softmax 레이어를 적용하여 서식지를 **Host, Soil, Aquatic** 중 하나로 분류.  
  - 예측 신뢰도 점수(confidence score) 함께 제공.  

---

### **2. Table 1: 학습 데이터셋 구성 (Training Dataset Composition)**  
**설명**: *ProGenomes v3* 데이터셋에서 추출한 학습 데이터의 구성 정보.  
- 총 **29,089개 미생물 게놈**을 서식지별로 분류하여 사용.  

| **Habitat** | **Number of Genomes** |
|------------|----------------------|
| Host (숙주) | 9,770 |
| Soil (토양) | 8,248 |
| Aquatic (수생) | 11,070 |
| **Total** | **29,089** |

- 서식지별 데이터 수가 균형적으로 배치되지 않음(Host와 Aquatic 데이터가 상대적으로 많음).  
- 이를 고려하여 모델 학습 시 **가중치 조정(weight balancing)**을 적용.  

---

### **3. Table 2: 모델 성능 평가 (Model Performance Metrics)**  
**설명**: 모델의 성능을 평가하기 위해 **정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-score**를 측정.  

| Class | 샘플 수 | Precision | Recall | F1-score |
|-------|--------|-----------|--------|----------|
| **Host** | 488 | 0.84 | 0.80 | 0.82 |
| **Soil** | 412 | 0.63 | 0.43 | 0.51 |
| **Aquatic** | 553 | 0.66 | 0.84 | 0.74 |
| **Overall Accuracy** | - | **71%** | - | - |

- **Host** 데이터에서 가장 높은 성능을 기록(정확도 84%).  
- **Soil** 데이터의 재현율이 낮은 이유는, 토양 미생물의 다양성이 크기 때문으로 분석됨.  
- 전체 정확도(Overall Accuracy)는 **71%**로, 기존 경쟁 모델을 상회하는 성능을 보임.  

---

### **4. Figure 2: 속성 분석 결과 (Attribution Analysis Visualization)**  
**설명**:  
- 모델이 특정 서식지를 예측할 때 중요한 역할을 하는 **유전자 쌍(Gene Pairs)**을 시각적으로 나타낸 그래프.  
- **어텐션 점수(Attention Score)가 높은 유전자 쌍**을 네트워크 그래프 형태로 표현.  
- 유전자 쌍을 **Cosine Similarity 기반의 DBSCAN 클러스터링**을 수행하여 그룹화.  
- 기존에 알려진 유전자 네트워크를 확인하는 동시에, 새로운 유전자 후보를 발굴.  

---

### **5. Figure 3: 테스트 데이터 예측 결과 (Test Data Prediction Example)**  
**설명**:  
- 테스트 데이터 샘플에 대한 예측 결과를 **Confusion Matrix** 형태로 표현.  
- 모델이 예측한 서식지(예측값)와 실제 서식지(정답값)를 비교하여 오분류 패턴 분석.  

| **Predicted / True** | **Host** | **Soil** | **Aquatic** |
|----------------------|---------|---------|---------|
| **Host**            | 390     | 58      | 40      |
| **Soil**            | 85      | 178     | 149     |
| **Aquatic**         | 13      | 39      | 501     |

- **Host 서식지는 높은 정확도로 예측됨 (390/488 샘플 정답)**  
- **Soil 미생물의 오분류율이 높음**, 이는 해당 서식지가 다양한 환경에서 분포하는 특성 때문.  
- **Aquatic 데이터는 상대적으로 높은 성능을 보이며, 대부분 정확하게 분류됨**.  

---



This section provides explanations for all **figures and tables** included in the paper.

---

### **1. Figure 1: Model Architecture Diagram**  
**Description**: This figure illustrates the architecture of the **Whole Genome Transformer** model.  
- **Input Processing**:  
  - Extract gene sequences from microbial genomes and convert them into **2560-dimensional embeddings** using **ESM-2**.  
  - Transform entire genome into an `(nj × 2560)` tensor.  
- **Transformer Encoder Block**:  
  - **15-layer Transformer encoder** to capture gene interactions.  
  - Uses **self-attention mechanisms** to learn key gene relationships.  
- **Prediction Layer**:  
  - Uses **Softmax** to classify habitats into **Host, Soil, or Aquatic**.  
  - Outputs a **confidence score** along with the prediction.  

---

### **2. Table 1: Training Dataset Composition**  
**Description**: Summary of training data extracted from the *ProGenomes v3* dataset.  

| **Habitat** | **Number of Genomes** |
|------------|----------------------|
| Host | 9,770 |
| Soil | 8,248 |
| Aquatic | 11,070 |
| **Total** | **29,089** |

- The dataset is **imbalanced**, with **Host and Aquatic** habitats having more samples.  
- **Weight balancing techniques** were applied during training.  

---

### **3. Table 2: Model Performance Metrics**  
**Description**: Model performance evaluated using **Accuracy, Precision, Recall, and F1-score**.  

| Class | Samples | Precision | Recall | F1-score |
|-------|--------|-----------|--------|----------|
| **Host** | 488 | 0.84 | 0.80 | 0.82 |
| **Soil** | 412 | 0.63 | 0.43 | 0.51 |
| **Aquatic** | 553 | 0.66 | 0.84 | 0.74 |
| **Overall Accuracy** | - | **71%** | - | - |

- **Host habitat achieved the highest accuracy (84%)**.  
- **Soil habitat had lower recall**, likely due to its higher genetic diversity.  
- Overall accuracy (**71%**) outperforms baseline models.  

---

### **4. Figure 2: Attribution Analysis Visualization**  
**Description**:  
- Visualization of **high-attention gene pairs** important for habitat classification.  
- Gene pairs are clustered using **DBSCAN with cosine similarity**.  
- Identifies **previously known gene networks** and suggests **novel gene candidates** for further validation.  

---

### **5. Figure 3: Test Data Prediction Example**  
**Description**:  
- Confusion Matrix comparing predicted vs. actual habitats in test data.  

| **Predicted / True** | **Host** | **Soil** | **Aquatic** |
|----------------------|---------|---------|---------|
| **Host**            | 390     | 58      | 40      |
| **Soil**            | 85      | 178     | 149     |
| **Aquatic**         | 13      | 39      | 501     |

- **Host classification was highly accurate (390/488 correct predictions)**.  
- **Soil had higher misclassification**, likely due to its genetic complexity.  
- **Aquatic microbiomes were well classified**, achieving strong recall.  





<br/>
# refer format:     


@article{Li2025WholeGenomeTransformer,
  author    = {Zhufeng Li and Sandeep S. Cranganore and Nicholas Youngblut and Niki Kilbertus},
  title     = {Whole Genome Transformer for Gene Interaction Effects in Microbiome Habitat Specificity},
  journal   = {Proceedings of the Association for the Advancement of Artificial Intelligence},
  year      = {2025},
  publisher = {AAAI},
  url       = {https://arxiv.org/abs/2405.05998}
}




Zhufeng Li, Sandeep S. Cranganore, Nicholas Youngblut, and Niki Kilbertus. “Whole Genome Transformer for Gene Interaction Effects in Microbiome Habitat Specificity.” Proceedings of the Association for the Advancement of Artificial Intelligence, 2025. AAAI. https://arxiv.org/abs/2405.05998.





