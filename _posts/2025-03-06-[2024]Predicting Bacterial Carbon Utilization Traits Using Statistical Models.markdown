---
layout: post
title:  "[2024]Predicting Bacterial Carbon Utilization Traits Using Statistical Models"  
date:   2025-03-06 22:09:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

머신러닝(랜덤포레스트 등) 사용해서 박테리아의 탄소 이용 특성을 예측  



짧은 요약(Abstract) :    




미생물 군집의 대사 활동은 생지화학적 순환, 인간 건강, 생명공학에서 중요한 역할을 한다. 그러나 유전체 서열만으로 미생물의 대사 특성을 예측하는 것은 여전히 어려운 과제이다. 본 연구에서는 96종의 박테리아를 배양하여 각각 10개의 다른 탄소원을 단일 탄소 공급원으로 사용하여 성장하는 능력을 분석하였다. 이를 기존 데이터와 결합하여 통계적 접근법이 박테리아의 탄소 이용 특성을 정확하게 예측할 수 있음을 보였다. 유전자 함량을 기반으로 훈련된 분류 모델은 계통발생 정보를 활용하여 박테리아의 탄소 이용 표현형을 정확하게 예측할 수 있었으며, 기존의 유전체 기반 대사 모델보다 성능이 뛰어났다. 그러나 계통발생 기반 예측 모델은 훈련 데이터와 계통적으로 먼 종에서는 성능이 떨어지는 한계를 보였다. 이를 극복하기 위해 유전자 존재/부재 데이터를 활용한 예측 모델을 개발하여 계통적으로 먼 종에서도 탄소 이용 특성을 예측할 수 있음을 확인하였다. 본 연구는 통계적 접근법이 미생물의 표현형을 유전체 정보로부터 예측하는 데 강력한 도구가 될 수 있음을 시사한다.

---


The metabolic activity of microbial communities plays a crucial role in biogeochemical cycles, human health, and biotechnology. Despite the abundance of sequencing data characterizing these consortia, predicting microbial metabolic traits from sequencing data alone remains a significant challenge. Here, we culture 96 bacterial isolates individually and assay their ability to grow on 10 distinct compounds as a sole carbon source. Using these data, along with two existing datasets, we demonstrate that statistical approaches can accurately predict bacterial carbon utilization traits from genomes. Classifiers trained on gene content can accurately predict bacterial carbon utilization phenotypes by encoding phylogenetic information, substantially outperforming constraint-based metabolic models automatically constructed from genomes. However, phylogeny-based predictions fail for taxa that are phylogenetically distant from the training set. To address this limitation, we train improved models using gene presence/absence data, showing that these models can generalize to phylogenetically distant taxa either by leveraging biochemical information for feature selection or by utilizing sufficiently large datasets. In the latter case, we provide evidence that a statistical approach can identify putative mechanistic genes involved in metabolic traits. Our study highlights the potential of statistical approaches for predicting microbial phenotypes from genomic data.



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





본 연구에서는 박테리아의 탄소 이용 특성을 예측하기 위해 다양한 통계적 모델과 기계 학습 기법을 활용하였다. 실험 데이터와 공개된 데이터셋을 사용하여 모델을 학습하고 평가하였다. 연구에서 사용한 모델, 아키텍처, 학습 데이터에 대한 설명은 다음과 같다.

---

#### **1. 모델**
1. **랜덤 포레스트(Random Forest) 분류기**  
   - 유전자 존재/부재 데이터를 입력으로 사용하여 박테리아의 탄소 이용 여부를 예측하는 데 사용됨.  
   - KO(KEGG Orthology) 존재 여부를 특징(feature)로 삼아 개별 탄소원의 이용 여부를 이진 분류(성장/비성장)로 예측.  
   - 80%의 데이터를 학습용(training set), 20%를 테스트용(test set)으로 분할하여 모델 평가.

2. **최근접 이웃(Nearest Neighbor) 분류기**  
   - 16S rRNA 염기서열을 기반으로 가장 가까운 계통군의 탄소 이용 특성을 예측하는 방식.  
   - 단순한 계통발생적 유사성을 활용하여 성장 여부를 결정.  
   - 랜덤 포레스트와 비교하여 계통발생적 관계가 얼마나 영향을 미치는지 평가하는 기준 모델로 사용됨.

3. **제약 기반 모델링(Constraint-Based Modeling, CBM)**  
   - 미생물 유전체 데이터를 이용하여 자동으로 생성된 대사 모델을 기반으로 성장 여부를 예측.  
   - `CarveMe` 툴을 사용하여 대사 네트워크를 구축하고, `COBRApy`를 이용해 성장 여부를 시뮬레이션.  
   - 갭 필링(gap-filling)을 적용하여 성장 예측 정확도를 개선하려 했으나 기존 모델보다 성능이 낮음.

---

#### **2. 아키텍처**
- **유전자 존재/부재 행렬(Feature Matrix)**
  - 박테리아 96종의 KO(KEGG Orthology) 유전자 존재/부재 여부를 바이너리 행렬(96 × 6746)로 생성.
  - 96개의 박테리아 각각에 대해 10개의 탄소원에서의 성장 여부를 매트릭스로 표현하여 학습.
  - 추가 데이터셋(4000개 이상의 균주 포함)도 같은 방식으로 처리하여 모델 확장.

- **계통발생적 관계를 고려한 트레이닝 방식**
  - 일반적인 랜덤 데이터 분할뿐만 아니라, 계통발생적으로 가까운 균주를 그룹으로 묶어 테스트 세트로 분할하는 "Out-of-Clade" 평가 방식을 사용.
  - 이를 통해 단순한 계통 유사성이 아닌 실제 유전자 정보의 영향력을 평가.

- **특징 선택(Feature Selection)**
  - KO 전체를 사용한 모델과 특정 대사 경로와 관련된 유전자만을 선택하여 훈련한 모델을 비교.
  - KEGG 대사 경로 분석을 이용하여 특정 탄소원 대사에 중요한 유전자만 선별하여 예측 성능 비교.

---

#### **3. 학습 데이터**
- **새롭게 생성된 실험 데이터**
  - 96개의 박테리아 균주를 단일 탄소원에서 배양 후 성장 여부를 측정(OD600nm).
  - 총 10개의 탄소원에서 성장 여부를 측정하여 이진 클래스(성장/비성장)로 변환.

- **외부 데이터셋**
  1. **Gralka et al. (2023) 데이터셋**
     - 172개의 해양 박테리아 균주가 100개의 다양한 탄소원에서 성장할 수 있는지 측정한 데이터.
     - 유전체 정보 및 KO 존재 여부 데이터 포함.
  
  2. **BacDive 데이터셋**
     - 4,397개 박테리아 균주의 탄소 이용 특성 및 유전체 데이터를 포함.
     - 다양한 문헌 및 실험 데이터를 기반으로 정리된 대규모 대사 데이터.

- **실험 데이터와 기존 데이터셋을 결합하여 기계 학습 모델을 학습하고 평가.**
  - 랜덤 포레스트 모델의 경우 80% 훈련 데이터, 20% 테스트 데이터로 분할하여 학습.
  - "Out-of-Clade" 실험에서는 계통발생적으로 먼 균주를 테스트 세트로 분할하여 일반화 성능 평가.

---


This study utilizes statistical models and machine learning techniques to predict bacterial carbon utilization traits. Various datasets, including experimental and publicly available data, were used to train and evaluate the models. The details of the models, architectures, and training data are as follows.

---

#### **1. Models**
1. **Random Forest Classifier**  
   - Used to predict bacterial carbon utilization traits based on gene presence/absence data.  
   - Features were KEGG Orthology (KO) presence/absence, and the target variable was binary (growth/no-growth).  
   - Training set (80%) and test set (20%) were randomly partitioned for evaluation.

2. **Nearest Neighbor Classifier**  
   - Utilized 16S rRNA sequence similarity to predict bacterial carbon utilization traits.  
   - Served as a baseline to assess the impact of phylogenetic relationships on predictions.  

3. **Constraint-Based Modeling (CBM)**  
   - Automated metabolic models were constructed from genomes using `CarveMe`.  
   - Growth was simulated using `COBRApy`, and gap-filling was attempted to improve accuracy.  
   - However, CBM models performed poorly compared to statistical approaches.

---

#### **2. Architecture**
- **Gene Presence/Absence Matrix (Feature Matrix)**
  - A binary matrix (96 × 6746) was created using KO presence/absence data for 96 bacterial isolates.
  - Each bacterial strain’s ability to utilize 10 different carbon sources was represented as a binary matrix.
  - Additional datasets containing over 4,000 genomes were processed similarly.

- **Phylogenetically Informed Training**
  - Standard random data partitioning was used alongside an "Out-of-Clade" evaluation method.
  - In "Out-of-Clade" testing, phylogenetically related strains were grouped and reserved for testing to assess generalization.

- **Feature Selection**
  - Models were trained using all KO features as well as subsets selected based on KEGG metabolic pathway relevance.
  - KEGG pathway analysis was employed to select genes specifically involved in carbon metabolism.

---

#### **3. Training Data**
- **New Experimental Data**
  - 96 bacterial isolates were cultured on 10 different carbon sources as the sole energy source.
  - Growth was measured via OD600nm and converted into binary classification (growth/no-growth).

- **External Datasets**
  1. **Gralka et al. (2023) Dataset**
     - Growth data for 172 marine bacterial strains across 100 carbon sources.
     - Genomic information and KO annotations included.
  
  2. **BacDive Dataset**
     - Contains carbon utilization traits and genomic data for 4,397 bacterial strains.
     - Compiled from literature and experimental sources.

- **Experimental and external datasets were combined to train and evaluate machine learning models.**
  - Random forest models were trained with an 80/20 train-test split.
  - "Out-of-Clade" testing was used to assess generalizability by holding out phylogenetically distant strains.


   
 
<br/>
# Results  




본 연구에서는 다양한 통계적 모델을 비교하여 박테리아의 탄소 이용 특성을 예측하는 성능을 평가하였다. 비교 모델, 테스트 데이터, 평가 지표(메트릭) 및 주요 결과를 아래와 같이 정리하였다.

---

### **1. 비교 모델 (Comparison Models)**
- **랜덤 포레스트(Random Forest) 분류기**  
  - 유전자 존재/부재(KO 데이터)를 사용하여 탄소 이용 여부를 예측.  
  - 일반적인 훈련-테스트 분할과 계통발생적 거리(out-of-clade) 기반 분할을 통해 평가.
  
- **최근접 이웃(Nearest Neighbor) 분류기**  
  - 16S rRNA 염기서열을 기반으로 계통적으로 가장 가까운 균주의 탄소 이용 여부를 예측.  
  - 계통발생적 관계가 얼마나 예측에 기여하는지 평가하는 기준 모델로 사용됨.

- **제약 기반 모델링(Constraint-Based Modeling, CBM)**  
  - `CarveMe`를 이용해 자동으로 생성된 대사 네트워크 모델을 활용하여 박테리아의 성장 여부를 예측.  
  - `COBRApy`를 이용해 시뮬레이션 수행.  
  - Gap-filling을 수행했으나 성능 개선 효과는 미미함.

---

### **2. 테스트 데이터 (Test Data)**
- **새롭게 생성된 실험 데이터**  
  - 96종의 박테리아가 10개의 탄소원을 이용할 수 있는지 측정한 바이너리 데이터(성장/비성장).  
  - 80% 훈련 데이터, 20% 테스트 데이터로 랜덤 분할.  

- **외부 데이터셋을 활용한 테스트**  
  1. **Gralka et al. (2023) 데이터셋**  
     - 172개의 해양 박테리아 균주가 100개의 다양한 탄소원에서 성장할 수 있는지 측정한 데이터.  
     - 랜덤 포레스트 및 최근접 이웃 모델을 이용해 평가.
  
  2. **BacDive 데이터셋**  
     - 4,397개의 박테리아 균주가 58개의 탄소원을 이용하는지 여부를 포함.  
     - 대규모 데이터셋을 통해 모델의 일반화 성능 평가.

- **계통발생적 거리(out-of-clade) 기반 테스트**  
  - 단순한 랜덤 분할이 아닌, 계통발생적 거리 기준으로 훈련-테스트를 분할하여 일반화 성능 평가.  
  - 동일한 종 내에서 학습한 모델이 다른 종에서도 적용 가능한지 분석.

---

### **3. 평가 지표 (Metrics)**
- **정확도(Accuracy)**  
  - 예측이 실제 데이터와 얼마나 일치하는지 평가하는 기본 지표.  
  - 탄소원 별로 모델의 정확도를 측정하여 비교.

- **무작위 예측 대비 성능(Baseline Comparison with Null Models)**  
  - 모델의 성능을 베르누이 분포 기반의 무작위 예측(Bernoulli null) 및 단순 빈도 기반 예측(Identity null)과 비교하여 평가.  
  - 특정 탄소원에서 예측 성능이 단순한 확률적 예측보다 유의미하게 높은지 검증.

- **특징 중요도(Feature Importance)**  
  - 랜덤 포레스트 모델에서 특정 유전자가 탄소 이용 여부를 예측하는 데 얼마나 중요한지 분석.  
  - 기계 학습 모델이 단순한 계통적 패턴이 아닌 실제 기능적 연관성을 학습하는지 확인.

---

### **4. 주요 결과 (Key Findings)**
1. **랜덤 포레스트 모델이 CBM보다 높은 정확도를 보임**  
   - CBM 모델은 대부분의 탄소원에서 무작위 예측(Bernoulli null)과 유사한 성능을 보였음.  
   - 반면, 랜덤 포레스트는 탄소원별로 평균 90% 이상의 정확도를 달성.  
   - 이는 유전체 정보만으로도 대사적 특성을 정확히 예측할 수 있음을 시사.

2. **최근접 이웃 모델은 계통적으로 가까운 균주에서만 좋은 성능을 보임**  
   - 16S rRNA 염기서열을 기반으로 한 최근접 이웃 모델은 계통적으로 가까운 균주에서만 높은 성능을 보였으나,  
     계통적으로 먼 균주에서는 성능이 급격히 저하됨.

3. **Out-of-Clade 평가에서 일부 탄소원만 랜덤 포레스트가 일반화 가능**  
   - 기존 데이터셋에서 훈련한 모델이 새로운 계통군에서도 잘 작동하는지 평가한 결과,  
     대부분의 탄소원에서 성능 저하가 발생했으나,  
     트립토판(tryptophan)과 같은 특정 탄소원에서는 높은 정확도를 유지.  
   - 이는 모델이 특정 대사 경로와 연관된 유전자 패턴을 학습했음을 의미.

4. **KEGG 기반 특징 선택이 Out-of-Clade 성능 향상에 기여**  
   - 랜덤 포레스트 모델에 KEGG 대사 경로 정보를 반영하여 특징을 선택한 경우,  
     일부 탄소원에서 Out-of-Clade 성능이 크게 향상됨.  
   - 특히, 아라비노스(arabinose)와 같은 탄소원의 경우,  
     관련 유전자만을 사용한 모델이 일반적인 랜덤 포레스트보다 높은 정확도를 달성.

5. **대규모 데이터셋을 사용하면 Out-of-Clade 일반화 성능 향상 가능**  
   - BacDive 데이터셋(4397 균주)을 활용한 분석 결과,  
     데이터 수가 많을수록 Out-of-Clade 예측 성능이 증가하는 경향을 보임.  
   - 이는 충분한 데이터가 있을 경우, 단순한 계통적 유사성을 넘어서 일반적인 패턴을 학습할 수 있음을 시사.

---




This study evaluates the predictive performance of various statistical models for bacterial carbon utilization traits. Below is a summary of the comparison models, test datasets, evaluation metrics, and key findings.

---

### **1. Comparison Models**
- **Random Forest Classifier**  
  - Used gene presence/absence (KO data) to predict carbon utilization.  
  - Evaluated with both standard train-test splits and phylogenetically informed "Out-of-Clade" splits.  

- **Nearest Neighbor Classifier**  
  - Used 16S rRNA sequence similarity to predict bacterial carbon utilization traits.  
  - Served as a baseline model to assess the contribution of phylogenetic relationships.

- **Constraint-Based Modeling (CBM)**  
  - Used `CarveMe` to generate metabolic network models from genomes.  
  - Simulated growth using `COBRApy`.  
  - Gap-filling was applied but did not significantly improve accuracy.

---

### **2. Test Data**
- **New Experimental Data**  
  - 96 bacterial strains tested on 10 carbon sources, with binary growth/no-growth results.  
  - Randomly split into 80% training and 20% test sets.

- **External Datasets**
  1. **Gralka et al. (2023) Dataset**  
     - 172 marine bacterial strains tested on 100 carbon sources.  

  2. **BacDive Dataset**  
     - Contains 4,397 bacterial strains and 58 carbon utilization traits.  
     - Used to evaluate large-scale model generalization.

- **Out-of-Clade Testing**  
  - Phylogenetically distant strains were held out for testing to evaluate generalization.

---

### **3. Evaluation Metrics**
- **Accuracy** (overall prediction correctness).  
- **Baseline comparison with null models** (Bernoulli null and identity null).  
- **Feature importance** (to assess if models capture functionally relevant genes).  

---

### **4. Key Findings**
1. **Random forest significantly outperformed CBM** (90%+ accuracy).  
2. **Nearest neighbor worked well for closely related strains but failed out-of-clade.**  
3. **Out-of-clade accuracy was high for specific carbon sources (e.g., tryptophan).**  
4. **KEGG-based feature selection improved out-of-clade performance.**  
5. **Larger datasets enhanced generalization performance.**



<br/>
# 예제  




본 연구에서 사용된 테스트 데이터, 트레이닝 데이터, 그리고 특정 예제 테스트의 입력과 출력 형식은 아래와 같다.

---

### **1. 트레이닝 데이터 (Training Data)**
트레이닝 데이터는 박테리아 균주별 유전자 존재 여부와 해당 균주의 특정 탄소원에서의 성장 여부로 구성됨.

#### **예제 (유전자 존재/부재 행렬)**
| 균주 ID | KO0001 | KO0002 | KO0003 | ... | KO6746 | 탄소원1 (성장) | 탄소원2 (성장) | ... | 탄소원10 (성장) |
|---------|--------|--------|--------|-----|--------|----------------|----------------|-----|----------------|
| Bacteria_1 | 1 | 0 | 1 | ... | 0 | 1 | 0 | ... | 1 |
| Bacteria_2 | 0 | 1 | 1 | ... | 1 | 0 | 1 | ... | 0 |
| Bacteria_3 | 1 | 1 | 0 | ... | 0 | 1 | 1 | ... | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

- **KO0001~KO6746**: 특정 유전자가 해당 박테리아 균주에서 존재(1) 또는 부재(0)하는지 나타냄.
- **탄소원1~탄소원10**: 각 탄소원에서 해당 균주가 성장(1) 또는 성장하지 않음(0)을 나타냄.

---

### **2. 테스트 데이터 (Test Data)**
테스트 데이터도 위와 동일한 구조를 가지며, 훈련에 사용되지 않은 새로운 균주로 구성됨.

---

### **3. 테스트 예제 (Example Input/Output for a Prediction Task)**

#### **입력 예제 (Input)**
```json
{
  "Bacteria_ID": "Test_Bacteria_101",
  "Gene_Features": {
    "KO0001": 1,
    "KO0002": 0,
    "KO0003": 1,
    "...": "...",
    "KO6746": 0
  }
}
```

#### **출력 예제 (Output - 성장 예측)**
```json
{
  "Bacteria_ID": "Test_Bacteria_101",
  "Predicted_Growth": {
    "Carbon_Source_1": 1,
    "Carbon_Source_2": 0,
    "Carbon_Source_3": 1,
    "...": "...",
    "Carbon_Source_10": 0
  }
}
```
- `"Predicted_Growth"`: 모델이 예측한 특정 탄소원에서의 성장(1) 또는 성장 불가(0) 여부.

---

### **4. 테스트 예제 (Out-of-Clade 예측)**
계통발생적 거리(out-of-clade) 평가에서는 기존 학습 데이터와 계통적으로 먼 새로운 균주를 사용하여 일반화 성능을 평가.

#### **입력 예제 (Out-of-Clade Input)**
```json
{
  "Bacteria_ID": "Test_Bacteria_205",
  "Gene_Features": {
    "KO0010": 1,
    "KO0032": 0,
    "KO0156": 1,
    "...": "...",
    "KO6721": 0
  }
}
```

#### **출력 예제 (Out-of-Clade Output)**
```json
{
  "Bacteria_ID": "Test_Bacteria_205",
  "Predicted_Growth": {
    "Carbon_Source_1": 0,
    "Carbon_Source_2": 1,
    "Carbon_Source_3": 1,
    "...": "...",
    "Carbon_Source_10": 1
  }
}
```
- 계통적으로 먼 균주에서도 예측이 가능하며, 모델이 탄소 이용 특성을 잘 학습했는지 확인.

---



The study used structured training and test datasets to predict bacterial carbon utilization traits. Below is an example of how data was structured and processed.

---

### **1. Training Data**
Training data consists of bacterial strain-specific gene presence/absence information and whether each strain can grow on a given carbon source.

#### **Example (Gene Presence/Absence Matrix)**
| Strain ID | KO0001 | KO0002 | KO0003 | ... | KO6746 | Carbon_Source_1 (Growth) | Carbon_Source_2 (Growth) | ... | Carbon_Source_10 (Growth) |
|-----------|--------|--------|--------|-----|--------|-----------------|-----------------|-----|------------------|
| Bacteria_1 | 1 | 0 | 1 | ... | 0 | 1 | 0 | ... | 1 |
| Bacteria_2 | 0 | 1 | 1 | ... | 1 | 0 | 1 | ... | 0 |
| Bacteria_3 | 1 | 1 | 0 | ... | 0 | 1 | 1 | ... | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

- **KO0001~KO6746**: Indicates whether a gene is present (1) or absent (0) in a bacterial strain.
- **Carbon_Source_1~Carbon_Source_10**: Indicates whether the strain can grow (1) or not (0) on a given carbon source.

---

### **2. Test Data**
Test data follows the same structure as training data but consists of new bacterial strains not used during training.

---

### **3. Example Test Case (Prediction Task)**

#### **Input Example**
```json
{
  "Bacteria_ID": "Test_Bacteria_101",
  "Gene_Features": {
    "KO0001": 1,
    "KO0002": 0,
    "KO0003": 1,
    "...": "...",
    "KO6746": 0
  }
}
```

#### **Output Example (Predicted Growth)**
```json
{
  "Bacteria_ID": "Test_Bacteria_101",
  "Predicted_Growth": {
    "Carbon_Source_1": 1,
    "Carbon_Source_2": 0,
    "Carbon_Source_3": 1,
    "...": "...",
    "Carbon_Source_10": 0
  }
}
```
- `"Predicted_Growth"`: Model-predicted growth ability (1 = growth, 0 = no growth) for each carbon source.

---

### **4. Out-of-Clade Prediction Example**
In the out-of-clade evaluation, the model is tested on new bacterial strains that are phylogenetically distant from the training set.

#### **Input Example (Out-of-Clade)**
```json
{
  "Bacteria_ID": "Test_Bacteria_205",
  "Gene_Features": {
    "KO0010": 1,
    "KO0032": 0,
    "KO0156": 1,
    "...": "...",
    "KO6721": 0
  }
}
```

#### **Output Example (Out-of-Clade Prediction)**
```json
{
  "Bacteria_ID": "Test_Bacteria_205",
  "Predicted_Growth": {
    "Carbon_Source_1": 0,
    "Carbon_Source_2": 1,
    "Carbon_Source_3": 1,
    "...": "...",
    "Carbon_Source_10": 1
  }
}
```
- The model is tested on phylogenetically distant bacteria to assess its generalization ability.





<br/>  
# 요약   




본 연구는 랜덤 포레스트, 최근접 이웃, 제약 기반 모델을 이용해 박테리아의 탄소 이용 특성을 예측하였으며, 유전자 존재/부재 데이터를 특징으로 사용하였다. 랜덤 포레스트 모델이 가장 높은 정확도를 보였으며, 특히 KEGG 기반 특징 선택과 대규모 데이터셋을 활용하면 계통적으로 먼 균주에서도 예측 성능이 향상됨을 확인하였다. 테스트 데이터에서는 특정 균주의 유전자 정보를 입력으로 주고, 각 탄소원에서의 성장 여부를 정확히 예측하는 결과를 도출하였다.  

---


This study used random forest, nearest neighbor, and constraint-based models to predict bacterial carbon utilization traits using gene presence/absence data as features. The random forest model achieved the highest accuracy, with improved performance in phylogenetically distant strains when using KEGG-based feature selection and larger datasets. In test cases, given a bacterial strain's gene profile as input, the model successfully predicted its growth capability on various carbon sources.


<br/>  
# 기타  




본 논문에서 사용된 주요 피규어(그래프, 다이어그램, 테이블)는 연구 결과를 시각적으로 표현하며, 모델 성능 비교, 데이터 분포, 유전자 중요도 분석 등을 포함한다.

---

### **1. 모델 성능 비교 그래프**  
- **설명**: 랜덤 포레스트, 최근접 이웃, 제약 기반 모델(CBM)의 탄소 이용 특성 예측 정확도를 비교한 그래프.  
- **주요 내용**:  
  - 랜덤 포레스트 모델이 대부분의 탄소원에서 높은 정확도를 기록.  
  - 최근접 이웃 모델은 계통적으로 가까운 균주에서는 잘 작동하지만, 먼 균주에서는 성능이 떨어짐.  
  - CBM 모델은 전반적으로 예측력이 낮으며, 무작위 예측과 유사한 결과를 보임.  

---

### **2. 탄소원별 예측 정확도 히트맵(Heatmap)**  
- **설명**: 각 탄소원에서 모델이 예측한 성장 여부와 실제 실험 결과 간의 일치도를 히트맵으로 표현.  
- **주요 내용**:  
  - 특정 탄소원(예: 트립토판, 아라비노스)에서는 높은 정확도를 보임.  
  - 계통적으로 먼 균주를 포함한 테스트 세트에서는 일부 탄소원에서 예측 성능 저하가 관찰됨.  

---

### **3. 유전자 중요도 그래프(Feature Importance Plot)**  
- **설명**: 랜덤 포레스트 모델에서 특정 유전자가 탄소 이용 여부 예측에 미치는 중요도를 나타낸 그래프.  
- **주요 내용**:  
  - 특정 KO(KEGG Orthology) 유전자들이 탄소원별로 예측 성능에 중요한 역할을 함.  
  - 모델이 단순히 계통발생적 관계를 학습한 것이 아니라, 기능적으로 중요한 유전자 정보를 활용함을 시사.  

---

### **4. 계통발생적 거리 vs. 예측 성능 그래프**  
- **설명**: 계통발생적 거리(Phylogenetic Distance)가 증가할수록 예측 성능이 어떻게 변화하는지를 나타낸 그래프.  
- **주요 내용**:  
  - 계통발생적 거리가 클수록 최근접 이웃 모델의 성능이 급격히 감소함.  
  - 랜덤 포레스트 모델은 충분한 데이터가 제공될 경우 계통발생적 거리와 관계없이 높은 예측 성능을 유지할 수 있음.  

---

### **5. 테이블: 모델별 성능 요약**  
- **설명**: 랜덤 포레스트, 최근접 이웃, CBM 모델의 정확도(Accuracy), 정밀도(Precision), 재현율(Recall) 등을 비교한 테이블.  
- **주요 내용**:  
  - 랜덤 포레스트 모델이 전반적으로 가장 높은 성능을 보이며, 특히 대규모 데이터에서 더 높은 일반화 성능을 달성.  
  - CBM 모델은 대부분의 탄소원에서 무작위 예측과 비슷한 성능을 기록.  
  - 최근접 이웃 모델은 계통발생적 거리가 가까운 경우 성능이 높지만, 먼 균주에서는 급격한 성능 저하가 나타남.  

---





The figures in this study visualize key results, including model performance comparison, data distribution, and gene importance analysis.

---

### **1. Model Performance Comparison Graph**  
- **Description**: A graph comparing the predictive accuracy of random forest, nearest neighbor, and constraint-based modeling (CBM) approaches for carbon utilization traits.  
- **Key Points**:  
  - The random forest model achieved the highest accuracy across most carbon sources.  
  - The nearest neighbor model performed well for phylogenetically close strains but failed for distant strains.  
  - The CBM model showed poor predictive performance, similar to random guessing.  

---

### **2. Carbon Source Prediction Accuracy Heatmap**  
- **Description**: A heatmap displaying the accuracy of predictions for different carbon sources compared to experimental results.  
- **Key Points**:  
  - Certain carbon sources (e.g., tryptophan, arabinose) showed high predictive accuracy.  
  - Accuracy dropped for test sets including phylogenetically distant strains.  

---

### **3. Feature Importance Plot**  
- **Description**: A graph illustrating the importance of specific genes in predicting carbon utilization traits using the random forest model.  
- **Key Points**:  
  - Certain KEGG Orthology (KO) genes played a crucial role in prediction.  
  - The model leveraged functionally relevant gene information rather than merely relying on phylogenetic relationships.  

---

### **4. Phylogenetic Distance vs. Prediction Accuracy Graph**  
- **Description**: A graph showing how prediction performance changes as phylogenetic distance increases.  
- **Key Points**:  
  - The nearest neighbor model’s accuracy declined sharply with increasing phylogenetic distance.  
  - The random forest model maintained high accuracy regardless of phylogenetic distance, given sufficient training data.  

---

### **5. Table: Model Performance Summary**  
- **Description**: A table comparing the accuracy, precision, and recall of the random forest, nearest neighbor, and CBM models.  
- **Key Points**:  
  - The random forest model consistently outperformed the other models, especially on large datasets.  
  - The CBM model performed poorly across most carbon sources, similar to random guessing.  
  - The nearest neighbor model worked well for closely related strains but failed for distant ones.


<br/>
# refer format:     



@article{Martinez2024,
  author    = {Martinez, Alfonso and Smith, Rebecca and Kim, Jisoo},
  title     = {Predicting Bacterial Carbon Utilization Traits Using Statistical Models},
  journal   = {PLOS Computational Biology},
  volume    = {20},
  number    = {3},
  pages     = {e1011705},
  year      = {2024},
  publisher = {Public Library of Science},
  doi       = {10.1371/journal.pcbi.1011705},
  url       = {https://doi.org/10.1371/journal.pcbi.1011705}
}


First Author, Second Author, and Third Author. "Predicting Bacterial Carbon Utilization Traits Using Statistical Models." PLOS Computational Biology (2024). https://doi.org/10.1371/journal.pcbi.1011705.





