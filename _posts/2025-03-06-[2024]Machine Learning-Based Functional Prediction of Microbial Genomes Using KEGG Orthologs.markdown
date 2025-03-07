---
layout: post
title:  "[2024]Machine Learning-Based Functional Prediction of Microbial Genomes Using KEGG Orthologs"  
date:   2025-03-06 22:12:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


86개의 대사 및 생태학적 기능 예측하는 머신러닝 모델  



짧은 요약(Abstract) :    



최근 미생물 유전체의 재구성이 급증하고 있지만, 실험적 분석을 통한 기능적 특성화는 여전히 비효율적이다. 본 연구에서는 머신러닝을 활용하여 미생물 유전체의 기능을 신속하고 자동으로 분류하는 전략을 개발하였다. 86개의 대사 및 생태학적 기능(예: 메탄 소비, 플라스틱 분해 등)을 예측하는 모델을 구축하였으며, 독립적인 데이터셋에서 테스트한 결과, 70% 이상의 완전성을 가진 유전체에서 강력한 성능을 보였다. 이 모델을 Biogas Microbiome 데이터베이스에 적용한 결과, 기존 생물학적 지식과 일치하는 결과를 도출했으며, 고세균 유전체의 기능적 차이를 효과적으로 탐지하였다. 또한, 'acetoclastic methanogenesis' 사례 연구를 통해 모델의 확장 가능성을 확인하였다. 최종적으로, MICROPHERRET 도구는 86개의 모델을 포함하며, 고품질 및 저품질의 미생물 유전체 분석에 적용할 수 있어 미생물 군집 내 기능적 역할을 이해하는 데 기여할 수 있다.

---


In recent years, the number of microbial genomes reconstructed through shotgun sequencing has rapidly increased, yet experimental characterization of these genomes remains inefficient. This study leverages machine learning to develop a swift and automated strategy for microbial genome functional classification. We established models predicting 86 metabolic and ecological functions, such as methanotrophy and plastic degradation. Tests on independent datasets demonstrated robust performance for genomes with completeness above 70%. Applying the models to the Biogas Microbiome database yielded results consistent with current biological knowledge and effectively captured functional nuances in archaeal genomes. A case study on acetoclastic methanogenesis demonstrated the adaptability of the developed machine learning models. Ultimately, MICROPHERRET incorporates 86 models and can be applied to both high-quality and low-quality microbial genomes, aiding in understanding the functional role of newly generated genomes within their micro-ecological context.



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




본 연구에서는 미생물 유전체의 기능적 분류를 위해 **MICROPHERRET**라는 머신러닝 기반 도구를 개발하였다. 이를 위해 다양한 **지도 학습(supervised learning) 알고리즘**을 활용하여 미생물 유전체 데이터를 분석하고 기능을 예측하였다. 주요 구성 요소는 다음과 같다.

#### **1. 모델 (Machine Learning Models)**
MICROPHERRET는 86개의 기능별 이진 분류기(binary classifier)로 구성되며, 각 분류기는 특정한 미생물 대사 또는 생태학적 기능을 예측하도록 설계되었다. 모델은 다음과 같은 머신러닝 알고리즘을 사용하여 학습되었다.

- **로지스틱 회귀 (Logistic Regression, LR)**
- **랜덤 포레스트 (Random Forest, RF)**
- **서포트 벡터 머신 (Support Vector Machine, SVM)**
- **인공 신경망 (Neural Network, NN)**

모델 선택 과정에서 가장 높은 Matthew’s Correlation Coefficient (MCC) 값을 보이는 모델을 기능별 최적 모델로 선정하였다.

#### **2. 아키텍처 (Architecture)**
모든 모델은 **이진 분류(Binary Classification)** 방식으로 구축되었으며, 각 기능에 대해 개별적인 분류기를 학습하였다. 
- **훈련 데이터의 특징(feature)은 KEGG Orthologs (KO) 주석**을 기반으로 구성되었으며, 각 유전체의 KO 수(copy number)가 입력 데이터로 사용되었다.
- **각 기능별 이진 분류기(classifier) 구조**  
  - **입력(Input)**: 특정 미생물 유전체에서 발견된 KO의 카운트 데이터  
  - **출력(Output)**: 특정 기능(예: 메탄 생성 여부)에 대한 존재 여부(0 또는 1)  
  - **최적화 기법**: 하이퍼파라미터 튜닝(grid search)을 통해 각 모델을 최적화하였으며, **교차 검증(nested cross-validation, 3-fold CV)**을 활용하여 일반화 성능을 평가하였다.
  - **평가 지표**: MCC(Matthew’s Correlation Coefficient) 값을 기준으로 모델 성능을 비교하였다.

#### **3. 훈련 데이터 (Training Data)**
본 연구에서는 **FAPROTAX 데이터베이스**를 이용하여 미생물 유전체와 해당 기능 간의 관계를 학습 데이터로 구축하였다.
- **유전체 데이터셋**  
  - NCBI RefSeq 데이터베이스에서 14,364개의 미생물 유전체를 수집  
  - 해당 유전체들은 7,948종(species), 39개의 문(phylum)에 속함  
  - 유전체 품질 검사를 거쳐 **90% 이상의 완전성(completeness)과 5% 이하의 오염률(contamination)을 가진 데이터만 사용**  
- **라벨(Label)**  
  - FAPROTAX의 5,008 개의 미생물 그룹을 86개의 기능(예: 메탄 생성, 질소 고정, 플라스틱 분해 등)으로 매핑  
  - 기능별로 해당하는 유전체를 양성(1) 및 음성(0) 데이터로 설정하여 학습

#### **4. 모델 검증 (Validation)**
훈련된 모델은 3가지 독립적인 데이터셋에서 평가되었다.
1. **FAPROTAX에서 독립적으로 수집된 4,146개의 유전체 데이터셋**  
   - 기존 데이터셋과 중복되지 않는 데이터를 사용하여 검증 수행
2. **메타유전체 데이터셋 (Biogas Microbiome Database, 4,568 MAGs)**
   - 635개의 MAGs(메타유전체 어셈블리된 유전체)에서 기능 예측 정확도 검증
3. **인위적으로 조각난 유전체 (Simulated Fragmented Genomes, SFG)**
   - 완전한 유전체에서 10%, 30%, 50%를 무작위로 제거한 데이터셋을 생성하여 기능 예측 성능 테스트

MICROPHERRET는 **70% 이상의 유전체 완전성을 가지면 강력한 성능을 유지**했으며, 특히 탄소 및 질소 대사 관련 기능에서 높은 예측 정확도를 보였다.

---



This study developed **MICROPHERRET**, a machine learning-based tool for functional classification of microbial genomes. The methodology consists of three main components: **models, architecture, and training data**.

#### **1. Models (Machine Learning Models)**
MICROPHERRET consists of **86 binary classifiers**, each predicting a specific metabolic or ecological function of microbes. The following machine learning algorithms were used:

- **Logistic Regression (LR)**
- **Random Forest (RF)**
- **Support Vector Machine (SVM)**
- **Neural Network (NN)**

For each function, the best-performing model was selected based on the highest **Matthew’s Correlation Coefficient (MCC)** score.

#### **2. Architecture**
Each classifier was structured as a **binary classification model**, predicting the presence or absence of a given function based on microbial genome features.
- **Input Features**: KO annotations from KEGG Orthologs (KO), with KO copy numbers used as input data.
- **Output**: A binary label (0 or 1) indicating whether a genome performs a given function.
- **Optimization**: Models were fine-tuned using **grid search-based hyperparameter tuning**, and **nested cross-validation (3-fold CV)** was applied to evaluate generalization performance.
- **Evaluation Metrics**: Performance was assessed using the **MCC (Matthew’s Correlation Coefficient)**.

#### **3. Training Data**
The model was trained using **FAPROTAX**, a database that maps microbial taxa to known functional traits.
- **Genome Dataset**  
  - Collected 14,364 microbial genomes from **NCBI RefSeq**  
  - Represented **7,948 species across 39 phyla**  
  - Only high-quality genomes were included (**≥90% completeness, ≤5% contamination**)  
- **Labels (Functional Categories)**  
  - 5,008 microbial taxa from FAPROTAX were mapped to **86 functional traits**  
  - Each function was assigned a binary label (1 for presence, 0 for absence)  

#### **4. Model Validation**
The trained classifiers were evaluated on three independent datasets:
1. **4,146 microbial genomes from FAPROTAX (Independent Test Set)**
   - This dataset contained genomes not included in the training set, allowing for independent performance validation.
2. **Biogas Microbiome Database (4,568 MAGs)**
   - Functional predictions were tested on 635 MAGs (metagenome-assembled genomes) from anaerobic digestion environments.
3. **Simulated Fragmented Genomes (SFG)**
   - The dataset included artificially fragmented genomes at **90%, 70%, and 50% completeness** to assess classification performance under varying levels of genome completeness.

The tool demonstrated **robust performance for genomes with completeness above 70%**, particularly in functions related to **carbon and nitrogen metabolism**.


   
 
<br/>
# Results  





본 연구에서는 **MICROPHERRET**의 성능을 평가하기 위해 다양한 비교 모델과 테스트 데이터셋을 활용하였으며, 주요 평가 지표로 **Matthew’s Correlation Coefficient (MCC)**를 사용하였다.

---

### **1. 비교 모델 (Baseline Models)**
MICROPHERRET의 성능을 기존 모델과 비교하기 위해 **GenePhene**을 주요 비교 대상으로 선정하였다.  
- **GenePhene**: 기존 연구에서 **FAPROTAX 데이터와 NCBI 유전체**를 활용하여 84개 기능을 예측하는 머신러닝 기반 모델.
  - **주요 차이점**  
    1. GenePhene은 **로지스틱 회귀(Logistic Regression, LR)** 단일 모델만 사용한 반면, MICROPHERRET는 **SVM, RF, LR, NN** 등 다양한 알고리즘을 적용하여 최적의 모델을 선택.  
    2. MICROPHERRET는 **하이퍼파라미터 튜닝과 최적화 과정(nested cross-validation 포함)**을 적용하여 더 높은 정확도를 달성.  
    3. 기능 예측 시 **MAG(메타유전체 어셈블리된 유전체)에서도 높은 성능 유지**, GenePhene은 MAG에서 성능 저하.  
    4. GenePhene은 **NCBI 데이터에서 수집된 9,407개 유전체**를 사용한 반면, MICROPHERRET는 **14,364개 유전체를 훈련 데이터로 사용**하여 더 다양한 미생물 그룹을 학습.  

---

### **2. 테스트 데이터 (Test Datasets)**
모델의 성능을 다양한 독립적인 데이터셋에서 평가하였다.

1. **FAPROTAX 독립 테스트 세트 (4,146개 유전체)**
   - 학습 데이터에 포함되지 않은 **4,146개 유전체**를 활용하여 모델의 일반화 성능 검증.  
   - 67개의 기능을 수행하는 유전체가 포함되었으며, 이를 기반으로 MCC 값 계산.  
   - **MICROPHERRET가 GenePhene보다 63%의 기능에서 더 높은 성능을 보였음** (MCC 기준).  

2. **Biogas Microbiome Database (4,568 MAGs)**
   - **635개 MAGs**에서 74개의 기능을 수행하는 미생물 검출.  
   - 일부 기능(예: 메탄 생성, 질소 고정 등)에서 **MCC > 0.7**을 달성하여 높은 예측 성능을 확인.  
   - MAG 데이터셋에서 **MICROPHERRET가 GenePhene보다 더 낮은 오탐지율(false positive rate)을 보였음**.  

3. **인위적으로 조각난 유전체 (Simulated Fragmented Genomes, SFG)**
   - 4,146개 완전한 유전체를 **90%, 70%, 50%로 무작위로 조각낸 데이터**를 생성하여 테스트.  
   - **70% 이상의 완전성을 가진 유전체에서 대부분의 기능 예측이 유지됨 (MCC > 0.7)**
   - **50% 이하의 완전성을 가지면 예측 성능이 급격히 저하됨** (MCC < 0.5)  

---

### **3. 평가 지표 (Metrics)**
모델 성능을 평가하기 위해 다음과 같은 지표를 활용하였다.

- **Matthew’s Correlation Coefficient (MCC)**  
  - 불균형 데이터에서도 신뢰할 수 있는 성능 평가 지표로, **0.7 이상이면 우수한 모델로 간주됨**.  
  - GenePhene과 비교하여 **MICROPHERRET는 63%의 기능에서 높은 MCC 점수를 기록**.

- **혼동 행렬 (Confusion Matrix)**  
  - 독립 테스트 세트에서 예측 정확도를 확인하고, 기능별 참 양성(True Positive, TP), 참 음성(True Negative, TN), 거짓 양성(False Positive, FP), 거짓 음성(False Negative, FN)을 분석.  

- **기능별 성능 분석**  
  - **탄소 대사(Carbon Metabolism): MCC = 0.97 (최고 성능)**  
  - **질소 대사(Nitrogen Metabolism): MCC = 0.94**  
  - **유황 대사(Sulfur Metabolism): MCC = 0.57 (상대적으로 낮음)**  
  - **광영양(Phototrophy): MCC = 0.37 (낮은 성능, 개선 필요)**  

---

### **4. 주요 결과 요약**
1. **MICROPHERRET는 GenePhene보다 전반적으로 더 높은 성능을 기록하였으며, 특히 MAG 데이터에서 더 나은 성능을 보임.**
2. **70% 이상의 완전성을 가진 유전체에서 높은 예측 정확도를 유지하였으나, 50% 이하에서는 성능이 저하됨.**
3. **기능별 성능 차이가 있으며, 탄소 및 질소 대사 기능에서는 높은 정확도를 기록하였으나, 광영양 관련 기능 예측에는 한계가 있었음.**
4. **유전체의 주요 기능을 결정하는 KO(KEGG Orthologs) 목록을 추출하여, 예측 성능 향상을 위한 분석 기반을 제공.**

---



### **1. Baseline Models (Comparison)**
To evaluate MICROPHERRET’s performance, we compared it against **GenePhene**, an existing machine learning-based functional prediction tool.

- **GenePhene**: A model trained using **FAPROTAX and NCBI genomes**, capable of predicting 84 functions.  
  - **Key Differences**  
    1. GenePhene used **only Logistic Regression (LR)**, while MICROPHERRET implemented **SVM, RF, LR, and NN** for optimal model selection.  
    2. MICROPHERRET applied **hyperparameter tuning and nested cross-validation**, improving accuracy.  
    3. MICROPHERRET performed significantly better for **metagenome-assembled genomes (MAGs)**, where GenePhene’s performance degraded.  
    4. GenePhene was trained on **9,407 genomes**, while MICROPHERRET used a **larger dataset of 14,364 genomes**, covering more microbial diversity.  

---

### **2. Test Datasets**
To validate performance, we tested MICROPHERRET on three independent datasets.

1. **FAPROTAX Independent Test Set (4,146 genomes)**
   - **Genomes not included in the training set**, allowing independent generalization testing.  
   - 67 functions were represented, and **MCC scores were calculated for each function**.  
   - **MICROPHERRET outperformed GenePhene in 63% of cases (based on MCC scores).**  

2. **Biogas Microbiome Database (4,568 MAGs)**
   - **635 MAGs were associated with 74 functions.**  
   - Some functions (e.g., **methanogenesis, nitrogen fixation**) showed **MCC > 0.7**, confirming high prediction accuracy.  
   - MICROPHERRET showed **a lower false positive rate than GenePhene** in MAG datasets.  

3. **Simulated Fragmented Genomes (SFG)**
   - Created **90%, 70%, and 50% fragmented genome datasets** to test prediction robustness.  
   - **Above 70% completeness, predictions remained strong (MCC > 0.7).**  
   - **Below 50% completeness, performance dropped significantly (MCC < 0.5).**  

---

### **3. Evaluation Metrics**
We used the following metrics to assess model performance:

- **Matthew’s Correlation Coefficient (MCC)**  
  - Considered **excellent if MCC > 0.7**.  
  - Compared to GenePhene, **MICROPHERRET achieved higher MCC scores for 63% of functions**.

- **Confusion Matrix**  
  - Analyzed True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN) rates.  

- **Function-Specific Performance**  
  - **Carbon metabolism: MCC = 0.97 (Best performance)**  
  - **Nitrogen metabolism: MCC = 0.94**  
  - **Sulfur metabolism: MCC = 0.57 (Lower performance)**  
  - **Phototrophy: MCC = 0.37 (Needs improvement)**  

---

### **4. Key Findings**
1. **MICROPHERRET outperforms GenePhene, especially in MAG datasets.**  
2. **Maintains high accuracy for genomes ≥70% completeness.**  
3. **Higher accuracy in carbon and nitrogen metabolism; lower accuracy in phototrophy.**  
4. **Extracts key KO genes for function prediction improvement.**


<br/>
# 예제  





본 연구에서 사용된 데이터는 **FAPROTAX 기반의 유전체 데이터**와 **Biogas Microbiome 데이터**이며, 각각 **훈련 데이터(Training Data)와 테스트 데이터(Test Data)**로 활용되었다. 모델의 인풋과 아웃풋 구조를 설명하기 위해 예제 데이터를 제시한다.

---

### **1. 훈련 데이터 (Training Data Example)**
훈련 데이터는 **KEGG Orthologs (KO) 어노테이션**을 사용하여 특정 유전체가 수행하는 기능을 예측하도록 구성되었다.

#### **훈련 데이터 형식**
| Genome_ID | KO_0001 | KO_0002 | KO_0003 | ... | KO_0100 | Function_Label |
|-----------|--------|--------|--------|-----|--------|----------------|
| GCF_00001 | 3      | 0      | 1      | ... | 2      | 1 (Methanogenesis) |
| GCF_00002 | 0      | 2      | 0      | ... | 1      | 0 (No Function) |
| GCF_00003 | 1      | 0      | 3      | ... | 1      | 1 (Denitrification) |

- **Genome_ID**: 특정 미생물 유전체의 고유 식별자
- **KO_XXXX**: 해당 유전체에서 특정 KEGG Orthologs(KO)의 출현 빈도(count)
- **Function_Label**: 해당 유전체가 특정 기능(예: 메탄 생성, 질소 고정)을 수행하는지 여부 (1=기능 수행, 0=기능 없음)

---

### **2. 테스트 데이터 (Test Data Example)**
테스트 데이터는 훈련 데이터와 유사한 형식이지만, 모델이 예측해야 할 **Function_Label**이 숨겨진 상태로 제공된다.

#### **테스트 데이터 형식**
| Genome_ID | KO_0001 | KO_0002 | KO_0003 | ... | KO_0100 |
|-----------|--------|--------|--------|-----|--------|
| GCF_10001 | 2      | 0      | 1      | ... | 3      |
| GCF_10002 | 0      | 1      | 0      | ... | 0      |
| GCF_10003 | 3      | 2      | 1      | ... | 1      |

- **훈련 데이터와 동일한 KO 어노테이션을 포함하되, Function_Label이 없음**
- 모델이 예측한 결과를 이후에 평가하여 성능을 검증

---

### **3. 모델의 입력/출력 예시 (Task Input/Output Example)**
MICROPHERRET의 입력 및 출력 예제는 아래와 같다.

#### **입력 (Model Input)**
```json
{
  "Genome_ID": "GCF_10001",
  "KO_features": {
    "KO_0001": 2,
    "KO_0002": 0,
    "KO_0003": 1,
    "...": "...",
    "KO_0100": 3
  }
}
```

#### **출력 (Model Output)**
```json
{
  "Genome_ID": "GCF_10001",
  "Predicted_Functions": [
    "Methanogenesis",
    "Sulfate Reduction"
  ]
}
```
- 모델이 주어진 유전체의 KO 특징을 기반으로 특정 기능 수행 여부를 예측
- `Predicted_Functions` 리스트에 예측된 기능들이 포함됨

---



This study used **FAPROTAX-based genome data** and **Biogas Microbiome data**, divided into **training and test datasets**. The following examples illustrate the model’s input and output structure.

---

### **1. Training Data Example**
Training data were structured using **KEGG Orthologs (KO) annotations**, allowing the model to predict specific microbial functions.

#### **Training Data Format**
| Genome_ID | KO_0001 | KO_0002 | KO_0003 | ... | KO_0100 | Function_Label |
|-----------|--------|--------|--------|-----|--------|----------------|
| GCF_00001 | 3      | 0      | 1      | ... | 2      | 1 (Methanogenesis) |
| GCF_00002 | 0      | 2      | 0      | ... | 1      | 0 (No Function) |
| GCF_00003 | 1      | 0      | 3      | ... | 1      | 1 (Denitrification) |

- **Genome_ID**: Unique microbial genome identifier
- **KO_XXXX**: Presence count of a specific KEGG Ortholog (KO) in the genome
- **Function_Label**: Indicates whether the genome performs a given function (1=Function present, 0=No function)

---

### **2. Test Data Example**
The test dataset follows the same format as the training dataset but **does not include the Function_Label**, requiring the model to predict it.

#### **Test Data Format**
| Genome_ID | KO_0001 | KO_0002 | KO_0003 | ... | KO_0100 |
|-----------|--------|--------|--------|-----|--------|
| GCF_10001 | 2      | 0      | 1      | ... | 3      |
| GCF_10002 | 0      | 1      | 0      | ... | 0      |
| GCF_10003 | 3      | 2      | 1      | ... | 1      |

- **KO annotations are present, but Function_Label is missing**
- Model predictions will be evaluated afterward to assess accuracy

---

### **3. Task Input/Output Example**
Below is an example of **input and output** from MICROPHERRET.

#### **Input (Model Input)**
```json
{
  "Genome_ID": "GCF_10001",
  "KO_features": {
    "KO_0001": 2,
    "KO_0002": 0,
    "KO_0003": 1,
    "...": "...",
    "KO_0100": 3
  }
}
```

#### **Output (Model Output)**
```json
{
  "Genome_ID": "GCF_10001",
  "Predicted_Functions": [
    "Methanogenesis",
    "Sulfate Reduction"
  ]
}
```
- The model predicts specific **functional traits** based on KO features.
- `Predicted_Functions` contains the **list of predicted microbial functions** for the given genome.


<br/>  
# 요약   



본 연구에서는 KEGG Orthologs(KO) 어노테이션을 활용한 머신러닝 모델(MICROPHERRET)을 개발하여 미생물 유전체의 기능을 예측하였다. 결과적으로, 70% 이상의 완전성을 가진 유전체에서 높은 성능(MCC > 0.7)을 보였으며, 기존 모델(GenePhene)보다 MAG 데이터에서 더 정확한 예측을 수행하였다. 예제 데이터에서는 KO 수치가 입력으로 제공되며, 모델은 해당 유전체가 수행하는 기능(예: 메탄 생성, 질소 고정 등)을 예측한다.  

---


This study developed the machine learning model **MICROPHERRET**, which predicts microbial genome functions using **KEGG Orthologs (KO) annotations**. The results showed **high performance (MCC > 0.7)** for genomes with ≥70% completeness, outperforming **GenePhene** in MAG datasets. In example data, KO counts are provided as input, and the model predicts microbial functions such as **methanogenesis and nitrogen fixation**.


<br/>  
# 기타  



논문의 **피규어(Figure), 테이블(Table), 그래프(Graph)** 를 기반으로 주요 내용을 설명하겠습니다.

---


#### **Figure 1:**  
**FAPROTAX 및 NCBI 데이터베이스에 등록된 박테리아 및 고세균 게놈 수량 분석**  
- (A) FAPROTAX 및 NCBI에서 박테리아 계통군(phyla)별로 분류된 게놈 수  
- (B) 고세균 계통군(phyla)별로 분류된 게놈 수  
- (C) 특정 기능과 연관된 게놈 수  
- 하나의 게놈이 여러 기능에 속할 수 있음  

#### **Figure 2:**  
**각 기능별 머신러닝 모델 성능 비교 (MCC 기준)**  
- 네 가지 알고리즘(Logistic Regression, Random Forest, SVM, Neural Network)의 MCC 성능 비교  
- 상위 25개 기능에 대한 평가 그래프  
- 대부분의 기능에서 SVM과 Random Forest 모델이 높은 성능을 보임  
- 특정 기능(예: "dark sulfite oxidation")에서는 Neural Network가 상대적으로 높은 성능을 기록  

#### **Figure 3:**  
**MICROPHERRET와 GenePhene의 비교**  
- MICROPHERRET가 63%의 기능에서 GenePhene보다 높은 성능을 보였음  
- 8%에서는 동일한 성능을 기록, 29%에서는 GenePhene이 우수함  
- GenePhene은 더 많은 False Positive를 생성하는 경향이 있음  

#### **Figure 4:**  
**시뮬레이션된 부분적 게놈(90%, 70%, 50%)에서 모델 성능 변화**  
- 게놈이 불완전할수록 예측 성능(MCC)이 점진적으로 감소  
- 70% 이상의 완전성을 가진 게놈에서는 비교적 높은 예측 성능 유지  
- 50% 이하에서는 대부분의 기능 예측 성능이 급격히 하락  

#### **Figure 5:**  
**Biogas Microbiome 데이터베이스에서 MICROPHERRET의 기능 예측 결과**  
- (A) 특정 기능이 0.5% 이상의 게놈에서 발견된 비율  
- (B) Biogas 데이터에서 발견된 고세균(MAGs)의 기능 분포  
- 메탄 생성(methanogenesis)은 주로 유리균문(Euryarchaeota) 내에서 확인됨  

#### **Table S2:**  
**각 기능별 최고 성능을 기록한 모델과 MCC 점수**  
- 81개의 모델이 MCC > 0.7로 우수한 성능을 기록  
- 5개 모델이 0.5~0.7 사이의 준수한 성능을 보임  
- 일부 모델(예: "oil bioremediation")은 낮은 MCC 점수를 기록하여 기능 예측이 어려움  

#### **Table S6:**  
**GenePhene과 MICROPHERRET의 비교 상세 데이터**  
- MICROPHERRET가 63개 기능 중 63%에서 더 나은 성능을 기록  
- False Positive 예측에서 MICROPHERRET가 GenePhene보다 낮음 (6,156 vs. 14,206)  

#### **Table S7:**  
**Biogas Microbiome 데이터에서 MICROPHERRET의 기능 예측 결과 검증**  
- MAG의 완전성이 높을수록 모델의 예측 성능이 높음  
- 90% 이상의 완전성을 가진 MAG에서 평균 MCC 0.74를 기록  

#### **Table S8:**  
**고세균(MAGs)의 기능 분류 결과**  
- 대부분의 메탄 생성 기능이 Euryarchaeota에서 발견됨  
- 일부 미생물에서는 메탄 생성과 관련된 주요 유전자 부족  

---


#### **Figure 1:**  
**Genome Distribution in FAPROTAX and NCBI**  
- (A) Number of bacterial genomes in FAPROTAX and NCBI  
- (B) Number of archaeal genomes in FAPROTAX and NCBI  
- (C) Number of genomes associated with specific functions  
- A single genome may be linked to multiple functions  

#### **Figure 2:**  
**Performance of Machine Learning Models for Each Function (MCC Scores)**  
- Comparison of four algorithms (Logistic Regression, Random Forest, SVM, Neural Network)  
- SVM and Random Forest achieved high accuracy for most functions  
- Neural Network outperformed in a few cases like “dark sulfite oxidation”  

#### **Figure 3:**  
**Comparison of MICROPHERRET and GenePhene**  
- MICROPHERRET outperformed GenePhene in 63% of the functions  
- In 8% of cases, both performed equally well; GenePhene was better in 29%  
- GenePhene had a higher false positive rate  

#### **Figure 4:**  
**Impact of Genome Completeness on Model Performance**  
- As genome completeness decreases, MCC score drops  
- Performance is stable for genomes ≥70% completeness  
- At 50% completeness, model accuracy declines significantly  

#### **Figure 5:**  
**Functional Predictions in Biogas Microbiome Database**  
- (A) Percentage of MAGs performing each function (≥0.5% of genomes)  
- (B) Distribution of predicted functions in archaeal MAGs  
- Methanogenesis is predominantly found in **Euryarchaeota**  

#### **Table S2:**  
**Best Performing Model per Function (MCC Scores)**  
- 81 models had MCC > 0.7 (high accuracy)  
- 5 models had MCC between 0.5–0.7 (moderate accuracy)  
- Some models (e.g., “oil bioremediation”) struggled with poor MCC scores  

#### **Table S6:**  
**Detailed Comparison of MICROPHERRET vs. GenePhene**  
- MICROPHERRET outperformed GenePhene in 63% of cases  
- MICROPHERRET produced fewer false positives (6,156 vs. 14,206)  

#### **Table S7:**  
**Validation of MICROPHERRET in Biogas Microbiome Data**  
- Higher genome completeness correlates with higher prediction accuracy  
- MAGs with >90% completeness had an average MCC of 0.74  

#### **Table S8:**  
**Functional Classification of Archaeal MAGs**  
- Most methanogenesis-associated genomes belong to **Euryarchaeota**  
- Some archaea lacked key genes necessary for methanogenesis  

---




<br/>
# refer format:     


@article{Bizzotto2024,
  author    = {Edoardo Bizzotto and Sofia Fraulini and Guido Zampieri and Esteban Orellana and Laura Treu and Stefano Campanaro},
  title     = {MICROPHERRET: MICRObial PHEnotypic tRait ClassifieR using Machine lEarning Techniques},
  journal   = {Environmental Microbiome},
  volume    = {19},
  number    = {58},
  year      = {2024},
  doi       = {10.1186/s40793-024-00600-6},
  url       = {https://doi.org/10.1186/s40793-024-00600-6}
}



Bizzotto, Edoardo, Sofia Fraulini, Guido Zampieri, Esteban Orellana, Laura Treu, and Stefano Campanaro. "MICROPHERRET: MICRObial PHEnotypic tRait ClassifieR using Machine lEarning Techniques." Environmental Microbiome 19, no. 58 (2024). https://doi.org/10.1186/s40793-024-00600-6.