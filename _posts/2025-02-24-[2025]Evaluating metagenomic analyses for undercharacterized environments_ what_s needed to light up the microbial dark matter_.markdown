---
layout: post
title:  "[2025]Evaluating metagenomic analyses for undercharacterized environments: what’s needed to light up the microbial dark matter?"  
date:   2025-02-24 12:52:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    



비인간 관련 미생물 군집은 중요한 생물학적 역할을 하지만, 인간과 연관된 미생물 군집보다 상대적으로 덜 연구되어 왔습니다. 이 논문에서는 다양한 최신 메타게놈 분석 방법들이 환경 샘플의 주요 특성에 어떻게 영향을 받는지 평가하였습니다. 

시뮬레이션된 데이터셋에서는 모든 방법이 높은 분류학적 수준에서는 유사한 성능을 보였지만, 메타게놈 어셈블된 게놈(MAGs)을 활용하는 최신 마커 기반 방법들이 낮은 분류 수준에서 더 뛰어난 성능을 발휘했습니다. 실제 환경 샘플에서는 같은 샘플에 대해 서로 다른 방법들이 할당한 분류학적 프로파일이 낮은 수준에서 일치하지 않았지만, 군집의 다양성 평가 및 환경 변수와 미생물 프로파일 간의 관계 추정에서는 더 나은 일관성을 보였습니다. 

이 연구는 환경 미생물 군집을 보다 효과적으로 해석하고, 미생물 "암흑 물질(microbial dark matter)"을 밝히는 데 필요한 분석 방법 개선을 위한 방향을 제시합니다.


Non-human-associated microbial communities play important biological roles, but they remain less understood than human-associated communities. Here, we assess the impact of key environmental sample properties on a variety of state-of-the-art metagenomic analysis methods.

In simulated datasets, all methods performed similarly at high taxonomic ranks, but newer marker-based methods incorporating metagenomic assembled genomes (MAGs) outperformed others at lower taxonomic levels. In real environmental data, taxonomic profiles assigned to the same sample by different methods showed little agreement at lower taxonomic levels, but the methods agreed better on community diversity estimates and estimates of the relationships between environmental parameters and microbial profiles.

This analysis will help inform the interpretation of environmental microbial communities and guide future development of taxonomic profilers, particularly in illuminating microbial "dark matter."

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





이 논문에서는 다양한 최신 메타게놈 분석 방법을 평가하기 위해 **합성 데이터셋과 실제 환경 데이터**를 활용하였다. 연구 방법은 크게 다음과 같이 정리할 수 있다.

#### **1. 데이터셋**
- **합성 데이터셋**: 다양한 환경(토양, 해양, 동물 장내 미생물 군집)을 반영한 300개의 합성 샘플을 생성함.
- **실제 환경 데이터셋**: 다양한 생태계(산림 토양, 해안 퇴적물, 극지 해양, 염습지, 산성 광산 배수 등)에서 얻은 **348개 환경 샘플**을 활용함.

#### **2. 메타게놈 분석 방법**
두 가지 주요 접근법이 사용되었다.
1. **참조 기반(Reference-based) 방법**:
   - Centrifuge, Kraken 2/Bracken 2 (k-mer 기반)
   - mOTUs3, Metaxa2 (유니버설 마커 기반)
   - MetaPhlAn 2/3/4 (특정 마커 기반)
2. **어셈블리 기반(Assembly-based) 방법**:
   - MEGAHIT 또는 metaSPAdes (컨티그 생성)
   - MetaBAT (게놈 클러스터링)
   - CheckM2 (품질 검사)
   - PhyloPhlAn 또는 GTDB-Tk (계통 분석 및 분류)

#### **3. 모델 및 훈련 데이터**
- **모델 사용 방식**: 각 방법에 대해 **2022년 최신 데이터베이스**를 활용하여 사전 학습된 모델을 평가함.
- **훈련 데이터**: 데이터베이스 업데이트 당시의 **RefSeq, GTDB-Tk, PhyloPhlAn** 등의 공공 데이터셋을 이용함.
- **평가 방법**: 300개의 합성 샘플과 348개의 실제 환경 샘플에서 다양한 샘플 특성을 변화시키면서 분석함.

#### **4. 평가 지표**
- **분류 정확도**: F1-score 및 Bray-Curtis dissimilarity로 평가.
- **군집 구조 평가**: α-다양성(Inverse Simpson Index) 및 β-다양성(Bray-Curtis 거리) 비교.
- **환경 변수와의 연관성**: PERMANOVA, MaAsLin 2를 사용하여 환경 변수와 미생물 프로파일 간의 관계 분석.

---



This study evaluates various state-of-the-art **metagenomic analysis methods** using both **synthetic and real environmental datasets**.

#### **1. Datasets**
- **Synthetic datasets**: 300 synthetic samples were generated to reflect different environments (soil, ocean, animal gut microbiomes).
- **Real environmental datasets**: A total of **348 environmental samples** were collected from diverse ecosystems, including forest soil, coastal sediment, polar ocean, salt marsh, and acid mine drainage.

#### **2. Metagenomic Analysis Methods**
The study employs two major approaches:
1. **Reference-based methods**:
   - Centrifuge, Kraken 2/Bracken 2 (k-mer-based)
   - mOTUs3, Metaxa2 (universal marker-based)
   - MetaPhlAn 2/3/4 (unique marker-based)
2. **Assembly-based methods**:
   - MEGAHIT or metaSPAdes (contig construction)
   - MetaBAT (genome binning)
   - CheckM2 (quality control)
   - PhyloPhlAn or GTDB-Tk (phylogenetic classification)

#### **3. Models and Training Data**
- **Model Usage**: Pre-trained models from **2022 updated databases** were used for evaluation.
- **Training Data**: Public datasets from **RefSeq, GTDB-Tk, and PhyloPhlAn** were used at the time of database updates.
- **Evaluation Method**: The models were tested on **300 synthetic samples and 348 real environmental samples**, varying sample parameters.

#### **4. Evaluation Metrics**
- **Classification Accuracy**: Measured using **F1-score** and **Bray-Curtis dissimilarity**.
- **Community Structure Analysis**: **Alpha diversity (Inverse Simpson Index)** and **Beta diversity (Bray-Curtis distance)** comparisons.
- **Environmental Correlation Analysis**: **PERMANOVA** and **MaAsLin 2** were used to examine relationships between microbial profiles and environmental parameters.






   
 
<br/>
# Results  



---

### **1. 메타게놈 분석 방법 성능 비교**
- 모든 방법이 **높은 분류학적 수준(예: 문, 과 수준)**에서는 유사한 성능을 보였다.
- **최신 마커 기반 방법(MetaPhlAn 4, mOTUs3)**이 **낮은 분류학적 수준(속, 종 수준)**에서 다른 방법보다 더 뛰어난 성능을 보였다.
- 실제 환경 데이터에서, 동일한 샘플에 대해 서로 다른 방법들이 낮은 분류 수준에서 일치도가 낮았지만, **군집 다양성 평가 및 환경 변수와의 연관성 분석에서는 더 높은 일치도를 보였다.**

---

### **2. 모델 비교 (경쟁 모델 성능)**
#### **1) 참조 기반 방법 (Reference-based Methods)**
- **MetaPhlAn 4**: 모든 환경에서 가장 높은 F1-score와 낮은 Bray-Curtis dissimilarity를 기록함.
- **mOTUs3**: 비교적 높은 정밀도를 보였지만 리콜이 낮음.
- **Kraken 2/Bracken 2**: 중간 수준의 성능, 정밀도와 리콜이 균형적.
- **Metaxa2**: 성능이 가장 낮았으며 환경별 편차가 컸음.

#### **2) 어셈블리 기반 방법 (Assembly-based Methods)**
- **MEGAHIT/metaSPAdes + MetaBAT + GTDB-Tk**: 
  - 낮은 분류 수준에서 매우 낮은 리콜을 보임 (F1 평균 0.16-0.24).
  - 충분한 서열 깊이(≥7.5Gbp)가 없으면 정확도가 급격히 하락함.
  - 기존 데이터베이스에 없는 종을 일부 복원할 수 있다는 강점이 있음.

---

### **3. 테스트 데이터 결과**
- **합성 데이터셋(300개 샘플) 결과**:
  - 모든 방법이 **상위 계통수준(Phylum, Class)**에서는 유사한 성능을 보였음.
  - **MetaPhlAn 4**가 **종(Species) 수준에서 가장 높은 정확도를 기록**.
  - **어셈블리 기반 방법은 낮은 깊이에서 성능이 떨어짐**.

- **실제 환경 데이터셋(348개 샘플) 결과**:
  - 낮은 분류 수준에서는 서로 다른 방법 간 **일관성이 매우 낮았음**.
  - 그러나 **군집 구조 분석(Alpha/Beta Diversity) 및 환경 요인과의 상관관계에서는 높은 일관성을 보임**.
  - PERMANOVA 및 MaAsLin 2를 통해 환경 요인과 미생물 군집 간의 중요한 관계를 도출함.

---

### **4. 핵심 결론**
- **마커 기반 방법(MetaPhlAn 4, mOTUs3)이 가장 높은 종 수준 정확도를 기록**.
- **어셈블리 기반 방법은 기존 데이터베이스에 없는 종을 복원하는 데 유용**.
- **같은 샘플에 대해 서로 다른 방법이 매우 다른 프로파일을 생성**하므로, 환경 미생물 연구에서는 다중 접근법을 사용하는 것이 중요함.

---

---



---

### **1. Metagenomic Analysis Method Performance Comparison**
- All methods performed **similarly at higher taxonomic ranks (e.g., phylum, family).**
- **Newer marker-based methods (MetaPhlAn 4, mOTUs3) outperformed others at lower taxonomic levels (genus, species).**
- In real environmental data, taxonomic profiles assigned to the same sample **varied significantly at lower taxonomic levels**, but **showed higher agreement in community diversity estimates and environmental correlations.**

---

### **2. Model Comparison (Competitor Model Performance)**
#### **1) Reference-based Methods**
- **MetaPhlAn 4**: Achieved **the highest F1-score and lowest Bray-Curtis dissimilarity across all environments**.
- **mOTUs3**: Showed **high precision but lower recall**.
- **Kraken 2/Bracken 2**: Moderate performance, with **balanced precision and recall**.
- **Metaxa2**: Had the **lowest performance with high variation across environments**.

#### **2) Assembly-based Methods**
- **MEGAHIT/metaSPAdes + MetaBAT + GTDB-Tk**:
  - **Very low recall at lower taxonomic levels** (F1 average: 0.16–0.24).
  - **Requires deep sequencing (≥7.5Gbp) for reasonable accuracy**.
  - **Can reconstruct previously uncharacterized species** missing from reference databases.

---

### **3. Test Data Performance**
- **Synthetic dataset (300 samples) results**:
  - All methods performed **similarly at higher taxonomic ranks (phylum, class)**.
  - **MetaPhlAn 4 had the highest accuracy at the species level.**
  - **Assembly-based methods struggled at lower sequencing depths**.

- **Real-world environmental dataset (348 samples) results**:
  - **Low agreement between methods at the species level**.
  - However, **higher agreement in community structure analysis (alpha/beta diversity) and environmental factor correlations**.
  - **PERMANOVA and MaAsLin 2 identified key relationships between environmental factors and microbial communities**.

---

### **4. Key Takeaways**
- **Marker-based methods (MetaPhlAn 4, mOTUs3) achieved the highest species-level accuracy**.
- **Assembly-based methods are useful for recovering novel species absent from reference databases**.
- **Different methods produce highly different profiles for the same sample**, highlighting the need for **a multi-method approach in environmental microbiome studies**.





### **평가 메트릭 (Evaluation Metrics)   **

이 논문에서는 다양한 메타게놈 분석 방법을 평가하기 위해 다음과 같은 평가 메트릭을 사용했다.

---

### **1. 분류 정확도 (Taxonomic Classification Accuracy)**
- **F1-score**: 예측된 미생물 군집이 실제 군집과 얼마나 일치하는지를 평가함.
- **Bray-Curtis Dissimilarity**: 예측된 미생물 군집과 실제 군집 사이의 유사도를 측정하는 거리 기반 지표. 값이 낮을수록 더 유사함.

---

### **2. 군집 구조 분석 (Community Structure Analysis)**
- **Alpha Diversity (Intra-Sample Diversity)**
  - **Inverse Simpson Index**: 하나의 샘플 내에서 미생물 다양성을 측정하는 지표. 값이 높을수록 군집 내 미생물 분포가 균등함.
  - **Shannon Index** (추가 분석): 미생물 종의 풍부도와 균등도를 평가.

- **Beta Diversity (Inter-Sample Dissimilarity)**
  - **Bray-Curtis Dissimilarity**: 샘플 간 미생물 군집의 차이를 비교하는 지표.
  - **Mantel Test**: 샘플 간 거리 행렬의 상관관계를 측정하여 메타게놈 분석 방법 간 유사도를 평가.

---

### **3. 환경 요인과 미생물 군집 연관성 (Environmental Associations)**
- **PERMANOVA (Permutational Multivariate Analysis of Variance)**
  - 환경 요인이 미생물 군집의 조성에 미치는 영향을 평가하는 다변량 통계 방법.
  - R² 값이 클수록 환경 요인이 군집 구성에 미치는 영향이 큼.

- **MaAsLin 2 (Multivariable Association with Linear Models)**
  - 환경 변수와 특정 미생물 종 간의 관계를 분석하는 통계 기법.
  - FDR (False Discovery Rate) 조정을 적용하여 다중 검정 문제를 해결.

---

### **4. 데이터베이스 기반 평가 (Database-Based Evaluation)**
- **Taxonomic Agreement Across Methods**  
  - **Intersection Over Union (Jaccard Index)**: 서로 다른 분석 방법이 동일한 샘플에서 얼마나 유사한 미생물 프로파일을 예측하는지 평가. 값이 높을수록 두 방법이 비슷한 결과를 냄.
  - **Proportion of Unknown Species Identified**: 기존 데이터베이스에 없는 종을 얼마나 효과적으로 복원할 수 있는지 평가.

---

---

### **Evaluation Metrics **

---

### **1. Taxonomic Classification Accuracy**
- **F1-score**: Measures how well the predicted microbial community matches the actual community.
- **Bray-Curtis Dissimilarity**: A distance-based metric that quantifies dissimilarity between predicted and actual microbial communities. Lower values indicate higher similarity.

---

### **2. Community Structure Analysis**
- **Alpha Diversity (Intra-Sample Diversity)**
  - **Inverse Simpson Index**: Measures microbial diversity within a single sample. Higher values indicate more even distribution of species.
  - **Shannon Index** (additional analysis): Evaluates species richness and evenness.

- **Beta Diversity (Inter-Sample Dissimilarity)**
  - **Bray-Curtis Dissimilarity**: Compares microbial community composition across samples.
  - **Mantel Test**: Measures the correlation between distance matrices to assess the similarity between metagenomic analysis methods.

---

### **3. Environmental Associations with Microbial Communities**
- **PERMANOVA (Permutational Multivariate Analysis of Variance)**
  - A multivariate statistical method used to assess the impact of environmental factors on microbial composition.
  - A higher R² value indicates a stronger influence of environmental factors on microbial communities.

- **MaAsLin 2 (Multivariable Association with Linear Models)**
  - A statistical method that identifies associations between environmental variables and specific microbial taxa.
  - Uses **False Discovery Rate (FDR) correction** to address multiple testing issues.

---

### **4. Database-Based Evaluation**
- **Taxonomic Agreement Across Methods**  
  - **Intersection Over Union (Jaccard Index)**: Evaluates how similarly different methods classify microbial profiles from the same sample. Higher values indicate more agreement.
  - **Proportion of Unknown Species Identified**: Assesses the ability of methods to recover species absent from reference databases.





<br/>
# 예제  



---

### **1. 테스트 데이터 및 훈련 데이터 예시**
#### **1) 합성 데이터셋 예시 (Synthetic Dataset Example)**
실험에서는 다양한 환경을 반영한 **300개의 합성 샘플**을 생성했다. 각 샘플에는 특정 환경에서 발견될 수 있는 미생물 종의 상대적 풍부도(abundance)가 포함됨.

| 샘플 ID | 환경 유형 | Bacteroides (%) | Firmicutes (%) | Proteobacteria (%) | 기타 미생물 (%) |
|---------|----------|---------------|---------------|----------------|----------------|
| S1      | 해양     | 10.5          | 25.3          | 40.2           | 24.0           |
| S2      | 토양     | 5.2           | 30.1          | 35.6           | 29.1           |
| S3      | 장내     | 50.0          | 20.0          | 10.0           | 20.0           |

- 이 데이터는 실제 샘플처럼 **특정 환경에서의 미생물 군집 구조를 시뮬레이션**하여 생성됨.
- 데이터의 풍부도 분포를 모델이 얼마나 정확히 예측하는지 평가.

#### **2) 실제 환경 데이터 예시 (Real Environmental Data Example)**
- **산림 토양(Forest Soil) 샘플 예제**:
  - Shannon Index: 3.4
  - 주요 미생물 종: *Acidobacteria*, *Actinobacteria*, *Proteobacteria*

- **극지 해양(Polar Ocean) 샘플 예제**:
  - Shannon Index: 2.1
  - 주요 미생물 종: *Flavobacteria*, *Alphaproteobacteria*, *Cyanobacteria*

---

### **2. 예측 작업 예시 (Prediction Task Example)**
실험에서 모델들은 **주어진 환경 샘플에 대해 미생물 종을 예측하는 작업**을 수행했다.

**입력 (Input) 예시:**
```json
{
  "샘플_ID": "S1",
  "환경": "해양",
  "염분": "3.5%",
  "온도": "15°C",
  "pH": "8.1"
}
```

**모델 예측 (Predicted Output) 예시:**
```json
{
  "예측된 미생물 군집": {
    "Bacteroides": 12.1,
    "Firmicutes": 22.5,
    "Proteobacteria": 38.9,
    "기타 미생물": 26.5
  }
}
```
- 모델이 **환경 정보(염분, 온도, pH 등)를 기반으로 특정 미생물 군집의 풍부도를 예측**.
- 실제 데이터와의 차이를 비교하여 **F1-score 및 Bray-Curtis Dissimilarity로 성능 평가**.

---

---

---

### **1. Example of Test and Training Data**
#### **1) Synthetic Dataset Example**
The experiment included **300 synthetic samples**, each reflecting microbial abundance distributions found in different environments.

| Sample ID | Environment Type | Bacteroides (%) | Firmicutes (%) | Proteobacteria (%) | Other Microbes (%) |
|-----------|-----------------|-----------------|-----------------|---------------------|---------------------|
| S1        | Ocean           | 10.5            | 25.3            | 40.2                | 24.0                |
| S2        | Soil            | 5.2             | 30.1            | 35.6                | 29.1                |
| S3        | Gut             | 50.0            | 20.0            | 10.0                | 20.0                |

- This dataset simulates **the microbial community composition in specific environments**.
- The goal is to assess how accurately a model can predict the **relative abundance of microbes**.

#### **2) Real Environmental Data Example**
- **Forest Soil Sample Example**:
  - Shannon Index: 3.4
  - Dominant microbial taxa: *Acidobacteria*, *Actinobacteria*, *Proteobacteria*

- **Polar Ocean Sample Example**:
  - Shannon Index: 2.1
  - Dominant microbial taxa: *Flavobacteria*, *Alphaproteobacteria*, *Cyanobacteria*

---

### **2. Prediction Task Example**
The models were trained to **predict microbial composition for a given environmental sample**.

**Input Example:**
```json
{
  "sample_ID": "S1",
  "environment": "ocean",
  "salinity": "3.5%",
  "temperature": "15°C",
  "pH": "8.1"
}
```

**Predicted Output Example:**
```json
{
  "predicted_microbial_abundance": {
    "Bacteroides": 12.1,
    "Firmicutes": 22.5,
    "Proteobacteria": 38.9,
    "Other_Microbes": 26.5
  }
}
```
- The model predicts **microbial composition** based on environmental factors (e.g., salinity, temperature, pH).
- Performance is evaluated using **F1-score and Bray-Curtis Dissimilarity** by comparing predictions with actual data.





<br/>  
# 요약   



이 연구는 메타게놈 분석 방법을 평가하기 위해 **300개의 합성 데이터 샘플과 348개의 실제 환경 데이터 샘플**을 사용하였다. 실험에서는 **마커 기반 방법(MetaPhlAn 4, mOTUs3)과 어셈블리 기반 방법(MEGAHIT, MetaBAT, GTDB-Tk)**을 비교하여 다양한 환경에서의 성능을 분석하였다. 평가 결과, **MetaPhlAn 4가 종 수준에서 가장 높은 정확도를 보였으며, 어셈블리 기반 방법은 기존 데이터베이스에 없는 종을 복원하는 데 유리**하였다. 예측 작업에서는 **환경 변수(염분, 온도, pH 등)를 입력으로 받아 특정 미생물 군집의 상대적 풍부도를 예측**하는 모델을 사용하였다. 결과적으로, 서로 다른 방법들이 낮은 분류 수준에서 불일치를 보였으나, **환경 요인과의 연관성 분석에서는 높은 일관성을 유지**하였다.  

---


This study evaluated metagenomic analysis methods using **300 synthetic samples and 348 real-world environmental samples**. The experiments compared **marker-based methods (MetaPhlAn 4, mOTUs3) and assembly-based methods (MEGAHIT, MetaBAT, GTDB-Tk)** to analyze their performance across different environments. The results showed that **MetaPhlAn 4 had the highest species-level accuracy, while assembly-based methods were better at recovering novel species absent from reference databases**. The prediction task involved **using environmental variables (e.g., salinity, temperature, pH) as inputs to predict the relative abundance of microbial taxa**. Ultimately, while different methods produced inconsistent taxonomic profiles at lower ranks, **they showed strong consistency in analyzing environmental associations**.  



<br/>  
# 기타  






### ** 피규어(Figures) 요약**  
논문에 포함된 주요 시각 자료들은 실험 결과를 시각적으로 나타내며, 다음과 같이 분류할 수 있다.  

#### ** Figure 1: 실험 개요 다이어그램**  
- **연구 방법론 개요**: 데이터셋 구성(합성 vs. 실제 환경), 분석 방법(마커 기반 vs. 어셈블리 기반), 평가 지표 등을 보여줌.  
- 샘플들이 어떻게 분석되었으며, 각 모델이 어떤 프로세스를 거쳤는지 정리된 흐름도(flowchart) 형태.  

#### ** Figure 2: 분류 정확도 비교 (Classification Performance)**  
- 다양한 방법들이 **Phylum~Species 수준에서 어떻게 성능이 달라지는지**를 막대 그래프로 표현.  
- **F1-score, Precision-Recall Curve**를 비교하여, **MetaPhlAn 4가 종(Species) 수준에서 가장 높은 정확도를 보임**을 강조.  
- **Bray-Curtis Dissimilarity 분석**을 통해 서로 다른 모델들이 생성한 프로파일 간의 차이를 보여줌.  

#### ** Figure 3: 환경 데이터에서 군집 다양성 비교 (Community Structure in Environmental Data)**  
- **Alpha Diversity (Inverse Simpson Index)**, **Beta Diversity (Bray-Curtis Dissimilarity)**를 계산하여 시각화.  
- **Principal Coordinate Analysis (PCoA) 플롯**을 사용해 서로 다른 방법으로 분석된 샘플들의 군집 차이를 나타냄.  
- 특정 환경 요인(예: 해양 vs. 토양)의 미생물 군집 차이를 보여주는 그래프 포함.  

#### ** Figure 4: 환경 요인과 미생물 군집의 관계 (Environmental Factors & Microbial Community)**  
- **PERMANOVA 분석 결과**를 막대 그래프로 표현하여, 특정 환경 요인이 미생물 군집에 얼마나 영향을 미치는지 나타냄.  
- **MaAsLin 2 분석 결과**를 통해 특정 미생물이 특정 환경에서 얼마나 중요한지 상관 계수를 보여줌.  

#### ** Figure 5: 어셈블리 기반 방법의 성능 (Assembly-Based Methods Performance)**  
- **MEGAHIT, MetaBAT 기반 게놈 복원율과 정확도 비교**.  
- **샘플 당 복원된 MAG (Metagenome Assembled Genome) 개수**를 박스 플롯(box plot)으로 표현.  

---

### **어펜딕스(Appendix) 요약**  
논문의 보충 자료(어펜딕스)에는 추가적인 실험 결과와 상세 방법론이 포함됨.  

#### ** Appendix A: 추가 실험 및 데이터셋 설명**  
- **300개 합성 샘플 및 348개 실제 환경 샘플의 원본 데이터 상세 정보** 제공.  
- 각 환경(산림, 해양, 토양 등)에서 수집된 샘플의 메타데이터 설명.  

#### ** Appendix B: 평가 방법 추가 설명**  
- PERMANOVA 및 MaAsLin 2의 세부 통계 계산 과정.  
- **Bray-Curtis Dissimilarity 계산 방법 및 공식** 포함.  

#### ** Appendix C: 코드 및 재현성 (Reproducibility & Code)**  
- 논문의 분석을 **재현(reproducibility)**하기 위한 스크립트 및 GitHub 링크 포함.  
- 사용된 소프트웨어 및 버전 설명 (Kraken2, MetaPhlAn 4, MEGAHIT, CheckM2 등).  

---

---



### ** Figures Summary**  
The paper includes several key figures that visually represent the results.  

#### ** Figure 1: Experimental Overview Diagram**  
- **A schematic flowchart** showing dataset structure (synthetic vs. real-world), analysis methods (marker-based vs. assembly-based), and evaluation metrics.  

#### ** Figure 2: Classification Performance Comparison**  
- A **bar graph comparing F1-scores across Phylum to Species levels** for different methods.  
- **Precision-Recall Curve** analysis, showing **MetaPhlAn 4’s superior accuracy at the species level**.  
- **Bray-Curtis Dissimilarity analysis** illustrating differences between taxonomic profiles generated by different models.  

#### ** Figure 3: Community Diversity in Environmental Data**  
- **Alpha Diversity (Inverse Simpson Index)** and **Beta Diversity (Bray-Curtis Dissimilarity) visualized**.  
- **Principal Coordinate Analysis (PCoA) plot** showing how sample clustering differs based on the applied method.  
- Environmental microbiome differences (e.g., **ocean vs. soil**) are highlighted.  

#### ** Figure 4: Environmental Factors & Microbial Community**  
- **PERMANOVA results** presented in a bar chart, illustrating the impact of environmental factors on microbial composition.  
- **MaAsLin 2 results** showing correlation scores for key microbes in different environmental conditions.  

#### ** Figure 5: Performance of Assembly-Based Methods**  
- **Comparative accuracy and genome recovery rates for MEGAHIT & MetaBAT**.  
- **Box plots showing the number of recovered Metagenome Assembled Genomes (MAGs) per sample**.  

---

### ** Appendix Summary**  
The appendix contains additional **experimental details and supplementary results**.  

#### ** Appendix A: Additional Experiments & Dataset Details**  
- Detailed metadata for **300 synthetic samples and 348 real-world samples**.  
- Breakdown of sample distribution across different environments (e.g., **forest, ocean, soil**).  

#### ** Appendix B: Evaluation Method Details**  
- **PERMANOVA and MaAsLin 2 statistical analysis methodology**.  
- **Bray-Curtis Dissimilarity formula and calculation steps**.  

#### ** Appendix C: Code & Reproducibility**  
- **Scripts and GitHub repository links** for full reproducibility.  
- **Software and version details** (Kraken2, MetaPhlAn 4, MEGAHIT, CheckM2, etc.).  

---



<br/>
# refer format:     



@article{Nickols2024,
  author    = {William A. Nickols and Lauren J. McIver and Aaron Walsh and Yancong Zhang and Jacob T. Nearing and Francesco Asnicar and Michal Punčochář and Nicola Segata and Long H. Nguyen and Erica M. Hartmann and Eric A. Franzosa and Curtis Huttenhower and Kelsey N. Thompson},
  title     = {Evaluating metagenomic analyses for undercharacterized environments: what’s needed to light up the microbial dark matter?},
  journal   = {bioRxiv},
  year      = {2024},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {10.1101/2024.11.08.622677},
  url       = {https://doi.org/10.1101/2024.11.08.622677}
}



Nickols, William A., Lauren J. McIver, Aaron Walsh, Yancong Zhang, Jacob T. Nearing, Francesco Asnicar, Michal Punčochář, Nicola Segata, Long H. Nguyen, Erica M. Hartmann, Eric A. Franzosa, Curtis Huttenhower, and Kelsey N. Thompson. "Evaluating Metagenomic Analyses for Undercharacterized Environments: What’s Needed to Light up the Microbial Dark Matter?" bioRxiv (November 9, 2024). https://doi.org/10.1101/2024.11.08.622677.




