---
layout: post
title:  "[2020]Metagenomic Analysis of Colorectal Cancer Datasets Identifies Cross-Cohort Microbial Diagnostic Signatures and a Link with Choline Degradation"  
date:   2025-04-15 05:23:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

여러 출처 데이터들 모음, 그리고 Kraken2 알고리즘을 사용해 각 시퀀스를 분류, 각 미생물 분류 수준(속, 종, 과 등)을 특징(feature)으로 사용하여 예측(건강한 대조군 (Normal),대장 선종 (CRA),대장암 (CRC))   


짧은 요약(Abstract) :    

이 논문은 대장암(colorectal carcinoma, CRC)과 그 주변 조직(adjecent tissue)의 미생물 군집(microbiota) 변화를 대규모 16S rRNA 데이터를 메타분석하여 연구한 것입니다. 연구팀은 CRC, 대장 선종(colorectal adenoma, CRA), 그리고 건강한 대조군 간의 미생물 차이를 머신러닝 기반 방법으로 분석했습니다. 총 15개의 연구로부터 2,099개의 샘플을 수집하여 다양한 분석(차등 풍부도 분석, 랜덤 포레스트 분류, 네트워크 분석 등)을 수행했고, CRC 인접 조직의 미생물 불균형이 뚜렷하며 예측에 유용하다는 것을 발견했습니다. 특히 CRC 인접 조직의 마이크로바이옴은 질병 유무를 정확히 분류하는 데 있어 대변보다 더 효과적일 수 있음이 밝혀졌습니다. 이 결과는 대장암 발병 과정에서 미생물의 역할을 이해하고 진단에도 응용될 수 있는 가능성을 시사합니다.

This study performs a meta-analysis of 16S rRNA microbial datasets to explore microbiota differences among colorectal carcinoma (CRC), colorectal adenoma (CRA), and healthy controls. Using 2,099 samples from 15 studies, the authors applied machine learning methods—such as random forest classifiers, differential abundance analysis, and network analysis—to identify microbial dysbiosis in both tumor and adjacent tissues. Notably, they found that the microbiota of adjacent tissues (off-site) showed distinct and predictive patterns, with high AUC scores (up to 95.8%) for CRC detection. The microbiome profiles of adjacent tissues closely resembled those of tumor tissues, suggesting their diagnostic value and supporting the notion that these microbial changes may play a role in tumorigenesis.   





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



**1. 데이터 수집 및 전처리**  
총 15개의 공개된 연구에서 2,099개의 16S rRNA 시퀀싱 데이터를 수집했어요.  
- 샘플은 다음 세 가지 그룹으로 나뉘었어요:  
  - 건강한 대조군 (Normal)  
  - 대장 선종 (CRA)  
  - 대장암 (CRC)  
그리고 샘플은 대변(stool), 병변 조직(on-site tissue), 병변 인접 조직(adjacent/off-site tissue)으로 구분되었어요.  

**2. 특성 추출 및 정규화**  
- Kraken2 알고리즘을 사용해 각 시퀀스를 분류하고, 각 미생물 분류 수준(속, 종, 과 등)을 특징(feature)으로 사용했어요.  
- 각 feature의 상대적 풍부도(relative abundance)를 계산했고, 10% 이상 샘플에서 나타나는 feature만 분석에 사용했어요.  

**3. 머신러닝 모델 - Random Forest (RF)**  
- 주요 분류 모델은 **Random Forest**이며, Qiime과 R의 `randomForest` 패키지를 사용했어요.  
- 각 전략별로 모델을 훈련하고 10-fold cross-validation으로 평가했어요.  
- AUC(Area Under Curve)를 기준으로 가장 중요한 feature들을 선택했어요.  

**4. 모델 평가 방식**  
- 전체 데이터를 통합하여 학습한 **Pooling RF 모델**,  
- 각 코호트 간 학습/테스트를 수행한 **Cohort-to-Cohort 모델 (C2C RF)**,  
- 한 코호트를 빼고 학습해 남은 코호트로 테스트하는 **Leave-One-Cohort-Out (LOCO) RF 모델**  
이 세 가지 방식으로 일반화 성능을 평가했어요.  

---


**1. Data Collection and Preprocessing**  
The study used **2,099 samples** from **15 publicly available 16S rRNA sequencing datasets**, covering three groups:  
- Healthy controls  
- Colorectal adenoma (CRA)  
- Colorectal carcinoma (CRC)  
Samples were classified into **stool**, **on-site lesion tissue**, and **adjacent/off-site tissue** groups.

**2. Feature Extraction and Normalization**  
- Taxonomic classification was performed using the **Kraken2** algorithm.  
- Each taxonomic level (e.g., genus, species) was treated as a feature.  
- Only features present in **at least 10% of samples** were retained.  
- Relative abundance of features was computed per sample.

**3. Machine Learning Model – Random Forest (RF)**  
- The main classifier used was **Random Forest**, implemented via **Qiime** and the R package `randomForest`.  
- Models were trained for each strategy and evaluated using **10-fold cross-validation**.  
- Feature importance was determined by AUC maximization or plateauing during iterative feature addition.

**4. Evaluation Strategies**  
Three model types were applied to assess generalizability:  
- **Pooling RF model** (combined samples from all cohorts)  
- **Cohort-to-Cohort (C2C) RF model** (train on one cohort, test on others)  
- **Leave-One-Cohort-Out (LOCO) RF model** (train on all but one cohort, test on the excluded one)

---



   
 
<br/>
# Results  




1. **인접 조직의 마이크로바이옴도 예측에 매우 효과적이다.**  
   - CRC 인접 조직 vs 정상 조직 AUC: **95.8%**  
   - CRA 인접 조직 vs 정상 조직 AUC: **89.5%**  
   → 종양 부위가 아닌 인접 조직만으로도 질병을 높은 정확도로 구분 가능

2. **대변보다 조직 기반 마이크로바이옴이 더 높은 분류 성능을 보인다.**  
   - CRC 대변 vs 정상 대변 AUC: **80.7%**  
   - CRC 조직 vs 정상 조직 AUC: **96.0%**

3. **CRA(선종)과 CRC(암) 간 마이크로바이옴 구성 변화는 점진적이다.**  
   - Fusobacterium, Parvimonas 등은 암에 가까울수록 점점 증가  
   - Ruminococcus, Dorea, Blautia 등은 건강할수록 높고 점차 감소

4. **조직과 인접 조직의 마이크로바이옴 네트워크는 매우 유사하다.**  
   → 인접 조직은 미생물적으로도 “정상”이라고 보기 어렵다

5. **코호트 간 일반화는 어려우나, 일부 공통 특징은 반복적으로 관측됨.**  
   → 메타커뮤니티 분석에서 지리적/기술적 차이가 주요 변수로 작용

---



1. **Adjacent tissues have strong predictive microbiome signals.**  
   - AUC for CRC adjacent tissue vs normal: **95.8%**  
   - AUC for CRA adjacent tissue vs normal: **89.5%**  
   → Even off-site tissue microbiota can effectively distinguish disease.

2. **Tissue-based microbiota outperforms stool-based models.**  
   - CRC stool vs normal stool AUC: **80.7%**  
   - CRC tissue vs normal tissue AUC: **96.0%**

3. **CRA and CRC show gradual microbiome shifts.**  
   - Fusobacterium, Parvimonas: increase along disease progression  
   - Ruminococcus, Dorea, Blautia: decline from healthy to cancer

4. **Adjacent tissues closely resemble on-site tumor tissues in microbiome networks.**  
   → Adjacent tissues are **not microbiologically “normal”**

5. **Cross-cohort generalization is limited, but some features are reproducible.**  
   → Metacommunity analysis reveals cohort- and region-specific patterns




<br/>
# 예제  




### 1. **데이터 유형 및 구조**
논문에서 사용한 데이터는 크게 두 부분으로 구성돼요:
- **(A) 메타데이터 테이블**: 샘플의 속성 정보
- **(B) 시퀀스 기반 taxonomic feature 테이블**: 각 샘플의 미생물 풍부도

---

###  2. **예제 A: 메타데이터 (metadata) 구조 예시**

| sample_id    | group              | cohort      | sample_type     | platform     | region   |
|--------------|-------------------|-------------|------------------|--------------|----------|
| CRC_T_001    | CRC_Tissue         | China_GBA   | tissue           | 454          | V1–V4    |
| CRC_TA_002   | CRC_Tissue_Adjacent| China_GBA   | adjacent_tissue  | 454          | V1–V4    |
| Normal_T_003 | Normal_Tissue      | Zeller      | tissue           | Illumina     | V4       |
| CRA_S_004    | CRA_Stool          | Flemer      | stool            | Illumina     | V3–V4    |

- `group`: 샘플 분류 (정상/암/선종/인접조직)
- `cohort`: 샘플이 나온 연구 데이터셋 이름
- `sample_type`: 조직인지 대변인지 등
- `platform`: 시퀀싱 장비 종류
- `region`: 16S rRNA의 어느 하이퍼변이 영역을 사용했는지

---

###  3. **예제 B: 시퀀스 기반 taxonomic feature 예시**

| sample_id   | Fusobacterium | Ruminococcus | Blautia | Parvimonas | ... | Total_Reads |
|-------------|---------------|--------------|---------|------------|-----|-------------|
| CRC_T_001   | 0.231         | 0.003        | 0.012   | 0.154      | ... | 20432       |
| CRC_TA_002  | 0.195         | 0.006        | 0.021   | 0.142      | ... | 19875       |
| Normal_T_003| 0.001         | 0.120        | 0.089   | 0.002      | ... | 18291       |
| CRA_S_004   | 0.034         | 0.034        | 0.043   | 0.007      | ... | 21356       |

- 각 열은 특정 균 속의 **상대적 풍부도 (0~1)**  
- `Total_Reads`는 정규화 전 read 개수  
- Kraken2로 분류 후 생성됨

---

###  4. **모델에 실제 사용된 feature**

- 전체 수천 개 feature 중 **10% 이상 샘플에서 나타나는 taxon만 유지** → 약 657개
- 이 중 **Random Forest 중요도 기준 상위 수십 개만 사용**
- 예:  
  - **Fusobacterium, Parvimonas**: 암에서 증가  
  - **Ruminococcus, Blautia, Dorea**: 정상에서 풍부

---

###  5. **예제 분석 시나리오 (전략 S8)**

- 분류 문제: **CRC 인접 조직 vs 정상 조직**
- 샘플 수: 약 422 vs 175
- 입력 데이터:  
  - 메타정보: cohort, platform, region  
  - taxon feature: 위에서 추린 중요 균속들의 풍부도
- 결과:  
  - AUC: **95.8%**  
  - 주요 feature: Fusobacterium, Parvimonas ↑ / Ruminococcus, Dorea ↓



###  1. **Data Structure**
Two main types of data were used:
- (A) **Metadata table**: Sample-level attributes
- (B) **Taxonomic abundance table**: Microbial features (Kraken2-derived)

---

###  2. **Example A: Metadata format**

| sample_id    | group              | cohort      | sample_type     | platform     | region   |
|--------------|-------------------|-------------|------------------|--------------|----------|
| CRC_T_001    | CRC_Tissue         | China_GBA   | tissue           | 454          | V1–V4    |
| CRC_TA_002   | CRC_Tissue_Adjacent| China_GBA   | adjacent_tissue  | 454          | V1–V4    |
| Normal_T_003 | Normal_Tissue      | Zeller      | tissue           | Illumina     | V4       |
| CRA_S_004    | CRA_Stool          | Flemer      | stool            | Illumina     | V3–V4    |

---

###  3. **Example B: Microbiome features**

| sample_id   | Fusobacterium | Ruminococcus | Blautia | Parvimonas | ... | Total_Reads |
|-------------|---------------|--------------|---------|------------|-----|-------------|
| CRC_T_001   | 0.231         | 0.003        | 0.012   | 0.154      | ... | 20432       |
| Normal_T_003| 0.001         | 0.120        | 0.089   | 0.002      | ... | 18291       |

- Features are **relative abundances** from Kraken2 taxonomic profiling
- Only taxa present in >10% of samples were retained (657 features)
- Only **top-ranked features (e.g., 50–70)** were used per model

---

###  4. **Real scenario: Strategy S8**

- Task: **CRC adjacent tissue vs normal tissue classification**
- Data:  
  - Sample metadata: cohort, platform, region  
  - Microbial features: top discriminative taxa  
- Performance:  
  - AUC = **95.8%**  
  - Important taxa:  
    - ↑ Fusobacterium, Parvimonas (cancer)  
    - ↓ Ruminococcus, Dorea (healthy)




<br/>  
# 요약   


이 논문은 15개 코호트에서 수집한 2,099개의 16S rRNA 데이터를 Kraken2로 분석하고, Random Forest 모델을 통해 정상, 선종, 대장암 간의 미생물 차이를 다양한 이진 분류 전략으로 평가하였다.
그 결과, 대장암 인접 조직의 마이크로바이옴만으로도 정상 조직과 95.8%의 AUC로 구분 가능하며, 대변보다 조직 기반 정보가 높은 분류 성능을 보였다.
예를 들어 CRC_TA_002 샘플은 Fusobacterium이 높고 Ruminococcus가 낮은 특성을 보이며 CRC 인접조직으로 분류되었다.   



This study analyzed 2,099 16S rRNA samples from 15 cohorts using Kraken2 and evaluated microbial differences between normal, adenoma, and colorectal cancer groups via Random Forest-based binary classification.
The results showed that adjacent tissue microbiomes alone could distinguish CRC from healthy controls with an AUC of 95.8%, outperforming stool-based models.
For instance, sample CRC_TA_002 showed high Fusobacterium and low Ruminococcus abundance, characteristic of CRC-adjacent tissue.    




<br/>  
# 기타  



###  1. **Figure (그림)**  
- **Figure 1**: Principal Coordinate Analysis (PCoA)를 통해 각 샘플 그룹 간의 군집화(β-diversity)를 시각화  
- **Figure 2**: 8개 전략에서 차등 풍부도 분석 결과 (enriched/depleted taxa)와 겹치는 feature를 Venn diagram으로 표현  
- **Figure 3**: Random Forest 모델 성능(AUC), 중요 feature들의 heatmap  
- **Figure 4**: CRA→CRC로의 진행에 따라 주요 균속들의 상대적 변화(시간 흐름에 따른 boxplot)  
- **Figure 5**: microbial network 상호작용을 SparCC로 분석한 correlation matrix, Mantel test 결과 포함  
- **Figure 6–7**: DMM 기반 메타커뮤니티 군집화 결과, cohort-to-cohort 및 LOCO 모델 평가 결과 heatmap  

###  2. **Table (표)**  
- **Table 1**: 15개 코호트별 샘플 수, 시퀀싱 플랫폼, 사용된 16S 영역 정리  
- **Supplemental Table S2–S4 (보조자료)**:  
  - 차등 풍부도 분석에 의해 식별된 균속 목록  
  - 각 전략에서 선택된 top feature 목록  
  - 반복적으로 검출된 reproducible taxa 리스트 등 제공  

### 3. **Appendix / Supplement**  
- **Figure S1–S6**: 추가적인 군집화 시각화, feature 중요도 그래프, validation AUC 분포 등  
- DESeq2나 Random Forest 관련 **상세 파라미터 및 필터링 기준** 포함  
- 데이터 전처리 파이프라인 (Trimmomatic, Kraken2 적용 방식) 정리

---



###  1. **Figures**  
- **Figure 1**: PCoA plots show microbial beta-diversity differences across groups  
- **Figure 2**: Differential abundance results with Venn diagrams across 8 strategies  
- **Figure 3**: AUC results from Random Forest models, feature importance heatmaps  
- **Figure 4**: Temporal progression of microbial shifts (e.g., Fusobacterium increasing, Ruminococcus decreasing)  
- **Figure 5**: SparCC-based microbial co-occurrence networks and Mantel correlation  
- **Figure 6–7**: DMM clustering of metacommunities and C2C/LOCO performance heatmaps

###  2. **Tables**  
- **Table 1**: Sample distribution per cohort, platform used, and 16S region  
- **Supplementary Tables (S2–S4)**:  
  - Differentially abundant taxa  
  - Top selected features per classification strategy  
  - Reproducible microbial features shared across cohorts

###  3. **Appendix & Supplemental Files**  
- **Figure S1–S6**: Extended clustering results, model performance visualizations, feature selection curves  
- DESeq2 parameters and Kraken2 classification rules  
- Processing pipeline details: Trimmomatic filtering, sample exclusion criteria, etc.




<br/>
# refer format:     


@article{Thomas2020,
  author    = {Andrew M. Thomas and Edoardo Manghi and Federica Asnicar and Paolo Pasolli and Moreno Armanini and Francesco Zolfo and Francesco Beghini and Adrian Manara and Claudia Karcher and Vincenzo Pozzi and Alberto Gandini and Paolo Serrano and Francesco Bolzan and Stefano Margiotta and Jacopo Longo and Guido Nicolini and Alessio Canto and Silvia L. Bertasi and Giorgio Francavilla and Nicola Cellini and Giorgio Masetti and Piergiuseppe De Bellis and Alessandra Briukhovetskaia and Erica Cereser and Paolo Gasperini and Claudio Bassi and Aldo Scarpa and Francesco Turroni and Marco Ventura and Stefano Bicciato and Riccardo Valdagni and Saverio Bettuzzi and Renato Ugo and Nico Curti and Francesco Tesei and Silvia Franceschi and Alessandra Renzi and Matteo Cescon and Marco Lucarini and Maurizio Brigotti and Silvia Di Bella and Jacopo Sanchini and Federica Facciotti and Nicola Segata},
  title     = {Metagenomic Analysis of Colorectal Cancer Datasets Identifies Cross-Cohort Microbial Diagnostic Signatures and a Link with Choline Degradation},
  journal   = {mSystems},
  volume    = {5},
  number    = {1},
  year      = {2020},
  pages     = {e0138-20},
  doi       = {10.1128/mSystems.00138-20},
  publisher = {American Society for Microbiology},
  url       = {https://journals.asm.org/doi/10.1128/mSystems.00138-20}
}





Thomas, Andrew M., Edoardo Manghi, Federica Asnicar, Paolo Pasolli, Moreno Armanini, Francesco Zolfo, Francesco Beghini, et al. 2020. “Metagenomic Analysis of Colorectal Cancer Datasets Identifies Cross-Cohort Microbial Diagnostic Signatures and a Link with Choline Degradation.” mSystems 5 (1): e0138-20. https://doi.org/10.1128/mSystems.00138-20.   





