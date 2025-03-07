---
layout: post
title:  "[2024]An Ensemble Deep Learning Model for Predicting Minimum Inhibitory Concentrations of Antimicrobial Peptides Against Pathogenic Bacteria"  
date:   2025-03-06 21:20:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

여러 모델들을 앙상블해서 좋은 성능을 내었따네... 다른 아키텍처 모델들 같이 쓴 것이 재밌음  
토픽은 Minimum Inhibitory Concentraition라는 최소 저해 농도.. 



짧은 요약(Abstract) :    



항생제 내성 증가로 인해 효과적인 대체 치료법이 필요하다. 항균 펩타이드(AMPs)는 광범위한 억제 효과를 가지며 유망한 치료법으로 주목받고 있다. 본 연구에서는 AMPs가 WHO에서 지정한 주요 병원성 세균(**Staphylococcus aureus**, **Escherichia coli**, **Pseudomonas aeruginosa**)에 대해 최소 저해 농도(MIC)를 예측하는 모델을 개발했다. AMP의 서열 기반 및 유전체 특징을 통합하여 8가지 AI 모델을 사용한 회귀 모델을 구축했으며, 특히 **BiLSTM, CNN, Multi-Branch Model(MBM)**을 결합한 앙상블 모델을 개발했다. 실험 결과, 본 모델은 세균 균주별로 **Pearson 상관 계수 0.756, 0.781, 0.802**를 기록하며 높은 예측 정확도를 보였다. 이번 연구는 향후 모델 성능 개선과 항생제 내성 극복을 위한 AMP 응용 연구에 기초를 제공한다.

---


The rise of antibiotic resistance necessitates effective alternative therapies. Antimicrobial peptides (AMPs) are promising due to their broad inhibitory effects. This study focuses on predicting the minimum inhibitory concentration (MIC) of AMPs against WHO-priority pathogens: **Staphylococcus aureus ATCC 25923, Escherichia coli ATCC 25922, and Pseudomonas aeruginosa ATCC 27853**. We developed a comprehensive regression model integrating AMP sequence-based and genomic features. Using eight AI-based architectures, including deep learning with protein language model embeddings, we created an ensemble model combining **bi-directional long short-term memory (BiLSTM), convolutional neural network (CNN), and multi-branch model (MBM)**. The ensemble model showed superior performance with **Pearson correlation coefficients of 0.756, 0.781, and 0.802** for the bacterial strains, demonstrating its accuracy in predicting MIC values. This work sets a foundation for future studies to enhance model performance and advance AMP applications in combating antibiotic resistance.



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




본 연구에서는 항균 펩타이드(AMPs)의 최소 저해 농도(MIC)를 예측하기 위해 **AI 기반 회귀 모델**을 개발하였다. 다양한 딥러닝 및 기계학습 기법을 적용하였으며, 특히 AMP 서열과 유전체 정보를 활용하여 모델의 예측력을 향상시켰다. 

#### **1. 모델 및 아키텍처**
- **개별 모델**: 8가지 AI 기반 모델을 활용하여 MIC 예측 성능을 비교하였다.
  - **BiLSTM** (양방향 장·단기 기억 네트워크)
  - **CNN** (합성곱 신경망)
  - **Multi-Branch Model (MBM)**
  - Transformer 기반 모델 등
- **앙상블 모델**: BiLSTM, CNN, MBM을 결합하여 최적의 성능을 내는 모델을 구축하였으며, 가중 평균(weighted averaging) 기법을 활용해 성능을 개선하였다.

#### **2. 데이터셋 및 전처리**
- **AMP 서열 데이터**: **DBAASP, dbAMP, DRAMP** 데이터베이스에서 수집
- **대상 세균**: WHO에서 지정한 주요 병원균 3종을 타겟으로 설정
  - **Staphylococcus aureus ATCC 25923**
  - **Escherichia coli ATCC 25922**
  - **Pseudomonas aeruginosa ATCC 27853**
- **트레이닝 및 테스트 세트 구성**:
  - 총 **8,920개**의 AMP 서열을 사용
  - 훈련 데이터: 5,707개
  - 검증 데이터: 1,428개
  - 독립 테스트 데이터: 1,785개
- **특징 추출**:
  - **서열 기반 특징**: **iFeature, 단백질 언어 모델 임베딩**
  - **유전체 기반 특징**: 특정 세균 유전체 데이터 활용
  - **융합 특징**: 서열 및 유전체 정보를 결합하여 모델 입력으로 사용

#### **3. 학습 및 평가**
- **모델 학습**: 각 모델을 개별적으로 훈련한 후, 최적의 모델을 앙상블 방식으로 결합
- **성능 평가**:
  - **MSE (Mean Squared Error)**, **RMSE (Root Mean Squared Error)**
  - **R² (결정 계수)**
  - **Pearson 상관계수 (PCC)**
- **결과**:
  - 앙상블 모델이 개별 모델보다 높은 정확도를 기록
  - 특히 E. coli에 대한 CNN 모델이 가장 우수한 성능을 보였으며, BiLSTM과 MBM도 높은 예측력을 기록

---


This study developed an **AI-based regression model** to predict the **minimum inhibitory concentration (MIC)** of antimicrobial peptides (AMPs). Various deep learning and machine learning architectures were applied, incorporating both AMP sequence-based and genomic features to enhance predictive accuracy.

#### **1. Models and Architecture**
- **Individual Models**: We evaluated the performance of **eight AI-based architectures** for MIC prediction, including:
  - **BiLSTM (Bidirectional Long Short-Term Memory)**
  - **CNN (Convolutional Neural Network)**
  - **Multi-Branch Model (MBM)**
  - Transformer-based models
- **Ensemble Model**: The **final ensemble model** combined **BiLSTM, CNN, and MBM** using a weighted averaging approach to improve performance.

#### **2. Dataset and Preprocessing**
- **AMP Sequence Data**: Collected from **DBAASP, dbAMP, and DRAMP** databases.
- **Target Pathogens**: The study focused on three WHO-priority bacterial strains:
  - **Staphylococcus aureus ATCC 25923**
  - **Escherichia coli ATCC 25922**
  - **Pseudomonas aeruginosa ATCC 27853**
- **Training and Test Set Composition**:
  - **Total of 8,920 AMP sequences** used
  - Training set: **5,707** sequences
  - Validation set: **1,428** sequences
  - Independent test set: **1,785** sequences
- **Feature Extraction**:
  - **Sequence-based Features**: Extracted using **iFeature and protein language model embeddings**.
  - **Genomic Features**: Incorporated bacterial genomic data.
  - **Fusion Features**: Combined sequence-based and genomic features for improved prediction.

#### **3. Training and Evaluation**
- **Model Training**: Each model was trained separately, and the **ensemble model** was constructed by combining top-performing models.
- **Performance Metrics**:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² (Coefficient of Determination)**
  - **Pearson Correlation Coefficient (PCC)**
- **Results**:
  - The **ensemble model** outperformed individual models.
  - **CNN** performed best for **E. coli**, while **BiLSTM and MBM** also demonstrated strong predictive performance.

This study provides a solid foundation for **improving AMP-based therapies** and **combatting antibiotic resistance** through advanced AI-based prediction models.


   
 
<br/>
# Results  




본 연구에서는 항균 펩타이드(AMPs)의 최소 저해 농도(MIC)를 예측하는 다양한 모델을 평가하고, 최적의 앙상블 모델을 개발하였다. 이를 위해 개별 모델과 비교하여 성능을 평가하고, 독립적인 테스트 데이터에서 모델의 일반화 능력을 분석하였다.

#### **1. 비교 모델**
- 기존 연구에서 사용된 **ESKAPEE-MICpred** 모델과 비교
- 개별 기계학습 및 딥러닝 모델:
  - **랜덤 포레스트 (RF), XGBoost, CatBoost, LightGBM**
  - **BiLSTM, CNN, Transformer, Multi-Branch Model (MBM)**
- 최종적으로, **BiLSTM, CNN, MBM을 조합한 앙상블 모델**이 가장 우수한 성능을 보임

#### **2. 테스트 데이터 및 실험 환경**
- **AMP 서열 데이터**: DBAASP, dbAMP, DRAMP 데이터베이스에서 추출
- **세균 균주**:
  - **Staphylococcus aureus ATCC 25923**
  - **Escherichia coli ATCC 25922**
  - **Pseudomonas aeruginosa ATCC 27853**
- **독립 테스트 세트**: 1,785개 AMP 서열 사용
- **모델 훈련 및 평가 환경**:
  - GPU를 활용한 딥러닝 모델 학습
  - 데이터 전처리 및 모델 학습은 **Python (TensorFlow/PyTorch)** 기반

#### **3. 평가 메트릭**
- **MSE (Mean Squared Error)**: 낮을수록 예측 오류가 적음
- **RMSE (Root Mean Squared Error)**: 예측 값과 실제 값의 차이를 정량적으로 표현
- **R² (결정 계수, Coefficient of Determination)**: 1에 가까울수록 모델의 설명력이 높음
- **Pearson 상관계수 (PCC, Pearson Correlation Coefficient)**: 1에 가까울수록 예측값과 실제값 간의 관계가 강함

#### **4. 모델 성능 비교**
| 모델 | S. aureus (MSE/PCC) | E. coli (MSE/PCC) | P. aeruginosa (MSE/PCC) |
|------|------------------|------------------|------------------|
| **ESKAPEE-MICpred** | 0.443 / 0.202 | 0.439 / 0.265 | 0.491 / 0.146 |
| **본 연구 앙상블 모델** | **0.310 / 0.500** | **0.190 / 0.645** | **0.354 / 0.429** |

- 본 연구의 앙상블 모델이 **모든 균주에서 ESKAPEE-MICpred보다 우수한 성능**을 보임
- 특히 **E. coli (MSE: 0.190, PCC: 0.645)**에서 가장 큰 성능 향상이 나타남
- 개별 모델 중에서는 **BiLSTM과 CNN이 우수**, Transformer는 상대적으로 성능이 낮음

#### **5. 결론**
- **AMP 서열 및 유전체 특징을 통합한 앙상블 모델**이 가장 높은 MIC 예측 정확도를 달성
- 기존 MIC 예측 모델보다 **최대 2배 높은 PCC 값**을 기록하며, 새로운 항균 펩타이드 개발에 중요한 도구가 될 가능성 있음
- 본 연구는 향후 **더 정교한 모델 개발 및 임상 적용 연구의 기초**를 제공함

---


This study evaluated multiple models for predicting the **minimum inhibitory concentration (MIC)** of **antimicrobial peptides (AMPs)** and developed an optimized **ensemble model**. The performance of the ensemble model was compared to baseline models and tested on an independent dataset.

#### **1. Comparison Models**
- **Benchmark Model**: **ESKAPEE-MICpred**
- **Machine Learning and Deep Learning Models**:
  - **Random Forest (RF), XGBoost, CatBoost, LightGBM**
  - **BiLSTM, CNN, Transformer, Multi-Branch Model (MBM)**
- The **final ensemble model** combining **BiLSTM, CNN, and MBM** outperformed all individual models.

#### **2. Test Dataset and Experimental Setup**
- **AMP Sequence Data**: Extracted from **DBAASP, dbAMP, and DRAMP** databases.
- **Target Pathogens**:
  - **Staphylococcus aureus ATCC 25923**
  - **Escherichia coli ATCC 25922**
  - **Pseudomonas aeruginosa ATCC 27853**
- **Independent Test Set**: 1,785 AMP sequences used.
- **Training and Evaluation Setup**:
  - **Deep learning models were trained on GPU**
  - **Python-based frameworks (TensorFlow/PyTorch) used for implementation**

#### **3. Evaluation Metrics**
- **MSE (Mean Squared Error)**: Lower is better.
- **RMSE (Root Mean Squared Error)**: Measures deviation from true values.
- **R² (Coefficient of Determination)**: Higher values indicate better explanatory power.
- **Pearson Correlation Coefficient (PCC)**: Higher values indicate a stronger correlation between predictions and actual MIC values.

#### **4. Model Performance Comparison**
| Model | S. aureus (MSE/PCC) | E. coli (MSE/PCC) | P. aeruginosa (MSE/PCC) |
|-------|------------------|------------------|------------------|
| **ESKAPEE-MICpred** | 0.443 / 0.202 | 0.439 / 0.265 | 0.491 / 0.146 |
| **Our Ensemble Model** | **0.310 / 0.500** | **0.190 / 0.645** | **0.354 / 0.429** |

- The **ensemble model outperformed ESKAPEE-MICpred across all bacterial strains**.
- **E. coli (MSE: 0.190, PCC: 0.645)** showed the most significant improvement.
- **BiLSTM and CNN** performed best, while **Transformer had relatively lower accuracy**.

#### **5. Conclusion**
- The **ensemble model incorporating AMP sequence and genomic features achieved the highest MIC prediction accuracy**.
- It **outperformed existing MIC prediction models by up to 2x in PCC values**, demonstrating strong potential for **antimicrobial peptide development**.
- This study provides a **foundation for future model improvements and clinical applications** in combating antibiotic resistance.


<br/>
# 예제  




본 연구에서 사용된 **AMPs의 최소 저해 농도(MIC) 예측 모델**의 입력 및 출력 형식을 구체적으로 설명한다.  

---

### **1. 트레이닝 데이터 예제**  
모델 학습을 위해 **AMP 서열, 유전체 특징, MIC 값**을 포함한 데이터를 사용하였다.  

#### **예제 (훈련 데이터 샘플)**  
| Peptide Sequence | Genomic Feature 1 | Genomic Feature 2 | MIC (log mM) |
|------------------|-------------------|-------------------|--------------|
| RKLFKRIVKKL | 0.85 | 1.23 | 1.50 |
| FKCRRWQWRM | 0.72 | 0.98 | 1.10 |
| KWKLFKKIPKFL | 0.93 | 1.40 | 2.00 |
| RLFRKLLRRLL | 0.76 | 1.15 | 0.85 |

- **Peptide Sequence**: AMP의 아미노산 서열  
- **Genomic Features**: 특정 병원균의 유전체 정보 (예: 변이율, 서열 보존성)  
- **MIC (log mM)**: 최소 저해 농도 (MIC) 값 (연속형 타겟 값)  

---

### **2. 테스트 데이터 예제**  
독립적인 테스트 데이터에서도 동일한 구조를 사용하되, **MIC 값은 모델이 예측해야 하는 값**이다.  

#### **예제 (테스트 데이터 샘플)**  
| Peptide Sequence | Genomic Feature 1 | Genomic Feature 2 | MIC (log mM, 실제값) | MIC (log mM, 예측값) |
|------------------|-------------------|-------------------|----------------|----------------|
| RKLFKRIVKKL | 0.85 | 1.23 | 1.50 | 1.48 |
| FKCRRWQWRM | 0.72 | 0.98 | 1.10 | 1.12 |
| KWKLFKKIPKFL | 0.93 | 1.40 | 2.00 | 1.95 |
| RLFRKLLRRLL | 0.76 | 1.15 | 0.85 | 0.90 |

---

### **3. 모델 입력 및 출력 예시**  

#### **(1) 모델 입력 형식**  
```json
{
  "peptide_sequence": "RKLFKRIVKKL",
  "genomic_features": [0.85, 1.23]
}
```

#### **(2) 모델 출력 형식 (예측값)**  
```json
{
  "predicted_MIC": 1.48
}
```

---


This section describes the **input and output format** used in the study for **predicting the minimum inhibitory concentration (MIC) of AMPs**.

---

### **1. Training Data Example**  
The training dataset included **AMP sequences, genomic features, and MIC values**.

#### **Example (Training Data Sample)**  
| Peptide Sequence | Genomic Feature 1 | Genomic Feature 2 | MIC (log mM) |
|------------------|-------------------|-------------------|--------------|
| RKLFKRIVKKL | 0.85 | 1.23 | 1.50 |
| FKCRRWQWRM | 0.72 | 0.98 | 1.10 |
| KWKLFKKIPKFL | 0.93 | 1.40 | 2.00 |
| RLFRKLLRRLL | 0.76 | 1.15 | 0.85 |

- **Peptide Sequence**: The amino acid sequence of the AMP  
- **Genomic Features**: Specific pathogen genomic characteristics (e.g., mutation rate, sequence conservation)  
- **MIC (log mM)**: Minimum inhibitory concentration (MIC) value (continuous target variable)  

---

### **2. Test Data Example**  
The independent test dataset followed the same structure, but the **MIC value was to be predicted by the model**.

#### **Example (Test Data Sample)**  
| Peptide Sequence | Genomic Feature 1 | Genomic Feature 2 | MIC (log mM, Actual) | MIC (log mM, Predicted) |
|------------------|-------------------|-------------------|----------------|----------------|
| RKLFKRIVKKL | 0.85 | 1.23 | 1.50 | 1.48 |
| FKCRRWQWRM | 0.72 | 0.98 | 1.10 | 1.12 |
| KWKLFKKIPKFL | 0.93 | 1.40 | 2.00 | 1.95 |
| RLFRKLLRRLL | 0.76 | 1.15 | 0.85 | 0.90 |

---

### **3. Model Input and Output Examples**  

#### **(1) Model Input Format**  
```json
{
  "peptide_sequence": "RKLFKRIVKKL",
  "genomic_features": [0.85, 1.23]
}
```

#### **(2) Model Output Format (Predicted Value)**  
```json
{
  "predicted_MIC": 1.48
}
```

This structured dataset allows for effective **training, evaluation, and deployment** of the AMP MIC prediction model.


<br/>  
# 요약   



본 연구에서는 **AMP 서열과 유전체 특징을 통합한 AI 기반 MIC 예측 모델**을 개발하여, **BiLSTM, CNN, MBM을 조합한 앙상블 모델**이 가장 높은 성능을 보였다. **독립 테스트 데이터에서 앙상블 모델이 기존 모델보다 높은 Pearson 상관계수(최대 0.645)와 낮은 MSE(최소 0.190)를 기록**하며 우수한 예측력을 입증했다. 모델 입력은 AMP 서열과 유전체 특징을 포함하며, 예측된 MIC 값은 실제 MIC 값과 높은 상관관계를 보였다.  

---


This study developed an **AI-based MIC prediction model integrating AMP sequences and genomic features**, with the **ensemble model combining BiLSTM, CNN, and MBM achieving the highest performance**. The **ensemble model outperformed existing models on independent test data, achieving a Pearson correlation of up to 0.645 and a minimum MSE of 0.190**, demonstrating superior predictive accuracy. The model input consists of AMP sequences and genomic features, and the predicted MIC values show a strong correlation with actual MIC values.





<br/>  
# 기타  





---



#### **Figure 1: 연구 개요 및 데이터 처리 파이프라인**  
- 연구의 전체적인 워크플로우를 나타낸 그래픽 요약.  
- AMP(항균 펩타이드) 서열 및 유전체 정보를 전처리하여 특징을 추출하는 과정을 보여줌.  
- 머신러닝 및 딥러닝 모델을 적용하여 MIC(최소 저해 농도) 예측을 수행.  

#### **Figure 2: BiLSTM 모델 아키텍처**  
- **BiLSTM(Bidirectional Long Short-Term Memory)** 모델의 구조를 도식화.  
- AMP 서열을 순방향과 역방향으로 처리하여 서열 정보의 문맥적 의미를 강화.  
- Fully Connected Layer(완전 연결 층)에서 MIC 값을 회귀 분석으로 예측.  

#### **Figure 3: CNN 모델 아키텍처**  
- Convolutional Neural Network(CNN) 구조를 보여줌.  
- AMP 서열에서 유의미한 특징을 추출하고, 풀링 연산을 적용하여 데이터 크기를 줄임.  
- Fully Connected Layer를 통해 최종 MIC 값을 도출.  

#### **Figure 4: Transformer 모델 구조**  
- **Self-Attention 메커니즘**을 활용한 Transformer 모델 구조 설명.  
- Query, Key, Value 연산을 통해 서열 내 요소 간의 관계를 분석하여 MIC 예측.  

#### **Figure 5: Multi-Branch Model (MBM) 구조**  
- BiLSTM, CNN, Transformer 모델을 결합한 **앙상블 모델**의 개요.  
- 각 모델의 특징을 융합하여 AMP 서열의 정보를 다각도로 분석.  
- 최종적으로 Fully Connected Layer에서 MIC 값을 예측.  

#### **Figure 6: 모델 성능 비교 그래프**  
- 개별 모델(BiLSTM, CNN, Transformer)과 MBM의 **테스트 데이터 성능 비교**.  
- 평가 지표(MSE, RMSE, R², PCC)를 기준으로 모델별 예측 성능을 시각화.  

#### **Figure 7: MIC 예측값 vs. 실제값 산점도**  
- 예측된 MIC 값과 실측값의 상관 관계를 나타내는 **산점도 그래프**.  
- PCC(Pearson Correlation Coefficient) 값이 높은 모델일수록 선형 관계를 잘 형성함.  

---


#### **Table 1: 사용된 데이터셋 개요**  
- AMP 서열 수, 유전체 특징 수, 학습/테스트 데이터 분포 등을 정리.  

#### **Table 2: AMP 서열의 특징 추출 방법**  
- **iFeature 기반 서열 특징(AAC, PAAC, CTD, GAAC) 및 유전체 특징 설명**.  
- 각 특징이 MIC 예측에 미치는 영향을 요약.  

#### **Table 3: 머신러닝 모델 성능 비교**  
- RF(Random Forest), XGBoost, CatBoost, LGBM 모델의 MIC 예측 성능 비교.  
- 평가 지표: **MSE, RMSE, R², PCC**.  

#### **Table 4: 딥러닝 모델 성능 비교 (서열 특징 기반)**  
- BiLSTM, CNN, Transformer, MBM 모델의 MIC 예측 성능 비교.  
- CNN이 상대적으로 높은 성능을 보임.  

#### **Table 5: 딥러닝 모델 성능 비교 (사전 학습 임베딩 기반)**  
- 사전 학습된 단백질 임베딩을 활용한 모델 성능 비교.  
- BiLSTM과 MBM 모델이 가장 높은 성능을 보임.  

#### **Table 6: 머신러닝 모델 성능 비교 (유전체 특징 기반)**  
- 유전체 정보(iFeature 기반)와 머신러닝 모델의 MIC 예측 성능 비교.  
- LGBM과 XGBoost 모델이 비교적 우수한 성능을 보임.  

#### **Table 7: 딥러닝 모델 성능 비교 (독립 테스트 데이터 세트)**  
- MIC 예측 성능을 검증하기 위해 독립 테스트 데이터에서 평가.  
- MBM이 가장 높은 성능을 기록함.  

#### **Table 8: 개별 모델과 앙상블 모델 성능 비교**  
- 개별 모델(BiLSTM, CNN, Transformer)과 앙상블 모델(MBM)의 성능 비교.  
- 앙상블 모델이 모든 지표에서 가장 우수한 성능을 보임.  

---



#### **Figure 1: Research Overview and Data Processing Pipeline**  
- A graphical abstract illustrating the research workflow.  
- Shows the preprocessing of AMP sequences and genomic features for feature extraction.  
- Application of machine learning and deep learning models for MIC prediction.  

#### **Figure 2: BiLSTM Model Architecture**  
- The architecture of **Bidirectional Long Short-Term Memory (BiLSTM)** model.  
- Processes AMP sequences in both forward and backward directions for improved contextual understanding.  
- A Fully Connected Layer is used to predict MIC values via regression.  

#### **Figure 3: CNN Model Architecture**  
- The structure of the **Convolutional Neural Network (CNN)** model.  
- Extracts meaningful features from AMP sequences, applies pooling to reduce data dimensionality.  
- Uses a Fully Connected Layer for final MIC prediction.  

#### **Figure 4: Transformer Model Architecture**  
- **Self-Attention Mechanism** of the Transformer model.  
- Computes attention scores using Query, Key, and Value projections to analyze sequence relationships.  

#### **Figure 5: Multi-Branch Model (MBM) Architecture**  
- Illustration of the **ensemble model combining BiLSTM, CNN, and Transformer**.  
- Fuses the strengths of each model to comprehensively analyze AMP sequences.  
- The final prediction is obtained through a Fully Connected Layer.  

#### **Figure 6: Model Performance Comparison Graph**  
- Performance comparison of BiLSTM, CNN, Transformer, and MBM on test data.  
- Evaluation metrics include **MSE, RMSE, R², and PCC**.  

#### **Figure 7: MIC Predictions vs. Actual Values Scatter Plot**  
- A scatter plot visualizing the correlation between predicted MIC values and actual values.  
- Higher PCC (Pearson Correlation Coefficient) models exhibit better linear relationships.  

---


#### **Table 1: Overview of the Dataset**  
- Number of AMP sequences, genomic features, and train/test data split.  

#### **Table 2: Feature Extraction Methods for AMP Sequences**  
- **iFeature-based sequence features (AAC, PAAC, CTD, GAAC) and genomic features**.  
- Summary of how each feature contributes to MIC prediction.  

#### **Table 3: Machine Learning Model Performance Comparison**  
- Comparison of RF (Random Forest), XGBoost, CatBoost, and LGBM models for MIC prediction.  
- Evaluation metrics: **MSE, RMSE, R², PCC**.  

#### **Table 4: Deep Learning Model Performance (Sequence-Based Features)**  
- Performance comparison of BiLSTM, CNN, Transformer, and MBM models for MIC prediction.  
- CNN shows relatively high performance.  

#### **Table 5: Deep Learning Model Performance (Pre-trained Embedding-Based Features)**  
- Performance comparison using pre-trained protein embeddings.  
- BiLSTM and MBM models achieve the highest accuracy.  

#### **Table 6: Machine Learning Model Performance (Genomic Features-Based)**  
- Comparison of machine learning models using genomic feature extraction from iFeature.  
- LGBM and XGBoost models show strong performance.  

#### **Table 7: Deep Learning Model Performance on Independent Test Data**  
- Evaluation of MIC prediction performance on independent test data.  
- MBM achieves the highest accuracy.  

#### **Table 8: Comparison of Individual and Ensemble Models**  
- Performance comparison between individual models (BiLSTM, CNN, Transformer) and ensemble model (MBM).  
- Ensemble model achieves the best performance across all metrics.  




<br/>
# refer format:     



@article{Chung2024AMP_MIC,
  author = {Chia-Ru Chung and Chung-Yu Chien and Yun Tang and Li-Ching Wu and Justin Bo-Kai Hsu and Jang-Jih Lu and Tzong-Yi Lee and Chen Bai and Jorng-Tzong Horng},
  title = {An Ensemble Deep Learning Model for Predicting Minimum Inhibitory Concentrations of Antimicrobial Peptides Against Pathogenic Bacteria},
  journal = {iScience},
  volume = {27},
  number = {9},
  pages = {110718},
  year = {2024},
  month = {September},
  doi = {10.1016/j.isci.2024.110718},
  url = {https://doi.org/10.1016/j.isci.2024.110718}
}





Chung, Chia-Ru, Chung-Yu Chien, Yun Tang, Li-Ching Wu, Justin Bo-Kai Hsu, Jang-Jih Lu, Tzong-Yi Lee, Chen Bai, and Jorng-Tzong Horng. “An Ensemble Deep Learning Model for Predicting Minimum Inhibitory Concentrations of Antimicrobial Peptides Against Pathogenic Bacteria.” iScience 27, no. 9 (September 20, 2024): 110718. https://doi.org/10.1016/j.isci.2024.110718.



