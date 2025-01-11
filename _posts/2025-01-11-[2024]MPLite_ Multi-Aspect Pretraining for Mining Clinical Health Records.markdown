---
layout: post
title:  "[2024]MPLite: Multi-Aspect Pretraining for Mining Clinical Health Records"  
date:   2025-01-11 16:03:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    



MPLite는 전자 건강 기록(EHR)의 단일 방문 데이터를 활용하여 환자 건강 결과를 예측할 수 있는 가벼운 신경망 기반의 다중 측면 사전 학습 프레임워크입니다. 이 연구는 MIMIC-III 및 MIMIC-IV 데이터셋을 사용하여 심부전 예측 및 진단 예측 과제를 개선하는 데 있어 이 접근 방식이 기존 모델보다 높은 성능을 보인다는 것을 입증했습니다. 특히 실험 결과, MPLite는 구조화된 의료 데이터와 실험실 결과 데이터를 결합하여 단일 방문 데이터에서도 효과적인 의료 개념 표현 학습을 가능하게 했으며, 다중 방문 데이터와 통합할 경우 모델의 예측 정확도가 크게 향상되었습니다.

---


MPLite is a lightweight neural network-based framework designed for multi-aspect pretraining to predict patient health outcomes using single-visit electronic health records (EHR). This study demonstrates its efficacy in improving diagnostic and heart failure prediction tasks using the MIMIC-III and MIMIC-IV datasets. The approach effectively leverages structured medical data and lab results, enabling robust medical concept representation. Experimental results highlight significant performance improvements over existing models, showcasing MPLite's potential to enhance predictive modeling in healthcare by integrating diverse data aspects.



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




MPLite는 전자 건강 기록(EHR) 데이터에서 단일 방문 및 다중 방문 환자 데이터를 모두 활용하여 예측 모델의 성능을 높이는 다중 측면 사전 학습 프레임워크입니다. 주요 특징 및 방법론은 다음과 같습니다:

1. **모델 구조**:
   - MPLite는 가벼운 신경망(MLP, 다층 퍼셉트론)을 핵심 모델로 사용합니다. 이 모델은 실험실 결과를 기반으로 의료 진단 코드를 예측하는 사전 학습 모듈을 제공합니다.
   - **사전 학습 모듈**은 실험실 결과를 입력으로 받아 숨겨진 표현(hidden representation)을 생성하고, 이를 기반으로 진단 코드를 예측하는 다중 레이블 분류 작업을 수행합니다.
   - MLP를 사용하는 이유는 다음과 같습니다:
     - 계산 자원이 적게 소요됩니다.
     - 간단한 구조로도 경쟁력 있는 예측 성능을 제공합니다.

2. **통합 및 학습 방법**:
   - 사전 학습된 모듈은 "플러그 앤 플레이" 방식으로 기존 모델과 통합됩니다. 이로 인해 기존 예측 모델의 출력과 실험실 결과를 기반으로 한 숨겨진 표현을 결합하여 최종 예측을 수행합니다.
   - 이 접근법은 기존 예측 모델과의 결합 시 더 높은 정확도와 재현율을 제공하며, 특히 단일 방문 데이터에서도 효율적으로 작동합니다.

3. **사용 데이터셋**:
   - **MIMIC-III**와 **MIMIC-IV** 데이터셋을 사용했습니다. 두 데이터셋은 집중 치료실(ICU)에서 수집된 방대한 환자 기록을 포함하며, 각각의 환자 기록에는 진단 코드, 실험실 결과, 약물 정보 등이 포함되어 있습니다.
   - MIMIC-III 데이터셋에서 총 46,520명의 환자가 포함되었으며, 이 중 83.8%는 단일 방문 환자 데이터로 구성되어 있습니다. 단일 방문 데이터는 기존 연구에서 주로 제외되었지만, MPLite는 이를 추가 학습 데이터로 활용해 성능을 개선했습니다.

4. **특이점**:
   - 기존 Transformer 기반 모델이나 그래프 기반 모델과 달리, MPLite는 가벼운 구조와 높은 확장성을 통해 다양한 예측 작업에 쉽게 통합될 수 있습니다.
   - 실험실 결과를 의료 개념의 중요한 요소로 간주하여, 이를 사전 학습 단계에 통합함으로써 모델이 의료 데이터의 복잡한 관계를 더 잘 학습할 수 있도록 했습니다.

---



MPLite is a multi-aspect pretraining framework designed to leverage both single-visit and multi-visit patient data from electronic health records (EHR) to improve prediction models. The key features and methodology are as follows:

1. **Model Structure**:
   - MPLite employs a lightweight neural network (MLP, Multi-Layer Perceptron) as the core model. The pretraining module predicts medical diagnosis codes based on lab results.
   - **Pretraining Module**:
     - Takes lab results as input to generate hidden representations.
     - Performs a multi-label classification task to predict diagnosis codes.
   - Reasons for using MLP:
     - Requires minimal computational resources.
     - Achieves competitive performance with a simple architecture.

2. **Integration and Training**:
   - The pretrained module is integrated into existing models in a "plug-and-play" manner. This integration combines the predictions of the existing model with hidden representations derived from lab results to make final predictions.
   - This approach enhances accuracy and recall when combined with existing models, especially for single-visit data.

3. **Datasets Used**:
   - **MIMIC-III** and **MIMIC-IV** datasets were utilized. These datasets contain extensive de-identified clinical data collected from ICU patients, including diagnosis codes, lab results, and medication information.
   - The MIMIC-III dataset includes 46,520 patients, 83.8% of whom have single-visit data. While single-visit data are often excluded in prior studies, MPLite effectively utilizes these data to improve performance.

4. **Distinctive Features**:
   - Unlike Transformer or graph-based models, MPLite's lightweight structure and high scalability make it easy to integrate with various prediction tasks.
   - By treating lab results as crucial medical concepts, MPLite incorporates them into the pretraining phase, enabling the model to better capture the complex relationships in medical data.



   
 
<br/>
# Results  




MPLite는 MIMIC-III와 MIMIC-IV 데이터셋을 기준으로 다양한 예측 과제를 수행했으며, 기존 모델 대비 유의미한 성능 향상을 보여주었습니다. 주요 결과는 다음과 같습니다:

1. **평가 메트릭**:
   - 진단 예측(Diagnosis Prediction)에서는 **가중 F1 점수(weighted F1)**와 **상위 k개 재현율(Recall@k)**를 사용했습니다. 여기서 \(k=10\)과 \(k=20\)으로 설정되었습니다.
   - 심부전 예측(Heart Failure Prediction)에서는 **AUC(ROC 곡선 아래 면적)**와 **F1 점수**를 사용했습니다.

2. **MIMIC-III 데이터셋 결과**:
   - MPLite를 통합한 GRU 모델은 기존 GRU 모델 대비 가중 F1 점수가 **1.76%** 상승했습니다(17.82 → 19.58).
   - 같은 모델에서 Recall@10과 Recall@20은 각각 **2.26%**와 **2.33%** 증가했습니다.
   - Dipole 모델에서도 가중 F1 점수가 **3.61%** 상승했고(14.66 → 18.27), Recall@10과 Recall@20에서 각각 **2.18%**와 **3.53%**의 향상이 있었습니다.

3. **MIMIC-IV 데이터셋 결과**:
   - MPLite를 사용한 GRU 모델은 기존 대비 가중 F1 점수가 **1.32%** 상승했으며(19.55 → 21.87), Recall@10과 Recall@20에서는 각각 **2.72%** 증가했습니다.
   - Dipole 모델에서는 가중 F1 점수가 **3.47%** 상승했고, Recall@10과 Recall@20에서도 각각 **5.91%**와 **2.01%**의 향상이 관찰되었습니다.

4. **심부전 예측 결과**:
   - GRU와 Dipole 모델 모두 AUC와 F1 점수에서 성능이 증가했습니다.
   - 예를 들어, GRU 모델의 AUC는 MIMIC-III 데이터셋에서 **80.54% → 82.01%**로, F1 점수는 **68.93% → 70.56%**로 상승했습니다.

5. **주요 향상 요인**:
   - 실험실 결과 데이터를 추가로 통합함으로써 환자의 생리적, 생화학적 상태를 더 잘 반영하여 모델의 민감도와 정확도를 높였습니다.
   - 단일 방문 데이터를 효과적으로 활용함으로써 기존 모델의 제약을 극복했습니다.

---



MPLite demonstrated significant improvements over baseline models for various prediction tasks using the MIMIC-III and MIMIC-IV datasets. The key results are as follows:

1. **Evaluation Metrics**:
   - For diagnosis prediction, the **weighted F1 score** and **Recall@k** (with \(k=10\) and \(k=20\)) were used.
   - For heart failure prediction, **AUC (Area Under the ROC Curve)** and **F1 score** were employed.

2. **MIMIC-III Dataset Results**:
   - The GRU model with MPLite integration showed a **1.76%** increase in weighted F1 score (from 17.82 to 19.58).
   - Similarly, Recall@10 and Recall@20 improved by **2.26%** and **2.33%**, respectively.
   - The Dipole model achieved a **3.61%** increase in weighted F1 score (from 14.66 to 18.27) and improvements of **2.18%** in Recall@10 and **3.53%** in Recall@20.

3. **MIMIC-IV Dataset Results**:
   - The GRU model with MPLite showed a **1.32%** increase in weighted F1 score (from 19.55 to 21.87) and a **2.72%** improvement in both Recall@10 and Recall@20.
   - The Dipole model achieved a **3.47%** increase in weighted F1 score and notable gains of **5.91%** in Recall@10 and **2.01%** in Recall@20.

4. **Heart Failure Prediction Results**:
   - Both GRU and Dipole models exhibited improved AUC and F1 scores.
   - For example, the GRU model's AUC increased from **80.54% to 82.01%**, and the F1 score rose from **68.93% to 70.56%** on the MIMIC-III dataset.

5. **Key Factors for Improvement**:
   - The integration of lab results provided detailed physiological and biochemical indicators, enhancing the model's sensitivity and accuracy.
   - Effective utilization of single-visit data helped overcome the limitations of existing models.



<br/>
# 예제  


논문에 구체적인 테스트 데이터 샘플이나 비교된 결과에 대한 사례는 명시되어 있지 않습니다. 하지만, 논문에서 설명된 주요 결과를 기반으로 가상의 예시를 생성하여 설명할 수 있습니다.

---



#### 데이터 샘플:
- **환자 ID**: 12345
- **입원 기록**:
  - **진단 코드**: 고혈압(ICD-9: 401), 당뇨병(ICD-9: 250)
  - **실험실 결과**: 
    - 혈당: 180 mg/dL (정상 범위 초과)
    - 크레아티닌: 1.8 mg/dL (정상 범위 초과, 신장 기능 저하)
    - 헤모글로빈: 10 g/dL (정상 범위 이하, 빈혈 의심)
- **예측 목표**: 다음 방문에서 심부전(Heart Failure, ICD-9: 428) 진단 여부

#### 결과 비교:
- **기존 모델(GRU)**:
  - 예측: 심부전 없음(False Negative)
  - 이유: 실험실 결과를 포함하지 않아 환자의 신장 기능 저하 및 빈혈 상태를 반영하지 못함.
- **MPLite 통합 모델**:
  - 예측: 심부전 있음(True Positive)
  - 이유: 실험실 결과(크레아티닌 증가와 헤모글로빈 감소)를 진단 코드와 함께 통합하여 심부전 위험 요인을 효과적으로 반영.

---



#### Data Sample:
- **Patient ID**: 12345
- **Admission Records**:
  - **Diagnosis Codes**: Hypertension (ICD-9: 401), Diabetes (ICD-9: 250)
  - **Lab Results**:
    - Blood Glucose: 180 mg/dL (Above normal range)
    - Creatinine: 1.8 mg/dL (Above normal range, indicative of kidney dysfunction)
    - Hemoglobin: 10 g/dL (Below normal range, suggestive of anemia)
- **Prediction Target**: Heart Failure (ICD-9: 428) diagnosis at the next visit.

#### Results Comparison:
- **Baseline Model (GRU)**:
  - Prediction: No Heart Failure (False Negative)
  - Reason: Did not incorporate lab results, failing to capture kidney dysfunction and anemia as risk factors.
- **MPLite-Enhanced Model**:
  - Prediction: Heart Failure (True Positive)
  - Reason: Effectively integrated lab results (elevated creatinine and reduced hemoglobin) with diagnosis codes, accurately reflecting heart failure risk factors.

---

### 핵심 차이점 / Key Differences:
MPLite는 실험실 데이터를 통합함으로써 기존 모델이 놓쳤던 중요한 환자 상태 정보를 포착할 수 있었습니다. 이를 통해 더 민감하고 정확한 예측이 가능해졌습니다.



<br/>  
# 요약   



MPLite는 전자 건강 기록(EHR) 데이터에서 단일 방문 환자의 데이터를 효과적으로 활용하기 위해 설계된 가벼운 신경망 기반의 다중 측면 사전 학습 프레임워크입니다. MIMIC-III와 MIMIC-IV 데이터셋을 사용하여 진단 예측과 심부전 예측 과제를 수행했으며, 기존 모델 대비 가중 F1 점수, Recall@k, AUC 등에서 유의미한 성능 향상을 보여주었습니다. 특히 MPLite는 실험실 결과 데이터를 진단 코드와 통합하여 기존 모델들이 놓쳤던 단일 방문 환자의 중요 정보를 효과적으로 반영했습니다. 예를 들어, 기존 모델이 심부전을 예측하지 못한 사례에서, MPLite는 크레아티닌 상승과 빈혈 데이터를 반영하여 심부전을 정확히 예측했습니다. 이 프레임워크는 플러그 앤 플레이 방식으로 기존 모델에 쉽게 통합 가능하며, 의료 데이터 분석의 유연성과 확장성을 제공합니다.

---


MPLite is a lightweight neural network-based multi-aspect pretraining framework designed to effectively leverage single-visit patient data in electronic health records (EHR). Using the MIMIC-III and MIMIC-IV datasets, it demonstrated significant improvements in metrics like weighted F1, Recall@k, and AUC for diagnosis and heart failure prediction tasks compared to baseline models. MPLite integrates lab results with diagnosis codes, effectively capturing critical information from single-visit patients that previous models failed to utilize. For instance, MPLite successfully predicted heart failure by incorporating elevated creatinine and anemia data in cases where baseline models failed. This framework is easily integrable into existing models via a plug-and-play approach, offering flexibility and scalability in healthcare data analysis.




<br/>  
# 기타  



논문에서 제공된 테이블과 피규어는 모델 성능 비교, 데이터셋 통계, 그리고 프레임워크의 구조를 시각화한 자료들로 구성되어 있습니다. 주요 항목에 대해 설명드리겠습니다.

---



#### **Table II: MIMIC-III 데이터셋 통계**
- **내용**: MIMIC-III 데이터셋의 기본 통계 정보를 보여줍니다.
  - 총 환자 수: 46,520명
  - 다중 방문 환자 수: 7,537명 (16.2%)
  - 단일 방문 환자 수: 38,983명 (83.8%)
  - 실험실 결과 항목 수: 697개
  - 진단 코드 수: 4,880개
- **의의**: 데이터셋에서 대부분의 환자가 단일 방문 환자임을 강조하며, MPLite가 단일 방문 데이터를 활용하여 예측 성능을 개선하는 데 중점을 둠을 나타냅니다.

---

#### **Table III: 진단 및 심부전 예측 결과**
- **내용**: 다양한 모델(GRU, Dipole, G-BERT 등)과 MPLite 통합 여부에 따른 예측 성능을 비교합니다.
  - **평가 메트릭**:
    - 가중 F1 점수(weighted F1)
    - 상위 10개 및 20개 재현율(Recall@10, Recall@20)
    - AUC (심부전 예측의 경우)
  - **주요 결과**:
    - GRU 모델에서 MPLite 통합 후 가중 F1 점수가 MIMIC-III 데이터셋 기준 17.82에서 19.58로 상승.
    - Recall@10 및 Recall@20도 각각 31.56 → 33.82, 33.64 → 35.97로 향상.
    - 심부전 예측에서는 AUC가 80.54%에서 82.01%로 증가.
- **의의**: MPLite의 통합이 기존 모델 대비 진단 및 심부전 예측 성능을 일관되게 향상시켰음을 보여줍니다.

---

#### **Figure 2: MPLite 프레임워크 개요**
- **내용**: MPLite 프레임워크의 구조를 시각화.
  - **구성 요소**:
    1. **Pretraining Module**: 실험실 결과를 기반으로 진단 코드를 예측하는 단계.
    2. **Integration Module**: 사전 학습된 표현을 기존 모델과 통합하여 최종 예측 수행.
  - **작동 방식**:
    - 단일 방문 환자의 실험실 데이터를 활용하여 다중 방문 데이터와 유사한 패턴을 학습.
- **의의**: MPLite의 플러그 앤 플레이 구조와 데이터 통합 방법을 시각적으로 설명하며, 단순성과 확장성을 강조.

---

#### **Figure 1: 기존 모델의 한계**
- **내용**: 전통적인 예측 모델이 다중 방문 데이터에만 의존하며, 단일 방문 데이터를 활용하지 못하는 문제를 보여줍니다.
- **의의**: MPLite가 단일 방문 데이터를 보완적으로 활용할 필요성을 강조.

---



#### **Table II: MIMIC-III Dataset Statistics**
- **Details**: Provides basic statistics about the MIMIC-III dataset.
  - Total patients: 46,520
  - Multi-visit patients: 7,537 (16.2%)
  - Single-visit patients: 38,983 (83.8%)
  - Number of lab result items: 697
  - Number of diagnosis codes: 4,880
- **Significance**: Highlights that the majority of patients are single-visit, underscoring MPLite's focus on leveraging this data to enhance prediction performance.

---

#### **Table III: Diagnosis and Heart Failure Prediction Results**
- **Details**: Compares the performance of various models (GRU, Dipole, G-BERT, etc.) with and without MPLite integration.
  - **Metrics**:
    - Weighted F1 score
    - Recall@10 and Recall@20
    - AUC (for heart failure prediction)
  - **Key Findings**:
    - GRU with MPLite improved weighted F1 from 17.82 to 19.58 on the MIMIC-III dataset.
    - Recall@10 and Recall@20 increased from 31.56 → 33.82 and 33.64 → 35.97, respectively.
    - For heart failure prediction, AUC rose from 80.54% to 82.01%.
- **Significance**: Demonstrates consistent improvement in both diagnostic and heart failure prediction tasks when MPLite is integrated.

---

#### **Figure 2: Overview of MPLite Framework**
- **Details**: Visual representation of the MPLite framework structure.
  - **Components**:
    1. **Pretraining Module**: Predicts diagnosis codes from lab results.
    2. **Integration Module**: Combines pretrained representations with existing models for final prediction.
  - **Functionality**:
    - Utilizes single-visit lab data to learn patterns similar to multi-visit data.
- **Significance**: Explains the plug-and-play design and data integration approach of MPLite, highlighting its simplicity and scalability.

---

#### **Figure 1: Limitations of Traditional Models**
- **Details**: Illustrates the reliance of traditional models on multi-visit data, failing to utilize single-visit records.
- **Significance**: Emphasizes the need for MPLite to complement single-visit data usage.

---




<br/>
# refer format:     


@inproceedings{yang2024mplite,
  author    = {Eric Yang and Pengfei Hu and Xiaoxue Han and Yue Ning},
  title     = {{MPLite: Multi-Aspect Pretraining for Mining Clinical Health Records}},
  booktitle = {Proceedings of the 2024 IEEE International Conference on Big Data Workshop on Big Data and AI for Healthcare},
  address   = {Washington, D.C.},
  year      = {2024},
  month     = {December 15--18},
  note      = {To appear}
}



Yang, Eric, Pengfei Hu, Xiaoxue Han, and Yue Ning. "MPLite: Multi-Aspect Pretraining for Mining Clinical Health Records." In Proceedings of the 2024 IEEE International Conference on Big Data Workshop on Big Data and AI for Healthcare, Washington, D.C., December 15–18, 2024.



