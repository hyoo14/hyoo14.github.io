---
layout: post
title:  "[2026]MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events"
date:   2026-07-16 21:29:30 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 의료 기기 부작용 사건 보고서를 기반으로 한 다중 레이블 텍스트 분류(Multi-Label Text Classification, MLTC) 벤치마크인 MADE를 소개하고, 다양한 모델의 예측 성능과 불확실성 정량화(uncertainty quantification, UQ) 능력을 평가하였다.


짧은 요약(Abstract) :


이 논문에서는 의료 기기 부작용 사건에 대한 다중 레이블 텍스트 분류(Multi-Label Text Classification, MLTC)를 위한 새로운 벤치마크인 MADE를 소개합니다. 의료 분야와 같은 고위험 도메인에서는 강력한 예측 성능뿐만 아니라 신뢰할 수 있는 불확실성 정량화(uncertainty quantification, UQ)가 필요합니다. 기존의 MLTC 벤치마크는 데이터 오염, 레이블 불균형, 의존성 및 조합 복잡성으로 인해 한계가 있으며, MADE는 이러한 문제를 해결하기 위해 지속적으로 업데이트되는 데이터셋으로 구성되어 있습니다. MADE는 계층적 레이블의 긴 꼬리 분포를 특징으로 하며, 엄격한 시간 분할을 통해 재현 가능한 평가를 가능하게 합니다. 연구 결과, 작은 모델이 강력한 정확도를 유지하면서 경쟁력 있는 UQ를 제공하는 반면, 생성적 미세 조정은 가장 신뢰할 수 있는 UQ를 제공하는 것으로 나타났습니다. 이 연구는 MLTC의 신뢰성을 높이기 위한 기초를 제공하며, 향후 연구를 위한 기반을 마련합니다.



This paper introduces MADE, a new benchmark for Multi-Label Text Classification (MLTC) derived from medical device adverse event reports. In high-stakes domains like healthcare, strong predictive performance is not enough; reliable uncertainty quantification (UQ) is also essential. Existing MLTC benchmarks face limitations due to data contamination, label imbalances, dependencies, and combinatorial complexity. MADE addresses these issues by being a continuously updated dataset. It features a long-tailed distribution of hierarchical labels and enables reproducible evaluation with strict temporal splits. The results show that smaller models achieve strong accuracy while maintaining competitive UQ, whereas generative fine-tuning provides the most reliable UQ. This work lays the foundation for improving the reliability of MLTC and sets the stage for future research.


* Useful sentences :


* **예측 거부율**: 모델이 불확실하다고 판단해 예측을 내리지 않은 샘플의 비율입니다.  
* **Spearman 상관계수**: 모델의 불확실성 점수와 실제 오류 크기 사이의 순위 상관관계를 나타냅니다.(불확실성은 1-소프트맥스 점수)   
* **ECE+**: 모델이 예측한 신뢰도와 실제 정확도의 차이를 측정한 보정 오류로, 낮을수록 좋습니다.  
(예를 들어 신뢰도 80%인 예측 10개 중 7개가 정답이면, 해당 구간의 오차는 ∣0.7−0.8∣=0.1입니다. ECE는 0에 가까울수록 모델의 신뢰도가 실제 정확도와 잘 맞는다는 뜻 )   
   


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



이 논문에서는 의료 기기 부작용 사건 보고서를 기반으로 한 다중 레이블 텍스트 분류(Multi-Label Text Classification, MLTC) 벤치마크인 MADE를 소개합니다. 이 벤치마크는 다음과 같은 방법론을 사용하여 구축되었습니다.

1. **데이터 수집 및 전처리**: FDA에서 제공하는 의료 기기 부작용 사건 보고서를 수집하여, 각 보고서에서 사건 설명, 관련 메타데이터(사건 유형 및 기기 정보), 그리고 제품 및 환자 문제 레이블을 추출합니다. 이 과정에서 IMDRF(International Medical Device Regulatory Forum)에서 제공하는 계층적 코드로 레이블을 매핑하고, 레이블의 계층 구조를 활용하여 부모 및 조부모 레이블을 포함하도록 합니다. 데이터는 2015년부터 2025년 중반까지의 보고서를 포함하며, 각 레이블의 최소 발생 수를 설정하여 최종적으로 1,154개의 레이블을 포함하는 긴 꼬리 분포를 형성합니다.

2. **모델 아키텍처**: 다양한 모델 아키텍처를 사용하여 MLTC 작업을 수행합니다. 여기에는 인코더 전용 모델과 디코더 전용 모델이 포함됩니다. 특히, Llama 및 Ettin 모델을 사용하여 분류 작업을 수행하며, 이들 모델은 각각의 아키텍처에 따라 다르게 구성됩니다. 모델은 크게 두 가지 학습 패러다임인 **판별적 학습(Discriminative Learning)**과 **생성적 학습(Generative Learning)**으로 나뉘며, 각 패러다임에 따라 모델을 미세 조정합니다.

3. **훈련 기법**: 모델은 AdamW 옵티마이저와 코사인 스케줄러를 사용하여 훈련됩니다. 판별적 학습에서는 이진 교차 엔트로피 손실을 사용하고, 생성적 학습에서는 클래스 레이블을 토큰으로 생성하도록 훈련합니다. 또한, LoRA(Low-Rank Adaptation)와 같은 파라미터 효율적인 훈련 기법을 사용하여 모델의 성능을 향상시킵니다.

4. **불확실성 정량화(Quantification)**: 모델의 예측 불확실성을 정량화하기 위해 정보 기반 및 일관성 기반의 불확실성 측정 방법을 사용합니다. 정보 기반 불확실성(Uinfo)은 토큰 기반 메트릭(예: 엔트로피, 로그 확률 등)을 통해 측정되며, 일관성 기반 불확실성(Ucons)은 여러 번의 확률적 전방 패스를 통해 계산됩니다. 이 두 가지를 결합하여 최종 불확실성 점수를 산출합니다.

5. **평가 방법**: 모델의 성능은 매크로 F1 점수, Jaccard 점수, 예측 거부율(PRR), 스피어만 상관계수(ρ), 기대 교정 오차(ECE+) 등을 통해 평가됩니다. 이러한 평가 지표는 모델의 예측 성능과 불확실성 정량화의 신뢰성을 동시에 측정하는 데 사용됩니다.

이러한 방법론을 통해 MADE는 지속적으로 업데이트되는 벤치마크로, MLTC 분야에서의 모델 성능을 평가하는 데 중요한 역할을 합니다.

---




This paper introduces MADE, a living benchmark for Multi-Label Text Classification (MLTC) based on medical device adverse event reports. The methodology used to construct this benchmark includes the following:

1. **Data Collection and Preprocessing**: Adverse event reports from the FDA are collected, and from each report, the event description, relevant metadata (event type and device information), and product and patient problem labels are extracted. The labels are mapped using hierarchical codes provided by the International Medical Device Regulatory Forum (IMDRF), and the hierarchy of labels is utilized to include parent and grandparent labels. The data spans reports from 2015 to mid-2025, and by setting a minimum occurrence threshold for each label, a final set of 1,154 labels is formed, exhibiting a long-tailed distribution.

2. **Model Architecture**: Various model architectures are employed to perform the MLTC task, including encoder-only and decoder-only models. Specifically, Llama and Ettin models are used for classification tasks, with each model configured differently based on its architecture. The models are divided into two main learning paradigms: **Discriminative Learning** and **Generative Learning**, with fine-tuning performed according to each paradigm.

3. **Training Techniques**: The models are trained using the AdamW optimizer and a cosine scheduler. In discriminative learning, binary cross-entropy loss is utilized, while in generative learning, the models are trained to generate class labels as tokens. Additionally, parameter-efficient training techniques such as Low-Rank Adaptation (LoRA) are employed to enhance model performance.

4. **Uncertainty Quantification**: To quantify the uncertainty of model predictions, both information-based and consistency-based uncertainty measurement methods are used. Information-based uncertainty (Uinfo) is measured through token-based metrics (e.g., entropy, log probabilities), while consistency-based uncertainty (Ucons) is calculated through multiple stochastic forward passes. These two are combined to yield a final uncertainty score.

5. **Evaluation Methods**: The performance of the models is evaluated using metrics such as macro F1 score, Jaccard score, Prediction Rejection Rate (PRR), Spearman correlation (ρ), and Expected Calibration Error (ECE+). These evaluation metrics are used to simultaneously measure the predictive performance and the reliability of uncertainty quantification.

Through these methodologies, MADE serves as a continuously updated benchmark, playing a crucial role in evaluating model performance in the field of MLTC.


<br/>
# Results



이 논문에서는 MADE라는 새로운 다중 레이블 텍스트 분류(Multi-Label Text Classification, MLTC) 벤치마크를 소개하고, 의료 기기 부작용 보고서를 기반으로 한 다양한 모델의 성능을 평가합니다. MADE는 FDA에서 제공하는 의료 기기 부작용 보고서를 사용하여 생성된 데이터셋으로, 지속적으로 업데이트되어 데이터 오염을 방지합니다. 이 데이터셋은 1,154개의 상호 의존적인 레이블을 포함하고 있으며, 세 가지 계층적 수준으로 구성되어 있습니다.

#### 경쟁 모델
연구에서는 20개 이상의 인코더 및 디코더 모델을 사용하여 성능을 비교했습니다. 주요 모델로는 Llama-3.1-8B-Base, Llama-3.1-70B-Base, Ettin-1B-Encoder 등이 있으며, 이들은 각각의 학습 패러다임(차별적 미세 조정, 생성적 미세 조정, 몇 샷 프롬프트 등)에서 평가되었습니다.

#### 테스트 데이터
테스트 데이터는 2024년 7월부터 2025년 6월까지의 FDA 보고서를 포함하며, 이 데이터는 모델의 사전 학습 데이터와 겹치지 않도록 설계되었습니다. 데이터는 훈련 세트(2015-2023), 검증 세트(2024년 1-6월), 테스트 세트(2024년 7월 - 2025년 6월)로 나뉘어 있습니다.

#### 메트릭
모델의 성능은 매크로 F1 점수, Jaccard 지수, 예측 거부율(Precision Rejection Rate, PRR), 스피어만 상관계수(ρ), 기대 보정 오류(Expected Calibration Error, ECE+) 등의 메트릭을 사용하여 평가되었습니다. 매크로 F1 점수는 전체 클래스에 대한 평균 성능을 나타내며, Jaccard 지수는 예측의 정확성을 측정합니다. PRR은 불확실한 예측을 식별하는 모델의 능력을 평가하며, ρ는 불확실성과 정확성 간의 상관관계를 나타냅니다. ECE+는 긍정 클래스의 보정 오류를 측정하여 모델의 신뢰도를 평가합니다.

#### 결과
결과적으로, Llama-3.1-8B-Base 모델이 전체 매크로 F1 점수에서 0.54로 가장 높은 성능을 보였으며, Ettin-1B-Encoder는 PRR에서 0.52로 가장 높은 점수를 기록했습니다. 생성적 미세 조정 모델은 드문 레이블에 대한 성능을 향상시키는 데 유리했지만, 차별적 미세 조정 모델이 전반적으로 더 안정적이고 일관된 성능을 보였습니다. 또한, 불확실성 정량화(UQ) 방법으로는 토큰 엔트로피 기반의 방법이 가장 효과적이었으며, 자가 언급된 불확실성은 신뢰할 수 없는 것으로 평가되었습니다.

이 연구는 MADE 벤치마크를 통해 MLTC 분야에서의 모델 성능을 지속적으로 평가할 수 있는 기초를 제공하며, 향후 연구 방향으로는 생성적 미세 조정의 이점, 사고 모델의 불확실성 실패 원인, 새로운 추론 모델의 훈련 방법 등이 제안되었습니다.

---




This paper introduces a new benchmark for Multi-Label Text Classification (MLTC) called MADE, which is based on medical device adverse event reports. MADE is derived from reports provided by the FDA and is continuously updated to prevent data contamination. The dataset features 1,154 interdependent labels organized into three hierarchical levels.

#### Competing Models
The study evaluates over 20 encoder and decoder models, including key models such as Llama-3.1-8B-Base, Llama-3.1-70B-Base, and Ettin-1B-Encoder, comparing their performance across different learning paradigms (discriminative fine-tuning, generative fine-tuning, few-shot prompting, etc.).

#### Test Data
The test data includes FDA reports from July 2024 to June 2025, designed to avoid overlap with the models' pre-training data. The data is divided into training sets (2015-2023), validation sets (January to June 2024), and test sets (July 2024 - June 2025).

#### Metrics
Model performance is evaluated using metrics such as macro F1 score, Jaccard index, Prediction Rejection Rate (PRR), Spearman correlation (ρ), and Expected Calibration Error (ECE+). The macro F1 score indicates average performance across all classes, while the Jaccard index measures prediction accuracy. PRR assesses the model's ability to identify unreliable predictions, and ρ indicates the correlation between uncertainty and accuracy. ECE+ measures the calibration error for positive classes, assessing the model's reliability.

#### Results
Overall, the Llama-3.1-8B-Base model achieved the highest macro F1 score of 0.54, while the Ettin-1B-Encoder recorded the highest PRR at 0.52. Generative fine-tuning models showed advantages in performance for rare labels, but discriminative fine-tuning models exhibited more stable and consistent performance overall. Additionally, token-entropy-based uncertainty quantification methods proved to be the most effective, while self-verbalized uncertainty was deemed unreliable.

This research provides a foundation for ongoing evaluation of model performance in the MLTC field through the MADE benchmark, suggesting future research directions such as investigating the benefits of generative fine-tuning, understanding the failure of uncertainty in reasoning models, and training new reasoning models.


<br/>
# 예제



이 논문에서는 의료 기기 부작용 보고서를 다루는 다중 레이블 텍스트 분류(Multi-Label Text Classification, MLTC) 작업을 위한 새로운 벤치마크인 MADE를 소개합니다. MADE는 FDA(미국 식품의약국)에서 제공하는 의료 기기 부작용 보고서를 기반으로 하며, 지속적으로 업데이트되어 데이터 오염을 방지합니다. 이 벤치마크는 계층적 레이블을 포함하고 있으며, 각 레이블은 부모 및 자식 레이블과의 관계를 반영합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **샘플 수**: 298,825개
   - **기간**: 2015년부터 2023년까지의 보고서
   - **구성**: 각 보고서는 사건 설명, 관련 메타데이터(사건 유형 및 기기 정보), 그리고 제품 및 환자 문제 레이블을 포함합니다.
   - **예시**:
     - **입력**: "환자가 당뇨병성 케톤산증으로 입원했으며, 혈당 수치는 500 mg/dl였습니다."
     - **출력 레이블**: 
       ```
       A04: 제품 문제
       A0404: 균열
       E12: 내분비 및 영양
       E1205: 고혈당
       E120501: 당뇨병성 케톤산증
       ```

2. **테스트 데이터**:
   - **샘플 수**: 118,177개
   - **기간**: 2024년 7월부터 2025년 6월까지의 보고서
   - **구성**: 트레이닝 데이터와 유사하게 사건 설명 및 레이블을 포함합니다.
   - **예시**:
     - **입력**: "환자가 인슐린 주입 펌프를 사용 중에 부작용을 경험했습니다."
     - **출력 레이블**: 
       ```
       A05: 기계적 문제
       A0508: 소음
       E22: 진단 검사
       ```

#### 작업 설명
- **작업 목표**: 주어진 의료 기기 부작용 보고서를 분석하여 관련된 모든 레이블을 정확하게 분류하는 것입니다.
- **입력 형식**: 각 보고서는 사건 설명과 함께 제공되며, 레이블 정의가 포함된 태그가 있습니다.
- **출력 형식**: 모델은 각 레이블에 대해 확신 점수를 포함한 JSON 형식으로 결과를 반환합니다. 예를 들어:
  ```json
  {
    "A04": 0.92,
    "A0404": 0.74,
    "E12": 0.85
  }
  ```

이러한 방식으로 MADE 벤치마크는 MLTC 작업에서 모델의 성능을 평가하고, 불확실성 정량화(uncertainty quantification) 방법을 비교하는 데 사용됩니다.

---




This paper introduces MADE, a new benchmark for Multi-Label Text Classification (MLTC) tasks dealing with medical device adverse event reports. MADE is based on reports provided by the FDA (U.S. Food and Drug Administration) and is continuously updated to prevent data contamination. This benchmark features hierarchical labels, where each label reflects relationships with parent and child labels.

#### Training Data and Test Data

1. **Training Data**:
   - **Number of Samples**: 298,825
   - **Period**: Reports from 2015 to 2023
   - **Composition**: Each report includes an event description, relevant metadata (event type and device information), and product and patient problem labels.
   - **Example**:
     - **Input**: "The patient was hospitalized due to diabetic ketoacidosis, with a blood glucose level of 500 mg/dl."
     - **Output Labels**: 
       ```
       A04: Product Problems
       A0404: Crack
       E12: Endocrine and Nutrition
       E1205: Hyperglycemia
       E120501: Diabetic Ketoacidosis
       ```

2. **Test Data**:
   - **Number of Samples**: 118,177
   - **Period**: Reports from July 2024 to June 2025
   - **Composition**: Similar to the training data, including event descriptions and labels.
   - **Example**:
     - **Input**: "The patient experienced an adverse event while using the insulin infusion pump."
     - **Output Labels**: 
       ```
       A05: Mechanical Problems
       A0508: Noise
       E22: Diagnostic Tests
       ```

#### Task Description
- **Task Objective**: The goal is to analyze the given medical device adverse event report and accurately classify all relevant labels.
- **Input Format**: Each report is provided along with a tag containing the event description and label definitions.
- **Output Format**: The model returns results in JSON format, including confidence scores for each label. For example:
  ```json
  {
    "A04": 0.92,
    "A0404": 0.74,
    "E12": 0.85
  }
  ```

In this way, the MADE benchmark is used to evaluate model performance in MLTC tasks and compare uncertainty quantification methods.

<br/>
# 요약


이 논문에서는 의료 기기 부작용 사건 보고서를 기반으로 한 다중 레이블 텍스트 분류(Multi-Label Text Classification, MLTC) 벤치마크인 MADE를 소개하고, 다양한 모델의 예측 성능과 불확실성 정량화(uncertainty quantification, UQ) 능력을 평가하였다. 실험 결과, 특화된 판별 모델이 가장 높은 예측 성능을 보였으며, 생성적 미세 조정이 드문 클래스에 대한 성능을 향상시키는 동시에 UQ를 크게 개선하는 것으로 나타났다. 또한, 자가 언급된 불확실성은 신뢰할 수 없는 결과를 초래할 수 있어 주의가 필요하다는 점을 강조하였다.

---

This paper introduces MADE, a benchmark for multi-label text classification (MLTC) based on medical device adverse event reports, and evaluates the predictive performance and uncertainty quantification (UQ) capabilities of various models. The experimental results show that specialized discriminative models achieve the highest predictive performance, while generative fine-tuning significantly improves performance on rare classes and enhances UQ. Additionally, it emphasizes the need for caution as self-verbalized uncertainty can lead to unreliable outcomes.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Figure 1**: 벤치마킹 설정 개요
   - 이 다이어그램은 다양한 언어 모델과 학습 패러다임(판별적 또는 생성적 미세 조정 및 몇 샷 프롬프트)과 불확실성 정량화(UQ) 접근 방식을 보여줍니다. 이는 연구자들이 다양한 모델을 비교하고 평가하는 데 도움을 줍니다.

2. **Figure 2**: 제품 및 환자 문제의 계층적 다중 레이블
   - 이 피규어는 MADE 데이터셋의 레이블 구조를 시각적으로 나타내며, 각 레이블이 부모 및 조부모 레이블과 어떻게 연결되는지를 보여줍니다. 이는 모델이 레이블 간의 관계를 이해하는 데 중요한 정보를 제공합니다.

3. **Figure 3**: 레이블 빈도의 로그 스케일 분포
   - 이 그래프는 레이블의 빈도 분포를 보여주며, 긴 꼬리(long tail) 패턴을 강조합니다. 이는 다중 레이블 텍스트 분류에서 빈번한 레이블과 드문 레이블 간의 불균형 문제를 시각적으로 나타냅니다.

#### 테이블
1. **Table 1**: MADE 데이터셋 요약 통계
   - 데이터셋의 총 샘플 수, 훈련 세트, 검증 세트, 테스트 세트의 크기 및 레이블 수를 포함한 통계 정보를 제공합니다. 이는 연구자들이 데이터셋의 규모와 구조를 이해하는 데 도움을 줍니다.

2. **Table 2**: 예측 성능 및 UQ 결과
   - 다양한 모델과 학습 패러다임에 대한 예측 성능(매크로 F1, Jaccard J) 및 불확실성 정량화(예측 거부율, Spearman 상관계수, ECE+)를 비교합니다. 이 테이블은 각 모델의 강점과 약점을 명확히 보여줍니다.

3. **Table A.1**: BCE와 HYDRA 손실 비교
   - 두 가지 손실 함수(BCE와 HYDRA)를 사용한 모델의 성능을 비교합니다. HYDRA 손실이 특정 모델에서 더 나은 성능을 보이는 것을 확인할 수 있습니다.

#### 어펜딕스
- **A.1 데이터 가용성**: 데이터의 출처와 라이센스 정보를 제공합니다. 이는 연구의 투명성을 높이고, 다른 연구자들이 데이터를 재현할 수 있도록 돕습니다.
- **A.4 불확실성 정량화 방법**: 판별적 및 생성적 모델에서 불확실성을 정량화하는 방법을 설명합니다. 이는 연구자들이 모델의 신뢰성을 평가하는 데 필요한 정보를 제공합니다.




#### Diagrams and Figures
1. **Figure 1**: Overview of the Benchmarking Setup
   - This diagram illustrates the various language models and learning paradigms (discriminative or generative fine-tuning and few-shot prompting) along with uncertainty quantification (UQ) approaches. It aids researchers in comparing and evaluating different models.

2. **Figure 2**: Hierarchical Multi-Labels of Product and Patient Problems
   - This figure visually represents the label structure of the MADE dataset, showing how each label is connected to parent and grandparent labels. This provides crucial information for models to understand the relationships between labels.

3. **Figure 3**: Log-Scaled Distribution of Label Frequencies
   - This graph displays the frequency distribution of labels, emphasizing the long-tail pattern. It visually represents the imbalance issue between frequent and rare labels in multi-label text classification.

#### Tables
1. **Table 1**: Summary Statistics of the MADE Dataset
   - Provides statistical information including the total number of samples, sizes of training, validation, and test sets, and the number of labels. This helps researchers understand the scale and structure of the dataset.

2. **Table 2**: Predictive Performance and UQ Results
   - Compares predictive performance (macro F1, Jaccard J) and uncertainty quantification (Prediction Rejection Rate, Spearman correlation, ECE+) across various models and learning paradigms. This table clearly shows the strengths and weaknesses of each model.

3. **Table A.1**: Comparison of BCE and HYDRA Loss
   - Compares the performance of models using two loss functions (BCE and HYDRA). It confirms that the HYDRA loss performs better in certain models.

#### Appendix
- **A.1 Data Availability**: Provides information about the source and licensing of the data. This enhances the transparency of the research and helps other researchers reproduce the data.
- **A.4 Methods for Uncertainty Quantification**: Describes how uncertainty is quantified in both discriminative and generative models. This provides necessary information for researchers to assess the reliability of the models.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{agarwal2026made,
  title={MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events},
  author={Raunak Agarwal and Markus Wenzel and Simon Baur and Jonas Zimmer and George Harvey and Jackie Ma},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={46308--46328},
  year={2026},
  month={July},
  publisher={Association for Computational Linguistics},
}
```

### 시카고 스타일

Agarwal, Raunak, Markus Wenzel, Simon Baur, Jonas Zimmer, George Harvey, and Jackie Ma. "MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 46308–46328. Association for Computational Linguistics, July 2026. 
