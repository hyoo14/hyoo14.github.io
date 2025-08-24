---
layout: post
title:  "[2024]Conceptualizing bias in EHR data: A case study in performance disparities by demographic subgroups for a pediatric obesity incidence classifier"
date:   2025-08-24 20:33:15 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

연구진은 로지스틱 회귀, 랜덤 포레스트, 그래디언트 부스티드 트리, 신경망 등 네 가지 알고리즘을 훈련시켜 소아 환자를 비만 여부로 분류  

흥미롭게도 아프리카계 미국인 환자나 Medicaid 가입자 같은 소수자 집단에서 더 좋은 성능이 나타났습니다. 이는 이들 집단에서 비만과 강하게 연관된 특징(예: 특정 질환 진단)이 더 자주 관찰되었기 때문으로 해석


이러한 결과는 EHR 데이터와 기계학습 모델에서 편향이 단순히 불리하게만 작용하지 않고 복잡한 방식으로 나타날 수 있음을 보여줌  




짧은 요약(Abstract) :





이 논문은 **전자건강기록(EHR)** 데이터를 활용해 소아 비만 발생을 예측하는 기계학습 모델을 개발하고, 그 과정에서 나타나는 **편향(bias)** 을 분석한 연구입니다. 연구진은 로지스틱 회귀, 랜덤 포레스트, 그래디언트 부스티드 트리, 신경망 등 네 가지 알고리즘을 훈련시켜 소아 환자를 비만 여부로 분류했습니다. 평균 AUC-ROC 값은 0.72\~0.80 범위로 비교적 안정적인 성능을 보였으며, 흥미롭게도 아프리카계 미국인 환자나 Medicaid 가입자 같은 소수자 집단에서 더 좋은 성능이 나타났습니다. 이는 이들 집단에서 비만과 강하게 연관된 특징(예: 특정 질환 진단)이 더 자주 관찰되었기 때문으로 해석됩니다. 이러한 결과는 EHR 데이터와 기계학습 모델에서 **편향이 단순히 불리하게만 작용하지 않고 복잡한 방식으로 나타날 수 있음**을 보여주며, 향후 보다 공정한 모델 개발을 위해 고려해야 할 분석 틀을 제시합니다.

---


This study develops machine learning models using **Electronic Health Records (EHRs)** to predict pediatric obesity incidence and examines potential **biases** in model performance. Four algorithms—Logistic Regression, Random Forest, Gradient Boosted Trees, and Neural Networks—were trained to classify patients as obese or non-obese. Mean AUC-ROC values ranged from 0.72 to 0.80, showing consistent performance across models. Interestingly, the models performed better for minority subgroups such as African American patients and Medicaid enrollees. Permutation analysis suggested that these groups were more likely to exhibit diagnostic patterns strongly associated with obesity, which contributed to better predictive performance. These findings highlight that bias in EHR-based machine learning models can emerge in complex ways, sometimes favoring underrepresented groups, and underscore the need for comprehensive frameworks to identify and mitigate bias when building equitable predictive models.

---




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


---


### 1. 데이터셋 (EHR 기반 소아 비만 연구 데이터)

* 연구 데이터는 **Children’s Hospital of Philadelphia(CHOP)** 의 Pediatric Big Data (PBD) 자원에서 추출한 전자건강기록(EHR) 자료입니다.
* 연구 대상은 2009\~2017년 사이 **새롭게 비만으로 진단된 아동 환자(사례군, case)** 와 이와 매칭된 **건강한 BMI를 가진 대조군(control)** 환자들입니다.
* BMI 기준은 **CDC 아동 비만 정의**를 따랐으며, pre-index(비만 전)와 index(비만 진단 시점) 방문 기록을 모두 보유한 환자들만 포함했습니다.
* 최종적으로 **총 9,554명(4,777 쌍의 case-control pair)** 이 분석에 사용되었습니다.

---

### 2. 특징 추출 (Temporal Condition Patterns)

* 기존 연구(Campbell et al. 2020)에서 도출된 **시간적 질환 패턴(temporal condition patterns)** 을 주요 특징으로 사용했습니다.
* EHR 데이터에서 index 이전(pre-index)과 index 방문 시점(index)의 **진단 코드 조합**을 기반으로 패턴을 뽑았습니다.
* 원래 80개의 패턴이 있었으나, 본 연구에서는 **70개 패턴**을 선택해 모델 입력 특징(feature)으로 사용했습니다.
* 각 패턴은 **존재 여부(0/1 이진 변수)** 로 환자별로 부여되었습니다.
* 진단 코드는 **Expanded Diagnostic Clusters (EDCs)** 체계를 기반으로 표준화했습니다.

---

### 3. 추가 인구학적 변수

* 환자의 **성별, 인종/민족, 연령(2–4세, 5–11세, 12–18세), Medicaid 가입 여부**를 변수로 포함했습니다.
* Medicaid 가입은 사회경제적 지위를 반영하는 지표로 사용되었습니다.
* 인종/민족 정보가 누락된 경우 “Unknown”으로 분류했습니다.

---

### 4. 기계학습 모델 (Algorithms)

연구진은 **네 가지 대표적 분류 알고리즘**을 훈련시켰습니다:

1. **로지스틱 회귀 (Logistic Regression)**
2. **랜덤 포레스트 (Random Forest)**
3. **그래디언트 부스티드 트리 (Gradient Boosted Trees)**
4. **다층 퍼셉트론 신경망 (Neural Network, MLP)**

* 특히 **MLP 모델**은 **3개의 은닉층(hidden layers)** 구조를 사용했으며,

  * 활성화 함수: tanh, ReLU
  * 학습률: 고정(constant) 및 adaptive (0.1, 0.01, 0.001 초기값)
  * 은닉층 크기 조합: (30,50,30), (100,100,100), (50,100,50), (75,50,25)
  * 최적화 알고리즘: **Adam optimizer**

---

### 5. 학습 및 최적화 절차

* 데이터를 **층화(stratified) 방식**으로 50:50 class balance (비만 vs 비비만 환자 수 동일)로 훈련/검증 세트로 분할.
* **부트스트래핑(bootstrapping) 기반 최적화**:

  * 무작위 분할을 **200회 반복**
  * 각 모델·각 하이퍼파라미터 조합에 대해 **AUC-ROC** 측정
  * 평균 AUC-ROC가 가장 높은 조합을 최종 하이퍼파라미터로 선택
* 모든 알고리즘은 **Scikit-learn (Python 3)** 으로 구현했습니다.

---

### 6. 성능 평가

* 전체 집단과 인구학적 하위 집단별(인종, 성별, 연령, Medicaid 여부) **평균 AUC-ROC 값**을 계산해 모델 성능을 비교했습니다.
* 추가적으로 **Permutation Feature Importance 분석**을 실시해, 각 특징(질환 패턴)이 예측 성능에 기여하는 정도를 평가했습니다.

---



### 1. Dataset (EHR-based Pediatric Obesity Cohort)

* Data were obtained from the **Pediatric Big Data (PBD)** resource at the **Children’s Hospital of Philadelphia (CHOP)**.
* The study population included children newly diagnosed with obesity (cases) and matched healthy BMI controls between 2009–2017.
* Obesity was defined using **CDC BMI z-score thresholds** (≥95th percentile for age and sex).
* Both **pre-index** (before obesity diagnosis) and **index** (at diagnosis) visits were required.
* The final dataset contained **9,554 patients (4,777 case-control pairs)**.

---

### 2. Feature Extraction (Temporal Condition Patterns)

* Leveraged previously identified **temporal condition patterns** from EHR data (Campbell et al. 2020).
* Derived from diagnostic code sequences at **pre-index** and **index** visits.
* From 80 original patterns, **70 were selected** for this study.
* Each pattern was encoded as a **binary feature (0/1)** per patient.
* Diagnostic codes were standardized using the **Expanded Diagnostic Clusters (EDCs)** system.

---

### 3. Demographic Variables

* Included **sex, race/ethnicity, age group (2–4, 5–11, 12–18 years), and Medicaid enrollment**.
* Medicaid/CHIP enrollment served as a proxy for socioeconomic status.
* Missing race/ethnicity values were classified as “Unknown.”

---

### 4. Machine Learning Models

Four supervised classification algorithms were trained:

1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosted Trees (GBT)**
4. **Multi-layer Perceptron (MLP) Neural Networks**

* The MLP had **3 hidden layers**, with hyperparameters including:

  * Activation: tanh, ReLU
  * Learning rates: constant/adaptive (0.1, 0.01, 0.001)
  * Hidden layer configurations: (30,50,30), (100,100,100), (50,100,50), (75,50,25)
  * Optimizer: **Adam**

---

### 5. Training & Hyperparameter Optimization

* Data split into training and validation sets with **stratified sampling** ensuring a balanced 50:50 ratio of obese vs non-obese cases.
* **Bootstrapping-based optimization**:

  * 200 randomized stratified train-test splits.
  * Each hyperparameter combination evaluated by **AUC-ROC**.
  * The setting with the highest mean AUC-ROC selected as optimal.
* Implemented using **Scikit-learn (Python 3)**.

---

### 6. Evaluation

* **Mean AUC-ROC scores** were reported for the overall population and across demographic subgroups (e.g., race, sex, Medicaid).
* **Permutation feature importance analysis** was conducted to identify the most predictive temporal condition patterns and demographic characteristics.

---




<br/>
# Results



### 1. 연구 대상 집단 특성

* 최종 연구 집단은 **총 9,554명**으로, 사례군(case, 비만 아동) 4,777명과 대조군(control, 정상 BMI 아동) 4,777명으로 구성됨.
* 인구학적 특성:

  * 성별: 남아 55.4%, 여아 44.6%
  * 인종/민족: 백인(60.1%) > 아프리카계 미국인(25.3%) > 히스패닉(3.5%) > 아시아인(2.9%)
  * 사회경제적 지위: 약 **32.1%가 Medicaid/CHIP 가입자**
* 사례군(case)에서는 아프리카계 미국인(31.7%)과 Medicaid 가입자(38.1%) 비율이 전체 집단보다 높았으며, 대조군(control)에서는 백인(66.3%) 비율이 더 높게 나타남.

---

### 2. 경쟁 모델별 성능 (AUC-ROC 기준)

* 평가 지표는 **AUC-ROC (Receiver Operating Characteristic curve 면적)** 사용.
* 전체 집단 및 하위 인구학적 집단에서 **0.72\~0.80 범위**의 일관된 성능을 보임.
* 모델별 평균 성능:

  * **Logistic Regression:** 0.78
  * **Gradient Boosted Trees (GBT):** 0.78
  * **Random Forest:** 0.77
  * **Neural Network (MLP):** 0.76
* 즉, **로지스틱 회귀와 GBT가 가장 높은 성능**을 기록했으며, 복잡한 모델(Random Forest, Neural Net) 대비 큰 우위를 보이지는 않음.

---

### 3. 하위 그룹별 성능 차이 (편향 분석)

* **가장 높은 성능:**

  * Medicaid 가입자: AUC 0.80 (GBT, Logistic Regression)
  * 아동 연령 5–11세 그룹: AUC 0.80
* **아프리카계 미국인 환자:** 평균 AUC 0.78–0.79 (모든 모델에서 높은 성능 유지)
* **가장 낮은 성능:**

  * 인종 정보가 없는 집단(Unknown): 0.72–0.73
  * 다인종 그룹(Multiple Race): 0.73–0.76
  * 백인 집단: 0.74–0.76 (상대적으로 낮음)
* **해석:** 일반적으로 소수자 집단이 성능에서 불리한 경우가 많은데, 이번 연구에서는 오히려 Medicaid 가입자 및 아프리카계 미국인 집단에서 성능이 더 높게 나타남. 이는 이들 집단에서 비만과 강하게 연관된 진단 패턴(예: 천식, 알레르기 비염)이 더 자주 관찰되었기 때문으로 분석됨.

---

### 4. Permutation Feature Importance 분석 결과

* 네 가지 모델 모두에서 공통적으로 가장 중요한 예측 특징(top predictors):

  1. **천식 진단 (pre-index 방문)**
  2. **근골격계 증상 (index 방문)**
  3. **알레르기 비염 (index 방문)**
  4. **여드름 (index 방문)**
* 특히, **천식 진단**과 **알레르기 비염**은 아프리카계 미국인 환자와 Medicaid 가입자에서 과대표집(over-represented) 되어 있었음.
* 예: 천식 진단(1-ALL04)을 가진 환자의 49.4%가 아프리카계 미국인(전체 집단 내 비율은 25.3%임).

---

### 5. 주요 해석

* 전반적으로 모델 간 성능 차이는 크지 않았으며, 로지스틱 회귀와 GBT가 가장 안정적 성능을 보임.
* 소수자 집단에서 더 나은 성능이 나온 것은 **해당 집단에서 예측에 강하게 기여하는 질환 패턴이 더 빈번하게 나타났기 때문**으로 설명됨.
* 즉, 모델이 공평(fair)하게 학습되었다기보다, **질환 패턴 분포 차이가 모델 성능 편향을 유리하게 만든 사례**라 할 수 있음.

---



### 1. Study Population Characteristics

* Final cohort: **9,554 patients** (4,777 obese cases, 4,777 matched controls).
* Demographics:

  * Gender: 55.4% male, 44.6% female
  * Race/Ethnicity: White (60.1%), African American (25.3%), Hispanic (3.5%), Asian (2.9%)
  * Socioeconomic status: **32.1% enrolled in Medicaid/CHIP**
* Compared to the full cohort, the case group had higher proportions of **African American (31.7%)** and **Medicaid enrollees (38.1%)**, while the control group was predominantly White (66.3%).

---

### 2. Model Performance (AUC-ROC)

* Evaluation metric: **AUC-ROC**.
* Overall performance across models: **0.72–0.80**.
* Mean AUC-ROC by model:

  * Logistic Regression: **0.78**
  * Gradient Boosted Trees: **0.78**
  * Random Forest: **0.77**
  * Neural Networks (MLP): **0.76**
* **Logistic Regression and GBT performed best**, with only marginal improvements over more complex models.

---

### 3. Subgroup Performance Disparities (Bias Analysis)

* **Best-performing subgroups:**

  * Medicaid/CHIP enrollees: AUC up to **0.80** (GBT, Logistic Regression)
  * Age 5–11 years: AUC **0.80**
* **African American patients:** consistently high performance (AUC 0.78–0.79 across all models).
* **Lowest-performing subgroups:**

  * Unknown race: 0.72–0.73
  * Multiple race: 0.73–0.76
  * White patients: 0.74–0.76
* **Interpretation:** Contrary to expectations, models performed **better for minority and low-income groups**. This was likely due to these groups being more frequently associated with predictive diagnostic patterns strongly linked to obesity (e.g., asthma, allergic rhinitis).

---

### 4. Permutation Feature Importance Findings

* Top predictive features across all models:

  1. **Asthma (pre-index visit)**
  2. **Musculoskeletal symptoms (index visit)**
  3. **Allergic rhinitis (index visit)**
  4. **Acne (index visit)**
* Example: **Asthma diagnosis** was observed in 49.4% of African American patients (vs 25.3% in the overall cohort) and 47.5% of Medicaid enrollees (vs 32.1% overall), explaining their over-representation in high-performing groups.

---

### 5. Key Interpretation

* No single model substantially outperformed others; Logistic Regression and GBT offered slightly better consistency.
* Better performance for minority groups reflects **disease pattern distributions**, not necessarily fairer modeling.
* Thus, bias in EHR-based ML can sometimes manifest in ways that **favor underrepresented subgroups**, depending on feature prevalence.

---




<br/>
# 예제




### 1. 학습(Task) 정의

* 본 연구의 주요 테스크는 **소아 비만 발생 예측(Classification of pediatric obesity incidence)** 입니다.
* 목표: **아동이 특정 시점(index visit)에서 비만(Obese)인지 아닌지(Non-obese) 예측**
* 문제 유형: **이진 분류(binary classification)**

---

### 2. 입력 데이터 (Input Features)

#### (1) 시간적 질환 패턴 (Temporal Condition Patterns)

* **Pre-index visit (비만 진단 이전 방문)** 과 **Index visit (비만으로 처음 진단된 방문)** 에서 환자에게 기록된 **질환 코드**를 조합해 **패턴**으로 정의.
* 총 70개 패턴(예: 천식 진단, 알레르기 비염, 여드름, 근골격계 증상 등).
* 각 패턴은 환자별로 **0/1 이진 변수**로 인코딩됨.

  * 예:

    * Input: {천식=1, 알레르기 비염=0, 근골격계 증상=1, 여드름=0, …}

#### (2) 인구학적 변수 (Demographics)

* 성별(Sex, Male/Female)
* 연령대(Age group: 2–4세 / 5–11세 / 12–18세)
* 인종/민족(Race/Ethnicity: White, Black, Hispanic, Asian, Multiple, Unknown)
* Medicaid 가입 여부 (Socioeconomic status proxy: Yes/No)

---

### 3. 출력 데이터 (Output Labels)

* 각 환자는 **Binary label** 부여:

  * **1 = Obese (사례군, Case)** : pre-index 방문에서 정상 BMI → index 방문에서 비만 BMI
  * **0 = Non-obese (대조군, Control)** : pre-index, index 방문 모두 정상 BMI

예:

* Input: {Sex=Male, Age=10, Race=Black, Medicaid=Yes, 천식=1, 알레르기 비염=0, …}
* Output: 1 (비만 발생)

---

### 4. 트레이닝 데이터 (Training Data)

* **Case-control matched dataset** 기반:

  * 사례군(case): 4,777명
  * 대조군(control): 4,777명
* 입력: 70개 temporal condition 패턴 + 4개 인구학적 변수
* 출력: 비만 여부(0/1)
* 데이터 분할: **층화(Stratified split)** 방식으로 50:50 비만/비비만 균형 유지

---

### 5. 테스트 데이터 (Test Data)

* 동일한 구조의 입력 특징(패턴 + 인구학 변수)과 레이블(0/1)을 포함.
* 훈련 과정에서 보지 않은 새로운 환자 데이터로 구성.
* 테스트에서 모델은 **예측 확률(Patient Obese probability)** 을 산출 → ROC curve 기반 AUC 계산

---

### 6. 구체적인 테스크 예시

* **Task:** "이 환자가 다음 방문(Index)에서 비만 판정을 받을 가능성이 있는가?"
* **Input (예시 환자):**

  * 나이: 9세 (5–11 그룹)
  * 성별: 남
  * 인종: African American
  * Medicaid: 가입
  * Temporal patterns: 천식=1, 알레르기 비염=1, 근골격계 증상=0, 여드름=0, …
* **Output:** 1 (예측: 비만 발생)

---



### 1. Task Definition

* The primary task was **predicting pediatric obesity incidence**.
* Goal: Determine whether a child would be classified as **Obese vs Non-obese** at their index visit.
* Problem type: **Binary classification**

---

### 2. Input Data (Features)

#### (1) Temporal Condition Patterns

* Derived from **pre-index (before obesity diagnosis)** and **index (at diagnosis)** visits.
* 70 selected diagnostic patterns (e.g., asthma, allergic rhinitis, musculoskeletal symptoms, acne).
* Each encoded as **binary variable (0/1)**.

  * Example:

    * Input: {Asthma=1, Allergic rhinitis=0, Musculoskeletal symptoms=1, Acne=0, …}

#### (2) Demographic Variables

* Sex (Male/Female)
* Age group (2–4, 5–11, 12–18 years)
* Race/Ethnicity (White, Black, Hispanic, Asian, Multiple, Unknown)
* Medicaid/CHIP enrollment (Yes/No, socioeconomic proxy)

---

### 3. Output Data (Labels)

* Each patient labeled as:

  * **1 = Obese (Case):** non-obese BMI at pre-index → obese BMI at index visit
  * **0 = Non-obese (Control):** healthy BMI at both pre-index and index visits

Example:

* Input: {Sex=Male, Age=10, Race=Black, Medicaid=Yes, Asthma=1, Allergic rhinitis=0, …}
* Output: 1 (Obese incidence)

---

### 4. Training Data

* Case-control matched dataset:

  * 4,777 obese cases, 4,777 matched controls
* Inputs: 70 temporal patterns + 4 demographic variables
* Output: Obesity label (0/1)
* Data split: **Stratified 50:50** balance of obese vs non-obese patients

---

### 5. Test Data

* Same input structure (patterns + demographics) with labels.
* Comprised of unseen patient records.
* Model generates **probability of obesity incidence**, evaluated using **AUC-ROC**.

---

### 6. Concrete Example Task

* **Task:** "Will this child be diagnosed as obese at their next (index) visit?"
* **Input (example patient):**

  * Age: 9 (5–11 group)
  * Sex: Male
  * Race: African American
  * Medicaid: Enrolled
  * Temporal patterns: Asthma=1, Allergic rhinitis=1, Musculoskeletal symptoms=0, Acne=0, …
* **Output:** 1 (Predicted Obesity incidence)

---






<br/>
# 요약




1. 본 연구는 CHOP의 전자건강기록(EHR) 데이터를 활용하여 70개의 시간적 질환 패턴과 인구학적 변수를 입력으로 한 소아 비만 발생 예측 모델을 개발했다.
2. 로지스틱 회귀, 랜덤 포레스트, 그래디언트 부스티드 트리, 신경망 모델을 비교한 결과 AUC-ROC 0.72\~0.80 범위의 성능을 보였으며, 소수자 집단(아프리카계 미국인, Medicaid 가입자)에서 오히려 더 높은 성능을 기록했다.
3. 예시는 환자의 나이, 성별, 인종, Medicaid 여부, 천식·알레르기 비염 등 진단 패턴을 입력으로 하고, 비만 여부(0/1)를 출력하는 이진 분류 테스크로 제시되었다.

---



1. This study developed pediatric obesity incidence prediction models using CHOP EHR data, with 70 temporal condition patterns and demographic variables as inputs.
2. Logistic Regression, Random Forest, Gradient Boosted Trees, and Neural Networks achieved AUC-ROC values ranging from 0.72–0.80, with minority groups (African Americans, Medicaid enrollees) showing unexpectedly higher performance.
3. The task was framed as binary classification, where inputs included age, sex, race, Medicaid status, and diagnostic patterns (e.g., asthma, allergic rhinitis), and the output was obesity status (0/1).

---



<br/>
# 기타






### 1. 다이어그램 (Diagram)

* **Fig 1: Bias Framework in EHR Data**

  * 전자건강기록(EHR) 데이터에서 발생 가능한 **편향의 세 가지 출처**(측정 오류, 샘플링 편향, 맥락적/사회적 편향)를 도식화.
  * 이러한 편향이 모델 설계와 알고리즘 과정에 반영되어 **성능 격차 및 불평등을 초래**할 수 있음을 보여줌.
  * 인사이트: **편향은 데이터 수집 단계부터 모델 구현까지 전 과정에서 발생 가능**하며, 단일 원인으로 설명되지 않음.

---

### 2. 피규어 (Figures)

* **Fig 2: 환자 선택 플로우차트**

  * 최종 분석에 포함된 환자(총 9,554명)가 어떻게 선별되었는지를 단계별로 제시.
  * 비만 사례군(case)와 대조군(control)을 매칭하고, BMI·보험 정보 기준을 충족하는지 검증하는 과정이 시각적으로 표현됨.
  * 인사이트: 연구의 **데이터 정제 과정과 포함·제외 기준**을 명확히 보여주며, 편향 가능성을 이해하는 데 도움을 줌.

* **Fig 3\~4 (Permutation Analysis 관련)**

  * 특정 진단 패턴(예: 천식, 알레르기 비염, 근골격계 증상, 여드름)이 모델 성능에 얼마나 기여했는지 시각적으로 표현.
  * 아프리카계 미국인과 Medicaid 가입자가 이 특징을 더 많이 보유해 **모델이 해당 집단에서 더 잘 작동한 이유**를 뒷받침.
  * 인사이트: **질환-비만 연관 패턴이 인구학적 편향 성능 차이를 설명하는 핵심 요인**임을 보여줌.

---

### 3. 테이블 (Tables)

* **Table 1: 연구 집단 인구학적 특성**

  * 사례군과 대조군 간 성별, 인종, Medicaid 비율 차이를 정량적으로 제시.
  * 인사이트: 사례군에 소수자 비율이 높아 **모델 성능 편차의 배경**이 되었음을 확인 가능.

* **Table 2: 모델별 AUC-ROC 성능**

  * 로지스틱 회귀, GBT, 랜덤 포레스트, 신경망 각각의 평균 AUC-ROC 제시 (0.72–0.80).
  * 인사이트: **단순한 로지스틱 회귀가 복잡한 모델과 성능 면에서 큰 차이가 없다는 점**을 보여줌.

* **Table 3: 가장 중요한 예측 변수(Top Predictors)**

  * 네 모델 공통적으로 중요한 특징: 천식, 알레르기 비염, 근골격계 증상, 여드름.
  * 인사이트: **특정 진단 패턴이 소수자 집단에서 더 흔하기 때문에 성능이 더 높게 나타남**을 입증.

* **Table 4: 상위 특징 보유 환자의 인구학적 특성**

  * 예: 천식 진단을 가진 환자의 절반 가까이가 아프리카계 미국인, 절반이 Medicaid 가입자.
  * 인사이트: **질환 패턴과 사회적 요인이 결합해 모델 성능 차이를 만들어냄**을 보여줌.

---

### 4. 어펜딕스 (Appendix, Supporting Info)

* **S1 Table:** 사례군/대조군 포함·제외 기준 정리.
* **S2 Table:** 분석에 사용된 70개 질환 패턴 리스트.
* **S3 Table:** 트레이닝/테스트 세트 인구학적 특성.
* **S4–S5 Tables:** ANOVA 결과 – 모델 및 하위 집단 간 성능 차이가 통계적으로 유의함을 검증.
* 인사이트: 부록은 연구의 **재현 가능성과 통계적 신뢰성**을 뒷받침. 특히, 성능 차이가 단순한 우연이 아니라 통계적으로 유의하다는 점을 강조.

---



### 1. Diagram

* **Fig 1: Framework of Bias in EHR Data**

  * Illustrates three major sources of bias in EHR data: measurement error, sampling bias, and contextual/social bias.
  * Shows how these biases propagate through model design and algorithmic processes, leading to **performance disparities and inequities**.
  * Insight: Bias is **multi-layered and present throughout the pipeline**, from data collection to model deployment.

---

### 2. Figures

* **Fig 2: Patient Selection Flowchart**

  * Visualizes how the final cohort of 9,554 patients was selected step by step, including BMI and insurance criteria.
  * Insight: Clarifies the **data curation and filtering process**, highlighting where potential biases may enter.

* **Fig 3–4 (Permutation Analysis)**

  * Show the contribution of top features (e.g., asthma, allergic rhinitis, musculoskeletal symptoms, acne) to predictive performance.
  * Support the finding that African American and Medicaid patients were overrepresented in high-predictive patterns.
  * Insight: Feature distribution explains why **models performed better on minority groups**.

---

### 3. Tables

* **Table 1: Demographics of Study Population**

  * Provides distribution of sex, race, and Medicaid status across cases and controls.
  * Insight: Higher minority representation in cases partly explains performance differences.

* **Table 2: AUC-ROC Across Models**

  * Summarizes performance of Logistic Regression, GBT, Random Forest, and Neural Nets (0.72–0.80).
  * Insight: **Simple Logistic Regression achieved comparable performance to more complex models**.

* **Table 3: Top Predictive Features**

  * Lists most predictive features shared across models (asthma, allergic rhinitis, musculoskeletal symptoms, acne).
  * Insight: Minority patients disproportionately carried these features, explaining favorable performance.

* **Table 4: Demographics of Patients with Top Features**

  * Example: Nearly half of patients with asthma diagnosis were African American and Medicaid enrolled.
  * Insight: Shows how **disease patterns intersect with socioeconomic factors to shape model performance**.

---

### 4. Appendix (Supporting Information)

* **S1 Table:** Case/control inclusion and exclusion criteria.
* **S2 Table:** List of 70 temporal condition patterns used.
* **S3 Table:** Training and test set demographics.
* **S4–S5 Tables:** ANOVA tests confirming statistically significant subgroup performance differences.
* Insight: Provides **robustness and reproducibility evidence**, showing that observed disparities are statistically valid, not random.

---





<br/>
# refer format:




```bibtex
@article{Campbell2024EHRBias,
  author    = {Elizabeth A. Campbell and Saurav Bose and Aaron J. Masino},
  title     = {Conceptualizing bias in EHR data: A case study in performance disparities by demographic subgroups for a pediatric obesity incidence classifier},
  journal   = {PLOS Digital Health},
  year      = {2024},
  volume    = {3},
  number    = {10},
  pages     = {e0000642},
  doi       = {10.1371/journal.pdig.0000642},
  url       = {https://doi.org/10.1371/journal.pdig.0000642}
}
```

---



Campbell, Elizabeth A., Saurav Bose, and Aaron J. Masino. “Conceptualizing Bias in EHR Data: A Case Study in Performance Disparities by Demographic Subgroups for a Pediatric Obesity Incidence Classifier.” *PLOS Digital Health* 3, no. 10 (October 23, 2024): e0000642. [https://doi.org/10.1371/journal.pdig.0000642](https://doi.org/10.1371/journal.pdig.0000642).

---


