---
layout: post
title:  "[2025]mRNA2vec: mRNA Embedding with Language Model in the 5′ UTR-CDS for mRNA Design"  
date:   2025-03-04 09:43:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


내 기억에는.. CDS영역에 더 가중치를 준 마스킹이 특이했음  
그리고 코돈 수준에서만 토크나이징한 것도 특이한 rna LM!  

짧은 요약(Abstract) :    





mRNA(전령 RNA) 기반 백신은 신약 개발을 가속화하고 제약 산업에 혁신을 일으키고 있다. 그러나 방대한 mRNA 라이브러리에서 백신과 치료제에 적합한 특정 mRNA 서열을 선택하는 것은 비용이 많이 든다. 효과적인 mRNA 치료제는 최적화된 발현 수준과 안정성을 갖춘 정밀한 서열 설계가 필요하다. 이 연구에서는 새로운 문맥적 언어 모델 기반 임베딩 방법인 **mRNA2vec**을 제안한다. 기존 mRNA 임베딩 접근 방식과 달리, 본 방법은 **data2vec**의 교사-학생(Self-Supervised Teacher-Student) 학습 프레임워크를 기반으로 한다. 또한, 5′ 비번역 영역(5′ UTR)과 코딩 서열(CDS) 영역을 하나의 입력 서열로 결합하여 처리한다. 

mRNA2vec의 핵심 특징은 다음과 같다:
1. mRNA 서열 내 위치의 중요성을 반영하여 확률적 마스킹 전략을 적용한다.
2. 최소 자유 에너지(MFE) 예측 및 이차 구조(SS) 분류를 추가적인 사전 학습 과제로 활용한다.

mRNA2vec은 기존 최첨단 방법(SOTA)인 **UTR-LM**보다 5′ UTR에서 **번역 효율(TE)과 발현 수준(EL) 예측 성능이 크게 향상**됨을 보였다. 또한, mRNA 안정성과 단백질 생산 수준을 예측하는 CDS 기반 과제에서도 **CodonBERT**와 비교해 경쟁력 있는 성능을 나타냈다.  

---



Messenger RNA (mRNA)-based vaccines are accelerating drug discovery and revolutionizing the pharmaceutical industry. However, selecting specific mRNA sequences for vaccines and therapeutics from large mRNA libraries is costly. Effective mRNA therapeutics require carefully designed sequences with optimized expression levels and stability. This paper proposes **mRNA2vec**, a novel contextual language model-based embedding method. Unlike existing mRNA embedding approaches, this method is based on the **self-supervised teacher-student learning framework of data2vec**. Additionally, it treats the **5′ untranslated region (UTR) and coding sequence (CDS) as a single input sequence**.

The key features of mRNA2vec include:
1. A **probabilistic masking strategy** that accounts for positional importance in mRNA sequences.
2. **Minimum Free Energy (MFE) prediction and Secondary Structure (SS) classification** as auxiliary pretraining tasks.

mRNA2vec significantly improves **Translation Efficiency (TE) and Expression Level (EL) prediction** in the 5′ UTR compared to the state-of-the-art **UTR-LM** method. It also shows competitive performance in **mRNA stability and protein production level prediction** tasks in CDS when compared to **CodonBERT**.





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






#### **1. 모델 개요 (mRNA2vec)**
mRNA2vec는 **5′ 비번역 영역(UTR)과 코딩 서열(CDS)**을 하나의 입력으로 통합하여 mRNA 서열을 임베딩하는 새로운 언어 모델이다. 기존의 mRNA 임베딩 방법과 달리, **data2vec 기반의 교사-학생(Self-Supervised Teacher-Student) 학습 프레임워크**를 사용하여 문맥적 표현을 학습한다.  

#### **2. 아키텍처 개요**  
- **데이터 입력**: 5′ UTR과 CDS를 연결하여 하나의 입력 시퀀스로 처리  
- **토크나이제이션**: 3-뉴클레오타이드 단위로 서열을 토큰화 (64개의 토큰)  
- **마스킹 전략**:  
  - 일반적인 마스크드 언어 모델(MLM)과 달리, 특정 위치(5′ UTR의 끝과 CDS의 시작)에 **확률적 하드 마스킹 적용**  
  - 이를 통해 번역 조절에 중요한 서열의 의미적 표현을 더욱 효과적으로 학습  

- **사전 학습 기법**:
  - **data2vec 프레임워크 적용**: 기존 MLM 방식(T5 등)과 달리, 마스킹된 서열뿐만 아니라 **비마스킹된 서열의 문맥까지 반영**하여 보다 풍부한 특징 학습 가능  
  - **보조 과제(Auxiliary Tasks)**:
    1. **최소 자유 에너지(MFE) 예측** → mRNA 구조 안정성을 고려한 학습  
    2. **이차 구조(SS) 분류** → 뉴클레오타이드 구조적 특성을 반영한 학습  

#### **3. 학습 데이터 (Training Data)**  
- **사전 학습 데이터**: NIH 데이터셋에서 수집한 5종(인간, 쥐, 생쥐, 닭, 제브라피시) mRNA 서열 (510k 개)  
- **다운스트림 작업 평가 데이터**:
  1. **5′ UTR 데이터** (발현 수준 및 번역 효율 예측)  
     - HEK293T (14k 개), PC3 (12k 개), 근육 세포(1k 개)  
  2. **CDS 데이터** (mRNA 안정성 및 단백질 생산 수준 예측)  
     - mRNA 안정성 데이터셋 (25k 개)  
     - 단백질 생산성 데이터셋 (6k 개)  

#### **4. 모델 학습 설정**  
- **모델 아키텍처**:  
  - T5 인코더를 기반으로 한 데이터2벡(data2vec) 구조  
  - 4개 **자기-어텐션(Self-Attention) 헤드**, 4개 은닉 레이어, 256차원 벡터 임베딩  
  - 모델 파라미터: 총 **10M** (실제 학습 가중치 3M)  
- **최적화 설정**:
  - 옵티마이저: AdamW (learning rate = 0.001, weight decay = 0.001)  
  - 배치 크기: 256  
  - 학습 환경: 4개의 NVIDIA RTX 4090 GPU  

#### **5. 모델 성능 평가**  
- **5′ UTR 기반 번역 효율(TE) 및 발현 수준(EL) 예측**에서 **SOTA 방법(U.TR-LM) 대비 최대 14% 성능 향상**  
- **CDS 기반 mRNA 안정성과 단백질 생산성 예측**에서도 CodonBERT와 비교하여 경쟁력 있는 성능을 보임  

---



#### **1. Model Overview (mRNA2vec)**
mRNA2vec is a **novel language model designed to embed mRNA sequences** by treating the **5′ untranslated region (UTR) and coding sequence (CDS) as a single input sequence**. Unlike traditional mRNA embedding approaches, it employs a **self-supervised teacher-student framework inspired by data2vec**, allowing for contextual sequence representation learning.  

#### **2. Architecture Overview**  
- **Input Processing**: Concatenates the 5′ UTR and CDS into a single input sequence  
- **Tokenization**: Uses 3-nucleotide tokenization (64 tokens)  
- **Masking Strategy**:  
  - Unlike traditional Masked Language Models (MLMs) such as T5, **probabilistic hard masking** is applied to specific regions (the end of 5′ UTR and the beginning of CDS)  
  - This enhances the model’s ability to learn critical regulatory sequences  

- **Pretraining Approach**:
  - **data2vec Framework**: Unlike standard MLMs, data2vec **considers both masked and unmasked sequences**, capturing richer sequence information  
  - **Auxiliary Tasks**:
    1. **Minimum Free Energy (MFE) Prediction** → Incorporates mRNA stability  
    2. **Secondary Structure (SS) Classification** → Integrates structural properties of nucleotides  

#### **3. Training Data**  
- **Pretraining Dataset**: 510k mRNA sequences collected from NIH datasets across five species (human, rat, mouse, chicken, zebrafish)  
- **Evaluation Datasets**:
  1. **5′ UTR Tasks** (Expression Level and Translation Efficiency Prediction)  
     - HEK293T (14k sequences), PC3 (12k sequences), Muscle cells (1k sequences)  
  2. **CDS Tasks** (mRNA Stability and Protein Production Level Prediction)  
     - mRNA Stability Dataset (25k sequences)  
     - Protein Production Dataset (6k sequences)  

#### **4. Training Configuration**  
- **Model Architecture**:  
  - Based on the T5 encoder with a **data2vec** framework  
  - 4 **self-attention heads**, 4 hidden layers, and **256-dimensional embeddings**  
  - Total **10M parameters** (3M trainable parameters)  
- **Optimization Settings**:
  - Optimizer: AdamW (learning rate = 0.001, weight decay = 0.001)  
  - Batch size: 256  
  - Hardware: **4 NVIDIA RTX 4090 GPUs**  

#### **5. Model Performance**  
- **Achieved up to 14% improvement in Translation Efficiency (TE) and Expression Level (EL) prediction over the SOTA method (UTR-LM)**  
- **Demonstrated competitive performance against CodonBERT in mRNA stability and protein production level prediction**

   
 
<br/>
# Results  




#### **1. 경쟁 모델 비교 (Baseline Models)**
mRNA2vec의 성능을 검증하기 위해 **기존 최첨단(SOTA) 모델들과 비교**하였다. 비교 대상 모델은 다음과 같다:  

1. **CodonBERT** (Li et al., 2023) - 코돈 기반 mRNA 임베딩 모델  
2. **UTR-LM** (Chu et al., 2024) - 5′ UTR 서열을 활용한 번역 효율 예측 모델  
3. **RNA-BERT** (Akiyama & Sakakibara, 2022) - RNA 서열 분석을 위한 BERT 기반 모델  
4. **RNA-FM** (Chen et al., 2022) - RNA 구조 및 기능 예측을 위한 기초 모델  
5. **Cao-RF** (Cao et al., 2021) - 랜덤 포레스트 기반 5′ UTR 번역 효율 예측 모델  
6. **RiNALMo** (Penić et al., 2024) - 36M개의 비코딩 RNA 서열로 훈련된 대형 모델  

#### **2. 테스트 데이터 (Evaluation Datasets)**
모델 성능은 5′ UTR과 CDS 영역에서 각각 평가되었다.  

**(1) 5′ UTR 테스트 데이터** (번역 효율 및 발현 수준 예측)  
- HEK293T 세포 (14k 서열)  
- PC3 전립선 암 세포 (12k 서열)  
- 근육 세포 (1k 서열)  

**(2) CDS 테스트 데이터** (mRNA 안정성 및 단백질 생산성 예측)  
- **mRNA 안정성 데이터셋** (25k 서열)  
- **mRFP 단백질 생산성 데이터셋** (6k 서열)  

#### **3. 평가 메트릭 (Evaluation Metrics)**  
모델의 성능을 측정하기 위해 **스피어만 순위 상관계수(Spearman Rank Correlation, ρ)**를 사용하였다.  
스피어만 순위 상관계수는 두 변수 간의 **비선형 상관관계**를 측정하는 메트릭으로, 예측값과 실제값 사이의 일관성을 평가하는 데 적합하다.  

#### **4. 실험 결과 (Experimental Results)**  
**(1) 5′ UTR 영역에서의 성능 비교**  
- 번역 효율(TE) 및 발현 수준(EL) 예측에서 mRNA2vec이 기존 모델 대비 **최대 14% 향상**된 성능을 기록함  
- 특히 **HEK, PC3, Muscle 데이터셋에서 UTR-LM보다 TE에서 각각 13%, 12%, 14% 더 높은 성능을 보임**  
- EL 예측에서도 **UTR-LM 대비 6~31% 향상**  

**(2) CDS 영역에서의 성능 비교**  
- mRNA 안정성 예측에서 **CodonBERT(0.34) 대비 mRNA2vec(0.53) 성능 대폭 향상**  
- 단백질 생산 수준 예측에서는 CodonBERT와 **비슷한 성능**을 보였으나, CodonBERT는 E. coli 데이터로 훈련되었음에도 불구하고 mRNA2vec은 범용적으로 적용 가능함을 입증  

---



#### **1. Baseline Model Comparison**
To validate the performance of **mRNA2vec**, it was compared against **state-of-the-art (SOTA) models**, including:  

1. **CodonBERT** (Li et al., 2023) - Codon-based mRNA embedding model  
2. **UTR-LM** (Chu et al., 2024) - Translation efficiency prediction model for 5′ UTR sequences  
3. **RNA-BERT** (Akiyama & Sakakibara, 2022) - BERT-based model for RNA sequence analysis  
4. **RNA-FM** (Chen et al., 2022) - Foundational model for RNA structure and function prediction  
5. **Cao-RF** (Cao et al., 2021) - Random forest-based 5′ UTR translation efficiency model  
6. **RiNALMo** (Penić et al., 2024) - Large pre-trained model with 36M non-coding RNA sequences  

#### **2. Evaluation Datasets**
The model’s performance was evaluated on both 5′ UTR and CDS datasets.  

**(1) 5′ UTR Evaluation Data** (Translation Efficiency and Expression Level Prediction)  
- HEK293T cell line (14k sequences)  
- PC3 prostate cancer cell line (12k sequences)  
- Muscle cells (1k sequences)  

**(2) CDS Evaluation Data** (mRNA Stability and Protein Production Level Prediction)  
- **mRNA Stability Dataset** (25k sequences)  
- **mRFP Protein Production Dataset** (6k sequences)  

#### **3. Evaluation Metrics**
The model performance was measured using the **Spearman Rank Correlation (ρ)**.  
This metric assesses **nonlinear correlations** between predictions and actual values, making it well-suited for ranking-based evaluations.  

#### **4. Experimental Results**
**(1) Performance on 5′ UTR Tasks**  
- **Achieved up to 14% improvement** in **Translation Efficiency (TE) and Expression Level (EL) prediction** compared to previous models  
- Specifically, **outperformed UTR-LM in TE by 13%, 12%, and 14% on HEK, PC3, and Muscle datasets, respectively**  
- EL prediction also showed **6-31% improvement over UTR-LM**  

**(2) Performance on CDS Tasks**  
- **mRNA Stability prediction improved significantly**, with mRNA2vec achieving **0.53 Spearman correlation compared to CodonBERT’s 0.34**  
- For **protein production level prediction**, mRNA2vec performed comparably to CodonBERT, despite **not being explicitly trained on E. coli data**, suggesting broader generalization capabilities  



<br/>
# 예제  




#### **1. 학습 데이터 (Training Data)**
mRNA2vec 모델의 학습을 위해 NIH 데이터셋에서 **5종(인간, 쥐, 생쥐, 닭, 제브라피시)**의 mRNA 서열을 수집하였다.  
- **총 서열 개수**: **510,000개**  
- **서열 구성**: 5′ 비번역 영역(UTR)과 코딩 서열(CDS)을 포함  
- **데이터 처리**:
  - **5′ UTR**: 시작 코돈(AUG) 이전 서열을 102 뉴클레오타이드(34 토큰)로 제한  
  - **CDS**: 종료 코돈(UAG, UAA, UGA) 이전까지 포함  
  - **MFE (최소 자유 에너지) 및 SS (이차 구조) 값 추가**  

#### **2. 테스트 데이터 (Test Data)**
모델 평가를 위해 5′ UTR과 CDS 영역에서 각각 테스트 데이터를 설정하였다.  

**(1) 5′ UTR 테스트 데이터**  
- **목적**: 번역 효율(TE) 및 발현 수준(EL) 예측  
- **테스트 세트**:
  - HEK293T 세포 (14k 개 서열)  
  - PC3 전립선 암 세포 (12k 개 서열)  
  - 근육 세포 (1k 개 서열)  
- **입력 서열 예제**:  
  ```
  AUGGUGCCUUUACGUGAUGCUA...
  ```
- **출력 값 예제** (TE 예측):  
  ```
  0.85
  ```

**(2) CDS 테스트 데이터**  
- **목적**: mRNA 안정성과 단백질 생산 수준 예측  
- **테스트 세트**:
  - **mRNA 안정성 데이터셋** (25k 개 서열)  
  - **mRFP 단백질 생산 데이터셋** (6k 개 서열)  
- **입력 서열 예제**:  
  ```
  AUGGCUAAGUCUCGAAUGU...
  ```
- **출력 값 예제** (mRNA 안정성 예측):  
  ```
  0.65
  ```

#### **3. 테스트 태스크의 입력/출력 (Task Input/Output)**
- **입력**: 5′ UTR + CDS 서열 (토큰화된 형태)  
- **출력**: TE, EL, mRNA 안정성, 단백질 생산 수준 등  
- **예제**  
  - **입력 (5′ UTR + CDS 서열)**
    ```
    AUGGCUAAGUCUCGAAUGU...
    ```
  - **출력 (예측된 번역 효율)**
    ```
    0.85
    ```  

---



#### **1. Training Data**
The mRNA2vec model was trained using **mRNA sequences from five species** (human, rat, mouse, chicken, zebrafish) collected from the NIH dataset.  
- **Total sequences**: **510,000**  
- **Sequence composition**: Includes **5′ untranslated region (UTR) and coding sequence (CDS)**  
- **Data processing**:
  - **5′ UTR**: Limited to **102 nucleotides (34 tokens) before the start codon (AUG)**  
  - **CDS**: Includes sequence up to **the stop codon (UAG, UAA, UGA)**  
  - **MFE (Minimum Free Energy) and SS (Secondary Structure) values were included**  

#### **2. Test Data**
For evaluation, test datasets were created for both 5′ UTR and CDS tasks.  

**(1) 5′ UTR Test Data**  
- **Objective**: Predict **Translation Efficiency (TE) and Expression Level (EL)**  
- **Test Sets**:
  - HEK293T cell line (14k sequences)  
  - PC3 prostate cancer cell line (12k sequences)  
  - Muscle cell line (1k sequences)  
- **Example Input Sequence**:  
  ```
  AUGGUGCCUUUACGUGAUGCUA...
  ```
- **Example Output (TE Prediction)**:  
  ```
  0.85
  ```

**(2) CDS Test Data**  
- **Objective**: Predict **mRNA Stability and Protein Production Level**  
- **Test Sets**:
  - **mRNA Stability Dataset** (25k sequences)  
  - **mRFP Protein Production Dataset** (6k sequences)  
- **Example Input Sequence**:  
  ```
  AUGGCUAAGUCUCGAAUGU...
  ```
- **Example Output (mRNA Stability Prediction)**:  
  ```
  0.65
  ```

#### **3. Task Input/Output**
- **Input**: **5′ UTR + CDS sequence (tokenized format)**  
- **Output**: Predictions for **TE, EL, mRNA stability, protein production level**  
- **Example**  
  - **Input (5′ UTR + CDS Sequence)**
    ```
    AUGGCUAAGUCUCGAAUGU...
    ```
  - **Output (Predicted Translation Efficiency)**
    ```
    0.85
    ```  


<br/>  
# 요약   



이 연구에서는 5′ UTR과 CDS를 통합하여 mRNA 서열을 임베딩하는 **mRNA2vec** 모델을 제안하며, data2vec 기반의 **교사-학생(Self-Supervised Teacher-Student) 학습**과 **확률적 하드 마스킹**을 적용하였다. 실험 결과, **번역 효율(TE) 및 발현 수준(EL) 예측에서 기존 SOTA 모델 대비 최대 14% 향상**되었으며, **mRNA 안정성 및 단백질 생산성 예측에서도 CodonBERT와 경쟁력 있는 성능**을 보였다. 테스트 데이터로는 **HEK293T, PC3, Muscle, mRNA Stability, mRFP Expression 데이터셋**을 활용했으며, 입력은 **5′ UTR+CDS 서열**이고, 출력은 **번역 효율, 발현 수준, 안정성, 단백질 생산성 등의 예측값**이다.  

---


This study proposes **mRNA2vec**, a model that embeds mRNA sequences by integrating **5′ UTR and CDS** while employing **self-supervised teacher-student learning with probabilistic hard masking** based on data2vec. Experimental results show **up to a 14% improvement in Translation Efficiency (TE) and Expression Level (EL) prediction** over previous SOTA models, along with **competitive performance in mRNA stability and protein production level prediction** compared to CodonBERT. The test datasets include **HEK293T, PC3, Muscle, mRNA Stability, and mRFP Expression datasets**, where the **input is a 5′ UTR + CDS sequence** and the output consists of **predicted TE, EL, stability, and protein production levels**.


<br/>  
# 기타  







#### **1. 모델 개요 및 학습 과정 다이어그램 (Figure 1)**  
- **개요**: mRNA2vec의 **사전 학습 과정**을 나타내는 다이어그램  
- **주요 구성 요소**:
  - **입력 서열**: 5′ UTR과 CDS가 결합된 형태  
  - **마스킹 전략**: 특정 위치(5′ UTR의 끝과 CDS의 시작)에 **확률적 하드 마스킹 적용**  
  - **교사-학생 모델 (Teacher-Student Framework)**:  
    - **학생 모델(Student Model)**: 마스킹된 서열을 입력으로 받아 특징을 학습  
    - **교사 모델(Teacher Model)**: 마스킹되지 않은 서열을 활용하여 학생 모델이 예측한 값을 학습 목표로 제공  
  - **보조 학습 과제(Auxiliary Tasks)**:  
    - **MFE(최소 자유 에너지) 예측**  
    - **SS(이차 구조) 분류**  

---

#### **2. 번역 효율(TE) 및 발현 수준(EL) 비교 그래프 (Figure 2)**  
- **개요**: 사전 학습 전략(data2vec vs. T5 vs. 미훈련 모델) 비교  
- **실험 결과**:  
  - **data2vec 기반의 mRNA2vec이 T5 및 미훈련 모델 대비 TE 및 EL 예측에서 더 높은 성능**을 보임  
  - **특히 5′ UTR 데이터셋(HEK, PC3, Muscle)에서 성능 차이가 명확**하게 나타남  

---

#### **3. 사전 학습 기법 비교 테이블 (Table 2)**  
- **개요**: 마스킹 전략과 보조 과제(MFE, SS) 적용 여부에 따른 성능 비교  
- **실험 결과**:  
  - **MFE와 SS를 추가한 모델이 기본 모델(data2vec만 적용한 경우)보다 성능이 향상됨**  
  - **Muscle 데이터셋에서는 TE 예측 성능이 0.550 → 0.573으로 증가**  

---

#### **4. UTR-LM 대비 성능 비교 그래프 (Figure 3)**  
- **개요**: 기존 5′ UTR 기반 모델(UTR-LM)과의 비교  
- **실험 결과**:  
  - **mRNA2vec이 UTR-LM 대비 번역 효율(TE)과 발현 수준(EL)에서 더 높은 성능을 기록**  
  - **특히 MFE 및 SS 보조 학습 과제가 추가될 경우 더욱 성능이 개선됨**  

---

#### **5. CDS 기반 모델 성능 비교 그래프 (Figure 4)**  
- **개요**: mRNA 안정성과 단백질 생산 수준 예측에서 기존 모델과의 비교  
- **실험 결과**:  
  - **mRNA 안정성 예측에서는 CodonBERT(0.34) 대비 mRNA2vec(0.53) 성능이 대폭 향상**  
  - **단백질 생산 수준 예측에서는 CodonBERT와 유사한 성능을 보였음**  

---

#### **6. 입력 서열 길이에 따른 성능 변화 그래프 (Figure 5)**  
- **개요**: 5′ UTR 입력 서열 길이가 TE 예측 성능에 미치는 영향  
- **실험 결과**:  
  - **HEK, PC3, Muscle 데이터셋에서 12~18개의 토큰(뉴클레오타이드 36~54개)이 최적의 길이로 나타남**  
  - **전체 서열을 입력했을 때보다 일부 서열을 선택하는 것이 더 좋은 성능을 보임**  

---



#### **1. Model Overview and Training Process Diagram (Figure 1)**  
- **Overview**: Diagram illustrating the **pretraining process of mRNA2vec**  
- **Key Components**:
  - **Input Sequence**: Combined **5′ UTR and CDS** sequence  
  - **Masking Strategy**: **Probabilistic hard masking** applied to specific positions (end of 5′ UTR and start of CDS)  
  - **Teacher-Student Model Framework**:
    - **Student Model**: Learns features from the masked sequence  
    - **Teacher Model**: Provides learning targets based on the unmasked sequence  
  - **Auxiliary Tasks**:
    - **Minimum Free Energy (MFE) Prediction**  
    - **Secondary Structure (SS) Classification**  

---

#### **2. Translation Efficiency (TE) and Expression Level (EL) Comparison Graph (Figure 2)**  
- **Overview**: Performance comparison of different pretraining strategies (data2vec vs. T5 vs. untrained model)  
- **Results**:
  - **mRNA2vec using data2vec outperforms T5 and untrained models in TE and EL prediction**  
  - **The performance gap is particularly noticeable on 5′ UTR datasets (HEK, PC3, Muscle)**  

---

#### **3. Pretraining Strategy Comparison Table (Table 2)**  
- **Overview**: Performance comparison based on different masking strategies and auxiliary tasks (MFE, SS)  
- **Results**:
  - **Adding MFE and SS tasks improves performance over the base data2vec model**  
  - **For the Muscle dataset, TE prediction performance increased from 0.550 → 0.573**  

---

#### **4. Performance Comparison Against UTR-LM (Figure 3)**  
- **Overview**: Performance comparison between **mRNA2vec and UTR-LM**  
- **Results**:
  - **mRNA2vec achieves higher performance than UTR-LM in both TE and EL prediction**  
  - **The inclusion of MFE and SS auxiliary tasks further enhances performance**  

---

#### **5. Performance Comparison on CDS-Based Tasks (Figure 4)**  
- **Overview**: Performance comparison for **mRNA stability and protein production level prediction**  
- **Results**:
  - **mRNA2vec significantly outperforms CodonBERT (0.34 → 0.53) in mRNA stability prediction**  
  - **For protein production level prediction, mRNA2vec performs comparably to CodonBERT**  

---

#### **6. Effect of Input Sequence Length on Performance (Figure 5)**  
- **Overview**: The impact of **5′ UTR sequence length on TE prediction performance**  
- **Results**:
  - **The optimal sequence length is found to be 12-18 tokens (36-54 nucleotides) for HEK, PC3, and Muscle datasets**  
  - **Using the full sequence results in lower performance compared to selecting a subset of the sequence**  


<br/>
# refer format:     




@article{zhang2025mrna2vec,
  author    = {Honggen Zhang and Xiangrui Gao and June Zhang and Lipeng Lai},
  title     = {mRNA2vec: mRNA Embedding with Language Model in the 5′ UTR-CDS for mRNA Design},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  volume    = {39},
  number    = {1},
  pages     = {1--12},
  publisher = {Association for the Advancement of Artificial Intelligence},
  url       = {https://www.aaai.org},
}




Honggen Zhang, Xiangrui Gao, June Zhang, and Lipeng Lai. “mRNA2vec: mRNA Embedding with Language Model in the 5′ UTR-CDS for mRNA Design.” Proceedings of the AAAI Conference on Artificial Intelligence 39, no. 1 (2025): 1–12. Association for the Advancement of Artificial Intelligence.





