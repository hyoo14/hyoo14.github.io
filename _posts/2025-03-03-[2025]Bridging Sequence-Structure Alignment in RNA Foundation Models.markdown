---
layout: post
title:  "[2025]Bridging Sequence-Structure Alignment in RNA Foundation Models"  
date:   2025-03-03 14:03:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


rna 이차구조 예측 기반으로 트레인?한 LM    
텍스트로 표현된 구조를 학습에 포함한듯  


짧은 요약(Abstract) :    



기존 RNA 기반 기초 모델(FM)은 RNA 서열과 구조 간 정렬(sequence-structure alignment)을 효과적으로 수행하지 못해, 서열과 구조 간 유전자 정보 흐름이 제한되었다. 본 연구에서는 RNA의 이차 구조를 기반으로 서열과 구조 간 정렬을 가능하게 하는 새로운 RNA FM **OmniGenome**을 제안한다. OmniGenome은 유연한 RNA 모델링 패러다임을 적용하여, 서열 또는 구조를 입력과 출력으로 자유롭게 활용할 수 있도록 설계되었다. 이를 평가하기 위해 RNA 디자인과 제로샷 이차 구조 예측(secondary structure prediction) 실험을 수행하였으며, **EternaV2 벤치마크**에서 기존 FM이 3%의 퍼즐만 해결한 것과 달리 OmniGenome은 **74%의 퍼즐을 해결**하여 뛰어난 성능을 보였다. 또한, 4개의 종합적인 **인 실리코(in silico) 유전체 모델링 벤치마크**를 통해 RNA 및 DNA 유전체 분석에서 최첨단 성능을 달성하였으며, DNA 유전체에 대한 추가 학습 없이도 높은 성능을 유지하였다.  

---


The alignment between RNA sequences and structures in foundation models (FMs) has yet to be thoroughly investigated. Existing FMs have struggled to establish sequence-structure alignment, hindering the free flow of genomic information between RNA sequences and structures. In this study, we introduce **OmniGenome**, an RNA FM trained to align RNA sequences with respect to secondary structures based on structure-contextualized modeling. The alignment enables free and bidirectional mappings between sequences and structures by utilizing a flexible RNA modeling paradigm that supports versatile input and output modalities, i.e., sequence and/or structure as input/output. We implement RNA design and zero-shot secondary structure prediction as case studies to evaluate the **Seq2Str and Str2Seq** mapping capacity of OmniGenome. Results on the **EternaV2 benchmark** show that **OmniGenome solved 74% of puzzles**, whereas existing FMs only solved up to **3%** of the puzzles due to the oversight of sequence-structure alignment. We leverage four comprehensive **in-silico genome modeling benchmarks** to evaluate performance across a diverse set of genome downstream tasks, where the results show that OmniGenome achieves state-of-the-art performance on RNA and DNA benchmarks, even without any training on DNA genomes.



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




본 연구에서는 RNA 서열과 이차 구조 간 정렬(sequence-structure alignment)을 실현하는 새로운 RNA 기반 기초 모델 **OmniGenome**을 제안하였다. OmniGenome은 다양한 유전체 모델링 작업에서 높은 성능을 보이며, 기존 RNA FMs과 차별화되는 핵심 요소는 다음과 같다.  

#### **1. 제안 모델 및 학습 목표**  
OmniGenome은 **Str2Seq (Structure-to-Sequence)** 및 **Seq2Str (Sequence-to-Structure)** 매핑을 모두 수행할 수 있도록 설계되었다. 이를 위해 **Masked RNA Language Modeling (MRLM)**을 포함한 세 가지 주요 사전 학습 목표를 설정하였다.  
- **LStr2Seq**: 이차 구조를 기반으로 마스킹된 RNA 서열을 복원하는 목표  
- **LSeq2Str**: 주어진 RNA 서열로부터 이차 구조를 예측하는 목표  
- **LMRLM**: RNA 언어 모델링을 통해 전체 서열 이해도를 높이는 목표  

이러한 학습 목표를 통해 RNA의 서열과 구조 간 정보 흐름을 정렬하고, 구조적 맥락을 고려한 서열 및 구조 예측을 가능하게 하였다.  

---

#### **2. 아키텍처 및 모델 설계**  
OmniGenome은 **Transformer 기반의 인코더 아키텍처**를 채택하며, 두 가지 변형 모델을 제안하였다.  
- **OmniGenome52M**: 52M(5200만) 개의 파라미터  
- **OmniGenome186M**: 186M(1억8600만) 개의 파라미터  

모델은 **양방향 Multi-Head Attention**을 활용하며, 최신 아키텍처인 Mamba 및 Hyena보다 RNA 서열을 이해하는 데 더 적합한 것으로 나타났다.  

- 학습률(learning rate): **5 × 10⁻⁵**  
- 옵티마이저: **AdamW (β₁=0.9, β₂=0.999)**  
- 배치 크기(batch size): **2048**  
- 드롭아웃 없음, **로터리 위치 임베딩(rotary position embeddings)** 적용  

모델 학습에는 **8개의 NVIDIA RTX 4090 GPU**를 사용하였으며, OmniGenome52M과 OmniGenome186M의 사전 학습은 각각 약 **1주, 3주 소요**되었다.  

---

#### **3. 트레이닝 데이터 및 전처리**  
OmniGenome은 다양한 RNA 서열과 구조 데이터를 활용하여 학습되었으며, 특히 **OneKP 데이터베이스**를 주요 학습 데이터로 사용하였다.  

- **OneKP 데이터베이스**: 1124개 식물종의 대규모 RNA 전사체 데이터 포함  
- **데이터 전처리**:  
  - **중복 제거** (duplicate filtering)  
  - **50개 뉴클레오타이드 미만의 서열 제거**  
  - **ViennaRNA 소프트웨어를 사용하여 이차 구조 생성**  
  - **cd-hit-est 및 BLAST를 사용한 중복 데이터 필터링**  

이를 통해 OmniGenome은 **고품질의 RNA 서열과 구조 데이터**를 학습할 수 있었으며, Seq2Str 및 Str2Seq 예측 성능을 극대화하였다.  

---

#### **4. 벤치마크 및 성능 평가**  
OmniGenome의 성능을 평가하기 위해 네 가지 주요 유전체 벤치마크를 활용하였다.  
1. **RNA Genomics Benchmark (RGB)**: RNA 분해 속도 예측, mRNA 설계 등  
2. **Plant Genomics Benchmark (PGB)**: DNA 기반 유전체 분석, 다중 종 간 일반화 성능 평가  
3. **Genomics Benchmark (GB)** 및 **Genomics Understanding Evaluation (GUE)**: 비식물성 유전체 데이터셋에 대한 일반화 성능 평가  

이러한 벤치마크 실험을 통해 **OmniGenome은 기존 모델보다 최대 35% 성능 향상**을 기록하며, **RNA 및 DNA 기반 유전체 분석에서 최첨단 성능을 달성**하였다.  

---



In this study, we propose **OmniGenome**, a novel RNA foundation model designed to achieve sequence-structure alignment. OmniGenome demonstrates superior performance across various genomics tasks and introduces key innovations distinguishing it from existing RNA foundation models.  

---

#### **1. Proposed Model and Training Objectives**  
OmniGenome is designed to handle both **Structure-to-Sequence (Str2Seq)** and **Sequence-to-Structure (Seq2Str)** mapping. To achieve this, we introduce three key **pre-training objectives**, including **Masked RNA Language Modeling (MRLM)**:  
- **LStr2Seq**: Predict masked RNA sequences given secondary structure context  
- **LSeq2Str**: Predict secondary structures from RNA sequences  
- **LMRLM**: Improve sequence understanding through masked language modeling  

These training objectives ensure proper information flow between RNA sequences and structures, enabling structure-aware sequence prediction and vice versa.  

---

#### **2. Model Architecture**  
OmniGenome employs a **Transformer-based encoder architecture**, with two model variants:  
- **OmniGenome52M**: 52 million parameters  
- **OmniGenome186M**: 186 million parameters  

The model adopts **bidirectional Multi-Head Attention**, outperforming newer architectures such as Mamba and Hyena in RNA sequence understanding.  

- Learning rate: **5 × 10⁻⁵**  
- Optimizer: **AdamW (β₁=0.9, β₂=0.999)**  
- Batch size: **2048**  
- No dropout applied; **rotary position embeddings** incorporated  

Training was conducted using **8 NVIDIA RTX 4090 GPUs**, taking approximately **1 week for OmniGenome52M and 3 weeks for OmniGenome186M**.  

---

#### **3. Training Data and Preprocessing**  
OmniGenome was trained on **high-quality RNA sequences and secondary structure data**, primarily from the **OneKP database**.  

- **OneKP dataset**: Large-scale transcriptome data from **1,124 plant species**  
- **Preprocessing pipeline**:  
  - **Duplicate removal**  
  - **Filtering sequences shorter than 50 nucleotides**  
  - **Generating secondary structures using ViennaRNA software**  
  - **Removing redundant sequences using cd-hit-est and BLAST**  

This rigorous preprocessing ensured that OmniGenome learned from **high-quality RNA sequence-structure data**, enhancing **Seq2Str and Str2Seq prediction capabilities**.  

---

#### **4. Benchmarks and Performance Evaluation**  
OmniGenome was evaluated using four **comprehensive genome modeling benchmarks**:  
1. **RNA Genomics Benchmark (RGB)**: Tasks such as RNA degradation rate prediction and mRNA design  
2. **Plant Genomics Benchmark (PGB)**: DNA-based genome analysis and multi-species generalization  
3. **Genomics Benchmark (GB)** and **Genomics Understanding Evaluation (GUE)**: Generalization performance on non-plant genomes  

Experimental results demonstrated that **OmniGenome outperforms existing models by up to 35%**, achieving **state-of-the-art performance in both RNA and DNA genomics tasks**.

   
 
<br/>
# Results  





본 연구에서는 OmniGenome의 성능을 평가하기 위해 다양한 **경쟁 모델**, **테스트 데이터셋**, **평가 메트릭**을 사용하여 비교 실험을 수행하였다. OmniGenome은 RNA 및 DNA 기반 유전체 분석에서 기존 기초 모델(FM) 대비 **최대 35% 성능 향상**을 기록하며, 최첨단 성능을 달성하였다.  

---

### **1. 경쟁 모델 (Baseline Models)**  
OmniGenome의 성능을 비교하기 위해 다양한 RNA 및 DNA 기초 모델과 비교 실험을 수행하였다. 경쟁 모델은 크게 RNA 및 DNA FMs으로 나뉜다.  

#### **(1) RNA Foundation Models (RNA FMs)**
- **RNA-FM**  
- **RNA-MSM**  
- **SpliceBERT**  
- **3UTRBERT**  

위 모델들은 RNA 시퀀스 분석을 수행하는 기초 모델들이지만, 대부분이 서열 기반 예측만 가능하며 **서열-구조 정렬(sequence-structure alignment)이 부재**하여 Str2Seq 및 Seq2Str 작업에서 한계를 보인다.  

#### **(2) DNA Foundation Models (DNA FMs)**
- **DNABERT2**  
- **HyenaDNA**  
- **Caduceus**  
- **NT-V2**  
- **Agro-NT**  

DNA FMs는 유전체 모델링을 위한 대규모 사전학습 모델이지만, **RNA 서열 및 구조 분석에는 적합하지 않으며**, Str2Seq 및 Seq2Str 작업에서 낮은 성능을 보였다.  

---

### **2. 테스트 데이터 및 벤치마크**  
OmniGenome의 성능을 평가하기 위해 **4가지 주요 유전체 벤치마크 데이터셋**을 사용하였다.  

1. **RNA Genomics Benchmark (RGB)**  
   - RNA 기반 유전체 분석 평가 (예: RNA 분해율 예측, RNA 디자인)  
2. **Plant Genomics Benchmark (PGB)**  
   - DNA 기반 유전체 분석 및 다중 종 간 일반화 평가  
3. **Genomics Benchmark (GB)**  
   - RNA 및 DNA 시퀀스 예측 및 구조 분석  
4. **Genomics Understanding Evaluation (GUE)**  
   - 비식물성 유전체 데이터셋에 대한 일반화 성능 평가  

---

### **3. 평가 메트릭 (Evaluation Metrics)**  
다양한 유전체 분석 작업에 대해 OmniGenome의 성능을 평가하기 위해 **F1-score, RMSE, Accuracy 등**의 메트릭을 사용하였다.  

- **RNA 디자인 (Str2Seq 예측 정확도)**
  - **벤치마크**: EternaV2  
  - **메트릭**: 퍼즐 해결률(Accuracy)  
  - **결과**: **OmniGenome(74%) vs 기존 FMs(최대 3%)**  

- **제로샷 이차 구조 예측 (Seq2Str 예측 성능)**
  - **벤치마크**: Archive2, bpRNA, Stralign  
  - **메트릭**: Macro-F1 Score  
  - **결과**: **OmniGenome(최대 74.85%) vs 기존 RNA FMs(최대 66%)**  

- **DNA 유전체 분석 (Plant Genomics Benchmark, PGB)**
  - **평가 작업**: 유전자 발현 예측, 염색질 접근성 예측, 전사 개시 예측  
  - **결과**: **OmniGenome이 Agro-NT, NT-V2, Caduceus 대비 최대 35% 향상된 성능**  

---

### **4. 종합 결과**  
- OmniGenome은 **기존 RNA 및 DNA FMs보다 뛰어난 성능을 달성**하였으며, 특히 **서열-구조 정렬(Sequence-Structure Alignment)**을 통해 Str2Seq 및 Seq2Str 예측 성능을 크게 향상시켰다.  
- RNA 디자인 작업에서 OmniGenome은 기존 FMs 대비 **20배 이상 높은 정확도**를 기록하였다.  
- DNA 유전체 분석에서도 OmniGenome은 기존 모델보다 **일관된 성능 향상**을 보였으며, 다양한 유전체 벤치마크에서 최첨단 성능을 기록하였다.  

---


We evaluated **OmniGenome** against various **competitive models**, **test datasets**, and **evaluation metrics**. OmniGenome outperformed existing RNA and DNA foundation models (FMs) by **up to 35%** in genomics tasks, setting a new state-of-the-art performance standard.  

---

### **1. Competitive Models (Baseline Comparisons)**  
We compared OmniGenome with various RNA and DNA **foundation models (FMs)**, divided into two categories.  

#### **(1) RNA Foundation Models (RNA FMs)**
- **RNA-FM**  
- **RNA-MSM**  
- **SpliceBERT**  
- **3UTRBERT**  

These models primarily focus on sequence-based prediction but **lack sequence-structure alignment**, making them less effective in Str2Seq and Seq2Str tasks.  

#### **(2) DNA Foundation Models (DNA FMs)**
- **DNABERT2**  
- **HyenaDNA**  
- **Caduceus**  
- **NT-V2**  
- **Agro-NT**  

While these models perform well in DNA-based tasks, they **struggle with RNA sequence and structure alignment**, resulting in lower performance in Str2Seq and Seq2Str tasks.  

---

### **2. Test Datasets and Benchmarks**  
OmniGenome was evaluated using **four major genomics benchmarks**:  

1. **RNA Genomics Benchmark (RGB)**  
   - Evaluates RNA-based genomics tasks (e.g., RNA degradation rate prediction, RNA design)  
2. **Plant Genomics Benchmark (PGB)**  
   - Evaluates DNA-based genomics tasks and cross-species generalization  
3. **Genomics Benchmark (GB)**  
   - Evaluates RNA and DNA sequence and structure modeling  
4. **Genomics Understanding Evaluation (GUE)**  
   - Measures generalization performance on non-plant genomes  

---

### **3. Evaluation Metrics**  
We used **F1-score, RMSE, and Accuracy** to evaluate OmniGenome’s performance across various tasks.  

- **RNA Design (Str2Seq Prediction Accuracy)**
  - **Benchmark**: EternaV2  
  - **Metric**: Puzzle-solving Accuracy  
  - **Results**: **OmniGenome (74%) vs Existing FMs (Max 3%)**  

- **Zero-Shot Secondary Structure Prediction (Seq2Str Performance)**
  - **Benchmarks**: Archive2, bpRNA, Stralign  
  - **Metric**: Macro-F1 Score  
  - **Results**: **OmniGenome (Up to 74.85%) vs Existing RNA FMs (Max 66%)**  

- **DNA Genomics Tasks (Plant Genomics Benchmark, PGB)**
  - **Tasks**: Gene Expression Prediction, Chromatin Accessibility, Transcription Initiation Prediction  
  - **Results**: **OmniGenome outperformed Agro-NT, NT-V2, and Caduceus by up to 35%**  

---

### **4. Summary of Results**  
- **OmniGenome outperformed existing RNA and DNA FMs across all benchmarks**, demonstrating the effectiveness of sequence-structure alignment.  
- OmniGenome achieved **20× higher accuracy than existing FMs in RNA design tasks**.  
- OmniGenome also **consistently outperformed DNA models** in genome-wide analysis, achieving **state-of-the-art performance** across diverse genomics benchmarks.  



<br/>
# 예제  




OmniGenome의 성능을 검증하기 위해 **트레이닝 및 테스트 데이터셋**을 구성하고, 다양한 유전체 분석 작업을 수행하였다. 실험에는 RNA 및 DNA 기반 데이터를 포함한 **대규모 유전체 데이터셋**이 사용되었으며, **각 벤치마크별 대표적인 예제 데이터**를 포함하여 설명한다.  

---

### **1. 트레이닝 데이터 (Training Data)**  

OmniGenome의 사전 학습(Pre-training)에는 **다양한 RNA 및 DNA 데이터셋**이 사용되었다.  
주요 데이터셋은 **OneKP** 데이터베이스를 기반으로 하며, 다음과 같은 전처리 과정을 거쳤다.  

- **데이터셋: OneKP**  
  - 1124개 식물종의 대규모 RNA 전사체 데이터 포함  
  - 중복 제거 및 품질 관리 수행  

- **데이터 필터링 및 정제**  
  - **50 뉴클레오타이드 미만의 짧은 서열 제거**  
  - **ViennaRNA를 사용하여 이차 구조 생성 및 정렬**  
  - **cd-hit-est 및 BLAST를 사용하여 중복 및 진화적으로 보존된 서열 제거**  

이러한 과정을 거쳐 **RNA 서열과 이차 구조가 정렬된 데이터셋**이 구성되었으며, OmniGenome의 Seq2Str 및 Str2Seq 성능을 극대화할 수 있도록 하였다.  

---

### **2. 테스트 데이터 (Test Data & Benchmark Datasets)**  

OmniGenome의 성능을 측정하기 위해 다양한 **RNA 및 DNA 벤치마크 데이터셋**을 활용하였다.  
각 데이터셋에는 대표적인 예제(RNA 서열 및 이차 구조 정보 포함)가 포함되며, 주요 벤치마크는 다음과 같다.  

#### **(1) RNA Genomics Benchmark (RGB)**
- **목적**: RNA 기반 유전체 분석 (예: RNA 분해율 예측, RNA 디자인)  
- **데이터 예제**:
  ```
  >Example_RNA_1
  AUGCUAGCUAGCUAGCUA
  (((....)))....(((..)))
  ```
  - RNA 서열과 해당 이차 구조의 정렬 예제  

#### **(2) Plant Genomics Benchmark (PGB)**
- **목적**: DNA 기반 유전체 분석 및 다중 종 간 일반화 성능 평가  
- **데이터 예제**:
  ```
  >Example_DNA_1
  ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
  ```
  - 다양한 식물 유전체 데이터를 활용한 DNA 분석  

#### **(3) EternaV2 RNA Design Dataset**
- **목적**: RNA 디자인(서열-구조 변환 Str2Seq) 성능 평가  
- **데이터 예제**:
  ```
  Target Structure: (((....)))....(((..)))
  Generated Sequence: AUGCUAGCUAGCUAGCUA
  ```
  - 주어진 이차 구조를 기반으로 RNA 서열을 설계하는 문제  

#### **(4) Zero-shot Secondary Structure Prediction (Seq2Str)**
- **목적**: RNA 서열을 기반으로 이차 구조 예측(Seq2Str) 성능 측정  
- **데이터 예제**:
  ```
  Input Sequence: AUGCUAGCUAGCUAGCUA
  Predicted Structure: (((....)))....(((..)))
  ```

---

### **3. 실험 환경 (Experimental Setup)**  
OmniGenome의 훈련 및 평가를 위한 하드웨어 및 소프트웨어 환경은 다음과 같다.  

- **하드웨어**:  
  - NVIDIA RTX 4090 GPU × 8개  
  - 메모리: 512GB RAM  

- **소프트웨어**:  
  - PyTorch 2.0.0  
  - Transformers 4.37.1  
  - ViennaRNA 2.6.4  
  - 데이터 전처리: cd-hit-est, BLAST  

이러한 실험 환경을 통해 **대규모 RNA 및 DNA 데이터셋을 효과적으로 학습**하고 평가할 수 있었다.  

---



To evaluate **OmniGenome**, we designed a **comprehensive training and test dataset** using large-scale genomic data. The **benchmark datasets include both RNA and DNA** sequences and cover a variety of genomics tasks.  

---

### **1. Training Data**  

OmniGenome was pre-trained on **diverse RNA and DNA datasets**, primarily based on the **OneKP** database.  
The following preprocessing steps were applied to ensure high-quality data alignment.  

- **Dataset: OneKP**  
  - Includes **transcriptome data from 1,124 plant species**  
  - **Duplicate removal and quality control applied**  

- **Data Filtering and Processing**  
  - **Sequences shorter than 50 nucleotides were removed**  
  - **Secondary structures were generated using ViennaRNA**  
  - **cd-hit-est and BLAST were used to filter redundant sequences**  

Through these processes, **high-quality RNA sequence-structure aligned data** was constructed, enhancing OmniGenome’s **Seq2Str and Str2Seq capabilities**.  

---

### **2. Test Data (Benchmark Datasets & Example Data)**  

OmniGenome was evaluated on multiple **RNA and DNA benchmark datasets**.  
Each dataset includes **representative examples (RNA sequences and secondary structures)**, as listed below.  

#### **(1) RNA Genomics Benchmark (RGB)**
- **Purpose**: Evaluates RNA-based genomics tasks (e.g., RNA degradation rate prediction, RNA design)  
- **Example Data**:
  ```
  >Example_RNA_1
  AUGCUAGCUAGCUAGCUA
  (((....)))....(((..)))
  ```
  - Example of RNA sequence and corresponding secondary structure  

#### **(2) Plant Genomics Benchmark (PGB)**
- **Purpose**: Evaluates DNA-based genomics tasks and multi-species generalization  
- **Example Data**:
  ```
  >Example_DNA_1
  ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
  ```
  - Example of DNA sequence used in plant genomics  

#### **(3) EternaV2 RNA Design Dataset**
- **Purpose**: Evaluates RNA design performance (Str2Seq prediction)  
- **Example Data**:
  ```
  Target Structure: (((....)))....(((..)))
  Generated Sequence: AUGCUAGCUAGCUAGCUA
  ```
  - Task involves generating RNA sequences that match the given secondary structure  

#### **(4) Zero-shot Secondary Structure Prediction (Seq2Str)**
- **Purpose**: Evaluates RNA secondary structure prediction performance (Seq2Str)  
- **Example Data**:
  ```
  Input Sequence: AUGCUAGCUAGCUAGCUA
  Predicted Structure: (((....)))....(((..)))
  ```

---

### **3. Experimental Setup**  

To train and evaluate OmniGenome, we used the following **hardware and software environment**:  

- **Hardware**:  
  - **NVIDIA RTX 4090 GPUs × 8**  
  - **512GB RAM**  

- **Software**:  
  - **PyTorch 2.0.0**  
  - **Transformers 4.37.1**  
  - **ViennaRNA 2.6.4**  
  - **Data preprocessing: cd-hit-est, BLAST**  

With this setup, we efficiently **trained and evaluated OmniGenome on large-scale RNA and DNA datasets**.


<br/>  
# 요약   



OmniGenome의 성능을 검증하기 위해 **트레이닝 및 테스트 데이터셋**을 구성하고, 다양한 유전체 분석 작업을 수행하였다. 실험에는 RNA 및 DNA 기반 데이터를 포함한 **대규모 유전체 데이터셋**이 사용되었으며, **각 벤치마크별 대표적인 예제 데이터**를 포함하여 설명한다.  

---

### **1. 트레이닝 데이터 (Training Data)**  

OmniGenome의 사전 학습(Pre-training)에는 **다양한 RNA 및 DNA 데이터셋**이 사용되었다.  
주요 데이터셋은 **OneKP** 데이터베이스를 기반으로 하며, 다음과 같은 전처리 과정을 거쳤다.  

- **데이터셋: OneKP**  
  - 1124개 식물종의 대규모 RNA 전사체 데이터 포함  
  - 중복 제거 및 품질 관리 수행  

- **데이터 필터링 및 정제**  
  - **50 뉴클레오타이드 미만의 짧은 서열 제거**  
  - **ViennaRNA를 사용하여 이차 구조 생성 및 정렬**  
  - **cd-hit-est 및 BLAST를 사용하여 중복 및 진화적으로 보존된 서열 제거**  

이러한 과정을 거쳐 **RNA 서열과 이차 구조가 정렬된 데이터셋**이 구성되었으며, OmniGenome의 Seq2Str 및 Str2Seq 성능을 극대화할 수 있도록 하였다.  

---

### **2. 테스트 데이터 (Test Data & Benchmark Datasets)**  

OmniGenome의 성능을 측정하기 위해 다양한 **RNA 및 DNA 벤치마크 데이터셋**을 활용하였다.  
각 데이터셋에는 대표적인 예제(RNA 서열 및 이차 구조 정보 포함)가 포함되며, 주요 벤치마크는 다음과 같다.  

#### **(1) RNA Genomics Benchmark (RGB)**
- **목적**: RNA 기반 유전체 분석 (예: RNA 분해율 예측, RNA 디자인)  
- **데이터 예제**:
  ```
  >Example_RNA_1
  AUGCUAGCUAGCUAGCUA
  (((....)))....(((..)))
  ```
  - RNA 서열과 해당 이차 구조의 정렬 예제  

#### **(2) Plant Genomics Benchmark (PGB)**
- **목적**: DNA 기반 유전체 분석 및 다중 종 간 일반화 성능 평가  
- **데이터 예제**:
  ```
  >Example_DNA_1
  ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
  ```
  - 다양한 식물 유전체 데이터를 활용한 DNA 분석  

#### **(3) EternaV2 RNA Design Dataset**
- **목적**: RNA 디자인(서열-구조 변환 Str2Seq) 성능 평가  
- **데이터 예제**:
  ```
  Target Structure: (((....)))....(((..)))
  Generated Sequence: AUGCUAGCUAGCUAGCUA
  ```
  - 주어진 이차 구조를 기반으로 RNA 서열을 설계하는 문제  

#### **(4) Zero-shot Secondary Structure Prediction (Seq2Str)**
- **목적**: RNA 서열을 기반으로 이차 구조 예측(Seq2Str) 성능 측정  
- **데이터 예제**:
  ```
  Input Sequence: AUGCUAGCUAGCUAGCUA
  Predicted Structure: (((....)))....(((..)))
  ```

---

### **3. 실험 환경 (Experimental Setup)**  
OmniGenome의 훈련 및 평가를 위한 하드웨어 및 소프트웨어 환경은 다음과 같다.  

- **하드웨어**:  
  - NVIDIA RTX 4090 GPU × 8개  
  - 메모리: 512GB RAM  

- **소프트웨어**:  
  - PyTorch 2.0.0  
  - Transformers 4.37.1  
  - ViennaRNA 2.6.4  
  - 데이터 전처리: cd-hit-est, BLAST  

이러한 실험 환경을 통해 **대규모 RNA 및 DNA 데이터셋을 효과적으로 학습**하고 평가할 수 있었다.  

---



To evaluate **OmniGenome**, we designed a **comprehensive training and test dataset** using large-scale genomic data. The **benchmark datasets include both RNA and DNA** sequences and cover a variety of genomics tasks.  

---

### **1. Training Data**  

OmniGenome was pre-trained on **diverse RNA and DNA datasets**, primarily based on the **OneKP** database.  
The following preprocessing steps were applied to ensure high-quality data alignment.  

- **Dataset: OneKP**  
  - Includes **transcriptome data from 1,124 plant species**  
  - **Duplicate removal and quality control applied**  

- **Data Filtering and Processing**  
  - **Sequences shorter than 50 nucleotides were removed**  
  - **Secondary structures were generated using ViennaRNA**  
  - **cd-hit-est and BLAST were used to filter redundant sequences**  

Through these processes, **high-quality RNA sequence-structure aligned data** was constructed, enhancing OmniGenome’s **Seq2Str and Str2Seq capabilities**.  

---

### **2. Test Data (Benchmark Datasets & Example Data)**  

OmniGenome was evaluated on multiple **RNA and DNA benchmark datasets**.  
Each dataset includes **representative examples (RNA sequences and secondary structures)**, as listed below.  

#### **(1) RNA Genomics Benchmark (RGB)**
- **Purpose**: Evaluates RNA-based genomics tasks (e.g., RNA degradation rate prediction, RNA design)  
- **Example Data**:
  ```
  >Example_RNA_1
  AUGCUAGCUAGCUAGCUA
  (((....)))....(((..)))
  ```
  - Example of RNA sequence and corresponding secondary structure  

#### **(2) Plant Genomics Benchmark (PGB)**
- **Purpose**: Evaluates DNA-based genomics tasks and multi-species generalization  
- **Example Data**:
  ```
  >Example_DNA_1
  ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
  ```
  - Example of DNA sequence used in plant genomics  

#### **(3) EternaV2 RNA Design Dataset**
- **Purpose**: Evaluates RNA design performance (Str2Seq prediction)  
- **Example Data**:
  ```
  Target Structure: (((....)))....(((..)))
  Generated Sequence: AUGCUAGCUAGCUAGCUA
  ```
  - Task involves generating RNA sequences that match the given secondary structure  

#### **(4) Zero-shot Secondary Structure Prediction (Seq2Str)**
- **Purpose**: Evaluates RNA secondary structure prediction performance (Seq2Str)  
- **Example Data**:
  ```
  Input Sequence: AUGCUAGCUAGCUAGCUA
  Predicted Structure: (((....)))....(((..)))
  ```

---

### **3. Experimental Setup**  

To train and evaluate OmniGenome, we used the following **hardware and software environment**:  

- **Hardware**:  
  - **NVIDIA RTX 4090 GPUs × 8**  
  - **512GB RAM**  

- **Software**:  
  - **PyTorch 2.0.0**  
  - **Transformers 4.37.1**  
  - **ViennaRNA 2.6.4**  
  - **Data preprocessing: cd-hit-est, BLAST**  

With this setup, we efficiently **trained and evaluated OmniGenome on large-scale RNA and DNA datasets**.


<br/>  
# 기타  




논문에는 **모델 구조, 성능 비교, 실험 결과 등을 시각적으로 나타내는 다양한 피규어(Figure), 테이블(Table), 그래프(Graph), 다이어그램(Diagram)**이 포함되어 있다. 각각의 요소가 전달하는 핵심 내용을 설명한다.  

---

### **1. 모델 구조 및 학습 개요 (Figure 1: OmniGenome 모델 개요 다이어그램)**  
- **설명**: OmniGenome의 학습 및 예측 과정을 설명하는 다이어그램  
- **주요 구성 요소**:  
  - **Seq2Str Input** (RNA 서열 입력)  
  - **Masked Language Model (MLM) Input** (마스킹된 서열 입력)  
  - **Str2Seq Input** (RNA 구조 입력)  
  - **출력 과정**: 구조 예측(Seq2Str) 및 서열 복원(Str2Seq)  
- **의미**: OmniGenome이 서열과 구조 정보를 양방향으로 활용하여 학습되는 방식과, 이를 통해 Seq2Str 및 Str2Seq 예측이 가능함을 시각적으로 보여준다.  

---

### **2. RNA 디자인 성능 비교 (Table 2: EternaV2 RNA Design Benchmark 성능 비교 테이블)**  
- **설명**: RNA 디자인(서열-구조 변환, Str2Seq) 성능을 비교  
- **주요 비교 모델**:  
  - **RNAInverse** (전통적인 RNA 디자인 기법)  
  - **3UTRBERT, DNABERT2, SpliceBERT** (기존 RNA 및 DNA FM)  
  - **OmniGenome (52M, 186M, OmniGenome+)**  
- **주요 결과**:  
  - OmniGenome186M+가 **74%의 정확도**를 기록하며 가장 우수한 성능을 보임  
  - 기존 RNA FMs(RNA-MSM, RNA-FM 등)은 **3% 이하의 퍼즐 해결률**을 보임  
  - RNAInverse는 30% 해결률을 기록하며, 기존 FM보다 우수하지만 OmniGenome보다는 낮은 성능을 보임  
- **의미**: OmniGenome이 Str2Seq 작업에서 기존 RNA FMs보다 **20배 이상의 성능 향상**을 달성했음을 강조  

---

### **3. 제로샷 이차 구조 예측 성능 (Table 3: Zero-shot SSP 성능 비교 테이블)**  
- **설명**: RNA 서열을 입력받아 이차 구조를 예측하는 Seq2Str 성능 비교  
- **사용 데이터셋**: Archive2, bpRNA, Stralign  
- **주요 결과**:  
  - OmniGenome186M+ 모델이 **최대 74.85% F1-score**를 기록  
  - ViennaRNA(전통적인 구조 예측 모델) 대비 유사하거나 더 나은 성능을 보임  
  - 기존 RNA FMs는 최대 66%에 머무름  
- **의미**: OmniGenome이 **기존의 생물정보학 기반 구조 예측 모델(ViennaRNA)과 대등한 성능을 가지며**, 기존 RNA FMs 대비 **Seq2Str 성능이 우수함**을 보여줌  

---

### **4. DNA 유전체 분석 성능 비교 (Table 4: Plant Genomics Benchmark 성능 비교 테이블)**  
- **설명**: DNA 기반 유전체 분석에서 OmniGenome과 경쟁 모델 비교  
- **평가 작업**:  
  - PolyA site prediction  
  - Chromatin Accessibility 예측  
  - Promoter Strength 예측  
  - Enhancer Region 분석 등  
- **주요 결과**:  
  - OmniGenome186M+가 대부분의 작업에서 기존 DNA FMs(NT-V2, Agro-NT, Caduceus 등)을 초월하는 성능 기록  
  - 특히 **Gene Expression, Enhancer Region 예측에서 가장 높은 F1-score** 달성  
- **의미**: OmniGenome이 **RNA뿐만 아니라 DNA 기반 유전체 분석에서도 강력한 성능을 보이며, 다목적 유전체 분석 모델로 활용 가능함**을 입증  

---

### **5. RNA 및 DNA 유전체 분석 종합 성능 비교 (Table 5: RGB Benchmark 성능 비교 테이블)**  
- **설명**: RNA 및 DNA 기반 유전체 분석에서 OmniGenome과 경쟁 모델의 성능 비교  
- **주요 평가 작업**:  
  - RNA 분해율 예측  
  - 염기 변이 감지 (SNM detection)  
  - 이차 구조 예측 (Seq2Str)  
- **주요 결과**:  
  - OmniGenome186M+가 **모든 작업에서 기존 RNA/DNA FMs을 초월하는 성능**을 기록  
  - RNA 구조 예측에서는 ViennaRNA와 대등하거나 더 높은 성능을 보임  
  - SNM 변이 탐지에서 기존 DNA 모델보다 우수한 결과  
- **의미**: OmniGenome이 **다양한 유전체 분석 작업에서 기존 RNA 및 DNA FMs 대비 일관된 성능 향상**을 보였음을 강조  

---

### **6. 모델 하이퍼파라미터 및 학습 설정 (Table 1: OmniGenome 모델 스펙 테이블)**  
- **설명**: OmniGenome52M 및 OmniGenome186M의 모델 구성 비교  
- **주요 항목**:  
  - 모델 계층 수 (Layers)  
  - 임베딩 차원 (Embedding Dimension)  
  - 헤드 수 (Number of Heads)  
  - 파라미터 개수 (Number of Parameters)  
- **의미**: OmniGenome의 모델 스펙을 명확히 제시하고, 작은 모델(52M)에서도 충분히 우수한 성능을 보인다는 점을 강조  

---



This paper includes **figures, tables, graphs, and diagrams** that illustrate the **model architecture, performance comparisons, and experimental results**. Below is a detailed explanation of each component.  

---

### **1. Model Architecture and Training Overview (Figure 1: OmniGenome Diagram)**  
- **Description**: A diagram illustrating the training and inference process of OmniGenome  
- **Key Components**:  
  - **Seq2Str Input** (RNA sequence input)  
  - **Masked Language Model (MLM) Input** (Masked sequence input)  
  - **Str2Seq Input** (RNA structure input)  
  - **Output Process**: Secondary structure prediction (Seq2Str) and sequence reconstruction (Str2Seq)  
- **Significance**: Shows how OmniGenome learns by **leveraging both sequence and structure information**, enabling Seq2Str and Str2Seq predictions.  

---

### **2. RNA Design Performance (Table 2: EternaV2 RNA Design Benchmark)**  
- **Description**: Performance comparison for RNA design (Str2Seq task)  
- **Key Findings**:  
  - **OmniGenome186M+ achieved 74% accuracy**, outperforming all other models  
  - Existing RNA FMs (RNA-MSM, RNA-FM) solved only **up to 3% of puzzles**  
  - RNAInverse (traditional RNA design) solved 30%  
- **Significance**: OmniGenome shows **20× improvement** over existing RNA FMs in Str2Seq tasks.  

---

### **3. Zero-shot Secondary Structure Prediction (Table 3: Zero-shot SSP Benchmark)**  
- **Description**: Evaluates Seq2Str performance on RNA structure prediction tasks  
- **Key Findings**:  
  - OmniGenome186M+ **achieved up to 74.85% F1-score**  
  - Comparable or superior to ViennaRNA (traditional structure predictor)  
  - Existing RNA FMs scored **up to 66%**  
- **Significance**: Demonstrates **OmniGenome’s superior Seq2Str capability**, even in a zero-shot setting.  

---

### **4. DNA Genomics Performance (Table 4: Plant Genomics Benchmark)**  
- **Description**: DNA-based genomics task performance comparison  
- **Key Findings**:  
  - **OmniGenome186M+ consistently outperformed DNA FMs (NT-V2, Agro-NT, Caduceus, etc.)**  
  - Achieved the **highest F1-score in Gene Expression and Enhancer Region predictions**  
- **Significance**: OmniGenome is not only effective for RNA but also **excels in DNA-based genomics tasks**.  

---

### **5. Comprehensive Genomics Performance (Table 5: RGB Benchmark)**  
- **Description**: Evaluates OmniGenome against RNA and DNA FMs across various tasks  
- **Key Findings**: OmniGenome **outperforms all baselines in RNA degradation, SNM detection, and structure prediction**  
- **Significance**: Establishes **OmniGenome as the new state-of-the-art model** in both RNA and DNA genomics.  




<br/>
# refer format:     


@article{Yang2025OmniGenome,
  author    = {Heng Yang and Renzhi Chen and Ke Li},
  title     = {Bridging Sequence-Structure Alignment in RNA Foundation Models},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  volume    = {39},
  number    = {4},
  pages     = {TBD},
  publisher = {Association for the Advancement of Artificial Intelligence},
  doi       = {TBD},
  url       = {https://arxiv.org/abs/2407.11242}
}



Yang, Heng, Renzhi Chen, and Ke Li. 2025. "Bridging Sequence-Structure Alignment in RNA Foundation Models." Proceedings of the AAAI Conference on Artificial Intelligence 39 (4): TBD. Association for the Advancement of Artificial Intelligence. https://arxiv.org/abs/2407.11242.





