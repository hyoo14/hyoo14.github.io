---
layout: post
title:  "[2024]MP-RNA: Unleashing Multi-species RNA Foundation Model via Calibrated Secondary Structure Prediction"  
date:   2025-04-03 01:56:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

RNA 파운데이션 모델 제안(이차구조로 사전학습)   



짧은 요약(Abstract) :    



---



이 논문에서는 RNA 기반의 기초 모델(RNA Foundation Model, FM)이 다양한 유전체(in-silico genomic) 과제를 해결하는 데 사용되고 있지만, 대부분의 기존 모델들이 RNA의 **이차 구조(secondary structure)** 정보를 사전 학습(pretraining)에 반영하지 않아 성능이 제한된다는 점을 지적합니다.

이 문제를 해결하기 위해, 저자들은 **고품질의 이차 구조 주석(annotations)**을 필터링하여 FM의 학습에 사용함으로써, **단일 염기 수준(single nucleotide resolution)**의 과제에서 모델의 표현력을 높였습니다. 이들이 제안한 모델 **MP-RNA**는 네 가지 주요 유전체 벤치마크에서 기존 모델보다 우수한 성능을 보였으며, 특히 **RNA 이차 구조 예측에서 40% 향상**된 결과를 보였습니다. 놀랍게도 DNA에 대해 사전 학습되지 않았음에도 불구하고 DNA 유전체 벤치마크에서도 최고 수준의 결과를 달성했습니다.

저자들은 **코드와 튜토리얼, 모델을 공개**하여, 생물학적 현실과 컴퓨터 기반 예측 간의 간극을 줄이기 위한 추가 연구를 장려하고 있습니다.

---



RNA foundation models (FMs) have been extensively used to interpret genomic sequences and address a wide range of in-silico genomic tasks. However, current RNA FMs often overlook the incorporation of secondary structures in the pretraining of FMs, which impedes the effectiveness in various genomic tasks. To address this problem, we leverage filtered high-fidelity structure annotations for structure pretraining to enhance the modeling ability of FMs in single nucleotide resolution tasks. Experimental evaluations across four comprehensive genomic benchmarks demonstrate that our FM (MP-RNA) consistently outperforms existing RNA FMs, achieving a 40% improvement in RNA secondary structure prediction and obtaining top-tier results on DNA genomic benchmarks even though it has not been pretrained on any DNA genome. We release the code and tutorials and models to encourage further research to bridge the gap between in-silico predictions and biological reality.

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



###  1. 백본 및 아키텍처  
MP-RNA는 Transformer 기반의 언어 모델을 기반(backbone)으로 사용합니다.  
모델은 두 가지 버전으로 개발되었습니다:

- **MP-RNA-52M**:  
  - 레이어 수: 16  
  - 임베딩 차원: 480  
  - 헤드 수: 24  
  - 파라미터 수: 약 5,200만 개  

- **MP-RNA-186M**:  
  - 레이어 수: 32  
  - 임베딩 차원: 720  
  - 헤드 수: 30  
  - 파라미터 수: 약 1억 8,600만 개  

공통적으로 Rotary Position Embedding, AdamW 옵티마이저, Linear Decay 학습률 스케줄러가 사용되었고, 드롭아웃은 적용되지 않았습니다.

---

###  2. 토크나이제이션 및 입력  
MP-RNA는 기존의 k-mer나 BPE 기반 토크나이제이션 대신, **SNT (Single-Nucleotide Tokenization)** 방식을 채택하여 염기 수준의 정밀한 표현이 가능하도록 설계되었습니다.  
이 방식은 RNA의 이차 구조 예측이나 돌연변이 탐지 같은 **단일 염기 수준의 과제**에 적합합니다.

---

###  3. 학습 목표 및 손실 함수  
사전 학습(pretraining)은 아래 세 가지 목적을 함께 학습합니다:

1. **RNA 이차 구조 예측 (SSP)**:  
   ViennaRNA를 사용하여 생성된 구조 정보를 바탕으로 토큰 수준 예측 (Loss: Cross-Entropy)

2. **단일 염기 돌연변이 복원 (SNMR)**:  
   임의로 돌연변이를 삽입한 뒤 원래 염기를 복원하도록 학습 (Loss: Cross-Entropy)

3. **마스킹된 RNA 언어 모델링 (MRLM)**:  
   염기 및 구조를 일부 마스킹하고 복원하는 일반적인 언어 모델링 (Loss: Cross-Entropy)

---

###  4. 트레이닝 데이터  
- **데이터 출처**: OneKP 이니셔티브의 1,124개 식물 종으로부터 얻은 대규모 전사체 데이터  
- **전처리 과정**:
  1. 긴 서열을 1,024 염기 단위로 슬라이싱  
  2. 중복 제거  
  3. 50 염기 미만의 서열 제거  

최종적으로 **54.2B개의 토큰**이 학습에 사용되었습니다.

---


###  1. Backbone and Architecture  
MP-RNA is built on a Transformer backbone. Two model sizes were implemented:

- **MP-RNA-52M**:  
  - 16 transformer layers  
  - Embedding dim: 480  
  - 24 attention heads  
  - ~52 million parameters

- **MP-RNA-186M**:  
  - 32 transformer layers  
  - Embedding dim: 720  
  - 30 attention heads  
  - ~186 million parameters

Both versions use rotary positional embeddings, the AdamW optimizer, a linear decay scheduler, and zero dropout.

---

###  2. Tokenization and Input  
Instead of k-mer or BPE, MP-RNA adopts **Single-Nucleotide Tokenization (SNT)**, where each token corresponds to a single base (A, U, C, G, etc.).  
This enables precise base-level modeling, essential for tasks like **RNA secondary structure prediction** and **mutation detection**.

---

###  3. Pretraining Objectives and Losses  
The model is trained in a multi-objective fashion with the following tasks:

1. **Secondary Structure Prediction (SSP)**:  
   Predicts structure tokens (‘(’, ‘)’, ‘.’) from ViennaRNA-annotated data.  
   *(Loss: Cross-entropy)*

2. **Single Nucleotide Mutation Repair (SNMR)**:  
   Detects and corrects randomly introduced base mutations.  
   *(Loss: Cross-entropy)*

3. **Masked RNA Language Modeling (MRLM)**:  
   Recovers masked bases and structure tokens.  
   *(Loss: Cross-entropy)*

---

###  4. Training Data  
- **Source**: RNA transcriptomes from 1,124 plant species (OneKP initiative)  
- **Preprocessing**:
  - Sequences sliced into 1,024-base segments  
  - Duplicate removal  
  - Discarding sequences shorter than 50 bases  

This results in a total of **54.2 billion tokens** used for training.

---




   
 
<br/>
# Results  



---


###  주요 테스크

MP-RNA는 **RNA 및 DNA 유전체 이해**를 위한 다음 세 가지 주요 벤치마크에서 평가되었습니다:

1. **RGB (RNA Genomic Benchmark)**  
   - 단일 염기 수준(Single-nucleotide level)의 RNA 분석을 목표로 하는 6가지 과제 포함  
   - 예: mRNA 분해 속도 예측, 이차 구조 예측, 돌연변이 탐지 및 복원  

2. **PGB (Plant Genomic Benchmark)**  
   - 식물 DNA에 기반한 8가지 과제로, 모델의 전이 학습 능력 평가  
   - 예: Polyadenylation, 스플라이스 사이트, 유전자 발현 예측  

3. **GB & GUE Benchmarks**  
   - 다양한 종의 DNA 분석을 포함한 추가 벤치마크 (논문 부록에 상세)

---

###  테스트 데이터

- RGB는 실제 및 합성된 RNA 데이터로 구성되며, mRNA 과제는 Kaggle COVID-19 백신 대회 데이터를 사용함  
- PGB는 다수의 식물 종에서 수집된 DNA 유전체 데이터로 구성됨  
- GUE 및 GB는 공개 DNA 벤치마크를 사용하며, MP-RNA는 해당 데이터로 **사전 학습되지 않음**

---

###  비교한 경쟁 모델 (Baseline Models)

총 13개의 모델과 성능 비교가 이루어졌습니다. 주요 모델은 다음과 같습니다:

- **RNA 기반**: RNA-FM, RNA-MSM, RNA-BERT, Uni-RNA, SpliceBERT 등  
- **DNA 기반**: DNABERT-2, NT-V2, HyenaDNA, Caduceus, Agro-NT, CDSBERT 등  

---

###  메트릭 비교 결과 요약

| Task | MP-RNA 성능 요약 | 주요 메트릭 |
|------|------------------|--------------|
| RNA 구조 예측 | 최대 40% 성능 향상 | F1-score |
| mRNA 분해 속도 예측 | RMSE 기준 최저 오차 | RMSE ↓ |
| 돌연변이 탐지/복원 | 다른 모델보다 높은 AUC 및 F1 | AUC, F1 |
| DNA 유전체 과제 (PGB) | DNA 전용 모델보다 우수 | F1, RMSE |
| 제로샷 구조 예측 | ViennaRNA보다 정확도 높음 | F1 |

- 특히 **MP-RNA-186M** 모델은 **모든 RNA 과제에서 최고 성능**, DNA 과제에서도 **사전 학습 없이 상위권 성능**을 기록함  
- 기존 모델들은 대부분 k-mer 또는 BPE 토크나이제이션을 사용하여 단일 염기 수준의 과제에서 제한적인 성능을 보임

---



###  Key Tasks

MP-RNA was evaluated across three major genomic benchmarks:

1. **RGB (RNA Genomic Benchmark)**  
   - Focuses on SN-resolution tasks like mRNA degradation rate prediction, RNA secondary structure prediction, and mutation detection/repair.

2. **PGB (Plant Genomic Benchmark)**  
   - Evaluates transferability of MP-RNA to DNA-based tasks in plant genomes (e.g., polyadenylation, splice site classification).

3. **GB & GUE Benchmarks**  
   - General DNA understanding tasks across species (see appendix).

---

###  Test Data

- **RGB** includes both real and synthetic RNA sequences. The mRNA task is based on the Kaggle COVID-19 vaccine design competition.  
- **PGB** contains large-scale DNA sequences from edible plant species.  
- **GUE and GB** include curated genomic datasets not seen during MP-RNA pretraining.

---

###  Baseline Models

13 baselines were used for comparison, including:

- **RNA-specific FMs**: RNA-FM, RNA-MSM, RNA-BERT, Uni-RNA, SpliceBERT  
- **DNA-specific FMs**: DNABERT-2, NT-V2, HyenaDNA, Caduceus, Agro-NT, CDSBERT

---

###  Metric Comparison Highlights

| Task | MP-RNA Performance | Metrics |
|------|--------------------|---------|
| RNA Secondary Structure | Up to 40% improvement | F1-score |
| mRNA Degradation Rate | Lowest RMSE | RMSE ↓ |
| Mutation Detection/Repair | Best AUC and F1 scores | AUC, F1 |
| DNA Tasks (PGB) | Outperforms DNA-specialized models | F1, RMSE |
| Zero-shot Structure Prediction | Outperforms ViennaRNA | F1-score |

- **MP-RNA-186M** achieves **state-of-the-art performance on RNA tasks** and **competitive results on DNA tasks**, even without DNA pretraining.  
- Existing models using **k-mer or BPE tokenization** struggle with SN-level tasks, while MP-RNA's SNT approach proves more effective.

---






<br/>
# 예제  



---


###  1. 트레이닝 데이터 실제 예시

MP-RNA는 주로 OneKP 프로젝트에서 수집한 **식물 RNA 전사체 서열**을 사용했습니다.  
이 서열들은 1,024 염기 길이로 슬라이스 되어 모델에 입력됩니다.

#### 🧾 예시 (입력 RNA 서열):
```
AAGUACCUAGGCUUUGACUAACCGAGUUGCUAACUGGAGCUU...
```

####  학습 목적에 따라 추가되는 정보:

- **이차 구조 주석 (SSP)**:
  ```
  (((..((...))..)))....((..)).....
  ```

- **돌연변이 삽입 (SNMR)**:
  ```
  원래: AAGUACCUAG...
  변형: AAGUACCAAG... ← 'C'가 'A'로 돌연변이됨
  ```

- **마스킹 (MRLM)**:
  ```
  입력: AAGUA[M]CUAG...
  ```

---

###  2. 테스트 데이터 실제 예시

**RGB 벤치마크**에 포함된 RNA 테스트 데이터는 다양한 종류의 SN(단일 염기) 수준의 태스크로 구성되어 있습니다.

#### 예시 (RNA 이차 구조 예측용):
- 입력 RNA:
  ```
  ACUAGGUUACGAUGCUGA...
  ```
- 예측해야 할 구조 (라벨):
  ```
  ..((....))((....))...
  ```

#### 예시 (SNMR - 돌연변이 복원):
- 입력:
  ```
  AGC**T**GA → 실제는 AGC**A**GA
  ```
- 출력 (복원 결과):
  ```
  A
  ```

---

###  3. 테스크별 실제 Input/Output 예

| Task | Input 예시 | Output 예시 |
|------|------------|-------------|
| **SSP (이차 구조 예측)** | RNA 서열 | 구조표현 (‘(’, ‘)’, ‘.’) |
| **SNMR (돌연변이 복원)** | 변이된 RNA 서열 | 원래 염기 |
| **MRLM (마스킹 언어 모델링)** | 마스킹된 서열 | 마스크된 염기 |
| **mRNA 분해속도 예측** | RNA 서열 | 각 염기별 실수값 (회귀) |

---



###  1. Training Data Example

MP-RNA is pretrained on RNA transcriptome sequences from 1,124 plant species (OneKP).  
Sequences are split into 1,024-base segments.

####  Sample Input (RNA sequence):
```
AAGUACCUAGGCUUUGACUAACCGAGUUGCUAACUGGAGCUU...
```

####  Depending on the training objective:

- **With Secondary Structure (SSP)**:
  ```
  (((..((...))..)))....((..)).....
  ```

- **With Mutation (SNMR)**:
  ```
  Original: AAGUACCUAG...
  Mutated:  AAGUACCAAG...
  ```

- **With Masking (MRLM)**:
  ```
  Input: AAGUA[M]CUAG...
  ```

---

###  2. Test Data Example

**RGB benchmark** provides SN-level RNA datasets with true labels for multiple tasks.

#### Example (RNA Secondary Structure Prediction):
- Input RNA:
  ```
  ACUAGGUUACGAUGCUGA...
  ```
- Target structure:
  ```
  ..((....))((....))...
  ```

#### Example (SNMR - Mutation Repair):
- Input sequence:
  ```
  AGC**T**GA → true: AGC**A**GA
  ```
- Output:
  ```
  A
  ```

---

###  3. Task-specific Input/Output Examples

| Task | Input Example | Output Example |
|------|----------------|----------------|
| **SSP (Structure Prediction)** | RNA sequence | Paired structure (e.g., '(', ')', '.') |
| **SNMR (Mutation Repair)** | Mutated RNA | Correct nucleotide |
| **MRLM (Masked LM)** | Sequence with masks | Masked base |
| **mRNA Degradation** | RNA sequence | Real-valued regression per base |

---





<br/>  
# 요약   





MP-RNA는 단일 염기 수준의 표현을 위한 Single-Nucleotide Tokenization을 기반으로 하여, RNA 이차 구조 예측, 돌연변이 복원 등 세 가지 과제를 멀티태스크로 학습한 RNA 기반 기초 모델이다.  
이 모델은 RNA와 DNA 유전체 벤치마크에서 기존 RNA/DNA 특화 모델보다 우수한 성능을 보였으며, RNA 구조 예측에서 최대 40% 성능 향상을 기록했다.  
트레이닝에는 1,124개 식물 종의 RNA 전사체가 사용되었으며, 실제 입력은 RNA 서열이고 출력은 구조 정보나 염기 복원 결과 등 태스크별로 다양하다.

---


MP-RNA is a foundation model for RNA that uses Single-Nucleotide Tokenization and is trained in a multi-task setup to predict RNA secondary structures, repair mutations, and perform masked language modeling.  
It outperforms existing RNA and DNA models across multiple benchmarks, achieving up to 40% improvement in RNA structure prediction.  
The training data consists of transcriptomes from 1,124 plant species, with tasks taking raw RNA sequences as input and predicting structures or nucleotide-level outputs depending on the task.

---




<br/>  
# 기타  




---


###  Figure (그림)

- **Figure 1**: 전체 모델 학습 구조를 시각적으로 보여줍니다.  
  - 세 가지 학습 목적(SSP, SNMR, MRLM)을 통합하는 멀티태스크 학습 구성도입니다.
  - RNA 서열이 구조 정보 및 마스킹/돌연변이를 통해 처리되는 과정을 단계별로 보여줍니다.

- **Figure 2**: RGB 및 PGB 벤치마크에서의 성능 비교 바 그래프.  
  - MP-RNA가 다른 모델들(RNA-BERT, NT-V2 등)보다 일관되게 높은 성능을 보이는 것을 강조합니다.

- **Figure 3**: 제로샷 RNA 구조 예측 성능 시각화.  
  - ViennaRNA, RNA-BERT와 비교하여 MP-RNA가 더 정밀한 구조 예측을 하는 예시를 제공합니다.

---

###  Table (표)

- **Table 1**: RGB 벤치마크의 6개 세부 태스크에서 MP-RNA-52M/186M과 기존 모델들의 정확도 및 F1 점수를 정리한 테이블.  
- **Table 2**: PGB DNA 기반 벤치마크에서 모델별 성능 비교 (F1, RMSE 등).  
- **Table 3**: RNA 구조 예측 태스크에서 ViennaRNA, RNA-BERT, MP-RNA 간의 상세 비교.

모든 테이블은 결과를 bold 처리하여 SOTA(SOTA: 최고 성능)을 강조합니다.

---

###  Appendix (부록)

- RGB와 PGB 외에 **GB (Genomic Benchmark)**와 **GUE (Genome Understanding Evaluation)**의 상세 구성 및 태스크 설명이 포함되어 있습니다.
- **모델 아키텍처 세부 정보**, 학습 하이퍼파라미터, 데이터 전처리 전략, 도메인별 분석 결과 등이 부록에 포함되어 있습니다.
- **오픈소스 링크 및 사용법**도 부록에서 제공됩니다 (코드, 모델, 튜토리얼 등).

---


###  Figures

- **Figure 1**: Visual overview of MP-RNA’s multi-task training architecture.  
  - Shows how SSP, SNMR, and MRLM are jointly trained with RNA sequences processed through structural and mutational annotations.

- **Figure 2**: Bar charts comparing model performance on RGB and PGB benchmarks.  
  - Demonstrates MP-RNA’s consistent improvement over RNA-BERT, NT-V2, and others.

- **Figure 3**: Zero-shot RNA structure prediction comparison.  
  - Illustrates MP-RNA’s higher fidelity prediction versus ViennaRNA and RNA-BERT.

---

###  Tables

- **Table 1**: Accuracy and F1-score results on six RGB tasks for MP-RNA (both 52M and 186M) compared to existing models.  
- **Table 2**: PGB benchmark results showing MP-RNA outperforming other models on DNA tasks.  
- **Table 3**: Detailed comparison of RNA structure prediction among ViennaRNA, RNA-BERT, and MP-RNA.

All tables bold the best-performing results to highlight MP-RNA’s superiority.

---

###  Appendix

- Includes detailed task descriptions for additional benchmarks: **GB** and **GUE**.  
- Provides full model architecture details, training hyperparameters, data preprocessing steps, and domain-specific analyses.  
- Also includes links to **open-source code, pretrained models, and tutorials** for reproducibility and further research.

---




<br/>
# refer format:     



@inproceedings{yang2024mp,
  title = "MP-RNA: Unleashing Multi-species RNA Foundation Model via Calibrated Secondary Structure Prediction",
  author = "Yang, Heng and Li, Ke",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
  year = "2024",
  pages = "5278--5296",
  publisher = "Association for Computational Linguistics",
  address = "Miami, USA",
  url = "https://aclanthology.org/2024.findings-emnlp.304"
}




Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik, D., & Rost, B. (2022). ProtTrans: Toward understanding the language of life through self-supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10), 7112–7127. https://doi.org/10.1109/TPAMI.2021.3095381   

   





