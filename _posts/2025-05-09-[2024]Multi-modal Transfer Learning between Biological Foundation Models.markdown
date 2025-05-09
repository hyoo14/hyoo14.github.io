---
layout: post
title:  "[2024]Multi-modal Transfer Learning between Biological Foundation Models"  
date:   2025-05-09 16:42:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

DNA+RNA+PROTEIN LM들의 임베딩을 통합한 일종의 멀티모달 통합 모델.. 나도 생각해본..  


짧은 요약(Abstract) :    






이 논문은 DNA, RNA, 단백질이라는 서로 다른 생물학적 서열 정보를 함께 활용할 수 있는 **다중모달 생물학 파운데이션 모델**을 제안합니다. 지금까지의 대부분의 모델은 하나의 서열 종류만을 처리했지만, 실제 생물학적 현상은 DNA에서 RNA, 다시 단백질로 이어지는 복잡한 흐름을 따릅니다. 저자들은 **DNA, RNA, 단백질 각각에 대해 사전학습된 인코더**를 활용하고 이들 정보를 \*\*통합하는 아키텍처 (IsoFormer)\*\*를 구축하여, 유전자 하나에서 다양한 RNA 전사체(isoforms)가 어떻게 생성되고 조직별로 다르게 발현되는지를 예측하는 문제에 도전합니다. 이 모델은 기존 방법보다 더 높은 정확도를 보여주었고, 모달리티 간 지식 이전까지 가능함을 실험적으로 보였습니다. 저자들은 모델을 오픈소스로 공개하며, 향후 생물학 문제에서의 다중모달 접근 가능성을 제시합니다.

---



This paper introduces a **multi-modal biological foundation model** that integrates DNA, RNA, and protein sequence data, addressing the limitations of current approaches that focus on a single modality. Biological processes inherently involve all three types of sequences, as described by the central dogma (DNA → RNA → Protein). The authors propose **IsoFormer**, a model that uses **pre-trained encoders** for each modality and an aggregation layer to combine the information. IsoFormer is applied to the challenging task of **predicting RNA transcript isoform expression levels across human tissues**, which cannot be solved with DNA alone. The model outperforms existing methods by leveraging cross-modality knowledge transfer and achieves **state-of-the-art results**. This work represents a step forward in modeling complex biological phenomena using unified sequence representations, and the model is released open-source to foster further research.

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





이 논문은 **IsoFormer**라는 다중모달 모델을 제안하며, DNA, RNA, 단백질 서열을 동시에 입력으로 받아 **RNA 전사체의 조직별 발현량을 예측**하는 것이 목표입니다.

####  1. 입력 및 인코더

* DNA: ACGT 염기로 구성된 서열
* RNA: ACGU 염기로 구성된 서열 (실제로는 U→T 치환으로 DNA 인코더 재사용)
* 단백질: 20개의 아미노산 문자로 구성된 서열
* 각 서열은 해당 modality에 맞는 **사전학습(pre-trained) 인코더**로 임베딩됨:

  * **DNA**: Enformer 또는 Nucleotide Transformer (NT-v2)
  * **RNA**: NT-v2 사용 (RNA 사전학습 모델이 공개되지 않아 NT 재사용)
  * **단백질**: ESM2 (MLM 방식으로 학습된 150M 파라미터 모델)

####  2. Aggregation Module (통합 모듈)

* 각 모달리티 인코더에서 나온 임베딩을 **크로스 어텐션 기반 통합 모듈**로 통합
* Residual connection을 포함해 모달리티 간 정보 상호작용을 유도
* **하나 이상의 모달리티가 없을 경우에도 작동 가능** (해당 어텐션은 0 처리)

####  3. 예측 대상 및 Head

* DNA 중심의 유전자 위치를 중심으로 서열을 자름 (프로모터 및 조절 요소 포함)
* 해당 유전자의 하나의 RNA isoform, 대응되는 단백질 서열을 함께 입력
* 다중모달 임베딩을 통합한 후, **expression prediction head**가 조직별 발현량 예측

####  4. 트레이닝

* 손실 함수는 **Mean Squared Error (MSE)**: 예측한 발현량과 실제 발현량 사이 오차 최소화
* 데이터: **GTEx v8**에서 5,000명 이상의 개인으로부터 30개 조직의 RNA 전사체 발현량 (TPM)
* 총 17만 개 전사체 중 약 9만 개는 단백질로 번역 가능 (protein-coding)
* 학습 데이터셋은 gene 단위로 train/test 분리하여 정보 누수를 방지

---



This paper presents **IsoFormer**, a multi-modal architecture designed to predict **RNA transcript isoform expression levels** across tissues by integrating DNA, RNA, and protein sequences.

####  1. Inputs and Encoders

* **DNA**: sequences over ACGT, encoded using either Enformer (supervised on gene expression) or NT-v2 (self-supervised).
* **RNA**: sequences over ACGU, encoded by NT-v2 (by replacing U with T, as RNA-specific models are not publicly available).
* **Protein**: amino acid sequences (20 residues), encoded using **ESM2** (150M parameters, MLM-pretrained).

Each modality is independently processed by a **pre-trained encoder**.

####  2. Aggregation Module

* Uses **cross-attention layers with residual connections** to integrate the embeddings from each modality.
* Modality-specific multi-modal embeddings are created (e.g., $h'_{DNA} = f_{\text{agg}}(h_{DNA}, h_{RNA}, h_{protein})$).
* The method is **robust to missing modalities** (cross-attention is skipped or zeroed out).

####  3. Expression Prediction Head

* The input DNA sequence is centered on the **transcription start site** (TSS) of the gene.
* Together with a transcript’s RNA and matching protein sequence, the model predicts **tissue-specific expression**.
* A global average pooling and a linear layer per tissue produce the final output.

####  4. Training Setup

* Objective: **Minimize MSE loss** between predicted and actual RNA transcript expression across tissues.
* Dataset: Derived from **GTEx v8**, covering \~170,000 transcripts across 30 tissues from 5,000 individuals.
* Only expression-averaged values per transcript per tissue are used.
* **Gene-level data split** ensures no leakage between train and test sets.
* Optimizer: Adam (learning rate $3 \times 10^{-5}$), batch size 64, early stopping on 5% validation split.

---



   
 
<br/>
# Results  





#### 테스트 데이터

* 출처: **GTEx v8** 데이터셋의 **RNA 전사체 발현량 (TPM)** 데이터
* 범위: **30개 조직**, **약 17만 개 전사체**, **5,000명 이상**으로부터 수집
* 각 전사체의 RNA, DNA, 단백질 서열을 매핑하여 사용
* 학습-테스트는 **유전자 단위 분리**로 구성하여 정보 누수 방지

#### 평가 지표 (Metrics)

1. **R² (Coefficient of Determination)**

   * 예측값이 실제 발현값을 얼마나 잘 설명하는지를 수치화
2. **Spearman Correlation**

   * 전사체 간 발현 수준의 **순위 상관성** 측정 (순서 일치에 초점)

#### 비교 대상 모델 (경쟁 모델들)

* **단일모달 모델**:

  * DNA only (NT 또는 Enformer 기반)
  * RNA only (NT 기반)
  * Protein only (ESM 기반)
* **이중모달 조합**:

  * DNA + Protein
  * DNA + RNA
* **삼중모달 (IsoFormer)**:

  * DNA + RNA + Protein

또한 **DNA 인코더**로는 세 가지 모델 비교:

* **NT (Nucleotide Transformer)**
* **Enformer** (유전자 발현 예측 특화)
* **Borzoi** (RNA-seq coverage 예측 특화, 512kb 컨텍스트 지원)

#### 주요 결과

* IsoFormer는 **DNA + RNA + Protein** 조합에서 가장 높은 성능:

  * R²: **0.53**
  * Spearman: **0.72**
* 단일모달보다 모달리티를 늘릴수록 성능 상승:

  * DNA only: R² 0.13 / Spearman 0.43
  * RNA only: R² 0.36 / Spearman 0.61
  * Protein only: R² 0.20 / Spearman 0.46
* **Enformer**를 DNA 인코더로 사용할 경우 NT보다 더 높은 성능
* **Ablation** 실험 결과: cross-attention 기반 aggregation이 최적
* 사전학습(pre-training)된 인코더 사용이 매우 중요:

  * 세 인코더 모두 랜덤 초기화 시 R² 0.10로 급감

---


####  Test Dataset

* Source: **GTEx v8**, RNA transcript TPMs across 30 human tissues
* Total: \~170,000 transcripts from over 5,000 individuals
* Triplets constructed per sample: DNA (centered on TSS), RNA transcript, and corresponding protein
* Data split **by gene** to avoid information leakage

####  Evaluation Metrics

1. **R² (Coefficient of Determination)**: measures how well predicted expression values match actual values
2. **Spearman Correlation**: ranks transcripts by expression level to assess ordinal agreement

####  Baseline and Competitor Models

* **Single-modality models**:

  * DNA only (NT or Enformer)
  * RNA only (NT)
  * Protein only (ESM2)
* **Multi-modality combinations**:

  * DNA + Protein
  * DNA + RNA
  * DNA + RNA + Protein (**IsoFormer**)
* **DNA encoders tested**:

  * **NT** (self-supervised transformer)
  * **Enformer** (supervised on gene expression)
  * **Borzoi** (pre-trained on RNA-seq coverage prediction)

####  Key Results

* **Best performance** with full multi-modal IsoFormer:

  * **R² = 0.53**, **Spearman = 0.72**
* Performance improves with each added modality:

  * DNA only: R² 0.13, Spearman 0.43
  * RNA only: R² 0.36, Spearman 0.61
  * Protein only: R² 0.20, Spearman 0.46
* Using **Enformer** as DNA encoder leads to better results than NT
* **Ablation studies** showed that the **cross-attention aggregation** outperforms other strategies
* **Pre-trained encoders are critical**: R² drops from 0.53 to 0.10 when all encoders are randomly initialized






<br/>
# 예제  





####  트레이닝 / 테스트 데이터 구성 예시

모델은 다음과 같은 3가지 서열 정보를 **하나의 샘플 단위**로 받아들입니다:

| 구성 요소      | 실제 데이터 예시                                         |
| ---------- | ------------------------------------------------- |
| **DNA 서열** | `ACGTACGTACGT...` (12,288 염기; 중심은 TSS)            |
| **RNA 서열** | `ACGUACGUACGU...` (전사체 길이, 최대 12kb까지 crop)        |
| **단백질 서열** | `MTEYKLVVVGAGGVGKSALTIQLIQNHFV...` (아미노산 서열)      |
| **타겟 값**   | `[1.24, 0.83, ..., 2.01]` ← 30개 조직의 발현량 (log TPM) |

* DNA 서열은 해당 전사체의 \*\*transcription start site (TSS)\*\*를 중심으로 자름
* RNA 서열은 전사체 isoform 자체
* 단백질 서열은 해당 전사체가 번역되어 생성하는 아미노산 서열
* 타겟은 GTEx에서 측정된 **30개 조직의 RNA 발현량 (TPM → log 변환 후 정규화)**

####  모델 입력 형식 (예시)

```python
{
  "dna_sequence": "ACGTACGT... (length=12,288)",
  "rna_sequence": "ACGUACGU... (length=up to 12,000)",
  "protein_sequence": "MVHLTPEEKSAVTALWGKVNVD... (length=up to 1,200)",
}
```

#### 모델 출력 형식 (예시)

```python
{
  "predicted_expression": [1.23, 0.95, ..., 1.87]  # size = 30 (tissues)
}
```

→ 이 출력값은 해당 전사체 isoform이 각 조직에서 **어느 정도로 발현되는지**를 예측한 결과입니다.

---


####  Example of Training / Test Data

Each sample in the dataset consists of:

| Component          | Example                                                    |
| ------------------ | ---------------------------------------------------------- |
| **DNA sequence**   | `"ACGTACGTACGT..."` (12,288 nucleotides, centered at TSS)  |
| **RNA transcript** | `"ACGUACGUACGU..."` (cropped to 12kb if needed)            |
| **Protein seq.**   | `"MTEYKLVVVGAGGVGKSALTIQLIQ..."` (amino acids)             |
| **Target output**  | `[1.24, 0.83, ..., 2.01]` (expression in 30 human tissues) |

* The DNA sequence is centered on the **Transcription Start Site (TSS)** of the gene.
* The RNA sequence is the **specific isoform transcript**.
* The protein sequence is the **translated protein** (if applicable).
* The target values are **log-transformed and normalized transcript expression levels** per tissue.

####  Model Input (JSON-style)

```json
{
  "dna_sequence": "ACGTACGTACGTACGT... (12,288 nt)",
  "rna_sequence": "ACGUACGUACGUACGU... (up to 12,000 nt)",
  "protein_sequence": "MVHLTPEEKSAVTALWGKVNVD... (up to 1,200 aa)"
}
```

####  Model Output

```json
{
  "predicted_expression": [1.23, 0.95, ..., 1.87]  // 30 values, one per tissue
}
```

→ This output vector represents **predicted tissue-specific expression levels** for the given transcript isoform.

---





<br/>  
# 요약   





이 논문은 DNA, RNA, 단백질 서열을 통합하는 다중모달 모델 **IsoFormer**를 제안하여 RNA 전사체 isoform의 조직별 발현량을 예측한다. GTEx 데이터 기반으로 약 17만 개의 전사체에 대해 DNA 중심 서열, RNA 전사체, 단백질 서열을 입력으로 사용하고, 30개 조직에 대한 발현값을 출력으로 학습한다. 실험 결과 IsoFormer는 기존 단일모달 모델보다 우수한 성능(R²=0.53, Spearman=0.72)을 보였으며, 사전학습된 인코더 활용이 핵심 성능 요인임을 입증했다.

---



This paper introduces **IsoFormer**, a multi-modal model that integrates DNA, RNA, and protein sequences to predict tissue-specific expression levels of RNA transcript isoforms. Using GTEx data, each sample includes a DNA sequence centered at the transcription start site, an RNA isoform, and its corresponding protein, with output being expression values across 30 tissues. IsoFormer achieves superior performance (R²=0.53, Spearman=0.72) compared to single-modality baselines, demonstrating the importance of using pre-trained encoders and cross-modal aggregation.



<br/>  
# 기타  




####  대표 Figure

* **Figure 1**: IsoFormer 구조도
  → DNA, RNA, 단백질 인코더를 통과한 후, 크로스어텐션 기반으로 임베딩을 통합해 발현량 예측
* **Figure 2**: 생물학적 흐름 (DNA → RNA isoform → protein) 및 입력 예시
  → 하나의 DNA에서 다양한 RNA isoform이 생성되고, 조직별로 발현량이 달라짐을 시각화
* **Figure 3 (왼쪽)**: 조직별 성능
  → IsoFormer는 모든 조직에서 Enformer보다 높은 R² 달성 (예: 뇌, 간, 폐 등)
* **Figure 3 (오른쪽)**: RNA 인코더 내부 attention 분포 분석
  → 3’UTR, CDS, 5’UTR 등 영역에 따라 attention이 달라지며, IsoFormer는 중간\~상위층에서 기능별 특화된 attention을 보임

####  주요 테이블

* **Table 2**: 모달리티별 성능 비교
  → DNA + RNA + Protein 조합이 가장 높고, RNA 단독도 DNA보다 훨씬 강력함
* **Table 3**: DNA 인코더별 성능 (NT vs. Enformer vs. Borzoi)
  → Enformer 기반 IsoFormer가 최고 성능 (R²=0.53), Borzoi도 높은 성능
* **Table 4**: Aggregation 방법 ablation
  → Perceiver Resampler, C-Abstractor보다 cross-attention이 가장 효율적
* **Table 5**: Pre-training 여부에 따른 성능
  → 세 인코더 모두 랜덤 초기화 시 성능 급감 (R²=0.10 → 0.53)

####  어펜딕스 인사이트

* **Appendix A**: GTEx 기반 데이터 생성 방식 명시 (30개 조직, 5,000명 이상)
* **Appendix C**: RNA attention 분석 공식과 유의미한 layer/head에 대한 t-test 결과
  → IsoFormer에서는 3’UTR에 대한 attention이 중간 layer에서 유의미하게 증가

---


####  Key Figures

* **Figure 1**: IsoFormer architecture
  → Shows modality-specific encoders followed by cross-attention layers to produce multi-modal embeddings for expression prediction
* **Figure 2**: Central dogma + input composition
  → Illustrates how a single gene gives rise to multiple RNA isoforms with tissue-dependent expression levels
* **Figure 3 (Left)**: R² per tissue
  → IsoFormer consistently outperforms Enformer across all tissues (e.g., brain, liver, lung)
* **Figure 3 (Right)**: Attention maps in RNA encoder
  → Reveals that IsoFormer develops specialized attention patterns: 3’UTR is emphasized in middle layers, CDS in upper layers

####  Key Tables

* **Table 2**: Performance by modality combinations
  → Best with DNA+RNA+Protein; RNA alone is more predictive than DNA alone
* **Table 3**: DNA encoder comparison (NT, Enformer, Borzoi)
  → IsoFormer with Enformer achieves the best overall results (R²=0.53)
* **Table 4**: Ablation of aggregation strategies
  → Cross-attention outperforms Perceiver Resampler and C-Abstractor
* **Table 5**: Importance of pre-training
  → Removing pre-training leads to severe performance drops (R² from 0.53 to 0.10)

####  Appendix Insights

* **Appendix A**: Describes dataset construction from GTEx v8 (30 tissues, 5,000+ individuals)
* **Appendix C**: Attention analysis with statistical tests
  → Shows significantly increased attention to 3’UTR in IsoFormer’s mid layers (via t-test)




<br/>
# refer format:     



@inproceedings{garau2024isoformer,
  title={Multi-modal Transfer Learning between Biological Foundation Models},
  author={Garau-Luis, Juan Jose and Bordes, Patrick and Gonzalez, Liam and Roller, Masa and de Almeida, Bernardo P. and Hexemer, Lorenz and Blum, Christopher and Laurent, Stefan and Grzegorzewski, Jan and Lang, Maren and Pierrot, Thomas and Richard, Guillaume},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  note={To appear},
  publisher={Curran Associates, Inc.}
}




Garau-Luis, Juan Jose, Patrick Bordes, Liam Gonzalez, Masa Roller, Bernardo P. de Almeida, Lorenz Hexemer, Christopher Blum, Stefan Laurent, Jan Grzegorzewski, Maren Lang, Thomas Pierrot, and Guillaume Richard. “Multi-modal Transfer Learning between Biological Foundation Models.” Advances in Neural Information Processing Systems (NeurIPS), 2024 (to appear).   




