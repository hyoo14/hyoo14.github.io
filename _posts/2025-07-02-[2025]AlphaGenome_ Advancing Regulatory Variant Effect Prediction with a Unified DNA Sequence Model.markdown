---
layout: post
title:  "[2025]AlphaGenome: Advancing Regulatory Variant Effect Prediction with a Unified DNA Sequence Model"  
date:   2025-07-01 19:11:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


U-net 스타일의 파운데이션 언어 모델 및 정제 모델    
(다운스트림테스크는 랜덤포레스트 쓰기도 함, 다운스트림에 맞게 다른 스코어링을 통해 쓰레쉬홀드 설정하기도 함..(이게 좀 휴리스틱한거 아닌지..))     



짧은 요약(Abstract) :    


이 논문은 AlphaGenome이라는 새로운 딥러닝 모델을 소개합니다. AlphaGenome은 인간과 생쥐의 DNA 염기서열 100만 염기쌍(1Mb)을 입력받아 염기 단위 해상도에서 수천 개의 유전체 기능적 특성(예: 유전자 발현, 전사 시작, 염색질 접근성, 히스톤 변형, 전사인자 결합, 3차원 유전체 접촉, 스플라이싱 등)을 예측할 수 있습니다.

이 모델은 단일 입력으로 다양한 모달리티에 대한 기능 예측과 변이 효과 예측을 동시에 수행할 수 있으며, 총 26개의 변이 예측 과제 중 24개에서 기존 최고 성능 모델을 능가하거나 동일한 성능을 보여줍니다. 특히 TAL1 암유전자 주변의 병리학적 변이의 기능적 메커니즘을 성공적으로 복원합니다.

AlphaGenome은 실험 데이터를 기반으로 훈련되었으며, 유전체 트랙과 변이 효과 예측을 위한 도구도 함께 제공합니다.


The paper presents AlphaGenome, a novel deep learning model that takes 1 megabase of human or mouse DNA sequence as input and predicts thousands of functional genomic tracks at single-base resolution. These tracks span diverse modalities including gene expression, transcription initiation, chromatin accessibility, histone modifications, transcription factor binding, chromatin contact maps, and splicing-related features.

AlphaGenome is trained on both human and mouse genomes and demonstrates state-of-the-art performance on 24 out of 26 variant effect prediction benchmarks. It can simultaneously predict variant effects across all modalities, accurately capturing the regulatory mechanisms of clinically relevant variants, such as those near the TAL1 oncogene.

The authors provide tools for generating genome track and variant effect predictions directly from sequence, making AlphaGenome a unified and extensible platform for regulatory genomics.





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




####  1. 모델 아키텍처

* **입력**: 1Mb(100만 염기) 길이의 인간 또는 생쥐 DNA 서열.
* **출력**: 총 11개 모달리티에서 5,930개(인간), 1,128개(생쥐)의 유전체 트랙.
* **백본 구조**: U-Net 스타일의 구조.

  * **Encoder**: 로컬 염기 서열 패턴을 추출하는 CNN 레이어 포함.
  * **Transformer**: 염색체의 장거리 상호작용(예: enhancer-promoter)을 포착.
  * **Decoder**: sequence resolution 복원.
  * **Output Head**: 각 모달리티에 따라 resolution이 다른 출력을 생성 (1bp, 128bp, 2048bp).
  * **2D 임베딩**: 3차원 유전체 접촉 지도(contact map)를 위한 2048bp 단위의 2D 임베딩 사용.

####  2. 트레이닝 데이터

* **인간 및 생쥐 유전체**에서 수집된 다양한 실험 기반 유전체 트랙 사용:

  * RNA-seq, ATAC-seq, DNase-seq, ChIP-seq, Hi-C 등.
* 총 11개의 유전체 기능 모달리티를 포함.
* Supplementary Table 1, 2에 세부 목록 포함.

####  3. 트레이닝 전략

* **단계 1: Pre-training**

  * 4-fold cross-validation 방식으로 훈련된 teacher 모델들을 생성.
  * 각 fold는 참조 유전체의 3/4을 학습용, 1/4을 검증용으로 사용.
* **단계 2: Distillation**

  * \*\*학생 모델(student model)\*\*이 모든 fold의 teacher 예측 출력을 따라 하도록 훈련.
  * 입력 DNA는 변형(shift, reverse complement, mutation)을 통해 다양화.
  * 단일 모델이 모든 모달리티와 세포 타입에 대해 예측 가능하게 설계.
  * 이 distillation은 예측의 일관성과 변이 효과 예측(VEP) 성능을 높임.

####  4. 변이 효과 예측 방식

* 참조(reference)와 변이(mutated) 서열에 대해 예측 차이를 비교하여 효과를 계산.
* 스플라이싱, 발현량, 접근성 등 각 모달리티별 맞춤 스코어를 설계.
* 모델 예측값은 주로 1초 내외(NVIDIA H100 GPU 기준)로 처리 가능.

---


####  1. Model Architecture

* **Input**: 1 megabase (1Mb) of human or mouse DNA sequence.
* **Output**: Predictions across 11 genomic modalities — 5,930 human and 1,128 mouse genomic tracks.
* **Backbone**: U-Net–style architecture.

  * **Encoder** with convolutional layers to capture local sequence patterns.
  * **Transformer blocks** to model long-range dependencies such as enhancer-promoter interactions.
  * **Decoder** to upsample sequence resolution.
  * **Task-specific output heads** for modality-specific resolutions (1bp, 128bp, 2048bp).
  * **2D embeddings** used for modeling chromatin contact maps (2048bp resolution).

####  2. Training Data

* Experimental genomic tracks from human and mouse genomes.

  * Modalities include RNA-seq, ATAC-seq, DNase-seq, ChIP-seq, PRO-cap, Hi-C, etc.
* Tracks span 11 different modalities, described in Supplementary Tables 1 and 2.

####  3. Training Procedure

* **Stage 1: Pre-training**

  * 4-fold cross-validation generates fold-specific teacher models.
  * Each model trains on 3/4 of the genome and validates on the held-out 1/4.
* **Stage 2: Distillation**

  * A **single student model** is trained to mimic predictions from all-fold teacher ensembles.
  * Inputs are randomly augmented via shifting, reverse complementation, and mutation.
  * The distilled model can predict all modalities and cell types efficiently in a single forward pass.
  * This process improves prediction robustness and variant effect prediction (VEP) accuracy.

####  4. Variant Effect Prediction

* VEP is performed by comparing model predictions between reference and alternative alleles.
* Modality-specific scoring strategies are applied (e.g., for splicing, expression).
* Highly efficient: predictions take <1 second on an NVIDIA H100 GPU.




   
 
<br/>
# Results  



####  1. **테스트 데이터 및 벤치마크**

* 총 **24개 genome track 예측** 벤치마크와 **26개 variant effect prediction (VEP)** 벤치마크 사용.
* **유전체 트랙 벤치마크**:

  * 예측 대상: RNA-seq, CAGE, PRO-cap, splice sites, junctions, usage, chromatin accessibility (DNase, ATAC), histone marks, TF binding, Hi-C contact maps 등.
  * **인간 및 생쥐 유전체**에서 다양한 cell/tissue type 기반.
  * 모델은 **held-out genome intervals**에 대해 평가.
* **변이 효과 예측 벤치마크**:

  * GTEx eQTL, sQTL, paQTL, ClinVar, MPRA (e.g. MFASS), CRISPRi-based enhancer-gene linking 등 포함.
  * zero-shot, supervised, fine-mapping 기반 벤치마크 다양하게 포함.

####  2. **경쟁 모델 (Baselines)**

* 단일 모달리티 특화 모델들:

  * **SpliceAI** (splicing),
  * **ChromBPNet** (chromatin accessibility),
  * **Orca** (3D contact maps),
  * **DeltaSplice**, **Pangolin**, 등.
* 멀티모달 모델들:

  * **Borzoi**, **Enformer**, **DeepSEA**, **Basenji**, **Sei**.
* AlphaGenome은 이들 모두와 비교됨.

####  3. **사용된 평가 지표 (Metrics)**

* **Pearson correlation** (유전체 트랙 예측 정확도),
* **Area under the Precision-Recall Curve (auPRC)**,
* **Area under the ROC Curve (auROC)**,
* **Spearman correlation (ρ)** (예: eQTL 효과 크기),
* Sign accuracy (eQTL 방향성 예측),
* MFASS 실험 데이터의 정답 유무 등.

####  4. **AlphaGenome의 주요 성과**

* **24개 genome track 예측 중 22개에서 SOTA 달성.**
* **26개 변이 예측 중 24개에서 SOTA 달성.**
* 예시 성과:

  * eQTL 방향 예측 → **+25.5% 성능 향상** (vs Borzoi),
  * splice junction 예측 → **ClinVar에서 auPRC 0.66**, Pangolin보다 우수,
  * contact map → Orca 대비 **+42.3% 향상** (cell-type specific difference),
  * MFASS 실험 벤치마크에서도 일부 모델보다 우수.

---


####  1. **Test Datasets & Benchmarks**

* **24 genome track prediction benchmarks** and **26 variant effect prediction (VEP) benchmarks** were used.
* **Genome track prediction**:

  * Tasks include RNA-seq, CAGE, PRO-cap, splice sites/junctions/usage, DNase, ATAC, histone marks, TF binding, and Hi-C contact maps.
  * Evaluations done on **held-out genome intervals** from human and mouse.
* **VEP Benchmarks**:

  * Include fine-mapped **GTEx eQTLs**, **sQTLs**, **paQTLs**, **ClinVar pathogenicity**, **MPRA datasets** (e.g., MFASS), and **CRISPRi enhancer-gene validation** (ENCODE-rE2G).
  * Tasks span **zero-shot**, **supervised**, and **causality inference**.

####  2. **Baseline Models**

* **Modality-specific models**:

  * *SpliceAI* (splicing),
  * *ChromBPNet* (accessibility),
  * *Orca* (3D chromatin contacts),
  * *DeltaSplice*, *Pangolin*, etc.
* **Multimodal models**:

  * *Borzoi*, *Enformer*, *DeepSEA*, *Basenji*, *Sei*.
* AlphaGenome was benchmarked against all these models.

####  3. **Evaluation Metrics**

* **Pearson correlation** (for track prediction accuracy),
* **Area under Precision-Recall Curve (auPRC)**,
* **Area under ROC Curve (auROC)**,
* **Spearman ρ** (for effect size correlations),
* **Sign prediction accuracy** (direction of eQTL effects),
* Classification accuracy (e.g., ClinVar variant classification).

####  4. **Key Results of AlphaGenome**

* **Achieved state-of-the-art (SOTA)** on:

  * **22 of 24** genome track tasks,
  * **24 of 26** variant effect prediction tasks.
* Examples:

  * eQTL sign prediction: **+25.5% improvement** over Borzoi,
  * ClinVar splice variants: **auPRC 0.66** vs Pangolin’s 0.64,
  * Contact map prediction: **+42.3% improvement** vs Orca (cell type-specific),
  * Strong performance on MPRA (CAGI5), outperforming ChromBPNet and Borzoi.




<br/>
# 예제  





####  1. **트레이닝 데이터 실제 예**

* 모델은 **인간/생쥐의 1Mb 유전체 서열**을 입력으로 받아, 해당 구간에 대한 다양한 **실험 기반 유전체 트랙**을 예측하도록 학습됨.
* **예측 대상 트랙**(output label)은 실험 데이터를 기반으로 한 다음 항목들:

  * RNA-seq (유전자 발현량, +/– strand별),
  * Splice site 위치 및 사용도, Splice junction coverage,
  * ATAC-seq, DNase-seq (염색질 접근성),
  * ChIP-seq (히스톤 변형, TF 결합),
  * Hi-C (contact map 형태의 2D 상호작용 매트릭스).
* 예: `chr19:10587331-11635907` 구간의 HepG2 세포주 실험 데이터를 학습 라벨로 사용 (그림 2a).

####  2. **테스트 데이터 실제 예**

* **Held-out genome interval**에서 예측 성능을 측정.
* 예시: `chr19:11086619-11136619` 영역에서 **LDLR 유전자** 주변의 스플라이싱 이벤트 예측.

  * RNA-seq 커버리지 예측, 스플라이스 donor/acceptor 예측, splice site usage 예측, junction 예측이 실제 RNA-seq과 잘 일치함 (그림 2b).

####  3. **제안 모델의 인풋/아웃풋 예시**

* **입력**:

  * DNA 염기서열 1Mb (e.g., "AGTCAGTC..."),
  * 생물종 정보 (인간/마우스),
  * 실험별로 세포 유형 선택 가능 (e.g., HepG2, CD34+ CMP 등).
* **출력**:

  * 각 염기에 대한 예측 값들 (1bp 단위 예: RNA-seq, ATAC, DNase),
  * 특정 resolution의 예측 (128bp 히스톤 마크, 2048bp contact maps),
  * 예: RNA-seq에서 특정 유전자의 엑손 위치에서 발현량 상승 예측, contact map에서 특정 영역 간 3D 상호작용 예측 등.

####  4. **변이 예측 실제 예시**

* 예: chr3:197081044\:TACTC>T 변이는 **DLG1 유전자에서 엑손 스킵 현상** 유발.

  * AlphaGenome은 splice site usage 감소, 새로운 junction 등장, RNA-seq 커버리지 감소를 정확히 예측함 (그림 3b).
* 예: chr22:36201698\:A>C 변이는 **APOL4 유전자 발현 감소**와 연관된 eQTL.

  * 모델은 RNA-seq 커버리지 감소와 함께 스플라이싱 변화를 예측 (그림 4b).

---


####  1. **Training Data Example**

* Input DNA segments of **1 megabase (1Mb)** from human or mouse genomes.
* **Supervision targets** are experimentally measured genomic tracks:

  * RNA-seq (expression levels, strand-specific),
  * Splice sites, splice junctions, and their usage,
  * ATAC-seq, DNase-seq (chromatin accessibility),
  * ChIP-seq (histone marks, TF binding),
  * Hi-C (3D chromatin contact maps).
* Example: **HepG2 cell line** data over the region `chr19:10587331–11635907` is used for supervised learning (Figure 2a).

####  2. **Test Data Example**

* Evaluation is performed on **held-out intervals** of the genome.
* Example: Region `chr19:11086619–11136619` includes the **LDLR gene**.

  * Predictions include RNA-seq coverage, splice site activity, splice site usage, and junction usage — all matching the experimental data (Figure 2b).

####  3. **Model Input/Output Example**

* **Input**:

  * A 1Mb DNA sequence (e.g., “AGTCAGTC...”),
  * Species identifier (human or mouse),
  * Cell-type metadata (e.g., HepG2, CD34+ CMP).
* **Output**:

  * Predictions at 1bp resolution (RNA-seq, ATAC, DNase),
  * Predictions at 128bp or 2048bp resolution (histone marks, Hi-C),
  * Example: Predicted increased RNA-seq coverage at exon boundaries, or high contact score in 3D maps between distal genomic loci.

####  4. **Variant Prediction Example**

* **chr3:197081044\:TACTC>T**: causes exon skipping in the **DLG1** gene.

  * AlphaGenome correctly predicts reduced splice site usage, lost junctions, new junctions, and decreased RNA-seq signal (Figure 3b).
* **chr22:36201698\:A>C**: a known eQTL affecting **APOL4** expression.

  * Model predicts decreased RNA expression and splicing disruption consistent with GTEx observations (Figure 4b).




<br/>  
# 요약   


AlphaGenome은 1Mb DNA 서열을 입력으로 받아 RNA 발현, 스플라이싱, 염색질 접근성 등 11개 모달리티를 단일 모델로 예측하는 U-Net 기반 멀티모달 모델이다.
24개 유전체 트랙과 26개 변이 예측 벤치마크 중 각각 22개, 24개에서 기존 최고 모델을 능가하며, GTEx eQTL 및 ClinVar 변이 등 실제 생물학적 예시에서도 정확한 결과를 보였다.
입력은 DNA 염기서열과 생물종 정보이며, 출력은 실험 기반 트랙의 염기 단위 예측값으로 구성된다.


AlphaGenome is a U-Net–based multimodal model that takes a 1Mb DNA sequence as input and predicts 11 functional genomic modalities, including RNA expression, splicing, and chromatin accessibility.
It outperforms prior models on 22 of 24 genome track tasks and 24 of 26 variant effect benchmarks, showing accurate predictions on real-world examples like GTEx eQTLs and ClinVar mutations.
The input is a DNA sequence with species metadata, and the output consists of base-resolution predictions across genomic tracks.


<br/>  
# 기타  



####  **Figure 1: AlphaGenome 모델 구조 및 전체 평가 요약**

* **내용**: 모델 아키텍처(a), pretraining(b), distillation(c), 트랙 예측(d), 변이 예측(e) 전반을 요약.
* **의미**: AlphaGenome의 입력-출력 구조, 학습 방식, 그리고 다양한 modality별 성능 향상을 시각화함.
* **결과**: 22/24개의 유전체 트랙과 24/26개의 변이 효과 예측에서 SOTA(최고 성능) 성능 달성.

####  **Figure 2: 유전체 트랙 예측 예시 및 정량 평가**

* **내용**: `chr19` 구간의 RNA-seq, ATAC, DNase, H3K27ac, CTCF 등의 실험값과 예측값 비교 (a), LDLR 유전자 영역 확대 시각화 (b), Pearson 상관계수 분포 (c), 조직 특이적 발현 예측 (d), 스플라이싱 junction 예측 (e).
* **의미**: AlphaGenome이 실제 염기서열 상에서 실험값과 정밀하게 일치하는 트랙 예측을 한다는 증거.
* **결과**: RNA-seq 및 스플라이싱 예측 정확도가 높고, 다양한 생물종 및 조직에서도 높은 상관도 유지.

####  **Figure 3: 스플라이싱 변이 예측 분석**

* **내용**: AlphaGenome vs. 다른 모델들의 스플라이싱 관련 예측 항목 비교 (a), GTEx에서 관측된 변이 사례 예측 (b–c), ISM 분석 (d), 변이 점수 계산 방식(e), 벤치마크 비교 성능(f–i).
* **의미**: AlphaGenome은 splice site, usage, junction 모두를 예측 가능하며, 실제 변이 사례에서도 정확한 기능적 결과를 예측함.
* **결과**: sQTL, ClinVar, MFASS 등 주요 벤치마크에서 Pangolin, DeltaSplice 등을 능가함.

####  **Figure 4: 유전자 발현 관련 변이 예측**

* **내용**: eQTL 효과 예측 방법 (a), 실제 eQTL 예시 (b), 예측 vs 관측 비교 (c–d), eQTL 방향성, 거리별 분석 (e–g), GWAS 적용 범위 (h), 인과성 평가 (i), enhancer–gene 연결 예측 (j), paQTL 예측 성능 (k).
* **의미**: AlphaGenome은 발현량 예측뿐 아니라 GWAS 해석, enhancer 연결까지 확장 가능함.
* **결과**: eQTL 방향 예측 정확도 +25.5%, GWAS 변이 중 49%에 방향성 할당 가능.

####  **Supplementary Tables/Figures**

* **내용**: Supplementary Table 1–5: 사용된 데이터 목록, 정확한 수치, 실험 설정, 세부 평가 기준 등.
* **의미**: 재현성 및 확장성 확보, 세부 성능 비교 근거 제공.
* **결과**: 모든 수치는 정량화되어 통계적으로 비교 가능함.

---


####  **Figure 1: Model Architecture and Overall Performance**

* **What it shows**: Overview of AlphaGenome’s architecture (a), pretraining (b), distillation (c), and relative performance in genome track prediction (d) and variant effect prediction (e).
* **What it means**: Demonstrates AlphaGenome’s unified structure and training approach, and its superior performance across most tasks.
* **Key findings**: Outperforms baselines in 22/24 track tasks and 24/26 variant tasks.

####  **Figure 2: Genome Track Prediction Examples and Metrics**

* **What it shows**: Predicted vs observed tracks across chr19 (a), zoom into LDLR region (b), correlation across modalities (c), gene expression prediction (d), and splice junction prediction (e).
* **What it means**: Confirms AlphaGenome’s ability to precisely match experimental measurements across cell types and organisms.
* **Key findings**: High Pearson correlation and accurate base-resolution predictions across modalities.

####  **Figure 3: Splicing Variant Effect Predictions**

* **What it shows**: Comparison with other models (a), specific variant examples from GTEx (b–c), in silico mutagenesis (d), scoring strategy (e), and benchmark comparisons (f–i).
* **What it means**: AlphaGenome uniquely predicts all splicing dimensions and accurately identifies pathogenic variants.
* **Key findings**: Outperforms Pangolin and DeltaSplice on sQTL, ClinVar, and MFASS benchmarks.

####  **Figure 4: Gene Expression Variant Effects**

* **What it shows**: eQTL scoring (a), real example (b), effect size and direction prediction (c–f), Sign accuracy (g), GWAS variant coverage (h), causal inference (i), enhancer-gene links (j), and paQTL prediction (k).
* **What it means**: AlphaGenome generalizes well to regulatory tasks and supports mechanistic interpretation of variants.
* **Key findings**: +25.5% improvement in eQTL sign prediction, 49% of GWAS loci resolved for effect direction.

####  **Supplementary Tables/Figures**

* **What they show**: Detailed benchmark datasets, metric scores, model ablations, and cross-validation settings.
* **What they mean**: Ensure full reproducibility and transparency of the experiments.
* **Key findings**: All numerical evaluations are statistically grounded and benchmark-comparable.




<br/>
# refer format:     


@article{Avsec2025AlphaGenome,
  title     = {AlphaGenome: Advancing Regulatory Variant Effect Prediction with a Unified DNA Sequence Model},
  author    = {Žiga Avsec and Natasha Latysheva and Jun Cheng and Guido Novati and Kyle R. Taylor and Tom Ward and Clare Bycroft and Lauren Nicolaisen and Eirini Arvaniti and Joshua Pan and Raina Thomas and Vincent Dutordoir and Matteo Perino and Soham De and Alexander Karollus and Adam Gayoso and Toby Sargeant and Anne Mottram and Lai Hong Wong and Pavol Drotár and Adam Kosiorek and Andrew Senior and Richard Tanburn and Taylor Applebaum and Souradeep Basu and Demis Hassabis and Pushmeet Kohli},
  year      = {2025},
  journal   = {bioRxiv},
  doi       = {10.1101/2025.06.25.661532},
  note      = {Preprint},
  url       = {https://doi.org/10.1101/2025.06.25.661532}
}




Avsec, Žiga, Natasha Latysheva, Jun Cheng, Guido Novati, Kyle R. Taylor, Tom Ward, Clare Bycroft, Lauren Nicolaisen, Eirini Arvaniti, Joshua Pan, Raina Thomas, Vincent Dutordoir, Matteo Perino, Soham De, Alexander Karollus, Adam Gayoso, Toby Sargeant, Anne Mottram, Lai Hong Wong, Pavol Drotár, Adam Kosiorek, Andrew Senior, Richard Tanburn, Taylor Applebaum, Souradeep Basu, Demis Hassabis, and Pushmeet Kohli.
"AlphaGenome: Advancing Regulatory Variant Effect Prediction with a Unified DNA Sequence Model." bioRxiv (2025). https://doi.org/10.1101/2025.06.25.661532.   

