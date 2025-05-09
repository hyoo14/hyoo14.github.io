---
layout: post
title:  "[2024]DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA"  
date:   2025-05-09 15:28:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

벤치마크 제안 페이퍼인데.. 제로샷, 프로빙(모델 프리징 및 최종 레이어만 튜닝), 파인튜닝 세팅을 잘 한 듯..?   
클러스터링이나 임베딩간 거리 기반 평가도 제안하고(스피어맨 코릴레이션이랑 같이 평가하는..)   



짧은 요약(Abstract) :    


이 논문에서는 최근 발전한 DNA 언어 모델(DNALMs)을 체계적으로 평가할 수 있는 새로운 벤치마크인 DART-Eval을 소개합니다. 기존의 평가 기준은 유전자 발현을 조절하는 비부호화 조절 DNA 영역에 대한 모델의 성능을 제대로 측정하지 못했습니다. 이를 보완하기 위해 DART-Eval은 zero-shot, probe, fine-tuning 환경에서 DNALM의 성능을 다양한 태스크에 대해 평가합니다.

이 태스크들은:

전사 인자 결합 모티프 탐색

세포 유형 특이적 조절 활성 예측

유전 변이의 반사실적 예측(counterfactual prediction) 등입니다.

실험 결과, 현재의 DNALM들은 일관되지 않은 성능을 보였고, 많은 계산 자원이 요구됨에도 불구하고 대부분의 태스크에서 기존의 간단한 ab initio 모델보다 성능이 뛰어나지 않았습니다. 저자들은 향후 모델링, 데이터 구축, 평가 전략의 방향성에 대해 논의하며, DART-Eval 벤치마크 도구 및 데이터셋을 오픈소스로 제공합니다.


This paper introduces DART-Eval, a new benchmarking suite to systematically evaluate DNA language models (DNALMs), especially on non-coding regulatory DNA, which governs gene activity. Existing benchmarks fall short in assessing how well DNALMs capture the function of regulatory sequences.

DART-Eval includes tasks such as:

Transcription factor motif detection

Cell-type specific regulatory activity prediction

Counterfactual predictions of genetic variants

The benchmark evaluates models under zero-shot, probed, and fine-tuned settings, comparing them against ab initio models. Results show that current DNALMs exhibit inconsistent performance and often fail to outperform much simpler models, despite their higher computational demands. The paper concludes with suggestions for improving future models, data, and evaluation strategies, and publicly releases the full benchmark.



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



다음은 DART-Eval 논문에서 사용된 **메서드(Method)** 중 백본 아키텍처, 토크나이징 전략, 트레이닝 데이터에 대한 설명입니다. 논문에서는 여러 DNALMs를 비교 실험하며, 각 모델의 사전학습 방식과 구조, 그리고 벤치마크 방법론을 상세히 설명합니다.

---


#### 1. **백본 아키텍처 (Backbone Architectures)**

DART-Eval에서는 다음과 같은 다양한 DNALM들을 평가합니다:

* **DNABERT-2**:

  * Masked Language Modeling(Masked LM) 방식
  * Byte-Pair Encoding (BPE) 토크나이저 사용
  * Transformer 기반 구조
  * 멀티스피시즈 사전학습 (10kb 컨텍스트 길이)

* **Nucleotide Transformer (NT)**:

  * Masked LM 방식
  * Non-overlapping k-mer 토크나이징 (v2 모델은 6-mer 등)
  * Transformer 기반
  * 멀티스피시즈 사전학습, 12kb 컨텍스트 길이

* **HyenaDNA**:

  * Autoregressive 방식
  * 단일 뉴클레오타이드 토크나이징
  * Hyena 구조 (서브쿼드러틱 시간복잡도의 implicit long convolutions)
  * 휴먼 단일 종 기준, 1Mbp까지의 길이 모델링 가능

* **Caduceus**:

  * Masked LM 방식
  * 단일 뉴클레오타이드 토크나이징
  * Bi-directional equivariant 구조 사용
  * 131kbp 컨텍스트 길이

* **Mistral-DNA**:

  * Autoregressive 방식
  * Byte-Pair Encoding 사용
  * 1.6B 파라미터 Transformer 구조
  * 휴먼 기반 10kb 컨텍스트 길이

* **GENA-LM**:

  * Masked LM 방식
  * Byte-Pair Encoding
  * Transformer 기반, 336M 파라미터
  * 멀티스피시즈 학습, 4.5kb 컨텍스트

#### 2. **사전학습 데이터 (Pretraining Data)**

* DNALMs는 주로 **인간 유전체 전체** 또는 **다종(multi-species)** 유전체를 대상으로 사전학습됨.
* 유전체 전체에서 annotation 없이 학습하며, regulatory 영역에 대한 태스크에서는 annotation-free 상태에서 평가됨.

#### 3. **학습 방식 및 평가 (Training Strategy & Evaluation)**

* **Zero-shot**:
  사전학습된 모델에서 직접 추론 (e.g., likelihood or embedding distance)

* **Probed**:
  모델을 고정(frozen)하고 최종 히든 레이어 위에 단순한 분류기(CNN)만 학습

* **Fine-tuned**:
  LoRA를 활용한 파라미터 효율적 미세조정 (선택된 파라미터만 학습)

* **Baselines**:
  ChromBPNet (ab initio CNN) 및 probing-head 스타일의 ab initio 분류기를 비교 기준으로 사용

---



#### 1. **Backbone Architectures**

DART-Eval evaluates several distinct DNA Language Models (DNALMs):

* **DNABERT-2**:

  * Pretrained using Masked Language Modeling
  * Byte-Pair Encoding tokenizer
  * Transformer-based, 117M parameters
  * Multi-species genome training, max context \~10kb

* **Nucleotide Transformer (NT)**:

  * Masked LM with non-overlapping k-mer tokenization
  * Transformer-based architecture (v2 version: 500M parameters)
  * Multi-species training, context length \~12kb

* **HyenaDNA**:

  * Autoregressive objective
  * Single nucleotide tokenizer
  * Hyena architecture (implicit long convolutions with sub-quadratic complexity)
  * Human genome only, 1Mbp context

* **Caduceus**:

  * Masked LM objective
  * Single nucleotide tokens
  * Bi-directional equivariant long-range architecture
  * Context length \~131kbp

* **Mistral-DNA**:

  * Autoregressive Transformer (1.6B parameters)
  * Byte-Pair Encoding
  * Human genome, 10kb context

* **GENA-LM**:

  * Masked LM, Transformer-based
  * Byte-Pair Encoding, 336M parameters
  * Trained on multi-species data with 4.5kb context

#### 2. **Pretraining Datasets**

* Models are pretrained on full genomes (either human or multi-species) without genomic annotations.
* Their regulatory performance is evaluated **annotation-agnostically**, relying solely on raw sequence learning.

#### 3. **Training Strategies & Evaluation Settings**

* **Zero-shot**:
  Models are evaluated using output log-likelihoods or embedding distances directly, without any task-specific training.

* **Probed**:
  The final hidden layer is frozen and used to train a shallow CNN classifier.

* **Fine-tuned**:
  Models are fine-tuned using parameter-efficient LoRA on selected layers.

* **Baselines**:
  Compared against **ab initio supervised models** like ChromBPNet and custom CNNs designed for regulatory sequence prediction.

---




   
 
<br/>
# Results  




### 1. **테스트 데이터**

* **총 5개 태스크**로 구성됨:

  1. 조절 DNA와 컨트롤 구분
  2. TF 모티프 민감도 평가 (1443개 모티프)
  3. 세포 유형 특이적 조절 요소 구분 (5개 세포주: GM12878, H1ESC, HEPG2, IMR90, K562)
  4. 정량적 조절 활성 예측 (DNase-seq 기반)
  5. 유전 변이의 반사실적 효과 예측 (caQTL, dsQTL)

* 데이터 출처:

  * ENCODE cCRE (350bp, 2.3M)
  * HOCOMOCO v12 TF motif DB
  * DNase-seq 및 ATAC-seq 실험 데이터
  * QTL 데이터 (African, Yoruban 유래)

---

### 2. **평가지표 (Metrics)**

* **Classification**: Accuracy, AUROC, AUPRC
* **Regression**: Pearson r, Spearman r
* **Ranking**: Paired accuracy (positive vs. control)
* **Clustering**: Adjusted Mutual Information Score (AMI)
* **Counterfactual variant tasks**: AUROC + correlation between predicted and observed variant effect

---

### 3. **경쟁 모델들 (Models Compared)**

#### DNALMs (총 6개)

* DNABERT-2
* Nucleotide Transformer (NT v2)
* GENA-LM
* HyenaDNA
* Mistral-DNA
* Caduceus

#### Ab Initio Baselines

* ChromBPNet (회귀, base-resolution)
* CNN probing-head-like 모델

---

### 4. **결과 요약 (대표 결과)**

| Task       | DNALM 성능                                 | Baseline 성능                    | 비고                           |
| ---------- | ---------------------------------------- | ------------------------------ | ---------------------------- |
| 조절 DNA 판별  | 모든 DNALM이 zero-shot으로도 좋은 성능 보임          | ab initio CNN과 유사              | Fine-tuning 시 향상             |
| TF 모티프 민감도 | Likelihood 기반 성능은 양호하나, embedding 기반은 부진 | -                              | DNALM이 일부 모티프만 학습함           |
| 세포 특이 조절   | Fine-tuned DNALM만 경쟁력 있음                 | ab initio CNN이 probing 모델보다 우수 |                              |
| 정량적 예측     | Fine-tuned DNALM이 준수함                    | ChromBPNet과 유사, 때론 열세          | Probed는 부진                   |
| 유전변이 효과 예측 | 모든 DNALM 부진                              | ChromBPNet이 압도적 우위             | Counterfactual task에서 한계 드러남 |

---



### 1. **Test Data**

DART-Eval evaluates DNALMs across 5 tasks using:

* **ENCODE cCREs** (2.3 million 350bp sequences)
* **HOCOMOCO v12** TF motifs (1443 motifs)
* **Cell-type regulatory peaks** from ATAC/DNase-seq (5 cell types)
* **QTL variant datasets** (African, Yoruban populations)

---

### 2. **Evaluation Metrics**

* **Binary classification**: Accuracy, AUROC, AUPRC
* **Regression**: Pearson and Spearman correlation
* **Ranking**: Paired accuracy (positive vs. control sequences)
* **Clustering**: Adjusted Mutual Information (AMI)
* **Variant effect prediction**: AUROC, AUPRC, Pearson r with ground-truth effect size

---

### 3. **Compared Models**

#### DNALMs:

* DNABERT-2
* Nucleotide Transformer (NT v2)
* GENA-LM
* HyenaDNA
* Mistral-DNA
* Caduceus

#### Baselines:

* **ChromBPNet**: A high-resolution base-level CNN
* **Ab initio probing-head-like CNN**

---

### 4. **Main Findings**

| Task                               | DNALMs                                                       | Baseline                                 | Notes                                                 |
| ---------------------------------- | ------------------------------------------------------------ | ---------------------------------------- | ----------------------------------------------------- |
| Regulatory sequence discrimination | All DNALMs perform well in zero-shot                         | Similar to ab initio CNN                 | Fine-tuning improves further                          |
| TF motif detection                 | Likelihood-based scores effective; embeddings perform poorly | -                                        | DNALMs capture only frequent motifs                   |
| Cell-type specific regulation      | Only fine-tuned DNALMs competitive                           | Baseline CNN outperforms probed DNALMs   |                                                       |
| Quantitative chromatin activity    | Fine-tuned DNALMs perform well                               | On par or slightly worse than ChromBPNet | Probed models are weak                                |
| Variant effect prediction          | DNALMs underperform                                          | ChromBPNet significantly better          | Major limitation exposed in counterfactual prediction |








<br/>
# 예제  





###  예시 1: **Task 1 – 조절 DNA 구분**

* **트레이닝/테스트 데이터**

  * **양성 (Positive)**: ENCODE에서 정의한 2.3백만 개의 cCREs (candidate cis-regulatory elements), 길이 350bp
  * **음성 (Negative)**: 위의 cCREs를 di-nucleotide 빈도를 유지한 채 무작위로 섞은 시퀀스
* **입력 예시 (input)**

  ```
  NNNGATTACAAGTCGT... (총 350bp)
  ```
* **출력 예시 (output)**

  * 1 (조절 시퀀스) vs 0 (배경 시퀀스)
* **학습 방식**

  * Zero-shot: 두 시퀀스 중 log-likelihood가 더 높은 쪽을 조절 엘리먼트로 판단
  * Probing/Fine-tuning: CNN을 붙여 supervised classification 수행

---

###  예시 2: **Task 2 – TF Motif 감지**

* **테스트 데이터**

  * 1443개의 TF 모티프(HOCOMOCO v12)
  * 각 모티프를 di-nucleotide shuffled background에 삽입하여 positive 시퀀스 구성
  * Negative는 모티프 순서를 섞거나 reverse complement하여 구성
* **입력 예시**

  * Positive: `NNNNN[ACTG]Motif[ACTG]NNNNN`
  * Negative: `NNNNN[ACTG]ShuffledMotif[ACTG]NNNNN`
* **출력 예시**

  * log-likelihood 값 또는 embedding distance
  * 예: `logP(pos) = -10.2`, `logP(neg) = -11.6` → 모티프 감지 성공

---

###  예시 3: **Task 3 – 세포 특이적 조절 구분**

* **데이터**

  * ATAC-seq 데이터로부터 유래된 5개 세포주(GM12878, H1ESC, HEPG2, IMR90, K562)에서 특이적 활성을 보이는 엘리먼트
  * 총 약 25만 개 시퀀스, 클래스별로 균형 맞춤
* **입력**

  * 350bp 조절 시퀀스
* **출력**

  * 5개 세포주 중 하나로 분류

---

###  예시 4: **Task 4 – 정량적 조절 활성 예측**

* **데이터**

  * DNase-seq 측정값을 갖는 2kb 길이 시퀀스, G/C 조성 매칭된 negative 포함
* **입력 예시**

  ```
  DNA sequence (2000bp): AGCTGATTCC... 
  ```
* **출력 예시**

  * 로그 정규화된 DNase read count 값 (예: 5.32)
  * 또는 classification 용도로 이 값을 기준으로 binary label 부여 (High vs. Low accessibility)

---

###  예시 5: **Task 5 – 유전변이 효과 예측**

* **데이터**

  * QTL 실험 기반으로 정해진 변이 위치 (caQTL, dsQTL)
  * Positive: 활성을 변화시키는 known variant
  * Negative: 같은 엘리먼트 내에 있으나 효과가 없는 변이
* **입력 예시**

  * REF allele 시퀀스: `...ACGTCTGAT[C]TGGAC...`
  * ALT allele 시퀀스: `...ACGTCTGAT[A]TGGAC...`
* **출력 예시**

  * 예측된 활성 차이 (ΔActivity)

    * `Activity_REF = 5.2`, `Activity_ALT = 3.7`, Δ = -1.5
  * 또는 분류 결과 (유의미 vs 무의미 변이)

---


###  Example 1: **Task 1 – Discriminating Regulatory DNA**

* **Training/Test Data**

  * **Positive**: 2.3M ENCODE cCREs (350bp)
  * **Negative**: Dinucleotide-shuffled versions of the same
* **Input Example**

  ```
  NNNGATTACAAGTCGT... (350bp)
  ```
* **Output**

  * 1 (regulatory) or 0 (background)
* **Evaluation**

  * Zero-shot: Based on likelihood comparison
  * Supervised: Trained CNN classifier using model embeddings

---

###  Example 2: **Task 2 – TF Motif Sensitivity**

* **Test Data**

  * 1443 motifs from HOCOMOCO v12
  * Positive = motif embedded in shuffled cCRE background
  * Negative = shuffled or reversed motifs
* **Input**

  * Positive: `NNNNN[ACTG]Motif[ACTG]NNNNN`
  * Negative: `NNNNN[ACTG]ShuffledMotif[ACTG]NNNNN`
* **Output**

  * log-likelihood score or embedding distance
  * e.g., `logP(pos) = -10.2`, `logP(neg) = -11.6` → correct prediction

---

###  Example 3: **Task 3 – Cell-Type Specific Regulation**

* **Data**

  * ATAC-seq derived peaks unique to 5 cell types
* **Input**

  * 350bp DNA sequences
* **Output**

  * Classification into one of 5 cell types (e.g., "K562")

---

###  Example 4: **Task 4 – Quantitative Activity Regression**

* **Data**

  * 2kb sequences with measured DNase-seq signal
* **Input**

  ```
  DNA sequence (2000bp): AGCTGATTCC... 
  ```
* **Output**

  * Log-transformed DNase count (e.g., 5.32)
  * Or binary: high vs low accessibility

---

###  Example 5: **Task 5 – Variant Effect Prediction**

* **Data**

  * QTL-identified functional variants in accessible chromatin regions
* **Input**

  * REF sequence: `...ACGTCTGAT[C]TGGAC...`
  * ALT sequence: `...ACGTCTGAT[A]TGGAC...`
* **Output**

  * ΔActivity = Predicted\_ALT − Predicted\_REF (e.g., -1.5)
  * Or classification: causal vs. neutral variant

---




<br/>  
# 요약   



DART-Eval은 다양한 DNA 언어 모델(DNALM)을 평가하기 위해 설계된 벤치마크로, 조절 DNA 판별, TF 모티프 감지, 변이 예측 등 5가지 태스크로 구성된다. 모델은 사전학습만 된 상태(zero-shot), probing, fine-tuning 설정에서 평가되며, 기존의 간단한 ab initio 모델들이 DNALM보다 우수한 성능을 보이기도 했다. 각 태스크는 실제 cCRE, ATAC-seq, QTL 데이터 등을 사용하며, 입력은 350\~2000bp DNA 시퀀스, 출력은 분류 결과나 활성 수치 등으로 구성된다.

---



DART-Eval is a benchmark suite for evaluating DNA language models (DNALMs) across five tasks such as regulatory sequence detection, motif recognition, and variant effect prediction. Models are tested in zero-shot, probed, and fine-tuned settings, with simpler ab initio models sometimes outperforming DNALMs. Tasks use real datasets (e.g., cCREs, ATAC-seq, QTLs) with DNA sequences as input (350–2000bp) and outputs like class labels or activity scores.



<br/>  
# 기타  



좋습니다. 아래는 DART-Eval 논문에 포함된 **기타 구성요소**(피규어, 테이블, 어펜딕스 등)와 그로부터 얻을 수 있는 **핵심 인사이트 및 결과 분석 요약**입니다.

---



###  1. **Figure 1: 전체 태스크 구조 요약도**

* 각 DNALM이 수행하는 태스크 유형(분류, 회귀, zero-shot 등)을 시각화함.
* TF motif 감지, regulatory vs control, variant effect 등 다양한 입력/출력 예시가 함께 제시되어 전반적 파이프라인 이해에 도움.

###  2. **Figure 3, 4: TF 모티프/세포주 구분 성능**

* Figure 3: DNALM은 likelihood 기반으로는 TF 모티프 감지가 가능하나, embedding 기반의 clustering은 실패.
* Figure 4: Cell-type 분류에서 DNALM의 embedding은 단순 motif-count 기반보다 낮은 분리력을 보임.

###  3. **Table 3\~6: 모델별 성능 정량 비교**

* Table 3: 조절 DNA 분류에서 fine-tuned DNALM이 일부 향상된 성능 보이나, probing 모델은 ab initio와 유사.
* Table 4: Cell-type classification에서 ChromBPNet이 probing 모델보다 명확히 우수함.
* Table 5: 정량적 활성 예측에서 fine-tuned DNALM이 괜찮은 성능이나, ChromBPNet이 전체적으로 안정적.
* Table 6: 변이 효과 예측에서는 모든 DNALM이 ChromBPNet에 비해 열세, 특히 zero-shot에서는 매우 낮은 AUROC.

###  4. **Appendix**

* Appendix B: 유전체 및 비부호화 조절 영역에 대한 생물학적 배경 설명
* Appendix C: 각 데이터셋 출처 및 전처리 방법 명시
* Appendix D: 모델 구조, 학습 설정, 하드웨어 자원 정리
* Appendix E: 평가 지표 세부 정의 및 추가 실험 결과 포함 (e.g., confidence interval, motif별 accuracy)

###  주요 인사이트

* DNALM은 시퀀스 구별력은 있으나, **TF 모티프 표현 학습에 한계**가 있음.
* fine-tuning을 해야만 좋은 성능이 나오며, **zero-shot 및 probing은 제한적**.
* **ChromBPNet은 간단하면서도 매우 강력한 baseline**이며, 특히 변이 예측에서 압도적 우위.
* BPE tokenizer는 단일 염기 변이 분석에 비직관적인 문제를 야기함.

---



###  1. **Figure 1: Overview of Task Types**

* Illustrates various evaluation settings (classification, regression, zero-shot) and input-output formats.
* Shows concrete examples of TF motif detection, regulatory/control classification, variant prediction.

###  2. **Figure 3, 4: TF Motif and Cell Type Visualization**

* Figure 3: DNALMs can detect motifs via likelihoods but **fail in embedding-based detection**.
* Figure 4: DNALM embeddings cluster poorly for cell-type regulatory regions, **worse than motif-count baseline**.

###  3. **Tables 3–6: Quantitative Comparisons**

* **Table 3**: Regulatory classification—fine-tuned DNALMs are decent; probing ≈ ab initio.
* **Table 4**: Cell-type classification—ChromBPNet outperforms all probing DNALMs.
* **Table 5**: Quantitative activity prediction—fine-tuned DNALMs nearly match ChromBPNet, but probed ones lag.
* **Table 6**: Variant effect prediction—**ChromBPNet dominates**, DNALMs underperform especially in zero-shot AUROC.

###  4. **Appendices**

* **Appendix B**: Biological background on non-coding regulatory elements.
* **Appendix C**: Dataset curation and preprocessing details.
* **Appendix D**: Model architectures, training setups, and compute resource usage.
* **Appendix E**: Metric definitions and extended results (e.g., CI, motif-wise breakdown).

###  Key Insights

* DNALMs distinguish sequences well but **struggle to learn rich motif representations**.
* **Fine-tuning is essential** for performance; probing and zero-shot are often weak.
* **ChromBPNet is a surprisingly strong baseline**, especially for counterfactual variant predictions.
* **Byte-Pair Encoding** hinders interpretation of single-nucleotide variants due to token fragmentation.




<br/>
# refer format:     


@inproceedings{patel2024dart,
  title={{DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA}},
  author={Patel, Aman and Singhal, Arpita and Wang, Austin and Pampari, Anusri and Kasowski, Maya and Kundaje, Anshul},
  booktitle={Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024), Datasets and Benchmarks Track},
  year={2024},
  url={https://github.com/kundajelab/DART-Eval}
}




Patel, Aman, Arpita Singhal, Austin Wang, Anusri Pampari, Maya Kasowski, and Anshul Kundaje. “DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA.” In Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024), Datasets and Benchmarks Track, 2024. https://github.com/kundajelab/DART-Eval.    

