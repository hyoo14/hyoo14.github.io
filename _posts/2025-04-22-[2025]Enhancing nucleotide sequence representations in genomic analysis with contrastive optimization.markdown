---
layout: post
title:  "[2020]Enhancing nucleotide sequence representations in genomic analysis with contrastive optimization"  
date:   2025-04-22 11:51:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


contrastive learning을 이용한 dna lm 학습  


짧은 요약(Abstract) :    





이 논문은 **Scorpio**라는 프레임워크를 소개하며, 기존의 alignment 기반 분석 방법들이 가진 한계를 극복하고자 **대조 학습(contrastive learning)** 기반으로 DNA 서열 임베딩을 최적화하는 방법을 제안합니다. Scorpio는 **6-mer 빈도 정보**와 **BigBird 기반의 임베딩**을 결합해 메타지놈 데이터를 빠르고 정확하게 분류할 수 있습니다. 이를 통해 **AMR 유전자 예측**, **프로모터 검출**, **유전자 및 분류학적 레벨 예측** 등 다양한 생물정보학 문제에 활용되며, **새로운 유전자나 미탐색 종에도 잘 일반화**됩니다. 또한, Scorpio의 임베딩은 **Codon Adaptation Index (CAI)** 같은 생물학적 지표와도 유의미한 상관을 가지며, 단백질 구조나 기능과도 연결되는 정보를 포착할 수 있음을 시각화로 보여줍니다. 실험적으로도 Scorpio는 MMseqs2, Kraken2, DeepMicrobes 등과 비교해 **높은 정확도와 일반화 능력**을 보였습니다.

---


This paper introduces **Scorpio**, a contrastive learning-based framework designed to enhance nucleotide sequence embeddings for genomic analysis. Scorpio leverages **pretrained language models** and **6-mer frequency embeddings** to learn representations that generalize well to **novel DNA sequences and taxa**, overcoming limitations of alignment-based methods. The framework achieves **competitive results** in tasks such as **AMR gene prediction**, **taxonomic and gene classification**, and **promoter detection**. Scorpio embeddings also capture **biological signals**, including correlations with **codon adaptation index (CAI)** and sequence similarity. Empirical evaluations show that Scorpio outperforms existing methods like MMseqs2, Kraken2, and DeepMicrobes in both accuracy and generalizability, highlighting its potential as a versatile tool in metagenomic and genomic research.

---

필요하면 t-SNE 시각화나 성능표 중심으로 조금 더 구체적으로 설명해줄 수도 있어!


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



**1. 백본 모델 및 구조:**

Scorpio는 **대조 학습(contrastive learning)**을 기반으로 설계된 트리플렛 네트워크(triplet network) 구조를 사용해 DNA 서열 임베딩을 학습합니다. 세 가지 주요 인코더 구조를 사용합니다:

- **Scorpio-6Freq**: 6-mer 빈도 기반 벡터 (4096차원)를 입력으로 받아 256차원 임베딩으로 변환  
- **Scorpio-BigEmbed**: BigBird 사전학습 임베딩을 사용하고 모든 층을 고정 (frozen)  
- **Scorpio-BigDynamic**: BigBird 기반으로 마지막 임베딩 층은 학습 가능하게 조정

각 트리플렛 (anchor, positive, negative)은 3개의 동일한 네트워크를 통과하고, 최종 임베딩 벡터의 유클리드 거리를 최소화/최대화하는 방식으로 학습됩니다. 이때 **Margin Ranking Loss**를 사용합니다.

**2. 트레이닝 데이터 구성:**

Scorpio는 NCBI에서 수집한 1929개의 박테리아 및 고세균 유전체 데이터를 사용해 **497개의 보편적이고 보존된 유전자**를 중심으로 학습합니다. 이 유전자들은 대부분 명명된 housekeeping genes이며, 각 유전자는 최소 1,000개의 인스턴스를 포함합니다. 이로써 클래스 간 불균형을 줄이고 **계층적 분류 학습(hierarchical classification)**에 적합한 구조를 갖춥니다.

**3. 계층적 학습:**

Scorpio는 유전자(gene)를 최상위 레벨로, 그 아래로 **phylum, class, order, family, genus** 순으로 계층 구조를 학습합니다. 학습 중 각 트리플렛은 이 계층 중 하나의 레벨에서 anchor–positive–negative 쌍을 선택하여, 다층적인 생물학적 정보를 반영합니다. 이 구조는 단일 모델에서 유전자 및 분류학 정보 모두를 동시에 학습할 수 있도록 합니다.

**4. 임베딩 검색 및 추론:**

Scorpio는 추론 시 Facebook AI의 FAISS 라이브러리를 사용해 학습된 임베딩 벡터의 최근접 이웃을 빠르게 검색하며, **거리 기반 신뢰도 점수(confidence score)**를 계산하여 분류의 신뢰도를 함께 제공합니다.

---

**1. Backbone Model and Architecture:**

Scorpio adopts a **triplet network** architecture with **contrastive learning**, aiming to optimize embeddings for nucleotide sequences. Three types of encoders are used:

- **Scorpio-6Freq**: Uses 6-mer frequency vectors (4096 dimensions), reduced to 256-dimensional embeddings.  
- **Scorpio-BigEmbed**: Utilizes **pre-trained BigBird embeddings** with all layers frozen.  
- **Scorpio-BigDynamic**: Uses BigBird embeddings with the final embedding layer **trainable**, allowing fine-tuning.

Each triplet (anchor, positive, negative) is processed through the same network, and **margin ranking loss** is applied to bring similar sequences closer and push dissimilar ones apart in the embedding space.

**2. Training Data:**

The training dataset was curated using **1929 bacterial and archaeal genomes** from NCBI. A total of **497 well-characterized genes** were selected—each gene having over 1,000 instances. These were mainly housekeeping genes to ensure biological relevance and class balance, making the model suitable for **hierarchical classification**.

**3. Hierarchical Training:**

Scorpio explicitly models **gene-taxonomy hierarchies**. It places **genes at the highest level**, followed by phylum, class, order, family, and genus. Each triplet is sampled with a defined similarity level across this hierarchy, enabling the model to learn both **functional and taxonomic information** concurrently within a single framework.

**4. Embedding Search and Inference:**

For inference, Scorpio employs **FAISS (Facebook AI Similarity Search)** to retrieve nearest neighbors efficiently in the embedding space. It also provides a **confidence score** derived from query distances and neighborhood class probabilities, offering interpretability and robustness for downstream predictions.

---




   
 
<br/>
# Results  




**1. 비교한 경쟁 모델들:**

Scorpio는 다음과 같은 기존 방법들과 비교 평가되었어:

- **MMseqs2**: 빠른 시퀀스 정렬 기반 도구
- **Kraken2**: k-mer 기반 분류 시스템
- **DeepMicrobes**: 딥러닝 기반의 유전자 분류 모델
- **BERTax**: 트랜스포머 기반의 사전학습 분류기
- **BigBird**: 사전학습된 LLM 임베딩 모델

**2. 평가 메트릭:**

- **Accuracy (정확도)**
- **F1-macro score (다중 클래스 간 평균 F1)**
- 각 메트릭은 **taxonomy level (phylum, class, order, family)** 과 **gene level** 에 대해 개별적으로 측정됨

**3. 테스트 데이터 종류 및 실험 설정:**

테스트는 총 네 가지 시나리오에서 진행됨:

- **Memorization Test (Test Set)**: 학습 데이터에 포함된 유전자 및 분류 레벨은 같지만 조합은 다른 시퀀스들
- **Gene-Out Test**: 학습 데이터에 없는 유전자
- **Taxa-Out Test**: 학습 데이터에 없는 계통 분류 (phylum 등)
- **Short Fragment Test**: 400bp 짧은 서열로 구성된 테스트 (시퀀싱 현실 반영)

**4. 주요 결과:**

- **Full-length 시퀀스**에서 Scorpio-BigDynamic은 gene 수준에서 **98.8% 정확도**, phylum 수준에서도 **89.0%**로 매우 높은 성능을 기록.
- **Gene-Out/Taxa-Out 테스트**에서도 기존 방법들보다 **일반화 성능**이 높았고, 특히 BigBird 대비 gene 수준에서는 Scorpio가 더 뛰어났고, taxonomic 레벨에서는 BigBird가 약간 더 나은 경우도 있었음.
- **Short fragments (400bp)**의 경우에도 Scorpio는 gene 수준에서 **98.9%**, phylum 수준에서 **91.0%**로 거의 최고 성능.
- 특히 **Kraken2**와 **MMseqs2**는 새로운 유전자나 계통에 대해 **극단적으로 낮은 정확도**를 보였고, **Scorpio는 이들을 10배~60배 이상 능가**함.

---



**1. Baseline Models Compared:**

Scorpio’s performance was benchmarked against several prominent tools:

- **MMseqs2**: A fast alignment-based sequence search tool  
- **Kraken2**: A popular k-mer based taxonomic classifier  
- **DeepMicrobes**: A supervised deep learning model for gene classification  
- **BERTax**: A transformer-based taxonomy classifier  
- **BigBird**: A pre-trained LLM embedding generator

**2. Evaluation Metrics:**

- **Accuracy (%)**
- **F1-macro Score (%)**
- Metrics were reported across **multiple taxonomic levels** (phylum, class, order, family) and the **gene level**

**3. Test Datasets:**

Scorpio was evaluated on four primary test sets:

- **Memorization Test (Test Set)**: Sequences contain known genes and taxa but in unseen combinations  
- **Gene-Out**: Sequences with novel genes not present in training  
- **Taxa-Out**: Sequences from novel phyla or taxa  
- **Short Fragment Test (400bp)**: Simulating real-world short-read sequencing data

**4. Key Findings:**

- For **full-length sequences**, Scorpio-BigDynamic achieved **98.8% gene-level accuracy** and **89.0% phylum-level accuracy**, outperforming all other models.
- In **generalization tasks** (Gene-Out and Taxa-Out), Scorpio outperformed Kraken2, MMseqs2, and DeepMicrobes significantly, especially at **higher taxonomy levels**.
- On **short fragment sequences**, Scorpio again led with **98.9% accuracy at the gene level** and **91.0% at the phylum level**.
- Traditional tools like **Kraken2** and **MMseqs2** performed poorly on novel genes or taxa, often achieving <5% accuracy, while **Scorpio was 10× to 60× better** in some cases.

---





<br/>
# 예제  



**1. 트레이닝 데이터 예시**

- **출처**: NCBI에서 수집한 1929개의 박테리아 및 고세균 유전체
- **유전자 수**: 명명된 보편적 유전자(gene) 497개만 선택 (예: *gyrA*, *uvrA*, *aspS* 등)
- **총 시퀀스 수**: 약 72,000,000개의 CDS로부터 선별하여 **800,318개 유전자 서열**로 구성
- **전처리 조건**:
  - hypothetical/unknown 유전자 제외
  - 각 유전자는 **1000개 이상의 인스턴스** 포함
  - 각 시퀀스에는 해당 **유전자 이름 + 분류학 정보(phylum~genus)** 라벨이 부여됨

**2. 테스트 데이터 예시**

- **Test Set (memorization)**:
  - 학습 시 사용된 유전자와 분류군을 포함하되, 조합은 처음 보는 시퀀스
  - 예: *gyrA* 유전자가 속한 *E. coli* 대신 *Salmonella* 속의 *gyrA* 시퀀스

- **Gene-Out**:
  - 훈련 데이터에 **존재하지 않는 유전자**
  - 예: 학습에 없는 *kamB* 유전자를 포함하고 있는 새로운 균주

- **Taxa-Out**:
  - 훈련에 쓰이지 않은 **새로운 phylum**에서 유래한 시퀀스
  - 예: 기존에 없는 고세균(Archaea)의 *uvrA* 유전자

- **Short Fragments (400bp)**:
  - 긴 유전자 시퀀스를 랜덤하게 자른 **400bp 조각**
  - ORF가 아닐 수도 있고, in-frame이 아닐 수도 있음
  - 실제 NGS 실험의 short-read와 유사한 설정

**3. 태스크(Task) 예시**

- **분류(Classification)**:
  - 유전자 이름 (gene name)
  - 분류학적 계층: phylum, class, order, family, genus

- **추론(Inference)**:
  - **쿼리 시퀀스**를 임베딩 후, FAISS를 통해 최근접 시퀀스를 찾아 분류
  - 출력: 분류 결과 + confidence score

- **응용 태스크**:
  - **프로모터 검출**: 81bp 길이의 promoter vs. non-promoter 구분
  - **AMR 예측**: MEGARes와 CARD 데이터 기반으로 **항생제 저항성 유전자 분류**



**1. Training Dataset Example**

- **Source**: 1929 bacterial and archaeal genomes from NCBI
- **Gene Selection**: 497 well-known genes (e.g., *gyrA*, *uvrA*, *aspS*)  
- **Sequence Count**: 800,318 coding sequences (CDS), selected from over 72 million total sequences
- **Filtering Criteria**:
  - Removed hypothetical or unnamed genes
  - Each gene included had at least **1000 instances**
  - Labels include **gene name + taxonomy levels (phylum to genus)**

**2. Test Dataset Examples**

- **Test Set (Memorization)**:
  - Contains genes and taxa seen during training, but with unseen combinations  
  - Example: *gyrA* gene from *Salmonella* not seen during training, though *gyrA* was in *E. coli*

- **Gene-Out**:
  - Contains **novel genes** absent from the training set  
  - Example: *kamB* gene not seen during training

- **Taxa-Out**:
  - Contains genes from **unseen phyla**, like certain archaeal sequences  
  - Example: *uvrA* gene from a newly introduced archaeal species

- **Short Fragments (400bp)**:
  - Random 400bp subsequences from full-length gene sequences  
  - Not guaranteed to be in-frame or within an ORF  
  - Mimics real-world NGS short-read data

**3. Task Examples**

- **Classification Tasks**:
  - Predict **gene identity** (e.g., *uvrA*, *gyrB*)  
  - Predict **taxonomy levels**: phylum, class, order, family, genus

- **Inference**:
  - Embed the query DNA sequence
  - Use **FAISS nearest-neighbor search** for label prediction
  - Return predicted labels along with a **confidence score**

- **Application Tasks**:
  - **Promoter detection**: classify 81bp sequences as promoter or non-promoter  
  - **AMR gene classification**: classify resistance genes using MEGARes and CARD databases

---



<br/>  
# 요약   




Scorpio는 6-mer 빈도 및 사전학습된 BigBird 임베딩을 기반으로 triplet network와 대조 학습을 활용하여 유전자 및 계통 정보를 동시에 학습한다. 학습에는 497개의 보편적 유전자가 포함된 80만 개 이상의 시퀀스를 사용하였고, 테스트에서는 새로운 유전자나 분류군에 대한 일반화 능력을 평가하였다. 그 결과 Scorpio는 기존의 Kraken2, MMseqs2, DeepMicrobes보다 전반적으로 높은 정확도와 F1-score를 기록하며, 특히 짧은 시퀀스와 낯선 데이터셋에서도 우수한 성능을 보였다.

---



Scorpio leverages contrastive learning with triplet networks using 6-mer frequencies and pretrained BigBird embeddings to jointly learn gene and taxonomic representations. It was trained on over 800,000 sequences from 497 universal genes and tested on both familiar and novel genes and taxa. Scorpio consistently outperformed baseline models like Kraken2, MMseqs2, and DeepMicrobes, especially showing strong generalization and accuracy even on short and out-of-distribution sequences.

---




<br/>  
# 기타  





####  **Table 1: 주요 실험 성능 테이블**

- Table 1은 총 4개의 하위 섹션으로 구성됨:
  - **(a)** Memorization Test (전체 시퀀스): 학습된 조합과 다른, 그러나 이미 본 유전자/분류군으로 구성된 테스트셋
  - **(b)** Generalization Test (Gene-Out/Taxa-Out): 새로운 유전자 또는 분류군으로부터 생성된 테스트셋
  - **(c)** Memorization on 400bp short reads: 짧은 시퀀스에서의 예측 정확도
  - **(d)** Generalization on short reads: 새로운 유전자 또는 분류군을 짧은 시퀀스로 표현한 실험
- 각 테이블에는 **Accuracy 및 F1-macro**가 각각의 분류 계층(phylum, class, order, family, gene)에 대해 제시되어 있음
- Scorpio-BigDynamic이 전반적으로 gene-level에서 가장 높은 정확도(98% 이상)를 달성함

---

####  **Figure 1: Scorpio 프레임워크 개요**

- (A) Woltka 파이프라인을 통해 유전자-분류군 데이터셋 생성
- (B) 6-mer 또는 BigBird 기반 임베딩 생성 + Triplet 학습 구조
- (C) FAISS 기반 검색 + 계층적 confidence score 계산 및 예측
- 계층적 정보: gene → phylum → class → order → family → genus

---

####  **Figure 2, 3, 5: 시각화 기반 임베딩 해석**

- **Figure 2**: Scorpio-BigEmbed와 BigBird(pretrained)의 t-SNE 시각화 비교
  - Scorpio는 gene이나 phylum 단위로 뚜렷한 클러스터 형성을 보임
- **Figure 3**: *uvrA* 유전자의 시퀀스들을 계층적으로 컬러링하여, gene–phylum–class–...–genus 수준의 분류 정보를 임베딩이 잘 반영하는지 확인
- **Figure 5**: Codon Adaptation Index(CAI)와 Scorpio 임베딩 간의 상관관계 시각화
  - 평균 CAI와 임베딩 간 t-SNE 축 상의 좌표 간 **음의 상관관계(Pearson −0.60)** 존재

---

####  **Figure 6, 7: 응용 테스크 결과**

- **Figure 6**: 프로모터 검출 과제에서 Scorpio가 MCC와 Accuracy에서 기존 방법보다 18% 이상 향상
- **Figure 7**:
  - **(a)** AMR 분류 정확도 비교: Scorpio가 모든 다른 모델들보다 전반적으로 높은 성능
  - **(b)** Cross-attention 분석: KamB와 Kmr 유전자에서 고주의 영역(βN1, βN2 등)을 강조
  - **(c)** 시퀀스 로고와 3D 구조 시각화 → 기능적으로 중요한 영역과 Scorpio 모델이 주목한 영역이 일치함을 보여줌

---

### Supplemental Figures 및 기타

- **Supplementary Figure 8**: DeepMicrobes와 비교하여 Scorpio의 학습 시간이 얼마나 효율적인지 보여줌
- **Supplementary Figure 9–10**:
  - 유사 기능 또는 구조의 유전자들이 임베딩 공간에서 가까운 위치에 있음
  - edit distance와 임베딩 유클리드 거리 간 **R² ≈ 0.41** 수준의 상관관계 존재

---



####  **Table 1: Core Performance Comparison Table**

- Divided into four parts:
  - **(a)** Memorization test on full-length sequences  
  - **(b)** Generalization on unseen genes and taxa (Gene-Out / Taxa-Out)
  - **(c)** Memorization test on 400bp fragments  
  - **(d)** Generalization on short reads from novel classes
- Metrics: **Accuracy and F1-macro**, evaluated for **phylum, class, order, family, gene**
- Scorpio-BigDynamic achieves up to **98.8% gene-level accuracy**, outperforming all baselines

---

####  **Figure 1: Scorpio Framework Overview**

- (A) Dataset creation using Woltka with 497 genes across 1929 genomes  
- (B) Encoder branches using 6-mer or BigBird embeddings + triplet learning  
- (C) Nearest neighbor retrieval with FAISS and hierarchical confidence scoring  
- Prediction operates across multiple taxonomic levels simultaneously

---

####  **Figures 2, 3, 5: Embedding Visualization**

- **Figure 2**: t-SNE plots show that Scorpio creates more **distinct gene and phylum clusters** than the pretrained BigBird
- **Figure 3**: Hierarchical coloring of *uvrA* fragments illustrates Scorpio’s **multi-level taxonomic discrimination**
- **Figure 5**: Significant **negative correlation** (Pearson −0.60) between **Codon Adaptation Index (CAI)** and embedding space location, indicating biologically meaningful structure

---

####  **Figures 6 & 7: Downstream Tasks**

- **Figure 6**: Promoter detection task—Scorpio outperforms previous methods by **more than 18% in accuracy and MCC**
- **Figure 7**:
  - **(a)** Scorpio achieves the **highest AMR gene classification performance**
  - **(b)** Attention maps on KamB and Kmr genes identify functionally relevant regions (e.g., β6/7 linker, mutation sites)
  - **(c)** Structure-aligned heatmaps show that **model-attended regions align with known AMR functional domains**

---

####  **Supplemental Figures**

- **Supp. Fig 8**: Training time comparison shows Scorpio is more efficient than DeepMicrobes
- **Supp. Fig 9–10**:
  - Functionally similar genes cluster in embedding space
  - **R² ≈ 0.41** between edit distance and embedding distance validates biological structure capture

---



<br/>
# refer format:     



@article{refahi2025scorpio,
  title = {Enhancing nucleotide sequence representations in genomic analysis with contrastive optimization},
  author = {Refahi, Mohammadsaleh and Sokhansanj, Bahrad A. and Mell, Joshua C. and Brown, James R. and Yoo, Hyunwoo and Hearne, Gavin and Rosen, Gail L.},
  journal = {Communications Biology},
  volume = {8},
  number = {517},
  year = {2025},
  publisher = {Nature Portfolio},
  doi = {10.1038/s42003-025-07902-6},
  url = {https://doi.org/10.1038/s42003-025-07902-6}
}



Refahi, Mohammadsaleh, Bahrad A. Sokhansanj, Joshua C. Mell, James R. Brown, Hyunwoo Yoo, Gavin Hearne, and Gail L. Rosen. “Enhancing Nucleotide Sequence Representations in Genomic Analysis with Contrastive Optimization.” Communications Biology 8, no. 517 (2025). https://doi.org/10.1038/s42003-025-07902-6.





