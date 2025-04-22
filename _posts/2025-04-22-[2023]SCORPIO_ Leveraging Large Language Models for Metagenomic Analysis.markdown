---
layout: post
title:  "[2020]SCORPIO: Leveraging Large Language Models for Metagenomic Analysis"  
date:   2025-04-22 11:46:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


RoBERTa 를 이용한 DNA LM 제안   


짧은 요약(Abstract) :    


다음은 논문의 **초록(abstract)**에 대한 한글 설명과 영문 원문입니다.

---



이 논문에서는 마이크로바이옴 시퀀싱 데이터의 고차원성과 복잡성을 해결하기 위해 대형 언어 모델 **RoBERTa**를 도입합니다. 먼저 RoBERTa를 유전체 시퀀스 데이터에 맞게 사전 학습시켜 다양한 메타지놈 데이터셋(16S, 28S rRNA, ITS)에 적용 가능한 임베딩을 생성합니다. 이 임베딩은 특히 **분류(classification)** 작업에서 기존 방법보다 높은 성능을 보였으며, 사전 학습된 모델이 다양한 생물종 데이터에 일반화 가능하다는 것을 보여주었습니다. k-mer 기반 전처리 및 하이퍼파라미터 튜닝을 통해 모델 성능을 최적화하였으며, 실험 결과 RoBERTa 임베딩이 기존 k-mer 또는 DNABERT 방식보다 더 정교한 군집 형성과 높은 분류 정확도를 달성함을 보였습니다.

---



> Analyzing sequencing data from microbiome experiments is challenging, since samples can contain tens of thousands of unique taxa (and their genes) and populations of millions of cells. Reducing the dimensionality of metagenomic data is a crucial step in improving the interpretability of complex genetic information, as metagenomic datasets typically encompass a wide range of genetic diversity and variations.

> In this study, we implement RoBERTa, a state-of-the-art large language model, and pre-train it on relatively large genomic datasets to obtain a model that can be used to generate embeddings that can help simplify complex metagenomic data sets. The pre-training process enables RoBERTa to capture the inherent characteristics and patterns present in the genomic sequences. We then evaluate the effectiveness of embeddings generated using the pre-trained RoBERTa model in downstream tasks, with a particular focus on taxonomic classification. To assess whether our method can be generalizable, we conduct extensive downstream analysis on three distinct datasets: 16s rRNA, 28s rRNA, and ITS. By utilizing datasets containing 16S rRNA exclusive to bacteria and eukaryotic mitochondria, as well as datasets containing 28S rRNA and ITS specific to eukaryotes (such as fungi), we were able to assess the performance of RoBERTa embeddings across diverse genomic regions. We tune the RoBERTa model through hyperparameter optimization on each dataset. Our results demonstrate that RoBERTa embeddings exhibit promising results in taxonomic classification compared to conventional methods.

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




1. **백본 모델: RoBERTa 기반**
   - 이 연구는 **RoBERTa**를 백본 모델로 사용하였고, DNA 서열 분석에 맞게 커스터마이징하였다.
   - 원래 RoBERTa는 BERT의 변형으로, 더 큰 배치사이즈, 더 긴 트레이닝 시간, 더 많은 데이터로 학습되며, 마스킹된 토큰 예측을 중심으로 학습한다.

2. **구조 및 트레이닝 설정**
   - DNA 시퀀스는 단어가 없기 때문에, 이를 **k-mer (3~6-mer)** 단위로 나누어 토크나이징함.
   - MLM (Masked Language Modeling) 방식으로 학습함: 시퀀스 일부를 [MASK]로 대체하고 해당 토큰을 예측하게 함.
   - **임베딩 크기**: 512, 768, 1024, 2048 실험 → 최종적으로 2048이 가장 성능 좋았음.
   - **포지셔널 임베딩**: **상대적(relative)** 포지셔널 임베딩이 절대적(absolute) 방식보다 우수.
   - **최대 시퀀스 길이**: 200bp로 설정 (Illumina의 짧은 리드 길이에 맞춤).
   - **모델 파라미터 수**: 약 103M과 209M 모델 비교 → 반드시 큰 모델이 더 나은 성능을 내는 것은 아님.

3. **트레이닝 데이터**
   - **사전 학습(pretraining)**: NCBI에서 수집한 **33,902개 박테리아 유전체** (평균 3.4 Mb).
     - 여기서 7M / 51M 개의 200bp 프래그먼트 생성하여 두 가지 버전의 사전학습 데이터로 사용.
   - **다운스트림 데이터셋**:
     - **DairyDB (16S rRNA)**: 유제품에서 추출된 박테리아 데이터.
     - **Fungi ITS/28S rRNA**: 곰팡이 진핵생물의 ITS 및 28S 리전.

---



1. **Backbone Model: RoBERTa**
   - The study uses **RoBERTa**, a robustly optimized variant of BERT, as the base model. RoBERTa is well-suited for sequence modeling due to its enhanced training setup, including dynamic masking and longer training durations.

2. **Architecture and Training Procedure**
   - DNA sequences were tokenized into **k-mers** ranging from 3 to 6-mers, with 6-mer yielding the best performance.
   - The model was trained using the **Masked Language Modeling (MLM)** objective, where random tokens are masked and predicted.
   - **Embedding sizes** of 512, 768, 1024, and 2048 were tested, with **2048** performing best.
   - **Relative positional embeddings** outperformed absolute embeddings in downstream tasks.
   - The **maximum sequence length** was set to **200 bp**, considering Illumina sequencing constraints.
   - Two model sizes were evaluated (~103M and ~209M parameters), but larger size did not guarantee better performance.

3. **Training Data**
   - **Pretraining Dataset**: Comprised of 33,902 **complete prokaryotic genomes** from NCBI, yielding **7 million and 51 million** 200bp fragments for model training.
   - **Downstream Datasets**:
     - **DairyDB (16S rRNA)**: Bacterial sequences from dairy-associated microbes.
     - **Fungal ITS and 28S datasets**: Eukaryotic sequences from fungal genomes covering diverse taxonomic levels.

---



   
 
<br/>
# Results  




1. **비교 대상 모델**
   - 제안된 **RoBERTa 기반 임베딩 모델**은 다음과 비교됨:
     - **k-mer 빈도 기반 모델 (3-mer, 6-mer)**  
     - **DNABERT 임베딩 기반 모델**
   - 이 모델들은 **KNN** 또는 **Random Forest (RF)** 분류기와 함께 사용되어 비교되었음.

2. **사용된 메트릭**
   - **정확도 (Accuracy)**를 기준으로 세 개의 다운스트림 데이터셋에 대해 **10-fold 교차검증** 수행.
   - 비교는 다양한 분류 수준: **Phylum, Order, Family, Genus** 수준으로 구분하여 수행됨.

3. **테스트 데이터셋**
   - **DairyDB (16S)**: 박테리아 중심.
   - **Fungi ITS** / **Fungi 28S**: 곰팡이류 유전체 리전.
   - 총 3개 데이터셋 × 3개 분류 수준 × 다양한 모델 조합이 비교됨.

4. **결과 요약**
   - 전반적으로 **6-mer 빈도 기반 + RF** 조합이 가장 뛰어난 정확도를 기록.
     - 특히 **Genus 수준 분류 정확도**에서 0.95~0.99에 달하는 고성능을 보임.
   - **RoBERTa 임베딩 모델**은 DNABERT보다 높은 정확도를 보였으며, **t-SNE 시각화**에서 클래스 간 클러스터링이 더 뚜렷하게 구분됨.
   - 특히 16S 데이터셋에서는 **포지션 정보를 임베딩에서 반영**한 결과, 단순 k-mer로는 포착되지 않는 정보까지 잘 표현되었음.

---



1. **Competitor Models**
   - The proposed **RoBERTa-based embedding model** was compared against:
     - **k-mer frequency-based models** (3-mer and 6-mer)
     - **DNABERT embeddings**
   - These representations were evaluated using **K-Nearest Neighbors (KNN)** and **Random Forest (RF)** classifiers.

2. **Evaluation Metrics**
   - **Accuracy** was used as the main metric.
   - Evaluations were performed using **10-fold cross-validation** across three taxonomic levels: **Phylum**, **Order**, **Genus**.

3. **Test Datasets**
   - **DairyDB (16S rRNA)**: Bacterial sequences from dairy products.
   - **Fungi ITS** and **Fungi 28S rRNA**: Eukaryotic (fungal) sequences from two distinct genomic regions.
   - All models were tested on the same datasets across multiple taxonomic resolutions.

4. **Key Findings**
   - The combination of **6-mer frequency + RF** achieved the best accuracy overall, with **Genus-level accuracy reaching up to 0.99**.
   - The **RoBERTa-based embeddings** outperformed DNABERT in classification and exhibited **better cluster separability in t-SNE visualizations**.
   - For the **16S dataset**, RoBERTa embeddings also captured **positional information** that was not evident in traditional k-mer representations.

---





<br/>
# 예제  




####  1. 트레이닝 데이터 (Pretraining)

- **출처**: NCBI에서 수집한 **33,902개의 박테리아(prokaryotic) 전체 유전체**
- **프래그먼트 방식**: 각 유전체에서 **200bp 단위**로 잘라 프래그먼트 생성
  - **7M** (700만 개) 버전과 **51M** (5100만 개) 버전 모두 실험에 사용됨
-  사용 목적: RoBERTa 모델의 **사전학습 (Masked Language Modeling)**

---

####  2. 테스트 및 다운스트림 태스크용 데이터

- **DairyDB (16S rRNA)**  
  - 총 **10,612개 전체 시퀀스**  
  - 이로부터 **200bp 단위로 80,227개 프래그먼트** 생성  
  - 분류 수준: **42개 문(phylum), 197개 목(order), 1069개 속(genus)**

- **Fungi ITS 데이터셋**  
  - **15,551개 ITS 시퀀스**  
  - 200bp 프래그먼트 **50,068개 생성**  
  - 분류 수준: **6개 문, 235개 목, 516개 과(family)**

- **Fungi 28S rRNA 데이터셋**  
  - **8,506개 시퀀스** → 200bp 프래그먼트 **42,766개 생성**  
  - 분류 수준: **8개 문, 105개 목, 293개 과**

---

####  3. 태스크 (Downstream Tasks)

- **목적**: 각각의 200bp DNA 프래그먼트를 기반으로 해당 서열의 **계통학적 분류(Taxonomic Classification)** 수행
- **수행 방식**:
  - 입력: DNA 프래그먼트
  - 출력: 분류 수준(문, 목, 과, 속)에 해당하는 레이블
- **모델 비교**: RoBERTa 임베딩 vs k-mer 빈도 vs DNABERT 임베딩
- **분류기**: KNN, RF 사용

---


####  1. Pretraining Data

- **Source**: 33,902 complete **prokaryotic genomes** from NCBI
- **Fragmentation**: Each genome was split into **200bp fragments**
  - Two sets were prepared: **7 million** and **51 million** fragments
-  Purpose: **Pretrain the RoBERTa model** using **Masked Language Modeling (MLM)**

---

####  2. Downstream/Test Datasets

- **DairyDB (16S rRNA)**  
  - **10,612 full-length sequences**  
  - **80,227 fragments** of 200bp were extracted  
  - Taxonomic coverage: **42 phyla, 197 orders, 1069 genera**

- **Fungi ITS Dataset**  
  - **15,551 ITS sequences**  
  - **50,068 200bp fragments**  
  - Taxonomic coverage: **6 phyla, 235 orders, 516 families**

- **Fungi 28S rRNA Dataset**  
  - **8,506 sequences**, yielding **42,766 fragments**  
  - Taxonomic coverage: **8 phyla, 105 orders, 293 families**

---

#### 3. Tasks (Downstream Tasks)

- **Objective**: Classify each 200bp DNA fragment into its correct **taxonomic level** (Phylum, Order, Family, Genus)
- **Input**: DNA sequence fragment (200bp)
- **Output**: Taxonomic label at a given level
- **Baselines Compared**: RoBERTa embeddings, k-mer frequency representation, DNABERT embeddings
- **Classifiers Used**: K-Nearest Neighbors (KNN), Random Forest (RF)

---




<br/>  
# 요약   




이 연구는 RoBERTa 언어 모델을 DNA k-mer 시퀀스에 맞게 사전학습하고, 이를 통해 생성한 임베딩으로 계통학적 분류 작업을 수행하였다. 16S, ITS, 28S 데이터셋에서 실험한 결과, 제안된 RoBERTa 임베딩은 기존 k-mer 빈도 기반 및 DNABERT보다 뛰어난 분류 성능과 표현력을 보였다. 특히 200bp 프래그먼트 기반 테스트에서, 상대 포지셔널 임베딩과 6-mer 구성의 조합이 가장 높은 정확도를 달성하였다.

---


This study pretrains a RoBERTa language model on DNA k-mer sequences and applies the resulting embeddings for taxonomic classification tasks. Experiments on 16S, ITS, and 28S datasets show that the proposed RoBERTa embeddings outperform traditional k-mer frequency representations and DNABERT in accuracy and representation quality. In particular, 200bp fragment-based tests demonstrated that the combination of 6-mer tokenization and relative positional embedding achieved the highest classification performance.

---



<br/>  
# 기타  





####  **Figure 1**: K-mer 크기별 분류 정확도
- 3-mer부터 6-mer까지 실험했으며, **6-mer일 때 가장 높은 정확도**를 기록함.
- 각 k-mer에 따라 분류 정확도와 loss가 어떻게 변화하는지 시각화.
- 6-mer는 아미노산 단위(2 codons)와 유사하여 표현력이 높다는 점이 성능 향상의 원인으로 분석됨.

####  **Figure 2**: K-mer 길이에 따른 Zipf’s Law 분포
- k-mer 길이가 늘어날수록 희귀 단어가 많아지고, 전체 분포가 점점 치우치는 현상이 시각화됨.
- 6-mer는 자주 등장하는 k-mer와 희귀 k-mer 사이의 균형이 잘 잡혀 있음.

####  **Figure 3(a~e)**: 다양한 하이퍼파라미터가 성능에 미치는 영향
- 각각의 서브피규어는 다음을 보여줌:
  - (a) 임베딩 크기 (256~2048) → 2048에서 가장 좋음
  - (b) 포지셔널 임베딩 타입 → 상대 positional이 절대 positional보다 우수
  - (c) 최대 시퀀스 길이 → 200에서 가장 좋음
  - (d) 프리트레인 데이터 양 → 7M 데이터셋이 51M보다 더 일반화 잘됨
  - (e) 모델 파라미터 수 → 큰 모델이 항상 좋은 성능을 보이지 않음

####  **Figure 4(a~d)**: t-SNE 시각화
- 각 데이터셋(16S, ITS, 28S)에서 임베딩된 벡터들을 시각화함.
- RoBERTa 임베딩은 k-mer 빈도 기반보다 훨씬 **더 분명한 군집(cluster)**을 형성.
- 16S에서는 위치 기반 클러스터가 형성됨 → positional 정보가 잘 반영됨.

####  **Figure 5–7**: 분류 정확도 비교
- **16S, ITS, 28S** 각각에 대해 다양한 조합(KNN + DNABERT, RF + 6mer 등) 비교.
- **RF + 6-mer** 조합이 모든 레벨(Phylum, Order, Genus)에서 가장 우수.
- 제안된 RoBERTa 모델도 Genus 수준에서 높은 정확도를 보임 (예: 0.95 이상).

####  **Appendix / 기타**
- Appendix는 따로 존재하지 않지만, **모든 소스코드와 데이터**는 GitHub에 공개됨:  
  https://github.com/EESI/MetaBERTa

---



####  **Figure 1**: Classification Accuracy by K-mer Size
- Compared 3-mer to 6-mer inputs; **6-mers achieved the highest accuracy**.
- Shows that longer k-mers capture more sequence context, with 6-mers aligning with codon structure (two codons = one amino acid).

####  **Figure 2**: Zipf’s Law of K-mer Frequencies
- As k-mer length increases, the vocabulary becomes more skewed.
- 6-mers demonstrate a good balance between common and rare tokens, aiding model learning.

####  **Figure 3 (a–e)**: Impact of Model Parameters on Performance
- (a) **Embedding size**: 2048 yielded the best results.
- (b) **Positional encoding**: **Relative** embeddings outperformed absolute ones.
- (c) **Max sequence length**: 200 bp was optimal for the datasets.
- (d) **Dataset size**: Surprisingly, **7M fragment set outperformed 51M**, likely due to overfitting in larger data.
- (e) **Model size**: Larger models didn’t always correlate with better performance.

####  **Figure 4 (a–d)**: t-SNE Visualizations of Embeddings
- Visualized embeddings from 16S, ITS, and 28S datasets.
- RoBERTa embeddings show **clearer clustering than k-mer frequency-based vectors**, especially in the ITS and 28S datasets.
- For 16S, position-based clustering appears, suggesting that RoBERTa encodes positional information effectively.

####  **Figures 5–7**: Classification Accuracy Comparisons
- Compared KNN and RF classifiers using DNABERT, 3-mer, 6-mer, and RoBERTa embeddings.
- **RF + 6-mer consistently achieved the best accuracy** across Phylum, Order, and Genus levels.
- RoBERTa also showed competitive performance, particularly at the Genus level (up to 0.95).

####  **Appendix / Others**
- While no formal appendix is provided, **all code and data are available** at:  
  https://github.com/EESI/MetaBERTa

---



<br/>
# refer format:     


@inproceedings{refahi2023scorpio,
  author    = {M. S. Refahi and B. A. Sokhansanj and G. L. Rosen},
  title     = {SCORPIO: Leveraging Large Language Models for Metagenomic Analysis},
  booktitle = {2023 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  year      = {2023},
  pages     = {1--6},
  doi       = {10.1109/SPMB59478.2023.10372773},
  publisher = {IEEE},
  address   = {Philadelphia, PA, USA}
}


Refahi, M. S., B. A. Sokhansanj, and G. L. Rosen. “SCORPIO: Leveraging Large Language Models for Metagenomic Analysis.” In 2023 IEEE Signal Processing in Medicine and Biology Symposium (SPMB), 1–6. Philadelphia, PA, USA: IEEE, 2023. https://doi.org/10.1109/SPMB59478.2023.10372773.   


