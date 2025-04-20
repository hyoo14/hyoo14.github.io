---
layout: post
title:  "[2019]16S rRNA sequence embeddings: Meaningful numeric feature representations of nucleotide sequences that are convenient for downstream analyses"  
date:   2025-04-18 12:16:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


16rRNA 워드투백 제안  
근데 핵심은 기존 OTU 대신에, k-mer 임베딩을 기반으로 한 OTU-유사 구조, 즉 "pseudo-OTU"라는 개념을 만들어 사용  


짧은 요약(Abstract) :    



1. 본 연구는 16S rRNA 서열을 k‑mer 기반 Skip‑Gram word2vec로 임베딩하고, 문장 임베딩 기법을 적용해 서열·샘플 수준의 저차원 벡터를 생성했습니다.  
2. 이렇게 얻은 임베딩은 k‑mer 문맥 정보를 보존하면서도 서열 분류(계통 수준)와 샘플 분류(인체 부위 예측)에서 OTU 풍부도와 동등한 성능을 나타냈습니다.  
3. 서열 임베딩 간 코사인 유사도는 전통적인 염기서열 정렬 (identity)와 유사한 거리를 보여, 계통학적 미세 차이를 효과적으로 반영했습니다.  
4. 클러스터링 실험에서는 종(species) 수준에서 가장 높은 군집 충실도를 보였고, 임베딩으로 생성한 pseudo‑OTU는 차원이 크게 줄어든 상태에서도 분류 성능 손실이 거의 없었습니다.  
5. 임베딩이 비지도 학습으로 훈련되므로 라벨 없는 대규모 데이터도 활용 가능하며, 추후 딥러닝 모델의 입력 특징으로서 확장성이 큽니다. 

---


1. This study embeds 16S rRNA sequences by training Skip‑Gram word2vec on k‑mers and then applies a sentence‑embedding technique to build compact vector representations for individual sequences and whole samples.  
2. The resulting embeddings preserve k‑mer context and achieve sample‑level body‑site classification accuracy comparable to conventional OTU abundance profiles.  
3. Pairwise cosine similarity between sequence embeddings mirrors sequence‑alignment identity scores, capturing fine‑grained taxonomic differences.  
4. Clustering experiments show highest fidelity at the species level, and the authors’ “pseudo‑OTU” counts—derived from sequence embeddings—maintain predictive power while drastically reducing dimensionality.  
5. Because the embeddings are trained unsupervised, unlabeled data can augment downstream supervised learning, making these numeric representations a versatile foundation for microbiome machine‑learning pipelines. 



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




1. **학습 데이터 준비**  
   * **코퍼스 구축:** GreenGenes v13.5에서 길이 전체 16S rRNA 서열 1,262,986개를 가져와 k‑mer(길이 k=4, 6, 8, 10, 12, 15)로 잘라 어휘집을 만들었습니다. 변형 염기가 포함된 k‑mer는 제거했습니다.   
   * **임베딩 학습:** gensim Skip‑Gram word2vec으로 5 epoch 학습하고, 등장 빈도 < 100인 k‑mer는 무시했습니다. 6‑mer와 10‑mer에 대해 차원(64, 128, 256), 음수 샘플 수(10, 20), 다운샘플링 임계값(1e‑4, 1e‑6), 윈도우(20, 50)를 조합해 48가지 모델을 실험했습니다. 8‑, 12‑, 15‑mer는 계산 자원 문제로 최종 모델에서 제외했습니다.   

2. **서열·샘플 임베딩 생성**  
   * **서열 임베딩:** 서열 r의 모든 k‑mer 임베딩 ε_m을 가중 평균해 (가중치 a/(a+f_m), f_m은 해당 k‑mer의 빈도) 벡터 ε_r^(raw)을 만든 뒤, 1차 주성분 투영을 빼서(denoising) 최종 ε_r을 얻었습니다.   
   * **샘플/클러스터/바디사이트 임베딩:** 동일 공식을 모든 서열의 k‑mer에 적용해 하나의 총합 벡터를 만든 뒤 denoising 하였습니다. 서열에 교차하지 않는 k‑mer가 있으면 해당 서열은 임베딩에서 제외됩니다. 

3. **비교 기준(Baseline)**  
   * 전통적 **k‑mer 빈도 테이블**(4‑, 6‑, 8‑mer)과 **QIIME OTU 풍부도**(clr 변환) 및 계통학 수준(속·과·목) 풍부도를 생성해 임베딩과 성능을 비교했습니다.   

4. **Pseudo‑OTU 생성**  
   * American Gut 샘플의 서열 임베딩 약 100만 개를 K‑means(k=1000)로 군집화해 **pseudo‑OTU** 중심을 얻고, 각 서열을 가장 가까운 중심에 할당하여 1000차원 시퀀스 카운트 벡터를 만들었습니다.   

5. **분류 및 평가 절차**  
   * American Gut(분변·피부·구강) 11,341 샘플을 90/10(훈련/테스트)로 분할하고, 다항 lasso 회귀(10‑fold CV로 λ 선택)를 사용해 샘플 임베딩, clr‑OTU, pseudo‑OTU, k‑mer 빈도 등을 입력 특징으로 **바디사이트 예측**을 수행했습니다. 균형 정확도와 F1으로 성능을 비교했습니다.   

6. **임베딩 해석**  
   * 분류 모델의 스파스 회귀계수를 이용해 **노드/서열/k‑mer 활성값**을 계산하고, 예측 결정 과정과 중요 k‑mer의 위치·계통 연관성을 추적했습니다.   

7. **기타 분석**  
   * KEGG 16S 서열(16,699개) 임베딩을 K‑means로 클러스터링하여 종·속 수준 군집 충실도를 평가하고, 클러스터 임베딩과 VSEARCH 합의 서열(con‑sensus) 임베딩의 코사인 유사도를 비교했습니다.   
   * 임베딩 공간의 구조 탐색을 위해 t‑SNE(샘플 임베딩)와 독립 성분 분석(10‑mer 임베딩)을 사용했습니다.   

---



1. **Training corpus**  
   * 1,262,986 full‑length 16S rRNA sequences from GreenGenes v13.5 were tokenised into k‑mers (k = 4, 6, 8, 10, 12, 15); k‑mers containing degenerate bases were discarded.   
   * Skip‑Gram word2vec (gensim) was trained for 5 epochs, ignoring k‑mers occurring < 100 times. For 6‑ and 10‑mers the team varied embedding size (64/128/256), negative samples (10/20), down‑sampling thresholds (1e‑4/1e‑6) and window sizes (20/50), yielding 48 model variants; 8‑/12‑/15‑mer models were dropped due to run‑time limits.   

2. **Sequence and higher‑level embeddings**  
   * **Sequence embedding:** For a sequence r, each k‑mer embedding ε_m was weighted by a/(a + f_m) (f_m = corpus frequency) and averaged, then de‑noised by subtracting the first principal component.   
   * **Sample / cluster / body‑site embedding:** The same weighted average was taken over the union of k‑mers from all sequences in the entity, followed by identical de‑noising.   

3. **Baselines**  
   * Constructed classical k‑mer frequency tables (k = 4, 6, 8) and clr‑transformed QIIME OTU abundances (plus genus/family/order collapsed counts) for head‑to‑head comparisons.   

4. **Pseudo‑OTUs**  
   * One million sequence embeddings from American Gut were clustered with K‑means (k = 1000). Counts of sequences per centroid yielded 1000‑dimensional pseudo‑OTU profiles.   

5. **Classification workflow**  
   * American Gut samples (fecal, skin, tongue) were split 90/10 into train/test (7,526 / 835). A multinomial lasso (10‑fold CV for λ) predicted body site using sample embeddings, clr‑OTUs, principal components, k‑mer frequencies, and pseudo‑OTUs. Balanced accuracy and F1 were reported.   

6. **Interpretability analyses**  
   * Sparse regression weights enabled computation of node‑, sequence‑, and k‑mer‑level activations, revealing which taxa and nucleotide regions drove classification decisions.   

7. **Additional evaluations**  
   * KEGG 16S embeddings were K‑means clustered; species‑level clusters showed highest homogeneity and ARI. Consensus‑sequence embeddings aligned closely with cluster embeddings via cosine similarity.   
   * t‑SNE (sample level) and ICA (10‑mer space) provided low‑dimensional visualisations of embedding structure.   

These steps collectively demonstrate how k‑mer–based word2vec embeddings can generate compact, information‑rich representations of microbiome data that perform competitively with—and often surpass—traditional OTU or raw k‑mer approaches across clustering, visualisation, and supervised prediction tasks.

   
 
<br/>
# Results  



| 입력 특징 | 정확도 (Acc) | 정밀도 (Prec) | 재현율 (Rec) | F1 | 차원 수 | 비고 |
|-----------|-------------|---------------|--------------|----|--------|------|
| OTU 풍부도 (clr) | **0.922** | 0.958 | 0.891 | 0.922 | 21,749 | QIIME 97 % OTU 벡터  |
| 속(genus) 수준 풍부도 | 0.934 | 0.960 | 0.905 | 0.931 | 775 |  |
| 과(family) 수준 풍부도 | 0.938 | 0.963 | 0.916 | 0.939 | 247 | 247 차원에서 OTU보다 우수  |
| 목(order) 수준 풍부도 | 0.916 | 0.899 | 0.881 | 0.890 | 110 | 정보 손실로 성능 하락  |
| k‑mer 빈도 (6‑mer) | 0.979 | 0.942 | 0.969 | 0.953 | 4,096 | 원‑핫 빈도 기반 baseline 최고  |
| 샘플 임베딩 (10‑mer, denoise) | 0.960 | 0.929 | 0.945 | 0.936 | 256 | 차원 99 %↓에도 유지  |
| 샘플 임베딩 (15‑mer, denoise) | 0.968 | 0.937 | 0.957 | 0.946 | 256 | 긴 k‑mer 사용 시 최고 임베딩 성능  |
| **Pseudo‑OTU (임베딩 10‑mer)** | **0.977** | **0.971** | **0.964** | **0.968** | 1,000 | 임베딩+K‑means, 최고 종합 성능  |

* **테스트 데이터**: American Gut 835 샘플(분변 749, 피부 31, 혀 54) — 훈련 90 %(7,526) / 테스트 10 %(835) 분할로 평가했습니다.   
* **주요 관찰점**  
  1. 고차원 OTU(21 k 차원)는 정확도 0.922에 그쳤으나, **256‑차원 임베딩**은 0.96 이상으로 향상.  
  2. **Pseudo‑OTU**는 차원을 1 만 분의 1 수준(1 k)으로 줄이면서도 F1 0.968로 최고 성능을 보였습니다.  
  3. 빈도 기반 6‑mer 테이블도 높았지만(0.979), 메모리 4,096 차원 > Pseudo‑OTU 1,000 차원보다 큼.  
  4. Denoising(주성분 제거)은 특히 긴 k‑mer(≥10)에서 안정적 이득을 주었습니다.  

#### 클러스터링 평가 (KEGG 16 S 서열 14,520개)  
| 세부 수준 | ARI | Homogeneity | Completeness |
|-----------|-----|-------------|--------------|
| 종(Species) | **0.84** | 0.98 | 0.94 |
| 과(Family) | 0.43 | 0.96 | 0.79 |
| 문(Phylum) | ‑0.01 | 0.11 | 0.02 |   

*종 수준에서 Adjusted Rand Index 0.84로 VSEARCH 기반 컨센서스와 높은 일치도를 보였으며, 상위 계통으로 갈수록 정보가 희석되었습니다.*

---

  

**Sample‑level body‑site classification (American Gut test set, n = 835)**  

| Feature set | Acc | Prec | Rec | F1 | Dim | Notes |
|-------------|-----|------|-----|----|-----|-------|
| OTU clr counts | **0.922** | 0.958 | 0.891 | 0.922 | 21 ,749 | QIIME 97 % OTUs  |
| Genus counts | 0.934 | 0.960 | 0.905 | 0.931 | 775 |  |
| Family counts | 0.938 | 0.963 | 0.916 | 0.939 | 247 | Best among taxon‑collapsed counts  |
| Order counts | 0.916 | 0.899 | 0.881 | 0.890 | 110 | Information loss at higher level  |
| 6‑mer frequency | 0.979 | 0.942 | 0.969 | 0.953 | 4 ,096 | Strong classical baseline  |
| Embedding 10‑mer (denoise) | 0.960 | 0.929 | 0.945 | 0.936 | 256 | 99 % dimension reduction vs OTU  |
| Embedding 15‑mer (denoise) | 0.968 | 0.937 | 0.957 | 0.946 | 256 | Best pure‑embedding model  |
| **Pseudo‑OTU (10‑mer)** | **0.977** | **0.971** | **0.964** | **0.968** | 1 ,000 | K‑means clusters of sequence embeddings; top overall  |

*Test configuration:* 90 %/10 % split (7,526 train / 835 test; fecal 749, skin 31, tongue 54) and multinomial lasso with 10‑fold CV for λ selection. 

**Key findings**  
1. Compact 256‑dimensional embeddings outperform 21 k‑dimensional OTU vectors (Acc 0.96 vs 0.922).  
2. Pseudo‑OTUs deliver the best F1 (0.968) with only 1 k features.  
3. Raw 6‑mer tables are accurate but higher‑dimensional (4 k) than embeddings.  
4. Removing the first principal component (denoising) stabilises long‑k‑mer embeddings.

**Clustering of KEGG sequences (k‑means on embeddings)**  

| Taxonomic level | ARI | Homogeneity | Completeness |
|-----------------|-----|-------------|--------------|
| Species | **0.84** | 0.98 | 0.94 |
| Family | 0.43 | 0.96 | 0.79 |
| Phylum | –0.01 | 0.11 | 0.02 | 

Species‑level clusters align closely with VSEARCH consensus sequences, confirming that the embedding space preserves fine‑grained sequence similarity, while higher‑level distinctions attenuate.

Overall, k‑mer–based embeddings and their derived pseudo‑OTU counts provide strong, low‑dimensional alternatives to traditional OTU or raw k‑mer profiles for both supervised classification and unsupervised clustering tasks.
 
   


<br/>
# 예제  




1. **k‑mer 임베딩**  
   상단 표는 30 염기 예시 서열을 6‑mer로 나눈 뒤, 각 k‑mer에 대해 앞 8차원만 표시한 임베딩 벡터입니다. 실제 모델에서는 64–256차원의 실수 벡터가 저장되며, 샘플·서열 임베딩은 이런 k‑mer 벡터를 가중 평균한 뒤 1차 주성분을 제거(denoising)해 만듭니다.  

2. **샘플 임베딩**  
   두번째 표는 256차원 샘플 임베딩 중 처음 12개 차원만 보여 줍니다. 이 벡터 1개가 OTU 테이블 전체(수만 차원)를 대체하며, Lasso 분류기나 딥러닝 입력으로 사용됩니다.  

3. **Pseudo‑OTU 카운트 벡터**  
   마지막 표는 K‑means(1000 군집)로 얻은 pseudo‑OTU 중 앞 10개 카운트 예시입니다. 각 샘플은 1 × 1000 정수 벡터가 되며, F1 0.968로 가장 높은 예측 성능을 보였습니다.  

> 실제 연구에서는 이러한 구조가 자동으로 생성·축적되어 분류·클러스터링·가시화를 수행합니다.

---



1. **k‑mer embeddings**  
   The first table shows a 30‑nt toy read split into 6‑mers, with the first 8 dimensions of each k‑mer embedding. In practice the model stores 64‑ to 256‑dimensional float vectors; sequence or sample embeddings are obtained by a weighted average of these vectors followed by first‑PC removal.  

2. **Sample embedding**  
   The second table displays the first 12 of 256 dimensions for one sample embedding. This single dense vector replaces a 20 k‑feature OTU table and feeds directly into a multinomial lasso or deep‑learning classifier.  

3. **Pseudo‑OTU counts**  
   The last table illustrates the first 10 out of 1,000 pseudo‑OTU counts derived by K‑means clustering of sequence embeddings. A 1 × 1000 integer vector per sample yielded the top F1 = 0.968 in body‑site prediction.  

These compact representations dramatically reduce dimensionality while retaining—often enhancing—predictive power and interpretability for microbiome machine‑learning tasks.



      







<br/>  
# 요약   


1) 본 연구는 GreenGenes 16S 전체서열의 k‑mer를 Skip‑Gram word2vec로 임베딩한 뒤 가중 평균과 1차 주성분 제거로 256‑차원 서열·샘플 벡터를 만들고, 이를 K‑means로 1000개 pseudo‑OTU로 집계하는 파이프라인을 제안했다 .  
2) American Gut 테스트셋에서 pseudo‑OTU가 F1 0.968로 clr‑OTU(0.922)와 6‑mer 빈도(0.953)를 능가했으며, KEGG 서열 클러스터링에서는 종 수준 ARI 0.84를 기록해 미세 계통 차이를 잘 보존했다 .  
3) 30염기 예시 서열의 6‑mer 임베딩, 256‑차원 샘플 벡터 일부, 1000‑차원 pseudo‑OTU 카운트 예시를 통해 임베딩 기반 특성 생성·축약 과정을 직관적으로 보여주었다.  

1) The pipeline trains Skip‑Gram word2vec on GreenGenes k‑mers, produces 256‑dimensional sequence and sample embeddings via weighted averaging plus first‑PC removal, and clusters them into 1,000 pseudo‑OTUs .  
2) On the American Gut test set, pseudo‑OTUs reached an F1 of 0.968, surpassing clr‑OTUs (0.922) and 6‑mer frequency tables (0.953), while KEGG embeddings yielded a species‑level clustering ARI of 0.84 .  
3) A toy demonstration with 6‑mer embeddings from a 30‑nt read, the first 12 dimensions of a 256‑dim sample vector, and the first 10 entries of a 1 k‑component pseudo‑OTU profile illustrates how the compact embedding workflow operates end‑to‑end.




<br/>  
# 기타  





| 구분 | 내용 요약 | 의미 |
|------|-----------|------|
| **Fig. 1, 2, 3** | 10‑mer 임베딩의 t‑SNE 투영(KEGG 전장 서열)에서 문·과·속별 클러스터가 뚜렷하며, 코사인 유사도와 글로벌 정렬 identity가 높은 상관을 보여 임베딩이 계통적 거리를 보존함을 입증  | 임베딩 공간이 FASTA 정렬 없이도 유사 서열을 가깝게 배치 |
| **Fig. 4** | KEGG 클러스터별 컨센서스 서열 임베딩과 전체 서열 클러스터 임베딩의 코사인 행렬—대각선이 가장 진하게 나타나 두 임베딩이 사실상 동일(=컨센서스)함을 시각적 확인  | “서열 평균 임베딩 ≈ 합의 서열” 논리 검증 |
| **Fig. 5, 6** | American Gut k‑mer/서열/샘플 임베딩과 6‑mer 빈도 테이블의 저차원 투영: 바디사이트별(분변·피부·혀) 군집과 주요 문(phyla) 분리가 모두 재현되어 정보 손실 없음  | 임베딩 시각화·빈도표 대비 손실 검증 |
| **Fig. 7** | 잘못 분류된 혀 샘플 1건을 따라가며 읽기(서열) 도입 순서에 따른 누적 노드 활성과 주요 k‑mer/서열의 영향력을 시각화—모델 해석 사례  | 임베딩 기반 모델의 결정 과정 트레이스 |
| **Fig. 8** | Lachnospiraceae 속(Genus)별로 바디사이트 특이적 상위‑활성 k‑mer가 16S 다양 영역 상 위치를 달리함을 히트맵으로 표시  | k‑mer 수준 해석·변이영역 파악 |
| **Fig. 9** | k‑mer·서열·샘플 임베딩 간 코사인 유사도 분포: k가 짧을수록·서열 수가 많을수록 유사도가 치우치며, 주성분 제거로 배경 신호가 줄어드는 효과를 박스플롯으로 제시  | denoising 필요성 정량화 |
| **Table 1** | KEGG 서열 클러스터링 지표(ARI, Homogeneity, Completeness): 종 수준 ARI 0.84로 최고, 상위 계통으로 갈수록 감소  | 임베딩이 미세 분류에 강함 |
| **Table 2** | American Gut 바디사이트 분류 성능(다양한 특징): pseudo‑OTU F1 0.968, 6‑mer 빈도 0.953, OTU 0.922 등  | 특징별 예측력 비교 |
| **S1 Appendix** | 추가 실험(6‑mer 빈도 클러스터 ARI, 파라미터 스윕, Fig B 등)과 전체 파라미터 테이블 제공 | 재현·세부 검증 자료 |

---

 

| Item | Brief description | Take‑home message |
|------|-------------------|-------------------|
| **Fig. 1–3** | t‑SNE projections of 10‑mer embeddings (KEGG): clear phylum/genus clusters; cosine similarity tracks global‑alignment identity, confirming taxonomic fidelity  | Embedding space preserves biological distance without alignments |
| **Fig. 4** | Cosine matrix between cluster embeddings and their VSEARCH consensus‑sequence embeddings: dark diagonal shows they are nearly identical  | Averaging sequence embeddings ≈ consensus sequence |
| **Fig. 5 & 6** | Low‑dim views of k‑mer, sequence & sample embeddings vs 6‑mer frequency table for American Gut: body‑site and phylum separations appear in both  | Embedding does not lose structure vs raw features |
| **Fig. 7** | Trace of node, sequence and k‑mer activations while adding reads to a mis‑classified tongue sample—model interpretability showcase  | Reveals how specific taxa flip decision boundary |
| **Fig. 8** | Heat‑map of top‑activated k‑mers within Lachnospiraceae reads; body‑site‑specific k‑mers map to distinct 16S regions  | Links high‑impact k‑mers to variable regions |
| **Fig. 9** | Distributions of cosine similarities across embedding types/sizes; denoising reduces background similarity, especially with many reads  | Quantifies importance of first‑PC removal |
| **Table 1** | KEGG clustering metrics: species ARI 0.84 > genus 0.47 > phylum –0.01  | Fine‑grained separation strongest |
| **Table 2** | Body‑site classification: pseudo‑OTU F1 0.968, 6‑mer freq 0.953, clr‑OTU 0.922  | Embedding‑derived features outperform baselines |
| **S1 Appendix** | Extra experiments—6‑mer clustering, parameter grids, additional visualisations (Fig B) and implementation details | Provides reproducibility and robustness checks |

These supplementary elements collectively validate that k‑mer embeddings capture biologically meaningful patterns, outperform classical features, and remain interpretable down to individual nucleotide motifs.




<br/>
# refer format:     


@article{Woloszynek2019_16SEmbeddings,
  author    = {Woloszynek, Stephen and Zhao, Zhengqiao and Chen, Jian and Rosen, Gail L.},
  title     = {16S rRNA sequence embeddings: Meaningful numeric feature representations of nucleotide sequences that are convenient for downstream analyses},
  journal   = {PLOS Computational Biology},
  year      = {2019},
  volume    = {15},
  number    = {2},
  pages     = {e1006721},
  doi       = {10.1371/journal.pcbi.1006721},
  url       = {https://doi.org/10.1371/journal.pcbi.1006721},
  month     = feb,
  day       = {26},
  publisher = {Public Library of Science}
}



Woloszynek, Stephen, Zhengqiao Zhao, Jian Chen, and Gail L. Rosen. 2019. “16S rRNA Sequence Embeddings: Meaningful Numeric Feature Representations of Nucleotide Sequences That Are Convenient for Downstream Analyses.” PLOS Computational Biology 15 (2): e1006721. https://doi.org/10.1371/journal.pcbi.1006721. ​  







  