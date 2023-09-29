---
layout: post
title:  "[2021]NOVEL APPROACHES TO MARKER GENE REPRESENTATION LEARNING SUING TRAINED TOKENIZERS AND JOINTLY TRAINED TRANSFORMER MODELS"
date:   2023-09-28 15:40:11 -0400
categories: study
---


{% highlight ruby %}


짧은 요약(Abstract) :    
* 학습된 토크나이저와 학습된 트랜스포머 모델들을 사용하는 marker gene을 표현하는 새로운 접근법(모델)  
** 차세대 DNA시퀀스 기술은 마커 DNA seq 데이터의 활용가능성을 높여주었음  
** LM은 기존의 어려움 극복을 위해 사용됨  
** seq는 유의미한 token으로 나누어지도록 함  
** 이 때 학습된 토크나이저 사용, BPE나 Unigram LM  
*** 기존의 slide-window기반 seq tokeni 대체  
** 이것들이 DNA marker gene에 대한 새로운 접근법(DNA seq임베딩 제안-dense임베딩)  
** 트랜스포머 LM 사용  
** 성능이 이전 압도하거나 비슷(고정길이, 미생물 개수 적을 때)  

{% endhighlight %}  

<br/>

# [Papers with my notes]()  


[~~Lecture link~~]()  

<br/>

# 단어정리  
* marker gene: 특정 세포나 조직, 상태 등을 식별할 수 있는 유전자  
* cardinality: 원소개수, 크기, 고유값  
* intra-study dependencies: 연구 내의 종속성, 의존성  
* sensible tokens: 유의미한 토큰  
* ASV: Amplicon Sequence Variant  
* BLAST: Basic Local Alignment Search Tool  
* CLR: Centered Log-Ratio  
* DADA2: Divisive Amplicon Denoising Algorithm 2  
* KEGG: Kyoto Encyclopedia of Genes and Genomes  
* NCBI: National Center for Biotechnology Information  
* OTU: Operation Taxonomic Unit  
* PCoA: Principal Co-ordinates Analysis - 다차원 스케일링의 일종: 유사성 또는 거리 행렬 기반으로 개체들 간의 관계 시각화 - 낮은 차원으로 축소가 PCA와 유사해보이지만 PCoA는 거리 또는 유사성 행렬에 추원 축소 적용(PCA는 원본데이터에 적용)  
* QIIME2: Quantitative Insights Into Microbial Ecology 2: 미생물 군집 구성과 기능을 분석하기 위한 소프트웨어 패키지, 주로 16s rRNA 유전자 sequencing 같은 고처리량 sequencing 데이터 처리 분석(+시각화)  
* SIF: Smooth Inverse Frequency: 문장 또는 문서의 의미를 벡터 형태로 표현하기 위한 방법  
* SVD: Singular Value Decomposition  
* proxies: 대리지표(생물 다양성의 대리지표)  
* SSU: Small Sub Unit  
* LSU: Large Usb Unit  
* prokarytic: 원핵생물(핵을 갖지 않는 단순구조 세포로된 생물)  
* segmentation: 분할, 의미있게 분리  
* k-mer: 연속 k개로 이루어진 부분 sequence, k=3이라면 3개씩 중첩되게 분할  
* Archaea: 고세균  
* agnostic: 구애받지 않는  
* agnostic text compression: 텍스트 언어에 구애받지 않는 압축  
* kingdom phylum class order family genus species species :  계문강목과속종명


<br/>


# Chapter 1  

## Introduction  

### 1.1 Motivation  
동기  
DNA를 잘 이용해보려는 거겠지  


### 1.2 Marker Genes and Microbial Community Analysis  
미커 유전자(상태 등 알려주는)와 미생물 군집 분석  


#### 1.2.1 Ribosomal RNA Genes  
리보솜 RNA - rRNA 같은 거  
* rRNA 유전자는 리보솜 RNA를 코딩하며 이 rRNA는 셒포 내에서 단백질 합성을 수행하는 리보솜의 주요 구성 요소  
* rRNA 유전자는 미생물 분류 및 동정(identification)에서 중요한 마커임  
* 16rRNA 유전자는 박테리아와 고세균분류의 identification에서 주로 사용됨  


#### 1.2.2 Amplicon Sequence Data  
증폭된 seq data  


#### 1.2.3 Challenges and shortcomings of amplicon sequence data  
증폭된 seq data의 약점  


### 1.3 Machine-learning applications with amplicon sequence data  
ML 접근 -> 증폭된 seq data에 적용  


#### 1.3.1 Considerations in machine learning: Model selection  
모델 선택  


#### 1.3.2 Considerations in machine learning: Feature selection  
피처 선택  



### 1.4 Segmentation of DNA sequences  
DNA seq segmentation 분할로 분리, 세분화  


#### 1.4.1 k-mer based segmentation  
k사이즈의 k-1개씩 중첩되게 뿔어난 분할 


#### 1.4.2 Statistically driven segmentation of DNA sequences  
통계적 seg seq   


### 1.5 Representation learning of text data   
임베딩 for text(1. 단어임베딩, 2. 서브워드 토크나이저, 3.트랜스포머, 4.센탠스트랜스포머 4개 소개)  


##### 1.5.1 Word embedding  
단어 임베딩  


#### 1.5.2 Sub-word tokenization  
서브워드 토크나이저(BPE, Unigram LM같은)  


#### 1.5.3 Transformers  
트랜스포머  



#### 1.5.4 Sentence transformer  
센텐스 트랜스포머  



### 1.6 Representation learning of DNA sequence data  
DNA seq로 임베딩학습(유사도 기반, 근데 여기선 k-mer 임베딩만 소개하네)  


#### 1.6.1 k-mer embedding  
k 크기 단위로 k-1개가 중첩되는 seg로 임베딩 생성  


### 1.7 Contribution  
공헌.. k-mer랑 bpe등 써보고 성능 좋았다..?  


# Chapter 2  
## Statistically Driven Segmentation of DNA Sequences  
통계기반 DNA seq seg(분할)  


### 2.1 Data acquisition and preprocessing  
데이터 획득과 전처리  
BPE와 unigram tokenization algorithm 사용  
SILVA NR-Ref version 138.1 database of SSU rRNA 사용   
taxonomic prediction을 위해서 더 작은 데이터인 KEGG database사용(16s sequences)  
즉, SILVA + KEGG 사용  


### 2.2 Byte-pair encoding  
알고리즘 대략: 타겟사이즈 잡고, 코퍼스와 캐릭터셋 초기에 가짐  
캐릭터 단위로 다 쪼개서 초기 캐릭터셋에 추가  
함께 나오는 pair를 묶어서 다빈출을 캐릭터셋에 추가  
타겟사이즈 달성할 때까지 반복  


### 2.3 Unigram tokenization  
BPE가 sub-optimal이라는 연구에 따라 유니그램 토크아니저도 test  
유니그램 토크나이저 -> data 기반 최적 subword 집합 만드는 것 목표  
1. 초기화 - 문자단위 분해(위와 같네)  
2. 빈도기반 서브워드 후보 생성  
3. 정보 손실이 가장 적은 서브워드 선택(가장 정보 손실이 적은 token 선택)  
4. 타겟 사이즈 될 때까지 반복  


### 2.4 16S Sequence Embedding  
16s sequence: 16s rRNA Bacteria and Archaea의 ribosson의 일부를 나타내는 유전자 seq  (높은 보존성 있음, 오랜시간동안 미생물 분류 및 식별에 사용됨)  
3가지 usage  
* 미생물 식별  
* phylogenetic Analysis 계통 발생학적 분석   
* Microbio community analysis:환경 샘플에서 미생물 군집 구성 파악    


* Smooth Inverse Frequency  
문장이나 문서 표현 생성방법(임베딩)   
개별 단어의 벡터 조합하여 문장의 전체 벡터 표현  
코퍼스와 임베딩 모델 주어짐, 토큰들에 해당되는 임베딩 가져옴, 임베딩 평균내고 다만 빈출토큰은 더 적은 가중치를 줌, 결과 매트릭스 만들고, svd로 첫번째 주성분 얻어서 삭세(공통적으로 많이 나오는 요소이기 때문에 삭제)  


### 2.5 Results  
#### 2.5.1 Properties of encoded sequences  
결과: BPE/unigram이 확실히 sequence 줄여주는 압축효과 있음(k-mer 대비)  

Compression rate of tokens  
Distribution of tokens  
Inverse rank frequencey of tokens  
각 토큰의 순위와 역수 빈도인데.. vocab 크기와 토크나이즈 방식에 따라 빈도 랭킹 분포 달라짐 나타냄  
Length of encoded sequences  
encoded seq의 길이들 다름(BPE, Unigram, K-mer)  


#### 2.5.2 Machine-learning assignment of taxonomy  
머신러닝으로 taxonomy 할당, ML classify for taxonomy  


Experimental Design  
실험디자인 - P, O, G 3개   phylum, order, genus (문,강,속) (계문강목과속종명 중에서)  


Classification results  
RF가 NBC보다 성능 좋았고, bpe/unigram보다 k-mer가 좋았음  
ANOVA-Analysis of Variance = 그룹간변동/그룹내변동  : 세개 이상의 그룹간의 평균차이가 통계적 유의미한지 검증  


Feature Correlations  
피처 상관성 -> pearson 상관관계 봄, 토크나이즈 기법간 차이를 보려고 internal correation 봄  


#### 2.5.3 Sequence embedding assessment  
seq임베딩 평가  


Classification of embedded sequences  
임베딩(SIF와 RoBERTa) 분류->RF  


Clustering of embedded sequences  
클러스터링 k-menas 씀  


평가방법들로는 homogeneity, completeness, ARI, AMI 있는데 이건 잘 와닿지 않아서... 논문 한번 다시 봐야할 듯, 이 평가 척도를 제안한 논문  


### 2.6 Conclusions  
k-mer가 거의 좋음.. 대신 시간은 10배 걸림  



# Chapter 3  
## Bi-Directional Transformers Produce High-Quality Sequence Embeddings with Multiple Use Cases   
MLM이니깐..ㅎ  


### 3.1 Masked Language Model  
MLM 굿  


### 3.1.1 Transformer model architecture  
트랜스포머 굿  


#### 3.1.2 Data set preparation  
데이터 준비  


#### 3.1.3 Training  
학습  


### 3.2 Sequence embedding: Siamese network architecture  
seq임베딩을 siamese network 사용하여 학습  


#### 3.2.1 Sequence transformer architecture  
시퀀스 트랜스포머 구조  


#### 3.2.2 Sequence transformer loss and optimization  
loss와 opti(loss function과 최적화)  



#### 3.2.3 Data set preparation   
SILVA와 KEGG 사용  


#### 3.2.4 Training  
sent transformer로 학습(SRoBERTa)  


### 3.3 Quality of sequence transformer embeddings  
퀄리티..  SIF와 비교  


#### 3.3.1 Spearman correlation of pairwise-alignment scores and cosine similarities  
스피어맨상관계수 - 두 변수 순위 사이의 통계적 종속성 측정, 두 변수 모노토닉할 때 유용하다함  
pairwise-alignment(페이의 정렬)-DNA/RNA/protein 서열 최적 일치 찾는 과정  
상관계수 SRoBERTa가 훨씬 높음  



#### 3.3.2 Clustering of embedded sequences  
임베딩의 클러스터링 KEGG로 평가  


#### 3.3.3 Classification of embedded sequences  
분류  

5-fold 분류에서 성능 차이 크지 않음  
분류에서는 SRoBERTa가 성능 나음  
(SIF 대비)  


#### 3.3.4 Nearest sequence lookup  
근접 seq 탐색, search도 테스트해보나봄  



### 3.4 Classification of host phenotype from amplicon sequence data  
주 표현형 분류  


#### 3.4.1 Data set preparation  
Operational Taxonomic Unit(OTU) : used micro biome research, DNA sequencing based find similarity  



### 3.4.2 Sample-level embedding  
sample microbiome 단위 임베딩  


Amplicon Sequence Varient - 노이즈 줄이고 정밀도 높임   
OTU/ASV frequencies: 특정 OTU/ASV가 샘플내에서 발견되는 횟수  


### 3.4.3 Classification of samples  
샘플 분류  



## 3.5 Conclusion  
SOTA  



# Chapter 4  
## Conclusion  
토크나이징 전략 달리한 점이 좀 크게 보이는 듯  


### 4.1 Implication  
역시 토큰 방법과 압축에 대한 걸 얘기함  
LM 자체가 압축 관련이라고 보는 듯  


### 4.2 Limitations  
사용한 데이터가 제한적... 



### 4.3 Future directions  
DNA model extending  






