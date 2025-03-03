---
layout: post
title:  "[2025]SpotDiff: Spatial Gene Expression Imputation Diffusion with Single-Cell RNA Sequencing Data Integration"  
date:   2025-03-03 16:19:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

ST(Spatial Transcriptomics)이라는 테이블 형태의 수치들로 표현되는 RNA를 디퓨전 스럽게 매꾸는 것?  
이게 스퍼스 테이블이라 결측이 많아서 매꿀 때 디퓨전 스럽게 매꾼다는 것인듯   
impute가 매꾸는거니까 인터폴레이션처럼  


짧은 요약(Abstract) :    




공간 전사체학(Spatial Transcriptomics, ST)은 조직 내 유전자 발현 패턴을 고해상도로 시각화하는 기술로, 조직 구조에 대한 이해를 혁신적으로 변화시켰다. 하지만 ST 데이터는 낮은 캡처율로 인해 희소성이 높아, 생물학적 신호를 복원하기 위한 보완(imputation)이 필요하다.

본 연구에서는 ST 데이터 보완을 위한 Spatial Gene Expression Imputation Diffusion 모델, SpotDiff를 제안한다. SpotDiff는 특정 위치(spot)와 유전자 간의 관계를 학습하는 spot-gene prompt learning 모듈을 포함하며, 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터를 통합하여 각 위치에서의 유전자 발현을 보완한다.

이 방법은 다수의 단일 세포 측정을 결합하여 더 안정적인 유전자 발현 프로파일을 제공하므로, 보완 과정의 불확실성을 줄일 수 있다. 여러 벤치마크 실험 결과, SpotDiff는 기존 방법들보다 더 정확하고 생물학적으로 의미 있는 유전자 발현 프로파일을 생성하는 것으로 나타났으며, 특히 희소성이 높은 환경에서 우수한 성능을 보였다.



The advent of Spatial Transcriptomics (ST) has revolutionized understanding of tissue architecture by creating high-resolution maps of gene expression patterns. However, the low capture rate of ST leads to significant sparsity. The aim of imputation is to recover biological signals by imputing the dropouts in ST data to approximate the true expression values.

In this paper, we introduce a Spatial Gene Expression Imputation Diffusion model, referred to as SpotDiff, to facilitate ST data imputation. Specifically, we incorporate a spot-gene prompt learning module to capture the association between spots and genes. Further, SpotDiff integrates single-cell RNA sequencing data to impute gene expression at each spot.

The proposed approach is able to reduce the uncertainty in the imputation process, since the aggregation of multiple single-cell measurements yields a stable representation of the corresponding spot expression profile. Extensive experiments have been performed to demonstrate that SpotDiff outperforms existing imputation methods across multiple benchmarks, yielding more accurate and biologically relevant gene expression profiles, particularly in highly sparse scenarios.
 

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




본 연구에서는 SpotDiff라는 새로운 다중 모달 조건부 확산 모델을 제안하여 공간 전사체학(Spatial Transcriptomics, ST) 데이터의 보완(imputation) 문제를 해결하고자 한다. SpotDiff는 다음과 같은 주요 요소로 구성된다.

1. Spot-Gene Prompt Learning 모듈

SpotDiff는 특정 위치(spot)와 유전자 간의 관계를 학습하기 위해 spot-gene prompt learning 모듈을 도입하였다. 이 모듈은 각 spot의 유전자 발현 정보를 자연어 형식의 프롬프트로 변환하고, T5 인코더를 활용하여 이를 임베딩 벡터로 변환한다. 이를 통해 유전자 간 연관성을 학습하고, 보완(imputation) 과정에서 보다 정교한 조건을 제공할 수 있다.

		•	Spot의 유전자 발현 정보를 서술하는 템플릿(T)을 구성하여 각 spot의 특징을 학습
	•	T5 인코더를 활용하여 spot 설명과 유전자 토큰을 임베딩 벡터로 변환
	•	Spot과 유전자 간의 상호작용을 모델링하기 위한 Spot-Gene Cross-Attention 적용

2. 단일 세포 RNA 시퀀싱 데이터(scRNA-seq) 통합

기존 ST 데이터는 희소성이 크기 때문에, 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터를 활용하여 보완 성능을 향상시킨다. SpotDiff는 scRNA-seq 데이터를 ST 데이터에 정렬(alignment) 및 통합하여 보다 정확한 유전자 발현 정보를 제공한다.

		•	Integration Network (ω): scRNA-seq 데이터를 ST 데이터와 정렬하는 6층 U-Net 기반 네트워크
	•	Cosine Similarity 기반 최근접 이웃 탐색: ST 데이터와 가장 유사한 scRNA-seq 데이터를 찾아 가중 합산하여 보완 데이터 생성
	•	Loss Functions:
	•	LLOC (Locality Loss): 공간적으로 가까운 spot 간 유사한 유전자 발현을 유지하도록 학습
	•	LMSE (Mean Squared Error Loss): 원본 ST 데이터와 보완된 데이터 간 오차 최소화
	•	LPEA (Pearson Correlation Loss): 유전자 발현 패턴의 상관관계를 유지하도록 학습

3. Diffusion Model 기반 보완(Imputation)

SpotDiff의 핵심은 **확산 모델(Diffusion Model)**을 활용하여 ST 데이터의 결측치를 보완하는 것이다. 확산 모델은 노이즈가 추가된 데이터를 점진적으로 복원하는 방식으로 동작한다.

		•	전방 과정(Forward Process): 가우시안 노이즈를 점진적으로 추가하여 원본 데이터를 변형
	•	역방향 과정(Reverse Process): 학습된 네트워크가 노이즈를 제거하여 원본 ST 데이터에 가까운 보완된 데이터를 생성
	•	Diffusion Model Loss (LDIFF): 원본 ST 데이터와의 차이를 최소화하는 방향으로 최적화

트레이닝 및 벤치마킹

SpotDiff는 osmFISH, FISH, STARmap, MERFISH, 10x BA, 10x HBC 등 총 6개의 공간 전사체학 데이터셋에서 평가되었다. 각 데이터셋에 대해 SpotDiff의 성능을 기존 보완 기법(gimVI, Tangram, GraphST, SpaFormer, stMDCI, stDiff, STEM)과 비교하였으며, Pearson Correlation Coefficient (PCC), Structural Similarity Index (SSIM), Root Mean Square Error (RMSE), Cosine Similarity (COSSIM) 등의 다양한 지표를 활용하여 성능을 검증하였다.

결과적으로 SpotDiff는 기존 방법보다 더 높은 정확도와 생물학적으로 유의미한 결과를 제공하며, 특히 희소성이 높은 데이터에서 뛰어난 성능을 보였다.




In this study, we propose SpotDiff, a novel multi-modal conditional diffusion model to address the imputation problem in spatial transcriptomics (ST) data. The SpotDiff framework consists of the following key components:

1. Spot-Gene Prompt Learning Module

SpotDiff introduces a spot-gene prompt learning module to learn associations between spots and genes. This module transforms each spot’s gene expression information into a natural language prompt, which is then embedded using a T5 encoder. This allows for better understanding of gene-gene relationships and provides refined conditions during the imputation process.

		•	Constructing a template (T) that describes the gene expression profile of each spot
	•	Utilizing a T5 encoder to convert spot descriptions and gene tokens into embedding vectors
	•	Applying a Spot-Gene Cross-Attention mechanism to model the interactions between spot descriptions and individual gene expressions

2. Integration of Single-Cell RNA Sequencing (scRNA-seq) Data

Since ST data is highly sparse, we integrate single-cell RNA sequencing (scRNA-seq) data to improve imputation performance. SpotDiff aligns and integrates scRNA-seq data with ST data to provide more reliable gene expression profiles.

		•	Integration Network (ω): A 6-layer U-Net-based network for aligning scRNA-seq data with ST data
	•	Cosine Similarity Nearest Neighbor Search: Identifying the most similar scRNA-seq data points for each ST spot and computing a weighted sum for imputation
	•	Loss Functions:
	•	LLOC (Locality Loss): Encourages spatial consistency by ensuring that adjacent spots share similar gene expression patterns
	•	LMSE (Mean Squared Error Loss): Minimizes the difference between the original ST data and the imputed data
	•	LPEA (Pearson Correlation Loss): Preserves gene expression correlation across samples

3. Diffusion Model-Based Imputation

At its core, SpotDiff utilizes a diffusion model to impute missing values in ST data. The diffusion model progressively refines noisy data to reconstruct biologically meaningful gene expression profiles.

		•	Forward Process: Gradually adding Gaussian noise to the data
	•	Reverse Process: Removing noise using a trained neural network to recover a refined ST data representation
	•	Diffusion Model Loss (LDIFF): Optimizing the model to minimize the discrepancy between imputed and ground-truth ST data

Training and Benchmarking

SpotDiff is evaluated on six spatial transcriptomics datasets: osmFISH, FISH, STARmap, MERFISH, 10x BA, 10x HBC. The model’s performance is compared against state-of-the-art imputation methods such as gimVI, Tangram, GraphST, SpaFormer, stMDCI, stDiff, and STEM using multiple evaluation metrics: Pearson Correlation Coefficient (PCC), Structural Similarity Index (SSIM), Root Mean Square Error (RMSE), and Cosine Similarity (COSSIM).

Experimental results demonstrate that SpotDiff significantly outperforms existing imputation techniques, particularly in highly sparse scenarios, while producing more accurate and biologically meaningful spatial gene expression profiles.
   




   
 
<br/>
# Results  




본 연구에서는 SpotDiff의 성능을 검증하기 위해 다양한 공간 전사체학(Spatial Transcriptomics, ST) 데이터셋을 활용하고, 기존의 최신 보완 기법과 비교 실험을 수행하였다.

1. 비교 모델 (Competing Models)

SpotDiff는 다음과 같은 최신 ST 데이터 보완 기법과 비교되었다.

모델명	설명
gimVI	Variational Autoencoder 기반의 공간 전사체학 보완 기법
Tangram	단일 세포 RNA-seq 데이터를 ST 데이터와 정렬하여 보완
GraphST	그래프 신경망(GNN)을 활용한 공간 데이터 보완 기법
SpaFormer	트랜스포머 모델을 활용하여 ST 데이터 보완
stMDCI	그래프 신경망과 확산 모델을 결합한 방법
stDiff	단일 세포 RNA-seq 데이터를 통합한 확산 모델
STEM	전이 학습을 적용한 ST 데이터 보완 모델

SpotDiff는 위의 모델들과 비교하여 성능을 분석하였다.

2. 테스트 데이터셋 (Test Datasets)

본 연구에서는 다양한 공간 전사체학(ST) 데이터셋을 사용하여 SpotDiff의 성능을 평가하였다.

데이터셋	조직 유형	ST 데이터 크기 (Spots/Cells)	유전자 수 (Genes)	단일 세포 RNA 데이터 크기
osmFISH	Somatosensory Cortex	3,405	33	30,527
FISH	Embryo	3,039	84	8,924
STARmap	Primary Visual Cortex	1,549	1,020	34,041
MERFISH	Primary Visual Cortex	2,399	268	34,041
10x BA	Brain Anterior	2,695	32,285	22,764
10x HBC	Human Breast Cancer	3,798	3,6601	5,000

SpotDiff는 위의 6개 데이터셋에서 평가되었으며, 단일 세포 RNA-seq 데이터를 활용하여 보완 성능을 측정하였다.

3. 성능 평가 메트릭 (Evaluation Metrics)

SpotDiff의 성능은 다음과 같은 4가지 주요 평가 지표를 활용하여 분석되었다.

메트릭	설명
PCC (Pearson Correlation Coefficient)	원본 ST 데이터와 보완된 데이터 간의 상관관계 측정
SSIM (Structural Similarity Index)	구조적 유사도를 평가하여 ST 데이터 보완의 품질 분석
RMSE (Root Mean Square Error)	원본 데이터와 보완된 데이터 간의 평균 오차 측정
COSSIM (Cosine Similarity)	보완된 데이터가 원본 데이터의 벡터 공간에서 얼마나 유사한지 평가

이러한 지표를 활용하여 SpotDiff의 성능을 기존 모델들과 비교하였다.

4. 성능 비교 결과 (Performance Comparison Results)

(1) 정량적 성능 평가 (Quantitative Evaluation)

SpotDiff는 대부분의 데이터셋에서 **모든 지표(PCC, SSIM, RMSE, COSSIM)**에서 기존 모델보다 우수한 성능을 보였다.

모델	osmFISH (PCC)	FISH (PCC)	STARmap (PCC)	MERFISH (PCC)	10x BA (PCC)	10x HBC (PCC)
gimVI	0.2011	0.4984	0.1854	0.2365	0.1845	0.1837
Tangram	0.2124	0.4622	0.2254	0.3421	0.1847	0.1829
GraphST	0.2354	0.3432	0.1754	0.2157	0.1896	0.1663
SpaFormer	0.3020	0.5112	0.2454	0.3325	0.2401	0.2100
stMDCI	0.2954	0.5876	0.2874	0.3412	0.2311	0.2244
stDiff	0.2775	0.4109	0.1754	0.2451	0.1954	0.2085
STEM	0.2745	0.3874	0.1984	0.1854	0.1896	0.1832
SpotDiff (Ours)	0.3721	0.6215	0.3215	0.3784	0.2559	0.2345

SpotDiff는 모든 데이터셋에서 기존 모델보다 높은 PCC 값을 기록하였으며, 특히 희소성이 높은 ST 데이터에서 우수한 성능을 보였다.

(2) 마커 유전자 복원 성능 (Marker Gene Recovery)

SpotDiff는 특정 마커 유전자 복원 실험에서도 가장 정확한 유전자 발현 패턴을 재현하는 것으로 나타났다. 10x BA 및 10x HBC 데이터셋에서 수행된 실험에서는, SpotDiff가 기존 모델보다 유전자 발현의 공간적 패턴을 더 정확하게 복원하는 것으로 확인되었다.

(3) 클러스터링 분석 (Clustering Analysis)

SpotDiff는 Mop 데이터셋을 이용한 클러스터링 분석에서도 뛰어난 성능을 보였다.

모델	ARI	AMI	NMI
Raw ST Data	0.6404	0.8073	0.8097
gimVI	0.6832	0.8038	0.8074
Tangram	0.6732	0.7936	0.7972
stDiff	0.7254	0.8345	0.8361
stMDCI	0.7022	0.7784	0.7841
SpotDiff (Ours)	0.7336	0.8440	0.8561

SpotDiff는 기존 방법들보다 세포 집단을 더욱 정확하게 복원하여 높은 ARI, AMI, NMI 값을 기록하였다.

결론 (Conclusion)

SpotDiff는 공간 전사체학(ST) 데이터 보완을 위한 새로운 접근법으로, 확산 모델, 단일 세포 RNA-seq 데이터 통합, Spot-Gene 프롬프트 학습을 결합하여 기존 방법보다 우수한 성능을 달성하였다. 실험 결과, SpotDiff는 모든 벤치마크에서 기존 보완 모델을 능가하며, 특히 희소성이 높은 ST 데이터에서 강력한 성능을 발휘하는 것으로 나타났다.




The SpotDiff model was evaluated using six spatial transcriptomics (ST) datasets and compared against several state-of-the-art imputation methods (gimVI, Tangram, GraphST, SpaFormer, stMDCI, stDiff, STEM). SpotDiff outperformed existing models across PCC, SSIM, RMSE, and COSSIM metrics, particularly excelling in highly sparse datasets.

Additionally, SpotDiff demonstrated superior marker gene recovery, clustering accuracy, and overall gene expression profile consistency compared to competing methods. The results suggest that SpotDiff is a robust and biologically meaningful imputation framework for spatial transcriptomics data analysis.



 
<br/>
# 예제  



SpotDiff의 성능을 검증하기 위해 다양한 실험을 수행하였으며, 이를 통해 모델의 입출력 구조, 훈련 및 테스트 데이터 구성, 태스크(task) 정의를 설명한다.

1. 학습 데이터 (Training Data)

SpotDiff는 공간 전사체학(Spatial Transcriptomics, ST) 데이터와 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터를 결합하여 학습되었다. 모든 데이터셋에서 ST 데이터는 보완 대상이며, scRNA-seq 데이터는 보완을 보조하는 역할을 한다.

데이터셋	조직 유형	ST 데이터 크기 (Spots/Cells)	유전자 수 (Genes)	단일 세포 RNA 데이터 크기
osmFISH	Somatosensory Cortex	3,405	33	30,527
FISH	Embryo	3,039	84	8,924
STARmap	Primary Visual Cortex	1,549	1,020	34,041
MERFISH	Primary Visual Cortex	2,399	268	34,041
10x BA	Brain Anterior	2,695	32,285	22,764
10x HBC	Human Breast Cancer	3,798	3,6601	5,000

훈련 과정:
	1.	**ST 데이터의 일부를 무작위로 마스킹(masking)**하여 보완해야 할 목표를 생성
	2.	scRNA-seq 데이터를 활용하여 보완 신호를 제공
	3.	Spot-Gene Prompt Learning 모듈을 통해 spot과 유전자 간 연관성을 학습
	4.	확산 모델(Diffusion Model) 기반으로 노이즈를 제거하며 보완 수행
	5.	생성된 데이터와 원본 데이터 비교 후 오차 최소화

2. 테스트 데이터 및 태스크 인풋/아웃풋 (Test Data and Task Input/Output)

SpotDiff의 성능을 평가하기 위해 ST 데이터에서 무작위로 유전자 발현 값을 제거하고, 이를 모델이 얼마나 정확하게 복원하는지 실험하였다.

(1) 입력 데이터 (Task Input)
	•	Xmst: 마스킹된 공간 전사체학 데이터 (ST 데이터에서 일부 유전자 값이 제거됨)
	•	Xrna: 단일 세포 RNA 시퀀싱 데이터 (scRNA-seq 데이터)
	•	ftext: Spot-Gene Prompt 학습을 통해 생성된 텍스트 임베딩
	•	frna: scRNA-seq 데이터에서 변환된 임베딩 벡터

(2) 출력 데이터 (Task Output)
	•	X̂st: 보완된 ST 데이터 (SpotDiff가 복원한 유전자 발현 값)
	•	성능 평가 지표: 원본 ST 데이터와 비교하여 보완 정확도 평가

예제

입력 데이터 (Masked ST Data)

Spot 1: Gene A - ?, Gene B - 12.5, Gene C - ?, Gene D - 8.3
Spot 2: Gene A - 4.2, Gene B - ?, Gene C - 5.6, Gene D - ?
Spot 3: Gene A - ?, Gene B - 10.1, Gene C - ?, Gene D - 7.9

(여기서 ?는 결측값을 의미)

출력 데이터 (Imputed ST Data by SpotDiff)

Spot 1: Gene A - 6.3, Gene B - 12.5, Gene C - 3.8, Gene D - 8.3
Spot 2: Gene A - 4.2, Gene B - 9.7, Gene C - 5.6, Gene D - 6.2
Spot 3: Gene A - 5.9, Gene B - 10.1, Gene C - 4.4, Gene D - 7.9

SpotDiff는 공간적 정보와 scRNA-seq 데이터를 활용하여 결측값을 보완함.

3. 태스크 성능 평가 (Task Performance Evaluation)

SpotDiff의 성능은 원본 ST 데이터와의 유사성을 기준으로 평가되었다.
PCC (Pearson Correlation Coefficient), SSIM (Structural Similarity Index), RMSE (Root Mean Square Error), COSSIM (Cosine Similarity) 등을 활용하여 기존 모델과 비교하였다.

모델	PCC (상관관계)	SSIM (구조적 유사도)	RMSE (오차)	COSSIM (코사인 유사도)
gimVI	0.2011	0.0876	1.3071	0.3845
Tangram	0.2124	0.1120	1.3158	0.4011
GraphST	0.2354	0.1211	1.2741	0.4658
SpaFormer	0.3020	0.2421	1.1003	0.5721
stMDCI	0.2954	0.2444	1.1021	0.5845
SpotDiff (Ours)	0.3721	0.3021	1.0442	0.6615

SpotDiff는 모든 메트릭에서 기존 모델 대비 향상된 성능을 기록하였다.





To evaluate SpotDiff’s performance, we conducted various experiments, defining input/output structures, training and test data configurations, and task objectives.

1. Training Data

SpotDiff is trained using spatial transcriptomics (ST) data and single-cell RNA sequencing (scRNA-seq) data to enhance gene expression imputation.

Dataset	Tissue Type	ST Data Size (Spots/Cells)	Genes	scRNA-seq Data Size
osmFISH	Somatosensory Cortex	3,405	33	30,527
FISH	Embryo	3,039	84	8,924
STARmap	Primary Visual Cortex	1,549	1,020	34,041
MERFISH	Primary Visual Cortex	2,399	268	34,041
10x BA	Brain Anterior	2,695	32,285	22,764
10x HBC	Human Breast Cancer	3,798	3,6601	5,000

Training Process:
	1.	Masking a portion of ST data to create missing values
	2.	Using scRNA-seq data to assist imputation
	3.	Learning spot-gene associations with Spot-Gene Prompt Learning
	4.	Diffusion model reconstructs the missing data
	5.	Error minimization through loss functions

2. Test Data and Task Input/Output

Task Input:
	•	Xmst: Masked ST Data
	•	Xrna: Single-cell RNA-seq Data
	•	ftext: Spot-Gene Prompt embeddings
	•	frna: Transformed embeddings from scRNA-seq data

Task Output:
	•	X̂st: Imputed ST Data
	•	Performance Metrics: PCC, SSIM, RMSE, COSSIM

Example

Input (Masked ST Data)

Spot 1: Gene A - ?, Gene B - 12.5, Gene C - ?, Gene D - 8.3
Spot 2: Gene A - 4.2, Gene B - ?, Gene C - 5.6, Gene D - ?
Spot 3: Gene A - ?, Gene B - 10.1, Gene C - ?, Gene D - 7.9

Output (Imputed ST Data by SpotDiff)

Spot 1: Gene A - 6.3, Gene B - 12.5, Gene C - 3.8, Gene D - 8.3
Spot 2: Gene A - 4.2, Gene B - 9.7, Gene C - 5.6, Gene D - 6.2
Spot 3: Gene A - 5.9, Gene B - 10.1, Gene C - 4.4, Gene D - 7.9

SpotDiff reconstructs missing values based on spatial and scRNA-seq data.

3. Performance Metrics

SpotDiff outperforms state-of-the-art methods across multiple benchmarks, showing higher PCC, SSIM, lower RMSE, and better COSSIM.





<br/>  
# 요약   






SpotDiff는 공간 전사체학(ST) 데이터 보완을 위해 **확산 모델(Diffusion Model)**과 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터 통합을 활용하는 새로운 기법이다. 실험 결과, SpotDiff는 기존 보완 모델보다 **더 높은 정확도(PCC 0.3721, SSIM 0.3021)와 낮은 오차(RMSE 1.0442)**를 기록하며, 특히 희소성이 높은 데이터에서 우수한 성능을 보였다. 실제 테스트에서 마스킹된 유전자 발현 값을 정확히 복원했으며, 단일 세포 데이터를 활용한 최적화된 보완 과정이 기존 모델 대비 성능 향상을 가져왔다.



SpotDiff is a novel approach that utilizes a diffusion model and integrates single-cell RNA sequencing (scRNA-seq) data for spatial transcriptomics (ST) imputation. Experimental results show that SpotDiff outperforms existing methods with higher accuracy (PCC 0.3721, SSIM 0.3021) and lower error (RMSE 1.0442), particularly in highly sparse datasets. In real-world testing, SpotDiff accurately reconstructed masked gene expression values, demonstrating significant improvements over prior models through optimized integration of single-cell data.



<br/>  
# 기타  



Figure 1: 기존 모델 vs. SpotDiff의 ST 데이터 보완 과정
	•	(a) 기존 ST 데이터 보완 모델들은 단일 소스(ST 데이터만 또는 scRNA-seq 데이터만)를 사용하여 결측치를 보완하는 방식이다.
	•	(b) SpotDiff는 Spot-Gene Prompt Learning과 scRNA-seq 데이터 통합을 활용하여 다중 모달 정보를 결합하여 보완을 수행한다.
	•	SpotDiff는 공간적 연관성과 세포 수준 정보를 결합함으로써 기존 모델 대비 더 높은 정확도와 생물학적 일관성을 유지한다.

Figure 2: SpotDiff 아키텍처 개요
	•	(a) Spot-Gene Prompt Learning 모듈: T5 인코더를 사용하여 spot과 유전자 간 관계를 학습
	•	(b) Integration Network: scRNA-seq 데이터를 ST 데이터에 정렬하여 보완에 활용
	•	(c) Diffusion Model 기반 보완 과정: 결측 유전자 발현 값을 점진적으로 복원하는 구조

Table 1: 성능 비교 결과 (Benchmarking Results)

모델	osmFISH (PCC)	FISH (PCC)	STARmap (PCC)	MERFISH (PCC)	10x BA (PCC)	10x HBC (PCC)
gimVI	0.2011	0.4984	0.1854	0.2365	0.1845	0.1837
Tangram	0.2124	0.4622	0.2254	0.3421	0.1847	0.1829
GraphST	0.2354	0.3432	0.1754	0.2157	0.1896	0.1663
SpaFormer	0.3020	0.5112	0.2454	0.3325	0.2401	0.2100
stMDCI	0.2954	0.5876	0.2874	0.3412	0.2311	0.2244
SpotDiff (Ours)	0.3721	0.6215	0.3215	0.3784	0.2559	0.2345

	•	SpotDiff는 모든 데이터셋에서 **가장 높은 Pearson Correlation Coefficient (PCC)**를 기록하며, 기존 모델 대비 우수한 성능을 보임.
	•	특히 FISH (PCC 0.6215) 및 MERFISH (PCC 0.3784) 데이터셋에서 성능 향상이 두드러짐.

Figure 3: 마커 유전자 복원 결과 (Marker Gene Recovery)
	•	10x BA 및 10x HBC 데이터셋에서 특정 마커 유전자를 보완한 결과를 비교.
	•	SpotDiff는 유전자 발현 패턴과 공간적 정보까지 정밀하게 복원하는 것으로 나타남.
	•	기존 모델들은 마커 유전자 발현량을 과대/과소 평가하는 경향이 있음.

Figure 4: UMAP 기반 데이터 분포 시각화
	•	ST 원본 데이터, 기존 모델, SpotDiff가 생성한 데이터를 시각적으로 비교.
	•	gimVI, Tangram 등 기존 모델들은 ST 데이터와의 분포 차이가 크며, 일부는 원본 데이터와 동떨어진 클러스터를 형성함.
	•	SpotDiff는 원본 ST 데이터의 분포와 가장 유사한 패턴을 보이며, 더 신뢰할 수 있는 보완 결과를 생성함.

Table 3: Ablation Study (구성 요소별 성능 분석)

모델	frna 포함	ftext 포함	osmFISH (PCC)	FISH (PCC)	STARmap (PCC)	MERFISH (PCC)
Baseline	✗	✗	0.2987	0.5124	0.2315	0.3025
+ frna	✓	✗	0.3215	0.5651	0.2655	0.3315
+ ftext	✗	✓	0.3612	0.6055	0.3154	0.3644
SpotDiff (Both)	✓	✓	0.3721	0.6215	0.3215	0.3784

	•	SpotDiff의 두 가지 주요 요소인 frna(scRNA-seq 임베딩)와 ftext(Spot-Gene Prompt Learning)를 모두 활용할 때 가장 높은 성능을 기록.
	•	frna와 ftext를 함께 사용하지 않으면 성능이 감소하는 것을 확인할 수 있음.







Figure 1: Comparison of Traditional Models vs. SpotDiff
	•	(a) Traditional models rely on either ST data alone or scRNA-seq data for imputation.
	•	(b) SpotDiff combines Spot-Gene Prompt Learning with scRNA-seq integration to enhance imputation accuracy.
	•	By incorporating spatial relationships and single-cell measurements, SpotDiff achieves higher biological consistency.

Figure 2: SpotDiff Architecture Overview
	•	(a) Spot-Gene Prompt Learning Module: Uses a T5 encoder to learn relationships between spots and genes.
	•	(b) Integration Network: Aligns and integrates scRNA-seq data with ST data for improved imputation.
	•	(c) Diffusion Model-Based Imputation: Gradually refines missing gene expressions through noise removal.

Table 1: Performance Comparison (Benchmarking Results)

Model	osmFISH (PCC)	FISH (PCC)	STARmap (PCC)	MERFISH (PCC)	10x BA (PCC)	10x HBC (PCC)
gimVI	0.2011	0.4984	0.1854	0.2365	0.1845	0.1837
Tangram	0.2124	0.4622	0.2254	0.3421	0.1847	0.1829
GraphST	0.2354	0.3432	0.1754	0.2157	0.1896	0.1663
SpaFormer	0.3020	0.5112	0.2454	0.3325	0.2401	0.2100
stMDCI	0.2954	0.5876	0.2874	0.3412	0.2311	0.2244
SpotDiff (Ours)	0.3721	0.6215	0.3215	0.3784	0.2559	0.2345

	•	SpotDiff consistently achieves higher PCC values across all datasets, outperforming existing methods.
	•	Significant performance improvement is observed in highly sparse datasets like FISH (PCC 0.6215) and MERFISH (PCC 0.3784).

Figure 3: Marker Gene Recovery
	•	SpotDiff accurately reconstructs marker gene expression patterns in 10x BA and 10x HBC datasets.
	•	Competing models tend to overestimate or underestimate gene expression, while SpotDiff captures biologically relevant details.

Figure 4: UMAP-Based Data Distribution
	•	SpotDiff’s imputed data distribution closely matches the original ST data, unlike gimVI or Tangram, which show deviations.
	•	This demonstrates SpotDiff’s ability to maintain spatial structure and biological consistency.

Table 3: Ablation Study

Model	frna Included	ftext Included	osmFISH (PCC)	FISH (PCC)	STARmap (PCC)	MERFISH (PCC)
Baseline	✗	✗	0.2987	0.5124	0.2315	0.3025
+ frna	✓	✗	0.3215	0.5651	0.2655	0.3315
+ ftext	✗	✓	0.3612	0.6055	0.3154	0.3644
SpotDiff (Both)	✓	✓	0.3721	0.6215	0.3215	0.3784

	•	SpotDiff performs best when both frna (scRNA-seq embeddings) and ftext (Spot-Gene Prompt Learning) are used together.



<br/>
# refer format:     


@article{Chen2025SpotDiff,
  author    = {Tianyi Chen and Yunfei Zhang and Lianxin Xie and Wenjun Shen and Si Wu and Hau-San Wong},
  title     = {SpotDiff: Spatial Gene Expression Imputation Diffusion with Single-Cell RNA Sequencing Data Integration},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  volume    = {39},
  number    = {4},
  pages     = {476–489},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  
}


Tianyi Chen, Yunfei Zhang, Lianxin Xie, Wenjun Shen, Si Wu, and Hau-San Wong. “SpotDiff: Spatial Gene Expression Imputation Diffusion with Single-Cell RNA Sequencing Data Integration.” Proceedings of the AAAI Conference on Artificial Intelligence 39, no. 4 (2025): 476–489. 


