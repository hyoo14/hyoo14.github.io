---
layout: post
title:  "[2019]Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"  
date:   2025-04-22 20:18:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

Set 내부 요소 간 상호작용을 학습 가능한 attention 구조로 정교하게 통합한 아키텍처  
(내부 요소 간의 상호작용을 self-attention과 ISAB로 정교하게 학습하고, attention 기반 pooling(PMA)으로 통합하는, permutation-invariant한 아키텍처)  



짧은 요약(Abstract) :    



  
이 논문은 순서에 무관한(set) 입력 데이터를 처리해야 하는 다양한 머신러닝 과제(예: multiple instance learning, 3D 객체 인식, few-shot 학습 등)를 해결하기 위해 고안된 새로운 딥러닝 구조인 **Set Transformer**를 소개합니다. 기존의 평균/합계 풀링 방식의 한계를 극복하고, **self-attention 메커니즘**을 활용해 입력 집합 내 요소 간의 복잡한 상호작용을 포착합니다. 특히 **Inducing Point** 기반의 attention 구조를 도입하여 계산 복잡도를 줄였으며, 다양한 실험을 통해 기존 방법보다 더 나은 성능을 입증하였습니다. 이 모델은 permutation-invariant 조건을 만족하고, 다양한 크기의 입력 집합을 효과적으로 처리할 수 있습니다.

---


Many machine learning tasks, such as multiple instance learning, 3D shape recognition, and few-shot image classification, are defined on sets of instances. Since solutions to these problems do not depend on the order of elements in the set, models must be permutation invariant. This paper introduces **Set Transformer**, an attention-based neural network module specifically designed to model interactions among elements in a set. It consists of an encoder and a decoder that both rely on attention mechanisms. To reduce computational complexity, the authors propose an attention scheme inspired by inducing point methods from sparse Gaussian process literature, which reduces the time complexity of self-attention from quadratic to linear in the number of set elements. Theoretical appeal and experimental results across multiple tasks show the model outperforms recent methods for set-structured data.





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





**1. 모델 구조 (Structure)**  
Set Transformer는 **attention 기반의 set 처리 모델**로, 입력 집합의 순서에 영향을 받지 않고 작동하도록 설계되었습니다.  
전체 구조는 크게 **Encoder**와 **Decoder**로 구성됩니다.

- **Encoder**:  
  - 주로 **Set Attention Block (SAB)** 또는 **Induced Set Attention Block (ISAB)** 을 사용함  
  - ISAB는 self-attention의 계산량을 줄이기 위해 **inducing point (m개)**를 도입하여 attention 연산을 효율적으로 수행 (O(n²) → O(nm))  
  - 모든 block은 permutation-equivariant 성질을 유지

- **Decoder**:  
  - **Pooling by Multihead Attention (PMA)** 구조를 사용하여 출력 벡터들을 생성  
  - PMA는 k개의 learnable seed vectors를 사용하여 aggregation 수행  
  - 이후 SAB를 추가로 적용해 다수 출력 간 상호작용을 학습

**2. 핵심 구성 요소 (Blocks)**  
- **MAB (Multihead Attention Block)**: Query-Value-Key 구조에 기반한 attention 연산을 수행  
- **SAB (Set Attention Block)**: MAB의 self-attention 버전  
- **ISAB (Induced SAB)**: 계산 효율을 위한 inducing point 도입  
- **PMA (Pooling by Multihead Attention)**: 학습 가능한 seed vector로 feature aggregation

**3. 학습 데이터 (Training Data)**  
다양한 종류의 set-input 문제에서 학습 및 평가를 수행:

- **최댓값 회귀 문제 (Max Regression)**: 임의의 수열에서 최대값을 추정  
- **문자 수 세기 (Omniglot)**: Omniglot 데이터셋에서 서로 다른 문자의 개수 예측  
- **혼합 가우시안 클러스터링 (Meta-Clustering)**: synthetic 2D Gaussian mixture, CIFAR-100 임베딩  
- **이상치 탐지 (Anomaly Detection)**: CelebA 이미지에서 이질적인 이미지 탐지  
- **포인트 클라우드 분류**: ModelNet40 데이터셋의 3D 점 집합 분류

---



**1. Model Structure**  
The Set Transformer is an attention-based architecture designed to handle set-structured data in a permutation-invariant manner. The overall structure consists of two main components:

- **Encoder**:  
  - Composed of **Set Attention Blocks (SAB)** or **Induced Set Attention Blocks (ISAB)**  
  - ISAB introduces a fixed number of **inducing points (m)** to reduce computational complexity from O(n²) to O(nm)  
  - All blocks are permutation-equivariant

- **Decoder**:  
  - Uses **Pooling by Multihead Attention (PMA)** with k learnable seed vectors to aggregate features  
  - Followed by a SAB to model interaction among multiple outputs

**2. Key Components (Blocks)**  
- **MAB (Multihead Attention Block)**: Applies dot-product attention using query, key, and value inputs  
- **SAB (Set Attention Block)**: Self-attention over input set  
- **ISAB (Induced SAB)**: Attention using trainable inducing points for scalability  
- **PMA (Pooling by Multihead Attention)**: Aggregates set features using learnable queries

**3. Training Data**  
The model was evaluated on a variety of set-input tasks:

- **Max Regression Task**: Predicting the maximum value from a set of real numbers  
- **Unique Character Counting**: Predicting the number of unique characters using the Omniglot dataset  
- **Meta-Clustering**: Amortized maximum likelihood estimation using 2D synthetic Gaussians and CIFAR-100 embeddings  
- **Set Anomaly Detection**: Detecting anomalies in image sets from the CelebA dataset  
- **Point Cloud Classification**: Classifying 3D object shapes from ModelNet40 using sets of points

---



   
 
<br/>
# Results  





Set Transformer는 다양한 **set-input task**에서 기존 모델들과 비교하여 **우수한 성능**을 보였습니다. 주요 비교 대상은 다음과 같은 기존 방법들이었습니다:

- **기존 방법들**  
  - `rFF + Pooling`: 평균, 합, 최대 풀링 등 고정 방식의 pooling 사용  
  - `rFF + Dotprod`: attention을 활용한 pooling  
  - `SAB + Pooling`, `ISAB + Pooling`: attention 기반 encoder, 고정형 pooling  
  - `rFF/SAB/ISAB + PMA`: attention 기반 decoder를 추가한 구조 (Set Transformer의 핵심 설계)

**주요 실험 결과**:

1. **최댓값 회귀 (Max Regression)**  
   - 목표: 입력 숫자 집합에서 최대값 예측  
   - MAE (Mean Absolute Error) 기준  
   - Set Transformer (`SAB + PMA`)는 max pooling과 유사하거나 더 낮은 오차를 달성함

2. **문자 수 세기 (Unique Character Counting)**  
   - 데이터셋: Omniglot  
   - 메트릭: 정확도 (Accuracy)  
   - Set Transformer는 `SAB + PMA` 구조로 **60.3%**, 기존 평균 풀링 기반은 약 **43–46%** 수준

3. **혼합 가우시안 클러스터링 (Meta-Clustering)**  
   - 데이터: 2D synthetic mixture, CIFAR-100 임베딩  
   - 메트릭: 로그우도(Log-Likelihood per data), ARI (Adjusted Rand Index)  
   - `ISAB + PMA`가 모든 구조 중 **최고 성능**  
     - 예: CIFAR에서 ARI = **0.9223**, 기존 rFF + pooling은 ARI = **0.5693**

4. **이상치 탐지 (Anomaly Detection)**  
   - 데이터: CelebA 이미지 세트  
   - 메트릭: AUROC, AUPR  
   - Set Transformer (`SAB + PMA`)가 AUROC = **0.5941**, AUPR = **0.4386**으로 가장 뛰어남

5. **3D 포인트 클라우드 분류 (Point Cloud Classification)**  
   - 데이터: ModelNet40  
   - 입력 수: 100, 1000, 5000개 점  
   - 큰 입력에선 `ISAB + Pooling`이 성능이 좋았고, 작은 입력에선 `Set Transformer`가 강점을 보임

---



The Set Transformer outperformed traditional models across various **set-input tasks**, showing clear benefits of attention-based architecture.

- **Baselines Compared**:  
  - `rFF + Pooling`: traditional pooling with sum, mean, or max  
  - `rFF + Dotprod`: attention-based weighted pooling  
  - `SAB/ISAB + Pooling`: attention encoder + simple pooling  
  - `rFF/SAB/ISAB + PMA`: full Set Transformer architectures with attention-based decoder

**Key Findings**:

1. **Max Regression**  
   - Task: Predict max from a set of real numbers  
   - Metric: Mean Absolute Error (MAE)  
   - `SAB + PMA` achieved low MAE comparable to hard max pooling

2. **Unique Character Counting**  
   - Dataset: Omniglot  
   - Metric: Accuracy  
   - `SAB + PMA`: **60.3% accuracy**, outperforming baselines (~43–46%)

3. **Meta-Clustering (Gaussian Mixtures)**  
   - Datasets: Synthetic 2D, CIFAR-100 embeddings  
   - Metrics: Log-likelihood per data, Adjusted Rand Index (ARI)  
   - `ISAB + PMA` was **best-performing**, e.g., CIFAR-100 ARI = **0.9223**, compared to rFF + Pooling (ARI = **0.5693**)

4. **Set Anomaly Detection**  
   - Dataset: CelebA  
   - Metrics: AUROC, AUPR  
   - `SAB + PMA` scored AUROC = **0.5941**, AUPR = **0.4386**, outperforming all other methods

5. **Point Cloud Classification**  
   - Dataset: ModelNet40  
   - Inputs: 100, 1000, 5000 points  
   - For small inputs, Set Transformer performed best; for large inputs, `ISAB + Pooling` yielded better results

---



<br/>
# 예제  



**1. 최댓값 회귀 (Max Regression)**  
- **목표**: 실수 값들로 구성된 집합에서 **최댓값을 예측**  
- **입력 데이터**: `{x₁, x₂, ..., xₙ}` (랜덤한 실수들로 구성된 set)  
- **출력 값**: `max(x₁, ..., xₙ)`  
- **테스트 목적**: attention이 실제로 중요한 요소(max)를 잘 찾아낼 수 있는지를 평가  
- **학습 데이터**: 다양한 크기의 숫자 집합을 생성하여 모델 학습  

**2. 문자 수 세기 (Unique Character Counting)**  
- **목표**: 이미지 set에서 **서로 다른 문자 수 예측**  
- **데이터셋**: [Omniglot](https://github.com/brendenlake/omniglot) — 1623개의 서로 다른 문자를 포함  
- **입력 데이터**: 6~10장의 이미지로 구성된 집합 (동일/다른 문자 섞임)  
- **출력 값**: 서로 다른 문자 종류의 수 (정수)  
- **학습/테스트 분리**: 학습은 훈련 문자에 대해서만 진행, 테스트는 보지 않은 문자에 대해 수행

**3. 가우시안 혼합 클러스터링 (Meta-Clustering with MoG)**  
- **목표**: 입력 점 집합을 기반으로 **가우시안 혼합 모델의 파라미터(클러스터 중심 등) 예측**  
- **Synthetic 2D 데이터**: 2D 평면에 무작위로 생성된 4개의 Gaussian에서 샘플링된 100~500개의 점  
- **CIFAR-100 기반**: VGG로 추출한 512차원 임베딩으로 구성된 세트 (4개의 클래스로부터)  
- **출력 값**: 각 클러스터의 평균, 분산, weight 등 (MoG 파라미터)  
- **테스크 성격**: meta-learning처럼 세트를 보고 분포 파라미터를 한 번에 추정하는 방식

**4. 이상치 탐지 (Set Anomaly Detection)**  
- **목표**: 주어진 이미지 세트 중 **어울리지 않는 한 이미지를 탐지**  
- **데이터셋**: CelebA (202,599장, 40개의 얼굴 속성)  
- **입력 데이터**: 1세트 = 8장 (7장: 공통 속성 2개 포함, 1장: 둘 다 없는 이상치)  
- **출력 값**: 이상치 이미지의 index 또는 score  
- **테스트 목적**: attention 기반 모델이 컨텍스트 내에서 어긋나는 요소를 잘 잡는지 평가

**5. 포인트 클라우드 분류 (Point Cloud Classification)**  
- **목표**: 3D 점군 데이터로 구성된 객체를 **분류(40개 클래스)**  
- **데이터셋**: ModelNet40 (각 객체는 3차원 점 집합으로 표현됨)  
- **입력 크기**: 100, 1000, 5000개의 점을 사용한 세 가지 실험  
- **출력 값**: 클래스 레이블 (예: airplane, chair, table 등)

---



**1. Max Regression Task**  
- **Objective**: Predict the **maximum value** in a set of real numbers  
- **Input**: Sets like `{x₁, x₂, ..., xₙ}` where each `xᵢ` is a real number  
- **Output**: The scalar `max(x₁, ..., xₙ)`  
- **Training**: Randomly generated sets of varying size and values  
- **Purpose**: Test whether the model can attend to the most relevant element

**2. Unique Character Counting**  
- **Objective**: Count how many **unique characters** are in an image set  
- **Dataset**: [Omniglot dataset](https://github.com/brendenlake/omniglot) with 1,623 unique handwritten characters  
- **Input**: Sets of 6–10 images (some repeated, some unique characters)  
- **Output**: Integer representing the number of distinct characters  
- **Training/Test Split**: Only train on a subset of characters; test on unseen characters

**3. Meta-Clustering with Gaussian Mixtures**  
- **Objective**: Estimate the parameters of a **mixture of Gaussians (MoG)** from a given set  
- **Synthetic 2D Data**: 100–500 points sampled from 4 Gaussians on 2D plane  
- **CIFAR-100 Data**: Image embeddings (512D from VGG) sampled from 4 classes  
- **Output**: Cluster centers (means), variances, and mixture weights  
- **Task Type**: Amortized meta-learning—directly mapping a set to MoG parameters

**4. Set Anomaly Detection**  
- **Objective**: Identify the **anomalous image** in a set  
- **Dataset**: CelebA with 202,599 images and 40 face attributes  
- **Input**: 8 images per set (7 share two attributes, 1 lacks both = anomaly)  
- **Output**: Index or score indicating the outlier  
- **Purpose**: Evaluate model’s ability to detect contextual mismatch in set

**5. Point Cloud Classification**  
- **Objective**: Classify 3D objects represented by sets of points  
- **Dataset**: ModelNet40 with 40 object categories  
- **Input Sizes**: 100, 1000, 5000 points per object  
- **Output**: Class label (e.g., airplane, chair, etc.)  
- **Goal**: Evaluate model's ability to reason over large point sets

---




<br/>  
# 요약   





Set Transformer는 self-attention과 inducing point 기반의 ISAB 모듈을 활용하여 permutation-invariant한 set 입력을 효과적으로 처리하는 신경망 구조이다. 다양한 세트 입력 과제들(최댓값 회귀, 문자 수 세기, 클러스터링, 이상치 탐지, 3D 분류)에서 기존의 평균/합 풀링 방식보다 뛰어난 성능을 보였다. 특히 attention 기반 디코더(PMA)를 통한 학습 가능한 풀링이 성능 향상에 결정적 역할을 했다.



Set Transformer is an attention-based neural architecture that processes set-structured inputs using self-attention and the efficient ISAB module with inducing points. It outperforms traditional pooling methods across diverse tasks such as max regression, character counting, clustering, anomaly detection, and 3D point cloud classification. Notably, the learnable pooling via PMA plays a critical role in boosting performance.




<br/>  
# 기타  




####  주요 테이블 요약

**Table 1: Max Regression 실험 (MAE 측정)**  
- 목표는 숫자 집합에서 최대값 예측  
- 단순 평균(mean)과 합(sum) pooling 모델은 MAE가 크고,  
- max pooling은 MAE = **0.1355**로 가장 낮았음  
- Set Transformer (`SAB + PMA`)도 **0.2085**로 매우 근접, attention이 최대값 탐색 기능을 학습했음을 입증

---

**Table 2: Omniglot 기반 Unique Character Counting (Accuracy)**  
- rFF+Pooling: **~44%** 수준  
- `SAB + PMA`: **60.3%** 정확도로 가장 뛰어남  
- attention 구조가 set 내 객체 간 관계를 학습해 더 정확한 판단 가능

---

**Table 3: Meta-Clustering (GMM 파라미터 추정)**  
- Synthetic 2D와 CIFAR-100에서 실험  
- 메트릭: Log-likelihood per data (LL), Adjusted Rand Index (ARI)  
- `ISAB + PMA` 구조가 두 실험 모두에서 **최고 성능**  
  - CIFAR ARI = **0.9223**, 기존 평균 풀링보다 크게 앞섬  
  - EM 1회 후에도 oracle보다 높은 성능 보이는 경우 존재

---

**Table 4: Point Cloud Classification**  
- 데이터셋: ModelNet40, 입력 수: 100/1000/5000 포인트  
- 작은 입력(100개)에서는 Set Transformer 계열이 강했지만,  
- 큰 입력(5000개)에서는 `ISAB + Pooling`이 연산 효율과 정확도 면에서 우위

---

**Table 5: Set Anomaly Detection (CelebA)**  
- 메트릭: AUROC / AUPR  
- `SAB + PMA`: AUROC = **0.5941**, AUPR = **0.4386**  
- 단순 pooling 구조보다 월등히 높은 정확도

---

####  주요 피규어 설명

**Figure 1: Set Transformer 구조 및 구성 블록**  
- (a) 전체 구조: Encoder(SAB/ISAB) + Decoder(PMA + SAB)  
- (b) MAB: 기본적인 attention block  
- (c) SAB: 입력 set 간의 self-attention  
- (d) ISAB: inducing point를 활용한 효율적 attention 구조

---

**Figure 3: Inducing Point 수 변화에 따른 정확도 (Omniglot Task)**  
- inducing point 수를 늘릴수록 성능이 향상됨  
- ISAB의 효율성과 성능 균형을 보여주는 그래프

---

**Figure 4: 클러스터링 시각화 결과 (2D Synthetic Dataset)**  
- 다양한 모델별 클러스터링 결과 비교 (rFF + Pooling, SAB + Pooling, rFF + PMA, Set Transformer)  
- Set Transformer가 가장 자연스럽고 정확하게 클러스터 중심을 추정함을 시각적으로 표현

---

**Figure 5: CelebA 이상치 탐지 샘플**  
- 각 행: 하나의 이미지 세트 (7개 정상, 1개 이상치)  
- 이상치는 빨간 박스로 표시됨  
- attention이 문맥 기반 이상 탐지에 효과적임을 시각적으로 보여줌

---


####  Key Tables

**Table 1: Max Regression (MAE)**  
- Task: Predict the max of a set of numbers  
- Mean/sum pooling had large errors, max pooling achieved MAE = **0.1355**  
- Set Transformer (`SAB + PMA`) achieved **0.2085**, showing it learned to identify key elements

---

**Table 2: Unique Character Counting (Omniglot)**  
- Baseline (rFF + pooling): ~**44%**  
- `SAB + PMA`: achieved **60.3%**, highest performance  
- Attention enabled modeling inter-instance relationships effectively

---

**Table 3: Meta-Clustering with Gaussian Mixtures**  
- Datasets: synthetic 2D and CIFAR-100 embeddings  
- Metrics: Log-likelihood per data (LL), Adjusted Rand Index (ARI)  
- `ISAB + PMA` outperformed all, reaching **ARI = 0.9223** on CIFAR  
- Sometimes even surpassed the EM oracle after just one step

---

**Table 4: Point Cloud Classification**  
- Dataset: ModelNet40 with 100/1000/5000 points  
- For small inputs, Set Transformer excelled  
- For large inputs, `ISAB + Pooling` performed better due to reduced complexity

---

**Table 5: Set Anomaly Detection (CelebA)**  
- Metrics: AUROC / AUPR  
- `SAB + PMA`: AUROC = **0.5941**, AUPR = **0.4386**, best among all models

---

#### Key Figures

**Figure 1: Architecture Blocks**  
- (a) Full pipeline: Encoder (SAB/ISAB) → PMA → Decoder  
- (b) MAB: Multihead Attention Block  
- (c) SAB: Self-attention among set elements  
- (d) ISAB: Efficient self-attention using inducing points

---

**Figure 3: Accuracy vs. Number of Inducing Points**  
- Shows that more inducing points lead to better accuracy  
- Highlights the balance between scalability and performance in ISAB

---

**Figure 4: Clustering Visualization (Synthetic 2D)**  
- Comparison across models: rFF + Pooling, SAB + Pooling, rFF + PMA, Set Transformer  
- Set Transformer’s clustering is the most coherent and aligned with true clusters

---

**Figure 5: Anomaly Detection Samples (CelebA)**  
- Each row shows a set of 8 images, one of which is an anomaly (red box)  
- Demonstrates the model's contextual anomaly detection ability




<br/>
# refer format:     


@inproceedings{lee2019set,
  title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
  author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam R. and Choi, Seungjin and Teh, Yee Whye},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  pages={3744--3753},
  year={2019},
  organization={PMLR}
}   





Lee, Juho, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. “Set Transformer: A Framework for Attention-Based Permutation-Invariant Neural Networks.” In Proceedings of the 36th International Conference on Machine Learning, 3744–3753. PMLR, 2019.  






