---
layout: post
title:  "[2025]When Hypergraph Meets Heterophily: New Benchmark Datasets and Baseline"  
date:   2025-03-04 09:28:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

완전 잘 모르던 분야...   
하이퍼그래프는 그래프의 확장으로 하나의 edge가 두개 이상을 연결.. 일종의 3d?   


Heterophily는.. 일종의 같은 클러스터상에 있지만 다른 클래스들을 가지는..?   


Homophily(동질성): 같은 취향, 같은 학력, 같은 국적의 사람들이 서로 친구가 되는 경향    
Heterophily(이질성): 서로 다른 배경을 가진 사람들이 연결되는 경우 (예: 교수와 학생, 의사와 환자)   


그래서 이를 위한 데이터셋(영화나, 쇼핑 도메인 등)을 제공하는 논문임   
그리고 이 데이터 바탕으로 새 구조도 제안...  (동형/이형 따로 학습을 하는? 방식인듯)   

짧은 요약(Abstract) :    





이 논문은 **이형성 하이퍼그래프 학습 (Heterophilic Hypergraph Learning, HHL)** 분야의 연구 부족을 해결하고자 한다. 기존 **하이퍼그래프 신경망 (Hypergraph Neural Networks, HNNs)** 은 높은 차원의 상관관계를 처리하는 데 성공적이었지만, **이형성 (Heterophily)** 환경에서의 연구는 상대적으로 부족했다. 논문에서는 이를 보완하기 위해 다음 세 가지 측면에서 기여한다:

1. **이형성을 측정하는 새로운 메트릭 도입**  
   - 하이퍼그래프의 **동질성/이형성 비율**을 정량적으로 평가할 수 있는 지표를 개발하여, 기존 연구의 한계를 극복하고 보다 체계적인 분석을 가능하게 한다.

2. **이형성을 반영한 새로운 벤치마크 데이터셋 구축**  
   - 실제 응용 사례를 반영한 다양한 데이터셋을 개발하여, 기존 HNN 모델들의 성능을 평가하고 향후 연구를 촉진할 수 있도록 한다.

3. **새로운 기초 모델 ‘HyperUFG’ 제안**  
   - 프레임렛 (Framelet) 기반의 HNN을 도입하여 저주파 (Low-pass) 및 고주파 (High-pass) 필터를 통합적으로 활용하는 방식을 제시한다.  
   - 이를 통해 기존 모델들이 **이형성 환경에서 성능이 낮은 문제**를 해결하고, 다양한 데이터셋에서 우수한 성능을 보임을 실험적으로 입증하였다.

이 연구는 **이형성 하이퍼그래프 학습 (HHL)** 의 중요성을 강조하며, 해당 분야의 미래 연구를 촉진할 수 있는 기반을 마련하는 것을 목표로 한다.

---



Hypergraph neural networks (HNNs) have shown promise in handling tasks characterized by high-order correlations, achieving notable success across various applications. However, there has been limited focus on **heterophilic hypergraph learning (HHL)**, in contrast to the increasing attention given to **heterophilic graph neural networks (GNNs)**. 

This paper addresses key gaps in HHL research from multiple perspectives:

1. **New Metrics for Measuring Heterophily in Hypergraphs**  
   - Introduces novel metrics to quantify the homophily/heterophily ratio in hypergraphs, providing a numerical basis for systematic evaluation.

2. **Development of Diverse Benchmark Datasets**  
   - Creates new benchmark datasets covering real-world heterogeneous scenarios to facilitate comprehensive evaluations of existing HNNs and further advance HHL research.

3. **Proposing HyperUFG: A Novel Framelet-Based HNN**  
   - Introduces **HyperUFG**, an HNN that integrates both **low-pass and high-pass filters** using a framelet-based approach, addressing the performance limitations of existing models on heterophilic hypergraphs.  
   - Extensive experiments on synthetic and benchmark datasets demonstrate that HyperUFG performs competitively and often outperforms existing models in such scenarios.

Overall, this study underscores the urgent need for further exploration and development in **heterophilic hypergraph learning (HHL)** and provides a strong foundation for future research in this emerging field.



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






#### **1. 모델: HyperUFG (Hypergraph Unified Framelet-based Graph Network)**
HyperUFG는 **프레임렛 기반의 하이퍼그래프 신경망 (Hypergraph Neural Network, HNN)** 으로, 기존 HNN들이 이형성 하이퍼그래프에서 성능이 저하되는 문제를 해결하기 위해 설계되었다.  
이 모델은 **저주파 (Low-pass) 및 고주파 (High-pass) 필터를 결합한 프레임렛 변환 (Framelet Transform)** 을 활용하여, 하이퍼그래프 내 다양한 유형의 관계를 효과적으로 학습할 수 있도록 한다.

**핵심 개념:**
- **저주파 필터 (Low-pass Filter):** 동질적인 (homophilic) 관계를 학습하는 데 사용됨.
- **고주파 필터 (High-pass Filter):** 이형성 (heterophilic) 환경에서 중요한 차별적 정보 (distinctive information) 를 학습하는 데 사용됨.
- **프레임렛 변환 (Framelet Transform):** 하이퍼그래프의 구조적 정보를 다중 해상도에서 캡처할 수 있도록 하며, 다양한 계층적 특징을 학습 가능하게 함.

---

#### **2. 아키텍처**
HyperUFG는 **스펙트럴 기반 (Spectral-based)** 아키텍처로, 하이퍼그래프 라플라시안 (Hypergraph Laplacian) 을 사용하여 프레임렛 변환을 수행한다.  
모델의 구조는 다음과 같다:

1. **입력 (Input):**  
   - 하이퍼그래프 **\( G = (V, E) \)** (노드 집합 **\( V \)**, 하이퍼엣지 집합 **\( E \)**)  
   - 노드 특성 행렬 **\( X \in \mathbb{R}^{N \times m} \)** (N: 노드 개수, m: 특성 차원)

2. **하이퍼그래프 라플라시안 계산 (Hypergraph Laplacian Computation)**  
   - 라플라시안 고유벡터 **\( U \)** 및 고유값 **\( \Lambda \)** 계산.

3. **프레임렛 기반 필터링 (Framelet-based Filtering)**  
   - 저주파 및 고주파 필터링을 수행하여 정보 추출:
     - 저주파 필터: 정보 통합 및 일반적 패턴 학습.
     - 고주파 필터: 중요한 국소적 변화를 감지하고 이형성 데이터 학습.

4. **프레임렛 변환 계수 계산 (Framelet Coefficients Computation)**  
   - 저주파 및 고주파 변환 계수 **\( V_0 \)**, **\( W_j^r \)** 계산하여 특징 맵 생성.

5. **하이퍼그래프 컨볼루션 레이어 (Hypergraph Convolution Layers)**  
   - 다중 계층으로 구성되며, 각 계층에서 프레임렛 필터를 학습하여 특징을 추출함.

6. **출력 (Output):**  
   - 노드 분류 (Node Classification) 또는 기타 하이퍼그래프 학습 태스크 수행.

---

#### **3. 트레인 데이터 (Training Data)**
HyperUFG는 **다양한 이형성 하이퍼그래프 데이터셋** 으로 학습되었다.  
주요 벤치마크 데이터셋은 다음과 같다:

1. **Actor (공연자 네트워크)**  
   - 영화 속 배우들의 공동 출연 관계를 기반으로 한 하이퍼그래프.  
   - 노드: 배우, 감독, 작가  
   - 하이퍼엣지: 영화에서 함께 작업한 팀  
   - 태스크: 직업군 (배우, 감독 등) 분류

2. **Amazon-ratings (상품 공동 구매 네트워크)**  
   - 아마존에서 동일한 사용자가 함께 구매한 상품들로 구성된 하이퍼그래프.  
   - 노드: 상품 (책, DVD 등)  
   - 하이퍼엣지: 공동 구매된 제품 그룹  
   - 태스크: 상품의 평균 리뷰 평점 예측

3. **Twitch-gamers (스트리밍 사용자 네트워크)**  
   - 트위치 스트리머들의 공동 생성 관계를 나타낸 하이퍼그래프.  
   - 노드: 트위치 계정  
   - 하이퍼엣지: 같은 시간대에 활동한 계정들  
   - 태스크: 계정이 성인 콘텐츠를 포함하는지 여부 예측 (이진 분류)

4. **Pokec (소셜 네트워크 데이터)**  
   - 슬로바키아의 대표적인 SNS인 Pokec의 친구 관계를 기반으로 한 하이퍼그래프.  
   - 노드: 사용자  
   - 하이퍼엣지: 특정 사용자의 모든 친구 관계  
   - 태스크: 사용자 성별 분류

이러한 데이터셋을 사용하여 HyperUFG는 **이형성 하이퍼그래프 환경에서도 높은 성능을 보이는 신경망 모델** 로 검증되었다.

---



### **1. Model: HyperUFG (Hypergraph Unified Framelet-based Graph Network)**
HyperUFG is a **framelet-based hypergraph neural network (HNN)** designed to address the challenges posed by **heterophilic hypergraph learning (HHL)**. Unlike existing HNNs, which struggle with heterophilic settings, HyperUFG incorporates **low-pass and high-pass filters** using **framelet transforms** to effectively capture diverse structural relationships in hypergraphs.

**Key Concepts:**
- **Low-pass filter:** Captures homophilic relationships.
- **High-pass filter:** Extracts distinctive information in heterophilic settings.
- **Framelet transform:** Enables multi-scale feature extraction to model hypergraph structures effectively.

---

### **2. Architecture**
HyperUFG adopts a **spectral-based** architecture leveraging the **hypergraph Laplacian** to perform framelet transformations.  
The architecture consists of:

1. **Input:**  
   - A hypergraph **\( G = (V, E) \)** with node set **\( V \)** and hyperedge set **\( E \)**.  
   - Node feature matrix **\( X \in \mathbb{R}^{N \times m} \)** (N: number of nodes, m: feature dimension).

2. **Hypergraph Laplacian Computation:**  
   - Eigenvectors **\( U \)** and eigenvalues **\( \Lambda \)** of the Laplacian are computed.

3. **Framelet-Based Filtering:**  
   - **Low-pass filtering:** Aggregates information from homophilic relationships.  
   - **High-pass filtering:** Extracts crucial distinguishing features in heterophilic structures.

4. **Framelet Coefficients Computation:**  
   - Computes low-pass and high-pass transform coefficients **\( V_0 \)**, **\( W_j^r \)** to generate feature maps.

5. **Hypergraph Convolution Layers:**  
   - Multi-layer architecture that applies framelet filters to extract hierarchical features.

6. **Output:**  
   - Performs **node classification** or other hypergraph learning tasks.

---

### **3. Training Data**
HyperUFG is trained on **diverse heterophilic hypergraph datasets** to validate its effectiveness.  
The benchmark datasets include:

1. **Actor (Co-occurrence Network)**  
   - Hypergraph based on actors co-starring in films.  
   - Nodes: Actors, directors, and writers.  
   - Hyperedges: Groups working on the same film.  
   - Task: Job classification (e.g., actor, director).

2. **Amazon-ratings (Co-purchase Network)**  
   - Hypergraph constructed from Amazon product co-purchases.  
   - Nodes: Products (books, DVDs, etc.).  
   - Hyperedges: Groups of products frequently bought together.  
   - Task: Predicting average review ratings.

3. **Twitch-gamers (Co-creation Network)**  
   - Hypergraph modeling relationships among Twitch streamers.  
   - Nodes: Twitch accounts.  
   - Hyperedges: Accounts that streamed together.  
   - Task: Binary classification (explicit content or not).

4. **Pokec (Social Network Data)**  
   - Hypergraph based on the Pokec social networking platform in Slovakia.  
   - Nodes: Users.  
   - Hyperedges: User's full friend list.  
   - Task: Predicting user gender.

These datasets demonstrate HyperUFG’s capability to **excel in heterophilic hypergraph learning scenarios**.

   
 
<br/>
# Results  





#### **1. 경쟁 모델 (Baseline Models)**
HyperUFG의 성능을 평가하기 위해 기존 하이퍼그래프 신경망(HNN) 및 그래프 신경망(GNN) 모델들과 비교하였다. 실험에서 사용된 주요 경쟁 모델은 다음과 같다.

1. **MLP (Multi-Layer Perceptron)**  
   - 그래프 구조를 고려하지 않는 **그래프 비의존적 (graph-agnostic) 모델**  
   - 하이퍼그래프 구조를 활용하지 않고 노드의 특징만을 학습  

2. **HGNN (Hypergraph Neural Network)**  
   - 하이퍼그래프 기반의 고전적인 모델로, 메시지 패싱(Message Passing) 방식 적용  

3. **HyperGCN**  
   - 하이퍼그래프의 구조를 그래프 기반의 컨볼루션 방식으로 변환하여 학습  

4. **UniGCNII (Unified Graph Convolutional Network II)**  
   - 그래프 및 하이퍼그래프에서 공통적으로 사용할 수 있도록 확장된 GCN 모델  

5. **HyperND (Nonlinear Feature Diffusion on Hypergraphs)**  
   - 하이퍼그래프에서 비선형 특징 확산(Nonlinear Feature Diffusion) 기법을 적용  

6. **AllDeepSets / AllSetTransformer**  
   - 하이퍼그래프를 다중집합(Multiset) 기반으로 표현하여 학습하는 최신 모델  

7. **ED-HNN (Equivariant Hypergraph Diffusion Neural Network)**  
   - 하이퍼그래프 내에서 동형성과 이형성을 모두 학습할 수 있도록 설계된 모델  

8. **SheafHyperGNN**  
   - 세이프 이론(Sheaf Theory)을 적용하여 하이퍼그래프의 복잡한 관계를 반영  

이 모델들과 비교하여 HyperUFG는 **이형성(Heterophily)이 높은 환경에서도 더 나은 성능을 보여주는지 검증**하였다.

---

#### **2. 테스트 데이터 (Test Datasets)**
실험은 **동질성(Homophily)과 이형성(Heterophily)을 갖춘 다양한 하이퍼그래프 데이터셋** 에서 수행되었다.

1. **Homophilic Datasets (동질성 데이터셋)**  
   - **Cora, Citeseer, Pubmed**: 논문 인용 네트워크  
   - **Cora-CA, DBLP-CA**: 인용 및 학술 데이터  

2. **Heterophilic Datasets (이형성 데이터셋)**  
   - **Actor**: 영화 공동 출연 네트워크  
   - **Amazon-ratings**: 상품 공동 구매 네트워크  
   - **Twitch-gamers**: 트위치 사용자 네트워크  
   - **Pokec**: 소셜 네트워크  

**이형성 데이터셋에서의 성능이 특히 중요**하며, 기존 HNN 모델들은 대부분 이형성 환경에서 성능 저하를 보이는 경향이 있다.

---

#### **3. 평가 메트릭 (Evaluation Metrics)**
모델의 성능을 평가하기 위해 다음과 같은 주요 **정확도 기반 지표** 가 사용되었다.

1. **Accuracy (정확도)**  
   - 전체 예측값 중 정답을 맞춘 비율  
   - 데이터셋 내 다양한 이형성 환경에서 모델의 기본 성능 평가  

2. **Top-K Accuracy (Top-K 정확도)**  
   - 모델이 예측한 상위 K개의 라벨 중 정답이 포함되는 비율  
   - 다중 클래스 분류에서 중요한 지표  

3. **Homophily Ratio (동질성 비율, \(H_{\text{edge}}\) 및 \(H_{\text{node}}\))**  
   - 하이퍼그래프 내에서 연결된 노드들이 같은 클래스일 확률  
   - 모델이 동질성과 이형성 환경을 어떻게 학습하는지 측정  

---

#### **4. 실험 결과 요약**
- **동질성 데이터셋 (Homophilic Datasets):**  
  - 기존 HNN 모델들이 좋은 성능을 보였으며, HyperUFG도 높은 성능 유지.  
- **이형성 데이터셋 (Heterophilic Datasets):**  
  - 기존 HNN 모델들은 MLP보다 성능이 낮은 경우가 많았음.  
  - HyperUFG는 **이형성 환경에서도 높은 성능을 유지**하며, 기존 모델들을 능가함.  
  - 특히 **Amazon-ratings 및 Pokec 데이터셋에서 큰 성능 향상**을 보임.  

결과적으로 HyperUFG는 **이형성 환경에서도 우수한 성능을 보이며, 새로운 하이퍼그래프 학습 기준을 제시**하였다.

---



### **1. Baseline Models**
To evaluate HyperUFG's performance, it was compared against existing **hypergraph neural networks (HNNs) and graph neural networks (GNNs)**. The key baseline models used in the experiments are:

1. **MLP (Multi-Layer Perceptron)**  
   - A **graph-agnostic model** that does not consider graph structures.  
   - Learns node features without utilizing hypergraph connectivity.  

2. **HGNN (Hypergraph Neural Network)**  
   - A classic HNN model that applies **message passing** on hypergraphs.  

3. **HyperGCN**  
   - Uses graph convolution to transform hypergraph structures into standard GCNs.  

4. **UniGCNII (Unified Graph Convolutional Network II)**  
   - A generalizable GCN model that can be applied to both graphs and hypergraphs.  

5. **HyperND (Nonlinear Feature Diffusion on Hypergraphs)**  
   - Utilizes **nonlinear feature diffusion** for hypergraph learning.  

6. **AllDeepSets / AllSetTransformer**  
   - Models hypergraphs as **multi-set learning problems**.  

7. **ED-HNN (Equivariant Hypergraph Diffusion Neural Network)**  
   - Handles both homophilic and heterophilic relationships within hypergraphs.  

8. **SheafHyperGNN**  
   - Incorporates **sheaf theory** to model complex hypergraph interactions.  

HyperUFG was tested to determine whether it could **outperform existing models in heterophilic settings**.

---

### **2. Test Datasets**
Experiments were conducted on **a diverse set of hypergraph datasets** that include both **homophilic and heterophilic** structures.

1. **Homophilic Datasets**  
   - **Cora, Citeseer, Pubmed**: Citation networks.  
   - **Cora-CA, DBLP-CA**: Academic citation and collaboration networks.  

2. **Heterophilic Datasets**  
   - **Actor**: Movie co-occurrence network.  
   - **Amazon-ratings**: Product co-purchase network.  
   - **Twitch-gamers**: Twitch user co-creation network.  
   - **Pokec**: Social network dataset from Slovakia.  

**Performance on heterophilic datasets is particularly crucial**, as traditional HNN models often struggle in such environments.

---

### **3. Evaluation Metrics**
The following accuracy-based metrics were used to evaluate the models:

1. **Accuracy**  
   - The proportion of correctly classified samples.  
   - A key metric for assessing general model performance.  

2. **Top-K Accuracy**  
   - The proportion of instances where the correct label appears in the model’s top-K predictions.  

3. **Homophily Ratio (\(H_{\text{edge}}\) and \(H_{\text{node}}\))**  
   - The probability that nodes within the same hyperedge belong to the same class.  
   - Measures how well a model adapts to homophilic vs. heterophilic settings.  

---

### **4. Summary of Results**
- **Homophilic Datasets:**  
  - Traditional HNN models performed well, and HyperUFG maintained high performance.  
- **Heterophilic Datasets:**  
  - Many existing HNNs performed worse than MLP.  
  - **HyperUFG consistently outperformed other models**, maintaining strong accuracy.  
  - **Significant performance improvement was observed on Amazon-ratings and Pokec datasets.**  

Overall, HyperUFG **sets a new benchmark in heterophilic hypergraph learning**, demonstrating its robustness across different hypergraph structures.


<br/>
# 예제  




#### **1. 트레인 데이터 (Training Data)**
HyperUFG 모델은 다양한 **이형성(Heterophilic) 및 동질성(Homophilic) 하이퍼그래프 데이터셋** 에 대해 학습되었다. 주요 트레인 데이터셋은 다음과 같다.

1. **Actor (영화 공동 출연 네트워크)**  
   - **노드 (Nodes):** 배우, 감독, 작가  
   - **하이퍼엣지 (Hyperedges):** 하나의 영화에 함께 참여한 사람들의 집합  
   - **노드 특성 (Node Features):** 위키피디아에서 추출한 키워드 기반 특성  
   - **레이블 (Labels):** 직업군 (배우, 감독, 작가)

2. **Amazon-ratings (상품 공동 구매 네트워크)**  
   - **노드:** 다양한 상품 (책, 음악 CD, DVD 등)  
   - **하이퍼엣지:** 동일한 사용자가 함께 구매한 제품 그룹  
   - **노드 특성:** 제품 설명에서 추출된 Bag of Words(BOW) 표현  
   - **레이블:** 상품의 평균 리뷰 평점 (10개 클래스)

3. **Twitch-gamers (트위치 사용자 네트워크)**  
   - **노드:** 트위치 사용자 계정  
   - **하이퍼엣지:** 동일한 시간대에 스트리밍한 계정들  
   - **노드 특성:** 조회수, 계정 생성/업데이트 시간, 사용 언어, 방송 시간 등  
   - **레이블:** 성인 콘텐츠 여부 (이진 분류)

4. **Pokec (소셜 네트워크 데이터)**  
   - **노드:** 소셜 네트워크 사용자  
   - **하이퍼엣지:** 특정 사용자의 친구 목록  
   - **노드 특성:** 연령, 관심사, 학력, 지역, 가입 시기 등  
   - **레이블:** 사용자 성별

---

#### **2. 테스트 데이터 (Test Data)**
테스트 데이터는 학습 데이터와 동일한 구조를 가지며, **모델의 일반화 성능을 평가**하는 데 사용된다.  
각 데이터셋은 **학습(train) / 검증(validation) / 테스트(test) 데이터셋으로 분할** 되며, 일반적인 분할 비율은 **40% / 20% / 40%** 이다.

- **훈련 데이터 (Training Set):** 모델 학습에 사용됨.  
- **검증 데이터 (Validation Set):** 하이퍼파라미터 튜닝 및 모델 성능 모니터링에 사용됨.  
- **테스트 데이터 (Test Set):** 모델의 최종 성능을 평가하는 데 사용됨.

---

#### **3. 태스크 입력/출력 (Task Input/Output)**

**(1) 입력 (Input):**  
- **하이퍼그래프**: **\( G = (V, E) \)** (노드 집합 **\( V \)**, 하이퍼엣지 집합 **\( E \)**)  
- **노드 특성 행렬**: **\( X \in \mathbb{R}^{N \times m} \)** (N: 노드 개수, m: 특성 차원)  

예제 입력:
```python
{
  "nodes": ["User1", "User2", "User3", "User4"],
  "hyperedges": [["User1", "User2", "User3"], ["User2", "User3", "User4"]],
  "node_features": {
      "User1": [0.1, 0.2, 0.3],
      "User2": [0.5, 0.4, 0.6],
      "User3": [0.7, 0.8, 0.9],
      "User4": [0.2, 0.3, 0.5]
  }
}
```

**(2) 출력 (Output):**  
- **노드 레이블 (Node Labels) 예측**
- **출력 형식:** 각 노드에 대한 예측된 클래스

예제 출력:
```python
{
  "predicted_labels": {
      "User1": "Non-Explicit",
      "User2": "Explicit",
      "User3": "Non-Explicit",
      "User4": "Explicit"
  }
}
```

**출력 설명:**  
- 모델이 각 노드의 속성과 하이퍼그래프 구조를 학습하여 **해당 노드가 속할 클래스를 예측**함.  
- 예제에서는 Twitch-gamers 데이터셋의 **성인 콘텐츠 포함 여부(Explicit vs. Non-Explicit)를 예측**한 결과를 보여줌.  

이처럼 HyperUFG는 **노드 분류, 상품 평점 예측, 소셜 네트워크 분석 등 다양한 이형성 하이퍼그래프 태스크에서 사용 가능**하다.

---



### **1. Training Data**
HyperUFG was trained on various **heterophilic and homophilic hypergraph datasets**. The key training datasets include:

1. **Actor (Movie Co-occurrence Network)**  
   - **Nodes:** Actors, directors, writers.  
   - **Hyperedges:** Groups of people who worked on the same movie.  
   - **Node Features:** Extracted keyword-based features from Wikipedia.  
   - **Labels:** Job categories (actor, director, writer).  

2. **Amazon-ratings (Product Co-purchase Network)**  
   - **Nodes:** Various products (books, music CDs, DVDs, etc.).  
   - **Hyperedges:** Groups of products frequently bought together.  
   - **Node Features:** Bag of Words (BOW) representation extracted from product descriptions.  
   - **Labels:** Average product review rating (10-class classification).  

3. **Twitch-gamers (Streaming User Network)**  
   - **Nodes:** Twitch user accounts.  
   - **Hyperedges:** Users who streamed at the same time.  
   - **Node Features:** View count, account creation/update time, language preferences, streaming hours.  
   - **Labels:** Binary classification (explicit content vs. non-explicit content).  

4. **Pokec (Social Network Data)**  
   - **Nodes:** Social network users.  
   - **Hyperedges:** A user's complete list of friends.  
   - **Node Features:** Age, interests, education level, region, registration time.  
   - **Labels:** User gender classification.  

---

### **2. Test Data**
Test data follows the same structure as training data but is used to evaluate the **generalization performance of the model**.  
Each dataset is **split into training (40%), validation (20%), and testing (40%) sets**.

- **Training Set:** Used to train the model.  
- **Validation Set:** Used for hyperparameter tuning and performance monitoring.  
- **Test Set:** Used to evaluate the final model performance.  

---

### **3. Task Input/Output**

**(1) Input:**  
- **Hypergraph:** **\( G = (V, E) \)** (node set **\( V \)**, hyperedge set **\( E \)**).  
- **Node Feature Matrix:** **\( X \in \mathbb{R}^{N \times m} \)** (N: number of nodes, m: feature dimension).  

Example Input:
```python
{
  "nodes": ["User1", "User2", "User3", "User4"],
  "hyperedges": [["User1", "User2", "User3"], ["User2", "User3", "User4"]],
  "node_features": {
      "User1": [0.1, 0.2, 0.3],
      "User2": [0.5, 0.4, 0.6],
      "User3": [0.7, 0.8, 0.9],
      "User4": [0.2, 0.3, 0.5]
  }
}
```

**(2) Output:**  
- **Predicted node labels.**  
- **Format:** Predicted class for each node.  

Example Output:
```python
{
  "predicted_labels": {
      "User1": "Non-Explicit",
      "User2": "Explicit",
      "User3": "Non-Explicit",
      "User4": "Explicit"
  }
}
```

**Explanation:**  
- The model predicts the **class of each node based on its features and hypergraph structure**.  
- The example output shows the **Twitch-gamers dataset**, where the model predicts whether a user streams explicit content.  

HyperUFG is applicable to **various hypergraph-based tasks**, including **node classification, product rating prediction, and social network analysis**.




<br/>  
# 요약   



HyperUFG는 **프레임렛 기반의 하이퍼그래프 신경망**으로, 저주파 및 고주파 필터를 결합하여 이형성 하이퍼그래프에서도 효과적으로 학습할 수 있도록 설계되었다. 실험 결과, 기존 HNN 모델들이 이형성 환경에서 성능이 저하되는 반면, HyperUFG는 다양한 데이터셋에서 일관되게 우수한 성능을 보였다. 예제 실험에서는 노드 특징과 하이퍼그래프 구조를 기반으로 **사용자 성별, 콘텐츠 유형, 제품 평점 등의 분류 작업을 정확하게 수행**함을 확인하였다.  

---



HyperUFG is a **framelet-based hypergraph neural network** designed to effectively learn in heterophilic hypergraph settings by integrating both low-pass and high-pass filters. Experimental results show that while existing HNN models suffer from performance degradation in heterophilic environments, HyperUFG consistently outperforms them across various datasets. Example experiments demonstrate its ability to **accurately classify user gender, content type, and product ratings based on node features and hypergraph structures**.

<br/>  
# 기타  






#### **1. Figure 1: MLP vs. HNN 모델의 성능 비교 그래프**
- **내용:** 동질성(Homophily) 비율에 따른 MLP(그래프 비의존적 모델)와 HNN(하이퍼그래프 신경망) 성능 비교.  
- **결과:** 동질성 비율이 낮을 때(≤0.5)는 MLP가 HNN보다 높은 성능을 보이지만, 동질성이 증가하면 HNN이 우수한 성능을 발휘.  
- **의미:** 기존 HNN 모델이 이형성 환경에서 취약함을 보여주며, 이를 해결하기 위한 새로운 접근 방식이 필요함을 시사.  

---

#### **2. Table 1: 기존 데이터셋의 동질성 비율 (Homophily Ratios)**
- **내용:** Congress, Senate, House, Walmart 등 기존 데이터셋의 노드 및 하이퍼엣지 동질성 비율 정리.  
- **결과:** 대부분의 데이터셋이 0.5 이상의 동질성 비율을 가지고 있어, 실제로 이형성 환경을 반영하기 어려움.  
- **의미:** 기존 벤치마크 데이터셋의 한계를 보완하기 위해 새로운 이형성 하이퍼그래프 데이터셋이 필요함을 강조.  

---

#### **3. Table 2: 새로운 벤치마크 데이터셋의 통계 정보**
- **내용:** Actor, Amazon-ratings, Twitch-gamers, Pokec 데이터셋의 노드 수, 하이퍼엣지 수, 평균 하이퍼엣지 크기, 클래스 수 등의 정보 제공.  
- **결과:** 기존 데이터셋보다 다양한 크기와 구조를 가지고 있으며, 특히 동질성 비율이 낮아 이형성 연구에 적합함.  
- **의미:** HyperUFG와 같은 모델을 평가하기 위한 보다 현실적인 실험 환경 제공.  

---

#### **4. Table 3 & Table 4: 동질성 및 이형성 데이터셋에서 모델 성능 비교**
- **Table 3:** Cora, Citeseer, Pubmed 등 동질성 데이터셋에서의 모델 성능 비교.  
- **Table 4:** Actor, Amazon-ratings 등 이형성 데이터셋에서의 모델 성능 비교.  
- **결과:** 동질성 데이터셋에서는 기존 HNN 모델들이 우수한 성능을 보였지만, 이형성 데이터셋에서는 대부분 MLP보다 성능이 낮음.  
- **의미:** HyperUFG는 동질성뿐만 아니라 이형성 환경에서도 높은 성능을 유지하는 모델임을 입증.  

---

#### **5. Figure 3: 다양한 동질성 비율에서 모델 성능 비교 그래프**
- **내용:** 동질성 비율(Hedge 및 Hnode)에 따른 여러 모델의 성능 변화 분석.  
- **결과:** 대부분의 HNN 모델들은 동질성이 낮을수록 성능이 급격히 감소하는 반면, HyperUFG는 일정 수준의 성능을 유지.  
- **의미:** HyperUFG가 기존 모델들이 해결하지 못한 **이형성 하이퍼그래프 학습 문제를 효과적으로 해결**함을 입증.  

---



#### **1. Figure 1: Performance Comparison of MLP vs. HNN Models**
- **Description:** Compares the performance of MLP (graph-agnostic model) and HNN (hypergraph neural network) based on homophily ratio.  
- **Results:** MLP performs better when the homophily ratio is low (≤0.5), while HNN models excel as homophily increases.  
- **Implication:** This highlights the **weakness of existing HNN models in heterophilic environments**, emphasizing the need for improved models.  

---

#### **2. Table 1: Homophily Ratios of Existing Datasets**
- **Description:** Summarizes node and hyperedge homophily ratios for datasets like Congress, Senate, House, and Walmart.  
- **Results:** Most datasets have a homophily ratio **above 0.5**, making them less suitable for studying heterophilic hypergraphs.  
- **Implication:** Indicates the **necessity of new benchmark datasets** that better represent heterophilic real-world scenarios.  

---

#### **3. Table 2: Statistics of Newly Developed Benchmark Datasets**
- **Description:** Provides details on the number of nodes, hyperedges, average hyperedge size, number of classes, and homophily ratios for datasets like Actor, Amazon-ratings, Twitch-gamers, and Pokec.  
- **Results:** The new datasets are **larger, more diverse, and have lower homophily ratios**, making them better suited for heterophilic hypergraph learning.  
- **Implication:** These datasets provide a **realistic experimental setup for evaluating models like HyperUFG**.  

---

#### **4. Table 3 & Table 4: Model Performance Comparison on Homophilic and Heterophilic Datasets**
- **Table 3:** Evaluates models on **homophilic datasets** (Cora, Citeseer, Pubmed, etc.).  
- **Table 4:** Evaluates models on **heterophilic datasets** (Actor, Amazon-ratings, etc.).  
- **Results:**  
  - On **homophilic datasets**, traditional HNN models perform well.  
  - On **heterophilic datasets**, most existing HNNs perform worse than MLP.  
  - **HyperUFG consistently outperforms other models across both settings.**  
- **Implication:** HyperUFG is **effective in both homophilic and heterophilic environments**, proving its robustness in various hypergraph structures.  

---

#### **5. Figure 3: Model Performance Comparison Across Different Homophily Ratios**
- **Description:** Shows how different models perform based on varying homophily ratios (Hedge and Hnode).  
- **Results:**  
  - Most HNN models **struggle as homophily decreases**, with a **steep performance drop**.  
  - **HyperUFG maintains strong performance across all homophily levels.**  
- **Implication:** HyperUFG successfully addresses the **challenges of heterophilic hypergraph learning**, making it a superior model compared to existing approaches.  


<br/>
# refer format:     



@article{Li2025Hypergraph,
  author = {Ming Li and Yongchun Gu and Yi Wang and Yujie Fang and Lu Bai and Xiaosheng Zhuang and Pietro Lio},
  title = {When Hypergraph Meets Heterophily: New Benchmark Datasets and Baseline},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2025},
  url = {https://kellysylvia77.github.io/HHL}
}




Ming Li, Yongchun Gu, Yi Wang, Yujie Fang, Lu Bai, Xiaosheng Zhuang, and Pietro Lio. When Hypergraph Meets Heterophily: New Benchmark Datasets and Baseline. Proceedings of the AAAI Conference on Artificial Intelligence, 2025. Accessed at https://kellysylvia77.github.io/HHL.  




