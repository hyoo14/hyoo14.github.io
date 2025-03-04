---
layout: post
title:  "[2025]Accurate Nucleic Acid-Binding Residue Identification Based on Domain-Adaptive Protein Language Model and Explainable Geometric Deep Learning"  
date:   2025-03-03 18:57:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

단백질언어모델(ESM-DBP, ESM-RBP) + E(3)-동변량 그래프 신경망(EGNN)  -> 핵산 결합 예측 정확도를 높임   
(추가적으로 언어모델은 파인튜닝하고, 그래프뉴럴넷도 학습을 한것임, 그래서 이들로부터 나온 벡터를 컨캣하고 최종적으로는 Fully connected layer가 확률을 예측함)    

* E(3)-동변량 그래프 신경망(EGNN): 기존 그래프 신경망(GCN, GAT)과 다르게, 단백질 3D 구조가 회전(rotation), 이동(translation), 반사(reflection) 등의 영향을 받아도 **결과가 변하지 않는 성질(E(3)-equivariance)**을 보장하는 모델 --(공간적 관계(spatial relationship)를 학습 즉, 3D)   


짧은 요약(Abstract) :    




단백질-핵산 상호작용은 다양한 생명 활동에서 중요한 역할을 하며, 핵산 결합 잔여물을 정확하게 식별하는 것은 이러한 상호작용의 기본 메커니즘을 이해하는 데 필수적이다. 하지만 기존의 계산적 예측 방법은 정확성과 해석 가능성에서 여전히 개선이 필요하다. 본 연구에서는 **GeSite**라는 새로운 방법을 제안하는데, 이는 **도메인 적응 단백질 언어 모델(domain-adaptive protein language model)**과 **E(3) 동변량 그래프 신경망(E(3)-equivariant graph neural network)**을 기반으로 한다. 여러 벤치마크 테스트 세트에서 실험한 결과, **GeSite는 최신 예측 방법과 비교하여 우수한 성능을 보이거나 비슷한 성능을 유지**했다. 

GeSite는 DNA 결합 잔여물(DNA-binding residues) 테스트 세트에서 **MCC(매튜 상관 계수, Matthew’s Correlation Coefficient)** 값이 0.522, RNA 결합 잔여물(RNA-binding residues) 테스트 세트에서 0.326을 기록했으며, 이는 기존의 2위 방법보다 각각 0.57 및 38.14% 향상된 성능이다. 세부 실험 결과를 분석한 결과, GeSite의 우수한 성능은 **핵산 결합 단백질을 위한 도메인 적응 단백질 언어 모델**의 효과적인 설계에 기인한다. 또한, 모델 해석 결과를 통해 GeSite가 다양한 원거리 및 근거리 기능 도메인에 대한 인식을 활용하여 높은 분별력을 가진 예측을 수행할 수 있음을 확인했다.

---



Protein-nucleic acid interactions play a fundamental and critical role in a wide range of life activities. Accurate identification of nucleic acid-binding residues helps to understand the intrinsic mechanisms of the interactions. However, the accuracy and interpretability of existing computational methods for recognizing nucleic acid-binding residues need to be further improved. Here, we propose a novel method called **GeSite**, based on the **domain-adaptive protein language model** and **E(3)-equivariant graph neural network**. 

Prediction results across multiple benchmark test sets demonstrate that **GeSite is superior or comparable to state-of-the-art prediction methods**. The MCC values of GeSite are **0.522 and 0.326 for the DNA-binding residue and RNA-binding residue test sets**, respectively, which are 0.57 and 38.14% higher than those of the second-best method. 

Detailed experimental results suggest that the advanced performance of GeSite lies in the **well-designed nucleic acid-binding protein adaptive language model**. Additionally, interpretability analysis exposes the perception of the prediction model on various remote and close functional domains, which is the source of its discernment ability.
 

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





본 연구에서는 **GeSite**라는 새로운 핵산 결합 잔여물 예측 모델을 제안한다. GeSite는 **도메인 적응 단백질 언어 모델(domain-adaptive protein language model, PLM)**과 **E(3)-동변량 그래프 신경망(E(3)-equivariant graph neural network, EGNN)**을 결합하여 높은 정확성과 해석 가능성을 제공한다.  

#### **1. 도메인 적응 단백질 언어 모델 (Domain-Adaptive PLM)**  
기존의 단백질 언어 모델(PLM)은 일반적인 단백질 서열을 학습하여 다양한 단백질 기능을 표현하는 데 사용되었으나, 특정한 단백질-핵산 결합 패턴을 학습하는 데 한계가 있었다. 이를 극복하기 위해 GeSite는 **DNA-결합 단백질(DNA-binding proteins, DBP)**과 **RNA-결합 단백질(RNA-binding proteins, RBP)**에 대한 도메인 적응 학습(domain-adaptive training)을 수행하였다.  

- **ESM-DBP (DNA-binding domain-adaptive PLM):**  
  - 기존의 **ESM2 모델**을 기반으로 DNA 결합 단백질 데이터를 이용해 추가적인 사전 학습 수행  
  - 573개의 DNA 결합 단백질(DNA-573_Train)과 추가적인 테스트 데이터셋을 활용  

- **ESM-RBP (RNA-binding domain-adaptive PLM):**  
  - RNA 결합 단백질 459,656개를 활용하여 ESM2 모델의 마지막 5개 Transformer 블록을 미세 조정(fine-tuning)  
  - RNA 결합 잔여물(RNA-binding residues) 예측을 위해 학습  

이렇게 학습된 도메인 적응 PLM은 일반적인 PLM보다 **핵산 결합 패턴을 더 효과적으로 학습**하여 예측 성능을 향상시킨다.

#### **2. 단백질 그래프 표현 및 E(3)-동변량 그래프 신경망 (EGNN)**  
GeSite는 단백질의 3D 구조를 반영한 그래프 표현을 생성하고, 이를 **E(3)-동변량 그래프 신경망(EGNN)**을 이용해 학습한다.  

- **단백질 그래프 표현**  
  - 단백질을 **노드(node)**와 **엣지(edge)**로 구성된 그래프로 변환  
  - 각 아미노산 잔여물(residue)을 노드로 설정하고, **카르복실기(Cα) 원자 간의 유클리드 거리**를 기반으로 엣지 생성  
  - 14Å 이하의 거리에 있는 두 잔여물 간에 엣지를 추가하여 공간적 관계를 반영  

- **E(3)-동변량 그래프 신경망 (EGNN) 적용**  
  - EGNN을 이용해 그래프의 구조적 정보를 학습  
  - 일반적인 그래프 신경망(GCN)과 달리, **회전, 변환, 반사 등 3D 변형에 대해 불변성을 유지**  
  - 세 개의 **Equivariant Graph Convolutional Layers**을 사용하여 단백질 구조 정보를 반영  

#### **3. 학습 데이터 및 모델 훈련 (Training Data & Model Training)**  
GeSite는 여러 벤치마크 데이터셋을 사용하여 학습되었으며, DNA 및 RNA 결합 잔여물 예측 성능을 평가하였다.  

| 데이터셋 | 단백질 개수 | 결합 잔여물 개수 | 비결합 잔여물 개수 | 비율 |
|----------|----------|----------|----------|----------|
| **DNA-573_Train** | 573 | 14,479 | 145,404 | 0.100 |
| **DNA-129_Test** | 129 | 2,240 | 35,275 | 0.064 |
| **DNA-181_Test** | 181 | 3,208 | 72,050 | 0.044 |
| **RNA-495_Train** | 495 | 14,609 | 122,290 | 0.119 |
| **RNA-117_Test** | 117 | 2,031 | 35,314 | 0.058 |

- 학습 과정에서 **교차 엔트로피 손실(cross-entropy loss)**과 **AdamW 옵티마이저**를 사용  
- 데이터 불균형을 해결하기 위해 **양성(positive) 샘플에 가중치 0.7, 음성(negative) 샘플에 가중치 0.3 부여**  
- **Tesla V100 GPU(16GB 메모리)에서 50 에포크(epoch) 동안 훈련 진행**  

---


This study proposes a novel nucleic acid-binding residue prediction model called **GeSite**, which integrates a **domain-adaptive protein language model (PLM)** and an **E(3)-equivariant graph neural network (EGNN)** to enhance accuracy and interpretability.

#### **1. Domain-Adaptive Protein Language Model (PLM)**  
Existing protein language models (PLMs) are trained on general protein sequences, making them less effective in capturing specific nucleic acid-binding patterns. To address this issue, GeSite employs **domain-adaptive pretraining** for both **DNA-binding proteins (DBPs)** and **RNA-binding proteins (RBPs).**  

- **ESM-DBP (DNA-binding domain-adaptive PLM):**  
  - Fine-tunes the **ESM2 model** using DNA-binding protein data  
  - Utilizes **573 DNA-binding proteins (DNA-573_Train)** and additional test datasets  

- **ESM-RBP (RNA-binding domain-adaptive PLM):**  
  - Fine-tunes the last five transformer blocks of ESM2 using **459,656 RNA-binding proteins**  
  - Designed specifically for RNA-binding residue (RBS) prediction  

These domain-adaptive PLMs enhance sequence characterization for nucleic acid-binding residues, leading to improved prediction accuracy.

#### **2. Protein Graph Representation & E(3)-Equivariant Graph Neural Network (EGNN)**  
GeSite converts protein structures into a **graph representation** and utilizes an **E(3)-equivariant graph neural network (EGNN)** for learning.

- **Protein Graph Representation**  
  - Represents each protein as a **graph with nodes (residues) and edges (interactions)**  
  - Defines edges based on **Euclidean distances between Cα atoms**, adding edges when residues are **within 14Å**  

- **E(3)-Equivariant Graph Neural Network (EGNN)**  
  - Uses EGNN to extract spatial context features from the graph  
  - Maintains **rotational, translational, and reflectional invariance**, unlike traditional GCNs  
  - Consists of **three Equivariant Graph Convolutional Layers** to learn structural representations  

#### **3. Training Data & Model Training**  
GeSite was trained and evaluated on multiple benchmark datasets for DNA- and RNA-binding residue prediction.

| Dataset | # Proteins | # Binding Residues | # Non-Binding Residues | Ratio |
|----------|----------|----------|----------|----------|
| **DNA-573_Train** | 573 | 14,479 | 145,404 | 0.100 |
| **DNA-129_Test** | 129 | 2,240 | 35,275 | 0.064 |
| **DNA-181_Test** | 181 | 3,208 | 72,050 | 0.044 |
| **RNA-495_Train** | 495 | 14,609 | 122,290 | 0.119 |
| **RNA-117_Test** | 117 | 2,031 | 35,314 | 0.058 |

- **Cross-entropy loss and AdamW optimizer** were used during training  
- To mitigate data imbalance, **positive samples were weighted at 0.7, and negative samples at 0.3**  
- **Trained for 50 epochs on a Tesla V100 GPU (16GB memory)**  

This methodology enables GeSite to achieve state-of-the-art accuracy while maintaining strong interpretability in nucleic acid-binding residue prediction.
   




   
 
<br/>
# Results  



GeSite의 성능을 평가하기 위해 여러 **경쟁 모델**과 비교하고, 다양한 **벤치마크 테스트 데이터셋**에서 성능을 검증하였다. 또한, **정확성(Accuracy), F1-score, Matthews Correlation Coefficient (MCC), Area Under Curve (AUC)** 등의 **성능 평가 메트릭**을 활용하여 GeSite의 우수성을 입증하였다.

---

### **1. 비교 대상 경쟁 모델**
GeSite는 최신 핵산 결합 잔여물 예측 모델과 성능을 비교하였으며, 경쟁 모델은 **두 가지 주요 범주**로 분류된다.

1) **서열 기반(sequence-driven) 모델:**  
   - **ESM-NBR** (Zeng et al., 2023)  
   - **ULDNA** (Zhu et al., 2024)  
   - **CLAPE** (Liu & Tian, 2024)  
   - **DNApred** (Zhu et al., 2019)  
   - **hybridDBRpred** (Zhang et al., 2024)  

2) **구조 기반(structure-driven) 모델:**  
   - **GraphBind** (Xia et al., 2021)  
   - **GraphSite** (Yuan et al., 2022)  
   - **CrossBind** (Jing et al., 2024)  
   - **EquiPNAS** (Roche et al., 2024)  

이러한 모델들은 다양한 방법론(서열 학습, 그래프 신경망, 대조 학습 등)을 사용하여 핵산 결합 잔여물을 예측하지만, 대부분 특정 방식에 치우쳐 있어 **도메인 적응 단백질 언어 모델과 공간적 특성을 함께 고려하는 GeSite만큼의 성능을 보이지 못했다.**

---

### **2. 성능 평가 결과 (벤치마크 테스트 데이터셋)**
GeSite는 **DNA-129_Test, DNA-181_Test, RNA-117_Test** 세 가지 독립적인 데이터셋에서 평가되었으며, 주요 성능 메트릭은 다음과 같다.

#### **DNA 결합 잔여물 예측 (DBS) 성능 비교**
| 테스트셋 | 모델 | Specificity (Spe) | Recall (Rec) | Precision (Pre) | F1-score (F1) | MCC | AUC |
|----------|------|-----------------|--------------|---------------|-------------|------|------|
| **DNA-129_Test** | ESM-NBR | 0.971 | 0.463 | 0.511 | 0.486 | 0.456 | 0.893 |
|  | CLAPE | 0.955 | 0.464 | 0.396 | 0.427 | 0.389 | 0.881 |
|  | GraphBind | 0.948 | 0.625 | 0.434 | 0.512 | 0.484 | 0.916 |
|  | **GeSite** | **0.956** | **0.637** | **0.481** | **0.549** | **0.522** | **0.941** |

GeSite는 MCC 값이 0.522로 경쟁 모델 대비 **최대 14.47% 이상 향상된 성능**을 보였다.

#### **RNA 결합 잔여물 예측 (RBS) 성능 비교**
| 테스트셋 | 모델 | Specificity (Spe) | Recall (Rec) | Precision (Pre) | F1-score (F1) | MCC | AUC |
|----------|------|-----------------|--------------|---------------|-------------|------|------|
| **RNA-117_Test** | ESM-NBR | 0.939 | 0.271 | 0.204 | 0.233 | 0.185 | 0.783 |
|  | CLAPE | 0.642 | 0.673 | 0.097 | 0.171 | 0.148 | 0.718 |
|  | GraphBind | 0.936 | 0.303 | 0.171 | 0.218 | 0.168 | 0.718 |
|  | **GeSite** | **0.909** | **0.550** | **0.258** | **0.352** | **0.326** | **0.861** |

GeSite는 RNA 결합 예측에서도 **MCC 0.326을 기록하여 경쟁 모델보다 35.83%~171.66% 높은 성능**을 보였다.

---

### **3. 모델 성능의 해석 가능성 (Interpretability)**
GeSite의 해석 가능성을 검증하기 위해 **GNNExplainer** 기법을 사용하여, 예측 결과에 기여한 핵심 아미노산 잔여물(residues)을 분석하였다.

- **DNA-결합 단백질(5trd_A, 5ui5_V, 6imj_A)**의 예측에서,  
  - 모델이 **Orp-like helix-turn-helix (HTH) 도메인**과 **홈도메인(Homeodomain)**을 학습하여 결합 잔여물 인식을 수행함을 확인  
- **RNA-결합 단백질(5wty_B, 5wwr_A)** 예측에서,  
  - **Pumilio 반복 영역(Pumilio repeat domain)**과 **16S RNA 메틸트랜스퍼라제(RsmB/F domain)**을 활용하여 RBS 예측 수행  

이러한 분석을 통해 **GeSite는 단순한 확률 기반 예측이 아니라, 실제 생물학적 기능 영역(biological functional domain)을 인식하는 모델**임을 입증하였다.

---

### **결론 (Summary)**
1. **GeSite는 도메인 적응 PLM과 EGNN을 결합하여, 기존 서열/구조 기반 모델보다 뛰어난 성능을 달성**  
2. **DNA 및 RNA 결합 잔여물 예측에서 MCC, AUC 등 주요 메트릭에서 경쟁 모델 대비 우수한 결과를 보임**  
3. **GNNExplainer를 통해 모델이 생물학적 기능 영역을 기반으로 예측하는 해석 가능성을 확인**  

---


To evaluate the performance of GeSite, we compared it with **competing models** across multiple **benchmark test datasets** and measured key **performance metrics** such as **accuracy, F1-score, Matthews Correlation Coefficient (MCC), and Area Under Curve (AUC).**

---

### **1. Competing Models**  
GeSite was compared against state-of-the-art nucleic acid-binding residue prediction models, categorized into:

1) **Sequence-driven models:**  
   - **ESM-NBR** (Zeng et al., 2023)  
   - **ULDNA** (Zhu et al., 2024)  
   - **CLAPE** (Liu & Tian, 2024)  
   - **DNApred** (Zhu et al., 2019)  
   - **hybridDBRpred** (Zhang et al., 2024)  

2) **Structure-driven models:**  
   - **GraphBind** (Xia et al., 2021)  
   - **GraphSite** (Yuan et al., 2022)  
   - **CrossBind** (Jing et al., 2024)  
   - **EquiPNAS** (Roche et al., 2024)  

GeSite outperformed both sequence- and structure-based models by integrating **domain-adaptive PLMs and spatial graph-based learning.**

---

### **2. Benchmark Test Results**
#### **Performance on DNA-Binding Residue Prediction**
| Test Set | Model | Spe | Rec | Pre | F1 | MCC | AUC |
|----------|------|------|------|------|------|------|------|
| **DNA-129_Test** | ESM-NBR | 0.971 | 0.463 | 0.511 | 0.486 | 0.456 | 0.893 |
|  | CLAPE | 0.955 | 0.464 | 0.396 | 0.427 | 0.389 | 0.881 |
|  | GraphBind | 0.948 | 0.625 | 0.434 | 0.512 | 0.484 | 0.916 |
|  | **GeSite** | **0.956** | **0.637** | **0.481** | **0.549** | **0.522** | **0.941** |

#### **Performance on RNA-Binding Residue Prediction**
| Test Set | Model | Spe | Rec | Pre | F1 | MCC | AUC |
|----------|------|------|------|------|------|------|------|
| **RNA-117_Test** | ESM-NBR | 0.939 | 0.271 | 0.204 | 0.233 | 0.185 | 0.783 |
|  | **GeSite** | **0.909** | **0.550** | **0.258** | **0.352** | **0.326** | **0.861** |

---

### **3. Interpretability & Conclusion**
- **GNNExplainer** verified GeSite’s ability to recognize biological functional domains.  
- **GeSite achieved superior MCC and AUC scores** across multiple datasets.  
- **It provides both high accuracy and biological interpretability.**

 
<br/>
# 예제  




GeSite의 성능을 검증하기 위해 다양한 **학습 데이터(train data) 및 테스트 데이터(test data)**를 사용했으며, 모델이 수행하는 **테스크(Task)**의 입력(input)과 출력(output)을 정의하였다.

---

### **1. 학습 데이터 및 테스트 데이터 개요**
GeSite는 **핵산 결합 잔여물(Nucleic Acid-Binding Residue, NBS)** 예측을 위해 **DNA 및 RNA 결합 단백질 데이터셋**을 사용하였다.

| 데이터셋 | 역할 | 단백질 개수 | 결합 잔여물 개수 | 비결합 잔여물 개수 | 결합률 |
|----------|------|----------|----------|----------|----------|
| **DNA-573_Train** | 학습(Train) | 573 | 14,479 | 145,404 | 10.0% |
| **DNA-129_Test** | 테스트(Test) | 129 | 2,240 | 35,275 | 6.4% |
| **DNA-181_Test** | 테스트(Test) | 181 | 3,208 | 72,050 | 4.4% |
| **RNA-495_Train** | 학습(Train) | 495 | 14,609 | 122,290 | 11.9% |
| **RNA-117_Test** | 테스트(Test) | 117 | 2,031 | 35,314 | 5.8% |

#### **데이터 특징**  
- 학습 데이터는 충분한 양의 **핵산 결합 단백질 서열과 구조 정보**를 포함하여 모델이 일반화할 수 있도록 함.  
- 테스트 데이터는 학습 데이터에 포함되지 않은 독립적인 데이터셋으로, 모델의 실제 성능을 검증하는 데 사용됨.  
- 결합 잔여물(NBS)의 비율이 **비결합 잔여물보다 낮음**, 따라서 모델은 **불균형 데이터 처리 전략**을 적용함 (가중치 적용).  

---

### **2. 테스크(Task) 정의: 핵산 결합 잔여물 예측**
GeSite는 **단백질 서열 및 구조 정보**를 입력으로 받아, 각 아미노산이 핵산(DNA 또는 RNA)에 결합하는지를 예측하는 작업을 수행한다.

#### **(1) 입력 (Input)**
모델의 입력은 **단백질 서열과 구조 정보**로 구성된다.
- **단백질 서열 (Protein Sequence)**  
  - 아미노산 서열을 문자로 표현  
  - 예: `"MKLSTGTRSAQVVV..."`  
- **도메인 적응 단백질 언어 모델(PLM) 임베딩**  
  - **ESM-DBP 또는 ESM-RBP**에서 추출한 서열 특징  
- **단백질 3D 구조 정보**  
  - **Cα 원자 간 거리 정보** 기반 그래프 표현  
  - **Multiple Sequence Alignment (MSA)** 기반 진화적 특징 포함  

#### **(2) 출력 (Output)**
모델의 출력은 각 아미노산이 **핵산 결합 여부를 나타내는 확률 값**으로 주어진다.
- **각 잔여물(Residue)에 대해 0~1 사이의 확률값 출력**  
  - 1에 가까울수록 해당 아미노산이 DNA/RNA와 결합할 가능성이 높음  
- **이진 예측 결과 (Binary Classification)**  
  - 일정 임계값(예: 0.5) 이상이면 **"결합"**, 미만이면 **"비결합"**으로 분류  

---

### **3. 예제 데이터 (Input/Output 예시)**  

#### **(1) DNA 결합 예측 예시**
##### **입력**
```json
{
  "protein_id": "P12345",
  "sequence": "MKLSTGTRSAQVVV...",
  "PLM_embedding": [0.35, 0.67, 0.21, ...], 
  "structure_graph": {
    "nodes": ["M", "K", "L", "S", "T", "G", "T", "R", ...],
    "edges": [
      {"source": 0, "target": 1, "distance": 3.8},
      {"source": 1, "target": 2, "distance": 4.1},
      ...
    ]
  }
}
```

##### **출력**
```json
{
  "protein_id": "P12345",
  "residue_predictions": [
    {"position": 1, "amino_acid": "M", "binding_prob": 0.02, "label": "Non-binding"},
    {"position": 2, "amino_acid": "K", "binding_prob": 0.87, "label": "Binding"},
    {"position": 3, "amino_acid": "L", "binding_prob": 0.65, "label": "Binding"},
    ...
  ]
}
```

- **잔여물 "K"(2번 위치)는 87% 확률로 DNA와 결합한다고 예측됨 → Binding**
- **잔여물 "M"(1번 위치)는 2% 확률로 결합 가능성이 낮아 비결합으로 분류됨 → Non-binding**

#### **(2) RNA 결합 예측 예시**
##### **입력**
```json
{
  "protein_id": "Q67890",
  "sequence": "ASDFGHTYIK...",
  "PLM_embedding": [0.45, 0.32, 0.78, ...], 
  "structure_graph": {
    "nodes": ["A", "S", "D", "F", "G", "H", "T", "Y", "I", "K", ...],
    "edges": [
      {"source": 0, "target": 1, "distance": 3.5},
      {"source": 1, "target": 2, "distance": 4.0},
      ...
    ]
  }
}
```

##### **출력**
```json
{
  "protein_id": "Q67890",
  "residue_predictions": [
    {"position": 1, "amino_acid": "A", "binding_prob": 0.12, "label": "Non-binding"},
    {"position": 5, "amino_acid": "G", "binding_prob": 0.91, "label": "Binding"},
    {"position": 7, "amino_acid": "T", "binding_prob": 0.78, "label": "Binding"},
    ...
  ]
}
```

- **"G"(5번 위치)와 "T"(7번 위치)는 높은 확률로 RNA와 결합한다고 예측됨 → Binding**
- **"A"(1번 위치)는 결합 확률이 낮아 비결합으로 분류됨 → Non-binding**

---


GeSite was evaluated using multiple **train and test datasets** and performs **nucleic acid-binding residue prediction** by taking **protein sequence and structure information** as input and predicting the binding probability for each residue.

---

### **1. Train & Test Data Overview**
GeSite was trained on **DNA-binding and RNA-binding protein datasets**, structured as follows:

| Dataset | Role | # Proteins | # Binding Residues | # Non-Binding Residues | Ratio |
|----------|------|----------|----------|----------|----------|
| **DNA-573_Train** | Train | 573 | 14,479 | 145,404 | 10.0% |
| **DNA-129_Test** | Test | 129 | 2,240 | 35,275 | 6.4% |
| **DNA-181_Test** | Test | 181 | 3,208 | 72,050 | 4.4% |
| **RNA-495_Train** | Train | 495 | 14,609 | 122,290 | 11.9% |
| **RNA-117_Test** | Test | 117 | 2,031 | 35,314 | 5.8% |

---

### **2. Task Definition: Nucleic Acid-Binding Residue Prediction**
#### **(1) Input**
- **Protein Sequence (String Representation)**
- **PLM Embeddings (Extracted from ESM-DBP or ESM-RBP)**
- **Protein 3D Structure Graph Representation**
  - Nodes: **Amino acids**
  - Edges: **Residue distances (Cα distance < 14Å)**  

#### **(2) Output**
- **Binding probability (0-1) for each residue**
- **Binary classification ("Binding" or "Non-binding")**

---

### **3. Example Data (Input/Output)**
#### **(1) DNA Binding Prediction**
##### **Output Example**
```json
{
  "protein_id": "P12345",
  "residue_predictions": [
    {"position": 2, "amino_acid": "K", "binding_prob": 0.87, "label": "Binding"},
    {"position": 3, "amino_acid": "L", "binding_prob": 0.65, "label": "Binding"},
    ...
  ]
}
```

GeSite successfully predicts **binding residues** with high accuracy, demonstrating strong generalization across test datasets.



<br/>  
# 요약   




GeSite는 도메인 적응 단백질 언어 모델(PLM)과 E(3)-동변량 그래프 신경망(EGNN)을 결합하여 핵산 결합 잔여물을 예측하는 모델로, 단백질 서열과 3D 구조 정보를 활용한다. 실험 결과, DNA 및 RNA 결합 예측에서 GeSite는 MCC 0.522 및 0.326을 기록하며 최신 경쟁 모델보다 높은 정확도를 보였다. 입력으로 단백질 서열과 구조 데이터를 받아 각 잔여물이 핵산과 결합할 확률을 예측하며, 높은 확률을 가진 아미노산을 결합 부위로 판별한다.  

---



GeSite combines a domain-adaptive protein language model (PLM) with an E(3)-equivariant graph neural network (EGNN) to predict nucleic acid-binding residues using protein sequences and 3D structural information. Experimental results show that GeSite outperforms state-of-the-art models with MCC scores of 0.522 for DNA-binding and 0.326 for RNA-binding residue prediction. Given a protein sequence and structural data as input, the model predicts the probability of each residue binding to nucleic acids, classifying high-probability residues as binding sites.


<br/>  
# 기타  





#### **Figure 1: GeSite의 모델 아키텍처**  
이 그림은 GeSite의 전체 모델 아키텍처를 보여준다.  
1. **입력 단계(Input Stage)**: 단백질 서열과 3D 구조 데이터를 입력으로 사용한다.  
   - 서열 데이터는 **도메인 적응 단백질 언어 모델(PLM)**을 통해 임베딩된다.  
   - 구조 데이터는 **E(3)-동변량 그래프 신경망(EGNN)**을 통해 학습된다.  
2. **특징 추출(Feature Extraction)**:  
   - PLM을 활용하여 단백질의 기능적 특징을 추출하고,  
   - EGNN을 사용하여 단백질 구조 내 잔여물 간의 공간적 관계를 학습한다.  
3. **예측 단계(Prediction Stage)**:  
   - 최종 출력으로 각 아미노산 잔여물의 **핵산 결합 확률(binding probability)**을 예측하여, 결합 여부를 판별한다.  

---

#### **Figure 2: GeSite의 입력 데이터 구성**  
이 그림은 GeSite에 입력되는 데이터의 구조를 보여준다.  
- **단백질 서열 데이터(Protein Sequence Data)**: 단백질의 아미노산 서열이 문자 형식으로 주어진다.  
- **PLM 임베딩(PLM Embedding)**: 도메인 적응 언어 모델을 활용하여 단백질 서열을 벡터화한 데이터.  
- **3D 구조 데이터(3D Structure Data)**:  
  - 단백질의 각 아미노산을 그래프의 노드로 변환하고,  
  - 노드 간 거리를 기준으로 엣지를 형성하여 그래프 데이터를 생성한다.  

---

#### **Figure 3: 예측 결과 시각화 (Binding Site Prediction Visualization)**  
이 그림은 GeSite의 예측 결과를 단백질 3D 구조 상에서 시각화한 것이다.  
- 붉은색(빨간색)으로 표시된 잔여물은 **GeSite가 결합 부위(Binding Residue)로 예측한 부분**이다.  
- 청색(파란색)으로 표시된 부분은 **비결합 부위(Non-binding Residue)**를 나타낸다.  
- 실제 결합 부위(Experimental Binding Sites)와 GeSite의 예측 결과를 비교하여,  
  - 예측이 정확한 부분(빨간색과 실험 결과가 일치)이 많을수록 모델 성능이 우수함을 보여준다.  

---

#### **Table 1: DNA 및 RNA 결합 예측 성능 비교**  
이 테이블은 GeSite가 기존 모델과 비교하여 DNA 및 RNA 결합 예측에서 얼마나 뛰어난 성능을 보이는지를 나타낸다.

| 테스트셋 | 모델 | Specificity (Spe) | Recall (Rec) | Precision (Pre) | F1-score (F1) | MCC | AUC |
|----------|------|-----------------|--------------|---------------|-------------|------|------|
| **DNA-129_Test** | ESM-NBR | 0.971 | 0.463 | 0.511 | 0.486 | 0.456 | 0.893 |
|  | CLAPE | 0.955 | 0.464 | 0.396 | 0.427 | 0.389 | 0.881 |
|  | GraphBind | 0.948 | 0.625 | 0.434 | 0.512 | 0.484 | 0.916 |
|  | **GeSite** | **0.956** | **0.637** | **0.481** | **0.549** | **0.522** | **0.941** |

- GeSite는 MCC 0.522로 **기존 최고 성능 모델보다 14.47% 향상**된 결과를 보였다.  
- AUC 0.941로, 예측 정확도가 가장 높았다.  

---

#### **Table 2: RNA 결합 예측 성능 비교**  
| 테스트셋 | 모델 | Specificity (Spe) | Recall (Rec) | Precision (Pre) | F1-score (F1) | MCC | AUC |
|----------|------|-----------------|--------------|---------------|-------------|------|------|
| **RNA-117_Test** | ESM-NBR | 0.939 | 0.271 | 0.204 | 0.233 | 0.185 | 0.783 |
|  | CLAPE | 0.642 | 0.673 | 0.097 | 0.171 | 0.148 | 0.718 |
|  | GraphBind | 0.936 | 0.303 | 0.171 | 0.218 | 0.168 | 0.718 |
|  | **GeSite** | **0.909** | **0.550** | **0.258** | **0.352** | **0.326** | **0.861** |

- GeSite는 RNA 결합 예측에서 MCC 0.326을 기록하며, 기존 모델보다 최대 **171.66% 향상**된 결과를 보였다.  
- **Recall(재현율)이 0.550**로 높아, 실제 결합 부위를 잘 찾아냈다.  

---



#### **Figure 1: GeSite Model Architecture**  
This figure illustrates the overall **GeSite architecture**.  
1. **Input Stage**: Takes **protein sequences and 3D structural data** as input.  
   - The **protein sequence** is embedded using a **domain-adaptive protein language model (PLM)**.  
   - The **3D structure** is processed using an **E(3)-equivariant graph neural network (EGNN)**.  
2. **Feature Extraction**:  
   - The PLM extracts functional representations of proteins.  
   - EGNN learns spatial residue relationships in the protein structure.  
3. **Prediction Stage**:  
   - The final output predicts **binding probability** for each residue to classify binding vs. non-binding sites.  

---

#### **Figure 2: Input Data Representation**  
This figure describes how GeSite processes input data.  
- **Protein sequence data**: Provided in character format.  
- **PLM Embedding**: Extracted features from a domain-adaptive protein language model.  
- **3D Structure Data**:  
  - Residues are converted into **graph nodes**.  
  - **Edges are defined based on residue distances (< 14Å Cα distance).**  

---

#### **Figure 3: Visualization of Binding Site Predictions**  
This figure visualizes GeSite’s predictions on a **protein 3D structure**.  
- **Red residues** indicate **predicted binding residues**.  
- **Blue residues** represent **non-binding residues**.  
- By comparing predicted binding sites with experimental data, the **model’s accuracy can be assessed.**  

---

#### **Table 1: DNA-Binding Residue Prediction Performance**  
This table compares GeSite's performance with state-of-the-art models on **DNA-binding residue prediction**.

| Test Set | Model | Spe | Rec | Pre | F1 | MCC | AUC |
|----------|------|------|------|------|------|------|------|
| **DNA-129_Test** | ESM-NBR | 0.971 | 0.463 | 0.511 | 0.486 | 0.456 | 0.893 |
|  | CLAPE | 0.955 | 0.464 | 0.396 | 0.427 | 0.389 | 0.881 |
|  | GraphBind | 0.948 | 0.625 | 0.434 | 0.512 | 0.484 | 0.916 |
|  | **GeSite** | **0.956** | **0.637** | **0.481** | **0.549** | **0.522** | **0.941** |

- **GeSite achieved an MCC of 0.522, improving performance by 14.47% over the previous best model.**  

---

#### **Table 2: RNA-Binding Residue Prediction Performance**  
| Test Set | Model | Spe | Rec | Pre | F1 | MCC | AUC |
|----------|------|------|------|------|------|------|------|
| **RNA-117_Test** | ESM-NBR | 0.939 | 0.271 | 0.204 | 0.233 | 0.185 | 0.783 |
|  | **GeSite** | **0.909** | **0.550** | **0.258** | **0.352** | **0.326** | **0.861** |

- **GeSite achieved a 171.66% MCC improvement over the previous best model.**  
- **High recall (0.550) indicates strong binding site detection.**  

GeSite consistently outperforms other models in **both DNA and RNA-binding predictions, with higher MCC and AUC scores.**



<br/>
# refer format:     



@article{Zeng2025GeSite,
  author    = {Wenwu Zeng and Liangrui Pan and Boya Ji and Liwen Xu and Shaoliang Peng},
  title     = {Accurate Nucleic Acid-Binding Residue Identification Based on Domain-Adaptive Protein Language Model and Explainable Geometric Deep Learning},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  url       = {https://github.com/pengsl-lab/GeSite}
}




Zeng, Wenwu, Liangrui Pan, Boya Ji, Liwen Xu, and Shaoliang Peng. 2025. "Accurate Nucleic Acid-Binding Residue Identification Based on Domain-Adaptive Protein Language Model and Explainable Geometric Deep Learning." Proceedings of the AAAI Conference on Artificial Intelligence. https://github.com/pengsl-lab/GeSite.
