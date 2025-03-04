---
layout: post
title:  "[2025]CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction"  
date:   2025-03-03 20:18:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


크게보면 단백질언어모델+rna언어모델인데  
임베딩을 추출해서 새로 트랜스포머 모델에서 학습을 함(파인튜닝)  
근데 단순 컨캣은 아니고 제3의 정보(거리나 결합 강도)기반.....   

즉,

단백질과 rna서열을 입력하면 트랜스포머구조에서 결합도나 거리를 내뱉도록 학습   
->그 학습 결과의 트랜스포머로부터 임베딩을 뽑아내면   
->그 임베딩이 사실상 결합된 임베딩   
->그러면 그렇게 나온 임베딩을 새로운 파운데이션 모델로 써서 뭐 다른 테스크들 수행   

어렵다..   

짧은 요약(Abstract) :    




단백질-RNA 결합 친화도를 정확하게 측정하는 것은 다양한 생물학적 과정과 신약 개발에서 중요한 역할을 한다. 기존의 예측 방법들은 단백질-RNA 결합 메커니즘을 완전하게 포착하지 못하며, 주로 서열 또는 구조적 특징 중 하나에만 의존하는 한계를 가진다. 최근 단백질과 RNA의 대규모 비지도 학습 데이터를 활용한 사전 학습 언어 모델이 등장하면서, 결합 부위 예측 등 다양한 생물학적 작업에서 뛰어난 성능을 보이고 있다. 하지만 서로 다른 생물학적 도메인의 언어 모델을 협력적으로 활용하여 복합 수준의 작업을 수행하는 연구는 거의 이루어지지 않았다. 본 논문에서는 **CoPRA(Co-Former for Protein-RNA binding Affinity prediction)**를 제안하여, 서로 다른 생물학적 도메인의 사전 학습된 언어 모델을 복합 구조를 통해 연결하는 방식을 제시한다. 이를 통해 처음으로 교차 생물학적 언어 모델이 협력하여 단백질-RNA 결합 친화도 예측 성능을 향상시킬 수 있음을 입증하였다. 

CoPRA는 **Co-Former**를 도입하여 교차 모달(sequence & structure) 정보를 결합하며, 모델의 상호작용 이해를 향상시키기 위해 **이중 시야 사전 학습 전략(bi-scope pre-training strategy)**을 사용한다. 또한, 가장 큰 단백질-RNA 결합 친화도 데이터셋인 **PRA310**을 구축하여 모델의 성능을 평가하였으며, 돌연변이 효과 예측(public dataset)을 통해도 테스트를 진행했다. CoPRA는 모든 데이터셋에서 최신(state-of-the-art) 성능을 달성했으며, 다음과 같은 주요 기여점을 가진다:
1. 단백질과 RNA 언어 모델을 복합 구조 정보를 통해 결합하여 결합 친화도를 예측하는 최초의 방법론을 제시함.
2. **Co-Former**를 활용하여 교차 모달 정보를 효과적으로 결합하며, 이중 시야 사전 학습(CPRI 및 MIDM)을 적용하여 상호작용을 다각적으로 학습할 수 있도록 설계됨.
3. **PRA310**이라는 가장 큰 단백질-RNA 결합 친화도 데이터셋을 구축하고, 여러 공개 데이터셋에서 CoPRA의 성능을 평가하여 최고 수준의 결과를 도출함.



---



Accurately measuring protein-RNA binding affinity is crucial in many biological processes and drug design. Previous computational methods for protein-RNA binding affinity prediction rely on either sequence or structure features, unable to capture the binding mechanisms comprehensively. The recent emerging pre-trained language models trained on massive unsupervised sequences of protein and RNA have shown strong representation ability for various in-domain downstream tasks, including binding site prediction. However, applying different-domain language models collaboratively for complex-level tasks remains unexplored. 

In this paper, we propose **CoPRA** to bridge pre-trained language models from different biological domains via **Complex structure for Protein-RNA binding Affinity prediction**. We demonstrate for the first time that cross-biological modal language models can collaborate to improve binding affinity prediction. We propose a **Co-Former** to combine the cross-modal sequence and structure information and a **bi-scope pre-training strategy** for improving Co-Former’s interaction understanding. Meanwhile, we build the largest **protein-RNA binding affinity dataset PRA310** for performance evaluation. We also test our model on a public dataset for mutation effect prediction. CoPRA reaches **state-of-the-art performance** on all the datasets. 

Our key contributions are:
1. We propose **CoPRA**, the first attempt to combine protein and RNA language models with complex structure information for protein-RNA binding affinity prediction.
2. We design a **Co-Former** to integrate sequence and structure embeddings and introduce a **bi-scope pre-training** approach, including **CPRI** (Contrastive Protein-RNA Interaction Modeling) and **MIDM** (Mask Interface Distance Modeling), to enhance interaction understanding.
3. We curate **PRA310**, the largest protein-RNA binding affinity dataset, and evaluate our model’s performance across multiple datasets, achieving **state-of-the-art results**.



 

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





본 논문에서는 **CoPRA(Co-Former for Protein-RNA binding Affinity prediction)**라는 새로운 모델을 제안하여, 서로 다른 생물학적 도메인의 사전 학습된 언어 모델을 결합하여 단백질-RNA 결합 친화도를 예측한다. 이 모델의 주요 구성 요소는 **Co-Former 아키텍처**, **이중 시야 사전 학습 전략(bi-scope pre-training strategy)**, 그리고 **새롭게 구축된 데이터셋(PRA310, PRI30k)**으로 구성된다.

#### **1. CoPRA 모델 및 아키텍처**  
CoPRA는 단백질 언어 모델(Protein Language Model, PLM)과 RNA 언어 모델(RNA Language Model, RLM)을 결합하여 복합 수준의 단백질-RNA 결합 친화도를 예측하는 최초의 시도이다. 이를 위해, CoPRA는 다음과 같은 주요 모듈을 포함한다:

- **Co-Former**: 단백질과 RNA 서열 및 구조 정보를 통합하여 결합 친화도를 예측하는 경량화된 Transformer 기반 모델.  
- **인터페이스 시퀀스 임베딩**: 단백질 및 RNA 서열을 개별적으로 PLM과 RLM에 입력하고, 상호작용 인터페이스에서 중요한 정보를 선택하여 크로스 모달 학습에 활용.  
- **인터페이스 구조 임베딩**: 결합 인터페이스에서 구조 정보를 추출하여 Pair Embedding을 생성하고, Co-Former의 구조-서열 융합 모듈에 입력.  

Co-Former는 **구조-서열 융합 모듈(Structure-Sequence Fusion Module)**을 포함하여, 구조 및 서열 정보를 결합하는 **이중 경로 Transformer(Dual-path Transformer)** 아키텍처를 기반으로 한다.

#### **2. 이중 시야 사전 학습 전략 (Bi-Scope Pre-training Strategy)**  
CoPRA는 두 가지 사전 학습 기법을 적용하여 단백질-RNA 상호작용을 효과적으로 학습한다:

1. **CPRI(Contrastive Protein-RNA Interaction Modeling)**: 단백질과 RNA가 결합할 가능성을 예측하는 대조 학습(contrastive learning) 기반 모델링 기법.
2. **MIDM(Mask Interface Distance Modeling)**: 결합 인터페이스에서 원자 간 거리를 예측하는 정밀한 구조 예측 모델링 기법.

CPRI는 상호작용이 있는 단백질-RNA 쌍을 식별하는 반면, MIDM은 결합 인터페이스에서 세밀한 거리를 예측하여 CoPRA가 보다 정확한 결합 친화도를 학습하도록 한다.

#### **3. 데이터셋 구축 및 학습 과정**  
CoPRA는 대규모 데이터셋을 구축하고 이를 활용하여 학습한다:

- **PRI30k**: 사전 학습을 위한 데이터셋으로, BioLiP2에서 수집된 **5,909개의 단백질-RNA 복합체**를 기반으로 구성.
- **PRA310**: 결합 친화도 예측을 위한 **최대 규모의 데이터셋**, 기존의 PDBbind, PRBABv2, ProNAB 데이터셋을 통합하여 310개 복합체로 구성.
- **PRA201**: PRA310의 서브셋으로, 단백질과 RNA가 각각 하나의 체인으로 이루어진 더 엄격한 기준의 데이터셋.
- **mCSM 블라인드 테스트 세트**: 단일 아미노산 돌연변이 효과를 평가하는 독립적인 벤치마크 데이터셋.

모델 학습은 **NVIDIA A100-80G GPU 4개**를 사용하여 수행되었으며, 학습 과정에서 Adam 옵티마이저(learning rate 3e-5)를 적용하였다. Co-Former는 6개 블록으로 구성되며, 임베딩 차원은 320(시퀀스) 및 40(페어)로 설정되었다.

#### **4. 성능 평가 및 실험 결과**  
CoPRA는 **PRA310, PRA201, mCSM 블라인드 테스트 세트**에서 최신 성능을 달성하였다. 기존 모델과 비교하여, CoPRA는 결합 친화도 예측에서 **최고의 RMSE(1.391), MAE(1.129), PCC(0.580), SCC(0.589)** 값을 기록하였다. 특히, 기존 구조 기반 및 서열 기반 방법보다 뛰어난 결과를 보였으며, Co-Former의 구조 융합 모듈과 이중 시야 사전 학습 전략이 결합 친화도 예측에 효과적임을 입증하였다.

---


This paper proposes **CoPRA (Co-Former for Protein-RNA binding Affinity prediction)**, a novel approach that integrates pre-trained language models from different biological domains to predict protein-RNA binding affinity. The model consists of three key components: **Co-Former architecture**, **bi-scope pre-training strategy**, and **newly curated datasets (PRA310, PRI30k).**

#### **1. CoPRA Model and Architecture**  
CoPRA is the **first approach to integrate protein and RNA language models** for complex-level binding affinity prediction. The architecture consists of:

- **Co-Former**: A lightweight Transformer-based model that integrates sequence and structural information for binding affinity prediction.  
- **Interface Sequence Embedding**: Protein and RNA sequences are separately input into the PLM and RLM, and embeddings from the interaction interface are selected for cross-modal learning.  
- **Interface Structure Embedding**: Structural information at the binding interface is extracted to generate **pair embeddings**, which are fed into the **Co-Former’s structure-sequence fusion module.**  

Co-Former employs a **dual-path Transformer architecture** with a **structure-sequence fusion module** to combine sequence and structural representations effectively.

#### **2. Bi-Scope Pre-Training Strategy**  
CoPRA adopts **two pre-training strategies** to effectively model protein-RNA interactions:

1. **CPRI (Contrastive Protein-RNA Interaction Modeling)**: A contrastive learning-based strategy that predicts the likelihood of protein-RNA binding.  
2. **MIDM (Mask Interface Distance Modeling)**: A fine-grained structural prediction method that estimates atomic-level distances at the binding interface.

CPRI identifies interacting protein-RNA pairs, whereas MIDM refines the binding affinity predictions by accurately modeling the interface’s structural details.

#### **3. Dataset Curation and Training Process**  
CoPRA leverages **large-scale datasets** for training:

- **PRI30k**: A pre-training dataset curated from BioLiP2, containing **5,909 protein-RNA complexes** and 150k protein-RNA pairs.
- **PRA310**: The **largest protein-RNA binding affinity dataset**, created by integrating samples from **PDBbind, PRBABv2, and ProNAB** (310 complexes).
- **PRA201**: A subset of PRA310 with stricter constraints (one protein chain and one RNA chain per complex).
- **mCSM Blind Test Set**: A benchmark dataset for **single-point mutation effect prediction** in protein-RNA complexes.

Training was conducted on **four NVIDIA A100-80G GPUs** with the **Adam optimizer (learning rate 3e-5).** The Co-Former model consists of **six blocks**, with **sequence embedding size = 320** and **pair embedding size = 40.**

#### **4. Performance Evaluation and Results**  
CoPRA achieves **state-of-the-art performance** on **PRA310, PRA201, and the mCSM blind test set**. Compared to existing methods, CoPRA achieves the **best RMSE (1.391), MAE (1.129), PCC (0.580), and SCC (0.589)** values in binding affinity prediction. The results demonstrate that CoPRA outperforms both sequence-based and structure-based approaches, validating the effectiveness of its **structure-sequence fusion module** and **bi-scope pre-training strategy.**
   




   
 
<br/>
# Results  




 

본 연구에서는 **CoPRA** 모델의 성능을 기존의 경쟁 모델들과 비교하고, 다양한 데이터셋에서 평가하여 그 우수성을 입증하였다. 실험에서는 **결합 친화도 예측(Binding Affinity Prediction)**과 **단백질 돌연변이 효과 예측(Mutation Effect Prediction)**을 수행하며, 평가에 사용된 데이터셋과 메트릭은 다음과 같다.  

---

### **1. 경쟁 모델 (Baseline Models)**  
CoPRA의 성능을 비교하기 위해, 기존 연구에서 사용된 **서열 기반(Sequence-based) 모델**과 **구조 기반(Structure-based) 모델**을 포함한 다양한 경쟁 모델과 비교하였다.  

#### **(1) 서열 기반 모델 (Sequence-based Models)**
- **LM+LR**: 사전 학습된 언어 모델(ESM-2, RiNALMo)의 출력을 로지스틱 회귀(Logistic Regression)로 학습.
- **LM+RF**: 랜덤 포레스트(Random Forest)를 사용한 모델.
- **LM+MLP**: 다층 퍼셉트론(MLP)을 활용하여 결합 친화도를 예측.
- **LM+SVR**: 서포트 벡터 회귀(SVR)를 적용한 모델.
- **LM+Transformer**: 트랜스포머 기반 모델로, 언어 모델 임베딩을 활용하여 예측.
- **DeepNAP**: 1D CNN을 활용하여 단백질-RNA 서열 정보를 처리하는 기존의 딥러닝 모델.  

#### **(2) 구조 기반 모델 (Structure-based Models)**
- **PredPRBA**: 결합 인터페이스의 구조적 특징을 활용하는 기계 학습 모델.
- **FoldX**: 결합 자유 에너지(ΔG)를 예측하는 물리 기반 모델.
- **GCN, GAT, EGNN, GVP**: 그래프 신경망(GNN) 기반의 모델로, 단백질-RNA 구조 정보를 그래프로 표현하여 학습.
- **IPA (Invariant Point Attention)**: AlphaFold2에서 사용된 구조 정보 학습 기법을 적용한 모델.
- **LM+IPA**: IPA 모델에 사전 학습된 언어 모델 임베딩을 추가하여 성능을 향상.

---

### **2. 테스트 데이터 및 실험 설정**  
#### **(1) 평가 데이터셋**
- **PRA310**: 310개의 단백질-RNA 복합체를 포함한 가장 큰 결합 친화도 예측 데이터셋.
- **PRA201**: PRA310의 서브셋으로, 단백질과 RNA가 각각 하나의 체인만 포함하는 데이터셋.
- **mCSM 블라인드 테스트 세트**: 단백질 돌연변이 효과 예측을 위한 79개의 변이 데이터셋.

#### **(2) 평가 메트릭**
모델 성능 평가는 기존 연구에서 일반적으로 사용되는 다음 네 가지 메트릭을 기준으로 하였다:
1. **RMSE (Root Mean Squared Error)**: 낮을수록 좋은 성능.
2. **MAE (Mean Absolute Error)**: 낮을수록 좋은 성능.
3. **PCC (Pearson Correlation Coefficient)**: 높을수록 좋은 성능.
4. **SCC (Spearman Correlation Coefficient)**: 높을수록 좋은 성능.

---

### **3. 실험 결과 및 성능 분석**  
#### **(1) 결합 친화도 예측 (Binding Affinity Prediction)**
- **CoPRA는 PRA310 및 PRA201에서 최고의 성능을 기록함.**
- **PRA310에서 RMSE 1.391, MAE 1.129, PCC 0.580, SCC 0.589**를 기록하여, 기존의 모든 모델을 능가함.
- 기존의 서열 기반 및 구조 기반 모델보다 **5~20% 더 높은 예측 정확도**를 보임.

#### **(2) 단백질 돌연변이 효과 예측 (Mutation Effect Prediction)**
- CoPRA는 mCSM 블라인드 테스트 세트에서 **RMSE 0.957, MAE 0.833, PCC 0.550, SCC 0.570**을 기록함.
- 기존의 FoldX(물리 기반 모델), mCSM(머신러닝 모델)보다 더 높은 상관 계수를 보이며, 단백질 변이 효과 예측에서도 높은 성능을 입증함.

---

### **4. 분석 및 결론**
- **CoPRA는 구조와 서열 정보를 함께 활용하는 하이브리드 모델로, 기존 서열 또는 구조 단독 기반 모델보다 뛰어난 성능을 보임.**
- **사전 학습(Bi-Scope Pretraining)을 거친 CoPRA는 기본 모델 대비 성능이 10% 이상 향상됨.**
- **PRA310 데이터셋이 더욱 복잡한 문제를 포함하고 있으며, 기존 데이터셋(PRA201)보다 더 높은 난이도를 가짐.**
- **CoPRA의 구조-서열 융합(Structure-Sequence Fusion) 기법이 결합 친화도 예측의 핵심적인 성능 향상 요소로 작용함.**

---

---



This study evaluates **CoPRA** by comparing it with existing **sequence-based** and **structure-based** baseline models across multiple datasets, demonstrating its superior performance.

---

### **1. Baseline Models**  
To validate CoPRA’s performance, we compared it against multiple **sequence-based** and **structure-based** methods.

#### **(1) Sequence-Based Models**
- **LM+LR**: Logistic Regression using embeddings from pre-trained language models (ESM-2, RiNALMo).
- **LM+RF**: Random Forest-based model.
- **LM+MLP**: Multi-layer perceptron (MLP) for binding affinity prediction.
- **LM+SVR**: Support Vector Regression (SVR) model.
- **LM+Transformer**: Transformer-based model using pre-trained language model embeddings.
- **DeepNAP**: A 1D CNN model processing protein-RNA sequences.

#### **(2) Structure-Based Models**
- **PredPRBA**: A machine learning model leveraging interface structural features.
- **FoldX**: A physics-based model predicting binding free energy (ΔG).
- **GCN, GAT, EGNN, GVP**: Graph-based models processing protein-RNA structure.
- **IPA (Invariant Point Attention)**: A structural learning method derived from AlphaFold2.
- **LM+IPA**: IPA combined with pre-trained language model embeddings.

---

### **2. Test Data and Experiment Setup**  
#### **(1) Evaluation Datasets**
- **PRA310**: The largest binding affinity dataset containing 310 protein-RNA complexes.
- **PRA201**: A subset of PRA310 with stricter chain constraints (one protein, one RNA per complex).
- **mCSM Blind Test Set**: A benchmark dataset with **79 single-point mutations** for mutation effect prediction.

#### **(2) Evaluation Metrics**
Four widely used metrics were employed for performance assessment:
1. **RMSE (Root Mean Squared Error)** – Lower is better.
2. **MAE (Mean Absolute Error)** – Lower is better.
3. **PCC (Pearson Correlation Coefficient)** – Higher is better.
4. **SCC (Spearman Correlation Coefficient)** – Higher is better.

---

### **3. Experimental Results and Performance Analysis**  
#### **(1) Binding Affinity Prediction**
- **CoPRA achieved the best performance on PRA310 and PRA201.**
- **PRA310 results: RMSE 1.391, MAE 1.129, PCC 0.580, SCC 0.589.**
- Outperformed all prior methods, achieving **5-20% higher accuracy than sequence-based and structure-based models.**

#### **(2) Mutation Effect Prediction**
- CoPRA’s **mCSM blind test set results: RMSE 0.957, MAE 0.833, PCC 0.550, SCC 0.570.**
- Outperformed **FoldX (physics-based) and mCSM (ML-based) models, demonstrating superior correlation scores.**

---

### **4. Analysis and Conclusion**
- **CoPRA’s hybrid approach integrating sequence and structure significantly outperforms previous models.**
- **Bi-scope pre-training improves CoPRA’s accuracy by over 10%.**
- **PRA310 is a more challenging dataset than PRA201, demonstrating CoPRA’s robustness.**
- **Structure-Sequence Fusion in CoPRA is a key factor in enhancing binding affinity prediction.**




<br/>
# 예제  





본 연구에서는 **CoPRA**의 성능을 평가하기 위해 다양한 **테스트 데이터, 트레이닝 데이터, 및 실제 예제(task input/output)**를 구성하였다. 이를 통해 모델이 단백질-RNA 결합 친화도를 얼마나 정확하게 예측하는지, 그리고 돌연변이 효과를 얼마나 정밀하게 분석하는지 검증하였다.  

---

## **1. 트레이닝 데이터 (Training Data)**
CoPRA는 대규모의 **사전 학습 데이터**와 **결합 친화도 예측을 위한 데이터**로 학습되었다.

### **(1) 사전 학습 데이터 (Pre-training Data)**
- **PRI30k**:  
  - BioLiP2에서 수집된 **5,909개의 단백질-RNA 복합체**  
  - 각 복합체에서 단백질-RNA 상호작용 인터페이스를 정의하여 **총 150,000개의 단백질-RNA 쌍**을 구성  
  - CoPRA의 **Contrastive Learning(CPRI)** 및 **Interface Distance Modeling(MIDM)** 사전 학습에 활용  

### **(2) 결합 친화도 예측 데이터 (Binding Affinity Prediction Data)**
- **PRA310**:  
  - **310개의 단백질-RNA 복합체**로 구성된 결합 친화도 데이터셋  
  - 기존 PDBbind, PRBABv2, ProNAB 데이터셋을 통합하여 구축  
  - 단백질과 RNA 간의 결합 친화도를 나타내는 **실제 실험 데이터(ΔG 값, kcal/mol 단위)** 포함  
- **PRA201**:  
  - PRA310의 서브셋으로, 단일 단백질 및 RNA 체인만 포함  
  - 결합 복잡도를 줄인 대신 더 정밀한 평가 가능  

### **(3) 단백질 돌연변이 효과 예측 데이터 (Mutation Effect Prediction Data)**
- **mCSM 블라인드 테스트 세트**:  
  - **79개의 단백질-RNA 복합체 변이 데이터** 포함  
  - 단백질의 단일 아미노산 돌연변이가 결합 친화도(ΔΔG)에 미치는 영향을 평가  

---

## **2. 테스크 인풋/아웃풋 (Task Input/Output)**
CoPRA는 두 가지 주요 테스크를 수행한다:  
1) **단백질-RNA 결합 친화도 예측 (Binding Affinity Prediction)**  
2) **단백질 돌연변이 효과 예측 (Mutation Effect Prediction)**  

### **(1) 단백질-RNA 결합 친화도 예측**
- **입력(Input)**:
  - 단백질 서열(Protein Sequence)  
  - RNA 서열(RNA Sequence)  
  - 단백질-RNA 결합 인터페이스의 3D 구조 정보(Protein-RNA Complex Structure)  

- **출력(Output)**:
  - 결합 자유 에너지 (Binding Free Energy, ΔG)  
  - kcal/mol 단위의 값으로 낮을수록 결합이 강한 구조 의미  
  - 예: **ΔG = -8.32 kcal/mol** (낮을수록 강한 결합)  

#### ** 예제 (Example)**
```python
input_protein = "MTSAGKQSR..."
input_rna = "GGGCUAACGG..."
predicted_binding_affinity = model.predict(input_protein, input_rna)
print(predicted_binding_affinity)  
# Output: -8.32 kcal/mol
```

---

### **(2) 단백질 돌연변이 효과 예측**
- **입력(Input)**:
  - 변이 전 단백질 서열(Wild-type Protein Sequence)  
  - 변이 후 단백질 서열(Mutant Protein Sequence)  
  - RNA 서열(RNA Sequence)  
  - 단백질-RNA 결합 인터페이스의 3D 구조 정보  

- **출력(Output)**:
  - 돌연변이로 인한 결합 자유 에너지 변화(ΔΔG = ΔG_mut - ΔG_wild)  
  - ΔΔG > 0 → 결합 친화도 감소 (돌연변이로 인해 결합이 약해짐)  
  - ΔΔG < 0 → 결합 친화도 증가 (돌연변이로 인해 결합이 강해짐)  

#### ** 예제 (Example)**
```python
wild_type = "MTSAGKQSR..."
mutant_type = "MTSAGKHSR..."  # 특정 아미노산 돌연변이
input_rna = "GGGCUAACGG..."
predicted_mutation_effect = model.predict_mutation_effect(wild_type, mutant_type, input_rna)
print(predicted_mutation_effect)  
# Output: 1.25 kcal/mol (결합 친화도 감소)
```

---

## **3. 평가 및 테스트 과정**
- **5-Fold Cross Validation**을 활용하여 PRA310에서 모델을 검증  
- **독립적인 mCSM 블라인드 테스트 세트**에서 돌연변이 예측 성능 검증  
- 모든 실험은 **NVIDIA A100-80G GPU 4개**에서 실행  

---

---



This study evaluates **CoPRA** using a variety of **test datasets, training datasets, and real examples (task input/output).** The goal is to validate the model’s ability to accurately predict protein-RNA binding affinity and analyze mutation effects.

---

## **1. Training Data**
CoPRA is trained on both **pre-training datasets** and **binding affinity datasets.**

### **(1) Pre-training Data**
- **PRI30k**:  
  - **5,909 protein-RNA complexes** collected from BioLiP2  
  - Defined protein-RNA interaction interfaces, forming **150,000 protein-RNA pairs**  
  - Used for **contrastive learning (CPRI)** and **interface distance modeling (MIDM)** in pre-training  

### **(2) Binding Affinity Prediction Data**
- **PRA310**:  
  - **310 protein-RNA complexes** curated from **PDBbind, PRBABv2, and ProNAB**  
  - Includes experimental **binding free energy (ΔG) values in kcal/mol**  
- **PRA201**:  
  - A subset of PRA310 with **one protein and one RNA chain per complex**  
  - Allows for a more controlled evaluation  

### **(3) Mutation Effect Prediction Data**
- **mCSM Blind Test Set**:  
  - **79 mutation samples from 14 protein-RNA complexes**  
  - Evaluates **the impact of single amino acid mutations** on binding affinity  

---

## **2. Task Input/Output**
CoPRA performs two primary tasks:  
1) **Protein-RNA Binding Affinity Prediction**  
2) **Mutation Effect Prediction**  

### **(1) Protein-RNA Binding Affinity Prediction**
- **Input:**
  - Protein sequence  
  - RNA sequence  
  - Protein-RNA complex structure  

- **Output:**
  - **Binding free energy (ΔG) in kcal/mol**  
  - Lower ΔG indicates stronger binding  
  - Example: **ΔG = -8.32 kcal/mol**  

#### ** Example**
```python
input_protein = "MTSAGKQSR..."
input_rna = "GGGCUAACGG..."
predicted_binding_affinity = model.predict(input_protein, input_rna)
print(predicted_binding_affinity)  
# Output: -8.32 kcal/mol
```

---

### **(2) Mutation Effect Prediction**
- **Input:**
  - Wild-type protein sequence  
  - Mutant protein sequence  
  - RNA sequence  
  - Protein-RNA complex structure  

- **Output:**
  - **Binding affinity change (ΔΔG = ΔG_mut - ΔG_wild)**  
  - ΔΔG > 0 → Binding affinity decreases  
  - ΔΔG < 0 → Binding affinity increases  

#### ** Example**
```python
wild_type = "MTSAGKQSR..."
mutant_type = "MTSAGKHSR..."  
input_rna = "GGGCUAACGG..."
predicted_mutation_effect = model.predict_mutation_effect(wild_type, mutant_type, input_rna)
print(predicted_mutation_effect)  
# Output: 1.25 kcal/mol (weakened binding)
```

---

## **3. Evaluation and Testing**
- **5-Fold Cross Validation** performed on PRA310  
- **Independent mCSM blind test set** used for mutation effect evaluation  
- All experiments were conducted on **4 NVIDIA A100-80G GPUs**  





<br/>  
# 요약   




본 연구에서는 단백질-RNA 결합 친화도를 예측하기 위해 **CoPRA** 모델을 제안하며, 사전 학습된 단백질 및 RNA 언어 모델을 결합하여 구조-서열 정보를 통합하는 **Co-Former**를 도입하였다. CoPRA는 PRA310, PRA201, mCSM 블라인드 테스트 세트에서 기존 모델보다 높은 성능을 기록하였으며, 특히 결합 자유 에너지(ΔG) 및 돌연변이 효과(ΔΔG) 예측에서 정확도를 향상시켰다. 예제 실험을 통해 CoPRA는 단백질 및 RNA 서열과 구조 정보를 입력으로 받아 결합 친화도를 예측하고, 돌연변이에 따른 결합 변화도 정밀하게 분석할 수 있음을 입증하였다.  

---


This study proposes **CoPRA**, a model for protein-RNA binding affinity prediction, integrating pre-trained protein and RNA language models through **Co-Former**, which fuses structural and sequence information. CoPRA outperforms existing models on PRA310, PRA201, and the mCSM blind test set, significantly improving accuracy in predicting binding free energy (ΔG) and mutation effects (ΔΔG). Through example experiments, CoPRA demonstrates its ability to take protein/RNA sequences and structural data as input, accurately predicting binding affinity and analyzing mutation-induced changes.


<br/>  
# 기타  





논문에는 **CoPRA 모델의 구조, 실험 결과, 성능 비교** 등을 시각적으로 보여주는 다양한 **피규어(Figure)와 테이블(Table)**이 포함되어 있다. 주요 그림과 표를 설명하면 다음과 같다.  

---

### **1. CoPRA 모델 구조 (Figure 1 & 2)**  
- **Figure 1: CoPRA 개요**  
  - 단백질 언어 모델(PLM)과 RNA 언어 모델(RLM)을 활용하여 **결합 친화도 예측을 위한 Co-Former**를 구성하는 과정이 시각적으로 표현됨.  
  - 주요 구성 요소: **단백질 및 RNA 언어 모델**, **구조 정보 추출**, **Co-Former**, **인터페이스 노드 및 결합 자유 에너지(ΔG) 예측**  
  - CPRI(Contrastive Protein-RNA Interaction)와 MIDM(Mask Interface Distance Modeling)이라는 사전 학습 기법을 사용하여, 단백질-RNA 결합 정보를 학습하는 구조를 설명함.  

- **Figure 2: CoPRA의 작동 방식**  
  - 단백질-RNA 복합체를 입력으로 받아, **서열 정보는 PLM과 RLM으로 처리**하고, **구조 정보는 인터페이스에서 추출**하여 **Co-Former**로 결합하는 과정을 나타냄.  
  - **Co-Former의 내부 구조**를 보여주며, **1D 임베딩, 쌍(pair) 임베딩, 구조-서열 융합 모듈** 등을 포함함.  
  - CoPRA가 **결합 친화도를 예측하는 단계별 과정**을 설명하는 핵심 그림.  

---

### **2. 실험 결과 및 모델 성능 비교 (Table 1 & 2 & 3)**  

- **Table 1: PRA310 및 PRA201 데이터셋에서의 결합 친화도 예측 성능 비교**  
  - RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), PCC (Pearson Correlation Coefficient), SCC (Spearman Correlation Coefficient)를 비교.  
  - CoPRA가 모든 지표에서 **기존 모델보다 높은 성능을 기록**하였으며, 특히 **PCC와 SCC에서 0.58, 0.59로 가장 높은 상관 관계를 보임**.  
  - 서열 기반 모델(DeepNAP, LM+Transformer)과 구조 기반 모델(IPA, GVP)보다 우수한 결과를 보이며, **구조-서열 통합 방식(Co-Former)의 효과**를 입증.  

- **Table 2: mCSM 블라인드 테스트 세트에서 돌연변이 효과 예측 성능 비교**  
  - CoPRA는 돌연변이 후 결합 자유 에너지 변화(ΔΔG) 예측에서도 **기존의 FoldX, mCSM 모델을 뛰어넘음**.  
  - 특히 **RMSE 0.957, MAE 0.833, PCC 0.550, SCC 0.570**으로, 기존 모델 대비 높은 정확도를 보임.  
  - 이는 **CoPRA의 사전 학습 기법(Bi-Scope Pretraining)이 돌연변이 효과 예측에도 강력한 성능을 제공함**을 의미함.  

- **Table 3: CoPRA의 주요 구성 요소에 대한 ablation study 결과**  
  - 모델의 각 요소(사전 학습, 구조 정보, 인터페이스 특성, Co-Former 등)를 제거했을 때 성능 변화를 비교.  
  - **사전 학습을 제거하면 성능이 가장 크게 감소**(PCC 0.580 → 0.522), 이는 **CPRI와 MIDM 사전 학습의 중요성을 강조**함.  
  - **Co-Former가 없는 경우 성능이 가장 낮아지며**, 이는 CoPRA의 핵심 구성 요소가 서열과 구조 정보를 효과적으로 융합하는 것임을 입증.  

---

### **3. 모델 크기 변화에 따른 성능 영향 (Figure 3)**  
- CoPRA의 **ESM-2 모델 크기**에 따른 성능 변화를 분석한 그림.  
- **모델 크기가 증가할수록 예측 정확도가 향상됨**을 보여줌.  
- 650M 모델에서 가장 높은 성능을 기록하였으며, 이는 **사전 학습된 대규모 모델이 단백질-RNA 결합 친화도 예측에서 효과적임을 시사**.  

---

### **결론**  
논문의 피규어와 테이블들은 **CoPRA의 혁신적인 구조(서열-구조 융합), 강력한 성능(최신 모델 대비 우수), 그리고 사전 학습 기법의 중요성**을 강조하고 있다. 특히, Co-Former와 Bi-Scope Pretraining이 결합 친화도 및 돌연변이 효과 예측에서 중요한 역할을 한다는 점이 실험을 통해 입증되었다.  

---

---



The paper includes various **figures (Figures) and tables (Tables)** that visually represent **CoPRA’s architecture, experimental results, and performance comparisons.** Below is a summary of the key figures and tables.  

---

### **1. CoPRA Model Architecture (Figure 1 & 2)**  

- **Figure 1: Overview of CoPRA**  
  - Illustrates the **integration of protein and RNA language models (PLM, RLM)** with the **Co-Former** to predict protein-RNA binding affinity.  
  - Key components: **Protein & RNA Language Models, Structure Extraction, Co-Former, Interface Nodes, and Binding Free Energy (ΔG) Prediction.**  
  - The model utilizes **contrastive learning (CPRI) and masked interface distance modeling (MIDM) as pre-training strategies.**  

- **Figure 2: CoPRA’s Functional Process**  
  - Shows the **input of protein-RNA complexes**, where **sequence data is processed by PLM and RLM**, while **structural information is extracted from the interaction interface** before being fused in **Co-Former**.  
  - Depicts **Co-Former’s internal structure**, including **1D embedding, pair embedding, and structure-sequence fusion module**.  
  - Provides a step-by-step explanation of how CoPRA **predicts binding affinity using fused sequence and structural information**.  

---

### **2. Experimental Results and Performance Comparison (Table 1 & 2 & 3)**  

- **Table 1: Binding Affinity Prediction Performance on PRA310 & PRA201**  
  - Compares RMSE, MAE, PCC, and SCC across multiple models.  
  - **CoPRA achieves the highest performance across all metrics**, particularly with **PCC = 0.580 and SCC = 0.589**, surpassing both sequence-based and structure-based models.  
  - Demonstrates the **effectiveness of Co-Former in integrating sequence and structure information.**  

- **Table 2: Mutation Effect Prediction Performance on the mCSM Blind Test Set**  
  - CoPRA outperforms **FoldX and mCSM models** in **predicting mutation effects on binding affinity (ΔΔG)**.  
  - Notably, **RMSE = 0.957, MAE = 0.833, PCC = 0.550, SCC = 0.570**, showing higher accuracy than existing methods.  
  - Highlights the **importance of Bi-Scope Pretraining in mutation effect prediction.**  

- **Table 3: Ablation Study on CoPRA’s Key Components**  
  - Examines performance changes when removing key components like **pre-training, structural features, interface information, and Co-Former**.  
  - **Removing pre-training leads to the largest performance drop (PCC from 0.580 to 0.522), emphasizing the importance of CPRI and MIDM.**  
  - **Without Co-Former, performance significantly degrades**, proving that **structure-sequence fusion is crucial for accurate predictions.**  

---

### **3. Model Size Impact on Performance (Figure 3)**  
- Analyzes how **increasing the size of the ESM-2 model** impacts performance.  
- **Larger models improve predictive accuracy**, with the **650M model achieving the best results.**  
- This supports the idea that **pre-trained large-scale models are highly effective for protein-RNA binding affinity prediction.**  

---

### **Conclusion**  
The figures and tables in the paper highlight **CoPRA’s innovative architecture (sequence-structure fusion), superior performance (outperforming state-of-the-art models), and the importance of Bi-Scope Pretraining.** Particularly, the Co-Former module and large-scale pre-training significantly improve binding affinity and mutation effect prediction accuracy.



<br/>
# refer format:     


@article{Han2025CoPRA,
  author    = {Rong Han and Xiaohong Liu and Tong Pan and Jing Xu and Xiaoyu Wang and Wuyang Lan and Zhenyu Li and Zixuan Wang and Jiangning Song and Guangyu Wang and Ting Chen},
  title     = {CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  volume    = {39},
  number    = {3},
  url       = {https://arxiv.org/abs/2409.03773},
  note      = {Extended version available at \url{https://arxiv.org/abs/2409.03773}}
}




Han, Rong, Xiaohong Liu, Tong Pan, Jing Xu, Xiaoyu Wang, Wuyang Lan, Zhenyu Li, Zixuan Wang, Jiangning Song, Guangyu Wang, and Ting Chen. "CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction." Proceedings of the AAAI Conference on Artificial Intelligence 39, no. 3 (2025)  





