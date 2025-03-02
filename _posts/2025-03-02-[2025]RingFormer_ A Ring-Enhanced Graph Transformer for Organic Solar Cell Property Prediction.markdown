---
layout: post
title:  "[2025]RingFormer: A Ring-Enhanced Graph Transformer for Organic Solar Cell Property Prediction"  
date:   2025-03-02 13:18:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


그래프네트워크랑 트랜스포머를 합친 링포머제안, 유기 태양 전지의 분자 시스템을 잘 반영토록함..  
그래프와 언어모델의 합친 접근방법이 인상적인..!  



짧은 요약(Abstract) :    



유기 태양 전지(Organic Solar Cells, OSCs)는 지속 가능한 에너지원으로 주목받고 있지만, 원하는 특성을 가진 분자를 찾는 실험적 연구는 많은 시간과 비용이 소요된다. 이를 해결하기 위해 머신 러닝 모델을 활용하여 OSC 분자의 특성을 정확히 예측하는 것이 중요하다. 기존의 그래프 학습 방법은 분자 특성 예측에서 성과를 보였지만, OSC 분자의 독특한 구조적 특성을 충분히 반영하지 못했다. 특히, OSC 성능에 큰 영향을 미치는 고리(ring) 시스템을 제대로 고려하지 않아 최적의 성능을 내지 못하는 문제가 있다.

이 연구에서는 **RingFormer**라는 새로운 그래프 트랜스포머 프레임워크를 제안한다. 이 모델은 원자 수준과 고리 수준의 구조적 패턴을 모두 포착하도록 설계되었다. RingFormer는 원자 및 고리 구조를 통합한 계층적 그래프를 구성하고, 지역 메시지 전달(Local Message Passing)과 글로벌 어텐션(Global Attention) 메커니즘을 결합하여 보다 표현력 있는 그래프 표현을 생성한다. 다섯 개의 OSC 분자 데이터셋을 활용한 실험에서 RingFormer는 기존 모델을 뛰어넘는 성능을 보여주었으며, 특히 CEPDB 데이터셋에서 가장 가까운 경쟁 모델보다 **22.77% 향상된 예측 성능**을 기록했다.

---


Organic Solar Cells (OSCs) are a promising technology for sustainable energy production. However, the identification of molecules with desired OSC properties typically involves laborious experimental research. To accelerate progress in the field, it is crucial to develop machine learning models capable of accurately predicting the properties of OSC molecules. While graph representation learning has demonstrated success in molecular property prediction, it remains under-explored for OSC-specific tasks. Existing methods fail to capture the unique structural features of OSC molecules, particularly the intricate ring systems that critically influence OSC properties, leading to suboptimal performance.

To fill the gap, we present **RingFormer**, a novel graph transformer framework specially designed to capture both atom and ring-level structural patterns in OSC molecules. RingFormer constructs a hierarchical graph that integrates atomic and ring structures and employs a combination of local message-passing and global attention mechanisms to generate expressive graph representations for accurate OSC property prediction. We evaluate RingFormer’s effectiveness on five curated OSC molecule datasets through extensive experiments. The results demonstrate that **RingFormer consistently outperforms existing methods, achieving a 22.77% relative improvement over the nearest competitor on the CEPDB dataset.**



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




#### **1. RingFormer 모델 개요**  
RingFormer는 유기 태양 전지(Organic Solar Cells, OSCs) 분자의 특성을 보다 정확하게 예측하기 위해 개발된 **그래프 트랜스포머(Graph Transformer)** 기반 모델이다. 이 모델은 원자 수준과 고리(ring) 수준의 구조적 패턴을 모두 반영하는 계층적 그래프(Hierarchical Graph)를 구축하고, 이를 학습하여 더욱 정교한 분자 표현을 생성한다.  

#### **2. 계층적 OSC 그래프 (Hierarchical OSC Graph)**
RingFormer는 기존의 그래프 신경망(Graph Neural Networks, GNNs) 모델이 OSC 분자의 복잡한 고리 시스템을 적절히 반영하지 못한다는 문제를 해결하기 위해, **세 가지 수준(Level)의 계층적 그래프**를 구축한다.
- **원자-수준 그래프 (Atom-Level Graph, \(G_A\))**: 분자의 원자(atom) 간의 결합 관계를 표현한다.
- **고리-수준 그래프 (Ring-Level Graph, \(G_R\))**: 분자의 주요 고리(ring) 구조와 그 연결성을 표현한다.
- **원자-고리 상호작용 그래프 (Inter-Level Graph, \(G_I\))**: 원자와 고리 간의 관계를 나타내며, 두 수준 간 정보 전달을 돕는다.  

이 세 가지 그래프를 통합하여 **RingFormer의 입력 계층적 그래프 \(G = \{G_A, G_R, G_I\}\) 를 구성**한다. 이를 통해 기존 모델들이 놓쳤던 **고리 간의 상호작용 정보**까지 학습할 수 있다.

#### **3. RingFormer 아키텍처**
RingFormer는 **메시지 전달(message passing)**과 **글로벌 어텐션(global attention)**을 결합하여 구조적 정보를 효과적으로 학습한다.
1. **원자-수준 메시지 전달 (Atom-Level Message Passing)**  
   - GNN을 활용하여 원자 간 결합 및 기능성 그룹(functional group)과 같은 지역적 화학 구조 정보를 학습한다.
   
2. **고리-수준 크로스 어텐션 (Ring-Level Cross-Attention)**  
   - 트랜스포머 기반의 **크로스 어텐션(cross-attention) 메커니즘**을 도입하여 고리 간의 관계를 학습한다.  
   - 일반적인 트랜스포머는 단순한 self-attention을 수행하지만, RingFormer는 고리 간 연결 정보를 더욱 효과적으로 반영하기 위해 **엣지 속성(edge attributes)을 포함한 크로스 어텐션**을 활용한다.  

3. **원자-고리 상호작용 메시지 전달 (Inter-Level Message Passing)**  
   - 원자와 고리 간의 관계를 모델링하여, 원자 표현이 전체 고리 시스템 구조를 인식할 수 있도록 돕는다.

4. **계층적 정보 융합 (Hierarchical Fusion)**  
   - 원자 수준과 고리 수준에서 학습된 정보를 종합하여 더욱 정교한 분자 표현을 생성한다.
   - 최종적으로, 원자와 고리 노드의 정보를 모두 포함하는 **그래프 표현(graph representation)을 생성**하여 OSC 분자의 특성을 예측한다.

#### **4. 훈련 데이터 및 실험 설정**
- **데이터셋**: 다섯 개의 OSC 분자 데이터셋에서 RingFormer를 평가했다.
  - **CEPDB**: 밀도 범함수 이론(DFT) 기반으로 생성된 230만 개 이상의 OSC 분자를 포함.
  - **HOPV**: 실험적으로 검증된 350개의 OSC 분자를 포함.
  - **PFD, NFA, PD**: 다양한 종류의 OSC 분자를 포함한 실험적 데이터셋.  
- **평가 지표**: **Mean Absolute Error (MAE)** 를 사용하여 예측 성능을 비교했다.
- **베이스라인 모델**:  
  - **기존 GNN 기반 모델**: GINE, AttentiveFP, O-GNN 등.  
  - **기존 트랜스포머 기반 모델**: Graphormer, GPS, GraphViT 등.  
  - **기존 분자 지문(Fingerprint) 기반 모델**: MACCS, ECFP 등.  

실험 결과, RingFormer는 **CEPDB 데이터셋에서 기존 최고 모델 대비 22.77% 더 정확한 예측 성능**을 보였다. 또한, 다른 OSC 데이터셋에서도 일관되게 높은 성능을 기록했다.

---



#### **1. RingFormer Model Overview**  
RingFormer is a **graph transformer-based model** designed to improve the prediction accuracy of organic solar cell (OSC) molecular properties. The model captures both atomic and ring-level structural patterns, utilizing a hierarchical graph representation to enhance molecular property predictions.

#### **2. Hierarchical OSC Graph Construction**  
To address the limitations of traditional graph neural networks (GNNs) in capturing intricate **ring systems** in OSC molecules, **RingFormer constructs a three-level hierarchical graph \(G = \{G_A, G_R, G_I\}\)**:
- **Atom-Level Graph (\(G_A\))**: Represents atomic bonding structures.
- **Ring-Level Graph (\(G_R\))**: Explicitly captures rings and their interconnections.
- **Inter-Level Graph (\(G_I\))**: Models interactions between atoms and rings, bridging local and global structural representations.  

By integrating these three levels, RingFormer **accurately models both atomic and ring-level structures**, which are crucial for predicting OSC properties.

#### **3. RingFormer Architecture**
RingFormer employs **a combination of message passing and global attention** to learn expressive graph representations:
1. **Atom-Level Message Passing**  
   - GNNs are applied to encode local chemical structures, including functional groups.

2. **Ring-Level Cross-Attention**  
   - A novel **cross-attention mechanism** is introduced to capture global ring system patterns.  
   - Unlike standard transformers that use simple self-attention, RingFormer **integrates edge attributes into the attention mechanism**, improving its ability to model ring connectivity.

3. **Inter-Level Message Passing**  
   - Bridges atomic and ring-level representations, allowing atomic nodes to perceive the global ring system structure.

4. **Hierarchical Fusion Strategy**  
   - Aggregates learned information from different hierarchical levels to construct a **comprehensive molecular representation**, ultimately predicting OSC properties with higher accuracy.

#### **4. Training Data & Experimental Setup**
- **Datasets**: Five OSC molecular datasets were used to evaluate RingFormer:
  - **CEPDB**: A large-scale dataset with over 2.3M OSC molecules, generated using density functional theory (DFT).
  - **HOPV**: Experimentally validated dataset with 350 OSC molecules.
  - **PFD, NFA, PD**: Datasets containing various OSC molecules with experimental validation.  
- **Evaluation Metric**: Mean Absolute Error (MAE) was used to assess predictive performance.
- **Baselines**:  
  - **GNN-based models**: GINE, AttentiveFP, O-GNN.  
  - **Transformer-based models**: Graphormer, GPS, GraphViT.  
  - **Fingerprint-based models**: MACCS, ECFP.  

The experimental results show that **RingFormer outperformed existing methods across all datasets, achieving a remarkable 22.77% relative improvement on CEPDB compared to the best-performing baseline.**



   
 
<br/>
# Results  




#### **1. 경쟁 모델 (Baseline Models)**  
RingFormer의 성능을 평가하기 위해 **11개의 기존 모델**과 비교하였다.  
이들은 크게 **네 가지 카테고리**로 나뉜다:

1. **지문(Fingerprint) 기반 방법**  
   - **MACCS**: 분자의 구조적 특징을 166비트 길이의 지문으로 표현하는 방법.  
   - **ECFP (Extended Connectivity Fingerprint)**: 보다 확장된 분자 지문을 제공하는 방법.  

2. **GNN 기반 방법**  
   - **GINE**: 그래프 합성곱 신경망(GNN) 모델로, 화학 구조 학습을 위해 설계됨.  
   - **GINE-VN**: 가상 노드(Virtual Node)를 추가하여 GINE을 개선한 모델.  
   - **AttentiveFP**: 어텐션(attention) 기반 그래프 신경망을 활용하여 분자 표현 학습.  
   - **O-GNN**: 기존 GNN의 성능을 개선하기 위해 고리 정보를 추가한 모델.  

3. **그래프 풀링(Graph Pooling) 기반 방법**  
   - **TopKPool**: 그래프에서 중요한 노드를 선별하는 풀링(pooling) 기법 적용.  
   - **SAGPool**: 자기 어텐션(self-attention) 기반 그래프 풀링 방법.  

4. **트랜스포머 기반 방법**  
   - **Graphormer**: 트랜스포머 구조를 그래프 데이터에 적용한 모델.  
   - **GPS (General, Powerful, Scalable Graph Transformer)**: 그래프 내 지역 정보와 전역 정보를 결합한 트랜스포머.  
   - **GraphViT**: 비전 트랜스포머(ViT) 기법을 그래프 모델에 적용한 방법.  

---

#### **2. 테스트 데이터 (Test Datasets)**
RingFormer는 **다섯 개의 OSC(유기 태양 전지) 분자 데이터셋**을 사용하여 평가되었다:

- **CEPDB**:  
  - 가장 큰 데이터셋으로 **230만 개(2.3M) 이상의 OSC 분자** 포함.  
  - 밀도 범함수 이론(DFT, Density Functional Theory)을 기반으로 생성됨.  

- **HOPV**:  
  - 실험적으로 검증된 **350개 OSC 분자** 포함.  
  - Harvard Organic Photovoltaic Dataset에서 가져옴.  

- **PFD**:  
  - **1055개 분자** 포함.  
  - 태양전지 재료 후보군을 평가하는 데이터셋.  

- **NFA**:  
  - **654개 분자** 포함.  
  - OSC에서 중요하게 고려되는 비퓨즈(non-fused) 및 퓨즈(fused) 구조 포함.  

- **PD**:  
  - **277개 분자** 포함.  
  - OSC의 고성능 소자 후보군으로 구성된 실험적 데이터셋.  

---

#### **3. 평가 지표 (Evaluation Metric)**
모델 성능은 **평균 절대 오차 (Mean Absolute Error, MAE)**로 평가되었다.  
- MAE 값이 낮을수록 예측이 더 정확함을 의미.  
- 각 데이터셋에서 **전력 변환 효율(Power Conversion Efficiency, PCE)**과 함께 **HOMO, LUMO, Band Gap, Voc, Jsc** 등의 속성도 추가로 평가됨.

---

#### **4. 실험 결과 (Experimental Results)**
- **RingFormer는 모든 데이터셋에서 경쟁 모델을 능가**했다.  
- 특히, **CEPDB 데이터셋에서 기존 최고 모델 대비 22.77% 성능 향상**을 기록.  
- PCE(전력 변환 효율) 예측에서 가장 낮은 MAE(오차)를 달성함.  
- **트랜스포머 기반 모델(GPS, Graphormer)보다도 높은 성능**을 보이며,  
  - GNN 기반 모델보다 전역 구조를 더 잘 반영함을 확인.  

테스트 데이터셋별 **PCE 예측 성능 (MAE, 낮을수록 좋음)**:
| **모델**       | **CEPDB** | **HOPV** | **PFD** | **NFA** | **PD**  |
|---------------|---------|--------|-------|-------|-------|
| MACCS        | 0.898   | 1.632  | 1.770 | 2.614 | 2.594 |
| ECFP         | 0.510   | 1.544  | 1.787 | 2.377 | 2.704 |
| GINE         | 0.460   | 1.614  | 1.826 | 2.620 | 2.528 |
| O-GNN        | 0.267   | 1.727  | 1.868 | 2.587 | 2.866 |
| Graphormer   | -       | 1.609  | 1.799 | 2.689 | 2.522 |
| **RingFormer** | **0.189** | **1.477** | **1.776** | **2.259** | **2.482** |

- RingFormer는 **PFD 데이터셋을 제외한 모든 데이터셋에서 1위를 기록**.  
- 특히 **CEPDB에서는 가장 성능이 좋은 기존 모델(GraphViT)의 0.244에서 0.189로 22.77% 향상**.  
- OSC 분자에서 **고리 시스템(Ring System)의 중요성을 반영한 RingFormer의 설계가 효과적**임을 입증.  

---



#### **1. Baseline Models**  
RingFormer was evaluated against **11 existing models**, categorized into four types:

1. **Fingerprint-Based Methods**  
   - **MACCS**: Uses a 166-bit molecular fingerprint representation.  
   - **ECFP (Extended Connectivity Fingerprint)**: Provides an extended representation of molecular structure.  

2. **GNN-Based Methods**  
   - **GINE**: A graph convolutional network (GNN) model for molecular property prediction.  
   - **GINE-VN**: An improved version of GINE incorporating a virtual node.  
   - **AttentiveFP**: Attention-based GNN that learns molecular representations.  
   - **O-GNN**: Incorporates ring-based features into traditional GNN models.  

3. **Graph Pooling-Based Methods**  
   - **TopKPool**: Uses node selection for graph pooling.  
   - **SAGPool**: Self-attention-based graph pooling method.  

4. **Transformer-Based Methods**  
   - **Graphormer**: A transformer model adapted for graph data.  
   - **GPS**: Combines local and global information in graph transformer architectures.  
   - **GraphViT**: Uses Vision Transformer (ViT) techniques for graphs.  

---

#### **2. Test Data**  
RingFormer was evaluated on **five OSC molecular datasets**:

- **CEPDB**:  
  - The **largest dataset**, containing **2.3M+ molecules**.  
  - Generated using **density functional theory (DFT)**.  

- **HOPV**:  
  - Contains **350 experimentally validated molecules**.  

- **PFD**:  
  - **1,055 molecules** for OSC material screening.  

- **NFA**:  
  - **654 molecules** with fused and non-fused structures.  

- **PD**:  
  - **277 molecules** experimentally validated as high-performance OSC candidates.  

---

#### **3. Evaluation Metric**  
The **Mean Absolute Error (MAE)** was used to measure model accuracy:  
- Lower MAE values indicate **higher prediction accuracy**.  
- Additional OSC properties (HOMO, LUMO, Band Gap, Voc, Jsc) were also assessed.  

---

#### **4. Performance Results**  
- **RingFormer outperformed all baselines** across datasets.  
- **Achieved 22.77% relative improvement over the best existing model on CEPDB.**  
- Outperformed even **Graphormer and GPS (transformer-based models).**  

RingFormer effectively **captures ring systems, leading to superior molecular property predictions.**



<br/>
# 예제  




#### **1. 훈련 데이터 예제 (Training Data Example)**
RingFormer는 **OSC(유기 태양 전지) 분자 데이터셋**을 사용하여 훈련된다.  
훈련 데이터는 일반적으로 **그래프 형식의 분자 구조 정보와 해당 분자의 물리적 특성 값**을 포함한다.

**입력 (예제 분자 데이터 - Graph Format)**  
| 원자 ID | 원자 유형 | 결합된 원자들 | 고리 포함 여부 |
|--------|---------|--------------|--------------|
| 1      | C (탄소)  | 2, 3          | O            |
| 2      | C (탄소)  | 1, 3, 4       | O            |
| 3      | O (산소)  | 1, 2, 5       | O            |
| 4      | H (수소)  | 2            | X            |
| 5      | N (질소)  | 3            | O            |

**출력 (해당 분자의 목표 예측값 - Regression Targets)**  
| 특성(Property) | 값(Value) |
|--------------|---------|
| **전력 변환 효율 (PCE, %)** | 13.2 |
| **최고 점유 분자 오비탈 (HOMO, eV)** | -5.1 |
| **최저 비점유 분자 오비탈 (LUMO, eV)** | -3.2 |
| **밴드 갭 (Band Gap, eV)** | 1.9 |
| **개방 회로 전압 (Voc, V)** | 0.95 |
| **단락 전류 밀도 (Jsc, mA/cm²)** | 18.7 |

---

#### **2. 실험 데이터 예제 (Experimental Data Example)**
RingFormer는 **테스트 데이터셋(CEPDB, HOPV, PFD, NFA, PD)**에서 성능을 검증했다.  
실험 데이터는 새로운 OSC 분자들을 포함하며, 훈련 데이터와 유사한 입력 구조를 갖는다.

**입력 (새로운 OSC 분자 - Graph Input Example)**  
- **분자 그래프 구조**: 원자 노드(탄소, 산소, 질소 등) 및 결합 관계
- **고리 정보**: 벤젠 고리(6원자), 퓨란(5원자) 등

**출력 (예측된 분자 특성 - Model Output Example)**  
| 특성(Property) | 실제 값 (True) | 예측 값 (Predicted) |
|--------------|-------------|----------------|
| **PCE (%)**  | 12.8        | **12.9**       |
| **HOMO (eV)** | -5.0        | **-5.1**       |
| **LUMO (eV)** | -3.3        | **-3.2**       |
| **Band Gap (eV)** | 1.7        | **1.8**        |
| **Voc (V)**  | 0.92        | **0.93**       |
| **Jsc (mA/cm²)** | 19.5       | **19.2**       |

RingFormer는 **실제 값과 거의 일치하는 예측 성능을 보이며, 기존 모델 대비 낮은 MAE(평균 절대 오차)를 기록**했다.

---

#### **3. 태스크 입력/출력 예제 (Task Input/Output Example)**
RingFormer의 실제 사용 예제는 다음과 같다.

**입력 (Task Input Example - OSC 분자 예측 요청)**  
```python
input_molecule = {
    "atoms": ["C", "C", "O", "H", "N"],
    "bonds": [(1, 2), (2, 3), (3, 5), (2, 4)],
    "ring_systems": [["C", "C", "O", "N"]],
}

predicted_properties = model.predict(input_molecule)
```

 **출력 (Task Output Example - 예측 결과)**  
```json
{
    "PCE (%)": 12.9,
    "HOMO (eV)": -5.1,
    "LUMO (eV)": -3.2,
    "Band Gap (eV)": 1.8,
    "Voc (V)": 0.93,
    "Jsc (mA/cm²)": 19.2
}
```

---

### **English Explanation (Examples: Training Data, Experimental Data, Task Input/Output)**  

#### **1. Training Data Example**
RingFormer is trained on **organic solar cell (OSC) molecular datasets**.  
The training data typically consists of **graph-structured molecular representations and their physical properties**.

 **Input (Example Molecular Graph - Graph Format)**  
| Atom ID | Atom Type | Bonded Atoms | Part of Ring? |
|--------|---------|--------------|--------------|
| 1      | C (Carbon)  | 2, 3          | Yes        |
| 2      | C (Carbon)  | 1, 3, 4       | Yes        |
| 3      | O (Oxygen)  | 1, 2, 5       | Yes        |
| 4      | H (Hydrogen) | 2            | No         |
| 5      | N (Nitrogen) | 3            | Yes        |

 **Output (Target Molecular Properties - Regression Targets)**  
| Property | Value |
|--------------|---------|
| **Power Conversion Efficiency (PCE, %)** | 13.2 |
| **Highest Occupied Molecular Orbital (HOMO, eV)** | -5.1 |
| **Lowest Unoccupied Molecular Orbital (LUMO, eV)** | -3.2 |
| **Band Gap (eV)** | 1.9 |
| **Open Circuit Voltage (Voc, V)** | 0.95 |
| **Short-Circuit Current Density (Jsc, mA/cm²)** | 18.7 |

---

#### **2. Experimental Data Example**
RingFormer was evaluated on **five OSC molecular datasets (CEPDB, HOPV, PFD, NFA, PD)**.  
The test data consists of **new OSC molecules with similar graph structures**.

 **Input (New OSC Molecule - Graph Input Example)**  
- **Graph representation**: Nodes (atoms like Carbon, Oxygen, Nitrogen) and bond relations.
- **Ring system information**: Benzene (6-membered), Furan (5-membered), etc.

 **Output (Predicted vs. True Molecular Properties - Model Output Example)**  
| Property | True Value | Predicted Value |
|--------------|-------------|----------------|
| **PCE (%)**  | 12.8        | **12.9**       |
| **HOMO (eV)** | -5.0        | **-5.1**       |
| **LUMO (eV)** | -3.3        | **-3.2**       |
| **Band Gap (eV)** | 1.7        | **1.8**        |
| **Voc (V)**  | 0.92        | **0.93**       |
| **Jsc (mA/cm²)** | 19.5       | **19.2**       |

RingFormer demonstrates **high prediction accuracy**, achieving lower MAE compared to baseline models.

---

#### **3. Task Input/Output Example**
RingFormer can be applied to **real-world OSC molecule property prediction**.

 **Input (Task Input Example - OSC Molecule Prediction Request)**  
```python
input_molecule = {
    "atoms": ["C", "C", "O", "H", "N"],
    "bonds": [(1, 2), (2, 3), (3, 5), (2, 4)],
    "ring_systems": [["C", "C", "O", "N"]],
}

predicted_properties = model.predict(input_molecule)
```

 **Output (Task Output Example - Predicted Properties)**  
```json
{
    "PCE (%)": 12.9,
    "HOMO (eV)": -5.1,
    "LUMO (eV)": -3.2,
    "Band Gap (eV)": 1.8,
    "Voc (V)": 0.93,
    "Jsc (mA/cm²)": 19.2
}
```

RingFormer effectively **predicts key OSC molecular properties, making it highly useful for material discovery.**


<br/>  
# 요약   


RingFormer는 유기 태양 전지(OSC) 분자의 특성을 예측하기 위해 원자 및 고리 수준의 구조 정보를 통합한 계층적 그래프 트랜스포머 모델이다. 실험 결과, RingFormer는 기존 GNN 및 트랜스포머 기반 모델을 능가하며, 특히 CEPDB 데이터셋에서 기존 최고 성능 대비 22.77% 향상된 예측 정확도를 기록했다. 예제 실험에서도 RingFormer는 실제 분자 특성과 유사한 예측 결과를 생성하며, OSC 분자 설계 및 최적화에 효과적인 도구로 활용될 수 있음을 입증했다.  

---


RingFormer is a hierarchical graph transformer model designed to predict the properties of organic solar cell (OSC) molecules by integrating atomic and ring-level structural information. Experimental results show that RingFormer outperforms existing GNN and transformer-based models, achieving a 22.77% improvement over the best competitor on the CEPDB dataset. In example experiments, RingFormer generates predictions closely matching real molecular properties, demonstrating its potential as an effective tool for OSC molecular design and optimization.


<br/>  
# 기타  



#### **1. RingFormer 아키텍처 다이어그램 (Figure: RingFormer Architecture)**
- **계층적 OSC 그래프 (Hierarchical OSC Graph)**  
  - 원자-수준 그래프 (\(G_A\)): 원자와 결합을 표현.  
  - 고리-수준 그래프 (\(G_R\)): 분자의 고리 시스템과 상호 연결을 반영.  
  - 원자-고리 간 인터페이스 그래프 (\(G_I\)): 원자와 고리 간 관계를 모델링.  

- **RingFormer 모델 구조**  
  - **원자-수준 메시지 전달 (Atom-Level Message Passing)**: 원자 간 결합과 지역적 화학 구조 학습.  
  - **고리-수준 크로스 어텐션 (Ring-Level Cross-Attention)**: 고리 시스템 내 전역 구조 학습.  
  - **원자-고리 상호작용 메시지 전달 (Inter-Level Message Passing)**: 원자와 고리 사이 정보 교환.  
  - **계층적 융합 (Hierarchical Fusion)**: 다양한 수준에서 학습된 정보를 통합하여 최종 분자 표현 생성.  

---

#### **2. 성능 비교 테이블 (Table: Performance Comparison)**
테스트 데이터셋에서 **RingFormer가 기존 모델 대비 우수한 성능을 보였음을 나타냄**.  
- **평가 지표**: **Mean Absolute Error (MAE, 낮을수록 좋음)**.  
- **RingFormer가 모든 데이터셋에서 최고 또는 준최고 성능을 기록**.

| **모델**       | **CEPDB** | **HOPV** | **PFD** | **NFA** | **PD**  |
|---------------|---------|--------|-------|-------|-------|
| MACCS        | 0.898   | 1.632  | 1.770 | 2.614 | 2.594 |
| ECFP         | 0.510   | 1.544  | 1.787 | 2.377 | 2.704 |
| GINE         | 0.460   | 1.614  | 1.826 | 2.620 | 2.528 |
| O-GNN        | 0.267   | 1.727  | 1.868 | 2.587 | 2.866 |
| Graphormer   | -       | 1.609  | 1.799 | 2.689 | 2.522 |
| **RingFormer** | **0.189** | **1.477** | **1.776** | **2.259** | **2.482** |

- **CEPDB 데이터셋에서 RingFormer가 기존 최고 모델 대비 22.77% 향상된 성능 기록**.  
- 다른 데이터셋에서도 **일관되게 가장 낮은 MAE를 유지하며 우수한 성능을 입증**.  

---

#### **3. RingFormer 성능 시각화 (Figure: Performance Visualization)**
- **RingFormer의 성능 향상 정도를 분자의 고리 수에 따라 분석**.  
- **UMAP 시각화**를 통해 RingFormer가 분자의 고리 개수에 따라 더욱 명확한 구조적 구분을 수행함을 확인.  
- 고리 수가 증가할수록 RingFormer의 성능이 향상됨을 **상대적 성능 향상 그래프(Relative Improvement Graph)**에서 확인 가능.  

---



#### **1. RingFormer Architecture Diagram (Figure: RingFormer Architecture)**
- **Hierarchical OSC Graph Construction**  
  - **Atom-Level Graph (\(G_A\))**: Represents atomic bonds.  
  - **Ring-Level Graph (\(G_R\))**: Captures ring systems and interconnections.  
  - **Inter-Level Graph (\(G_I\))**: Models relationships between atoms and rings.  

- **RingFormer Model Components**  
  - **Atom-Level Message Passing**: Captures local chemical structures.  
  - **Ring-Level Cross-Attention**: Learns global structural patterns within ring systems.  
  - **Inter-Level Message Passing**: Enables interaction between atom and ring representations.  
  - **Hierarchical Fusion**: Merges multi-level structural information into a final molecular representation.  

---

#### **2. Performance Comparison Table (Table: Performance Comparison)**
The table presents **RingFormer’s superior performance compared to existing models**.  
- **Evaluation Metric**: **Mean Absolute Error (MAE, lower is better)**.  
- **RingFormer achieves the best or second-best performance across all datasets**.

| **Model**       | **CEPDB** | **HOPV** | **PFD** | **NFA** | **PD**  |
|---------------|---------|--------|-------|-------|-------|
| MACCS        | 0.898   | 1.632  | 1.770 | 2.614 | 2.594 |
| ECFP         | 0.510   | 1.544  | 1.787 | 2.377 | 2.704 |
| GINE         | 0.460   | 1.614  | 1.826 | 2.620 | 2.528 |
| O-GNN        | 0.267   | 1.727  | 1.868 | 2.587 | 2.866 |
| Graphormer   | -       | 1.609  | 1.799 | 2.689 | 2.522 |
| **RingFormer** | **0.189** | **1.477** | **1.776** | **2.259** | **2.482** |

- **RingFormer achieves a 22.77% improvement over the best-performing competitor on CEPDB**.  
- Consistently **achieves the lowest MAE across datasets, proving its effectiveness**.  

---

#### **3. RingFormer Performance Visualization (Figure: Performance Visualization)**
- **Performance improvement analyzed based on the number of rings in molecules**.  
- **UMAP visualization** shows that RingFormer **effectively differentiates molecular structures based on ring count**.  
- **Relative improvement graph** confirms that **RingFormer's advantage increases as ring system complexity grows**.


<br/>
# refer format:     

@article{Ding2025,
  author    = {Zhihao Ding and Ting Zhang and Yiran Li and Jieming Shi and Chen Jason Zhang},
  title     = {RingFormer: A Ring-Enhanced Graph Transformer for Organic Solar Cell Property Prediction},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  url       = {https://arxiv.org/abs/2412.09030}
}




Ding, Zhihao, Ting Zhang, Yiran Li, Jieming Shi, and Chen Jason Zhang. "RingFormer: A Ring-Enhanced Graph Transformer for Organic Solar Cell Property Prediction." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. Association for the Advancement of Artificial Intelligence (AAAI). https://arxiv.org/abs/2412.09030.

