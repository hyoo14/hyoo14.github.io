---
layout: post
title:  "[2023]GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning"  
date:   2025-02-25 00:22:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

그래프 대신 언어모델로 그래프 구조까지(설명으로) 포함하여 분자-텍스트 테스크 수행  



짧은 요약(Abstract) :    




최근 분자 특성 예측은 많은 주목을 받고 있지만, 실험적 데이터의 부족이 주요한 장애 요소로 작용하고 있다. 이를 해결하기 위해 자연어 지시문을 이용한 무감독 학습 방식(Zero-shot learning)을 적용하는 연구가 진행되었다. 그러나 기존의 분자-텍스트 모델들은 이러한 방식에서 성능이 낮은 것으로 나타났다. 이에 연구팀은 GIMLET이라는 새로운 모델을 제안하였다. GIMLET은 언어 모델을 그래프 및 텍스트 데이터 모두에 적용하여, 추가적인 그래프 인코딩 모듈 없이 그래프 구조와 지시문을 함께 처리할 수 있도록 설계되었다. 또한 그래프 인코딩과 지시문을 주의 메커니즘(Attention Mechanism)에서 분리하여 그래프 특성의 일반화를 향상시켰다. 연구진은 2,000개 이상의 분자 관련 과제와 해당 지시문으로 구성된 데이터셋을 구축하여 모델을 사전 학습하였으며, 이를 통해 다양한 과제로의 전이 학습이 가능함을 입증하였다. 실험 결과, GIMLET은 기존 분자-텍스트 모델보다 월등한 성능을 보였으며, 일부 과제에서는 지도 학습된 그래프 신경망(GNN) 모델과 유사한 수준의 결과를 보였다.

---


Molecule property prediction has gained significant attention in recent years. The main bottleneck is the label insufficiency caused by expensive lab experiments. In order to alleviate this issue and to better leverage textual knowledge for tasks, this study investigates the feasibility of employing natural language instructions to accomplish molecule-related tasks in a zero-shot setting. We discover that existing molecule-text models perform poorly in this setting due to inadequate treatment of instructions and limited capacity for graphs. To overcome these issues, we propose GIMLET, which unifies language models for both graph and text data. By adopting generalized position embedding, our model is extended to encode both graph structures and instruction text without additional graph encoding modules. GIMLET also decouples encoding of the graph from tasks instructions in the attention mechanism, enhancing the generalization of graph features across novel tasks. We construct a dataset consisting of more than two thousand molecule tasks with corresponding instructions derived from task descriptions. We pretrain GIMLET on the molecule tasks along with instructions, enabling the model to transfer effectively to a broad range of tasks. Experimental results demonstrate that GIMLET significantly outperforms molecule-text baselines in instruction-based zero-shot learning, even achieving closed results to supervised GNN models on tasks such as toxcast and muv.



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





#### **사용한 모델**
- **GIMLET 모델**: GIMLET(Graph Instruction-based Molecule Zero-shot Learning)은 그래프와 텍스트 데이터를 동시에 처리할 수 있도록 설계된 통합된 언어 모델이다.
- **Transformer 기반**: 기존의 그래프 신경망(GNN)을 따로 사용하지 않고, **일반화된 위치 임베딩(Generalized Position Embedding)** 기법을 활용하여 Transformer 모델 내에서 그래프 구조와 텍스트 지시문을 함께 인코딩할 수 있도록 구성되었다.
- **그래프와 지시문 디커플링(Decoupling)**: 그래프 데이터와 지시문을 **어텐션 마스크(Attention Masking)** 를 이용해 분리하여, 그래프 특징이 다양한 과제에서 일반화될 수 있도록 설계되었다.

#### **사용한 학습 데이터**
- **사전 학습 데이터 (Pretraining Data)**
  - **ChEMBL**: 1,310개의 생물학적 실험 데이터에서 예측해야 할 목표 라벨을 포함하는 대규모 데이터셋으로, 주로 신약 개발에 사용됨.
  - **ChEMBL-Property 데이터셋**: 화합물의 물리·화학적 특성을 포함하는 별도의 데이터셋으로, 분자의 다양한 속성(분자량, 수소 결합 수, 극성 표면적 등)이 포함됨.
  - **사전 학습 방식**: 지도 학습 방식으로 진행되며, 분자 그래프와 지시문을 함께 입력하여 모델이 다양한 분자 관련 과제를 해결할 수 있도록 훈련됨.

- **다운스트림(Zero-Shot & Few-Shot) 평가 데이터**
  - **PCBA**: 128개의 생물학적 실험을 포함하는 대규모 데이터셋.
  - **ChEMBL Zero-Shot**: 사전 학습 데이터와 겹치지 않는 262개의 새로운 분자 과제.
  - **MoleculeNet 벤치마크**: 물리·화학적 특성, 독성 예측, 약물 동태학 등 다양한 분자 예측 과제를 포함.
  - **Tox21, ToxCast**: 독성 예측을 위한 데이터셋.
  - **CYP450, BBBP**: 약물 대사 및 혈액-뇌 장벽(Blood-Brain Barrier) 투과성 예측을 포함.

#### **특별한 점**
1. **GIMLET 모델의 독창성**
   - 그래프와 텍스트 데이터를 추가적인 GNN 인코딩 없이 직접 Transformer에 통합.
   - 그래프의 구조를 **일반화된 위치 임베딩(Generalized Position Embedding)** 방식으로 반영.
   - **어텐션 마스크(Attention Masking)** 를 통해 그래프와 지시문의 인코딩을 분리하여 다양한 과제에 대한 일반화 성능 강화.

2. **사전 학습 과정의 특이점**
   - 기존의 **SMILES 문자 표현**이 아닌, **자연어 지시문(Natural Language Instructions)** 기반 학습.
   - 단순한 단일 태스크 학습이 아닌, 다양한 생물학적 실험과 물리·화학적 과제들을 **지시문 기반 전이 학습(Instruction-based Transfer Learning)** 방식으로 훈련.

3. **Zero-Shot & Few-Shot 평가**
   - 기존 분자-텍스트 모델들은 Zero-Shot 학습에서 성능이 저조했으나, GIMLET은 지도 학습(GNN) 모델과 비슷한 성능을 달성.
   - Few-Shot 학습에서도 빠르게 성능이 향상되며, 적은 데이터로도 우수한 결과를 보임.

---


#### **Model Used**
- **GIMLET Model**: GIMLET (Graph Instruction-based Molecule Zero-shot Learning) is a unified model that simultaneously processes **graph and text data**.
- **Transformer-Based**: Instead of using a separate **Graph Neural Network (GNN)**, GIMLET utilizes a **generalized position embedding** technique, enabling the Transformer to encode both graph structures and text instructions seamlessly.
- **Decoupled Attention Mechanism**: Graph data and task instructions are separated using **attention masks**, enhancing the generalization of graph features across various novel tasks.

#### **Training Data Used**
- **Pretraining Data**
  - **ChEMBL**: A dataset containing **1,310 bioassay tasks** with prediction target labels, widely used in drug discovery.
  - **ChEMBL-Property Dataset**: A dataset containing various **physicochemical properties** (e.g., molecular weight, hydrogen bond acceptors, polar surface area, etc.).
  - **Pretraining Method**: The model was **supervised pre-trained** with both molecular graphs and natural language instructions to enhance generalizability across multiple molecular tasks.

- **Downstream Evaluation Datasets (Zero-Shot & Few-Shot)**
  - **PCBA**: A large-scale dataset with **128 bioassay tasks**.
  - **ChEMBL Zero-Shot**: **262 novel molecular tasks** not included in the pretraining dataset.
  - **MoleculeNet Benchmark**: Includes multiple molecular prediction tasks, such as physicochemical property estimation, toxicity prediction, and pharmacokinetics.
  - **Tox21, ToxCast**: Datasets for **toxicity prediction**.
  - **CYP450, BBBP**: Datasets for **drug metabolism and blood-brain barrier permeability prediction**.

#### **Key Innovations**
1. **Uniqueness of GIMLET**
   - Directly integrates **graph and text data** within a single Transformer model without using an additional **GNN encoder**.
   - Introduces **generalized position embedding** to represent graph structures.
   - Uses **attention masking** to separate the encoding of graphs from instructions, improving generalization across diverse tasks.

2. **Novel Pretraining Process**
   - Instead of relying on **SMILES string representations**, GIMLET **incorporates natural language instructions** for learning.
   - Uses **instruction-based transfer learning**, where the model learns from a wide variety of molecular tasks with **task descriptions in natural language**.

3. **Zero-Shot & Few-Shot Evaluation**
   - Unlike existing molecule-text models that perform poorly in **zero-shot learning**, GIMLET achieves performance **comparable to supervised GNN models**.
   - In **few-shot learning**, GIMLET quickly improves performance with minimal data, demonstrating strong representation learning.

This methodology allows GIMLET to bridge the gap between **molecular graph learning and natural language processing**, making it a powerful model for **zero-shot molecular property prediction**.
   
 
<br/>
# Results  



#### **비교 대상 경쟁 모델**
- **Zero-Shot 비교 모델**:
  - **KVPLM** (110M parameters): SMILES 기반의 분자-텍스트 언어 모델.
  - **MoMu** (113M parameters): 그래프 신경망(GNN)과 자연어를 결합한 분자-텍스트 모델.
  - **Galactica** (125M/1.3B parameters): 분자 데이터를 포함한 대규모 사전 훈련된 자연어 모델.
  - **GIMLET** (64M parameters): 연구에서 제안한 그래프-텍스트 통합 Transformer 모델.

- **Supervised 비교 모델** (지도 학습 성능과 비교):
  - **GCN** (0.5M parameters): 그래프 컨볼루션 신경망.
  - **GAT** (1.0M parameters): 그래프 어텐션 네트워크.
  - **GIN** (1.8M parameters): 그래프 임베딩 네트워크.
  - **Graphormer** (48M parameters): Transformer 기반 그래프 학습 모델.
  - **Graphormer-p** (48M parameters): 연구에서 사용한 사전 학습 데이터로 추가 학습된 Graphormer.

---

#### **테스트 데이터셋**
1. **Bio-Activity Tasks (생물학적 활성 예측)**
   - **BACE**: 베타-세크레타제(BACE) 억제제 예측.
   - **HIV**: HIV 억제 활성 예측.
   - **MUV**: 대규모 생물학적 실험 데이터.

2. **Toxicity Tasks (독성 예측)**
   - **Tox21**: 21세기 독성 예측 과제.
   - **ToxCast**: 다양한 생물학적 독성 실험 데이터.

3. **Pharmacokinetic Tasks (약물 동태학 예측)**
   - **BBBP**: 혈액-뇌 장벽(Blood-Brain Barrier) 투과성 예측.
   - **CYP450**: CYP450 효소에 의한 대사 과정 예측.

4. **Large-Scale Molecule Tasks**
   - **PCBA**: 128개의 생물학적 실험 포함.
   - **ChEMBL Zero-Shot**: 사전 학습 데이터와 겹치지 않는 새로운 분자 과제.

5. **Physicochemical Property Prediction (물리·화학적 특성 예측)**
   - **ESOL**: 물 용해도 예측.
   - **Lipophilicity**: 지용성(LogP) 예측.
   - **FreeSolv**: 자유 에너지 예측.

---

#### **사용한 평가 지표 (Evaluation Metrics)**
- **ROC-AUC** (높을수록 좋음): 분류 모델의 예측 성능을 평가하는 지표.
  - Bio-Activity, Toxicity, Pharmacokinetics, Large-Scale Molecule Tasks에서 사용.
- **RMSE** (낮을수록 좋음): 회귀 모델에서 예측값과 실제값 사이의 평균 제곱근 오차.
  - Physicochemical Property Prediction에서 사용.

---

#### **결과 요약**
1. **Zero-Shot Learning (지도 학습 없이 성능 평가)**
   - GIMLET은 **모든 Zero-Shot 분자-텍스트 모델을 능가**하며, 일부 데이터셋(BACE, MUV, ToxCast 등)에서는 **지도 학습된 GNN 모델과 유사한 성능**을 보임.
   - **PCBA, ChEMBL Zero-Shot**과 같은 대규모 데이터셋에서도 **기존 모델 대비 높은 성능**을 기록.

2. **Few-Shot Learning (일부 샘플을 활용한 추가 학습)**
   - 소수의 데이터만 추가 학습해도 GIMLET의 성능이 빠르게 향상됨.
   - 기존 분자-텍스트 모델들은 성능 개선이 불안정하지만, GIMLET은 **안정적으로 성능을 향상**.

3. **Physicochemical Property Prediction**
   - 회귀 모델 평가에서 다른 Zero-Shot 모델들은 **올바른 숫자 형식조차 생성하지 못하는 경우**가 많았음.
   - 반면, **GIMLET은 98% 이상의 샘플에서 정확한 숫자 형식을 생성**, RMSE 기준으로도 가장 낮은 오차를 기록.

---


#### **Competitor Models for Comparison**
- **Zero-Shot Baseline Models**:
  - **KVPLM** (110M parameters): A SMILES-based molecule-text language model.
  - **MoMu** (113M parameters): A GNN-based molecule-text model with natural language.
  - **Galactica** (125M/1.3B parameters): A large-scale pre-trained language model incorporating molecular data.
  - **GIMLET** (64M parameters): The proposed graph-text unified Transformer model.

- **Supervised Baselines for Upper Bound Comparison**:
  - **GCN** (0.5M parameters): Graph Convolutional Network.
  - **GAT** (1.0M parameters): Graph Attention Network.
  - **GIN** (1.8M parameters): Graph Isomorphism Network.
  - **Graphormer** (48M parameters): Transformer-based graph learning model.
  - **Graphormer-p** (48M parameters): Graphormer trained on the study's pretraining dataset.

---

#### **Test Datasets**
1. **Bio-Activity Tasks**
   - **BACE**: Beta-secretase (BACE) inhibition prediction.
   - **HIV**: HIV inhibition prediction.
   - **MUV**: Large-scale biological assay dataset.

2. **Toxicity Prediction**
   - **Tox21**: Toxicity assessment dataset.
   - **ToxCast**: Comprehensive biological toxicity experiments.

3. **Pharmacokinetics Tasks**
   - **BBBP**: Blood-Brain Barrier permeability prediction.
   - **CYP450**: Prediction of metabolism by CYP450 enzymes.

4. **Large-Scale Molecular Tasks**
   - **PCBA**: 128 biological assay tasks.
   - **ChEMBL Zero-Shot**: New molecular tasks that were not in the pretraining dataset.

5. **Physicochemical Property Prediction**
   - **ESOL**: Water solubility prediction.
   - **Lipophilicity**: LogP estimation.
   - **FreeSolv**: Free energy of solvation.

---

#### **Evaluation Metrics**
- **ROC-AUC** (Higher is better): Evaluates classification model performance.
  - Used for Bio-Activity, Toxicity, Pharmacokinetics, and Large-Scale Molecular Tasks.
- **RMSE** (Lower is better): Root Mean Square Error used for regression tasks.
  - Applied in Physicochemical Property Prediction.

---

#### **Key Findings**
1. **Zero-Shot Learning Performance**
   - **GIMLET outperforms all existing molecule-text baselines**, achieving performance close to **supervised GNN models** on datasets like BACE, MUV, and ToxCast.
   - **Excels in large-scale tasks** such as PCBA and ChEMBL Zero-Shot, showing strong **generalization**.

2. **Few-Shot Learning Performance**
   - GIMLET exhibits **consistent performance gains** with few-shot finetuning, whereas other molecule-text models show **unstable improvements**.
   - With just a small number of labeled samples, GIMLET achieves **performance comparable to supervised models**.

3. **Physicochemical Property Prediction**
   - Other zero-shot models often **fail to generate valid numerical outputs** for regression tasks.
   - **GIMLET successfully produces correctly formatted numbers for over 98% of test cases** and achieves the lowest RMSE among all zero-shot models.

---

### **Conclusion**
GIMLET successfully bridges the gap between **molecular property prediction and natural language instructions**, demonstrating **state-of-the-art zero-shot learning capabilities**. It **outperforms existing baselines**, adapts well to **few-shot learning**, and **achieves near-supervised performance** in several tasks. This validates the effectiveness of instruction-based learning for molecular prediction.


<br/>
# 예제  



#### **훈련 데이터 (Pretraining Data)**
- **ChEMBL**: 1,310개의 생물학적 실험 과제 포함 (예: 특정 단백질 저해 여부 예측).
- **ChEMBL-Property**: 분자의 물리·화학적 특성 포함 (예: 용해도, 극성 표면적 등).
- **자연어 지시문 데이터**: 모델이 분자 관련 과제를 해결하도록 유도하는 텍스트 명령 포함.

#### **테스트 데이터 (Evaluation Data)**
- **PCBA**: 128개의 생물학적 실험 데이터셋.
- **Tox21, ToxCast**: 독성 예측을 위한 데이터셋.
- **BBBP**: 혈액-뇌 장벽 투과성 예측.
- **ESOL, Lipophilicity, FreeSolv**: 물리·화학적 성질 예측.

#### **수행한 테스크 예시**
1. **Bio-Activity 예측**  
   **입력:** "이 분자는 BACE 단백질을 억제할 가능성이 있습니까?"  
   **출력:** "가능성이 높음 / 낮음" (Binary classification)

2. **독성 예측**  
   **입력:** "이 화합물은 간독성을 유발할 가능성이 있습니까?"  
   **출력:** "가능성 있음 / 없음"

3. **물리·화학적 속성 예측**  
   **입력:** "이 화합물의 물 용해도(logS)는 얼마입니까?"  
   **출력:** "-2.5" (Regression task)

---


#### **Pretraining Data**
- **ChEMBL**: 1,310 bioassay tasks (e.g., predicting inhibition of specific proteins).
- **ChEMBL-Property**: Includes physicochemical properties (e.g., solubility, polar surface area).
- **Natural Language Instructions**: Text-based instructions guiding molecular tasks.

#### **Evaluation Data**
- **PCBA**: 128 biological assay tasks.
- **Tox21, ToxCast**: Toxicity prediction datasets.
- **BBBP**: Blood-brain barrier permeability prediction.
- **ESOL, Lipophilicity, FreeSolv**: Physicochemical property prediction.

#### **Example Tasks**
1. **Bio-Activity Prediction**  
   **Input:** "Does this molecule inhibit BACE protein?"  
   **Output:** "High probability / Low probability" (Binary classification)

2. **Toxicity Prediction**  
   **Input:** "Is this compound likely to cause liver toxicity?"  
   **Output:** "Yes / No"

3. **Physicochemical Property Prediction**  
   **Input:** "What is the water solubility (logS) of this compound?"  
   **Output:** "-2.5" (Regression task)




<br/>  
# 요약   



이 연구에서는 GIMLET이라는 그래프-텍스트 통합 모델을 제안하여, 자연어 지시문과 그래프 데이터를 함께 처리하도록 설계되었다. 모델은 ChEMBL 등의 데이터로 사전 학습되었으며, 독성 예측, 생물학적 활성 예측, 물리·화학적 특성 예측과 같은 다양한 분자 관련 과제를 수행할 수 있도록 훈련되었다. 실험 결과, GIMLET은 기존의 분자-텍스트 모델을 능가하는 성능을 보였으며, 일부 과제에서는 지도 학습된 그래프 신경망과 유사한 성능을 기록하였다.  

---


This study introduces GIMLET, a unified graph-text model designed to process both natural language instructions and molecular graph data. The model was pretrained on datasets such as ChEMBL and trained to perform various molecular tasks, including toxicity prediction, bioactivity prediction, and physicochemical property estimation. Experimental results demonstrate that GIMLET outperforms existing molecule-text models and achieves performance comparable to supervised GNNs in some tasks.


<br/>  
# 기타  



#### **1. 주요 피규어**
- **Figure 6**: GIMLET과 기존 모델(KVPLM, MoMu)의 성능 비교를 시각화한 산점도. 대각선(x=y) 아래에 위치한 점들이 많을수록 GIMLET이 기존 모델보다 우수함을 나타냄.
- **Figure 7**: 회귀 작업(Regression tasks)에 대한 예측값과 실제값의 상관관계를 나타내는 산점도. ESOL과 Lipophilicity의 경우 강한 상관관계를 보임.
- **Figure 8**: Few-shot 학습에서 GIMLET의 성능 향상을 나타내는 그래프. 샘플 수가 증가함에 따라 성능이 개선됨을 보여줌.
- **Figure 9~18**: 분자 구조에 대한 어텐션 시각화. 특정 분자의 생물학적 활성(BACE 억제제 예측) 및 혈액-뇌 장벽 투과성(BBBP) 예측을 수행할 때 모델이 주목하는 원자 및 결합을 강조.

---

#### **2. 주요 테이블**
- **Table 1**: Zero-shot 성능 비교(ROC-AUC 기준) - 생물학적 활성(Bio-Activity), 독성(Toxicity), 약물동태학(Pharmacokinetics) 분야에서 기존 모델(KVPLM, MoMu, Galactica) 대비 GIMLET이 높은 성능을 기록.
- **Table 2**: 대규모 분자 데이터셋(PCBA, ChEMBL Zero-Shot)에서의 성능. GIMLET이 기존 모델 대비 뛰어난 일반화 성능을 보임.
- **Table 3**: 물리·화학적 특성 예측(Regression task) 결과 비교. Zero-shot 환경에서 다른 모델들은 올바른 숫자를 출력하지 못하는 경우가 많았으나, GIMLET은 98% 이상의 샘플에서 정확한 숫자 형식을 생성함.
- **Table 4**: GIMLET의 구성 요소별 Ablation Study 결과. "Unifying"과 "Decoupling" 기법을 제거한 경우 성능이 저하됨을 확인.
- **Table 5**: Pretraining 방식 비교 - Bio-Activity, Toxicity, Pharmacokinetics, Physicochemical 특성 예측에서 ChEMBL 기반 사전학습이 큰 영향을 미친다는 점을 보여줌.

---

#### **3. 어펜딕스 핵심 내용**
- **C.2 Zero-Shot 세부 결과**: 테이블 10~12에 GIMLET과 기존 모델(KVPLM, MoMu, Galactica)의 세부 성능 비교 포함. 최근 CLAMP 모델과의 비교 결과도 포함.
- **C.3 Few-Shot 학습 결과**: Few-shot 설정에서 GIMLET이 기존 모델 대비 더 안정적으로 성능이 개선됨을 보여주는 실험 결과 제공.
- **C.4 Pretraining Ablation Study**: Bioactivity와 Physicochemical pretraining이 모델 성능 향상에 미치는 영향을 분석한 실험 결과 포함.
- **C.5 Instruction Robustness**: GPT-3.5를 사용해 지시문을 변형(단순 변형, 확장, 세부화, 축약)하여 실험한 결과, GIMLET이 기존 모델보다 강건한 성능을 보임.

---



#### **1. Key Figures**
- **Figure 6**: Scatter plot comparing GIMLET to KVPLM and MoMu. Points below the diagonal (x=y) indicate that GIMLET outperforms existing models.
- **Figure 7**: Scatter plot for regression tasks (ESOL, Lipophilicity). Strong correlation between predicted and actual values shows GIMLET’s effectiveness.
- **Figure 8**: Few-shot learning performance graph. Performance improves consistently as the number of examples increases.
- **Figures 9~18**: Attention visualization for molecular structure tasks. Highlights atoms and bonds crucial for tasks like BACE inhibition and BBB permeability.

---

#### **2. Key Tables**
- **Table 1**: Zero-shot performance comparison (ROC-AUC) for Bio-Activity, Toxicity, and Pharmacokinetics. GIMLET outperforms KVPLM, MoMu, and Galactica.
- **Table 2**: Performance on large-scale molecule datasets (PCBA, ChEMBL Zero-Shot). GIMLET demonstrates strong generalization.
- **Table 3**: Regression task performance. Other models fail to generate valid numerical outputs, but GIMLET produces correctly formatted numbers for over 98% of test samples.
- **Table 4**: Ablation study of GIMLET’s architecture. Removing "Unifying" and "Decoupling" components significantly degrades performance.
- **Table 5**: Pretraining methods comparison. ChEMBL-based pretraining has a substantial impact on downstream task performance.

---

#### **3. Key Appendix Points**
- **C.2 Detailed Zero-Shot Results**: Tables 10–12 provide an in-depth comparison of GIMLET against KVPLM, MoMu, and Galactica, including recent CLAMP results.
- **C.3 Few-Shot Learning Performance**: Demonstrates that GIMLET improves performance more consistently than existing models under few-shot conditions.
- **C.4 Pretraining Ablation Study**: Investigates the impact of Bioactivity and Physicochemical pretraining on performance.
- **C.5 Instruction Robustness**: GPT-3.5 was used to rephrase instructions (rewrite, expand, detail, shorten), showing that GIMLET is more robust to instruction variations compared to baselines.


<br/>
# refer format:     


@inproceedings{gimlet2023,
  author    = {Anonymous},
  title     = {GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning},
  booktitle = {Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}



Anonymous. 2023. "GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning." Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS).  





