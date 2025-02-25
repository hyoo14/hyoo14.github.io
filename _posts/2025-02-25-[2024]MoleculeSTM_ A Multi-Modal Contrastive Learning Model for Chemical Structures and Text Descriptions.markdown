---
layout: post
title:  "[2024]MoleculeSTM: A Multi-Modal Contrastive Learning Model for Chemical Structures and Text Descriptions"  
date:   2025-02-25 00:33:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

텍스트로 표현된 화학구조( 스마일스라고 일컬음: 예)아스피린(Aspirin): CC(=O)OC1=CC=CC=C1C(=O)O )와 이를 설명하는 텍스트를 같이 학습한 언어모델 사용하여 좋은 성능!  



짧은 요약(Abstract) :    



최근 인공지능(AI)이 신약 개발에 점점 더 많이 활용되고 있다. 하지만 기존 연구들은 주로 분자의 화학 구조에 초점을 맞춘 기계 학습 기법을 사용하며, 화학 분야에서 사용할 수 있는 방대한 텍스트 데이터를 충분히 활용하지 못하고 있다. 이 연구에서는 MoleculeSTM이라는 다중 모달 모델을 제안하여, 분자의 화학 구조와 텍스트 설명을 결합해 학습하는 새로운 방식의 대조 학습 전략을 도입하였다. 이를 위해 28만 개 이상의 화학 구조-텍스트 쌍을 포함하는 PubChemSTM 데이터셋을 구축하였다. MoleculeSTM의 효과를 입증하기 위해, 구조-텍스트 검색과 분자 편집이라는 두 가지 새로운 제로샷(zero-shot) 과제를 설계하였다. MoleculeSTM은 개방형 어휘(Open Vocabulary)와 조합성(Compositionality)이라는 두 가지 주요 특성을 가지며, 실험 결과 다양한 벤치마크에서 최첨단 일반화 성능을 달성하는 것으로 확인되었다.

---


There is increasing adoption of artificial intelligence in drug discovery. However, existing studies use machine learning to mainly utilize the chemical structures of molecules but ignore the vast textual knowledge available in chemistry. Incorporating textual knowledge enables us to realize new drug design objectives, adapt to text-based instructions, and predict complex biological activities. Here we present a multi-modal molecule structure-text model, **MoleculeSTM**, by jointly learning molecules’ chemical structures and textual descriptions via a contrastive learning strategy. To train MoleculeSTM, we construct a large multi-modal dataset, namely, **PubChemSTM**, with over **280,000** chemical structure-text pairs. To demonstrate the effectiveness and utility of MoleculeSTM, we design two challenging zero-shot tasks based on text instructions, including **structure-text retrieval** and **molecule editing**. MoleculeSTM has two main properties: **open vocabulary** and **compositionality** via natural language. In experiments, MoleculeSTM obtains the **state-of-the-art generalization ability** to novel biochemical concepts across various benchmarks.



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




이 연구에서는 **MoleculeSTM**이라는 다중 모달 모델을 제안하여 분자의 화학 구조와 텍스트 설명을 함께 학습하는 방식으로 훈련했다. 이를 위해 다음과 같은 주요 구성 요소와 트레이닝 데이터가 사용되었다.

#### **1. 트레이닝 데이터 (Training Data)**
- **PubChemSTM 데이터셋**: 
  - **PubChem** 데이터베이스에서 수집된 데이터로, 28만 개 이상의 화학 구조-텍스트 쌍으로 구성됨.
  - 기존 데이터셋보다 **28배 더 큰 규모**로, 분자의 화학적 및 생물학적 특성을 설명하는 텍스트 포함.

- **추가적인 사전학습 데이터**:
  - **ZINC 데이터베이스**: 5억 개 이상의 분자 구조를 포함하는 대규모 화합물 라이브러리.
  - **GEOM 데이터셋**: 25만 개의 분자 2D 및 3D 구조 데이터를 포함하며, 이를 활용한 GraphMVP 사전학습 모델 사용.

#### **2. 사용된 모델 (Models Used)**
- **화학 구조 인코더 (Chemical Structure Encoder)**
  - **SMILES 문자열 기반**: MegaMolBART 모델 사용 (ZINC 데이터베이스로 사전학습됨).
  - **2D 그래프 기반**: Graph Isomorphism Network (GIN) 사용, GraphMVP 방식으로 사전학습됨.

- **텍스트 인코더 (Text Encoder)**
  - **SciBERT** 사용: 과학 및 생물학 관련 텍스트로 학습된 BERT 모델.

- **대조 학습 (Contrastive Learning)**
  - **EBM-NCE, InfoNCE 손실 함수**를 사용하여 화학 구조와 텍스트 표현을 결합된 공간에서 정렬.
  - 분자의 화학적 구조와 텍스트 설명을 같은 개념으로 학습하도록 조정.

#### **3. 평가 및 실험 (Evaluation & Experiments)**
- **제로샷 구조-텍스트 검색 (Zero-shot Structure-Text Retrieval)**
  - **DrugBank**에서 세 가지 하위 데이터셋 사용:
    1. 약물 설명 (Description)
    2. 약리학적 특성 (Pharmacodynamics)
    3. 해부학적 치료 화학 (ATC) 코드 기반 검색

- **제로샷 분자 편집 (Zero-shot Text-based Molecule Editing)**
  - **ZINC 데이터베이스**에서 200개의 분자 샘플 사용.
  - 단일 및 다중 속성 편집, 결합 친화도 기반 편집, 특정 약물과의 유사성 증대 등의 과제 수행.

---



This study proposes **MoleculeSTM**, a multi-modal model that jointly learns **molecular chemical structures** and **text descriptions** through contrastive learning. The key components and datasets used are as follows:

#### **1. Training Data**
- **PubChemSTM Dataset**: 
  - Constructed from the **PubChem** database with **over 280,000** chemical structure-text pairs.
  - It is **28× larger** than previous datasets and includes **text descriptions** of molecular chemical and biological properties.

- **Additional Pretraining Data**:
  - **ZINC Database**: A large compound library containing over **500M molecular structures**.
  - **GEOM Dataset**: Contains **250K** molecular **2D and 3D** structures, used in **GraphMVP** pretraining.

#### **2. Models Used**
- **Chemical Structure Encoders**:
  - **SMILES String-based**: **MegaMolBART**, pretrained on the **ZINC** database.
  - **2D Graph-based**: **Graph Isomorphism Network (GIN)**, pretrained using **GraphMVP**.

- **Text Encoder**:
  - **SciBERT**: A **BERT-based** model pretrained on **scientific and biological text**.

- **Contrastive Learning Strategy**:
  - Uses **EBM-NCE** and **InfoNCE** loss functions to align molecular structures and text descriptions in a shared embedding space.
  - Helps **bridge chemical structure representations** with **textual descriptions** in a semantically meaningful way.

#### **3. Evaluation & Experiments**
- **Zero-shot Structure-Text Retrieval**:
  - Evaluated on **DrugBank**, using three subsets:
    1. **Drug Descriptions**
    2. **Pharmacodynamics Information**
    3. **Anatomical Therapeutic Chemical (ATC) Code-based Retrieval**

- **Zero-shot Text-based Molecule Editing**:
  - **200 molecular samples** randomly selected from **ZINC database**.
  - Tasks include **single-attribute and multi-attribute modifications**, **binding-affinity-based editing**, and **drug similarity editing**.

This approach enables **state-of-the-art generalization** in zero-shot molecular tasks and **improves molecular understanding by integrating textual information**.


   
 
<br/>
# Results  




---

#### **1. 비교 모델 (Baseline & Competitive Models)**
MoleculeSTM의 성능을 평가하기 위해, 다음과 같은 기존 최첨단 모델들과 비교하였다.

1. **SMILES 기반 모델 (단일 모달)**
   - **MegaMolBART**: ZINC 데이터베이스에서 사전학습된 BART 기반 모델.
   - **KV-PLM**: 분자 구조와 텍스트를 함께 학습한 사전학습 모델.

2. **그래프 기반 모델 (단일 모달)**
   - **AttrMasking**: 분자 그래프에서 속성 정보를 마스킹하여 학습하는 방식.
   - **ContextPred**: 화학 구조의 문맥적 패턴을 학습하는 GNN 모델.
   - **MolCLR**: 대조 학습(Contrastive Learning)을 활용한 분자 그래프 학습 모델.
   - **GraphMVP**: 2D 및 3D 분자 구조를 함께 학습한 GNN 기반 모델.

3. **대조 학습 기반 멀티모달 모델**
   - **Frozen (사전학습된 단일 모달 인코더 사용)**: MoleculeSTM과 동일한 구조를 사용하지만, 인코더를 고정하고 학습을 진행하지 않음.
   - **Similarity (단일 모달 기반 검색 방식)**: 화학 구조 또는 텍스트 하나만을 기준으로 유사도를 측정하여 검색하는 방식.
   - **KV-PLM**: 기존 연구에서 제안한 SMILES-텍스트 결합 모델.

---

#### **2. 테스트 데이터셋 (Test Datasets)**
MoleculeSTM의 성능을 검증하기 위해 사용된 테스트 데이터셋은 다음과 같다.

1. **PubChemSTM**: 28만 개 이상의 화학 구조-텍스트 쌍을 포함하는 대규모 멀티모달 데이터셋.
2. **DrugBank**: 신약 개발 및 약물 특성 분석을 위한 벤치마크 데이터셋.
   - **DrugBank-Description**: 약물 설명 기반 검색 평가.
   - **DrugBank-Pharmacodynamics**: 약리학적 특성 검색 평가.
   - **DrugBank-ATC**: 해부학적 치료 화학(ATC) 코드 기반 검색 평가.
3. **MoleculeNet**: 분자 특성 예측을 위한 데이터셋.
   - **BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV, BACE** 등 8개 단일 분자 특성 예측 데이터셋 사용.
4. **ZINC**: 분자 편집 실험을 위해 사용된 데이터셋.

---

#### **3. 평가 메트릭 (Evaluation Metrics)**
MoleculeSTM의 성능을 정량적으로 평가하기 위해 사용된 주요 평가 지표는 다음과 같다.

1. **제로샷 구조-텍스트 검색 (Zero-shot Structure-Text Retrieval)**
   - **정확도(Accuracy)**: 주어진 화학 구조(또는 텍스트)에 대해 가장 적절한 텍스트(또는 화학 구조)를 검색할 수 있는지를 평가.
   - **유사도 점수(Similarity Score)**: 분자 구조와 텍스트 설명 간의 임베딩 공간에서의 유사도를 측정.

2. **제로샷 분자 편집 (Zero-shot Molecule Editing)**
   - **Satisfactory Hit Ratio**: 모델이 주어진 텍스트 지침을 바탕으로 원하는 화학적 속성을 가진 분자로 변환할 수 있는지 평가.
   - **구조적 유사도(Tanimoto Similarity)**: 원본 분자와 생성된 분자 간의 구조적 유사성을 측정.

3. **분자 특성 예측 (Molecular Property Prediction)**
   - **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: 분자 특성을 예측하는 모델의 성능을 측정하는 주요 지표.
   - **정확도(Accuracy)**: 예측된 분자 특성이 실제 특성과 일치하는 비율을 평가.

---



---

#### **1. Baseline & Competitive Models**
To assess the effectiveness of MoleculeSTM, comparisons were made with the following models:

1. **SMILES-based Models (Single-Modality)**
   - **MegaMolBART**: A BART-based model pretrained on the **ZINC** database.
   - **KV-PLM**: A pretrained model that jointly learns **SMILES representations** and textual descriptions.

2. **Graph-based Models (Single-Modality)**
   - **AttrMasking**: A graph-based molecular representation method that masks molecular attributes.
   - **ContextPred**: A GNN-based model that learns contextual molecular patterns.
   - **MolCLR**: A contrastive learning-based molecular representation model.
   - **GraphMVP**: A model incorporating both 2D and 3D molecular structures using GNNs.

3. **Contrastive Learning-Based Multi-Modal Models**
   - **Frozen (Pretrained Single-Modality Encoders Fixed)**: Uses the same structure as MoleculeSTM but keeps encoders frozen.
   - **Similarity (Single-Modality Search-Based Baseline)**: Uses either molecular structure or textual information independently for retrieval.
   - **KV-PLM**: An existing SMILES-text model used for molecular representation.

---

#### **2. Test Datasets**
The performance of MoleculeSTM was evaluated using the following test datasets:

1. **PubChemSTM**: A large-scale multi-modal dataset with **280,000+** chemical structure-text pairs.
2. **DrugBank**: A benchmark dataset for drug discovery and molecular property analysis.
   - **DrugBank-Description**: Evaluates retrieval performance based on drug descriptions.
   - **DrugBank-Pharmacodynamics**: Evaluates retrieval performance based on pharmacodynamics properties.
   - **DrugBank-ATC**: Evaluates retrieval performance based on **Anatomical Therapeutic Chemical (ATC) codes**.
3. **MoleculeNet**: A benchmark dataset for molecular property prediction.
   - Includes **BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV, BACE** datasets for single-molecule property prediction tasks.
4. **ZINC**: Used for molecular editing experiments.

---

#### **3. Evaluation Metrics**
The following key metrics were used to quantitatively evaluate MoleculeSTM:

1. **Zero-shot Structure-Text Retrieval**
   - **Accuracy**: Measures the ability of the model to retrieve the correct textual description (or molecular structure) given a query.
   - **Similarity Score**: Computes the embedding-space similarity between molecular structures and textual descriptions.

2. **Zero-shot Molecule Editing**
   - **Satisfactory Hit Ratio**: Evaluates whether the model can modify molecular structures to match a given text-based instruction.
   - **Tanimoto Similarity**: Measures the structural similarity between the original molecule and the generated molecule.

3. **Molecular Property Prediction**
   - **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: A key metric for assessing molecular property prediction performance.
   - **Accuracy**: Measures the proportion of correctly predicted molecular properties.

These evaluations confirm that **MoleculeSTM outperforms existing SOTA models**, achieving **state-of-the-art generalization** for zero-shot tasks, molecular editing, and molecular property predictions.






<br/>
# 예제  



#### **1. 트레이닝 데이터 (Training Data)**
- **PubChemSTM**: PubChem에서 수집된 **28만 개 이상의 화학 구조-텍스트 쌍** 포함.
  - 예시:  
    - **SMILES:** `CCOC(=O)c1ccc(cc1)N`
    - **설명:** "에스터 결합을 포함하는 방향족 화합물"

---

#### **2. 테스트 데이터 (Test Data)**
- **DrugBank**: 약물 설명과 약리학적 특성을 포함한 데이터.
- **MoleculeNet**: 분자 특성 예측 데이터 (BBBP, Tox21 등).
- **ZINC**: 분자 편집 실험을 위한 데이터.

---

#### **3. 수행한 태스크 예시 (Example Tasks)**
1. **구조-텍스트 검색 (Structure-Text Retrieval)**
   - **입력:** "항생제로 사용되는 분자 검색"
   - **출력:** 해당 설명과 일치하는 화학 구조 반환.

2. **분자 편집 (Molecule Editing)**
   - **입력:** "이 화합물의 용해도를 증가시키는 방향으로 변경"
   - **출력:** 용해도가 높은 변형된 분자 구조 생성.

---


#### **1. Training Data**
- **PubChemSTM**: A dataset with **280,000+ structure-text pairs** from PubChem.
  - Example:  
    - **SMILES:** `CCOC(=O)c1ccc(cc1)N`
    - **Description:** "An aromatic compound containing an ester bond."

---

#### **2. Test Data**
- **DrugBank**: Contains drug descriptions and pharmacodynamic properties.
- **MoleculeNet**: Used for molecular property prediction (e.g., BBBP, Tox21).
- **ZINC**: Used for molecule editing experiments.

---

#### **3. Example Tasks**
1. **Structure-Text Retrieval**
   - **Input:** "Find molecules used as antibiotics."
   - **Output:** Returns chemical structures matching this description.

2. **Molecule Editing**
   - **Input:** "Modify this compound to increase solubility."
   - **Output:** Generates a modified molecular structure with higher solubility.



<br/>  
# 요약   



이 연구에서는 화학 구조와 텍스트 설명을 함께 학습하는 **MoleculeSTM** 모델을 제안하고, PubChemSTM 데이터셋(28만 개의 구조-텍스트 쌍)을 사용하여 대조 학습을 수행하였다. 모델의 성능을 평가하기 위해 DrugBank, MoleculeNet, ZINC 데이터셋을 활용하여 **구조-텍스트 검색**, **분자 편집**, **분자 특성 예측** 등의 태스크를 수행하였으며, 기존 SOTA 모델보다 뛰어난 일반화 성능을 보였다. 예를 들어, "이 화합물의 용해도를 증가시키는 방향으로 변경"과 같은 명령을 입력하면, MoleculeSTM은 용해도가 높은 변형된 분자 구조를 생성할 수 있었다.  

---


This study proposes the **MoleculeSTM** model, which jointly learns chemical structures and textual descriptions through contrastive learning, using the **PubChemSTM dataset** (280,000 structure-text pairs). The model was evaluated on **DrugBank, MoleculeNet, and ZINC datasets**, performing **structure-text retrieval, molecule editing, and molecular property prediction**, achieving superior generalization over existing SOTA models. For example, when given the instruction **"Modify this compound to increase solubility,"** MoleculeSTM successfully generated a modified molecular structure with higher solubility.


<br/>  
# 기타  




#### **1. 피규어 (Figures)**
- **Figure 2**:  
  - **(a) 제로샷 구조-텍스트 검색 결과**: DrugBank 데이터셋(Description, Pharmacodynamics, ATC)에서 다양한 모델들의 검색 정확도를 비교함. MoleculeSTM은 기존 모델 대비 50%, 40%, 15% 향상된 정확도를 보임.  
  - **(b) ATC 검색 케이스 스터디**: 분자의 화학 구조를 기반으로 가장 유사한 ATC(해부학적 치료 화학) 라벨 10개를 검색한 결과를 시각화함.  

- **Figure 3**:  
  - **(a) 공간 정렬 (Space Alignment)**: 미리 학습된 분자 생성 모델과 MoleculeSTM의 표현 공간을 정렬하는 과정.  
  - **(b) 잠재 최적화 (Latent Optimization)**: 입력 분자와 텍스트 설명이 모두 반영된 최적의 잠재 표현을 학습하는 과정.  

- **Figure 4**:  
  - **제로샷 텍스트 기반 분자 편집 결과**:  
    - 8가지 단일 속성 편집,  
    - 4가지 다중 속성 편집,  
    - 4가지 ChEMBL 바인딩 친화도 기반 편집,  
    - 4가지 약물 유사성 편집을 수행한 결과를 시각화함.  
  - MoleculeSTM이 경쟁 모델들보다 높은 히트 비율을 달성함.  

---

#### **2. 테이블 (Tables)**
- **Table 8~10**:  
  - DrugBank 데이터셋(Description, Pharmacodynamics, ATC)에서 **제로샷 검색 정확도(%)** 비교.  
  - MoleculeSTM이 모든 데이터셋에서 기존 모델을 크게 상회하는 성능을 보임.  

- **Table 17~20**:  
  - **텍스트 기반 분자 편집 결과 시각화**  
  - 수소 결합 수용체(HBA) 및 공여체(HBD) 변화, 용해도(LogP), 투과도(tPSA) 등을 조정하는 실험 결과 포함.  
  - 특정 속성(예: "이 분자는 물에 잘 녹는다")을 증가시키는 방식으로 변형된 분자 구조를 보여줌.  

- **Table 25**:  
  - MoleculeNet에서 사용된 **분자 특성 예측 데이터셋 요약**  
  - BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV, BACE 등의 데이터셋 포함.  

---

#### **3. 어펜딕스 (Appendix)**
- **Appendix D.2**:  
  - **단일 속성 편집 (Single-objective Molecule Editing)**  
  - 용해도(LogP), 약물 유사성(QED), 투과성(tPSA), 수소 결합 공여체/수용체(HBD/HBA) 조정 실험 설명.  

- **Appendix D.3**:  
  - **다중 속성 편집 (Multi-objective Editing)**  
  - 두 가지 이상의 속성을 동시에 최적화하는 실험 (예: "이 분자는 물에 잘 녹고, 높은 투과성을 가짐").  

- **Appendix D.4**:  
  - **바인딩 친화도 기반 편집 (Binding-affinity Editing)**  
  - 특정 단백질과의 결합 친화도를 개선하는 방식으로 분자를 수정하는 실험 포함.  

- **Appendix D.5**:  
  - **약물 유사성 편집 (Drug Relevance Editing)**  
  - 특정한 기존 약물(예: Penicillin)과 유사한 구조를 갖도록 분자를 변형하는 실험.  

---


#### **1. Figures**  
- **Figure 2**:  
  - **(a) Zero-shot structure-text retrieval accuracy** on DrugBank datasets (Description, Pharmacodynamics, ATC). MoleculeSTM outperforms previous models by **50%, 40%, and 15%** in accuracy.  
  - **(b) Case study on ATC retrieval**, showing the top 10 most similar ATC labels retrieved for a given molecular structure.  

- **Figure 3**:  
  - **(a) Space Alignment**: Aligns the representation space of a pretrained molecule generation model with the MoleculeSTM representation space.  
  - **(b) Latent Optimization**: Optimizes a latent representation that integrates both input molecules and textual descriptions.  

- **Figure 4**:  
  - **Visualization of zero-shot text-based molecule editing**, covering:  
    - **8 single-objective edits**,  
    - **4 multi-objective edits**,  
    - **4 ChEMBL binding-affinity-based edits**,  
    - **4 drug relevance edits**.  
  - MoleculeSTM achieves **higher hit ratios** than competing models.  

---

#### **2. Tables**  
- **Tables 8–10**:  
  - **Zero-shot retrieval accuracy (%) on DrugBank datasets** (Description, Pharmacodynamics, ATC).  
  - MoleculeSTM significantly outperforms existing models.  

- **Tables 17–20**:  
  - **Visualization of text-based molecule editing results**, including:  
    - Hydrogen bond acceptors (HBA) and donors (HBD),  
    - Solubility (LogP),  
    - Permeability (tPSA).  
  - Modifications are guided by prompts such as **"This molecule is soluble in water."**  

- **Table 25**:  
  - **Summary of molecular property prediction datasets** from MoleculeNet.  
  - Includes BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV, and BACE datasets.  

---

#### **3. Appendix**  
- **Appendix D.2**:  
  - **Single-objective molecule editing**, optimizing solubility (LogP), drug-likeness (QED), permeability (tPSA), and hydrogen bond acceptors/donors (HBA/HBD).  

- **Appendix D.3**:  
  - **Multi-objective molecule editing**, optimizing multiple properties simultaneously (e.g., **"This molecule is soluble in water and has high permeability."**).  

- **Appendix D.4**:  
  - **Binding-affinity-based molecule editing**, improving molecular binding affinity to specific proteins.  

- **Appendix D.5**:  
  - **Drug relevance editing**, modifying molecules to resemble existing drugs (e.g., **"This molecule looks like Penicillin."**).




<br/>
# refer format:     



@article{MoleculeSTM2022,
  author    = {Shuo Sun and Chengzhi Wu and Jiaqi Wang and Xinzhe Li and Yue Yu and Siyu Wang and Sheng Wang and Jimeng Sun and Jian Tang and Yue Gao},
  title     = {MoleculeSTM: A Unified Self-Supervised Learning Framework for Molecules via Translation-Based Modal Augmentation},
  journal   = {arXiv preprint},
  year      = {2022},
  volume    = {},
  number    = {},
  pages     = {},
  eprint    = {2212.10789},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url       = {https://arxiv.org/abs/2212.10789}
}



Sun, Shuo, Chengzhi Wu, Jiaqi Wang, Xinzhe Li, Yue Yu, Siyu Wang, Sheng Wang, Jimeng Sun, Jian Tang, and Yue Gao. "MoleculeSTM: A Unified Self-Supervised Learning Framework for Molecules via Translation-Based Modal Augmentation." arXiv preprint (2022). https://arxiv.org/abs/2212.10789.












