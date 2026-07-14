---
layout: post
title:  "[2026]FACT: Functional Group Alignment and Consistency in Token Space for Structure-aware Molecular Representation Learning"
date:   2026-07-14 00:51:35 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 FACT(Functional Group Alignment and Consistency in Token Space)라는 구조 인식 SMILES 기반 분자 표현 학습 프레임워크를 제안하며, 이는 원자-토큰 정렬 모듈을 통해 기능 그룹(FG)의 정확한 범위를 식별하고, 다양한 SMILES 표현 간의 FG 일관성을 유지하는 손실 함수를 도입합니다.


짧은 요약(Abstract) :


이 논문의 초록에서는 분자 표현 학습이 다양한 하위 작업, 특히 분자의 물리화학적 및 생물학적 특성을 정확하게 예측하는 데 중요한 역할을 한다고 설명합니다. 그러나 SMILES 기반 모델에 기능 그룹(FG) 정보를 통합하는 것은 여전히 도전 과제가 있습니다. 그래프 정의 FG 원자 집합과 시퀀스의 토큰 간의 명시적 정렬이 없기 때문에 완전한 하위 구조 마스킹이 불가능하며, 동일한 분자의 여러 유효한 SMILES 형태는 토큰 공간에서 일관되지 않은 FG 표현을 초래합니다. 이러한 문제를 해결하기 위해, 저자들은 FACT(Functional Group Alignment and Consistency in Token Space)라는 구조 인식 SMILES 기반 표현 학습을 위한 엔드 투 엔드 프레임워크를 제안합니다. FACT는 사전 훈련 중 FG 범위 마스킹을 완전하게 수행하기 위한 원자-토큰 정렬 모듈을 도입하고, 미세 조정 중에는 서로 다른 SMILES 형태 간의 FG 일관성을 강제합니다. MoleculeNet 벤치마크에서의 실험 결과, FACT는 8개의 작업에서 최첨단 또는 경쟁력 있는 성능을 달성하여 분자 표현을 위한 정렬 및 일관성 학습의 효과를 입증합니다.



The abstract of this paper describes that molecular representation learning plays a crucial role in various downstream tasks, particularly in accurately predicting the physicochemical and biological properties of molecules. However, incorporating functional group (FG) information into SMILES-based models remains a challenge. The absence of explicit alignment between graph-defined FG atom sets and tokens in the sequence prevents complete substructure masking, while multiple valid SMILES forms of the same molecule lead to inconsistent FG representations in token space. To address these issues, the authors propose FACT (Functional Group Alignment and Consistency in Token Space), an end-to-end framework for structure-aware SMILES-based representation learning. FACT introduces an atom-token alignment module for complete FG span masking during pre-training and enforces FG consistency across different SMILES forms during fine-tuning. Experiments on MoleculeNet benchmarks show that FACT achieves state-of-the-art or competitive performance on eight tasks, demonstrating the effectiveness of alignment and consistency learning for molecular representation.


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


FACT(Functional Group Alignment and Consistency in Token Space)는 SMILES 기반의 분자 표현 학습을 위한 종단 간(end-to-end) 프레임워크로, 분자의 화학적 구조를 효과적으로 캡처하기 위해 설계되었습니다. 이 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 기능 그룹(FG) 탐지 및 원자-토큰 정렬, FG 인식 사전 훈련, FG 보존 일관성 학습입니다.

1. **FG 탐지 및 원자-토큰 정렬**: 
   - FACT는 먼저 분자의 기능 그룹을 식별하고, 이를 SMILES 문자열의 원자 토큰과 정렬합니다. 기존의 방법들이 미리 정의된 SMARTS 패턴에 의존하는 것과 달리, FACT는 Ertl 알고리즘을 사용하여 패턴 없는 기능 그룹 탐지를 수행합니다. 이 과정에서 분자의 원자와 SMILES 문자열의 원자 토큰 간의 일대일 대응 관계를 설정하여, 기능 그룹의 정확한 토큰 범위를 식별할 수 있습니다.

2. **FG 인식 사전 훈련**: 
   - 정렬된 기능 그룹 범위를 활용하여, 모델은 완전한 FG 범위 마스킹을 통해 훈련됩니다. 그러나 FG 마스킹만으로는 충분하지 않기 때문에, FACT는 완전 FG 범위 마스킹과 추가적인 무작위 토큰 마스킹을 결합한 하이브리드 마스킹 전략을 제안합니다. 이 방법은 기능 그룹이 완전한 하위 구조로 마스킹되도록 하면서, 나머지 컨텍스트 토큰도 무작위로 마스킹하여 훈련 신호를 증가시킵니다.

3. **FG 보존 일관성 학습**: 
   - 사전 훈련된 인코더는 FG 보존 일관성 목표를 사용하여 다운스트림 분자 속성 예측(MPP) 작업에 대해 미세 조정됩니다. 동일한 분자의 원래 SMILES와 무작위 SMILES 변형을 인코딩하여, 서로 다른 SMILES 표현 간의 기능 그룹 표현의 일관성을 강화합니다. 이를 통해 모델은 동일한 분자의 다양한 SMILES 표현에서 기능 그룹의 일관된 표현을 학습할 수 있습니다.

이러한 방법론을 통해 FACT는 MoleculeNet 벤치마크에서 최첨단 성능을 달성하며, 화학적으로 의미 있는 하위 구조 표현을 학습하는 데 효과적임을 입증하였습니다.

---


FACT (Functional Group Alignment and Consistency in Token Space) is an end-to-end framework designed for SMILES-based molecular representation learning, aimed at effectively capturing the chemical structure of molecules. This framework consists of three main components: Functional Group (FG) Detection and Atom-Token Alignment, FG-Aware Pre-training, and FG-Preserving Consistency Learning.

1. **FG Detection and Atom-Token Alignment**: 
   - FACT first identifies functional groups in a molecule and aligns them with atom tokens in the SMILES string. Unlike existing methods that rely on predefined SMARTS patterns, FACT employs the Ertl algorithm for pattern-free functional group detection. This process establishes a one-to-one correspondence between graph atoms and atom tokens in the SMILES sequence, allowing for precise identification of the token spans corresponding to functional groups.

2. **FG-Aware Pre-training**: 
   - Utilizing the aligned functional group spans, the model is trained with complete FG span masking. However, FG masking alone is insufficient, so FACT proposes a hybrid masking strategy that combines complete FG span masking with additional random token masking. This approach ensures that functional groups are masked as complete substructures while also covering the remaining context tokens through random masking, thereby increasing the training signal.

3. **FG-Preserving Consistency Learning**: 
   - The pretrained encoder is fine-tuned on downstream molecular property prediction (MPP) tasks using an FG-preserving consistency objective. By encoding both the original SMILES and a randomized SMILES variant of the same molecule, the model reinforces the consistency of functional group representations across different SMILES forms. This allows the model to learn invariant representations of functional groups across various representations of the same molecule.

Through these methodologies, FACT achieves state-of-the-art performance on MoleculeNet benchmarks, demonstrating its effectiveness in learning chemically meaningful substructure representations.


<br/>
# Results


FACT 프레임워크는 MoleculeNet 벤치마크에서 여러 분자 속성 예측 작업에 대해 경쟁 모델들과 비교하여 우수한 성능을 보였습니다. 다음은 주요 결과에 대한 요약입니다.

1. **경쟁 모델**: FACT는 다양한 모델과 비교되었습니다. 여기에는 RoBERTa, MoLFormer, BROBERG, FG-BERT, MLM-FG 등이 포함됩니다. 이들 모델은 각각 다른 방식으로 분자 표현을 학습하고, 기능적 그룹(FG) 정보를 활용하는 방법이 다릅니다.

2. **테스트 데이터**: MoleculeNet 벤치마크는 여러 분자 속성 예측 작업을 포함하고 있으며, 이 작업들은 이진 분류, 다중 레이블 분류 및 회귀 작업으로 나뉩니다. 각 작업은 분자의 화학적 특성을 예측하는 데 중점을 두고 있습니다.

3. **메트릭**: 성능 평가는 ROC-AUC (이진 분류 작업)와 RMSE (회귀 작업)로 측정되었습니다. ROC-AUC는 모델의 분류 성능을 나타내며, RMSE는 예측 값과 실제 값 간의 차이를 측정합니다.

4. **비교 결과**:
   - **이진 분류 작업**: FACT는 BACE, BBBP, ClinTox, Tox21, SIDER, HIV, MUV와 같은 여러 이진 분류 작업에서 최첨단 성능을 달성했습니다. 예를 들어, BACE 데이터셋에서 FACT는 0.931의 ROC-AUC를 기록하여 가장 높은 성능을 보였습니다.
   - **회귀 작업**: FACT는 FreeSolv 데이터셋에서 0.964의 RMSE를 기록하여 최상의 성능을 달성했습니다. 그러나 ESOL과 Lipophilicity 데이터셋에서는 다른 모델에 비해 상대적으로 낮은 성능을 보였습니다. 이는 이들 작업이 분자의 전반적인 특성에 더 의존하기 때문으로 분석되었습니다.

5. **결론**: FACT 프레임워크는 기능적 그룹 정렬 및 일관성 학습을 통해 분자 표현에서 화학적으로 의미 있는 하위 구조를 학습하는 데 효과적임을 입증했습니다. 실험 결과는 FACT가 여러 작업에서 최첨단 성능을 달성했음을 보여주며, 이는 분자 속성 예측에서의 유용성을 강조합니다.

---



The FACT framework demonstrated superior performance compared to various competing models on the MoleculeNet benchmark across multiple molecular property prediction tasks. Here is a structured summary of the key results:

1. **Competing Models**: FACT was compared against several models, including RoBERTa, MoLFormer, BROBERG, FG-BERT, and MLM-FG. Each of these models employs different strategies for learning molecular representations and utilizing functional group (FG) information.

2. **Test Data**: The MoleculeNet benchmark includes a variety of molecular property prediction tasks, which are categorized into binary classification, multi-label classification, and regression tasks. Each task focuses on predicting the chemical properties of molecules.

3. **Metrics**: Performance was evaluated using ROC-AUC (for binary classification tasks) and RMSE (for regression tasks). ROC-AUC indicates the classification performance of the model, while RMSE measures the difference between predicted and actual values.

4. **Comparison Results**:
   - **Binary Classification Tasks**: FACT achieved state-of-the-art performance on several binary classification tasks, including BACE, BBBP, ClinTox, Tox21, SIDER, HIV, and MUV. For instance, on the BACE dataset, FACT recorded a ROC-AUC of 0.931, the highest among all models.
   - **Regression Tasks**: FACT achieved the best performance on the FreeSolv dataset with an RMSE of 0.964. However, it showed relatively lower performance on the ESOL and Lipophilicity datasets compared to other models. This was attributed to the nature of these tasks being more influenced by global molecular properties.

5. **Conclusion**: The FACT framework effectively learns chemically meaningful substructure representations through functional group alignment and consistency learning. The experimental results demonstrate that FACT achieves state-of-the-art performance across multiple tasks, highlighting its utility in molecular property prediction.


<br/>
# 예제


이 논문에서는 FACT(Functional Group Alignment and Consistency in Token Space)라는 새로운 프레임워크를 제안하여 SMILES 기반의 분자 표현 학습을 개선하고자 합니다. 이 프레임워크는 두 가지 주요 문제를 해결합니다: 첫째, SMILES 토큰 공간에서의 기능 그룹(Functional Group, FG) 정렬 부족, 둘째, 동일한 분자의 여러 SMILES 표현 간의 FG 일관성 부족입니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **데이터셋**: PubChem 데이터베이스에서 무작위로 샘플링한 1,000만 개의 SMILES 문자열.
   - **입력**: 각 SMILES 문자열은 분자의 구조를 나타내는 문자열로, 예를 들어 "CCO"는 에탄올을 나타냅니다.
   - **출력**: 각 SMILES 문자열에 대해 모델은 분자의 물리화학적 및 생물학적 속성을 예측합니다. 예를 들어, 에탄올의 경우, 물질의 용해도, 독성, 생물학적 활성 등을 예측할 수 있습니다.

2. **테스트 데이터**:
   - **데이터셋**: MoleculeNet 벤치마크에서 제공하는 다양한 분자 속성 예측 데이터셋.
   - **입력**: 테스트 데이터는 분자의 SMILES 표현으로 구성되며, 예를 들어 "CC(=O)O"는 아세트산을 나타냅니다.
   - **출력**: 모델은 각 SMILES 문자열에 대해 이 분자의 특정 속성(예: 독성, 생물학적 활성 등)을 예측합니다. 예를 들어, 아세트산의 경우, 모델은 이 물질이 특정 생물학적 타겟에 대해 얼마나 활성인지 예측할 수 있습니다.

#### 구체적인 테스크

- **분자 속성 예측(Molecular Property Prediction, MPP)**: 이 테스크는 주어진 SMILES 문자열에 대해 분자의 다양한 물리화학적 및 생물학적 속성을 예측하는 것입니다. 예를 들어, BACE-1 억제제인지 여부를 예측하는 이진 분류 문제, 특정 물질의 용해도를 예측하는 회귀 문제 등이 포함됩니다.

이러한 방식으로 FACT 프레임워크는 SMILES 기반의 분자 표현 학습에서 기능 그룹의 정렬과 일관성을 강화하여, 더 정확한 분자 속성 예측을 가능하게 합니다.

---




In this paper, we propose a new framework called FACT (Functional Group Alignment and Consistency in Token Space) to improve SMILES-based molecular representation learning. This framework addresses two main issues: first, the lack of functional group (FG) alignment in the SMILES token space, and second, the absence of FG consistency across multiple SMILES representations of the same molecule.

#### Training Data and Test Data

1. **Training Data**:
   - **Dataset**: 10 million SMILES strings randomly sampled from the PubChem database.
   - **Input**: Each SMILES string represents the structure of a molecule; for example, "CCO" represents ethanol.
   - **Output**: For each SMILES string, the model predicts the physicochemical and biological properties of the molecule. For instance, for ethanol, it could predict properties like solubility, toxicity, and biological activity.

2. **Test Data**:
   - **Dataset**: Various molecular property prediction datasets provided by the MoleculeNet benchmark.
   - **Input**: The test data consists of SMILES representations of molecules, such as "CC(=O)O" representing acetic acid.
   - **Output**: The model predicts specific properties of each SMILES string, such as toxicity or biological activity. For example, for acetic acid, the model might predict how active this substance is against a specific biological target.

#### Specific Tasks

- **Molecular Property Prediction (MPP)**: This task involves predicting various physicochemical and biological properties of a molecule given its SMILES string. This includes binary classification problems, such as predicting whether a compound is a BACE-1 inhibitor, and regression problems, such as predicting the solubility of a specific substance.

In this way, the FACT framework enhances the alignment and consistency of functional groups in SMILES-based molecular representation learning, enabling more accurate predictions of molecular properties.

<br/>
# 요약


이 논문에서는 FACT(Functional Group Alignment and Consistency in Token Space)라는 구조 인식 SMILES 기반 분자 표현 학습 프레임워크를 제안하며, 이는 원자-토큰 정렬 모듈을 통해 기능 그룹(FG)의 정확한 범위를 식별하고, 다양한 SMILES 표현 간의 FG 일관성을 유지하는 손실 함수를 도입합니다. 실험 결과, FACT는 MoleculeNet 벤치마크에서 여러 분자 속성 예측 작업에서 최첨단 성능을 달성하였으며, FG 관련 구조를 효과적으로 학습하는 것으로 나타났습니다. 이 연구는 FG의 정렬과 일관성 학습이 분자 표현의 품질을 향상시킬 수 있음을 보여줍니다.

---

This paper proposes FACT (Functional Group Alignment and Consistency in Token Space), a structure-aware SMILES-based molecular representation learning framework that introduces an atom-token alignment module to accurately identify functional group (FG) spans and enforces FG consistency across different SMILES representations through a consistency loss. Experimental results show that FACT achieves state-of-the-art performance on various molecular property prediction tasks in the MoleculeNet benchmark, effectively learning FG-related structures. This study demonstrates that alignment and consistency learning of FGs can enhance the quality of molecular representations.

<br/>
# 기타




1. **다이어그램 및 피규어**
   - **Figure 1**: FACT 프레임워크의 개요를 보여줍니다. 이 다이어그램은 기능 그룹(FG) 감지 및 원자-토큰 정렬, FG-aware 사전 훈련, FG 보존 일관성 학습의 세 가지 주요 구성 요소를 설명합니다. 이 구조는 FG의 정확한 식별과 일관성을 통해 화학적으로 의미 있는 분자 표현을 학습하는 데 기여합니다.
   - **Figure 2**: UMAP을 사용하여 SMILES 임베딩을 시각화한 결과입니다. FACT와 그 변형들이 MLM-FG보다 더 조직적인 패턴을 보이며, FG 관련 구조를 더 잘 포착하고 있음을 나타냅니다. 특히, FACT의 전체 구성은 가장 집중된 분포를 보여줍니다.
   - **Figure 3**: BACE 데이터셋에서의 모델 예측 시나리오에 대한 기여 분석을 보여줍니다. FACT는 화학적으로 관련된 하위 구조에 더 강한 정렬을 보이며, 잘못된 예측에서도 여전히 관련 하위 구조에 기여를 할당합니다.

2. **테이블**
   - **Table 1**: 사전 훈련에 사용된 10M 분자의 통계 요약입니다. 기능 그룹(FG) 수, FG당 원자 수, SMILES 토큰 길이 등의 정보를 제공합니다. 이 데이터는 FG 기반 마스킹의 자연스러운 범위와 모델 훈련의 맥락을 이해하는 데 도움이 됩니다.
   - **Table 2**: MoleculeNet 분류 벤치마크에서의 성능 비교입니다. FACT는 7개의 분류 작업 중 5개에서 최첨단 성능을 달성하며, FG 보존 일관성 학습이 하위 구조 수준의 표현에서 일관된 개선을 제공함을 보여줍니다.
   - **Table 3**: MoleculeNet 회귀 벤치마크에서의 성능 비교입니다. FACT는 FreeSolv에서 최첨단 성능을 달성하지만, ESOL 및 Lipophilicity에서는 다른 모델에 비해 낮은 성능을 보입니다. 이는 이러한 작업이 분자의 전반적인 특성에 더 영향을 받기 때문으로 해석됩니다.

3. **어펜딕스**
   - 어펜딕스는 실험 설정, 데이터셋, 하이퍼파라미터 및 추가적인 실험 결과를 포함할 수 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 FACT 프레임워크를 기반으로 추가 연구를 수행할 수 있도록 돕습니다.

---



1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates the overview of the FACT framework. It describes the three main components: functional group (FG) detection and atom-token alignment, FG-aware pre-training, and FG-preserving consistency learning. This structure contributes to learning chemically meaningful molecular representations through accurate identification and consistency of FGs.
   - **Figure 2**: This shows the UMAP visualization of SMILES embeddings. FACT and its variants exhibit more organized patterns compared to MLM-FG, indicating that they better capture FG-related structures. Notably, the full configuration of FACT shows the most concentrated distribution.
   - **Figure 3**: This figure presents the attribution analysis for prediction scenarios on the BACE dataset. FACT demonstrates stronger alignment with chemically relevant substructures and assigns attribution to relevant substructures even in incorrect predictions.

2. **Tables**
   - **Table 1**: A summary of statistics for the 10M molecules used for pre-training. It provides information on the number of functional groups (FGs), the number of atoms per FG, and the length of SMILES tokens. This data helps understand the natural range of FG-based masking and the context for model training.
   - **Table 2**: Performance comparison on MoleculeNet classification benchmarks. FACT achieves state-of-the-art performance on five out of seven classification tasks, demonstrating that FG-preserving consistency learning provides consistent improvements in substructure-level representations.
   - **Table 3**: Performance comparison on MoleculeNet regression benchmarks. FACT achieves state-of-the-art performance on FreeSolv but shows lower performance on ESOL and Lipophilicity compared to other models. This is interpreted as these tasks being more influenced by global molecular properties.

3. **Appendix**
   - The appendix may include experimental setups, datasets, hyperparameters, and additional experimental results. This enhances the reproducibility of the research and helps other researchers build upon the FACT framework for further studies.

<br/>
# refer format:


### BibTeX 형식

```bibtex
@inproceedings{Nam2026,
  author    = {Hyeonyeong Nam and Woojae Choi and Deok-Joong Lee and Young-Han Son and Sangwoon Lee and Bogyeong Kang and Eunjung Jo and Tae-Eui Kam},
  title     = {FACT: Functional Group Alignment and Consistency in Token Space for Structure-aware Molecular Representation Learning},
  booktitle = {Proceedings of the 25th Workshop on Biomedical Language Processing (BioNLP 2026)},
  pages     = {695--703},
  year      = {2026},
  month     = {July},
  publisher  = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Hyeonyeong Nam, Woojae Choi, Deok-Joong Lee, Young-Han Son, Sangwoon Lee, Bogyeong Kang, Eunjung Jo, and Tae-Eui Kam. "FACT: Functional Group Alignment and Consistency in Token Space for Structure-aware Molecular Representation Learning." In *Proceedings of the 25th Workshop on Biomedical Language Processing (BioNLP 2026)*, 695–703. Association for Computational Linguistics, July 3-4, 2026.
    