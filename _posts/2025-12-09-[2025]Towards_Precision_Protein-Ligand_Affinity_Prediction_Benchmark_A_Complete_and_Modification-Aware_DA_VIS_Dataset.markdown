---
layout: post
title:  "[2025]Towards Precision Protein-Ligand Affinity Prediction Benchmark: A Complete and Modification-Aware DA VIS Dataset"
date:   2025-12-09 20:52:58 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 DA VIS 데이터셋을 수정된 단백질-리간드 쌍을 포함하여 보완하고, 세 가지 벤치마크 설정을 통해 모델의 일반화 능력을 평가하였다.


짧은 요약(Abstract) :


이 논문의 초록에서는 인공지능(AI) 기술의 발전이 단백질-리간드 결합 친화도 예측과 같은 중요한 약물 발견 작업에 대한 새로운 가능성을 열어주고 있음을 강조합니다. 그러나 현재의 모델들은 생물학적으로 관련성이 있는 단백질 변형을 반영하지 않는 단순화된 데이터셋에 과적합(overfitting)되고 있다는 문제를 지적합니다. 이 연구에서는 4,032개의 키나제-리간드 쌍을 포함하여 변형을 고려한 DA VIS 데이터셋의 완전하고 수정된 버전을 구축하였습니다. 이 데이터셋은 생물학적으로 현실적인 조건에서 예측 모델의 성능을 평가할 수 있는 기준을 제공합니다. 연구진은 이 새로운 데이터셋을 기반으로 세 가지 벤치마크 설정을 제안하며, 이를 통해 모델의 변형에 대한 강건성을 평가합니다. 평가 결과, 도킹 기반 모델이 제로샷(zero-shot) 설정에서 더 잘 일반화되는 반면, 도킹 없는 모델은 야생형(wild-type) 단백질에 과적합되는 경향이 있음을 발견했습니다. 이 연구는 단백질 변형에 더 잘 일반화되는 모델 개발을 위한 귀중한 기초를 제공할 것으로 기대합니다.



The abstract of this paper emphasizes that advancements in artificial intelligence (AI) technology open new possibilities for critical drug discovery tasks, such as protein-ligand binding affinity prediction. However, it points out that current models overfit to simplified datasets that do not reflect biologically relevant protein modifications. In this study, the authors curate a complete and modification-aware version of the DA VIS dataset, incorporating 4,032 kinase-ligand pairs. This enriched dataset provides a benchmark for evaluating predictive models under biologically realistic conditions. The authors propose three benchmark settings based on this new dataset to assess model robustness in the presence of protein modifications. Evaluation results indicate that docking-based models generalize better in zero-shot settings, while docking-free models tend to overfit to wild-type proteins. This study anticipates providing a valuable foundation for developing models that better generalize to protein modifications, ultimately advancing precision medicine in drug discovery.


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



이 논문에서는 단백질-리간드 결합 친화도 예측을 위한 새로운 데이터셋인 DA VIS-complete를 소개하고, 이를 기반으로 세 가지 벤치마크 설정을 제안합니다. 이 데이터셋은 4,032개의 키나아제-리간드 쌍을 포함하며, 단백질의 변형(예: 치환, 삽입, 삭제, 인산화 등)을 고려하여 생물학적으로 더 현실적인 조건에서 모델의 성능을 평가할 수 있도록 설계되었습니다.

#### 1. 데이터셋
DA VIS-complete 데이터셋은 기존의 DA VIS 데이터셋을 확장하여, 수정된 키나아제 단백질과 그에 대한 리간드 쌍을 포함합니다. 이 데이터셋은 단백질 변형이 결합 친화도에 미치는 영향을 분석하는 데 중요한 역할을 합니다. 데이터셋은 Entrez Gene Symbols를 사용하여 키나아제 단백질을 식별하고, UniProt 데이터베이스에서 해당 아미노산 서열을 가져와 수동으로 수정된 아미노산 서열을 추가했습니다.

#### 2. 모델
이 연구에서는 두 가지 유형의 모델을 평가합니다: 도킹 기반 모델과 도킹 없는 모델. 도킹 없는 모델은 단백질의 아미노산 서열이나 예측된 단백질 접촉 맵을 사용하여 결합 친화도를 예측합니다. 반면, 도킹 기반 모델은 고해상도 단백질 구조를 사용하여 원자 수준의 상호작용을 고려합니다. 이 두 가지 접근 방식의 성능을 비교하여, 각 모델이 단백질 변형에 대한 일반화 능력을 어떻게 발휘하는지를 평가합니다.

#### 3. 벤치마크 설정
세 가지 벤치마크 설정은 다음과 같습니다:
- **증강 데이터셋 예측(Augmented Dataset Prediction)**: 수정된 단백질-리간드 쌍을 포함한 데이터셋을 사용하여 모델의 일반적인 예측 능력을 평가합니다.
- **와일드타입에서 수정으로의 일반화(Wild-Type to Modification Generalization)**: 와일드타입 단백질로 훈련된 모델이 수정된 단백질에 대해 얼마나 잘 일반화되는지를 평가합니다.
- **소수 샷 수정 일반화(Few-Shot Modification Generalization)**: 제한된 수정된 단백질-리간드 쌍으로 모델을 미세 조정하여, 새로운 변형에 대한 일반화 능력을 평가합니다.

이러한 설정을 통해, 모델이 단백질 변형에 대한 예측을 얼마나 잘 수행하는지를 평가하고, 정밀 의학의 발전에 기여할 수 있는 가능성을 탐구합니다.

---




This paper introduces a new dataset for protein-ligand binding affinity prediction called DA VIS-complete, and proposes three benchmark settings based on this dataset. The dataset includes 4,032 kinase-ligand pairs and is designed to evaluate model performance under biologically more realistic conditions by considering protein modifications (e.g., substitutions, insertions, deletions, phosphorylation, etc.).

#### 1. Dataset
The DA VIS-complete dataset expands upon the existing DA VIS dataset by including modified kinase proteins and their corresponding ligand pairs. This dataset plays a crucial role in analyzing the impact of protein modifications on binding affinity. It identifies kinase proteins using Entrez Gene Symbols and retrieves the corresponding amino acid sequences from the UniProt database, manually adding modified amino acid sequences.

#### 2. Models
The study evaluates two types of models: docking-based models and docking-free models. Docking-free models predict binding affinity using protein amino acid sequences or predicted protein contact maps. In contrast, docking-based models utilize high-resolution protein structures to consider atom-level interactions. The performance of these two approaches is compared to assess how well each model generalizes to protein modifications.

#### 3. Benchmark Settings
The three benchmark settings are as follows:
- **Augmented Dataset Prediction**: This setting evaluates the general predictive capability of models using a dataset that includes modified protein-ligand pairs.
- **Wild-Type to Modification Generalization**: This benchmark assesses how well models trained on wild-type proteins generalize to unseen modified variants.
- **Few-Shot Modification Generalization**: This setting evaluates model adaptability by fine-tuning on a limited number of modified protein-ligand pairs to assess generalization to unseen variants.

Through these settings, the study aims to evaluate how well models perform in predicting protein modifications and explore the potential contributions to advancing precision medicine.


<br/>
# Results



이 논문에서는 단백질-리간드 결합 친화도 예측을 위한 새로운 데이터셋인 DA VIS-complete를 기반으로 여러 모델의 성능을 평가했습니다. 연구에서는 다섯 가지 도킹 없는 모델(DeepDTA, AttentionDTA, GraphDTA, DGraphDTA, MGraphDTA)과 두 가지 도킹 기반 모델(FDA, Boltz-2)을 비교했습니다. 

#### 1. 데이터셋 및 테스트 설정
- **데이터셋**: DA VIS-complete는 4,032개의 키나제-리간드 쌍을 포함하며, 이 데이터셋은 단백질 변형을 고려하여 구성되었습니다.
- **테스트 설정**: 세 가지 주요 벤치마크 설정이 사용되었습니다:
  1. **증강 데이터셋 예측**: 기존의 DA VIS 데이터셋에 변형된 단백질-리간드 쌍을 추가하여 모델 성능을 평가했습니다.
  2. **와일드타입에서 변형으로의 일반화**: 모델이 와일드타입 단백질에서 변형된 단백질로의 일반화 능력을 평가했습니다.
  3. **소수 샷 변형 일반화**: 제한된 수의 변형된 단백질-리간드 쌍으로 모델을 미세 조정하여 일반화 능력을 평가했습니다.

#### 2. 성능 메트릭
- **평가 메트릭**: 평균 제곱 오차(MSE)와 피어슨 상관 계수(Rp)를 사용하여 모델의 예측 성능을 평가했습니다.

#### 3. 결과 요약
- **증강 데이터셋 예측**: FDA 모델이 도킹 없는 모델들보다 전반적으로 더 나은 성능을 보였으며, 특히 새로운 단백질과 리간드 조합에 대한 예측에서 우수한 결과를 나타냈습니다.
- **와일드타입에서 변형으로의 일반화**: 도킹 없는 모델들은 와일드타입 데이터에 과적합되는 경향을 보였고, 변형된 단백질에 대한 예측 성능이 낮았습니다. 반면, 도킹 기반 모델은 변형된 단백질에 대한 예측에서 더 나은 일반화 성능을 보였습니다.
- **소수 샷 변형 일반화**: 모든 도킹 없는 모델이 미세 조정 후 성능이 향상되었으나, 여전히 낮은 성능을 보였습니다. 특히 AttentionDTA 모델이 가장 좋은 성능을 보였으며, MSE가 0.42로 감소하고 Rp가 0.80으로 증가했습니다.

이 연구는 단백질 변형이 단백질-리간드 상호작용에 미치는 영향을 고려한 데이터셋과 벤치마크를 제공함으로써, 정밀 의학 및 약물 발견 분야에서의 모델 개발에 기여할 것으로 기대됩니다.

---




In this paper, the authors evaluated the performance of several models for protein-ligand binding affinity prediction based on a new dataset called DA VIS-complete. The study compared five docking-free models (DeepDTA, AttentionDTA, GraphDTA, DGraphDTA, MGraphDTA) and two docking-based models (FDA, Boltz-2).

#### 1. Dataset and Test Settings
- **Dataset**: DA VIS-complete includes 4,032 kinase-ligand pairs and is curated to account for protein modifications.
- **Test Settings**: Three main benchmark settings were used:
  1. **Augmented Dataset Prediction**: The existing DA VIS dataset was augmented with modified protein-ligand pairs to evaluate model performance.
  2. **Wild-Type to Modification Generalization**: This benchmark assessed the model's ability to generalize from wild-type proteins to modified variants.
  3. **Few-Shot Modification Generalization**: The model was fine-tuned on a limited number of modified protein-ligand pairs to evaluate its adaptability.

#### 2. Performance Metrics
- **Evaluation Metrics**: Mean Squared Error (MSE) and Pearson correlation coefficient (Rp) were used to assess the predictive performance of the models.

#### 3. Summary of Results
- **Augmented Dataset Prediction**: The FDA model consistently outperformed the docking-free models, especially in predicting new combinations of proteins and ligands.
- **Wild-Type to Modification Generalization**: Docking-free models tended to overfit to wild-type data, showing lower performance on modified proteins. In contrast, docking-based models demonstrated better generalization to modified proteins.
- **Few-Shot Modification Generalization**: All docking-free models showed improved performance after fine-tuning, but still exhibited low overall performance. Notably, the AttentionDTA model achieved the best results, with MSE decreasing to 0.42 and Rp increasing to 0.80.

This study provides a curated dataset and benchmarks that consider the impact of protein modifications on protein-ligand interactions, contributing to the development of models in precision medicine and drug discovery.


<br/>
# 예제



이 논문에서는 단백질-리간드 결합 친화도 예측을 위한 새로운 데이터셋인 DA VIS-complete를 소개하고, 이를 기반으로 세 가지 벤치마크 설정을 제안합니다. 각 설정은 모델의 성능을 평가하기 위해 다양한 훈련 및 테스트 데이터 분할을 사용합니다.

1. **Augmented Dataset Prediction**: 
   - **훈련 데이터**: DA VIS-complete 데이터셋의 모든 단백질-리간드 쌍(P*L)으로 구성됩니다. 여기에는 수정된 단백질과 일반 단백질이 모두 포함됩니다.
   - **테스트 데이터**: 세 가지 주요 분할로 나뉩니다:
     - **New-ligand**: 훈련 세트와 테스트 세트 간에 리간드 이름이 겹치지 않도록 하거나, Tanimoto 유사도가 0.5 이하인 리간드를 사용합니다.
     - **New-protein**: 훈련 세트에 포함된 단백질과 겹치지 않는 새로운 단백질을 테스트합니다.
     - **Both-new**: 훈련 세트와 테스트 세트 모두에서 새로운 단백질과 리간드를 사용합니다.

2. **Wild-Type to Modification Generalization**:
   - **훈련 데이터**: 일반 단백질-리간드 쌍(PwL)으로만 구성됩니다.
   - **테스트 데이터**: 세 가지 설정으로 나뉩니다:
     - **Global modification generalization**: 모든 수정된 단백질-리간드 쌍(PmL)을 포함합니다.
     - **Same-ligand, different-modifications**: 동일한 리간드에 대해 서로 다른 수정된 단백질을 평가합니다.
     - **Same-modification, different-ligands**: 동일한 수정된 단백질에 대해 서로 다른 리간드를 평가합니다.

3. **Few-Shot Modification Generalization**:
   - **훈련 데이터**: 일반 단백질-리간드 쌍(PwL)으로 구성됩니다.
   - **테스트 데이터**: 두 가지 설정으로 나뉩니다:
     - **Same-ligand, different-modifications**: 동일한 리간드에 대해 수정된 단백질을 평가합니다.
     - **Same-modification, different-ligands**: 동일한 수정된 단백질에 대해 서로 다른 리간드를 평가합니다.

이러한 설정을 통해 모델의 일반화 능력을 평가하고, 단백질 수정이 결합 친화도 예측에 미치는 영향을 분석합니다.




This paper introduces a new dataset for protein-ligand binding affinity prediction called DA VIS-complete and proposes three benchmark settings based on it. Each setting uses various training and testing data splits to evaluate model performance.

1. **Augmented Dataset Prediction**: 
   - **Training Data**: Composed of all protein-ligand pairs from the DA VIS-complete dataset (P*L), including both modified and wild-type proteins.
   - **Testing Data**: Divided into three main splits:
     - **New-ligand**: Ensures no overlap in ligand names between training and test sets or uses ligands with a Tanimoto similarity of less than 0.5.
     - **New-protein**: Tests new proteins that do not overlap with those in the training set.
     - **Both-new**: Uses new proteins and ligands in both training and test sets.

2. **Wild-Type to Modification Generalization**:
   - **Training Data**: Composed solely of wild-type protein-ligand pairs (PwL).
   - **Testing Data**: Divided into three settings:
     - **Global modification generalization**: Includes all modified protein-ligand pairs (PmL).
     - **Same-ligand, different-modifications**: Evaluates different modified proteins against the same ligand.
     - **Same-modification, different-ligands**: Evaluates different ligands against the same modified protein.

3. **Few-Shot Modification Generalization**:
   - **Training Data**: Composed of wild-type protein-ligand pairs (PwL).
   - **Testing Data**: Divided into two settings:
     - **Same-ligand, different-modifications**: Evaluates modified proteins against the same ligand.
     - **Same-modification, different-ligands**: Evaluates different ligands against the same modified protein.

These settings allow for the assessment of the model's generalization capabilities and the analysis of how protein modifications impact binding affinity predictions.

<br/>
# 요약


이 연구에서는 DA VIS 데이터셋을 수정된 단백질-리간드 쌍을 포함하여 보완하고, 세 가지 벤치마크 설정을 통해 모델의 일반화 능력을 평가하였다. 실험 결과, 도킹 기반 모델이 제로샷 설정에서 더 나은 일반화를 보였으며, 도킹 없는 모델은 야생형 단백질에 과적합되는 경향이 있었다. 또한, 소수의 수정된 예제에 대한 미세 조정이 도킹 없는 모델의 성능을 향상시킬 수 있음을 발견하였다.

---

This study curated the DA VIS dataset by incorporating modified protein-ligand pairs and evaluated model generalization capabilities through three benchmark settings. Experimental results showed that docking-based models generalized better in zero-shot settings, while docking-free models tended to overfit to wild-type proteins. Additionally, fine-tuning on a small set of modified examples was found to improve the performance of docking-free models.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: DA VIS-complete 데이터셋의 확장 과정을 보여줍니다. 이 피규어는 수정된 키나제 단백질-리간드 쌍이 기존의 데이터셋에 어떻게 추가되었는지를 시각적으로 설명합니다. 특히, ABL1 단백질의 여러 변형이 포함된 예시를 통해 단백질 수정이 리간드 결합에 미치는 영향을 강조합니다.
   - **Figure 1(c)**: Augmented Dataset Prediction 벤치마크의 세 가지 주요 분할을 보여줍니다. 이는 모델의 일반화 능력을 평가하는 데 중요한 역할을 합니다.
   - **Figure 1(d)**: Wild-Type to Modification Generalization 벤치마크의 평가 설정을 설명합니다. 이는 모델이 야생형 단백질에서 수정된 변형으로 얼마나 잘 일반화되는지를 평가하는 데 중점을 둡니다.
   - **Figure 1(e)**: Few-Shot Modification Generalization 벤치마크의 설정을 보여줍니다. 이는 모델이 제한된 수정된 단백질-리간드 쌍에 대해 얼마나 잘 적응하는지를 평가합니다.

2. **테이블**
   - **Table 1**: 각 벤치마크의 하위 작업 및 데이터셋 분할을 요약합니다. 이는 모델 훈련, 미세 조정 및 테스트 세트를 명확히 하여 연구의 구조를 이해하는 데 도움을 줍니다.
   - **Table 2**: 다양한 모델의 성능 비교를 보여줍니다. 각 모델의 평균 제곱 오차(MSE)와 피어슨 상관 계수(Rp)를 통해 모델의 예측 성능을 평가합니다. 이 테이블은 모델 간의 성능 차이를 명확히 보여주며, 특정 모델이 특정 데이터 분할에서 더 나은 성능을 보이는지를 강조합니다.
   - **Table 3**: Wild-Type to Modification Generalization 벤치마크의 성능 비교를 제공합니다. 이는 수정된 단백질-리간드 쌍에 대한 모델의 일반화 능력을 평가하는 데 중요한 정보를 제공합니다.
   - **Table 4**: Same-ligand, different-modifications 및 Same-modification, different-ligands 벤치마크의 결과를 요약합니다. 이는 모델이 수정된 단백질에 대해 얼마나 잘 작동하는지를 보여줍니다.
   - **Table 5**: Few-Shot Modification Generalization 벤치마크의 결과를 보여줍니다. 모델이 제한된 수정된 단백질-리간드 쌍에 대해 얼마나 잘 적응하는지를 평가합니다.

3. **어펜딕스**
   - 어펜딕스에는 모델 훈련 세부사항, 데이터 전처리 방법, 하이퍼파라미터 설정 등이 포함되어 있습니다. 이는 연구 결과의 재현성을 높이는 데 중요한 역할을 합니다. 또한, 각 모델의 성능을 평가하기 위한 추가적인 실험 결과와 분석이 포함되어 있어, 연구의 신뢰성을 높입니다.

---



1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the process of expanding the DA VIS-complete dataset. This figure visually explains how modified kinase protein-ligand pairs were added to the existing dataset, emphasizing the impact of protein modifications on ligand binding through examples like various ABL1 variants.
   - **Figure 1(c)**: Shows the three main splits of the Augmented Dataset Prediction benchmark, which plays a crucial role in assessing the generalization ability of the models.
   - **Figure 1(d)**: Describes the evaluation settings for the Wild-Type to Modification Generalization benchmark, focusing on how well models generalize from wild-type proteins to modified variants.
   - **Figure 1(e)**: Displays the setup for the Few-Shot Modification Generalization benchmark, assessing how well models adapt to a limited number of modified protein-ligand pairs.

2. **Tables**
   - **Table 1**: Summarizes the sub-tasks and dataset splits for each benchmark, clarifying the structure of the research and aiding in understanding the model training, fine-tuning, and test sets.
   - **Table 2**: Compares the performance of various models, showing the mean squared error (MSE) and Pearson correlation coefficient (Rp) for each model. This table highlights performance differences among models and emphasizes which models perform better under specific data splits.
   - **Table 3**: Provides a performance comparison for the Wild-Type to Modification Generalization benchmark, offering critical insights into the models' generalization capabilities on modified protein-ligand pairs.
   - **Table 4**: Summarizes results for the Same-ligand, different-modifications and Same-modification, different-ligands benchmarks, demonstrating how well models perform on modified proteins.
   - **Table 5**: Displays results for the Few-Shot Modification Generalization benchmark, evaluating how well models adapt to limited modified protein-ligand pairs.

3. **Appendices**
   - The appendices include details on model training, data preprocessing methods, and hyperparameter settings, which are crucial for enhancing the reproducibility of the research findings. Additionally, further experimental results and analyses are provided, bolstering the reliability of the study.

<br/>
# refer format:

### BibTeX 

```bibtex
@inproceedings{Wu2025,
  author = {Ming-Hsiu Wu and Ziqian Xie and Shuiwang Ji and Degui Zhi},
  title = {Towards Precision Protein-Ligand Affinity Prediction Benchmark: A Complete and Modification-Aware DA VIS Dataset},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year = {2025},
  institution = {The University of Texas Health Science Center at Houston and Texas A\&M University},
  url = {https://github.com/ZhiGroup/DAVIS-complete}
}
```

### 시카고 스타일

Ming-Hsiu Wu, Ziqian Xie, Shuiwang Ji, and Degui Zhi. 2025. "Towards Precision Protein-Ligand Affinity Prediction Benchmark: A Complete and Modification-Aware DA VIS Dataset." In *Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)*. The University of Texas Health Science Center at Houston and Texas A&M University. https://github.com/ZhiGroup/DAVIS-complete.
