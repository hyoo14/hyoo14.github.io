---
layout: post
title:  "[2026]BiomeGPT: A Foundation Model for the Human Gut Microbiome"
date:   2026-01-17 18:41:30 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

인간 장내 메타게놈 시퀀스로 LM학습 아닐까했는데 그게 아니구...

BiomeGPT는 마스크 모델링(masked modeling) 기법을 사용하여 미생물의 상대적 풍부도 프로파일을 구조화된 시퀀스로 처리합니다. 이 과정에서 모델은 미생물의 조합 패턴과 샘플 간의 신호를 학습하여, 미생물군의 복잡한 상호작용을 이해할 수 있도록 함... 어번던스를 학습하는거였음  


짧은 요약(Abstract) :


인간의 장내 미생물군은 숙주 건강에 대한 풍부한 정보를 담고 있지만, 현재의 분석 파이프라인은 개별 작업에 최적화되어 있어 미생물군이 건강과 질병에 미치는 영향을 종합적으로 이해하는 데 한계가 있다. 본 연구에서는 13,300개 이상의 인간 장내 메타게놈을 기반으로 사전 훈련된 BiomeGPT라는 변환기 기반의 기초 모델을 소개한다. 이 모델은 건강한 상태와 31가지 다양한 질병을 포함한 32가지 표현형에 대한 맥락 인식, 종 수준의 장내 미생물군 표현을 학습한다. BiomeGPT는 미생물 군집 프로파일에 내재된 정량적 조성 구조와 복잡한 종 간 의존성을 포착한다. 숙주 건강 상태 예측을 위해 미세 조정된 BiomeGPT는 건강한 미생물군과 질병 미생물군을 정확하게 구별하고, 광범위한 임상 스펙트럼에서 개별 질병 상태를 해결한다. 또한, 모델의 주의 패턴은 생물학적으로 그럴듯한 미생물 서명을 드러내며, 숙주 표현형과 연결된 공유 및 질병 특이적 미생물 종을 강조한다. BiomeGPT는 종 수준의 장내 미생물군 표현 학습 및 예측을 위한 통합되고 확장 가능한 프레임워크를 제공함으로써 바이오마커 발견, 질병 분류 및 미생물군 기반 정밀 의학을 위한 새로운 경로를 열어준다.

---



The human gut microbiome encodes rich information about host health, yet current analysis pipelines remain narrowly optimized for individual tasks, limiting our ability to gain a thorough view of how the microbiome impacts health and disease. Here, we introduce BiomeGPT, a transformer-based foundation model pretrained on over 13,300 human gut metagenomes spanning 32 phenotypes—including healthy and 31 diverse diseases—to learn context-aware, species-level gut microbiome representations. The model captures quantitative compositional structure and intricate cross-species dependencies embedded within community profiles. When fine-tuned for predicting host health status, BiomeGPT accurately distinguishes healthy from diseased microbiomes and resolves individual disease states across a broad clinical spectrum. Furthermore, its attention patterns reveal biologically plausible microbial signatures, highlighting both shared and disease-specific microbial species linked to host phenotypes. By providing a unified, scalable framework for species-level gut microbiome representation learning and prediction, BiomeGPT enables new avenues for biomarker discovery, disease stratification, and microbiome-driven precision medicine.


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



**모델 및 아키텍처**
BiomeGPT는 인간 장내 미생물군의 분석을 위해 설계된 변환기 기반의 기초 모델입니다. 이 모델은 13,300개 이상의 인간 장내 메타게놈 샘플을 사용하여 사전 훈련되었으며, 각 샘플은 종 수준의 세균 분포를 포함하고 있습니다. BiomeGPT는 마스크 모델링(masked modeling) 기법을 사용하여 미생물의 상대적 풍부도 프로파일을 구조화된 시퀀스로 처리합니다. 이 과정에서 모델은 미생물의 조합 패턴과 샘플 간의 신호를 학습하여, 미생물군의 복잡한 상호작용을 이해할 수 있도록 합니다.

**훈련 데이터**
모델 훈련에 사용된 데이터셋은 14,451개의 인간 메타게놈 샘플로 구성되어 있으며, 이 샘플들은 여러 공개 데이터베이스에서 수집되었습니다. 이 데이터셋은 건강한 샘플과 35개의 질병 상태를 포함하고 있으며, 각 샘플은 MetaPhlAn을 사용하여 종 수준으로 세분화된 분류를 받았습니다. 훈련 데이터는 90%를 훈련 세트로, 10%를 검증 세트로 나누어 사용하였습니다.

**특별한 기법**
BiomeGPT는 두 단계의 훈련 과정을 거칩니다. 첫 번째 단계에서는 장과 비장 부위의 메타게놈 샘플을 포함한 대규모 데이터셋에서 사전 훈련을 수행합니다. 두 번째 단계에서는 장내 미생물군에 특화된 데이터셋을 사용하여 모델을 추가로 조정합니다. 이 과정에서 모델은 각 샘플의 미생물 종과 그들의 풍부도를 쌍으로 묶어 입력으로 사용하며, 각 종은 학습 가능한 임베딩 벡터로 변환됩니다.

모델의 훈련 과정에서, 25%의 비어 있지 않은 풍부도 종이 무작위로 마스킹되어, 모델이 마스킹된 종의 풍부도를 예측하도록 훈련됩니다. 이때, 마스킹된 종은 서로 간의 주의(attention)를 가질 수 없지만, 비마스킹된 종과는 상호작용할 수 있습니다. 이러한 방식은 모델이 미생물 간의 공존 및 풍부도 관계를 학습하는 데 도움을 줍니다.

**세부 사항**
모델의 아키텍처는 8개의 변환기 층으로 구성되어 있으며, 각 층은 8개의 주의 헤드를 포함하고 있습니다. 훈련 후, BiomeGPT는 건강 상태 예측을 위한 여러 임상 분류 작업에 대해 미세 조정(fine-tuning)됩니다. 이 과정에서 모델은 각 샘플의 <cls> 토큰을 사용하여 샘플 수준의 정보를 집계하고, 이를 통해 건강한 샘플과 질병 샘플을 구분하는 데 필요한 정보를 학습합니다.

### English Version

**Model and Architecture**
BiomeGPT is a transformer-based foundation model designed for the analysis of the human gut microbiome. This model is pretrained on over 13,300 human gut metagenome samples, each containing species-level bacterial distributions. BiomeGPT employs a masked modeling technique to process microbial relative abundance profiles as structured sequences. During this process, the model learns compositional patterns and inter-sample signals, enabling it to understand the complex interactions within microbiomes.

**Training Data**
The dataset used for model training consists of 14,451 human metagenomic samples collected from various public databases. This dataset includes healthy samples and 35 disease states, with each sample taxonomically profiled at the species level using MetaPhlAn. The training data is split into 90% for the training set and 10% for the validation set.

**Special Techniques**
BiomeGPT undergoes a two-phase training process. In the first phase, it is pretrained on a large dataset that includes metagenomic samples from both gut and non-gut sites. In the second phase, the model is further refined using a gut-specific dataset. During this process, each sample's microbial species and their abundances are paired and used as input, with each species transformed into a learnable embedding vector.

During the training process, 25% of the non-zero abundance species are randomly masked, allowing the model to predict the abundances of the masked species. In this case, masked species cannot attend to each other but can interact with unmasked species. This approach helps the model learn co-occurrence and abundance relationships among microbes.

**Details**
The model architecture consists of 8 stacked transformer layers, each containing 8 attention heads. After training, BiomeGPT is fine-tuned for several clinical classification tasks aimed at predicting health status. In this process, the model aggregates information from the sample-level <cls> token to learn the necessary information to distinguish between healthy and diseased samples.


<br/>
# Results



BiomeGPT 모델은 건강한 미생물군과 질병이 있는 미생물군을 구별하는 능력을 평가하기 위해 여러 실험을 수행했습니다. 이 모델은 10-겹 교차 검증을 통해 내부 데이터셋에서 평균 정확도 0.851, F1 점수 0.852, AUROC 0.921을 달성하여 강력한 일반화 성능을 보여주었습니다. 이는 모델이 질병 상태와 관련된 미생물 조성의 광범위한 변화를 효과적으로 포착할 수 있음을 나타냅니다.

외부 검증을 위해 927개의 샘플로 구성된 독립 데이터셋을 사용하여 모델의 일반화 능력을 평가했습니다. 이 과정에서 BiomeGPT는 건강한 미생물군과 질병이 있는 미생물군을 구별하는 작업에서 평균 정확도 0.762, 평균 F1 점수 0.702, 평균 AUROC 0.897을 기록했습니다. 이러한 결과는 BiomeGPT가 다양한 임상 연구에 잘 일반화되며, 미생물 조성의 집단적 변동성에 강한 내성을 가지고 있음을 보여줍니다.

BiomeGPT의 성능을 기존의 기계 학습 모델인 Random Forest와 XGBoost와 비교한 결과, BiomeGPT는 건강한 미생물군과 질병이 있는 미생물군을 구별하는 작업에서 더 높은 평균 정확도(0.762)와 평균 F1 점수(0.702)를 기록했습니다. XGBoost는 각각 0.712와 0.634의 성능을 보였으며, Random Forest는 0.711과 0.633의 성능을 보였습니다. 이러한 결과는 BiomeGPT가 외부 데이터셋에 대한 일반화 능력이 뛰어남을 나타냅니다.

또한, BiomeGPT는 기존의 미생물군 기반 모델인 Pope et al.과 MGM과 비교했을 때도 우수한 성능을 보였습니다. BiomeGPT는 IBD(염증성 장 질환)와 건강한 미생물군을 구별하는 작업에서 AUROC 0.993, Crohn's Disease(CD)와 건강한 미생물군을 구별하는 작업에서 AUROC 0.999를 기록했습니다. 반면, Pope et al.은 AUROC 0.687, MGM은 AUROC 0.829와 0.844를 기록했습니다. 이러한 비교 결과는 BiomeGPT의 기초 모델 설계가 미생물군 예측 작업에서 강력한 성능을 발휘할 수 있도록 해준다는 것을 보여줍니다.




The BiomeGPT model was evaluated for its ability to distinguish between healthy and diseased microbiomes through several experiments. The model achieved a mean accuracy of 0.851, an F1 score of 0.852, and an AUROC of 0.921 in 10-fold cross-validation on the internal dataset, demonstrating strong generalization performance. This indicates that the model effectively captures broad alterations in microbial composition associated with disease status.

For external validation, an independent dataset comprising 927 samples was used to assess the model's generalizability. In this evaluation, BiomeGPT maintained strong predictive performance, achieving a macro accuracy of 0.762, a macro F1 score of 0.702, and a macro AUROC of 0.897 for the healthy vs. diseased classification task. These results demonstrate that BiomeGPT generalizes well to heterogeneous clinical studies and is robust to cohort-specific variations in microbiome composition.

To contextualize BiomeGPT's performance, it was benchmarked against classical machine learning models such as Random Forest and XGBoost. In the healthy vs. diseased classification task, BiomeGPT achieved a higher macro accuracy (0.762) and macro F1 score (0.702) compared to XGBoost (0.712 and 0.634) and Random Forest (0.711 and 0.633). This indicates that BiomeGPT exhibits superior generalization to external cohorts.

Additionally, BiomeGPT outperformed existing microbiome models such as those proposed by Pope et al. and MGM. For classifying IBD (Inflammatory Bowel Disease) versus healthy microbiomes, BiomeGPT achieved an AUROC of 0.993, and for Crohn's Disease (CD) versus healthy microbiomes, it achieved an AUROC of 0.999. In contrast, Pope et al. reported an AUROC of 0.687, while MGM recorded AUROCs of 0.829 and 0.844. These comparisons highlight the advantages of BiomeGPT's foundation model design, which enables robust performance in clinical microbiome prediction tasks.


<br/>
# 예제
### 한글 설명




1. **훈련 데이터 구성**
   - **데이터셋**: BiomeGPT는 13,524개의 인간 장내 메타게놈 샘플로 훈련되었습니다. 이 샘플들은 건강한 상태와 35개의 다양한 질병 상태를 포함합니다.
   - **입력 형식**: 각 샘플은 미생물 종과 그들의 이진화된 상대적 풍부도(binned abundance)로 구성된 시퀀스로 표현됩니다. 예를 들어, 샘플 1의 입력은 다음과 같을 수 있습니다:
     ```
     <cls>, 0
     (Escherichia coli, 3)
     (Bacteroides fragilis, 2)
     (Lactobacillus rhamnosus, 1)
     ```
   - **출력 형식**: 모델은 각 샘플의 건강 상태를 예측합니다. 예를 들어, 샘플 1이 건강한 경우 출력은 "Healthy"가 될 수 있습니다.

2. **테스트 데이터 구성**
   - **데이터셋**: 외부 검증을 위해 927개의 샘플이 사용되었습니다. 이 샘플들은 훈련 데이터셋에 포함되지 않은 연구에서 수집되었습니다.
   - **입력 형식**: 테스트 샘플도 훈련 샘플과 동일한 형식으로 제공됩니다. 예를 들어, 테스트 샘플 1의 입력은 다음과 같을 수 있습니다:
     ```
     <cls>, 0
     (Faecalibacterium prausnitzii, 4)
     (Bifidobacterium longum, 2)
     (Clostridium difficile, 1)
     ```
   - **출력 형식**: 모델은 이 샘플의 건강 상태를 예측합니다. 예를 들어, 샘플 1이 질병 상태인 경우 출력은 "Diseased"가 될 수 있습니다.

3. **구체적인 태스크**
   - **이진 분류 태스크**: 모델은 건강한 샘플과 질병 샘플을 구분하는 이진 분류 태스크를 수행합니다. 예를 들어, 훈련 데이터에서 모델은 "Healthy"와 "Diseased"를 구분하는 방법을 학습합니다.
   - **다중 클래스 분류 태스크**: 모델은 특정 질병(예: 크론병, 대장암 등)과 건강한 샘플을 구분하는 다중 클래스 분류 태스크도 수행합니다. 이 경우, 모델은 각 질병에 대해 별도의 이진 분류기를 학습합니다.




**Example of Training and Testing Data for the BiomeGPT Model**

1. **Training Data Composition**
   - **Dataset**: BiomeGPT was trained on 13,524 human gut metagenome samples, which include healthy states and 35 different disease states.
   - **Input Format**: Each sample is represented as a sequence of microbial species and their binned relative abundances. For example, the input for Sample 1 might look like this:
     ```
     <cls>, 0
     (Escherichia coli, 3)
     (Bacteroides fragilis, 2)
     (Lactobacillus rhamnosus, 1)
     ```
   - **Output Format**: The model predicts the health status of each sample. For instance, if Sample 1 is healthy, the output could be "Healthy".

2. **Testing Data Composition**
   - **Dataset**: An external validation set of 927 samples was used. These samples were collected from studies not included in the training dataset.
   - **Input Format**: Testing samples are provided in the same format as training samples. For example, the input for Test Sample 1 might look like this:
     ```
     <cls>, 0
     (Faecalibacterium prausnitzii, 4)
     (Bifidobacterium longum, 2)
     (Clostridium difficile, 1)
     ```
   - **Output Format**: The model predicts the health status of this sample. For example, if Sample 1 is in a diseased state, the output could be "Diseased".

3. **Specific Tasks**
   - **Binary Classification Task**: The model performs a binary classification task to distinguish between healthy and diseased samples. For example, in the training data, the model learns how to differentiate "Healthy" from "Diseased".
   - **Multi-Class Classification Task**: The model also performs a multi-class classification task to distinguish specific diseases (e.g., Crohn's disease, colorectal cancer) from healthy samples. In this case, the model learns separate binary classifiers for each disease.

<br/>
# 요약


BiomeGPT는 13,300개 이상의 인간 장내 메타게놈을 기반으로 한 변환기 기반의 기초 모델로, 건강 상태 예측을 위한 임상 분류 작업에 대해 강력한 성능을 보였다. 모델은 건강한 샘플과 질병 샘플을 구별하는 데 평균 정확도 85.1%를 달성했으며, 26개의 질병 유형에 대한 예측에서도 높은 성능을 나타냈다. 주목할 만한 미생물 종의 중요성을 분석하여 건강과 질병 간의 연관성을 해석할 수 있는 통찰력을 제공하였다.



BiomeGPT is a transformer-based foundation model pretrained on over 13,300 human gut metagenomes, demonstrating strong performance in clinical classification tasks for health status prediction. The model achieved an average accuracy of 85.1% in distinguishing healthy samples from diseased ones and showed high performance across predictions for 26 disease types. It provided insights into the importance of notable microbial species, enabling the interpretation of associations between health and disease.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **다이어그램 1 (Gut Metagenomic Dataset Overview)**:
   - 이 다이어그램은 BiomeGPT의 훈련 및 검증 데이터셋의 구성과 각 질병에 대한 샘플 수를 보여줍니다. 13,349개의 샘플이 포함되어 있으며, 32개의 다양한 임상 상태(건강 및 31개의 질병)를 나타냅니다. 이 데이터는 모델이 다양한 질병 상태를 학습하고 예측하는 데 중요한 역할을 합니다.

2. **다이어그램 2 (Input Representation and Masked Attention Mechanism)**:
   - 이 다이어그램은 BiomeGPT의 입력 표현 및 마스킹된 주의 메커니즘을 설명합니다. 각 샘플은 미생물 종과 그들의 빈도 값으로 구성된 시퀀스로 표현되며, 마스킹된 주의 메커니즘을 통해 모델은 비어 있지 않은 종들 간의 관계를 학습합니다. 이는 모델이 미생물의 상호작용을 이해하는 데 도움을 줍니다.

3. **다이어그램 3 (Fine-tuning BiomeGPT for Health Status Prediction Tasks)**:
   - 이 다이어그램은 BiomeGPT의 미세 조정 과정과 건강 상태 예측을 위한 분류 헤드를 추가하는 방법을 보여줍니다. 모델은 건강과 질병 상태를 구분하는 데 강력한 성능을 발휘하며, 다양한 질병에 대한 예측을 수행합니다.

4. **다이어그램 4 (Species Most Emphasized by BiomeGPT)**:
   - 이 다이어그램은 BiomeGPT가 건강과 질병 예측에서 가장 중요하게 여기는 미생물 종을 보여줍니다. 각 질병 카테고리에서 높은 주의 점수를 받은 종들이 나열되어 있으며, 이는 모델이 특정 질병과 관련된 미생물 신호를 포착하는 데 도움을 줍니다.

#### 테이블
- **테이블 (Performance Metrics)**:
   - BiomeGPT의 성능 메트릭을 요약한 테이블은 건강과 질병 상태를 구분하는 데 있어 모델의 정확도, F1 점수 및 AUROC를 보여줍니다. 모델은 건강과 질병을 구분하는 데 평균 0.851의 정확도를 기록했으며, 이는 모델의 강력한 일반화 능력을 나타냅니다.

#### 어펜딕스
- **어펜딕스 (Methods and Additional Data)**:
   - BiomeGPT의 훈련 및 검증 방법론, 데이터 전처리 과정, 주의 분석 방법 등이 포함되어 있습니다. 이 정보는 모델의 신뢰성과 재현성을 높이는 데 기여하며, 향후 연구에 대한 기초 자료로 활용될 수 있습니다.

---



#### Diagrams and Figures
1. **Figure 1 (Gut Metagenomic Dataset Overview)**:
   - This figure illustrates the composition of the training and validation datasets for BiomeGPT, showing the number of samples for each disease. It includes 13,349 samples representing 32 diverse clinical conditions (healthy and 31 diseases), which is crucial for the model to learn and predict various disease states.

2. **Figure 2 (Input Representation and Masked Attention Mechanism)**:
   - This diagram explains the input representation and masked attention mechanism of BiomeGPT. Each sample is represented as a sequence of microbial species paired with their abundance values, and the masked attention mechanism allows the model to learn relationships among non-masked species. This helps the model understand microbial interactions.

3. **Figure 3 (Fine-tuning BiomeGPT for Health Status Prediction Tasks)**:
   - This figure shows the fine-tuning process of BiomeGPT and how a classification head is added for health status prediction. The model demonstrates strong performance in distinguishing between healthy and diseased states and performs predictions across various diseases.

4. **Figure 4 (Species Most Emphasized by BiomeGPT)**:
   - This figure summarizes the microbial species that BiomeGPT considers most influential for health and disease prediction. It lists species with high attention scores across disease categories, indicating the model's ability to capture disease-linked microbial signals.

#### Tables
- **Table (Performance Metrics)**:
   - A summary table of performance metrics for BiomeGPT shows accuracy, F1 scores, and AUROC for distinguishing between healthy and diseased states. The model achieved an average accuracy of 0.851, indicating its strong generalization capability.

#### Appendix
- **Appendix (Methods and Additional Data)**:
   - The appendix includes the training and validation methodologies for BiomeGPT, data preprocessing steps, and attention analysis methods. This information contributes to the model's reliability and reproducibility, serving as a foundational resource for future research.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@article{Medearis2026,
  author = {Nicholas A. Medearis and Siyao Zhu and Ali R. Zomorrodi},
  title = {BiomeGPT: A Foundation Model for the Human Gut Microbiome},
  journal = {bioRxiv},
  year = {2026},
  month = {January},
  doi = {10.64898/2026.01.05.697599},
  url = {https://doi.org/10.64898/2026.01.05.697599},
  note = {Preprint}
}
```

### 시카고 스타일 인용
Medearis, Nicholas A., Siyao Zhu, and Ali R. Zomorrodi. "BiomeGPT: A Foundation Model for the Human Gut Microbiome." bioRxiv, January 5, 2026. https://doi.org/10.64898/2026.01.05.697599. Preprint.
