---
layout: post
title:  "[2025]BioGeoFormer: A deep learning approach to classify unknown genes associated with critical biogeochemical cycles"
date:   2026-03-05 11:21:11 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 BioGeoFormer(BGF)라는 단백질 언어 모델을 사용하여 메탄, 질소, 황 및 인과 관련된 생물지구화학적 경로를 분류하는 방법을 제시합니다.


짧은 요약(Abstract) :


이 논문에서는 미생물 생태학에서 기능적 주석을 달기 위한 새로운 접근법인 BioGeoFormer를 소개합니다. 기존의 정렬 기반 방법들은 여전히 많은 미생물 서열의 기능을 해결하지 못하고 있으며, 이로 인해 미생물의 기능에 대한 이해가 제한되고 있습니다. 반면, 사전 훈련된 자연어 처리 접근법은 다양한 생물학적 서열에서 기능을 추론하는 데 강력한 잠재력을 보여주고 있습니다. 본 연구에서는 단백질 언어 모델을 활용하여 메탄, 황, 질소 및 인과 같은 4개의 주요 생물지구화학적 사이클에 관련된 37개의 주요 경로 카테고리로 서열을 분류할 수 있는 방법을 제안합니다. ESM2-8m 모델을 미세 조정하여 생물지구화학적 사이클 경로에 대한 데이터베이스를 구축하였고, 이를 통해 BioGeoFormer(BGF)가 높은 성능을 보임을 입증했습니다. BGF는 메탄 연료를 사용하는 심해의 "콜드 시프" 환경에서 구축된 메타유전체 조립 유전체(MAG) 데이터셋에 적용되어 현재의 정보학적 접근법과 비교하여 유용성을 입증했습니다. BGF는 1.05M개의 유전자에 생물지구화학적 기능을 할당하였으며, 이 중 0.49M(46%)의 유전자는 다른 접근법에서 미지의 것으로 분류되었습니다. BGF는 HMMs 및 정렬 기반 접근법에 비해 평균적으로 6배 더 많은 유전자를 식별했습니다. 이 연구는 다양한 시스템에서 과정 기반 가설을 제시할 수 있는 새로운 도구를 제공하며, 미생물 다크 매터와 관련된 신비로운 단백질을 밝혀내는 데 기여합니다.

---




This paper introduces a novel approach for functional annotation in microbial ecology called BioGeoFormer. Traditional alignment-based methods still leave a significant portion of microbial sequences functionally unresolved, limiting our understanding of microbial functions. In contrast, pre-trained natural language processing approaches have shown strong potential for inferring functions from diverse biological sequences. We propose a protein language modeling approach that allows us to classify sequences into 37 defined key pathway categories involved in four major biogeochemical cycles: methane, sulfur, nitrogen, and phosphorus. By fine-tuning the ESM2-8m model using databases curated for biogeochemical cycling pathways, we demonstrate that our BioGeochemical cycling transFormer (BioGeoFormer or BGF) performs well. BGF was applied to a dataset of metagenome-assembled genomes (MAGs) constructed from methane-fueled deep-sea "cold seep" environments, showcasing its utility compared to current informatics approaches. A total of 1.05 million genes were assigned biogeochemical functions, with BGF suggesting putative ecosystem roles for 0.49 million (46%) of these genes, which were classified as unknown by other approaches. On average, BGF identified six times as many genes as Hidden Markov models (HMMs) and alignment-based approaches across various pathways. This study provides a novel tool capable of informing process-based hypotheses in diverse systems, highlighting cryptic proteins notably linked to microbial dark matter.


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



이 연구에서는 BioGeoFormer(BGF)라는 새로운 단백질 언어 모델을 개발하여 미생물의 생화학적 사이클과 관련된 유전자의 기능을 분류하는 방법을 제안합니다. BGF는 ESM-2이라는 사전 훈련된 단백질 언어 모델을 기반으로 하며, 4개의 주요 생화학적 사이클(메탄, 황, 질소, 인)에 관련된 37개의 경로 카테고리로 유전자를 분류합니다.

#### 1. 데이터 수집 및 전처리
BGF의 훈련 데이터는 MCycDB(메탄), SCycDB(황), NCycDB(질소), PCycDB(인)라는 4개의 데이터베이스에서 수집되었습니다. 이 데이터베이스들은 각각의 생화학적 경로에 따라 유전자를 수집하고, UniProt 데이터베이스에서 단백질 패밀리를 다운로드하여 구성되었습니다. 최종적으로, CD-HIT를 사용하여 중복된 시퀀스를 제거하고, 1.9M의 경로 특정 시퀀스를 포함하는 BioGeoFormer-db(BGFdb)를 생성했습니다.

#### 2. 모델 아키텍처
BGF는 ESM-2 모델을 기반으로 하며, 8백만 개의 파라미터를 가진 6개의 레이어로 구성되어 있습니다. 이 모델은 37개의 유닛을 가진 분류 레이어를 추가하여 각 유전자가 속하는 생화학적 경로를 예측할 수 있도록 설계되었습니다. BGF는 훈련, 검증, 테스트 데이터셋을 60/20/20 비율로 나누어 훈련되었습니다.

#### 3. 훈련 및 최적화
BGF는 AdamW 옵티마이저를 사용하여 1 에폭 동안 훈련되었으며, 초기 학습률은 0.0003으로 설정되었습니다. 훈련 과정에서 dropout 확률은 0.1로 설정하여 과적합을 방지했습니다. 모델의 성능을 평가하기 위해 Matthew's Correlation Coefficient(MCC), 정확도, 정밀도, 재현율, F1 점수 등의 다양한 성능 지표를 사용했습니다.

#### 4. 온도 스케일링을 통한 신뢰도 조정
모델의 예측 신뢰도를 조정하기 위해 온도 스케일링 기법을 적용했습니다. 이 기법은 softmax 출력의 확률을 조정하여 모델의 과신을 줄이고, 예측의 정확성을 높이는 데 기여했습니다. 이를 통해 BGF는 원거리 유사성을 가진 단백질을 보다 정확하게 예측할 수 있었습니다.

#### 5. 성능 평가 및 비교
BGF는 DIAMOND 및 Hidden Markov Models(HMMs)와 같은 기존의 정렬 기반 방법들과 비교하여 우수한 성능을 보였습니다. 특히, BGF는 원거리 유사성을 가진 단백질을 식별하는 데 있어 HMMs보다 6배 더 많은 유전자를 식별할 수 있었습니다. BGF는 또한 미생물 다크 매터와 관련된 유전자의 기능을 밝혀내는 데 중요한 도구로 자리 잡았습니다.




This study introduces a novel protein language model called BioGeoFormer (BGF) designed to classify the functions of genes associated with biogeochemical cycles in microorganisms. BGF is based on the pre-trained protein language model ESM-2 and classifies genes into 37 defined pathway categories related to four major biogeochemical cycles (methane, sulfur, nitrogen, and phosphorus).

#### 1. Data Collection and Preprocessing
The training data for BGF was collected from four databases: MCycDB (methane), SCycDB (sulfur), NCycDB (nitrogen), and PCycDB (phosphorus). These databases were constructed by manually retrieving genes based on metabolic pathways and downloading corresponding protein families from the UniProt database. Finally, CD-HIT was used to remove duplicate sequences, resulting in the creation of BioGeoFormer-db (BGFdb) containing 1.9 million pathway-specific sequences.

#### 2. Model Architecture
BGF is built upon the ESM-2 model, which consists of 8 million parameters and 6 layers. A classification layer with 37 units was added to predict the biogeochemical pathway to which each gene belongs. BGF was trained using a 60/20/20 split for training, validation, and test datasets.

#### 3. Training and Optimization
BGF was trained for 1 epoch using the AdamW optimizer with an initial learning rate set to 0.0003. A dropout probability of 0.1 was applied during training to prevent overfitting. Various performance metrics, including Matthew's Correlation Coefficient (MCC), accuracy, precision, recall, and F1 score, were used to evaluate the model's performance.

#### 4. Confidence Calibration through Temperature Scaling
To adjust the prediction confidence of the model, temperature scaling was applied. This technique helps to mitigate model overconfidence and improve the accuracy of predictions. As a result, BGF was able to more accurately predict proteins with distant homologous relationships.

#### 5. Performance Evaluation and Comparison
BGF demonstrated superior performance compared to existing alignment-based methods such as DIAMOND and Hidden Markov Models (HMMs). Notably, BGF identified six times more genes than HMMs when detecting proteins with remote homologies. BGF has established itself as an important tool for uncovering the functions of genes related to microbial dark matter.


<br/>
# Results



BioGeoFormer(BGF)는 다양한 생물학적 경로에 대한 단백질 기능 예측을 위해 설계된 딥러닝 모델로, 여러 경쟁 모델과 비교하여 성능을 평가하였다. BGF는 20%에서 90%까지의 다양한 시퀀스 유사성 임계값에서 테스트 세트를 사용하여 성능을 평가하였다. 

1. **경쟁 모델**: BGF는 DIAMOND와 Hidden Markov Models(HMMs)와 비교되었다. DIAMOND는 개별 쌍 비교를 통해 높은 유사성에서 우수한 성능을 보였으나, BGF는 낮은 유사성(20%)에서 HMMs와 DIAMOND보다 더 나은 성능을 보였다. BGF는 20% 유사성에서 MCC(매튜 상관 계수)가 0.09, 정밀도가 0.15로 나타났고, HMMs는 각각 0.01과 0.04로 나타났다. 이는 BGF가 원거리 동종 유전자를 기능적으로 식별할 수 있는 잠재력을 보여준다.

2. **테스트 데이터**: BGF는 1,050,000개의 유전자를 생물지구화학적 기능으로 할당하였으며, 이 중 490,000개(46%)는 다른 접근 방식에서 '알 수 없는' 것으로 분류되었다. BGF는 평균적으로 HMMs 및 정렬 기반 접근 방식보다 6배 더 많은 유전자를 식별하였다.

3. **메트릭**: BGF는 모든 유사성 임계값에서 높은 성능 점수를 기록하였으며, 특히 90% 유사성에서 94-95%의 정확도를 보였다. 그러나 20% 유사성에서는 9-15%의 정확도를 기록하였다. BGF의 정밀도는 재현율보다 높았으며, 이는 모델이 상대적으로 적은 수의 거짓 긍정 결과를 생성했음을 나타낸다.

4. **비교**: BGF는 HMMs와 비교하여 모든 유사성 임계값에서 더 나은 성능을 보였으며, 특히 원거리 유사성에서 HMMs보다 우수한 성능을 발휘하였다. BGF는 특정 경로에서 더 많은 단백질을 예측하였으며, 예를 들어 유기 인산 에스터 가수분해 경로에서 BGF는 43,564개의 단백질을 예측한 반면, HMMs는 2,452개에 불과하였다.

이러한 결과는 BGF가 생물지구화학적 경로와 관련된 단백질 기능 예측에 있어 강력한 도구임을 입증하며, 미생물 다크 매터를 탐색하는 데 중요한 기여를 할 수 있음을 보여준다.

---




BioGeoFormer (BGF) is a deep learning model designed for predicting protein functions related to various biological pathways, and its performance was evaluated against several competing models. BGF was tested using a dataset across multiple sequence identity thresholds ranging from 20% to 90%.

1. **Competing Models**: BGF was compared with DIAMOND and Hidden Markov Models (HMMs). DIAMOND showed superior performance at high similarity levels through individual pairwise comparisons, but BGF outperformed both HMMs and DIAMOND at lower similarity levels (20%). At 20% similarity, BGF achieved a Matthews Correlation Coefficient (MCC) of 0.09 and a precision of 0.15, while HMMs recorded 0.01 and 0.04, respectively. This indicates BGF's potential to functionally identify remote homologous genes.

2. **Test Data**: BGF assigned biogeochemical functions to a total of 1,050,000 genes, with 490,000 (46%) classified as 'unknown' by other approaches. BGF identified, on average, six times more genes than HMMs and alignment-based methods.

3. **Metrics**: BGF returned high performance scores across all identity thresholds, particularly achieving 94-95% accuracy at the 90% identity split. However, at the 20% identity split, accuracy ranged from 9-15%. BGF's precision exceeded its recall, indicating that the model produced relatively few false positives.

4. **Comparison**: BGF outperformed HMMs at all identity thresholds, particularly demonstrating superior performance at remote similarities. BGF predicted significantly more proteins related to certain pathways; for instance, in the organic phosphoester hydrolysis pathway, BGF predicted 43,564 proteins compared to just 2,452 by HMMs.

These results demonstrate that BGF is a powerful tool for predicting protein functions related to biogeochemical pathways and can significantly contribute to exploring microbial dark matter.


<br/>
# 예제



이 논문에서는 BioGeoFormer(BGF)라는 단백질 언어 모델을 사용하여 미생물의 생화학적 경로와 관련된 유전자를 분류하는 방법을 제안합니다. BGF는 4개의 주요 생화학적 사이클(메탄, 황, 질소, 인)의 37개 경로 카테고리로 유전자를 분류하는 데 사용됩니다. 이 모델은 ESM2-8m이라는 사전 훈련된 단백질 언어 모델을 기반으로 하며, 이를 통해 미생물의 기능을 예측하는 데 도움을 줍니다.

#### 데이터셋 구성
1. **데이터셋 커리레이션**: 
   - MCycDB (메탄 관련 데이터베이스)
   - SCycDB (황 관련 데이터베이스)
   - NCycDB (질소 관련 데이터베이스)
   - PCycDB (인 관련 데이터베이스)
   
   이 데이터베이스들은 각각의 생화학적 경로에 따라 유전자를 수집하고, 이를 바탕으로 1.9M의 단백질 서열을 포함하는 BioGeoFormer-db(BGFdb)를 생성합니다.

2. **훈련 및 테스트 데이터 분할**:
   - CD-HIT를 사용하여 각 경로에 대해 20%에서 90%까지의 다양한 유사도 기준으로 클러스터링을 수행합니다.
   - 훈련 데이터(60%), 검증 데이터(20%), 테스트 데이터(20%)로 나누어 모델을 훈련합니다.

#### 모델 훈련
- BGF는 ESM2-8m 모델을 기반으로 하여 37개의 클래스(생화학적 경로)에 대한 분류 레이어를 추가합니다.
- 각 유사도 기준에 따라 별도의 모델을 훈련시키며, 훈련 과정에서 AdamW 옵티마이저와 드롭아웃을 사용하여 과적합을 방지합니다.

#### 테스트 및 성능 평가
- 테스트 데이터셋에 대해 BGF의 성능을 평가합니다. 예를 들어, 20% 유사도 기준에서 BGF는 0.15의 정밀도와 0.10의 재현율을 기록합니다.
- BGF는 HMMs 및 DIAMOND와 같은 기존 방법들과 비교하여 성능을 평가하며, 특히 원거리 유사도에서 더 나은 성능을 보입니다.

#### 예시
- **입력**: 특정 유전자 서열 (예: "ATGCGTACGTAGC...")
- **출력**: 해당 유전자가 속하는 생화학적 경로 (예: "메탄 사이클")

이러한 방식으로 BGF는 미생물의 기능을 예측하고, 미생물 다크 매터를 탐색하는 데 기여합니다.

---




In this paper, a protein language model called BioGeoFormer (BGF) is proposed to classify genes associated with biogeochemical cycles in microorganisms. BGF is used to classify genes into 37 defined pathway categories related to four major biogeochemical cycles (methane, sulfur, nitrogen, and phosphorus). This model is based on a pre-trained protein language model, ESM2-8m, which helps predict the functions of microorganisms.

#### Dataset Composition
1. **Dataset Curation**: 
   - MCycDB (methane-focused database)
   - SCycDB (sulfur-focused database)
   - NCycDB (nitrogen-focused database)
   - PCycDB (phosphorus-focused database)
   
   These databases collect genes based on their respective biogeochemical pathways, resulting in the creation of BioGeoFormer-db (BGFdb) containing 1.9 million protein sequences.

2. **Training and Testing Data Split**:
   - CD-HIT is used to cluster sequences based on various similarity thresholds from 20% to 90%.
   - The data is divided into training (60%), validation (20%), and testing (20%) sets for model training.

#### Model Training
- BGF is built on the ESM2-8m model with an added classification layer for 37 classes (biogeochemical pathways).
- Separate models are trained for each similarity threshold, using the AdamW optimizer and dropout to prevent overfitting.

#### Testing and Performance Evaluation
- The performance of BGF is evaluated on the test dataset. For example, at a 20% similarity threshold, BGF achieves a precision of 0.15 and a recall of 0.10.
- BGF is compared against existing methods like HMMs and DIAMOND, showing superior performance, especially at distant similarities.

#### Example
- **Input**: A specific gene sequence (e.g., "ATGCGTACGTAGC...")
- **Output**: The biogeochemical pathway to which the gene belongs (e.g., "Methane Cycle")

In this way, BGF contributes to predicting microbial functions and exploring microbial dark matter.

<br/>
# 요약


이 논문에서는 BioGeoFormer(BGF)라는 단백질 언어 모델을 사용하여 메탄, 질소, 황 및 인과 관련된 생물지구화학적 경로를 분류하는 방법을 제시합니다. BGF는 1.05M 유전자를 기능적으로 주석 처리하고, 기존의 HMM 및 정렬 기반 접근 방식보다 평균 6배 더 많은 유전자를 식별하여 높은 성능을 보였습니다. 이 모델은 심해의 차가운 샘 환경에서의 유전자 기능 예측에 성공적으로 적용되었습니다.

---

This paper presents a method using a protein language model called BioGeoFormer (BGF) to classify biogeochemical pathways related to methane, nitrogen, sulfur, and phosphorus. BGF annotated 1.05M genes functionally and demonstrated a performance that identified, on average, six times more genes than existing HMM and alignment-based approaches. The model was successfully applied to predict gene functions in cold seep environments.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Fig 1**: BioGeoFormer(BGF)의 개발 및 평가 파이프라인을 보여줍니다. 이 다이어그램은 데이터베이스의 구성과 BGF의 성능 평가 방법을 시각적으로 설명합니다.
   - **Fig 2**: BGF, HMMs, DIAMOND의 성능 메트릭스를 비교합니다. BGF는 모든 정체성 임계값에서 HMMs보다 우수한 성능을 보였으며, 특히 낮은 정체성에서 더 많은 진정한 양성을 탐지했습니다.
   - **Fig 3**: BGF의 혼동 행렬을 보여주며, 특정 사이클(예: 질산화)에서의 오분류를 강조합니다. 이는 BGF가 특정 경로에서 일관되게 잘못 분류된다는 것을 나타냅니다.
   - **Fig 4**: t-SNE 시각화로, BGF가 분류한 단백질의 고차원 임베딩을 2D로 표현합니다. 이 결과는 BGF가 생화학적 사이클을 잘 구분하고 있음을 보여줍니다.
   - **Fig 5**: BGF 모델의 온도 스케일링을 통한 신뢰도 분포 개선을 보여줍니다. 온도 스케일링 후, 모델의 신뢰도 분포가 더 현실적으로 조정되었습니다.
   - **Fig 6**: 다양한 모델 간의 주석 일치를 보여주는 UpSetR 플롯입니다. BGF는 719,686개의 단백질을 주석 처리했으며, HMMs는 362,133개로 두 모델 간의 일치가 두드러집니다.
   - **Fig 7**: 각 예측 방법에 따른 주석의 크기를 나타내는 버블 플롯입니다. BGF는 특정 경로(예: 유기 인산 에스터 가수분해)에서 다른 방법보다 훨씬 더 많은 단백질을 예측했습니다.

2. **테이블**
   - **Table 1**: BGF의 성능 메트릭스를 정체성 임계값에 따라 나열합니다. BGF는 90% 정체성에서 94-95%의 정확도를 보였으며, 20% 정체성에서는 10%의 정확도를 기록했습니다. 이는 BGF가 원거리 유사성에서 더 많은 단백질을 탐지할 수 있는 잠재력을 보여줍니다.

3. **어펜딕스**
   - 어펜딕스에는 데이터 커링, 모델 훈련 및 평가 방법에 대한 세부 정보가 포함되어 있습니다. BGF는 1.9M의 경로 특정 단백질을 사용하여 훈련되었으며, 이는 생화학적 사이클에 대한 새로운 통찰력을 제공합니다.

### Summary of Results and Insights

1. **Diagrams and Figures**
   - **Fig 1**: Illustrates the development and evaluation pipeline of BioGeoFormer (BGF). This diagram visually explains the construction of the database and the methods used to evaluate BGF's performance.
   - **Fig 2**: Compares performance metrics of BGF, HMMs, and DIAMOND. BGF outperformed HMMs across all identity thresholds, particularly excelling in detecting true positives at lower identity levels.
   - **Fig 3**: Displays the confusion matrix for BGF, highlighting consistent misclassifications in specific cycles (e.g., nitrification). This indicates that BGF has a tendency to misclassify certain pathways.
   - **Fig 4**: Shows t-SNE visualization of high-dimensional embeddings produced by BGF, demonstrating its ability to distinguish between biogeochemical cycles effectively.
   - **Fig 5**: Illustrates the improvement in confidence distribution of BGF models through temperature scaling. After scaling, the model's confidence distribution became more representative of true accuracy.
   - **Fig 6**: An UpSetR plot showing annotation agreement between different models. BGF annotated 719,686 proteins, while HMMs annotated 362,133, highlighting significant overlap between the two models.
   - **Fig 7**: A bubble plot representing the magnitude of annotations for each prediction method. BGF predicted significantly more proteins related to certain pathways (e.g., organic phosphoester hydrolysis) compared to other methods.

2. **Tables**
   - **Table 1**: Lists performance metrics of BGF across identity thresholds. BGF achieved 94-95% accuracy at 90% identity and only 10% at 20% identity, showcasing its potential to detect proteins at remote homologies.

3. **Appendix**
   - The appendix contains detailed information on data curation, model training, and evaluation methods. BGF was trained using 1.9 million pathway-specific proteins, providing new insights into biogeochemical cycles.

<br/>
# refer format:


### BibTeX 형식

```bibtex
@article{Wynne2025,
  author = {Jacob H. Wynne and Nima Azbijari and Andrew R. Thurber and Maude M. David},
  title = {BioGeoFormer: A deep learning approach to classify unknown genes associated with critical biogeochemical cycles},
  journal = {bioRxiv},
  year = {2025},
  month = {December},
  doi = {10.64898/2025.12.17.695047},
  url = {https://doi.org/10.64898/2025.12.17.695047}
}
```

### 시카고 스타일

Wynne, Jacob H., Nima Azbijari, Andrew R. Thurber, and Maude M. David. "BioGeoFormer: A Deep Learning Approach to Classify Unknown Genes Associated with Critical Biogeochemical Cycles." *bioRxiv*, December 18, 2025. https://doi.org/10.64898/2025.12.17.695047.
