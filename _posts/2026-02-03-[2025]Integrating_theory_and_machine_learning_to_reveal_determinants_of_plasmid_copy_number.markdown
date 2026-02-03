---
layout: post
title:  "[2025]Integrating theory and machine learning to reveal determinants of plasmid copy number"
date:   2026-02-03 19:03:48 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 11,338개의 플라스미드 데이터를 기반으로 다양한 유전적 특성을 포함한 머신러닝 모델(랜덤 포레스트 회귀 모델을)을 개발하여 플라스미드 복제 수(PCN)를 예측하였다.


짧은 요약(Abstract) :


이 논문은 플라스미드 복제 수(PCN)가 미생물의 진화, 항생제 내성 및 병원성에 미치는 중요한 영향을 다루고 있습니다. 플라스미드는 염색체 외부에 존재하는 이동 가능한 유전 요소로, 그 복제 수는 생태적, 진화적, 분자적 요인에 의해 결정됩니다. 연구자들은 플라스미드 크기와 복제 수 간의 경험적 파워-로우 관계를 설명하는 이론 모델을 제시하였으나, 이 관계만으로는 예측력이 제한적임을 지적합니다. 이를 개선하기 위해, 10,000개 이상의 플라스미드를 기반으로 다양한 특성을 포함한 데이터 기반 접근 방식을 도입하여 머신러닝 모델을 개발하였습니다. 이 모델은 플라스미드-인코딩 단백질 도메인이 주요 예측 인자로 부각되며, 메타유전체 플라스미드와 임상 분리주에서 PCN 분포를 포괄적으로 분석하여 생태적 적응과 특정 세균군의 PCN 핫스팟을 발견하였습니다. 이 연구는 플라스미드 생태학, 항생제 내성 유전자 감시 및 인간 미생물군의 "어두운 물질"에 대한 중요한 통찰을 제공합니다.



This paper addresses the significant impact of plasmid copy number (PCN) on microbial evolution, antibiotic resistance, and pathogenicity. Plasmids are extrachromosomal mobile genetic elements, and their copy numbers are determined by ecological, evolutionary, and molecular factors. The authors present a theoretical model to explain the empirical power-law relationship between plasmid size and copy number, but note that this relationship alone has limited predictive power. To improve this, they introduce a data-driven approach incorporating diverse features based on over 10,000 plasmids, developing a machine learning model. This model highlights plasmid-encoded protein domains as key predictors and conducts a comprehensive analysis of PCN distributions across metagenomic plasmids and clinical isolates, uncovering niche-specific taxonomic PCN hotspots and ecological adaptations. The findings provide critical insights into plasmid ecology, antibiotic resistance gene surveillance, and the "dark matter" of the human microbiome.


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



이 연구에서는 플라스미드 복제 수(PCN)를 예측하기 위해 두 가지 주요 접근 방식을 통합한 기계 학습 프레임워크를 개발했습니다. 첫 번째는 이론적 모델링을 통해 플라스미드 크기와 PCN 간의 관계를 설명하는 것이고, 두 번째는 데이터 기반 접근 방식을 통해 다양한 유전적 특성을 포함하여 PCN 예측의 정확성을 향상시키는 것입니다.

#### 1. 이론적 모델링
이론적 모델은 플라스미드의 생존과 복제 수가 두 가지 선택 압력, 즉 숙주 수준의 선택과 플라스미드 수준의 선택에 의해 형성된다는 가정을 기반으로 합니다. 숙주 수준의 선택은 플라스미드가 숙주 세포에 미치는 대사 부담을 최소화하려고 하며, 플라스미드 수준의 선택은 플라스미드의 자기 복제 및 전이 효율을 극대화하려고 합니다. 이 두 가지 상반된 힘의 상호작용을 통해 최적의 PCN이 결정됩니다.

이 모델은 수학적으로 표현되며, 플라스미드의 경쟁력은 총 DNA 양(플라스미드 크기 × PCN)으로 정의됩니다. 각 선택 압력의 비용 함수는 힐 함수로 유도되며, 이 두 비용 함수를 최소화하는 PCN 값을 찾는 최적화 문제로 다루어집니다.

#### 2. 기계 학습 프레임워크
기계 학습 프레임워크는 랜덤 포레스트 회귀 모델을 사용하여 다양한 유전적 및 플라스미드 수준의 특성을 통합하여 PCN을 예측합니다. 이 모델은 플라스미드에서 인코딩된 단백질 도메인에 중점을 두어, 기능 주석이 부족한 플라스미드에서도 예측의 신뢰성을 높입니다. 

- **특징 추출**: Prodigal을 사용하여 플라스미드의 단백질 코딩 서열(CDS)을 식별하고, Pfam 데이터베이스를 통해 도메인 주석을 수행합니다. 이 과정에서 11,533개의 고유한 Pfam 도메인이 식별되었습니다. PCN과 유의미한 상관관계를 가진 도메인을 찾기 위해 포인트-바이세리얼 상관 분석을 수행하고, 다중 테스트 보정을 통해 1,288개의 도메인을 최종적으로 선택했습니다.

- **모델 훈련**: 다양한 특징 조합을 테스트하여 PCN 예측의 정확성을 평가합니다. 모델은 4:1 비율로 훈련 및 테스트 세트로 나누어 훈련되며, R², MSE 및 스피어만 상관계수를 사용하여 성능을 평가합니다. 최종적으로, 플라스미드 도메인, k-머, 플라스미드 길이 및 숙주 염색체 길이를 포함한 전체 컨텍스트 모델과 플라스미드 중심 모델을 개발하여 다양한 상황에서 PCN을 예측할 수 있도록 합니다.

이러한 접근 방식은 플라스미드 복제 수 예측의 정확성을 크게 향상시키며, 다양한 생태계에서 플라스미드의 생물학적 역할을 이해하는 데 기여합니다.

---




In this study, we developed a machine learning framework that integrates two main approaches to predict plasmid copy number (PCN). The first approach involves theoretical modeling to explain the relationship between plasmid size and PCN, while the second employs a data-driven approach to enhance the accuracy of PCN predictions by incorporating diverse genetic features.

#### 1. Theoretical Modeling
The theoretical model is based on the assumption that the survival and copy number of plasmids are shaped by two levels of selective pressures: host-level selection and plasmid-level selection. Host-level selection aims to minimize the metabolic burden imposed by plasmids on the host cell, while plasmid-level selection seeks to maximize the self-replication and transfer efficiency of the plasmids. The optimal PCN emerges from the interplay of these opposing forces.

This model is mathematically formulated, where plasmid competitiveness is defined as the total DNA content (plasmid size × PCN). The cost functions for each level of selection are derived analytically as Hill functions, and the optimal PCN is determined by minimizing the combined cost function.

#### 2. Machine Learning Framework
The machine learning framework utilizes a random forest regression model to predict PCN by integrating various genomic and plasmid-level features. This model emphasizes plasmid-encoded protein domains, enhancing prediction reliability even for plasmids with sparse functional annotations.

- **Feature Extraction**: Prodigal is used to identify protein-coding sequences (CDSs) of plasmids, followed by domain annotation against the Pfam database. This process identifies 11,533 unique Pfam domains. To find domains significantly associated with PCN, point-biserial correlation analysis is performed, and multiple testing correction is applied to retain 1,288 domains.

- **Model Training**: Various feature combinations are systematically tested to assess their impact on predictive accuracy. The model is trained and evaluated through independent training-test splits at a 4:1 ratio, using R², MSE, and Spearman's correlation for performance evaluation. Ultimately, two models are developed: a full-context model incorporating plasmid domains, k-mers, plasmid length, and chromosomal length, and a plasmid-centric model using only plasmid-derived features for broader applicability.

These approaches significantly enhance the accuracy of plasmid copy number predictions and contribute to understanding the biological roles of plasmids across diverse ecosystems.


<br/>
# Results



이 연구에서는 플라스미드 복제 수(PCN)를 예측하기 위해 다양한 유전체 및 플라스미드 수준의 특징을 통합한 머신러닝 모델을 개발했습니다. 연구의 주요 결과는 다음과 같습니다.

1. **모델 개발**: 연구팀은 랜덤 포레스트 회귀 모델을 사용하여 PCN 예측을 위한 다중 특징 접근 방식을 설계했습니다. 이 모델은 플라스미드에서 인코딩된 단백질 도메인, 플라스미드 길이, 호스트 염색체 길이, k-머 빈도수 등 다양한 특징을 포함했습니다.

2. **테스트 데이터**: 모델은 11,338개의 플라스미드로 구성된 데이터셋을 사용하여 훈련되었습니다. 이 데이터셋은 4,317개의 프로카리오틱 게놈에서 수집된 것으로, 대부분이 박테리아에서 유래했습니다.

3. **성능 메트릭**: 모델의 성능은 R², 평균 제곱 오차(MSE), 스피어만 상관계수 등을 사용하여 평가되었습니다. 다양한 특징 조합을 테스트한 결과, 다중 특징 모델이 단일 특징 모델보다 일관되게 높은 R² 값을 기록했습니다. 특히, 플라스미드 도메인, 플라스미드 길이, 호스트 염색체 길이를 결합한 모델이 가장 높은 정확도를 보였습니다.

4. **비교 분석**: 연구팀은 다양한 특징 조합을 통해 모델의 예측 정확도를 평가했습니다. 단일 특징 모델은 중간 정도의 예측력을 보였으나, 다중 특징 모델은 높은 정확도를 달성했습니다. 예를 들어, 플라스미드 길이만을 사용한 모델은 제한된 예측력을 보였으며, 이는 다중 특징 접근 방식의 필요성을 강조합니다.

5. **결과의 유용성**: 이 연구의 결과는 플라스미드 복제 수 예측의 정확성을 크게 향상시켰으며, 플라스미드-인코딩 단백질 도메인이 PCN 조절에 중요한 역할을 한다는 것을 밝혔습니다. 이러한 접근 방식은 메타게놈 데이터와 같은 다양한 환경에서 플라스미드 복제 수를 예측하는 데 유용하게 활용될 수 있습니다.




In this study, a machine learning model was developed to predict plasmid copy number (PCN) by integrating various genomic and plasmid-level features. The main results of the study are as follows:

1. **Model Development**: The research team designed a random forest regression model that employs a multi-feature approach for PCN prediction. This model incorporates diverse features, including plasmid-encoded protein domains, plasmid length, host chromosomal length, and k-mer frequencies.

2. **Test Data**: The model was trained using a dataset consisting of 11,338 plasmids, collected from 4,317 prokaryotic genomes, the majority of which were derived from bacteria.

3. **Performance Metrics**: The model's performance was evaluated using R², Mean Squared Error (MSE), and Spearman's correlation coefficient. Various feature combinations were tested, and the results showed that multi-feature models consistently achieved higher R² values compared to single-feature models. Notably, the model combining plasmid domains, plasmid length, and chromosomal length achieved the highest accuracy.

4. **Comparative Analysis**: The research team assessed the predictive accuracy of the model through various feature combinations. Single-feature models exhibited moderate predictive power, while multi-feature models achieved high accuracy. For instance, a model using only plasmid length demonstrated limited predictive power, reinforcing the necessity of a multi-feature approach.

5. **Utility of Results**: The findings of this study significantly enhance the accuracy of PCN predictions and reveal that plasmid-encoded protein domains play a crucial role in PCN regulation. This approach can be effectively utilized for predicting plasmid copy numbers in diverse contexts, including metagenomic datasets.


<br/>
# 예제



이 논문에서는 플라스미드 복제 수(PCN)를 예측하기 위해 머신러닝 모델을 개발했습니다. 이 모델은 다양한 유전적 특성을 포함하여 훈련되었습니다. 훈련 데이터와 테스트 데이터의 구체적인 입력 및 출력, 그리고 작업의 세부 사항은 다음과 같습니다.

1. **데이터 수집**: 연구진은 11,338개의 플라스미드 시퀀스를 포함하는 데이터셋을 수집했습니다. 이 데이터셋은 4,317개의 프로카리오틱 유전체에서 유래하였으며, 각 플라스미드의 복제 수는 평균 시퀀싱 커버리지를 기반으로 추정되었습니다.

2. **특성 추출**: 각 플라스미드에 대해 다음과 같은 특성을 추출했습니다:
   - 플라스미드 길이
   - 호스트 염색체 길이 (가능한 경우)
   - 플라스미드에서의 k-머 빈도 (k=1, 2, 3)
   - 플라스미드에 인코딩된 단백질 도메인

3. **훈련 및 테스트 데이터 분할**: 데이터셋은 4:1 비율로 훈련 데이터와 테스트 데이터로 나누어졌습니다. 훈련 데이터는 모델을 학습하는 데 사용되었고, 테스트 데이터는 모델의 성능을 평가하는 데 사용되었습니다.

4. **모델 훈련**: 랜덤 포레스트 회귀 모델이 선택되어 다양한 특성 조합을 통해 훈련되었습니다. 예를 들어, 플라스미드 길이, 호스트 염색체 길이, k-머 빈도 및 플라스미드 도메인을 포함한 모델이 훈련되었습니다.

5. **출력**: 모델의 출력은 각 플라스미드의 예측된 복제 수(PCN)입니다. 예를 들어, 특정 플라스미드의 입력 특성이 다음과 같다고 가정해 보겠습니다:
   - 플라스미드 길이: 50 kb
   - 호스트 염색체 길이: 4 Mb
   - k-머 빈도: [0.1, 0.2, 0.3] (1-머, 2-머, 3-머 빈도)
   - 도메인: [도메인 A, 도메인 B]

   이 입력을 모델에 제공하면, 모델은 예측된 PCN을 출력합니다. 예를 들어, 예측된 PCN이 15라고 가정할 수 있습니다.

6. **모델 평가**: 테스트 데이터셋을 사용하여 모델의 성능을 평가했습니다. R² 값, 평균 제곱 오차(MSE), 스피어만 상관계수 등을 통해 모델의 예측 정확도를 측정했습니다.

이러한 방식으로, 연구진은 플라스미드 복제 수를 예측하는 데 있어 머신러닝 모델의 유용성을 입증했습니다.

---




In this paper, the authors developed a machine learning model to predict plasmid copy number (PCN). The model was trained using various genetic features. The specific inputs and outputs of the training and testing data, as well as the details of the tasks, are as follows:

1. **Data Collection**: The researchers collected a dataset comprising 11,338 plasmid sequences derived from 4,317 prokaryotic genomes. The copy numbers of each plasmid were estimated based on average sequencing coverage.

2. **Feature Extraction**: For each plasmid, the following features were extracted:
   - Plasmid length
   - Host chromosomal length (when available)
   - Plasmid k-mer frequencies (for k=1, 2, 3)
   - Plasmid-encoded protein domains

3. **Training and Testing Data Split**: The dataset was divided into training and testing data at a 4:1 ratio. The training data was used to train the model, while the testing data was used to evaluate the model's performance.

4. **Model Training**: A random forest regression model was selected and trained using various combinations of features. For example, a model incorporating plasmid length, host chromosomal length, k-mer frequencies, and plasmid domains was trained.

5. **Output**: The output of the model is the predicted copy number (PCN) for each plasmid. For instance, if the input features for a specific plasmid are as follows:
   - Plasmid length: 50 kb
   - Host chromosomal length: 4 Mb
   - k-mer frequencies: [0.1, 0.2, 0.3] (1-mer, 2-mer, 3-mer frequencies)
   - Domains: [Domain A, Domain B]

   The model would output a predicted PCN. For example, the predicted PCN might be 15.

6. **Model Evaluation**: The model's performance was evaluated using the testing dataset. Metrics such as R², mean squared error (MSE), and Spearman's correlation coefficient were used to measure the accuracy of the model's predictions.

Through this approach, the authors demonstrated the utility of machine learning models in predicting plasmid copy numbers.

<br/>
# 요약
이 연구에서는 11,338개의 플라스미드 데이터를 기반으로 다양한 유전적 특성을 포함한 머신러닝 모델을 개발하여 플라스미드 복제 수(PCN)를 예측하였다. 결과적으로, 플라스미드에 인코딩된 단백질 도메인이 PCN 예측의 주요 요소로 나타났으며, 이를 통해 생태계별 PCN 분포와 항생제 내성 유전자(ARG)와의 관계를 분석하였다. 이 연구는 플라스미드 생태학 및 ARG 감시를 위한 중요한 통찰력을 제공한다.

---

This study developed a machine learning model to predict plasmid copy number (PCN) based on a dataset of 11,338 plasmids, incorporating various genetic features. The results indicated that plasmid-encoded protein domains emerged as key predictors of PCN, allowing for the analysis of PCN distributions across ecosystems and their relationship with antibiotic resistance genes (ARGs). This research provides critical insights for plasmid ecology and ARG surveillance.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 플라스미드 크기와 복제 수(PCN) 간의 역 파워 법칙 관계를 설명하는 다이어그램. 이 다이어그램은 플라스미드 크기가 클수록 복제 수가 낮아지는 경향을 보여주며, 이는 호스트 수준의 선택과 플라스미드 수준의 선택 간의 상호작용에 의해 형성된다는 것을 나타냅니다.
   - **Figure 2**: 머신러닝 프레임워크를 통해 PCN을 예측하는 과정과 결과를 보여줍니다. 다양한 특징을 통합한 모델이 단일 특징 모델보다 높은 정확도를 보였으며, 이는 플라스미드의 복잡한 생물학적 특성을 반영합니다.
   - **Figure 3**: PCN과 관련된 플라스미드-인코딩 단백질 도메인의 분포를 시각화한 네트워크. 이 네트워크는 기능적으로 유사한 플라스미드 간의 관계를 보여주며, 복제 수에 따른 클러스터링 패턴을 확인할 수 있습니다.
   - **Figure 4**: 임상 플라스미드에서의 PCN과 항생제 내성 유전자(ARG) 분포를 보여주는 지리적 맵과 박스 플롯. 이는 특정 지역에서 높은 복제 수를 가진 플라스미드가 어떻게 분포하는지를 나타냅니다.
   - **Figure 5**: 다양한 생태계에서의 PCN 분포 패턴을 보여줍니다. 이는 플라스미드가 특정 환경에서 어떻게 다르게 나타나는지를 시각적으로 설명합니다.

2. **테이블**
   - 테이블은 플라스미드의 복제 수와 관련된 다양한 특성(예: 도메인, 길이 등)을 정리하여, 각 특성이 PCN에 미치는 영향을 비교합니다. 이를 통해 특정 도메인이 PCN 조절에 중요한 역할을 한다는 것을 확인할 수 있습니다.

3. **어펜딕스**
   - 어펜딕스에는 연구에 사용된 데이터셋, 분석 방법, 머신러닝 모델의 세부 사항 등이 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 유사한 분석을 수행할 수 있도록 돕습니다.




1. **Diagrams and Figures**
   - **Figure 1**: A diagram explaining the inverse power-law relationship between plasmid size and copy number (PCN). This figure illustrates the trend that larger plasmids tend to have lower copy numbers, shaped by the interplay between host-level selection and plasmid-level selection.
   - **Figure 2**: Shows the process and results of predicting PCN using a machine learning framework. The model that integrates various features outperformed single-feature models, reflecting the complex biological characteristics of plasmids.
   - **Figure 3**: A network visualizing the distribution of plasmid-encoded protein domains associated with PCN. This network highlights the relationships among functionally similar plasmids and allows for the examination of clustering patterns based on copy number.
   - **Figure 4**: A geographical map and box plots showing the distribution of PCN and antibiotic resistance genes (ARGs) in clinical plasmids. This illustrates how high-copy-number plasmids are distributed in specific regions.
   - **Figure 5**: Displays the distribution patterns of PCN across different ecosystems. This visually explains how plasmids manifest differently in specific environments.

2. **Tables**
   - Tables summarize various characteristics related to plasmid copy number (e.g., domains, length) and compare the impact of each characteristic on PCN. This helps confirm that specific domains play a crucial role in copy number regulation.

3. **Appendix**
   - The appendix includes datasets used in the study, analysis methods, and details of the machine learning models. This enhances the reproducibility of the research and assists other researchers in conducting similar analyses.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@article{Shahzadi2025,
  author = {Iqra Shahzadi and Wenzhi Xue and Hasan Ubaid Ullah and Rohan Maddamsetti and Lingchong You and Teng Wang},
  title = {Integrating theory and machine learning to reveal determinants of plasmid copy number},
  journal = {bioRxiv},
  year = {2025},
  month = {October},
  doi = {10.1101/2025.10.23.684078},
  url = {https://doi.org/10.1101/2025.10.23.684078}
}
```

### 시카고 스타일

Iqra Shahzadi, Wenzhi Xue, Hasan Ubaid Ullah, Rohan Maddamsetti, Lingchong You, and Teng Wang. "Integrating Theory and Machine Learning to Reveal Determinants of Plasmid Copy Number." *bioRxiv*, October 23, 2025. https://doi.org/10.1101/2025.10.23.684078.
