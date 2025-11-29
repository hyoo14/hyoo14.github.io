---
layout: post
title:  "[2021]A semi-supervised learning framework for quantitative structure–activity regression modelling"
date:   2025-11-29 01:38:08 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 QSAR 모델링의 세 가지 주요 문제를 해결하기 위한 방법론을 제시하고, 반지도 학습 프레임워크를 통해 예측 정확도를 향상시키는 방법을 설명  
성능도 기존 경쟁 모델보다 더 올렸다고 함  


짧은 요약(Abstract) :


이 연구는 정량적 구조-활성 관계(QSAR) 모델링에서 세 가지 문제를 해결하는 방법을 제시합니다. 첫째, 특정 타겟에 대한 정보 내용을 비교하는 방법을 제공합니다. 둘째, 모델 예측의 정확도가 테스트 데이터와 훈련 데이터 간의 거리 함수에 따라 어떻게 저하되는지를 정량화하는 방법을 제시합니다. 셋째, 많은 훈련 데이터셋에 내재된 스크리닝 의존 선택 편향을 조정하는 방법을 제공합니다. 예를 들어, 가장 극단적인 경우에는 특정 활성도 기준을 통과한 화합물만 보고됩니다. 이 연구는 반지도 학습 프레임워크를 통해 (ii)와 (iii)를 결합하여, 테스트 화합물이 훈련 데이터와의 유사성을 고려하고 보고 선택 편향을 조정하는 예측을 수행할 수 있도록 합니다. 우리는 GlaxoSmithKline이 보고한 대규모 화합물 세트인 Tres Cantos AntiMalarial Set(TCAMS)을 사용하여 이 세 가지 방법을 설명합니다.





This work provides methods that solve three problems in quantitative structure–activity relationship (QSAR) modeling. First, it presents a method for comparing the information content between finite-dimensional representations of molecular structures with respect to the target of interest. Second, it quantifies how the accuracy of model predictions degrades as a function of the distance between the testing and training data. Third, it offers a method to adjust for screening-dependent selection bias inherent in many training datasets. For example, in the most extreme cases, only compounds that pass an activity-dependent screening threshold are reported. A semi-supervised learning framework combines (ii) and (iii) and can make predictions that take into account the similarity of the testing compounds to those in the training data and adjust for the reporting selection bias. We illustrate the three methods using publicly available structure–activity data for a large set of compounds reported by GlaxoSmithKline (the Tres Cantos AntiMalarial Set, TCAMS) to inhibit asexual in vitro Plasmodium falciparum growth.


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



이 논문에서는 정량적 구조-활성 관계(QSAR) 모델링을 위한 반지도 학습 프레임워크를 제안합니다. 이 프레임워크는 세 가지 주요 문제를 해결하는 방법을 제공합니다.

1. **정보 내용 비교 방법**: 다양한 분자 구조의 유한 차원 표현(지문) 간의 정보 손실을 정량화하는 방법을 제시합니다. 이는 특정 타겟에 대한 정보의 유용성을 평가하는 데 중요합니다. 예를 들어, 두 분자가 서로 다른 생물활성 프로파일을 가질 수 있지만, 선택된 지문에 따라 가까운 거리에서 위치할 수 있습니다.

2. **예측 정확도 저하의 거리 의존성**: 모델의 예측 정확도가 훈련 데이터와 테스트 데이터 간의 거리 함수로 어떻게 저하되는지를 정량화하는 방법을 제공합니다. 이는 모델의 적용 가능 영역(applicability domain)을 정의하는 데 중요한 요소입니다. 훈련 데이터와 유사한 화합물에 대해서만 모델을 제한하는 것이 일반적이지만, 일반적인 예측 목적을 위해서는 모델이 거리 의존성을 적절히 고려해야 합니다.

3. **선택 편향 조정 방법**: 많은 구조-활성 데이터셋은 특정 최소 타겟 활성도를 초과하는 분자만 포함되어 있어 선택 편향이 존재합니다. 이로 인해 새로운 분자의 활성 예측이 과도하게 낙관적일 수 있습니다. 반지도 학습 프레임워크는 이러한 선택 편향을 조정하고, 훈련 데이터와의 유사성을 고려하여 예측을 수행합니다.

이 프레임워크는 두 가지 데이터 소스를 사용합니다: 첫 번째는 레이블이 있는 구조-활성 데이터(Ln)로, 이는 특정 타겟에 대한 활성도를 가진 화합물로 구성됩니다. 두 번째는 레이블이 없는 구조 데이터(UN)로, 이는 활성도 정보가 없는 화합물로 구성됩니다. 이 두 데이터 소스를 결합하여 새로운 화합물의 활성도를 예측합니다.

모델은 레이블이 있는 데이터로 훈련된 후, 레이블이 없는 데이터의 정보를 사용하여 예측을 조정합니다. 이 과정에서 Tanimoto 거리 메트릭을 사용하여 분자 간의 유사성을 평가하고, 훈련 데이터와의 거리 정보를 통해 예측의 정확도를 조정합니다. 최종적으로, 이 프레임워크는 새로운 화합물이 특정 활성도 임계값을 초과할 확률을 추정하는 데 중점을 둡니다.




This paper proposes a semi-supervised learning framework for quantitative structure-activity relationship (QSAR) modeling, addressing three main issues.

1. **Method for Comparing Information Content**: It presents a method to quantify the information loss between different finite-dimensional representations (fingerprints) of molecular structures. This is crucial for assessing the utility of information related to a specific target. For instance, two molecules may have different bioactivity profiles but can be positioned close together in fingerprint space depending on the chosen fingerprint.

2. **Distance-Dependent Degradation of Predictive Accuracy**: The framework provides a method to quantify how the accuracy of model predictions degrades as a function of the distance between training and testing data. This is an important factor in defining the applicability domain of the model. While it is common to restrict models to compounds similar to those in the training set, it is desirable for the model to properly account for this distance-dependent effect for general predictive purposes.

3. **Adjustment for Selection Bias**: Many structure-activity datasets inherently contain selection bias, as they only include molecules that exceed a certain minimum target-specific activity. This can lead to overly optimistic predictions of activity for new molecules. The semi-supervised learning framework adjusts for this selection bias and makes predictions that consider the similarity of testing compounds to those in the training data.

The framework utilizes two sources of data: the first is labeled structure-activity data (Ln), which consists of compounds with known activity against a specific target. The second is unlabeled structure data (UN), which consists of compounds without activity information. By combining these two data sources, the framework predicts the activity of new compounds.

The model is trained on the labeled data and then adjusts predictions using information from the unlabeled data. In this process, the Tanimoto distance metric is employed to assess molecular similarity, and the distance information from the training data is used to calibrate the accuracy of predictions. Ultimately, the framework focuses on estimating the probability that a new compound exceeds a specified threshold of activity.


<br/>
# Results



이 논문에서는 정량적 구조-활성 관계(QSAR) 모델링을 위한 반지도 학습 프레임워크를 제안하고, 이를 통해 세 가지 주요 문제를 해결하고자 하였다. 연구의 결과는 다음과 같은 주요 요소로 구성된다.

1. **경쟁 모델**: 연구에서는 두 가지 주요 모델인 랜덤 포레스트(Random Forest)와 릿지 회귀(Ridge Regression)를 사용하여 QSAR 모델을 구축하였다. 이 두 모델은 각각의 데이터 세트에 대해 훈련되고 테스트되었으며, 반지도 학습 프레임워크(S3)와 전통적인 감독 학습 모델(S0) 간의 성능을 비교하였다.

2. **테스트 데이터**: 테스트 데이터는 TCAMS(Tres Cantos AntiMalarial Set)에서 파생된 13,533개의 화합물로 구성되며, 이 화합물들은 P. falciparum 3D7의 성장을 80% 이상 억제하는 것으로 선택되었다. 연구에서는 두 가지 테스트 세트를 사용하였으며, 하나는 7.5 pIC50 이상의 활성 화합물(237개)로 구성되었고, 다른 하나는 8.0 pIC50 이상의 활성 화합물(170개)로 구성되었다.

3. **메트릭**: 연구에서는 Tanimoto 거리(Tanimoto distance)를 사용하여 화합물 간의 유사성을 측정하였다. 이 메트릭은 화합물의 구조적 유사성을 정량화하는 데 적합하며, QSAR 모델의 성능을 평가하는 데 중요한 역할을 하였다.

4. **비교**: 연구 결과, 반지도 학습 프레임워크(S3)는 전통적인 감독 학습 모델(S0)보다 우수한 성능을 보였다. 특히, S3는 테스트 세트에서 더 많은 활성 화합물을 발견하는 데 성공하였으며, 이는 S0 모델이 예측한 값보다 더 높은 정확도를 나타냈다. S3는 또한 거리 의존적인 조정과 선택 편향을 고려하여 예측을 개선하였다. 

결과적으로, 반지도 학습 프레임워크는 QSAR 모델링에서 예측의 정확성을 높이고, 새로운 화합물의 활성 예측을 보다 신뢰할 수 있게 만드는 데 기여하였다.

---




This paper proposes a semi-supervised learning framework for quantitative structure-activity relationship (QSAR) modeling, aiming to address three major issues. The results of the study consist of the following key elements:

1. **Competing Models**: The study utilized two primary models, Random Forest and Ridge Regression, to construct QSAR models. These models were trained and tested on respective datasets, and the performance of the semi-supervised learning framework (S3) was compared against traditional supervised learning models (S0).

2. **Test Data**: The test data comprised 13,533 compounds derived from the Tres Cantos AntiMalarial Set (TCAMS), which were selected based on their ability to inhibit the growth of P. falciparum 3D7 by more than 80%. The study employed two testing sets: one consisting of active compounds with activity greater than 7.5 pIC50 (237 compounds) and another with activity greater than 8.0 pIC50 (170 compounds).

3. **Metrics**: The study used the Tanimoto distance as a metric to measure the similarity between compounds. This metric is suitable for quantifying structural similarity among compounds and played a crucial role in evaluating the performance of the QSAR models.

4. **Comparison**: The results indicated that the semi-supervised learning framework (S3) outperformed the traditional supervised learning model (S0). Specifically, S3 successfully identified a greater number of active compounds in the test set, demonstrating higher accuracy than the predictions made by the S0 model. S3 also improved predictions by accounting for distance-dependent adjustments and selection bias.

In conclusion, the semi-supervised learning framework contributes to enhancing the accuracy of predictions in QSAR modeling, making the predictions of the activity of novel compounds more reliable.


<br/>
# 예제



이 논문에서는 반감독 학습 프레임워크를 사용하여 정량적 구조-활성 관계(QSAR) 모델링을 수행하는 방법을 제시합니다. 이 방법은 세 가지 주요 문제를 해결하는 데 중점을 두고 있습니다: 

1. **정보 손실 비교**: 다양한 분자 구조의 유한 차원 표현(지문) 간의 정보 손실을 비교하는 방법을 제공합니다.
2. **예측 정확도 저하**: 테스트 데이터와 훈련 데이터 간의 거리 함수에 따라 모델 예측의 정확도가 어떻게 저하되는지를 정량화하는 방법을 제시합니다.
3. **선택 편향 조정**: 많은 훈련 데이터셋에서 발생하는 스크리닝 의존 선택 편향을 조정하는 방법을 제공합니다.

#### 예시

**트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋**

- **트레이닝 데이터 (L_n)**: 
  - 13,533개의 화합물로 구성되며, 이 화합물들은 P. falciparum 3D7의 성장을 80% 이상 억제하는 것으로 알려져 있습니다. 이 데이터는 각 화합물의 구조와 해당 화합물의 활성 값(pIC50)을 포함합니다. 예를 들어, 화합물 A는 pIC50 값이 7.5이고, 화합물 B는 pIC50 값이 8.0입니다.

- **테스트 데이터 (T)**: 
  - 테스트 데이터는 훈련 데이터에서 사용되지 않은 화합물로 구성되며, 이 화합물들은 pIC50 값이 7.5 이상인 237개의 화합물로 이루어져 있습니다. 예를 들어, 화합물 C는 pIC50 값이 8.5입니다.

**구체적인 테스크**

1. **모델 훈련**: 
   - 훈련 데이터 L_n을 사용하여 랜덤 포레스트 또는 릿지 회귀 모델을 훈련합니다. 이 모델은 화합물의 구조를 입력으로 받아 해당 화합물의 활성 값을 예측합니다.

2. **모델 예측**: 
   - 훈련된 모델을 사용하여 테스트 데이터의 화합물 C에 대한 예측을 수행합니다. 예를 들어, 모델이 화합물 C의 pIC50 값을 8.0으로 예측할 수 있습니다.

3. **예측 조정**: 
   - 테스트 화합물 C와 훈련 데이터 간의 Tanimoto 거리(d)를 계산하고, 이 거리를 기반으로 예측 값을 조정합니다. 예를 들어, 화합물 C가 훈련 데이터의 화합물 A와 가까운 경우, 예측 값이 더 높게 조정될 수 있습니다.

4. **결과 평가**: 
   - 최종적으로, 조정된 예측 값이 실제 활성 값과 얼마나 일치하는지를 평가합니다. 예를 들어, 화합물 C의 실제 pIC50 값이 8.5일 경우, 모델의 예측이 얼마나 정확한지를 평가합니다.




This paper presents a semi-supervised learning framework for quantitative structure-activity relationship (QSAR) modeling, focusing on solving three main problems:

1. **Comparison of Information Loss**: It provides a method for comparing the information loss between different finite-dimensional representations (fingerprints) of molecular structures.
2. **Degradation of Predictive Accuracy**: It quantifies how the accuracy of model predictions degrades as a function of the distance between testing and training data.
3. **Adjustment for Selection Bias**: It offers a method to adjust for the screening-dependent selection bias inherent in many training datasets.

#### Example

**Specific Inputs and Outputs of Training and Testing Data**

- **Training Data (L_n)**: 
  - Composed of 13,533 compounds known to inhibit the growth of P. falciparum 3D7 by more than 80%. This dataset includes the structure of each compound and its corresponding activity value (pIC50). For example, compound A has a pIC50 value of 7.5, and compound B has a pIC50 value of 8.0.

- **Testing Data (T)**: 
  - The testing data consists of compounds not used in the training set, specifically 237 compounds with pIC50 values greater than 7.5. For instance, compound C has a pIC50 value of 8.5.

**Specific Tasks**

1. **Model Training**: 
   - A random forest or ridge regression model is trained using the training data L_n. This model takes the structure of the compounds as input and predicts their activity values.

2. **Model Prediction**: 
   - The trained model is used to make predictions for the testing data compound C. For example, the model might predict a pIC50 value of 8.0 for compound C.

3. **Prediction Adjustment**: 
   - The Tanimoto distance (d) between testing compound C and the training data is calculated, and the predicted value is adjusted based on this distance. For instance, if compound C is close to training compound A, the predicted value may be adjusted upwards.

4. **Result Evaluation**: 
   - Finally, the adjusted predicted value is compared to the actual activity value. For example, if the actual pIC50 value of compound C is 8.5, the accuracy of the model's prediction is assessed.

<br/>
# 요약


이 논문에서는 QSAR 모델링의 세 가지 주요 문제를 해결하기 위한 방법론을 제시하고, 반지도 학습 프레임워크를 통해 예측 정확도를 향상시키는 방법을 설명합니다. 연구 결과, 제안된 방법이 기존의 감독 학습 모델보다 더 나은 예측 성능을 보였으며, Tres Cantos AntiMalarial Set(TCAMS) 데이터를 사용하여 그 효과를 입증했습니다. 이 연구는 화합물의 구조와 활성 간의 관계를 보다 정확하게 모델링할 수 있는 새로운 접근 방식을 제공합니다.




This paper presents methodologies to address three major issues in QSAR modeling and describes a semi-supervised learning framework to enhance predictive accuracy. The results demonstrate that the proposed methods outperform traditional supervised learning models, validated using the Tres Cantos AntiMalarial Set (TCAMS) data. This research offers a novel approach to more accurately model the relationship between compound structure and activity.

<br/>
# 기타
### 결과 및 인사이트 요약

#### 1. 다이어그램 및 피규어
- **Figure 1A**: TCAMS 데이터의 pIC50 값 분포를 보여줍니다. 이 분포는 정규 분포보다 긴 오른쪽 꼬리를 가지며, 이는 높은 활성을 가진 화합물이 상대적으로 적다는 것을 나타냅니다.
- **Figure 1B**: TCAMS 데이터의 두 가지 서로 다른 비트 길이(128비트 및 1024비트)에서의 Tanimoto 거리 분포를 비교합니다. 이는 서로 다른 지문 표현이 화합물 간의 유사성을 어떻게 다르게 나타내는지를 보여줍니다.
- **Figure 1C**: Tanimoto 거리 의존적인 활성의 공분산을 나타냅니다. 이 그래프는 거리와 활성 간의 관계를 시각적으로 나타내며, 가까운 화합물 간의 활성 값이 더 유사하다는 것을 보여줍니다.
- **Figure 1D**: 알려진 활성 화합물에 대한 거리 함수로서 무작위 화합물이 활성일 확률을 보여줍니다. 이는 거리와 활성 간의 관계를 정량화합니다.
- **Figure 1E**: 거리 의존적인 예측 정확도 감소를 나타내며, 이는 훈련 데이터와의 거리 증가에 따라 예측 정확도가 감소함을 보여줍니다.
- **Figure 1F**: 특정 화합물의 활성 확률을 거리 함수로 나타내며, 이는 예측 모델의 조정된 결과를 시각화합니다.

#### 2. 테이블
- **Table**: 다양한 예측 모델의 성능을 비교하는 테이블이 포함되어 있습니다. 각 모델의 예측 정확도와 활성 화합물 발견 비율을 보여줍니다. 이 테이블은 각 모델의 장단점을 명확히 하여, 연구자들이 최적의 모델을 선택하는 데 도움을 줍니다.

#### 3. 어펜딕스
- **Appendix**: 추가적인 데이터와 방법론적 세부사항이 포함되어 있습니다. 이 부분은 연구의 재현성을 높이고, 다른 연구자들이 이 방법론을 적용할 수 있도록 돕습니다.

---

### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Figure 1A**: Shows the distribution of pIC50 values in the TCAMS dataset. This distribution has a longer right tail than a normal distribution, indicating that there are relatively few compounds with high activity.
- **Figure 1B**: Compares the Tanimoto distance distributions for two different fingerprint lengths (128-bit and 1024-bit) in the TCAMS dataset. This illustrates how different fingerprint representations can affect the similarity between compounds.
- **Figure 1C**: Displays the distance-dependent covariance of activity, visually representing the relationship between distance and activity, showing that compounds that are closer together tend to have more similar activity values.
- **Figure 1D**: Illustrates the probability of a random compound being active as a function of its distance to known active compounds, quantifying the relationship between distance and activity.
- **Figure 1E**: Depicts the degradation of predictive accuracy as a function of distance, indicating that predictive accuracy decreases as the distance from the training data increases.
- **Figure 1F**: Visualizes the probability of activity for a specific compound as a function of distance, showcasing the adjusted results of the predictive model.

#### 2. Tables
- **Table**: Contains a comparison of the performance of various predictive models, showing the prediction accuracy and the rate of discovery of active compounds. This table clarifies the strengths and weaknesses of each model, aiding researchers in selecting the optimal model.

#### 3. Appendix
- **Appendix**: Includes additional data and methodological details. This section enhances the reproducibility of the research and assists other researchers in applying the methodology.

<br/>
# refer format:
### BibTeX 

```bibtex
@article{watson2020semi,
  title={A semi-supervised learning framework for quantitative structure–activity regression modelling},
  author={Watson, Oliver and Cortes-Ciriano, Isidro and Watson, James A.},
  journal={Bioinformatics},
  volume={37},
  number={3},
  pages={342--350},
  year={2021},
  publisher={Oxford University Press},
  doi={10.1093/bioinformatics/btaa711},
  url={https://academic.oup.com/bioinformatics/article/37/3/342/5890674}
}
```

### 시카고 스타일

Oliver Watson, Isidro Cortes-Ciriano, and James A. Watson. "A Semi-Supervised Learning Framework for Quantitative Structure–Activity Regression Modelling." *Bioinformatics* 37, no. 3 (2021): 342-350. https://doi.org/10.1093/bioinformatics/btaa711.
