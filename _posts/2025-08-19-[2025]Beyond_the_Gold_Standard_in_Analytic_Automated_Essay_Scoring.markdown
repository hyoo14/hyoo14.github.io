---
layout: post
title:  "[2025]Beyond the Gold Standard in Analytic Automated Essay Scoring"
date:   2025-08-19 03:13:03 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 자동화된 에세이 평가(AES) 시스템의 신뢰성을 높이기 위해 다수의 채점자 데이터를 활용하는 새로운 접근 방식을 제안(평가자 간의 불일치의 원인을 탐구하고, 금본위 레이블 대신 개별 평가자로부터 학습하는 새로운 AES 시스템 설계를 제안)     

앙상블, 다중 레이블, 다중 작업 학습(에세이의 전반적인 점수와 함께 개별 특성 점수를 동시에 학습)으로 다중채점자 아키텍처 구성  


짧은 요약(Abstract) :


이 논문은 자동화된 에세이 채점(AES) 시스템의 발전을 다루고 있습니다. 기존의 AES 연구는 주로 전체적인 점수 부여에 초점을 맞추었으나, 이는 학생들에게 유용한 피드백을 제공하지 못합니다. 최근에는 에세이를 다양한 언어적 특성에 따라 평가하는 분석적 AES로의 전환이 이루어지고 있습니다. 이 접근법은 더 세부적인 피드백을 생성할 수 있는 가능성을 가지고 있지만, 인간 평가자 간의 불일치로 인해 편향이 발생할 수 있습니다. 현재의 AES 시스템은 평가자 간의 불일치를 단일 금본위(label)로 집계하는 경향이 있으며, 이는 실제 평가자 간의 변동성을 반영하지 못합니다. 따라서 이 연구는 평가자 간의 불일치의 원인을 탐구하고, 금본위 레이블 대신 개별 평가자로부터 학습하는 새로운 AES 시스템 설계를 제안합니다.



This paper addresses the advancement of Automated Essay Scoring (AES) systems. Previous AES research has primarily focused on holistic scoring, which fails to provide useful feedback to students. Recently, there has been a shift towards analytic AES, which evaluates essays according to various linguistic traits. This approach holds the potential for generating more detailed feedback but is susceptible to bias due to disagreements among human raters. The current AES systems tend to aggregate disagreements between raters into a single gold-standard label, which does not account for genuine variability among examiners. Therefore, this research aims to explore the sources of disagreements among raters and propose a novel AES system design that learns from individual raters instead of relying on gold-standard labels.


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



이 연구에서는 자동화된 에세이 채점(AES) 시스템을 개선하기 위해 여러 가지 방법론을 제안하고 있습니다. 특히, 기존의 금표준(gold standard) 레이블 대신 개별 채점자의 의견을 반영하는 새로운 모델 아키텍처를 개발하는 데 중점을 두고 있습니다. 이 과정에서 다음과 같은 주요 요소들이 포함됩니다.

1. **모델 선택**: 연구에서는 DeBERTa라는 사전 훈련된 언어 모델을 선택했습니다. DeBERTa는 뛰어난 성능을 보이며, AES 연구에서 성공적으로 사용된 바 있습니다. 이 모델은 다양한 하이퍼파라미터 설정을 통해 최적화됩니다.

2. **다중 채점자 아키텍처**: 연구에서는 세 가지 아키텍처를 제안합니다:    
   - **앙상블 모델**: 여러 개의 모델 출력을 결합하여 최종 예측을 생성합니다. 이는 각 모델의 강점을 활용하여 더 나은 성능을 도출할 수 있습니다.
   - **다중 레이블 모델**: 각 에세이에 대해 여러 특성을 동시에 평가할 수 있는 구조입니다. 이를 통해 다양한 평가 기준을 동시에 고려할 수 있습니다.
   - **다중 작업 학습(Multi-task Learning, MTL)**: 이 접근법은 여러 관련 작업의 정보를 활용하여 하나의 작업의 성능을 향상시키는 방법입니다. 예를 들어, 에세이의 전반적인 점수와 함께 개별 특성 점수를 동시에 학습할 수 있습니다.

3. **훈련 데이터**: 연구에서는 ELLIPSE 데이터셋을 사용합니다. 이 데이터셋은 6,482개의 에세이로 구성되어 있으며, 각 에세이는 두 명 이상의 채점자에 의해 채점되었습니다. 이 데이터셋은 다양한 특성(예: 응집력, 문법, 어휘 등)에 대해 채점된 정보를 포함하고 있습니다.

4. **훈련 방법**: 모델은 원시 다중 채점 데이터를 기반으로 훈련됩니다. 이는 기존의 금표준 레이블을 사용하는 대신, 채점자 간의 불일치를 신호로 활용하여 모델의 성능을 향상시키는 데 중점을 둡니다. 훈련 과정에서 RMSE(평균 제곱근 오차)와 같은 평가 지표를 사용하여 모델의 성능을 측정합니다.

5. **신뢰도 측정**: 모델의 출력에서 신뢰도를 추정하는 방법도 제안됩니다. 각 모델의 출력 간의 분산을 기반으로 신뢰도를 계산하여, 채점 결과의 신뢰성을 높이는 데 기여합니다.

이러한 방법론을 통해 연구자는 AES 시스템의 투명성과 신뢰성을 높이고, 학습자에게 보다 세분화된 피드백을 제공할 수 있는 가능성을 모색하고 있습니다.

---




This research proposes several methodologies to improve Automated Essay Scoring (AES) systems, focusing particularly on developing new model architectures that reflect individual raters' opinions instead of relying solely on gold standard labels. The key components of this approach include:

1. **Model Selection**: The study selects the DeBERTa pre-trained language model, which has demonstrated excellent performance and has been successfully utilized in AES research. This model is optimized through various hyperparameter settings.

2. **Multi-Rater Architectures**: The research proposes three different architectures:
   - **Ensemble Models**: These combine the outputs of multiple models to generate a final prediction. This approach leverages the strengths of each model to achieve better performance.
   - **Multi-Label Models**: This structure allows for the simultaneous evaluation of multiple traits for each essay, enabling consideration of various assessment criteria at once.
   - **Multi-Task Learning (MTL)**: This approach enhances the performance of one task by utilizing information from multiple related tasks. For instance, it can learn both the overall score of an essay and its individual trait scores simultaneously.

3. **Training Data**: The study utilizes the ELLIPSE dataset, which consists of 6,482 essays, each scored by two or more raters. This dataset includes information scored across various traits (e.g., cohesion, grammar, vocabulary).

4. **Training Method**: The models are trained on raw multi-rater data, focusing on using the disagreements between raters as signals to enhance model performance rather than relying on gold standard labels. The training process employs evaluation metrics such as RMSE (Root Mean Square Error) to measure model performance.

5. **Confidence Measurement**: The research also proposes methods for estimating the confidence of the model's outputs. By calculating the variance between the outputs of different models, it aims to enhance the reliability of the scoring results.

Through these methodologies, the researchers seek to improve the transparency and trustworthiness of AES systems, while exploring the potential to provide learners with more nuanced feedback.


<br/>
# Results



이 연구에서는 자동화된 에세이 평가(AES) 시스템의 성능을 향상시키기 위해 다양한 모델을 비교하고 평가하는 과정을 진행했습니다. 특히, ELLIPSE 데이터셋을 사용하여 다중 평가자 모델을 구축하고, 이 모델의 성능을 기존의 경쟁 모델과 비교했습니다.

#### 1. 경쟁 모델
연구에서는 DeBERTa 모델을 기본으로 설정하고, 여러 다른 사전 훈련된 모델들과 성능을 비교했습니다. 이 모델들은 각각의 하이퍼파라미터 설정에 따라 최적화되었으며, RMSE(평균 제곱근 오차), Pearson 상관계수, Spearman 순위 상관계수, 정확도, 정밀도, 재현율, F1 점수와 같은 다양한 메트릭을 사용하여 평가되었습니다.

#### 2. 테스트 데이터
테스트 데이터는 ELLIPSE 데이터셋에서 분리된 15%의 샘플로 구성되었습니다. 이 데이터셋은 6,482개의 에세이로 이루어져 있으며, 각 에세이는 최소 두 명의 평가자에 의해 평가되었습니다. 이 데이터는 다양한 주제와 난이도를 포함하고 있어, 모델의 일반화 능력을 평가하는 데 적합합니다.

#### 3. 메트릭
모델의 성능을 평가하기 위해 다음과 같은 메트릭을 사용했습니다:
- **RMSE**: 모델의 예측 점수와 실제 점수 간의 차이를 측정합니다.
- **Pearson 상관계수**: 두 변수 간의 선형 관계를 평가합니다.
- **Spearman 순위 상관계수**: 두 변수 간의 순위 관계를 평가합니다.
- **정확도**: 모델이 올바르게 예측한 비율을 나타냅니다.
- **정밀도**: 모델이 긍정으로 예측한 것 중 실제 긍정인 비율을 나타냅니다.
- **재현율**: 실제 긍정 중 모델이 긍정으로 예측한 비율을 나타냅니다.
- **F1 점수**: 정밀도와 재현율의 조화 평균으로, 두 메트릭의 균형을 평가합니다.

#### 4. 비교 결과
DeBERTa 모델은 다른 모델들에 비해 RMSE 점수에서 가장 낮은 값을 기록하며, 모든 평가 메트릭에서 우수한 성능을 보였습니다. 특히, RMSE는 2.308로 기록되었으며, 이는 모델의 예측이 실제 점수와 매우 근접함을 나타냅니다. 다른 모델들은 상대적으로 높은 RMSE 값을 기록하였으며, 이는 이들 모델이 DeBERTa에 비해 성능이 떨어짐을 의미합니다.

이러한 결과는 DeBERTa 모델이 AES 시스템에서 효과적으로 활용될 수 있음을 시사하며, 향후 연구에서 이 모델을 기반으로 한 다양한 접근 방식이 필요함을 강조합니다.

---




In this study, we conducted a systematic evaluation of various models to enhance the performance of Automated Essay Scoring (AES) systems. Specifically, we built multi-rater models using the ELLIPSE dataset and compared their performance against existing competitive models.

#### 1. Competitive Models
The study set DeBERTa as the baseline model and compared it with several other pre-trained models. These models were optimized based on different hyperparameter settings, and their performance was evaluated using various metrics, including RMSE (Root Mean Square Error), Pearson correlation coefficient, Spearman rank correlation coefficient, accuracy, precision, recall, and F1 score.

#### 2. Test Data
The test data consisted of 15% of samples separated from the ELLIPSE dataset. This dataset comprises 6,482 essays, each evaluated by at least two raters. It includes a variety of topics and difficulty levels, making it suitable for assessing the generalization ability of the models.

#### 3. Metrics
The following metrics were used to evaluate the performance of the models:
- **RMSE**: Measures the difference between the predicted scores and the actual scores.
- **Pearson Correlation Coefficient**: Assesses the linear relationship between two variables.
- **Spearman Rank Correlation Coefficient**: Evaluates the rank relationship between two variables.
- **Accuracy**: Represents the proportion of correct predictions made by the model.
- **Precision**: Indicates the proportion of true positives among the predicted positives.
- **Recall**: Represents the proportion of true positives that were predicted as positive by the model.
- **F1 Score**: The harmonic mean of precision and recall, assessing the balance between the two metrics.

#### 4. Comparison Results
The DeBERTa model recorded the lowest RMSE score compared to other models, demonstrating superior performance across all evaluation metrics. Specifically, the RMSE was recorded at 2.308, indicating that the model's predictions were very close to the actual scores. Other models recorded relatively higher RMSE values, suggesting that their performance was inferior to that of DeBERTa.

These results indicate that the DeBERTa model can be effectively utilized in AES systems and highlight the need for various approaches based on this model in future research.


<br/>
# 예제



이 연구에서는 자동화된 에세이 채점(AES) 시스템을 개발하기 위해 다양한 데이터셋을 사용하여 모델을 훈련하고 평가하는 과정을 설명합니다. 이 과정은 크게 세 가지 단계로 나눌 수 있습니다: 데이터 준비, 모델 훈련, 그리고 평가입니다.

1. **데이터 준비**:
   - **트레이닝 데이터**: ELLIPSE 데이터셋을 사용합니다. 이 데이터셋은 6,482개의 에세이로 구성되어 있으며, 각 에세이는 최소 두 명의 평가자에 의해 채점되었습니다. 각 에세이는 다음과 같은 여섯 가지 분석적 특성(코히전, 구문, 어휘, 구문구성, 문법, 규범)에 대해 1에서 5까지의 점수로 평가됩니다.
   - **테스트 데이터**: 트레이닝 데이터의 15%를 테스트 데이터로 사용합니다. 이 데이터는 모델의 성능을 평가하는 데 사용됩니다.

2. **모델 훈련**:
   - **모델 선택**: DeBERTa라는 사전 훈련된 언어 모델을 선택합니다. 이 모델은 자연어 처리(NLP) 작업에서 우수한 성능을 보이는 것으로 알려져 있습니다.
   - **훈련 과정**: 모델은 트레이닝 데이터에 대해 훈련됩니다. 이 과정에서 모델은 각 에세이에 대해 예측 점수를 생성합니다. 예를 들어, 특정 에세이에 대해 모델이 "코히전" 특성에 대해 4.0, "구문" 특성에 대해 3.5, "어휘" 특성에 대해 4.5를 예측할 수 있습니다.

3. **모델 평가**:
   - **성능 평가**: 테스트 데이터에 대해 모델의 예측 점수를 실제 평가자 점수와 비교하여 모델의 성능을 평가합니다. 평가 지표로는 RMSE(평균 제곱근 오차), Pearson 상관계수, Spearman 순위 상관계수 등이 사용됩니다.
   - **예시**: 만약 모델이 특정 에세이에 대해 "코히전" 특성에 대해 4.0을 예측했지만 실제 평가자 점수가 3.0이었다면, 이 경우 모델의 예측이 부정확하다는 것을 의미합니다. 이러한 예측과 실제 점수 간의 차이를 통해 모델의 성능을 분석하고 개선할 수 있는 기회를 찾습니다.

이러한 과정을 통해 연구자는 AES 시스템의 신뢰성과 투명성을 높이고, 교육적 피드백을 제공하는 데 기여하고자 합니다.

---




This research describes the process of developing an Automated Essay Scoring (AES) system using various datasets to train and evaluate the model. This process can be divided into three main stages: data preparation, model training, and evaluation.

1. **Data Preparation**:
   - **Training Data**: The ELLIPSE dataset is used, which consists of 6,482 essays, each scored by at least two raters. Each essay is evaluated on six analytic traits (Cohesion, Syntax, Vocabulary, Phraseology, Grammar, and Conventions) on a scale from 1 to 5.
   - **Test Data**: 15% of the training data is used as test data. This data is used to evaluate the performance of the model.

2. **Model Training**:
   - **Model Selection**: A pre-trained language model called DeBERTa is chosen. This model is known for its excellent performance in natural language processing (NLP) tasks.
   - **Training Process**: The model is trained on the training data. During this process, the model generates predicted scores for each essay. For example, the model might predict a score of 4.0 for the "Cohesion" trait, 3.5 for "Syntax," and 4.5 for "Vocabulary" for a specific essay.

3. **Model Evaluation**:
   - **Performance Evaluation**: The model's predicted scores are compared to the actual scores given by raters on the test data to evaluate the model's performance. Evaluation metrics include RMSE (Root Mean Square Error), Pearson correlation coefficient, and Spearman rank correlation coefficient.
   - **Example**: If the model predicts a score of 4.0 for the "Cohesion" trait for a specific essay, but the actual rater score is 3.0, this indicates that the model's prediction is inaccurate. Analyzing the differences between these predictions and actual scores provides opportunities to identify areas for improvement in the model.

Through this process, the researchers aim to enhance the reliability and transparency of the AES system and contribute to providing educational feedback.

<br/>
# 요약

이 연구에서는 자동화된 에세이 평가(AES) 시스템의 신뢰성을 높이기 위해 다수의 채점자 데이터를 활용하는 새로운 접근 방식을 제안하였다. 연구 결과, 다수의 채점자 모델이 기존의 금표준 레이블보다 더 나은 성능을 보였으며, 채점자 간의 불일치를 활용하여 피드백의 질을 향상시킬 수 있음을 보여주었다. 이 방법은 교육 현장에서의 AES 시스템의 투명성과 신뢰성을 높이는 데 기여할 것으로 기대된다.

---

This study proposes a novel approach to enhance the reliability of Automated Essay Scoring (AES) systems by leveraging data from multiple raters. The results indicate that multi-rater models outperform traditional gold-standard labels, demonstrating the potential to improve feedback quality by utilizing rater disagreements. This method is expected to contribute to the transparency and trustworthiness of AES systems in educational settings.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 이 피규어는 두 개의 에세스가 세 명의 평가자에 의해 다중 채점된 후 평균을 내어 금 표준 레이블을 생성하는 과정을 보여줍니다. 이 과정에서 각 에세스의 점수 변동성을 기반으로 신뢰도 점수를 계산하는 방법을 설명합니다. 이는 평가자 간의 불일치를 단일 금 표준 레이블로 통합하는 기존 접근 방식의 한계를 강조합니다.
   - **Figure 2**: 이 다이어그램은 제안된 다중 평가자 AES 모델의 구조를 보여줍니다. 앙상블, 다중 레이블, 다중 작업 학습(MTL) 아키텍처를 통해 각 특성에 대한 모델을 구축하는 방법을 설명합니다. 이는 평가자 간의 불일치를 활용하여 더 나은 예측을 가능하게 합니다.

2. **테이블**
   - **Table 1**: 다양한 분석 AES 데이터셋을 비교한 표로, 각 데이터셋의 에세이 유형, 작가 정보, 에세이 수, 분석 특성, 평가자 수, 다중 채점 가능 여부, 점수 범위를 나열합니다. 이 표는 연구자들이 데이터셋을 선택할 때 고려해야 할 요소들을 명확히 하여, 데이터셋의 다양성과 특성을 이해하는 데 도움을 줍니다.
   - **Table 2**: CLC FCE 데이터셋에서 다양한 모델의 하이퍼파라미터 설정을 보여줍니다. 각 모델의 성능을 비교하여 DeBERTa 모델이 다른 모델들보다 우수한 성능을 보였음을 나타냅니다. 이는 AES 시스템의 성능 향상을 위한 모델 선택에 중요한 정보를 제공합니다.

3. **어펜딕스**
   - **Appendix A**: 분석 AES 데이터셋에 대한 자세한 설명과 각 데이터셋의 특성을 나열합니다. 이는 연구자들이 데이터셋을 이해하고 활용하는 데 필요한 정보를 제공합니다.
   - **Appendix B**: DeBERTa 모델을 선택한 이유와 그 성능을 평가한 방법을 설명합니다. 이 정보는 모델 선택의 근거를 제공하며, 향후 연구에서 유사한 접근 방식을 사용할 수 있는 기초 자료가 됩니다.
   - **Appendix C**: 실험 방법론을 상세히 설명하며, 재현 가능성을 높이기 위한 절차를 제시합니다. 이는 연구의 신뢰성을 높이고, 다른 연구자들이 동일한 방법을 적용할 수 있도록 돕습니다.




1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates the process of two essays being multi-marked by three raters, leading to the creation of a gold standard label through averaging. It highlights the method of calculating confidence scores based on the variance of scores, emphasizing the limitations of aggregating rater disagreements into a single gold standard label.
   - **Figure 2**: This diagram presents the structure of the proposed multi-rater AES models. It outlines how ensemble, multi-label, and multi-task learning architectures will be constructed for each trait, showcasing the potential to leverage rater disagreements for improved predictions.

2. **Tables**
   - **Table 1**: A comparative table of various analytic AES datasets, detailing essay types, writer information, number of essays, analytic traits, number of raters, availability of multi-marks, and score ranges. This table clarifies the factors researchers should consider when selecting datasets, aiding in understanding the diversity and characteristics of the datasets.
   - **Table 2**: This table presents the hyper-parameter settings for different models evaluated on the CLC FCE dataset. It shows that the DeBERTa model outperforms other models across all evaluation metrics, providing critical information for model selection aimed at enhancing AES system performance.

3. **Appendices**
   - **Appendix A**: Provides detailed descriptions of the analytic AES datasets, listing their characteristics. This information is essential for researchers to understand and utilize the datasets effectively.
   - **Appendix B**: Explains the rationale behind selecting the DeBERTa model and the methods used to evaluate its performance. This serves as a foundation for model selection and can guide similar approaches in future research.
   - **Appendix C**: Outlines the experimental methodology in detail, enhancing reproducibility and providing a clear procedure for conducting the experiments. This increases the reliability of the research and assists other researchers in applying the same methods.

<br/>
# refer format:
### BibTeX Entry

```bibtex
@inproceedings{Gaudeau2025,
  author    = {Gabrielle Gaudeau},
  title     = {Beyond the Gold Standard in Analytic Automated Essay Scoring},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)},
  pages     = {18--39},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  address   = {July 28-29, 2025}
}
```

### Chicago Style Citation

Gabrielle Gaudeau. 2025. "Beyond the Gold Standard in Analytic Automated Essay Scoring." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)*, 18–39. Association for Computational Linguistics. July 28-29, 2025.
