---
layout: post
title:  "[2026]Evaluation Pitfalls and Sparsity Limitations in LLM-based Confidence Estimates for Classification"
date:   2026-07-16 21:38:07 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 LLM 기반의 분류에서 신뢰도 추정의 희소성 문제를 다루고, 이를 해결하기 위해 'verbalization logprobs'라는 새로운 방법을 제안합니다.  
verbalization logprobs란 구술화 로그 확률 방법, 구술화된 신뢰도 값을 토큰 확률로 가중치를 두어 희소성을 해결  
즉, 모델이 말한 신뢰도 숫자와 그 숫자 토큰들의 생성 확률을 함께 사용해 더 세밀한 신뢰도 점수를 계산하는 방법  
(일반 verbalization: 그대로 95%,  
verbalization logprobs: 십의 자리에서 9가 나올 확률과 다른 숫자들의 확률, 일의 자리에서 5가 나올 확률 등을 모두 반영해 94.7% 같은 연속적인 값을 계산 )  


짧은 요약(Abstract) :


이 논문의 초록에서는 대형 언어 모델(LLM)을 분류 작업에 사용할 때 신뢰도 추정의 중요성을 강조하고 있습니다. 신뢰도 추정은 예측이 신뢰할 수 있는지를 나타내며, 일반적인 접근 방식인 언어화(verbalization)는 매우 희소한 출력을 생성하는 경향이 있습니다. 예를 들어, Qwen3-32B 모델은 SST-2 데이터셋에서 단 8개의 고유 신뢰도 값을 생성하며, 그 중 절반 이상이 정확히 95%입니다. 이러한 희소성은 실제 유용성을 제한할 뿐만 아니라 평가에도 중대한 영향을 미칩니다. 정확도-거부 곡선(AUARC)에서의 보간(interpolation) 선택이 순위를 극적으로 변경할 수 있으며, 단계적 보간(stepwise interpolation)을 표준화할 것을 제안합니다. 공정한 평가 하에서, 우리는 언어화된 숫자를 토큰 확률로 가중치 부여하는 방법인 '언어화 로그확률(verbalization logprobs)'이 희소성을 해결하고 최상의 AUARC를 달성한다고 보고합니다.



The abstract of this paper emphasizes the importance of confidence estimation when using large language models (LLMs) for classification tasks. Confidence estimation indicates when predictions can be trusted, but common approaches like verbalization tend to produce extremely sparse outputs. For instance, the Qwen3-32B model verbalizes only eight unique confidence values on the SST-2 dataset, with over half being exactly 95%. This sparsity limits practical utility and critically affects evaluation, as the choice of interpolation in the area under the accuracy-rejection curve (AUARC) can dramatically alter rankings. The authors advocate for standardizing stepwise interpolation for fairer comparisons. Under such fair evaluation, they find that weighting verbalized digits by token probabilities—a method termed verbalization logprobs—addresses sparsity and achieves the best AUARC.


* Useful sentences :


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
* pitfall 함정  


<br/>
# Methodology



이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 분류 작업에서의 신뢰도 추정 방법에 대해 다루고 있습니다. 특히, 신뢰도 추정의 중요성과 함께 기존의 방법들이 가지는 한계, 특히 출력의 희소성 문제를 강조합니다. 다음은 이 연구에서 제안하는 주요 방법론입니다.

1. **신뢰도 추정 방법의 분류**: 연구에서는 신뢰도 추정 방법을 크게 세 가지로 분류합니다:
   - **구술화 기반 방법 (Verbalization-based methods)**: 모델이 자연어로 신뢰도를 표현하도록 유도하는 방법입니다. 예를 들어, 모델이 "신뢰도는 95%입니다"와 같은 형식으로 응답하도록 합니다.
   - **샘플링 기반 방법 (Sampling-based methods)**: 여러 번의 예측을 통해 신뢰도를 추정하는 방법입니다. 예를 들어, 모델이 여러 번 예측을 수행하고 가장 빈번하게 나타나는 예측을 신뢰도로 사용합니다.
   - **로짓 기반 방법 (Logit-based methods)**: 모델의 내부 상태를 활용하여 신뢰도를 추정하는 방법입니다. 예를 들어, 특정 클래스 레이블의 로그 확률을 사용하여 신뢰도를 계산합니다.

2. **희소성 문제**: 연구에서는 구술화 기반 방법이 매우 희소한 신뢰도 값을 생성한다는 점을 지적합니다. 예를 들어, Qwen3-32B 모델은 SST-2 데이터셋에서 단 8개의 고유한 신뢰도 값만을 생성하며, 이 중 절반 이상이 95%로 동일합니다. 이러한 희소성은 신뢰도 추정의 실용성을 제한하고, 평가 방법에도 중대한 영향을 미칩니다.

3. **평가 방법론**: 연구에서는 신뢰도 추정의 평가에서 보간(interpolation) 방법의 선택이 성능 순위에 큰 영향을 미친다는 점을 강조합니다. 특히, 선형 보간(linear interpolation) 대신 단계적 보간(stepwise interpolation)을 사용하는 것이 더 공정한 비교를 가능하게 한다고 주장합니다.

4. **구술화 로그 확률 (Verbalization Logprobs)**: 연구의 핵심 기여 중 하나는 구술화 로그 확률 방법입니다. 이 방법은 구술화된 신뢰도 값을 토큰 확률로 가중치를 두어 희소성을 해결하고, 더 나은 AUARC 점수를 달성합니다. 이 방법은 추가적인 추론 비용 없이 기존의 구술화 방법보다 성능을 향상시킵니다.

이 연구는 LLM 기반의 분류 작업에서 신뢰도 추정의 희소성 문제를 해결하기 위한 새로운 접근 방식을 제안하며, 평가 방법론의 중요성을 강조합니다.

---




This paper addresses confidence estimation methods based on large language models (LLMs) for classification tasks. It emphasizes the importance of confidence estimation and highlights the limitations of existing methods, particularly the issue of output sparsity. Below are the key methodologies proposed in this research.

1. **Classification of Confidence Estimation Methods**: The study categorizes confidence estimation methods into three main types:
   - **Verbalization-based methods**: These methods prompt the model to express confidence in natural language. For example, the model might respond with "The confidence is 95%."
   - **Sampling-based methods**: These methods estimate confidence by generating multiple predictions and using the most frequently occurring prediction as the confidence score. For instance, the model performs several predictions and counts the frequency of each class label.
   - **Logit-based methods**: These methods leverage the internal states of the model to estimate confidence. For example, they might use the log probability of a specific class label to calculate confidence.

2. **Sparsity Issue**: The research points out that verbalization-based methods produce extremely sparse confidence values. For instance, the Qwen3-32B model generates only eight unique confidence values on the SST-2 dataset, with more than half being exactly 95%. This sparsity limits the practical utility of confidence estimation and has significant implications for evaluation methods.

3. **Evaluation Methodology**: The study emphasizes that the choice of interpolation method in evaluating confidence estimation critically impacts performance rankings. It argues for the use of stepwise interpolation instead of linear interpolation to enable fairer comparisons.

4. **Verbalization Logprobs**: One of the key contributions of this research is the verbalization logprobs method. This approach addresses sparsity by weighting verbalized confidence values by their token probabilities, achieving better AUARC scores without incurring additional inference costs compared to traditional verbalization methods.

This research proposes a novel approach to solving the sparsity problem in confidence estimation for LLM-based classification tasks and underscores the importance of evaluation methodologies.


<br/>
# Results



이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 분류 작업에서의 신뢰도 추정의 문제점과 희소성 한계를 다루고 있습니다. 연구자들은 Qwen3-32B와 Claude 3.7 Sonnet이라는 두 가지 모델을 사용하여 여러 데이터셋(SST-2, SST-5, Amazon ESCI, Yahoo! Answers)에서 신뢰도 추정 방법을 평가했습니다.

#### 주요 결과
1. **모델 및 데이터셋**:
   - **모델**: Qwen3-32B, Claude 3.7 Sonnet
   - **데이터셋**: SST-2(영화 리뷰 감정 분석), SST-5(다양한 감정 분석), Amazon ESCI(제품-쿼리 관련성), Yahoo! Answers(주제 분류)

2. **신뢰도 추정 방법**:
   - **일반적인 방법**: Vanilla verbalization, Top-2 verbalization, Consistency sampling, Verbalization sampling, Token logprobs
   - **메트릭**: AUARC(정확도-거부 곡선 아래 면적), AUROC(수신자 조작 특성 곡선 아래 면적), ECE(예상 보정 오류)

3. **결과 비교**:
   - **Qwen3-32B**:
     - Vanilla verbalization: AUARC 0.758 (linear interpolation) → 0.713 (stepwise interpolation)
     - Consistency sampling: AUARC 0.808 (linear) → 0.683 (stepwise)
     - Token logprobs: AUARC 0.723 (변화 없음)
   - **Claude 3.7 Sonnet**:
     - Vanilla verbalization: AUARC 0.801 (linear) → 0.771 (stepwise)
     - Consistency sampling: AUARC 0.814 (linear) → 0.703 (stepwise)

4. **희소성 문제**:
   - 연구자들은 LLM이 특정 숫자 토큰을 선호하여 신뢰도 추정이 극도로 희소하다는 것을 발견했습니다. 예를 들어, Qwen3-32B는 SST-2에서 단 8개의 고유 신뢰도 값을 생성했습니다. 이로 인해 신뢰도 임계값 선택이 제한되어 실제 유용성이 감소합니다.

5. **제안된 방법**:
   - **Verbalization logprobs**: 이 방법은 신뢰도 값을 토큰 확률로 가중치를 두어 희소성을 해결하고, 기존의 vanilla verbalization보다 AUARC를 2.3 포인트 향상시켰습니다.

6. **결론**:
   - 신뢰도 추정의 희소성 문제는 LLM을 분류 작업에 활용하는 데 심각한 제한을 초래합니다. 따라서 평가 메트릭은 이러한 희소성을 반영하여 stepwise interpolation을 사용해야 하며, 이는 성능 점수를 인위적으로 부풀리는 linear interpolation의 문제를 해결합니다.




This paper addresses the pitfalls of confidence estimation and the sparsity limitations in classification tasks based on large language models (LLMs). The researchers evaluated two models, Qwen3-32B and Claude 3.7 Sonnet, across multiple datasets (SST-2, SST-5, Amazon ESCI, Yahoo! Answers).

#### Key Findings
1. **Models and Datasets**:
   - **Models**: Qwen3-32B, Claude 3.7 Sonnet
   - **Datasets**: SST-2 (sentiment analysis of movie reviews), SST-5 (varied sentiment analysis), Amazon ESCI (product-query relevance), Yahoo! Answers (topic classification)

2. **Confidence Estimation Methods**:
   - **Common Approaches**: Vanilla verbalization, Top-2 verbalization, Consistency sampling, Verbalization sampling, Token logprobs
   - **Metrics**: AUARC (Area Under the Accuracy-Rejection Curve), AUROC (Area Under the Receiver Operating Characteristic Curve), ECE (Expected Calibration Error)

3. **Results Comparison**:
   - **Qwen3-32B**:
     - Vanilla verbalization: AUARC 0.758 (linear interpolation) → 0.713 (stepwise interpolation)
     - Consistency sampling: AUARC 0.808 (linear) → 0.683 (stepwise)
     - Token logprobs: AUARC 0.723 (no change)
   - **Claude 3.7 Sonnet**:
     - Vanilla verbalization: AUARC 0.801 (linear) → 0.771 (stepwise)
     - Consistency sampling: AUARC 0.814 (linear) → 0.703 (stepwise)

4. **Sparsity Issue**:
   - The researchers found that LLMs tend to prefer certain numerical tokens, leading to extremely sparse confidence estimates. For instance, Qwen3-32B produced only 8 unique confidence values on SST-2, limiting the choice of confidence thresholds and reducing practical utility.

5. **Proposed Method**:
   - **Verbalization logprobs**: This method addresses sparsity by weighting verbalized confidence values by their token probabilities, achieving a 2.3 point improvement in AUARC over vanilla verbalization.

6. **Conclusion**:
   - The sparsity of confidence estimates poses a significant limitation on the practical use of LLMs for classification tasks. Therefore, evaluation metrics should account for this sparsity by using stepwise interpolation, which resolves the issues of artificially inflating performance scores associated with linear interpolation.


<br/>
# 예제



이 논문에서는 대형 언어 모델(LLM)을 사용한 분류 작업에서의 신뢰도 추정 방법에 대해 다루고 있습니다. 특히, 신뢰도 추정의 중요성과 함께, 기존의 방법들이 가지는 한계점인 '희소성' 문제를 강조하고 있습니다. 

#### 데이터셋 및 작업 설명

1. **데이터셋**:
   - **SST-2**: 영화 리뷰에 대한 감정 분석 데이터셋으로, 긍정(positive)과 부정(negative) 두 가지 레이블이 있습니다. 테스트 데이터는 872개의 샘플로 구성되어 있습니다.
   - **SST-5**: SST-2와 유사하지만, 더 세분화된 감정 레이블(매우 긍정적, 긍정적, 중립적 등)을 포함하고 있으며, 2,210개의 샘플로 구성되어 있습니다.
   - **Yahoo! Answers**: 다양한 주제에 대한 질문 제목을 분류하는 멀티 클래스 분류 작업으로, 10개의 클래스가 있으며, 6,000개의 샘플이 테스트에 사용됩니다.
   - **Amazon ESCI**: 제품 쿼리와 관련된 분류 작업으로, 4개의 클래스(정확한, 대체, 보완 등)가 있으며, 8,604개의 샘플이 테스트에 사용됩니다.

2. **작업**:
   - 각 데이터셋에 대해 LLM을 사용하여 주어진 입력(예: 영화 리뷰, 질문 제목 등)에 대해 감정 레이블 또는 주제 레이블을 예측합니다. 
   - 모델은 예측과 함께 신뢰도 점수를 출력해야 하며, 이 신뢰도 점수는 모델이 예측한 결과의 확신 정도를 나타냅니다.

#### 예시

- **입력**: "너무 많은 유머가 실패한다." (SST-2 데이터셋의 영화 리뷰)
- **출력**: 
  - **레이블**: "부정적" (negative)
  - **신뢰도**: "85%" (모델이 이 예측에 대해 85%의 확신을 가지고 있다는 의미)

이와 같은 방식으로 모델은 각 입력에 대해 레이블과 신뢰도를 출력하며, 이 신뢰도는 모델의 예측이 얼마나 신뢰할 수 있는지를 나타냅니다.




This paper discusses confidence estimation methods in classification tasks using large language models (LLMs). It emphasizes the importance of confidence estimation and highlights the limitation of existing methods, particularly the issue of 'sparsity'.

#### Dataset and Task Description

1. **Datasets**:
   - **SST-2**: A sentiment analysis dataset for movie reviews with two labels: positive and negative. The test set consists of 872 samples.
   - **SST-5**: Similar to SST-2 but includes more granular sentiment labels (very positive, positive, neutral, etc.) and consists of 2,210 samples.
   - **Yahoo! Answers**: A multi-class classification task for classifying question titles into various topics, with 10 classes and 6,000 samples used for testing.
   - **Amazon ESCI**: A classification task for product-query relevance with 4 classes (exact, substitute, complement, etc.) and 8,604 samples used for testing.

2. **Task**:
   - For each dataset, the LLM is tasked with predicting a sentiment label or topic label based on the given input (e.g., movie review, question title).
   - The model is required to output a confidence score along with its prediction, indicating the degree of certainty the model has in its prediction.

#### Example

- **Input**: "Too much of the humor falls flat." (A movie review from the SST-2 dataset)
- **Output**: 
  - **Label**: "Negative"
  - **Confidence**: "85%" (indicating that the model is 85% confident in this prediction)

In this manner, the model outputs a label and a confidence score for each input, where the confidence score reflects how trustworthy the model's prediction is.

<br/>
# 요약


이 논문에서는 LLM 기반의 분류에서 신뢰도 추정의 희소성 문제를 다루고, 이를 해결하기 위해 'verbalization logprobs'라는 새로운 방법을 제안합니다. 실험 결과, 이 방법은 기존의 'vanilla verbalization'보다 2.3 AUARC 포인트 향상되었으며, 추가적인 계산 비용 없이 신뢰도 추정의 희소성을 줄이는 데 효과적임을 보여주었습니다. 예를 들어, Qwen3-32B 모델을 사용한 실험에서, 'verbalization logprobs'는 630개의 고유 신뢰도 값을 생성하여 실용성을 높였습니다.

---

This paper addresses the issue of sparsity in confidence estimation for LLM-based classification and proposes a new method called 'verbalization logprobs' to mitigate this problem. Experimental results show that this method improves AUARC by 2.3 points over the existing 'vanilla verbalization' without incurring additional computational costs, effectively reducing the sparsity in confidence estimates. For instance, experiments using the Qwen3-32B model generated 630 unique confidence values with 'verbalization logprobs', enhancing practical utility.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1(a)**: SST-2 데이터셋에서 Qwen3-32B의 신뢰도 점수 히스토그램을 보여줍니다. 이 그래프는 신뢰도 점수가 0.6에서 0.98 사이의 단 8개의 고유 값으로 극도로 희소하다는 것을 나타냅니다. 특히, 95%의 신뢰도 값이 절반 이상을 차지하고 있습니다. 이는 신뢰도 추정의 실용성을 제한합니다.
  
- **Figure 1(b)**: 정확도-거부 곡선(Accuracy-Rejection Curve)을 보여줍니다. 이 곡선은 신뢰도 점수의 희소성으로 인해 선택할 수 있는 임계값이 제한적임을 나타냅니다. 98% 이상의 정확도를 달성하기 위해서는 39.7%의 예측을 거부해야 하며, 이는 97.9%의 정확도를 달성하기 위해 23.6%를 거부하는 것과 비교됩니다.

#### 2. 테이블
- **Table 2**: AUARC 점수와 순위에 대한 보간(interpolation) 방법의 영향을 보여줍니다. 선형 보간(linear interpolation)과 단계적 보간(stepwise interpolation) 간의 차이를 강조하며, 선형 보간이 성능 점수를 인위적으로 부풀리는 경향이 있음을 보여줍니다. 예를 들어, 일관성 샘플링(consistency sampling)은 선형 보간에서 1위였으나, 단계적 보간에서는 최하위로 떨어졌습니다.

- **Table 3**: 다양한 신뢰도 추정 방법의 성능을 비교합니다. 'Verbalization logprobs' 방법이 기존의 'Vanilla verbalization'보다 AUARC 점수를 2.3 포인트 향상시키며, 추가적인 계산 비용 없이 성능을 개선하는 것을 보여줍니다.

- **Table 4**: 'Vanilla verbalization'과 'Verbalization logprobs' 간의 신뢰도 점수의 상관관계를 보여줍니다. 두 방법 간의 상관계수(Spearman’s ρ)는 평균 0.89로, 두 방법이 높은 상관관계를 가지지만 여전히 차이가 있음을 나타냅니다.

#### 3. 어펜딕스
- **Appendix A**: 사용된 데이터셋에 대한 개요를 제공합니다. SST-2와 SST-5는 영화 리뷰에 대한 감정 분석을 다루며, Yahoo! Answers는 다중 클래스 주제 분류를, Amazon ESCI는 제품-쿼리 관련성을 테스트합니다.

- **Appendix B**: Claude 3.7 Sonnet 모델에 대한 결과를 제시하여, 다른 모델에서도 희소성 패턴이 지속됨을 확인합니다.

- **Appendix C**: 각 신뢰도 추정 방법의 상대적 비용을 계산하여, 샘플링 기반 접근 방식이 더 높은 비용을 초래함을 보여줍니다.

- **Appendix D**: 추론 모드가 활성화된 상태에서의 결과를 제시하여, 희소성 패턴이 여전히 존재함을 확인합니다.

- **Appendix E**: 0-9의 대체 신뢰도 범위를 사용하여 실험한 결과를 보여줍니다. 이 범위에서도 희소성이 지속되며, 선형 보간이 AUARC 점수를 부풀리는 경향이 있음을 확인합니다.

### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Figure 1(a)**: Displays a histogram of confidence scores for the SST-2 dataset with Qwen3-32B, showing extreme sparsity with only 8 unique values between 0.6 and 0.98. Notably, over half of the scores are exactly 95%, limiting the practical utility of confidence estimation.

- **Figure 1(b)**: Shows the accuracy-rejection curve, indicating limited threshold choices due to sparsity. Achieving at least 98% accuracy requires rejecting 39.7% of predictions, compared to 23.6% for 97.9% accuracy.

#### 2. Tables
- **Table 2**: Illustrates the impact of interpolation methods on AUARC scores and rankings. It highlights how linear interpolation tends to artificially inflate performance scores, with consistency sampling dropping from first to last place when switching from linear to stepwise interpolation.

- **Table 3**: Compares the performance of various confidence estimation methods, showing that the 'Verbalization logprobs' method improves AUARC by 2.3 points over 'Vanilla verbalization' without incurring additional computational costs.

- **Table 4**: Displays the correlation (Spearman’s ρ) between confidence scores of 'Vanilla verbalization' and 'Verbalization logprobs', averaging 0.89, indicating a high correlation but meaningful divergence.

#### 3. Appendix
- **Appendix A**: Provides an overview of the datasets used, including SST-2 and SST-5 for sentiment analysis, Yahoo! Answers for multi-class topic classification, and Amazon ESCI for product-query relevance.

- **Appendix B**: Presents results for the Claude 3.7 Sonnet model, confirming that sparsity patterns persist across different models.

- **Appendix C**: Calculates relative costs for each confidence estimation method, showing that sampling-based approaches incur higher costs.

- **Appendix D**: Shows results with reasoning mode enabled, confirming that sparsity patterns remain.

- **Appendix E**: Experiments with an alternative confidence range of 0-9, confirming that sparsity and inflation from linear interpolation persist.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Merdjanovska2026,
  author    = {Elena Merdjanovska and Omar Zaidan and Andreas Rücklé},
  title     = {Evaluation Pitfalls and Sparsity Limitations in LLM-based Confidence Estimates for Classification},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {33424--33435},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Elena Merdjanovska, Omar Zaidan, and Andreas Rücklé. "Evaluation Pitfalls and Sparsity Limitations in LLM-based Confidence Estimates for Classification." In *Findings of the Association for Computational Linguistics: ACL 2026*, 33424–33435. Association for Computational Linguistics, July 2026.
