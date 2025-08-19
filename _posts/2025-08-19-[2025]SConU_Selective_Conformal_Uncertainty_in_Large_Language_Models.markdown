---
layout: post
title:  "[2025]SConU: Selective Conformal Uncertainty in Large Language Models"
date:   2025-08-19 21:44:26 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 선택적 비순응 불확실성(SConU)이라는 새로운 접근 방식을 제안하여, 대형 언어 모델의 불확실성 데이터 아웃라이어를 식별하고 제거하는 방법을 설명합니다.

SConU는 두 가지 정형 p-값을 개발하여 특정 위험 수준에서 주어진 샘플이 불확실성 분포에서 벗어나는지를 판단하는 유의성 검정을 구현

SConU는 불확실성 데이터 아웃라이어를 제거하여 예측 세트의 신뢰성을 높이고, 잘못된 커버리지 비율을 관리


추가 정보:

기존 conformal p value는 테스트 샘플의 불확실성을 calibration 샘플들의 불확실성과 단순 비교하여 비슷한 수준인지 확인하는 방식이다. 이 값은 테스트 샘플의 불확실성보다 크거나 같은 calibration 샘플의 비율로 계산된다. 그러나 calibration 샘플 중에는 정답을 포함하지 못한 경우도 있어서 그대로 사용하면 신뢰성이 떨어질 수 있다. 개선된 conformal p value는 calibration 샘플이 정답을 prediction set에 포함한 경우에만 비교 대상으로 삼는다. 이로 인해 불확실성 이상치를 더 엄격하고 안정적으로 걸러낼 수 있다.






---

기존 conformal p value는 테스트 샘플의 불확실성을 calibration 샘플과 비교하여 비슷한 수준인지 확인하는 방식이다. 계산식은 아래와 같다.

```markdown
p_{N+1} = (1 + Σ_{i=1}^N 1{u_i ≥ u_{N+1}}) / (N + 1)
```

여기서 $u_i$는 i번째 calibration 샘플의 불확실성, $u_{N+1}$은 테스트 샘플의 불확실성을 의미한다. 그러나 calibration 샘플 중 정답을 포함하지 못하는 경우도 있어 그대로 세면 신뢰성이 떨어진다.

이를 보완한 개선된 conformal p value는 정답을 prediction set에 포함한 calibration 샘플만 비교 대상으로 삼는다. 계산식은 다음과 같다.

```markdown
p'_{N+1} = (1 + Σ_{i=1}^N 1{u_i ≥ u_{N+1}, y_i* ∈ E(x_i, D_cal, α)}) / (N + 1)
```

따라서 개선된 방식은 불확실성 이상치를 더 엄격하고 안정적으로 판별할 수 있다.

---







짧은 요약(Abstract) :


이 논문에서는 대형 언어 모델(LLM)의 신뢰할 수 있는 배포를 위해 작업별 성능 보장을 제공하는 새로운 접근 방식인 선택적 정형 불확실성(Selective Conformal Uncertainty, SConU)을 제안합니다. 기존의 정형 불확실성 프레임워크는 교환 가능성 가정을 위반하는 불확실성 데이터 이상치를 식별하는 데 실패하여 예측 세트의 신뢰성을 저하시킵니다. SConU는 두 가지 정형 p-값을 개발하여 특정 위험 수준에서 주어진 샘플이 불확실성 분포에서 벗어나는지를 판단하는 유의성 검정을 구현합니다. 이 방법은 단일 도메인 및 다학제적 맥락에서의 잘못된 커버리지 비율을 엄격하게 관리할 수 있도록 하며, 예측의 효율성을 향상시킵니다. 또한, 고위험 질문-응답 작업에서 조건부 커버리지를 근사화하기 위한 정형 절차의 구성 요소를 포괄적으로 분석합니다.



This paper proposes a novel approach called Selective Conformal Uncertainty (SConU) to provide guarantees for task-specific performance in the reliable deployment of large language models (LLMs). Existing conformal uncertainty frameworks often fail to identify uncertainty data outliers that violate the exchangeability assumption, leading to unreliable prediction sets. SConU implements significance tests by developing two conformal p-values that determine whether a given sample deviates from the uncertainty distribution at a specific manageable risk level. This approach facilitates rigorous management of miscoverage rates across both single-domain and interdisciplinary contexts while enhancing the efficiency of predictions. Additionally, it comprehensively analyzes the components of the conformal procedures to approximate conditional coverage, particularly in high-stakes question-answering tasks.


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



이 논문에서 제안하는 방법은 "선택적 정형 불확실성(Selective Conformal Uncertainty, SConU)"이라고 불리며, 대규모 언어 모델(LLM)의 불확실성을 관리하기 위한 새로운 접근 방식을 제공합니다. SConU는 기존의 정형 예측(conformal prediction) 프레임워크의 한계를 극복하기 위해 설계되었습니다. 기존의 정형 예측 방법은 교환 가능성(exchangeability) 가정을 기반으로 하여 새로운 샘플에 대한 커버리지 보장을 제공하지만, 불확실성 데이터 아웃라이어를 식별하는 데 실패하여 잘못된 커버리지 비율(miscoverage rate)을 초래할 수 있습니다.

SConU는 두 가지 정형 p-값을 개발하여 특정 위험 수준에서 주어진 샘플이 보정 세트의 불확실성 분포에서 얼마나 벗어나는지를 판단합니다. 이 방법은 불확실성 데이터 아웃라이어를 필터링하여 단일 도메인 및 교차 도메인 QA 데이터셋에서의 잘못된 커버리지 비율을 엄격하게 관리할 수 있도록 합니다. 또한, SConU는 예측의 효율성을 향상시키고, 고위험 질문-응답 작업에서 조건부 커버리지(conditional coverage)를 근사화하는 데 중점을 둡니다.

SConU의 주요 구성 요소는 다음과 같습니다:

1. **정형 p-값의 개발**: SConU는 두 가지 정형 p-값을 사용하여 각 테스트 데이터 포인트의 불확실성 상태가 보정 데이터 분포에서 유의미하게 벗어나는지를 평가합니다. 이를 통해 교환 가능성 조건을 위반하는 샘플을 식별할 수 있습니다.

2. **불확실성 데이터 아웃라이어 필터링**: SConU는 불확실성 데이터 아웃라이어를 제거하여 예측 세트의 신뢰성을 높이고, 잘못된 커버리지 비율을 관리합니다.

3. **조건부 커버리지 근사화**: SConU는 다양한 예측 세트 크기에서 조건부 커버리지를 근사화하기 위해 여러 내부 구성 요소를 분석합니다. 이는 고위험 QA 시나리오에서의 성능을 향상시키는 데 기여합니다.

4. **실험적 검증**: SConU는 다양한 데이터셋에서 실험을 통해 그 효과를 검증하였으며, 기존의 정형 예측 방법보다 더 나은 성능을 보였습니다.

이러한 방법론은 대규모 언어 모델의 신뢰성과 효율성을 높이는 데 기여하며, 실제 응용 프로그램에서의 활용 가능성을 높입니다.

---




The method proposed in this paper is called "Selective Conformal Uncertainty (SConU)," which provides a new approach for managing uncertainty in large language models (LLMs). SConU is designed to overcome the limitations of existing conformal prediction frameworks. Traditional conformal prediction methods offer coverage guarantees for new samples based on the assumption of exchangeability, but they often fail to identify uncertainty data outliers, leading to unbounded miscoverage rates.

SConU implements two conformal p-values to assess whether a given sample significantly deviates from the uncertainty distribution of the calibration set at a specific risk level. This method allows for rigorous management of miscoverage rates across both single-domain and cross-domain QA datasets by filtering out uncertainty data outliers. Additionally, SConU focuses on enhancing the efficiency of predictions and approximating conditional coverage in high-stakes question-answering tasks.

The key components of SConU are as follows:

1. **Development of Conformal p-values**: SConU uses two conformal p-values to evaluate whether the uncertainty state of each test data point significantly deviates from the distribution of the calibration data. This helps identify samples that violate the exchangeability condition.

2. **Filtering of Uncertainty Data Outliers**: SConU removes uncertainty data outliers to enhance the reliability of prediction sets and manage miscoverage rates effectively.

3. **Approximation of Conditional Coverage**: SConU analyzes various internal components to approximate conditional coverage across different prediction set sizes, contributing to improved performance in high-risk QA scenarios.

4. **Experimental Validation**: SConU has been experimentally validated across various datasets, demonstrating better performance compared to traditional conformal prediction methods.

This methodology contributes to enhancing the trustworthiness and efficiency of large language models, increasing their applicability in real-world applications.


<br/>
# Results



이 논문에서는 대형 언어 모델(LLM)의 불확실성을 관리하기 위한 새로운 접근 방식인 선택적 정형 불확실성(Selective Conformal Uncertainty, SConU)을 제안합니다. SConU는 기존의 정형 예측(conformal prediction) 프레임워크의 한계를 극복하고, 불확실성 데이터 아웃라이어를 식별하여 예측의 신뢰성을 높이는 데 중점을 둡니다. 

#### 실험 결과
1. **경쟁 모델**: SConU는 여러 대형 언어 모델을 사용하여 실험을 수행했습니다. 여기에는 LLaMA, Qwen, OpenChat 등이 포함됩니다.
2. **테스트 데이터**: MMLU, MMLU-Pro, MedMCQA, TriviaQA, CoQA와 같은 다양한 데이터셋을 사용하여 모델의 성능을 평가했습니다.
3. **메트릭**: 주요 성능 지표로는 경험적 미커버리지 비율(Empirical Miscoverage Rate, EMR)과 평균 예측 세트 크기(Average Prediction Set Size, APSS)를 사용했습니다. EMR은 예측 세트가 통계적 보장을 충족하는지를 평가하며, APSS는 예측의 효율성을 측정합니다.
4. **비교**: SConU는 기존의 정형 불확실성 프레임워크와 비교하여 EMR과 APSS 모두에서 개선된 성능을 보였습니다. 예를 들어, SConU를 적용한 후 EMR이 사용자 지정 위험 수준을 초과하지 않도록 엄격하게 관리되었으며, APSS는 대부분의 테스트 샘플에서 1에 가까운 값을 기록하여 예측의 정확성을 높였습니다.

이러한 결과는 SConU가 대형 언어 모델의 불확실성을 효과적으로 관리하고, 다양한 도메인에서의 예측 성능을 향상시킬 수 있음을 보여줍니다. 특히, SConU는 교차 도메인 시나리오에서도 강력한 성능을 발휘하여, 모델이 다양한 질문에 대해 신뢰할 수 있는 예측을 제공할 수 있도록 합니다.

---




This paper introduces a novel approach called Selective Conformal Uncertainty (SConU) aimed at managing the uncertainty of large language models (LLMs). SConU focuses on overcoming the limitations of existing conformal prediction frameworks and enhancing the reliability of predictions by identifying uncertainty data outliers.

#### Experimental Results
1. **Competing Models**: Experiments were conducted using several large language models, including LLaMA, Qwen, and OpenChat.
2. **Test Data**: Various datasets such as MMLU, MMLU-Pro, MedMCQA, TriviaQA, and CoQA were utilized to evaluate the performance of the models.
3. **Metrics**: The primary performance metrics used were the Empirical Miscoverage Rate (EMR) and the Average Prediction Set Size (APSS). EMR assesses whether the prediction sets meet statistical guarantees, while APSS measures the efficiency of predictions.
4. **Comparison**: SConU demonstrated improved performance compared to existing conformal uncertainty frameworks in both EMR and APSS. For instance, after applying SConU, the EMR was rigorously managed to not exceed user-specified risk levels, and the APSS recorded values close to 1 for most test samples, indicating enhanced prediction accuracy.

These results illustrate that SConU effectively manages the uncertainty of large language models and improves prediction performance across various domains. Notably, SConU also exhibits strong performance in cross-domain scenarios, enabling the model to provide reliable predictions for diverse questions.


<br/>
# 예제



이 논문에서는 대규모 언어 모델(LLM)을 활용한 질문-답변(QA) 시스템에서의 불확실성 관리 방법인 선택적 정형 불확실성(Selective Conformal Uncertainty, SConU)을 제안합니다. SConU는 불확실성 데이터 아웃라이어를 식별하고, 이를 통해 예측의 신뢰성을 높이는 방법입니다. 이 방법은 두 가지 정형 p-값을 개발하여, 주어진 샘플이 특정 위험 수준에서 보정 세트의 불확실성 분포에서 얼마나 벗어나는지를 판단합니다.

#### 예시 1: MMLU 데이터셋
- **입력**: "What is penetration testing?"
  - 선택지: A: A procedure for testing libraries or other program components for vulnerabilities; B: Whole-system testing for security flaws and bugs; C: A security-minded form of unit testing that applies early in the development process; D: All of the above
- **출력**: B

#### 예시 2: MMLU-Pro 데이터셋
- **입력**: "In contrast to , aim to reward favourable behaviour by companies."
  - 선택지: A: Boycotts, Buyalls, Blockchain technology, Increased Sales; B: Buycotts, Boycotts, Digital technology, Decreased Sales; C: Boycotts, Buycotts, Digital technology, Decreased Sales; D: Buycotts, Boycotts, Blockchain technology, Charitable donations; E: Boycotts, Buyalls, Blockchain technology, Charitable donations; F: Boycotts, Buycotts, Digital technology, Increased Sales; G: Buycotts, Boycotts, Digital technology, Increased Sales; H: Boycotts, Buycotts, Physical technology, Increased Sales; I: Buycotts, Buyalls, Blockchain technology, Charitable donations; J: Boycotts, Buycotts, Blockchain technology, Decreased Sales
- **출력**: F

#### 예시 3: MedMCQA 데이터셋
- **입력**: "Kamlesh, a 2 year old girl, has Down’s syndrome. What is the risk of recurrence in subsequent pregnancies if the father is a balanced translocation carrier?"
  - 선택지: A: 100 %; B: 50 %; C: 25 %; D: 0 %
- **출력**: A

#### 예시 4: TriviaQA 데이터셋
- **입력**: "In 1968, who did radical feminist Valerie Solanas shoot and wound as he entered his New York studio?"
- **출력**: "Andy Warhol"

이러한 예시들은 SConU가 다양한 QA 태스크에서 어떻게 작동하는지를 보여줍니다. 각 데이터셋은 특정한 질문과 선택지를 제공하며, 모델은 가장 적절한 답변을 선택하여 출력합니다. SConU는 이러한 예측의 신뢰성을 높이기 위해 불확실성을 관리하고, 잘못된 예측을 줄이는 데 기여합니다.

---




This paper proposes a method for managing uncertainty in question-answering (QA) systems using large language models (LLMs), called Selective Conformal Uncertainty (SConU). SConU identifies uncertainty data outliers and enhances the reliability of predictions. It develops two conformal p-values to determine how much a given sample deviates from the uncertainty distribution of the calibration set at a specific risk level.

#### Example 1: MMLU Dataset
- **Input**: "What is penetration testing?"
  - Options: A: A procedure for testing libraries or other program components for vulnerabilities; B: Whole-system testing for security flaws and bugs; C: A security-minded form of unit testing that applies early in the development process; D: All of the above
- **Output**: B

#### Example 2: MMLU-Pro Dataset
- **Input**: "In contrast to , aim to reward favourable behaviour by companies."
  - Options: A: Boycotts, Buyalls, Blockchain technology, Increased Sales; B: Buycotts, Boycotts, Digital technology, Decreased Sales; C: Boycotts, Buycotts, Digital technology, Decreased Sales; D: Buycotts, Boycotts, Blockchain technology, Charitable donations; E: Boycotts, Buyalls, Blockchain technology, Charitable donations; F: Boycotts, Buycotts, Digital technology, Increased Sales; G: Buycotts, Boycotts, Digital technology, Increased Sales; H: Boycotts, Buycotts, Physical technology, Increased Sales; I: Buycotts, Buyalls, Blockchain technology, Charitable donations; J: Boycotts, Buycotts, Blockchain technology, Decreased Sales
- **Output**: F

#### Example 3: MedMCQA Dataset
- **Input**: "Kamlesh, a 2 year old girl, has Down’s syndrome. What is the risk of recurrence in subsequent pregnancies if the father is a balanced translocation carrier?"
  - Options: A: 100 %; B: 50 %; C: 25 %; D: 0 %
- **Output**: A

#### Example 4: TriviaQA Dataset
- **Input**: "In 1968, who did radical feminist Valerie Solanas shoot and wound as he entered his New York studio?"
- **Output**: "Andy Warhol"

These examples illustrate how SConU operates across various QA tasks. Each dataset provides specific questions and options, and the model selects the most appropriate answer for output. SConU contributes to enhancing the reliability of these predictions by managing uncertainty and reducing incorrect predictions.

<br/>
# 요약
이 논문에서는 선택적 비순응 불확실성(SConU)이라는 새로운 접근 방식을 제안하여, 대형 언어 모델의 불확실성 데이터 아웃라이어를 식별하고 제거하는 방법을 설명합니다. 실험 결과, SConU는 기존의 비순응 프레임워크보다 더 엄격한 정확성 보장을 제공하며, 다양한 도메인에서 예측 효율성을 향상시킵니다. 이 방법은 고위험 질문-응답 작업에서 조건부 커버리지를 근사화하는 데 효과적임을 보여줍니다.

---

This paper proposes a novel approach called Selective Conformal Uncertainty (SConU) to identify and eliminate uncertainty data outliers in large language models. Experimental results demonstrate that SConU provides stricter accuracy guarantees compared to existing conformal frameworks and enhances prediction efficiency across various domains. The method effectively approximates conditional coverage in high-stakes question-answering tasks.

<br/>
# 기타



이 논문에서는 SConU(Selective Conformal Uncertainty)라는 새로운 접근 방식을 제안하고 있습니다. 이 방법은 대규모 언어 모델에서 불확실성 데이터 아웃라이어를 식별하고 제거하여 예측의 신뢰성을 높이는 데 중점을 두고 있습니다. 다음은 논문에서 제시된 주요 결과와 인사이트입니다.

1. **EMR(경험적 미커버리지 비율) 결과**:
   - SConU 프레임워크를 적용한 후, EMR이 사용자 지정 위험 수준을 초과하지 않도록 엄격하게 관리되었습니다. 이는 SConU가 불확실성 데이터 아웃라이어를 효과적으로 제거하여 예측의 정확성을 높였음을 나타냅니다.
   - 예를 들어, MMLU-Pro 데이터셋에서 SConU를 적용한 결과, EMR이 0.2의 위험 수준을 초과하는 경우가 줄어들었습니다.

2. **APSS(평균 예측 세트 크기) 결과**:
   - SConU 프레임워크를 적용한 후 APSS가 1에 가까워졌습니다. 이는 대부분의 테스트 샘플에 대해 정확한 답변을 제공했음을 의미합니다. 
   - SConU-Pro를 적용한 경우, APSS가 더욱 개선되어 예측 효율성이 높아졌습니다.

3. **조건부 커버리지**:
   - SConU는 다양한 세트 크기에서 조건부 커버리지를 근사화하는 데 성공했습니다. 이는 고위험 QA 작업에서 개별 샘플의 정확성을 보장하는 데 중요한 요소입니다.

4. **최소 위험 수준**:
   - SConU는 원래의 보정 세트를 유지하면서 관리 가능한 최소 위험 수준을 도출했습니다. 이는 사용자가 지정한 위험 수준에 따라 예측의 신뢰성을 높이는 데 기여합니다.

5. **다양한 데이터셋에서의 성능**:
   - SConU는 MMLU, MMLU-Pro, MedMCQA, TriviaQA, CoQA와 같은 다양한 데이터셋에서 효과적으로 작동하여, 각 데이터셋의 특성에 맞춰 예측의 정확성을 높였습니다.

이러한 결과들은 SConU가 대규모 언어 모델의 불확실성을 관리하고, 예측의 신뢰성을 높이는 데 효과적임을 보여줍니다.

---




This paper introduces a novel approach called SConU (Selective Conformal Uncertainty), which focuses on identifying and removing uncertainty data outliers in large language models to enhance the reliability of predictions. Here are the key results and insights presented in the paper:

1. **EMR (Empirical Miscoverage Rate) Results**:
   - After applying the SConU framework, the EMR was strictly managed to not exceed user-specified risk levels. This indicates that SConU effectively removed uncertainty data outliers, thereby improving prediction accuracy.
   - For instance, in the MMLU-Pro dataset, the application of SConU significantly reduced instances where EMR exceeded the risk level of 0.2.

2. **APSS (Average Prediction Set Size) Results**:
   - Following the application of the SConU framework, the APSS approached 1, indicating that accurate answers were provided for most test samples.
   - The application of SConU-Pro further improved APSS, enhancing prediction efficiency.

3. **Conditional Coverage**:
   - SConU successfully approximated conditional coverage across various set sizes, which is crucial for ensuring the accuracy of individual samples in high-stakes QA tasks.

4. **Minimum Risk Level**:
   - SConU derived the minimum manageable risk level while maintaining the integrity of the original calibration set. This contributes to enhancing the reliability of predictions based on user-specified risk levels.

5. **Performance Across Various Datasets**:
   - SConU effectively operated across diverse datasets such as MMLU, MMLU-Pro, MedMCQA, TriviaQA, and CoQA, improving prediction accuracy tailored to the characteristics of each dataset.

These results demonstrate that SConU is effective in managing uncertainty in large language models and enhancing the reliability of predictions.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Wang2025,
  author    = {Zhiyuan Wang and Qingni Wang and Yue Zhang and Tianlong Chen and Xiaofeng Zhu and Xiaoshuang Shi and Kaidi Xu},
  title     = {SConU: Selective Conformal Uncertainty in Large Language Models},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {19052--19075},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  url       = {https://github.com/Zhiyuan-GG/SConU}
}
```

### 시카고 스타일

Wang, Zhiyuan, Qingni Wang, Yue Zhang, Tianlong Chen, Xiaofeng Zhu, Xiaoshuang Shi, and Kaidi Xu. 2025. "SConU: Selective Conformal Uncertainty in Large Language Models." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 19052–19075. July. Association for Computational Linguistics. https://github.com/Zhiyuan-GG/SConU.
