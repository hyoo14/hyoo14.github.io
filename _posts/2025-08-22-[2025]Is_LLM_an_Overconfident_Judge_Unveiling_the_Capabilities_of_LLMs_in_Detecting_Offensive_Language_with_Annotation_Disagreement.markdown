---
layout: post
title:  "[2025]Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement"
date:   2025-08-22 01:29:10 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

연구 결과, LLM은 낮은 합의 수준의 샘플에서 성능이 저하되고, 이러한 모호한 경우에 대해 과도한 자신감을 보이는 경향이 있음을 발견  

그러나 주석 불일치 샘플을 학습에 활용하면 탐지 정확도와 인간 판단과의 일치를 개선할 수 있음을 보여줌  




짧은 요약(Abstract) :


이 연구는 대형 언어 모델(LLM)이 공격적인 언어를 탐지하는 능력을 평가하며, 특히 주석 불일치(annotation disagreement) 상황에서의 성능을 분석합니다. 주석 불일치는 주관적인 해석에서 발생하는 샘플로, 모호한 특성 때문에 LLM이 이를 처리하는 능력이 부족할 수 있습니다. 연구 결과, LLM은 낮은 합의 수준의 샘플에서 성능이 저하되고, 이러한 모호한 경우에 대해 과도한 자신감을 보이는 경향이 있음을 발견했습니다. 그러나 주석 불일치 샘플을 학습에 활용하면 탐지 정확도와 인간 판단과의 일치를 개선할 수 있음을 보여주었습니다. 이 연구는 LLM 기반의 공격적인 언어 탐지 시스템을 향상시키기 위한 기초 자료를 제공합니다.



This study evaluates the capabilities of large language models (LLMs) in detecting offensive language, particularly focusing on their performance in situations of annotation disagreement. Annotation disagreement arises from subjective interpretations, posing a unique challenge due to their ambiguous nature. The findings reveal that LLMs struggle with low-agreement samples and often exhibit overconfidence in these ambiguous cases. However, utilizing disagreement samples in training improves both detection accuracy and model alignment with human judgment. These insights provide a foundation for enhancing LLM-based offensive language detection systems.


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



이 논문에서는 대규모 언어 모델(LLM)이 공격적인 언어를 탐지하는 데 있어 주석 불일치(annotation disagreement)를 처리하는 능력을 평가하기 위해 여러 가지 방법론을 사용했습니다. 연구의 주요 목표는 LLM이 주석 불일치가 있는 경우에도 효과적으로 공격적인 언어를 탐지할 수 있는지를 분석하는 것입니다. 이를 위해 다음과 같은 방법론을 적용했습니다.

1. **모델 선택**: 연구에서는 여러 개의 폐쇄형(closed-source) 및 개방형(open-source) LLM을 평가했습니다. 폐쇄형 모델로는 GPT-3.5, GPT-4, Claude-3.5, Gemini-1.5 등이 포함되었으며, 개방형 모델로는 LLaMa3(8B, 70B), Qwen2.5(7B, 72B) 등이 사용되었습니다. 다양한 아키텍처와 파라미터 크기를 가진 모델을 포함하여 포괄적인 평가를 수행했습니다.

2. **데이터셋**: MD-Agreement 데이터셋을 사용하여 주석 불일치가 있는 공격적인 언어 샘플을 평가했습니다. 이 데이터셋은 10,753개의 트윗으로 구성되어 있으며, 각 샘플은 다섯 명의 훈련된 주석자에 의해 라벨링되었습니다. 데이터셋은 하드 라벨(다수결 라벨)과 소프트 라벨(주석자 간의 합의 수준)을 모두 제공합니다. 주석자 간의 합의 수준에 따라 샘플을 세 가지 수준(A++, A+, A0)으로 분류했습니다.

3. **평가 방법**: 연구에서는 LLM의 이진 분류 성능을 평가하기 위해 정확도(accuracy)와 F1 점수를 사용했습니다. 또한, 모델의 신뢰도(confidence)와 주석자 간의 불일치 간의 관계를 분석했습니다. 이를 위해 모델의 출력을 여러 온도 설정에서 재샘플링하여 신뢰도를 추정하는 자기 일관성(self-consistency) 방법을 적용했습니다.

4. **학습 기법**: 연구에서는 몇 가지 학습 기법을 사용하여 LLM의 성능을 향상시켰습니다. 특히, 몇 샷 학습(few-shot learning)과 지시 미세 조정(instruction fine-tuning)을 통해 주석 불일치 샘플을 포함한 다양한 샘플을 모델에 학습시켰습니다. 이러한 기법은 모델이 주석 불일치가 있는 샘플을 더 잘 처리할 수 있도록 도와주었습니다.

5. **결과 분석**: 연구 결과, LLM은 주석자 간의 합의가 높은 샘플에서는 높은 정확도를 보였으나, 합의가 낮은 샘플에서는 성능이 급격히 저하되는 경향을 보였습니다. 또한, LLM은 불확실한 경우에 과도한 신뢰를 보이는 경향이 있었습니다. 주석 불일치 샘플을 학습에 포함시키는 것이 모델의 탐지 정확도와 인간의 판단과의 정렬을 개선하는 데 도움이 된다는 것을 발견했습니다.

이러한 방법론을 통해 연구는 LLM이 공격적인 언어 탐지에서 주석 불일치를 처리하는 능력을 체계적으로 평가하고, 향후 모델 개선을 위한 기초 자료를 제공하고자 했습니다.

---




This paper employs a variety of methodologies to evaluate the capabilities of Large Language Models (LLMs) in detecting offensive language while handling annotation disagreement. The primary objective of the study is to analyze whether LLMs can effectively detect offensive language in cases of annotation disagreement. The following methodologies were applied:

1. **Model Selection**: The study evaluates several closed-source and open-source LLMs. Closed-source models include GPT-3.5, GPT-4, Claude-3.5, and Gemini-1.5, while open-source models include LLaMa3 (8B, 70B) and Qwen2.5 (7B, 72B). A comprehensive evaluation was conducted by including models with various architectures and parameter sizes.

2. **Dataset**: The MD-Agreement dataset was utilized to assess offensive language samples with annotation disagreement. This dataset consists of 10,753 tweets, each labeled by five trained annotators. It provides both hard labels (majority-voted labels) and soft labels (levels of agreement among annotators). Samples were classified into three levels based on the degree of agreement (A++, A+, A0).

3. **Evaluation Method**: The study employed accuracy and F1 score as metrics to evaluate the binary classification performance of LLMs. Additionally, the relationship between model confidence and human disagreement was analyzed. The self-consistency method was applied to estimate confidence by resampling model outputs under various temperature settings.

4. **Learning Techniques**: Several learning techniques were employed to enhance the performance of LLMs. In particular, few-shot learning and instruction fine-tuning were used to train the models on a variety of samples, including those with annotation disagreement. These techniques helped the models better handle samples with annotation disagreement.

5. **Result Analysis**: The results indicated that LLMs achieved high accuracy on samples with high agreement among annotators, but their performance sharply declined on samples with low agreement. Furthermore, LLMs tended to exhibit overconfidence in uncertain cases. The study found that incorporating disagreement samples into training improved both detection accuracy and alignment with human judgment.

Through these methodologies, the research systematically evaluates how LLMs handle annotation disagreement in offensive language detection and aims to provide a foundation for future model improvements.


<br/>
# Results



이 논문에서는 대규모 언어 모델(LLM)이 공격적인 언어를 탐지하는 능력을 평가하고, 주석 불일치(annotation disagreement) 상황에서의 성능을 분석합니다. 연구의 주요 목표는 LLM이 주석 불일치가 있는 경우에도 얼마나 잘 작동하는지를 이해하고, 이러한 불일치가 모델의 결정 과정에 미치는 영향을 조사하는 것입니다.

#### 1. 경쟁 모델
연구에서는 여러 개의 폐쇄형(closed-source) 및 개방형(open-source) LLM을 평가했습니다. 폐쇄형 모델로는 GPT-3.5, GPT-4, GPT-4o, Claude-3.5, Gemini-1.5 등이 포함되었고, 개방형 모델로는 LLaMa3(8B, 70B), Qwen2.5(7B, 72B), Mixtral(8x7B, 8x22B) 등이 사용되었습니다.

#### 2. 테스트 데이터
MD-Agreement 데이터셋을 사용하여 모델의 성능을 평가했습니다. 이 데이터셋은 10,753개의 트윗으로 구성되어 있으며, 각 트윗은 다섯 명의 훈련된 주석자에 의해 주석이 달렸습니다. 데이터셋은 주석자 간의 합의 수준에 따라 세 가지 수준(전원 합의, 경미한 합의, 약한 합의)으로 나뉘어 있습니다.

#### 3. 메트릭
모델의 성능을 평가하기 위해 정확도(accuracy)와 F1 점수를 주요 메트릭으로 사용했습니다. 또한, 모델의 신뢰도와 주석자 간의 합의 정도 간의 관계를 분석하기 위해 평균 제곱 오차(MSE)와 스피어만 순위 상관 계수(ρ)를 사용했습니다.

#### 4. 성능 비교
- **전원 합의(A++)**: 모든 모델이 높은 정확도를 기록했습니다. 예를 들어, GPT-4o는 93.96%의 정확도를 달성했습니다.
- **경미한 합의(A+)**: 모델의 성능이 다소 감소했지만 여전히 높은 정확도를 유지했습니다. GPT-4o는 85.24%의 F1 점수를 기록했습니다.
- **약한 합의(A0)**: 모델의 성능이 급격히 떨어졌습니다. GPT-4o의 F1 점수는 57.06%로 감소했습니다. 이는 LLM이 주관적인 경우를 처리하는 데 어려움을 겪고 있음을 나타냅니다.

#### 5. 결론
LLM은 전원 합의가 있는 경우에는 높은 성능을 보이지만, 주석 불일치가 있는 경우에는 성능이 급격히 저하됩니다. 특히, LLM은 불확실한 경우에 과도한 자신감을 보이는 경향이 있으며, 이는 실제 콘텐츠 조정에서 신뢰성을 저하시킬 수 있습니다. 따라서, 주석 불일치 샘플을 학습에 포함시키는 것이 모델의 성능을 향상시키고 인간의 판단과의 정렬을 개선하는 데 도움이 됩니다.

---




This paper evaluates the capabilities of large language models (LLMs) in detecting offensive language and analyzes their performance in situations of annotation disagreement. The primary goal of the study is to understand how well LLMs operate even in the presence of annotation disagreement and to investigate the impact of such disagreements on the model's decision-making process.

#### 1. Competing Models
The study evaluates several closed-source and open-source LLMs. Closed-source models include GPT-3.5, GPT-4, GPT-4o, Claude-3.5, and Gemini-1.5, while open-source models include LLaMa3 (8B, 70B), Qwen2.5 (7B, 72B), and Mixtral (8x7B, 8x22B).

#### 2. Test Data
The MD-Agreement dataset was used to assess the models' performance. This dataset consists of 10,753 tweets, each annotated by five trained annotators. The dataset is divided into three levels based on the degree of agreement among annotators (unanimous agreement, mild agreement, and weak agreement).

#### 3. Metrics
Accuracy and F1 score were used as the primary metrics to evaluate model performance. Additionally, Mean Squared Error (MSE) and Spearman's Rank Correlation Coefficient (ρ) were employed to analyze the relationship between model confidence and the degree of agreement among annotators.

#### 4. Performance Comparison
- **Unanimous Agreement (A++)**: All models achieved high accuracy. For instance, GPT-4o reached an accuracy of 93.96%.
- **Mild Agreement (A+)**: The performance of the models slightly decreased but still maintained high accuracy. GPT-4o recorded an F1 score of 85.24%.
- **Weak Agreement (A0)**: The performance of the models dropped significantly. The F1 score for GPT-4o decreased to 57.06%. This indicates that LLMs struggle to handle subjective cases effectively.

#### 5. Conclusion
LLMs perform well in cases of unanimous agreement but experience a sharp decline in performance when faced with annotation disagreement. Notably, LLMs tend to exhibit overconfidence in uncertain cases, which can undermine their reliability in real-world moderation tasks. Therefore, incorporating disagreement samples into training can enhance model performance and improve alignment with human judgment.


<br/>
# 예제



이 논문에서는 대규모 언어 모델(LLM)이 공격적인 언어를 탐지하는 능력을 평가하기 위해 MD-Agreement 데이터셋을 사용합니다. 이 데이터셋은 각 샘플에 대해 다섯 명의 훈련된 인간 주석자가 주석을 달아, 주석자 간의 동의 정도에 따라 샘플을 분류합니다. 주석자 간의 동의 정도는 세 가지로 나뉘며, 각각은 다음과 같습니다:

1. **A++ (전적인 동의)**: 모든 주석자가 동일한 레이블을 부여한 경우.
2. **A+ (부분적인 동의)**: 네 명의 주석자가 동일한 레이블을 부여하고 한 명은 다른 레이블을 부여한 경우.
3. **A0 (약한 동의)**: 세 명의 주석자가 동일한 레이블을 부여하고 두 명은 다른 레이블을 부여한 경우.

#### 예시

- **트레이닝 데이터 예시**:
  - **입력**: "Come on man! Lock’em up!!! #maga"
  - **출력**: 0.2 (부분적인 동의, 비공격적)
  
  - **입력**: "#DISGUSTING #Democrats terrorize old folks just before #elections2020"
  - **출력**: 1 (전적인 동의, 공격적)

- **테스트 데이터 예시**:
  - **입력**: "Set fire to Fox News (metaphorically)"
  - **출력**: 0.4 (약한 동의, 비공격적)

  - **입력**: "Y’all trending about kpop stans instead of #BlackLivesMatter big fack you"
  - **출력**: 1 (전적인 동의, 공격적)

이러한 데이터는 LLM이 공격적인 언어를 탐지하는 데 있어 주석자 간의 동의 정도가 모델의 성능에 미치는 영향을 평가하는 데 사용됩니다. 모델은 주어진 입력에 대해 공격적 또는 비공격적이라는 레이블을 예측하며, 이 과정에서 모델의 신뢰도와 주석자 간의 동의 정도 간의 관계를 분석합니다.





This paper evaluates the capabilities of large language models (LLMs) in detecting offensive language using the MD-Agreement dataset. This dataset is annotated by five trained human annotators for each sample, categorizing the samples based on the degree of agreement among the annotators. The degrees of agreement are divided into three categories:

1. **A++ (Unanimous Agreement)**: All annotators provide the same label.
2. **A+ (Mild Agreement)**: Four annotators agree on the same label, while one annotator provides a different label.
3. **A0 (Weak Agreement)**: Three annotators agree on the same label, while two annotators provide different labels.

#### Examples

- **Training Data Example**:
  - **Input**: "Come on man! Lock’em up!!! #maga"
  - **Output**: 0.2 (Mild Agreement, Non-offensive)
  
  - **Input**: "#DISGUSTING #Democrats terrorize old folks just before #elections2020"
  - **Output**: 1 (Unanimous Agreement, Offensive)

- **Test Data Example**:
  - **Input**: "Set fire to Fox News (metaphorically)"
  - **Output**: 0.4 (Weak Agreement, Non-offensive)

  - **Input**: "Y’all trending about kpop stans instead of #BlackLivesMatter big fack you"
  - **Output**: 1 (Unanimous Agreement, Offensive)

These data points are used to assess how the LLMs perform in detecting offensive language, particularly focusing on how the degree of agreement among annotators affects the model's performance. The model predicts whether the given input is offensive or non-offensive, and the analysis includes examining the relationship between the model's confidence and the degree of agreement among annotators.

<br/>
# 요약

이 연구에서는 대규모 언어 모델(LLM)이 공격적인 언어를 탐지하는 능력을 평가하고, 주석 불일치가 모델의 성능에 미치는 영향을 분석하였다. 결과적으로, LLM은 주석이 일치하는 경우에는 높은 정확도를 보였으나, 불일치하는 경우에는 과도한 자신감을 보이며 성능이 저하되었다. 예를 들어, LLM은 모호한 사례에서 비공격적인 언어를 공격적으로 잘못 분류하는 경향이 있었다.

---

This study evaluates the capabilities of large language models (LLMs) in detecting offensive language and analyzes the impact of annotation disagreement on model performance. The results show that while LLMs achieve high accuracy in cases of unanimous agreement, their performance declines significantly in disagreement cases, often exhibiting overconfidence. For instance, LLMs tend to misclassify non-offensive language as offensive in ambiguous scenarios.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: LLMs의 정확도는 주어진 주석 동의 수준에 따라 다르게 나타나며, 특히 동의가 높은 경우(A++)에서 높은 정확도를 보인다. 반면, 동의가 낮은 경우(A0)에서는 정확도가 급격히 떨어진다. 이는 LLM들이 주관적인 언어 판단을 처리하는 데 어려움을 겪고 있음을 시사한다.
   - **Figure 2**: LLM의 자기 일관성을 보여주며, 주석 동의 수준이 낮을수록 일관성이 감소하는 경향이 있다. 이는 LLM들이 모호한 사례에서 불확실성을 잘 인식하지 못함을 나타낸다.
   - **Figure 3**: GPT-4o의 신뢰도 점수와 소프트 레이블 간의 관계를 시각화한 혼동 행렬로, LLM이 주석 동의 수준에 관계없이 높은 신뢰도를 부여하는 경향이 있음을 보여준다. 이는 LLM의 과신 문제를 강조한다.

2. **테이블**
   - **Table 1**: 다양한 주석 동의 수준을 가진 샘플의 예시를 제공하며, 각 샘플의 하드 레이블과 소프트 레이블을 통해 주관적인 언어 판단의 복잡성을 보여준다.
   - **Table 3**: LLM의 이진 분류 성능을 요약하며, 동의 수준이 높을수록 성능이 우수하나, 동의가 낮은 경우 성능이 급격히 저하됨을 나타낸다.
   - **Table 4**: LLM의 신뢰도와 주석 동의 간의 관계를 평가하며, 동의 수준이 낮을수록 MSE가 증가하고 상관관계가 약해지는 경향을 보여준다.
   - **Table 5**: GPT-4o의 성능을 다양한 주석 동의 수준에서 평가하며, 주석 동의 수준이 모델의 성능에 미치는 영향을 분석한다.

3. **어펜딕스**
   - **Appendix A**: 데이터셋의 품질 관리 과정과 평가 지표에 대한 세부 정보를 제공하며, MSE와 Spearman의 순위 상관 계수를 사용하여 LLM의 신뢰도와 주석 동의 간의 관계를 평가하는 방법을 설명한다.
   - **Appendix B**: 다양한 실험 결과를 포함하여 LLM의 성능을 평가하고, 온도 샘플링이 LLM의 성능에 미치는 영향을 분석한다. 또한, 오류 분석을 통해 LLM이 주관적인 언어 판단에서 겪는 문제를 심층적으로 탐구한다.




1. **Diagrams and Figures**
   - **Figure 1**: The accuracy of LLMs varies significantly based on the level of annotation agreement, showing high accuracy for unanimous cases (A++) but a sharp decline for low-agreement cases (A0). This indicates that LLMs struggle with subjective language judgments.
   - **Figure 2**: This figure illustrates the self-consistency of LLMs, revealing that consistency decreases as the level of annotation agreement drops. This suggests that LLMs do not effectively recognize uncertainty in ambiguous cases.
   - **Figure 3**: The confusion matrix visualizes the relationship between GPT-4o's confidence scores and soft labels, indicating a tendency for LLMs to assign high confidence regardless of annotation agreement, highlighting the issue of overconfidence.

2. **Tables**
   - **Table 1**: Provides examples of samples with varying levels of annotation agreement, illustrating the complexity of subjective language judgments through hard and soft labels.
   - **Table 3**: Summarizes the binary classification performance of LLMs, showing that performance is strong for high agreement cases but drops significantly for low agreement cases.
   - **Table 4**: Evaluates the relationship between LLM confidence and annotation agreement, indicating that as agreement decreases, the mean squared error (MSE) increases and correlation weakens.
   - **Table 5**: Assesses the performance of GPT-4o across different levels of annotation agreement, analyzing how these levels impact model performance.

3. **Appendix**
   - **Appendix A**: Provides detailed information on the quality control process of the dataset and the metrics used for evaluation, explaining how MSE and Spearman's rank correlation coefficient are employed to assess the relationship between LLM confidence and annotation agreement.
   - **Appendix B**: Includes various experimental results evaluating LLM performance and analyzes the impact of temperature sampling on LLM accuracy. It also conducts error analysis to explore the challenges LLMs face in subjective language judgments.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Lu2025,
  author    = {Junyu Lu and Kai Ma and Kaichun Wang and Kelaiti Xiao and Roy Ka-Wei Lee and Bo Xu and Liang Yang and Hongfei Lin},
  title     = {Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {5609--5626},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  address   = {July 27 - August 1, 2025}
}
```

### 시카고 스타일

Junyu Lu, Kai Ma, Kaichun Wang, Kelaiti Xiao, Roy Ka-Wei Lee, Bo Xu, Liang Yang, and Hongfei Lin. "Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement." In *Findings of the Association for Computational Linguistics: ACL 2025*, 5609–5626. Association for Computational Linguistics, July 27 - August 1, 2025.
