---
layout: post
title:  "[2025]Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection"
date:   2025-08-13 16:09:01 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

성차별 감지 문제 데이터셋의 클래스 불균형 문제 해결 위해 프롬프트 기반 데이터 증강 기법을 제안: 정의 기반 데이터 증강(DDA)과 문맥적 의미 확장(CSE)  


짧은 요약(Abstract) :
논문의 초록 부분은 온라인 콘텐츠에서의 성차별 감지 문제를 다루고 있습니다. 성차별적 언어는 여성과 소외된 그룹에 불균형적으로 영향을 미치며, 이를 감지하기 위한 자동화 시스템이 개발되었지만 여전히 두 가지 주요 문제에 직면하고 있습니다: 데이터 부족과 성차별적 언어의 미묘한 특성입니다. 대규모 데이터셋에서도 심각한 클래스 불균형이 모델의 일반화를 방해하며, 세부적인 카테고리의 경계가 겹치고 모호하여 주석자 간의 의견 불일치를 초래합니다. 이러한 문제를 해결하기 위해, 저자들은 두 가지 프롬프트 기반 데이터 증강 기법을 제안합니다: 정의 기반 데이터 증강(DDA)과 문맥적 의미 확장(CSE)입니다. 또한, 여러 언어 모델의 보완적 관점을 결합하여 예측을 개선하는 앙상블 전략을 도입했습니다. 실험 결과, 제안된 방법이 모든 작업에서 최첨단 성능을 달성했으며, 특히 이진 분류와 세부 분류에서 성능이 향상되었습니다.


The detection of sexism in online content remains an open problem, as harmful language disproportionately affects women and marginalized groups. While automated systems for sexism detection have been developed, they still face two key challenges: data sparsity and the nuanced nature of sexist language. Even in large, well-curated datasets like the Explainable Detection of Online Sexism (EDOS), severe class imbalance hinders model generalization. Additionally, the overlapping and ambiguous boundaries of fine-grained categories introduce substantial annotator disagreement, reflecting the difficulty of interpreting nuanced expressions of sexism. To address these challenges, we propose two prompt-based data augmentation techniques: Definition-based Data Augmentation (DDA), which leverages category-specific definitions to generate semantically-aligned synthetic examples, and Contextual Semantic Expansion (CSE), which targets systematic model errors by enriching examples with task-specific semantic features. To further improve reliability in fine-grained classification, we introduce an ensemble strategy that resolves prediction ties by aggregating complementary perspectives from multiple language models. Our experimental evaluation on the EDOS dataset demonstrates state-of-the-art performance across all tasks, with notable improvements of macro F1 by 1.5 points for binary classification (Task A) and 4.1 points for fine-grained classification (Task C).


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


이 논문에서는 온라인 콘텐츠에서 성차별을 탐지하는 데 있어 두 가지 주요 문제를 해결하기 위한 새로운 방법론을 제안합니다. 첫 번째 문제는 데이터 희소성이고, 두 번째 문제는 성차별 언어의 미묘한 해석입니다. 이를 해결하기 위해 두 가지 데이터 증강 기법을 도입했습니다: 정의 기반 데이터 증강(Definition-based Data Augmentation, DDA)과 맥락적 의미 확장(Contextual Semantic Expansion, CSE)입니다.

#### 정의 기반 데이터 증강 (DDA)

DDA는 각 성차별 카테고리에 대한 명확한 정의를 활용하여 의미적으로 정렬된 합성 예제를 생성하는 데이터 증강 기법입니다. 이 방법은 각 카테고리의 경계를 명확히 하여 학습 중 모호성을 줄이는 데 중점을 둡니다. DDA는 대형 언어 모델(LLM)을 사용하여 각 데이터셋의 인스턴스에 대해 여러 개의 합성 예제를 생성합니다. 이 합성 예제들은 원래의 성차별 의도를 유지하면서도 언어, 톤, 스타일을 다양화하여 소셜 미디어 상의 비공식적인 상호작용을 모방합니다.

#### 맥락적 의미 확장 (CSE)

CSE는 모델의 예측 한계를 해결하기 위해 설계된 자기 개선 방법론입니다. 이 방법은 전통적인 확률 기반의 자기 수정 기법과 달리, 도전적인 예제에 대한 추가적인 맥락과 설명을 생성하여 모델의 이해를 명시적으로 향상시킵니다. CSE는 특정 작업에 맞춘 프롬프트를 사용하여 도전적인 인스턴스를 분석하고, 그들의 의미적 특성을 식별하며, 이러한 통찰을 학습 데이터에 통합합니다. 이 방법은 체계적인 의미 분석을 통해 모델의 오분류를 해결하고, 미묘한 패턴을 명시적으로 포착하는 데 기여합니다.

#### Mistral-7B 폴백 앙상블 (M7-FE)

M7-FE는 다중 클래스 분류에서 예측 신뢰성을 향상시키기 위해 여러 모델의 예측을 결합하는 앙상블 전략입니다. 이 방법은 다수결 투표를 통해 예측을 집계하고, 동점 상황에서는 Mistral-7B 모델을 폴백 모델로 사용하여 최종 결정을 내립니다. 이 앙상블 방법은 개별 모델의 상호 보완적인 강점을 활용하여 미묘한 클래스 간의 구분을 포착하고, 맥락에 의존하는 성차별 인스턴스를 처리하는 데 도움을 줍니다.



This paper proposes a novel methodology to address two major challenges in detecting sexism in online content: data sparsity and the nuanced interpretation of sexist language. To tackle these issues, two data augmentation techniques are introduced: Definition-based Data Augmentation (DDA) and Contextual Semantic Expansion (CSE).

#### Definition-based Data Augmentation (DDA)

DDA is a data augmentation technique that generates semantically aligned synthetic examples by leveraging clear definitions for each sexism category. This method focuses on clarifying the boundaries of each category to reduce ambiguity during training. DDA uses large language models (LLMs) to generate multiple synthetic examples for each instance in the dataset. These synthetic examples maintain the harmful nature of the original sexist intent while diversifying language, tone, and style to mimic informal interactions on social media.

#### Contextual Semantic Expansion (CSE)

CSE is a self-refinement methodology designed to address limitations in model predictions by expanding the context of challenging examples through explicit semantic expansion. Unlike traditional confidence-based self-correction techniques, CSE explicitly enhances model understanding by generating additional context and explanations for challenging examples. This method operates iteratively, using task-specific prompts to analyze challenging instances, identify their semantic characteristics, and incorporate these insights into the training data. CSE contributes to resolving model misclassifications and explicitly capturing subtle patterns through systematic semantic analysis.

#### Mistral-7B Fallback Ensemble (M7-FE)

M7-FE is an ensemble strategy that combines predictions from multiple models to improve prediction reliability in multi-class classification. It aggregates predictions through majority voting and resolves tie-breaking scenarios using the Mistral-7B model as the fallback. This ensemble method leverages the complementary strengths of individual models to capture subtle distinctions between classes and handle context-dependent instances of sexism.


<br/>
# Results


이 연구에서는 EDOS 데이터셋을 기반으로 한 성차별 탐지 작업에서 여러 모델의 성능을 비교했습니다. 실험은 세 가지 주요 작업(Task A, B, C)으로 나뉘며, 각 작업은 성차별 언어를 다양한 수준에서 분류하는 것을 목표로 합니다.

#### Task A: 이진 분류
Task A는 댓글이 성차별적인지 여부를 이진 분류하는 작업입니다. 이 작업에서 가장 높은 성능을 보인 모델은 Contextual Semantic Expansion (CSE) 기법을 사용한 모델로, 매크로 F1 점수 0.8819를 기록했습니다. 이는 기존의 최고 성능을 보인 모델보다도 높은 점수로, CSE 기법이 이진 분류에서의 체계적인 편향을 효과적으로 교정했음을 보여줍니다.

#### Task B: 다중 클래스 분류
Task B는 성차별적인 댓글을 네 가지 카테고리로 분류하는 작업입니다. 이 작업에서는 PaLM 앙상블 모델이 0.7326의 매크로 F1 점수로 가장 높은 성능을 보였습니다. 그러나, Definition-based Data Augmentation (DDA) 기법을 사용한 모델도 0.7277의 점수를 기록하며 경쟁력 있는 성능을 보였습니다. 이는 DDA 기법이 다중 클래스 분류에서의 데이터 불균형 문제를 효과적으로 해결했음을 시사합니다.

#### Task C: 세분화된 분류
Task C는 성차별적인 댓글을 11개의 세부 카테고리로 분류하는 작업입니다. 이 작업에서 DDA 기법을 사용한 모델은 0.6018의 매크로 F1 점수를 기록하며, 기존의 최고 성능을 보인 모델보다도 높은 점수를 기록했습니다. 이는 DDA 기법이 세분화된 분류 작업에서의 미세한 차이를 잘 포착했음을 보여줍니다.

#### 결론
전반적으로, 제안된 DDA와 CSE 기법은 기존의 모델들보다 우수한 성능을 보였으며, 특히 세분화된 분류 작업에서의 성능 향상이 두드러졌습니다. 이는 데이터 증강 기법이 성차별 탐지 작업에서의 데이터 희소성과 의미적 모호성을 효과적으로 해결할 수 있음을 시사합니다.

---



In this study, we compared the performance of various models on the task of sexism detection based on the EDOS dataset. The experiments were divided into three main tasks (Task A, B, C), each aiming to classify sexist language at different levels.

#### Task A: Binary Classification
Task A involves binary classification to determine whether a comment is sexist. The model that showed the highest performance in this task used the Contextual Semantic Expansion (CSE) technique, achieving a macro F1 score of 0.8819. This score surpasses the previous best-performing models, indicating that the CSE technique effectively corrected systematic biases in binary classification.

#### Task B: Multi-Class Classification
Task B classifies sexist comments into four categories. In this task, the PaLM ensemble model achieved the highest macro F1 score of 0.7326. However, the model using the Definition-based Data Augmentation (DDA) technique also showed competitive performance with a score of 0.7277, suggesting that the DDA technique effectively addressed data imbalance issues in multi-class classification.

#### Task C: Fine-Grained Classification
Task C involves classifying sexist comments into 11 fine-grained categories. In this task, the model using the DDA technique achieved a macro F1 score of 0.6018, surpassing the previous best-performing models. This indicates that the DDA technique effectively captured subtle distinctions in fine-grained classification tasks.

#### Conclusion
Overall, the proposed DDA and CSE techniques outperformed existing models, with particularly notable improvements in fine-grained classification tasks. This suggests that data augmentation techniques can effectively address data sparsity and semantic ambiguity in sexism detection tasks.


<br/>
# 예제
논문 "Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection"에서는 온라인 콘텐츠에서 성차별을 탐지하는 문제를 다루고 있습니다. 이 논문에서는 성차별 탐지의 두 가지 주요 과제인 데이터 희소성과 성차별 언어의 미묘한 특성을 해결하기 위해 두 가지 데이터 증강 기법을 제안합니다. 이 기법들은 Definition-based Data Augmentation (DDA)와 Contextual Semantic Expansion (CSE)입니다. 또한, Mistral-7B Fallback Ensemble (M7-FE)라는 앙상블 전략을 도입하여 모델의 신뢰성을 향상시킵니다.

### 예시 설명

#### 트레이닝 데이터
- **입력 (Input):** 온라인 플랫폼(예: Reddit, Gab)에서 수집된 소셜 미디어 댓글
- **출력 (Output):** 각 댓글이 성차별적인지 여부를 나타내는 레이블
  - **Task A:** 이진 분류 (성차별적/비성차별적)
  - **Task B:** 성차별적인 댓글을 네 가지 카테고리로 분류 (위협, 비방, 적대감, 편견적 논의)
  - **Task C:** 성차별적인 댓글을 11개의 세부 카테고리로 분류

#### 테스트 데이터
- **입력 (Input):** 트레이닝 데이터와 유사한 형식의 소셜 미디어 댓글
- **출력 (Output):** 각 댓글에 대한 성차별 여부 및 세부 카테고리 레이블

### 구체적인 테스크
1. **Task A (이진 분류):** 댓글이 성차별적인지 여부를 판단합니다.
   - 예: "여성은 감정적이어서 중요한 결정을 내릴 수 없다." -> 성차별적
2. **Task B (다중 클래스 분류):** 성차별적인 댓글을 네 가지 카테고리로 분류합니다.
   - 예: "여성은 감정적이어서 중요한 결정을 내릴 수 없다." -> 비방
3. **Task C (세부 카테고리 분류):** 성차별적인 댓글을 11개의 세부 카테고리로 분류합니다.
   - 예: "여성은 감정적이어서 중요한 결정을 내릴 수 없다." -> 2.1 Descriptive Attacks



The paper "Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection" addresses the problem of detecting sexism in online content. It proposes two data augmentation techniques, Definition-based Data Augmentation (DDA) and Contextual Semantic Expansion (CSE), to tackle the challenges of data sparsity and the nuanced nature of sexist language. Additionally, it introduces an ensemble strategy called Mistral-7B Fallback Ensemble (M7-FE) to improve model reliability.

### Example Explanation

#### Training Data
- **Input:** Social media comments collected from online platforms (e.g., Reddit, Gab)
- **Output:** Labels indicating whether each comment is sexist
  - **Task A:** Binary classification (sexist/non-sexist)
  - **Task B:** Classify sexist comments into four categories (threats, derogation, animosity, prejudiced discussions)
  - **Task C:** Classify sexist comments into 11 fine-grained categories

#### Test Data
- **Input:** Social media comments in a similar format to the training data
- **Output:** Labels for each comment indicating sexism and fine-grained category

### Specific Tasks
1. **Task A (Binary Classification):** Determine whether a comment is sexist.
   - Example: "Women are too emotional to make important decisions." -> Sexist
2. **Task B (Multi-class Classification):** Classify sexist comments into four categories.
   - Example: "Women are too emotional to make important decisions." -> Derogation
3. **Task C (Fine-grained Classification):** Classify sexist comments into 11 fine-grained categories.
   - Example: "Women are too emotional to make important decisions." -> 2.1 Descriptive Attacks

<br/>
# 요약

이 논문에서는 온라인 성차별 탐지를 개선하기 위해 정의 기반 데이터 증강(DDA)과 문맥적 의미 확장(CSE)이라는 두 가지 데이터 증강 기법을 제안합니다. 실험 결과, 제안된 방법은 EDOS 데이터셋에서 모든 작업에 걸쳐 최첨단 성능을 달성했으며, 특히 이진 분류(Task A)에서 매크로 F1 점수가 1.5포인트, 세분화된 분류(Task C)에서 4.1포인트 향상되었습니다. 예시로, DDA는 성차별적 의도를 반영하고 소셜 미디어의 비공식적 언어를 모방하여 문장을 생성합니다.


This paper proposes two data augmentation techniques, Definition-based Data Augmentation (DDA) and Contextual Semantic Expansion (CSE), to improve online sexism detection. Experimental results demonstrate that the proposed methods achieve state-of-the-art performance across all tasks on the EDOS dataset, with notable improvements of 1.5 points in macro F1 for binary classification (Task A) and 4.1 points for fine-grained classification (Task C). As an example, DDA generates sentences that reflect sexist intent and mimic informal social media language.

<br/>
# 기타
논문에서 제공된 다이어그램, 피규어, 테이블, 어펜딕스는 주로 연구 결과와 그에 따른 인사이트를 시각적으로 표현하고 있습니다. 각 요소의 주요 내용을 요약하면 다음과 같습니다:

1. **Figure 1: DDA Framework** - 이 다이어그램은 Definition-based Data Augmentation (DDA) 프레임워크의 구조를 보여줍니다. DDA는 카테고리 정의를 활용하여 미세한 성차별 카테고리의 의미적 경계를 명확히 하고, 모델이 성차별 의도를 반영하는 변형을 생성하도록 지시합니다.

2. **Table 1: Annotator Agreement and Disagreement** - 이 테이블은 Task C의 각 카테고리에 대한 주석자 간의 합의 및 불일치 비율을 보여줍니다. 이는 성차별 언어의 미묘한 해석이 얼마나 어려운지를 반영하며, 주석자 간의 높은 불일치율이 모델 성능에 부정적인 영향을 미칠 수 있음을 시사합니다.

3. **Figure 2: Pipeline Overview** - 이 다이어그램은 제안된 파이프라인의 전반적인 구조를 보여줍니다. 여기에는 사전 훈련, 데이터 증강(DDA 또는 CSE), 미세 조정, 앙상블 모델링이 포함됩니다. 각 단계는 희소하거나 모호한 사례에 대한 성차별 분류 성능을 향상시키기 위해 설계되었습니다.

4. **Table 2: Model Performance Comparison** - 이 테이블은 다양한 모델의 성능을 매크로 F1 점수로 비교합니다. 제안된 방법(DDA 및 CSE)이 모든 작업에서 최첨단 성능을 달성했음을 보여줍니다.

5. **Figure 4: Confusion Matrix** - 이 혼동 행렬은 DDA가 적용된 모델과 그렇지 않은 모델 간의 차이를 보여줍니다. DDA는 특히 저자원 및 미세한 카테고리에서 성능을 향상시키며, 카테고리 간의 혼동을 줄이는 데 효과적임을 나타냅니다.

6. **Appendix A: Prompt Structures** - 이 부록은 데이터 증강을 위한 기본 프롬프트와 DDA 프롬프트의 구조를 비교합니다. DDA 프롬프트는 카테고리 정의를 통합하여 생성된 데이터가 특정 카테고리와 잘 맞도록 합니다.

7. **Appendix B: Hyperparameters** - 이 부록은 각 모델에 대한 최적의 하이퍼파라미터를 나열합니다. 이는 모델의 성능을 최적화하기 위한 중요한 요소입니다.

8. **Appendix C: Class-Specific Data Augmentation** - 이 부록은 Task C의 각 클래스에 대한 데이터 증강 전략을 자세히 설명합니다. 특히, 저자원 클래스에 대한 증강이 모델 성능을 크게 향상시킴을 보여줍니다.

9. **Appendix D: DDA Prompt** - 이 부록은 DDA 프롬프트에 사용된 카테고리 정의를 제공합니다. 이러한 정의는 성차별 언어의 미묘한 특성을 반영하도록 설계되었습니다.



1. **Figure 1: DDA Framework** - This diagram illustrates the structure of the Definition-based Data Augmentation (DDA) framework. DDA leverages category definitions to clarify semantic boundaries of fine-grained sexist categories and instructs the model to generate variations that reflect sexist intent.

2. **Table 1: Annotator Agreement and Disagreement** - This table shows the rates of agreement and disagreement among annotators for each category in Task C. It reflects the difficulty of interpreting nuanced sexist language, with high annotator disagreement rates potentially undermining model performance.

3. **Figure 2: Pipeline Overview** - This diagram presents the overall structure of the proposed pipeline, including pre-training, data augmentation (via DDA or CSE), fine-tuning, and ensemble modeling. Each step is designed to improve performance on fine-grained sexism classification by enhancing contextual understanding and robustness to sparse or ambiguous cases.

4. **Table 2: Model Performance Comparison** - This table compares the performance of various models using macro F1 scores. It shows that the proposed methods (DDA and CSE) achieve state-of-the-art performance across all tasks.

5. **Figure 4: Confusion Matrix** - This confusion matrix shows the difference between models with and without DDA. DDA significantly improves performance, especially in underrepresented and fine-grained categories, and reduces confusion between categories.

6. **Appendix A: Prompt Structures** - This appendix compares the structures of baseline prompts and DDA prompts for data augmentation. DDA prompts incorporate category definitions to ensure generated data aligns closely with specific categories.

7. **Appendix B: Hyperparameters** - This appendix lists the optimal hyperparameters for each model, which are crucial for optimizing model performance.

8. **Appendix C: Class-Specific Data Augmentation** - This appendix details data augmentation strategies for each class in Task C, showing that augmenting underrepresented classes significantly improves model performance.

9. **Appendix D: DDA Prompt** - This appendix provides the category definitions used in the DDA prompt, designed to reflect the nuanced characteristics of sexist language.

<br/>
# refer format:


**BibTeX:**
```bibtex
@inproceedings{khan2025explaining,
  title={Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection},
  author={Khan, Sahrish and Jhumka, Arshad and Pergola, Gabriele},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={16553--16571},
  year={2025},
  organization={Association for Computational Linguistics}
}
```

**Chicago Style:**
Khan, Sahrish, Arshad Jhumka, and Gabriele Pergola. 2025. "Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 16553-16571. Association for Computational Linguistics.
