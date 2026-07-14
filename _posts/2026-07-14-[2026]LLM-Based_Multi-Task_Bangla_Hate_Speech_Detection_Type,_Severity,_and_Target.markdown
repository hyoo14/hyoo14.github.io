---
layout: post
title:  "[2026]LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target"
date:   2026-07-14 01:07:10 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 BanglaMultiHate라는 첫 번째 다중 작업 방글라 증오 발언 데이터셋을 개발하고, 이를 통해 다양한 모델을 비교하여 방글라어 증오 발언 탐지의 성능을 평가하였다.


짧은 요약(Abstract) :


이 논문의 초록에서는 온라인 소셜 미디어 플랫폼이 커뮤니케이션과 정보 교환의 중심이 되었지만, 동시에 개인과 커뮤니티를 겨냥한 증오 발언, 공격적인 언어, 괴롭힘의 온상이 되고 있다는 점을 강조합니다. 이러한 콘텐츠는 온라인 안전과 포용성을 저해하며, 특히 제한된 조정 도구가 있는 저자원 언어에서 신뢰할 수 있는 탐지 시스템의 필요성을 강조합니다. 기존의 방글라어 관련 연구는 유용한 자원과 모델을 제공하지만, 대부분이 단일 작업(예: 이진 증오/공격)으로 제한되어 있으며, 증오의 유형, 심각성 및 대상을 포함한 주요 차원에 대한 범위가 좁습니다. 이 연구는 첫 번째 다중 작업 방글라 증오 발언 데이터셋인 BanglaMultiHate를 소개하며, 이는 현재까지 수작업으로 주석이 달린 가장 큰 데이터셋 중 하나입니다. 이 자원을 사용하여 다양한 기준선, 단일 언어 사전 훈련 모델 및 LLM을 비교 연구하였으며, LoRA 미세 조정 설정에서 제로샷 및 몇 샷 설정을 포함했습니다. 연구 결과, LoRA로 조정된 LLM이 BanglaBERT와 경쟁할 수 있지만, 문화적으로 기반이 있는 사전 훈련이 강력한 성능을 위해 여전히 중요하다는 것을 보여줍니다. 전반적으로 BanglaMultiHate는 저자원 맥락에서 증오 발언 탐지를 위한 더 강력한 기준을 설정합니다.



The abstract of this paper emphasizes that online social media platforms have become central to communication and information exchange, but they also serve as fertile ground for hate speech, offensive language, and bullying targeting individuals and communities. Such content undermines online safety and inclusion, highlighting the need for reliable detection systems—especially in low-resource languages with limited moderation tools. Existing work on Bangla provides valuable resources and models, but they are mostly single-task (e.g., binary hate/offense) with narrow coverage of key dimensions such as type, severity, and target of hate. This study addresses these gaps by introducing the first multi-task Bangla hate speech dataset, BanglaMultiHate, which is one of the largest manually annotated datasets to date. Using this resource, the authors performed a comparative study across different baselines, monolingual pretrained models, and LLMs under zero-shot, few-shot, and LoRA fine-tuning settings. The findings show that while LoRA-tuned LLMs rival BanglaBERT, culturally grounded pretraining remains crucial for robust performance. Overall, BanglaMultiHate establishes a stronger benchmark for hate speech detection in low-resource contexts.


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



이 논문에서는 방글라어의 혐오 발언 탐지를 위한 첫 번째 다중 작업 데이터셋인 BanglaMultiHate를 개발하고, 이를 통해 다양한 모델을 평가하는 방법론을 제시합니다. 연구의 주요 목표는 방글라어에서 혐오 발언의 유형, 심각도 및 대상을 식별하는 것입니다. 이를 위해 다음과 같은 방법론을 사용했습니다.

1. **데이터 수집**: 연구팀은 YouTube API를 사용하여 방글라어 댓글을 수집했습니다. 이 댓글들은 비즈니스, 연예인, 재난, 정치 등 19개의 다양한 카테고리로 분류되었습니다. 최종적으로 약 50,746개의 댓글이 수집되었으며, 이 데이터는 수작업으로 주석이 달렸습니다.

2. **주석 작업**: 주석 팀은 방글라어를 모국어로 사용하는 대학생들로 구성되었으며, 각 댓글에 대해 혐오 발언의 유형(모욕적, 성차별, 종교적 혐오 등), 심각도(경미, 중간, 심각), 그리고 대상(개인, 조직, 커뮤니티 등)을 평가했습니다. 주석 작업은 세 명의 주석자가 독립적으로 수행하였고, 다수결에 의해 최종 레이블이 결정되었습니다.

3. **모델 선택**: 연구에서는 SVM(서포트 벡터 머신), 방글라BERT와 같은 단일 언어 모델, 그리고 대형 언어 모델(LLMs)인 Llama, Qwen, Gemini 등을 사용하여 성능을 비교했습니다. 각 모델은 제로샷, 몇 샷, LoRA(저랭크 적응) 방식으로 미세 조정되었습니다.

4. **평가 방법**: 모델의 성능은 정확도, 마이크로 F1 점수, 가중 정밀도 및 재현율을 사용하여 평가되었습니다. 특히, 클래스 불균형을 고려하여 가중 지표를 사용했습니다.

5. **결과 분석**: 연구 결과, 방글라BERT가 모든 작업에서 가장 높은 성능을 보였으며, LLMs는 특정 작업에서 유의미한 성능 향상을 보였지만, 여전히 방글라BERT에 비해 부족한 성능을 보였습니다. LoRA를 사용한 미세 조정이 모델의 성능을 크게 향상시켰음을 확인했습니다.

이 연구는 방글라어의 혐오 발언 탐지에 대한 새로운 기준을 설정하고, 저자들은 향후 데이터셋을 확장하여 더 나은 모델을 개발할 계획입니다.

---




This paper presents a methodology for developing the first multi-task dataset for hate speech detection in Bangla, named BanglaMultiHate, and evaluates various models using this dataset. The primary goal of the research is to identify the type, severity, and target of hate speech in Bangla. The methodology includes the following components:

1. **Data Collection**: The research team collected Bangla comments using the YouTube API. These comments were categorized into 19 different categories, including Business, Celebrities, Disaster, Politics, etc. A total of approximately 50,746 comments were collected, and this data was manually annotated.

2. **Annotation Process**: The annotation team consisted of native Bangla-speaking undergraduate students, who evaluated each comment for the type of hate (e.g., Abusive, Sexism, Religious Hate), severity (e.g., Little to None, Mild, Severe), and target (e.g., Individual, Organization, Community). The annotation was performed independently by three annotators, and the final label was determined by majority agreement.

3. **Model Selection**: The study employed various models, including SVM (Support Vector Machine), monolingual pretrained models like BanglaBERT, and large language models (LLMs) such as Llama, Qwen, and Gemini, to compare performance. Each model was evaluated under zero-shot, few-shot, and LoRA (Low-Rank Adaptation) fine-tuning settings.

4. **Evaluation Metrics**: Model performance was assessed using accuracy, micro F1 score, weighted precision, and recall. Weighted metrics were particularly used to account for class imbalance.

5. **Results Analysis**: The results indicated that BanglaBERT achieved the highest performance across all tasks, while LLMs showed significant performance improvements in certain tasks but still lagged behind BanglaBERT. The use of LoRA for fine-tuning significantly enhanced model performance.

This research establishes a new benchmark for hate speech detection in Bangla and the authors plan to extend the dataset to develop better models in the future.


<br/>
# Results



이 연구에서는 BanglaMultiHate 데이터셋을 사용하여 방글라어의 혐오 발언 탐지 성능을 평가했습니다. 다양한 모델을 비교하여 성능을 분석하였으며, 주요 결과는 다음과 같습니다.

1. **모델 성능 비교**:
   - **SVM (Support Vector Machine)**: 전통적인 기계 학습 모델인 SVM은 TF-IDF 기반의 특성을 사용하여 모든 작업에서 비교적 일관된 성능을 보였습니다. 예를 들어, 혐오 유형(task)에서 SVM은 0.609의 micro-F1 점수를 기록했습니다.
   - **BanglaBERT**: 방글라어에 특화된 사전 훈련된 모델인 BanglaBERT는 모든 작업에서 가장 높은 성능을 보였습니다. 혐오 유형(task)에서 0.712, 혐오의 심각성(task)에서 0.722, 혐오의 대상(task)에서 0.715의 micro-F1 점수를 기록했습니다. 이는 문화적 및 언어적 맥락을 잘 반영한 결과입니다.
   - **LLMs (Large Language Models)**: Gemini, GPT-5, Llama3, Qwen3와 같은 대형 언어 모델들은 제로샷 및 몇 샷 학습 설정에서 성능을 평가했습니다. 예를 들어, Gemini는 혐오의 심각성(task)에서 0.698의 micro-F1 점수를 기록하며, Qwen3는 0.589로 뒤를 이었습니다. 그러나 이러한 모델들은 여전히 BanglaBERT에 비해 낮은 성능을 보였습니다.

2. **제로샷 및 몇 샷 학습**:
   - 제로샷 학습에서는 BanglaLLM이 모든 작업에서 상대적으로 낮은 성능을 보였으며, Qwen3는 혐오 유형(task)에서 0.520의 micro-F1 점수를 기록했습니다. 이는 제로샷 접근 방식이 방글라어의 혐오 발언 탐지에 충분하지 않음을 시사합니다.
   - 몇 샷 학습에서는 GPT-5가 혐오의 대상(task)에서 약간의 성능 향상을 보였지만, 전반적으로 제로샷 성능과 큰 차이를 보이지 않았습니다.

3. **모델의 세부 성능**:
   - BanglaBERT는 모든 작업에서 가장 높은 성능을 보였으며, 특히 혐오의 심각성(task)에서 0.722로 가장 높은 점수를 기록했습니다.
   - Llama3와 Qwen3는 LoRA(저랭크 적응)로 미세 조정했을 때 성능이 향상되었으며, Llama3는 혐오 유형(task)에서 0.620의 micro-F1 점수를 기록했습니다.

4. **데이터셋의 불균형**:
   - 데이터셋은 클래스 간 불균형이 존재하여, 일부 희귀한 카테고리(예: 성차별, 종교적 혐오)의 탐지 성능이 저하되었습니다. 이러한 불균형은 모델의 성능 평가에 영향을 미쳤습니다.

5. **결론**:
   - 연구 결과는 방글라어의 혐오 발언 탐지에서 언어 특화된 사전 훈련의 중요성을 강조하며, 제로샷 접근 방식이 충분하지 않음을 보여줍니다. 또한, LoRA와 같은 미세 조정 기법이 모델 성능을 향상시키는 데 효과적임을 입증했습니다.




In this study, we evaluated the performance of hate speech detection in Bangla using the BanglaMultiHate dataset. Various models were compared, and the key findings are as follows:

1. **Model Performance Comparison**:
   - **SVM (Support Vector Machine)**: The traditional machine learning model SVM showed relatively consistent performance across all tasks using TF-IDF based features. For instance, in the hate type task, SVM achieved a micro-F1 score of 0.609.
   - **BanglaBERT**: The language-specific pre-trained model BanglaBERT outperformed all other models across all tasks, achieving micro-F1 scores of 0.712 for hate type, 0.722 for hate severity, and 0.715 for hate target. This reflects its ability to capture cultural and linguistic nuances effectively.
   - **LLMs (Large Language Models)**: Large language models such as Gemini, GPT-5, Llama3, and Qwen3 were evaluated under zero-shot and few-shot learning settings. For example, Gemini achieved a micro-F1 score of 0.698 in the hate severity task, while Qwen3 followed with 0.589. However, these models still performed lower than BanglaBERT.

2. **Zero-shot and Few-shot Learning**:
   - In zero-shot learning, BanglaLLM showed relatively low performance across all tasks, with Qwen3 achieving a micro-F1 score of 0.520 in the hate type task. This indicates that zero-shot approaches are insufficient for hate speech detection in Bangla.
   - In few-shot learning, GPT-5 showed slight performance improvements only in the hate target task, while overall, it did not significantly differ from zero-shot performance.

3. **Detailed Model Performance**:
   - BanglaBERT consistently achieved the highest scores across all tasks, particularly in the hate severity task with a score of 0.722.
   - Llama3 and Qwen3 showed performance improvements when fine-tuned with LoRA, with Llama3 achieving a micro-F1 score of 0.620 in the hate type task.

4. **Dataset Imbalance**:
   - The dataset exhibited class imbalance, which affected the detection performance of some rare categories (e.g., sexism, religious hate). This imbalance influenced the evaluation of model performance.

5. **Conclusion**:
   - The findings emphasize the importance of language-specific pre-training for hate speech detection in Bangla and indicate that zero-shot approaches are insufficient. Additionally, fine-tuning techniques like LoRA effectively enhance model performance.


<br/>
# 예제



이 논문에서는 BanglaMultiHate라는 다중 작업을 지원하는 방글라어 혐오 발언 데이터셋을 소개합니다. 이 데이터셋은 방글라어로 작성된 댓글을 수집하여, 혐오 발언의 유형, 심각도, 그리고 타겟을 식별하는 세 가지 작업을 수행할 수 있도록 설계되었습니다. 데이터셋은 총 50,746개의 댓글로 구성되어 있으며, 각 댓글은 다음과 같은 세 가지 작업에 대해 레이블이 지정됩니다.

1. **혐오 발언 유형 (Type of Hate)**: 댓글이 어떤 유형의 혐오 발언인지 분류합니다. 가능한 레이블은 'Abusive' (모욕적), 'Sexism' (성차별), 'Religious Hate' (종교적 혐오), 'Political Hate' (정치적 혐오), 'Profane' (욕설), 'None' (혐오 없음)입니다.
   
   - **예시**: 
     - 입력: "তুমি একেবারে অকার্যকর।" (영어: "You are completely useless.")
     - 출력: {"type_of_hate": "Abusive"}

2. **혐오 발언 심각도 (Severity of Hate)**: 댓글의 혐오 발언이 얼마나 심각한지를 평가합니다. 레이블은 'Little to None' (거의 없음), 'Mild' (경미함), 'Severe' (심각함)입니다.
   
   - **예시**: 
     - 입력: "তুমি পিগের ছেলে, তোমার সাহস আছে।" (영어: "Son of a pig, you got guts.")
     - 출력: {"severity_of_hate": "Severe"}

3. **혐오 발언 타겟 (Target of Hate)**: 혐오 발언이 어떤 대상을 향하고 있는지를 식별합니다. 가능한 레이블은 'Community' (커뮤니티), 'Individual' (개인), 'Organization' (조직), 'Society' (사회), 'None' (대상 없음)입니다.
   
   - **예시**: 
     - 입력: "এই ধর্মের লোকেরা সব খারাপ।" (영어: "All people in this religion are bad.")
     - 출력: {"target_of_hate": "Community"}

이 데이터셋은 훈련 데이터와 테스트 데이터로 나뉘며, 각 작업에 대해 다양한 모델을 평가하는 데 사용됩니다. 예를 들어, SVM, BanglaBERT, 그리고 대형 언어 모델(LLM)들이 사용되어 성능을 비교합니다. 훈련 데이터는 모델이 학습하는 데 사용되며, 테스트 데이터는 모델의 일반화 능력을 평가하는 데 사용됩니다.




This paper introduces the BanglaMultiHate dataset, which supports multiple tasks for hate speech detection in the Bangla language. The dataset consists of a total of 50,746 comments collected from YouTube, and it is designed to perform three tasks: identifying the type of hate speech, the severity of hate, and the target of hate. Each comment is labeled for these three tasks as follows:

1. **Type of Hate**: This task classifies the type of hate speech expressed in the comment. Possible labels include 'Abusive', 'Sexism', 'Religious Hate', 'Political Hate', 'Profane', and 'None'.
   
   - **Example**: 
     - Input: "তুমি একেবারে অকার্যকর।" (English: "You are completely useless.")
     - Output: {"type_of_hate": "Abusive"}

2. **Severity of Hate**: This task assesses the degree of hate expressed in the comment. The labels are 'Little to None', 'Mild', and 'Severe'.
   
   - **Example**: 
     - Input: "তুমি পিগের ছেলে, তোমার সাহস আছে।" (English: "Son of a pig, you got guts.")
     - Output: {"severity_of_hate": "Severe"}

3. **Target of Hate**: This task focuses on identifying the specific target of the hateful expression. Possible labels include 'Community', 'Individual', 'Organization', 'Society', and 'None'.
   
   - **Example**: 
     - Input: "এই ধর্মের লোকেরা সব খারাপ।" (English: "All people in this religion are bad.")
     - Output: {"target_of_hate": "Community"}

The dataset is divided into training and testing data, which are used to evaluate various models. For instance, models such as SVM, BanglaBERT, and large language models (LLMs) are employed to compare their performance. The training data is used for model learning, while the testing data is used to assess the model's generalization ability.

<br/>
# 요약


이 논문에서는 BanglaMultiHate라는 첫 번째 다중 작업 방글라 증오 발언 데이터셋을 개발하고, 이를 통해 다양한 모델을 비교하여 방글라어 증오 발언 탐지의 성능을 평가하였다. 실험 결과, LoRA로 미세 조정된 대형 언어 모델(LLM)이 BanglaBERT와 유사한 성능을 보였지만, 문화적으로 기반한 사전 훈련이 여전히 중요하다는 것을 강조하였다. 이 연구는 방글라어와 같은 저자원 언어에서의 증오 발언 탐지에 대한 강력한 기준을 설정하였다.

---

This paper develops the first multi-task Bangla hate speech dataset, BanglaMultiHate, and evaluates various models to assess performance in Bangla hate speech detection. The results show that LoRA-tuned large language models (LLMs) rival BanglaBERT, but emphasize the continued importance of culturally grounded pretraining. This research establishes a stronger benchmark for hate speech detection in low-resource languages like Bangla.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **피규어 1**: 증오 발언의 예시와 그에 대한 유형, 심각도, 타겟을 보여줍니다. 이는 연구의 주요 목표인 다중 작업 분류의 필요성을 강조합니다.
   - **피규어 2**: BanglaMultiHate 데이터셋의 카테고리별 분포를 나타내며, 대부분의 댓글이 특정 카테고리에 집중되어 있음을 보여줍니다. 이는 데이터의 불균형 문제를 시사합니다.
   - **피규어 3 및 4**: 증오 유형과 심각도, 증오 유형과 타겟 간의 관계를 시각화합니다. 이들 피규어는 특정 유형의 증오가 특정 심각도와 타겟에 어떻게 연결되는지를 보여줍니다.

2. **테이블**
   - **테이블 1**: 기존 데이터셋과 BanglaMultiHate 데이터셋의 비교를 통해, BanglaMultiHate가 가장 큰 수동 주석 데이터셋임을 강조합니다.
   - **테이블 3**: 각 작업에 대한 클래스 레이블 분포를 보여주며, 데이터셋의 불균형을 명확히 드러냅니다. 이는 모델 훈련 시 특정 클래스의 성능 저하를 초래할 수 있습니다.
   - **테이블 4**: 다양한 모델의 성능을 비교하여, BanglaBERT가 모든 작업에서 가장 높은 성능을 보임을 나타냅니다. 이는 언어 특화된 사전 훈련 모델의 중요성을 강조합니다.

3. **어펜딕스**
   - **어펜딕스 A**: 주석 가이드라인을 제공하여, 각 작업의 정의와 주석 기준을 명확히 합니다. 이는 주석의 일관성을 높이는 데 기여합니다.
   - **어펜딕스 C**: 훈련 및 검증 손실 분석을 통해 모델의 안정성을 평가합니다. 손실이 안정적으로 감소하는 경향을 보여주며, 과적합의 증거가 없음을 나타냅니다.
   - **어펜딕스 D**: 교차 도메인 실험 결과를 통해, BanglaLLM이 데이터셋 훈련된 기준선보다 낮은 성능을 보임을 보여줍니다. 이는 도메인 간 전이의 어려움을 강조합니다.

### Insights from Figures, Tables, and Appendices

1. **Figures**
   - **Figure 1**: Provides an example of hate speech, illustrating the type, severity, and target. This emphasizes the need for multi-task classification, which is the main goal of the study.
   - **Figure 2**: Shows the distribution of categories in the BanglaMultiHate dataset, indicating that most comments are concentrated in specific categories. This suggests potential issues with data imbalance.
   - **Figures 3 and 4**: Visualize the relationships between hate type and severity, and hate type and target, respectively. These figures demonstrate how certain types of hate correlate with specific severity levels and targets.

2. **Tables**
   - **Table 1**: Compares existing datasets with BanglaMultiHate, highlighting that it is the largest manually annotated dataset to date.
   - **Table 3**: Displays the class label distribution for each task, clearly revealing the imbalance in the dataset. This could lead to performance degradation for certain classes during model training.
   - **Table 4**: Compares the performance of various models, showing that BanglaBERT consistently outperforms others across all tasks. This underscores the importance of language-specific pre-trained models.

3. **Appendices**
   - **Appendix A**: Provides annotation guidelines, clarifying the definitions and criteria for each task. This contributes to the consistency of the annotations.
   - **Appendix C**: Analyzes training and validation loss to assess model stability. The steady decrease in loss indicates stable optimization with no evidence of overfitting.
   - **Appendix D**: Presents cross-domain experiment results, showing that BanglaLLM performs significantly lower than dataset-trained baselines, reinforcing the challenges of domain transfer.

These insights collectively highlight the importance of dataset quality, model selection, and the need for task-specific adaptations in hate speech detection, particularly in low-resource languages like Bangla.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{hasan2026llm,
  author    = {Md Arid Hasan and Firoj Alam and Md Fahad Hossain and Usman Naseem and Syed Ishtiaque Ahmed},
  title     = {LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {33962--33980},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### Chicago Style Citation

Hasan, Md Arid, Firoj Alam, Md Fahad Hossain, Usman Naseem, and Syed Ishtiaque Ahmed. 2026. "LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 33962–33980. Association for Computational Linguistics.
    