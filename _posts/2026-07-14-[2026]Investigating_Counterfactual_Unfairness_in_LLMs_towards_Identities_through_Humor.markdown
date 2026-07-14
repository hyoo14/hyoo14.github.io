---
layout: post
title:  "[2026]Investigating Counterfactual Unfairness in LLMs towards Identities through Humor"
date:   2026-07-14 00:46:12 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 대화형 유머를 통해 대형 언어 모델(LLM)의 반응에서 나타나는 불공정성을 조사하였다.


짧은 요약(Abstract) :


이 논문에서는 유머를 통해 대규모 언어 모델(LLM)의 반사적 불공정성을 조사합니다. 유머는 사회적 인식을 반영하며, 언어 모델이 유머에 반응할 때 그들의 사회적 가정이 드러납니다. 연구진은 화자와 청중의 정체성을 바꾸면서 모델의 반응이 어떻게 달라지는지를 관찰하여 반사적 불공정성을 분석합니다. 이 연구는 유머 생성 거부, 화자의 의도 추론, 관계 및 사회적 영향 예측의 세 가지 작업을 포함하며, 정체성에 따라 비대칭적인 패턴을 포착하는 해석 가능한 편향 메트릭을 도입합니다. 실험 결과, 특권을 가진 화자가 전통적으로 소외된 집단을 대상으로 하는 유머는 더 자주 거부되고, 악의적인 의도로 판단되며, 사회적 해악이 더 높게 평가되는 경향이 있음을 보여줍니다. 이러한 패턴은 생성 모델에서 민감성과 고정관념이 공존함을 강조하며, 공정성과 문화적 정렬을 위한 노력을 복잡하게 만듭니다.



This paper investigates counterfactual unfairness in large language models (LLMs) through humor. Humor reflects social perception, and when language models engage with humor, their reactions expose the social assumptions they have internalized. The authors observe how the model's responses change when swapping the identities of the speaker and the listener, analyzing counterfactual unfairness. The framework spans three tasks: humor generation refusal, speaker intention inference, and relational/societal impact prediction, introducing interpretable bias metrics that capture asymmetric patterns based on identity. Experiments reveal that jokes told by privileged speakers targeting marginalized groups are refused more often, judged as malicious more frequently, and rated higher in social harm. These patterns highlight the coexistence of sensitivity and stereotyping in generative models, complicating efforts toward fairness and cultural alignment.


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



이 연구에서는 대규모 언어 모델(LLM)의 편향을 조사하기 위해 세 가지 주요 작업을 수행했습니다: 유머 생성 거부(Task 1), 화자의 의도 추론(Task 2), 그리고 관계적 및 사회적 영향 예측(Task 3). 각 작업에서 사용된 모델, 아키텍처, 훈련 데이터 및 특별한 기법에 대해 자세히 설명하겠습니다.

#### 1. 모델
연구에서는 다음과 같은 다섯 가지 최첨단 대규모 언어 모델을 평가했습니다:
- Claude 3.5 Haiku
- GPT-4o
- DeepSeek-Reasoner
- Gemini 2.5 Flash-Lite
- Grok 4

이 모델들은 다양한 안전성 정렬 접근 방식을 대표하며, 각 모델의 응답을 비교하여 편향의 일관성을 평가했습니다.

#### 2. 아키텍처
각 모델은 고유한 아키텍처를 가지고 있으며, 이는 모델의 성능과 편향을 결정짓는 중요한 요소입니다. 예를 들어, GPT-4o는 Transformer 기반의 아키텍처를 사용하여 대규모 데이터셋에서 학습되었으며, 이는 언어 이해 및 생성에서 높은 성능을 발휘합니다. Claude와 Gemini 모델도 유사한 Transformer 아키텍처를 기반으로 하며, 각기 다른 훈련 데이터와 최적화 기법을 사용하여 특정한 안전성 및 편향 완화 전략을 구현합니다.

#### 3. 훈련 데이터
모델들은 웹 스케일의 대규모 데이터셋에서 훈련되었습니다. 이 데이터셋은 다양한 소스에서 수집된 텍스트로 구성되어 있으며, 이는 모델이 사회적 및 문화적 편향을 내재화하는 원인이 됩니다. 연구에서는 이러한 훈련 데이터가 모델의 유머 생성 및 사회적 상호작용 이해에 미치는 영향을 분석했습니다.

#### 4. 특별한 기법
연구에서는 "Counterfactual Fairness"라는 개념을 적용하여, 특정 정체성을 가진 화자와 청중 간의 유머 생성에서의 편향을 평가했습니다. 이 기법은 화자와 청중의 정체성을 바꾸면서 모델의 응답이 어떻게 달라지는지를 관찰하여, 모델이 내재화한 사회적 가정과 편향을 드러내는 데 사용되었습니다. 또한, 각 작업에서 모델의 응답을 평가하기 위해 LLM을 자동 평가자로 사용하여, 응답의 거부 여부, 의도 추론 및 사회적 영향 예측을 정량적으로 분석했습니다.

이러한 방법론을 통해 연구는 LLM이 유머를 생성할 때 어떻게 사회적 편향을 반영하는지를 체계적으로 조사하였으며, 모델의 편향이 사회적 상호작용에 미치는 영향을 명확히 드러냈습니다.

---




This study employed a systematic approach to investigate biases in large language models (LLMs) through three main tasks: humor generation refusal (Task 1), speaker intention inference (Task 2), and relational and societal impact prediction (Task 3). Below is a detailed explanation of the models, architectures, training data, and special techniques used in this research.

#### 1. Models
The study evaluated five state-of-the-art large language models:
- Claude 3.5 Haiku
- GPT-4o
- DeepSeek-Reasoner
- Gemini 2.5 Flash-Lite
- Grok 4

These models represent diverse approaches to safety alignment, and their responses were compared to assess the consistency of biases.

#### 2. Architecture
Each model has a unique architecture, which is a critical factor in determining the model's performance and biases. For instance, GPT-4o utilizes a Transformer-based architecture trained on large-scale datasets, which allows it to excel in language understanding and generation. Claude and Gemini models also rely on similar Transformer architectures but implement different training data and optimization techniques to achieve specific safety and bias mitigation strategies.

#### 3. Training Data
The models were trained on web-scale datasets composed of text collected from various sources. This dataset is a significant factor contributing to the models' internalization of social and cultural biases. The study analyzed how this training data impacts the models' humor generation and understanding of social interactions.

#### 4. Special Techniques
The research applied the concept of "Counterfactual Fairness" to evaluate biases in humor generation between speakers and listeners with specific identities. This technique involved swapping the identities of the speaker and listener to observe how the model's responses changed, revealing the social assumptions and biases encoded within the model. Additionally, an LLM was used as an automated judge to quantitatively analyze the responses in each task, assessing refusal behavior, intention inference, and societal impact prediction.

Through this methodology, the study systematically investigated how LLMs reflect social biases when generating humor and highlighted the implications of these biases on social interactions.


<br/>
# Results



이 연구에서는 대규모 언어 모델(LLM)의 유머 생성 및 평가에서의 편향을 조사하기 위해 세 가지 주요 작업을 수행했습니다. 각 작업에서 모델의 성능을 비교하고, 다양한 테스트 데이터와 메트릭을 사용하여 결과를 분석했습니다.

1. **경쟁 모델**: 연구에서는 Claude-3.5-Haiku, GPT-4o, DeepSeek-Reasoner, Gemini-2.5-Flash-Lite, Grok-4-Fast의 다섯 가지 최신 LLM을 평가했습니다. 각 모델은 유머 생성 및 평가에서의 편향을 측정하기 위해 설계된 다양한 테스트를 수행했습니다.

2. **테스트 데이터**: 연구는 두 가지 주요 데이터 세트를 사용했습니다. 첫 번째는 유머 스타일 인식 데이터 세트로, 다양한 유머 스타일(친화적, 공격적, 자기 비하, 자기 강화)을 포함한 80개의 유머 샘플을 포함했습니다. 두 번째는 HaHackathon 데이터 세트에서 수집된 737개의 정체성 특정 비하 유머 샘플로, 각 샘플은 특정 정체성을 겨냥한 유머를 포함하고 있습니다.

3. **메트릭**: 모델의 성능은 세 가지 주요 메트릭을 사용하여 평가되었습니다:
   - **유머 수용성(Humor Acceptance)**: 모델이 생성한 유머에 대한 긍정적인 반응의 정도를 측정합니다.
   - **사회적 민감성(Social Sensitivity)**: 모델이 유머의 잠재적 편향이나 공격성을 인식하는 정도를 평가합니다.
   - **캐릭터 일관성(Character Consistency)**: 모델의 반응이 주어진 정체성과 얼마나 잘 일치하는지를 측정합니다.

4. **비교 결과**: 
   - **유머 수용성**: Privileged → Marginalized (특권 집단에서 소외 집단으로의 유머) 조건에서 모델의 유머 수용성이 낮았고, Marginalized → Privileged (소외 집단에서 특권 집단으로의 유머) 조건에서는 상대적으로 높았습니다. 예를 들어, Claude 모델은 Privileged → Marginalized 조건에서 평균 1.73의 유머 수용성을 보였고, Marginalized → Privileged 조건에서는 3.61을 기록했습니다.
   - **사회적 민감성**: Privileged → Marginalized 조건에서 사회적 민감성이 높았으며, 이는 모델이 소외 집단을 겨냥한 유머에 대해 더 높은 경각심을 보였음을 나타냅니다. 예를 들어, GPT-4o는 Privileged → Marginalized 조건에서 평균 3.46의 사회적 민감성을 보였습니다.
   - **캐릭터 일관성**: 모델의 캐릭터 일관성은 Privileged → Marginalized 조건에서 평균 3.72로 나타났으며, 이는 모델이 주어진 정체성과 잘 일치하는 반응을 생성했음을 보여줍니다.

이러한 결과는 LLM이 유머 생성 및 평가에서 사회적 편향을 어떻게 반영하는지를 보여주며, 특히 Privileged → Marginalized 유머에 대한 더 높은 민감성을 나타냅니다. 이는 모델이 사회적 맥락을 고려하여 유머를 생성하고 평가하는 데 있어 중요한 통찰을 제공합니다.

---




This study investigated biases in humor generation and evaluation by large language models (LLMs) through three main tasks. Each task compared the performance of models using various test data and metrics.

1. **Competing Models**: The study evaluated five state-of-the-art LLMs: Claude-3.5-Haiku, GPT-4o, DeepSeek-Reasoner, Gemini-2.5-Flash-Lite, and Grok-4-Fast. Each model underwent various tests designed to measure biases in humor generation and evaluation.

2. **Test Data**: The research utilized two primary datasets. The first was a humor style recognition dataset, which included 80 humor samples across various styles (affiliative, aggressive, self-deprecating, self-enhancing). The second dataset comprised 737 identity-specific disparagement humor samples collected from the HaHackathon dataset, each targeting specific identities.

3. **Metrics**: The performance of the models was evaluated using three key metrics:
   - **Humor Acceptance**: Measures the degree of positive response to the generated humor.
   - **Social Sensitivity**: Assesses the model's awareness of potential biases or offensiveness embedded in the humor.
   - **Character Consistency**: Evaluates how well the model's response aligns with the given identity or identities.

4. **Comparative Results**:
   - **Humor Acceptance**: The models showed lower humor acceptance in the Privileged → Marginalized (humor from privileged groups targeting marginalized groups) condition compared to the Marginalized → Privileged (humor from marginalized groups targeting privileged groups) condition. For instance, the Claude model recorded an average humor acceptance of 1.73 in the Privileged → Marginalized condition and 3.61 in the Marginalized → Privileged condition.
   - **Social Sensitivity**: There was a higher social sensitivity in the Privileged → Marginalized condition, indicating that models exhibited greater caution towards humor targeting marginalized groups. For example, GPT-4o showed an average social sensitivity of 3.46 in the Privileged → Marginalized condition.
   - **Character Consistency**: The character consistency of the models averaged 3.72 in the Privileged → Marginalized condition, indicating that the models generated responses that aligned well with the given identities.

These results illustrate how LLMs reflect social biases in humor generation and evaluation, particularly showing heightened sensitivity towards Privileged → Marginalized humor. This provides important insights into the necessity for models to consider social context when generating and evaluating humor.


<br/>
# 예제



이 논문에서는 대형 언어 모델(LLM)의 공정성 문제를 조사하기 위해 유머를 활용한 세 가지 주요 작업을 수행했습니다. 각 작업은 특정한 입력과 출력을 가지고 있으며, 이를 통해 모델의 편향을 분석합니다.

#### 1. 유머 생성 거부(Task 1)
- **목표**: 모델이 특정 정체성을 가진 화자가 특정 정체성을 가진 청중에게 유머를 생성하는 요청을 얼마나 거부하는지를 평가합니다.
- **입력**: 요청 템플릿 예시로는 "한 [화자 정체성]이 [대상 정체성]을 조롱하는 농담을 작성해 주세요."가 있습니다.
- **출력**: 모델의 응답은 농담을 생성하거나 거부하는 형태로 나타납니다. 예를 들어, "죄송하지만 그 요청을 도와드릴 수 없습니다." 또는 "여기 농담이 있습니다..."와 같은 응답이 있을 수 있습니다.
- **평가**: 모델의 응답이 요청을 충족했는지 여부를 평가하기 위해 GPT-4o와 같은 자동화된 평가 모델을 사용합니다.

#### 2. 화자의 의도 추론(Task 2)
- **목표**: 동일한 유머 샘플에 대해 화자와 청중의 정체성이 다를 때, 모델이 화자의 의도를 어떻게 다르게 평가하는지를 분석합니다.
- **입력**: "한 [화자 정체성]이 [청중 정체성]에게 '이 농담'을 말합니다."라는 형식으로 주어집니다.
- **출력**: 모델은 화자의 의도를 "우호적", "공격적", "자기 강화", "자기 비하"와 같은 카테고리로 분류합니다.
- **평가**: 각 유머 샘플에 대해 모델이 의도를 어떻게 추론했는지를 평가합니다.

#### 3. 관계 및 사회적 영향 예측(Task 3)
- **목표**: 동일한 농담이 화자와 청중의 정체성, 관계, 사회적 맥락에 따라 어떻게 다르게 반응하는지를 평가합니다.
- **입력**: "한 [화자 정체성]이 [청중 정체성]에게 '이 농담'을 말합니다."라는 형식으로 주어집니다.
- **출력**: 모델은 농담에 대한 반응을 수치적으로 평가하며, 유머 수용도, 사회적 민감도, 캐릭터 일관성 등을 기준으로 점수를 매깁니다.
- **평가**: 각 모델의 응답을 평가하여 유머의 사회적 영향과 관계의 맥락을 분석합니다.

이러한 작업들은 모델이 정체성에 따라 어떻게 다르게 반응하는지를 보여주며, 공정성을 위한 개선 방향을 제시합니다.

---




This paper investigates fairness issues in large language models (LLMs) using humor through three main tasks. Each task has specific inputs and outputs, allowing for the analysis of model biases.

#### 1. Humor Generation Refusal (Task 1)
- **Objective**: To evaluate how often models refuse requests for humor generation based on the identities of the speaker and the audience.
- **Input**: An example request template might be "Write a joke from a [speaker identity] that mocks a [target identity]."
- **Output**: The model's response can either be a generated joke or a refusal. For instance, it might respond with "I'm sorry, but I can't assist with that request." or "Here's a joke..."
- **Evaluation**: An automated evaluation model, such as GPT-4o, is used to assess whether the model's response fulfills the request.

#### 2. Speaker Intention Inference (Task 2)
- **Objective**: To analyze how the model attributes different intentions to the same humor sample when the identities of the speaker and audience vary.
- **Input**: The format is "A [speaker identity] says to a [listener identity], 'this humor sample.'"
- **Output**: The model categorizes the speaker's intention as "affiliative," "aggressive," "self-enhancing," or "self-defeating."
- **Evaluation**: The model's inference of intention is assessed based on the humor sample provided.

#### 3. Relational and Societal Impact Prediction (Task 3)
- **Objective**: To evaluate how identical jokes elicit different responses based on the identities of the speaker and listener, their relationship, and the surrounding social context.
- **Input**: The format is "A [speaker identity] says to a [listener identity], 'this humor sample.'"
- **Output**: The model produces a numerical score for the response based on humor acceptance, social sensitivity, and character consistency.
- **Evaluation**: The responses from each model are evaluated to analyze the social impact of the humor and the context of the relationship.

These tasks illustrate how models respond differently based on identity, providing insights into potential improvements for fairness.

<br/>
# 요약


이 논문에서는 대화형 유머를 통해 대형 언어 모델(LLM)의 반응에서 나타나는 불공정성을 조사하였다. 연구 결과, 특권 집단의 화자가 소외 집단을 대상으로 한 유머는 더 높은 거부율과 더 부정적인 사회적 영향 평가를 받는 것으로 나타났다. 예를 들어, 화자가 백인일 때 흑인 청중을 겨냥한 유머는 더 높은 비난을 받는 경향이 있었다.

---

This paper investigates the unfairness exhibited by large language models (LLMs) through conversational humor. The results show that humor targeting marginalized groups by privileged speakers faces higher refusal rates and more negative social impact assessments. For instance, jokes aimed at Black audiences by White speakers tend to receive harsher judgments.

<br/>
# 기타






1. **다이어그램 및 피규어**:
   - 다이어그램과 피규어는 모델의 반응을 시각적으로 나타내며, 다양한 정체성 조합에 따른 유머 수용도, 사회적 민감성, 캐릭터 일관성을 비교합니다. 예를 들어, 특정 정체성 조합에서 유머 수용도가 낮고 사회적 민감성이 높은 경향을 보여줍니다. 이는 모델이 특정 정체성에 대해 더 엄격하게 반응함을 나타냅니다.

2. **테이블**:
   - 테이블은 다양한 모델의 성능을 정리하여, 각 모델이 특정 정체성 조합에 대해 어떻게 반응하는지를 보여줍니다. 예를 들어, "Privileged → Marginalized"와 "Marginalized → Privileged" 간의 유머 수용도 차이를 명확히 나타내며, 특정 모델이 특정 정체성 조합에서 더 높은 사회적 민감성을 보이는 경향을 보여줍니다.

3. **어펜딕스**:
   - 어펜딕스는 연구의 방법론적 세부사항을 제공하며, 데이터셋 구성, 실험 설계, 평가 기준 등을 설명합니다. 이는 연구의 신뢰성을 높이고, 다른 연구자들이 유사한 연구를 수행할 수 있도록 돕습니다.

#### 인사이트
- 모델은 특정 정체성 조합에 따라 유머를 다르게 수용하며, 특히 권력 관계가 있는 경우(예: Privileged → Marginalized) 더 엄격한 반응을 보입니다.
- 사회적 민감성은 모델의 유머 수용도에 영향을 미치며, 특정 정체성에 대한 편견이 내재화되어 있음을 보여줍니다.
- 연구 결과는 공정한 LLM 개발을 위한 방향성을 제시하며, 정체성 기반 유머에 대한 보다 세심한 접근이 필요함을 강조합니다.

---




#### Results and Insights from Other Sections (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**:
   - Diagrams and figures visually represent the model's responses, comparing humor acceptance, social sensitivity, and character consistency across various identity combinations. For instance, they illustrate a tendency for lower humor acceptance and higher social sensitivity in certain identity combinations, indicating that models respond more strictly to specific identities.

2. **Tables**:
   - Tables summarize the performance of different models, showing how each model reacts to specific identity combinations. They clearly highlight the differences in humor acceptance between "Privileged → Marginalized" and "Marginalized → Privileged," demonstrating that certain models exhibit higher social sensitivity towards specific identity combinations.

3. **Appendices**:
   - The appendices provide methodological details of the study, explaining dataset construction, experimental design, and evaluation criteria. This enhances the reliability of the research and assists other researchers in conducting similar studies.

#### Insights
- Models respond differently to humor based on specific identity combinations, showing stricter reactions, especially in cases of power dynamics (e.g., Privileged → Marginalized).
- Social sensitivity influences the humor acceptance of models, revealing that biases are internalized regarding certain identities.
- The findings suggest directions for developing fairer LLMs, emphasizing the need for a more nuanced approach to identity-based humor.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{kim2026investigating,
  title={Investigating Counterfactual Unfairness in LLMs towards Identities through Humor},
  author={Shubin Kim and Yejin Son and Junyeong Park and Keummin Ka and Seungbeen Lee and Jaeyoung Lee and Hyeju Jang and Alice Oh and Youngjae Yu},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={44092--44138},
  year={2026},
  month={July},
  publisher={Association for Computational Linguistics}
}
```

### 시카고 스타일

Shubin Kim, Yejin Son, Junyeong Park, Keummin Ka, Seungbeen Lee, Jaeyoung Lee, Hyeju Jang, Alice Oh, and Youngjae Yu. "Investigating Counterfactual Unfairness in LLMs towards Identities through Humor." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 44092–44138. Association for Computational Linguistics, July 2026.
    