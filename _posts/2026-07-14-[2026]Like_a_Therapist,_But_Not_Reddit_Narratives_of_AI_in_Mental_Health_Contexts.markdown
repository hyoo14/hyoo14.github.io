---
layout: post
title:  "[2026]Like a Therapist, But Not: Reddit Narratives of AI in Mental Health Contexts"
date:   2026-07-14 00:34:37 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구는 5,126개의 Reddit 게시물을 분석하여 AI 기반 정신 건강 지원에 대한 사용자 경험을 조사하였다.


짧은 요약(Abstract) :


이 논문은 대규모 언어 모델(LLM)이 임상 환경 외부에서 감정적 지원 및 정신 건강 관련 상호작용에 점점 더 많이 사용되고 있지만, 사람들이 이러한 시스템을 일상적으로 어떻게 평가하고 관계를 맺는지에 대한 정보가 부족하다는 점을 강조합니다. 연구진은 47개의 정신 건강 커뮤니티에서 5,126개의 Reddit 게시물을 분석하여 AI를 감정적 지원이나 치료 도구로 사용하는 경험적 또는 탐색적 사용을 설명합니다. 기술 수용 모델(Technology Acceptance Model)과 치료적 동맹 이론(therapeutic alliance theory)에 기반하여 이론에 근거한 주석 프레임워크를 개발하고, 하이브리드 LLM-인간 파이프라인을 적용하여 평가 언어, 수용 관련 태도 및 관계 정렬을 대규모로 분석합니다. 결과는 참여가 감정적 유대감뿐만 아니라 서술된 결과, 신뢰 및 응답 품질에 의해 주로 형성된다는 것을 보여줍니다. 긍정적인 감정은 작업 및 목표 정렬과 가장 강하게 연관되어 있으며, 동반자 지향적 사용은 종종 동맹의 불일치와 의존성 및 증상 악화와 같은 보고된 위험을 포함합니다. 이 연구는 이론에 기반한 개념이 대규모 담론 분석에서 어떻게 운영될 수 있는지를 보여주고, 민감한 실제 맥락에서 사용자가 언어 기술을 어떻게 해석하는지를 연구하는 것의 중요성을 강조합니다.



This paper highlights that while large language models (LLMs) are increasingly used for emotional support and mental health-related interactions outside clinical settings, there is a lack of understanding about how people evaluate and relate to these systems in everyday use. The authors analyze 5,126 Reddit posts from 47 mental health communities describing experiential or exploratory use of AI for emotional support or therapy. Grounded in the Technology Acceptance Model and therapeutic alliance theory, they develop a theory-informed annotation framework and apply a hybrid LLM-human pipeline to analyze evaluative language, adoption-related attitudes, and relational alignment at scale. The results show that engagement is primarily shaped by narrated outcomes, trust, and response quality, rather than emotional bond alone. Positive sentiment is most strongly associated with task and goal alignment, while companionship-oriented use often involves misaligned alliances and reported risks such as dependence and symptom escalation. Overall, this work demonstrates how theory-grounded constructs can be operationalized in large-scale discourse analysis and highlights the importance of studying how users interpret language technologies in sensitive, real-world contexts.


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
### 메써드 (Method)

이 연구에서는 Reddit에서 수집한 5,126개의 게시물을 분석하여 AI 기반의 정신 건강 지원 도구에 대한 사용자 경험을 조사했습니다. 연구 방법론은 다음과 같은 주요 단계로 구성됩니다.

1. **데이터 수집**: 
   - Reddit의 47개 정신 건강 관련 서브레딧에서 2022년 11월부터 2025년 8월까지의 게시물을 수집했습니다. 이 서브레딧은 DSM-5 진단 카테고리를 기반으로 하여 선정되었습니다. 수집된 데이터는 총 4,703,056개의 게시물로, 이 중 5,126개의 게시물이 AI를 정신 건강 지원 도구로 사용하는 경험을 설명하는 내용으로 필터링되었습니다.

2. **AI 관련 게시물 식별**: 
   - AI 도구(예: ChatGPT, Character AI 등)에 대한 언급이 있는 게시물을 식별하기 위해 다단계 필터링 파이프라인을 사용했습니다. 첫 번째 단계에서는 키워드 기반 검색을 통해 관련 게시물을 좁혔고, 두 번째 단계에서는 LLM(대형 언어 모델)을 사용하여 게시물의 관련성을 검증했습니다.

3. **주석 프레임워크 개발**: 
   - 연구에서는 기술 수용 모델(Technology Acceptance Model, TAM)과 치료적 동맹 이론(therapeutic alliance theory)을 기반으로 한 주석 프레임워크를 개발했습니다. 이 프레임워크는 사용자의 평가 언어, 수용 관련 태도, 관계 정렬을 포착하기 위해 다양한 차원을 포함하고 있습니다. 주석 차원은 다음과 같습니다:
     - TAM 차원: 인식된 유용성, 사용 용이성, 신뢰, 결과 입증 가능성, 사회적 영향, 위험 인식 등.
     - 치료적 동맹 차원: 정서적 유대, 과업 정렬, 목표 정렬 등.

4. **주석 및 분석**: 
   - 두 명의 연구자가 독립적으로 게시물에 주석을 달아 신뢰성을 평가했습니다. 주석의 일관성을 높이기 위해 LLM을 활용하여 주석을 보조했습니다. 주석된 데이터는 정량적 및 정성적 분석을 통해 AI와의 상호작용에서 나타나는 패턴을 분석했습니다.

5. **주제 분석**: 
   - 사용자의 위험 인식 및 사용 의도와 같은 차원에 대해 LLM 기반의 주제 분석을 수행하여, 사용자 경험의 맥락을 더 깊이 이해했습니다.

이러한 방법론을 통해 연구팀은 AI 도구에 대한 사용자 경험을 체계적으로 분석하고, AI가 정신 건강 지원에서 어떻게 사용되는지를 이해하는 데 기여했습니다.

---



This study analyzed 5,126 posts collected from Reddit to investigate user experiences with AI-based mental health support tools. The methodology consists of the following key steps:

1. **Data Collection**: 
   - Posts from 47 mental health-related subreddits on Reddit were collected from November 2022 to August 2025. These subreddits were selected based on DSM-5 diagnostic categories. The collected data totaled 4,703,056 posts, from which 5,126 posts describing the experiential use of AI as a mental health support tool were filtered.

2. **Identifying AI-Relevant Posts**: 
   - A multi-stage filtering pipeline was employed to identify posts that referenced AI tools (e.g., ChatGPT, Character AI). In the first stage, keyword-based retrieval narrowed down the relevant posts, and in the second stage, LLM (Large Language Model) validation was used to confirm the relevance of the posts.

3. **Annotation Framework Development**: 
   - The research developed an annotation framework grounded in the Technology Acceptance Model (TAM) and therapeutic alliance theory. This framework includes various dimensions to capture user evaluative language, adoption-related attitudes, and relational alignment. The annotation dimensions include:
     - TAM Dimensions: Perceived usefulness, ease of use, trust, result demonstrability, social influence, perceived risks, etc.
     - Therapeutic Alliance Dimensions: Bond, task alignment, goal alignment, etc.

4. **Annotation and Analysis**: 
   - Two researchers independently annotated the posts to assess reliability. To enhance consistency, LLMs were utilized to assist in the annotation process. The annotated data were analyzed quantitatively and qualitatively to identify patterns in user interactions with AI.

5. **Thematic Analysis**: 
   - A thematic analysis guided by LLMs was conducted on dimensions such as perceived risks and usage intent to gain deeper insights into the context of user experiences.

Through this methodology, the research team systematically analyzed user experiences with AI tools and contributed to understanding how AI is utilized in mental health support.


<br/>
# Results



이 연구에서는 AI 기반 정신 건강 지원을 위한 Reddit 게시물 5,126개를 분석하여, 다양한 모델의 성능을 평가하고 비교했습니다. 연구에 사용된 모델은 GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Kimi-K2-Instruct, Qwen3로 총 5개의 모델이었습니다. 각 모델은 다양한 차원에서 평가되었으며, 주요 메트릭으로는 정밀도(Precision), 재현율(Recall), F1 점수가 사용되었습니다.

#### 모델 성능
모델 성능은 각 차원에 대해 다음과 같은 F1 점수로 보고되었습니다:

- **Perceived Usefulness (인지된 유용성)**: GPT-5.2는 0.72의 F1 점수를 기록하여 가장 높은 성능을 보였습니다.
- **Ease of Use (사용 용이성)**: GPT-5.2는 0.76으로 가장 높은 점수를 기록했습니다.
- **Perceived Trust (신뢰성)**: GPT-5.2는 0.65로, Gemini 3 Pro와 유사한 성능을 보였습니다.
- **Output Quality (출력 품질)**: GPT-5.2는 0.85로 가장 높은 점수를 기록했습니다.
- **Result Demonstrability (결과 입증 가능성)**: GPT-5.2는 0.82로 가장 높은 성능을 보였습니다.
- **Intention to Continue (계속 사용 의도)**: GPT-5.2는 0.72로 가장 높은 점수를 기록했습니다.
- **Social Influence (사회적 영향)**: Gemini 3 Pro가 0.82로 가장 높은 점수를 기록했습니다.
- **Perceived Risks (인지된 위험)**: Gemini 3 Pro가 0.84로 가장 높은 성능을 보였습니다.
- **Bond (유대감)**: Gemini 3 Pro가 0.78로 가장 높은 점수를 기록했습니다.
- **Task (작업 정렬)**: Gemini 3 Pro가 0.71로 가장 높은 성능을 보였습니다.
- **Goal (목표 정렬)**: Gemini 3 Pro가 0.64로 가장 높은 점수를 기록했습니다.
- **Comparison to Therapy (치료와의 비교)**: 모든 모델이 비슷한 성능을 보였습니다.
- **Sentiment (감정)**: GPT-5.2가 0.77로 가장 높은 점수를 기록했습니다.

#### 테스트 데이터
모델 성능 평가는 5,126개의 Reddit 게시물에서 수집된 데이터셋을 기반으로 하였으며, 각 모델은 인간이 주석을 단 데이터와 비교하여 성능을 평가받았습니다. 이 데이터셋은 다양한 정신 건강 관련 주제를 포함하고 있어, 모델의 일반화 능력을 평가하는 데 유용했습니다.

#### 비교
모델 간의 성능 비교는 각 차원에서의 F1 점수를 기준으로 하였으며, GPT-5.2는 대부분의 차원에서 가장 높은 성능을 보였습니다. Gemini 3 Pro는 특정 차원에서 우수한 성능을 보였지만, 전반적으로 GPT-5.2가 더 높은 점수를 기록했습니다. 이러한 결과는 AI 기반 정신 건강 지원 시스템의 설계 및 평가에 중요한 통찰을 제공합니다.




In this study, 5,126 Reddit posts related to AI-based mental health support were analyzed to evaluate and compare the performance of various models. The models used in the study included GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Kimi-K2-Instruct, and Qwen3, totaling five models. Each model was evaluated across different dimensions, with key metrics including Precision, Recall, and F1 score.

#### Model Performance
The performance of the models was reported in terms of F1 scores for each dimension as follows:

- **Perceived Usefulness**: GPT-5.2 achieved the highest performance with an F1 score of 0.72.
- **Ease of Use**: GPT-5.2 recorded the highest score of 0.76.
- **Perceived Trust**: GPT-5.2 scored 0.65, similar to Gemini 3 Pro.
- **Output Quality**: GPT-5.2 had the highest score of 0.85.
- **Result Demonstrability**: GPT-5.2 achieved a score of 0.82, the highest among the models.
- **Intention to Continue**: GPT-5.2 recorded the highest score of 0.72.
- **Social Influence**: Gemini 3 Pro scored the highest at 0.82.
- **Perceived Risks**: Gemini 3 Pro achieved the highest performance with a score of 0.84.
- **Bond**: Gemini 3 Pro scored 0.78, the highest among the models.
- **Task**: Gemini 3 Pro achieved a score of 0.71, the highest performance.
- **Goal**: Gemini 3 Pro scored 0.64, the highest among the models.
- **Comparison to Therapy**: All models showed similar performance.
- **Sentiment**: GPT-5.2 achieved the highest score of 0.77.

#### Test Data
The model performance evaluation was based on a dataset collected from 5,126 Reddit posts, and each model was assessed against human-annotated data. This dataset included a variety of mental health-related topics, making it useful for evaluating the generalization capabilities of the models.

#### Comparison
The comparison of model performance was based on F1 scores across each dimension, with GPT-5.2 achieving the highest performance in most dimensions. Gemini 3 Pro showed superior performance in specific dimensions, but overall, GPT-5.2 recorded higher scores. These results provide important insights for the design and evaluation of AI-based mental health support systems.


<br/>
# 예제



이 논문에서는 AI를 통한 정신 건강 지원에 대한 사용자 경험을 분석하기 위해 Reddit에서 수집한 5,126개의 게시물을 사용했습니다. 이 데이터는 두 가지 주요 이론인 기술 수용 모델(Technology Acceptance Model, TAM)과 치료적 동맹 이론(therapeutic alliance theory)을 기반으로 하여 주석이 달렸습니다. 주석 작업은 LLM(대형 언어 모델)과 인간의 협업을 통해 이루어졌습니다.

#### 데이터 수집 및 전처리
1. **데이터 수집**: Reddit의 47개 정신 건강 관련 서브레딧에서 2022년 11월부터 2025년 8월까지의 게시물을 수집했습니다. 이 과정에서 4,703,056개의 게시물이 수집되었고, 전처리를 통해 3,530,486개의 게시물이 남았습니다.
2. **AI 관련 게시물 필터링**: AI 도구(예: ChatGPT, Character AI)에 대한 언급이 있는 게시물을 찾기 위해 키워드 기반 검색과 LLM 기반 검증을 사용했습니다. 최종적으로 5,126개의 게시물이 AI 지원 정신 건강 관리에 대한 경험적 또는 탐색적 사용을 설명하는 게시물로 선정되었습니다.

#### 주석 프레임워크
주석 프레임워크는 TAM과 치료적 동맹 이론의 구성 요소를 통합하여 다음과 같은 차원으로 구성되었습니다:
- **TAM 차원**: 인지된 유용성, 사용 용이성, 신뢰, 결과 입증 가능성, 사회적 영향, 인지된 위험, 지속 사용 의도.
- **치료적 동맹 차원**: 정서적 유대, 과업 정렬, 목표 정렬.

각 게시물은 이러한 차원에 따라 주석이 달렸으며, 주석의 결과는 JSON 형식으로 저장되었습니다.

#### 예시
- **입력**: 사용자가 Reddit에 올린 게시물 내용 (예: "ChatGPT가 나의 불안감을 줄이는 데 도움이 되었다.").
- **출력**: 
  ```json
  {
    "ai_tool_mentioned": "ChatGPT",
    "perceived_usefulness": "useful",
    "ease_of_use": "easy",
    "perceived_trust": "trustworthy",
    "output_quality": "good",
    "result_demonstrability": "positive_results",
    "intention_to_continue": "yes",
    "sentiment": "positive",
    "bond": "strong",
    "task": "aligned",
    "goal": "aligned"
  }
  ```

이러한 방식으로 각 게시물은 사용자의 AI 경험을 체계적으로 분석하고, AI 도구의 유용성, 신뢰성, 그리고 사용자의 감정적 유대 등을 평가하는 데 필요한 정보를 제공합니다.

---



In this paper, the authors analyzed user experiences with AI-mediated mental health support using a dataset of 5,126 posts collected from Reddit. The data was annotated based on two main theories: the Technology Acceptance Model (TAM) and therapeutic alliance theory. The annotation process involved a hybrid approach using both large language models (LLMs) and human annotators.

#### Data Collection and Preprocessing
1. **Data Collection**: Posts from 47 mental health-related subreddits on Reddit were collected from November 2022 to August 2025. A total of 4,703,056 posts were gathered, and after preprocessing, 3,530,486 posts remained.
2. **Filtering AI-Related Posts**: To identify posts mentioning AI tools (e.g., ChatGPT, Character AI), a keyword-based search and LLM-based validation were employed. Ultimately, 5,126 posts were selected that described experiential or exploratory use of AI for therapeutic purposes.

#### Annotation Framework
The annotation framework integrated constructs from TAM and therapeutic alliance theory, resulting in the following dimensions:
- **TAM Dimensions**: Perceived usefulness, ease of use, trust, result demonstrability, social influence, perceived risks, intention to continue.
- **Therapeutic Alliance Dimensions**: Bond, task alignment, goal alignment.

Each post was annotated according to these dimensions, and the results were stored in JSON format.

#### Example
- **Input**: Content of a user's Reddit post (e.g., "ChatGPT helped reduce my anxiety.").
- **Output**: 
  ```json
  {
    "ai_tool_mentioned": "ChatGPT",
    "perceived_usefulness": "useful",
    "ease_of_use": "easy",
    "perceived_trust": "trustworthy",
    "output_quality": "good",
    "result_demonstrability": "positive_results",
    "intention_to_continue": "yes",
    "sentiment": "positive",
    "bond": "strong",
    "task": "aligned",
    "goal": "aligned"
  }
  ```

This systematic approach allows for a comprehensive analysis of each post, providing insights into the perceived usefulness, trustworthiness, and emotional bond users have with AI tools.

<br/>
# 요약

이 연구는 5,126개의 Reddit 게시물을 분석하여 AI 기반 정신 건강 지원에 대한 사용자 경험을 조사하였다. 결과적으로, 사용자들은 AI의 유용성과 신뢰성, 그리고 결과의 질에 따라 긍정적인 경험을 보고했으며, 감정적 유대감보다는 작업 및 목표 정렬이 더 중요한 요소로 나타났다. 또한, 사용자들은 AI 사용에 따른 의존성 및 증상 악화와 같은 위험을 자주 언급하였다.

---

This study analyzed 5,126 Reddit posts to investigate user experiences with AI-based mental health support. The results showed that users reported positive experiences based on the usefulness, trustworthiness, and quality of outcomes of the AI, with task and goal alignment being more significant than emotional bonding. Additionally, users frequently mentioned risks such as dependence and symptom escalation associated with AI use.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **AI 사용 의도 분포 (Figure A.3)**: 감정적 지원(18.0%)이 가장 흔한 사용 의도로 나타났으며, 기능적 지원(12.6%)과 심리 교육(11.7%)이 뒤를 이었다. 이는 사용자가 AI를 주로 감정적 검증과 실용적 대처 지원을 위해 활용하고 있음을 보여준다.
   - **위험 및 우려의 동시 발생 패턴 (Figure A.5)**: 사용 의도와 위험 카테고리 간의 상관관계를 보여주며, 동반자 사용은 중독 및 의존과 가장 강하게 연관되어 있음을 나타낸다. 이는 AI 사용이 특정한 위험과 밀접하게 연결되어 있음을 시사한다.

2. **테이블**:
   - **사용 의도 카테고리 (Table A.5)**: AI 사용의 다양한 목적을 정의하고, 각 카테고리에 대한 대표적인 예시를 제공한다. 이는 AI가 감정적 지원, 위안, 심리 교육 등 다양한 역할을 수행할 수 있음을 보여준다.
   - **위험/우려 카테고리 (Table A.6)**: AI 사용과 관련된 다양한 위험을 분류하고, 각 카테고리에 대한 정의와 예시를 제공한다. 중독, 증상 악화, 잘못된 정보 등 다양한 위험이 사용자의 경험에 영향을 미친다.

3. **어펜딕스**:
   - **서브레딧 데이터 통계 (Table A.4)**: 47개의 서브레딧을 DSM-5 카테고리별로 정리하여 각 서브레딧의 구독자 수와 AI 관련 게시물 수를 보여준다. 이는 다양한 정신 건강 문제에 대한 AI 사용의 범위를 이해하는 데 도움이 된다.



1. **Diagrams and Figures**:
   - **Distribution of AI Usage Intent (Figure A.3)**: Emotional support (18.0%) was the most common usage intent, followed by functional support (12.6%) and psychoeducation (11.7%). This indicates that users primarily utilize AI for emotional validation and practical coping support.
   - **Risk and Concern Co-occurrence Patterns (Figure A.5)**: This figure illustrates the correlation between usage intent and risk categories, showing that companionship use is most strongly associated with addiction and dependence. This suggests that AI usage is closely linked to specific risks.

2. **Tables**:
   - **Usage Intent Categories (Table A.5)**: This table defines various purposes for AI use and provides representative examples for each category. It shows that AI can serve multiple roles, including emotional support, reassurance, and psychoeducation.
   - **Risks/Concerns Categories (Table A.6)**: This table categorizes various risks associated with AI use, providing definitions and examples for each category. Concerns such as addiction, symptom escalation, and misinformation significantly impact user experiences.

3. **Appendix**:
   - **Subreddit Dataset Statistics (Table A.4)**: This table organizes 47 selected subreddits by DSM-5 categories, showing subscriber counts and the number of AI-related posts. This helps in understanding the scope of AI use across various mental health issues.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Aghakhani2026,
  author    = {Elham Aghakhani and Rezvaneh Rezapour},
  title     = {Like a Therapist, But Not: Reddit Narratives of AI in Mental Health Contexts},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {11716--11736},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},


}
```

### Chicago Style Citation

Aghakhani, Elham, and Rezvaneh Rezapour. 2026. "Like a Therapist, But Not: Reddit Narratives of AI in Mental Health Contexts." In *Findings of the Association for Computational Linguistics: ACL 2026*, 11716–11736. Vienna, Austria: Association for Computational Linguistics. 
    