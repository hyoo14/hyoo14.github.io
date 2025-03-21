---
layout: post
title:  "[2024]Large Language Models Reflect the Ideology of their Creators"  
date:   2024-11-19 11:33:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

LLM은 배경지식에 의존적.. 당연히 나라마다 언어마다 다른 결과.. 편향된 결과 보이겠죠   



짧은 요약(Abstract) :    




이 논문은 대형 언어 모델(LLMs)이 창작자의 이념적 관점을 반영할 가능성을 연구합니다. 다양한 LLM들이 영어와 중국어로 생성하는 주요 인물들에 대한 설명을 비교하여, 언어 및 모델의 출처에 따라 도덕적 평가와 이념적 차이가 나타나는지를 분석했습니다. 연구 결과, 동일한 모델이라도 언어에 따라 상당한 차이를 보였으며, 서구와 비서구 모델 간에도 가치관 차이가 존재했습니다. 이는 모델의 설계, 데이터 선택 및 정렬 과정에서 나타나는 인간의 선택이 이념적 편향을 초래할 수 있음을 시사합니다. 이러한 결과는 LLM의 중립성을 보장하려는 기술적, 규제적 노력에 도전 과제를 제기하며, 정치적 도구화의 위험성도 지적합니다.

---



This paper investigates the potential for large language models (LLMs) to reflect the ideological perspectives of their creators. By comparing the descriptions of prominent figures generated by diverse LLMs in English and Chinese, the study examines moral evaluations and ideological differences influenced by language and the origin of the models. Findings reveal significant variations in moral stances depending on the language used and between Western and non-Western models. These results suggest that human choices during model design, data selection, and alignment processes can embed ideological biases. This raises challenges for efforts to ensure LLM neutrality and highlights the risks of political instrumentalization.



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




이 논문에서는 대형 언어 모델(LLM)의 이념적 입장을 정량화하기 위해 두 단계로 구성된 실험 방법을 사용했습니다. 주요 메서드는 다음과 같습니다:

1. **1단계: 개방형 설명 생성**
   모델에 특정 정치적 인물에 대해 설명을 요청하는 자연스러운 질문을 제시하여, 사용자가 일반적으로 모델을 사용할 때 얻을 수 있는 응답을 수집했습니다. 이 단계에서는 모델이 주어진 인물에 대해 도덕적 평가를 명시적으로 요구하지 않도록 설계되었습니다.

2. **2단계: 도덕적 평가 추출**
   1단계에서 생성된 응답을 다시 모델에 제시하고, 해당 응답에서 드러나는 도덕적 평가를 "매우 부정적", "부정적", "중립적", "긍정적", "매우 긍정적" 중 하나로 분류하도록 요청했습니다. 이 과정에서 모델의 응답을 강제적으로 하나의 범주로 제한하는 구성을 사용했습니다.

### 기존 메서드와의 비교
이 연구는 기존의 LLM 이념 분석 메서드인 직접적인 질문(예: 정치적 성향 설문조사) 방식과는 차별화됩니다. 기존 연구들은 모델의 응답이 질문의 구성에 민감하며 일관성이 낮다는 점을 지적해왔습니다. 이 논문에서는 개방형 질문 방식을 채택함으로써 모델의 자연스러운 사용 환경을 반영하려고 했고, 평가를 위한 Likert 척도를 적용해 정량화를 시도했습니다. 또한, 이전 연구들이 주로 좌-우 이념 축에 초점을 맞췄다면, 이 연구는 다양한 이념적 태그(국제주의, 다문화주의, 사회 정의 등)를 도입하여 보다 복잡한 이념적 다양성을 분석하려고 했습니다.

---



The paper uses a two-stage experimental methodology to quantify the ideological stances of large language models (LLMs). The primary methods include:

1. **Stage 1: Open-ended Description Generation**
   The models were prompted with natural questions about specific political figures to collect responses typical of everyday user interactions. This stage avoided explicitly asking the model for moral evaluations to maintain ecological validity.

2. **Stage 2: Moral Assessment Extraction**
   The responses from Stage 1 were reintroduced to the model, which was asked to classify the moral assessment embedded in the description into one of five categories: "very negative," "negative," "neutral," "positive," or "very positive." The prompts in this stage were carefully designed to enforce single-label responses.

### Comparison with Existing Methods
This study differs from prior approaches that primarily used direct questioning (e.g., political orientation surveys). Such methods have been criticized for their susceptibility to prompt phrasing and lack of response consistency. By adopting an open-ended prompting approach, this paper aims to reflect natural model usage and leverages a Likert scale to quantify moral stances. Additionally, while earlier studies focused on ideological dimensions like the left-right political spectrum, this research expands the analysis by incorporating diverse ideological tags (e.g., internationalism, multiculturalism, social justice), enabling a more nuanced understanding of ideological diversity.



   
 
<br/>
# Results  




#### 결과
이 연구의 주요 결과는 다음과 같습니다:
1. **이념적 차이 발견**:
   - 동일한 모델이라도 영어와 중국어와 같은 언어에 따라 도덕적 평가와 이념적 태도가 달라지는 것으로 나타났습니다.
   - 서구와 비서구에서 개발된 모델 간에도 이념적 차이가 존재했습니다. 예를 들어, 서구 모델은 개인 자유와 평등 같은 가치에 더 높은 점수를 부여하는 반면, 비서구 모델은 국가 통제 및 중앙 집권적 경제에 대해 더 긍정적이었습니다.

2. **정량적 분석**:
   - 다양한 LLM을 평가하고 이념적 태그를 기준으로 평균 점수 차이를 비교했습니다.
   - 특정 태그(예: "중국 긍정" 또는 "다문화주의")에서 언어별로 점수 차이가 통계적으로 유의미한 것으로 나타났습니다.

#### 비교 모델
이 연구는 OpenAI의 GPT-4, Meta의 LLaMA-2, Google의 Gemini-Pro, Baidu의 ERNIE-Bot 등 17개의 대형 언어 모델을 비교했습니다. 모델은 영어와 중국어로 평가되었으며, 각 모델-언어 쌍을 별도의 응답자로 간주하여 분석을 수행했습니다.

#### 사용된 데이터셋
- 연구는 **Pantheon 데이터셋**을 기반으로 하여 정치적 인물 4,339명을 선정했습니다.
- 정치적 인물은 위키피디아의 영어 및 중국어 요약본을 바탕으로 태그가 지정되었습니다.

#### 성능 향상
- 이 연구는 기존 연구보다 더 높은 생태학적 타당성을 제공했습니다. 기존 연구들이 주로 직접적인 질문을 통한 평가에 의존했다면, 이 연구는 보다 자연스러운 사용 환경을 반영했습니다.
- 또한, Likert 척도를 기반으로 평가를 정량화함으로써 응답 간 비교 가능성을 향상시켰습니다.

---



#### Results
The main findings of this study are as follows:
1. **Discovery of Ideological Differences**:
   - The same model exhibited differing moral assessments and ideological attitudes depending on the language (e.g., English vs. Chinese).
   - There were also significant ideological differences between Western and non-Western models. For example, Western models rated values like individual liberty and equality more highly, while non-Western models were more positive about state control and centralized economies.

2. **Quantitative Analysis**:
   - The study evaluated various LLMs and compared their average scores across ideological tags.
   - Statistically significant differences were observed for specific tags (e.g., "Pro-China" or "Multiculturalism") based on language.

#### Comparison Models
The study compared 17 large language models, including OpenAI’s GPT-4, Meta’s LLaMA-2, Google’s Gemini-Pro, and Baidu’s ERNIE-Bot. Each model-language pair was treated as a separate respondent for analysis.

#### Dataset
- The study used the **Pantheon dataset**, selecting 4,339 political figures.
- Political figures were tagged based on their English and Chinese Wikipedia summaries.

#### Performance Improvements
- This approach demonstrated higher ecological validity compared to previous studies, which relied primarily on direct questioning.
- By leveraging a Likert scale for evaluation, the study improved the comparability of responses across models.



<br/>
# 예제  




#### 구체적인 예시

**데이터셋 처리 및 분석 과정**
- **데이터셋 예시**: 연구는 Pantheon 데이터셋에서 4,339명의 정치적 인물을 선택했습니다. 예를 들어, **에드워드 스노든**은 이 데이터셋에 포함된 인물 중 하나입니다. 
- **데이터 처리**:
  1. 위키피디아의 영어 및 중국어 요약본에서 스노든에 대한 설명을 추출합니다.
     - 영어: "Edward Snowden is a former U.S. intelligence contractor who leaked classified information..."
     - 중국어: "爱德华·斯诺登是一名前美国情报承包商，他泄露了机密信息..."
  2. 모델은 각 언어로 스노든에 대해 설명하도록 요청받고, 도덕적 평가를 "매우 부정적"에서 "매우 긍정적"의 Likert 척도로 분류했습니다.

**예측 및 결과**
- **처리 방법**: 모델이 생성한 스노든 설명은 다음과 같은 태그로 분류되었습니다:
  - 영어 응답: "자유와 인권(긍정적)", "미국에 대한 부정적 견해"
  - 중국어 응답: "러시아 긍정", "국가 생활 방식(부정적)"
- **결과**:
  - 영어로 생성된 응답은 스노든의 행동을 "긍정적"으로 평가하는 경향이 강했으며, 자유와 인권에 대해 강조했습니다.
  - 반면, 중국어로 생성된 응답은 스노든의 미국 비판을 부각하며 "중립적" 또는 "부정적" 평가를 보였습니다.
- 이 결과는 동일한 정치적 인물이라도 언어와 문화적 맥락에 따라 모델의 도덕적 평가가 달라질 수 있음을 보여줍니다.

---



#### Specific Example

**Dataset Processing and Analysis Steps**
- **Dataset Example**: The study selected 4,339 political figures from the Pantheon dataset. One such figure is **Edward Snowden**.
- **Data Processing**:
  1. Extract descriptions of Snowden from English and Chinese Wikipedia summaries.
     - English: "Edward Snowden is a former U.S. intelligence contractor who leaked classified information..."
     - Chinese: "爱德华·斯诺登是一名前美国情报承包商，他泄露了机密信息..."
  2. Models were prompted to describe Snowden in both languages and classify their moral assessment on a Likert scale from "very negative" to "very positive."

**Prediction and Results**
- **Methodology**: The generated descriptions were classified with the following tags:
  - English Response: "Freedom and Human Rights (Positive)", "Negative View of the United States"
  - Chinese Response: "Pro-Russia", "National Way of Life (Negative)"
- **Results**:
  - English-generated responses tended to evaluate Snowden’s actions as "positive," emphasizing freedom and human rights.
  - Chinese-generated responses highlighted Snowden’s critique of the U.S., resulting in more "neutral" or "negative" evaluations.
- This illustrates how the same political figure can receive different moral evaluations depending on language and cultural context.



<br/>  
# 요약   


이 연구는 Pantheon 데이터셋에서 4,339명의 정치적 인물을 선정하고, 영어와 중국어로 대형 언어 모델(LLMs)을 평가했습니다. 두 단계 실험을 통해 모델이 생성한 설명을 분석하고, 도덕적 평가를 Likert 척도로 정량화했습니다. 에드워드 스노든의 사례에서 영어 응답은 자유와 인권을 강조하며 긍정적인 평가를 보였지만, 중국어 응답은 미국 비판을 강조하며 중립적 평가를 보였습니다. 이는 동일한 모델이라도 언어와 문화적 맥락에 따라 도덕적 평가가 달라질 수 있음을 보여줍니다. 결과적으로, 연구는 LLM이 설계와 데이터 선택 과정에서 창작자의 이념적 관점을 반영할 가능성이 있음을 시사합니다.  

---


This study evaluated 4,339 political figures from the Pantheon dataset using large language models (LLMs) in English and Chinese. A two-stage experiment was conducted to analyze model-generated descriptions and quantify moral evaluations using a Likert scale. In the case of Edward Snowden, English responses emphasized freedom and human rights with positive assessments, whereas Chinese responses highlighted U.S. criticism with neutral assessments. This demonstrates that moral evaluations by the same model can vary depending on language and cultural context. The findings suggest that LLMs may reflect the ideological perspectives of their creators due to design and data selection choices.  
<br/>  
# 기타  


<br/>
# refer format:     



@article{buyl2024ideology,
  title={Large Language Models Reflect the Ideology of their Creators},
  author={Buyl, Maarten and Rogiers, Alexander and Noels, Sander and Dominguez-Catena, Iris and Heiter, Edith and Romero, Raphael and Johary, Iman and Mara, Alexandru-Cristian and Lijffijt, Jefrey and De Bie, Tijl},
  journal={arXiv preprint arXiv:2410.18417},
  year={2024},
  url={https://arxiv.org/abs/2410.18417}
}



Buyl, Maarten, Alexander Rogiers, Sander Noels, Iris Dominguez-Catena, Edith Heiter, Raphael Romero, Iman Johary, Alexandru-Cristian Mara, Jefrey Lijffijt, and Tijl De Bie. “Large Language Models Reflect the Ideology of Their Creators.” arXiv preprint arXiv:2410.18417, 2024.





