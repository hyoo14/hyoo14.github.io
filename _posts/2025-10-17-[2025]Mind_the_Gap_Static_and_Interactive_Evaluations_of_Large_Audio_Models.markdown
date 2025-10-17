---
layout: post
title:  "[2025]Mind the Gap: Static and Interactive Evaluations of Large Audio Models"
date:   2025-10-17 21:27:26 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 484명의 참가자로부터 7,500개의 대화 데이터를 수집하여 대형 오디오 모델(LAM)의 상호작용 성능을 평가했습니다.


짧은 요약(Abstract) :


이 연구는 대화형 AI 챗봇의 사용이 증가함에 따라 음성 상호작용이 의미적 및 사회적 신호를 전달하는 효과적인 방법이 될 수 있음을 강조합니다. 이를 위해 대규모 오디오 모델(Large Audio Models, LAMs)의 개발이 필요하며, 사용자 목표와의 정렬을 위해 사용자 요구와 선호를 명확히 이해해야 한다고 주장합니다. 연구진은 484명의 참가자로부터 7,500개의 LAM 상호작용을 수집하고, 사용자 쿼리의 주제 모델링을 통해 오디오 인터페이스의 주요 사용 사례를 식별했습니다. 또한, 사용자 선호 순위와 질적 피드백을 분석하여 어떤 모델이 사용자 요구에 가장 잘 부합하는지를 평가했습니다. 마지막으로, 정적 벤치마크가 대화형 성능을 예측하는 방법을 평가했으며, 분석 결과 어떤 개별 벤치마크도 대화형 결과와 강한 상관관계를 보이지 않았습니다. 이는 사용자 선호와 더 잘 일치하는 LAM 평가 방법의 개발 필요성을 시사합니다.



This study emphasizes that as the use of conversational AI chatbots increases, voice interaction can serve as an effective means of conveying semantic and social signals. It argues for the necessity of developing Large Audio Models (LAMs) to align with user goals, which requires a clear understanding of user needs and preferences. The researchers collected 7,500 LAM interactions from 484 participants and identified primary use cases for audio interfaces through topic modeling of user queries. They also analyzed user preference rankings and qualitative feedback to determine which models best align with user needs. Finally, they evaluated how static benchmarks predict interactive performance, revealing that no individual benchmark strongly correlates with interactive results. This suggests a clear need for developing LAM evaluations that better correlate with user preferences.


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



이 연구에서는 대규모 오디오 모델(Large Audio Models, LAMs)의 성능을 평가하기 위해 두 가지 주요 방법론을 사용했습니다: 정적 평가와 상호작용 평가입니다. 정적 평가는 기존의 벤치마크 데이터셋을 기반으로 하여 모델의 특정 능력을 측정하는 반면, 상호작용 평가는 실제 사용자와의 상호작용을 통해 모델의 성능을 평가합니다.

1. **모델 아키텍처**: 연구에서 평가한 LAMs는 Whisper와 Llama와 같은 다양한 아키텍처를 포함합니다. Whisper는 자동 음성 인식(ASR) 시스템으로, 음성을 텍스트로 변환하는 데 사용됩니다. Llama는 텍스트 기반의 대규모 언어 모델로, 음성 입력을 처리하는 데 필요한 언어적 이해를 제공합니다. 이 두 모델을 결합하여 ASR 파이프라인을 구성하였으며, 이는 음성 입력을 텍스트로 변환한 후 Llama를 통해 처리하는 방식입니다.

2. **트레이닝 데이터**: LAMs는 대규모의 비지도 학습 데이터셋을 사용하여 훈련되었습니다. 이 데이터셋은 다양한 음성 샘플을 포함하고 있으며, 모델이 일반화된 오디오 표현을 학습할 수 있도록 돕습니다. Whisper는 공개 도메인 오디오 북을 기반으로 한 Librispeech와 같은 데이터셋을 사용하여 훈련되었습니다.

3. **상호작용 평가 방법**: 연구팀은 484명의 참가자로부터 7,500개의 상호작용 데이터를 수집했습니다. 참가자들은 음성 기반 AI 어시스턴트가 수행할 것으로 기대되는 다양한 작업을 테스트하도록 유도되었습니다. 이 과정에서 사용자는 두 개의 모델로부터 응답을 받고, 선호하는 모델을 선택하는 방식으로 평가가 이루어졌습니다. 이러한 상호작용을 통해 수집된 데이터는 모델의 사용자 선호도를 분석하는 데 사용되었습니다.

4. **정적 벤치마크**: 연구에서는 20개의 정적 벤치마크 데이터셋을 사용하여 LAMs의 성능을 평가했습니다. 이 데이터셋은 음성 인식, 감정 인식, 의도 탐지 등 다양한 작업을 포함하고 있으며, 각 모델의 성능을 비교하는 데 사용되었습니다. 그러나 연구 결과에 따르면, 정적 벤치마크는 사용자 선호도와의 상관관계가 낮아, 새로운 평가 방법론이 필요하다는 결론에 도달했습니다.

이러한 방법론을 통해 연구팀은 LAMs의 성능을 보다 정확하게 평가하고, 사용자 요구에 맞는 모델 개발 방향을 제시할 수 있었습니다.

---




In this study, two main methodologies were employed to evaluate the performance of Large Audio Models (LAMs): static evaluation and interactive evaluation. Static evaluation measures specific capabilities of the models based on existing benchmark datasets, while interactive evaluation assesses model performance through real user interactions.

1. **Model Architecture**: The LAMs evaluated in this research include various architectures such as Whisper and Llama. Whisper is an Automatic Speech Recognition (ASR) system used to convert speech into text. Llama is a text-based large language model that provides the necessary linguistic understanding to process audio inputs. These two models are combined to form an ASR pipeline, where speech input is first converted to text by Whisper and then processed by Llama.

2. **Training Data**: LAMs were trained using large-scale unsupervised datasets. These datasets include a variety of speech samples, enabling the models to learn generalized audio representations. Whisper was trained on datasets like Librispeech, which is based on public domain audiobooks.

3. **Interactive Evaluation Method**: The research team collected 7,500 interaction data points from 484 participants. Participants were prompted to test various tasks they would expect from a voice-based AI assistant. In this process, users received responses from two models and selected their preferred model. The data collected from these interactions were used to analyze user preferences for the models.

4. **Static Benchmarks**: The study utilized 20 static benchmark datasets to evaluate the performance of LAMs. These datasets cover a range of tasks, including speech recognition, emotion detection, and intent detection, and were used to compare the performance of each model. However, the results indicated that static benchmarks had low correlation with user preferences, leading to the conclusion that new evaluation methodologies are needed.

Through these methodologies, the research team was able to more accurately assess the performance of LAMs and provide directions for model development that align with user needs.


<br/>
# Results



이 연구에서는 대규모 오디오 모델(Large Audio Models, LAMs)의 상호작용 성능을 평가하기 위해 484명의 참가자로부터 7,500개의 상호작용 데이터를 수집했습니다. 연구의 주요 목표는 사용자 선호도와 모델 성능 간의 관계를 이해하고, 기존의 정적 벤치마크가 이러한 상호작용 성능을 얼마나 잘 예측하는지를 평가하는 것이었습니다.

#### 경쟁 모델
연구에서 평가된 모델은 다음과 같습니다:
1. **Whisper (ASR 파이프라인)**: Whisper는 음성을 텍스트로 변환하는 자동 음성 인식(ASR) 시스템입니다.
2. **Llama3-8B-Instruct**: 텍스트 기반의 대규모 언어 모델로, Whisper와 결합하여 사용됩니다.
3. **DiV A**: 텍스트 LLM을 증류하여 훈련된 모델입니다.
4. **GPT-4o**: 상업적으로 사용되는 모델로, 정적 벤치마크에서 가장 높은 성능을 보였습니다.
5. **Gemini 1.5**: 또 다른 상업 모델로, 다양한 작업에서 우수한 성능을 보였습니다.
6. **Qwen2-Audio**: 공개 소스 모델로, 여러 작업에서 좋은 성능을 보였습니다.

#### 테스트 데이터
연구에서는 20개의 서로 다른 정적 벤치마크를 사용하여 모델의 성능을 평가했습니다. 이 벤치마크는 음성 인식, 감정 인식, 의도 탐지, 나이 및 성별 분류 등 다양한 작업을 포함합니다. 각 모델의 성능은 매크로 F1 점수, 단어 오류율(Word Error Rate, WER) 등으로 측정되었습니다.

#### 메트릭
모델의 성능을 평가하기 위해 사용된 주요 메트릭은 다음과 같습니다:
- **매크로 F1 점수**: 분류 작업에서 각 클래스의 중요성을 고려하여 평균을 계산합니다.
- **단어 오류율 (WER)**: ASR 작업에서 음성 인식의 정확성을 측정합니다.
- **브래들리-테리 점수**: 사용자 선호도를 기반으로 모델의 상대적 성능을 평가하는 데 사용됩니다.

#### 비교 결과
연구 결과, 정적 벤치마크와 상호작용 성능 간의 상관관계는 낮았습니다. 특히, 단일 벤치마크는 사용자 선호도와 강한 상관관계를 보이지 않았으며, 20개의 벤치마크 중에서 두 개의 벤치마크(공식 SG-Speech 및 CommonVoice - Age)만이 사용자 선호도와 긍정적인 상관관계를 보였습니다. 이 결과는 정적 벤치마크가 LAM의 상호작용 성능을 예측하는 데 한계가 있음을 시사합니다.

결론적으로, 연구는 LAM의 성능을 평가하기 위해 사용자 선호도와 상호작용을 반영하는 새로운 정적 평가 방법의 필요성을 강조합니다. 사용자 피드백을 통해 모델의 상호작용 능력을 향상시키기 위한 방향성을 제시하였습니다.

---




This study collected 7,500 interaction data from 484 participants to evaluate the interactive performance of Large Audio Models (LAMs). The main goal of the research was to understand the relationship between user preferences and model performance, and to assess how well existing static benchmarks predict this interactive performance.

#### Competing Models
The models evaluated in the study include:
1. **Whisper (ASR Pipeline)**: An automatic speech recognition (ASR) system that converts speech to text.
2. **Llama3-8B-Instruct**: A text-based large language model used in conjunction with Whisper.
3. **DiV A**: A model trained by distilling a text LLM.
4. **GPT-4o**: A commercially used model that performed the best on static benchmarks.
5. **Gemini 1.5**: Another commercial model that showed strong performance across various tasks.
6. **Qwen2-Audio**: An open-source model that performed well on multiple tasks.

#### Test Data
The study utilized 20 different static benchmarks to evaluate model performance. These benchmarks included various tasks such as speech recognition, emotion recognition, intent detection, age and gender classification. The performance of each model was measured using metrics like macro F1 score and Word Error Rate (WER).

#### Metrics
The key metrics used to evaluate model performance included:
- **Macro F1 Score**: An average calculated considering the importance of each class in classification tasks.
- **Word Error Rate (WER)**: A measure of accuracy in ASR tasks.
- **Bradley-Terry Score**: Used to evaluate the relative performance of models based on user preferences.

#### Comparison Results
The results indicated a low correlation between static benchmarks and interactive performance. Notably, no single benchmark strongly correlated with user preferences, and only two benchmarks (Public-SG-Speech and CommonVoice - Age) showed positive correlations with user preferences. This suggests that static benchmarks have limitations in predicting the interactive performance of LAMs.

In conclusion, the study emphasizes the need for new static evaluation methods that reflect user preferences and interactions to assess LAM performance. User feedback was used to provide directions for improving the interactive capabilities of the models.


<br/>
# 예제



이 논문에서는 대규모 오디오 모델(Large Audio Models, LAMs)의 상호작용 평가를 통해 사용자 선호도를 이해하고, 기존의 정적 벤치마크가 이러한 상호작용 성능을 얼마나 잘 예측하는지를 분석합니다. 연구의 주요 목표는 LAM의 개발이 사용자 요구와 선호에 맞춰 이루어질 수 있도록 하는 것입니다.

#### 1. 데이터 수집
연구팀은 484명의 참가자로부터 7,500개의 LAM 상호작용 데이터를 수집했습니다. 참가자들은 음성 기반 AI 어시스턴트가 수행할 것으로 기대하는 다양한 작업을 테스트하기 위해 음성 입력을 제공했습니다. 이 과정에서 사용자는 두 개의 모델로부터 응답을 받고, 더 나은 응답을 선택하는 방식으로 선호도를 평가했습니다.

#### 2. 예시 입력 및 출력
- **입력 예시**: 사용자가 "우주란 무엇인가요?"라는 질문을 음성으로 입력합니다.
- **모델 A의 출력**: "우주는 모든 물질과 에너지가 존재하는 공간입니다."
- **모델 B의 출력**: "우주는 별, 행성, 그리고 모든 물체가 포함된 공간입니다."

사용자는 두 모델의 응답을 비교하고, 더 나은 응답을 선택합니다. 예를 들어, 사용자가 모델 A의 응답이 더 구체적이고 유용하다고 판단하면, 모델 A를 선택합니다.

#### 3. 평가 기준
연구팀은 사용자의 선호도를 기반으로 모델의 성능을 평가하기 위해 브래들리-테리 모델(Bradley-Terry model)을 사용했습니다. 이 모델은 각 모델의 상대적인 우수성을 추정하는 데 사용됩니다. 사용자의 피드백을 통해 어떤 모델이 특정 작업에서 더 나은 성능을 보이는지를 분석합니다.

#### 4. 정적 벤치마크와의 비교
연구팀은 20개의 정적 벤치마크를 사용하여 LAM의 성능을 평가했습니다. 그러나 이들 벤치마크는 사용자 선호도와 강한 상관관계를 보이지 않았습니다. 예를 들어, 특정 벤치마크에서 높은 점수를 받은 모델이 실제 사용자 상호작용에서는 낮은 선호도를 받을 수 있음을 발견했습니다. 이는 정적 벤치마크가 LAM의 상호작용 성능을 충분히 반영하지 못한다는 것을 시사합니다.




This paper investigates the interactive evaluation of Large Audio Models (LAMs) to understand user preferences and analyzes how well existing static benchmarks predict such interactive performance. The primary goal of the research is to align LAM development with user needs and preferences.

#### 1. Data Collection
The research team collected 7,500 LAM interaction data from 484 participants. Participants provided voice inputs to test various tasks they would expect from a voice-based AI assistant. In this process, users received responses from two models and evaluated their preferences by choosing the better response.

#### 2. Example Input and Output
- **Input Example**: A user asks, "What is the universe?" via voice input.
- **Output from Model A**: "The universe is the space that contains all matter and energy."
- **Output from Model B**: "The universe includes stars, planets, and all objects within space."

The user compares the responses from the two models and selects the better one. For instance, if the user finds Model A's response to be more specific and useful, they will choose Model A.

#### 3. Evaluation Criteria
To evaluate model performance based on user preferences, the research team employed the Bradley-Terry model. This model is used to estimate the relative superiority of each model. User feedback is analyzed to determine which models perform better on specific tasks.

#### 4. Comparison with Static Benchmarks
The research team evaluated LAM performance using 20 static benchmarks. However, these benchmarks did not show strong correlations with user preferences. For example, a model that scored high on a specific benchmark might receive low preference in actual user interactions. This suggests that static benchmarks do not adequately reflect the interactive performance of LAMs.

<br/>
# 요약

이 연구에서는 484명의 참가자로부터 7,500개의 대화 데이터를 수집하여 대형 오디오 모델(LAM)의 상호작용 성능을 평가했습니다. 분석 결과, 기존의 정적 벤치마크는 사용자 선호도와 강한 상관관계를 보이지 않았으며, 사용자 피드백을 통해 LAM의 개선 방향을 제시했습니다. 특히, 사용자는 모델의 유용성, 세부 정보 수준, 언어 적합성 등을 중요하게 여겼습니다.

---

In this study, 7,500 interaction data points were collected from 484 participants to evaluate the interactive performance of Large Audio Models (LAMs). The analysis revealed that existing static benchmarks did not strongly correlate with user preferences, and user feedback provided insights for improving LAMs. Notably, users valued factors such as model helpfulness, level of detail, and language appropriateness.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 정적 평가와 상호작용 평가의 비교를 보여줍니다. 이 다이어그램은 LAM(대형 오디오 모델)의 평가 방법의 차이를 강조하며, 사용자 선호도와 상호작용을 기반으로 한 평가의 필요성을 강조합니다.
   - **Figure 2**: 사용자 쿼리의 주요 주제를 식별한 결과를 보여줍니다. 주제는 작업 실행, 지식 확장, 대화, 조언 요청으로 나뉘며, 사용자의 77%가 효율성을 목적으로 음성을 사용하고 있음을 나타냅니다.
   - **Figure 3**: 모델 간의 선호도 비교를 보여줍니다. 사용자 선호도에 따라 ASR 파이프라인과 Llama 모델이 가장 높은 선호도를 보였으며, 이는 사용자가 텍스트 의미에 더 의존하고 있음을 시사합니다.
   - **Figure 4**: 정적 벤치마크의 성능 차이가 사용자 선호도에 미치는 영향을 보여주는 혼합 효과 회귀 분석 결과입니다. 이 분석은 정적 벤치마크가 사용자 선호도를 예측하는 데 제한적인 예측력을 가지고 있음을 나타냅니다.

2. **테이블**
   - **Table 1**: 20개의 정적 벤치마크에서 LAM의 평균 성능을 보여줍니다. 이 표는 상업적 LAM이 일반적으로 더 높은 성능을 보이는 반면, 오픈 소스 모델도 경쟁력 있는 성능을 보임을 나타냅니다.
   - **Table 2**: 사용자 선호도와 정적 벤치마크 간의 상관관계를 보여줍니다. 이 표는 특정 벤치마크가 사용자 선호도와 긍정적인 상관관계를 보이는 반면, 대부분의 벤치마크는 낮은 상관관계를 보임을 나타냅니다.

3. **어펜딕스**
   - **Appendix A**: 다양한 작업에 대한 프롬프트 예시를 제공합니다. 이 프롬프트들은 감정 인식, 의도 탐지, 나이 및 성별 분류 등 다양한 작업을 위한 것입니다. 이러한 프롬프트는 LAM의 성능을 평가하는 데 사용됩니다.
   - **Appendix B**: 정적 벤치마크 간의 상관관계를 보여주는 분석 결과를 포함합니다. 이 분석은 정적 벤치마크가 LAM의 상호작용 능력을 반영하는 데 한계가 있음을 강조합니다.




1. **Diagrams and Figures**
   - **Figure 1**: This figure compares static and interactive evaluation methods, highlighting the differences in evaluating Large Audio Models (LAMs) and emphasizing the need for user preference-based evaluations.
   - **Figure 2**: It shows the main topics identified from user queries, categorized into task execution, knowledge expansion, chat, and advice seeking. Notably, 77% of users utilize speech for efficiency purposes.
   - **Figure 3**: This figure illustrates the preference comparison among models. The ASR pipeline and Llama model received the highest preference, indicating that users rely more on text semantics.
   - **Figure 4**: It presents the results of a mixed-effects regression analysis showing the impact of benchmark performance differences on user preferences. The analysis indicates that static benchmarks have limited predictive power for user preferences.

2. **Tables**
   - **Table 1**: This table displays the average performance of LAMs across 20 static benchmarks, indicating that commercial LAMs generally outperform open-source models, although the latter also show competitive performance.
   - **Table 2**: It shows the correlation between user preferences and static benchmarks, revealing that while some benchmarks exhibit positive correlations with user preferences, most show low correlation.

3. **Appendix**
   - **Appendix A**: Provides examples of prompts for various tasks, including emotion recognition, intent detection, and age and gender classification. These prompts are used to evaluate the performance of LAMs.
   - **Appendix B**: Contains analysis results showing the correlation among static benchmarks, emphasizing the limitations of static benchmarks in reflecting the interactive capabilities of LAMs.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Li2025,
  author    = {Minzhi Li and William Held and Michael J. Ryan and Kunat Pipatanakul and Potsawee Manakul and Hao Zhu and Diyi Yang},
  title     = {Mind the Gap: Static and Interactive Evaluations of Large Audio Models},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {8749--8766},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  address   = {Online},
}
```

### 시카고 스타일

Minzhi Li, William Held, Michael J. Ryan, Kunat Pipatanakul, Potsawee Manakul, Hao Zhu, and Diyi Yang. "Mind the Gap: Static and Interactive Evaluations of Large Audio Models." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 8749–8766. July 2025. Association for Computational Linguistics.
