---
layout: post
title:  "[2024]AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension"
date:   2025-10-17 21:25:22 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 AIR-Bench라는 새로운 벤치마크를 소개하여 대규모 오디오-언어 모델(LALMs)의 이해 및 상호작용 능력을 평가합니다.


짧은 요약(Abstract) :


이 논문의 초록에서는 최근 인간과 오디오 간의 상호작용을 위한 지침을 따르는 오디오-언어 모델(large audio-language models, LALMs)에 대한 관심이 높아지고 있음을 언급하고 있습니다. 그러나 오디오 중심의 상호작용 능력을 평가할 수 있는 벤치마크의 부재가 이 분야의 발전을 저해하고 있다는 점을 지적합니다. 기존 모델들은 주로 자동 음성 인식과 같은 기본 작업의 평가에 집중하고 있으며, 오디오를 중심으로 한 개방형 생성 능력에 대한 평가는 부족합니다. 따라서 LALMs 분야의 발전을 추적하고 향후 개선 방향을 제시하는 데 어려움이 있습니다. 이 논문에서는 LALMs의 다양한 오디오 신호 이해 능력과 인간과의 텍스트 형식 상호작용 능력을 평가하기 위해 AIR-Bench(오디오 지침 벤치마크)를 소개합니다. AIR-Bench는 기본 벤치마크와 채팅 벤치마크의 두 가지 차원을 포함하며, 각각 19개의 작업과 약 19,000개의 단일 선택 질문, 2,000개의 개방형 질문-답변 데이터를 포함하고 있습니다. 이 벤치마크는 모델이 직접 가설을 생성하도록 요구하며, GPT-4와 같은 고급 언어 모델을 활용하여 생성된 가설의 점수를 평가하는 통합 프레임워크를 설계했습니다. 실험 결과는 GPT-4 기반 평가와 인간 평가 간의 높은 일관성을 보여줍니다. AIR-Bench는 기존 LALMs의 한계를 드러내고 향후 연구 방향에 대한 통찰을 제공할 수 있습니다.



The abstract of this paper discusses the growing interest in instruction-following audio-language models (LALMs) for human-audio interaction. However, it points out that the lack of benchmarks capable of evaluating audio-centric interaction capabilities has hindered advancements in this field. Existing models primarily focus on assessing fundamental tasks such as automatic speech recognition, and there is a lack of assessment of open-ended generative capabilities centered around audio. This makes it challenging to track progress in the LALMs domain and to provide guidance for future improvements. The paper introduces AIR-Bench (Audio Instruction Benchmark), the first benchmark designed to evaluate LALMs' ability to understand various types of audio signals and to interact with humans in textual format. AIR-Bench encompasses two dimensions: foundation and chat benchmarks, consisting of 19 tasks with approximately 19,000 single-choice questions and 2,000 instances of open-ended question-and-answer data. Both benchmarks require the model to generate hypotheses directly. A unified framework is designed that leverages advanced language models, such as GPT-4, to evaluate the scores of generated hypotheses. Experimental results demonstrate a high level of consistency between GPT-4-based evaluation and human evaluation. By revealing the limitations of existing LALMs through evaluation results, AIR-Bench can provide insights into the direction of future research.


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



이 논문에서 제안하는 AIR-Bench는 대규모 오디오-언어 모델(Large Audio-Language Models, LALMs)의 성능을 평가하기 위한 최초의 생성적 평가 벤치마크입니다. AIR-Bench는 두 가지 주요 차원으로 구성되어 있습니다: 기초 벤치마크와 채팅 벤치마크입니다.

1. **모델 아키텍처**: AIR-Bench는 다양한 오디오 신호(인간의 음성, 자연 소리, 음악 등)를 이해하고, 인간의 지시에 따라 상호작용할 수 있는 LALMs의 능력을 평가하기 위해 설계되었습니다. 이 벤치마크는 19개의 오디오 작업으로 구성된 기초 벤치마크와 2,000개 이상의 오픈 엔디드 질문으로 구성된 채팅 벤치마크로 나뉩니다.

2. **트레이닝 데이터**: 기초 벤치마크는 약 19,000개의 단일 선택 질문을 포함하고 있으며, 각 질문은 특정 기초 능력에 초점을 맞추고 있습니다. 채팅 벤치마크는 복잡한 오디오에 대한 모델의 이해도를 직접 평가하는 오픈 엔디드 질문-답변 데이터로 구성되어 있습니다. 데이터는 GPT-4를 사용하여 자동 필터링되고 수동 검증을 거쳐 품질이 유지됩니다.

3. **특별한 기법**: AIR-Bench는 오디오의 복잡성을 높이기 위해 새로운 오디오 믹싱 전략을 제안합니다. 이 전략은 음량 조절과 시간적 위치 이동을 포함하여, 실제 상황에서의 오디오와 유사한 복잡한 오디오 신호를 생성합니다. 또한, 평가 프레임워크는 GPT-4를 활용하여 생성된 가설의 품질을 평가합니다. 이 과정에서 모델은 오디오와 질문을 기반으로 직접 답변을 생성해야 하며, GPT-4는 메타 정보를 바탕으로 참조 답변을 생성하여 평가를 수행합니다.

4. **모델 평가**: AIR-Bench는 9개의 LALMs를 평가하여, 기존 모델들이 오디오 이해 및 지시 따르기 능력에서 제한적임을 보여주었습니다. 이 평가 결과는 향후 연구 방향에 대한 통찰을 제공합니다.




The paper introduces AIR-Bench, the first generative evaluation benchmark designed specifically for Large Audio-Language Models (LALMs). AIR-Bench is structured around two main dimensions: the foundation benchmark and the chat benchmark.

1. **Model Architecture**: AIR-Bench is designed to evaluate the ability of LALMs to understand various types of audio signals (including human speech, natural sounds, and music) and to interact with humans based on instructions. The benchmark is divided into a foundation benchmark consisting of 19 audio tasks and a chat benchmark containing over 2,000 open-ended questions.

2. **Training Data**: The foundation benchmark includes approximately 19,000 single-choice questions, each focusing on a specific foundational ability. The chat benchmark consists of open-ended question-and-answer data that directly assesses the model's comprehension of complex audio. The data quality is maintained through automated filtering by GPT-4, followed by manual verification.

3. **Special Techniques**: AIR-Bench proposes a novel audio mixing strategy to enhance the complexity of audio. This strategy includes loudness control and temporal dislocation, creating complex audio signals that resemble real-world scenarios. Additionally, the evaluation framework leverages GPT-4 to assess the quality of generated hypotheses. In this process, models are required to generate answers directly based on audio and questions, while GPT-4 generates reference answers based on meta-information for evaluation.

4. **Model Evaluation**: AIR-Bench evaluates nine LALMs, revealing that existing models have limited audio understanding and instruction-following capabilities. The evaluation results provide insights into future research directions.


<br/>
# Results



이 논문에서는 AIR-Bench라는 새로운 벤치마크를 소개하며, 이는 대규모 오디오-언어 모델(LALMs)의 성능을 평가하기 위해 설계되었습니다. AIR-Bench는 두 가지 주요 벤치마크로 구성되어 있습니다: 기초 벤치마크와 채팅 벤치마크입니다. 기초 벤치마크는 19개의 오디오 작업으로 구성되어 있으며, 약 19,000개의 단일 선택 질문을 포함하고 있습니다. 채팅 벤치마크는 2,000개 이상의 오픈 엔디드 질문을 포함하여 모델이 복잡한 오디오를 이해하고 지시를 따르는 능력을 평가합니다.

#### 경쟁 모델
논문에서는 여러 LALMs 모델을 평가하였으며, 그 중에는 SALMONN, Qwen-Audio-Chat, Qwen-Audio Turbo, BLSP, PandaGPT, SpeechGPT, NExT-GPT 등이 포함됩니다. 각 모델은 기초 벤치마크와 채팅 벤치마크에서 성능을 비교하였습니다.

#### 테스트 데이터
기초 벤치마크는 다양한 오디오 유형(음성, 소리, 음악)에 대한 19개의 작업을 포함하고 있으며, 각 작업은 특정 능력을 평가하기 위해 설계되었습니다. 채팅 벤치마크는 오디오 신호에 기반한 질문과 답변 쌍을 포함하여, 모델이 실제 상황에서 사용자 지시를 따르는 능력을 평가합니다.

#### 메트릭
모델의 성능은 정확도(정확히 맞춘 비율)와 GPT-4를 활용한 평가 점수로 측정되었습니다. 기초 벤치마크에서는 단일 선택 질문에 대한 정확도를 평가하였고, 채팅 벤치마크에서는 생성된 답변의 유용성, 관련성, 정확성 및 포괄성을 기준으로 점수를 매겼습니다.

#### 비교
결과적으로, Qwen-Audio Turbo는 기초 벤치마크와 채팅 벤치마크 모두에서 가장 높은 평균 점수를 기록하였으며, SALMONN과 Qwen-Audio-Chat이 뒤를 이었습니다. 그러나 대부분의 모델은 단일 선택 지시를 따르는 데 어려움을 겪었으며, 이는 모델의 출력 형식의 다양성 때문입니다. Whisper와 GPT-4의 조합은 채팅 벤치마크에서 가장 높은 점수를 기록하였습니다.

이러한 결과는 LALMs의 오디오 이해 및 지시 따르기 능력에 한계가 있음을 보여주며, 향후 연구 방향에 대한 통찰을 제공합니다.

---



This paper introduces AIR-Bench, a new benchmark designed to evaluate the performance of large audio-language models (LALMs). AIR-Bench consists of two main benchmarks: the foundation benchmark and the chat benchmark. The foundation benchmark comprises 19 audio tasks and includes approximately 19,000 single-choice questions. The chat benchmark contains over 2,000 open-ended questions, assessing the model's ability to understand complex audio and follow instructions.

#### Competing Models
The paper evaluates several LALM models, including SALMONN, Qwen-Audio-Chat, Qwen-Audio Turbo, BLSP, PandaGPT, SpeechGPT, and NExT-GPT. Each model's performance is compared across both the foundation and chat benchmarks.

#### Test Data
The foundation benchmark includes 19 tasks covering various audio types (speech, sound, music), with each task designed to assess specific capabilities. The chat benchmark includes question-and-answer pairs based on audio signals, evaluating the model's ability to follow user instructions in real-world scenarios.

#### Metrics
Model performance is measured using accuracy (the proportion of exact matches) and scores evaluated using GPT-4. The foundation benchmark assesses accuracy on single-choice questions, while the chat benchmark scores the generated answers based on usefulness, relevance, accuracy, and comprehensiveness.

#### Comparison
Ultimately, Qwen-Audio Turbo achieved the highest average scores in both the foundation and chat benchmarks, followed by SALMONN and Qwen-Audio-Chat. However, most models struggled with following single-choice instructions, attributed to the variability in output formats. The combination of Whisper and GPT-4 recorded the highest score in the chat benchmark.

These results reveal limitations in the audio understanding and instruction-following capabilities of LALMs, providing insights into future research directions.


<br/>
# 예제



AIR-Bench는 대규모 오디오-언어 모델(LALMs)의 성능을 평가하기 위해 설계된 벤치마크입니다. 이 벤치마크는 두 가지 주요 차원으로 구성되어 있습니다: 기초 벤치마크와 채팅 벤치마크입니다.

1. **기초 벤치마크 (Foundation Benchmark)**:
   - **목적**: LALMs의 기본적인 오디오 이해 능력을 평가합니다.
   - **구성**: 19개의 서로 다른 오디오 작업으로 구성되어 있으며, 총 19,000개 이상의 단일 선택 질문이 포함되어 있습니다.
   - **예시 작업**:
     - **음성 인식 (Speech Recognition)**: 주어진 오디오에서 특정 단어가 발음되는 시점을 선택하는 작업입니다.
       - **질문 예시**: "‘hate’라는 단어가 발음된 시점을 선택하세요."
       - **선택지 예시**: A. [7.67, 8.05], B. [1.03, 1.53], C. [3.07, 3.27], D. [7.02, 7.21]
     - **감정 인식 (Emotion Recognition)**: 화자의 감정을 추론하는 작업입니다.
       - **질문 예시**: "화자의 감정은 무엇인가요?"
       - **선택지 예시**: A. 화남, B. 행복, C. 슬픔, D. 중립
     - **음악 장르 분류 (Music Genre Classification)**: 주어진 음악의 장르를 식별하는 작업입니다.
       - **질문 예시**: "이 음악의 장르는 무엇인가요?"
       - **선택지 예시**: A. 재즈, B. 록, C. 컨트리, D. 실험적

2. **채팅 벤치마크 (Chat Benchmark)**:
   - **목적**: LALMs가 복잡한 오디오 기반 질문에 대해 개방형 응답을 생성하는 능력을 평가합니다.
   - **구성**: 2,000개 이상의 오디오 기반 개방형 질문이 포함되어 있습니다.
   - **예시 작업**:
     - **음악 질문 응답 (Music Question Answering)**: 주어진 음악에 대한 질문에 대해 설명하는 작업입니다.
       - **질문 예시**: "이 음악이 전달하는 감정은 무엇인가요?"
       - **참고 답변 예시**: "이 음악은 평화롭고 차분한 감정을 전달합니다."
     - **혼합 오디오 질문 응답 (Mixed Audio Question Answering)**: 여러 오디오 유형이 혼합된 경우에 대한 질문입니다.
       - **질문 예시**: "이 오디오에서 들리는 소음은 무엇인가요?"
       - **참고 답변 예시**: "이 소음은 배경 음악과 함께하는 사람의 목소리입니다."

이러한 벤치마크는 LALMs의 성능을 체계적으로 평가하고, 향후 연구 방향을 제시하는 데 기여합니다.

---




AIR-Bench is a benchmark designed to evaluate the performance of large audio-language models (LALMs). This benchmark consists of two main dimensions: the foundation benchmark and the chat benchmark.

1. **Foundation Benchmark**:
   - **Purpose**: To assess the basic audio understanding capabilities of LALMs.
   - **Composition**: It consists of 19 different audio tasks, with over 19,000 single-choice questions.
   - **Example Tasks**:
     - **Speech Recognition**: A task where the model selects the time when a specific word is spoken in the given audio.
       - **Question Example**: "Choose when the word ‘hate’ is spoken."
       - **Choice Example**: A. [7.67, 8.05], B. [1.03, 1.53], C. [3.07, 3.27], D. [7.02, 7.21]
     - **Emotion Recognition**: A task to infer the emotion of the speaker.
       - **Question Example**: "What emotion is at the forefront of the speaker's words?"
       - **Choice Example**: A. Angry, B. Happy, C. Sad, D. Neutral
     - **Music Genre Classification**: A task to identify the genre of the given music.
       - **Question Example**: "What is the genre of this music?"
       - **Choice Example**: A. Jazz, B. Rock, C. Country, D. Experimental

2. **Chat Benchmark**:
   - **Purpose**: To evaluate the ability of LALMs to generate open-ended responses to complex audio-based questions.
   - **Composition**: It includes over 2,000 open-ended questions based on audio.
   - **Example Tasks**:
     - **Music Question Answering**: A task where the model explains the answer to a question about the given music.
       - **Question Example**: "What emotion does this music convey?"
       - **Reference Answer Example**: "This music conveys a sense of peace and calm."
     - **Mixed Audio Question Answering**: Questions related to audio that combines multiple types.
       - **Question Example**: "What noise is heard in this audio?"
       - **Reference Answer Example**: "The noise is the voice of a person along with background music."

These benchmarks systematically evaluate the performance of LALMs and contribute to guiding future research directions.

<br/>
# 요약
이 논문에서는 AIR-Bench라는 새로운 벤치마크를 소개하여 대규모 오디오-언어 모델(LALMs)의 이해 및 상호작용 능력을 평가합니다. AIR-Bench는 19개의 오디오 작업과 19,000개 이상의 단일 선택 질문으로 구성된 기초 벤치마크와 2,000개 이상의 개방형 질문으로 구성된 채팅 벤치마크를 포함합니다. 실험 결과, 기존 LALMs는 제한된 오디오 이해 및 지시 따르기 능력을 보였으며, AIR-Bench는 향후 연구 방향에 대한 통찰력을 제공합니다.

---

This paper introduces AIR-Bench, a new benchmark designed to evaluate the understanding and interaction capabilities of large audio-language models (LALMs). AIR-Bench consists of a foundation benchmark with 19 audio tasks and over 19,000 single-choice questions, along with a chat benchmark featuring over 2,000 open-ended questions. Experimental results show that existing LALMs exhibit limited audio understanding and instruction-following abilities, and AIR-Bench provides insights into future research directions.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **AIR-Bench 개요 (Figure 1)**: AIR-Bench는 오디오 신호의 다양한 유형을 이해하고 인간과 상호작용하는 LALMs의 능력을 평가하기 위해 설계된 계층적 벤치마크입니다. 두 가지 주요 차원인 기초 및 채팅 벤치마크로 나뉘며, 기초 벤치마크는 19개의 오디오 작업과 19,000개 이상의 단일 선택 질문으로 구성되어 있습니다. 채팅 벤치마크는 2,000개 이상의 오픈 엔디드 질문으로 구성되어 있습니다.

- **오디오 혼합 전략 (Figure 2)**: 이 그림은 오디오 혼합 전략을 설명합니다. 볼륨 조절과 시간적 위치 이동을 통해 두 개의 오디오 클립을 혼합하여 복잡한 오디오 신호를 생성합니다. 이 과정에서 생성된 메타 정보는 오디오의 텍스트 표현을 더욱 풍부하게 만듭니다.

- **자동 생성 평가 (Figure 3)**: LALMs가 오디오 입력과 질문을 기반으로 가설을 생성하고, GPT-4가 이 가설을 평가하는 방법을 보여줍니다. 기초 벤치마크에서는 정답이 금본위 선택으로 사용되며, 채팅 벤치마크에서는 GPT-4가 생성한 참조 답변이 사용됩니다.

#### 2. 테이블
- **기초 벤치마크 성능 (Table 3)**: 다양한 LALMs의 성능을 비교한 결과, Qwen-Audio Turbo와 Qwen-Audio-Chat이 기초 벤치마크에서 가장 높은 점수를 기록했습니다. SALMONN과 PandaGPT도 주목할 만한 성능을 보였습니다.

- **정확도 (Table 6)**: 각 모델의 기초 벤치마크에서의 정확도를 보여줍니다. Qwen-Audio Turbo는 여러 작업에서 높은 정확도를 기록했으며, 특히 음성 및 음악 관련 작업에서 두드러진 성과를 보였습니다.

- **인간 평가 (Figure 4)**: GPT-4와 인간 평가자 간의 일관성을 비교한 결과, GPT-4가 인간의 판단과 높은 일관성을 보였습니다. 채팅 벤치마크에서는 모델 간의 쌍 비교를 통해 성능을 평가했습니다.

#### 3. 어펜딕스
- **질문 및 선택 예시 (Table 5)**: 기초 벤치마크의 각 작업에 대한 질문과 선택 예시를 제공합니다. 이 표는 각 작업의 질문 형식과 선택지를 명확히 보여줍니다.

- **모델 응답 예시 (Figure 6, Figure 7)**: 다양한 모델의 응답을 보여주는 예시로, 각 모델이 기초 벤치마크와 채팅 벤치마크에서 어떻게 반응했는지를 시각적으로 나타냅니다.




#### 1. Diagrams and Figures
- **Overview of AIR-Bench (Figure 1)**: AIR-Bench is a hierarchical benchmark designed to evaluate the ability of LALMs to understand various types of audio signals and interact with humans. It is divided into two main dimensions: foundation and chat benchmarks, with the foundation benchmark consisting of 19 audio tasks and over 19,000 single-choice questions. The chat benchmark includes over 2,000 open-ended questions.

- **Audio Mixing Strategy (Figure 2)**: This figure illustrates the audio mixing strategy, which combines two audio clips through loudness control and temporal dislocation to create complex audio signals. The resulting meta-information enriches the textual representation of the audio.

- **Automated Generative Evaluation (Figure 3)**: It shows how LALMs generate hypotheses based on audio input and questions, with GPT-4 evaluating these hypotheses. In the foundation benchmark, the correct answer is used as the golden choice, while in the chat benchmark, the reference answer is generated by GPT-4.

#### 2. Tables
- **Performance on Foundation Benchmark (Table 3)**: The results comparing various LALMs show that Qwen-Audio Turbo and Qwen-Audio-Chat achieved the highest scores in the foundation benchmark. SALMONN and PandaGPT also demonstrated noteworthy performances.

- **Accuracy (Table 6)**: This table presents the accuracy of each model across tasks in the foundation benchmark. Qwen-Audio Turbo recorded high accuracy across several tasks, particularly in speech and music-related tasks.

- **Human Evaluation (Figure 4)**: The comparison of consistency between GPT-4 and human evaluators shows that GPT-4 exhibited high consistency with human judgments. In the chat benchmark, pairwise comparisons among models were conducted to assess performance.

#### 3. Appendix
- **Examples of Questions and Choices (Table 5)**: This table provides examples of questions and choices for each task in the foundation benchmark, clearly illustrating the question formats and options.

- **Examples of Model Responses (Figure 6, Figure 7)**: These figures show representative responses from various models in the foundation and chat benchmarks, visually demonstrating how each model reacted.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{yang2024airbench,
  title={AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension},
  author={Qian Yang and Jin Xu and Wenrui Liu and Yunfei Chu and Ziyue Jiang and Xiaohuan Zhou and Yichong Leng and Yuanjun Lv and Zhou Zhao and Chang Zhou and Jingren Zhou},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1979--1998},
  year={2024},
  month={August},
  publisher={Association for Computational Linguistics},
  url={https://github.com/OFA-Sys/AIR-Bench}
}
```

### 시카고 스타일

Yang, Qian, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, and Jingren Zhou. 2024. "AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension." In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 1979–1998. Association for Computational Linguistics. https://github.com/OFA-Sys/AIR-Bench.
