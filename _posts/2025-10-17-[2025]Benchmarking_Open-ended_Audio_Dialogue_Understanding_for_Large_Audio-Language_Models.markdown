---
layout: post
title:  "[2025]Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models"
date:   2025-10-17 21:23:22 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 대규모 오디오-언어 모델(LALMs)의 오디오 대화 이해 능력을 평가하기 위해 ADU-Bench라는 벤치마크를 제안하였다.


짧은 요약(Abstract) :


이 논문에서는 최근 오디오 대화 능력을 갖춘 대형 오디오-언어 모델(LALMs)의 성능을 평가하기 위한 포괄적인 벤치마크인 오디오 대화 이해 벤치마크(ADU-Bench)를 제안합니다. ADU-Bench는 4개의 벤치마크 데이터셋으로 구성되어 있으며, 이는 LALMs의 오픈 엔디드 오디오 대화 능력을 3가지 일반 시나리오, 12가지 기술, 9개 다국어 및 4가지 모호성 처리 범주에서 평가합니다. 특히, 우리는 문장의 동일한 문자적 의미를 넘어서는 다양한 의도를 표현하는 오디오 대화에서의 모호성 처리 평가를 처음으로 제안합니다. ADU-Bench는 20,000개 이상의 오픈 엔디드 오디오 대화를 포함하고 있으며, 16개의 LALMs에 대한 광범위한 실험을 통해 기존 LALMs가 수학 기호 및 공식, 역할극과 같은 인간 행동 이해, 여러 언어 이해, 그리고 억양, 일시 정지 위치 및 동음이의어와 같은 다양한 음성 요소에서의 오디오 대화 모호성 처리에 어려움을 겪고 있음을 밝혀냈습니다.



This paper proposes a comprehensive benchmark called the Audio Dialogue Understanding Benchmark (ADU-Bench) to evaluate the performance of Large Audio-Language Models (LALMs) that have recently unlocked audio dialogue capabilities. ADU-Bench consists of four benchmark datasets that assess the open-ended audio dialogue ability of LALMs across three general scenarios, twelve skills, nine multilingual languages, and four categories of ambiguity handling. Notably, we introduce the evaluation of ambiguity handling in audio dialogues that express different intentions beyond the same literal meaning of sentences. ADU-Bench includes over 20,000 open-ended audio dialogues, and extensive experiments on 16 LALMs reveal that existing LALMs struggle with mathematical symbols and formulas, understanding human behavior such as roleplay, comprehending multiple languages, and handling audio dialogue ambiguities arising from various phonetic elements, such as intonations, pause positions, and homophones.


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



이 논문에서는 대규모 오디오-언어 모델(Large Audio-Language Models, LALMs)의 오디오 대화 이해 능력을 평가하기 위한 새로운 벤치마크인 ADU-Bench를 제안합니다. ADU-Bench는 네 가지 주요 데이터셋으로 구성되어 있으며, 각 데이터셋은 다양한 시나리오와 기술을 포함하여 LALMs의 성능을 종합적으로 평가합니다.

1. **모델 아키텍처**: LALMs는 오디오와 언어를 통합하여 다양한 오디오 관련 작업을 수행할 수 있도록 설계된 모델입니다. 이 모델들은 일반적으로 두 가지 유형으로 나뉘며, 첫 번째는 오디오 이해에 특화된 엔드 투 엔드 모델이고, 두 번째는 오디오와 언어를 결합하여 다양한 작업을 수행할 수 있는 모델입니다. 예를 들어, Whisper와 같은 자동 음성 인식 모델과 GPT-4와 같은 언어 모델을 결합하여 오디오 데이터를 처리합니다.

2. **데이터셋 구성**: ADU-Bench는 총 20,715개의 오픈 엔드 오디오 대화로 구성되어 있으며, 이는 실제 녹음과 합성 오디오 샘플을 포함합니다. 데이터셋은 다음과 같은 네 가지 주요 카테고리로 나뉩니다:
   - **ADU-General**: 일반적인 대화 이해를 평가하며, 유용한 질문, 일상적인 질문, 일상적인 진술을 포함합니다.
   - **ADU-Skill**: 수학, 물리학, 코딩 등 12가지 기술 기반 대화를 평가합니다.
   - **ADU-Multilingual**: 9개 언어(영어, 프랑스어, 중국어 등)에서의 다국어 대화 이해 능력을 평가합니다.
   - **ADU-Ambiguity**: 억양, 일시 정지, 동음이의어 등 다양한 음성 요소에서의 모호성 처리 능력을 평가합니다.

3. **훈련 데이터**: LALMs는 대규모의 텍스트 및 오디오 데이터로 훈련됩니다. 이 데이터는 다양한 소스에서 수집되며, 특히 ADU-Bench의 경우, GPT-4와 인간의 검토를 통해 생성된 텍스트 대화가 포함됩니다. 합성 오디오 샘플은 Speech Synthesis Markup Language(SSML)를 사용하여 생성됩니다.

4. **평가 방법**: LALMs의 성능은 GPT-4와 같은 고급 언어 모델을 사용하여 평가됩니다. 모델은 사용자 오디오 입력에 대해 텍스트 응답을 생성하고, 이 응답의 품질을 평가하기 위해 텍스트 전사본과 참조를 GPT-4에 입력합니다. 평가 점수는 0에서 10까지의 척도로 제공되며, 높은 점수는 더 나은 대화 이해 능력을 나타냅니다.

이러한 방법론을 통해 ADU-Bench는 LALMs의 오디오 대화 이해 능력을 체계적으로 평가하고, 향후 연구 및 개발에 기여할 수 있는 기초 자료를 제공합니다.

---




This paper proposes a new benchmark called ADU-Bench to evaluate the audio dialogue understanding capabilities of Large Audio-Language Models (LALMs). ADU-Bench consists of four main datasets, each designed to comprehensively assess the performance of LALMs across various scenarios and skills.

1. **Model Architecture**: LALMs are designed to integrate audio and language to perform a variety of audio-related tasks. These models are generally classified into two types: end-to-end models specialized in audio understanding and models that combine audio and language to perform diverse tasks. For instance, they may combine automatic speech recognition models like Whisper with language models like GPT-4 to process audio data.

2. **Dataset Composition**: ADU-Bench comprises a total of 20,715 open-ended audio dialogues, which include both real recordings and synthetic audio samples. The datasets are categorized into four main categories:
   - **ADU-General**: Evaluates general dialogue understanding, including helpful questions, daily questions, and daily statements.
   - **ADU-Skill**: Assesses skill-based dialogues across 12 different domains, such as mathematics, physics, and coding.
   - **ADU-Multilingual**: Tests multilingual dialogue understanding across 9 languages, including English, French, and Chinese.
   - **ADU-Ambiguity**: Evaluates the ability to handle ambiguities arising from various phonetic elements, such as intonation, pause positions, and homophones.

3. **Training Data**: LALMs are trained on large-scale text and audio data. This data is collected from various sources, and specifically for ADU-Bench, it includes text dialogues generated through GPT-4 and human review. Synthetic audio samples are generated using Speech Synthesis Markup Language (SSML).

4. **Evaluation Method**: The performance of LALMs is evaluated using advanced language models like GPT-4. The models generate textual responses to user audio inputs, and the quality of these responses is assessed by inputting the textual transcriptions and references into GPT-4. Evaluation scores are provided on a scale from 0 to 10, with higher scores indicating better dialogue understanding capabilities.

Through this methodology, ADU-Bench systematically evaluates the audio dialogue understanding of LALMs and provides foundational resources that can contribute to future research and development.


<br/>
# Results



이 논문에서는 대규모 오디오-언어 모델(Large Audio-Language Models, LALMs)의 오디오 대화 이해 능력을 평가하기 위한 새로운 벤치마크인 ADU-Bench를 제안합니다. ADU-Bench는 4개의 데이터셋으로 구성되어 있으며, 각 데이터셋은 다양한 시나리오와 기술, 언어, 모호성 처리 방법을 포함하고 있습니다. 총 20,715개의 오픈 엔드 오디오 대화가 포함되어 있으며, 이를 통해 LALMs의 성능을 평가합니다.

#### 경쟁 모델
논문에서는 16개의 LALMs를 평가하였으며, 이들 모델은 다음과 같습니다:
- PandaGPT
- NExT-GPT
- Qwen-Audio
- Mini-Omni
- SALMONN
- Qwen-Audio-Chat
- SpeechGPT
- Moshi
- SALMONN (13B)
- BLSP
- Step-Audio-Chat
- Whisper + LLaMA-2
- Whisper + LLaMA-3
- Whisper + LLaMA-3 (70B)
- Whisper + GPT-4
- GPT-4o

#### 테스트 데이터
ADU-Bench는 다음과 같은 4개의 데이터셋으로 구성됩니다:
1. **ADU-General Dataset**: 일반적인 대화 이해를 평가하며, 유용한 질문, 일상적인 질문, 일상적인 진술을 포함합니다.
2. **ADU-Skill Dataset**: 수학, 물리학, 화학, 생물학, 컴퓨터 과학, 코딩, 법률, 금융, 상식, 글쓰기, 역할극, 의학 등 12개의 기술을 평가합니다.
3. **ADU-Multilingual Dataset**: 아랍어, 중국어, 영어, 프랑스어, 독일어, 일본어, 한국어, 러시아어, 스페인어 등 9개 언어의 대화 이해 능력을 평가합니다.
4. **ADU-Ambiguity Dataset**: 억양, 일시 정지, 동음이의어, 반복 등 4가지 유형의 모호성을 처리하는 능력을 평가합니다.

#### 메트릭
모델의 성능은 GPT-4와 같은 고급 LLM을 사용하여 평가되며, 각 모델의 응답은 0에서 10까지의 점수로 평가됩니다. 이 점수는 유용성, 관련성, 정확성 및 포괄성을 기준으로 하여 산출됩니다. 또한, 평가 과정에서 위치 편향을 제거하기 위해 두 번의 평가를 수행하고 평균 점수를 보고합니다.

#### 결과 비교
결과적으로, GPT-4o 모델이 가장 높은 평균 점수인 8.16을 기록하여 가장 우수한 성능을 보였습니다. 반면, PandaGPT와 NExT-GPT는 상대적으로 낮은 성능을 보였으며, 이들은 오디오 대화 이해에서 개선이 필요함을 나타냅니다. LALMs는 수학 기호와 공식을 처리하는 데 어려움을 겪고 있으며, 인간 행동 이해와 같은 복잡한 대화 상황에서도 한계를 보였습니다.

### English Version

This paper proposes a new benchmark called ADU-Bench to evaluate the audio dialogue understanding capabilities of Large Audio-Language Models (LALMs). ADU-Bench consists of four benchmark datasets, each encompassing various scenarios, skills, languages, and ambiguity handling methods. In total, it includes 20,715 open-ended audio dialogues to assess the performance of LALMs.

#### Competing Models
The paper evaluates 16 LALMs, which include:
- PandaGPT
- NExT-GPT
- Qwen-Audio
- Mini-Omni
- SALMONN
- Qwen-Audio-Chat
- SpeechGPT
- Moshi
- SALMONN (13B)
- BLSP
- Step-Audio-Chat
- Whisper + LLaMA-2
- Whisper + LLaMA-3
- Whisper + LLaMA-3 (70B)
- Whisper + GPT-4
- GPT-4o

#### Test Data
ADU-Bench is composed of the following four datasets:
1. **ADU-General Dataset**: Evaluates general dialogue understanding, including helpful questions, daily questions, and daily statements.
2. **ADU-Skill Dataset**: Assesses domain-specific skills across 12 areas, including mathematics, physics, chemistry, biology, computer science, coding, law, finance, common sense, writing, roleplay, and medicine.
3. **ADU-Multilingual Dataset**: Tests multilingual dialogue understanding across 9 languages: Arabic, Chinese, English, French, German, Japanese, Korean, Russian, and Spanish.
4. **ADU-Ambiguity Dataset**: Evaluates the ability to handle ambiguity from four types: intonation-based, pause-based, homophone-based, and repetition-based.

#### Metrics
The performance of the models is evaluated using advanced LLMs like GPT-4, with each model's responses scored on a scale from 0 to 10. These scores are derived based on criteria such as helpfulness, relevance, accuracy, and comprehensiveness. Additionally, to eliminate position bias in the evaluation process, two evaluations are conducted, and the average score is reported.

#### Result Comparison
Ultimately, the GPT-4o model achieved the highest average score of 8.16, indicating the best performance among the evaluated models. In contrast, PandaGPT and NExT-GPT exhibited relatively low performance, suggesting a need for improvement in audio dialogue understanding. LALMs struggled with processing mathematical symbols and formulas and faced limitations in understanding complex dialogue situations involving human behavior.


<br/>
# 예제



이 논문에서는 대화형 오디오 이해를 평가하기 위한 새로운 벤치마크인 ADU-Bench를 제안합니다. ADU-Bench는 4개의 데이터셋으로 구성되어 있으며, 각 데이터셋은 다양한 시나리오와 기술을 포함하여 대규모 오디오-언어 모델(LALMs)의 성능을 평가합니다. 이 데이터셋은 다음과 같은 구성 요소를 포함합니다:

1. **ADU-General 데이터셋**: 일반적인 대화 이해를 평가합니다. 이 데이터셋은 세 가지 시나리오(유용한 질문, 일상 질문, 일상 진술)를 포함하며, 각 시나리오에 대해 4,000개의 오디오 대화가 포함되어 있습니다. 예를 들어, "프랑스의 수도는 어디인가요?"라는 질문에 대한 대답은 "프랑스의 수도는 파리입니다."와 같은 형식입니다.

2. **ADU-Skill 데이터셋**: 특정 기술에 대한 대화 능력을 평가합니다. 이 데이터셋은 수학, 물리학, 화학, 생물학, 컴퓨터 과학, 코딩, 법률, 금융, 상식, 글쓰기, 역할극, 의학 등 12개의 도메인에 대해 3,750개의 오디오 대화를 포함합니다. 예를 들어, "2 + 2는 얼마인가요?"라는 질문에 대한 대답은 "2 + 2는 4입니다."와 같은 형식입니다.

3. **ADU-Multilingual 데이터셋**: 다국어 대화 이해 능력을 평가합니다. 이 데이터셋은 아랍어, 중국어, 영어, 프랑스어, 독일어, 일본어, 한국어, 러시아어, 스페인어 등 9개 언어로 구성되어 있으며, 각 언어에 대해 400개의 오디오 대화가 포함되어 있습니다. 예를 들어, "2023년 NBA 챔피언은 어떤 팀인가요?"라는 질문이 중국어로 "2023年NBA总冠军是哪支队伍？"로 표현될 수 있습니다.

4. **ADU-Ambiguity 데이터셋**: 오디오 대화에서의 모호성 처리 능력을 평가합니다. 이 데이터셋은 억양 기반, 일시 정지 기반, 동음이의어 기반, 반복 기반의 모호성을 포함하여 총 1,390개의 오디오 대화를 포함합니다. 예를 들어, "정말 완벽한 해변의 날이네요."라는 문장은 억양에 따라 긍정적이거나 부정적인 의미로 해석될 수 있습니다.

각 데이터셋은 LALMs에 대한 입력(오디오 쿼리)과 기대되는 출력(텍스트 참조)으로 구성됩니다. 모델은 오디오 쿼리를 입력받아 텍스트 응답을 생성하며, 이 응답의 품질은 GPT-4와 같은 평가 모델을 통해 점수화됩니다. 




This paper proposes a new benchmark for evaluating open-ended audio dialogue understanding called ADU-Bench. ADU-Bench consists of four datasets, each designed to assess the performance of Large Audio-Language Models (LALMs) across various scenarios and skills. The datasets include the following components:

1. **ADU-General Dataset**: This dataset evaluates general dialogue understanding. It includes 12,000 audio dialogues across three scenarios: helpful questions, daily questions, and daily statements. For example, a question like "What is the capital of France?" would have the answer "The capital of France is Paris."

2. **ADU-Skill Dataset**: This dataset assesses the dialogue capabilities related to specific skills. It contains 3,750 audio dialogues across 12 domains, including Mathematics, Physics, Chemistry, Biology, Computer Science, Coding, Law, Finance, Common Sense, Writing, Roleplay, and Medicine. For instance, a question like "What is 2 + 2?" would have the answer "2 + 2 is 4."

3. **ADU-Multilingual Dataset**: This dataset evaluates multilingual dialogue understanding capabilities. It covers 9 languages: Arabic, Chinese, English, French, German, Japanese, Korean, Russian, and Spanish, with 400 audio dialogues for each language. For example, the question "Which team won the NBA championship in 2023?" could be expressed in Chinese as "2023年NBA总冠军是哪支队伍？"

4. **ADU-Ambiguity Dataset**: This dataset assesses the ability to handle ambiguities in audio dialogues. It includes 1,390 audio dialogues that cover intonation-based, pause-based, homophone-based, and repetition-based ambiguities. For example, the sentence "What a perfect day for the beach." can be interpreted positively or negatively depending on the intonation used.

Each dataset consists of inputs (audio queries) and expected outputs (text references) for the LALMs. The models receive audio queries and generate textual responses, which are then scored for quality using evaluation models like GPT-4.

<br/>
# 요약


이 논문에서는 대규모 오디오-언어 모델(LALMs)의 오디오 대화 이해 능력을 평가하기 위해 ADU-Bench라는 벤치마크를 제안하였다. ADU-Bench는 일반 대화, 기술 기반 대화, 다국어 대화, 모호성 처리 등 4개의 데이터셋으로 구성되어 있으며, 20,715개의 오픈 엔드 오디오 대화를 포함하고 있다. 실험 결과, 기존 LALMs는 수학 기호 및 공식 처리, 인간 행동 이해, 다국어 처리, 오디오 대화의 모호성 처리에서 여전히 많은 개선이 필요함을 보여주었다.

---

This paper proposes a benchmark called ADU-Bench to evaluate the audio dialogue understanding capabilities of Large Audio-Language Models (LALMs). ADU-Bench consists of four datasets covering general dialogue, skill-based dialogue, multilingual dialogue, and ambiguity handling, containing a total of 20,715 open-ended audio dialogues. The experimental results reveal that existing LALMs still require significant improvements in handling mathematical symbols and formulas, understanding human behavior, multilingual processing, and addressing ambiguities in audio dialogues.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **ADU-Bench 구조**: ADU-Bench는 4개의 데이터셋(ADU-General, ADU-Skill, ADU-Multilingual, ADU-Ambiguity)으로 구성되어 있으며, 각 데이터셋은 다양한 시나리오와 기술을 평가합니다. 이 구조는 LALMs의 오픈 엔드 오디오 대화 이해 능력을 종합적으로 평가하는 데 중점을 두고 있습니다.

2. **성능 평가 결과**: 각 LALM의 성능은 다양한 시나리오에서 평가되며, 특히 ADU-General 데이터셋에서 유용한 질문에 대한 성능이 더 높게 나타났습니다. 이는 LALMs가 특정 정보 검색을 요구하는 질문에 더 잘 대응할 수 있음을 시사합니다.

#### 테이블
1. **성능 점수**: 각 LALM의 평균 점수는 ADU-Bench의 각 데이터셋에서 평가되며, GPT-4o가 가장 높은 점수를 기록했습니다. 이는 GPT-4o가 오디오 대화 이해에서 가장 뛰어난 성능을 보임을 나타냅니다.

2. **ADU-Skill 데이터셋**: LALMs는 생물학, 컴퓨터 과학, 법률 등 특정 분야에서 상대적으로 높은 성능을 보였으나, 수학 및 코딩과 같은 분야에서는 낮은 성능을 보였습니다. 이는 이러한 분야가 수학 기호나 프로그래밍 언어를 포함하고 있어 LALMs가 이해하는 데 어려움을 겪기 때문입니다.

3. **ADU-Multilingual 데이터셋**: LALMs는 영어에서 가장 높은 성능을 보였으며, 그 다음으로 독일어, 스페인어, 프랑스어 순으로 성능이 나타났습니다. 이는 LALMs가 영어에 대한 훈련 데이터가 많기 때문입니다.

4. **ADU-Ambiguity 데이터셋**: LALMs는 반복 기반 모호성을 처리하는 데 상대적으로 더 나은 성능을 보였으나, 억양 기반, 일시 정지 기반, 동음이의어 기반 모호성 처리에서는 어려움을 겪었습니다.

#### 어펜딕스
- **데이터 생성 방법**: ADU-Bench의 데이터는 실제 오디오와 합성 오디오를 포함하며, 합성 오디오는 SSML을 사용하여 생성되었습니다. 이는 다양한 음성 특성을 반영하여 실제 대화와 유사한 오디오를 생성하는 데 기여합니다.




#### Diagrams and Figures
1. **Structure of ADU-Bench**: ADU-Bench consists of 4 datasets (ADU-General, ADU-Skill, ADU-Multilingual, ADU-Ambiguity), each designed to evaluate various scenarios and skills. This structure focuses on comprehensively assessing the open-ended audio dialogue understanding capabilities of LALMs.

2. **Performance Evaluation Results**: The performance of each LALM is evaluated across various scenarios, with particularly high performance observed in the ADU-General dataset for helpful questions. This suggests that LALMs are better at responding to questions that require specific information retrieval.

#### Tables
1. **Performance Scores**: The average scores of each LALM are evaluated across the datasets in ADU-Bench, with GPT-4o achieving the highest score. This indicates that GPT-4o exhibits the best performance in audio dialogue understanding.

2. **ADU-Skill Dataset**: LALMs showed relatively high performance in specific domains such as Biology, Computer Science, and Law, but struggled in areas like Mathematics and Coding. This is attributed to the inclusion of mathematical symbols or programming languages, which pose challenges for LALMs.

3. **ADU-Multilingual Dataset**: LALMs performed best in English, followed by German, Spanish, and French. This is likely due to the abundance of training data available in English.

4. **ADU-Ambiguity Dataset**: LALMs exhibited relatively better performance in handling repetition-based ambiguity, while they struggled with intonation-based, pause-based, and homophone-based ambiguities.

#### Appendix
- **Data Generation Method**: The data in ADU-Bench includes both real-world and synthetic audio, with synthetic audio generated using SSML. This contributes to creating audio that closely resembles real conversations by reflecting various voice characteristics.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{gao2025benchmarking,
  title={Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models},
  author={Kuofeng Gao and Shu-Tao Xia and Ke Xu and Philip Torr and Jindong Gu},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={4763--4784},
  year={2025},
  month={July},
  publisher={Association for Computational Linguistics},
  url={https://adu-bench.github.io/}
}
```

### 시카고 스타일

Kuofeng Gao, Shu-Tao Xia, Ke Xu, Philip Torr, and Jindong Gu. "Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 4763–4784. July 2025. Association for Computational Linguistics. https://adu-bench.github.io/.
