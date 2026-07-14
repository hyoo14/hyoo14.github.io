---
layout: post
title:  "[2026]When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms"
date:   2026-07-14 00:57:01 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 오디오 플랫폼에서의 허위 정보 검증을 위한 새로운 접근 방식을 제안합니다.


짧은 요약(Abstract) :


이 논문은 오디오 플랫폼에서의 허위 정보 문제를 다루고 있습니다. 팟캐스트, 라디오, WhatsApp 음성 메시지 및 라이브 스트림과 같은 오디오 플랫폼이 대중 담론의 중심으로 자리 잡으면서, 수백만 개의 프로그램과 수억 명의 청취자가 존재하게 되었습니다. 그러나 기존의 사실 확인 시스템은 주로 서면 주장에 맞춰 설계되어 있어, 구술 미디어의 고유한 특성을 간과하고 있습니다. 저자들은 오디오 허위 정보가 단순히 텍스트 내용의 전사본이 아니라는 점을 강조하며, 구술적이고 대화적인 특성이 사실 확인의 어려움을 증가시킨다고 주장합니다. 이 논문은 다양한 매체와 플랫폼에서의 증거를 종합하고, 데이터셋과 방법론을 검토하며, 기존의 사실 확인 시스템이 오디오에서 실패하는 이유를 강조합니다. 저자들은 사실 확인을 발전시키기 위해 구술적이고 대화적인 현실에 맞춰 검증 파이프라인을 재구성해야 한다고 주장합니다.



This paper addresses the issue of misinformation on audio platforms. With podcasts, radio, WhatsApp voice notes, and live streams becoming central to public discourse, there are now millions of programs and hundreds of millions of listeners. However, existing fact-checking systems are primarily designed for written claims, overlooking the unique properties of spoken media. The authors emphasize that audio misinformation is not merely a transcript of textual content, arguing that its spoken and conversational characteristics increase the difficulties of verification. The paper synthesizes evidence across modalities and platforms, examines datasets and methods, and highlights why existing fact-checking pipelines fail in the audio context. The authors argue that advancing fact-checking requires rethinking verification pipelines to align with the spoken and conversational realities of audio.


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



이 논문에서는 오디오 플랫폼에서의 사실 확인을 위한 새로운 접근 방식을 제안합니다. 기존의 사실 확인 시스템은 주로 텍스트 기반으로 설계되어 있으며, 오디오와 대화의 특성을 고려하지 못하고 있습니다. 따라서, 저자들은 오디오 미디어의 고유한 속성을 반영한 새로운 모델과 방법론을 개발해야 한다고 주장합니다.

#### 1. 모델 아키텍처
저자들은 오디오 플랫폼에서의 사실 확인을 위해 다음과 같은 네 가지 주요 단계로 구성된 파이프라인을 제안합니다:

- **Claim Detection (CD)**: 이 단계에서는 오디오에서 체크할 가치가 있는 주장을 식별합니다. 전통적인 CD 모델은 일반적으로 문장 단위로 주장을 처리하지만, 대화형 오디오에서는 주장이 여러 턴에 걸쳐 분산되어 있을 수 있습니다. 따라서, 저자들은 턴과 화자 정보를 고려하여 인접한 턴을 집계하여 독립적인 주장 단위를 구성하는 방법을 제안합니다. 또한, 프로소디(음조, 강세, 속도)와 화자 역할을 조건으로 하여 주장을 탐지하는 것이 중요합니다.

- **Evidence Retrieval (ER)**: 이 단계에서는 주장을 뒷받침할 수 있는 증거를 검색합니다. 오디오 플랫폼에서는 증거가 연속적이며 클릭할 수 있는 텍스트가 아니기 때문에, 시간에 기반한 증거 검색이 필요합니다. 저자들은 에피소드 ID와 시작/종료 시간을 포함한 시간 기반 증거를 반환하는 방법을 제안합니다. 이를 통해 대화의 맥락을 유지하고, 반복되는 주장을 여러 에피소드에 걸쳐 연결할 수 있습니다.

- **Claim Verification (VER)**: 이 단계에서는 주장을 증거와 비교하여 진위를 판단합니다. 전통적인 텍스트 기반 모델은 단일 문장 쌍을 가정하지만, 오디오에서는 다중 턴과 시간적 제약이 존재합니다. 저자들은 다중 스팬, 시간 인식 추론을 통해 주장을 검증하고, 다이어리제이션 불확실성을 모델링하여 성능을 향상시키는 방법을 제안합니다.

- **Explanation Generation (GEN)**: 마지막으로, 이 단계에서는 검증 결과에 대한 설명을 생성합니다. 오디오 기반의 설명은 텍스트만으로는 전달할 수 없는 전달 방식(예: 비꼼, 강조, 주저)을 포함해야 합니다. 저자들은 각 설명이 읽을 수 있는 전사 스팬과 짧은 오디오 클립을 쌍으로 제공하여 사용자가 전달 방식을 판단할 수 있도록 하는 방법을 제안합니다.

#### 2. 트레이닝 데이터
저자들은 오디오 플랫폼에서의 사실 확인을 위한 대규모 데이터셋의 필요성을 강조합니다. 이 데이터셋은 사실 확인 가능성 또는 진위 레이블이 포함된 전사본, 화자 및 역할 메타데이터와 함께 정렬되어야 합니다. 또한, 저자들은 다양한 발음, 중첩, 잡음 등을 포함한 타겟 합성 변형을 통해 데이터셋을 보강할 것을 제안합니다.

이러한 방법론은 오디오 플랫폼에서의 사실 확인을 보다 효과적으로 수행할 수 있도록 하며, 기존의 텍스트 기반 접근 방식의 한계를 극복하는 데 기여할 것입니다.

---




This paper proposes a new approach to fact-checking in audio platforms. Existing fact-checking systems are primarily designed for text and do not account for the unique characteristics of audio and conversation. Therefore, the authors argue that new models and methodologies must be developed that reflect the unique properties of audio media.

#### 1. Model Architecture
The authors propose a pipeline consisting of four main stages for fact-checking in audio platforms:

- **Claim Detection (CD)**: In this stage, check-worthy claims are identified from audio. Traditional CD models typically process claims at the sentence level, but in conversational audio, claims may be distributed across multiple turns. Therefore, the authors suggest aggregating adjacent turns to form a standalone claim unit while considering turn and speaker information. Additionally, it is crucial to condition the detection on prosody (pitch, stress, speed) and speaker roles.

- **Evidence Retrieval (ER)**: This stage involves retrieving evidence that can support the claims. Since evidence in audio platforms is continuous and not clickable text, time-based evidence retrieval is necessary. The authors propose returning time-anchored evidence that includes episode IDs and start/end times. This approach helps maintain the context of the conversation and allows for linking recurring claims across multiple episodes.

- **Claim Verification (VER)**: In this stage, claims are compared against the retrieved evidence to determine their truthfulness. Traditional text-based models assume single-span claims, but audio presents multi-turn and time-qualified evidence. The authors suggest using multi-span, time-aware reasoning to verify claims and explicitly model diarization uncertainty to enhance performance.

- **Explanation Generation (GEN)**: Finally, this stage generates explanations for the verification results. Audio-based explanations must include delivery aspects (e.g., sarcasm, emphasis, hesitation) that cannot be conveyed through text alone. The authors propose that each explanation should pair a readable transcript span with a short audio clip, allowing users to assess the delivery.

#### 2. Training Data
The authors emphasize the need for large-scale datasets for fact-checking in audio platforms. These datasets should include transcripts with fact-checking labels or veracity labels, aligned with speaker and role metadata. Additionally, the authors suggest augmenting the datasets with targeted synthetic variants that include accents, overlaps, and noise.

These methodologies aim to enable more effective fact-checking in audio platforms and contribute to overcoming the limitations of existing text-based approaches.


<br/>
# Results



이 논문에서는 오디오 플랫폼에서의 사실 확인을 위한 새로운 접근 방식을 제안하고, 기존의 텍스트 기반 사실 확인 시스템이 오디오 대화에 어떻게 적용되지 않는지를 분석합니다. 연구의 주요 결과는 다음과 같습니다:

1. **경쟁 모델**: 기존의 사실 확인 모델들은 주로 텍스트 기반으로 설계되어 있으며, 단일 문장을 주장으로 간주합니다. 그러나 오디오 대화에서는 주장이 여러 턴에 걸쳐 분산되어 나타나기 때문에, 이러한 모델들은 효과적으로 작동하지 않습니다. 연구에서는 BERT 및 RoBERTa와 같은 문장 수준의 체크-워시니스 및 주장 분류기를 사용한 기존 모델들이 오디오 대화에서의 주장을 놓치거나 잘못 해석하는 경향이 있음을 보여주었습니다.

2. **테스트 데이터**: 연구에서는 MAD2라는 새로운 벤치마크 데이터셋을 사용하여, 1,000개의 대화와 8,192개의 문장, 3,368개의 체크-워시니스 주장을 포함하고 있습니다. 이 데이터셋은 각 주장에 대해 이진 진위 레이블이 부여되어 있으며, 주장이 발화된 정확한 시간 정보도 포함되어 있습니다. 이를 통해 모델이 대화의 맥락을 고려하여 주장을 검증할 수 있도록 합니다.

3. **메트릭**: 연구에서는 F1 점수와 AUC(Area Under Curve)와 같은 메트릭을 사용하여 모델의 성능을 평가했습니다. 예를 들어, MAD2 데이터셋에서 15초 전후의 맥락을 고려한 모델이 단순히 문장 단위로 주장을 검증하는 모델보다 더 높은 성능을 보였습니다. 이는 대화의 시간적 순서와 맥락이 주장의 진위를 판단하는 데 중요한 역할을 한다는 것을 시사합니다.

4. **비교**: 연구 결과, 기존의 텍스트 기반 모델들은 오디오 대화의 복잡성을 처리하는 데 한계가 있음을 보여주었습니다. 특히, 대화의 흐름, 발화자의 역할, 그리고 비언어적 신호(예: 억양, 감정 등)를 고려하지 않으면 주장을 정확히 검증하기 어렵습니다. 연구에서는 이러한 한계를 극복하기 위해, 턴과 발화자 정보를 고려한 새로운 모델 설계를 제안하였습니다.

이러한 결과들은 오디오 플랫폼에서의 사실 확인 연구가 기존의 텍스트 중심 접근 방식에서 벗어나, 대화의 맥락과 비언어적 요소를 통합한 새로운 방법론으로 나아가야 함을 강조합니다.

---




This paper proposes a new approach to fact-checking in audio platforms and analyzes how existing text-based fact-checking systems fail to apply to audio dialogue. The main results of the study are as follows:

1. **Competing Models**: Existing fact-checking models are primarily designed for text and treat single sentences as claims. However, in audio dialogue, claims often appear distributed across multiple turns, leading these models to miss or misinterpret claims. The study shows that existing models, such as sentence-level check-worthiness and claim classifiers based on BERT and RoBERTa, tend to overlook or misinterpret claims in audio dialogue.

2. **Test Data**: The study utilizes a new benchmark dataset called MAD2, which includes 1,000 dialogues, 8,192 sentences, and 3,368 check-worthy claims. Each claim is annotated with a binary true/false label and includes precise timing information for when the claim was made. This allows models to verify claims while considering the context of the dialogue.

3. **Metrics**: The study employs metrics such as F1 score and AUC (Area Under Curve) to evaluate model performance. For instance, models that consider the context of 15 seconds before a claim perform better than those that verify claims based solely on isolated sentences. This suggests that the temporal order and context of dialogue play a crucial role in determining the veracity of claims.

4. **Comparison**: The results indicate that existing text-based models have limitations in handling the complexities of audio dialogue. Specifically, without considering the flow of conversation, the roles of speakers, and non-verbal cues (e.g., prosody, emotion), it becomes challenging to accurately verify claims. The study proposes a new model design that incorporates turn and speaker information to overcome these limitations.

These findings emphasize the need for research on fact-checking in audio platforms to move beyond traditional text-centric approaches and adopt new methodologies that integrate the context of conversation and non-verbal elements.


<br/>
# 예제



이 논문에서는 오디오 플랫폼에서의 사실 확인을 위한 새로운 접근 방식을 제안하고 있습니다. 특히, 대화형 오디오 콘텐츠에서의 허위 정보 검증을 위한 데이터셋과 그 사용 방법에 대해 설명합니다. 여기서는 예시로 사용된 데이터셋 MAD2를 중심으로 설명하겠습니다.

#### 데이터셋 구성
MAD2 데이터셋은 약 1,000개의 대화로 구성되어 있으며, 각 대화는 8,192개의 문장과 3,368개의 검증 가능한 주장(check-worthy claims)을 포함하고 있습니다. 각 주장은 이진 true/false 레이블로 주석이 달려 있습니다. 데이터셋은 다음과 같은 구조를 가지고 있습니다:

- **입력(Input)**: 각 대화의 전체 텍스트와 해당 대화의 음성 파일. 각 문장은 시간 기반의 타임스탬프와 함께 제공되어, 모델이 특정 주장에 대한 문맥을 이해할 수 있도록 합니다.
- **출력(Output)**: 각 주장에 대한 진위 여부(true/false)와 그 주장을 뒷받침하는 증거를 찾기 위한 정보입니다. 모델은 주장을 검증하기 위해 대화의 맥락을 고려해야 하며, 이 과정에서 대화의 흐름과 발화자의 역할이 중요합니다.

#### 태스크(Task)
MAD2 데이터셋을 사용하여 수행하는 주요 태스크는 다음과 같습니다:

1. **주장 탐지(Claim Detection)**: 대화에서 검증 가능한 주장을 식별하는 작업입니다. 이 단계에서는 각 문장이 독립적으로 처리되지 않고, 대화의 흐름과 발화자의 역할을 고려하여 주장을 탐지합니다.
   
2. **증거 검색(Evidence Retrieval)**: 탐지된 주장에 대한 증거를 찾는 작업입니다. 이 과정에서는 대화의 특정 부분에서 주장을 뒷받침하는 증거를 찾기 위해 시간 기반의 검색이 필요합니다.

3. **주장 검증(Claim Verification)**: 수집된 증거를 바탕으로 주장의 진위를 판단하는 작업입니다. 이 단계에서는 다중 발화와 시간적 맥락을 고려하여 주장을 평가합니다.

4. **설명 생성(Explanation Generation)**: 검증 결과를 바탕으로 사용자가 이해할 수 있는 형태로 설명을 생성하는 작업입니다. 이 과정에서는 주장의 진위 여부와 관련된 증거를 명확히 제시해야 합니다.

이러한 태스크들은 대화형 오디오 콘텐츠의 특성을 반영하여 설계되었으며, 기존의 텍스트 기반 사실 확인 시스템과는 다른 접근 방식을 요구합니다.

---




This paper proposes a new approach to fact-checking in audio platforms, particularly focusing on the verification of misinformation in conversational audio content. Here, we will explain using the example of the dataset MAD2.

#### Dataset Composition
The MAD2 dataset consists of approximately 1,000 dialogues, containing 8,192 sentences and 3,368 check-worthy claims. Each claim is annotated with a binary true/false label. The dataset has the following structure:

- **Input**: The entire text of each dialogue along with the corresponding audio files. Each sentence is provided with time-based timestamps, allowing the model to understand the context of specific claims.
- **Output**: The truth value (true/false) for each claim and information needed to find evidence supporting that claim. The model must consider the context of the dialogue to verify the claims, where the flow of conversation and the roles of speakers are crucial.

#### Task
The main tasks performed using the MAD2 dataset are as follows:

1. **Claim Detection**: The task of identifying check-worthy claims within the dialogue. In this stage, sentences are not processed independently; instead, the flow of conversation and the roles of speakers are considered to detect claims.

2. **Evidence Retrieval**: The task of finding evidence for the detected claims. This process requires time-based retrieval to locate evidence supporting the claims from specific parts of the dialogue.

3. **Claim Verification**: The task of determining the truthfulness of claims based on the collected evidence. In this stage, multiple turns and temporal context are considered to evaluate the claims.

4. **Explanation Generation**: The task of generating explanations in a user-understandable format based on the verification results. This process must clearly present the evidence related to the truth value of the claims.

These tasks are designed to reflect the characteristics of conversational audio content and require a different approach compared to traditional text-based fact-checking systems.

<br/>
# 요약
이 논문에서는 오디오 플랫폼에서의 허위 정보 검증을 위한 새로운 접근 방식을 제안합니다. 연구진은 대화형 오디오의 특성을 반영한 데이터셋과 방법론을 개발하고, 기존의 텍스트 중심 검증 파이프라인이 오디오 콘텐츠에 적합하지 않음을 보여줍니다. 예를 들어, 대화의 맥락과 음성의 억양이 정보의 신뢰성에 미치는 영향을 분석하여, 오디오 기반 검증 시스템의 필요성을 강조합니다.

---

This paper proposes a new approach to fact-checking misinformation on audio platforms. The authors develop datasets and methodologies that reflect the characteristics of conversational audio, demonstrating that existing text-centric verification pipelines are inadequate for audio content. For instance, they analyze how the context of dialogue and vocal prosody affect the perceived credibility of information, emphasizing the need for audio-based verification systems.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: COVID-19에 대한 교차 에피소드 허위 정보의 타임라인을 보여줍니다. 이 피규어는 허위 정보가 시간이 지남에 따라 어떻게 축적되고 반복되는지를 시각적으로 나타내며, 특정 주제에 대한 허위 주장이 여러 에피소드에 걸쳐 어떻게 강화되는지를 강조합니다. 이는 청취자에게 허위 정보의 지속성과 신뢰성을 높이는 데 기여할 수 있음을 시사합니다.
   - **Figure 2**: 팟캐스트에서의 주장 탐지 및 검증 작업을 설명합니다. 이 피규어는 전통적인 텍스트 기반 검증 시스템이 어떻게 다중 턴 대화의 복잡성을 처리하는 데 어려움을 겪는지를 보여줍니다. 이는 오디오 플랫폼에서의 주장 검증이 단순한 텍스트 검증과는 다르다는 점을 강조합니다.

2. **테이블**
   - **Table 1**: 오디오 허위 정보의 구술적 및 대화적 특성과 이들이 사실 확인에 미치는 도전 과제를 정리합니다. 이 표는 오디오 미디어의 고유한 특성이 어떻게 사실 확인 프로세스를 복잡하게 만드는지를 보여줍니다. 예를 들어, 음성의 억양, 감정, 대화의 흐름 등이 허위 정보의 수용에 영향을 미친다는 점을 강조합니다.
   - **Table 2**: MAD2 데이터셋에서의 F1 점수와 AUC를 비교하여, 대화의 맥락이 주장 검증 성능에 미치는 영향을 보여줍니다. 이 표는 대화의 순서와 맥락이 주장 탐지의 정확성에 중요한 역할을 한다는 것을 시사합니다.
   - **Table 3**: 다양한 데이터셋의 오디오, 대화, 전사 및 사실성 차원을 정리합니다. 이 표는 현재의 데이터셋들이 어떻게 제한적이며, 오디오 기반 사실 확인 연구의 필요성을 강조합니다.
   - **Table 4**: 전통적인 사실 확인 파이프라인의 각 구성 요소가 구술 대화에서 어떻게 작동하는지를 요약합니다. 이 표는 각 단계에서의 한계와 구술 대화에 적합한 설계 요구 사항을 제시합니다.

3. **어펜딕스**
   - 어펜딕스는 연구에서 사용된 데이터셋, 실험 설정 및 추가적인 분석 결과를 포함할 수 있습니다. 이는 연구의 투명성을 높이고, 다른 연구자들이 동일한 방법론을 재현할 수 있도록 돕습니다.

---

### Insights and Results from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates the timeline of cross-episode misinformation regarding COVID-19. It visually represents how misinformation accumulates over time, highlighting how specific false claims are reinforced across multiple episodes. This suggests that the persistence and credibility of misinformation can increase for listeners.
   - **Figure 2**: This figure explains the tasks of claim detection and verification in podcasts. It shows how traditional text-based verification systems struggle to handle the complexities of multi-turn dialogue. This emphasizes that claim verification in audio platforms is fundamentally different from simple text verification.

2. **Tables**
   - **Table 1**: This table summarizes the spoken and conversational properties of audio misinformation and the distinct challenges they introduce for fact-checking. It highlights how unique features of audio media complicate the fact-checking process, such as the influence of prosody, emotion, and the flow of conversation on the reception of misinformation.
   - **Table 2**: This table compares F1 scores and AUC from the MAD2 dataset, demonstrating the impact of conversational context on claim verification performance. It suggests that the order and context of dialogue play a crucial role in the accuracy of claim detection.
   - **Table 3**: This table summarizes various datasets across audio, dialogue, transcripts, and factuality dimensions. It highlights the limitations of current datasets and underscores the need for more comprehensive resources for audio-based fact-checking research.
   - **Table 4**: This table summarizes how traditional fact-checking pipeline components behave in spoken dialogue, outlining limitations and design requirements for each stage. It provides insights into how existing systems can be improved to better accommodate the nuances of spoken dialogue.

3. **Appendices**
   - The appendices may include datasets used in the research, experimental setups, and additional analysis results. This enhances the transparency of the research and helps other researchers replicate the methodologies used.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{chun2026when,
  author = {Chaewan Chun and Delvin Ce Zhang and Dongwon Lee},
  title = {When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {2060--2075},
  year = {2026},
  month = {July},
  publisher = {Association for Computational Linguistics},
  address = {USA}
}
```

### 시카고 스타일 인용
Chaewan Chun, Delvin Ce Zhang, and Dongwon Lee. 2026. "When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2060–2075. USA: Association for Computational Linguistics.
    