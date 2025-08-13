---
layout: post
title:  "[2025]SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script"
date:   2025-08-13 15:53:51 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

SHARE 데이터셋은 영화 대본에서 추출된 대화 데이터를 기반으로 하며, 두 참가자 간의 공유된 경험을 포함  
EPISODE 프레임워크는 대화 시스템의 자연스러운 대화를 촉진하기 위한 새로운 접근 방식(응답 생성 절차=기억 선택, 응답 생성, 기억 관리 절차=정보 추출 각 대화 세션이 끝날 때마다, 기억 업데이트)  

짧은 요약(Abstract) :


이 연구는 두 사람 간의 공유된 기억이 그들의 관계를 강화하고 대화를 지속시키는 데 중요한 역할을 한다는 점에 주목합니다. 연구의 목표는 이러한 공유된 기억을 활용하여 장기적인 대화를 더 흥미롭게 만드는 것입니다. 이를 위해 영화 대본에서 구축된 새로운 장기 대화 데이터셋인 SHARE를 소개합니다. 이 데이터셋은 대화에서 명시적으로 드러나는 두 사람의 인물 정보와 사건 요약뿐만 아니라 암묵적으로 추출 가능한 공유된 기억도 포함하고 있습니다. 또한 SHARE를 기반으로 한 장기 대화 프레임워크인 EPISODE를 소개하며, 이는 두 사람 간의 공유된 경험을 활용합니다. SHARE를 사용한 실험을 통해 두 사람 간의 공유된 기억이 장기 대화를 더 흥미롭고 지속 가능하게 만든다는 것을 입증하였으며, EPISODE가 대화 중 공유된 기억을 효과적으로 관리한다는 것을 보여줍니다. 데이터셋과 코드는 https://github.com/e1kim/SHARE에서 이용할 수 있습니다.



Shared memories between two individuals strengthen their bond and are crucial for facilitating their ongoing conversations. This study aims to make long-term dialogue more engaging by leveraging these shared memories. To this end, we introduce a new long-term dialogue dataset named SHARE, constructed from movie scripts, which are a rich source of shared memories among various relationships. Our dialogue dataset contains the summaries of persona information and events of two individuals, as explicitly revealed in their conversation, along with implicitly extractable shared memories. We also introduce EPISODE, a long-term dialogue framework based on SHARE that utilizes shared experiences between individuals. Through experiments using SHARE, we demonstrate that shared memories between two individuals make long-term dialogues more engaging and sustainable, and that EPISODE effectively manages shared memories during dialogue. Our dataset and code are available at https://github.com/e1kim/SHARE.


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


이 연구에서는 장기 대화 시스템을 개선하기 위해 SHARE라는 새로운 데이터셋과 EPISODE라는 대화 프레임워크를 제안합니다. SHARE 데이터셋은 영화 대본에서 추출된 대화 데이터를 기반으로 하며, 두 참가자 간의 공유된 경험을 포함하고 있습니다. 이 데이터셋은 장기 대화 시스템 개발에 중요한 기초를 제공합니다.

EPISODE 프레임워크는 대화 시스템의 자연스러운 대화를 촉진하기 위한 새로운 접근 방식입니다. 이 프레임워크는 공유된 기억을 대화 시스템에 통합하여 반영성과 일관성을 개선하고, 더 풍부하고 일관된 상호작용을 이끌어냅니다. EPISODE는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **응답 생성 절차**: 
   - **기억 선택**: 현재 대화의 맥락에 따라 적절한 기억을 선택하여 대화 생성에 활용합니다. 이는 이전 대화에서의 기억을 활용하여 장기 대화를 지원합니다.
   - **응답 생성**: 선택된 기억을 활용하여 대화 모델이 적절한 응답을 생성합니다. 이를 통해 대화 모델은 기억을 반영하여 개인화되고 일관된 대화를 생성할 수 있습니다.

2. **기억 관리 절차**:
   - **정보 추출**: 각 대화 세션이 끝날 때마다, 향후 상호작용에 유용할 수 있는 독특하고 기억에 남는 세부 정보를 식별합니다. 추출된 정보는 이후 세션에 통합됩니다.
   - **기억 업데이트**: 대화가 진행됨에 따라 기억을 지속적으로 정제하고 관리합니다. 이는 대화의 질을 유지하기 위해 중요합니다.

EPISODE 프레임워크는 GPT-4o를 활용한 새로운 평가 방법을 사용하여 전통적인 메트릭을 넘어 다양한 평가 기준을 통합합니다. 공유된 기억을 대화 시스템에 통합함으로써 반영성과 일관성이 개선되어 더 풍부하고 일관된 상호작용을 이끌어냅니다.



In this study, we propose a new dataset called SHARE and a dialogue framework named EPISODE to enhance long-term dialogue systems. The SHARE dataset is based on dialogue data extracted from movie scripts and includes shared experiences between two participants. This dataset provides a critical foundation for the development of long-term dialogue systems.

The EPISODE framework is a novel approach designed to promote more natural conversations in dialogue systems. By integrating shared memory into the dialogue system, EPISODE improves reflectiveness and consistency, leading to richer and more coherent interactions. The EPISODE framework consists of the following key components:

1. **Response Generation Procedure**:
   - **Memory Selection**: Selects appropriate memories based on the current context to be used in dialogue generation. This supports long-term dialogue by utilizing memories from previous conversations.
   - **Response Generation**: The dialogue model generates appropriate responses using the selected memories. This allows the dialogue model to produce personalized and consistent dialogues by reflecting the memories.

2. **Memory Management Procedure**:
   - **Information Extraction**: At the end of each dialogue session, distinct and memorable details that may be useful for future interactions are identified. The extracted information is then incorporated into subsequent sessions.
   - **Memory Update**: As the dialogue progresses, the memory is continuously refined and managed. This is crucial for maintaining the quality of the dialogue.

The EPISODE framework employs a novel evaluation method using GPT-4o, extending beyond traditional metrics to incorporate diverse evaluation criteria. By integrating shared memory into the dialogue system, improvements in both reflectiveness and consistency are demonstrated, leading to richer and more coherent interactions.


<br/>
# Results


#### 실험 결과

이 연구에서는 SHARE 데이터셋을 사용하여 다양한 대화 모델의 성능을 평가하였습니다. 실험은 자동 평가와 다중 세션 평가로 나누어 진행되었습니다.

##### 자동 평가

자동 평가는 각 세션의 마지막 발화에 대해 수행되었으며, BLEU-3/4, ROUGE (ROUGE-1/2/L), BERTscore, Distinct-1/2, PPL 등의 다양한 메트릭을 사용하여 모델의 성능을 평가하였습니다. 각 백본 모델에 대해 다음과 같은 세 가지 버전의 대화 생성 모델을 비교하였습니다.

1. **SHARE w/o memory**: SHARE 데이터셋의 대화만을 사용하여 훈련된 모델로, 메모리를 사용하지 않습니다.
2. **SHARE w/ predicted memory**: SHARE 데이터셋의 대화와 주석된 메모리를 사용하여 훈련된 모델로, 대화 컨텍스트에 따라 메모리를 선택하여 응답을 생성합니다.
3. **SHARE w/ predicted individual memory**: 각 화자가 독립적으로 메모리를 관리하여 응답을 생성하는 베이스라인 모델입니다.
4. **SHARE w/ gold memory**: 실제 메모리를 사용하여 응답을 생성하는 모델입니다.

결과적으로, SHARE 데이터셋을 사용한 훈련은 다음 응답을 생성하는 데 있어 대부분의 성능 메트릭을 향상시켰습니다. 특히, Llama-3 모델에서 SHARE w/ predicted memory는 BLEU-3/4 및 ROUGE 점수에서 더 높은 성능을 보였으며, 이는 SHARE 데이터셋, 특히 공유 메모리를 사용하는 것이 대화 생성에 있어 더 나은 성능을 발휘함을 나타냅니다. 또한, SHARE w/ predicted memory는 SHARE w/ predicted individual memory보다 더 다양한 대화를 생성하여 대화의 풍부함을 증가시켰습니다.

##### 다중 세션 평가

다중 세션 평가는 세션 수가 증가함에 따라 업데이트 모델이 메모리를 효율적으로 조직하는지를 평가하였습니다. GPT 기반 평가 방법을 사용하여, Coherence, Engagingness, Closeness, Reflectiveness의 네 가지 기준에 따라 평가하였습니다. 각 기준은 0에서 3점까지의 척도로 평가되었습니다.

SHARE +EPISODE 모델은 Reflectiveness에서 일관되게 다른 모델보다 우수한 성능을 보였습니다. Llama-3 모델은 초기 세션에서는 Closeness 점수가 낮았으나, 이후 세션에서 성능이 향상되어 Accumulate 모델을 능가하였습니다. 이는 공유 경험을 효과적으로 활용하고 참가자 간의 관계를 명확하게 묘사하는 데 성공했음을 보여줍니다.

##### EPISODE 평가

EPISODE 모델은 전체 세션을 함께 평가하였으며, Consistency, Reflectiveness, Engagingness의 세 가지 기준에 따라 평가되었습니다. 결과적으로, 공유 메모리를 적용한 모델은 그렇지 않은 모델에 비해 더 나은 성능을 보였습니다. 특히, Llama-3와 Gemma 모델에서 SHARE +EPISODE는 Reflectiveness와 Consistency에서 더 높은 점수를 기록하였습니다. 이는 공유 메모리가 두 참가자 간의 관계를 더 잘 이해하게 하여 대화를 더 일관되게 만든다는 것을 나타냅니다.



#### Experimental Results

In this study, the performance of various dialogue models was evaluated using the SHARE dataset. The experiments were divided into automatic evaluation and multi-session evaluation.

##### Automatic Evaluation

The automatic evaluation was conducted on the final utterance of each session, using a comprehensive set of metrics including BLEU-3/4, ROUGE (ROUGE-1/2/L), BERTscore, Distinct-1/2, and PPL to assess the model's performance. For each backbone model, the following three versions of dialogue generation were compared:

1. **SHARE w/o memory**: A model trained only with the dialogue from the SHARE dataset without using memories.
2. **SHARE w/ predicted memory**: A model trained with both the dialogue and annotated memories from the SHARE dataset, generating responses based on memory selection for the given dialogue context.
3. **SHARE w/ predicted individual memory**: A baseline where each speaker independently manages their memory for response generation.
4. **SHARE w/ gold memory**: A model that uses the actual memory to generate responses.

As a result, training with the SHARE dataset improved most performance metrics for generating the next response. In the Llama-3 model, SHARE w/ predicted memory showed higher BLEU-3/4 and ROUGE scores, indicating better performance in dialogue generation, especially when using the SHARE dataset and shared memory. Additionally, SHARE w/ predicted memory outperformed SHARE w/ predicted individual memory, demonstrating that incorporating the interlocutor’s persona and personal event information leads to more diverse and enriched conversations.

##### Multi-Session Evaluation

The multi-session evaluation assessed how efficiently the updated model organizes memory as the number of sessions increases. Using a GPT-based evaluation approach, the model was evaluated on four criteria: Coherence, Engagingness, Closeness, and Reflectiveness, each scored on a scale from 0 to 3 points.

The SHARE +EPISODE model consistently outperformed others in Reflectiveness. Although the Llama-3 model showed lower scores in Closeness in the initial session, its performance improved in subsequent sessions, eventually surpassing the Accumulate model. This highlights the model’s effective use of shared experiences and clear depiction of the participants’ relationship across multiple sessions.

##### EPISODE Evaluation

The EPISODE model was evaluated on the entire set of sessions together, based on three criteria: Consistency, Reflectiveness, and Engagingness. The results indicated that applying shared memory in each model led to better performance compared to when it was not used. In both the Llama-3 and Gemma models, SHARE +EPISODE achieved higher scores in Reflectiveness and Consistency compared to SHARE (w/o shared memory). This result indicates that shared memory enhances the model’s understanding of the relationship between the two participants, leading to more consistent dialogues.


<br/>
# 예제


#### 예시 설명

이 연구에서는 영화 대본에서 추출한 대화 데이터를 기반으로 한 장기 대화 데이터셋인 SHARE를 소개합니다. 이 데이터셋은 두 참가자 간의 공유 경험을 포함하여 장기 대화 시스템 개발에 중요한 기초를 제공합니다. SHARE 데이터셋을 사용하여 대화 모델을 훈련하고, EPISODE라는 새로운 장기 대화 프레임워크를 제안합니다. 이 프레임워크는 더 자연스러운 대화를 촉진합니다.

#### 트레이닝 데이터

- **인풋**: 영화 대본에서 추출한 대화
  - 대화는 두 명의 참가자 간의 여러 세션으로 구성됩니다.
  - 각 세션은 여러 발화로 구성되며, 각 발화는 대화의 맥락과 관련된 정보(예: 인물 정보, 개인 이벤트, 공유 메모리 등)를 포함합니다.

- **아웃풋**: 대화 모델이 생성한 응답
  - 모델은 주어진 대화 맥락과 메모리 세트를 기반으로 다음 발화를 생성합니다.
  - 생성된 응답은 대화의 일관성, 참여도, 반영성을 평가하는 데 사용됩니다.

#### 테스트 데이터

- **인풋**: 새로운 대화 세션
  - 테스트 데이터는 훈련 데이터와 유사한 형식으로 구성되며, 모델이 학습한 내용을 평가하는 데 사용됩니다.

- **아웃풋**: 모델의 성능 평가
  - BLEU, ROUGE, BERTscore와 같은 자동 평가 지표를 사용하여 모델의 성능을 평가합니다.
  - GPT-4o를 사용하여 대화의 일관성, 참여도, 반영성을 평가합니다.

#### 구체적인 테스크

1. **메모리 선택**: 대화 맥락에 따라 적절한 메모리를 선택합니다.
2. **응답 생성**: 선택된 메모리를 사용하여 다음 발화를 생성합니다.
3. **메모리 관리**: 세션이 끝난 후, 새로운 정보를 추출하고 기존 메모리를 업데이트합니다.

---



#### Example Description

In this study, we introduce SHARE, a long-term dialogue dataset extracted from movie scripts. This dataset provides a critical foundation for the development of long-term dialogue systems, including shared experiences between two participants. Using the SHARE dataset, we train dialogue models and propose a new long-term dialogue framework called EPISODE, which promotes more natural conversations.

#### Training Data

- **Input**: Dialogue extracted from movie scripts
  - The dialogue consists of multiple sessions between two participants.
  - Each session comprises multiple utterances, each containing information related to the dialogue context (e.g., persona information, personal events, shared memory).

- **Output**: Responses generated by the dialogue model
  - The model generates the next utterance based on the given dialogue context and memory set.
  - The generated responses are used to evaluate the coherence, engagement, and reflectiveness of the dialogue.

#### Test Data

- **Input**: New dialogue sessions
  - The test data is structured similarly to the training data and is used to evaluate the model's learned capabilities.

- **Output**: Model performance evaluation
  - The model's performance is evaluated using automatic metrics such as BLEU, ROUGE, and BERTscore.
  - GPT-4o is used to assess the coherence, engagement, and reflectiveness of the dialogue.

#### Specific Tasks

1. **Memory Selection**: Select appropriate memories based on the dialogue context.
2. **Response Generation**: Generate the next utterance using the selected memories.
3. **Memory Management**: After the session ends, extract new information and update existing memories.

<br/>
# 요약

이 연구에서는 영화 대본에서 추출한 공유 기억을 포함한 장기 대화 데이터셋인 SHARE를 소개하고, 이를 활용한 장기 대화 프레임워크 EPISODE를 제안합니다. 실험 결과, 공유 기억을 활용한 대화가 더 일관되고 흥미로우며, 참여자 간의 관계를 잘 반영하는 것으로 나타났습니다. 예시로, 공유 기억을 사용한 대화는 참여자 간의 과거 경험을 자연스럽게 통합하여 대화를 더욱 풍부하고 매력적으로 만듭니다.


This study introduces SHARE, a long-term dialogue dataset incorporating shared memories extracted from movie scripts, and proposes the EPISODE framework to utilize it. Experimental results show that dialogues using shared memories are more coherent, engaging, and reflective of the participants' relationship. For example, dialogues with shared memories naturally integrate past experiences between participants, making the conversation richer and more engaging.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: 이 다이어그램은 두 사람 간의 대화에서 공유된 기억이 어떻게 대화의 질을 향상시키는지를 보여줍니다. 공유된 기억을 활용한 대화는 더 깊이 있는 관계를 반영하고, 대화를 더 흥미롭게 만듭니다.
   - **Figure 2**: EPISODE 프레임워크의 아키텍처를 보여줍니다. 이 프레임워크는 대화 중에 공유된 기억을 선택하고, 이를 기반으로 응답을 생성하며, 세션이 끝난 후에는 기억을 관리합니다.
   - **Figure 3**: 공유된 기억을 사용한 대화와 사용하지 않은 대화의 예시를 보여줍니다. 공유된 기억을 사용한 대화는 더 깊이 있는 관계와 감정을 드러내며, 대화를 더 몰입감 있게 만듭니다.

2. **테이블**
   - **Table 1**: 다양한 장기 대화 데이터셋을 비교합니다. SHARE 데이터셋은 다른 데이터셋과 달리 공유된 기억을 포함하고 있어, 두 사람 간의 깊이 있는 대화를 가능하게 합니다.
   - **Table 2**: SHARE 데이터셋의 통계 정보를 제공합니다. 이 데이터셋은 다양한 장르의 영화 대본에서 추출된 대화로 구성되어 있으며, 많은 에피소드가 공유된 기억을 포함하고 있습니다.
   - **Table 3**: 자동 평가 메트릭 결과를 비교합니다. SHARE 데이터셋을 사용한 모델은 더 나은 성능을 보이며, 특히 공유된 기억을 활용한 대화 생성에서 우수한 결과를 보여줍니다.
   - **Table 4**: GPT-4o를 사용한 다중 세션 평가 결과를 보여줍니다. EPISODE 프레임워크는 대화의 일관성, 흥미로움, 관계 반영 측면에서 우수한 성능을 보입니다.
   - **Table 5**: 전체 대화 세션을 평가한 결과를 보여줍니다. 공유된 기억을 활용한 대화는 일관성과 반영성 측면에서 더 나은 성능을 보입니다.

3. **어펜딕스**
   - **Appendix A.1**: 데이터셋 수집 과정과 장르 분포를 설명합니다. 다양한 장르의 영화 대본에서 데이터를 수집하여, 다양한 대화 스타일을 반영할 수 있습니다.
   - **Appendix A.2**: 구현 및 훈련 세부 사항을 설명합니다. 각 모델의 훈련 설정과 소요 시간을 제공합니다.
   - **Appendix A.6**: 실험 설정과 평가 방법을 설명합니다. GPT Eval을 사용하여 대화의 일관성, 흥미로움, 반영성을 평가합니다.

---



1. **Diagrams and Figures**
   - **Figure 1**: This diagram illustrates how shared memories between two individuals enhance the quality of dialogue. Conversations utilizing shared memories reflect deeper relationships and make the dialogue more engaging.
   - **Figure 2**: Shows the architecture of the EPISODE framework. This framework selects shared memories during dialogue, generates responses based on them, and manages memories after the session ends.
   - **Figure 3**: Provides examples of dialogues with and without shared memory. Dialogues using shared memory reveal deeper relationships and emotions, making the conversation more immersive.

2. **Tables**
   - **Table 1**: Compares various long-term dialogue datasets. The SHARE dataset uniquely includes shared memories, enabling deeper conversations between two individuals.
   - **Table 2**: Provides statistical information about the SHARE dataset. It consists of dialogues extracted from movie scripts of various genres, with many episodes containing shared memories.
   - **Table 3**: Compares automatic evaluation metric results. Models trained with the SHARE dataset show better performance, especially in generating dialogues using shared memory.
   - **Table 4**: Shows multi-session evaluation results using GPT-4o. The EPISODE framework demonstrates superior performance in terms of coherence, engagingness, and reflectiveness of relationships.
   - **Table 5**: Evaluates the entire conversation sessions. Dialogues utilizing shared memory show better performance in consistency and reflectiveness.

3. **Appendix**
   - **Appendix A.1**: Describes the dataset collection process and genre distribution. Data collected from movie scripts of various genres allows for the reflection of diverse dialogue styles.
   - **Appendix A.2**: Details implementation and training specifics. Provides training settings and time for each model.
   - **Appendix A.6**: Describes experimental setup and evaluation methods. Uses GPT Eval to assess coherence, engagingness, and reflectiveness of dialogues.

<br/>
# refer format:


**BibTeX:**
```bibtex
@inproceedings{kim2025share,
  title={SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script},
  author={Eunwon Kim and Chanho Park and Buru Chang},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={14474--14498},
  year={2025},
  organization={Association for Computational Linguistics}
}
```


Kim, Eunwon, Chanho Park, and Buru Chang. 2025. "SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 14474-14498. Association for Computational Linguistics.
