---
layout: post
title:  "[2026]MPCEval: A Benchmark for Multi-Party Conversation Generation"
date:   2026-06-20 14:46:33 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

Ask the right question to the right people at the right time.  
올바른 질문을 올바른 사람에게 올바른 타이밍에 하는지를 체크!  


짧은 요약(Abstract) :


이 논문에서는 다자간 대화 생성의 평가 방법론인 MPCEval을 소개합니다. 다자간 대화는 스마트 답변 및 협업 도우미와 같은 생성 AI의 중요한 기능이지만, 그 평가 방법은 여전히 큰 병목 현상입니다. 다자간 대화는 복잡한 턴 교환, 역할에 따른 화자 행동, 긴 대화 구조, 여러 개의 유효한 연속성 등 두 사람 간의 대화와는 다른 도전 과제를 포함합니다. MPCEval은 생성 품질을 화자 모델링, 콘텐츠 품질, 화자-콘텐츠 일관성으로 분해하고, 지역적인 다음 턴 예측과 전체 대화 생성을 명확히 구분합니다. 이 평가 프레임워크는 데이터셋과 모델에 걸쳐 확장 가능한 새로운 정량적, 참조 없는, 재현 가능한 메트릭을 제공합니다. MPCEval을 다양한 공개 및 실제 데이터셋에 적용하여 현대 생성 방법과 인간이 작성한 대화를 평가한 결과, 참여 균형, 콘텐츠 진행 및 참신성, 화자-콘텐츠 일관성에서 체계적이고 차원별 모델 특성이 드러났습니다. 이러한 결과는 평가 목표가 모델 평가에 중요한 영향을 미친다는 것을 보여주며, 단일 점수 평가가 다자간 대화 행동의 근본적인 차이를 가리는 경향이 있음을 나타냅니다.



This paper introduces MPCEval, a benchmarking and evaluation methodology for multi-party conversation generation. Multi-party conversation, such as smart replies and collaborative assistants, is an increasingly important capability of generative AI, yet its evaluation remains a critical bottleneck. Multi-party settings introduce distinct challenges compared to two-party dialogue, including complex turn-taking, role-dependent speaker behavior, long-range conversational structure, and multiple equally valid continuations. MPCEval decomposes generation quality into speaker modeling, content quality, and speaker-content consistency, explicitly distinguishing local next-turn prediction from global full-conversation generation. It provides novel, quantitative, reference-free, and reproducible metrics that scale across datasets and models. Applying MPCEval to diverse public and real-world datasets reveals systematic, dimension-specific model characteristics in participation balance, content progression and novelty, and speaker-content consistency, demonstrating that evaluation objectives critically shape model assessment and that single-score evaluation obscures fundamental differences in multi-party conversational behavior.


* Useful sentences :

DeliData: 사람들이 온라인에서 함께 협업하며 퍼즐이나 문제를 해결하는 과정을 기록한 실제 대화 데이터입니다.

MPDD: TV 드라마, 영화 스크립트 등 대사 작가들이 작성한 다자간 대화 데이터셋입니다.  

이거 두가지가 대화 소스임  


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology
  

**MPCEval: 다자간 대화 생성 평가를 위한 벤치마크**

MPCEval은 다자간 대화 생성의 평가를 위한 새로운 프레임워크로, 대화 생성의 품질을 세 가지 주요 차원으로 분해하여 측정합니다: 화자 모델링, 콘텐츠 품질, 화자-콘텐츠 일관성. 이 프레임워크는 다음과 같은 두 가지 주요 평가 작업을 구분합니다: 다음 메시지 예측(로컬 평가)과 전체 대화 생성(글로벌 평가).

1. **화자 모델링**: 이 차원은 다음 화자가 대화의 맥락에서 적절한지를 평가합니다. MPCEval은 화자 선택의 여러 신호를 고려하여, 화자가 최근에 얼마나 자주 참여했는지, 주제와의 일치성 등을 평가합니다. 예를 들어, 화자가 명시적으로 언급되었는지, 최근에 대화에 참여했는지, 주제와 얼마나 잘 맞는지를 측정합니다.

2. **콘텐츠 품질**: 이 차원은 생성된 메시지가 대화의 흐름에 얼마나 적절하게 기여하는지를 평가합니다. MPCEval은 메시지의 참신함, 일관성, 그리고 대화의 목표에 대한 기여도를 측정합니다. 예를 들어, 메시지가 이전 대화와 얼마나 중복되는지, 주제를 얼마나 잘 확장하는지를 평가합니다.

3. **화자-콘텐츠 일관성**: 이 차원은 생성된 메시지가 예측된 화자의 과거 발언과 얼마나 일관성이 있는지를 평가합니다. 이는 화자의 역할이나 스타일에 따라 메시지가 적절한지를 판단하는 데 도움을 줍니다.

MPCEval은 이러한 세 가지 차원을 통해 다자간 대화 생성의 복잡한 특성을 정량적으로 평가할 수 있는 새로운 메트릭을 제공합니다. 이 프레임워크는 기존의 평가 방법들이 간과했던 다자간 대화의 고유한 특성을 반영하여, 보다 정교하고 신뢰할 수 있는 평가를 가능하게 합니다.




**MPCEval: A Benchmark for Multi-Party Conversation Generation Evaluation**

MPCEval is a new framework for evaluating multi-party conversation generation, decomposing the quality of dialogue generation into three main dimensions: speaker modeling, content quality, and speaker-content consistency. This framework distinguishes between two primary evaluation tasks: next-message prediction (local evaluation) and full conversation generation (global evaluation).

1. **Speaker Modeling**: This dimension assesses whether the next speaker is appropriate within the context of the conversation. MPCEval considers various signals for speaker selection, evaluating how frequently the speaker has participated recently, and their alignment with the topic. For instance, it measures whether the speaker has been explicitly mentioned, participated recently, and how well they fit the current topic.

2. **Content Quality**: This dimension evaluates how appropriately the generated message contributes to the flow of the conversation. MPCEval measures the novelty, coherence, and contribution of the message to the conversation's goals. For example, it assesses how much the message overlaps with previous content and how well it expands the topic.

3. **Speaker-Content Consistency**: This dimension evaluates how consistent the generated message is with the predicted speaker's past utterances. It helps determine whether the message is appropriate based on the speaker's role or style.

Through these three dimensions, MPCEval provides a new set of metrics that quantitatively assess the complex characteristics of multi-party conversation generation. This framework enables a more nuanced and reliable evaluation that reflects the unique properties of multi-party dialogue, which have often been overlooked by existing evaluation methods.


<br/>
# Results



이 논문에서는 다자간 대화 생성의 평가를 위한 새로운 벤치마크인 MPCEval을 소개합니다. MPCEval은 대화 생성 품질을 세 가지 차원으로 분해하여 평가합니다: 화자 모델링, 콘텐츠 품질, 화자-콘텐츠 일관성. 이 평가 프레임워크는 로컬(next-message prediction)과 글로벌(full-conversation generation) 두 가지 평가 작업을 구분하여, 각 작업에 적합한 메트릭을 제공합니다.

#### 실험 결과 요약

1. **경쟁 모델**: MPCEval을 사용하여 여러 최신 대화 생성 모델을 평가했습니다. 여기에는 LLaMa-3.3, GPT-4-Turbo, DeepSeek, Claude-3.5, MultiLIGHT, ChatGPT-solver 등이 포함됩니다. 각 모델은 DeliData와 MPDD 데이터셋에서 평가되었습니다.

2. **테스트 데이터**: DeliData는 협업 문제 해결을 위한 대화 데이터셋으로, 500개의 대화가 포함되어 있으며, MPDD는 TV 스크립트에서 파생된 다자간 대화 데이터셋으로, 1,774개의 대화가 포함되어 있습니다.

3. **메트릭**: MPCEval은 다양한 메트릭을 사용하여 모델의 성능을 평가합니다. 예를 들어, DeliData에서의 로컬 화자 모델링 메트릭에서는 ChatGPT-solver가 가장 높은 참여 빈도(PF) 점수를 기록했으며, DeepSeek는 직접 이름 참조(DNR)에서 가장 높은 점수를 기록했습니다. 글로벌 평가에서는 MPC-constraints와 LLaMa-3.1 조합이 가장 높은 정규화된 화자 엔트로피(NSE) 점수를 기록했습니다.

4. **비교**: MPCEval의 메트릭은 기존의 참조 기반 메트릭(BLEU, ROUGE 등)과 비교하여 더 나은 진단 능력을 보여주었습니다. 예를 들어, BLEU는 생성된 응답이 단일 참조와 다를 경우 낮은 점수를 부여하는 반면, MPCEval은 대화의 맥락과 일관성을 고려하여 더 균형 잡힌 평가를 제공합니다.

5. **결론**: MPCEval은 다자간 대화 생성의 평가를 위한 강력한 도구로, 다양한 모델의 성능을 세밀하게 비교할 수 있는 능력을 제공합니다. 실험 결과는 인간이 작성한 대화가 항상 최상의 기준이 아니라는 점을 강조하며, MPCEval의 메트릭이 대화의 질을 보다 정확하게 평가할 수 있음을 보여줍니다.

---




This paper introduces MPCEval, a new benchmark for evaluating multi-party conversation generation. MPCEval decomposes the quality of conversation generation into three dimensions: speaker modeling, content quality, and speaker-content consistency. This evaluation framework distinguishes between local (next-message prediction) and global (full-conversation generation) tasks, providing metrics suitable for each task.

#### Summary of Experimental Results

1. **Competing Models**: Various state-of-the-art conversation generation models were evaluated using MPCEval. These include LLaMa-3.3, GPT-4-Turbo, DeepSeek, Claude-3.5, MultiLIGHT, and ChatGPT-solver. Each model was assessed on the DeliData and MPDD datasets.

2. **Test Data**: DeliData is a collaborative problem-solving dialogue dataset containing 500 dialogues, while MPDD is a multi-party dialogue dataset derived from TV scripts, containing 1,774 dialogues.

3. **Metrics**: MPCEval employs various metrics to evaluate model performance. For instance, in local speaker modeling metrics on DeliData, ChatGPT-solver achieved the highest participation frequency (PF) score, while DeepSeek recorded the highest direct name reference (DNR) score. In global evaluation, the combination of MPC-constraints and LLaMa-3.1 achieved the highest normalized speaker entropy (NSE) score.

4. **Comparison**: The metrics from MPCEval demonstrated better diagnostic power compared to existing reference-based metrics (such as BLEU and ROUGE). For example, BLEU assigns low scores when the generated response diverges from a single reference, whereas MPCEval considers the context and consistency of the conversation, providing a more balanced assessment.

5. **Conclusion**: MPCEval serves as a powerful tool for evaluating multi-party conversation generation, enabling fine-grained comparisons of various models' performances. The experimental results emphasize that human-authored conversations should not always be treated as the gold standard, highlighting that MPCEval's metrics can more accurately assess the quality of conversations.


<br/>
# 예제


**MPCEval: 다자간 대화 생성 평가 벤치마크**

MPCEval은 다자간 대화 생성을 위한 평가 및 벤치마크 프레임워크로, 대화 생성의 품질을 세 가지 주요 차원으로 나누어 평가합니다: 화자 모델링, 내용 품질, 화자-내용 일관성. 이 프레임워크는 다음과 같은 두 가지 주요 작업을 구분합니다:

1. **다음 메시지 예측 (Next-Message Prediction)**: 주어진 대화 이력을 바탕으로 다음 화자와 그 화자가 할 메시지를 예측하는 작업입니다. 이 작업에서는 화자의 선택이 적절한지, 생성된 메시지가 대화의 맥락에 맞는지를 평가합니다.

2. **전체 대화 생성 (Full-Conversation Generation)**: 주어진 주제나 조건에 따라 전체 대화를 생성하는 작업입니다. 이 작업에서는 대화의 전체적인 흐름, 참여 균형, 정보의 흐름 등을 평가합니다.

#### 예시

**트레이닝 데이터 (Training Data)**:
- 대화 이력: 
  ```
  Alice: "안녕하세요, 오늘 회의의 주제는 무엇인가요?"
  Bob: "오늘은 프로젝트 진행 상황에 대해 이야기해요."
  Chris: "좋아요, 그럼 각자 맡은 부분을 공유해볼까요?"
  ```
- 주어진 조건: "다음 화자는 Bob이며, 그가 할 메시지는 프로젝트의 특정 부분에 대한 업데이트입니다."

**테스트 데이터 (Test Data)**:
- 대화 이력:
  ```
  Alice: "우리는 다음 주까지 이 작업을 완료해야 해요."
  Bob: "네, 저는 이미 작업을 시작했어요."
  Chris: "그럼 각자 진행 상황을 공유해볼까요?"
  ```
- 예측해야 할 다음 메시지:
  - 예측된 화자: Bob
  - 예측된 메시지: "저는 디자인 부분을 맡고 있으며, 현재 50% 완료했습니다."

**출력 (Output)**:
- JSON 형식으로 출력:
  ```json
  [
    {
      "dialogue_id": "1",
      "next_speaker": "Bob",
      "next_message": "저는 디자인 부분을 맡고 있으며, 현재 50% 완료했습니다."
    }
  ]
  ```

---



**MPCEval: A Benchmark for Multi-Party Conversation Generation**

MPCEval is an evaluation and benchmarking framework for multi-party conversation generation that decomposes the quality of dialogue generation into three main dimensions: speaker modeling, content quality, and speaker-content consistency. This framework distinguishes between two primary tasks:

1. **Next-Message Prediction**: This task involves predicting the next speaker and the message that speaker would say based on the given conversation history. In this task, the evaluation focuses on whether the choice of speaker is appropriate and whether the generated message aligns with the context of the conversation.

2. **Full-Conversation Generation**: This task involves generating an entire conversation based on a given topic or constraints. The evaluation in this task emphasizes the overall flow of the conversation, participation balance, and information flow.

#### Example

**Training Data**:
- Conversation History: 
  ```
  Alice: "Hello, what is the topic of today's meeting?"
  Bob: "Today, we will discuss the progress of the project."
  Chris: "Great, shall we share our parts?"
  ```
- Given Condition: "The next speaker is Bob, and he will provide an update on a specific part of the project."

**Test Data**:
- Conversation History:
  ```
  Alice: "We need to complete this task by next week."
  Bob: "Yes, I have already started working on it."
  Chris: "Then shall we share our progress?"
  ```
- Next Message to Predict:
  - Predicted Speaker: Bob
  - Predicted Message: "I am in charge of the design part, and I am currently 50% done."

**Output**:
- Output in JSON format:
  ```json
  [
    {
      "dialogue_id": "1",
      "next_speaker": "Bob",
      "next_message": "I am in charge of the design part, and I am currently 50% done."
    }
  ]
  ```

<br/>
# 요약



MPCEval은 다자간 대화 생성을 평가하기 위한 새로운 벤치마크로, 발화자 모델링, 콘텐츠 품질, 발화자-콘텐츠 일관성을 세 가지 차원으로 나누어 평가한다. 이 프레임워크는 로컬 및 글로벌 평가를 통해 대화의 질을 정량적으로 측정하며, 다양한 생성 모델의 성능을 비교할 수 있는 참조 없는 메트릭을 제공한다. 실험 결과, MPCEval은 기존 평가 방법의 한계를 극복하고, 인간이 작성한 대화와 기계 생성 대화 간의 차이를 명확히 드러낸다.



MPCEval is a new benchmark for evaluating multi-party conversation generation, decomposing the assessment into three dimensions: speaker modeling, content quality, and speaker-content consistency. This framework quantitatively measures the quality of conversations through local and global evaluations, providing reference-free metrics for comparing the performance of various generation models. Experimental results demonstrate that MPCEval overcomes the limitations of existing evaluation methods and clearly highlights the differences between human-authored and machine-generated conversations.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **MPCEval 프레임워크 구조**: 다이어그램은 MPCEval의 구조를 시각적으로 나타내며, 세 가지 주요 구성 요소(컨텍스트, 평가 객체, 측정)를 강조합니다. 이 구조는 다차원 평가를 가능하게 하여 대화 생성의 질을 보다 정교하게 분석할 수 있도록 합니다.
   - **대화 흐름**: 대화의 흐름을 보여주는 피규어는 각 발화가 어떻게 이어지는지를 시각적으로 표현하여, 대화의 맥락과 흐름을 이해하는 데 도움을 줍니다.

2. **테이블**:
   - **성능 비교 테이블**: DeliData 및 MPDD 데이터셋에서 다양한 모델의 성능을 비교한 테이블은 각 모델의 지역적 및 전역적 평가 지표를 보여줍니다. 이 테이블은 각 모델이 특정 평가 지표에서 어떻게 성능을 발휘하는지를 명확하게 나타내어, 연구자들이 모델의 강점과 약점을 파악하는 데 유용합니다.
   - **인간-작성 대화 vs. 기계 생성 대화**: 인간-작성 대화와 기계 생성 대화의 비교 결과는 두 유형의 대화가 서로 다른 특성을 지니고 있음을 보여줍니다. 예를 들어, 인간-작성 대화는 더 높은 암시적 참조 점수를 가지지만, 기계 생성 대화는 더 높은 일관성과 주제 적합성을 보입니다.

3. **어펜딕스**:
   - **데이터셋 설명**: 어펜딕스에서는 DeliData, MPDD, Tanka 데이터셋의 특성과 평가 작업을 상세히 설명합니다. 각 데이터셋의 구조와 목적을 이해하는 것은 MPCEval의 적용 가능성을 높이는 데 중요합니다.
   - **세부 평가 메트릭**: 각 평가 메트릭의 정의와 계산 방법을 설명하는 섹션은 연구자들이 MPCEval을 활용하여 대화 생성 모델을 평가할 때 필요한 정보를 제공합니다.

### Insights from Diagrams, Figures, Tables, and Appendices

1. **Diagrams and Figures**:
   - **Structure of MPCEval Framework**: The diagram visually represents the structure of MPCEval, highlighting its three main components (context, evaluation objects, measures). This structure enables multi-dimensional evaluation, allowing for a more nuanced analysis of conversation generation quality.
   - **Conversation Flow**: Figures illustrating the flow of conversation provide a visual representation of how each utterance connects, aiding in the understanding of context and flow in dialogues.

2. **Tables**:
   - **Performance Comparison Tables**: Tables comparing the performance of various models on the DeliData and MPDD datasets present local and global evaluation metrics for each model. These tables clearly indicate how each model performs on specific evaluation metrics, helping researchers identify strengths and weaknesses.
   - **Human vs. Machine-Generated Conversations**: The comparison results between human-authored and machine-generated conversations show that the two types exhibit different characteristics. For instance, human-authored conversations tend to have higher implicit reference scores, while machine-generated dialogues demonstrate better consistency and topic alignment.

3. **Appendices**:
   - **Dataset Descriptions**: The appendices provide detailed descriptions of the DeliData, MPDD, and Tanka datasets, outlining their characteristics and intended evaluation tasks. Understanding the structure and purpose of each dataset is crucial for enhancing the applicability of MPCEval.
   - **Detailed Evaluation Metrics**: Sections explaining the definitions and calculation methods for each evaluation metric offer essential information for researchers looking to utilize MPCEval for assessing conversation generation models.

<br/>
# refer format:



```bibtex
@article{zhang2026mpceval,
  title={MPCEval: A Benchmark for Multi-Party Conversation Generation},
  author={Zhang, Minxing and Yang, Yi and Jia, Zhuofan and Yang, Xuan and Pei, Jian and Zang, Yuchen and Deng, Xingwang and Chen, Xianglong},
  journal={arXiv preprint arXiv:2603.04969},
  year={2026},
  url={https://github.com/Owen-Yang-18/MPCEval}
}
```




Minxing Zhang, Yi Yang, Zhuofan Jia, Xuan Yang, Jian Pei, Yuchen Zang, Xingwang Deng, and Xianglong Chen. "MPCEval: A Benchmark for Multi-Party Conversation Generation." arXiv preprint arXiv:2603.04969 (2026). https://github.com/Owen-Yang-18/MPCEval.
