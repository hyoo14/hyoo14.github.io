---
layout: post
title:  "[2026]Confidence Estimation for LLMs in Multi-turn Interactions"
date:   2026-07-17 02:52:54 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 다중 턴 대화에서 대형 언어 모델(LLM)의 신뢰도 추정 방법을 체계적으로 연구하고, 정보가 축적됨에 따라 신뢰도가 증가해야 한다는 두 가지 주요 요구사항(보정 및 단조성)을 설정하였다.

길이 정규화된 기대 보정 오류(InfoECE)와 통제된 평가 데이터셋 생성을 위한 "Hinter-Guesser" 패러다임을 도입  
**Verbalized Confidence**: 모델이 스스로의 답변에 대한 신뢰도를 수치로 보고하도록 하는 방법  
**Self-Consistency (SC)**: 모델이 여러 번 답변을 생성하고, 이 중 가장 일관된 답변을 바탕으로 신뢰도를 계산하는 방법  
**Logit-based Probes**: 모델의 내부 신호를 활용하여 신뢰도를 추정하는 방법입니다. 특히, P(TRUE)와 P(SUFFICIENT)라는 두 가지 방법이 사용되며, P(SUFFICIENT)는 주어진 정보가 특정 답변을 지지하는 데 충분한지를 평가  

**평가 메트릭**: 연구에서는 신뢰도 추정의 정확성을 평가하기 위해 두 가지 주요 메트릭을 사용합니다. 첫째, **Expected Calibration Error (ECE)**는 모델의 신뢰도가 실제 정확성과 얼마나 잘 일치하는지를 측정합니다. 둘째, **Kendall’s τ**는 대화가 진행됨에 따라 신뢰도가 일관되게 증가하는지를 평가      




짧은 요약(Abstract) :


이 논문의 초록에서는 대형 언어 모델(LLMs)의 신뢰도 추정이 환각 현상을 완화하는 데 유망한 방향임에도 불구하고, 현재 연구가 주로 단일 턴 설정에 집중되고 있다는 점을 지적합니다. 다중 턴 대화에서 모델의 신뢰도가 어떻게 변화하는지는 아직 충분히 탐구되지 않았습니다. 다중 턴 설정에서 신뢰도 추정의 신뢰성은 자율 에이전트 및 인간-기계 협업 시스템과 같은 많은 하위 응용 프로그램에 필수적입니다. 이 연구는 다중 턴 상호작용에서 신뢰도 추정에 대한 첫 번째 체계적인 연구를 제시하며, 두 가지 주요 요구 사항인 턴별 보정과 정보가 추가됨에 따라 신뢰도가 일관되게 증가하는 것을 기반으로 한 평가 프레임워크를 수립합니다. 이를 위해 새로운 메트릭인 길이 정규화된 기대 보정 오류(InfoECE)와 통제된 평가 데이터셋 생성을 위한 "Hinter-Guesser" 패러다임을 도입합니다. 실험 결과, 널리 사용되는 신뢰도 기술들이 다중 턴 대화에서 보정 및 일관성 유지에 어려움을 겪고 있음을 보여주며, 비교적 더 나은 성능을 보이는 로짓 기반 프로브인 P(SUFFICIENT)를 제안합니다. 그러나 이 과제가 완전히 해결되지는 않았습니다. 이 연구는 더 신뢰할 수 있고 신뢰할 수 있는 대화형 에이전트를 개발하기 위한 기초적인 방법론을 제공합니다.



The abstract of this paper points out that while confidence estimation for large language models (LLMs) is a promising direction for mitigating hallucinations, current research predominantly focuses on single-turn settings. The dynamics of model confidence in multi-turn conversations remain largely unexplored. Reliable confidence estimation in multi-turn settings is critical for many downstream applications, such as autonomous agents and human-in-the-loop systems. This work presents the first systematic study of confidence estimation in multi-turn interactions, establishing a formal evaluation framework grounded in two key desiderata: per-turn calibration and monotonicity of confidence as more information becomes available. To facilitate this, we introduce novel metrics, including a length-normalized Expected Calibration Error (InfoECE), and a new "Hinter-Guesser" paradigm for generating controlled evaluation datasets. Our experiments reveal that widely-used confidence techniques struggle with calibration and monotonicity in multi-turn dialogues. We propose P(SUFFICIENT), a logit-based probe that achieves comparatively better performance, although the task remains far from solved. Our work provides a foundational methodology for developing more reliable and trustworthy conversational agents.


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



이 논문에서는 다중 턴 대화에서 대형 언어 모델(LLM)의 신뢰도 추정 방법을 체계적으로 연구합니다. 연구의 주요 목표는 다중 턴 대화에서 모델의 신뢰도를 정확하게 추정하고, 이를 통해 모델의 "환각" 현상을 완화하는 것입니다. 이를 위해 연구진은 다음과 같은 방법론을 제안합니다.

1. **모델 아키텍처**: 연구에서는 Llama3.1과 Qwen2.5와 같은 최신 LLM을 사용합니다. 이 모델들은 대화의 맥락을 이해하고, 사용자의 질문에 대해 적절한 답변을 생성하는 데 최적화되어 있습니다. 각 모델은 다양한 크기(예: 8B, 70B)로 제공되어, 모델의 크기에 따른 성능 차이를 분석할 수 있습니다.

2. **훈련 데이터**: 연구에서는 20Q, GUESS, GRACE, TRICK ME와 같은 여러 데이터셋을 사용하여 모델을 평가합니다. 이 데이터셋들은 다중 턴 대화의 특성을 반영하며, 각 턴에서 모델이 제공하는 정보의 양이 점진적으로 증가하는 구조를 가지고 있습니다. 이를 통해 모델의 신뢰도와 정확성을 평가할 수 있습니다.

3. **신뢰도 추정 기법**: 연구에서는 여러 신뢰도 추정 방법을 비교합니다. 여기에는 다음과 같은 기법이 포함됩니다:
   - **Verbalized Confidence**: 모델이 스스로의 답변에 대한 신뢰도를 수치로 보고하도록 하는 방법입니다. 이 방법은 두 가지 변형(VANILLA-VERB, COT-VERB)으로 나뉘며, 각기 다른 방식으로 신뢰도를 추정합니다.
   - **Self-Consistency (SC)**: 모델이 여러 번 답변을 생성하고, 이 중 가장 일관된 답변을 바탕으로 신뢰도를 계산하는 방법입니다.
   - **Logit-based Probes**: 모델의 내부 신호를 활용하여 신뢰도를 추정하는 방법입니다. 특히, P(TRUE)와 P(SUFFICIENT)라는 두 가지 방법이 사용되며, P(SUFFICIENT)는 주어진 정보가 특정 답변을 지지하는 데 충분한지를 평가합니다.

4. **평가 메트릭**: 연구에서는 신뢰도 추정의 정확성을 평가하기 위해 두 가지 주요 메트릭을 사용합니다. 첫째, **Expected Calibration Error (ECE)**는 모델의 신뢰도가 실제 정확성과 얼마나 잘 일치하는지를 측정합니다. 둘째, **Kendall’s τ**는 대화가 진행됨에 따라 신뢰도가 일관되게 증가하는지를 평가합니다.

이러한 방법론을 통해 연구진은 다중 턴 대화에서 LLM의 신뢰도 추정이 어떻게 이루어지는지를 분석하고, 기존의 신뢰도 추정 기법들이 다중 턴 대화에서 어떤 한계를 가지는지를 밝혀냅니다. 최종적으로, P(SUFFICIENT)라는 새로운 방법이 기존 방법들보다 더 나은 성능을 보임을 보여주며, 향후 연구 방향에 대한 기초를 제공합니다.

---

### English Version

This paper systematically studies confidence estimation methods for large language models (LLMs) in multi-turn conversations. The primary goal of the research is to accurately estimate the model's confidence in multi-turn dialogues and mitigate the phenomenon of "hallucination." To achieve this, the authors propose the following methodology:

1. **Model Architecture**: The study utilizes state-of-the-art LLMs such as Llama3.1 and Qwen2.5. These models are optimized to understand the context of conversations and generate appropriate responses to user queries. Each model is available in various sizes (e.g., 8B, 70B), allowing for an analysis of performance differences based on model size.

2. **Training Data**: The research employs several datasets, including 20Q, GUESS, GRACE, and TRICK ME, to evaluate the models. These datasets reflect the characteristics of multi-turn dialogues and are structured such that the amount of information provided by the model increases progressively with each turn. This allows for the assessment of the model's confidence and accuracy.

3. **Confidence Estimation Techniques**: The study compares various confidence estimation methods, including:
   - **Verbalized Confidence**: A method where the model self-reports its confidence in its answers numerically. This method is divided into two variations (VANILLA-VERB, COT-VERB), each estimating confidence in different ways.
   - **Self-Consistency (SC)**: A method where the model generates multiple answers and calculates confidence based on the most consistent response.
   - **Logit-based Probes**: A method that leverages internal signals from the model to estimate confidence. Specifically, two approaches, P(TRUE) and P(SUFFICIENT), are used, with P(SUFFICIENT) assessing whether the given information sufficiently supports a specific answer.

4. **Evaluation Metrics**: The study employs two key metrics to evaluate the accuracy of confidence estimation. First, **Expected Calibration Error (ECE)** measures how well the model's confidence aligns with actual accuracy. Second, **Kendall’s τ** assesses whether confidence consistently increases as the conversation progresses.

Through this methodology, the authors analyze how confidence estimation occurs in multi-turn dialogues and reveal the limitations of existing confidence estimation techniques in such contexts. Ultimately, they demonstrate that the new method, P(SUFFICIENT), outperforms existing methods, providing a foundation for future research directions.


<br/>
# Results



이 논문에서는 다중 턴 대화에서 대형 언어 모델(LLM)의 신뢰도 추정에 대한 체계적인 연구를 수행하였습니다. 연구의 주요 목표는 다중 턴 대화에서 모델의 신뢰도를 정확하게 추정하는 방법을 개발하고 평가하는 것이었습니다. 이를 위해 두 가지 주요 요구 사항인 "턴별 보정(per-turn calibration)"과 "정보가 추가됨에 따라 신뢰도가 일관되게 증가하는 것(monotonicity)"을 설정하였습니다.

#### 경쟁 모델
연구에서는 여러 신뢰도 추정 방법을 비교하였습니다. 이들 방법에는 다음이 포함됩니다:
1. **VANILLA-VERB**: 모델이 제안한 답변에 대한 신뢰도를 직접 보고하도록 요구하는 방법.
2. **COT-VERB**: 모델이 단계별로 사고한 후 신뢰도를 보고하도록 요구하는 방법.
3. **Self-consistency (SC)**: 여러 번의 답변을 샘플링하여 가장 많이 일치하는 답변의 비율을 신뢰도로 사용하는 방법.
4. **P(TRUE)**: 모델이 제안한 답변이 정답인지 여부를 이진 선택으로 판단하는 방법.
5. **P(SUFFICIENT)**: 현재 정보가 제안된 답변이 유일한 정답인지 여부를 판단하는 방법.

#### 테스트 데이터
연구에서는 두 가지 유형의 데이터셋을 사용하였습니다:
1. **Under-specified datasets**: 초기 질문이 여러 가능한 답변을 허용하는 경우로, 점진적으로 정보가 제공됩니다. 예를 들어, 20Q와 GUESS 데이터셋이 이에 해당합니다.
2. **Fully-specified datasets**: 초기 질문이 유일한 답변을 지시하지만, 모델의 지식 한계로 인해 답변하기 어려운 경우로, GRACE와 TRICK ME 데이터셋이 이에 해당합니다.

#### 메트릭
연구에서는 다음과 같은 메트릭을 사용하여 신뢰도를 평가하였습니다:
- **InfoECE (Expected Calibration Error)**: 신뢰도와 정확도를 비교하여 보정 정도를 측정합니다. 값이 낮을수록 보정이 잘 이루어졌음을 나타냅니다.
- **Kendall’s τ**: 신뢰도가 턴에 따라 일관되게 증가하는지를 측정하는 비모수적 순위 상관 계수입니다. 값이 1에 가까울수록 신뢰도가 일관되게 증가함을 나타냅니다.

#### 비교 결과
실험 결과, P(SUFFICIENT) 방법이 다른 방법들에 비해 상대적으로 더 나은 성능을 보였습니다. 특히, P(SUFFICIENT)는 정보가 추가됨에 따라 신뢰도가 일관되게 증가하는 경향을 보였으며, 이는 다중 턴 대화에서 신뢰도 추정의 중요성을 강조합니다. 반면, 다른 방법들은 보정이나 일관성에서 어려움을 겪는 경향이 있었습니다.

결론적으로, 이 연구는 다중 턴 대화에서 신뢰도 추정의 필요성을 강조하며, 향후 연구 방향으로는 신뢰도와 보정, 정보의 유용성을 효과적으로 구분할 수 있는 방법 개발이 필요하다고 제안합니다.

---




This paper presents a systematic study on confidence estimation for large language models (LLMs) in multi-turn conversations. The primary goal of the research was to develop and evaluate methods for accurately estimating model confidence in multi-turn dialogues. To achieve this, two key desiderata were established: "per-turn calibration" and "monotonicity," where confidence should consistently increase as more information is provided.

#### Competing Models
The study compared several confidence estimation methods, including:
1. **VANILLA-VERB**: A method where the model is required to directly report the confidence in its proposed answer.
2. **COT-VERB**: A method where the model is required to think step-by-step before reporting its confidence.
3. **Self-consistency (SC)**: A method that estimates confidence based on the fraction of sampled answers that match.
4. **P(TRUE)**: A binary choice method where the model determines if the proposed answer is correct.
5. **P(SUFFICIENT)**: A method that assesses whether the current information is sufficient to conclude that the proposed answer is the only correct one.

#### Test Data
The research utilized two types of datasets:
1. **Under-specified datasets**: Cases where the initial question allows for multiple plausible answers, with information revealed progressively. Examples include the 20Q and GUESS datasets.
2. **Fully-specified datasets**: Cases where the initial question points to a unique answer but is difficult to answer due to the model's knowledge limitations. Examples include the GRACE and TRICK ME datasets.

#### Metrics
The study employed the following metrics to evaluate confidence:
- **InfoECE (Expected Calibration Error)**: Measures the calibration by comparing confidence and accuracy. Lower values indicate better calibration.
- **Kendall’s τ**: A non-parametric rank correlation coefficient that measures whether confidence consistently increases over turns. Values closer to 1 indicate a consistent increase in confidence.

#### Comparison Results
The experimental results revealed that the P(SUFFICIENT) method outperformed other methods in terms of performance. Notably, P(SUFFICIENT) showed a consistent increase in confidence as information was added, highlighting the importance of confidence estimation in multi-turn dialogues. In contrast, other methods struggled with calibration and monotonicity.

In conclusion, this study emphasizes the necessity of confidence estimation in multi-turn conversations and suggests future research directions focused on developing methods that effectively distinguish between confidence and calibration, as well as the utility of information.


<br/>
# 예제



이 논문에서는 다중 턴 대화에서 대형 언어 모델(LLM)의 신뢰도 추정에 대한 체계적인 연구를 수행합니다. 연구의 주요 목표는 다중 턴 대화에서 모델의 신뢰도를 어떻게 평가하고 개선할 수 있는지를 탐구하는 것입니다. 이를 위해 두 가지 주요 요구 사항인 "턴별 보정(per-turn calibration)"과 "정보가 추가됨에 따라 신뢰도가 일관되게 증가하는 것(monotonicity)"을 설정합니다.

#### 데이터셋 구성

1. **훈련 데이터셋 (Training Dataset)**:
   - **20Q**: 사용자가 비밀 엔티티를 추측하는 게임입니다. 질문자는 예/아니오 질문을 통해 정보를 점진적으로 수집합니다. 예를 들어, 사용자가 "이것은 인간이 만든 것인가요?"라고 질문하면, 모델은 "예"라고 대답할 수 있습니다. 이 과정은 사용자가 비밀 엔티티를 추측할 때까지 계속됩니다.
   - **GUESS**: 사용자가 도시를 추측하는 게임입니다. 질문자는 개방형 질문을 통해 정보를 수집합니다. 예를 들어, "이 도시는 아시아에 있나요?"라는 질문에 모델이 "예"라고 대답할 수 있습니다.

2. **테스트 데이터셋 (Test Dataset)**:
   - **GRACE**: 퀴즈볼 스타일의 질문으로, 점진적으로 구체적인 단서가 제공됩니다. 각 단서는 독립적이며 명확합니다. 예를 들어, "이 사람은 Charlie Parker가 아니지만, Stravinsky의 'The Firebird'를 편곡한 뮤지션의 성은 무엇인가요?"와 같은 질문이 주어집니다.
   - **TRICK ME**: 인간이 작성한 적대적인 질문 데이터셋으로, 모델이 실수를 유도하도록 설계된 질문이 포함되어 있습니다. 예를 들어, "이 사람은 'Power'와 'Gold Diggers'의 작곡가로 알려져 있습니다. 그의 이름은 무엇인가요?"와 같은 질문이 주어집니다.

#### 구체적인 인풋과 아웃풋

- **인풋 (Input)**: 각 턴에서 모델은 이전 대화의 히스토리와 현재 질문을 포함한 프롬프트를 받습니다. 예를 들어, 20Q 게임에서 사용자가 "이것은 인간이 만든 것인가요?"라고 질문하면, 모델은 이전 질문과 대답을 포함한 히스토리를 바탕으로 대답해야 합니다.
  
- **아웃풋 (Output)**: 모델은 각 턴에서 대답과 함께 신뢰도 점수를 출력합니다. 신뢰도 점수는 0에서 1 사이의 값으로, 모델이 자신의 대답이 맞을 확률을 나타냅니다. 예를 들어, 모델이 "예"라고 대답하고 신뢰도 점수를 0.85로 출력할 수 있습니다.

이러한 방식으로, 연구는 다중 턴 대화에서 신뢰도 추정의 정확성과 일관성을 평가하고, 다양한 신뢰도 추정 방법의 성능을 비교합니다.

---




This paper presents a systematic study of confidence estimation for large language models (LLMs) in multi-turn conversations. The main goal of the research is to explore how to evaluate and improve the confidence of models in multi-turn dialogues. To achieve this, two key desiderata are established: "per-turn calibration" and "monotonicity," where confidence should consistently increase as more information is provided.

#### Dataset Construction

1. **Training Dataset**:
   - **20Q**: A game where the user guesses a secret entity. The questioner asks yes/no questions to incrementally gather information. For example, if the user asks, "Is it human-made?" the model might respond "Yes." This process continues until the user can guess the secret entity.
   - **GUESS**: A game where the user guesses a city. The questioner asks open-ended questions to gather information. For example, the question "Is the city in Asia?" could receive a "Yes" response from the model.

2. **Test Dataset**:
   - **GRACE**: A quiz bowl-style question set where increasingly specific clues are provided. Each clue is self-contained and unambiguous. For example, a question might be, "This person is not Charlie Parker, but what is the surname of the musician who arranged excerpts from Stravinsky's 'The Firebird'?"
   - **TRICK ME**: A human-authored adversarial question dataset designed to elicit mistakes from the model. An example question could be, "This person is known for songs like 'Power' and 'Gold Diggers.' What is his name?"

#### Specific Inputs and Outputs

- **Input**: At each turn, the model receives a prompt that includes the history of previous dialogue and the current question. For instance, in the 20Q game, if the user asks, "Is it human-made?" the model must respond based on the history of previous questions and answers.

- **Output**: The model outputs an answer along with a confidence score at each turn. The confidence score is a value between 0 and 1, indicating the model's certainty about its answer being correct. For example, the model might respond "Yes" and output a confidence score of 0.85.

Through this approach, the research evaluates the accuracy and consistency of confidence estimation in multi-turn dialogues and compares the performance of various confidence estimation methods.

<br/>
# 요약


이 논문에서는 다중 턴 대화에서 대형 언어 모델(LLM)의 신뢰도 추정 방법을 체계적으로 연구하고, 정보가 축적됨에 따라 신뢰도가 증가해야 한다는 두 가지 주요 요구사항(보정 및 단조성)을 설정하였다. 실험 결과, 기존의 신뢰도 추정 기법들이 다중 턴 대화에서 보정 및 단조성을 유지하는 데 어려움을 겪는 반면, 제안된 P(SUFFICIENT) 방법이 상대적으로 더 나은 성능을 보였다. 이 연구는 신뢰할 수 있는 대화형 에이전트를 개발하기 위한 기초적인 방법론을 제공한다.

---

This paper systematically studies confidence estimation methods for large language models (LLMs) in multi-turn conversations, establishing two key desiderata (calibration and monotonicity) that require confidence to increase as information accumulates. Experimental results reveal that existing confidence estimation techniques struggle to maintain calibration and monotonicity in multi-turn dialogues, while the proposed P(SUFFICIENT) method demonstrates comparatively better performance. This research provides a foundational methodology for developing reliable conversational agents.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: InfoECE 그래프는 모델의 신뢰도가 정보가 제공됨에 따라 어떻게 변화하는지를 보여줍니다. 이상적으로는 신뢰도가 증가해야 하며, 신뢰도 곡선이 정확도 곡선에 가까워질수록 보정이 개선됩니다. P(S UFFICIENT) 방법이 가장 잘 이 두 가지 조건을 만족하는 것으로 나타났습니다.
   - **Figure 2**: 대화가 진행됨에 따라 평균 신뢰도와 정확도가 어떻게 변화하는지를 시각화합니다. 신뢰도는 정보가 축적됨에 따라 증가해야 하며, P(S UFFICIENT) 방법이 가장 일관되게 이 경향을 따르는 것으로 나타났습니다.
   - **Figure 3**: 각 턴의 정답에 대한 신뢰도를 평가했을 때, 모든 방법이 상당한 증가를 보였으며, P(S UFFICIENT)가 가장 높은 성능을 보였습니다. 이는 모델이 현재 힌트가 정답과 일치할 때 신뢰도를 인식할 수 있음을 시사합니다.

2. **테이블**
   - **Table 2**: 다양한 모델과 데이터셋에서의 InfoECE와 Kendall’s τ를 보여줍니다. P(S UFFICIENT) 방법이 다른 방법들에 비해 더 나은 성능을 보였으며, 특히 다중 턴 대화에서 신뢰도 보정과 일관성을 유지하는 데 효과적임을 나타냅니다.
   - **Table 3**: 다양한 모델과 데이터셋에서의 평균 신뢰도를 비교합니다. P(S UFFICIENT) 방법이 정보가 제공될 때 신뢰도가 어떻게 변화하는지를 잘 추적하는 것으로 나타났습니다.
   - **Appendix A**: 다양한 프롬프트 템플릿을 제공하여 모델이 신뢰도를 어떻게 보고하도록 유도하는지를 설명합니다. 이는 신뢰도 추정의 일관성을 높이는 데 기여합니다.

3. **어펜딕스**
   - **Appendix B**: GRACE와 TRICK ME 데이터셋의 세부 정보를 제공하며, 이 데이터셋들이 어떻게 신뢰도 평가를 위한 기준을 제공하는지를 설명합니다.
   - **Appendix C**: 다중 턴 대화와 단일 턴 요약 간의 성능 비교를 통해, 두 설정 간의 정확도 차이가 미미하다는 것을 보여줍니다. 이는 다중 턴 대화가 단일 턴 요약에 비해 인지적 부담이 적다는 것을 시사합니다.

### Insights  (Results and Insights from Diagrams, Figures, Tables, and Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: The InfoECE graph illustrates how model confidence changes as information is provided. Ideally, confidence should increase, and calibration improves when the confidence curve approaches the accuracy curve. The P(S UFFICIENT) method was found to best satisfy both conditions.
   - **Figure 2**: This visualizes how average confidence and accuracy evolve as the dialogue progresses. Confidence should increase as information accumulates, and the P(S UFFICIENT) method consistently followed this trend.
   - **Figure 3**: When evaluating confidence against the ground truth at each turn, all methods showed substantial increases, with P(S UFFICIENT) demonstrating the highest performance. This suggests that models can recognize when current hints align with the correct answer.

2. **Tables**
   - **Table 2**: Displays InfoECE and Kendall’s τ across various models and datasets. The P(S UFFICIENT) method outperformed others, particularly in maintaining calibration and monotonicity in multi-turn dialogues.
   - **Table 3**: Compares average confidence across different models and datasets. The P(S UFFICIENT) method effectively tracks how confidence changes with the provision of information.
   - **Appendix A**: Provides various prompt templates that guide models on how to report confidence, contributing to the consistency of confidence estimation.

3. **Appendices**
   - **Appendix B**: Offers details on the GRACE and TRICK ME datasets, explaining how these datasets serve as benchmarks for confidence evaluation.
   - **Appendix C**: Compares performance between multi-turn dialogues and single-turn summaries, showing that accuracy differences are minimal, suggesting that multi-turn dialogues do not impose significant cognitive burdens compared to single-turn summaries.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{zhang2026confidence,
  author = {Caiqi Zhang and Ruihan Yang and Xiaochen Zhu and Chengzu Li and Tiancheng Hu and Yijiang Dong and Deqing Yang and Nigel Collier},
  title = {Confidence Estimation for LLMs in Multi-turn Interactions},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages = {25661--25676},
  year = {2026},
  month = {July},
  publisher = {Association for Computational Linguistics},
}
```

### 시카고 스타일

Caiqi Zhang, Ruihan Yang, Xiaochen Zhu, Chengzu Li, Tiancheng Hu, Yijiang Dong, Deqing Yang, and Nigel Collier. "Confidence Estimation for LLMs in Multi-turn Interactions." In *Findings of the Association for Computational Linguistics: ACL 2026*, 25661–25676. Association for Computational Linguistics, July 2026.
