---
layout: post
title:  "[2025]DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues"
date:   2025-08-13 16:01:37 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 


대화형 도구 호출 기능을 평가하기 위한 새로운 벤치마크인 DICE-BENCH를 소개(앱 호출과 같은)  


짧은 요약(Abstract) :

이 논문은 대화형 도구 호출 기능을 평가하기 위한 새로운 벤치마크인 DICE-BENCH를 소개합니다. 기존의 벤치마크는 단일 회차 상호작용에 초점을 맞추고 있어 현실 세계의 복잡성을 간과하고 있습니다. 이를 해결하기 위해 DICE-SCORE라는 새로운 지표를 도입하여 대화 전반에 걸쳐 도구 관련 정보의 분산 정도를 평가합니다. DICE-BENCH는 도구 그래프와 다중 에이전트 시스템을 사용하여 현실적인 대화 데이터를 생성하며, 1,607개의 고품질 데이터셋을 포함합니다. 19개의 대형 언어 모델(LLM)을 대상으로 한 실험 결과, 이러한 모델들이 실제 환경에서 효과적으로 배포되기 위해서는 여전히 많은 발전이 필요함을 보여줍니다. 코드와 데이터는 공개되어 있습니다.


This paper introduces DICE-BENCH, a new benchmark for evaluating the tool-use capabilities of large language models (LLMs) in multi-round, multi-party dialogues. Existing function-calling benchmarks focus on single-turn interactions, overlooking the complexity of real-world scenarios. To address this, we introduce DICE-SCORE, a metric that evaluates the dispersion of tool-related information such as function names and parameter values throughout the dialogue. Analyzing existing benchmarks through DICE-SCORE reveals notably low scores, highlighting the need for more realistic scenarios. DICE-BENCH constructs practical function-calling datasets by synthesizing conversations through a tool graph that maintains dependencies across rounds and a multi-agent system with distinct personas to enhance dialogue naturalness. The final dataset comprises 1,607 high-DICE-SCORE instances. Our experiments on 19 LLMs with DICE-BENCH show that significant advances are still required before such models can be deployed effectively in real-world settings. Our code and data are all publicly available.


* Useful sentences :


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>

논문 "DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues"에서는 대규모 언어 모델(LLM)의 도구 사용 능력을 평가하기 위한 새로운 벤치마크인 DICE-BENCH를 소개합니다. 이 벤치마크는 현실적인 다중 라운드, 다중 참여자 대화에서의 함수 호출 성능을 평가하는 데 중점을 둡니다. 기존의 함수 호출 벤치마크는 주로 단일 사용자와의 상호작용에 초점을 맞추고 있어, 실제 그룹 채팅 시나리오에서의 복잡성을 간과하고 있습니다. DICE-BENCH는 이러한 한계를 극복하기 위해 설계되었습니다.

DICE-BENCH는 세 가지 주요 단계로 데이터 구축을 진행합니다. 첫째, 도구 그래프 구축 단계에서는 현실적인 일상 시나리오를 반영하기 위해 다양한 도구를 수집하고, 이들 간의 의존성을 나타내는 도구 그래프를 생성합니다. 둘째, 시나리오 구성 단계에서는 도구 체인을 샘플링하고, 대화 유형, 참여자 수, 라운드 수 등을 설정하여 다양한 대화 시나리오를 구성합니다. 마지막으로 대화 생성 단계에서는 다중 에이전트 시스템을 사용하여 각 에이전트가 고유한 페르소나를 가지도록 하여 자연스러운 대화를 생성합니다.

DICE-BENCH는 또한 DICE-SCORE라는 새로운 메트릭을 도입하여 대화 내에서 도구 관련 정보가 얼마나 분산되어 있는지를 평가합니다. 이 메트릭은 정보가 여러 턴에 걸쳐 얼마나 흩어져 있는지를 측정하여, LLM이 이러한 정보를 통합하는 데 필요한 능력을 평가합니다. 실험 결과, DICE-SCORE가 높을수록 모델의 성능이 크게 저하되는 것으로 나타났으며, 이는 대화 추적 및 문맥 통합 전략의 개선 필요성을 강조합니다.

이 연구의 주요 기여는 다음과 같습니다:
1. DICE-BENCH는 현실적인 그룹 채팅 데이터를 기반으로 한 최초의 다중 라운드, 다중 참여자 함수 호출 벤치마크입니다.
2. DICE-SCORE는 다중 참여자 대화의 복잡성을 포착하는 새로운 메트릭으로, 흩어진 함수 호출 정보를 검색하는 난이도를 평가합니다.
3. 다양한 LLM에 대한 철저한 평가를 통해, 분산된 다중 라운드 대화 문맥을 처리하는 데 있어 모델의 한계를 분석하고 귀중한 통찰을 제공합니다.

---



The paper "DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues" introduces a new benchmark called DICE-BENCH, designed to evaluate the tool-use capabilities of large language models (LLMs) in realistic multi-round, multi-party dialogues. Existing function-calling benchmarks primarily focus on single-user interactions, overlooking the complexities of real-world group chat scenarios. DICE-BENCH is designed to address these limitations.

DICE-BENCH constructs data through three main steps. First, in the Tool Graph Construction phase, various tools are collected to reflect realistic everyday scenarios, and a tool graph is generated to represent dependencies among these tools. Second, in the Scenario Configuration phase, tool chains are sampled, and dialogue types, the number of participants, and the number of rounds are set to configure diverse dialogue scenarios. Finally, in the Dialogue Generation phase, a multi-agent system is used to generate natural dialogues, with each agent having a distinct persona.

DICE-BENCH also introduces a new metric called DICE-SCORE, which evaluates how dispersed tool-related information is within a dialogue. This metric measures how scattered the information is across multiple turns, assessing the LLM's ability to integrate such information. Experimental results show a significant drop in model performance as the DICE-SCORE increases, highlighting the need for improved dialogue-tracking and context-integration strategies.

The key contributions of this study are as follows:
1. DICE-BENCH is the first multi-round, multi-party benchmark for function-calling, grounded in realistic group chat data.
2. DICE-SCORE is a novel metric that captures the complexity of multi-party conversation by assessing the difficulty of retrieving scattered function call information.
3. A thorough evaluation of various LLMs provides valuable insights into their limitations in handling fragmented multi-round dialogue contexts.


<br/>
# Results


#### 경쟁 모델
DICE-BENCH에서 평가된 모델은 총 19개로, 이들은 최소 8k의 컨텍스트 윈도우 크기를 지원하는 모델들입니다. 이 중 폐쇄형 모델로는 GPT-4o, GPT-4o-mini, Gemini 2 Flash, Gemini 2 Flash Lite가 포함되었고, 개방형 모델로는 LLaMA3, Qwen2.5, Mistral, EXAONE, Phi4, GLM4-Chat 등이 포함되었습니다. 또한, Hammer2.1, ToolAce, CALM, NexusRaven-V2, Granite와 같은 도구 특화 모델들도 평가되었습니다.

#### 테스트 데이터
DICE-BENCH는 다중 라운드, 다중 참여자 대화에서의 도구 호출 성능을 평가하기 위해 설계된 벤치마크입니다. 총 1,607개의 대화 인스턴스가 포함되어 있으며, 각 대화는 여러 라운드와 다양한 참여자 구성을 통해 복잡한 시나리오를 제공합니다. 이 데이터는 현실적인 그룹 채팅 데이터를 기반으로 하여, 실제 상황에서의 도구 호출 성능을 평가할 수 있도록 구성되었습니다.

#### 메트릭
평가 메트릭으로는 정확한 함수와 해당 매개변수를 선택하는지를 평가하는 정확한 일치(Exact Match, EM) 점수가 사용되었습니다. 이 점수는 각 구성 데이터셋에 대한 EM을 평균하여 최종 점수를 산출합니다. 또한, DICE-SCORE라는 새로운 메트릭이 도입되어, 대화 내에서 도구 관련 정보가 얼마나 분산되어 있는지를 측정하여 작업의 난이도를 평가합니다.

#### 비교
모델 성능을 비교한 결과, GPT-4o가 대부분의 라운드와 참여자 구성에서 가장 높은 성능을 보였습니다. 개방형 모델 중에서는 Phi4-15B가 대부분의 시나리오에서 가장 높은 점수를 기록했으며, 이는 폐쇄형 모델과 비교해도 손색이 없는 성능을 보여주었습니다. 7B–9B 모델 중에서는 GLM-9B가 가장 높은 점수를 기록했으며, 32B–70B 모델 중에서는 Qwen 32B가 대부분의 설정에서 최고 점수를 기록했습니다. 이는 Qwen 2.5의 128k 토큰 컨텍스트 윈도우가 긴 대화 시나리오에서의 강점을 유지하는 데 기여했음을 시사합니다.



#### Competing Models
A total of 19 models were evaluated on DICE-BENCH, all supporting at least an 8k context window size. Among these, closed-source models included GPT-4o, GPT-4o-mini, Gemini 2 Flash, and Gemini 2 Flash Lite. Open-source models included LLaMA3, Qwen2.5, Mistral, EXAONE, Phi4, and GLM4-Chat. Additionally, tool-specific models such as Hammer2.1, ToolAce, CALM, NexusRaven-V2, and Granite were also evaluated.

#### Test Data
DICE-BENCH is a benchmark designed to evaluate tool-calling performance in multi-round, multi-party dialogues. It includes a total of 1,607 dialogue instances, each providing complex scenarios through multiple rounds and various participant configurations. The data is based on realistic group chat data, allowing for the assessment of tool-calling performance in real-world settings.

#### Metrics
The evaluation metric used was the Exact Match (EM) score, which assesses whether the model selects the exact function along with its corresponding parameters. The final score is obtained by averaging the EM across the configuration dataset. Additionally, a new metric called DICE-SCORE was introduced to measure how dispersed tool-related information is within the dialogue, thereby assessing the task's difficulty.

#### Comparison
In terms of model performance comparison, GPT-4o achieved the highest performance in most rounds and participant configurations. Among open-source models, Phi4-15B recorded the highest scores in most scenarios, showing performance comparable to closed-source models. Among the 7B–9B models, GLM-9B achieved the highest overall score, while in the 32B–70B category, the Qwen 32B model secured top scores in most settings. This suggests that Qwen 2.5's 128k-token context window helps maintain resilience in extended dialogue scenarios.


<br/>
# 예제


#### 트레이닝 데이터와 테스트 데이터의 인풋과 아웃풋

1. **트레이닝 데이터 인풋**:
   - 다중 라운드, 다중 참여자 대화: 각 대화는 여러 라운드로 구성되며, 각 라운드에는 여러 사용자의 발화와 시스템 응답이 포함됩니다.
   - 도구 그래프: 각 대화는 도구 그래프에 기반하여 생성되며, 이는 각 라운드 간의 의존성을 유지합니다.
   - 대화 유형: 설득, 탐구, 논쟁 등 다양한 대화 유형이 포함됩니다.

2. **트레이닝 데이터 아웃풋**:
   - 함수 호출: 각 대화는 특정 함수 호출과 그에 필요한 매개변수 값을 생성합니다.
   - DICE-SCORE: 대화 내에서 도구 관련 정보의 분산 정도를 평가하는 점수입니다.

3. **테스트 데이터 인풋**:
   - 트레이닝 데이터와 유사한 구조의 다중 라운드, 다중 참여자 대화.
   - 도구 그래프와 대화 유형이 포함됩니다.

4. **테스트 데이터 아웃풋**:
   - 모델이 생성한 함수 호출과 매개변수 값.
   - 모델의 성능을 평가하기 위한 정확한 매칭 점수(Exact Match Score).

#### 구체적인 테스크

- **목표**: LLM(대형 언어 모델)이 다중 라운드, 다중 참여자 대화에서 함수 호출을 정확히 수행할 수 있는지를 평가합니다.
- **테스크 구성**:
  1. 대화의 각 라운드에서 주어진 대화와 도구 문서를 기반으로 적절한 함수와 매개변수 값을 식별합니다.
  2. 대화의 맥락을 통합하여 분산된 정보를 수집하고, 이를 통해 정확한 함수 호출을 생성합니다.
  3. DICE-SCORE를 사용하여 대화의 복잡성을 평가하고, 모델의 성능과의 상관관계를 분석합니다.

---



#### Input and Output of Training and Test Data

1. **Training Data Input**:
   - Multi-round, multi-party dialogues: Each dialogue consists of multiple rounds, with each round containing multiple user utterances and system responses.
   - Tool Graph: Each dialogue is generated based on a tool graph that maintains dependencies across rounds.
   - Dialogue Types: Includes various dialogue types such as persuasion, inquiry, and eristic.

2. **Training Data Output**:
   - Function Calls: Each dialogue generates specific function calls and the necessary parameter values.
   - DICE-SCORE: A score that evaluates the dispersion of tool-related information within the dialogue.

3. **Test Data Input**:
   - Multi-round, multi-party dialogues with a structure similar to the training data.
   - Includes tool graphs and dialogue types.

4. **Test Data Output**:
   - Function calls and parameter values generated by the model.
   - Exact Match Score for evaluating the model's performance.

#### Specific Task

- **Objective**: To evaluate whether LLMs (Large Language Models) can accurately perform function-calling in multi-round, multi-party dialogues.
- **Task Structure**:
  1. Identify the appropriate function and parameter values based on the given dialogue and tool documents in each round of the dialogue.
  2. Integrate the context of the dialogue to gather dispersed information and generate accurate function calls.
  3. Use DICE-SCORE to assess the complexity of the dialogue and analyze its correlation with model performance.

<br/>
# 요약

DICE-BENCH는 다중 라운드, 다중 참여자 대화에서 대형 언어 모델의 도구 사용 능력을 평가하기 위한 벤치마크로, 대화 내 도구 관련 정보의 분산을 측정하는 DICE-SCORE를 도입하여 기존 벤치마크의 한계를 보완합니다. 실험 결과, DICE-BENCH는 1,607개의 고품질 대화 인스턴스를 통해 모델이 여러 라운드와 참여자에 걸쳐 분산된 정보를 통합하는 데 어려움을 겪는다는 것을 보여줍니다. 이 연구는 복잡한 대화 맥락을 통합하는 AI 비서의 발전을 촉진하는 데 기여합니다.


DICE-BENCH is a benchmark designed to evaluate the tool-use capabilities of large language models in multi-round, multi-party dialogues, introducing the DICE-SCORE to measure the dispersion of tool-related information within dialogues, addressing limitations of existing benchmarks. Experimental results demonstrate that DICE-BENCH, with 1,607 high-quality dialogue instances, reveals the challenges models face in integrating dispersed information across multiple rounds and participants. This research contributes to advancing AI assistants capable of integrating complex dialogue contexts.

<br/>
# 기타


1. **Figure 1: Single-Round, Four-Party Dialogue**
   - **결과**: DICE-BENCH의 대화 예시로, 여러 사용자가 참여하는 대화에서 LLM이 함수 호출에 필요한 정보를 어떻게 식별하는지를 보여줍니다.
   - **인사이트**: 다중 사용자 대화에서 정보가 어떻게 분산되어 있는지를 시각적으로 이해할 수 있으며, 이는 LLM이 정보를 통합하는 데 필요한 복잡성을 강조합니다.

2. **Table 1: Baseline Comparison**
   - **결과**: 다양한 함수 호출 벤치마크와 DICE-BENCH를 비교한 표입니다.
   - **인사이트**: DICE-BENCH는 다중 사용자 및 다중 라운드 대화를 포함하는 유일한 벤치마크로, 더 현실적인 작업을 처리할 수 있음을 보여줍니다.

3. **Figure 2: DICE-BENCH Data-Generation Pipeline**
   - **결과**: DICE-BENCH 데이터 생성 과정의 단계별 흐름을 보여줍니다.
   - **인사이트**: 도구 그래프 구축, 시나리오 구성, 대화 생성의 세 가지 주요 단계가 어떻게 상호작용하여 현실적인 대화 데이터를 생성하는지를 이해할 수 있습니다.

4. **Table 2: Filtering Statistics per Round**
   - **결과**: 각 라운드별로 필터링된 데이터의 수를 보여줍니다.
   - **인사이트**: 자동화된 평가, 규칙 기반 필터링, 기준 기반 필터링의 세 단계를 통해 데이터의 품질을 보장합니다.

5. **Table 3: Data Statistics of DICE-BENCH**
   - **결과**: DICE-BENCH의 데이터 통계를 보여줍니다.
   - **인사이트**: 각 구성 요소(라운드, 파티, 대화 유형)별로 데이터가 어떻게 분포되어 있는지를 이해할 수 있습니다.

6. **Figure 3: Inverse Correlation between DICE-SCORE and Model Performance**
   - **결과**: DICE-SCORE와 모델 성능 간의 역상관 관계를 보여줍니다.
   - **인사이트**: DICE-SCORE가 높을수록 모델 성능이 낮아지는 경향이 있으며, 이는 정보가 분산될수록 작업이 더 어려워짐을 나타냅니다.

7. **Table 4: Human Evaluation Results by Round**
   - **결과**: 각 라운드별 인간 평가 결과를 보여줍니다.
   - **인사이트**: DICE-SCORE가 높아질수록 인간의 정확도가 낮아지는 경향이 있으며, 이는 DICE-SCORE가 작업의 난이도를 잘 반영하고 있음을 시사합니다.

8. **Table 5: Main Experiment Results of DICE-BENCH**
   - **결과**: 다양한 LLM의 DICE-BENCH 성능을 보여줍니다.
   - **인사이트**: GPT-4o가 가장 높은 성능을 보였으며, Phi4-15B가 오픈 소스 모델 중 가장 우수한 성능을 보였습니다.

9. **Figure 4: EM Performance Scores vs DICE-SCORE**
   - **결과**: EM 성능 점수와 DICE-SCORE 간의 관계를 보여줍니다.
   - **인사이트**: DICE-SCORE가 높을수록 모델의 EM 성능이 낮아지는 경향이 있으며, 이는 DICE-SCORE가 작업의 복잡성을 잘 반영하고 있음을 나타냅니다.

10. **Appendices (A to P)**
    - **결과**: 다양한 추가 정보와 실험 설정, 데이터 생성 및 평가 방법에 대한 세부 사항을 제공합니다.
    - **인사이트**: 연구의 재현 가능성을 높이고, 연구 방법론에 대한 깊은 이해를 제공합니다.

---



1. **Figure 1: Single-Round, Four-Party Dialogue**
   - **Result**: An example of a dialogue in DICE-BENCH, showing how LLMs identify function-related information in multi-user dialogues.
   - **Insight**: Provides a visual understanding of how information is dispersed in multi-user dialogues, highlighting the complexity required for LLMs to integrate information.

2. **Table 1: Baseline Comparison**
   - **Result**: A comparison of various function-calling benchmarks with DICE-BENCH.
   - **Insight**: Demonstrates that DICE-BENCH is the only benchmark encompassing both multi-party and multi-round dialogues, capable of handling more realistic tasks.

3. **Figure 2: DICE-BENCH Data-Generation Pipeline**
   - **Result**: Shows the step-by-step flow of the DICE-BENCH data generation process.
   - **Insight**: Understands how the three main stages—Tool Graph Construction, Scenario Configuration, and Dialogue Generation—interact to produce realistic dialogue data.

4. **Table 2: Filtering Statistics per Round**
   - **Result**: Shows the number of data points filtered at each round.
   - **Insight**: Ensures data quality through three stages: automated evaluation, rule-based filtering, and criteria-based filtering.

5. **Table 3: Data Statistics of DICE-BENCH**
   - **Result**: Shows the data statistics of DICE-BENCH.
   - **Insight**: Understands how data is distributed across each component (round, party, dialogue type).

6. **Figure 3: Inverse Correlation between DICE-SCORE and Model Performance**
   - **Result**: Shows the inverse correlation between DICE-SCORE and model performance.
   - **Insight**: Indicates that higher DICE-SCOREs correlate with lower model performance, suggesting that tasks become more difficult as information is more dispersed.

7. **Table 4: Human Evaluation Results by Round**
   - **Result**: Shows human evaluation results by round.
   - **Insight**: Indicates that as DICE-SCORE increases, human accuracy tends to decrease, suggesting that DICE-SCORE effectively reflects task difficulty.

8. **Table 5: Main Experiment Results of DICE-BENCH**
   - **Result**: Shows the performance of various LLMs on DICE-BENCH.
   - **Insight**: GPT-4o showed the highest performance, with Phi4-15B being the best-performing open-source model.

9. **Figure 4: EM Performance Scores vs DICE-SCORE**
   - **Result**: Shows the relationship between EM performance scores and DICE-SCORE.
   - **Insight**: Indicates that higher DICE-SCOREs correlate with lower EM performance, suggesting that DICE-SCORE effectively reflects task complexity.

10. **Appendices (A to P)**
    - **Result**: Provides various additional information and details on experimental settings, data generation, and evaluation methods.
    - **Insight**: Enhances the reproducibility of the study and provides a deep understanding of the research methodology.

<br/>
# refer format:


**BibTeX 형식:**
```bibtex
@inproceedings{jang2025dicebench,
  title={DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues},
  author={Jang, Kyochul and Lee, Donghyeon and Kim, Kyusik and Heo, Dongseok and Lee, Taewhoo and Kim, Woojeong and Suh, Bongwon},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={26822--26846},
  year={2025},
  organization={Association for Computational Linguistics}
}
```

**시카고 스타일 줄글 형식:**
Jang, Kyochul, Donghyeon Lee, Kyusik Kim, Dongseok Heo, Taewhoo Lee, Woojeong Kim, and Bongwon Suh. 2025. "DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues." In *Findings of the Association for Computational Linguistics: ACL 2025*, 26822–26846. Association for Computational Linguistics.
