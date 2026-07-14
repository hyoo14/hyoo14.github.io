---
layout: post
title:  "[2026]SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation"
date:   2026-07-14 01:08:21 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: SAFARI는 긴 수명의 에이전트 오류 귀속을 위한 능동적 조사 프레임워크로, 전체 경로를 LLM의 제한된 컨텍스트 창에 로드하는 대신 도구 보강 진단 루프를 사용하여 진단 정확도를 향상시킵니다.


짧은 요약(Abstract) :


이 논문에서는 복잡한 다단계, 다중 에이전트 작업을 수행하는 자율 에이전트의 실패를 진단하는 새로운 방법인 SAFARI(Scaling Long Horizon Agentic Fault Attribution via Active Investigation)를 소개합니다. 기존의 방법들은 전체 실행 경로를 LLM의 컨텍스트 창에 로드하여 진단을 수행하는데, 이는 컨텍스트 한계를 초과할 경우 주의력 희석 문제를 겪고 실패합니다. SAFARI는 선형 컨텍스트 로딩을 도구 보강 진단 루프로 대체하여, LLM이 특정 도구를 사용하여 경로 세그먼트를 읽고 검색할 수 있도록 하며, 지속적인 단기 기억(Short-Term Memory, STM)을 통해 여러 턴 간의 추론을 가능하게 합니다. 실험 결과, SAFARI는 1M 토큰 예산 내에서 Who&When 데이터셋에서 20% 향상된 성능을 보였고, 25K 토큰 예산의 TRAIL GAIA 하위 집합에서도 19% 향상된 성능을 기록했습니다. 특히, SAFARI는 목표 결함이 모델의 기본 컨텍스트 창보다 5배 이상 떨어져 있을 때도 0.58의 정밀도를 유지하여, 전통적인 평가자들이 완전히 실패하는 상황에서도 효과적으로 작동합니다.



This paper introduces SAFARI (Scaling Long Horizon Agentic Fault Attribution via Active Investigation), a new method for diagnosing failures in autonomous agents performing complex multi-step, multi-agent tasks. Current methods load the entire execution trajectory into an LLM's context window for diagnosis, which suffers from attention dilution and fails when the context limits are inevitably exceeded. SAFARI replaces linear context loading with a tool-augmented diagnostic loop, allowing LLMs to read and search trajectory segments using specialized tools while enabling cross-turn reasoning through a persistent Short-Term Memory (STM). Experimental results show that SAFARI outperforms state-of-the-art results by 20% on the Who&When dataset within a 1M token budget and by 19% on the TRAIL GAIA subset on a 25K token budget. Most significantly, SAFARI maintains a precision of 0.58 even when the target fault resides 5x beyond the model's native context window, a scenario where traditional evaluators fail entirely.


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



SAFARI(Scaling Long Horizon Agentic Fault Attribution via Active Investigation)는 복잡한 다단계 및 다중 에이전트 작업을 수행하는 자율 에이전트의 실패를 진단하기 위한 새로운 프레임워크입니다. 기존의 방법들은 전체 실행 경로를 LLM(대형 언어 모델)의 컨텍스트 창에 로드하여 진단을 수행하는데, 이로 인해 주의력 희석(attention dilution) 문제가 발생하고, 에이전트의 흔적이 컨텍스트 한계를 초과할 경우 실패하게 됩니다. SAFARI는 이러한 문제를 해결하기 위해 선형 컨텍스트 로딩을 도구 보강 진단 루프로 대체합니다.

SAFARI의 주요 구성 요소는 다음과 같습니다:

1. **조사자 에이전트(Investigator Agent)**: 이 에이전트는 주어진 경로(trajectory)와 상호작용하며, 반복적인 도구 호출을 통해 증거를 수집합니다. 각 조사 단계에서 에이전트는 현재 대화 이력을 관찰하고, 도구 스키마를 기반으로 도구 호출을 생성합니다.

2. **도구 호출**: SAFARI는 두 가지 주요 도구를 사용합니다:
   - `read(offset, limit)`: 경로의 특정 세그먼트를 체계적으로 탐색합니다.
   - `search(pattern)`: 정규 표현식을 사용하여 직렬화된 경로에서 특정 패턴을 검색합니다.

3. **단기 기억(Short-Term Memory, STM)**: SAFARI는 조사 과정에서 수집된 정보를 유지하기 위해 STM을 사용합니다. STM은 조사 목표, 가설 및 실패 단계의 증거, 조사 계획 등을 저장합니다. 이 기억은 대화 이력의 끝에 추가되어, 이전의 도구 호출과 관찰을 보존합니다.

4. **검증 단계**: 충분한 증거가 수집되면, 조사자 에이전트는 가설을 세분화하여 각 주장에 대한 검증을 수행합니다. 이 과정에서 각 주장은 독립적으로 평가되며, 평가 결과에 따라 결론을 내리게 됩니다.

SAFARI는 Who&When 및 TRAIL 데이터셋을 통해 평가되었으며, 기존의 최첨단 방법들보다 20% 이상의 성능 향상을 보여주었습니다. 특히, SAFARI는 목표 결함이 모델의 기본 컨텍스트 창을 5배 초과하는 경우에도 0.58의 정밀도를 유지하여, 전통적인 평가자들이 완전히 실패하는 상황에서도 효과적으로 작동합니다.



SAFARI (Scaling Long Horizon Agentic Fault Attribution via Active Investigation) is a novel framework designed for diagnosing failures in autonomous agents performing complex multi-step and multi-agent tasks. Traditional methods load the entire execution trajectory into a Large Language Model (LLM)'s context window for diagnosis, which leads to attention dilution issues and fails when agentic traces inevitably exceed context limits. SAFARI addresses these challenges by replacing linear context loading with a tool-augmented diagnostic loop.

The key components of SAFARI are as follows:

1. **Investigator Agent**: This agent interacts with the given trajectory and collects evidence through iterative tool calls. At each investigation step, the agent observes the current conversation history and generates tool calls based on the tool schema.

2. **Tool Calls**: SAFARI employs two main tools:
   - `read(offset, limit)`: Systematically traverses segments of the trajectory.
   - `search(pattern)`: Executes case-insensitive regex queries over the serialized trajectory to find specific patterns.

3. **Short-Term Memory (STM)**: SAFARI uses STM to retain information gathered during the investigation process. The STM stores the investigation goals, hypotheses, evidence of failure steps, and investigative plans. This memory is appended at the end of the conversation history, preserving previous tool calls and observations.

4. **Verification Stage**: Once sufficient evidence is gathered, the Investigator Agent decomposes its hypothesis into atomic claims and performs verification for each claim. Each claim is evaluated independently, and based on the evaluation results, a conclusion is drawn.

SAFARI has been evaluated on the Who&When and TRAIL datasets, demonstrating over a 20% performance improvement compared to state-of-the-art methods. Notably, SAFARI maintains a precision of 0.58 even when the target fault resides 5 times beyond the model's native context window, effectively operating in scenarios where traditional evaluators fail completely.


<br/>
# Results



SAFARI는 다양한 테스트 데이터셋에서 경쟁 모델들과 비교하여 성능을 평가하였습니다. 주요 테스트 데이터셋으로는 Who&When과 TRAIL이 있으며, 이 두 데이터셋은 각각 다른 특성과 도전 과제를 가지고 있습니다.

1. **경쟁 모델**:
   - SAFARI는 여러 기존 모델과 비교되었습니다. 여기에는 단일 샷(single shot) 모델, 단계별(step-by-step) 모델, 그리고 RAFFLES와 같은 최신 모델이 포함됩니다. RAFFLES는 다단계 추론을 통해 결함을 정제하는 방법을 사용하지만, 여전히 모델의 기본 컨텍스트 창에 제한을 받습니다.

2. **테스트 데이터**:
   - **Who&When**: 이 데이터셋은 두 가지 하위 집합으로 나뉘며, 알고리즘 생성(Algorithm-Generated)과 수작업 생성(Hand-crafted)으로 구분됩니다. 수작업 생성 집합은 더 긴 경로 복잡성을 가지고 있으며, 더 많은 에이전트 단계와 토큰을 포함합니다.
   - **TRAIL**: 이 데이터셋은 긴 수평 에이전트 경로를 제공하며, 일부 경로는 2M 토큰을 초과합니다. TRAIL/SWE 벤치는 토큰 밀도가 높지만 단계 수는 적고, TRAIL/GAIA는 가장 긴 경로를 포함하고 있습니다.

3. **메트릭**:
   - SAFARI는 결정적 결함 속성(Decisive Fault Attribution)에서의 성능을 평가하기 위해 여러 메트릭을 사용했습니다. 여기에는 단계별 정확도(Step-level Accuracy)와 관용 정확도(Tolerant Accuracy)가 포함됩니다. TRAIL 데이터셋에서는 모든 결함이 레이블링되어 있지만, 단일 결정적 결함은 레이블링되지 않았기 때문에, 정밀도(Precision)와 엄격 정밀도(Strict Precision)라는 두 가지 특수 메트릭을 도입했습니다.

4. **비교 결과**:
   - SAFARI는 Who&When 데이터셋에서 1M 토큰 예산 내에서 기존 최첨단 결과보다 20% 향상된 성능을 보였고, TRAIL GAIA 하위 집합에서는 25K 토큰 예산 내에서 19% 향상된 성능을 기록했습니다. 특히, SAFARI는 모델의 기본 컨텍스트 창을 5배 초과하는 위치에 결함이 있을 때도 0.58의 정밀도를 유지했습니다. 이는 전통적인 평가자들이 완전히 실패하는 시나리오에서의 성과입니다.

이러한 결과는 SAFARI가 긴 수평 경로에서 결함을 효과적으로 식별할 수 있는 능력을 보여주며, 기존의 방법들이 직면한 한계를 극복할 수 있는 가능성을 제시합니다.

---




SAFARI was evaluated against various competitive models using different test datasets. The primary test datasets include Who&When and TRAIL, each presenting distinct characteristics and challenges.

1. **Competitive Models**:
   - SAFARI was compared with several existing models, including single-shot models, step-by-step models, and state-of-the-art models like RAFFLES. RAFFLES employs multi-turn reasoning to refine fault attribution but remains constrained by the model's native context window.

2. **Test Data**:
   - **Who&When**: This dataset is divided into two subsets: Algorithm-Generated and Hand-crafted. The Hand-crafted subset features significantly longer trajectory complexity with more agentic steps and tokens.
   - **TRAIL**: This dataset provides long-horizon agentic trajectories, with some traces exceeding 2M tokens. The TRAIL/SWE benchmark is token-dense despite having fewer steps, while TRAIL/GAIA contains the longest trajectories.

3. **Metrics**:
   - SAFARI employed several metrics to evaluate performance in Decisive Fault Attribution. These include Step-level Accuracy and Tolerant Accuracy. Since the TRAIL dataset labels all faults but not a single decisive fault, two specialized metrics—Precision and Strict Precision—were introduced.

4. **Comparison Results**:
   - SAFARI outperformed state-of-the-art results by 20% on the Who&When dataset within a 1M token budget and by 19% on the TRAIL GAIA subset on a 25K token budget. Most notably, SAFARI maintained a precision of 0.58 even when the target fault resided 5 times beyond the model's native context window, a scenario where traditional evaluators fail entirely.

These results demonstrate SAFARI's capability to effectively identify faults in long-horizon trajectories, presenting a potential solution to the limitations faced by existing methods.


<br/>
# 예제



SAFARI는 복잡한 다단계, 다중 에이전트 작업에서 에이전트의 실패를 진단하기 위한 새로운 프레임워크입니다. 이 시스템은 에이전트의 실행 경로를 분석하여 특정 실패 지점을 식별하는 데 중점을 두고 있습니다. SAFARI는 전통적인 방법이 아닌, 능동적인 조사(loop)를 통해 에이전트의 실패를 추적합니다. 이 과정에서 SAFARI는 에이전트의 실행 경로를 여러 단계로 나누어 분석하고, 각 단계에서 발생한 입력과 출력을 기반으로 오류를 식별합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**: 
   - **입력**: 에이전트의 실행 경로(예: `τ = {τ1, τ2, ..., τT}`)로 구성된 데이터. 각 단계 `τt`는 에이전트의 입력과 출력을 포함합니다.
   - **출력**: 각 단계에서 발생한 오류를 식별하는 것이 목표입니다. 예를 들어, 특정 단계에서 잘못된 입력이 주어졌거나, 에이전트가 잘못된 출력을 생성한 경우 이를 기록합니다.

2. **테스트 데이터**:
   - **입력**: 실제 환경에서 수집된 에이전트의 실행 경로. 이 경로는 다양한 복잡성을 가진 다단계 작업을 포함합니다.
   - **출력**: SAFARI가 특정 실패 지점을 정확히 식별할 수 있는지 평가합니다. 예를 들어, 에이전트가 특정 작업을 수행하는 동안 발생한 오류를 찾아내고, 그 오류가 발생한 최초의 단계(step_id)를 반환합니다.

#### 구체적인 테스크 예시
- **테스크**: 에이전트가 특정 웹 페이지에서 정보를 검색하는 작업을 수행한다고 가정합니다. 
  - **입력**: 
    - `τ1`: "웹 페이지를 열어라."
    - `τ2`: "검색어를 입력하라."
    - `τ3`: "결과를 클릭하라."
    - `τ4`: "정보를 읽어라."
  - **출력**: 
    - 만약 `τ3`에서 에이전트가 잘못된 링크를 클릭했다면, SAFARI는 `τ3`을 최초의 오류 단계로 식별하고, 그 이유를 설명합니다.




SAFARI is a new framework designed to diagnose agent failures in complex multi-step, multi-agent tasks. This system focuses on analyzing the execution trajectories of agents to identify specific failure points. Instead of traditional methods, SAFARI employs an active investigation loop to trace agent failures. In this process, SAFARI breaks down the agent's execution path into multiple steps and identifies errors based on the inputs and outputs that occur at each step.

#### Training Data and Test Data

1. **Training Data**: 
   - **Input**: Data consisting of the agent's execution trajectory (e.g., `τ = {τ1, τ2, ..., τT}`). Each step `τt` includes the agent's input and output.
   - **Output**: The goal is to identify errors that occur at each step. For example, if a specific step has incorrect input or the agent generates an incorrect output, this is recorded.

2. **Test Data**:
   - **Input**: Execution trajectories of agents collected from real-world environments. These trajectories include multi-step tasks with varying complexities.
   - **Output**: Evaluates whether SAFARI can accurately identify specific failure points. For instance, if the agent encounters an error while performing a specific task, it returns the step_id of the first step where the error occurred.

#### Specific Task Example
- **Task**: Assume the agent is performing a task to retrieve information from a specific web page.
  - **Input**: 
    - `τ1`: "Open the web page."
    - `τ2`: "Enter the search term."
    - `τ3`: "Click on the result."
    - `τ4`: "Read the information."
  - **Output**: 
    - If the agent clicks on the wrong link at `τ3`, SAFARI identifies `τ3` as the first error step and explains the reason for the failure.

<br/>
# 요약


SAFARI는 긴 수명의 에이전트 오류 귀속을 위한 능동적 조사 프레임워크로, 전체 경로를 LLM의 제한된 컨텍스트 창에 로드하는 대신 도구 보강 진단 루프를 사용하여 진단 정확도를 향상시킵니다. 실험 결과, SAFARI는 Who&When 데이터셋에서 20% 향상된 성능을 보였고, TRAIL GAIA 서브셋에서는 19% 향상된 성능을 기록했습니다. 특히, SAFARI는 목표 오류가 모델의 기본 컨텍스트 창을 5배 초과하는 경우에도 0.58의 정밀도를 유지하여 기존 평가자들이 완전히 실패하는 상황에서도 효과적으로 작동합니다.

---

SAFARI is an active investigation framework for long-horizon agentic fault attribution that enhances diagnostic accuracy by using a tool-augmented diagnostic loop instead of loading the entire trajectory into the LLM's limited context window. Experimental results show that SAFARI outperforms state-of-the-art methods by 20% on the Who&When dataset and by 19% on the TRAIL GAIA subset. Notably, SAFARI maintains a precision of 0.58 even when the target fault resides 5 times beyond the model's native context window, effectively operating in scenarios where traditional evaluators fail completely.

<br/>
# 기타



#### 다이어그램 및 피규어
SAFARI의 다이어그램은 Active Investigation 루프의 구조를 시각적으로 설명합니다. 이 루프는 Investigator Agent가 에이전트의 실행 경로를 분석하고, 가설을 생성하며, 도구를 사용하여 증거를 수집하는 과정을 보여줍니다. 이 구조는 전통적인 방법과의 차별성을 강조하며, SAFARI가 어떻게 긴 경로에서도 효과적으로 오류를 식별할 수 있는지를 설명합니다.

#### 테이블
1. **성능 비교 테이블**: SAFARI는 다양한 컨텍스트 예산에서 기존 방법들과 비교하여 성능을 보여줍니다. 예를 들어, TRAIL GAIA 데이터셋에서 SAFARI는 25K 토큰 예산에서 19% 더 높은 정밀도를 기록했습니다. 이는 SAFARI가 긴 경로에서도 효과적으로 오류를 식별할 수 있음을 나타냅니다.

2. **Ablation Study 결과**: STM(Short-Term Memory) 구성 요소의 유무에 따른 성능 차이를 보여줍니다. STM이 포함된 SAFARI는 모든 예산에서 성능이 향상되었으며, 이는 STM이 단순한 요약 이상의 역할을 한다는 것을 시사합니다.

#### 어펜딕스
어펜딕스에는 SAFARI의 작동 예시와 함께, Investigator Agent의 출력 예시가 포함되어 있습니다. 이 예시는 SAFARI가 어떻게 특정 오류를 식별하고, 다음 단계로 나아가는지를 보여줍니다. 또한, 각 단계에서의 의사결정 과정과 그에 따른 결과를 명확히 설명합니다.

---




#### Diagrams and Figures
The diagram of SAFARI visually explains the structure of the Active Investigation loop. This loop illustrates how the Investigator Agent analyzes the agent's execution trajectory, generates hypotheses, and uses tools to gather evidence. This structure emphasizes the differences from traditional methods and explains how SAFARI can effectively identify errors even in long trajectories.

#### Tables
1. **Performance Comparison Table**: SAFARI shows its performance compared to existing methods across various context budgets. For instance, on the TRAIL GAIA dataset, SAFARI achieved a 19% higher precision at a 25K token budget. This indicates that SAFARI can effectively identify errors even in long trajectories.

2. **Ablation Study Results**: This table shows the performance differences with and without the Short-Term Memory (STM) component. SAFARI with STM outperformed across all budgets, suggesting that STM plays a role beyond simple summarization.

#### Appendix
The appendix includes examples of SAFARI's operation, along with outputs from the Investigator Agent. These examples demonstrate how SAFARI identifies specific errors and progresses to the next steps. Additionally, it clearly explains the decision-making process at each stage and the resulting outcomes.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{zhu2026safari,
  author = {Chenyang Zhu and Jiayu Yao and Kushal Chawla and Youbing Yin and Nathan Wolfe and Pengshan Cai and Jingyu Wu and Spencer Hong and Sangwoo Cho and Shi-Xiong Zhang and Daben Liu and Sambit Sahu and Erin Babinsky},
  title = {SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation},
  booktitle = {Proceedings of the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML},
  year = {2026},
}
```

### 시카고 스타일 인용
Chenyang Zhu, Jiayu Yao, Kushal Chawla, Youbing Yin, Nathan Wolfe, Pengshan Cai, Jingyu Wu, Spencer Hong, Sangwoo Cho, Shi-Xiong Zhang, Daben Liu, Sambit Sahu, and Erin Babinsky. "SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation." In *Proceedings of the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML*, 2026. 
    