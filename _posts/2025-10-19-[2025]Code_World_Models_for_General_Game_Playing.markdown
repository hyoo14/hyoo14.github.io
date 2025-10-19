---
layout: post
title:  "[2025]Code World Models for General Game Playing"
date:   2025-10-19 19:20:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 대규모 언어 모델(LLM)을 사용하여 게임 규칙과 게임 경로를 파이썬 코드로 변환하는 "코드 월드 모델(CWM)"을 생성하는 방법을 제안합니다.

이전에 들은 임용 인터뷰 질문과 대답과 매우 유사하군.. 신기  


짧은 요약(Abstract) :


이 논문의 초록에서는 대형 언어 모델(LLM)의 추론 능력이 전통적인 보드 및 카드 게임에 점점 더 많이 적용되고 있지만, 현재의 주된 접근 방식인 직접적인 수 생성 요청이 몇 가지 단점이 있음을 지적합니다. 이 방법은 모델의 취약한 패턴 인식 능력에 의존하여 불법적인 수를 자주 발생시키고 전략적으로 얕은 플레이를 초래합니다. 이에 대한 대안으로, 저자들은 LLM을 사용하여 자연어 규칙과 게임 경로를 파이썬 코드로 표현된 형식적이고 실행 가능한 세계 모델로 변환하는 방법을 제안합니다. 이 생성된 모델은 상태 전이, 합법적인 수 열거 및 종료 확인을 위한 함수로 구성되어 있으며, 고성능 계획 알고리즘인 몬테카를로 트리 탐색(MCTS)의 검증 가능한 시뮬레이션 엔진 역할을 합니다. 저자들은 또한 LLM이 MCTS의 효율성을 높이기 위한 휴리스틱 가치 함수와 불완전 정보 게임에서 숨겨진 상태를 추정하기 위한 추론 함수를 생성하도록 요청합니다. 이 방법은 LLM을 정책으로 직접 사용하는 것에 비해 세 가지 뚜렷한 장점을 제공합니다: (1) 검증 가능성, (2) 전략적 깊이, (3) 일반화. 저자들은 이 에이전트를 10개의 서로 다른 게임에서 평가하였으며, 그 중 4개는 이 논문을 위해 새로 생성된 게임입니다. 결과적으로, 이 방법은 고려된 10개 게임 중 9개에서 Gemini 2.5 Pro를 초과하거나 동등한 성능을 보였습니다.



In the abstract of this paper, the authors point out that the reasoning abilities of large language models (LLMs) are increasingly being applied to traditional board and card games. However, the dominant approach—prompting for direct move generation—has significant drawbacks. This method relies on the model's fragile pattern-matching capabilities, leading to frequent illegal moves and strategically shallow play. As an alternative, the authors propose using the LLM to translate natural language rules and game trajectories into a formal, executable world model represented as Python code. This generated model consists of functions for state transition, legal move enumeration, and termination checks, serving as a verifiable simulation engine for high-performance planning algorithms like Monte Carlo tree search (MCTS). The authors also prompt the LLM to generate heuristic value functions to enhance the efficiency of MCTS and inference functions to estimate hidden states in imperfect information games. Their method offers three distinct advantages compared to directly using the LLM as a policy: (1) Verifiability, (2) Strategic Depth, and (3) Generalization. They evaluate their agent on 10 different games, of which 4 are novel and created for this paper. They find that their method outperforms or matches Gemini 2.5 Pro in 9 out of the 10 considered games.


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



이 논문에서 제안하는 메서드는 "코드 월드 모델(Code World Model, CWM)"을 생성하고 이를 통해 게임을 플레이하는 새로운 접근 방식을 제시합니다. 이 메서드는 대규모 언어 모델(LLM)을 활용하여 자연어로 표현된 게임 규칙과 게임 진행 데이터를 Python 코드로 변환하는 과정을 포함합니다. 이 과정은 다음과 같은 단계로 구성됩니다.

1. **게임 규칙 및 데이터 수집**: 새로운 게임을 접할 때, 에이전트는 먼저 몇 게임을 무작위 정책으로 플레이하여 게임의 진행 데이터를 수집합니다. 이 데이터는 관찰, 보상, 합법적인 행동 및 각 시간 단계의 상태를 포함합니다.

2. **코드 월드 모델 생성**: 수집된 게임 데이터와 게임 규칙의 텍스트 설명을 바탕으로 LLM을 사용하여 CWM을 학습합니다. CWM은 게임의 상태 정의, 합법적인 행동을 결정하는 함수, 상태 전이 함수, 관찰 함수, 보상 함수 및 종료 조건을 포함하는 실행 가능한 모델입니다.

3. **모델 정제**: 초기 CWM은 종종 정확하지 않기 때문에, 반복적인 정제 과정을 통해 모델의 품질을 향상시킵니다. 이 과정에서는 오프라인 게임 데이터에서 생성된 단위 테스트를 사용하여 CWM의 예측이 원래의 게임 진행 데이터와 일치하는지 확인합니다. 정제 과정은 LLM에 피드백을 제공하여 모델을 개선하는 데 도움을 줍니다.

4. **추론 함수 생성**: 불완전한 정보 게임의 경우, LLM을 사용하여 추론 함수를 생성합니다. 이 함수는 관찰된 데이터를 바탕으로 숨겨진 상태를 추정하는 데 사용됩니다. 이를 통해 에이전트는 게임 진행 중에 숨겨진 정보를 추정할 수 있습니다.

5. **가치 함수 생성**: CWM을 기반으로 MCTS(몬테카를로 트리 탐색) 및 ISMCTS(정보 집합 몬테카를로 트리 탐색)와 같은 계획 알고리즘의 효율성을 높이기 위해 LLM을 사용하여 가치 함수를 생성합니다. 이 함수는 각 상태에서의 예상 보상을 추정하는 데 사용됩니다.

6. **게임 플레이**: 최종적으로, 에이전트는 생성된 CWM을 사용하여 MCTS 또는 ISMCTS를 통해 게임을 플레이합니다. 이 과정에서 에이전트는 CWM의 정확성과 효율성을 바탕으로 최적의 행동을 선택합니다.

이러한 접근 방식은 LLM을 단순한 정책 생성기로 사용하는 대신, 게임의 규칙을 명확하게 정의하고 이를 기반으로 계획을 세우는 데 중점을 두고 있습니다. 이로 인해 에이전트는 새로운 게임에 대한 적응력이 높아지고, 전략적 깊이를 갖춘 플레이를 할 수 있게 됩니다.




The method proposed in this paper introduces a novel approach to generating a "Code World Model" (CWM) and using it to play games. This method leverages large language models (LLMs) to translate natural language descriptions of game rules and gameplay data into executable Python code. The process consists of the following steps:

1. **Game Rules and Data Collection**: When encountering a new game, the agent first plays a few games using a random policy to collect gameplay data. This data includes observations, rewards, legal actions, and states at each time step.

2. **Code World Model Generation**: Using the collected gameplay data and a textual description of the game rules, the LLM is employed to learn a CWM. The CWM comprises functions for defining the game state, determining legal actions, state transition functions, observation functions, reward functions, and termination checks.

3. **Model Refinement**: The initial CWM is often insufficiently accurate, so an iterative refinement process is employed to improve the model's quality. This process utilizes unit tests generated from offline gameplay data to verify that the CWM's predictions align with the original gameplay data. The refinement process provides feedback to the LLM to help improve the model.

4. **Inference Function Generation**: In the case of imperfect information games, the LLM is used to synthesize inference functions. These functions are used to estimate hidden states based on observed data, allowing the agent to infer hidden information during gameplay.

5. **Value Function Generation**: To enhance the efficiency of planning algorithms like Monte Carlo Tree Search (MCTS) and Information Set MCTS (ISMCTS), the LLM is used to generate value functions based on the CWM. This function estimates the expected rewards for each state.

6. **Gameplay**: Finally, the agent uses the generated CWM to play the game using MCTS or ISMCTS. In this process, the agent selects optimal actions based on the accuracy and efficiency of the CWM.

This approach emphasizes defining the rules of the game clearly and planning based on them, rather than simply using the LLM as a policy generator. As a result, the agent gains higher adaptability to new games and can engage in deeper strategic play.


<br/>
# Results



이 연구에서는 Code World Models (CWM) 접근 방식을 사용하여 다양한 게임에서의 성능을 평가했습니다. 실험은 10개의 서로 다른 게임에서 수행되었으며, 이 중 4개는 본 논문을 위해 새로 생성된 게임입니다. 실험의 주요 목표는 CWM을 통해 생성된 모델이 기존의 LLM 정책과 비교하여 얼마나 효과적인지를 평가하는 것이었습니다.

#### 경쟁 모델
CWM 접근 방식은 Gemini 2.5 Pro라는 최신 LLM 정책과 비교되었습니다. Gemini 2.5 Pro는 일반적인 LLM을 정책으로 사용하는 접근 방식으로, 각 단계에서 최적의 수를 선택하는 데 사용됩니다. CWM은 LLM을 사용하여 게임의 규칙과 경과를 코드로 변환하여 실행 가능한 모델을 생성하고, 이를 통해 Monte Carlo Tree Search (MCTS)와 같은 고성능 계획 알고리즘을 사용할 수 있도록 합니다.

#### 테스트 데이터
각 게임에 대해 5개의 오프라인 게임 플레이 데이터를 수집하여 CWM을 학습하는 데 사용했습니다. 이 데이터는 각 게임의 규칙과 함께 제공되었으며, CWM은 이 데이터를 기반으로 게임의 상태 전이, 합법적인 수, 종료 조건 등을 정의하는 함수를 생성했습니다.

#### 메트릭
성능 평가는 다음과 같은 메트릭을 사용하여 이루어졌습니다:
1. **전환 정확도 (Transition Accuracy)**: CWM이 생성한 모델이 실제 게임의 상태 전이를 얼마나 잘 예측하는지를 측정합니다.
2. **추론 정확도 (Inference Accuracy)**: CWM이 생성한 추론 함수가 주어진 관찰에 대해 얼마나 정확하게 상태를 추정하는지를 평가합니다.
3. **게임 플레이 성능**: CWM을 사용하여 생성된 에이전트가 Gemini 2.5 Pro 및 다른 경쟁 모델과의 대결에서 얼마나 잘 수행되는지를 측정합니다.

#### 비교 결과
CWM 접근 방식은 10개의 게임 중 9개에서 Gemini 2.5 Pro와 비교하여 우수한 성능을 보였습니다. 특히, 완전 정보 게임에서는 CWM이 모든 게임에서 높은 전환 정확도를 달성했으며, 불완전 정보 게임에서도 상당한 성과를 거두었습니다. CWM은 특히 새로운 게임에 대한 적응력이 뛰어나며, 기존 LLM 정책보다 더 깊이 있는 전략적 사고를 가능하게 했습니다.

결과적으로, CWM 접근 방식은 LLM을 단순한 정책으로 사용하는 것보다 더 나은 성능을 보여주었으며, 이는 게임 이론 및 인공지능 분야에서의 새로운 가능성을 제시합니다.

---




In this study, the Code World Models (CWM) approach was used to evaluate performance across various games. Experiments were conducted on 10 different games, of which 4 were newly created for this paper. The primary goal of the experiments was to assess how effectively the models generated through CWM performed compared to existing LLM policies.

#### Competing Models
The CWM approach was compared against the latest LLM policy, Gemini 2.5 Pro. Gemini 2.5 Pro employs a standard approach of using a general LLM as a policy, selecting optimal moves at each step. In contrast, CWM utilizes the LLM to translate the rules and trajectories of the game into executable code, allowing for the use of high-performance planning algorithms like Monte Carlo Tree Search (MCTS).

#### Test Data
For each game, 5 offline gameplay data points were collected and used to train the CWM. This data was provided alongside the rules of each game, and the CWM generated functions defining state transitions, legal moves, and termination conditions based on this data.

#### Metrics
Performance evaluation was conducted using the following metrics:
1. **Transition Accuracy**: Measures how well the CWM-generated model predicts actual game state transitions.
2. **Inference Accuracy**: Assesses how accurately the inference functions generated by the CWM estimate states given observations.
3. **Gameplay Performance**: Measures how well the agent using the CWM performs in matches against Gemini 2.5 Pro and other competing models.

#### Comparison Results
The CWM approach outperformed Gemini 2.5 Pro in 9 out of the 10 games considered. Notably, in perfect information games, the CWM achieved high transition accuracy across all games, while also demonstrating significant performance in imperfect information games. The CWM showed remarkable adaptability to new games and enabled deeper strategic thinking compared to existing LLM policies.

In conclusion, the CWM approach demonstrated superior performance compared to using LLMs as simple policies, highlighting new possibilities in the fields of game theory and artificial intelligence.


<br/>
# 예제



이 논문에서는 "Code World Models" (CWM)라는 새로운 접근 방식을 사용하여 다양한 게임을 플레이하는 에이전트를 훈련시키는 방법을 제안합니다. 이 방법은 대규모 언어 모델(LLM)을 활용하여 게임의 규칙과 플레이 데이터를 기반으로 실행 가능한 코드 형태의 세계 모델을 생성합니다. 이 모델은 게임의 상태 전이, 합법적인 이동 열거, 종료 체크 등의 기능을 포함합니다.

#### 훈련 데이터와 테스트 데이터의 예시

1. **훈련 데이터 (Training Data)**:
   - **게임 규칙 설명**: 예를 들어, "틱택토" 게임의 규칙을 설명하는 텍스트가 주어집니다. 이 규칙에는 플레이어가 어떻게 이동할 수 있는지, 승리 조건은 무엇인지 등이 포함됩니다.
   - **플레이 데이터**: 에이전트가 무작위로 플레이한 게임의 기록이 포함됩니다. 예를 들어, 다음과 같은 게임 진행이 있을 수 있습니다:
     - 플레이어 0: x(0,0) (첫 번째 행, 첫 번째 열에 'x'를 놓음)
     - 플레이어 1: o(0,1) (첫 번째 행, 두 번째 열에 'o'를 놓음)
     - 플레이어 0: x(1,1) (두 번째 행, 두 번째 열에 'x'를 놓음)
     - 플레이어 1: o(0,2) (첫 번째 행, 세 번째 열에 'o'를 놓음)
   - 이와 같은 데이터는 LLM이 게임의 규칙을 이해하고, 게임의 상태를 코드로 변환하는 데 사용됩니다.

2. **테스트 데이터 (Test Data)**:
   - **게임 규칙 설명**: 새로운 게임의 규칙이 주어집니다. 예를 들어, "연합 카드 게임"의 규칙이 설명됩니다.
   - **플레이 데이터**: 이 게임에 대한 새로운 플레이 데이터가 제공됩니다. 예를 들어, 플레이어 0이 "1,2,3"이라는 카드를 제안하고, 플레이어 1이 이를 수락하는 등의 기록이 포함됩니다.
   - 테스트 데이터는 에이전트가 새로운 게임에 대해 얼마나 잘 학습했는지를 평가하는 데 사용됩니다.

#### 구체적인 태스크 (Task)
- **태스크 1**: LLM이 주어진 게임 규칙을 기반으로 CWM을 생성하도록 요청합니다. 이 CWM은 게임의 상태 전이, 합법적인 이동, 종료 조건 등을 포함해야 합니다.
- **태스크 2**: 생성된 CWM을 사용하여 에이전트가 게임을 플레이하도록 하고, 그 성능을 평가합니다. 예를 들어, 에이전트가 100번의 게임을 플레이한 후 승률을 계산합니다.
- **태스크 3**: 새로운 게임에 대한 테스트 데이터를 사용하여 에이전트의 성능을 평가하고, 이전에 학습한 게임과의 비교를 통해 일반화 능력을 측정합니다.

---




This paper proposes a new approach using "Code World Models" (CWM) to train agents that can play various games. This method utilizes large language models (LLMs) to generate a formal, executable world model in the form of code based on the rules of the game and gameplay data. This model includes functions for state transition, legal move enumeration, and termination checks.

#### Example of Training Data and Test Data

1. **Training Data**:
   - **Game Rule Description**: For example, a text describing the rules of the "Tic-Tac-Toe" game is provided. This description includes how players can move, what the winning conditions are, etc.
   - **Gameplay Data**: Records of games played randomly by the agent are included. For instance, the following game progression might occur:
     - Player 0: x(0,0) (places 'x' in the first row, first column)
     - Player 1: o(0,1) (places 'o' in the first row, second column)
     - Player 0: x(1,1) (places 'x' in the second row, second column)
     - Player 1: o(0,2) (places 'o' in the first row, third column)
   - Such data is used for the LLM to understand the rules of the game and to convert the game states into code.

2. **Test Data**:
   - **Game Rule Description**: A new game's rules are provided. For example, the rules of a "Bargaining Card Game" are described.
   - **Gameplay Data**: New gameplay data for this game is provided. For instance, Player 0 might propose "1,2,3" and Player 1 accepts it.
   - The test data is used to evaluate how well the agent has learned about the new game.

#### Specific Tasks
- **Task 1**: Request the LLM to generate a CWM based on the given game rules. This CWM should include state transitions, legal moves, and termination conditions.
- **Task 2**: Use the generated CWM to have the agent play the game and evaluate its performance. For example, calculate the win rate after the agent plays 100 games.
- **Task 3**: Use the test data for the new game to evaluate the agent's performance and measure its generalization ability by comparing it with previously learned games.

<br/>
# 요약


이 논문에서는 대규모 언어 모델(LLM)을 사용하여 게임 규칙과 게임 경로를 파이썬 코드로 변환하는 "코드 월드 모델(CWM)"을 생성하는 방법을 제안합니다. 이 방법은 LLM의 패턴 인식 능력에 의존하지 않고, 검증 가능한 시뮬레이션 엔진을 통해 전략적 깊이를 더하며, 새로운 게임에 대한 일반화 능력을 향상시킵니다. 실험 결과, 제안된 방법이 10개의 다양한 게임에서 Gemini 2.5 Pro를 초과하거나 동등한 성능을 보였습니다.

---

This paper introduces a method for generating a "Code World Model" (CWM) by translating game rules and trajectories into Python code using large language models (LLMs). This approach enhances strategic depth through a verifiable simulation engine, avoiding reliance on the LLM's pattern recognition capabilities, and improves generalization to new games. Experimental results show that the proposed method outperforms or matches Gemini 2.5 Pro across 10 different games.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - 논문에서는 다양한 게임의 성능을 비교하는 그래프와 다이어그램이 포함되어 있습니다. 이들은 CWM 기반 에이전트와 기존 LLM 정책의 성능을 시각적으로 나타내며, 특히 CWM이 다양한 게임에서 어떻게 더 나은 성능을 발휘하는지를 보여줍니다. 예를 들어, CWM-MCTS 에이전트가 Gemini 2.5 Pro보다 우수한 성능을 보인 게임들이 강조됩니다. 이러한 시각적 자료는 독자가 결과를 쉽게 이해하고 비교할 수 있도록 도와줍니다.

2. **테이블**:
   - 테이블은 각 게임에서의 성능 지표를 정리하여 보여줍니다. 예를 들어, 각 게임의 승률, 패배율, 무승부율 등을 포함하여 CWM-MCTS와 Gemini 2.5 Pro의 성능을 비교합니다. 이러한 데이터는 CWM의 효과성을 수치적으로 입증하며, 특히 새로운 게임에서의 성능을 강조합니다. 테이블을 통해 독자는 각 게임의 특성과 에이전트의 성능을 한눈에 파악할 수 있습니다.

3. **어펜딕스**:
   - 어펜딕스에서는 실험에 사용된 게임의 규칙, CWM의 세부 구현, 그리고 각 게임의 코드 예시가 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 이 방법론을 적용할 수 있도록 돕습니다. 또한, 어펜딕스는 CWM의 설계 및 구현 과정에서의 세부 사항을 제공하여, 독자가 이 연구의 기초를 이해하는 데 도움을 줍니다.

### Insights
- **CWM의 장점**: CWM은 LLM을 정책으로 사용하는 것보다 더 높은 성능을 보여주며, 특히 새로운 게임에 대한 적응력이 뛰어납니다. 이는 CWM이 게임의 규칙을 명확하게 정의하고, 이를 기반으로 계획을 세울 수 있도록 돕기 때문입니다.
- **전략적 깊이**: CWM은 LLM의 의미적 이해와 고전적 계획 알고리즘의 깊은 탐색 능력을 결합하여, 더 전략적인 플레이를 가능하게 합니다. 이는 특히 복잡한 게임에서 두드러집니다.
- **일반화 능력**: CWM은 새로운 게임에 대한 적응력이 뛰어나며, 이는 LLM이 데이터에서 코드로 변환하는 메타 작업에 집중할 수 있도록 유도하기 때문입니다. 이는 다양한 게임 환경에서의 성능 향상으로 이어집니다.

---




1. **Diagrams and Figures**:
   - The paper includes graphs and diagrams that compare the performance of various games. These visually represent the performance of the CWM-based agent against existing LLM policies, particularly highlighting how CWM outperforms in various games. For instance, the superior performance of the CWM-MCTS agent over Gemini 2.5 Pro in certain games is emphasized. Such visual materials help readers easily understand and compare the results.

2. **Tables**:
   - Tables summarize performance metrics for each game, including win rates, loss rates, and draw rates, comparing CWM-MCTS with Gemini 2.5 Pro. This data quantitatively proves the effectiveness of CWM, especially in new games. Through the tables, readers can grasp the characteristics of each game and the performance of the agents at a glance.

3. **Appendix**:
   - The appendix includes the rules of the games used in the experiments, detailed implementations of CWM, and code examples for each game. This enhances the reproducibility of the research and assists other researchers in applying this methodology. Additionally, the appendix provides details on the design and implementation process of CWM, aiding readers in understanding the foundation of this research.

### Insights
- **Advantages of CWM**: CWM demonstrates higher performance than using LLM as a policy, particularly showing adaptability to new games. This is due to CWM's clear definition of game rules, allowing for effective planning.
- **Strategic Depth**: CWM combines the semantic understanding of LLM with the deep search capabilities of classical planning algorithms, enabling more strategic play, especially in complex games.
- **Generalization Ability**: CWM's focus on the meta-task of translating data to code allows it to adapt more easily to new games, leading to performance improvements across various game environments.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@article{Lehrach2025,
  author = {Wolfgang Lehrach and Daniel Hennes and Miguel Lázaro-Gredilla and Xinghua Lou and Carter Wendelken and Zun Li and Antoine Dedieu and Jordi Grau-Moya and Marc Lanctot and Atil Iscen and John Schultz and Marcus Chiam and Ian Gemp and Piotr Zielinski and Satinder Singh and Kevin P. Murphy},
  title = {Code World Models for General Game Playing},
  journal = {arXiv preprint arXiv:2510.04542},
  year = {2025},
  url = {https://arxiv.org/abs/2510.04542}
}
```

### 시카고 스타일 인용
Lehrach, Wolfgang, Daniel Hennes, Miguel Lázaro-Gredilla, Xinghua Lou, Carter Wendelken, Zun Li, Antoine Dedieu, Jordi Grau-Moya, Marc Lanctot, Atil Iscen, John Schultz, Marcus Chiam, Ian Gemp, Piotr Zielinski, Satinder Singh, and Kevin P. Murphy. 2025. "Code World Models for General Game Playing." arXiv preprint arXiv:2510.04542. https://arxiv.org/abs/2510.04542.
