---
layout: post
title:  "[2024]ReTA: Recursively Thinking Ahead to Improve the Strategic Reasoning of Large Language Models"  
date:   2024-08-27 14:12:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


ReTA는 LLM으로 게임 행동 묘사를 생성하고, 도메인 지식과 게임 이론에 기반한 보상 함수를 통해 해당 행동이 게임에서 얼마나 유리한지 점수를 계산합니다. 이를 통해 각 행동에 대해 시뮬레이션을 실행하고, 미래의 결과에 따라 보상 신호를 할당하여 최적의 행동을 선택하게 합니다.


짧은 요약(Abstract) :    



이 논문은 대형 언어 모델(LLM)의 다중 턴 전략적 추론 능력을 분석하는 연구입니다. 기존에는 주로 단일 턴에서의 논리적 문제 해결 능력만 평가되었지만, 이 연구에서는 보드 게임(Tic-Tac-Toe, Connect-4)과 포커 같은 텍스트 기반 게임을 통해 다중 턴에서의 전략적 사고 능력을 평가했습니다. 실험 결과, 기존의 LLM과 추론 방법들은 전략적 추론에서 효과적이지 않다는 결론을 도출했습니다. 이를 극복하기 위해 연구진은 **ReTA(Recursively Thinking-Ahead)**라는 에이전트를 제안했습니다. ReTA는 **재귀적 프롬프팅**을 통해 상대방의 미래 행동을 예측하고, 각 상황에 대한 보상 신호를 계산하여 더 나은 전략적 결정을 내리도록 설계되었습니다. 실험 결과, ReTA는 기존의 최첨단 방법들보다 더 나은 성능을 보였습니다.


This paper analyzes the multi-turn strategic reasoning capabilities of Large Language Models (LLMs), focusing on environments such as board games (e.g., Tic-Tac-Toe, Connect-4) and poker games. Existing LLM evaluations have mainly focused on single-turn and static tasks like arithmetic, but this work extends the evaluation to multi-turn, strategy-driven tasks. The experiments show that current state-of-the-art LLMs and reasoning frameworks are largely ineffective in handling strategic reasoning. To address these limitations, the authors propose **ReTA (Recursively Thinking-Ahead)**, a simple yet effective agent that uses recursive prompting to predict opponents' future actions and assign reward signals to improve strategic reasoning. The results demonstrate that ReTA significantly outperforms existing methods in multi-turn reasoning scenarios.


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




### 1. **LLM 기반 행동 생성**:
ReTA는 **대형 언어 모델(LLM)**을 사용하여 현재 게임 상태에 맞는 여러 행동을 묘사하고 예측합니다. LLM은 각 게임 상황에서 가능한 행동들을 생성하며, 여기에는 게임의 규칙에 맞춘 다양한 전략적 선택들이 포함됩니다. 예를 들어, Tic-Tac-Toe 게임에서 특정 위치에 말을 두는 행동이나, 포커 게임에서 베팅을 올리는 행동 등이 LLM을 통해 텍스트로 생성됩니다.

### 2. **도메인 지식과 게임 이론에 기반한 보상 함수**:
ReTA는 생성된 행동에 대한 평가를 위해 **보상 함수(reward function)**를 사용합니다. 이 보상 함수는 각 행동이 게임에서 얼마나 유리한지를 계산하는데, 이를 위해 **도메인 지식**과 **게임 이론**이 활용됩니다. 
- **도메인 지식**: 각 게임의 규칙과 전략적 요소에 따라 보상 함수를 정의합니다. 예를 들어, Tic-Tac-Toe에서는 상대방의 승리를 막는 행동이 높은 보상을 받을 수 있고, 포커에서는 상대방의 베팅 패턴을 고려한 최적의 베팅 전략이 높은 보상을 받습니다.
- **게임 이론**: **최소-최대(minimax) 전략**과 같은 게임 이론적 접근을 사용하여 상대방이 최적의 선택을 할 것이라고 가정한 상황에서 자신에게 가장 유리한 행동을 선택하도록 합니다.

### 3. **미래 결과 시뮬레이션**:
ReTA는 LLM이 생성한 행동을 기반으로 **시뮬레이션**을 실행합니다. 시뮬레이션을 통해 현재 행동이 몇 턴 후의 게임 상태에 어떻게 영향을 미칠지를 예측하고, 각 행동이 승리, 패배, 또는 무승부로 이어질 가능성을 평가합니다.

### 4. **보상 신호 할당과 행동 선택**:
시뮬레이션 결과를 바탕으로 각 행동에 **보상 신호(reward signal)**를 할당합니다. 보상 신호는 게임에서의 유리한 상태를 얼마나 잘 만들어낼 수 있는지를 나타냅니다. 이 보상 신호는 최종 결과에서 역으로 현재 행동까지 **되돌려(backtracking)** 계산되며, 보상이 가장 높은 행동이 선택됩니다.

### 사용하는 요소:
- **LLM (Large Language Model)**: 각 게임 상황에서 가능한 행동을 자연어로 생성하는 데 사용됩니다.
- **도메인 지식**: 게임의 규칙과 전략을 반영한 보상 함수를 정의하는 데 사용됩니다.
- **게임 이론 (minimax 전략)**: 상대방의 최적 행동을 가정하여, 그 상황에서도 최선의 결정을 내리는 전략적 선택을 가능하게 합니다.
- **시뮬레이션**: 미래 턴에서 발생할 수 있는 결과를 예측하여 보상 신호를 할당합니다.

### 전체 요약:
ReTA는 LLM을 사용하여 게임 행동을 묘사하고, 도메인 지식과 게임 이론에 기반한 보상 함수를 통해 각 행동을 평가합니다. 이후 시뮬레이션을 통해 행동의 결과를 예측하고, 보상 신호를 통해 최적의 행동을 선택하여 전략적 결정을 내리도록 설계되었습니다.

---



### 1. **Action Generation via LLM**:
ReTA uses a **Large Language Model (LLM)** to generate action descriptions for the current game state. The LLM generates various strategic options based on the rules of the game, such as placing a mark in Tic-Tac-Toe or raising a bet in poker.

### 2. **Reward Function Based on Domain Knowledge and Game Theory**:
To evaluate the generated actions, ReTA employs a **reward function**. This function assesses how advantageous each action is, based on **domain knowledge** and **game theory**:
- **Domain Knowledge**: The rules and strategies specific to each game are used to define the reward function. For instance, blocking the opponent in Tic-Tac-Toe or betting optimally in poker would yield higher rewards.
- **Game Theory**: **Minimax strategies** are employed, assuming that the opponent will make optimal moves. ReTA selects the action that maximizes its advantage even under the worst-case scenario.

### 3. **Future Outcome Simulation**:
ReTA runs **simulations** based on the generated actions, predicting how each move will influence the game in future turns. The simulation evaluates the likelihood of winning, losing, or drawing based on each action.

### 4. **Assigning Reward Signals and Action Selection**:
Each action is assigned a **reward signal** based on the simulation results. The reward reflects how well the action leads to a favorable state in the game. This signal is **backtracked** from the predicted future state to the current action, and the action with the highest reward is selected.

### Key Components:
- **LLM (Large Language Model)**: Used to generate natural language descriptions of possible actions.
- **Domain Knowledge**: Utilized to define the reward function based on game rules and strategies.
- **Game Theory (Minimax Strategy)**: Ensures that the chosen action is optimal even if the opponent plays perfectly.
- **Simulation**: Predicts future game outcomes to assign reward signals.

### Summary:
ReTA leverages LLM to generate game actions and evaluates these actions using a reward function based on domain knowledge and game theory. By running simulations of future outcomes and backtracking reward signals, it selects the optimal action for strategic decision-making.

<br/>
# Results  




### 1. **사용된 언어 모델**:
ReTA는 **GPT-3.5-turbo**와 같은 상용 LLM뿐만 아니라 **Llama-2-chat**와 같은 오픈소스 LLM을 사용하여 테스트되었습니다. 특히, 전략적 추론을 위해 다양한 LLM 기반 에이전트를 평가했으며, 각 모델은 ReTA와 함께 **Chain-of-Thought (CoT)**, **Self-Consistent CoT (CoT-SC)**, **Tree-of-Thought (ToT)** 및 **ReAct** 에이전트와 같은 기존의 최첨단 추론 기법들과 비교되었습니다.

### 2. **평가된 게임**:
ReTA는 다음과 같은 게임 환경에서 테스트되었습니다:
- **Tic-Tac-Toe**
- **Connect-4**
- **Texas Hold’em Poker**

이 게임들은 완전 정보(모든 참가자가 게임 상태를 알고 있음) 및 불완전 정보(참가자가 모든 정보를 알 수 없음) 시나리오에서 사용되어, 다중 턴 전략적 추론을 평가했습니다.

### 3. **평가 지표**:
성능 평가 지표는 주로 각 에이전트가 얼마나 많은 **승리 비율(Win Ratio)**과 **남은 칩(포커 게임의 경우)**을 보유하는지에 따라 결정되었습니다. Tic-Tac-Toe와 같은 게임에서는 각 에이전트가 100번의 매치를 실행하여, 승리와 패배를 계산했고, Texas Hold'em Poker에서는 게임당 손 승리 비율을 사용했습니다.

### 4. **결과 요약**:
- **Tic-Tac-Toe**: ReTA는 다양한 LLM을 사용해 평가된 결과, **ToT(Tree of Thought)** 및 **CoT**와 같은 기존 에이전트보다 높은 승리 비율을 기록했습니다. 예를 들어, GPT-3.5-turbo와 함께 사용된 ReTA는 ToT에 비해 **30% 더 높은 승리 비율**(61% 대 31%)을 보였습니다.
- **Connect-4**: ReTA는 이 게임에서도 **ToT** 및 **ReAct** 에이전트보다 높은 성능을 보여, **20% 이상의 승리 비율** 향상을 기록했습니다.
- **Texas Hold’em Poker**: 포커 게임에서 ReTA는 **손 승리 비율**에서 특히 높은 성과를 보였으며, **Prompt 에이전트**에 비해 7.6%, **CoT-SC**에 비해 26.4%, 그리고 **ToT**에 비해 44.2% 더 높은 승리 비율을 기록했습니다. ReTA는 이러한 게임 환경에서 **최종 칩 보유량**에서도 다른 에이전트들보다 뛰어난 성과를 보였습니다.

### 5. **주요 성과**:
- ReTA는 **최적화 기반 솔버(예: MinMax 에이전트)**보다는 아직 부족할 수 있지만, **기존의 최첨단 추론 기법들보다 높은 승리 비율**과 **더 나은 전략적 추론**을 제공하는 것으로 나타났습니다.
- ReTA는 상용 LLM과 오픈소스 LLM 모두에서 **높은 성능을 유지**했으며, 특히 GPT-3.5와 Llama-2-chat과 함께 사용할 때, 각 게임에서 **대폭적인 승리 비율 향상**을 보여주었습니다.

---


### 1. **Language Models Used**:
ReTA was tested using both commercial models like **GPT-3.5-turbo** and open-source models like **Llama-2-chat**. The evaluations included comparisons with state-of-the-art reasoning agents such as **Chain-of-Thought (CoT)**, **Self-Consistent CoT (CoT-SC)**, **Tree-of-Thought (ToT)**, and **ReAct**.

### 2. **Games Evaluated**:
ReTA was tested on games that require both complete information and incomplete information strategies, such as:
- **Tic-Tac-Toe**
- **Connect-4**
- **Texas Hold’em Poker**

These games were used to assess multi-turn strategic reasoning in different contexts.

### 3. **Evaluation Metrics**:
The key performance metrics included:
- **Win ratio**: The percentage of games won out of a set number of matches.
- **Remaining chips** (for poker): The number of chips left after each match.

In Tic-Tac-Toe, each agent was tested over 100 matches, while in Texas Hold'em Poker, the win ratio per hand was used as the metric.

### 4. **Results Summary**:
- **Tic-Tac-Toe**: ReTA outperformed other agents like **ToT** and **CoT** in win ratios. For instance, using GPT-3.5-turbo, ReTA achieved a **30% higher win ratio** compared to ToT (61% vs. 31%).
- **Connect-4**: ReTA also showed superior performance, with more than a **20% improvement in win ratio** compared to ToT and ReAct agents.
- **Texas Hold’em Poker**: In poker, ReTA demonstrated particularly strong results, achieving a **7.6% higher hand win ratio** than the Prompt agent, **26.4% higher than CoT-SC**, and **44.2% higher than ToT**. ReTA also performed better in terms of **final chip count** compared to other agents.

### 5. **Key Achievements**:
- While ReTA may still fall short compared to optimization-based solvers like the **MinMax agent**, it significantly outperformed state-of-the-art reasoning methods, achieving **higher win ratios** and **better strategic reasoning**.
- ReTA maintained high performance across both commercial (GPT-3.5) and open-source (Llama-2-chat) LLMs, showing **notable improvements in win ratios** in all games tested.


<br/>
# 예시  




### 게임 플레이 예시 (Tic-Tac-Toe):

#### 1. **게임 상태**:
- 현재 보드: 
  ```
  O |   |   
  ---------
    | X |   
  ---------
    |   |   
  ```
- ReTA는 'X' 플레이어이고, 상대방이 'O'를 놓았습니다. 현재 ReTA가 둘 차례입니다.

#### 2. **LLM이 생성한 행동 묘사**:
ReTA의 LLM은 다음과 같이 행동을 예측하고, 각 행동에 대해 시뮬레이션을 진행합니다. 예를 들어, ReTA가 생각할 수 있는 행동 후보는 다음과 같습니다.

- **생성된 행동 1**: "C1R3에 X를 놓습니다."
  - **LLM이 생성한 문장**: "If I place an 'X' in the bottom left corner (C1R3), I can block the opponent from winning diagonally in future turns."
  - **보상 평가**: 이 행동은 상대방의 다음 턴에서 승리를 막을 수 있기 때문에 높은 보상을 받습니다.

- **생성된 행동 2**: "C3R2에 X를 놓습니다."
  - **LLM이 생성한 문장**: "Placing an 'X' in the middle right position (C3R2) allows me to set up a potential winning move horizontally."
  - **보상 평가**: 이 행동은 다음 턴에서 이길 기회를 만들 수 있어 높은 보상을 받습니다.

#### 3. **시뮬레이션 및 보상 평가**:
ReTA는 각 행동에 대한 시뮬레이션을 실행하여 상대방이 어떻게 반응할지를 예측합니다. 예를 들어, "C1R3"에 'X'를 놓을 경우 상대방이 어떤 반응을 할 수 있을지에 대해 LLM이 예측한 시나리오는 다음과 같습니다.

- **예측된 상대방 행동**: "상대방은 C1R2에 'O'를 놓습니다."
  - 이 경우, 상대방의 승리를 막았기 때문에 **높은 보상 신호**를 할당받습니다.
- **예측된 최종 상태**: 상대방이 다음 턴에서 이길 기회를 막는 것이 중요하므로, 이 행동이 유리한 것으로 평가됩니다.

#### 4. **최종 행동 선택**:
시뮬레이션 결과 "C1R3"에 'X'를 놓는 것이 상대방의 승리를 차단할 수 있는 최적의 행동으로 평가되며, ReTA는 이 행동을 선택합니다.

```
최종 행동: "C1R3에 X를 놓습니다."
```

### 게임 진행 후 결과:
보드는 다음과 같이 업데이트됩니다:
```
O |   |   
---------
  | X |   
---------
X |   |   
```
이후, 상대방은 중앙 좌측(C1R2)에 'O'를 놓고 게임이 계속됩니다.

### LLM이 생성한 문장들 요약:
- "If I place an 'X' in the bottom left corner (C1R3), I can block the opponent from winning diagonally in future turns."
- "Placing an 'X' in the middle right position (C3R2) allows me to set up a potential winning move horizontally."

이러한 문장들은 게임 상황에 맞춘 LLM의 자연어 생성 예시이며, 각각의 행동이 게임에서 유리한지 아닌지에 대한 판단을 시뮬레이션과 보상 신호를 통해 결정하게 됩니다.

---


### Example of Gameplay (Tic-Tac-Toe):

#### 1. **Game State**:
- Current board:
  ```
  O |   |   
  ---------
    | X |   
  ---------
    |   |   
  ```
- ReTA is the 'X' player, and the opponent has placed 'O'. It is ReTA's turn.

#### 2. **LLM-Generated Action Descriptions**:
ReTA’s LLM generates several possible actions based on the current game state. Some examples of action candidates are:

- **Generated Action 1**: "Place an X in C1R3 (bottom-left corner)."
  - **LLM's Generated Sentence**: "If I place an 'X' in the bottom left corner (C1R3), I can block the opponent from winning diagonally in future turns."
  - **Reward Evaluation**: This action receives a high reward as it prevents the opponent from winning.

- **Generated Action 2**: "Place an X in C3R2 (middle-right position)."
  - **LLM's Generated Sentence**: "Placing an 'X' in the middle right position (C3R2) allows me to set up a potential winning move horizontally."
  - **Reward Evaluation**: This action is also favorable as it sets up a potential winning move for the next turn.

#### 3. **Simulation and Reward Evaluation**:
ReTA simulates the outcomes of each action by predicting how the opponent might respond. For example, if 'X' is placed in C1R3, the LLM might predict:

- **Predicted Opponent Action**: "The opponent places an 'O' in C1R2."
  - This prevents the opponent from winning, so this action receives a **high reward signal**.
- **Final Outcome**: Since this move blocks the opponent's potential win, it is evaluated as advantageous.

#### 4. **Final Action Selection**:
Based on the simulation results, placing 'X' in C1R3 is evaluated as the optimal move, and ReTA selects this action.

```
Final Action: "Place an X in C1R3."
```

### Game Progress:
The board is updated as follows:
```
O |   |   
---------
  | X |   
---------
X |   |   
```
The game continues with the opponent placing 'O' in C1R2.

### Summary of LLM-Generated Sentences:
- "If I place an 'X' in the bottom left corner (C1R3), I can block the opponent from winning diagonally in future turns."
- "Placing an 'X' in the middle right position (C3R2) allows me to set up a potential winning move horizontally."

These sentences are examples of how the LLM generates natural language descriptions of possible actions, which are then evaluated through simulation and reward signals to make optimal decisions.


<br/>  
# 요약 



ReTA는 대형 언어 모델(LLM)을 사용해 게임 상황에 맞는 행동을 생성하고, 도메인 지식과 게임 이론에 기반한 보상 함수로 각 행동의 유리함을 평가합니다. 행동들은 시뮬레이션을 통해 미래 결과를 예측한 후 보상 신호를 통해 평가되며, 최적의 행동을 선택합니다. ReTA는 GPT-3.5-turbo 및 Llama-2-chat을 사용해 Tic-Tac-Toe, Connect-4, Texas Hold'em Poker에서 평가되었고, 기존 방법들보다 높은 승리 비율을 기록했습니다. 예를 들어, Tic-Tac-Toe에서 30% 더 높은 승리 비율을 보였으며, 포커에서는 ToT와 비교해 44.2% 더 높은 손 승리 비율을 보였습니다. ReTA는 전략적 추론에 있어 기존의 최첨단 방법들보다 우수한 성능을 발휘했습니다.

---


ReTA uses Large Language Models (LLMs) to generate game-specific actions and evaluates them based on domain knowledge and game theory via a reward function. The actions are assessed through simulations that predict future outcomes, and reward signals are used to select the optimal move. ReTA, tested with GPT-3.5-turbo and Llama-2-chat, demonstrated superior performance in games like Tic-Tac-Toe, Connect-4, and Texas Hold'em Poker, achieving higher win ratios than existing methods. For example, it achieved a 30% higher win ratio in Tic-Tac-Toe and a 44.2% higher hand win ratio in poker compared to ToT. ReTA significantly outperformed state-of-the-art reasoning approaches in multi-turn strategic reasoning.

# 기타  

음.. 몇 편 요약하면서 느낀거는 멀티합? 멀티 턴을 좀 주의깊게 보고 여기에서 뭔가를 하는 느낌이고  
그 턴과 턴 사이에 있는 status, 하나의 턴을 주목하는 듯    
그 status를 평가하여 더 나은 선택을 하게 하거나.. 아니면 뭐 이러한 멀티 status한 걸 데이터셋으로 제공하거나...  


<br/>
# refer format:     
@inproceedings{duan2024reta,
  title={ReTA: Recursively Thinking Ahead to Improve the Strategic Reasoning of Large Language Models},
  author={Duan, Jinhao and Wang, Shiqi and Diffenderfer, James and Sun, Lichao and Chen, Tianlong and Kailkhura, Bhavya and Xu, Kaidi},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={2232--2246},
  year={2024},
  organization={Association for Computational Linguistics}
}


다음은 시카고 스타일로 작성된 참고 문헌입니다:

Duan Jinhao, Shiqi Wang, James Diffenderfer, Lichao Sun, Tianlong Chen, Bhavya Kailkhura, and Kaidi Xu. "ReTA: Recursively Thinking Ahead to Improve the Strategic Reasoning of Large Language Models." In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, 2232-2246. Association for Computational Linguistics, 2024.

