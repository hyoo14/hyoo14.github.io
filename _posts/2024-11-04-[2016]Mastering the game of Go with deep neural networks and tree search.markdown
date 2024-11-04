---
layout: post
title:  "[2016]Mastering the game of Go with deep neural networks and tree search"  
date:   2024-11-04 00:55:30 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


짧은 요약(Abstract) :    

### 한글 요약 (Abstract)

고전 게임 중 하나인 바둑은 인공지능 분야에서 가장 해결하기 어려운 문제로 여겨져 왔습니다. 이는 바둑의 방대한 탐색 공간과 복잡한 보드 위치 및 수를 평가하는 어려움 때문입니다. 이 논문은 보드의 상태를 평가하는 '가치 네트워크'와 수를 선택하는 '정책 네트워크'를 이용한 새로운 바둑 컴퓨터 접근법을 제시합니다. 이러한 딥 신경망은 인간 전문가의 대국으로부터 지도 학습을 하고, 자가 대국을 통한 강화 학습을 결합하여 학습됩니다. 이 신경망은 추가적인 예측 탐색 없이도 수천 번의 무작위 자가 대국을 시뮬레이션하는 기존 몬테카를로 트리 탐색 프로그램과 유사한 수준의 바둑 실력을 보여줍니다. 이 연구에서 제안된 탐색 알고리즘을 통해, AlphaGo는 다른 바둑 프로그램을 상대로 99.8%의 승률을 달성했고, 유럽 바둑 챔피언을 상대로 5대 0으로 승리했습니다. 이는 컴퓨터가 처음으로 인간 프로 바둑 선수를 상대로 전체 바둑판에서 승리를 거둔 사례로, 이전에는 최소한 10년 뒤에나 가능할 것이라 예측된 성과입니다.  

---

### English Abstract

The game of Go has long been viewed as one of the most challenging classic games for artificial intelligence due to its vast search space and the difficulty of evaluating board positions and moves. This paper introduces a new approach to computer Go using 'value networks' to evaluate board positions and 'policy networks' to select moves. These deep neural networks are trained by a novel combination of supervised learning from human expert games and reinforcement learning from games of self-play. Without any lookahead search, the neural networks play Go at a level comparable to state-of-the-art Monte Carlo tree search programs that simulate thousands of random games of self-play. With this search algorithm, AlphaGo achieved a 99.8% winning rate against other Go programs and defeated the human European Go champion by 5 games to 0. This is the first time a computer program has defeated a human professional player in the full-sized game of Go, a feat previously thought to be at least a decade away.   



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




이 논문에서 AlphaGo가 바둑을 두기 위해 사용한 주요 메서드들은 크게 세 가지로 나뉩니다: 정책 네트워크, 가치 네트워크, 그리고 몬테카를로 트리 탐색(MCTS) 알고리즘입니다.

1. **정책 네트워크**: AlphaGo의 정책 네트워크는 주어진 바둑판 상태에서 가장 가능성 높은 다음 수를 예측하도록 학습되었습니다. 이는 인공지능이 인간 전문가들의 기보를 통해 지도 학습을 하도록 설계된 것입니다. 정책 네트워크는 각 위치에서의 유효한 수에 대해 확률 분포를 제공하여 가능한 행동을 평가하고 선택하는 데 도움을 줍니다.

2. **가치 네트워크**: 가치 네트워크는 주어진 상태에서 바둑판의 승패 결과를 예측하는 역할을 합니다. 가치 네트워크는 자가 대국을 통해 강화 학습을 수행하며, 각 상태가 종국에 미치는 영향을 평가합니다. 이를 통해 AlphaGo는 위치별로 보드를 평가하고, 승리 확률이 높은 수를 두는 방식을 발전시켰습니다.

3. **몬테카를로 트리 탐색 (MCTS)**: MCTS는 AlphaGo가 전체적인 시뮬레이션을 통해 다음 최선의 수를 결정할 수 있도록 해줍니다. 이 과정에서 정책 네트워크는 수를 선택하는 우선순위를 정하고, 가치 네트워크는 시뮬레이션된 결과를 바탕으로 그 수의 가치를 평가하여 결정합니다. 이 탐색 알고리즘을 통해 AlphaGo는 인간과 유사하게 직관적이면서도 계산된 수를 둘 수 있었습니다.

---



The primary methods used in this paper to enable AlphaGo to play Go are divided into three main components: the policy network, value network, and Monte Carlo Tree Search (MCTS) algorithm.

1. **Policy Network**: AlphaGo's policy network is trained to predict the most probable next moves given a board state. This was achieved through supervised learning on human expert games. The policy network provides a probability distribution over legal actions at each position, assisting in evaluating and selecting possible moves.

2. **Value Network**: The value network estimates the outcome of the game from a given board position. It is trained through reinforcement learning from self-play games to assess the influence of each position on the game's final outcome. This enables AlphaGo to evaluate board states and develop moves that have a high probability of leading to a win.

3. **Monte Carlo Tree Search (MCTS)**: MCTS allows AlphaGo to determine the optimal next move by simulating potential game outcomes. The policy network guides the selection priorities, and the value network evaluates the outcome of the simulated moves. Through this search algorithm, AlphaGo could play moves that are both intuitive and calculated, similar to human play.


<br/>
# Results  





이 논문에서는 AlphaGo의 성능을 다양한 바둑 프로그램과 비교하고, Fan Hui와의 대결에서 사용된 메트릭을 통해 평가했습니다. 주요 비교 결과는 다음과 같습니다:

1. **Elo 레이팅**: AlphaGo는 다양한 바둑 프로그램과의 대결에서 압도적인 승률을 보여, 기존 프로그램에 비해 높은 Elo 레이팅을 기록했습니다. 단일 기계에서의 AlphaGo는 494판 중 493승을 기록하며 기존 바둑 프로그램 대비 높은 수준을 입증했고, 분산 버전에서는 더욱 높은 성능을 발휘했습니다.

2. **승률**: AlphaGo는 Zen, Crazy Stone, Pachi와 같은 프로그램을 상대로 최대 4개의 핸디캡을 주고도 77%, 86%, 99%의 승률을 기록했습니다. 또한, Fan Hui와의 대결에서는 5판 전부를 승리하여 인간 전문가를 상대로 컴퓨터 프로그램이 승리한 최초의 사례로 남았습니다.

3. **모델 변형 간 성능**: AlphaGo의 정책 네트워크, 가치 네트워크, 롤아웃을 결합한 모델이 다른 구성의 AlphaGo 모델 변형보다 안정적으로 높은 성능을 나타냈습니다. 특히 가치 네트워크와 롤아웃을 동시에 사용했을 때 성능이 가장 높았으며, 이는 두 평가 방식이 상호 보완적임을 시사합니다.

4. **성능 확장성**: AlphaGo는 비동기 및 분산 MCTS 탐색을 통해 다중 CPU와 GPU를 활용하는 높은 확장성을 보여줬습니다. CPU 1202개와 GPU 176개를 활용한 분산 AlphaGo는 Fan Hui를 상대로 최고의 성능을 발휘했습니다.   

---



This paper evaluates AlphaGo's performance through comparisons with various Go programs and metrics used in its match against Fan Hui. The main findings are as follows:

1. **Elo Rating**: AlphaGo demonstrated a dominant win rate across games with other Go programs, earning a significantly higher Elo rating compared to existing programs. On a single machine, AlphaGo won 494 out of 495 games, showcasing its superiority over previous Go programs, and its distributed version performed even better.

2. **Win Rates**: AlphaGo achieved win rates of 77%, 86%, and 99% when playing against programs like Zen, Crazy Stone, and Pachi with up to four handicap stones. Additionally, AlphaGo won all five formal games against Fan Hui, marking the first time a computer program defeated a human professional player in Go.

3. **Performance Among Model Variants**: The version of AlphaGo combining the policy network, value network, and rollouts outperformed other configurations of AlphaGo models. Specifically, using both value networks and rollouts yielded the best performance, suggesting a complementary relationship between the two evaluation mechanisms.

4. **Scalability**: AlphaGo demonstrated high scalability by utilizing asynchronous and distributed MCTS, effectively leveraging multiple CPUs and GPUs. The distributed version with 1,202 CPUs and 176 GPUs achieved optimal performance in its match against Fan Hui.   



<br/>
# 예제  


### 한글 설명 (Example)

AlphaGo의 성능을 보여주는 예로 Fan Hui와의 경기 결과가 있습니다. AlphaGo는 프랑스의 바둑 챔피언인 Fan Hui와 5번의 대국을 벌여 모두 승리했습니다. AlphaGo는 이러한 경기에서 가치 네트워크와 정책 네트워크를 효과적으로 활용하여 수를 예측하고, 몬테카를로 트리 탐색을 통해 최적의 수를 결정했습니다. Fan Hui는 AlphaGo의 움직임을 인간 전문가와 유사하다고 평하였으며, 이 대결을 통해 AlphaGo가 프로 수준의 바둑 실력을 갖췄음을 입증했습니다.

이 예제는 AlphaGo가 인간 프로 선수와 대등한 수준에서 경기할 수 있음을 보여주며, 기존의 바둑 프로그램과의 차별성을 나타냅니다. AlphaGo는 Deep Blue가 체스에서 Kasparov를 상대로 수천 개의 위치를 평가한 것과 달리, 훨씬 적은 수의 위치를 평가하면서도 정교한 정책 및 가치 네트워크를 사용하여 높은 성과를 거두었습니다.

---

### English Explanation (Example)

An example demonstrating AlphaGo's performance is its match against Fan Hui. AlphaGo played five games against Fan Hui, a Go champion from France, and won all five. In these games, AlphaGo effectively utilized its value and policy networks to predict moves and determined optimal moves through Monte Carlo Tree Search. Fan Hui noted that AlphaGo’s moves resembled those of a human expert, and this match validated AlphaGo’s proficiency at a professional Go level.

This example illustrates AlphaGo’s ability to compete at par with human professional players, showcasing its distinction from prior Go programs. Unlike Deep Blue, which evaluated thousands of positions in its chess match against Kasparov, AlphaGo evaluated far fewer positions, achieving high performance by relying on precise policy and value networks.



<br/>  
# 요약   





AlphaGo는 정책 네트워크와 가치 네트워크를 사용하여 바둑판의 상태를 평가하고 최적의 수를 예측합니다. 이 두 네트워크는 지도 학습과 강화 학습을 통해 훈련되었으며, 몬테카를로 트리 탐색(MCTS)과 결합해 높은 성능을 보여줍니다. AlphaGo는 다양한 바둑 프로그램을 상대로 압도적인 승률을 기록했고, 인간 프로 바둑 챔피언인 Fan Hui를 5대 0으로 승리하며 인간과 대등한 수준의 성과를 입증했습니다. 이 성과는 인공지능이 복잡한 결정을 요구하는 게임에서도 인간 수준의 실력을 갖출 수 있음을 보여줍니다.

---



AlphaGo evaluates board positions and predicts optimal moves using its policy and value networks. These networks are trained through supervised and reinforcement learning and are combined with Monte Carlo Tree Search (MCTS) to achieve high performance. AlphaGo demonstrated a dominant win rate against various Go programs and defeated the human professional Go champion Fan Hui 5-0, proving its capability at a human-competitive level. This achievement illustrates that artificial intelligence can attain human-level expertise in complex decision-making games.


<br/>  
# 기타 



이 이미지는 AlphaGo에서 정책 네트워크와 가치 네트워크를 학습하고 활용하는 과정을 설명합니다. 각 네트워크는 바둑의 상태를 평가하고, 다음 수를 예측하는 데 중요한 역할을 합니다.

1. **롤아웃 정책 (Rollout Policy)**: 초기 정책 네트워크는 인간 전문가의 기보를 학습하여 수를 예측하는 분류 작업을 수행합니다. 여기서 학습된 모델을 `pπ`라고 부릅니다.

2. **지도 학습 (SL) 정책 네트워크**: 정책 네트워크(`pσ`)는 인간 전문가의 데이터를 바탕으로 학습되며, 분류 과정을 통해 수를 예측합니다. 이는 AlphaGo가 다음 수를 정밀하게 선택할 수 있도록 돕습니다.

3. **강화 학습 (RL) 정책 네트워크**: 지도 학습에서 학습된 정책 네트워크는 자가 대국을 통해 추가로 학습됩니다. 정책 그래디언트를 통해 강화 학습이 이루어지며, 이 네트워크는 `pρ`로 불립니다.

4. **가치 네트워크 (Value Network)**: 가치 네트워크(`vθ`)는 보드의 상태를 평가하고, 주어진 위치에서 승리할 확률을 예측하는 회귀 과정을 수행합니다. 이 네트워크는 자가 대국 데이터를 바탕으로 학습되며, AlphaGo가 상태에 따라 최적의 결정을 내리도록 돕습니다.

이렇게 훈련된 정책 네트워크와 가치 네트워크는 AlphaGo가 바둑판에서 다음 최선의 수를 예측하고 평가하는 데 중요한 역할을 합니다.

---



This image explains the training and utilization process of the policy and value networks in AlphaGo. Each network plays a critical role in evaluating the game state and predicting the next move.

1. **Rollout Policy**: The initial policy network, trained on human expert games, performs a classification task to predict moves. This trained model is referred to as `pπ`.

2. **Supervised Learning (SL) Policy Network**: The policy network (`pσ`) is trained on human expert data and predicts moves through classification. This enables AlphaGo to accurately choose the next move.

3. **Reinforcement Learning (RL) Policy Network**: The policy network trained with supervised learning is further refined through self-play reinforcement learning, using policy gradients. This refined network is known as `pρ`.

4. **Value Network**: The value network (`vθ`) evaluates the board state and predicts the probability of winning from a given position through a regression task. This network is trained on self-play data and aids AlphaGo in making optimal decisions based on the game state.

These trained policy and value networks are essential for AlphaGo to accurately predict and evaluate the best moves on the Go board. 







이 이미지는 AlphaGo가 바둑에서 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS)을 사용하는 과정의 각 단계를 설명하고 있습니다. 각 단계는 선택(Selection), 확장(Expansion), 평가(Evaluation), 백업(Backup)으로 나뉘어 있으며, 이를 통해 AlphaGo는 최적의 수를 찾아갑니다.

1. **선택 (Selection)**: 트리의 루트에서 시작하여 선택 규칙(Q + u(P))을 사용하여 최적의 경로를 따라 이동합니다. 이때 가장 높은 가치를 가진 수를 선택합니다.
2. **확장 (Expansion)**: 현재 트리에 없는 새로운 노드가 도달하면 트리가 확장되며, 가능한 다음 수들이 추가됩니다. 여기서 확률(P)을 기반으로 확장 노드들이 선택됩니다.
3. **평가 (Evaluation)**: 확장된 노드는 가치 네트워크(vθ)에 의해 평가되며, 보드 상태에 대한 예측이 이루어집니다. 여기서 승리 확률을 계산하여 그 값을 평가합니다.
4. **백업 (Backup)**: 평가된 결과(vθ와 r)가 트리를 따라 역방향으로 전파되어 상위 노드에 반영됩니다. 이를 통해 선택된 경로의 가치를 업데이트하며, 최적의 수를 찾는 과정이 반복됩니다.

이 과정을 통해 AlphaGo는 현재 보드 상태에서 다음 최선의 수를 효율적으로 탐색하여 인간 수준의 직관적인 수를 둘 수 있게 됩니다.

---



The image illustrates the stages of Monte Carlo Tree Search (MCTS) used by AlphaGo in the game of Go. The process is divided into four stages: Selection, Expansion, Evaluation, and Backup, which collectively help AlphaGo find the optimal move.

1. **Selection**: Starting from the root, AlphaGo follows a selection rule (Q + u(P)) to traverse the tree along the path with the highest value, selecting moves that maximize the combined evaluation score.
2. **Expansion**: Upon reaching a node that isn’t already in the tree, the tree expands by adding new possible moves. These new nodes are chosen based on probabilities (P).
3. **Evaluation**: The expanded node is evaluated by the value network (vθ), which predicts the win probability from that board state. This step assesses the potential outcome of the current board position.
4. **Backup**: The evaluated results (vθ and r) are propagated back up the tree, updating the values of the traversed nodes. This updates the values along the selected path, refining the search for the optimal move.

Through this iterative process, AlphaGo efficiently explores potential moves, enabling it to make intuitive yet calculated moves at a human-competitive level.


<br/>
# refer format:     



@article{silver2016alphago,
    title={Mastering the game of Go with deep neural networks and tree search},
    author={Silver, David and Huang, Aja and Maddison, Chris J. and Guez, Arthur and Sifre, Laurent and van den Driessche, George and Schrittwieser, Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot, Marc and Dieleman, Sander and Grewe, Dominik and Nham, John and Kalchbrenner, Nal and Sutskever, Ilya and Lillicrap, Timothy and Leach, Madeleine and Kavukcuoglu, Koray and Graepel, Thore and Hassabis, Demis},
    journal={Nature},
    volume={529},
    number={7587},
    pages={484--489},
    year={2016},
    publisher={Macmillan Publishers Limited}
}




Silver, David, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. 2016. "Mastering the Game of Go with Deep Neural Networks and Tree Search." Nature 529, no. 7587: 484–89.  