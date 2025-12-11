---
layout: post
title:  "[2023]Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"
date:   2025-12-11 21:28:49 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

체인 오브 씽킹(Chain-of-Thought) 프롬프트의 더 나은 문제 해결력을 위해 복잡한 문제를 일련의 간단한 하위 문제로 나누고, 이를 순차적으로 해결하는 리스트 투 모스트 프롬프트 제안  


짧은 요약(Abstract) :


이 논문에서는 체인 오브 씽킹(Chain-of-Thought) 프롬프트가 다양한 자연어 추론 작업에서 뛰어난 성능을 보였지만, 프롬프트에 제시된 예시보다 더 어려운 문제를 해결하는 데는 한계가 있음을 지적합니다. 이를 극복하기 위해 저자들은 '리스트-투-모스트(Least-to-Most) 프롬프트'라는 새로운 프롬프트 전략을 제안합니다. 이 전략의 핵심 아이디어는 복잡한 문제를 일련의 간단한 하위 문제로 나누고, 이를 순차적으로 해결하는 것입니다. 각 하위 문제의 해결은 이전에 해결된 하위 문제의 답변에 의해 촉진됩니다. 실험 결과, 리스트-투-모스트 프롬프트는 기호 조작, 조합 일반화 및 수학적 추론과 관련된 작업에서 더 어려운 문제로 일반화할 수 있는 능력을 보여주었습니다. 특히, GPT-3 모델을 사용한 실험에서 리스트-투-모스트 프롬프트는 체인 오브 씽킹 프롬프트에 비해 훨씬 높은 정확도를 기록했습니다.



This paper points out that while Chain-of-Thought prompting has demonstrated remarkable performance on various natural language reasoning tasks, it has limitations in solving problems that are harder than the exemplars shown in the prompts. To overcome this challenge, the authors propose a novel prompting strategy called "Least-to-Most prompting." The key idea of this strategy is to break down a complex problem into a series of simpler subproblems and then solve them in sequence. The solution to each subproblem is facilitated by the answers to previously solved subproblems. Experimental results reveal that Least-to-Most prompting is capable of generalizing to more difficult problems related to symbolic manipulation, compositional generalization, and math reasoning. Notably, when using the GPT-3 model, Least-to-Most prompting achieved significantly higher accuracy compared to Chain-of-Thought prompting.


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



이 논문에서 제안하는 "Least-to-Most Prompting" 방법은 복잡한 문제를 해결하기 위해 문제를 더 간단한 하위 문제로 분해하고, 이를 순차적으로 해결하는 방식입니다. 이 방법은 두 가지 주요 단계로 구성됩니다.

1. **문제 분해 (Decomposition)**: 이 단계에서는 주어진 문제를 여러 개의 하위 문제로 나누는 과정을 포함합니다. 이 과정에서 모델은 이전에 해결된 하위 문제의 답변을 활용하여 문제를 분해합니다. 예를 들어, "엘사가 5개의 사과를 가지고 있고, 안나가 엘사보다 2개 더 많은 사과를 가지고 있다면, 두 사람이 함께 몇 개의 사과를 가지고 있는가?"라는 문제를 "안나가 몇 개의 사과를 가지고 있는가?"와 "엘사와 안나가 함께 몇 개의 사과를 가지고 있는가?"라는 두 개의 하위 문제로 나눌 수 있습니다.

2. **하위 문제 해결 (Subproblem Solving)**: 이 단계에서는 분해된 하위 문제를 순차적으로 해결합니다. 각 하위 문제의 답변은 다음 하위 문제를 해결하는 데 사용됩니다. 예를 들어, 첫 번째 하위 문제의 답변을 바탕으로 두 번째 하위 문제를 해결하는 방식입니다.

이 방법은 기존의 "Chain-of-Thought Prompting" 방법과 비교하여 더 어려운 문제를 해결하는 데 효과적입니다. 특히, "Least-to-Most Prompting"은 모델이 훈련 데이터에서 본 적이 없는 더 복잡한 문제를 해결할 수 있도록 도와줍니다. 실험 결과에 따르면, 이 방법은 수학적 추론, 기호 조작, 조합 일반화와 같은 다양한 자연어 처리 작업에서 우수한 성능을 보였습니다.

이 방법은 특별한 아키텍처나 훈련 데이터 없이도 사용할 수 있으며, 기존의 모델에 간단히 적용할 수 있는 장점이 있습니다. 또한, 이 방법은 교육 심리학에서 유래된 개념으로, 학생이 새로운 기술을 배우는 데 도움을 주기 위해 점진적인 프롬프트를 사용하는 방식에서 영감을 받았습니다.




The "Least-to-Most Prompting" method proposed in this paper is designed to enable complex problem-solving by breaking down a complex problem into simpler subproblems and solving them sequentially. This method consists of two main stages:

1. **Decomposition**: In this stage, the process of breaking down the given problem into several subproblems is included. During this process, the model utilizes the answers to previously solved subproblems to aid in the decomposition. For example, a problem like "Elsa has 5 apples, and Anna has 2 more apples than Elsa. How many apples do they have together?" can be decomposed into two subproblems: "How many apples does Anna have?" and "How many apples do Elsa and Anna have together?"

2. **Subproblem Solving**: In this stage, the decomposed subproblems are solved sequentially. The answer to each subproblem is used to solve the next subproblem. For instance, the answer to the first subproblem is used to address the second subproblem.

This method is particularly effective in solving more difficult problems compared to the existing "Chain-of-Thought Prompting" method. Notably, "Least-to-Most Prompting" helps the model tackle more complex problems that it has not encountered in the training data. Experimental results indicate that this method demonstrates superior performance across various natural language processing tasks, including mathematical reasoning, symbolic manipulation, and compositional generalization.

The method can be applied without requiring special architectures or training data, making it advantageous for straightforward implementation on existing models. Additionally, this approach is inspired by concepts from educational psychology, where a progressive sequence of prompts is used to assist students in learning new skills.


<br/>
# Results



이 논문에서는 "Least-to-Most Prompting"이라는 새로운 프롬프트 기법을 제안하고, 이를 기존의 "Chain-of-Thought Prompting"과 비교하여 성능을 평가합니다. 실험은 여러 자연어 추론 작업에서 수행되었으며, 특히 수학 문제 해결, 기호 조작, 조합 일반화와 같은 복잡한 문제를 다루었습니다.

#### 결과 요약

1. **경쟁 모델**: 실험에서는 GPT-3의 여러 버전(예: code-davinci-002, text-davinci-002 등)을 사용하여 성능을 비교했습니다. 특히, code-davinci-002 모델이 가장 높은 성능을 보였습니다.

2. **테스트 데이터**: 다양한 테스트 데이터셋이 사용되었습니다. 예를 들어, SCAN 벤치마크와 GSM8K 데이터셋이 포함되어 있으며, 이들은 복잡한 문제 해결을 요구하는 데이터셋입니다.

3. **메트릭**: 성능 평가는 정확도(accuracy)를 기준으로 하였습니다. 각 프롬프트 기법의 정확도를 비교하여, Least-to-Most Prompting이 Chain-of-Thought Prompting보다 더 높은 정확도를 기록했습니다.

4. **비교**: 
   - **Chain-of-Thought Prompting**: 이 방법은 문제를 단계별로 해결하는 접근 방식을 사용하지만, 복잡한 문제에 대한 일반화에서 한계를 보였습니다. 예를 들어, Chain-of-Thought Prompting은 길이가 긴 리스트에 대한 문제에서 성능이 급격히 떨어졌습니다.
   - **Least-to-Most Prompting**: 이 방법은 복잡한 문제를 더 간단한 하위 문제로 분해하여 순차적으로 해결하는 방식으로, 더 어려운 문제에 대해서도 높은 정확도를 보였습니다. 실험 결과, Least-to-Most Prompting은 SCAN 벤치마크에서 99.7%의 정확도를 기록하며, Chain-of-Thought Prompting의 16%에 비해 월등한 성과를 보였습니다.

5. **결과 분석**: Least-to-Most Prompting은 특히 문제의 복잡성이 증가할수록 더 큰 이점을 보였습니다. 예를 들어, 5단계 이상의 문제에서 Chain-of-Thought Prompting의 정확도가 39.07%인 반면, Least-to-Most Prompting은 45.23%로 더 높은 성과를 기록했습니다.

이러한 결과는 Least-to-Most Prompting이 복잡한 문제 해결에 있어 더 효과적인 접근 방식임을 시사합니다. 이 방법은 문제를 단계적으로 해결할 수 있도록 도와주며, 기존의 방법보다 더 나은 일반화 능력을 보여줍니다.

---




This paper introduces a novel prompting technique called "Least-to-Most Prompting" and evaluates its performance against the existing "Chain-of-Thought Prompting." The experiments were conducted on various natural language reasoning tasks, particularly focusing on complex problems such as mathematical reasoning, symbolic manipulation, and compositional generalization.

#### Summary of Results

1. **Competing Models**: The experiments utilized various versions of GPT-3 (e.g., code-davinci-002, text-davinci-002) to compare performance. Notably, the code-davinci-002 model exhibited the highest performance.

2. **Test Data**: A variety of test datasets were employed, including the SCAN benchmark and the GSM8K dataset, which require complex problem-solving capabilities.

3. **Metrics**: Performance evaluation was based on accuracy. The accuracy of each prompting technique was compared, showing that Least-to-Most Prompting achieved higher accuracy than Chain-of-Thought Prompting.

4. **Comparison**:
   - **Chain-of-Thought Prompting**: This method uses a step-by-step approach to solve problems but shows limitations in generalization for complex problems. For instance, Chain-of-Thought Prompting significantly drops in performance on problems with longer lists.
   - **Least-to-Most Prompting**: This method breaks down complex problems into simpler subproblems and solves them sequentially, demonstrating high accuracy even on more difficult problems. Experimental results showed that Least-to-Most Prompting achieved 99.7% accuracy on the SCAN benchmark, compared to only 16% for Chain-of-Thought Prompting.

5. **Results Analysis**: Least-to-Most Prompting exhibited greater advantages as the complexity of the problems increased. For example, in problems requiring five or more steps, Chain-of-Thought Prompting achieved an accuracy of 39.07%, while Least-to-Most Prompting recorded 45.23%, indicating superior performance.

These results suggest that Least-to-Most Prompting is a more effective approach for solving complex problems. This method aids in solving problems step-by-step and demonstrates better generalization capabilities compared to existing methods.


<br/>
# 예제



이 논문에서는 "Least-to-Most Prompting"이라는 새로운 프롬프트 기법을 제안하고, 이를 통해 복잡한 문제를 해결하는 방법을 설명합니다. 이 기법은 문제를 더 간단한 하위 문제로 분해한 후, 이 하위 문제들을 순차적으로 해결하는 방식입니다. 

#### 예시 1: 마지막 글자 연결 작업
- **문제**: "think, machine, learning"이라는 단어 목록이 주어졌을 때, 각 단어의 마지막 글자를 연결하여 결과를 도출해야 합니다.
- **입력**: "think, machine, learning"
- **출력**: "keg" (think의 마지막 글자 'k', machine의 마지막 글자 'e', learning의 마지막 글자 'g'를 연결한 결과)

**Least-to-Most Prompting**:
1. **하위 문제 분해**: "think, machine"을 먼저 처리하여 "ke"를 얻습니다.
2. **하위 문제 해결**: "think, machine"의 결과를 사용하여 "learning"의 마지막 글자 'g'를 추가하여 최종 결과 "keg"를 도출합니다.

#### 예시 2: SCAN 벤치마크
- **문제**: 자연어 명령을 행동 시퀀스로 변환하는 작업입니다.
- **입력**: "look opposite right thrice after walk"
- **출력**: "TURN LEFT TURN LEFT TURN RIGHT TURN RIGHT WALK" (명령어를 행동으로 변환한 결과)

**Least-to-Most Prompting**:
1. **하위 문제 분해**: "look opposite right thrice"와 "walk"로 나누어 각각의 행동을 도출합니다.
2. **하위 문제 해결**: 각 하위 문제의 결과를 결합하여 최종 행동 시퀀스를 생성합니다.

이러한 방식으로, Least-to-Most Prompting은 복잡한 문제를 해결하는 데 있어 더 높은 정확도를 보여주며, 특히 훈련 데이터보다 더 어려운 문제를 해결하는 데 효과적입니다.

---




This paper introduces a new prompting technique called "Least-to-Most Prompting," which explains how to solve complex problems. This technique involves breaking down a complex problem into simpler subproblems and then solving these subproblems sequentially.

#### Example 1: Last Letter Concatenation Task
- **Problem**: Given a list of words "think, machine, learning," the task is to concatenate the last letters of each word to produce a result.
- **Input**: "think, machine, learning"
- **Output**: "keg" (the last letter of think is 'k', the last letter of machine is 'e', and the last letter of learning is 'g', concatenated together)

**Least-to-Most Prompting**:
1. **Decomposition**: First, handle "think, machine" to get "ke."
2. **Subproblem Solving**: Use the result from "think, machine" to add the last letter 'g' from "learning" to produce the final result "keg."

#### Example 2: SCAN Benchmark
- **Problem**: The task is to convert natural language commands into action sequences.
- **Input**: "look opposite right thrice after walk"
- **Output**: "TURN LEFT TURN LEFT TURN RIGHT TURN RIGHT WALK" (the result of converting the command into actions)

**Least-to-Most Prompting**:
1. **Decomposition**: Break down into "look opposite right thrice" and "walk" to derive each action.
2. **Subproblem Solving**: Combine the results of each subproblem to generate the final action sequence.

In this way, Least-to-Most Prompting demonstrates higher accuracy in solving complex problems, especially in addressing tasks that are more difficult than those seen in the training data.

<br/>
# 요약


이 논문에서는 복잡한 문제를 간단한 하위 문제로 나누어 해결하는 'Least-to-Most Prompting' 방법을 제안하고, 이를 통해 기존의 'Chain-of-Thought Prompting'보다 더 어려운 문제를 해결할 수 있음을 보여준다. 실험 결과, Least-to-Most Prompting은 SCAN 및 수학 문제 해결에서 높은 정확도를 기록하며, 특히 하위 문제 해결 과정에서 이전 문제의 답변을 활용하는 방식이 효과적임을 입증하였다. 예를 들어, GPT-3 모델을 사용하여 SCAN 벤치마크에서 99.7%의 정확도를 달성하였다.



This paper proposes a method called 'Least-to-Most Prompting,' which breaks down complex problems into simpler subproblems, demonstrating its ability to solve more difficult problems than the existing 'Chain-of-Thought Prompting.' Experimental results show that Least-to-Most Prompting achieves high accuracy in tasks like SCAN and math problem-solving, particularly by effectively utilizing answers from previously solved subproblems. For instance, it achieved an accuracy of 99.7% on the SCAN benchmark using the GPT-3 model.

<br/>
# 기타



이 논문에서는 "Least-to-Most Prompting"이라는 새로운 프롬프트 기법을 제안하고, 이를 통해 복잡한 문제를 더 쉽게 해결할 수 있는 방법을 제시합니다. 이 기법은 문제를 더 간단한 하위 문제로 분해하고, 이를 순차적으로 해결하는 방식으로 작동합니다. 다음은 논문에서 제시된 주요 결과와 인사이트입니다.

1. **다이어그램 및 피규어**: 
   - Figure 1에서는 Least-to-Most Prompting의 두 단계(문제 분해 및 하위 문제 해결)를 시각적으로 설명합니다. 이 다이어그램은 모델이 문제를 어떻게 분해하고, 각 하위 문제를 해결하는지를 보여줍니다.

2. **테이블**:
   - **테이블 4**: 다양한 프롬프트 방법의 정확도를 비교합니다. Least-to-Most Prompting은 Chain-of-Thought Prompting보다 모든 리스트 길이에서 더 높은 정확도를 보였습니다. 특히 리스트 길이가 길어질수록 그 차이는 더욱 두드러졌습니다.
   - **테이블 8**: SCAN 벤치마크에서의 성능을 보여줍니다. Least-to-Most Prompting은 99.7%의 정확도로, Chain-of-Thought Prompting보다 월등한 성능을 보였습니다.
   - **테이블 11**: GSM8K 및 DROP 벤치마크에서의 다양한 프롬프트 방법의 정확도를 비교합니다. Least-to-Most Prompting은 Chain-of-Thought Prompting보다 더 높은 정확도를 기록했습니다.

3. **어펜딕스**:
   - 어펜딕스에서는 각 실험의 세부 사항과 추가적인 결과를 제공합니다. 예를 들어, Least-to-Most Prompting의 오류 분석이 포함되어 있으며, 모델이 잘못된 답변을 생성한 이유를 설명합니다. 이 분석은 모델의 한계를 이해하는 데 도움을 줍니다.

이러한 결과들은 Least-to-Most Prompting이 복잡한 문제를 해결하는 데 있어 효과적인 방법임을 보여줍니다. 특히, 이 기법은 기존의 Chain-of-Thought Prompting보다 더 높은 일반화 능력을 가지고 있으며, 더 어려운 문제를 해결할 수 있는 잠재력을 가지고 있습니다.

---




This paper introduces a new prompting technique called "Least-to-Most Prompting," which provides a method for solving complex problems more easily. This technique operates by breaking down a problem into simpler subproblems and solving them sequentially. Here are the key results and insights presented in the paper:

1. **Diagrams and Figures**: 
   - Figure 1 visually explains the two stages of Least-to-Most Prompting (problem decomposition and subproblem solving). This diagram illustrates how the model decomposes the problem and solves each subproblem.

2. **Tables**:
   - **Table 4**: Compares the accuracy of various prompting methods. Least-to-Most Prompting outperformed Chain-of-Thought Prompting across all list lengths, with the difference becoming more pronounced as the list length increased.
   - **Table 8**: Shows performance on the SCAN benchmark. Least-to-Most Prompting achieved an accuracy of 99.7%, significantly outperforming Chain-of-Thought Prompting.
   - **Table 11**: Compares the accuracy of various prompting methods on GSM8K and DROP benchmarks. Least-to-Most Prompting recorded higher accuracy than Chain-of-Thought Prompting.

3. **Appendix**:
   - The appendix provides details of each experiment and additional results. For instance, it includes an error analysis of Least-to-Most Prompting, explaining why the model generated incorrect answers. This analysis helps in understanding the limitations of the model.

These results demonstrate that Least-to-Most Prompting is an effective method for solving complex problems. Notably, this technique exhibits a higher generalization ability than the existing Chain-of-Thought Prompting and has the potential to tackle more difficult problems.

<br/>
# refer format:


### BibTeX   

```bibtex
@inproceedings{zhou2023least,
  title={Least-to-Most Prompting Enables Complex Reasoning in Large Language Models},
  author={Denny Zhou and Nathanael Schärli and Le Hou and Jason Wei and Nathan Scales and Xuezhi Wang and Dale Schuurmans and Claire Cui and Olivier Bousquet and Quoc Le and Ed Chi},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2023},
}
```

### 시카고 스타일

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models." In *Proceedings of the International Conference on Learning Representations (ICLR).
---


