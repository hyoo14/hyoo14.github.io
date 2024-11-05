---
layout: post
title:  "[2024]PDDLEGO Iterative Planning in Textual Environments"  
date:   2024-11-04 21:06:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

음 결국 플래닝을 위한 제안인데.. LLM을 직접 튜닝하는 것이 아니라 포맷을 제공하는 거임.. 포맷을 통해 LLM을 더 잘 활용하자 느낌  
그리고 이 포맷은 약간 랭기지처럼 보다 포멀한, 정형화된 포맷임  
이를 통해 플래닝, 그리고 그 안에 액션, 스테이트, 컨디션 등등을 반영하여 LLM이 더 잘 이해하고 더 잘 싸우게 하겠다...  


짧은 요약(Abstract) :    



이 논문은 **PDDLEGO**라는 시스템을 제안합니다. 이 시스템은 **LLM(대형 언어 모델)**과 **PDDL(계획 영역 정의 언어)**을 결합하여 텍스트 환경에서 목표를 점진적으로 달성하는 방법을 제안합니다. PDDLEGO는 기존 방식과 달리, 모든 정보가 초기부터 주어지지 않는 환경(부분적으로 관찰 가능한 환경)에서도 작동합니다. 초기 계획을 세우는 데 필요한 정보를 얻기 위해 부분 목표를 설정하고 이를 달성함으로써 최종 목표로 나아갑니다. 실험 결과, 이 방식은 목표를 일괄적으로 생성하는 기존 방법보다 **43% 더 효율적**이며, 특히 복잡한 요리 시뮬레이션 환경에서 기존 LLM이 실패한 곳에서 **98%의 성공률**을 보였습니다. 이는 이 방법이 탐색 과정에서 환경을 점진적으로 이해하고 목표 달성을 위한 계획을 업데이트할 수 있음을 시사합니다.


This paper proposes **PDDLEGO**, a system that integrates **Large Language Models (LLMs)** and **PDDL (Planning Domain Definition Language)** to achieve goals iteratively in text-based environments. Unlike previous methods that require full initial information, PDDLEGO operates effectively in partially observed environments. It achieves this by setting and achieving sub-goals, gradually accumulating the information needed to plan for the final goal. Experimental results indicate that this approach is **43% more efficient** than conventional methods that generate end-to-end plans. Notably, it achieves a **98% success rate** in a complex cooking simulation environment where traditional LLMs fail to create coherent plans, highlighting its ability to progressively update plans based on exploration.


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



1. PDDL-gen (PDDL 생성)

PDDL-gen은 탐색 초기 단계에서 문제 파일을 생성하기 위한 메서드입니다. 처음으로 주어진 환경에 대한 설명과 목표 상태를 바탕으로 현재 상태와 목표 상태를 정의하는 문제 파일을 작성합니다. 이 문제 파일은 PDDL 형식으로 표현되며, 탐색 중에 필요한 기본적인 정보가 포함됩니다. 이 메서드는 목표를 달성할 수 있는 충분한 정보가 없는 경우 일단 초기 파일을 만들어 탐색을 시작할 수 있도록 돕습니다.

예를 들어, 사용자가 처음 방에 들어가서 주어진 목표가 "코인 찾기"일 때, 이 메서드는 방의 현재 상태와 목표 상태를 기반으로 문제 파일을 만듭니다.


PDDL-gen is a method used to create an initial problem file at the beginning of the exploration phase. It defines the current state and goal state based on the initial description of the environment and the end-goal. This file, expressed in PDDL format, provides the basic structure needed to start exploring the environment, even if there is not yet enough information to reach the end-goal.

2. PDDL-edit (PDDL 편집)

PDDL-edit는 탐색 도중 새로운 정보를 반영하기 위해 사용됩니다. 새로운 방을 방문하거나, 추가적인 객체와의 상호작용이 발생할 때마다, PDDL-edit는 현재 문제 파일을 업데이트하여 최신 상태를 반영합니다. 이 메서드는 환경 탐색 중 얻어진 관찰 정보만 추가하거나 변경하여 문제 파일을 수정하므로, 전체 파일을 다시 생성할 필요 없이 필요한 부분만 수정하게 됩니다. 이 방법으로 효율적이고 일관성 있게 탐색을 진행할 수 있습니다.

예를 들어, 주방에서 거실로 이동했을 때 PDDL-edit는 방 위치 및 방문 상태 등을 문제 파일에 추가하여 업데이트합니다.


PDDL-edit is used to update the problem file as new information is acquired during exploration. When the agent enters a new room or interacts with additional objects, PDDL-edit modifies the current problem file to reflect the latest state. By adding or changing only the newly observed information, this method efficiently updates the problem file without regenerating it from scratch, allowing for consistent and streamlined exploration.


<br/>
# Results  



이 논문에서는 **PDDLEGO 모델의 성능을 평가**하기 위해 **텍스트 기반 시뮬레이션 환경**에서 모델이 목표를 얼마나 잘 달성하는지를 확인하는 방식으로 평가를 진행했습니다. 경쟁 상대와 비교 기준(메트릭)은 다음과 같습니다.

---

### 1. **평가 방법**: 시뮬레이션 환경에서의 성능

논문에서는 PDDLEGO가 **Coin Collector**와 **Cooking World**라는 두 가지 텍스트 기반 시뮬레이션 환경에서 목표를 얼마나 잘 달성하는지를 평가했습니다.

- **Coin Collector**: 여러 방을 돌아다니며 특정 위치에 숨겨진 코인을 찾는 게임입니다. 코인의 위치는 처음에는 알려지지 않기 때문에, 탐색을 통해 정보를 얻어가며 목표에 도달해야 합니다.
  
- **Cooking World**: 이 환경은 좀 더 복잡하여, 다양한 방을 탐색해 요리에 필요한 재료를 찾고, 해당 재료를 특정 방법으로 요리해 목표 요리를 완성해야 합니다. 단계별로 재료 수집, 조리 과정이 필요해 목표를 점진적으로 달성하는 과정이 요구됩니다.

이 두 환경은 **점진적으로 정보를 얻고 목표를 수정하는** PDDLEGO의 특성을 평가하는 데 적합합니다.

---

### 2. **경쟁 상대**: 기존의 LLM 기반 계획 모델

PDDLEGO의 경쟁 모델은 기존에 사용되던 **LLM 기반 계획 생성 모델**입니다. 일반적인 LLM 모델들은 주어진 목표를 달성하기 위해 **한 번에 완전한 계획을 세우려는 접근 방식**을 취합니다. 이를 `Action-gen`이라고 부르며, LLM이 직접 환경을 이해하고 필요한 행동을 단번에 생성하려고 합니다. 

이와 달리 PDDLEGO는 **PDDL 형식을 기반으로 부분 목표를 설정하고 점진적으로 계획을 수정**하므로, 환경이 점진적으로 관찰될 때 더 높은 성능을 기대할 수 있습니다.

---

### 3. **메트릭 (평가지표)**

PDDLEGO의 성능을 측정하기 위해 사용된 주요 메트릭은 다음과 같습니다:

- **성공률(Success Rate)**: 주어진 시간 내에 목표를 달성하는 비율입니다. 예를 들어, Coin Collector에서 주어진 최대 이동 횟수 내에 코인을 찾는 비율이 포함됩니다.
  
- **효율성(Efficiency)**: 목표를 달성하기 위해 걸린 평균 **행동 수(step count)**입니다. 행동 수가 적을수록 효율적입니다.
  
- **안정성(Stability)**: 여러 시도에서 일관되게 목표를 달성하는지를 측정합니다. 특정 목표에 대해 PDDLEGO가 일관된 계획을 세워 목표에 도달할 수 있는지를 평가합니다.

이 메트릭을 통해 PDDLEGO는 기존 LLM 계획 모델보다 **성공률이 높고, 효율적이며, 일관된 성능**을 보였음을 실험에서 입증했습니다.

---



The paper evaluates the PDDLEGO model by testing it in **text-based simulation environments** to measure how well it achieves goals compared to other models. The primary competitors are **LLM-based planning models** that generate plans in one go, unlike PDDLEGO, which iteratively updates its plans. The main metrics used are:

1. **Success Rate**: The proportion of successful attempts within a given step limit.
2. **Efficiency**: The average number of steps needed to reach the goal.
3. **Stability**: The model’s consistency in achieving the goal across multiple runs.

PDDLEGO demonstrated superior performance in success rate, efficiency, and stability compared to traditional LLM planning models.


이 논문에서는 **GPT-3.5 Turbo**와 **GPT-4 Turbo** 두 가지 LLM을 사용하여 PDDLEGO의 성능을 테스트했습니다. 

### 사용한 LLM 버전

- **GPT-3.5 Turbo**: OpenAI의 3.5세대 모델로, 다양한 언어 이해와 생성 작업에 적합한 모델입니다.
- **GPT-4 Turbo**: GPT-4의 경량화된 버전으로, 일반적으로 더 복잡한 문제나 계획 수립에서 GPT-3.5보다 더 높은 성능을 보입니다.

### 이유

이 두 모델을 사용하여, **다양한 환경에서 계획을 생성하고 수정**하는 작업에서 PDDLEGO가 얼마나 일관되고 효율적으로 동작하는지 확인했습니다. 또한, 각 모델의 성능 차이를 비교하여 더 높은 성능의 모델(GPT-4 Turbo)이 PDDL 형식의 계획 수정에 얼마나 일관적이고 정확한 결과를 내는지 보여주었습니다.

GPT-4 Turbo는 GPT-3.5에 비해 **더 복잡한 계획을 생성**하고 **부분 목표를 설정**하는 작업에서 더 높은 성과를 보였으며, 이는 논문에서도 언급한 주요 결과 중 하나입니다.



<br/>
# 예제  


PDDL(Planning Domain Definition Language)은 AI와 자동화 계획에서 사용하는 **계획 언어**로, 환경의 상태와 목표를 형식적으로 표현하여 문제 해결을 자동화하는 데 사용됩니다. PDDL은 환경에서의 **행위(action)**와 **상태(state)**를 정의하고, 이들 간의 관계를 명시함으로써 **AI 에이전트가 목표를 달성하기 위해 수행해야 할 일련의 행동을 계획**할 수 있도록 돕습니다.

### PDDL의 주요 요소

PDDL은 일반적으로 두 가지 파일로 구성됩니다: **도메인 파일**과 **문제 파일**입니다.

1. **도메인 파일**  
   도메인 파일은 특정 환경에서 가능한 **행동(action)**과 **전제 조건(precondition)**, 그리고 **효과(effect)**를 정의합니다. 예를 들어, "문 열기"라는 행동이 가능하다고 하면, 이 행동을 수행하기 위한 전제 조건으로 "문이 닫혀 있어야 함"을, 효과로는 "문이 열림"을 명시할 수 있습니다.

2. **문제 파일**  
   문제 파일은 **현재 상태(initial state)**와 **목표 상태(goal state)**를 정의합니다. 예를 들어, 문제 파일에서는 "에이전트가 A방에 있음"을 현재 상태로, "에이전트가 코인을 얻음"을 목표 상태로 설정할 수 있습니다. 이 문제 파일에 따라 계획 알고리즘은 목표를 달성하기 위해 어떤 행동을 취해야 하는지 계산합니다.

### PDDL의 역할

PDDL은 단순한 논리 이상의 역할을 합니다. **형식 언어**로서, AI 시스템이 환경의 상태를 논리적으로 이해하고, 상태 간의 변화를 공식화하며, 최종 목표를 달성할 수 있는 구체적인 **행동 계획**을 생성하는 데 필요한 정보를 제공합니다. PDDLEGO 시스템에서는 **LLM(대형 언어 모델)**이 PDDL 형식으로 상태와 행동을 업데이트하여, 에이전트가 목표를 향해 점진적으로 나아갈 수 있도록 합니다.

---

### 예시

예를 들어, "코인 수집" 게임에서 PDDL을 사용해 다음과 같은 문제 파일을 작성할 수 있습니다:

- **도메인 파일**:
   - **행동**: 이동(move), 문 열기(open door)
   - **전제 조건**: 이동하려면 연결된 방이 있어야 하고, 문을 열려면 문이 닫혀 있어야 함
   - **효과**: 이동하면 현재 위치가 변경되고, 문을 열면 닫힌 상태가 열린 상태로 변경됨

- **문제 파일**:
   - **현재 상태**: 에이전트는 주방에 있음
   - **목표 상태**: 에이전트가 코인을 획득함

이러한 설정을 바탕으로, PDDLEGO는 현재 상태를 업데이트하며 목표를 향해 진행할 계획을 세웁니다.



PDDL (Planning Domain Definition Language) is a **planning language** used in AI and automated planning, which formally represents the state of an environment and goals to enable problem-solving. PDDL defines **actions, states, and their relationships** in a way that allows an AI agent to plan a series of actions to achieve a goal.

In PDDLEGO, **LLMs** iteratively update this PDDL structure to adjust plans based on new observations in the environment.


<br/>  
# 요약   

일종의 템플릿을 사용해서 LLM이 컨디션, 스테이트, 액션등을 포함하는 플래닝을 잘 하도록 함  
This research uses plannning language to make the LLM perform well in planning including condition, state, action.  

<br/>  
# 기타  



PDDL은 일종의 **틀(템플릿)** 같은 형식입니다. AI가 환경을 이해하고, 계획을 세우는 데 필요한 **항목(예: 행동, 상태)**과 그에 **맞는 값들(예: 현재 위치, 문이 열린 상태)**로 이루어진 구조라고 할 수 있습니다. 

### PDDL 형식을 LLM에 어떻게 사용하는지

1. **PDDL 템플릿 생성**:  
   처음에는 기본적인 PDDL 템플릿을 만듭니다. 이 템플릿은 특정 환경에서 사용할 수 있는 행동, 상태 전이, 전제 조건 등을 정의합니다.

2. **문제 파일 업데이트**:  
   환경을 탐색하면서 새로운 정보를 얻게 되면, 현재 상태와 목표 상태를 PDDL 템플릿에 맞춰 업데이트합니다. 예를 들어, 새로운 방을 방문하면 "현재 위치" 항목의 값을 그 방의 이름으로 수정하는 식입니다.

3. **LLM 입력으로 제공**:  
   최신 상태로 업데이트된 PDDL 문제 파일을 LLM에 입력으로 제공합니다. LLM은 이 문제 파일을 바탕으로 새로운 계획을 세우거나 필요한 행동을 제안할 수 있습니다.

### 예시로 보면

- PDDL 템플릿을 다음과 같이 정의할 수 있습니다:
   - **행동**: `move`, `open_door`
   - **상태**: `at(location)`, `closed_door(location1, location2)`

- 그 후 텍스트로 입력에 넣는 문제 파일 예시는 다음과 같습니다:
   ```
   (define (problem find-coin)
       (:init 
           (at kitchen)
           (closed_door kitchen living_room)
       )
       (:goal 
           (have coin)
       )
   )
   ```
   여기서 `at`과 `closed_door`는 항목이고, `kitchen`과 `living_room`은 그에 맞는 값입니다. 이렇게 텍스트 형식으로 채운 PDDL을 LLM에 주면, LLM은 이 파일을 참고해 다음 행동을 결정합니다.

이런 방식으로, **PDDL 형식을 통해 LLM이 환경을 이해하고, 행동을 계획하는 데 필요한 정보를 체계적으로 제공**하게 됩니다.  




이 논문에서는 **PDDLEGO 모델의 성능을 평가**하기 위해 **텍스트 기반 시뮬레이션 환경**에서 모델이 목표를 얼마나 잘 달성하는지를 확인하는 방식으로 평가를 진행했습니다. 경쟁 상대와 비교 기준(메트릭)은 다음과 같습니다.

---

### 1. **평가 방법**: 시뮬레이션 환경에서의 성능

논문에서는 PDDLEGO가 **Coin Collector**와 **Cooking World**라는 두 가지 텍스트 기반 시뮬레이션 환경에서 목표를 얼마나 잘 달성하는지를 평가했습니다.

- **Coin Collector**: 여러 방을 돌아다니며 특정 위치에 숨겨진 코인을 찾는 게임입니다. 코인의 위치는 처음에는 알려지지 않기 때문에, 탐색을 통해 정보를 얻어가며 목표에 도달해야 합니다.
  
- **Cooking World**: 이 환경은 좀 더 복잡하여, 다양한 방을 탐색해 요리에 필요한 재료를 찾고, 해당 재료를 특정 방법으로 요리해 목표 요리를 완성해야 합니다. 단계별로 재료 수집, 조리 과정이 필요해 목표를 점진적으로 달성하는 과정이 요구됩니다.

이 두 환경은 **점진적으로 정보를 얻고 목표를 수정하는** PDDLEGO의 특성을 평가하는 데 적합합니다.

---

### 2. **경쟁 상대**: 기존의 LLM 기반 계획 모델

PDDLEGO의 경쟁 모델은 기존에 사용되던 **LLM 기반 계획 생성 모델**입니다. 일반적인 LLM 모델들은 주어진 목표를 달성하기 위해 **한 번에 완전한 계획을 세우려는 접근 방식**을 취합니다. 이를 `Action-gen`이라고 부르며, LLM이 직접 환경을 이해하고 필요한 행동을 단번에 생성하려고 합니다. 

이와 달리 PDDLEGO는 **PDDL 형식을 기반으로 부분 목표를 설정하고 점진적으로 계획을 수정**하므로, 환경이 점진적으로 관찰될 때 더 높은 성능을 기대할 수 있습니다.

---

### 3. **메트릭 (평가지표)**

PDDLEGO의 성능을 측정하기 위해 사용된 주요 메트릭은 다음과 같습니다:

- **성공률(Success Rate)**: 주어진 시간 내에 목표를 달성하는 비율입니다. 예를 들어, Coin Collector에서 주어진 최대 이동 횟수 내에 코인을 찾는 비율이 포함됩니다.
  
- **효율성(Efficiency)**: 목표를 달성하기 위해 걸린 평균 **행동 수(step count)**입니다. 행동 수가 적을수록 효율적입니다.
  
- **안정성(Stability)**: 여러 시도에서 일관되게 목표를 달성하는지를 측정합니다. 특정 목표에 대해 PDDLEGO가 일관된 계획을 세워 목표에 도달할 수 있는지를 평가합니다.

이 메트릭을 통해 PDDLEGO는 기존 LLM 계획 모델보다 **성공률이 높고, 효율적이며, 일관된 성능**을 보였음을 실험에서 입증했습니다.

---



The paper evaluates the PDDLEGO model by testing it in **text-based simulation environments** to measure how well it achieves goals compared to other models. The primary competitors are **LLM-based planning models** that generate plans in one go, unlike PDDLEGO, which iteratively updates its plans. The main metrics used are:

1. **Success Rate**: The proportion of successful attempts within a given step limit.
2. **Efficiency**: The average number of steps needed to reach the goal.
3. **Stability**: The model’s consistency in achieving the goal across multiple runs.

PDDLEGO demonstrated superior performance in success rate, efficiency, and stability compared to traditional LLM planning models.
<br/>
# refer format:     



@inproceedings{zhang2024pddlego,
  title = {PDDLEGO: Iterative Planning in Textual Environments},
  author = {Li Zhang and Peter Jansen and Peter Clark and Chris Callison-Burch and Niket Tandon},
  booktitle = {Proceedings of the 2024 *SEM Conference},
  year = {2024}
}



Zhang, Li, Peter Jansen, Peter Clark, Chris Callison-Burch, and Niket Tandon. 2024. "PDDLEGO: Iterative Planning in Textual Environments." In *Proceedings of the 2024 SEM Conference.  