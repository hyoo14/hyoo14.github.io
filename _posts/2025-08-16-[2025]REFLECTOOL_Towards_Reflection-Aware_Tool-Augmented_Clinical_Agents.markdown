---
layout: post
title:  "[2025]REFLECTOOL: Towards Reflection-Aware Tool-Augmented Clinical Agents"
date:   2025-08-16 16:11:47 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

이 논문에서는 REFLECTOOL이라는 새로운 프레임워크를 제안하여 임상 에이전트가 도구를 효과적으로 활용할 수 있도록 지원  

단순 텍스트뿐만 아니라 롱텀 기억까지 적용? 근데 이것도 텍스트 아니냐..  기억하고 기억 확장(다듬는?) 과정을 거친다 이말인듯  


짧은 요약(Abstract) :


이 논문에서는 대규모 언어 모델(LLM)이 의료 분야에서 임상 노트 생성 및 환자 소통과 같은 작업을 지원하는 데 유망한 가능성을 보여주고 있지만, 현재 LLM이 텍스트 기반 커뮤니케이션에만 국한되어 있어 다양한 정보 형태와 상호작용하는 데 한계가 있음을 지적합니다. 이를 해결하기 위해, 저자들은 ClinicalAgent Bench (CAB)라는 포괄적인 의료 에이전트 벤치마크를 제안하며, 이는 5개의 주요 임상 차원에서 18개의 작업으로 구성되어 있습니다. 이 벤치마크를 기반으로 REFLECTOOL이라는 새로운 프레임워크를 소개하며, 이는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 성공적인 문제 해결 과정을 저장하여 장기 기억을 확장하고, 두 번째 단계에서는 이미 구축된 장기 기억에서 지원적인 성공 사례를 검색하여 도구 선택 전략을 안내합니다. REFLECTOOL은 기존의 LLM 및 에이전트 기반 방법보다 더 높은 성능을 보여주며, 복잡한 임상 작업을 해결하는 데 있어 적응성과 효과성을 강조합니다.


This paper highlights the promising potential of large language models (LLMs) in the medical domain, assisting with tasks such as clinical note generation and patient communication. However, it points out that current LLMs are limited to text-based communication, hindering their ability to interact with diverse forms of information. To address this limitation, the authors propose ClinicalAgent Bench (CAB), a comprehensive medical agent benchmark consisting of 18 tasks across five key clinical dimensions. Building on this benchmark, they introduce REFLECTOOL, a novel framework that operates in two stages. The first stage progressively expands long-term memory by saving successful problem-solving processes, while the second stage retrieves supportive successful demonstrations from the established long-term memory to guide the tool selection strategy. REFLECTOOL demonstrates superior performance compared to existing LLMs and agent-based methods, highlighting its adaptability and effectiveness in solving complex clinical tasks.


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


이 논문에서 제안하는 REFLECTOOL은 의료 분야에서의 임상 에이전트를 위한 혁신적인 도구 증강 프레임워크입니다. REFLECTOOL은 두 가지 주요 단계로 구성되어 있습니다: 최적화 단계와 추론 단계입니다.

1. **최적화 단계**: 이 단계에서는 에이전트가 작은 샘플 집합을 사용하여 문제를 해결하려고 시도합니다. 에이전트는 성공적인 해결 과정을 기록하고, 이를 장기 기억(long-term memory)에 저장합니다. 이 과정에서 에이전트는 성공적인 경로와 실패한 경로를 비교하여 도구 사용에 대한 제안을 생성합니다. 이 제안은 에이전트가 다음 단계에서 더 나은 결정을 내리는 데 도움을 줍니다.

2. **추론 단계**: 이 단계에서는 에이전트가 장기 기억에서 유사한 성공 사례를 검색하여 도구 선택 전략을 최적화합니다. 에이전트는 각 도구 사용 시, 최적화 단계에서 축적된 도구별 경험을 바탕으로 도구 사용을 개선합니다. 이 과정에서 두 가지 검증 방법인 반복 정제(Iterative Refinement)와 후보 선택(Candidate Selection)을 사용하여 도구 사용의 효과성을 평가합니다.

REFLECTOOL은 다양한 임상 시나리오에서의 적응성과 효율성을 높이기 위해 설계되었습니다. 이 프레임워크는 의료 지식, 다중 모달 데이터 처리, 수치 분석, 데이터 이해, 신뢰성 등 다섯 가지 주요 차원에서 에이전트의 성능을 평가하는 ClinicalAgent Bench (CAB)라는 포괄적인 벤치마크를 기반으로 합니다. REFLECTOOL은 기존의 대형 언어 모델(LLM)과 비교하여 10점 이상의 성능 향상을 보여주며, 이는 복잡한 임상 작업을 해결하는 데 있어 그 효과성을 입증합니다.




The REFLECTOOL proposed in this paper is an innovative tool-augmented framework for clinical agents in the medical domain. REFLECTOOL consists of two main stages: the optimization stage and the inference stage.

1. **Optimization Stage**: In this stage, the agent attempts to solve problems using a small set of samples. The agent records successful solving processes and saves them into long-term memory. During this process, the agent compares successful trajectories with failed ones to generate suggestions for tool usage. These suggestions help the agent make better decisions in subsequent steps.

2. **Inference Stage**: In this stage, the agent retrieves similar successful cases from long-term memory to optimize the tool selection strategy. Each time a tool is used, the agent improves its usage based on the accumulated tool-wise experience from the optimization stage. Two verification methods, namely Iterative Refinement and Candidate Selection, are employed to evaluate the effectiveness of tool usage.

REFLECTOOL is designed to enhance adaptability and efficiency across various clinical scenarios. This framework is built upon the ClinicalAgent Bench (CAB), a comprehensive benchmark that evaluates agent performance across five key dimensions: medical knowledge, multimodal data processing, numerical analysis, data understanding, and trustworthiness. REFLECTOOL demonstrates over 10 points of performance improvement compared to existing large language models (LLMs), showcasing its effectiveness in solving complex clinical tasks.


<br/>
# Results



이 논문에서는 REFLECTOOL이라는 새로운 프레임워크를 제안하고, ClinicalAgent Bench (CAB)라는 포괄적인 의료 에이전트 벤치마크를 기반으로 성능을 평가합니다. REFLECTOOL은 도구 사용 최적화를 위해 장기 기억과 도구별 검증을 활용하며, 두 가지 검증 방법인 반복 정제(Iterative Refinement)와 후보 선택(Candidate Selection)을 채택합니다.

#### 실험 결과
1. **경쟁 모델**: REFLECTOOL은 여러 기존 모델과 비교되었습니다. 여기에는 MedLlama, Qwen2, Llama3, GPT-3.5-turbo, ReAct, CRITIC, Reﬂexion 등이 포함됩니다. 각 모델은 CAB의 다양한 차원에서 성능을 평가받았습니다.

2. **테스트 데이터**: CAB는 18개의 다양한 임상 작업을 포함하고 있으며, 각 작업은 Knowledge & Reasoning, MultiModal, Numerical Analysis, Data Understanding, Trustworthiness의 다섯 가지 주요 차원으로 나뉩니다. 각 차원은 특정한 임상 시나리오를 반영하여 설계되었습니다.

3. **메트릭**: 성능 평가는 각 모델의 정확도(accuracy)로 측정되었습니다. 각 차원에서의 평균 성능을 비교하여 REFLECTOOL의 효과성을 입증했습니다.

4. **비교 결과**: REFLECTOOL은 기존의 LLMs 및 에이전트 기반 방법들에 비해 평균적으로 10점 이상의 성능 향상을 보였습니다. 특히, Qwen2-72B 모델을 사용할 경우, REFLECTOOL은 Reﬂexion보다 최소 3점 높은 성능을 기록했습니다. 이는 REFLECTOOL이 도구 사용에 있어 더 효과적임을 나타냅니다.

5. **도구 사용 오류**: REFLECTOOL은 도구 선택 오류를 줄이는 데에도 효과적이었습니다. 실험 결과, REFLECTOOL은 다른 에이전트 기반 방법들에 비해 도구 선택 오류율이 현저히 낮았습니다.

이러한 결과들은 REFLECTOOL이 복잡한 임상 작업을 해결하는 데 있어 높은 적응성과 효율성을 제공함을 보여줍니다.

---




This paper introduces a novel framework called REFLECTOOL and evaluates its performance based on a comprehensive medical agent benchmark known as ClinicalAgent Bench (CAB). REFLECTOOL optimizes tool utilization through long-term memory and tool-wise verification, employing two verification methods: Iterative Refinement and Candidate Selection.

#### Experimental Results
1. **Competing Models**: REFLECTOOL was compared against several existing models, including MedLlama, Qwen2, Llama3, GPT-3.5-turbo, ReAct, CRITIC, and Reﬂexion. Each model was evaluated across various dimensions of CAB.

2. **Test Data**: CAB consists of 18 diverse clinical tasks, categorized into five key dimensions: Knowledge & Reasoning, MultiModal, Numerical Analysis, Data Understanding, and Trustworthiness. Each dimension is designed to reflect specific clinical scenarios.

3. **Metrics**: Performance was measured in terms of accuracy. The average performance across each dimension was compared to demonstrate the effectiveness of REFLECTOOL.

4. **Comparison Results**: REFLECTOOL showed an average performance improvement of over 10 points compared to existing LLMs and agent-based methods. Notably, when using the Qwen2-72B model, REFLECTOOL outperformed Reﬂexion by at least 3 points, indicating its superior effectiveness in tool utilization.

5. **Tool Usage Errors**: REFLECTOOL also significantly reduced tool selection errors. Experimental results indicated that REFLECTOOL had a markedly lower error rate in tool selection compared to other agent-based methods.

These results demonstrate that REFLECTOOL provides high adaptability and efficiency in solving complex clinical tasks.


<br/>
# 예제



이 논문에서는 REFLECTOOL이라는 새로운 프레임워크를 제안하며, 이를 통해 의료 분야에서의 임상 에이전트의 성능을 향상시키기 위한 방법을 설명합니다. REFLECTOOL은 두 가지 주요 단계로 구성되어 있습니다: 최적화 단계와 추론 단계입니다.

1. **최적화 단계**:
   - 이 단계에서는 에이전트가 주어진 임상 문제를 해결하기 위해 도구를 사용하여 초기 경로(C1)를 생성합니다. 
   - 에이전트는 이 경로를 바탕으로 정답(y)과 비교하여 제안(S1)을 생성합니다. 
   - 이후, 제안에 따라 경로를 재생성하여 새로운 경로(C2)를 만듭니다. 
   - C2가 성공적으로 문제를 해결하면, 이 경로는 장기 기억(M)에 저장됩니다. 
   - 이 과정에서 각 도구의 사용 경험을 수집하여 도구별 경험(E)을 업데이트합니다.

2. **추론 단계**:
   - 이 단계에서는 에이전트가 장기 기억에서 유사한 사례를 검색하여 문제를 해결합니다. 
   - 에이전트는 도구별 경험을 바탕으로 최적의 도구를 선택하여 문제를 해결합니다. 
   - 이 과정에서 두 가지 검증 방법인 반복 정제(Iterative Refinement)와 후보 선택(Candidate Selection)을 사용하여 도구 사용의 효과성을 높입니다.

#### 예시
- **트레이닝 데이터**:
  - 입력: "환자가 경련을 일으켰습니다. 이 환자의 상태는 무엇인가요?"
  - 출력: "A. 뇌전증 B. 뇌종양 C. 기타"
  
- **테스트 데이터**:
  - 입력: "27세 남성이 응급실에 경련으로 입원했습니다. 이 환자의 증상은 무엇인가요?"
  - 출력: "A. 뇌전증"

이와 같은 방식으로 REFLECTOOL은 다양한 임상 시나리오에서 에이전트의 성능을 평가하고, 도구 사용을 최적화하여 더 나은 결과를 도출할 수 있도록 설계되었습니다.

---




This paper proposes a new framework called REFLECTOOL, which aims to enhance the performance of clinical agents in the medical field. REFLECTOOL consists of two main stages: the optimization stage and the inference stage.

1. **Optimization Stage**:
   - In this stage, the agent generates an initial trajectory (C1) by using tools to solve a given clinical problem.
   - The agent compares this trajectory with the ground truth (y) to generate a suggestion (S1).
   - Based on the suggestion, the agent regenerates a refined trajectory (C2).
   - If C2 successfully completes the task, this trajectory is saved into long-term memory (M).
   - During this process, the agent collects usage experiences for each tool to update the tool-wise experience (E).

2. **Inference Stage**:
   - In this stage, the agent retrieves similar cases from long-term memory to solve the task.
   - The agent selects the optimal tool based on the accumulated tool-wise experience to complete the task.
   - Two verification methods, Iterative Refinement and Candidate Selection, are employed to enhance the effectiveness of tool usage.

#### Example
- **Training Data**:
  - Input: "The patient has experienced seizures. What is the condition of this patient?"
  - Output: "A. Epilepsy B. Brain tumor C. Other"
  
- **Test Data**:
  - Input: "A 27-year-old male is brought to the emergency room after experiencing seizures. What is the condition of this patient?"
  - Output: "A. Epilepsy"

In this way, REFLECTOOL is designed to evaluate the performance of agents across various clinical scenarios and optimize tool usage to achieve better outcomes.

<br/>
# 요약

이 논문에서는 REFLECTOOL이라는 새로운 프레임워크를 제안하여 임상 에이전트가 도구를 효과적으로 활용할 수 있도록 지원합니다. 실험 결과, REFLECTOOL은 기존의 임상 에이전트보다 10점 이상 높은 성능을 보이며, 다양한 임상 시나리오에서의 적응성과 효율성을 입증했습니다. 이 프레임워크는 장기 기억과 도구별 검증을 통해 도구 사용을 최적화하는 두 가지 단계로 구성됩니다.

---

This paper introduces a novel framework called REFLECTOOL that aids clinical agents in effectively utilizing tools. Experimental results demonstrate that REFLECTOOL outperforms existing clinical agents by over 10 points, showcasing its adaptability and efficiency across various clinical scenarios. The framework consists of two stages that optimize tool usage through long-term memory and tool-wise verification.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **ClinicalAgent Bench (CAB) 개요**: CAB는 5개의 주요 차원에서 18개의 임상 작업을 평가하기 위한 벤치마크입니다. 이 다이어그램은 각 차원에서의 데이터 비율과 크기를 시각적으로 나타내어, 임상 에이전트의 성능을 종합적으로 평가할 수 있는 방법을 제시합니다.

2. **REFLECTOOL 개요**: REFLECTOOL의 구조를 보여주는 다이어그램은 최적화 단계와 추론 단계로 나뉘어 있습니다. 최적화 단계에서는 성공적인 경로를 저장하고, 추론 단계에서는 장기 기억에서 유사한 사례를 검색하여 도구 선택 전략을 최적화합니다.

3. **도구 분포**: 다양한 작업에서 REFLECTOOL이 사용하는 도구의 분포를 보여주는 그래프는 REFLECTOOL이 다른 방법들에 비해 더 많은 도구를 활용하고 있음을 나타냅니다. 이는 임상 작업을 수행하는 데 있어 더 높은 유연성과 효과성을 보여줍니다.

#### 테이블
1. **성능 비교 테이블**: 다양한 모델과 REFLECTOOL의 성능을 비교한 테이블은 REFLECTOOL이 기존의 임상 에이전트보다 평균적으로 10점 이상 높은 성능을 보임을 보여줍니다. 이는 REFLECTOOL의 도구 활용 전략이 효과적임을 나타냅니다.

2. **Ablation Study 결과**: 각 모듈의 효과를 검증하기 위한 ablation study 결과는 REFLECTOOL의 각 구성 요소가 성능에 미치는 영향을 보여줍니다. 특히, 장기 기억의 부재가 성능 저하에 큰 영향을 미친다는 점이 강조됩니다.

3. **도구 선택 오류**: 도구 선택 오류율을 비교한 테이블은 REFLECTOOL이 다른 방법들에 비해 도구 선택 오류를 현저히 줄였음을 보여줍니다. 이는 REFLECTOOL의 도구-wise 반영 메커니즘이 에이전트의 도구 사용 능력을 향상시킨다는 것을 시사합니다.

#### 어펜딕스
어펜딕스에서는 CAB의 각 차원과 작업에 대한 세부 정보를 제공하며, 각 도구의 구현 및 사용 방법에 대한 설명이 포함되어 있습니다. 이는 REFLECTOOL이 다양한 임상 시나리오에서 어떻게 적용될 수 있는지를 이해하는 데 도움을 줍니다.

---




#### Diagrams and Figures
1. **Overview of ClinicalAgent Bench (CAB)**: CAB is a benchmark designed to evaluate 18 clinical tasks across 5 key dimensions. This diagram visually represents the data proportions and sizes in each dimension, providing a comprehensive method for assessing the performance of clinical agents.

2. **Overview of REFLECTOOL**: The diagram illustrating REFLECTOOL's structure is divided into optimization and inference stages. In the optimization stage, successful trajectories are stored, while in the inference stage, similar cases are retrieved from long-term memory to optimize the tool selection strategy.

3. **Tool Distribution**: The graph showing the distribution of tools used by REFLECTOOL across various tasks indicates that REFLECTOOL utilizes a higher number of tools compared to other methods. This demonstrates greater flexibility and effectiveness in performing clinical tasks.

#### Tables
1. **Performance Comparison Table**: The table comparing the performance of various models and REFLECTOOL shows that REFLECTOOL outperforms existing clinical agents by an average of over 10 points. This indicates the effectiveness of REFLECTOOL's tool utilization strategy.

2. **Ablation Study Results**: The results of the ablation study, which validate the effectiveness of each module, highlight the significant impact of the absence of long-term memory on performance.

3. **Tool Selection Error**: The table comparing tool selection error rates shows that REFLECTOOL significantly reduces tool selection errors compared to other methods. This suggests that the tool-wise reflection mechanism of REFLECTOOL enhances the agent's ability to use appropriate tools.

#### Appendix
The appendix provides detailed information about each dimension and task of CAB, along with descriptions of the implementation and usage of each tool. This aids in understanding how REFLECTOOL can be applied across various clinical scenarios.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{Liao2025,
  author    = {Yusheng Liao and Shuyang Jiang and Yanfeng Wang and Yu Wang},
  title     = {REFLECTOOL: Towards Reflection-Aware Tool-Augmented Clinical Agents},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {13507--13531},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일 인용
Liao, Yusheng, Shuyang Jiang, Yanfeng Wang, and Yu Wang. 2025. "REFLECTOOL: Towards Reflection-Aware Tool-Augmented Clinical Agents." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 13507–13531. Association for Computational Linguistics.
