---
layout: post
title:  "[2026]CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation"
date:   2026-07-14 01:03:42 -0000
categories: study
---

{% highlight ruby %}

한줄 요약:  계획과 코드 생성을 동시 발전시키는 협업적 의사결정 모듈을 통해 코드 생성의 효율성을 향상시키는 새로운 프레임워크  


짧은 요약(Abstract) :


자동화된 코드 생성은 소프트웨어 공학에서 지속적인 도전 과제가 되고 있으며, 기존의 다중 에이전트 프레임워크는 정적 계획, 고립된 실행, 높은 계산 오버헤드 및 복잡한 작업에 대한 제한된 적응성으로 인해 제약을 받습니다. 이 논문에서는 코드 생성을 개선하기 위해 동적 다중 에이전트 협업을 통한 새로운 계획-코드 공동 진화 프레임워크인 CollabCoder를 소개합니다. 핵심 아이디어는 계획 에이전트와 코드 에이전트 간의 협업적 의사결정 모듈을 설계하여 디버깅 과정에서 어떤 작업을 실행할지를 결정하는 것입니다. 광범위한 실험을 통해 CollabCoder는 다양한 작업에서 코드 품질과 견고성을 일관되게 향상시키며, 현재의 최첨단 방법과 비교할 때 성능이 동등하거나 이를 초과하면서 계산 오버헤드를 줄이는 데 성공했습니다. 특히, CollabCoder는 LiveCodeBench 및 xCodeEval과 같은 더 어려운 벤치마크에서 성능을 11-20% 향상시키고, 실행당 평균 4-10개의 API 호출을 줄였습니다.



Automated code generation remains a persistent challenge in software engineering, as conventional multi-agent frameworks are often constrained by static planning, isolated execution, high computational overhead, and limited adaptability to complex tasks. This paper introduces CollabCoder, a novel Plan-Code Co-Evolution framework that improves code generation through dynamic multi-agent collaboration. The core idea is to design a collaborative decision-making module between the plan agent and the code agent to decide which should be executed for the debugging process. Extensive experiments on widely used benchmarks demonstrate that CollabCoder consistently improves code quality and robustness across tasks. Importantly, CollabCoder achieves performance comparable to or exceeding current state-of-the-art methods while reducing computational overhead, with efficiency gains becoming more pronounced as benchmark difficulty increases. On the more challenging LiveCodeBench and xCodeEval benchmarks, our approach improves performance by 11-20% over strong baselines while reducing the number of API calls by an average of 4-10 per execution.


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


**CollabCoder 메서드 개요**

CollabCoder는 코드 생성을 위한 혁신적인 프레임워크로, 계획(Plan)과 코드(Code)의 공동 진화를 통해 효율적인 코드 생성을 목표로 합니다. 이 메서드는 다중 에이전트 협업을 통해 코드 생성의 품질과 강건성을 향상시키며, 기존의 정적 계획 및 고립된 실행 방식의 한계를 극복합니다.

**1. 아키텍처**

CollabCoder는 세 가지 주요 에이전트로 구성됩니다: 계획 에이전트(A_plan), 코드 에이전트(A_code), 디버깅 에이전트(A_debug). 이들은 협력적 의사결정 모듈(ACDM)을 통해 상호작용하며, 각 에이전트는 다음과 같은 역할을 수행합니다.

- **계획 에이전트(A_plan)**: 주어진 문제에 대한 해결책을 제시하는 계획을 생성합니다. 이 계획은 문제의 요구 사항을 충족하는 알고리즘적 접근 방식을 포함합니다.
  
- **코드 에이전트(A_code)**: 계획에 따라 실행 가능한 코드를 생성합니다. 이 과정에서 코드의 구조와 형식을 준수해야 합니다.

- **디버깅 에이전트(A_debug)**: 코드 실행 결과를 분석하고, 발생한 오류를 진단하여 계획 또는 코드의 수정을 결정합니다. 이 에이전트는 협력적 의사결정 모듈을 통해 계획과 코드의 일관성을 평가합니다.

**2. 협력적 의사결정 모듈(ACDM)**

ACDM은 두 가지 주요 단계로 구성됩니다: 분석 단계와 결정 단계입니다. 분석 단계에서는 현재 상태를 여러 관점에서 평가하고, 결정 단계에서는 분석 결과를 바탕으로 계획을 업데이트할지 코드 수정을 할지를 결정합니다. 이 과정에서 계획 수준 분석, 코드 수준 분석, 계획-코드 정렬 분석을 수행하여 각 요소의 신뢰성을 평가합니다.

**3. Reasoning Trajectory 모듈**

Reasoning Trajectory 모듈은 이전의 디버깅 전략을 유지하면서 새로운 진단 정보를 통합하여 지속적인 개선을 가능하게 합니다. 이 모듈은 각 반복에서 발생한 오류를 기록하고, 이를 바탕으로 다음 반복에서의 수정 방향을 제시합니다. 이를 통해 CollabCoder는 반복적인 수정 과정에서 발생할 수 있는 비효율성을 줄이고, 더 나은 성능을 발휘할 수 있습니다.

**4. 실험 및 평가**

CollabCoder는 HumanEval, MBPP, LiveCodeBench, xCodeEval과 같은 다양한 벤치마크에서 평가되었습니다. 실험 결과, CollabCoder는 기존의 최첨단 방법들과 비교하여 코드 품질과 효율성을 모두 향상시켰으며, 특히 복잡한 문제에 대한 성능이 두드러졌습니다.




**Overview of the CollabCoder Method**

CollabCoder is an innovative framework for code generation that aims to achieve efficient code generation through the co-evolution of plans and code. This method enhances the quality and robustness of code generation through multi-agent collaboration, overcoming the limitations of traditional static planning and isolated execution approaches.

**1. Architecture**

CollabCoder consists of three main agents: the planning agent (A_plan), the coding agent (A_code), and the debugging agent (A_debug). These agents interact through a Collaborative Decision-Making module (ACDM), with each agent performing the following roles:

- **Planning Agent (A_plan)**: Generates a plan that outlines a solution to the given problem. This plan includes an algorithmic approach that meets the requirements of the problem.

- **Coding Agent (A_code)**: Produces executable code based on the generated plan. This process must adhere to the structure and format of the code.

- **Debugging Agent (A_debug)**: Analyzes the execution results of the code, diagnoses any errors, and decides whether to update the plan or refine the code. This agent evaluates the consistency between the plan and the code through the collaborative decision-making module.

**2. Collaborative Decision-Making Module (ACDM)**

The ACDM consists of two main phases: the analysis phase and the decision phase. In the analysis phase, the current state is evaluated from multiple perspectives, and in the decision phase, the results of the analysis are used to determine whether to update the plan or modify the code. This process involves performing plan-level analysis, code-level analysis, and plan-code alignment analysis to assess the reliability of each component.

**3. Reasoning Trajectory Module**

The Reasoning Trajectory module enables continuous improvement by integrating new diagnostic information while maintaining the previous debugging strategy. This module records errors that occur in each iteration and suggests directions for modifications in the next iteration. By doing so, CollabCoder reduces inefficiencies that may arise during repetitive modification processes and can deliver better performance.

**4. Experiments and Evaluation**

CollabCoder has been evaluated on various benchmarks, including HumanEval, MBPP, LiveCodeBench, and xCodeEval. Experimental results demonstrate that CollabCoder improves both code quality and efficiency compared to existing state-of-the-art methods, particularly excelling in performance on complex problems.


<br/>
# Results


**결과 요약**

이 논문에서는 CollabCoder라는 새로운 코드 생성 프레임워크를 제안하고, 이를 기존의 경쟁 모델들과 비교하여 성능을 평가하였다. 실험은 HumanEval, MBPP, LiveCodeBench, xCodeEval과 같은 다양한 벤치마크 데이터셋을 사용하여 진행되었으며, 각 데이터셋의 난이도에 따라 성능을 측정하였다.

**경쟁 모델**

CollabCoder는 여러 기존 모델과 비교되었다. 이들 모델은 다음과 같다:
- **Direct Prompting**: 언어 모델이 직접적으로 코드를 생성하는 방식.
- **Chain-of-Thought (CoT)**: 모델이 중간 자연어 추론을 생성하여 코드 생성을 유도하는 방식.
- **Self-Planning**: 계획 단계와 구현 단계를 분리하여 진행하는 방식.
- **MapCoder**: 다중 에이전트 아키텍처를 사용하여 예제 검색, 계획, 코딩 및 디버깅을 포함하는 방식.
- **CodeSIM**: 시뮬레이션 기반의 입력/출력 실행을 포함하여 반복적인 검증을 수행하는 방식.

**테스트 데이터**

테스트 데이터는 다음과 같은 벤치마크로 구성되었다:
- **HumanEval**: 164개의 문제로 구성된 데이터셋.
- **MBPP**: 397개의 문제로 구성된 데이터셋.
- **LiveCodeBench**: 실제 코딩 대회와 유사한 문제를 포함하는 데이터셋.
- **xCodeEval**: 경쟁 프로그래밍 문제를 평가하기 위한 데이터셋.

**메트릭**

성능 평가는 다음과 같은 메트릭을 사용하여 이루어졌다:
- **Pass@1**: 주어진 문제에 대해 모델이 생성한 코드가 테스트 케이스를 통과하는 비율.
- **Token I/O**: 입력 및 출력 토큰의 평균 수.
- **API 호출 수**: 문제당 평균 API 호출 수.

**비교 결과**

CollabCoder는 모든 벤치마크에서 기존 모델들보다 우수한 성능을 보였다. 예를 들어, LiveCodeBench와 xCodeEval에서 CollabCoder는 각각 41.96%와 47.16%의 Pass@1 정확도를 기록하며, MapCoder와 CodeSIM보다 각각 6.6-7.1% 및 4.7-5.3% 높은 성능을 보였다. 또한, CollabCoder는 총 토큰 소비량을 MapCoder보다 약 57% 줄이고, CodeSIM보다 42% 줄이는 등 효율성에서도 우수한 성과를 나타냈다.

이러한 결과는 CollabCoder가 복잡한 문제를 해결하는 데 있어 더 나은 적응성과 안정성을 제공함을 보여준다. 특히, 문제의 난이도가 증가할수록 CollabCoder의 성능이 더욱 두드러지며, 이는 계획과 코드의 동적 공동 진화 덕분이다.




**Summary of Results**

This paper introduces a novel code generation framework called CollabCoder and evaluates its performance against existing competitive models. Experiments were conducted using various benchmark datasets such as HumanEval, MBPP, LiveCodeBench, and xCodeEval, measuring performance based on the difficulty of each dataset.

**Competitive Models**

CollabCoder was compared with several existing models, including:
- **Direct Prompting**: A method where the language model generates code directly.
- **Chain-of-Thought (CoT)**: A method that encourages the model to generate intermediate natural language reasoning to guide code generation.
- **Self-Planning**: A method that separates the process into planning and implementation phases.
- **MapCoder**: A multi-agent architecture covering example retrieval, planning, coding, and debugging.
- **CodeSIM**: A framework that incorporates simulated input/output execution for iterative verification.

**Test Data**

The test data consisted of the following benchmarks:
- **HumanEval**: A dataset comprising 164 problems.
- **MBPP**: A dataset with 397 problems.
- **LiveCodeBench**: A dataset containing problems that closely resemble real-world coding competitions.
- **xCodeEval**: A dataset designed to evaluate competitive programming problems.

**Metrics**

Performance evaluation was conducted using the following metrics:
- **Pass@1**: The ratio of test cases passed by the code generated by the model for a given problem.
- **Token I/O**: The average number of input and output tokens.
- **API Calls**: The average number of API calls per problem.

**Comparison Results**

CollabCoder outperformed all existing models across all benchmarks. For instance, on LiveCodeBench and xCodeEval, CollabCoder achieved Pass@1 accuracies of 41.96% and 47.16%, respectively, surpassing MapCoder and CodeSIM by approximately 6.6-7.1% and 4.7-5.3%. Additionally, CollabCoder reduced total token consumption by about 57% compared to MapCoder and by 42% compared to CodeSIM, demonstrating superior efficiency.

These results indicate that CollabCoder provides better adaptability and robustness in solving complex problems. Notably, as the difficulty of the problems increases, the performance of CollabCoder becomes more pronounced, thanks to its dynamic co-evolution of plans and code.


<br/>
# 예제



**CollabCoder의 예시:**

CollabCoder는 코드 생성 문제를 해결하기 위해 설계된 프레임워크로, 주어진 문제 설명에 따라 코드를 생성하고, 이를 디버깅하는 과정을 반복합니다. 이 과정에서 사용되는 트레이닝 데이터와 테스트 데이터의 구체적인 예시는 다음과 같습니다.

#### 트레이닝 데이터 예시

1. **문제 설명**: "이진 탐색 트리(BST) 클래스를 생성하고, 삽입, 검색, 삭제 메서드를 포함하되, 엣지 케이스 처리 및 균형 트리를 위한 최적화를 포함하라."
   
   - **샘플 입력/출력**:
     - 입력: 
       ```python
       bst = BinarySearchTree()
       bst.insert(50)
       bst.insert(30)
       bst.insert(70)
       bst.search(30)  # True
       bst.delete(30)
       bst.search(30)  # False
       ```
     - 출력: 
       ```python
       True
       False
       ```

2. **문제 설명**: "주어진 리스트에서 k번째 요소를 제거하는 함수를 작성하라."
   
   - **샘플 입력/출력**:
     - 입력: 
       ```python
       remove_kth_element([1, 2, 3, 4, 5], 3)
       ```
     - 출력: 
       ```python
       [1, 2, 4, 5]
       ```

#### 테스트 데이터 예시

1. **문제 설명**: "주어진 리스트에서 모든 짝수를 제거하는 함수를 작성하라."
   
   - **샘플 입력/출력**:
     - 입력: 
       ```python
       remove_even_numbers([1, 2, 3, 4, 5, 6])
       ```
     - 출력: 
       ```python
       [1, 3, 5]
       ```

2. **문제 설명**: "리스트의 첫 번째와 마지막 요소를 각각 분리하여 반환하는 함수를 작성하라."
   
   - **샘플 입력/출력**:
     - 입력: 
       ```python
       merge_first_last([[1, 2], [3, 4], [5, 6]])
       ```
     - 출력: 
       ```python
       [[1, 3, 5], [2, 4, 6]]
       ```

이러한 예시들은 CollabCoder가 문제를 이해하고, 적절한 코드를 생성하며, 디버깅을 통해 최종적으로 정확한 결과를 도출하는 데 사용됩니다. 각 문제는 자연어로 설명되며, 샘플 입력과 출력은 코드의 기능을 검증하는 데 사용됩니다.

---




**Example of CollabCoder:**

CollabCoder is a framework designed to solve code generation problems by generating code based on given problem descriptions and iteratively debugging it. The specific examples of training data and test data used in this process are as follows.

#### Training Data Example

1. **Problem Description**: "Create a Binary Search Tree (BST) class with insert, search, and delete methods, including edge case handling and optimization for balanced trees."
   
   - **Sample Input/Output**:
     - Input: 
       ```python
       bst = BinarySearchTree()
       bst.insert(50)
       bst.insert(30)
       bst.insert(70)
       bst.search(30)  # True
       bst.delete(30)
       bst.search(30)  # False
       ```
     - Output: 
       ```python
       True
       False
       ```

2. **Problem Description**: "Write a function to remove the k-th element from a given list."
   
   - **Sample Input/Output**:
     - Input: 
       ```python
       remove_kth_element([1, 2, 3, 4, 5], 3)
       ```
     - Output: 
       ```python
       [1, 2, 4, 5]
       ```

#### Test Data Example

1. **Problem Description**: "Write a function to remove all even numbers from a given list."
   
   - **Sample Input/Output**:
     - Input: 
       ```python
       remove_even_numbers([1, 2, 3, 4, 5, 6])
       ```
     - Output: 
       ```python
       [1, 3, 5]
       ```

2. **Problem Description**: "Write a function to merge the first and last elements separately in a list of lists."
   
   - **Sample Input/Output**:
     - Input: 
       ```python
       merge_first_last([[1, 2], [3, 4], [5, 6]])
       ```
     - Output: 
       ```python
       [[1, 3, 5], [2, 4, 6]]
       ```

These examples illustrate how CollabCoder understands the problem, generates appropriate code, and ultimately derives accurate results through debugging. Each problem is described in natural language, and the sample inputs and outputs are used to validate the functionality of the code.

<br/>
# 요약



CollabCoder는 계획과 코드 생성을 동시 발전시키는 협업적 의사결정 모듈을 통해 코드 생성의 효율성을 향상시키는 새로운 프레임워크입니다. 실험 결과, CollabCoder는 기존의 최첨단 방법들보다 11-20%의 성능 향상과 함께 API 호출 수를 평균 4-10회 줄이는 성과를 보였습니다. 이 프레임워크는 복잡한 프로그래밍 작업에서도 높은 정확도와 효율성을 유지하며, 코드 품질과 견고성을 지속적으로 개선합니다.




CollabCoder is a novel framework that enhances code generation efficiency through a collaborative decision-making module that co-evolves planning and coding. Experimental results demonstrate that CollabCoder achieves a performance improvement of 11-20% over existing state-of-the-art methods while reducing the average number of API calls by 4-10. This framework maintains high accuracy and efficiency even in complex programming tasks, continuously improving code quality and robustness.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: CollabCoder의 구조를 보여주는 다이어그램으로, 전통적인 코드 생성 방식과 CollabCoder의 차별점을 강조합니다. CollabCoder는 실행 중에 계획을 수정할 수 있는 능력을 가지고 있어, 여러 에이전트가 협력하여 중간 결과를 평가하고 필요한 업데이트를 결정합니다.
   - **Figure 2**: CollabCoder의 아키텍처를 설명하며, 동적 계획 에이전트와 적응형 코딩 에이전트, 협업 디버그 에이전트 간의 상호작용을 보여줍니다. 이 구조는 지속적인 피드백 루프를 통해 계획과 코드의 공동 진화를 가능하게 합니다.
   - **Figure 6**: CollabCoder의 자기 개선 디버깅 과정을 보여주는 예시로, 초기 솔루션에서 발생한 오류를 식별하고 수정하는 과정을 단계별로 설명합니다. 이는 계획과 코드의 정렬을 명확히 하여 문제를 해결하는 방법을 보여줍니다.

2. **테이블**
   - **Table 1**: 다양한 벤치마크 데이터셋에서 CollabCoder의 정확도와 효율성을 비교한 결과를 보여줍니다. CollabCoder는 다른 방법들에 비해 높은 정확도를 유지하면서도 API 호출 수와 토큰 사용량을 줄이는 데 성공했습니다.
   - **Table 2**: LiveCodeBench와 xCodeEval과 같은 복잡한 코드 생성 벤치마크에서의 성능을 비교합니다. CollabCoder는 다른 최신 방법들보다 높은 정확도와 효율성을 보여주며, 특히 어려운 문제에서 더 나은 성능을 발휘합니다.
   - **Table 3**: CDM(협업 의사결정 모듈)과 RT(추론 경로 모듈)의 영향을 분석한 결과로, 두 모듈이 모두 활성화된 경우 CollabCoder의 성능이 가장 높다는 것을 보여줍니다.

3. **어펜딕스**
   - 어펜딕스에서는 CollabCoder의 구현 세부사항과 각 모듈의 작동 방식을 설명하는 알고리즘을 제공합니다. 또한, 다양한 프롬프트 템플릿을 통해 문제 해결 과정에서의 입력과 출력을 명확히 정의하고 있습니다. 이는 재현성을 높이고, 다른 연구자들이 동일한 방법론을 적용할 수 있도록 돕습니다.

### Insights in English

1. **Diagrams and Figures**
   - **Figure 1**: This diagram illustrates the structure of CollabCoder, highlighting the differences from traditional code generation methods. CollabCoder's ability to revise plans during execution allows multiple agents to collaborate in assessing intermediate results and deciding on necessary updates.
   - **Figure 2**: It describes the architecture of CollabCoder, showcasing the interactions between the dynamic planning agent, adaptive coding agent, and collaborative debug agent. This structure enables continuous feedback loops that facilitate the co-evolution of plans and code.
   - **Figure 6**: An example of CollabCoder's self-improving debugging process, illustrating how it identifies and corrects errors in the initial solution step-by-step. This demonstrates how to clarify the alignment between plans and code to solve problems effectively.

2. **Tables**
   - **Table 1**: It presents a comparison of accuracy and efficiency of CollabCoder across various benchmark datasets. CollabCoder successfully maintains higher accuracy while reducing the number of API calls and token usage compared to other methods.
   - **Table 2**: This table compares performance on complex code generation benchmarks like LiveCodeBench and xCodeEval. CollabCoder shows superior accuracy and efficiency, particularly excelling in more challenging problems.
   - **Table 3**: It analyzes the impact of the Collaborative Decision-Making (CDM) and Reasoning Trajectory (RT) modules, demonstrating that performance is highest when both modules are activated.

3. **Appendix**
   - The appendix provides detailed implementation specifics of CollabCoder, including algorithms that describe how each module operates. It also includes various prompt templates that clearly define inputs and outputs during the problem-solving process, enhancing reproducibility and enabling other researchers to apply the same methodology.

<br/>
# refer format:   
### BibTeX Citation

```bibtex
@inproceedings{Doan2026,
  author    = {Duy Tung Doan and Quang Huy Phung and Ngoc Dung Nguyen and Khac-Hoai Nam Bui},
  title     = {CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {27964--27985},
  year      = {2026},
  month     = {July 2-7},
  publisher  = {Association for Computational Linguistics},
}
```

### Chicago Style Citation

Doan, Duy Tung, Quang Huy Phung, Ngoc Dung Nguyen, and Khac-Hoai Nam Bui. "CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation." In *Findings of the Association for Computational Linguistics: ACL 2026*, 27964–27985. July 2-7, 2026. Association for Computational Linguistics.   