---
layout: post
title:  "[2026]CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents"
date:   2026-07-16 21:30:53 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 CostBench라는 새로운 벤치마크를 소개하여 대형 언어 모델(LLM) 에이전트의 비용 최적 계획 및 적응 능력을 평가합니다.


짧은 요약(Abstract) :



현재 대규모 언어 모델(LLM) 에이전트의 평가에서는 주로 작업 완료에 초점을 맞추고 있으며, 자원 효율성과 적응성은 간과되고 있습니다. 이는 에이전트가 변화하는 환경에 대응하여 비용 최적의 계획을 수립하고 조정하는 능력이라는 중요한 기능을 간과하는 것입니다. 이를 해결하기 위해, 우리는 CostBench라는 확장 가능한 비용 중심 벤치마크를 소개합니다. 이 벤치마크는 에이전트의 경제적 추론 및 재계획 능력을 평가하기 위해 설계되었습니다. 여행 계획 도메인에 위치한 CostBench는 다양한 사용자 정의 비용을 가진 원자 및 복합 도구의 여러 시퀀스를 통해 해결할 수 있는 작업으로 구성되어 있습니다. 또한 도구 고장 및 비용 변화와 같은 네 가지 유형의 동적 차단 이벤트를 지원하여 실제 세계의 예측 불가능성을 시뮬레이션하고 에이전트가 실시간으로 적응하도록 요구합니다. CostBench에서 선도적인 오픈 소스 및 상용 모델을 평가한 결과, 비용 인식 계획에서 상당한 격차가 드러났습니다. 에이전트는 정적 환경에서도 비용 최적의 솔루션을 식별하는 데 자주 실패하며, 가장 어려운 작업에서 GPT-5조차도 75% 미만의 정확도를 기록했습니다. 동적 조건에서는 성능이 더욱 크게 저하되었습니다. 이러한 약점을 진단함으로써, CostBench는 경제적으로 합리적이고 강력한 미래 에이전트를 개발하기 위한 기초를 마련합니다.

---




Current evaluations of Large Language Model (LLM) agents primarily emphasize task completion, often overlooking resource efficiency and adaptability. This neglects a crucial capability: agents’ ability to devise and adjust cost-optimal plans in response to changing environments. To bridge this gap, we introduce CostBench, a scalable, cost-centric benchmark designed to evaluate agents’ economic reasoning and replanning abilities. Situated in the travel-planning domain, CostBench comprises tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs. It also supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability and require agents to adapt in real time. Evaluating leading open-sourced and proprietary models on CostBench reveals a substantial gap in cost-aware planning: agents frequently fail to identify cost-optimal solutions in static settings, with even GPT-5 achieving less than a 75% exact match rate on the hardest tasks, and performance further drops significantly under dynamic conditions. By diagnosing these weaknesses, CostBench lays the groundwork for developing future agents that are both economically rational and robust.


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



이 논문에서는 CostBench라는 새로운 벤치마크를 소개하고 있습니다. CostBench는 대규모 언어 모델(LLM) 에이전트의 비용 최적 계획 및 적응 능력을 평가하기 위해 설계된 확장 가능한 벤치마크입니다. 이 벤치마크는 여행 계획 도메인에 기반하여 여러 개의 원자 도구와 복합 도구를 포함하고 있으며, 각 도구는 무작위로 할당된 비용을 가지고 있습니다. 이를 통해 에이전트가 다양한 비용 구조에 대해 얼마나 민감하게 반응하는지를 평가할 수 있습니다.

#### 모델 아키텍처
CostBench는 LLM이 도구를 사용하여 다단계 작업을 수행하는 능력을 평가합니다. 이 과정에서 LLM은 외부 도구와 상호작용하며, 각 도구 호출은 특정 비용을 발생시킵니다. 에이전트는 주어진 작업을 완료하기 위해 최적의 도구 호출 경로를 선택해야 하며, 이 경로는 비용이 최소화되어야 합니다. CostBench는 에이전트가 비용 최적의 경로를 찾는 능력을 평가하기 위해 여러 가지 동적 차단 이벤트를 도입합니다. 이러한 이벤트는 도구의 실패, 비용 변화, 사용자 선호의 변화 등을 포함하여 에이전트가 실시간으로 적응하고 재계획할 수 있도록 요구합니다.

#### 훈련 데이터 및 기법
CostBench는 다양한 사용자 선호 조합을 생성하여 훈련 데이터를 만듭니다. 각 작업은 사용자 요구 사항에 따라 다르게 구성되며, 이 요구 사항은 카테고리, 계층, 스타일, 기능 패키지 등으로 나뉩니다. 이러한 조합은 LLM이 실제 환경에서의 불확실성을 처리하는 데 필요한 다양한 시나리오를 제공합니다. 또한, CostBench는 에이전트의 성능을 평가하기 위해 여러 메트릭을 사용합니다. 여기에는 비용 격차, 평균 편집 거리, 정확한 일치 비율 등이 포함됩니다.

이러한 방법론을 통해 CostBench는 LLM의 비용 인식 및 계획 능력을 평가하는 데 있어 중요한 기초를 제공합니다. 이 벤치마크는 향후 LLM 에이전트의 경제적 합리성과 강건성을 향상시키기 위한 연구의 기초가 될 것입니다.

---



This paper introduces a new benchmark called CostBench, designed to evaluate the cost-optimal planning and adaptation capabilities of Large Language Model (LLM) agents. CostBench is situated in the travel planning domain and comprises multiple atomic and composite tools, each assigned a randomly generated cost. This allows for the assessment of how sensitive agents are to various cost structures.

#### Model Architecture
CostBench evaluates the ability of LLMs to perform multi-step tasks using tools. In this process, the LLM interacts with external tools, and each tool invocation incurs a specific cost. The agent must select the optimal tool call path to complete the given task, minimizing the total cost. CostBench introduces several dynamic blocking events to assess the agent's ability to find cost-optimal paths. These events include tool failures, cost changes, and shifts in user preferences, requiring the agent to adapt and replan in real-time.

#### Training Data and Techniques
CostBench generates training data by creating various combinations of user preferences. Each task is structured according to user requirements, which are categorized into dimensions such as category, tier, style, and feature package. These combinations provide scenarios that are essential for the LLM to handle uncertainties in real-world environments. Additionally, CostBench employs multiple metrics to evaluate agent performance, including cost gap, average edit distance, and exact match ratio.

Through this methodology, CostBench lays a crucial foundation for assessing the cost-awareness and planning capabilities of LLMs. This benchmark will serve as a basis for future research aimed at enhancing the economic rationality and robustness of LLM agents.


<br/>
# Results



이 논문에서는 CostBench라는 새로운 벤치마크를 소개하며, 이는 대형 언어 모델(LLM) 에이전트의 비용 최적 계획 및 적응 능력을 평가하기 위해 설계되었습니다. 기존의 LLM 평가가 주로 작업 완료에 중점을 두고 자원 효율성과 적응성을 간과하는 경향이 있었기 때문에, CostBench는 이러한 격차를 해소하고자 합니다.

#### 경쟁 모델
CostBench에서 평가된 모델은 다음과 같습니다:
- GPT-5
- Gemini-2.5-Pro
- Qwen 시리즈 (Qwen3-8B, Qwen3-14B, Qwen3-32B)
- Llama-3.1-8B-Instruct
- GLM-4.5
- Deepseek-V3.1
- Claude-Sonnet-4
- GPT-4o

이 모델들은 다양한 비용 최적화 작업을 수행하는 능력을 평가받았습니다.

#### 테스트 데이터
CostBench는 여행 계획 도메인에 기반하여 설계된 6개의 작업을 포함하고 있습니다. 각 작업은 사용자 요구 사항에 따라 다양한 경로를 통해 해결할 수 있으며, 각 도구의 비용은 무작위로 할당됩니다. 또한, CostBench는 도구 실패, 비용 변화 등 4가지 유형의 동적 차단 이벤트를 지원하여 실제 환경의 예측 불가능성을 시뮬레이션합니다.

#### 메트릭
모델의 성능은 다음과 같은 메트릭을 통해 평가되었습니다:
1. **비용 격차 (Cost Gap)**: 에이전트의 총 비용과 정답 경로의 총 비용 간의 차이를 측정합니다.
2. **평균 편집 거리 (Average Edit Distance, AED)**: 에이전트의 경로와 정답 경로 간의 구조적 유사성을 측정합니다.
3. **정확한 일치 비율 (Exact Match Ratio, EMR)**: 에이전트의 경로가 정답 경로와 정확히 일치하는 비율입니다.
4. **작업 완료 비율 (Task Completion Ratio, TCR)**: 에이전트가 올바른 최종 답변을 반환한 비율입니다.
5. **유효 도구 사용 비율 (Invalid Tool-Use Ratio, ITUR)**: 잘못된 도구 호출의 비율입니다.

#### 비교 결과
모델 성능 비교 결과, GPT-5가 가장 높은 성능을 보였지만, 가장 어려운 정적 설정에서 75% 미만의 정확한 일치 비율을 기록했습니다. 다른 모델들은 성능이 더 낮았으며, 특히 Qwen 시리즈 모델들은 그리디 정책과 유사한 성능을 보였습니다. 모든 모델은 동적 조건에서 성능이 크게 저하되었으며, 이는 현재 모델들이 비용 인식 및 적응 능력에서 한계를 가지고 있음을 나타냅니다.

이러한 결과는 CostBench가 LLM 에이전트의 비용 인식 및 적응 능력을 평가하는 데 있어 중요한 기초를 제공하며, 향후 연구에서 경제적으로 합리적이고 자원 효율적인 의사 결정을 할 수 있는 모델 개발을 촉진할 것으로 기대됩니다.

---




This paper introduces a new benchmark called CostBench, designed to evaluate the cost-optimal planning and adaptation abilities of Large Language Model (LLM) agents. Existing evaluations of LLMs have primarily focused on task completion, often overlooking resource efficiency and adaptability, which CostBench aims to address.

#### Competing Models
The models evaluated on CostBench include:
- GPT-5
- Gemini-2.5-Pro
- Qwen series (Qwen3-8B, Qwen3-14B, Qwen3-32B)
- Llama-3.1-8B-Instruct
- GLM-4.5
- Deepseek-V3.1
- Claude-Sonnet-4
- GPT-4o

These models were assessed for their ability to perform various cost-optimized tasks.

#### Test Data
CostBench is built around six tasks in the travel planning domain. Each task can be solved through various paths based on user requirements, with the costs of each tool assigned randomly. Additionally, CostBench supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability.

#### Metrics
Model performance was evaluated using the following metrics:
1. **Cost Gap**: Measures the difference between the total cost incurred by the agent and the total cost of the ground-truth trajectory.
2. **Average Edit Distance (AED)**: Assesses the structural similarity between the agent's trajectory and the ground-truth trajectory.
3. **Exact Match Ratio (EMR)**: The proportion of cases where the agent's trajectory exactly matches the ground-truth trajectory.
4. **Task Completion Ratio (TCR)**: The proportion of cases where the agent returns the unique correct answer.
5. **Invalid Tool-Use Ratio (ITUR)**: The proportion of invalid tool calls.

#### Comparison Results
In the performance comparison, GPT-5 achieved the highest performance but recorded less than 75% exact match rate on the hardest static settings. Other models performed even lower, with Qwen series models showing performance similar to a greedy policy. All models exhibited significant performance drops under dynamic conditions, indicating that current models have limitations in cost awareness and adaptability.

These results suggest that CostBench provides a crucial foundation for evaluating the cost awareness and adaptability of LLM agents, encouraging the development of models that can make economically rational and resource-efficient decisions in complex real-world scenarios.


<br/>
# 예제



**CostBench**는 대규모 언어 모델(LLM) 에이전트의 비용 최적 계획 및 적응 능력을 평가하기 위해 설계된 벤치마크입니다. 이 벤치마크는 여행 계획 도메인에 기반하여 다양한 사용자 요구 사항을 반영한 쿼리를 생성하고, 이를 통해 에이전트가 여러 도구를 사용하여 비용을 최소화하면서 목표를 달성할 수 있는지를 평가합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **쿼리 구성**: 각 쿼리는 사용자의 요구 사항을 기반으로 하며, 다음과 같은 네 가지 차원으로 구성됩니다:
     - **카테고리 요구 사항**: 예를 들어, "도시", "마을", "산" 등.
     - **티어 요구 사항**: "소규모 마을", "외진 지역" 등.
     - **스타일 우선순위**: "자연과 평화로운", "모험적인" 등.
     - **특징 패키지**: "미식의 수도", "역사적인" 등.
   - **예시**: 
     - 사용자 요구 사항: "나는 도시 카테고리에서 역사적인 장소를 원합니다."
     - 생성된 쿼리: "나는 역사적인 도시에서 모험을 찾고 있습니다."

2. **테스트 데이터**:
   - **테스트 쿼리**: 트레이닝 데이터와 유사하지만, 필터링 과정을 거쳐 일반 상식에 위배되지 않는 조합으로 구성됩니다.
   - **예시**:
     - 사용자 요구 사항: "나는 조용한 마을에서 자연을 즐기고 싶습니다."
     - 생성된 쿼리: "나는 평화로운 마을에서 자연을 만끽하고 싶습니다."

#### 구체적인 테스크

- **테스크**: 에이전트는 주어진 사용자 요구 사항에 따라 최적의 여행 계획을 세워야 합니다. 이 과정에서 에이전트는 여러 도구를 호출하여 정보를 검색하고, 후보지를 필터링하며, 최종 결정을 내려야 합니다.
- **도구 사용 예시**:
  - **도구 1**: "위치 선호 결정" - 사용자의 위치 선호를 결정합니다.
  - **도구 2**: "위치 후보 검색" - 사용자의 요구 사항에 맞는 위치 후보를 검색합니다.
  - **도구 3**: "위치 필터링" - 검색된 후보지 중에서 사용자의 추가 요구 사항에 따라 필터링합니다.
  - **도구 4**: "최종 위치 선택" - 최종적으로 선택된 위치를 결정합니다.

이러한 도구들은 각각 고유한 비용이 있으며, 에이전트는 최소 비용으로 목표를 달성해야 합니다. CostBench는 에이전트가 이러한 도구를 어떻게 활용하는지를 평가하여, 비용 최적화 및 적응 능력을 측정합니다.

---




**CostBench** is a benchmark designed to evaluate the cost-optimal planning and adaptation abilities of Large Language Model (LLM) agents. This benchmark is based in the travel planning domain and generates queries reflecting various user requirements, assessing whether agents can use multiple tools to achieve goals while minimizing costs.

#### Training Data and Test Data

1. **Training Data**:
   - **Query Construction**: Each query is based on user requirements and consists of four dimensions:
     - **Category Requirement**: For example, "city", "village", "mountain", etc.
     - **Tier Requirement**: "small_town", "secluded_area", etc.
     - **Style Priority**: "natural_and_serene", "adventurous", etc.
     - **Feature Package**: "culinary_capital", "historical", etc.
   - **Example**: 
     - User Requirement: "I want a historical place in the city category."
     - Generated Query: "I am looking for an adventure in a historical city."

2. **Test Data**:
   - **Test Queries**: Similar to training data but filtered to ensure no commonsense conflicts.
   - **Example**:
     - User Requirement: "I want to enjoy nature in a quiet village."
     - Generated Query: "I want to immerse myself in nature in a peaceful village."

#### Specific Tasks

- **Task**: The agent must create an optimal travel plan based on the given user requirements. In this process, the agent calls various tools to search for information, filter candidates, and make final decisions.
- **Tool Usage Example**:
  - **Tool 1**: "Decide Location Preference" - Determines the user's location preference.
  - **Tool 2**: "Search Location Candidates" - Searches for location candidates based on user requirements.
  - **Tool 3**: "Filter Location" - Filters the searched candidates based on additional user requirements.
  - **Tool 4**: "Select Final Location" - Decides on the final selected location.

These tools each have unique costs, and the agent must achieve the goal at the minimum cost. CostBench evaluates how agents utilize these tools to measure their cost optimization and adaptability capabilities.

<br/>
# 요약


이 논문에서는 CostBench라는 새로운 벤치마크를 소개하여 대형 언어 모델(LLM) 에이전트의 비용 최적 계획 및 적응 능력을 평가합니다. 실험 결과, 현재의 LLM 모델들은 정적 환경에서 75% 미만의 정확도를 보이며, 동적 조건에서는 성능이 35%로 급격히 떨어지는 것으로 나타났습니다. 이 연구는 LLM 에이전트의 비용 인식 및 적응 능력의 한계를 진단하고, 향후 연구 방향을 제시합니다.

---

This paper introduces CostBench, a new benchmark designed to evaluate the cost-optimal planning and adaptation capabilities of large language model (LLM) agents. Experimental results show that current LLM models achieve less than 75% accuracy in static environments, with performance dropping to around 35% under dynamic conditions. The study diagnoses the limitations of LLM agents in cost awareness and adaptability, providing directions for future research.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **다이어그램 (Figure 1)**: CostBench의 파이프라인 개요를 보여줍니다. 사용자 쿼리에서 시작하여 에이전트가 계획을 세우고, 도구를 사용하여 목표를 달성하는 과정을 시각적으로 나타냅니다. 이 다이어그램은 각 단계에서 에이전트가 수행해야 할 작업을 명확히 보여줍니다.

2. **피규어 (Figure 7)**: CostBench에서의 도구 실행 흐름을 보여줍니다. 각 도구가 어떻게 연결되어 있는지를 나타내며, 에이전트가 각 도구를 호출하는 순서를 명확히 합니다. 이 피규어는 도구 간의 의존성을 강조하여 에이전트가 올바른 순서로 도구를 호출해야 함을 보여줍니다.

3. **테이블 (Table 2)**: CostBench에서 생성된 쿼리의 통계 정보를 제공합니다. 각 여행 관련 작업에 대한 사용자 선호 조합의 수를 보여주며, 데이터의 다양성을 강조합니다. 이 테이블은 다양한 사용자 요구를 반영하는 데 중요한 역할을 합니다.

4. **어펜딕스 (Appendix)**: 실험 설정, 도구 구성, 메트릭 정의 등 다양한 세부 정보를 제공합니다. 이 정보는 연구의 재현성을 높이고, 다른 연구자들이 CostBench를 활용할 수 있도록 돕습니다.

#### 결과 및 인사이트
- **CostBench의 설계**: CostBench는 에이전트의 비용 최적화 계획 능력을 평가하기 위해 설계된 확장 가능한 벤치마크입니다. 다양한 도구와 동적 차단 이벤트를 통해 에이전트의 적응 능력을 평가합니다.
  
- **모델 성능 분석**: 여러 모델을 평가한 결과, 대부분의 모델이 비용 최적화 계획에서 낮은 성능을 보였습니다. 특히, GPT-5는 가장 높은 성능을 보였지만, 여전히 75% 미만의 정확도를 기록했습니다. 이는 현재 모델들이 복잡한 동적 환경에서 비용 인식 및 적응 능력이 부족함을 나타냅니다.

- **비용 인식의 중요성**: 에이전트가 비용 인식 능력을 갖추는 것은 실제 환경에서의 효율적인 의사결정에 필수적입니다. CostBench는 이러한 능력을 평가하는 데 중요한 역할을 하며, 향후 연구에서 더 나은 에이전트를 개발하는 기초를 제공합니다.

---



#### Diagrams and Figures
1. **Diagram (Figure 1)**: It provides an overview of the CostBench pipeline. It visually represents the process starting from user queries, where the agent constructs a plan and uses tools to achieve its goals. This diagram clearly illustrates the tasks the agent must perform at each stage.

2. **Figure (Figure 7)**: It shows the execution flow of tools in CostBench. It indicates how each tool is connected and clarifies the order in which the agent should call each tool. This figure emphasizes the dependencies between tools, highlighting the need for the agent to call them in the correct sequence.

3. **Table (Table 2)**: It provides statistical information about the queries generated in CostBench. It shows the number of user preference combinations for each travel-related task, emphasizing the diversity of the data. This table plays a crucial role in reflecting various user needs.

4. **Appendix**: It offers various details such as experimental settings, tool construction, and metric definitions. This information enhances the reproducibility of the research and helps other researchers utilize CostBench.

#### Results and Insights
- **Design of CostBench**: CostBench is a scalable benchmark designed to evaluate agents' ability to plan cost-optimally. It assesses agents' adaptability through various tools and dynamic blocking events.

- **Model Performance Analysis**: The evaluation of several models revealed that most performed poorly in cost-optimal planning. Notably, GPT-5 achieved the highest performance but still recorded less than 75% accuracy. This indicates that current models lack cost awareness and adaptability in complex dynamic environments.

- **Importance of Cost Awareness**: Equipping agents with cost awareness is essential for efficient decision-making in real-world scenarios. CostBench plays a vital role in assessing this capability and lays the groundwork for developing better agents in future research.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{liu2026costbench,
  title={CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents},
  author={Jiayu Liu and Cheng Qian and Zhaochen Su and Qing Zong and Shijue Huang and Bingxiang He and Yi R. (May) Fung},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={12826--12858},
  year={2026},
  month={July},
  publisher={Association for Computational Linguistics}
}
```

### Chicago Style Citation

Jiayu Liu, Cheng Qian, Zhaochen Su, Qing Zong, Shijue Huang, Bingxiang He, and Yi R. (May) Fung. "CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 12826–12858. July 2026. Association for Computational Linguistics.
