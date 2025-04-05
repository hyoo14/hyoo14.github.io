---
layout: post
title:  "[2024]Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios"  
date:   2025-04-05 03:32:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

LLM이 실제 복잡한 시나리오에서 도구를 계획·생성·활용하는 능력을 평가하기위한 벤치마크 제공   
예를 들어 "500달러와 300유로를 현재 환율로 위안화로 환전한 총 금액을 계산해주세요." 일 경우,  
환율 찾고(도구사용), 계산하고(도구사용), 총합      


짧은 요약(Abstract) :    




---


최근 대형 언어 모델(LLM)을 실제 응용에서 도구(agent)로 활용하려는 흐름이 늘어나면서, 이들이 복잡한 실제 상황에서 어떻게 도구를 기획·생성·사용할 수 있는지를 평가하는 것이 중요해졌습니다. 그러나 기존 벤치마크는 대부분 단순한 질문 위주로 설계되어 실제 도구 활용 능력을 평가하는 데 한계가 있습니다. 이 논문은 이러한 문제를 해결하기 위해 **UltraTool**이라는 새로운 벤치마크를 제안합니다. UltraTool은 현실적인 복잡한 시나리오 속에서의 도구 활용 과정을 **계획(Planning)**, **도구 생성(Creation)**, **도구 사용(Usage)**의 세 가지 측면, 총 6가지 하위 평가 항목으로 세분화하여 종합적으로 평가합니다. 특히 사전 정의된 도구 목록에 얽매이지 않고, 자연어 기반 계획 과정을 독립적으로 평가한다는 점이 특징입니다. 다양한 LLM에 대해 실험을 수행하여 도구 활용 능력을 심층 분석하였으며, 관련 연구 분야에 새로운 관점을 제시합니다.

---



The recent trend of using Large Language Models (LLMs) as tool agents in real-world applications underscores the necessity for comprehensive evaluations of their capabilities, particularly in complex scenarios involving planning, creating, and using tools. However, existing benchmarks typically focus on simple synthesized queries that do not reflect real-world complexity, thereby offering limited perspectives in evaluating tool utilization. To address this issue, we present UltraTool, a novel benchmark designed to improve and evaluate LLMs’ ability in tool utilization within real-world scenarios. UltraTool focuses on the entire process of using tools - from planning and creating to applying them in complex tasks. It emphasizes real-world complexities, demanding accurate, multi-step planning for effective problem-solving. A key feature of UltraTool is its independent evaluation of planning with natural language, which happens before tool usage and simplifies the task solving by mapping out the intermediate steps. Thus, unlike previous work, it eliminates the restriction of pre-defined toolset. Through extensive experiments on various LLMs, we offer novel insights into the evaluation of capabilities of LLMs in tool utilization, thereby contributing a fresh perspective to this rapidly evolving field. The benchmark is publicly available at https://github.com/JoeYing1019/UltraTool.





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




### 1. 전반적인 구성 개요
UltraTool은 대형 언어 모델(LLM)의 도구 활용 능력을 종합적으로 평가하기 위해 고안된 벤치마크로, 현실 세계의 복잡한 작업을 반영한 질의(Query)를 바탕으로 다음과 같은 세 가지 핵심 요소를 포함합니다:
- **계획 수립 (Planning)**: 복잡한 목표를 단계별 작업으로 세분화
- **도구 생성 (Tool Creation)**: 기존 도구가 부족할 경우 새로운 도구 설계
- **도구 사용 (Tool Usage)**: 적절한 도구를 선택해 실제 입력값을 넣어 문제 해결

### 2. 구성 절차

#### (1) 질의 수집 (Query Collection)
- **20개 이상의 실제 도메인**(예: 금융, 호텔, 항공, 알람 등)에서 도메인 전문가들이 복잡한 실제 사용자 질의를 수집.
- 이 질의들은 GPT-4를 통해 **일반화(Generalization)** 및 **복잡화(Complication)** 과정을 거쳐 더욱 다양한 형태로 확장됨.
- 이후 모든 질의는 전문가들이 수동으로 검토하여 최종 정제됨.

#### (2) 솔루션 주석 (Solution Annotation)
GPT-4를 활용하여 아래의 단계별로 자동 주석 처리됨:
- **계획 작성**: 트리 구조로 자연어 기반 작업 계획 수립
- **도구 생성 및 계획 개선**: 도구가 부족한 경우 새로운 도구 생성 + 기존 계획 보완
- **도구 호출 메시지 생성**: 각 단계별로 호출해야 할 도구와 인자 정의
- **도구 통합**: 유사한 기능의 도구들을 병합하여 중복 제거

#### (3) 수동 정제 (Manual Refinement)
- 총 6명의 전문가가 모든 데이터 샘플을 **2회 이상 수동으로 검토**함.
- 불필요한 단계 제거, 누락된 단계 보완, 도구 선택 오류 수정 등을 포함.

### 3. 데이터 통계
- 총 **5,824개 샘플**, **22개 도메인**, **2,032개 도구**
- 각 샘플당 평균 **12.27개의 작업 단계**, **2.74번 도구 호출**, 각 도구 호출 시 평균 **3.05개 인자**
- **39.61%**의 샘플에서 도구 호출이 중첩됨 (nested)

### 4. 학습 (Training)
- UltraTool은 **학습용 벤치마크가 아니라 평가용**으로 사용되며, 학습은 따로 진행되지 않음.
- 대신, GPT-4, GPT-3.5, LLaMA2, Mistral 등 다양한 모델의 **사전학습 모델을 평가**함.

---



### 1. Overview
UltraTool is a comprehensive benchmark that evaluates LLMs' tool utilization ability across:
- **Planning**: Decomposing complex user queries into structured sub-steps
- **Tool Creation**: Creating new tools if existing ones are insufficient
- **Tool Usage**: Selecting and using the appropriate tools with specific parameters

### 2. Construction Process

#### (1) Query Collection
- Real-world queries were collected across **22 domains** (e.g., finance, travel, alarm, document).
- Experts generated complex queries and suggested toolsets, which were then enhanced via GPT-4 through **generalization** and **complication**.
- All queries were manually reviewed for realism and utility.

#### (2) Solution Annotation (via GPT-4)
- **Plan Annotation**: Hierarchical, tree-structured plan generated in natural language.
- **Tool Creation & Plan Refinement**: GPT-4 determines if new tools are needed and creates them.
- **Tool Calling Message Annotation**: GPT-4 assigns tools and arguments to each sub-step.
- **Tool Merge**: Similar tools are merged manually to eliminate redundancy.

#### (3) Manual Refinement
- Six experts conducted **double-pass refinement** on every sample to ensure logical consistency and data quality.
- Redundant steps were removed, and tools were corrected or reassigned as necessary.

### 3. Data Statistics
- UltraTool contains **5,824 samples**, **2,032 tools**, and covers **22 domains**.
- Average of **12.27 steps** per sample and **2.74 tool calls**, with **39.61%** involving nested tool calls.
- Each tool call has an average of **3.05 arguments**.

### 4. Training
- UltraTool is **not used for training** but for evaluation only.
- It benchmarks the performance of various LLMs (GPT-3.5, GPT-4, LLaMA2, Mistral, etc.) on the same set of real-world tool-use tasks.




   
 
<br/>
# Results  




### 1. 테스트 데이터 구성
- 전체 5,824개 샘플 중 **1,000개 샘플**이 테스트용으로 사용됨.
- 평가 대상: 중국어 버전(Chinese-dataset)과 영어 버전(English-dataset) 각각 존재.
- 평가 항목은 6개(① 계획, ② 도구 생성 인식, ③ 도구 생성, ④ 도구 사용 인식, ⑤ 도구 선택, ⑥ 도구 사용)로 구성됨.

---

### 2. 평가 메트릭
UltraTool에서는 다음 3가지 메트릭을 활용함:

1. **Multi-Dimensional Point-Wise LLM-as-Judge Method**
   - GPT-4가 평가자 역할을 하며, 정답과 예측을 비교해 **6~7개의 세부 항목**(정확성, 완성도, 실행 가능성 등)에서 1~10점 척도로 점수를 부여함.
   
2. **Key-Value 기반 정확도 (Accuracy)**
   - 키(단계명)가 일치할 때만 값(결과물)의 정확도 계산.

3. **Key-Value 기반 Levenshtein 거리**
   - JSON 값 간 유사도를 거리 기반으로 평가하여 다양한 표현을 허용.

---

### 3. 벤치마크 대상 모델 (총 14개)
- **폐쇄형(Closed-source)**: GPT-4, GPT-3.5
- **오픈소스(Open-source)**:
  - 소형: LLaMA2-7B, ChatGLM3-6B, Baichuan2-7B, Vicuna-7B, Qwen-7B, Mistral-7B
  - 중형: LLaMA2-13B, Qwen-14B, Vicuna-13B, Baichuan2-13B
  - 대형: LLaMA2-70B, Qwen-72B

---

### 4. 주요 비교 결과

| 모델 | 전체 평균 점수 (중국어/영어) | 특징 |
|------|----------------------|--------|
| **GPT-4** | **76.04 / 74.58** | 전체 1위, 도구 생성/사용 능력 탁월 |
| GPT-3.5 | 59.68 / 58.90 | GPT-4보다 성능 낮지만 상위권 |
| **Qwen-72B** | 64.12 / 62.94 | 오픈소스 중 최고 성능 |
| Mistral-7B | 55.05 / 54.76 | 소형 중 효율적 성능 |
| Vicuna, Baichuan 등 | 중간 수준, 규모 커질수록 성능 향상 |

---

### 5. 주요 분석 결과
- **모델 규모가 클수록 도구 활용 능력도 증가**하는 경향이 뚜렷함.
- 오픈소스 모델은 계획 능력은 양호하나, 도구 생성/사용 단계에서는 GPT 계열에 비해 부족.
- **중국어와 영어 성능 차이** 존재: 중국어 중심 모델은 중국어에서, 영어 중심 모델은 영어에서 성능이 높음.
- **JSON 포맷 정확률과 성능 사이 양의 상관관계** 확인됨.

---



### 1. Test Data
- Among 5,824 samples, **1,000 samples** were used as the test set.
- Evaluation conducted on **Chinese-dataset** and **English-dataset**.
- Six evaluation aspects: ① Planning, ② Tool Creation Awareness, ③ Tool Creation, ④ Tool Usage Awareness, ⑤ Tool Selection, ⑥ Tool Usage.

---

### 2. Metrics Used

1. **Multi-Dimensional Point-Wise LLM-as-Judge**
   - GPT-4 acts as a judge, scoring model responses from 1 to 10 in multiple dimensions like Accuracy, Completeness, Executability, etc.

2. **Key-Value Accuracy**
   - Compares task results only when step keys match.

3. **Key-Value Levenshtein Distance**
   - Computes normalized string difference, allowing for varied expression styles in JSON outputs.

---

### 3. Baseline Models (14 total)

- **Closed-source**: GPT-4, GPT-3.5  
- **Open-source**:  
  - Small-scale: LLaMA2-7B, ChatGLM3-6B, Baichuan2-7B, Vicuna-7B, Qwen-7B, Mistral-7B  
  - Mid-scale: LLaMA2-13B, Qwen-14B, Vicuna-13B, Baichuan2-13B  
  - Large-scale: LLaMA2-70B, Qwen-72B

---

### 4. Main Results

| Model | Avg. Overall Score (CN/EN) | Notes |
|-------|-----------------------------|-------|
| **GPT-4** | **76.04 / 74.58** | Best overall, especially in tool creation and usage |
| GPT-3.5 | 59.68 / 58.90 | Lower than GPT-4 but strong |
| **Qwen-72B** | 64.12 / 62.94 | Best among open-source models |
| Mistral-7B | 55.05 / 54.76 | Strong tradeoff for small model size |
| Others (Vicuna, Baichuan) | Moderate; scale matters |

---

### 5. Key Observations
- **Model size positively correlates** with tool utilization performance.
- **Open-source models show good planning**, but struggle with tool creation and JSON formatting.
- **Language-specific models perform better** in their native languages.
- **Higher JSON format accuracy leads to higher scores** overall.




<br/>
# 예제  




---


### 1. 데이터 구성 개요
- 전체 샘플 수: **5,824개**
- 도메인 수: **22개** (금융, 비행기, 호텔, 알람 등)
- 도구 수: **2,032개**
- 각 샘플 평균:  
  - **12.27단계의 계획**
  - **2.74번 도구 호출**
  - **3.05개의 인자(argument)** 포함
- **중첩 호출(nested tool calling)** 비율: **39.61%**

---

### 2. 학습/테스트 데이터 분리
- **Test set**: 1,000개 샘플 (모델 평가에만 사용)
- **Dev set**: 4,824개 샘플 (개발 및 분석용, 학습에는 사용되지 않음)
- 언어: **중국어 / 영어** 병렬 제공 (GPT-4와 수동 정제를 통해 번역)

---

### 3. 실제 예제 (UltraTool 샘플)

####  사용자 질의:
> 500달러와 300유로를 현재 환율로 위안화로 환전한 총 금액을 계산해주세요.

####  트리 기반 계획(Plan):
```text
1. 현재 환율 조회 (도구 없음)
2. 환전 금액 계산 (도구 없음)
3. 총합 계산 (도구 없음)

세부 단계:
1.1 USD→CNY 환율 조회 → 도구: currency_exchange_rate
1.2 EUR→CNY 환율 조회 → 도구: currency_exchange_rate
2.1 500 USD 환전 → 도구: foreign_currency_exchange
2.2 300 EUR 환전 → 도구: foreign_currency_exchange
3.1 합계 계산 → 도구: sum_amounts
3.2 결과 출력 (도구 없음)
```

####  사용된 도구 예시 (툴셋):
```json
{
  "name": "currency_exchange_rate",
  "description": "Query the current exchange rate",
  "arguments": {
    "currency_from": "USD/EUR",
    "currency_to": "CNY"
  }
}
```

---

### 4. 예제의 특징
- 각 단계별로 **도구가 필요한지** 여부와, 필요한 경우 **어떤 도구와 어떤 인자(arguments)**를 써야 하는지가 명확히 표시됨.
- **트리 구조**로 각 계획이 하위 단계로 세분화되어 있음.
- 일부 단계는 도구 없이도 수행 가능한 "tool-free step"으로 설정됨.
- **실제 도구는 실행 불가능한 skeleton 형태**이지만, 구조적으로 실현 가능한 JSON 포맷을 따름.

---



### 1. Dataset Overview
- **5,824 total samples**
- **22 domains** (finance, hotel, flight, alarm, etc.)
- **2,032 unique tools**
- Per sample (on average):  
  - **12.27 planning steps**  
  - **2.74 tool calls**  
  - **3.05 arguments per call**
- **39.61%** samples include **nested tool calls**

---

### 2. Data Splits
- **Test set**: 1,000 samples (for evaluation only)
- **Dev set**: 4,824 samples (not used for training)
- Both **Chinese and English versions** available  
  (translated and refined by GPT-4 + human experts)

---

### 3. Real Example (from UltraTool)

#### User Query:
> I need to calculate the total amount of RMB required to exchange 500 US dollars and 300 euros at the current exchange rate.

####  Tree-structured Plan:
```text
1. Obtain current exchange rate (no tool)
2. Calculate exchanged amount (no tool)
3. Sum the total amount (no tool)

Sub-steps:
1.1 Get USD→CNY rate → tool: currency_exchange_rate
1.2 Get EUR→CNY rate → tool: currency_exchange_rate
2.1 Exchange 500 USD → tool: foreign_currency_exchange
2.2 Exchange 300 EUR → tool: foreign_currency_exchange
3.1 Calculate total → tool: sum_amounts
3.2 Display result (no tool)
```

####  Tool Example:
```json
{
  "name": "currency_exchange_rate",
  "description": "Query the current exchange rate",
  "arguments": {
    "currency_from": "USD/EUR",
    "currency_to": "CNY"
  }
}
```

---

### 4. Key Characteristics
- Explicit **tool-awareness at each step**: whether a tool is needed and what arguments are required.
- **Tree-structured natural language plan** supports decomposition into substeps.
- "Tool-free" steps are allowed where LLMs can reason without tools.
- Tools are not executable but are **well-structured skeletons in JSON**, representing realistic API interfaces.




<br/>  
# 요약   



이 논문은 LLM이 실제 복잡한 시나리오에서 도구를 계획·생성·활용하는 능력을 평가하기 위해, 트리 구조의 자연어 계획과 JSON 기반 도구 호출이 포함된 UltraTool 벤치마크를 제안한다. 평가 결과 GPT-4가 모든 항목에서 가장 뛰어난 성능을 보였으며, 모델 크기와 언어별 특성이 도구 활용 능력에 영향을 미쳤다. 실제 예제에서는 ‘환전 계산’과 같은 복잡한 요청을 다단계로 나누고 각 단계에 필요한 도구와 인자를 정의해 평가한다.

---


This paper introduces UltraTool, a benchmark designed to evaluate LLMs’ ability to plan, create, and use tools in real-world scenarios using tree-structured natural language plans and JSON-based tool calls. GPT-4 outperformed all other models, and the results show that model scale and language alignment significantly affect tool utilization performance. Example tasks like “currency exchange calculation” are decomposed into multiple steps with explicitly defined tools and arguments for each stage.

--- 




<br/>  
# 기타  



---


###  주요 테이블 (Tables)
1. **Table 1**: UltraTool의 예시 출력 구성 – 질의, 계획, 도구 호출 양식 등
2. **Table 2**: 각 모델별 성능 요약 (전체 점수 및 각 평가 항목별 점수 포함)
3. **Table 3**: 모델별 JSON 형식 정확도(Structure Accuracy) 및 도구 호출 성공률
4. **Table 4**: 오픈소스 모델들의 성능 비교 (모델 크기별, 언어별 등)

---

###  주요 그림 (Figures)
1. **Figure 1**: UltraTool의 전체 파이프라인 – 질의 수집 → 계획 작성 → 도구 생성 및 호출 → 출력
2. **Figure 2**: 트리 구조 기반의 계획 예시
3. **Figure 3**: 모델별 점수 분포 히트맵
4. **Figure 4**: 언어별 및 모델 크기별 성능 비교 그래프
5. **Figure 5**: JSON 출력 형식 정확도와 모델 점수 간의 상관관계 그래프

---

### 부록 (Appendix) 내용
- **A. 도메인 목록**: 총 22개의 도메인 나열 (예: 금융, 교통, 건강 등)
- **B. 도구 통합 방식**: 비슷한 도구들을 병합하는 기준과 예시
- **C. 추가 예제**: 더 많은 사용자 질의, 계획 트리, 도구 호출 예시 포함
- **D. 평가 항목 설명**: 각 항목별 GPT-4가 부여하는 기준 상세 설명
- **E. 포맷 기준**: JSON 구조의 명확한 작성 방식

---



###  Key Tables
1. **Table 1**: Example output format of UltraTool – including query, plan, tool calls.
2. **Table 2**: Evaluation results of models across six dimensions.
3. **Table 3**: Structure accuracy of JSON and tool-call success rates.
4. **Table 4**: Open-source model performance comparison by size and language.

---

###  Key Figures
1. **Figure 1**: Overall pipeline of UltraTool – from query collection to final tool-based outputs.
2. **Figure 2**: Example of tree-structured planning.
3. **Figure 3**: Heatmap of model performance across tasks.
4. **Figure 4**: Comparison of performance by language and model size.
5. **Figure 5**: Correlation plot between JSON structure accuracy and model score.

---

###  Appendix Contents
- **A. Domain List**: 22 application domains (e.g., finance, travel, health).
- **B. Tool Merging Strategy**: How similar tools are consolidated with examples.
- **C. Additional Examples**: Extended examples of queries, plans, and tool calls.
- **D. Evaluation Criteria**: Explanation of GPT-4 scoring dimensions.
- **E. Format Specification**: JSON output formatting rules.

---



<br/>
# refer format:     



@inproceedings{ying2024ultratool,
  title     = {Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios},
  author    = {Ying, Yuhang and Zhu, Lingxiao and Sun, Kai and Li, Xiaoyang and Gong, Ming and Tang, Jianshu and Feng, Zhenzhong},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  year      = {2024},
  url       = {https://aclanthology.org/2024.findings-acl.259}
}



Ying, Yuhang, Lingxiao Zhu, Kai Sun, Xiaoyang Li, Ming Gong, Jianshu Tang, and Zhenzhong Feng.
2024. "Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios." Findings of the Association for Computational Linguistics: ACL 2024. https://aclanthology.org/2024.findings-acl.259.   








