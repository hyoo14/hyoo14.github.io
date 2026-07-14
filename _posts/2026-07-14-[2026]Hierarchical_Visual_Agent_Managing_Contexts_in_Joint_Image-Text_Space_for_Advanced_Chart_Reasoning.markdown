---
layout: post
title:  "[2026]Hierarchical Visual Agent: Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning"
date:   2026-07-14 00:44:39 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 HIERVA라는 계층적 비주얼 에이전트를 제안하여 복잡한 차트 질문 응답을 위한 멀티모달 컨텍스트 관리를 수행합니다.


짧은 요약(Abstract) :



고급 차트 질문 응답은 작은 시각적 요소에 대한 정확한 인식과 여러 서브플롯에 걸친 다단계 추론을 요구합니다. 기존의 다중 모달 대형 언어 모델(MLLM)은 단일 플롯을 이해하는 데 강하지만, 여러 서브플롯에 걸친 다단계 추론에서는 종종 어려움을 겪습니다. 우리는 HIERVA라는 계층적 시각적 에이전트 프레임워크를 제안하여 차트 추론을 위해 이미지-텍스트 공간에서 작업 맥락을 반복적으로 구성하고 업데이트합니다. 고수준 관리자는 계획을 생성하고 핵심 정보만 포함된 간결한 맥락을 유지하며, 전문화된 작업자는 추론을 수행하고 증거를 수집하여 결과를 반환합니다. 특히, 이 에이전트는 별도의 시각적 및 텍스트 맥락을 유지하며, 시각적 맥락을 제한하기 위해 확대 도구를 사용합니다. CHARXIV 추론 하위 집합에 대한 실험 결과는 강력한 다중 모달 기준선에 비해 일관된 개선을 보여주며, 계층적 아키텍처, 범위가 제한된 시각적 맥락, 그리고 정제된 맥락이 상호 보완적인 이점을 제공함을 검증합니다.




Advanced chart question answering requires precise perception of small visual elements and multi-step reasoning across several subplots. While existing multimodal large language models (MLLMs) are strong at understanding single plots, they often struggle with multi-step reasoning across multiple subplots. We propose HIERVA, a hierarchical visual agent framework for chart reasoning that iteratively constructs and updates a working context in a joint image-text space. A high-level manager generates plans and maintains a compact context containing only key information, while specialized workers perform reasoning, gather evidence, and return results. In particular, the agent maintains separate visual and textual contexts, using a zoom-in tool to restrict the visual context. Experiments on the CHARXIV reasoning subset demonstrate consistent improvements over strong multimodal baselines, and ablation studies verify that hierarchical architecture, scoped visual context, and distilled context contribute complementary gains.


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



**모델 및 아키텍처: HIERVA (Hierarchical Visual Agent)**

HIERVA는 고급 차트 질문 응답을 위한 계층적 비주얼 에이전트 프레임워크로, 이미지-텍스트 공간에서의 맥락 관리를 통해 복잡한 차트 추론을 수행합니다. 이 모델은 두 가지 주요 구성 요소로 나뉩니다: **매니저**와 **워커**. 매니저는 고수준의 계획을 수립하고, 필요한 정보를 포함하는 간결한 맥락을 유지합니다. 반면, 워커는 특정 작업을 수행하고 증거를 수집하여 결과를 반환합니다.

1. **매니저**: 매니저는 전체 작업의 계획을 수립하고, 각 워커에게 필요한 정보를 전달합니다. 매니저는 원래 질문과 차트 참조, 정제된 계획, 완료된 하위 작업의 요약을 포함하는 글로벌 작업 맥락을 유지합니다. 매니저는 각 단계의 필요에 따라 텍스트 및 비주얼 맥락을 제한하는 '줌 인' 도구를 사용하여 작업을 수행합니다.

2. **워커**: 각 워커는 고유한 로컬 맥락을 유지하며, 주어진 작업을 수행합니다. 워커는 매니저가 지정한 이미지 입력(원본 차트 또는 이전에 생성된 크롭)과 최소한의 배경 정보를 포함하여 작업을 수행합니다. 각 워커는 특정 작업에 필요한 정보만을 처리하여 효율성을 높이고 주의 산만을 줄입니다.

3. **계층적 구조**: HIERVA의 계층적 구조는 매니저와 워커 간의 상호작용을 통해 복잡한 작업을 분해하고, 각 워커가 독립적으로 작업을 수행할 수 있도록 합니다. 이 구조는 긴 추론 체인을 관리하는 데 필수적이며, 각 워커는 자신에게 할당된 작업에만 집중할 수 있습니다.

4. **맥락 관리**: HIERVA는 맥락을 관리하는 데 있어 세 가지 주요 원칙을 따릅니다:
   - **계층적 위임**: 매니저는 작업을 워커에게 위임하여 각 워커가 독립적으로 작업을 수행하도록 합니다.
   - **맥락 증류**: 매니저는 중간 결과를 정제하여 불필요한 정보를 제거하고, 각 단계에서 필요한 정보만을 유지합니다.
   - **스코프된 맥락**: 각 워커는 자신에게 필요한 정보만을 포함하는 로컬 맥락에서 작업을 수행하여, 전체 맥락의 복잡성을 줄입니다.

5. **기술적 세부사항**: HIERVA는 훈련이 필요 없는 프레임워크로, 매니저는 두 단계의 계획을 수립하고, 각 워커는 주어진 작업을 수행한 후 결과를 매니저에게 반환합니다. 이 과정에서 매니저는 각 워커의 작업 이력을 포함하지 않으며, 오직 최종 결과만을 유지합니다.

이러한 구조와 원칙을 통해 HIERVA는 복잡한 차트 질문 응답 작업에서 높은 정확도를 달성하며, 특히 다단계 비교, 집계 및 계산이 필요한 질문에서 두드러진 성능 향상을 보입니다.

---




**Model and Architecture: HIERVA (Hierarchical Visual Agent)**

HIERVA is a hierarchical visual agent framework designed for advanced chart question answering, managing context in a joint image-text space to perform complex chart reasoning. The model is divided into two main components: the **manager** and the **workers**. The manager generates high-level plans and maintains a compact context containing only essential information, while specialized workers perform specific tasks, gather evidence, and return results.

1. **Manager**: The manager establishes the overall plan for the task and communicates the necessary information to each worker. It maintains a global working context that includes the original question and chart reference, a refined plan, and structured summaries of completed subtasks. The manager uses a 'zoom-in' tool to restrict the visual and textual contexts according to the needs of each step.

2. **Workers**: Each worker operates within its own isolated local context, performing the assigned task. Workers receive a single image input (either the original chart or a previously generated crop) along with minimal required background information. This separation enforces encapsulation, allowing each worker to focus solely on the task at hand, improving efficiency and reducing distractions.

3. **Hierarchical Structure**: The hierarchical structure of HIERVA facilitates the decomposition of complex tasks into manageable subtasks, allowing each worker to operate independently. This structure is essential for managing long-horizon reasoning chains, as each worker can concentrate on the specific task assigned to them.

4. **Context Management**: HIERVA adheres to three key principles for managing context:
   - **Hierarchical Delegation**: The manager delegates tasks to workers, enabling them to perform their tasks independently.
   - **Context Distillation**: The manager refines intermediate results, removing unnecessary information and retaining only what is needed for each step.
   - **Scoped Context**: Each worker operates within a local context that includes only the information necessary for their task, reducing the complexity of the overall context.

5. **Technical Details**: HIERVA is a training-free framework where the manager performs a two-stage planning process, and each worker executes the assigned task before returning the results to the manager. Throughout this process, the manager does not inherit the worker's deliberation traces, retaining only the final results.

Through this structure and these principles, HIERVA achieves high accuracy in complex chart question answering tasks, demonstrating significant performance improvements, especially in questions requiring multi-step comparisons, aggregations, and computations.


<br/>
# Results


이 논문에서는 HIERVA(Hierarchical Visual Agent)라는 새로운 프레임워크를 제안하여 고급 차트 질문 응답을 수행합니다. 이 프레임워크는 이미지-텍스트 공간에서의 맥락 관리를 통해 다단계 추론을 효과적으로 처리합니다. 실험은 CHARXIV reasoning subset에서 수행되었으며, 이 데이터셋은 복잡한 차트 질문을 포함하고 있습니다.

#### 경쟁 모델
HIERVA는 여러 강력한 멀티모달 모델과 비교되었습니다. 주요 경쟁 모델로는 Direct, Chain-of-Thought (CoT), CoT-Plan, 그리고 "Thinking with Images" 접근 방식이 있습니다. 각 모델은 동일한 백본 모델인 Qwen3VL-A22B를 사용하여 평가되었습니다.

#### 테스트 데이터
CHARXIV reasoning subset은 1000개의 예제를 포함하고 있으며, 이 데이터셋은 다단계 비교, 집계 및 복잡한 차트 질문을 요구합니다. 질문 유형은 읽기, 극대값 찾기, 첫 번째 찾기, 역 읽기, 비교, 빈도 수 세기 등으로 분류됩니다.

#### 메트릭
모델의 성능은 정확도(accuracy)로 측정되었습니다. 각 모델의 질문 유형별 정확도는 다음과 같습니다:
- Direct: 45.7%
- CoT: 62.1%
- CoT-Plan: 62.4%
- Thinking with Images (zoom): 58.9%
- HIERVA: 64.2%

HIERVA는 CoT-Plan보다 1.8% 높은 정확도를 기록하였고, "Thinking with Images"보다 5.3% 높은 성능을 보였습니다. 특히, HIERVA는 복잡한 추론 카테고리에서 가장 큰 성과를 보였으며, 극대값 찾기(+1.5%), 비교(+4.8%), 빈도 수 세기(+2.2%)에서 두드러진 성과를 나타냈습니다.

#### 비교
HIERVA는 다른 모델들과 비교했을 때, 다단계 질문에서의 성능이 가장 뛰어난 것으로 나타났습니다. 특히, HIERVA는 맥락을 효과적으로 관리하여 불필요한 정보를 줄이고, 필요한 정보만을 유지하는 방식으로 성능을 향상시켰습니다. 또한, HIERVA는 도구와 기술을 동적으로 관리하여 더 높은 정확도를 달성했습니다.




This paper proposes a new framework called HIERVA (Hierarchical Visual Agent) for advanced chart question answering. This framework effectively handles multi-step reasoning through context management in the image-text space. Experiments were conducted on the CHARXIV reasoning subset, which includes complex chart questions.

#### Competing Models
HIERVA was compared against several strong multimodal models. The main competing models include Direct, Chain-of-Thought (CoT), CoT-Plan, and the "Thinking with Images" approach. Each model was evaluated using the same backbone model, Qwen3VL-A22B.

#### Test Data
The CHARXIV reasoning subset contains 1,000 examples and requires multi-step comparisons, aggregations, and complex chart questions. The question types are classified into reading values, finding extremes, finding the first occurrence, reverse reading, comparing, and counting frequencies.

#### Metrics
The performance of the models was measured using accuracy. The accuracy for each model by question type is as follows:
- Direct: 45.7%
- CoT: 62.1%
- CoT-Plan: 62.4%
- Thinking with Images (zoom): 58.9%
- HIERVA: 64.2%

HIERVA achieved an accuracy that is 1.8% higher than CoT-Plan and 5.3% higher than "Thinking with Images." Notably, HIERVA showed the largest gains in complex reasoning categories, with improvements of +1.5% in finding extremes, +4.8% in comparisons, and +2.2% in frequency counting.

#### Comparison
HIERVA demonstrated superior performance compared to other models, particularly in multi-step questions. It effectively managed context by reducing unnecessary information and retaining only the necessary details, leading to improved performance. Additionally, HIERVA achieved higher accuracy by dynamically managing tools and skills.


<br/>
# 예제



이 논문에서는 고급 차트 질문 응답을 위한 HIERVA(계층적 시각 에이전트) 프레임워크를 제안합니다. 이 프레임워크는 이미지-텍스트 공간에서 작업 컨텍스트를 관리하는 데 중점을 두고 있습니다. HIERVA는 매니저와 여러 작업자로 구성되어 있으며, 매니저는 계획을 세우고 작업자에게 작업을 할당합니다. 각 작업자는 특정 작업에 필요한 최소한의 정보만을 가지고 작업을 수행합니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 차트 이미지와 자연어 질문
     - 예: 차트 이미지 (예: 막대 그래프)와 질문 "이 그래프에서 가장 높은 값의 색깔은 무엇인가요?"
   - **출력**: 질문에 대한 답변
     - 예: "파란색"

2. **테스트 데이터**:
   - **입력**: 복합 차트 이미지와 질문
     - 예: 여러 개의 서브플롯이 포함된 차트 이미지와 질문 "서브플롯 (a), (b), (c), (d) 중에서 가장 높은 중앙값을 가진 것은 무엇인가요?"
   - **출력**: 질문에 대한 답변
     - 예: "(b)"

#### 구체적인 테스크
- **테스크 1**: 단일 서브플롯에서 값 읽기
  - **입력**: 차트 이미지와 질문 "서브플롯 (a)의 중앙값은 얼마인가요?"
  - **출력**: "3.5"

- **테스크 2**: 여러 서브플롯 비교
  - **입력**: 복합 차트 이미지와 질문 "서브플롯 (a)와 (b)의 중앙값을 비교하세요."
  - **출력**: "(b)가 (a)보다 높습니다."

- **테스크 3**: 다단계 계산
  - **입력**: 차트 이미지와 질문 "서브플롯 (a)와 (b)의 차이를 계산하세요."
  - **출력**: "2.0"

이러한 방식으로 HIERVA는 각 작업을 세분화하여 매니저가 작업을 계획하고, 작업자가 필요한 정보를 수집하여 최종 결과를 도출하는 구조로 작동합니다.

---



This paper proposes HIERVA (Hierarchical Visual Agent) framework for advanced chart question answering, focusing on managing working contexts in the image-text space. HIERVA consists of a manager and multiple workers, where the manager plans and assigns tasks to the workers. Each worker operates with only the minimal information necessary for their specific task.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Input**: Chart image and natural language question
     - Example: Chart image (e.g., bar graph) and question "What is the color of the highest value in this graph?"
   - **Output**: Answer to the question
     - Example: "Blue"

2. **Test Data**:
   - **Input**: Composite chart image and question
     - Example: Chart image with multiple subplots and question "Which subplot (a), (b), (c), or (d) has the highest median value?"
   - **Output**: Answer to the question
     - Example: "(b)"

#### Specific Tasks
- **Task 1**: Reading a value from a single subplot
  - **Input**: Chart image and question "What is the median value of subplot (a)?"
  - **Output**: "3.5"

- **Task 2**: Comparing multiple subplots
  - **Input**: Composite chart image and question "Compare the median values of subplot (a) and (b)."
  - **Output**: "(b) is higher than (a)."

- **Task 3**: Multi-step computation
  - **Input**: Chart image and question "Calculate the difference between subplot (a) and (b)."
  - **Output**: "2.0"

In this way, HIERVA operates by breaking down each task, allowing the manager to plan the tasks and the workers to gather the necessary information to derive the final results.

<br/>
# 요약


이 논문에서는 HIERVA라는 계층적 비주얼 에이전트를 제안하여 복잡한 차트 질문 응답을 위한 멀티모달 컨텍스트 관리를 수행합니다. 실험 결과, HIERVA는 CHARXIV 데이터셋에서 64.2%의 정확도로 기존의 강력한 기준선보다 1.8%에서 5.3% 향상된 성능을 보였습니다. 예를 들어, HIERVA는 여러 서브플롯에서 중간값을 비교하는 질문을 효과적으로 처리하여 정확한 결과를 도출했습니다.

---

This paper proposes HIERVA, a hierarchical visual agent that manages multimodal context for complex chart question answering. Experimental results show that HIERVA achieves an accuracy of 64.2% on the CHARXIV dataset, outperforming strong baselines by 1.8% to 5.3%. For instance, HIERVA effectively handles questions comparing median values across multiple subplots to produce accurate results.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 다양한 차트 추론 패러다임을 비교합니다. HIERVA는 계층적 비주얼 에이전트를 사용하여 이미지-텍스트 작업 컨텍스트를 유지하며, 각 하위 작업이 특정 지역의 스코프에서 작동하도록 합니다. 이는 불필요한 정보의 혼합을 방지하고, 각 단계에서 필요한 정보만을 유지하여 효율성을 높입니다.
   - **Figure 2**: 다양한 방법이 고급 차트 추론 질문을 처리하는 방식을 보여줍니다. CoT는 전역 이미지를 통해 순차적으로 추론하지만 세부 사항을 잘못 판단할 수 있습니다. 반면, HIERVA는 작업을 세분화하여 각 하위 작업이 관련된 뷰에서 작동하도록 하여 정확한 결과를 도출합니다.

2. **테이블**
   - **Table 1**: 기본 차트 이해와 고급 차트 추론 작업의 차이를 보여줍니다. 기본 작업은 단일 지역을 로컬라이징하고 한 단계에서 정보를 검색하는 반면, 고급 작업은 여러 지역에서 증거를 집계하고 다단계 계산을 요구합니다.
   - **Table 2**: ChartQA 및 합성 차트에서 다양한 방법의 성능을 비교합니다. HIERVA는 복잡한 차트에서 더 나은 성능을 보이며, 특히 하위 플롯 수가 증가할수록 성능이 감소하는 경향을 보여줍니다.
   - **Table 3**: CHARXIV 추론 하위 집합에서의 성능을 보여줍니다. HIERVA는 64.2%의 정확도로 가장 높은 성능을 기록하며, 복잡한 추론 카테고리에서 가장 큰 이점을 보입니다.
   - **Table 4**: HIERVA의 주요 구성 요소에 대한 ablation 연구 결과를 보여줍니다. 계층적 위임이 가장 중요한 요소로 나타났으며, 컨텍스트 증류와 스코프가 있는 컨텍스트도 성능 향상에 기여합니다.

3. **어펜딕스**
   - 어펜딕스에서는 HIERVA의 프롬프트 및 작업 스키마에 대한 세부 정보를 제공합니다. 각 하위 작업은 명확한 구조를 가지고 있으며, 작업 생성 및 종료 과정에서 필요한 정보를 명확히 전달합니다. 이러한 구조는 HIERVA의 효율성을 높이고, 각 작업이 필요한 정보만을 처리하도록 합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: Compares various chart reasoning paradigms. HIERVA uses hierarchical visual agents to maintain a joint image-text working context, allowing each subtask to operate within a scoped region. This prevents the mixing of irrelevant information and keeps only the necessary information at each step, enhancing efficiency.
   - **Figure 2**: Illustrates how different methods handle an advanced chart reasoning question. CoT reasons sequentially over the global image but can misjudge fine details. In contrast, HIERVA decomposes the task so that each subtask operates with the relevant view, leading to accurate results.

2. **Tables**
   - **Table 1**: Shows the difference between basic chart understanding and advanced chart reasoning tasks. Basic tasks require localizing a single region and retrieving information in one step, while advanced tasks require aggregating evidence across multiple regions and multi-step computations.
   - **Table 2**: Compares the performance of various methods on ChartQA and synthetic charts. HIERVA demonstrates better performance on complex charts, particularly as the number of subplots increases, showing a trend of performance degradation.
   - **Table 3**: Displays performance on the CHARXIV reasoning subset. HIERVA achieves the highest accuracy of 64.2%, with the largest gains observed in complex reasoning categories.
   - **Table 4**: Presents the results of an ablation study on the key components of HIERVA. Hierarchical delegation is identified as the most critical factor, with context distillation and scoped context also contributing to performance improvements.

3. **Appendix**
   - The appendix provides detailed information on the prompts and task schema used in HIERVA. Each subtask is clearly structured, ensuring that necessary information is communicated during task creation and termination. This structure enhances the efficiency of HIERVA and ensures that each task processes only the required information.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Dong2026,
  author    = {Qihua Dong and Ruozhen He and Junwen Chen and Yizhou Wang and Xu Ma and Songyao Jiang and Yun Fu},
  title     = {Hierarchical Visual Agent: Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {38390--38401},
  year      = {2026},
  month     = {July 2-7},
  publisher  = {Association for Computational Linguistics},
}
```

### Chicago Style Citation

Dong, Qihua, Ruozhen He, Junwen Chen, Yizhou Wang, Xu Ma, Songyao Jiang, and Yun Fu. "Hierarchical Visual Agent: Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning." In *Findings of the Association for Computational Linguistics: ACL 2026*, 38390–38401. July 2-7, 2026. Association for Computational Linguistics.
    