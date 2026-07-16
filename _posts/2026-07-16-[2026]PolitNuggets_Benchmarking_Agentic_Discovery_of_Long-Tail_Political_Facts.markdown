---
layout: post
title:  "[2026]PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts"
date:   2026-07-16 21:34:09 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 PolitNuggets라는 다국어 벤치마크를 통해 정치적 전기 작성을 위한 에이전틱 정보 합성 능력을 평가하였다.


짧은 요약(Abstract) :


이 논문의 초록에서는 대규모 추론 모델(Large Reasoning Models, LRM)이 정보 검색 방식을 정적이고 긴 맥락의 질문 응답에서 개방형 탐색으로 변화시켰다고 설명합니다. 그러나 실제 사용에서는 모델이 분산된 출처에서 "롱테일(long-tail)" 사실을 발견하고 종합하는 능력이 필요하지만, 이 능력은 아직 충분히 평가되지 않았습니다. 이를 해결하기 위해 저자들은 400명의 글로벌 엘리트에 대한 정치적 전기를 구성하여 10,000개 이상의 정치적 사실을 포함하는 다국어 벤치마크인 PolitNuggets를 소개합니다. 평가를 표준화하기 위해 최적화된 Supervisor–Searcher 다중 에이전트 시스템을 사용하고, 발견, 세부 정확성 및 효율성을 점수화하는 증거 조건 프로토콜인 FactNet을 제안합니다. 모델과 설정에 따라 현재 시스템은 세부 사항에 대한 정확성이 부족하고 효율성에서 상당한 변동성을 보입니다. 마지막으로, 벤치마크 진단을 통해 에이전트 성능과 기본 모델 능력 간의 관계를 설명하며, 짧은 맥락 추출, 다국어 강건성 및 신뢰할 수 있는 도구 사용의 중요성을 강조합니다.



The abstract of this paper describes how Large Reasoning Models (LRMs) have transformed information retrieval from static, long-context question answering into open-ended exploration. However, real-world applications require models to discover and synthesize "long-tail" facts from dispersed sources, a capability that remains under-evaluated. To address this, the authors introduce PolitNuggets, a multilingual benchmark for constructing political biographies of 400 global elites, covering over 10,000 political facts. They standardize evaluation using an optimized Supervisor–Searcher multi-agent system and propose FactNet, an evidence-conditional protocol that scores discovery, fine-grained accuracy, and efficiency. Across models and settings, current systems often struggle with fine-grained details and exhibit substantial variability in efficiency. Finally, using benchmark diagnostics, they relate agent performance to underlying model capabilities, highlighting the importance of short-context extraction, multilingual robustness, and reliable tool use.


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



이 논문에서 제안하는 PolitNuggets 벤치마크는 정치적 인물의 전기를 구성하는 작업을 통해 정보 합성과 추론 능력을 평가하는 시스템입니다. 이 시스템은 Supervisor–Searcher 아키텍처를 기반으로 하며, 두 개의 주요 구성 요소로 나뉩니다: Supervisor와 Searcher입니다.

1. **Supervisor**: Supervisor는 전체 연구 상태를 관리하며, 검색 요약과 할 일 목록을 유지합니다. 이 구성 요소는 전반적인 목표를 설정하고, 검색 결과를 바탕으로 다음 검색 작업을 지시합니다. Supervisor는 수집된 정보를 통합하여 글로벌 요약을 업데이트하고, 남아 있는 정보의 격차를 반영하여 할 일 목록을 조정합니다.

2. **Searcher**: Searcher는 웹에서 정보를 검색하고, 관련 문서를 검색하여 필요한 정보를 수집합니다. 이 구성 요소는 Supervisor의 지시에 따라 검색 쿼리를 실행하고, 검색된 문서에서 유용한 정보를 추출하여 아카이브에 저장합니다. Searcher는 여러 번의 검색을 통해 점진적으로 정보를 수집하며, 수집된 정보를 바탕으로 Supervisor에게 결과를 전달합니다.

이 시스템은 두 가지 주요 평가 조건을 사용하여 모델의 성능을 측정합니다: 
- **Event-Level F1**: 특정 사건의 존재를 올바르게 식별하는 능력을 평가합니다. 
- **Attribute-Level F1**: 사건에 대한 세부 정보를 정확하게 추출하는 능력을 평가합니다.

또한, 이 시스템은 다국어 증거를 처리할 수 있는 능력을 요구하며, 다양한 언어의 문서를 탐색하고 이해하는 데 필요한 멀티링구얼 강건성을 강조합니다. 이를 통해, 모델이 비영어권 정치 인물에 대한 정보를 수집할 때 발생하는 국제적 증거 격차를 해결할 수 있도록 합니다.

마지막으로, 이 연구는 모델의 성능을 평가하기 위해 FactNet이라는 동적 평가 프로토콜을 도입하여, 발견된 정보의 정확성과 효율성을 측정합니다. 이 프로토콜은 모델이 수집한 정보를 기반으로 새로운 사실을 검증하고, 이를 통해 모델의 전반적인 성능을 평가합니다.




The PolitNuggets benchmark proposed in this paper evaluates the information synthesis and reasoning capabilities of models through the task of constructing political biographies. This system is based on a Supervisor–Searcher architecture, which is divided into two main components: the Supervisor and the Searcher.

1. **Supervisor**: The Supervisor manages the overall research state, maintaining a running search summary and a to-do list. This component sets the overarching goals and directs the next search tasks based on the results obtained. The Supervisor updates the global summary by integrating the collected information and adjusts the to-do list to reflect any remaining gaps in information.

2. **Searcher**: The Searcher is responsible for searching the web for information, retrieving relevant documents, and collecting the necessary data. This component executes search queries as instructed by the Supervisor and extracts useful information from the retrieved documents to store in an archive. The Searcher incrementally gathers information through multiple searches and communicates the results back to the Supervisor.

The system measures model performance using two main evaluation conditions:
- **Event-Level F1**: Assesses the ability to correctly identify the existence of specific events.
- **Attribute-Level F1**: Evaluates the ability to accurately extract detailed information about those events.

Additionally, this system requires the ability to handle multilingual evidence, emphasizing the multilingual robustness necessary for exploring and understanding documents in various languages. This is crucial for addressing the International Evidence Gap that arises when models collect information about non-English political figures.

Finally, the study introduces a dynamic evaluation protocol called FactNet to assess model performance, measuring the accuracy and efficiency of the information collected. This protocol validates new facts based on the evidence gathered by the model, allowing for a comprehensive evaluation of the model's overall performance.


<br/>
# Results



이 논문에서는 PolitNuggets라는 새로운 벤치마크를 소개하며, 이는 정치적 인물의 전기 작성을 위한 에이전틱 정보 합성 능력을 평가하는 데 중점을 두고 있습니다. 이 벤치마크는 400명의 글로벌 엘리트에 대한 10,000개 이상의 정치적 사실을 포함하고 있으며, 다국어로 구성되어 있습니다. 평가 시스템은 Supervisor-Searcher 다중 에이전트 시스템을 최적화하여 구성되었으며, FactNet이라는 증거 조건 프로토콜을 통해 발견, 세부 정확성 및 효율성을 점수화합니다.

#### 경쟁 모델
논문에서는 Grok-4-Fast, Gemini-2.5-Flash, Qwen-3(80B 및 225B) 모델을 평가했습니다. 이 모델들은 각각 에이전틱 정보 합성의 최전선에서 성능을 발휘하는 것으로 선택되었습니다. Grok-4-Fast는 두 가지 평가 수준(Event-Level 및 Attribute-Level) 모두에서 가장 높은 성능을 보였으며, 특히 비영어권 정치인에 대한 성능 저하가 두드러졌습니다.

#### 테스트 데이터
테스트 데이터는 WhoGov 데이터셋에서 추출된 400명의 정치적 인물로 구성되어 있으며, 미국과 비미국 정치인 각각 200명씩 무작위로 샘플링되었습니다. 이 데이터는 정치적 경력, 교육 및 가족 관계에 대한 정보를 포함하고 있습니다.

#### 메트릭
평가는 두 가지 주요 메트릭으로 나뉘어 있습니다:
1. **Event-Level F1 (발견)**: 에이전트가 전기적 사건의 존재를 올바르게 식별했는지를 측정합니다.
2. **Attribute-Level F1 (세부 사항)**: 사건에 대한 세부 속성을 올바르게 채웠는지를 측정합니다.

이 두 메트릭은 각각의 모델이 얼마나 잘 정보를 발견하고 세부 사항을 정확하게 추출하는지를 평가합니다.

#### 비교
결과적으로 Grok-4-Fast는 Event-Level F1에서 0.768, Attribute-Level F1에서 0.501을 기록하며 가장 높은 성능을 보였습니다. 반면 Gemini 모델은 Event-Level F1에서 0.638, Attribute-Level F1에서 0.407로 뒤처졌습니다. Qwen 모델은 두 메트릭 모두에서 상대적으로 낮은 성능을 보였습니다. 특히 비미국 정치인에 대한 성능 저하는 최대 40%까지 감소하는 경향을 보였습니다.

이 연구는 에이전틱 정보 합성의 성능을 평가하는 데 있어 다국어 및 비미국 정치인에 대한 정보의 부족이 중요한 문제임을 강조하고 있습니다. 또한, 짧은 맥락에서의 정보 추출 능력이 에이전트의 전반적인 성능에 큰 영향을 미친다는 점을 발견했습니다.

---




This paper introduces a new benchmark called PolitNuggets, which focuses on evaluating the agentic information synthesis capabilities for constructing political biographies. This benchmark includes over 10,000 political facts about 400 global elites and is multilingual. The evaluation system is optimized using a Supervisor-Searcher multi-agent system and employs FactNet, an evidence-conditional protocol that scores discovery, fine-grained accuracy, and efficiency.

#### Competing Models
The paper evaluates three models: Grok-4-Fast, Gemini-2.5-Flash, and Qwen-3 (80B and 225B). These models were selected for their performance at the forefront of agentic information synthesis. Grok-4-Fast demonstrated the highest performance across both evaluation levels (Event-Level and Attribute-Level), with a notable degradation in performance for non-US entities.

#### Test Data
The test data consists of 400 political figures sampled from the WhoGov dataset, with 200 US and 200 non-US politicians randomly selected. This data includes information on political careers, education, and family relations.

#### Metrics
The evaluation is divided into two main metrics:
1. **Event-Level F1 (Discovery)**: Measures whether the agent correctly identifies the existence of a biographical event.
2. **Attribute-Level F1 (Granularity)**: Measures whether the agent can accurately fill fine-grained attributes for an event.

These two metrics assess how well each model discovers information and accurately extracts details.

#### Comparison
Ultimately, Grok-4-Fast achieved an Event-Level F1 of 0.768 and an Attribute-Level F1 of 0.501, marking the highest performance. In contrast, the Gemini model scored 0.638 for Event-Level F1 and 0.407 for Attribute-Level F1, trailing behind. The Qwen models exhibited relatively lower performance across both metrics. Notably, the performance for non-US politicians showed a decline of up to 40%.

This study highlights the significant issue of the lack of information regarding multilingual and non-US politicians in evaluating agentic information synthesis performance. Additionally, it found that the ability to extract information from short contexts greatly impacts the overall performance of the agents.


<br/>
# 예제



**예시: PolitNuggets 벤치마크의 트레이닝 데이터와 테스트 데이터**

1. **트레이닝 데이터**
   - **입력**: 정치인에 대한 기본 정보 (이름, 생년월일, 직업 등)와 관련된 웹 페이지 링크들.
   - **출력**: 정치인의 경력, 교육 이력, 가족 관계 등 상세한 생애 정보가 포함된 JSON 형식의 데이터.
   - **예시**:
     - 입력: 
       ```json
       {
         "name": "Erik Solheim",
         "birth_date": "1955-01-18",
         "occupation": "Politician",
         "sources": [
           "https://no.wikipedia.org/wiki/Erik_Solheim",
           "https://regjeringen.no/en/dep/mdg/"
         ]
       }
       ```
     - 출력:
       ```json
       {
         "full_name": "Erik Solheim",
         "birth_date": "1955-01-18",
         "education_experiences": [
           {
             "organization_name": "Oslo katedralskole",
             "education_level": "High school"
           },
           {
             "organization_name": "Universitetet i Oslo",
             "education_level": "Master"
           }
         ],
         "occupation_experiences": [
           {
             "time_range": "2007-2012",
             "position_title": "Minister of the Environment"
           }
         ],
         "relatives": [
           {
             "relation": "spouse",
             "name": "Gry Høeg Ulverud"
           }
         ]
       }
       ```

2. **테스트 데이터**
   - **입력**: 정치인에 대한 정보가 포함된 웹 페이지 링크들, 그러나 이 정보는 트레이닝 데이터와는 다를 수 있음.
   - **출력**: 모델이 수집한 정보의 정확성을 평가하기 위한 F1 점수와 같은 메트릭.
   - **예시**:
     - 입력:
       ```json
       {
         "name": "Erik Solheim",
         "sources": [
           "https://example.com/erik_solheim_bio",
           "https://anotherexample.com/erik_solheim"
         ]
       }
       ```
     - 출력:
       ```json
       {
         "Event-Level F1": 0.75,
         "Attribute-Level F1": 0.60,
         "Efficiency": {
           "search_steps": 12,
           "tokens_used": 1500
         }
       }
       ```

3. **구체적인 테스크**
   - **테스크**: 주어진 정치인에 대한 생애 정보를 수집하고, 이를 바탕으로 정확한 경력 이력을 작성하는 것.
   - **과정**:
     1. Supervisor가 Search Agent에게 특정 정치인에 대한 정보를 수집하라는 지시를 내림.
     2. Search Agent는 웹에서 관련 정보를 검색하고, 이를 수집하여 Archive에 저장.
     3. 수집된 정보는 Coder에 의해 JSON 형식으로 정리됨.
     4. 최종적으로 Supervisor는 수집된 정보를 바탕으로 전체 생애 이력을 작성하고, 이를 평가함.




**Example: Training and Testing Data for the PolitNuggets Benchmark**

1. **Training Data**
   - **Input**: Basic information about a politician (name, date of birth, occupation, etc.) and related web page links.
   - **Output**: Detailed life information including career, education history, family relations, formatted in JSON.
   - **Example**:
     - Input: 
       ```json
       {
         "name": "Erik Solheim",
         "birth_date": "1955-01-18",
         "occupation": "Politician",
         "sources": [
           "https://no.wikipedia.org/wiki/Erik_Solheim",
           "https://regjeringen.no/en/dep/mdg/"
         ]
       }
       ```
     - Output:
       ```json
       {
         "full_name": "Erik Solheim",
         "birth_date": "1955-01-18",
         "education_experiences": [
           {
             "organization_name": "Oslo katedralskole",
             "education_level": "High school"
           },
           {
             "organization_name": "Universitetet i Oslo",
             "education_level": "Master"
           }
         ],
         "occupation_experiences": [
           {
             "time_range": "2007-2012",
             "position_title": "Minister of the Environment"
           }
         ],
         "relatives": [
           {
             "relation": "spouse",
             "name": "Gry Høeg Ulverud"
           }
         ]
       }
       ```

2. **Testing Data**
   - **Input**: Web page links containing information about a politician, which may differ from the training data.
   - **Output**: Metrics such as F1 scores to evaluate the accuracy of the information collected by the model.
   - **Example**:
     - Input:
       ```json
       {
         "name": "Erik Solheim",
         "sources": [
           "https://example.com/erik_solheim_bio",
           "https://anotherexample.com/erik_solheim"
         ]
       }
       ```
     - Output:
       ```json
       {
         "Event-Level F1": 0.75,
         "Attribute-Level F1": 0.60,
         "Efficiency": {
           "search_steps": 12,
           "tokens_used": 1500
         }
       }
       ```

3. **Specific Task**
   - **Task**: To collect life information about a given politician and create an accurate career history based on that information.
   - **Process**:
     1. The Supervisor instructs the Search Agent to gather information about a specific politician.
     2. The Search Agent searches the web for relevant information and archives it.
     3. The collected information is organized into JSON format by the Coder.
     4. Finally, the Supervisor compiles the information into a complete life history and evaluates it.

<br/>
# 요약


이 논문에서는 PolitNuggets라는 다국어 벤치마크를 통해 정치적 전기 작성을 위한 에이전틱 정보 합성 능력을 평가하였다. 실험 결과, 현재 시스템은 세부 사항에 대한 재현율이 낮고 비효율적이며, 특히 비미국 정치인에 대한 성능 저하가 두드러졌다. 최종적으로, 짧은 맥락 추출, 다국어 강건성, 신뢰할 수 있는 도구 사용이 에이전트 성능에 중요한 영향을 미친다는 것을 발견하였다.

---

This paper evaluates the agentic information synthesis capabilities for constructing political biographies using a multilingual benchmark called PolitNuggets. The results show that current systems struggle with low recall and efficiency, particularly for non-US politicians. Ultimately, it was found that short-context extraction, multilingual robustness, and reliable tool use significantly impact agent performance.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 에이전트 성능 히트맵은 "헤드"와 "롱테일" 합성의 격차를 보여줍니다. 이는 에이전트가 주요 사실을 잘 찾지만, 덜 알려진 세부사항을 놓치는 경향이 있음을 나타냅니다.
   - **Figure 2**: 각국에서 검색된 증거의 언어 구성 비율을 보여줍니다. 영어와 비영어의 비율이 나타나며, 비영어 자료의 부족이 비미국 정치인에 대한 성능 저하에 기여하고 있음을 시사합니다.
   - **Figure 4**: 효율성 분석을 통해 검색 단계와 F1 점수 간의 관계를 시각화합니다. Grok-4-Fast 모델이 높은 F1 점수를 유지하면서도 적은 검색 단계를 소모하는 효율적인 경향을 보입니다.

2. **테이블**
   - **Table 1**: 다양한 모델의 성능을 Event-Level F1과 Attribute-Level F1으로 비교합니다. Grok-4-Fast 모델이 모든 조건에서 가장 높은 성능을 보이며, 특히 비미국 정치인에 대한 성능이 저조한 것을 확인할 수 있습니다.
   - **Table 4**: 정밀도, 재현율, F1 점수를 Event-Level과 Attribute-Level에서 비교합니다. 전반적으로 정밀도는 높지만, 재현율이 낮아 정보의 누락이 발생하고 있음을 보여줍니다.
   - **Table 2**: 비미국 정치인에 대한 성능 저하를 통계적으로 검증합니다. 비미국 정치인에 대한 성능이 미국 정치인에 비해 유의미하게 낮다는 것을 보여줍니다.

3. **어펜딕스**
   - **A.1.1 Deliverables**: 연구의 재현성을 위해 제공되는 자료 목록을 포함합니다. 이는 연구 결과의 신뢰성을 높이는 데 기여합니다.
   - **A.3.1 LRM baseline results**: LRM 모델의 성능을 정리하여, 에이전트 시스템과의 비교를 통해 에이전트의 장점과 한계를 명확히 합니다.

### Insights
- **에이전트 성능**: 에이전트는 주요 사실을 잘 찾지만, 세부사항에서 성능이 저조하여 "롱테일" 사실을 놓치는 경향이 있습니다. 이는 정보 검색의 효율성을 높이기 위한 전략적 접근이 필요함을 시사합니다.
- **국제적 증거 격차**: 비미국 정치인에 대한 성능 저하가 관찰되었으며, 이는 비영어 자료의 부족과 관련이 있습니다. 이는 다국적 정보 검색 시스템의 개선이 필요함을 강조합니다.
- **효율성**: Grok-4-Fast 모델이 높은 성능을 유지하면서도 적은 검색 단계를 소모하는 경향을 보였으며, 이는 에이전트 시스템의 설계에서 중요한 요소로 작용합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: The agent performance heatmap illustrates the gap between "head" and "long-tail" synthesis. It indicates that agents excel at finding major facts but tend to miss lesser-known details.
   - **Figure 2**: This figure shows the language composition of retrieved evidence across countries. The ratio of English to non-English materials suggests that the lack of non-English resources contributes to the performance degradation for non-US politicians.
   - **Figure 4**: The efficiency analysis visualizes the relationship between search steps and F1 scores. The Grok-4-Fast model demonstrates an efficient trend, maintaining high F1 scores while consuming fewer search steps.

2. **Tables**
   - **Table 1**: Compares the performance of various models using Event-Level F1 and Attribute-Level F1. The Grok-4-Fast model consistently shows the highest performance across all conditions, particularly highlighting the lower performance for non-US politicians.
   - **Table 4**: Compares precision, recall, and F1 scores at both Event-Level and Attribute-Level. Overall, precision is high, but low recall indicates missing information.
   - **Table 2**: Statistically validates the performance degradation for non-US politicians, showing a significant drop compared to US politicians.

3. **Appendix**
   - **A.1.1 Deliverables**: Lists the materials provided to support reproducibility of the research, enhancing the reliability of the findings.
   - **A.3.1 LRM baseline results**: Summarizes the performance of LRM models, clarifying the advantages and limitations of the agent system through comparison.

### Insights
- **Agent Performance**: Agents perform well in finding major facts but struggle with details, indicating a need for strategic approaches to improve information retrieval efficiency.
- **International Evidence Gap**: The observed performance degradation for non-US politicians is linked to the lack of non-English resources, emphasizing the need for improvements in multinational information retrieval systems.
- **Efficiency**: The Grok-4-Fast model's ability to maintain high performance while consuming fewer search steps highlights a critical factor in the design of agent systems.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Zhu2026,
  author    = {Yifei Zhu},
  title     = {PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {45012--45035},
  year      = {2026},
  month     = {July 2-7},
  publisher = {Association for Computational Linguistics},
}
```

### 시카고 스타일

Yifei Zhu. "PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 45012–45035. Association for Computational Linguistics, July 2-7, 2026.   