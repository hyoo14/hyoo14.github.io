---
layout: post
title:  "[2026]CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems"
date:   2026-06-21 08:45:47 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 고대 중국 문화 시스템의 사건 분류 체계를 구축하기 위해 CHisAgent라는 다중 에이전트 프레임워크를 제안합니다.


짧은 요약(Abstract) :


이 논문에서는 고대 중국 문화 시스템에서 사건 분류 체계를 구축하기 위한 다중 에이전트 프레임워크인 CHisAgent를 제안합니다. 대형 언어 모델(LLM)은 많은 작업에서 뛰어난 성능을 보이지만, 역사적 및 문화적 추론, 특히 비영어권 맥락에서는 제한된 능력을 보입니다. 수작업으로 분류 체계를 구축하는 것은 비용이 많이 들고 확장하기 어렵기 때문에, CHisAgent는 세 가지 역할로 분류 체계 구축을 분해합니다: 원시 역사 자료에서 초기 계층 구조를 유도하는 'Inducer', LLM의 세계 지식을 사용하여 누락된 중간 개념을 도입하는 'Expander', 외부 구조화된 역사 자원을 통합하여 신뢰성을 보장하는 'Enricher'입니다. 이 프레임워크를 통해 고대 중국의 정치, 군사, 외교 및 사회 생활을 포괄하는 대규모 사건 분류 체계를 구축하였으며, 평가 결과는 구조적 일관성과 범위가 개선되었음을 보여줍니다.



This paper proposes CHisAgent, a multi-agent framework for constructing an event taxonomy in ancient Chinese cultural systems. Large language models (LLMs) demonstrate strong performance on many tasks but show limited ability in historical and cultural reasoning, particularly in non-English contexts. Manual taxonomy construction is costly and difficult to scale, which is why CHisAgent decomposes taxonomy construction into three roles: an 'Inducer' that derives an initial hierarchy from raw historical corpora, an 'Expander' that introduces missing intermediate concepts using LLM world knowledge, and an 'Enricher' that integrates external structured historical resources to ensure faithfulness. Through this framework, a large-scale event taxonomy covering politics, military, diplomacy, and social life in ancient China has been constructed, and evaluation results demonstrate improved structural coherence and coverage.


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



이 논문에서 제안하는 CHisAgent는 고대 중국 문화 시스템의 사건 분류 체계를 구축하기 위한 다중 에이전트 프레임워크입니다. 이 프레임워크는 세 가지 주요 단계로 구성되어 있으며, 각 단계는 특정 역할을 수행하는 LLM(대형 언어 모델) 에이전트로 구성됩니다. 이 방법론은 다음과 같은 세 가지 주요 구성 요소로 나뉩니다:

1. **Inducer (유도기)**: 이 단계는 원시 역사 자료에서 초기 사건 계층 구조를 유도하는 역할을 합니다. Inducer는 고대 중국의 역사적 문서에서 사건 인스턴스와 사건 유형을 추출하여 초기 사건 세트를 생성합니다. 이 과정에서 여러 추출기(Extractor)를 사용하여 다양한 사건 유형을 포괄적으로 수집하고, 각 사건 유형을 사전 정의된 도메인에 할당합니다.

2. **Expander (확장기)**: Expander는 Inducer가 생성한 초기 계층 구조의 불완전성을 해결하기 위해 상위에서 하위로 구조를 확장하는 역할을 합니다. 이 단계에서는 현재 계층의 노드와 정의를 바탕으로 누락된 중간 개념을 식별하고, 필요에 따라 새로운 하위 노드를 생성하여 계층의 일관성을 높입니다.

3. **Enricher (풍부화기)**: Enricher는 외부의 구조화된 역사 자원과 원시 자료에서 유도된 증거를 통합하여 생성된 분류 체계의 충실성을 보장합니다. 이 단계에서는 고빈도 사건, 주제 기반 사건, 그리고 기존의 역사 지식 기반에서 수집된 관계를 통해 후보 사건 세트를 구성하고, 이를 기존의 분류 체계에 통합하여 최종 분류 체계를 완성합니다.

이러한 세 가지 단계는 각각의 역할에 따라 전문화된 LLM 에이전트가 협력하여 역사적 사건 분류 체계를 효과적으로 구축할 수 있도록 설계되었습니다. 이 방법론은 고대 중국의 역사와 문화에 대한 깊이 있는 이해를 제공하며, 다양한 문화적 맥락을 포괄하는 분류 체계를 생성하는 데 기여합니다.

---




The method proposed in this paper, CHisAgent, is a multi-agent framework for constructing an event taxonomy in ancient Chinese cultural systems. This framework is divided into three main stages, each performed by specialized LLM (Large Language Model) agents. The methodology consists of the following three key components:

1. **Inducer**: This stage is responsible for deriving an initial event hierarchy from raw historical corpora. The Inducer extracts event instances and event types from historical documents of ancient China to create an initial event set. Multiple extractors are employed in this process to comprehensively gather various event types, and each event type is assigned to predefined domains.

2. **Expander**: The Expander addresses structural incompleteness caused by the Inducer's output by performing top-down expansion. In this stage, it identifies missing intermediate concepts based on the current layer's nodes and definitions, generating new child nodes as necessary to improve hierarchical coherence.

3. **Enricher**: The Enricher aims to enhance the completeness and coverage of the generated taxonomy while ensuring fidelity to historical evidence. It constructs a candidate event set from high-frequency events, topic-based events, and relations derived from existing historical knowledge bases, integrating these into the existing taxonomy to finalize the classification system.

These three stages are designed for collaboration among specialized LLM agents, allowing for the effective construction of a historical event taxonomy. This methodology provides a deeper understanding of ancient Chinese history and culture, contributing to the creation of a taxonomy that encompasses diverse cultural contexts.


<br/>
# Results


이 논문에서는 CHisAgent라는 다중 에이전트 프레임워크를 제안하여 고대 중국 문화 시스템의 사건 분류 체계를 구축하는 방법을 설명합니다. 이 프레임워크는 세 가지 주요 단계로 구성되어 있습니다: Inducer(유도기), Expander(확장기), Enricher(풍부화기). 각 단계는 서로 다른 역할을 수행하는 LLM(대형 언어 모델) 에이전트에 의해 수행됩니다.

#### 결과
1. **경쟁 모델**: CHisAgent는 여러 경쟁 모델과 비교되었습니다. 이 모델들은 LLM + Domains, LLM + Event Corpus, LLM + Event Corpus + Domains, 그리고 인간이 제작한 CHED 분류 체계 등입니다. 각 모델은 고대 중국 역사와 문화에 대한 사건 분류 체계를 구축하는 데 사용되었습니다.

2. **테스트 데이터**: 테스트 데이터는 고대 중국의 공식 역사서인 '사기'와 '이십사사'에서 추출된 사건들로 구성되었습니다. 이 데이터는 다양한 정치, 군사, 외교 및 사회적 사건을 포함하고 있습니다.

3. **메트릭**: 성능 평가는 다음과 같은 메트릭을 사용하여 수행되었습니다:
   - **Path Granularity**: 노드가 부모보다 더 구체적인지를 평가합니다.
   - **CSC (Structural Consistency)**: 계층 구조와 의미적 관계 간의 일치를 평가합니다.
   - **Coverage Rate**: 생성된 분류 체계가 역사적 사건을 얼마나 잘 포괄하는지를 측정합니다.
   - **Node Recall**: 자동 생성된 분류 체계가 인간이 제작한 분류 체계에서 얼마나 많은 노드를 포착했는지를 평가합니다.
   - **Novelty**: 생성된 분류 체계에서 새로운 노드의 비율을 측정합니다.
   - **Significance**: 생성된 분류 체계가 인간 제작 분류 체계에 비해 얼마나 세분화되어 있는지를 평가합니다.

4. **비교**: CHisAgent는 다른 모델들과 비교했을 때, 특히 Coverage Rate와 Node Recall에서 우수한 성능을 보였습니다. 예를 들어, CHisAgent는 75.13%의 Coverage Rate를 기록했으며, Node Recall에서도 68.89%를 달성했습니다. 반면, 경쟁 모델들은 상대적으로 낮은 성능을 보였습니다. 또한, CHisAgent는 Novelty와 Significance에서도 높은 점수를 기록하여, 생성된 분류 체계가 더 많은 새로운 정보를 포함하고 있음을 보여주었습니다.

이러한 결과는 CHisAgent가 고대 중국 역사와 문화에 대한 사건 분류 체계를 효과적으로 구축할 수 있는 강력한 도구임을 입증합니다.

---




This paper proposes a multi-agent framework called CHisAgent for constructing an event taxonomy in ancient Chinese cultural systems. The framework consists of three main stages: Inducer, Expander, and Enricher, each performed by LLM (Large Language Model) agents with distinct roles.

#### Results
1. **Competing Models**: CHisAgent was compared with several competing models, including LLM + Domains, LLM + Event Corpus, LLM + Event Corpus + Domains, and the human-crafted CHED taxonomy. Each model was used to construct an event taxonomy related to ancient Chinese history and culture.

2. **Test Data**: The test data consisted of events extracted from authoritative historical texts such as the "Records of the Grand Historian" and the "Twenty-Four Histories." This data included various political, military, diplomatic, and social events.

3. **Metrics**: Performance evaluation was conducted using the following metrics:
   - **Path Granularity**: Assesses whether a node is more specific than its parent.
   - **CSC (Structural Consistency)**: Evaluates the alignment between hierarchical structure and semantic relationships.
   - **Coverage Rate**: Measures how well the generated taxonomy covers historical events.
   - **Node Recall**: Evaluates how many nodes in the human taxonomy are captured by the automatically generated taxonomy.
   - **Novelty**: Quantifies the proportion of new nodes in the generated taxonomy.
   - **Significance**: Measures how much finer-grained the generated taxonomy is compared to the human taxonomy.

4. **Comparison**: CHisAgent demonstrated superior performance compared to other models, particularly in Coverage Rate and Node Recall. For instance, CHisAgent achieved a Coverage Rate of 75.13% and a Node Recall of 68.89%. In contrast, competing models showed relatively lower performance. Additionally, CHisAgent scored high in Novelty and Significance, indicating that the generated taxonomy contains more new information.

These results demonstrate that CHisAgent is a powerful tool for effectively constructing an event taxonomy related to ancient Chinese history and culture.


<br/>
# 예제



**예시: 트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋, 구체적인 테스크**

이 연구에서는 고대 중국 역사에 대한 사건 중심의 분류 체계를 구축하기 위해 다중 에이전트 프레임워크인 CHisAgent를 제안합니다. 이 프레임워크는 세 가지 주요 단계로 구성됩니다: 유도기(Inducer), 확장기(Expander), 그리고 풍부화기(Enricher). 각 단계에서 LLM(대형 언어 모델)이 특정 역할을 수행합니다.

1. **유도기(Inducer)**:
   - **입력**: 고대 중국 역사서인 '사기'와 같은 원시 역사 문서에서 추출한 사건 인스턴스와 사건 유형.
   - **출력**: 초기 사건 분류 체계. 예를 들어, "정치", "군사", "외교"와 같은 상위 도메인으로 분류된 사건 유형 목록.
   - **작업**: 원시 문서에서 사건을 추출하고, 이를 기반으로 사건 유형을 분류하여 초기 분류 체계를 생성합니다.

2. **확장기(Expander)**:
   - **입력**: 유도기에서 생성된 초기 분류 체계와 도메인 설명.
   - **출력**: 구조적 일관성을 보장하기 위해 추가된 중간 개념을 포함한 확장된 분류 체계.
   - **작업**: 상위 레벨에서 하위 레벨로의 계층적 확장을 통해 누락된 개념을 식별하고 추가합니다.

3. **풍부화기(Enricher)**:
   - **입력**: 확장기에서 생성된 분류 체계와 외부 구조화된 역사 자원.
   - **출력**: 역사적 증거와 외부 지식을 통합하여 완전성과 신뢰성을 높인 최종 분류 체계.
   - **작업**: 후보 사건을 수집하고, 이를 기존 분류 체계에 통합하여 최종적으로 풍부화된 분류 체계를 생성합니다.

**테스트 데이터 예시**:
- **입력**: "왕이 전쟁을 선포하다"라는 문장.
- **출력**: 
  - 사건 유형: "전쟁 선포"
  - 트리거: "선포"
  - 도메인: "군사"

이러한 방식으로, 연구자는 고대 중국 역사에 대한 포괄적이고 체계적인 사건 분류 체계를 구축하고, 이를 통해 LLM의 역사적 및 문화적 이해를 향상시키고자 합니다.

---




**Example: Specific Inputs and Outputs of Training and Testing Data, Specific Tasks**

In this study, we propose a multi-agent framework called CHisAgent for constructing an event-centered taxonomy of ancient Chinese history. This framework consists of three main stages: Inducer, Expander, and Enricher. Each stage involves a specific role performed by a large language model (LLM).

1. **Inducer**:
   - **Input**: Event instances and event types extracted from raw historical documents such as the "Records of the Grand Historian."
   - **Output**: An initial taxonomy of events. For example, a list of event types categorized under higher domains such as "Politics," "Military," and "Diplomacy."
   - **Task**: Extract events from raw documents and classify them to create an initial taxonomy.

2. **Expander**:
   - **Input**: The initial taxonomy generated by the Inducer and domain descriptions.
   - **Output**: An expanded taxonomy that includes additional intermediate concepts to ensure structural coherence.
   - **Task**: Perform hierarchical expansion from higher to lower levels to identify and add missing concepts.

3. **Enricher**:
   - **Input**: The taxonomy generated by the Expander and external structured historical resources.
   - **Output**: A final enriched taxonomy that improves completeness and fidelity by integrating historical evidence and external knowledge.
   - **Task**: Collect candidate events and integrate them into the existing taxonomy to produce a richly detailed final taxonomy.

**Example of Test Data**:
- **Input**: A sentence "The king declares war."
- **Output**: 
  - Event Type: "War Declaration"
  - Trigger: "Declares"
  - Domain: "Military"

Through this approach, the researchers aim to construct a comprehensive and systematic event taxonomy of ancient Chinese history, thereby enhancing the historical and cultural understanding of LLMs.

<br/>
# 요약

이 논문에서는 고대 중국 문화 시스템의 사건 분류 체계를 구축하기 위해 CHisAgent라는 다중 에이전트 프레임워크를 제안합니다. 이 프레임워크는 사건 계층 구조를 유도하는 Inducer, 누락된 개념을 추가하는 Expander, 외부 자료를 통합하여 신뢰성을 보장하는 Enricher의 세 가지 역할로 구성됩니다. 실험 결과, CHisAgent는 구조적 일관성과 범위에서 향상된 성능을 보여주며, 고대 중국 역사와 문화에 대한 포괄적인 사건 분류 체계를 성공적으로 구축했습니다.

---

This paper proposes a multi-agent framework called CHisAgent for constructing an event taxonomy in ancient Chinese cultural systems. The framework consists of three roles: the Inducer, which derives the event hierarchy; the Expander, which adds missing concepts; and the Enricher, which integrates external resources to ensure fidelity. Experimental results demonstrate that CHisAgent achieves improved structural coherence and coverage, successfully creating a comprehensive event taxonomy for ancient Chinese history and culture.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: CHisAgent 프레임워크의 구조를 보여줍니다. 이 다이어그램은 Inducer, Expander, Enricher의 세 가지 주요 역할을 시각적으로 설명하며, 각 단계에서의 작업 흐름을 명확히 나타냅니다. 이를 통해 각 역할이 어떻게 협력하여 고대 중국 역사 및 문화의 사건 분류 체계를 구축하는지를 이해할 수 있습니다.
   - **Figure 2**: 다양한 방법론에 따른 평가 메트릭을 비교하는 레이더 차트입니다. 이 차트는 각 방법이 특정 도메인에서 어떻게 성능을 발휘하는지를 보여주며, CHisAgent가 여러 도메인에서 균형 잡힌 성능을 보임을 강조합니다.

2. **테이블**
   - **Table 1**: 다양한 방법으로 생성된 분류 체계의 평가 결과를 요약합니다. 각 방법의 Path Granularity, CSC, Coverage Rate, Node Recall, Novelty, Significance 등의 메트릭을 비교하여 CHisAgent가 다른 방법들보다 높은 Coverage Rate와 Node Recall을 달성했음을 보여줍니다.
   - **Table 2**: CHisAgent의 세 가지 구성 요소(Inducer, Expander, Enricher)의 기여도를 평가한 결과입니다. 각 구성 요소의 유무에 따른 성능 변화를 보여주며, Enricher의 추가가 전체 성능을 크게 향상시킨다는 것을 나타냅니다.

3. **어펜딕스**
   - **Appendix A**: 사용된 데이터셋의 통계 정보를 제공합니다. 이는 연구의 기초가 되는 데이터의 규모와 특성을 이해하는 데 도움을 줍니다.
   - **Appendix B**: 도메인별 이벤트 유형의 분포를 보여줍니다. 이는 각 도메인에서 어떤 사건들이 중요한지를 파악하는 데 유용합니다.
   - **Appendix C**: 클러스터링된 역사적 주제를 나열하여, 분류 체계 구축에 사용된 주제의 맥락을 제공합니다.

### Insights from Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the structure of the CHisAgent framework. This diagram visually explains the roles of the Inducer, Expander, and Enricher, clearly depicting the workflow at each stage. It helps in understanding how these roles collaborate to construct a taxonomy of events in ancient Chinese history and culture.
   - **Figure 2**: A radar chart comparing evaluation metrics across different methodologies. This chart highlights how each method performs in specific domains, emphasizing that CHisAgent exhibits balanced performance across various domains.

2. **Tables**
   - **Table 1**: Summarizes the evaluation results of taxonomies generated by different methods. It compares metrics such as Path Granularity, CSC, Coverage Rate, Node Recall, Novelty, and Significance, demonstrating that CHisAgent achieves higher Coverage Rate and Node Recall compared to other methods.
   - **Table 2**: Presents the contributions of the three components of CHisAgent (Inducer, Expander, Enricher). It shows how the presence or absence of each component affects performance, indicating that the addition of the Enricher significantly enhances overall performance.

3. **Appendices**
   - **Appendix A**: Provides statistical information about the dataset used. This helps in understanding the scale and characteristics of the data that form the foundation of the research.
   - **Appendix B**: Displays the distribution of event types across domains, which is useful for identifying which events are significant within each domain.
   - **Appendix C**: Lists clustered historical topics, providing context for the themes used in constructing the taxonomy.

These elements collectively contribute to a comprehensive understanding of the methodology, results, and implications of the research on historical taxonomy construction in ancient Chinese cultural systems.

<br/>
# refer format:


### BibTeX Citation

```bibtex
@inproceedings{Tang2026,
  author    = {Xuemei Tang and Chengxi Yan and Jianghang Gu and Chu-Ren Huang},
  title     = {CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems},
  publisher = {Association for Computational Linguistics},
  email     = {xuemeitang00@gmail.com},
  url       = {https://anonymous.4open.science/r/ACL-BC1E/models}
}
```

### Chicago Style Citation

Tang, Xuemei, Chengxi Yan, Jianghang Gu, and Chu-Ren Huang. 2026. "CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems." https://anonymous.4open.science/r/ACL-BC1E/models.
