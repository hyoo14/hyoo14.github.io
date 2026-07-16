---
layout: post
title:  "[2026]Generating Literature-Driven Scientific Theories at Scale"
date:   2026-07-16 21:36:43 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 THEORIZER라는 시스템을 사용하여 13,744개의 과학 논문에서 이론을 생성하는 방법을 제시하고, 문헌 기반 이론 생성이 파라메트릭 지식에 비해 더 높은 정확성과 예측력을 제공함을 보여준다.


짧은 요약(Abstract) :


이 논문의 초록에서는 현대의 자동화된 과학 발견 시스템이 과학 실험을 생성하는 데 중점을 두고 있지만, 이론 구축과 같은 고차원 과학 활동을 수행하는 시스템은 상대적으로 덜 탐구되었다고 언급하고 있습니다. 연구자들은 대규모 과학 문헌에서 질적 및 양적 법칙으로 구성된 이론을 합성하는 문제를 정의하고, 13,700개의 출처 논문을 사용하여 2,900개의 이론을 합성하는 실험을 수행했습니다. 이 과정에서 문헌 기반 생성 방법과 매개변수 지식 기반 생성 방법, 정확성 중심의 목표와 참신성 중심의 목표가 이론의 특성에 미치는 영향을 분석했습니다. 실험 결과, 문헌 지원 방법이 매개변수 LLM 메모리를 사용하는 방법에 비해 기존 증거와의 일치도 및 이후 작성된 4,600개의 논문에서의 예측 정확도가 유의미하게 향상된 이론을 생성한다는 것을 보여주었습니다.



The abstract of this paper states that contemporary automated scientific discovery systems have focused on generating scientific experiments, while systems that perform higher-level scientific activities, such as theory building, remain relatively underexplored. The researchers define the problem of synthesizing theories consisting of qualitative and quantitative laws from large corpora of scientific literature and conduct experiments using 13,700 source papers to synthesize 2,900 theories. They analyze how generation using literature grounding versus parametric knowledge, and accuracy-focused versus novelty-focused generation objectives affect theory properties. The experiments show that the literature-supported method creates theories that significantly outperform those generated using parametric LLM memory in terms of matching existing evidence and predicting future results from 4,600 subsequently written papers.


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



이 논문에서 제안하는 방법론은 "THEORIZER"라는 시스템을 기반으로 하며, 이는 대규모 과학 문헌에서 이론을 생성하는 데 중점을 두고 있습니다. 이 시스템은 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. **문헌 검색**: 사용자가 제공한 이론 쿼리를 바탕으로 관련 연구 논문을 검색합니다. 이 과정에서 AI2 PAPER FINDER를 사용하여 최대 100개의 관련 논문을 수집합니다. 수집된 논문은 SEMANTIC SCHOLAR를 통해 다운로드되고, OCR 기반의 추출 파이프라인을 통해 전체 텍스트로 변환됩니다. 만약 초기 검색에서 충분한 논문이 확보되지 않으면, 검색된 논문의 참고 문헌을 통해 추가 논문을 찾아내어 후보 집합을 확장합니다.

2. **증거 추출**: 각 논문에서 이론 생성을 지원하기 위해 필요한 정보를 추출합니다. 이 과정에서 이론 쿼리에 맞춘 추출 스키마를 생성하고, 이를 통해 각 논문에서 이론 관련 정보를 구조화된 형식으로 수집합니다. 예를 들어, 메모리 증강 언어 모델에 대한 쿼리의 경우, 사용된 메모리 메커니즘, 평가 설정, 성능 비교 등의 정보를 포함할 수 있습니다.

3. **이론 합성**: 추출된 증거를 바탕으로 후보 이론을 생성합니다. 이 과정에서 언어 모델을 사용하여 증거를 기반으로 한 이론을 유도합니다. 생성된 이론은 자기 반성 단계를 거쳐 내부 일관성, 증거 귀속 및 구체성을 개선합니다.

4. **이론 평가**: 생성된 이론의 품질을 평가하기 위해 새로운 실험을 수행하는 것이 비현실적이므로, 백테스팅 패러다임을 개발하여 이론의 예측이 이후 발표된 실험 결과와 얼마나 일치하는지를 평가합니다. 이 과정에서 이론의 예측이 실제로 얼마나 잘 지지되는지를 측정합니다.

5. **문헌 기반 vs. 매개변수 기반 생성**: 이 시스템은 문헌 기반 생성과 매개변수 기반 생성의 두 가지 접근 방식을 비교합니다. 문헌 기반 생성은 실제 연구 결과에 기반하여 이론을 생성하는 반면, 매개변수 기반 생성은 사전 훈련된 언어 모델의 지식만을 사용하여 이론을 생성합니다. 이 두 접근 방식의 결과를 비교하여 이론의 구체성, 경험적 지원, 예측 정확성, 참신성 및 그럴듯함을 분석합니다.

이러한 방법론은 과학적 이론 생성을 자동화하고, 문헌에 기반한 증거를 통해 이론의 품질을 높이는 데 기여합니다. 이 시스템은 다양한 주제에 걸쳐 수천 개의 이론을 생성하고, 이론의 특성을 체계적으로 비교하여 과학적 발견을 촉진하는 데 중요한 역할을 할 수 있습니다.

---




The methodology proposed in this paper is based on a system called "THEORIZER," which focuses on generating theories from large-scale scientific literature. This system consists of the following key components:

1. **Literature Discovery**: It searches for relevant research papers based on a user-provided theory query. In this process, AI2 PAPER FINDER is used to collect up to 100 relevant papers. The collected papers are downloaded through SEMANTIC SCHOLAR and converted to full text using an OCR-based extraction pipeline. If insufficient papers are obtained in the initial search, the system expands the candidate set by searching the references of the retrieved papers.

2. **Evidence Extraction**: The system extracts necessary information from each paper to support theory generation. It generates an extraction schema tailored to the theory query and uses this schema to collect theory-relevant information in a structured format from each paper. For example, for queries concerning memory-augmented language models, the schema might include the memory mechanisms used, evaluation settings, and performance comparisons.

3. **Theory Synthesis**: The extracted evidence is used to generate candidate theories. This process involves using a language model to induce theories grounded in the evidence. The generated theories undergo a self-reflection step to improve internal consistency, evidence attribution, and specificity.

4. **Theory Evaluation**: To assess the quality of the generated theories, a backtesting paradigm is developed, as conducting new experiments at scale is impractical. This paradigm evaluates how well the predictions made by the theories align with experimental results reported in subsequently published literature.

5. **Literature-Supported vs. Parametric Generation**: The system compares two approaches: literature-supported generation and parametric generation. Literature-supported generation creates theories based on actual research findings, while parametric generation relies solely on the knowledge of a pre-trained language model. The results of these two approaches are compared to analyze the specificity, empirical support, predictive accuracy, novelty, and plausibility of the theories.

This methodology automates the generation of scientific theories and enhances the quality of theories by grounding them in literature-based evidence. The system can generate thousands of theories across diverse topics and systematically compare their characteristics, playing a crucial role in facilitating scientific discovery.


<br/>
# Results



이 연구에서는 문헌 기반 이론 생성 시스템인 THEORIZER를 사용하여 두 가지 주요 조건(정확성 중심 및 참신성 중심)에서 이론을 생성하고, 이를 통해 생성된 이론의 품질을 평가했습니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구에서는 두 가지 생성 방법을 비교했습니다. 하나는 문헌 기반의 이론 생성 방법(문헌 지원 이론)이고, 다른 하나는 파라메트릭 지식만을 사용하는 방법(파라메트릭 이론)입니다. 이 두 방법은 각각의 이론 생성 조건에 따라 정확성 중심과 참신성 중심으로 나뉘어 평가되었습니다.

2. **테스트 데이터**: 이론의 예측 정확성을 평가하기 위해, 연구팀은 2025년 7월부터 12월 사이에 발표된 4,554개의 논문을 사용했습니다. 이 논문들은 생성된 이론의 예측을 검증하는 데 사용되었습니다.

3. **메트릭**: 이론의 품질은 여러 메트릭을 통해 평가되었습니다. 주요 메트릭은 다음과 같습니다:
   - **특이성(Specificity)**: 이론이 얼마나 구체적이고 테스트 가능한 주장을 포함하는지를 평가합니다.
   - **경험적 지원(Empirical Support)**: 이론의 주장이 기존의 경험적 증거와 얼마나 일치하는지를 평가합니다.
   - **예측 정확성(Predictive Accuracy)**: 이론의 주장이 미래의 실험 결과와 얼마나 잘 일치하는지를 평가합니다.
   - **참신성(Novelty)**: 이론이 기존 연구에서 명시적으로 언급되지 않은 새로운 통찰을 얼마나 제공하는지를 평가합니다.
   - **그럴듯함(Plausibility)**: 이론이 과학적으로 그럴듯한 이유나 메커니즘 설명을 얼마나 잘 제공하는지를 평가합니다.

4. **비교 결과**: 연구 결과에 따르면, 문헌 지원 이론은 파라메트릭 이론에 비해 모든 메트릭에서 우수한 성능을 보였습니다. 특히, 문헌 지원 이론은 경험적 지원과 예측 정확성에서 유의미한 향상을 보였으며, 이는 문헌 기반의 증거를 활용한 결과로 해석됩니다. 반면, 참신성 중심 조건에서는 문헌 지원 이론이 더 많은 경험적 지원과 그럴듯함을 보였지만, 특이성에서는 큰 차이를 보이지 않았습니다.

5. **결론**: 문헌 기반 이론 생성 방법은 파라메트릭 방법에 비해 이론의 품질을 높이는 데 효과적이며, 특히 예측 정확성과 경험적 지원에서 두드러진 성과를 보였습니다. 그러나 참신성 중심의 이론 생성에서는 더 많은 위험을 감수해야 하며, 이론의 예측이 문헌에서 충분히 검증되지 않을 수 있습니다.

---




In this study, the literature-based theory generation system, THEORIZER, was utilized to generate theories under two main conditions: accuracy-focused and novelty-focused. The key findings of the study are as follows:

1. **Competing Models**: The study compared two generation methods. One was the literature-supported theory generation method, and the other was the parametric knowledge-only method. These two methods were evaluated based on their respective theory generation conditions, categorized as accuracy-focused and novelty-focused.

2. **Test Data**: To evaluate the predictive accuracy of the theories, the research team used 4,554 papers published between July and December 2025. These papers were used to validate the predictions made by the generated theories.

3. **Metrics**: The quality of the theories was assessed using several metrics. The main metrics included:
   - **Specificity**: Evaluates how specific and testable the claims of the theories are.
   - **Empirical Support**: Assesses how well the claims of the theories align with existing empirical evidence.
   - **Predictive Accuracy**: Measures how well the claims of the theories match future experimental results.
   - **Novelty**: Evaluates the extent to which the theories provide new insights not explicitly stated in prior work.
   - **Plausibility**: Assesses how well the theories provide plausible scientific rationale or mechanistic explanations.

4. **Comparative Results**: According to the study results, literature-supported theories outperformed parametric theories across all metrics. Notably, literature-supported theories showed significant improvements in empirical support and predictive accuracy, which can be interpreted as a result of leveraging literature-based evidence. In contrast, under the novelty-focused condition, literature-supported theories exhibited more empirical support and plausibility, but did not show significant differences in specificity.

5. **Conclusion**: The literature-based theory generation method is effective in enhancing the quality of theories compared to the parametric method, particularly in terms of predictive accuracy and empirical support. However, generating theories focused on novelty entails greater risks, as the predictions made by these theories may not be sufficiently validated in the literature.


<br/>
# 예제



이 논문에서는 과학 이론 생성을 위한 시스템인 THEORIZER를 소개하고, 이 시스템이 어떻게 문헌 기반의 이론을 생성하는지에 대한 구체적인 과정을 설명합니다. 이 과정은 크게 세 가지 단계로 나눌 수 있습니다: 문헌 검색, 증거 추출, 이론 합성입니다.

1. **문헌 검색**: 사용자가 제공한 이론 쿼리를 바탕으로 관련 연구 논문을 검색합니다. 예를 들어, "LLM(대형 언어 모델)과 메모리의 관계"라는 쿼리를 입력하면, THEORIZER는 관련된 최대 100개의 논문을 검색하여 이론 생성을 위한 기초 자료로 사용합니다.

2. **증거 추출**: 검색된 논문에서 이론 생성을 위한 관련 정보를 추출합니다. 이 단계에서는 각 논문에서 메모리 유형, 성능 비교, 실험 설정 등과 같은 구체적인 데이터를 추출하여 JSON 형식으로 정리합니다. 예를 들어, 특정 논문에서 "LLM이 메모리를 사용할 때 성능이 20% 향상되었다"는 정보를 추출할 수 있습니다.

3. **이론 합성**: 추출된 증거를 바탕으로 이론을 생성합니다. 이 단계에서는 언어 모델을 사용하여 추출된 정보를 기반으로 이론을 작성합니다. 예를 들어, "LLM은 메모리 유형에 따라 성능이 달라지며, 특정 작업에 맞는 메모리 형식을 사용하는 것이 효과적이다"라는 이론을 생성할 수 있습니다.

이러한 과정을 통해 THEORIZER는 수천 개의 이론을 생성하고, 각 이론의 예측 정확성을 평가하여 문헌에 기반한 이론의 품질을 분석합니다. 이 시스템은 문헌 기반의 이론 생성을 통해 과학적 발견을 지원하는 데 기여할 수 있습니다.




This paper introduces a system for generating scientific theories called THEORIZER and explains the specific process by which this system generates literature-based theories. This process can be divided into three main stages: literature discovery, evidence extraction, and theory synthesis.

1. **Literature Discovery**: The system searches for relevant research papers based on a user-provided theory query. For example, if the query is "the relationship between LLMs (large language models) and memory," THEORIZER will search for up to 100 related papers to use as foundational material for theory generation.

2. **Evidence Extraction**: The system extracts relevant information from the searched papers to support theory generation. In this stage, specific data such as memory types, performance comparisons, and experimental settings are extracted and organized in JSON format. For instance, it might extract information like "the performance of LLM improved by 20% when using memory."

3. **Theory Synthesis**: Based on the extracted evidence, the system generates theories. In this stage, a language model is used to write theories based on the extracted information. For example, it might generate a theory stating, "LLMs perform differently based on the type of memory used, and using a memory format suited to specific tasks is effective."

Through this process, THEORIZER generates thousands of theories and evaluates the predictive accuracy of each theory to analyze the quality of literature-based theories. This system can contribute to scientific discovery by supporting the generation of theories grounded in literature.

<br/>
# 요약


이 논문에서는 THEORIZER라는 시스템을 사용하여 13,744개의 과학 논문에서 이론을 생성하는 방법을 제시하고, 문헌 기반 이론 생성이 파라메트릭 지식에 비해 더 높은 정확성과 예측력을 제공함을 보여준다. 실험 결과, 문헌 지원 이론은 기존 증거와의 일치도 및 미래 결과 예측에서 유의미한 개선을 보였다. 예를 들어, "다중 에이전트 사회적 존재 이론"은 LLM 기반 지능형 튜터링 시스템에서의 학생 참여와 동기 부여에 대한 새로운 통찰을 제공한다.

---

This paper presents a method using a system called THEORIZER to generate theories from 13,744 scientific papers, demonstrating that literature-based theory generation offers higher accuracy and predictive power compared to parametric knowledge. Experimental results show that literature-supported theories significantly improve alignment with existing evidence and predictions of future outcomes. For instance, the "Emergent Multi-Agent Social Presence Theory" provides new insights into student engagement and motivation in LLM-powered intelligent tutoring systems.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: THEORIZER 시스템의 개요를 보여줍니다. 이 시스템은 사용자가 제공한 이론 쿼리를 기반으로 관련 논문을 검색하고, 이들 논문에서 이론에 필요한 정보를 추출하여 이론을 생성합니다. 이 과정은 이론 생성의 구조적 흐름을 시각적으로 나타내며, 문헌 기반 이론 생성의 복잡성을 이해하는 데 도움을 줍니다.
   - **Figure 2**: 예측 정확성 평가 절차를 설명합니다. 이 피규어는 각 이론 법칙에 대해 언어 모델이 생성한 예측을 기반으로 최근 문헌에서 이를 평가하는 방법을 보여줍니다. 이 과정은 이론의 예측이 실제 실험 결과와 얼마나 잘 일치하는지를 평가하는 데 중요한 역할을 합니다.

2. **테이블**
   - **Table 1**: 문헌 기반 이론과 파라메트릭 이론의 특성을 비교합니다. 이 표는 두 가지 생성 조건(정확성 중심 및 참신성 중심)에서 이론의 구체성, 경험적 지원, 예측 정확성, 참신성 및 그럴듯함을 평가한 결과를 보여줍니다. 문헌 지원 이론이 파라메트릭 이론보다 전반적으로 더 높은 점수를 받았음을 나타내며, 이는 문헌 기반 접근 방식이 이론의 질을 향상시키는 데 기여함을 시사합니다.
   - **Table 2**: 예측 정확성을 평가한 결과를 요약합니다. 이 표는 문헌 기반 이론이 예측 정확성에서 파라메트릭 이론보다 더 높은 성과를 보였음을 보여줍니다. 이는 문헌을 활용한 이론 생성이 더 신뢰할 수 있는 예측을 가능하게 한다는 것을 강조합니다.
   - **Table 3**: 이론의 참신성을 평가한 결과를 보여줍니다. 이 표는 문헌 기반 이론이 파라메트릭 이론보다 더 높은 참신성을 보였음을 나타내며, 이는 문헌을 통해 새로운 통찰을 도출하는 데 효과적임을 시사합니다.

3. **어펜딕스**
   - 어펜딕스에는 생성된 이론, 추출 스키마, 예측 정확성 평가 및 참신성 평가와 같은 추가 자료가 포함되어 있습니다. 이 자료들은 연구의 투명성을 높이고, 다른 연구자들이 이론 생성 및 평가 방법을 재현할 수 있도록 돕습니다. 또한, 다양한 이론의 예시가 제공되어 이론 생성의 실제 적용 사례를 보여줍니다.

---

### Results and Insights from Other Components (Diagrams, Figures, Tables, Appendix)

1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates the overview of the THEORIZER system. The system searches for relevant papers based on a user-provided theory query, extracts necessary information from these papers, and generates theories. This visual representation helps understand the structural flow of literature-based theory generation and the complexity involved.
   - **Figure 2**: This figure explains the evaluation procedure for predictive accuracy. It shows how predictions generated by a language model for each theory law are evaluated against recent literature. This process plays a crucial role in assessing how well the predictions of the theories align with actual experimental results.

2. **Tables**
   - **Table 1**: This table compares the characteristics of literature-supported theories and parametric theories. It presents the evaluation results of theories across two generation conditions (accuracy-focused and novelty-focused) in terms of specificity, empirical support, predictive accuracy, novelty, and plausibility. The results indicate that literature-supported theories scored higher overall, suggesting that the literature-based approach contributes to the quality of theories.
   - **Table 2**: This table summarizes the results of predictive accuracy evaluations. It shows that literature-supported theories outperformed parametric theories in predictive accuracy. This emphasizes that literature-informed theory generation enables more reliable predictions.
   - **Table 3**: This table presents the results of novelty evaluations for the theories. It indicates that literature-supported theories exhibited higher novelty compared to parametric theories, suggesting that leveraging literature is effective in deriving new insights.

3. **Appendix**
   - The appendix includes additional materials such as generated theories, extraction schemas, predictive accuracy evaluations, and qualified novelty evaluations. These materials enhance the transparency of the research and assist other researchers in reproducing the theory generation and evaluation methods. Furthermore, various examples of theories are provided, showcasing practical applications of theory generation.

<br/>
# refer format:
### BibTeX Entry

```bibtex
@inproceedings{jansen2026generating,
  title={Generating Literature-Driven Scientific Theories at Scale},
  author={Peter Jansen and Peter Clark and Doug Downey and Daniel S. Weld},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={14677--14698},
  year={2026},
  month={July},
  publisher={Association for Computational Linguistics},
  email={peterj@allenai.org}
}
```

### Chicago Style Citation

Peter Jansen, Peter Clark, Doug Downey, and Daniel S. Weld. "Generating Literature-Driven Scientific Theories at Scale." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 14677–14698. Association for Computational Linguistics, July 2026. 
