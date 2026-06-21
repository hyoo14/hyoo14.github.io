---
layout: post
title:  "[2026]Workflow Matters: Comparing Human Translators and Multi-Agent LLMs in Literary Translation"
date:   2026-06-21 08:41:14 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 두 가지 다중 에이전트 번역 시스템(PD-MAS와 LCD-MAS)을 비교하여 문학 번역의 품질을 평가하였다.


짧은 요약(Abstract) :



이 연구는 문학 번역에서 전문 인간 번역가와 다중 에이전트 기반의 대형 언어 모델(LLM) 간의 번역 품질을 비교합니다. 연구에서는 두 가지 AI 시스템을 사용하여 번역 작업을 수행했습니다. 첫 번째 시스템은 전문 인간 번역의 관행을 모방하여 초안 작성과 수정 단계로 나누어 작업합니다. 두 번째 시스템은 LLM의 능력에 맞춰 번역 과정을 재설계하여, 전략적 계획, 스타일 수정 및 일관성 검사를 전문으로 하는 AI 에이전트가 각각의 세부 작업을 처리합니다. 전문가 평가 결과, 두 AI 시스템 모두 전문 인간 번역가와 유사한 정확도를 달성했으며, LLM 능력 기반 시스템은 스타일적 품질과 시적 언어에서 우수한 성과를 보였지만 가끔 불필요한 내용을 추가하기도 했습니다. 반면, 관행 기반 시스템은 간결한 번역을 제공했지만 때때로 일관성이 부족했습니다. 블라인드 평가에서는 두 AI 시스템의 번역이 종종 인간 번역보다 선호되었으며, 특히 유창성 측면에서 두드러졌습니다. 이 연구는 LLM의 능력에 맞춰 번역 작업 흐름을 재구성하는 것이 뛰어난 결과를 낳을 수 있음을 보여주며, 특정 측면에서 인간 성과를 초월할 수 있음을 시사합니다.




This study compares the translation quality between professional human translators and multi-agent-based large language models (LLMs) in literary translation. Two AI systems were employed to perform the translation tasks. The first system mimics professional human translation practices, dividing the work into drafting and revision phases. The second system redesigns the process specifically for LLM capabilities, with specialized AI agents handling strategic planning, stylistic refinement, and coherence checking. Expert evaluations revealed that both AI systems achieved accuracy comparable to professional human translators, with the LLM-capability-driven system producing superior stylistic qualities and poetic language, although it occasionally added extraneous content. Meanwhile, the practice-derived system delivered concise translations but sometimes lacked cohesive flow. Blind evaluations showed that translations from both AI systems were frequently preferred over human translations, particularly in terms of fluency. This study demonstrates that rethinking translation workflows around LLM capabilities can yield exceptional results, sometimes surpassing human performance in certain aspects.


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



이 연구에서는 두 가지 다중 에이전트 번역 시스템을 비교하여 문학 번역의 품질을 평가했습니다. 첫 번째 시스템은 인간 번역가의 전통적인 작업 흐름을 모방한 '실천 기반 다중 에이전트 시스템(Practice-Derived Multi-Agent System, PD-MAS)'이고, 두 번째 시스템은 대형 언어 모델(LLM)의 능력에 맞춰 설계된 'LLM 능력 기반 다중 에이전트 시스템(LLM-Capability-Driven Multi-Agent System, LCD-MAS)'입니다.

#### 1. PD-MAS (실천 기반 다중 에이전트 시스템)
PD-MAS는 ISO 17100:2015 번역 서비스 요구 사항에 맞춰 설계되었습니다. 이 시스템은 두 개의 주요 단계로 구성됩니다: 

- **사전 제작 단계(Pre-Production Stage)**: 이 단계에서는 두 개의 전문 에이전트가 필수 자원을 준비합니다. 텍스트 분석가는 원본 텍스트의 특성(장르, 도메인, 목적 및 스타일적 특징)을 분석하고, 용어 전문가는 일관성을 위해 이중 언어 용어 목록을 작성합니다.

- **제작 단계(Production Stage)**: 이 단계에서는 번역가가 분석 및 용어 자원을 바탕으로 목표 텍스트를 생성합니다. 번역가가 자가 점검을 한 후, 텍스트는 수정자에게 전달되어 원본 텍스트와 목표 텍스트 간의 정확성과 완전성을 비교 분석합니다. 마지막으로, 검토자가 언어적 및 스타일적 일관성을 보장하고, 교정자가 최종 품질 검사를 수행합니다.

#### 2. LCD-MAS (LLM 능력 기반 다중 에이전트 시스템)
LCD-MAS는 LLM의 계산적 특성에 맞춰 설계되었으며, 세분화된 작업 분해와 전용 스타일 처리 단계를 포함합니다. 이 시스템은 다음과 같은 세 가지 상호 연결된 단계로 운영됩니다:

- **사전 번역 계획(Pre-Translation Planning)**: 이 단계에서는 두 개의 에이전트가 글로벌 컨텍스트를 설정합니다. 요약 생성기는 주요 사건, 캐릭터 및 주제를 포착하는 간결한 내러티브 요약을 생성하고, 전략 계획자는 청중의 기대, 텍스트 유형 및 문화적 참조를 다루는 포괄적인 번역 계획을 개발합니다.

- **번역 및 스타일 재작성(Translation and Stylistic Rewriting)**: 각 원본 청크에 대해 번역가는 사전 번역 단계에서 생성된 요약 및 전략 계획을 사용하여 초기 번역을 생성합니다. 수정자는 정확성을 확인하고, 스타일 가이드 생성기는 적절한 스타일 향상을 식별합니다. 마지막으로, 스타일 재작성자는 이러한 권장 사항을 적용하여 의미를 보존하면서 문학적 장치를 통합합니다.

- **최종화(Finalization)**: 이 단계에서는 독립적으로 번역된 세그먼트를 연결하여 일관성을 보장합니다. 텍스트가 연결된 후, 스타일 가이드 생성기는 불일치 및 어색한 전환을 감지하고 편집자가 구현할 수 있는 지침을 생성하여 글로벌 일관성을 유지합니다.

이러한 아키텍처는 LLM의 계산적 특성에 맞춰 번역 프로세스를 재구성하여 의미 전달과 스타일적 정제를 분리함으로써 문학 번역의 품질을 향상시키는 것을 목표로 합니다.

---




This study compared two multi-agent translation systems to evaluate the quality of literary translation. The first system is the 'Practice-Derived Multi-Agent System (PD-MAS)', which mimics traditional workflows of human translators, and the second system is the 'LLM-Capability-Driven Multi-Agent System (LCD-MAS)', designed around the capabilities of large language models (LLMs).

#### 1. PD-MAS (Practice-Derived Multi-Agent System)
PD-MAS is designed in accordance with the ISO 17100:2015 translation service requirements. This system consists of two main stages:

- **Pre-Production Stage**: In this stage, two specialized agents prepare essential resources. The text analyst analyzes the characteristics of the source text (genre, domain, purpose, and stylistic features), while the terminology expert creates bilingual terminology lists for consistency.

- **Production Stage**: In this stage, the translator generates the target text based on the analysis and terminology resources. After self-checking, the text is passed to the reviser, who conducts a comparative analysis between the source and target texts, focusing on accuracy and completeness. Finally, the reviewer ensures linguistic and stylistic coherence, and the proofreader performs the final quality check.

#### 2. LCD-MAS (LLM-Capability-Driven Multi-Agent System)
LCD-MAS is designed around the computational characteristics of LLMs, featuring granular task decomposition and dedicated stylistic processing stages. This system operates through three interconnected stages:

- **Pre-Translation Planning**: In this stage, two agents establish global context. The summarizer generates a concise narrative summary capturing main events, characters, and themes, while the strategy planner develops a comprehensive translation plan addressing audience expectations, text type, and cultural references.

- **Translation and Stylistic Rewriting**: For each source chunk, the translator produces an initial translation using the summary and strategy plan from the pre-translation stage. The reviser checks for accuracy, and then the style guide generator identifies appropriate stylistic enhancements. Finally, the stylistic rewriter applies these recommendations, incorporating literary devices while preserving semantic content.

- **Finalization**: This stage ensures coherence across independently translated segments, which are concatenated at this stage. After text concatenation, the style guide generator detects inconsistencies and awkward transitions, producing guidelines for the editor to implement, maintaining global coherence while preserving established stylistic qualities.

This architecture reconfigures the translation process around the computational characteristics of LLMs, aiming to enhance the quality of literary translation by separating semantic transfer from stylistic refinement.


<br/>
# Results



이 연구에서는 두 가지 다중 에이전트 번역 시스템(실무 기반 다중 에이전트 시스템, PD-MAS와 LLM 기능 기반 다중 에이전트 시스템, LCD-MAS)의 번역 품질을 전문 인간 번역과 비교했습니다. 평가 기준은 정확성, 유창성, 그리고 전체 선호도였습니다. 

1. **정확성 분석**: 
   - 정확성 평가의 일관성을 확인하기 위해 두 번역 전문가 간의 상관 계수를 계산했습니다. 결과적으로 ICC(2,k)는 0.74로 나타났으며, 이는 평가 기준의 일관된 적용을 나타냅니다. 
   - 세 가지 번역 접근 방식(인간 번역, PD-MAS, LCD-MAS)의 평균 정확성 점수를 비교한 결과, Friedman 검정에서 통계적으로 유의미한 차이가 발견되지 않았습니다(χ²(2) = 0.37, p = .832). 모든 접근 방식의 중앙값 정확성 점수는 동일하게 8로 나타났습니다.

2. **유창성 분석**: 
   - 유창성 평가에서는 두 평가자 간의 높은 일관성을 보였으며, ICC는 0.83으로 나타났습니다. 
   - 유창성 점수는 세 가지 번역 유형 간에 통계적으로 유의미한 차이를 보였습니다(χ²(2) = 14.92, p < .001). LCD-MAS는 인간 번역(Mdn = 7)과 PD-MAS(Mdn = 7.5)보다 유의미하게 높은 유창성 점수(Mdn = 8)를 받았습니다.

3. **번역 선호도 분석**: 
   - 총 120개의 평가에서, LCD-MAS가 52표(43.33%)로 가장 많이 선호되었고, PD-MAS는 39표(32.50%), 인간 번역은 29표(24.17%)로 나타났습니다. 
   - 평가자들은 유창성이 번역 선호도에 더 큰 영향을 미쳤다고 언급했습니다. 

이 연구의 결과는 LLM 기반 다중 에이전트 시스템이 인간 번역가와 유사한 정확성을 달성할 수 있으며, 특정 측면에서 유창성과 선호도에서 인간 번역을 초월할 수 있음을 보여줍니다. 특히, LLM 기능 기반 시스템은 문학적 품질과 스타일적 풍부함을 향상시키는 데 효과적이었지만, 때때로 원본 텍스트에 없는 내용을 추가하는 경향이 있었습니다. 반면, 실무 기반 시스템은 더 간결한 번역을 제공했지만, 유기적인 흐름이 부족한 경우가 있었습니다.




This study compared the translation quality of two multi-agent translation systems (Practice-Derived Multi-Agent System, PD-MAS, and LLM-Capability-Driven Multi-Agent System, LCD-MAS) against professional human translations. The evaluation criteria included accuracy, fluency, and overall preference.

1. **Accuracy Analysis**: 
   - To assess the consistency of accuracy evaluations, the intra-class correlation coefficient (ICC) was calculated between the two translation experts. The result showed an ICC of 0.74, indicating consistent application of the evaluation criteria.
   - Statistical comparisons of average accuracy scores across the three translation approaches (human translations, PD-MAS, and LCD-MAS) revealed no statistically significant differences (Friedman test: χ²(2) = 0.37, p = .832). The median accuracy score was identical across all three approaches, with a median of 8.

2. **Fluency Analysis**: 
   - Fluency evaluations showed high consistency between raters, with an ICC of 0.83. 
   - In contrast to accuracy scores, fluency ratings revealed statistically significant differences among the three translation types (Friedman test: χ²(2) = 14.92, p < .001). The LCD-MAS received significantly higher fluency scores (Mdn = 8) compared to both human translations (Mdn = 7) and PD-MAS (Mdn = 7.5).

3. **Translation Preference Analysis**: 
   - In the overall preference analysis, LCD-MAS emerged as the most preferred translation approach, receiving 52 votes (43.33%), followed by PD-MAS with 39 votes (32.50%) and human translations with 29 votes (24.17%).
   - Raters indicated that fluency had a stronger influence on overall preference than accuracy.

The results of this study demonstrate that LLM-based multi-agent systems can achieve accuracy comparable to human translators while potentially surpassing them in fluency and preference in certain aspects. Notably, the LLM-capability-driven system produced translations with enhanced literary quality and stylistic richness, although it occasionally introduced content absent from the source text. In contrast, the practice-derived system generated more concise translations but sometimes lacked cohesive flow.


<br/>
# 예제



이 연구에서는 두 가지 다중 에이전트 시스템(Practice-Derived Multi-Agent System, PD-MAS와 LLM-Capability-Driven Multi-Agent System, LCD-MAS)을 사용하여 문학 번역의 품질을 비교했습니다. 이 시스템들은 각각 다른 방식으로 번역 작업을 수행하며, 그 과정에서 사용되는 입력과 출력의 구체적인 예시를 설명하겠습니다.

#### 1. Practice-Derived Multi-Agent System (PD-MAS)

**입력:**
- 원본 텍스트: 중국어 소설의 특정 장(chapter) (예: "尘埃落定"의 3장)
- 텍스트 분석기: 원본 텍스트의 장르, 주제, 스타일적 특징을 분석하여 번역에 필요한 정보를 제공합니다.
- 용어 전문가: 일관성을 유지하기 위해 이중 언어 용어 목록을 생성합니다.

**과정:**
1. **전처리 단계**: 텍스트 분석기와 용어 전문가가 협력하여 번역에 필요한 자료를 준비합니다.
2. **번역 단계**: 번역가가 원본 텍스트를 기반으로 번역을 수행합니다.
3. **검토 단계**: 번역가가 자가 점검 후, 수정자가 번역의 정확성과 완전성을 비교 분석합니다.
4. **최종 검토 단계**: 리뷰어가 언어적 및 스타일적 일관성을 확인하고, 교정자가 최종 품질 검사를 수행합니다.

**출력:**
- 번역된 텍스트: 원본 텍스트의 의미를 충실히 전달하며, 문법적으로 올바르고 자연스러운 영어로 작성된 번역 결과.

#### 2. LLM-Capability-Driven Multi-Agent System (LCD-MAS)

**입력:**
- 원본 텍스트: 동일한 중국어 소설의 특정 장
- 요약 생성기: 원본 텍스트의 주요 사건, 등장인물, 주제를 요약합니다.
- 전략 계획자: 독자 기대, 텍스트 유형, 문화적 참조를 고려한 번역 계획을 수립합니다.

**과정:**
1. **전처리 단계**: 요약 생성기와 전략 계획자가 협력하여 번역의 전반적인 방향을 설정합니다.
2. **번역 및 스타일 수정 단계**: 번역가가 요약과 계획을 바탕으로 초기 번역을 수행하고, 수정자가 정확성을 확인합니다. 스타일 가이드 생성기가 적절한 스타일 수정을 제안합니다.
3. **스타일 수정 단계**: 스타일 수정자가 제안된 스타일 수정을 적용하여 문학적 장치를 포함한 번역을 완성합니다.
4. **최종화 단계**: 독립적으로 번역된 세그먼트를 연결하고, 스타일 가이드 생성기가 일관성을 확인합니다.

**출력:**
- 번역된 텍스트: 문학적 품질이 높고, 생동감 있는 표현을 포함하며, 원본 텍스트의 의미를 충실히 전달하는 번역 결과.

이 두 시스템의 비교를 통해, PD-MAS는 전통적인 번역 프로세스를 따르며, LCD-MAS는 LLM의 능력을 최대한 활용하여 번역 품질을 향상시키는 데 중점을 두고 있습니다.

---




This study compared the quality of literary translation using two multi-agent systems: the Practice-Derived Multi-Agent System (PD-MAS) and the LLM-Capability-Driven Multi-Agent System (LCD-MAS). Each system operates differently, and here are specific examples of the inputs and outputs involved in their processes.

#### 1. Practice-Derived Multi-Agent System (PD-MAS)

**Input:**
- Source Text: A specific chapter from a Chinese novel (e.g., Chapter 3 of "Dust Settles" - 尘埃落定).
- Text Analyst: Analyzes the genre, themes, and stylistic features of the source text to provide necessary information for translation.
- Terminology Expert: Creates bilingual terminology lists to maintain consistency.

**Process:**
1. **Pre-production Stage**: The text analyst and terminology expert collaborate to prepare resources needed for translation.
2. **Production Stage**: The translator generates the target text based on the analysis and terminology resources.
3. **Review Stage**: The translator performs a self-check, then the reviser conducts a comparative analysis between the source and target texts focusing on accuracy and completeness.
4. **Final Review Stage**: The reviewer ensures linguistic and stylistic coherence, followed by a proofreader performing the final quality check.

**Output:**
- Translated Text: A translation that faithfully conveys the meaning of the source text, written in grammatically correct and natural English.

#### 2. LLM-Capability-Driven Multi-Agent System (LCD-MAS)

**Input:**
- Source Text: The same chapter from the Chinese novel.
- Summarizer: Generates a concise narrative summary capturing main events, characters, and themes.
- Strategy Planner: Develops a comprehensive translation plan addressing audience expectations, text type, and cultural references.

**Process:**
1. **Pre-translation Planning**: The summarizer and strategy planner collaborate to establish the overall direction for the translation.
2. **Translation and Stylistic Rewriting Stage**: The translator produces an initial translation using the summary and strategy plan, followed by a reviser checking for accuracy. The style guide generator identifies appropriate stylistic enhancements.
3. **Stylistic Rewriting Stage**: The stylistic rewriter applies the recommendations, incorporating literary devices while preserving semantic content.
4. **Finalization Stage**: Independently translated segments are concatenated, and the style guide generator detects inconsistencies and awkward transitions.

**Output:**
- Translated Text: A translation characterized by enhanced literary quality and vivid expression, faithfully conveying the meaning of the source text.

By comparing these two systems, it is evident that PD-MAS follows traditional translation processes, while LCD-MAS focuses on maximizing the capabilities of LLMs to improve translation quality.

<br/>
# 요약

이 연구에서는 두 가지 다중 에이전트 번역 시스템(PD-MAS와 LCD-MAS)을 비교하여 문학 번역의 품질을 평가하였다. 결과적으로, LCD-MAS는 유창성에서 인간 번역가보다 높은 점수를 받았고, 두 시스템 모두 정확성에서 유사한 성과를 보였다. 그러나 LCD-MAS는 때때로 원문에 없는 내용을 추가하는 경향이 있어, 번역의 충실성을 저해할 수 있다는 비판을 받았다.

---

This study compared two multi-agent translation systems (PD-MAS and LCD-MAS) to evaluate the quality of literary translation. The results showed that LCD-MAS received higher scores in fluency compared to human translators, while both systems demonstrated similar performance in accuracy. However, LCD-MAS was criticized for occasionally adding content not present in the source text, potentially compromising translation fidelity.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Practice-Derived Multi-Agent Workflow (PD-MAS) 다이어그램**: 이 다이어그램은 PD-MAS의 두 가지 주요 단계인 사전 제작(pre-production)과 제작(production)을 보여줍니다. 사전 제작 단계에서는 텍스트 분석가와 용어 전문가가 각각 소스 텍스트의 특성과 일관성을 위한 용어 목록을 준비합니다. 제작 단계에서는 번역가가 초기 번역을 생성하고, 수정자가 정확성과 완전성을 검토하며, 리뷰어가 언어적 및 스타일적 일관성을 확인합니다. 이 구조는 전통적인 번역 프로세스를 반영하며, LLM 기반 에이전트가 어떻게 작동하는지를 보여줍니다.

2. **LLM-Capability-Driven Multi-Agent Workflow (LCD-MAS) 다이어그램**: LCD-MAS는 세 가지 상호 연결된 단계로 구성됩니다. 사전 번역 계획 단계에서는 요약자와 전략 계획자가 각각 주요 사건과 주제를 요약하고 번역 계획을 수립합니다. 번역 및 스타일 수정 단계에서는 번역가가 초기 번역을 생성하고, 수정자가 정확성을 확인한 후 스타일 가이드 생성기가 스타일적 개선을 제안합니다. 마지막으로, 최종화 단계에서는 독립적으로 번역된 세그먼트를 연결하여 일관성을 유지합니다. 이 구조는 LLM의 계산적 특성을 활용하여 번역 품질을 향상시키는 방법을 보여줍니다.

#### 테이블
1. **표 1: 샘플 길이 요약 통계**: 이 표는 번역된 텍스트의 길이를 비교합니다. PD-MAS 번역은 평균 99 단어, LCD-MAS 번역은 평균 149.5 단어, 인간 번역은 평균 113.4 단어로 나타났습니다. LCD-MAS는 일반적으로 더 긴 번역을 생성했으며, 이는 스타일적 경향을 분석하는 데 중요한 요소로 작용합니다.

2. **표 2: 번역 접근 방식에 대한 평가자 선호도**: 이 표는 120개의 평가에서 각 번역 접근 방식에 대한 선호도를 보여줍니다. LCD-MAS가 52표(43.33%)로 가장 많이 선호되었고, PD-MAS는 39표(32.50%), 인간 번역은 29표(24.17%)로 나타났습니다. 이는 LCD-MAS의 번역이 평가자들에게 더 매력적으로 인식되었음을 시사합니다.

#### 어펜딕스
1. **어펜딕스 A**: 실험에 사용된 텍스트 샘플의 출처를 나열합니다. 이 목록은 다양한 장르의 현대 중국 소설에서 발췌한 28개의 장을 포함하고 있으며, 각 장의 원작자와 번역자, 출판 정보를 제공합니다. 이는 연구의 대표성을 높이는 데 기여합니다.

2. **어펜딕스 B**: 번역 품질 평가를 위한 점수 매기기 기준을 제시합니다. 이 기준은 정확성과 유창성을 평가하는 다섯 가지 수준으로 나뉘며, 각 수준에 대한 설명과 점수 범위를 제공합니다. 이는 평가자들이 번역 품질을 일관되게 평가할 수 있도록 돕습니다.

---




#### Diagrams and Figures
1. **Practice-Derived Multi-Agent Workflow (PD-MAS) Diagram**: This diagram illustrates the two main stages of PD-MAS: pre-production and production. In the pre-production stage, a text analyst and a terminology expert prepare essential resources by analyzing the source text's characteristics and creating bilingual terminology lists for consistency. The production stage encompasses translation and quality assurance, where the translator generates the initial translation, the reviser checks for accuracy and completeness, and the reviewer ensures linguistic and stylistic coherence. This structure reflects traditional translation processes and demonstrates how LLM-based agents operate.

2. **LLM-Capability-Driven Multi-Agent Workflow (LCD-MAS) Diagram**: LCD-MAS consists of three interconnected stages. The pre-translation planning stage establishes global context through a summarizer and a strategy planner, who generate a concise narrative summary and develop a comprehensive translation plan. The translation and stylistic rewriting stage separates semantic transfer from stylistic refinement, where the translator produces an initial translation, the reviser checks for accuracy, and the style guide generator identifies stylistic enhancements. Finally, the finalization stage ensures coherence across independently translated segments. This structure leverages the computational characteristics of LLMs to enhance translation quality.

#### Tables
1. **Table 1: Summary Statistics for Sample Lengths**: This table compares the lengths of translated texts. PD-MAS translations averaged 99 words, LCD-MAS translations averaged 149.5 words, and human translations averaged 113.4 words. LCD-MAS produced generally longer translations, which is a significant factor in analyzing stylistic tendencies.

2. **Table 2: Rater Preferences for Translation Approaches**: This table shows the preferences for each translation approach across 120 evaluations. LCD-MAS received 52 votes (43.33%) as the most preferred, followed by PD-MAS with 39 votes (32.50%) and human translations with 29 votes (24.17%). This suggests that LCD-MAS translations were perceived as more appealing by evaluators.

#### Appendices
1. **Appendix A**: Lists the sources of the text samples used for the experiment. This list includes 28 chapters from contemporary Chinese fiction, providing the original authors and translators, along with publication information. This contributes to the representativeness of the study.

2. **Appendix B**: Presents the scoring rubric for translation quality evaluation. This rubric is divided into five levels assessing accuracy and fluency, with descriptions and score ranges for each level. It helps ensure that evaluators can assess translation quality consistently.

<br/>
# refer format:


```bibtex
@article{Wang2026,
  author = {Wang, Lulu and Sun, Sanjun and Wang, Xing and Gu, Jinghang and Liu, Kanglong},
  title = {Workflow Matters: Comparing Human Translators and Multi-Agent LLMs in Literary Translation},
  journal = {Target},
  volume = {36},
  number = {1},
  pages = {1--30},
  year = {2026},
  publisher = {John Benjamins Publishing Company},
  doi = {10.1075/target.25081.wan},
  url = {https://doi.org/10.1075/target.25081.wan}
}
```




Lulu Wang, Sanjun Sun, Xing Wang, Jinghang Gu, and Kanglong Liu. "Workflow Matters: Comparing Human Translators and Multi-Agent LLMs in Literary Translation." *Target* 36, no. 1 (2026): 1–30. https://doi.org/10.1075/target.25081.wan.
