---
layout: post
title:  "[2026]MED-COREASONER: Reducing Language Disparities in Medical Reasoning via Language-Informed Co-Reasoning"
date:   2026-07-14 18:54:01 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: MED-COREASONER라는 언어 정보 기반의 공동 추론 프레임워크를 제안합니다. 이 프레임워크는 영어와 지역 언어의 병렬 추론을 유도하고, 이를 구조화된 개념으로 추상화한 후, 지역 임상 지식을 영어 논리 구조에 통합  


짧은 요약(Abstract) :



이 논문에서는 영어 의료 작업에서 강력한 성능을 보이는 추론 강화 대형 언어 모델(LLM)이 다국어 환경에서는 지속적인 격차가 존재함을 지적합니다. 특히, 지역 언어에서의 추론 성능이 상당히 약해져 공정한 글로벌 의료 배포를 제한하고 있습니다. 이를 해결하기 위해, MED-COREASONER라는 언어 정보 기반의 공동 추론 프레임워크를 제안합니다. 이 프레임워크는 영어와 지역 언어의 병렬 추론을 유도하고, 이를 구조화된 개념으로 추상화한 후, 지역 임상 지식을 영어 논리 구조에 통합합니다. 실험 결과, MED-COREASONER는 다국어 추론 성능을 평균 5% 향상시키며, 특히 자원이 부족한 언어에서 큰 성과를 보였습니다. 또한, 모델 증류 및 전문가 평가 분석을 통해 MED-COREASONER가 임상적으로 타당하고 문화적으로 적합한 추론 흔적을 생성함을 확인했습니다.




This paper highlights the persistent multilingual gap in reasoning capabilities of reasoning-enhanced large language models (LLMs) that perform strongly on English medical tasks. Specifically, there is significantly weaker reasoning in local languages, which limits equitable global medical deployment. To address this gap, we introduce MED-COREASONER, a language-informed co-reasoning framework that elicits parallel reasoning in English and local languages, abstracts them into structured concepts, and integrates local clinical knowledge into an English logical scaffold. Experimental results show that MED-COREASONER improves multilingual reasoning performance by an average of 5%, with particularly substantial gains in low-resource languages. Furthermore, model distillation and expert evaluation analysis confirm that MED-COREASONER produces clinically sound and culturally grounded reasoning traces.


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



**메써드: MED-COREASONER**

MED-COREASONER는 의료 분야에서의 다국어 추론 격차를 줄이기 위해 설계된 언어 정보 기반의 공동 추론 프레임워크입니다. 이 프레임워크는 영어와 지역 언어에서의 병렬 추론을 유도하고, 이를 구조화된 개념으로 추상화한 후, 지역 임상 지식을 영어의 논리적 스캐폴드에 통합하는 방식으로 작동합니다. MED-COREASONER는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **병렬 추론 생성**: 주어진 질문에 대해 영어와 지역 언어 각각에서 독립적인 추론 경로를 생성합니다. 이 과정에서 각 언어의 자연스러운 추론 경로를 따르도록 하여, 서로의 편향 없이 다양한 관점을 수집합니다.

2. **개념 체인 추출**: 생성된 추론 경로에서 핵심 의료 개념을 추출하여, 이를 정렬된 개념 체인으로 변환합니다. 이 단계는 서로 다른 언어 간의 정밀한 매핑과 융합을 가능하게 합니다.

3. **교차 언어 개념 융합**: 영어 개념 체인을 백본으로 삼고, 지역 언어 개념을 보완적으로 통합하여 일관된 의미를 유지합니다. 이 과정은 논리적 일관성과 임상적 맥락을 보장합니다.

4. **지식 검색**: 융합된 개념 체인은 최종 답변 생성을 위한 구조적 뼈대를 제공합니다. 이 단계에서는 지역적 및 언어적 맥락에 맞는 권위 있는 다국어 의료 지침을 기반으로 추가 정보를 검색하여, 최종 응답의 신뢰성을 높입니다.

5. **최종 답변 생성**: 질문, 개념 체인, 검색된 문서를 바탕으로 최종 응답을 생성합니다. 이 과정에서 개념 체인은 추론의 경로를 제공하고, 검색된 문서는 필요한 실증적 근거를 제공합니다.

이러한 방법론을 통해 MED-COREASONER는 특히 저자원 언어에서의 의료 추론 성능을 향상시키며, 임상적 정확성과 문화적 적합성을 동시에 달성하는 것을 목표로 합니다.



**Method: MED-COREASONER**

MED-COREASONER is a language-informed co-reasoning framework designed to reduce the multilingual reasoning gap in the medical domain. This framework operates by eliciting parallel reasoning in English and local languages, abstracting them into structured concepts, and integrating local clinical knowledge into an English logical scaffold. MED-COREASONER consists of the following key components:

1. **Parallel Reasoning Generation**: For a given question, independent reasoning paths are generated in both English and the local language. This process ensures that each chain follows its natural reasoning path without bias from the other language, allowing for diverse perspectives to be reconciled.

2. **Concept Chain Extraction**: Key medical concepts are extracted from the generated reasoning paths and transformed into an ordered concept chain. This step enables precise mapping and fusion across different languages.

3. **Cross-Lingual Concept Fusion**: The English concept chain serves as a backbone, and local language concepts are integrated complementarily to maintain logical consistency and semantic coherence.

4. **Knowledge Retrieval**: The fused concept chain serves as the structural backbone for generating the final answer. In this phase, authoritative multilingual medical guidelines are retrieved based on regional and linguistic contexts to enhance the reliability of the reasoning process.

5. **Final Answer Generation**: Guided by the original question, the fused concept chain, and the retrieved documents, the model synthesizes a response. In this stage, the concept chain provides the reasoning trajectory, while the retrieved documents offer the necessary empirical grounding.

Through this methodology, MED-COREASONER aims to improve medical reasoning performance, particularly in low-resource languages, while achieving both clinical accuracy and cultural relevance.


<br/>
# Results



이 논문에서는 MED-COREASONER라는 다국어 의료 추론 프레임워크를 소개하고, 이를 통해 영어와 지역 언어 간의 추론 격차를 줄이는 방법을 제안합니다. MED-COREASONER는 영어와 지역 언어에서의 병렬 추론을 유도하고, 이를 구조화된 개념으로 추상화하여 영어의 논리적 구조를 기반으로 지역 임상 지식을 통합합니다. 이를 통해 영어의 구조적 강인성과 지역 언어의 실무 기반 전문성을 결합하여 의료 AI의 공정한 배포를 목표로 합니다.

#### 실험 결과
MED-COREASONER의 성능을 평가하기 위해 MultiMed-X라는 새로운 벤치마크를 구축하였으며, 이는 7개 언어로 구성된 장문 질문 응답 및 자연어 추론 작업을 포함합니다. 실험 결과, MED-COREASONER는 평균 5%의 다국어 추론 성능 향상을 보였으며, 특히 자원이 부족한 언어에서 상당한 개선을 나타냈습니다. 모델 증류 및 전문가 평가 분석을 통해 MED-COREASONER가 임상적으로 타당하고 문화적으로 적합한 추론 흔적을 생성한다는 것을 확인했습니다.

#### 경쟁 모델
MED-COREASONER는 여러 경쟁 모델과 비교되었습니다. 예를 들어, GPT-4o, GPT-5.1, DeepSeek-3.2와 같은 모델들이 포함되었습니다. 실험 결과, MED-COREASONER는 Global-MMLU 및 MMLU-ProX 벤치마크에서 우수한 성능을 보였으며, 특히 자원이 부족한 언어에서 더 큰 성과를 달성했습니다.

#### 테스트 데이터 및 메트릭
테스트 데이터는 Global-MMLU와 MMLU-ProX의 의료 하위 집합을 포함하며, 각 언어에 대해 1,505개 및 687개의 항목이 포함되어 있습니다. 평가 메트릭으로는 정확도, 완전성, 안전성, 환각률 등이 사용되었습니다. MultiMed-X에서는 LFQA(장문 질문 응답)와 NLI(자연어 추론) 작업을 포함하여 다양한 작업에서 MED-COREASONER의 성능을 평가했습니다.

### English Version

This paper introduces MED-COREASONER, a multilingual medical reasoning framework aimed at reducing the reasoning gap between English and local languages. MED-COREASONER elicits parallel reasoning in both English and local languages, abstracts them into structured concepts, and integrates local clinical knowledge into an English logical scaffold. This design combines the structural robustness of English reasoning with the practice-grounded expertise encoded in local languages, aiming for equitable global deployment of medical AI.

#### Experimental Results
To evaluate the performance of MED-COREASONER, a new benchmark called MultiMed-X was constructed, covering long-form question answering and natural language inference tasks in seven languages. The experimental results show that MED-COREASONER improves multilingual reasoning performance by an average of 5%, with particularly substantial gains in low-resource languages. Model distillation and expert evaluation analysis further confirm that MED-COREASONER produces clinically sound and culturally grounded reasoning traces.

#### Competing Models
MED-COREASONER was compared against several competing models, including GPT-4o, GPT-5.1, and DeepSeek-3.2. The results indicated that MED-COREASONER outperformed these models on the Global-MMLU and MMLU-ProX benchmarks, achieving particularly significant improvements in low-resource languages.

#### Test Data and Metrics
The test data includes medical subsets of Global-MMLU and MMLU-ProX, comprising 1,505 and 687 items per language, respectively. Evaluation metrics included accuracy, completeness, safety, and hallucination rates. MultiMed-X also incorporated LFQA (long-form question answering) and NLI (natural language inference) tasks to assess the performance of MED-COREASONER across diverse tasks.


<br/>
# 예제



**예시: 트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋, 구체적인 테스크 설명**

1. **트레이닝 데이터**
   - **인풋**: 의료 질문과 해당 질문에 대한 정답 및 이유가 포함된 데이터셋. 예를 들어, "환자가 3주 전에 동상으로 병원에 입원했습니다. 현재 상태는 어떠한가요?"라는 질문이 있을 수 있습니다.
   - **아웃풋**: 질문에 대한 정답과 함께, 그 정답에 도달하기 위한 단계별 이유가 포함됩니다. 예를 들어, "환자는 동상으로 인해 조직 괴사가 발생했으며, 이는 감염의 위험이 있습니다. 따라서 항생제 치료가 필요합니다."와 같은 형태입니다.

2. **테스트 데이터**
   - **인풋**: 새로운 의료 질문이 주어집니다. 예를 들어, "환자가 발열과 혼란 상태를 보이고 있습니다. 어떤 치료가 필요할까요?"라는 질문이 있을 수 있습니다.
   - **아웃풋**: 모델이 생성한 답변과 그에 대한 이유가 포함됩니다. 예를 들어, "이 환자는 감염의 징후가 있으며, 즉각적인 항생제 치료가 필요합니다."와 같은 형태입니다.

3. **구체적인 테스크**
   - **질문 응답**: 주어진 의료 질문에 대해 적절한 답변을 생성하는 작업입니다.
   - **이유 생성**: 답변을 뒷받침하는 논리적이고 단계적인 이유를 생성하는 작업입니다.
   - **다국어 처리**: 다양한 언어로 질문과 답변을 처리할 수 있도록 하는 작업입니다.




**Example: Detailed Explanation of Training Data and Test Data Inputs and Outputs, Specific Tasks**

1. **Training Data**
   - **Input**: A dataset containing medical questions along with their answers and reasoning. For example, a question might be, "A patient was admitted to the hospital with frostbite three weeks ago. What is their current condition?"
   - **Output**: The answer to the question along with a step-by-step reasoning process leading to that answer. For instance, "The patient has developed tissue necrosis due to frostbite, which poses a risk of infection. Therefore, antibiotic treatment is necessary."

2. **Test Data**
   - **Input**: A new medical question is presented. For example, "The patient is showing signs of fever and confusion. What treatment is needed?"
   - **Output**: The model generates an answer and the reasoning behind it. For example, "This patient shows signs of infection, and immediate antibiotic treatment is required."

3. **Specific Tasks**
   - **Question Answering**: The task of generating appropriate answers to given medical questions.
   - **Reasoning Generation**: The task of producing logical and step-by-step reasoning that supports the answer.
   - **Multilingual Processing**: The task of handling questions and answers in various languages.

<br/>
# 요약


MED-COREASONER는 영어와 지역 언어의 병행 추론을 통해 의료 추론의 언어 격차를 줄이는 프레임워크로, MultiMed-X라는 다국어 의료 추론 벤치마크를 도입하여 7개 언어에서의 성능을 평가하였다. 실험 결과, MED-COREASONER는 특히 자원이 부족한 언어에서 평균 5%의 성능 향상을 보였으며, 임상적으로 타당하고 문화적으로 적합한 추론을 생성하는 것으로 확인되었다. 이 연구는 영어 중심의 추론과 지역 언어의 임상 지식을 통합하여 의료 AI의 공정한 배포를 촉진하는 데 기여한다.



MED-COREASONER is a framework that reduces language disparities in medical reasoning through parallel reasoning in English and local languages, introducing the MultiMed-X benchmark to evaluate performance across seven languages. Experimental results show that MED-COREASONER improves performance by an average of 5%, particularly in low-resource languages, while producing clinically sound and culturally grounded reasoning. This research contributes to promoting equitable deployment of medical AI by integrating English-centric reasoning with local clinical knowledge.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **MED-COREASONER 프레임워크 다이어그램**: 이 다이어그램은 MED-COREASONER의 작동 방식을 시각적으로 설명합니다. 사용자의 입력을 영어로 번역한 후, 영어와 로컬 언어에서 독립적으로 병렬 추론을 수행하고, 이를 개념 체인으로 추출하여 융합하는 과정을 보여줍니다. 이 구조는 영어의 논리적 뼈대와 로컬 언어의 문화적 세부사항을 결합하여 최종 출력을 생성합니다.
   - **성능 비교 그래프**: MultiMed-X 벤치마크에서 MED-COREASONER의 성능을 다른 모델과 비교한 그래프가 포함되어 있습니다. 이 그래프는 MED-COREASONER가 특히 저자원 언어에서 성능 향상을 보여주며, 영어와 비영어 간의 성능 격차를 줄이는 데 효과적임을 나타냅니다.

2. **테이블**:
   - **Global-MMLU 및 MMLU-ProX 결과 테이블**: 이 테이블은 다양한 언어에서 MED-COREASONER의 성능을 다른 모델과 비교한 결과를 보여줍니다. MED-COREASONER는 평균적으로 5%의 성능 향상을 보였으며, 특히 저자원 언어에서 두드러진 개선을 나타냈습니다.
   - **MultiMed-X 평가 결과 테이블**: 이 테이블은 MultiMed-X에서의 MED-COREASONER의 성능을 다양한 평가 지표(정확성, 완전성, 안전성 등)로 나누어 보여줍니다. MED-COREASONER는 모든 언어에서 높은 점수를 기록하며, 특히 완전성 점수에서 큰 향상을 보였습니다.

3. **어펜딕스**:
   - **실험 설정 및 평가 메트릭스**: 어펜딕스에서는 실험의 설정, 사용된 평가 메트릭스, 그리고 각 모델의 세부 사항을 설명합니다. MED-COREASONER는 다양한 언어 모델을 기반으로 하여 평가되었으며, 각 모델의 성능을 비교하는 데 필요한 정보를 제공합니다.
   - **전문가 평가 결과**: 전문가들이 MED-COREASONER와 다른 모델의 추론 품질을 비교한 결과가 포함되어 있습니다. 이 평가에서는 MED-COREASONER가 특히 지역적 임상 관행에 대한 적합성과 명확성에서 우수한 성과를 보였음을 나타냅니다.

### Insights from Diagrams, Figures, Tables, and Appendices

1. **Diagrams and Figures**:
   - **MED-COREASONER Framework Diagram**: This diagram visually explains how MED-COREASONER operates. It shows the process of translating user input into English, performing parallel reasoning in both English and local languages, and extracting and fusing these into a concept chain. This structure combines the logical backbone of English with the cultural specifics of local languages to generate the final output.
   - **Performance Comparison Graph**: A graph comparing the performance of MED-COREASONER against other models on the MultiMed-X benchmark. It illustrates that MED-COREASONER shows significant performance improvements, especially in low-resource languages, effectively narrowing the performance gap between English and non-English languages.

2. **Tables**:
   - **Global-MMLU and MMLU-ProX Results Table**: This table presents the performance of MED-COREASONER across various languages compared to other models. It shows an average improvement of 5% in performance, with particularly notable gains in low-resource languages.
   - **MultiMed-X Evaluation Results Table**: This table breaks down the performance of MED-COREASONER on the MultiMed-X benchmark across different evaluation metrics (accuracy, completeness, safety, etc.). MED-COREASONER achieved high scores across all languages, with significant improvements in completeness.

3. **Appendices**:
   - **Experimental Setup and Evaluation Metrics**: The appendices detail the experimental setup, evaluation metrics used, and specifics of each model. MED-COREASONER was evaluated based on various language models, providing necessary information for comparing performance.
   - **Expert Evaluation Results**: Included are the results of expert evaluations comparing the reasoning quality of MED-COREASONER and other models. This evaluation highlights that MED-COREASONER excels in clarity and localization, demonstrating its effectiveness in producing culturally grounded outputs.

These insights collectively emphasize the effectiveness of MED-COREASONER in enhancing multilingual medical reasoning, particularly in low-resource settings, while maintaining clinical relevance and safety.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Gao2026,
  author = {Fan Gao and Sherry T. Tong and Jiwoong Sohn and Jiahao Huang and Junfeng Jiang and Ding Xia and Piyalitt Ittichaiwong and Kanyakorn Veerakanjana and Hyunjae Kim and Qingyu Chen and Edison Marrese Taylor and Kazuma Kobayashi and Akiko Aizawa and Irene Li},
  title = {MED-COREASONER: Reducing Language Disparities in Medical Reasoning via Language-Informed Co-Reasoning},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {24868--24888},
  year = {2026},
  month = {July},
  publisher = {Association for Computational Linguistics},
  
  
}
```

### 시카고 스타일

Gao, Fan, Sherry T. Tong, Jiwoong Sohn, Jiahao Huang, Junfeng Jiang, Ding Xia, Piyalitt Ittichaiwong, Kanyakorn Veerakanjana, Hyunjae Kim, Qingyu Chen, Edison Marrese Taylor, Kazuma Kobayashi, Akiko Aizawa, and Irene Li. "MED-COREASONER: Reducing Language Disparities in Medical Reasoning via Language-Informed Co-Reasoning." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 24868–24888. Association for Computational Linguistics, 2026.
