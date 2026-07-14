---
layout: post
title:  "[2026]Enhancing Mental Health Counseling Support in Bangladesh using Culturally-Grounded Knowledge"
date:   2026-07-14 00:50:31 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 방글라데시의 정신 건강 상담 지원을 개선하기 위해 두 가지 접근 방식(검색 보강 생성(RAG) 및 지식 그래프(KG) 기반 방법)을 사용하여 대규모 언어 모델(LLM)을 평가하였다.


짧은 요약(Abstract) :


이 연구는 방글라데시에서 정신 건강 상담 지원을 향상시키기 위해 문화적으로 기반한 지식을 통합하는 방법을 다룹니다. 대형 언어 모델(LLMs)은 정신 건강 및 상담 응용 프로그램에서 지원적인 응답을 생성하는 데 유망하지만, 종종 문화적 민감성, 맥락적 기반 및 임상적으로 적절한 지침이 부족합니다. 이 연구는 LLM의 상담 품질을 개선하기 위해 도메인 특정의 임상적으로 검증된 지식을 체계적으로 통합하는 방법의 격차를 해결합니다. 우리는 두 가지 접근 방식, 즉 검색 보강 생성(RAG)과 지식 그래프(KG) 기반 방법을 활용하고 비교합니다. KG는 수동으로 구축되고 임상적으로 검증되어 스트레스 요인, 개입 및 결과 간의 인과 관계를 포착합니다. 여러 LLM을 평가한 결과, KG 기반 접근 방식이 RAG 단독보다 맥락적 관련성, 임상 적합성 및 실용성을 일관되게 개선하는 것으로 나타났습니다. 이는 구조화된 전문가 검증 지식이 상담 작업에서 LLM의 한계를 해결하는 데 중요한 역할을 한다는 것을 보여줍니다.




This study addresses the integration of culturally grounded knowledge to enhance mental health counseling support in Bangladesh. Large language models (LLMs) show promise in generating supportive responses for mental health and counseling applications; however, they often lack cultural sensitivity, contextual grounding, and clinically appropriate guidance. This work addresses the gap in systematically incorporating domain-specific, clinically validated knowledge into LLMs to improve counseling quality. We utilize and compare two approaches: retrieval-augmented generation (RAG) and a knowledge graph (KG)–based method. Our KG is manually constructed and clinically validated, capturing causal relationships between stressors, interventions, and outcomes. Evaluations of multiple LLMs show that KG-based approaches consistently improve contextual relevance, clinical appropriateness, and practical usability compared to RAG alone, demonstrating that structured, expert-validated knowledge plays a critical role in addressing LLMs' limitations in counseling tasks.


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



이 연구에서는 방글라데시의 정신 건강 상담 지원을 향상시키기 위해 두 가지 주요 접근 방식을 사용했습니다: **검색 증강 생성(RAG)**와 **지식 그래프(KG) 기반 방법**입니다. 이 두 가지 방법은 대규모 언어 모델(LLM)을 활용하여 상담 품질을 개선하는 데 중점을 두었습니다.

1. **모델 선택**: 연구에서는 여러 언어 모델을 평가했습니다. Gemini, Llama, DeepSeek, Gemma, GPT와 같은 다양한 모델 패밀리에서 여러 변형을 테스트하여 성능 차이를 분석했습니다. 이 과정에서 경량 모델과 대규모 모델을 모두 포함하여, 모델 선택이 RAG 및 KG 기반 실험의 결과에 미치는 영향을 평가했습니다.

2. **데이터셋 구축**: 상담 세션에서 수집된 데이터를 기반으로 한 데이터셋을 구축했습니다. 이 데이터셋은 방글라데시의 저소득 커뮤니티에서 진행된 상담 세션의 구조화된 기록으로, 각 기록은 개인의 인구 통계적 특성, 스트레스 요인, 감정적 반응 및 제공된 개입을 문서화했습니다. 이 데이터는 전문가의 검토를 거쳐 임상적으로 유효한 지식 그래프로 변환되었습니다.

3. **지식 그래프(KG) 구축**: KG는 스트레스 요인, 개입 및 결과 간의 인과 관계를 모델링하기 위해 수작업으로 구축되었습니다. 이 과정에는 경험이 있는 사람들, 공공 건강 연구자, 심리학자 및 정신과 의사와의 공동 설계 워크숍이 포함되었습니다. KG는 상담 내러티브에서의 인과 관계를 명확히 하여, LLM이 보다 해석 가능하고 임상적으로 타당한 추론을 할 수 있도록 지원합니다.

4. **실험 설정**: RAG 기반 실험에서는 주어진 쿼리에 대해 관련 정보를 검색하여 모델의 입력 컨텍스트에 통합했습니다. KG 기반 실험에서는 KG에서 구조화된 지식을 활용하여 응답 생성 과정을 안내했습니다. 이 두 가지 접근 방식을 비교하여 비구조적 검색과 구조적 지식 표현이 생성된 응답의 품질과 관련성에 미치는 영향을 평가했습니다.

5. **평가 메트릭**: 자동 평가를 통해 생성된 응답의 품질을 정량적으로 평가했습니다. BERTScore F1과 SBERT 코사인 유사도를 사용하여 생성된 응답과 참조 텍스트 간의 의미적 유사성을 측정했습니다. 또한, 인간 평가를 통해 상담 품질과 맥락 이해도를 평가했습니다.

이 연구의 결과는 KG 기반 접근 방식이 RAG 단독 사용보다 상담 응답의 맥락적 관련성, 임상 적합성 및 실용성을 일관되게 개선함을 보여주었습니다. 이는 구조화된 전문가 검증 지식이 LLM의 상담 작업에서의 한계를 해결하는 데 중요한 역할을 한다는 것을 강조합니다.

---




In this study, two main approaches were utilized to enhance mental health counseling support in Bangladesh: **Retrieval-Augmented Generation (RAG)** and a **Knowledge Graph (KG)-based method**. These approaches focused on leveraging large language models (LLMs) to improve counseling quality.

1. **Model Selection**: The study evaluated multiple language models, testing various variants from different model families, including Gemini, Llama, DeepSeek, Gemma, and GPT. This included both lightweight and large-scale models to analyze performance differences and assess how model selection impacts the results of RAG and KG-based experiments.

2. **Dataset Construction**: A dataset was constructed based on data collected from counseling sessions. This dataset consists of structured records of counseling sessions conducted in low-income communities in Bangladesh, documenting each participant's demographic characteristics, stressors, emotional responses, and interventions provided. The data was clinically validated and transformed into a knowledge graph.

3. **Knowledge Graph (KG) Construction**: The KG was manually constructed to model causal relationships between stressors, interventions, and outcomes. This process involved co-design workshops with individuals with lived experience, public health researchers, psychologists, and psychiatrists. The KG clarifies causal relationships in counseling narratives, enabling LLMs to perform more interpretable and clinically valid reasoning.

4. **Experimental Settings**: In the RAG-based experiments, relevant information was retrieved and incorporated into the model's input context based on given queries. In the KG-based experiments, structured knowledge from the KG was utilized to guide the response generation process. This comparison allowed for the evaluation of how unstructured retrieval and structured knowledge representations influence the quality and relevance of generated responses.

5. **Evaluation Metrics**: Automatic evaluations were conducted to quantitatively assess the quality of generated responses. BERTScore F1 and SBERT cosine similarity were used to measure semantic similarity between generated responses and reference texts. Additionally, human evaluations were performed to assess counseling quality and contextual understanding.

The results of this study demonstrated that KG-based approaches consistently improve contextual relevance, clinical appropriateness, and practical usability of counseling responses compared to RAG alone. This highlights the critical role of structured, expert-validated knowledge in addressing the limitations of LLMs in counseling tasks.


<br/>
# Results



이 연구에서는 두 가지 접근 방식, 즉 검색 증강 생성(RAG)과 지식 그래프(KG) 기반 방법을 사용하여 방글라데시의 정신 건강 상담 지원을 개선하기 위해 여러 대형 언어 모델(LLMs)을 평가했습니다. 연구의 주요 목표는 LLM이 제공하는 상담 지원의 질을 향상시키기 위해 문화적으로 적합하고 임상적으로 검증된 지식을 통합하는 것이었습니다.

#### 경쟁 모델
연구에서는 Gemini, Llama, DeepSeek, Gemma, OpenAI의 여러 모델을 포함하여 총 17개의 후보 모델을 평가했습니다. 이들 모델은 다양한 아키텍처와 크기를 포함하여 성능 차이를 분석할 수 있도록 설계되었습니다.

#### 테스트 데이터
테스트 데이터는 Sajida Foundation의 커뮤니티 정신 건강 이니셔티브의 상담 세션에서 수집된 402개의 참가자 기록으로 구성되었습니다. 이 데이터는 참가자의 인구 통계적 특성, 스트레스 요인, 감정적 반응 및 제공된 개입을 포함하여 실제 상담 시나리오를 반영합니다.

#### 메트릭
모델의 성능은 BERTScore F1과 SBERT 코사인 유사도를 사용하여 자동 평가되었으며, 인간 평가를 통해 상담 품질과 맥락 이해도를 평가했습니다. BERTScore F1은 생성된 응답의 내용 품질을 측정하고, SBERT 코사인 유사도는 생성된 응답과 참조 응답 간의 의미적 유사성을 평가합니다.

#### 비교 결과
자동 평가 결과, BERTScore F1 값은 84.95에서 86.84 사이로 나타났으며, SBERT 코사인 유사도는 69.31에서 86.22 사이로 더 넓은 범위를 보였습니다. GPT-4.1 모델이 SBERT 코사인 유사도에서 가장 높은 점수를 기록했으며, Llama-3.3-70B 모델이 BERTScore F1에서 가장 높은 점수를 기록했습니다.

인간 평가에서는 Llama-3.3-70B와 GPT-4.1 모델이 평균적으로 가장 높은 점수를 기록했으며, 이 두 모델은 각각 2.3의 평균 점수를 기록했습니다. KG 기반 접근 방식이 RAG 기반 모델에 비해 모든 모델에서 BERTScore F1을 향상시켰으며, 평균 인간 점수도 감소했습니다. 이는 KG가 상담 지원의 질을 향상시키는 데 중요한 역할을 했음을 나타냅니다.

결론적으로, 연구 결과는 구조화된 임상 지식의 통합이 상담 지원의 질을 향상시키는 데 필수적임을 보여주었으며, 향후 연구는 더 넓은 맥락 요인을 포함하고 가족, 개인 및 경제적 맥락을 더 잘 표현하는 데 초점을 맞출 필요가 있음을 강조했습니다.

---




This study evaluated several large language models (LLMs) using two approaches: Retrieval-Augmented Generation (RAG) and Knowledge Graph (KG)-based methods, aiming to enhance mental health counseling support in Bangladesh. The primary goal of the research was to systematically incorporate culturally appropriate and clinically validated knowledge into LLMs to improve the quality of counseling support.

#### Competing Models
The study assessed a total of 17 candidate models, including those from Gemini, Llama, DeepSeek, Gemma, and OpenAI. These models encompassed various architectures and sizes to allow for a comprehensive analysis of performance differences.

#### Test Data
The test data consisted of 402 participant records collected from counseling sessions conducted as part of the Sajida Foundation's Community Mental Health Initiative. This data reflects real-world counseling scenarios, including participants' demographic characteristics, stressors, emotional responses, and the interventions provided.

#### Metrics
The performance of the models was evaluated using BERTScore F1 and SBERT cosine similarity for automatic evaluation, along with human evaluation to assess the quality of counseling and contextual understanding. BERTScore F1 measures the content quality of generated responses, while SBERT cosine similarity assesses the semantic similarity between generated and reference responses.

#### Comparison Results
The automated evaluation results showed that BERTScore F1 values ranged from 84.95 to 86.84, while SBERT cosine similarity exhibited a wider range from 69.31 to 86.22. The GPT-4.1 model achieved the highest score in SBERT cosine similarity, while the Llama-3.3-70B model recorded the highest BERTScore F1.

In human evaluations, both Llama-3.3-70B and GPT-4.1 models scored the highest on average, with both achieving an average score of 2.3. The KG-based approach consistently improved BERTScore F1 across all models compared to RAG-based models, and the average human scores decreased, indicating that KG played a critical role in enhancing the quality of counseling support.

In conclusion, the findings demonstrate that the integration of structured clinical knowledge is essential for improving the quality of counseling support, and future research should focus on incorporating broader contextual factors and better representing familial, personal, and economic contexts.


<br/>
# 예제


이 연구에서는 방글라데시의 정신 건강 상담 지원을 향상시키기 위해 두 가지 접근 방식을 사용하여 대규모 언어 모델(LLM)을 평가했습니다: 검색 증강 생성(RAG)과 지식 그래프(KG) 기반 방법입니다. 이 과정에서 사용된 데이터셋은 실제 상담 세션에서 수집된 사례 기록으로 구성되어 있으며, 각 기록은 개인의 인구 통계적 특성, 스트레스의 원인, 정서적 반응 및 상담 세션에서 제공된 개입을 문서화합니다.

#### 데이터셋 구성
1. **트레이닝 데이터**: 
   - **입력**: 상담 세션의 요약, 참가자의 문제, 가족 및 경제적 배경 등.
   - **출력**: 상담자가 제공해야 할 적절한 개입 및 조언.
   - 예시: 
     - 입력: "30세 여성, 경제적 어려움으로 인한 우울증 증세. 가족과의 갈등이 심화됨."
     - 출력: "예산 관리 기법을 논의하고, 가족과의 의사소통 개선을 위한 전략을 제안합니다."

2. **테스트 데이터**: 
   - **입력**: 새로운 상담 세션의 요약 및 참가자의 문제.
   - **출력**: 모델이 생성한 상담 응답.
   - 예시: 
     - 입력: "25세 남성, 직장 스트레스로 인한 불안 증세."
     - 출력: "스트레스 관리 기법을 소개하고, 필요시 전문가에게 의뢰하는 방법을 안내합니다."

#### 구체적인 태스크
- **RAG 기반 실험**: 
  - 모델은 입력된 상담 세션 요약을 바탕으로 관련된 과거 사례를 검색하여 응답을 생성합니다.
  - 예를 들어, "경제적 스트레스"라는 키워드로 과거 사례를 검색하고, 그 사례에서 유용한 개입을 추출하여 응답을 생성합니다.

- **KG 기반 실험**: 
  - 모델은 지식 그래프를 활용하여 입력된 문제와 관련된 원인, 개입 및 결과 간의 관계를 분석합니다.
  - 예를 들어, "경제적 스트레스"가 "수면 부족"으로 이어질 수 있다는 관계를 기반으로, 수면 개선을 위한 개입을 제안합니다.

이러한 방식으로, 연구는 LLM이 방글라데시의 저소득 커뮤니티에서 문화적으로 민감한 상담 지원을 제공하는 데 어떻게 기여할 수 있는지를 평가합니다.

---




In this study, two approaches were used to evaluate large language models (LLMs) to enhance mental health counseling support in Bangladesh: Retrieval-Augmented Generation (RAG) and Knowledge Graph (KG)-based methods. The dataset used in this process consists of case records collected from actual counseling sessions, where each record documents the individual's demographic characteristics, causes of stress, emotional responses, and interventions provided during counseling sessions.

#### Dataset Composition
1. **Training Data**: 
   - **Input**: Summaries of counseling sessions, participant problems, family and economic backgrounds, etc.
   - **Output**: Appropriate interventions and advice that the counselor should provide.
   - Example: 
     - Input: "30-year-old female, showing symptoms of depression due to financial difficulties. Increasing conflict with family."
     - Output: "Discuss budgeting techniques and suggest strategies for improving communication with family."

2. **Test Data**: 
   - **Input**: Summaries of new counseling sessions and participant problems.
   - **Output**: Counseling responses generated by the model.
   - Example: 
     - Input: "25-year-old male, experiencing anxiety due to workplace stress."
     - Output: "Introduce stress management techniques and guide on how to refer to a professional if necessary."

#### Specific Tasks
- **RAG-based Experiments**: 
  - The model generates responses by searching for relevant past cases based on the input summary of the counseling session.
  - For instance, it might search for past cases using the keyword "economic stress" and extract useful interventions from those cases to generate a response.

- **KG-based Experiments**: 
  - The model utilizes a knowledge graph to analyze the relationships between causes, interventions, and outcomes related to the input problem.
  - For example, it might suggest interventions for improving sleep based on the relationship that "economic stress" can lead to "sleep deprivation."

Through these methods, the study assesses how LLMs can contribute to providing culturally sensitive counseling support in low-income communities in Bangladesh.

<br/>
# 요약

이 연구에서는 방글라데시의 정신 건강 상담 지원을 개선하기 위해 두 가지 접근 방식(검색 보강 생성(RAG) 및 지식 그래프(KG) 기반 방법)을 사용하여 대규모 언어 모델(LLM)을 평가하였다. KG 기반 접근 방식이 RAG보다 일관되게 더 나은 상담 품질을 제공하며, 문화적 민감성과 임상 적합성을 향상시키는 데 중요한 역할을 한다는 결과를 도출하였다. 연구 결과는 LLM이 상담 지원에 있어 문화적 맥락을 반영하는 것이 중요하다는 점을 강조한다.

---

This study evaluated two approaches (retrieval-augmented generation (RAG) and knowledge graph (KG)-based methods) to improve mental health counseling support in Bangladesh using large language models (LLMs). The results showed that KG-based approaches consistently provided better counseling quality than RAG alone, demonstrating the critical role of structured, culturally sensitive knowledge in enhancing contextual relevance and clinical appropriateness. The findings emphasize the importance of reflecting cultural context in LLMs for effective counseling support.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 연구 방법론의 개요를 보여주며, RAG 기반 실험과 KG 기반 실험의 구조를 설명합니다. 이는 LLM이 어떻게 도메인 특화 지식과 결합되어 상담 지원을 제공하는지를 시각적으로 나타냅니다.
   - **Figure 2**: 지식 그래프의 서브그래프를 보여주며, 빈곤의 원인, 문제 상태 및 개입 간의 관계를 시각화합니다. 이는 상담 과정에서의 인과 관계를 명확히 하고, LLM이 보다 구조화된 지식을 활용할 수 있도록 돕습니다.
   - **Figure 3**: 자동화된 성능 평가 결과를 보여주며, 다양한 모델 간의 BERTScore F1과 SBERT 코사인 유사성을 비교합니다. 이 결과는 모델 선택이 상담 지원의 질에 미치는 영향을 강조합니다.
   - **Figure 4**: RAG와 KG 기반 모델의 성능 비교를 보여주며, KG 기반 모델이 BERTScore F1과 인간 평가에서 일관되게 개선된 결과를 나타냅니다. 이는 구조화된 지식이 상담 지원의 질을 향상시키는 데 중요한 역할을 한다는 것을 시사합니다.
   - **Figure 5**: KG 기반 모델의 카테고리별 인간 평가 점수를 보여주며, KG 기반 모델이 모든 평가 카테고리에서 평균 점수를 낮추는 경향을 보입니다. 이는 KG가 상담 지원의 질을 향상시키는 데 기여함을 나타냅니다.

2. **테이블**
   - **Table 1**: 지식 그래프의 관계 분포를 보여주며, 다양한 인과 관계와 개입 간의 연결을 정리합니다. 이는 상담 과정에서의 복잡한 관계를 이해하는 데 도움을 줍니다.
   - **Table 4**: 인간 평가 결과를 요약하여 각 모델의 성능을 비교합니다. Llama-3.3-70B와 GPT-4.1이 가장 높은 평균 점수를 기록하며, 이는 이 모델들이 상담 지원에 있어 더 나은 성능을 발휘함을 나타냅니다.
   - **Table 5**: RAG 기반 모델과 KG 기반 모델의 성능 비교를 보여주며, KG 기반 모델이 모든 모델에서 BERTScore F1을 개선한 것을 나타냅니다. 이는 KG가 상담 지원의 질을 높이는 데 기여함을 시사합니다.
   - **Table 6**: 자동화된 평가 결과를 보여주며, 다양한 모델의 BERTScore F1과 SBERT 점수를 비교합니다. 이는 모델 선택이 상담 지원의 질에 미치는 영향을 강조합니다.
   - **Table 7**: KG 기반 모델의 카테고리별 인간 평가 점수를 보여주며, KG가 상담 지원의 질을 향상시키는 데 기여함을 나타냅니다.

3. **어펜딕스**
   - **Appendix A**: 상담 세션 진행 가이드라인을 제공하며, 각 세션의 목표와 절차를 설명합니다. 이는 상담 과정의 구조를 명확히 하고, 상담자와 참여자 간의 신뢰 구축을 돕습니다.
   - **Appendix A.2**: 주석 가이드라인을 제공하여 세션 내러티브의 적합성 및 데이터 프라이버시 검토 절차를 설명합니다. 이는 데이터의 정확성과 임상적 유효성을 보장하는 데 기여합니다.

---

### Summary of Results and Insights from Diagrams, Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Figure 1**: Provides an overview of the research methodology, illustrating the structure of RAG-based and KG-based experiments. It visually represents how LLMs integrate domain-specific knowledge to provide counseling support.
   - **Figure 2**: Displays a subgraph of the knowledge graph, visualizing the relationships between poverty drivers, problem states, and interventions. This clarifies the causal relationships in the counseling process, enabling LLMs to utilize more structured knowledge.
   - **Figure 3**: Shows the results of automated performance evaluations, comparing BERTScore F1 and SBERT cosine similarity across various models. This highlights the impact of model selection on the quality of counseling support.
   - **Figure 4**: Compares the performance of RAG and KG-based models, indicating that KG-based models consistently improve BERTScore F1 and human evaluations. This suggests that structured knowledge plays a critical role in enhancing the quality of counseling support.
   - **Figure 5**: Displays category-level human evaluation scores for KG-based models, showing a trend of lower average scores after grounding with KG. This indicates that KG contributes to improving the quality of counseling support.

2. **Tables**
   - **Table 1**: Shows the distribution of relationships in the knowledge graph, summarizing various causal relationships and connections between interventions. This aids in understanding the complex relationships in the counseling process.
   - **Table 4**: Summarizes human evaluation results, comparing the performance of each model. Llama-3.3-70B and GPT-4.1 achieve the highest average scores, indicating their superior performance in counseling support.
   - **Table 5**: Compares the performance of RAG-based and KG-based models, showing that KG-based models improve BERTScore F1 across all models. This suggests that KG contributes to enhancing the quality of counseling support.
   - **Table 6**: Displays automated evaluation results, comparing BERTScore F1 and SBERT scores across various models. This emphasizes the impact of model selection on the quality of counseling support.
   - **Table 7**: Shows category-level human evaluation scores for KG-based models, indicating that KG contributes to improving the quality of counseling support.

3. **Appendices**
   - **Appendix A**: Provides guidelines for conducting counseling sessions, outlining the goals and procedures for each session. This clarifies the structure of the counseling process and helps build trust between counselors and participants.
   - **Appendix A.2**: Offers annotation guidelines detailing the appropriateness and data privacy review process for session narratives. This contributes to ensuring the accuracy and clinical validity of the data.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{hasan2026enhancing,
  author    = {Md Arid Hasan and Azhagu Meena SP and Aditya Khan and Abu Md Akteruzzaman Bhuiyan and Helal Uddin Ahmed and Joysree Debi and Farig Sadeque and Annie En-Shiun Lee and Syed Ishtiaque Ahmed},
  title     = {Enhancing Mental Health Counseling Support in Bangladesh using Culturally-Grounded Knowledge},
  booktitle = {Proceedings of the 10th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026)},
  pages     = {164--177},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},
}
```

### Chicago Style Citation

Md Arid Hasan, Azhagu Meena SP, Aditya Khan, Abu Md Akteruzzaman Bhuiyan, Helal Uddin Ahmed, Joysree Debi, Farig Sadeque, Annie En-Shiun Lee, and Syed Ishtiaque Ahmed. "Enhancing Mental Health Counseling Support in Bangladesh using Culturally-Grounded Knowledge." In *Proceedings of the 10th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026)*, 164–177. Association for Computational Linguistics, July 2026.
    