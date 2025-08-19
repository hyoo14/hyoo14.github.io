---
layout: post
title:  "[2025]The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering"
date:   2025-08-19 03:10:39 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 


대형 언어 모델(LLM)과 지식 그래프(KG)를 통합하여 환각 현상을 완화하는 방법을 탐구(특히 작은 LLM으로)    

이 논문에서는 작은 언어 모델(SLM)이 지식 그래프(KG)를 활용한 질문 응답에서 탐색 단계의 한계로 인해 성능이 저하된다는 점을 강조   

경량 탐색 모듈을 도입한 SLMs는 기존의 ToG와 CoT(Chain-of-Thought) 방법보다 성능이 향상  




짧은 요약(Abstract) :


이 연구에서는 대형 언어 모델(LLM)과 지식 그래프(KG)를 통합하여 환각 현상을 완화하는 방법을 탐구합니다. 그러나 기존 연구는 대개 비공식적이거나 매우 큰 모델에 의존하여 접근성과 확장성을 제한합니다. 우리는 소형 언어 모델(SLM)이 KG 기반 질문 응답에서 기존 통합 방법의 성능이 제한적임을 관찰하였고, 이는 지식 그래프를 탐색하고 추론하는 능력이 부족하기 때문이라고 주장합니다. 이를 해결하기 위해, 우리는 언어 모델 대신 지식 그래프 탐색을 처리할 수 있는 간단하고 효율적인 탐색 모듈을 활용하는 방안을 제안합니다. 실험 결과, 이러한 경량 모듈이 SLM의 KG 질문 응답 성능을 효과적으로 향상시킨다는 것을 보여줍니다.




This study investigates the integration of knowledge graphs (KGs) into the reasoning processes of large language models (LLMs) as a promising approach to mitigate hallucination. However, existing work in this area often relies on proprietary or extremely large models, limiting accessibility and scalability. We observe that the performance of existing integration methods for small language models (SLMs) in KG-based question answering is often constrained by their limited ability to traverse and reason over knowledge graphs. To address this limitation, we propose leveraging simple and efficient exploration modules to handle knowledge graph traversal in place of the language model itself. Experimental results demonstrate that these lightweight modules effectively improve the performance of small language models on knowledge graph question answering tasks.


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



이 논문에서는 작은 언어 모델(SLMs)이 지식 그래프(KG)를 활용하여 질문 응답을 수행하는 데 있어 겪는 한계와 이를 극복하기 위한 방법을 제안합니다. 연구의 주요 초점은 SLMs가 KG를 효과적으로 탐색하고 추론하는 능력이 제한적이라는 점입니다. 이를 해결하기 위해 저자들은 탐색 모듈을 도입하여 SLMs의 성능을 향상시키는 방법을 제안합니다.

#### 1. 연구 배경
대형 언어 모델(LLMs)은 자연어 처리 작업에서 뛰어난 성능을 보이지만, 해석 가능성 부족과 환각(hallucination) 문제에 직면해 있습니다. 이러한 문제를 해결하기 위해 Think-on-Graph(ToG)와 같은 접근 방식이 제안되었으며, 이는 LLM을 지식 그래프와 상호작용하는 에이전트로 간주합니다. 그러나 이러한 방법은 대형 모델에 의존하기 때문에 자원 제약이 있는 환경에서는 접근성이 떨어집니다.

#### 2. SLMs의 한계
SLMs는 KG 기반 질문 응답에서 ToG를 적용할 때 성능이 저하되는 경향이 있습니다. 저자들은 SLMs가 KG를 탐색하고 추론하는 데 필요한 정보를 충분히 얻지 못하는 것이 주요 원인이라고 분석합니다. 이로 인해 SLMs는 질문에 대한 정확한 답변을 생성하는 데 어려움을 겪습니다.

#### 3. 탐색 모듈의 도입
저자들은 SLMs의 탐색 단계를 개선하기 위해 경량의 탐색 모듈을 도입합니다. 이 모듈은 문장 임베딩을 생성하는 SentenceBERT와 같은 모델을 사용하여 KG 탐색을 지원합니다. 이러한 모듈은 추가적인 훈련 없이도 사용할 수 있으며, SLMs가 KG를 효과적으로 탐색할 수 있도록 돕습니다.

#### 4. 실험 결과
실험 결과, 경량 탐색 모듈을 도입한 SLMs는 기존의 ToG와 CoT(Chain-of-Thought) 방법보다 성능이 향상되었습니다. 저자들은 이러한 결과가 SLMs의 탐색 품질이 성능의 주요 병목 현상임을 보여준다고 주장합니다.

#### 5. 결론
이 연구는 SLMs가 KG를 활용하여 질문 응답을 수행하는 데 있어 탐색 단계의 중요성을 강조하며, 경량 탐색 모듈을 통해 SLMs의 성능을 개선할 수 있음을 보여줍니다. 이러한 접근은 자원 제약이 있는 환경에서도 KG를 효과적으로 활용할 수 있는 기반을 마련합니다.

---




This paper investigates the limitations of small language models (SLMs) in leveraging knowledge graphs (KGs) for question answering and proposes methods to overcome these challenges. The primary focus of the study is on the constrained ability of SLMs to effectively traverse and reason over KGs. To address this limitation, the authors introduce exploration modules to enhance the performance of SLMs.

#### 1. Background
Large language models (LLMs) have demonstrated impressive performance across various natural language processing tasks; however, they face issues such as a lack of interpretability and hallucination. To tackle these challenges, approaches like Think-on-Graph (ToG) have been proposed, which treat LLMs as agents interacting with knowledge graphs. However, these methods often rely on large models, limiting accessibility in resource-constrained settings.

#### 2. Limitations of SLMs
SLMs tend to underperform when applying ToG for KG-based question answering. The authors analyze that the primary reason for this is the inability of SLMs to access sufficient information needed to generate accurate answers. Consequently, SLMs struggle to provide correct responses to questions.

#### 3. Introduction of Exploration Modules
To improve the exploration stage of SLMs, the authors propose the integration of lightweight exploration modules. These modules utilize models like SentenceBERT, which generate semantic embeddings to assist in KG traversal. Importantly, these modules can be used without additional training, enabling SLMs to effectively explore KGs.

#### 4. Experimental Results
The experimental results demonstrate that SLMs equipped with lightweight exploration modules significantly outperform both the original ToG and Chain-of-Thought (CoT) methods. The authors argue that these findings highlight the quality of exploration as a key bottleneck for SLM performance.

#### 5. Conclusion
This study emphasizes the importance of the exploration stage for SLMs in utilizing KGs for question answering and shows that lightweight exploration modules can enhance SLM performance. This approach lays the groundwork for effectively leveraging KGs in practical, resource-constrained environments.


<br/>
# Results



이 논문에서는 작은 언어 모델(SLMs)이 지식 그래프(KG)를 활용한 질문 응답(KGQA)에서의 성능을 평가하고, 그 한계를 극복하기 위한 방법을 제안합니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구에서는 다양한 크기의 언어 모델을 비교했습니다. 특히, GPT-4.1과 같은 대형 언어 모델(LLMs)과 Qwen2, Gemma2, Phi-3-mini, Llama 등 여러 SLMs를 사용하여 성능을 비교했습니다.

2. **테스트 데이터**: 두 개의 벤치마크 데이터셋인 ComplexWebQuestions (CWQ)와 WebQSP를 사용했습니다. CWQ는 최대 4단계의 복잡한 질문을 포함하고 있으며, WebQSP는 주로 1-2단계의 질문을 포함합니다.

3. **메트릭**: 성능 평가는 정확한 답변 문자열이 주어진 정답과 일치하는지를 측정하는 정확도(Exact Match, EM) 점수를 사용했습니다.

4. **비교 결과**: 
   - 대형 언어 모델인 GPT-4.1은 ToG(Think-on-Graph) 프레임워크를 적용했을 때 성능이 크게 향상되었습니다. 예를 들어, CWQ 데이터셋에서 GPT-4.1은 ToG를 사용하여 0.575의 EM 점수를 기록했습니다.
   - 반면, SLMs는 ToG를 적용했을 때 성능 향상이 제한적이었고, 때로는 Chain-of-Thought (CoT) 기준선보다도 낮은 성능을 보였습니다. 예를 들어, Qwen2-0.5b 모델은 ToG를 사용했을 때 CWQ에서 0.175의 EM 점수를 기록했습니다.
   - SLMs의 성능 저하는 주로 탐색 단계에서의 한계로 인해 발생했습니다. SLMs는 KG를 효과적으로 탐색하고 추론하는 데 필요한 정보를 충분히 얻지 못했습니다.

5. **탐색 모듈의 효과**: 연구에서는 SLMs의 탐색 성능을 개선하기 위해 경량의 패시지 검색 모델(SentenceBERT, GTR)을 도입했습니다. 이들 모델은 SLMs의 KG 탐색을 지원하여 성능을 크게 향상시켰습니다. 예를 들어, Qwen2-0.5b 모델은 SentenceBERT를 사용했을 때 CWQ에서 0.210의 EM 점수를 기록하여 ToG보다 개선된 성능을 보였습니다.

이 연구는 SLMs가 KG를 활용한 질문 응답에서 효과적으로 작동할 수 있도록 탐색 단계를 개선하는 방법을 제시하며, 향후 연구에 대한 기초를 제공합니다.

---




This paper evaluates the performance of small language models (SLMs) in knowledge graph (KG)-based question answering (KGQA) and proposes methods to overcome their limitations. The main results of the study are as follows:

1. **Competing Models**: The study compares various sizes of language models, particularly large language models (LLMs) like GPT-4.1 with several SLMs such as Qwen2, Gemma2, Phi-3-mini, and Llama.

2. **Test Data**: Two benchmark datasets, ComplexWebQuestions (CWQ) and WebQSP, were used. CWQ contains complex questions requiring up to 4-hop reasoning, while WebQSP primarily involves 1-2 hop questions.

3. **Metrics**: Performance evaluation was conducted using the Exact Match (EM) score, which measures whether the predicted answer string exactly matches the given answer.

4. **Comparison Results**: 
   - The large language model GPT-4.1 showed significant performance improvement when applying the Think-on-Graph (ToG) framework. For instance, GPT-4.1 achieved an EM score of 0.575 on the CWQ dataset using ToG.
   - In contrast, SLMs showed limited performance improvement with ToG and sometimes performed worse than the Chain-of-Thought (CoT) baseline. For example, the Qwen2-0.5b model recorded an EM score of 0.175 on CWQ when using ToG.
   - The performance drop in SLMs was primarily attributed to limitations in the exploration stage, where SLMs failed to retrieve sufficient information necessary for effective reasoning over the KG.

5. **Effectiveness of Exploration Modules**: The study introduced lightweight passage retrieval models (SentenceBERT, GTR) to improve the exploration performance of SLMs. These models significantly enhanced the KG exploration capabilities of SLMs. For instance, the Qwen2-0.5b model achieved an EM score of 0.210 on CWQ when using SentenceBERT, showing improved performance over ToG.

This research presents a method to enhance the exploration stage, allowing SLMs to operate effectively in KG-based question answering, providing a foundation for future research.


<br/>
# 예제



이 논문에서는 작은 언어 모델(SLMs)이 지식 그래프(KG)를 활용하여 질문에 답하는 과정에서의 한계를 다루고 있습니다. 특히, SLMs가 KG를 탐색하는 데 있어 효과적이지 않다는 점을 강조하고, 이를 해결하기 위해 경량의 탐색 모듈을 도입하는 방법을 제안합니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: "북부 지구가 있는 나라의 정부 형태는 무엇인가요?"
   - **출력**: "이스라엘은 의회제 정부 형태를 가지고 있습니다."

   이 예시는 모델이 질문을 이해하고, 관련된 지식 그래프의 정보를 바탕으로 답변을 생성하는 과정을 보여줍니다. 모델은 "북부 지구"와 "이스라엘" 간의 관계를 탐색하여 정부 형태에 대한 정보를 찾습니다.

2. **테스트 데이터**:
   - **입력**: "1980년에 브라후이 언어가 주요 언어인 나라의 대통령은 누구인가요?"
   - **출력**: "브라후이 언어가 주요 언어인 나라는 파키스탄이며, 그 당시의 대통령은 지아 울 하크입니다."

   이 테스트 데이터는 모델이 복잡한 질문을 처리할 수 있는지를 평가합니다. 모델은 KG에서 "브라후이 언어"와 관련된 정보를 탐색하여 해당 국가와 대통령에 대한 정보를 찾아야 합니다.

#### 구체적인 테스크
- **질문 응답**: 주어진 질문에 대해 지식 그래프를 활용하여 정확한 답변을 생성하는 작업입니다. 이 과정에서 모델은 질문의 주제와 관련된 엔티티를 식별하고, KG에서 이들 간의 관계를 탐색하여 최종 답변을 도출합니다.

### English Version

This paper addresses the limitations of small language models (SLMs) in leveraging knowledge graphs (KGs) for question answering. It emphasizes that SLMs are often ineffective in exploring KGs and proposes the introduction of lightweight exploration modules to address this issue.




1. **Training Data**:
   - **Input**: "What type of government is used in the country with the Northern District?"
   - **Output**: "Israel has a parliamentary system of government."

   This example illustrates how the model understands the question and generates an answer based on relevant information from the knowledge graph. The model explores the relationship between "Northern District" and "Israel" to find information about the type of government.

2. **Test Data**:
   - **Input**: "Who was the president of the country where Brahui was the main language in 1980?"
   - **Output**: "The country where Brahui was the main language is Pakistan, and the president at that time was Zia-ul-Haq."

   This test data evaluates whether the model can handle complex questions. The model needs to explore the KG to find information about the country and the president related to "Brahui language."

#### Specific Task
- **Question Answering**: The task of generating accurate answers to given questions using knowledge graphs. In this process, the model identifies the entities related to the question and explores the relationships between them in the KG to derive the final answer.

<br/>


이 논문에서는 작은 언어 모델(SLM)이 지식 그래프(KG)를 활용한 질문 응답에서 탐색 단계의 한계로 인해 성능이 저하된다는 점을 강조합니다. 저자들은 경량 탐색 모듈을 도입하여 SLM의 KG 탐색 능력을 향상시키고, 이를 통해 성능이 크게 개선되었음을 실험 결과로 보여줍니다. 예를 들어, SLM이 GPT-4.1의 도움을 받아 KG를 탐색할 때, 정확한 답변을 생성할 수 있는 능력이 향상되었습니다.

---

In this paper, the authors emphasize that small language models (SLMs) suffer from performance degradation in knowledge graph (KG) question answering due to limitations in the exploration stage. They introduce lightweight exploration modules to enhance the KG traversal capabilities of SLMs, demonstrating significant performance improvements through experimental results. For instance, when SLMs leverage GPT-4.1 for KG exploration, their ability to generate accurate answers is notably enhanced.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1**: SLMs의 탐색 출력과 GPT-4.1의 탐색 결정 간의 교차 엔트로피(Cross-Entropy) 정렬을 보여줍니다. 모델 크기가 증가함에 따라 정렬이 일관되게 증가하는 것을 확인할 수 있으며, 이는 탐색 품질이 SLMs의 성능 병목 현상임을 지지합니다.
  
- **Figure 2**: 제약 디코딩(Constrained Decoding) 적용 전후의 관계 정리 오류를 보여줍니다. 작은 모델인 Qwen2-0.5b와 Qwen2-1.5b에서 포맷 오류가 크게 줄어드는 것을 확인할 수 있으며, 이는 제약 디코딩 전략의 효과를 나타냅니다.

- **Figure 3**: 모델이 검색한 관계 경로와 진짜 경로 간의 평균 교차 엔트로피를 보여줍니다. 제약 디코딩 적용 전후의 CE 값이 안정적으로 유지되며, 이는 제약 디코딩이 모델의 탐색 능력에 부정적인 영향을 미치지 않음을 확인합니다.

#### 2. 테이블
- **Table 1**: ToG와 CoT의 성능 비교를 보여줍니다. GPT-4.1은 ToG에서 상당한 성능 향상을 보였지만, SLMs는 CoT 기준선보다도 낮은 성능을 보였습니다. 이는 ToG가 SLMs에 효과적으로 적용되지 않음을 나타냅니다.

- **Table 3**: GPT-4.1의 탐색 지원을 받았을 때 SLMs의 성능을 보여줍니다. GPT-4.1의 도움으로 SLMs는 CoT 기준선보다 더 나은 성능을 보였으며, 이는 탐색이 SLMs의 성능 병목임을 강조합니다.

- **Table 4**: 경량화된 패시지 검색 방법이 KG 탐색에 미치는 효과를 보여줍니다. SentenceBERT와 GTR이 모든 SLM에서 ToG 및 CoT보다 상당한 성능 향상을 가져왔음을 확인할 수 있습니다.

#### 3. 어펜딕스
- **Appendix A**: KG 탐색을 위한 패시지 검색 구현 세부사항을 설명합니다. 질문과 후보 패시지 간의 관련성 점수를 계산하여 상위 k개의 패시지를 선택하는 방법을 제시합니다.

- **Appendix B**: 제약 디코딩을 통해 SLMs와 LLMs 간의 성능 차이가 단순한 포맷 불일치 때문이 아님을 확인하기 위한 방법을 설명합니다.

- **Appendix C**: LLMs에 대한 패시지 검색의 효과를 다룹니다. Sun et al. (2024)의 연구와 대조적으로, SLMs에 대한 패시지 검색 모델의 통합이 성능 향상에 기여함을 보여줍니다.

---




#### 1. Diagrams and Figures
- **Figure 1**: Shows the cross-entropy alignment between the exploration outputs of SLMs and the exploration decisions of GPT-4.1. The consistent increase in alignment with model size supports the notion that exploration quality is a key bottleneck for SLM performance.

- **Figure 2**: Illustrates relation cleaning errors before and after applying constrained decoding. A significant reduction in formatting errors for smaller models like Qwen2-0.5b and Qwen2-1.5b indicates the effectiveness of the constrained decoding strategy.

- **Figure 3**: Displays the average cross-entropy between model-retrieved relation paths and the ground-truth paths before and after applying constrained decoding. The stable CE values confirm that the constrained decoding strategy does not negatively impact the exploration capability of the models.

#### 2. Tables
- **Table 1**: Compares the performance of ToG and CoT. While GPT-4.1 shows significant improvement with ToG, SLMs perform even worse than the CoT baseline, indicating that ToG does not effectively translate to SLMs.

- **Table 3**: Shows the performance of SLMs when assisted by GPT-4.1 for exploration. With the context provided by GPT-4.1, SLMs outperform the CoT baseline, emphasizing exploration as a key bottleneck.

- **Table 4**: Demonstrates the effectiveness of lightweight passage retrieval methods for KG exploration. Both SentenceBERT and GTR provide substantial performance gains across all SLMs, validating their effectiveness.

#### 3. Appendix
- **Appendix A**: Details the implementation of passage retrieval for KG exploration, explaining how relevance scores are computed between questions and candidate passages to select the top-k passages.

- **Appendix B**: Describes the method used to ensure that the performance gap between SLMs and LLMs is not simply due to formatting inconsistencies.

- **Appendix C**: Discusses the effects of passage retrieval on LLMs, contrasting with previous findings and showing that integrating passage retrieval models can enhance SLM performance without the trade-offs observed in LLMs.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Cheng2025,
  author    = {Yi-Jie Cheng and Oscar Chew and Yun-Nung Chen},
  title     = {The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)},
  pages     = {919--928},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  address   = {July 28-29, 2025}
}
```

### 시카고 스타일

Cheng, Yi-Jie, Oscar Chew, and Yun-Nung Chen. 2025. "The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)*, 919–928. Association for Computational Linguistics. July 28-29, 2025.
