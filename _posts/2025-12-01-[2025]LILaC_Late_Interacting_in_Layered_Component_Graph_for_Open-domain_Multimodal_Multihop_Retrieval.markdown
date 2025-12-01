---
layout: post
title:  "[2025]LILaC: Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval"
date:   2025-12-01 02:10:06 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: ColBERT의 토큰-수준 late interaction을, 멀티모달 노드 그래프로 확장한 버전 느낌  
(이미지+텍스트 등 멀티모달 멀티홉 검색에서 late-interaction 아이디어를 layered component graph 위에 얹은 구조.)  


짧은 요약(Abstract) :



이 논문에서는 멀티모달 문서 검색을 위한 새로운 프레임워크인 LILaC를 제안합니다. 멀티모달 문서 검색은 텍스트, 표, 이미지 등 다양한 요소로 구성된 문서에서 쿼리와 관련된 구성 요소를 검색하는 것을 목표로 합니다. 효과적인 멀티모달 검색기는 두 가지 주요 과제를 해결해야 합니다: (1) 고정된 단일 세분화 검색 단위로 인한 불필요한 내용의 영향을 줄이고, (2) 문서 내 및 문서 간의 구성 요소 간의 의미적 관계를 효과적으로 포착하여 다중 홉 추론을 지원해야 합니다. 이를 해결하기 위해, LILaC는 두 가지 핵심 혁신을 특징으로 합니다. 첫째, 우리는 멀티모달 정보를 두 개의 레이어로 명시적으로 표현하는 레이어드 컴포넌트 그래프를 도입하여 효율적이면서도 정밀한 추론을 촉진합니다. 둘째, 우리는 지연 상호작용 기반의 서브그래프 검색 방법을 개발하여, 먼저 효율적인 후보 생성을 위해 세분화된 노드를 식별한 후, 지연 상호작용을 통해 세밀한 추론을 수행합니다. 광범위한 실험 결과, LILaC는 추가적인 미세 조정 없이도 모든 다섯 개 벤치마크에서 최첨단 검색 성능을 달성함을 보여줍니다.




This paper proposes a novel framework called LILaC for multimodal document retrieval. Multimodal document retrieval aims to retrieve query-relevant components from documents composed of various elements such as text, tables, and images. An effective multimodal retriever needs to address two main challenges: (1) mitigating the impact of irrelevant content caused by fixed, single-granular retrieval units, and (2) supporting multihop reasoning by effectively capturing semantic relationships among components within and across documents. To tackle these challenges, LILaC features two core innovations. First, we introduce a layered component graph that explicitly represents multimodal information at two layers, facilitating efficient yet precise reasoning. Second, we develop a late-interaction-based subgraph retrieval method that initially identifies coarse-grained nodes for efficient candidate generation and then performs fine-grained reasoning through late interaction. Extensive experimental results demonstrate that LILaC achieves state-of-the-art retrieval performance on all five benchmarks, notably without additional fine-tuning.


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



LILaC(레이어드 컴포넌트 그래프를 이용한 오픈 도메인 멀티모달 멀티홉 검색)는 멀티모달 문서 검색을 위한 혁신적인 프레임워크입니다. 이 방법은 두 가지 주요 혁신을 통해 기존의 검색 방법의 한계를 극복하고자 합니다.

1. **레이어드 컴포넌트 그래프 구성**: LILaC는 멀티모달 문서 집합을 두 개의 레이어로 구성된 그래프로 표현합니다. 이 그래프는 각 레이어가 서로 다른 세분화 수준을 나타내며, 상위 레이어는 문서와 컴포넌트 간의 관계를 모델링하여 멀티홉 검색을 지원합니다. 하위 레이어는 각 컴포넌트를 더 세부적인 하위 구성 요소로 분해하여 정밀한 검색을 가능하게 합니다. 이러한 구조는 컴포넌트 간의 의미적 관계를 명시적으로 인코딩하여 효과적인 멀티홉 추론을 지원합니다.

2. **레이트 인터랙션 기반 서브그래프 검색**: LILaC는 쿼리와 관련된 서브그래프를 검색하는 과정에서 레이트 인터랙션 방식을 사용합니다. 이 방법은 초기 후보 노드를 식별한 후, 이들 노드와 연결된 엣지를 탐색하면서 동적으로 관련성을 평가합니다. 이를 통해 모든 엣지에 대한 점수를 미리 계산하는 대신, 쿼리의 세부 사항에 따라 필요한 엣지의 점수를 실시간으로 계산하여 효율성을 높입니다.

LILaC는 이러한 두 가지 혁신을 통해 멀티모달 문서 검색에서의 성능을 크게 향상시킵니다. 실험 결과, LILaC는 다섯 개의 벤치마크에서 최첨단 검색 성능을 달성하였으며, 추가적인 파인튜닝 없이도 뛰어난 결과를 보여주었습니다.




LILaC (Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval) is an innovative framework designed for multimodal document retrieval. This method aims to overcome the limitations of existing retrieval approaches through two key innovations.

1. **Layered Component Graph Construction**: LILaC represents a collection of multimodal documents as a graph structured in two layers. Each layer represents different levels of granularity, with the upper layer modeling the relationships between documents and components to support multihop retrieval. The lower layer decomposes each component into finer subcomponents, enabling precise retrieval. This structure explicitly encodes semantic relationships among components, facilitating effective multihop reasoning.

2. **Late Interaction-based Subgraph Retrieval**: In the process of retrieving a query-relevant subgraph, LILaC employs a late interaction approach. This method identifies an initial set of candidate nodes and then explores the edges connected to these nodes while dynamically evaluating relevance. Instead of pre-computing scores for all edges, it calculates the scores in real-time based on the specifics of the query, enhancing efficiency.

Through these two innovations, LILaC significantly improves performance in multimodal document retrieval. Experimental results demonstrate that LILaC achieves state-of-the-art retrieval performance across five benchmarks, notably without additional fine-tuning.


<br/>
# Results


이 논문에서는 LILaC라는 새로운 다중 모달 검색 프레임워크를 제안합니다. LILaC는 두 가지 주요 혁신을 통해 다중 모달 문서 검색의 한계를 극복하고자 합니다. 첫 번째는 다층 구성 요소 그래프(layered component graph)를 도입하여, 문서 내의 다양한 모달 정보(텍스트, 표, 이미지 등)를 두 개의 레이어(거칠고 세밀한 레이어)로 명시적으로 표현합니다. 두 번째는 지연 상호작용 기반의 서브그래프 검색 방법을 개발하여, 초기 후보 생성을 위한 거칠고 세밀한 추론을 효율적으로 수행합니다.

#### 실험 설정
LILaC의 성능을 평가하기 위해 다섯 가지 벤치마크 데이터셋(MP-DocVQA, SlideVQA, InfoVQA, MultimodalQA, MMCoQA)을 사용하였으며, 각 데이터셋에서의 검색 정확도를 Mean Reciprocal Rank at 10 (MRR@10)과 Recall@3으로 측정했습니다. 

#### 경쟁 모델
LILaC는 두 가지 주요 경쟁 모델인 VisRAG와 ColPali와 비교되었습니다. VisRAG는 문서 이미지를 VLMs(비전 언어 모델)를 통해 직접 인코딩하는 방법을 사용하며, ColPali는 문서 이미지에서 다중 벡터 임베딩을 사용하는 방법입니다. 이 두 모델은 LILaC와 비교하여 상대적으로 낮은 성능을 보였습니다.

#### 결과
LILaC는 모든 벤치마크에서 최첨단 성능을 달성하였으며, 특히 MultimodalQA와 MMCoQA와 같은 복잡한 다중 홉 추론이 필요한 데이터셋에서 두드러진 성능 향상을 보였습니다. 예를 들어, LILaC는 평균적으로 Recall@3에서 60.68%의 향상과 MRR@10에서 59.90%의 향상을 기록했습니다. 

이러한 결과는 LILaC가 다중 모달 검색에서의 효과적인 다중 홉 추론을 지원하고, 세밀한 구성 요소의 관련성을 동적으로 평가하는 능력을 갖추고 있음을 보여줍니다. 




This paper proposes a novel multimodal retrieval framework called LILaC, which aims to overcome the limitations of existing methods through two key innovations. The first is the introduction of a layered component graph that explicitly represents various modalities (text, tables, images, etc.) in two layers (coarse and fine granularity). The second is the development of a late-interaction-based subgraph retrieval method that efficiently performs both coarse and fine-grained reasoning for initial candidate generation.

#### Experimental Setup
To evaluate the performance of LILaC, five benchmark datasets (MP-DocVQA, SlideVQA, InfoVQA, MultimodalQA, MMCoQA) were used, measuring retrieval accuracy using Mean Reciprocal Rank at 10 (MRR@10) and Recall@3.

#### Competing Models
LILaC was compared against two main competing models, VisRAG and ColPali. VisRAG employs a method that directly encodes document images via Vision Language Models (VLMs), while ColPali uses multi-vector embeddings from document images. Both models exhibited relatively lower performance compared to LILaC.

#### Results
LILaC achieved state-of-the-art performance across all benchmarks, particularly showing significant improvements on datasets requiring complex multihop reasoning, such as MultimodalQA and MMCoQA. For instance, LILaC recorded an average improvement of 60.68% in Recall@3 and 59.90% in MRR@10.

These results demonstrate that LILaC effectively supports multihop reasoning in multimodal retrieval and possesses the capability to dynamically evaluate the relevance of fine-grained components.


<br/>
# 예제



LILaC 논문에서는 멀티모달 문서 검색을 위한 새로운 프레임워크를 제안하고 있습니다. 이 프레임워크는 두 가지 주요 혁신을 포함하고 있습니다: 레이어드 컴포넌트 그래프와 레이트 인터랙션 기반 서브그래프 검색 방법입니다. 이 시스템은 다양한 입력과 출력 형식을 사용하여 멀티모달 정보를 효과적으로 검색합니다.

#### 1. 트레이닝 데이터와 테스트 데이터

- **트레이닝 데이터**: 모델은 다양한 멀티모달 문서(텍스트, 표, 이미지 등)로 구성된 데이터셋을 사용하여 훈련됩니다. 각 문서는 여러 컴포넌트로 나뉘며, 이들 컴포넌트는 서로 연결된 관계를 가집니다. 예를 들어, 한 문서에는 "타지마할"에 대한 설명이 포함된 텍스트, 그에 대한 표, 그리고 타지마할의 이미지가 있을 수 있습니다.

- **테스트 데이터**: 테스트 데이터는 모델이 훈련되지 않은 새로운 멀티모달 문서로 구성됩니다. 이 문서들은 질문에 대한 답변을 찾기 위해 사용됩니다. 예를 들어, "타지마할의 중앙 돔을 둘러싼 슬렌더 미나렛의 수는?"과 같은 질문이 주어질 수 있습니다.

#### 2. 구체적인 인풋과 아웃풋

- **인풋**: 사용자가 입력하는 질문은 자연어로 작성됩니다. 예를 들어, "타지마할의 중앙 돔을 둘러싼 슬렌더 미나렛의 수는?"이라는 질문이 있을 수 있습니다. 이 질문은 LILaC 시스템에 의해 처리되어야 합니다.

- **아웃풋**: 시스템은 질문에 대한 답변을 제공해야 합니다. 예를 들어, "4개"라는 답변이 출력될 수 있습니다. 이 답변은 시스템이 검색한 멀티모달 문서에서 추출된 정보에 기반합니다.

#### 3. 구체적인 테스크

- **객체 탐지**: 시스템은 이미지에서 객체를 탐지하고, 이를 JSON 형식으로 반환합니다. 예를 들어, "타지마할" 이미지에서 "타지마할"이라는 객체와 그 경계 상자를 반환합니다.

- **질문 분해**: 시스템은 입력된 질문을 여러 개의 하위 질문으로 분해하여 각 하위 질문이 특정 컴포넌트를 타겟으로 하도록 합니다. 예를 들어, "타지마할의 중앙 돔을 둘러싼 슬렌더 미나렛의 수는?"이라는 질문은 "타지마할의 중앙 돔"과 "슬렌더 미나렛"에 대한 하위 질문으로 나뉘어질 수 있습니다.

- **모달리티 선택**: 각 하위 질문에 대해 가장 적합한 모달리티(텍스트, 표, 이미지)를 선택합니다. 예를 들어, "타지마할의 중앙 돔"에 대한 질문은 텍스트 모달리티로 선택될 수 있습니다.

- **답변 생성**: 최종적으로, 시스템은 검색된 컴포넌트를 기반으로 질문에 대한 답변을 생성합니다. 예를 들어, "4개"라는 답변이 생성될 수 있습니다.




The LILaC paper proposes a new framework for multimodal document retrieval, incorporating two main innovations: a layered component graph and a late-interaction-based subgraph retrieval method. This system effectively retrieves multimodal information using various input and output formats.

#### 1. Training Data and Test Data

- **Training Data**: The model is trained on datasets composed of various multimodal documents (text, tables, images, etc.). Each document is divided into multiple components, which are interconnected. For example, a document may contain text describing the "Taj Mahal," a table related to it, and an image of the Taj Mahal.

- **Test Data**: The test data consists of new multimodal documents that the model has not been trained on. These documents are used to find answers to questions. For instance, a question like "How many slender minarets surround the central dome of the Taj Mahal?" may be posed.

#### 2. Specific Input and Output

- **Input**: The question entered by the user is written in natural language. For example, the question "How many slender minarets surround the central dome of the Taj Mahal?" is provided to the LILaC system for processing.

- **Output**: The system is expected to provide an answer to the question. For example, the output could be "4," which is based on the information extracted from the multimodal documents searched by the system.

#### 3. Specific Tasks

- **Object Detection**: The system detects objects in images and returns them in JSON format. For example, it may return the object "Taj Mahal" along with its bounding box from an image.

- **Query Decomposition**: The system decomposes the input question into several sub-queries, ensuring that each sub-query targets a specific component. For instance, the question "How many slender minarets surround the central dome of the Taj Mahal?" could be broken down into sub-queries about "the central dome of the Taj Mahal" and "the slender minarets."

- **Modality Selection**: For each sub-query, the system selects the most appropriate modality (text, table, image) for obtaining the answer. For example, the sub-query about "the central dome of the Taj Mahal" might be selected as text modality.

- **Answer Generation**: Finally, the system generates an answer based on the retrieved components. For example, the answer "4" could be generated based on the information found in the documents.

<br/>
# 요약


LILaC는 다중 모달 문서 검색을 위한 새로운 프레임워크로, 계층화된 구성 요소 그래프와 지연 상호작용 기반의 하위 그래프 검색 방법을 사용하여 효과적인 다중 홉 추론을 지원합니다. 실험 결과, LILaC는 다섯 개의 벤치마크에서 최첨단 검색 성능을 달성하며, 추가적인 미세 조정 없이도 우수한 결과를 보였습니다. 이 방법은 다양한 모달리티의 정보를 효과적으로 통합하여 검색 정확도를 높이는 데 기여합니다.



LILaC is a novel framework for multimodal document retrieval that utilizes a layered component graph and late-interaction-based subgraph retrieval method to support effective multihop reasoning. Experimental results demonstrate that LILaC achieves state-of-the-art retrieval performance across five benchmarks, notably without additional fine-tuning. This approach contributes to enhancing retrieval accuracy by effectively integrating information from various modalities.

<br/>
# 기타


1. **LILaC의 성능**: LILaC는 다중 모드 문서 검색에서 최신 기술을 초월하는 성능을 보여주었습니다. 모든 벤치마크에서 최첨단 성능을 달성했으며, 특히 MultimodalQA 및 MMCoQA와 같은 복잡한 멀티홉 추론이 필요한 데이터셋에서 두드러진 성과를 보였습니다. 이는 LILaC의 계층적 구성 요소 그래프와 지연 상호작용 기반 하위 그래프 검색 방법이 효과적으로 작용했음을 나타냅니다.

2. **계층적 구성 요소 그래프**: LILaC는 문서의 다중 모드 정보를 두 개의 계층으로 명시적으로 표현하는 계층적 구성 요소 그래프를 도입했습니다. 이 구조는 멀티홉 추론을 지원하고, 각 구성 요소 간의 관계를 명확히 하여 검색의 효율성과 정확성을 높였습니다.

3. **지연 상호작용 기반 하위 그래프 검색**: LILaC는 초기 후보 노드를 식별한 후, 지연 상호작용을 통해 세부적인 추론을 수행하는 방법을 개발했습니다. 이 접근 방식은 검색의 정확성을 높이고, 불필요한 계산을 줄이는 데 기여했습니다.

4. **실험 결과**: LILaC는 모든 벤치마크에서 평균적으로 14.24% 및 15.75%의 성능 향상을 보여주었으며, 이는 기존의 VisRAG 및 ColPali 모델에 비해 상당한 개선을 나타냅니다. 특히, 시각적 요소가 포함된 질문 응답 데이터셋에서 LILaC의 성능이 두드러졌습니다.

5. **제한 사항**: LILaC는 사전 훈련된 모델을 사용하여 성능을 향상시키지만, 하위 구성 요소 추출의 품질에 크게 의존합니다. 또한, 최종 생성 작업에서 여전히 개선의 여지가 있습니다.




1. **Performance of LILaC**: LILaC demonstrated superior performance, surpassing state-of-the-art methods in open-domain multimodal document retrieval. It achieved state-of-the-art results across all benchmarks, particularly excelling in datasets like MultimodalQA and MMCoQA that require complex multihop reasoning. This indicates the effectiveness of LILaC's layered component graph and late-interaction-based subgraph retrieval method.

2. **Layered Component Graph**: LILaC introduced a layered component graph that explicitly represents multimodal information at two distinct layers. This structure supports multihop reasoning and clarifies the relationships among components, enhancing both retrieval efficiency and effectiveness.

3. **Late-Interaction-Based Subgraph Retrieval**: LILaC developed a method that identifies initial candidate nodes and then performs fine-grained reasoning through late interaction. This approach improves retrieval accuracy while reducing unnecessary computations.

4. **Experimental Results**: LILaC showed an average performance improvement of 14.24% and 15.75% across all benchmarks compared to existing models like VisRAG and ColPali, indicating significant advancements. Notably, its performance was particularly strong in question-answering datasets that include visual elements.

5. **Limitations**: While LILaC enhances performance using pretrained models, it heavily relies on the quality of subcomponent extraction. Additionally, there remains substantial room for improvement in end-to-end generation tasks.

<br/>
# refer format:


### BibTeX 
```bibtex
@inproceedings{Yun2025,
  author    = {Joohyung Yun and Doyup Lee and Wook-Shin Han},
  title     = {LILaC: Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages     = {20552--20571},
  year      = {2025},
  month     = {November},
  publisher = {Association for Computational Linguistics},
  address   = {Republic of Korea},
  url       = {https://github.com/joohyung00/lilac}
}
```

### 시카고 스타일
Joohyung Yun, Doyup Lee, and Wook-Shin Han. "LILaC: Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval." In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, 20552–20571. November 4-9, 2025. Association for Computational Linguistics. https://github.com/joohyung00/lilac.
