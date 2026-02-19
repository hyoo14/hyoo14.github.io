---
layout: post
title:  "[2022]Learning to Rank Instant Search Results with Multiple Indices: A Case Study in Search Aggregation for Entertainment"
date:   2026-02-19 21:53:23 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Xfinity의 즉각 검색 시스템을 위한 다중 인덱스를 활용한 결과 순위 학습 방법을 제안합니다.


짧은 요약(Abstract) :


이 논문에서는 Xfinity의 즉각 검색 시스템을 다루고 있습니다. 이 시스템은 사용자가 입력하는 각 키에 대해 다양한 출처에서 결과를 제공하며, 결과에는 영화, TV 시리즈, 스포츠 이벤트, 음악 비디오, 뉴스 클립 등이 포함됩니다. 사용자는 Xfinity 음성 리모컨을 통해 더 긴 쿼리를 제출할 수 있으며, 이러한 쿼리는 불완전한 단어, 주제 검색, 또는 특정한 검색을 포함할 수 있습니다. 결과는 어휘적 일치, 의미적 일치, 항목 간 유사성 일치 등 다양한 방식으로 생성되며, 이러한 결과를 하나의 목록으로 결합하는 것이 주요 도전 과제입니다. 이를 해결하기 위해, 저자들은 검색 쿼리를 고려한 학습 기반 순위 매기기(Learning to Rank, LTR) 신경망 모델을 제안합니다. 이 결합된 목록은 사용자의 검색 기록과 프로그램 메타데이터를 반영하여 개인화될 수 있습니다. 즉각 검색에 대한 연구가 부족한 상황에서, 저자들은 다른 실무자들에게 도움이 될 수 있는 연구 결과를 제시합니다.




This paper addresses the instant search system at Xfinity, which provides a variety of results from different sources for each keystroke entered by the user. The results can include movies, television series, sporting events, music videos, news clips, and more. Users can also submit longer queries using the Xfinity Voice Remote, which may include incomplete words, topical searches, or more specific searches. The results can be generated through various methods such as lexical matches, semantic matches, and item-to-item similarity matches, presenting a key challenge in how to combine these results into a single list. To tackle this, the authors propose a Learning to Rank (LTR) neural model that takes the search query into account. This combined list can be personalized based on the user's search history and metadata of the programs. Given the underrepresentation of instant search in the literature, the authors present their findings to aid other practitioners.


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



이 논문에서는 Xfinity의 인스턴트 검색 시스템을 위한 학습 기반 랭킹 모델을 제안합니다. 이 시스템은 다양한 출처에서 제공되는 검색 결과를 통합하여 사용자에게 제공하는 것을 목표로 합니다. 이 과정에서 두 가지 주요 단계가 있습니다: 후보 생성과 재랭킹입니다.

1. **후보 생성**: 후보 생성 단계에서는 여러 인덱스에 비동기 호출을 하여 검색 쿼리에 대한 후보 결과를 생성합니다. 이 단계에서는 다음과 같은 다양한 매칭 기법이 사용됩니다:
   - **Lexical Matching (어휘 매칭)**: 쿼리의 접두사가 포함된 제목을 가진 항목을 후보 목록에 포함시키고, 글로벌 인기 점수로 재랭킹합니다.
   - **Semantic Search Model (의미 검색 모델)**: 쌍둥이 신경망(시암 신경망)을 사용하여 쿼리와 콘텐츠의 의미적 유사성을 평가합니다. 이 모델은 사전 훈련된 자연어 처리(NLP) 모델을 활용하여 쿼리와 콘텐츠의 벡터 표현을 생성합니다.
   - **Item-to-Item Similarity Candidates (항목 간 유사성 후보)**: 협업 필터링 기반 접근 방식을 사용하여 유사한 항목 목록을 미리 계산하고 저장합니다.
   - **Trending Candidates (트렌드 후보)**: 최근 클릭 데이터를 기반으로 트렌드 항목을 식별하고 이를 후보 목록의 상위에 배치합니다.

2. **재랭킹**: 후보 목록이 생성된 후, 두 개의 딥러닝 모델을 사용하여 재랭킹을 수행합니다.
   - **첫 번째 모델**: 쿼리와 항목 ID를 입력으로 받아, 쿼리의 n-그램을 임베딩하고 평균화하여 후보 목록을 결합합니다. 이 모델은 쿼리와 항목의 인기도를 캡처하기 위해 쌍별 학습을 통해 훈련됩니다.
   - **두 번째 모델**: 개인화된 결과를 제공하기 위해 사용자 검색 클릭 이력을 고려하여 상위 N개의 결과를 개인화합니다. 이 모델은 LSTM을 사용하여 쿼리와 항목 제목 간의 유사성을 식별합니다.

이 시스템은 대규모 사용자에게 서비스를 제공하기 위해 설계되었으며, 다양한 비즈니스 로직을 적용하여 최종 결과를 생성합니다. 예를 들어, 어휘 매칭이 없는 항목 간 유사성 매칭은 어휘 매칭보다 낮은 순위로 배치됩니다. 이러한 방식으로, 사용자는 더 나은 검색 경험을 얻을 수 있습니다.




This paper proposes a learning-based ranking model for Xfinity's instant search system, aimed at integrating search results from various sources and presenting them to users. The process consists of two main stages: candidate generation and reranking.

1. **Candidate Generation**: In the candidate generation stage, asynchronous calls are made to multiple indices to generate candidate results for a search query. Various matching techniques are employed in this stage:
   - **Lexical Matching**: Items whose titles contain the query as a prefix are included in the candidate list, and results are reranked by a global popularity score.
   - **Semantic Search Model**: A twin neural network (Siamese network) is used to evaluate the semantic similarity between the query and the content. This model leverages a pre-trained natural language processing (NLP) model to generate vector representations of the query and content.
   - **Item-to-Item Similarity Candidates**: A collaborative filtering-based approach is used to pre-compute and store lists of similar items.
   - **Trending Candidates**: Trending items are identified based on recent click data and boosted to the top of the candidate list.

2. **Reranking**: After generating the candidate lists, two deep learning models are employed for reranking.
   - **First Model**: This model takes the query and item ID as input, embedding and averaging the n-grams of the query to combine the candidate lists. It is trained using a pairwise learning approach to capture the popularity of items for a given query.
   - **Second Model**: This model personalizes the top N results by considering the user's search click history. It uses LSTM to identify the similarity between the search query and the item title.

The system is designed to serve millions of users at scale and applies various business logic to generate the final results. For instance, item-to-item similarity matches that do not also contain a lexical match are ranked lower than lexical matches. This approach aims to provide users with a better search experience.


<br/>
# Results



이 논문에서는 Xfinity의 인스턴트 검색 시스템에서 제안된 두 단계의 재순위 모델을 통해 얻은 결과를 다룹니다. 연구팀은 A/B 테스트를 통해 두 가지 주요 메트릭을 평가했습니다: 검색 성공률(SSR)과 평균 키 입력 수(ANK). 

1. **경쟁 모델**: 제안된 모델은 기존의 글로벌 인기 정렬 알고리즘과 비교되었습니다. A/B 테스트는 약 2주 동안 진행되었으며, 실험군과 대조군은 동일한 양의 트래픽을 받았습니다. 

2. **테스트 데이터**: 테스트 데이터는 최근 2주간의 사용자 검색 세션 데이터를 기반으로 하였으며, 각 세션은 최소 하나의 검색 결과 클릭 이벤트를 포함해야 했습니다. 총 150백만 개의 예제가 훈련에 사용되었습니다.

3. **메트릭**: 
   - **검색 성공률(SSR)**: 사용자가 검색 후 클릭한 결과가 후속 검색 없이 𝑇분 이내에 이루어진 비율로 정의됩니다. 
   - **평균 키 입력 수(ANK)**: 사용자가 원하는 검색 결과를 클릭하기 위해 입력한 평균 키 수입니다. 

4. **비교 결과**: 
   - 제안된 재순위 모델을 도입한 결과, SSR은 0.5-5% 향상되었고, ANK는 10-20% 감소했습니다. 
   - 특히, 개인화 모델이 추가되었을 때 짧은 쿼리(예: "MA")에서 더 큰 개선이 나타났습니다. 
   - 사용자가 입력하는 쿼리의 경우, 검색이 실패하는 이유는 사용자가 세션을 포기하거나 검색한 항목이 사용 가능하지 않기 때문입니다. 따라서 기계 학습 기반의 순위 매김이 SSR에 큰 영향을 미치지 않았지만, 소폭의 개선이 있었습니다. 
   - 반면, 새로운 인덱스를 도입함으로써 SSR에서 가장 큰 향상이 관찰되었습니다. 이는 더 많은 검색 사용 사례를 처리함으로써 사용자가 더 많은 콘텐츠를 찾을 수 있도록 도와주었습니다.

이러한 결과는 제안된 인스턴트 검색 시스템이 다양한 쿼리 유형에 대해 효과적으로 작동하며, 사용자 경험을 개선하는 데 기여할 수 있음을 보여줍니다.

---





This paper discusses the results obtained from the proposed two-step reranking model in Xfinity's instant search system. The research team evaluated two main metrics through A/B testing: Search Success Rate (SSR) and Average Number of Keystrokes (ANK).

1. **Competing Model**: The proposed model was compared against a global popularity sorting algorithm. The A/B tests were conducted over approximately two weeks, with both the treatment and control groups receiving equal amounts of traffic.

2. **Test Data**: The test data was based on user search session data from the last two weeks, with each session required to include at least one search result click event. A total of 150 million examples were used for training.

3. **Metrics**: 
   - **Search Success Rate (SSR)**: Defined as the percentage of sessions ending in a search result click without a follow-up search within T minutes.
   - **Average Number of Keystrokes (ANK)**: The average number of keystrokes a user inputs before clicking on the desired search result.

4. **Comparison Results**: 
   - The introduction of the proposed reranking model resulted in a 0.5-5% improvement in SSR and a 10-20% reduction in ANK.
   - Notably, the addition of the personalization model led to greater improvements in shorter queries (e.g., "MA").
   - In cases where users typed queries, search failures were typically due to users abandoning the session or the searched item being unavailable. Therefore, machine learning-based ranking did not significantly impact SSR, but a slight improvement was observed.
   - Conversely, the introduction of new indices resulted in the largest gains in SSR. By handling more search use cases, the system enabled users to find more content.

These results demonstrate that the proposed instant search system operates effectively across various query types and contributes to enhancing the user experience.


<br/>
# 예제



이 논문에서는 Xfinity의 인스턴트 검색 시스템을 위한 학습 기반 랭킹 모델을 제안하고 있습니다. 이 시스템은 사용자가 입력하는 쿼리에 대해 다양한 출처에서 결과를 제공하며, 각 키 입력마다 새로운 결과를 화면에 렌더링합니다. 이 시스템의 주요 목표는 여러 인덱스에서 검색 결과를 통합하고, 이를 사용자 맞춤형으로 제공하는 것입니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 사용자의 검색 쿼리와 해당 쿼리에 대한 클릭 데이터. 예를 들어, 사용자가 "영화"라는 쿼리를 입력했을 때, 이 쿼리에 대해 클릭된 결과 목록이 수집됩니다. 이 데이터는 (쿼리, 아이템 ID) 쌍으로 구성되며, 각 쌍은 사용자가 클릭한 횟수로 가중치가 부여됩니다.
   - **출력**: 각 쿼리에 대해 랭킹된 아이템 목록. 예를 들어, "영화"라는 쿼리에 대해 "어벤져스", "타이타닉", "인셉션"과 같은 영화들이 랭킹되어 출력됩니다.

2. **테스트 데이터**:
   - **입력**: 새로운 사용자의 검색 쿼리. 예를 들어, "코미디 영화"라는 쿼리를 입력했을 때, 이 쿼리에 대한 결과를 예측하기 위해 사용됩니다.
   - **출력**: 모델이 예측한 랭킹된 아이템 목록. 예를 들어, "코미디 영화"라는 쿼리에 대해 "슈렉", "내 남자친구의 결혼식", "21 점프 스트리트"와 같은 영화들이 출력될 수 있습니다.

#### 구체적인 작업

- **작업 1**: 후보 생성
  - 여러 인덱스에서 후보 아이템을 생성합니다. 예를 들어, "코미디 영화"라는 쿼리에 대해, Lexical Matching, Semantic Search, Item-to-Item Similarity 등 다양한 방법을 통해 후보 아이템을 생성합니다.

- **작업 2**: 필터링
  - 사용자가 접근할 수 있는 아이템인지 확인하여 필터링합니다. 예를 들어, 사용자가 구독하지 않은 영화는 결과에서 제외됩니다.

- **작업 3**: 재랭킹
  - 두 개의 딥러닝 모델을 사용하여 최종 결과를 재랭킹합니다. 첫 번째 모델은 쿼리와 아이템 ID를 입력으로 받아 랭킹을 생성하고, 두 번째 모델은 사용자 검색 이력과 메타데이터를 기반으로 개인화된 랭킹을 생성합니다.

이러한 과정을 통해, 사용자는 보다 관련성 높은 검색 결과를 얻을 수 있으며, 시스템의 성능은 A/B 테스트를 통해 평가됩니다.

---




This paper proposes a learning-based ranking model for Xfinity's instant search system. The system provides results from various sources for user-input queries, rendering new results on the screen for each keystroke. The main goal of this system is to integrate search results from multiple indices and present them in a personalized manner.

#### Training Data and Test Data

1. **Training Data**:
   - **Input**: User search queries and click data corresponding to those queries. For example, when a user inputs the query "movie," the list of results clicked for that query is collected. This data is structured as (query, item ID) pairs, with weights assigned based on the number of clicks for each pair.
   - **Output**: A ranked list of items for each query. For instance, for the query "movie," the output might include ranked movies like "Avengers," "Titanic," and "Inception."

2. **Test Data**:
   - **Input**: New user search queries. For example, when a user inputs the query "comedy movie," this query is used to predict results.
   - **Output**: A ranked list of items predicted by the model. For example, for the query "comedy movie," the output might include movies like "Shrek," "My Best Friend's Wedding," and "21 Jump Street."

#### Specific Tasks

- **Task 1**: Candidate Generation
  - Generate candidate items from multiple indices. For example, for the query "comedy movie," candidates are generated using various methods such as Lexical Matching, Semantic Search, and Item-to-Item Similarity.

- **Task 2**: Filtering
  - Filter candidates based on whether the user has access to the items. For instance, movies that the user is not subscribed to are excluded from the results.

- **Task 3**: Reranking
  - Use two deep learning models to rerank the final results. The first model takes the query and item ID as input to generate rankings, while the second model generates personalized rankings based on user search history and metadata.

Through these processes, users can obtain more relevant search results, and the system's performance is evaluated through A/B testing.

<br/>
# 요약



이 논문에서는 Xfinity의 즉각 검색 시스템을 위한 다중 인덱스를 활용한 결과 순위 학습 방법을 제안합니다. 두 개의 딥러닝 모델을 사용하여 후보 목록을 결합하고 개인화된 결과를 생성하며, A/B 테스트를 통해 검색 성공률과 클릭 수에서 유의미한 개선을 보였습니다. 예를 들어, 개인화 모델은 짧은 쿼리에서 평균 키 입력 수와 성공 시간에서 10-20%의 개선을 달성했습니다.

---

This paper proposes a learning-to-rank method for instant search results using multiple indices in Xfinity's instant search system. Two deep learning models are employed to combine candidate lists and generate personalized results, showing significant improvements in search success rates and click counts through A/B testing. For instance, the personalization model achieved a 10-20% improvement in average keystrokes and time to success for short queries.



<br/>
# 기타
논문 "Learning to Rank Instant Search Results with Multiple Indices: A Case Study in Search Aggregation for Entertainment"에서 다루어진 다이어그램, 피규어, 테이블, 어펜딕스의 주요 결과와 인사이트는 다음과 같습니다.

### 1. 다이어그램 및 피규어
- **Figure 1**: 검색 흐름 다이어그램
  - 이 다이어그램은 사용자가 "Park"라는 쿼리를 입력했을 때의 검색 흐름을 보여줍니다. 다양한 후보 결과(lexical match, synonyms, semantic search 등)가 생성되고, 필터링 및 재순위화 과정을 통해 최종 결과가 도출됩니다. 이 과정은 사용자가 다양한 유형의 콘텐츠(영화, TV 프로그램 등)를 쉽게 찾을 수 있도록 돕습니다.

- **Figure 3**: 재순위화 모델 아키텍처
  - 두 개의 타워로 구성된 모델이 쿼리와 아이템 ID를 임베딩하여 결합하는 과정을 보여줍니다. 이 모델은 쿼리와 아이템 간의 관계를 학습하여 최종 순위를 매기는 데 사용됩니다. 이는 다양한 후보 리스트를 통합하여 일관된 결과를 생성하는 데 기여합니다.

- **Figure 4**: 개인화 모델 아키텍처
  - 이 모델은 LSTM을 사용하여 쿼리와 아이템 제목 간의 유사성을 파악합니다. 사용자 검색 클릭 이력을 기반으로 최종 순위를 계산하는 과정이 포함되어 있습니다. 이는 사용자 맞춤형 검색 결과를 제공하는 데 중요한 역할을 합니다.

### 2. 테이블
- **Table of Results**: A/B 테스트 결과
  - A/B 테스트를 통해 재순위화 단계 도입 후 주요 메트릭에서 개선이 관찰되었습니다. 예를 들어, 검색 성공률, 클릭까지의 시간, 입력된 키스트로크 수에서 각각 0.5-5%, 10-20%의 개선이 있었습니다. 이는 재순위화 모델이 사용자 경험을 향상시키는 데 효과적임을 나타냅니다.

### 3. 어펜딕스
- 어펜딕스에서는 실험에 사용된 데이터 세트, 메트릭 정의, 세션화 로직 등 추가적인 세부 사항이 제공됩니다. 이러한 정보는 연구 결과의 신뢰성을 높이고, 다른 연구자들이 유사한 시스템을 구현하는 데 도움을 줄 수 있습니다.

---




### 1. Diagrams and Figures
- **Figure 1**: Search Flow Diagram
  - This diagram illustrates the search flow when a user inputs the query "Park." It shows how various candidate results (lexical match, synonyms, semantic search, etc.) are generated, filtered, and reranked to produce the final results. This process helps users easily find different types of content (movies, TV shows, etc.).

- **Figure 3**: Reranking Model Architecture
  - This figure depicts a two-tower model that embeds the query and item ID, combining them for final ranking. This model learns the relationship between the query and items, contributing to the generation of a cohesive list of results by integrating various candidate lists.

- **Figure 4**: Personalization Model Architecture
  - This model uses LSTM to identify similarities between the query and item titles. It incorporates user search click history to compute the final ranking. This plays a crucial role in providing personalized search results.

### 2. Tables
- **Table of Results**: A/B Test Results
  - The A/B tests showed improvements in key metrics after introducing the reranking step. For instance, there were improvements of 0.5-5% in search success rate and 10-20% in time to click and number of keystrokes. This indicates that the reranking model effectively enhances user experience.

### 3. Appendix
- The appendix provides additional details such as the datasets used in experiments, metric definitions, and sessionization logic. This information enhances the reliability of the research findings and can assist other researchers in implementing similar systems.

<br/>
# refer format:


### BibTeX 
```bibtex
@inproceedings{rome2022learning,
  author = {Scott Rome and Sardar Hamidian and Richard Walsh and Kevin Foley and Ferhan Ture},
  title = {Learning to Rank Instant Search Results with Multiple Indices: A Case Study in Search Aggregation for Entertainment},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22)},
  pages = {1--5},
  year = {2022},
  month = {July},
  publisher = {ACM},
  address = {New York, NY, USA},
  doi = {10.1145/3477495.3536334},
  isbn = {978-1-4503-8732-3}
}
```

### 시카고 스타일
Scott Rome, Sardar Hamidian, Richard Walsh, Kevin Foley, and Ferhan Ture. 2022. "Learning to Rank Instant Search Results with Multiple Indices: A Case Study in Search Aggregation for Entertainment." In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22)*, 1-5. New York, NY: ACM. https://doi.org/10.1145/3477495.3536334.
