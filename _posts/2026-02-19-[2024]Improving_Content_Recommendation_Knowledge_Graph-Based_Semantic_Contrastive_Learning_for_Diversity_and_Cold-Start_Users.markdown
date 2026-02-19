---
layout: post
title:  "[2024]Improving Content Recommendation: Knowledge Graph-Based Semantic Contrastive Learning for Diversity and Cold-Start Users"
date:   2026-02-19 21:40:22 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 지식 그래프와 콘텐츠 기반 대조 학습을 활용하여 추천 시스템의 성능을 향상시키는 하이브리드 다중 작업 학습 접근 방식을 제안합니다.


짧은 요약(Abstract) :

이 논문은 추천 시스템에서 데이터 희소성, 콜드 스타트 문제, 그리고 다양성 관련 도전 과제를 해결하기 위한 방법을 제안합니다. 기존의 많은 접근 방식은 지식 그래프를 활용하여 아이템 기반 및 사용자-아이템 협업 신호를 결합하여 이러한 문제를 해결하고자 합니다. 그러나 이러한 방법들은 종종 모델의 복잡성을 증가시키고 다양성을 감소시키며, 단순히 높은 순위 기반 성능을 추구하는 경향이 있습니다. 본 연구에서는 사용자-아이템 및 아이템-아이템 상호작용을 학습하는 하이브리드 다중 작업 학습 접근 방식을 제안합니다. 우리는 아이템 메타데이터를 기반으로 긍정적 및 부정적 쌍을 샘플링하여 설명 텍스트에 대해 아이템 기반 대조 학습을 적용합니다. 이 접근 방식은 지식 그래프 내의 엔티티 간의 관계를 더 잘 이해할 수 있도록 하여, 보다 정확하고 관련성 있으며 다양한 사용자 추천을 가능하게 합니다. 또한, 이 방법은 아이템과의 상호작용이 적은 콜드 스타트 사용자에게도 이점을 제공합니다. 우리는 두 개의 널리 사용되는 데이터셋에서 실험을 수행하여 제안한 방법의 효과를 검증하였습니다.


This paper proposes a method to address challenges related to data sparsity, cold-start problems, and diversity in recommendation systems. Many existing approaches leverage knowledge graphs to tackle these issues by combining item-based and user-item collaborative signals. However, these methods often tend to increase model complexity and reduce diversity, focusing solely on achieving high rank-based performance. In this study, we propose a hybrid multi-task learning approach that trains on user-item and item-item interactions. We apply item-based contrastive learning on descriptive text, sampling positive and negative pairs based on item metadata. This approach allows the model to better understand the relationships between entities within the knowledge graph, leading to more accurate, relevant, and diverse user recommendations. Additionally, it provides benefits even for cold-start users who have few interactions with items. We perform extensive experiments on two widely used datasets to validate the effectiveness of our proposed method.


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


이 논문에서는 콘텐츠 추천 시스템의 성능을 향상시키기 위해 지식 그래프(Knowledge Graph, KG)와 의미론적 대조 학습(Semantic Contrastive Learning)을 결합한 하이브리드 다중 작업 학습 접근 방식을 제안합니다. 이 방법은 사용자-아이템 상호작용과 아이템-아이템 상호작용을 동시에 학습하여 추천의 정확성과 다양성을 높이는 것을 목표로 합니다.

#### 모델 아키텍처
제안된 모델은 두 가지 주요 구성 요소로 이루어져 있습니다: 
1. **지식 그래프 컨볼루션 네트워크(KGCN)**: 이 네트워크는 사용자와 콘텐츠 간의 관계를 모델링하는 데 사용됩니다. KGCN은 각 노드의 특성을 업데이트하는 메커니즘을 통해 작동하며, 사용자와 콘텐츠 간의 상호작용을 기반으로 중요도 점수를 적용하여 이웃 노드의 특성을 집계합니다.
2. **대조 학습 손실 함수**: 콘텐츠의 메타데이터를 기반으로 긍정적 및 부정적 샘플을 생성하여 대조 학습을 수행합니다. 이 과정에서 콘텐츠의 장르나 제목 정보를 활용하여 문장을 생성하고, 이를 임베딩하여 유사성을 평가합니다. 긍정적 샘플은 유사한 콘텐츠로, 부정적 샘플은 비유사한 콘텐츠로 정의됩니다.

#### 트레이닝 데이터
모델은 두 가지 데이터셋을 사용하여 훈련됩니다: 
- **MovieLens-20M**: 약 2000만 개의 영화 평점 데이터로 구성되어 있습니다.
- **Book-Crossing**: 약 100만 개의 책 평점 데이터로 구성되어 있습니다.

각 데이터셋에서 콘텐츠의 메타데이터(예: 장르, 제목)를 수집하여 콘텐츠 임베딩을 생성합니다. 이 임베딩은 사전 훈련된 언어 모델(BERT 등)을 사용하여 생성됩니다.

#### 특별한 기법
- **다중 작업 학습**: 사용자-아이템 상호작용과 아이템-아이템 상호작용을 동시에 학습하여 추천의 다양성과 개인화를 향상시킵니다.
- **대조 학습**: 콘텐츠의 메타데이터를 활용하여 긍정적 및 부정적 샘플을 생성하고, 이를 통해 콘텐츠 간의 관계를 더 잘 이해하도록 모델을 훈련합니다.
- **차가운 시작 문제 해결**: 사용자와 콘텐츠 간의 상호작용이 적은 새로운 사용자에게도 추천의 질을 높이는 데 기여합니다.

이러한 방법론을 통해 제안된 모델은 추천 성능, 개인화, 다양성 및 임베딩 품질을 향상시키며, 특히 차가운 시작 시나리오에서도 효과적인 성능을 보입니다.

---





This paper proposes a hybrid multi-task learning approach that combines Knowledge Graph (KG) and Semantic Contrastive Learning to enhance the performance of content recommendation systems. The goal of this method is to improve the accuracy and diversity of recommendations by jointly learning user-item interactions and item-item interactions.

#### Model Architecture
The proposed model consists of two main components:
1. **Knowledge Graph Convolutional Network (KGCN)**: This network is used to model the relationships between users and content. KGCN operates through a mechanism that updates the features of each node and aggregates the features of neighboring nodes by applying importance scores based on user-content interactions.
2. **Contrastive Learning Loss Function**: This function performs contrastive learning by generating positive and negative samples based on the metadata of the content. In this process, genre or title information is used to create sentences, which are then embedded to evaluate similarity. Positive samples are defined as similar content, while negative samples are defined as dissimilar content.

#### Training Data
The model is trained using two datasets:
- **MovieLens-20M**: Composed of approximately 20 million movie rating data.
- **Book-Crossing**: Composed of approximately 1 million book rating data.

Metadata (e.g., genre, title) of the content is collected from each dataset to generate content embeddings. These embeddings are created using pre-trained language models (e.g., BERT).

#### Special Techniques
- **Multi-task Learning**: This approach enhances the diversity and personalization of recommendations by jointly learning user-item interactions and item-item interactions.
- **Contrastive Learning**: By utilizing the metadata of the content to generate positive and negative samples, the model is trained to better understand the relationships between contents.
- **Cold-Start Problem Resolution**: The method contributes to improving the quality of recommendations for new users with few interactions with items.

Through these methodologies, the proposed model enhances recommendation performance, personalization, diversity, and embedding quality, demonstrating effective performance even in cold-start scenarios.


<br/>
# Results



이 논문에서는 추천 시스템의 성능을 향상시키기 위해 지식 그래프 기반의 의미적 대조 학습을 활용한 하이브리드 다중 작업 학습 접근 방식을 제안합니다. 연구의 주요 목표는 데이터 희소성, 콜드 스타트 문제, 그리고 추천의 다양성을 해결하는 것입니다. 이를 위해 MovieLens-20M과 Book-Crossing 데이터셋을 사용하여 실험을 수행하였습니다.

#### 실험 설정
- **경쟁 모델**: 본 연구의 기본 모델은 KGCN(Knowledge Graph Convolutional Network)으로 설정하였으며, 제안된 모델은 기존 KGCN 모델에 대조 학습을 추가하여 성능을 비교하였습니다.
- **테스트 데이터**: MovieLens-20M 데이터셋은 약 2000만 개의 사용자 평가로 구성되어 있으며, Book-Crossing 데이터셋은 100만 개의 평가로 이루어져 있습니다. 두 데이터셋 모두 사용자와 콘텐츠 간의 상호작용을 포함하고 있습니다.
- **메트릭**: 추천 성능을 평가하기 위해 Click-Through Rate (CTR), Area Under the Curve (AUC), F1-score, Recall@K, Normalized Discounted Cumulative Gain (NDCG) 등의 메트릭을 사용하였습니다. 특히, 콜드 스타트 시나리오에서의 NDCG 성능을 분석하여 새로운 사용자에 대한 추천의 질을 평가하였습니다.

#### 결과
- **CTR 및 F1-score**: 제안된 모델은 기본 모델에 비해 AUC와 F1-score에서 유의미한 성능 향상을 보였습니다. 예를 들어, MovieLens 데이터셋에서 제안된 모델은 AUC 0.9780을 기록하여 기본 모델보다 통계적으로 유의미한 개선을 나타냈습니다.
- **Recall@K 및 NDCG@K**: 제안된 모델은 모든 K 값에서 기본 모델보다 높은 Recall과 NDCG를 기록하였습니다. 특히, 콜드 스타트 사용자에 대한 NDCG 성능이 개선되어, 추천 시스템이 새로운 사용자에게도 효과적으로 작동함을 보여주었습니다.
- **다양성 평가**: 추천 목록의 다양성을 평가하기 위해 Inter-list diversity와 Intra-list diversity 메트릭을 사용하였습니다. 제안된 모델은 다양한 추천을 제공하는 데 있어 기본 모델보다 우수한 성능을 보였습니다.

이러한 결과들은 제안된 접근 방식이 추천 시스템의 성능을 향상시키고, 특히 콜드 스타트 문제를 해결하는 데 효과적임을 입증합니다.

---



This paper proposes a hybrid multi-task learning approach that utilizes knowledge graph-based semantic contrastive learning to improve the performance of recommendation systems. The main goal of the study is to address challenges related to data sparsity, cold-start problems, and diversity in recommendations. Experiments were conducted using the MovieLens-20M and Book-Crossing datasets.

#### Experimental Setup
- **Competing Model**: The baseline model for this study is the Knowledge Graph Convolutional Network (KGCN), and the proposed model incorporates contrastive learning to compare performance against the baseline.
- **Test Data**: The MovieLens-20M dataset consists of approximately 20 million user ratings, while the Book-Crossing dataset contains 1 million ratings. Both datasets include user-item interactions.
- **Metrics**: To evaluate recommendation performance, metrics such as Click-Through Rate (CTR), Area Under the Curve (AUC), F1-score, Recall@K, and Normalized Discounted Cumulative Gain (NDCG) were employed. The NDCG performance under cold-start scenarios was specifically analyzed to assess the quality of recommendations for new users.

#### Results
- **CTR and F1-score**: The proposed model demonstrated significant performance improvements over the baseline model in terms of AUC and F1-score. For instance, in the MovieLens dataset, the proposed model achieved an AUC of 0.9780, indicating a statistically significant enhancement compared to the baseline.
- **Recall@K and NDCG@K**: The proposed model outperformed the baseline model across all K values in terms of Recall and NDCG. Notably, the NDCG performance for cold-start users improved, demonstrating the effectiveness of the recommendation system for new users.
- **Diversity Evaluation**: To assess the diversity of the recommendation lists, metrics for Inter-list diversity and Intra-list diversity were utilized. The proposed model exhibited superior performance in providing diverse recommendations compared to the baseline model.

These results validate that the proposed approach effectively enhances the performance of recommendation systems, particularly in addressing cold-start problems.


<br/>
# 예제



이 논문에서는 콘텐츠 추천 시스템을 개선하기 위해 지식 그래프 기반의 의미적 대조 학습(Semantic Contrastive Learning) 접근 방식을 제안합니다. 이 방법은 사용자-아이템 상호작용과 아이템-아이템 상호작용을 동시에 학습하는 하이브리드 다중 작업 학습(multi-task learning) 접근 방식을 사용합니다. 

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **영화 데이터셋 (MovieLens-20M)**: 약 2000만 개의 사용자 평가가 포함되어 있으며, 각 영화에 대한 시놉시스(줄거리) 정보가 TMDB 데이터베이스에서 수집됩니다. 예를 들어, "Despicable Me"라는 영화의 시놉시스는 "The genre(s) of the film is/are Animation, Comedy."와 같은 형식으로 변환됩니다.
   - **책 데이터셋 (Book-Crossing)**: 약 100만 개의 사용자 평가가 포함되어 있으며, 각 책에 대한 메타데이터는 Goodreads와 Google Books에서 수집됩니다. 예를 들어, "Harry Potter"라는 책의 시놉시스는 "A book title is Harry Potter (released_year). The genre(s) of the book is/are Fantasy."와 같은 형식으로 변환됩니다.

2. **테스트 데이터**:
   - 테스트 데이터는 트레이닝 데이터와 동일한 형식으로 구성되지만, 사용자-아이템 상호작용이 적은 '콜드 스타트(cold-start)' 사용자 그룹을 포함합니다. 이 그룹은 사용자-아이템 상호작용이 가장 적은 1%의 사용자로 정의됩니다.

#### 구체적인 태스크

- **추천 성능 평가**: 
  - **Click-Through Rate (CTR)**: 추천된 콘텐츠가 얼마나 클릭되는지를 측정합니다. 예를 들어, 추천된 영화 목록에서 사용자가 클릭한 영화의 비율을 계산합니다.
  - **Recall@K**: 추천된 K개의 콘텐츠 중 실제로 사용자가 선호한 콘텐츠의 비율을 측정합니다. 예를 들어, 사용자가 10개의 추천 영화 중 3개를 선호했다면 Recall@10은 0.3이 됩니다.
  - **NDCG@K**: 추천된 콘텐츠의 순위 품질을 평가합니다. 높은 순위에 있는 추천이 실제로 사용자가 선호하는 콘텐츠일수록 NDCG 점수가 높아집니다.

- **다양성 평가**:
  - **Inter-list Diversity**: 추천된 콘텐츠 목록 간의 다양성을 측정합니다. 예를 들어, 서로 다른 사용자에게 추천된 콘텐츠 목록 간의 유사성을 평가합니다.
  - **Intra-list Diversity**: 동일한 사용자에게 추천된 콘텐츠 목록 내의 다양성을 측정합니다. 예를 들어, 한 사용자에게 추천된 영화 목록에서 서로 다른 장르의 영화가 얼마나 포함되어 있는지를 평가합니다.

이러한 평가를 통해 제안된 모델이 기존의 추천 시스템보다 더 나은 성능을 발휘하는지를 검증합니다.

---




This paper proposes a knowledge graph-based semantic contrastive learning approach to improve content recommendation systems. The method employs a hybrid multi-task learning approach that simultaneously learns from user-item interactions and item-item interactions.

#### Training Data and Test Data

1. **Training Data**:
   - **Movie Dataset (MovieLens-20M)**: Contains approximately 20 million user ratings, with synopsis information for each movie collected from the TMDB database. For example, the synopsis for the movie "Despicable Me" is transformed into a format like "The genre(s) of the film is/are Animation, Comedy."
   - **Book Dataset (Book-Crossing)**: Contains about 1 million user ratings, with metadata for each book sourced from Goodreads and Google Books. For example, the synopsis for the book "Harry Potter" is transformed into a format like "A book title is Harry Potter (released_year). The genre(s) of the book is/are Fantasy."

2. **Test Data**:
   - The test data is structured similarly to the training data but includes a 'cold-start' user group with minimal user-item interactions. This group is defined as the bottom 1% of users with the least interactions.

#### Specific Tasks

- **Recommendation Performance Evaluation**:
  - **Click-Through Rate (CTR)**: Measures how often recommended content is clicked. For instance, it calculates the ratio of movies clicked from a recommended list.
  - **Recall@K**: Measures the proportion of actual preferred content among the K recommended items. For example, if a user prefers 3 out of 10 recommended movies, Recall@10 would be 0.3.
  - **NDCG@K**: Evaluates the quality of the ranking of recommended content. The higher the rank of a recommendation that matches the user's preference, the higher the NDCG score.

- **Diversity Evaluation**:
  - **Inter-list Diversity**: Measures the diversity between lists of recommended content for different users. For example, it assesses the similarity between content lists recommended to different users.
  - **Intra-list Diversity**: Measures the diversity within the list of recommended content for the same user. For example, it evaluates how many different genres are included in a list of movies recommended to a single user.

These evaluations validate whether the proposed model outperforms existing recommendation systems.

<br/>
# 요약
이 논문에서는 지식 그래프와 콘텐츠 기반 대조 학습을 활용하여 추천 시스템의 성능을 향상시키는 하이브리드 다중 작업 학습 접근 방식을 제안합니다. 실험 결과, 제안된 방법이 기존 모델보다 추천 성능, 개인화, 다양성 및 임계 사용자(Cold-start user) 상황에서의 성능을 모두 개선함을 보여주었습니다. 예를 들어, 영화 추천에서 제안된 모델은 NDCG@100에서 기존 모델보다 더 적은 성능 저하를 보였습니다.

---

This paper proposes a hybrid multi-task learning approach that leverages knowledge graphs and content-based contrastive learning to enhance the performance of recommendation systems. Experimental results demonstrate that the proposed method improves recommendation performance, personalization, diversity, and performance in cold-start user scenarios compared to baseline models. For instance, in movie recommendations, the proposed model shows less performance degradation in NDCG@100 compared to the existing model.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: Cold-start 시나리오에서의 성능 감소를 보여줍니다. 이 그림은 추천 시스템이 사용자-콘텐츠 상호작용이 적은 사용자에게서 성능이 어떻게 감소하는지를 시각적으로 나타냅니다. 이는 추천 시스템의 설계에서 cold-start 문제를 해결하는 것이 얼마나 중요한지를 강조합니다.
   - **Figure 2**: 제안된 모델의 구조를 보여줍니다. 이 그림은 지식 그래프와 다중 목표를 가진 모델의 상호작용을 설명하며, 사용자-콘텐츠 상호작용 손실과 콘텐츠 기반 대조 손실을 함께 최적화하는 방법을 나타냅니다.
   - **Figure 3**: 긍정적/부정적 샘플링 프로세스를 설명합니다. 이 그림은 콘텐츠의 메타데이터를 기반으로 긍정적 및 부정적 샘플을 선택하는 방법을 보여줍니다.
   - **Figure 4**: 훈련된 콘텐츠 및 사용자 임베딩을 사용하는 추론 프로세스를 설명합니다. 이 그림은 사용자와 콘텐츠 간의 내적 곱을 통해 추천을 생성하는 방법을 보여줍니다.
   - **Figure 5**: 사용자 활동 수준에 따른 모델 성능 비교를 보여줍니다. 이 그림은 사용자-콘텐츠 상호작용 수에 따라 추천 성능이 어떻게 달라지는지를 시각적으로 나타냅니다.
   - **Figure 6**: 임베딩 품질을 비교합니다. 이 그림은 다양한 모델의 정렬 및 균일성을 보여주며, 임베딩 품질이 추천 성능에 미치는 영향을 강조합니다.

2. **테이블**
   - **Table 1**: 두 데이터셋의 기본 통계 및 하이퍼파라미터 설정을 보여줍니다. 이 표는 MovieLens-20M과 Book-Crossing 데이터셋의 사용자 수, 콘텐츠 수, 사용자-콘텐츠 상호작용 수 등을 나열하여 실험의 기초를 제공합니다.
   - **Table 2**: AUC 및 F1-score 성능을 비교합니다. 이 표는 제안된 모델이 기존 모델보다 성능이 우수하다는 것을 보여줍니다.
   - **Table 3**: Recall@K 및 NDCG@K 성능을 비교합니다. 이 표는 제안된 모델이 추천 품질을 향상시키는 데 효과적임을 나타냅니다.
   - **Table 4**: 추천의 다양성을 평가합니다. 이 표는 제안된 모델이 추천 목록의 다양성을 높이는 데 기여함을 보여줍니다.
   - **Table 5**: LLaMA로 생성된 텍스트와 인간이 생성한 텍스트를 사용한 모델의 성능을 비교합니다. 이 표는 생성 AI 모델이 추천 성능에 미치는 영향을 평가합니다.

3. **어펜딕스**
   - 어펜딕스는 추가적인 실험 결과나 데이터, 코드 등을 포함할 수 있으며, 연구의 재현성을 높이는 데 기여합니다. 연구의 세부 사항이나 추가적인 분석을 제공하여 독자가 연구 결과를 더 깊이 이해할 수 있도록 돕습니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: Shows the performance decline in cold-start scenarios. This figure visually represents how the recommendation system's performance decreases for users with limited user-content interactions, emphasizing the importance of addressing the cold-start problem in recommendation system design.
   - **Figure 2**: Illustrates the structure of the proposed model. This figure explains the interactions of the knowledge graph and the multi-objective model, depicting how user-content interaction loss and content-based contrastive loss are optimized together.
   - **Figure 3**: Describes the positive/negative sampling process. This figure shows how positive and negative samples are selected based on the metadata of the content.
   - **Figure 4**: Explains the inference process using trained content and user embeddings. This figure demonstrates how recommendations are generated through the inner product between users and content.
   - **Figure 5**: Displays a comparison of model performance across different user activity levels. This figure visually represents how recommendation performance varies based on the number of user-content interactions.
   - **Figure 6**: Compares the quality of embeddings. This figure shows the alignment and uniformity of various models' embeddings, highlighting the impact of embedding quality on recommendation performance.

2. **Tables**
   - **Table 1**: Presents basic statistics and hyperparameter settings for the two datasets. This table lists the number of users, contents, and user-content interactions for the MovieLens-20M and Book-Crossing datasets, providing a foundation for the experiments.
   - **Table 2**: Compares AUC and F1-score performance. This table shows that the proposed model outperforms the baseline model.
   - **Table 3**: Compares Recall@K and NDCG@K performance. This table indicates that the proposed model is effective in enhancing recommendation quality.
   - **Table 4**: Evaluates the diversity of recommendations. This table shows that the proposed model contributes to increasing the diversity of the recommendation lists.
   - **Table 5**: Compares the performance of models using LLaMA-generated text and human-generated text. This table assesses the impact of generative AI models on recommendation performance.

3. **Appendix**
   - The appendix may include additional experimental results, data, or code, contributing to the reproducibility of the research. It provides detailed information or further analysis to help readers gain a deeper understanding of the research findings.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{kim2024improving,
  title={Improving Content Recommendation: Knowledge Graph-Based Semantic Contrastive Learning for Diversity and Cold-Start Users},
  author={Yejin Kim and Scott Rome and Kevin Foley and Mayur Nankani and Rimon Melamed and Javier Morales and Abhay Yadav and Maria Peifer and Sardar Hamidian and H. Howie Huang},
  booktitle={Proceedings of the LREC-COLING 2024},
  pages={8743--8755},
  year={2024},
  publisher={ELRA Language Resource Association},
  note={CC BY-NC 4.0}
}
```

### 시카고 스타일

Yejin Kim, Scott Rome, Kevin Foley, Mayur Nankani, Rimon Melamed, Javier Morales, Abhay Yadav, Maria Peifer, Sardar Hamidian, and H. Howie Huang. 2024. "Improving Content Recommendation: Knowledge Graph-Based Semantic Contrastive Learning for Diversity and Cold-Start Users." In *Proceedings of the LREC-COLING 2024*, 8743–8755. ELRA Language Resource Association. CC BY-NC 4.0.
