---
layout: post
title:  "[2025]TRIAL: Token Relations and Importance Aware Late-interaction for Accurate Text Retrieval"
date:   2025-12-01 02:07:01 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: TRIAL은 토큰 관계와 중요성을 고려한 지연 상호작용 모델로, 문서 검색의 정확성을 향상시키기 위해 토큰 간의 관계를 명시적으로 모델링하고 쿼리 토큰의 중요성을 가중치로 반영한다.
-> 토큰간의 릴레이션을 명시적으로 추가  
(토큰별 max-sim 합이 멀티토큰 표현/불용어 때문에 이상해지는 문제를 지적, token importance weight + token 간 relation score를 추가)    
(phrase-level 매칭 + 문맥 dependency를 더 잘 반영하는 스코어링 함수 제안)     
(ColBERT 구조는 유지하면서 스코어 정의만 더 리치하게 만듬)  

token importance weight + token 간 relation score를 추가

짧은 요약(Abstract) :


이 논문에서는 TRIAL(Token Relations and Importance Aware Late-interaction)이라는 새로운 텍스트 검색 모델을 제안합니다. 기존의 늦은 상호작용 기반 다중 벡터 검색 시스템은 토큰 수준의 유사도 점수를 단순히 합산하는 방식에 의존하여, 의미 단위의 토큰화로 인한 부정확한 관련성 추정과 저내용 단어의 영향을 초래합니다. TRIAL은 이러한 문제를 해결하기 위해 토큰 간의 관계와 중요성을 명시적으로 모델링하여 관련성 점수를 향상시킵니다. 세 가지 널리 사용되는 벤치마크에서의 광범위한 실험 결과, TRIAL은 MSMARCO에서 nDCG@10 점수 46.3을 기록하며, BEIR와 LoTTE Search에서 각각 평균 nDCG@10 점수 51.09와 72.15를 달성하여 최첨단 정확도를 보여줍니다. TRIAL은 뛰어난 정확도를 유지하면서도 기존의 늦은 상호작용 방법들과 비교해 경쟁력 있는 검색 속도를 유지하여 대규모 텍스트 검색에 실용적인 솔루션이 됩니다.



This paper proposes a new text retrieval model called TRIAL (Token Relations and Importance Aware Late-interaction). Existing late-interaction based multi-vector retrieval systems rely on a naive summation of token-level similarity scores, which often leads to inaccurate relevance estimation due to the tokenization of semantic units and the influence of low-content words. To address these challenges, TRIAL explicitly models token relations and token importance in relevance scoring. Extensive experiments on three widely used benchmarks show that TRIAL achieves state-of-the-art accuracy, with an nDCG@10 score of 46.3 on MSMARCO, and average nDCG@10 scores of 51.09 and 72.15 on BEIR and LoTTE Search, respectively. With superior accuracy, TRIAL maintains competitive retrieval speed compared to existing late-interaction methods, making it a practical solution for large-scale text retrieval.


* Useful sentences :

* Useful information :

토큰별 max-sim 합 : query 각 토큰에 대해 doc에서 제일 비슷한 토큰을 하나 골라 max 유사도를 구하고, 그걸 query구성하는 토큰 별 max 유사도를 전부 합친 것!  

문서/쿼리의 각 토큰마다 벡터를 하나씩 뽑은 다음,  
그 토큰들끼리 유사도를 계산해서 점수를 만듬,  
스코어는 각 쿼리 토큰마다, 그 토큰과 제일 비슷한 문서의 토큰을 찾고 (max), 그 max 유사도들을 다 더한 것 (sum)  
즉, 더 fine-grained임  

{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



TRIAL(토큰 관계 및 중요도 인식 지연 상호작용 모델)은 정보 검색을 위한 혁신적인 접근 방식을 제안합니다. 이 모델은 두 가지 주요 구성 요소인 토큰 관계 모델링과 토큰 중요도 가중치를 통해 텍스트 검색의 정확성을 향상시킵니다.

1. **모델 아키텍처**:
   TRIAL은 지연 상호작용 기반의 다중 벡터 검색 시스템으로, 쿼리와 문서의 토큰을 독립적으로 인코딩하여 다중 벡터 표현을 생성합니다. 이 모델은 각 쿼리 토큰과 문서 토큰 간의 유사성을 계산하는 데 있어, 단순한 합산 대신 토큰 간의 관계를 명시적으로 모델링합니다. 이를 통해 다중 토큰 단어 및 구문을 보다 효과적으로 처리할 수 있습니다.

2. **토큰 관계 점수**:
   TRIAL은 쿼리와 문서의 토큰 간의 관계를 모델링하여, 의미적 의존성을 캡처합니다. 이를 위해 쿼리 측의 바이그램을 사용하여 각 토큰의 이전 토큰과의 관계를 계산합니다. 이 관계는 두 개의 토큰 임베딩을 결합한 후, 다층 퍼셉트론(MLP)을 통해 처리되어 최종 유사성 점수에 기여합니다.

3. **토큰 중요도 가중치**:
   TRIAL은 쿼리 내의 각 토큰의 중요도를 평가하여, 중요도가 높은 토큰에 더 많은 가중치를 부여합니다. 이를 위해 두 개의 레이어로 구성된 신경망을 사용하여 각 토큰의 중요도를 학습합니다. 이 가중치 메커니즘은 쿼리의 핵심 토큰을 강조하여 문서의 관련성을 보다 정확하게 평가할 수 있도록 합니다.

4. **훈련 방법**:
   TRIAL은 대조 학습을 통해 훈련됩니다. KL 발산 손실과 교차 엔트로피 손실을 결합하여 모델이 관련 문서와 비관련 문서를 구별하도록 학습합니다. 또한, 토큰 중요도 가중치를 최적화하기 위해 정규화 손실을 도입하여, 모델이 의미 있는 토큰에 집중할 수 있도록 합니다.

5. **실험 및 평가**:
   TRIAL은 MSMARCO, BEIR, LoTTE와 같은 다양한 벤치마크 데이터셋에서 평가되었으며, nDCG@10 및 Success@5와 같은 메트릭을 사용하여 성능을 측정합니다. 실험 결과, TRIAL은 기존의 지연 상호작용 모델보다 우수한 성능을 보이며, 대규모 텍스트 검색에 실용적인 솔루션으로 자리 잡았습니다.




TRIAL (Token Relations and Importance Aware Late-interaction) proposes an innovative approach for information retrieval. This model enhances the accuracy of text retrieval through two main components: token relation modeling and token importance weighting.

1. **Model Architecture**:
   TRIAL is a late-interaction based multi-vector retrieval system that independently encodes tokens from queries and documents to generate multi-vector representations. Instead of relying on a naive summation of token-level similarities, this model explicitly models the relationships between tokens, allowing for more effective handling of multi-token words and phrases.

2. **Token Relation Score**:
   TRIAL models the relationships between tokens in queries and documents to capture semantic dependencies. To achieve this, it uses query-side bigrams to compute the relationship between each token and its preceding token. This relationship is processed through a two-layer Multilayer Perceptron (MLP) after concatenating the embeddings of the two tokens, contributing to the final similarity score.

3. **Token Importance Weight**:
   TRIAL evaluates the importance of each token within a query, assigning higher weights to more significant tokens. A two-layer neural network is employed to learn the importance of each token. This weighting mechanism emphasizes critical tokens in the query, allowing for more accurate relevance scoring of documents.

4. **Training Method**:
   TRIAL is trained using contrastive learning. It combines KL divergence loss and cross-entropy loss to enable the model to distinguish between relevant and irrelevant documents. Additionally, a regularization loss is introduced to optimize the token importance weights, ensuring that the model focuses on meaningful tokens.

5. **Experiments and Evaluation**:
   TRIAL is evaluated on various benchmark datasets, including MSMARCO, BEIR, and LoTTE, using metrics such as nDCG@10 and Success@5 to measure performance. Experimental results demonstrate that TRIAL outperforms existing late-interaction models, establishing itself as a practical solution for large-scale text retrieval.


<br/>
# Results



TRIAL 모델은 MSMARCO와 BEIR, LoTTE 데이터셋을 사용하여 다양한 경쟁 모델과 비교하여 성능을 평가했습니다. 이 연구에서는 여러 가지 메트릭을 사용하여 모델의 정확도를 측정했습니다. 주요 메트릭으로는 nDCG@10과 Success@5가 있습니다. nDCG@10은 상위 10개의 결과에서의 정답 문서의 순위를 평가하는 지표로, 높은 값일수록 더 나은 성능을 나타냅니다. Success@5는 상위 5개의 결과 중에 적어도 하나의 정답 문서가 포함되어 있는지를 평가하는 지표입니다.

#### 경쟁 모델
TRIAL은 다음과 같은 다양한 모델과 비교되었습니다:
1. **Sparse Retrieval**: BM25, SPLADEv2, SPLADE++와 같은 전통적인 희소 검색 모델.
2. **Full-Interaction Dense Retrieval**: MonoBERT, CELI와 같은 모델로, 쿼리와 문서 간의 전체 상호작용을 통해 높은 정확도를 달성합니다.
3. **Representation-Based Dense Retrieval**: DPR, ANCE, GTR와 같은 모델로, 쿼리와 문서를 고정된 표현으로 매핑하여 효율적인 검색을 수행합니다.
4. **Late-Interaction Dense Retrieval**: ColBERT, ColBERTv2, XTR, LITE와 같은 모델로, 토큰 수준의 표현을 활용하여 효율성과 정확성을 동시에 추구합니다.

#### 실험 결과
TRIAL 모델은 MSMARCO 데이터셋에서 nDCG@10 점수 46.3을 기록하며, 이는 기존의 다른 모델들보다 높은 성능을 보였습니다. BEIR 데이터셋에서도 TRIAL은 평균 nDCG@10 점수 51.09를 기록하며, 여러 개별 데이터셋에서 최상의 성능을 달성했습니다. 특히, TRIAL은 SPLADE++와 같은 희소 검색 모델의 이전 최고 성능을 초과했습니다.

LoTTE 데이터셋에 대한 평가에서도 TRIAL은 검색 벤치마크에서 평균적으로 가장 높은 정확도를 기록했습니다. 그러나 포럼 벤치마크에서는 TRIAL이 기술 관련 데이터셋에서만 최고 점수를 기록했으며, 나머지 데이터셋에서는 ColBERTv2가 더 높은 정확도를 보였습니다. 이는 검색 쿼리와 포럼 쿼리 간의 스타일과 의도의 차이에서 기인한 것으로 보입니다.

#### 결론
TRIAL 모델은 기존의 모델들에 비해 뛰어난 성능을 보여주었으며, 특히 nDCG@10과 Success@5 메트릭에서 두드러진 결과를 나타냈습니다. 이러한 결과는 TRIAL이 텍스트 검색에서의 정확도를 높이는 데 기여할 수 있음을 시사합니다.

---



The TRIAL model was evaluated against various competitive models using the MSMARCO, BEIR, and LoTTE datasets. The performance of the models was measured using several metrics, with the primary metrics being nDCG@10 and Success@5. nDCG@10 evaluates the ranking of relevant documents within the top 10 results, with higher values indicating better performance. Success@5 assesses whether at least one relevant document appears in the top 5 results.

#### Competitive Models
TRIAL was compared against a diverse set of models, including:
1. **Sparse Retrieval**: Traditional models like BM25, SPLADEv2, and SPLADE++.
2. **Full-Interaction Dense Retrieval**: Models such as MonoBERT and CELI, which achieve high accuracy by leveraging full interaction between queries and documents.
3. **Representation-Based Dense Retrieval**: Models like DPR, ANCE, and GTR, which efficiently map queries and documents into fixed representations for scalable retrieval.
4. **Late-Interaction Dense Retrieval**: Models such as ColBERT, ColBERTv2, XTR, and LITE, which utilize token-level representations to balance efficiency and accuracy.

#### Experimental Results
The TRIAL model achieved an nDCG@10 score of 46.3 on the MSMARCO dataset, outperforming other existing models. On the BEIR dataset, TRIAL recorded an average nDCG@10 score of 51.09, achieving state-of-the-art performance across several individual datasets. Notably, TRIAL surpassed the previous best performance achieved by the sparse retrieval method SPLADE++.

In evaluations on the LoTTE dataset, TRIAL also achieved the highest average accuracy across the search benchmark. However, in the forum benchmark, TRIAL only achieved the highest score for the technology-related dataset, while ColBERTv2 showed higher accuracy on the remaining datasets. This performance disparity is likely due to differences in query patterns between search queries and forum queries.

#### Conclusion
The TRIAL model demonstrated superior performance compared to existing models, particularly in the nDCG@10 and Success@5 metrics. These results suggest that TRIAL can significantly contribute to improving accuracy in text retrieval tasks.


<br/>
# 예제



TRIAL 모델은 정보 검색을 위한 텍스트 검색 시스템으로, 주어진 쿼리와 문서 간의 관련성을 평가하는 데 중점을 둡니다. 이 모델은 두 가지 주요 요소인 토큰 관계와 중요성 가중치를 활용하여 정확한 관련성 점수를 계산합니다. 다음은 TRIAL 모델의 훈련 및 테스트 과정에서 사용되는 데이터와 작업에 대한 구체적인 설명입니다.

#### 1. 데이터셋
- **훈련 데이터**: MSMARCO 데이터셋을 사용합니다. 이 데이터셋은 약 880만 개의 웹 페이지에서 수집된 문서와 약 7,000개의 실제 쿼리로 구성되어 있습니다. 각 쿼리는 사용자가 검색 엔진에 입력할 수 있는 질문 형식입니다.
- **테스트 데이터**: BEIR와 LoTTE 데이터셋을 사용하여 모델의 일반화 성능을 평가합니다. BEIR 데이터셋은 다양한 정보 검색 작업을 포함하는 13개의 데이터셋으로 구성되어 있으며, LoTTE 데이터셋은 자연어 검색 쿼리와 긴 꼬리 주제를 다룹니다.

#### 2. 작업(Task)
- **쿼리-문서 관련성 평가**: 모델은 주어진 쿼리에 대해 문서의 관련성을 평가합니다. 예를 들어, 쿼리 "영화 '바다 너머'의 노래를 부른 사람은 누구인가?"에 대해 모델은 관련 문서(예: "바다 너머 영화")를 높은 점수로 평가해야 합니다.
- **정확도 측정**: 모델의 성능은 nDCG@10과 Success@5와 같은 메트릭을 사용하여 평가됩니다. nDCG@10은 상위 10개의 문서의 관련성을 기반으로 한 점수이며, Success@5는 상위 5개 문서 중 하나라도 관련 문서가 포함되어 있는지를 평가합니다.

#### 3. 훈련 및 테스트 과정
- **훈련 과정**: 모델은 주어진 쿼리와 관련된 문서 쌍을 입력으로 받아, 각 문서의 관련성 점수를 예측하도록 학습합니다. 이 과정에서 KL-발산 손실과 교차 엔트로피 손실을 사용하여 모델의 성능을 최적화합니다.
- **테스트 과정**: 훈련된 모델은 BEIR와 LoTTE 데이터셋의 쿼리를 입력으로 받아, 각 쿼리에 대해 문서의 관련성 점수를 계산합니다. 이 점수를 기반으로 문서를 순위별로 정렬하고, 최종적으로 nDCG@10과 Success@5 메트릭을 통해 성능을 평가합니다.




The TRIAL model is a text retrieval system focused on evaluating the relevance between a given query and documents. This model utilizes two main components: token relations and importance weights, to compute accurate relevance scores. Below is a detailed explanation of the data and tasks used in the training and testing processes of the TRIAL model.

#### 1. Datasets
- **Training Data**: The MSMARCO dataset is used, which consists of approximately 8.8 million documents collected from the web and around 7,000 real-world queries. Each query represents a question that a user might input into a search engine.
- **Testing Data**: The BEIR and LoTTE datasets are used to evaluate the model's generalization performance. The BEIR dataset includes 13 datasets covering various information retrieval tasks, while the LoTTE dataset focuses on natural language search queries and long-tail topics.

#### 2. Task
- **Query-Document Relevance Evaluation**: The model evaluates the relevance of documents for a given query. For example, for the query "Who sang the songs in the movie 'Beyond the Sea'?", the model should score relevant documents (e.g., "Beyond the Sea film") highly.
- **Accuracy Measurement**: The model's performance is evaluated using metrics such as nDCG@10 and Success@5. nDCG@10 is a score based on the relevance of the top 10 documents, while Success@5 assesses whether at least one relevant document is included in the top 5 documents.

#### 3. Training and Testing Process
- **Training Process**: The model takes pairs of queries and relevant documents as input and learns to predict the relevance scores for each document. During this process, KL-divergence loss and cross-entropy loss are used to optimize the model's performance.
- **Testing Process**: The trained model takes queries from the BEIR and LoTTE datasets as input and computes relevance scores for each document. Based on these scores, documents are ranked, and the final performance is evaluated using the nDCG@10 and Success@5 metrics.

<br/>
# 요약


TRIAL은 토큰 관계와 중요성을 고려한 지연 상호작용 모델로, 문서 검색의 정확성을 향상시키기 위해 토큰 간의 관계를 명시적으로 모델링하고 쿼리 토큰의 중요성을 가중치로 반영한다. 실험 결과, TRIAL은 MSMARCO에서 nDCG@10 점수 46.3을 기록하며, BEIR와 LoTTE Search에서 각각 평균 nDCG@10 점수 51.09와 72.15를 달성하여 기존 방법들보다 우수한 성능을 보였다. 예를 들어, TRIAL은 "movie beyond the sea"와 같은 다중 토큰 구문을 처리할 때, 관련 문서의 순위를 올리는 데 효과적임을 보여주었다.

---

TRIAL is a late-interaction model that considers token relations and importance to enhance the accuracy of document retrieval by explicitly modeling relationships between tokens and weighting the importance of query tokens. Experimental results show that TRIAL achieves an nDCG@10 score of 46.3 on MSMARCO and average nDCG@10 scores of 51.09 and 72.15 on BEIR and LoTTE Search, respectively, outperforming existing methods. For instance, TRIAL effectively improves the ranking of relevant documents when handling multi-token phrases like "movie beyond the sea."

<br/>
# 기타
### 한국어 설명

#### 다이어그램 및 피규어
1. **토큰 가중치 분석 (Figure 2)**: TRIAL 모델에서 다양한 품사 태그에 대한 평균 토큰 가중치를 보여줍니다. 의미 있는 정보가 많은 토큰(예: 감탄사, 형용사, 명사, 고유명사)은 높은 가중치를 부여받고, 기능적 토큰(예: 보조 동사, 한정사)은 낮은 가중치를 받습니다. 이는 TRIAL이 의미 있는 용어를 우선시하는 능력을 보여줍니다.

2. **토큰 점수 비교 (Figure 3)**: TRIAL과 ColBERTv2 간의 토큰 점수를 비교합니다. TRIAL은 의미 있는 토큰에 대해 더 높은 점수를 부여하고, 덜 관련된 토큰의 영향을 최소화합니다. 이는 TRIAL이 쿼리-문서 관련성 계산에서 더 정확한 결과를 도출할 수 있도록 합니다.

#### 테이블
1. **MSMARCO 및 BEIR 데이터셋에서의 검색 정확도 (Table 1)**: TRIAL은 다양한 검색 방법에서 최첨단 성능을 달성했습니다. 특히 MSMARCO에서 nDCG@10 점수는 46.3으로, BEIR 데이터셋에서도 평균 51.09를 기록했습니다. 이는 TRIAL이 기존의 방법들보다 더 정확한 검색 결과를 제공함을 나타냅니다.

2. **LoTTE 데이터셋에서의 검색 정확도 (Table 2 & Table 3)**: TRIAL은 LoTTE 검색 데이터셋에서 평균적으로 가장 높은 정확도를 기록했습니다. 그러나 포럼 데이터셋에서는 일부 데이터셋에서 ColBERTv2에 비해 낮은 성능을 보였습니다. 이는 쿼리 패턴의 차이로 인한 것으로 보입니다.

3. **후보 검색 정확도 및 최종 검색 정확도 (Table 8)**: TRIAL의 후보 검색 정확도는 PLAID 알고리즘을 통해 개선되었으며, 이는 최종 검색 정확도에도 긍정적인 영향을 미쳤습니다. 오라클 후보를 사용할 경우 nDCG@10 점수가 소폭 향상되었지만, 후보 검색의 품질이 최종 검색 정확도에 미치는 영향은 제한적임을 보여줍니다.

#### 어펜딕스
- **실험 설정 (A.1)**: TRIAL은 MSMARCO, BEIR, LoTTE 데이터셋을 사용하여 평가되었습니다. 각 데이터셋의 특성과 평가 지표(nDCG@10, Success@5)에 대한 설명이 포함되어 있습니다.
- **후보 검색 (A.3)**: 후보 검색의 중요성을 강조하며, TRIAL 모델이 후보 검색의 품질을 개선하기 위해 토큰 중요성을 통합한 방법을 설명합니다.





#### Diagrams and Figures
1. **Token Weight Analysis (Figure 2)**: This figure shows the average token weights for different part-of-speech tags in the TRIAL model. Tokens with significant semantic information (e.g., interjections, adjectives, nouns, proper nouns) receive higher weights, while functional tokens (e.g., auxiliary verbs, determiners) receive lower weights. This demonstrates TRIAL's ability to prioritize meaningful terms.

2. **Token Score Comparison (Figure 3)**: This figure compares the token scores between TRIAL and ColBERTv2. TRIAL amplifies the relevance of semantically important tokens while suppressing less relevant ones. This allows TRIAL to derive more accurate results in query-document relevance computation.

#### Tables
1. **Retrieval Accuracy on MSMARCO and BEIR Datasets (Table 1)**: TRIAL achieved state-of-the-art performance across various retrieval methods. Notably, it recorded an nDCG@10 score of 46.3 on MSMARCO and an average of 51.09 on BEIR datasets. This indicates that TRIAL provides more accurate search results compared to existing methods.

2. **Retrieval Accuracy on LoTTE Datasets (Table 2 & Table 3)**: TRIAL achieved the highest average accuracy on the LoTTE search dataset. However, in the forum dataset, it performed lower than ColBERTv2 on some datasets. This is likely due to differences in query patterns.

3. **Candidate Retrieval Accuracy and End-to-End Retrieval Accuracy (Table 8)**: The candidate retrieval accuracy of TRIAL improved through the PLAID algorithm, positively impacting the end-to-end retrieval accuracy. Using oracle candidates resulted in a slight increase in nDCG@10 scores, but the quality of candidate retrieval had a limited impact on final retrieval accuracy.

#### Appendix
- **Experimental Setting (A.1)**: TRIAL was evaluated using the MSMARCO, BEIR, and LoTTE datasets. The characteristics of each dataset and the evaluation metrics (nDCG@10, Success@5) are described.
- **Candidate Retrieval (A.3)**: This section emphasizes the importance of candidate retrieval and explains how the TRIAL model integrates token importance to improve the quality of candidate retrieval.

<br/>
# refer format:


### BibTeX 형식

```bibtex
@inproceedings{Kang2025,
  author    = {Hyukkyu Kang and Injung Kim and Wook-Shin Han},
  title     = {TRIAL: Token Relations and Importance Aware Late-interaction for Accurate Text Retrieval},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages     = {16876--16889},
  year      = {2025},
  month     = {November},
  publisher = {Association for Computational Linguistics},
  address   = {Online}
}
```

### 시카고 스타일

Hyukkyu Kang, Injung Kim, and Wook-Shin Han. "TRIAL: Token Relations and Importance Aware Late-interaction for Accurate Text Retrieval." In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, 16876–16889. Online: Association for Computational Linguistics, November 2025.
