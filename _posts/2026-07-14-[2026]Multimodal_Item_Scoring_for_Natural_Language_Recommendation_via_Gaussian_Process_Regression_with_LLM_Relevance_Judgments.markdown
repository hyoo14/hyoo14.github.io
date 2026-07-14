---
layout: post
title:  "[2026]Multimodal Item Scoring for Natural Language Recommendation via Gaussian Process Regression with LLM Relevance Judgments"
date:   2026-07-14 01:00:23 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 자연어 추천(NLRec)을 위한 GPR-LLM 방법을 제안하며, 이는 여러 LLM-판단 기준을 활용하여 가우시안 프로세스 회귀(GPR)를 통해 아이템의 관련성을 추정한다.


짧은 요약(Abstract) :


이 논문의 초록에서는 자연어 추천(Natural Language Recommendation, NLRec) 시스템이 사용자 요청과 아이템 설명 간의 관련성을 기반으로 아이템 제안을 생성하는 방법을 다룹니다. 기존의 NLRec 접근 방식은 밀집 검색(Dense Retrieval, DR)을 사용하여 사용자 요청 임베딩과 관련된 패시지 임베딩 간의 내적을 통해 아이템의 관련성 점수를 계산합니다. 그러나 DR은 요청을 유일한 관련성 신호로 간주하여 단일 모드의 점수 함수를 생성하는데, 이는 실제 관련성을 반영하기에는 부족합니다. 이 논문에서는 여러 LLM(대형 언어 모델) 판단을 기반으로 한 앵커 패시지를 사용하여 기본적인 관련성 함수를 추정하는 가우시안 프로세스 회귀(Gaussian Process Regression, GPR) 방법인 GPR-LLM을 제안합니다. 실험 결과, GPR-LLM은 DR, 크로스 인코더, 포인트와이즈 LLM 기반 관련성 점수와 같은 기존 방법들보다 최대 65% 더 높은 성능을 보였습니다.



The abstract of this paper discusses how Natural Language Recommendation (NLRec) systems generate item suggestions based on the relevance between user-issued requests and item descriptions. Existing NLRec approaches often use Dense Retrieval (DR) to compute item relevance scores through the inner product between user request embeddings and relevant passage embeddings. However, DR treats the request as the sole relevance signal, resulting in a unimodal scoring function that often fails to capture true relevance. This paper proposes GPR-LLM, which uses Gaussian Process Regression (GPR) to estimate the underlying relevance function from multiple LLM-judged anchor passages instead of relying solely on the request. Experimental results demonstrate that GPR-LLM consistently outperforms baseline methods, including DR, cross-encoder, and pointwise LLM-based relevance scoring, by up to 65%.


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


GPR-LLM(Gaussian Process Regression with LLM Relevance Judgments)은 자연어 추천(Natural Language Recommendation, NLRec) 시스템을 위한 새로운 접근 방식으로, 여러 LLM(대형 언어 모델)에서 판단한 앵커 패시지의 정보를 활용하여 기본적인 관련성 함수를 추정하는 방법입니다. 이 방법은 기존의 Dense Retrieval(DR) 방식이 단일 쿼리 중심의 관련성 신호에 의존하는 것과는 달리, 여러 관련성 신호를 결합하여 다중 관련성 모드를 포착할 수 있도록 설계되었습니다.

#### 1. 모델 아키텍처
GPR-LLM은 Gaussian Process Regression(GPR)을 기반으로 하며, 이는 비모수적 베이지안 회귀 방법입니다. GPR은 주어진 입력에 대해 잠재적인 함수 값을 추정하는 데 사용되며, 이 함수는 가우시안 프로세스 사전 분포를 따릅니다. GPR-LLM에서는 쿼리와 관련된 패시지의 임베딩을 입력으로 사용하여, 각 패시지의 관련성 점수를 예측합니다.

#### 2. 데이터 샘플링
GPR-LLM은 전체 패시지 집합에서 R개의 패시지를 샘플링하여 LLM의 관련성 판단을 받습니다. 이 샘플링 과정은 ϵ-탐욕적(ϵ-greedy) 전략을 사용하여, 상위 DR 점수를 가진 패시지와 무작위로 선택된 패시지를 혼합하여 구성됩니다. 이를 통해 다양한 관련성 패턴을 포착할 수 있습니다.

#### 3. LLM 관련성 판단
샘플링된 패시지에 대해 LLM을 사용하여 관련성 점수를 부여합니다. 이 점수는 LLM이 쿼리와 패시지 간의 관계를 평가하여 생성한 것입니다. LLM의 판단 결과는 GPR의 학습에 사용되며, 이를 통해 GPR은 패시지의 관련성 점수를 추정합니다.

#### 4. GPR 모델 학습
GPR 모델은 쿼리와 관련된 패시지의 임베딩을 입력으로 받아, 이들 간의 유사성을 측정하는 커널 함수를 사용하여 학습됩니다. GPR-LLM에서는 RBF(Radial Basis Function) 커널을 사용하여, 각 앵커 패시지가 주변 패시지에 미치는 영향을 조절합니다. 이를 통해 GPR-LLM은 다중 관련성 모드를 효과적으로 포착할 수 있습니다.

#### 5. 최종 점수 집계
각 패시지의 관련성 점수는 GPR의 후방 평균을 사용하여 계산되며, 이 점수는 최종적으로 아이템 수준의 점수로 집계됩니다. 이 과정에서 상위 T개의 패시지를 선택하여 평균 또는 최대값을 사용하여 최종 점수를 계산합니다.

GPR-LLM은 이러한 방식으로 다중 관련성 모드를 포착하고, 기존의 DR 및 LLM 기반 방법들보다 더 나은 성능을 보여줍니다.

---




GPR-LLM (Gaussian Process Regression with LLM Relevance Judgments) is a novel approach for Natural Language Recommendation (NLRec) systems that utilizes information from multiple LLM (Large Language Model) judged anchor passages to estimate the underlying relevance function. This method is designed to capture multiple relevance modes by combining various relevance signals, in contrast to the existing Dense Retrieval (DR) methods that rely on a single query-centered relevance signal.

#### 1. Model Architecture
GPR-LLM is based on Gaussian Process Regression (GPR), which is a non-parametric Bayesian regression method. GPR is used to estimate potential function values given inputs, and this function follows a Gaussian process prior distribution. In GPR-LLM, the embeddings of passages related to the query are used as inputs to predict the relevance scores of each passage.

#### 2. Data Sampling
GPR-LLM samples R passages from the entire set of passages to receive relevance judgments from the LLM. This sampling process employs an ϵ-greedy strategy, mixing top-ranked passages based on DR scores with randomly selected passages. This approach allows for capturing diverse relevance patterns.

#### 3. LLM Relevance Judgments
For the sampled passages, relevance scores are assigned using the LLM. These scores are generated based on the LLM's evaluation of the relationship between the query and the passages. The judgments from the LLM are then used in the training of the GPR, allowing it to estimate the relevance scores of the passages.

#### 4. GPR Model Training
The GPR model takes the embeddings of the passages related to the query as input and uses a kernel function to measure the similarity between them. In GPR-LLM, the Radial Basis Function (RBF) kernel is employed to control the influence of each anchor passage on nearby passages. This enables GPR-LLM to effectively capture multiple relevance modes.

#### 5. Final Score Aggregation
The relevance scores for each passage are calculated using the posterior mean of the GPR, which is then aggregated into item-level scores. In this process, the top T passages are selected, and the final score is computed using either the mean or maximum of these scores.

Through this methodology, GPR-LLM captures multiple relevance modes and demonstrates superior performance compared to traditional DR and LLM-based methods.


<br/>
# Results



이 논문에서는 GPR-LLM(Gaussian Process Regression with LLM Relevance Judgments)이라는 새로운 자연어 추천(NLRec) 방법을 제안하고, 이를 기존의 여러 경쟁 모델과 비교하여 성능을 평가하였다. GPR-LLM은 여러 LLM(대형 언어 모델)에서 판단한 앵커 패시지의 정보를 활용하여 다중 모드의 관련성을 포착하는 데 중점을 두고 있다.

#### 실험 데이터셋
연구에서는 네 가지 공개 벤치마크 데이터셋을 사용하였다:
1. **TravelDest**: 여행 도시 추천을 위한 데이터셋.
2. **POINTREC**: 포인트 오브 인터레스트 추천을 위한 데이터셋.
3. **TripAdvisor Hotel**: 호텔 추천을 위한 사용자 리뷰 데이터셋.
4. **Yelp Restaurant**: 레스토랑 추천을 위한 사용자 리뷰 데이터셋.

각 데이터셋은 복잡하고 다면적인 자연어 요청을 포함하고 있으며, 다양한 텍스트 패시지로 구성된 아이템을 나타낸다.

#### 경쟁 모델
GPR-LLM은 다음과 같은 여러 경쟁 모델과 비교되었다:
- **BM25**: 전통적인 정보 검색 모델로, 문서의 관련성을 점수화하는 데 사용된다.
- **Dense Retrieval (DR)**: 쿼리와 패시지 임베딩 간의 내적을 기반으로 관련성을 점수화하는 방법.
- **Cross-Encoder (CE)**: 쿼리-패시지 쌍을 공동으로 인코딩하여 세밀한 관련성 점수를 출력하는 모델.
- **Pointwise LLM-based Relevance Scoring (PW)**: LLM을 사용하여 쿼리-패시지 쌍의 관련성을 점수화하는 방법.

#### 평가 메트릭
성능 평가는 다음과 같은 메트릭을 사용하여 수행되었다:
- **Precision@K**: 상위 K개의 추천 아이템 중 실제 관련 아이템의 비율.
- **NDCG@K**: 추천 리스트의 품질을 평가하는 메트릭으로, 관련성 수준과 아이템 위치를 모두 고려한다.

#### 결과
GPR-LLM은 다양한 LLM 백본과 함께 실험되었으며, 다음과 같은 주요 결과를 보였다:
- GPR-LLM은 모든 데이터셋과 LLM 백본에서 기존의 모든 기준선 모델에 비해 일관되게 우수한 성능을 보였다.
- 특히, GPR-LLM은 점수화 예산이 동일한 경우, Pointwise LLM 기반 모델에 비해 최대 65%의 성능 향상을 달성하였다.
- GPR-LLM은 RBF 커널을 사용할 때, 다중 모드의 관련성을 더 효과적으로 포착하는 것으로 나타났다.

이러한 결과는 GPR-LLM이 복잡한 NLRec 데이터에서 다중 모드의 관련성을 포착하는 데 있어 효과적이고 효율적인 접근법임을 입증하였다.

---




In this paper, a novel Natural Language Recommendation (NLRec) method called GPR-LLM (Gaussian Process Regression with LLM Relevance Judgments) is proposed, and its performance is evaluated against several existing competitive models. GPR-LLM focuses on capturing multiple modes of relevance by leveraging information from multiple LLM (Large Language Model)-judged anchor passages.

#### Experimental Datasets
The study utilized four publicly available benchmark datasets:
1. **TravelDest**: A dataset for travel city recommendations.
2. **POINTREC**: A dataset for point-of-interest recommendations.
3. **TripAdvisor Hotel**: A dataset of user reviews for hotel recommendations.
4. **Yelp Restaurant**: A dataset of user reviews for restaurant recommendations.

Each dataset contains complex, multi-faceted natural language requests and represents items through diverse collections of textual passages.

#### Competitive Models
GPR-LLM was compared against several competitive models, including:
- **BM25**: A traditional information retrieval model used to score document relevance.
- **Dense Retrieval (DR)**: A method that scores relevance based on the inner product between query and passage embeddings.
- **Cross-Encoder (CE)**: A model that jointly encodes query-passage pairs to output fine-grained relevance scores.
- **Pointwise LLM-based Relevance Scoring (PW)**: A method that uses LLMs to score the relevance of query-passage pairs.

#### Evaluation Metrics
Performance evaluation was conducted using the following metrics:
- **Precision@K**: The proportion of actual relevant items among the top K recommended items.
- **NDCG@K**: A metric that assesses the quality of the recommendation list by considering both relevance levels and item positions.

#### Results
GPR-LLM was tested with various LLM backbones, yielding the following key findings:
- GPR-LLM consistently outperformed all baseline models across all datasets and LLM backbones.
- Notably, GPR-LLM achieved up to a 65% performance improvement over the Pointwise LLM-based model when using the same scoring budget.
- GPR-LLM demonstrated more effective capture of multimodal relevance when using the RBF kernel.

These results establish GPR-LLM as an effective and efficient approach for capturing multimodal relevance in complex NLRec data.


<br/>
# 예제



이 논문에서는 자연어 추천(Natural Language Recommendation, NLRec) 시스템을 위한 새로운 접근 방식인 GPR-LLM(Gaussian Process Regression with LLM Relevance Judgments)을 제안합니다. 이 시스템은 사용자가 입력한 자연어 요청과 관련된 아이템 설명을 기반으로 추천을 생성합니다. 기존의 NLRec 방법들은 주로 Dense Retrieval(DR) 기법을 사용하여 사용자 요청과 관련된 패시지의 내적을 통해 아이템의 관련성을 평가합니다. 그러나 DR은 요청을 유일한 관련성 신호로 간주하여 단일 모드의 점수 함수를 생성하는 경향이 있습니다. 이는 실제 관련성을 잘 반영하지 못할 수 있습니다.

GPR-LLM은 여러 LLM(대형 언어 모델)에서 판단한 앵커 패시지를 사용하여 기본적인 관련성 함수를 추정합니다. 이 방법은 다음과 같은 단계로 구성됩니다:

1. **후보 샘플링**: 전체 패시지 집합에서 ϵ-탐욕적 전략을 사용하여 R개의 패시지를 샘플링합니다. 이때, 상위 DR 점수를 가진 패시지와 무작위로 선택된 패시지를 혼합합니다.
   
2. **LLM 관련성 판단**: 샘플링된 각 패시지에 대해 LLM을 사용하여 관련성 점수를 부여합니다.

3. **GPR 적합**: LLM에서 얻은 관련성 점수를 기반으로 GPR 모델을 적합시킵니다.

4. **모든 패시지 점수화**: GPR의 후방 평균을 사용하여 모든 패시지에 대한 점수를 예측합니다.

5. **아이템 점수화**: 패시지 점수를 집계하여 최종 아이템 점수를 생성합니다.

#### 예시
- **입력**: 사용자가 "가족 친화적인 도시"라는 요청을 입력합니다.
- **샘플링**: 시스템은 관련성이 높은 패시지(예: "밴쿠버는 가족 친화적인 도시입니다.")와 무작위로 선택된 패시지(예: "샌디에고는 해변이 아름답습니다.")를 포함하여 R개의 패시지를 샘플링합니다.
- **LLM 판단**: 각 패시지에 대해 LLM이 관련성 점수를 부여합니다. 예를 들어, "밴쿠버는 가족 친화적인 도시입니다."는 3점, "샌디에고는 해변이 아름답습니다."는 1점을 받을 수 있습니다.
- **GPR 적합**: 이 점수를 기반으로 GPR 모델이 적합됩니다.
- **모든 패시지 점수화**: GPR 모델을 사용하여 모든 패시지에 대한 점수를 예측합니다.
- **아이템 점수화**: 최종적으로 관련성이 높은 아이템이 추천됩니다.




This paper proposes a new approach for Natural Language Recommendation (NLRec) systems called GPR-LLM (Gaussian Process Regression with LLM Relevance Judgments). This system generates recommendations based on user-issued natural language requests and the associated item descriptions. Existing NLRec methods primarily use Dense Retrieval (DR) techniques to compute item relevance scores based on the inner product between user requests and relevant passages. However, DR tends to treat the request as the sole relevance signal, resulting in a unimodal scoring function that may not accurately reflect true relevance.

GPR-LLM estimates the underlying relevance function from multiple LLM-judged anchor passages. The method consists of the following steps:

1. **Candidate Sampling**: A small set of R passages is sampled from the entire set using an ϵ-greedy strategy, mixing top DR-ranked passages with uniformly random selections.

2. **LLM Relevance Judgments**: An LLM is used to assign relevance scores to each sampled passage.

3. **Fitting GPR**: A GPR model is fitted based on the relevance scores obtained from the LLM.

4. **Scoring All Passages**: The posterior mean from the GPR is used to predict scores for all passages.

5. **Item Scoring**: The passage scores are aggregated to generate final item scores.

#### Example
- **Input**: A user inputs the request "family-friendly cities."
- **Sampling**: The system samples R passages, including relevant ones (e.g., "Vancouver is a family-friendly city.") and randomly selected passages (e.g., "San Diego has beautiful beaches.").
- **LLM Judgments**: The LLM assigns relevance scores to each passage. For instance, "Vancouver is a family-friendly city." might receive a score of 3, while "San Diego has beautiful beaches." might receive a score of 1.
- **Fitting GPR**: The GPR model is fitted based on these scores.
- **Scoring All Passages**: The GPR model is used to predict scores for all passages.
- **Item Scoring**: Finally, the most relevant items are recommended.

<br/>
# 요약

이 논문에서는 자연어 추천(NLRec)을 위한 GPR-LLM 방법을 제안하며, 이는 여러 LLM-판단 기준을 활용하여 가우시안 프로세스 회귀(GPR)를 통해 아이템의 관련성을 추정한다. 실험 결과, GPR-LLM은 기존의 밀집 검색(DR), 크로스 인코더, 포인트와이즈 LLM 기반 관련성 점수 산정 방법보다 최대 65% 더 우수한 성능을 보였다. 이 방법은 복잡한 NLRec 데이터에서 다중 관련성 모드를 효과적으로 캡처하여 추천 품질을 향상시킨다.

---

This paper proposes the GPR-LLM method for Natural Language Recommendation (NLRec), which utilizes multiple LLM judgments to estimate item relevance through Gaussian Process Regression (GPR). Experimental results show that GPR-LLM consistently outperforms existing methods, including Dense Retrieval (DR), Cross-Encoder, and Pointwise LLM-based relevance scoring by up to 65%. This approach effectively captures multiple relevance modes in complex NLRec data, enhancing the quality of recommendations.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: Dense Retrieval (DR) 방법이 단일 쿼리 중심의 유일한 관련 신호를 가정하는 반면, GPR-LLM은 여러 관련 신호를 통합하여 다중 모드의 관련성을 포착할 수 있음을 보여줍니다. 이는 GPR-LLM이 더 복잡한 NLRec 데이터에서 더 나은 성능을 발휘할 수 있는 이유를 설명합니다.
   - **Figure 2**: GPR-LLM의 전체 프로세스를 시각화하여, 샘플링, LLM의 관련성 판단, GPR 적합, 모든 패시지에 대한 점수 예측 및 최종 아이템 점수 집계 단계를 보여줍니다.
   - **Figure 3**: RBF 커널이 다른 커널에 비해 모든 데이터셋에서 일관되게 우수한 성능을 보이는 것을 나타냅니다. 이는 RBF 커널이 다중 모드의 관련성을 더 잘 포착할 수 있음을 시사합니다.
   - **Figure 4**: 다양한 샘플링 전략이 GPR-LLM의 성능에 미치는 영향을 보여줍니다. ϵ=0.3의 소규모 탐색적 샘플링이 성능을 향상시키는 것을 확인할 수 있습니다.
   - **Figure 5**: DR과 GPR-LLM의 상위 패시지 분포를 비교하여, DR이 쿼리 주변에 밀집된 패시지를 선택하는 반면, GPR-LLM은 여러 지역에 분산된 패시지를 선택함을 보여줍니다. 이는 GPR-LLM이 다중 모드의 관련성을 포착하는 데 더 효과적임을 나타냅니다.

2. **테이블**
   - **Table 1**: GPR-LLM의 쿼리당 시간 복잡도와 지연 시간을 보여줍니다. GPR-LLM은 DR에 비해 약간의 추가 오버헤드가 있지만, 여전히 효율적입니다.
   - **Table 2**: 실험에 사용된 데이터셋의 통계 정보를 제공합니다. 각 데이터셋은 복잡하고 다면적인 쿼리를 포함하고 있어 GPR-LLM의 성능을 평가하는 데 적합합니다.
   - **Table 3**: GPR-LLM이 다양한 LLM 백본과 레이블 예산에서 다른 기준선 방법들과 비교하여 성능을 보여줍니다. GPR-LLM은 대부분의 경우 기준선보다 우수한 성능을 보이며, 특히 포인트 LLM 기반 점수보다 최대 65% 향상된 결과를 나타냅니다.

3. **어펜딕스**
   - **Appendix A**: UMBRELA 프롬프트를 사용하여 LLM의 관련성 판단을 수행하는 방법을 설명합니다. 이는 쿼리와 패시지 간의 관련성을 평가하는 데 중요한 역할을 합니다.
   - **Appendix B**: BM25, DR, Cross-Encoder 및 Pointwise LLM 기반 점수와 같은 기준선 방법의 구현 세부 사항을 제공합니다.
   - **Appendix C**: PW 점수의 비대칭적 행동을 분석하여 GPR-LLM이 적은 레이블 예산에서도 경쟁력을 유지할 수 있음을 보여줍니다.
   - **Appendix D**: 샘플링 품질이 GPR-LLM에 미치는 영향을 조사하여, 고품질 샘플링이 성능 향상에 기여할 수 있음을 나타냅니다.
   - **Appendix E**: 다양한 임베딩의 영향을 평가하여 GPR-LLM의 성능이 임베딩 모델에 의존하지 않음을 보여줍니다.

---

### Insights and Results from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: Illustrates that Dense Retrieval (DR) methods assume a single query-centered relevance signal, while GPR-LLM integrates multiple relevance signals to capture multimodal relevance. This explains why GPR-LLM performs better in complex NLRec data.
   - **Figure 2**: Visualizes the overall process of GPR-LLM, showing the steps of sampling, LLM relevance judgment, GPR fitting, scoring all passages, and aggregating final item scores.
   - **Figure 3**: Demonstrates that the RBF kernel consistently outperforms other kernels across all datasets, suggesting that it is better suited for capturing multimodal relevance.
   - **Figure 4**: Shows the impact of various sampling strategies on GPR-LLM's performance, confirming that a small exploratory sampling fraction (ϵ=0.3) enhances performance.
   - **Figure 5**: Compares the distribution of top-ranked passages from DR and GPR-LLM, indicating that DR clusters passages tightly around the query, while GPR-LLM retrieves passages dispersed across multiple regions, effectively capturing multimodal relevance.

2. **Tables**
   - **Table 1**: Displays the per-query time complexity and latency of GPR-LLM, indicating that while there is some additional overhead compared to DR, it remains efficient.
   - **Table 2**: Provides statistics of the datasets used in the experiments, highlighting their complexity and suitability for evaluating GPR-LLM's performance.
   - **Table 3**: Compares GPR-LLM against various baseline methods across different LLM backbones and labeling budgets, showing that GPR-LLM consistently outperforms baselines, achieving up to 65% improvement over pointwise LLM relevance scoring.

3. **Appendices**
   - **Appendix A**: Describes the UMBRELA prompt used for LLM relevance judgment, which plays a crucial role in evaluating the relevance between queries and passages.
   - **Appendix B**: Provides implementation details for baseline methods such as BM25, DR, Cross-Encoder, and Pointwise LLM-based scoring.
   - **Appendix C**: Analyzes the asymptotic behavior of PW scoring, demonstrating that GPR-LLM remains competitive even with a smaller labeling budget.
   - **Appendix D**: Investigates the effect of sampling quality on GPR-LLM, indicating that high-quality sampling can lead to performance improvements.
   - **Appendix E**: Evaluates the impact of different embeddings, showing that GPR-LLM's performance is not dependent on the embedding model used.

<br/>
# refer format:

### BibTeX 형식

```bibtex
@inproceedings{liu2026multimodal,
  author = {Yifan Simon Liu and Qianfeng Wen and Jiazhou Liang and Mark Zhao and Justin Cui and Anton Korikov and Armin Toroghi and Junyoung Kim and Scott Sanner},
  title = {Multimodal Item Scoring for Natural Language Recommendation via Gaussian Process Regression with LLM Relevance Judgments},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages = {36859--36876},
  year = {2026},
  month = {July},
  publisher = {Association for Computational Linguistics},
}
```

### 시카고 스타일

Liu, Yifan Simon, Qianfeng Wen, Jiazhou Liang, Mark Zhao, Justin Cui, Anton Korikov, Armin Toroghi, Junyoung Kim, and Scott Sanner. 2026. "Multimodal Item Scoring for Natural Language Recommendation via Gaussian Process Regression with LLM Relevance Judgments." In *Findings of the Association for Computational Linguistics: ACL 2026*, 36859–36876. Association for Computational Linguistics.
    