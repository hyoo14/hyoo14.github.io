---
layout: post
title:  "[2026]Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring for Dense Passage Retrieval"
date:   2026-07-14 00:43:31 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 LLM의 관련성 점수를 기반으로 한 가우시안 프로세스를 활용한 베이지안 능동 학습(BAGEL) 프레임워크를 제안합니다.


짧은 요약(Abstract) :


이 논문의 초록에서는 대형 언어 모델(LLM)이 뛰어난 제로샷 관련성 모델링 능력을 가지고 있지만, 높은 계산 비용으로 인해 패시지 검색을 예산 제약이 있는 글로벌 최적화 문제로 설정해야 한다고 설명합니다. 기존의 접근 방식은 첫 번째 단계의 밀집 검색기(dense retriever)에 수동적으로 의존하여 두 가지 한계를 초래합니다: 첫째, 의미적으로 구별되는 클러스터에서 관련 패시지를 검색하지 못하고, 둘째, 관련성 신호를 더 넓은 코퍼스로 전파하지 못합니다. 이러한 한계를 극복하기 위해, 저자들은 LLM의 관련성 점수에 의해 안내되는 가우시안 프로세스를 활용한 베이지안 능동 학습(BAGEL)이라는 새로운 프레임워크를 제안합니다. BAGEL은 LLM의 희소한 관련성 신호를 임베딩 공간 전반에 걸쳐 전파하여 글로벌 탐색을 안내합니다. 이 프레임워크는 LLM의 예산 내에서 LLM 재정렬 방법보다 효과적으로 복잡한 관련성 분포를 탐색하고 포착할 수 있음을 실험을 통해 입증합니다.



The abstract of this paper explains that while Large Language Models (LLMs) exhibit exceptional zero-shot relevance modeling capabilities, their high computational cost necessitates framing passage retrieval as a budget-constrained global optimization problem. Existing approaches passively rely on first-stage dense retrievers, leading to two limitations: first, they fail to retrieve relevant passages in semantically distinct clusters, and second, they do not propagate relevance signals to the broader corpus. To address these limitations, the authors propose a novel framework called Bayesian Active Learning with Gaussian Processes Guided by LLM relevance scoring (BAGEL). BAGEL propagates sparse LLM relevance signals across the embedding space to guide global exploration. The framework demonstrates that it effectively explores and captures complex relevance distributions, outperforming LLM reranking methods under the same LLM budget through extensive experiments.


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



BAGEL(Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring)은 대규모 언어 모델(LLM)의 관련성 점수를 활용하여 패시지 검색을 위한 새로운 프레임워크입니다. 이 방법은 두 가지 주요 구성 요소로 이루어져 있습니다: 가우시안 프로세스(GP) 기반의 능동 학습과 LLM 기반의 관련성 점수입니다.

1. **가우시안 프로세스(GP)**: BAGEL은 GP를 사용하여 쿼리-패시지 관련성 분포를 모델링합니다. GP는 비모수 베이지안 모델로, 함수에 대한 사전 분포를 정의하여 불확실성을 정량화할 수 있습니다. GP는 커널 함수를 통해 데이터 포인트 간의 상관관계를 모델링하며, 이를 통해 패시지 임베딩 공간에서의 관련성 신호를 보간할 수 있습니다.

2. **능동 학습**: BAGEL은 초기 단계에서 쿼리와 관련된 패시지를 선택하기 위해 '웜 스타트' 초기화 단계를 사용합니다. 이 단계에서는 쿼리 자체와 밀접하게 관련된 상위 M개의 패시지를 선택하여 GP 모델에 강력한 신호를 제공합니다. 이후, GP의 예측 평균과 불확실성을 기반으로 다음 패시지를 선택하는 능동 학습 단계를 진행합니다. 이 과정에서 BAGEL은 탐색(exploration)과 활용(exploitation) 간의 균형을 맞추어 패시지 임베딩 공간을 효율적으로 탐색합니다.

3. **LLM 기반의 관련성 점수**: BAGEL은 LLM을 사용하여 쿼리-패시지 쌍의 관련성을 점수화합니다. 이 점수는 LLM의 출력으로부터 얻어지며, GP는 이 점수를 기반으로 패시지의 관련성을 추정합니다. BAGEL은 LLM의 계산 비용을 고려하여 제한된 예산 내에서 최대한의 관련성을 탐색할 수 있도록 설계되었습니다.

4. **획득 함수**: BAGEL은 GP의 예측 평균과 불확실성을 결합하여 다음 패시지를 선택하는 데 사용되는 획득 함수를 정의합니다. 이 함수는 UCB(Upper Confidence Bound)와 같은 다양한 전략을 통해 탐색과 활용을 조절합니다.

BAGEL은 이러한 구성 요소들을 통해 패시지 검색의 성능을 크게 향상시키며, 기존의 LLM 재정렬 방법보다 더 효과적으로 관련 패시지를 탐색할 수 있습니다. 실험 결과, BAGEL은 여러 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다.

---




BAGEL (Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring) is a novel framework for passage retrieval that leverages relevance scoring from large language models (LLMs). This method consists of two main components: Gaussian Process (GP)-based active learning and LLM-based relevance scoring.

1. **Gaussian Processes (GP)**: BAGEL utilizes GP to model the query-passage relevance distribution. GP is a non-parametric Bayesian model that defines a prior distribution over functions, allowing for principled uncertainty quantification. It models correlations between data points through kernel functions, enabling the interpolation of relevance signals across the passage embedding space.

2. **Active Learning**: BAGEL employs a 'warm start' initialization phase to select passages relevant to the query in the initial stage. In this phase, the query itself and the top M passages closely related to it are selected to provide strong signals to the GP model. Subsequently, an active learning phase is conducted, where the next passage is selected based on the GP's predicted mean and uncertainty. This process allows BAGEL to efficiently explore the passage embedding space by balancing exploration and exploitation.

3. **LLM-based Relevance Scoring**: BAGEL uses LLMs to score the relevance of query-passage pairs. This score is derived from the output of the LLM, and the GP estimates the relevance of passages based on this score. BAGEL is designed to maximize the exploration of relevant passages within a constrained budget, considering the computational cost of LLMs.

4. **Acquisition Function**: BAGEL defines an acquisition function that combines the GP's predicted mean and uncertainty to select the next passage. This function employs various strategies, such as Upper Confidence Bound (UCB), to balance exploration and exploitation.

Through these components, BAGEL significantly enhances the performance of passage retrieval and can explore relevant passages more effectively than traditional LLM reranking methods. Experimental results demonstrate that BAGEL outperforms existing methods across multiple datasets.


<br/>
# Results



이 논문에서는 BAGEL(Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring)이라는 새로운 프레임워크를 제안하고, 이를 통해 대규모 문서 집합에서의 패시지 검색 성능을 향상시키는 방법을 다룹니다. BAGEL은 대형 언어 모델(LLM)의 점수와 가우시안 프로세스를 결합하여, 제한된 예산 내에서 효과적으로 관련 패시지를 탐색할 수 있도록 설계되었습니다.

#### 실험 설정
BAGEL은 네 가지 데이터셋(Covid, NFCorpus, Robust04, TravelDest)에서 평가되었습니다. 각 데이터셋은 서로 다른 특성을 가지고 있으며, BAGEL은 이들 데이터셋에서 기존의 LLM 재정렬 방법들과 비교되었습니다. 실험에서는 NDCG@10, NDCG@50, Recall@10, Recall@50과 같은 메트릭을 사용하여 성능을 평가했습니다.

#### 경쟁 모델
BAGEL은 다음과 같은 다섯 가지 경쟁 모델과 비교되었습니다:
1. **BM25**: 전통적인 희소 검색 방법.
2. **Dense Retriever**: 쿼리와 패시지를 밀집 벡터로 인코딩하여 유사성을 기반으로 검색하는 모델.
3. **Cross Encoder**: 쿼리-패시지 쌍을 공동으로 인코딩하여 세밀한 관련 점수를 예측하는 BERT 기반 재정렬 모델.
4. **Pointwise LLM**: 각 쿼리-패시지 쌍을 독립적으로 점수화하는 LLM 기반 방법.
5. **Listwise LLM**: 여러 패시지를 동시에 입력하여 전역 컨텍스트를 기반으로 순위를 매기는 LLM 기반 재정렬 방법.

#### 결과
BAGEL은 모든 데이터셋에서 기존의 LLM 재정렬 방법들보다 일관되게 우수한 성능을 보였습니다. 예를 들어, TravelDest 데이터셋에서 NDCG@50 메트릭이 29.3에서 41.6으로 향상되었습니다. BAGEL은 초기 후보 집합에 국한되지 않고, 전체 임베딩 공간을 탐색하여 관련 패시지를 발견하는 데 성공했습니다. 이는 BAGEL이 고유한 탐색-활용 균형을 통해 다양한 관련 클러스터를 효과적으로 식별할 수 있음을 보여줍니다.

#### 결론
BAGEL은 LLM 기반의 점수화와 가우시안 프로세스를 결합하여, 제한된 예산 내에서 효과적으로 패시지를 검색할 수 있는 가능성을 보여주었습니다. 이 연구는 LLM의 활용을 극대화하고, 패시지 검색의 효율성을 높이는 데 기여할 수 있는 방법론을 제시합니다.

---




This paper introduces a novel framework called BAGEL (Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring) aimed at improving passage retrieval performance over large document collections. BAGEL is designed to effectively explore relevant passages within a constrained budget by combining the scoring of large language models (LLMs) with Gaussian processes.

#### Experimental Setup
BAGEL was evaluated on four datasets: Covid, NFCorpus, Robust04, and TravelDest. Each dataset has distinct characteristics, and BAGEL was compared against existing LLM reranking methods on these datasets. Metrics such as NDCG@10, NDCG@50, Recall@10, and Recall@50 were used to assess performance.

#### Competing Models
BAGEL was compared against five representative models:
1. **BM25**: A traditional sparse retrieval method.
2. **Dense Retriever**: A model that encodes queries and passages into dense vectors and retrieves based on similarity.
3. **Cross Encoder**: A BERT-based reranker that jointly encodes a query-passage pair to predict fine-grained relevance scores.
4. **Pointwise LLM**: An LLM-based method that scores each query-passage pair independently.
5. **Listwise LLM**: An LLM-based reranking approach that inputs multiple passages simultaneously to generate a reordered list based on global context.

#### Results
BAGEL consistently outperformed all baseline methods across all datasets. For instance, on the TravelDest dataset, the NDCG@50 metric improved from 29.3 to 41.6. BAGEL successfully discovered relevant passages beyond the initial candidate set by actively exploring the entire embedding space. This demonstrates BAGEL's ability to effectively identify diverse relevant clusters through a unique exploration-exploitation balance.

#### Conclusion
BAGEL showcases the potential of combining LLM-based scoring with Gaussian processes to effectively retrieve passages within a limited budget. This research presents a methodology that maximizes the utility of LLMs and enhances the efficiency of passage retrieval.


<br/>
# 예제



이 논문에서는 "Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring" (BAGEL)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대규모 언어 모델(LLM)을 활용하여 패시지 검색을 최적화하는 방법을 다룹니다. BAGEL은 LLM의 점수 기반으로 가우시안 프로세스를 사용하여 패시지의 관련성을 평가하고, 이를 통해 검색 공간을 효율적으로 탐색합니다.

#### 예시

1. **트레이닝 데이터**:
   - **쿼리**: "COVID-19의 전파 경로는 무엇인가요?"
   - **패시지**:
     1. "COVID-19는 주로 비말을 통해 전파됩니다."
     2. "COVID-19의 증상으로는 발열, 기침, 호흡 곤란이 있습니다."
     3. "COVID-19 예방을 위해 마스크 착용이 권장됩니다."
   - **라벨**: 
     - 패시지 1: 3 (정확한 답변)
     - 패시지 2: 2 (부분적인 답변)
     - 패시지 3: 3 (정확한 답변)

2. **테스트 데이터**:
   - **쿼리**: "COVID-19의 예방 방법은?"
   - **패시지**:
     1. "손 씻기와 마스크 착용이 중요합니다."
     2. "COVID-19는 전 세계적으로 확산되고 있습니다."
     3. "백신 접종이 예방에 효과적입니다."
   - **예상 아웃풋**:
     - 패시지 1: 3 (정확한 답변)
     - 패시지 2: 0 (관련 없음)
     - 패시지 3: 3 (정확한 답변)

이 예시에서 BAGEL은 LLM을 사용하여 각 패시지의 관련성을 점수화하고, 가우시안 프로세스를 통해 불확실성을 평가하여 다음에 어떤 패시지를 선택할지를 결정합니다. 이 과정은 초기의 패시지 세트를 기반으로 하여, 더 많은 패시지를 탐색하고 관련성을 높이는 방향으로 진행됩니다.




This paper proposes a novel framework called "Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring" (BAGEL). This framework addresses the optimization of passage retrieval using large language models (LLMs). BAGEL utilizes LLM-based scoring to evaluate the relevance of passages and employs Gaussian processes to efficiently explore the search space.

#### Example

1. **Training Data**:
   - **Query**: "What are the transmission routes of COVID-19?"
   - **Passages**:
     1. "COVID-19 is primarily transmitted through respiratory droplets."
     2. "Symptoms of COVID-19 include fever, cough, and shortness of breath."
     3. "Wearing masks is recommended to prevent COVID-19."
   - **Labels**: 
     - Passage 1: 3 (Exact answer)
     - Passage 2: 2 (Partial answer)
     - Passage 3: 3 (Exact answer)

2. **Test Data**:
   - **Query**: "What are the prevention methods for COVID-19?"
   - **Passages**:
     1. "Hand washing and wearing masks are important."
     2. "COVID-19 is spreading globally."
     3. "Vaccination is effective for prevention."
   - **Expected Output**:
     - Passage 1: 3 (Exact answer)
     - Passage 2: 0 (Not relevant)
     - Passage 3: 3 (Exact answer)

In this example, BAGEL uses the LLM to score the relevance of each passage and assesses uncertainty through Gaussian processes to determine which passage to select next. This process is based on the initial set of passages and aims to explore more passages while increasing relevance.

<br/>
# 요약


이 논문에서는 LLM의 관련성 점수를 기반으로 한 가우시안 프로세스를 활용한 베이지안 능동 학습(BAGEL) 프레임워크를 제안합니다. BAGEL은 초기 후보 집합에 의존하지 않고, 전체 임베딩 공간을 탐색하여 관련성을 효과적으로 전파함으로써 다양한 클러스터에서 관련된 구절을 발견합니다. 실험 결과, BAGEL은 고정된 LLM 예산 하에서 기존 LLM 재정렬 방법보다 모든 데이터셋에서 우수한 성능을 보였습니다.

---

This paper proposes a Bayesian Active Learning framework (BAGEL) that utilizes Gaussian Processes guided by LLM relevance scores. BAGEL effectively explores the entire embedding space and propagates relevance signals without relying on an initial candidate set, allowing it to discover relevant passages from diverse clusters. Experimental results demonstrate that BAGEL outperforms existing LLM reranking methods across all datasets under a fixed LLM budget.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Figure 1**: BAGEL과 기존 LLM 포인트 재정렬 방식의 비교를 보여줍니다. BAGEL은 고신뢰 지역의 활용과 불확실한 지역의 탐색을 균형 있게 조절하여 다양한 클러스터에서 관련 패시지를 발견하는 데 효과적임을 나타냅니다.

2. **Figure 3**: Covid와 TravelDest 데이터셋에서 BAGEL의 탐색 전략을 시각화합니다. Covid에서는 관련 패시지가 쿼리 임베딩 근처에 집중되어 있는 반면, TravelDest에서는 관련 패시지가 전반적으로 분산되어 있습니다. BAGEL은 이러한 분포에 따라 탐색 전략을 조정하여 더 많은 관련 패시지를 발견합니다.

3. **Figure 4**: 다양한 LLM 예산에 따른 BAGEL의 성능을 보여줍니다. BAGEL은 예산이 적을 때도 기존 포인트 LLM보다 우수한 성능을 보이며, 예산을 효율적으로 활용하는 방법을 제시합니다.

#### 테이블
1. **Table 1**: BAGEL의 전반적인 성능을 보여줍니다. 모든 데이터셋에서 BAGEL이 기존의 LLM 재정렬 방법보다 우수한 성능을 보이며, 특히 NDCG@50에서 큰 성과를 나타냅니다.

2. **Table 2**: 다양한 커널 함수의 성능을 비교합니다. RBF와 Matérn 커널이 Linear 커널보다 우수한 성능을 보이며, 이는 복잡한 다중 모드 관련 구조를 효과적으로 모델링할 수 있음을 나타냅니다.

3. **Table 3**: 다양한 획득 함수의 성능을 비교합니다. UCB와 같은 베이지안 획득 함수가 무작위 선택이나 밀집 선택보다 일관되게 우수한 성능을 보입니다.

#### 어펜딕스
- **Appendix A**: 다양한 커널 함수에 대한 설명과 성능 비교를 포함합니다.
- **Appendix B**: 여러 획득 함수의 정의와 성능 비교를 제공합니다.
- **Appendix C**: 실험 설정 및 데이터셋 통계에 대한 자세한 정보를 제공합니다.




#### Diagrams and Figures
1. **Figure 1**: Compares BAGEL with the existing LLM pointwise reranking method. It shows that BAGEL effectively discovers relevant passages from diverse clusters by balancing the exploitation of high-confidence regions and the exploration of uncertain areas.

2. **Figure 3**: Visualizes BAGEL's selection strategy in the Covid and TravelDest datasets. In Covid, relevant passages are concentrated near the query embedding, while in TravelDest, they are dispersed across the embedding space. BAGEL adjusts its exploration strategy according to these distributions to uncover more relevant passages.

3. **Figure 4**: Displays BAGEL's performance under varying LLM budgets. It shows that BAGEL consistently outperforms the pointwise LLM baseline even with a lower budget, demonstrating an effective use of resources.

#### Tables
1. **Table 1**: Shows the overall performance of BAGEL. It consistently outperforms existing LLM reranking methods across all datasets, particularly achieving significant results in NDCG@50.

2. **Table 2**: Compares the performance of different kernel functions. Both RBF and Matérn kernels outperform the Linear kernel, indicating their effectiveness in modeling complex multimodal relevance structures.

3. **Table 3**: Compares the performance of various acquisition functions. Bayesian acquisition functions like UCB consistently show superior performance compared to random or dense selection.

#### Appendix
- **Appendix A**: Contains descriptions and performance comparisons of various kernel functions.
- **Appendix B**: Provides definitions and performance comparisons of several acquisition functions.
- **Appendix C**: Offers detailed information on experimental setup and dataset statistics.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{kim2026bayesian,
  author    = {Junyoung Kim and Anton Korikov and Jiazhou Liang and Justin Cui and Yifan Simon Liu and Qianfeng Wen and Mark Zhao and Scott Sanner},
  title     = {Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring for Dense Passage Retrieval},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {9884--9898},
  year      = {2026},
  month     = {July 2-7},
  publisher  = {Association for Computational Linguistics},
}
```

### Chicago Style Citation

Junyoung Kim, Anton Korikov, Jiazhou Liang, Justin Cui, Yifan Simon Liu, Qianfeng Wen, Mark Zhao, and Scott Sanner. "Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring for Dense Passage Retrieval." In *Findings of the Association for Computational Linguistics: ACL 2026*, 9884–9898. July 2-7, 2026. Association for Computational Linguistics.
    