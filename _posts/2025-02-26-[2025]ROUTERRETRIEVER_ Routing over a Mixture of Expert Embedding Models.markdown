---
layout: post
title:  "[2025]ROUTERRETRIEVER: Routing over a Mixture of Expert Embedding Models"  
date:   2025-02-26 00:39:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    




정보 검색 방법은 일반적으로 MSMARCO와 같은 대규모 범용 데이터셋에서 학습된 단일 임베딩 모델에 의존합니다. 이러한 방식은 전체적으로 적절한 성능을 제공할 수 있지만, 특정 도메인에서 테스트할 때 해당 도메인에서 학습된 모델보다 성능이 떨어지는 경우가 많습니다. 기존 연구에서는 다중 작업 학습을 통해 이를 해결하려 했으나, 도메인별 전문가 검색 모델을 조합하여 최적의 모델을 선택하는 접근 방식은 아직 충분히 연구되지 않았습니다.

이에 따라 본 연구에서는 **ROUTERRETRIEVER**라는 검색 모델을 제안합니다. 이 모델은 질의에 대해 적절한 도메인 전문가를 선택하는 라우팅 메커니즘을 활용하여, 여러 도메인별 전문가 모델을 조합하는 방식을 사용합니다. ROUTERRETRIEVER는 가벼운 모델로, 추가적인 학습 없이도 새로운 전문가를 쉽게 추가하거나 제거할 수 있는 장점을 가집니다. 

BEIR 벤치마크 평가 결과, ROUTERRETRIEVER는 MSMARCO로 학습된 단일 모델보다 **nDCG@10 기준 2.1점**, 다중 작업 학습 모델보다 **3.2점 향상된 성능**을 보였습니다. 또한 기존 언어 모델 연구에서 사용되던 라우팅 기술을 적용했을 때보다 **평균 1.8점 높은 성능**을 기록했습니다. 특히, 특정 도메인 전문가가 없는 상황에서도 일반적인 단일 모델보다 우수한 성능을 보여, 다양한 도메인 검색에서 효과적인 대안임을 입증하였습니다.

---



Information retrieval methods often rely on a single embedding model trained on large, general-domain datasets like MSMARCO. While this approach can produce a retriever with reasonable overall performance, it often underperforms models trained on domain-specific data when tested in their respective domains. Prior work in information retrieval has tackled this through multi-task training, but the idea of routing over a mixture of domain-specific expert retrievers remains unexplored despite its popularity in language model generation research.

In this work, we introduce **ROUTERRETRIEVER**, a retrieval model that leverages a mixture of domain-specific experts by using a routing mechanism to select the most appropriate expert for each query. ROUTERRETRIEVER is lightweight and allows easy addition or removal of experts without additional training.

Evaluation on the BEIR benchmark demonstrates that ROUTERRETRIEVER outperforms both models trained on MSMARCO (**+2.1 absolute nDCG@10**) and multi-task models (**+3.2**). This is achieved by employing our routing mechanism, which surpasses other routing techniques (**+1.8 on average**) commonly used in language modeling. Furthermore, the benefit generalizes well to other datasets, even in the absence of a specific expert on the dataset. ROUTERRETRIEVER is the first work to demonstrate the advantages of routing over a mixture of domain-specific expert embedding models as an alternative to a single, general-purpose embedding model, especially when retrieving from diverse, specialized domains.



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



이 논문에서는 **ROUTERRETRIEVER**라는 검색 모델을 제안하며, 도메인별 전문가 임베딩 모델을 조합하여 적절한 전문가를 선택하는 **라우팅 메커니즘**을 도입합니다. 이를 통해 다양한 도메인에서 효과적인 검색 성능을 달성하고, 새로운 도메인 전문가를 추가하거나 제거할 때 재학습이 필요 없는 경량화된 구조를 갖습니다.

#### **1. 모델 구조**
ROUTERRETRIEVER는 **기본 검색 모델(Base Encoder)** 과 **여러 도메인 전문가(Experts)** 로 구성됩니다. 기본 검색 모델은 모든 전문가와 공유되며, 각 전문가 모델은 특정 도메인에 맞춰 훈련된 **LoRA 모듈**입니다.

- **기본 검색 모델 (Base Encoder)**: Contriever (Izacard et al., 2021) 기반의 사전 학습된 모델을 사용하여, 모든 전문가들이 공유하는 공통 모델로 활용됩니다.
- **전문가 모델 (Experts)**: 특정 도메인에서 학습된 LoRA 기반 모듈로, 도메인별로 개별적으로 훈련됩니다.
- **파일럿 임베딩 라이브러리 (Pilot Embedding Library)**: 질의와 각 전문가 모델의 유사도를 비교하기 위해 미리 계산된 대표 임베딩을 저장하는 라이브러리입니다.
- **라우팅 메커니즘 (Routing Mechanism)**: 질의 임베딩과 파일럿 임베딩 간의 유사도를 계산하여 가장 적합한 전문가 모델을 선택합니다.

#### **2. 훈련 과정**
ROUTERRETRIEVER의 훈련은 두 가지 단계로 진행됩니다.

1. **도메인별 전문가 훈련**  
   - 각 도메인 데이터셋을 사용하여 LoRA 기반 전문가 모델을 훈련합니다.
   - Base Encoder는 고정(frozen)된 상태에서, LoRA 모듈만 미세 조정됩니다.
   - 결과적으로 각 도메인별 최적화된 전문가 모델이 생성됩니다.

2. **파일럿 임베딩 생성**  
   - 각 도메인 데이터셋에서 대표적인 문장을 선택하여, 모든 전문가 모델을 사용해 임베딩을 생성합니다.
   - 각 질의에 대해 가장 적절한 전문가를 선택하는 기준으로 활용됩니다.

#### **3. 라우팅 메커니즘**
질의가 입력되면, 다음과 같은 과정으로 최적의 전문가 모델을 선택합니다.

1. 기본 검색 모델을 사용하여 질의 임베딩을 생성합니다.
2. 파일럿 임베딩 라이브러리에서 미리 계산된 전문가별 임베딩과 질의 임베딩 간 유사도를 계산합니다.
3. 가장 높은 평균 유사도를 가지는 전문가 모델을 선택합니다.
4. 선택된 전문가 모델을 활용하여 최종 질의 임베딩을 생성합니다.

이 과정은 기존의 룰 기반 라우팅보다 유연하고, 새로운 도메인을 추가할 때도 추가적인 학습이 필요하지 않다는 장점이 있습니다.

#### **4. 데이터셋 및 실험 설정**
- **훈련 데이터**: BEIR(Benchmarking Information Retrieval) 벤치마크를 사용하며, MSMARCO, SciFact, NFCorpus 등 여러 도메인 데이터셋을 포함합니다.
- **평가 데이터**: BEIR 벤치마크에서 제공하는 다양한 검색 데이터셋을 사용하여 성능을 검증합니다.
- **하이퍼파라미터**:
  - **기본 모델**: Contriever 기반
  - **LoRA 설정**: 랭크 8, 알파 값 32, 약 100만 개의 파라미터 학습
  - **학습률**: 1e-4
  - **배치 크기**: 256
  - **최대 500 에포크** 진행 (얼리 스토핑 적용)

#### **5. 실험 결과**
- ROUTERRETRIEVER는 단일 MSMARCO 모델보다 **nDCG@10 기준 +2.1 점** 향상된 성능을 보였으며, 다중 작업 학습(Multi-task Learning) 모델보다 **+3.2 점** 향상된 성능을 기록했습니다.
- 기존 언어 모델에서 사용되는 일반적인 라우팅 기법보다 **평균 +1.8 점 높은 성능**을 달성했습니다.
- 전문가 모델의 수가 증가할수록 성능이 향상되지만, 일정 수준 이상에서는 **성능 개선이 완만해지는 경향**을 보였습니다.
- 특정 도메인 전문가가 없는 상황에서도 ROUTERRETRIEVER는 단일 모델보다 **더 나은 일반화 성능**을 나타냈습니다.

---


This paper introduces **ROUTERRETRIEVER**, a retrieval model that combines multiple domain-specific expert embedding models with a **routing mechanism** to select the most appropriate expert for each query. The model is lightweight, allowing easy addition or removal of experts without additional training.

#### **1. Model Architecture**
ROUTERRETRIEVER consists of:
- **Base Encoder**: A shared retrieval model based on **Contriever (Izacard et al., 2021)**, which is frozen during training.
- **Domain-specific Experts**: LoRA-based modules trained on individual domain datasets.
- **Pilot Embedding Library**: Precomputed embeddings that help determine the best expert for a given query.
- **Routing Mechanism**: Determines the most suitable expert by calculating the similarity between the query embedding and the pilot embeddings.

#### **2. Training Process**
ROUTERRETRIEVER training follows two main steps:

1. **Training Domain-specific Experts**  
   - Each expert model is fine-tuned on a domain-specific dataset.
   - The base encoder remains **frozen**, while only the LoRA modules are trained.
   - The result is a set of **specialized expert models**, each optimized for a specific domain.

2. **Constructing the Pilot Embedding Library**  
   - Representative sentences from each domain dataset are encoded using all expert models.
   - The expert providing the **most relevant** representation for each instance is selected.
   - These embeddings are stored as a reference for **routing decisions**.

#### **3. Routing Mechanism**
When a query is received, ROUTERRETRIEVER follows these steps:

1. Generate the query embedding using the **base encoder**.
2. Compute the similarity between the query embedding and the **precomputed pilot embeddings**.
3. Select the expert model with the highest **average similarity**.
4. Use the chosen expert to generate the final query embedding.

This method provides a **flexible alternative** to rule-based routing and does not require retraining when adding new domains.

#### **4. Dataset & Experimental Setup**
- **Training Data**: BEIR benchmark datasets, including MSMARCO, SciFact, NFCorpus, and others.
- **Evaluation Data**: Various datasets from BEIR for benchmarking retrieval performance.
- **Hyperparameters**:
  - **Base Model**: Contriever
  - **LoRA Configuration**: Rank 8, Alpha 32, ~1M parameters per expert
  - **Learning Rate**: 1e-4
  - **Batch Size**: 256
  - **Maximum Epochs**: 500 (early stopping applied)

#### **5. Experimental Results**
- ROUTERRETRIEVER outperforms a single MSMARCO-trained model by **+2.1 absolute nDCG@10** and multi-task models by **+3.2**.
- It **exceeds standard routing techniques** used in language modeling by an average of **+1.8**.
- Performance **improves as more experts are added**, though **diminishing returns** occur beyond a certain number of experts.
- Even in domains **without a dedicated expert**, ROUTERRETRIEVER achieves **better generalization** than a single, general-purpose model.

This demonstrates the effectiveness of **routing over domain-specific experts**, providing a **more flexible** and **efficient** alternative to single-model retrievers.



   
 
<br/>
# Results  




ROUTERRETRIEVER의 성능은 **BEIR 벤치마크 데이터셋**에서 평가되었으며, 여러 경쟁 모델과 비교하여 분석되었습니다. 실험에서는 **nDCG@10 (Normalized Discounted Cumulative Gain at rank 10)**을 주요 평가 지표로 사용하여 모델의 검색 성능을 정량적으로 측정하였습니다.

#### **1. 비교 대상 (경쟁 모델)**
ROUTERRETRIEVER의 성능을 검증하기 위해 다음과 같은 비교 대상 모델을 설정하였습니다.

1. **단일 모델 (Single Model)**
   - **MSMARCO 모델**: 대규모 범용 데이터셋인 MSMARCO에서 학습된 단일 임베딩 모델.
   - **다중 작업 학습 (Multi-Task Training)**: MSMARCO와 도메인별 데이터셋을 함께 학습한 단일 모델.

2. **라우팅 기법 비교**
   - **ExpertClassifierRouter**: 전문가 모델별 확률을 계산하여 선택.
   - **ClassificationHeadRouter**: 분류 헤드를 사용하여 최적의 전문가 모델 선택.
   - **DatasetRouter**: 기존 데이터셋 레이블 기반으로 전문가 모델 선택.

3. **오라클 모델 (Oracle Models)**
   - **DatasetOracle**: 각 데이터셋에서 가장 성능이 좋은 전문가 모델을 선택하는 이상적인 설정.
   - **InstanceOracle**: 개별 질의별로 최적의 전문가를 동적으로 선택하는 이상적인 설정.

#### **2. 테스트 데이터 (BEIR 벤치마크)**
실험은 **BEIR(Benchmarking Information Retrieval) 데이터셋**을 사용하여 진행되었으며, 다음과 같은 다양한 도메인 데이터셋에서 모델을 평가하였습니다.

- **MSMARCO**: 범용 검색 데이터셋
- **Quora**: 중복 질문 검색
- **ArguAna**: 논증 분석 검색
- **HotpotQA**: 다중 문서 질의응답
- **NFCorpus**: 의료 정보 검색
- **SciFact**: 과학적 사실 검색
- **FiQA**: 금융 질의응답

추가적으로 **TREC-COVID**, **SciDocs**, **DBPedia**, **FEVER**, **NaturalQuestions** 등의 데이터셋에서도 **제로샷 성능(Zero-shot Generalization)**을 평가하였습니다.

#### **3. 주요 결과 (성능 분석)**
1. **ROUTERRETRIEVER vs 단일 모델**
   - MSMARCO 모델보다 **+2.1 absolute nDCG@10** 향상.
   - Multi-Task 모델보다 **+3.2 absolute nDCG@10** 향상.

2. **ROUTERRETRIEVER vs 라우팅 기법**
   - 기존 라우팅 기법(DatasetRouter, ClassificationHeadRouter, ExpertClassifierRouter)보다 평균 **+1.8** 높은 성능.
   - DatasetRouter보다 더 정교한 라우팅을 수행하여 **다양한 도메인에서 일관된 성능 향상**.

3. **ROUTERRETRIEVER vs 오라클 모델**
   - DatasetOracle과 유사한 성능을 보였으며, **nDCG@10에서 평균적으로 50.9점 달성**.
   - InstanceOracle보다는 낮은 성능을 보였지만, 라우팅 메커니즘 개선을 통해 추가적인 성능 향상이 가능함을 시사.

4. **전문가 모델 수 증가 효과**
   - 전문가 모델을 추가할수록 성능이 지속적으로 증가.
   - 하지만 일정 수준 이상(5개 이상)에서는 성능 향상의 폭이 감소하는 **수확체감 현상(Diminishing Returns)**이 관찰됨.

5. **제로샷 테스트 성능**
   - 특정 도메인 전문가가 없는 데이터셋에서도 단일 MSMARCO 모델보다 높은 성능을 보임.
   - 이는 **도메인별 전문가 모델의 조합이 범용 임베딩 모델보다 더 효과적임**을 입증.

#### **4. 결론 및 시사점**
- ROUTERRETRIEVER는 **단일 모델 대비 성능이 우수**하며, **다중 작업 학습보다도 효과적인 검색 성능을 제공**.
- 새로운 도메인을 추가하거나 제거할 때 **추가 학습 없이 쉽게 확장 가능**.
- 기존의 언어 모델 기반 라우팅 기법보다 **검색 환경에서 더 적합한 라우팅 방식**을 제공.
- 추가적인 연구를 통해 InstanceOracle에 더 가까운 라우팅 방식으로 개선 가능.

---


ROUTERRETRIEVER was evaluated using the **BEIR benchmark dataset**, and its performance was compared against several baseline models. The primary evaluation metric used was **nDCG@10 (Normalized Discounted Cumulative Gain at rank 10)**, which quantifies retrieval performance.

#### **1. Baselines (Competitive Models)**
To validate ROUTERRETRIEVER, it was compared against the following models:

1. **Single Model Baselines**
   - **MSMARCO Model**: A single embedding model trained on the large-scale MSMARCO dataset.
   - **Multi-Task Training Model**: A model trained on both MSMARCO and multiple domain-specific datasets.

2. **Routing Baselines**
   - **ExpertClassifierRouter**: Uses probability estimation for expert selection.
   - **ClassificationHeadRouter**: Uses a classifier head to select the best expert.
   - **DatasetRouter**: Selects the expert based on dataset labels.

3. **Oracle Models (Upper Bound Performance)**
   - **DatasetOracle**: Routes all queries in a dataset to the best-performing expert for that dataset.
   - **InstanceOracle**: Selects the best-performing expert dynamically for each query.

#### **2. Test Data (BEIR Benchmark)**
The model was evaluated on multiple datasets from **BEIR (Benchmarking Information Retrieval)**, covering diverse retrieval tasks:

- **MSMARCO**: General-purpose search dataset.
- **Quora**: Duplicate question retrieval.
- **ArguAna**: Argument analysis retrieval.
- **HotpotQA**: Multi-hop question answering.
- **NFCorpus**: Medical information retrieval.
- **SciFact**: Scientific claim verification.
- **FiQA**: Finance question answering.

Additional **zero-shot generalization** was tested on **TREC-COVID, SciDocs, DBPedia, FEVER, and NaturalQuestions**.

#### **3. Key Results (Performance Analysis)**
1. **ROUTERRETRIEVER vs Single Model Baselines**
   - Outperforms MSMARCO model by **+2.1 absolute nDCG@10**.
   - Outperforms Multi-Task model by **+3.2 absolute nDCG@10**.

2. **ROUTERRETRIEVER vs Routing Techniques**
   - Outperforms other routing approaches (DatasetRouter, ClassificationHeadRouter, ExpertClassifierRouter) by an **average of +1.8**.
   - Achieves more **consistent performance across domains** compared to DatasetRouter.

3. **ROUTERRETRIEVER vs Oracle Models**
   - Achieves performance **close to DatasetOracle**, reaching **50.9 nDCG@10** on average.
   - Performs below **InstanceOracle**, suggesting room for further improvements in routing accuracy.

4. **Effect of Increasing Experts**
   - Performance **improves as more experts are added**.
   - However, **diminishing returns** occur beyond five experts.

5. **Zero-shot Performance**
   - Outperforms a single MSMARCO-trained model even in **datasets without dedicated domain experts**.
   - This indicates that **a mixture of domain-specific experts is more effective** than a general-purpose embedding model.

#### **4. Conclusion & Implications**
- **ROUTERRETRIEVER consistently outperforms single models and multi-task learning approaches**.
- **Easily scalable** to new domains without requiring retraining.
- **More effective routing mechanism** than existing language model-based routing methods.
- **Future improvements** can enhance routing accuracy to match **InstanceOracle** performance.

These results highlight the **effectiveness of expert-based retrieval models** and the **importance of adaptive routing** for diverse retrieval tasks.




<br/>
# 예제  



---

#### **1. 훈련 데이터 예제**
ROUTERRETRIEVER는 다양한 도메인에서 학습된 전문가 모델을 활용합니다. 각 도메인별 데이터셋을 기반으로 특정 전문가를 훈련하며, 아래는 BEIR 데이터셋의 예제입니다.

##### **(1) MSMARCO (일반 웹 검색)**
- **질의(Query)**: *What are the symptoms of COVID-19?*
- **문서(Document)**: *The symptoms of COVID-19 include fever, cough, and shortness of breath. Some patients may also experience loss of taste and smell.*
- **관련도(Relatedness Score)**: 1 (관련 있음)

##### **(2) SciFact (과학적 주장 검증)**
- **질의(Query)**: *Does vitamin D help prevent the flu?*
- **문서(Document)**: *A study suggests that vitamin D supplementation reduces the risk of respiratory infections, but further research is needed to confirm its effectiveness for the flu.*
- **관련도(Relatedness Score)**: 1 (관련 있음)

##### **(3) NFCorpus (의료 정보 검색)**
- **질의(Query)**: *What are the side effects of ibuprofen?*
- **문서(Document)**: *Ibuprofen can cause stomach pain, nausea, and, in some cases, kidney damage with prolonged use.*
- **관련도(Relatedness Score)**: 1 (관련 있음)

---

#### **2. 테스트 데이터 예제**
ROUTERRETRIEVER는 훈련된 전문가 모델을 사용하여 특정 도메인의 검색 성능을 평가합니다. 테스트 데이터는 도메인별로 다르게 구성됩니다.

##### **(1) HotpotQA (다중 문서 질의응답)**
- **질의(Query)**: *Who was the president of the U.S. when NASA was founded?*
- **문서(Document 1)**: *NASA was established on July 29, 1958, during the presidency of Dwight D. Eisenhower.*
- **문서(Document 2)**: *Dwight D. Eisenhower served as the 34th president of the United States from 1953 to 1961.*
- **정답(Answer)**: *Dwight D. Eisenhower*

##### **(2) FiQA (금융 질의응답)**
- **질의(Query)**: *Is Apple stock a good investment in 2024?*
- **문서(Document)**: *Apple's stock is projected to rise due to strong earnings, new product launches, and increasing demand for AI-driven devices.*
- **관련도(Relatedness Score)**: 1 (관련 있음)

##### **(3) ArguAna (논증 검색)**
- **질의(Query)**: *Should social media platforms be regulated for misinformation?*
- **문서(Document)**: *Some experts argue that regulating social media is necessary to curb misinformation, while others believe it infringes on free speech.*
- **관련도(Relatedness Score)**: 0.8 (부분 관련 있음)

---

#### **3. 논문에서 다룬 테스크 예제**
ROUTERRETRIEVER는 다양한 도메인별 검색 작업을 수행할 수 있으며, 주요 테스크의 예시는 다음과 같습니다.

##### **(1) 정보 검색 (Information Retrieval)**
- **질의(Query)**: *What is the capital of Canada?*
- **문서(Document)**: *Ottawa is the capital city of Canada, known for its Parliament Hill and historical landmarks.*
- **정확한 검색 모델은 Ottawa를 포함한 문서를 상위에 배치해야 함.**

##### **(2) 문서 순위 결정 (Document Ranking)**
- **입력(Input)**: 질의와 다수의 문서 리스트  
- **출력(Output)**: 가장 관련성이 높은 문서 순으로 정렬  
- **예제**:
  - **질의**: *Who discovered penicillin?*
  - **문서 1 (관련도 1.0)**: *Alexander Fleming discovered penicillin in 1928.*
  - **문서 2 (관련도 0.7)**: *Penicillin is a widely used antibiotic.*
  - **문서 3 (관련도 0.4)**: *Medical research has led to the development of many antibiotics.*

##### **(3) 전문가 라우팅 (Expert Routing)**
- **입력(Input)**: 특정 도메인의 질의
- **출력(Output)**: 해당 도메인에 적합한 전문가 모델 선택
- **예제**:
  - **질의**: *What is the latest research on black holes?* → **과학 분야 전문가 모델 선택**
  - **질의**: *How do stock market trends impact cryptocurrency?* → **금융 분야 전문가 모델 선택**
  - **질의**: *What are common side effects of antibiotics?* → **의료 분야 전문가 모델 선택**

---



---

#### **1. Training Data Examples**
ROUTERRETRIEVER trains multiple domain-specific expert models. The following are sample queries from the BEIR dataset:

##### **(1) MSMARCO (General Web Search)**
- **Query**: *What are the symptoms of COVID-19?*
- **Document**: *The symptoms of COVID-19 include fever, cough, and shortness of breath. Some patients may also experience loss of taste and smell.*
- **Relevance Score**: 1 (Relevant)

##### **(2) SciFact (Scientific Claim Verification)**
- **Query**: *Does vitamin D help prevent the flu?*
- **Document**: *A study suggests that vitamin D supplementation reduces the risk of respiratory infections, but further research is needed to confirm its effectiveness for the flu.*
- **Relevance Score**: 1 (Relevant)

##### **(3) NFCorpus (Medical Information Retrieval)**
- **Query**: *What are the side effects of ibuprofen?*
- **Document**: *Ibuprofen can cause stomach pain, nausea, and, in some cases, kidney damage with prolonged use.*
- **Relevance Score**: 1 (Relevant)

---

#### **2. Test Data Examples**
The model was evaluated on various datasets from the **BEIR benchmark**.

##### **(1) HotpotQA (Multi-hop Question Answering)**
- **Query**: *Who was the president of the U.S. when NASA was founded?*
- **Document 1**: *NASA was established on July 29, 1958, during the presidency of Dwight D. Eisenhower.*
- **Document 2**: *Dwight D. Eisenhower served as the 34th president of the United States from 1953 to 1961.*
- **Answer**: *Dwight D. Eisenhower*

##### **(2) FiQA (Financial Question Answering)**
- **Query**: *Is Apple stock a good investment in 2024?*
- **Document**: *Apple's stock is projected to rise due to strong earnings, new product launches, and increasing demand for AI-driven devices.*
- **Relevance Score**: 1 (Relevant)

##### **(3) ArguAna (Argument Retrieval)**
- **Query**: *Should social media platforms be regulated for misinformation?*
- **Document**: *Some experts argue that regulating social media is necessary to curb misinformation, while others believe it infringes on free speech.*
- **Relevance Score**: 0.8 (Partially Relevant)

---

#### **3. Task Examples in the Paper**
ROUTERRETRIEVER performs multiple retrieval tasks, including:

##### **(1) Information Retrieval**
- **Query**: *What is the capital of Canada?*
- **Document**: *Ottawa is the capital city of Canada, known for its Parliament Hill and historical landmarks.*
- **A good retrieval model should rank documents mentioning "Ottawa" at the top.**

##### **(2) Document Ranking**
- **Input**: A query and a list of documents.
- **Output**: Ranking documents based on relevance.
- **Example**:
  - **Query**: *Who discovered penicillin?*
  - **Document 1 (Relevance 1.0)**: *Alexander Fleming discovered penicillin in 1928.*
  - **Document 2 (Relevance 0.7)**: *Penicillin is a widely used antibiotic.*
  - **Document 3 (Relevance 0.4)**: *Medical research has led to the development of many antibiotics.*

##### **(3) Expert Routing**
- **Input**: Query from a specific domain.
- **Output**: Selecting the appropriate expert model.
- **Example**:
  - **Query**: *What is the latest research on black holes?* → **Science Expert Selected**
  - **Query**: *How do stock market trends impact cryptocurrency?* → **Finance Expert Selected**
  - **Query**: *What are common side effects of antibiotics?* → **Medical Expert Selected**




<br/>  
# 요약   




ROUTERRETRIEVER는 도메인별 전문가 모델을 조합하여 최적의 검색 성능을 제공하는 라우팅 기반 검색 모델로, 파일럿 임베딩을 활용한 동적 전문가 선택 방식을 적용한다. 실험 결과, BEIR 벤치마크에서 단일 모델 및 기존 라우팅 기법보다 높은 nDCG@10 성능을 기록하며, 특정 도메인 전문가 없이도 일반 모델보다 우수한 검색 성능을 보였다. 다양한 도메인의 질의에 대해 적절한 전문가 모델을 선택하는 방식으로, 예제에서는 의료, 금융, 과학 분야 등에서 효과적인 검색 결과를 제공하는 사례를 확인할 수 있다.  

---


ROUTERRETRIEVER is a routing-based retrieval model that leverages domain-specific expert models and dynamically selects the most suitable expert using pilot embeddings. Experimental results show that it outperforms single models and existing routing techniques on the BEIR benchmark, achieving higher nDCG@10 scores and demonstrating superior retrieval performance even without a dedicated expert for a given domain. By selecting appropriate expert models for queries from various domains, examples illustrate its effectiveness in retrieving relevant information across medical, financial, and scientific fields.


<br/>  
# 기타  




---

### **1. Figure 1: ROUTERRETRIEVER 아키텍처 개요**  
**설명:**  
ROUTERRETRIEVER의 동작 과정을 시각적으로 표현한 다이어그램입니다.  
1. **질의(Query)**가 입력되면, **기본 인코더(Base Encoder)**가 질의 임베딩을 생성합니다.  
2. 이 임베딩은 **파일럿 임베딩(Pilot Embeddings)**과 비교되어, 가장 유사한 전문가(Expert)가 선택됩니다.  
3. 선택된 전문가(예: Expert A)가 질의를 다시 처리하여 최종 검색 임베딩을 생성합니다.  
**핵심 개념:**  
- 각 전문가(Expert)는 특정 도메인에 최적화된 LoRA 기반 모듈입니다.  
- 파일럿 임베딩을 활용하여 최적의 전문가를 자동으로 선택합니다.  

---

### **2. Table 1: ROUTERRETRIEVER vs 경쟁 모델 성능 비교 (BEIR 데이터셋, nDCG@10 기준)**  
**설명:**  
- BEIR 데이터셋에서 ROUTERRETRIEVER가 다른 검색 모델보다 우수한 성능을 보이는지를 정량적으로 평가한 테이블입니다.  
- **비교 모델**:  
  - 단일 모델 (MSMARCO 기반, Multi-task 학습)  
  - 기존 라우팅 기법 (ExpertClassifierRouter, ClassificationHeadRouter, DatasetRouter)  
  - 오라클(Oracle) 모델 (DatasetOracle, InstanceOracle)  

**주요 결과:**  
- ROUTERRETRIEVER는 **MSMARCO 단일 모델 대비 +2.1점, Multi-Task 모델 대비 +3.2점** 성능 향상.  
- 기존 라우팅 기법보다 평균 **+1.8점 높은 nDCG@10** 성능 기록.  
- 오라클 모델(DatasetOracle)에 근접한 성능을 보이며, InstanceOracle에는 미치지 못함.  

---

### **3. Table 2: 전문가 없는 데이터셋에서도 ROUTERRETRIEVER 성능 유지**  
**설명:**  
- 특정 도메인 전문가(Expert)가 없는 환경에서 ROUTERRETRIEVER가 어떻게 일반화되는지 평가한 결과.  
- "w/ Experts" (전문가 포함 데이터셋)와 "w/o Experts" (전문가 없는 데이터셋)로 나누어 성능 비교.  
- ROUTERRETRIEVER는 전문가가 없는 도메인에서도 MSMARCO 단일 모델보다 높은 성능을 유지함.  
**결론:**  
- 전문가 모델 없이도 기존 단일 모델 대비 높은 일반화 성능을 보이며, 이는 다양한 도메인에 적합한 검색 방식임을 입증.  

---

### **4. Figure 2: BEIR 데이터셋의 TSNE 임베딩 시각화**  
**설명:**  
- BEIR 데이터셋의 질의 임베딩을 TSNE 방식으로 2D 공간에 투영한 그래프.  
- **군집 분석 결과:**  
  - **일반 도메인 데이터셋 (MSMARCO, ArguAna)**: 넓게 분포 → 범용성이 높은 검색 데이터  
  - **도메인 특화 데이터셋 (HotpotQA, SciFact, NFCorpus, FiQA)**: 특정 영역에 밀집 → 특정 도메인에 최적화됨  
**결론:**  
- 도메인별 전문가 모델이 필요한 이유를 설명하는 그래프.  
- 특정 도메인 데이터는 밀집된 패턴을 보이며, 일반 검색 모델이 이를 효과적으로 처리하기 어려움.  

---

### **5. Figure 3: 훈련 데이터 크기 vs 모델 성능 (nDCG@10 기준)**  
**설명:**  
- 단일 전문가 모델의 성능이 훈련 데이터 크기에 따라 어떻게 변화하는지 분석한 그래프.  
- **결과:**  
  - 도메인 내 성능(in-domain performance): 데이터 크기가 증가할수록 성능 증가.  
  - 도메인 외 성능(out-of-domain performance): 데이터 크기가 증가해도 성능 개선 없음.  
- **결론:**  
  - 단일 대규모 데이터셋을 학습하는 것보다 여러 도메인별 전문가를 활용하는 것이 효과적임.  

---

### **6. Figure 4 & Figure 5: 전문가 수 증가에 따른 성능 변화**  
**설명:**  
- Figure 4: 전문가 수가 증가할 때 ROUTERRETRIEVER의 평균 성능(nDCG@10)이 어떻게 변하는지 분석.  
- Figure 5: InstanceOracle 기반의 최적의 전문가 선택이 가능한 경우, 성능이 얼마나 증가할 수 있는지 평가.  
**결과:**  
- **전문가 수가 증가할수록 성능 향상** → 하지만 일정 수준 이상에서는 **수확 체감 (Diminishing Returns)** 발생.  
- **InstanceOracle 기반의 전문가 선택이 가능할 경우 성능이 더 향상될 가능성 존재**.  

---

### **7. Table 3: 전문가 추가에 따른 성능 변화 (Sequential Expert Addition Analysis)**  
**설명:**  
- 전문가를 하나씩 추가하면서 검색 성능이 어떻게 변화하는지 평가.  
- 초기에는 특정 도메인의 전문가 추가가 다른 도메인의 성능에 영향을 미치지만, 전문가 수가 많아질수록 변화폭이 적어짐.  
- **결론:**  
  - 적절한 수의 전문가를 추가하면 성능 향상을 극대화할 수 있음.  
  - 하지만 일정 수준 이상 전문가를 추가하면 성능 향상 효과가 점점 감소.  

---

### **8. Figure 6: ROUTERRETRIEVER vs InstanceOracle 라우팅 비교**  
**설명:**  
- 어떤 질의가 어떤 전문가 모델로 라우팅되는지를 분석.  
- **InstanceOracle (a)**: 특정 도메인의 전문가 모델을 적극적으로 활용.  
- **ROUTERRETRIEVER (b)**: 데이터셋 경계를 따르는 패턴을 보이며, 일부 질의에서 최적 전문가를 선택하지 못함.  
**결론:**  
- ROUTERRETRIEVER의 라우팅 방식 개선 여지가 있음.  
- InstanceOracle처럼 보다 세밀한 전문가 선택이 가능하다면 성능 향상 가능.  

---


### **1. Figure 1: ROUTERRETRIEVER Architecture**  
- Illustrates the workflow of ROUTERRETRIEVER.  
- Queries are first encoded using the **Base Encoder**.  
- The model selects the best expert based on **pilot embedding similarity**.  
- The **selected expert** then processes the query to generate the final embedding.  

---

### **2. Table 1: Performance Comparison on BEIR Benchmark (nDCG@10)**  
- **ROUTERRETRIEVER vs Single Model & Routing Techniques.**  
- Outperforms MSMARCO by **+2.1 nDCG@10** and Multi-Task by **+3.2 nDCG@10**.  
- Achieves higher scores than standard routing techniques (**+1.8 on average**).  

---

### **3. Table 2: Performance in Absence of Domain Experts**  
- Evaluates generalization performance when no domain expert is present.  
- ROUTERRETRIEVER **outperforms MSMARCO even without a dedicated expert**.  

---

### **4. Figure 2: TSNE Visualization of BEIR Dataset Embeddings**  
- **General-domain datasets** (e.g., MSMARCO) are widely spread.  
- **Domain-specific datasets** (e.g., SciFact, NFCorpus) are tightly clustered.  

---

### **5. Figure 3: Impact of Training Data Size on Performance**  
- **In-domain performance improves with more training data**.  
- **Out-of-domain performance does not significantly improve**.  

---

### **6. Figure 4 & 5: Impact of Increasing Experts**  
- Performance improves with more experts, but **diminishing returns occur**.  
- InstanceOracle suggests **potential for further improvements**.  

---

### **7. Table 3: Performance Changes with Expert Additions**  
- Initially, adding experts significantly impacts performance.  
- Beyond a certain number, **additional experts provide minimal improvement**.  

---

### **8. Figure 6: Expert Selection Patterns in ROUTERRETRIEVER vs InstanceOracle**  
- **ROUTERRETRIEVER follows dataset boundaries**, while **InstanceOracle selects more appropriate experts dynamically**.  
- Indicates room for **improving the routing mechanism**.


<br/>
# refer format:     


@inproceedings{lee2025routerretriever,
  author    = {Hyunji Lee and Luca Soldaini and Arman Cohan and Minjoon Seo and Kyle Lo},
  title     = {ROUTERRETRIEVER: Routing over a Mixture of Expert Embedding Models},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  publisher = {Association for the Advancement of Artificial Intelligence},
  url       = {https://arxiv.org/abs/2409.02685}
}



Lee, Hyunji, Luca Soldaini, Arman Cohan, Minjoon Seo, and Kyle Lo. "ROUTERRETRIEVER: Routing over a Mixture of Expert Embedding Models." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. Association for the Advancement of Artificial Intelligence. https://arxiv.org/abs/2409.02685.  





