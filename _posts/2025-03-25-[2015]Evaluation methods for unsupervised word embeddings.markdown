---
layout: post
title:  "[2015]Evaluation methods for unsupervised word embeddings"  
date:   2025-03-15 11:14:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

다소 꽤 많이 오래된 임베딩 평가 논문  
파인튜닝도 없었으니.. 임베딩은 한 번만 학습(Wikipedia 2008 3월)  
(통일된 전처리 및 벡터차원 및 사용 단어)  
단어에 해당되는 임베딩 가져와서 테스크 수행(logistic regression같은걸로 테스크 수행..또는 파싱은 그냥 파싱  )      
 
 
짧은 요약(Abstract) :    




이 논문은 비지도 학습 기반의 단어 임베딩(word embeddings) 기법들을 평가하는 다양한 방법을 다룬 종합적인 연구입니다. 전통적으로 단어 임베딩의 성능을 측정하는 데 사용되는 평가 방식들이 서로 다른 결과를 낳는다는 점을 지적하면서, 하나의 "최적 벡터 표현"이 존재한다는 기존의 가정을 재검토합니다.

논문은 다음과 같은 새로운 평가 방식을 제안합니다:

특정 쿼리에 대한 임베딩의 직접적인 비교 평가를 통해 편향을 줄이고 통찰력을 높이며,

크라우드소싱을 통해 빠르고 정확하게 데이터 기반의 판단을 수집합니다.

이러한 새로운 방식은 기존의 점수 기반 평가보다 더 정밀하고, 실제 사용 사례에 더 적합한 정보를 제공할 수 있습니다.



We present a comprehensive study of evaluation methods for unsupervised embedding techniques that obtain meaningful representations of words from text. Different evaluations result in different orderings of embedding methods, calling into question the common assumption that there is one single optimal vector representation. We present new evaluation techniques that directly compare embeddings with respect to specific queries. These methods reduce bias, provide greater insight, and allow us to solicit data-driven relevance judgments rapidly and accurately through crowdsourcing.





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


 **기존의 여러 비지도 학습 기반 단어 임베딩 모델을 평가하는 다양한 방법론**을 제안하고 실험하는 데 초점을 맞춘 논문   

**제안한 평가 메서드(evaluation methods)**와 **사용한 임베딩 모델 및 트레이닝 데이터**에 대한 설명을 아래와 같이 정리.     

---



###  제안된 평가 방법

1. **절대적 평가 (Absolute Evaluation)**
   - 기존 평가 방식으로, 단어쌍 유사도, 유추(analogy), 범주화(categorization), 선택적 선호(selectional preference) 등 기존 데이터셋을 활용하여 모델을 평가.
   - 문제점: 데이터셋이 편향되거나 임베딩 모델 간의 정밀한 비교가 어려움.

2. **비교 평가 (Comparative Intrinsic Evaluation)**
   - 저자들이 제안한 방식으로, **여러 임베딩 모델이 반환한 단어 중 어떤 것이 가장 적절한지** 사람들에게 직접 묻는 방식.
   - 크라우드소싱(MTurk)을 통해 수집된 사람의 선택 데이터를 기반으로 모델 비교 수행.
   - 평가 항목: 쿼리 단어의 이웃 단어(top-k)에 대한 선호도 비교.

3. **코히런스(Coherence) 평가**
   - 단어 임베딩 공간에서 가까운 단어들의 집합이 서로 의미적으로 얼마나 관련 있는지를 평가.
   - "Intruder Detection" 태스크: 세 개의 관련 단어와 하나의 무관한 단어를 제시하고, 사람이 무관한 단어(침입자)를 골라내는 방식.

---

### 사용된 임베딩 모델

총 6개의 **기존 비지도 단어 임베딩 모델**이 비교 대상이 되었어요:

1. **CBOW (Continuous Bag of Words)** – Word2Vec의 한 방식.
2. **C&W 모델** – Collobert & Weston 모델.
3. **GloVe** – 전역 통계 기반 임베딩.
4. **TSCCA** – Two-step Canonical Correlation Analysis.
5. **Hellinger PCA** – H-PCA, 통계적 PCA 기법.
6. **Random Projections** – 매우 희소한 임의 투영 기법.

---

###  트레이닝 데이터

- **Wikipedia 2008년 3월 덤프**를 사용하여 모든 임베딩을 학습.
- **Stanford Tokenizer**로 토크나이징.
- **소문자 처리 및 숫자 0으로 치환** 등 전처리 수행.
- 모든 모델은 **50차원 벡터**를 사용하고, **103,647개 단어**의 공통된 어휘 집합을 기반으로 실험.

---



###  Proposed Evaluation Methods

1. **Absolute Intrinsic Evaluation**
   - Traditional evaluation using standard datasets (e.g., WordSim-353, MEN) measuring word similarity, analogy, categorization, and selectional preference.
   - Drawback: Biased datasets, hard to make fine-grained model comparisons.

2. **Comparative Intrinsic Evaluation** *(Proposed)*
   - The authors propose a method where human annotators compare the nearest neighbors retrieved by different embeddings for a given query word.
   - Judgments are collected through **crowdsourcing on Mechanical Turk**.
   - Directly compares embeddings based on human preferences rather than metric aggregation.

3. **Coherence Evaluation**
   - Measures how semantically coherent the neighborhoods of words are in the embedding space.
   - "Intruder Detection" task: Three similar words and one unrelated word are shown; users select the unrelated one.
   - Designed to assess the **local semantic consistency** of embeddings.

---

###  Embedding Models Used

Six **unsupervised word embeddings** were evaluated:

1. **CBOW (Continuous Bag of Words)** – from Word2Vec.
2. **C&W** – Collobert & Weston model.
3. **GloVe** – Global Vectors for Word Representation.
4. **TSCCA** – Two-step Canonical Correlation Analysis.
5. **Hellinger PCA** – PCA-based distributional model.
6. **Random Projection** – Very sparse random projections.

---

### Training Data

- All embeddings were trained on a **Wikipedia dump from March 1, 2008**.
- **Stanford Tokenizer** was used.
- **Lowercasing and digit normalization** (digits replaced with zeros) were applied.
- All embeddings are in **50-dimensional space**, and only the **intersection vocabulary of 103,647 words** was used for comparison.

---


 
<br/>
# Results  


---



###  1. 절대적 내재적 평가 (Absolute Intrinsic Evaluation)

- **테스크**:  
  1. **단어 관련성(relatedness)**  
  2. **단어 유추(analogy)**  
  3. **범주화(categorization)**  
  4. **선택적 선호(selectional preference)**

- **테스트 데이터셋**:
  - WordSim-353, MEN, TOEFL, Battig, ESSLLI, AP, MCrae, AN, AnSem 등 총 **14개 데이터셋** 사용

- **메트릭**:
  - 관련성: 인간 유사도 점수와의 **Spearman/Pearson 상관계수**
  - 유추: 정확도 (Accuracy)
  - 범주화: 군집 **Purity**
  - 선택적 선호: 상관계수

- **결과 요약**:
  - **CBOW**가 14개 중 10개 테스크에서 최고 성능
  - **GloVe**와 **TSCCA**는 중간 수준
  - **Random Projection**, **Hellinger PCA**는 낮은 성능

---

###  2. 비교 평가 (Comparative Intrinsic Evaluation)

- **테스크**:  
  - 사용자들에게 주어진 쿼리 단어에 대해 **가장 관련 있는 이웃 단어(top-1, 5, 50)**를 선택하게 함

- **테스트 세팅**:
  - 100개의 쿼리 단어 (품사와 추상성/구체성 다양화)
  - 각 단어당 6개의 임베딩 모델 결과 제시
  - **MTurk** 사용자에게 클릭 기반 선택 수집

- **메트릭**:
  - **승률(win ratio)**: 모델이 선택된 비율

- **결과 요약**:
  - CBOW > GloVe > TSCCA > C&W > H-PCA > Random
  - CBOW는 특히 top-1에서 강력하지만 top-50에서는 GloVe가 더 좋을 수 있음

---

###  3. 코히런스(Coherence) 평가 (Intruder Detection)

- **테스크**:  
  - 단어 세 개 + 침입자 단어 하나를 제시 → 침입자 골라내기

- **메트릭**:
  - **정확도 (Precision)**: 사용자들이 침입자를 맞힌 비율

- **결과 요약**:
  - **TSCCA, CBOW, GloVe**가 상위권
  - Random Projection은 최하위
  - 낮은 빈도 단어에서는 모델 성능 격차가 커짐

---

###  4. 외재적 평가 (Extrinsic Evaluation)

#### 1) **명사구 청킹 (Chunking)**
- **데이터셋**: CoNLL 2000
- **모델**: CRF (조건부 무작위장)
- **메트릭**: F1 Score  
- **결과 요약**: C&W, TSCCA > CBOW > GloVe > Others

#### 2) **감성 분류 (Sentiment Analysis)**
- **데이터셋**: IMDB 리뷰 (Maas et al., 2011)
- **모델**: Logistic Regression (LIBLINEAR)
- **메트릭**: F1 Score  
- **결과 요약**: CBOW > TSCCA > GloVe > Others

- **종합 결론**:
  - 테스크에 따라 성능 좋은 임베딩이 다름 → **범용적 최고 모델은 없음**
  - 단어 빈도(frequency)가 성능에 영향을 많이 미침

---



###  1. Absolute Intrinsic Evaluation

- **Tasks**:
  1. **Word Relatedness**
  2. **Analogy**
  3. **Categorization**
  4. **Selectional Preference**

- **Test Datasets**:
  - WordSim-353, MEN, TOEFL, Battig, ESSLLI, AP, MCrae, AN, AnSem, etc. (14 in total)

- **Metrics**:
  - Relatedness: **Spearman/Pearson correlation**
  - Analogy: **Accuracy**
  - Categorization: **Purity**
  - Selectional Preference: **Correlation coefficient**

- **Key Findings**:
  - **CBOW** performed best on 10/14 tasks
  - **GloVe** and **TSCCA** were middle-tier
  - **Random Projection** and **Hellinger PCA** underperformed

---

###  2. Comparative Intrinsic Evaluation

- **Task**:
  - Users chose the most relevant neighbor word among top-1, 5, or 50 candidates generated by each embedding

- **Setup**:
  - 100 diverse query words
  - Amazon MTurk annotators evaluated neighbor choices
  - Responses were collected per query and rank

- **Metric**:
  - **Win Ratio**: Proportion of times an embedding’s result was preferred

- **Results**:
  - Ranking: **CBOW > GloVe > TSCCA > C&W > H-PCA > Random**
  - CBOW strongest at top-1; GloVe better at top-50

---

###  3. Coherence Evaluation (Intruder Detection)

- **Task**:
  - Present 3 similar words + 1 intruder → users select the intruder

- **Metric**:
  - **Precision**: % of annotators who correctly identified the intruder

- **Findings**:
  - **TSCCA, CBOW, GloVe** showed high coherence
  - **Random Projection** was weakest
  - Frequency affected model robustness (rare words harder)

---

###  4. Extrinsic Evaluation

#### 1) **Noun Phrase Chunking**
- **Dataset**: CoNLL 2000
- **Model**: Conditional Random Fields (CRF)
- **Metric**: **F1 Score**
- **Result**: C&W and TSCCA best; CBOW and GloVe performed well

#### 2) **Sentiment Analysis**
- **Dataset**: IMDB Reviews (Maas et al., 2011)
- **Model**: Logistic Regression (LIBLINEAR)
- **Metric**: **F1 Score**
- **Result**: CBOW > TSCCA > GloVe

---


       






<br/>
# 예제  


**임베딩 모델(CBOW, GloVe 등)**을 동일한 조건에서 학습시키고, 다양한 **평가 데이터셋과 사용자 태스크**를 통해 **정량적/정성적 비교**를 수행했어요.  




---


###  1. 트레이닝 데이터 예시

- **Corpus**: Wikipedia 2008년 3월 덤프
- **전처리**:  
  - Stanford Tokenizer로 토큰화  
  - 모든 단어를 소문자로 변환  
  - 숫자는 0으로 치환 (예: “2023년” → “0000년”)

- **예시 문장** (가상의 예):
  ```
  "Natural language processing (NLP) is a subfield of artificial intelligence."
  → "natural language processing nlp is a subfield of artificial intelligence ."
  ```

이런 문장들이 학습에 사용되어 단어 벡터를 학습함.  
예: `"language"`와 `"processing"`이 자주 함께 등장 → 가까운 임베딩 벡터로 학습됨.

---

###  2. 테스트 데이터 및 태스크 예시

#### 1) **단어 관련성 (Relatedness)**

- **데이터셋**: WordSim-353  
- **입력 쌍**:
  - (car, automobile)
  - (student, school)
  - (coffee, tree)

- **출력**:
  - 인간 유사도 점수 vs. 모델 cosine similarity 비교
  - 예: 인간 점수 = 9.1 / 모델 similarity = 0.82 → 상관계수로 평가

---

#### 2) **단어 유추 (Analogy)**

- **예시 쿼리**:  
  - “man : king = woman : ?”  
- **출력**:  
  - 모델이 ‘queen’을 찾을 수 있는지 여부

---

#### 3) **비교 평가 태스크 (Comparative Intrinsic Evaluation)**

- **입력 쿼리**: `skillfully`
- **각 임베딩이 반환한 top-1 단어**:
  - (a) swiftly – C&W  
  - (b) expertly – CBOW, GloVe, TSCCA  
  - (c) cleverly – Random  
  - (d) pointedly – H-PCA

- **MTurk 사용자에게 질문**:  
  > “Which of the following words is most similar to ‘skillfully’?”

- **출력**:  
  - 사용자 클릭 수로 가장 유사하다고 판단된 임베딩 선택 → win ratio 계산

---

#### 4) **코히런스 평가 (Intruder Detection)**

- **입력 단어군**:
  - (a) finally (쿼리)
  - (b) eventually (이웃)
  - (c) immediately (이웃)
  - (d) put (침입자)

- **태스크**:  
  > “Which word doesn’t belong in this group?”

- **출력**:  
  - 침입자 선택 정확도(정답률)

---



###  1. Training Data Example

- **Corpus**: Wikipedia dump from March 2008  
- **Preprocessing**:
  - Tokenization using **Stanford Tokenizer**
  - Lowercasing all words
  - Replacing digits with zeros (e.g., “2023” → “0000”)

- **Example Sentence**:
  ```
  Original: "Natural language processing (NLP) is a subfield of artificial intelligence."
  Preprocessed: "natural language processing nlp is a subfield of artificial intelligence ."
  ```

These sentences help the model learn that `"language"` and `"processing"` are semantically close.

---

###  2. Test Data & Task Input-Output Examples

#### 1) **Word Relatedness**

- **Dataset**: WordSim-353  
- **Input Pairs**:
  - (car, automobile), (student, school), (coffee, tree)
- **Output**:
  - Compare model’s **cosine similarity** with **human scores**
  - e.g., Human: 9.1 / Model: 0.82 → evaluated by Spearman correlation

---

#### 2) **Word Analogy Task**

- **Input Query**:
  > “man is to king as woman is to ?”
- **Expected Output**:
  - Model should return: `queen`

---

#### 3) **Comparative Evaluation Task**

- **Query Word**: `skillfully`
- **Top results from each embedding**:
  - (a) swiftly (C&W)
  - (b) expertly (CBOW, GloVe, TSCCA)
  - (c) cleverly (Random Projection)
  - (d) pointedly (H-PCA)

- **MTurk Question**:
  > “Which word is most similar to ‘skillfully’?”

- **Output**:
  - Selected word → embedding win ratio calculated

---

#### 4) **Coherence / Intruder Detection Task**

- **Word Set**:
  - (a) finally (query)
  - (b) eventually (neighbor)
  - (c) immediately (neighbor)
  - (d) put (intruder)

- **Task**:
  > “Which word does not belong?”

- **Output**:
  - Accuracy: % of users who picked the intruder correctly

---







<br/>  
# 요약   





이 논문은 단어 임베딩을 평가하기 위한 새로운 비교 기반 평가 메서드와 코히런스 평가를 제안한다.  
CBOW가 대부분의 관련성 평가에서 우수했지만, 테스크에 따라 성능이 달라 다양한 기준이 필요함을 보여준다.  
예제는 WordSim-353 유사도 판단, 유추 문제(king:man=queen:? 등), 침입어 탐지 등 실제 사용자 입력 기반 태스크를 포함한다.

---


This paper proposes novel comparative evaluation and coherence-based methods for assessing word embeddings.  
CBOW outperforms others on most relatedness tasks, but results vary by task, highlighting the need for multiple evaluation criteria.  
Examples include similarity judgments from WordSim-353, analogy tasks (e.g., king:man=queen:?), and intruder detection by human users.

---




<br/>  
# 기타  






---

##  기타 구성 요소 요약 (한글)

- **테이블**:
  - **Table 1**: 14개 내재적 평가 데이터셋에 대한 6개 임베딩 모델의 정량적 성능 결과 (관련성, 유추, 범주화 등)
  - **Table 2**: 사용자 비교 평가 태스크의 실제 예시 (쿼리 단어 `skillfully`에 대한 후보 단어들)
  - **Table 3**: 코히런스 평가에서 사용된 침입자 탐지 예시 (`finally`, `eventually`, `immediately`, `put`)
  - **Table 4 & 5**: 외재적 평가에서 청킹과 감성분류 F1 점수 및 통계적 유의성 (p-value) 결과

- **그림 (Figures)**:
  - **Figure 1(a–d)**: 비교 평가 결과를 단어 빈도, 이웃 순위, 품사, 추상성 기준으로 시각화한 막대 그래프
  - **Figure 2**: 침입자 탐지 실험의 평균 정밀도를 단어 빈도에 따라 시각화한 그래프
  - **Figure 3**: 단어 임베딩 벡터만으로 단어 빈도를 예측한 정확도 결과 (Frequency Classification)
  - **Figure 4**: 단어 빈도와 이웃 랭킹 간의 상관관계를 보여주는 파워 법칙 그래프



---



- **Tables**:
  - **Table 1**: Performance comparison of six embedding models across 14 intrinsic evaluation datasets (relatedness, analogy, categorization, etc.)
  - **Table 2**: Example item from the comparative evaluation task using the query word `skillfully`
  - **Table 3**: Intruder detection example showing a word group with one unrelated word
  - **Tables 4 & 5**: F1 scores and p-values for extrinsic tasks (chunking and sentiment analysis)

- **Figures**:
  - **Figure 1 (a–d)**: Bar plots showing comparative evaluation results based on word frequency, neighbor rank, part-of-speech, and abstractness
  - **Figure 2**: Intrusion detection precision across word frequencies
  - **Figure 3**: Frequency prediction accuracy using word embeddings alone
  - **Figure 4**: Power-law relationship between corpus frequency and nearest-neighbor rank in embeddings


---






<br/>
# refer format:     

@inproceedings{schnabel2015evaluation,
  title={Evaluation methods for unsupervised word embeddings},
  author={Schnabel, Tobias and Labutov, Igor and Mimno, David and Joachims, Thorsten},
  booktitle={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={298--307},
  year={2015},
  organization={Association for Computational Linguistics},
  address={Lisbon, Portugal}
}




Schnabel, Tobias, Igor Labutov, David Mimno, and Thorsten Joachims. 2015. “Evaluation Methods for Unsupervised Word Embeddings.” Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP), 298–307. Lisbon, Portugal: Association for Computational Linguistics.