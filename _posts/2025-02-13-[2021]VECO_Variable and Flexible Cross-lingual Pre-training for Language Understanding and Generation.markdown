---
layout: post
title:  "[2021]VECO: Variable and Flexible Cross-lingual Pre-training for Language Understanding and Generation"  
date:   2025-02-13 11:52:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

Cross-Attention(Self-Attention은 문장 내 단어들 사이의 관계를 학습 Cross-Attention은 다른 문장 또는 다른 언어와의 관계를 학습) 사용 멀티링구얼 프리트레이닝(버트계열-인코더기반)  


짧은 요약(Abstract) :    



기존의 다국어 사전 훈련 모델은 Transformer 인코더를 여러 언어에 적용하여 교차 언어 전이를 가능하게 했다. 하지만, 대부분의 연구는 공유된 어휘와 이중언어 문맥만을 활용하여 언어 간 상관성을 구축하는데, 이러한 방식은 언어 간 문맥 표현을 효과적으로 정렬하기에는 한계가 있다. 

본 연구에서는 Transformer 인코더에 **교차 주의 (Cross-Attention) 모듈**을 추가하여 언어 간 상호 의존성을 명시적으로 구축한다. 이를 통해 단어 예측이 단일 언어 문맥에만 의존하는 문제를 해결할 수 있다. 더 나아가, 다운스트림 작업에서 이 모듈을 필요에 따라 추가하거나 제거할 수 있어 언어 이해(NLU) 및 생성(NLG) 작업 모두에서 유연하게 활용 가능하다. 

제안된 **VECO 모델**은 XTREME 벤치마크의 다양한 언어 이해 태스크(텍스트 분류, 시퀀스 라벨링, 질의응답, 문장 검색)에서 새로운 최고 성능을 기록했다. 또한, 언어 생성 작업에서도 기존 다국어 모델 및 최신 Transformer 변종보다 높은 성능을 보이며, WMT14 영어-독일어 및 영어-프랑스어 번역 데이터셋에서 BLEU 점수를 최대 1~2점 향상시켰다.

---


Existing work in multilingual pretraining has demonstrated the potential of cross-lingual transferability by training a unified Transformer encoder for multiple languages. However, much of this work only relies on the shared vocabulary and bilingual contexts to encourage the correlation across languages, which is loose and implicit for aligning the contextual representations between languages.

In this paper, we plug a **cross-attention module** into the Transformer encoder to explicitly build the interdependence between languages. This effectively prevents the degeneration of predicting masked words only conditioned on the context in its own language. More importantly, when fine-tuning on downstream tasks, the cross-attention module can be plugged in or out on demand, thus naturally benefiting a wider range of cross-lingual tasks, from language understanding to generation.

As a result, the proposed **VECO model** delivers new state-of-the-art results on various cross-lingual understanding tasks of the XTREME benchmark, covering text classification, sequence labeling, question answering, and sentence retrieval. For cross-lingual generation tasks, it also outperforms all existing cross-lingual models and state-of-the-art Transformer variants on WMT14 English-to-German and English-to-French translation datasets, with gains of up to **1–2 BLEU**.



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




VECO는 **다국어 이해(NLU) 및 생성(NLG)** 작업을 동시에 지원하는 **유연한 크로스-링구얼 사전 훈련 모델**이다. 기존의 Transformer 인코더와는 다르게 **교차 주의(Cross-Attention) 모듈**을 추가하여 언어 간 상호 의존성을 명시적으로 구축한다. 이를 통해 다국어 문맥을 더욱 효과적으로 정렬하고 학습할 수 있다.  

#### **1. 아키텍처**  
VECO는 **24개 층을 가진 Transformer 인코더**를 기반으로 하며, 각 층에는 다음과 같은 주요 모듈이 포함된다.  
- **자기 주의(Self-Attention) 모듈**: 기존 Transformer 인코더와 동일하게, 문맥 정보를 고려하여 토큰 간의 관계를 학습한다.  
- **교차 주의(Cross-Attention) 모듈**: 다른 언어의 문맥 정보를 추가적으로 활용하여 보다 강한 언어 간 정렬을 수행한다.  
- **피드포워드(Feed-Forward) 네트워크**: 각 토큰의 표현을 변환하고, 학습된 정보를 효과적으로 처리한다.  
이러한 구조를 통해 VECO는 **기존 mBERT, XLM, XLM-R과 비교해 더욱 정밀한 언어 간 표현 정렬**을 수행할 수 있다.  

#### **2. 훈련 데이터**  
VECO는 **50개 언어**를 포함하는 **대규모 단일 언어 및 이중 언어 데이터**를 활용하여 훈련되었다.  
- **단일 언어 데이터(Monolingual Data)**: CommonCrawl에서 수집한 **1.36TB 분량**의 다국어 문서(6.5억 개 문장) 사용  
- **이중 언어 데이터(Bilingual Data)**: OPUS 코퍼스에서 가져온 **6.4억 개의 병렬 문장**(879개 언어 쌍 포함) 활용  

#### **3. 훈련 목표 (Pretraining Objectives)**  
VECO는 두 가지 주요 훈련 목표를 사용한다.  
- **CA-MLM(Cross-Attention Masked Language Modeling)**: 기존 MLM과 달리, 크로스-어텐션을 통해 다른 언어의 문맥 정보를 적극적으로 반영하여 마스크된 단어를 예측한다.  
- **TLM(Translation Language Modeling)**: 병렬 문장을 활용하여 한 언어에서 다른 언어로의 전이를 학습한다.  
이러한 훈련 방식을 통해 VECO는 **언어 이해 및 생성 작업에서 기존 다국어 모델보다 강력한 성능을 발휘**한다.  

---

VECO is a **flexible cross-lingual pre-training model** designed to support both **natural language understanding (NLU) and natural language generation (NLG)**. Unlike traditional Transformer encoders, it incorporates a **cross-attention module**, which explicitly builds interdependence between languages. This allows VECO to better align multilingual contexts compared to previous models.  

#### **1. Architecture**  
VECO is built on a **24-layer Transformer encoder** and includes the following core components:  
- **Self-Attention Module**: Similar to standard Transformer encoders, it captures contextual relationships between tokens.  
- **Cross-Attention Module**: Explicitly models interdependencies between different languages, improving cross-lingual representation alignment.  
- **Feed-Forward Network**: Transforms token representations and processes learned information efficiently.  
This architecture enables VECO to **outperform models like mBERT, XLM, and XLM-R in cross-lingual representation learning**.  

#### **2. Training Data**  
VECO is trained on **a large-scale monolingual and bilingual corpus covering 50 languages**.  
- **Monolingual Data**: 1.36TB of multilingual text (6.5 billion sentences) sourced from CommonCrawl.  
- **Bilingual Data**: 6.4 billion parallel sentences from the OPUS corpus, covering 879 language pairs.  

#### **3. Pretraining Objectives**  
VECO utilizes two key pretraining objectives:  
- **CA-MLM (Cross-Attention Masked Language Modeling)**: Unlike traditional MLM, this approach leverages cross-attention to incorporate contextual information from other languages when predicting masked words.  
- **TLM (Translation Language Modeling)**: Learns cross-lingual transfer by leveraging parallel sentences.  
These objectives make VECO significantly more **effective in both language understanding and generation** than prior multilingual models.


   
 
<br/>
# Results  




VECO는 **다양한 다국어 이해(NLU) 및 생성(NLG) 작업**에서 기존 모델보다 **우수한 성능**을 보였다. 성능 비교를 위해 **XTREME 벤치마크 및 기계 번역(MT) 데이터셋**을 활용했다.  

---

#### **1. 다국어 이해(NLU) 성능 – XTREME 벤치마크**  
VECO는 **XTREME 벤치마크**(40개 언어, 9개 태스크)에서 기존 다국어 사전 훈련 모델들과 비교하여 평가되었다.  

- **비교 모델**: XLM, XLM-R, mBERT, MMTE  
- **테스트 데이터**:  
  - **문장 분류**: XNLI, PAWS-X  
  - **구조적 예측**: POS 태깅, Named Entity Recognition(NER)  
  - **질의응답(QA)**: XQuAD, MLQA, TyDiQA  
  - **문장 검색**: BUCC 2018, Tatoeba  

- **평가 메트릭**:  
  - 분류 및 검색 태스크: **정확도(Accuracy, Acc)**  
  - 구조적 예측 및 질의응답 태스크: **F1 Score 및 정확한 일치율(EM, Exact Match)**  

- **결과 요약**:  
  - VECO는 **XTREME 벤치마크에서 1위 기록**  
  - **XLM-R 대비 5.0~6.6 포인트 성능 향상**  
  - 문장 검색 태스크(BUCC, Tatoeba)에서 **XLM-R 대비 10포인트 이상 향상**, 다국어 문맥 정렬 강화  

| 모델 | XNLI (Acc) | PAWS-X (Acc) | NER (F1) | XQuAD (F1/EM) | MLQA (F1/EM) | BUCC (F1) | Tatoeba (Acc) | 평균 |
|------|-----------|-------------|---------|--------------|--------------|-----------|-------------|------|
| XLM-R | 79.2 | 86.4 | 65.4 | 76.6 / 60.8 | 71.6 / 53.2 | 66.0 | 57.3 | 68.1 |
| VECO | **84.3** | **92.8** | **71.0** | **83.9 / 70.9** | **77.5 / 59.3** | **92.6** | **91.1** | **81.0** |

---

#### **2. 다국어 생성(NLG) 성능 – WMT14 기계 번역 평가**  
VECO는 **WMT14 영어-독일어(En-De) 및 영어-프랑스어(En-Fr) 번역 태스크**에서 기존 모델과 비교되었다.  

- **비교 모델**: mBART, mRASP, XLM-R, Transformer 기반 모델  
- **테스트 데이터**:  
  - WMT14 En-De (4.5M 문장)  
  - WMT14 En-Fr (36M 문장)  
- **평가 메트릭**:  
  - **BLEU Score** (번역 성능 평가 지표)  

- **결과 요약**:  
  - VECO는 **XLM-R보다 평균 0.8 BLEU, Transformer보다 2.3 BLEU 향상**  
  - **10 Epoch만에 SacreBLEU 28점 이상 달성**, 빠른 수렴 속도  

| 모델 | WMT14 En-Fr (BLEU) | WMT14 En-De (BLEU) |
|------|------------------|------------------|
| Transformer (랜덤 초기화) | 42.9 | 28.7 |
| XLM-R | 43.8 | 30.9 |
| VECO | **44.5** | **31.7** |

---



VECO outperforms previous multilingual pre-trained models on **both cross-lingual understanding (NLU) and generation (NLG) tasks**. The performance was benchmarked on **XTREME for NLU** and **WMT14 machine translation datasets for NLG**.  

---

#### **1. Multilingual Understanding (NLU) – XTREME Benchmark**  
VECO was evaluated on the **XTREME benchmark**, covering **40 languages and 9 tasks**.  

- **Baseline Models**: XLM, XLM-R, mBERT, MMTE  
- **Test Datasets**:  
  - **Sentence Classification**: XNLI, PAWS-X  
  - **Structured Prediction**: POS tagging, Named Entity Recognition (NER)  
  - **Question Answering (QA)**: XQuAD, MLQA, TyDiQA  
  - **Sentence Retrieval**: BUCC 2018, Tatoeba  

- **Evaluation Metrics**:  
  - Classification & retrieval: **Accuracy (Acc)**  
  - Structured prediction & QA: **F1 Score and Exact Match (EM)**  

- **Results Summary**:  
  - VECO **ranked 1st on XTREME leaderboard**  
  - **5.0–6.6 points improvement over XLM-R**  
  - **Significant boost (10+ points) in retrieval tasks (BUCC, Tatoeba), improving cross-lingual alignment**  

| Model | XNLI (Acc) | PAWS-X (Acc) | NER (F1) | XQuAD (F1/EM) | MLQA (F1/EM) | BUCC (F1) | Tatoeba (Acc) | Average |
|------|-----------|-------------|---------|--------------|--------------|-----------|-------------|------|
| XLM-R | 79.2 | 86.4 | 65.4 | 76.6 / 60.8 | 71.6 / 53.2 | 66.0 | 57.3 | 68.1 |
| VECO | **84.3** | **92.8** | **71.0** | **83.9 / 70.9** | **77.5 / 59.3** | **92.6** | **91.1** | **81.0** |

---

#### **2. Multilingual Generation (NLG) – WMT14 Machine Translation**  
VECO was evaluated on the **WMT14 English-to-German (En-De) and English-to-French (En-Fr) translation tasks**.  

- **Baseline Models**: mBART, mRASP, XLM-R, Transformer-based models  
- **Test Datasets**:  
  - WMT14 En-De (4.5M sentences)  
  - WMT14 En-Fr (36M sentences)  
- **Evaluation Metric**:  
  - **BLEU Score** (standard metric for translation quality)  

- **Results Summary**:  
  - VECO **outperformed XLM-R by an average of 0.8 BLEU and Transformer by 2.3 BLEU**  
  - **Achieved 28+ SacreBLEU in just 10 epochs, demonstrating rapid convergence**  

| Model | WMT14 En-Fr (BLEU) | WMT14 En-De (BLEU) |
|------|------------------|------------------|
| Transformer (Random Init.) | 42.9 | 28.7 |
| XLM-R | 43.8 | 30.9 |
| VECO | **44.5** | **31.7** |  





<br/>
# 예제  




#### **1. 예제 훈련 데이터 (Training Data I/O)**
VECO의 훈련 데이터는 **50개 언어의 단일 언어 데이터와 이중 언어 병렬 데이터**를 포함한다.  
- **입력 (Input)**:  
  - 단일 언어 데이터: CommonCrawl에서 수집된 원문 문장  
  - 병렬 데이터: OPUS 코퍼스에서 제공하는 번역된 문장 쌍  
  - 마스크된 단어를 포함한 문장 (MLM, Masked Language Modeling 방식)  

- **출력 (Output)**:  
  - 마스크된 단어 복원 (MLM 예측)  
  - 번역 예측 (TLM, Translation Language Modeling 방식)  
  - 문맥 기반 예측 결과  

---

#### **2. 테스트 데이터 (Testing Data I/O)**
VECO는 다양한 다국어 평가 벤치마크를 사용해 테스트되었다.  
- **입력 (Input)**:  
  - XNLI: 다국어 자연어 추론 (예: "이 문장은 사실인가?" – 주어진 가설과 비교)  
  - PAWS-X: 다국어 문장 유사성 판별 (예: "이 두 문장은 동일한 의미인가?")  
  - XQuAD, MLQA: 질의응답 (예: "1980년에 올림픽이 열린 도시는?" → 문서 내에서 정답 찾기)  
  - WMT14 En-De, En-Fr: 기계 번역 (예: "The weather is nice today." → "Das Wetter ist heute schön.")  

- **출력 (Output)**:  
  - 자연어 추론 결과 (참/거짓/중립)  
  - 문장 유사성 점수  
  - 질문에 대한 정답 위치 및 정답 문구  
  - 번역된 문장 (BLEU 점수로 평가)  

---

#### **3. 본 연구와 기존 모델의 예측 차이**  

**예제 1: 다국어 자연어 추론 (XNLI)**
- **입력**:  
  - 문장 1: "그는 새 차를 샀다."  
  - 문장 2 (가설): "그는 돈이 많다."  
- **VECO 예측**: **중립 (Neutral)**  
- **XLM-R 예측**: 참 (Entailment) → 잘못된 예측  

**차이점**:  
VECO는 다국어 데이터에서 **문맥의 미묘한 차이를 더 잘 인식**하며, '새 차를 샀다'는 '돈이 많다'를 함축할 수도 있지만 반드시 그렇다고 할 수 없음을 학습했다.

---

**예제 2: 다국어 문장 유사성 (PAWS-X)**  
- **입력**:  
  - 문장 1: "그녀는 저녁을 먹고 영화를 봤다."  
  - 문장 2: "그녀는 영화를 본 후 저녁을 먹었다."  
- **VECO 예측**: **유사하지 않음 (Dissimilar)**  
- **XLM-R 예측**: 유사 (Similar) → 잘못된 예측  

**차이점**:  
VECO는 단순한 단어 수준 매칭이 아닌 **문장 구조와 순서의 의미 차이를 반영**하여 유사성을 더 정밀하게 판단한다.

---

**예제 3: 다국어 기계 번역 (WMT14 En-De)**
- **입력 문장 (영어)**:  
  - "The book on the table is mine."  
- **출력 (VECO)**:  
  - "Das Buch auf dem Tisch gehört mir." (정확한 번역)  
- **출력 (XLM-R)**:  
  - "Das Buch ist auf meinem Tisch." (오류: "on my table"로 잘못 번역)  

**차이점**:  
VECO는 문맥을 더 정확하게 반영하며, XLM-R은 문장을 직역하면서 원래 의미가 왜곡되었다.  

---



#### **1. Example Training Data (Training I/O)**
VECO was trained on **monolingual and bilingual corpora spanning 50 languages**.  
- **Input (Training Data)**:  
  - Monolingual text from CommonCrawl  
  - Parallel sentence pairs from the OPUS corpus  
  - Sentences with masked words (Masked Language Modeling - MLM)  

- **Output (Training Data)**:  
  - Predicted masked words (MLM)  
  - Translation predictions (TLM - Translation Language Modeling)  
  - Context-aware sentence predictions  

---

#### **2. Example Testing Data (Testing I/O)**  
VECO was evaluated on multiple multilingual benchmarks:  
- **Input (Testing Data)**:  
  - **XNLI** (Multilingual Natural Language Inference) – "Is this sentence true based on the given hypothesis?"  
  - **PAWS-X** (Sentence Similarity) – "Do these two sentences mean the same thing?"  
  - **XQuAD, MLQA** (Question Answering) – "Which city hosted the Olympics in 1980?" (Find the answer in text)  
  - **WMT14 En-De, En-Fr** (Machine Translation) – "The weather is nice today." → "Das Wetter ist heute schön."  

- **Output (Testing Data)**:  
  - NLI prediction (Entailment/Neutral/Contradiction)  
  - Sentence similarity score  
  - Answer span and extracted phrase for QA  
  - Translated sentence (evaluated via BLEU score)  

---

#### **3. Differences Between VECO and Previous Models**  

**Example 1: Multilingual Natural Language Inference (XNLI)**  
- **Input**:  
  - Sentence 1: "He bought a new car."  
  - Hypothesis: "He is wealthy."  
- **VECO Prediction**: **Neutral**  
- **XLM-R Prediction**: Entailment (Incorrect)  

**Difference**:  
VECO **better captures the nuances of context**—buying a car might imply wealth, but it does not necessarily entail it.  

---

**Example 2: Multilingual Sentence Similarity (PAWS-X)**  
- **Input**:  
  - Sentence 1: "She had dinner and then watched a movie."  
  - Sentence 2: "She watched a movie and then had dinner."  
- **VECO Prediction**: **Dissimilar**  
- **XLM-R Prediction**: Similar (Incorrect)  

**Difference**:  
VECO captures **semantic differences in sequence and event ordering**, while XLM-R relies too heavily on lexical overlap.  

---

**Example 3: Multilingual Machine Translation (WMT14 En-De)**  
- **Input (English)**:  
  - "The book on the table is mine."  
- **VECO Output (German)**:  
  - "Das Buch auf dem Tisch gehört mir." (Accurate translation)  
- **XLM-R Output (German)**:  
  - "Das Buch ist auf meinem Tisch." (Error: Incorrectly translates "on my table")  

**Difference**:  
VECO **preserves the correct meaning and syntactic structure**, while XLM-R produces a literal but incorrect translation.



<br/>  
# 요약   



VECO는 **크로스 어텐션 모듈**을 추가하여 다국어 문맥을 효과적으로 정렬하는 **Transformer 기반 사전 훈련 모델**이다. 50개 언어의 단일 언어 및 이중 언어 데이터를 활용해 훈련되었으며, **CA-MLM(Cross-Attention Masked Language Modeling)과 TLM(Translation Language Modeling)**을 적용하여 언어 간 전이를 강화했다. XTREME 벤치마크에서 **XLM-R보다 5~6포인트 높은 성능**을 기록했으며, 특히 문장 검색 태스크(BUCC, Tatoeba)에서 10포인트 이상 향상되었다. WMT14 기계 번역 평가에서는 기존 Transformer 및 XLM-R을 능가하며 BLEU 점수가 평균 1~2포인트 증가했다. 예제 테스트에서는 VECO가 기존 모델보다 문맥 이해와 번역에서 더 정확한 결과를 제공하며, 특히 자연어 추론(XNLI)과 문장 유사성(PAWS-X)에서 향상된 정밀도를 보였다.  

---


VECO is a **Transformer-based pre-trained model** that enhances multilingual alignment using a **cross-attention module**. It is trained on **monolingual and bilingual data across 50 languages**, incorporating **CA-MLM (Cross-Attention Masked Language Modeling) and TLM (Translation Language Modeling)** to improve cross-lingual transfer. VECO **outperforms XLM-R by 5-6 points** on the XTREME benchmark, with a **10+ point improvement** in sentence retrieval tasks (BUCC, Tatoeba). In WMT14 machine translation, it surpasses previous Transformer models and XLM-R, achieving **1-2 BLEU score gains on average**. Test examples show that VECO delivers **more accurate contextual understanding and translations**, particularly excelling in natural language inference (XNLI) and sentence similarity (PAWS-X).




<br/>  
# 기타  




#### **1. 모델 아키텍처 (Figure 1)**  
Figure 1은 VECO의 **모델 구조**를 보여준다. 기존 Transformer 인코더와 유사하지만, **교차 주의(Cross-Attention) 모듈**이 추가되어 다국어 문맥을 효과적으로 조정할 수 있다. 이 모듈을 통해 단일 언어 기반이 아닌, **다른 언어의 문맥 정보를 함께 고려**하여 예측을 수행할 수 있다. 또한, 훈련 단계에서는 크로스 어텐션을 활용하지만, 다운스트림 작업에서는 **필요에 따라 제거 가능**하여 유연성을 제공한다.  

---

#### **2. 훈련 목표 및 데이터 구성 (Table 1, Figure 2)**  
Table 1은 VECO의 **훈련 데이터 분포**를 정리한 표로, **단일 언어(CommonCrawl, 1.36TB, 6.5억 개 문장) 및 이중 언어(OPUS 병렬 코퍼스, 6.4억 개 문장)**를 사용했음을 보여준다.  
Figure 2는 **VECO의 사전 훈련 목표(Pretraining Objectives)**를 시각적으로 설명하며, **CA-MLM(Cross-Attention Masked Language Modeling)과 TLM(Translation Language Modeling)**이 언어 간 전이를 어떻게 강화하는지를 나타낸다.  

---

#### **3. XTREME 벤치마크 성능 비교 (Table 2, Figure 3)**  
Table 2는 **XTREME 벤치마크(40개 언어, 9개 태스크)**에서 VECO와 기존 모델(XLM-R, mBERT, XLM 등)의 성능을 비교한 표다. VECO는 **XLM-R보다 평균 5~6포인트 향상된 성능을 보이며**, 특히 **BUCC(문장 검색)에서 92.6% F1 스코어를 달성**해 기존 모델 대비 10포인트 이상 향상되었다.  
Figure 3은 이러한 성능 개선을 **시각적으로 표현한 그래프**로, VECO가 전반적인 XTREME 태스크에서 일관되게 높은 성능을 보이는 것을 확인할 수 있다.  

---

#### **4. 기계 번역 성능 비교 (Table 3, Figure 4)**  
Table 3은 **WMT14 영어-독일어(En-De) 및 영어-프랑스어(En-Fr) 번역 태스크**에서 VECO와 기존 모델(Transformer, XLM-R)의 BLEU 점수를 비교한 표다. VECO는 XLM-R 대비 **평균 0.8~1.0 BLEU 향상**, Transformer 대비 **2.3 BLEU 향상**을 기록했다.  
Figure 4는 BLEU 점수 변화를 그래프로 표현하며, VECO가 훈련 초기 단계에서도 빠르게 수렴하는 것을 보여준다.  

---

#### **5. 예제 테스트 결과 비교 (Table 4, Figure 5)**  
Table 4는 **XNLI(자연어 추론), PAWS-X(문장 유사성), XQuAD(질의응답) 태스크**에서 VECO와 기존 모델이 예측한 결과를 비교한 표다.  
Figure 5는 **번역 태스크에서 VECO와 XLM-R의 예측 차이**를 비교한 사례로, VECO가 보다 **정확한 문맥을 반영하여 번역을 수행**하는 것을 보여준다.  

---



#### **1. Model Architecture (Figure 1)**  
Figure 1 illustrates the **architecture of VECO**, which is similar to a traditional Transformer encoder but incorporates a **cross-attention module**. This module allows the model to **integrate contextual information from other languages** rather than relying solely on monolingual context. While cross-attention is used during pretraining, it can be **removed for downstream tasks**, providing flexibility in model adaptation.  

---

#### **2. Training Objectives and Data Composition (Table 1, Figure 2)**  
Table 1 presents a **summary of the training data** used for VECO, which includes **monolingual data (CommonCrawl, 1.36TB, 650M sentences) and bilingual data (OPUS parallel corpus, 640M sentences)**.  
Figure 2 visually explains **VECO’s pretraining objectives**, illustrating how **CA-MLM (Cross-Attention Masked Language Modeling) and TLM (Translation Language Modeling)** enhance cross-lingual transferability.  

---

#### **3. XTREME Benchmark Performance (Table 2, Figure 3)**  
Table 2 compares **VECO’s performance on the XTREME benchmark (40 languages, 9 tasks) against existing models (XLM-R, mBERT, XLM, etc.)**. VECO **outperforms XLM-R by 5-6 points on average** and achieves **a 92.6% F1 score on BUCC (sentence retrieval), a 10+ point improvement** over previous models.  
Figure 3 is a **visual representation of VECO’s improvements**, showing its consistently superior performance across all XTREME tasks.  

---

#### **4. Machine Translation Performance (Table 3, Figure 4)**  
Table 3 compares BLEU scores for **WMT14 English-to-German (En-De) and English-to-French (En-Fr) translation tasks** among VECO, Transformer, and XLM-R. VECO achieves a **0.8-1.0 BLEU improvement over XLM-R and a 2.3 BLEU increase over Transformer**.  
Figure 4 visualizes **BLEU score trends**, illustrating how VECO converges more rapidly even in early training stages.  

---

#### **5. Example Test Results Comparison (Table 4, Figure 5)**  
Table 4 compares **predictions from VECO and previous models on tasks such as XNLI (Natural Language Inference), PAWS-X (Sentence Similarity), and XQuAD (Question Answering)**.  
Figure 5 provides **examples of translation differences between VECO and XLM-R**, demonstrating VECO’s **better contextual understanding and translation accuracy**.  





<br/>
# refer format:     


@inproceedings{luo2021veco,
  author    = {Fuli Luo and Wei Wang and Jiahao Liu and Yijia Liu and Bin Bi and Songfang Huang and Fei Huang and Luo Si},
  title     = {{VECO}: Variable and Flexible Cross-lingual Pre-training for Language Understanding and Generation},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing},
  year      = {2021},
  pages     = {3980--3994},
  month     = {August},
  doi       = {10.18653/v1/2021.acl-long.308},
  url       = {https://aclanthology.org/2021.acl-long.308}
}
  


Luo, Fuli, Wei Wang, Jiahao Liu, Yijia Liu, Bin Bi, Songfang Huang, Fei Huang, and Luo Si. 2021. "VECO: Variable and Flexible Cross-lingual Pre-training for Language Understanding and Generation." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, August, 3980–3994. https://aclanthology.org/2021.acl-long.308.  




