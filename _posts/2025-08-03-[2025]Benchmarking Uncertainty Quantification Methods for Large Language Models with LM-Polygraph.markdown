---
layout: post
title:  "[2025]Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph"  
date:   2025-08-03 11:22:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 

그간 나온 다양한 불확실성척도 방법(LLM을 위한) 것들을 통합한 프레임워크  


PRR (Prediction Rejection Ratio)**: 낮은 불확실성을 가진 샘플의 평균 품질을 기반으로 UQ 성능을 평가.  
**ROC-AUC / PR-AUC**: 클레임 수준에서는 허위 주장 탐지를 위한 정확도 측정.  
**MSE (Mean Squared Error)**: 정규화된 신뢰도 점수와 실제 품질 간의 차이를 측정해 calibration 성능을 평가.  



짧은 요약(Abstract) :    




이 논문은 대형 언어 모델(LLM)의 **불확실성 정량화(Uncertainty Quantification, UQ)** 기법을 체계적으로 비교하고 평가할 수 있는 새로운 벤치마크를 제안합니다. 기존 연구들은 다양한 UQ 기법들을 개별적으로 제안했지만, 평가 방식이 제각각이라 비교가 어려웠습니다. 본 논문에서는 다양한 텍스트 생성 과제에 걸쳐 **최신 UQ 기법들을 구현한 통합된 평가 환경**을 제공하며, 특히 **신뢰도 점수의 해석 가능성**을 평가할 수 있는 정규화(normalization) 방법도 포함됩니다. 저자들은 이 벤치마크를 기반으로 **11가지 과제를 대상으로 한 대규모 실험**을 수행하였고, 그 결과 가장 효과적인 UQ 및 정규화 기법들을 도출합니다. 이를 통해 LLM의 신뢰성과 안전성을 향상시키는 데 기여하며, 후속 연구자들에게 진입 장벽을 낮추는 기반을 마련합니다.

---



The rapid proliferation of large language models (LLMs) has stimulated researchers to seek effective and efficient approaches to deal with LLM hallucinations and low-quality outputs. Uncertainty quantification (UQ) is a key element of machine learning applications in dealing with such challenges. However, research to date on UQ for LLMs has been fragmented in terms of techniques and evaluation methodologies. In this work, we address this issue by introducing a novel benchmark that implements a collection of state-of-the-art UQ baselines and offers an environment for controllable and consistent evaluation of novel UQ techniques over various text generation tasks. Our benchmark also supports the assessment of confidence normalization methods in terms of their ability to provide interpretable scores. Using our benchmark, we conduct a large-scale empirical investigation of UQ and normalization techniques across eleven tasks, identifying the most effective approaches.

---




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





이 논문은 대형 언어 모델(LLM)의 출력에 대한 불확실성을 정량화하기 위한 다양한 방법을 통합적으로 구현하고 비교 평가할 수 있는 벤치마크를 제안합니다. **LM-Polygraph**라는 프레임워크를 기반으로 하며, 여기에 다양한 UQ 기법들을 확장하여 포함하였습니다.
메서드는 크게 \*\*화이트박스(White-box)\*\*와 **블랙박스(Black-box)** 방식으로 나뉘며, 각 방식은 접근 가능한 정보 수준(예: 로짓 접근 가능 여부)에 따라 다릅니다.

1. **화이트박스 방법**:

   * 내부 로짓(logits), 확률 분포 등을 활용하여 예측 불확실성을 정량화합니다.
   * 대표적인 기법:

     * **MSP (Maximum Sequence Probability)**: 전체 시퀀스 확률을 기반으로 계산.
     * **Perplexity**, **Token Entropy**, **Pointwise Mutual Information (PMI)** 등.
     * **Claim-Conditioned Probability (CCP)**: 특정 문장(클레임)에 대한 의미 변화에 민감한 방식.
     * **SAR (Shifting Attention to Relevance)**: 의미 유사도 기반으로 시퀀스 다양성을 반영.

2. **샘플 다양성 기반 기법 (Sample Diversity)**:

   * LLM의 다양한 출력을 샘플링하고, 그들의 분산이나 의미 차이를 분석하여 불확실성을 평가.
   * 대표적으로 **Semantic Entropy**, **MC-Entropy**, **SAR** 등이 있음.

3. **밀도 기반 기법 (Density-based)**:

   * 훈련 데이터의 임베딩 분포를 근사하여 입력이 훈련 분포와 얼마나 다른지를 측정.
   * 예: Mahalanobis Distance, Hybrid UQ 등.

4. **리플렉시브 기법 (Reflexive Methods)**:

   * 모델 자신에게 "이 출력이 사실이냐?"를 다시 묻는 방식.
   * 예: **P(True)**, **Verbalized Confidence** 등.

5. **블랙박스 방법**:

   * 모델 출력만을 기반으로 판단. API 기반 LLM에 유용.
   * 의미 기반 클러스터링, 그래프 유사도 지표 (예: Degree Matrix, Eccentricity, Eigenvalue of Laplacian 등)를 사용.

6. **Claim-level UQ 확장**:

   * 문장 전체가 아니라 개별 클레임 단위에서 불확실성을 측정할 수 있도록 CCP 등을 확장 적용.

7. **정규화 및 신뢰도 점수 보정 (Normalization & Calibration)**:

   * 불확실성 점수를 해석 가능한 \[0,1] 범위로 보정하는 기법 개발.
   * 예: Linear, Quantile Scaling 외에 **Isotonic PCC** 같은 보정 기반 방법 사용.

이들은 모두 **Mistral 7B**, **StableLM 12B**, **Jais 13B**, **Vikhr**, **Yi** 등의 LLM에 적용되며, 영어, 중국어, 러시아어, 아랍어 등 다국어 실험 환경에서 평가됩니다.

---



This work introduces a benchmark based on the **LM-Polygraph** framework for systematically evaluating Uncertainty Quantification (UQ) methods in large language models (LLMs). The benchmark implements a wide range of recent UQ techniques and allows consistent, controlled comparisons across various text generation tasks.

Key components of the methodology include:

1. **White-box UQ methods**:

   * These use access to internal model states like logits or token probabilities.
   * Techniques include:

     * **Maximum Sequence Probability (MSP)**, **Perplexity**, **Token Entropy**, **PMI**, etc.
     * **Claim-Conditioned Probability (CCP)**: focuses on how meaning changes with token substitutions.
     * **SAR (Shifting Attention to Relevance)**: soft aggregates uncertainty using semantic similarity.

2. **Sample Diversity-Based Methods**:

   * These sample multiple responses and assess their diversity to estimate uncertainty.
   * Includes **Semantic Entropy**, **Monte Carlo Entropy**, and **SAR**.

3. **Density-Based Methods**:

   * These model the distribution of training embeddings and compute how likely a new input fits that distribution.
   * Examples: Mahalanobis Distance, Robust Density Estimation, and Hybrid UQ.

4. **Reflexive Methods**:

   * Ask the model to assess the truth or confidence of its own output (e.g., **P(True)**, **Verbalized Confidence**).

5. **Black-box Methods**:

   * Designed for API-only access. They operate solely on generated outputs.
   * Use semantic similarity graphs (e.g., **DegMat**, **Eccentricity**, **EigVal**) to compute uncertainty scores.

6. **Claim-level Extensions**:

   * Adapt UQ methods to operate on individual claims (fragments) rather than whole texts. CCP is used for this purpose.

7. **Confidence Score Normalization**:

   * Raw uncertainty scores are normalized to \[0,1] for interpretability using techniques like **Isotonic PCC** and **Binned PCC**.

The methods are applied across multiple LLMs (Mistral 7B, StableLM 12B, Jais 13B, Vikhr, Yi) and multiple languages (English, Chinese, Arabic, Russian).

---




   
 
<br/>
# Results  



---



이 논문은 제안된 LM-Polygraph 벤치마크를 활용하여 다양한 불확실성 정량화(UQ) 및 정규화 기법들을 **11개 태스크**, **여러 LLM (예: Mistral 7B, StableLM 12B, GPT-4o-mini, Jais 13B 등)**, **다국어 환경 (영어, 중국어, 러시아어, 아랍어)** 에서 평가했습니다.

####  주요 실험 구성:

* **태스크 종류**:

  1. **Selective QA (선택적 질의응답)**: CoQA, TriviaQA, MMLU, GSM8k
  2. **Selective Generation (선택적 생성)**: XSum 요약, WMT14/WMT19 기계번역
  3. **Claim-level Fact-checking**: 전기 생성 후 클레임 단위로 사실 검증 (GPT-4 기반 자동 평가)

* **평가 지표**:

  * **PRR (Prediction Rejection Ratio)**: 낮은 불확실성을 가진 샘플의 평균 품질을 기반으로 UQ 성능을 평가.
  * **ROC-AUC / PR-AUC**: 클레임 수준에서는 허위 주장 탐지를 위한 정확도 측정.
  * **MSE (Mean Squared Error)**: 정규화된 신뢰도 점수와 실제 품질 간의 차이를 측정해 calibration 성능을 평가.

####  주요 결과:

1. **정보 기반 방법 (MSP, CCP)**:

   * 짧은 출력(ex. MMLU, GSM8k)에서는 **MSP**, **CCP**가 매우 강력한 성능을 보임.
   * MSP는 단순한 기법임에도 불구하고 여러 태스크에서 복잡한 기법을 능가함.

2. **샘플 다양성 기반 기법 (SAR, Semantic Entropy, DegMat)**:

   * 긴 출력(ex. 요약, 번역)에서는 **SAR**, **Semantic Entropy**, **DegMat** 등의 기법이 우수.
   * 특히 SAR은 길이에 관계없이 안정적인 성능을 보임.

3. **Reflexive Methods (예: P(True))**:

   * 대부분의 소형 모델(Mistral 등)에서는 성능이 낮았음.
   * GPT-4o-mini와 같은 대형 모델에서는 Verbalized 방식이 효과적이었음.

4. **Claim-level Fact-checking 결과**:

   * **CCP**가 모든 언어(영어, 아랍어, 러시아어)에서 가장 높은 ROC-AUC와 PR-AUC 달성.
   * **중국어**에서는 CCP와 P(True)가 유사한 성능.

5. **신뢰도 정규화 평가 (Calibration)**:

   * **Isotonic PCC**와 **Binned PCC**가 가장 좋은 보정(calibration) 성능을 보였고, 원래의 PRR 성능도 유지함.

---


Using the proposed LM-Polygraph benchmark, the authors conducted a large-scale empirical evaluation of uncertainty quantification (UQ) and normalization techniques across **11 tasks**, multiple **LLMs (e.g., Mistral 7B, StableLM 12B, GPT-4o-mini, Jais 13B)**, and **four languages** (English, Chinese, Arabic, Russian).

####  Evaluation Setup:

* **Task Types**:

  1. **Selective QA**: CoQA, TriviaQA, MMLU, GSM8k
  2. **Selective Generation**: XSum summarization, WMT14/WMT19 translation
  3. **Claim-Level Fact-Checking**: Fact-checking atomic claims extracted from LLM-generated biographies (using GPT-4)

* **Evaluation Metrics**:

  * **Prediction Rejection Ratio (PRR)** for ranking quality by uncertainty.
  * **ROC-AUC / PR-AUC** for hallucination detection in claim-level fact-checking.
  * **MSE (Mean Squared Error)** to measure calibration performance of normalized confidence scores.

####  Key Findings:

1. **Information-based Methods (MSP, CCP)**:

   * For short outputs (e.g., MMLU, GSM8k), **MSP** and **CCP** outperformed more complex approaches.
   * MSP remains a surprisingly strong baseline.

2. **Sample Diversity-Based Methods (SAR, Semantic Entropy, DegMat)**:

   * For long outputs (e.g., summarization and MT), methods like **SAR**, **Semantic Entropy**, and **DegMat** performed best.
   * **SAR** was robust across both short and long generation tasks.

3. **Reflexive Methods (e.g., P(True))**:

   * Underperformed on small models (e.g., Mistral).
   * On larger models like **GPT-4o-mini**, verbalized UQ methods showed competitive performance.

4. **Claim-Level Fact-Checking**:

   * **CCP** consistently achieved the best ROC-AUC and PR-AUC across all languages except Chinese.
   * In Chinese, **CCP** and **P(True)** were comparable.

5. **Uncertainty Normalization**:

   * **Isotonic PCC** and **Binned PCC** produced the best-calibrated confidence scores without degrading PRR.

---





<br/>
# 예제  





논문은 다양한 언어 및 과제 유형에 걸쳐 **텍스트 생성 기반 태스크들**에서 불확실성 정량화(UQ) 기법을 테스트합니다. 각 태스크는 \*\*입력 텍스트 (prompt)\*\*에 대해 **LLM이 생성한 출력 텍스트**를 평가하며, 해당 출력이 얼마나 신뢰할 수 있는지를 **불확실성 점수로 측정**합니다.

#### 1. **선택적 질의응답 (Selective QA)**:

* **입력**: 질문 또는 문맥+질문 (예: "Who is the president of France?" 또는 대화 문맥 포함)
* **출력**: LLM이 생성한 자연어 답변 (예: "Emmanuel Macron")
* **목표**: 모델이 정답을 모를 경우 높은 불확실성을 보여야 하며, 확신이 있는 답변일 경우 낮은 불확실성을 보여야 함
* **사용된 데이터셋**:

  * *CoQA*: 대화 기반 자유형식 질의응답
  * *TriviaQA*: 구성적, 복잡한 질문들 (문맥 없음)
  * *MMLU*: 객관식 문제
  * *GSM8k*: 수학 서술형 문제

#### 2. **선택적 생성 (Selective Generation)**:

* **입력**: 원문 텍스트 (예: 프랑스어 문장 또는 뉴스 기사)
* **출력**: 번역문 또는 요약문 (영어 문장 등)
* **목표**: 낮은 품질의 생성물에 대해 높은 불확실성 점수를 부여하는지 평가
* **사용된 데이터셋**:

  * *WMT14 (Fr→En), WMT19 (De→En)*: 기계 번역
  * *XSum*: 뉴스 기사 요약

#### 3. **클레임 수준 사실 검증 (Claim-Level Fact-checking)**:

* **입력**: 예: “Tell me a biography of \[Famous Person]”
* **출력**: 생성된 전기 텍스트 (예: “John Doe was born in 1950 and won the Nobel Prize in 1990.”)
* **처리**:

  * GPT-4로 문장을 원자적 클레임 단위로 분할함
  * 각 클레임이 사실인지 아닌지 판단 (supported / unsupported)
* **목표**: LLM이 **허위 정보를 생성할 때** 높은 불확실성 점수를 부여하는지 측정
* **데이터 생성 방식**: GPT-4를 사용해 1900년 이후 유명 인물 100명에 대해 전기를 생성하고 자동 평가

---


The paper tests UQ methods on a variety of **text generation tasks** where the input is a textual prompt and the output is the LLM-generated response. The goal is to measure whether uncertainty scores correctly reflect the **quality and reliability** of the generated output.

#### 1. **Selective Question Answering (QA)**:

* **Input**: A question or a conversational context + question
  (e.g., “Who is the president of France?” or a dialog history)
* **Output**: Natural language answer generated by the LLM
  (e.g., “Emmanuel Macron”)
* **Objective**: The model should show **low uncertainty** for correct confident answers and **high uncertainty** when unsure or wrong
* **Datasets Used**:

  * *CoQA*: Conversational QA with free-form answers
  * *TriviaQA*: Complex factual questions without context
  * *MMLU*: Multiple-choice questions
  * *GSM8k*: Grade-school math word problems

#### 2. **Selective Generation**:

* **Input**: Source text (e.g., a French sentence or a news article)
* **Output**: Target text, such as a translation or a summary
  (e.g., English sentence or headline)
* **Objective**: Evaluate whether low-quality generations are assigned high uncertainty
* **Datasets Used**:

  * *WMT14 (Fr→En), WMT19 (De→En)*: Machine Translation
  * *XSum*: Abstractive summarization

#### 3. **Claim-Level Fact-Checking**:

* **Input**: Prompt like “Tell me a biography of \[Person]”
* **Output**: Generated biography (e.g., “John Doe was born in 1950 and won a Nobel Prize in 1990.”)
* **Processing**:

  * GPT-4 decomposes the text into **atomic claims**
  * GPT-4 then classifies each claim as **supported or unsupported**
* **Objective**: The UQ method should assign **higher uncertainty** to unsupported or hallucinated claims
* **Data Creation**: Biographies were generated for 100 famous people (post-1900) using various LLMs and assessed via GPT-4

---




<br/>  
# 요약   




이 논문은 LLM의 불확실성 정량화(UQ) 기법들을 일관되게 비교할 수 있는 **LM-Polygraph 벤치마크**를 제안하며, 정보 기반, 샘플 다양성 기반, 밀도 기반 등 다양한 기법을 구현한다.
11개의 QA, 번역, 요약, 사실 검증 태스크에서 실험한 결과, **SAR, CCP, MSP**와 같은 기법들이 출력 길이에 따라 강점을 보이며, 특히 CCP는 다국어 사실 검증에서 최고 성능을 기록했다.
예를 들어, "Who is the president of France?"와 같은 질문에 대해 생성된 답변의 **정확성과 불확실성 점수 간 상관성**을 평가하거나, 생성된 전기에서 개별 문장을 클레임으로 나눠 사실 여부를 검증하는 방식으로 테스트가 수행된다.

---



This paper proposes **LM-Polygraph**, a unified benchmark for evaluating uncertainty quantification (UQ) methods in LLMs, implementing techniques such as information-based, sample diversity-based, and density-based approaches.
Across 11 tasks—including QA, summarization, translation, and fact-checking—methods like **SAR, CCP, and MSP** demonstrated strength depending on output length, with CCP achieving top performance in multilingual fact-checking.
For example, the system evaluates whether the uncertainty score aligns with answer correctness in a QA like "Who is the president of France?", or checks factuality of atomic claims extracted from LLM-generated biographies.



<br/>  
# 기타  



####  다이어그램 및 피규어

* **Figure 1–3 (평균 PRR 비교)**

  * 선택적 QA/생성에서 \*\*PRR(예측 거절 비율)\*\*을 시각화함.
  * 정보 기반 기법(MSP, CCP)은 짧은 응답에서 강력했고, 샘플 다양성 기반 기법(SAR, DegMat 등)은 긴 응답에서 우수.
  * 특히 **SAR**은 짧고 긴 응답 모두에서 안정적인 성능을 보이며 가장 범용적임을 시사.

* **Figure 4 (Calibration MSE 비교)**

  * 정규화 기법별로 예측된 confidence score와 실제 품질 점수 간 평균제곱오차(MSE)를 비교.
  * **Isotonic PCC**와 **Binned PCC**가 가장 낮은 MSE를 보이며, 사용자에게 신뢰할 수 있는 confidence 출력을 제공함을 나타냄.

####  테이블

* **Table 2 (데이터셋 통계)**

  * CoQA, TriviaQA, MMLU, GSM8k, WMT14/19, XSum 등에서 입력 길이와 평균 출력 길이 등을 정리.
  * 다양한 생성 길이와 도메인을 다룸으로써 UQ 기법의 일반화 가능성을 높임.

* **Table 4 (출력 길이에 따른 PRR 변화)**

  * 출력이 짧을수록 \*\*정보 기반 기법(MSP, CCP)\*\*의 성능이 높고, 출력이 길어질수록 \*\*샘플 다양성 기반 기법(SAR 등)\*\*이 유리함.
  * 이는 각 기법이 포착하는 불확실성의 형태가 다름을 시사.

* **Table 5 (사실 검증 성능)**

  * 다국어 클레임 기반 평가에서 **CCP**가 거의 모든 언어에서 최고 성능(ROC-AUC 및 PR-AUC)을 달성.
  * 단, **중국어**에서는 P(True)도 유사한 수준으로, 특정 언어/모델간 fine-tuning 차이가 성능에 영향을 미침.

####  어펜딕스 (Appendix C, D, E)

* **Appendix C**: 각 모델과 태스크별 세부 성능 수치 제공. PRR 기준으로 **MSP, CCP, SAR** 등이 강세.
* **Appendix D**: 정규화 이후 성능 변화 분석. **Isotonic PCC**는 calibration 성능 향상 + 기존 성능 유지.
* **Appendix E**: 각 모델의 기본 생성 품질 점수 정리. 베이스라인 비교를 위한 참조용.

---

####  Diagrams & Figures

* **Figure 1–3 (Mean PRR Comparison)**

  * Visualizes PRR (Prediction Rejection Ratio) across selective QA/generation tasks.
  * Information-based methods like **MSP** and **CCP** perform best on short outputs, while **SAR** and **DegMat** excel on long ones.
  * **SAR** stands out for its consistent performance across output lengths, suggesting high generalizability.

* **Figure 4 (Confidence Calibration MSE)**

  * Shows Mean Squared Error between normalized confidence scores and actual quality scores.
  * **Isotonic PCC** and **Binned PCC** yield the lowest errors, making them most interpretable for end users.

####  Tables

* **Table 2 (Dataset Statistics)**

  * Summarizes input/output lengths and domains across datasets like CoQA, MMLU, XSum, WMT.
  * Ensures diversity in generation conditions, increasing robustness of UQ evaluation.

* **Table 4 (PRR by Output Length)**

  * Short outputs favor **information-based methods** like MSP/CCP; long outputs benefit from **diversity-based methods** like SAR.
  * Highlights that different UQ techniques capture different aspects of uncertainty.

* **Table 5 (Fact-Checking Performance)**

  * **CCP** achieves highest ROC-AUC and PR-AUC across all languages except Chinese.
  * For Chinese, **P(True)** performs similarly, possibly due to fine-tuning differences in the Yi model.

#### Appendix Insights

* **Appendix C**: Provides granular PRR scores across models/tasks, confirming SAR, CCP, MSP as top performers.
* **Appendix D**: Evaluates normalization impact; **Isotonic PCC** improves calibration without harming PRR.
* **Appendix E**: Includes base text generation quality metrics for each model, useful for baseline comparisons.

---




<br/>
# refer format:     



@article{vashurin2025lmpolygraph,
  title={Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph},
  author={Vashurin, Roman and Fadeeva, Ekaterina and Vazhentsev, Artem and Rvanova, Lyudmila and Vasilev, Daniil and Tsvigun, Akim and Petrakov, Sergey and Xing, Rui and Sadallah, Abdelrahman and Grishchenkov, Kirill and Panchenko, Alexander and Baldwin, Timothy and Nakov, Preslav and Panov, Maxim and Shelmanov, Artem},
  journal={Transactions of the Association for Computational Linguistics},
  volume={13},
  pages={220--248},
  year={2025},
  doi={10.1162/tacl_a_00737},
  publisher={MIT Press}
}



Vashurin, Roman, Ekaterina Fadeeva, Artem Vazhentsev, Lyudmila Rvanova, Daniil Vasilev, Akim Tsvigun, Sergey Petrakov, Rui Xing, Abdelrahman Sadallah, Kirill Grishchenkov, Alexander Panchenko, Timothy Baldwin, Preslav Nakov, Maxim Panov, and Artem Shelmanov. “Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph.” Transactions of the Association for Computational Linguistics 13 (2025): 220–248. https://doi.org/10.1162/tacl_a_00737.




