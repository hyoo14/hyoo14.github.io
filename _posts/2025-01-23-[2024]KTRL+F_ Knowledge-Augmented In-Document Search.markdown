---
layout: post
title:  "[2024]KTRL+F: Knowledge-Augmented In-Document Search"  
date:   2025-01-23 22:52:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


쿼리에서 핵심 키워드(엔티티) 식별 → Wikipedia에서 대상 검색 → 검색 결과에 외부 지식 정보(검색 결과를 다시 위키에 검색해서 얻은)를 추가    
(평가 셋도 만들어서 제공)  



짧은 요약(Abstract) :    



KTRL+F는 실시간으로 문서 내의 모든 의미론적 대상을 찾아내는 새로운 과제입니다. 이 과제는 외부 지식을 활용하여 대상에 대한 추가 정보를 확장하고, 속도와 성능의 균형을 맞추는 데 초점을 둡니다. 기존 모델의 한계(예: 환각, 높은 지연 시간, 외부 지식 활용의 어려움)를 분석한 후, 외부 지식을 구문 임베딩에 단순히 추가하여 속도와 성능을 모두 향상시키는 지식 증강 구문 검색 모델을 제안합니다. 사용자 연구 결과, 이 모델은 검색 쿼리 수를 줄이고 다른 소스를 탐색하는 시간을 감소시키며 사용자 경험을 크게 향상시키는 것으로 나타났습니다. 연구자 커뮤니티가 KTRL+F 문제를 해결하여 문서 내 정보 접근의 효율성을 높이기를 권장합니다.

---


We introduce a new problem, KTRL+F, a knowledge-augmented in-document search that necessitates real-time identification of all semantic targets within a document with awareness of external sources through a single natural query. KTRL+F addresses the unique challenges of in-document search: 1) utilizing knowledge outside the document for extended use of additional information about targets, and 2) balancing real-time applicability with performance. Analyzing various baselines, we identify limitations such as hallucinations, high latency, and difficulties in leveraging external knowledge. We propose a Knowledge-Augmented Phrase Retrieval model, which balances speed and performance by simply augmenting external knowledge in phrase embedding. A user study demonstrates that even with this simple model, users reduce search time and queries while making fewer extra visits to other sources. We encourage the research community to tackle KTRL+F to enhance efficient in-document information access.



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





이 논문에서는 **KTRL+F(Knowledge-Augmented In-Document Search)**를 구현하기 위해 새로운 데이터셋과 모델 아키텍처를 제안합니다. 

1. **데이터셋 생성**:  
   - **실제 뉴스 기사**에서 문서를 샘플링하여 입력 데이터로 사용했습니다. 문서에 포함된 엔티티를 식별하고 외부 지식(예: Wikipedia)과 연결하였습니다.
   - **질문-대상 쌍 생성**: LLAMA-2 모델을 활용해 다양한 질문과 대상을 자동 생성하고, 이를 바탕으로 외부 지식과의 연결성을 검증했습니다. 이후, DeBERTa 모델을 사용해 외부 지식 없이도 답변할 수 있는 질문을 필터링하여 최종적으로 512개의 질문-대상 쌍을 구축했습니다.

2. **모델 아키텍처**:  
   - **지식 증강 구문 검색 모델(Knowledge-Augmented Phrase Retrieval)**: DensePhrases 모델을 기반으로 설계된 이 모델은 문서 내 구문 임베딩에 외부 지식 임베딩을 추가하여 성능을 향상시켰습니다.  
   - **외부 지식 연계 모듈**: 엔티티를 식별하고 관련된 Wikipedia 페이지를 연결하여 해당 지식을 임베딩에 포함합니다.  
   - **질문 및 구문 인코더**: DensePhrases의 사전 학습된 구문 인코더를 사용하여 각 구문과 질문을 임베딩으로 변환하였습니다.  
   - **지식 통합 모듈**: 구문 임베딩과 외부 지식 임베딩을 단순히 요소별로 더하는 방식으로 통합하였으며, 이를 통해 성능과 속도 모두에서 우수한 결과를 도출했습니다.

3. **모델 평가**:  
   - 제안된 모델은 검색 속도와 성능(정확도)을 균형 있게 유지하며, 특히 **List Overlap F1** 점수에서 우수한 결과를 나타냈습니다.  
   - 또한, 사용자가 외부 지식을 통해 효율적으로 정보를 검색할 수 있도록 Chrome 확장 플러그인을 개발하여 실제 사용자 테스트를 진행했습니다.

---



This paper introduces **KTRL+F (Knowledge-Augmented In-Document Search)** with a novel dataset and model architecture.

1. **Dataset Construction**:  
   - **Document Sampling**: Real-world news articles were selected as input documents, with entities identified and linked to external knowledge (e.g., Wikipedia).  
   - **Query-Target Pair Generation**: Diverse queries and targets were automatically generated using the LLAMA-2 model, and the connection to external knowledge was validated. DeBERTa was used to filter queries that could be answered without external knowledge, resulting in a final set of 512 query-target pairs.

2. **Model Architecture**:  
   - **Knowledge-Augmented Phrase Retrieval Model**: Based on the DensePhrases architecture, the model incorporates external knowledge embeddings into in-document phrase embeddings to improve performance.  
   - **External Knowledge Linking Module**: This module identifies entities and links them to relevant Wikipedia pages, embedding this external knowledge into the model.  
   - **Query and Phrase Encoders**: Pre-trained DensePhrases encoders were used to transform document phrases and queries into embeddings.  
   - **Knowledge Aggregation Module**: Phrase embeddings and external knowledge embeddings were combined using element-wise addition, achieving both high performance and low latency.

3. **Model Evaluation**:  
   - The proposed model strikes a balance between search speed and performance, excelling in metrics like **List Overlap F1**.  
   - A Chrome extension plugin was also developed to test the model’s real-world applicability, demonstrating improved efficiency in user search experiences.


   
 
<br/>
# Results  




**테스트 데이터 및 평가 메트릭**  
KTRL+F의 성능을 평가하기 위해 총 512개의 질문-대상 쌍으로 구성된 KTRL+F 데이터셋을 사용했습니다. 성능은 다음과 같은 주요 메트릭으로 평가되었습니다:
1. **List EM (Exact Match)**: 모델이 대상 리스트를 정확히 예측했는지 평가.
2. **List Overlap F1**: 대상 리스트의 일부라도 정확히 예측했는지 평가.
3. **Latency (ms/Q)**: 쿼리당 처리 속도를 밀리초 단위로 측정하여 실시간 성능 평가.

**비교 모델**  
KTRL+F 모델은 다양한 베이스라인 모델과 비교되었습니다:
- **생성 기반 모델**: GPT-3.5, GPT-4, LLAMA-2, VICUNA.
- **추출 기반 모델**: MultiSpanQA를 활용한 BERT 기반 SequenceTagger.
- **검색 기반 모델**: DensePhrases 기반의 Retrieval-Augmented 모델.

**결과 요약**  
1. **성능 우위**  
   - KTRL+F 모델은 **List Overlap F1** 기준으로 생성 기반 모델보다 더 높은 점수를 기록했으며, 특히 "Ours (w/ Gold)" 설정에서는 가장 높은 점수(53.69%)를 달성했습니다.  
   - **List EM**에서도 "Ours (w/ Gold)" 모델이 46.17%로 우위를 보였습니다.  

2. **속도 우위**  
   - KTRL+F 모델("Ours (w/ Wikifier)")은 평균 **15ms/Q**로 다른 생성 기반 모델(GPT-4: 2,420ms/Q, LLAMA-2: 3,176ms/Q) 대비 압도적으로 빠른 검색 속도를 보였습니다.

3. **비교 분석**  
   - 생성 기반 모델은 고성능을 보였으나, 처리 속도가 느리고 환각(hallucination) 문제가 있었습니다.  
   - 추출 기반 모델은 외부 지식을 활용하지 못해 KTRL+F와 같은 복합적인 검색 과제를 해결하는 데 부적합했습니다.  
   - KTRL+F는 성능과 속도 모두에서 균형 잡힌 결과를 보여주었습니다.

---



**Test Data and Metrics**  
The KTRL+F dataset, comprising 512 query-target pairs, was used to evaluate the model. Performance was assessed using the following key metrics:
1. **List EM (Exact Match)**: Evaluates whether the model predicts the exact list of targets.
2. **List Overlap F1**: Measures partial matches between predicted and actual target lists.
3. **Latency (ms/Q)**: Assesses real-time performance by measuring processing time per query.

**Comparison Models**  
The KTRL+F model was compared against various baselines:
- **Generative Models**: GPT-3.5, GPT-4, LLAMA-2, VICUNA.
- **Extractive Model**: A BERT-based SequenceTagger fine-tuned on MultiSpanQA.
- **Retrieval-Based Models**: Retrieval-Augmented DensePhrases models.

**Summary of Results**  
1. **Performance Superiority**  
   - The KTRL+F model outperformed generative models in **List Overlap F1**, achieving the highest score (53.69%) under the "Ours (w/ Gold)" setting.  
   - It also excelled in **List EM**, with "Ours (w/ Gold)" scoring 46.17%.

2. **Speed Superiority**  
   - The KTRL+F model ("Ours (w/ Wikifier)") demonstrated significantly faster search speeds, averaging **15ms/Q**, compared to generative models (e.g., GPT-4: 2,420ms/Q, LLAMA-2: 3,176ms/Q).

3. **Comparative Analysis**  
   - Generative models exhibited high performance but suffered from slow speeds and issues like hallucinations.  
   - The extractive model failed to leverage external knowledge, making it unsuitable for complex tasks like KTRL+F.  
   - The KTRL+F model balanced both performance and speed effectively, making it a robust solution for the task.



<br/>
# 예제  




**테스트 데이터 예시**  
다음은 KTRL+F 데이터셋의 질문과 입력 데이터 예시입니다.

- **입력 문서**:  
  "중국의 주요 소셜 네트워크 플랫폼으로는 WeChat, Baidu, Weibo가 있다. WeChat은 Tencent가 개발한 소셜 미디어 및 모바일 결제 애플리케이션이다. Baidu는 검색 엔진과 소셜 네트워크 서비스를 제공한다. Weibo는 ‘중국의 트위터’라고 불린다."

- **질문**:  
  "중국에서 소셜 네트워크 플랫폼으로 사용되는 모든 플랫폼을 나열하시오."

- **정답 대상 리스트 (Ground Truth)**:  
  ["WeChat", "Baidu", "Weibo"]

---

**모델별 결과 비교**  

1. **제안된 KTRL+F 모델 ("Ours w/ Wikifier")**  
   - 예측 결과: ["WeChat", "Baidu", "Weibo"]  
   - 정확도 (List EM): 100%  
   - 설명: 제안된 모델은 문서에서 모든 정답을 정확히 식별했으며, WeChat의 변형(Weixin)과 같은 외부 지식을 활용하여 추가적인 의미적 연결성을 제공했습니다.

2. **생성 기반 모델 (예: GPT-4)**  
   - 예측 결과: ["WeChat", "Weibo"]  
   - 정확도 (List EM): 66%  
   - 문제점: Baidu를 누락했으며, 외부 지식을 완전히 활용하지 못한 결과를 보였습니다. 또한, WeChat과 Weixin을 별도로 인식하지 못했습니다.

3. **추출 기반 모델 (SequenceTagger)**  
   - 예측 결과: ["Baidu"]  
   - 정확도 (List EM): 33%  
   - 문제점: 문서 내에서 단일 키워드 기반의 추출만 수행했으며, 의미적 연결성을 이해하지 못해 WeChat과 Weibo를 놓쳤습니다.

---

**결론**  
KTRL+F 모델은 외부 지식을 활용하여 문서 내 모든 정답을 정확히 찾아냄으로써, 생성 모델과 추출 모델의 한계를 극복했습니다. 특히, 외부 지식을 요구하는 복잡한 질문에서 우위를 보여줍니다.

---


**Test Data Example**  
Below is a sample from the KTRL+F dataset:

- **Input Document**:  
  "Major social network platforms in China include WeChat, Baidu, and Weibo. WeChat is a social media and mobile payment app developed by Tencent. Baidu provides search engine and social networking services. Weibo is known as the 'Twitter of China'."

- **Question**:  
  "List all social network platforms used in China."

- **Ground Truth Target List**:  
  ["WeChat", "Baidu", "Weibo"]

---

**Model Results Comparison**

1. **Proposed KTRL+F Model ("Ours w/ Wikifier")**  
   - Predicted Output: ["WeChat", "Baidu", "Weibo"]  
   - Accuracy (List EM): 100%  
   - Explanation: The proposed model accurately identified all targets in the document. It also leveraged external knowledge, recognizing semantic variations such as "WeChat" and "Weixin."

2. **Generative Model (e.g., GPT-4)**  
   - Predicted Output: ["WeChat", "Weibo"]  
   - Accuracy (List EM): 66%  
   - Issue: It failed to identify "Baidu" and did not fully utilize external knowledge. Variants like "WeChat" and "Weixin" were not connected properly.

3. **Extractive Model (SequenceTagger)**  
   - Predicted Output: ["Baidu"]  
   - Accuracy (List EM): 33%  
   - Issue: This model relied on keyword matching and failed to understand semantic connections, missing "WeChat" and "Weibo."

---

**Conclusion**  
The KTRL+F model excelled by leveraging external knowledge to identify all correct targets, outperforming generative and extractive models. It demonstrated superior performance, especially for complex queries requiring external knowledge.


<br/>  
# 요약   



KTRL+F 모델은 외부 지식을 활용하여 문서 내 의미적 대상을 실시간으로 식별하는 새로운 접근 방식을 제안합니다. DensePhrases를 기반으로 설계된 이 모델은 외부 지식 임베딩과 구문 임베딩을 통합하여 성능과 속도의 균형을 맞췄습니다. 테스트 데이터 예시에서 "중국에서 사용되는 모든 소셜 네트워크 플랫폼"을 묻는 질문에 대해 KTRL+F는 WeChat, Baidu, Weibo를 정확히 찾아냈습니다. 반면, GPT-4는 Baidu를 누락했으며, SequenceTagger는 WeChat과 Weibo를 식별하지 못했습니다. 이처럼 KTRL+F는 외부 지식과 문서 간의 의미적 연결성을 효과적으로 활용해 기존 모델의 한계를 극복하고 높은 정확도를 보여줍니다.

---


The KTRL+F model introduces a novel approach for real-time identification of semantic targets within documents by leveraging external knowledge. Built on DensePhrases, the model integrates external knowledge embeddings with phrase embeddings, achieving a balance between performance and speed. In a test example asking for "all social network platforms used in China," KTRL+F accurately identified WeChat, Baidu, and Weibo. In contrast, GPT-4 missed Baidu, and SequenceTagger failed to recognize WeChat and Weibo. This demonstrates KTRL+F's effectiveness in overcoming the limitations of existing models by effectively utilizing semantic connections between external knowledge and the document, achieving high accuracy.


<br/>  
# 기타  




1. **Figure 1**:  
   KTRL+F와 기존 문서 검색 시스템(Ctrl+F, 정규 표현식, MRC)의 비교를 시각적으로 보여줍니다. Ctrl+F와 정규 표현식은 단순한 텍스트 매칭을 기반으로 하며, 외부 지식을 고려하지 않습니다. 반면, KTRL+F는 외부 지식을 활용하여 문서 내 모든 의미적 대상을 효율적으로 식별하는 과제를 강조합니다.

2. **Figure 2**:  
   KTRL+F 데이터셋 생성 파이프라인을 설명합니다. 뉴스 기사에서 문서를 선택하고(Llama-2로) 질문-대상 쌍을 생성하며, 최종적으로 외부 지식과 함께 질의와 대상을 검증 및 필터링하는 단계를 포함합니다.

3. **Figure 3**:  
   제안된 Knowledge-Augmented Phrase Retrieval 모델의 구조를 설명합니다. 외부 지식 연결 모듈, 구문 인코더, 질문 인코더, 지식 통합 모듈 등 주요 구성 요소를 시각적으로 보여줍니다.

4. **Figure 4**:  
   사용자 연구 결과를 요약한 그래프입니다. KTRL+F 플러그인은 Ctrl+F와 정규 표현식보다 짧은 검색 시간, 적은 쿼리 수, 더 높은 List EM F1 점수를 보여줍니다.

---



1. **Figure 1**:  
   Compares KTRL+F with traditional in-document search systems (Ctrl+F, regex, and MRC). Ctrl+F and regex rely on simple text matching without utilizing external knowledge. KTRL+F addresses this limitation by leveraging external knowledge for efficient real-time semantic target identification within documents.

2. **Figure 2**:  
   Explains the KTRL+F dataset construction pipeline. It includes selecting documents from news articles, generating query-target pairs using Llama-2, and verifying/filtering these pairs with external knowledge.

3. **Figure 3**:  
   Visualizes the architecture of the proposed Knowledge-Augmented Phrase Retrieval model, showing components like the external knowledge linking module, phrase encoder, query encoder, and knowledge aggregation module.

4. **Figure 4**:  
   Summarizes user study results. The KTRL+F plugin outperforms Ctrl+F and regex in terms of shorter search times, fewer queries, and higher List EM F1 scores.


<br/>
# 페이퍼 구조 및 구성     



1. **서론 (Introduction)**  
   논문의 첫 부분에서는 KTRL+F 문제의 중요성과 필요성을 강조합니다. 기존 문서 검색 시스템의 한계를 지적하며, 특히 외부 지식을 활용하지 못한다는 점을 부각합니다. 또한, KTRL+F가 해결하고자 하는 과제와 목표를 명확히 설정하며, 이 문제를 해결하기 위해 제안된 접근 방식의 핵심 아이디어를 간략히 소개합니다.

2. **관련 연구 (Related Work)**  
   기존의 문서 검색, 정보 검색, 및 MRC(기계 독해) 분야에서 사용된 기술들을 정리하고, KTRL+F가 이들과 어떻게 다른지 설명합니다. 특히, 기존 모델들이 외부 지식을 통합하는 데 겪는 어려움을 중심으로 논의가 이루어집니다.

3. **KTRL+F 문제 정의 (Problem Definition)**  
   KTRL+F의 구체적인 문제 정의와 평가 기준이 이 섹션에 포함됩니다. 여기서는 문서 내 의미적 대상 탐색과 외부 지식 활용이라는 두 가지 주요 도전 과제가 명확히 정의됩니다.

4. **데이터셋 및 방법론 (Dataset and Methodology)**  
   KTRL+F 데이터셋 생성 과정과 제안된 Knowledge-Augmented Phrase Retrieval 모델의 구조 및 작동 원리를 상세히 설명합니다. 데이터셋 생성은 뉴스 기사 샘플링, 질문-대상 쌍 생성, 외부 지식 연결을 포함하며, 모델 아키텍처는 구문 인코더와 외부 지식 통합 모듈로 구성됩니다.

5. **실험 및 결과 (Experiments and Results)**  
   다양한 베이스라인 모델(GPT-4, LLAMA-2, SequenceTagger)과 KTRL+F의 성능을 비교한 결과가 포함됩니다. 성능 평가는 List EM, List Overlap F1, 검색 속도 등의 메트릭을 사용하여 이루어졌으며, KTRL+F가 성능과 속도 모두에서 우수한 결과를 보여줍니다.

6. **사용자 연구 (User Study)**  
   사용자 연구를 통해 KTRL+F의 실질적인 효용성을 평가합니다. 사용자가 KTRL+F를 사용하여 더 적은 시간과 쿼리로 문서 내 대상을 검색했음을 강조하며, Ctrl+F와 정규 표현식 대비 이점이 상세히 논의됩니다.

7. **결론 및 향후 연구 (Conclusion and Future Work)**  
   논문의 마지막 부분에서는 연구 결과를 요약하고, KTRL+F가 가져올 실질적인 기여를 강조합니다. 동시에, 향후 연구 방향으로 KTRL+F의 확장 가능성과 다른 도메인에의 적용 가능성을 제안합니다.



<br/>
# refer format:     



@inproceedings{oh2024ktrl,
  title={KTRL+F: Knowledge-Augmented In-Document Search},
  author={Oh, Hanseok and Shin, Haebin and Ko, Miyoung and Lee, Hyunji and Seo, Minjoon},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={2416--2436},
  year={2024},
  organization={Association for Computational Linguistics},
  address={June 16-21, 2024},
  publisher={ACL},
  url={https://github.com/kaistAI/KtrlF}
}




Oh, Hanseok, Haebin Shin, Miyoung Ko, Hyunji Lee, and Minjoon Seo. "KTRL+F: Knowledge-Augmented In-Document Search." In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2416–2436. Association for Computational Linguistics, 2024. Accessed via GitHub: https://github.com/kaistAI/KtrlF.








