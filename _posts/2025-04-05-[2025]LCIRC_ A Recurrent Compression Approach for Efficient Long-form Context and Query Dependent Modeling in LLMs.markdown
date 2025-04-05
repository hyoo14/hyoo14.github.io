---
layout: post
title:  "[2025]LCIRC A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs"  
date:   2025-04-05 02:32:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

LLM 인풋에 요약을 넣어줘서 더 긴 텍스트도 처리하게끔함   




짧은 요약(Abstract) :    




---



기존의 대형 언어 모델(LLM)은 문맥을 잘 이해하고 풍부한 텍스트 생성을 잘하지만, **긴 입력을 효율적으로 처리하는 데 한계**가 있습니다. 이는 **고정된 위치 임베딩**과 **길이에 따라 급격히 증가하는 연산 비용** 때문입니다. 이러한 문제를 해결하기 위해, 이 논문에서는 전체 모델을 재학습하지 않고도 긴 입력을 처리할 수 있도록 하는 방법인 **LCIRC(Long-form Context Injection with Recurrent Compression)**를 제안합니다. LCIRC는 긴 문맥을 반복적으로 압축하여 핵심 정보를 요약하고 이를 모델에 다시 주입합니다.

또한, 사용자의 **질문(query)에 따라 중요한 정보만 선별적으로 압축**하는 **질문-의존 문맥 모델링(Query Dependent Context Modeling)** 기법도 함께 도입했습니다. 실험 결과, 이 방법(QD-LCIRC)은 LLM의 긴 문맥 처리 성능을 크게 향상시키며, **문맥 이해와 질문 관련성**이 동시에 필요한 작업에 적합함을 보였습니다.

---



While large language models (LLMs) excel in generating coherent and contextually rich outputs, their capacity to efficiently handle long-form contexts is limited by fixed-length position embeddings. Additionally, the computational cost of processing long sequences increases quadratically, making it challenging to extend context length. To address these challenges, we propose Long-form Context Injection with Recurrent Compression (LCIRC), a method that enables the efficient processing long-form sequences beyond the model’s length limit through recurrent compression without retraining the entire model. We further introduce query dependent context modeling, which selectively compresses query-relevant information, ensuring that the model retains the most pertinent content. Our empirical results demonstrate that Query Dependent LCIRC (QD-LCIRC) significantly improves LLM’s ability to manage extended contexts, making it well-suited for tasks that require both comprehensive context understanding and query relevance.

---

필요하시면 본문이나 실험 결과에 대한 부분도 요약해드릴 수 있어요!


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





---


#### 1. **백본(Backbone) 모델**
- 이 논문에서는 **Llama2-7B** 모델을 기반으로 사용합니다.
- 기존 Llama2는 최대 4K 토큰까지만 처리할 수 있어서 긴 문맥을 다루기에 한계가 있었는데, 이 논문은 이를 확장하는 구조를 제안합니다.

#### 2. **구조(모델 아키텍처)**
- 제안하는 모델은 **LCIRC (Long-form Context Injection with Recurrent Compression)** 구조로 이루어져 있으며, 두 가지 핵심 구성 요소가 있습니다:
  
  1) **Recurrent Context Compression (반복 문맥 압축)**  
     - 긴 입력 시퀀스를 여러 개의 세그먼트로 나눠 순차적으로 처리하며, 이전 단계에서 압축된 결과를 다음 세그먼트의 압축에 활용하는 구조입니다.  
     - 이 과정에서 **Perceiver 아키텍처**를 활용하여, 세그먼트별로 cross-attention 기반으로 문맥 정보를 압축합니다.
  
  2) **Compressed Context Injection (압축된 문맥 삽입)**  
     - 이렇게 생성된 압축 벡터는 기존 LLM에 **Gated Cross Attention 블록**을 통해 삽입되어, 기존 문맥 처리에 통합됩니다.  
     - LLM의 파라미터는 **동결(frozen)**되어 있으며, 새로운 블록만 학습됩니다.

- 추가적으로, **Query Dependent LCIRC(QD-LCIRC)**는 질문에 따라 중요한 정보만 압축하도록 **질문 정보(query embedding)**를 추가적으로 사용하는 구조를 포함합니다.

#### 3. **트레이닝 데이터**
- **FineWeb-Edu**: 1.3조 토큰으로 구성된 대규모 텍스트 코퍼스에서 최소 길이 4K 이상 텍스트만 추출해 사용. 최대 길이 339K까지 포함.
- **FineWeb-LQA**: FineWeb-Edu 기반으로 생성한 long-form QA 데이터셋. Llama3.1-70B를 활용해 질문-답변 쌍을 자동 생성함. 이 데이터는 QD-LCIRC의 fine-tuning에 사용됨.

---



#### 1. **Backbone Model**
- The backbone is **Llama2-7B**, a transformer-based large language model.
- Since Llama2 can only handle up to 4K tokens natively, LCIRC was developed to overcome this limitation by enabling long-context processing.

#### 2. **Architecture (Model Structure)**

LCIRC consists of two main components:

1) **Recurrent Context Compression**  
   - Long input sequences are split into multiple segments.
   - Each segment is compressed using the **Perceiver architecture**, where the compressed output from the previous segment becomes the query for the next.
   - This results in a compact and memory-efficient representation of long-form context.

2) **Compressed Context Injection**  
   - The compressed vectors are injected into the frozen LLM through a **Gated Cross Attention (GCA) block**.
   - Only the GCA layers are trained, while the base LLM remains unchanged, preserving its original capabilities.

An extension, **Query Dependent LCIRC (QD-LCIRC)**, introduces a query-aware compression mechanism by incorporating **query embeddings** into the compression step, enabling relevance-aware context selection.

#### 3. **Training Data**
- **FineWeb-Edu**: A 1.3T-token corpus extracted from FineWeb, filtered to include texts longer than 4K tokens (up to 339K).
- **FineWeb-LQA**: A query-answering dataset automatically generated from FineWeb-Edu using Llama3.1-70B-Instruct. Used to fine-tune QD-LCIRC for query-dependent modeling.





   
 
<br/>
# Results  




---


#### 1. **경쟁 모델 (Baseline Models)**
- **Llama2-7B**: 기본 백본 모델 (최대 4K 토큰)
- **ExtendedFA**: RoPE θ를 수정하여 Llama2의 문맥 길이를 8K로 확장한 모델
- **AutoCompressor**: 세그먼트 단위로 입력을 나눠 반복적으로 처리하는 기존 압축 방식
- **LCIRC (제안)**: 반복 압축 및 문맥 삽입을 도입한 본 논문의 모델
- **QD-LCIRC (제안)**: LCIRC에 질의 기반 문맥 압축을 추가한 확장 모델

####  2. **테스트 데이터셋 (Evaluation Datasets)**
- **FineWeb-Edu**: 언어 모델링(perplexity) 평가용. 최대 339K 토큰까지 포함.
- **FineWeb-LQA**: FineWeb 기반으로 생성한 긴 질의응답 데이터셋. QD-LCIRC 훈련용.
- **InfiniteBench**: 100K 이상 초장문 컨텍스트를 요구하는 QA 벤치마크.
- **LongBench**: 다양한 단일/다중 문서 QA 작업 포함 (최대 82K 토큰).
- **L-Eval**: 도메인별 508개 문서 기반 QA 평가셋. 최대 200K 토큰.

####  3. **평가 메트릭 (Evaluation Metrics)**
- **Perplexity (언어 모델링 정확도)** – FineWeb-Edu에서 사용
- **F1 Score** – 주관식 QA 평가 (LongBench, InfiniteBench, L-Eval)
- **Accuracy** – 객관식 QA 평가

####  4. **결과 비교 요약**
- **Perplexity**:
  - LCIRC와 QD-LCIRC는 기존 모델보다 **더 긴 컨텍스트(최대 128K)**에서 낮은 perplexity 유지.
  - AutoCompressor는 64K 이상에서는 성능 악화.

- **연산 효율성**:
  - ExtendedFA는 토큰 수에 따라 **복잡도 급증 (10K TFLOPs 이상)**.
  - LCIRC는 동일한 성능에서 **최대 99% 연산량 절감**.
  - QD-LCIRC는 소폭 추가 연산이 있지만 효율 유지.

- **벤치마크 QA 성능**:
  - **QD-LCIRC**는 **모든 데이터셋에서 평균 성능 최고** 달성.
    - InfiniteBench: Llama2 대비 **308% 향상**
    - LongBench: **90% 향상**
    - L-Eval: ExtendedFA fine-tuned 모델 대비 **11.5% 향상**

- **Selective State BPTT** 학습 방식은 Truncated BPTT보다 성능 더 우수.

---


####  1. **Baseline Models**
- **Llama2-7B**: Original backbone, 4K token limit.
- **ExtendedFA**: Llama2 with modified RoPE to support up to 8K tokens.
- **AutoCompressor**: Recurrent prompt compression using segment feeding.
- **LCIRC (Ours)**: Our model using recurrent compression and context injection.
- **QD-LCIRC (Ours)**: Our extended model with query-dependent compression.

####  2. **Evaluation Datasets**
- **FineWeb-Edu**: Used for perplexity evaluation; contains texts up to 339K tokens.
- **FineWeb-LQA**: Automatically generated long-form QA dataset used to fine-tune QD-LCIRC.
- **InfiniteBench**: Benchmark for ultra-long context tasks (>100K tokens).
- **LongBench**: Multiple real/synthetic QA tasks with context up to 82K tokens.
- **L-Eval**: Long document QA benchmark with context up to 200K tokens.

####  3. **Metrics**
- **Perplexity**: For language modeling on FineWeb-Edu.
- **F1 Score**: For open-ended QA tasks.
- **Accuracy**: For multiple-choice QA tasks.

####  4. **Performance Comparison Summary**
- **Perplexity**:
  - LCIRC and QD-LCIRC achieve lower perplexity at large context sizes (up to 128K).
  - AutoCompressor performance degrades beyond 64K tokens.

- **Efficiency**:
  - ExtendedFA has rapidly increasing FLOPs with token length (up to 10K+ TFLOPs).
  - LCIRC achieves up to **99% reduction in computation** while maintaining performance.
  - QD-LCIRC adds minimal overhead while retaining efficiency.

- **QA Benchmark Results**:
  - **QD-LCIRC** outperforms all baselines across all datasets:
    - InfiniteBench: ~**308% improvement** over Llama2.
    - LongBench: ~**90% improvement**.
    - L-Eval: ~**11.5% improvement** over the best finetuned baseline (ExtendedFA).

- **Selective State BPTT**: Outperforms Truncated BPTT in all benchmark tests.

---



<br/>
# 예제  



---


#### 1. **훈련 데이터 예시 (Training Data Examples)**

#####  **FineWeb-Edu**
- 약 **1.3조 토큰** 규모의 텍스트로, 교육적이고 일반적인 웹 문서에서 추출됨.
- 각 문서는 **최소 4K 토큰 이상**, 최대 **339K 토큰**까지 포함.
- 문서 예시 (구체 문장은 논문에 없지만 일반적인 형식은 다음과 같음):
  ```
  [문서 내용] "The Treaty of Versailles was signed in 1919 and officially ended World War I. The treaty imposed..."
  ```

#####  **FineWeb-LQA (Long-form QA 데이터셋)**
- FineWeb-Edu에서 문맥을 추출한 후, **Llama3.1-70B-Instruct** 모델을 이용해 질문-답변 쌍을 자동 생성.
- 예시:
  - **문맥 (context)**: 128K 토큰 길이의 실제 문서 일부
  - **질문 (question)**: “What were the main consequences of the Treaty of Versailles?”
  - **답변 (answer)**: “It imposed harsh reparations on Germany and led to political unrest.”

####  2. **테스트 데이터 예시 (Test Data Examples)**

##### **InfiniteBench**
- 초장문 QA 벤치마크.
- 예시:
  - **문맥**: 장편 소설 또는 수십만 토큰짜리 리포트
  - **질문**: “What was the author’s main argument in chapter 17?”
  - **형식**: 객관식 또는 주관식 응답

#####  **LongBench**
- 단일 또는 다중 문서 QA 테스트.
- 예시:
  - **문맥**: 여러 기사 혹은 논문 결합
  - **질문**: “Which country had the largest carbon emissions in 2020?”
  - **정답**: “China”

#####  **L-Eval**
- 도메인별 실제 문서 기반의 QA (뉴스, 논문, SF, 에듀 등)
- 예시:
  - **문맥**: "In the Coursera course on machine learning, one key concept was regularization..."
  - **질문**: "What is the purpose of regularization?"
  - **정답**: "To prevent overfitting by penalizing complex models."

---



####  1. **Training Data Examples**

#####  **FineWeb-Edu**
- A 1.3 trillion token corpus sourced from educational and general web documents.
- Each document contains at least 4K tokens and up to 339K tokens.
- Example (approximate):
  ```
  [Document] "The Treaty of Versailles was signed in 1919 and officially ended World War I. The treaty imposed..."
  ```

#####  **FineWeb-LQA**
- Generated from FineWeb-Edu using **Llama3.1-70B-Instruct** to produce QA pairs.
- Example:
  - **Context**: Excerpt from a long document (~128K tokens)
  - **Question**: “What were the main consequences of the Treaty of Versailles?”
  - **Answer**: “It imposed harsh reparations on Germany and led to political unrest.”

####  2. **Test Data Examples**

#####  **InfiniteBench**
- Benchmark for ultra-long context QA.
- Example:
  - **Context**: Long novel or report (>100K tokens)
  - **Question**: “What was the author’s main argument in chapter 17?”
  - **Answer type**: Multiple-choice or open-ended

#####  **LongBench**
- Evaluates both single- and multi-document QA.
- Example:
  - **Context**: Merged documents such as multiple articles or reports
  - **Question**: “Which country had the largest carbon emissions in 2020?”
  - **Answer**: “China”

#####  **L-Eval**
- Long document QA across various domains (e.g., news, education, sci-fi).
- Example:
  - **Context**: "In the Coursera course on machine learning, one key concept was regularization..."
  - **Question**: "What is the purpose of regularization?"
  - **Answer**: "To prevent overfitting by penalizing complex models."

---





<br/>  
# 요약   



이 논문은 LLM이 긴 문맥을 효율적으로 처리할 수 있도록, 문맥을 반복적으로 압축하고 이를 다시 모델에 주입하는 **LCIRC** 구조를 제안하며, **질문에 따라 중요한 정보만 선별적으로 반영하는 QD-LCIRC**도 함께 도입했다.  
실험 결과, QD-LCIRC는 기존 Llama2 및 AutoCompressor, ExtendedFA 등 경쟁 모델보다 **최대 308% 향상된 성능**을 보이며, **128K 이상의 토큰 처리와 낮은 연산 복잡도**를 동시에 달성했다.  
학습에는 최대 339K 토큰까지 포함된 **FineWeb-Edu**, 테스트에는 **InfiniteBench, LongBench, L-Eval** 등 초장문 QA 데이터셋이 활용되었으며, 실제 예시로 긴 문서 속 특정 질문에 대한 정답을 생성하는 과제가 포함된다.

---


This paper proposes **LCIRC**, a method that recurrently compresses long contexts and injects them back into LLMs, along with **QD-LCIRC**, which selectively retains query-relevant content.  
Experimental results show that QD-LCIRC outperforms baselines like Llama2, AutoCompressor, and ExtendedFA, achieving up to **308% performance gains** while supporting **128K+ token sequences with low computational cost**.  
Training used the **FineWeb-Edu** dataset (up to 339K tokens), and evaluation was conducted on **InfiniteBench, LongBench, and L-Eval**, with tasks involving answering questions from extremely long documents.


<br/>  
# 기타  



---

###  **부록 (Appendix)**
부록에서는 **Selective State BPTT (Backpropagation Through Time)**와 **Truncated BPTT**를 비교한 추가 실험 결과를 제시합니다Selective State BPTT는 Truncated BPTT보다 긴 문맥을 더 효과적으로 학습할 수 있음을 보여줍니다이러한 결과는 **InfiniteBench**, **LongBench**, **L-Eval** 벤치마크에서의 성능 향상을 통해 확인되었습니다

---

###  **그림 (Figures)**

- **Figure 1** **LCIRC의 전체 프로세스**를 시각화한 다이어그램입니. 왼쪽은 **Recurrent Context Compression**을, 오른쪽은 **Compressed Context Injection**을 보여줍니. 각 단계에서 Perceiver 모듈과 Gated Cross Attention 블록의 상호작용을 상세히 나타냅니.

- **Figure 2** **질문 의존적 문맥 압축(Query Dependent Context Modeling)**을 기존의 문맥 압축과 비교한 도표입니. 추가된 cross-attention 모듈이 어떻게 질문 정보를 압축된 특징에 주입하는지 설명합니.

- **Figure 3** **Selective State BPTT**와 **Truncated BPTT**의 비교를 시각화한 그래프입니. Selective State BPTT가 긴 문맥 학습에서 더 나은 성능을 보임을 강조합니.

---

###  **표 (Tables)**

- **Table 6*: **Selective State BPTT**와 **Truncated BPTT**의 성능 비교 표입다 각 벤치마크(InfiniteBench, LongBench, L-Eval)에서 Selective State BPTT가 더 높은 점수를 기록했음을 보여줍다.

--


---

### **Appendix*

The appendix provides additional experimental results comparing **Selective State BPTT (Backpropagation Through Time)** and **Truncated BPT*. It demonstrates that Selective State BPTT is more effective in learning long contexts, as evidenced by performance improvements in benchmarks like **InfiniteBench**, **LongBench**, and **L-Eva**.

---

###  **Figures**

- **Figure **: A diagram illustrating the **overall process of LCIC*. The left side depicts **Recurrent Context Compression**, while the right side shows **Compressed Context Injection**, detailing the interactions between the Perceiver module and Gated Cross Attention blcks.

- **Figure **: A comparison of **Query Dependent Context Modeling** with standard context compresin. It explains how the additional cross-attention module injects query information into the compressed featres.

- **Figure **: A graph comparing **Selective State BPTT** and **Truncated BPTT**, highlighting the superior performance of Selective State BPTT in learning long contxts.

---

###  **Tables**

- **Table6**: A performance comparison between **Selective State BPTT** and **Truncated BT**. It shows that Selective State BPTT achieves higher scores across benchmarks like InfiniteBench, LongBench, and LEval.
---




<br/>
# refer format:     


@article{an2025lcirc,
  title={LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs},
  author={An, Sumin and Sung, Junyoung and Park, Wonpyo and Park, Chanjun and Seo, Paul Hongsuck},
  journal={arXiv preprint arXiv:2502.06139},
  year={2025}
}



An, Sumin, Junyoung Sung, Wonpyo Park, Chanjun Park, and Paul Hongsuck Seo. “LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs.” arXiv preprint arXiv:2502.06139 (2025).  




