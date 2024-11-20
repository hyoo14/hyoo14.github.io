---
layout: post
title:  "[2024]Lost in the Middle: How Language Models Use Long Contexts"  
date:   2024-11-19 10:38:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


아, 이거 수업때 봤던거네... 중간보다는 처음이나 끝 질문에 더 잘 대답했다는 연구내용! (LLM이)    


짧은 요약(Abstract) :    



이 논문은 긴 입력 컨텍스트를 활용하는 언어 모델의 성능과 한계를 분석합니다. 구체적으로, 멀티 문서 질문 응답 및 키-값 검색 작업을 통해 입력 컨텍스트 내에서 관련 정보의 위치에 따라 성능이 어떻게 변화하는지 연구했습니다. 연구 결과, 관련 정보가 컨텍스트의 시작이나 끝에 있을 때 성능이 가장 높았으며, 중간에 있을 때 성능이 크게 저하되었습니다. 이는 언어 모델이 긴 컨텍스트의 정보를 일관성 있게 활용하지 못함을 시사합니다. 또한, 모델 아키텍처, 쿼리 중심 컨텍스트화, 명령어 미세 조정의 역할을 추가적으로 분석했으며, 긴 컨텍스트를 더 잘 활용하기 위한 평가 프로토콜을 제안합니다.


This study investigates how well language models utilize long input contexts through controlled experiments on tasks like multi-document question answering and key-value retrieval. The findings reveal significant performance degradation when the position of relevant information changes, with models performing best when the information is at the beginning or end of the context and poorly when it is in the middle. This indicates that current models struggle to consistently leverage information in long contexts. Additional analyses of model architecture, query-aware contextualization, and instruction fine-tuning provide insights, and new evaluation protocols for long-context models are introduced.



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




이 논문은 언어 모델이 긴 입력 컨텍스트에서 정보를 활용하는 방식을 연구하기 위해 두 가지 작업을 수행했습니다: **멀티 문서 질문 응답(Multi-Document Question Answering, QA)**와 **키-값 검색(Key-Value Retrieval)**입니다. 이 작업들은 입력 컨텍스트 내에서 관련 정보를 찾고 이를 활용하여 정답을 생성해야 하는 설정을 기반으로 했습니다. 연구에서는 다음과 같은 모델을 사용했습니다:  
1. **MPT-30B-Instruct**: 최대 8192 토큰의 컨텍스트를 처리할 수 있으며, 추가적으로 시퀀스 길이 적응 훈련을 수행.  
2. **LongChat-13B**: 기존 LLaMA-13B 모델의 컨텍스트 창을 2048에서 16384 토큰으로 확장.  
3. **GPT-3.5-Turbo 및 Claude-1.3**: OpenAI와 Anthropic의 상용 언어 모델을 사용, 확장된 컨텍스트 창(예: 16K, 100K 토큰)을 포함.  
4. **Flan-T5-XXL 및 Flan-UL2**: 인코더-디코더 기반 모델로, 상대적 위치 임베딩을 통해 긴 컨텍스트에서 더 나은 성능을 보임.  

모델의 성능은 다음을 기준으로 평가되었습니다:  
- 관련 정보의 위치(시작, 중간, 끝)에 따른 성능 변화.  
- 입력 컨텍스트의 길이를 늘리거나 줄임에 따른 성능 변화.  



- **기존 연구와의 유사점**: 이 연구는 기존의 Transformer 기반 모델처럼 긴 입력 시퀀스를 다루기 위한 작업을 기반으로 했습니다. 특히, 기존 연구에서도 컨텍스트의 시작 부분 또는 끝 부분에서 더 나은 성능을 보이는 "Primacy"와 "Recency" 편향이 관찰되었습니다.  
- **차이점**: 이 연구는 단순히 긴 시퀀스를 다룰 수 있는 모델의 성능을 평가하는 데 그치지 않고, 모델이 관련 정보를 정확히 식별하고 활용하는 데 필요한 메커니즘과 제약을 상세히 분석했습니다. 특히, JSON 기반 키-값 검색 작업처럼 언어적 의미를 배제한 단순 검색 작업에서도 U자형 성능 곡선을 확인하며 모델이 중간 컨텍스트를 처리하는 데 어려움을 겪는 이유를 탐구했습니다.  

---

**English Explanation**  
The paper investigated how language models utilize information within long input contexts through two tasks: **Multi-Document Question Answering (QA)** and **Key-Value Retrieval**. These tasks require models to locate relevant information in the input context and generate answers based on it. The following models were analyzed:  
1. **MPT-30B-Instruct**: Handles up to 8192 tokens and incorporates sequence length adaptation training.  
2. **LongChat-13B**: Extends the LLaMA-13B model's context window from 2048 to 16384 tokens.  
3. **GPT-3.5-Turbo and Claude-1.3**: Commercial models by OpenAI and Anthropic, including extended context windows (e.g., 16K, 100K tokens).  
4. **Flan-T5-XXL and Flan-UL2**: Encoder-decoder models with relative positional embeddings, providing robust performance with long contexts.  

The model performance was evaluated based on:  
- The effect of relevant information's position (beginning, middle, end).  
- Changes in performance with varying input context lengths.  

**Similarities and Differences with Prior Methods**  
- **Similarities**: Like prior Transformer-based studies, this work also observed "Primacy" and "Recency" biases, where models performed better on information at the beginning or end of the context.  
- **Differences**: This study went beyond evaluating models' ability to handle long sequences by analyzing mechanisms and limitations in identifying and leveraging relevant information. For instance, the JSON-based Key-Value Retrieval task removed linguistic semantics to isolate token retrieval abilities. The findings confirmed that even in such simplified tasks, models struggle with information in the middle of the context, highlighting fundamental challenges in long-context processing.


   
 
<br/>
# Results  

 

### 결과
연구 결과, 언어 모델은 입력 컨텍스트 내에서 관련 정보의 위치에 따라 성능이 크게 변화했습니다. 특히, 관련 정보가 **컨텍스트의 시작(Primacy Bias)**이나 **끝(Recency Bias)**에 있을 때 성능이 가장 높았으며, 중간에 있을 때는 성능이 크게 저하되는 **U자형 곡선**을 보였습니다. 일부 모델의 경우, 관련 정보가 중간에 위치할 경우 성능이 "닫힌 책(closed-book)" 설정보다도 낮아졌습니다.

- **멀티 문서 질문 응답(Multi-Document QA)** 작업에서:
  - **GPT-3.5-Turbo**는 중간 위치 정보에서 약 20% 성능 감소를 보였습니다. 
  - 확장된 컨텍스트 모델 (e.g., GPT-3.5-Turbo 16K)도 기본 모델과 동일한 경향을 보이며, 컨텍스트 창 확장이 반드시 성능 향상을 의미하지 않음을 보여줍니다.

- **키-값 검색(Key-Value Retrieval)** 작업에서:
  - Claude-1.3 모델은 거의 모든 컨텍스트 길이에서 완벽한 성능을 보였지만, 다른 모델(GPT-3.5-Turbo, LongChat-13B 등)은 중간 위치에서 성능이 저하되었습니다.

---

### 비교 모델
다양한 최신 모델들이 비교되었습니다:
1. **MPT-30B-Instruct**  
2. **LongChat-13B**  
3. **GPT-3.5-Turbo (4K 및 16K 컨텍스트 창)**  
4. **Claude-1.3 (8K 및 100K 컨텍스트 창)**  
5. **Flan-T5-XXL 및 Flan-UL2 (인코더-디코더 기반 모델)**  

---

### 사용한 데이터셋
1. **NaturalQuestions-Open**: 구글 검색 엔진에서 수집된 질문과 인간 주석이 포함된 위키피디아 데이터를 활용하여 QA 작업을 수행.  
2. **Synthetic Key-Value Dataset**: JSON 형식의 키-값 쌍 데이터를 생성하여 기본 검색 기능을 평가.  

---

### 성능 향상
- 멀티 문서 QA 작업에서, 관련 정보가 컨텍스트 끝에 있을 때 **Claude-1.3**의 성능은 약 76%에서 100K 컨텍스트 모델로 소폭 향상되었습니다.  
- 키-값 검색 작업에서 쿼리를 앞뒤로 배치하는 쿼리 중심 컨텍스트화(Query-Aware Contextualization)는 일부 모델(GPT-3.5-Turbo)의 성능을 45%에서 100%로 향상시켰습니다.  
- 그러나 확장된 컨텍스트 창이 직접적인 성능 향상을 보장하지 않으며, 모델이 컨텍스트 중간 정보를 처리하는 데 어려움을 겪었습니다.

---



### Results
The study revealed that the performance of language models significantly depends on the position of relevant information within the input context. Specifically:  
- Models performed best when relevant information was located at the **beginning (Primacy Bias)** or **end (Recency Bias)** of the input context.  
- A **U-shaped curve** was observed, where performance dropped sharply when the information was located in the middle of the context.  
- For some models, performance on mid-context information was even lower than the **closed-book** setting (without any input documents).

- **Multi-Document QA** results:  
  - **GPT-3.5-Turbo** experienced a performance drop of up to 20% when relevant information was in the middle of the context.  
  - Extended-context models (e.g., GPT-3.5-Turbo 16K) showed similar trends, indicating that larger context windows do not necessarily translate to better performance.  

- **Key-Value Retrieval** results:  
  - While Claude-1.3 achieved near-perfect performance across all settings, other models (e.g., GPT-3.5-Turbo, LongChat-13B) struggled with mid-context information.

---

### Comparison Models
The study evaluated a variety of state-of-the-art models:  
1. **MPT-30B-Instruct**  
2. **LongChat-13B**  
3. **GPT-3.5-Turbo (4K and 16K context windows)**  
4. **Claude-1.3 (8K and 100K context windows)**  
5. **Flan-T5-XXL and Flan-UL2 (encoder-decoder models)**  

---

### Datasets Used
1. **NaturalQuestions-Open**: Questions sourced from Google Search, annotated with answers from Wikipedia, were used for the QA task.  
2. **Synthetic Key-Value Dataset**: JSON-formatted key-value pairs were generated to evaluate basic retrieval capabilities.  

---

### Performance Improvements
- In the Multi-Document QA task, placing relevant information at the end of the context slightly improved **Claude-1.3**'s accuracy, from around 76% to higher with the 100K context model.  
- Query-aware contextualization (placing queries before and after the context) improved the Key-Value Retrieval task performance for models like GPT-3.5-Turbo from **45% to 100%**.  
- However, extending the context window did not guarantee better performance, as models still struggled with mid-context information.




<br/>
# 예제  




### 구체적인 예시: 멀티 문서 질문 응답(Multi-Document QA)

**데이터셋 예시**:  
질문: "Who wrote *To Kill a Mockingbird*?"  
입력 컨텍스트:  
1. 문서 1: "Harper Lee는 유명한 미국 작가로, *To Kill a Mockingbird*의 저자이다." (정답 포함)  
2. 문서 2: "이 소설은 1960년에 출간되었다."  
3. 문서 3: "책의 배경은 1930년대 미국 남부이다."  

**처리 방식**:  
모델은 입력된 컨텍스트를 분석하여 질문과 가장 관련 있는 정보를 찾습니다. 이 예시에서, **문서 1**에 정답이 포함되어 있지만 문서 순서를 바꿔서 정보를 컨텍스트의 시작, 중간, 끝에 배치했습니다.  

**결과**:  
- 관련 정보가 **문서 1이 컨텍스트의 시작**에 위치할 때 성능이 가장 높았습니다.  
- 관련 정보가 **중간**에 위치할 때, **GPT-3.5-Turbo**의 성능은 약 20% 감소했으며, 모델은 정보를 찾는 데 어려움을 겪었습니다.  
- 관련 정보가 **끝**에 위치했을 때도 높은 성능을 보였지만 시작보다 약간 낮았습니다.

---

### 구체적인 예시: 키-값 검색(Key-Value Retrieval)

**데이터셋 예시**:  
입력 JSON:  
```json
{
  "key1": "value1",
  "key2": "value2",
  "key3": "value3"
}
```  
질문: `"What is the value of key2?"`

**처리 방식**:  
모델은 JSON 데이터를 분석하여 질문에서 제공된 키(`key2`)와 일치하는 값을 찾습니다. 쿼리를 JSON 데이터 앞뒤로 배치하는 **쿼리 중심 컨텍스트화(Query-Aware Contextualization)** 방식을 사용했습니다.  

**결과**:  
- 쿼리가 앞뒤에 배치되었을 때, GPT-3.5-Turbo 모델은 **100% 정확도**로 `"value2"`를 반환했습니다.  
- 쿼리가 뒤에만 배치된 경우, 중간 위치 키(`key2`)에서 성능이 45%로 떨어졌습니다.  
- Claude-1.3은 모든 설정에서 완벽한 성능을 보였습니다.

---


### Specific Example: Multi-Document QA

**Dataset Example**:  
Question: "Who wrote *To Kill a Mockingbird*?"  
Input Context:  
1. Document 1: "Harper Lee is a famous American author and the writer of *To Kill a Mockingbird*." (Contains the answer)  
2. Document 2: "This novel was published in 1960."  
3. Document 3: "The story is set in the 1930s in the American South."  

**Processing**:  
The model analyzes the input context to find the information most relevant to the question. In this example, **Document 1** contains the answer, but the order of the documents was altered to position the relevant information at the beginning, middle, and end of the context.  

**Results**:  
- When the relevant information was at the **beginning** of the context, performance was highest.  
- When it was in the **middle**, **GPT-3.5-Turbo**'s accuracy dropped by about 20%, as the model struggled to locate the answer.  
- At the **end**, performance was still high but slightly lower than when the answer was at the start.

---

### Specific Example: Key-Value Retrieval

**Dataset Example**:  
Input JSON:  
```json
{
  "key1": "value1",
  "key2": "value2",
  "key3": "value3"
}
```  
Query: `"What is the value of key2?"`

**Processing**:  
The model processes the JSON data to identify the key (`key2`) mentioned in the query and retrieve its corresponding value. The **query-aware contextualization** method was applied, placing the query both before and after the JSON data.

**Results**:  
- With the query placed before and after the JSON, GPT-3.5-Turbo achieved **100% accuracy**, returning `"value2"`.  
- When the query was only placed after the JSON, performance dropped to 45% for mid-context keys like `key2`.  
- Claude-1.3 consistently performed perfectly across all settings.

<br/>  
# 요약   


<br/>  
# 기타  


<br/>
# refer format:     

@article{liu2024lost,
  author    = {Nelson F. Liu and Kevin Lin and John Hewitt and Ashwin Paranjape and Michele Bevilacqua and Fabio Petroni and Percy Liang},
  title     = {Lost in the Middle: How Language Models Use Long Contexts},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {12},
  pages     = {157--173},
  year      = {2024},
  doi       = {10.1162/tacl_a_00638},
  publisher = {Association for Computational Linguistics}
}




Liu, Nelson F., Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. "Lost in the Middle: How Language Models Use Long Contexts." Transactions of the Association for Computational Linguistics 12: 157–173.

