---
layout: post
title:  "[2024]DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts Language Model"  
date:   2024-11-19 11:20:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


MLA(Multi-head Latent Attention)는 Key-Value(KV) 캐시를 저차원 벡터로 압축하는 기술을 사용하여 파라미터 개수를 줄이고도 성능을 유지할 수 있었습니다.   


짧은 요약(Abstract) :    




DeepSeek-V2는 강력하고 경제적인 Mixture-of-Experts(MoE) 언어 모델로, 총 2360억 개의 매개변수를 포함하며, 각 토큰당 활성화된 매개변수는 210억 개입니다. 이 모델은 128K 토큰의 컨텍스트 길이를 지원하며, Multi-head Latent Attention(MLA)와 DeepSeekMoE 같은 혁신적인 아키텍처를 채택했습니다. MLA는 Key-Value(KV) 캐시를 압축하여 추론 효율을 극대화하며, DeepSeekMoE는 희소 계산을 통해 훈련 비용을 줄이는 데 기여합니다. DeepSeek 67B와 비교하여 DeepSeek-V2는 훈련 비용을 42.5% 절감하고, KV 캐시 크기를 93.3% 줄이며, 최대 생성 처리량을 5.76배 향상시켰습니다. 이 모델은 8.1조 개의 토큰으로 구성된 고품질 데이터로 사전 학습되었으며, 감독 학습(SFT) 및 강화 학습(RL)을 통해 성능을 더욱 향상시켰습니다. DeepSeek-V2는 활성화된 매개변수가 적음에도 불구하고, 오픈 소스 모델 중 최고 성능을 달성했습니다.

---


DeepSeek-V2 is a powerful and economical Mixture-of-Experts (MoE) language model, featuring 236 billion total parameters with 21 billion activated per token. It supports a context length of 128K tokens and incorporates innovative architectures like Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA significantly enhances inference efficiency by compressing the Key-Value (KV) cache, while DeepSeekMoE reduces training costs through sparse computation. Compared to DeepSeek 67B, DeepSeek-V2 achieves a 42.5% reduction in training costs, a 93.3% reduction in KV cache size, and a 5.76x increase in maximum generation throughput. Pretrained on 8.1 trillion tokens of high-quality data, the model undergoes Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for further performance improvements. Despite fewer activated parameters, DeepSeek-V2 achieves top-tier performance among open-source models.



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




DeepSeek-V2는 다음과 같은 핵심 메서드를 통해 성능을 최적화했습니다:

1. **Multi-head Latent Attention (MLA)**  
   - 기존 Multi-Head Attention(MHA)의 Key-Value(KV) 캐시가 추론 과정에서 병목현상을 일으키는 문제를 해결하기 위해, MLA는 Key와 Value를 저차원 벡터로 압축하는 **저차원 Key-Value 압축(joint compression)** 방식을 도입했습니다.  
   - 기존 MHA 대비 성능은 유지하거나 향상하면서도, KV 캐시 크기를 대폭 줄임으로써 추론 효율성을 높였습니다.  
   - **차이점**: 기존의 Multi-Query Attention(MQA) 및 Grouped-Query Attention(GQA) 방식은 KV 캐시 크기를 줄이지만, 성능이 저하될 가능성이 있었습니다. MLA는 이 문제를 극복하여, GQA 수준의 캐시 크기를 유지하면서 MHA 이상의 성능을 달성했습니다.

2. **DeepSeekMoE 아키텍처**  
   - Feed-Forward Networks(FFNs)에서 사용되는 Mixture-of-Experts(MoE) 구조를 개선하기 위해, 전문가를 더 세분화(fine-grained)하고 일부 전문가를 공유(shared)하도록 설계했습니다.  
   - **경제적 훈련**: 전문가 병렬화를 활용하면서, 추가적인 통신 비용을 보완하기 위한 **디바이스 제한 라우팅(device-limited routing)** 메커니즘을 적용했습니다. 이 메커니즘은 한 토큰이 최대 3개의 디바이스에만 라우팅되도록 제한하여 통신 효율성을 보장합니다.  
   - **차이점**: 기존 GShard 구조와 달리, DeepSeekMoE는 더 높은 전문가 특화(specialization)를 통해 강력한 성능을 제공합니다. 

3. **훈련 및 추론 최적화**  
   - 사전 학습: 8.1조 개의 고품질 데이터를 사용하여 사전 학습을 수행하며, 데이터 품질 필터링을 통해 데이터 편향을 줄였습니다.  
   - Supervised Fine-Tuning(SFT) 및 Reinforcement Learning(RL): 모델을 인간의 선호도에 맞춰 조정하고 성능을 더욱 강화했습니다.  
   - KV 캐시 양자화 및 FP8 정밀도를 활용하여 추론 효율성을 높였습니다. 

**요약**:  
DeepSeek-V2는 MHA 대비 MLA의 효율성과 GShard 대비 DeepSeekMoE의 경제성을 통해 기존의 모델들과 차별화되며, KV 캐시 크기 감소, 훈련 비용 절감, 그리고 추론 처리량 증가 측면에서 강력한 성능을 발휘합니다.

---


DeepSeek-V2 employs the following key methods to optimize its performance:

1. **Multi-head Latent Attention (MLA)**  
   - To address the bottleneck caused by the Key-Value (KV) cache in standard Multi-Head Attention (MHA), MLA introduces **low-rank Key-Value joint compression**.  
   - This significantly reduces the KV cache size while maintaining or even enhancing the performance compared to MHA.  
   - **Difference**: Unlike Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), which reduce KV cache size at the expense of performance, MLA achieves GQA-level efficiency with stronger performance than MHA.

2. **DeepSeekMoE Architecture**  
   - For Feed-Forward Networks (FFNs), DeepSeekMoE employs **fine-grained expert segmentation** and **shared expert isolation** to achieve higher specialization and efficiency.  
   - **Economic Training**: Using expert parallelism, it applies a **device-limited routing** mechanism that ensures each token is routed to at most three devices, minimizing communication overheads.  
   - **Difference**: Unlike the GShard architecture, DeepSeekMoE delivers significantly stronger performance due to its finer granularity and better expert specialization.

3. **Training and Inference Optimization**  
   - Pretraining: DeepSeek-V2 is pretrained on 8.1 trillion tokens of high-quality data, incorporating filtering to reduce biases.  
   - Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL): These steps align the model with human preferences and further enhance performance.  
   - Optimizations like KV cache quantization and FP8 precision are used to improve inference efficiency.

**Summary**:  
DeepSeek-V2 distinguishes itself by combining MLA's efficiency over MHA and DeepSeekMoE's economic advantage over GShard, achieving remarkable reductions in KV cache size, training costs, and enhanced inference throughput.



   
 
<br/>
# Results  




### 비교 모델
DeepSeek-V2는 다음과 같은 대표 오픈소스 모델과 비교되었습니다:
- **DeepSeek 67B** (이전 버전)
- **Qwen1.5 72B**
- **LLaMA3 70B**
- **Mixtral 8x22B**

### 사용한 데이터셋
평가는 영어 및 중국어로 이루어진 다양한 벤치마크를 통해 진행되었습니다:
1. **언어 이해 및 추론**: HellaSwag, PIQA, ARC, BBH  
2. **폐쇄형 질문 응답**: TriviaQA, NaturalQuestions  
3. **독해**: RACE, DROP, CMRC  
4. **다중 선택 문제**: MMLU, C-Eval, CMMLU  
5. **코드**: HumanEval, MBPP  
6. **수학**: GSM8K, MATH, CMath  
7. **중국어 데이터셋**: CHID, CCPM 등

### 성능 향상
1. DeepSeek-V2는 활성화된 21B 매개변수만으로도 **DeepSeek 67B** 대비 거의 모든 벤치마크에서 성능을 능가했습니다.
   - 예: MMLU 정확도에서 DeepSeek 67B의 71.3%에서 78.5%로 향상.  
2. **Qwen1.5 72B**와 비교했을 때, DeepSeek-V2는 영어와 수학 벤치마크에서 더 높은 점수를 기록했으며, 중국어 다중 선택 문제에서는 유사하거나 더 나은 성능을 보였습니다.  
3. **LLaMA3 70B** 대비, 영어 기본 성능에서는 약간의 차이가 있지만, 코드와 수학에서는 유사한 성능을, 중국어 데이터셋에서는 더 나은 성능을 보였습니다.  
4. **Mixtral 8x22B** 대비, DeepSeek-V2는 영어와 코드에서 비슷하거나 더 나은 성능을 보였으며, 중국어에서는 월등히 앞섰습니다.

### 요약
DeepSeek-V2는 모델 크기를 효율적으로 줄이고, 훈련 비용을 42.5% 절감하며, 추론 속도를 5.76배로 향상시켰음에도 불구하고 상위 수준의 성능을 달성했습니다. 특히, 수학과 코딩 벤치마크에서 큰 향상을 보여줬습니다.

---


### Comparison Models
DeepSeek-V2 was compared against the following representative open-source models:
- **DeepSeek 67B** (previous version)
- **Qwen1.5 72B**
- **LLaMA3 70B**
- **Mixtral 8x22B**

### Datasets Used
Evaluations were conducted on a variety of benchmarks in English and Chinese:
1. **Language Understanding and Reasoning**: HellaSwag, PIQA, ARC, BBH  
2. **Closed-Book QA**: TriviaQA, NaturalQuestions  
3. **Reading Comprehension**: RACE, DROP, CMRC  
4. **Multi-Choice QA**: MMLU, C-Eval, CMMLU  
5. **Code**: HumanEval, MBPP  
6. **Math**: GSM8K, MATH, CMath  
7. **Chinese Datasets**: CHID, CCPM, etc.

### Performance Improvements
1. With only 21B activated parameters, DeepSeek-V2 significantly outperformed **DeepSeek 67B** on almost all benchmarks.
   - Example: On MMLU, accuracy improved from 71.3% (DeepSeek 67B) to 78.5%.  
2. Compared to **Qwen1.5 72B**, DeepSeek-V2 achieved higher scores on English and math benchmarks and showed comparable or better performance on Chinese multi-choice tasks.  
3. Against **LLaMA3 70B**, while slightly behind in basic English capabilities, DeepSeek-V2 demonstrated comparable performance on code and math benchmarks and substantially outperformed on Chinese datasets.  
4. Compared to **Mixtral 8x22B**, DeepSeek-V2 showed comparable or better performance in English and code, and significantly outperformed in Chinese benchmarks.

### Summary
DeepSeek-V2 achieves top-tier performance despite reducing model size, cutting training costs by 42.5%, and increasing inference speed by 5.76x. Notably, it shows substantial improvements in math and coding benchmarks.



<br/>
# 예제  




### 구체적인 예시: MMLU 데이터셋
#### 데이터셋 설명  
MMLU(Massive Multitask Language Understanding)는 57개의 도메인에 걸쳐 다양한 멀티초이스 질문을 포함하며, 모델의 일반화된 언어 이해 능력을 평가하는 데 사용됩니다.

#### 처리 과정  
1. **입력 데이터**:  
   - 질문: *"What is the capital of France?"*  
   - 선택지: (A) Madrid, (B) Paris, (C) Rome, (D) Berlin  
   - 정답: (B) Paris  

2. **모델 처리 방식**:  
   - DeepSeek-V2는 질문과 선택지를 함께 입력받아 학습된 Mixture-of-Experts(MoE) 아키텍처를 통해 적절한 전문가 레이어를 활성화합니다.  
   - Multi-head Latent Attention(MLA)을 통해 Key-Value 캐시를 압축하여 선택지 간 관계를 효율적으로 분석합니다.  

3. **예측 결과**:  
   - DeepSeek-V2는 문맥 정보와 선택지 간의 관계를 이해하여 (B) Paris를 선택합니다.  
   - MMLU에서 DeepSeek-V2는 78.5%의 정확도를 기록하며, DeepSeek 67B(71.3%)를 크게 상회했습니다.

---

### 구체적인 예시: HumanEval 데이터셋
#### 데이터셋 설명  
HumanEval은 주어진 코딩 문제에 대한 정확한 코드를 작성하는 모델의 능력을 평가합니다.

#### 처리 과정  
1. **입력 데이터**:  
   - 문제: *"Write a function that returns the sum of two integers."*  

2. **모델 처리 방식**:  
   - DeepSeek-V2는 문제를 자연어로 이해하고, Supervised Fine-Tuning(SFT)을 통해 학습된 코딩 데이터를 활용하여 정답 코드를 생성합니다.  
   - 추론 단계에서는 Reinforcement Learning(RL)을 통해 학습된 코드 품질 보상 모델이 정답 코드의 품질을 평가합니다.  

3. **예측 결과**:  
   - 출력 코드:
     ```python
     def add(a, b):
         return a + b
     ```
   - DeepSeek-V2는 HumanEval에서 81.1%의 정확도를 기록하며, Mixtral 8x22B(75.0%)를 능가했습니다.

---



### Specific Example: MMLU Dataset
#### Dataset Description  
MMLU (Massive Multitask Language Understanding) includes multi-choice questions across 57 domains to evaluate a model's generalized language understanding capabilities.

#### Processing Steps  
1. **Input Data**:  
   - Question: *"What is the capital of France?"*  
   - Choices: (A) Madrid, (B) Paris, (C) Rome, (D) Berlin  
   - Correct Answer: (B) Paris  

2. **Model Processing**:  
   - DeepSeek-V2 processes the question and choices using its Mixture-of-Experts (MoE) architecture to activate the relevant expert layers.  
   - Multi-head Latent Attention (MLA) compresses the Key-Value cache, enabling efficient analysis of relationships between the choices.  

3. **Prediction Result**:  
   - DeepSeek-V2 selects (B) Paris by understanding the context and relationships among the choices.  
   - On MMLU, DeepSeek-V2 achieved 78.5% accuracy, significantly outperforming DeepSeek 67B (71.3%).

---

### Specific Example: HumanEval Dataset
#### Dataset Description  
HumanEval evaluates a model's ability to write correct code for given coding problems.

#### Processing Steps  
1. **Input Data**:  
   - Problem: *"Write a function that returns the sum of two integers."*  

2. **Model Processing**:  
   - DeepSeek-V2 understands the problem in natural language and utilizes code data learned through Supervised Fine-Tuning (SFT) to generate the correct answer.  
   - During inference, a code-quality reward model trained via Reinforcement Learning (RL) evaluates the output.  

3. **Prediction Result**:  
   - Generated Code:
     ```python
     def add(a, b):
         return a + b
     ```
   - DeepSeek-V2 achieved 81.1% accuracy on HumanEval, surpassing Mixtral 8x22B (75.0%).


<br/>  
# 요약   



DeepSeek-V2는 Multi-head Latent Attention(MLA)과 DeepSeekMoE라는 두 가지 혁신적인 방법론을 도입하여 성능과 효율성을 극대화했습니다. MLA는 Key-Value(KV) 캐시를 저차원으로 압축하여 활성화 파라미터 수를 줄이는 동시에, 기존 Multi-Head Attention(MHA)의 성능을 유지했습니다. DeepSeekMoE는 Mixture-of-Experts(MoE) 구조를 활용해 전문가의 세분화를 강화하고, 훈련 비용을 42.5% 절감했습니다. MMLU 데이터셋에서 DeepSeek-V2는 DeepSeek 67B의 정확도 71.3%에서 78.5%로 향상되었으며, HumanEval 데이터셋에서는 정확도 81.1%로 코딩 문제 해결 능력을 크게 개선했습니다. 이처럼 DeepSeek-V2는 저차원화, 훈련 최적화, 성능 향상을 동시에 달성하며, 다양한 벤치마크에서 최첨단 성능을 기록했습니다.

---


DeepSeek-V2 employs two innovative methodologies: Multi-head Latent Attention (MLA) and DeepSeekMoE, to maximize performance and efficiency. MLA reduces the Key-Value (KV) cache to a low-dimensional space, cutting active parameter counts while maintaining the performance of standard Multi-Head Attention (MHA). DeepSeekMoE leverages Mixture-of-Experts (MoE) architecture to enhance expert specialization, achieving a 42.5% reduction in training costs. On the MMLU dataset, DeepSeek-V2 improved accuracy from 71.3% (DeepSeek 67B) to 78.5%, and on the HumanEval dataset, it achieved 81.1% accuracy, significantly enhancing code problem-solving capabilities. In summary, DeepSeek-V2 excels in dimensionality reduction, training optimization, and performance enhancement, achieving state-of-the-art results across various benchmarks.


<br/>  
# 기타  



MLA(Multi-head Latent Attention)는 Key-Value(KV) 캐시를 저차원 벡터로 압축하는 기술을 사용하여 파라미터 개수를 줄이고도 성능을 유지할 수 있었습니다.

### 어떻게 파라미터 개수를 줄였는가?
MLA는 다음 방식으로 파라미터 개수를 줄였습니다:
1. **저차원 압축(joint compression)**  
   - 기존 Multi-Head Attention(MHA)은 토큰별로 Key와 Value를 고차원 벡터로 저장하여 많은 메모리를 소모합니다.  
   - MLA는 Key와 Value를 **공동(latent)** 벡터로 압축하여, 차원을 크게 줄입니다.  
   - 이를 통해, 저장해야 할 KV 캐시 크기가 대폭 감소했습니다. (예: \(d_c + d_R \ll n_h \cdot d_h\)).

2. **파라미터 재활용**  
   - MLA는 Key와 Value를 생성하는 과정에서, 모델의 학습된 행렬을 효율적으로 재활용하여 추가 파라미터 생성을 억제합니다.  
   - 이는 MHA와 비교했을 때, 성능 저하 없이 필요한 연산량과 파라미터 개수를 줄이는 데 기여했습니다.

### 왜 성능이 유지될 수 있었는가?
- MLA는 Key-Value 캐시를 단순히 줄이는 것이 아니라, 이를 **저차원에서 효율적으로 학습 및 사용**할 수 있도록 설계되었습니다.
- **로터리 위치 임베딩(RoPE)**와 결합된 새로운 구조를 통해, 모델이 필요한 문맥 정보를 충분히 보존하며 추론 효율성을 극대화했습니다.
- 기존 GQA(Grouped-Query Attention) 및 MQA(Multi-Query Attention)가 성능 저하를 겪었던 반면, MLA는 **저차원 압축과 성능 균형**을 유지합니다.

---

### 결론적으로
MLA는 Key와 Value를 저차원으로 압축하여 모델의 활성화 파라미터 개수를 줄였습니다. 이 과정에서 저장 공간과 연산량은 감소했지만, 설계된 압축 메커니즘 덕분에 모델이 여전히 고품질의 성능을 유지할 수 있었습니다.

---


MLA (Multi-head Latent Attention) uses low-rank compression of Key-Value (KV) caches, reducing the parameter count while maintaining model performance.

### How Did It Reduce Parameters?
MLA achieves parameter reduction through the following mechanisms:
1. **Low-Rank Compression (Joint Compression)**  
   - Standard Multi-Head Attention (MHA) stores Key and Value vectors as high-dimensional representations for each token, consuming significant memory.  
   - MLA compresses the Key and Value vectors into **joint latent vectors**, significantly reducing their dimensionality.  
   - This drastically decreases the size of the KV cache (e.g., \(d_c + d_R \ll n_h \cdot d_h\)).

2. **Parameter Reuse**  
   - MLA efficiently reuses learned matrices to generate Key and Value, avoiding the need to introduce extra parameters.  
   - Compared to MHA, this design reduces computational complexity and parameter count without compromising performance.

### Why Was Performance Maintained?
- MLA is not just about reducing the KV cache size; it is designed to **learn and utilize contextual information efficiently** in a low-dimensional space.  
- Coupled with **Rotary Position Embedding (RoPE)**, MLA ensures that contextual and positional information is preserved, even with reduced dimensions.  
- While other techniques like GQA (Grouped-Query Attention) and MQA (Multi-Query Attention) suffered from performance trade-offs, MLA strikes a balance between **compression and effectiveness**.

---

### Conclusion
MLA compresses Key and Value into low-dimensional vectors, significantly reducing the active parameters in the model. Despite these reductions, MLA’s efficient compression mechanism allows it to maintain high-quality performance.


<br/>
# refer format:     



@article{DeepSeek2024,
    author = {DeepSeek-AI},
    title = {DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
    journal = {arXiv preprint arXiv:2405.04434},
    year = {2024},
    url = {https://github.com/deepseek-ai/DeepSeek-V2}
}



DeepSeek-AI. “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.” arXiv preprint arXiv:2405.04434, 2024.




