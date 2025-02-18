---
layout: post
title:  "[2025]Large Language Diffusion Models"  
date:   2025-02-18 10:32:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


MLM의 마스킹을 동적으로(개수 비율이 마구 바뀌게..->이게 디퓨전 느낌)   


짧은 요약(Abstract) :    




대부분의 대형 언어 모델(LLM)은 자회귀 모델(Autoregressive Models, ARM)을 기반으로 개발되었지만, 본 연구에서는 이를 대체할 수 있는 새로운 접근법인 **LLaDA**를 소개합니다. LLaDA는 데이터 마스킹을 활용한 확산 모델(Diffusion Model)로, 사전 학습 및 지도학습 방식의 미세 조정을 거쳐 개발되었습니다. 모델은 순방향 마스킹 과정과 역방향 복원 과정을 통해 확률적 생성 모델링을 수행하며, 기존 ARM 기반 모델과 달리 양방향 정보를 활용하여 더 효과적인 확률 추론이 가능합니다.

LLaDA는 다양한 벤치마크 테스트에서 기존 ARM 모델과 비교하여 높은 확장성을 보이며, 특히 **LLaDA 8B 모델**은 LLaMA3 8B와 동등한 수준의 성능을 보였습니다. 또한, LLaDA는 **명령어 수행(instruction-following) 능력과 문맥 학습(in-context learning)에서 뛰어난 성능**을 입증하였으며, 심지어 **GPT-4o를 능가하는 역순 시(poem reversal completion) 생성 능력**도 보였습니다.

이 연구는 대형 언어 모델이 반드시 ARM 방식에 의존해야 한다는 기존 가설을 재검토하고, 확산 모델이 충분히 강력한 대안이 될 수 있음을 보여줍니다.

---


Autoregressive models (ARMs) are widely regarded as the cornerstone of large language models (LLMs). We challenge this notion by introducing **LLaDA**, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA models distributions through a forward data masking process and a reverse process, parameterized by a vanilla Transformer to predict masked tokens. By optimizing a likelihood bound, it provides a principled generative approach for probabilistic inference.

Across extensive benchmarks, LLaDA demonstrates strong scalability, outperforming our self-constructed ARM baselines. Remarkably, **LLaDA 8B is competitive with strong LLMs like LLaMA3 8B** in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multi-turn dialogue. Moreover, **LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task**. 

Our findings establish diffusion models as a viable and promising alternative to ARMs, challenging the assumption that key LLM capabilities are inherently tied to ARMs.



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





#### **1. 방법론 (Method)**  
LLaDA는 기존의 자회귀 모델(ARM)과 달리 **마스킹 기반 확산 모델(Masked Diffusion Model, MDM)**을 활용하여 대형 언어 모델을 학습합니다. 모델은 다음과 같은 과정으로 작동합니다.  

1. **순방향 과정 (Forward Process)**: 입력 데이터에 대해 특정 비율(0~1)로 랜덤하게 토큰을 마스킹합니다.  
2. **역방향 과정 (Reverse Process)**: 마스킹된 토큰을 예측하여 원래 문장을 복원하는 과정을 학습합니다.  
3. **손실 함수 (Loss Function)**: 마스킹된 부분만을 대상으로 크로스 엔트로피 손실을 계산하여 모델을 최적화합니다.  
4. **샘플링 (Sampling)**: 학습된 마스킹 예측기를 활용하여, 문장을 점진적으로 복원하는 방식으로 새로운 문장을 생성합니다.  

기존 BERT와 같은 마스킹 기반 모델과 차별화되는 점은 **마스킹 비율이 고정되지 않고 동적으로 변화**한다는 점입니다. 또한, ARM 모델처럼 왼쪽에서 오른쪽으로 순차적으로 생성하는 방식이 아니라, **양방향 문맥을 활용할 수 있는 모델**로 학습됩니다.  

---

#### **2. 모델과 학습 데이터 (Model and Training Data)**  

- **모델 구조**: LLaDA는 트랜스포머(Transformer) 구조를 기반으로 합니다. 기존의 ARM 모델들과 달리, LLaDA는 **원활한 양방향 예측을 위해 인과적 마스킹(causal masking)을 사용하지 않습니다**.  
- **파라미터 크기**: LLaDA는 1B(10억) 및 8B(80억) 파라미터 모델로 학습되었습니다.  
- **학습 데이터**:
  - 총 **2.3조(Trillion) 개의 토큰**을 사용하여 사전 학습(pre-training)을 수행함.
  - 학습 데이터는 **온라인 코퍼스(corpus)에서 수집**되었으며, 저품질 데이터를 필터링하는 과정을 거침.
  - 데이터셋에는 **일반 텍스트뿐만 아니라 코드, 수학, 다국어 데이터**도 포함됨.
  - 학습된 토크나이저(tokenizer)는 기존 LLaMA 모델과 일부 차이가 있으며, LLaDA의 학습 데이터에 맞춰 최적화됨.  

LLaDA 8B 모델의 학습에는 **H800 GPU를 사용하여 총 13만 GPU 시간**이 소요되었습니다.  
사전 학습 이후에는 **4.5백만 개의 데이터 쌍**을 사용하여 지도 학습(SFT, Supervised Fine-Tuning)을 진행하였으며, 명령어 수행(instruction-following)과 같은 추가적인 능력을 강화하였습니다.  

---



#### **1. Method**  
Unlike traditional **Autoregressive Models (ARMs)**, LLaDA is a **Masked Diffusion Model (MDM)** that learns via a forward masking and reverse prediction process. The model operates as follows:  

1. **Forward Process**: Randomly masks tokens in an input sequence based on a probability between 0 and 1.  
2. **Reverse Process**: Recovers the original sequence by predicting masked tokens.  
3. **Loss Function**: Optimizes via cross-entropy loss, calculated only on masked tokens.  
4. **Sampling**: Generates new text by iteratively refining masked predictions.  

A key difference from traditional masked language models (e.g., BERT) is that **LLaDA does not use a fixed masking ratio but dynamically varies it**. Unlike ARM models, which generate text sequentially, LLaDA **leverages bidirectional context** for prediction.  

---

#### **2. Model and Training Data**  

- **Model Architecture**: LLaDA is based on the **Transformer** architecture. Unlike standard ARMs, it **does not use causal masking**, enabling bidirectional token predictions.  
- **Model Size**: LLaDA was trained in two configurations: **1B (1 billion) and 8B (8 billion) parameters**.  
- **Training Data**:
  - Pre-trained on **2.3 trillion tokens**.  
  - Data was collected from **online corpora**, with low-quality content filtered.  
  - Includes **general text, code, mathematical data, and multilingual data**.  
  - Tokenizer was adapted specifically for LLaDA, differing slightly from LLaMA models.  

The **LLaDA 8B model required 130,000 H800 GPU hours** for pre-training.  
For **Supervised Fine-Tuning (SFT)**, **4.5 million data pairs** were used to enhance instruction-following capabilities.


   
 
<br/>
# Results  




#### **1. 결과 (Results)**  
LLaDA 모델은 다양한 벤치마크에서 기존 **자회귀 모델(ARM)**을 기반으로 한 대형 언어 모델(LLM)과 비교하여 높은 성능을 보였습니다. 특히, **LLaDA 8B 모델은 LLaMA3 8B 모델과 경쟁 가능한 성능을 보이며, 일부 작업에서는 뛰어난 성과**를 기록했습니다.  

LLaDA의 주요 성과는 다음과 같습니다:  

1. **확장성(Scalability)**  
   - LLaDA는 ARM 모델과 유사한 수준으로 확장 가능하며, **MMLU, GSM8K 등 주요 벤치마크에서 자체 구축한 ARM 모델을 능가**하는 성과를 보였습니다.  
   - FLOPs(연산량) 증가에 따른 성능 향상이 ARM 모델과 비교할 때 동등하거나 더 높은 확장성을 보임.  

2. **문맥 학습 능력 (In-Context Learning)**  
   - LLaDA 8B 모델은 **LLaMA2 7B 모델보다 모든 벤치마크에서 더 나은 성능**을 보였으며, LLaMA3 8B 모델과 거의 동등한 결과를 기록.  
   - 특히, **15개의 대표적인 Zero/Few-shot 학습 태스크**에서 LLaMA2 7B보다 높은 성능을 기록함.  

3. **명령어 수행 능력 (Instruction-Following)**  
   - 지도 학습(SFT) 후, LLaDA는 **다중 턴 대화(multi-turn dialogue)와 같은 태스크에서 매우 우수한 결과**를 보였음.  
   - GPT-4o를 능가하는 **역순 시(poem reversal completion) 생성 능력**을 보임.  

4. **역순 추론 (Reversal Reasoning) 개선**  
   - GPT-4o 및 Qwen2.5 7B Instruct와 비교하여 **순방향(Forward) 및 역순(Reverse) 태스크 간의 성능 차이가 크지 않음**.  
   - 기존 ARM 모델이 가진 **순차적 생성 방식의 한계를 극복**하여 역순 추론에서도 균형 잡힌 성능을 보임.  

---

#### **2. 경쟁 모델 및 테스트 데이터 (Competitive Models and Test Data)**  

**① 경쟁 모델**  
LLaDA는 아래와 같은 대표적인 **자회귀(Autoregressive, AR) 모델들과 성능 비교**를 수행했습니다.  

- **LLaMA2 7B / LLaMA3 8B** (Meta AI)  
- **Qwen2 7B / Qwen2.5 7B** (Alibaba)  
- **Mistral 7B**  
- **DeepSeek 7B**  
- **GPT-4o** (OpenAI)  

대부분의 비교 모델은 **자회귀 모델(ARMs)이며, LLaDA는 유일한 확산 모델(Diffusion Model)**로서 평가되었습니다.  

**② 테스트 데이터 (Benchmark Datasets)**  
LLaDA의 성능을 평가하기 위해 다음과 같은 대표적인 15개 벤치마크를 사용했습니다.  

- **일반 과제(General Tasks)**
  - MMLU (Massive Multitask Language Understanding)  
  - BBH (Big-Bench Hard)  
  - ARC-C (AI2 Reasoning Challenge - Challenge Set)  
  - HellaSwag  
  - TruthfulQA  
  - WinoGrande  
  - PIQA  

- **수학 및 과학(Mathematics & Science)**  
  - GSM8K (Grade School Math 8K)  
  - Math (Mathematical Reasoning)  
  - GPQA (Graduate-Level Google-Proof Q&A)  

- **코드(Code Generation)**
  - HumanEval  
  - MBPP (Mostly Basic Python Programming)  

- **중국어(Chinese)**
  - CMMLU (Chinese MMLU)  
  - C-Eval (Chinese Evaluation Dataset)  

각 태스크별로 Zero-shot 및 Few-shot 평가를 진행했으며, LLaDA는 **MMLU, GSM8K, CMMLU 등의 여러 벤치마크에서 경쟁 모델 대비 우수한 성과를 보였습니다**.  

---



#### **1. Results**  
LLaDA demonstrated **competitive or superior performance** compared to traditional **autoregressive models (ARMs)** across multiple benchmarks. In particular, **LLaDA 8B matched or even outperformed LLaMA3 8B** on several tasks.  

Key findings include:  

1. **Scalability**  
   - LLaDA exhibits **scalability comparable to ARMs**, with **MMLU, GSM8K, and other benchmarks surpassing self-constructed ARM baselines**.  
   - The **performance increase with computational FLOPs matches or exceeds ARM models**.  

2. **In-Context Learning**  
   - LLaDA **outperforms LLaMA2 7B on all benchmarks** and achieves results close to **LLaMA3 8B**.  
   - The model **excels in Zero/Few-shot learning tasks** across **15 major benchmarks**.  

3. **Instruction-Following**  
   - After **Supervised Fine-Tuning (SFT)**, LLaDA exhibits strong **multi-turn dialogue** capabilities.  
   - **Outperforms GPT-4o in the reversal poem completion task**.  

4. **Reversal Reasoning**  
   - Compared to GPT-4o and Qwen2.5 7B Instruct, **LLaDA maintains balanced performance between forward and reverse tasks**.  
   - Unlike ARMs, **LLaDA does not suffer from the sequential generation bottleneck**, allowing it to perform well in **reversal reasoning**.  

---

#### **2. Competitive Models and Test Data**  

**① Competitive Models**  
LLaDA was evaluated against major **Autoregressive (AR) models**, including:  

- **LLaMA2 7B / LLaMA3 8B** (Meta AI)  
- **Qwen2 7B / Qwen2.5 7B** (Alibaba)  
- **Mistral 7B**  
- **DeepSeek 7B**  
- **GPT-4o** (OpenAI)  

Among these, **LLaDA is the only diffusion-based model**, competing against purely **autoregressive** baselines.  

**② Test Datasets**  
LLaDA was benchmarked using **15 standard datasets** across multiple domains:  

- **General Tasks:**
  - MMLU (Massive Multitask Language Understanding)  
  - BBH (Big-Bench Hard)  
  - ARC-C (AI2 Reasoning Challenge - Challenge Set)  
  - HellaSwag  
  - TruthfulQA  
  - WinoGrande  
  - PIQA  

- **Mathematics & Science:**  
  - GSM8K (Grade School Math 8K)  
  - Math (Mathematical Reasoning)  
  - GPQA (Graduate-Level Google-Proof Q&A)  

- **Code Generation:**
  - HumanEval  
  - MBPP (Mostly Basic Python Programming)  

- **Chinese Language Tasks:**
  - CMMLU (Chinese MMLU)  
  - C-Eval (Chinese Evaluation Dataset)  

Each task was evaluated under **Zero-shot and Few-shot settings**, and **LLaDA outperformed competitive models on key benchmarks such as MMLU, GSM8K, and CMMLU**.



<br/>
# 예제  





#### **1. 트레이닝 데이터 입력/출력 예제 (Training Data Input/Output Example)**  
- **입력(Input)**  
  ```
  "The capital of France is [MASK]."
  ```
- **출력(Output, 모델 예측)**  
  ```
  "The capital of France is Paris."
  ```

---

#### **2. 테스트 데이터 입력/출력 예제 (Test Data Input/Output Example)**  
- **입력(Input, Zero-shot Prompt)**  
  ```
  "Translate the sentence into French: 'Hello, how are you?'"
  ```
- **출력(Output, LLaDA 예측)**  
  ```
  "Bonjour, comment ça va?"
  ```

---

#### **3. 경쟁 모델과 비교 (Competitive Model Comparison)**  

- **질문 (Question)**  
  ```
  "What is the largest planet in the solar system?"
  ```
- **LLaDA 8B 응답**  
  ```
  "Jupiter is the largest planet in the solar system."
  ```
- **LLaMA3 8B 응답**  
  ```
  "The largest planet in our solar system is Jupiter."
  ```
- **GPT-4o 응답**  
  ```
  "Jupiter, the gas giant, is the largest planet in the solar system."
  ```
*(LLaDA의 응답이 경쟁 모델과 유사한 수준임을 확인 가능)*  

---


#### **1. Training Data Input/Output Example**  
- **Input**  
  ```
  "The capital of France is [MASK]."
  ```
- **Output (Model Prediction)**  
  ```
  "The capital of France is Paris."
  ```

---

#### **2. Test Data Input/Output Example**  
- **Input (Zero-shot Prompt)**  
  ```
  "Translate the sentence into French: 'Hello, how are you?'"
  ```
- **Output (LLaDA Prediction)**  
  ```
  "Bonjour, comment ça va?"
  ```

---

#### **3. Competitive Model Comparison**  

- **Question**  
  ```
  "What is the largest planet in the solar system?"
  ```
- **LLaDA 8B Response**  
  ```
  "Jupiter is the largest planet in the solar system."
  ```
- **LLaMA3 8B Response**  
  ```
  "The largest planet in our solar system is Jupiter."
  ```
- **GPT-4o Response**  
  ```
  "Jupiter, the gas giant, is the largest planet in the solar system."
  ```
*(LLaDA's response is comparable to leading models.)*


<br/>  
# 요약   



LLaDA는 기존 자회귀 모델(ARM)과 달리 마스킹 기반 확산 모델(Masked Diffusion Model, MDM)을 활용하여 학습된다. 순방향 마스킹을 통해 일부 토큰을 제거하고, 역방향 과정에서 이를 복원하는 방식으로 모델이 훈련된다. 이를 통해 기존의 단방향 생성 방식이 아닌 양방향 문맥을 활용한 자연어 모델링이 가능하다.  

LLaDA는 다양한 벤치마크에서 LLaMA3 8B와 유사한 성능을 보였으며, 일부 태스크에서는 우수한 결과를 기록했다. 특히 문맥 학습(in-context learning)과 명령어 수행(instruction-following) 능력이 뛰어나며, GPT-4o보다 역순 생성(reverse completion)에서 더 좋은 성능을 보였다. FLOPs 대비 확장성도 경쟁 모델과 동등하거나 더 우수한 것으로 나타났다.  

예제 비교 결과, LLaDA는 프랑스 수도 예측, 번역, 질문 응답 등의 태스크에서 경쟁 모델과 비슷한 성능을 보였다. "파리"를 정확히 예측하고, 영어 문장을 프랑스어로 자연스럽게 번역하며, 태양계에서 가장 큰 행성이 목성임을 정확히 답변했다. 이러한 결과는 확산 모델이 자회귀 모델의 대안으로 충분한 가능성을 가진다는 것을 시사한다.  

---


LLaDA differs from traditional autoregressive models (ARMs) by adopting a **Masked Diffusion Model (MDM)**. It removes tokens in a **forward masking** step and reconstructs them in a **reverse process**, allowing the model to leverage bidirectional context rather than relying on sequential generation.  

LLaDA performs competitively with **LLaMA3 8B** on multiple benchmarks and surpasses it in certain tasks. It excels in **in-context learning** and **instruction-following**, even **outperforming GPT-4o** in reverse completion tasks. Additionally, it demonstrates scalability on par with or superior to ARMs in terms of computational efficiency.  

In example comparisons, LLaDA correctly predicts **"Paris"** as the capital of France, translates **"Hello, how are you?"** into **"Bonjour, comment ça va?"**, and identifies **Jupiter** as the largest planet. These results indicate that diffusion models are a viable alternative to autoregressive architectures for large language models.


<br/>  
# 기타  



논문에는 LLaDA 모델의 성능을 시각적으로 표현한 다양한 **피규어(Figure)와 테이블(Table)**이 포함되어 있다.  

#### **1. 모델 성능 비교 (Figure 1 & Table 1, 2)**  
- **Figure 1**: LLaDA와 기존 자회귀 모델(ARM)의 성능을 FLOPs(연산량) 대비 비교한 그래프이다. LLaDA는 FLOPs 증가에 따라 성능이 안정적으로 상승하며, ARM 모델과 유사한 확장성을 보인다.  
- **Table 1**: LLaDA 8B 모델과 LLaMA3 8B, Qwen2.5 7B, Mistral 7B 등의 성능을 벤치마크별로 정리하였다. LLaDA는 주요 자연어 이해(NLU) 및 코드 생성(Code Generation) 태스크에서 경쟁 모델과 비슷하거나 더 나은 성능을 보였다.  
- **Table 2**: LLaDA의 문맥 학습(in-context learning) 성능을 보여주는 테이블로, LLaDA 8B가 LLaMA2 7B보다 우수하고 LLaMA3 8B와 유사한 성능을 보인다.  

#### **2. 명령어 수행 및 역순 생성 성능 (Figure 2 & Table 3)**  
- **Figure 2**: 명령어 수행 태스크에서 LLaDA가 지도 학습(SFT) 후 얼마나 향상되는지를 시각적으로 나타낸다.  
- **Table 3**: GPT-4o와 비교한 역순 생성(poem reversal completion) 성능을 보여준다. LLaDA는 GPT-4o보다 높은 정확도를 기록하며, ARM 모델이 가진 순차적 생성의 한계를 극복했음을 나타낸다.  

#### **3. 예제 비교 (Table 4)**  
- **Table 4**: LLaDA 8B, LLaMA3 8B, GPT-4o의 실제 출력 비교 테이블이다. "태양계에서 가장 큰 행성은?" 같은 질문에서 모든 모델이 "목성(Jupiter)"을 정답으로 도출했으며, 번역 및 일반 지식 문항에서도 LLaDA가 경쟁 모델과 유사한 응답을 생성했다.  

이러한 결과는 확산 모델(Diffusion Model)이 기존 자회귀 모델(ARM)과 유사한 성능을 보일 뿐만 아니라, 일부 작업에서는 더 나은 결과를 도출할 수 있음을 시사한다.  

---



The paper includes multiple **figures and tables** to illustrate LLaDA's performance.  

#### **1. Model Performance Comparison (Figure 1 & Table 1, 2)**  
- **Figure 1**: A graph comparing LLaDA’s performance to **autoregressive models (ARMs)** in terms of FLOPs (computational cost). LLaDA scales efficiently and performs on par with ARMs.  
- **Table 1**: A benchmark performance comparison of **LLaDA 8B** with **LLaMA3 8B, Qwen2.5 7B, Mistral 7B**, etc. LLaDA performs competitively in natural language understanding (NLU) and code generation tasks.  
- **Table 2**: Evaluates LLaDA’s **in-context learning** capabilities, showing **LLaDA 8B surpasses LLaMA2 7B and is comparable to LLaMA3 8B**.  

#### **2. Instruction Following & Reversal Performance (Figure 2 & Table 3)**  
- **Figure 2**: Demonstrates how LLaDA improves after **supervised fine-tuning (SFT)** in instruction-following tasks.  
- **Table 3**: Compares **GPT-4o** and LLaDA on **poem reversal completion** tasks. LLaDA outperforms **GPT-4o**, indicating that it overcomes ARMs’ sequential generation limitations.  

#### **3. Example Comparison (Table 4)**  
- **Table 4**: A direct comparison of responses from **LLaDA 8B, LLaMA3 8B, and GPT-4o**. For example, when asked **"What is the largest planet in the solar system?"**, all models correctly answer **"Jupiter"**, and LLaDA produces translations and factual responses similar to its competitors.  

These results suggest that **diffusion models can serve as strong alternatives to autoregressive models (ARMs), offering competitive or even superior performance in certain tasks**.


<br/>
# refer format:     


@article{nie2025llada,
  author    = {Shen Nie and Fengqi Zhu and Zebin You and Xiaolu Zhang and Jingyang Ou and Jun Hu and Jun Zhou and Yankai Lin and Ji-Rong Wen and Chongxuan Li},
  title     = {Large Language Diffusion Models},
  journal   = {arXiv preprint},
  volume    = {arXiv:2502.09992},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.09992}
}





Nie, Shen, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. **"Large Language Diffusion Models."** *arXiv preprint* arXiv:2502.09992 (2025). [https://arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992).

