---
layout: post
title:  "[2025]Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems"  
date:   2025-03-01 09:52:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    



이 연구는 인공지능(AI) 시스템의 수학적 문제 해결 능력을 평가하면서 기존 연구가 정답의 정확성에 초점을 맞춰왔던 점을 지적합니다. 연구진은 단순히 올바른 답을 생성하는 것뿐만 아니라, AI가 인간과 함께 새로운 해결책을 개발하는 능력을 가져야 한다고 주장합니다. 이에 따라 **CREATIVEMATH**라는 새로운 벤치마크를 도입하여, 중학교 수준부터 올림피아드 수준까지 다양한 난이도의 문제를 포함하고, 이미 알려진 해결책을 제공한 후 AI가 독창적인 해결책을 제안할 수 있는지를 평가합니다. 실험 결과, AI가 일반적인 수학 문제 해결에는 강하지만 창의적 문제 해결 능력에는 편차가 있는 것으로 나타났습니다. 특히 **Gemini-1.5-Pro 모델**이 다른 AI 모델보다 혁신적인 해결책을 제시하는 능력이 뛰어났습니다. 이 연구는 AI의 창의성 평가를 위한 새로운 기준을 마련하며, AI가 수학적 혁신을 촉진하는데 있어 강점과 한계를 탐구하고 향후 발전의 방향성을 제시합니다.

---


The mathematical capabilities of AI systems are complex and multifaceted. Most existing research has predominantly focused on the correctness of AI-generated solutions to mathematical problems. In this work, we argue that beyond producing correct answers, AI systems should also be capable of, or assist humans in, developing novel solutions to mathematical challenges. This study explores the creative potential of Large Language Models (LLMs) in mathematical reasoning, an aspect that has received limited attention in prior research. We introduce a novel framework and benchmark, **CREATIVEMATH**, which encompasses problems ranging from middle school curricula to Olympic-level competitions, designed to assess LLMs’ ability to propose innovative solutions after some known solutions have been provided. Our experiments demonstrate that, while LLMs perform well on standard mathematical tasks, their capacity for creative problem-solving varies considerably. Notably, the **Gemini-1.5-Pro model** outperformed other LLMs in generating novel solutions. This research opens a new frontier in evaluating AI creativity, shedding light on both the strengths and limitations of LLMs in fostering mathematical innovation, and setting the stage for future developments in AI-assisted mathematical discovery.



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



이 연구에서는 **CREATIVEMATH** 데이터셋을 이용하여 **대형 언어 모델(LLM)**이 수학 문제 해결에서 얼마나 창의적인 해결책을 제시할 수 있는지를 평가하는 프레임워크를 개발했습니다. 평가 방법은 크게 **네 가지 단계**로 구성됩니다:  

1. **새로운 해결책 생성 (Novel Solution Generation)**  
   - LLM이 주어진 수학 문제에 대해 기존 해결책(k개 제공)과는 다른 새로운 해결책을 생성하도록 유도합니다.  
   - 문제는 **중학교 수준부터 국제 수학 올림피아드(IMO) 수준까지 포함**되어 있으며, 다양한 난이도를 반영합니다.  
   - 해결책의 다양성을 보장하기 위해 **알고리즘적 차이, 중간 과정의 변화, 가정의 차이, 일반화 가능성, 복잡성 차이** 등의 기준을 적용합니다.  

2. **정확성 평가 (Correctness Evaluation)**  
   - 생성된 해결책이 수학적으로 올바른지 확인합니다.  
   - GPT-4, Claude-3.5 Sonnet, Gemini-1.5-Pro 모델을 활용하여 교차 검증합니다.  
   - **모든 모델이 정답으로 판단한 해결책만 다음 단계로 진행**됩니다.  

3. **대략적인 창의성 평가 (Coarse-Grained Novelty Assessment)**  
   - 제공된 k개의 기존 해결책과 비교하여 새로운 해결책이 독창적인지를 평가합니다.  
   - 해결책이 기존 해결책과 다르면 창의적인 것으로 판단하고 다음 단계로 진행합니다.  

4. **정밀 창의성 평가 (Fine-Grained Novelty Assessment)**  
   - 인간이 만든 모든 기존 해결책(n개)과 비교하여 완전히 새로운 해결책인지 확인합니다.  
   - 창의적인 해결책을 **Novel-Known (이미 존재하는 해결책과 유사)** 또는 **Novel-Unknown (완전히 새로운 해결책)** 으로 분류합니다.  

**사용한 모델 및 실험 환경**  
- **폐쇄형 LLMs**: GPT-4o, Claude-3-Opus, Gemini-1.5-Pro  
- **오픈소스 LLMs**: Llama-3-70B, Qwen1.5-72B, Yi-1.5-34B, Mixtral-8x22B, DeepSeek-V2  
- **수학 특화 모델**: DeepSeek-Math-7B-RL, Internlm2-Math-20B  
- 실험 환경: NVIDIA A100 (80G) GPU 1~4개 사용, `Hugging Face` 라이브러리를 활용하여 모델 실행  
- 폐쇄형 LLMs 및 DeepSeek-V2는 API를 이용해 평가  
- 실험 반복성을 보장하기 위해 **Greedy Decoding (탐욕적 디코딩) 사용, 최대 1024개의 토큰 생성 허용**  

**사용한 데이터셋**  
- **CREATIVEMATH 데이터셋**:  
  - **8개의 미국 주요 수학 경진대회(AIME, AMC, IMO 등) 문제 포함**  
  - **6,469개의 수학 문제 및 14,223개의 해결책**  
  - 각 문제에 대해 기존 해결책이 최소 1개, 최대 5개(k=1~5)까지 제공됨  
  - 데이터 정제 과정에서 **LaTeX 코드 및 HTML에서 수식을 추출, 중복 해결책 제거**  
  - 문제 난이도는 **Llama-3-70B 모델을 이용해 자동 태깅**  

**평가 지표**  
- **Correctness (C, 정확성 비율)**: 모델이 올바른 해결책을 생성한 비율  
- **Novelty (N, 창의성 비율)**: 기존 k개의 해결책과 다른 해결책을 만든 비율  
- **Novel-Unknown (Nu, 완전 창의적 해결책 비율)**: 인간이 만든 해결책과도 다른 해결책을 만든 비율  
- **Novelty-to-Correctness Ratio (N/C)**: 창의적인 해결책이 정확한 해결책 중 몇 %인지 측정  
- **Novel-Unknown-to-Novelty Ratio (Nu/N)**: 창의적인 해결책 중 완전히 새로운 해결책이 차지하는 비율  

이 실험을 통해, Gemini-1.5-Pro가 **가장 창의적인 해결책을 제시하는 모델**임이 밝혀졌으며, 일반적인 수학 문제 해결 능력과 창의적 문제 해결 능력 간에는 차이가 있음이 확인되었습니다.  

---


This study introduces a **new evaluation framework** using the **CREATIVEMATH dataset** to assess the creative problem-solving capabilities of **Large Language Models (LLMs)** in mathematical reasoning. The evaluation is conducted through **four key stages**:  

1. **Novel Solution Generation**  
   - LLMs generate new solutions to given mathematical problems after being provided with k known solutions.  
   - Problems span **middle school to International Mathematical Olympiad (IMO) level**, ensuring a wide range of difficulty.  
   - **Creativity is assessed based on methodological differences, intermediate step variations, assumptions, generalization, and complexity.**  

2. **Correctness Evaluation**  
   - Each generated solution is checked for mathematical correctness.  
   - **GPT-4, Claude-3.5 Sonnet, and Gemini-1.5-Pro** serve as LLM-based evaluators.  
   - **Only unanimously verified correct solutions proceed to the next stage.**  

3. **Coarse-Grained Novelty Assessment**  
   - The solution is compared against k reference solutions to determine if it is distinct.  
   - If unique, it is considered novel and moves to the fine-grained assessment stage.  

4. **Fine-Grained Novelty Assessment**  
   - The solution is compared against **all** n human-generated solutions.  
   - Novel solutions are categorized as either **Novel-Known (similar to an existing human solution)** or **Novel-Unknown (entirely new).**  

**Models and Experimental Setup**  
- **Closed-source LLMs**: GPT-4o, Claude-3-Opus, Gemini-1.5-Pro  
- **Open-source LLMs**: Llama-3-70B, Qwen1.5-72B, Yi-1.5-34B, Mixtral-8x22B, DeepSeek-V2  
- **Math-specialized LLMs**: DeepSeek-Math-7B-RL, Internlm2-Math-20B  
- **Hardware**: 1-4 NVIDIA A100 (80G) GPUs, `Hugging Face` library for open-source models  
- **API-based evaluation** for closed-source LLMs and DeepSeek-V2  
- To ensure reproducibility, **greedy decoding was used, with a max token generation limit of 1024**  

**Dataset: CREATIVEMATH**  
- **Derived from 8 major U.S. math competitions (AIME, AMC, IMO, etc.)**  
- **6,469 math problems and 14,223 solutions**  
- Each problem includes **1 to 5 prior reference solutions (k=1~5)**  
- **Data cleaning process**: Extracted mathematical expressions from LaTeX and HTML, removed duplicate solutions  
- **Problem difficulty** was labeled using **Llama-3-70B**  

**Evaluation Metrics**  
- **Correctness (C)**: Percentage of correct solutions  
- **Novelty (N)**: Percentage of correct solutions that differ from k known solutions  
- **Novel-Unknown (Nu)**: Percentage of solutions that differ from **all human-created solutions**  
- **Novelty-to-Correctness Ratio (N/C)**: Percentage of correct solutions that are novel  
- **Novel-Unknown-to-Novelty Ratio (Nu/N)**: Percentage of novel solutions that are entirely new  

The experiments revealed that **Gemini-1.5-Pro was the best-performing model in generating novel solutions**. Additionally, the study highlights that **mathematical accuracy does not necessarily correlate with creativity**, emphasizing the need for future research into AI-assisted mathematical discovery.
   
 
<br/>
# Results  



이 연구에서는 **대형 언어 모델(LLM)의 창의적 수학 문제 해결 능력**을 평가하기 위해 **CREATIVEMATH 데이터셋**을 활용하여 **다양한 LLM 모델들의 성능을 비교**하였습니다.  

#### **1. 비교 대상 모델 및 테스트 데이터**
- **비교 모델**
  - **폐쇄형 모델**: GPT-4o, Claude-3-Opus, Gemini-1.5-Pro  
  - **오픈소스 모델**: Llama-3-70B, Qwen1.5-72B, Yi-1.5-34B, Mixtral-8x22B, DeepSeek-V2  
  - **수학 특화 모델**: DeepSeek-Math-7B-RL, Internlm2-Math-20B  

- **테스트 데이터**  
  - **CREATIVEMATH 데이터셋의 서브셋 활용**  
  - **400개의 수학 문제와 605개의 해결책**  
  - 문제 난이도: **중학교 수준부터 국제 수학 올림피아드(IMO) 수준까지 포함**  
  - **각 문제마다 1~5개(k=1~5)의 기존 해결책이 제공됨**  

---

#### **2. 성능 비교 메트릭**
모델 비교는 다음과 같은 핵심 지표를 기준으로 평가되었습니다.  

- **Correctness (C, 정확성 비율)**: 모델이 수학적으로 올바른 해결책을 생성한 비율  
- **Novelty (N, 창의성 비율)**: 제공된 k개의 기존 해결책과 다른 해결책을 생성한 비율  
- **Novel-Unknown (Nu, 완전 창의적 해결책 비율)**: 인간이 만든 기존 해결책(n개)과도 다른 완전히 새로운 해결책을 생성한 비율  
- **Novelty-to-Correctness Ratio (N/C)**: 정확한 해결책 중 창의적인 해결책이 차지하는 비율  
- **Novel-Unknown-to-Novelty Ratio (Nu/N)**: 창의적인 해결책 중 완전히 새로운 해결책이 차지하는 비율  

---

#### **3. 주요 결과**
1. **Gemini-1.5-Pro가 가장 뛰어난 창의성을 보임**
   - **정확성(C) 69.92%**, **창의성(N) 66.94%**, **완전 창의적 해결책 비율(Nu) 65.45%**로 **모든 지표에서 가장 높은 성능**을 보임.
   - **Novelty-to-Correctness Ratio (N/C) 95.75%**로, **생성된 해결책 중 대부분이 기존 해결책과 달랐음**.  

2. **Llama-3-70B와 Claude-3-Opus도 창의적 해결책 생성 능력이 뛰어남**  
   - Llama-3-70B는 **창의성(N) 48.76%**, **정확성(C) 58.84%**로 오픈소스 모델 중 가장 높은 창의성 비율을 보임.  
   - Claude-3-Opus는 **창의성(N) 44.63%**, **정확성(C) 59.84%**로 GPT-4o보다 창의적 해결책 생성에 강점을 보임.  

3. **GPT-4o는 수학 문제 해결 능력은 뛰어나지만 창의적 해결책 생성 능력은 낮음**  
   - **정확성(C) 60.83%**로 높은 수준이지만, **창의성(N) 30.08%**로 상대적으로 낮았음.  
   - 이는 GPT-4o가 기존 해결책을 따라가는 경향이 강하고, 새로운 해결책을 창출하는 능력이 부족함을 시사함.  

4. **수학 특화 모델(DeepSeek-Math-7B-RL, Internlm2-Math-20B)은 창의성이 낮음**  
   - **정확성(C)이 38~40% 수준**, **창의성(N)은 10~12% 수준**으로 일반 LLM보다 낮은 창의성을 보임.  
   - 이는 해당 모델들이 특정한 정형화된 해결책을 학습했기 때문으로 추정됨.  

---

#### **4. 문제 난이도별 창의성 차이**
- **쉬운 문제 (AMC 8~12)에서는 정확성이 높지만 창의성은 낮음**  
  - AMC 8: **정확성(C) 71.80%, 창의성(N/C) 55.39%**  
- **난이도가 높은 문제 (USAMO, IMO)에서는 정확성이 낮지만 창의성이 증가함**  
  - USAMO: **정확성(C) 35.00%, 창의성(N/C) 83.01%**  
  - 이는 어려운 문제일수록 모델이 정형화된 해결책이 아닌 새로운 접근 방식을 찾아야 하기 때문으로 해석됨.  

---



This study evaluates the **creative problem-solving abilities of Large Language Models (LLMs)** using the **CREATIVEMATH dataset** by comparing their performance across multiple models and evaluation metrics.  

#### **1. Models and Test Dataset**
- **Compared Models**
  - **Closed-source LLMs**: GPT-4o, Claude-3-Opus, Gemini-1.5-Pro  
  - **Open-source LLMs**: Llama-3-70B, Qwen1.5-72B, Yi-1.5-34B, Mixtral-8x22B, DeepSeek-V2  
  - **Math-specialized LLMs**: DeepSeek-Math-7B-RL, Internlm2-Math-20B  

- **Test Dataset**
  - **Subset of the CREATIVEMATH dataset**  
  - **400 mathematical problems and 605 solutions**  
  - Problems range from **middle school to IMO (International Mathematical Olympiad) level**  
  - Each problem has **1 to 5 provided reference solutions (k=1~5)**  

---

#### **2. Performance Metrics**
Models were evaluated based on the following key metrics:  

- **Correctness (C)**: Percentage of correctly generated solutions  
- **Novelty (N)**: Percentage of correct solutions that differ from the k provided reference solutions  
- **Novel-Unknown (Nu)**: Percentage of solutions that differ from **all human-created solutions**  
- **Novelty-to-Correctness Ratio (N/C)**: Percentage of correct solutions that are also novel  
- **Novel-Unknown-to-Novelty Ratio (Nu/N)**: Percentage of novel solutions that are entirely unique  

---

#### **3. Key Findings**
1. **Gemini-1.5-Pro demonstrated the highest creativity**  
   - **Correctness (C): 69.92%**, **Novelty (N): 66.94%**, **Novel-Unknown (Nu): 65.45%**  
   - **Novelty-to-Correctness Ratio (N/C) 95.75%**, meaning most generated solutions were distinct.  

2. **Llama-3-70B and Claude-3-Opus also performed well in generating novel solutions**  
   - Llama-3-70B: **Novelty (N) 48.76%, Correctness (C) 58.84%** (best among open-source models)  
   - Claude-3-Opus: **Novelty (N) 44.63%, Correctness (C) 59.84%**, outperforming GPT-4o in creativity.  

3. **GPT-4o was highly accurate but less creative**  
   - **Correctness (C) 60.83%**, but **Novelty (N) only 30.08%**  
   - This suggests GPT-4o tends to follow existing patterns rather than generating innovative solutions.  

4. **Math-specialized models (DeepSeek-Math-7B-RL, Internlm2-Math-20B) showed low creativity**  
   - **Correctness (C) 38-40%**, **Novelty (N) only 10-12%**, likely due to their focus on structured solutions.  

---

#### **4. Impact of Problem Difficulty on Creativity**
- **Easier problems (AMC 8-12) had higher accuracy but lower creativity**  
  - AMC 8: **Correctness (C) 71.80%, Novelty-to-Correctness (N/C) 55.39%**  
- **Harder problems (USAMO, IMO) showed lower accuracy but increased creativity**  
  - USAMO: **Correctness (C) 35.00%, Novelty-to-Correctness (N/C) 83.01%**  
  - This suggests that as problem difficulty increases, LLMs rely less on known strategies and more on novel approaches.

<br/>
# 예제  


이 연구에서는 **CREATIVEMATH 데이터셋을 사용하여 LLM의 창의적 수학 문제 해결 능력을 평가**했습니다. 구체적으로 **모델이 제공된 해결책과 다른 새로운 해결책을 생성할 수 있는지**를 테스트하는 방식입니다.  

---

### **1. 테스트 데이터 예제**  
테스트 데이터는 **CREATIVEMATH 데이터셋**에서 가져온 **중학교 수준부터 올림피아드(IMO) 수준까지의 다양한 수학 문제**로 구성됩니다.  

#### **예제 1: 중학교 수준 문제 (AMC 8)**  
**문제:**  
> 어떤 수가 2와 3의 배수이지만 5의 배수가 아닌 가장 작은 자연수는 무엇인가?  

**기존 해결책 (참조 해결책, k=2 제공):**  
1. **최소공배수 접근법**: 2와 3의 최소공배수는 6. 6은 5의 배수가 아니므로 답은 6.  
2. **배수 체크 접근법**: 자연수를 순차적으로 확인하여 6이 조건을 만족하는 것을 찾음.  

**LLM이 생성한 새로운 해결책:**  
- **소인수분해 접근법**:  
  - 2와 3의 배수는 공통적으로 포함해야 하므로, 숫자는 \(2^a \times 3^b\) 형태여야 함.  
  - \(a = 1, b = 1\) 일 때 \(2^1 \times 3^1 = 6\), 5의 배수가 아니므로 정답은 6.  
- **일반화 접근법**:  
  - 조건을 만족하는 최소 자연수를 구하는 알고리즘을 제시하여, n이 \(2^a \times 3^b\) 형태일 때 \(5\)의 배수가 되지 않는 최소값을 계산함.  

**결과:**  
- 기존 해결책과 다른 방식으로 문제를 해결했으므로 **창의적인 해결책(Novel Solution)** 으로 평가됨.  

---

#### **예제 2: 올림피아드 수준 문제 (IMO)**  
**문제:**  
> 한 정삼각형 내부에 점 \( P \) 가 위치해 있고, 각 변에 대하여 점 \( P \) 에서 수선을 내려 얻은 세 점을 \( A_1, B_1, C_1 \) 이라고 하자.  
> \(\triangle A_1 B_1 C_1\) 의 넓이와 \(\triangle ABC\) 의 넓이 간의 관계를 구하시오.  

**기존 해결책 (참조 해결책, k=3 제공):**  
1. **기하학적 변환 접근법**: 세 수선이 정삼각형의 중심을 포함하며, 닮음비를 통해 넓이 비를 계산.  
2. **벡터 계산 접근법**: 삼각형의 넓이를 벡터 내적을 이용해 계산하여 비율 도출.  
3. **삼각함수 접근법**: 각을 이용해 넓이를 유도.  

**LLM이 생성한 새로운 해결책:**  
- **확률적 접근법**: 랜덤 점 \(P\) 를 선택한 후 몬테카를로 기법을 이용해 수학적으로 넓이 비를 도출.  
- **대칭성 이용 접근법**: 대칭성 개념을 활용하여 점 \( P \) 가 중심에 가까울수록 닮음비가 일정한 패턴을 따르는 것을 증명.  

**결과:**  
- 기존 해결책과 달리 확률론적 해석을 도입했으며, 대칭성을 이용한 새로운 풀이 방식도 포함되어 창의적 해결책으로 평가됨.  

---

### **2. 훈련 데이터 예제**  
훈련 데이터는 **CREATIVEMATH 데이터셋에서 수집된 실제 수학 문제와 기존 해결책들**을 포함합니다.  

- **총 6,469개 문제, 14,223개 해결책 포함**  
- 문제 유형: **산술, 대수, 기하, 확률, 조합론, 수론 등**  
- 난이도: **중학교(AMC 8) ~ 대학 수준(IMO)까지 포함**  
- 데이터 정제 과정: **중복 해결책 제거, 불완전한 데이터 삭제, LaTeX 변환 적용**  

---

### **3. 다른 모델 대비 나은 해결책 예제**  

#### **예제 3: 비교 실험 (GPT-4o vs. Gemini-1.5-Pro)**
**문제:**  
> 두 개의 주어진 정수 \( a, b \) 에 대해, \( f(n) = a^n + b^n \) 이 소수가 되는 \( n \) 값 중 가장 작은 값을 찾아라.  

**GPT-4o의 해결책:**  
- 기존 수학 공식들을 참고하여 \( n = 2, 3, 5 \) 를 대입해 확인.  
- 소수 판별 공식을 이용해 가장 작은 \( n \) 값을 도출.  

**Gemini-1.5-Pro의 해결책 (새로운 접근법 적용):**  
- **페르마 소정리 이용:**  
  - \( f(n) \) 에 대해 페르마 소정리를 적용하여 특정 조건을 만족하는 최소 \( n \) 값을 증명.  
- **패턴 인식 모델 활용:**  
  - 수열 패턴을 분석해 더 빠르게 \( n \) 값을 찾는 새로운 알고리즘 제안.  

**결과:**  
- Gemini-1.5-Pro는 보다 **추론적이고 일반화된 풀이 방식을 제안**하여 **창의성이 뛰어난 해결책으로 평가됨.**  

---


This study evaluates **the creative problem-solving capabilities of Large Language Models (LLMs)** using the **CREATIVEMATH dataset** to determine whether models can generate **novel solutions distinct from known ones**.  

---

### **1. Test Data Example**  
Test data is derived from the **CREATIVEMATH dataset**, containing problems from **middle school to IMO level.**  

#### **Example 1: Middle School Level (AMC 8)**  
**Problem:**  
> Find the smallest natural number that is a multiple of 2 and 3 but not a multiple of 5.  

**Reference Solutions (k=2 provided):**  
1. **LCM approach**: Compute LCM of 2 and 3 (which is 6). Since 6 is not a multiple of 5, the answer is 6.  
2. **Sequential checking approach**: Check numbers one by one until 6 is found.  

**LLM-Generated Novel Solutions:**  
- **Prime factorization approach**:  
  - A number must be of the form \(2^a \times 3^b\).  
  - Setting \(a = 1, b = 1\) gives \(6\), which is not a multiple of 5.  
- **Generalization approach**:  
  - An algorithmic approach is suggested to find such numbers systematically.  

**Result:**  
- The generated solution was **different from existing ones and considered novel.**  

---

#### **Example 2: Olympiad-Level Problem (IMO)**  
**Problem:**  
> Given a point \( P \) inside an equilateral triangle, perpendiculars are drawn to each side, forming points \( A_1, B_1, C_1 \).  
> Determine the relationship between the areas of \(\triangle A_1 B_1 C_1\) and \(\triangle ABC\).  

**Reference Solutions (k=3 provided):**  
1. **Geometric transformation approach**  
2. **Vector calculation approach**  
3. **Trigonometric approach**  

**LLM-Generated Novel Solutions:**  
- **Probabilistic approach**: Use Monte Carlo methods to estimate the ratio probabilistically.  
- **Symmetry-based approach**: Use the inherent symmetry to prove an invariant ratio.  

**Result:**  
- The novel approaches were **significantly different from known ones, making them highly creative solutions.**  

---

### **2. Comparison: GPT-4o vs. Gemini-1.5-Pro**  
**Problem:**  
> Find the smallest \( n \) such that \( f(n) = a^n + b^n \) is prime.  

**GPT-4o solution:**  
- Uses known formulas and sequentially checks \( n = 2, 3, 5 \).  

**Gemini-1.5-Pro solution:**  
- **Applies Fermat's Little Theorem** to derive conditions for prime outcomes.  
- **Utilizes pattern recognition** for faster inference.  

**Result:**  
- Gemini-1.5-Pro provided a **more generalized, innovative approach**, outperforming GPT-4o in **creative problem-solving.**







<br/>  
# 요약   



이 연구는 **CREATIVEMATH 데이터셋**을 활용하여 **대형 언어 모델(LLM)이 기존 해결책과 다른 창의적인 수학적 해결책을 제안할 수 있는지**를 평가하는 프레임워크를 개발했다. 실험 결과, **Gemini-1.5-Pro가 다른 모델보다 높은 창의성을 보이며, 기존 해결책과 완전히 다른 새로운 해결책을 가장 많이 생성**하는 것으로 나타났다. 특히, **일반적인 패턴 인식과 확률적 접근법을 활용한 혁신적인 풀이 방식이 기존 모델 대비 더 창의적인 해결책을 도출하는 데 효과적**임을 확인했다.  

---


This study introduces a framework utilizing the **CREATIVEMATH dataset** to evaluate whether **Large Language Models (LLMs) can generate novel mathematical solutions distinct from known ones**. The results show that **Gemini-1.5-Pro outperformed other models in creativity, producing the highest number of truly novel solutions**. Notably, **its ability to apply general pattern recognition and probabilistic approaches led to more innovative problem-solving strategies compared to existing models**.

<br/>  
# 기타  




#### **1. Figure 1: CREATIVEMATH 데이터셋 내 문제 분포**  
이 그림은 **CREATIVEMATH 데이터셋에서 문제 유형과 난이도에 따른 분포를 시각화**한 것이다. **대수(Algebra)와 기하(Geometry) 문제가 가장 많이 포함**되어 있으며, 이는 대부분의 수학 경진대회에서 중점적으로 다루는 주제임을 보여준다. 또한, **AMC 10~AIME 수준의 문제들이 가장 많으며, IMO 수준 문제는 상대적으로 적다**, 이는 초급~중급 난이도의 문제들이 더 많기 때문으로 분석된다.  

---

#### **2. Figure 2: 각 경진대회별 해결책 개수 분포**  
이 그림은 **CREATIVEMATH 데이터셋에서 각 수학 경진대회별 문제당 해결책의 개수 분포를 보여준다**. **AMC 10~AIME와 같은 중간 난이도의 대회에서는 다양한 해결책이 존재하지만, AMC 8과 같은 초급 문제나 IMO 수준의 고난이도 문제에서는 해결책의 개수가 적다**. 이는 쉬운 문제일수록 풀이 방법이 정형화되어 있고, 어려운 문제일수록 해결책을 찾기 어렵기 때문으로 보인다.  

---

#### **3. Figure 3: LLM의 해결책 생성 및 평가 파이프라인**  
이 그림은 **LLM이 생성한 해결책이 어떻게 평가되는지를 나타내는 파이프라인**이다. 모델이 문제와 기존 해결책 일부(k개)를 입력받아 새로운 해결책을 생성하면, 이를 ① **정확성 평가(Correctness Evaluation)**, ② **대략적 창의성 평가(Coarse-Grained Novelty Assessment)**, ③ **정밀 창의성 평가(Fine-Grained Novelty Assessment)** 단계를 거쳐 평가한다. 이 과정에서 **정확성이 보장된 해결책만 창의성 평가로 이동하며, 기존 해결책과 차별성이 인정될 경우 창의적인 해결책으로 간주된다**.  

---

#### **4. Table 1: 평가 메트릭 정의**  
이 표는 **실험에서 사용된 주요 평가 지표의 정의**를 정리한 것이다.  
- **Correctness (C, 정확성 비율):** 올바른 해결책을 생성한 비율  
- **Novelty (N, 창의성 비율):** 기존 해결책과 다른 해결책을 생성한 비율  
- **Novel-Unknown (Nu, 완전 창의적 해결책 비율):** 인간이 만든 모든 해결책과도 차별되는 완전히 새로운 해결책을 생성한 비율  
- **Novelty-to-Correctness Ratio (N/C):** 생성된 정확한 해결책 중 창의적인 해결책의 비율  
- **Novel-Unknown-to-Novelty Ratio (Nu/N):** 창의적인 해결책 중 완전히 새로운 해결책의 비율  

---

#### **5. Table 2: 모델별 성능 비교 결과**  
이 표는 **각 LLM이 CREATIVEMATH 데이터셋에서 생성한 해결책의 성능을 비교한 것**이다.  
- **Gemini-1.5-Pro가 모든 지표에서 가장 높은 성능을 보이며, 특히 Novelty(N)와 Novel-Unknown(Nu)에서 두드러진 성과를 나타냄**.  
- **Llama-3-70B와 Claude-3-Opus도 비교적 높은 창의성을 보였지만, GPT-4o는 높은 정확성(C)에도 불구하고 창의성이 낮음**.  
- **DeepSeek-Math-7B-RL 및 Internlm2-Math-20B 등 수학 특화 모델은 창의성이 낮고 정형화된 해결책을 주로 생성함**.  

---

#### **6. Table 3: 제공된 해결책 개수(k)에 따른 정확성(C) 변화**  
이 표는 **제공된 해결책(k)이 증가할 때 LLM의 정확성(C)이 어떻게 변화하는지를 보여준다**.  
- **모든 모델에서 k 값이 증가할수록 정확성이 향상됨**.  
- **Gemini-1.5-Pro는 k=4에서 100% 정확도를 기록하며, 기존 해결책을 효과적으로 학습하는 능력이 뛰어남**.  
- **반면, Yi-1.5-34B 및 DeepSeek-Math-7B-RL과 같은 모델은 k 값 증가에 따른 정확성 향상이 크지 않음**.  

---

#### **7. Table 4: 제공된 해결책 개수(k)와 창의성(N/C) 간의 관계**  
이 표는 **참조 해결책 개수(k)가 증가할수록 모델이 창의적인 해결책을 생성하는 능력에 어떤 영향을 미치는지를 보여준다**.  
- **k 값이 증가할수록 창의성(N/C)이 감소하는 경향이 나타남**, 즉, 많은 해결책을 제공하면 모델이 기존 해결책을 따르는 경향이 강해짐.  
- **Gemini-1.5-Pro는 k=2에서 N/C가 100%로 가장 높은 창의성을 보임**.  

---

#### **8. Figure 6: LLM 간 해결책 유사성 맵**  
이 그림은 **다양한 LLM이 생성한 해결책이 얼마나 유사한지를 시각적으로 나타낸 것**이다.  
- **Llama-3-70B와 Yi-1.5-34B의 해결책이 가장 차별화되어 있음(유사성 6%)** → 서로 다른 접근 방식을 탐색하는 모델임을 의미.  
- **Mixtral-8x22B와 Claude-3-Opus의 해결책은 유사성이 높음(47%)** → 해결책 패턴이 정형화된 경향이 있음.  
- **이 결과는 창의적인 해결책을 생성하려면 서로 다른 유형의 모델을 조합하여 사용하는 것이 효과적임을 시사**.  

---



### **1. Figure 1: Distribution of Problems in CREATIVEMATH**  
This figure **visualizes the distribution of problems by category and difficulty in the CREATIVEMATH dataset**. **Algebra and Geometry are the most represented topics**, reflecting their prominence in math competitions. **Intermediate-level competitions like AMC 10–AIME contain the most problems, while IMO-level problems are less frequent**, as harder problems tend to be rarer.  

---

### **2. Figure 2: Distribution of Solutions per Competition**  
This figure **shows the number of solutions per problem across different math competitions**. **Mid-level contests like AMC 10 and AIME have more solutions**, whereas **simpler contests (AMC 8) and high-level ones (IMO) have fewer solutions**, indicating that easier problems are more standardized while harder ones are more challenging to solve.  

---

### **3. Figure 3: Solution Generation and Evaluation Pipeline**  
This figure **illustrates the multi-stage evaluation pipeline** used to assess LLM-generated solutions. The process includes ① **Correctness Evaluation**, ② **Coarse-Grained Novelty Assessment**, and ③ **Fine-Grained Novelty Assessment**. **Only solutions that pass correctness checks proceed to creativity evaluations, ensuring quality and originality.**  

---

### **4. Table 1: Evaluation Metric Definitions**  
This table defines key evaluation metrics, including **Correctness (C), Novelty (N), and Novel-Unknown (Nu)**, measuring how well LLMs generate correct and novel solutions.  

---

### **5. Table 2: Model Performance Comparison**  
This table compares **LLM performance on CREATIVEMATH**:  
- **Gemini-1.5-Pro had the best overall performance, excelling in Novelty (N) and Novel-Unknown (Nu).**  
- **GPT-4o had high correctness but low creativity.**  
- **Math-specialized models (DeepSeek-Math-7B-RL, Internlm2-Math-20B) produced rigid, less novel solutions.**  

---

### **6. Table 3: Impact of k on Correctness (C)**  
This table shows that **as k increases, correctness improves for all models**, with **Gemini-1.5-Pro reaching 100% at k=4.**  

---

### **7. Table 4: Impact of k on Novelty (N/C)**  
This table reveals that **increasing k reduces novelty**, as models tend to follow provided examples rather than generate new solutions.  

---

### **8. Figure 6: Solution Similarity Map Across LLMs**  
This figure shows how similar the solutions of different LLMs are:  
- **Llama-3-70B and Yi-1.5-34B explored different solutions (6% similarity).**  
- **Mixtral-8x22B and Claude-3-Opus had higher similarity (47%), suggesting rigid patterns.**  
- **Using diverse models enhances creativity in AI-generated solutions.**


<br/>
# refer format:     

@article{Ye2025CREATIVEMATH,
  author    = {Junyi Ye and Jingyi Gu and Xinyun Zhao and Wenpeng Yin and Guiling Wang},
  title     = {Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems},
  journal   = {Proceedings of the Association for the Advancement of Artificial Intelligence},
  year      = {2025},
  url       = {https://github.com/NJIT-AI-Center/CreativeMath}
}
  


Ye, Junyi, Jingyi Gu, Xinyun Zhao, Wenpeng Yin, and Guiling Wang. 2025. "Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems." Proceedings of the Association for the Advancement of Artificial Intelligence. Accessed from https://github.com/NJIT-AI-Center/CreativeMath.





