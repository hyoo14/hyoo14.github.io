---
layout: post
title:  "[2025]Mercury: Ultra-Fast Language Models Based on Diffusion"
date:   2025-12-09 19:57:32 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: Mercury는 병렬로 여러 토큰을 예측하는 확산 기반의 대형 언어 모델로, 코딩 애플리케이션에 최적화된 Mercury Coder Mini와 Small 모델을 포함한다.

앞에서부터 한 글자씩 타자 치는 방식이 아니라,   
긴 문장 초안을 한 번에 써놓고 여러 번 고쳐 나가는 방식   


짧은 요약(Abstract) :


이 논문에서는 Mercury라는 새로운 상업용 대규모 언어 모델(LLM) 세대를 소개합니다. 이 모델들은 확산(diffusion) 기반으로 설계되었으며, Transformer 아키텍처를 통해 매개변수가 설정되고 여러 토큰을 병렬로 예측하도록 훈련되었습니다. Mercury Coder는 코딩 응용 프로그램을 위해 설계된 첫 번째 확산 LLM 세트로, 현재 Mini와 Small 두 가지 크기로 제공됩니다. 이 모델들은 속도와 품질의 최전선에서 새로운 최첨단을 설정하며, NVIDIA H100 GPU에서 Mercury Coder Mini는 초당 1109 토큰, Mercury Coder Small은 초당 737 토큰의 처리량을 달성하여 속도 최적화 모델보다 평균 10배 더 빠르면서도 유사한 품질을 유지합니다. 다양한 프로그래밍 언어와 사용 사례에 걸친 코드 벤치마크에서 추가 결과를 논의하며, 개발자들이 사용하는 Copilot Arena에서 현재 품질 면에서 두 번째로 높은 순위를 기록하고 있으며, 전체적으로 가장 빠른 모델로 평가받고 있습니다. 또한, public API와 무료 플레이그라운드를 제공하고 있습니다.



This paper presents Mercury, a new generation of commercial-scale large language models (LLMs) based on diffusion. These models are parameterized via the Transformer architecture and trained to predict multiple tokens in parallel. Mercury Coder is our first set of diffusion LLMs designed for coding applications, currently available in two sizes: Mini and Small. These models set a new state-of-the-art on the speed-quality frontier, achieving throughputs of 1109 tokens/sec for Mercury Coder Mini and 737 tokens/sec for Mercury Coder Small on NVIDIA H100 GPUs, outperforming speed-optimized frontier models by up to 10× on average while maintaining comparable quality. We discuss additional results on a variety of code benchmarks spanning multiple languages and use cases, as well as real-world validation by developers on Copilot Arena, where the model currently ranks second in quality and is the fastest model overall. We also release a public API and a free playground.


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



**모델 및 아키텍처**
Mercury 모델은 Transformer 아키텍처를 기반으로 하며, 이는 최근 몇 년간 대규모 언어 모델의 훈련 및 추론을 위한 다양한 최적화 기법과 호환성을 보장합니다. 이 모델은 전통적인 오토회귀 모델과는 달리, 여러 토큰을 병렬로 생성할 수 있는 능력을 가지고 있습니다. 이러한 병렬 생성 방식은 속도를 크게 향상시키고, 더 세밀한 제어 및 추론 능력을 제공합니다.

**트레이닝 데이터**
Mercury 모델은 수조 개의 토큰에 해당하는 대규모 데이터셋으로 훈련됩니다. 이 데이터셋은 웹 크롤링을 통해 수집된 데이터와 함께, 신뢰할 수 있는 실제 및 합성 데이터셋으로 구성됩니다. 이러한 데이터는 모델이 다양한 프로그래밍 언어와 사용 사례에 대해 높은 정확도와 올바른 코드를 생성할 수 있도록 돕습니다.

**특별한 기법**
Mercury 모델은 디퓨전 프로세스를 통해 훈련됩니다. 이 과정은 초기의 무작위 노이즈에서 시작하여 점진적으로 데이터 분포에서 샘플을 생성하는 방식으로 진행됩니다. 모델은 노이즈가 있는 데이터에서 깨끗한 데이터를 생성하는 역 프로세스를 통해 학습하며, 이 과정에서 손실 함수를 최소화하는 방향으로 파라미터를 조정합니다. 이러한 디퓨전 모델은 기존의 오토회귀 모델에 비해 더 높은 연산 효율성을 제공하며, 특히 코드 생성과 같은 지연 민감한 작업에서 사용자 경험을 크게 향상시킵니다.

**추론 및 응용**
Mercury 모델은 조건부 생성이 가능하여, 주어진 프롬프트나 컨텍스트에 따라 유연하게 응답을 생성할 수 있습니다. 이 모델은 기존의 오토회귀 모델과의 호환성을 유지하면서도, 더 빠른 추론 속도를 제공합니다. 또한, 다양한 프로그래밍 언어에 대한 코드 생성 성능이 뛰어나며, 실제 개발자들에 의해 검증된 바 있습니다.

---




**Model and Architecture**
The Mercury model is based on the Transformer architecture, which ensures compatibility with various optimization techniques developed in recent years for the training and inference of large-scale language models. Unlike traditional autoregressive models, this model has the capability to generate multiple tokens in parallel. This parallel generation significantly enhances speed and provides finer control and reasoning capabilities.

**Training Data**
The Mercury model is trained on a large-scale dataset comprising trillions of tokens. This dataset consists of data collected through web crawling, along with carefully curated real and synthetic datasets. Such data helps the model achieve high accuracy and correctness in generating code across various programming languages and use cases.

**Special Techniques**
The Mercury model is trained using a diffusion process. This process starts from random noise and gradually transforms it into a sample from the data distribution. The model learns by generating clean data from noisy data through a reverse process, adjusting parameters to minimize a loss function. This diffusion model offers higher computational efficiency compared to traditional autoregressive models, significantly improving user experience, especially in latency-sensitive tasks like code generation.

**Inference and Applications**
The Mercury model supports conditional generation, allowing it to flexibly respond based on a given prompt or context. It maintains compatibility with existing autoregressive models while providing faster inference speeds. Additionally, it demonstrates excellent performance in code generation across various programming languages, validated by real-world developers.


<br/>
# Results



**결과 요약**

Mercury Coder 모델은 다양한 코드 생성 벤치마크에서 성능을 평가받았으며, 그 결과는 다음과 같습니다:

1. **경쟁 모델**: Mercury Coder Mini와 Small 모델은 여러 오픈 웨이트 모델 및 프론티어 모델과 비교되었습니다. 오픈 웨이트 모델로는 Llama 3.1, Mistral Small 3, Qwen 2.5 Coder 등이 있으며, 프론티어 모델로는 GPT-4o, Claude 3.5 등이 포함되었습니다.

2. **테스트 데이터**: 모델의 성능은 HumanEval, MBPP, EvalPlus, MultiPL-E, LiveCodeBench, BigCodeBench와 같은 표준 벤치마크를 통해 평가되었습니다. 이 벤치마크들은 Python 코드 생성, 다국어 코드 생성, 코드 완성 능력 등을 측정합니다.

3. **메트릭**: 성능 평가는 주로 'pass@1' 비율과 처리 속도(초당 출력 토큰 수)로 이루어졌습니다. 'pass@1' 비율은 모델이 주어진 테스트 케이스에서 올바른 솔루션을 생성한 비율을 나타냅니다. 처리 속도는 모델이 얼마나 빠르게 코드를 생성할 수 있는지를 나타냅니다.

4. **비교 결과**:
   - **Mercury Coder Mini**는 모든 오픈 웨이트 모델을 초과하는 성능을 보였으며, 초당 약 1,100 토큰의 속도로 8배 이상 빠른 성능을 기록했습니다.
   - **Mercury Coder Small**은 Claude 3.5 Haiku 및 Gemini 2.0 Flash와 같은 프론티어 속도 최적화 모델과 동등한 성능을 보였으며, 속도 면에서도 훨씬 빠른 결과를 나타냈습니다.
   - MultiPL-E 벤치마크에서 Mercury Coder Mini는 Java 및 JavaScript에서 특히 우수한 성능을 보였으며, 코드 완성 시나리오에서도 모든 평가 모델을 초과하는 성능을 기록했습니다.

5. **인간 평가**: Copilot Arena에서 Mercury Coder Mini는 다른 모델들과 비교하여 두 번째로 높은 품질 점수를 기록했으며, 평균 지연 시간은 25ms로 GPT-4o Mini보다 약 4배 빠른 성능을 보였습니다.

이러한 결과들은 Mercury Coder 모델이 코드 생성 작업에서 높은 효율성과 품질을 제공함을 보여줍니다. 특히, 이 모델들은 높은 처리 속도와 함께 경쟁력 있는 품질을 유지하여 실제 애플리케이션에서의 사용 가능성을 높이고 있습니다.

---


**Summary of Results**

The Mercury Coder models were evaluated on various code generation benchmarks, and the results are as follows:

1. **Competing Models**: The Mercury Coder Mini and Small models were compared against several open-weight models and frontier models. Open-weight models included Llama 3.1, Mistral Small 3, and Qwen 2.5 Coder, while frontier models included GPT-4o and Claude 3.5.

2. **Test Data**: The performance of the models was assessed using standard benchmarks such as HumanEval, MBPP, EvalPlus, MultiPL-E, LiveCodeBench, and BigCodeBench. These benchmarks measure Python code generation, multi-language code generation, and code completion capabilities.

3. **Metrics**: Performance evaluation was primarily based on the 'pass@1' rate and throughput (tokens per second). The 'pass@1' rate indicates the proportion of test cases for which the model generated the correct solution. Throughput indicates how quickly the model can generate code.

4. **Comparison Results**:
   - **Mercury Coder Mini** outperformed all open-weight models, achieving a throughput of approximately 1,100 tokens per second, making it over 8 times faster.
   - **Mercury Coder Small** demonstrated performance on par with frontier speed-optimized models like Claude 3.5 Haiku and Gemini 2.0 Flash, while also being significantly faster.
   - On the MultiPL-E benchmark, Mercury Coder Mini showed particularly strong performance in Java and JavaScript, and achieved state-of-the-art results in code completion scenarios, surpassing all evaluated models.

5. **Human Evaluation**: In the Copilot Arena, Mercury Coder Mini ranked second in quality compared to other models, with an average latency of just 25 ms, making it about 4 times faster than GPT-4o Mini.

These results demonstrate that the Mercury Coder models provide high efficiency and quality in code generation tasks. Notably, these models maintain competitive quality while achieving high throughput, enhancing their applicability in real-world scenarios.


<br/>
# 예제



Mercury 모델은 코드 생성 및 관련 작업을 위해 설계된 대규모 확산 기반 언어 모델입니다. 이 모델은 다양한 프로그래밍 언어에 대한 코드 생성 능력을 평가하기 위해 여러 벤치마크를 사용합니다. 예를 들어, HumanEval과 MBPP는 Python 코드 생성을 평가하는 데 사용되며, 이들 벤치마크는 주어진 입력에 대해 올바른 코드를 생성하는 비율(즉, 테스트 통과율)을 측정합니다.

1. **트레이닝 데이터**: Mercury 모델은 웹 크롤링 데이터와 함께 실제 및 합성 데이터 세트를 포함하여 수조 개의 토큰으로 구성된 대규모 데이터 세트에서 훈련됩니다. 이 데이터는 다양한 프로그래밍 언어와 관련된 코드 샘플을 포함하고 있습니다.

2. **테스트 데이터**: 모델의 성능을 평가하기 위해 HumanEval과 MBPP와 같은 벤치마크에서 제공하는 테스트 데이터가 사용됩니다. 예를 들어, HumanEval은 특정 기능을 수행하는 Python 함수를 생성하는 문제를 포함하고 있으며, 각 문제는 함수의 입력과 예상 출력이 명시되어 있습니다.

3. **구체적인 인풋과 아웃풋**:
   - **인풋**: "주어진 정수 리스트에서 최대값을 찾는 함수를 작성하시오."
   - **아웃풋**: 
     ```python
     def find_max(nums):
         return max(nums)
     ```

4. **구체적인 테스크**: 모델은 주어진 문제 설명에 따라 코드를 생성해야 하며, 생성된 코드는 주어진 테스트 케이스를 통과해야 합니다. 예를 들어, "find_max([1, 2, 3])"를 호출했을 때, 함수는 3을 반환해야 합니다.

이러한 방식으로 Mercury 모델은 다양한 프로그래밍 언어에 대한 코드 생성 능력을 평가하고, 그 성능을 다른 모델과 비교하여 우수성을 입증합니다.

---




The Mercury model is a large-scale diffusion-based language model designed for code generation and related tasks. It employs various benchmarks to evaluate its code generation capabilities across different programming languages. For instance, HumanEval and MBPP are used to assess Python code generation, measuring the rate of correct code generation (i.e., pass rates) for given inputs.

1. **Training Data**: The Mercury model is trained on a large dataset comprising trillions of tokens, which includes web-crawled data along with real and synthetic datasets. This data encompasses code samples related to various programming languages.

2. **Test Data**: To evaluate the model's performance, test data from benchmarks like HumanEval and MBPP is utilized. For example, HumanEval includes problems that require generating Python functions, with each problem specifying the function's input and expected output.

3. **Specific Input and Output**:
   - **Input**: "Write a function that finds the maximum value in a given list of integers."
   - **Output**: 
     ```python
     def find_max(nums):
         return max(nums)
     ```

4. **Specific Task**: The model must generate code based on the provided problem description, and the generated code should pass the given test cases. For instance, when calling "find_max([1, 2, 3])", the function should return 3.

In this way, the Mercury model evaluates its code generation capabilities across various programming languages and demonstrates its superiority by comparing its performance with other models.

<br/>


Mercury는 병렬로 여러 토큰을 예측하는 확산 기반의 대형 언어 모델로, 코딩 애플리케이션에 최적화된 Mercury Coder Mini와 Small 모델을 포함한다. 이 모델들은 NVIDIA H100 GPU에서 각각 1109 및 737 토큰/초의 속도를 기록하며, 기존의 속도 최적화 모델보다 최대 10배 빠르면서도 유사한 품질을 유지한다. 다양한 프로그래밍 언어에 대한 벤치마크에서 우수한 성능을 보이며, Copilot Arena에서 품질과 속도 모두에서 높은 평가를 받았다.


Mercury is a diffusion-based large language model that predicts multiple tokens in parallel, featuring the Mercury Coder Mini and Small models optimized for coding applications. These models achieve speeds of 1109 and 737 tokens/sec on NVIDIA H100 GPUs, outperforming existing speed-optimized models by up to 10 times while maintaining comparable quality. They demonstrate strong performance across various programming languages in benchmarks and received high ratings for both quality and speed on Copilot Arena.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: Mercury Coder 모델의 품질과 속도 간의 트레이드오프를 보여줍니다. 이 피규어는 Mercury Coder Mini와 Small 모델이 다른 최신 모델들에 비해 최대 10배 더 높은 처리량을 기록하면서도 유사한 품질을 유지하고 있음을 나타냅니다. 이는 Mercury 모델이 속도와 품질 모두에서 우수한 성능을 발휘함을 강조합니다.

2. **테이블**
   - **Table 1**: 다양한 모델의 성능(정확도 및 속도)을 비교합니다. Mercury Coder Mini는 모든 오픈 웨이트 모델을 초과하는 성능을 보이며, 1,100 토큰/초의 속도로 작동합니다. Mercury Coder Small은 유명한 속도 최적화 모델들과 동등한 성능을 보이면서도 훨씬 빠른 속도를 자랑합니다. 이는 Mercury 모델이 실제 애플리케이션에서 높은 효율성을 제공할 수 있음을 시사합니다.
   - **Table 2**: MultiPL-E 벤치마크에서 다양한 프로그래밍 언어에 대한 모델 성능을 비교합니다. Mercury Coder Mini와 Small은 오픈 웨이트 모델을 초과하는 성능을 보이며, 특히 Java와 JavaScript에서 잘 작동합니다. 이는 다국어 코드 생성에서의 효과성을 강조합니다.
   - **Table 3**: Fill-in-the-middle(FIM) 작업에서의 성능을 비교합니다. Mercury Coder Mini와 Small은 모든 평가된 모델을 초과하는 성능을 기록하여 코드 완성 시나리오에서의 효과성을 보여줍니다.
   - **Table 4**: Copilot Arena에서의 모델 비교를 보여줍니다. Mercury Coder Mini는 평균 대기 시간이 25ms로 가장 빠른 모델이며, 이는 사용자 경험을 크게 향상시킬 수 있음을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에는 모델의 훈련 및 평가 방법론, 데이터 세트, 그리고 실험 설정에 대한 자세한 정보가 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 유사한 접근 방식을 사용할 수 있도록 돕습니다.

---



1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates the trade-off between quality and speed for the Mercury Coder models. It shows that the Mercury Coder Mini and Small models achieve up to 10 times higher throughput compared to other state-of-the-art models while maintaining comparable quality. This emphasizes the superior performance of Mercury models in both speed and quality.

2. **Tables**
   - **Table 1**: This table compares the performance (accuracy and speed) of various models. The Mercury Coder Mini outperforms all open-weight models and operates at a speed of 1,100 tokens/second. The Mercury Coder Small matches the performance of popular speed-optimized models while being significantly faster. This suggests that Mercury models can provide high efficiency in real-world applications.
   - **Table 2**: This table compares model performance on the MultiPL-E benchmark across different programming languages. Both Mercury Coder Mini and Small exceed the performance of open-weight models, particularly excelling in Java and JavaScript. This highlights their effectiveness in multi-language code generation.
   - **Table 3**: This table compares performance on fill-in-the-middle (FIM) tasks. Mercury Coder Mini and Small achieve state-of-the-art performance, surpassing all evaluated models, which demonstrates their effectiveness in code completion scenarios.
   - **Table 4**: This table presents a comparison of models in the Copilot Arena. Mercury Coder Mini is the fastest model with an average latency of just 25 ms, significantly enhancing user experience.

3. **Appendices**
   - The appendices contain detailed information about the training and evaluation methodologies, datasets, and experimental setups. This enhances the reproducibility of the research and helps other researchers to adopt similar approaches.

<br/>
# refer format:
### BibTeX 


```bibtex
@article{Khanna2025,
  author = {Samar Khanna and Siddhant Kharbanda and Shufan Li and Harshit Varma and Eric Wang and Sawyer Birnbaum and Ziyang Luo and Yanis Miraoui and Akash Palrecha and Stefano Ermon and Aditya Grover and Volodymyr Kuleshov},
  title = {Mercury: Ultra-Fast Language Models Based on Diffusion},
  journal = {arXiv preprint arXiv:2506.17298},
  year = {2025},
  url = {https://arxiv.org/abs/2506.17298},
}
```

### 시카고 스타일


Khanna, Samar, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, and Volodymyr Kuleshov. "Mercury: Ultra-Fast Language Models Based on Diffusion." arXiv preprint arXiv:2506.17298 (2025). https://arxiv.org/abs/2506.17298.
