---
layout: post
title:  "[2026]Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention"
date:   2026-07-02 17:01:35 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Minimal Test-Time Intervention (MTI)라는 방법을 제안하여 대형 언어 모델(LLM)의 추론 정확성과 안정성을 향상시킵니다.


짧은 요약(Abstract) :


이 논문에서는 대형 언어 모델(LLM)의 추론 능력을 향상시키기 위한 새로운 접근법인 최소 테스트 시간 개입(MTI)을 제안합니다. 기존의 연구들은 추론 시 더 많은 계산을 할당하여 reasoning을 개선하려고 했지만, 이는 종종 효율성을 저해했습니다. 저자들은 reasoning 불확실성이 고도로 국소화되어 있다는 점을 발견하였고, 이는 높은 엔트로피를 가진 소수의 토큰이 출력의 정확성에 주로 영향을 미친다는 것을 의미합니다. MTI는 이러한 고엔트로피 위치에서만 선택적으로 개입하여 reasoning의 정확성과 안정성을 향상시키며, 추가적인 계산 비용을 최소화합니다. MTI는 일반, 코딩, STEM 작업에서 일관된 성과 향상을 보여주며, DeepSeek-R1-7B 모델에서 평균 9.28%의 성과 향상을 달성했습니다.



This paper proposes a new approach called Minimal Test-Time Intervention (MTI) to enhance the reasoning capabilities of large language models (LLMs). Previous research has focused on allocating more computation during inference to improve reasoning, but this often comes at the cost of efficiency. The authors discovered that reasoning uncertainty is highly localized, meaning that only a small subset of high-entropy tokens predominantly affects output correctness. MTI selectively intervenes only at these high-entropy positions to improve reasoning accuracy and stability while minimizing additional computational costs. MTI demonstrates consistent performance gains across general, coding, and STEM tasks, achieving an average improvement of 9.28% on the DeepSeek-R1-7B model.


* Useful sentences :

pre-reasoning improvement와  
post training with human feedback(DPO, GRPO맞나?)  
그 중에서 pre-reasoning improvement차원, 일종의 CoT 파생으로 봐야하나  


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



이 논문에서 제안하는 방법은 "Minimal Test-Time Intervention (MTI)"라는 프레임워크로, 대규모 언어 모델(LLM)의 추론 성능을 향상시키기 위해 설계되었습니다. MTI는 훈련 없이도 적용할 수 있으며, 주로 두 가지 주요 기법을 포함합니다: 선택적 CFG(클래시파이어-프리 가이던스) 개입과 경량화된 부정 프롬프트 가이던스입니다.

1. **선택적 CFG 개입**: MTI는 모델이 생성하는 동안 각 토큰의 엔트로피를 측정하여 불확실성이 높은 토큰(즉, 높은 엔트로피를 가진 토큰)을 식별합니다. 이러한 높은 엔트로피 토큰은 모델의 예측에서 오류가 발생할 가능성이 높기 때문에, MTI는 이러한 특정 위치에서만 CFG를 적용하여 모델의 출력을 조정합니다. 이 방법은 불필요한 계산을 줄이고, 모델의 추론 안정성을 높이는 데 기여합니다.

2. **경량화된 부정 프롬프트 가이던스**: MTI는 부정 프롬프트를 사용하여 모델이 잘못된 출력을 생성하는 것을 방지합니다. 이 부정 프롬프트는 "OUTPUT ERROR"와 같은 짧은 문구로 구성되어 있으며, 모델의 입력에 추가되어 높은 엔트로피 토큰을 생성하기 전에 사용됩니다. 이 방식은 모델의 키-값 캐시를 재사용하여 메모리 사용량을 줄이고, 추가적인 계산 비용 없이도 모델의 예측 분포를 조정할 수 있게 합니다.

MTI는 다양한 벤치마크에서 일관된 성능 향상을 보여주며, 일반적인 작업, 코딩 작업, STEM(과학, 기술, 공학, 수학) 작업에서 모두 효과적입니다. 예를 들어, DeepSeek-R1-7B 모델을 사용한 실험에서는 평균 9.28%의 성능 향상을 기록하였고, AIME2024 벤치마크에서는 11.25%의 향상을 보였습니다. 이러한 결과는 MTI가 LLM의 추론 성능을 향상시키는 데 있어 효율적이고 실용적인 접근법임을 보여줍니다.




The method proposed in this paper is called "Minimal Test-Time Intervention (MTI)," designed to enhance the inference performance of large language models (LLMs). MTI is applicable without any training and primarily consists of two key techniques: selective Classifier-Free Guidance (CFG) intervention and lightweight negative prompt guidance.

1. **Selective CFG Intervention**: MTI identifies uncertain tokens (i.e., tokens with high entropy) by measuring the entropy of each token during generation. These high-entropy tokens are likely to be sources of errors in the model's predictions. Therefore, MTI applies CFG only at these specific positions to adjust the model's output. This approach reduces unnecessary computations and contributes to improving the stability of the model's reasoning.

2. **Lightweight Negative Prompt Guidance**: MTI employs negative prompts to prevent the model from generating incorrect outputs. These negative prompts consist of short phrases like "OUTPUT ERROR," which are added to the model's input before generating high-entropy tokens. This method reuses the model's key-value cache, reducing memory usage and allowing for adjustments to the model's predictive distribution without incurring additional computational costs.

MTI demonstrates consistent performance improvements across various benchmarks, effectively addressing general tasks, coding tasks, and STEM (Science, Technology, Engineering, and Mathematics) tasks. For instance, experiments using the DeepSeek-R1-7B model achieved an average performance improvement of 9.28%, while a notable 11.25% improvement was observed on the AIME2024 benchmark. These results indicate that MTI is an efficient and practical approach for enhancing the reasoning capabilities of LLMs.


<br/>
# Results



이 논문에서는 Minimal Test-Time Intervention (MTI)라는 새로운 접근 방식을 제안하여 대형 언어 모델(LLM)의 추론 성능을 향상시키고자 하였습니다. MTI는 고엔트로피 토큰에만 선택적으로 개입하여 모델의 추론 정확성과 안정성을 높이는 방법입니다. 이 방법은 기존의 여러 테스트 시간 최적화 기법들과 비교하여 효율성을 유지하면서도 성능을 개선하는 데 중점을 두고 있습니다.

#### 실험 결과
MTI의 성능을 평가하기 위해 여러 벤치마크 데이터셋에서 실험을 진행하였으며, 그 결과는 다음과 같습니다:

1. **경쟁 모델**: MTI는 Direct Inference (DI)와 Vanilla Classifier-Free Guidance (VC)와 비교되었습니다. DI는 모델이 직접적으로 출력을 생성하는 방식이며, VC는 모든 토큰에 대해 CFG를 적용하는 방법입니다.

2. **테스트 데이터**: MTI는 다양한 데이터셋에서 평가되었습니다. 여기에는 일반적인 작업(MMLU-Pro), 코딩 작업(HumanEval, HumanEvalPlus, LiveCodeBench), STEM 작업(GPQA-Diamond, MATH500, AIME2024) 등이 포함됩니다.

3. **메트릭**: 성능 평가는 정확도(Accuracy)로 측정되었습니다. MTI는 각 데이터셋에서 DI와 VC에 비해 일관된 성능 향상을 보였습니다.

4. **비교 결과**:
   - **DeepSeek-R1-7B 모델**: MTI는 평균 70.73%의 정확도를 기록하며, DI에 비해 9.28% 향상되었습니다. VC와 비교했을 때도 MTI는 더 나은 성능을 보였습니다.
   - **Ling-mini-2.0 모델**: AIME2024 벤치마크에서 MTI는 60.00%에서 71.25%로 향상되어 11.25%의 성과를 올렸습니다.
   - **Qwen3-14B 모델**: MTI는 85.33%의 평균 정확도를 기록하며, DI에 비해 3.00% 향상되었습니다.

MTI는 고엔트로피 토큰에만 개입하여 불필요한 계산을 줄이고, 모델의 추론 경로를 안정화하는 데 효과적임을 보여주었습니다. 이러한 결과는 MTI가 기존의 방법들보다 더 효율적이고 효과적인 접근 방식임을 입증합니다.




This paper proposes a novel approach called Minimal Test-Time Intervention (MTI) to enhance the reasoning performance of large language models (LLMs). MTI focuses on selectively intervening only at high-entropy tokens to improve the model's reasoning accuracy and stability. This method emphasizes maintaining efficiency while improving performance compared to various existing test-time optimization techniques.

#### Experimental Results
To evaluate the performance of MTI, experiments were conducted across several benchmark datasets, with the following results:

1. **Competing Models**: MTI was compared against Direct Inference (DI) and Vanilla Classifier-Free Guidance (VC). DI represents the model generating outputs directly, while VC applies CFG uniformly across all tokens.

2. **Test Data**: MTI was evaluated on diverse datasets, including general tasks (MMLU-Pro), coding tasks (HumanEval, HumanEvalPlus, LiveCodeBench), and STEM tasks (GPQA-Diamond, MATH500, AIME2024).

3. **Metrics**: Performance was measured using accuracy. MTI consistently demonstrated performance improvements over both DI and VC across all datasets.

4. **Comparison Results**:
   - **DeepSeek-R1-7B Model**: MTI achieved an average accuracy of 70.73%, representing a 9.28% improvement over DI. It also outperformed VC.
   - **Ling-mini-2.0 Model**: On the AIME2024 benchmark, MTI improved from 60.00% to 71.25%, yielding an 11.25% increase.
   - **Qwen3-14B Model**: MTI recorded an average accuracy of 85.33%, which is a 3.00% improvement over DI.

MTI effectively intervenes only at high-entropy tokens, reducing unnecessary computations and stabilizing the model's reasoning trajectory. These results demonstrate that MTI is a more efficient and effective approach compared to existing methods.


<br/>
# 예제



이 논문에서는 Minimal Test-Time Intervention (MTI)라는 새로운 접근 방식을 제안합니다. MTI는 대규모 언어 모델(LLM)의 추론 성능을 향상시키기 위해 설계된 방법으로, 주로 높은 엔트로피를 가진 토큰에만 선택적으로 개입하여 오류를 줄이고 정확성을 높이는 데 중점을 둡니다. 이 방법은 훈련 없이 적용할 수 있으며, 기존의 방법들보다 효율적입니다.

#### 예시 설명

1. **트레이닝 데이터**: 
   - **입력**: "4321을 57·28로 나누었을 때, 소수점 이하의 숫자 합은 얼마인가요? 단계별로 설명하고 최종 답을 괄호 안에 적어주세요."
   - **출력**: "14"

2. **테스트 데이터**:
   - **입력**: "4321을 57·28로 나누었을 때, 소수점 이하의 숫자 합은 얼마인가요? 단계별로 설명하고 최종 답을 괄호 안에 적어주세요."
   - **MTI 적용 전 출력**: "41" (잘못된 답변)
   - **MTI 적용 후 출력**: "14" (올바른 답변)

3. **구체적인 작업**:
   - 모델은 주어진 수학 문제를 해결하기 위해 단계별로 접근해야 합니다. 
   - 첫 번째 단계에서 모델은 분모의 구조를 이해하고, 이를 통해 소수점 이하의 숫자를 계산합니다.
   - MTI는 높은 엔트로피를 가진 토큰에 개입하여 모델이 잘못된 경로로 가지 않도록 돕습니다. 예를 들어, "28·57= 27·2·57= 27·107"와 같은 잘못된 계산을 방지하고, 올바른 계산으로 유도합니다.

이러한 방식으로 MTI는 모델의 추론 과정을 안정화하고, 최종적으로 더 정확한 결과를 도출할 수 있도록 합니다.

---




This paper proposes a new approach called Minimal Test-Time Intervention (MTI). MTI is designed to enhance the inference performance of large language models (LLMs) by selectively intervening only on tokens with high entropy, thereby reducing errors and improving accuracy. This method can be applied without retraining and is more efficient than existing methods.

#### Example Explanation

1. **Training Data**: 
   - **Input**: "What is the sum of the digits in the terminating decimal representation of the fraction 4321 divided by 57·28? Please reason step by step and put your final answer in parentheses."
   - **Output**: "14"

2. **Test Data**:
   - **Input**: "What is the sum of the digits in the terminating decimal representation of the fraction 4321 divided by 57·28? Please reason step by step and put your final answer in parentheses."
   - **Output before applying MTI**: "41" (incorrect answer)
   - **Output after applying MTI**: "14" (correct answer)

3. **Specific Task**:
   - The model is required to solve the given math problem step by step.
   - In the first step, the model needs to understand the structure of the denominator and calculate the digits in the decimal representation.
   - MTI intervenes at high-entropy tokens to prevent the model from going down the wrong path. For example, it helps avoid incorrect calculations like "28·57= 27·2·57= 27·107" and guides the model towards the correct calculation.

In this way, MTI stabilizes the model's reasoning process and ultimately leads to more accurate results.

<br/>
# 요약
이 논문에서는 Minimal Test-Time Intervention (MTI)라는 방법을 제안하여 대형 언어 모델(LLM)의 추론 정확성과 안정성을 향상시킵니다. MTI는 높은 엔트로피 토큰에만 선택적으로 개입하여 오류 전파를 방지하고, 경량화된 부정 프롬프트를 사용하여 메모리 오버헤드를 최소화합니다. 실험 결과, MTI는 다양한 벤치마크에서 일관된 성능 향상을 보여주며, 특히 AIME2024에서 11.25%의 정확도 향상을 달성했습니다.

---

This paper proposes a method called Minimal Test-Time Intervention (MTI) to enhance the reasoning accuracy and stability of large language models (LLMs). MTI selectively intervenes only at high-entropy tokens to prevent error propagation and employs lightweight negative prompts to minimize memory overhead. Experimental results demonstrate consistent performance improvements across various benchmarks, achieving a 11.25% accuracy increase specifically on the AIME2024 benchmark.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 이 그림은 정답과 오답의 평균 엔트로피를 비교합니다. 정답은 낮은 평균 엔트로피를 보이는 반면, 오답은 높은 엔트로피를 나타내며, 이는 불확실성이 높은 토큰이 오류의 주요 원인임을 시사합니다.
   - **Figure 2**: MTI와 기존의 Classifier-Free Guidance (CFG) 방법을 비교합니다. MTI는 높은 엔트로피 토큰에만 선택적으로 적용되어 효율성을 높이고, 불필요한 계산을 줄입니다.
   - **Figure 4**: 토큰의 엔트로피와 CFG의 효과를 분석합니다. 높은 엔트로피를 가진 토큰에서 CFG가 더 효과적으로 작용하는 경향이 있음을 보여줍니다.
   - **Figure 5**: MTI 적용 전후의 단어 구름을 시각화하여, MTI가 모델의 어휘 다양성을 어떻게 향상시키는지를 보여줍니다.

2. **테이블**
   - **Table 1**: 다양한 벤치마크에서 MTI의 성능을 Direct Inference (DI) 및 Vanilla CFG (VC)와 비교합니다. MTI는 모든 설정에서 DI를 초과하는 성능을 보이며, 특히 높은 엔트로피 토큰에 대한 선택적 개입이 효과적임을 보여줍니다.
   - **Table 2**: AIME2024 벤치마크에서 MTI의 성능을 평가합니다. MTI는 랜덤 샘플링을 통해도 성능을 향상시키며, DI와 VC를 초과하는 결과를 나타냅니다.
   - **Table 3**: MTI의 추론 비용을 평가합니다. MTI는 DI 및 VC보다 더 나은 성능을 보이면서도 지연 시간이 적은 결과를 보여줍니다.
   - **Table 5**: 최신 테스트 시간 스케일링 방법과 MTI를 비교합니다. MTI는 다른 방법들보다 일관되게 우수한 성능을 보입니다.

3. **어펜딕스**
   - **A.1**: 실험 설정 및 파일에 대한 정보를 제공합니다. OpenCompass를 사용하여 일관된 평가를 수행합니다.
   - **A.4**: 다양한 엔트로피 범위에서의 성능을 평가합니다. MTI는 높은 엔트로피 토큰에만 개입하여 성능을 향상시키는 반면, 낮은 엔트로피 토큰에 대한 개입은 성능을 저하시킵니다.
   - **A.6**: 작업별 부정 프롬프트의 효과를 분석합니다. 특정 작업에 맞춘 부정 프롬프트가 성능을 더욱 향상시킬 수 있음을 보여줍니다.

---

### Insights and Results from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: This figure compares the average entropy of correct and incorrect answers. Correct answers exhibit lower average entropy, while incorrect ones show higher entropy, indicating that high-uncertainty tokens are a major source of errors.
   - **Figure 2**: It compares MTI with the existing Classifier-Free Guidance (CFG) method. MTI selectively applies guidance only to high-entropy tokens, enhancing efficiency and reducing unnecessary computations.
   - **Figure 4**: Analyzes the relationship between token entropy and the effectiveness of CFG. It shows that CFG tends to be more effective on tokens with high entropy.
   - **Figure 5**: Visualizes word clouds before and after applying MTI, demonstrating how MTI enhances lexical diversity in the model's outputs.

2. **Tables**
   - **Table 1**: Compares the performance of MTI against Direct Inference (DI) and Vanilla CFG (VC) across various benchmarks. MTI consistently outperforms DI in all settings, highlighting the effectiveness of selective intervention on high-entropy tokens.
   - **Table 2**: Evaluates MTI's performance on the AIME2024 benchmark. MTI shows improvements even under random sampling, surpassing both DI and VC.
   - **Table 3**: Assesses the inference cost of MTI. It demonstrates that MTI achieves better performance with lower latency compared to DI and VC.
   - **Table 5**: Compares MTI with state-of-the-art test-time scaling methods, showing that MTI consistently delivers superior performance.

3. **Appendices**
   - **A.1**: Provides information on experimental setup and files, using OpenCompass for consistent evaluation.
   - **A.4**: Evaluates performance under different entropy scopes. MTI improves performance by intervening only on high-entropy tokens, while intervention on low-entropy tokens degrades performance.
   - **A.6**: Analyzes the effectiveness of task-specific negative prompts. It shows that aligning the semantics of negative prompts with task characteristics can further enhance performance.

<br/>
# refer format:
### BibTeX   

```bibtex
@inproceedings{yang2026less,
  title={Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention},
  author={Zhen Yang and Mingyang Zhang and Feng Chen and Ganggui Ding and Liang Hou and Xin Tao and Ying-Cong Chen},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={20124--20137},
  year={2026},
  month={July},
  publisher={Association for Computational Linguistics},
  address={GZ, China},
  email={zheny.cs@gmail.com, yingcongchen@ust.hk}
}
```

### Chicago style  

Zhen Yang, Mingyang Zhang, Feng Chen, Ganggui Ding, Liang Hou, Xin Tao, and Ying-Cong Chen. "Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 20124–20137. GZ, China: Association for Computational Linguistics, July 2026.
