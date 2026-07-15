---
layout: post
title:  "[2026]EPICAR: Knowing What You Don’t Know Matters for Better Reasoning in LLMs"
date:   2026-07-14 18:51:18 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: EPICAR(에피스테믹 캘리브레이션 추론)는 대규모 언어 모델(LLM)의 추론 능력을 향상시키기 위해 설계된 새로운 훈련 프레임워크입니다. EPICAR는 Llama-3와 Qwen-3 모델 계열을 기반  

**이중 목표 훈련**을 통해 모델이 문제 해결 능력과 자기 평가 능력을 동시에 학습하도록 합니다. 모델은 올바른 경로와 잘못된 경로 모두를 학습하여, 잘못된 경로에 대한 부정적인 신호를 통해 자기 평가를 강화합니다. 둘째, **적응형 주입 디코딩(AID)** 기법을 통해 모델이 생성하는 경로의 형식 오류를 방지하고, 이를 통해 잘못된 경로가 "잘못된" 신호로 잘못 레이블링되는 것을 방지합니다. AID는 모델의 출력을 모니터링하고, 필요한 경우 형식 준수를 강제하여 최종 출력을 개선  

epistemic uncertainty: 문제 자체가 애매해서 생기는 불확실성이 아니라, 모델의 지식이나 추론 능력이 부족해서 생기는 불확실성입니다.    

이 논문에서는 uncertainty를 모델에게 숫자로 직접 말하게 하는 게 아니라, yes/no 토큰의 확률로 뽑습니다.  
문제와 모델이 생성한 풀이·정답을 다시 보여주고 다음처럼 자기평가를 시킵니다.  
“이 답이 맞습니까? yes / no”  
그때 모델의 마지막 출력 분포에서 yes와 no의 logit을 가져와 정규화합니다.  
즉 이 로짓값인 컨피던스 스코어를 가져와서 언써틴티 = 1-컨피던스 -> 이런식으로   


짧은 요약(Abstract) :


이 논문에서는 대형 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 새로운 접근법인 EPICAR(에피스테믹 보정 추론)를 제안합니다. 기존의 자기 훈련 방법들은 주로 성공적인 추론 경로를 강화하는 데 집중하여 모델이 과신하게 만들고 불확실성을 표현하는 능력을 잃게 만듭니다. 이러한 문제를 해결하기 위해, EPICAR는 모델이 언제 자신의 추론을 신뢰해야 하는지를 학습하도록 하는 에피스테믹 학습 문제로 재구성합니다. 이 방법은 추론 성능과 보정을 동시에 최적화하는 훈련 목표를 제시하며, 메타 인지적 자기 평가 신호를 명시적으로 추출하여 반복적인 감독 세부 조정 프레임워크 내에서 구현됩니다. 실험 결과, EPICAR는 Llama-3 및 Qwen-3 모델에서 정확성과 보정 모두에서 기존 기준선보다 우수한 성능을 보였으며, 특히 충분한 추론 능력을 가진 모델에서 두드러진 성과를 나타냈습니다. 궁극적으로, 이 접근법은 전체 추론 계산 예산을 3배 줄이면서도 외부 검증자의 다중 모델 오버헤드 없이도 높은 성능을 달성할 수 있게 합니다.



This paper proposes a new approach called EPICAR (Epistemically Calibrated Reasoning) to enhance the reasoning abilities of large language models (LLMs). Existing self-training methods primarily focus on reinforcing successful reasoning paths, which leads to models becoming overconfident and losing the ability to represent uncertainty. To address this issue, EPICAR reframes reasoning training as an epistemic learning problem, where models learn not only how to reason but also when to trust their reasoning. This method introduces a training objective that jointly optimizes reasoning performance and calibration, and it is instantiated within an iterative supervised fine-tuning framework using explicitly extracted meta-cognitive self-evaluation signals. Experimental results demonstrate that EPICAR achieves superior performance in both accuracy and calibration compared to standard baselines on the Llama-3 and Qwen-3 models, particularly in models with sufficient reasoning capacity. Ultimately, this approach enables a 3× reduction in the overall inference compute budget while achieving high performance without the multi-model overhead of external verifiers.


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
이 연구에서 제안하는 EPICAR(에피스테믹 캘리브레이션 추론)는 대규모 언어 모델(LLM)의 추론 능력을 향상시키기 위해 설계된 새로운 훈련 프레임워크입니다. EPICAR는 Llama-3와 Qwen-3 모델 계열을 기반으로 하며, 이들 모델은 각각 1B, 3B, 4B, 8B 파라미터를 가진 다양한 크기로 구성되어 있습니다. 이러한 모델들은 복잡한 수학적 문제 해결 및 코드 생성과 같은 고차원적 추론 작업을 수행할 수 있도록 훈련되었습니다.

**훈련 데이터**  
훈련 데이터로는 MATH 데이터셋과 GSM8K, MBPP와 같은 다양한 벤치마크가 사용되었습니다. MATH 데이터셋은 수학 문제를 포함하고 있으며, GSM8K는 수학적 문제 해결을 위한 자연어 질문을 포함하고 있습니다. MBPP는 코드 생성 문제를 다루고 있습니다. 이러한 데이터셋은 모델이 다양한 문제를 해결할 수 있도록 훈련하는 데 사용됩니다.

**특별한 기법**  
EPICAR는 두 가지 주요 기법을 사용하여 모델의 추론 성능과 신뢰성을 동시에 최적화합니다. 첫째, **이중 목표 훈련**을 통해 모델이 문제 해결 능력과 자기 평가 능력을 동시에 학습하도록 합니다. 모델은 올바른 경로와 잘못된 경로 모두를 학습하여, 잘못된 경로에 대한 부정적인 신호를 통해 자기 평가를 강화합니다. 둘째, **적응형 주입 디코딩(AID)** 기법을 통해 모델이 생성하는 경로의 형식 오류를 방지하고, 이를 통해 잘못된 경로가 "잘못된" 신호로 잘못 레이블링되는 것을 방지합니다. AID는 모델의 출력을 모니터링하고, 필요한 경우 형식 준수를 강제하여 최종 출력을 개선합니다.

이러한 기법들은 모델이 자신의 신뢰성을 평가하고, 잘못된 경로에 대해 낮은 신뢰도를 부여함으로써, 고신뢰도의 추론 결과를 생성할 수 있도록 돕습니다. EPICAR는 이러한 과정을 통해 모델이 고차원적 추론 작업에서 더 나은 성능을 발휘할 수 있도록 합니다.




**Model and Architecture**  
The proposed EPICAR (Epistemically-Calibrated Reasoning) is a novel training framework designed to enhance the reasoning capabilities of large language models (LLMs). EPICAR is based on the Llama-3 and Qwen-3 model families, which consist of various sizes with 1B, 3B, 4B, and 8B parameters. These models are trained to perform complex reasoning tasks such as mathematical problem-solving and code generation.

**Training Data**  
The training data includes the MATH dataset, as well as various benchmarks like GSM8K and MBPP. The MATH dataset contains mathematical problems, while GSM8K includes natural language questions for mathematical problem-solving. MBPP focuses on code generation tasks. These datasets are utilized to train the models to solve a wide range of problems.

**Special Techniques**  
EPICAR employs two main techniques to optimize both the reasoning performance and reliability of the models simultaneously. First, it uses a **dual-objective training** approach that allows the model to learn both problem-solving skills and self-evaluation capabilities. The model learns from both correct and incorrect paths, reinforcing self-evaluation through negative signals from incorrect paths. Second, it incorporates **Adaptive Injection Decoding (AID)** to prevent formatting errors in the generated paths, ensuring that incorrect paths are not mislabeled as "incorrect" due to minor formatting issues. AID monitors the model's output and enforces format compliance when necessary, improving the final output.

These techniques help the model assess its own reliability and assign low confidence to incorrect paths, ultimately generating high-confidence reasoning results. EPICAR facilitates the model's ability to perform better in high-dimensional reasoning tasks through this process.


<br/>
# Results


이 논문에서는 EPICAR(에피스테믹 캘리브레이션 추론)라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLM)의 추론 능력을 향상시키고, 모델의 신뢰성을 높이는 방법을 다루고 있습니다. EPICAR는 모델이 자신의 추론 결과에 대한 신뢰도를 평가할 수 있도록 훈련하는 것을 목표로 하며, 이를 통해 모델의 과신을 줄이고 불확실성을 더 잘 표현할 수 있도록 합니다.

#### 실험 결과

1. **경쟁 모델**: EPICAR는 STaR(자기 훈련 추론) 및 Slow Thinking(느린 사고)와 같은 기존 모델과 비교되었습니다. STaR는 긍정적인 피드백 루프를 통해 모델의 정확성을 높이지만, 종종 신뢰성에 부정적인 영향을 미치는 경향이 있습니다.

2. **테스트 데이터**: 실험은 MATH, GSM8K, MBPP와 같은 다양한 데이터셋에서 수행되었습니다. MATH 데이터셋은 수학 문제 해결을 위한 데이터셋이며, GSM8K는 수학적 추론의 OOD(Out-Of-Distribution) 일반화를 평가하기 위한 데이터셋입니다. MBPP는 코드 생성 작업을 위한 데이터셋입니다.

3. **메트릭**: 모델의 성능은 정확도(Accuracy), AUROC(Receiver Operating Characteristic Area Under Curve), ECE(Expected Calibration Error), Brier Score와 같은 다양한 메트릭을 통해 평가되었습니다. AUROC는 모델의 신뢰성을 평가하는 데 사용되며, ECE는 모델의 신뢰도와 실제 정확도 간의 차이를 측정합니다.

4. **비교 결과**:
   - **Llama-3 모델**: EPICAR는 Llama-3 모델에서 STaR보다 높은 AUROC와 낮은 ECE를 기록했습니다. 예를 들어, Llama-3-3B 모델에서 EPICAR는 AUROC 0.568, ECE 0.108을 달성하여 STaR의 성능을 초월했습니다.
   - **Qwen-3 모델**: Qwen-3-8B 모델에서도 EPICAR는 AUROC 0.797, ECE 0.131을 기록하여 STaR의 AUROC 0.710, ECE 0.179를 초과했습니다. 이는 EPICAR가 신뢰성 있는 추론을 제공하는 데 효과적임을 보여줍니다.

5. **결론**: EPICAR는 기존의 자기 훈련 방법에서 발생하는 신뢰성 저하 문제를 해결하며, 모델이 자신의 신뢰도를 평가할 수 있도록 훈련함으로써 성능과 신뢰성을 동시에 향상시킬 수 있음을 입증했습니다. 이 연구는 LLM의 신뢰성을 높이는 데 중요한 기여를 하며, 향후 다양한 분야에서의 적용 가능성을 제시합니다.

---




This paper introduces a new framework called EPICAR (Epistemically Calibrated Reasoning) aimed at enhancing the reasoning capabilities of large language models (LLMs) while improving their reliability. EPICAR focuses on training models to evaluate their confidence in their reasoning outputs, thereby reducing overconfidence and better representing uncertainty.

#### Experimental Results

1. **Competing Models**: EPICAR was compared against existing models such as STaR (Self-Training Reasoner) and Slow Thinking. STaR improves model accuracy through a positive feedback loop but often negatively impacts reliability.

2. **Test Data**: Experiments were conducted on various datasets, including MATH, GSM8K, and MBPP. The MATH dataset is used for mathematical problem-solving, GSM8K evaluates out-of-distribution (OOD) generalization in mathematical reasoning, and MBPP is for code generation tasks.

3. **Metrics**: Model performance was evaluated using various metrics, including Accuracy, AUROC (Area Under the Receiver Operating Characteristic Curve), ECE (Expected Calibration Error), and Brier Score. AUROC assesses the model's reliability, while ECE measures the discrepancy between the model's confidence and actual accuracy.

4. **Comparison Results**:
   - **Llama-3 Model**: EPICAR achieved higher AUROC and lower ECE compared to STaR in the Llama-3 model. For instance, the Llama-3-3B model recorded an AUROC of 0.568 and an ECE of 0.108 with EPICAR, surpassing STaR's performance.
   - **Qwen-3 Model**: In the Qwen-3-8B model, EPICAR recorded an AUROC of 0.797 and an ECE of 0.131, exceeding STaR's AUROC of 0.710 and ECE of 0.179. This demonstrates EPICAR's effectiveness in providing reliable reasoning.

5. **Conclusion**: EPICAR addresses the reliability degradation issues inherent in traditional self-training methods and proves that training models to assess their confidence can simultaneously enhance performance and reliability. This research makes a significant contribution to improving the reliability of LLMs and suggests potential applications in various fields.


<br/>
# 예제


이 논문에서는 EPICAR(에피스테믹 캘리브레이션 추론)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대규모 언어 모델(LLM)의 추론 능력을 향상시키기 위해 설계되었습니다. EPICAR는 모델이 추론을 수행하는 방법뿐만 아니라, 언제 자신의 추론을 신뢰해야 하는지를 학습하도록 합니다. 이를 위해, 모델은 정확한 답변을 생성하는 것뿐만 아니라, 자신이 생성한 답변의 신뢰성을 평가하는 메타 인지적 자기 평가 신호를 사용합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**: 
   - **데이터셋**: MATH 데이터셋
   - **입력**: 수학 문제 (예: "5 + 7은 얼마인가?")
   - **출력**: 정답 (예: "12")
   - **추론 경로**: 모델이 문제를 해결하기 위해 생성한 단계별 과정 (예: "5와 7을 더한다. 결과는 12이다.")

2. **테스트 데이터**:
   - **데이터셋**: GSM8K (Out-of-Distribution)
   - **입력**: 수학 문제 (예: "A와 B의 합은 얼마인가? A=3, B=4")
   - **출력**: 정답 (예: "7")
   - **신뢰성 평가**: 모델이 자신의 답변이 맞는지 여부를 평가하는 과정 (예: "이 답변이 맞습니까? 예/아니오")

#### 구체적인 작업

- **작업 1**: 모델은 주어진 수학 문제에 대해 Chain-of-Thought (CoT) 방식으로 단계별로 문제를 해결합니다.
- **작업 2**: 모델은 자신의 답변에 대한 신뢰성을 평가합니다. 예를 들어, "이 답변이 맞습니까?"라는 질문에 대해 "예" 또는 "아니오"로 대답합니다.
- **작업 3**: 모델은 여러 번의 추론을 통해 생성된 답변을 집계하여 최종 답변을 결정합니다. 이 과정에서 신뢰성 점수를 고려하여 최종 답변을 선택합니다.

이러한 방식으로 EPICAR는 모델이 더 나은 추론을 수행하고, 자신의 신뢰성을 평가할 수 있도록 돕습니다.

---




This paper proposes a new framework called EPICAR (Epistemically Calibrated Reasoning) designed to enhance the reasoning capabilities of large language models (LLMs). EPICAR enables models to learn not only how to reason but also when to trust their reasoning. To achieve this, the model uses explicitly extracted meta-cognitive self-evaluation signals to assess the reliability of the answers it generates.

#### Training Data and Test Data

1. **Training Data**:
   - **Dataset**: MATH dataset
   - **Input**: Math problems (e.g., "What is 5 + 7?")
   - **Output**: Correct answers (e.g., "12")
   - **Reasoning Path**: The step-by-step process generated by the model to solve the problem (e.g., "Add 5 and 7. The result is 12.")

2. **Test Data**:
   - **Dataset**: GSM8K (Out-of-Distribution)
   - **Input**: Math problems (e.g., "What is the sum of A and B? A=3, B=4")
   - **Output**: Correct answers (e.g., "7")
   - **Reliability Assessment**: The process where the model evaluates whether its answer is correct (e.g., "Is this answer correct? Yes/No")

#### Specific Tasks

- **Task 1**: The model solves the given math problem using a Chain-of-Thought (CoT) approach, reasoning step-by-step.
- **Task 2**: The model evaluates the reliability of its answer. For example, it answers the question, "Is this answer correct?" with "Yes" or "No."
- **Task 3**: The model aggregates multiple generated answers from reasoning to determine the final answer, considering the reliability scores in the process.

Through this approach, EPICAR helps the model perform better reasoning and assess its own reliability.

<br/>
# 요약


EPICAR는 대규모 언어 모델의 추론 성능과 신뢰성을 동시에 최적화하기 위해 메타인지 자기 평가 신호를 활용하는 훈련 목표를 제안한다. 실험 결과, EPICAR는 기존의 STaR 방법에 비해 정확도와 신뢰성 모두에서 우수한 성능을 보이며, 특히 3B 및 8B 모델에서 두드러진 개선을 나타낸다. 이 방법은 코드 생성 및 수학적 추론과 같은 다양한 작업에서 일반화 가능성을 보여준다.



EPICAR proposes a training objective that leverages meta-cognitive self-evaluation signals to jointly optimize reasoning performance and reliability in large language models. Experimental results demonstrate that EPICAR outperforms the existing STaR method in both accuracy and reliability, particularly showing significant improvements in the 3B and 8B models. This approach also exhibits generalization capabilities across various tasks, including code generation and mathematical reasoning.

<br/>
# 기타

#### 다이어그램 및 피규어
1. **신뢰도 다이어그램 (Reliability Diagram)**: MATH 데이터셋에서 EPICAR와 STaR의 보정 성능을 비교한 결과, EPICAR가 더 나은 신뢰도를 보여주었다. 이는 모델이 자신의 확신을 더 잘 표현하고, 잘못된 경로에 대한 신뢰를 낮추는 데 기여한다는 것을 나타낸다.

2. **신뢰도 다이어그램 (Slow Thinking)**: 느린 사고(Slow Thinking) 행동이 EPICAR와 결합될 때, 모델의 보정 성능이 어떻게 향상되는지를 시각화하였다. EPICAR는 느린 사고 행동을 통해 더 나은 결과를 도출할 수 있음을 보여준다.

3. **신뢰도 다이어그램 (GSM8K)**: OOD(Out-of-Distribution) 상황에서의 보정 성능을 평가한 결과, EPICAR가 더 높은 AUROC를 기록하며, 모델이 새로운 데이터에 대해 더 잘 일반화할 수 있음을 나타낸다.

4. **신뢰도 다이어그램 (MBPP)**: 코드 생성 작업에서 EPICAR의 교차 도메인 강건성을 평가한 결과, EPICAR가 더 낮은 ECE를 기록하며, 이는 모델이 코드 생성에서도 신뢰성을 유지할 수 있음을 보여준다.

#### 테이블
1. **모델 성능 비교 (Llama-3 및 Qwen-3)**: EPICAR는 STaR와 비교하여 정확도와 보정에서 일관되게 우수한 성능을 보였다. 특히, 8B 모델에서 EPICAR는 가장 낮은 Brier Score를 기록하며, 이는 전반적인 예측 품질이 우수함을 나타낸다.

2. **모델 병합 결과**: 모델 병합 실험에서 EPICAR는 STaR보다 더 높은 AUROC를 기록하며, 이는 EPICAR가 더 나은 신뢰성을 제공함을 나타낸다. 특히, 4B 및 8B 모델에서 EPICAR의 성능이 두드러진다.

3. **신뢰도 분석 (AUROC 및 ECE)**: EPICAR는 다양한 K 값에서 STaR보다 더 높은 AUROC를 기록하며, ECE는 낮은 값을 유지하였다. 이는 EPICAR가 신뢰도와 정확도 간의 균형을 잘 맞추고 있음을 보여준다.

#### 어펜딕스
1. **AID (Adaptive Injection Decoding)**: AID의 중요성을 강조하는 실험 결과, AID가 없을 경우 모델의 성능이 크게 저하됨을 보여준다. 이는 AID가 형식 오류로 인한 잘못된 신호를 방지하는 데 중요한 역할을 한다는 것을 나타낸다.

2. **프롬프트 민감도 분석**: 다양한 자기 평가 템플릿을 사용하여 EPICAR의 신뢰도와 보정 성능을 평가한 결과, EPICAR가 더 낮은 변동성을 보이며, 이는 모델이 자기 평가 신호를 잘 내재화하고 있음을 나타낸다.

---



#### Diagrams and Figures
1. **Reliability Diagram**: The comparison of calibration performance between EPICAR and STaR on the MATH dataset shows that EPICAR demonstrates better reliability. This indicates that the model is better at expressing its confidence and reducing trust in incorrect paths.

2. **Reliability Diagram (Slow Thinking)**: This visualizes how the performance of EPICAR improves when combined with slow thinking behaviors. EPICAR shows that it can achieve better results through these cognitive patterns.

3. **Reliability Diagram (GSM8K)**: Evaluating calibration performance in an out-of-distribution (OOD) scenario, EPICAR achieves a higher AUROC, indicating that the model generalizes better to new data.

4. **Reliability Diagram (MBPP)**: In the code generation task, the evaluation of EPICAR's cross-domain robustness shows that it records a lower ECE, demonstrating that the model can maintain reliability even in code generation.

#### Tables
1. **Model Performance Comparison (Llama-3 and Qwen-3)**: EPICAR consistently outperforms STaR in both accuracy and calibration. Notably, in the 8B model, EPICAR achieves the lowest Brier Score, indicating superior overall predictive quality.

2. **Model Merging Results**: In the model merging experiments, EPICAR shows higher AUROC compared to STaR, indicating that EPICAR provides better reliability. This is particularly evident in the 4B and 8B models.

3. **Reliability Analysis (AUROC and ECE)**: EPICAR achieves higher AUROC than STaR across various K values while maintaining lower ECE. This demonstrates that EPICAR effectively balances reliability and accuracy.

#### Appendix
1. **AID (Adaptive Injection Decoding)**: The results highlight the importance of AID, showing significant performance degradation when AID is removed. This indicates that AID plays a crucial role in preventing noise from formatting errors.

2. **Prompt Sensitivity Analysis**: The evaluation of EPICAR's reliability and calibration performance using diverse self-evaluation templates shows that EPICAR exhibits lower variance, indicating that the model effectively internalizes self-evaluation signals.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Yeom2026EPICAR,
  author    = {Jewon Yeom and Jaewon Sok and Seonghyeon Park and Jeongjae Park and Taesup Kim},
  title     = {EPICAR: Knowing What You Don’t Know Matters for Better Reasoning in LLMs},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {22414--22443},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  
}
```

### Chicago Style Citation

Yeom, Jewon, Jaewon Sok, Seonghyeon Park, Jeongjae Park, and Taesup Kim. 2026. "EPICAR: Knowing What You Don’t Know Matters for Better Reasoning in LLMs." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 22414–22443. Association for Computational Linguistics.
    