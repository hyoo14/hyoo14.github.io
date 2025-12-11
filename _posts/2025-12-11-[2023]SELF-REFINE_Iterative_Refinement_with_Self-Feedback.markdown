---
layout: post
title:  "[2023]SELF-REFINE: Iterative Refinement with Self-Feedback"
date:   2025-12-11 21:26:30 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: SELF-REFINE는 LLM을 사용하여 초기 출력을 생성하고, 그 출력에 대한 피드백을 제공한 후, 이 피드백을 바탕으로 출력을 반복적으로 정제(이 방법 제안)     


짧은 요약(Abstract) :


이 논문에서는 SELF-REFINE이라는 새로운 접근 방식을 소개합니다. 이 방법은 대형 언어 모델(LLM)이 초기 출력물을 개선하기 위해 반복적인 피드백과 정제를 통해 성능을 향상시키는 방법입니다. SELF-REFINE는 LLM을 사용하여 초기 출력을 생성하고, 그 출력에 대한 피드백을 제공한 후, 이 피드백을 바탕으로 출력을 반복적으로 정제합니다. 이 과정은 감독된 학습 데이터나 추가적인 훈련 없이 진행되며, 단일 LLM을 생성기, 정제기 및 피드백 제공자로 사용합니다. 다양한 7개 작업에서 SELF-REFINE를 평가한 결과, 인간과 자동화된 메트릭 모두에서 SELF-REFINE를 통해 생성된 출력물이 기존의 단일 단계 생성 방식보다 평균적으로 약 20% 향상된 성능을 보였습니다. 이 연구는 최신 LLM인 GPT-4와 같은 모델도 테스트 시점에서 SELF-REFINE를 통해 추가적인 개선이 가능하다는 것을 보여줍니다.



This paper introduces a novel approach called SELF-REFINE, which enhances the performance of large language models (LLMs) through iterative feedback and refinement of initial outputs. The main idea is to generate an initial output using an LLM, then provide feedback on that output, and subsequently refine it based on the feedback. This process does not require supervised training data or additional training and utilizes a single LLM as the generator, refiner, and feedback provider. Evaluating SELF-REFINE across seven diverse tasks shows that outputs generated with this method are preferred by both humans and automatic metrics, achieving an average performance improvement of approximately 20% over conventional one-step generation. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test time using this simple, standalone approach.


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



SELF-REFINE는 대형 언어 모델(LLM)의 초기 출력을 반복적으로 개선하기 위한 접근 방식으로, 인간의 글쓰기 수정 과정을 모방하여 설계되었습니다. 이 방법은 LLM이 스스로 피드백을 제공하고 이를 바탕으로 출력을 개선하는 방식으로 작동합니다. SELF-REFINE의 주요 구성 요소는 다음과 같습니다:

1. **모델(M)**: SELF-REFINE는 GPT-3.5, GPT-4와 같은 최신 LLM을 사용하여 초기 출력을 생성하고, 피드백을 제공하며, 출력을 개선합니다. 이 모델은 대규모 데이터셋에서 사전 훈련되어 있으며, 다양한 자연어 처리 작업에 대한 높은 성능을 보입니다.

2. **피드백 단계**: 초기 출력이 생성된 후, 동일한 모델이 이 출력을 평가하여 피드백을 생성합니다. 이 피드백은 출력의 여러 측면(예: 관련성, 정보성, 흥미로움 등)을 평가하며, 각 측면에 대한 점수를 제공합니다. 피드백은 구체적이고 실행 가능한 형태로 제공되어야 하며, 이는 모델이 개선할 수 있는 방향을 제시합니다.

3. **개선 단계**: 피드백이 제공된 후, 모델은 이를 바탕으로 초기 출력을 수정합니다. 이 과정은 반복적으로 이루어지며, 모델은 이전의 피드백과 출력을 기억하여 점진적으로 품질을 향상시킵니다. 이 단계에서 모델은 피드백을 반영하여 더 나은 출력을 생성하는 데 집중합니다.

4. **반복적 과정**: SELF-REFINE는 피드백과 개선 단계를 여러 번 반복하여 최종 출력을 생성합니다. 이 과정은 사전 정의된 중지 조건이 충족될 때까지 계속됩니다. 예를 들어, 특정 횟수의 반복 후 또는 피드백이 더 이상 개선되지 않을 때 중지할 수 있습니다.

5. **훈련 데이터**: SELF-REFINE는 추가적인 감독 학습 데이터나 강화 학습을 필요로 하지 않습니다. 대신, 모델은 몇 가지 예시를 통해 피드백과 개선을 위한 프롬프트를 학습합니다. 이는 모델이 다양한 작업에 대해 적응할 수 있도록 합니다.

SELF-REFINE는 대화 응답 생성, 코드 최적화, 수학적 추론 등 다양한 작업에서 평가되었으며, 기존의 단일 생성 방식보다 평균적으로 20% 이상의 성능 향상을 보여주었습니다. 이 방법은 LLM이 초기 출력에서 최적의 결과를 생성하지 못할 때 유용하며, 반복적인 피드백과 개선을 통해 더 나은 출력을 얻을 수 있는 효과적인 방법임을 입증하였습니다.

---




SELF-REFINE is an approach designed to iteratively improve the initial outputs of large language models (LLMs), mimicking the human process of text refinement. This method operates by allowing the LLM to provide self-feedback and refine its outputs based on that feedback. The main components of SELF-REFINE are as follows:

1. **Model (M)**: SELF-REFINE utilizes state-of-the-art LLMs such as GPT-3.5 and GPT-4 to generate initial outputs, provide feedback, and refine those outputs. These models are pre-trained on large datasets and demonstrate high performance across various natural language processing tasks.

2. **Feedback Step**: After generating an initial output, the same model evaluates this output to generate feedback. This feedback assesses multiple aspects of the output (e.g., relevance, informativeness, interest) and provides scores for each aspect. The feedback is expected to be specific and actionable, guiding the model on how to improve.

3. **Refinement Step**: Following the feedback, the model refines the initial output based on the provided feedback. This process is iterative, with the model retaining memory of previous feedback and outputs to gradually enhance quality. In this step, the model focuses on generating better outputs by incorporating the feedback.

4. **Iterative Process**: SELF-REFINE alternates between the feedback and refinement steps multiple times to produce the final output. This process continues until a predefined stopping condition is met, such as reaching a specified number of iterations or when feedback no longer leads to improvements.

5. **Training Data**: SELF-REFINE does not require additional supervised training data or reinforcement learning. Instead, the model learns prompts for feedback and refinement through a few examples. This allows the model to adapt to various tasks effectively.

SELF-REFINE has been evaluated across diverse tasks, including dialogue response generation, code optimization, and mathematical reasoning, showing an average performance improvement of over 20% compared to conventional one-step generation methods. This approach proves useful when LLMs do not generate optimal outputs on their first attempt, demonstrating an effective way to obtain better outputs through iterative self-feedback and refinement.


<br/>
# Results



SELF-REFINE는 다양한 자연어 생성 작업에서 성능을 향상시키기 위해 설계된 접근 방식으로, 여러 경쟁 모델과 비교하여 그 효과를 입증했습니다. 이 연구에서는 7개의 다양한 작업을 평가하였으며, 각 작업에서 SELF-REFINE의 성능을 기존의 강력한 언어 모델(GPT-3.5, GPT-4 등)과 비교했습니다.

1. **작업 및 데이터셋**: SELF-REFINE는 다음과 같은 작업에서 평가되었습니다:
   - 감정 전환 (Sentiment Reversal)
   - 대화 응답 생성 (Dialogue Response Generation)
   - 코드 최적화 (Code Optimization)
   - 코드 가독성 개선 (Code Readability Improvement)
   - 수학 추론 (Math Reasoning)
   - 약어 생성 (Acronym Generation)
   - 제약 생성 (Constrained Generation)

2. **경쟁 모델**: SELF-REFINE는 다음과 같은 모델과 비교되었습니다:
   - GPT-3.5 (text-davinci-003)
   - CHATGPT (gpt-3.5-turbo)
   - GPT-4

3. **메트릭**: 각 작업의 성능은 다음과 같은 메트릭을 사용하여 평가되었습니다:
   - 감정 전환 및 대화 응답 생성에서는 인간의 선호도 비율을 측정했습니다.
   - 코드 최적화에서는 최적화된 프로그램의 비율을 측정했습니다.
   - 수학 추론에서는 문제 해결 비율을 사용했습니다.

4. **비교 결과**: SELF-REFINE는 모든 작업에서 기존 모델에 비해 일관되게 성능이 향상되었습니다. 예를 들어, 감정 전환 작업에서 SELF-REFINE는 GPT-4에 비해 32.4%의 선호도 향상을 보였고, 대화 응답 생성에서는 49.2%의 향상을 기록했습니다. 코드 최적화 작업에서는 8.7%의 성능 향상을 보였습니다.

5. **결론**: SELF-REFINE는 기존의 강력한 언어 모델을 사용하여 추가적인 훈련 없이도 성능을 향상시킬 수 있는 효과적인 방법임을 입증했습니다. 이 연구는 SELF-REFINE가 다양한 자연어 생성 작업에서 유용하게 적용될 수 있음을 보여주며, 향후 연구에 기여할 수 있는 가능성을 제시합니다.

---




SELF-REFINE is an approach designed to enhance performance across various natural language generation tasks, demonstrating its effectiveness through comparisons with several competitive models. This study evaluated SELF-REFINE across seven diverse tasks, comparing its performance against existing strong language models (GPT-3.5, GPT-4, etc.).

1. **Tasks and Datasets**: SELF-REFINE was evaluated on the following tasks:
   - Sentiment Reversal
   - Dialogue Response Generation
   - Code Optimization
   - Code Readability Improvement
   - Math Reasoning
   - Acronym Generation
   - Constrained Generation

2. **Competitive Models**: SELF-REFINE was compared against:
   - GPT-3.5 (text-davinci-003)
   - CHATGPT (gpt-3.5-turbo)
   - GPT-4

3. **Metrics**: The performance of each task was evaluated using the following metrics:
   - For sentiment reversal and dialogue response generation, human preference rates were measured.
   - For code optimization, the percentage of optimized programs was measured.
   - For math reasoning, the problem-solving rate was used.

4. **Comparison Results**: SELF-REFINE consistently improved performance across all tasks compared to existing models. For instance, in the sentiment reversal task, SELF-REFINE showed a 32.4% improvement in preference over GPT-4, while in dialogue response generation, it recorded a 49.2% improvement. In code optimization, a performance increase of 8.7% was observed.

5. **Conclusion**: SELF-REFINE has proven to be an effective method for enhancing the performance of existing strong language models without requiring additional training. This research demonstrates the applicability of SELF-REFINE across various natural language generation tasks and suggests potential contributions to future research.


<br/>
# 예제



SELF-REFINE는 대형 언어 모델(LLM)이 자신의 출력을 반복적으로 피드백하고 개선하는 방법을 제안하는 접근 방식입니다. 이 방법은 여러 가지 작업에 적용될 수 있으며, 각 작업에 대해 구체적인 입력과 출력을 설정합니다. 아래는 SELF-REFINE의 몇 가지 예시와 함께 트레이닝 데이터와 테스트 데이터의 구체적인 입력 및 출력에 대한 설명입니다.

#### 1. 감정 전환 (Sentiment Reversal)
- **작업 설명**: 주어진 리뷰의 감정을 반전시키는 작업입니다. 예를 들어, 긍정적인 리뷰를 부정적으로 바꾸거나 그 반대의 경우입니다.
- **트레이닝 데이터 예시**:
  - **입력**: "이 음식은 정말 맛있었습니다! 다음에 또 오고 싶어요."
  - **출력**: "이 음식은 정말 맛없었습니다! 다시는 오고 싶지 않아요."
- **테스트 데이터 예시**:
  - **입력**: "이 호텔은 환상적이었습니다! 서비스가 훌륭했어요."
  - **출력**: "이 호텔은 끔찍했습니다! 서비스가 형편없었어요."

#### 2. 대화 응답 생성 (Dialogue Response Generation)
- **작업 설명**: 주어진 대화 맥락에 따라 적절한 응답을 생성하는 작업입니다.
- **트레이닝 데이터 예시**:
  - **입력**: "A: 당신의 취미는 무엇인가요? B: 저는 주로 친구들과 시간을 보냅니다."
  - **출력**: "A: 정말 재미있겠네요! 어떤 활동을 주로 하시나요?"
- **테스트 데이터 예시**:
  - **입력**: "A: 요즘 어떤 음악을 듣고 있나요? B: 다양한 음악을 듣고 있어요."
  - **출력**: "A: 어떤 아티스트의 음악이 가장 좋나요?"

#### 3. 코드 최적화 (Code Optimization)
- **작업 설명**: 주어진 코드의 성능을 개선하는 작업입니다.
- **트레이닝 데이터 예시**:
  - **입력**: 
    ```python
    def slow_function(n):
        total = 0
        for i in range(n):
            total += i
        return total
    ```
  - **출력**: 
    ```python
    def optimized_function(n):
        return n * (n - 1) // 2
    ```
- **테스트 데이터 예시**:
  - **입력**: 
    ```python
    def inefficient_function(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    ```
  - **출력**: 
    ```python
    def optimized_factorial(n):
        from math import factorial
        return factorial(n)
    ```




SELF-REFINE is an approach that allows large language models (LLMs) to iteratively provide self-feedback and refine their outputs. This method can be applied to various tasks, and specific inputs and outputs are set for each task. Below are examples of SELF-REFINE along with detailed descriptions of the training and testing data inputs and outputs.

#### 1. Sentiment Reversal
- **Task Description**: The task is to reverse the sentiment of a given review. For example, changing a positive review to a negative one or vice versa.
- **Training Data Example**:
  - **Input**: "The food was amazing! I want to come back again."
  - **Output**: "The food was terrible! I never want to come back."
- **Testing Data Example**:
  - **Input**: "This hotel was fantastic! The service was excellent."
  - **Output**: "This hotel was awful! The service was terrible."

#### 2. Dialogue Response Generation
- **Task Description**: The task is to generate an appropriate response based on the given dialogue context.
- **Training Data Example**:
  - **Input**: "A: What are your hobbies? B: I mostly spend time with friends."
  - **Output**: "A: That sounds fun! What activities do you usually do?"
- **Testing Data Example**:
  - **Input**: "A: What music are you listening to these days? B: I listen to all sorts of music."
  - **Output**: "A: Which artist do you like the most?"

#### 3. Code Optimization
- **Task Description**: The task is to improve the performance of a given piece of code.
- **Training Data Example**:
  - **Input**: 
    ```python
    def slow_function(n):
        total = 0
        for i in range(n):
            total += i
        return total
    ```
  - **Output**: 
    ```python
    def optimized_function(n):
        return n * (n - 1) // 2
    ```
- **Testing Data Example**:
  - **Input**: 
    ```python
    def inefficient_function(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    ```
  - **Output**: 
    ```python
    def optimized_factorial(n):
        from math import factorial
        return factorial(n)
    ```

<br/>
# 요약


SELF-REFINE는 대형 언어 모델이 초기 출력을 생성한 후, 스스로 피드백을 제공하고 이를 바탕으로 출력을 반복적으로 개선하는 방법론이다. 이 방법은 대화 응답 생성, 코드 최적화 등 다양한 작업에서 기존 모델보다 평균 20% 이상의 성능 향상을 보여주었다. 예를 들어, 대화 응답 생성 작업에서 SELF-REFINE를 적용한 모델은 인간 평가에서 74.6%의 선호도를 기록했다.



SELF-REFINE is a methodology where large language models generate initial outputs, provide self-feedback, and iteratively improve the outputs based on that feedback. This approach demonstrated an average performance improvement of over 20% compared to existing models across various tasks, including dialogue response generation and code optimization. For instance, in the dialogue response generation task, the model using SELF-REFINE achieved a preference score of 74.6% in human evaluations.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **SELF-REFINE 프로세스 다이어그램**: 이 다이어그램은 SELF-REFINE의 전반적인 프로세스를 시각적으로 설명합니다. 초기 출력 생성, 피드백 생성, 그리고 피드백을 기반으로 한 출력 개선의 반복 과정을 보여줍니다. 이 과정은 LLM이 스스로 피드백을 제공하고 이를 통해 출력을 개선할 수 있음을 강조합니다.
   - **예시 피규어**: 다양한 작업에서 SELF-REFINE의 적용 예시를 보여주는 피규어들이 포함되어 있습니다. 예를 들어, 대화 응답 생성 및 코드 최적화 작업에서 초기 출력과 피드백, 개선된 출력을 비교하여 SELF-REFINE의 효과를 시각적으로 나타냅니다.

2. **테이블**
   - **성능 비교 테이블**: SELF-REFINE의 성능을 다양한 작업에서 기존 모델과 비교한 테이블이 있습니다. 이 테이블은 SELF-REFINE이 기존 모델에 비해 얼마나 개선되었는지를 수치적으로 보여줍니다. 예를 들어, 감정 반전 작업에서 SELF-REFINE이 기존 모델보다 32.4% 더 높은 성능을 보였다는 결과가 있습니다.
   - **인간 평가 결과**: 인간 평가를 통해 SELF-REFINE의 출력이 기존 모델보다 얼마나 선호되는지를 보여주는 테이블이 있습니다. 이 결과는 SELF-REFINE의 출력이 더 매력적이고 유용하다는 것을 나타냅니다.

3. **어펜딕스**
   - **어펜딕스 V**: 다양한 작업에 대한 프롬프트 예시와 함께 SELF-REFINE의 세부 구현을 설명합니다. 각 작업에 대한 초기 생성, 피드백, 개선 단계에서 사용된 프롬프트를 포함하여, SELF-REFINE의 적용 방법을 구체적으로 보여줍니다.
   - **어펜딕스 K**: SELF-REFINE의 피드백 및 개선 과정에서 발생한 오류 분석을 포함합니다. 이 분석은 피드백의 질이 SELF-REFINE의 성능에 미치는 영향을 강조하며, 피드백이 구체적이고 실행 가능할수록 더 나은 결과를 도출할 수 있음을 보여줍니다.

### Insights
- **SELF-REFINE의 효과**: SELF-REFINE는 LLM이 스스로 피드백을 제공하고 이를 통해 출력을 개선할 수 있는 강력한 방법임을 보여줍니다. 다양한 작업에서 성능이 향상되었으며, 특히 대화 응답 생성 및 감정 반전 작업에서 두드러진 성과를 보였습니다.
- **피드백의 중요성**: 피드백의 질이 SELF-REFINE의 성능에 큰 영향을 미친다는 점이 강조되었습니다. 구체적이고 실행 가능한 피드백이 제공될 때, LLM은 더 나은 출력을 생성할 수 있습니다.
- **다양한 적용 가능성**: SELF-REFINE는 코드 최적화, 대화 응답 생성, 감정 반전 등 다양한 작업에 적용 가능하며, 이는 LLM의 활용 범위를 넓히는 데 기여할 수 있습니다.

---



1. **Diagrams and Figures**
   - **SELF-REFINE Process Diagram**: This diagram visually explains the overall process of SELF-REFINE, illustrating the iterative steps of generating initial outputs, providing feedback, and refining outputs based on that feedback. It emphasizes the capability of LLMs to self-generate feedback and improve their outputs.
   - **Example Figures**: Figures showcasing examples of applying SELF-REFINE across various tasks, such as dialogue response generation and code optimization, compare initial outputs, feedback, and improved outputs, visually demonstrating the effectiveness of SELF-REFINE.

2. **Tables**
   - **Performance Comparison Table**: A table comparing the performance of SELF-REFINE across various tasks against existing models. This table quantitatively shows how much SELF-REFINE improves upon previous models, such as a 32.4% increase in performance in sentiment reversal tasks.
   - **Human Evaluation Results**: A table showing the results of human evaluations indicating how much more preferred the outputs from SELF-REFINE are compared to those from baseline models. This result suggests that outputs from SELF-REFINE are more engaging and useful.

3. **Appendices**
   - **Appendix V**: Provides examples of prompts used in various tasks along with detailed implementations of SELF-REFINE. It includes prompts for initial generation, feedback, and refinement, illustrating how SELF-REFINE can be applied.
   - **Appendix K**: Contains an error analysis of the feedback and refinement process in SELF-REFINE. This analysis highlights the impact of feedback quality on the performance of SELF-REFINE, showing that more specific and actionable feedback leads to better results.

### Insights
- **Effectiveness of SELF-REFINE**: SELF-REFINE demonstrates a powerful method for LLMs to provide self-feedback and improve their outputs. It shows significant performance improvements across various tasks, particularly in dialogue response generation and sentiment reversal.
- **Importance of Feedback**: The quality of feedback significantly impacts the performance of SELF-REFINE. When feedback is specific and actionable, LLMs can generate better outputs.
- **Diverse Applicability**: SELF-REFINE can be applied to various tasks, including code optimization, dialogue response generation, and sentiment reversal, contributing to a broader utilization of LLMs.

<br/>
# refer format:
### BibTeX 


```bibtex
@inproceedings{madaan2023selfrefine,
  title={SELF-REFINE: Iterative Refinement with Self-Feedback},
  author={Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Nouha Dziri and Shrimai Prabhumoye and Yiming Yang and Shashank Gupta and Bodhisattwa Prasad Majumder and Katherine Hermann and Sean Welleck and Amir Yazdanbakhsh and Peter Clark},
  booktitle={Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)},
  year={2023},
}
```

### 시카고 스타일

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. "SELF-REFINE: Iterative Refinement with Self-Feedback." In *Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)*, 2023. 
