---
layout: post
title:  "[2025]Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection"
date:   2025-08-19 21:27:08 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Exemplar-Guided Reflection with Memory (ERM) 메커니즘을 제안하여 자동 프롬프트 최적화를 효율적이고 정확하게 수행

ERM은 다음처럼 구성: 예시 기반 반영(잘못된 샘플을 기반으로 LLM이 생성한 예시를 통해 문제를 해결하는 방법을 제시)+피드백 메모리(최적화 과정에서 생성된 피드백을 저장하고 각 피드백에 우선 순위 점수를 부여, 유용한 피드백을 효율적으로 검색하고 활용할 수 있도록 하여 최적화 과정을 가속화) + 예시 공장(예시를 저장하고 평가하여 예시가 특정 질문에 미치는 영향을 분석)


짧은 요약(Abstract) :


이 논문에서는 대형 언어 모델(LLM)의 생성 품질을 향상시키기 위한 자동 프롬프트 최적화 방법을 제안합니다. 최근 연구들은 오류 사례에서 생성된 피드백을 활용하여 프롬프트 최적화를 안내하는 방법을 사용하고 있습니다. 그러나 기존 방법들은 현재 단계에서의 피드백만을 활용하고, 역사적이고 선택되지 않은 피드백을 무시하여 최적화 과정에서 비효율성을 초래합니다. 또한, 예시의 선택은 일반적인 의미적 관계만을 고려하여 최적의 성능을 보장하지 못합니다. 본 연구에서는 메모리 메커니즘을 활용한 예시 안내 반영(Exemplar-Guided Reflection with Memory, ERM) 방법을 제안하여 보다 효율적이고 정확한 프롬프트 최적화를 실현합니다. 이 방법은 예시를 통해 생성된 피드백을 추가로 안내하고, 역사적 피드백 정보를 최대한 활용하여 더 효과적인 예시 검색을 지원합니다. 실험 결과, 제안된 방법이 이전의 최첨단 기술보다 우수한 성능을 보이며, 최적화 단계 수를 절반으로 줄이는 성과를 달성했습니다.



This paper proposes an automatic prompt optimization method aimed at enhancing the generation quality of large language models (LLMs). Recent works utilize feedback generated from erroneous cases to guide the prompt optimization process. However, existing methods only leverage feedback at the current step, ignoring historical and unselected feedback, which leads to inefficiencies in the optimization process. Additionally, the selection of exemplars only considers general semantic relationships, which may not guarantee optimal performance. In this work, we introduce an Exemplar-Guided Reflection with Memory mechanism (ERM) to achieve more efficient and accurate prompt optimization. This method further guides feedback generation through exemplars and fully utilizes historical feedback information to support more effective exemplar retrieval. Empirical evaluations show that our method surpasses previous state-of-the-art approaches while reducing the number of optimization steps by half.


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



이 논문에서 제안하는 방법은 "Exemplar-Guided Reflection with Memory" (ERM)라는 새로운 접근 방식을 통해 효율적이고 정확한 프롬프트 최적화를 달성하는 것입니다. 이 방법은 세 가지 주요 구성 요소로 이루어져 있습니다: 

1. **Exemplar-Guided Reflection (예시 기반 반영)**: 이 단계에서는 지침 메타 프롬프트를 사용하여 LLM(대형 언어 모델)이 잘못된 샘플을 선택하고 이에 대한 상세한 해결 과정을 제공하도록 유도합니다. 이를 통해 LLM은 더 유용한 피드백을 생성할 수 있습니다. 예를 들어, 잘못된 샘플을 기반으로 LLM이 생성한 예시를 통해 문제를 해결하는 방법을 제시합니다.

2. **Feedback Memory (피드백 메모리)**: 이 구성 요소는 최적화 과정에서 생성된 피드백을 저장하고 각 피드백에 우선 순위 점수를 부여합니다. 피드백 메모리는 유용한 피드백을 효율적으로 검색하고 활용할 수 있도록 하여 최적화 과정을 가속화합니다. 피드백의 우선 순위 점수는 새로운 프롬프트를 평가한 후 업데이트되며, 성능이 향상되면 점수가 증가하고 그렇지 않으면 감소합니다.

3. **Exemplar Factory (예시 공장)**: 이 단계에서는 예시를 저장하고 평가하여 예시가 특정 질문에 미치는 영향을 분석합니다. 예시 메모리 저장소에 저장된 예시는 우선 순위 점수를 부여받고, 최적화 과정에서 예시를 검색하여 예측의 정확성을 높이는 데 사용됩니다. 

이 방법은 기존의 피드백 기반 방법들이 가지는 한계를 극복하고, 과거의 피드백을 효과적으로 활용하여 프롬프트 최적화의 효율성과 정확성을 높입니다. 실험 결과, ERM은 다양한 데이터셋에서 이전의 최첨단 방법들보다 우수한 성능을 보였으며, 최적화 단계 수를 대폭 줄일 수 있었습니다.




The method proposed in this paper is called "Exemplar-Guided Reflection with Memory" (ERM), which aims to achieve efficient and accurate prompt optimization. This method consists of three main components:

1. **Exemplar-Guided Reflection**: In this stage, an instructive meta-prompt is used to guide the LLM (Large Language Model) to select erroneous samples and provide detailed solution processes for them. This allows the LLM to generate more informative feedback. For instance, the LLM generates examples based on the erroneous samples to illustrate how to solve the problem.

2. **Feedback Memory**: This component stores the feedback generated during the optimization process and assigns a priority score to each piece of feedback. The feedback memory enables efficient retrieval and utilization of valuable feedback, accelerating the optimization process. The priority scores of the feedback are updated after evaluating the new prompts, increasing if performance improves and decreasing otherwise.

3. **Exemplar Factory**: In this stage, exemplars are stored and evaluated to analyze their impact on specific questions. Each exemplar in the memory storage is assigned a priority score, and during the optimization process, exemplars are retrieved to enhance prediction accuracy.

This method overcomes the limitations of existing feedback-based approaches by effectively utilizing historical feedback, thereby improving both the efficiency and accuracy of prompt optimization. Empirical results demonstrate that ERM outperforms previous state-of-the-art methods across various datasets while significantly reducing the number of optimization steps.


<br/>
# Results



이 논문에서는 Exemplar-Guided Reflection with Memory (ERM) 메커니즘을 제안하여 자동 프롬프트 최적화의 효율성과 정확성을 향상시키고자 하였습니다. ERM은 두 가지 주요 메모리 구조인 피드백 메모리와 예시 메모리를 활용하여 과거의 피드백과 예시를 효과적으로 저장하고 검색합니다. 이를 통해 모델이 더 나은 프롬프트를 생성할 수 있도록 돕습니다.

#### 실험 결과
논문에서는 ERM을 여러 데이터셋에서 기존의 최첨단 모델들과 비교하여 성능을 평가하였습니다. 다음은 주요 결과입니다:

1. **LIAR 데이터셋**: ERM은 F1 점수에서 68.6을 기록하여, ProTeGi 모델보다 10.1 포인트 향상되었습니다. ProTeGi는 58.5의 F1 점수를 기록했습니다.
2. **BBH-Navigate 데이터셋**: ERM은 86.1의 F1 점수를 기록하여, 이전 모델들보다 우수한 성능을 보였습니다.
3. **ETHOS 데이터셋**: ERM은 98.0의 F1 점수를 기록하여, 기존 모델들보다 높은 정확도를 달성했습니다.
4. **WebNLG 데이터셋**: ERM은 Rouge-L 점수에서 59.6을 기록하여, ProTeGi보다 3.9 포인트 향상되었습니다.
5. **GSM8K 데이터셋**: ERM은 93.3의 정확도를 기록하여, 이전 모델들보다 더 나은 성능을 보였습니다.
6. **WSC 데이터셋**: ERM은 86.0의 정확도를 기록하여, 기존 모델들보다 높은 성능을 보였습니다.

이러한 결과는 ERM이 피드백 메모리와 예시 메모리를 통해 과거의 정보를 효과적으로 활용하여 프롬프트 최적화의 효율성을 높이고, 더 나은 예시를 통해 모델의 예측 정확도를 향상시킬 수 있음을 보여줍니다. ERM은 최적화 단계 수를 대폭 줄이면서도 성능을 향상시켰습니다.




In this paper, the authors propose the Exemplar-Guided Reflection with Memory (ERM) mechanism to enhance the efficiency and accuracy of automatic prompt optimization. ERM utilizes two main memory structures, Feedback Memory and Exemplar Memory, to effectively store and retrieve past feedback and exemplars, thereby assisting the model in generating better prompts.

#### Experimental Results
The paper evaluates ERM's performance by comparing it with existing state-of-the-art models across several datasets. Here are the key results:

1. **LIAR Dataset**: ERM achieved an F1 score of 68.6, surpassing the ProTeGi model by 10.1 points, which recorded an F1 score of 58.5.
2. **BBH-Navigate Dataset**: ERM recorded an F1 score of 86.1, demonstrating superior performance compared to previous models.
3. **ETHOS Dataset**: ERM achieved an F1 score of 98.0, attaining higher accuracy than existing models.
4. **WebNLG Dataset**: ERM recorded a Rouge-L score of 59.6, improving by 3.9 points over ProTeGi.
5. **GSM8K Dataset**: ERM achieved an accuracy of 93.3, outperforming previous models.
6. **WSC Dataset**: ERM recorded an accuracy of 86.0, showing higher performance than existing models.

These results indicate that ERM effectively utilizes past information through Feedback Memory and Exemplar Memory to enhance the efficiency of prompt optimization and improve prediction accuracy through better exemplars. ERM significantly reduced the number of optimization steps while improving performance.


<br/>
# 예제



이 논문에서는 "Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection"이라는 제목으로, 대형 언어 모델(LLM)의 성능을 향상시키기 위한 자동 프롬프트 최적화 방법을 제안합니다. 이 방법은 특히 메모리 메커니즘을 활용하여 과거의 피드백과 예시를 효과적으로 활용하는 데 중점을 둡니다.

#### 1. 데이터셋 및 태스크
이 연구에서는 여러 데이터셋을 사용하여 모델의 성능을 평가합니다. 주요 데이터셋은 다음과 같습니다:

- **LIAR**: 진위 여부를 판단하는 데이터셋으로, 3681개의 훈련 샘플과 461개의 테스트 샘플로 구성됩니다. 각 샘플은 진술과 그에 대한 진위 레이블(참/거짓)을 포함합니다.
- **BBH-Navigate**: 96개의 훈련 샘플과 144개의 테스트 샘플로 구성된 데이터셋으로, 주어진 지침을 따라 시작점으로 돌아오는지를 판단하는 태스크입니다.
- **ETHOS**: 440개의 훈련 샘플과 200개의 테스트 샘플로 구성된 데이터셋으로, 혐오 발언을 탐지하는 태스크입니다.
- **ArSarcasm**: 8437개의 훈련 샘플과 2110개의 테스트 샘플로 구성된 아랍어의 풍자 탐지 데이터셋입니다.
- **WebNLG**: 200개의 훈련 샘플과 300개의 테스트 샘플로 구성된 데이터셋으로, 주어진 트리플을 자연어 문장으로 변환하는 태스크입니다.
- **GSM8K**: 200개의 훈련 샘플과 300개의 테스트 샘플로 구성된 데이터셋으로, 수학 문제를 해결하는 태스크입니다.
- **WSC**: 100개의 훈련 샘플과 150개의 테스트 샘플로 구성된 데이터셋으로, 주어진 문장에서 대명사가 가리키는 대상을 찾는 태스크입니다.

#### 2. 예시
각 데이터셋에 대한 예시는 다음과 같습니다:

- **LIAR 데이터셋 예시**:
  - **입력**: "A study of private bail bond systems showed that Wisconsin has a higher no-show rate than other states of defendants skipping court appearances."
  - **출력**: "Yes" (이 진술은 거짓입니다.)

- **BBH-Navigate 데이터셋 예시**:
  - **입력**: "Move 7 steps forward, then turn left and move 3 steps."
  - **출력**: "No" (시작점으로 돌아오지 않습니다.)

- **ETHOS 데이터셋 예시**:
  - **입력**: "All immigrants are criminals."
  - **출력**: "Yes" (이 문장은 혐오 발언입니다.)

- **ArSarcasm 데이터셋 예시**:
  - **입력**: "Oh great, another rainy day. Just what I needed!"
  - **출력**: "Yes" (이 문장은 풍자입니다.)

- **WebNLG 데이터셋 예시**:
  - **입력**: "Anders_Osborne | associatedBand/associatedMusicalArtist | Billy_Iuso"
  - **출력**: "Rock musician Anders Osborne has worked with the band Galactic and also with the musical artists Tab Benoit and Billy Iuso."

- **GSM8K 데이터셋 예시**:
  - **입력**: "If a burger costs $5 and fries cost $3, how much do I spend on 2 burgers and 2 fries?"
  - **출력**: "16" (총 비용은 $16입니다.)

- **WSC 데이터셋 예시**:
  - **입력**: "The sculpture rolled off the shelf because it wasn’t level. What does the pronoun 'it' refer to?"
  - **출력**: "B" (대명사 'it'은 선반을 가리킵니다.)

이와 같은 방식으로, 각 데이터셋은 특정 태스크에 대한 입력과 출력의 예시를 제공하며, 연구에서는 이러한 예시를 통해 제안된 방법의 효과를 평가합니다.

---




This paper presents a method titled "Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection," which focuses on automatic prompt optimization to enhance the performance of large language models (LLMs). The method emphasizes the use of memory mechanisms to effectively utilize past feedback and examples.

#### 1. Datasets and Tasks
The study evaluates the model's performance using several datasets. The main datasets include:

- **LIAR**: A true/false classification dataset consisting of 3681 training samples and 461 test samples. Each sample includes a statement and its truth label (true/false).
- **BBH-Navigate**: A dataset with 96 training samples and 144 test samples, where the task is to determine if following given instructions will return to the starting point.
- **ETHOS**: A hate speech detection dataset with 440 training samples and 200 test samples.
- **ArSarcasm**: An Arabic sarcasm detection dataset with 8437 training samples and 2110 test samples.
- **WebNLG**: A dataset with 200 training samples and 300 test samples, where the task is to convert given triples into natural language sentences.
- **GSM8K**: A dataset with 200 training samples and 300 test samples, focusing on solving math problems.
- **WSC**: A dataset with 100 training samples and 150 test samples, where the task is to identify the noun or noun phrase that each pronoun refers to in given sentences.

#### 2. Examples
Examples for each dataset are as follows:

- **LIAR Dataset Example**:
  - **Input**: "A study of private bail bond systems showed that Wisconsin has a higher no-show rate than other states of defendants skipping court appearances."
  - **Output**: "Yes" (This statement is false.)

- **BBH-Navigate Dataset Example**:
  - **Input**: "Move 7 steps forward, then turn left and move 3 steps."
  - **Output**: "No" (You do not return to the starting point.)

- **ETHOS Dataset Example**:
  - **Input**: "All immigrants are criminals."
  - **Output**: "Yes" (This statement is hate speech.)

- **ArSarcasm Dataset Example**:
  - **Input**: "Oh great, another rainy day. Just what I needed!"
  - **Output**: "Yes" (This statement is sarcastic.)

- **WebNLG Dataset Example**:
  - **Input**: "Anders_Osborne | associatedBand/associatedMusicalArtist | Billy_Iuso"
  - **Output**: "Rock musician Anders Osborne has worked with the band Galactic and also with the musical artists Tab Benoit and Billy Iuso."

- **GSM8K Dataset Example**:
  - **Input**: "If a burger costs $5 and fries cost $3, how much do I spend on 2 burgers and 2 fries?"
  - **Output**: "16" (The total cost is $16.)

- **WSC Dataset Example**:
  - **Input**: "The sculpture rolled off the shelf because it wasn’t level. What does the pronoun 'it' refer to?"
  - **Output**: "B" (The pronoun 'it' refers to the shelf.)

In this manner, each dataset provides examples of inputs and outputs for specific tasks, and the research evaluates the effectiveness of the proposed method through these examples.

<br/>
# 요약

이 논문에서는 Exemplar-Guided Reflection with Memory (ERM) 메커니즘을 제안하여 자동 프롬프트 최적화를 효율적이고 정확하게 수행합니다. 실험 결과, ERM은 LIAR 데이터셋에서 F1 점수를 10.1 향상시키고, ProTeGi에 비해 최적화 단계를 절반으로 줄이는 성과를 보였습니다. 예를 들어, ERM은 잘못된 예시를 통해 더 나은 피드백을 생성하고, 이를 메모리에 저장하여 최적화 과정에서 활용합니다.

---

This paper proposes the Exemplar-Guided Reflection with Memory (ERM) mechanism to achieve efficient and accurate automatic prompt optimization. Experimental results show that ERM improves the F1 score by 10.1 on the LIAR dataset and reduces the optimization steps by half compared to ProTeGi. For instance, ERM generates better feedback through erroneous examples and utilizes this feedback by storing it in memory during the optimization process.

<br/>
# 기타



이 논문에서는 Exemplar-Guided Reflection with Memory (ERM) 메커니즘을 통해 효율적이고 정확한 프롬프트 최적화를 달성하는 방법을 제안합니다. 이 방법은 세 가지 주요 구성 요소로 이루어져 있습니다: Exemplar-Guided Reflection, Feedback Memory, Exemplar Factory. 각 구성 요소의 결과와 인사이트는 다음과 같습니다.

1. **Exemplar-Guided Reflection**:
   - 이 메커니즘은 LLM이 잘못된 샘플을 선택하고 그에 대한 상세한 해결 과정을 제공하여 더 유용한 피드백을 생성하도록 유도합니다. 이를 통해 LLM은 더 많은 정보를 포함한 피드백을 생성할 수 있습니다.

2. **Feedback Memory**:
   - 피드백 메모리는 최적화 과정에서 생성된 피드백을 저장하고 우선 순위 점수를 부여하여, 유용한 피드백을 효율적으로 검색하고 활용할 수 있도록 합니다. 이 메커니즘은 피드백의 선택적 망각을 통해 최적화 과정에서 가치 있는 피드백만을 지속적으로 활용할 수 있게 합니다.

3. **Exemplar Factory**:
   - 이 공장은 예시를 저장하고 평가하여, 예시가 특정 질문에 미치는 영향을 분석합니다. 이를 통해 예시를 검색하고 프롬프트에 통합하여 예측 정확도를 높입니다.

### 결과 및 인사이트
- **성능 향상**: ERM은 LIAR 데이터셋에서 F1 점수를 10.1 향상시키고, ProTeGi보다 최적화 단계를 절반으로 줄였습니다. 이는 ERM이 피드백과 예시를 효과적으로 활용하여 최적화 속도를 크게 향상시켰음을 보여줍니다.
- **효율성**: ERM은 최적화 과정에서 필요한 단계를 줄여, 다른 방법들에 비해 더 빠른 성능을 달성했습니다. 예를 들어, LIAR 데이터셋에서 ERM은 7단계 만에 F1 점수 68.6을 달성했습니다.
- **구성 요소의 효과**: 각 구성 요소의 효과를 분석한 결과, Exemplar-Guided Reflection과 Feedback Memory, Exemplar Factory를 모두 포함했을 때 성능이 가장 크게 향상되었습니다. 특히, 피드백 메모리와 예시 공장이 성능 향상에 기여한 바가 큽니다.




This paper proposes the Exemplar-Guided Reflection with Memory (ERM) mechanism to achieve efficient and accurate prompt optimization. The method consists of three main components: Exemplar-Guided Reflection, Feedback Memory, and Exemplar Factory. The results and insights from each component are as follows:

1. **Exemplar-Guided Reflection**:
   - This mechanism guides the LLM to select erroneous samples and provide detailed solution processes, leading to the generation of more informative feedback. As a result, the LLM can produce feedback that includes more insights.

2. **Feedback Memory**:
   - The Feedback Memory stores the feedback generated during the optimization process and assigns priority scores, allowing for efficient retrieval and utilization of valuable feedback. This mechanism ensures that only beneficial feedback is continuously utilized through selective forgetting.

3. **Exemplar Factory**:
   - The Exemplar Factory stores and evaluates exemplars, analyzing their impact on specific questions. This allows for the retrieval of exemplars and their integration into prompts to enhance prediction accuracy.

### Results and Insights
- **Performance Improvement**: ERM improves the F1 score by 10.1 on the LIAR dataset and reduces the optimization steps by half compared to ProTeGi. This demonstrates that ERM effectively utilizes feedback and exemplars to significantly enhance optimization speed.
- **Efficiency**: ERM reduces the number of steps required in the optimization process, achieving faster performance compared to other methods. For instance, on the LIAR dataset, ERM reaches an F1 score of 68.6 in just 7 steps.
- **Effectiveness of Components**: Analyzing the impact of each component revealed that including Exemplar-Guided Reflection, Feedback Memory, and Exemplar Factory together resulted in the greatest performance improvement. Notably, the Feedback Memory and Exemplar Factory contributed significantly to the performance enhancement.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{yan2025efficient,
  title={Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection},
  author={Cilin Yan and Jingyun Wang and Lin Zhang and Ruihui Zhao and Xiaopu Wu and Kai Xiong and Qingsong Liu and Guoliang Kang and Yangyang Kang},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={753--779},
  year={2025},
  month={July},
  publisher={Association for Computational Linguistics},
  address={Beihang University, ByteDance, Zhejiang University},
}
```

### Chicago Style Citation

Yan, Cilin, Jingyun Wang, Lin Zhang, Ruihui Zhao, Xiaopu Wu, Kai Xiong, Qingsong Liu, Guoliang Kang, and Yangyang Kang. "Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 753–779. July 27 - August 1, 2025. Beihang University, ByteDance, Zhejiang University: Association for Computational Linguistics.
