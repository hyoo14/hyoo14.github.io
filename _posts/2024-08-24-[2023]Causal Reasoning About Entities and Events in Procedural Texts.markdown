---
layout: post
title:  "[2023]Causal Reasoning About Entities and Events in Procedural Texts"  
date:   2024-08-24 08:25:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

이벤트와 엔티티의 상태 간의 인과적 관계를 추론할 수 있는 데이터셋인 CREPE를 제안하고, 이를 통해 언어 모델이 멀티홉 이벤트 추론을 할 수 있도록 코드 스타일의 프롬프트를 사용해 성능을 향상. (이벤트와 엔티티 간의 인과적 관계는, 예를 들어, "뜨거운 기름이 든 팬에 물을 넣으면 팬에서 지글지글 소리가 난다"처럼, 물과 기름의 상태가 이벤트(지글지글 소리)의 발생 여부에 영향을 미치는 상황이 있음)  

짧은 요약(Abstract) :    

절차 텍스트에서 개체와 이벤트는 자연어 추론에서 매우 중요합니다. 기존 연구는 개체 상태 추적(예: 팬이 뜨거운지 여부) 또는 이벤트 추론(예: 팬을 만지면 화상을 입을지 여부)에만 집중했지만, 이 두 작업은 종종 인과적으로 연결됩니다. 우리는 이벤트의 개연성과 개체 상태에 대한 인과적 추론의 첫 번째 벤치마크인 CREPE를 제안합니다. 우리는 대부분의 언어 모델(GPT-3 포함)이 .35 F1 점수로 인간의 .87 F1보다 훨씬 뒤쳐져 있다는 것을 보여줍니다. 코드를 학습한 언어 모델을 활용해 이벤트를 프로그래밍 언어로 창의적으로 표현함으로써 모델 성능을 .59 F1까지 향상시켰습니다. 또한 개체와 이벤트 간의 인과적 관계를 중간 추론 단계로 삽입함으로써 성능을 .67 F1까지 더 향상시켰습니다. 우리의 연구 결과는 CREPE가 언어 모델에 도전 과제를 제공함을 시사하며, 코딩 스타일의 프롬프트와 생각의 흐름 체인을 결합하여 멀티홉 이벤트 추론에 대한 효율성을 보여줍니다.


Entities and events are crucial to natural language reasoning and common in procedural texts. Existing work has focused either exclusively on entity state tracking (e.g., whether a pan is hot) or on event reasoning (e.g., whether one would burn themselves by touching the pan), while these two tasks are often causally related. We propose CREPE, the first benchmark on causal reasoning of event plausibility and entity states. We show that most language models, including GPT-3, perform close to chance at .35 F1, lagging far behind human at .87 F1. We boost model performance to .59 F1 by creatively representing events as programming languages while prompting language models pretrained on code. By injecting the causal relations between entities and events as intermediate reasoning steps in our representation, we further boost the performance to .67 F1. Our findings indicate not only the challenge that CREPE brings for language models, but also the efficacy of code-like prompting combined with chain-of-thought prompting for multihop event reasoning.




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



1. **CREPE 데이터셋 구축**: 저자들이 절차 텍스트를 수동으로 수집하고 주석을 달아 개체 상태와 이벤트 간의 인과적 관계를 표현했습니다. 총 183개의 절차, 1219개의 단계, 324개의 이벤트 변화를 포함하며, 다양한 주제를 다루는 데이터셋을 만들었습니다. 테스트 세트는 펜실베니아 대학 인공지능 수업 학생들이 추가로 주석 작업에 참여해 제작되었습니다.

2. **프롬프팅 방법**: 언어 모델에 코드를 학습시키는 방식을 도입하여, 이벤트와 개체 상태를 Python과 같은 프로그래밍 언어 형식으로 표현했습니다. 이를 통해 개체 상태 변화를 중간 단계로 삽입하여 복잡한 논리적 사고를 가능하게 했고, 다단계 추론을 위한 코드 스타일의 프롬프팅을 제안했습니다.

3. **체인 오브 쏘트(Chain-of-Thought, CoT) 추론**: 이벤트 가능성을 예측하기 위해 CoT 프롬프팅을 활용하여 여러 단계로 논리적 사고를 유도하는 방식으로 추론을 진행했습니다. 이를 통해 개체 상태의 변화가 이벤트 가능성에 미치는 영향을 체계적으로 분석했습니다.

4. **개체 상태 추적 통합**: OpenPI와 같은 기존 개체 상태 추적 모델을 활용하여, 절차 텍스트에서 개체 상태를 추적한 후 이를 기반으로 이벤트 가능성을 예측하는 방식으로 모델의 성능을 향상시키려고 시도했습니다. 이를 통해 개체 상태 변화가 이벤트 추론에 미치는 중요한 역할을 강조했습니다.

5. **수동 주석 데이터 검증**: CREPE 데이터셋은 저자들이 직접 주석을 달았으며, 일부는 펜실베니아 대학 학생들이 주석 작업을 수행해 데이터의 품질을 검증했습니다. 개발 세트와 테스트 세트를 구분하여 모델의 일반화 성능을 평가했습니다.

---



1. **CREPE Dataset Construction**: The authors manually collected and annotated procedural texts to express causal relationships between entity states and events. The dataset contains 183 procedures, 1219 steps, and 324 event likelihood changes, covering various topics. Students from a University of Pennsylvania AI class participated in annotating the test set.

2. **Prompting Methods**: A novel method of prompting language models was introduced, where events and entity states were represented in a programming language format (e.g., Python). This allowed intermediate reasoning steps to be incorporated, enabling complex multihop reasoning with code-like prompts.

3. **Chain-of-Thought (CoT) Reasoning**: CoT prompting was employed to guide the logical reasoning process, breaking down event prediction into multiple steps. This method systematically predicted event likelihoods based on changes in entity states.

4. **Entity State Tracking Integration**: The authors integrated existing entity state tracking models, such as OpenPI, to track entity states in procedural texts and use these predictions for event reasoning. This highlighted the critical role of entity states in causal reasoning.

5. **Manual Annotation and Validation**: The CREPE dataset was manually annotated by the authors, with additional annotation work carried out by students to validate the dataset. The dataset was divided into development and test sets to evaluate the models' generalization performance.

<br/>
# Results  




1. **기존 모델 성능**: 기존의 대형 언어 모델(T5, T0, GPT-3)은 약 **0.35 F1** 수준으로 성능이 매우 낮았습니다. 이는 인간 성능(약 **0.87 F1**)에 크게 뒤떨어지는 결과였습니다.

2. 오픈에이아이 **Codex 모델의 성능 향상**: 이벤트와 개체 상태를 코드처럼 표현하는 새로운 프롬프팅 방식을 사용한 **Codex 모델**은 **0.585 F1**을 기록하며, 기존 모델에 비해 성능이 크게 향상되었습니다.

3. **체인 오브 쏘트(Chain-of-Thought, CoT) 방식**: 개체 상태 변화를 중간 추론 단계로 사용하여 **0.667 F1**로 성능을 더욱 향상시켰습니다. CoT 방식은 논리적 사고를 다단계로 확장해 더 정교한 추론을 가능하게 했습니다.

4. **인간 성능**: 인간이 개발 세트에서 기록한 성능은 **0.868 F1**으로, 모델 성능과 비교했을 때 여전히 큰 차이를 보였습니다.



1. **Baseline Model Performance**: Large language models such as T5, T0, and GPT-3 showed poor performance with an **F1 score of around 0.35**, significantly lower than the human performance of **0.87 F1**.

2. **Codex Model Improvement**: The **Codex model**, which utilized a novel code-like prompting method for representing events and entity states, achieved a **0.585 F1 score**, indicating a significant improvement over previous models.

3. **Chain-of-Thought (CoT) Approach**: By incorporating entity state changes as intermediate reasoning steps, the performance further increased to **0.667 F1**. This method allowed for more sophisticated multihop reasoning.

4. **Human Performance**: Human performance on the development set was **0.868 F1**, which still remains higher than the model performance, indicating room for further improvement.



<br/>
# 예시  



#### 예시: 신발 세탁 절차
- **목표**: 운동화를 세탁하는 것
- **상황**: 신발 끈을 제거하고 운동화를 헹굽니다.
- **질문**: 운동화를 신었을 때 발이 젖을 가능성은 얼마나 될까요?

이 과정에서, Codex 모델은 다음과 같은 **추론 단계**를 거칩니다:
1. **운동화의 상태**: 운동화가 젖었는지 여부를 먼저 묻습니다. (중간 추론)
2. **발이 젖을 가능성**: 젖은 운동화를 신으면 발이 젖을 가능성이 더 높다는 결론을 도출합니다.

이 **체인 오브 쏘트(CoT)** 스타일의 중간 추론을 통해, Codex는 **발이 젖을 가능성이 더 높음**이라는 결론에 도달하게 됩니다.

#### 부족한 점:
이 모델은 기본적인 논리적 추론을 잘 수행하지만, 복잡한 상황에서는 여전히 인간 수준의 추론 성능에 미치지 못합니다. 예를 들어, **개체 상태 추적**에서의 성능 향상에도 불구하고, 더 복잡한 인과 관계를 다루는 데 어려움을 겪습니다.

---



#### Example: Washing Sneakers
- **Goal**: Washing sneakers.
- **Context**: The shoelaces are removed, and the sneakers are rinsed.
- **Question**: What is the likelihood that my feet will get wet by wearing the sneakers?

In this process, the Codex model follows these **reasoning steps**:
1. **State of the Sneakers**: It first asks whether the sneakers are wet. (Intermediate reasoning)
2. **Likelihood of Wet Feet**: It concludes that wearing wet sneakers makes it more likely for the feet to get wet.

By following this **Chain-of-Thought (CoT)** reasoning, Codex reaches the conclusion that the **likelihood of wet feet increases**.

#### Limitations:
While the model performs well with basic logical reasoning, it still struggles to match human-level performance in more complex scenarios. For instance, despite improvements in **entity state tracking**, it faces challenges when dealing with more intricate causal relationships.




<br/>  
# 요약 

이 논문은 개체 상태와 이벤트 간의 인과적 관계를 추론하기 위한 CREPE 데이터셋을 구축하고, 이를 통해 언어 모델이 멀티홉 추론을 수행하도록 했습니다. Codex 모델은 이벤트와 개체 상태를 코드처럼 표현해, 체인 오브 쏘트(CoT) 방식으로 중간 추론 단계를 거쳐 성능을 향상시켰습니다. 예를 들어, 운동화를 헹군 후 발이 젖을 가능성을 추론할 때, 운동화가 젖었는지를 중간 단계로 물어 결론을 도출했습니다. Codex는 이러한 방법으로 0.667 F1까지 성능을 높였으나, 여전히 복잡한 인과 관계에서는 부족한 면이 있었습니다. 이러한 결과는 개체 상태의 추론이 중요한 중간 단계임을 강조합니다.

---

This paper introduces the CREPE dataset to infer causal relationships between entity states and events, enabling multihop reasoning in language models. The Codex model represents events and entity states in a code-like format and improves performance using the Chain-of-Thought (CoT) method by incorporating intermediate reasoning steps. For example, when predicting the likelihood of feet getting wet after rinsing sneakers, Codex asks whether the sneakers are wet as an intermediate step before reaching a conclusion. Codex achieved an F1 score of 0.667, but still struggles with more complex causal relationships. These results highlight the importance of entity state reasoning as a critical intermediate step.  



# 기타  


<br/>
# refer format:     
@inproceedings{zhang2023crepe,
  title={Causal Reasoning About Entities and Events in Procedural Texts},
  author={Li Zhang and Hainiu Xu and Yue Yang and Shuyan Zhou and Weiqiu You and Manni Arora and Chris Callison-Burch},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2023},
  pages={415--431},
  year={2023}
}



Zhang, Li, Hainiu Xu, Yue Yang, Shuyan Zhou, Weiqiu You, Manni Arora, and Chris Callison-Burch. "Causal Reasoning About Entities and Events in Procedural Texts." Findings of the Association for Computational Linguistics: EACL 2023, 415–431, 2023.

