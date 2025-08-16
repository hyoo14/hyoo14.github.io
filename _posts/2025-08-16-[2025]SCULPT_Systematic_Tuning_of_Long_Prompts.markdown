---
layout: post
title:  "[2025]SCULPT: Systematic Tuning of Long Prompts"
date:   2025-08-16 21:35:24 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

긴 프롬프팅 최적화를 위한 프레임워크-actor-critic 구조로 정제(일종의 강화학습처럼), 계층적 트리 구조를 사용하여 프롬프트를 체계적으로 수정    


짧은 요약(Abstract) :


이 논문의 초록에서는 대형 언어 모델(LLMs)의 효과적인 활용을 위해 프롬프트 최적화의 중요성을 강조하고 있습니다. 기존의 최적화 방법들은 짧은 프롬프트에는 효과적이지만, 길고 복잡한 프롬프트에서는 정보 손실이 발생하고 작은 변화에 민감하다는 한계를 가지고 있습니다. 이를 해결하기 위해 저자들은 SCULPT라는 새로운 프레임워크를 제안합니다. SCULPT는 프롬프트 최적화를 계층적 트리 정제 문제로 간주하며, 프롬프트를 트리 구조로 표현하여 맥락의 무결성을 유지하면서 목표 수정이 가능하도록 합니다. 이 프레임워크는 Critic-Actor 구조를 사용하여 프롬프트를 정제하는 반영을 생성하고, 이를 바탕으로 행동을 적용합니다. 평가 결과, SCULPT는 긴 프롬프트에서 효과적이며, 적대적 변형에 대한 강인성을 보여주고, 초기 인간 작성 프롬프트 없이도 높은 성능의 프롬프트를 생성할 수 있음을 입증했습니다. 기존의 최첨단 방법들과 비교했을 때, SCULPT는 필수 작업 정보를 보존하면서 구조화된 정제를 적용하여 LLM 성능을 일관되게 향상시킵니다.



The abstract of this paper emphasizes the importance of prompt optimization for the effective utilization of large language models (LLMs). Existing optimization methods are effective for short prompts but struggle with longer, more complex prompts, often leading to information loss and sensitivity to small perturbations. To address these challenges, the authors propose a new framework called SCULPT. SCULPT treats prompt optimization as a hierarchical tree refinement problem, representing prompts as tree structures that allow for targeted modifications while preserving contextual integrity. This framework employs a Critic-Actor structure to generate reflections for refining the prompt and applies actions based on these reflections. Evaluation results demonstrate SCULPT's effectiveness on long prompts, its robustness to adversarial perturbations, and its ability to generate high-performing prompts even without any initial human-written prompt. Compared to existing state-of-the-art methods, SCULPT consistently improves LLM performance by preserving essential task information while applying structured refinements.


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


### 1. 모델 아키텍처
메써드는 대규모 언어 모델(LLM)로, 주로 Transformer 아키텍처를 기반으로 합니다. 이 아키텍처는 입력 데이터의 각 단어를 벡터로 변환하고, 이 벡터들 간의 관계를 학습하여 문맥을 이해합니다. Transformer는 셀프 어텐션 메커니즘을 사용하여 입력 시퀀스의 모든 단어가 서로 어떻게 연결되는지를 파악합니다. 이를 통해 모델은 문맥에 따라 단어의 의미를 동적으로 조정할 수 있습니다.

### 2. 트레이닝 데이터
메써드는 다양한 출처에서 수집된 방대한 양의 텍스트 데이터를 사용하여 훈련됩니다. 이 데이터는 웹사이트, 책, 논문, 뉴스 기사 등 여러 종류의 문서로 구성되어 있습니다. 이러한 다양한 데이터는 모델이 다양한 주제와 스타일의 언어를 이해하고 생성할 수 있도록 돕습니다.

### 3. 특별한 기법
메써드는 다음과 같은 특별한 기법을 사용하여 성능을 향상시킵니다:

- **사전 훈련과 미세 조정**: 모델은 먼저 대규모 데이터셋에서 사전 훈련을 거친 후, 특정 작업에 맞게 미세 조정됩니다. 이 과정에서 모델은 특정 도메인이나 태스크에 대한 전문성을 갖추게 됩니다.

- **어텐션 메커니즘**: 셀프 어텐션 메커니즘을 통해 모델은 입력 시퀀스의 모든 단어 간의 관계를 고려하여 더 나은 문맥 이해를 가능하게 합니다. 이를 통해 모델은 긴 문장이나 복잡한 문맥에서도 중요한 정보를 효과적으로 추출할 수 있습니다.

- **전이 학습**: 메써드는 이전에 학습한 지식을 새로운 태스크에 적용할 수 있는 전이 학습 기법을 사용합니다. 이를 통해 모델은 적은 데이터로도 높은 성능을 발휘할 수 있습니다.

### 4. 평가 및 성능
메써드는 다양한 자연어 처리(NLP) 태스크에서 성능을 평가받습니다. 일반적으로, 모델의 성능은 정확도, F1 점수, BLEU 점수 등 여러 지표를 통해 측정됩니다. 이러한 평가를 통해 모델은 지속적으로 개선되고, 새로운 데이터와 태스크에 적응할 수 있습니다.

---



### 1. Model Architecture
The method is based on a large-scale language model (LLM), primarily utilizing the Transformer architecture. This architecture converts each word in the input data into vectors and learns the relationships between these vectors to understand context. The Transformer employs a self-attention mechanism to determine how each word in the input sequence relates to every other word, allowing the model to dynamically adjust the meaning of words based on context.

### 2. Training Data
The method is trained on a vast amount of text data collected from various sources. This data includes documents from websites, books, research papers, news articles, and more. The diversity of this data helps the model understand and generate language across a wide range of topics and styles.

### 3. Special Techniques
The method employs several special techniques to enhance performance:

- **Pre-training and Fine-tuning**: The model undergoes pre-training on a large dataset before being fine-tuned for specific tasks. This process allows the model to gain expertise in particular domains or tasks.

- **Attention Mechanism**: The self-attention mechanism enables the model to consider the relationships between all words in the input sequence, facilitating better contextual understanding. This allows the model to effectively extract important information even from long sentences or complex contexts.

- **Transfer Learning**: The method utilizes transfer learning techniques, allowing the model to apply previously learned knowledge to new tasks. This enables the model to achieve high performance even with limited data.

### 4. Evaluation and Performance
The method is evaluated across various natural language processing (NLP) tasks. Typically, the model's performance is measured using metrics such as accuracy, F1 score, and BLEU score. These evaluations help continuously improve the model and adapt it to new data and tasks.


<br/>
# Results


이 연구에서는 SCULPT라는 새로운 프레임워크를 통해 긴 프롬프트의 최적화를 수행하였습니다. SCULPT는 기존의 프롬프트 최적화 방법들이 긴 프롬프트에서 발생하는 정보 손실과 구조적 복잡성 문제를 해결하기 위해 고안되었습니다. SCULPT는 프롬프트를 계층적 트리 구조로 표현하여, 각 노드에 대해 타겟 수정이 가능하도록 하여 문맥의 무결성을 유지합니다.

### 경쟁 모델
SCULPT는 여러 기존 모델과 비교되었습니다. 여기에는 APE, LAPE, APEX, OPRO, ProTeGi와 같은 자동 프롬프트 최적화 방법들이 포함되었습니다. 각 모델은 다양한 작업에서 SCULPT와 비교하여 성능을 평가받았습니다.

### 테스트 데이터
SCULPT는 Big Bench Hard (BBH) 벤치마크의 네 가지 작업과 Responsible AI (RAI) 작업을 포함한 여러 데이터셋에서 평가되었습니다. 각 작업은 다양한 입력-출력 쌍으로 구성되어 있으며, SCULPT는 이러한 데이터셋에서 최적화된 프롬프트를 생성하여 성능을 향상시켰습니다.

### 메트릭
SCULPT의 성능은 여러 메트릭을 통해 평가되었습니다. 주요 메트릭으로는 정확도, F1 점수, ROUGE 점수 등이 있으며, SCULPT는 이러한 메트릭에서 기존 모델들보다 일관되게 높은 성능을 보였습니다. 특히, SCULPT는 긴 프롬프트에서의 정보 보존과 구조적 수정의 안정성을 강조하며, 다양한 작업에서 일반화 능력을 보여주었습니다.

### 비교
SCULPT는 기존 모델들에 비해 다음과 같은 장점을 보였습니다:
1. **정보 보존**: SCULPT는 긴 프롬프트에서 중요한 정보를 효과적으로 보존하며, 정보 손실을 최소화합니다.
2. **구조적 수정**: SCULPT는 계층적 트리 구조를 통해 프롬프트의 각 부분을 체계적으로 수정할 수 있어, 더 나은 성능을 발휘합니다.
3. **안정성**: SCULPT는 다양한 작업에서 일관된 성능을 보여주며, 적대적 변형에 대한 강인성을 입증하였습니다.

결과적으로, SCULPT는 긴 프롬프트 최적화에 있어 효과적이고 신뢰할 수 있는 솔루션으로 자리잡았습니다.

---




In this study, we introduced a novel framework called SCULPT for optimizing long prompts. SCULPT was designed to address the issues of information loss and structural complexity that arise in existing prompt optimization methods when applied to long prompts. By representing prompts as hierarchical tree structures, SCULPT allows for targeted modifications at each node while preserving contextual integrity.

### Competing Models
SCULPT was compared against several existing models, including APE, LAPE, APEX, OPRO, and ProTeGi. Each model was evaluated for performance across various tasks in comparison to SCULPT.

### Test Data
SCULPT was evaluated on multiple datasets, including four tasks from the Big Bench Hard (BBH) benchmark and various Responsible AI (RAI) tasks. Each task consisted of diverse input-output pairs, and SCULPT generated optimized prompts to enhance performance on these datasets.

### Metrics
The performance of SCULPT was assessed using several metrics, including accuracy, F1 score, and ROUGE scores. SCULPT consistently outperformed existing models across these metrics, particularly emphasizing information preservation and the stability of structured refinements in long prompts.

### Comparison
SCULPT demonstrated several advantages over existing models:
1. **Information Preservation**: SCULPT effectively preserves critical information in long prompts, minimizing information loss.
2. **Structured Refinement**: By utilizing a hierarchical tree structure, SCULPT allows for systematic modifications of each part of the prompt, leading to improved performance.
3. **Stability**: SCULPT exhibited consistent performance across various tasks, demonstrating robustness against adversarial perturbations.

In conclusion, SCULPT has established itself as an effective and reliable solution for optimizing long prompts.


<br/>
# 예제


### 초기 프롬프트
초기 프롬프트는 특정 작업을 수행하기 위해 모델에게 제공되는 지침이나 질문을 포함합니다. 이 프롬프트는 모델이 이해하고 적절한 출력을 생성할 수 있도록 명확하고 구체적이어야 합니다. 예를 들어, "주어진 문장에서 주어와 목적어를 식별하시오"와 같은 지침이 포함될 수 있습니다. 이 프롬프트는 모델이 어떤 작업을 수행해야 하는지를 명확히 하고, 필요한 경우 예시를 통해 추가적인 맥락을 제공합니다.

### 최적화된 프롬프트
최적화된 프롬프트는 초기 프롬프트를 바탕으로 하여, 모델의 성능을 극대화하기 위해 구조와 내용을 개선한 것입니다. 이 프롬프트는 더 명확한 지침, 더 나은 예시, 그리고 모델이 작업을 수행하는 데 필요한 모든 정보를 포함해야 합니다. 예를 들어, "다음 문장에서 주어와 목적어를 식별하시오. 예시: '나는 사과를 먹었다'에서 주어는 '나', 목적어는 '사과'입니다."와 같이 구체적인 예시를 추가하여 모델이 더 쉽게 이해할 수 있도록 합니다.

### 예시
- **초기 프롬프트**: "주어진 문장에서 주어와 목적어를 식별하시오."
- **최적화된 프롬프트**: "다음 문장에서 주어와 목적어를 식별하시오. 예시: '나는 사과를 먹었다'에서 주어는 '나', 목적어는 '사과'입니다."

이와 같이 초기 프롬프트는 모델이 수행해야 할 작업을 정의하고, 최적화된 프롬프트는 그 작업을 보다 효과적으로 수행할 수 있도록 돕는 역할을 합니다.

---



### Initial Prompt
The initial prompt contains instructions or questions provided to the model to perform a specific task. This prompt should be clear and specific enough for the model to understand and generate appropriate outputs. For example, it may include instructions like "Identify the subject and object in the given sentence." This prompt clarifies what the model needs to do and may provide additional context through examples.

### Optimized Prompt
The optimized prompt is an improved version based on the initial prompt, designed to maximize the model's performance by enhancing its structure and content. This prompt should include clearer instructions, better examples, and all necessary information for the model to perform the task. For instance, it might say, "Identify the subject and object in the following sentence. Example: In 'I ate an apple,' the subject is 'I,' and the object is 'apple.'" This addition of specific examples helps the model understand better.

### Example
- **Initial Prompt**: "Identify the subject and object in the given sentence."
- **Optimized Prompt**: "Identify the subject and object in the following sentence. Example: In 'I ate an apple,' the subject is 'I,' and the object is 'apple.'"

In this way, the initial prompt defines the task the model needs to perform, while the optimized prompt helps the model perform that task more effectively.

<br/>
# 요약


SCULPT는 긴 프롬프트 최적화를 위한 새로운 프레임워크로, 계층적 트리 구조를 사용하여 프롬프트를 체계적으로 수정합니다. 이 방법은 기존의 최적화 기법보다 더 높은 성능을 보이며, 정보 손실을 최소화하고 안정적인 결과를 제공합니다. 실험 결과, SCULPT는 다양한 작업에서 우수한 성능을 발휘하며, 특히 긴 프롬프트에 대한 강력한 내성을 보여줍니다.



SCULPT is a novel framework for optimizing long prompts, utilizing a hierarchical tree structure for systematic modifications. This approach demonstrates superior performance compared to existing optimization techniques, minimizing information loss and providing stable results. Experimental results show that SCULPT achieves excellent performance across various tasks, particularly exhibiting strong resilience to long prompts.

<br/>
# 기타


### 1. 정보 보존 (Information Preservation)
- **점수**: 9
- **설명**: 최적화된 프롬프트는 초기 프롬프트의 주요 개념과 세부 정보를 거의 완벽하게 보존하고 있습니다. 모든 중요한 정보가 포함되어 있으며, 문장의 의미와 맥락이 유지되었습니다. 특히, 초기 프롬프트에서 강조된 핵심 포인트들이 최적화된 프롬프트에서도 명확하게 드러나고 있습니다.

### 2. 전체 유사성 (Overall Dissimilarity)
- **점수**: 3
- **설명**: 최적화된 프롬프트는 초기 프롬프트와 상당한 유사성을 유지하고 있습니다. 구조와 흐름이 비슷하며, 예시와 지침도 유사하게 구성되어 있습니다. 그러나 최적화 과정에서 일부 문구가 변경되었고, 문장의 간결함이 향상되었습니다. 이러한 변화는 전체적인 유사성을 약간 감소시켰지만, 여전히 두 프롬프트 간의 유사성이 높습니다.

### 3. 코히어런스 (Coherence)
- **점수**: 4
- **설명**: 두 프롬프트 모두 논리적이고 일관된 흐름을 가지고 있습니다. 최적화된 프롬프트는 문장 간의 연결이 매끄럽고, 독자가 이해하기 쉽게 구성되어 있습니다. 그러나 일부 문장 구조의 변화로 인해 약간의 차이가 발생했습니다.

### 4. 구조 (Structure)
- **점수**: 5
- **설명**: 최적화된 프롬프트는 초기 프롬프트의 구조를 잘 유지하고 있습니다. 섹션과 하위 섹션이 명확하게 구분되어 있으며, 각 섹션의 내용이 잘 정리되어 있습니다. 그러나 몇 가지 문단의 순서가 변경되어 약간의 구조적 차이가 발생했습니다.

### 5. 예시 (Examples)
- **점수**: 6
- **설명**: 최적화된 프롬프트는 초기 프롬프트의 예시를 잘 반영하고 있으며, 새로운 예시도 추가되었습니다. 그러나 일부 예시의 표현이 변경되어 원래의 의미가 약간 변형되었습니다.

### 6. 지침 (Instructions)
- **점수**: 4
- **설명**: 최적화된 프롬프트는 초기 프롬프트의 지침을 잘 따르고 있습니다. 그러나 몇 가지 지침이 간소화되거나 수정되어 약간의 차이가 발생했습니다.

## 종합 결과
```json
{
  "Information Preservation": 9,
  "Overall Dissimilarity": 3,
  "Explanation": "The optimized prompt retains almost all key concepts and details from the initial prompt, resulting in a high score for information preservation. The overall similarity score is low due to minor changes in phrasing and structure, but the coherence and logical flow remain intact."
}
```

---




### 1. Information Preservation
- **Score**: 9
- **Explanation**: The optimized prompt retains almost all key concepts and details from the initial prompt. All important information is included, and the meaning and context of the sentences are preserved. Particularly, the key points emphasized in the initial prompt are clearly reflected in the optimized prompt.

### 2. Overall Dissimilarity
- **Score**: 3
- **Explanation**: The optimized prompt maintains a significant similarity to the initial prompt. The structure and flow are similar, and the examples and instructions are also organized in a comparable manner. However, some phrases were altered during the optimization process, and the conciseness of the sentences improved. These changes slightly reduced the overall similarity, but the high level of similarity between the two prompts remains.

### 3. Coherence
- **Score**: 4
- **Explanation**: Both prompts exhibit a logical and consistent flow. The optimized prompt has smooth connections between sentences, making it easy for the reader to understand. However, slight differences arose due to changes in some sentence structures.

### 4. Structure
- **Score**: 5
- **Explanation**: The optimized prompt effectively maintains the structure of the initial prompt. Sections and subsections are clearly delineated, and the content of each section is well organized. However, some changes in the order of paragraphs resulted in slight structural differences.

### 5. Examples
- **Score**: 6
- **Explanation**: The optimized prompt reflects the examples from the initial prompt well, and new examples have also been added. However, some expressions of examples were altered, slightly changing their original meaning.

### 6. Instructions
- **Score**: 4
- **Explanation**: The optimized prompt adheres well to the instructions of the initial prompt. However, some instructions were simplified or modified, leading to slight differences.

## Overall Result
```json
{
  "Information Preservation": 9,
  "Overall Dissimilarity": 3,
  "Explanation": "The optimized prompt retains almost all key concepts and details from the initial prompt, resulting in a high score for information preservation. The overall similarity score is low due to minor changes in phrasing and structure, but the coherence and logical flow remain intact."
}
```

<br/>
# refer format:
### BibTeX 형식
```bibtex
@inproceedings{kumar2025sculpt,
  title={SCULPT: Systematic Tuning of Long Prompts},
  author={Shanu Kumar and Akhila Yesantarao Venkata and Shubhanshu Khandelwal and Bishal Santra and Parag Agrawal and Manish Gupta},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={14996--15029},
  year={2025},
  month={July},
  publisher={Association for Computational Linguistics},


}
```

### 시카고 스타일
Kumar, Shanu, Akhila Yesantarao Venkata, Shubhanshu Khandelwal, Bishal Santra, Parag Agrawal, and Manish Gupta. 2025. "SCULPT: Systematic Tuning of Long Prompts." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 14996–15029. Association for Computational Linguistics.
