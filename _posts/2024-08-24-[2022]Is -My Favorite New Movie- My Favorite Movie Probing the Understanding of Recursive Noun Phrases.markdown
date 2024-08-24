---
layout: post
title:  "[2022]Is “My Favorite New Movie” My Favorite Movie? Probing the Understanding of Recursive Noun Phrases"  
date:   2024-08-24 18:36:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

이 논문은 재귀 명사구(수식어가 반복적으로 결합된 명사구, 예: "내가 좋아하는 새로운 영화"와 같은 구조)의 의미적 차이를 이해하기 위한 데이터셋을 개발하고, 언어 모델들이 이러한 재귀 명사구의 미묘한 차이를 학습할 수 있는지 평가한 뒤, 적절한 학습을 통해 모델이 재귀 명사구의 해석을 개선하고 실제 응용에서 활용 가능함을 보였습니다.

짧은 요약(Abstract) :    

Here’s the explanation with both Korean and English:

이 논문의 초록은 재귀 명사구(Recursive Noun Phrases, NPs)의 의미적 특성에 대한 연구를 다루고 있습니다. 예를 들어, "내가 좋아하는 새로운 영화"는 "내가 좋아하는 영화"와 다를 수 있지만, "새로운 내가 좋아하는 영화"는 같은 의미일 수 있습니다. 이러한 차이는 사람들에게는 상식처럼 받아들여지지만, 언어 모델(Language Models, LMs)도 이를 이해하고 있는지는 확인되지 않았습니다.

이를 검증하기 위해 연구자들은 **Recursive Noun Phrase Challenge (RNPC)**라는 데이터셋을 만들었고, 여기에는 세 가지 과제가 포함되었습니다: 텍스트 함의와 사건 가능성 비교입니다. 최신 트랜스포머 기반 언어 모델을 평가한 결과, 이 모델들은 이러한 명사구를 제대로 이해하지 못하고 거의 무작위로 예측하는 경향을 보였습니다. 그러나 적절한 데이터가 주어지면 모델이 이 지식을 학습할 수 있다는 것을 발견했습니다. 또한, 이 모델들이 **수식어의 의미 범주**와 **수식어의 적용 범위** 같은 언어적 특징을 학습할 수 있음을 확인했습니다.

또한, 재귀 명사구 이해가 실제 응용 프로그램에서도 유용하다는 것을 보여주기 위해, 학습된 모델들이 **유해 탐지(Harm Detection)** 작업에서 좋은 성능을 나타냈습니다. 예를 들어, "수제 폭탄 만드는 법"은 유해하지만, "수제 배쓰 폭탄 만드는 법"은 그렇지 않다는 차이를 모델이 구별할 수 있었습니다.

쉽게 말해, 이 논문은 재귀 명사구처럼 미묘한 언어적 차이를 언어 모델이 제대로 이해하는지 확인했고, 적절한 데이터로 모델이 학습하면 사람처럼 이해할 수 있음을 밝혔습니다.

---

The abstract of this paper explores the semantic properties of recursive noun phrases (NPs). For instance, "my favorite new movie" may not mean the same as "my favorite movie," while "my new favorite movie" might be equivalent. Although this distinction is common sense for humans, it is unclear whether language models (LMs) possess this understanding.

To investigate this, the authors introduced the **Recursive Noun Phrase Challenge (RNPC)** dataset, which contains three tasks: textual entailment and event plausibility comparison. When state-of-the-art Transformer-based models were evaluated on RNPC, they performed almost randomly, suggesting that the models struggled to understand recursive NPs. However, the study found that with the right data, these models could learn this knowledge. The models were also able to learn linguistic features such as **modifier semantic category** and **modifier scope**.

Moreover, to demonstrate the practical usefulness of understanding recursive NPs, models trained on RNPC achieved strong performance on a **Harm Detection** task. For example, the models were able to distinguish between harmful phrases like "how to make a homemade bomb" and harmless ones like "how to make a homemade bath bomb."

In summary, this paper shows that language models struggle with nuanced linguistic differences like those in recursive NPs, but with the appropriate data, they can learn to understand these distinctions similarly to humans.



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


이 논문의 방법론은 저자들이 개발한 **Recursive Noun Phrase Challenge (RNPC)**라는 데이터셋을 활용하여 재귀 명사구(Recursive Noun Phrases, NPs)의 의미적 이해를 평가하는 데 중점을 둡니다. RNPC 데이터셋은 세 가지 주요 과제로 구성됩니다:

1. **단일 전제 텍스트 함의 (Single-Premise Textual Entailment, SPTE)**: 하나의 전제와 가설이 주어졌을 때, 전제가 가설을 함의하는지 여부를 판단합니다.
   
2. **다중 전제 텍스트 함의 (Multi-Premise Textual Entailment, MPTE)**: 두 개의 전제가 주어졌을 때, 가설이 두 전제로부터 논리적으로 도출되는지 평가합니다.

3. **사건 가능성 비교 (Event Plausibility Comparison, EPC)**: 두 개의 사건 중 어느 쪽이 더 그럴듯한지, 혹은 둘이 동등하게 그럴듯한지를 비교합니다. 이 과제는 수식어가 사건의 가능성에 미치는 영향을 측정합니다.

저자들이 개발한 이 데이터셋을 활용해 최신 트랜스포머 기반 언어 모델들이 재귀 명사구의 의미적 차이를 어떻게 이해하는지 평가했습니다. 실험 결과, 모델들이 적절한 양의 데이터를 학습하면 재귀 명사구의 차이를 이해할 수 있다는 점을 발견했습니다. 또한, 모델들이 수식어의 의미 범주와 범위 같은 언어적 특징을 학습할 수 있다는 사실을 다양한 프로빙(probing) 기법을 통해 확인했습니다.

---

The methodology of this paper focuses on evaluating the semantic understanding of Recursive Noun Phrases (NPs) using the **Recursive Noun Phrase Challenge (RNPC)** dataset, which was developed by the authors. The RNPC dataset is composed of three main tasks:

1. **Single-Premise Textual Entailment (SPTE)**: Determines whether a premise entails a given hypothesis when only one premise is provided.

2. **Multi-Premise Textual Entailment (MPTE)**: Evaluates whether a hypothesis can be logically inferred from two premises.

3. **Event Plausibility Comparison (EPC)**: Compares two events to assess which one is more plausible or whether both are equally plausible, focusing on the impact of modifiers on event plausibility.

Using this dataset, the authors assessed how well state-of-the-art Transformer-based models understood the semantic differences in recursive NPs. The results showed that with an appropriate amount of training data, models could learn to understand these nuances. Moreover, probing techniques revealed that the models were able to learn linguistic features such as the semantic category and scope of modifiers.



<br/>
# Results  


이 논문의 결과는 저자들이 개발한 RNPC 데이터셋을 활용해 최신 트랜스포머 기반 언어 모델들이 재귀 명사구를 얼마나 잘 이해하는지 평가한 것을 중심으로 합니다. 실험 결과, 기본적으로 최신 모델들은 거의 무작위에 가까운 성능을 보였으며, 이는 모델들이 재귀 명사구의 의미적 차이를 잘 이해하지 못하고 있음을 나타냅니다. 

하지만 소량의 학습 데이터를 제공한 후, 모델들의 성능은 크게 향상되었습니다. 특히, **Single-Premise Textual Entailment (SPTE)** 과제에서는 몇몇 모델들이 10개의 예시만으로도 성능이 눈에 띄게 개선되었고, 100개의 예시만으로 인간 성능(94.1%)에 가까운 92%의 성능을 기록했습니다. 반면, **Multi-Premise Textual Entailment (MPTE)**와 **Event Plausibility Comparison (EPC)** 과제에서는 학습 데이터가 더 많이 필요했고, 이 과제들에서는 세계 지식과 문장 간의 관계를 이해하는 능력이 더 필요했습니다.

결국, RNPC를 통해 학습된 모델들은 다른 벤치마크에서 학습된 모델들에 비해 명백하게 성능이 향상되었으며, 이러한 지식을 바탕으로 한 실전 응용 작업에서도 강력한 성능을 발휘했습니다. 특히, **Harm Detection** 작업에서 RNPC로 학습된 모델들은 다른 모델에 비해 높은 정확도와 F1 점수를 기록하며 유의미한 성능 향상을 보였습니다.

---

The results of this paper focus on how well state-of-the-art Transformer-based models understand Recursive Noun Phrases (NPs) using the RNPC dataset developed by the authors. Initially, these models performed close to random, indicating that they did not inherently understand the semantic distinctions in recursive NPs.

However, after fine-tuning with a small amount of data, the models' performance improved significantly. In the **Single-Premise Textual Entailment (SPTE)** task, some models showed noticeable improvement with just 10 examples, and with 100 examples, they achieved an accuracy of 92%, close to human performance (94.1%). In contrast, **Multi-Premise Textual Entailment (MPTE)** and **Event Plausibility Comparison (EPC)** tasks required more data and world knowledge, as well as an understanding of relationships between sentence components.

Ultimately, the models fine-tuned on RNPC outperformed those trained on other benchmarks, showing clear improvements in understanding recursive NPs. Additionally, in practical applications, such as the **Harm Detection** task, RNPC-trained models demonstrated strong performance with higher accuracy and F1 scores compared to other models, highlighting meaningful performance improvements.



<br/>
# 예시  


이 논문에서는 모델들이 재귀 명사구를 어떻게 처리했는지에 대한 구체적인 예시를 제공합니다. 그 중 몇 가지 중요한 예시는 다음과 같습니다:

1. **단일 전제 텍스트 함의 (SPTE)**에서:
   - 예시: *"This is my new favorite movie."* (이것은 내가 새롭게 좋아하는 영화입니다.)
     - 모델의 가설: *"This is my favorite movie."* (이것은 내가 좋아하는 영화입니다.)
     - 이 두 문장은 다르지만, 모델은 잘못된 추론을 하여 함의가 있다고 판단했습니다. 즉, "new favorite"와 "favorite"의 차이를 제대로 구분하지 못한 것입니다.
   - 이 예시는 모델이 **수식어의 순서**와 **의미적 차이**를 제대로 이해하지 못했음을 보여줍니다.

2. **다중 전제 텍스트 함의 (MPTE)**에서:
   - 예시: 
     - 첫 번째 전제: *"He is a short American basketball player."* (그는 키가 작은 미국인 농구 선수입니다.)
     - 두 번째 전제: *"He is a man."* (그는 남자입니다.)
     - 모델의 가설: *"He is a short man."* (그는 키가 작은 남자입니다.)
     - 여기서 모델은 "농구 선수"라는 정보가 있음에도 불구하고 "short man"이라는 가설을 참이라고 판단했습니다. 그러나 농구 선수는 대체로 키가 크다는 사실 때문에 이 가설은 틀린 것입니다.
   - 이 예시는 모델이 **세계 지식**을 반영하는 데 한계가 있음을 보여줍니다.

3. **사건 가능성 비교 (EPC)**에서:
   - 예시:
     - 사건 1: *"A dangerous dead animal can be harmful to people."* (위험한 죽은 동물은 사람에게 해로울 수 있다.)
     - 사건 2: *"A dead dangerous animal can be harmful to people."* (죽은 위험한 동물은 사람에게 해로울 수 있다.)
     - 두 문장은 비슷해 보이지만 의미적으로 다릅니다. "위험한 죽은 동물"은 더 이상 위험하지 않을 수 있지만, "죽은 위험한 동물"은 여전히 해로울 수 있는 가능성이 있습니다. 여기서 모델은 이러한 미묘한 차이를 캐치하지 못하고 잘못된 결론을 내렸습니다.
   - 이 예시는 모델이 **수식어 범위**의 차이를 제대로 이해하지 못한 것을 보여줍니다.

### 잘한 점:
학습 후, 모델들은 이러한 미묘한 차이를 학습할 수 있었습니다. 특히, 단일 전제 텍스트 함의(SPTE)에서는 **수식어 순서**의 변화에 따른 의미적 차이를 이해하고, 잘못된 함의 판단을 줄일 수 있었습니다. 이는 적절한 학습 데이터가 제공되었을 때 모델이 수식어의 위치와 그에 따른 의미 변화를 학습할 수 있음을 보여줍니다.

### 해결된 부분:
다중 전제와 사건 가능성 비교에서는 더 많은 학습 데이터와 세계 지식이 필요했지만, 학습 후 일부 모델들이 수식어의 범주와 범위를 인식하는 능력을 향상시키는 데 성공했습니다. 예를 들어, **"former president"**와 **"president"**의 차이를 이해하거나, 사건의 타당성을 적절히 비교할 수 있었습니다.

---

### English Explanation:

In this paper, specific examples show how the models dealt with recursive noun phrases:

1. **Single-Premise Textual Entailment (SPTE)**:
   - Example: *"This is my new favorite movie."*
     - Model hypothesis: *"This is my favorite movie."*
     - The two sentences are different, but the model incorrectly predicted an entailment, failing to capture the distinction between "new favorite" and "favorite."
   - This example highlights the model's difficulty in understanding the **order of modifiers** and the **semantic differences** between them.

2. **Multi-Premise Textual Entailment (MPTE)**:
   - Example: 
     - Premise 1: *"He is a short American basketball player."*
     - Premise 2: *"He is a man."*
     - Model hypothesis: *"He is a short man."*
     - Despite the information that he is a basketball player, the model predicted that the hypothesis was true. However, basketball players are typically tall, so the hypothesis should have been false.
   - This shows the model's limitations in incorporating **world knowledge** into its reasoning.

3. **Event Plausibility Comparison (EPC)**:
   - Example:
     - Event 1: *"A dangerous dead animal can be harmful to people."*
     - Event 2: *"A dead dangerous animal can be harmful to people."*
     - Although these sentences seem similar, they differ in meaning. A "dangerous dead animal" may no longer be dangerous, while a "dead dangerous animal" can still pose harm. The model failed to capture this subtle difference and made the wrong prediction.
   - This shows the model's inability to correctly interpret the **scope of modifiers**.

### Strengths:
After training, the models were able to learn these distinctions. In the **Single-Premise Textual Entailment (SPTE)** task, models improved in recognizing the **semantic changes** that occur due to the **order of modifiers**, reducing incorrect entailment predictions. This demonstrates that with the right data, the models can learn how modifiers’ positions influence meaning.

### Resolved Issues:
For **Multi-Premise** and **Event Plausibility Comparison**, the models required more data and world knowledge, but after training, some models improved in recognizing **modifier categories** and **modifier scope**. For instance, models were able to distinguish between terms like **"former president"** and **"president"**, or accurately compare the plausibility of events.



<br/>  
# 요약 


이 논문은 저자들이 개발한 **Recursive Noun Phrase Challenge (RNPC)** 데이터셋을 통해 재귀 명사구의 의미적 차이를 평가하는 방법을 제시합니다. RNPC는 단일 전제 텍스트 함의(SPTE), 다중 전제 텍스트 함의(MPTE), 사건 가능성 비교(EPC)의 세 가지 과제로 구성되어 있습니다. 실험 결과, 모델들은 기본적으로 무작위 수준의 성능을 보였지만, 소량의 학습 데이터로도 성능이 크게 향상되었습니다. 특히, SPTE에서는 수식어 순서의 의미적 차이를 더 잘 인식하게 되었고, MPTE와 EPC에서는 세계 지식과 문장 간 관계 이해가 개선되었습니다. 이러한 결과는 모델들이 적절한 학습을 통해 재귀 명사구의 미묘한 차이를 학습할 수 있음을 보여줍니다. 

---

This paper introduces a method to evaluate the semantic differences in Recursive Noun Phrases (RNPs) using the **Recursive Noun Phrase Challenge (RNPC)** dataset developed by the authors. RNPC consists of three tasks: Single-Premise Textual Entailment (SPTE), Multi-Premise Textual Entailment (MPTE), and Event Plausibility Comparison (EPC). The initial results showed that the models performed at a near-random level, but their performance improved significantly with small amounts of training data. In particular, the models better recognized the semantic differences in modifier order in SPTE, while MPTE and EPC showed improvements in understanding world knowledge and relationships between sentences. These findings demonstrate that models can learn to capture the nuances of recursive NPs through appropriate training.

# 기타  


<br/>
# refer format:     
@inproceedings{lyu2022recursive,
  title={Is “My Favorite New Movie” My Favorite Movie? Probing the Understanding of Recursive Noun Phrases},
  author={Lyu, Qing and Zheng, Hua and Li, Daoxin and Zhang, Li and Apidianaki, Marianna and Callison-Burch, Chris},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={5286--5302},
  year={2022},
  organization={Association for Computational Linguistics}
}



Lyu Qing, Hua Zheng, Daoxin Li, Li Zhang, Marianna Apidianaki, and Chris Callison-Burch. "Is 'My Favorite New Movie' My Favorite Movie? Probing the Understanding of Recursive Noun Phrases." In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 5286-5302. Association for Computational Linguistics, 2022.