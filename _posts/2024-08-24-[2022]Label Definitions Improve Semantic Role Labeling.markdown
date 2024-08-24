---
layout: post
title:  "[2022]Label Definitions Improve Semantic Role Labeling"  
date:   2024-08-24 12:11:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

이 논문은 CoNLL09 데이터셋의 프레임 파일에서 제공되는 PropBank 주석 지침 기반의 레이블 정의를 활용해 SRL 성능을 향상시키는 방법을 제안하며, 예를 들어 “일하다(work)“라는 술어에 대해 “일하는 사람(worker)” 등의 구체적인 정의를 적용해 더 정확한 의미 파악을 돕습니다.


짧은 요약(Abstract) :    


 이 논문은 의미역 결정(SRL)이라는 작업을 다룹니다. SRL은 문장에서 누가 무엇을, 언제, 어디서, 어떻게 했는지를 분석하는 기술입니다. 이 과정에서 각각의 행위나 사건에 대해 그 역할을 나타내는 레이블을 붙이는 것이 중요합니다. 기존 연구에서는 이러한 레이블을 단순히 기호로 처리했으나, 저자들은 이 레이블에 대한 정의를 실제로 활용하여 성능을 향상시키는 방법을 제안했습니다. 예를 들어, “일하다(work)“라는 동사에 대해 “일하는 사람(worker)”, “일(job)”, “고용주(employer)“와 같은 정의를 사용하여 더욱 정확한 의미를 파악할 수 있습니다. 이 방법을 통해 적은 데이터로도 더 좋은 결과를 얻을 수 있었으며, 특히 데이터가 부족한 상황에서도 성능이 크게 개선되었습니다.


This paper addresses the task of Semantic Role Labeling (SRL), which involves analyzing sentences to identify who did what, when, where, and how. In this process, assigning role labels to each action or event is essential. While previous research treated these labels symbolically, the authors propose leveraging the actual definitions of these labels to improve performance. For instance, for the verb “work,” using definitions like “worker,” “job,” and “employer” helps understand the meaning more accurately. Their approach leads to better results, especially in low-resource settings where training data is scarce.




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


이 논문에서 사용한 방법론은 **CoNLL09 데이터셋의 프레임 파일**에서 제공되는 레이블의 정의를 활용하여 SRL(의미역 결정) 성능을 향상시키는 것입니다. 일반적인 SRL 작업에서는 각 술어(predicate)에 대해 논항(argument)들에게 기호적인 레이블(A0, A1 등)을 부여합니다. 그러나 이 레이블들은 문맥에 따라 그 의미가 다를 수 있어, 단순한 기호만으로는 정확한 역할을 파악하는 데 한계가 있습니다.

이에 저자들은 CoNLL09 데이터셋에 포함된 프레임 파일을 사용하여 각 술어의 의미에 해당하는 정의를 모델에 주입하는 방식을 제안했습니다. 예를 들어, 술어 ”work(일하다)“에 대해 A0 레이블은 ”worker(일하는 사람)“, A1은 ”job(직무)“, A2는 ”employer(고용주)“로 정의됩니다. 이렇게 각 논항에 대한 정의를 모델에 추가로 제공함으로써, 단순히 기호적인 레이블을 사용하는 기존 방법보다 논항의 실제 의미를 더 잘 이해하게 만들어 성능을 향상시킵니다.

이러한 방법론은 특히 데이터가 부족한 상황이나 새로운 술어를 다룰 때 성능이 더욱 개선되는 효과를 보였습니다. 데이터가 적을 때도 각 술어의 논항이 무엇을 의미하는지를 모델이 학습할 수 있기 때문입니다.

The methodology in this paper leverages the label definitions provided in the frame files from the CoNLL09 dataset to improve the performance of Semantic Role Labeling (SRL). Typically, SRL assigns symbolic labels (like A0, A1) to the arguments of a predicate, but these symbolic labels can vary in meaning depending on the context, making them less informative. To address this, the authors use the frame files from the dataset, which provide definitions for each argument of a predicate.

For instance, for the verb ”work,“ A0 is defined as ”worker,“ A1 as ”job,“ and A2 as ”employer.“ By incorporating these definitions into the model, the authors enable the model to learn the actual meaning behind each label, rather than relying solely on symbolic representations. This allows the model to more accurately understand the role of each argument in the sentence, leading to better performance, particularly in low-resource settings or when dealing with unfamiliar predicates.








<br/>
# Results  





1. **모델 비교**: 
   - 논문에서는 RoBERTa 기반 모델을 사용해, 기존 AC(기호적인 레이블만 사용) 모델과 ACD(레이블 정의를 포함한 모델)를 비교했습니다.
   - CoNLL09 데이터셋을 사용한 실험에서, ACD 모델이 AC 모델보다 더 나은 성능을 보여주었습니다. 
   - 특히 **out-domain** 테스트 세트에서 ACD 모델은 **gold sense**를 사용했을 때 기존 최첨단 모델보다 **1.1 F1 점수**가 높았고, **predicted sense**를 사용했을 때에도 **0.5 F1** 더 높았습니다 .

2. **성능 향상**:
   - In-domain 테스트에서 ACD 모델은 AC 모델과 비교해 작은 차이로 성능이 개선되었지만, **정확도**와 **정밀도**에서 더 나은 성능을 보였습니다.
   - **p-CoNLL09 데이터셋**에서 ACD 모델은 in-domain에서 더욱 두드러진 성능 향상을 보였으며, out-domain에서도 비슷한 성능을 유지했습니다 .

3. **저빈도 술어에 대한 성능**:
   - ACD 모델은 특히 **저빈도 술어**에 대해 큰 성능 향상을 보였습니다. 예를 들어, ACD 모델은 AC 모델보다 최대 **4.4 F1** 더 높은 성능을 보여주었으며, 특히 데이터셋에서 거의 등장하지 않는 술어에 대해 성능이 더 두드러졌습니다.
   

1. **Model Comparison**: 
   - They use a RoBERTa-based model to compare the conventional AC (symbolic labels only) model with the ACD (including label definitions) model.
   - On the CoNLL09 dataset, the ACD model outperformed the AC model. Specifically, on the **out-domain** test set, the ACD model with **gold sense** outperformed the state-of-the-art by **1.1 F1**, and with **predicted sense**, it outperformed by **0.5 F1** .

2. **Performance Improvements**:
   - In the in-domain test, the ACD model also showed improved precision and accuracy compared to the AC model, though the differences were smaller.
   - On the **p-CoNLL09 dataset**, the advantage of the ACD model was more pronounced in-domain and similar out-domain .

3. **Performance on Low-Frequency Predicates**:
   - The ACD model particularly excelled on **low-frequency predicates**, achieving up to **4.4 F1** higher than the AC model, demonstrating significant improvement for rarely seen predicates .





<br/>
# 예시  


논문의 주요 결과 부분을 설명하며, ACD 모델의 성능 향상 및 구체적 예시를 아래와 같이 설명할 수 있습니다:

**결과 요약**:
ACD 모델은 의미역 레이블링(SRL) 작업에서 기존의 AC 모델과 비교하여 더 나은 성능을 보여주었습니다. 특히 **CoNLL09 데이터셋**을 사용한 실험에서 ACD 모델은 **out-domain** 테스트에서 **gold sense** 기준으로 **1.1 F1**, **predicted sense** 기준으로 **0.5 F1** 점수의 성능 향상을 기록했습니다 

**구체적 예시**:
예를 들어, 문장 ”**Any good decorator these days can make you a tasteful home**“에서 술어인 ”make“의 의미는 ”create“로, ”you“는 간접 목적어로 해석됩니다. ACD 모델은 올바른 정의에 따라, ”decorator“를 창작자(A0)로, ”home“을 창작물(A1)로 정확히 예측했습니다. 반면, AC 모델은 기호적인 레이블만 사용했기 때문에 ”you“를 잘못된 A1으로 예측하는 등의 오류를 범했습니다 

이러한 예시는 ACD 모델이 정의를 활용함으로써 복잡한 문장이나 드문 술어에 대해 더 나은 성능을 발휘할 수 있음을 보여줍니다.



**Results Summary**:
The ACD model showed superior performance in SRL tasks compared to the traditional AC model. In particular, on the **CoNLL09 dataset**, the ACD model achieved an improvement of **1.1 F1** with **gold sense** and **0.5 F1** with **predicted sense** on the **out-domain** test 

**Specific Example**:
For example, in the sentence ”**Any good decorator these days can make you a tasteful home**,“ the predicate ”make“ has the sense ”create,“ and ”you“ is interpreted as an indirect object. The ACD model, using the correct definitions, accurately predicted ”decorator“ as the creator (A0) and ”home“ as the creation (A1). In contrast, the AC model, which only relied on symbolic labels, incorrectly predicted ”you“ as A1, making mistakes in identifying the roles of arguments.

This example demonstrates how the ACD model, by leveraging definitions, performs better on complex sentences or rare predicates.



<br/>  
# 요약 
이 논문은 의미역 결정(SRL)에서 레이블 정의를 활용해 성능을 향상시키는 방법을 제안했습니다. CoNLL09 데이터셋의 프레임 파일에서 제공되는 레이블 정의를 주입하여, 단순한 기호적 레이블 대신 각 논항의 실제 의미를 학습하게 했습니다. 예를 들어, ”make“라는 술어가 포함된 문장에서 ACD 모델은 ”decorator“를 A0, ”home“을 A1로 정확히 예측했습니다. ACD 모델은 특히 저빈도 술어에서 기존 AC 모델보다 최대 4.4 F1 점수의 성능 향상을 보였으며, out-domain 테스트에서는 gold sense 기준으로 1.1 F1 향상을 기록했습니다. 이러한 결과는 SRL 작업에서 레이블 정의를 활용하는 것이 특히 데이터가 부족한 상황에서 효과적임을 보여줍니다.


This paper proposes a method to improve Semantic Role Labeling (SRL) performance by using label definitions. By incorporating definitions from the frame files of the CoNLL09 dataset, the model learns the actual meaning of each argument rather than just symbolic labels. For instance, in a sentence with the predicate ”make,“ the ACD model accurately predicted ”decorator“ as A0 and ”home“ as A1. The ACD model showed up to 4.4 F1 improvement on low-frequency predicates compared to the AC model and achieved a 1.1 F1 improvement in the out-domain test with gold sense. These results demonstrate the effectiveness of using label definitions, especially in low-resource settings.

# 기타  


<br/>
# refer format:     
@inproceedings{zhang2022label,
  title={Label Definitions Improve Semantic Role Labeling},
  author={Zhang, Li and Jindal, Ishan and Li, Yunyao},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={5613--5620},
  year={2022}
}


Li Zhang, Ishan Jindal, and Yunyao Li. “Label Definitions Improve Semantic Role Labeling.” In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 5613-5620. 2022.



