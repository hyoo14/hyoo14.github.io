---
layout: post
title:  "[2020]Small but Mighty: New Benchmarks for Split and Rephrase"  
date:   2024-08-24 19:04:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

이 논문은 자동 생성된 데이터셋의 문제를 해결하기 위해 사람들을 통해 크라우드소싱으로 생성된 두 개의 새로운 벤치마크 데이터셋을 제안하고, 다양한 문장 구조에서 더 정확한 평가를 제공하는 방법을 제시합니다.

짧은 요약(Abstract) :    



이 논문은 **Split and Rephrase** 작업을 위한 새로운 벤치마크 데이터셋을 제안하고 있습니다. 기존의 WebSplit 데이터셋은 자동으로 생성된 문장이 많아 자연스럽지 않고 반복적인 문장 구조로 인해 간단한 규칙 기반 모델로도 높은 성능을 얻을 수 있었습니다. 이를 해결하기 위해 저자들은 Wikipedia와 법률 계약서 문서에서 복잡한 문장을 수집하여 두 개의 새로운 데이터셋, **Wiki Benchmark (Wiki-BM)**와 **Contract Benchmark (Cont-BM)**를 크라우드소싱으로 만들었습니다. 이 데이터셋은 더 다양한 문법 구조를 포함하며, 문장을 나누고 재구성하는 작업에 더 도전적인 환경을 제공합니다. 저자들은 seq2seq 모델과 규칙 기반 모델을 사용하여 실험을 진행했으며, 기존 데이터셋에서는 규칙 기반 모델이 비슷한 성능을 보였지만, 새로운 데이터셋에서는 더 복잡한 문장 구조로 인해 성능이 낮았습니다. 평가 과정에서 BLEU 점수가 인간 평가와 일치하지 않는 문제를 지적하며, 크라우드소싱을 통해 수작업 평가를 실시하여 모델 성능을 더 정교하게 평가했습니다.

---


This paper introduces new benchmark datasets for the **Split and Rephrase** task. The existing WebSplit dataset contains many automatically generated sentences that are unnatural and repetitive, allowing even simple rule-based models to achieve high performance. To address this, the authors crowdsourced two new datasets, **Wiki Benchmark (Wiki-BM)** and **Contract Benchmark (Cont-BM)**, which collect complex sentences from Wikipedia and legal documents. These datasets contain more diverse syntactic structures and provide a more challenging environment for models performing sentence splitting and rephrasing tasks. The authors experimented with both seq2seq models and rule-based models, finding that while the rule-based model performed similarly on the old dataset, it struggled with the more complex sentences in the new datasets. Furthermore, the authors criticized the use of BLEU as a metric, highlighting its inconsistency with human evaluations, and opted for manual evaluation through crowdsourcing to better assess model performance.



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


1. **데이터셋 생성**:
   - 두 개의 새로운 벤치마크 데이터셋, **Wiki Benchmark (Wiki-BM)**와 **Contract Benchmark (Cont-BM)**를 만들기 위해 Wikipedia와 법률 계약서 문장에서 복잡한 문장을 수집했습니다.
   - 크라우드소싱을 통해 작업자들이 복잡한 문장을 나누고 단순화한 후, 다른 작업자들이 이 작업을 평가하여 데이터셋의 품질을 보장했습니다.

2. **모델 사용**:
   - 실험에는 두 가지 모델이 사용되었습니다. 첫 번째는 **sequence-to-sequence (seq2seq)** 모델이고, 두 번째는 **규칙 기반 모델**입니다.
   - seq2seq 모델은 기존 데이터셋에서 많이 사용되었고, 규칙 기반 모델은 문법적 단서를 사용하여 문장을 단순화하는 방법입니다.
   - 실험 결과, WebSplit에서는 두 모델이 비슷한 성능을 보였지만, 새로운 데이터셋에서는 규칙 기반 모델의 성능이 떨어졌습니다.

3. **평가 방법**:
   - BLEU 점수를 사용해 평가를 진행했지만, BLEU가 인간 평가와 일치하지 않는 문제를 해결하기 위해 **수작업 평가**가 사용되었습니다. 이를 위해 6가지 기준에 따라 크라우드소싱으로 모델 성능을 평가했습니다.

---


1. **Dataset Creation**:
   - Two new benchmark datasets, **Wiki Benchmark (Wiki-BM)** and **Contract Benchmark (Cont-BM)**, were created by collecting complex sentences from Wikipedia and legal contracts.
   - Through crowdsourcing, workers split and simplified the complex sentences, and other workers evaluated the quality of these simplifications to ensure dataset quality.

2. **Model Usage**:
   - Two models were used for the experiments. The first was a **sequence-to-sequence (seq2seq)** model, which has been widely used in sentence simplification tasks, and the second was a **rule-based model** that uses grammatical cues to split and simplify sentences.
   - The rule-based model performed similarly to the seq2seq model on the WebSplit dataset but struggled with the more complex structures in the newly created datasets.

3. **Evaluation Method**:
   - While BLEU scores were initially used for evaluation, the authors highlighted the inconsistency of BLEU with human judgments. To address this, they conducted manual evaluations through crowdsourcing, using six well-defined criteria to assess model performance more accurately.


<br/>
# Results  





1. **데이터셋 생성**:
   - 두 개의 새로운 벤치마크 데이터셋, **Wiki Benchmark (Wiki-BM)**와 **Contract Benchmark (Cont-BM)**를 만들기 위해 Wikipedia와 법률 계약서 문장에서 복잡한 문장을 수집했습니다.
   - 크라우드소싱을 통해 작업자들이 복잡한 문장을 나누고 단순화한 후, 다른 작업자들이 이 작업을 평가하여 데이터셋의 품질을 보장했습니다.

2. **모델 사용**:
   - 실험에는 두 가지 모델이 사용되었습니다. 첫 번째는 **sequence-to-sequence (seq2seq)** 모델이고, 두 번째는 **규칙 기반 모델**입니다.
   - 사용된 seq2seq 모델은 **Aharoni와 Goldberg (2018)**의 모델로, **Bahdanau et al. (2015)**의 **어텐션 메커니즘**과 **Gu et al. (2016)**의 **카피 메커니즘(copy mechanism)**을 기반으로 합니다. 이 모델은 문장을 최대한 변경하지 않고 원래의 의미를 유지하면서 단순화 작업을 수행합니다.
   - WebSplit에서는 두 모델이 비슷한 성능을 보였지만, 새로운 데이터셋에서는 규칙 기반 모델의 성능이 떨어졌습니다.

3. **평가 방법**:
   - BLEU 점수를 사용해 평가를 진행했지만, BLEU가 인간 평가와 일치하지 않는 문제를 해결하기 위해 **수작업 평가**가 사용되었습니다. 이를 위해 6가지 기준에 따라 크라우드소싱으로 모델 성능을 평가했습니다.

---



1. **Dataset Creation**:
   - Two new benchmark datasets, **Wiki Benchmark (Wiki-BM)** and **Contract Benchmark (Cont-BM)**, were created by collecting complex sentences from Wikipedia and legal contracts.
   - Through crowdsourcing, workers split and simplified the complex sentences, and other workers evaluated the quality of these simplifications to ensure dataset quality.

2. **Model Usage**:
   - Two models were used for the experiments. The first was a **sequence-to-sequence (seq2seq)** model, and the second was a **rule-based model**.
   - The seq2seq model was based on **Aharoni and Goldberg (2018)**, which incorporates an **attention mechanism** from **Bahdanau et al. (2015)** and a **copy mechanism** from **Gu et al. (2016)**. This model focuses on simplifying sentences while preserving the original meaning.
   - While the two models performed similarly on the WebSplit dataset, the rule-based model struggled with the more complex sentence structures in the new datasets.

3. **Evaluation Method**:
   - BLEU scores were initially used for evaluation, but since they did not align well with human judgments, **manual evaluations** were conducted through crowdsourcing, using six well-defined criteria to assess model performance more accurately.


<br/>
# 예시  


논문에서는 여러 예시가 제공되며, 특히 **Jonathan Thirkield**라는 인물을 설명하는 문장을 나누는 작업이 있습니다. 예를 들어:

- **복잡한 원문**: 
  - "Jonathan Thirkield, currently living in New York City, is an American poet who is known to be prolific."
  
- **좋은 나눔 예시** (모델이 잘 처리한 경우):
  - "Jonathan Thirkield is an American poet. He is currently living in New York City. He is known to be prolific."
  - 이 예시는 원문의 의미를 정확히 유지하고, 문장도 문법적으로 적절하며 새로운 사실을 추가하지 않았습니다. **BLEU 점수**는 이러한 경우 높은 값을 얻습니다.
  
- **잘못된 나눔 예시** (모델이 잘못 처리한 경우):
  - "Jonathan Thirkield is a poet. He is American. He is currently living somewhere. That somewhere is New York City."
  - 이 경우, 불필요하게 많은 문장으로 나뉘어지고, 일부 문장은 의미가 모호하거나 중복된 내용을 포함합니다. 이럴 때 BLEU 점수가 낮게 평가됩니다.

**BLEU 점수와 수작업 평가의 비교**:

- **WebSplit 데이터셋**에서는 규칙 기반 모델이 seq2seq 모델보다 BLEU 점수(61.3 대 56.0)가 더 높았습니다. 그러나 새로운 데이터셋인 **Wiki-BM**과 **Cont-BM**에서는 문장이 더 복잡해지면서 두 모델 모두 BLEU 점수가 크게 떨어졌습니다. 특히, **Wiki-BM**에서는 seq2seq 모델이 BLEU 점수 37.3을 기록했으며, **Cont-BM**에서는 BLEU 점수가 16.7로 매우 낮았습니다 .

이러한 결과는 BLEU 점수가 실제로 인간 평가와 잘 맞지 않으며, 문장 나누기 작업에서는 BLEU 외의 평가 지표가 필요함을 강조합니다.

---


The paper provides several examples, notably a task involving the splitting of a sentence about **Jonathan Thirkield**. For example:

- **Complex Original Sentence**: 
  - "Jonathan Thirkield, currently living in New York City, is an American poet who is known to be prolific."
  
- **Good Splitting Example** (Correctly Handled by the Model):
  - "Jonathan Thirkield is an American poet. He is currently living in New York City. He is known to be prolific."
  - This example maintains the original meaning, is grammatically correct, and does not introduce new facts. The **BLEU score** would be high in such cases.
  
- **Poor Splitting Example** (Incorrectly Handled by the Model):
  - "Jonathan Thirkield is a poet. He is American. He is currently living somewhere. That somewhere is New York City."
  - In this case, the sentence is split into too many unnecessary parts, some of which are ambiguous or redundant. The BLEU score would be low here.

**Comparison of BLEU Score and Manual Evaluation**:

- On the **WebSplit dataset**, the rule-based model scored higher on BLEU (61.3 vs. 56.0) compared to the seq2seq model. However, on the new **Wiki-BM** and **Cont-BM** datasets, as the sentences became more complex, both models saw a significant drop in BLEU scores. Specifically, on **Wiki-BM**, the seq2seq model scored 37.3, and on **Cont-BM**, it scored as low as 16.7.

These results highlight that BLEU scores do not correlate well with human evaluations and that alternative evaluation metrics are necessary for sentence splitting tasks.




<br/>  
# 요약 





이 논문은 문장을 단순화하고 분할하는 **Split and Rephrase** 작업을 위한 두 개의 새로운 벤치마크 데이터셋을 제안합니다. 기존의 자동화된 WebSplit 데이터셋의 문제점을 해결하기 위해 **Wiki-BM**과 **Cont-BM**이 크라우드소싱을 통해 만들어졌습니다. 저자들은 **seq2seq 모델**과 **규칙 기반 모델**을 사용해 실험을 진행했으며, WebSplit에서는 두 모델 모두 비슷한 성능을 보였으나, 새로운 데이터셋에서는 **seq2seq 모델**이 더 나은 성능을 보였습니다. 규칙 기반 모델은 의미역 라벨링과 의존 구문 분석을 사용해 문법적 단서를 기반으로 문장을 분할하지만, 복잡한 문장 구조에서는 어려움을 겪었습니다. BLEU 점수는 WebSplit에서 규칙 기반 모델이 높은 성능을 보였으나, 새로운 데이터셋에서는 BLEU 점수가 크게 떨어졌고 인간 평가와도 상관관계가 낮았습니다.

---


This paper introduces two new benchmark datasets, **Wiki-BM** and **Cont-BM**, for the **Split and Rephrase** task, addressing limitations in the existing WebSplit dataset. These datasets were created through crowdsourcing to ensure diverse and high-quality sentence structures. The authors used both a **seq2seq model** and a **rule-based model** for the experiments, finding similar performance on WebSplit, but the **seq2seq model** outperformed the rule-based model on the new datasets. The rule-based model, which uses **semantic role labeling** and **dependency parsing** to split sentences based on grammatical cues, struggled with more complex sentence structures. While the rule-based model performed well on WebSplit, its BLEU scores dropped significantly on the new datasets, showing poor correlation with human evaluations.

# 기타  


<br/>
# refer format:     

@inproceedings{zhang2020small,
  title={Small but Mighty: New Benchmarks for Split and Rephrase},
  author={Zhang, Li and Zhu, Huaiyu and Brahma, Siddhartha and Li, Yunyao},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={1198--1205},
  year={2020},
  organization={Association for Computational Linguistics}
}



Zhang Li, Huaiyu Zhu, Siddhartha Brahma, and Yunyao Li. "Small but Mighty: New Benchmarks for Split and Rephrase." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1198-1205. Association for Computational Linguistics, 2020.

