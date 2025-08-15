---
layout: post
title:  "[2025]Structured Discourse Representation for Factual Consistency Verification"
date:   2025-08-15 15:37:36 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 


FDSpotter로 텍스트에서 추출된 원자 사실(atomic facts)과 담화 관계(discourse relations)의 존재 여부를 검증 및 분류  

DiscInfer는 적대적 담화 연결어(adversarial discourse connectives)를 포함하는 텍스트 쌍을 생성하여, 담화 관계의 모순을 식별하는 데 필요한 데이터셋 제안  

이 방법은 RDF 스타일의 삼중 구조를 확장하여, 원자 사실을 구성하는 주어, 술어, 직접 목적어, 간접 목적어, 짧은 부사구 및 보어를 포함, 원자 사실 간의 담화 관계를 보존하여 텍스트의 의미를 더 풍부하게 표현  


담화 관계는 시간적, 비교적, 조건적 관계로 나뉘며, 각 관계는 특정 연결어로 표현됩니다. 이 구조는 텍스트 간의 의미 비교를 가능하게 하여 사실 일관성을 검증하는 데 도움



짧은 요약(Abstract) :

이 논문의 초록에서는 사건이 텍스트 간에 어떻게 표현되는지의 차이를 분석하고, 언어 모델 생성물이 허위 정보를 포함하고 있는지를 검증하기 위해 콘텐츠를 체계적으로 비교할 수 있는 능력이 필요하다고 강조합니다. 이러한 비교를 지원하기 위해, 세부 정보를 포착하는 구조적 표현이 중요한 역할을 합니다. 특히, 개별 사실과 이들을 연결하는 담화 관계를 식별함으로써 더 깊은 의미 비교가 가능해집니다. 제안된 접근 방식은 구조적 담화 정보 추출과 사실 일관성 검증을 위한 분류기인 FDSpotter를 결합합니다. 연구 결과, 적대적 담화 관계가 언어 모델에 도전 과제를 제기하지만, 우리의 주석 데이터인 DiscInfer로 미세 조정함으로써 경쟁력 있는 성능을 달성할 수 있음을 보여줍니다. 이 방법은 언어 구조에 기반하여 사실 일관성 검증을 발전시키며, 이를 해석 가능한 구성 요소로 분해합니다. 우리는 데이터-텍스트 생성 및 텍스트 요약의 두 가지 작업에서 방법의 효과를 입증합니다. 우리의 코드와 데이터셋은 GitHub에서 공개될 예정입니다.


The abstract of this paper emphasizes the need for the ability to systematically compare content to analyze the differences in how events are represented across texts or to verify whether language model generations contain hallucinations. To support such comparisons, structured representations that capture fine-grained information play a vital role. In particular, identifying distinct atomic facts and the discourse relations connecting them enables deeper semantic comparisons. Our proposed approach combines structured discourse information extraction with a classifier, FDSpotter, for factual consistency verification. The findings show that adversarial discourse relations pose challenges for language models, but fine-tuning on our annotated data, DiscInfer, achieves competitive performance. This method advances factual consistency verification by grounding it in linguistic structure and decomposing it into interpretable components. We demonstrate the effectiveness of our method on the evaluation of two tasks: data-to-text generation and text summarization. Our code and dataset will be publicly available on GitHub.


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

이 논문에서 제안하는 방법은 사실 일관성 검증을 위한 구조화된 담화 표현을 사용하는 것입니다. 이 방법은 두 가지 주요 구성 요소로 이루어져 있습니다: **FDSpotter**라는 분류기와 **DiscInfer**라는 데이터셋입니다.

1. **FDSpotter**: 
   - FDSpotter는 텍스트에서 추출된 원자 사실(atomic facts)과 담화 관계(discourse relations)의 존재 여부를 검증하는 분류기입니다. 이 모델은 Transformer 기반의 아키텍처를 사용하여, 주어진 텍스트에서 특정 원자 사실이나 담화 관계가 표현될 확률을 계산합니다. 
   - FDSpotter는 두 가지 데이터 소스를 사용하여 훈련됩니다:
     - **사실 포함 데이터**: 이 데이터는 사실이 텍스트에 포함되어 있는지 여부를 판단하는 데 사용됩니다. 예를 들어, WebNLG와 같은 데이터셋에서 원자 사실과 그에 해당하는 텍스트 쌍을 사용하여 긍정 샘플을 생성하고, 원자 사실이 텍스트에서 유도될 수 없도록 변형하여 부정 샘플을 생성합니다.
     - **자연어 추론(NLI) 데이터**: NLI 데이터는 가설-전제 쌍을 포함하고 있으며, 각 가설이 전제로부터 유도되는지, 모순되는지 또는 중립적인지를 판단하는 데 사용됩니다. 이 데이터는 모델이 더 긴 텍스트에서 유도 가능한 사실을 인식할 수 있도록 돕습니다.

2. **DiscInfer 데이터셋**:
   - DiscInfer는 적대적 담화 연결어(adversarial discourse connectives)를 포함하는 텍스트 쌍을 생성하여, 담화 관계의 모순을 식별하는 데 필요한 데이터셋입니다. 이 데이터셋은 NLI 데이터에서 가설의 담화 연결어를 다른 연결어로 교체하여 생성됩니다. 
   - 이 과정에서, 교체된 연결어가 전제와 모순되는지 여부를 확인하기 위해 인간 검증자가 필요합니다. 이 데이터셋은 920개의 주석이 달린 (가설, 전제) 쌍으로 구성되어 있습니다.

3. **구조화된 정보 표현**:
   - 이 방법은 RDF 스타일의 삼중 구조를 확장하여, 원자 사실을 구성하는 주어, 술어, 직접 목적어, 간접 목적어, 짧은 부사구 및 보어를 포함합니다. 또한, 원자 사실 간의 담화 관계를 보존하여 텍스트의 의미를 더 풍부하게 표현합니다.
   - 담화 관계는 시간적, 비교적, 조건적 관계로 나뉘며, 각 관계는 특정 연결어로 표현됩니다. 이 구조는 텍스트 간의 의미 비교를 가능하게 하여 사실 일관성을 검증하는 데 도움을 줍니다.

이러한 방법론은 데이터-텍스트 생성 및 텍스트 요약 평가에서 효과적으로 검증되었으며, 기존의 방법들과 비교하여 경쟁력 있는 성능을 보여주었습니다.

---



The method proposed in this paper utilizes structured discourse representation for factual consistency verification. This approach consists of two main components: a classifier called **FDSpotter** and a dataset known as **DiscInfer**.

1. **FDSpotter**:
   - FDSpotter is a classifier designed to verify the presence of atomic facts and discourse relations extracted from text. This model employs a Transformer-based architecture to compute the probability that a specific atomic fact or discourse relation is expressed in a given text.
   - FDSpotter is trained using two data sources:
     - **Fact Inclusion Data**: This data is used to determine whether a fact is included in the text. For instance, positive samples are generated from datasets like WebNLG, where pairs of atomic facts and their corresponding texts are used, while negative samples are created by perturbing the atomic facts or the corresponding descriptive texts so that the atomic fact can no longer be entailed from the text.
     - **Natural Language Inference (NLI) Data**: NLI data contains hypothesis-premise pairs and is used to determine if each hypothesis is entailed, contradictory, or neutral with respect to the given premise text. This data helps the model recognize facts that can be inferred across longer spans of text.

2. **DiscInfer Dataset**:
   - DiscInfer is a dataset that includes text pairs with adversarial discourse connectives, which are used to identify contradictions in discourse relations. This dataset is generated by replacing one discourse connective in the selected hypothesis with another connective that could potentially render the hypothesis contradictory to the premise.
   - Human verification is required to ensure that the replaced connective indeed contradicts the premise. The dataset consists of 920 annotated (hypothesis, premise) pairs.

3. **Structured Information Representation**:
   - The method extends the RDF-style triple structure by incorporating elements such as subject, predicate, direct object, indirect object, short adverbial, and complement into the atomic facts. Additionally, it preserves the discourse relations between atomic facts to provide a richer representation of the text's meaning.
   - Discourse relations are categorized into temporal, comparative, and conditional relations, each expressed by specific connectives. This structure enables meaningful comparisons between texts, facilitating the verification of factual consistency.

This methodology has been effectively validated in the evaluation of data-to-text generation and text summarization, demonstrating competitive performance compared to existing approaches.


<br/>
# Results


이 논문에서는 사실 일관성 검증을 위한 구조화된 담화 표현을 제안하고, 이를 통해 데이터-텍스트 생성 및 텍스트 요약의 평가에서 경쟁력 있는 성능을 달성하는 방법을 설명합니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델 성능**: 다양한 모델을 사용하여 사실 일관성 검증을 수행한 결과, GPT-4 모델이 전반적으로 가장 높은 성능을 보였습니다. 예를 들어, GPT-4 Turbo 모델은 인간의 성능에 근접한 결과를 나타냈으며, Llama3.1 70B 모델도 경쟁력 있는 성능을 보였습니다. 특히, Llama3.1 8B 모델은 담화 관계 추출에서 좋은 성능을 보였지만, 원자 사실 추출에서는 어려움을 겪었습니다.

2. **테스트 데이터**: 연구에서는 Causal News Corpus와 DiscInfer 데이터셋을 사용하여 모델의 성능을 평가했습니다. DiscInfer 데이터셋은 적대적 담화 연결어를 포함한 텍스트 쌍으로 구성되어 있어, 모델이 이러한 연결어에 대한 이해도를 평가하는 데 도움을 줍니다.

3. **메트릭**: 사실 일관성 검증을 위해 Factual Inclusion Score (FI)와 Factual Overlap Score (FO)를 도입했습니다. FI는 한 텍스트의 원자 사실과 담화 관계가 다른 텍스트에 포함되어 있는지를 평가하며, FO는 두 텍스트 간의 내용 일치를 대칭적으로 평가합니다. 이 메트릭들은 기존의 n-그램 기반 메트릭보다 더 정교한 평가를 가능하게 합니다.

4. **비교**: 기존의 Open Information Extraction (OIE) 시스템과 비교했을 때, 제안된 구조화된 표현 방식은 원자 사실과 담화 관계를 통합하여 더 풍부한 정보를 제공하며, 이는 사실 일관성 검증의 성능을 향상시킵니다. 실험 결과, 원자 사실과 담화 관계를 포함한 구조화된 표현이 단순한 주어-술어-목적어 형식보다 더 나은 성능을 보였습니다.

5. **결론**: 이 연구는 사실 일관성 검증을 위한 새로운 접근 방식을 제안하며, 언어 모델이 담화 관계를 이해하는 데 있어 중요한 역할을 한다는 것을 보여줍니다. 또한, 제안된 방법은 데이터-텍스트 생성 및 텍스트 요약 평가에서 경쟁력 있는 성능을 달성하였으며, 향후 연구에서 다른 언어로의 확장 가능성을 제시합니다.



This paper proposes a structured discourse representation for factual consistency verification and describes how this approach achieves competitive performance in evaluating data-to-text generation and text summarization. The main findings of the study are as follows:

1. **Competitive Model Performance**: Various models were used to perform factual consistency verification, with the GPT-4 model generally showing the highest performance. For instance, the GPT-4 Turbo model approached human performance, while the Llama3.1 70B model also demonstrated competitive results. Notably, the Llama3.1 8B model excelled in discourse relation extraction but struggled with atomic fact extraction.

2. **Test Data**: The study utilized the Causal News Corpus and the DiscInfer dataset to evaluate model performance. The DiscInfer dataset consists of text pairs that include adversarial discourse connectives, aiding in assessing the model's understanding of these connectives.

3. **Metrics**: The study introduced the Factual Inclusion Score (FI) and the Factual Overlap Score (FO) for factual consistency verification. FI evaluates whether atomic facts and discourse relations from one text are included in another text, while FO symmetrically assesses content alignment between two texts. These metrics enable more nuanced evaluations compared to traditional n-gram-based metrics.

4. **Comparison**: Compared to existing Open Information Extraction (OIE) systems, the proposed structured representation method integrates atomic facts and discourse relations to provide richer information, enhancing the performance of factual consistency verification. Experimental results showed that structured representations including atomic facts and discourse relations outperformed simple subject-predicate-object formats.

5. **Conclusion**: This research presents a novel approach to factual consistency verification, demonstrating the significant role of discourse relations in language models' understanding. The proposed method achieved competitive performance in evaluating data-to-text generation and text summarization, suggesting potential for future research to extend this approach to other languages.


<br/>
# 예제


이 논문에서는 사실 일관성 검증을 위한 구조화된 담화 표현을 제안하고, 이를 위해 두 가지 주요 데이터셋인 훈련 데이터와 테스트 데이터를 생성합니다. 이 데이터셋은 담화 관계를 포함한 원자 사실을 추출하고, 이를 통해 텍스트 간의 사실 일관성을 평가하는 데 사용됩니다.

#### 1. 훈련 데이터 생성
훈련 데이터는 두 가지 주요 소스에서 생성됩니다:
- **사실 추출 데이터**: 이 데이터는 웹에서 수집된 사실과 그에 해당하는 텍스트 쌍으로 구성됩니다. 예를 들어, "바락 오바마는 2008년에 미국의 대통령으로 선출되었다"라는 문장에서 "바락 오바마"는 주어, "선출되었다"는 술어, "미국의 대통령"은 목적어로 구성된 원자 사실로 표현됩니다.
- **자연어 추론(NLI) 데이터**: NLI 데이터셋에서 가설-전제 쌍을 사용하여 원자 사실을 생성합니다. 예를 들어, "그는 시험에서 부정행위를 했다"라는 전제가 주어졌을 때, "그는 학교에서 퇴학당하지 않았다"라는 가설이 원자 사실로 변환될 수 있습니다.

#### 2. 테스트 데이터 생성
테스트 데이터는 훈련 데이터와 유사한 방식으로 생성되지만, 여기서는 담화 관계가 포함된 예시를 사용합니다. 예를 들어, "그는 시험에서 부정행위를 했지만, 학교에서 퇴학당하지 않았다"라는 문장에서 "하지만"이라는 접속사가 두 원자 사실을 연결하는 담화 관계로 표현됩니다.

#### 3. 예시
- **훈련 데이터 예시**:
  - 입력: "바락 오바마는 2008년에 미국의 대통령으로 선출되었다."
  - 출력: 원자 사실: ⟨바락 오바마, 선출되었다, 미국의 대통령, 2008년⟩

- **테스트 데이터 예시**:
  - 입력: "그는 시험에서 부정행위를 했지만, 학교에서 퇴학당하지 않았다."
  - 출력: 원자 사실: ⟨그, 부정행위를 했다, 시험에서⟩, 담화 관계: ⟨⟨그, 부정행위를 했다, 시험에서⟩, 하지만, ⟨그, 퇴학당하지 않았다, 학교에서⟩⟩

이러한 방식으로 훈련 데이터와 테스트 데이터가 생성되며, 이 데이터는 모델이 사실 일관성을 평가하는 데 사용됩니다.

---



This paper proposes a structured discourse representation for factual consistency verification and generates two main datasets: training data and test data. These datasets are used to extract atomic facts that include discourse relations and evaluate the factual consistency between texts.

#### 1. Training Data Generation
The training data is generated from two main sources:
- **Fact Extraction Data**: This data consists of pairs of facts and their corresponding texts collected from the web. For example, the sentence "Barack Obama was elected as the President of the USA in 2008" can be represented as an atomic fact where "Barack Obama" is the subject, "was elected" is the predicate, and "the President of the USA" is the object.
- **Natural Language Inference (NLI) Data**: Pairs of hypotheses and premises from NLI datasets are used to generate atomic facts. For instance, given the premise "He cheated on the exam," the hypothesis "He was not dismissed by the school" can be transformed into an atomic fact.

#### 2. Test Data Generation
The test data is generated similarly to the training data, but it includes examples with discourse relations. For example, in the sentence "He cheated on the exam, but he was not dismissed by the school," the conjunction "but" connects two atomic facts, which can be represented as a discourse relation.

#### 3. Examples
- **Training Data Example**:
  - Input: "Barack Obama was elected as the President of the USA in 2008."
  - Output: Atomic Fact: ⟨Barack Obama, was elected, President of the USA, in 2008⟩

- **Test Data Example**:
  - Input: "He cheated on the exam, but he was not dismissed by the school."
  - Output: Atomic Fact: ⟨He, cheated, on the exam⟩, Discourse Relation: ⟨⟨He, cheated, on the exam⟩, but, ⟨He, was not dismissed, by the school⟩⟩

In this way, the training and test datasets are generated, and they are used for the model to evaluate factual consistency.

<br/>
# 요약
이 논문에서는 사실 일관성 검증을 위한 구조적 담화 표현을 제안하며, 이를 위해 원자 사실과 담화 관계를 추출하는 FDSpotter 분류기를 사용합니다. 실험 결과, 제안된 방법이 데이터-텍스트 생성 및 텍스트 요약 평가에서 경쟁력 있는 성능을 보였으며, 특히 담화 관계를 포함한 구조적 표현이 사실 일관성 검증에 효과적임을 입증했습니다. 이 연구는 언어 모델의 사실 일관성 평가를 위한 새로운 데이터셋인 DiscInfer를 소개합니다.

---

This paper proposes a structured discourse representation for factual consistency verification, utilizing the FDSpotter classifier to extract atomic facts and discourse relations. Experimental results demonstrate that the proposed method achieves competitive performance in data-to-text generation and text summarization evaluation, particularly highlighting the effectiveness of structured representations that include discourse relations for factual consistency verification. The study introduces a new dataset, DiscInfer, for evaluating factual consistency in language models.

<br/>
# 기타


1. **테이블 1: 원자 사실 및 담화 관계 표현**
   - 이 테이블은 주어진 텍스트에서 추출된 원자 사실과 담화 관계를 보여줍니다. 원자 사실은 주어, 술어, 직접 목적어, 간접 목적어, 짧은 부사 및 보어로 구성된 튜플로 표현됩니다. 담화 관계는 두 원자 사실을 연결하는 접속사와 함께 표현됩니다.
   - **인사이트**: 원자 사실과 담화 관계를 구조적으로 표현함으로써 텍스트의 의미를 더 잘 이해하고 비교할 수 있습니다.

2. **테이블 2: 다양한 대형 언어 모델의 추출 품질**
   - 이 테이블은 다양한 모델(GPT4-Turbo, Llama3 등)의 원자 사실 및 담화 관계 추출 성능을 비교합니다. 인간의 성능과 비교하여 각 모델의 정밀도(Precision), 재현율(Recall), F1 점수를 제공합니다.
   - **인사이트**: GPT4 모델이 전반적으로 가장 높은 성능을 보이며, Llama3 모델은 원자 사실 추출에서 어려움을 겪는 것으로 나타났습니다. 이는 모델의 크기와 성능 간의 관계를 보여줍니다.

3. **테이블 3: DiscInfer 데이터셋의 정확도**
   - DiscInfer 데이터셋에서 DeBERTa와 GPT4 모델의 성능을 비교합니다. DiscInfer로 훈련된 모델이 훈련되지 않은 모델보다 성능이 크게 향상된 것을 보여줍니다.
   - **인사이트**: DiscInfer 데이터셋이 담화 관계 추출의 정확도를 높이는 데 중요한 역할을 한다는 것을 나타냅니다.

4. **테이블 4: WebNLG에서의 텍스트 일관성 평가**
   - WebNLG 벤치마크에서 다양한 메트릭(예: Factual Inclusion, Factual Overlap)과 인간 평가 간의 상관관계를 보여줍니다. Factual Inclusion과 Factual Overlap이 가장 높은 상관관계를 보입니다.
   - **인사이트**: Factual Inclusion과 Factual Overlap 메트릭이 텍스트 생성의 사실적 일관성을 평가하는 데 효과적임을 나타냅니다.

5. **테이블 5: SummEval에서의 일관성 평가**
   - SummEval 벤치마크에서 Factual Inclusion 메트릭과 기존 메트릭(BERTScore, BARTScore 등) 간의 성능을 비교합니다. Factual Inclusion이 인간 평가와 높은 상관관계를 보입니다.
   - **인사이트**: Factual Inclusion 메트릭이 요약의 사실적 일관성을 평가하는 데 유용하다는 것을 보여줍니다.

6. **어펜딕스: DiscInfer 데이터셋 생성 과정**
   - DiscInfer 데이터셋은 NLI 데이터셋에서 담화 접속사를 교체하여 생성된 예제를 포함합니다. 이 과정에서 인간 검증자가 새로운 가설-전제 쌍의 관계를 확인합니다.
   - **인사이트**: 이 데이터셋은 담화 관계의 모호성을 해결하고, 모델이 적대적 담화 관계를 인식하는 데 도움을 줍니다.



1. **Table 1: Atomic Facts and Discourse Relation Representation**
   - This table shows the extracted atomic facts and discourse relations from the given text. Atomic facts are represented as tuples consisting of subject, predicate, direct object, indirect object, short adverbial, and complement. Discourse relations are represented with the connective linking two atomic facts.
   - **Insight**: Structurally representing atomic facts and discourse relations allows for a better understanding and comparison of the text's meaning.

2. **Table 2: Quality of Extraction from Various Large Language Models**
   - This table compares the performance of various models (GPT4-Turbo, Llama3, etc.) in extracting atomic facts and discourse relations. It provides precision, recall, and F1 scores for each model compared to human performance.
   - **Insight**: The GPT4 model generally shows the highest performance, while the Llama3 model struggles with atomic fact extraction, highlighting the relationship between model size and performance.

3. **Table 3: Accuracy of the DiscInfer Dataset**
   - This table compares the performance of DeBERTa and GPT4 models on the DiscInfer dataset. It shows a significant improvement in performance for models trained with DiscInfer compared to those without.
   - **Insight**: The DiscInfer dataset plays a crucial role in enhancing the accuracy of discourse relation extraction.

4. **Table 4: Text Consistency Evaluation on WebNLG**
   - This table shows the correlation between various metrics (e.g., Factual Inclusion, Factual Overlap) and human evaluations on the WebNLG benchmark. Factual Inclusion and Factual Overlap show the highest correlations.
   - **Insight**: The Factual Inclusion and Factual Overlap metrics are effective in evaluating the factual consistency of text generation.

5. **Table 5: Consistency Evaluation on SummEval**
   - This table compares the performance of the Factual Inclusion metric with existing metrics (BERTScore, BARTScore, etc.) on the SummEval benchmark. Factual Inclusion shows a strong correlation with human evaluations.
   - **Insight**: The Factual Inclusion metric is useful for assessing the factual consistency of summaries.

6. **Appendix: Creation Process of the DiscInfer Dataset**
   - The DiscInfer dataset includes examples generated by replacing discourse connectives in NLI datasets. Human verifiers check the relationships of the new hypothesis-premise pairs.
   - **Insight**: This dataset helps address the ambiguity of discourse relations and aids models in recognizing adversarial discourse relations.

<br/>
# refer format:


### BibTeX 형식
```bibtex
@inproceedings{zhang2025structured,
  title={Structured Discourse Representation for Factual Consistency Verification},
  author={Kun Zhang and Oana Balalau and Ioana Manolescu},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={820--838},
  year={2025},
  publisher={Association for Computational Linguistics},
  address={Albuquerque, New Mexico, USA},
  date={July 27 - August 1, 2025}
}
```

### 시카고 스타일 인용
Zhang, Kun, Oana Balalau, and Ioana Manolescu. "Structured Discourse Representation for Factual Consistency Verification." In *Findings of the Association for Computational Linguistics: ACL 2025*, 820–838. Albuquerque, New Mexico, USA: Association for Computational Linguistics, July 27 - August 1, 2025.
