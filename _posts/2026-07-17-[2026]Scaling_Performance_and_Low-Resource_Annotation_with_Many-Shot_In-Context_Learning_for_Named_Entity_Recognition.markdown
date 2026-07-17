---
layout: post
title:  "[2026]Scaling Performance and Low-Resource Annotation with Many-Shot In-Context Learning for Named Entity Recognition"
date:   2026-07-17 02:56:33 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 많은 샷 인-컨텍스트 학습(ICA)을 활용하여 저자원 명명된 개체 인식(NER) 작업을 위한 데이터 주석을 자동화하는 방법을 제안합니다.


-> 많은 샷 ICL이 NER 작업에서 강력한 성능을 발휘할 수 있음을 보여주며, LLM을 활용한 주석 생성이 저자원 환경에서도 효과적이라는 것을 입증했습니다. 또한, ICA 프레임워크는 LLM의 주석을 효율적으로 활용하여, 더 작은 모델(BERT 등)을 훈련시키고 배포하는 데 있어 비용 효율성을 높일 수 있음   
결국 이를 통해 뭐 주석도 얻고 이걸로 훈련도 시키고 이득이라는 듯   


짧은 요약(Abstract) :



이 논문에서는 대형 언어 모델(LLM)을 활용한 인컨텍스트 학습(ICL)이 명명된 개체 인식(NER) 작업에서 강력한 성능을 발휘할 수 있음을 보여줍니다. 기존 연구에 따르면 LLM은 적은 양의 주석으로도 좋은 성능을 내지만, 완전 감독 모델인 BERT와 같은 모델에 비해 여전히 부족한 성능을 보입니다. 본 연구는 수백 개의 예시를 활용한 다수 샷 ICL의 가능성을 탐구하며, 이를 통해 저자원 NER 작업을 위한 데이터 주석 및 정제의 효과를 평가합니다. 실험 결과, 수백 개의 인컨텍스트 예시를 사용함으로써 LLM이 완전 감독 BERT 모델의 성능을 초과하거나 동등한 성능을 달성할 수 있음을 보여주었으며, 약 100개의 인간 주석 예시를 사용하여 생성된 고품질 주석 데이터는 기존의 최첨단 접근 방식에 비해 약 10%의 F1 점수 향상을 가져왔습니다.




This paper demonstrates that in-context learning (ICL) with large language models (LLMs) can achieve strong performance in Named Entity Recognition (NER) tasks with minimal annotation. Previous studies have shown that while LLMs perform well with limited supervision, they still lag behind fully supervised models like fine-tuned BERT. This study explores the potential of many-shot ICL, utilizing hundreds of examples, to annotate and refine data for low-resource NER tasks. Our experiments show that scaling to hundreds of in-context examples allows LLMs to match or even surpass the performance of fully supervised BERT models. Additionally, using around 100 human-labeled examples to generate high-quality annotated data leads to approximately a 10% absolute F1 improvement over existing state-of-the-art approaches.


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



이 논문에서는 이름 개체 인식(Named Entity Recognition, NER) 작업을 위한 많은 샷 인-컨텍스트 학습(many-shot in-context learning, ICL) 접근 방식을 제안합니다. 이 방법은 대규모 언어 모델(LLM)을 활용하여 적은 수의 인간 주석 예제를 기반으로 고품질의 주석 데이터를 생성하는 것을 목표로 합니다. 

#### 모델 및 아키텍처
1. **대규모 언어 모델(LLM)**: 이 연구에서는 여러 개의 LLM을 평가합니다. 여기에는 OpenAI의 GPT-4o, DeepSeekV3, Qwen-2.5(7B, 32B, 72B) 및 LLaMA-3.1(8B, 70B) 모델이 포함됩니다. 이 모델들은 긴 컨텍스트를 지원하며, 수백 개의 인-컨텍스트 예제를 처리할 수 있는 능력을 가지고 있습니다.

2. **인-컨텍스트 학습(ICL)**: NER 작업을 조건부 생성 문제로 정의하고, 주어진 입력 문장에 대해 주석이 달린 버전을 생성하는 방식으로 모델을 학습시킵니다. 이 과정에서, 모델은 주어진 예제들로부터 주석 달기 행동을 학습하고 이를 테스트 인스턴스에 적용합니다.

3. **데이터 수집 및 주석**: 연구에서는 CrossNER 벤치마크를 사용하여 다섯 개의 다양한 도메인에서 실험을 수행합니다. 각 도메인에서 100개의 인간 주석 예제를 샘플링하여 인-컨텍스트 예제로 사용하고, 이를 통해 LLM이 2,000개의 비주석 문장을 주석 달 수 있도록 합니다.

#### 특별한 기법
1. **인-컨텍스트 주석(ICA) 프레임워크**: ICA는 고품질의 주석 데이터를 생성하기 위해 LLM을 오프라인 주석자로 활용하는 방법입니다. 이 프레임워크는 주석 품질을 향상시키기 위해 세 가지 정제 전략(자기 일관성, 자기 수정, 오류 인식 정제)을 적용합니다.

2. **정제 전략**:
   - **자기 일관성**: 동일한 문장에 대해 여러 번 예측을 수행하고, 다수결 투표를 통해 최종 주석을 결정합니다.
   - **자기 수정**: LLM이 자신의 출력을 검토하고 수정하도록 유도하는 추가적인 프롬프트를 사용합니다.
   - **오류 인식 정제(EAR)**: 특정 오류 유형(스푸리어스 엔티티, 누락된 엔티티, 타입 오류)에 대해 개별적으로 처리하는 프롬프트를 사용하여 주석 품질을 향상시킵니다.

이러한 방법론을 통해, 연구는 LLM이 적은 수의 주석 예제를 기반으로도 높은 성능을 발휘할 수 있음을 보여주며, 기존의 최첨단 방법들보다 약 10%의 절대 F1 점수 향상을 달성했습니다.

---




This paper proposes a many-shot in-context learning (ICL) approach for Named Entity Recognition (NER) tasks. The goal of this method is to utilize large language models (LLMs) to generate high-quality annotated data based on a small number of human-annotated examples.

#### Models and Architecture
1. **Large Language Models (LLMs)**: The study evaluates several LLMs, including OpenAI's GPT-4o, DeepSeekV3, and Qwen-2.5 (7B, 32B, 72B) as well as LLaMA-3.1 (8B, 70B). These models support long contexts and have the capability to handle hundreds of in-context examples.

2. **In-Context Learning (ICL)**: The NER task is formulated as a conditional generation problem, where the model learns to generate an annotated version of a given input sentence. In this process, the model learns the annotation behavior from the provided examples and applies it to test instances.

3. **Data Collection and Annotation**: The research uses the CrossNER benchmark to conduct experiments across five diverse domains. In each domain, 100 human-annotated examples are sampled to serve as in-context examples, allowing the LLM to annotate 2,000 unlabeled sentences.

#### Special Techniques
1. **In-Context Annotation (ICA) Framework**: The ICA framework leverages LLMs as offline annotators to generate high-quality annotated data. This framework employs three refinement strategies (self-consistency, self-correction, and error-aware refinement) to enhance annotation quality.

2. **Refinement Strategies**:
   - **Self-Consistency**: Multiple predictions are made for the same sentence, and a majority voting mechanism is used to determine the final annotation.
   - **Self-Correction**: An additional prompt is used to encourage the LLM to review and correct its own outputs.
   - **Error-Aware Refinement (EAR)**: This strategy uses separate prompts to address specific error types (spurious entities, missing entities, and type errors) to improve annotation quality.

Through these methodologies, the study demonstrates that LLMs can achieve high performance based on a modest number of annotated examples, achieving approximately a 10% absolute F1 score improvement over existing state-of-the-art methods.


<br/>
# Results



이 논문에서는 많은 샷 인-컨텍스트 학습(Many-Shot In-Context Learning, ICL)을 활용하여 명명된 개체 인식(Named Entity Recognition, NER) 성능을 향상시키고, 저자원 주석 작업을 개선하는 방법을 제안합니다. 연구 결과는 다음과 같은 주요 요소로 요약될 수 있습니다.

1. **경쟁 모델**: 연구에서는 BERT 기반의 모델(BERT-FT)과 여러 대형 언어 모델(LLMs)인 GPT-4o, DeepSeekV3, Qwen-2.5(7B, 32B, 72B) 등을 비교했습니다. 이들 모델은 다양한 데이터셋에서 성능을 평가받았습니다.

2. **테스트 데이터**: 연구는 MIT-Movie, MIT-Restaurant, CoNLL2003, WNUT 2017 등 네 가지 데이터셋을 사용하여 ICL의 성능을 평가했습니다. 각 데이터셋은 서로 다른 도메인과 개체 유형을 포함하고 있어, 모델의 일반화 능력을 테스트하는 데 적합합니다.

3. **메트릭**: 성능 평가는 주로 F1 점수로 이루어졌습니다. F1 점수는 정밀도(Precision)와 재현율(Recall)의 조화 평균으로, 모델의 예측 정확성을 평가하는 데 유용합니다. 연구에서는 특히 ICL-MS(100-shot)와 ICA(우리의 방법) 간의 F1 점수를 비교하여 성능 향상을 입증했습니다.

4. **비교 결과**: 연구 결과에 따르면, 많은 샷 ICL을 활용한 ICA 프레임워크는 BERT-FT 모델보다 평균적으로 약 10%의 F1 점수 향상을 보였습니다. 특히, ICA는 100개의 인간 주석 예제를 사용하여 1,000개의 LLM 주석 샘플을 생성함으로써, 총 주석 비용을 크게 줄이면서도 성능을 향상시켰습니다. 예를 들어, ICA w/ EAR(오류 인식 정제 방법을 포함한 ICA)는 평균 F1 점수 0.7964를 달성하여, ICL-MS(0.7403)보다 7.6% 향상된 결과를 보였습니다.

5. **결론**: 이 연구는 많은 샷 ICL이 NER 작업에서 강력한 성능을 발휘할 수 있음을 보여주며, LLM을 활용한 주석 생성이 저자원 환경에서도 효과적이라는 것을 입증했습니다. 또한, ICA 프레임워크는 LLM의 주석을 효율적으로 활용하여, 더 작은 모델(BERT 등)을 훈련시키고 배포하는 데 있어 비용 효율성을 높일 수 있음을 강조합니다.

---




This paper proposes a method to enhance Named Entity Recognition (NER) performance and improve low-resource annotation tasks using Many-Shot In-Context Learning (ICL). The findings can be summarized in the following key elements:

1. **Competing Models**: The study compares BERT-based models (BERT-FT) with several large language models (LLMs) such as GPT-4o, DeepSeekV3, and Qwen-2.5 (7B, 32B, 72B). These models were evaluated across various datasets.

2. **Test Data**: The research utilized four datasets: MIT-Movie, MIT-Restaurant, CoNLL2003, and WNUT 2017, which encompass different domains and entity types, making them suitable for testing the models' generalization capabilities.

3. **Metrics**: Performance evaluation was primarily conducted using the F1 score, which is the harmonic mean of precision and recall, useful for assessing the accuracy of the models' predictions. The study specifically compared the F1 scores between ICL-MS (100-shot) and ICA (our method) to demonstrate performance improvements.

4. **Comparative Results**: According to the findings, the ICA framework utilizing many-shot ICL showed an average F1 score improvement of approximately 10% over the BERT-FT model. Notably, ICA generated 1,000 LLM-annotated samples using only 100 human-annotated examples, significantly reducing total annotation costs while enhancing performance. For instance, ICA w/ EAR (which includes error-aware refinement) achieved an average F1 score of 0.7964, representing a 7.6% improvement over ICL-MS (0.7403).

5. **Conclusion**: This study demonstrates that many-shot ICL can yield strong performance in NER tasks and validates that leveraging LLMs for annotation generation is effective even in low-resource settings. Furthermore, the ICA framework emphasizes the potential for efficiently utilizing LLM annotations to train and deploy smaller models (like BERT) while enhancing cost-effectiveness.


<br/>
# 예제



이 논문에서는 Named Entity Recognition (NER) 작업을 위한 많은 샷 인-컨텍스트 학습(many-shot in-context learning, ICL) 접근 방식을 제안합니다. NER은 텍스트 내에서 사람, 조직, 위치 및 기타 도메인 특정 개념과 같은 엔티티를 식별하고 분류하는 작업입니다. 이 연구에서는 LLM(대형 언어 모델)을 사용하여 NER을 수행하는 방법을 설명하고, 이를 통해 적은 양의 주석 데이터로도 높은 성능을 달성할 수 있음을 보여줍니다.

#### 예시 설명

1. **테스크 설명**: NER 작업은 주어진 문장에서 엔티티를 식별하고 해당 엔티티에 XML 스타일의 태그를 추가하는 것입니다. 예를 들어, "Canada's were the worst performing bonds."라는 문장이 주어졌을 때, "Canada"라는 단어는 위치 엔티티로 식별되어 `<entity type="location">Canada</entity>`와 같이 태그가 추가됩니다.

2. **트레이닝 데이터 예시**:
   - **입력**: "Apple is looking at buying U.K. startup for $1 billion."
   - **출력**: `<entity type="organization">Apple</entity> is looking at buying <entity type="location">U.K.</entity> startup for $1 billion.`

3. **테스트 데이터 예시**:
   - **입력**: "Barack Obama was born in Hawaii."
   - **출력**: `<entity type="person">Barack Obama</entity> was born in <entity type="location">Hawaii</entity>.`

4. **모델 학습 과정**: 모델은 주어진 트레이닝 데이터에서 엔티티의 예시를 학습하고, 이를 바탕으로 새로운 문장에서 엔티티를 식별하는 방법을 배웁니다. 예를 들어, 모델은 "Apple"과 "U.K."와 같은 단어가 각각 조직과 위치를 나타낸다는 것을 학습합니다.

5. **성능 평가**: 모델의 성능은 F1 점수로 평가되며, 이는 모델이 얼마나 정확하게 엔티티를 식별했는지를 나타냅니다. 연구에서는 LLM을 사용하여 수백 개의 예시를 제공함으로써 모델의 성능을 향상시키는 방법을 제안합니다.

이러한 방식으로, 연구는 LLM을 활용하여 적은 양의 주석 데이터로도 높은 품질의 NER 데이터를 생성할 수 있음을 보여줍니다.

---




This paper proposes a many-shot in-context learning (ICL) approach for Named Entity Recognition (NER) tasks. NER is the task of identifying and classifying entities such as persons, organizations, locations, and other domain-specific concepts within text. The study demonstrates how to perform NER using large language models (LLMs) and shows that high performance can be achieved with minimal annotation data.

#### Example Explanation

1. **Task Description**: The NER task involves identifying entities in a given sentence and adding XML-style tags to those entities. For instance, given the sentence "Canada's were the worst performing bonds.", the word "Canada" is identified as a location entity and tagged as `<entity type="location">Canada</entity>`.

2. **Training Data Example**:
   - **Input**: "Apple is looking at buying U.K. startup for $1 billion."
   - **Output**: `<entity type="organization">Apple</entity> is looking at buying <entity type="location">U.K.</entity> startup for $1 billion.`

3. **Test Data Example**:
   - **Input**: "Barack Obama was born in Hawaii."
   - **Output**: `<entity type="person">Barack Obama</entity> was born in <entity type="location">Hawaii</entity>.`

4. **Model Training Process**: The model learns from the provided training data how to identify entities, using examples to understand that words like "Apple" and "U.K." represent an organization and a location, respectively.

5. **Performance Evaluation**: The model's performance is evaluated using F1 scores, which indicate how accurately the model identifies entities. The study proposes using LLMs to provide hundreds of examples to enhance the model's performance.

Through this approach, the research demonstrates that LLMs can be utilized to generate high-quality NER data with minimal human annotation effort.

<br/>
# 요약

이 논문에서는 많은 샷 인-컨텍스트 학습(ICA)을 활용하여 저자원 명명된 개체 인식(NER) 작업을 위한 데이터 주석을 자동화하는 방법을 제안합니다. 실험 결과, 약 100개의 인간 주석 예제를 사용하여 LLM이 생성한 주석 데이터가 기존의 최첨단 방법보다 약 10%의 F1 점수 향상을 가져오는 것으로 나타났습니다. 이 방법은 LLM을 오프라인 주석 생성기로 활용하여 효율적이고 비용 효과적인 NER 데이터 세트를 구축할 수 있음을 보여줍니다.

---

This paper proposes a method for automating data annotation for low-resource Named Entity Recognition (NER) tasks using Many-Shot In-Context Learning (ICA). Experimental results show that using around 100 human-annotated examples allows LLM-generated annotations to achieve approximately 10% F1 score improvement over existing state-of-the-art methods. This approach demonstrates the potential of leveraging LLMs as offline annotators to build efficient and cost-effective NER datasets.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: XML 스타일 태깅 프롬프트를 보여주며, NER 작업의 구조적 출력을 강조합니다. 이 형식은 엔티티 경계를 명확히 하고, LLM의 경계 감지 능력을 향상시킵니다.
   - **Figure 2**: MIT-Movie, MIT-Restaurant, CoNLL2003, WNUT2017 데이터셋에서 샘플 수가 증가함에 따라 F1 점수가 어떻게 변화하는지를 보여줍니다. 이 결과는 많은 샷 ICL이 성능을 향상시킬 수 있음을 나타냅니다.
   - **Figure 3**: ICA 프레임워크의 세 가지 정제 방법(자기 일관성, 자기 수정, 오류 인식 정제)을 설명합니다. 각 방법은 LLM의 초기 예측을 개선하는 데 기여합니다.
   - **Figure 4**: AI 및 Literature 도메인에서 LLM 주석 데이터의 양이 증가함에 따라 NER 모델의 성능이 어떻게 변화하는지를 보여줍니다. 초기 1.5k-2k 주석 예제가 성능 향상에 기여함을 나타냅니다.
   - **Figure 13 및 14**: ICA의 자기 정제 방법이 주석을 어떻게 수정하는지를 보여주는 사례 연구입니다. 이들은 LLM의 초기 주석에서 발생한 오류를 수정하는 데 효과적임을 보여줍니다.

2. **테이블**
   - **Table 1**: ICA 기반 주석 전략의 성능을 여러 강력한 기준선과 비교합니다. ICA는 100개의 인간 주석 예제를 사용하여 이전 SOTA 방법보다 평균 10 F1 포인트 향상된 성능을 보여줍니다.
   - **Table 3**: 다양한 LLM의 성능을 여러 데이터셋에서 비교합니다. 많은 샷 ICL이 LLM의 성능을 향상시키는 경향이 있음을 보여줍니다.
   - **Table 5**: AI 도메인에서 클래스별 성능을 보여줍니다. ICA w/ EAR이 12/14 유형에서 가장 높은 점수를 기록하며, 이는 오류 인식 정제가 성능 향상에 기여했음을 나타냅니다.

3. **어펜딕스**
   - **Appendix A**: 많은 샷 ICL의 실험 설정 및 데이터셋 통계에 대한 세부 정보를 제공합니다. 다양한 도메인에서의 데이터셋 통계는 연구의 다양성과 적용 가능성을 강조합니다.
   - **Appendix B**: ICA 프레임워크의 정제 방법에 대한 자세한 설명을 제공합니다. 각 정제 방법의 구조와 작동 방식을 명확히 하여 연구의 신뢰성을 높입니다.

---

### Insights and Results from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: Displays the XML-style tagging prompt, emphasizing the structured output of the NER task. This format clarifies entity boundaries and enhances the boundary detection capabilities of LLMs.
   - **Figure 2**: Shows how the F1 score changes as the number of samples increases across the MIT-Movie, MIT-Restaurant, CoNLL2003, and WNUT2017 datasets. This result indicates that many-shot ICL can improve performance.
   - **Figure 3**: Illustrates the three refinement methods (self-consistency, self-correction, and error-aware refinement) in the ICA framework. Each method contributes to improving the initial predictions of the LLM.
   - **Figure 4**: Demonstrates how the performance of NER models changes as the amount of LLM-annotated data increases in the AI and Literature domains. It indicates that annotating around 1.5k-2k examples significantly contributes to performance improvement.
   - **Figures 13 and 14**: Case studies showing how the self-refinement methods correct annotations. They effectively demonstrate the ability of the LLM to rectify errors in initial annotations.

2. **Tables**
   - **Table 1**: Compares the performance of ICA-based annotation strategies against several strong baselines. It shows that ICA achieves an average improvement of approximately 10 F1 points over previous state-of-the-art methods using only 100 human-labeled examples.
   - **Table 3**: Compares the performance of various LLMs across multiple datasets. It indicates a trend where many-shot ICL enhances the performance of LLMs.
   - **Table 5**: Reports per-entity-type performance in the AI domain. It shows that ICA w/ EAR achieves the best score on 12 out of 14 types, indicating that error-aware refinement contributes significantly to performance improvement.

3. **Appendices**
   - **Appendix A**: Provides detailed statistics on the datasets used in the many-shot ICL experiments. The diversity of datasets highlights the applicability and robustness of the research.
   - **Appendix B**: Offers detailed descriptions of the refinement methods in the ICA framework. It clarifies the structure and functioning of each refinement method, enhancing the credibility of the research.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{zhang2026scaling,
  author    = {Qi Zhang and Fangping Lan and Cornelia Caragea and Longin Jan Latecki and Eduard Dragut},
  title     = {Scaling Performance and Low-Resource Annotation with Many-Shot In-Context Learning for Named Entity Recognition},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {28653--28673},
  year      = {2026},
  month     = {July 2-7},
  publisher  = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Zhang, Qi, Fangping Lan, Cornelia Caragea, Longin Jan Latecki, and Eduard Dragut. 2026. "Scaling Performance and Low-Resource Annotation with Many-Shot In-Context Learning for Named Entity Recognition." In *Findings of the Association for Computational Linguistics: ACL 2026*, 28653–28673. Association for Computational Linguistics. July 2-7.
