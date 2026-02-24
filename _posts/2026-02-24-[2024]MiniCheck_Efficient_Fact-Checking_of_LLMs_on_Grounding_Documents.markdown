---
layout: post
title:  "[2024]MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents"
date:   2026-02-24 14:26:24 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 LLM의 사실 확인을 위한 효율적인 시스템인 MiniCheck를 제안하며, 이를 위해 두 가지 합성 데이터 생성 방법(C2D 및 D2C)을 사용하여 모델을 훈련시켰습니다.


짧은 요약(Abstract) :


이 논문에서는 대형 언어 모델(LLM)의 출력이 증거에 기반하여 사실인지 확인하는 것이 자연어 처리(NLP)에서 중요한 과제임을 강조합니다. 현재의 사실 확인 접근법은 모델 생성의 각 부분을 LLM을 사용하여 잠재적 증거와 대조하는 방식으로 이루어지지만, 이 과정은 매우 계산 비용이 많이 듭니다. 본 연구에서는 GPT-4 수준의 성능을 가지면서도 400배 더 저렴한 소형 사실 확인 모델을 구축하는 방법을 제시합니다. 이를 위해 GPT-4를 사용하여 사실 오류의 현실적이고 도전적인 사례를 생성하는 합성 훈련 데이터를 구성하였습니다. 이 데이터로 훈련된 모델은 주장을 구성하는 각 사실을 확인하고 문장 간의 정보를 종합적으로 인식하는 능력을 배웁니다. 평가를 위해 최근의 사실 확인 및 LLM 생성의 기초 데이터셋을 통합하여 새로운 벤치마크인 LLM-AGGRE FACT를 만들었습니다. 우리의 최적 시스템인 MiniCheck-FT5(770M 파라미터)는 유사한 크기의 모든 시스템을 초월하며 GPT-4의 정확도에 도달합니다. 우리는 LLM-AGGRE FACT, 데이터 합성 코드 및 모델을 공개합니다.



This paper emphasizes the importance of recognizing whether the output of large language models (LLMs) can be grounded in evidence, which is central to many tasks in natural language processing (NLP). Current approaches to this kind of fact-checking involve verifying each piece of a model's generation against potential evidence using an LLM, but this process can be very computationally expensive. In this work, we show how to build small fact-checking models that achieve GPT-4-level performance at 400 times lower cost. We do this by constructing synthetic training data with GPT-4, which involves creating realistic yet challenging instances of factual errors through a structured generation procedure. Training on this data teaches models to check each fact in the claim and recognize the synthesis of information across sentences. For evaluation, we unify datasets from recent work on fact-checking and grounding LLM generations into a new benchmark, LLM-AGGRE FACT. Our best system, MiniCheck-FT5 (770M parameters), outperforms all systems of comparable size and reaches GPT-4 accuracy. We release LLM-AGGRE FACT, code for data synthesis, and models.


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




**MiniCheck: 효율적인 사실 확인 시스템**

MiniCheck는 대형 언어 모델(LLM)의 출력을 사실 확인하는 데 필요한 효율적인 시스템으로, 특히 문서 기반의 사실 확인 작업에 중점을 두고 있습니다. 이 시스템은 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. **모델 아키텍처**: MiniCheck는 Flan-T5 아키텍처를 기반으로 하며, 이는 다양한 자연어 처리(NLP) 작업에 적합하도록 설계된 모델입니다. Flan-T5는 사전 훈련된 T5 모델을 기반으로 하여, 특정 작업에 맞게 미세 조정(fine-tuning)된 버전입니다. MiniCheck는 770M 파라미터를 가진 Flan-T5 모델을 사용하여, 상대적으로 작은 크기에도 불구하고 높은 성능을 발휘합니다.

2. **트레이닝 데이터**: MiniCheck는 두 가지 주요 방법을 통해 합성 데이터를 생성하여 훈련합니다. 첫 번째 방법인 Claim to Document (C2D) 방법은 주어진 주장(claim)을 기반으로 여러 문장을 포함하는 문서를 생성합니다. 이 과정에서 GPT-4를 사용하여 주장을 지원하는 문장 쌍을 생성하고, 이를 통해 문서가 주장을 지지하는지 여부를 확인합니다. 두 번째 방법인 Document to Claim (D2C) 방법은 실제 문서에서 요약 문장을 생성하고, 이 요약 문장이 문서의 여러 부분과 어떻게 연결되는지를 평가합니다.

3. **특별한 기법**: MiniCheck는 사실 확인의 복잡성을 해결하기 위해 두 가지 합성 데이터 생성 방법을 사용합니다. C2D 방법은 주장을 여러 개의 원자적 사실(atomic facts)으로 분해하고, 이들 사실이 문서에서 어떻게 지지되는지를 평가합니다. D2C 방법은 문서의 여러 청크(chunk)에서 요약 문장을 생성하고, 이들이 주장을 지지하는지를 평가합니다. 이러한 접근 방식은 모델이 여러 문장 간의 관계를 이해하고, 주장의 사실성을 평가하는 데 필요한 복잡한 추론을 수행할 수 있도록 돕습니다.

4. **효율성**: MiniCheck는 기존의 LLM 기반 사실 확인 시스템에 비해 400배 저렴한 비용으로 GPT-4 수준의 성능을 달성합니다. 이는 모델이 각 주장을 검증하는 데 필요한 계산 비용을 크게 줄여줍니다. MiniCheck는 문서 기반의 사실 확인 작업에서 높은 정확도를 유지하면서도 효율성을 극대화하는 데 중점을 두고 있습니다.

이러한 구성 요소와 기법을 통해 MiniCheck는 LLM의 출력을 효과적으로 사실 확인할 수 있는 강력한 도구로 자리 잡고 있습니다.

---



**MiniCheck: An Efficient Fact-Checking System**

MiniCheck is an efficient system designed for fact-checking the outputs of large language models (LLMs), with a particular focus on document-grounded fact-checking tasks. This system consists of the following key components:

1. **Model Architecture**: MiniCheck is based on the Flan-T5 architecture, which is designed to be suitable for various natural language processing (NLP) tasks. Flan-T5 is a fine-tuned version of the pre-trained T5 model, optimized for specific tasks. MiniCheck utilizes a Flan-T5 model with 770M parameters, achieving high performance despite its relatively small size.

2. **Training Data**: MiniCheck trains on synthetic data generated through two main methods. The first method, Claim to Document (C2D), generates documents containing multiple sentences based on a given claim. In this process, GPT-4 is used to create sentence pairs that support the claim, allowing for verification of whether the document supports the claim. The second method, Document to Claim (D2C), generates summary sentences from actual documents and evaluates how these summaries relate to the various parts of the document.

3. **Special Techniques**: MiniCheck employs two synthetic data generation methods to address the complexities of fact-checking. The C2D method decomposes claims into multiple atomic facts and assesses how these facts are supported by the document. The D2C method generates summary sentences from various chunks of the document and evaluates whether these summaries support the claim. This approach helps the model understand relationships across multiple sentences and perform the complex reasoning required to assess the factuality of claims.

4. **Efficiency**: MiniCheck achieves GPT-4-level performance at a cost that is 400 times lower than existing LLM-based fact-checking systems. This significantly reduces the computational cost required to verify each claim. MiniCheck focuses on maximizing efficiency while maintaining high accuracy in document-based fact-checking tasks.

Through these components and techniques, MiniCheck establishes itself as a powerful tool for effectively fact-checking the outputs of LLMs.


<br/>
# Results


이 논문에서는 MiniCheck라는 새로운 사실 확인 시스템을 제안하고 있습니다. MiniCheck는 GPT-4 수준의 성능을 가지면서도 400배 더 저렴한 비용으로 작동합니다. 이 시스템은 두 가지 주요 방법론을 통해 훈련된 모델을 사용하여, LLM(대형 언어 모델) 생성물의 사실성을 검증합니다. 

#### 경쟁 모델
MiniCheck는 여러 기존의 사실 확인 모델과 비교됩니다. 특히 AlignScore, T5-NLI-Mixed, QAFactEval, SummaC-ZS, SummaC-CV와 같은 모델들이 포함됩니다. 이들 모델은 각각 다른 접근 방식을 사용하여 사실 확인을 수행하며, MiniCheck는 이들보다 더 나은 성능을 보여줍니다.

#### 테스트 데이터
MiniCheck는 LLM-A GGRE FACT라는 새로운 벤치마크를 사용하여 평가됩니다. 이 벤치마크는 10개의 기존 데이터셋을 통합하여 구성되었으며, 각 데이터셋은 LLM 생성물의 사실성을 평가하기 위해 인간이 주석을 달았습니다. 데이터셋은 CNN, XSum, MediaSum, WICE, REVEAL, CLAIM VERIFY, FACTCHECK-GPT, EXPERT QA, LFQA 등 다양한 출처에서 수집되었습니다.

#### 메트릭
모델의 성능은 균형 정확도(Balanced Accuracy, BAcc)로 측정됩니다. BAcc는 TP(참 긍정), TN(참 부정), FP(거짓 긍정), FN(거짓 부정)의 비율을 기반으로 하여 계산됩니다. MiniCheck는 이 메트릭에서 기존의 모든 모델을 초과하는 성능을 보여주었습니다.

#### 비교
MiniCheck-FT5 모델은 69.9%의 BAcc를 기록하며, AlignScore보다 4.3% 높은 성능을 보였습니다. 또한, MiniCheck는 GPT-4와 유사한 성능을 보이면서도 훨씬 더 적은 비용으로 운영될 수 있습니다. 예를 들어, MiniCheck-FT5는 770M 파라미터를 가진 모델로, 35K의 훈련 데이터로 훈련되었습니다. 반면, AlignScore는 4,700K의 훈련 데이터를 사용합니다. 이러한 차이는 MiniCheck의 효율성을 더욱 부각시킵니다.

결론적으로, MiniCheck는 기존의 사실 확인 모델들에 비해 성능과 비용 효율성 모두에서 우수한 결과를 보여주며, LLM의 사실 확인 작업에 있어 새로운 기준을 제시합니다.

---




This paper introduces a new fact-checking system called MiniCheck, which operates at a cost 400 times lower than that of GPT-4 while achieving similar performance levels. The system utilizes models trained through two main methodologies to verify the factuality of outputs generated by large language models (LLMs).

#### Competing Models
MiniCheck is compared against several existing fact-checking models, including AlignScore, T5-NLI-Mixed, QAFactEval, SummaC-ZS, and SummaC-CV. Each of these models employs different approaches to perform fact-checking, and MiniCheck demonstrates superior performance compared to them.

#### Test Data
MiniCheck is evaluated using a new benchmark called LLM-A GGRE FACT. This benchmark aggregates 10 existing datasets, each annotated by humans to assess the factuality of LLM-generated outputs. The datasets are sourced from various origins, including CNN, XSum, MediaSum, WICE, REVEAL, CLAIM VERIFY, FACTCHECK-GPT, EXPERT QA, and LFQA.

#### Metrics
The performance of the models is measured using Balanced Accuracy (BAcc). BAcc is calculated based on the ratios of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). MiniCheck outperforms all existing models on this metric.

#### Comparison
The MiniCheck-FT5 model achieved a BAcc of 69.9%, which is 4.3% higher than that of AlignScore. Additionally, MiniCheck exhibits similar performance to GPT-4 while being significantly more cost-effective. For instance, MiniCheck-FT5 is a model with 770M parameters trained on 35K data points, whereas AlignScore uses 4,700K training data points. This difference further highlights the efficiency of MiniCheck.

In conclusion, MiniCheck presents superior results in both performance and cost-effectiveness compared to existing fact-checking models, setting a new standard for fact-checking tasks involving LLMs.


<br/>
# 예제



이 논문에서는 MiniCheck라는 효율적인 사실 확인 시스템을 제안합니다. 이 시스템은 대규모 언어 모델(LLM)의 출력을 검증하기 위해 설계되었으며, 특히 문서에 기반한 사실 확인을 수행합니다. MiniCheck는 GPT-4 수준의 성능을 가지면서도 비용은 400배 낮은 모델입니다. 이 시스템은 두 가지 주요 방법론을 통해 훈련 데이터를 생성합니다: Claim to Document (C2D)와 Document to Claim (D2C)입니다.

#### 1. Claim to Document (C2D) 방법
- **입력**: 주어진 주장(claim) c를 입력으로 받습니다. 예를 들어, "바르셀로나는 스페인의 수도이다."라는 주장을 사용할 수 있습니다.
- **출력**: 이 주장을 지원하는 문서 D를 생성합니다. 문서 D는 주장을 뒷받침하는 여러 문장을 포함해야 하며, 이 문장들은 서로 결합되어야만 주장을 지지할 수 있습니다.
- **예시**:
  - 주장: "바르셀로나는 스페인의 수도이다."
  - 생성된 문서: "스페인의 수도는 마드리드이다. 그러나 바르셀로나는 스페인에서 가장 큰 도시 중 하나이다."

#### 2. Document to Claim (D2C) 방법
- **입력**: 인간이 작성한 문서 D를 입력으로 받습니다. 예를 들어, "스페인의 수도는 마드리드이다."라는 문서가 있을 수 있습니다.
- **출력**: 이 문서에 기반하여 주장을 생성합니다. 이 주장은 문서의 여러 문장을 결합하여 만들어지며, 주장을 지지하는 문서의 특정 부분을 참조해야 합니다.
- **예시**:
  - 문서: "스페인의 수도는 마드리드이다. 바르셀로나는 스페인에서 가장 큰 도시 중 하나이다."
  - 생성된 주장: "바르셀로나는 스페인에서 가장 큰 도시이다."

#### 3. 훈련 및 평가
- MiniCheck는 C2D와 D2C 방법을 통해 생성된 데이터로 훈련됩니다. 훈련 데이터는 각 주장과 문서 쌍에 대해 지원 여부를 레이블링합니다.
- 평가를 위해 LLM-AGGRE FACT라는 새로운 벤치마크를 사용하여 다양한 모델의 성능을 비교합니다. 이 벤치마크는 사실 확인의 정확성을 평가하기 위해 여러 데이터셋을 통합합니다.





This paper proposes an efficient fact-checking system called MiniCheck, designed to validate the outputs of large language models (LLMs), particularly focusing on document-grounded fact-checking. MiniCheck achieves performance comparable to GPT-4 while being 400 times cheaper. The system generates training data through two main methodologies: Claim to Document (C2D) and Document to Claim (D2C).

#### 1. Claim to Document (C2D) Method
- **Input**: Takes a given claim c as input. For example, "Barcelona is the capital of Spain."
- **Output**: Generates a document D that supports this claim. The document D must contain multiple sentences that, when combined, support the claim.
- **Example**:
  - Claim: "Barcelona is the capital of Spain."
  - Generated Document: "The capital of Spain is Madrid. However, Barcelona is one of the largest cities in Spain."

#### 2. Document to Claim (D2C) Method
- **Input**: Takes a human-written document D as input. For example, "The capital of Spain is Madrid."
- **Output**: Generates a claim based on this document. The claim should reference specific parts of the document that support it.
- **Example**:
  - Document: "The capital of Spain is Madrid. Barcelona is one of the largest cities in Spain."
  - Generated Claim: "Barcelona is the largest city in Spain."

#### 3. Training and Evaluation
- MiniCheck is trained on data generated from both C2D and D2C methods. The training data is labeled for support or non-support for each claim-document pair.
- For evaluation, a new benchmark called LLM-AGGRE FACT is used to compare the performance of various models. This benchmark aggregates multiple datasets to assess the accuracy of fact-checking.

<br/>
# 요약
이 논문에서는 LLM의 사실 확인을 위한 효율적인 시스템인 MiniCheck를 제안하며, 이를 위해 두 가지 합성 데이터 생성 방법(C2D 및 D2C)을 사용하여 모델을 훈련시켰습니다. 실험 결과, MiniCheck는 기존의 전문화된 사실 확인 시스템보다 4%에서 10% 더 높은 성능을 보였으며, GPT-4와 유사한 정확도를 달성했습니다. 이 연구는 LLM의 사실 확인을 위한 새로운 벤치마크인 LLM-AGGRE FACT를 소개하고, 저비용으로 높은 성능을 제공하는 모델을 개발하는 데 기여합니다.

---

This paper proposes an efficient system for fact-checking LLMs called MiniCheck, utilizing two synthetic data generation methods (C2D and D2C) for model training. Experimental results show that MiniCheck outperforms existing specialized fact-checking systems by 4% to 10% and achieves accuracy comparable to GPT-4. The study introduces a new benchmark for LLM fact-checking, LLM-AGGRE FACT, contributing to the development of high-performance models at a low cost.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 다양한 사실 확인 설정을 통합하여 문서 기반 사실 확인의 필요성을 강조합니다. 이 다이어그램은 MiniCheck 시스템이 문서에서 여러 사실을 동시에 검증하는 방법을 시각적으로 설명합니다.
   - **Figure 2**: LLM이 생성한 요약 문장과 관련된 대화 스니펫을 보여줍니다. 이 예시는 LLM이 생성한 문장이 여러 개의 원자적 사실로 나뉘어져 있음을 보여줍니다.
   - **Figure 3**: C2D 및 D2C 방법론을 통해 생성된 데이터의 흐름을 시각적으로 나타냅니다. 이 피규어는 각 단계에서의 데이터 생성 과정을 명확히 보여줍니다.

2. **테이블**
   - **Table 1**: C2D 및 D2C 방법으로 생성된 합성 데이터의 통계 정보를 제공합니다. 이 표는 데이터의 크기, 고유한 주장 및 문서 수, 평균 문서 및 주장 길이, 부정적인 주장 비율을 포함합니다.
   - **Table 2**: MiniCheck 모델의 성능을 다른 모델과 비교한 결과를 보여줍니다. MiniCheck-FT5는 AlignScore보다 4% 높은 성능을 보이며, GPT-4와 유사한 성능을 나타냅니다.
   - **Table 6**: MiniCheck 모델의 훈련 데이터에서 C2D 및 D2C의 제거가 성능에 미치는 영향을 보여주는 ablation study 결과입니다. 두 가지 합성 데이터 모두 모델 성능에 긍정적인 영향을 미친다는 것을 확인할 수 있습니다.

3. **어펜딕스**
   - **Appendix A**: MiniCheck 모델의 성능을 평가하기 위한 추가 분석 및 ablation study 결과를 포함합니다. 이 섹션은 모델의 강점과 약점을 더 깊이 이해하는 데 도움을 줍니다.
   - **Appendix D**: C2D 및 D2C 방법으로 생성된 데이터의 예시를 제공합니다. 이 예시는 합성 데이터의 질을 평가하는 데 유용합니다.
   - **Appendix H**: 합성 데이터 생성 및 평가에 사용된 프롬프트를 포함합니다. 이 섹션은 데이터 생성 과정의 투명성을 높이고, 재현 가능성을 보장합니다.

### Insights from Diagrams, Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Figure 1**: Highlights the necessity of document-based fact-checking by integrating various fact-checking settings. This diagram visually explains how the MiniCheck system verifies multiple facts simultaneously from documents.
   - **Figure 2**: Displays a dialogue snippet related to an LLM-generated summary sentence. This example illustrates how the LLM-generated sentence can be broken down into several atomic facts.
   - **Figure 3**: Visually represents the flow of data generated through the C2D and D2C methodologies. This figure clarifies the data generation process at each step.

2. **Tables**
   - **Table 1**: Provides statistical information about the synthetic data generated through the C2D and D2C methods. This table includes the size of the data, the number of unique claims and documents, the average length of documents and claims, and the proportion of unsupported claims.
   - **Table 2**: Shows the performance of the MiniCheck models compared to other models. MiniCheck-FT5 outperforms AlignScore by 4% and demonstrates performance similar to GPT-4.
   - **Table 6**: Presents the results of an ablation study showing the impact of removing C2D and D2C from the training data on the performance of the MiniCheck models. It confirms that both types of synthetic data positively influence model performance.

3. **Appendices**
   - **Appendix A**: Contains additional analyses and results from ablation studies to evaluate the performance of the MiniCheck models. This section helps in understanding the strengths and weaknesses of the models in greater depth.
   - **Appendix D**: Provides examples of data generated using the C2D and D2C methods. These examples are useful for assessing the quality of the synthetic data.
   - **Appendix H**: Includes prompts used for generating and evaluating synthetic data. This section enhances the transparency of the data generation process and ensures reproducibility.

These components collectively contribute to a comprehensive understanding of the methodologies, results, and implications of the research presented in the paper.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Tang2024,
  author = {Liyan Tang and Philippe Laban and Greg Durrett},
  title = {MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages = {8818--8847},
  year = {2024},
  month = {November},
  publisher = {Association for Computational Linguistics},
  url = {https://github.com/Liyan06/MiniCheck}
}
```

### Chicago Style Citation

Liyan Tang, Philippe Laban, and Greg Durrett. "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents." In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 8818–8847. Association for Computational Linguistics, November 2024. https://github.com/Liyan06/MiniCheck.
