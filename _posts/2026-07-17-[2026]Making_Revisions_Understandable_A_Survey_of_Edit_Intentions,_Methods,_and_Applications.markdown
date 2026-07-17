---
layout: post
title:  "[2026]Making Revisions Understandable: A Survey of Edit Intentions, Methods, and Applications"
date:   2026-07-17 02:55:10 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 텍스트 수정의 이해를 돕기 위해 수정 의도, 방법 및 응용에 대한 설문조사를 수행하였다.  
(수정 의도(edit intentions)를 이해하도록, 수동/자동/하이브리드 주석을 얻어오고 이로 분석.. 데이터셋 제안 겸 분석  )  
  

짧은 요약(Abstract) :


이 논문의 초록에서는 텍스트 수정이 문서 작성 과정에서 핵심적인 과정임을 강조하고 있습니다. 저자들은 수정 이력이 대규모로 수집될 수 있는 플랫폼(예: 위키피디아, arXiv)의 증가로 인해, 자연어 처리(NLP) 연구가 단순히 어떤 변화가 이루어졌는지를 모델링하는 것을 넘어, 왜 이러한 변화가 이루어졌는지를 이해하는 방향으로 나아가고 있다고 설명합니다. 즉, 수정 의도(edit intentions)를 이해하는 것이 중요하다는 것입니다. 이 논문은 수정 의도를 중심으로 텍스트 수정 연구를 종합적으로 정리한 첫 번째 조사로, 데이터셋, 분류 체계, 식별 방법 및 응용 프로그램에 대한 통합된 관점을 제공합니다. 저자들은 수정 작업 흐름 전반에 걸쳐 이전 연구를 검토하고, 대표적인 데이터셋과 방법을 분류하며, 글쓰기 지원 및 문서 수정 요약과 같은 하위 응용 프로그램을 요약하고, 주요 연구 방향을 제시합니다.




The abstract of this paper emphasizes that text revision is a core process in document creation. The authors explain that with the increasing availability of large-scale revision histories from platforms like Wikipedia and arXiv, NLP research has begun to move beyond merely modeling what changes are made to understanding why these changes are made, i.e., the underlying edit intentions. This paper is the first survey that synthesizes text revision research through the lens of edit intentions, providing a unified view of datasets, taxonomies, identification methods, and applications. The authors review prior work across the full revision workflow, categorize representative datasets and methods, summarize downstream applications such as writing assistance and document edit summarization, and highlight key open research directions.


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



이 논문에서는 텍스트 수정 과정에서의 편집 의도를 이해하고 모델링하기 위한 다양한 방법론을 제시합니다. 편집 의도 식별을 위한 접근 방식은 크게 수동 주석과 자동 주석으로 나눌 수 있습니다.

1. **수동 주석 (Manual Annotation)**:
   - 수동 주석은 일반적으로 Amazon Mechanical Turk와 같은 크라우드소싱 플랫폼이나 훈련된 학생들을 통해 수행됩니다. 주석자는 언어 능력과 특정 도메인 또는 편집 전문성을 갖춘 사람들로 선정됩니다.
   - 주석의 일관성을 보장하기 위해, 연구자들은 구조화된 교육 절차를 사용하여 주석자에게 상세한 지침, 예시, 실시간 시연 및 피드백을 제공합니다.
   - 주석의 신뢰성은 Krippendorff의 α와 MASI와 같은 상호 주석자 일치 메트릭을 사용하여 평가됩니다.

2. **자동 주석 (Automatic Annotation)**:
   - 자동 편집 의도 식별은 전통적인 기계 학습 방법과 신경망 접근 방식을 모두 사용하여 탐구되었습니다. 초기 연구는 이진 또는 다중 레이블 분류 문제로 편집 의도 식별을 공식화하며, 텍스트 차이, 담화 신호 및 메타데이터에서 파생된 특성을 사용합니다.
   - 최근의 접근 방식은 원본 텍스트와 수정된 텍스트를 함께 인코딩하여 의미적 의도를 더 잘 포착할 수 있는 신경망 및 대형 언어 모델(LLM)을 채택합니다. 특히, LLM을 사용한 인컨텍스트 학습은 특정 작업 훈련 없이 유연한 의도 생성을 가능하게 하지만, 계산 비용, 재현성 및 출력 일관성과 관련된 도전 과제를 동반합니다.

3. **하이브리드 전략 (Hybrid Strategies)**:
   - 많은 연구는 수동 주석 데이터셋을 금 표준으로 사용하여 자동 모델의 훈련, 평가 또는 보정을 수행하는 하이브리드 전략을 채택합니다. 이는 주석 품질과 확장성을 균형 있게 유지하는 방법입니다.

이러한 방법론은 편집 의도 식별의 정확성을 높이고, 다양한 도메인에서의 적용 가능성을 확장하는 데 기여합니다. 특히, 편집 의도는 텍스트 수정의 동적 과정을 이해하고, 쓰기 지원 시스템 및 문서 편집 요약과 같은 다운스트림 응용 프로그램에서 중요한 역할을 합니다.

---




This paper presents various methodologies for understanding and modeling edit intentions in the text revision process. The approaches for edit intention identification can be broadly categorized into manual annotation and automatic annotation.

1. **Manual Annotation**:
   - Manual annotation is typically conducted through crowdsourcing platforms like Amazon Mechanical Turk or by trained students. Annotators are selected based on their language proficiency and, in some cases, domain or editing expertise.
   - To ensure annotation consistency, researchers employ structured training procedures that include detailed guidelines, illustrative examples, live demonstrations, and iterative practice with feedback.
   - The reliability of annotations is assessed using inter-annotator agreement metrics such as Krippendorff's α and MASI.

2. **Automatic Annotation**:
   - Automatic edit intention identification has been explored using both traditional machine learning methods and neural approaches. Early work formulates the task as binary or multi-label classification, utilizing features derived from textual differences, discourse cues, and metadata.
   - More recent approaches adopt neural networks and large language models (LLMs) that encode the original and revised texts jointly, capturing semantic intent more effectively. In particular, in-context learning with LLMs allows for flexible intent generation without task-specific training, but introduces challenges related to computational cost, reproducibility, and output consistency.

3. **Hybrid Strategies**:
   - Many studies adopt hybrid strategies where manually annotated datasets serve as gold standards for training, evaluation, or calibration of automatic models. This balances annotation quality with scalability.

These methodologies contribute to enhancing the accuracy of edit intention identification and expanding applicability across various domains. Notably, edit intentions play a crucial role in understanding the dynamic processes of text revision and in downstream applications such as writing assistance systems and document edit summarization.


<br/>
# Results



이 논문에서는 텍스트 수정 과정에서의 편집 의도(edit intention)를 이해하고 분석하기 위한 연구를 종합적으로 다루고 있습니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 편집 의도 식별을 위한 다양한 모델이 제안되었습니다. 초기 연구는 전통적인 기계 학습 모델을 사용하여 이진 또는 다중 레이블 분류 문제로 편집 의도를 모델링했습니다. 최근에는 신경망 기반 모델과 대형 언어 모델(LLM)을 활용하여 원본 텍스트와 수정된 텍스트를 동시에 인코딩하고, 의미적 의도를 더 잘 포착하는 방법이 제안되었습니다. 이러한 모델들은 편집 의도를 더 정확하게 식별할 수 있는 가능성을 보여주었습니다.

2. **테스트 데이터**: 연구에서 사용된 데이터셋은 주로 위키피디아, 학술 논문, 학생 에세이 등에서 수집되었습니다. 각 데이터셋은 편집 의도와 편집 행동이 주석 처리되어 있으며, 이는 모델 훈련과 평가에 중요한 역할을 합니다. 예를 들어, Yang et al. (2017)에서는 5,700개의 문장 쌍을 포함한 데이터셋을 사용하여 편집 의도를 식별했습니다.

3. **메트릭**: 편집 의도 식별 모델의 성능을 평가하기 위해 다양한 메트릭이 사용되었습니다. 예를 들어, 정확도(ACC), 정밀도(Precision), 재현율(Recall), F1 점수 등이 있습니다. 다중 레이블 분류의 경우, 예제 기반 메트릭과 레이블 기반 메트릭이 사용되며, 각 메트릭은 모델의 성능을 다각도로 평가할 수 있도록 설계되었습니다.

4. **비교**: 기존의 수작업 주석 방식과 자동 주석 방식 간의 비교가 이루어졌습니다. 수작업 주석은 높은 정확도를 제공하지만 비용이 많이 들고 확장성이 떨어지는 반면, 자동 주석 방식은 대규모 분석이 가능하지만 도메인 변화에 민감하다는 한계가 있습니다. 따라서 많은 연구가 수작업 주석 데이터셋을 금 표준으로 사용하여 자동 모델의 훈련 및 평가에 활용하는 하이브리드 전략을 채택하고 있습니다.

이러한 결과들은 편집 의도 연구의 발전을 위한 기초 자료를 제공하며, 향후 연구 방향에 대한 통찰을 제공합니다.

---




This paper comprehensively addresses the research on understanding and analyzing edit intentions in the text revision process. The main findings are as follows:

1. **Competing Models**: Various models have been proposed for identifying edit intentions. Early studies utilized traditional machine learning models to frame the task as a binary or multi-label classification problem. More recent approaches have employed neural network-based models and large language models (LLMs) to jointly encode the original and revised texts, capturing semantic intent more effectively. These models demonstrate the potential for more accurate identification of edit intentions.

2. **Test Data**: The datasets used in the research were primarily collected from platforms like Wikipedia, academic papers, and student essays. Each dataset is annotated with edit intentions and edit actions, playing a crucial role in model training and evaluation. For instance, Yang et al. (2017) utilized a dataset containing 5,700 sentence pairs to identify edit intentions.

3. **Metrics**: Various metrics were employed to evaluate the performance of edit intention identification models. These include accuracy (ACC), precision, recall, and F1 score. In the case of multi-label classification, both example-based and label-based metrics were used, allowing for a multifaceted assessment of model performance.

4. **Comparison**: A comparison was made between manual annotation methods and automatic annotation methods. Manual annotation provides high accuracy but is costly and lacks scalability, while automatic methods allow for large-scale analysis but are sensitive to domain shifts. Consequently, many studies adopt hybrid strategies where manually annotated datasets serve as gold standards for training and evaluating automatic models.

These findings provide foundational insights for the advancement of edit intention research and offer perspectives on future research directions.


<br/>
# 예제



이 논문에서는 텍스트 수정 과정에서의 편집 의도(edit intention)를 이해하고 모델링하기 위한 다양한 방법과 데이터셋을 다루고 있습니다. 특히, 편집 의도를 식별하기 위한 트레이닝 데이터와 테스트 데이터의 구체적인 예시를 통해 이 과정을 설명하겠습니다.

#### 트레이닝 데이터 예시
트레이닝 데이터는 일반적으로 원본 텍스트와 수정된 텍스트 쌍으로 구성됩니다. 예를 들어, 다음과 같은 문장이 있을 수 있습니다:

- **원본 텍스트**: "그녀는 병으로 죽었다."
- **수정된 텍스트**: "그녀는 1949년에 병으로 죽었다."

이 경우, 편집 의도는 "정보 추가"로 분류될 수 있습니다. 이 데이터는 모델이 편집 의도를 학습하는 데 사용됩니다.

#### 테스트 데이터 예시
테스트 데이터는 모델의 성능을 평가하기 위해 사용됩니다. 예를 들어, 다음과 같은 문장이 있을 수 있습니다:

- **원본 텍스트**: "그는 책을 읽고 있다."
- **수정된 텍스트**: "그는 새로운 책을 읽고 있다."

모델은 이 수정된 텍스트를 보고 "정보 추가"라는 편집 의도를 예측해야 합니다.

#### 구체적인 태스크
이러한 데이터셋을 사용하여 수행하는 태스크는 주로 다음과 같습니다:

1. **편집 의도 식별**: 주어진 수정된 텍스트에 대해 어떤 편집 의도가 있는지를 분류하는 작업입니다. 이는 다중 클래스 분류 문제로 모델이 각 수정된 텍스트에 대해 적절한 편집 의도를 선택하도록 합니다.

2. **편집 요약**: 여러 개의 수정된 텍스트를 요약하여 어떤 변경이 있었는지를 설명하는 작업입니다. 이 작업은 편집 의도를 기반으로 하여 수행됩니다.

이러한 태스크는 자연어 처리(NLP) 분야에서 중요한 연구 주제이며, 다양한 응용 프로그램에서 활용될 수 있습니다. 예를 들어, 글쓰기 보조 도구나 문서 편집 요약 시스템에서 이러한 편집 의도 식별 기술이 사용될 수 있습니다.

---




This paper discusses various methods and datasets for understanding and modeling edit intentions in the text revision process. Specifically, I will explain the training and testing data examples systematically to illustrate this process.

#### Training Data Example
Training data typically consists of pairs of original and revised texts. For example, consider the following sentences:

- **Original Text**: "She died from an illness."
- **Revised Text**: "She died in 1949 from an illness."

In this case, the edit intention can be classified as "information addition." This data is used to train the model to learn edit intentions.

#### Testing Data Example
Testing data is used to evaluate the performance of the model. For instance, the following sentences might be used:

- **Original Text**: "He is reading a book."
- **Revised Text**: "He is reading a new book."

The model should predict the edit intention as "information addition" based on the revised text.

#### Specific Tasks
The datasets are used to perform tasks primarily such as:

1. **Edit Intention Identification**: This task involves classifying what edit intention is present in a given revised text. It is framed as a multi-class classification problem where the model selects the appropriate edit intention for each revised text.

2. **Edit Summarization**: This task involves summarizing multiple revised texts to explain what changes occurred. This is done based on the edit intentions.

These tasks are significant research topics in the field of Natural Language Processing (NLP) and can be applied in various applications. For example, edit intention identification techniques can be used in writing assistance tools or document edit summarization systems.

<br/>
# 요약

이 논문에서는 텍스트 수정의 이해를 돕기 위해 수정 의도, 방법 및 응용에 대한 설문조사를 수행하였다. 연구진은 다양한 데이터셋과 수정 의도 분류 체계를 정리하고, 수정 의도 식별 방법을 검토하여 쓰기 지원 및 문서 수정 요약과 같은 하위 응용 프로그램을 강조하였다. 마지막으로, 향후 연구 방향으로 수정 의도의 진화와 일반화 가능성에 대한 논의가 포함되었다.

---

This paper conducts a survey on edit intentions, methods, and applications to enhance the understanding of text revision. The authors organize various datasets and edit intention taxonomies, reviewing identification methods and highlighting downstream applications such as writing assistance and document edit summarization. Finally, the discussion includes future research directions focusing on the evolution and generalizability of edit intentions.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: Edit intention taxonomy를 보여주는 다이어그램으로, 연구의 다양한 측면(데이터셋, 방법론, 응용 분야 등)을 통합하여 시각적으로 정리하였다. 이는 연구자들이 edit intention 관련 연구를 이해하고 접근하는 데 도움을 준다.
   - **Figure 2**: Revision dataset construction의 워크플로우를 설명하는 다이어그램으로, 세분화, 버전 정렬, 차이 분석의 단계가 포함되어 있다. 이 과정은 데이터셋 구축의 복잡성을 강조하며, 각 단계에서의 설계 선택이 후속 분석에 미치는 영향을 보여준다.
   - **Figure 3**: Edit intention taxonomy의 진화 과정을 나타내는 다이어그램으로, 다양한 EIT 간의 인용 네트워크를 통해 EIT의 발전을 시각적으로 표현하였다. 이는 연구자들이 기존의 EIT를 기반으로 새로운 EIT를 개발하는 데 유용하다.
   - **Figure 5**: Revision dataset construction의 구체적인 예시를 보여주는 다이어그램으로, 두 개의 문단을 세분화하고 버전 정렬 및 차이 분석을 통해 수정된 내용을 시각적으로 나타낸다. 이는 실제 데이터셋 구축 과정에서의 단계별 접근 방식을 이해하는 데 도움을 준다.

2. **테이블**
   - **Table 1**: 다양한 텍스트 수정 데이터셋의 통계를 정리한 표로, 각 데이터셋의 출처, 편집 의도 및 행동 레이블의 유무, 추가 정보 등을 포함하고 있다. 이는 연구자들이 적절한 데이터셋을 선택하는 데 유용한 정보를 제공한다.
   - **Table 2**: 텍스트 수정 데이터셋의 상세 목록으로, 각 데이터셋의 특성과 출처를 정리하였다. 이는 연구자들이 데이터셋의 특성을 비교하고 선택하는 데 도움을 준다.
   - **Table 3**: Edit intention taxonomy의 특성을 요약한 표로, 각 연구의 적용 도메인, 카테고리 정의의 유무, 예시 제공 여부 등을 정리하였다. 이는 EIT의 비교 및 분석에 유용하다.
   - **Table 4**: 연구에서 제공된 코드 및 데이터의 유무를 요약한 표로, 연구자들이 재현 가능성을 평가하는 데 도움을 준다.

3. **어펜딕스**
   - 어펜딕스에서는 연구의 세부 사항, 추가 데이터셋, EIT의 특성, 평가 메트릭스 등을 제공하여 연구의 깊이를 더하고, 연구자들이 추가적인 정보를 쉽게 찾을 수 있도록 돕는다.

---

### Insights and Results from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: This diagram illustrates the edit intention taxonomy, providing a visual synthesis of various aspects of the research (datasets, methodologies, applications, etc.). It aids researchers in understanding and approaching edit intention-related studies.
   - **Figure 2**: This figure outlines the workflow for revision dataset construction, including segmentation, version alignment, and differencing. It emphasizes the complexity of dataset construction and shows how design choices at each stage impact subsequent analyses.
   - **Figure 3**: This diagram represents the evolution of edit intention taxonomies (EITs) through a citation network, visually depicting the development of various EITs. It is useful for researchers looking to build new EITs based on existing ones.
   - **Figure 5**: This figure provides a concrete example of the revision dataset construction process, illustrating how two paragraphs are segmented, aligned, and analyzed for differences. It helps in understanding the step-by-step approach in actual dataset construction.

2. **Tables**
   - **Table 1**: This table summarizes statistics for various text revision datasets, including their sources, the presence of edit intention and action labels, and additional features. It provides useful information for researchers selecting appropriate datasets.
   - **Table 2**: This table lists detailed characteristics of text revision datasets, helping researchers compare and choose datasets based on their specific needs.
   - **Table 3**: This table summarizes the characteristics of edit intention taxonomies, including application domains, the presence of definitions, and examples. It aids in the comparison and analysis of EITs.
   - **Table 4**: This table summarizes whether the literature provided code and data, helping researchers assess reproducibility and the availability of resources.

3. **Appendices**
   - The appendices provide detailed information on the study's specifics, additional datasets, characteristics of EITs, evaluation metrics, and more, enhancing the depth of the research and facilitating easy access to supplementary information for researchers.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Lan2026,
  author    = {Fangping Lan and Qi Zhang and Eduard C. Dragut},
  title     = {Making Revisions Understandable: A Survey of Edit Intentions, Methods, and Applications},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {35003--35019},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Fangping Lan, Qi Zhang, and Eduard C. Dragut. "Making Revisions Understandable: A Survey of Edit Intentions, Methods, and Applications." In *Findings of the Association for Computational Linguistics: ACL 2026*, 35003–35019. Association for Computational Linguistics, July 2026.  
