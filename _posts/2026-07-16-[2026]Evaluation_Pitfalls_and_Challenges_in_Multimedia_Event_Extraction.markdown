---
layout: post
title:  "[2026]Evaluation Pitfalls and Challenges in Multimedia Event Extraction"
date:   2026-07-16 21:28:02 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 멀티미디어 이벤트 추출(MEE)의 평가에서 발생하는 문제점들을 체계적으로 분석하고, 이를 해결하기 위한 엄격한 평가 프레임워크인 STRICT EVAL을 제안한다.


짧은 요약(Abstract) :


이 논문의 초록에서는 멀티미디어 이벤트 추출(Multimedia Event Extraction, MEE)의 목표가 여러 모달리티(예: 텍스트와 이미지)를 통해 이벤트와 그 인자를 공동으로 식별하여 보다 포괄적인 이벤트 이해를 지원하는 것이라고 설명하고 있습니다. 최근 연구들은 지속적이고 상당한 진전을 보고하고 있지만, 이러한 결과의 신뢰성과 비교 가능성은 일관되고 철저한 평가에 크게 의존합니다. 본 연구에서는 MEE의 평가에서 발생할 수 있는 함정과 도전 과제를 체계적으로 분석하고, 세 가지 주요 문제의 원인을 식별합니다: 일관되지 않은 데이터 처리, 일관되지 않은 작업 가정, 그리고 지나치게 느슨한 평가 설정입니다. 우리는 엄격한 평가 프레임워크 하에서 수행한 일련의 통제된 실험을 통해, 사소한 평가 선택이 성능 변동을 초래하고 모델의 실제 이벤트 기반 능력을 과대평가할 수 있음을 보여줍니다. 이러한 발견은 비교 가능한 평가 기준의 필요성을 강조하며, MEE에서 보다 철저한 평가로의 전환을 촉구합니다.



The abstract of this paper describes that the goal of Multimedia Event Extraction (MEE) is to jointly identify events and their arguments across multiple modalities, such as text and images, to support more comprehensive event understanding. While recent work reports steady and substantial progress, the reliability and comparability of these results critically depend on consistent and rigorous evaluation. In this work, we present the first systematic analysis of evaluation pitfalls in MEE and identify three major sources of issues: inconsistent data processing, inconsistent task assumptions, and overly relaxed evaluation settings. We demonstrate, through a series of controlled experiments under a strict evaluation framework, that minor evaluation choices can cause large performance variations and lead to overestimation of a model’s ability to ground real-world events across modalities. Our findings highlight the need for comparable evaluation standards and encourage a shift toward more rigorous evaluation in multimedia event extraction.


* Useful sentences :




{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리  

* Pitfalls: 함정  


<br/>
# Methodology



이 논문에서는 멀티미디어 이벤트 추출(Multimedia Event Extraction, MEE)을 위한 새로운 평가 프레임워크인 STRICT EVAL을 제안합니다. 이 프레임워크는 기존의 평가 방법에서 발생할 수 있는 여러 가지 문제점을 해결하기 위해 설계되었습니다. MEE는 텍스트와 이미지와 같은 여러 모달리티에서 이벤트와 그 인자를 동시에 추출하는 작업으로, 최근 연구에서 주목받고 있습니다. 그러나 기존의 평가 방법은 일관성이 부족하고, 평가 기준이 느슨하여 모델의 실제 성능을 과대평가할 수 있습니다.

#### 모델 및 아키텍처
이 연구에서는 SINGLE TASK 모델을 사용하여 각 서브태스크(텍스트 및 비주얼 이벤트 탐지, 이벤트 인자 추출)를 독립적으로 처리합니다. 텍스트 이벤트 탐지(ED)와 인자 추출(EAE)는 BERT 모델을 기반으로 하며, 입력 문장을 토큰 레벨로 분류하여 이벤트 유형을 예측합니다. 비주얼 이벤트 탐지와 인자 추출은 CLIP 모델을 사용하여 이미지에서 객체를 탐지하고, 각 객체에 대해 역할을 분류합니다.

#### 트레이닝 데이터
MEE 모델은 M2E2 벤치마크를 사용하여 훈련됩니다. 이 벤치마크는 6,167개의 문장과 1,014개의 이미지로 구성되어 있으며, 8개의 이벤트 유형과 15개의 인자 역할을 포함합니다. 모델은 ACE와 SWiG와 같은 외부 데이터셋을 사용하여 훈련되며, 이 데이터셋은 사전 정의된 훈련, 개발 및 테스트 분할을 제공합니다.

#### 특별한 기법
STRICT EVAL 프레임워크는 세 가지 주요 요소를 포함합니다:
1. **일관된 데이터 사용**: 오라클 후처리나 테스트 데이터 노출 없이 평가를 수행합니다.
2. **명확한 작업 정의**: 전체 벤치마크에서 평가를 수행하며, 특정 이벤트가 포함된 샘플로 제한하지 않습니다.
3. **엄격한 매칭 기준**: 텍스트, 비주얼 및 멀티미디어 예측 간의 구조적 제약을 유지하는 엄격한 매칭 기준을 적용합니다.

이러한 접근 방식은 모델의 성능을 보다 정확하게 반영하고, 실제 환경에서의 성능을 더 잘 나타내도록 설계되었습니다.

---




This paper proposes a new evaluation framework called STRICT EVAL for Multimedia Event Extraction (MEE). This framework is designed to address various issues that can arise from existing evaluation methods. MEE involves the simultaneous extraction of events and their arguments across multiple modalities, such as text and images, and has gained attention in recent research. However, existing evaluation methods often lack consistency and have relaxed criteria, which can lead to an overestimation of a model's actual performance.

#### Model and Architecture
The study employs SINGLE TASK models to independently handle each subtask (textual and visual event detection, event argument extraction). Textual event detection (ED) and argument extraction (EAE) are based on the BERT model, which classifies input sentences at the token level to predict event types. Visual event detection and argument extraction utilize the CLIP model to detect objects in images and classify roles for each object.

#### Training Data
The MEE models are trained using the M2E2 benchmark, which consists of 6,167 sentences and 1,014 images, covering 8 event types and 15 argument roles. The models leverage external datasets such as ACE and SWiG for training, which provide predefined training, development, and test splits.

#### Special Techniques
The STRICT EVAL framework incorporates three main elements:
1. **Consistent Data Usage**: Evaluations are conducted without oracle post-processing or exposure to test data.
2. **Clear Task Definition**: Evaluations are performed on the full benchmark without restricting to samples containing specific events.
3. **Strict Matching Criteria**: Strict matching criteria are applied to maintain structural constraints across textual, visual, and multimedia predictions.

This approach is designed to more accurately reflect model performance and better represent real-world performance challenges.


<br/>
# Results



이 논문에서는 멀티미디어 이벤트 추출(MEE) 분야에서의 평가 문제와 도전 과제를 체계적으로 분석하고, 이를 해결하기 위한 새로운 평가 프레임워크인 STRICT EVAL을 제안합니다. 연구자들은 MEE의 평가에서 발생하는 주요 문제를 세 가지 범주로 나누어 설명합니다: 

1. **일관성 없는 데이터 처리**: MEE 연구에서 사용되는 데이터 세트는 종종 외부 데이터 세트에 의존하며, 이로 인해 훈련 세트 구성, 전처리 및 후처리 절차에서 상당한 차이가 발생합니다. 예를 들어, M2E2 벤치마크는 ACE 및 SWiG와 같은 외부 데이터 세트를 사용하여 훈련하지만, 후속 연구에서는 추가적인 훈련 데이터를 포함하여 모델을 최적화하는 경우가 많습니다. 이로 인해 모델의 성능이 다르게 평가될 수 있습니다.

2. **일관성 없는 작업 가정**: 연구자들은 일부 연구가 텍스트와 이미지에서만 이벤트를 평가하는 반면, 다른 연구는 멀티미디어 이벤트에만 초점을 맞추는 등 작업 가정에서의 불일치를 지적합니다. 이러한 차이는 보고된 결과의 비교 가능성을 저해합니다.

3. **과도하게 느슨한 평가 설정**: MEE의 평가 메트릭은 종종 느슨한 매칭 기준이나 구조적 제약이 결여되어 있어 정확성이 떨어집니다. 예를 들어, 텍스트 이벤트의 경우, 여러 이벤트가 동일한 문장에 존재할 때 트리거 오프셋을 무시하는 경우가 많아 성능이 부풀려질 수 있습니다.

이 연구는 STRICT EVAL 프레임워크를 통해 이러한 문제를 해결하고, 평가의 일관성을 높이며, 실제 성능을 더 잘 반영할 수 있도록 합니다. 연구자들은 다양한 실험을 통해 평가 설정의 변화가 성능에 미치는 영향을 정량적으로 분석하였고, 이로 인해 MEE의 성능이 과대 평가될 수 있음을 보여주었습니다.

결과적으로, STRICT EVAL을 적용한 후, 기존 모델의 성능이 크게 하락하는 것을 관찰하였으며, 이는 평가 설정의 미세한 변화가 성능에 미치는 영향을 강조합니다. 이 연구는 MEE 분야에서의 평가 투명성을 높이고, 향후 연구에서의 일관된 평가 프로토콜의 필요성을 강조합니다.




This paper systematically analyzes the evaluation pitfalls and challenges in the field of Multimedia Event Extraction (MEE) and proposes a new evaluation framework called STRICT EVAL to address these issues. The authors categorize the main problems encountered in MEE evaluation into three major areas:

1. **Inconsistent Data Processing**: MEE research often relies on external datasets, leading to significant variations in training set construction, preprocessing, and postprocessing procedures. For instance, while the M2E2 benchmark uses external datasets like ACE and SWiG for training, subsequent studies may incorporate additional training data, resulting in different performance evaluations for the models.

2. **Inconsistent Task Assumptions**: The authors point out discrepancies in task assumptions, where some studies evaluate events only present in texts and images, while others focus exclusively on multimedia events. Such differences hinder the comparability of reported results.

3. **Overly Relaxed Evaluation Settings**: The evaluation metrics for MEE are often imprecise due to relaxed matching criteria or missing structural constraints. For example, in textual event extraction, ignoring trigger offsets when multiple events of the same type appear in a sentence can inflate performance metrics.

The study introduces the STRICT EVAL framework to address these issues, aiming to enhance evaluation consistency and better reflect real-world performance. The researchers conducted various experiments to quantitatively analyze how changes in evaluation settings affect performance, demonstrating that MEE performance can be overestimated.

As a result, after applying STRICT EVAL, a significant drop in performance for existing models was observed, highlighting the impact of minor changes in evaluation settings on performance. This research emphasizes the need for increased transparency in MEE evaluation and calls for consistent evaluation protocols in future research.


<br/>
# 예제



이 논문에서는 멀티미디어 이벤트 추출(Multimedia Event Extraction, MEE)의 평가에서 발생할 수 있는 여러 가지 문제점과 도전 과제를 다루고 있습니다. 특히, MEE의 평가가 일관되지 않거나 비효율적일 수 있는 세 가지 주요 원인을 제시합니다: 데이터 처리의 불일치, 작업 가정의 불일치, 그리고 지나치게 느슨한 평가 설정입니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 텍스트와 이미지 쌍으로 구성된 데이터셋. 예를 들어, 뉴스 기사에서 발췌한 문장과 해당 문장과 관련된 이미지.
   - **출력**: 각 이벤트의 유형과 그에 대한 인수(예: 주체, 객체 등)를 포함하는 주석. 예를 들어, "이민자들이 리비아-니제르 국경을 넘었다"라는 문장에서 "이민자"는 주체, "국경"은 장소로 주석이 달릴 수 있습니다.

2. **테스트 데이터**:
   - **입력**: 새로운 뉴스 기사와 그에 대한 이미지. 예를 들어, "EU는 니제르에 다섯 개의 이민자 센터를 두었다"라는 문장과 관련된 이미지.
   - **출력**: 모델이 예측한 이벤트와 그에 대한 인수. 예를 들어, 모델이 "이민자 센터"라는 이벤트를 감지하고, "EU"를 주체로, "니제르"를 장소로 예측할 수 있습니다.

#### 구체적인 테스크
- **이벤트 감지(Event Detection)**: 주어진 텍스트에서 이벤트를 식별하고 해당 이벤트의 유형을 분류합니다. 예를 들어, "이민자들이 국경을 넘었다"라는 문장에서 "국경을 넘다"라는 이벤트를 감지합니다.
- **이벤트 인수 추출(Event Argument Extraction)**: 감지된 이벤트에 대해 관련된 인수를 식별하고 그 역할을 할당합니다. 예를 들어, "이민자"는 주체 역할, "국경"은 장소 역할로 할당됩니다.

이러한 작업은 멀티모달 데이터(텍스트와 이미지)를 통합하여 수행되며, 각 단계에서 발생할 수 있는 평가의 불일치와 문제점을 분석합니다. 이 논문은 이러한 문제를 해결하기 위한 엄격한 평가 프레임워크인 STRICT EVAL을 제안합니다.

---




This paper addresses various pitfalls and challenges in the evaluation of Multimedia Event Extraction (MEE). It specifically identifies three major sources of issues that can lead to inconsistent or ineffective evaluations: inconsistent data processing, inconsistent task assumptions, and overly relaxed evaluation settings.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Input**: A dataset consisting of text and image pairs. For example, sentences extracted from news articles along with images related to those sentences.
   - **Output**: Annotations that include the type of each event and its arguments (e.g., subject, object). For instance, in the sentence "Migrants crossed the Libya-Niger border," "migrants" could be annotated as the subject, and "border" as the location.

2. **Test Data**:
   - **Input**: New news articles and their corresponding images. For example, the sentence "The EU has five migrant centers located in Niger" along with an image related to this sentence.
   - **Output**: The events predicted by the model and their associated arguments. For example, the model might detect the event "migrant center" and predict "EU" as the subject and "Niger" as the location.

#### Specific Tasks
- **Event Detection**: Identifying events in the given text and classifying the type of those events. For example, detecting the event "crossing the border" in the sentence "Migrants crossed the border."
- **Event Argument Extraction**: Identifying the arguments related to the detected events and assigning roles to them. For example, assigning "migrants" as the subject role and "border" as the location role.

These tasks are performed by integrating multimodal data (text and images), and the paper analyzes the discrepancies and issues that can arise at each stage of evaluation. It proposes a rigorous evaluation framework called STRICT EVAL to address these issues.

<br/>
# 요약
이 논문에서는 멀티미디어 이벤트 추출(MEE)의 평가에서 발생하는 문제점들을 체계적으로 분석하고, 이를 해결하기 위한 엄격한 평가 프레임워크인 STRICT EVAL을 제안한다. 연구 결과, 데이터 처리의 불일치, 작업 가정의 불일치, 그리고 느슨한 평가 설정이 성능 보고에 큰 영향을 미친다는 것을 보여준다. 이를 통해 MEE 연구의 신뢰성과 비교 가능성을 높이기 위한 보다 엄격한 평가 기준의 필요성을 강조한다.

---

This paper systematically analyzes the pitfalls in the evaluation of multimedia event extraction (MEE) and proposes a rigorous evaluation framework called STRICT EVAL to address these issues. The findings demonstrate that inconsistencies in data processing, task assumptions, and relaxed evaluation settings significantly impact reported performance. This highlights the need for more stringent evaluation standards to enhance the reliability and comparability of MEE research.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: M2E2 멀티미디어 이벤트 추출 파이프라인을 보여줍니다. 이 다이어그램은 텍스트와 이미지에서 이벤트를 추출하는 과정을 시각적으로 설명하며, 각 단계에서 발생할 수 있는 평가의 함정을 강조합니다.
   - **Figure 2**: 평가의 느슨한 설정을 설명하는 예시로, 잘못된 예측이 어떻게 올바른 것으로 간주될 수 있는지를 보여줍니다. 이로 인해 평가 점수가 부풀려질 수 있음을 강조합니다.
   - **Figure 3**: 다양한 코어퍼런스 해상도 기술을 사용한 멀티미디어 이벤트 추출의 성능을 비교합니다. 이 결과는 코어퍼런스 해상도 방법에 따라 성능이 어떻게 달라지는지를 보여줍니다.

2. **테이블**
   - **Table 1**: 최근 연구에서 사용된 다양한 평가 설정을 요약합니다. 각 설정이 어떤 평가 함정을 포함하고 있는지를 나타내며, STRICT EVAL이 모든 함정을 제거하는 가장 엄격한 설정임을 강조합니다.
   - **Table 2**: 단일 작업 모델의 unimodal 평가 결과를 보여줍니다. 각 설정에서의 성능 차이를 통해 데이터 처리 선택이 성능에 미치는 영향을 분석합니다.
   - **Table 4**: 원래 설정과 STRICT EVAL 설정에서의 재평가 결과를 비교합니다. 이 표는 평가 설정의 차이가 성능에 미치는 영향을 명확히 보여줍니다.

3. **어펜딕스**
   - **A.1**: 체계적 분석에 포함된 연구 목록을 제공합니다. 이 목록은 다양한 모델링 패러다임을 포함하며, 각 연구의 보고된 평가 점수를 나열합니다.
   - **A.6**: STRICT EVAL의 분석을 통해 평가 프로토콜의 엄격함이 성능 점수에 미치는 영향을 설명합니다. 이 섹션은 평가의 엄격함이 잘못된 긍정 예측을 드러내는 방법을 설명합니다.

### Insights and Results from Other Sections (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the M2E2 multimedia event extraction pipeline, visually explaining the process of extracting events from text and images while highlighting potential evaluation pitfalls at each stage.
   - **Figure 2**: Provides an example of relaxed evaluation settings, showing how incorrect predictions can be counted as correct, which can inflate evaluation scores.
   - **Figure 3**: Compares the performance of multimedia event extraction using different coreference resolution techniques, demonstrating how performance varies based on the method used.

2. **Tables**
   - **Table 1**: Summarizes various evaluation setups used in recent studies, indicating which evaluation pitfalls each setup includes, emphasizing that STRICT EVAL is the strictest setup that eliminates all identified pitfalls.
   - **Table 2**: Displays unimodal evaluation results of single-task models, analyzing how different data processing choices impact performance.
   - **Table 4**: Compares reevaluation results under original and STRICT EVAL settings, clearly showing how differences in evaluation setups affect reported performance.

3. **Appendices**
   - **A.1**: Provides a comprehensive list of studies included in the systematic analysis, covering diverse modeling paradigms and reporting their evaluation scores.
   - **A.6**: Analyzes the impact of stricter evaluation protocols on performance scores, explaining how stricter settings reveal false positives that would otherwise be ignored.

이러한 결과와 인사이트는 멀티미디어 이벤트 추출의 평가에서 발생할 수 있는 다양한 문제를 이해하고, 더 나은 평가 프로토콜을 개발하는 데 중요한 기초 자료를 제공합니다.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Seeberger2026,
  author    = {Philipp Seeberger and Steffen Freisinger and Tobias Bocklet and Korbinian Riedhammer},
  title     = {Evaluation Pitfalls and Challenges in Multimedia Event Extraction},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {30885--30900},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},
}
```

### 시카고 스타일

Philipp Seeberger, Steffen Freisinger, Tobias Bocklet, and Korbinian Riedhammer. "Evaluation Pitfalls and Challenges in Multimedia Event Extraction." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 30885–30900. July 2026. Association for Computational Linguistics.
