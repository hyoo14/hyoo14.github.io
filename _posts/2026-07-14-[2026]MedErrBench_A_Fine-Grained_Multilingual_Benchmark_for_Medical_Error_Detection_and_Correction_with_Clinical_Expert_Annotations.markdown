---
layout: post
title:  "[2026]MedErrBench: A Fine-Grained Multilingual Benchmark for Medical Error Detection and Correction with Clinical Expert Annotations"
date:   2026-07-14 00:48:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 의료 오류 탐지 및 수정에 대한 첫 번째 다국어 벤치마크인 MedErrBench  


짧은 요약(Abstract) :

이 논문에서는 의료 오류 탐지 및 수정에 대한 첫 번째 다국어 벤치마크인 MedErrBench를 소개합니다. 기존의 임상 텍스트에서 발생할 수 있는 오류는 잘못된 진단이나 치료 제안으로 이어질 수 있으며, 이는 심각한 결과를 초래할 수 있습니다. 대형 언어 모델(LLM)이 다양한 의료 응용 프로그램에서 점점 더 많이 사용됨에 따라, 이러한 모델의 성능을 평가하기 위한 전용 벤치마크가 필요합니다. 그러나 현재 이러한 데이터셋은 부족하며, 특히 다양한 언어와 맥락에서 더욱 그렇습니다. MedErrBench는 경험이 풍부한 임상의의 지침 아래 개발되었으며, 영어, 아랍어, 중국어를 포함한 세 가지 언어로 오류 탐지, 위치 지정 및 수정 작업을 지원합니다. 이 데이터셋은 10가지 일반적인 오류 유형에 대한 확장된 분류 체계를 기반으로 하며, 임상 전문가에 의해 주석이 달린 자연스러운 의료 사례를 포함하고 있습니다. 연구 결과는 비영어 환경에서 특히 성능 격차가 두드러지며, 임상적으로 기반이 있는 언어 인식 시스템의 필요성을 강조합니다. MedErrBench와 평가 프로토콜을 공개함으로써, 우리는 다국어 임상 자연어 처리(NLP)를 발전시키고, 전 세계적으로 더 안전하고 공정한 AI 기반 의료를 촉진하고자 합니다.



This paper introduces MedErrBench, the first multilingual benchmark for medical error detection and correction. Inaccuracies in existing or generated clinical text can lead to serious consequences, especially if they result in misdiagnosis or incorrect treatment suggestions. As Large Language Models (LLMs) are increasingly used across diverse healthcare applications, comprehensive evaluation through dedicated benchmarks is crucial. However, such datasets remain scarce, particularly across diverse languages and contexts. MedErrBench is developed under the guidance of experienced clinicians and supports error detection, localization, and correction tasks in three languages: English, Arabic, and Chinese. The dataset is based on an expanded taxonomy of ten common error types and includes natural medical cases annotated by domain experts. Our results reveal notable performance gaps, particularly in non-English settings, highlighting the need for clinically grounded, language-aware systems. By making MedErrBench and our evaluation protocols publicly available, we aim to advance multilingual clinical NLP to promote safer and more equitable AI-based healthcare globally.


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



**메서드(Methodology)**

이 연구에서는 의료 오류 탐지 및 수정에 대한 다국어 벤치마크 데이터셋인 MedErrBench를 구축하기 위해 여러 가지 방법론을 사용했습니다. 이 데이터셋은 영어, 아랍어, 중국어로 구성되어 있으며, 각 언어의 의료 오류를 탐지하고 수정하는 데 필요한 다양한 오류 유형을 포함하고 있습니다.

1. **데이터셋 구성**: 
   - MedErrBench는 세 가지 언어(영어, 아랍어, 중국어)로 구성된 다국어 데이터셋입니다. 각 언어의 데이터는 의료 교육 시스템의 언어적 및 지역적 다양성을 반영합니다. 영어는 글로벌 과학의 공통 언어로, 중국어는 세계에서 가장 많이 사용되는 언어이며, 아랍어는 중동 및 북아프리카 지역을 대표합니다.
   - 데이터는 여러 출처에서 수집되었으며, 각 언어의 데이터는 번역이 아닌 원어로 수집되었습니다. 이를 통해 다국어의 특성을 유지하고, 다양한 오류 유형을 포괄적으로 다룰 수 있도록 하였습니다.

2. **오류 유형 분류**:
   - 연구팀은 경험이 풍부한 임상의와 협력하여 10가지 대표적인 오류 유형을 정의했습니다. 이 오류 유형은 진단, 관리, 치료, 약물 치료, 병원체 식별, 실험실 값 해석, 생리학, 조직학, 해부학, 역학 등으로 구성됩니다. 각 오류 유형에 대한 명확한 정의와 예시를 제공하여 데이터 주석 및 시스템 평가의 기초가 되도록 하였습니다.

3. **오류 주입 및 데이터 구축**:
   - 원본 데이터셋에서 오류를 주입하여 각 질문에 대해 올바른 답변을 유지하면서도 그에 대한 잘못된 대안을 무작위로 선택했습니다. 이를 통해 두 가지 버전의 임상 노트를 생성하였습니다: 하나는 올바른 답변이 자연스럽게 통합된 버전, 다른 하나는 잘못된 답변이 삽입된 버전입니다.

4. **전문가 검토 및 품질 관리**:
   - 데이터셋의 품질을 보장하기 위해 두 단계의 전문가 검토 과정을 도입했습니다. 첫 번째 단계에서는 NLP 연구자들이 초기 주석 및 검증을 수행하였고, 두 번째 단계에서는 두 명의 임상의가 모든 인스턴스를 검토하여 의료적 유효성을 확인했습니다.

5. **모델 평가**:
   - MedErrBench는 오류 탐지, 오류 위치 지정, 오류 수정의 세 가지 주요 작업을 지원합니다. 다양한 일반 목적 LLM, 언어 특화 LLM, 의료 도메인 LLM을 평가하여 각 모델의 성능을 비교하였습니다. 평가 지표로는 정확도, ROUGE, BLEU, BERTScore, BLEURT 등을 사용하였습니다.

이러한 방법론을 통해 MedErrBench는 의료 오류 탐지 및 수정에 대한 연구의 기초를 마련하고, 향후 연구에서 안전하고 공정한 AI 기반 의료 시스템을 발전시키는 데 기여할 것입니다.

---




**Methodology**

In this study, we employed several methodologies to construct MedErrBench, a multilingual benchmark dataset for medical error detection and correction. This dataset is composed of English, Arabic, and Chinese, encompassing various types of medical errors necessary for detecting and correcting errors in each language.

1. **Dataset Construction**:
   - MedErrBench is a multilingual dataset consisting of three languages (English, Arabic, and Chinese). The data reflects linguistic and regional diversity in medical education systems, with English serving as the global scientific lingua franca, Chinese as the most spoken language in the world, and Arabic representing the Middle East and North Africa region.
   - The data was collected from multiple sources, ensuring that each language's dataset was gathered in its native form rather than through translation. This approach maintains the characteristics of each language and allows for comprehensive coverage of various error types.

2. **Error Type Classification**:
   - The research team collaborated with experienced clinicians to define ten representative error types. These error types include Diagnosis, Management, Treatment, Pharmacotherapy, Causal Organism/Pathogen, Lab Value Interpretation, Physiology, Histology, Anatomy, and Epidemiology. Clear definitions and examples for each error type were provided to serve as a foundational schema for data annotation and system evaluation.

3. **Error Injection and Dataset Construction**:
   - Errors were injected into the original datasets by preserving the correct answer while randomly selecting one plausible but incorrect alternative. This process resulted in two versions of a clinical note: one where the correct answer was naturally integrated into the context and another where the incorrect answer was inserted.

4. **Expert Review and Quality Control**:
   - To ensure the quality of the dataset, a two-stage expert review process was implemented. In the first stage, NLP researchers performed initial annotation and verification, while in the second stage, two clinicians reviewed all instances to validate medical plausibility.

5. **Model Evaluation**:
   - MedErrBench supports three key tasks: error detection, localization, and correction. A range of general-purpose LLMs, language-specific LLMs, and medical-domain LLMs were evaluated to compare the performance of each model. Evaluation metrics included Accuracy, ROUGE, BLEU, BERTScore, and BLEURT.

Through these methodologies, MedErrBench lays the groundwork for research in medical error detection and correction, contributing to the advancement of safer and more equitable AI-based healthcare systems in future research.


<br/>
# Results



이 논문에서는 MedErrBench라는 다국어 의료 오류 탐지 및 수정 벤치마크 데이터셋을 소개하고, 다양한 모델의 성능을 평가한 결과를 제시합니다. MedErrBench는 영어, 아랍어, 중국어로 구성되어 있으며, 각 언어에서 의료 오류의 탐지, 위치 파악, 수정 작업을 지원합니다. 이 데이터셋은 임상 전문가의 주도로 개발되었으며, 10가지의 일반적인 오류 유형에 대한 세부 분류가 포함되어 있습니다.

#### 경쟁 모델
논문에서는 여러 모델을 평가하였으며, 크게 세 가지 그룹으로 나누어 성능을 비교했습니다:
1. **일반 목적 LLMs**: GPT-4o, Gemini 2.0 Flash 등
2. **언어 특화 LLMs**: Qwen2.5-7B-Instruct, Deepseek-R1 등
3. **의료 도메인 LLMs**: MedGemma-4B, HuatuoGPT-o1-7B 등

#### 테스트 데이터
각 모델은 영어, 아랍어, 중국어 데이터셋에서 테스트되었으며, 각 데이터셋은 오류가 포함된 임상 노트로 구성되어 있습니다. 영어 데이터셋은 1,024개의 인스턴스를 포함하고, 중국어 데이터셋은 1,000개, 아랍어 데이터셋은 482개로 구성되어 있습니다.

#### 메트릭
모델의 성능은 다음과 같은 메트릭을 사용하여 평가되었습니다:
- **정확도 (Accuracy)**: 오류 탐지 및 위치 파악의 정확성을 측정합니다.
- **ROUGE**: 생성된 텍스트와 참조 텍스트 간의 유사성을 평가합니다.
- **BLEU**: 기계 번역의 품질을 평가하는 데 사용됩니다.
- **BERTScore**: 문장의 의미적 유사성을 평가합니다.
- **BLEURT**: 생성된 텍스트의 유창성과 적합성을 평가합니다.

#### 성능 비교
모델의 성능은 각 언어별로 비교되었으며, Deepseek-V3와 Doubao-1.5-Thinking-Pro 모델이 전반적으로 우수한 성능을 보였습니다. 특히, 의료 도메인 LLMs는 일반 목적 모델에 비해 일관되게 높은 성능을 보이지 않았습니다. 이는 의료 텍스트의 복잡성과 오류 탐지 및 수정 작업의 도전 과제를 반영합니다.

결과적으로, MedErrBench는 의료 오류 탐지 및 수정 연구의 기초를 마련하며, 향후 연구에서 더 나은 다국어 모델 개발을 위한 중요한 자원으로 작용할 것입니다.

---




This paper introduces MedErrBench, a multilingual benchmark dataset for medical error detection and correction, and presents the results of evaluating various models' performance. MedErrBench is composed of English, Arabic, and Chinese, supporting tasks of error detection, localization, and correction in each language. The dataset was developed under the guidance of clinical experts and includes a detailed classification of ten common error types.

#### Competing Models
The paper evaluates several models, categorizing them into three main groups for performance comparison:
1. **General-purpose LLMs**: GPT-4o, Gemini 2.0 Flash, etc.
2. **Language-specific LLMs**: Qwen2.5-7B-Instruct, Deepseek-R1, etc.
3. **Medical-domain LLMs**: MedGemma-4B, HuatuoGPT-o1-7B, etc.

#### Test Data
Each model was tested on datasets in English, Arabic, and Chinese, consisting of clinical notes with embedded errors. The English dataset contains 1,024 instances, the Chinese dataset includes 1,000, and the Arabic dataset consists of 482 instances.

#### Metrics
The models' performance was evaluated using the following metrics:
- **Accuracy**: Measures the correctness of error detection and localization.
- **ROUGE**: Assesses the similarity between generated text and reference text.
- **BLEU**: Used to evaluate the quality of machine translation.
- **BERTScore**: Evaluates the semantic similarity of sentences.
- **BLEURT**: Assesses the fluency and adequacy of generated text.

#### Performance Comparison
The performance of the models was compared across each language, with Deepseek-V3 and Doubao-1.5-Thinking-Pro showing overall superior performance. Notably, medical-domain LLMs did not consistently outperform general-purpose models, reflecting the complexity of medical text and the challenges of error detection and correction tasks.

In conclusion, MedErrBench lays a solid foundation for research in medical error detection and correction, serving as a crucial resource for advancing the development of better multilingual models in future research.


<br/>
# 예제



**MedErrBench 데이터셋 개요**

MedErrBench는 의료 오류 탐지 및 수정에 대한 다국어 벤치마크 데이터셋으로, 영어, 아랍어, 중국어로 구성되어 있습니다. 이 데이터셋은 의료 전문가의 주도로 개발되었으며, 다양한 오류 유형을 포함하고 있습니다. 데이터셋은 오류 탐지, 오류 위치 지정, 오류 수정의 세 가지 주요 작업을 지원합니다.

**1. 데이터셋 구성**

- **트레이닝 데이터**: 각 언어별로 의료 사례가 포함되어 있으며, 각 사례는 오류가 포함된 문장과 오류가 없는 문장으로 나뉩니다. 예를 들어, 영어 데이터셋의 경우, 1,024개의 인스턴스가 있으며, 각 인스턴스는 평균 755.7자의 길이를 가집니다.
  
- **테스트 데이터**: 테스트 데이터는 모델의 성능을 평가하기 위해 사용됩니다. 각 테스트 인스턴스는 오류가 포함된 문장과 그에 대한 정답(수정된 문장)을 포함합니다.

**2. 예시**

- **입력 형식**: 각 문장은 고유한 ID와 함께 제공되며, 오류가 있는 경우 오류 유형과 수정된 문장이 포함됩니다. 예를 들어, 다음과 같은 형식으로 제공됩니다:
  
  ```
  Text ID: 1
  "환자는 심장병으로 진단되었으나, ECG 결과는 위염을 나타냅니다."
  ```

- **출력 형식**: 모델의 출력은 다음과 같은 형식으로 제공됩니다:
  
  ```
  1 1 1 "환자는 심장병으로 진단되었으나, ECG 결과는 위염을 나타냅니다."
  ```

  여기서 첫 번째 숫자는 오류가 있는지 여부(1: 오류 있음, 0: 오류 없음), 두 번째 숫자는 오류가 있는 문장의 ID, 세 번째는 수정된 문장입니다.

**3. 작업 설명**

- **오류 탐지**: 주어진 임상 노트에 오류가 있는지 여부를 판단합니다. 이 작업은 이진 분류 문제로, 오류가 없는 경우 0, 오류가 있는 경우 1로 표시됩니다.

- **오류 위치 지정**: 오류가 포함된 특정 문장을 식별합니다. 이 작업은 문장 수준에서 오류를 찾는 데 중점을 둡니다.

- **오류 수정**: 오류가 포함된 문장을 수정하여 올바른 문장을 생성합니다. 이 작업은 문맥 이해와 의료 지식이 필요합니다.




**Overview of the MedErrBench Dataset**

MedErrBench is a multilingual benchmark dataset for medical error detection and correction, consisting of English, Arabic, and Chinese. This dataset was developed under the guidance of medical experts and includes various types of errors. It supports three main tasks: error detection, error localization, and error correction.

**1. Dataset Composition**

- **Training Data**: Each language contains medical cases, which are divided into sentences with errors and sentences without errors. For example, the English dataset contains 1,024 instances, with an average length of 755.7 characters per instance.

- **Test Data**: The test data is used to evaluate the performance of the models. Each test instance includes a sentence with an error and the corresponding correct answer (the corrected sentence).

**2. Example**

- **Input Format**: Each sentence is provided with a unique ID, and if there is an error, the type of error and the corrected sentence are included. For example, it is provided in the following format:
  
  ```
  Text ID: 1
  "The patient is diagnosed with heart disease, but the ECG results indicate gastritis."
  ```

- **Output Format**: The model's output is provided in the following format:
  
  ```
  1 1 1 "The patient is diagnosed with heart disease, but the ECG results indicate gastritis."
  ```

  Here, the first number indicates whether there is an error (1: error present, 0: no error), the second number is the ID of the erroneous sentence, and the third is the corrected sentence.

**3. Task Description**

- **Error Detection**: Determine whether there is an error in the given clinical note. This task is formulated as a binary classification problem, where 0 indicates no error and 1 indicates an error.

- **Error Localization**: Identify the specific sentence that contains the error. This task focuses on finding errors at the sentence level.

- **Error Correction**: Generate a corrected version of the clinical note that contains the error. This task requires contextual understanding and medical knowledge to produce plausible corrections.

<br/>
# 요약


이 연구에서는 의료 오류 탐지 및 수정에 대한 다국어 벤치마크 데이터셋인 MedErrBench를 제안하였다. 데이터셋은 영어, 아랍어, 중국어로 구성되며, 10가지 오류 유형에 대한 전문가 주도의 주석이 포함되어 있다. 다양한 언어 모델을 평가한 결과, 특히 비영어 환경에서 성능 격차가 두드러지며, 임상적으로 기반한 언어 인식 시스템의 필요성이 강조되었다.



This study introduces MedErrBench, a multilingual benchmark dataset for medical error detection and correction. The dataset comprises English, Arabic, and Chinese, with expert annotations on ten types of errors. Evaluation of various language models reveals significant performance gaps, particularly in non-English settings, highlighting the need for clinically grounded, language-aware systems.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: MedErrBench의 개요를 보여주며, 데이터셋의 구조와 오류 유형을 시각적으로 설명합니다. 이는 연구자들이 데이터셋의 구성 요소를 이해하는 데 도움을 줍니다.
   - **Figure 2**: 각 언어별로 난이도와 추론 유형의 분포를 나타냅니다. 영어 데이터셋은 높은 난이도의 질문이 많고, 아랍어 데이터셋은 사실 회상 기반 질문이 많다는 점에서 언어별 차이를 강조합니다.
   - **Figure 3**: 예제 난이도와 모델 성능 간의 관계를 보여줍니다. 이는 모델이 다양한 난이도의 질문에 어떻게 반응하는지를 분석하는 데 유용합니다.
   - **Figure 4**: 지식 기반과 시나리오 기반 데이터에서의 모델 성능을 비교합니다. 대부분의 모델이 시나리오 기반 작업에서 더 나은 성능을 보였으며, 이는 맥락적 패턴 인식에 의존하고 있음을 시사합니다.

2. **테이블**
   - **Table 1**: 의료 오류 유형의 분류와 정의를 제공합니다. 이는 연구자들이 오류 유형을 이해하고, 데이터셋을 구축하는 데 필요한 기초 정보를 제공합니다.
   - **Table 2-4**: MedErrBench의 각 언어(영어, 중국어, 아랍어)에서의 모델 성능을 비교합니다. 각 모델의 오류 탐지, 위치 지정, 수정 성능을 평가하여, 어떤 모델이 특정 언어에서 더 잘 작동하는지를 보여줍니다.
   - **Table S2**: 데이터셋의 언어별 통계 정보를 제공합니다. 각 언어의 훈련, 검증, 테스트 세트의 샘플 수와 평균 길이를 포함하여, 데이터셋의 구조를 이해하는 데 도움을 줍니다.

3. **어펜딕스**
   - **Appendix A**: 오류 유형의 정의와 예시를 제공합니다. 이는 연구자들이 오류 유형을 명확히 이해하고, 데이터셋을 활용하는 데 필요한 정보를 제공합니다.
   - **Appendix B**: 데이터 통계 및 오류 유형 분포를 보여줍니다. 이는 각 언어에서 어떤 오류 유형이 더 빈번하게 발생하는지를 분석하는 데 유용합니다.
   - **Appendix C**: MedErrBench가 지원하는 주요 작업(오류 탐지, 위치 지정, 수정)에 대한 설명을 제공합니다. 이는 연구자들이 데이터셋을 활용하여 어떤 작업을 수행할 수 있는지를 명확히 이해하는 데 도움을 줍니다.

### Insights from Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Figure 1**: Provides an overview of MedErrBench, visually explaining the structure of the dataset and types of errors. This aids researchers in understanding the components of the dataset.
   - **Figure 2**: Illustrates the distribution of difficulty levels and reasoning types across languages. The English dataset has a higher proportion of difficult questions, while the Arabic dataset emphasizes factual recall, highlighting linguistic differences.
   - **Figure 3**: Shows the relationship between example difficulty and model performance. It analyzes how models respond to questions of varying difficulty.
   - **Figure 4**: Compares model performance on knowledge-based versus scenario-based data. Most models performed better on scenario-based tasks, indicating reliance on contextual pattern recognition.

2. **Tables**
   - **Table 1**: Classifies medical error types and provides definitions. This foundational information is essential for researchers constructing datasets.
   - **Tables 2-4**: Compare model performance across languages (English, Chinese, Arabic) on MedErrBench. They evaluate error detection, localization, and correction, revealing which models perform better in specific languages.
   - **Table S2**: Provides statistics on the dataset, including sample sizes and average lengths for training, validation, and test sets, helping to understand the dataset's structure.

3. **Appendices**
   - **Appendix A**: Offers definitions and examples of error types, clarifying the understanding of errors for researchers using the dataset.
   - **Appendix B**: Displays data statistics and distribution of error types, useful for analyzing which errors are more prevalent in each language.
   - **Appendix C**: Describes the key tasks supported by MedErrBench (error detection, localization, correction), helping researchers understand how to utilize the dataset effectively.

<br/>
# refer format:
다음은 요청하신 논문의 BibTeX 형식과 시카고 스타일 인용입니다.   

### BibTeX 형식
```bibtex
@inproceedings{ma2026mederrbench,
  author    = {Congbo Ma and Yichun Zhang and Yousef Al-Jazzazi and Ahamed Foisal and Laasya Sharma and Yousra Sadqi and Khaled Saleh and Jihad Mallat and Farah E. Shamout},
  title     = {MedErrBench: A Fine-Grained Multilingual Benchmark for Medical Error Detection and Correction with Clinical Expert Annotations},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {11802--11827},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  address   = {July 2-7, 2026}
}
```

### 시카고 스타일 인용
Congbo Ma, Yichun Zhang, Yousef Al-Jazzazi, Ahamed Foisal, Laasya Sharma, Yousra Sadqi, Khaled Saleh, Jihad Mallat, and Farah E. Shamout. "MedErrBench: A Fine-Grained Multilingual Benchmark for Medical Error Detection and Correction with Clinical Expert Annotations." In *Findings of the Association for Computational Linguistics: ACL 2026*, 11802–11827. Association for Computational Linguistics, July 2-7, 2026.
    