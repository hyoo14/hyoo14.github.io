---
layout: post
title:  "[2025]Data Caricatures: On the Representation of African American Language in Pretraining Corpora"
date:   2025-08-16 21:42:43 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 12개의 오픈 소스 사전 훈련 코퍼스에서 아프리카계 미국인 언어(AAL)의 양과 질을 평가하기 위해 정량적 실험, 인간 판단 및 질적 분석을 결합하여 사용


짧은 요약(Abstract) :


이 논문에서는 12개의 주로 영어로 구성된 오픈 소스 사전 훈련 코퍼스에서 아프리카계 미국인 언어(AAL)의 양과 질을 평가합니다. 연구자들은 AAL 텍스트의 출처, 변이 및 자연스러움에 중점을 두고 AAL을 사용하는 커뮤니티를 대표하는 텍스트를 분석합니다. 연구 결과, AAL은 평가된 모든 사전 훈련 코퍼스에서 미국 인구 통계에 비해 과소 대표되며, 문서의 0.007%에서 0.18%에 불과합니다. 또한, C4 코퍼스의 AAL 텍스트 중 25% 이상이 LLM(대형 언어 모델)이 생성하기에 부적절하게 인식될 수 있으며, 해로운 고정관념을 강화할 수 있습니다. 마지막으로, 대부분의 자동 필터는 AAL보다 백인 주류 영어(WME) 텍스트를 보존할 가능성이 더 높다는 것을 발견했습니다.



This paper evaluates the quantity and quality of African American Language (AAL) representation in 12 predominantly English, open-source pretraining corpora. The authors focus on the sources, variation, and naturalness of AAL texts that represent the AAL-speaking community. They find that AAL is underrepresented in all evaluated pretraining corpora compared to US demographics, constituting as few as 0.007% and at most 0.18% of documents. Additionally, more than 25% of AAL texts in the C4 corpus may be perceived as inappropriate for LLMs to generate and could reinforce harmful stereotypes. Finally, they find that most automated filters are more likely to conserve White Mainstream English (WME) texts over AAL.


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



이 논문에서는 아프리카계 미국인 언어(AAL)의 표현을 평가하기 위해 다양한 방법론을 사용했습니다. 연구의 주요 목표는 AAL이 포함된 12개의 오픈 소스 사전 훈련 데이터 세트에서 AAL의 양과 질을 분석하는 것이었습니다. 이를 위해 다음과 같은 방법론을 적용했습니다.

1. **데이터 세트 선택**: 연구팀은 AAL의 표현을 평가하기 위해 12개의 오픈 소스 사전 훈련 데이터 세트를 선택했습니다. 이 데이터 세트는 주로 영어로 작성된 자료로 구성되어 있으며, AAL이 포함된 문서의 비율을 분석하기 위해 사용되었습니다.

2. **AAL 문서 추출**: AAL 문서를 식별하기 위해, 연구팀은 인구 통계적 정렬 분류기를 사용했습니다. 이 분류기는 AAL의 일반적인 언어적 특징을 기반으로 하여 문서의 AAL 확률을 계산했습니다. AAL 확률이 0.3 이상인 문서를 추출하여 분석에 포함시켰습니다.

3. **인간 평가**: AAL 문서의 질을 평가하기 위해, 연구팀은 AAL 사용자가 자발적으로 참여한 세 명의 평가자를 모집했습니다. 이들은 AAL 문서의 인간성, 언어적 일치성, 원어민 작성 여부, 그리고 고정관념이나 부적절함을 평가했습니다. 각 평가자는 4점 리커트 척도를 사용하여 문서를 평가했습니다.

4. **자동화된 특징 추출**: AAL의 형태 통사적 특징을 자동으로 추출하기 위해 CGEdit 모델을 사용했습니다. 이 모델은 AAL의 17가지 형태 통사적 구조를 식별할 수 있도록 훈련되었습니다. 이를 통해 AAL 문서에서 특정 언어적 특징의 분포를 분석했습니다.

5. **필터링 전략 평가**: 연구팀은 AAL 문서에 대한 자동화된 필터링 전략의 효과를 평가했습니다. 16개의 자동화된 필터를 사용하여 AAL 문서와 백인 주류 영어(WME) 문서의 보존 가능성을 비교했습니다. 이 과정에서 필터가 AAL 문서를 얼마나 잘 보존하는지를 분석했습니다.

이러한 방법론을 통해 연구팀은 AAL의 양과 질을 평가하고, AAL이 사전 훈련 데이터에서 어떻게 표현되는지를 분석했습니다. 이 연구는 AAL의 표현이 언어 모델의 성능에 미치는 영향을 이해하는 데 중요한 기초 자료를 제공합니다.

---




In this paper, the authors employed various methodologies to evaluate the representation of African American Language (AAL). The primary goal of the study was to analyze the quantity and quality of AAL representation in 12 predominantly English open-source pretraining corpora. The following methodologies were applied:

1. **Dataset Selection**: The research team selected 12 open-source pretraining datasets to evaluate the representation of AAL. These datasets are primarily composed of English texts and were used to analyze the proportion of documents containing AAL.

2. **AAL Document Extraction**: To identify AAL documents, the team utilized a demographic alignment classifier. This classifier calculates the probability of AAL in documents based on common linguistic features of AAL. Documents with an AAL probability of 0.3 or higher were extracted for analysis.

3. **Human Evaluation**: To assess the quality of AAL documents, the team recruited three evaluators who self-identified as AAL speakers. They evaluated the documents based on dimensions such as human-likeness, linguistic match, native speaker authenticity, and perceptions of stereotypes or appropriateness. Each evaluator used a 4-point Likert scale to rate the documents.

4. **Automated Feature Extraction**: To automatically extract morphosyntactic features of AAL, the CGEdit model was employed. This model was trained to identify 17 morphosyntactic constructions characteristic of AAL. This allowed the analysis of the distribution of specific linguistic features within AAL documents.

5. **Evaluation of Filtering Strategies**: The research team assessed the effectiveness of automated filtering strategies on AAL documents. They compared the preservation likelihood of AAL documents against that of White Mainstream English (WME) documents using 16 automated filters. This analysis aimed to understand how well the filters preserved AAL documents.

Through these methodologies, the research team evaluated the quantity and quality of AAL representation and analyzed how AAL is depicted in pretraining data. This study provides essential foundational insights into understanding the impact of AAL representation on the performance of language models.


<br/>
# Results



이 연구에서는 12개의 오픈 소스 사전 훈련 코퍼스를 분석하여 아프리카계 미국인 언어(AAL)의 양과 질을 평가했습니다. 연구 결과는 다음과 같습니다:

1. **AAL의 양적 분석**:
   - AAL은 평가된 모든 사전 훈련 코퍼스에서 심각하게 저조하게 나타났습니다. AAL 문서는 전체 문서의 0.007%에서 0.18%에 불과했습니다. 예를 들어, RedPajama-v2 코퍼스에서는 AAL 문서가 0.18%를 차지했지만, FineWeb-Edu 코퍼스에서는 0.0009%로 가장 낮은 비율을 보였습니다.
   - AAL 사용자는 미국 내 아프리카계 미국인 중 약 80%에 해당하는데, 이는 전체 미국 인구의 약 10%에 해당합니다. 따라서 AAL의 저조한 비율은 AAL 사용자의 언어적 요구를 충족하지 못하는 문제를 나타냅니다.

2. **AAL의 질적 분석**:
   - AAL 문서의 질에 대한 평가에서는 많은 문서가 자연스러운 AAL을 반영하지 않는 것으로 나타났습니다. 예를 들어, AAL 문서 중 51%는 원주율 AAL 화자가 작성하지 않았다고 판단되었습니다. 이는 비원주율 AAL 화자나 기업 소셜 미디어 계정에서 작성된 문서가 포함되어 있음을 시사합니다.
   - AAL 문서의 25% 이상이 LLM(대형 언어 모델)이 생성하기에 부적절하다고 평가되었으며, 이는 AAL 사용자를 고정관념적으로 묘사하는 내용을 포함하고 있었습니다.

3. **자동화된 필터의 영향**:
   - 연구에서는 16개의 자동화된 필터를 평가하여 AAL 문서와 백인 주류 영어(WME) 문서의 보존 가능성을 비교했습니다. 대부분의 필터(13개 중 16개)는 AAL 문서를 WME 문서보다 더 많이 제거하는 경향을 보였습니다.
   - 특히, 대화체 AAL 문서가 힙합 가사나 소셜 미디어 문서보다 더 잘 보존되는 경향이 있었습니다. 이는 자연스러운 대화체가 AAL의 더 나은 표현을 가능하게 한다는 것을 나타냅니다.

4. **결론**:
   - AAL의 저조한 양과 질은 LLM이 AAL을 이해하는 데 어려움을 겪는 원인으로 작용할 수 있으며, 이는 AAL 사용자의 기술 접근성을 제한할 수 있습니다. 따라서 AAL의 적절한 표현을 보장하기 위해 사전 훈련 데이터의 출처와 필터링 과정을 신중하게 고려해야 합니다.




This study analyzed 12 open-source pretraining corpora to evaluate the quantity and quality of African American Language (AAL) representation. The findings are as follows:

1. **Quantitative Analysis of AAL**:
   - AAL was found to be severely underrepresented across all evaluated pretraining corpora, constituting only 0.007% to 0.18% of the total documents. For instance, the RedPajama-v2 corpus contained 0.18% AAL documents, while the FineWeb-Edu corpus had the lowest representation at 0.0009%.
   - Approximately 80% of African Americans are AAL speakers, which corresponds to about 10% of the total U.S. population. Thus, the low representation of AAL indicates a failure to meet the linguistic needs of AAL speakers.

2. **Qualitative Analysis of AAL**:
   - The evaluation of the quality of AAL documents revealed that many documents did not reflect naturalistic AAL. For example, 51% of AAL documents were judged to be unlikely written by a native AAL speaker, suggesting the inclusion of texts authored by non-native speakers or corporate social media accounts.
   - Over 25% of AAL documents were deemed inappropriate for LLMs to generate, often containing content that reinforces harmful stereotypes about AAL speakers.

3. **Impact of Automated Filters**:
   - The study evaluated 16 automated filters to compare the preservation likelihood of AAL documents versus White Mainstream English (WME) documents. Most filters (13 out of 16) were more likely to remove AAL texts than WME texts.
   - Notably, dialogue-based AAL documents were more likely to be preserved compared to hip-hop lyrics or social media texts, indicating that naturalistic dialogue allows for better representation of AAL.

4. **Conclusion**:
   - The low quantity and quality of AAL representation may contribute to LLMs' difficulties in understanding AAL, which can restrict access to technology for AAL speakers. Therefore, careful consideration of the sources and filtering processes in pretraining data is necessary to ensure appropriate representation of AAL.


<br/>
# 예제



이 논문에서는 아프리카계 미국인 언어(AAL)의 표현을 평가하기 위해 다양한 데이터 세트를 사용하여 실험을 수행했습니다. 연구의 주요 목표는 AAL이 현대의 오픈 소스 프리트레이닝 데이터에서 얼마나 잘 표현되고 있는지를 분석하는 것이었습니다. 이를 위해 연구자들은 다음과 같은 방법론을 사용했습니다.

1. **데이터 세트 선택**: 연구자들은 12개의 오픈 소스 프리트레이닝 데이터 세트를 선택했습니다. 이 데이터 세트는 주로 영어로 작성된 문서들로 구성되어 있으며, AAL의 표현을 평가하기 위해 사용되었습니다.

2. **AAL 문서 추출**: AAL 문서를 추출하기 위해, 연구자들은 AAL의 특징을 식별할 수 있는 분류기를 사용했습니다. 이 분류기는 AAL의 언어적 특징을 기반으로 문서를 분류했습니다. 예를 들어, "He be running"과 같은 문장은 AAL의 특징을 나타내며, 이러한 문서들이 AAL 문서로 분류되었습니다.

3. **인간 평가**: 연구자들은 AAL 문서의 품질을 평가하기 위해 세 명의 AAL 화자를 모집하여 문서에 대한 평가를 수행했습니다. 이들은 각 문서가 AAL 화자에 의해 작성된 것처럼 보이는지, 그리고 문서가 고유한 AAL의 특징을 반영하는지를 평가했습니다. 평가 기준은 다음과 같았습니다:
   - **인간 유사성(Human-Likeness)**: 문서가 인간에 의해 작성된 것처럼 보이는 정도.
   - **언어적 일치(Linguistic Match)**: 문서가 AAL의 특징을 얼마나 잘 반영하는지.
   - **원어민(Native Speaker)**: 문서가 AAL 화자에 의해 자연스럽게 작성된 것처럼 보이는지.
   - **적절성(Appropriateness)**: 문서가 언어 모델이 생성하기에 적절한지.
   - **고정관념(Stereotype)**: 문서가 AAL이나 그 화자에 대한 고정관념을 강화하는지.

4. **자동화된 필터링**: 연구자들은 16개의 자동화된 필터를 사용하여 AAL 문서의 보존 가능성을 평가했습니다. 이 필터들은 AAL 문서와 백인 주류 영어(WME) 문서 간의 차이를 분석하여, AAL 문서가 얼마나 잘 보존되는지를 평가했습니다.

5. **결과 분석**: 연구 결과, AAL은 모든 평가된 프리트레이닝 데이터 세트에서 심각하게 저평가되었으며, AAL 문서의 상당 부분이 비자연적이거나 비적절하다고 평가되었습니다. 또한, 자동화된 필터는 AAL 문서보다 WME 문서를 더 잘 보존하는 경향이 있었습니다.

이러한 방법론을 통해 연구자들은 AAL의 표현이 현대의 언어 모델에서 어떻게 다루어지고 있는지를 체계적으로 분석하였습니다.

---



In this paper, the authors conducted experiments to evaluate the representation of African American Language (AAL) in various datasets. The main goal of the research was to analyze how well AAL is represented in modern open-source pretraining data. The researchers employed the following methodologies:

1. **Dataset Selection**: The researchers selected 12 open-source pretraining datasets. These datasets are primarily composed of documents written in English and were used to evaluate the representation of AAL.

2. **AAL Document Extraction**: To extract AAL documents, the researchers used a classifier capable of identifying features of AAL. This classifier categorized documents based on linguistic characteristics of AAL. For example, sentences like "He be running" are indicative of AAL features and were classified as AAL documents.

3. **Human Evaluation**: The researchers recruited three AAL speakers to evaluate the quality of the AAL documents. They assessed whether each document appeared to be authored by an AAL speaker and whether it reflected unique characteristics of AAL. The evaluation criteria included:
   - **Human-Likeness**: The degree to which the document appears to be authored by a human.
   - **Linguistic Match**: How well the document reflects features of AAL.
   - **Native Speaker**: Whether the document seems to be naturally written by an AAL speaker.
   - **Appropriateness**: Whether the document is appropriate for language models to generate.
   - **Stereotype**: Whether the document reinforces stereotypes about AAL or its speakers.

4. **Automated Filtering**: The researchers employed 16 automated filters to assess the likelihood of preserving AAL documents. These filters analyzed the differences between AAL documents and White Mainstream English (WME) documents to evaluate how well AAL documents were preserved.

5. **Results Analysis**: The findings indicated that AAL was severely underrepresented in all evaluated pretraining datasets, with a significant portion of AAL documents being deemed unnatural or inappropriate. Additionally, the automated filters tended to preserve WME documents better than AAL documents.

Through these methodologies, the researchers systematically analyzed how AAL is treated in modern language models.

<br/>


이 논문에서는 12개의 오픈 소스 사전 훈련 코퍼스에서 아프리카계 미국인 언어(AAL)의 양과 질을 평가하기 위해 정량적 실험, 인간 판단 및 질적 분석을 결합하여 사용하였다. 결과적으로 AAL은 모든 평가된 코퍼스에서 미국 인구 통계에 비해 심각하게 저조하게 나타났으며, 많은 문서가 자연스러운 AAL을 잘 반영하지 못하고 있었다. 예를 들어, C4 코퍼스의 AAL 문서 중 25% 이상이 LLM이 생성하기에 부적절하다고 인식될 수 있는 내용이 포함되어 있었다.

---

This paper combines quantitative experiments, human judgments, and qualitative analyses to evaluate the quantity and quality of African American Language (AAL) representation in 12 open-source pretraining corpora. The results show that AAL is severely underrepresented across all evaluated corpora compared to US demographics, with many documents failing to authentically reflect naturalistic AAL. For instance, over 25% of AAL texts in the C4 corpus may contain content deemed inappropriate for LLM generation.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: AAL 문서의 양과 질을 평가하는 방법을 보여줍니다. AAL이 모든 평가된 프리트레인 코퍼스에서 저조하게 나타나며, 많은 문서가 자연스러운 AAL을 잘 반영하지 못하고 있음을 강조합니다.
   - **Figure 2**: AAL 문서의 중복 수를 보여줍니다. 여러 코퍼스 간에 AAL 문서가 상당히 중복되어 있음을 나타내며, 이는 AAL의 다양성이 부족함을 시사합니다.
   - **Figure 3**: C4 코퍼스 내에서 AAL의 형태통사적 특징의 빈도를 보여줍니다. 필터링이 AAL 특징의 분포에 미치는 영향을 강조합니다.
   - **Figure 4**: C4 코퍼스에서 힙합 가사와 겹치는 문서의 비율을 나타냅니다. AAL 문서의 상당 부분이 힙합 가사와 겹치며, 이는 AAL의 대표성을 왜곡할 수 있음을 시사합니다.
   - **Figure 5**: AAL 문서의 적절성과 고정관념을 평가한 결과를 보여줍니다. 많은 문서가 부적절하거나 고정관념을 강화하는 것으로 평가되었습니다.

2. **테이블**
   - **Table 1**: 12개의 오픈 소스 프리트레인 데이터셋의 AAL 문서 비율을 보여줍니다. AAL이 모든 코퍼스에서 극도로 저조하게 나타나며, 이는 AAL 사용자의 대표성이 부족함을 나타냅니다.
   - **Table 2**: AAL 문서의 인간 평가 결과를 요약합니다. AAL 특징이 있는 문서의 비율이 낮음을 보여줍니다.
   - **Table 3**: C4 코퍼스에서 발견된 힙합 가사의 예시를 제공합니다. AAL 문서가 힙합 가사와 겹치는 비율이 높음을 보여줍니다.
   - **Table 4**: AAL 문서의 적절성과 고정관념에 대한 평가 결과를 요약합니다. 많은 문서가 부적절하거나 고정관념을 강화하는 것으로 평가되었습니다.

3. **어펜딕스**
   - **Appendix A**: AAL 문서 추출 방법론을 설명합니다. AAL 특징을 가진 문서를 식별하기 위한 분류기를 사용한 방법을 설명합니다.
   - **Appendix D**: 인간 평가자 간의 합의 수준을 보여줍니다. AAL 문서의 평가에서 평가자 간의 합의가 상당히 높음을 나타냅니다.
   - **Appendix G**: AAL 특징의 분포를 보여줍니다. 다양한 코퍼스에서 AAL 특징의 빈도를 비교합니다.

### English Version



   - **Figure 1**: Illustrates the methods used to evaluate the quantity and quality of AAL documents. It emphasizes that AAL is underrepresented in all evaluated pretraining corpora and that many documents do not accurately reflect naturalistic AAL.
   - **Figure 2**: Shows the count of overlapping AAL documents among the analyzed corpora. It indicates that there is substantial overlap in AAL documents across multiple corpora, suggesting a lack of diversity in AAL representation.
   - **Figure 3**: Displays the frequency of morphosyntactic features of AAL within the C4 corpus. It highlights the impact of filtering on the distribution of AAL features.
   - **Figure 4**: Represents the percentage of documents in the C4 corpus that overlap with hip hop lyrics. A significant portion of AAL documents overlaps with hip hop lyrics, which may distort the representation of AAL.
   - **Figure 5**: Shows the results of evaluations on the appropriateness and stereotypes of AAL documents. Many documents were rated as inappropriate or reinforcing stereotypes.

2. **Tables**
   - **Table 1**: Displays the proportion of AAL documents across 12 open-source pretraining datasets. It indicates that AAL is extremely underrepresented in all corpora, highlighting a lack of representation for AAL speakers.
   - **Table 2**: Summarizes the results of human evaluations of AAL documents. It shows a low percentage of documents identified as containing AAL features.
   - **Table 3**: Provides examples of hip hop lyrics found in the C4 corpus. It illustrates the high overlap of AAL documents with hip hop lyrics.
   - **Table 4**: Summarizes the evaluation results regarding the appropriateness and stereotypes of AAL documents. Many documents were rated as inappropriate or reinforcing stereotypes.

3. **Appendices**
   - **Appendix A**: Describes the methodology for extracting AAL documents. It explains the use of classifiers to identify documents containing AAL features.
   - **Appendix D**: Shows the level of agreement among human evaluators. It indicates that there is substantial agreement among evaluators in assessing AAL documents.
   - **Appendix G**: Displays the distribution of AAL features. It compares the frequency of AAL features across various corpora.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{deas2025data,
  title={Data Caricatures: On the Representation of African American Language in Pretraining Corpora},
  author={Nicholas Deas and Blake Vente and Amith Ananthram and Jessica A. Grieser and Desmond Patton and Shana Kleiner and James Shepard and Kathleen McKeown},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={29192--29217},
  year={2025},
  month={July},
  publisher={Association for Computational Linguistics},

}
```

### 시카고 스타일

Deas, Nicholas, Blake Vente, Amith Ananthram, Jessica A. Grieser, Desmond Patton, Shana Kleiner, James Shepard, and Kathleen McKeown. 2025. "Data Caricatures: On the Representation of African American Language in Pretraining Corpora." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 29192–29217. Bangkok, Thailand: Association for Computational Linguistics.
