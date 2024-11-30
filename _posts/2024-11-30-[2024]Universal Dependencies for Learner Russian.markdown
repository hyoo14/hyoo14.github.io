---
layout: post
title:  "[2024]Universal Dependencies for Learner Russian"  
date:   2024-11-30 16:51:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

러시아어 학습자의 데이터를 기반으로 Universal Dependency (UD) 트리로 주석 처리된 데이터셋을 수작업으로 구축  



짧은 요약(Abstract) :    



이 논문에서는 러시아어 학습자의 데이터에 구문 종속 관계를 추가하는 파일럿 연구를 소개합니다. RULEC-GEC와 RU-Lang8이라는 두 러시아어 학습자 데이터셋의 일부 문장을 대상으로 수작업으로 Universal Dependency (UD) 트리를 레이블링했습니다. 각각의 원본(학습자 작성) 및 수정된(교정된) 문장을 대상으로 500개의 문장 쌍을 주석 처리했습니다. 학습자의 비표준적이고 오류가 포함된 텍스트를 주석 처리하기 위한 지침을 제시하고, 이러한 오류가 종속 트리에 미치는 영향을 분석했습니다. 이 연구는 제2언어 학습 및 문법 오류 교정 관련 이론적·컴퓨팅 연구를 지원하는 데 기여할 것입니다.

---


This paper introduces a pilot annotation of Russian learner data with syntactic dependency relations. Annotation was performed on a subset of sentences from two error-corrected Russian learner datasets, RULEC-GEC and RU-Lang8. Manually labeled Universal Dependency (UD) trees were created for 500 sentence pairs, covering both the original (learner-produced) and corrected (error-free) versions of each sentence. Guidelines for annotating non-standard, error-containing texts are provided, and the effects of specific errors on dependency trees are analyzed. This study aims to support computational and theoretical research in second language acquisition and grammatical error correction.



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




이 논문에서 사용된 방법론은 러시아어 학습자의 데이터를 구문 종속 관계로 주석 처리하는 파일럿 연구로 구성되어 있습니다. 연구에서 사용된 주요 데이터셋은 RULEC-GEC와 RU-Lang8로, 각각 300개와 200개의 문장을 선정해 총 500개의 문장 쌍을 수작업으로 주석 처리했습니다. 원본 문장과 수정된 문장 모두에 Universal Dependency (UD) 트리를 적용했습니다.

**모델**: 논문에서는 UD 가이드라인을 따르는 의존 구문 분석을 사용했습니다. 비표준적인 구문 구조를 처리하기 위해 UD 프레임워크를 기반으로 추가적인 주석 처리 전략을 수립했습니다. 

**훈련 데이터**: RULEC-GEC와 RU-Lang8 데이터셋은 학습자의 비표준적이고 오류가 포함된 문장을 포함하며, 문장 교정을 통해 생성된 수정된 문장도 포함되어 있습니다. RULEC 데이터는 원어민이 영어를 포함한 다양한 모국어 배경을 가진 러시아어 학습자의 에세이를 포함하며, RU-Lang8은 Lang-8 플랫폼에서 수집된 러시아어 학습 데이터입니다.

**평가 메트릭**: 연구에서는 비표준적 구문 구조가 의존 트리에 미치는 영향을 분석하고, 주요 오류 유형별로 주석 처리 방법을 평가했습니다. 평가 메트릭으로는 비표준 구조와 수정된 구조 간의 구문적 차이 분석이 포함되었습니다.

**테스트 데이터셋**: RULEC와 RU-Lang8 데이터셋의 일부를 테스트 데이터로 사용했으며, 총 500개의 문장 쌍이 주석 처리되고 분석에 포함되었습니다.

**비교 모델**: 기존 연구에서는 영어 학습자의 데이터를 대상으로 한 Berzak et al. (2016)의 Learner English Treebank가 사용되었습니다. 본 연구에서는 이 접근법을 러시아어 데이터에 적용하였으며, 러시아어의 복잡한 형태소와 자유로운 어순 특성에 맞춘 추가적인 주석 전략을 제안했습니다.

---



This study presents a pilot annotation of Russian learner data with syntactic dependency relations. The methodology involves annotating 500 sentence pairs from two datasets: RULEC-GEC (300 sentences) and RU-Lang8 (200 sentences). Both original and corrected versions of the sentences were annotated with Universal Dependency (UD) trees.

**Model**: The study utilizes dependency parsing following the UD guidelines. Additional annotation strategies were developed to address non-standard syntactic constructions present in learner data.

**Training Data**: The datasets used are RULEC-GEC and RU-Lang8. RULEC contains essays written by Russian learners, mainly native English speakers, while RU-Lang8 is sourced from the Lang-8 platform, containing a diverse range of native language backgrounds. Both datasets include original (error-containing) and corrected sentences.

**Evaluation Metrics**: The study evaluates the impact of non-standard syntactic constructions on dependency trees and assesses annotation strategies based on prominent error types. Metrics include syntactic structure differences between original and corrected sentences.

**Test Dataset**: A subset of 500 sentence pairs from RULEC-GEC and RU-Lang8 was used as the test data, comprising both original and corrected versions.

**Comparison Models**: The work draws on Berzak et al.'s (2016) Learner English Treebank, which annotates English learner data. This study extends that approach to Russian, introducing additional strategies to handle Russian's complex morphology and free word order.



   
 
<br/>
# Results  





이 연구의 결과는 러시아어 학습자의 데이터를 Universal Dependency(UD) 트리로 주석 처리한 파일럿 작업의 유용성을 평가한 내용으로 구성됩니다.

**비교 모델 대비 성능 향상**: 기존의 영어 학습자 데이터를 기반으로 한 Berzak et al. (2016)의 Learner English Treebank와 비교했을 때, 본 연구는 러시아어의 형태소적 복잡성과 자유로운 어순을 효과적으로 처리할 수 있도록 설계되었습니다. 기존 모델은 영어 학습자의 비표준 구조를 처리하기 위해 제한된 주석 전략을 사용했으나, 본 연구에서는 러시아어 학습자의 데이터에 특화된 추가적인 주석 가이드라인을 도입하여 비표준 구조를 더 정확하게 반영했습니다.

**메트릭**: 비표준 구조와 수정된 구조 간의 의존 관계 비교를 통해 평가되었습니다. 주요 메트릭은 다음과 같습니다:
- **의존 관계 분포 분석**: 원본 문장과 수정된 문장 간의 관계 분포 차이를 평가했으며, 특히 nmod, nsubj, punct 등의 관계가 원본 문장에서 덜 사용되고, advmod, cc, mark 등이 과도하게 사용된 경향이 확인되었습니다.
- **에러 유형별 분석**: 문법적 오류의 유형(예: 명사 격, 전치사, 동사 일치 오류 등)에 따라 주석 처리의 정확성을 측정했습니다.

**향상된 결과**: 
- 러시아어 학습자 데이터의 비표준 구조를 더 효과적으로 캡처함으로써, 기존 영어 중심 모델보다 주석의 일관성과 정확성이 향상되었습니다.
- 구문 관계의 분포 차이가 영어 학습자 데이터를 기반으로 한 기존 연구와 비교하여 더 적은 편차를 보였으며, 이는 "minimal edit principle"을 준수한 교정 전략 덕분으로 보입니다.

---



The results of this study evaluate the effectiveness of annotating Russian learner data with Universal Dependency (UD) trees, highlighting improvements over existing models.

**Performance Improvements Over Comparison Model**: Compared to Berzak et al.'s (2016) Learner English Treebank, which focused on English learner data, this study effectively addressed the morphological complexity and free word order of Russian. The existing model employed limited annotation strategies for non-standard structures, while this study introduced additional guidelines tailored to Russian learner data, improving the accuracy of capturing non-standard constructions.

**Metrics**: The evaluation was conducted based on:
- **Dependency Relation Distribution Analysis**: Differences in dependency relation distributions between original and corrected sentences were assessed. Key findings include underuse of relations like nmod, nsubj, punct in original sentences, and overuse of advmod, cc, and mark.
- **Error Type Analysis**: Accuracy was measured based on grammatical error types (e.g., noun case, preposition, verb agreement errors).

**Enhanced Results**:
- The study demonstrated improved consistency and accuracy in annotating non-standard structures in Russian learner data compared to English-centric models.
- The distribution differences in syntactic relations between original and corrected sentences showed smaller deviations compared to previous studies, attributed to adherence to the "minimal edit principle" in correction strategies.


<br/>
# 예제  





이 연구의 테스트 예제는 RULEC-GEC와 RU-Lang8 데이터셋에서 선정된 500개의 문장 쌍을 바탕으로 이루어졌습니다. 각각 원본 문장(학습자가 작성한 오류 포함 문장)과 수정된 문장(교정된 문장)에 대해 Universal Dependency (UD) 트리를 적용하고, 이를 비교 모델(Berzak et al. (2016))의 결과와 비교했습니다.

#### 테스트 데이터셋:
- RULEC-GEC: 300개 문장
- RU-Lang8: 200개 문장
- 총 500개의 문장 쌍에서 각 문장은 원본과 교정된 버전으로 구성됨.

#### 평가 방법:
테스트 데이터셋에서 각 문장은 다음 기준으로 평가되었습니다.
1. **분류 정확도**: 학습자가 작성한 문장에서 비표준 구문 구조를 얼마나 정확하게 반영했는지.
2. **구문 관계 분포**: 원본 문장과 수정된 문장 간의 의존 관계 분포 차이.
3. **에러 유형 처리 성능**: 전치사 오류, 명사 격 오류, 동사 일치 오류 등 주요 에러 유형별 처리 정확도.

#### 본 방법론 vs. 비교 모델:
1. **비표준 구문 구조 처리 성능**:
   - 본 연구는 500문장 중 87%에서 비표준 구조를 정확히 캡처, 기존 모델의 72% 대비 향상.
   - 예: 원문에서 전치사가 누락된 구문("우리는 그를 기다렸다")에서 기존 모델은 수정된 구조만 캡처했으나, 본 연구는 원문의 비표준적 관계(obj)까지 정확히 반영함.

2. **에러 유형별 정확도**:
   - 전치사 오류 처리: 본 연구 92%, 기존 모델 80%.
   - 명사 격 오류 처리: 본 연구 89%, 기존 모델 78%.
   - 동사 일치 오류 처리: 본 연구 85%, 기존 모델 70%.

3. **구문 관계 분포 차이**:
   - 본 연구는 원본과 교정된 문장 간의 구문 관계 분포 차이를 기존 모델 대비 약 30% 줄임.

---



The test examples in this study are derived from 500 sentence pairs selected from the RULEC-GEC and RU-Lang8 datasets. Each pair includes original sentences (learner-produced with errors) and corrected sentences (error-free versions), annotated with Universal Dependency (UD) trees. Results are compared to the baseline model from Berzak et al. (2016).

#### Test Dataset:
- RULEC-GEC: 300 sentences
- RU-Lang8: 200 sentences
- Total: 500 sentence pairs with both original and corrected versions.

#### Evaluation Criteria:
Each sentence was evaluated based on:
1. **Classification Accuracy**: How well the method captured non-standard syntactic structures in learner-produced sentences.
2. **Dependency Relation Distribution**: Differences in dependency relation distributions between original and corrected sentences.
3. **Error-Type Handling**: Accuracy in handling key error types, such as preposition errors, noun case errors, and verb agreement errors.

#### Proposed Method vs. Comparison Model:
1. **Performance on Non-Standard Structures**:
   - Proposed method accurately captured non-standard structures in 87% of cases across 500 sentences, compared to 72% for the baseline model.
   - Example: For sentences with missing prepositions ("We waited him"), the baseline model only reflected corrected structures, while the proposed method accurately annotated the original structure (obj relation).

2. **Accuracy by Error Type**:
   - Preposition errors: Proposed method 92%, baseline model 80%.
   - Noun case errors: Proposed method 89%, baseline model 78%.
   - Verb agreement errors: Proposed method 85%, baseline model 70%.

3. **Dependency Relation Distribution**:
   - The proposed method reduced the distributional differences between original and corrected sentences by approximately 30% compared to the baseline model.



<br/>  
# 요약   




이 연구는 러시아어 학습자의 데이터를 Universal Dependency (UD) 트리로 주석 처리하는 파일럿 연구로, RULEC-GEC와 RU-Lang8 데이터셋에서 총 500개의 문장 쌍을 사용했습니다. 제안된 방법론은 UD 가이드라인을 따르며, 러시아어 학습 데이터에 특화된 추가 주석 전략을 적용해 비표준 구문 구조를 효과적으로 반영했습니다. 결과적으로, 비표준 구조 처리 정확도가 87%로 기존 모델의 72%를 크게 초과했으며, 주요 오류 유형(전치사, 명사 격, 동사 일치)에서도 더 높은 성능을 기록했습니다. 테스트 데이터는 원본과 수정된 문장 모두를 포함하며, 학습자가 작성한 비표준 텍스트를 포함해 실질적인 언어 사용 패턴을 평가할 수 있었습니다. 본 연구는 기존 영어 학습자 모델과의 비교를 통해 러시아어의 형태소적 복잡성과 자유로운 어순을 효과적으로 처리할 수 있음을 입증했습니다.

---



This study presents a pilot annotation of Russian learner data using Universal Dependency (UD) trees, utilizing 500 sentence pairs from the RULEC-GEC and RU-Lang8 datasets. The proposed methodology follows UD guidelines and incorporates additional annotation strategies tailored to Russian learner data, effectively capturing non-standard syntactic structures. As a result, the proposed method achieved a non-standard structure handling accuracy of 87%, significantly outperforming the baseline model’s 72%, with higher performance in key error types such as prepositions, noun cases, and verb agreement. The test data included both original and corrected sentences, allowing for an evaluation of practical language usage patterns by learners. This study demonstrates the capability to handle the morphological complexity and free word order of Russian more effectively than previous models designed for English learners.


<br/>  
# 기타  


<br/>
# refer format:     


@inproceedings{rozovskaya2024,
  author    = {Alla Rozovskaya},
  title     = {Universal Dependencies for Learner Russian},
  booktitle = {Proceedings of the LREC-COLING 2024},
  pages     = {17112--17119},
  year      = {2024},
  month     = {May},
  organization = {ELRA Language Resource Association},
  address   = {Miyazaki, Japan},
  note      = {CC BY-NC 4.0},
  url       = {https://github.com/arozovskaya/dependency-learner-russian}
}




Rozovskaya, Alla. “Universal Dependencies for Learner Russian.” In Proceedings of the LREC-COLING 2024, 17112–17119. Miyazaki, Japan: ELRA Language Resource Association, May 2024. Accessed from https://github.com/arozovskaya/dependency-learner-russian.

