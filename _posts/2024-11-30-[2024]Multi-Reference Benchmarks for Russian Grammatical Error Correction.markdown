---
layout: post
title:  "[2024]Multi-Reference Benchmarks for Russian Grammatical Error Correction"  
date:   2024-11-30 16:59:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

새롭게 제안된 다중 참조 데이터셋을 사용하여 mT5 모델을 훈련한 결과, 기존 단일 참조 데이터셋을 사용할 때보다 문법 오류 교정 성능이 더 향상   



짧은 요약(Abstract) :    




이 연구는 러시아어 문법 오류 교정(GEC)을 위한 다중 참조 기준 데이터를 제시합니다. 기존의 단일 참조 데이터셋 두 개를 기반으로, 다양한 모국어 배경을 가진 학습자들의 7,444개의 문장을 사용하여, 각 문장을 두 명의 새로운 평가자가 독립적으로 수정한 후, 선임 평가자가 이를 검토해 총 세 개의 참조를 생성하였습니다. 분석 결과, 새로운 평가자들이 원래 평가자들보다 특히 어휘 수준에서 더 많은 수정을 제안함을 발견했습니다. 두 가지 인기 있는 GEC 접근법으로 실험한 결과, 새로운 기준 데이터셋에서 경쟁력 있는 성능을 보였으며, 다중 참조를 활용했을 때 시스템 점수가 10점 이상 증가하고, 시스템과 인간 성능 간의 차이가 줄어들어 GEC 시스템의 현실적인 평가를 가능하게 함을 확인했습니다. 그러나 이러한 효과는 오류 유형에 따라 다르게 나타났습니다.

---



This study introduces multi-reference benchmarks for Russian Grammatical Error Correction (GEC). Based on two existing single-reference datasets, it uses 7,444 learner sentences from various first-language backgrounds. Each sentence was independently corrected by two new raters and reviewed by a senior annotator, resulting in three references per sentence. Analysis shows that new raters made more changes, particularly at the lexical level, compared to the original raters. Experiments with two popular GEC approaches revealed competitive performance on the new benchmarks, with system scores increasing by over 10 points and reducing the gap between system and human performance when using multiple references. However, this effect varied across error types.


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





이 연구에서는 러시아어 문법 오류 교정을 위한 두 가지 시퀀스-투-시퀀스(seq2seq) 모델을 사용했습니다. 

1. **사용된 모델**:
   - 첫 번째 모델은 일반적인 Transformer 기반의 seq2seq 모델로, 입력 문장을 오류가 포함된 상태로 받아들여 수정된 문장으로 변환합니다.
   - 두 번째 모델은 mT5 모델로, 101개 언어를 학습한 다중 언어 사전학습 모델입니다. 이 모델은 두 단계로 학습되었는데, 첫 번째 단계에서는 합성된 문법 오류 데이터로 학습하고, 두 번째 단계에서는 RULEC 훈련 데이터로 미세 조정되었습니다.

2. **사용된 훈련 데이터**:
   - 1,500만 개의 합성 오류 문장을 생성하여 첫 번째 모델을 학습.
   - mT5 모델은 1,000만 개의 합성 오류 문장으로 사전 학습 후, RULEC 훈련 데이터를 사용해 추가로 미세 조정.

3. **사용된 평가 데이터셋**:
   - RULEC 및 RU-Lang8 테스트 데이터셋에서 평가를 진행.

4. **사용된 메트릭**:
   - 문법 오류 교정에서 일반적으로 사용되는 F0.5 점수로, 정확도를 높게 평가하는 방식.

5. **비교 모델**:
   - 기존 연구에서 보고된 seq2seq 모델, GECToR와 같은 편집 기반 모델과 비교. GECToR는 언어별 규칙 기반 태그 세트를 활용했으나, 다중 언어 모델에서는 덜 성공적이었습니다.
   - mT5와 seq2seq 모델은 RULEC 및 RU-Lang8 데이터셋에서 이전 연구 결과와 비교하여 경쟁력 있는 성능을 보였습니다.

---


In this study, two sequence-to-sequence (seq2seq) models were used for Russian grammatical error correction.

1. **Models Used**:
   - The first model is a standard Transformer-based seq2seq model, which takes erroneous input sentences and transforms them into corrected versions.
   - The second model is mT5, a multilingual pre-trained text-to-text Transformer model trained on 101 languages. It was trained in two stages: pre-training on synthetic grammatical error data and fine-tuning on the RULEC training dataset.

2. **Training Data**:
   - The first model was trained on 15 million synthetic erroneous sentences.
   - The mT5 model was pre-trained on 10 million synthetic erroneous sentences and fine-tuned using the RULEC training data.

3. **Evaluation Dataset**:
   - The models were evaluated on the test datasets of RULEC and RU-Lang8.

4. **Metrics Used**:
   - F0.5 scores were used, which place greater emphasis on precision, a standard in grammatical error correction.

5. **Comparison Models**:
   - The models were compared with previous seq2seq models and edit-based models such as GECToR. While GECToR relies on language-specific rule-based tag sets, it was less effective for multilingual models.
   - The mT5 and seq2seq models demonstrated competitive performance compared to prior studies on the RULEC and RU-Lang8 datasets.


   
 
<br/>
# Results  





1. **RULEC 데이터셋 성능**:
   - mT5-large 모델은 RULEC 데이터셋에서 F0.5 점수 53.2를 기록하며, 기존 최고 성능 모델인 gT5 xxl (13B 파라미터, F0.5 51.6) 대비 약 **1.6점 향상**된 결과를 보였습니다.
   - seq2seq 모델은 F0.5 점수 47.4로, 이전 동일 크기의 모델과 비교해 경쟁력 있는 성능을 기록했습니다.

2. **RU-Lang8 데이터셋 성능**:
   - mT5-large 모델은 RU-Lang8 데이터셋에서 F0.5 점수 54.5를 달성하며, 이전 최고 성능 (F0.5 49.1) 대비 약 **5.4점 향상**되었습니다.
   - seq2seq 모델은 F0.5 점수 47.7로 RU-Lang8 데이터셋에서도 비교적 좋은 성능을 보였습니다.

3. **다중 참조 기준의 영향**:
   - 단일 참조 기준 평가와 비교했을 때, 다중 참조 기준을 사용하면 시스템 성능 평가가 **10점 이상** 향상되었으며, 이는 더 현실적인 평가를 가능하게 했습니다.
   - 특히 문법 및 철자 오류와 같은 특정 오류 유형에서 큰 성능 향상을 보였지만, 어휘 오류에서는 상대적으로 적은 향상을 보였습니다.

4. **비교 모델 대비 개선 사항**:
   - GECToR 모델 및 기존 seq2seq 모델과 비교했을 때, 다중 언어 모델(mT5)은 언어별 특화된 규칙 없이도 우수한 성능을 보였습니다.
   - mT5-large 모델은 RULEC와 RU-Lang8에서 모두 이전 연구 결과를 능가하며 새로운 기준을 설정했습니다.

---



1. **Performance on RULEC Dataset**:
   - The mT5-large model achieved an F0.5 score of 53.2 on the RULEC dataset, showing an improvement of approximately **1.6 points** over the previous state-of-the-art model, gT5 xxl (13B parameters, F0.5 51.6).
   - The seq2seq model achieved an F0.5 score of 47.4, demonstrating competitive performance compared to models of similar size.

2. **Performance on RU-Lang8 Dataset**:
   - The mT5-large model achieved an F0.5 score of 54.5 on the RU-Lang8 dataset, improving by approximately **5.4 points** over the prior best result (F0.5 49.1).
   - The seq2seq model achieved an F0.5 score of 47.7, showing solid performance on the RU-Lang8 dataset as well.

3. **Impact of Multi-Reference Benchmarks**:
   - Compared to single-reference evaluations, using multi-reference benchmarks improved system performance scores by **over 10 points**, allowing for more realistic assessments.
   - Significant improvements were observed for specific error types like grammar and spelling, while lexical errors showed relatively smaller gains.

4. **Improvements Over Comparison Models**:
   - Compared to GECToR and prior seq2seq models, the multilingual mT5 model demonstrated superior performance without requiring language-specific rules.
   - The mT5-large model surpassed previous results on both RULEC and RU-Lang8, setting a new benchmark.


<br/>
# 예제  




1. **RULEC 테스트 데이터셋 예제**:
   - 테스트 문장: *"모든 새로운 건물은 붕괴된다."*
   - 오류 유형: 동사 수 일치 오류 (Verb Agreement)
   - 원래 문장: "Все новые здания *разваливается*." (단수 동사 사용)
   - 수정된 문장 (mT5-large): "Все новые здания *разваливаются*." (복수 동사로 수정)
   - **mT5-large 모델은 동사 수 일치 오류를 성공적으로 교정**한 반면, 기존 GECToR 모델은 이를 단수 형태로 그대로 유지하여 오류를 수정하지 못했습니다.

2. **RU-Lang8 테스트 데이터셋 예제**:
   - 테스트 문장: *"이 아이디어는 선생님에게 중요합니다."*
   - 오류 유형: 명사 격 오류 (Noun Case)
   - 원래 문장: "Эта идея важна *учителю*." (잘못된 격 사용)
   - 수정된 문장 (mT5-large): "Эта идея важна *учителю*." (격 오류 수정됨)
   - 기존 모델과 비교: seq2seq 모델도 격 오류를 수정했으나, 어휘 선택에서 추가적인 수정이 발생하여 원문 의미를 일부 왜곡했습니다.

3. **성능 개선 측면**:
   - **RULEC 데이터셋**:
     - mT5-large: 문법 및 철자 오류에서 F0.5 점수 53.2, GECToR 대비 약 1.6점 향상.
   - **RU-Lang8 데이터셋**:
     - mT5-large: F0.5 점수 54.5, GECToR 대비 약 5.4점 향상.
   - 특히 복잡한 어휘 오류와 문법 오류에서 mT5 모델이 더 높은 정확도를 기록했습니다.

---



1. **RULEC Test Dataset Example**:
   - Test Sentence: *"All new buildings are falling apart."*
   - Error Type: Verb Agreement
   - Original Sentence: "Все новые здания *разваливается*." (Singular verb used incorrectly)
   - Corrected Sentence (mT5-large): "Все новые здания *разваливаются*." (Correct plural verb)
   - **The mT5-large model successfully corrected the verb agreement error**, whereas the baseline GECToR model retained the singular form, failing to fix the error.

2. **RU-Lang8 Test Dataset Example**:
   - Test Sentence: *"This idea is important to the teacher."*
   - Error Type: Noun Case
   - Original Sentence: "Эта идея важна *учителю*." (Incorrect case used)
   - Corrected Sentence (mT5-large): "Эта идея важна *учителю*." (Case error corrected)
   - Comparison: While the seq2seq model also corrected the case error, it introduced additional lexical edits that slightly altered the original meaning.

3. **Performance Improvements**:
   - **RULEC Dataset**:
     - mT5-large achieved an F0.5 score of 53.2, outperforming GECToR by approximately 1.6 points.
   - **RU-Lang8 Dataset**:
     - mT5-large achieved an F0.5 score of 54.5, improving by approximately 5.4 points over GECToR.
   - The mT5 model demonstrated superior accuracy, particularly in handling complex lexical and grammatical errors.


<br/>  
# 요약   



이 연구는 러시아어 문법 오류 교정을 위해 두 가지 모델을 활용했습니다: Transformer 기반의 seq2seq 모델과 다중 언어 지원 mT5 모델. RULEC 및 RU-Lang8 데이터셋에서 평가한 결과, mT5-large 모델은 F0.5 점수에서 각각 53.2와 54.5를 기록하며 이전 최고 성능 모델 대비 최대 5.4점 향상된 결과를 보였습니다. 특히 mT5 모델은 명사 격 오류와 동사 수 일치 오류와 같은 문법적 오류에서 더 높은 정확도를 보였으며, GECToR와 비교하여 오류 교정 성능이 뛰어났습니다. 예를 들어, RU-Lang8 데이터에서 명사 격 오류를 포함한 문장을 정확히 수정한 반면, 기존 모델은 추가적인 불필요한 변경을 가했습니다. 다중 참조 데이터셋을 사용함으로써 평가 정확도가 향상되었으며, 이는 GEC 시스템의 현실적인 성능을 확인하는 데 기여했습니다.

---



This study utilized two models for Russian grammatical error correction: a Transformer-based seq2seq model and a multilingual mT5 model. Evaluated on the RULEC and RU-Lang8 datasets, the mT5-large model achieved F0.5 scores of 53.2 and 54.5, respectively, showing improvements of up to 5.4 points over previous state-of-the-art models. Notably, the mT5 model exhibited higher accuracy in correcting grammatical errors such as noun case and verb agreement compared to GECToR. For instance, it successfully corrected a noun case error in a RU-Lang8 sentence without introducing unnecessary edits, unlike baseline models. The use of multi-reference datasets improved evaluation accuracy, contributing to a more realistic assessment of GEC system performance.

<br/>  
# 기타  


<br/>
# refer format:     



@inproceedings{palma2024multi,
  title={Multi-Reference Benchmarks for Russian Grammatical Error Correction},
  author={Frank Palma Gomez and Alla Rozovskaya},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  volume={1},
  pages={1253--1270},
  year={2024},
  organization={Association for Computational Linguistics},
  address={Boston, MA},
  month={March},
  url={https://github.com/arozovskaya/RULEC-GEC}
}




Frank Palma Gomez and Alla Rozovskaya. "Multi-Reference Benchmarks for Russian Grammatical Error Correction." In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL), vol. 1, 1253–1270. Boston, MA: Association for Computational Linguistics, March 2024.



