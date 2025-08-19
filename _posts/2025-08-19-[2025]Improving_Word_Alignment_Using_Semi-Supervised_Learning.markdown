---
layout: post
title:  "[2025]Improving Word Alignment Using Semi-Supervised Learning"
date:   2025-08-19 21:54:42 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

단어 정렬(word alignment) 문제를 해결하기 위해 반지도 학습(semi-supervised learning, 각 언어 쌍에 대해 레이블이 있는 데이터와 레이블이 없는 데이터를 사용하여 모델을 훈련) 방법을 제안(단어 정렬은 번역된 문장 쌍에서 소스 단어와 타겟 단어 간의 대응 관계를 식별하는 작업)  



추가적으로 가짜 라벨이 추가된 파인튜닝이 성능을 더욱 향상시킨다는 것을 관찰  


짧은 요약(Abstract) :


이 논문에서는 단어 정렬(word alignment)의 중요성과 현재의 최첨단 방법인 BinaryAlign의 한계를 다루고 있습니다. 단어 정렬은 기계 번역, 문장 임베딩, 동시 음성 번역 데이터 구축 등 다양한 자연어 처리(NLP) 작업에서 중요한 역할을 합니다. 현재의 방법들은 주로 수퍼바이즈드 데이터와 대규모 약한 수퍼바이즈드 데이터를 사용하고 있으며, 멀티링구얼 트랜스포머 인코더 기반 모델에 의존하고 있습니다. 그러나 저자들은 BinaryAlign이 라벨이 부족한 문제로 성능이 저하된다는 것을 발견하고, 적은 양의 병렬 데이터를 사용한 자기 학습(self-training)으로 이를 개선합니다. 또한, 멀티링구얼 대형 언어 모델을 단어 정렬기로 활용할 가능성을 탐구하며, 라벨이 있는 데이터로 미세 조정(fine-tuning)한 결과가 괜찮은 성능을 보이지만, 가짜 라벨이 추가된 훈련이 성능을 더욱 향상시킨다는 것을 관찰합니다. 이 연구는 제안된 반수퍼바이즈드 프레임워크가 다양한 단어 정렬 데이터셋에서 현재의 최첨단 방법을 초월하는 성능을 달성함을 보여줍니다.



This paper addresses the importance of word alignment and the limitations of the current state-of-the-art method, BinaryAlign. Word alignment plays a crucial role in various natural language processing (NLP) tasks, such as machine translation, sentence embedding, and the construction of training data for simultaneous speech translation. Current methods primarily rely on supervised data and large-scale weakly supervised data, depending on multilingual transformer encoder-based models. However, the authors find that BinaryAlign suffers from performance degradation due to insufficient labeled data and improve it with self-training using a small amount of parallel data. They also explore the possibility of using multilingual large language models as word aligners, observing that while fine-tuning with labeled data yields acceptable performance, augmenting training with pseudo-labeled data further enhances model performance. The study demonstrates that the proposed semi-supervised framework achieves state-of-the-art performance on various word alignment datasets, surpassing the current leading methods.


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



이 논문에서는 단어 정렬(word alignment) 성능을 향상시키기 위해 반지도 학습(semi-supervised learning) 프레임워크인 SemiAlign을 제안합니다. 이 방법은 두 가지 주요 모델, 즉 다국어 인코더 기반 모델(SemiAlign-E)과 다국어 대형 언어 모델(SemiAlign-D)을 사용하여 단어 정렬 작업을 수행합니다.

1. **모델 아키텍처**:
   - **다국어 인코더 모델**: mDeBERTa-v3-base와 같은 다국어 인코더 모델을 사용하여 단어 정렬을 수행합니다. 이 모델은 입력 문장에서 단어 간의 관계를 파악하는 데 강력한 성능을 보입니다.
   - **다국어 대형 언어 모델**: Llama-3.1-8B-Instruct와 같은 대형 언어 모델을 사용하여 단어 정렬을 수행합니다. 이 모델은 자연어 처리의 다양한 작업에서 뛰어난 성능을 발휘합니다.

2. **트레이닝 데이터**:
   - **라벨링된 데이터**: KFTT(교토 자유 번역 작업)와 같은 고품질의 라벨링된 데이터셋을 사용하여 모델을 훈련합니다. 이 데이터셋은 다양한 언어 쌍에 대한 단어 정렬 정보를 포함하고 있습니다.
   - **비라벨링된 데이터**: 대량의 비라벨링된 평행 텍스트를 활용하여 모델의 성능을 향상시킵니다. 이 데이터는 라벨링된 데이터와 결합되어 모델의 학습에 사용됩니다.

3. **특별한 기법**:
   - **자기 학습(self-training)**: 모델이 비라벨링된 데이터를 사용하여 스스로 라벨을 생성하고, 이를 통해 학습을 강화합니다. 이 과정에서 생성된 가짜 라벨(pseudo-labels)은 모델의 성능을 더욱 향상시키는 데 기여합니다.
   - **LoRA(저차원 적응)**: 대형 언어 모델의 파라미터를 효율적으로 조정하기 위해 LoRA 기법을 사용합니다. 이 방법은 모델의 성능을 유지하면서도 학습 비용을 줄이는 데 도움을 줍니다.

4. **성능 평가**:
   - 모델의 성능은 정밀도(Precision), 재현율(Recall), F1 점수, 정렬 오류율(AER) 등의 지표를 사용하여 평가됩니다. 실험 결과, 제안된 SemiAlign 방법이 기존의 최첨단 방법인 BinaryAlign보다 우수한 성능을 보임을 확인했습니다.

이러한 방법론을 통해, 논문은 단어 정렬 작업에서의 성능 향상을 위한 새로운 접근 방식을 제시하고 있습니다.

---




This paper proposes a semi-supervised learning framework called SemiAlign to improve word alignment performance. The method employs two main models: a multilingual encoder-based model (SemiAlign-E) and a multilingual large language model (SemiAlign-D) for the word alignment task.

1. **Model Architecture**:
   - **Multilingual Encoder Model**: The framework utilizes a multilingual encoder model, such as mDeBERTa-v3-base, to perform word alignment. This model demonstrates strong performance in understanding the relationships between words in input sentences.
   - **Multilingual Large Language Model**: A large language model like Llama-3.1-8B-Instruct is employed for word alignment. This model excels in various natural language processing tasks.

2. **Training Data**:
   - **Labeled Data**: High-quality labeled datasets, such as the Kyoto Free Translation Task (KFTT), are used to train the models. These datasets contain word alignment information for various language pairs.
   - **Unlabeled Data**: A large amount of unlabeled parallel text is leveraged to enhance the model's performance. This data is combined with labeled data to aid in the model's training.

3. **Special Techniques**:
   - **Self-Training**: The model uses unlabeled data to generate its own labels, thereby reinforcing its learning. The pseudo-labels generated in this process contribute to further enhancing the model's performance.
   - **LoRA (Low-Rank Adaptation)**: The LoRA technique is employed to efficiently adjust the parameters of large language models. This approach helps reduce training costs while maintaining model performance.

4. **Performance Evaluation**:
   - The model's performance is evaluated using metrics such as Precision, Recall, F1 score, and Alignment Error Rate (AER). Experimental results confirm that the proposed SemiAlign method outperforms the current state-of-the-art method, BinaryAlign.

Through these methodologies, the paper presents a novel approach to enhancing performance in word alignment tasks.


<br/>
# Results



이 논문에서는 단어 정렬(word alignment) 성능을 향상시키기 위해 반지도 학습(semi-supervised learning) 방법을 제안합니다. 연구팀은 기존의 최첨단 모델인 BinaryAlign이 레이블이 있는 데이터의 부족으로 인해 성능이 제한된다는 점을 발견하였고, 이를 해결하기 위해 소량의 병렬 데이터를 활용한 자기 학습(self-training) 기법을 도입했습니다. 또한, 다국어 대형 언어 모델(multilingual large language models, LLMs)을 단어 정렬기로 활용하는 가능성도 탐구하였습니다.

#### 실험 결과
1. **경쟁 모델**: 연구에서는 SpanAlign, WSPAlign, BinaryAlign과 같은 기존의 단어 정렬 모델과 비교하였습니다. BinaryAlign은 다국어 인코더 모델을 기반으로 하여 새로운 최첨단 성능을 달성하였습니다.
   
2. **테스트 데이터**: 실험은 일본어-영어(ja-en), 독일어-영어(de-en), 루마니아어-영어(ro-en), 중국어-영어(zh-en) 등 네 가지 언어 쌍에서 수행되었습니다. 각 언어 쌍에 대해 레이블이 있는 데이터와 레이블이 없는 데이터를 사용하여 모델을 훈련했습니다.

3. **메트릭**: 성능 평가는 정밀도(Precision), 재현율(Recall), F1 점수, 정렬 오류율(Alignment Error Rate, AER)로 측정되었습니다. F1 점수는 정밀도와 재현율의 조화 평균으로, 높은 점수가 더 나은 성능을 의미합니다. AER은 낮을수록 더 좋은 성능을 나타냅니다.

4. **비교 결과**: 제안된 SemiAlign 모델은 BinaryAlign보다 모든 언어 쌍에서 더 나은 성능을 보였습니다. 예를 들어, 일본어-영어 데이터셋에서 SemiAlign-E는 AER 점수를 12.38로 개선하였고, SemiAlign-D는 11.64로 개선되었습니다. 이는 기존 모델에 비해 상당한 성능 향상을 나타냅니다.

5. **제로샷 및 몇 샷 설정**: 제로샷 설정에서는 훈련 데이터가 없는 언어 쌍에 대해 SemiAlign 모델이 우수한 성능을 보였으며, 몇 샷 설정에서는 32개의 레이블이 있는 예제를 사용하여 훈련한 결과, SemiAlign-E가 BinaryAlign보다 더 나은 성능을 기록했습니다.

이러한 결과들은 제안된 반지도 학습 프레임워크가 단어 정렬 성능을 효과적으로 향상시킬 수 있음을 보여줍니다.

---




This paper proposes a semi-supervised learning method to improve word alignment performance. The research team discovered that the current state-of-the-art model, BinaryAlign, suffers from limited performance due to a lack of labeled data. To address this, they introduced a self-training technique that utilizes a small amount of parallel data. Additionally, they explored the possibility of using multilingual large language models (LLMs) as word aligners.

#### Experimental Results
1. **Competing Models**: The study compared existing word alignment models such as SpanAlign, WSPAlign, and BinaryAlign. BinaryAlign, based on multilingual encoder models, achieved new state-of-the-art performance.

2. **Test Data**: Experiments were conducted on four language pairs: Japanese-English (ja-en), German-English (de-en), Romanian-English (ro-en), and Chinese-English (zh-en). Both labeled and unlabeled data were used to train the models for each language pair.

3. **Metrics**: Performance was measured using Precision, Recall, F1 score, and Alignment Error Rate (AER). The F1 score is the harmonic mean of precision and recall, with higher scores indicating better performance. A lower AER indicates better performance.

4. **Comparison Results**: The proposed SemiAlign models outperformed BinaryAlign across all language pairs. For instance, on the Japanese-English dataset, SemiAlign-E improved the AER score to 12.38, while SemiAlign-D achieved 11.64. This indicates a significant performance enhancement over existing models.

5. **Zero-Shot and Few-Shot Settings**: In the zero-shot setting, where no training data is available for the target language pair, the SemiAlign model demonstrated excellent performance. In the few-shot setting, using 32 labeled examples for training, SemiAlign-E outperformed BinaryAlign.

These results demonstrate that the proposed semi-supervised learning framework can effectively enhance word alignment performance.


<br/>
# 예제



이 논문에서는 단어 정렬(word alignment) 문제를 해결하기 위해 반지도 학습(semi-supervised learning) 방법을 제안합니다. 단어 정렬은 번역된 문장 쌍에서 소스 단어와 타겟 단어 간의 대응 관계를 식별하는 작업입니다. 이 연구에서는 두 가지 주요 모델, 즉 다국어 인코더 기반 모델(SemiAlign-E)과 다국어 대형 언어 모델(SemiAlign-D)을 사용하여 성능을 향상시키는 방법을 탐구합니다.

#### 트레이닝 데이터와 테스트 데이터 예시

1. **트레이닝 데이터**:
   - **소스 문장**: "足利 _0 義満 _1 （_2 あしかが _3 よしみつ _4 ）_5 は_6 室町 _7 幕府 _8 の_9 第_10 3_11 代_12 征夷 _13 大_14 将軍 _15 （_16 在位 _17 1368_18 年_19 -_20 1394_21 年_22 ）_23 で_24 あ_25 る_26 。_27"
   - **타겟 문장**: "yoshimitsu_0 ashikaga_1 was_2 the_3 3rd_4 seii_5 taishogun_6 of_7 the_8 muromachi_9 shogunate_10 and_11 reigned_12 from_13 1368_14 to1394_15 ._16"
   - 이 데이터는 일본어와 영어 간의 병렬 문장으로, 각 단어에 ID가 부여되어 있습니다.

2. **테스트 데이터**:
   - **소스 문장**: "¶ 足利 _0 ¶ 義満 _1（_2 あしかが _3 よしみつ _4）_5 は_6 室町 _7 幕府 _8 の_9 第_10 3_11 代_12 征夷 _13 大_14 将軍 _15（_16 在位 _17 1368_18 年_19 -_20 1394_21 年_22）_23 で_24 あ_25 る_26。_27"
   - **타겟 문장**: "yoshimitsu_0 ashikaga_1 was_2 the_3 3rd_4 seii_5 taishogun_6 of_7 the_8 muromachi_9 shogunate_10 and_11 reigned_12 from_13 1368_14 to1394_15 ._16"
   - 여기서 "¶" 기호는 사용자가 관심 있는 단어를 표시합니다.

#### 아웃풋 예시
- **출력**: 
```json
{"足利 _0": ["ashikaga_1"]}
```
이 출력은 소스 문장 "足利 _0"가 타겟 문장 "ashikaga_1"과 정렬된다는 것을 나타냅니다.




This paper proposes a semi-supervised learning approach to address the word alignment problem. Word alignment aims to identify correspondences between source and target words in translated sentence pairs. The study explores two main models: a multilingual encoder-based model (SemiAlign-E) and a multilingual large language model (SemiAlign-D) to enhance performance.

#### Example of Training Data and Test Data

1. **Training Data**:
   - **Source Sentence**: "足利 _0 義満 _1 （_2 あしかが _3 よしみつ _4 ）_5 は_6 室町 _7 幕府 _8 の_9 第_10 3_11 代_12 征夷 _13 大_14 将軍 _15 （_16 在位 _17 1368_18 年_19 -_20 1394_21 年_22 ）_23 で_24 あ_25 る_26 。_27"
   - **Target Sentence**: "yoshimitsu_0 ashikaga_1 was_2 the_3 3rd_4 seii_5 taishogun_6 of_7 the_8 muromachi_9 shogunate_10 and_11 reigned_12 from_13 1368_14 to1394_15 ._16"
   - This data consists of parallel sentences in Japanese and English, with each word assigned an ID.

2. **Test Data**:
   - **Source Sentence**: "¶ 足利 _0 ¶ 義満 _1（_2 あしかが _3 よしみつ _4）_5 は_6 室町 _7 幕府 _8 の_9 第_10 3_11 代_12 征夷 _13 大_14 将軍 _15（_16 在位 _17 1368_18 年_19 -_20 1394_21 年_22）_23 で_24 あ_25 る_26。_27"
   - **Target Sentence**: "yoshimitsu_0 ashikaga_1 was_2 the_3 3rd_4 seii_5 taishogun_6 of_7 the_8 muromachi_9 shogunate_10 and_11 reigned_12 from_13 1368_14 to1394_15 ._16"
   - Here, the "¶" symbol marks the word of interest for the user.

#### Output Example
- **Output**: 
```json
{"足利 _0": ["ashikaga_1"]}
```
This output indicates that the source word "足利 _0" aligns with the target word "ashikaga_1".

<br/>
# 요약

이 논문에서는 세미-슈퍼바이즈드 학습을 통해 단어 정렬 성능을 향상시키기 위한 방법인 SemiAlign을 제안합니다. 실험 결과, SemiAlign-E와 SemiAlign-D 모델이 기존의 최첨단 방법인 BinaryAlign보다 우수한 성능을 보였으며, 특히 적은 양의 병렬 데이터를 활용하여 성능을 개선할 수 있음을 보여주었습니다. 예를 들어, 일본어-영어 데이터셋에서 SemiAlign-E는 AER 점수를 12.38로 개선했습니다.

In this paper, the authors propose a method called SemiAlign to improve word alignment performance using semi-supervised learning. Experimental results show that both the SemiAlign-E and SemiAlign-D models outperform the current state-of-the-art method, BinaryAlign, demonstrating the ability to enhance performance with a small amount of parallel data. For instance, SemiAlign-E improved the AER score to 12.38 on the Japanese-English dataset.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **Figure 1**: 일본어-영어 단어 정렬 예시를 보여줍니다. 이 그림은 소스와 타겟 문장 간의 단어 정렬 관계를 시각적으로 나타내며, 각 단어의 위치 정보도 고려하고 있습니다.
   - **Figure 2**: SemiAlign-E와 SemiAlign-D의 F1 점수가 레이블이 있는 데이터의 양에 따라 어떻게 변화하는지를 보여줍니다. 이 그래프는 레이블이 적은 경우에도 SemiAlign이 BinaryAlign보다 더 나은 성능을 보임을 나타냅니다.
   - **Figure 3**: SemiAlign의 훈련 프로세스를 설명하는 다이어그램으로, 멀티링구얼 인코더와 디코더 모델의 학습 과정을 보여줍니다.
   - **Figure 4 & 5**: LLM 기반 단어 정렬의 프롬프트 예시로, "full mode"와 "marker mode"의 차이를 설명합니다. "marker mode"는 특정 단어에 대한 정렬 정보를 요청하는 방식입니다.

2. **테이블**:
   - **Table 1**: 다양한 단어 정렬 방법의 성능을 비교합니다. SemiAlign-E와 SemiAlign-D는 BinaryAlign보다 높은 F1 점수와 낮은 AER 점수를 기록하여 새로운 최첨단 성능을 달성했습니다.
   - **Table 2**: 제로샷 설정에서의 성능을 보여줍니다. SemiAlign-E는 다른 언어 쌍에서 훈련된 모델이 저자원 언어에 대해 좋은 성능을 보임을 나타냅니다.
   - **Table 3**: 몇 개의 레이블이 있는 예제만으로 훈련했을 때의 성능을 보여줍니다. SemiAlign-E는 BinaryAlign보다 더 나은 성능을 보였습니다.
   - **Table 4**: 실제 추론 시간을 비교합니다. SemiAlign-D는 더 큰 모델이지만, 추론 시간이 더 길어지는 단점이 있습니다.
   - **Table 15**: 가능한 정렬을 포함한 데이터와 포함하지 않은 데이터의 성능을 비교합니다. 가능한 정렬을 사용하지 않은 경우 성능이 더 좋음을 보여줍니다.

3. **어펜딕스**:
   - 어펜딕스에서는 데이터 세트 통계, 병렬 데이터 세트의 세부 사항, 훈련 세부 사항, LoRA 훈련 세부 사항, 평가 메트릭 세부 사항 등을 제공합니다. 이 정보는 연구의 재현성과 이해를 돕습니다.




1. **Diagrams and Figures**:
   - **Figure 1**: Shows an example of word alignment between Japanese and English. This figure visually represents the correspondence between words in the source and target sentences, considering the positional information of each word.
   - **Figure 2**: Illustrates how the F1 scores of SemiAlign-E and SemiAlign-D change with the amount of labeled data. This graph indicates that even with limited labeled data, SemiAlign outperforms BinaryAlign.
   - **Figure 3**: A diagram explaining the training process of SemiAlign, showing the learning process of multilingual encoder and decoder models.
   - **Figures 4 & 5**: Prompt examples for LLM-based word alignment, illustrating the differences between "full mode" and "marker mode." "Marker mode" requests alignment information for a specific word.

2. **Tables**:
   - **Table 1**: Compares the performance of various word alignment methods. SemiAlign-E and SemiAlign-D achieve higher F1 scores and lower AER scores than BinaryAlign, marking a new state-of-the-art performance.
   - **Table 2**: Shows performance in a zero-shot setting. SemiAlign-E demonstrates good performance on low-resource languages when trained on other language pairs.
   - **Table 3**: Displays performance when trained with only a few labeled examples. SemiAlign-E outperforms BinaryAlign in this setting.
   - **Table 4**: Compares actual inference times. While SemiAlign-D is a larger model, it has the drawback of longer inference times.
   - **Table 15**: Compares performance using labeled data with and without possible alignments. Results show better performance without using possible alignments.

3. **Appendix**:
   - The appendix provides details on dataset statistics, parallel data specifics, training details, LoRA training specifics, and evaluation metric details. This information aids in the reproducibility and understanding of the research.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{Miao2025,
  author    = {Zhongtao Miao and Qiyu Wu and Masaaki Nagata and Yoshimasa Tsuruoka},
  title     = {Improving Word Alignment Using Semi-Supervised Learning},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {19871--19888},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  address   = {Bangkok, Thailand},
  month     = {July 27 - August 1}
}
```

### 시카고 스타일 인용
Miao, Zhongtao, Qiyu Wu, Masaaki Nagata, and Yoshimasa Tsuruoka. "Improving Word Alignment Using Semi-Supervised Learning." In *Findings of the Association for Computational Linguistics: ACL 2025*, 19871–19888. Bangkok, Thailand: Association for Computational Linguistics, July 27 - August 1, 2025.
