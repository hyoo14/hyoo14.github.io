---
layout: post
title:  "[2026]Learning to Combine AI Annotations for Improved Biomedical Relevance Labeling"
date:   2026-07-17 02:46:06 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 생물 의학 초록 간의 관련성을 정확하게 라벨링할 때, AI가 생성한 주석을 비전문가의 기여로 간주하고 이를 학습 순위 프레임워크를 사용하여 결합합니다. 연구 결과는 전체 주석 품질이 상당히 향상    


짧은 요약(Abstract) :


이 논문의 초록에서는 생물 의학 초록 간의 관련성을 정확하게 라벨링하는 것이 정보 검색, 의미 유사성 모델링, 순위 시스템 훈련 및 기타 자연어 처리 작업을 개선하는 데 필수적이라고 강조하고 있습니다. 그러나 수작업으로 주석을 달기는 시간과 노력이 많이 들고 비용이 많이 듭니다. 대형 언어 모델(LLMs)이 자동 주석 달기를 촉진할 수 있지만, 특히 도메인 특정 작업에서는 여전히 인간 전문가 수준의 정확도에 미치지 못합니다. 여러 비전문가의 주석을 결합하면 훈련된 전문가의 성능과 비슷하거나 이를 초과할 수 있다는 연구 결과를 바탕으로, AI가 생성한 주석을 비전문가의 기여로 간주하고 이를 학습 순위 프레임워크를 사용하여 결합합니다. 연구 결과는 전체 주석 품질이 상당히 향상되었음을 보여줍니다. 제안된 방법은 대규모 생물 의학 응용 프로그램에서 신뢰할 수 있는 성능을 유지하면서 인간 주석에 대한 의존도를 줄이는 데 유망합니다.



The abstract of this paper emphasizes that accurate labeling of relevance between biomedical abstracts is essential for improving information retrieval, semantic similarity modeling, training ranking systems, and other Natural Language Processing tasks. However, manual annotation is time-consuming, labor-intensive, and costly. While large language models (LLMs) can facilitate automated annotation, their performance still falls short of human expert-level accuracy, especially in domain-specific tasks. Based on studies showing that combining annotations from multiple non-expert annotators can achieve performance comparable to or even exceeding that of trained experts, the authors treat AI-generated annotations as contributions from non-expert annotators and combine them using a Learning to Rank framework. The results demonstrate significant improvement in overall annotation quality. The proposed method looks promising for reducing reliance on human annotation while maintaining reliable performance for large-scale biomedical applications.


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



이 연구에서는 생물 의학 관련 초록 간의 관련성을 평가하기 위해 여러 가지 주석 방법을 결합하는 새로운 접근 방식을 제안합니다. 연구의 주요 목표는 AI가 생성한 주석을 비전문가의 주석으로 간주하고, 이를 효과적으로 결합하여 주석 품질을 향상시키는 것입니다. 이를 위해 다음과 같은 방법론을 사용합니다.

1. **주석 방법**: 연구에서는 총 7가지 주석 방법을 고려합니다. 이 중 3가지는 전통적인 텍스트 유사도 접근 방식(BM25, MedCPT, LitSense2)이고, 4가지는 다양한 프롬프트를 사용하여 생성된 AI 기반 주석입니다. AI 주석은 GPT-5.1 모델을 사용하여 생성됩니다.

2. **학습 방법**: 주석 품질을 향상시키기 위해, 연구팀은 Learning to Rank (LTR) 프레임워크를 채택합니다. 이 방법은 주어진 쿼리에 대해 AI 주석의 순위를 학습하고, 이를 통해 인간의 판단과 최대한 일치하도록 모델을 훈련합니다. 특히, 연구에서는 절대적인 동의보다는 순위의 일치를 강조하여, 서로 다른 주석 방법에서 발생하는 다양한 신호를 통합합니다.

3. **데이터셋**: 연구에서는 RELISH 데이터셋과 NCBI 데이터셋을 사용하여 모델을 평가합니다. RELISH 데이터셋은 약 196,000개의 PubMed 초록 쌍으로 구성되어 있으며, NCBI 데이터셋은 5,000개의 PubMed 초록 쌍으로 구성되어 있습니다. 각 데이터셋은 서로 다른 주석 기준을 가지고 있어, 연구 결과의 일반화에 한계가 있을 수 있습니다.

4. **모델 평가**: 모델의 성능은 Spearman 상관 계수를 사용하여 평가됩니다. 연구팀은 훈련 세트에서 최적의 특성 조합을 선택하고, 이를 통해 최종 모델을 훈련시킵니다. 최종 모델은 테스트 세트에서 0.460의 Spearman 상관 계수를 달성하여, 개별 주석 방법보다 우수한 성능을 보였습니다.

5. **결론**: 연구 결과, 제안된 방법은 AI 주석을 효과적으로 결합하여 인간 전문가의 주석 품질에 근접하는 성능을 달성할 수 있음을 보여줍니다. 이는 대규모 생물 의학 응용 프로그램에서 인간 주석에 대한 의존도를 줄이는 데 기여할 수 있습니다.

### English Version

This study proposes a novel approach to combine multiple annotation methods for evaluating the relevance between biomedical abstracts. The main goal of the research is to treat AI-generated annotations as contributions from non-expert annotators and effectively combine them to improve annotation quality. The methodology includes the following components:

1. **Annotation Methods**: The study considers a total of seven annotation methods. Among these, three are traditional text similarity approaches (BM25, MedCPT, LitSense2), and four are AI-based annotations generated using different prompting strategies. The AI annotations are produced using the GPT-5.1 model.

2. **Learning Method**: To enhance annotation quality, the research team adopts a Learning to Rank (LTR) framework. This method learns to rank AI annotations for a given query, training the model to align as closely as possible with human judgments. Specifically, the study emphasizes agreement in ranking rather than absolute agreement, integrating diverse signals from different annotation methods.

3. **Datasets**: The study utilizes the RELISH dataset and the NCBI dataset for model evaluation. The RELISH dataset consists of approximately 196,000 annotated pairs of PubMed abstracts, while the NCBI dataset comprises 5,000 pairs. Each dataset has different annotation criteria, which may limit the generalization of the study's findings.

4. **Model Evaluation**: The model's performance is evaluated using Spearman correlation coefficients. The research team selects the optimal combination of features from the training set and trains the final model based on this selection. The final model achieves a Spearman correlation of 0.460 on the test set, outperforming individual annotation methods.

5. **Conclusion**: The results indicate that the proposed method can effectively combine AI annotations to achieve performance comparable to that of human experts. This could contribute to reducing reliance on human annotation in large-scale biomedical applications.


<br/>
# Results



이 연구에서는 생물 의학 관련 초록 간의 관련성을 평가하기 위해 여러 가지 주석 방법을 비교하고, 이를 통해 최적의 주석 모델을 개발하는 과정을 다루고 있습니다. 연구의 주요 목표는 AI 기반 주석과 전통적인 정보 검색 방법을 결합하여 주석 품질을 향상시키는 것입니다.

#### 데이터셋
연구에 사용된 데이터셋은 RELISH와 NCBI 데이터셋으로, RELISH는 약 196,000개의 PubMed 초록 쌍으로 구성되어 있으며, NCBI 데이터셋은 5,000개의 PubMed 초록 쌍으로 구성되어 있습니다. RELISH 데이터셋은 1-3의 세 점 척도로 주석이 달려 있으며, NCBI 데이터셋은 1-4의 척도로 주석이 달려 있습니다.

#### 방법론
연구에서는 총 7가지 주석 방법을 고려했습니다. 이 중 4가지는 AI 기반 주석 방법(GPT-5.1을 사용)이고, 나머지 3가지는 전통적인 정보 검색 방법(BM25, MedCPT, LitSense2)입니다. 각 주석 방법의 성능은 Spearman 상관 계수를 사용하여 평가되었습니다.

#### 결과
모델의 성능을 평가하기 위해 데이터셋을 세 개의 폴드로 나누어 두 개의 폴드로 훈련하고 나머지 하나로 테스트를 진행했습니다. 각 AI 주석 방법의 Spearman 상관 계수는 다음과 같았습니다:

- F1: 0.326
- F2: 0.320
- F3: 0.354
- F4: 0.397
- F5: 0.280
- F6: 0.314
- F7: 0.362

이러한 결과는 AI 주석 방법이 훈련된 전문가의 주석과 비교했을 때 여전히 부족하다는 것을 보여줍니다. 그러나 최적의 특성 조합([F1, F2, F3, F4, F7])을 사용하여 훈련된 모델 Moptimal은 테스트 세트에서 Spearman 상관 계수 0.460을 달성하였으며, 이는 모든 개별 특성의 성능을 초과하는 결과입니다. 이 결과는 Moptimal 모델이 전문가의 주석 품질에 필적하는 성능을 달성할 수 있음을 시사합니다.

#### 결론
이 연구는 AI 주석과 전통적인 정보 검색 방법을 결합하여 주석 품질을 향상시키는 방법을 제시하였으며, 최적의 모델이 전문가의 주석 품질에 근접할 수 있음을 보여주었습니다. 향후 연구에서는 더 많은 데이터셋과 다양한 LLM을 사용하여 모델의 일반화 가능성을 높일 계획입니다.

---



This study focuses on comparing various annotation methods for evaluating the relevance between biomedical abstracts and developing an optimal annotation model. The primary goal of the research is to enhance annotation quality by combining AI-based annotations with traditional information retrieval methods.

#### Datasets
The datasets used in the study include RELISH and NCBI datasets. RELISH consists of approximately 196,000 pairs of PubMed abstracts, while the NCBI dataset comprises 5,000 pairs of PubMed abstracts. The RELISH dataset is annotated on a three-point scale (1-3), while the NCBI dataset is annotated on a four-point scale (1-4).

#### Methodology
The study considers a total of seven annotation methods. Four of these are AI-based annotation methods (using GPT-5.1), and the remaining three are traditional information retrieval methods (BM25, MedCPT, LitSense2). The performance of each annotation method is evaluated using Spearman correlation coefficients.

#### Results
To evaluate the model's performance, the dataset was split into three folds, using two folds for training and the remaining one for testing. The Spearman correlation coefficients for each AI annotation method were as follows:

- F1: 0.326
- F2: 0.320
- F3: 0.354
- F4: 0.397
- F5: 0.280
- F6: 0.314
- F7: 0.362

These results indicate that AI annotation methods still fall short compared to trained expert annotations. However, the model Moptimal, trained using the optimal feature combination ([F1, F2, F3, F4, F7]), achieved a Spearman correlation of 0.460 on the test set, surpassing the performance of all individual features. This result suggests that the Moptimal model can achieve annotation quality comparable to that of human experts.

#### Conclusion
This study presents a method for improving annotation quality by combining AI annotations with traditional information retrieval methods, demonstrating that the optimal model can approach the quality of expert annotations. Future research will aim to enhance the model's generalizability by utilizing larger datasets and various LLMs.


<br/>
# 예제



이 논문에서는 생물 의학 문헌의 관련성 레이블링을 개선하기 위해 AI 주석을 결합하는 방법을 제안합니다. 연구의 주요 목표는 대규모 생물 의학 애플리케이션에서 신뢰할 수 있는 성능을 유지하면서 인간 주석에 대한 의존도를 줄이는 것입니다. 이를 위해, 연구팀은 여러 AI 모델의 주석을 결합하여 최적의 결과를 도출하는 방법을 사용합니다.

#### 데이터셋
1. **RELISH 데이터셋**: 약 196,000개의 PubMed 초록 쌍으로 구성되어 있으며, 각 쌍은 1(관련 없음)에서 3(관련 있음)까지의 세 점 척도로 레이블이 지정됩니다.
2. **NCBI 데이터셋**: 5,000개의 PubMed 초록 쌍으로 구성되어 있으며, 13명의 주석가(7명의 훈련된 주석가와 6명의 비훈련된 주석가)가 1에서 4까지의 척도로 레이블을 지정합니다.

#### 방법론
연구팀은 다음과 같은 7가지 주석 방법을 고려합니다:
- **텍스트 유사도 접근법**: BM25, MedCPT, LitSense2
- **AI 기반 변형**: GPT-5.1을 사용하여 다양한 프롬프트(프롬프트 1-4)를 통해 생성된 주석

각 AI 주석의 성능을 평가하기 위해 Spearman 상관 계수를 사용하여 훈련 세트와 테스트 세트에서 개별 AI 주석의 품질을 측정합니다. 최적의 주석 조합을 찾기 위해 Learning-to-Rank(LTR) 접근법을 사용하여 여러 주석 방법의 조합을 평가합니다.

#### 예시
- **훈련 데이터**: 특정 쿼리 문서에 대해 AI 모델이 생성한 주석과 인간 주석 간의 일치도를 평가합니다. 예를 들어, 쿼리 문서 A와 B가 주어졌을 때, AI 모델이 A와 B의 관련성을 2로 평가하고, 인간 주석가가 3으로 평가한 경우, 이 두 주석 간의 Spearman 상관 계수를 계산합니다.
- **테스트 데이터**: 훈련 과정에서 사용되지 않은 새로운 쿼리 문서에 대해 최적의 모델을 적용하여 예측 점수를 생성하고, 이 점수를 인간 주석과 비교하여 모델의 성능을 평가합니다.

이러한 과정을 통해 연구팀은 AI 주석의 결합이 인간 주석의 품질에 근접할 수 있음을 보여주고, 최종 모델이 인간 전문가의 주석 품질과 유사한 성능을 달성할 수 있음을 입증합니다.

---




This paper proposes a method for improving biomedical relevance labeling by combining AI annotations. The main goal of the study is to reduce reliance on human annotations while maintaining reliable performance for large-scale biomedical applications. To achieve this, the research team uses a method to combine annotations from multiple AI models to derive optimal results.

#### Datasets
1. **RELISH Dataset**: Composed of approximately 196,000 pairs of PubMed abstracts, each labeled on a three-point scale from 1 (irrelevant) to 3 (relevant).
2. **NCBI Dataset**: Consists of 5,000 pairs of PubMed abstracts, annotated on a scale of 1 to 4 by 13 judges (7 trained judges and 6 untrained judges).

#### Methodology
The research team considers seven annotation methods:
- **Text Similarity Approaches**: BM25, MedCPT, LitSense2
- **AI-based Variants**: Annotations generated using GPT-5.1 with various prompts (Prompt 1-4)

To evaluate the performance of each AI annotation, Spearman correlation is used to measure the quality of individual AI annotations in both the training and test sets. A Learning-to-Rank (LTR) approach is employed to find the optimal combination of multiple annotation methods.

#### Example
- **Training Data**: The agreement between AI-generated annotations and human annotations is evaluated for specific query documents. For instance, if AI rates the relevance of documents A and B as 2, while a human annotator rates it as 3, the Spearman correlation between these two annotations is calculated.
- **Test Data**: The optimal model is applied to new query documents that were not used during the training process to generate predicted scores, which are then compared to human annotations to assess the model's performance.

Through this process, the research team demonstrates that combining AI annotations can approach the quality of human annotations, proving that the final model can achieve performance comparable to that of human experts.

<br/>
# 요약


이 연구에서는 여러 AI 기반 주석 방법을 결합하여 생물 의학 관련성 레이블링의 품질을 향상시키기 위해 Learning to Rank(LTR) 접근 방식을 사용하였다. 실험 결과, 최적의 모델이 인간 전문가의 주석 품질과 유사한 성능을 보였으며, 이는 AI 주석의 조합이 효과적임을 나타낸다. 최종 모델은 Spearman 상관계수 0.460을 기록하여 개별 AI 주석 방법보다 유의미하게 향상된 결과를 보여주었다.



This study employed a Learning to Rank (LTR) approach to improve the quality of biomedical relevance labeling by combining various AI-based annotation methods. The results demonstrated that the optimal model achieved performance comparable to that of human experts, indicating the effectiveness of combining AI annotations. The final model recorded a Spearman correlation of 0.460, significantly outperforming individual AI annotation methods.

<br/>
# 기타



1. **테이블 1 및 테이블 2**: 
   - 이 테이블들은 훈련 세트와 테스트 세트에서 각 AI 랭커의 Spearman 상관 계수를 보여줍니다. 
   - 결과는 AI 기반 방법들이 훈련 세트에서 평균 0.336, 테스트 세트에서 평균 0.350의 상관 계수를 기록했음을 나타냅니다. 이는 AI 모델들이 인간의 주석과 어느 정도 일치함을 보여줍니다.

2. **테이블 3**: 
   - 훈련된 주석자와 비훈련된 주석자 간의 Spearman 상관 계수를 보여줍니다. 
   - 비훈련된 주석자들의 평균 상관 계수는 0.389로, 이는 비훈련된 주석자들이 훈련된 주석자들과 유사한 성능을 보임을 나타냅니다.

3. **테이블 4**: 
   - 다양한 특성 조합의 Spearman 상관 계수를 보여줍니다. 
   - [F1, F2, F3, F4, F7] 조합이 가장 높은 성능을 보였으며, 이는 최적의 특성 조합으로 선택되었습니다.

4. **테이블 5**: 
   - 최적 모델 Moptimal의 Spearman 상관 계수는 0.460으로, 이는 모든 개별 특성 점수보다 유의미하게 높은 성능을 나타냅니다.

5. **테이블 6**: 
   - 훈련된 주석자 간의 Spearman 상관 계수를 보여줍니다. 
   - 평균 상관 계수는 0.453으로, 이는 최적 모델 Moptimal의 성능과 유사함을 나타냅니다.

### 결론
이 연구는 다양한 AI 기반 주석 방법과 전통적인 정보 검색 방법을 결합하여 최적의 주석 품질을 달성하는 방법을 탐구했습니다. 최적 모델은 인간 전문가와 유사한 수준의 주석 품질을 달성할 수 있음을 보여주며, 이는 대규모 생물 의학 응용 프로그램에서의 활용 가능성을 시사합니다.

---




1. **Table 1 and Table 2**: 
   - These tables present the Spearman correlation coefficients for each AI ranker in the training and test sets. 
   - The results indicate that AI-based methods achieved an average correlation of 0.336 in the training set and 0.350 in the test set, demonstrating a degree of alignment with human annotations.

2. **Table 3**: 
   - This table shows the Spearman correlation coefficients between trained and untrained annotators. 
   - The average correlation for untrained annotators was 0.389, indicating that untrained annotators performed similarly to trained annotators.

3. **Table 4**: 
   - This table displays the Spearman correlation coefficients for various feature combinations. 
   - The combination [F1, F2, F3, F4, F7] achieved the highest performance and was selected as the optimal feature set.

4. **Table 5**: 
   - The Spearman correlation coefficient for the optimal model Moptimal was 0.460, significantly outperforming all individual feature scores.

5. **Table 6**: 
   - This table presents the Spearman correlation coefficients among trained annotators. 
   - The average correlation was 0.453, indicating that the performance of the optimal model Moptimal is comparable to that of human experts.

### Conclusion
This study explored how to combine various AI-based annotation methods and traditional information retrieval methods to achieve optimal annotation quality. The optimal model demonstrated the potential to achieve annotation quality comparable to that of human experts, suggesting its applicability in large-scale biomedical applications.

<br/>
# refer format:
다음은 요청하신 논문의 BibTeX 형식과 시카고 스타일 인용입니다.

### BibTeX 형식
```bibtex
@inproceedings{Kim2026,
  author    = {Won G. Kim and Lana Yeganova and Shubo Tian and Donald C. Comeau and W. John Wilbur and Zhiyong Lu},
  title     = {Learning to Combine AI Annotations for Improved Biomedical Relevance Labeling},
  booktitle = {Proceedings of the 25th Workshop on Biomedical Language Processing (BioNLP 2026)},
  pages     = {502--507},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일 인용
Won G. Kim, Lana Yeganova, Shubo Tian, Donald C. Comeau, W. John Wilbur, and Zhiyong Lu. "Learning to Combine AI Annotations for Improved Biomedical Relevance Labeling." In *Proceedings of the 25th Workshop on Biomedical Language Processing (BioNLP 2026)*, 502–507. Association for Computational Linguistics, July 3-4, 2026.
