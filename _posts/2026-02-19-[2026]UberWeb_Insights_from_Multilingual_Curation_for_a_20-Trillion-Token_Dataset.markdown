---
layout: post
title:  "[2026]UberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset"
date:   2026-02-19 21:50:51 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 다국어 모델 훈련의 성능을 향상시키기 위해 데이터 품질을 개선하는 방법을 연구하였다.


짧은 요약(Abstract) :




이 논문은 다국어 모델의 훈련에서 데이터 품질과 구성의 중요성을 강조합니다. 다국어 모델을 훈련하는 데 있어 데이터의 가용성이 언어마다 불균형하게 분포되어 있어 고품질 다국어 모델을 훈련하는 데 어려움이 있습니다. 또한, 다국어 훈련에서 발생할 수 있는 성능 간섭 문제인 "다국어의 저주"에 대해서도 논의합니다. 연구 결과, 데이터 품질과 구성의 결함이 다국어 훈련의 성능 저하를 초래하는 주요 원인임을 보여주며, 특정 언어에 대한 데이터 품질을 개선하면 다른 언어의 성능도 향상된다는 것을 발견했습니다. 맞춤형 데이터 큐레이션을 통해 다국어 간섭을 완화하고, 컴퓨팅 효율성을 높일 수 있음을 입증했습니다. 이 연구는 20조 토큰으로 구성된 대규모 데이터셋을 활용하여 다국어 모델의 성능을 향상시키는 방법을 제시합니다.





This paper emphasizes the importance of data quality and composition in training multilingual models. The availability of data for training high-quality multilingual models is unevenly distributed across languages, making it challenging to train effective models. Additionally, the paper discusses the performance interference issue known as the "curse of multilinguality" that can arise during multilingual training. The findings reveal that deficiencies in data quality and composition are the primary causes of performance regressions in multilingual training, and improving data quality for any single language can benefit the performance of others. The study demonstrates that targeted data curation can mitigate multilingual interference and enhance compute efficiency. It presents methods for improving multilingual model performance using a large-scale dataset comprising 20 trillion tokens.


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



이 논문에서는 다국어 기초 모델 훈련을 위한 데이터 큐레이션의 중요성을 강조하고 있습니다. 연구팀은 20조 토큰 규모의 데이터셋을 활용하여 다국어 모델의 성능을 향상시키기 위한 여러 방법론을 제시합니다. 주요 방법론은 다음과 같습니다.

1. **모델 아키텍처**: 연구팀은 Llama 기반의 아키텍처를 사용하여 3B 및 8B 파라미터 모델을 훈련합니다. 이 아키텍처는 대규모 언어 모델 훈련에 최적화되어 있으며, 다양한 언어를 지원할 수 있는 능력을 가지고 있습니다.

2. **데이터 큐레이션**: 데이터의 품질을 높이기 위해 각 언어에 맞춤형 큐레이션 파이프라인을 개발합니다. 영어 데이터의 품질을 개선하면 다른 언어의 성능도 향상되는 상호작용을 발견하였으며, 이는 데이터 품질이 다국어 모델의 성능에 미치는 영향을 강조합니다. 연구팀은 영어와 비영어 데이터 모두에 대해 큐레이션을 수행하여 성능을 극대화합니다.

3. **훈련 데이터**: 연구팀은 공개 소스에서 수집한 데이터셋을 사용하여 훈련합니다. 영어 데이터는 DCLM, FineWeb, Nemotron CC v1 등을 활용하고, 비영어 데이터는 FineWeb2를 사용합니다. 이 데이터셋은 다양한 언어와 스크립트를 포함하고 있으며, 각 언어의 특성에 맞게 큐레이션됩니다.

4. **다국어 훈련 전략**: 연구팀은 다국어 훈련에서 발생할 수 있는 성능 간섭을 줄이기 위해, 다국어 데이터의 비율을 조정합니다. 훈련 과정에서 다국어 데이터의 비율을 점진적으로 증가시키는 다단계 데이터 커리큘럼을 적용하여, 각 언어의 성능을 최적화합니다.

5. **번역을 통한 데이터 증강**: 연구팀은 고품질 영어 문서를 비영어 언어로 번역하여 훈련 데이터의 양을 늘리는 방법을 사용합니다. 번역의 효과는 원본 데이터의 품질에 크게 의존하며, 고품질 문서를 번역할 때 성능 향상이 두드러집니다.

이러한 방법론을 통해 연구팀은 다국어 모델의 성능을 크게 향상시키고, 훈련 효율성을 높이며, 다국어 모델의 성능-계산 효율성을 재정의하는 데 성공했습니다.

---




This paper emphasizes the importance of data curation for training multilingual foundation models. The research team presents several methodologies to enhance the performance of multilingual models using a dataset of 20 trillion tokens. The key methodologies are as follows:

1. **Model Architecture**: The research team utilizes a Llama-based architecture to train 3B and 8B parameter models. This architecture is optimized for large-scale language model training and is capable of supporting various languages.

2. **Data Curation**: To improve data quality, tailored curation pipelines are developed for each language. The discovery that enhancing the quality of English data improves the performance of other languages highlights the impact of data quality on multilingual model performance. The team performs curation on both English and non-English data to maximize performance.

3. **Training Data**: The research team uses datasets collected from public sources for training. English data is sourced from DCLM, FineWeb, and Nemotron CC v1, while non-English data is sourced from FineWeb2. This dataset includes a variety of languages and scripts, and is curated according to the characteristics of each language.

4. **Multilingual Training Strategy**: To reduce performance interference that can arise in multilingual training, the team adjusts the proportion of multilingual data. They apply a multi-phase data curriculum that progressively increases the ratio of multilingual data during the training process to optimize the performance of each language.

5. **Translation for Data Augmentation**: The research team employs the method of translating high-quality English documents into non-English languages to increase the volume of training data. The effectiveness of translation heavily depends on the quality of the source data, with significant performance improvements observed when translating high-quality documents.

Through these methodologies, the research team successfully enhances the performance of multilingual models, increases training efficiency, and redefines the performance-compute efficiency of multilingual models.


<br/>
# Results



이 논문에서는 다국어 모델의 성능을 향상시키기 위한 데이터 큐레이션의 중요성을 강조하고 있습니다. 연구팀은 20조 토큰의 데이터셋을 사용하여 다국어 모델을 훈련시키고, 다양한 언어에서의 성능을 평가했습니다. 주요 결과는 다음과 같습니다.

1. **경쟁 모델과의 비교**: DatologyAI의 모델은 3B 및 8B 파라미터를 가진 모델로, 1T 토큰의 무작위 하위 집합에서 훈련되었습니다. 이 모델들은 강력한 공개 기준선 모델에 비해 4배에서 10배 더 적은 훈련 FLOPs로 경쟁력 있는 다국어 정확도를 달성했습니다. 예를 들어, DatologyAI의 3B 모델은 1.8 × 10^22 FLOPs로 훈련되어 LFM-2.5-1.2B 모델(1.2B 파라미터, 28T 토큰, 1.9 × 10^23 FLOPs)보다 더 나은 성능을 보였습니다.

2. **테스트 데이터**: 모델의 성능은 Multilingual MMLU, Multilingual ARC Challenge, Belebele와 같은 다양한 다국어 벤치마크를 통해 평가되었습니다. 이 평가들은 STEM, 인문학, 사회 과학 등 다양한 주제를 포함하고 있으며, 모델의 다국어 이해 및 추론 능력을 측정합니다.

3. **메트릭**: 성능 평가는 정확도(accuracy)와 오류율(error rate)로 측정되었습니다. 오류율은 로그 스케일로 표현되며, 낮은 값이 더 나은 성능을 나타냅니다. DatologyAI 모델은 기존의 공개 모델들에 비해 낮은 오류율을 기록하여 성능-계산 효율성의 새로운 경계를 설정했습니다.

4. **비교**: DatologyAI 모델은 LFM, SmolLM3, Qwen3와 같은 다른 모델들과 비교되었습니다. 이들 모델은 각각 12%에서 20%의 다국어 토큰을 사용했지만, DatologyAI는 전체 훈련 토큰의 7.75%만을 다국어 데이터에 할당하여도 경쟁력 있는 성능을 달성했습니다. 이는 데이터 큐레이션의 효율성을 강조합니다.

5. **결론**: 이 연구는 다국어 모델의 성능을 향상시키기 위해서는 단순히 모델의 크기를 늘리거나 훈련 토큰 수를 증가시키는 것이 아니라, 데이터의 품질과 큐레이션 전략이 중요하다는 것을 보여줍니다. 특히, 언어별 맞춤형 큐레이션이 다국어 성능을 극대화하는 데 필수적임을 강조합니다.





This paper emphasizes the importance of data curation in enhancing the performance of multilingual models. The research team trained multilingual models using a dataset of 20 trillion tokens and evaluated their performance across various languages. The key results are as follows:

1. **Comparison with Competing Models**: The DatologyAI models, with 3B and 8B parameters, were trained on a random subset of 1T tokens. These models achieved competitive multilingual accuracy with 4 to 10 times fewer training FLOPs compared to strong public baselines. For instance, the DatologyAI 3B model, trained with 1.8 × 10^22 FLOPs, outperformed the LFM-2.5-1.2B model (1.2B parameters, 28T tokens, 1.9 × 10^23 FLOPs).

2. **Test Data**: The performance of the models was evaluated using various multilingual benchmarks, including Multilingual MMLU, Multilingual ARC Challenge, and Belebele. These evaluations cover a wide range of subjects, including STEM, humanities, and social sciences, measuring the models' multilingual understanding and reasoning capabilities.

3. **Metrics**: Performance was measured using accuracy and error rate. The error rate is expressed on a logarithmic scale, where lower values indicate better performance. The DatologyAI models achieved lower error rates compared to existing public models, establishing a new frontier in performance-compute efficiency.

4. **Comparison**: DatologyAI models were compared with other models such as LFM, SmolLM3, and Qwen3, which used 12% to 20% multilingual tokens. In contrast, DatologyAI allocated only 7.75% of the total training tokens to multilingual data while still achieving competitive performance, highlighting the efficiency of data curation.

5. **Conclusion**: This study demonstrates that improving multilingual model performance requires not just increasing model size or the number of training tokens, but also focusing on data quality and curation strategies. It emphasizes that tailored per-language curation is essential for maximizing multilingual performance.


<br/>
# 예제



이 논문에서는 다국어 모델 훈련을 위한 데이터 큐레이션의 중요성을 강조하고 있습니다. 연구팀은 13개 언어에 걸쳐 다국어 데이터 큐레이션을 수행하여, 데이터 품질이 모델 성능에 미치는 영향을 분석했습니다. 특히, 영어 데이터의 품질을 개선하면 다른 언어의 성능도 향상된다는 것을 발견했습니다. 

#### 예시: 데이터 큐레이션 및 모델 훈련

1. **트레이닝 데이터**: 
   - 영어 데이터: 고품질의 영어 문서 60억 토큰
   - 비영어 데이터: 스페인어, 독일어, 프랑스어 등 12개 언어의 비영어 문서 60억 토큰

2. **테스트 데이터**: 
   - 다국어 MMLU, ARC 챌린지, Belebele와 같은 벤치마크 데이터셋을 사용하여 모델 성능을 평가합니다.

3. **구체적인 작업(Task)**:
   - 모델은 주어진 질문에 대해 다국어로 답변을 생성하는 작업을 수행합니다. 예를 들어, "지구의 대기 구성은 무엇인가요?"라는 질문에 대해 영어로는 "The Earth's atmosphere is composed of nitrogen, oxygen, argon, and other gases."라고 답변하고, 스페인어로는 "La atmósfera de la Tierra está compuesta por nitrógeno, oxígeno, argón y otros gases."라고 답변합니다.

4. **모델 훈련**:
   - 3B 및 8B 파라미터 모델을 사용하여 1T 토큰의 랜덤 서브셋으로 훈련합니다. 이 과정에서 영어 데이터의 품질을 개선하면, 비영어 언어의 성능이 평균 3.91% 향상되는 결과를 얻었습니다.

5. **결과 평가**:
   - 모델의 성능은 다국어 MMLU, ARC 챌린지, Belebele와 같은 벤치마크를 통해 평가되며, 각 언어별로 성능을 비교합니다. 예를 들어, 영어 데이터의 품질을 개선한 경우, 스페인어와 독일어의 성능이 각각 8.56% 및 3.94% 향상되었습니다.

이러한 방식으로, 데이터 큐레이션이 다국어 모델의 성능을 향상시키는 데 중요한 역할을 한다는 것을 보여줍니다.

---




This paper emphasizes the importance of data curation for training multilingual models. The research team conducted multilingual data curation across 13 languages to analyze the impact of data quality on model performance. Notably, they found that improving the quality of English data also enhances the performance of other languages.

#### Example: Data Curation and Model Training

1. **Training Data**: 
   - English Data: High-quality English documents totaling 6 billion tokens.
   - Non-English Data: Non-English documents in 12 languages, such as Spanish, German, and French, also totaling 6 billion tokens.

2. **Test Data**: 
   - Benchmark datasets such as Multilingual MMLU, ARC Challenge, and Belebele are used to evaluate model performance.

3. **Specific Task**:
   - The model performs the task of generating answers in multiple languages to given questions. For example, for the question "What is the composition of the Earth's atmosphere?", the model responds in English with "The Earth's atmosphere is composed of nitrogen, oxygen, argon, and other gases." and in Spanish with "La atmósfera de la Tierra está compuesta por nitrógeno, oxígeno, argón y otros gases."

4. **Model Training**:
   - Models with 3B and 8B parameters are trained on a random subset of 1 trillion tokens. During this process, improving the quality of English data resulted in an average performance increase of 3.91% for non-English languages.

5. **Result Evaluation**:
   - The model's performance is evaluated using benchmarks like Multilingual MMLU, ARC Challenge, and Belebele, comparing performance across languages. For instance, improving the quality of English data led to performance increases of 8.56% for Spanish and 3.94% for German.

Through this approach, the paper demonstrates that data curation plays a critical role in enhancing the performance of multilingual models.

<br/>
# 요약   


이 논문에서는 다국어 모델 훈련의 성능을 향상시키기 위해 데이터 품질을 개선하는 방법을 연구하였다. 실험 결과, 영어 데이터의 품질을 높이면 다른 언어의 성능도 향상되며, 맞춤형 데이터 큐레이션이 최적의 성능을 달성하는 데 필수적임을 보여주었다. 예를 들어, 맞춤형 큐레이션을 통해 3B 모델에서 16.87%의 성능 향상을 달성하였다.

---

This paper investigates methods to improve the performance of multilingual models by enhancing data quality. The results show that improving the quality of English data also boosts performance in other languages, and tailored data curation is essential for achieving optimal performance. For instance, bespoke curation led to a 16.87% performance improvement in a 3B model.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1**: 이 다이어그램은 DatologyAI 모델이 기존의 오픈 소스 모델들과 비교하여 성능-계산 효율성의 새로운 경계를 설정했음을 보여줍니다. DatologyAI 모델은 더 적은 계산 비용으로 더 낮은 오류율을 달성하여 멀티링구얼 성능을 향상시켰습니다.
  
- **Figure 2**: 영어 데이터 품질 개선이 비영어 언어의 성능에 미치는 영향을 보여줍니다. 영어 데이터의 품질을 높이면 12개 언어 중 11개 언어에서 성능이 향상되었으며, 평균적으로 3.91%의 상대적 개선이 있었습니다.

- **Figure 3**: 언어 유사성과 영어 데이터 품질 개선의 상관관계를 나타냅니다. 언어 유사성이 높을수록 영어 데이터 품질 개선의 혜택이 더 크게 나타났습니다.

- **Figure 5**: 번역의 효과를 보여주는 그래프입니다. 고품질 영어 문서를 번역할 때 성능이 크게 향상되며, 무작위로 선택된 영어 데이터의 번역보다 더 나은 결과를 보여줍니다.

#### 2. 테이블
- **Table 1**: 연구에 포함된 13개 비영어 언어의 자원 수준과 문서 수를 보여줍니다. 자원 수준이 높은 언어(예: 스페인어)와 낮은 언어(예: 힌디어) 간의 차이를 강조합니다.

- **Table 4-16**: 각 언어별로 MMLU, ARC, Belebele 평가에서의 성능을 보여줍니다. DatologyAI 모델은 다양한 언어에서 경쟁력 있는 성능을 보이며, 특히 영어와 비영어 데이터의 품질 개선이 서로에게 긍정적인 영향을 미친다는 것을 보여줍니다.

#### 3. 어펜딕스
- **Appendix A.1**: 각 언어별 평가 데이터셋을 정리하여, 어떤 데이터셋이 사용되었는지를 명확히 합니다. 이는 평가의 신뢰성을 높이는 데 기여합니다.

- **Appendix A.3**: 각 모델의 언어별 성능과 훈련 계산량을 비교하여, DatologyAI 모델이 다른 모델들에 비해 데이터 효율성이 뛰어남을 보여줍니다.




#### 1. Diagrams and Figures
- **Figure 1**: This diagram illustrates that DatologyAI models have established a new frontier in performance-compute efficiency compared to existing open-source models. DatologyAI models achieve lower error rates with less computational cost, enhancing multilingual performance.

- **Figure 2**: It shows the impact of improving English data quality on the performance of non-English languages. Enhancing the quality of English data resulted in performance improvements in 11 out of 12 languages, with an average relative gain of 3.91%.

- **Figure 3**: This figure depicts the correlation between language similarity and the benefits of improving English data quality. The closer the linguistic similarity, the greater the benefits from enhancing English data quality.

- **Figure 5**: It demonstrates the effectiveness of translation. Translating high-quality English documents leads to significant performance improvements compared to translating randomly selected English data.

#### 2. Tables
- **Table 1**: It presents the resource levels and document counts for the 13 non-English languages included in the study, highlighting the differences between high-resource languages (e.g., Spanish) and low-resource languages (e.g., Hindi).

- **Tables 4-16**: These tables show the performance of each language on MMLU, ARC, and Belebele evaluations. DatologyAI models demonstrate competitive performance across various languages, indicating that improvements in the quality of both English and non-English data positively affect each other.

#### 3. Appendix
- **Appendix A.1**: It organizes the evaluation datasets used for each language, clarifying which datasets were utilized, thus contributing to the reliability of the evaluations.

- **Appendix A.3**: It compares the performance and training compute for each model by language, showing that DatologyAI models exhibit superior data efficiency compared to other models.

<br/>
# refer format:  


### BibTeX   


```bibtex
@article{datologyai2026,
  title={UberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset},
  author={DatologyAI Team},
  year={2026},
  journal={arXiv preprint arXiv:2602.15210},
  url={https://arxiv.org/abs/2602.15210}
}
```

### 시카고 스타일

DatologyAI Team. 2026. "UberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset." arXiv preprint arXiv:2602.15210. https://arxiv.org/abs/2602.15210.
