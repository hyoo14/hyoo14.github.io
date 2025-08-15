---
layout: post
title:  "[2025]Experiential Semantic Information and Brain Alignment: Are Multimodal Models Better than Language Models?"
date:   2025-08-15 19:28:34 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 언어 모델과 멀티모달 모델(오디오)의 단어 표현을 비교하여 경험적 의미 정보를 얼마나 잘 포착하는지를 평가(뇌MRI와 잘 일맥상통하나)    


짧은 요약(Abstract) :

이 논문의 초록에서는 다중 모달 모델이 언어 전용 모델보다 더 풍부하고 인간과 유사한 텍스트 표현을 학습한다는 일반적인 가정에 대해 논의하고 있습니다. 이러한 모델은 이미지나 오디오에 기반하여 학습되며, 인간 언어가 실제 경험에 기반을 두고 있다는 점에서 유사하다고 주장됩니다. 그러나 이러한 주장을 뒷받침하는 실증적 연구는 부족합니다. 본 연구에서는 대조적 다중 모달 모델과 언어 전용 모델의 단어 표현을 비교하여 경험적 정보를 얼마나 잘 포착하고 인간의 fMRI 반응과 얼마나 잘 일치하는지를 조사했습니다. 결과적으로, 언어 전용 모델이 두 가지 측면 모두에서 다중 모달 모델보다 우수하다는 것을 발견했습니다. 또한, 언어 전용 모델은 경험적 모델과 공유되는 정보 외에도 더 독특한 뇌 관련 의미 정보를 학습하는 것으로 나타났습니다. 전반적으로 이 연구는 다중 모달 데이터 소스가 제공하는 보완적인 의미 정보를 더 잘 통합하는 계산 모델을 개발할 필요성을 강조합니다.


The abstract of this paper discusses the common assumption that multimodal models learn richer and more human-like text representations than language-only models, as they are grounded in images or audio—similar to how human language is grounded in real-world experiences. However, empirical studies supporting this claim are largely lacking. This study addresses this gap by comparing word representations from contrastive multimodal models versus language-only models in terms of how well they capture experiential information and align with human fMRI responses. Surprisingly, the results indicate that language-only models outperform multimodal ones in both respects. Additionally, they learn more unique brain-relevant semantic information beyond what is shared with the experiential model. Overall, the study highlights the need to develop computational models that better integrate the complementary semantic information provided by multimodal data sources.


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


이 연구에서는 언어 모델과 멀티모달 모델 간의 비교를 통해 경험적 의미 정보를 포착하고 뇌 반응과의 정렬을 평가하기 위해 여러 가지 방법론을 사용했습니다. 연구의 주요 방법론은 다음과 같습니다.

1. **모델 선택**: 연구에서는 언어 전용 모델(SimCSE, BERT)과 멀티모달 모델(MCSE, VisualBERT)을 비교했습니다. 모든 모델은 BERT 아키텍처를 기반으로 하며, 언어 인코더로 사전 훈련된 BERT를 사용했습니다. SimCSE는 언어 전용 문장 인코더로, 100만 개의 문장으로 대조적으로 미세 조정되었습니다. MCSE는 비전-언어 문장 인코더로, SimCSE 목표와 CLIP과 유사한 목표를 동시에 최적화하여 훈련되었습니다.

2. **데이터 및 경험적 모델**: 연구에서는 320개의 명사를 사용하여 fMRI 반응과 경험적 모델(EXP48)과의 정렬을 평가했습니다. EXP48 모델은 48개의 차원으로 구성된 경험적 정보를 포착하는 모델로, 각 단어는 사람들의 경험에 대한 평가로 표현됩니다. fMRI 데이터는 36명의 참가자로부터 수집되었으며, 각 단어에 대해 그들이 일상에서 얼마나 자주 경험하는지를 평가했습니다.

3. **표현 추출**: 단어 표현은 각 모델의 숨겨진 상태에서 추출되었습니다. 단어 자극을 문장 템플릿에 포함시켜 모델에 전달함으로써, 단어의 맥락화된 표현을 얻었습니다. 이 과정에서 단어의 표현은 각 템플릿에 대해 평균화되었습니다.

4. **정렬 평가**: 모델 표현과 EXP48 및 fMRI 반응 간의 정렬을 비교하기 위해 표현 유사성 분석(RSA)을 사용했습니다. RSA는 두 표현 공간 간의 정렬을 정량화하는 방법으로, 표현 불일치 행렬(RDM)을 사용하여 모델 표현과 뇌 반응 간의 상관관계를 계산했습니다.

5. **부분 상관 분석**: 모델의 뇌 정렬이 EXP48과 공유하는 정보의 양을 평가하기 위해 부분 상관 분석을 수행했습니다. 이를 통해 각 모델이 EXP48과 공유하는 정보가 뇌 반응에 얼마나 기여하는지를 확인했습니다.

이러한 방법론을 통해 연구는 멀티모달 모델이 언어 전용 모델보다 경험적 의미 정보를 덜 포착하고 뇌 반응과의 정렬이 낮다는 결과를 도출했습니다.

---


In this study, several methodologies were employed to compare language models and multimodal models in terms of capturing experiential semantic information and evaluating alignment with brain responses. The main methodologies are as follows:

1. **Model Selection**: The study compared language-only models (SimCSE, BERT) with multimodal models (MCSE, VisualBERT). All models are based on the BERT architecture and utilize a pre-trained BERT as the language encoder. SimCSE is a language-only sentence encoder fine-tuned contrastively on 1 million sentences. MCSE is a vision-language sentence encoder that was trained by jointly optimizing a SimCSE objective and a CLIP-like objective.

2. **Data and Experiential Model**: The study utilized 320 nouns to evaluate the alignment with fMRI responses and the experiential model (EXP48). The EXP48 model captures experiential information across 48 dimensions, where each word is represented by human ratings of their experiences. fMRI data were collected from 36 participants, who rated how frequently they experienced the corresponding entities in daily life.

3. **Extracting Representations**: Word representations were extracted from the hidden states of each model. The word stimuli were embedded in sentence templates to obtain contextualized representations from the models. In this process, the representations of the words were averaged across the templates.

4. **Alignment Evaluation**: Representational Similarity Analysis (RSA) was used to compare the alignment between model representations and EXP48 as well as fMRI responses. RSA quantifies the alignment between two representational spaces by calculating the correlation between representational dissimilarity matrices (RDMs).

5. **Partial Correlation Analysis**: To assess how much of the brain alignment of the models is attributable to the information shared with EXP48, partial correlation analysis was performed. This allowed the researchers to determine how much of the brain-relevant information each model contributes independently of EXP48.

Through these methodologies, the study found that multimodal models capture less experiential semantic information and have lower alignment with brain responses compared to language-only models.


<br/>
# Results


이 연구에서는 언어 모델과 멀티모달 모델 간의 경험적 의미 정보와 뇌 반응의 정렬을 비교하여, 멀티모달 모델이 언어 모델보다 더 나은 성능을 보이는지 여부를 조사했습니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구에서는 세 가지 모델을 비교했습니다. 언어 전용 모델인 SimCSE와 BERT, 그리고 멀티모달 모델인 MCSE(비전-언어 모델)와 VisualBERT(비전-언어 모델)입니다. 또한, CLAP(오디오-언어 모델)도 포함되었으나, 이 모델은 통계적으로 유의미한 결과를 보이지 않았습니다.

2. **테스트 데이터**: 실험에 사용된 단어 자극은 320개의 명사로 구성되어 있으며, 이 중 절반은 객체를, 나머지 절반은 사건을 나타냅니다. fMRI 반응은 36명의 참가자로부터 수집되었으며, 각 단어 자극에 대해 참가자들이 일상 생활에서 얼마나 자주 해당 개체를 경험했는지를 평가했습니다.

3. **메트릭**: 모델의 성능은 두 가지 주요 메트릭을 통해 평가되었습니다. 첫 번째는 EXP48 모델과의 정렬을 측정하는 Spearman 상관계수이며, 두 번째는 fMRI 반응과의 정렬을 측정하는 Spearman 상관계수입니다. 이 두 메트릭을 통해 각 모델이 경험적 정보와 뇌 반응을 얼마나 잘 반영하는지를 평가했습니다.

4. **비교 결과**: 연구 결과에 따르면, 언어 전용 모델(BERT와 SimCSE)은 멀티모달 모델(MCSE와 VisualBERT)보다 EXP48 모델 및 fMRI 반응과 더 높은 정렬을 보였습니다. 특히 BERT는 EXP48과의 정렬에서 0.53의 상관계수를 기록하며 가장 높은 성능을 보였고, SimCSE는 0.52, MCSE는 0.45, VisualBERT는 0.27의 상관계수를 기록했습니다. 또한, 언어 전용 모델은 뇌 반응과의 정렬에서도 더 높은 성능을 보였으며, MCSE와 VisualBERT는 상대적으로 낮은 정렬을 보였습니다.

5. **결론**: 이 연구는 멀티모달 모델이 언어 전용 모델보다 더 많은 경험적 의미 정보를 학습하지 못한다는 점을 강조합니다. 이는 멀티모달 데이터가 언어 모델의 성능을 향상시키는 데 기여하지 못할 수 있음을 시사합니다. 따라서, 향후 연구에서는 멀티모달 모델이 뇌와의 정렬을 개선할 수 있는 방법을 모색해야 할 필요성이 있습니다.

---




This study investigates the empirical semantic information and brain response alignment between language models and multimodal models to determine whether multimodal models outperform language-only models. The main results of the study are as follows:

1. **Competing Models**: The study compared three models: the language-only models SimCSE and BERT, and the multimodal models MCSE (vision-language model) and VisualBERT (vision-language model). Additionally, CLAP (audio-language model) was included, but this model did not yield statistically significant results.

2. **Test Data**: The word stimuli used in the experiments consisted of 320 nouns, half of which referred to objects and the other half to events. fMRI responses were collected from 36 participants, who rated how frequently they experienced the corresponding entities in daily life.

3. **Metrics**: The performance of the models was evaluated using two main metrics. The first was the Spearman correlation coefficient measuring alignment with the EXP48 model, and the second was the Spearman correlation coefficient measuring alignment with fMRI responses. These two metrics assessed how well each model reflected experiential information and brain responses.

4. **Comparison Results**: The results indicated that the language-only models (BERT and SimCSE) exhibited higher alignment with both the EXP48 model and fMRI responses compared to the multimodal models (MCSE and VisualBERT). Specifically, BERT achieved the highest performance with a correlation of 0.53 with EXP48, followed by SimCSE at 0.52, MCSE at 0.45, and VisualBERT at 0.27. Furthermore, the language-only models demonstrated better alignment with brain responses, while MCSE and VisualBERT showed relatively lower alignment.

5. **Conclusion**: This study highlights that multimodal models do not learn more experiential semantic information than language-only models. This suggests that multimodal data may not contribute to enhancing the performance of language models. Therefore, future research should explore ways to improve the alignment of multimodal models with brain responses.


<br/>
# 예제



이 논문에서는 언어 모델과 멀티모달 모델의 성능을 비교하기 위해 실험을 설계했습니다. 실험의 주요 목표는 두 가지 모델이 어떻게 경험적 의미 정보를 포착하고 인간의 뇌 반응과 얼마나 잘 정렬되는지를 평가하는 것이었습니다. 

#### 1. 데이터셋
- **단어 자극**: 320개의 명사로 구성되어 있으며, 이 중 절반은 사물(예: '음식', '차량', '동물', '도구')을, 나머지 절반은 사건(예: '사회적 사건', '부정적 사건', '소리', '의사소통')을 나타냅니다.
- **fMRI 반응**: 36명의 참가자가 각 단어 자극을 보면서 그 단어와 관련된 경험의 빈도를 평가했습니다. 이 과정에서 뇌의 반응을 측정하기 위해 fMRI를 사용했습니다.

#### 2. 모델
- **언어 전용 모델 (SimCSE)**: 100만 개의 문장에서 대조 학습을 통해 미세 조정된 문장 인코더입니다. 입력으로는 문장이 사용되며, 출력은 문장 임베딩입니다.
- **멀티모달 모델 (MCSE)**: 이미지-텍스트 쌍을 사용하여 대조 학습을 통해 미세 조정된 비전-언어 문장 인코더입니다. 입력으로는 이미지와 해당 캡션이 사용되며, 출력은 이미지와 텍스트의 정렬된 임베딩입니다.

#### 3. 실험 절차
- **입력**: 각 모델에 대해 단어 자극을 포함한 문장 템플릿을 사용하여 입력을 생성했습니다. 예를 들어, "Someone mentioned the <word>."와 같은 문장 구조를 사용했습니다.
- **출력**: 각 모델은 단어 자극에 대한 임베딩을 생성하며, 이 임베딩은 모델의 숨겨진 상태에서 추출됩니다. 

#### 4. 평가 방법
- **표현 유사성 분석 (RSA)**: 모델의 표현과 EXP48 모델 및 fMRI 반응 간의 정렬을 평가하기 위해 표현 유사성 분석을 사용했습니다. 이 분석은 각 모델의 표현 간의 쌍별 코사인 거리로 구성된 표현 불일치 행렬(RDM)을 생성하고, 이를 통해 상관관계를 계산합니다.

이러한 실험을 통해 언어 전용 모델이 멀티모달 모델보다 더 많은 뇌 관련 정보를 포착하고, 경험적 의미 정보를 더 잘 반영한다는 결과를 도출했습니다.

---




In this paper, an experiment was designed to compare the performance of language models and multimodal models. The main goal of the experiment was to evaluate how well the two models capture experiential semantic information and align with human brain responses.

#### 1. Dataset
- **Word Stimuli**: Composed of 320 nouns, half of which refer to objects (e.g., 'food', 'vehicles', 'animals', 'tools'), and the other half refer to events (e.g., 'social event', 'negative event', 'sound', 'communication').
- **fMRI Responses**: 36 participants rated the frequency with which they experienced the entities corresponding to each word stimulus while viewing them. fMRI was used to measure brain responses during this process.

#### 2. Models
- **Language-Only Model (SimCSE)**: A sentence encoder fine-tuned with contrastive learning on 1 million sentences. The input consists of sentences, and the output is sentence embeddings.
- **Multimodal Model (MCSE)**: A vision-language sentence encoder fine-tuned with contrastive learning using image-text pairs. The input consists of images and their corresponding captions, and the output is aligned embeddings of images and text.

#### 3. Experimental Procedure
- **Input**: Sentence templates containing the word stimuli were used to generate inputs for each model. For example, a sentence structure like "Someone mentioned the <word>." was employed.
- **Output**: Each model generates embeddings for the word stimuli, which are extracted from the hidden states of the model.

#### 4. Evaluation Method
- **Representational Similarity Analysis (RSA)**: RSA was used to evaluate the alignment between the model representations and the EXP48 model as well as the fMRI responses. This analysis generates representational dissimilarity matrices (RDMs) based on pairwise cosine distances between the representations of each model, allowing for correlation calculations.

Through these experiments, it was found that language-only models capture more brain-relevant information and better reflect experiential semantic information than multimodal models.

<br/>
# 요약
이 연구에서는 언어 모델과 멀티모달 모델의 단어 표현을 비교하여 경험적 의미 정보를 얼마나 잘 포착하는지를 평가하였다. 실험 결과, 언어 전용 모델이 멀티모달 모델보다 뇌 반응과 더 높은 정렬을 보였으며, 경험적 모델과의 정렬에서도 우수한 성능을 나타냈다. 이러한 결과는 멀티모달 모델이 언어 전용 모델보다 더 많은 의미 정보를 학습하지 못한다는 것을 시사한다.

---

This study evaluated how well language models and multimodal models capture experiential semantic information by comparing their word representations. The results indicated that language-only models exhibited higher alignment with brain responses and performed better in alignment with the experiential model than multimodal models. These findings suggest that multimodal models do not learn more semantic information than language-only models.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 실험 설정의 개요를 보여줍니다. 단어 자극, fMRI 반응, 그리고 경험적 모델(EXP48)에서 파생된 표현 간의 관계를 시각화하여, 각 모델의 표현이 어떻게 상호작용하는지를 설명합니다.
   - **Figure 2**: 모델 표현과 EXP48 및 fMRI 반응 간의 정량적 정합성을 보여주는 Spearman 상관관계를 나타냅니다. 언어 전용 모델(BERT, SimCSE)이 다중 모달 모델(MCSE, VisualBERT)보다 더 높은 정합성을 보이는 경향이 있음을 보여줍니다.
   - **Figure 3**: 부분 상관 분석 결과를 보여줍니다. EXP48의 정보를 제거한 후 모델의 뇌 정합성을 평가하여, 언어 전용 모델이 더 많은 독립적인 기여를 한다는 것을 나타냅니다.
   - **Figure 4**: 캡션 유사 템플릿을 사용했을 때의 모델 표현과 fMRI 반응 간의 정합성을 보여줍니다. 모든 모델이 캡션 유사 템플릿을 사용할 때 더 높은 정합성을 보이며, 이는 템플릿의 구조가 모든 모델에 더 적합하다는 것을 시사합니다.
   - **Figure 5**: 객체와 사건 단어에 대한 모델 표현의 정합성을 비교합니다. 사건 단어에 대한 정합성이 객체 단어보다 더 높다는 것을 보여주며, 이는 사건 개념의 쌍 유사성이 더 높은 변동성을 가진다는 것을 나타냅니다.

2. **테이블**
   - **Table 1**: 다양한 모델의 EXP48 및 뇌 반응과의 정합성을 보여주는 Spearman 상관관계를 정리합니다. 언어 전용 모델이 다중 모달 모델보다 더 높은 정합성을 보이며, CLAP 모델은 유의미한 정합성을 보이지 않습니다.

3. **어펜딕스**
   - **Appendix A**: 단어 자극을 포함한 문장 템플릿의 목록을 제공합니다. 이 템플릿들은 모델이 단어 표현을 추출하는 데 사용되었습니다.
   - **Appendix B**: 추가 RSA 결과를 포함하여, 단어 표현을 단독으로 사용하는 것과 문장 템플릿을 사용하는 것의 차이를 보여줍니다. 문장 템플릿을 사용할 때 더 높은 정합성을 보이는 경향이 있음을 확인합니다.




1. **Diagrams and Figures**
   - **Figure 1**: Provides an overview of the experimental setup, visualizing the relationships between word stimuli, fMRI responses, and the experiential model (EXP48), explaining how the representations of each model interact.
   - **Figure 2**: Displays the Spearman correlations quantifying the alignment between model representations and EXP48 and fMRI responses. It shows a tendency for language-only models (BERT, SimCSE) to have higher alignment than multimodal models (MCSE, VisualBERT).
   - **Figure 3**: Shows the results of the partial correlation analysis, evaluating the brain alignment of models after regressing out the information from EXP48, indicating that language-only models contribute more independent information.
   - **Figure 4**: Illustrates the alignment between model representations and fMRI responses when using caption-like templates. All models show higher alignment with these templates, suggesting that the structure of the templates is more suitable for all models.
   - **Figure 5**: Compares the alignment of model representations for object and event words. It shows that event words have higher alignment than object words, indicating that event concepts exhibit greater variability in pairwise similarities.

2. **Table**
   - **Table 1**: Summarizes the Spearman correlations of various models with EXP48 and brain responses. It highlights that language-only models exhibit higher alignment than multimodal models, with the CLAP model showing no significant alignment.

3. **Appendix**
   - **Appendix A**: Lists the sentence templates used to embed word stimuli for extracting contextualized representations from the models.
   - **Appendix B**: Includes additional RSA results, demonstrating the difference in alignment when using single words versus sentence templates. It confirms that using sentence templates yields higher alignment.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{bavaresco2025experiential,
  title={Experiential Semantic Information and Brain Alignment: Are Multimodal Models Better than Language Models?},
  author={Anna Bavaresco and Raquel Fernández},
  booktitle={Proceedings of the 29th Conference on Computational Natural Language Learning},
  pages={141--155},
  year={2025},
  publisher={Association for Computational Linguistics},
  address={Amsterdam, Netherlands},
  url={https://github.com/dmg-illc/exp-info-models-brain}
}
```

### 시카고 스타일

Bavaresco, Anna, and Raquel Fernández. 2025. "Experiential Semantic Information and Brain Alignment: Are Multimodal Models Better than Language Models?" In *Proceedings of the 29th Conference on Computational Natural Language Learning*, 141–155. Amsterdam, Netherlands: Association for Computational Linguistics. https://github.com/dmg-illc/exp-info-models-brain.
