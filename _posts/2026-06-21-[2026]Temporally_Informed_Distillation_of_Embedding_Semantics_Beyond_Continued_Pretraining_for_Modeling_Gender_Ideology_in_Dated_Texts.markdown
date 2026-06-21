---
layout: post
title:  "[2026]Temporally Informed Distillation of Embedding Semantics: Beyond Continued Pretraining for Modeling Gender Ideology in Dated Texts"
date:   2026-06-21 08:23:00 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Temporally Informed Distillation of Embedding Semantics (TIDES)라는 방법론을 제안하여, 역사적 텍스트에서 성 이데올로기를 모델링하기 위해 대형 임베딩 모델의 지식을 소형 모델로 전이하는 과정을 설명합니다.


짧은 요약(Abstract) :


이 논문의 초록에서는 역사적으로 위치한 성 이데올로기를 모델링하는 것이 언어 모델에게 여전히 도전 과제가 되고 있음을 설명합니다. 현대의 임베딩은 표면적인 어휘 패턴을 넘어서는 시간적으로 특정한 의미 구조를 반영하는 데 어려움을 겪고 있습니다. 대형 언어 모델은 광범위한 일반 목적 성능을 보이지만, 역사적 언어 패턴과 현대 훈련 데이터 간의 분포 불일치로 인해 역사적 의미 분석에 직접 사용하기에는 한계가 있습니다. 이러한 문제를 해결하기 위해, 저자들은 Temporally Informed Distillation of Embedding Semantics (TIDES)를 제안합니다. TIDES는 시간적으로 특정한 말뭉치에 대한 지속적인 재훈련과 대형 임베딩 모델로부터의 특징 수준의 증류를 통합합니다. 저자들은 TIDES를 다양한 교사 아키텍처에서 평가하며, 지속적인 재훈련이 어휘 및 구문 적응을 제공하지만, 이데올로기 모델링의 개선은 추가적인 훈련 노출만으로는 설명될 수 없음을 보여줍니다. 오히려 교사-학생 구조적 정렬이 효과적인 전이의 핵심이라는 점을 강조합니다. 대조적이고 인코더 정렬된 교사는 역사적으로 위치한 미세한 의미 구분을 보다 안정적으로 보존하는 데 기여합니다. 이러한 발견은 시간적 이데올로기 전이가 표현 의존적임을 시사하며, 이데올로기적 의미는 임베딩 공간의 기하학과 훈련 목표에 의해 형성될 수 있음을 보여줍니다. TIDES를 도입하고 아키텍처 호환성이 이데올로기 상속에 영향을 미칠 수 있다는 증거를 제공함으로써, 이 연구는 시간적으로 기반한 의미 연구에서 이데올로기를 모델링하는 데 있어 표현 중심의 접근 방식을 발전시킵니다.



The abstract of this paper explains that modeling historically situated gender ideology remains a challenge for language models, as contemporary embeddings struggle to reflect temporally specific semantic structures beyond surface lexical patterns. Although large language models exhibit extensive general-purpose performance, their direct use in history-specific semantic analysis is limited by the distributional mismatch between contemporary training data and historical linguistic patterns. To address this issue, the authors propose Temporally Informed Distillation of Embedding Semantics (TIDES), which integrates continued pretraining on temporally specific corpora with feature-level distillation from large embedding teachers. The authors evaluate TIDES across various teacher architectures, showing that while continued pretraining provides lexical and syntactic adaptation, improvements in ideological modeling cannot be attributed to additional training exposure alone. Rather, teacher-student structural alignment is critical to transfer effectiveness. Contrastive, encoder-aligned teachers yield substantially more stable preservation of fine-grained, historically situated semantic distinctions. These findings suggest that temporal ideology transfer is representation-dependent: ideological meaning can be shaped by the geometry and training objectives of embedding spaces. By introducing TIDES and providing evidence that architectural compatibility can influence ideological inheritance, this study advances a representation-centered account of modeling ideology in temporally grounded semantic research.


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


이 논문에서는 Temporally Informed Distillation of Embedding Semantics (TIDES)라는 새로운 방법론을 제안합니다. TIDES는 역사적으로 특정한 성 이데올로기를 모델링하기 위해 대규모 임베딩 모델에서 소형 BERT 기반 학생 모델로 지식을 전이하는 프레임워크입니다. 이 방법은 두 가지 주요 모듈로 구성됩니다: 역사적 기초 모듈과 특징 수준의 증류 모듈입니다.

1. **모델 아키텍처**:
   - TIDES는 BERT(Bidirectional Encoder Representations from Transformers) 아키텍처를 기반으로 하며, 대형 교사 모델(예: bge-large-en, gte-large-en-v1.5)에서 학습된 지식을 소형 학생 모델(bert-base-uncased)로 전이합니다. 
   - 교사 모델은 24개의 층과 1024의 숨겨진 크기를 가지며, 학생 모델은 12개의 층과 768의 숨겨진 크기를 가집니다. 이러한 구조적 차이는 학생 모델이 더 적은 파라미터로도 효과적으로 학습할 수 있도록 합니다.

2. **트레이닝 데이터**:
   - TIDES는 1990년대의 역사적 텍스트를 포함하는 COHA(Corpus of Historical American English) 데이터를 사용하여 학생 모델을 훈련합니다. 이 데이터는 신문, 잡지, 소설 및 비소설을 포함하여 해당 시대의 언어 환경을 반영합니다.
   - 학생 모델은 마스크드 언어 모델링(MLM) 기법을 사용하여 역사적 데이터에 적응합니다. 이 과정에서 15%의 토큰이 무작위로 마스킹되어 모델이 원래의 토큰을 복원하도록 훈련됩니다.

3. **특징 수준의 증류**:
   - TIDES는 교사 모델과 학생 모델 간의 내부 표현을 정렬하여 지식을 전이합니다. 이 과정은 세 가지 주요 손실 함수로 구성됩니다: 
     - **임베딩 손실**: 입력 층에서의 임베딩 정렬.
     - **숨겨진 상태 손실**: 중간 층에서의 숨겨진 상태 정렬.
     - **주의 손실**: 주의 메커니즘의 정렬.
   - 이러한 손실 함수들은 학생 모델이 교사 모델의 구조적 및 이데올로기적 표현을 내재화하도록 돕습니다.

4. **훈련 과정**:
   - TIDES는 두 가지 손실 함수를 결합하여 훈련합니다: MLM 손실과 특징 수준의 손실. 이 두 손실의 가중치는 α로 조정되며, α=0.2로 설정하여 특징 정렬을 강조합니다.
   - 훈련은 AdamW 옵티마이저를 사용하여 수행되며, 학습률은 5×10^-5로 설정됩니다. 훈련은 최대 50 에포크 동안 진행되며, 배치 크기는 32로 설정됩니다.

이러한 방법론을 통해 TIDES는 역사적 성 이데올로기를 효과적으로 모델링할 수 있는 가능성을 보여줍니다. TIDES는 단순히 데이터에 의존하는 것이 아니라, 교사와 학생 모델 간의 구조적 정렬이 이데올로기 전이에 중요한 역할을 한다는 점을 강조합니다.

---



This paper proposes a novel methodology called Temporally Informed Distillation of Embedding Semantics (TIDES). TIDES is a framework designed to transfer knowledge from large embedding models to smaller BERT-based student models for modeling historically situated gender ideology. This approach consists of two main modules: the historical grounding module and the feature-level distillation module.

1. **Model Architecture**:
   - TIDES is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, transferring knowledge from large teacher models (e.g., bge-large-en, gte-large-en-v1.5) to a smaller student model (bert-base-uncased).
   - The teacher models have 24 layers and a hidden size of 1024, while the student model has 12 layers and a hidden size of 768. This structural difference allows the student model to learn effectively with fewer parameters.

2. **Training Data**:
   - TIDES utilizes the COHA (Corpus of Historical American English) dataset, which includes historical texts from the 1990s to train the student model. This dataset reflects the linguistic environment of the era, including newspapers, magazines, fiction, and non-fiction.
   - The student model adapts to the historical data using a masked language modeling (MLM) technique, where 15% of the tokens are randomly masked, and the model is trained to recover the original tokens.

3. **Feature-Level Distillation**:
   - TIDES transfers knowledge by aligning internal representations between the teacher and student models. This process consists of three main loss functions:
     - **Embedding Loss**: Alignment at the input layer.
     - **Hidden State Loss**: Alignment of hidden states at intermediate layers.
     - **Attention Loss**: Alignment of attention mechanisms.
   - These loss functions help the student model internalize the structural and ideological representations of the teacher model.

4. **Training Process**:
   - TIDES trains by combining MLM loss and feature-level loss. The weights of these two losses are adjusted by α, set to 0.2 to emphasize feature alignment.
   - Training is performed using the AdamW optimizer with a learning rate of 5×10^-5. The training lasts for up to 50 epochs, with a batch size of 32.

Through this methodology, TIDES demonstrates the potential to effectively model historical gender ideology. It emphasizes that effective ideological transfer is not merely data-dependent but also relies on the structural alignment between teacher and student models.


<br/>
# Results
### 결과 요약 (한글)

이 연구에서는 Temporally Informed Distillation of Embedding Semantics (TIDES) 프레임워크를 통해 역사적 성 이데올로기를 모델링하는 데 있어 다양한 모델의 성능을 비교하고 평가하였다. 실험은 다음과 같은 주요 요소로 구성되었다.

1. **경쟁 모델**: 
   - **기본 모델**: BERT (bert-base-uncased)
   - **계속 사전 훈련된 모델**: 1990년대 COHA 코퍼스를 사용하여 훈련된 BERT
   - **교사 모델**: bge-large-en (대비 학습을 통해 훈련됨) 및 gte-large-en-v1.5 (다중 작업 학습을 통해 훈련됨)

2. **테스트 데이터**: 
   - 1990년대의 COHA 코퍼스를 사용하여 모델을 훈련하고 평가하였다. 이 코퍼스는 신문, 잡지, 소설 및 비소설을 포함하여 해당 시대의 언어 환경을 반영한다.

3. **메트릭**: 
   - 모델의 성능은 Spearman의 순위 상관계수(ρ), Kendall의 순위 상관계수(τ), 그리고 선형 회귀의 결정 계수(R²)를 사용하여 평가되었다. 이 메트릭들은 모델이 인간의 성 이데올로기 평가와 얼마나 잘 일치하는지를 측정하는 데 사용되었다.

4. **비교 결과**: 
   - TIDES-bge 모델은 모든 메트릭에서 가장 높은 평균 값을 기록하였으며, 계속 사전 훈련된 BERT 모델보다 약 35%의 상대적 개선을 보였다. TIDES-gte 모델은 계속 사전 훈련된 BERT 모델에 비해 일관되게 낮은 성능을 보였으며, 이는 교사 모델의 구조적 호환성이 성능에 중요한 영향을 미친다는 것을 시사한다.
   - 각 모델의 레이어별 R² 값은 TIDES-bge가 Layer 10에서 최대치를 기록한 반면, 계속 사전 훈련된 BERT는 Layer 3에서 최대치를 기록하였다. 이는 TIDES-bge가 더 높은 차원의 의미와 이데올로기적 표현을 효과적으로 전이했음을 나타낸다.

5. **결론**: 
   - TIDES 프레임워크는 역사적 성 이데올로기를 모델링하는 데 있어 효과적인 방법론을 제공하며, 교사 모델의 구조적 호환성이 이데올로기 전이에 중요한 역할을 한다는 것을 보여주었다. TIDES-bge 모델은 역사적 데이터와 현대 모델 간의 정합성을 유지하면서 이데올로기적 표현을 성공적으로 전이할 수 있는 가능성을 제시하였다.

---




This study compared and evaluated the performance of various models in modeling historical gender ideology through the Temporally Informed Distillation of Embedding Semantics (TIDES) framework. The experiments consisted of the following key elements:

1. **Competing Models**: 
   - **Baseline Model**: BERT (bert-base-uncased)
   - **Continued Pretrained Model**: BERT trained on the 1990s COHA corpus
   - **Teacher Models**: bge-large-en (trained via contrastive learning) and gte-large-en-v1.5 (trained via multi-task learning)

2. **Test Data**: 
   - The 1990s COHA corpus was used to train and evaluate the models. This corpus includes newspapers, magazines, fiction, and non-fiction, reflecting the linguistic environment of the decade.

3. **Metrics**: 
   - The performance of the models was evaluated using Spearman's rank correlation coefficient (ρ), Kendall's rank correlation coefficient (τ), and the coefficient of determination (R²) from linear regression. These metrics were used to measure how well the models aligned with human assessments of gender ideology.

4. **Comparison Results**: 
   - The TIDES-bge model achieved the highest average values across all metrics, showing approximately a 35% relative improvement over the continued pretrained BERT model. The TIDES-gte model consistently underperformed compared to the continued pretrained BERT model, indicating that the structural compatibility of the teacher model significantly impacts performance.
   - The layer-wise R² values showed that TIDES-bge peaked at Layer 10, while the continued pretrained BERT reached its maximum at Layer 3. This indicates that TIDES-bge effectively transferred higher-order semantic and ideological representations.

5. **Conclusion**: 
   - The TIDES framework provides an effective methodology for modeling historical gender ideology, demonstrating that the structural compatibility of the teacher model plays a crucial role in ideological transfer. The TIDES-bge model successfully showcased the potential to transfer ideological representations while maintaining coherence between historical data and contemporary models.


<br/>
# 예제



이 논문에서는 Temporally Informed Distillation of Embedding Semantics (TIDES)라는 방법론을 제안하여, 역사적 텍스트에서 성 이데올로기를 모델링하는 데 있어 언어 모델의 성능을 향상시키고자 하였습니다. TIDES는 두 가지 주요 모듈로 구성되어 있습니다: 역사적 기초 모듈과 특징 수준 증류 모듈입니다.

1. **트레이닝 데이터**: 
   - **입력**: 1990년대의 역사적 텍스트 데이터로 구성된 COHA(Corpus of Historical American English)에서 수집된 문장들입니다. 예를 들어, "그는 용감한 사람이다."와 같은 문장이 입력으로 사용될 수 있습니다.
   - **출력**: 모델은 각 문장에 대해 문맥화된 임베딩을 생성합니다. 이 임베딩은 문장의 의미를 포착하고, 성 이데올로기와 관련된 정보를 포함합니다.

2. **테스트 데이터**:
   - **입력**: 동일한 COHA 데이터에서 추출된 문장들로, 예를 들어 "그녀는 의존적인 사람이다."와 같은 문장이 포함될 수 있습니다.
   - **출력**: 모델은 이 문장에 대한 임베딩을 생성하고, 이를 통해 성별과 관련된 이데올로기 점수를 계산합니다. 이 점수는 모델이 얼마나 잘 성 이데올로기를 반영하는지를 나타냅니다.

3. **구체적인 테스크**:
   - 모델은 성별에 따라 단어의 연관성을 평가하는 작업을 수행합니다. 예를 들어, "용감한"이라는 단어가 남성적 특성과 얼마나 밀접하게 연관되어 있는지를 평가합니다. 이 과정에서 모델은 인간의 설문조사 결과와 비교하여 성 이데올로기를 얼마나 잘 반영하는지를 측정합니다.

4. **평가 방법**:
   - 모델의 성능은 R² 값, Spearman의 ρ, Kendall의 τ와 같은 통계적 지표를 사용하여 평가됩니다. 이 지표들은 모델이 생성한 성 이데올로기 점수가 인간의 평가와 얼마나 잘 일치하는지를 나타냅니다.

이러한 방식으로 TIDES는 역사적 텍스트에서 성 이데올로기를 효과적으로 모델링할 수 있는 방법을 제시하고 있습니다.

---




In this paper, the authors propose a methodology called Temporally Informed Distillation of Embedding Semantics (TIDES) to enhance the performance of language models in modeling gender ideology in historical texts. TIDES consists of two main modules: the historical grounding module and the feature-level distillation module.

1. **Training Data**:
   - **Input**: Sentences collected from the 1990s historical text data in the COHA (Corpus of Historical American English). For example, a sentence like "He is a brave person." could be used as input.
   - **Output**: The model generates contextualized embeddings for each sentence. These embeddings capture the meaning of the sentence and include information related to gender ideology.

2. **Test Data**:
   - **Input**: Sentences extracted from the same COHA data, such as "She is a dependent person."
   - **Output**: The model generates embeddings for these sentences and calculates gender-related ideology scores. These scores indicate how well the model reflects gender ideology.

3. **Specific Task**:
   - The model performs a task of evaluating the association of words with gender. For instance, it assesses how closely the word "brave" is associated with masculine traits. In this process, the model compares its results with human survey data to measure how well it captures gender ideology.

4. **Evaluation Method**:
   - The model's performance is evaluated using statistical metrics such as R², Spearman's ρ, and Kendall's τ. These metrics indicate how well the gender ideology scores generated by the model align with human evaluations.

Through this approach, TIDES presents a method for effectively modeling gender ideology in historical texts.

<br/>
# 요약

이 논문에서는 Temporally Informed Distillation of Embedding Semantics (TIDES)라는 방법론을 제안하여, 역사적 텍스트에서 성 이데올로기를 모델링하기 위해 대형 임베딩 모델의 지식을 소형 모델로 전이하는 과정을 설명합니다. TIDES는 지속적인 사전 훈련과 특성 수준의 증류를 결합하여, 역사적 성 이데올로기를 더 잘 회복할 수 있음을 보여주며, 특히 구조적 정합성이 전이 효과에 중요한 역할을 한다는 것을 입증합니다. 실험 결과, TIDES는 인간의 설문 기반 이데올로기 측정과의 정렬을 개선하며, 특히 bge-large-en 모델을 사용할 때 가장 효과적임을 나타냅니다.

---

This paper proposes a methodology called Temporally Informed Distillation of Embedding Semantics (TIDES) to transfer knowledge from large embedding models to smaller models for modeling gender ideology in historical texts. TIDES combines continued pretraining and feature-level distillation, demonstrating improved recovery of historical gender ideology, with structural alignment being critical to transfer effectiveness. Experimental results show that TIDES enhances alignment with human survey-based ideology measures, particularly when using the bge-large-en model.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1**: TIDES의 구조를 보여주는 다이어그램으로, 두 개의 모듈(역사적 기초 및 특징 수준 증류 모듈)로 구성되어 있습니다. 이 구조는 대규모 교사 모델과 소규모 학생 모델 간의 효율적인 지식 전이를 가능하게 합니다.
  
- **Figure 2**: 증류 계수 α의 민감도 분석 결과를 보여줍니다. α=0.2에서 모델이 인간 설문 기준과 가장 높은 정렬을 보였으며, 이는 안정적인 성능을 나타냅니다.

- **Figure 4**: 다양한 모델의 계층별 R² 값을 비교한 결과를 보여줍니다. TIDES-bge 모델이 계속 사전 훈련된 BERT 모델보다 전반적으로 우수한 성능을 보였으며, 이는 구조적 정렬의 중요성을 강조합니다.

- **Figure 5**: 부트스트랩 커널 밀도 추정 결과를 보여줍니다. TIDES-bge의 분포가 계속 사전 훈련된 BERT 및 TIDES-gte보다 우측으로 치우쳐 있어, TIDES-bge가 더 나은 성능을 나타냄을 시사합니다.

- **Figure 6**: 단어 수준의 성별 정렬을 보여주는 그래프입니다. TIDES-bge 모델이 1990년대의 성별 고정관념을 더 잘 반영하고 있음을 나타냅니다.

#### 2. 테이블
- **Table 2**: 모델의 주요 아키텍처 및 훈련 매개변수를 비교한 표입니다. 각 모델의 파라미터 수, 레이어 수, 훈련 목표 등이 나열되어 있어, TIDES의 성능을 이해하는 데 도움이 됩니다.

- **Table 4**: 특징 구성 요소의 제거 실험 결과를 보여줍니다. 모든 구성 요소가 성능에 긍정적인 기여를 하며, 전체 TIDES 모델이 가장 높은 평균 R² 값을 기록했습니다.

- **Table 5**: 모델 간 성능 비교 결과를 요약한 표입니다. TIDES-bge가 모든 평가 지표에서 가장 높은 평균 값을 기록했으며, 이는 TIDES의 효과를 뒷받침합니다.




### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Figure 1**: A diagram illustrating the structure of TIDES, consisting of two modules (historical grounding and feature-level distillation modules). This structure enables efficient knowledge transfer between large teacher models and smaller student models.

- **Figure 2**: Results of the sensitivity analysis for the distillation coefficient α. The model showed the highest alignment with human survey benchmarks at α=0.2, indicating stable performance.

- **Figure 4**: A comparison of layer-wise R² values across different models. The TIDES-bge model consistently outperformed the continued-pretrained BERT model, emphasizing the importance of structural alignment.

- **Figure 5**: Bootstrap kernel density estimates showing the distribution of performance. The distribution of TIDES-bge is skewed to the right compared to the continued-pretrained BERT and TIDES-gte, indicating better performance.

- **Figure 6**: A graph illustrating word-level gender alignment. The TIDES-bge model better reflects the gender stereotypes of the 1990s.

#### 2. Tables
- **Table 2**: A comparison of key architectural and training parameters across models. It lists the number of parameters, layers, and training objectives for each model, aiding in understanding TIDES' performance.

- **Table 4**: Results of the ablation study on feature components. It shows that all components contribute positively to performance, with the full TIDES model achieving the highest average R².

- **Table 5**: A summary of performance comparisons across models. TIDES-bge recorded the highest mean values across all evaluation metrics, supporting the effectiveness of TIDES.

<br/>
# refer format:



```bibtex
@article{Ge2026,
  author = {Yingqiu Ge and Jinghang Gu and Chu-Ren Huang},
  title = {Temporally Informed Distillation of Embedding Semantics: Beyond Continued Pretraining for Modeling Gender Ideology in Dated Texts},
  journal = {Data},
  volume = {11},
  number = {126},
  year = {2026},
  month = {May},
  pages = {1-22},
  doi = {10.3390/data11060126},
  publisher = {MDPI},
  copyright = {© 2026 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license.}
}
```



Ge, Yingqiu, Jinghang Gu, and Chu-Ren Huang. "Temporally Informed Distillation of Embedding Semantics: Beyond Continued Pretraining for Modeling Gender Ideology in Dated Texts." *Data* 11, no. 126 (May 2026): 1-22. https://doi.org/10.3390/data11060126.
