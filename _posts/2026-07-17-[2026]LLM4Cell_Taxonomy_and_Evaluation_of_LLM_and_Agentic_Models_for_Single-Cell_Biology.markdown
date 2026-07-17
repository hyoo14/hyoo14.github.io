---
layout: post
title:  "[2026]LLM4Cell: Taxonomy and Evaluation of LLM and Agentic Models for Single-Cell Biology"
date:   2026-07-17 03:02:28 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문은 LLM4Cell이라는 통합된 프레임워크를 통해 단일 세포 생물학을 위한 대형 언어 모델(LLM)과 에이전틱 모델의 분류 및 평가를 제시합니다.


짧은 요약(Abstract) :


이 논문은 LLM4Cell이라는 프로젝트를 통해 대규모 언어 모델(LLM)과 에이전트 모델이 단일 세포 생물학에 미치는 영향을 조사합니다. LLM과 에이전트 프레임워크는 자연어 인터페이스, 생성적 주석, 다중 모드 데이터 통합을 가능하게 하여 단일 세포 생물학의 발전에 기여하고 있습니다. 그러나 데이터 모달리티, 모델 패밀리, 평가 관행이 분산되어 있어 진전이 단편적입니다. LLM4Cell은 RNA, ATAC, 다중 오믹스 및 공간 모달리티를 포함한 58개의 기초 및 에이전트 모델을 통합적으로 조사하고, 이들을 다섯 가지 가족으로 조직하여 주석, 경로 추론, 교란 모델링, 약물 반응 예측 등 여덟 가지 주요 분석 작업에 매핑합니다. 40개 이상의 공개 데이터 세트를 활용하여 벤치마크 범위, 데이터 다양성, 윤리적 또는 확장성 제약을 분석하고, 생물학적 기초, 다중 모드 정렬, 공정성, 개인 정보 보호 및 해석 가능성과 관련된 10개 도메인 차원에서 보고된 기능을 종합합니다. LLM4Cell은 데이터 세트, 모델링 패러다임 및 평가 도메인을 명시적으로 연결하여 언어 기반 단일 세포 분석에 대한 통합된 관점을 제공하고, 표준화, 해석 가능성 및 신뢰할 수 있는 모델 개발의 개방된 도전 과제를 강조합니다.



This paper investigates the impact of large language models (LLMs) and agentic frameworks on single-cell biology through a project called LLM4Cell. LLMs and agentic frameworks enable natural language interfaces, generative annotation, and multimodal data integration, contributing to advancements in single-cell biology. However, progress remains fragmented across data modalities, model families, and evaluation practices. LLM4Cell presents a unified survey of 58 foundation and agentic models developed for single-cell research, spanning RNA, ATAC, multi-omic, and spatial modalities. These methods are organized into five families and mapped to eight key analytical tasks, including annotation, trajectory inference, perturbation modeling, and drug-response prediction. Drawing on over 40 public datasets, the paper analyzes benchmark coverage, data diversity, and ethical or scalability constraints, synthesizing reported capabilities across ten domain-level dimensions related to biological grounding, multimodal alignment, fairness, privacy, and interpretability. By explicitly linking datasets, modeling paradigms, and evaluation domains, LLM4Cell provides an integrated perspective on language-driven single-cell analysis and highlights open challenges in standardization, interpretability, and trustworthy model development.


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



LLM4Cell 논문에서 제시된 메써드는 단일 세포 생물학을 위한 다양한 대형 언어 모델(LLM)과 에이전틱 모델을 포함합니다. 이들 모델은 크게 다섯 가지 방법론적 가족으로 분류됩니다: 기초 모델(Foundation Models), 텍스트-브리지 모델(Text-Bridge Models), 공간 및 다중 모달 모델(Spatial and Multimodal Models), 후생유전학 모델(Epigenomic Models), 그리고 에이전틱 프레임워크(Agentic Frameworks)입니다.

1. **기초 모델 (Foundation Models)**: 이 모델들은 대규모 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터에서 직접 학습된 전이 가능한 세포 및 유전자 임베딩을 생성합니다. 예를 들어, scGPT, Geneformer, scFoundation과 같은 모델은 대규모 다기관 아틀라스를 사용하여 훈련되며, 마스킹된 유전자 예측 또는 순위 기반 재구성을 통해 유전자 발현 맥락을 포착합니다. 이러한 모델들은 주로 주석(annotation) 및 통합(integration) 작업에 강점을 보이지만, 생물학적 기초가 부족하고 해석 가능성이 제한적입니다.

2. **텍스트-브리지 모델 (Text-Bridge Models)**: 이 모델들은 분자 임베딩과 생물 의학 언어를 결합하여 단일 세포 표현을 의미론적으로 연결합니다. 예를 들어, CellLM과 GenePT는 유전자 또는 세포 임베딩을 텍스트 설명과 정렬하여 해석 가능성을 높이고, 제로샷 주석(zero-shot annotation)을 가능하게 합니다. 이러한 모델들은 주로 이론적 해석을 제공하지만, 여전히 비에이전틱 모델에 비해 제한된 일반화 능력을 보입니다.

3. **공간 및 다중 모달 모델 (Spatial and Multimodal Models)**: 이 모델들은 유전자 발현과 공간 좌표를 통합하여 조직 구조를 포착합니다. TransformerST와 OmiCLIP과 같은 모델은 공간적 맥락을 고려하여 세포 또는 스팟 수준의 맵을 재구성합니다. 이러한 모델들은 생물학적 현실성을 높이지만, 이질적인 해상도와 높은 계산 비용으로 인해 제한을 받습니다.

4. **후생유전학 모델 (Epigenomic Models)**: 이 모델들은 크로마틴 접근성 및 조절 데이터를 다루며, EpiFoundation과 EpiBERT와 같은 모델은 ATAC-seq 데이터를 사용하여 유전자 조절 네트워크를 추론합니다. 이러한 모델들은 생물학적 기초를 개선하지만, 데이터의 희소성과 통합된 벤치마크의 부족으로 인해 한계가 있습니다.

5. **에이전틱 프레임워크 (Agentic Frameworks)**: 이 시스템들은 사전 훈련된 모델과 추론 모듈을 통합하여 자율적인 단일 세포 분석을 가능하게 합니다. scAgent와 CellVerse와 같은 모델은 LLM 컨트롤러와 도구 인터페이스를 결합하여 대화 기반 주석 및 다단계 추론을 수행합니다. 그러나 이러한 시스템들은 추론 정확성에 대한 표준화된 벤치마크가 부족하여 평가가 어렵습니다.

이러한 메써드는 단일 세포 생물학의 다양한 분석 작업을 지원하며, 각 모델은 특정한 데이터 모달리티와 훈련 기법을 통해 고유한 강점과 한계를 지니고 있습니다. LLM4Cell은 이러한 모델들을 통합하여 단일 세포 분석의 언어 기반 접근 방식을 제시하고, 데이터 가용성 및 평가 관행이 모델의 성능에 미치는 영향을 분석합니다.

---




The methods presented in the LLM4Cell paper encompass various large language models (LLMs) and agentic models for single-cell biology. These models are categorized into five methodological families: Foundation Models, Text-Bridge Models, Spatial and Multimodal Models, Epigenomic Models, and Agentic Frameworks.

1. **Foundation Models**: These models generate transferable cell and gene embeddings directly from large-scale single-cell RNA sequencing (scRNA-seq) data. For instance, models like scGPT, Geneformer, and scFoundation are trained on large multi-tissue atlases, capturing expression context through masked gene prediction or rank-based reconstruction. These models excel primarily in annotation and integration tasks but lack explicit biological grounding and have limited interpretability.

2. **Text-Bridge Models**: These models explicitly align molecular embeddings with biomedical language to ground single-cell representations in semantics. For example, CellLM and GenePT align gene or cell embeddings with textual descriptors, enhancing interpretability and enabling zero-shot annotation. While these models provide theoretical insights, they still exhibit limited generalization compared to non-agentic models.

3. **Spatial and Multimodal Models**: These models integrate gene expression with spatial coordinates to capture tissue architecture. Models like TransformerST and OmiCLIP reconstruct cell or spot-level maps while considering spatial context. Although these models improve biological realism, they face limitations due to heterogeneous resolutions and high computational costs.

4. **Epigenomic Models**: These models address chromatin accessibility and regulatory data, with models like EpiFoundation and EpiBERT inferring gene regulatory networks from ATAC-seq data. While these models enhance biological grounding, they are constrained by sparse data and a lack of unified benchmarks across regulatory modalities.

5. **Agentic Frameworks**: These systems integrate pretrained models with reasoning modules for autonomous single-cell analysis. Frameworks like scAgent and CellVerse combine domain-specific encoders with LLM controllers to perform dialogue-based annotation and multi-step reasoning. However, these systems lack standardized benchmarks for evaluating reasoning fidelity, making it challenging to assess their performance.

These methods support various analytical tasks in single-cell biology, with each model possessing unique strengths and limitations based on specific data modalities and training techniques. LLM4Cell provides an integrated perspective on language-driven single-cell analysis, analyzing how dataset availability and evaluation practices shape model performance.


<br/>
# Results



LLM4Cell 논문에서는 단일 세포 생물학을 위한 대형 언어 모델(LLM)과 에이전틱 모델의 평가 및 분류를 다루고 있습니다. 이 연구는 58개의 모델을 분석하고, 이들 모델이 지원하는 데이터셋, 주요 작업, 평가 기준 등을 체계적으로 정리하였습니다. 

1. **모델 분류**: 모델은 크게 다섯 가지 가족으로 나뉘며, 각각의 모델은 특정한 데이터 모달리티와 작업에 최적화되어 있습니다. 이들 모델은 기초 모델(Foundation Models), 텍스트-브리지 모델(Text-Bridge Models), 공간 및 다중 모달 모델(Spatial and Multimodal Models), 후생유전학 모델(Epigenomic Models), 에이전틱 프레임워크(Agentic Frameworks)로 구분됩니다.

2. **경쟁 모델**: 각 모델은 특정 작업에서의 성능을 기준으로 비교되었습니다. 예를 들어, scGPT, Geneformer, CellLM 등은 주로 주석(annotation) 및 통합(integration) 작업에서 높은 성능을 보였습니다. 반면, 에이전틱 모델인 scAgent와 CellVerse는 다단계 추론(multi-step reasoning) 및 도구 사용(tool use)에서 두각을 나타냈습니다.

3. **테스트 데이터**: 연구에서는 40개 이상의 공개 데이터셋을 사용하여 모델의 성능을 평가하였습니다. 데이터셋은 RNA, ATAC, 다중 오믹스(multi-omics), 공간 데이터 등 다양한 모달리티를 포함하고 있습니다. 이러한 데이터셋은 모델의 훈련 및 평가에 필수적이며, 각 데이터셋의 특성과 규모에 따라 모델의 성능이 달라질 수 있습니다.

4. **메트릭**: 모델의 성능 평가는 생물학적 기초(biological grounding), 일반화(generalization), 해석 가능성(interpretability), 공정성(fairness), 개인 정보 보호(privacy) 등 10가지 차원에서 이루어졌습니다. 각 차원은 모델이 해당 특성을 얼마나 잘 충족하는지를 평가하는 기준으로 사용되었습니다.

5. **비교 결과**: 에이전틱 모델은 주석 및 공간 매핑에서 높은 성능을 보였으나, 비에이전틱 모델은 주로 생물학적 기초와 배치 효과(batch effects)에서 강점을 보였습니다. 또한, 에이전틱 시스템은 설명 가능성(explainability) 및 공정성(fairness)에서 높은 점수를 기록했지만, 생물학적 기초와 배치 효과에 대한 성능은 상대적으로 낮았습니다.

이 연구는 LLM과 에이전틱 모델이 단일 세포 생물학 분야에서 어떻게 발전하고 있는지를 보여주며, 향후 연구 방향과 개선이 필요한 영역을 제시합니다.

---




The LLM4Cell paper addresses the evaluation and taxonomy of large language models (LLMs) and agentic models for single-cell biology. This study analyzes 58 models and systematically organizes the datasets they support, primary tasks, and evaluation criteria.

1. **Model Classification**: The models are categorized into five families, each optimized for specific data modalities and tasks. These families include Foundation Models, Text-Bridge Models, Spatial and Multimodal Models, Epigenomic Models, and Agentic Frameworks.

2. **Competing Models**: Each model was compared based on its performance on specific tasks. For instance, models like scGPT, Geneformer, and CellLM excelled in annotation and integration tasks. In contrast, agentic models such as scAgent and CellVerse stood out in multi-step reasoning and tool use.

3. **Test Data**: The study utilized over 40 public datasets to evaluate model performance. These datasets encompass various modalities, including RNA, ATAC, multi-omics, and spatial data. The characteristics and scale of these datasets are crucial for training and evaluating the models, influencing their performance.

4. **Metrics**: Model performance was assessed across ten dimensions, including biological grounding, generalization, interpretability, fairness, and privacy. Each dimension served as a criterion for evaluating how well the models meet specific characteristics.

5. **Comparison Results**: Agentic models demonstrated high performance in annotation and spatial mapping, while non-agentic models showed strengths in biological grounding and batch effects. Additionally, agentic systems scored high in explainability and fairness, but their performance in biological grounding and batch effects was relatively lower.

This research illustrates how LLMs and agentic models are evolving in the field of single-cell biology and highlights future research directions and areas for improvement.


<br/>
# 예제



이 논문에서는 LLM4Cell이라는 모델을 통해 단일 세포 생물학에서의 대규모 언어 모델(LLM)과 에이전틱 모델의 분류 및 평가를 다루고 있습니다. 이 모델은 다양한 데이터 모달리티와 분석 작업을 통합하여 단일 세포 연구에 대한 포괄적인 관점을 제공합니다. 

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **데이터셋**: Tabula Sapiens, Human Cell Atlas, Mouse Cell Atlas 등
   - **모달리티**: RNA, ATAC, multi-omic 데이터
   - **스케일**: 예를 들어, Tabula Sapiens는 약 110만 개의 세포를 포함하고 있으며, 다양한 조직에서 수집된 데이터입니다.
   - **입력**: 각 세포의 유전자 발현 프로파일, 크로마틴 접근성 데이터, 공간적 정보 등
   - **출력**: 세포 유형 분류, 유전자 주석, 약물 반응 예측 등

2. **테스트 데이터**:
   - **데이터셋**: sci-Plex, Perturb-seq 등
   - **모달리티**: 약물 반응 데이터, 유전자 변형 데이터
   - **스케일**: 예를 들어, Perturb-seq 데이터는 250만 개의 세포를 포함하며, 다양한 유전자 변형에 대한 반응을 측정합니다.
   - **입력**: 유전자 변형 후의 세포 발현 프로파일
   - **출력**: 유전자 발현 변화, 세포의 반응 예측 등

#### 구체적인 테스크

- **주석 및 주석 매핑**: 세포의 유전자 발현 데이터를 기반으로 세포 유형을 자동으로 주석화합니다. 예를 들어, scGPT 모델은 Tabula Sapiens 데이터셋을 사용하여 세포의 유전자 발현 패턴을 분석하고, 이를 통해 세포 유형을 분류합니다.
  
- **궤적 및 교란 모델링**: 세포가 시간에 따라 어떻게 변화하는지를 모델링합니다. Geneformer 모델은 시간에 따른 유전자 발현 변화를 예측하기 위해 scRNA-seq 데이터를 사용합니다.

- **약물 반응 예측**: 약물 처리 후 세포의 유전자 발현 변화를 예측합니다. EpiFoundation 모델은 Perturb-seq 데이터를 사용하여 약물의 효과를 예측합니다.

이러한 작업들은 LLM4Cell의 다양한 기능을 통해 수행되며, 각 모델은 특정 데이터셋과 작업에 최적화되어 있습니다.

---




This paper discusses LLM4Cell, a model that addresses the taxonomy and evaluation of large language models (LLMs) and agentic models in single-cell biology. This model provides a comprehensive perspective on single-cell research by integrating various data modalities and analytical tasks.

#### Training Data and Test Data

1. **Training Data**:
   - **Datasets**: Tabula Sapiens, Human Cell Atlas, Mouse Cell Atlas, etc.
   - **Modalities**: RNA, ATAC, multi-omic data
   - **Scale**: For example, Tabula Sapiens includes approximately 1.1 million cells collected from various tissues.
   - **Input**: Gene expression profiles of each cell, chromatin accessibility data, spatial information, etc.
   - **Output**: Cell type classification, gene annotation, drug response prediction, etc.

2. **Test Data**:
   - **Datasets**: sci-Plex, Perturb-seq, etc.
   - **Modalities**: Drug response data, genetic perturbation data
   - **Scale**: For instance, Perturb-seq data includes 2.5 million cells and measures responses to various genetic perturbations.
   - **Input**: Cell expression profiles after genetic perturbation
   - **Output**: Changes in gene expression, predictions of cellular responses, etc.

#### Specific Tasks

- **Annotation and Ontology Mapping**: Automatically annotates cell types based on gene expression data. For example, the scGPT model analyzes the Tabula Sapiens dataset to classify cell types based on gene expression patterns.

- **Trajectory and Perturbation Modeling**: Models how cells change over time. The Geneformer model uses scRNA-seq data to predict temporal changes in gene expression.

- **Drug Response Prediction**: Predicts changes in gene expression following drug treatment. The EpiFoundation model uses Perturb-seq data to predict the effects of drugs.

These tasks are performed through the various capabilities of LLM4Cell, with each model optimized for specific datasets and tasks.

<br/>
# 요약


이 논문은 LLM4Cell이라는 통합된 프레임워크를 통해 단일 세포 생물학을 위한 대형 언어 모델(LLM)과 에이전틱 모델의 분류 및 평가를 제시합니다. 58개의 모델을 다섯 가지 방법론적 가족으로 나누고, 40개 이상의 공개 데이터셋을 기반으로 생물학적 이해와 추론 능력을 분석하여 현재의 한계와 향후 연구 방향을 제시합니다. 이 연구는 데이터 가용성, 모델 설계, 평가 관행이 생물학적 이해에 미치는 영향을 강조하며, 표준화된 평가의 필요성을 강조합니다.

---

This paper presents a unified framework called LLM4Cell for the taxonomy and evaluation of large language models (LLMs) and agentic models in single-cell biology. It categorizes 58 models into five methodological families and analyzes biological understanding and reasoning capabilities based on over 40 public datasets, highlighting current limitations and future research directions. The study emphasizes the impact of data availability, model design, and evaluation practices on biological understanding, calling for the need for standardized evaluation.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **모델 및 작업 히트맵 (Figure 4)**: 이 히트맵은 다양한 모델이 지원하는 작업의 범위를 시각적으로 나타냅니다. 각 모델이 수행할 수 있는 작업의 수를 보여주며, 특정 모델이 여러 작업을 지원하는 경향이 있음을 알 수 있습니다. 이는 연구자들이 특정 작업에 적합한 모델을 선택하는 데 유용합니다.

2. **작업 커버리지 비교 (Figure 5)**: 이 피규어는 에이전틱 모델과 비에이전틱 모델 간의 작업 커버리지를 비교합니다. 에이전틱 모델은 주로 주석, 온톨로지 매핑 및 공간 매핑에 중점을 두는 반면, 비에이전틱 모델은 궤적 추적, 교란 모델링 및 규제 및 경로 추론에 집중하는 경향이 있습니다. 이는 각 모델의 강점과 약점을 이해하는 데 도움이 됩니다.

3. **도메인 커버리지 비교 (Figure 6)**: 이 피규어는 에이전틱 모델과 비에이전틱 모델 간의 도메인 커버리지를 비교합니다. 에이전틱 모델은 설명 가능성, 공정성 및 새로운 패러다임에 중점을 두는 반면, 비에이전틱 모델은 생물학적 기초 및 배치 효과에 집중합니다. 이는 연구자들이 특정 도메인에서 어떤 모델이 더 적합한지를 판단하는 데 유용합니다.

#### 테이블
1. **모델 비교 테이블 (Table 10)**: 이 테이블은 다양한 LLM 및 에이전틱 방법을 비교합니다. 각 모델의 카테고리, 모달리티, 그라운딩 유형, 에이전틱 여부, 주요 작업 및 도메인 커버리지를 나열합니다. 이를 통해 연구자들은 각 모델의 특성과 강점을 쉽게 비교할 수 있습니다.

#### 어펜딕스
1. **데이터셋 요약 (Appendix D)**: 데이터셋의 범위와 전처리 방법을 설명합니다. RNA, ATAC, 다중 오믹스, 공간 전사체 및 약물 반응 데이터셋의 다양한 특성을 강조합니다. 이는 연구자들이 특정 데이터셋을 선택할 때 고려해야 할 요소를 이해하는 데 도움이 됩니다.

2. **모델 특성 요약 (Appendix G)**: 대표 모델의 주요 특성을 요약하여 아키텍처 유형, 지원하는 모달리티, 사전 훈련 데이터 및 보고된 규모를 제공합니다. 이는 연구자들이 모델의 성능과 효율성을 평가하는 데 유용합니다.




#### Diagrams and Figures
1. **Model and Task Heatmap (Figure 4)**: This heatmap visually represents the range of tasks supported by various models. It shows the number of tasks each model can perform, indicating that certain models tend to support multiple tasks. This is useful for researchers in selecting models suitable for specific tasks.

2. **Task Coverage Comparison (Figure 5)**: This figure compares the task coverage between agentic and non-agentic models. Agentic models tend to focus on annotation, ontology mapping, and spatial mapping, while non-agentic models concentrate on trajectory inference, perturbation modeling, and regulatory and pathway inference. This helps in understanding the strengths and weaknesses of each model.

3. **Domain Coverage Comparison (Figure 6)**: This figure compares the domain coverage between agentic and non-agentic models. Agentic models emphasize explainability, fairness, and emerging paradigms, while non-agentic models focus on biological grounding and batch effects. This is useful for researchers in determining which model is more suitable for specific domains.

#### Tables
1. **Model Comparison Table (Table 10)**: This table compares various LLM and agentic methods. It lists each model's category, modality, grounding type, agentic status, primary tasks, and domain coverage. This allows researchers to easily compare the characteristics and strengths of each model.

#### Appendix
1. **Dataset Summary (Appendix D)**: It describes the scope and preprocessing methods of the datasets. It highlights the various characteristics of RNA, ATAC, multi-omic, spatial transcriptomics, and drug response datasets. This helps researchers understand the factors to consider when selecting specific datasets.

2. **Model Characteristics Summary (Appendix G)**: It summarizes key characteristics of representative models, including architecture type, supported modalities, pretraining data, and reported scale. This is useful for researchers in evaluating model performance and efficiency.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Acharjee2026,
  author    = {Sajib Acharjee Dip and Adrika Zafor and Bikash Kumar Paul and Uddip Acharjee Shuvo and Muhit Islam Emon and Xuan Wang and Liqing Zhang},
  title     = {LLM4Cell: Taxonomy and Evaluation of LLM and Agentic Models for Single-Cell Biology},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {41913--41954},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### Chicago Style Citation

Acharjee Dip, Sajib, Adrika Zafor, Bikash Kumar Paul, Uddip Acharjee Shuvo, Muhit Islam Emon, Xuan Wang, and Liqing Zhang. "LLM4Cell: Taxonomy and Evaluation of LLM and Agentic Models for Single-Cell Biology." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 41913–41954. July 2-7, 2026. Association for Computational Linguistics.
