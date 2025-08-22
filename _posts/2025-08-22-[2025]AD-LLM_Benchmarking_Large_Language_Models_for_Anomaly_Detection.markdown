---
layout: post
title:  "[2025]AD-LLM: Benchmarking Large Language Models for Anomaly Detection"
date:   2025-08-22 01:43:11 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

이 논문에서는 대규모 언어 모델(LLM)을 활용한 이상 탐지(AD) 방법을 제안하고, 세 가지 주요 작업인 제로샷 탐지, 데이터 증강, 모델 선택을 평가하는 AD-LLM 벤치마크를 소개



짧은 요약(Abstract) :

이 논문의 초록에서는 이상 탐지(Anomaly Detection, AD)의 중요성과 대규모 언어 모델(LLMs)의 잠재력에 대해 설명하고 있습니다. 이상 탐지는 사기 탐지, 의료 진단, 산업 모니터링 등 다양한 실제 응용 분야에서 중요한 기계 학습 작업입니다. 자연어 처리(NLP) 분야에서는 스팸, 잘못된 정보, 비정상적인 사용자 활동과 같은 문제를 탐지하는 데 AD가 도움이 됩니다. 대규모 언어 모델은 텍스트 생성 및 요약과 같은 작업에서 큰 영향을 미쳤지만, AD에서의 잠재력은 충분히 연구되지 않았습니다. 이 논문은 LLM이 NLP 이상 탐지에 어떻게 기여할 수 있는지를 평가하는 첫 번째 벤치마크인 AD-LLM을 소개합니다. 세 가지 주요 작업을 살펴보며, LLM이 사전 훈련된 지식을 활용하여 태스크 특정 훈련 없이 AD를 수행하는 제로샷 탐지, 합성 데이터를 생성하여 AD 모델을 개선하는 데이터 증강, LLM을 사용하여 비지도 AD 모델을 제안하는 모델 선택을 포함합니다. 실험 결과, LLM이 제로샷 AD에서 잘 작동하며, 신중하게 설계된 증강 방법이 유용하다는 것을 발견했습니다. 또한 특정 데이터셋에 대한 모델 선택을 설명하는 것은 여전히 도전 과제가 남아 있습니다. 이러한 결과를 바탕으로 LLM을 AD에 활용하기 위한 여섯 가지 미래 연구 방향을 제시합니다.



The abstract of this paper discusses the importance of anomaly detection (AD) and the potential of large language models (LLMs). Anomaly detection is a crucial machine learning task with many real-world applications, including fraud detection, medical diagnosis, and industrial monitoring. In the field of natural language processing (NLP), AD helps detect issues such as spam, misinformation, and unusual user activity. While large language models have had a significant impact on tasks like text generation and summarization, their potential in AD has not been sufficiently studied. This paper introduces AD-LLM, the first benchmark that evaluates how LLMs can contribute to NLP anomaly detection. It examines three key tasks: zero-shot detection, which uses LLMs' pretrained knowledge to perform AD without task-specific training; data augmentation, which generates synthetic data to improve AD models; and model selection, which uses LLMs to suggest unsupervised AD models. Through experiments, it finds that LLMs can perform well in zero-shot AD, that carefully designed augmentation methods are useful, and that explaining model selection for specific datasets remains a challenge. Based on these results, the paper outlines six future research directions for integrating LLMs into AD tasks.


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



이 논문에서는 대규모 언어 모델(LLM)을 활용한 이상 탐지(Anomaly Detection, AD) 방법론을 제안합니다. 이 방법론은 세 가지 주요 작업으로 구성됩니다: 제로샷 탐지, 데이터 증강, 모델 선택입니다.

1. **제로샷 탐지 (Zero-shot Detection)**: LLM은 사전 훈련된 지식을 활용하여 특정 작업에 대한 추가 훈련 없이도 이상을 탐지할 수 있습니다. 이 과정에서 LLM은 주어진 텍스트 샘플이 정상 범주에 속하는지 아니면 이상 범주에 속하는지를 판단합니다. 이 방법은 레이블이 부족한 상황에서도 효과적으로 작동할 수 있습니다.

2. **데이터 증강 (Data Augmentation)**: LLM은 기존의 정상 샘플을 기반으로 새로운 합성 데이터를 생성하여 훈련 데이터의 부족 문제를 해결합니다. 이 과정에서 LLM은 다양한 키워드를 사용하여 주제에 맞는 텍스트를 생성하고, 이를 통해 모델의 학습 성능을 향상시킵니다. 데이터 증강은 특히 불균형한 데이터셋에서 유용합니다.

3. **모델 선택 (Model Selection)**: LLM은 주어진 데이터셋의 특성과 요구 사항에 맞는 최적의 AD 모델을 추천합니다. 전통적인 모델 선택 방법은 과거 성능 데이터에 의존하는 경우가 많지만, LLM은 사전 훈련된 지식을 바탕으로 새로운 데이터셋에 적합한 모델을 제안할 수 있습니다. 이 과정은 수작업의 부담을 줄이고, 도메인 전문 지식이 부족한 상황에서도 유용합니다.

이러한 방법론은 LLM의 강력한 언어 이해 능력을 활용하여 이상 탐지의 여러 측면을 개선하고, 실질적인 응용 가능성을 높입니다. 특히, LLM 기반의 제로샷 탐지와 데이터 증강은 기존의 전통적인 방법들보다 더 나은 성능을 보여주며, 모델 선택 과정에서도 효과적인 결과를 도출할 수 있음을 보여줍니다.

### English Version

This paper proposes a methodology for anomaly detection (AD) utilizing large language models (LLMs). The methodology consists of three main tasks: zero-shot detection, data augmentation, and model selection.

1. **Zero-shot Detection**: LLMs leverage their pre-trained knowledge to detect anomalies without additional task-specific training. In this process, the LLM determines whether a given text sample belongs to a normal category or an anomaly category. This approach can effectively operate even in situations with limited labeled data.

2. **Data Augmentation**: LLMs generate new synthetic data based on existing normal samples to address the issue of insufficient training data. In this process, LLMs use various keywords to create contextually relevant text, thereby enhancing the model's learning performance. Data augmentation is particularly useful in imbalanced datasets.

3. **Model Selection**: LLMs recommend the optimal AD model that aligns with the characteristics and requirements of the given dataset. Traditional model selection methods often rely on historical performance data, but LLMs can suggest suitable models based on their pre-trained knowledge, reducing the burden of manual effort and being useful in situations lacking domain expertise.

These methodologies improve various aspects of anomaly detection by leveraging the powerful language understanding capabilities of LLMs, enhancing practical applicability. In particular, LLM-based zero-shot detection and data augmentation demonstrate superior performance compared to traditional methods, and effective results can be achieved in the model selection process as well.


<br/>
# Results



이 논문에서는 AD-LLM이라는 새로운 벤치마크를 소개하며, 대규모 언어 모델(LLM)이 자연어 처리(NLP)에서 이상 탐지(AD) 작업에 어떻게 기여할 수 있는지를 평가합니다. 연구는 세 가지 주요 작업에 초점을 맞추고 있습니다: 

1. **제로샷 탐지**: LLM의 사전 훈련된 지식을 활용하여 특정 작업에 대한 추가 훈련 없이 이상을 탐지하는 방법입니다. 실험 결과, LLM은 기존의 훈련 기반 AD 알고리즘보다 우수한 성능을 보였습니다. 예를 들어, GPT-4o와 DeepSeek-V3는 여러 데이터셋에서 높은 AUROC(Receiver Operating Characteristic Curve 아래 면적) 및 AUPRC(Precision-Recall Curve 아래 면적) 점수를 기록했습니다.

2. **데이터 증강**: LLM을 사용하여 제한된 데이터셋을 보완하기 위해 합성 데이터를 생성하는 방법입니다. 연구에서는 LLM이 생성한 합성 데이터가 여러 AD 모델의 성능을 향상시키는 데 효과적임을 보여주었습니다. 특히, AE, ECOD, LUNAR와 같은 모델은 합성 데이터를 포함했을 때 AUROC와 AUPRC에서 상당한 개선을 보였습니다.

3. **모델 선택**: LLM을 활용하여 주어진 데이터셋에 적합한 AD 모델을 추천하는 방법입니다. LLM이 추천한 모델은 평균적으로 기존의 베이스라인 모델보다 우수한 성능을 보였으며, 이는 LLM이 데이터셋의 특성과 모델의 강점을 분석하여 적합한 모델을 선택할 수 있는 가능성을 보여줍니다.

이 연구의 결과는 LLM이 제로샷 AD, 데이터 증강, 모델 선택의 세 가지 핵심 작업에서 유망한 성능을 발휘할 수 있음을 시사합니다. 그러나 모델 선택에 대한 설명이 일반적이고 데이터셋 특성과의 연관성이 부족하다는 한계도 지적되었습니다. 향후 연구는 이러한 설명의 투명성을 높이고, LLM의 성능을 더욱 향상시키기 위한 방향으로 나아가야 할 것입니다.




This paper introduces a new benchmark called AD-LLM, evaluating how large language models (LLMs) can contribute to anomaly detection (AD) tasks in natural language processing (NLP). The research focuses on three key tasks:

1. **Zero-shot Detection**: This method utilizes the pre-trained knowledge of LLMs to detect anomalies without additional task-specific training. Experimental results show that LLMs outperform existing training-based AD algorithms. For instance, GPT-4o and DeepSeek-V3 achieved high AUROC (Area Under the Receiver Operating Characteristic Curve) and AUPRC (Area Under the Precision-Recall Curve) scores across multiple datasets.

2. **Data Augmentation**: This approach involves generating synthetic data using LLMs to supplement limited datasets. The study demonstrates that synthetic data generated by LLMs significantly enhances the performance of several AD models. Notably, models like AE, ECOD, and LUNAR showed substantial improvements in AUROC and AUPRC when synthetic data was included.

3. **Model Selection**: This task leverages LLMs to recommend suitable AD models for a given dataset. The models recommended by LLMs generally outperformed the average baseline models, indicating the potential of LLMs to analyze dataset characteristics and model strengths to make appropriate selections.

The findings of this research suggest that LLMs exhibit promising capabilities in zero-shot AD, data augmentation, and model selection. However, it is also noted that the justifications for model selection often lack specificity and connection to dataset characteristics. Future research should aim to enhance the transparency of these justifications and further improve the performance of LLMs in AD tasks.


<br/>
# 예제



이 논문에서는 LLM(대형 언어 모델)을 활용한 이상 탐지(Anomaly Detection, AD) 방법을 제안하고, 이를 평가하기 위한 AD-LLM이라는 벤치마크를 소개합니다. 이 벤치마크는 세 가지 주요 작업을 다룹니다: 제로샷 탐지, 데이터 증강, 모델 선택입니다.

1. **제로샷 탐지**: 
   - **목표**: 주어진 테스트 데이터에서 이상 샘플을 식별하는 것입니다. 
   - **입력**: 테스트 데이터는 정상 샘플과 이상 샘플로 구성됩니다. 예를 들어, "이 주의 TravelWatch 칼럼은 호주 레드 센터의 아보리진 소유 투어 회사인 Anangu Tours를 소개합니다."라는 텍스트가 주어질 수 있습니다.
   - **출력**: LLM은 이 텍스트가 이상인지 아닌지를 판단하고, 그 이유를 설명하며, 신뢰도 점수를 부여합니다. 예를 들어, JSON 형식으로 `{"reason": "이 텍스트는 여행과 관련이 있으므로 이상이 아닙니다.", "anomaly_score": 0.2}`와 같은 결과를 반환할 수 있습니다.

2. **데이터 증강**:
   - **목표**: 제한된 훈련 데이터에서 추가 샘플을 생성하여 모델의 성능을 향상시키는 것입니다.
   - **입력**: 정상 샘플로 구성된 작은 훈련 세트가 주어집니다. 예를 들어, "이것은 스포츠에 관한 기사입니다."라는 텍스트가 있을 수 있습니다.
   - **출력**: LLM은 이 텍스트를 기반으로 새로운 샘플을 생성합니다. 예를 들어, "스포츠는 다양한 활동을 포함하며, 축구, 농구, 야구 등이 있습니다."와 같은 새로운 텍스트를 생성할 수 있습니다.

3. **모델 선택**:
   - **목표**: 주어진 데이터셋에 가장 적합한 AD 모델을 선택하는 것입니다.
   - **입력**: 데이터셋의 특성과 후보 모델에 대한 설명이 주어집니다. 예를 들어, "AG 뉴스 데이터셋은 다양한 뉴스 카테고리로 구성되어 있으며, 이상 카테고리는 스팸입니다."라는 정보가 있을 수 있습니다.
   - **출력**: LLM은 가장 적합한 모델을 추천하고 그 이유를 설명합니다. 예를 들어, `{"reason": "AG 뉴스 데이터셋은 텍스트 기반이므로 BERT + LOF 모델이 적합합니다.", "choice": "BERT + LOF"}`와 같은 결과를 반환할 수 있습니다.

이러한 작업을 통해 LLM이 이상 탐지에서 어떻게 활용될 수 있는지를 평가하고, 향후 연구 방향을 제시합니다.

---




This paper proposes a method for Anomaly Detection (AD) using Large Language Models (LLMs) and introduces a benchmark called AD-LLM for evaluation. This benchmark addresses three main tasks: zero-shot detection, data augmentation, and model selection.

1. **Zero-shot Detection**:
   - **Objective**: To identify anomalous samples from a given test dataset.
   - **Input**: The test data consists of normal and anomalous samples. For example, a text like "This week's TravelWatch column profiles Anangu Tours, an Aborigine owned tour company in Australia’s Red Center." may be provided.
   - **Output**: The LLM determines whether this text is anomalous or not, explains its reasoning, and assigns a confidence score. For instance, it might return a result in JSON format like `{"reason": "This text relates to travel, so it is not anomalous.", "anomaly_score": 0.2}`.

2. **Data Augmentation**:
   - **Objective**: To generate additional samples from limited training data to improve model performance.
   - **Input**: A small training set consisting of normal samples is provided. For example, a text like "This is an article about sports." may be given.
   - **Output**: The LLM generates new samples based on this text. For instance, it might produce a new text like "Sports encompass various activities, including soccer, basketball, and baseball."

3. **Model Selection**:
   - **Objective**: To select the most suitable AD model for the given dataset.
   - **Input**: Information about the dataset's characteristics and descriptions of candidate models is provided. For example, "The AG News dataset consists of various news categories, with the anomaly category being spam." may be included.
   - **Output**: The LLM recommends the most appropriate model and explains its reasoning. For example, it might return `{"reason": "The AG News dataset is text-based, so the BERT + LOF model is suitable.", "choice": "BERT + LOF"}`.

Through these tasks, the paper evaluates how LLMs can be utilized in anomaly detection and suggests future research directions.

<br/>
# 요약
이 논문에서는 대규모 언어 모델(LLM)을 활용한 이상 탐지(AD) 방법을 제안하고, 세 가지 주요 작업인 제로샷 탐지, 데이터 증강, 모델 선택을 평가하는 AD-LLM 벤치마크를 소개합니다. 실험 결과, LLM은 제로샷 AD에서 기존 방법보다 우수한 성능을 보였으며, LLM 기반의 데이터 증강이 AD 성능을 향상시키는 데 효과적임을 확인했습니다. 또한, LLM을 활용한 모델 선택이 기존 성능 기준에 근접하는 결과를 보여주었으나, 선택의 해석 가능성은 여전히 개선이 필요하다는 점을 강조합니다.

---

This paper introduces the AD-LLM benchmark for evaluating anomaly detection (AD) methods using large language models (LLMs), focusing on three key tasks: zero-shot detection, data augmentation, and model selection. Experimental results show that LLMs outperform traditional methods in zero-shot AD, and LLM-based data augmentation effectively enhances AD performance. Additionally, LLM-driven model selection approaches the performance of existing baselines, although the interpretability of the selections still requires improvement.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: AD-LLM의 세 가지 주요 작업(제로샷 탐지, 데이터 증강, 모델 선택)을 시각적으로 설명합니다. 이 다이어그램은 LLM이 각 작업에서 어떻게 기여하는지를 보여주며, 각 작업의 상호작용을 강조합니다. 특히, LLM이 제로샷 탐지에서 어떻게 작동하는지, 데이터 증강을 통해 어떻게 모델 성능을 향상시키는지, 그리고 모델 선택에서 어떻게 적합한 모델을 추천하는지를 명확히 나타냅니다.
   - **Figure 2**: LLM이 생성한 합성 데이터가 다양한 AD 모델의 성능에 미치는 영향을 보여줍니다. 이 피규어는 합성 데이터가 특정 모델에서 성능을 크게 향상시키는 반면, 다른 모델에서는 성능 저하를 초래할 수 있음을 시각적으로 나타냅니다. 이는 LLM 기반 합성 데이터 생성의 효과성을 강조합니다.

2. **테이블**
   - **Table 1**: LLM 기반 탐지기와 기존 방법 간의 성능 비교를 보여줍니다. 이 표는 LLM이 제로샷 AD에서 기존 방법보다 우수한 성능을 발휘함을 나타내며, 특히 GPT-4o와 DeepSeek-V3가 여러 데이터셋에서 높은 AUROC 및 AUPRC 점수를 기록했습니다.
   - **Table 2**: LLM 기반 탐지기의 성능이 카테고리 설명을 추가했을 때 어떻게 향상되는지를 보여줍니다. 이 표는 카테고리 설명이 LLM의 탐지 성능을 개선하는 데 중요한 역할을 한다는 것을 강조합니다.
   - **Table A6**: LLM 생성 합성 데이터가 AD 성능에 미치는 영향을 보여주는 결과를 포함합니다. 이 표는 합성 데이터가 특정 모델에서 성능을 크게 향상시키는 반면, 다른 모델에서는 성능 저하를 초래할 수 있음을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에서는 실험 설정, 데이터셋 세부정보, 프롬프트 설계 및 오류 분석에 대한 추가 정보를 제공합니다. 이 정보는 연구의 재현성을 높이고, LLM 기반 AD의 성능을 향상시키기 위한 다양한 접근 방식을 이해하는 데 도움을 줍니다. 특히, LLM의 오류 유형과 그 원인에 대한 분석은 향후 연구 방향을 제시합니다.




1. **Diagrams and Figures**
   - **Figure 1**: Visually explains the three main tasks of AD-LLM (zero-shot detection, data augmentation, model selection). This diagram illustrates how LLM contributes to each task and emphasizes the interactions between them. It particularly highlights how LLM operates in zero-shot detection, enhances model performance through data augmentation, and recommends suitable models in model selection.
   - **Figure 2**: Shows the impact of LLM-generated synthetic data on the performance of various AD models. This figure visually represents that synthetic data significantly improves performance in certain models while potentially degrading performance in others, emphasizing the effectiveness of LLM-based synthetic data generation.

2. **Tables**
   - **Table 1**: Displays the performance comparison between LLM-based detectors and traditional methods. This table indicates that LLMs outperform traditional methods in zero-shot AD, with GPT-4o and DeepSeek-V3 achieving high AUROC and AUPRC scores across multiple datasets.
   - **Table 2**: Shows how the performance of LLM-based detectors improves when category descriptions are added. This table emphasizes the critical role of category descriptions in enhancing the detection performance of LLMs.
   - **Table A6**: Contains results showing the impact of LLM-generated synthetic data on AD performance. This table indicates that synthetic data significantly enhances performance in certain models while potentially degrading it in others.

3. **Appendix**
   - The appendix provides additional information on experimental settings, dataset details, prompt design, and error analysis. This information enhances the reproducibility of the research and aids in understanding various approaches to improve the performance of LLM-based AD. Notably, the analysis of error types and their causes in LLMs suggests future research directions.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{yang2025adllm,
  author = {Tiankai Yang and Yi Nian and Shawn Li and Ruiyao Xu and Yuangang Li and Jiaqi Li and Zhuo Xiao and Xiyang Hu and Ryan Rossi and Kaize Ding and Xia Hu and Yue Zhao},
  title = {AD-LLM: Benchmarking Large Language Models for Anomaly Detection},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages = {1524--1547},
  year = {2025},
  month = {July},
  publisher = {Association for Computational Linguistics},
  url = {https://github.com/USC-FORTIS/AD-LLM}
}
```

### 시카고 스타일 인용
Tiankai Yang, Yi Nian, Shawn Li, Ruiyao Xu, Yuangang Li, Jiaqi Li, Zhuo Xiao, Xiyang Hu, Ryan Rossi, Kaize Ding, Xia Hu, and Yue Zhao. "AD-LLM: Benchmarking Large Language Models for Anomaly Detection." In *Findings of the Association for Computational Linguistics: ACL 2025*, 1524–1547. July 27 - August 1, 2025. Association for Computational Linguistics. https://github.com/USC-FORTIS/AD-LLM.
