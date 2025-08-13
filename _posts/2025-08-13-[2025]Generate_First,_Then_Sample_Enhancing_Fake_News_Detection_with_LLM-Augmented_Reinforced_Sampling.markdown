---
layout: post
title:  "[2025]Generate First, Then Sample: Enhancing Fake News Detection with LLM-Augmented Reinforced Sampling"
date:   2025-08-13 17:18:13 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

이 논문에서는 GSFND(Generate First, Then Sample for Fake News Detection)라는 새로운 프레임워크를 제안
LLM을 사용하여 세 가지 다른 스타일의 가짜 뉴스를 생성하고 이를 훈련 세트에 포함시켜 가짜 뉴스의 표현을 보강
강화 학습을 적용하여 가짜 뉴스의 최적 비율을 동적으로 샘플링하여 모델이 효과적인 가짜 뉴스 탐지기를 학습


짧은 요약(Abstract) :

이 논문의 초록에서는 온라인 플랫폼에서의 가짜 뉴스 확산이 오랫동안 심각한 문제로 여겨져 왔음을 언급하고 있습니다. 이에 따라 가짜 뉴스 탐지기를 개발하기 위한 많은 노력이 있었지만, 이러한 모델들이 실제 뉴스에 비해 가짜 뉴스를 식별하는 데 20% 이상 낮은 성능을 보이는 주요 단점이 있다고 지적합니다. 이는 데이터셋의 불균형과 모델이 목표 플랫폼의 데이터 분포를 충분히 이해하지 못하기 때문일 수 있습니다. 이 연구에서는 가짜 뉴스 탐지 모델의 효과성을 개선하기 위해, 먼저 대형 언어 모델(LLM)을 사용하여 세 가지 다른 스타일의 가짜 뉴스를 생성하고 이를 훈련 세트에 포함시켜 가짜 뉴스의 표현을 보강합니다. 이후 강화 학습을 적용하여 가짜 뉴스의 최적 비율을 동적으로 샘플링하여 모델이 효과적인 가짜 뉴스 탐지기를 학습할 수 있도록 합니다. 이 접근 방식은 주석이 제한된 뉴스 데이터로도 모델이 효과적으로 작동할 수 있게 하며, 다양한 플랫폼에서 탐지 정확도를 지속적으로 향상시킵니다. 실험 결과, 이 방법이 두 개의 벤치마크 데이터셋에서 최첨단 성능을 달성하며, 각각 24.02%와 11.06%의 가짜 뉴스 탐지 성능 향상을 보여주었다고 보고합니다.


The abstract of this paper discusses the long-standing concern regarding the spread of fake news on online platforms. It notes that, despite extensive efforts to develop fake news detectors, these models exhibit a significant drawback: they lag by more than 20% in identifying fake news compared to real news. This gap is likely attributed to an imbalance in the dataset and the model's inadequate understanding of the data distribution on the targeted platform. The study focuses on improving the effectiveness of fake news detection models by first using a large language model (LLM) to generate fake news in three different styles, which are then incorporated into the training set to augment the representation of fake news. Subsequently, reinforcement learning is applied to dynamically sample fake news, allowing the model to learn the optimal real-to-fake news ratio for training an effective fake news detector. This approach enables the model to perform effectively even with a limited amount of annotated news data and consistently improves detection accuracy across different platforms. Experimental results demonstrate that this method achieves state-of-the-art performance on two benchmark datasets, improving fake news detection performance by 24.02% and 11.06%, respectively.


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


이 논문에서 제안하는 GSFND(Generate First, Then Sample for Fake News Detection) 모델은 가짜 뉴스 탐지를 위한 혁신적인 접근 방식을 제시합니다. 이 모델은 크게 세 가지 주요 구성 요소로 나뉩니다: 다각적 가짜 뉴스 생성, 강화 샘플링, 그리고 가짜 뉴스 탐지기입니다.

1. **다각적 가짜 뉴스 생성**: 
   GSFND의 첫 번째 단계는 대형 언어 모델(LLM)을 활용하여 다양한 스타일로 가짜 뉴스를 생성하는 것입니다. 이 과정에서 원본 가짜 뉴스를 바탕으로 세 가지 스타일(재작성, 확장, 변장)로 변형하여 데이터셋의 가짜 뉴스 샘플을 세 배로 늘립니다. 이를 통해 모델은 더 다양한 가짜 뉴스의 특성을 학습할 수 있게 됩니다.

2. **강화 샘플링**: 
   두 번째 단계에서는 강화 학습(RL)을 사용하여 훈련 과정에서 실제 뉴스와 가짜 뉴스의 최적 비율을 동적으로 학습합니다. 이 과정은 마르코프 결정 과정(MDP)으로 모델링되며, DQN(Deep Q-Network)을 통해 샘플링 비율을 조정합니다. 에이전트는 현재 상태를 관찰하고, 가능한 행동 중 하나를 선택하여 샘플링 비율을 조정합니다. 이때, 에이전트는 가짜 뉴스 탐지기의 성능을 극대화하는 방향으로 샘플링 비율을 최적화합니다.

3. **가짜 뉴스 탐지기**: 
   마지막으로, 가짜 뉴스 탐지기는 입력된 뉴스 텍스트를 분류하여 진짜인지 가짜인지를 판단합니다. 이 탐지기는 BERT 모델을 기반으로 하여 뉴스의 특징 벡터를 생성하고, 이를 통해 두 클래스(진짜 또는 가짜)에 대한 확률을 출력합니다. 탐지기의 성능은 강화 샘플링 과정에서 에이전트의 보상 신호로 사용되어, 모델의 학습을 더욱 효과적으로 만듭니다.

이러한 세 가지 구성 요소는 서로 상호작용하며, GSFND는 가짜 뉴스 탐지의 정확성을 높이고, 다양한 뉴스 플랫폼에서의 적용 가능성을 향상시킵니다. 실험 결과, GSFND는 두 개의 벤치마크 데이터셋에서 최첨단 성능을 달성하였으며, 가짜 뉴스 탐지 성능을 각각 24.02%와 11.06% 향상시켰습니다.



The GSFND (Generate First, Then Sample for Fake News Detection) model proposed in this paper presents an innovative approach to fake news detection. This model is divided into three main components: Multi-Perspective Fake News Generation, Reinforced Sampling, and the Fake News Detector.

1. **Multi-Perspective Fake News Generation**: 
   The first step of GSFND involves utilizing a Large Language Model (LLM) to generate fake news in various styles. In this process, the original fake news is transformed into three different styles (rewrite, expand, disguise), effectively tripling the number of fake news samples in the dataset. This allows the model to learn more diverse characteristics of fake news.

2. **Reinforced Sampling**: 
   The second step employs Reinforcement Learning (RL) to dynamically learn the optimal ratio of real to fake news during the training process. This process is modeled as a Markov Decision Process (MDP) and uses Deep Q-Networks (DQN) to adjust the sampling ratio. The agent observes the current state and selects one of the possible actions to adjust the sampling ratio. The agent aims to optimize the sampling ratio in a way that maximizes the performance of the fake news detector.

3. **Fake News Detector**: 
   Finally, the Fake News Detector classifies the input news text to determine whether it is real or fake. This detector is based on a BERT model, which generates a feature vector for the news and outputs probabilities for the two classes (real or fake). The performance of the detector serves as a feedback signal for the RL agent during the Reinforced Sampling process, enhancing the model's learning effectiveness.

These three components interact with each other, allowing GSFND to improve the accuracy of fake news detection and enhance its applicability across various news platforms. Experimental results demonstrate that GSFND achieves state-of-the-art performance on two benchmark datasets, improving fake news detection performance by 24.02% and 11.06%, respectively.


<br/>
# Results


이 논문에서는 GSFND(Generate First, Then Sample for Fake News Detection)라는 새로운 가짜 뉴스 탐지 모델을 제안하고, 이를 기존의 여러 경쟁 모델과 비교하여 성능을 평가하였습니다. GSFND는 대형 언어 모델(LLM)을 활용하여 다양한 스타일의 가짜 뉴스를 생성하고, 강화 학습을 통해 실제 뉴스와 가짜 뉴스의 최적 비율을 동적으로 학습하는 방식으로 작동합니다.

#### 실험 데이터셋
모델의 성능은 두 가지 주요 데이터셋인 Weibo21과 GossipCop에서 평가되었습니다. Weibo21 데이터셋은 중국어 뉴스로 구성되어 있으며, GossipCop 데이터셋은 영어 뉴스로 구성되어 있습니다. 각 데이터셋은 훈련, 검증, 테스트 세트로 나뉘어 있으며, 가짜 뉴스와 실제 뉴스의 비율은 검증 및 테스트 세트에서 1:1로 설정되었습니다.

#### 성능 메트릭
모델의 성능은 주로 매크로 F1 점수(mac F1)와 정확도(Acc.)를 사용하여 평가되었습니다. 또한, 각 클래스(실제 뉴스와 가짜 뉴스)에 대한 F1 점수(F1-real, F1-fake)도 별도로 보고되었습니다. 이러한 메트릭은 모델이 실제 뉴스와 가짜 뉴스를 얼마나 잘 구분하는지를 평가하는 데 중요한 역할을 합니다.

#### 결과 비교
GSFND는 Weibo21 데이터셋에서 0.880의 매크로 F1 점수와 0.899의 정확도를 기록하였으며, GossipCop 데이터셋에서는 각각 0.914와 0.925를 기록했습니다. 이는 기존의 여러 경쟁 모델에 비해 현저히 높은 성능입니다. 예를 들어, GossipCop 데이터셋에서 GSFND는 F1-fake 점수를 0.883으로 기록하여, 이전 최고 성능 모델인 LLM-GAN보다 24.02% 향상된 결과를 보였습니다. Weibo21 데이터셋에서도 F1-fake 점수가 0.884로, LLM-GAN보다 11.06% 향상되었습니다.

이러한 결과는 GSFND가 가짜 뉴스 탐지에서 기존 모델들보다 더 효과적임을 보여주며, 특히 가짜 뉴스 탐지 성능을 크게 향상시켰습니다. GSFND는 가짜 뉴스와 실제 뉴스 모두에 대해 균형 잡힌 탐지 능력을 보여주었으며, 이는 실제 뉴스 플랫폼에서의 적용 가능성을 높이는 데 기여합니다.



This paper introduces a novel fake news detection model called GSFND (Generate First, Then Sample for Fake News Detection) and evaluates its performance against several existing competitive models. GSFND operates by leveraging a large language model (LLM) to generate fake news in various styles and employs reinforcement learning to dynamically learn the optimal ratio of real to fake news during training.

#### Experimental Datasets
The model's performance was evaluated on two primary datasets: Weibo21 and GossipCop. The Weibo21 dataset consists of Chinese news, while the GossipCop dataset comprises English news. Each dataset was split into training, validation, and test sets, with a fixed real-to-fake ratio of 1:1 for the validation and test sets.

#### Performance Metrics
The model's performance was primarily assessed using the macro F1 score (mac F1) and accuracy (Acc.). Additionally, the F1 scores for each class (real news and fake news) were reported separately (F1-real, F1-fake). These metrics play a crucial role in evaluating how well the model distinguishes between real and fake news.

#### Results Comparison
GSFND achieved a macro F1 score of 0.880 and an accuracy of 0.899 on the Weibo21 dataset, while on the GossipCop dataset, it recorded scores of 0.914 and 0.925, respectively. These results significantly outperform those of several existing competitive models. For instance, on the GossipCop dataset, GSFND achieved an F1-fake score of 0.883, which represents a 24.02% improvement over the previous best-performing model, LLM-GAN. On the Weibo21 dataset, the F1-fake score was 0.884, showing an 11.06% improvement over LLM-GAN.

These results demonstrate that GSFND is more effective in detecting fake news compared to existing models, particularly in enhancing fake news detection performance. GSFND exhibited a balanced ability to identify both real and fake news, contributing to its applicability in real-world news platforms.


<br/>
# 예제


이 논문에서는 GSFND(Generate First, Then Sample for Fake News Detection)라는 새로운 프레임워크를 제안하여 가짜 뉴스 탐지의 성능을 향상시키고자 합니다. 이 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: LLM(대형 언어 모델)을 사용한 가짜 뉴스 생성과 강화 학습을 통한 샘플링 비율 최적화입니다.

#### 1. 트레이닝 데이터 생성
- **입력**: 원본 가짜 뉴스 기사
- **출력**: 다양한 스타일로 생성된 가짜 뉴스
  - **예시**: 
    - 원본 뉴스: "Alicia Silverstone와 Christopher Jarecki가 20년의 결혼 생활 끝에 이혼을 발표했다."
    - 생성된 뉴스(재작성): "Alicia Silverstone가 남편 Christopher Jarecki와의 결혼 생활을 마감하고 이혼 소송을 제기했다."
    - 생성된 뉴스(확장): "Alicia Silverstone는 남편 Christopher Jarecki와의 결혼 생활을 마감하고 이혼 소송을 제기했다. 이들은 20년 이상 함께한 커플로, 7세 아들을 두고 있다."
    - 생성된 뉴스(변장): "최근 보도에 따르면, Alicia Silverstone가 남편 Christopher Jarecki와의 결혼 생활을 끝내기로 결정했다."

이렇게 생성된 가짜 뉴스는 원본 데이터셋에 추가되어 모델의 학습에 사용됩니다.

#### 2. 테스트 데이터
- **입력**: 테스트 데이터셋의 뉴스 기사
- **출력**: 뉴스의 진위 여부(진짜 또는 가짜)
  - **예시**: 
    - 테스트 뉴스: "Alicia Silverstone와 Christopher Jarecki가 이혼을 발표했다."
    - 모델의 출력: "가짜 뉴스" (모델이 이 뉴스를 가짜로 분류함)

#### 3. 강화 학습을 통한 샘플링 비율 최적화
- **입력**: 현재 상태(진짜 뉴스와 가짜 뉴스의 비율, 주제 분포 등)
- **출력**: 최적의 진짜-가짜 뉴스 비율
  - **예시**: 
    - 현재 상태: 진짜 뉴스 60%, 가짜 뉴스 40%
    - 모델의 출력: "진짜 뉴스 50%, 가짜 뉴스 50%로 조정"

이러한 방식으로 GSFND는 가짜 뉴스 탐지의 성능을 향상시키기 위해 다양한 스타일의 가짜 뉴스를 생성하고, 강화 학습을 통해 최적의 샘플링 비율을 학습합니다.

---


This paper proposes a novel framework called GSFND (Generate First, Then Sample for Fake News Detection) to enhance the performance of fake news detection. The framework consists of two main components: generating fake news using a Large Language Model (LLM) and optimizing the sampling ratio through Reinforcement Learning (RL).

#### 1. Training Data Generation
- **Input**: Original fake news articles
- **Output**: Fake news generated in various styles
  - **Example**: 
    - Original News: "Alicia Silverstone and Christopher Jarecki announced their divorce after 20 years of marriage."
    - Generated News (Rewritten): "Alicia Silverstone has filed for divorce from her husband Christopher Jarecki."
    - Generated News (Expanded): "Alicia Silverstone has filed for divorce from her husband Christopher Jarecki. They have been a couple for over 20 years and have a 7-year-old son."
    - Generated News (Disguised): "According to reports, Alicia Silverstone has decided to end her marriage to Christopher Jarecki."

The generated fake news is then added to the original dataset for model training.

#### 2. Test Data
- **Input**: News articles from the test dataset
- **Output**: Authenticity label (real or fake)
  - **Example**: 
    - Test News: "Alicia Silverstone and Christopher Jarecki announced their divorce."
    - Model Output: "Fake News" (the model classifies this news as fake)

#### 3. Sampling Ratio Optimization through Reinforcement Learning
- **Input**: Current state (ratio of real to fake news, topic distribution, etc.)
- **Output**: Optimal real-to-fake news ratio
  - **Example**: 
    - Current State: 60% real news, 40% fake news
    - Model Output: "Adjust to 50% real news, 50% fake news"

In this way, GSFND enhances fake news detection performance by generating diverse styles of fake news and learning the optimal sampling ratio through reinforcement learning.

<br/>
# 요약
이 논문에서는 GSFND라는 새로운 프레임워크를 제안하여, 대형 언어 모델(LLM)을 활용해 다양한 스타일의 가짜 뉴스를 생성하고, 강화 학습을 통해 실제 뉴스와 가짜 뉴스의 최적 비율을 동적으로 학습하여 가짜 뉴스 탐지 성능을 향상시킵니다. 실험 결과, GSFND는 GossipCop과 Weibo21 데이터셋에서 각각 24.02%와 11.06%의 F1 점수 향상을 달성하며, 기존 방법들보다 우수한 성능을 보였습니다. 예를 들어, LLM을 통해 생성된 가짜 뉴스는 모델이 더 다양한 특징을 학습하는 데 기여하여 탐지 정확도를 높였습니다.

---

In this paper, a novel framework called GSFND is proposed, which utilizes a large language model (LLM) to generate fake news in various styles and employs reinforcement learning to dynamically learn the optimal ratio of real to fake news, enhancing fake news detection performance. Experimental results show that GSFND achieves improvements of 24.02% and 11.06% in F1 scores on the GossipCop and Weibo21 datasets, respectively, outperforming existing methods. For instance, the fake news generated by the LLM contributes to the model's ability to learn more diverse features, thereby increasing detection accuracy.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **GSFND 아키텍처 다이어그램**: GSFND의 전체 구조를 보여주는 다이어그램은 세 가지 주요 구성 요소(다양한 스타일의 가짜 뉴스 생성, 강화 샘플링, 가짜 뉴스 탐지기)를 시각적으로 설명합니다. 이 구조는 각 구성 요소가 어떻게 상호작용하여 가짜 뉴스 탐지 성능을 향상시키는지를 명확히 보여줍니다.
   - **F1 점수 비교 그래프**: GSFND와 기존 모델 간의 F1 점수 비교를 통해 GSFND가 가짜 뉴스 탐지에서 어떻게 성능을 향상시켰는지를 시각적으로 나타냅니다. GSFND는 가짜 뉴스에 대해 24.02%의 성능 향상을 보여주며, 이는 기존 모델들과의 성능 차이를 명확히 드러냅니다.

2. **테이블**
   - **성능 비교 테이블**: Weibo21 및 GossipCop 데이터셋에서 GSFND와 여러 기준 모델의 성능을 비교한 테이블은 GSFND가 모든 메트릭에서 최고 성능을 달성했음을 보여줍니다. 특히, 가짜 뉴스 탐지에서 F1 점수가 24.02% 향상된 것은 GSFND의 효과성을 강조합니다.
   - **데이터셋 통계 테이블**: Weibo21 및 GossipCop 데이터셋의 통계는 각 데이터셋의 실제 뉴스와 가짜 뉴스의 비율을 보여줍니다. 이 통계는 데이터 불균형 문제를 이해하는 데 중요한 정보를 제공합니다.

3. **어펜딕스**
   - **어펜딕스 A.1**: 기준 모델의 세부 사항을 제공하여 GSFND와의 비교를 위한 배경 정보를 제공합니다. 이는 연구의 신뢰성을 높이는 데 기여합니다.
   - **어펜딕스 A.5**: 다양한 스타일로 생성된 가짜 뉴스의 예시를 제공하여 GSFND의 다각적인 접근 방식을 설명합니다. 이는 모델이 어떻게 다양한 스타일의 가짜 뉴스를 생성하여 데이터셋을 보강하는지를 보여줍니다.

### Insights
- GSFND는 가짜 뉴스 탐지에서 기존 모델들보다 현저히 높은 성능을 보여주며, 이는 LLM을 활용한 데이터 보강과 강화 학습을 통한 샘플링 비율 최적화의 결합 덕분입니다.
- 데이터셋의 불균형 문제를 해결하기 위해 LLM이 생성한 다양한 스타일의 가짜 뉴스를 포함함으로써, 모델이 더 많은 특징을 학습할 수 있도록 하였습니다.
- 강화 학습을 통해 동적으로 조정된 실제 뉴스와 가짜 뉴스의 비율은 모델이 특정 플랫폼의 데이터 분포에 더 잘 적응할 수 있도록 도와줍니다.

---



1. **Diagrams and Figures**
   - **GSFND Architecture Diagram**: The diagram illustrating the overall structure of GSFND visually explains the three main components (multi-perspective fake news generation, reinforced sampling, and fake news detector). This structure clearly shows how each component interacts to enhance fake news detection performance.
   - **F1 Score Comparison Graph**: The graph comparing F1 scores between GSFND and existing models visually represents how GSFND improves performance in fake news detection. GSFND shows a 24.02% performance improvement for fake news, highlighting the performance gap with existing models.

2. **Tables**
   - **Performance Comparison Table**: The table comparing GSFND with various baseline models on the Weibo21 and GossipCop datasets shows that GSFND achieves the highest performance across all metrics. Notably, the 24.02% improvement in the F1 score for fake news emphasizes the effectiveness of GSFND.
   - **Dataset Statistics Table**: The statistics of the Weibo21 and GossipCop datasets provide insights into the ratio of real news to fake news in each dataset. This information is crucial for understanding the data imbalance issue.

3. **Appendix**
   - **Appendix A.1**: Provides details of the baseline models, offering background information for comparison with GSFND. This contributes to the credibility of the research.
   - **Appendix A.5**: Offers examples of fake news generated in various styles, explaining GSFND's multifaceted approach. This demonstrates how the model generates diverse styles of fake news to augment the dataset.

### Insights
- GSFND demonstrates significantly higher performance in fake news detection compared to existing models, attributed to the combination of LLM-augmented data and reinforcement learning for sampling ratio optimization.
- By incorporating various styles of fake news generated by LLMs, the model addresses the data imbalance issue, allowing it to learn more distinctive features.
- The dynamically adjusted real-to-fake news ratio through reinforcement learning helps the model better adapt to the data distribution of specific platforms.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Tong2025,
  author    = {Zhao Tong and Yimeng Gu and Huidong Liu and Qiang Liu and Shu Wu and Haichao Shi and Xiao-Yu Zhang},
  title     = {Generate First, Then Sample: Enhancing Fake News Detection with LLM-Augmented Reinforced Sampling},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {24276--24290},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  address   = {Vancouver, Canada},
}
```

### 시카고 스타일

Zhao Tong, Yimeng Gu, Huidong Liu, Qiang Liu, Shu Wu, Haichao Shi, and Xiao-Yu Zhang. "Generate First, Then Sample: Enhancing Fake News Detection with LLM-Augmented Reinforced Sampling." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 24276–24290. Vancouver, Canada: Association for Computational Linguistics, 2025.
