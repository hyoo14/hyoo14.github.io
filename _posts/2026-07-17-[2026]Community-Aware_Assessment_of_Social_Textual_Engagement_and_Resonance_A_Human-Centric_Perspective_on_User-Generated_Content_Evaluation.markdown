---
layout: post
title:  "[2026]Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation"
date:   2026-07-17 03:00:34 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 사용자 생성 콘텐츠(UGC)의 품질 평가를 위해 CASTER(Community-Aware Assessment of Social Textual Engagement and Resonance)라는 새로운 접근 방식을 제안하고, MEDEA(Multimodal Engagement-Driven Evaluation Architecture)를 통해 사회적 사고(Social-CoT)를 활용하여 커뮤니티의 반응을 시뮬레이션한다.  
(Social Chain-of-Thought, Social-CoT 경로(다양한 시청자 페르소나를 생성하고, 이들의 반응을 기반으로 품질을 평가)와 좋아요를 사용..)  


짧은 요약(Abstract) :


이 논문에서는 전통적인 비디오 품질 평가(Video Quality Assessment, VQA)가 사용자 생성 콘텐츠(User-Generated Content, UGC)의 품질을 정의하는 복잡한 사회적 역학을 간과하고 있다는 점을 지적합니다. 저자들은 신호 중심의 메트릭에서 인간 중심의 공명 평가로 패러다임 전환을 제안하며, CASTER(Community-Aware Assessment of Social Textual Engagement and Resonance)라는 새로운 작업을 소개합니다. 이 작업은 UGC 항목이 시각적 품질뿐만 아니라 다중 모달 속성을 기반으로 긍정적인 커뮤니티 공명을 달성했는지를 평가합니다. 이를 위해 MEDEA(Multimodal Engagement-Driven Evaluation Architecture)를 제안하며, 이는 사회적 사고(Social Chain-of-Thought, Social-CoT) 메커니즘을 도입하여 다양한 시청자 페르소나를 시뮬레이션하고 집단적 인지 및 감정 반응을 모사합니다. MEDEA는 두 단계의 접근 방식을 통해 훈련되며, 실험 결과 MEDEA가 CASTER-Bench에서 최첨단 기준을 크게 초월하는 성능을 보임을 입증합니다.



This paper points out that traditional Video Quality Assessment (VQA) overlooks the complex social dynamics that define quality in User-Generated Content (UGC). The authors propose a paradigm shift from signal-centric metrics to human-centric resonance assessment, introducing a new task called CASTER (Community-Aware Assessment of Social Textual Engagement and Resonance). This task evaluates whether a UGC item achieves positive community resonance based on its multimodal attributes rather than visual quality alone. To address this, they present MEDEA (Multimodal Engagement-Driven Evaluation Architecture), which introduces a Social Chain-of-Thought (Social-CoT) mechanism to simulate diverse viewer personas and collective cognitive and emotional reactions. MEDEA is trained via a two-stage approach, and experimental results demonstrate that it significantly outperforms state-of-the-art baselines on CASTER-Bench.


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



이 논문에서는 사용자 생성 콘텐츠(UGC)의 품질 평가를 위한 새로운 접근 방식을 제안합니다. 기존의 비디오 품질 평가(VQA) 방법은 주로 미적 충실도와 기술적 왜곡에 초점을 맞추고 있지만, UGC의 품질은 사회적 동적 요소와 사용자 경험에 의해 결정된다는 점을 강조합니다. 이를 위해 저자들은 CASTER(Community-Aware Assessment of Social Textual Engagement and Resonance)라는 새로운 작업을 도입하고, 이를 지원하기 위해 MEDEA(Multimodal Engagement-Driven Evaluation Architecture)라는 모델을 개발했습니다.

#### 모델 아키텍처
MEDEA는 세 가지 주요 단계로 구성됩니다:

1. **사회적 사고 경로(Social-CoT) 구축**: MEDEA는 커뮤니티 반응을 기반으로 다양한 시청자 페르소나를 생성하여, UGC 항목에 대한 공감적 반응 경로를 시뮬레이션합니다. 이를 위해, 사용자 댓글을 수집하고, 가장 많이 좋아요를 받은 댓글을 필터링하여 핵심 반응을 선택합니다.

2. **지도 학습(Supervised Fine-Tuning)**: MEDEA는 수집된 데이터와 전문가 주석을 결합하여 모델을 훈련합니다. 이 단계에서는 모델이 시각적 신호와 텍스트 메타데이터를 사회적 해석과 정렬하도록 학습합니다. 이를 통해 모델은 반응 경로를 생성한 후 품질 레이블을 예측하는 구조적 추론 과정을 내재화합니다.

3. **프로세스-감독 강화 학습(Process-Supervised Reinforcement Learning)**: 이 단계에서는 사회적 정렬 보상(Social Alignment Reward)을 도입하여 모델의 사회적 추론 과정을 개선합니다. 이 보상은 생성된 경로가 실제 사용자 댓글과 얼마나 유사한지를 측정하여, 모델이 진정한 인간의 감정 표현을 모방하도록 유도합니다.

#### 훈련 데이터
MEDEA는 대규모의 비주얼 및 텍스트 메타데이터를 포함한 UGC 항목을 사용하여 훈련됩니다. 이 데이터는 다양한 콘텐츠 카테고리에서 수집되며, 각 항목은 전문가에 의해 주석이 달려 있습니다. 주석은 제작 품질, 정보 유용성, 감정적 공명, 내러티브 우수성 등 여러 차원에서 이루어집니다.

MEDEA는 이러한 과정을 통해 UGC의 품질을 평가하는 데 있어 단순한 시각적 특성 분석을 넘어, 사회적 수용과 감정적 반응을 반영하는 모델로 발전합니다. 실험 결과, MEDEA는 기존의 최첨단 모델들보다 우수한 성능을 보이며, 커뮤니티 피드백과 일치하는 해석 가능한 추론 경로를 제공합니다.

---




This paper proposes a new approach for assessing the quality of user-generated content (UGC). Traditional Video Quality Assessment (VQA) methods primarily focus on aesthetic fidelity and technical distortions, but the quality of UGC is emphasized as being determined by social dynamics and user experiences. To address this, the authors introduce a new task called CASTER (Community-Aware Assessment of Social Textual Engagement and Resonance) and develop a model called MEDEA (Multimodal Engagement-Driven Evaluation Architecture) to support it.

#### Model Architecture
MEDEA consists of three main stages:

1. **Social Chain-of-Thought (Social-CoT) Construction**: MEDEA simulates diverse viewer personas by generating empathetic reaction paths based on community responses. To achieve this, user comments are collected, and the most liked comments are filtered to select core reactions.

2. **Supervised Fine-Tuning**: MEDEA trains the model by combining the collected data with expert annotations. In this stage, the model learns to align visual signals and textual metadata with social interpretations. This allows the model to internalize a structured reasoning process that generates reaction paths before predicting quality labels.

3. **Process-Supervised Reinforcement Learning**: In this stage, a Social Alignment Reward is introduced to improve the model's social reasoning process. This reward measures how similar the generated paths are to actual user comments, encouraging the model to mimic genuine human emotional expressions.

#### Training Data
MEDEA is trained using UGC items that include large-scale visual and textual metadata. This data is collected from various content categories and is annotated by experts. The annotations cover multiple dimensions, including production quality, informational utility, emotional resonance, and narrative excellence.

Through this process, MEDEA evolves into a model that assesses UGC quality by reflecting not just on visual characteristics but also on social acceptance and emotional responses. Experimental results show that MEDEA significantly outperforms existing state-of-the-art models while providing interpretable reasoning paths that align with community feedback.


<br/>
# Results



이 논문에서는 사용자 생성 콘텐츠(UGC)의 품질 평가를 위한 새로운 접근 방식을 제안합니다. 기존의 비디오 품질 평가(VQA) 방법들은 주로 미적 충실도와 기술적 왜곡에 초점을 맞추었으나, UGC의 품질은 사회적 동적 요소와 사용자 경험에 의해 결정된다는 점을 강조합니다. 이를 위해 저자들은 CASTER(Community-Aware Assessment of Social Textual Engagement and Resonance)라는 새로운 평가 작업을 도입하고, MEDEA(Multimodal Engagement-Driven Evaluation Architecture)라는 프레임워크를 제안합니다.

#### 실험 설정
- **테스트 데이터**: CASTER-Bench라는 새로운 벤치마크를 사용하여 1,485개의 UGC 항목을 평가합니다. 이 데이터셋은 다양한 콘텐츠 카테고리를 포함하고 있으며, 전문가에 의해 주관적으로 평가된 품질 레이블을 포함합니다.
- **경쟁 모델**: MEDEA는 여러 기존 VQA 모델 및 대형 멀티모달 모델과 비교됩니다. 여기에는 FastVQA, DOVER, MaxVQA, Q-Align, FineVQ, VQA2와 같은 전통적인 VQA 방법과 Qwen3-VL-Plus, GPT-5.2, Claude-4.5-opus와 같은 최신 대형 멀티모달 모델이 포함됩니다.
  
#### 성능 메트릭
- **정밀도(Precision)**: 모델이 고품질 콘텐츠로 분류한 항목 중 실제로 고품질인 항목의 비율.
- **재현율(Recall)**: 실제 고품질 콘텐츠 중 모델이 올바르게 분류한 항목의 비율.
- **F1 점수**: 정밀도와 재현율의 조화 평균으로, 모델의 전반적인 성능을 평가하는 데 사용됩니다.

#### 결과
MEDEA는 모든 경쟁 모델에 비해 우수한 성능을 보였습니다. 특히, MEDEA는 고품질 클래스에서 F1 점수 0.650을 달성하여 가장 강력한 기준선 모델을 크게 초과했습니다. MEDEA는 정밀도 0.603과 재현율 0.705를 기록하여, 실제 사용자 피드백과 잘 일치하는 해석 가능한 경로를 제공했습니다. 

기존의 전통적인 VQA 방법들은 낮은 F1 점수를 기록하며, 사회적 수용을 반영하는 데 실패했습니다. 반면, MEDEA는 사회적 정렬 보상을 통해 사용자 반응을 시뮬레이션하고, 다양한 관점을 반영하여 품질 평가를 수행했습니다. 이러한 결과는 MEDEA가 UGC의 품질을 평가하는 데 있어 더 나은 접근 방식을 제공함을 보여줍니다.




This paper proposes a new approach for assessing the quality of user-generated content (UGC). Traditional video quality assessment (VQA) methods have primarily focused on aesthetic fidelity and technical distortions, but the authors emphasize that the quality of UGC is determined by social dynamics and user experiences. To address this, they introduce a new evaluation task called CASTER (Community-Aware Assessment of Social Textual Engagement and Resonance) and propose a framework named MEDEA (Multimodal Engagement-Driven Evaluation Architecture).

#### Experimental Setup
- **Test Data**: The authors utilize a new benchmark called CASTER-Bench, which consists of 1,485 UGC items. This dataset covers various content categories and includes quality labels that have been subjectively evaluated by experts.
- **Competing Models**: MEDEA is compared against several existing VQA models and large multimodal models. This includes traditional VQA methods such as FastVQA, DOVER, MaxVQA, Q-Align, FineVQ, and VQA2, as well as state-of-the-art large multimodal models like Qwen3-VL-Plus, GPT-5.2, and Claude-4.5-opus.

#### Performance Metrics
- **Precision**: The proportion of items classified as high-quality by the model that are actually high-quality.
- **Recall**: The proportion of actual high-quality content that the model correctly classifies.
- **F1 Score**: The harmonic mean of precision and recall, used to evaluate the overall performance of the model.

#### Results
MEDEA significantly outperformed all competing models. Specifically, it achieved an F1 score of 0.650 on the high-quality class, surpassing the strongest baseline model by a large margin. MEDEA recorded a precision of 0.603 and a recall of 0.705, providing interpretable reasoning paths that align well with actual community feedback.

In contrast, traditional VQA methods recorded low F1 scores and failed to reflect social acceptance. MEDEA, through its Social Alignment Reward, simulates user reactions and incorporates diverse perspectives into its quality assessment. These results demonstrate that MEDEA offers a better approach for evaluating the quality of UGC.


<br/>
# 예제



이 논문에서는 사용자 생성 콘텐츠(UGC)의 품질을 평가하기 위한 새로운 접근 방식을 제안합니다. 기존의 비디오 품질 평가(VQA) 방법은 주로 기술적 완벽성과 미적 요소에 초점을 맞추었으나, UGC의 품질은 커뮤니티의 공감과 감정적 반응에 의해 결정된다고 주장합니다. 이를 위해 연구자들은 CASTER(Community-Aware Assessment of Social Textual Engagement and Resonance)라는 새로운 작업을 정의하고, MEDEA(Multimodal Engagement-Driven Evaluation Architecture)라는 평가 프레임워크를 제안합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 비디오의 커버 이미지, 7개의 주요 프레임, 제목, 태그, 자동 음성 인식(ASR) 텍스트, 카테고리 메타데이터 등.
   - **출력**: 각 비디오에 대한 품질 레이블(예: Excellent, Good, Average, Poor)과 함께, 커뮤니티의 반응을 시뮬레이션한 다양한 댓글.

2. **테스트 데이터**:
   - **입력**: 트레이닝 데이터와 유사한 형식으로, 새로운 비디오의 커버 이미지, 주요 프레임, 제목, 태그, ASR 텍스트 등.
   - **출력**: MEDEA 모델이 생성한 댓글과 함께, 최종 품질 판단(High-Quality 또는 Low-Quality).

#### 구체적인 작업(Task)

- **CASTER 작업**: 주어진 UGC 항목이 커뮤니티와 긍정적으로 공감하는지를 평가하는 작업. 이 작업은 비디오의 시각적 품질뿐만 아니라, 감정적 참여, 정보의 유용성, 내러티브의 일관성 등을 고려합니다.
- **MEDEA 프레임워크**: 다양한 관점에서 커뮤니티의 반응을 시뮬레이션하여 최종 품질 판단을 내리는 구조. 이 과정에서 Social Chain-of-Thought(Social-CoT) 메커니즘을 사용하여 다양한 시청자 페르소나를 생성하고, 이들의 반응을 기반으로 품질을 평가합니다.




This paper proposes a new approach to assessing the quality of user-generated content (UGC). Traditional Video Quality Assessment (VQA) methods primarily focus on technical perfection and aesthetic elements, but the authors argue that the quality of UGC is determined by community resonance and emotional responses. To address this, they define a new task called CASTER (Community-Aware Assessment of Social Textual Engagement and Resonance) and propose an evaluation framework called MEDEA (Multimodal Engagement-Driven Evaluation Architecture).

#### Training Data and Test Data

1. **Training Data**:
   - **Input**: Cover image of the video, 7 key frames, title, tags, automatic speech recognition (ASR) text, category metadata, etc.
   - **Output**: Quality labels for each video (e.g., Excellent, Good, Average, Poor) along with various comments simulating community reactions.

2. **Test Data**:
   - **Input**: Similar format to the training data, containing cover image, key frames, title, tags, ASR text of new videos.
   - **Output**: Comments generated by the MEDEA model, along with the final quality judgment (High-Quality or Low-Quality).

#### Specific Task

- **CASTER Task**: A task that evaluates whether a given UGC item resonates positively with the community. This task considers not only the visual quality of the video but also emotional engagement, informational utility, and narrative coherence.
- **MEDEA Framework**: A structure that simulates community reactions from diverse perspectives to derive a final quality judgment. In this process, the Social Chain-of-Thought (Social-CoT) mechanism is used to generate various viewer personas and assess quality based on their reactions.

<br/>
# 요약
이 논문에서는 사용자 생성 콘텐츠(UGC)의 품질 평가를 위해 CASTER(Community-Aware Assessment of Social Textual Engagement and Resonance)라는 새로운 접근 방식을 제안하고, MEDEA(Multimodal Engagement-Driven Evaluation Architecture)를 통해 사회적 사고(Social-CoT)를 활용하여 커뮤니티의 반응을 시뮬레이션한다. 실험 결과, MEDEA는 기존의 최첨단 방법들보다 우수한 성능을 보였으며, CASTER-Bench라는 새로운 벤치마크를 통해 다양한 UGC 카테고리에 대한 평가를 지원한다. 예를 들어, MEDEA는 고품질 콘텐츠를 정확하게 식별하는 데 있어 높은 정밀도와 재현율을 달성하였다.

---

This paper proposes a new approach for assessing the quality of user-generated content (UGC) through CASTER (Community-Aware Assessment of Social Textual Engagement and Resonance) and utilizes MEDEA (Multimodal Engagement-Driven Evaluation Architecture) to simulate community responses via Social Chain-of-Thought (Social-CoT). Experimental results show that MEDEA significantly outperforms existing state-of-the-art methods, supported by the introduction of CASTER-Bench, a new benchmark for evaluating various UGC categories. For instance, MEDEA achieved high precision and recall in accurately identifying high-quality content.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **다이어그램**: MEDEA 프레임워크의 구조를 보여주는 다이어그램은 Social-CoT 경로를 생성하고, 커뮤니티 반응을 수집하며, 최종 품질 판단을 내리는 과정을 시각적으로 설명합니다. 이 구조는 MEDEA가 어떻게 다양한 사용자 관점을 시뮬레이션하고 이를 통해 품질 평가를 수행하는지를 명확히 보여줍니다.
  
- **피규어**: CASTER-Bench의 카테고리 분포와 대표적인 UGC 예시를 보여주는 피규어는 다양한 콘텐츠 유형을 포함하고 있으며, 각 카테고리의 특성을 강조합니다. 이는 연구의 범위와 다양성을 강조하는 데 기여합니다.

#### 2. 테이블
- **성능 비교 테이블**: MEDEA와 기존의 다양한 VQA 방법들 간의 성능을 비교한 테이블은 MEDEA가 모든 기준에서 우수한 성능을 보였음을 나타냅니다. 특히, High-Quality 클래스에서 MEDEA는 F1 점수 0.650을 기록하며, 가장 강력한 기준선보다 큰 차이를 보였습니다. 이는 MEDEA가 커뮤니티의 진정한 품질을 인식하는 데 효과적임을 시사합니다.

- **Ablation Study 테이블**: MEDEA의 각 구성 요소가 성능에 미치는 영향을 분석한 테이블은 Social-CoT와 Social Alignment Reward의 중요성을 강조합니다. 이 연구는 Social-CoT가 없을 경우 성능이 크게 저하된다는 것을 보여주며, Social Alignment Reward가 없을 경우에는 생성된 경로가 반복적이고 비인간적인 결과를 초래함을 나타냅니다.

#### 3. 어펜딕스
- **어펜딕스 A**: MEDEA의 계산 비용과 토큰 소비를 비교한 데이터는 MEDEA가 높은 품질의 판단을 위해 더 많은 토큰을 생성해야 함을 보여줍니다. 이는 MEDEA의 깊이 있는 분석이 단순한 예측보다 더 나은 성능을 발휘하는 데 기여함을 나타냅니다.

- **어펜딕스 D**: 생성된 추론의 신뢰성과 다양성을 평가한 결과는 Social Alignment Reward가 신뢰성과 다양성을 모두 향상시킨다는 것을 보여줍니다. 이는 MEDEA가 보다 구조적이고 사회적으로 일관된 추론을 생성하는 데 기여함을 나타냅니다.




#### 1. Diagrams and Figures
- **Diagrams**: The diagram illustrating the structure of the MEDEA framework visually explains the process of generating Social-CoT paths, collecting community reactions, and making final quality judgments. This structure clearly shows how MEDEA simulates various user perspectives to perform quality assessments.

- **Figures**: The figures showing the category distribution of CASTER-Bench and representative UGC examples highlight the diversity of content types included in the study, emphasizing the scope and variety of the research.

#### 2. Tables
- **Performance Comparison Table**: The table comparing the performance of MEDEA with various existing VQA methods indicates that MEDEA outperformed all benchmarks. Notably, it achieved an F1 score of 0.650 on the High-Quality class, significantly surpassing the strongest baseline. This suggests that MEDEA is effective in recognizing the true quality endorsed by the community.

- **Ablation Study Table**: The table analyzing the impact of each component of MEDEA on performance emphasizes the importance of Social-CoT and Social Alignment Reward. The study shows that without Social-CoT, performance drops significantly, and without Social Alignment Reward, the generated paths become repetitive and robotic.

#### 3. Appendix
- **Appendix A**: Data comparing the computational costs and token consumption of MEDEA shows that it generates more tokens to achieve high-quality judgments. This indicates that the depth of analysis provided by MEDEA contributes to better performance than simple predictions.

- **Appendix D**: The evaluation of the faithfulness and diversity of generated reasoning indicates that the Social Alignment Reward enhances both aspects. This suggests that MEDEA contributes to producing more structured and socially coherent reasoning.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Li2026,
  author    = {Tianjiao Li and Kai Zhao and Xiang Li and Yang Liu and Huyang Sun},
  title     = {Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {21578--21600},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Tianjiao Li, Kai Zhao, Xiang Li, Yang Liu, and Huyang Sun. "Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 21578–21600. Association for Computational Linguistics, 2026.
