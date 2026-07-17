---
layout: post
title:  "[2026]LAMAR-2 at MedGenVidQA 2026: Visual Answer Localization in Medical Videos via Multimodal LLM and Context-Augmented Prompting"
date:   2026-07-17 02:57:46 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 의료 비디오에서 시각적 답변을 정확하게 로컬라이즈하기 위해 다중 모달 생성 파이프라인을 제안합니다.


짧은 요약(Abstract) :



이 논문은 MedGenVidQA 데이터셋을 사용하여 연속적인 의료 비디오 내에서 시각적 답변을 지역화하는 접근 방식을 제시합니다. 우리는 시각적 답변 지역화를 다중 모달 융합 문제로 설정하고, 원시 비디오, 타임스탬프가 있는 ASR(자동 음성 인식) 전사, VLM(비디오 언어 모델)에서 생성된 장면 설명을 구조화된 맥락 블록으로 통합하여 모델이 말로 된 해설과 관찰 가능한 물리적 사건을 교차 참조할 수 있도록 합니다. 우리는 모델이 오디오 전사를 관찰 가능한 시각적 움직임과 함께 보조 힌트로 처리하도록 강제하는 목표 지향적 안내가 기본 접근 방식을 크게 초월하는 성능을 발휘한다는 것을 보여줍니다. 이 방법은 테스트 리더보드에서 최첨단 성능을 달성하며, mIoU(Mean Intersection over Union) 점수는 79.55에 달하고, IoU@0.3, IoU@0.5, IoU@0.7 점수는 각각 93.75, 90.00, 77.50입니다. 우리의 발견은 다중 모달 맥락 융합과 목표 지향적 안내를 결합하는 것이 텍스트 편향을 극복하는 데 효과적임을 강조하며, 의료 분야에서 요구되는 미세 수준의 정밀도를 달성하기 위한 유망한 접근 방식을 확립합니다. 우리는 GitHub에 코드를 공개합니다.




This paper presents an approach to localizing visual answers within continuous medical videos using a multi-step multimodal generation pipeline with the MedGenVidQA dataset. We frame visual answer localization as a multimodal fusion problem, integrating raw video, timestamped ASR transcripts, and VLM-generated scene descriptions into structured contextual blocks, enabling the model to cross-reference spoken commentary against observable physical events. We show that targeted guidance, which forces the model to treat audio transcripts as supplementary hints with observable visual movements, significantly outperforms baseline approaches. It achieves state-of-the-art performance on the test leaderboard, yielding an mIoU of 79.55, alongside IoU@0.3, IoU@0.5, and IoU@0.7 scores of 93.75, 90.00, and 77.50, respectively. Our findings highlight the effectiveness of combining multimodal context fusion with targeted guidance to overcome text bias, establishing a promising approach for achieving the micro-level precision required in the medical domain. We release our code on GitHub.


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



이 논문에서 제안하는 메써드는 의료 비디오에서 시각적 답변을 로컬라이징하기 위한 다단계 다중 모달 생성 파이프라인입니다. 이 접근법은 MedGenVidQA 데이터셋을 활용하여 시각적 답변 로컬라이징을 다중 모달 융합 문제로 프레임화합니다. 이 과정에서 원시 비디오, 타임스탬프가 있는 ASR(자동 음성 인식) 전사본, 그리고 VLM(비주얼 언어 모델)에서 생성된 장면 설명을 통합하여 구조화된 맥락 블록을 생성합니다. 이를 통해 모델은 음성 해설과 관찰 가능한 물리적 사건을 교차 참조할 수 있습니다.

주요 기법으로는 다음과 같은 요소들이 포함됩니다:

1. **다중 모달 융합**: 원시 비디오와 ASR 전사본, VLM 생성 장면 설명을 통합하여 모델이 다양한 정보를 활용할 수 있도록 합니다.
2. **타겟 가이던스**: 모델이 음성 전사본을 보조 힌트로 취급하도록 유도하여, 관찰 가능한 시각적 움직임에 중점을 두게 합니다. 이로 인해 기존의 방법들보다 성능이 크게 향상됩니다.
3. **정확한 타임스탬프 예측**: 모델은 주어진 의료 비디오와 임상 질문을 바탕으로 관련 행동의 시작 및 종료 타임스탬프를 출력합니다. 이를 통해 의료 절차의 정확한 경계를 식별할 수 있습니다.
4. **맥락 강화**: LLM(대형 언어 모델)의 입력을 구조화된 다중 출처 맥락으로 풍부하게 하여, 단일 모달리티만으로는 얻을 수 없는 정확한 로컬라이징을 달성합니다.

이러한 방법론을 통해 논문은 의료 비디오 질문 응답에서의 시각적 답변 로컬라이징의 정확성을 크게 향상시켰으며, 테스트 리더보드에서 최첨단 성능을 달성했습니다.




The method proposed in this paper is a multi-step multimodal generation pipeline for localizing visual answers within medical videos. This approach frames visual answer localization as a multimodal fusion problem, utilizing the MedGenVidQA dataset. It integrates raw video, timestamped ASR (Automatic Speech Recognition) transcripts, and VLM (Visual Language Model)-generated scene descriptions into structured contextual blocks, enabling the model to cross-reference spoken commentary against observable physical events.

Key components of the method include:

1. **Multimodal Fusion**: The integration of raw video, ASR transcripts, and VLM-generated scene descriptions allows the model to leverage diverse information sources.
2. **Targeted Guidance**: The model is guided to treat audio transcripts as supplementary hints, focusing on observable visual movements. This significantly improves performance compared to baseline approaches.
3. **Accurate Timestamp Prediction**: The model outputs the start and end timestamps of relevant actions based on the provided medical video and clinical query, allowing for precise identification of clinical task boundaries.
4. **Context Augmentation**: By enriching the input to the LLM (Large Language Model) with structured, multi-source context, the method achieves more accurate localization than relying on any single modality alone.

Through this methodology, the paper demonstrates a significant enhancement in the accuracy of visual answer localization in medical video question answering, achieving state-of-the-art performance on the test leaderboard.


<br/>
# Results



이 논문에서는 MedGenVidQA 2026 대회에서의 성과를 바탕으로, 의료 비디오에서 시각적 답변을 정확하게 로컬라이징하는 방법을 제안합니다. 연구팀은 다양한 모델과 접근 방식을 비교하여 최종적으로 가장 높은 성능을 기록한 모델을 도출했습니다.

1. **경쟁 모델**: 연구팀은 여러 모델을 비교했습니다. 특히, Video-Only 모델이 가장 높은 mIoU(Mean Intersection over Union) 점수인 79.51을 기록했습니다. 이는 비디오만을 사용하여 시각적 정보를 기반으로 한 결과입니다. 반면, VLM-Enhanced Context 모델은 Heuristic Context(Loose) 전략을 사용하여 mIoU 79.55, IoU@0.3 93.75, IoU@0.5 90.00, IoU@0.7 77.50을 기록했습니다. 이 모델은 비디오와 함께 텍스트 정보를 결합하여 시각적 증거를 보완하는 방식으로 성능을 향상시켰습니다.

2. **테스트 데이터**: 연구팀은 MedGenVidQA 2026 Task C의 공식 테스트 데이터셋을 사용했습니다. 이 데이터셋은 80개의 의료 질문과 65개의 고유한 교육 비디오로 구성되어 있으며, 평균 비디오 길이는 544.72초입니다. 공식 테스트 데이터의 정답은 공개되지 않았으나, 연구팀은 수동으로 주어진 질문-답변 쌍을 주석 처리하여 성능을 평가했습니다.

3. **메트릭**: 성능 평가는 IoU(Intersection over Union) 메트릭을 사용하여 이루어졌습니다. IoU는 예측된 시간 간격과 실제 정답 간의 겹치는 비율을 측정합니다. 연구팀은 IoU@0.3, IoU@0.5, IoU@0.7의 세 가지 임계값에서 정확도를 보고했습니다. 이 메트릭은 모델의 시각적 로컬라이징 능력을 평가하는 데 중요한 역할을 합니다.

4. **비교**: 연구팀은 다양한 접근 방식을 비교하여 최적의 성능을 도출했습니다. 예를 들어, RAG(검색 증강 생성) 기반 접근 방식은 성능이 낮았고, Transcript-Video Fusion 방식이 가장 높은 mIoU를 기록했습니다. 그러나 VLM-Enhanced Context 모델은 Heuristic Context 전략을 통해 비디오-전용 모델보다 약간 높은 성능을 보였으며, 이는 텍스트 정보를 보조적으로 활용하여 시각적 증거를 강화한 결과입니다.

결론적으로, 연구팀은 텍스트 기반의 정보와 시각적 정보를 결합하여 의료 비디오에서의 시각적 답변 로컬라이징의 정확성을 높이는 방법을 제안하였으며, 이는 의료 분야에서의 질문-답변 시스템의 발전에 기여할 것으로 기대됩니다.

---




This paper presents the results based on the performance in the MedGenVidQA 2026 competition, proposing a method for accurately localizing visual answers in medical videos. The research team compared various models and approaches to derive the one with the highest performance.

1. **Competing Models**: The research team compared several models. Notably, the Video-Only model achieved the highest mIoU (Mean Intersection over Union) score of 79.51, relying solely on visual information from the video. In contrast, the VLM-Enhanced Context model, using the Heuristic Context (Loose) strategy, recorded an mIoU of 79.55, IoU@0.3 of 93.75, IoU@0.5 of 90.00, and IoU@0.7 of 77.50. This model improved performance by combining video with textual information to supplement visual evidence.

2. **Test Data**: The research team utilized the official test dataset from the MedGenVidQA 2026 Task C. This dataset consists of 80 medical questions assigned to 65 unique instructional videos, with an average video length of 544.72 seconds. Although the official test ground truth was withheld, the research team manually annotated the provided question-answer pairs to evaluate performance.

3. **Metrics**: Performance evaluation was conducted using the IoU (Intersection over Union) metric. IoU measures the overlap ratio between the predicted time intervals and the actual ground truth. The research team reported accuracy at three IoU thresholds: IoU@0.3, IoU@0.5, and IoU@0.7. This metric plays a crucial role in assessing the model's visual localization capabilities.

4. **Comparison**: The research team compared various approaches to derive optimal performance. For instance, RAG (Retrieval-Augmented Generation) based approaches performed poorly, while the Transcript-Video Fusion method achieved the highest mIoU. However, the VLM-Enhanced Context model slightly outperformed the Video-Only model using the Heuristic Context strategy, which enhanced visual evidence by utilizing textual information as a supplementary resource.

In conclusion, the research team proposed a method that combines text-based information with visual data to improve the accuracy of visual answer localization in medical videos, which is expected to contribute to advancements in question-answering systems in the medical field.


<br/>
# 예제



이 논문에서는 의료 비디오에서 시각적 답변을 로컬라이징하는 방법을 제안합니다. 연구의 주요 목표는 주어진 의료 비디오와 자연어 임상 질문에 대해 정확한 시작 및 종료 타임스탬프를 예측하는 것입니다. 이를 위해 MedGenVidQA 데이터셋을 사용하여 다단계 다중 모달 생성 파이프라인을 구축하였습니다.

#### 1. 데이터셋
- **트레이닝 데이터**: MedVidQA 데이터셋은 900개의 건강 관련 비디오에서 3,010개의 인간 주석 QA 쌍으로 구성되어 있습니다. 이 연구에서는 49개의 고유 비디오와 148개의 QA 쌍을 사용했습니다.
- **테스트 데이터**: 공식 MedGenVidQA 2026 Task C 테스트 데이터셋은 80개의 의료 질문과 65개의 고유한 교육 비디오로 구성되어 있습니다.

#### 2. 입력 및 출력
- **입력**: 각 비디오와 관련된 자연어 질문, ASR(자동 음성 인식) 전사본, VLM(비디오 언어 모델)에서 생성된 장면 설명이 포함됩니다.
- **출력**: 모델은 질문에 대한 정확한 시작 및 종료 타임스탬프를 JSON 형식으로 출력합니다. 예를 들어:
  ```json
  {
    "reasoning": "이 시각적 장면이 질문에 대한 답변을 제공하는 이유.",
    "answer_start": "MM:SS",
    "answer_end": "MM:SS"
  }
  ```

#### 3. 태스크
- **주요 태스크**: 주어진 비디오에서 질문에 대한 정확한 시각적 답변을 로컬라이징하는 것입니다. 모델은 비디오의 시각적 증거를 우선시하고, 오디오 정보는 보조적인 역할을 하도록 설계되었습니다.
- **모델 아키텍처**: Gemini-3-Flash 모델을 사용하여 최종 타임스탬프를 예측하며, 다양한 입력 구성(예: 비디오만, 전사본만, 비디오와 전사본 결합 등)을 평가합니다.

#### 4. 평가 방법
- **성능 측정**: 예측된 타임스탬프와 실제 타임스탬프 간의 교차 겹침 비율(Intersection over Union, IoU)을 사용하여 성능을 평가합니다. IoU@0.3, IoU@0.5, IoU@0.7의 세 가지 기준으로 정확도를 측정합니다.

이러한 방식으로, 연구팀은 의료 비디오에서 시각적 답변을 정확하게 로컬라이징하는 데 필요한 다양한 접근 방식을 실험하고, 최적의 성능을 달성하기 위한 전략을 개발하였습니다.

---




This paper presents a method for localizing visual answers in medical videos. The main goal of the research is to predict the precise start and end timestamps for a given medical video and natural language clinical query. To achieve this, a multi-step multimodal generation pipeline is constructed using the MedGenVidQA dataset.

#### 1. Datasets
- **Training Data**: The MedVidQA dataset consists of 3,010 human-annotated QA pairs from 900 health-related videos. In this study, 49 unique videos and 148 QA pairs were utilized.
- **Test Data**: The official MedGenVidQA 2026 Task C test dataset contains 80 medical questions assigned to 65 unique instructional videos.

#### 2. Input and Output
- **Input**: Each video is accompanied by a natural language question, an ASR (Automatic Speech Recognition) transcript, and VLM (Video Language Model)-generated scene descriptions.
- **Output**: The model outputs the precise start and end timestamps for the question in JSON format. For example:
  ```json
  {
    "reasoning": "The reason this visual scene answers the question.",
    "answer_start": "MM:SS",
    "answer_end": "MM:SS"
  }
  ```

#### 3. Task
- **Main Task**: The task is to localize the precise visual answer to the question within the given video. The model is designed to prioritize visual evidence from the video, with audio information serving a supplementary role.
- **Model Architecture**: The Gemini-3-Flash model is used to predict the final timestamps, evaluating various input configurations (e.g., video only, transcript only, combination of video and transcript).

#### 4. Evaluation Method
- **Performance Measurement**: The performance is evaluated using the Intersection over Union (IoU) between the predicted timestamps and the ground truth timestamps. Accuracy is measured at three thresholds: IoU@0.3, IoU@0.5, and IoU@0.7.

In this way, the research team experiments with various approaches necessary for accurately localizing visual answers in medical videos and develops strategies to achieve optimal performance.

<br/>
# 요약
이 논문에서는 의료 비디오에서 시각적 답변을 정확하게 로컬라이즈하기 위해 다중 모달 생성 파이프라인을 제안합니다. 실험 결과, 시각적 증거를 우선시하는 "Heuristic Context" 가이드를 사용하여 mIoU 79.55 및 IoU@0.5 90.00을 달성했습니다. 이 접근법은 텍스트 기반 맥락을 실제 물리적 실행과 분리하여 미세한 시간 정밀도를 확보하는 데 효과적임을 보여줍니다.

---

This paper presents a multimodal generation pipeline for accurately localizing visual answers in medical videos. The results demonstrate that using a "Heuristic Context" guidance that prioritizes visual evidence achieves an mIoU of 79.55 and an IoU@0.5 of 90.00. This approach effectively decouples text-based context from actual physical execution, ensuring micro-level temporal precision.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - 논문에서는 VLM-Enhanced Context 파이프라인의 개요를 보여주는 다이어그램이 포함되어 있습니다. 이 다이어그램은 비디오 분석을 위한 입력 및 출력 흐름을 시각적으로 설명하며, 각 단계에서 어떤 데이터가 처리되는지를 명확히 나타냅니다. 이를 통해 독자는 제안된 방법론의 구조와 작동 방식을 쉽게 이해할 수 있습니다.

2. **테이블**:
   - 성능 비교를 위한 여러 테이블이 포함되어 있습니다. 이 테이블들은 다양한 파이프라인 구성의 mIoU, IoU@0.3, IoU@0.5, IoU@0.7 점수를 비교하여 각 접근 방식의 효과를 명확히 보여줍니다. 예를 들어, VLM-Enhanced Context의 Heuristic Context(Loose) 전략이 mIoU 79.55로 가장 높은 성능을 기록했음을 보여줍니다. 이는 텍스트 기반의 맥락을 보조적인 힌트로 활용하면서도 시각적 증거에 기반한 경계 예측을 강화한 결과입니다.

3. **어펜딕스**:
   - 어펜딕스에서는 실험에 사용된 다양한 프롬프트 템플릿이 제공됩니다. 이는 각기 다른 제약 기반 가이던스 전략을 설명하며, 모델이 어떻게 시각적 증거를 우선시하고, 텍스트를 보조적으로 활용하는지를 보여줍니다. 이러한 템플릿은 연구자들이 향후 연구에서 유사한 접근 방식을 적용할 수 있도록 돕는 중요한 자료입니다.

### Insights and Results

1. **Diagrams and Figures**:
   - The paper includes a diagram that outlines the VLM-Enhanced Context pipeline. This diagram visually explains the flow of input and output for video analysis, clearly indicating what data is processed at each stage. This helps readers easily understand the structure and functioning of the proposed methodology.

2. **Tables**:
   - Several tables are included for performance comparison. These tables compare the mIoU, IoU@0.3, IoU@0.5, and IoU@0.7 scores of various pipeline configurations, clearly demonstrating the effectiveness of each approach. For instance, the Heuristic Context (Loose) strategy of the VLM-Enhanced Context achieved the highest performance with an mIoU of 79.55. This result highlights the success of leveraging textual context as supplementary hints while enhancing boundary predictions based on visual evidence.

3. **Appendix**:
   - The appendix provides various prompt templates used in the experiments. These templates describe different constraint-based guidance strategies and illustrate how the model prioritizes visual evidence while using text as a supportive tool. Such templates serve as valuable resources for researchers looking to apply similar approaches in future studies.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Sermsrisuwan2026,
  author    = {Watcharitpol Sermsrisuwan and Nopporn Lekuthai and Seksan Yoadsanit and Titipat Achakulvisut},
  title     = {LAMAR-2 at MedGenVidQA 2026: Visual Answer Localization in Medical Videos via Multimodal LLM and Context-Augmented Prompting},
  booktitle = {Proceedings of the BioNLP 2026 (Shared Tasks)},
  pages     = {233--242},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},


}
```

### 시카고 스타일

Sermsrisuwan, Watcharitpol, Nopporn Lekuthai, Seksan Yoadsanit, and Titipat Achakulvisut. "LAMAR-2 at MedGenVidQA 2026: Visual Answer Localization in Medical Videos via Multimodal LLM and Context-Augmented Prompting." In *Proceedings of the BioNLP 2026 (Shared Tasks)*, 233–242. 
Association for Computational Linguistics, July 2026.   
