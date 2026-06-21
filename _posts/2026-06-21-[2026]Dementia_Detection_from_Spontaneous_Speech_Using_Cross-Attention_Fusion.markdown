---
layout: post
title:  "[2026]Dementia Detection from Spontaneous Speech Using Cross-Attention Fusion"
date:   2026-06-21 08:51:29 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 알츠하이머병(AD) 탐지를 위해 텍스트, 오디오, 이미지 정보를 통합한 다중 모달 크로스 어텐션 프레임워크를 제안하였다.


짧은 요약(Abstract) :



이 연구는 알츠하이머병(AD)의 조기 발견을 위한 다중 모달 크로스 어텐션 융합 프레임워크를 제안합니다. 알츠하이머병은 노인들의 일상생활에 영향을 미치는 진행성 신경퇴행성 질환으로, 인지 능력과 언어 소통에 부정적인 영향을 미칩니다. 조기 발견은 적시 개입을 가능하게 하여 영향을 받는 사람들의 삶의 질을 향상시키는 데 중요합니다. 기존의 연구들은 대부분 단일 모달에 의존하고 있어 서로 보완적인 신호를 놓치고 있습니다. 본 연구에서는 텍스트, 음성, 이미지 정보를 통합하여 AD를 탐지하는 다중 모달 크로스 어텐션 프레임워크를 제시합니다. 이 프레임워크는 ADReSSo 2021 그림 설명 데이터셋을 사용하여 텍스트 데이터는 ModernBERT로 인코딩하고, 오디오 특징은 wav2vec 2.0-base-960으로 추출하며, Cookie Theft 이미지는 CLIP ViT-L/14를 통해 표현합니다. 이러한 임베딩은 공유 공간으로 선형 투영된 후 Transformer 기반의 크로스 어텐션을 통해 결합되어 AD 탐지를 위한 융합 벡터를 생성합니다. 결과적으로, trimodal 모델은 SVC 분류기와 함께 사용할 때 0.8732의 정확도와 0.8571의 F1 점수를 달성하여 단일 모달 및 이중 모달 구성보다 우수한 성능을 보였습니다. 해석 가능성을 위해 모달리티 기여도에 대한 민감도 분석을 수행한 결과, 텍스트가 주요 역할을 하며, 오디오는 보완적인 개선을 제공하고, 이미지는 적당한 안정적 맥락 지원을 제공하는 것으로 나타났습니다. 이러한 결과는 다중 모달 임베딩 융합 방법이 성능에 중요한 영향을 미친다는 것을 강조하며, 크로스 어텐션 블록이 정확성과 단순성 간의 효과적인 균형을 이루어 해석 가능한 다운스트림 분류기와 잘 정렬된 통합 표현을 생성함을 보여줍니다.

---




This study proposes a multimodal cross-attention fusion framework for the early detection of Alzheimer's disease (AD). Alzheimer's disease is a progressive neurodegenerative disorder that affects the daily lives of older adults, impacting their cognitive abilities and language communication. Early detection is crucial as it enables timely intervention and helps improve the quality of life for those affected. Most existing studies rely on unimodal approaches, missing complementary signals across modalities. In this work, we present a multimodal cross-attention framework that integrates lexical (text), acoustic (speech), and visual (image) information for dementia detection using the ADReSSo 2021 picture-description dataset. Within this framework, text data are encoded using ModernBERT, audio features are extracted using wav2vec 2.0-base-960, and the Cookie Theft image is represented through CLIP ViT-L/14. These embeddings are linearly projected to a shared space and then combined via Transformer-based cross-attention, yielding a fused vector for AD detection. Our results show that the trimodal model achieved the best overall performance when paired with an SVC classifier, reaching an accuracy of 0.8732 and an F1 score of 0.8571, surpassing both the top-performing unimodal and bimodal configurations. For interpretability, a sensitivity analysis of modality contributions reveals that text plays the primary role, audio provides complementary improvements, and image offers modest yet stabilizing contextual support. These results highlight that the method of multimodal embedding fusion significantly influences performance: a cross-attention block achieves an effective balance between accuracy and simplicity, producing integrated representations that align well with interpretable downstream classifiers.


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



이 연구에서는 알츠하이머병(AD) 탐지를 위한 다중 모달 크로스 어텐션(fusion) 프레임워크를 제안합니다. 이 프레임워크는 텍스트(어휘), 음성(음향), 이미지(시각) 정보를 통합하여 AD를 탐지하는 데 사용됩니다. 연구에 사용된 데이터셋은 ADReSSo 2021 그림 설명 데이터셋으로, 이 데이터셋은 임상적으로 진단된 AD 환자와 인지적으로 건강한 대조군이 '쿠키 도둑' 장면을 설명하는 237개의 오디오 녹음으로 구성되어 있습니다.

#### 모델 아키텍처
모델은 세 가지 모달리티(텍스트, 오디오, 이미지)를 통합하기 위해 Transformer 기반의 크로스 어텐션 메커니즘을 사용합니다. 각 모달리티는 다음과 같이 처리됩니다:

1. **텍스트**: ModernBERT를 사용하여 텍스트 데이터를 768차원 벡터로 인코딩합니다. 이 과정에서 mean-token pooling을 통해 각 녹음에 대해 하나의 벡터를 생성합니다.
   
2. **오디오**: wav2vec 2.0 모델을 사용하여 오디오 특징을 추출하고, 이 또한 768차원 벡터로 변환됩니다. 오디오 데이터는 16kHz의 샘플링 속도로 읽혀지며, 과도한 트리밍을 피하여 AD 스피치에서 중요한 데이터인 주저함이나 침묵 패턴을 보존합니다.

3. **이미지**: CLIP ViT-L/14 모델을 사용하여 '쿠키 도둑' 이미지를 768차원 벡터로 인코딩합니다. 이 이미지는 모든 참가자에게 공통적으로 사용되며, 각 녹음에 대한 맥락적 기초를 제공합니다.

이 세 가지 모달리티의 임베딩은 공통 차원으로 선형 변환된 후, 크로스 어텐션 메커니즘을 통해 결합되어 최종적으로 분류에 적합한 단일 벡터로 통합됩니다.

#### 트레이닝 및 검증 프로토콜
모델의 훈련 과정에서는 모든 인코더를 고정하고, 프로젝션 레이어와 크로스 어텐션 블록, 소규모 피드포워드 네트워크만을 훈련합니다. AdamW 옵티마이저를 사용하여 최적화하며, 훈련 세트에서 20%를 검증 세트로 분리하여 조기 중단 및 학습률 스케줄링을 수행합니다. 테스트 시, 최상의 검증 체크포인트를 복원하고, 선형 분류 헤드를 제거한 후, 표준 분류기를 사용하여 예측을 수행합니다.

#### 성능 평가
모델의 성능은 정확도, 정밀도, 재현율, F1 점수와 같은 지표를 사용하여 평가됩니다. 또한, 각 모달리티의 기여도를 정량화하기 위해 민감도 분석을 수행하여 각 모달리티가 최종 결정에 미치는 영향을 평가합니다.

이 연구는 다중 모달 크로스 어텐션 프레임워크가 AD 탐지에서 효과적임을 보여주며, 텍스트, 오디오, 이미지의 통합이 AD 탐지의 정확성을 높이는 데 기여함을 강조합니다.

---




This study proposes a multimodal cross-attention fusion framework for detecting Alzheimer's disease (AD). The framework integrates lexical (text), acoustic (speech), and visual (image) information to detect AD. The dataset used in the study is the ADReSSo 2021 picture-description dataset, which consists of 237 audio recordings where clinically diagnosed AD participants and cognitively healthy controls describe the "Cookie Theft" scene.

#### Model Architecture
The model employs a Transformer-based cross-attention mechanism to integrate three modalities (text, audio, and image). Each modality is processed as follows:

1. **Text**: Text data is encoded into a 768-dimensional vector using ModernBERT. This process involves mean-token pooling to generate a single vector for each recording.

2. **Audio**: Audio features are extracted using the wav2vec 2.0 model, which also converts the audio into a 768-dimensional vector. The audio data is read at a sampling rate of 16 kHz, avoiding aggressive trimming to preserve important data such as hesitations or silence patterns that are significant in AD speech.

3. **Image**: The "Cookie Theft" image is encoded into a 768-dimensional vector using the CLIP ViT-L/14 model. This image serves as a common stimulus for all participants, providing contextual grounding for each recording.

The embeddings from these three modalities are linearly projected to a common dimension and then combined through the cross-attention mechanism, resulting in a single vector suitable for classification.

#### Training and Validation Protocol
During the training process, all encoders are kept frozen, and only the projection layers, cross-attention block, and a small feed-forward network are trained. The AdamW optimizer is used for optimization, and 20% of the training set is reserved as a validation set for early stopping and learning rate scheduling. At test time, the best validation checkpoint is restored, and the linear classification head is discarded to perform predictions using standard classifiers.

#### Performance Evaluation
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1 score. Additionally, a sensitivity analysis is conducted to quantify the contribution of each modality to the final decision, assessing how each modality impacts the overall outcome.

This study demonstrates that the multimodal cross-attention framework is effective in AD detection, highlighting that the integration of text, audio, and image significantly enhances the accuracy of AD detection.


<br/>
# Results



이 연구에서는 알츠하이머병(AD) 탐지를 위한 다중 모달 크로스 어텐션 융합 모델의 성능을 평가하였습니다. ADReSSo 2021 데이터셋을 사용하여, 텍스트, 오디오, 이미지의 세 가지 모달리티를 통합한 모델이 단일 모달리티 및 이중 모달리티 모델보다 우수한 성능을 보였습니다.

#### 경쟁 모델
연구에서는 ADReSSo 2021 챌린지에서 발표된 여러 모델과 비교하였습니다. 특히, Luz et al.의 기본 모델(F1 = 0.7888)과 Balagopalan et al. (F1 = 0.7089), Pan et al. (F1 = 0.7813)과 같은 모델들과 성능을 비교했습니다. 연구의 다중 모달 융합 모델은 F1 점수 0.8571과 정확도 0.8732를 기록하여, 기존의 모델들보다 우수한 성능을 나타냈습니다.

#### 테스트 데이터
ADReSSo 2021 데이터셋은 237개의 오디오 녹음으로 구성되어 있으며, 이 중 166개는 훈련 세트로, 71개는 테스트 세트로 나누어졌습니다. 이 데이터셋은 알츠하이머병 진단을 받은 참가자와 인지적으로 건강한 대조군이 Cookie Theft 장면을 설명하는 내용을 포함하고 있습니다.

#### 메트릭
모델의 성능은 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수로 평가되었습니다. 연구 결과, 텍스트 모달리티가 가장 큰 기여를 하였으며, 오디오 모달리티는 보조적인 개선을 제공하고, 이미지는 제한적인 지원을 제공하는 것으로 나타났습니다. 특히, 텍스트와 이미지를 결합한 모델은 F1 점수 0.8484를 기록하였고, 텍스트와 오디오를 결합한 모델은 F1 점수 0.8307을 기록했습니다. 최종적으로, 세 가지 모달리티를 모두 포함한 모델은 F1 점수 0.8571을 기록하며 가장 높은 성능을 보였습니다.

#### 비교
이 연구의 다중 모달 크로스 어텐션 융합 모델은 단일 모달 및 이중 모달 모델에 비해 일관되게 성능이 향상되었으며, 특히 텍스트 모달리티가 가장 큰 기여를 하였습니다. 오디오 모달리티는 보조적인 역할을 하였고, 이미지는 주로 안정적인 컨텍스트를 제공하는 역할을 했습니다. 이러한 결과는 다중 모달 융합이 AD 탐지에서 효과적임을 보여줍니다.

---




This study evaluated the performance of a multimodal cross-attention fusion model for detecting Alzheimer's disease (AD). Using the ADReSSo 2021 dataset, the model that integrated three modalities—text, audio, and image—outperformed both unimodal and bimodal models.

#### Competing Models
The study compared its results with several models published in the ADReSSo 2021 challenge. Notably, it compared against Luz et al.'s baseline model (F1 = 0.7888), Balagopalan et al. (F1 = 0.7089), and Pan et al. (F1 = 0.7813). The multimodal fusion model achieved an F1 score of 0.8571 and an accuracy of 0.8732, surpassing the performance of these existing models.

#### Test Data
The ADReSSo 2021 dataset consists of 237 audio recordings, with 166 used for training and 71 for testing. This dataset includes descriptions of the Cookie Theft scene by participants diagnosed with Alzheimer's disease and cognitively healthy controls.

#### Metrics
The model's performance was evaluated using accuracy, precision, recall, and F1 score. The results indicated that the text modality contributed the most significant signal, while the audio modality provided complementary improvements, and the image offered limited support. Specifically, the model combining text and image achieved an F1 score of 0.8484, while the model combining text and audio recorded an F1 score of 0.8307. Ultimately, the model incorporating all three modalities achieved the highest performance with an F1 score of 0.8571.

#### Comparison
The multimodal cross-attention fusion model consistently improved performance over unimodal and bimodal models, with the text modality being the primary contributor. The audio modality played a supportive role, while the image primarily provided stabilizing context. These results demonstrate the effectiveness of multimodal fusion in AD detection.


<br/>
# 예제



이 논문에서는 알츠하이머병(AD) 감지를 위한 다중 모달 크로스 어텐션 프레임워크를 제안합니다. 연구에 사용된 데이터는 ADReSSo 2021 챌린지 데이터셋으로, 이 데이터셋은 임상적으로 진단된 알츠하이머 환자와 인지적으로 건강한 대조군이 '쿠키 도둑' 장면을 설명하는 237개의 오디오 녹음으로 구성되어 있습니다. 이 데이터셋은 훈련 세트와 테스트 세트로 나뉘며, 훈련 세트는 166개 샘플(87개 AD, 79개 대조군)로 구성되고, 테스트 세트는 71개 샘플로 구성됩니다.

#### 트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋

1. **트레이닝 데이터**:
   - **인풋**: 
     - 텍스트: '쿠키 도둑' 장면에 대한 설명을 포함하는 자동 전사된 텍스트. 이 텍스트는 ModernBERT를 통해 768차원 벡터로 변환됩니다.
     - 오디오: 녹음된 음성 데이터로, wav2vec 2.0을 사용하여 768차원 오디오 임베딩으로 변환됩니다.
     - 이미지: '쿠키 도둑' 장면의 고정된 이미지로, CLIP ViT-L/14 모델을 통해 768차원 이미지 임베딩으로 변환됩니다.
   - **아웃풋**: 
     - 각 샘플에 대해 AD(알츠하이머) 또는 Non-AD(비알츠하이머)로 분류되는 이진 레이블.

2. **테스트 데이터**:
   - **인풋**: 
     - 훈련 데이터와 동일한 방식으로 전처리된 텍스트, 오디오, 이미지 임베딩.
   - **아웃풋**: 
     - 테스트 세트의 각 샘플에 대해 AD 또는 Non-AD로 분류되는 이진 레이블.

#### 구체적인 테스크
- **테스크**: '쿠키 도둑' 장면에 대한 설명을 기반으로 알츠하이머병을 감지하는 것입니다. 이 작업은 텍스트, 오디오, 이미지의 세 가지 모달리티를 통합하여 수행됩니다. 크로스 어텐션 메커니즘을 통해 각 모달리티의 정보를 결합하여 최종적으로 AD 감지의 정확도를 높이는 것이 목표입니다.

---




This paper proposes a multimodal cross-attention framework for dementia detection, specifically targeting Alzheimer's disease (AD). The study utilizes the ADReSSo 2021 Challenge dataset, which consists of 237 audio recordings where clinically diagnosed AD participants and cognitively healthy controls describe the "Cookie Theft" scene. The dataset is split into training and testing sets, with the training set containing 166 samples (87 AD, 79 controls) and the testing set containing 71 samples.

#### Specific Inputs and Outputs of Training and Testing Data

1. **Training Data**:
   - **Input**: 
     - Text: Automatically transcribed text describing the "Cookie Theft" scene. This text is converted into a 768-dimensional vector using ModernBERT.
     - Audio: Recorded speech data, which is transformed into a 768-dimensional audio embedding using wav2vec 2.0.
     - Image: A fixed image of the "Cookie Theft" scene, encoded into a 768-dimensional image embedding using the CLIP ViT-L/14 model.
   - **Output**: 
     - A binary label indicating whether each sample is AD (Alzheimer's) or Non-AD (non-Alzheimer's).

2. **Testing Data**:
   - **Input**: 
     - Text, audio, and image embeddings that have been preprocessed in the same manner as the training data.
   - **Output**: 
     - A binary label for each sample in the test set indicating AD or Non-AD.

#### Specific Task
- **Task**: The goal is to detect Alzheimer's disease based on descriptions of the "Cookie Theft" scene. This task is performed by integrating three modalities: text, audio, and image. The aim is to enhance the accuracy of AD detection by combining information from each modality through a cross-attention mechanism.

<br/>
# 요약
이 연구에서는 알츠하이머병(AD) 탐지를 위해 텍스트, 오디오, 이미지 정보를 통합한 다중 모달 크로스 어텐션 프레임워크를 제안하였다. 결과적으로, 이 모델은 SVC 분류기를 사용하여 0.8732의 정확도와 0.8571의 F1 점수를 달성하며, 단일 및 이중 모달 구성보다 우수한 성능을 보였다. 이 연구는 다중 모달 융합이 AD 탐지 성능에 미치는 긍정적인 영향을 강조하며, 각 모달리티의 기여도를 분석하였다.

---

This study proposes a multimodal cross-attention framework that integrates text, audio, and image information for dementia detection. As a result, the model achieved an accuracy of 0.8732 and an F1 score of 0.8571 using an SVC classifier, outperforming both unimodal and bimodal configurations. The research highlights the positive impact of multimodal fusion on AD detection performance and analyzes the contributions of each modality.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1**: 이 다이어그램은 텍스트(ModernBERT), 오디오(wav2vec 2.0), 이미지(Cookie Theft)로부터의 768-D 임베딩을 정렬하여 Transformer 기반의 크로스-어텐션 메커니즘을 통해 융합하는 과정을 보여줍니다. 이 구조는 각 모달리티의 정보를 효과적으로 통합하여 AD(알츠하이머병) 탐지에 필요한 분류기 준비가 된 벡터를 생성합니다.

- **Figure 2**: 이 피규어는 각 모달리티가 최종 세 가지 융합 모델에 기여하는 정도를 보여줍니다. 텍스트가 가장 큰 기여를 하며, 오디오는 긍정적인 추가 기여를 하고, 이미지는 평균적으로 미미한 기여를 하는 것으로 나타났습니다. 이는 언어적 특징이 이 작업에서 가장 강력한 구별 신호를 제공한다는 것을 시사합니다.

#### 2. 테이블
- **Table 1**: 단일 모달리티(텍스트 및 오디오)의 성능을 비교합니다. 텍스트 임베딩이 오디오 임베딩보다 AD 예측에서 더 강력한 성능을 보이며, SVC(서포트 벡터 분류기)는 텍스트만으로 F1 점수 0.8358을 달성했습니다.

- **Table 2**: 텍스트와 이미지의 융합 성능을 보여줍니다. 텍스트만 사용할 때보다 SVC의 정확도가 0.8591로 증가하며, 이는 이미지가 텍스트의 정확도를 높이는 안정화 역할을 한다는 것을 나타냅니다.

- **Table 3**: 텍스트와 오디오의 융합 성능을 보여줍니다. SVC는 F1 점수 0.8307을 기록하며, 텍스트만 사용할 때와 비슷한 성능을 보입니다. 이는 텍스트가 여전히 주요한 정보 원천임을 시사합니다.

- **Table 4**: 오디오와 이미지의 융합 성능을 보여줍니다. 이미지가 오디오의 성능을 향상시키며, RF(랜덤 포레스트)는 오디오 전용 F1 점수 0.6769에서 0.7462로 개선되었습니다.

- **Table 5**: 텍스트, 오디오, 이미지의 세 가지 모달리티를 융합한 성능을 보여줍니다. SVC는 F1 점수 0.8571을 기록하며, 이는 단일 및 이중 모달리티보다 우수한 성능을 나타냅니다.

- **Table 6**: 쿼리 모달리티의 변형을 비교합니다. 학습된 쿼리를 사용하는 것이 가장 일관된 성능을 보이며, 특정 모달리티에서 쿼리를 유도할 경우 성능이 저하될 수 있음을 보여줍니다.

- **Table 7**: ADReSSo 2021 챌린지 시스템과의 성능 비교를 보여줍니다. 제안된 세 가지 모달리티 융합 모델이 가장 높은 F1 점수(0.8571)와 정확도(0.8732)를 기록하며, 기존의 다른 시스템보다 우수한 성능을 나타냅니다.

### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Figure 1**: This diagram illustrates the process of aligning 768-D embeddings from text (ModernBERT), audio (wav2vec 2.0), and image (Cookie Theft) and fusing them through a Transformer-based cross-attention mechanism. This structure effectively integrates information from each modality to produce a classifier-ready vector necessary for Alzheimer's disease (AD) detection.

- **Figure 2**: This figure summarizes the contributions of each modality to the final three-way fusion model. Text is shown to have the largest contribution, while audio provides a smaller positive gain, and the image contributes marginally on average. This indicates that linguistic features carry the strongest discriminative signal for this task.

#### 2. Tables
- **Table 1**: Compares the performance of unimodal modalities (text and audio). Text embeddings outperform audio embeddings in predicting AD, with SVC achieving an F1 score of 0.8358 using text alone.

- **Table 2**: Shows the performance of fusing text and image. The accuracy of SVC increases to 0.8591 compared to using text alone, indicating that the image plays a stabilizing role that enhances the accuracy of text.

- **Table 3**: Displays the performance of fusing text and audio. SVC achieves an F1 score of 0.8307, similar to using text alone, suggesting that text remains the primary source of information.

- **Table 4**: Illustrates the performance of fusing audio and image. The image enhances the performance of audio, with RF improving from an audio-only F1 score of 0.6769 to 0.7462.

- **Table 5**: Reports the performance of the trimodal fusion model (text, audio, and image). SVC achieves an F1 score of 0.8571, outperforming both unimodal and bimodal configurations.

- **Table 6**: Compares variations of query modalities. Using a learned query yields the most consistent performance, while using a specific modality as a query can lead to performance degradation.

- **Table 7**: Compares performance against other ADReSSo 2021 challenge systems. The proposed trimodal fusion model achieves the highest overall F1 score (0.8571) and accuracy (0.8732), surpassing other systems.

<br/>
# refer format:



```bibtex
@article{Agbavor2026,
  author = {Felix Agbavor and Hualou Liang},
  title = {Dementia Detection from Spontaneous Speech Using Cross-Attention Fusion},
  journal = {Journal of Dementia and Alzheimer’s Disease},
  year = {2026},
  volume = {3},
  pages = {12},
  doi = {10.3390/jdad3010012},
  publisher = {MDPI},
  note = {Open Access under the terms of the Creative Commons Attribution (CC BY) license}
}
```




Felix Agbavor and Hualou Liang. "Dementia Detection from Spontaneous Speech Using Cross-Attention Fusion." *Journal of Dementia and Alzheimer’s Disease* 3 (2026): 12. https://doi.org/10.3390/jdad3010012. Open Access under the terms of the Creative Commons Attribution (CC BY) license.
