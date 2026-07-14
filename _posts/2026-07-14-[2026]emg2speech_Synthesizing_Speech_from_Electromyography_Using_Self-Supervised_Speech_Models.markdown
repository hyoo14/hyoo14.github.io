---
layout: post
title:  "[2026]emg2speech: Synthesizing Speech from Electromyography Using Self-Supervised Speech Models"
date:   2026-07-14 01:02:17 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 전기근육신호(EMG)를 음성으로 변환하는 비침습적 방법을 제안하며, 자기 지도 학습(self-supervised learning) 모델을 활용하여 EMG 신호와 음성 간의 관계를 학습합니다.


짧은 요약(Abstract) :


이 논문에서는 구강 근육에서 기록된 전기근육도(EMG) 신호를 직접 오디오로 변환하는 신경근육 음성 인터페이스를 제안합니다. 연구 결과, 자기 지도 음성(S3) 표현이 근육 활동의 전기적 파워와 강한 선형 관계가 있음을 발견했습니다. 간단한 선형 매핑을 통해 S3 표현에서 EMG 파워를 예측할 수 있으며, 이 두 가지 요소는 서로 구별 가능한 클러스터를 형성합니다. 이러한 관찰은 S3 모델이 EMG 활동에 반영된 조음 메커니즘을 암묵적으로 인코딩하고 있음을 시사합니다. 이 구조를 활용하여 EMG 신호를 S3 표현 공간으로 매핑하고 음성을 합성함으로써, 명시적인 조음 모델링이나 보코더 훈련 없이 EMG에서 음성으로의 전환을 가능하게 합니다. 이 시스템은 근위축성 측삭경화증(ALS) 환자를 대상으로 시연되었으며, 그녀가 조용히 발음하는 동안 기록된 구강 EMG를 오디오로 변환했습니다.



This paper presents a neuromuscular speech interface that translates electromyographic (EMG) signals recorded from orofacial muscles during speech articulation directly into audio. The authors find that self-supervised speech (S3) representations are strongly linearly related to the electrical power of muscle activity: a simple linear mapping predicts EMG power from S3 representations with a correlation of r = 0.85. Additionally, EMG power vectors associated with distinct articulatory gestures form structured, separable clusters. Together, these observations suggest that S3 models implicitly encode articulatory mechanisms, as reflected in EMG activity. Leveraging this structure, the authors map EMG signals into the S3 representation space and synthesize speech, enabling end-to-end EMG-to-speech generation without explicit articulatory modeling or vocoder training. The system is demonstrated with a participant diagnosed with amyotrophic lateral sclerosis (ALS), converting orofacial EMG recorded while she silently articulated speech into audio.


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



이 논문에서는 전기근육신호(EMG)를 음성으로 변환하는 시스템을 제안합니다. 이 시스템은 자가 감독 음성(S3) 모델을 활용하여 EMG 신호를 직접 오디오로 변환하는 방법을 사용합니다. 주요 방법론은 다음과 같습니다.

1. **EMG 신호 수집**: EMG 신호는 목, 턱, 뺨, 입술 등에서 31개의 전극을 사용하여 5000Hz의 샘플링 주기로 수집됩니다. 이 신호는 발화 중의 근육 활동을 반영합니다.

2. **특징 추출**: 수집된 EMG 신호는 공분산 행렬을 사용하여 특징을 추출합니다. 이 공분산 행렬은 EMG 신호의 시간적 정보를 집계하여 고정 차원의 특징을 생성합니다. 이를 통해 각 전극에서의 근육 활동 전력을 나타내는 D(E) 벡터를 생성합니다.

3. **S3 모델 활용**: S3 모델은 EMG 신호와 음성 신호 간의 관계를 학습하는 데 사용됩니다. 이 모델은 EMG 신호에서 추출된 D(E) 벡터와 S3 모델에서 추출된 음성 특징 간의 선형 관계를 학습합니다. 이 관계는 EMG 신호가 음성의 발화 구조를 암시적으로 인코딩하고 있음을 나타냅니다.

4. **모델 아키텍처**: EMG 신호는 TDS(시간 깊이 분리) 컨볼루션 네트워크를 통해 처리됩니다. 이 네트워크는 지역적 시간적 맥락을 활용하여 EMG 신호를 처리하고, 두 개의 독립적인 출력 헤드를 통해 음성 단위와 음소를 예측합니다. 이 구조는 EMG 신호와 음성 단위 간의 비동기적 매핑을 가능하게 합니다.

5. **훈련 기법**: 모델은 CTC(연결주의 시간 분류) 손실을 사용하여 훈련됩니다. 이 손실 함수는 EMG 신호와 음성 단위 간의 정렬이 알려져 있지 않은 상황에서도 학습할 수 있도록 합니다. 또한, 음소 가이드를 통해 음성 단위 예측의 품질을 향상시키는 추가적인 일관성 손실을 도입합니다.

6. **데이터셋**: 연구에서는 건강한 참가자와 ALS(근위축성 측삭경화증) 환자로부터 수집된 EMG 데이터를 포함한 대규모 데이터셋을 사용합니다. 이 데이터셋은 약 9시간 분량의 EMG 음성 데이터로 구성되어 있으며, 6800개 이상의 고유 단어를 포함합니다.

이러한 방법론을 통해 EMG 신호를 효과적으로 음성으로 변환할 수 있는 시스템을 구축하였으며, 이는 비침습적인 방법으로 다양한 언어 장애를 가진 개인들이 기능적인 음성을 회복할 수 있도록 돕는 데 기여할 수 있습니다.

---




This paper presents a system for converting electromyography (EMG) signals into speech. The system utilizes self-supervised speech (S3) models to directly translate EMG signals into audio. The main methodologies are as follows:

1. **EMG Signal Collection**: EMG signals are collected from 31 electrode sites on the neck, chin, jaw, cheek, and lips at a sampling rate of 5000 Hz. These signals reflect muscle activity during speech articulation.

2. **Feature Extraction**: The collected EMG signals are processed to extract features using covariance matrices. This covariance matrix aggregates the temporal information of the EMG signals to produce fixed-dimensional features. This results in a D(E) vector that represents the muscle action potential power at each electrode.

3. **Utilization of S3 Models**: The S3 model is employed to learn the relationship between EMG signals and speech signals. This model learns a linear mapping between the D(E) vectors extracted from EMG signals and the speech features extracted from the S3 model. This relationship indicates that EMG signals implicitly encode the articulatory structure of speech.

4. **Model Architecture**: EMG signals are processed through a time-depth separable (TDS) convolutional network. This network leverages local temporal context to process EMG signals and predicts speech units and phonemes through two independent output heads. This structure allows for asynchronous mapping between EMG signals and speech units.

5. **Training Techniques**: The model is trained using a connectionist temporal classification (CTC) loss function. This loss function enables learning even when the alignment between EMG signals and speech units is unknown. Additionally, a phoneme-guided approach is introduced to enhance the quality of speech unit predictions through an additional consistency loss.

6. **Datasets**: The study utilizes a large dataset comprising EMG data collected from both healthy participants and a participant with amyotrophic lateral sclerosis (ALS). This dataset consists of approximately 9 hours of EMG speech data, including over 6,800 unique words.

Through these methodologies, a system capable of effectively converting EMG signals into speech has been developed, contributing to the potential recovery of functional speech for individuals with various speech impairments through a non-invasive approach.


<br/>
# Results



이 논문에서는 EMG(근전도) 신호를 음성으로 변환하는 시스템을 제안하고, 이를 통해 ALS(근위축성 측삭경화증) 환자와 건강한 참가자에게서 음성을 생성하는 성능을 평가하였다. 연구의 주요 결과는 다음과 같다.

1. **경쟁 모델과 비교**: 기존의 EMG-음성 변환 모델들과 비교했을 때, 본 연구에서 제안한 모델은 더 높은 성능을 보였다. Gaddy와 Klein(2020, 2021)의 연구에서는 EMG-음성 변환에서 각각 68%와 42%의 단어 오류율(WER)을 보고했으나, 본 연구에서는 62%의 WER을 달성하였다. 이는 본 연구가 시간 정렬된 EMG-오디오 쌍을 가정하지 않고도 성능을 개선했음을 보여준다.

2. **테스트 데이터**: 본 연구에서는 두 가지 데이터셋을 사용하였다. 첫 번째는 건강한 참가자로부터 수집된 약 9시간 분량의 EMG 음성 데이터로, 약 6,800개의 고유 단어를 포함하고 있다. 두 번째는 ALS 환자로부터 수집된 약 1시간 분량의 EMG 음성 데이터로, 약 300개의 고유 단어를 포함하고 있다. 각 데이터셋은 훈련, 검증, 테스트 세트로 나누어져 있으며, 테스트 세트는 훈련 및 검증 세트에 포함되지 않은 문장으로 구성되어 있다.

3. **메트릭**: 성능 평가는 단어 오류율(WER)과 음소 오류율(PER)을 사용하여 이루어졌다. 건강한 참가자 데이터셋에서, vec(E) 입력을 사용한 모델은 56.08%의 UER을 기록하였고, phoneme-guided decoding을 적용했을 때는 51.81%로 개선되었다. ALS 환자 데이터셋에서는 vec(E) 입력을 사용한 모델이 52.29%의 UER을 기록하였고, phoneme-guided decoding을 적용했을 때는 46.98%로 성능이 향상되었다.

4. **비교**: 본 연구의 결과는 기존의 EMG-음성 변환 연구와 비교할 때, 더 높은 성능을 보였으며, 특히 phoneme-guided decoding을 통해 음성 생성의 품질이 향상되었다. 이는 음소가 발음 구성에 의해 정의되며, EMG 신호가 음소 구조를 더 충실히 포착할 수 있음을 나타낸다.

이러한 결과들은 EMG 신호를 기반으로 한 비침습적 음성 인터페이스의 가능성을 보여주며, 향후 다양한 임상 환경에서의 적용 가능성을 제시한다.

---




This paper presents a system for converting EMG (electromyography) signals into speech and evaluates its performance in generating speech from both an ALS (amyotrophic lateral sclerosis) patient and a healthy participant. The main results of the study are as follows:

1. **Comparison with Competing Models**: Compared to existing EMG-to-speech conversion models, the proposed model demonstrated superior performance. Previous studies by Gaddy and Klein (2020, 2021) reported word error rates (WER) of 68% and 42%, respectively, while this study achieved a WER of 62%. This indicates that the proposed model improved performance without assuming time-aligned EMG-audio pairs.

2. **Test Data**: The study utilized two datasets. The first dataset consists of approximately 9 hours of EMG speech data collected from a healthy participant, containing around 6,800 unique words. The second dataset consists of about 1 hour of EMG speech data from an ALS patient, containing around 300 unique words. Each dataset was divided into training, validation, and test sets, with the test sets containing sentences not included in the training or validation sets.

3. **Metrics**: Performance evaluation was conducted using word error rate (WER) and phoneme error rate (PER). In the healthy participant dataset, the model using vec(E) input recorded a unit error rate (UER) of 56.08%, which improved to 51.81% with phoneme-guided decoding. In the ALS patient dataset, the model using vec(E) input achieved a UER of 52.29%, which further improved to 46.98% with phoneme-guided decoding.

4. **Comparison**: The results of this study showed higher performance compared to previous EMG-to-speech conversion research, particularly with the application of phoneme-guided decoding, which enhanced the quality of the generated speech. This indicates that phonemes, defined by articulatory configurations, can be more faithfully captured by EMG signals.

These results demonstrate the potential of non-invasive speech interfaces based on EMG signals and suggest future applicability in various clinical settings.


<br/>
# 예제



이 논문에서는 전기근육도(EMG) 신호를 사용하여 음성을 합성하는 시스템을 제안합니다. 이 시스템은 EMG 신호를 입력으로 받아 음성을 출력하는 방식으로 작동합니다. 다음은 트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋, 그리고 수행하는 태스크에 대한 설명입니다.

#### 1. 데이터셋 구성
- **DATA GENERAL**: 건강한 참가자로부터 수집된 데이터로, 약 9시간 분량의 EMG 음성 데이터가 포함되어 있습니다. 이 데이터셋은 약 6,800개의 고유 단어와 9,660개의 문장으로 구성되어 있습니다. 
  - **트레이닝 데이터**: 8,500개의 문장
  - **검증 데이터**: 760개의 문장
  - **테스트 데이터**: 400개의 문장

- **DATA ALS**: 근위축성 측삭경화증(ALS) 환자로부터 수집된 데이터로, 약 1시간 분량의 EMG 음성 데이터가 포함되어 있습니다. 이 데이터셋은 약 300개의 고유 단어와 600개의 문장으로 구성되어 있습니다.
  - **트레이닝 데이터**: 500개의 문장
  - **검증 데이터**: 40개의 문장
  - **테스트 데이터**: 60개의 문장

#### 2. 인풋과 아웃풋
- **인풋**: EMG 신호는 31개의 전극에서 수집되며, 각 전극에서의 신호는 시간에 따라 변화하는 전기적 활동을 나타냅니다. 이 신호는 5,000Hz의 샘플링 주기로 기록됩니다. EMG 신호는 다음과 같은 형태로 변환됩니다:
  - **D(E)**: 각 전극에서의 EMG 전력의 대각선 벡터
  - **vec(E)**: EMG 신호의 공분산 행렬을 벡터화한 것
  - **vec(B)**: EMG 신호의 스펙트로그램을 벡터화한 것

- **아웃풋**: 모델의 출력은 음성 합성을 위한 디스크리트 유닛(예: HuBERT 유닛) 또는 음소(phoneme) 시퀀스입니다. 이 출력은 EMG 신호와의 관계를 학습하여 생성됩니다.

#### 3. 태스크
- **EMG-to-Speech Synthesis**: EMG 신호를 입력으로 받아 음성을 생성하는 태스크입니다. 이 과정에서 모델은 EMG 신호와 음성 간의 관계를 학습하고, 이를 통해 새로운 음성을 합성합니다. 
- **디코딩**: 모델은 EMG 신호로부터 HuBERT 유닛 또는 음소를 예측하고, 이를 기반으로 음성을 생성합니다. 이 과정에서 CTC(연결주의 시간 분류) 손실을 사용하여 학습합니다.





This paper presents a system for synthesizing speech from electromyography (EMG) signals. The system operates by taking EMG signals as input and producing audio as output. Below is a detailed explanation of the training and testing data, including specific inputs and outputs, as well as the tasks performed.

#### 1. Dataset Composition
- **DATA GENERAL**: This dataset consists of approximately 9 hours of EMG speech data collected from a healthy participant. It includes around 6,800 unique words and 9,660 sentences.
  - **Training Data**: 8,500 sentences
  - **Validation Data**: 760 sentences
  - **Testing Data**: 400 sentences

- **DATA ALS**: This dataset consists of approximately 1 hour of EMG speech data collected from a participant diagnosed with amyotrophic lateral sclerosis (ALS). It includes around 300 unique words and 600 sentences.
  - **Training Data**: 500 sentences
  - **Validation Data**: 40 sentences
  - **Testing Data**: 60 sentences

#### 2. Inputs and Outputs
- **Inputs**: EMG signals are collected from 31 electrode sites, representing the electrical activity of the muscles over time. These signals are recorded at a sampling rate of 5,000 Hz. The EMG signals are transformed into the following forms:
  - **D(E)**: A diagonal vector representing the EMG power at each electrode
  - **vec(E)**: A vectorized representation of the covariance matrix of the EMG signals
  - **vec(B)**: A vectorized representation of the EMG signal's spectrogram

- **Outputs**: The model's output consists of discrete units (e.g., HuBERT units) or phoneme sequences for speech synthesis. This output is generated by learning the relationship between EMG signals and speech.

#### 3. Tasks
- **EMG-to-Speech Synthesis**: This task involves taking EMG signals as input and generating speech. The model learns the relationship between EMG signals and speech to synthesize new audio.
- **Decoding**: The model predicts HuBERT units or phonemes from the EMG signals, which are then used to generate speech. The connectionist temporal classification (CTC) loss is employed for training in this process.

<br/>
# 요약

이 논문에서는 전기근육신호(EMG)를 음성으로 변환하는 비침습적 방법을 제안하며, 자기 지도 학습(self-supervised learning) 모델을 활용하여 EMG 신호와 음성 간의 관계를 학습합니다. 실험 결과, EMG 신호의 전력 벡터가 자기 지도 음성 표현과 강한 선형 상관관계를 가지며, 이를 통해 ALS 환자의 EMG 신호를 음성으로 변환하는 데 성공했습니다. 최종적으로, 제안된 시스템은 EMG 신호를 기반으로 한 음성 합성에서 높은 품질을 보여주었습니다.

---

This paper presents a non-invasive method for converting electromyography (EMG) signals into speech, utilizing self-supervised learning models to learn the relationship between EMG signals and audio. Experimental results show that the power vectors of EMG signals have a strong linear correlation with self-supervised speech representations, successfully converting EMG signals from an ALS patient into audio. Ultimately, the proposed system demonstrates high quality in speech synthesis based on EMG signals.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **피규어 1**: EMG 신호의 대각선 벡터 D(E)가 서로 다른 오로페이셜 제스처에 대해 자연스럽게 구분되는 클러스터를 형성하는 것을 보여줍니다. 이는 EMG 신호가 발음 정보와 관련된 구조적 정보를 인코딩하고 있음을 시사합니다.
   - **피규어 2**: 다양한 자가 감독 음성 모델(HuBERT, WAV2VEC2.0 등)에서 D(E)와의 상관관계를 보여줍니다. 이 피규어는 D(E)가 자가 감독 음성 표현과 강한 선형 관계를 가지며, 특정 레이어에서 최적의 상관관계를 나타냄을 보여줍니다.
   - **피규어 4**: EMG 신호를 입력으로 받아 음성을 생성하는 전체 아키텍처를 설명합니다. 이 구조는 EMG 신호에서 직접 음성을 생성하는 방법을 시각적으로 나타냅니다.

2. **테이블**
   - **테이블 1**: DATA GENERAL에서 다양한 EMG 피처 표현을 사용하여 dis(H) HUBERT 유닛을 예측할 때의 단위 오류율(UER)을 보여줍니다. vec(E)가 가장 낮은 UER을 기록하여 EMG 신호의 표현이 음성 생성에 효과적임을 나타냅니다.
   - **테이블 2**: DATA GENERAL에서 음소 오류율(PER)을 비교합니다. vec(E)가 가장 낮은 PER을 기록하여 EMG 신호가 음소 수준에서의 정확성을 높이는 데 기여함을 보여줍니다.
   - **테이블 3**: 음소 유도 훈련이 dis(H) HUBERT 유닛 디코딩에 미치는 영향을 보여줍니다. 음소 유도 훈련이 UER을 유의미하게 감소시킴을 나타냅니다.
   - **테이블 4**: DATA ALS에서의 UER을 보여줍니다. vec(E)와 음소 유도 디코딩을 사용한 경우 가장 낮은 UER을 기록하여 ALS 환자에서도 효과적인 성능을 나타냅니다.
   - **테이블 5**: 인간 평가자에 의한 WER을 보여줍니다. DATA ALS에서 평균 WER이 낮아, ALS 환자에서도 음성 생성의 이해 가능성을 높임을 나타냅니다.

3. **어펜딕스**
   - **어펜딕스 A**: EMG 신호 수집 방법 및 전처리 과정을 상세히 설명합니다. 이는 EMG 신호의 품질과 관련된 중요한 정보를 제공합니다.
   - **어펜딕스 B**: 음소 유도 디코딩의 세부 사항을 설명합니다. 이는 음소와 HUBERT 유닛 간의 관계를 명확히 하여 음성 생성의 정확성을 높이는 데 기여합니다.
   - **어펜딕스 D**: 모델 아키텍처 및 훈련 절차에 대한 세부 정보를 제공합니다. 이는 연구의 재현 가능성을 높이는 데 중요한 역할을 합니다.

### Summary of Results and Insights from Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Figure 1**: Shows that the diagonal vectors D(E) of EMG signals form distinct clusters for different orofacial gestures, indicating that EMG signals encode structural information related to articulation.
   - **Figure 2**: Displays the correlation between D(E) and various self-supervised speech models (HuBERT, WAV2VEC2.0, etc.). This figure illustrates that D(E) has a strong linear relationship with self-supervised speech representations, with optimal correlation observed at specific layers.
   - **Figure 4**: Illustrates the overall architecture for generating speech directly from EMG signals, visually representing the method of synthesizing audio from EMG data.

2. **Tables**
   - **Table 1**: Shows the unit error rate (UER) when predicting dis(H) HUBERT units using various EMG feature representations on DATA GENERAL. The lowest UER is achieved with vec(E), indicating the effectiveness of this representation for speech generation.
   - **Table 2**: Compares phoneme error rates (PER) on DATA GENERAL, with vec(E) achieving the lowest PER, demonstrating that EMG signals contribute to higher accuracy at the phoneme level.
   - **Table 3**: Highlights the impact of phoneme-guided training on dis(H) HUBERT unit decoding, showing a significant reduction in UER due to this training approach.
   - **Table 4**: Displays UER on DATA ALS, with vec(E) and phoneme-guided decoding yielding the lowest UER, indicating effective performance even in ALS patients.
   - **Table 5**: Reports word error rates (WER) assessed by human raters, showing lower average WER on DATA ALS, which suggests improved intelligibility of synthesized speech for ALS patients.

3. **Appendices**
   - **Appendix A**: Provides detailed information on the EMG signal collection and preprocessing methods, which are crucial for ensuring the quality of EMG signals.
   - **Appendix B**: Explains the details of phoneme-guided decoding, clarifying the relationship between phonemes and HUBERT units, which enhances the accuracy of speech generation.
   - **Appendix D**: Offers detailed descriptions of model architectures and training procedures, contributing to the reproducibility of the research.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Gowda2026,
  author    = {Harshavardhana T. Gowda and Daniel C. Comstock and Lee M. Miller},
  title     = {emg2speech: Synthesizing Speech from Electromyography Using Self-Supervised Speech Models},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {16490--16507},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Gowda, Harshavardhana T., Daniel C. Comstock, and Lee M. Miller. "emg2speech: Synthesizing Speech from Electromyography Using Self-Supervised Speech Models." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 16490–16507. July 2026. Association for Computational Linguistics. 
    