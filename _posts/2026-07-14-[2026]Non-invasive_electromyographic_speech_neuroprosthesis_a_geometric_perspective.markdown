---
layout: post
title:  "[2026]Non-invasive electromyographic speech neuroprosthesis: a geometric perspective"
date:   2026-07-14 00:54:23 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 비침습적 근전도(EMG) 신호를 사용하여 침묵 속에서 발음된 음성을 텍스트로 변환하는 방법을 제안합니다.


짧은 요약(Abstract) :


이 논문에서는 비침습적인 신경근 speech 인터페이스를 제안합니다. 이 인터페이스는 사용자가 소리 없이 발음하는 내용을 직접 텍스트로 변환합니다. 연구팀은 참가자들이 소리 없이 발음할 때 얼굴과 목의 여러 발음 부위에서 표면 근전도(EMG) 신호를 기록하여 EMG-텍스트 변환을 가능하게 합니다. 이러한 인터페이스는 후두 절제술, 신경근 질환, 뇌졸중 또는 외상으로 인해 이해할 수 있는 말을 할 수 없는 개인들에게 의사소통을 회복할 수 있는 잠재력을 가지고 있습니다. 이전 연구들은 주로 소리 있는 발음 중 수집된 EMG를 시간 정렬된 오디오 목표에 매핑하거나 이러한 목표를 소리 없는 EMG 기록으로 전송하는 데 초점을 맞췄습니다. 그러나 이러한 방법은 오디오에 의존하며 더 이상 말을 할 수 없는 환자들에게는 적용이 제한적입니다. 반면, 본 연구에서는 고차원 EMG 신호의 효율적인 표현을 제안하고, 시간 정렬된 오디오에 의존하지 않고 음소 수준에서 직접 EMG-텍스트 변환을 수행하는 방법을 보여줍니다.


This paper presents a non-invasive neuromuscular speech interface that translates silently voiced articulations directly into text. The researchers record surface electromyographic (EMG) signals from multiple articulatory sites on the face and neck as participants silently articulate speech, enabling direct EMG-to-text translation. Such an interface has the potential to restore communication for individuals who have lost the ability to produce intelligible speech due to laryngectomy, neuromuscular disease, stroke, or trauma-induced damage. Prior work has largely focused on mapping EMG collected during audible articulation to time-aligned audio targets or transferring these targets to silent EMG recordings, which inherently requires audio and limits applicability to patients who can no longer speak. In contrast, this study proposes an efficient representation of high-dimensional EMG signals and demonstrates direct sequence-to-sequence EMG-to-text conversion at the phonemic level without relying on time-aligned audio.


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



이 논문에서 제안하는 방법은 비침습적인 전기근육도(Electromyography, EMG) 신경 보철 장치를 사용하여 침묵 속에서 발음된 언어를 텍스트로 변환하는 시스템입니다. 이 시스템은 여러 개의 근육 활성 패턴을 기록하여 고차원 EMG 신호를 효율적으로 표현하고, 이를 통해 음소 수준에서 직접 EMG-텍스트 변환을 수행합니다.

#### 모델 아키텍처
모델은 주로 게이트 순환 유닛(GRU) 아키텍처를 기반으로 하며, EMG 신호를 입력으로 받아 음소 시퀀스를 출력합니다. EMG 신호는 여러 전극에서 수집되며, 이 신호들은 대칭 양의 정부호(SPD) 행렬로 변환됩니다. SPD 행렬은 EMG 신호의 상관관계를 모델링하는 데 사용되며, 이를 통해 다양한 근육의 활동 패턴을 효과적으로 캡처할 수 있습니다.

#### 데이터 전처리
EMG 신호는 31개의 전극에서 수집되며, 각 전극은 얼굴과 목의 다양한 부위에 배치됩니다. 수집된 신호는 밴드패스 필터링을 통해 노이즈를 제거하고, z-정규화를 통해 각 채널의 신호를 정규화합니다. 이후, 이 신호들은 시간 창을 기준으로 SPD 행렬로 변환됩니다.

#### 훈련 방법
모델은 CTC(Connectionist Temporal Classification) 손실 함수를 사용하여 훈련됩니다. CTC 손실은 입력 신호와 출력 음소 시퀀스 간의 정밀한 정렬이 필요하지 않기 때문에, EMG 신호와 음소 간의 비정렬 쌍을 사용하여 훈련할 수 있습니다. 이 과정에서, 모델은 EMG 신호로부터 음소 확률 분포를 생성하고, 이를 바탕으로 가장 가능성이 높은 음소 시퀀스를 복원합니다.

#### 성능 평가
모델의 성능은 음소 오류율(PER)과 단어 오류율(WER)로 평가됩니다. 실험 결과, 제안된 방법은 기존의 EMG-음성 변환 시스템에 비해 상당히 낮은 오류율을 기록하였으며, 이는 EMG 신호만으로도 의미 있는 언어 구조를 추론할 수 있음을 보여줍니다.

이러한 방법론은 비침습적인 EMG 기반의 언어 인터페이스를 통해, 발음 기능을 잃은 개인들이 자연스럽게 의사소통할 수 있는 가능성을 제시합니다.

---




The method proposed in this paper is a non-invasive electromyographic (EMG) speech neuroprosthesis that translates silently articulated speech directly into text. This system records muscle activation patterns from multiple sites on the face and neck, enabling efficient representation of high-dimensional EMG signals and direct EMG-to-text conversion at the phonemic level.

#### Model Architecture
The model is primarily based on a Gated Recurrent Unit (GRU) architecture, which takes EMG signals as input and outputs phoneme sequences. The EMG signals are collected from multiple electrodes placed on various parts of the face and neck. These signals are transformed into symmetric positive definite (SPD) matrices, which are used to model the correlations in the EMG signals, effectively capturing the activity patterns of different muscles.

#### Data Preprocessing
EMG signals are collected from 31 electrodes, each positioned over anatomical regions that correspond to muscle groups involved in speech articulation. The collected signals undergo bandpass filtering to remove noise and are z-normalized to standardize the signals across channels. Subsequently, these signals are converted into SPD matrices based on time windows.

#### Training Method
The model is trained using the Connectionist Temporal Classification (CTC) loss function. CTC loss does not require precise alignment between input signals and output phoneme sequences, allowing the model to be trained on unaligned pairs of EMG signals and phonemes. During this process, the model generates phoneme probability distributions from the EMG signals and reconstructs the most likely phoneme sequences based on these probabilities.

#### Performance Evaluation
The model's performance is evaluated using Phoneme Error Rate (PER) and Word Error Rate (WER). Experimental results show that the proposed method achieves significantly lower error rates compared to existing EMG-to-speech systems, demonstrating that linguistically meaningful speech structure can be inferred from EMG signals alone.

This methodology presents the potential for a non-invasive EMG-based speech interface, enabling individuals who have lost their ability to produce intelligible speech to communicate naturally.


<br/>
# Results



이 논문에서는 비침습적 전기근육도(Electromyography, EMG) 신경 보철 장치를 통해 침묵 속에서 발음된 언어를 텍스트로 변환하는 방법을 제안합니다. 연구팀은 31개의 전극을 사용하여 목과 얼굴의 여러 부위에서 EMG 신호를 수집하였고, 이를 통해 6500개 이상의 고유 단어를 포함하는 약 8시간 분량의 EMG 음성 데이터를 수집했습니다. 이 데이터는 EMG 신호와 해당하는 텍스트 전사본으로 구성되어 있으며, 시간 정렬된 오디오 데이터 없이도 EMG 신호를 직접적으로 텍스트로 변환할 수 있는 모델을 훈련하는 데 사용되었습니다.

#### 결과
1. **경쟁 모델**: 기존의 EMG 기반 음성 인식 모델들은 주로 시간 정렬된 EMG-오디오 쌍을 필요로 했습니다. 예를 들어, Gaddy와 Klein(2020, 2021)의 연구에서는 EMG 신호와 오디오 신호 간의 정렬을 통해 음성을 인식하는 방법을 사용했습니다. 이와 달리, 본 연구에서는 EMG 신호만을 사용하여 직접적으로 음소(phoneme) 시퀀스를 예측하는 방법을 제안합니다.

2. **테스트 데이터**: 본 연구에서는 1970개의 문장으로 구성된 테스트 세트를 사용하였으며, 이 문장들은 훈련 및 검증 세트와는 별개로 구성되었습니다. 훈련 세트는 8000개의 문장, 검증 세트는 1000개의 문장으로 구성되었습니다.

3. **메트릭**: 성능 평가는 음소 오류율(Phoneme Error Rate, PER)과 단어 오류율(Word Error Rate, WER)로 측정되었습니다. PER은 48.47%로 보고되었으며, 이는 무작위 추측에 의한 PER 약 97.5%에 비해 상당히 낮은 수치입니다. WER은 73.53%로 보고되었습니다.

4. **비교**: 본 연구의 방법은 EMG 2QWERTY 데이터셋(Sivakumar et al., 2024)과 비교되었습니다. 이 데이터셋은 EMG 신호를 사용하여 QWERTY 키보드에서 입력된 문자를 디코딩하는 작업을 포함합니다. 본 연구에서는 SPD 행렬을 사용하여 성능을 개선하였으며, 기존의 스펙트로그램 기반 접근 방식보다 더 나은 성능을 보였습니다. 특히, SPD 행렬을 사용한 모델은 PER이 48.47%, WER이 73.53%로 나타났습니다.

이러한 결과는 EMG 신호만으로도 의미 있는 언어 구조를 추론할 수 있음을 보여주며, 향후 연구에서 더 나은 모델링과 디코딩 방법을 개발할 수 있는 기초를 마련합니다.

---



This paper proposes a non-invasive electromyographic (EMG) speech neuroprosthesis that translates silently articulated speech directly into text. The research team collected EMG signals from multiple sites on the face and neck using 31 electrodes, resulting in approximately 8 hours of EMG speech data covering over 6500 unique words. This data consists of EMG signals paired with their corresponding text transcriptions, allowing the training of a model to convert EMG signals into text without relying on time-aligned audio data.

#### Results
1. **Competing Models**: Previous EMG-based speech recognition models primarily required time-aligned EMG-audio pairs. For instance, the studies by Gaddy and Klein (2020, 2021) utilized alignment between EMG and audio signals to recognize speech. In contrast, this study proposes a method that directly predicts phoneme sequences from EMG signals alone.

2. **Test Data**: The study utilized a test set consisting of 1970 sentences, which were separate from the training and validation sets. The training set comprised 8000 sentences, while the validation set included 1000 sentences.

3. **Metrics**: Performance was measured using Phoneme Error Rate (PER) and Word Error Rate (WER). The reported PER was 48.47%, significantly lower than the chance-level PER of approximately 97.5%. The WER was reported at 73.53%.

4. **Comparison**: The methods in this study were compared against the EMG 2QWERTY dataset (Sivakumar et al., 2024), which involves decoding characters typed on a QWERTY keyboard using EMG signals. The study improved performance by using SPD matrices, outperforming previous spectrogram-based approaches. Specifically, the model using SPD matrices achieved a PER of 48.47% and a WER of 73.53%.

These results demonstrate that meaningful linguistic structures can be inferred from EMG signals alone, establishing a foundation for future research aimed at developing improved modeling and decoding methods.


<br/>
# 예제



이 논문에서는 비침습적 전기근육도(EMG) 신경 보철 장치를 사용하여 조용히 발음된 언어를 텍스트로 변환하는 방법을 제안합니다. 연구의 주요 목표는 발음이 불가능한 사람들에게 의사소통을 복원하는 것입니다. 이 시스템은 여러 개의 근육 전기 신호를 기록하여 발음의 세부 사항을 포착하고, 이를 통해 음소 수준에서 직접 EMG-텍스트 변환을 수행합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **데이터 수집**: 연구에서는 31개의 전극을 사용하여 목, 턱, 뺨, 입술 등에서 EMG 신호를 수집했습니다. 이 데이터는 8시간 분량의 조용한 발음 데이터를 포함하고 있으며, 6500개 이상의 고유 단어로 구성된 대규모 어휘 코퍼스를 포함합니다.
   - **입력**: EMG 신호는 시간에 따라 변화하는 신호로, 각 전극에서 수집된 EMG 신호의 시퀀스입니다. 예를 들어, 특정 문장을 조용히 발음할 때의 EMG 신호가 입력으로 사용됩니다.
   - **출력**: 각 EMG 신호에 대해 해당하는 음소의 시퀀스가 출력됩니다. 예를 들어, "FRIDAY"라는 단어를 조용히 발음할 때, EMG 신호는 <F-R-IY-D-AY>와 같은 음소 시퀀스로 변환됩니다.

2. **테스트 데이터**:
   - **데이터 수집**: 테스트 데이터는 훈련 데이터와는 별도로 수집된 1970개의 문장으로 구성되어 있습니다. 이 문장들은 훈련 및 검증 세트에 포함되지 않은 새로운 문장들입니다.
   - **입력**: 테스트 데이터의 EMG 신호는 훈련 데이터와 동일한 방식으로 수집되며, 각 문장에 대한 EMG 신호가 입력으로 사용됩니다.
   - **출력**: 모델은 테스트 데이터의 EMG 신호를 기반으로 음소 시퀀스를 예측하고, 이를 다시 단어로 변환합니다. 예를 들어, "IT WAS PAID FOR"라는 문장을 조용히 발음했을 때, 모델은 이를 <IH-T W-AA-Z P-EY-D F-O-R>와 같은 음소 시퀀스로 변환할 수 있습니다.

이러한 방식으로, 연구는 EMG 신호를 통해 조용한 발음을 텍스트로 변환하는 시스템의 가능성을 보여주고 있습니다.

---




This paper presents a non-invasive electromyographic (EMG) speech neuroprosthesis that translates silently voiced articulations directly into text. The main goal of the research is to restore communication for individuals who have lost the ability to produce intelligible speech. The system records multiple muscle electrical signals to capture the nuances of articulation, enabling direct EMG-to-text conversion at the phonemic level.

#### Training Data and Test Data

1. **Training Data**:
   - **Data Collection**: The study collected EMG signals from 31 electrodes placed on the face and neck, covering areas such as the chin, jaw, cheeks, and lips. The training dataset consists of approximately 8 hours of silent speech data, encompassing a large vocabulary corpus with over 6500 unique words.
   - **Input**: The input consists of sequences of EMG signals that vary over time, recorded from each electrode. For example, when a participant silently articulates a specific sentence, the corresponding EMG signals are used as input.
   - **Output**: For each EMG signal, the output is a sequence of corresponding phonemes. For instance, when silently articulating the word "FRIDAY," the EMG signals are converted into a phoneme sequence like <F-R-IY-D-AY>.

2. **Test Data**:
   - **Data Collection**: The test dataset comprises 1970 sentences that were collected separately from the training data. These sentences are new and not included in the training and validation sets.
   - **Input**: The EMG signals for the test data are collected in the same manner as the training data, with each sentence's EMG signals used as input.
   - **Output**: The model predicts phoneme sequences based on the EMG signals from the test data and converts them back into words. For example, when silently articulating the sentence "IT WAS PAID FOR," the model might convert this into a phoneme sequence like <IH-T W-AA-Z P-EY-D F-O-R>.

Through this approach, the research demonstrates the feasibility of translating silent speech into text using EMG signals.

<br/>
# 요약

이 논문에서는 비침습적 근전도(EMG) 신호를 사용하여 침묵 속에서 발음된 음성을 텍스트로 변환하는 방법을 제안합니다. 연구 결과, 제안된 방법은 49%의 음소 오류율(PER)과 73.53%의 단어 오류율(WER)을 기록하며, 이는 기존 방법들보다 우수한 성능을 보입니다. 예를 들어, "IT WAS PAID FOR"와 같은 문장이 정확하게 변환되었지만, "THE DEATH PENALTY"와 같은 문장은 부정확하게 변환되었습니다.

---

This paper proposes a method for converting silently articulated speech into text using non-invasive electromyographic (EMG) signals. The results show that the proposed method achieves a phoneme error rate (PER) of 49% and a word error rate (WER) of 73.53%, outperforming existing methods. For example, the sentence "IT WAS PAID FOR" was accurately transcribed, while "THE DEATH PENALTY" was transcribed inaccurately.

<br/>
# 기타



**기타(다이어그램, 피규어, 테이블, 어펜딕스 등) 결과와 인사이트**

1. **테이블 1**: 이 테이블은 모델의 성능을 나타내며, 음소 오류율(PER)과 단어 오류율(WER)을 비교합니다. 우리의 방법(σ(τ) 행렬 사용)은 기존의 스펙트로그램 기반 접근 방식보다 PER과 WER 모두에서 현저한 개선을 보여줍니다. 이는 EMG 신호의 공간 구조를 효과적으로 캡처하여 음소 수준의 디코딩을 가능하게 했음을 시사합니다.

2. **그림 3**: 모델 크기와 PER 간의 관계를 보여줍니다. 모델의 파라미터 수가 증가함에 따라 PER이 감소하는 경향을 보이며, 이는 모델 용량이 성능에 긍정적인 영향을 미친다는 것을 나타냅니다. 이는 신경망 모델의 성능이 파라미터 수에 따라 향상될 수 있음을 시사합니다.

3. **그림 4**: 훈련 데이터의 양과 PER 간의 관계를 보여줍니다. 훈련 데이터가 많아질수록 PER이 감소하는 경향을 보이며, 이는 데이터의 양이 성능에 중요한 영향을 미친다는 것을 나타냅니다.

4. **테이블 2**: EMG 2QWERTY 데이터셋에서 우리의 방법과 기존 방법을 비교합니다. 우리의 방법이 모든 피험자에서 성능을 향상시켰음을 보여주며, 이는 EMG 신호의 생리학적 구조를 반영한 접근 방식이 효과적임을 나타냅니다.

5. **그림 6**: 테스트 세트의 WER과 PER 분포를 요약합니다. 평균 PER은 48.47%로, 무작위 추측에 비해 현저히 낮은 수치입니다. 이는 EMG 신호에서 언어적 내용을 효과적으로 디코딩할 수 있음을 시사합니다.

6. **그림 7**: σ(τ) 행렬이 E(τ) 행렬보다 더 희소하다는 것을 보여줍니다. 이는 σ(τ)가 더 나은 디코딩 성능을 제공할 수 있는 이유 중 하나입니다.

7. **어펜딕스 A**: 이전 연구와의 비교를 통해 우리의 접근 방식이 기존의 EMG 기반 음성 인터페이스와 어떻게 다른지를 설명합니다. 특히, 시간 정렬된 EMG-오디오 쌍을 사용하지 않고도 성능을 향상시킬 수 있음을 강조합니다.




**Other (Diagrams, Figures, Tables, Appendix) Results and Insights**

1. **Table 1**: This table presents the performance of the models, comparing phoneme error rate (PER) and word error rate (WER). Our method (using σ(τ) matrices) shows significant improvements in both PER and WER compared to the existing spectrogram-based approaches. This suggests that effectively capturing the spatial structure of EMG signals enables phoneme-level decoding.

2. **Figure 3**: This figure illustrates the relationship between model size and PER. As the number of parameters in the model increases, PER tends to decrease, indicating that model capacity positively impacts performance. This suggests that the performance of neural network models can improve with an increase in the number of parameters.

3. **Figure 4**: This figure shows the relationship between the amount of training data and PER. As the amount of training data increases, PER tends to decrease, indicating that the quantity of data has a significant impact on performance.

4. **Table 2**: This table compares our methods with existing methods on the EMG 2QWERTY dataset. It shows that our approach improves performance for all subjects, indicating that an approach reflecting the physiological structure of EMG signals is effective.

5. **Figure 6**: This figure summarizes the distribution of WER and PER across the test set. The average PER is 48.47%, significantly lower than the chance-level PER, suggesting that linguistic content can be effectively decoded from EMG signals.

6. **Figure 7**: This figure shows that the σ(τ) matrices are sparser than the E(τ) matrices. This sparsity is one of the reasons why σ(τ) can provide better decoding performance.

7. **Appendix A**: This appendix provides a comparison with prior work, explaining how our approach differs from existing EMG-based speech interfaces. It emphasizes the ability to enhance performance without relying on time-aligned EMG-audio pairs.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Gowda2026,
  author    = {Harshavardhana T. Gowda and Lee M. Miller},
  title     = {Non-invasive electromyographic speech neuroprosthesis: a geometric perspective},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {11636--11650},
  year      = {2026},
  month     = {July 2-7},
  publisher  = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Harshavardhana T. Gowda and Lee M. Miller. "Non-invasive Electromyographic Speech Neuroprosthesis: A Geometric Perspective." In *Findings of the Association for Computational Linguistics: ACL 2026*, 11636–11650. July 2-7, 2026. Association for Computational Linguistics. 
    