---
layout: post
title:  "[2026]ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment"
date:   2026-07-17 02:50:21 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: ImmersiveTTS는 환경 인식 텍스트-음성 변환(TTS) 모델로, 멀티모달 확산 변환기와 도메인 특화 표현 정렬을 통해 자연스러운 음성을 생성합니다.


짧은 요약(Abstract) :


이 논문에서는 ImmersiveTTS라는 환경 인식 텍스트-음성 변환(TTS) 모델을 제안합니다. 이 모델은 텍스트와 환경 오디오 간의 상호작용을 명시적으로 모델링하여 자연스러운 음성을 생성하고, 이를 환경 맥락에 원활하게 통합합니다. ImmersiveTTS는 다중 모달 확산 변환기(multimodal diffusion transformer)를 기반으로 하며, 텍스트에 조건화된 환경 맥락과 전사에 정렬된 음성 잠재 변수를 결합하여 공동 주의를 통해 상호작용을 모델링합니다. 또한, 환경 인식 TTS에 맞춤화된 도메인 특정 표현 정렬 목표를 도입하여 의미적 일관성을 향상시킵니다. 실험 결과, ImmersiveTTS는 기존 방법들보다 더 높은 자연스러움, 이해 가능성 및 오디오 충실도를 달성했습니다.




This paper proposes ImmersiveTTS, an environment-aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. ImmersiveTTS builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to environment-aware TTS. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches.


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



**모델 및 아키텍처**

ImmersiveTTS는 환경 인식 텍스트-음성 변환(TTS) 모델로, 자연스러운 음성을 생성하면서 환경 오디오와 원활하게 통합하는 것을 목표로 합니다. 이 모델은 멀티모달 확산 변환기(MM-DiT) 아키텍처를 기반으로 하며, 텍스트에 조건화된 환경 맥락과 전사 정렬 음성 잠재 변수를 결합하여 교차 모달 상호작용을 명시적으로 모델링합니다. ImmersiveTTS는 두 개의 병렬 스트림을 사용하여 음성과 환경 정보를 처리합니다. 첫 번째 스트림은 음성 스트림으로, 전사된 텍스트에서 파생된 언어적 특징을 처리합니다. 두 번째 스트림은 환경 스트림으로, 환경 설명을 기반으로 한 세부 정보를 인코딩합니다. 이 두 스트림은 공동 주의(joint attention) 메커니즘을 통해 상호작용하여 음성 생성 과정에서 환경 맥락을 동적으로 반영합니다.

**훈련 데이터 및 기법**

ImmersiveTTS는 LibriTTS와 WavCaps라는 두 개의 데이터셋을 사용하여 훈련됩니다. LibriTTS는 고품질 음성을 제공하며, WavCaps는 다양한 환경 소리를 포함하고 있습니다. 이 두 데이터셋을 혼합하여 훈련 데이터셋을 구성하며, 환경 오디오 샘플은 신호 대 잡음 비율(SNR)을 2에서 10 dB 사이로 조정하여 혼합됩니다. 이를 통해 모델은 깨끗한 음성을 생성하는 동시에 환경 소리와의 통합을 학습합니다.

또한, 모델의 훈련 과정에서 도메인 특화 표현 정렬(objective) 기법이 도입됩니다. 이 기법은 음성과 환경 오디오의 중간 표현을 정렬하여 훈련의 안정성을 높이고, 언어적 명확성과 환경적 충실도를 동시에 유지할 수 있도록 돕습니다. 이를 위해 WavLM과 ATST-Frame이라는 두 개의 자가 지도 학습(SSL) 인코더를 사용하여 음성과 환경 오디오의 특성을 각각 캡처합니다.

**훈련 및 추론**

모델은 네 가지 손실 함수를 사용하여 최적화됩니다: 흐름 매칭 손실(LFlow), 도메인 특화 REPA 손실(LREPA), 텍스트 인코더와 지속 시간 예측기를 위한 사전 손실(LPrior), 그리고 지속 시간 손실(LDur)입니다. 이러한 손실 함수들은 모델이 언어적 내용과 환경적 맥락을 동시에 잘 생성할 수 있도록 돕습니다. 추론 과정에서는 샘플링을 통해 생성된 잠재 변수를 VAE 디코더를 사용하여 음성 파형으로 변환합니다.





**Model and Architecture**

ImmersiveTTS is an environment-aware text-to-speech (TTS) model designed to generate natural speech seamlessly integrated with environmental audio. The model is built on a multimodal diffusion transformer (MM-DiT) architecture, which explicitly models cross-modal interactions by fusing transcript-aligned speech latent with text-conditioned environmental context. ImmersiveTTS employs two parallel streams to process speech and environmental information. The first stream is the speech stream, which processes linguistic features derived from the transcribed text. The second stream is the environmental stream, which encodes details based on environmental descriptions. These two streams interact through a joint attention mechanism, allowing the speech generation process to dynamically reflect the environmental context.

**Training Data and Techniques**

ImmersiveTTS is trained using two datasets: LibriTTS and WavCaps. LibriTTS provides high-quality speech, while WavCaps contains diverse environmental sounds. The training dataset is constructed by mixing these two datasets, with environmental audio samples adjusted to a signal-to-noise ratio (SNR) between 2 and 10 dB. This approach enables the model to learn to generate clean speech while integrating environmental sounds.

Additionally, a domain-specific representation alignment (objective) technique is introduced during the training process. This technique aligns intermediate representations of speech and environmental audio to enhance training stability and maintain both linguistic clarity and environmental fidelity. To achieve this, two self-supervised learning (SSL) encoders, WavLM and ATST-Frame, are utilized to capture the distinct characteristics of speech and environmental audio, respectively.

**Training and Inference**

The model is optimized using four loss functions: flow matching loss (LFlow), domain-specific REPA loss (LREPA), prior loss (LPrior) for the text encoder and duration predictor, and duration loss (LDur). These loss functions help the model generate linguistic content and environmental context effectively. During inference, the generated latent variables are transformed into speech waveforms using a VAE decoder.


<br/>
# Results



이 논문에서는 ImmersiveTTS라는 환경 인식 텍스트-음성 변환(TTS) 모델을 제안하고, 이를 기존의 경쟁 모델들과 비교하여 성능을 평가하였다. 실험은 AudioCaps와 Seed-TTS 데이터셋을 사용하여 진행되었으며, 다양한 메트릭을 통해 모델의 성능을 평가하였다.

#### 경쟁 모델
ImmersiveTTS는 두 가지 주요 경쟁 모델인 V oiceLDM과 V oiceDiT와 비교되었다. 이들 모델은 자연어 환경 설명을 기반으로 음성을 생성하는 방식으로 설계되었다. ImmersiveTTS는 이러한 모델들과 비교하여 더 높은 자연스러움과 통합된 음성을 생성하는 것을 목표로 하였다.

#### 테스트 데이터
- **AudioCaps**: 이 데이터셋은 다양한 환경 소리와 함께 음성을 포함하고 있어, 환경 인식 TTS의 성능을 평가하는 데 적합하다.
- **Seed-TTS**: 이 데이터셋은 깨끗한 음성을 포함하고 있으며, 환경 소리와의 혼합을 통해 모델의 성능을 평가하는 데 사용되었다.

#### 메트릭
모델의 성능은 다음과 같은 메트릭을 통해 평가되었다:
- **SN-MOS (Speech Naturalness Mean Opinion Score)**: 생성된 음성의 자연스러움을 평가.
- **EC-MOS (Environmental Consistency Mean Opinion Score)**: 배경 소리가 주어진 환경 설명과 얼마나 잘 일치하는지를 평가.
- **ON-MOS (Overall Integration Naturalness Mean Opinion Score)**: 음성과 배경 소리가 얼마나 자연스럽게 통합되었는지를 평가.
- **WER (Word Error Rate)**: 음성 인식의 정확성을 평가.
- **FAD (Fréchet Audio Distance)**: 생성된 음성과 목표 음성 간의 분포 거리를 측정.
- **CLAP Score**: 텍스트와 생성된 음성 간의 의미적 일치를 평가.

#### 비교 결과
ImmersiveTTS는 다음과 같은 결과를 보였다:
- **SN-MOS**: ImmersiveTTS는 4.20의 점수를 기록하여 V oiceLDM(3.41) 및 V oiceDiT(3.47)보다 높은 자연스러움을 보였다.
- **EC-MOS**: ImmersiveTTS는 3.48로, V oiceLDM(3.33) 및 V oiceDiT(3.44)와 비슷한 성능을 보였다.
- **ON-MOS**: ImmersiveTTS는 3.47로, 기존 모델들보다 더 나은 통합 자연스러움을 보였다.
- **WER**: ImmersiveTTS는 8.06로, V oiceLDM(16.45) 및 V oiceDiT(11.68)보다 낮은 오류율을 기록하였다.
- **FAD**: ImmersiveTTS는 5.80으로, V oiceLDM(8.75) 및 V oiceDiT(9.07)보다 낮은 값을 보였다.
- **CLAP Score**: ImmersiveTTS는 0.308로, V oiceLDM(0.229) 및 V oiceDiT(0.263)보다 높은 의미적 일치를 보였다.

이러한 결과들은 ImmersiveTTS가 기존의 경쟁 모델들보다 더 높은 자연스러움과 음성-환경 일치를 제공함을 보여준다. 또한, ImmersiveTTS는 더 적은 샘플링 단계(25 NFE)로도 높은 품질을 달성할 수 있어 효율성에서도 우수한 성능을 보였다.

---




In this paper, the authors propose a model called ImmersiveTTS, which is an environment-aware text-to-speech (TTS) model, and evaluate its performance against existing competitive models. The experiments were conducted using the AudioCaps and Seed-TTS datasets, and various metrics were employed to assess the model's performance.

#### Competitive Models
ImmersiveTTS was compared with two main competitive models, V oiceLDM and V oiceDiT. These models are designed to generate speech based on natural language environmental descriptions. ImmersiveTTS aimed to generate speech with higher naturalness and integration compared to these models.

#### Test Data
- **AudioCaps**: This dataset contains various environmental sounds along with speech, making it suitable for evaluating the performance of environment-aware TTS.
- **Seed-TTS**: This dataset includes clean speech and was used to evaluate the model's performance through mixing with environmental sounds.

#### Metrics
The performance of the models was evaluated using the following metrics:
- **SN-MOS (Speech Naturalness Mean Opinion Score)**: Evaluates the naturalness of the generated speech.
- **EC-MOS (Environmental Consistency Mean Opinion Score)**: Assesses how well the background sound matches the given environmental description.
- **ON-MOS (Overall Integration Naturalness Mean Opinion Score)**: Evaluates how naturally the speech and background sound are blended.
- **WER (Word Error Rate)**: Measures the accuracy of speech recognition.
- **FAD (Fréchet Audio Distance)**: Measures the distribution distance between generated and target audio.
- **CLAP Score**: Evaluates the semantic alignment between the text and the generated audio.

#### Comparison Results
ImmersiveTTS showed the following results:
- **SN-MOS**: ImmersiveTTS achieved a score of 4.20, higher than V oiceLDM (3.41) and V oiceDiT (3.47), indicating better naturalness.
- **EC-MOS**: ImmersiveTTS scored 3.48, similar to V oiceLDM (3.33) and V oiceDiT (3.44).
- **ON-MOS**: ImmersiveTTS scored 3.47, indicating better overall integration than existing models.
- **WER**: ImmersiveTTS recorded a WER of 8.06, lower than V oiceLDM (16.45) and V oiceDiT (11.68).
- **FAD**: ImmersiveTTS achieved a FAD of 5.80, lower than V oiceLDM (8.75) and V oiceDiT (9.07).
- **CLAP Score**: ImmersiveTTS scored 0.308, higher than V oiceLDM (0.229) and V oiceDiT (0.263).

These results demonstrate that ImmersiveTTS provides higher naturalness and speech-environment alignment compared to existing competitive models. Additionally, ImmersiveTTS achieved high quality with fewer sampling steps (25 NFEs), showcasing its efficiency.


<br/>
# 예제



**ImmersiveTTS 모델의 훈련 및 테스트 데이터 예시**

1. **훈련 데이터 구성**
   - **입력**: 훈련 데이터는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 **텍스트 전사**(content prompt)로, 예를 들어 "나는 오늘 매우 행복하다."와 같은 문장이 포함됩니다. 두 번째는 **환경 설명**(environment prompt)으로, 예를 들어 "새가 지저귀고 있다."와 같은 문장이 포함됩니다.
   - **출력**: 모델의 출력은 두 입력을 기반으로 생성된 **오디오 파형**(audio waveform)입니다. 이 오디오는 주어진 텍스트 전사와 환경 설명이 잘 통합된 자연스러운 음성을 포함해야 합니다.

2. **테스트 데이터 구성**
   - **입력**: 테스트 데이터는 훈련 데이터와 유사한 형식으로 구성됩니다. 예를 들어, "비가 내리고 있다."라는 환경 설명과 "여성의 목소리로 말하고 있다."라는 텍스트 전사가 주어질 수 있습니다.
   - **출력**: 모델은 이 입력을 바탕으로 생성된 오디오를 출력합니다. 이 오디오는 주어진 환경 설명과 텍스트 전사가 잘 조화된 음성을 포함해야 하며, 자연스러움과 일관성을 평가받습니다.

3. **테스트 절차**
   - 모델의 성능을 평가하기 위해, 여러 테스트 샘플이 사용됩니다. 각 샘플은 주어진 텍스트 전사와 환경 설명에 따라 생성된 오디오를 포함하며, 평가자는 이 오디오의 자연스러움, 환경 일치성, 전반적인 통합 자연스러움 등을 평가합니다.

4. **예시**
   - **훈련 샘플**:
     - 입력: "나는 오늘 매우 행복하다." (텍스트 전사), "새가 지저귀고 있다." (환경 설명)
     - 출력: 자연스러운 여성의 목소리로 "나는 오늘 매우 행복하다."라는 문장이 새 소리와 함께 배경으로 들리는 오디오.
   - **테스트 샘플**:
     - 입력: "비가 내리고 있다." (환경 설명), "여성의 목소리로 말하고 있다." (텍스트 전사)
     - 출력: 비 소리와 함께 "여성의 목소리로 말하고 있다."라는 문장이 자연스럽게 들리는 오디오.




**Example of Training and Testing Data for the ImmersiveTTS Model**

1. **Training Data Composition**
   - **Input**: The training data consists of two main components. The first is the **text transcription** (content prompt), which may include sentences like "I am very happy today." The second is the **environment description** (environment prompt), which could be something like "A bird is chirping."
   - **Output**: The model's output is an **audio waveform** generated based on these two inputs. This audio should contain natural speech that is well-integrated with the environmental context.

2. **Testing Data Composition**
   - **Input**: The testing data is structured similarly to the training data. For example, it might include an environment description like "It is raining." and a text transcription like "A woman is speaking."
   - **Output**: The model generates audio based on this input. The output should include speech that harmonizes well with the given environment description and text transcription, and it will be evaluated for naturalness and coherence.

3. **Testing Procedure**
   - To evaluate the model's performance, multiple test samples are used. Each sample includes audio generated based on the provided text transcription and environment description, and evaluators assess the naturalness, environmental consistency, and overall integration of the audio.

4. **Example**
   - **Training Sample**:
     - Input: "I am very happy today." (text transcription), "A bird is chirping." (environment description)
     - Output: An audio where a natural female voice says "I am very happy today." with the background sound of chirping birds.
   - **Testing Sample**:
     - Input: "It is raining." (environment description), "A woman is speaking." (text transcription)
     - Output: An audio where the phrase "A woman is speaking." is heard naturally along with the sound of rain in the background.

<br/>
# 요약


ImmersiveTTS는 환경 인식 텍스트-음성 변환(TTS) 모델로, 멀티모달 확산 변환기와 도메인 특화 표현 정렬을 통해 자연스러운 음성을 생성합니다. 실험 결과, ImmersiveTTS는 기존 방법들보다 높은 자연스러움, 이해도 및 음질을 달성하며, 다양한 환경 오디오와의 통합에서도 우수한 성능을 보였습니다. 이 모델은 텍스트 프롬프트를 통해 환경 정보를 직접 지정하여 음성과 배경 오디오의 일관성을 높이는 데 기여합니다.

---

ImmersiveTTS is an environment-aware text-to-speech (TTS) model that generates natural speech through a multimodal diffusion transformer and domain-specific representation alignment. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity compared to existing methods, demonstrating superior performance in integrating with various environmental audio. This model enhances coherence between speech and background audio by directly specifying environmental information through text prompts.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: ImmersiveTTS의 전체 파이프라인을 보여주며, 이중 스트림 MM-DiT 아키텍처가 어떻게 환경 정보와 콘텐츠 정보를 통합하여 음성을 생성하는지를 설명합니다. 이 구조는 언어적 내용과 환경적 맥락 간의 상호작용을 명시적으로 모델링하여 자연스러운 음성을 생성하는 데 기여합니다.
   - **Figure 3**: 이중 스트림 DiT 블록의 내부 흐름을 보여줍니다. 이 구조는 환경 프롬프트와 콘텐츠 프롬프트 간의 상호작용을 통해 음성을 생성하는 과정을 시각적으로 설명합니다.

2. **테이블**
   - **Table 1 & Table 2**: ImmersiveTTS의 성능을 기존 모델들과 비교한 결과를 보여줍니다. ImmersiveTTS는 SN-MOS(음성 자연스러움) 및 ON-MOS(전체 통합 자연스러움)에서 높은 점수를 기록하며, 이는 모델이 더 자연스러운 음성을 생성하고 환경 오디오와 잘 통합된다는 것을 나타냅니다.
   - **Table 3**: 단일 작업 평가 결과를 보여주며, ImmersiveTTS가 낮은 WER(단어 오류율)과 높은 UTMOS(음성 자연스러움의 객관적 지표)를 기록하여 음성 인식의 정확성을 높였음을 나타냅니다.
   - **Table 4**: 다양한 교사 설정에 따른 REPA(Representation Alignment) 효과를 분석한 결과를 보여줍니다. WavLM과 ATST-Frame의 조합이 가장 우수한 성능을 보이며, 이는 각 교사가 제공하는 도메인별 지침이 상호 보완적임을 시사합니다.

3. **어펜딕스**
   - **Appendix A**: ImmersiveTTS의 구현 세부 사항을 설명하며, 모델의 구조와 훈련 방법에 대한 정보를 제공합니다.
   - **Appendix B**: 주관적 평가의 세부 사항을 설명하며, MOS 테스트에서 평가된 다양한 측면(음성 자연스러움, 환경 일관성 등)에 대한 정보를 제공합니다.
   - **Appendix C**: REPA 주입 위치에 대한 분석을 제공하며, 중간 레이어에서의 정렬이 가장 안정적이라는 것을 보여줍니다.

### Insights and Results from Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the overall pipeline of ImmersiveTTS, showing how the dual-stream MM-DiT architecture integrates environmental and content information to generate speech. This structure explicitly models the interactions between linguistic content and environmental context, contributing to the generation of natural-sounding speech.
   - **Figure 3**: Provides a visual representation of the internal flow of the double-stream DiT block, explaining the process of generating speech through interactions between the environmental and content prompts.

2. **Tables**
   - **Table 1 & Table 2**: Present the performance comparison of ImmersiveTTS with existing models. ImmersiveTTS achieves high scores in SN-MOS (speech naturalness) and ON-MOS (overall integration naturalness), indicating that the model generates more natural speech that is well-integrated with environmental audio.
   - **Table 3**: Shows single-task evaluation results, indicating that ImmersiveTTS records a lower WER (word error rate) and higher UTMOS (objective measure of speech naturalness), enhancing the accuracy of speech recognition.
   - **Table 4**: Analyzes the effect of different teacher configurations on REPA (Representation Alignment). The combination of WavLM and ATST-Frame yields the best performance, suggesting that the domain-specific guidance provided by each teacher is complementary.

3. **Appendices**
   - **Appendix A**: Describes the implementation details of ImmersiveTTS, providing information about the model's architecture and training methods.
   - **Appendix B**: Details the subjective evaluation process, explaining the various aspects assessed in the MOS test (speech naturalness, environmental consistency, etc.).
   - **Appendix C**: Provides an analysis of the injection position for REPA, showing that alignment in the middle layers is the most stable, consistent with prior findings.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Yun2026,
  author    = {Jun-Hak Yun and Seung-Bin Kim and Seong-Whan Lee},
  title     = {ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {38295--38314},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Jun-Hak Yun, Seung-Bin Kim, and Seong-Whan Lee. "ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 38295–38314. Association for Computational Linguistics, July 2026.
