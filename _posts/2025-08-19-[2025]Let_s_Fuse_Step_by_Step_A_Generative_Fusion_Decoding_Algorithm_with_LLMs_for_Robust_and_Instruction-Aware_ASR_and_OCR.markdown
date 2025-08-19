---
layout: post
title:  "[2025]Let’s Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Robust and Instruction-Aware ASR and OCR"
date:   2025-08-19 03:16:56 -0000
categories: study
---

{% highlight ruby %}

한줄 요약:  


언어 모델(LLMs)을 자동 음성 인식(ASR) 및 광학 문자 인식(OCR) 시스템에 통합하는 Generative Fusion Decoding (GFD)라는 새로운 얕은 융합 프레임워크를 제안(재훈련 없이 다양한 자기 회귀 모델과 호환되도록 설계)     


서로 다른 모델의 불일치하는 토큰 공간을 처리하기 위해 바이트 수준에서 우도(likelihood)를 계산하는 방식을 사용하여, 디코딩 과정에서의 원활한 융합과 동기적 진행을 가능     


짧은 요약(Abstract) :



이 논문에서는 "Generative Fusion Decoding" (GFD)라는 새로운 얕은 융합 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 자동 음성 인식(ASR) 및 광학 문자 인식(OCR) 시스템에 통합하기 위해 설계되었습니다. GFD는 서로 다른 모델의 불일치하는 토큰 공간에서 작동할 수 있도록 필요한 수식을 도출하여 바이트 수준에서 우도를 계산함으로써 원활한 융합과 동기화된 진행을 가능하게 합니다. GFD는 재훈련 없이 다양한 자기 회귀 모델과 호환되도록 설계되어 있으며, LLM과의 중간 및 빈번한 상호작용을 통해 일반 ASR 및 OCR 작업에서 효과적임을 입증하였습니다. GFD는 LLM의 맥락 내 학습 능력을 전이하고, 지침 인식 및 긴 맥락 설정에서 적응형 ASR을 가능하게 하여 최대 17.7%의 단어 오류율(WER) 감소를 가져옵니다.




In this paper, we propose "Generative Fusion Decoding" (GFD), a novel shallow fusion framework designed to integrate large language models (LLMs) into automatic speech recognition (ASR) and optical character recognition (OCR) systems. We derive the necessary formulations to enable GFD to operate across mismatched token spaces of different models by calculating likelihood at the byte level, thereby enabling seamless fusion and synchronous progression during the decoding process. GFD is plug-and-play by design, making it readily compatible with various auto-regressive models without the need for any re-training. GFD proves effective for general ASR and OCR tasks through intermediate and frequent interactions with LLMs, surpassing cascaded methods in English and Mandarin benchmarks. Additionally, GFD transfers in-context learning abilities of LLMs and allows for adaptive ASR in instruction-aware and long-context settings, yielding significant WER reductions of up to 17.7%.


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



본 논문에서 제안하는 방법론은 "Generative Fusion Decoding" (GFD)이라는 새로운 얕은 융합 프레임워크로, 대형 언어 모델(LLMs)을 자동 음성 인식(ASR) 및 광학 문자 인식(OCR) 시스템에 통합하기 위해 설계되었습니다. GFD는 서로 다른 모델의 불일치하는 토큰 공간을 처리하기 위해 바이트 수준에서 우도(likelihood)를 계산하는 방식을 사용하여, 디코딩 과정에서의 원활한 융합과 동기적 진행을 가능하게 합니다.

GFD는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **모델 아키텍처**: GFD는 기존의 ASR 및 OCR 모델과 LLM을 통합하는 방식으로, 각 모델의 아키텍처를 변경하지 않고도 사용할 수 있는 플러그 앤 플레이 구조를 가지고 있습니다. 이는 추가적인 재훈련 없이도 다양한 자동 회귀 모델과 호환될 수 있도록 설계되었습니다.

2. **바이트 수준의 우도 계산**: GFD는 서로 다른 모델의 토큰 공간이 불일치할 때 발생하는 문제를 해결하기 위해, 바이트 수준에서 우도를 계산합니다. 이를 통해 각 모델의 출력이 바이트 시퀀스와 일치하도록 하여, 디코딩 단계에서 효과적인 융합을 가능하게 합니다.

3. **실시간 오류 수정**: GFD는 ASR 및 OCR 모델이 제안한 시퀀스에 대해 LLM이 실시간으로 피드백을 제공하여 텍스트 인식 오류를 수정할 수 있도록 합니다. 이는 인식 정확도를 향상시키고, 다양한 ASR 시나리오에서의 성능을 개선하는 데 기여합니다.

4. **맥락 인식 및 적응형 ASR**: GFD는 LLM의 맥락 인식 능력을 활용하여 긴 맥락을 처리하고, 지시 사항에 따라 적응형 ASR을 수행할 수 있습니다. 이를 통해 특정 도메인에 대한 민감도와 지시 인식 능력을 향상시킬 수 있습니다.

5. **실험적 검증**: GFD는 영어 및 중국어(만다린) 벤치마크에서 기존의 방법론을 초월하는 성능을 입증하였으며, 특히 동음이의어와 코드 스위칭과 같은 도전적인 시나리오에서 효과적임을 보여주었습니다.

이러한 GFD의 설계는 LLM의 강력한 기능을 ASR 및 OCR 시스템에 통합하여, 다양한 조건에서 일관된 성능 향상을 이루는 것을 목표로 하고 있습니다.

---




The methodology proposed in this paper is called "Generative Fusion Decoding" (GFD), a novel shallow fusion framework designed to integrate large language models (LLMs) into automatic speech recognition (ASR) and optical character recognition (OCR) systems. GFD addresses the challenges of mismatched token spaces between different models by calculating likelihood at the byte level, enabling seamless fusion and synchronous progression during the decoding process.

GFD consists of the following key components:

1. **Model Architecture**: GFD integrates existing ASR and OCR models with LLMs in a plug-and-play structure that does not require changes to the architecture of each model. This design allows for compatibility with various auto-regressive models without the need for any retraining.

2. **Byte-Level Likelihood Calculation**: To address the issues arising from mismatched token spaces of different models, GFD calculates likelihoods at the byte level. This allows the outputs of each model to align with the byte sequences, facilitating effective fusion during the decoding stage.

3. **Real-Time Error Correction**: GFD enables LLMs to provide real-time feedback on sequences proposed by ASR and OCR models, allowing for the correction of text recognition errors. This contributes to improved recognition accuracy and enhances performance across various ASR scenarios.

4. **Contextual Awareness and Adaptive ASR**: GFD leverages the contextual awareness capabilities of LLMs to handle long contexts and perform adaptive ASR based on instructions. This enhances domain sensitivity and instruction awareness.

5. **Experimental Validation**: GFD has been empirically validated to outperform existing methods on English and Mandarin benchmarks, demonstrating effectiveness in challenging scenarios such as homophones and code-switching.

The design of GFD aims to integrate the powerful capabilities of LLMs into ASR and OCR systems, achieving consistent performance improvements across diverse conditions.


<br/>
# Results



이 논문에서는 "Generative Fusion Decoding" (GFD)라는 새로운 프레임워크를 제안하여 대규모 언어 모델(LLM)을 자동 음성 인식(ASR) 및 광학 문자 인식(OCR) 시스템에 통합하는 방법을 설명합니다. GFD는 다양한 테스트 데이터셋에서 기존의 경쟁 모델들과 비교하여 성능을 평가하였으며, 그 결과는 다음과 같습니다.

1. **경쟁 모델**: GFD는 Whisper, RobustGER, Clairaudience와 같은 여러 최신 ASR 모델과 비교되었습니다. Whisper는 대규모 음성 인식 모델로, 다양한 데이터셋에서 성능을 평가하는 데 사용되었습니다. RobustGER는 LLM을 활용한 오류 수정 모델로, GFD와의 성능 비교에 중요한 역할을 했습니다.

2. **테스트 데이터**: GFD는 Librispeech, Medical, NTUML2021, FormosaSpeech와 같은 다양한 데이터셋에서 평가되었습니다. Librispeech는 오디오북에서 수집된 데이터로, "Clean" 및 "Other" 하위 집합으로 나뉘어 있습니다. Medical 데이터셋은 의료 대화로 구성되어 있으며, NTUML2021은 대학교 강의 녹음으로 이루어져 있습니다. FormosaSpeech는 대만식 중국어 음성 데이터로, 다양한 발음과 억양을 포함하고 있습니다.

3. **메트릭**: 성능 평가는 주로 단어 오류율(Word Error Rate, WER)과 혼합 오류율(Mixed Error Rate, MER)로 측정되었습니다. OCR 작업에서는 문자 오류율(Character Error Rate, CER)과 정확한 일치율(Exact Match, EM)도 사용되었습니다.

4. **비교 결과**: GFD는 다양한 ASR 시나리오에서 성능을 개선하였으며, 특히 NTUML2021 데이터셋에서 8.83의 혼합 오류율을 기록하여 기존의 최적 모델인 Oracle N-Best 점수를 초과하는 성과를 보였습니다. GFD는 LLM의 문법 오류 수정 및 도메인 특정 용어 인식에서 우수한 성능을 발휘하였습니다. 또한, GFD는 노이즈가 있는 환경에서도 성능이 향상되었으며, 특히 중간 정도의 노이즈 조건에서 가장 큰 개선을 보였습니다.

5. **결론**: GFD는 LLM을 ASR 및 OCR 시스템에 효과적으로 통합할 수 있는 가능성을 보여주며, 다양한 조건에서 일관된 성능 향상을 입증하였습니다. GFD는 기존의 융합 방법들과 비교하여 유연성과 효율성을 제공하며, 향후 연구에서 LLM의 강점을 더욱 활용할 수 있는 기초를 마련하였습니다.




This paper proposes a novel framework called "Generative Fusion Decoding" (GFD) to integrate large language models (LLMs) into automatic speech recognition (ASR) and optical character recognition (OCR) systems. The performance of GFD was evaluated against various competitive models across different test datasets, and the results are as follows:

1. **Competitive Models**: GFD was compared with several state-of-the-art ASR models, including Whisper, RobustGER, and Clairaudience. Whisper is a large-scale speech recognition model used to evaluate performance across various datasets. RobustGER is an error correction model that utilizes LLMs, playing a crucial role in the performance comparison with GFD.

2. **Test Datasets**: GFD was evaluated on a variety of datasets, including Librispeech, Medical, NTUML2021, and FormosaSpeech. Librispeech consists of data collected from audiobooks, divided into "Clean" and "Other" subsets. The Medical dataset comprises medical conversations, while NTUML2021 consists of lecture recordings from a university. FormosaSpeech includes Taiwanese Mandarin speech data, featuring various pronunciations and tones.

3. **Metrics**: Performance was primarily measured using Word Error Rate (WER) and Mixed Error Rate (MER). For OCR tasks, Character Error Rate (CER) and Exact Match (EM) were also utilized.

4. **Comparison Results**: GFD demonstrated performance improvements across various ASR scenarios, achieving a mixed error rate of 8.83 on the NTUML2021 dataset, surpassing the best-performing Oracle N-Best score. GFD excelled in correcting grammatical errors and recognizing domain-specific terminology due to the capabilities of LLMs. Additionally, GFD showed enhanced performance in noisy environments, particularly under moderate noise conditions.

5. **Conclusion**: GFD showcases the potential for effectively integrating LLMs into ASR and OCR systems, demonstrating consistent performance improvements across diverse conditions. GFD offers flexibility and efficiency compared to existing fusion methods, laying the groundwork for future research to further exploit the strengths of pre-trained language models.


<br/>
# 예제



이 논문에서는 "Generative Fusion Decoding" (GFD)이라는 새로운 알고리즘을 제안합니다. GFD는 자동 음성 인식(ASR) 및 광학 문자 인식(OCR) 시스템에 대규모 언어 모델(LLM)을 통합하기 위해 설계된 얕은 융합 프레임워크입니다. GFD는 서로 다른 모델의 토큰 공간이 일치하지 않는 문제를 해결하기 위해 바이트 수준에서 우도(likelihood)를 계산하여 원활한 융합과 동기화된 진행을 가능하게 합니다.

#### 예시: ASR 작업

1. **트레이닝 데이터**:
   - **입력**: "안녕하세요, 오늘 날씨는 어떻습니까?" (음성 데이터)
   - **출력**: "안녕하세요, 오늘 날씨는 어떻습니까?" (텍스트 데이터)

2. **테스트 데이터**:
   - **입력**: "안녕하세요, 오늘 날씨는 어떻습니까?" (음성 데이터)
   - **출력**: "안녕하세요, 오늘 날씨는 어떻습니까?" (텍스트 데이터)

이 예시에서 ASR 시스템은 음성 입력을 텍스트로 변환하는 작업을 수행합니다. GFD는 LLM을 사용하여 ASR의 출력을 실시간으로 수정하고, 문맥에 따라 더 정확한 결과를 제공합니다.

#### 예시: OCR 작업

1. **트레이닝 데이터**:
   - **입력**: 이미지 파일 (예: 문서 스캔)
   - **출력**: "이 문서는 2023년 10월 1일에 작성되었습니다." (텍스트 데이터)

2. **테스트 데이터**:
   - **입력**: 이미지 파일 (예: 문서 스캔)
   - **출력**: "이 문서는 2023년 10월 1일에 작성되었습니다." (텍스트 데이터)

이 예시에서 OCR 시스템은 이미지에서 텍스트를 인식하는 작업을 수행합니다. GFD는 LLM을 통해 OCR의 출력을 개선하여 긴 텍스트 시퀀스의 인식 정확도를 높입니다.




This paper proposes a novel algorithm called "Generative Fusion Decoding" (GFD), designed to integrate large language models (LLMs) into automatic speech recognition (ASR) and optical character recognition (OCR) systems. GFD addresses the challenge of mismatched token spaces between different models by calculating likelihood at the byte level, enabling seamless fusion and synchronous progression during the decoding process.

#### Example: ASR Task

1. **Training Data**:
   - **Input**: "안녕하세요, 오늘 날씨는 어떻습니까?" (audio data)
   - **Output**: "안녕하세요, 오늘 날씨는 어떻습니까?" (text data)

2. **Test Data**:
   - **Input**: "안녕하세요, 오늘 날씨는 어떻습니까?" (audio data)
   - **Output**: "안녕하세요, 오늘 날씨는 어떻습니까?" (text data)

In this example, the ASR system performs the task of converting spoken input into text. GFD utilizes the LLM to correct the ASR output in real-time, providing more accurate results based on context.

#### Example: OCR Task

1. **Training Data**:
   - **Input**: Image file (e.g., document scan)
   - **Output**: "이 문서는 2023년 10월 1일에 작성되었습니다." (text data)

2. **Test Data**:
   - **Input**: Image file (e.g., document scan)
   - **Output**: "이 문서는 2023년 10월 1일에 작성되었습니다." (text data)

In this example, the OCR system performs the task of recognizing text from images. GFD enhances the OCR output by leveraging the LLM to improve recognition accuracy for long text sequences.

<br/>
# 요약


본 연구에서는 "Generative Fusion Decoding" (GFD)라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLM)을 자동 음성 인식(ASR) 및 광학 문자 인식(OCR) 시스템에 통합하였다. GFD는 바이트 수준에서의 가능성을 계산하여 서로 다른 모델 간의 토큰 공간 불일치를 해결하고, 실시간으로 텍스트 인식 오류를 수정하는 데 효과적임을 입증하였다. 실험 결과, GFD는 다양한 ASR 및 OCR 작업에서 기존 방법보다 우수한 성능을 보였으며, 특히 긴 맥락과 지시 기반 설정에서의 성능 향상이 두드러졌다.



This study proposes a novel framework called "Generative Fusion Decoding" (GFD) to integrate large language models (LLMs) into automatic speech recognition (ASR) and optical character recognition (OCR) systems. GFD effectively resolves token space mismatches between different models by calculating likelihoods at the byte level, enabling real-time correction of text recognition errors. Experimental results demonstrate that GFD outperforms existing methods across various ASR and OCR tasks, particularly excelling in long-context and instruction-aware settings.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: GFD 통합 프레임워크를 보여줍니다. 이 다이어그램은 ASR/OCR 모델과 LLM 간의 통합 과정을 시각적으로 설명하며, 바이트 수준에서의 가능성 계산을 통해 서로 다른 모델 간의 토큰 공간 불일치를 해결하는 방법을 강조합니다. 이로 인해 GFD는 다양한 입력 시퀀스를 지원할 수 있습니다.
   - **Figure 2**: 주요 시퀀스와 대체 토큰의 예시를 보여줍니다. 이 피규어는 바이트 시퀀스와 토큰 시퀀스 간의 관계를 설명하며, GFD가 어떻게 바이트 수준에서의 확률 계산을 통해 효과적인 통합을 수행하는지를 시각적으로 나타냅니다.

2. **테이블**
   - **Table 1**: 짧은 형식의 음성 인식 성능을 보여줍니다. GFD는 다양한 데이터셋에서 WER(Word Error Rate)를 줄이는 데 효과적임을 나타내며, 특히 NTUML2021 데이터셋에서 오라클 N-Best 점수를 초과하는 성능을 보였습니다. 이는 GFD가 코드 스위칭과 같은 복잡한 시나리오에서 LLM의 문법적 오류 수정 능력을 활용할 수 있음을 시사합니다.
   - **Table 2**: GFD와 RobustGER의 성능 비교를 보여줍니다. GFD는 RobustGER와 유사한 개선을 보이며, 두 방법의 조합이 최상의 결과를 도출함을 나타냅니다.
   - **Table 3**: 지시 인식 ASR 작업에서 GFD의 성능을 보여줍니다. GFD는 도메인 태그 및 희귀 단어 프롬프트를 활용하여 성능을 향상시키며, Whisper와 비교하여 더 나은 결과를 도출합니다.
   - **Table 4**: 장기 형식 음성 인식 성능을 보여줍니다. GFD는 NTUML2021 및 FormosaSpeech에서 Whisper보다 일관되게 우수한 성능을 보이며, LLM의 장기 컨텍스트 처리 능력을 효과적으로 활용하고 있음을 나타냅니다.
   - **Table 5**: OCR 작업에서 GFD의 성능을 보여줍니다. GFD는 TrOCR 모델과 통합하여 문자 오류율을 16.7% 줄이고, 정확한 일치를 38.07% 향상시킵니다.

3. **어펜딕스**
   - **Appendix A**: GFD의 계산 효율성 및 실험 세부 사항을 설명합니다. GFD의 바이트 수준 확률 계산이 O(1)로 효율적임을 보여주며, 이는 모델 포워딩에 대한 비용이 상대적으로 낮음을 나타냅니다. 또한, 실험에서 사용된 프롬프트 세부 사항을 제공하여 GFD의 성능 향상에 기여한 요소들을 명확히 합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: This figure illustrates the GFD integrated framework, visually explaining the integration process between ASR/OCR models and LLMs. It emphasizes how likelihood calculations at the byte level address token space mismatches between different models, enabling GFD to support various input sequences.
   - **Figure 2**: This figure provides an example of the main sequence and alternative tokens. It explains the relationship between byte sequences and token sequences, visually demonstrating how GFD performs effective integration through byte-level probability calculations.

2. **Tables**
   - **Table 1**: This table presents the performance of short-form speech recognition. GFD shows effectiveness in reducing WER (Word Error Rate) across various datasets, notably exceeding oracle N-Best scores in the NTUML2021 dataset. This suggests that GFD can leverage the LLM's ability to correct grammatical errors in complex scenarios like code-switching.
   - **Table 2**: This table compares the performance of GFD with RobustGER. GFD demonstrates similar improvements to RobustGER, indicating that the combination of both methods yields the best results.
   - **Table 3**: This table showcases the performance of GFD in instruction-aware ASR tasks. GFD enhances performance by utilizing domain tags and rare word prompts, achieving better results compared to Whisper.
   - **Table 4**: This table illustrates the performance of long-form speech recognition. GFD consistently outperforms Whisper in both NTUML2021 and FormosaSpeech, effectively utilizing the long-context capabilities of LLMs.
   - **Table 5**: This table evaluates GFD's performance in OCR tasks. GFD significantly reduces character error rates by 16.7% and improves exact match rates by 38.07% when integrated with the TrOCR model.

3. **Appendices**
   - **Appendix A**: This section explains the computational efficiency of GFD and provides experimental details. It shows that the byte-level probability calculations are efficient at O(1), indicating relatively low costs compared to model forwarding. Additionally, it provides details on the prompts used in experiments, clarifying the factors that contributed to the performance improvements of GFD.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Hsu2025,
  author    = {Chan-Jan Hsu and Yi-Chang Chen and Feng-Ting Liao and Pei-Chen Ho and Yu-Hsiang Wang and Po-Chun Hsu and Da-shan Shiu},
  title     = {Let’s Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Robust and Instruction-Aware ASR and OCR},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {24959--24973},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  address   = {July 27 - August 1, 2025}
}
```

### 시카고 스타일

Chan-Jan Hsu, Yi-Chang Chen, Feng-Ting Liao, Pei-Chen Ho, Yu-Hsiang Wang, Po-Chun Hsu, and Da-shan Shiu. "Let’s Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Robust and Instruction-Aware ASR and OCR." In *Findings of the Association for Computational Linguistics: ACL 2025*, 24959–24973. Association for Computational Linguistics, July 27 - August 1, 2025.
