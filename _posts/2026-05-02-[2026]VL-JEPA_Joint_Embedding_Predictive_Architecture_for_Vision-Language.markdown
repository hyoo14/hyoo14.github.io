---
layout: post
title:  "[2026]VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language"
date:   2026-05-02 03:02:48 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: VL-JEPA는 비전-언어 작업을 위한 Joint Embedding Predictive Architecture를 기반으로 하며, 연속적인 임베딩 예측을 통해 효율성을 높이고 훈련 파라미터를 50% 줄였습니다.


짧은 요약(Abstract) :

VL-JEPA는 비전-언어 모델로, Joint Embedding Predictive Architecture (JEPA)를 기반으로 구축되었습니다. 기존의 전통적인 비전-언어 모델(VLM)들이 토큰을 자가 회귀적으로 생성하는 방식과는 달리, VL-JEPA는 목표 텍스트의 연속적인 임베딩을 예측합니다. 이 모델은 추상적인 표현 공간에서 학습함으로써, 표면적인 언어적 변동성을 추상화하고 작업과 관련된 의미에 집중합니다. 동일한 비전 인코더와 훈련 데이터를 사용한 엄격한 비교에서, VL-JEPA는 50% 적은 학습 가능한 매개변수로 더 강력한 성능을 달성했습니다. 추론 시, 경량의 텍스트 디코더는 VL-JEPA가 예측한 임베딩을 텍스트로 변환할 필요가 있을 때만 호출됩니다. VL-JEPA는 선택적 디코딩을 본래 지원하여, 비적응형 균일 디코딩에 비해 약 2.85배의 디코딩 작업 수를 줄이면서 유사한 성능을 유지합니다. 생성 작업을 넘어, VL-JEPA의 임베딩 공간은 아키텍처 수정 없이도 개방 어휘 분류, 텍스트-비디오 검색 및 판별적 VQA를 자연스럽게 지원합니다.




VL-JEPA is a vision-language model built on a Joint Embedding Predictive Architecture (JEPA). Unlike traditional vision-language models (VLMs) that autoregressively generate tokens, VL-JEPA predicts continuous embeddings of the target texts. By learning in an abstract representation space, the model focuses on task-relevant semantics while abstracting away surface-level linguistic variability. In a strictly controlled comparison against standard token-space VLM training with the same vision encoder and training data, VL-JEPA achieves stronger performance while having 50% fewer trainable parameters. At inference time, a lightweight text decoder is invoked only when needed to translate VL-JEPA predicted embeddings into text. VL-JEPA natively supports selective decoding that reduces the number of decoding operations by approximately 2.85 times while maintaining similar performance compared to non-adaptive uniform decoding. Beyond generation, VL-JEPA’s embedding space naturally supports open-vocabulary classification, text-to-video retrieval, and discriminative VQA without any architecture modification.


* Useful sentences :

크게보면 비전트랜스포머의 임베딩과 llm의 임베딩을 가져와서 같이 잘 사용(align->joint embedding space)해서 답이 되는 임베딩을 출력하도록 학습   
근데 clip처럼 contrastive learning은 아니고 prediction task를 잘 하도록(뭐 예를 들면 vqa의 정답을)  



{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology


VL-JEPA(비전-언어 모델)는 Joint Embedding Predictive Architecture(JEPA)를 기반으로 한 새로운 비전-언어 모델입니다. 이 모델은 전통적인 비전-언어 모델(VLM)과는 달리, 연속적인 임베딩을 예측하는 방식으로 작동합니다. VL-JEPA는 다음과 같은 주요 구성 요소와 방법론을 포함합니다.

1. **모델 아키텍처**:
   - VL-JEPA는 네 가지 주요 구성 요소로 이루어져 있습니다: X-Encoder, Predictor, Y-Encoder, Y-Decoder.
   - **X-Encoder**는 시각적 입력(X_V)을 압축하여 시각적 임베딩(S_V)으로 변환합니다. 이 과정에서 고해상도 비디오 프레임을 처리할 수 있는 비전 트랜스포머(V-JEPA 2)를 사용합니다.
   - **Predictor**는 시각적 임베딩(S_V)과 텍스트 쿼리(X_Q)를 입력으로 받아 목표 임베딩(S_Y)을 예측합니다. 이 과정에서 Llama 3 트랜스포머 레이어를 초기화하여 사용합니다.
   - **Y-Encoder**는 텍스트 목표(Y)를 연속적인 잠재 공간으로 임베딩하여 예측 목표로 사용합니다. 이 임베딩은 작업과 관련 없는 정보를 추상화하는 데 중점을 둡니다.
   - **Y-Decoder**는 예측된 임베딩(S_Y)을 텍스트로 변환하는 역할을 하며, 주로 추론 시에만 사용됩니다.

2. **트레이닝 방법**:
   - VL-JEPA는 두 단계로 훈련됩니다. 첫 번째 단계는 대규모 캡션 데이터를 사용하여 비전-언어 정렬을 확립하는 사전 훈련 단계입니다. 이 단계에서는 이미지와 비디오 캡션을 포함한 대규모 데이터셋을 사용합니다.
   - 두 번째 단계는 감독된 미세 조정(Supervised Fine-Tuning, SFT) 단계로, 이 단계에서는 VQA(Visual Question Answering) 기능을 모델에 추가합니다. 이 과정에서 다양한 VQA 샘플과 캡션 샘플을 포함한 데이터가 사용됩니다.

3. **효율성**:
   - VL-JEPA는 비전-언어 모델의 훈련 및 추론 효율성을 크게 향상시킵니다. 특히, 임베딩 공간에서의 예측을 통해 훈련 효율성을 높이고, 선택적 디코딩을 통해 추론 시의 지연 시간을 줄입니다. 선택적 디코딩은 예측된 임베딩 스트림에서 중요한 변화가 감지될 때만 디코딩을 수행하여 불필요한 계산을 줄입니다.

4. **다중 작업 처리**:
   - VL-JEPA는 단일 아키텍처로 다양한 비전-언어 작업을 처리할 수 있습니다. 예를 들어, 캡션 생성, 개방형 VQA, 텍스트-비디오 검색 등 다양한 작업을 수행할 수 있습니다. 이 모델은 비전-언어 작업을 위한 통합된 접근 방식을 제공하여, 여러 작업을 동시에 처리할 수 있는 능력을 갖추고 있습니다.

이러한 방식으로 VL-JEPA는 기존의 비전-언어 모델보다 더 효율적이고 강력한 성능을 발휘하며, 실시간 비디오 애플리케이션에 적합한 특성을 가지고 있습니다.

---


VL-JEPA (Vision-Language Model) is a new vision-language model built on the Joint Embedding Predictive Architecture (JEPA). Unlike traditional Vision-Language Models (VLMs), this model operates by predicting continuous embeddings. VL-JEPA includes the following key components and methodologies:

1. **Model Architecture**:
   - VL-JEPA consists of four main components: X-Encoder, Predictor, Y-Encoder, and Y-Decoder.
   - The **X-Encoder** compresses visual input (X_V) into visual embeddings (S_V). It utilizes a vision transformer (V-JEPA 2) capable of processing high-resolution video frames.
   - The **Predictor** takes visual embeddings (S_V) and a textual query (X_Q) as input to predict the target embedding (S_Y). It initializes using Llama 3 transformer layers.
   - The **Y-Encoder** embeds the textual target (Y) into a continuous latent space as the prediction target, focusing on abstracting away task-irrelevant information.
   - The **Y-Decoder** translates the predicted embedding (S_Y) into text, primarily used during inference.

2. **Training Method**:
   - VL-JEPA is trained in two stages. The first stage is a pre-training phase using large caption data to establish robust vision-language alignment. This phase utilizes large datasets that include images and video captions.
   - The second stage is a supervised fine-tuning (SFT) phase, which equips the model with Visual Question Answering (VQA) capabilities. This process involves data that includes various VQA samples and caption samples.

3. **Efficiency**:
   - VL-JEPA significantly enhances the training and inference efficiency of vision-language models. In particular, it improves training efficiency through predictions in the embedding space and reduces inference latency through selective decoding. Selective decoding performs decoding only when significant changes are detected in the predicted embedding stream, minimizing unnecessary computations.

4. **Multi-tasking**:
   - VL-JEPA can handle a variety of vision-language tasks using a single, unified architecture. For example, it can perform caption generation, open-domain VQA, and text-to-video retrieval, providing an integrated approach for vision-language tasks and enabling the ability to handle multiple tasks simultaneously.

In this way, VL-JEPA demonstrates more efficient and powerful performance compared to existing vision-language models, making it well-suited for real-time video applications.


<br/>
# Results


VL-JEPA 모델은 여러 비전-언어 작업에서 경쟁 모델들과 비교하여 우수한 성능을 보여주었습니다. 이 모델은 두 가지 주요 버전인 VL-JEPA BASE와 VL-JEPA SFT로 나뉘며, 각각의 성능은 다양한 데이터셋에서 평가되었습니다.

1. **비디오 분류 및 텍스트-비디오 검색**:
   - VL-JEPA BASE는 1.6B 파라미터를 가지고 있으며, 8개의 비디오 분류 데이터셋과 8개의 텍스트-비디오 검색 데이터셋에서 평가되었습니다. 이 모델은 평균적으로 52.5%의 정확도를 기록하며, PE-Core 모델(2.3B 파라미터)보다 높은 성능을 보였습니다. 특히, VL-JEPA BASE는 SSv2, EK-100, EgoExo4D와 같은 동작 중심의 벤치마크에서 두드러진 성과를 보였습니다.
   - VL-JEPA SFT는 같은 파라미터 수를 유지하면서도, 75.4%의 평균 정확도를 기록하여, 기존의 전문 모델들과 비슷한 성능을 달성했습니다.

2. **시각적 질문 응답(VQA)**:
   - VL-JEPA SFT는 GQA, TallyQA, POPE, POPEv2와 같은 VQA 데이터셋에서 평가되었습니다. 이 모델은 GQA에서 61.5%, TallyQA에서 69.9%, POPE에서 85.7%, POPEv2에서 86.3%의 정확도를 기록하여, 기존의 VLM 모델들과 비교했을 때 경쟁력 있는 성능을 보였습니다.

3. **효율성**:
   - VL-JEPA는 선택적 디코딩을 통해 디코딩 작업의 수를 약 2.85배 줄이면서도 성능을 유지할 수 있었습니다. 이는 실시간 비디오 스트리밍과 같은 응용 프로그램에서 매우 유용합니다. 선택적 디코딩은 모델이 의미의 변화를 감지할 때만 디코딩을 수행하도록 하여, 불필요한 계산을 줄이고 응답 속도를 높입니다.

4. **비교 모델**:
   - VL-JEPA는 CLIP, SigLIP2, Perception Encoder와 같은 기존의 비전-언어 모델들과 비교되었습니다. VL-JEPA BASE는 평균적으로 52.5%의 정확도를 기록하며, PE-Core보다 높은 성능을 보였습니다. VL-JEPA SFT는 75.4%의 정확도로, 기존의 전문 모델들과 비슷한 성능을 달성했습니다.

이러한 결과들은 VL-JEPA가 비전-언어 작업에서 효율성과 성능을 동시에 달성할 수 있는 강력한 모델임을 보여줍니다.

---



The VL-JEPA model demonstrated superior performance compared to various competitive models across multiple vision-language tasks. This model is divided into two main versions: VL-JEPA BASE and VL-JEPA SFT, and their performance was evaluated on various datasets.

1. **Video Classification and Text-to-Video Retrieval**:
   - VL-JEPA BASE, with 1.6B parameters, was evaluated on 8 video classification datasets and 8 text-to-video retrieval datasets. It achieved an average accuracy of 52.5%, outperforming the PE-Core model (2.3B parameters). Notably, VL-JEPA BASE excelled in motion-centric benchmarks such as SSv2, EK-100, and EgoExo4D.
   - VL-JEPA SFT maintained the same parameter count while achieving an average accuracy of 75.4%, approaching the performance of existing specialist models.

2. **Visual Question Answering (VQA)**:
   - VL-JEPA SFT was evaluated on VQA datasets including GQA, TallyQA, POPE, and POPEv2. The model achieved an accuracy of 61.5% on GQA, 69.9% on TallyQA, 85.7% on POPE, and 86.3% on POPEv2, demonstrating competitive performance compared to existing VLM models.

3. **Efficiency**:
   - VL-JEPA reduced the number of decoding operations by approximately 2.85 times through selective decoding while maintaining performance. This is particularly beneficial for applications like real-time video streaming. Selective decoding allows the model to perform decoding only when significant semantic changes are detected, reducing unnecessary computations and improving response speed.

4. **Comparison Models**:
   - VL-JEPA was compared with existing vision-language models such as CLIP, SigLIP2, and Perception Encoder. VL-JEPA BASE achieved an average accuracy of 52.5%, outperforming PE-Core. VL-JEPA SFT reached an accuracy of 75.4%, comparable to existing specialist models.

These results indicate that VL-JEPA is a powerful model capable of achieving both efficiency and performance in vision-language tasks.


<br/>
# 예제


VL-JEPA 모델은 비전-언어 작업을 수행하기 위해 설계된 Joint Embedding Predictive Architecture를 기반으로 합니다. 이 모델은 훈련 데이터로 이미지와 텍스트 쌍을 사용하여 비전 입력(X_V), 텍스트 쿼리(X_Q), 그리고 예측할 텍스트 타겟(Y)의 삼중 쌍을 학습합니다. 

#### 훈련 데이터 예시
- **비전 입력 (X_V)**: 특정 이미지 또는 비디오 프레임의 시퀀스. 예를 들어, "고양이가 나무 위에 앉아 있는 이미지".
- **텍스트 쿼리 (X_Q)**: "이 이미지에서 무엇을 볼 수 있나요?"와 같은 질문.
- **타겟 텍스트 (Y)**: "고양이"와 같은 이미지에 대한 설명.

이러한 삼중 쌍을 사용하여 VL-JEPA는 비전 입력을 임베딩(S_V)으로 변환하고, 텍스트 쿼리를 조건으로 하여 타겟 임베딩(S_Y)을 예측합니다. 이 과정에서 모델은 비전 입력과 텍스트 쿼리의 조합을 통해 타겟 텍스트의 임베딩을 예측하는 방법을 학습합니다.

#### 테스트 데이터 예시
테스트 데이터는 훈련 데이터와 유사한 형식으로 제공됩니다. 예를 들어, 테스트 이미지로 "강아지가 공원에서 뛰어노는 모습"이 있을 수 있습니다. 이 경우, 텍스트 쿼리는 "이 이미지에서 어떤 동물이 있나요?"가 될 수 있습니다. 모델은 이 입력을 바탕으로 "강아지"라는 타겟 텍스트를 예측해야 합니다.

#### 구체적인 작업
VL-JEPA는 다양한 비전-언어 작업을 수행할 수 있습니다. 예를 들어:
1. **비디오 캡셔닝**: 비디오의 특정 장면에 대한 설명을 생성합니다.
   - 입력: 비디오 프레임과 질문
   - 출력: 해당 장면에 대한 설명 텍스트
2. **비디오 분류**: 비디오가 어떤 카테고리에 속하는지를 분류합니다.
   - 입력: 비디오 프레임
   - 출력: "스포츠", "음악", "교육" 등과 같은 카테고리
3. **비디오-텍스트 검색**: 주어진 텍스트 쿼리에 대해 관련 비디오를 검색합니다.
   - 입력: "고양이가 놀고 있는 비디오"
   - 출력: 해당 비디오의 임베딩

이러한 작업을 통해 VL-JEPA는 비전-언어 모델의 효율성을 높이고, 실시간 응답성을 개선하며, 다양한 비전-언어 작업을 동시에 처리할 수 있는 능력을 갖추게 됩니다.

---



The VL-JEPA model is built on a Joint Embedding Predictive Architecture designed for vision-language tasks. This model uses image-text pairs as training data to learn triplets consisting of visual input (X_V), textual query (X_Q), and the target text (Y) to be predicted.

#### Training Data Example
- **Visual Input (X_V)**: A specific image or a sequence of video frames. For example, "an image of a cat sitting on a tree."
- **Textual Query (X_Q)**: A question like "What can you see in this image?"
- **Target Text (Y)**: A description such as "cat."

Using these triplets, VL-JEPA transforms the visual input into an embedding (S_V) and predicts the target embedding (S_Y) conditioned on the textual query. In this process, the model learns how to predict the target text's embedding based on the combination of visual input and textual query.

#### Testing Data Example
The testing data is provided in a similar format to the training data. For instance, a test image might be "a dog playing in the park." In this case, the textual query could be "What animal is in this image?" The model is expected to predict the target text "dog."

#### Specific Tasks
VL-JEPA can perform various vision-language tasks. For example:
1. **Video Captioning**: Generating descriptions for specific scenes in a video.
   - Input: Video frames and a question
   - Output: Descriptive text for that scene
2. **Video Classification**: Classifying what category a video belongs to.
   - Input: Video frames
   - Output: Categories like "sports," "music," "education," etc.
3. **Text-to-Video Retrieval**: Searching for relevant videos based on a given text query.
   - Input: "A video of a cat playing"
   - Output: The embedding of that video

Through these tasks, VL-JEPA enhances the efficiency of vision-language models, improves real-time responsiveness, and possesses the capability to handle a wide range of vision-language tasks simultaneously.

<br/>


VL-JEPA는 비전-언어 작업을 위한 Joint Embedding Predictive Architecture를 기반으로 하며, 연속적인 임베딩 예측을 통해 효율성을 높이고 훈련 파라미터를 50% 줄였습니다. 실험 결과, VL-JEPA는 CLIP 및 기존 VLMs보다 우수한 성능을 보였으며, 특히 비디오 분류 및 텍스트-비디오 검색에서 뛰어난 결과를 기록했습니다. 이 모델은 선택적 디코딩을 통해 실시간 응답성을 유지하면서도 효율적인 추론을 가능하게 합니다.

---

VL-JEPA is based on a Joint Embedding Predictive Architecture for vision-language tasks, enhancing efficiency and reducing training parameters by 50% through continuous embedding prediction. Experimental results show that VL-JEPA outperforms CLIP and existing VLMs, particularly excelling in video classification and text-to-video retrieval. The model enables efficient inference while maintaining real-time responsiveness through selective decoding.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **VL-JEPA 아키텍처 (Figure 1)**: VL-JEPA는 비전 입력을 임베딩으로 변환하는 X-Encoder, 텍스트 타겟을 임베딩으로 변환하는 Y-Encoder, 그리고 이 두 임베딩을 기반으로 타겟 임베딩을 예측하는 Predictor로 구성되어 있습니다. 이 구조는 전통적인 VLM과 달리 비생성적이며, 임베딩 공간에서의 예측을 통해 학습 효율성을 높입니다.
   - **VL-JEPA의 응용 (Figure 2)**: VL-JEPA는 캡셔닝, VQA, 텍스트-비디오 검색 등 다양한 비전-언어 작업을 수행할 수 있는 단일 통합 아키텍처를 제공합니다. 이 구조는 선택적 디코딩을 지원하여 실시간 응답성을 높입니다.

2. **테이블**
   - **비디오 분류 및 텍스트-비디오 검색 성능 (Table 1)**: VL-JEPA BASE와 SFT 모델은 다양한 데이터셋에서 기존 모델들보다 우수한 성능을 보였습니다. 특히, VL-JEPA SFT는 1.6B 파라미터로도 전문 모델에 근접한 성능을 발휘했습니다.
   - **VQA 벤치마크 (Table 2)**: VL-JEPA SFT는 GQA, TallyQA, POPE, POPEv2 데이터셋에서 기존 VLM 모델들과 비교하여 경쟁력 있는 성능을 보였습니다. 이는 VL-JEPA가 비전-언어 작업에서의 일반화 능력을 갖추고 있음을 나타냅니다.

3. **어펜딕스**
   - **모델 아키텍처 및 훈련 방법론**: VL-JEPA는 두 단계의 훈련 과정을 거칩니다. 첫 번째 단계는 대규모 캡션 데이터를 사용하여 비전-언어 정렬을 확립하는 사전 훈련 단계이며, 두 번째 단계는 VQA 기능을 강화하기 위한 감독된 미세 조정 단계입니다. 이 과정에서 VL-JEPA는 다양한 작업을 수행할 수 있는 능력을 갖추게 됩니다.



1. **Diagrams and Figures**
   - **VL-JEPA Architecture (Figure 1)**: VL-JEPA consists of an X-Encoder that transforms visual inputs into embeddings, a Y-Encoder that converts textual targets into embeddings, and a Predictor that predicts target embeddings based on these two embeddings. This structure is non-generative compared to traditional VLMs, enhancing learning efficiency through predictions in the embedding space.
   - **Applications of VL-JEPA (Figure 2)**: VL-JEPA can perform various vision-language tasks such as captioning, VQA, and text-to-video retrieval using a single unified architecture. This structure supports selective decoding, improving real-time responsiveness.

2. **Tables**
   - **Video Classification and Text-to-Video Retrieval Performance (Table 1)**: Both VL-JEPA BASE and SFT models outperformed existing models across various datasets. Notably, VL-JEPA SFT achieved performance comparable to specialist models with only 1.6B parameters.
   - **VQA Benchmarks (Table 2)**: VL-JEPA SFT demonstrated competitive performance against established VLM models on datasets like GQA, TallyQA, POPE, and POPEv2. This indicates VL-JEPA's capability for generalization in vision-language tasks.

3. **Appendix**
   - **Model Architecture and Training Methodology**: VL-JEPA undergoes a two-stage training process. The first stage is a pre-training phase using large-scale caption data to establish robust vision-language alignment, while the second stage is a supervised fine-tuning phase that enhances VQA capabilities. Through this process, VL-JEPA acquires the ability to handle a wide range of tasks effectively.

<br/>
# refer format:


### BibTeX 형식

```bibtex
@inproceedings{chen2026vl,
  title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language},
  author={Delong Chen and Mustafa Shukor and Th\'eo Moutakanni and Willy Chung and Jade Yu and Tejaswi Kasarla and Allen Bolourchi and Yann LeCun and Pascale Fung},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026},
  organization={Meta FAIR, HKUST, Sorbonne Université, NYU, University of Amsterdam, USC}
}
```

### 시카고 스타일

Chen, Delong, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Allen Bolourchi, Yann LeCun, and Pascale Fung. 2026. "VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language." In *Proceedings of the International Conference on Learning Representations (ICLR)*. Meta FAIR, HKUST, Sorbonne Université, NYU, University of Amsterdam, USC.
