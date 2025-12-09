---
layout: post
title:  "[2025]Training Diffusion Language Models at Scale using Autoregressive Models"
date:   2025-12-09 20:50:24 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 RND1-Base라는 30B 파라미터의 희소 혼합 전문가(DLM) 모델을 제안, 이 모델은 간단하고 확장 가능한 지속적 사전 훈련 방법을 통해 훈련됨.


큰 그림은 머큐리와 같음: 길이 L인 시퀀스를 통으로 들고, 여러 디퓨전 스텝 동안
마스크된 토큰들을 병렬로 조금씩 고쳐나가는 DLM  
추가적으로 : 기존 AR 거대 모델(Qwen3)에서 어떻게 diffusion 모델로 잘 갈아탈까 (A2D) +  
지식 보존 + MoE로 30B까지 안정적으로 스케일링 레시피  
qwen사용은 사실 Qwen3 AR 모델을 그대로 가져와서, 마스크-디퓨전 loss로 계속 프리트레인하는 것임



즉, Qwen3 AR 체크포인트를 가져온다.
self-attention mask를 causal → bidirectional로 바꾼다.
[MASK] 기반 diffusion corruption 프로세스를 정의한다.
loss를 A2D(MDLM) loss로 바꾸고, 큰 배치/긴 토큰 수로 계속 프리트레인한다.
이때 attention만 세게 업데이트, 나머지는 거의 freeze 수준으로 천천히 움직이게 해서 지식 보존.



짧은 요약(Abstract) :


이 논문은 확산 언어 모델(Diffusion Language Models, DLMs)의 대규모 훈련 방법에 대해 다루고 있습니다. DLM은 병렬적이고 유연한 순서로 텍스트 생성을 가능하게 하여, 자기 회귀 모델(autoregressive models)과 비교할 때 추론 시간 최적화의 새로운 기회를 제공합니다. 그러나 DLM을 대규모로 훈련하는 방법은 아직 충분히 탐구되지 않았습니다. 저자들은 RND1-Base라는 30억 개의 파라미터를 가진 일반 목적의 희소 혼합 전문가 모델을 소개하며, 간단하고 확장 가능한 지속적 사전 훈련 레시피를 사용하여 훈련했습니다. 자기 회귀 모델을 5000억 개의 토큰으로 지속적으로 사전 훈련한 후, A2D(autoregressive-to-diffusion) 변환 레시피를 통해 고용량 DLM을 얻었습니다. RND1-Base는 일반적인 DLM 벤치마크에서 최첨단 성능을 달성하였으며, 8억 개 이상의 파라미터를 가진 DLM을 대규모로 훈련한 첫 번째 공개 연구로 알려져 있습니다. 저자들은 RND1-Base와 그 레시피를 공개하여 DLM의 후속 훈련, 추론 및 구조 혁신 연구를 촉진하고자 합니다.



This paper addresses the large-scale training methods for Diffusion Language Models (DLMs). DLMs enable parallel and flexible-order text generation, offering new opportunities for inference-time optimization compared to autoregressive models. However, recipes for training DLMs at scale remain largely unexplored. The authors introduce RND1-Base, a general-purpose 30 billion parameter sparse mixture-of-experts model trained using a simple and scalable continual pretraining recipe. After continually pretraining an autoregressive base model on 500 billion tokens, they obtain a high-capacity DLM through an autoregressive-to-diffusion (A2D) conversion recipe. RND1-Base achieves state-of-the-art performance on common benchmarks for DLMs, and it is noted as the first open effort to scale DLMs beyond 8 billion parameters. The authors release RND1-Base and their recipe to catalyze research in post-training, inference, and architectural innovation in DLMs.


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



이 논문에서는 Diffusion Language Models (DLMs)의 대규모 훈련을 위한 새로운 방법론을 제안합니다. 특히, RND1-Base라는 30B 파라미터를 가진 희소 혼합 전문가 모델을 소개하며, 이 모델은 간단하고 확장 가능한 지속적 사전 훈련(recipe)을 통해 훈련되었습니다. DLM은 기존의 오토회귀 모델(AR 모델)과 비교하여 병렬적이고 유연한 텍스트 생성을 가능하게 하여 추론 시간 최적화의 새로운 기회를 제공합니다.

#### 모델 아키텍처
RND1-Base는 Mixture-of-Experts (MoE) 아키텍처를 기반으로 하며, 이는 모델의 파라미터 수를 효율적으로 활용할 수 있게 해줍니다. MoE는 여러 전문가 모델을 사용하여 입력에 따라 활성화되는 전문가를 선택함으로써, 계산 자원을 절약하고 성능을 향상시킵니다. 이 모델은 500B 토큰을 사용하여 지속적으로 사전 훈련되었으며, 이는 DLM의 성능을 극대화하는 데 기여합니다.

#### 훈련 데이터
훈련 데이터는 FineWeb-Edu, Dolmino Text, FLAN, Dolmino Math, Stack-Exchange, Wikipedia 등 다양한 출처에서 수집된 500B 토큰으로 구성되어 있습니다. 이러한 다양한 데이터는 모델이 일반적인 상식, STEM(과학, 기술, 공학, 수학) 문제 해결, 코딩 등 여러 분야에서 높은 성능을 발휘할 수 있도록 돕습니다.

#### 특별한 기법
이 논문에서 제안하는 주요 기법은 오토회귀 모델에서 DLM으로의 전환을 위한 A2D(Autoregressive-to-Diffusion) 변환입니다. 이 과정에서 두 가지 주요 도전 과제가 있습니다: 
1. DLM이 양방향 컨텍스트를 활용할 수 있도록 하는 것.
2. A2D 변환 과정에서 AR 모델의 사전 훈련 지식을 보존하는 것.

이러한 도전 과제를 해결하기 위해, 저자들은 간단한 단일 단계 지속적 사전 훈련(recipe)을 제안합니다. 이 방법은 초기화 시 인과적 마스크를 양방향 마스크로 대체하고, A2D 변환 동안 업데이트를 밀집 레이어(MoE 레이어)에 제한하여 AR 모델의 지식을 보존합니다.

이러한 방법론을 통해 RND1-Base는 MMLU, GSM8K, MBPP와 같은 여러 벤치마크에서 최첨단 성능을 달성하였습니다. 이 연구는 DLM의 훈련, 추론 및 아키텍처 혁신에 대한 기초를 제공하며, 모델 가중치, 추론 코드 및 샘플을 공개하여 후속 연구를 촉진하고자 합니다.

---




This paper proposes a novel methodology for training Diffusion Language Models (DLMs) at scale. Specifically, it introduces RND1-Base, a 30B parameter sparse mixture-of-experts model trained using a simple and scalable continual pretraining recipe. DLMs offer parallel and flexible text generation compared to traditional autoregressive models (AR models), providing new opportunities for inference-time optimization.

#### Model Architecture
RND1-Base is based on a Mixture-of-Experts (MoE) architecture, which allows for efficient utilization of model parameters. MoE employs multiple expert models and selects the active expert based on the input, thereby saving computational resources and enhancing performance. The model is continually pretrained on 500B tokens, contributing to maximizing the performance of the DLM.

#### Training Data
The training data consists of 500B tokens collected from various sources, including FineWeb-Edu, Dolmino Text, FLAN, Dolmino Math, Stack-Exchange, and Wikipedia. This diverse dataset helps the model achieve high performance across various domains, including common sense reasoning, STEM problem-solving, and coding.

#### Special Techniques
A key technique proposed in this paper is the A2D (Autoregressive-to-Diffusion) conversion for transitioning from AR models to DLMs. This process presents two main challenges:
1. Enabling DLMs to leverage bidirectional context.
2. Preserving the pretrained knowledge of the AR model during the A2D conversion.

To address these challenges, the authors propose a simple single-stage continual pretraining recipe. This method replaces the causal mask with a bidirectional mask at initialization and restricts updates to dense layers (MoE layers) during A2D conversion to preserve the knowledge of the AR model.

Through this methodology, RND1-Base achieves state-of-the-art performance on various benchmarks such as MMLU, GSM8K, and MBPP. This research provides a foundation for training, inference, and architectural innovation in DLMs, and the authors release model weights, inference code, and samples to catalyze further research.


<br/>
# Results



RND1-Base 모델은 다양한 벤치마크에서 경쟁 모델들과 비교하여 우수한 성능을 보였습니다. 이 모델은 30B 파라미터를 가진 희소 혼합 전문가(Mixture-of-Experts, MoE) 구조로 설계되었으며, 500B 토큰을 사용하여 훈련되었습니다. RND1-Base는 다음과 같은 주요 벤치마크에서 성능을 평가받았습니다:

1. **상식/추론 (Common Sense/Reasoning)**:
   - **MMLU (Massive Multitask Language Understanding)**: RND1-Base는 69.6%의 정확도를 기록하여, Dream-7B (69.5%)와 LLaDA-8B (65.9%)를 초과했습니다.
   - **BBH (Big Bench Hard)**: RND1-Base는 67.5%의 정확도로, Dream-7B (57.9%)와 LLaDA-8B (47.4%)를 크게 앞섰습니다.
   - **ARC-C (AI2 Reasoning Challenge - Challenge)**: RND1-Base는 63.2%의 정확도를 기록하여, Dream-7B (59.8%)와 LLaDA-8B (47.5%)를 초과했습니다.

2. **수학 및 STEM (Science, Technology, Engineering, Mathematics)**:
   - **GSM8K (Grade School Math 8K)**: RND1-Base는 80.0%의 정확도로, Dream-7B (77.2%)와 LLaDA-8B (70.9%)를 초과했습니다.

3. **코딩 (Coding)**:
   - **MBPP (Massive Benchmarks for Programming Problems)**: RND1-Base는 65.4%의 정확도를 기록하여, Dream-7B (56.2%)와 LLaDA-8B (39.0%)를 초과했습니다.

이러한 결과는 RND1-Base가 다양한 작업에서 경쟁 모델들보다 우수한 성능을 발휘함을 보여줍니다. 특히, RND1-Base는 30B 파라미터를 가진 모델 중에서 처음으로 8B 파라미터를 초과하여 훈련된 모델로, DLM(확산 언어 모델) 분야에서의 새로운 기준을 설정했습니다. 

모델의 성능은 훈련 데이터의 양과 질, 그리고 A2D(Autoregressive-to-Diffusion) 전환 과정에서의 효과적인 지식 보존 전략 덕분에 가능했습니다. RND1-Base는 이러한 성과를 통해 DLM의 후속 연구와 아키텍처 혁신을 촉진할 것으로 기대됩니다.

---




The RND1-Base model demonstrated superior performance compared to competing models across various benchmarks. This model is designed with a 30B parameter sparse Mixture-of-Experts (MoE) architecture and was trained on 500B tokens. RND1-Base was evaluated on the following key benchmarks:

1. **Common Sense/Reasoning**:
   - **MMLU (Massive Multitask Language Understanding)**: RND1-Base achieved an accuracy of 69.6%, surpassing Dream-7B (69.5%) and LLaDA-8B (65.9%).
   - **BBH (Big Bench Hard)**: RND1-Base recorded an accuracy of 67.5%, significantly outperforming Dream-7B (57.9%) and LLaDA-8B (47.4%).
   - **ARC-C (AI2 Reasoning Challenge - Challenge)**: RND1-Base achieved 63.2% accuracy, exceeding Dream-7B (59.8%) and LLaDA-8B (47.5%).

2. **Math and STEM (Science, Technology, Engineering, Mathematics)**:
   - **GSM8K (Grade School Math 8K)**: RND1-Base achieved an accuracy of 80.0%, surpassing Dream-7B (77.2%) and LLaDA-8B (70.9%).

3. **Coding**:
   - **MBPP (Massive Benchmarks for Programming Problems)**: RND1-Base recorded an accuracy of 65.4%, exceeding Dream-7B (56.2%) and LLaDA-8B (39.0%).

These results indicate that RND1-Base outperforms competing models across a variety of tasks. Notably, RND1-Base is the first model trained with over 8B parameters in the DLM (Diffusion Language Model) space, setting a new standard in this field.

The model's performance is attributed to the quantity and quality of training data, as well as effective knowledge retention strategies during the A2D (Autoregressive-to-Diffusion) conversion process. RND1-Base is expected to catalyze further research and architectural innovation in the DLM domain due to these achievements.


<br/>
# 예제



이 논문에서는 Diffusion Language Models (DLMs)의 훈련 방법과 성능을 다루고 있습니다. 특히, RND1-Base라는 30B 파라미터를 가진 DLM을 소개하며, 이 모델은 500B 토큰을 사용하여 훈련되었습니다. DLM은 기존의 Autoregressive (AR) 모델과는 달리, 병렬적이고 유연한 순서로 텍스트를 생성할 수 있는 장점이 있습니다.

#### 훈련 데이터와 테스트 데이터

1. **훈련 데이터**: 
   - FineWeb-Edu 데이터셋을 사용하여 DLM을 훈련했습니다. 이 데이터셋은 교육 관련 콘텐츠로 구성되어 있으며, 다양한 주제에 대한 질문과 답변이 포함되어 있습니다.
   - 훈련 과정에서 사용된 데이터는 500B 토큰으로, 이는 대량의 텍스트 데이터를 포함하고 있습니다.

2. **테스트 데이터**:
   - 모델의 성능을 평가하기 위해 여러 벤치마크 데이터셋이 사용되었습니다. 예를 들어:
     - **MMLU (Massive Multitask Language Understanding)**: 일반적인 상식 및 추론 능력을 평가하는 데이터셋으로, RND1-Base는 69.6%의 정확도를 기록했습니다.
     - **GSM8K**: 수학 문제 해결 능력을 평가하는 데이터셋으로, RND1-Base는 80.0%의 정확도를 보였습니다.
     - **MBPP (Massive Benchmarks for Programming Problems)**: 코딩 문제 해결 능력을 평가하는 데이터셋으로, RND1-Base는 65.4%의 정확도를 기록했습니다.

#### 구체적인 작업(Task)

- **상식/추론**: MMLU와 같은 데이터셋을 통해 모델이 일반적인 상식과 논리적 추론을 수행할 수 있는지를 평가합니다.
- **STEM**: GSM8K와 같은 데이터셋을 통해 수학적 문제 해결 능력을 평가합니다.
- **코딩**: MBPP를 통해 프로그래밍 문제를 해결하는 능력을 평가합니다.

이러한 다양한 작업을 통해 RND1-Base는 DLM의 성능을 입증하며, 기존의 AR 모델에 비해 우수한 결과를 보여주었습니다.

---




This paper discusses the training methods and performance of Diffusion Language Models (DLMs). Specifically, it introduces RND1-Base, a 30B parameter DLM trained on 500B tokens. DLMs have the advantage of generating text in a parallel and flexible order, unlike traditional Autoregressive (AR) models.

#### Training Data and Test Data

1. **Training Data**:
   - The model was trained using the FineWeb-Edu dataset, which consists of educational content covering various topics, including questions and answers.
   - The training process utilized 500B tokens of data, which includes a vast amount of text.

2. **Test Data**:
   - Several benchmark datasets were used to evaluate the model's performance. For example:
     - **MMLU (Massive Multitask Language Understanding)**: A dataset that assesses general knowledge and reasoning abilities, where RND1-Base achieved an accuracy of 69.6%.
     - **GSM8K**: A dataset that evaluates mathematical problem-solving abilities, where RND1-Base scored 80.0% accuracy.
     - **MBPP (Massive Benchmarks for Programming Problems)**: A dataset that assesses coding problem-solving abilities, where RND1-Base recorded an accuracy of 65.4%.

#### Specific Tasks

- **Common Sense/Reasoning**: The model's ability to perform general knowledge and logical reasoning is evaluated using datasets like MMLU.
- **STEM**: The model's mathematical problem-solving capabilities are assessed using datasets like GSM8K.
- **Coding**: The model's ability to solve programming problems is evaluated using MBPP.

Through these diverse tasks, RND1-Base demonstrates the performance of DLMs, showing superior results compared to traditional AR models.

<br/>
# 요약


이 논문에서는 RND1-Base라는 30B 파라미터의 희소 혼합 전문가(DLM)를 제안하며, 간단하고 확장 가능한 지속적 사전 훈련 방법을 통해 훈련하였다. RND1-Base는 MMLU, GSM8K, MBPP와 같은 벤치마크에서 최첨단 성능을 달성하였으며, 이는 자가 회귀 모델에서 DLM으로의 전환(A2D) 과정에서 지식 보존을 효과적으로 수행했음을 보여준다. 연구팀은 RND1-Base와 A2D 방법을 공개하여 DLM의 후속 연구를 촉진하고자 한다.

---

This paper introduces RND1-Base, a 30B parameter sparse mixture-of-experts DLM trained using a simple and scalable continual pretraining method. RND1-Base achieves state-of-the-art performance on benchmarks such as MMLU, GSM8K, and MBPP, demonstrating effective knowledge retention during the autoregressive to diffusion model conversion (A2D). The research team releases RND1-Base and the A2D methods to catalyze further research in DLMs.

<br/>
# 기타



1. **Figure 1: Evaluation Benchmark Performance Comparison**
   - **결과**: RND1-Base는 MMLU, GSM8K, BBH, ARC-C, MBPP와 같은 다양한 벤치마크에서 다른 DLM 모델들보다 우수한 성능을 보였다.
   - **인사이트**: RND1-Base는 30B 파라미터를 가진 DLM으로, 기존의 8B 파라미터 모델을 초월하여 성능을 향상시켰음을 보여준다. 이는 DLM의 스케일링 가능성을 입증하는 중요한 결과이다.

2. **Figure 3.1: Probing A2D Recipes at 4B**
   - **결과**: Random initialization, Grafting, Simple Continual Pretraining (SCP) 세 가지 방법의 훈련 손실과 정확도를 비교한 결과, SCP가 가장 우수한 성능을 보였다.
   - **인사이트**: SCP 방법이 DLM 훈련에서 효과적인 접근법임을 나타내며, 기존의 Grafting 방법보다 더 간단하고 효율적이라는 점에서 DLM 훈련의 최적화 가능성을 제시한다.

3. **Figure 3.2: Critical Batch Size (CBS) Estimation for A2D Conversion**
   - **결과**: 배치 크기가 증가함에 따라 검증 손실이 감소하는 경향을 보였으며, CBS는 8M 토큰 이상으로 추정되었다.
   - **인사이트**: DLM 훈련에서 배치 크기와 학습률 조정이 중요하다는 점을 강조하며, DLM이 더 큰 배치 크기를 수용할 수 있음을 보여준다.

4. **Table 4.1: Benchmark Results**
   - **결과**: RND1-Base는 MMLU, GSM8K, MBPP 등에서 다른 모델들보다 높은 정확도를 기록하였다.
   - **인사이트**: RND1-Base의 성능이 기존의 AR 모델(Qwen3-30B-A3B)과 비교하여 여전히 차이가 있지만, A2D 훈련을 통해 이 격차가 줄어들 것으로 기대된다. 이는 DLM의 발전 가능성을 시사한다.

5. **Appendix: Infrastructure and Profiling**
   - **결과**: RND1-Base는 64개의 NVIDIA HGX B200 GPU 클러스터에서 훈련되었으며, 다양한 프로파일링 결과가 제시되었다.
   - **인사이트**: DLM 훈련의 효율성을 높이기 위한 하드웨어 및 소프트웨어 최적화의 중요성을 강조하며, 대규모 모델 훈련에서의 병렬 처리의 이점을 보여준다.

---




1. **Figure 1: Evaluation Benchmark Performance Comparison**
   - **Results**: RND1-Base demonstrated superior performance compared to other DLM models on various benchmarks such as MMLU, GSM8K, BBH, ARC-C, and MBPP.
   - **Insight**: RND1-Base, with its 30B parameters, surpasses the existing 8B parameter models, showcasing the potential for scaling DLMs. This is a significant result that validates the scalability of DLMs.

2. **Figure 3.1: Probing A2D Recipes at 4B**
   - **Results**: The comparison of training loss and accuracy among Random initialization, Grafting, and Simple Continual Pretraining (SCP) showed that SCP performed the best.
   - **Insight**: The effectiveness of the SCP method in DLM training indicates that it is a simpler and more efficient approach compared to the existing Grafting method, suggesting optimization possibilities for DLM training.

3. **Figure 3.2: Critical Batch Size (CBS) Estimation for A2D Conversion**
   - **Results**: There was a trend of decreasing validation loss with increasing batch size, with CBS estimated to be above 8M tokens.
   - **Insight**: This emphasizes the importance of batch size and learning rate adjustments in DLM training, demonstrating that DLMs can accommodate larger batch sizes.

4. **Table 4.1: Benchmark Results**
   - **Results**: RND1-Base achieved higher accuracy than other models on MMLU, GSM8K, and MBPP.
   - **Insight**: While there remains a gap between the performance of RND1-Base and the AR model (Qwen3-30B-A3B), it is expected that this gap will narrow with longer A2D training, indicating the potential for DLM advancement.

5. **Appendix: Infrastructure and Profiling**
   - **Results**: RND1-Base was trained on a cluster of 64 NVIDIA HGX B200 GPUs, with various profiling results presented.
   - **Insight**: This highlights the importance of hardware and software optimization to enhance the efficiency of DLM training, showcasing the benefits of parallel processing in large-scale model training.

<br/>
# refer format:


### BibTeX   



```bibtex
@article{RadicalNumerics2025,
  title={Training Diffusion Language Models at Scale using Autoregressive Models},
  author={Radical Numerics Inc.},
  year={2025},
  journal={arXiv preprint arXiv:2508.15487},
  url={https://arxiv.org/abs/2508.15487},
  note={Accessed: 2025-10-15}
}
```

### 시카고 스타일

Radical Numerics Inc. 2025. "Training Diffusion Language Models at Scale using Autoregressive Models." arXiv preprint arXiv:2508.15487. Accessed October 15, 2025. https://arxiv.org/abs/2508.15487.
