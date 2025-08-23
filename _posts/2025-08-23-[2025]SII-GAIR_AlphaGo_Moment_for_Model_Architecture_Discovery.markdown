---
layout: post
title:  "[2025]SII-GAIR: AlphaGo Moment for Model Architecture Discovery"
date:   2025-08-23 20:30:25 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

아래 3개의 모듈을 통해 스스로 새로운 구조 제안 및 실험까지 ai가 하도록 하는 것을 제안  
Researcher-propose new model architecture and implement, Engineer-do experiments, Analyst-analyzes results


짧은 요약(Abstract) :



이 논문은 인공지능 연구의 속도가 인간의 인지 능력에 의해 제한되는 문제를 해결하기 위해, 인공지능이 스스로 신경망 아키텍처를 혁신할 수 있는 완전 자율 시스템인 ASI-ARCH를 소개합니다. 기존의 신경망 아키텍처 검색(NAS)은 인간이 정의한 공간 내에서만 탐색할 수 있는 한계가 있었지만, ASI-ARCH는 자동 최적화에서 자동 혁신으로의 패러다임 전환을 제안합니다. 이 시스템은 새로운 아키텍처 개념을 가설로 세우고, 이를 실행 가능한 코드로 구현하며, 엄격한 실험을 통해 성능을 검증합니다. ASI-ARCH는 20,000 GPU 시간을 사용하여 1,773개의 자율 실험을 수행하였고, 106개의 혁신적이고 최첨단의 선형 주의 메커니즘 아키텍처를 발견했습니다. 이 시스템은 인간이 설계한 기준을 초월하는 설계 원칙을 보여주며, 과학적 발견 자체를 위한 첫 번째 경험적 스케일링 법칙을 확립하여 연구 진행을 인간의 한계에서 계산 확장 가능한 과정으로 전환합니다. 우리는 이러한 돌파구를 가능하게 한 설계 패턴과 자율 연구 능력에 대한 포괄적인 분석을 제공하며, AI 주도 연구를 민주화하기 위해 전체 프레임워크, 발견된 아키텍처 및 인지 흔적을 오픈 소스화합니다.




While AI systems demonstrate exponentially improving capabilities, the pace of AI research itself remains linearly bounded by human cognitive capacity, creating an increasingly severe development bottleneck. We present ASI-ARCH, the first demonstration of Artificial Superintelligence for AI research (ASI4AI) in the critical domain of neural architecture discovery—a fully autonomous system that shatters this fundamental constraint by enabling AI to conduct its own architectural innovation. Moving beyond traditional Neural Architecture Search (NAS), which is fundamentally limited to exploring human-defined spaces, we introduce a paradigm shift from automated optimization to automated innovation. ASI-ARCH can conduct end-to-end scientific research in the challenging domain of architecture discovery, autonomously hypothesizing novel architectural concepts, implementing them as executable code, training and empirically validating their performance through rigorous experimentation and past human and AI experience. ASI-ARCH conducted 1,773 autonomous experiments over 20,000 GPU hours, culminating in the discovery of 106 innovative, state-of-the-art (SOTA) linear attention architectures. Like AlphaGo’s Move 37 that revealed unexpected strategic insights invisible to human players, our AI-discovered architectures demonstrate emergent design principles that systematically surpass human-designed baselines and illuminate previously unknown pathways for architectural innovation. Crucially, we establish the first empirical scaling law for scientific discovery itself—demonstrating that architectural breakthroughs can be scaled computationally, transforming research progress from a human-limited to a computation-scalable process. We provide comprehensive analysis of the emergent design patterns and autonomous research capabilities that enabled these breakthroughs, establishing a blueprint for self-accelerating AI systems. To democratize AI-driven research, we open-source the complete framework, discovered architectures, and cognitive traces.


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



ASI-ARCH는 자율적인 아키텍처 발견을 위한 폐쇄 루프 시스템으로, 세 가지 핵심 역할을 중심으로 구성된 모듈형 프레임워크입니다. 이 시스템은 연구자 모듈, 엔지니어 모듈, 분석가 모듈로 구성되어 있으며, 각각의 역할은 다음과 같습니다:

1. **연구자 모듈**: 이 모듈은 AI가 독립적으로 새로운 모델 아키텍처를 제안하는 창의적인 엔진 역할을 합니다. 연구자 모듈은 역사적 경험과 인간의 전문 지식을 바탕으로 고품질의 아키텍처 혁신을 보장하고, 반복적인 탐색을 방지하기 위해 네 가지 주요 메커니즘을 구현합니다. 이 메커니즘에는 시드 선택, 모델 설계, 프로그램 구현, 그리고 참신성과 타당성 검사가 포함됩니다.

2. **엔지니어 모듈**: 이 모듈은 실제 코드 환경에서의 정량적 평가와 LLM-기반의 정성적 평가를 통해 모델을 실험적으로 평가합니다. ASI-ARCH는 강력한 자기 수정 메커니즘을 갖추고 있으며, 에러 로그를 분석하고 코드를 수정하는 반복적인 디버깅 루프를 통해 훈련이 성공할 때까지 지속적으로 개선합니다. 또한, 자동 품질 보증 시스템을 통해 훈련 로그를 실시간으로 모니터링하여 비효율적인 설계를 조기에 종료하고, 자원을 절약합니다.

3. **분석가 모듈**: 이 모듈은 실험 결과를 분석하여 새로운 통찰력을 얻고, 이를 바탕으로 다음 설계 단계를 위한 두 가지 지식 원천을 제공합니다. 하나는 인간의 전문 지식에서 파생된 인지 기반이고, 다른 하나는 시스템의 자체 실험 기록에서 동적으로 생성된 분석입니다. 인지 기반은 선형 주의 메커니즘 분야의 주요 논문에서 추출된 지식을 포함하며, 분석가는 실험의 특정 단점을 요약하여 인지 기반과의 연관성을 찾습니다.

이러한 모듈들은 협력하여 ASI-ARCH가 자율적으로 아키텍처를 탐색하고 검증할 수 있도록 하며, 이를 통해 인간의 개입 없이도 혁신적인 설계를 지속적으로 발전시킬 수 있습니다.




The ASI-ARCH framework operates as a closed-loop system for autonomous architecture discovery, structured around a modular framework with three core roles: the Researcher, the Engineer, and the Analyst. Each role is defined as follows:

1. **Researcher Module**: This module serves as the creative engine of the system, where AI independently proposes novel model architectures based on historical experience and human expertise. The Researcher module implements four key mechanisms to ensure high-quality architectural innovations while preventing repeated explorations: seed selection, model design, program implementation, and novelty and sanity checks.

2. **Engineer Module**: This module conducts empirical evaluations of models through quantitative evaluation in a real code environment and qualitative scoring by an LLM-based judge. ASI-ARCH features a robust self-revision mechanism, where error logs are analyzed and code is revised in an iterative debugging loop until training is successful. An automated quality assurance system monitors training logs in real-time to terminate inefficient designs early, saving resources.

3. **Analyst Module**: This module analyzes experimental results to acquire new insights, providing two distinct sources of knowledge for subsequent design steps: cognition derived from accumulated human expertise and analysis generated dynamically from the system's own experimental history. The cognition base includes knowledge extracted from seminal papers in the field of linear attention, while the Analyst summarizes specific shortcomings observed in experiments to find relevant solutions.

These modules work collaboratively to enable ASI-ARCH to autonomously explore and verify architectures, allowing for continuous innovation without human intervention.


<br/>
# Results



이 논문에서는 ASI-ARCH 시스템이 발견한 AI-설계 모델과 기존의 인간-설계 모델 간의 성능을 비교합니다. 성능 비교는 언어 모델링과 제로샷 상식 추론 작업을 통해 이루어졌습니다. 테스트 데이터셋으로는 Wiki, LMB, PIQA, Hella, Wino, ARC-e, ARC-c, SIQA, BoolQ가 사용되었습니다. 각 데이터셋에 대해 다양한 메트릭이 사용되었으며, 언어 모델링에서는 perplexity (ppl) 지표가, 상식 추론에서는 정확도 (acc) 지표가 사용되었습니다.

#### 경쟁 모델
- **Mamba2**: 인간-설계 모델
- **Gated DeltaNet**: 인간-설계 모델
- **DeltaNet**: 인간-설계 모델
- **PathGateFusionNet**: AI-설계 모델
- **ContentSharpRouter**: AI-설계 모델
- **FusionGatedFIRNet**: AI-설계 모델
- **HierGateNet**: AI-설계 모델
- **AdaMultiPathGateNet**: AI-설계 모델

#### 테스트 데이터 및 메트릭
- **Wiki, LMB**: 언어 모델링에서 perplexity (ppl) 지표 사용. 낮을수록 좋음.
- **PIQA, Hella, Wino, ARC-e, ARC-c, SIQA, BoolQ**: 상식 추론에서 정확도 (acc) 지표 사용. 높을수록 좋음.

#### 결과 비교
- **언어 모델링**: AI-설계 모델들이 대체로 인간-설계 모델보다 낮은 perplexity를 보여, 더 나은 성능을 나타냈습니다.
- **제로샷 상식 추론**: AI-설계 모델들이 여러 데이터셋에서 높은 정확도를 기록하며, 인간-설계 모델과 비교했을 때 경쟁력 있는 성능을 보였습니다.

이 결과는 AI-설계 모델들이 인간-설계 모델에 비해 더 효율적이고 효과적인 설계를 통해 다양한 작업에서 우수한 성능을 발휘할 수 있음을 시사합니다.

---




In this paper, the performance of AI-designed models discovered by the ASI-ARCH system is compared with that of existing human-designed models. The performance comparison is conducted through tasks of language modeling and zero-shot common-sense reasoning. The test datasets used include Wiki, LMB, PIQA, Hella, Wino, ARC-e, ARC-c, SIQA, and BoolQ. Various metrics are used for each dataset, with perplexity (ppl) being used for language modeling and accuracy (acc) for common-sense reasoning.

#### Competing Models
- **Mamba2**: Human-designed model
- **Gated DeltaNet**: Human-designed model
- **DeltaNet**: Human-designed model
- **PathGateFusionNet**: AI-designed model
- **ContentSharpRouter**: AI-designed model
- **FusionGatedFIRNet**: AI-designed model
- **HierGateNet**: AI-designed model
- **AdaMultiPathGateNet**: AI-designed model

#### Test Data and Metrics
- **Wiki, LMB**: Perplexity (ppl) metric used in language modeling. Lower is better.
- **PIQA, Hella, Wino, ARC-e, ARC-c, SIQA, BoolQ**: Accuracy (acc) metric used in common-sense reasoning. Higher is better.

#### Results Comparison
- **Language Modeling**: AI-designed models generally showed lower perplexity than human-designed models, indicating better performance.
- **Zero-shot Common-Sense Reasoning**: AI-designed models recorded higher accuracy across various datasets, demonstrating competitive performance compared to human-designed models.

These results suggest that AI-designed models can achieve superior performance across various tasks through more efficient and effective designs compared to human-designed models.


<br/>
# 예제
논문에서 제시된 ASI-ARCH 시스템의 예시를 구체적으로 설명하기 위해, 우리는 시스템이 어떻게 작동하는지에 대한 가상의 시나리오를 만들어 보겠습니다. 이 시나리오는 ASI-ARCH가 새로운 신경망 아키텍처를 발견하는 과정을 설명합니다.

### 한글 버전

#### 예시 시나리오: 새로운 신경망 아키텍처 발견

1. **트레이닝 데이터 준비**:
   - **입력 데이터**: 다양한 데이터셋이 사용됩니다. 예를 들어, 이미지 인식 데이터셋(CIFAR-10, ImageNet)이나 자연어 처리 데이터셋(GPT-3의 텍스트 데이터셋) 등이 포함될 수 있습니다.
   - **출력 데이터**: 각 데이터셋에 대한 모델의 성능 지표(정확도, 손실 등)가 출력으로 사용됩니다.

2. **테스트 데이터 준비**:
   - **입력 데이터**: 트레이닝 데이터와 유사한 형식의 데이터셋이 사용됩니다. 그러나 테스트 데이터는 모델의 일반화 능력을 평가하기 위해 트레이닝 데이터와는 다른 샘플로 구성됩니다.
   - **출력 데이터**: 모델이 테스트 데이터에서 달성한 성능 지표가 출력됩니다.

3. **구체적인 테스크**:
   - **아키텍처 제안**: ASI-ARCH의 연구자 모듈은 기존의 아키텍처와 실험 결과를 바탕으로 새로운 신경망 아키텍처를 제안합니다.
   - **코드 구현**: 제안된 아키텍처는 엔지니어 모듈에 의해 코드로 구현됩니다.
   - **모델 트레이닝 및 평가**: 구현된 모델은 실제 데이터셋을 사용하여 트레이닝되고, 성능이 평가됩니다.
   - **분석 및 피드백**: 분석가 모듈은 실험 결과를 분석하여 새로운 인사이트를 도출하고, 이를 다음 아키텍처 제안에 반영합니다.




#### Example Scenario: Discovering a New Neural Network Architecture

1. **Training Data Preparation**:
   - **Input Data**: Various datasets are used. For example, image recognition datasets (CIFAR-10, ImageNet) or natural language processing datasets (text datasets for GPT-3) may be included.
   - **Output Data**: Performance metrics of the model for each dataset (accuracy, loss, etc.) are used as outputs.

2. **Test Data Preparation**:
   - **Input Data**: Datasets similar in format to the training data are used. However, test data consists of different samples from the training data to evaluate the model's generalization ability.
   - **Output Data**: The performance metrics achieved by the model on the test data are output.

3. **Specific Tasks**:
   - **Architecture Proposal**: The Researcher module of ASI-ARCH proposes a new neural network architecture based on existing architectures and experimental results.
   - **Code Implementation**: The proposed architecture is implemented in code by the Engineer module.
   - **Model Training and Evaluation**: The implemented model is trained using real datasets, and its performance is evaluated.
   - **Analysis and Feedback**: The Analyst module analyzes the experimental results to derive new insights, which are incorporated into the next architecture proposal.

<br/>
# 요약



ASI-ARCH는 AI가 스스로 새로운 신경망 아키텍처를 제안하고 검증하는 자율적인 시스템으로, 연구자, 엔지니어, 분석가 모듈을 통해 진화적 개선 전략을 구현합니다. 이 시스템은 1,773개의 실험을 통해 106개의 혁신적인 선형 주의 메커니즘 아키텍처를 발견했으며, 이는 인간이 설계한 기준을 초과하는 성능을 보여줍니다. 이러한 발견은 AI가 인간의 인지적 한계를 넘어 과학적 발견을 컴퓨팅 자원으로 확장할 수 있음을 입증합니다.


ASI-ARCH is an autonomous system where AI independently proposes and validates new neural architectures, implementing an evolutionary improvement strategy through researcher, engineer, and analyst modules. The system discovered 106 innovative linear attention architectures through 1,773 experiments, demonstrating performance that surpasses human-designed baselines. These findings prove that AI can transcend human cognitive limits and scale scientific discovery with computational resources.

<br/>
# 기타



1. **Figure 1: Scaling Law for Scientific Discovery**
   - **결과**: 이 다이어그램은 발견된 최첨단(SOTA) 아키텍처의 누적 수를 소비된 총 컴퓨팅 시간에 대해 플롯합니다. 강한 선형 관계는 AI 시스템의 새로운 고성능 아키텍처 발견 능력이 할당된 계산 예산과 효과적으로 확장됨을 보여줍니다.
   - **인사이트**: 이는 연구 진행이 인간의 전문 지식이 아닌 계산 자원에 의해 확장될 수 있음을 시사합니다.

2. **Figure 2: A “Move 37” Moment in Design**
   - **결과**: AlphaGo의 전설적인 "Move 37"처럼, AI가 발견한 아키텍처는 우리의 가정을 도전하고 디자인 철학의 미지의 영역을 탐험하도록 영감을 줍니다.
   - **인사이트**: AI가 발견한 아키텍처는 인간이 설계한 기준을 체계적으로 능가하며, 아키텍처 혁신을 위한 이전에 알려지지 않은 경로를 밝힙니다.

3. **Figure 3: ASI-ARCH Exploration Trajectory Tree**
   - **결과**: 이 트리는 1,773개의 탐색된 아키텍처 간의 진화적 관계를 시각화하며, DeltaNet을 루트 노드로 합니다. 각 노드는 고유한 아키텍처를 나타내며 색상은 성능 점수를 나타냅니다.
   - **인사이트**: 이 트리는 AI가 어떻게 다양한 아키텍처를 탐색하고 발전시키는지를 보여주며, AI의 자율적 탐색 능력을 강조합니다.

4. **Figure 4: ASI-ARCH Framework Overview**
   - **결과**: 이 다이어그램은 닫힌 진화적 루프에서 작동하는 네 모듈 ASI-ARCH 프레임워크의 개요를 제공합니다. 사이클은 연구자가 역사적 데이터를 기반으로 새로운 아키텍처를 제안하는 것으로 시작됩니다.
   - **인사이트**: 이 프레임워크는 AI가 독립적으로 과학적 연구 과정을 수행할 수 있도록 하여, AI 주도의 연구를 민주화하는 데 기여합니다.

5. **Figure 5: Architectural Phylogenetic Tree**
   - **결과**: 이 트리는 새로운 아키텍처가 이전 아키텍처의 코드를 직접 수정하여 생성되는 부모-자식 관계를 정의합니다. 트리의 주변 색상은 다른 진화적 가지를 구별하는 데 사용됩니다.
   - **인사이트**: 이 트리는 AI가 아키텍처를 어떻게 발전시키고, 다양한 디자인 패턴을 탐색하는지를 보여줍니다.

6. **Table 1: Performance Comparison on Language Modeling and Zero-shot Common-sense Reasoning**
   - **결과**: 이 테이블은 언어 모델링 및 제로샷 상식 추론에서 AI가 발견한 아키텍처와 인간이 설계한 아키텍처의 성능을 비교합니다.
   - **인사이트**: AI가 발견한 아키텍처가 여러 작업에서 인간이 설계한 아키텍처와 경쟁할 수 있음을 보여줍니다.




1. **Figure 1: Scaling Law for Scientific Discovery**
   - **Result**: This diagram plots the cumulative count of discovered State-of-the-Art (SOTA) architectures against the total computing hours consumed. The strong linear relationship demonstrates that the AI system’s capacity for discovering novel, high-performing architectures scales effectively with the allocated computational budget.
   - **Insight**: This suggests that research progress can be scaled with computational resources rather than human expertise.

2. **Figure 2: A “Move 37” Moment in Design**
   - **Result**: Like AlphaGo’s legendary "Move 37," these AI-discovered architectures challenge our assumptions and inspire us to explore uncharted territories in design philosophy.
   - **Insight**: AI-discovered architectures systematically surpass human-designed baselines and illuminate previously unknown pathways for architectural innovation.

3. **Figure 3: ASI-ARCH Exploration Trajectory Tree**
   - **Result**: The tree visualizes the evolutionary relationships among 1,773 explored architectures, with DeltaNet as the root node. Each node represents a distinct architecture, and colors indicate performance scores.
   - **Insight**: This tree highlights how AI autonomously explores and evolves various architectures, emphasizing its autonomous exploration capabilities.

4. **Figure 4: ASI-ARCH Framework Overview**
   - **Result**: This diagram provides an overview of the four-module ASI-ARCH framework, which operates in a closed evolutionary loop. The cycle begins with the Researcher proposing a new architecture based on historical data.
   - **Insight**: This framework contributes to democratizing AI-driven research by enabling AI to independently conduct the scientific research process.

5. **Figure 5: Architectural Phylogenetic Tree**
   - **Result**: This tree defines a parent-child relationship where a new architecture is generated by directly modifying the code of a preceding one. The colors on the periphery are used to distinguish different evolutionary branches of the tree.
   - **Insight**: This tree shows how AI evolves architectures and explores various design patterns.

6. **Table 1: Performance Comparison on Language Modeling and Zero-shot Common-sense Reasoning**
   - **Result**: This table compares the performance of AI-discovered architectures with human-designed ones in language modeling and zero-shot common-sense reasoning.
   - **Insight**: It demonstrates that AI-discovered architectures can compete with human-designed ones across various tasks.

<br/>
# refer format:



**BibTeX:**
```bibtex
@article{Liu2025SII,
  title={SII-GAIR: AlphaGo Moment for Model Architecture Discovery},
  author={Yixiu Liu and Yang Nan and Weixian Xu and Xiangkun Hu and Lyumanshan Ye and Zhen Qin and Pengfei Liu},
  journal={arXiv preprint arXiv:2507.18074},
  year={2025}
}
```

**Chicago Style:**
Liu, Yixiu, Yang Nan, Weixian Xu, Xiangkun Hu, Lyumanshan Ye, Zhen Qin, and Pengfei Liu. 2025. "SII-GAIR: AlphaGo Moment for Model Architecture Discovery." *arXiv preprint arXiv:2507.18074*.
