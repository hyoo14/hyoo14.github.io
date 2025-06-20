---
layout: post
title:  "[2025]Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities"  
date:   2025-06-20 02:45:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

Mixture-of-Experts 기반 멀티모달 모델,  
포켓몬게임과 플래닝 등 다양한 테스크에서 좋은 성능  



짧은 요약(Abstract) :    


물론입니다. 아래는 Gemini 2.5 보고서의 핵심적인 **Abstract 요약**입니다.

---


Google의 Gemini 팀은 이 보고서에서 최신 AI 모델 시리즈인 **Gemini 2.5 Pro**와 **Gemini 2.5 Flash**를 소개합니다. Gemini 2.5 Pro는 코드 작성과 추론(task reasoning) 능력에서 최첨단 성능을 보여주며, 1백만 토큰 이상의 긴 문맥(long context)을 처리하고 텍스트, 이미지, 오디오, 비디오 등 다양한 모달리티를 동시에 다룰 수 있는 **다중모달 멀티태스킹** 모델입니다. 또한, 고급 추론을 위해 “생각하는(Thinking)” 능력을 학습하여 복잡한 에이전트적 문제 해결이 가능하도록 설계되었습니다. Flash 모델은 더 빠르고 저렴한 성능을 제공하면서도 복잡한 작업에 적절한 품질을 유지합니다. 전체 Gemini 2.X 시리즈는 **성능-비용 간 파레토 최적 경계**를 확장하여 다양한 응용 사례에 적합하게 구성되었습니다.

---


In this report, the Google Gemini team introduces the Gemini 2.X family of models, including **Gemini 2.5 Pro** and **Gemini 2.5 Flash**. Gemini 2.5 Pro is the most capable model to date, excelling in coding and reasoning benchmarks while supporting **multimodal input** (text, images, audio, video) and **long-context processing** of over 1 million tokens. The model incorporates a novel *Thinking* mechanism to perform advanced reasoning before generating outputs, enabling agentic behaviors and complex workflows. The **Gemini 2.5 Flash** model delivers strong reasoning performance at significantly lower latency and compute costs. Together, the Gemini 2.X models span the Pareto frontier of model capability vs. efficiency, offering scalable AI solutions across diverse tasks and domains.





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





### 1. 모델 아키텍처

Gemini 2.5 시리즈는 **Sparse Mixture-of-Experts (MoE)** 기반의 트랜스포머 모델입니다. 입력 토큰마다 전체 파라미터 중 일부 전문가(expert)만 활성화하여 계산량은 줄이면서도 모델 용량은 유지하는 구조입니다.

* 텍스트, 이미지, 오디오 등 **다중 모달 입력을 네이티브로 지원**하고,
* 최대 **100만 토큰 이상의 긴 문맥(long context)** 입력을 처리할 수 있습니다.

### 2. 트레이닝 데이터

학습에 사용된 데이터는 다음과 같은 **대규모 멀티모달 데이터셋**으로 구성됩니다.

* 공개 웹 문서, 다양한 프로그래밍 언어의 코드, 이미지, 음성, 동영상 등
* Gemini 2.0은 2024년 6월, Gemini 2.5는 **2025년 1월**까지의 정보를 학습에 반영했습니다.
* 중복 제거 및 품질 필터링을 강화하여 데이터 품질을 높였습니다.
* 사후 학습(post-training)에는 사용자 지침(instruction tuning) 데이터와 인간 피드백, 도구 사용 데이터를 포함합니다.

### 3. 트레이닝 인프라 및 전략

* **TPUv5p** 아키텍처 기반의 8960개 칩이 연결된 대규모 데이터 병렬 처리 환경에서 학습이 진행되었습니다.
* **탄력적 학습(elastic training)**: 일부 칩이 고장 나도 재배치 없이 즉시 학습을 계속할 수 있도록 설계되었습니다.
* **Silent Data Corruption(SDC) 탐지**: 오류 발생 시 체크섬 비교로 빠르게 원인을 찾아내고 문제 칩을 제외시킵니다.
* **강화학습(RL) 기반 ‘Thinking’ 학습**: 추론 시 계산 자원을 더 사용하여 보다 정교한 응답을 생성하도록 학습합니다.
* Post-training에는 **SFT, RM, RLHF, critic feedback** 등을 통해 **지시 따르기 능력과 안정성**을 동시에 개선하였습니다.

---


### 1. Model Architecture

The Gemini 2.5 models are built using **Sparse Mixture-of-Experts (MoE) Transformers**.
These models activate only a subset of parameters per token, significantly reducing computation cost while maintaining high capacity.

* They support **native multimodal input**, including text, images, audio, and video.
* Capable of processing **long-context inputs over 1 million tokens**.

### 2. Training Dataset

The pretraining data includes a large-scale, diverse collection of **multimodal datasets**:

* Public web documents, code from multiple programming languages, images, audio, and videos.
* Gemini 2.0 was trained with data until **June 2024**, and Gemini 2.5 up to **January 2025**.
* Improved **data filtering and deduplication** processes were applied.
* Post-training used **instruction-tuned multimodal data**, paired with human preference and tool-use data.

### 3. Training Infrastructure & Strategy

* Trained using **TPUv5p chips** across multiple datacenters in a **synchronous data-parallel** manner.
* Implemented **slice-granularity elasticity** to recover quickly from hardware failures without major downtime.
* **Split-phase Silent Data Corruption (SDC) detection** enables real-time detection and isolation of faulty devices.
* **Reinforcement Learning with Thinking**: Models are trained to reason using tens of thousands of forward passes at inference time.
* Post-training involved **SFT, Reward Modeling, and RLHF**, improving both **instruction following and safety** across tasks.




   
 
<br/>
# Results  





### 1. 평가 대상 테스크 및 메트릭

Gemini 2.5는 다양한 영역에 걸쳐 다음과 같은 벤치마크로 성능을 측정하였습니다.
주요 평가 지표는 **정확도(Accuracy), 통과율(Pass\@1), BLEU, WER, CIDEr** 등입니다.

| 영역      | 대표 테스크                                   | 메트릭                    |
| ------- | ---------------------------------------- | ---------------------- |
| 코드 생성   | LiveCodeBench, SWE-bench, Aider Polyglot | Pass\@1 (%)            |
| 수학·추론   | AIME, GPQA, HiddenMath-Hard              | Accuracy (%)           |
| 사실성     | SimpleQA, FACTS Grounding                | Accuracy (%)           |
| 멀티모달 이해 | MMMU, VideoMME, YouCook2                 | CIDEr, Accuracy        |
| 오디오 이해  | FLEURS, CoVoST2                          | WER↓, BLEU↑            |
| 긴 문맥 처리 | LOFT, MRCR-V2                            | Retrieval Accuracy (%) |

---

### 2. 경쟁 모델과의 비교 성능

Gemini 2.5 Pro는 다음과 같은 점에서 **경쟁 모델을 앞섰습니다**:

* **코딩 성능**: LiveCodeBench 기준 69.0%, SWE-bench verified 기준 67.2%로 GPT-4.1, Claude 4 등을 능가
* **추론 능력**: GPQA에서 86.4%, AIME에서 88.0% 달성
* **사실성**: FACTS Grounding 87.8%로 가장 높은 정답률
* **긴 문맥 처리**: 1M 토큰 처리 가능 모델 중 유일하며, LOFT와 MRCR-V2에서 최고 성능 기록
* **멀티모달 처리**: MMMU (82.0%) 및 VideoMME (84.3%) 등에서 GPT-4.1보다 높은 점수

---

### 3. Flash 모델 성능

Gemini 2.5 Flash는 성능 대비 속도와 비용이 매우 우수한 모델로,

* GPQA 82.8%,
* AIME 72.0%,
* MMMU 79.7%
  등의 높은 성능을 보였으며, Gemini 1.5 Pro보다 대부분의 테스크에서 앞섰습니다.

---


### 1. Evaluation Benchmarks and Metrics

Gemini 2.5 was evaluated across diverse tasks using standard academic and industry benchmarks.
Key metrics include **Accuracy, Pass\@1, BLEU, WER, CIDEr**, and others depending on task type.

| Domain                   | Task Benchmarks                          | Metrics                |
| ------------------------ | ---------------------------------------- | ---------------------- |
| Code Generation          | LiveCodeBench, SWE-bench, Aider Polyglot | Pass\@1 (%)            |
| Math & Reasoning         | AIME, GPQA, HiddenMath-Hard              | Accuracy (%)           |
| Factuality               | SimpleQA, FACTS Grounding                | Accuracy (%)           |
| Multimodal Understanding | MMMU, VideoMME, YouCook2                 | CIDEr, Accuracy        |
| Audio Understanding      | FLEURS, CoVoST2                          | WER↓, BLEU↑            |
| Long-context Retrieval   | LOFT, MRCR-V2                            | Retrieval Accuracy (%) |

---

### 2. Comparison to Competitors

Gemini 2.5 Pro demonstrates **state-of-the-art** performance in several areas:

* **Code**: 69.0% on LiveCodeBench, 67.2% on SWE-bench (multi-attempt), outperforming GPT-4.1 and Claude 4.
* **Reasoning**: Achieves 86.4% on GPQA and 88.0% on AIME.
* **Factuality**: Top score of 87.8% on FACTS Grounding benchmark.
* **Long-context**: The only model among peers supporting over 1M tokens, scoring highest on LOFT and MRCR-V2.
* **Multimodal**: 82.0% on MMMU and 84.3% on VideoMME, beating GPT-4.1 in video comprehension.

---

### 3. Gemini 2.5 Flash Performance

Gemini 2.5 Flash delivers **competitive accuracy with low latency**, including:

* GPQA: 82.8%
* AIME: 72.0%
* MMMU: 79.7%
  It surpasses Gemini 1.5 Pro in most tasks, making it a strong choice for cost-sensitive applications.




<br/>
# 예제  




### 1. 예시 1: Pokémon 게임 플레이 에이전트

Gemini 2.5 Pro는 \*\*Pokémon Blue 게임을 스스로 클리어한 에이전트(Gemini Plays Pokémon)\*\*로 활용되었습니다.

* 게임 화면을 텍스트로 전처리하여 인식하고,
* 100K\~150K 토큰 길이의 문맥을 기반으로 퍼즐 경로 탐색, 미로 문제 해결, 전략적 아이템 획득 등을 수행
* 813시간 걸려 1회차 클리어, 이후 고정 스캐폴드로 406시간 만에 2회차 클리어

 활용 능력: 장기 계획, 툴 사용, 비선형 경로 해결

---

### 2. 예시 2: 인터랙티브 도구 자동 생성

* **PDF 대본**을 입력하면, **학생용 대사 연습 도구**로 변환
* **책장 사진**을 입력하면, **책 추천 시스템** 구축
* **수학 개념을 표현한 포리에 급수 그래프**, **태양계 시뮬레이션**, **3D UI 생성** 등 고급 시각화까지 가능

 활용 능력: 멀티모달 이해, 구조적 이미지 처리, 창의적 생성

---

### 3. 예시 3: 오디오 대화 에이전트

* **감정, 속도, 스타일을 조절** 가능한 텍스트-음성 변환 (TTS)
* **다화자 대화, 팟캐스트 생성**, **사용자 음색 인식 및 응답 판단**
* 노트북LM에서 요약 내용을 오디오로 전환하는 기능에 활용됨

 활용 능력: 자연스러운 오디오 생성, 상황 인지 대화

---


### 1. Example 1: Gemini Plays Pokémon

Gemini 2.5 Pro was used as an autonomous agent to play and complete **Pokémon Blue**.

* The game screen was preprocessed into text inputs.
* The model handled **complex puzzles, item acquisition, and multi-step navigation** using 100K+ token context.
* The first run took 813 hours; the second optimized run completed in 406 hours.

 Strengths: Long-horizon planning, tool use, non-linear problem solving

---

### 2. Example 2: Interactive Application Generation

* Inputting a **PDF script** results in an interactive **line rehearsal tool** for students.
* A **bookshelf photo** is transformed into a **book recommendation app**.
* Advanced outputs like **solar system simulations**, **Fourier-series-based math visualizations**, and **3D UI generation** are also supported.

 Strengths: Multimodal reasoning, structural image-to-app generation, creative rendering

---

### 3. Example 3: Native Audio Dialog Agent

* Supports **style, emotion, and speed control** for text-to-speech (TTS).
* Enables **multi-speaker dialogue**, **podcast generation**, and **context-aware audio responses**.
* Used in applications like **NotebookLM Audio Overviews**.

 Strengths: Realistic audio generation, conversational timing, and tool integration

---



<br/>  
# 요약   




Gemini 2.5는 Sparse MoE 기반 멀티모달 트랜스포머로, 100만 토큰 이상의 긴 문맥과 텍스트·이미지·음성·비디오 입력을 네이티브로 처리하며 강화학습 기반 추론(Thinking) 기능을 탑재했습니다. 코드 생성, 수학 추론, 사실성, 멀티모달 이해 등 다양한 벤치마크에서 기존 모델을 능가하며 최고 성능을 기록했습니다. 포켓몬 게임 플레이, 대화형 애플리케이션 생성, 감정 기반 오디오 응답 등 실제 응용 사례에서도 장기 계획 수립과 도구 사용 능력을 입증했습니다.

---



Gemini 2.5 is a Sparse Mixture-of-Experts-based multimodal Transformer that natively supports text, image, audio, and video inputs, with long-context processing over 1 million tokens and reinforcement learning-based Thinking capabilities. It outperforms prior models across benchmarks in code generation, reasoning, factuality, and multimodal understanding. Real-world applications such as Pokémon gameplay, interactive tool generation, and emotion-aware audio response demonstrate its capabilities in long-horizon planning and agentic behavior.

---



<br/>  
# 기타   




### Figure 1 — 비용 대비 성능 그래프

* Gemini 2.5 Pro는 LMArena 기준 성능이 Gemini 1.5 Pro보다 120점 이상 향상됨.
* 성능과 비용(토큰당 가격)을 고려했을 때 Pareto frontier를 새롭게 정의함.

**인사이트**: 최상위 성능과 실용적 비용을 동시에 달성한 모델임을 시각적으로 보여줌.

---

### Figure 2 — 초당 출력 토큰 속도

* Gemini 2.5 Flash가 OpenAI o4-mini보다 빠른 출력 속도를 가짐.
* 고속 모델 중에서는 최고 수준의 생성 속도 달성.

**인사이트**: Flash 모델은 실시간 반응이 필요한 작업에 적합함.

---

### Table 3 — Gemini 계열 모델 성능 비교

* 코드: Gemini 2.5 Pro는 LiveCodeBench에서 69.0%로 이전 모델 대비 큰 향상
* 수학: AIME에서 88.0%, GPQA에서 86.4%
* 멀티모달: MMMU 82.0%, VideoMME 84.3%

**인사이트**: 거의 모든 벤치마크에서 1.5 시리즈 대비 20\~40% 이상 향상된 정확도 기록.

---

### Table 4 — 타 LLM과의 비교

* Gemini 2.5 Pro는 GPT-4, Claude 4, Grok-3 등과 비교해 Aider Polyglot, GPQA, FACTS Grounding 등에서 최고 점수 기록
* 유일하게 100만 토큰 이상의 컨텍스트를 네이티브로 지원

**인사이트**: 가장 광범위하고 강력한 범용 LLM 성능을 보여줌.

---

### Figure 5 — 포켓몬 게임 진행 타임라인

* Gemini 2.5 Pro가 포켓몬 게임 전체를 에이전트 형태로 813시간에 클리어
* 이후 개선된 구조로 406시간만에 완주

**인사이트**: 장기 목표 수행, 복잡한 문제 해결, 멀티스텝 계획 가능성 입증.

---

### Appendix 요약

* 8.1: 각 벤치마크의 정의 및 스코어 산출 방식 설명
* 8.2: Gemini Plays Pokémon의 도구 설계 및 행동 로그
* 8.4: 이미지-to-SVG 변환 결과 비교 (1.5 Pro vs. 2.5 Pro)
* 8.5: 3시간짜리 영상에서 1초 이벤트를 회상하는 능력 평가

**인사이트**: Gemini 2.5는 단순 정보 요약을 넘어, 복잡한 시공간적 정보 추출 및 생성에서 우수한 성능을 보임.

---


### Figure 1 — Cost vs. Performance

* Gemini 2.5 Pro exceeds Gemini 1.5 Pro by over 120 LMArena ELO points.
* Defines a new Pareto frontier by combining strong performance with lower token costs.

Insight: Establishes Gemini 2.5 Pro as a leader in performance-efficiency trade-offs.

---

### Figure 2 — Output Tokens per Second

* Gemini 2.5 Flash generates tokens faster than OpenAI's o4-mini.
* Among the fastest models in terms of throughput.

Insight: Flash is suitable for applications requiring real-time responsiveness.

---

### Table 3 — Gemini Model Family Comparison

* Code: 69.0% on LiveCodeBench, a major improvement over prior models
* Math: 88.0% on AIME, 86.4% on GPQA
* Multimodal: 82.0% on MMMU, 84.3% on VideoMME

Insight: Substantial performance gains of 20–40% across nearly all core tasks compared to Gemini 1.5.

---

### Table 4 — Comparison with Other LLMs

* Gemini 2.5 Pro outperforms GPT-4, Claude 4, and Grok-3 on several key benchmarks
* It is the only model supporting native 1M+ token context

Insight: Sets a new standard in general-purpose LLM performance.

---

### Figure 5 — Pokémon Agent Timeline

* Gemini 2.5 Pro autonomously completed Pokémon Blue in 813 hours, then in 406 hours in an optimized setup
* Demonstrated long-term task management and puzzle-solving

Insight: Validates the model’s capacity for long-horizon reasoning and agentic planning.

---

### Appendix Highlights

* Section 8.1: Definitions and scoring methods for benchmarks
* Section 8.2: Pokémon agent tool design and decision examples
* Section 8.4: Image-to-SVG conversion comparison between 1.5 and 2.5
* Section 8.5: Ability to recall 1-second events from a 3-hour video

Insight: Gemini 2.5 excels at complex multimodal, temporal, and spatial reasoning beyond surface-level understanding.




<br/>
# refer format:     



@techreport{gemini2025,
  title        = {Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities},
  author       = {{Gemini Team, Google}},
  year         = {2025},
  institution  = {Google DeepMind},
  note         = {Technical Report},
  url          = {https://deepmind.google/technologies/gemini}
}
  


Gemini Team, Google. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. Technical Report. Google DeepMind, 2025. https://deepmind.google/technologies/gemini.  




