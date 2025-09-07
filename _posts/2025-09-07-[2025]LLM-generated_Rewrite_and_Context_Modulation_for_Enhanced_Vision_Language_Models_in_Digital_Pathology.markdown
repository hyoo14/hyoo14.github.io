---
layout: post
title:  "[2025]LLM-generated Rewrite and Context Modulation for Enhanced Vision Language Models in Digital Pathology"
date:   2025-09-07 21:35:37 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 


대형 언어모델(LLM)로 기존 병리 캡션을 의미 보존 재서술(리라이트)하여 텍스트 다양성을 크게 늘리고, 이를 이용해 병리 특화 VLM을 학습합니다. 더 나아가, 이미지 임베딩을 텍스트 맥락(장기, 질병, 세포 등)에 맞게 조정하는 ‘컨텍스트 모듈레이션’ 임베딩 레이어를 제안하고, 컨텍스트별 리라이트 캡션으로 이 레이어를 학습


메서드: Mistral‑7B‑Instruct‑v0.2로 Quilt1M 캡션을 5,088,520개 일반 재작성과 3,053,112개 문맥별(장기·질병·세포) 재작성으로 확장하고, 대비학습 기반 VLM을 재학습했으며, 이미지 임베딩에 문맥 임베딩 h(c)를 곱하는 콘텍스트 모듈레이션 층을 추가하고 15개 데이터셋에 대해 각 2K(총 170K) 프롬프트 세트를 생성해 제로샷 평가의 프롬프트 민감도를 체계적으로 측정했습니다.


짧은 요약(Abstract) :


- 배경: 병리 영상 분야의 비전-언어 모델(VLM)은 대규모 이미지-캡션 쌍이 필요한데, 의료 도메인에서는 데이터가 적고, 슬라이드 캡션이 이미지의 일부 특징만을 지칭하는 경우가 많아 학습이 어렵습니다. 또한 제로샷 분류는 프롬프트 문구에 민감합니다.
- 핵심 아이디어: 대형 언어모델(LLM)로 기존 병리 캡션을 의미 보존 재서술(리라이트)하여 텍스트 다양성을 크게 늘리고, 이를 이용해 병리 특화 VLM을 학습합니다. 더 나아가, 이미지 임베딩을 텍스트 맥락(장기, 질병, 세포 등)에 맞게 조정하는 ‘컨텍스트 모듈레이션’ 임베딩 레이어를 제안하고, 컨텍스트별 리라이트 캡션으로 이 레이어를 학습합니다.
- 평가와 기여: 
  - 제로샷 분류, 텍스트-투-이미지/이미지-투-텍스트 검색에서 성능 향상을 입증.
  - LLM을 활용해 다수의 프롬프트 세트를 생성·평가함으로써 제로샷 결과의 프롬프트 민감성을 체계적으로 분석하는 확장 가능한 절차를 제안.
  - 800만 개 캡션, 17만 개 제로샷 프롬프트, 학습 코드와 가중치를 공개.
- 결론: LLM 기반 캡션 리라이트와 컨텍스트 모듈레이션을 통해 디지털 병리 VLM의 강건성과 성능을 크게 향상시킬 수 있음을 보였습니다.

Summary (English)
- Motivation: Vision-language models in digital pathology suffer from limited image–caption data, and slide-level captions often describe only small parts of the image. Zero-shot results are also highly prompt-sensitive.
- Approach: Use an LLM to produce semantically consistent language rewrites of pathology captions to greatly expand textual diversity and train a pathology-specific VLM. Introduce a context modulation layer that adjusts image embeddings to match caption context (e.g., organ, disease, cell), trained using context-specific rewrites.
- Evaluation and Contributions:
  - Demonstrate improvements on zero-shot classification and bidirectional retrieval tasks.
  - Provide a scalable LLM-driven procedure to quantify prompt sensitivity by generating large prompt sets.
  - Release a large resource: 8M captions, 170K prompts, training code, and model weights.
- Conclusion: LLM-generated rewrites and context-aware modulation substantially enhance VLM robustness and performance in digital pathology.


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



1) 개요: 핵심 아이디어
- 문제의식: 디지털 병리학에서는 이미지–자연어 페어가 대규모로 부족하고, 한 슬라이드의 캡션이 이미지 전체가 아니라 일부분 특징만을 말하는 경우가 많습니다. 또한 제로샷 분류는 프롬프트 문구에 민감합니다.
- 접근 요약:
  - 대규모 LLM(예: Mistral-7B-Instruct-v0.2)로 기존 병리 캡션의 “언어 재작성(rewrite)”을 대량 생성하여 텍스트 다양성을 확장.
  - 장기적 맥락(organ, disease, cell)에 맞춘 “컨텍스트 특화 캡션”을 별도로 생성.
  - 컨텍스트에 따라 이미지 임베딩을 조정하는 “컨텍스트 모듈레이션” 레이어를 도입.
  - 다양한 프롬프트에 대한 민감도를 정량화하기 위해 각 테스트 데이터셋마다 2,000개의 프롬프트 세트를 생성해 제로샷 성능을 평가.
  - 결과적으로, SynthPath(언어 재작성으로 파인튜닝된 VLM)와 SynthPath Context(컨텍스트 모듈레이션 포함) 두 모델을 제안.

2) 학습 데이터 구축 파이프라인
- 기반 데이터셋: Quilt1M
  - YouTube 교육 비디오에서 병리 이미지를 추출하고, 음성-텍스트 변환과 LLM 기반 정제를 거쳐 이미지–캡션 쌍을 구축한 공개 병리 데이터셋입니다.
- 일반 목적 언어 재작성(General rewrites)
  - 사용 LLM: Mistral-7B-Instruct-v0.2
  - 목표: 의미 보존을 전제로 다양한 표현으로 캡션을 재작성하여 텍스트 변이를 크게 늘림.
  - 규모 및 리소스: 총 5,088,520개의 재작성 캡션 생성, A6000 GPU에서 평균 추론 4.8초/문장, 총 약 6,800 GPU-시간.
- 컨텍스트 특화 캡션(Context-specific rewrites)
  - 세 가지 컨텍스트로 분리: 장기(organ), 질병(disease), 세포(cell).
  - LLM 프롬프트 설계:
    - Organ: 비교적 쉬워 예시 3개로 안정적 추출.
    - Disease: 두 단계로 유도(질병 여부 분류 → 근거 설명), 예시 5개가 최적.
    - Cell: 가장 어려워 예시 9개 사용, 관련 정보가 없으면 ‘None’을 출력하도록 설계(환각 방지).
  - 규모: 총 3,053,112개의 컨텍스트 특화 재작성 캡션.
- 전체 합산
  - 총 8,141,632개의 합성 캡션(일반 + 컨텍스트)이 생성되었고, 약 10,800 GPU-시간 사용.
  - 이는 병리학 영역 최대 규모의 공개 캡션 리소스로 제공됨.
- 제로샷 평가용 프롬프트 세트 생성
  - 각 테스트 데이터셋마다 2,000개 프롬프트 세트(클래스별 다양한 문장)를 LLM으로 생성.
  - 총 15개 데이터셋 대상, 프롬프트 총합 170,000개.
  - 목적: 프롬프트 문구 민감도를 체계적으로 평가하고, 모델 비교의 공정성을 높임.

3) 모델/아키텍처와 학습 절차
- 기본 VLM 구조
  - CLIP 계열의 이미지 인코더 f(·)와 텍스트 인코더 g(·)를 사용해, 매칭된 이미지–텍스트의 코사인 유사도를 높이고 비매칭 쌍을 낮추는 대조학습(InfoNCE류)을 수행.
  - 단일 이미지에 대해 K개의 캡션(원본+재작성)이 존재할 수 있으며, 학습 시 무작위로 하나를 샘플링하여 사용.
- SynthPath: 언어 재작성으로 파인튜닝된 VLM
  - 베이스: Quilt1M B-32 모델.
  - 파라미터 효율적 파인튜닝(PEFT)–LoRA 적용:
    - Rank=8, Alpha=8, Dropout=0.05, Linear/Conv1D 레이어에 적용.
  - 최적화 설정:
    - Optimizer: Adam, 초기 학습률 1e-5, weight decay 0.2.
    - Cosine 스케줄러(10 epoch 스케줄 참조), 총 30 epoch 학습(검증 손실 수렴까지).
  - 데이터/하드웨어:
    - 653,208개 유니크 이미지와 5,088,520개 합성 캡션 매칭.
    - Train:Val=0.85:0.15, 배치 4096, A6000 GPU 8장.
- SynthPath Context: 컨텍스트 모듈레이션 포함 모델
  - 아이디어: 캡션이 지시하는 컨텍스트(organ/disease/cell)에 따라 이미지 임베딩을 조정하면 텍스트와의 정합이 좋아짐.
  - 아키텍처: 컨텍스트 임베딩 레이어 h(c) 도입
    - 입력은 컨텍스트 ID c(1~3), 출력은 이미지/텍스트 임베딩 차원(512)에 맞는 벡터.
    - 이미지 임베딩 f(I)에 h(c)를 요소곱(해더마드 곱)으로 곱해 컨텍스트별로 변조.
  - 학습:
    - 이미지/텍스트 인코더는 SynthPath에서 고정(frozen).
    - 오직 h(c)만 학습(입력 크기 1, 출력 크기 512의 단일 임베딩 레이어).
    - 손실: 각 컨텍스트별(organ/disease/cell) 컨텍스트 특화 캡션과의 대조학습 손실을 합산.
    - Optimizer: AdamW, 초기 학습률 0.001, 배치 8192, 검증 수렴까지 학습.
    - 데이터: 3,053,112개 컨텍스트 특화 캡션(세 컨텍스트 균등 분배).

4) 추론(특히 제로샷) 절차
- 제로샷 분류:
  - 이미지 임베딩과 프롬프트 문구의 텍스트 임베딩 간 코사인 유사도를 비교해 클래스를 예측.
  - 프롬프트 민감도 문제를 보완하기 위해 데이터셋별로 2,000개 프롬프트 세트를 사용해 평균 및 분산을 보고.
- 컨텍스트 선택(SynthPath Context 사용 시):
  - 데이터셋의 과제 특성에 따라 organ/disease/cell 임베딩 중 하나를 선택해 적용.
  - 예: 장기 분류(OCeLOT Organ, PanNuke Organ)에는 organ 컨텍스트, 양성/악성 판별(BACH, LC Lung/Colon, Camelyon 등)에는 disease 컨텍스트, 세포 유형 중심 분류(PanNuke Cell, OCeLOT Cell, Skin 등)에는 cell 컨텍스트를 적용.
  - 어떤 데이터셋은 해석이 애매할 수 있어(예: Mhist) 선택이 성능에 영향을 줄 수 있음.

5) 본 연구의 차별점/특별 기법
- 대규모 LLM 기반 “의미 보존형” 언어 재작성으로 병리 텍스트 다양성을 비약적으로 확장(5M+ 캡션).
- 캡션의 숨은 맥락(organ/disease/cell)을 추출해 컨텍스트 특화 캡션을 생성하고(3M+), 그에 맞춘 “컨텍스트 임베딩 레이어”로 이미지 임베딩을 상황별로 조정.
- 모델 비교의 공정성과 신뢰도를 높이기 위해, 15개 데이터셋 × 2,000 프롬프트 세트(총 170K)로 제로샷 성능의 프롬프트 민감도를 정량화.
- 파라미터 효율적 파인튜닝(LoRA)로 대규모 재학습 비용을 절감하면서, 도메인 특화 능력을 증대.
- 모든 합성 캡션, 프롬프트 세트, 코드, 가중치를 공개(github.com/cagladbahadir/synthpath).

6) 실무적 고려와 한계
- 합성 캡션 생성은 GPU 비용이 큼(약 10,800 GPU-시간). 에너지–노력 간 트레이드오프가 존재.
- 컨텍스트 선택이 애매한 과제(복합 맥락 분류)에서는 단일 컨텍스트 모듈레이션이 최선이 아닐 수 있음. 향후 단계적/계층적 추론(예: 질병 여부 → 조직/세포 분류)이 유망.




1) Overview: Core Idea
- Motivation: Digital pathology lacks large-scale image–caption pairs; slide-level captions often describe only localized features. Zero-shot classification is sensitive to prompt wording.
- Approach summary:
  - Use a large language model (Mistral-7B-Instruct-v0.2) to generate massive “language rewrites” of existing pathology captions, expanding textual diversity at scale.
  - Create “context-specific” captions targeting organ, disease, and cell contexts.
  - Introduce a simple “context modulation” layer that adjusts image embeddings depending on the caption’s context, improving alignment with text.
  - Quantify prompt sensitivity by generating 2,000 prompt sets per test dataset (15 datasets; 170K prompts total).
  - Deliver two models: SynthPath (trained with language rewrites) and SynthPath Context (adds context modulation).

2) Data Construction Pipeline
- Base dataset: Quilt1M
  - Built from educational YouTube videos via speech-to-text, LLM refinement, and filtering; provides pathology image–caption pairs.
- General-purpose language rewrites
  - LLM: Mistral-7B-Instruct-v0.2.
  - Goal: paraphrase captions while preserving semantics to greatly increase textual variability.
  - Scale and resources: 5,088,520 rewrites; average 4.8 seconds per caption on an A6000; roughly 6,800 GPU-hours.
- Context-specific caption generation
  - Three contexts: organ, disease, cell.
  - Prompting strategy:
    - Organ: 3 examples suffice (easiest).
    - Disease: two-stage reasoning (diagnosis then explanation); 5 examples optimal.
    - Cell: hardest; 9 examples; force ‘None’ when no evidence exists (to control hallucination).
  - Scale: 3,053,112 context-specific captions.
- Total synthetic captions
  - 8,141,632 captions (general + context), about 10,800 GPU-hours; largest open pathology caption resource to date.
- Prompt sets for zero-shot evaluation
  - For each dataset, 2,000 diverse prompt sets were generated with an LLM, totaling 170K prompts across 15 datasets.
  - Purpose: ensure robust and fair comparison by accounting for prompt variability.

3) Model/Architecture and Training
- Base VLM
  - CLIP-style dual encoders: image encoder f(·), text encoder g(·).
  - Contrastive learning increases similarity for matched image–text pairs and decreases it for mismatches (InfoNCE-like).
  - For images with multiple captions (original + rewrites), one caption is randomly sampled per step.
- SynthPath: VLM fine-tuned with rewrites
  - Base: Quilt1M B-32.
  - Parameter-efficient fine-tuning with LoRA:
    - Rank=8, Alpha=8, Dropout=0.05 on Linear/Conv1D layers.
  - Optimization:
    - Adam, lr 1e-5, weight decay 0.2, cosine scheduler (10-epoch schedule reference), trained for 30 epochs until validation loss convergence.
  - Data/hardware:
    - 653,208 unique images paired with 5,088,520 synthetic captions.
    - Train:Val=0.85:0.15, batch size 4096, 8× A6000 GPUs.
- SynthPath Context: with context modulation
  - Idea: since captions encode different contexts (organ/disease/cell), modulate image embeddings accordingly to match context-specific texts.
  - Architecture: a small context embedding layer h(c)
    - Input: context ID (1..3); Output: a 512-dim vector matching embedding size.
    - Image embedding f(I) is element-wise multiplied by h(c) before similarity computation.
  - Training:
    - Freeze SynthPath’s image/text encoders.
    - Train only h(c) using context-specific captions; loss sums contrastive terms over all contexts.
    - AdamW, lr 0.001, batch size 8192, train until validation convergence.
    - Data: 3,053,112 context-specific rewrites (balanced across contexts).

4) Inference (Zero-shot) Procedure
- Zero-shot classification:
  - Compute cosine similarity between image embeddings and text embeddings from prompts; pick the best class.
  - Use 2,000 prompt sets per dataset to report average and variance, mitigating prompt-sensitivity bias.
- Context selection (for SynthPath Context):
  - Choose organ/disease/cell based on the dataset/task.
  - Examples: organ for PanNuke/OCeLOT Organ; disease for benign vs malignant/grading datasets (BACH, LC Lung/Colon, Camelyon, etc.); cell for PanNuke Cell, OCeLOT Cell, Skin, Renal Cell, CRC, and SICAP.
  - Some datasets are ambiguous (e.g., Mhist), and context choice may affect performance.

5) Key Contributions/Techniques
- Massive LLM-driven, semantics-preserving caption rewrites (5M+) tailored to computational pathology.
- Context-specific caption generation (3M+) and a lightweight context modulation layer that element-wise reweights image embeddings per context.
- Rigorous prompt-sensitivity characterization via 170K prompts across 15 datasets; ensures fair model comparison.
- Parameter-efficient fine-tuning (LoRA) for domain adaptation without full retraining.
- Public release of captions, prompt sets, code, and model weights (github.com/cagladbahadir/synthpath).

6) Practical Considerations and Limitations
- Synthetic caption generation is compute-intensive (~10,800 GPU-hours), implying an energy–effort tradeoff.
- Single-step context selection can be suboptimal for tasks spanning multiple contexts; future hierarchical inference (e.g., disease first, then tissue/cell) is promising.


<br/>
# Results



1) 평가 설정 개요
- 비교 대상(모델)
  - Clip: 자연 이미지 기반 일반 VLM 기준선.
  - Quilt1M: 디지털 병리 특화 VLM(YouTube 교육 영상에서 100만 쌍 병리 이미지-텍스트로 재학습).
  - SynthPath(제안): Quilt1M B-32를 기반으로, Mistral-7B-Instruct-v0.2로 생성한 일반 목적 언어 리라이트(총 5,088,520개)로 추가 학습.
  - SynthPath Context(제안): SynthPath의 인코더는 고정하고, 이미지 임베딩에 컨텍스트 임베딩 h(c)을 곱하는 얇은 층(512차)을 학습. 학습에는 컨텍스트별(장기/질병/세포) 리라이트 3,053,112개 사용. 추론 시 데이터셋별로 적절한 컨텍스트 선택.

- 테스트 데이터(총 15개 제로샷 분류 과제)
  - 이진/다중 클래스 병리 패치 분류: Camelyon, Databiox, Sicap, Crc, Skin, Skin Tumor, Mhist, BACH, LC Colon, LC Lung, Renal Cell
  - 장기/세포 분류: Pannuke(Organ/Cell), Ocelot(Organ/Cell)
  - 각 데이터셋의 클래스 구성과 샘플 수는 본문 3.6.3에 상세 기재.

- 프롬프트와 프롬프트 민감도
  - 각 데이터셋마다 2,000개 프롬프트 세트(클래스별 문장 묶음)를 LLM으로 생성.
  - 총 15개 데이터셋에 대해 170K개의 프롬프트(문장) 생성.
  - 제로샷 정확도는 프롬프트 세트별로 계산하고, 평균±표준편차를 보고.

- 메트릭과 비교 방식
  - 제로샷 분류: 이미지/텍스트 임베딩 코사인 유사도 기반 정확도. 2,000 프롬프트 세트 전반의 평균±표준편차.
  - 통계 검정(유의성): 동일 프롬프트 세트에서 모델 간 정확도를 짝지어 순열 기반 검정(비모수)으로 p-값 산출(Table 1).
  - 검색 성능: ARCH 데이터셋(PubMed n=3309, Textbooks n=4305)에서 텍스트→이미지, 이미지→텍스트 Top-k(1/5/20/50/100) 매칭 비율(Fig. 6).
  - Few-shot 선형 분류: 클래스당 1샷, 8개 랜덤 분류기 평균(표준편차) 성능(Table 2).
  - 프롬프트 앙상블: 2,000개 프롬프트의 소프트맥스 점수 평균(naive ensembling) 결과(Fig. 7).

- 컨텍스트 선택(제로샷 추론 시 SynthPath Context의 컨텍스트 h(c))
  - Organ: Ocelot Organ, Pannuke Organ
  - Disease: Bach, LC Lung, LC Colon, Camelyon, Databiox, Mhist, Skin Tumor
  - Cell: Pannuke Cell, Ocelot Cell, Skin, Renal Cell, Crc, Sicap
  - 주의: 일부 데이터셋은 컨텍스트 선택이 비자명(Mhist, Skin Tumor 등).

2) 제로샷 분류 결과(2,000 프롬프트 세트, Fig. 5)
- Quilt1M vs Clip
  - 15개 중 14개에서 Quilt1M가 Clip보다 우수, 평균 정확도 +0.07p 향상.

- SynthPath vs Quilt1M
  - 15개 중 11개에서 SynthPath가 우수, 1개 동일(Databiox), 3개 감소(Mhist, Sicap, Ocelot Cell).
  - 평균 +0.04p 향상. 이는 Quilt1M(100만 쌍) 대비 500만개의 LLM 리라이트 추가 만으로 이전 향상 폭의 약 57%를 추가로 확보했음을 의미.
  - 대표적 향상 폭: Crc +0.13, LC Lung +0.11, Ocelot Organ +0.09, LC Colon +0.08, Pannuke Cell +0.07.

- SynthPath Context vs SynthPath
  - 15개 중 11개에서 SynthPath Context가 최고 성능.
  - 평균 +0.015p 추가 향상. 비자명 컨텍스트(예: Crc, Skin, Renal Cell)에서도 이득 관찰.
  - 다만 컨텍스트 선택이 까다로운 일부(Mhist, Skin Tumor)에서는 평균 하락.

3) 제로샷 통계적 유의성(순열 검정, Table 1)
- SynthPath Context > Quilt1M: 15/15에서 우월, p<.001.
- SynthPath > Quilt1M: 15/15에서 우월, p<.001.
- SynthPath Context > SynthPath: 13/15에서 우월(대부분 p<.001). 예외로 Databiox는 유의하지 않음(p=.312), Skin Tumor는 유의하나 p=0.008로 상대적으로 완만.
- 모든 모델 대비 Clip: 제안 모델들이 전부 p<.001로 우월.

4) 텍스트-이미지/이미지-텍스트 검색(ARCH, Fig. 6)
- 데이터: PubMed(n=3309), Textbooks(n=4305).
- 결과: SynthPath가 20개 조건(2 데이터 × 2 태스크 × 5개 Top-k) 중 19개에서 최고. 유일한 예외는 Textbooks의 이미지→텍스트 Top-5 매칭율.
- 해석: 실제(실물) 이미지-텍스트 쌍에서도 LLM 리라이트로 학습한 SynthPath가 가장 견고함. 다중 컨텍스트가 섞인 캡션 환경에서는 컨텍스트 변조를 쓰지 않은 SynthPath가 SynthPath Context보다 적합하다는 사전 가정을 뒷받침.

5) Few-shot(1-shot) 선형 분류(Table 2)
- 총평: 제안 모델(SynthPath, SynthPath Context)이 15개 중 9개에서 기준선(Clip, Quilt1M)과 비기거나 우세.
- 예시
  - LC Lung: SynthPath Context 0.71로 최고(Clip 0.70, SynthPath 0.69, Quilt1M 0.68).
  - LC Colon: Quilt1M 0.97이 최고(SynthPath Context 0.94, SynthPath 0.90, Clip 0.74).
  - Camelyon: SynthPath Context 0.63로 최고(Quilt1M 0.60, SynthPath 0.59, Clip 0.65는 제로샷과 달리 few-shot에서 낮음).
- 결론: 프롬프트 공학을 배제하는 본 연구의 1차 목표와 별개로, 적은 표본의 선형 분류에서도 경쟁력 유지.

6) 프롬프트 앙상블(2,000개 평균, Fig. 7)
- 결과: 제안 모델들이 15개 중 10개에서 기준선과 비기거나 우세.
- 시사점: 프롬프트 앙상블을 사용해도 제안 모델의 상대적 우수성이 대체로 유지.

7) 베이스 모델 B-16 절편 실험(Table 3, 부록 D.2)
- SynthPath B-16(Quilt1M B-16 기반)에 500만 리라이트를 적용해 학습 시, 15개 중 14개에서 Quilt1M B-16을 유의하게 능가(p<.001 다수).
- B-32 실험과 동일한 경향 확인(일관성 검증).

8) 요약적 해석과 함의
- 대규모 LLM 리라이트(일반+컨텍스트)가 디지털 병리 VLM의 제로샷 강건성을 전체적으로 끌어올림.
- 프롬프트 다양성에 따른 성능 변동을 체계적으로 계량(2,000 프롬프트 세트/데이터셋)하고, 순열 검정으로 통계적 차이를 명확화.
- 컨텍스트 변조는 다수 과제에서 추가 이득을 제공하나, 컨텍스트가 혼재된 검색 태스크(ARCH)나 컨텍스트 선택이 애매한 일부 분류 과제에서는 기본 SynthPath가 더 적합할 수 있음.
- 공개 산출물: 800만 캡션, 17만 프롬프트, 코드/가중치 공개(synthpath), 재현성과 확장성 제공.

참고(그림/표): Fig. 5(제로샷), Table 1(유의성), Fig. 6(검색), Table 2(few-shot), Fig. 7(프롬프트 앙상블), Table 3(B-16 절편)

----------------------------------------






1) Evaluation setup
- Models compared
  - Clip: general-purpose VLM trained on natural images.
  - Quilt1M: pathology-specific VLM trained on ~1M histopathology image-text pairs from educational YouTube videos.
  - SynthPath (proposed): initialized from Quilt1M B-32 and further trained with 5,088,520 general-purpose language rewrites generated by Mistral-7B-Instruct-v0.2.
  - SynthPath Context (proposed): keeps SynthPath encoders frozen and trains a single 512-dim context embedding layer h(c) that multiplicatively modulates image embeddings; trained with 3,053,112 context-specific rewrites (organ/disease/cell). At inference, a dataset-appropriate context is selected.

- Datasets (15 zero-shot classification tasks)
  - Binary/multi-class patch classification: Camelyon, Databiox, Sicap, Crc, Skin, Skin Tumor, Mhist, BACH, LC Colon, LC Lung, Renal Cell.
  - Organ/cell classification: Pannuke (Organ/Cell), Ocelot (Organ/Cell).
  - Details of classes and sample sizes in Sec. 3.6.3.

- Prompts and prompt sensitivity
  - For each dataset, 2,000 prompt sets (a set is one sentence per class) generated by an LLM.
  - Total 170K prompts across the 15 datasets.
  - For zero-shot accuracy, report mean±std across the 2,000 prompt sets.

- Metrics and comparisons
  - Zero-shot classification: cosine similarity between image/text embeddings; accuracy averaged over 2,000 prompt sets with mean±std.
  - Statistical significance: permutation test with paired accuracies over the same prompt sets; report p-values (Table 1).
  - Retrieval: text-to-image and image-to-text on ARCH (PubMed n=3309, Textbooks n=4305), Top-k (1/5/20/50/100) match rates (Fig. 6).
  - Few-shot linear probing: 1-shot per class; average (std) over 8 random linear classifiers (Table 2).
  - Prompt ensembling: naive averaging of softmax scores over 2,000 prompts (Fig. 7).

- Context choice for SynthPath Context (inference)
  - Organ: Ocelot Organ, Pannuke Organ
  - Disease: Bach, LC Lung, LC Colon, Camelyon, Databiox, Mhist, Skin Tumor
  - Cell: Pannuke Cell, Ocelot Cell, Skin, Renal Cell, Crc, Sicap
  - Note: for some datasets, the optimal context is non-trivial (e.g., Mhist, Skin Tumor).

2) Zero-shot classification (2,000 prompt sets, Fig. 5)
- Quilt1M vs Clip
  - Quilt1M outperforms Clip in 14/15 datasets, average +0.07 accuracy.

- SynthPath vs Quilt1M
  - SynthPath better in 11/15; equal in 1 (Databiox); lower in 3 (Mhist, Sicap, Ocelot Cell).
  - Average +0.04 accuracy gain, i.e., an additional 57% improvement on top of Quilt1M’s gains, achieved purely via 5M language rewrites.
  - Notable per-dataset gains: Crc +0.13, LC Lung +0.11, Ocelot Organ +0.09, LC Colon +0.08, Pannuke Cell +0.07.

- SynthPath Context vs SynthPath
  - Best model in 11/15 datasets.
  - Additional +0.015 average gain; benefits appear even where context choice is non-trivial (e.g., Crc, Skin, Renal Cell).
  - Performance drops observed where context selection is challenging (e.g., Mhist, Skin Tumor).

3) Statistical significance (permutation tests, Table 1)
- SynthPath Context > Quilt1M: all 15 datasets, p<.001.
- SynthPath > Quilt1M: all 15 datasets, p<.001.
- SynthPath Context > SynthPath: 13/15 datasets (mostly p<.001); Databiox is not significant (p=.312); Skin Tumor significant with p=0.008.
- Proposed models vs Clip: all p<.001.

4) Retrieval on ARCH (Fig. 6)
- Data: PubMed(n=3309) and Textbooks(n=4305).
- Result: SynthPath is best in 19/20 settings (2 datasets × 2 tasks × 5 Top-k); the only exception is image-to-text Top-5 on Textbooks.
- Interpretation: SynthPath, trained with LLM rewrites, is most robust on real image-text retrieval; as anticipated, context modulation is less helpful when multiple contexts are mixed in captions.

5) Few-shot (1-shot) linear probing (Table 2)
- Overall: proposed models match or outperform baselines in 9/15 datasets.
- Examples
  - LC Lung: SynthPath Context 0.71 (best); Clip 0.70; SynthPath 0.69; Quilt1M 0.68.
  - LC Colon: Quilt1M 0.97 (best); SynthPath Context 0.94; SynthPath 0.90; Clip 0.74.
  - Camelyon: SynthPath Context 0.63 (best); Quilt1M 0.60; SynthPath 0.59; Clip 0.65 is lower in this few-shot setting.
- Takeaway: beyond minimizing prompt engineering, proposed models remain competitive under few-shot linear probes.

6) Prompt ensembling (2,000 prompts, Fig. 7)
- Result: proposed models outperform or tie baselines in 10/15 datasets under naive ensembling.
- Implication: advantages largely persist even when ensembling many prompts.

7) Base model ablation with B-16 (Table 3; Appendix D.2)
- SynthPath B-16 (Quilt1M B-16 + 5M rewrites) beats Quilt1M B-16 in 14/15 datasets (mostly p<.001), mirroring the B-32 trend.
- Confirms consistency of gains across backbones.

8) Bottom line
- Large-scale LLM rewrites (general + context-aware) substantially bolster pathology VLM robustness to prompt phrasing.
- Using 2,000 prompt sets per dataset plus permutation tests provides a principled measure of prompt sensitivity and model differences.
- Context modulation delivers further gains on many tasks, while plain SynthPath is preferable where captions mix multiple contexts (retrieval) or context choice is ambiguous for classification.
- All resources (8M captions, 170K prompts, code, weights) are publicly released, facilitating reproducibility and extension.

References to figures/tables: Fig. 5 (zero-shot), Table 1 (significance), Fig. 6 (retrieval), Table 2 (few-shot), Fig. 7 (prompt ensembling), Table 3 (B-16 ablation)


<br/>
# 예제



1) 어떤 모델을 무엇으로 학습하나? (학습 입력/출력 개요)
- 학습 대상 모델
  - SynthPath: Quilt1M B-32로 초기화된 디지털 병리 특화 VLM을 LoRA로 미세적응
  - SynthPath Context: 위 모델의 이미지/텍스트 인코더를 고정(frozen)하고, “컨텍스트 임베딩 레이어” 하나만 추가 학습(이미지 임베딩을 컨텍스트별로 조정하는 모듈)
- 학습 데이터(입력)
  - 이미지: Quilt1M에서 추출된 병리 이미지 패치(총 653,208개 고유 이미지 사용)
  - 텍스트: 원문 캡션에 대해 LLM(Mistral-7B-Instruct-v0.2)으로 생성한
    - 일반(문맥 제약 없음) 리라이트: 5,088,520개
    - 컨텍스트별 리라이트(장기/질병/세포): 3,053,112개 (각 컨텍스트에 맞춘 문장)
- 학습 출력(모델이 학습을 통해 얻게 되는 것)
  - SynthPath: 이미지/텍스트 인코더 파라미터(LoRA 어댑터)가 업데이트되어 이미지-텍스트 임베딩 정렬이 개선
  - SynthPath Context: 컨텍스트(장기/질병/세포)를 나타내는 작은 임베딩 레이어 h(c)을 학습하여, 이미지 임베딩 f(I)에 h(c)을 곱해(gating) 컨텍스트에 맞춘 정렬을 강화

2) 텍스트 생성(리라이트)과 컨텍스트 추출 방식
- 일반 리라이트(문맥 제약 없음)
  - 목적: 캡션 다양성 확대(패러프레이즈), 의미 보존
  - 규모: 5,088,520개(평균 추론 4.8초/샘플, A6000 GPU 약 6.8K 시간)
- 컨텍스트별 리라이트(컨텍스트 제약 있음)
  - 3개 컨텍스트: 장기(organ), 질병(disease), 세포(cell)
  - 입력: 원문 캡션
  - 출력: 각 컨텍스트에 맞춘 문장(예: 장기 정보만 추출한 문장, 질병 상태만 설명한 문장, 세포 수준 특징만 기술한 문장)
  - 장기 문맥은 비교적 쉬워 예시 3개, 질병은 5개 예시, 세포는 더 어려워 9개 예시를 LLM 프롬프트에 포함해 정확도 확보(‘해당 정보 없음’도 허용하여 환각 방지)
  - 규모: 3,053,112개
- 최종 합계: 8,141,632개(일반+컨텍스트 리라이트) 공개

3) 학습 형태(대조학습)와 배치별 입력/출력
- 공통(대조학습)
  - 입력(한 스텝에서의 배치): 이미지 배치 B개와, 각 이미지에 매칭된 텍스트(리라이트 중 랜덤 1개) 또는 컨텍스트별 텍스트
  - 출력: 이미지 임베딩과 텍스트 임베딩이 같은 쌍일수록 코사인 유사도가 커지도록 학습(반대쌍은 작아지도록)
- SynthPath 학습(일반 리라이트 사용)
  - 배치 입력: {(Ii, T̃i)}i=1..B (T̃i는 해당 이미지의 K개 리라이트 중 무작위 1개)
  - 배치 출력: 맞는 쌍 f(Ii)와 g(T̃i)의 유사도가 커지도록 파라미터 업데이트
- SynthPath Context 학습(컨텍스트 리라이트 사용)
  - 배치 입력: {(Ii, Tc_i)}i=1..B, c ∈ {organ, disease, cell}
  - 처리: 이미지 임베딩 f(Ii)에 컨텍스트 임베딩 h(c)을 원소곱(게이팅)하여 f(Ii)⊙h(c)로 조정 후, 해당 컨텍스트 문장 g(Tc_i)와 정렬을 학습
  - 배치 출력: 각 c에 대해 컨텍스트 합성 손실을 합산해 h(c)만 학습(인코더는 고정)

4) 테스트 데이터(입력/출력)와 구체적 테스크
A. 제로샷 분류(15개 데이터셋, 각 2,000 세트 프롬프트로 평균/표준편차 산출)
- 입력
  - 이미지: 각 벤치마크 데이터셋의 패치(예: Camelyon, LC Lung, LC Colon, CRC, Skin, Renal Cell, BACH, Databiox, Sicap, Mhist, Pannuke-Organ/Cell, Ocelot-Organ/Cell 등)
  - 텍스트: 데이터셋의 클래스들을 표현하는 “프롬프트 세트” 2,000개(데이터셋마다). 각 세트는 모든 클래스를 표현하는 문장 묶음
- 출력
  - 각 이미지에 대해 “가장 유사한 텍스트 프롬프트(=클래스)”를 선택 → 예측 라벨
  - 지표: 정확도(2,000 프롬프트 세트에 대한 평균±표준편차), 그리고 프롬프트 간 변동을 고려한 순열 검정 p-value
- 컨텍스트 모듈(SynthPath Context) 사용 시
  - 데이터셋 특성에 맞춰 organ/disease/cell 중 하나의 h(c)을 적용(논문 3.6.4에 데이터셋별 매핑)
  - 예: Pannuke-Organ/Ocelot-Organ → organ 컨텍스트, Camelyon/LC Lung/LC Colon/BACH/Databiox/Mhist/Skin Tumor → disease 컨텍스트, Pannuke-Cell/Ocelot-Cell/Skin/Renal Cell/CRC/Sicap → cell 컨텍스트

B. 텍스트-이미지 및 이미지-텍스트 검색(ARCH: PubMed, Textbooks)
- 입력
  - 텍스트-이미지: 텍스트 쿼리 1개와 이미지 후보 집합
  - 이미지-텍스트: 이미지 쿼리 1개와 텍스트 후보 집합
- 출력
  - 코사인 유사도 순 정렬 결과(Top-1/5/20/50/100 내에 GT 매칭 비율)
- 데이터
  - PubMed n=3309, Textbooks n=4305

C. 소량 샷(1-shot) 선형 프로빙
- 입력
  - 임베딩 특징: 사전학습된 VLM의 이미지 임베딩
  - 라벨: 각 클래스당 1개 샘플로 로지스틱 회귀 등 얕은 선형 분류기 학습
- 출력
  - 잔여 데이터에 대한 분류 정확도(여러 초기화 평균)

5) 하이퍼파라미터(핵심)
- SynthPath(LoRA)
  - LoRA rank/alpha=8, dropout=0.05(Linear/Conv1D에 적용)
  - Adam, lr=1e-5, weight decay=0.2, cosine 스케줄, batch=4096
  - 30 epoch, 8×A6000, Train/Val=0.85/0.15
- SynthPath Context
  - 인코더 고정, h(c)만 학습(출력 차원 512)
  - AdamW, lr=1e-3, batch=8192

6) 구체적 예시(가상의 인풋/아웃풋 사례)

예시 1) 학습(일반 리라이트, SynthPath)
- 입력 이미지(I): 유튜브 병리 강의에서 캡처된 H&E 폐 조직 패치(Quilt1M 파이프라인으로 확보)
- 원문 캡션(T): “Lung adenocarcinoma with glandular formations on H&E.”
- 일반 리라이트(T̃) 예(의미 보존, 표현 다양화)
  - “H&E-stained section of lung adenocarcinoma showing gland-like structures.”
  - “Histologic image of pulmonary adenocarcinoma demonstrating glandular architecture.”
  - “Adenocarcinoma of the lung on H&E with prominent gland formation.”
- 학습 단위 스텝의 입력
  - (I, 위 리라이트 중 랜덤 1개 T̃)
- 학습 스텝의 출력/목표
  - f(I)와 g(T̃)의 코사인 유사도를 최대화(동시에 배치 내 다른 텍스트와는 유사도 낮춤)

예시 2) 학습(컨텍스트 리라이트, SynthPath Context)
- 동일 이미지(I)와 원문 캡션에서 3개 컨텍스트 문장 생성:
  - Organ 컨텍스트(Torgan): “H&E-stained section of lung tissue.”
  - Disease 컨텍스트(Tdisease): “This slide shows lung adenocarcinoma.”
  - Cell 컨텍스트(Tcell): “Malignant gland-forming epithelial cells with nuclear atypia are present.”
- 학습 단위 스텝의 입력
  - (I, Torgan) → c=organ
  - (I, Tdisease) → c=disease
  - (I, Tcell) → c=cell
- 처리/목표
  - 각 c에 대해 f(I)⊙h(c)와 g(Tc)의 정렬을 동시에 좋게 만드는 손실을 합산해 h(c) 파라미터만 업데이트

예시 3) 제로샷 분류(LC Lung, 3클래스)
- 테스트 입력
  - 이미지: 테스트 패치 하나(예: 폐 선암)
  - 프롬프트 세트(가상 1세트 예)
    - “a histopathology image of benign lung parenchyma”
    - “a histopathology image of lung adenocarcinoma”
    - “a histopathology image of lung squamous cell carcinoma”
- 처리/출력
  - 각 프롬프트 문장 임베딩과 이미지 임베딩 유사도 계산
  - 가장 유사도가 큰 문장의 클래스(여기서는 ‘lung adenocarcinoma’)를 예측 라벨로 출력
  - SynthPath Context 사용 시: disease 컨텍스트 h(disease)를 이미지 임베딩에 곱해 분류

예시 4) 제로샷 분류(Organ vs Disease 컨텍스트 선택이 애매한 경우)
- 데이터셋: Mhist(과형성 용종 vs SSA)
  - 질병 분류 관점(disease)도 가능하지만 정확한 컨텍스트 선택이 애매
  - 논문에서는 disease 컨텍스트를 적용(SSA는 전암성 가능)
- 처리/출력
  - disease 컨텍스트로 모듈레이션 후 두 클래스 프롬프트 중 더 유사한 쪽을 선택

예시 5) 텍스트→이미지 검색(ARCH PubMed)
- 입력
  - 텍스트 쿼리: “H&E image of clear cell renal cell carcinoma with optically clear cytoplasm”
  - 이미지 후보: PubMed 서브셋의 모든 이미지
- 출력
  - 코사인 유사도 기준 상위 순위 목록 반환(Top-1/5/20/50/100 내 GT 이미지 포함 비율 계산)

예시 6) 이미지→텍스트 검색(ARCH Textbooks)
- 입력
  - 이미지 쿼리: 위장관 점막에서 술잔세포가 풍부한 정상 결장 조직 이미지
  - 텍스트 후보: 교과서 캡션들
- 출력
  - 코사인 유사도 순으로 정렬된 텍스트 목록(Top-k 매칭 비율)

예시 7) 1-shot 선형 프로빙(Camelyon, 정상 vs 전이)
- 입력(학습용)
  - 클래스별 1장씩(총 2장)의 이미지 임베딩과 라벨
- 처리
  - 간단한 로지스틱 회귀 분류기를 훈련
- 출력(평가용)
  - 나머지 테스트 샘플의 정확도

7) 평가 산출물(출력)
- 제로샷: 데이터셋별 정확도(평균±표준편차), 2,000 프롬프트 세트 기반 순열 검정 p-value
- 검색: Top-1/5/20/50/100 매칭 비율
- 선형 프로빙: 정확도 평균(여러 분류기 초기화 평균)

8) 데이터셋-컨텍스트 매핑(추론 시 SynthPath Context)
- organ: Pannuke-Organ, Ocelot-Organ
- disease: Bach, LC Lung, LC Colon, Camelyon, Databiox, Mhist, Skin Tumor
- cell: Pannuke-Cell, Ocelot-Cell, Skin, Renal Cell, CRC, Sicap




1) What is trained with what? (training inputs/outputs)
- Models
  - SynthPath: a digital pathology VLM initialized from Quilt1M B-32 and adapted with LoRA
  - SynthPath Context: freezes the encoders of SynthPath and trains a single “context embedding layer” h(c) that modulates image embeddings
- Training data (inputs)
  - Images: 653,208 unique patches from Quilt1M
  - Texts: LLM-generated rewrites of original captions using Mistral-7B-Instruct-v0.2
    - General rewrites (no context constraint): 5,088,520
    - Context-specific rewrites (organ/disease/cell): 3,053,112
- Training outputs
  - SynthPath: updated (LoRA) encoder weights that better align image/text embeddings
  - SynthPath Context: learned h(c) that gates image embeddings per context to better match context-specific texts

2) How are rewrites and contexts produced?
- General rewrites
  - Goal: increase textual diversity while preserving meaning
  - Scale: 5,088,520
- Context-specific rewrites
  - Contexts: organ, disease, cell
  - Prompts include sufficient examples (organ≈3, disease≈5, cell≈9) and allow “None” to avoid hallucinations
  - Scale: 3,053,112
- Total: 8,141,632 synthetic captions released

3) Training setup (contrastive learning)
- Common
  - Batch input: B image–text pairs; maximize similarity for matched pairs, minimize for mismatched
- SynthPath (general rewrites)
  - Input per step: {(Ii, T̃i)} with T̃i randomly drawn from the K rewrites for image Ii
  - Output: higher cosine similarity for f(Ii) and g(T̃i)
- SynthPath Context (context rewrites)
  - Input per step: {(Ii, Tc_i)}, c ∈ {organ, disease, cell}
  - Compute f(Ii)⊙h(c) and align it with g(Tc_i); sum loss over contexts; update only h(c)

4) Test data (inputs/outputs) and tasks
A. Zero-shot classification (15 datasets; 2,000 prompt sets per dataset)
- Inputs
  - Images from benchmark datasets
  - Text prompts: 2,000 sets per dataset; each set covers all classes with differently phrased sentences
- Outputs
  - Predicted class per image by selecting the prompt with highest similarity
  - Metrics: accuracy (mean±std across 2,000 sets) and permutation-test p-values
- With SynthPath Context
  - Apply the most suitable h(c) per dataset (mapping described in the paper)

B. Text-to-image and image-to-text retrieval (ARCH: PubMed, Textbooks)
- Inputs
  - Text-to-image: one text query and a set of images
  - Image-to-text: one image query and a set of captions
- Outputs
  - Ranked list by cosine similarity; report Top-1/5/20/50/100 match rates

C. Few-shot (1-shot) linear probing
- Inputs
  - Image embeddings from the frozen VLM; labels for 1 example per class
- Outputs
  - Accuracy of a shallow linear classifier on the remaining data

5) Key hyperparameters
- SynthPath (LoRA): rank/alpha=8, dropout=0.05; Adam lr=1e-5, wd=0.2, cosine schedule; batch=4096; 30 epochs on 8×A6000; split 0.85/0.15
- SynthPath Context: encoders frozen; learn h(c)∈R^512; AdamW lr=1e-3; batch=8192

6) Concrete examples (illustrative)

Example 1) Training (general rewrites, SynthPath)
- Input image (I): H&E lung patch from a YouTube lecture (via Quilt1M pipeline)
- Original caption (T): “Lung adenocarcinoma with glandular formations on H&E.”
- General rewrites (T̃) examples:
  - “H&E-stained section of lung adenocarcinoma showing gland-like structures.”
  - “Histologic image of pulmonary adenocarcinoma demonstrating glandular architecture.”
  - “Adenocarcinoma of the lung on H&E with prominent gland formation.”
- Per-step input/output
  - Feed (I, random T̃); optimize to increase cosine similarity f(I)·g(T̃)

Example 2) Training (context rewrites, SynthPath Context)
- Context-specific texts from the same image:
  - Organ (Torgan): “H&E-stained section of lung tissue.”
  - Disease (Tdisease): “This slide shows lung adenocarcinoma.”
  - Cell (Tcell): “Malignant gland-forming epithelial cells with nuclear atypia are present.”
- Per-step input/output
  - For each c, use (I, Tc); align f(I)⊙h(c) with g(Tc); sum losses across c; update h(c) only

Example 3) Zero-shot classification (LC Lung, 3 classes)
- Test input
  - Image: lung adenocarcinoma patch
  - Prompt set (sample):
    - “a histopathology image of benign lung parenchyma”
    - “a histopathology image of lung adenocarcinoma”
    - “a histopathology image of lung squamous cell carcinoma”
- Output
  - Predict the class with highest similarity (here: “lung adenocarcinoma”)
  - With SynthPath Context: apply h(disease) before similarity

Example 4) Zero-shot with ambiguous context choice (Mhist)
- Task: hyperplastic polyp vs sessile serrated adenoma
- The paper uses the disease context; proceed accordingly and select the higher-similarity class

Example 5) Text→image retrieval (ARCH PubMed)
- Input
  - Text query: “H&E image of clear cell renal cell carcinoma with optically clear cytoplasm”
  - Candidate images: all PubMed images
- Output
  - Ranked images by cosine similarity; compute Top-k match rates

Example 6) Image→text retrieval (ARCH Textbooks)
- Input
  - Image query: normal colon mucosa with abundant goblet cells
  - Candidate texts: textbook captions
- Output
  - Ranked caption list; report Top-k match rates

Example 7) 1-shot linear probing (Camelyon: normal vs metastasis)
- Input (train)
  - One image embedding per class with labels
- Process
  - Train logistic regression
- Output (test)
  - Classification accuracy on the remaining images

7) Reported outputs
- Zero-shot: accuracy mean±std across 2,000 prompt sets; permutation-test p-values
- Retrieval: Top-1/5/20/50/100 match rates
- Linear probing: mean accuracy over multiple classifier initializations

8) Dataset-to-context mapping (inference with SynthPath Context)
- organ: Pannuke-Organ, Ocelot-Organ
- disease: BACH, LC Lung, LC Colon, Camelyon, Databiox, Mhist, Skin Tumor
- cell: Pannuke-Cell, Ocelot-Cell, Skin, Renal Cell, CRC, Sicap

참고: 위 수치(데이터 규모, 하이퍼파라미터, 데이터셋 목록/매핑, 평가 방식)는 모두 논문 본문에 명시된 값을 정리한 것이고, 예시 문장(캡션/프롬프트)은 논문 절차를 충실히 따르는 이해용 가상 사례입니다.

<br/>
# 요약


메서드: Mistral‑7B‑Instruct‑v0.2로 Quilt1M 캡션을 5,088,520개 일반 재작성과 3,053,112개 문맥별(장기·질병·세포) 재작성으로 확장하고, 대비학습 기반 VLM을 재학습했으며, 이미지 임베딩에 문맥 임베딩 h(c)를 곱하는 콘텍스트 모듈레이션 층을 추가하고 15개 데이터셋에 대해 각 2K(총 170K) 프롬프트 세트를 생성해 제로샷 평가의 프롬프트 민감도를 체계적으로 측정했습니다. 
결과: SynthPath는 15개 중 11개 데이터셋에서 Quilt1M을, 14개에서 CLIP을 능가(평균 +0.04)했고, SynthPath Context는 추가로 +0.015 향상되어 11개에서 최고 성능을 보였으며, ARCH(PubMed·Textbooks) 이미지↔텍스트 검색에서도 20개 지표 중 19개에서 우수했고 쌍별 순열검정은 대부분 P<.001로 유의했습니다. 
예시: 총 8,141,632개의 합성 캡션으로 공개 병리 캡션 최대 규모를 구축했고, Crc(+0.13), LC Lung(+0.11), Ocelot Organ(+0.09)에서 큰 이득을 보였으며, Pannuke/Ocelot Organ에는 장기, LC Lung/Colon·Camelyon에는 질병, Pannuke Cell·Skin·Renal Cell에는 세포 콘텍스트를 적용했습니다.

Methods: Using Mistral‑7B‑Instruct‑v0.2, we enriched Quilt1M with 5,088,520 general rewrites and 3,053,112 context‑specific (organ, disease, cell) rewrites, retrained a contrastive VLM, introduced a context modulation layer h(c) that multiplies image embeddings, and built 2K prompt sets per 15 datasets (170K total) to rigorously assess prompt sensitivity in zero‑shot classification. 
Results: SynthPath outperformed Quilt1M on 11/15 and CLIP on 14/15 zero‑shot tasks (avg +0.04), while SynthPath Context added +0.015 and was best on 11/15; it also led on ARCH (PubMed/Textbooks) image↔text retrieval in 19/20 top‑k metrics, with pairwise permutation tests mostly P<.001. 
Examples: We released the largest open pathology caption set (8,141,632 synthetic captions), saw large gains on Crc (+0.13), LC Lung (+0.11), Ocelot Organ (+0.09), and applied organ context to Pannuke/Ocelot Organ, disease to LC Lung/Colon and Camelyon, and cell to Pannuke Cell/Skin/Renal Cell.

<br/>
# 기타



피규어(도표)
- Figure 1: 일반(컨텍스트 비제한) 캡션 리라이트 프롬프트 예시
  - 결과: Mistral-7B-Instruct로 의미 동등한 문장 리라이트를 대량 생성(5,088,520개, 평균 4.8초/샘플, 약 6.8K GPU-시간).
  - 인사이트: 간단한 프롬프트 설계로 병리 도메인의 텍스트 다양성을 크게 확장 가능. 일부 “추론적” 정보 추가도 허용해 텍스트-이미지 정렬 학습에 도움.

- Figure 2: 질병(disease) 컨텍스트 캡션 생성 프롬프트
  - 결과: 2단계(정상/질병 판정 → 근거 설명), 예시 5개가 최적. 질병 맥락 캡션 1/3 가량 확보.
  - 인사이트: 단계적 질의와 예시 수 조정으로 LLM 환각을 줄이고 정확한 질병 컨텍스트를 추출. 이후 컨텍스트-인식 학습의 토대가 됨.

- Figure 3: 제로샷 평가용 대규모 프롬프트 셋 생성(LC Lung 예시)
  - 결과: 데이터셋별 2,000개 프롬프트 셋, 총 170K 프롬프트 구축(15개 테스트셋). 클래스별 문장 변형을 LLM이 생성.
  - 인사이트: 제로샷 성능이 프롬프트 문구에 민감함을 정량화하고, 공정하고 강건한 비교를 가능하게 함(평균·분산·통계검정까지).

- Figure 4: 컨텍스트-모듈레이션(맥락 변조) 학습 구조
  - 결과: 이미지 임베딩 f(I)와 학습 가능한 컨텍스트 임베딩 h(c)를 Hadamard 곱으로 결합, 컨텍스트별 캡션과 대조학습(organ/disease/cell).
  - 인사이트: 이미지·텍스트 인코더는 동결하고, 얇은 단일 임베딩 레이어만 학습하여 “컨텍스트 정렬”을 달성. 경량 추가 학습으로 성능 향상.

- Figure 5: 15개 제로샷 과제(각 2K 프롬프트 셋) 성능 비교
  - 결과: SynthPath가 Quilt1M 대비 11/15개 데이터셋에서 향상(평균 +0.04). 특히 Crc(+0.13), LC Lung(+0.11), Ocelot Organ(+0.09), LC Colon(+0.08), Pannuke Cell(+0.07). SynthPath Context는 11/15개에서 최고 성능이며 SynthPath 대비 평균 +0.015 추가 향상.
  - 인사이트: LLM 리라이트만으로도 디지털 병리 VLM의 제로샷 강건성이 크게 개선. 컨텍스트 변조는 추가 이득 제공(기존 향상 대비 +37.5%). 에러바(표준편차)는 프롬프트 민감도를 보여주며, 제안 기법이 그 변동성을 낮춤.

- Figure 6: ARCH(PubMed, Textbooks) 텍스트↔이미지 검색(top-k)
  - 결과: SynthPath가 20개(두 세트 × 두 태스크 × top-k 5종) 중 19개 항목에서 최고. 예외: Textbooks에서 image-to-text top-5만 근소 열세.
  - 인사이트: 실제 문헌 캡션과의 정렬이 크게 개선됨. 다양한 컨텍스트가 섞인 데이터에서는 컨텍스트 모듈레이션 모델(=SynthPath Context)이 추가 이득을 내기 어렵다는 가정과 부합.

- Figure 7: 프롬프트 앙상블(2K 프롬프트 소프트맥스 평균) 결과
  - 결과: 앙상블을 해도 제안 모델들이 10/15 데이터셋에서 여전히 우수 혹은 동률.
  - 인사이트: “추론 시 프롬프트 공들임”보다 “학습 시 리라이트·컨텍스트 설계”가 더 큰 이득을 줌을 시사.

테이블
- Table 1: 모델간 유의성 검정(퍼뮤테이션, 2K 프롬프트 분산 반영)
  - 결과: SynthPath Context > SynthPath는 13/15에서 유의. SynthPath Context > Quilt1M 및 > CLIP은 전 과제에서 유의(p<.001). SynthPath > Quilt1M도 전 과제에서 유의.
  - 인사이트: 프롬프트 변동성을 고려해도 성능 우위가 통계적으로 견고. Databiox(=P=.312)는 컨텍스트 추가 이득이 거의 없는 예외 케이스.

- Table 2: 1-shot 선형 프로빙(8개 분류기 평균)
  - 결과: 제안 모델들이 15개 중 9개 데이터셋에서 베이스라인을 능가 또는 동률. 전반적 우세이나 제로샷만큼 일관되진 않음.
  - 인사이트: 본 연구의 주목적(제로샷 강건성 향상)에는 부합하며, 학습된 표현이 전이성 면에서도 경쟁력. 다만 모든 과제에서 일관된 우세를 보장하진 않음.

- Table 3: 백본 스케일(B-16) 절개 실험(기저 모델 선택)
  - 결과: SynthPath B-16이 Quilt1M B-16 대비 14/15에서 유의하게 우수. B-32에서 관측된 이득 패턴이 B-16에도 재현.
  - 인사이트: 개선 효과가 특정 백본/해상도 설정에 국한되지 않음. 방법의 일반성이 확인됨.

부록/서플리먼트(본문에 인용된 항목 중심)
- A.1(일반 리라이트 프롬프트 예시), A.2~A.4(세부 컨텍스트 프롬프트)
  - 결과: 장기(organ)는 3개 예시로 충분, 질병(disease)은 5개, 세포(cell)는 9개 예시와 ‘None’ 출력을 명시해 환각 억제. 총 8,141,632개 합성 캡션(일반+컨텍스트), 약 10,800 GPU-시간.
  - 인사이트: 컨텍스트별 난이도에 맞춘 예시 수·출력 규약이 고품질 합성 캡션을 담보. 대규모 캡션은 학습 강건성의 핵심.

- A.5.1~A.5.15(각 테스트셋용 프롬프트 셋 생성 입력/예시)
  - 결과: 15개 데이터셋 × 2,000셋 = 170K 프롬프트를 재현 가능하게 공개.
  - 인사이트: 제로샷 평가의 공정성과 재현성 제고. 향후 모델 비교의 표준 벤치마크로 활용 가치.

- C.1(클래스 출현 통계), C.2(선형 프로빙 상세), D.2(확장 결과)
  - 결과/인사이트: 데이터 편향·불균형 해석에 도움, 프로빙의 실험 안정성 확인, B-16 등 추가 결과로 방법론 일관성 뒷받침.

- 환각 분석(본문 4.6, 서플리먼트 Table 1)
  - 결과: 임의 100개 샘플 중 15개는 경미한 환각(숫자·약어 오류 등), 16개는 의학적으로 타당한 비자명 추가정보, 나머지는 주로 교정/개선. 
  - 인사이트: 합성 텍스트의 노이즈는 존재하나 대체로 경미하고, 유익한 정보 보강도 빈번. 전체 성능 향상에 비추어 관리 가능한 수준.

추가 맥락(컨텍스트 선택 가이드, 3.6.4)
- 결과: 질병 컨텍스트는 이분(정상/암) 및 암 등급/서브타입 과제에 적용, 장기 컨텍스트는 장기 분류에, 세포 컨텍스트는 세포·조직 유형 혼재 과제에 효과적. Mhist/ Skin Tumor처럼 경계적인 과제에서는 선택이 난해하여 성능 하락 사례도 존재.
- 인사이트: 컨텍스트 선택은 데이터셋 정의와 클래스 서술의 “의미적 초점”에 맞추는 것이 중요. 다단계(계층) 접근이 필요한 복합 과제는 향후 과제.

요약 인사이트
- 학습 시 LLM 리라이트(일반+컨텍스트)로 텍스트 다양성과 정렬 품질을 키우면, 추론 시 복잡한 프롬프트 엔지니어링 없이도 제로샷·검색 성능이 일관되게 향상.
- 프롬프트 민감성이 큰 영역(디지털 병리)에서 평균·분산·통계검정을 통한 “강건성” 비교가 필수이며, 제안 방법은 그 기준을 충족.
- 컨텍스트 모듈레이션은 인코더를 동결한 채 얇은 레이어만 학습하는 비용-효율적 개선책으로, 과제 적합 컨텍스트 선택 시 추가 이득을 제공.






Figures
- Figure 1: Prompt for general (context-free) caption rewrites
  - Result: Generated 5,088,520 semantically equivalent rewrites using Mistral-7B-Instruct (~4.8s per sample, ~6.8K GPU-hours).
  - Insight: Simple prompting scales textual diversity efficiently; occasional inferred details can aid alignment learning.

- Figure 2: Prompt for disease-context captions
  - Result: Two-stage answering (diagnosis then rationale) with five examples worked best.
  - Insight: Careful prompt design and example count reduces hallucinations and yields cleaner disease context.

- Figure 3: Framework to generate large zero-shot prompt sets (LC Lung example)
  - Result: 2,000 prompt sets per dataset, 170K prompts total across 15 datasets.
  - Insight: Enables fair, robust evaluation of prompt-sensitive zero-shot performance (means, variances, and statistical tests).

- Figure 4: Context-modulation training framework
  - Result: Learn a small context embedding h(c) and combine with image embeddings via Hadamard product for contrastive learning (organ/disease/cell).
  - Insight: Lightweight, plug-in head boosts image–text alignment without finetuning encoders.

- Figure 5: Zero-shot accuracy across 15 tasks with 2K prompt sets
  - Result: SynthPath improves over Quilt1M in 11/15 datasets (avg +0.04), notably Crc(+0.13), LC Lung(+0.11), Ocelot Organ(+0.09), LC Colon(+0.08), Pannuke Cell(+0.07). SynthPath Context is best on 11/15 with an additional +0.015 over SynthPath.
  - Insight: LLM rewrites substantially strengthen robustness to prompt variation; context modulation adds a further +37.5% relative gain over the +0.04 baseline improvement. Error bars highlight reduced prompt sensitivity.

- Figure 6: ARCH (PubMed, Textbooks) text-to-image and image-to-text retrieval (top-k)
  - Result: SynthPath wins in 19/20 metrics; only misses on image-to-text top-5 (Textbooks).
  - Insight: Stronger alignment with real captions; as expected, context modulation adds little when multiple contexts are mixed.

- Figure 7: Prompt ensembling with 2K prompts
  - Result: Proposed models still outperform or match in 10/15 datasets under naive ensembling.
  - Insight: Training-time rewrites/context learning yield gains beyond what test-time prompt tinkering can recover.

Tables
- Table 1: Permutation tests (accounting for variance across 2K prompt sets)
  - Result: SynthPath Context > SynthPath in 13/15; SynthPath Context > Quilt1M and > CLIP in all tasks (mostly p<.001); SynthPath > Quilt1M in all.
  - Insight: Performance advantages are statistically robust to prompt variability. Databiox is a near-tie case (P=.312).

- Table 2: 1-shot linear probing (mean over 8 classifiers)
  - Result: Proposed models outperform or match baselines in 9/15 datasets. Not universally superior.
  - Insight: Complements the main zero-shot objective; learned representations transfer reasonably but benefits are task-dependent.

- Table 3: Backbone scale ablation (B-16)
  - Result: SynthPath B-16 > Quilt1M B-16 in 14/15 with similar gains as B-32.
  - Insight: Improvements generalize across backbones/resolutions.

Appendices/Supplementary (as referenced)
- A.1 (general rewrite prompts), A.2–A.4 (context prompts)
  - Result: 3 examples for organ, 5 for disease, 9 for cell with an explicit ‘None’ output to curb hallucinations. Total 8,141,632 synthetic captions (general + context), ~10,800 GPU-hours.
  - Insight: Tailoring example counts/output constraints per context quality-controls the LLM and underpins robust training.

- A.5.1–A.5.15 (prompt-set inputs/examples per test dataset)
  - Result: 170K prompts released for reproducibility.
  - Insight: Establishes a fair, shareable benchmark for prompt-sensitive zero-shot evaluation.

- C.1 (class occurrence statistics), C.2 (linear probing details), D.2 (extended results)
  - Result/Insight: Support interpretation (class balance), stabilize probing evaluation, and confirm consistency at different backbones.

- Hallucination audit (Sec. 4.6; Supplementary Table 1)
  - Result: In 100 random samples, 15 minor hallucinations, 16 medically valid non-trivial additions; the rest primarily corrections/improvements.
  - Insight: Noise exists but is largely mild; beneficial information additions are common, and overall gains indicate noise is manageable.

Additional context (how context was chosen; Sec. 3.6.4)
- Result: Disease context for binary cancer/grade/subtype tasks; organ context for organ ID; cell context for cell/tissue-type mixtures. Some ambiguous datasets (e.g., Mhist, Skin Tumor) showed drops when context choice is non-trivial.
- Insight: Matching the context to the dataset’s semantic focus is critical; multi-stage hierarchical strategies may help for cross-context tasks.

Key takeaways
- Training-time LLM rewrites (general + context-aware) significantly improve zero-shot and retrieval without heavy inference-time prompt engineering.
- Properly measuring and controlling for prompt sensitivity (via large prompt sets and statistics) reveals consistent, significant advantages.
- Context modulation is a cost-effective head that adds meaningful gains when the right context is selected for the task.

<br/>
# refer format:



BibTeX
@inproceedings{Bahadir2025,
  title        = {LLM-generated Rewrite and Context Modulation for Enhanced Vision Language Models in Digital Pathology},
  author       = {Bahadir, Cagla Deniz and Akar, Gozde B. and Sabuncu, Mert R.},
  booktitle    = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages        = {327--336},
  year         = {2025}
}

Chicago 스타일

Bahadir, Cagla Deniz, Gozde B. Akar, and Mert R. Sabuncu. 2025. “LLM-generated Rewrite and Context Modulation for Enhanced Vision Language Models in Digital Pathology.” In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 327–336.
