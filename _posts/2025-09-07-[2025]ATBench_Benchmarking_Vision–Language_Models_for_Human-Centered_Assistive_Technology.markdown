---
layout: post
title:  "[2025]ATBench: Benchmarking Vision–Language Models for Human-Centered Assistive Technology"
date:   2025-09-07 20:07:42 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: - 메서드: 사용자 참여형 연구로 선정한 5개 과제(파노픽 분할/깊이/텍스트 인식/OCR/캡셔닝/VQA)를 묶은 @BENCH와, X-Decoder 기반에 태스크별 프롬프트·단일 가중치(62M)·OCR 문자 기반 토크나이저를 도입해 픽셀/토큰 출력을 동시에 수행하는 @MODEL을 제안합니다.  


짧은 요약(Abstract) :
한글 설명
- 이 논문은 시각장애인을 돕는 보조공학(AT)에 적용할 수 있는 비전-언어 모델(VLM)의 성능을 제대로 평가하기 위해, 사용자 참여형 사전 조사에 기반한 새로운 벤치마크(@BENCH)를 제안합니다.
- @BENCH는 시각장애인의 실제 요구를 반영해 다섯 가지 핵심 과제를 포함합니다: 파놉틱 세그멘테이션, 깊이 추정, OCR, 이미지 캡셔닝, VQA.
- 또한 이 다섯 가지 작업을 하나의 모델로 동시에 처리하고, 추후 더 많은 보조 기능으로 확장 가능한 AT 모델(@MODEL)도 소개합니다.
- 제안한 프레임워크는 멀티모달 정보를 결합해 다양한 과제에서 우수한 성능을 보이며, 시각장애인에게 보다 종합적인 도움을 제공합니다.
- 광범위한 실험을 통해 제안 방식의 효과성과 일반화 능력을 입증했습니다.

English (Abstract)
As Vision-Language Models (VLMs) advance, human-centered Assistive Technologies (ATs) for helping People with Visual Impairments (PVIs) are evolving into generalists, capable of performing multiple tasks simultaneously. However, benchmarking VLMs for ATs remains under-explored. To bridge this gap, we first create a novel AT benchmark (@BENCH). Guided by a pre-design user study with PVIs, our benchmark includes the five most crucial vision-language tasks: Panoptic Segmentation, Depth Estimation, Optical Character Recognition (OCR), Image Captioning, and Visual Question Answering (VQA). Besides, we propose a novel AT model (@MODEL) that addresses all tasks simultaneously and can be expanded to more assistive functions for helping PVIs. Our framework exhibits outstanding performance across tasks by integrating multi-modal information, and it offers PVIs a more comprehensive assistance. Extensive experiments prove the effectiveness and generalizability of our framework.


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



요약
- @BENCH: 시각장애인 보조기술(AT)을 위한 비전-언어 모델 평가 벤치마크. 사용자 참여형 설문으로 선정된 5개 핵심 과제(PS/DE/OCR/IC/VQA)와 효율-성능 절충 평가 기준을 포함.
- @MODEL: 단일 가중치로 다중 과제를 동시에 수행하는 범용 비전-언어 모델. 태스크-특화 프롬프트에 기반한 통일 입력(“이미지 + 프롬프트”), 픽셀/토큰 양방향 출력, OCR용 문자 단위 토크나이저 등으로 구성.

1) 문제 제기와 동기
- 기존 LVLM 벤치마크는 언어/크로스모달 과제 중심이며, 시각장애인에게 중요한 순수 비전 과제(세그멘테이션, 깊이 등)와 효율성을 충분히 다루지 못함.
- 실제 AT에서 요구되는 것은 다중 과제를 동시에, 효율적으로, 맥락에 맞게 처리하는 능력.
- 연구 질문: VLM은 시각장애인 보조기술을 실질적으로 뒷받침할 준비가 되었는가?

2) @BENCH 설계: 사용자 중심 벤치마크
- 사용자 참여형 조사: 접근성 전문가 2인과 사전 설계를 거친 뒤, 시각장애·저시력 당사자 7인을 대상으로 8가지 기능에 대해 흥미/사용빈도/중요도(각 1~5점)를 평가.
- 결과: 텍스트 인식(OCR)과 객체 인식(PS)이 가장 높은 점수. 장면 인식은 제외. 나머지 기능군을 포괄하도록 5개 과제를 최종 선정.
  - Panoptic Segmentation(PS): ADE20K, PQ
  - Depth Estimation(DE): NYU v2(실내 중심), RMSE
  - Optical Character Recognition(OCR): 대규모 합성데이터(MJ, ST) 학습, IC13/15, IIIT5K, SVT, SVTP, CUTE에서 정확도
  - Image Captioning(IC): VizWiz Cap, BLEU-1, CIDEr
  - Visual Question Answering(VQA): VizWiz VQA, 정확도
- 효율-성능 절충: AT 맥락에서 효율성을 중요한 축으로 포함. 파라미터 수를 효율 지표로 사용하여 비교.

3) @MODEL 아키텍처: X-Decoder 기반 범용 모델
- 구성 요소
  - 이미지 인코더: 시각 피처 추출.
  - 공유 텍스트 인코더(2개): CLIP 기반 텍스트 임베딩. 프롬프트/라벨/질문 등을 인코딩(가중치 공유).
  - 트랜스포머 디코더: latent queries(일반 잠재 쿼리) + textual queries(텍스트 쿼리)를 활용한 일반화된 디코딩.
- 출력 양식
  - 픽셀 수준 출력: PS, DE 등 밀집 예측.
  - 토큰 수준 출력: VQA, IC, OCR 등 언어 토큰 생성.

4) 핵심 설계 기법
- 태스크-특화 프롬프트에 기반한 통일 입력(“이미지 + 프롬프트”)
  - 동기: 멀티헤드(태스크별 출력 헤드) 방식은 구조가 비대해지고, 포터블 AT 배포에 부담. 또한 태스크 간 일반화가 제한.
  - 방식: 모든 태스크를 “이미지 + 프롬프트” 형태로 통일하여 입력. 예) VQA는 질문 자체를 프롬프트로 사용, 세그멘테이션/깊이/캡셔닝/OCR은 태스크를 명시하는 프롬프트 문구를 제공.
  - 장점
    1) 다양한 태스크 입력을 통일해 멀티태스크 학습을 단순화.
    2) 학습 초기에 태스크 구분이 명확해져 태스크별 표현학습이 용이.
    3) 별도 출력 헤드 증설 없이 파라미터 증가를 최소화.
- 단일 가중치로 모든 태스크 실행
  - 동일 파라미터 집합으로 5개 과제를 동시에 처리하여, 모델 수/메모리/배포 복잡도를 크게 줄임.
  - 휴대형(스마트 글래스 등) 디바이스에 실용적.

5) OCR 전용 토크나이저 설계(특수 기법)
- 문제: 기본 CLIP 서브워드 토크나이저(약 50k 어휘)는 OCR의 단순 문자 인식 특성과 불일치. OCR은 의미적 문장 생성보다 글자 자체 인식이 중요.
- 관찰: 영어 OCR 데이터는 단일 텍스트 인스턴스, 주로 26자+숫자 10개, 길이<15인 경우가 많음.
- 해법: 문자 단위 토크나이저(character-based)와 제한 어휘(알파벳+숫자 중심)를 사용
  - 대규모 서브워드 어휘 대신 문자 어휘를 사용하면, 모델의 예측 공간과 데이터 분포가 일치하여 학습이 안정되고 정확도 상승.
  - 실험적으로 문자 단위 전환과 제한 어휘 적용이 단계적으로 성능을 향상.

6) 학습·평가 전략과 결과 해석
- 학습 구성
  - 프리트레이닝 없이, 다중 데이터셋을 동시에 사용하는 멀티태스크 학습.
  - 태스크-특화 프롬프트를 통해 하나의 통합 학습 루프로 5개 과제를 모두 학습.
- 비교 평가
  - 범용 모델 비교: Unified-IO, X-Decoder, GIT, PaLI 등과 비교 시, @MODEL은 훨씬 작은 파라미터(약 62M)로 보조기술 관련 과제 전반에 걸쳐 경쟁적 성능을 보임.
  - 단일 과제 SOTA와 비교: PS/DE/OCR/IC/VQA의 전문 모델 대비, 프리트레이닝 없이도 여러 과제에서 근접하거나 일부는 능가.
- 멀티태스크 vs 싱글태스크
  - 관찰: 멀티태스크에서 캡셔닝과 VQA는 상호 보완적이라 개선, 반면 대규모 OCR 데이터로 인해 균형이 어려워 OCR은 일부 저하 가능.
  - 시사점: 데이터 크기/난이도 균형, 손실 가중 조정 등의 멀티태스크 최적화가 중요.

7) 적용 가능성 및 확장
- 파생 기능: 장애물 회피, 간단 내비게이션, 실내 거리/문·카운터 위치 안내, 표지판·문서 읽기, 장면·객체 파악 등으로 자연 확장.
- 배포 이점: 한 벌 파라미터로 다기능 제공이 가능하여, 온디바이스·웨어러블 환경에서 메모리·연산·지연을 절감.

8) 한계와 향후 과제
- 프리트레이닝 미적용: 프리트레이닝을 추가하면 더 높은 성능이 기대됨.
- 멀티태스크 데이터 불균형: OCR 대규모 데이터 편중 대응을 위한 리샘플링/손실 재가중/커리큘럼 학습 등이 유효할 수 있음.
- 기능 개발·실사용 배포: 사용자 피드백 기반의 기능 고도화와 실제 보급을 위한 시스템 엔지니어링이 다음 과제로 제시됨.






Summary
- @BENCH: An assistive-technology-centric benchmark for Vision-Language Models, selected via a human-in-the-loop user study. It covers five key tasks (PS/DE/OCR/IC/VQA) and explicitly evaluates the efficiency–performance trade-off.
- @MODEL: A generalist VL model that performs all five tasks with a single set of weights. It uses task-specific prompts to unify inputs (“image + prompt”), supports both pixel-level and token-level outputs, and introduces an OCR-specific character tokenizer.

1) Motivation
- Existing LVLM benchmarks emphasize language/cross-modal tasks, under-emphasizing vision-specific tasks crucial for people with visual impairments (e.g., segmentation, depth) and device-side efficiency.
- Real AT systems need multi-task, context-aware, and efficient inference.
- Research question: Are VLMs ready to empower assistive technology for people with visual impairments?

2) @BENCH: User-centered design
- User study: With 7 blind/low-vision participants, eight functions were rated on interest, usage frequency, and importance.
- Findings: Text recognition and object recognition rank highest. Scene recognition is excluded. Final five tasks:
  - Panoptic Segmentation (PS): ADE20K, PQ
  - Depth Estimation (DE): NYU v2 (indoor), RMSE
  - Optical Character Recognition (OCR): train on MJ/ST; evaluate on IC13/15, IIIT5K, SVT, SVTP, CUTE
  - Image Captioning (IC): VizWiz Cap, BLEU-1, CIDEr
  - Visual Question Answering (VQA): VizWiz VQA, accuracy
- Efficiency–performance: Includes parameter count as an efficiency metric, reflecting real AT constraints.

3) @MODEL architecture (built on X-Decoder)
- Components
  - Image encoder for visual features.
  - Two shared text encoders (CLIP-based) for prompts/labels/questions.
  - Transformer decoder with latent queries and textual queries for generalized decoding.
- Outputs
  - Pixel-level: dense predictions for PS and DE.
  - Token-level: generated sequences for VQA, IC, and OCR.

4) Key design choices
- Task-specific prompt and unified input (“image + prompt”)
  - Rationale: Multi-head outputs inflate architecture and hinder portable deployment; prompt-based unification improves generalization and simplifies training.
  - Advantages
    1) Unifies heterogeneous inputs across tasks.
    2) Enables early task disambiguation, improving feature learning.
    3) Minimizes parameter overhead by avoiding per-task heads.
  - Example: Use the natural question as the VQA prompt; use explicit task prompts for PS/DE/IC/OCR.
- Single weight set for all tasks
  - Reduces model count, memory, and complexity, important for wearables and on-device inference.

5) OCR-specific tokenizer (special technique)
- Problem: A 50k subword CLIP tokenizer mismatches OCR’s character-level nature; OCR focuses on literal recognition rather than semantic composition.
- Observations: Most English OCR samples contain one short text string composed of 26 letters and 10 digits.
- Solution: Character-based tokenizer with a limited vocabulary
  - Aligns prediction space with data, stabilizing learning and improving accuracy.
  - Empirically, switching to character tokens and then restricting vocabulary yield stepwise gains.

6) Training and evaluation strategy
- Training
  - Multi-task training without pretraining, using task-specific prompts in a unified loop.
  - A single set of parameters for all five tasks.
- Results
  - Against generalist baselines (Unified-IO, X-Decoder, GIT, PaLI), @MODEL achieves competitive performance on assistive-relevant tasks with far fewer parameters (~62M).
  - Against task-specific SOTAs, it approaches or surpasses several methods despite no pretraining.
- Multi-task vs single-task
  - Captioning and VQA benefit from joint training (mutual enhancement in scene understanding), while OCR may drop due to dataset imbalance.
  - Implication: Balancing strategies (re-weighting, re-sampling, curricula) are important.

7) Applicability and extensibility
- Derivable functions: Obstacle avoidance, simple navigation, indoor distance/landmark cues, sign/document reading, general scene/object understanding.
- Deployment: One-weight multi-function model suits on-device and wearable AT scenarios.

8) Limitations and future work
- No pretraining yet: Pretraining is expected to further improve performance.
- Multi-task balance: Address OCR dominance with data and loss balancing strategies.
- From lab to field: Function packaging, user testing, and robust on-device deployment are the next steps.


<br/>
# Results



1) 과업·데이터셋·평가지표
- 공통 목표: 시각장애인 보조(AT)를 위한 다과업 VLM 벤치마크(@BENCH)와 단일 가중치로 5개 과업을 동시에 처리하는 일반 AI 모델(@MODEL) 제안
- 과업/데이터/지표
  - Panoptic Segmentation (PS): ADE20K, 지표 PQ
  - Depth Estimation (DE): NYU v2(실내), 지표 RMSE(↓가 좋음)
  - OCR: 학습은 MJSynth+SynthText, 평가는 6개 데이터(IC13, IC15, IIIT5K, SVT, SVTP, CUTE), 지표 정확도(%)
  - Image Captioning (IC): VizWiz Caption(시각장애인 시점 이미지), 지표 BLEU-1, CIDEr
  - VQA: VizWiz VQA, 지표 Accuracy(%)
- 데이터 규모(표 2 요약)
  - PS: ADE20K(Train 25,574 / Val 2,000 / Test 2,000)
  - DE: NYU v2(Train 24,230 / Val 654 / Test 654)
  - OCR: MJ+ST로 학습, 테스트 합계 7,507
  - IC: VizWiz Cap(Train 23,431 / Val 7,750 / Test 8,000)
  - VQA: VizWiz VQA(Train 20,523 / Val 4,319 / Test 8,000)

2) 일반 모델들과의 비교(멀티태스크 학습 기준, 표 3)
- 설정: 사전학습·과업별 파인튜닝 없이 멀티태스크로 공동 학습. @MODEL 파라미터 62M
- 경쟁모델
  - Unified-IO (S/B/L)†: 다중모달 범용 모델(사전학습)
  - X-Decoder (T)†: PS/IC 가능(사전학습)
  - GIT†, PaLI†: 대규모 LVLM(사전학습), IC/VQA 성능 보고(일반 벤치마크 기준, @BENCH 과업과 완전 동치 아님)
- 결과(@MODEL vs 경쟁모델)
  - PS(PQ): @MODEL 38.5; X-Decoder(T)† 41.6 (사전학습·과업특화 헤드 기반)
  - DE(RMSE↓): @MODEL 0.425로 Unified-IO (S)† 0.649, (B)† 0.469보다 우수, (L)† 0.402와 근접
  - OCR(Acc avg): @MODEL 80.1 (타 일반모델들은 미보고)
  - IC(B@1/CIDEr, VizWiz Cap): @MODEL 61.0 / 52.5
  - VQA(Acc, VizWiz VQA): @MODEL 53.7로 Unified-IO (S)† 42.4, (B)† 45.8, (L)† 47.7 대비 크게 우수
- 요약: 비슷하거나 더 적은 파라미터(62M)로, @MODEL은 실내 깊이추정과 VQA에서 특히 강하고, PS/IC/OCR도 멀티태스크 설정임을 고려하면 경쟁력 있는 수치

3) 특화(Single-task) SOTA와의 비교(표 4, 단일과업 훈련)
- 설정: 각 과업을 단일 과업으로 학습(@MODEL 62M), 대표 SOTA와 비교
- PS(ADE20K, PQ)
  - MaskFormer† 34.7, Mask2Former† 39.7, kMaX-DeepLab† 41.5
  - @MODEL 39.2 → 사전학습 없는 단일 모델로 Mask2Former†에 근접
- DE(NYU v2, RMSE↓)
  - BTS† 0.392, DPT*† 0.357, GLP† 0.344
  - @MODEL 0.386 → BTS†보다 좋고 DPT†/GLP†보다는 낮음
- OCR(6개 세트 평균 정확도)
  - ASTER 86.7, SEED 88.3, MaskOCR† 93.1
  - @MODEL 90.0 → 비사전학습 계열(ASTER/SEED)보다 우수, MaskOCR†보다는 낮음
- IC(VizWiz Cap, B@1/CIDEr)
  - AoANet 65.9 / 59.7, 기준모델† 62.1 / 48.2
  - @MODEL 60.0 / 45.1 → 강력 캡셔닝 전용모델엔 미치지 못함
- VQA(VizWiz VQA, Acc)
  - 기준 47.5, S‑VQA 51.6, CS‑VQA 53.2
  - @MODEL 49.1 → 기준 대비 상회, 최신 특화모델보다는 낮음
- 요약: 단일과업 학습 시 PS/OCR에서 강하고, DE는 중상위, IC/VQA는 특화모델 대비 다소 낮음

4) 멀티태스크 vs 단일태스크(@MODEL 자체 비교)
- PS: 39.2(싱글) → 38.5(멀티)로 소폭 하락
- DE: 0.386(싱글) → 0.425(멀티)로 하락
- OCR(avg): 90.0(싱글) → 80.1(멀티)로 하락(대규모 OCR 데이터가 다른 과업 균형을 방해)
- IC(B@1): 60.0(싱글) → 61.0(멀티)로 상승
- VQA(Acc): 49.1(싱글) → 53.7(멀티)로 상승
- 해석: 장면이해 기반(IC/VQA)은 상호 보완 효과, 픽셀/수치 예측(PS/DE/OCR)은 멀티태스크 난이도로 성능 저하

5) 효율-성능 트레이드오프
- 파라미터 수 대비 성능
  - @MODEL 62M로 Unified-IO (S)† 71M보다 가볍고, DE/VQA에서 더 우수
  - Unified-IO (B)† 241M보다도 DE/VQA 우수
- 결론: 휴대형 보조기기 제약을 고려할 때, 단일 가중치·소형 모델로 다과업 처리 가능한 점이 강점

6) 설계 분석(어블레이션, 표 5·6)
- 프롬프트 기반 통합 입력(image + task prompt) vs 멀티헤드
  - 동일/유사 파라미터에서 프롬프트 기반이 PS/DE/OCR/IC 전반에서 더 좋음
  - 멀티헤드는 과업 확장 시 헤드 추가로 구조 비대화, 반면 프롬프트는 매개변수 증가 거의 없음(‘one suit of weights’에 유리)
- OCR 토크나이저/어휘
  - 서브워드+완전어휘: 평균 82.4%
  - 문자단위+완전어휘: 86.7%(+4.3)
  - 문자단위+제한어휘(영문 26자+숫자 10자 중심): 90.0%(추가 +3.3)
  - 결론: OCR은 텍스트 의미보다 글자 인식이 핵심이라 문자단위·제한어휘가 효과적

7) 정성 결과
- 하나의 입력 이미지에 대해 PS/DE/OCR/IC/VQA를 동시에 출력 가능(그림 4 예시)
- 실제 보조 응용(장애물/문 인식, 표지판 읽기, 실내 길찾기 등)으로의 확장성 강조

핵심 요약
- @BENCH: 시각장애인 사용자 연구 기반 5개 핵심 과업을 하나의 표준으로 묶고, 성능-효율 균형까지 평가
- @MODEL: 62M 소형, 단일 가중치로 다과업 동시 수행. 멀티태스크 학습에서 DE/VQA 강력, PS/IC/OCR도 경쟁적
- 프롬프트 기반 설계와 문자단위 토크나이저가 실전 성능과 경량화 모두에 기여





1) Tasks, datasets, and metrics
- Goal: Propose an AT-oriented multi-task VLM benchmark (@BENCH) and a single-weight generalist model (@MODEL) that performs 5 tasks simultaneously.
- Tasks/data/metrics
  - Panoptic Segmentation (PS): ADE20K, metric PQ
  - Depth Estimation (DE): NYU v2 (indoor), metric RMSE (lower is better)
  - OCR: train on MJSynth+SynthText; test on IC13, IC15, IIIT5K, SVT, SVTP, CUTE; metric accuracy
  - Image Captioning (IC): VizWiz Caption; metrics BLEU-1 and CIDEr
  - VQA: VizWiz VQA; metric accuracy
- Data sizes (Table 2)
  - ADE20K: 25,574/2,000/2,000
  - NYU v2: 24,230/654/654
  - OCR test total 7,507
  - VizWiz Cap: 23,431/7,750/8,000
  - VizWiz VQA: 20,523/4,319/8,000

2) Comparison with generalist models (multi-task training, Table 3)
- Setup: no pretraining or task-specific finetuning; @MODEL has 62M params
- Competitors: Unified-IO (S/B/L)†, X-Decoder (T)†, GIT†, PaLI†
- Results
  - PS (PQ): @MODEL 38.5; X-Decoder(T)† 41.6 (pretrained, task-specific heads)
  - DE (RMSE↓): @MODEL 0.425, better than Unified-IO (S)† 0.649 and (B)† 0.469, close to (L)† 0.402
  - OCR (avg acc): @MODEL 80.1 (others not reported)
  - IC (VizWiz Cap): @MODEL 61.0 B@1 / 52.5 CIDEr
  - VQA (VizWiz VQA): @MODEL 53.7, outperforming Unified-IO (S/B/L)† with large margins
- Takeaway: With only 62M params, @MODEL is particularly strong on DE and VQA; PS/IC/OCR are competitive given the multi-task setting.

3) Comparison with specialized single-task SOTA (Table 4)
- Setup: Train @MODEL per task and compare to task-specific SoTA
- PS (PQ): @MODEL 39.2 vs MaskFormer† 34.7; close to Mask2Former† 39.7; below kMaX-DeepLab† 41.5
- DE (RMSE↓): @MODEL 0.386 beats BTS† 0.392; below DPT*† 0.357 and GLP† 0.344
- OCR (avg acc): @MODEL 90.0 > ASTER 86.7 and SEED 88.3; < MaskOCR† 93.1
- IC (VizWiz Cap): @MODEL 60.0/45.1 < AoANet 65.9/59.7
- VQA (VizWiz VQA): @MODEL 49.1 > baseline 47.5; < S‑VQA 51.6 and CS‑VQA 53.2
- Takeaway: Single-task @MODEL is strong on PS/OCR, mid-high on DE, and trails top specialized IC/VQA models.

4) Multi-task vs single-task (within @MODEL)
- PS: 39.2 (single) → 38.5 (multi)
- DE: 0.386 → 0.425
- OCR avg: 90.0 → 80.1
- IC B@1: 60.0 → 61.0
- VQA Acc: 49.1 → 53.7
- Interpretation: Captioning and VQA benefit from joint training; pixel/dense tasks degrade under multi-task due to optimization balance (notably large OCR data).

5) Efficiency–performance trade-off
- @MODEL (62M) outperforms Unified-IO (S)† (71M) and even (B)† (241M) on shared tasks (DE/VQA), highlighting favorable accuracy–size trade-off for on-device AT.

6) Ablations (Tables 5–6)
- Prompt-based unified input (image + task prompt) vs multi-head
  - Prompting yields better PS/DE/OCR/IC with similar params; scales to new tasks without extra heads; aligns with “one suit of weights.”
- OCR tokenizer/vocab
  - Subword+full vocab: 82.4%
  - Char+full: 86.7% (+4.3)
  - Char+limited: 90.0% (+3.3)
  - Insight: OCR needs character recognition, not semantics; char-level with limited vocab works best.

7) Qualitative
- One image in, simultaneous outputs for PS/DE/OCR/IC/VQA; directly relatable to assistive functions like obstacle awareness, door/sign reading, and simple navigation.

Bottom line
- @BENCH delivers a user-centered, five-task standard with efficiency in mind.
- @MODEL is a compact, single-weight generalist that excels on DE/VQA and remains competitive on PS/IC/OCR, with design choices (prompting, char tokenizer) that enhance both accuracy and deployability.


<br/>
# 예제





1) 전반 개요: @BENCH와 @MODEL의 입력·출력 통합 방식
- 목적: 시각장애인(PVIs) 지원을 위해, 다중 테스크(PS, DE, OCR, IC, VQA)를 하나의 범용 VLM(@MODEL)으로 동시에 수행하고 이를 평가하는 벤치마크(@BENCH)를 제시.
- 통합 입력 형식: “이미지 + 프롬프트(prompt)”
  - 각 테스크에 맞는 텍스트 프롬프트를 함께 입력해, 모델이 어떤 작업을 수행할지 조기에 구분하도록 함.
  - 예: PS “What is the panoramic segmentation of this image?”, DE “What is the depth estimation of this image?”, OCR “What is the text on this image?”, IC “What does the image describe?”, VQA(질문 자체가 프롬프트) “Is here a dining room?”
- 통합 출력 형식: 두 가지 출력 타입
  - 픽셀 단위 출력(pixel-level): Panoptic Segmentation(PS), Depth Estimation(DE)
  - 토큰 단위 출력(token-level): OCR, Image Captioning(IC), Visual Question Answering(VQA)
- 효율성 기준: 보편적 성능 지표 외에, 효율성은 “파라미터 수”로 비교(경량·단일 가중치로 다중 테스크 수행).

2) 테스크별 훈련/테스트 데이터, 입력·출력, 메트릭, 예시
A. Panoptic Segmentation (PS)
- 목적(AT 관점): 장면 내 사물/재료(thing/stuff)를 모두 인식하고 인스턴스별로 분할하여 주변 환경을 정밀하게 파악(장애물 회피, 사물 인식 등 파생 기능에 중요).
- 데이터셋: ADE20K
  - 카테고리: 150개(365개 장면 범주, 실내/실외/도시/교외 등 다양)
  - 분할 수: Train 25,574 / Val 2,000 / Test 2,000
- 입력(훈련/테스트 공통):
  - 이미지 + 테스크 프롬프트(예: “What is the panoramic segmentation of this image?”)
- 대상 출력(정답/예측):
  - 픽셀 단위 panoptic 결과(클래스 라벨 + 인스턴스 구분)
- 평가 지표: Panoptic Quality(PQ)
- 논문 속 예시 출력(요지):
  - 같은 이미지에 대해 @MODEL이 다른 테스크와 함께 PS 결과를 동시 산출(그림 4 참조).

B. Depth Estimation (DE)
- 목적(AT 관점): 각 픽셀의 카메라 대비 거리 예측(실내 거리 추정, 문/카운터 등의 거리 안내 등).
- 데이터셋: NYU v2(실내 장면)
  - 분할 수: Train 24,230 / Val 654 / Test 654
- 입력(훈련/테스트 공통):
  - 이미지 + 테스크 프롬프트(예: “What is the depth estimation of this image?”)
- 대상 출력(정답/예측):
  - 픽셀 단위 연속값 깊이맵
- 평가 지표: RMSE
- 논문 속 예시 출력(요지):
  - 같은 이미지에 대해 DE depth map 결과를 동시 산출(그림 4 참조).

C. Optical Character Recognition (OCR)
- 목적(AT 관점): 이미지 내 텍스트를 기계가 읽을 수 있는 문자열로 인식(문서/표지판/간판/문패 등 텍스트 읽기).
- 훈련 데이터셋: 대규모 합성 데이터
  - MJSynth(MJ), SynthText(ST)
  - 총 학습 샘플 수: 15,895,356
- 평가(테스트) 데이터셋: 6개 공개 세트
  - IC13, IC15, IIIT5K, SVT, SVTP, CUTE
  - Val 7,507 / Test 7,507
- 입력(훈련/테스트 공통):
  - 이미지 + 테스크 프롬프트(예: “What is the text on this image?”)
- 대상 출력(정답/예측):
  - 텍스트 문자열(토큰 단위 생성)
  - 토크나이저 주의점: OCR은 “문자 기반(character-based) 토크나이저 + 제한된 어휘(26개 알파벳 + 10개 숫자 중심)”가 효과적이라고 보고
- 평가 지표: Accuracy(각 평가 세트 기준 정확도)
- 논문 속 예시 출력:
  - “cognac”, “jacquet” 등 단어 시퀀스(그림 4 Box Text 예시)
  - “quality / with / assessed / since1895 / star / superb” 같은 다단 텍스트 추출(그림 4)

D. Image Captioning (IC)
- 목적(AT 관점): 이미지의 전반적 내용을 사람과 유사한 문장으로 요약·서술(주변 상황 이해, 환경 묘사).
- 데이터셋: VizWiz Captioning(VizWiz Cap)
  - 시각장애인이 촬영한 이미지 기반 데이터
  - 분할 수: Train 23,431 / Val 7,750 / Test 8,000
- 입력(훈련/테스트 공통):
  - 이미지 + 테스크 프롬프트(예: “What does the image describe?”)
- 대상 출력(정답/예측):
  - 자연어 문장(토큰 단위 생성)
- 평가 지표: BLEU-1, CIDEr
- 논문 속 예시 출력:
  - “A room with table and chairs.”(그림 1)
  - “a dining room with a table and a big picture”, “a black car next to the wall”(그림 4)

E. Visual Question Answering (VQA)
- 목적(AT 관점): 사용자가 이미지에 대해 자유 질의를 던지고, 모델이 자연어로 답변(주변 이해를 질의응답 방식으로 보완).
- 데이터셋: VizWiz VQA
  - 공개 평가 스크립트 사용
  - 분할 수: Train 20,523 / Val 4,319 / Test 8,000
- 입력(훈련/테스트 공통):
  - 이미지 + 질문(질문 텍스트 자체가 프롬프트 역할, 예: “Is here a dining room?”)
- 대상 출력(정답/예측):
  - 자연어 짧은 답(토큰 단위 생성)
- 평가 지표: Accuracy
- 논문 속 예시 출력:
  - “Is here a dining room?” → “Yes”(그림 1)
  - “how many people are in this image?” → “two”(그림 4)
  - “is that a picture on the left?” → “yes”, “what is this?” → “car”(그림 4)

3) 하나의 예시 이미지에 대해 모든 테스크를 동시에 수행하는 흐름(논문 예시 기반)
- 입력:
  - 동일한 실내 이미지 1장
  - 프롬프트 묶음:
    - PS: “What is the panoramic segmentation of this image?”
    - DE: “What is the depth estimation of this image?”
    - OCR: “What is the text on this image?”
    - IC: “What does the image describe?”
    - VQA: ex) “Is here a dining room?”, ex) “how many people are in this image?”
- 출력:
  - PS: 사물/재료 전 범주에 대한 인스턴스 구분 마스크(픽셀 단위)
  - DE: 전체 픽셀의 연속값 깊이맵
  - OCR: 이미지 내 텍스트 문자열(예: “Exit”, “quality … superb” 등)
  - IC: 문장 1개(예: “This is a room with a table and chairs.”)
  - VQA: 질문별 짧은 답(예: “Yes”, “two”)
- 파생 기능 예시(논문 그림 1의 매핑):
  - 장애물 회피(Obstacle Avoidance): “There is a counter two meters ahead on the left.”
  - 단순 내비게이션(Simple Navigation): “There is a door 3 meters ahead on the right.”
  - 텍스트 인식(Text Recognition): “A sign with ‘Exit’ on the right.”
  - 장면 이해(Scene Understanding): 캡셔닝 문장
  - 객체 인식(Object Recognition): “6 chairs, 2 tables, 1 counter and 1 painting …”

4) 데이터셋 규모와 메트릭 요약(논문 보고 수치)
- PS (ADE20K): Train 25,574 / Val 2,000 / Test 2,000, 메트릭 PQ
- DE (NYU v2): Train 24,230 / Val 654 / Test 654, 메트릭 RMSE
- OCR (MJ, ST로 학습; IC13, IC15, IIIT5K, SVT, SVTP, CUTE로 평가): Train 15,895,356 / Val 7,507 / Test 7,507, 메트릭 Accuracy
- IC (VizWiz Cap): Train 23,431 / Val 7,750 / Test 8,000, 메트릭 BLEU-1, CIDEr
- VQA (VizWiz VQA): Train 20,523 / Val 4,319 / Test 8,000, 메트릭 Accuracy
- @BENCH 총합(사전 선택된 테스크의 합계): Train 15,989,114 / Val 22,230 / Test 26,161

5) 구현상 유의점(논문 직접 언급)
- 프롬프트 기반 통합 입출력: 다중 출력 헤드 대신 “이미지+프롬프트”로 테스크를 통합하면 파라미터 증가를 피하고 조기 테스크 구분으로 특징 추출을 명확히 할 수 있음.
- OCR 토크나이저: 서브워드(대규모 어휘) 대신 문자 기반(제한 어휘)이 OCR 성능을 유의미하게 향상.
- 효율성-성능 절충: 파라미터 수 대비 다중 테스크 성능을 함께 평가(경량 단일 가중치로 다중 테스크 수행 가능).



1) Overview: Unified Input/Output in @BENCH and @MODEL
- Goal: Help people with visual impairments (PVIs) by training and evaluating a single generalist VLM (@MODEL) that handles five tasks (PS, DE, OCR, IC, VQA) simultaneously on a dedicated benchmark (@BENCH).
- Unified input format: “image + prompt”
  - A task-specific textual prompt is concatenated with the image so the model distinguishes tasks early.
  - Examples: PS “What is the panoramic segmentation of this image?”, DE “What is the depth estimation of this image?”, OCR “What is the text on this image?”, IC “What does the image describe?”, VQA uses the question itself as the prompt (e.g., “Is here a dining room?”).
- Unified output types:
  - Pixel-level outputs: Panoptic Segmentation (PS) and Depth Estimation (DE)
  - Token-level outputs: OCR, Image Captioning (IC), Visual Question Answering (VQA)
- Efficiency metric: Number of parameters (besides task-specific performance metrics), emphasizing a single set of weights for all tasks.

2) Per-task training/testing data, inputs/outputs, metrics, examples
A. Panoptic Segmentation (PS)
- AT purpose: Recognize and segment all things/stuffs for precise scene understanding (supports obstacle avoidance, object recognition).
- Dataset: ADE20K
  - 150 semantic categories across 365 scenes (indoor/outdoor/urban/rural)
  - Splits: Train 25,574 / Val 2,000 / Test 2,000
- Input (train/test):
  - Image + task prompt (e.g., “What is the panoramic segmentation of this image?”)
- Output (ground truth/prediction):
  - Pixel-wise panoptic result (class + instance)
- Metric: Panoptic Quality (PQ)
- Example (from paper): PS prediction produced alongside other tasks on the same image (Fig. 4).

B. Depth Estimation (DE)
- AT purpose: Estimate per-pixel distances (e.g., indoor distance, door/counter distance).
- Dataset: NYU v2 (indoor)
  - Splits: Train 24,230 / Val 654 / Test 654
- Input (train/test):
  - Image + task prompt (e.g., “What is the depth estimation of this image?”)
- Output (ground truth/prediction):
  - Per-pixel continuous depth map
- Metric: RMSE
- Example (from paper): DE result jointly output with other tasks (Fig. 4).

C. Optical Character Recognition (OCR)
- AT purpose: Convert image text into machine-readable strings (documents, signs, door plates, posters).
- Training datasets (synthetic):
  - MJSynth (MJ), SynthText (ST)
  - Total training samples: 15,895,356
- Evaluation (test) datasets:
  - IC13, IC15, IIIT5K, SVT, SVTP, CUTE
  - Val 7,507 / Test 7,507
- Input (train/test):
  - Image + task prompt (e.g., “What is the text on this image?”)
- Output (ground truth/prediction):
  - Text string (token-level)
  - Tokenizer note: Character-based tokenizer with a limited vocabulary (26 letters + 10 digits) is more suitable than a large subword vocabulary for OCR.
- Metric: Accuracy
- Examples (from paper):
  - Recognized words like “cognac”, “jacquet” (Fig. 4)
  - Multi-line text: “quality / with / assessed / since1895 / star / superb” (Fig. 4)

D. Image Captioning (IC)
- AT purpose: Generate human-like, coherent natural language descriptions for overall scene understanding.
- Dataset: VizWiz Captioning (VizWiz Cap)
  - Images captured by PVIs
  - Splits: Train 23,431 / Val 7,750 / Test 8,000
- Input (train/test):
  - Image + task prompt (e.g., “What does the image describe?”)
- Output (ground truth/prediction):
  - A natural language sentence (token-level)
- Metrics: BLEU-1, CIDEr
- Examples (from paper):
  - “A room with table and chairs.” (Fig. 1)
  - “a dining room with a table and a big picture”, “a black car next to the wall” (Fig. 4)

E. Visual Question Answering (VQA)
- AT purpose: Users ask free-form questions about the image; the model answers in natural language.
- Dataset: VizWiz VQA
  - Uses the official VizWiz VQA evaluation scripts
  - Splits: Train 20,523 / Val 4,319 / Test 8,000
- Input (train/test):
  - Image + Question (the question serves as the task prompt, e.g., “Is here a dining room?”)
- Output (ground truth/prediction):
  - Short natural language answer (token-level)
- Metric: Accuracy
- Examples (from paper):
  - “Is here a dining room?” → “Yes” (Fig. 1)
  - “how many people are in this image?” → “two” (Fig. 4)
  - “is that a picture on the left?” → “yes”; “what is this?” → “car” (Fig. 4)

3) End-to-end example on a single image (from the paper’s illustrations)
- Inputs:
  - One indoor image
  - Prompts:
    - PS: “What is the panoramic segmentation of this image?”
    - DE: “What is the depth estimation of this image?”
    - OCR: “What is the text on this image?”
    - IC: “What does the image describe?”
    - VQA: e.g., “Is here a dining room?”, “how many people are in this image?”
- Outputs:
  - PS: Per-pixel panoptic map
  - DE: Per-pixel depth map
  - OCR: Recognized text (e.g., “Exit”, “quality … superb”)
  - IC: One-sentence caption (e.g., “This is a room with a table and chairs.”)
  - VQA: Short answer(s) (e.g., “Yes”, “two”)
- Downstream mappings (from Fig. 1):
  - Obstacle Avoidance: “There is a counter two meters ahead on the left.”
  - Simple Navigation: “There is a door 3 meters ahead on the right.”
  - Text Recognition: “A sign with ‘Exit’ on the right.”
  - Scene Understanding: Caption text
  - Object Recognition: “6 chairs, 2 tables, 1 counter and 1 painting …”

4) Dataset sizes and metrics (as reported)
- PS (ADE20K): Train 25,574 / Val 2,000 / Test 2,000, PQ
- DE (NYU v2): Train 24,230 / Val 654 / Test 654, RMSE
- OCR (train: MJ, ST; eval: IC13, IC15, IIIT5K, SVT, SVTP, CUTE): Train 15,895,356 / Val 7,507 / Test 7,507, Accuracy
- IC (VizWiz Cap): Train 23,431 / Val 7,750 / Test 8,000, BLEU-1, CIDEr
- VQA (VizWiz VQA): Train 20,523 / Val 4,319 / Test 8,000, Accuracy
- @BENCH totals (pre-selected tasks): Train 15,989,114 / Val 22,230 / Test 26,161

5) Implementation notes (explicitly stated in the paper)
- Prompt-based unification: Replacing multi-head outputs with “image+prompt” reduces parameter overhead and helps the model separate tasks earlier during feature extraction.
- OCR tokenizer: Character-level tokenizer with limited vocabulary improves OCR performance versus large subword vocabularies.
- Efficiency-performance trade-off: Emphasizes parameter count alongside task-specific metrics, enabling a single set of weights for all tasks on portable devices.

<br/>
# 요약


- 메서드: 사용자 참여형 연구로 선정한 5개 과제(파노픽 분할/깊이/텍스트 인식/OCR/캡셔닝/VQA)를 묶은 @BENCH와, X-Decoder 기반에 태스크별 프롬프트·단일 가중치(62M)·OCR 문자 기반 토크나이저를 도입해 픽셀/토큰 출력을 동시에 수행하는 @MODEL을 제안합니다. 
- 결과: 멀티태스크에서 ADE20K PQ 38.5, NYU-v2 RMSE 0.425, OCR 80.1%, VizWiz-Cap BLEU-1 61.0/CIDEr 52.5, VizWiz-VQA 53.7%를 기록하며 Unified-IO(S) 대비 깊이(0.649→0.425)·VQA(+11.3%p)에서 우수하고, 단일태스크에선 OCR 평균 90.0%로 ASTER/SEED를 능가하고 PS도 MaskFormer 대비 +4.5%p로 경쟁력을 보입니다. 
- 예시: 한 장의 이미지에서 동시에 파노픽 마스크와 깊이를 산출하고, “Exit” 등 텍스트를 읽고, 실내 장면을 캡션화하며, “사람이 몇 명인가?”→“둘” 같은 질의응답을 수행합니다.

- Method (EN): The paper introduces @BENCH, a user-driven benchmark covering five PVI-relevant tasks (PS/DE/OCR/IC/VQA), and @MODEL, an X-Decoder–based generalist using task-specific prompts, a single 62M-parameter weight set, and a character-based tokenizer for OCR to produce both pixel- and token-level outputs. 
- Results (EN): In multi-task training, @MODEL achieves ADE20K PQ 38.5, NYU-v2 RMSE 0.425, OCR 80.1%, VizWiz-Cap BLEU-1 61.0/CIDEr 52.5, and VizWiz-VQA 53.7%, outperforming Unified-IO(S) on depth (0.649→0.425) and VQA (+11.3 pp); in single-task settings it reaches 90.0% OCR average surpassing ASTER/SEED and exceeds MaskFormer by +4.5 pp while approaching Mask2Former/kMaX-DeepLab. 
- Examples (EN): With a single image, the model simultaneously outputs panoptic masks and depth, reads texts like “Exit,” generates indoor captions, and answers questions (e.g., “How many people?” → “Two”).

<br/>
# 기타



피규어(다이어그램) 요약과 인사이트
- Figure 1 (@MODEL/@BENCH 개요)
  - 결과: 단일 모델(@MODEL)로 5개 과제(파노픽 세그멘테이션, 깊이 추정, OCR, 캡셔닝, VQA)를 동시에 수행하고, 이를 통해 장애물 회피·간단 내비게이션·텍스트 인식·장면 이해·객체 인식 같은 다운스트림 보조 기능을 지원.
  - 인사이트: 보조공학(AT)에서 요구되는 다양한 시각·언어 기능을 한 번에 처리하는 “일반가(Generalist)형” 접근이 실사용 시나리오(스마트 글래스 등)에 적합함.

- Figure 2 (모델 아키텍처)
  - 결과: 이미지 인코더 + 공유되는 2개의 텍스트 인코더 + 트랜스포머 디코더(라텐트 쿼리/텍스트 쿼리) 구조. 픽셀 단위 출력(PS/DE)과 토큰 단위 출력(OCR/IC/VQA)을 모두 지원. 태스크 프롬프트로 입력을 통일(“image + prompt”).
  - 인사이트: 프롬프트 기반 통합 입력은 다양한 I/O 형식을 단일 프레임워크로 묶고, 초기 단계부터 태스크 구분 특화표현을 학습하게 해 멀티태스크 간 간섭을 줄임.

- Figure 3 (멀티태스크 패러다임 비교: 멀티헤드 vs 멀티프롬프트)
  - 결과: 멀티헤드는 태스크마다 별도 헤드(예: 헤드당 ~0.2M 파라미터)를 추가해야 하는 반면, 프롬프트는 태스크당 ~10K 수준의 미미한 추가로 충분. 전체 모델 파라미터는 유사(예: 62M)하되 구조적 비대화는 크게 줄어듦.
  - 인사이트: 보조기기의 제약(경량·단일 가중치 배포)을 고려하면 프롬프트 방식이 확장성·효율성 면에서 유리.

- Figure 4 (5개 태스크 동시 출력 사례)
  - 결과: 단일 이미지 입력으로 PS, DE, OCR(박스/텍스트), 캡션, VQA 응답을 동시에 산출.
  - 인사이트: 실제 사용에서 한 장면에 대해 다차원 정보를 즉시 제공 가능(거리·구성·텍스트·요약·질의응답). 현장성 높은 보조 기능(충돌회피·길찾기·표지판 읽기)에 실질적 이점.

- Figure 5 (싱글태스크 vs 멀티태스크 성능 상대 비교)
  - 결과: 멀티태스크 학습은 캡셔닝·VQA에서 유리, 세그멘테이션·깊이·OCR은 다소 저하.
  - 인사이트: 캡셔닝과 VQA는 상호 보완적 이해를 촉진하는 반면, 대규모 OCR 데이터가 멀티태스크 균형을 어렵게 함. 태스크 간 데이터 규모/난이도 균형 조절의 필요성.

테이블 요약과 인사이트
- Table 1 (사용자 연구 정량 결과)
  - 결과: 텍스트 인식(총점 13.43)과 객체 인식(11.57)이 최상위, 장면 인식(5.71)은 최하. 나머지 기능은 중간대. 이 결과를 반영해 PS/DE/OCR/IC/VQA 5개 태스크를 벤치마크에 채택.
  - 인사이트: 시각장애인의 실수요는 텍스트·객체 수준 이해에 강하게 집중. 벤치마크 태스크 선정의 사용자 주도성·현실 밀착성 확보.

- Table 2 (@BENCH 태스크/데이터셋/메트릭 통계)
  - 결과: PS(ADE20K, PQ), DE(NYU v2, RMSE), OCR(MJ/ST 학습, IC13/15/IIIT5K/SVT/SVTP/CUTE 평가, Acc), IC(VizWiz Cap, BLEU-1/CIDEr), VQA(VizWiz VQA, Acc). 일부 데이터셋은 val로 평가.
  - 인사이트: PVI(시각장애인) 관점의 실제성 높은 데이터 선택(VizWiz 등). 다양한 지표로 멀티모달 성능을 표준화 평가.

- Table 3 (일반가 모델 간 멀티태스크 비교)
  - 결과: 62M 파라미터의 @MODEL이, 사전학습 없이 멀티태스크 학습만으로 Unified-IO(S/B)를 DE와 VQA에서 능가(예: DE RMSE 0.425 vs 0.649/0.469; VQA 53.7%). PS는 사전학습 모델(X-Decoder 41.6 PQ) 대비 낮지만(38.5 PQ) 경쟁적.
  - 인사이트: 소형·단일 가중치 모델로도 효율-성능 균형이 우수. 사전학습/대형화 없이도 AT 맥락 핵심 과제에서 강한 성과.

- Table 4 (싱글태스크 @MODEL vs 각 태스크 특화 SoTA)
  - 결과:
    - PS: @MODEL 39.2 PQ로 MaskFormer(34.7) 상회, Mask2Former(39.7)·kMaX(41.5)에 근접.
    - DE: @MODEL 0.386 RMSE로 BTS(0.392)보다 낫고 GLP(0.344), DPT*(0.357)보단 낮음(두 모델은 사전학습/추가데이터).
    - OCR: @MODEL 평균 90.0%로 ASTER(86.7), SEED(88.3) 상회, MaskOCR(93.1)에는 미치지 못함.
    - IC: @MODEL BLEU-1/CIDEr 60.0/45.1로 AoANet(65.9/59.7)에는 못 미침.
    - VQA: @MODEL 49.1%로 베이스라인(47.5) 상회, 최고 성능(CS-VQA 53.2)에는 미달.
  - 인사이트: 단일 모델이 사전학습 없이도 여러 태스크에서 특화 모델군과 격차를 크게 좁히거나 일부 추월. 범용성-성능의 강한 절충.

- Table 5 (프롬프트 기반 통합 입력 vs 멀티헤드)
  - 결과: 프롬프트 방식이 멀티헤드 대비 모든 공통 지표에서 우위(예: PS 38.5 > 38.1, DE 0.425 < 0.432, OCR avg 80.1 > 79.4, IC B1/C 61.0/52.5 > 59.5/50.0). 파라미터 수는 유사(62–63M).
  - 인사이트: 프롬프트는 태스크별 헤드 없이도 멀티태스크 디코딩을 가능케 하고, 초기에 태스크 구분 표현을 학습시켜 효율·성능 동시 개선.

- Table 6 (OCR 토크나이저/어휘군 어블레이션)
  - 결과: 서브워드→문자 토크나이저 전환 시 +4.3%p(82.4→86.7), 문자 토크나이저에 제한 어휘군 적용 시 +3.3%p(86.7→90.0).
  - 인사이트: OCR은 의미적 서브워드보다 문자 단위·제한 어휘가 적합(26자+10숫자, 짧은 길이). 멀티태스크 내 서로 다른 언어 작업에 과제별 토크나이저/어휘 분리 설계가 유효.

보충자료(앱펜딕스/서플리) 관련 주요 포인트(본문 언급 기반)
- 구현/세부 설정: 트레이닝·토크나이저·어휘군 설계 등 세부는 보충자료에 수록.
- 멀티태스크 상호작용 분석: 캡셔닝·VQA는 상호 촉진, 대규모 OCR 데이터는 불균형 유발 가능성.
- 효율-성능 트레이드오프: 보조기기 제약 고려 시 파라미터 수를 핵심 효율 지표로 비교.
- 사용자 의견 반영: OCR(원거리 표지판·문패 등 인식 중요), 객체 인식(신뢰성·정확한 대상 지시) 요구 확인.
- 향후 과제: 사전학습 확대 시 성능 추가 향상 기대, @MODEL 기반 다기능 보조 시스템의 실제 배포 타당성.



- Figures
  - Fig. 1 (Overview): A single generalist model (@MODEL) handles PS/DE/OCR/IC/VQA simultaneously, enabling downstream assistive functions (obstacle avoidance, simple navigation, text reading, scene understanding, object recognition). Insight: This unified approach aligns well with real AT scenarios (e.g., smart glasses).
  - Fig. 2 (Architecture): Image encoder + two shared text encoders + transformer decoder with latent/text queries. Unified input as “image + task prompt,” with both pixel-level and token-level outputs. Insight: Prompt-based unification encourages early task-specific representation learning and reduces cross-task interference.
  - Fig. 3 (Multi-task paradigms): Multi-head requires per-task heads (~0.2M params each), while prompt-based adds only ~10K per task. Insight: Prompting offers superior scalability and efficiency for single-weight deployment on portable AT devices.
  - Fig. 4 (Qualitative multi-task outputs): From one image, the model outputs PS, DE, OCR (boxes/text), a caption, and VQA answers simultaneously. Insight: Practical multi-dimensional feedback per scene supports real-world AT needs.
  - Fig. 5 (Single vs multi-task training): Multi-task boosts IC and VQA but slightly degrades PS/DE/OCR. Insight: IC and VQA mutually reinforce global understanding; large OCR data can skew multi-task balance.

- Tables
  - Tab. 1 (User study): Text recognition (13.43) and object recognition (11.57) rank highest; scene recognition lowest (5.71). Selected benchmark tasks: PS/DE/OCR/IC/VQA. Insight: Strong user demand for text and object-level understanding.
  - Tab. 2 (Benchmark stats): Realistic datasets and metrics: ADE20K-PQ, NYUv2-RMSE, OCR Acc across 6 sets, VizWiz Cap BLEU-1/CIDEr, VizWiz VQA Acc. Insight: Standardized, PVI-relevant evaluation across modalities.
  - Tab. 3 (Generalist comparison, multi-task training): With 62M params and no pretraining, @MODEL surpasses Unified-IO (S/B) on DE and VQA (e.g., DE RMSE 0.425 vs 0.649/0.469; VQA 53.7%), while being competitive on PS. Insight: Strong efficiency-performance trade-off without scaling or pretraining.
  - Tab. 4 (Single-task @MODEL vs per-task SoTA): @MODEL outperforms MaskFormer on PS and approaches Mask2Former; better than BTS on DE; beats ASTER/SEED on OCR average but below MaskOCR; trails AoANet on IC and top VQA model. Insight: A single non-pretrained model narrows the gap or surpasses several per-task methods.
  - Tab. 5 (Prompt vs multi-head): Prompt-based input outperforms multi-head on shared metrics with similar parameter counts. Insight: Prompts enable better early task separation and decoding without head bloat.
  - Tab. 6 (OCR tokenizer/vocab ablation): Character-level tokenizer (+4.3%p) and limited vocabulary (+3.3%p) markedly improve OCR. Insight: OCR favors character-level, constrained vocab over semantic subwords; task-specific tokenizer/vocab is beneficial in multi-task setups.

- Supplementary (as referenced)
  - Implementation/tokenizer/vocab details in the supplement.
  - Task interplay: IC/VQA synergize; large OCR data can dominate training.
  - Efficiency metric: parameter count suits AT constraints.
  - User feedback: priority on reading distant signs/labels and reliable object identification.
  - Future: pretraining likely to boost performance further; @MODEL is a strong base for deployable multi-function AT systems.

<br/>
# refer format:



BibTeX
@inproceedings{Jiang2025ATBench,
  author    = {Xin Jiang and Junwei Zheng and Ruiping Liu and Jiahang Li and Jiaming Zhang and Sven Matthiesen and Rainer Stiefelhagen},
  title     = {ATBench: Benchmarking Vision–Language Models for Human-Centered Assistive Technology},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
  pages     = {3934--3943},
  publisher = {IEEE/CVF},
  note      = {Open Access version by CVF; final version available on IEEE Xplore}
}

Chicago 스타일


Jiang, Xin, Junwei Zheng, Ruiping Liu, Jiahang Li, Jiaming Zhang, Sven Matthiesen, and Rainer Stiefelhagen. 2025. “ATBench: Benchmarking Vision–Language Models for Human-Centered Assistive Technology.” In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 3934–3943. IEEE/CVF.
