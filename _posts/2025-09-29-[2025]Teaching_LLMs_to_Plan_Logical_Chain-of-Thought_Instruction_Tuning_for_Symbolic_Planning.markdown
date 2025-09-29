---
layout: post
title:  "[2025]Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning"
date:   2025-09-29 18:08:55 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

PDDL-INSTRUCT는 2단계 학습으로, ① Phase 1에서 정·오답 계획과 전제조건/효과 설명을 포함한 인스트럭션 튜닝, ② Phase 2에서 상태-행동-상태 CoT 체인을 생성해 VAL로 검증하고(이진/상세 피드백), 논리 사슬 품질(Reasoning loss)→최종 성능(Final loss) 순으로 최적화합니다.


짧은 요약(Abstract) :

이 논문은 LLM이 일반 자연어 추론에는 강하지만, PDDL 같은 형식적 표현을 요구하는 상징적(planning) 계획 문제에서는 여전히 취약하다는 한계를 지적합니다. 이를 해결하기 위해 저자들은 PDDL-INSTRUCT라는 새로운 인스트럭션 튜닝 프레임워크를 제안합니다. 핵심은 논리적 체인 오브 소트(Chain-of-Thought, CoT)를 활용해 모델이 각 단계에서 (1) 액션의 적용 가능성(전제조건 충족 여부), (2) 상태 전이(효과의 정확한 적용), (3) 계획의 타당성(불변식 유지, 목표 달성) 등을 명시적 추론 단계로 꼼꼼히 점검하도록 가르치는 것입니다. 이렇게 구성된 프롬프트와 지시 예시를 통해 모델이 스스로 자신의 계획을 검증·수정할 수 있는 구조적 반성 능력까지 학습하게 합니다. 여러 플래닝 도메인 실험에서 이 방식은 기존 모델 대비 크게 향상된 성능을 보였고, 표준 벤치마크에서 최대 94%의 계획 정확도(베이스라인 대비 절대 66%p 향상)에 도달했습니다. 결과적으로, 본 연구는 LLM의 일반적 추론 능력과 자동 계획에 필요한 논리적 정밀성 사이의 간극을 좁히는 유망한 방향을 제시합니다.



Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like the Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework, PDDL-INSTRUCT, designed to enhance LLMs’ symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems.


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



개요
- PDDL-INSTRUCT는 대규모 언어모델(LLM)에 상징적 계획(symbolic planning) 능력을 가르치기 위한 체계적 지시학습(instruction tuning) 프레임워크입니다.
- 핵심은 “논리적 Chain-of-Thought(CoT)”를 통해 액션 적용 가능성(전제조건), 상태 전이(효과), 불변조건/목표 충족 여부를 단계별로 명시하고, 외부 공식 검증기(VAL)로 각 단계의 논리를 실제로 검증해 피드백을 반영해 학습시키는 것입니다.
- 단순히 플랜을 생성하는 것을 넘어, 플랜이 왜(어떤 논리로) 유효한지를 모델이 설명하고, 검증기 피드백으로 그 논리를 교정하면서 점진적으로 계획 능력을 강화합니다.

모델/아키텍처
- 새로운 모델 아키텍처를 제안하는 것이 아니라, 사전학습 LLM(예: Llama-3-8B, GPT-4)을 대상으로 한 두 단계의 지시학습 기법입니다.
- 모델 외부에 PDDL 플랜 검증기 VAL(Howey et al., 2004)을 “진실의 원천”으로 두어, LLM의 자기비판(Self-reflection) 한계를 보완합니다. 즉, LLM이 만든 상태-행동-상태 추론체인을 VAL이 공식 논리로 판정하고, 그 피드백을 다시 학습 신호로 씁니다.
- 계획 최적성(최단/최소 코스트)보다는 “충분히 목표를 달성하는(satisficing)” 유효 플랜 생성에 초점을 둡니다.

입출력 정의
- 입력: (1) 사전학습 LLM M, (2) PDDL 도메인/문제/플랜(정답 및 오류 예시 포함) 데이터셋 D, (3) 플랜 검증기 V=VAL.
- 출력: 논리적 CoT 기반으로 액션 타당성 검증과 플랜 검증을 스스로 수행·설명할 수 있는 지시학습 완료 모델 Mθ*.
- 가정: 본 논문에서는 복잡한 PDDL 기능(조건부 효과, 지속시간, 파생 술어 등)을 제외한 STRIPS 계열로 제한해 논리 연쇄(CoT) 구성과 검증을 단순화합니다.

데이터셋 구성과 분할
- D는 여러 도메인(D1…Dn), 각 도메인의 다수 문제(Pi,j), 각 문제에 대해 유효/무효 플랜들과 그에 대한 설명(왜 맞고/틀린지)을 포함합니다.
- 세트 분할: Phase 1 학습용 D1, Phase 2(CoT) 학습용 D2, 최종 평가용 Dtest.
- Phase 1용 D1에는 의도적으로 잘못된 플랜도 다수 포함시켜, 전제조건 위반, 효과 전파 오류, 프레임 공리 위반, 목표 미달성 등 다양한 오류 유형을 모델이 식별·설명하도록 학습합니다.
- 문제 생성은 PlanBench와 PDDL generators 등을 활용해 자동화하며, 휴리스틱한 모델 생성 데이터보다 노이즈가 적게 관리합니다.

학습 절차 개요
- 전체 3단계 파이프라인: Phase 1 초기 지시학습 → Phase 2 CoT 지시학습(핵심 혁신) → 평가.
  
Phase 1: 초기 지시학습(Instruction Tuning)
- 입력 프롬프트에 도메인/문제/PDDL 플랜을 주고, 플랜의 각 행동에 대해 “전제조건 충족 여부, 효과 적용, 상태 변화”를 설명하도록 명시적으로 지시합니다.
- 유효 플랜과 무효 플랜을 모두 제시하며, 왜 유효/무효인지 상세 설명을 지도학습합니다.
- 목적: 플랜 구성지식과 논리적 설명 습관(말로만이 아니라 구조화된 근거 제시)을 모델에 심습니다.

Phase 2: 논리적 CoT 지시학습(핵심)
- 모델이 도메인/문제(D2)로부터 자체적으로 단계적 추론체인 ⟨s0,a1,s1⟩, ⟨s1,a2,s2⟩, …을 생성합니다. 각 단계에서:
  - 전제조건 만족 확인
  - 효과 적용으로 다음 상태 계산
  - 불변조건 유지 및 목표 진행 점검
- 외부 검증기 VAL이 각 단계의 타당성과 최종 플랜 유효성을 검증합니다.
- 피드백 유형 두 가지를 실험:
  - 이진 피드백: 유효/무효만 반환
  - 상세 피드백: 어떤 전제조건이 실패했고, 어떤 효과 적용이 잘못됐는지 등 세부 원인 반환
- 이 피드백을 학습 루프에 넣어 모델을 재훈련합니다. 이 루프의 반복 상한 η(예: 10, 15)를 두어 수렴을 유도합니다.
- 핵심 차별점: LLM이 스스로 비판하는 것이 아니라, 외부의 공식 검증으로 “반증 가능한” 논리 단계를 구성·교정한다는 점입니다.

두 단계 최적화(Phase 2 내부의 학습 설계)
- Stage 1: 추론체인 최적화
  - 목표: 모델이 생성하는 각 (상태-행동-상태) 단계가 논리적으로 맞도록 지도.
  - 손실은 “예상 상태와 모델 상태의 차이(상태 대칭차)”와 VAL의 오류 유형(전제조건 위반/효과 오류/목표 실패 등)에 따른 패널티를 결합.
  - 효과: 전제조건 점검, 효과 전파, 불변조건 유지, 논리 불일치 탐지 능력 강화.
- Stage 2: 최종 성능 최적화
  - 목표: Stage 1로 다진 논리 추론 능력이 실제 플랜 정답률 향상으로 이어지도록 조정.
  - 손실은 “완성 플랜의 유효성(VAL 라벨) 기반”으로 설계. 무효 플랜에는 고정 패널티 + 유효성 예측 BCE 손실을 부여.
- 두 단계는 서로 보완적: Stage 1이 논리적 정합성을, Stage 2가 과업 성능(플랜 유효성)을 최종 담보합니다.
- 학습률도 이원화: Stage 1(δ1) > Stage 2(δ2)보다 조금 크게 두어 논리 학습을 적극, 최종 조정을 보수적으로 수행.

프롬프트 설계
- Phase 1: 정오 플랜 모두에 대해 “각 액션의 전제조건/효과/상태전이”를 항목화해 설명하게 하는 프롬프트 사용.
- Phase 2(CoT): “현재 상태를 나열 → 후보 액션 제시 → 전제조건 체크 → 효과 적용으로 다음 상태 계산 → 목표 진행 확인”의 고정된 서술 구조를 강제. 피드백(이진/상세)을 넣은 재시도 프롬프트도 설계.

특수 기법 요약
- 외부 검증 기반의 CoT 지도: LLM의 추론체인을 원자적 논리 단계로 분해하고, 각 단계를 PDDL 검증기로 사실상 “정답 라벨링”하여 지도학습.
- 이원 손실과 이원 학습률: 추론품질(체인)과 최종 플랜 유효성을 분리 최적화.
- 오류 유형 라벨링: 전제조건 위반, 효과 적용 오류, 목표 미달 등 구체 오류를 패널티로 모델에 주입(상세 피드백의 효과가 큰 이유).
- 반복 한계 η: 피드백-재생성-재학습 루프의 계산량을 제어하며, 충분한 수렴 기회를 제공.

구현/훈련 세부
- 실험 모델: Llama-3-8B, GPT-4.
- 데이터/벤치마크: PlanBench(BlocksWorld, Mystery BlocksWorld, Logistics)에서 도메인 다양성과 난이도 스펙트럼 확보.
- 주요 하이퍼파라미터(요지):
  - Phase 1: 비교적 높은 학습률로 기초 계획지식·설명 습득(예: 2e-5), 시퀀스 길이 2048.
  - Phase 2: CoT 길이에 맞춰 컨텍스트/생성 길이 확장(4096/2048), 생성 온도 하향(0.3)로 논리 일관성 강화, η=10/15 반복.
  - Stage 1/2 별도 학습률(예: δ1=1e-5, δ2=5e-6), 상태 대칭차 기반 손실 + 오류유형 패널티, 플랜 유효성 BCE+고정 패널티.
- 계산 자원: 24GB GPU×2 기준으로 Phase 1/2 총 수십 시간 수준.

한계와 범위
- 현재는 STRIPS 하위셋 중심(조건부 효과, 시간/코스트, 파생 술어 미포함).
- 최적 계획이 아니라 만족 계획에 초점.
- 외부 검증기에 의존(자기 검증은 추후 과제). 다만 이로 인해 CoT의 비충실성 문제를 실질적으로 방지.

요약
- PDDL-INSTRUCT는 “논리적으로 검증 가능한 CoT”를 학습 신호로 삼아 LLM의 계획 능력을 크게 끌어올리는 지시학습 프레임워크입니다.
- 모델 내부의 막연한 자기반성 대신, 상태-행동-상태 단위의 논리 연쇄를 외부 검증으로 교정하는 폐루프 학습과, 추론체인/최종성능의 이원 최적화가 핵심적인 차별성입니다.
- 상세 피드백(전제조건/효과/목표 오류 원인)이 이진 피드백보다 일관되게 효과적이며, 반복 횟수 η를 늘리면 추가 향상이 관찰됩니다.



Overview
- PDDL-INSTRUCT is an instruction-tuning framework that teaches large language models (LLMs) symbolic planning via logical Chain-of-Thought (CoT).
- The core idea is to force models to reason explicitly about action preconditions, effects, and goal satisfaction at each step, and to verify every step with a formal PDDL plan validator (VAL). Validator feedback is fed back into training to correct the model’s reasoning.
- The model learns not only to produce plans but also to justify why each action is applicable and how states transition, enabling self-correction through externally validated reasoning chains.

Model/Architecture
- No new neural architecture is introduced. The method fine-tunes pretrained LLMs (e.g., Llama-3-8B, GPT-4) with a two-phase instruction-tuning process.
- An external validator (VAL) serves as an oracle to overcome LLM self-reflection limitations. The LLM generates state-action-state chains; VAL formally checks them; the feedback supervises the model.
- Focus is on satisficing planning (valid plans), not optimal planning.

I/O Definition
- Inputs: (1) a pretrained LLM M, (2) dataset D of PDDL domains/problems/plans (valid and invalid) with explanations, (3) plan validator V=VAL.
- Output: an instruction-tuned model Mθ* that can generate, verify, and explain plans via logical CoT.
- Assumption: STRIPS-style PDDL subset (no conditional effects, duratives, derived predicates, etc.) to simplify reasoning chains.

Dataset and Split
- D includes multiple domains, multiple problems per domain, and for each problem, a mix of valid/invalid plans with explanations (why right/wrong).
- Split into D1 (Phase 1 training), D2 (Phase 2 training), and Dtest (evaluation).
- D1 intentionally includes incorrect plans to cover error types: precondition violations, effect propagation mistakes, frame axiom violations, and goal failures.
- Tasks are generated via PlanBench/PDDL generators to reduce noise relative to free-form LLM-generated tasks.

Training Procedure
- Three phases: Phase 1 initial instruction tuning → Phase 2 CoT instruction tuning (key contribution) → Evaluation.
  
Phase 1: Initial Instruction Tuning
- Prompts pair PDDL domain + problem + plan with explicit, stepwise explanations: precondition checks, effect application, and resulting state.
- Both valid and invalid plans are used with detailed rationales.
- Goal: seed planning knowledge and enforce structured, logical justifications.

Phase 2: Logical CoT Instruction Tuning (core)
- The model generates its own CoT sequences ⟨s0,a1,s1⟩, ⟨s1,a2,s2⟩, … for D2 problems, explicitly checking preconditions, applying effects, tracking invariants/goal progress.
- VAL verifies each step and the whole plan.
- Two feedback modes:
  - Binary: valid/invalid
  - Detailed: explicit reasons (which preconditions failed, which effects misapplied, etc.)
- Feedback drives a training loop with a fixed iteration cap η (e.g., 10 or 15).
- Distinction: rather than self-critique, the method uses externally verifiable logical steps, turning CoT into a trustworthy training signal.

Two-Stage Optimization in Phase 2
- Stage 1: Reasoning chain optimization
  - Trains step-level correctness of (state, action, next-state).
  - Loss combines state mismatch (symmetric difference) with VAL-derived penalties for error types (precondition/effect/goal).
  - Strengthens precondition checking, effect propagation, invariant maintenance, and inconsistency detection.
- Stage 2: End-task performance optimization
  - Ensures chain-level improvements translate into higher plan validity.
  - Loss uses plan-level validity labels (fixed penalty for invalid plans plus BCE on predicted validity).
- Separate learning rates (δ1 for Stage 1 > δ2 for Stage 2) balance aggressive reasoning training with conservative end-task tuning.

Prompt Design
- Phase 1: Prompts require enumerating for each action the preconditions, their satisfaction, effects, and resulting state; include invalid plans with analyses.
- Phase 2: CoT prompts enforce a fixed structure: list current state → propose action → check preconditions → apply effects → compute next state → check goal progress; include feedback-conditioned retries (binary/detailed).

Key Techniques
- Validator-grounded CoT supervision: decompose reasoning into atomic, checkable steps; supervise with formal validation.
- Dual-objective training: separate losses for reasoning quality and final plan validity.
- Error-type supervision: explicit penalties for precondition/effect/goal errors—explains why detailed feedback outperforms binary.
- Iteration limit η: controls compute while allowing sufficient correction cycles.

Implementation/Training Details
- Models: Llama-3-8B and GPT-4.
- Data/Benchmark: PlanBench (BlocksWorld, Mystery BlocksWorld, Logistics) for varied domains and complexities.
- Hyperparameter highlights:
  - Phase 1: higher LR for foundational skill acquisition; context length ~2k.
  - Phase 2: longer context/generation for CoT (e.g., 4k/2k), lower temperature (0.3) for logical consistency, η in {10,15}.
  - Stage-specific LRs (e.g., δ1=1e-5, δ2=5e-6), state-diff + error-penalty loss for reasoning, plan-validity BCE + fixed penalty for end-task.
- Resources: fits on 2×24GB GPUs; total training time on the order of tens of hours.

Scope and Limitations
- Currently targets STRIPS subset (no conditional effects, temporal/cost features, derived predicates).
- Satisficing planning, not optimal planning.
- Relies on an external verifier; this significantly mitigates unfaithful CoT but reduces autonomy (self-verification is future work).

Bottom line
- PDDL-INSTRUCT converts CoT from a free-form narrative into a sequence of formally verifiable logical steps and uses validator feedback to supervise both reasoning chains and final plan validity.
- The combination of validator-grounded CoT and the two-stage optimization is the central methodological innovation, with detailed feedback and adequate iteration loops yielding the strongest gains.


<br/>
# Results
다음은 논문 내 결과를 바탕으로 한, 경쟁모델·테스트데이터·메트릭·비교 분석의 체계적 요약입니다.

개요
- 평가 목표: PDDL-INSTRUCT가 LLM의 기호적 계획(Planning) 능력을 얼마나 향상시키는지 검증
- 연구 질문:
  1) CoT 기반 논리적 instruction tuning이 표준 접근 대비 계획 유효성(Plan Validity)을 개선하는가?
  2) 검증 피드백의 질(이진 vs. 상세)이 성능에 어떤 영향을 주는가?
  3) 도메인 간 일반화는 어떤가?

비교 대상(경쟁모델/설정)
- Foundation 모델: Llama-3-8B, GPT-4
- 비교군:
  - Baseline: 사전학습 그대로(후처리/튜닝 없음)
  - Only Phase 1: 계획 예시와 정당화(왜 유효/무효인지)로 구성된 instruction tuning만 수행
  - Only Phase 2: CoT instruction tuning만 수행(상세 피드백 사용, 성능 좋은 설정으로 보고)
  - PDDL-INSTRUCT(제안법): Phase 1 + Phase 2 CoT instruction tuning
    - 피드백 종류: 이진(Valid/Invalid) vs 상세(어떤 전제조건 실패/효과 적용 오류 등)
    - 반복 한계(η): 10, 15

테스트 데이터(벤치마크/과제)
- 벤치마크: PlanBench
- 도메인 3종:
  - Blocksworld(고전 블록 쌓기)
  - Mystery Blocksworld(동일 술어 구조이나 이름이 난독화된 변형)
  - Logistics(트럭/비행기로 화물 이동)
- 평가 세트: 각 도메인별 100개 테스트 태스크(객체 수, 요구되는 계획 길이 등 난이도 다양)
- 검증기: VAL(외부 계획 검증기)로 모든 액션의 적용 가능성과 최종 목표 달성 여부를 공식 검증

메트릭
- Plan Accuracy: 각 테스트 태스크에서 모델이 생성한 계획이 VAL로 검증될 때 유효 판정을 받는 비율(%) 
  - 모든 단계에서 전제조건 충족, 효과 적용의 정합성, 최종 상태의 목표 충족이 필요

주요 정량 결과(표 1 요약)
- Llama-3-8B
  - Blocksworld: Baseline 28% → Only P1 78% → Only P2 72% → PDDL-INSTRUCT
    - Binary η=10: 84%
    - Detailed η=10: 89%
    - Binary η=15: 91%
    - Detailed η=15: 94%  ← 최고
  - Mystery Blocksworld: 1% → 32% → 17% → 47%/49%/59%/64%(Detailed η=15)
  - Logistics: 11% → 23% → 45% → 61%/72%/75%/79%(Detailed η=15)
  - 종합 개선: PDDL-INSTRUCT(상세, η=15)가 기본 instruction tuning 대비 평균 절대 +35%(표준편차 20%), Baseline 대비 +66%(표준편차 3%) 개선
- GPT-4
  - Blocksworld: 35% → 41% → 76% → 79%/84%/87%/91%(Detailed η=15)
  - Mystery Blocksworld: 3% → 17% → 19% → 39%/44%/54%/59%(Detailed η=15)
  - Logistics: 6% → 27% → 51% → 64%/69%/72%/78%(Detailed η=15)
  - 종합 개선: PDDL-INSTRUCT(상세, η=15)가 기본 instruction tuning 대비 평균 절대 +48%(표준편차 5%), Baseline 대비 +61%(표준편차 9%) 개선
  - 주: GPT-4 평가는 접근성 제약으로 제한된 범위에서 수행

피드백의 영향(RQ2)
- 상세 피드백이 이진 피드백보다 항상 우수
  - Llama-3, η=15 기준: Blocksworld +5%p, Mystery BW +15%p, Logistics +7%p (상세 대비 이진)
- 반복 한계(η) 증가의 효과
  - η=10 → η=15로 늘리면 전반적으로 성능 향상
  - 평균 개선폭: 상세 피드백 +4.3%p, 이진 피드백 +3.3%p
  - 더 많은 반복이 수렴에 도움(추가 실험으로 적정 η 탐색 여지)

도메인 일반화(RQ3)
- 난이도: Blocksworld > Logistics > Mystery Blocksworld 순으로 성능이 낮아짐
  - Llama-3(상세, η=15): 94%(BW), 79%(Logistics), 64%(Mystery BW)
- 가장 어려운 Mystery Blocksworld에서 상대적 개선이 가장 큼
  - Llama-3: 1% → 64%로 64배 향상
- 결론: 논리적 CoT + 검증 피드백 학습이 다양한 도메인에 일반적으로 전이됨

모듈 기여도(어블레이션, 표 4 요약: Llama-3)
- Phase 1만: 성능 크게 상승(예: BW 78%)
- Phase 2만: 단독으로도 향상 가능하나 Phase 1 대비 낮음(예: BW 72%)
- Phase 1 + Binary(η=15): 추가 향상(예: BW 89%)
- Phase 1 + Detailed(η=15): 최고 성능(예: BW 94%, Mystery 64%, Logistics 79%)
- 결론: 2단계(Phase 1→Phase 2)의 결합, 그리고 상세 피드백이 핵심

오류 유형 분석(표 5 요약: Llama-3, 상세+η=15)
- 실패율(총): Blocksworld 6.0%, Logistics 21.0%, Mystery BW 36.0%
- 주요 실패 유형 분포
  - Mystery BW에서 전제조건 위반(8.7%), 효과 적용 오류(12.4%), 목표 미달(9.2%), 잘못된 액션 순서(5.7%)가 두드러짐
  - 난독화된 술어로 인해 전제/효과 추적의 논리적 오류가 증가
- 시사점: 상세 피드백은 특히 전제조건/효과 추적 오류 교정에 유리

핵심 결론
- 논리적 CoT instruction tuning과 외부 검증기(VAL) 기반 피드백 루프를 결합한 PDDL-INSTRUCT가 계획 유효성에서 큰 폭의 향상을 달성
- 상세 피드백과 충분한 반복(η)이 추가 이득 제공
- 도메인 난이도에 따라 절대 성능은 달라도 상대적 개선은 일관되게 큼
- 여전히 100%에 도달하지는 않으나, 실용적으로 신뢰도 높은 계획을 생성하는 방향성을 입증

제약 및 주석
- GPT-4 실험은 접근성 제약으로 제한적
- 본 연구는 만족해(Optimal이 아닌 Satisficing) 계획을 목표로 함
- VAL 검증을 통해 CoT의 비충실성 문제를 완화




Overview
- Goal: Evaluate how much PDDL-INSTRUCT improves LLMs’ symbolic planning capabilities.
- Research Questions:
  1) Does logical CoT instruction tuning improve plan validity over standard approaches?
  2) How does the feedback type (binary vs. detailed) affect performance?
  3) How well does the approach generalize across domains?

Competitors/Settings
- Foundation models: Llama-3-8B, GPT-4
- Compared methods:
  - Baseline: off-the-shelf models
  - Only Phase 1: instruction tuning with plan examples and explanations of validity/invalidity
  - Only Phase 2: CoT instruction tuning alone (with detailed feedback in the ablation as the best-performing variant)
  - PDDL-INSTRUCT (proposed): Phase 1 + Phase 2 CoT instruction tuning
    - Feedback types: binary vs. detailed
    - Iteration limit η: 10, 15

Test Data/Benchmarks
- Benchmark: PlanBench
- Domains:
  - Blocksworld
  - Mystery Blocksworld (same predicates but obfuscated names)
  - Logistics (truck/airplane transport)
- Test set: 100 tasks per domain (varying object counts and plan lengths)
- Verifier: VAL for step-wise applicability and final goal satisfaction

Metric
- Plan Accuracy: percentage of tasks for which the generated plan is valid per VAL
  - Requires correct preconditions/effects at each step and goal satisfaction at the end

Key Quantitative Results (Table 1)
- Llama-3-8B
  - Blocksworld: 28% (Baseline) → 78% (Only P1) → 72% (Only P2) → 84%/89%/91%/94% with PDDL-INSTRUCT
    - Mapping: Binary η=10 (84%), Detailed η=10 (89%), Binary η=15 (91%), Detailed η=15 (94%, best)
  - Mystery Blocksworld: 1% → 32% → 17% → 47%/49%/59%/64% (Detailed η=15)
  - Logistics: 11% → 23% → 45% → 61%/72%/75%/79% (Detailed η=15)
  - Overall gains: Detailed η=15 improves over basic instruction tuning by +35% absolute on average (SD=20%), and over Baseline by +66% (SD=3%)
- GPT-4
  - Blocksworld: 35% → 41% → 76% → 79%/84%/87%/91% (Detailed η=15)
  - Mystery Blocksworld: 3% → 17% → 19% → 39%/44%/54%/59% (Detailed η=15)
  - Logistics: 6% → 27% → 51% → 64%/69%/72%/78% (Detailed η=15)
  - Overall gains: Detailed η=15 improves over basic instruction tuning by +48% absolute on average (SD=5%), and over Baseline by +61% (SD=9%)
  - Note: GPT-4 tests were limited due to access constraints

Impact of Feedback (RQ2)
- Detailed feedback consistently outperforms binary feedback across all domains and models.
  - For Llama-3 with η=15: +5%p (Blocksworld), +15%p (Mystery), +7%p (Logistics) over binary
- Iteration limit η:
  - Increasing from 10 to 15 improves performance consistently
  - Average gain: +4.3%p (detailed), +3.3%p (binary)

Cross-Domain Generalization (RQ3)
- Absolute performance: highest in Blocksworld, then Logistics, lowest in Mystery Blocksworld
  - Llama-3 (Detailed, η=15): 94% (BW), 79% (Logistics), 64% (Mystery BW)
- Largest relative improvements occur in the hardest domain
  - Llama-3: Mystery BW from 1% to 64% (64×)
- Conclusion: logical CoT with verifier-guided feedback generalizes across domains

Ablation Insights (Table 4, Llama-3)
- Phase 1 alone: large gains (e.g., 78% in BW)
- Phase 2 alone: improves but weaker than Phase 1 alone (e.g., 72% in BW)
- Phase 1 + Binary (η=15): further gains (e.g., 89% in BW)
- Phase 1 + Detailed (η=15): best overall (e.g., 94% BW, 64% Mystery, 79% Logistics)
- Takeaway: the two-phase combination plus detailed feedback is key

Error Analysis (Table 5, Llama-3 with Detailed+η=15)
- Total failure rates: 6.0% (BW), 21.0% (Logistics), 36.0% (Mystery BW)
- Error types:
  - Mystery BW shows higher rates of precondition violations (8.7%), incorrect effect application (12.4%), goal not achieved (9.2%), and invalid action sequence (5.7%)
  - Obfuscated predicates increase logical tracking difficulty
- Implication: detailed feedback is especially beneficial for correcting precondition/effect reasoning errors

Bottom Line
- PDDL-INSTRUCT, combining logical CoT and external validator feedback, substantially boosts plan validity
- Detailed feedback and sufficient iteration counts yield additional gains
- Despite domain difficulty variations, relative improvements are large and consistent
- While not achieving 100% across all domains, the method reliably produces valid plans and mitigates unfaithful CoT via formal verification

Notes/Constraints
- GPT-4 experiments were limited in scope due to access
- Focus is on satisficing plans, not optimality
- External VAL verification helps counter unfaithful CoT reasoning


<br/>
# 예제
아래 내용은 논문 “Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning”에서 제시된 예시(트레이닝/테스트 데이터의 구체적 인풋·아웃풋, 테스크 정의)를 바탕으로, 실제로 어떤 데이터가 모델에 들어가고 어떤 형식의 출력을 학습·생성하는지 길고 체계적으로 정리한 것입니다. 


1) 전체 과업 개요와 데이터 분할
- 과업(테스크):
  - 기호적 계획(PDDL) 도메인과 문제를 입력으로 받아, 유효한 계획(행동 시퀀스)을 생성하고 각 단계에서의 전제조건(preconditions)·효과(effects)·상태전이(state transitions)를 논리적 Chain-of-Thought(CoT)로 검증·설명하도록 모델을 가르치는 것.
- 데이터 분할:
  - D1 (Phase 1 학습용): 도메인·문제·계획(정답/오답 혼합)과 그 타당성 설명이 포함됨.
  - D2 (Phase 2 학습용): 도메인·문제 입력에 대해 모델이 스스로 CoT로 상태-행동-상태(⟨si−1, ai, si⟩)를 생성. 외부 검증기(VAL) 피드백(이진 또는 상세)을 받아 CoT 개선.
  - Dtest (평가용): 새로운 도메인·문제에 대해 모델이 CoT로 계획을 생성. VAL은 평가에만 사용(피드백 미반영).

2) 입력/출력 스키마(요약)
- Phase 1(초기 인스트럭션 튜닝) 샘플 단위:
  - 입력: [INSTRUCTION], [DOMAIN(PDDL)], [PROBLEM(PDDL)], [PLAN(행동열)]
  - 출력: [EXPECTED RESPONSE] = 유효/무효 판단 + 각 액션의 전제조건 검증, 효과 적용, 단계별 결과 상태(또는 오류 지점) 설명
- Phase 2(CoT 인스트럭션 튜닝) 샘플 단위:
  - Stage A(생성): 
    - 입력: [INSTRUCTION(CoT 강조)], [DOMAIN], [PROBLEM]
    - 출력: 초기 상태 요약, 각 단계 [si−1, ai, si]에 대한 전제조건 체크·효과 적용·결과 상태, 목표 달성 확인, 최종 계획
  - Stage B(피드백 반영):
    - 입력: [DOMAIN], [PROBLEM], [PREVIOUS PLAN or CoT], [VAL FEEDBACK(이진 또는 상세: 전제조건 실패 항목, 잘못된 효과 등)]
    - 출력: 피드백을 반영해 수정된 CoT와 계획(검증 가능한 논리적 단계 포함)
- 테스트(평가) 샘플 단위:
  - 입력: [DOMAIN], [PROBLEM]
  - 출력: [CoT로 생성된 ⟨s0, a1, s1⟩, …, ⟨sn−1, an, sn⟩]와 [FINAL PLAN]
  - 채점: 외부 검증기(VAL)로 계획 유효성 판단(모델에는 피드백 미제공)

3) 구체 예시: Blocksworld 도메인
아래는 논문 부록 B의 실제 프롬프트/응답 형식을 그대로 반영한 예시들입니다.

도메인(PDDL; 공통)
(define (domain blocksworld)
  (:requirements :strips)
  (:predicates
    (on ?x ?y) (ontable ?x) (clear ?x) (handempty) (holding ?x))
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (handempty))
                 (not (clear ?x)) (holding ?x)))
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (handempty)
                 (ontable ?x) (clear ?x)))
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y))
                 (handempty) (on ?x ?y) (clear ?x)))
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (not (on ?x ?y)) (not (handempty))
                 (holding ?x) (clear ?y))))

예시 1 — Phase 1(정답 계획 학습 샘플)
- 입력:
  - 문제(PDDL):
    (define (problem bw-simple)
      (:domain blocksworld)
      (:objects a b c)
      (:init (ontable a) (ontable b) (ontable c)
             (clear a) (clear b) (clear c) (handempty))
      (:goal (and (on a b) (on b c))))
  - 계획(정답):
    1. (pick-up b)
    2. (stack b c)
    3. (pick-up a)
    4. (stack a b)
- 출력(요구되는 이상적 응답):
  - “VALID” 선언 + 각 액션에 대해
    - 전제조건이 초기/중간 상태에서 만족되는지 설명
    - 효과 적용 후 결과 상태를 나열
  - 예: (pick-up b) 후 상태: (ontable a) (ontable c) (clear a) (clear c) (holding b)
  - 마지막에 목표 (on a b) ∧ (on b c) 달성 확인

예시 2 — Phase 1(오답 계획 학습 샘플)
- 입력:
  - 문제: 예시 1과 동일
  - 계획(오답):
    1. (stack a b)
    2. (stack b c)
- 출력(요구되는 이상적 응답):
  - “INVALID” 선언 + 오류 지점 상세
    - (stack a b)의 전제조건 (holding a) 불만족(초기엔 handempty이고 a는 ontable)
  - 수정 제안: (pick-up a) 선행 후 (stack a b), 이어서 b도 pick-up 후 stack 등

예시 3 — Phase 2(CoT 생성; 피드백 전) 
- 입력:
  - 문제(PDDL):
    (define (problem bw-tower)
      (:domain blocksworld)
      (:objects a b c)
      (:init (on c b) (ontable b) (ontable a)
             (clear c) (clear a) (handempty))
      (:goal (and (on a c) (on c b))))
- 출력(요구되는 CoT):
  - 초기 상태 요약
  - 단계별:
    - Step 1: a1=(pick-up a). 전제조건 체크(모두 TRUE) → 효과 적용 → s1
    - Step 2: a2=(stack a c). 전제조건 체크(holding a, clear c) → 효과 → s2
  - 목표 확인: (on a c), (on c b) 만족
  - 최종 계획: (pick-up a), (stack a c)

예시 4 — Phase 2(CoT + 이진 피드백; 오답 수정)
- 입력:
  - 문제: 예시 1과 동일(bw-simple)
  - 이전 계획(모델 출력): 
    - Step 1: (pick-up b) – VALID
    - Step 2: (stack b c) – VALID
    - Step 3: (stack a b) – INVALID
  - VAL 피드백(이진): 3번째에서 전제조건 불만족으로 실패
- 출력(요구되는 CoT):
  - 피드백을 반영해 CoT 수정
  - 수정된 계획 예: (pick-up b), (stack b c), (pick-up a), (stack a b)
  - 각 단계 전제조건·효과·결과 상태 재서술

예시 5 — Phase 2(CoT + 상세 피드백; 오답 수정)
- 입력:
  - 문제: 예시 1과 동일
  - 이전 계획: 예시 4와 동일
  - VAL 상세 피드백: “(stack a b) 시 (holding a)가 false” → “(holding a)를 true로 만들라”라는 수리 조언
- 출력(요구되는 CoT):
  - (pick-up a) 추가로 전제조건 보강
  - 수정 계획과 단계별 논리 검증

예시 6 — Phase 2(CoT + 피드백; 정답 확인)
- 입력:
  - 문제: 예시 1과 동일
  - 이전 계획(정답): (pick-up b) → (stack b c) → (pick-up a) → (stack a b)
  - VAL 피드백: 모든 단계에서 삭제/추가된 원자 술어를 나열하며 성공 판정
- 출력(요구되는 CoT):
  - 각 단계에 대한 유효성 설명 + 최종 목표 달성 재확인

4) 테스트(평가) 데이터의 구체 흐름
- 입력:
  - [DOMAIN], [PROBLEM]만 제공(새로운 도메인/문제 포함 가능)
- 모델 출력:
  - CoT 형태의 [⟨s0, a1, s1⟩, …, ⟨sn−1, an, sn⟩]와 [FINAL PLAN]
  - 각 단계 전제조건·효과·상태전이를 명시적으로 적는 것이 권장
- 평가:
  - 외부 검증기 VAL이 계획의 유효성(각 액션 적용 가능 여부 + 최종 목표 만족)을 자동 확인
  - 이때 피드백은 모델로 돌아가지 않으며, 평가지표(Plan Accuracy)만 집계

5) 도메인/테스크 범위
- 사용 도메인(PlanBench): Blocksworld, Mystery Blocksworld(술어명이 난독화), Logistics
- 테스크:
  - 계획 생성(Plan generation)
  - 액션 적용 가능성 검증(Action applicability verification)
  - 계획의 단계별 논리 검증(Plan verification via CoT)
  - 이유 설명(Reasoning transparency)

6) 데이터 구조(요약)
- D1(Phase 1):
  - (Domain file, Problem file, Plan file, Explanation) 페어들
  - Plan에는 정답/오답 혼합, 각 케이스에 대한 전제조건/효과/오류 설명
- D2(Phase 2):
  - (Domain file, Problem file)를 입력으로 모델이 CoT 생성
  - 생성된 (⟨s, a, s′⟩ 체인, 최종 Plan)에 대해 VAL 피드백(이진/상세)을 수집
  - 이 데이터를 바탕으로 2단계 최적화(Reasoning Chain → Final Performance)를 학습
- Dtest:
  - (Domain file, Problem file)만 제공, 모델 CoT/Plan을 VAL로 채점





1) Task overview and data splits
- Task:
  - Given a symbolic planning domain/problem in PDDL, the model must generate a valid plan and, via logical Chain-of-Thought (CoT), verify action preconditions, apply effects, track state transitions, and justify plan validity.
- Data splits:
  - D1 (Phase 1 training): Contains domain/problem/plan (mix of valid/invalid) along with detailed explanations of correctness or errors.
  - D2 (Phase 2 training): Model generates CoT state-action-state triplets; an external validator (VAL) provides binary or detailed feedback that is used to improve CoT and planning.
  - Dtest (Evaluation): Unseen domain/problems; model outputs CoT and plan; VAL is used only to score, not to feedback into the model.

2) Input/Output schema (summary)
- Phase 1 (initial instruction tuning) sample:
  - Input: [INSTRUCTION], [DOMAIN (PDDL)], [PROBLEM (PDDL)], [PLAN (action list)]
  - Output: [EXPECTED RESPONSE] = VALID/INVALID judgment plus per-action analysis: precondition checks, effect application, resulting state after each step, and final goal satisfaction (or error localization and repair hint).
- Phase 2 (CoT instruction tuning) sample:
  - Stage A (generation):
    - Input: [INSTRUCTION emphasizing CoT], [DOMAIN], [PROBLEM]
    - Output: Initial state analysis; for each step [si−1, ai, si] do precondition check, apply effects, produce next state; goal check; final plan.
  - Stage B (feedback integration):
    - Input: [DOMAIN], [PROBLEM], [PREVIOUS PLAN or CoT], [VAL FEEDBACK: binary or detailed (which preconditions failed, which effects were wrong, etc.)]
    - Output: Corrected CoT and plan incorporating feedback with explicit logical steps.
- Test (evaluation) sample:
  - Input: [DOMAIN], [PROBLEM]
  - Output: [CoT: ⟨s0, a1, s1⟩, …, ⟨sn−1, an, sn⟩] and [FINAL PLAN]
  - Scoring: VAL verifies applicability at each step and goal satisfaction. No feedback is returned to the model.

3) Concrete examples: Blocksworld domain
Domain (PDDL; shared)
(define (domain blocksworld)
  (:requirements :strips)
  (:predicates
    (on ?x ?y) (ontable ?x) (clear ?x) (handempty) (holding ?x))
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (handempty))
                 (not (clear ?x)) (holding ?x)))
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (handempty)
                 (ontable ?x) (clear ?x)))
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y))
                 (handempty) (on ?x ?y) (clear ?x)))
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (not (on ?x ?y)) (not (handempty))
                 (holding ?x) (clear ?y))))

Example 1 — Phase 1 (valid plan training sample)
- Input:
  - Problem:
    (define (problem bw-simple)
      (:domain blocksworld)
      (:objects a b c)
      (:init (ontable a) (ontable b) (ontable c)
             (clear a) (clear b) (clear c) (handempty))
      (:goal (and (on a b) (on b c))))
  - Plan (valid):
    1. (pick-up b)
    2. (stack b c)
    3. (pick-up a)
    4. (stack a b)
- Output (expected):
  - Declare “VALID,” and for each action:
    - Check preconditions against the current state,
    - Apply effects,
    - List the resulting state.
  - Finally verify the goal (on a b) ∧ (on b c) is satisfied.

Example 2 — Phase 1 (invalid plan training sample)
- Input:
  - Problem: same as Example 1
  - Plan (invalid):
    1. (stack a b)
    2. (stack b c)
- Output (expected):
  - Declare “INVALID” with error localization:
    - (stack a b) fails because (holding a) is false in the initial state.
  - Provide a repair suggestion (e.g., insert (pick-up a) before stacking).

Example 3 — Phase 2 (CoT generation; before feedback)
- Input:
  - Problem:
    (define (problem bw-tower)
      (:domain blocksworld)
      (:objects a b c)
      (:init (on c b) (ontable b) (ontable a)
             (clear c) (clear a) (handempty))
      (:goal (and (on a c) (on c b))))
- Output (expected CoT):
  - Initial state summary.
  - Step-by-step:
    - a1=(pick-up a): preconditions true → apply effects → s1
    - a2=(stack a c): preconditions true → apply effects → s2
  - Goal check: both (on a c) and (on c b) hold
  - Final plan: (pick-up a), (stack a c)

Example 4 — Phase 2 (CoT + binary feedback; correcting an invalid plan)
- Input:
  - Problem: same as Example 1
  - Previous plan (model output):
    - Step 1: (pick-up b) – VALID
    - Step 2: (stack b c) – VALID
    - Step 3: (stack a b) – INVALID
  - VAL feedback (binary): failure at step 3 due to unsatisfied preconditions
- Output (expected CoT):
  - Incorporate feedback by inserting (pick-up a) before stacking.
  - Revised plan: (pick-up b), (stack b c), (pick-up a), (stack a b)
  - Redo per-step precondition/effect reasoning.

Example 5 — Phase 2 (CoT + detailed feedback; repair using specific hints)
- Input:
  - Problem: same as Example 1
  - Previous plan: same as Example 4
  - VAL detailed feedback: “(stack a b) failed; make (holding a) true”
- Output (expected CoT):
  - Insert (pick-up a), then (stack a b)
  - Provide step-wise logical justifications.

Example 6 — Phase 2 (CoT + feedback; validated correct plan)
- Input:
  - Problem: same as Example 1
  - Previous plan (valid): (pick-up b) → (stack b c) → (pick-up a) → (stack a b)
  - VAL feedback: success; detailed add/delete predicate logs per step
- Output (expected CoT):
  - Confirm each step’s applicability and final goal achievement.

4) Test-time flow (evaluation set)
- Input:
  - [DOMAIN], [PROBLEM] only (may include unseen domains/problems)
- Model output:
  - CoT with state-action-state triplets and final plan
- Scoring:
  - VAL validates applicability and goal satisfaction; no feedback is returned to the model; only plan accuracy is computed.

5) Domains and tasks
- Domains (from PlanBench): Blocksworld, Mystery Blocksworld (with obfuscated predicate names), Logistics.
- Tasks:
  - Plan generation,
  - Action applicability verification,
  - Plan verification via CoT,
  - Reasoning transparency.

6) Data structure (summary)
- D1 (Phase 1):
  - Tuples of (Domain file, Problem file, Plan file, Explanation), mixing valid and invalid plans with step-wise justifications or error analyses.
- D2 (Phase 2):
  - Given (Domain, Problem), the model generates CoT plans.
  - VAL provides binary/detailed feedback for each step and for the whole plan.
  - Two-stage optimization learns (1) reasoning chains and (2) end-task planning performance.
- Dtest:
  - (Domain, Problem) only; model outputs CoT/plan; VAL scores plan validity.

위 예시들은 논문 본문 및 부록(B.1–B.2)의 실제 프롬프트·응답 형식을 충실히 반영하여, 학습/테스트 시 어떤 입력이 들어가고 어떤 출력이 기대되는지, 그리고 어떤 피드백이 어떻게 모델 학습에 사용되는지(특히 Phase 2에서) 구체적으로 보여줍니다.

<br/>
# 요약


메서드: PDDL-INSTRUCT는 2단계 학습으로, ① Phase 1에서 정·오답 계획과 전제조건/효과 설명을 포함한 인스트럭션 튜닝, ② Phase 2에서 상태-행동-상태 CoT 체인을 생성해 VAL로 검증하고(이진/상세 피드백), 논리 사슬 품질(Reasoning loss)→최종 성능(Final loss) 순으로 최적화합니다. 
결과: Llama-3-8B에서 상세 피드백·η=15 설정으로 Blocksworld 94%, Mystery BW 64%, Logistics 79%를 달성(GPT-4: 91/59/78), 기본 대비 평균 절대 향상 최대 66%이며 상세 피드백이 이진 피드백보다 일관되게 우수했습니다. 
예시: Blocksworld 도메인에 대해 (pick-up, stack) 단계별로 전제조건 확인·효과 적용·상태 갱신을 서술하는 CoT와 VAL의 오류 지시(예: 미충족 전제)를 활용한 수정 계획 예시를 제공합니다.



Method: PDDL-INSTRUCT trains in two phases: (1) Phase 1 instruction tuning with both valid/invalid plans plus explanations of preconditions/effects, and (2) Phase 2 CoT tuning that generates state–action–state chains verified by VAL using binary or detailed feedback, optimized via reasoning-chain loss then final-task loss. 
Results: With Llama-3-8B and detailed feedback at η=15, plan accuracy reaches 94% (Blocksworld), 64% (Mystery BW), 79% (Logistics) and GPT-4 achieves 91/59/78, yielding up to 66% absolute improvement over baselines, with detailed feedback consistently outperforming binary. 
Examples: The paper provides Blocksworld prompts showing stepwise CoT (checking preconditions, applying effects, tracking states) and VAL feedback that pinpoints errors (e.g., unmet preconditions) to produce repaired plans.

<br/>
# 기타
아래는 본 논문의 “기타(다이어그램/피규어, 테이블, 어펜딕스)”에 해당하는 핵심 결과와 인사이트 정리입니다.

[피규어]
- Figure 1: PDDL-INSTRUCT 파이프라인(Phase 1: 초기 Instruction Tuning → Phase 2: CoT Instruction Tuning → 평가)
  - 핵심 아이디어: Phase 2에서 LLM의 체인-오브-쏘트(CoT)를 상태-행동-상태 ⟨si−1, ai, si⟩ 단위로 쪼개고, 외부 검증기(VAL)로 각 전이를 공식 검증. 이 피드백을 다시 Instruction Tuning에 반영.
  - 인사이트:
    - “논리적 일관성(logical coherence)”을 강하게 보장(모든 전이가 전제-효과 규칙에 부합)하면서, LLM의 자기비판(Self-reflection) 한계를 외부 검증으로 보완.
    - 최적성(progressive refinement)은 목표가 아님(만족 가능한 계획(satisficing)에 집중). 그럼에도 도메인 일반화와 계획 검증 능력이 크게 향상.

[테이블]
- Table 1: 주 결과(계획 정확도, 100개 테스트/도메인)
  - Llama-3, Detailed feedback, η=15: Blocksworld 94%, Mystery-BW 64%, Logistics 79%.
  - GPT-4, Detailed feedback, η=15: Blocksworld 91%, Mystery-BW 59%, Logistics 78%.
  - 인사이트:
    - PDDL-INSTRUCT(특히 Detailed feedback)이 일관되게 최고 성능. Llama-3 기준, 기본 Instruction Tuning 대비 평균 절대 +35%p(표준편차 20), 베이스라인 대비 +66%p(표준편차 3).
    - Detailed > Binary 피드백: Llama-3(η=15)에서 Blocksworld +5%p, Mystery-BW +15%p, Logistics +7%p 차이.
    - η(피드백 반복) 증가 효과: 10→15로 늘리면 Detailed 평균 +4.3%p, Binary +3.3%p 추가 향상. 더 많은 검증 루프가 안정적 수렴에 기여.
    - Phase 1만/Phase 2만 vs 결합: 단일 단계보다 결합(Phase 1+Phase 2)이 항상 더 강함(특히 난도 높은 Mystery/Logistics에서 격차 큼).

- Table 2: 하이퍼파라미터(Phase 1 vs Phase 2)
  - 인사이트:
    - Phase 2는 긴 CoT를 담기 위해 시퀀스 길이(4096)와 생성 길이(2048) 확대, 배치 축소(8), 온도 하향(0.3)으로 결정론적 논리 전개 유도.
    - 2단 학습률(δ1>δ2) 설계는 “추론(논리 전이) 품질 향상”과 “최종 성능 최적화”의 간섭을 줄이고 균형 있게 개선하도록 고안.

- Table 3: 자원/시간
  - 2×RTX 3080(24GB)로 Phase 1 약 12h, Phase 2 약 18h, 총 30h. 추론 약 2.3s/문제.
  - 인사이트: 중급 GPU 2장으로 재현 가능한 비용/시간대. 실용적 파이프라인.

- Table 4: Ablation(Llama-3)
  - P2 only(상세 피드백, η=15)가 Logistics에서는 P1 only보다 큼(45% vs 23%)이지만, Blocksworld/Mystery-BW에서는 P1 only가 우세.
  - 최종적으로 P1 + Detailed(η=15)가 모든 도메인에서 최고(94/64/79%).
  - 인사이트: Phase 1은 도메인/문법 이해와 오류 설명능력을 구성, Phase 2는 논리적 검증 체인을 강화. 둘의 결합이 시너지를 만들어 가장 강한 일반화/정확도 달성.

- Table 5: 오류 유형 분석(Phase 1+Detailed, η=15, Llama-3)
  - 총 실패율: Blocksworld 6%, Mystery-BW 36%, Logistics 21%.
  - Mystery-BW는 “효과 적용 오류(12.4%)”와 “전제 위반(8.7%)”이 두드러짐(의미가 가려진 술어로 인한 의미 grounding 난이도).
  - 인사이트: CoT가 전제-효과 추론을 학습했음에도, 술어 의미가 난독화된 환경에서 여전히 의미/매핑 일반화가 어려움. 향후 술어 정합/grounding 강화가 고점 돌파의 열쇠.

[어펜딕스]
- Appendix A: 실험 세팅, 손실함수, 알고리즘
  - 손실 설계:
    - Lreasoning: 상태 대칭차(d_state)와 VAL 피드백(전제/효과/목표 실패 가중 패널티)로 단계별 전이의 논리 정확도를 직접 최적화.
    - Lfinal: 전체 계획의 유효성(breakdown 포함)을 벌점(β)과 BCE로 학습해 “추론 품질 → 최종 성공률” 전이를 보장.
  - 알고리즘 2단계(추론 사슬 → 최종 성능) 최적화는 “논리 기초 형성” 후 “과제 성능”을 미세하게 끌어올리는 구조.
  - 인사이트: 외부 검증 기반의 미시적 피드백이 CoT의 신뢰성을 담보. 손실의 분리로 역전파 신호 충돌을 완화하며 안정 수렴.

- Appendix B: 프롬프트 예시
  - Phase 1: “행동 타당성 이유”를 전제/효과/상태 변화로 설명하는 지시형 프롬프트(정/오 예시 제공).
  - Phase 2: 상태-행동-상태 CoT 생성 및 VAL 로그(추가/삭제된 술어)까지 포함한 상세 피드백 통합 프롬프트.
  - 인사이트: 모델에 “상태 저널링(state journaling)” 습관을 학습시키고, 검증기 로그를 노출함으로써 전이 감각을 정교화. Detailed 피드백의 성능 우위는 이 구조적 신호 덕분.

- Appendix C: 확장 결과
  - Ablation 재확인: Detailed 피드백과 Phase 결합이 결정적.
  - 오류 분석 재확인: Mystery-BW의 의미 난이도가 현 성능 병목. 전제 체크보다 “효과 전파/적용”이 상대적으로 더 취약한 패턴 확인.

종합 인사이트
- 외부 검증과 결합된 “논리적 CoT”는 계획(valid plan)의 신뢰성을 크게 높임. Detailed 피드백과 충분한 반복(η)을 주는 것이 핵심.
- Phase 1(개념/문법/오류설명)과 Phase 2(검증 중심 CoT)의 역할 분담이 시너지 창출.
- 난독화/은닉 의미 도메인(Mystery-BW)에서는 여전히 의미 grounding이 병목. 향후 술어 의미/정합 강화, 최적 계획(quality) 지향, 고급 PDDL(조건부 효과, 비용/시간) 확장이 유효한 다음 단계.


[English Version]

Figures
- Figure 1: PDDL-INSTRUCT pipeline (Phase 1: Initial Instruction Tuning → Phase 2: CoT Instruction Tuning → Evaluation)
  - Core idea: In Phase 2, decompose planning into state-action-state triplets and verify each transition with an external validator (VAL). Feed this back into instruction tuning.
  - Insights:
    - Strongly enforces logical coherence at each step while overcoming the limits of LLM self-reflection through formal external validation.
    - Does not target optimality; focuses on satisficing plans. Still yields notable gains in generalization and verification reliability.

Tables
- Table 1: Main plan accuracy (100 tests/domain)
  - Llama-3, Detailed feedback, η=15: 94% (Blocksworld), 64% (Mystery-BW), 79% (Logistics).
  - GPT-4, Detailed feedback, η=15: 91%, 59%, 78%.
  - Insights:
    - PDDL-INSTRUCT (especially with Detailed feedback) consistently tops all settings. For Llama-3, average absolute +35%p over basic instruction tuning (SD 20), and +66%p over baseline (SD 3).
    - Detailed > Binary feedback: with η=15 and Llama-3, +5%p (BW), +15%p (Mystery-BW), +7%p (Logistics).
    - Increasing η improves accuracy: 10→15 adds +4.3%p on average for Detailed, +3.3%p for Binary.
    - Phase 1+Phase 2 outperforms either alone, especially on harder domains.

- Table 2: Hyperparameters (Phase 1 vs Phase 2)
  - Insights:
    - Phase 2 extends context and generations to accommodate long CoT, reduces batch size, and lowers temperature to encourage deterministic, precise reasoning.
    - Dual learning rates separate “reasoning quality” from “final task” optimization, minimizing interference.

- Table 3: Resources/time
  - 2×RTX 3080 (24GB), ~12h (Phase 1) + ~18h (Phase 2) = 30h total; ~2.3s inference/problem.
  - Insight: Reproducible on mid-range hardware; practical for research and development.

- Table 4: Ablation (Llama-3)
  - P2 only (Detailed, η=15) beats P1 only on Logistics (45% vs 23%), but not on BW/Mystery-BW.
  - P1 + Detailed (η=15) is best across all domains (94/64/79%).
  - Insight: Phase 1 builds domain/format understanding and error explanation; Phase 2 strengthens step-level verification; the combination yields the strongest generalization/accuracy.

- Table 5: Error analysis (Phase 1+Detailed, η=15, Llama-3)
  - Total failure: 6% (BW), 36% (Mystery-BW), 21% (Logistics).
  - Mystery-BW dominated by incorrect effect application (12.4%) and precondition violations (8.7%) due to obfuscated predicates.
  - Insight: Even with CoT, semantic grounding is the bottleneck under predicate obfuscation; effect propagation is more fragile than precondition checking.

Appendices
- Appendix A: Setup, losses, algorithm
  - Losses:
    - Lreasoning enforces step correctness using symmetric set difference and VAL-driven penalties.
    - Lfinal penalizes invalid plans and aligns validity prediction via BCE, ensuring “reasoning quality → final success.”
  - Two-stage optimization reduces signal collision and stabilizes training.
  - Insight: Fine-grained, validator-grounded supervision makes CoT faithful and reliable.

- Appendix B: Prompt design
  - Phase 1: Instructs “why each action is valid/invalid” via explicit preconditions/effects/state updates (with correct/incorrect examples).
  - Phase 2: Enforces state journaling and integrates VAL logs (added/deleted predicates) into feedback prompts.
  - Insight: Exposing validator traces scaffolds accurate internalization of transition dynamics, explaining why Detailed feedback outperforms Binary.

- Appendix C: Extended results
  - Confirms ablation and error patterns: Detailed + phased training is decisive; semantic grounding limits remain in obfuscated domains.

Overall insights
- Validator-guided logical CoT markedly boosts plan validity and reliability; Detailed feedback and sufficient iteration (η) are key.
- The division of labor—Phase 1 (concept/format/error explanation) + Phase 2 (verification-centric CoT)—creates strong synergy.
- Predicate grounding is the current bottleneck on obfuscated domains; future gains likely from better semantic alignment, plan-quality objectives, and richer PDDL features.

<br/>
# refer format:


BibTeX
@misc{verma2025teaching,
  title         = {Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning},
  author        = {Verma, Pulkit and La, Ngoc and Favier, Anthony and Mishra, Swaroop and Shah, Julie A.},
  year          = {2025},
  eprint        = {2509.13351},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2509.13351},
  note          = {v1, 14 Sep 2025}
}



시카고 스타일(Notes and Bibliography)  
Verma, Pulkit, Ngoc La, Anthony Favier, Swaroop Mishra, and Julie A. Shah. “Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning.” arXiv, September 14, 2025. https://arxiv.org/abs/2509.13351.


시카고 스타일(Author–Date)  
Verma, Pulkit, Ngoc La, Anthony Favier, Swaroop Mishra, and Julie A. Shah. 2025. “Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning.” arXiv, September 14. https://arxiv.org/abs/2509.13351.
