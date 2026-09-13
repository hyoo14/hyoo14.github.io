---
layout: post
title:  "[2026]Verify Before You Commit: Towards Faithful Reasoning in LLM Agents via Self-Auditing"
date:   2026-09-13 21:37:04 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 여러 추론 만들게 해서 리즈닝 수정, 에이전트가 근거 기반하여 결론에 도달하도록 하는 프레임웍   



짧은 요약(Abstract) :

## 

이 논문은 **LLM 에이전트의 추론 과정이 겉보기에는 논리적이어도 실제로는 근거가 부족하거나 잘못된 경우**를 다룹니다. 에이전트는 이러한 중간 추론을 바탕으로 행동하고, 그 내용을 메모리에 저장하기 때문에 한 번의 잘못된 추론이 이후 의사결정에 반복적으로 영향을 줄 수 있습니다.

기존 방법들은 여러 추론 결과가 서로 일치하는지, 즉 **합의(consensus)**를 주로 확인합니다. 하지만 여러 추론이 같은 잘못된 가정이나 오류를 공유할 수 있으므로, 합의가 곧 올바른 추론을 의미하지는 않습니다.

이를 해결하기 위해 논문은 **SAVeR(Self-Audited Verified Reasoning)**라는 프레임워크를 제안합니다. SAVeR는 다음과 같은 절차를 사용합니다.

1. 서로 다른 추론 관점을 가진 여러 후보 추론을 생성합니다.
2. 추론 구조가 다양하고 품질이 높은 후보들을 선택합니다.
3. 각 추론의 중간 단계에서

   * 숨은 가정
   * 검증되지 않은 전제
   * 근거 없는 추론
   * 순환 논리
   * 모순
   * 과도한 일반화  
등의 문제를 찾아냅니다.
4. 문제가 발생한 부분만 최소한으로 수정하고, 명확한 검증 조건을 통과할 때까지 다시 점검합니다.
5. 모든 검증이 끝난 뒤에만 해당 추론을 행동이나 메모리에 반영합니다.

여섯 개의 벤치마크에서 실험한 결과, SAVeR는 최종 과제 성능을 크게 해치지 않으면서 **잘못된 중간 추론을 줄이고 추론의 충실성(faithfulness)을 높이는 효과**를 보였습니다. 핵심은 정답을 맞혔는지만 보는 것이 아니라, **에이전트가 그 결론에 도달한 과정이 실제 증거와 논리에 의해 뒷받침되는지 확인한다는 점**입니다.

---

## 

This paper addresses the problem that an LLM agent’s reasoning may appear coherent while still containing unsupported assumptions or invalid inferences. Since agents use their intermediate reasoning to decide actions and update memory, an unfaithful belief can be repeatedly propagated and cause behavioral drift over long interactions.

Existing approaches often rely on **consensus** among multiple reasoning traces. However, agreement does not guarantee faithfulness, because different traces may share the same hidden assumption or error.

To address this issue, the paper proposes **SAVeR (Self-Audited Verified Reasoning)**. The framework:

1. Generates candidate beliefs from diverse reasoning perspectives.
2. Selects structurally diverse and relatively high-quality candidates.
3. Audits their intermediate reasoning steps for problems such as missing assumptions, invalid preconditions, unjustified inferences, circular reasoning, contradictions, and overgeneralization.
4. Repairs only the localized faulty parts using explicit, verifiable acceptance criteria.
5. Commits the reasoning to actions or memory only after it passes the audit.

Experiments on six benchmark datasets show that SAVeR substantially improves reasoning faithfulness while maintaining competitive task performance. Its main contribution is to verify not only whether the final answer is correct, but also whether the reasoning leading to that answer is properly supported by evidence and logic.



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


### 1. 방법의 핵심 개념

이 논문은 **SAVeR(Self-Audited Verified Reasoning)**이라는 에이전트 추론 검증 프레임워크를 제안한다. 핵심 아이디어는 다음과 같다.

> 최종 답이 맞는지만 확인하지 말고, 에이전트가 행동하거나 메모리에 저장하기 전에 중간 추론 과정 자체가 근거와 논리에 부합하는지 검사한다.

LLM 에이전트에서는 추론 결과가 단순한 설명이 아니라 다음 행동, 도구 사용, 메모리 업데이트에 직접 사용된다. 따라서 중간 추론에 근거 없는 가정이나 잘못된 추론이 있으면 이후 단계로 반복적으로 전파될 수 있다.

---

### 2. 모델 및 아키텍처

* 별도의 새로운 신경망 구조를 제안한 것은 아니다.
* 하나의 LLM을 여러 역할 또는 **페르소나(persona)**로 프롬프트하여 여러 후보 추론을 생성한다.
* 실험 모델:

  * LLaMA-3.1-8B
  * LLaMA-3.2-3B
  * Qwen2.5-7B
* 모든 모델은 **zero-shot 방식**으로 사용되었으며, task-specific fine-tuning은 하지 않았다.
* 즉, SAVeR은 새로운 모델 자체라기보다 기존 LLM 위에 추가되는 **추론 생성–선택–감사–수리 파이프라인**이다.

---

### 3. 전체 처리 과정

#### 단계 1: 다양한 후보 추론 생성

하나의 LLM에 서로 다른 추론 관점을 부여하여 여러 후보 belief를 만든다.

예를 들면 다음과 같은 페르소나를 사용할 수 있다.

* 가정 우선형
* 증거 우선형
* 엔티티 구분형
* 감사 중심형

각 후보는 다음 두 부분으로 구성된다.

[
y_i=(c_i,r_i)
]

* (c_i): 최종 주장 또는 답
* (r_i): 단계별 추론 과정

이 단계의 목적은 단순히 답을 많이 생성하는 것이 아니라, 서로 다른 오류 유형과 추론 구조를 드러내는 것이다.

---

#### 단계 2: 구조를 고려한 후보 선택

모든 후보를 감사하면 비용이 크기 때문에, 일부 후보만 선택해 검사한다.

추론 구조를 다음과 같은 특징으로 표현한다.

* 추론 단계의 세분화 정도
* 암묵적 가정의 사용 여부
* 검증 행동의 정도
* 전체 추론 구조의 유형

그 후 **quality-aware diversity kernel**과 **k-DPP(k-Determinantal Point Process)**를 사용해 서로 비슷한 후보만 중복 선택하지 않도록 한다.

쉽게 말해:

> 품질이 괜찮으면서도 서로 다른 방식으로 추론한 후보를 골라 감사한다.

이를 통해 동일한 잘못된 가정을 공유하는 후보들을 여러 개 검사하는 문제를 줄인다.

---

#### 단계 3: 적대적 추론 감사

선택된 추론에 대해 별도의 감사 모듈이 각 단계의 논리적·증거적 타당성을 검사한다.

감사는 대체 답을 생성하거나 전체 추론을 다시 쓰는 것이 아니라, **문제가 발생한 구체적인 단계와 원인만 찾아내는 것**을 목표로 한다.

논문에서 정의한 주요 오류 유형은 다음과 같다.

1. **Missing Assumption**  
추론에 필요한 가정이 명시되지 않음
2. **Invalid Precondition**  
엔티티 확인이나 사실 조건 등 필요한 전제조건이 검증되지 않음
3. **Unjustified Inference**  
근거 없이 결론을 도출함. 예: 수치 추측, 단순한 패턴 연상
4. **Circular Reasoning**  
결론을 사실상 그 결론의 근거로 다시 사용하는 순환 논리
5. **Contradiction**  
앞선 단계나 증거와 모순됨
6. **Overgeneralization**  
제한된 증거를 더 넓은 상황에 부당하게 일반화함

감사 결과는 오류 유형, 문제가 발생한 단계, 문제 구간, 진단 질문, 수정 조건을 포함하는 구조화된 기록으로 출력된다.

---

#### 단계 4: 최소 수정 기반의 추론 수리

감사만으로는 추론이 개선되지 않기 때문에, SAVeR은 문제가 있는 부분만 최소한으로 수정한다.

핵심 원칙은 다음과 같다.

* 정상적인 추론 단계는 그대로 유지
* 오류가 발생한 부분만 수정
* 가능한 경우 외부 증거에 직접 연결
* 수정 후 만족해야 할 명확한 검증 조건을 설정

예를 들어, “경기장의 규모가 비슷하므로 수용 인원은 약 3,700명일 것”이라는 추론은 증거 없는 수치 추측으로 판정될 수 있다. 이 경우 전체 답변을 다시 생성하지 않고 다음처럼 수정한다.

* 경기장 이름을 먼저 확인
* 출처에서 좌석 수를 직접 추출
* 명시된 수치만 최종 답으로 사용

수정된 추론은 다시 감사한다. 오류가 남아 있으면 감사–수리 과정을 반복하며, 모든 조건을 만족한 경우에만 행동 또는 메모리 저장을 허용한다.

---

### 4. 추론의 충실성 정의

논문은 각 추론 단계가 당시 이용 가능한 정보에 의해 뒷받침되는지를 평가한다.

각 단계 (s_l)에 대해 다음 정보를 고려한다.

* 입력 문제
* 이전 추론 단계
* 검색 문서, 도구 출력, 환경 관찰 등 이용 가능한 증거

각 단계의 지원 점수가 임계값보다 낮으면 해당 단계를 **unfaithful**, 즉 충실하지 않은 단계로 본다.

따라서 최종 답이 맞더라도 다음과 같은 경우에는 추론이 충실하지 않다고 판단한다.

* 우연히 정답을 맞힘
* 근거 없이 숫자를 추정함
* 질문에 없는 대상을 임의로 연결함
* 결론을 다시 결론의 근거로 사용함

---

### 5. 학습 데이터와 실험 데이터

SAVeR 자체를 별도로 학습시키기 위한 새로운 training dataset은 사용하지 않았다. 모델은 zero-shot 프롬프트 방식으로 작동한다.

평가에는 다음 6개 데이터셋을 사용했다.

* **HotpotQA**: 다중 문서·다중 홉 질의응답
* **2WikiMHQA**: 위키피디아 기반 다중 홉 추론
* **MuSiQue**: 복합적인 다중 홉 질의응답
* **Natural Questions**: 실제 검색 질의 기반 개방형 QA
* **FEVER**: 증거 기반 사실 검증
* **Quoref**: 문서 내 대명사 및 핵심어 참조 추론

추가 실험에서는 다음 과제도 평가했다.

* GSM8K, MATH: 수학 추론
* HumanEval, MBPP: 코드 생성

---

### 6. 비교 기법

SAVeR은 다음 방법들과 비교되었다.

* **Vanilla LM**: 추론 없이 바로 답변
* **CoT**: 단계별 사고 유도
* **MAD**: 여러 에이전트의 토론과 합의
* **Self-Refine**: 자기 비판과 반복 수정
* **Best-of-2**: 두 후보 중 더 나은 답 선택

논문의 주장은 단순한 다수결이나 합의가 반드시 충실성을 보장하지 않는다는 것이다. 여러 후보가 같은 암묵적 가정을 공유하면, 잘못된 추론이 오히려 합의에 의해 강화될 수 있다.

---

### 7. 평가 결과

SAVeR은 최종 답변 성능을 크게 희생하지 않으면서 추론 충실성을 개선했다.

주요 평가 지표는 다음과 같다.

* **Avg Viol**: 추론 궤적당 평균 오류 수
* **VFR**: 오류가 전혀 없는 추론의 비율
* **USR**: 전체 단계 중 오류가 있는 단계의 비율
* **Post-Res**: 수리 후 남은 오류 비율

LLaMA-3.1-8B의 다중 홉 QA 실험에서 SAVeR은 예를 들어 HotpotQA에서 다음 결과를 보였다.

* 평균 오류 수: **0.37**
* 오류 없는 추론 비율: **81.36%**
* 오류 단계 비율: **9.12%**
* 수리 후 잔여 오류: **0.05**

즉, 논문의 핵심 성과는 단순히 정답률을 높이는 것이 아니라, **행동이나 메모리에 저장되기 전에 근거 없는 추론을 줄이는 것**이다.

---

### 8. 한계

* 여러 후보 생성과 반복적인 감사–수리 때문에 계산 비용이 증가한다.
* 논문 실험에서 SAVeR은 단일 추론 방식보다 더 많은 토큰과 API 호출을 사용했다.
* 간단한 문제에도 엄격한 검증을 적용하면 불필요한 연산이 발생할 수 있다.
* 현재는 문제 난이도에 따라 감사 깊이를 자동 조절하는 기능이 충분하지 않다.

---

### 한 줄 요약

**SAVeR은 기존 LLM이 여러 후보 추론을 생성한 뒤, 구조적으로 다양한 후보를 선택하고, 중간 추론의 오류를 단계별로 감사하며, 오류 부분만 증거와 검증 조건에 맞게 최소 수정한 후에야 행동이나 메모리 저장을 허용하는 프롬프트 기반 에이전트 추론 검증 프레임워크이다.**

---

## 



### 1. Core Idea

The paper proposes **SAVeR (Self-Audited Verified Reasoning)**, a framework for verifying the faithfulness of an agent’s intermediate reasoning before it commits to an action or writes information to memory.

The main idea is:

> A correct final answer is not sufficient. The intermediate reasoning steps must also be logically and evidentially supported.

This is especially important for LLM agents because their reasoning trajectories influence later actions, tool calls, and memory updates.

---

### 2. Models and Architecture

SAVeR does not introduce a new neural network architecture. Instead, it is an additional reasoning-control framework built on top of existing LLMs.

The experiments use:

* LLaMA-3.1-8B
* LLaMA-3.2-3B
* Qwen2.5-7B

The models are used in a **zero-shot setting**, without task-specific fine-tuning.

A single LLM is prompted with different reasoning personas, such as:

* assumption-first reasoning
* evidence-first reasoning
* entity-disambiguation reasoning
* audit-oriented reasoning

Thus, SAVeR is better understood as a **generation–selection–audit–repair pipeline**, rather than a new model architecture.

---

### 3. Main Pipeline

#### Step 1: Generate Diverse Candidate Beliefs

The model produces multiple candidate beliefs under different persona instructions.

Each belief consists of:

[
y_i=(c_i,r_i)
]

* (c_i): the final claim or answer
* (r_i): the step-by-step reasoning trajectory

The goal is not merely to sample more answers. It is to expose structurally different reasoning patterns and potential failure modes.

---

#### Step 2: Structure-Aware Belief Selection

Auditing every candidate would be expensive. Therefore, SAVeR selects a subset of candidates for auditing.

Each reasoning trajectory is represented using structural features such as:

* reasoning-step granularity
* use of implicit assumptions
* verification behavior
* global reasoning structure

A quality-aware diversity kernel and **k-DPP** are then used to select candidates that are both reasonably useful and structurally diverse.

In simple terms:

> Select high-quality candidates that reason in different ways, rather than repeatedly auditing similar reasoning patterns.

---

#### Step 3: Adversarial Reasoning Audit

An auditor examines the selected reasoning trajectories step by step.

The auditor does not generate alternative answers or rewrite the entire reasoning chain. Its purpose is to localize the exact reasoning step that lacks sufficient support.

The main violation types are:

1. **Missing Assumption**  
A necessary assumption is implicit or unstated.
2. **Invalid Precondition**  
A required condition, such as entity identity, has not been verified.
3. **Unjustified Inference**  
A conclusion is drawn without adequate evidence, such as numerical guessing.
4. **Circular Reasoning**  
The conclusion is indirectly used to justify itself.
5. **Contradiction**  
A step conflicts with previous reasoning or available evidence.
6. **Overgeneralization**  
A conclusion extends beyond the scope of its supporting evidence.

The audit output includes the violated step, the failing fragment, a diagnostic probe, the violation type, and an explicit acceptance criterion for repair.

---

#### Step 4: Constraint-Guided Minimal Repair

Auditing alone does not correct the reasoning. SAVeR therefore repairs only the localized faulty parts.

The repair principle is:

* preserve valid reasoning steps
* edit only the failing slice
* connect new claims directly to available evidence
* define a concrete condition that must be satisfied after repair

For example, instead of estimating an arena’s capacity from similar arenas, the system must retrieve the arena’s actual capacity from an evidence sentence.

The repaired trajectory is audited again. The audit–repair loop continues until no violations remain or the process reaches its iteration limit. Only then is the belief committed to an action or written to memory.

---

### 4. Formalization of Faithfulness

For each reasoning step, SAVeR evaluates whether it is supported by the information available at that point, including:

* the original task
* previous reasoning steps
* retrieved documents
* tool outputs
* environment observations

If the support score of a step falls below a predefined threshold, that step is marked as unfaithful.

Therefore, a reasoning process may be considered unfaithful even when its final answer is correct—for example, when it reaches the correct answer through guessing, unsupported assumptions, or circular justification.

---

### 5. Data and Training

SAVeR does not require a new training dataset or additional fine-tuning. It operates primarily through prompting and inference-time verification.

The main evaluation datasets are:

* **HotpotQA**: multi-hop question answering
* **2WikiMHQA**: Wikipedia-based multi-hop reasoning
* **MuSiQue**: compositional multi-hop QA
* **Natural Questions**: open-domain question answering
* **FEVER**: evidence-based fact verification
* **Quoref**: coreference reasoning in a single passage

Additional experiments evaluate:

* GSM8K and MATH for mathematical reasoning
* HumanEval and MBPP for code generation

---

### 6. Baselines

SAVeR is compared with:

* **Vanilla LM**: direct answer generation
* **CoT**: chain-of-thought prompting
* **MAD**: multi-agent debate
* **Self-Refine**: iterative self-critique and revision
* **Best-of-2**: selecting between two candidate outputs

The paper argues that consensus does not necessarily imply faithfulness. Multiple reasoning traces may share the same hidden assumption, causing an unsupported belief to be reinforced by majority agreement.

---

### 7. Results

SAVeR improves reasoning faithfulness while maintaining competitive task-level performance.

The main faithfulness metrics are:

* **Avg Viol**: average number of violations per trajectory
* **VFR**: proportion of violation-free trajectories
* **USR**: proportion of reasoning steps marked as unfaithful
* **Post-Res**: residual violations after repair

For example, with LLaMA-3.1-8B on HotpotQA, SAVeR achieves:

* Avg Viol: **0.37**
* VFR: **81.36%**
* USR: **9.12%**
* Post-Res: **0.05**

The main contribution is therefore not simply better answer accuracy, but preventing unsupported intermediate beliefs from being used for actions or memory updates.

---

### 8. Limitations

* Candidate generation and iterative audit–repair increase computational cost.
* SAVeR uses more tokens and API calls than single-pass prompting.
* Strict verification may be unnecessary for simple tasks.
* The current framework does not fully adapt the verification depth to task difficulty.

---

### One-Sentence Summary

**SAVeR is a prompt-based agent reasoning framework that generates diverse candidate beliefs, selects structurally different candidates, audits their intermediate reasoning for logical and evidential violations, minimally repairs faulty steps under explicit acceptance criteria, and commits the belief to actions or memory only after verification.**



<br/>
# Results


### 1. 비교 대상과 테스트 데이터

논문은 **SAV ER(Self-Audited Verified Reasoning)**를 다음 기준선들과 비교했다.

* **Vanilla LM**: 명시적 추론 없이 바로 답변
* **CoT**: Chain-of-Thought 기반 단계적 추론
* **MAD**: 여러 에이전트의 토론 결과를 결합하는 Multi-Agent Debate
* **Self-Refine**: 자기 비판과 반복 수정을 통한 개선
* **B-2**: 두 개의 후보 중 더 나은 답을 선택하는 방식

평가는 세 가지 유형의 총 **6개 테스트 데이터셋**에서 수행되었다.

|데이터 유형|데이터셋|평가 초점|
|-|-|-|
|다중 홉 질의응답|HotpotQA, 2WikiMHQA, MuSiQue|여러 문서와 추론 단계를 연결하는 능력|
|증거 민감형 QA|NQ, FEVER|답변이 실제 증거에 의해 충분히 뒷받침되는지|
|국소 추론|Quoref|한 문서 안에서 대명사·참조 관계를 올바르게 해결하는지|

사용한 모델은 **LLaMA-3.1-8B, LLaMA-3.2-3B, Qwen-2.5-7B**이며, 별도의 태스크 특화 파인튜닝 없이 zero-shot으로 평가했다.

---

### 2. 평가 메트릭

#### 과제 수행 성능

* **EM (Exact Match)**: 정답과 완전히 일치하는 비율
* **F1**: 정답과 예측 답변의 단어 단위 겹침 정도
* **Pass@1**: 코드 생성 문제에서 첫 번째 답변이 통과하는 비율

#### 추론 충실성(Faithfulness)

* **Avg Viol.**: 추론 궤적 하나당 평균 위반 횟수. 낮을수록 좋음
* **VFR (Violation-Free Rate)**: 위반이 하나도 없는 추론의 비율. 높을수록 좋음
* **USR (Unfaithful Step Rate)**: 전체 추론 단계 중 충실하지 않은 단계의 비율. 낮을수록 좋음
* **Post-Res.**: 감사(audit)와 수정(repair) 후에도 남은 위반 수. 낮을수록 좋음

즉, 이 논문은 단순히 최종 정답률만 보는 것이 아니라, **정답에 도달하는 중간 추론이 증거와 논리에 의해 뒷받침되는지**도 평가했다.

---

### 3. 최종 답변 성능

대표적으로 **LLaMA-3.1-8B**에서 SAV ER은 대부분의 데이터셋에서 기준선보다 경쟁력 있는 성능을 보였다.

|데이터셋|SAV ER EM / F1|주요 비교|
|-|-:|-|
|HotpotQA|**43.7 / 52.6**|MAD: 43.1 / 51.2|
|2WikiMHQA|**47.7 / 55.5**|MAD: 47.9 / 55.4|
|MuSiQue|**31.8 / 42.5**|MAD: 30.9 / 40.8|
|NQ|**37.1 / 47.8**|MAD: 36.6 / 46.9|
|Quoref|**37.2 / 45.7**|MAD: 36.3 / 45.2|
|FEVER|**61.1 EM**|MAD: 60.7|

핵심적으로:

* SAV ER은 **MuSiQue, NQ, Quoref, FEVER**에서 MAD보다 명확히 높은 성능을 보였다.
* 2WikiMHQA에서는 EM이 MAD와 비슷하거나 약간 낮았지만, F1은 더 높았다.
* 성능 향상이 특정 모델 크기에만 의존하지 않고 **LLaMA와 Qwen, 3B와 7–8B 모델에서 일관되게 나타났다.**
* 따라서 SAV ER은 추론 검증을 강화하면서도 최종 과제 성능을 크게 희생하지 않았다.

---

### 4. 추론 충실성 결과

LLaMA-3.1-8B 기준 다중 홉 QA 결과는 다음과 같다.

|방법|HotpotQA Avg Viol. / VFR / USR|2WikiMHQA|MuSiQue|
|-|-:|-:|-:|
|CoT|1.98 / 24.89% / 27.36%|2.21 / 17.41% / 32.11%|2.91 / 13.26% / 37.58%|
|MAD|1.33 / 36.74% / 23.94%|1.81 / 32.78% / 28.82%|2.16 / 26.17% / 36.51%|
|Self-Refine|1.48 / 31.80% / 21.77%|1.93 / 26.59% / 26.73%|2.03 / 21.96% / 31.45%|
|**SAV ER**|**0.37 / 81.36% / 9.12%**|**0.56 / 72.34% / 13.84%**|**0.83 / 69.38% / 19.73%**|

SAV ER의 특징은 다음과 같다.

* 평균 위반 횟수가 모든 데이터셋에서 가장 낮았다.
* 위반이 전혀 없는 추론의 비율인 VFR이 크게 높아졌다.
* USR은 MAD보다 대략 절반 이하 수준으로 감소했다.
* 수정 후 남은 위반(Post-Res)은 각각 **0.05, 0.08, 0.11**로 낮았다.
* MAD도 반복 토론을 통해 오류를 줄였지만, **검증 가능한 기준 없이 합의만 형성하기 때문에 잘못된 공통 가정이 남는 문제**가 있었다.

예를 들어 HotpotQA에서 SAV ER은:

* Avg Viol.: MAD **1.33 → SAV ER 0.37**
* VFR: MAD **36.74% → SAV ER 81.36%**
* USR: MAD **23.94% → SAV ER 9.12%**

로 개선되었다.

---

### 5. 구성 요소별 분석

절제 실험에서는 SAV ER의 각 모듈을 제거했을 때 성능이 어떻게 변하는지 확인했다.

* **Persona 생성 제거**: 다양한 추론 오류 유형을 발견하기 어려워져 위반률 증가
* **k-DPP 선택 제거**: 서로 비슷한 오류를 가진 후보들이 중복 선택되어 검증 효율 감소
* **Auditing 제거**: 오류를 발견할 수 없으므로 Avg Viol과 USR이 크게 증가
* **Repair 제거**: 오류를 발견해도 실제로 수정되지 않아 충실성 저하

LLaMA-3.1-8B의 HotpotQA에서:

|설정|Avg Viol.|VFR|USR|
|-|-:|-:|-:|
|SAV ER|**0.37**|**81.36%**|**9.12%**|
|Persona 제거|0.49|74.55%|11.97%|
|k-DPP 제거|0.64|71.47%|15.86%|
|Auditing 제거|1.37|42.65%|26.74%|
|Repair 제거|1.56|33.68%|37.63%|

따라서 가장 중요한 구성 요소는 **감사(auditing)와 수정(repair)**이며, persona와 k-DPP는 다양한 실패 유형을 효과적으로 노출하기 위한 역할을 한다.

---

### 6. 계산 비용과 추가 결과

SAV ER은 여러 후보를 만들고 반복적으로 감사·수정하기 때문에 계산 비용은 증가한다.

HotpotQA에서 LLaMA-3.1-8B 기준:

|방법|평균 토큰 수|평균 시간|API 호출|
|-|-:|-:|-:|
|Vanilla LM|2.3k|2.2초|1|
|CoT|2.8k|3.1초|1|
|MAD|11.2k|11.5초|6|
|Self-Refine|7.6k|8.7초|4|
|**SAV ER**|**12.4k**|**13.4초**|**8**|

즉, SAV ER은 가장 저렴한 방법은 아니지만, MAD와 비슷한 다중 호출 비용으로 훨씬 낮은 추론 오류율을 달성했다.

또한 부록 실험에서는 QA 외에도:

* **GSM8K**: 88.4 EM
* **MATH**: 58.2 EM
* **HumanEval**: 76.5 Pass@1
* **MBPP**: 76.1 Pass@1

을 기록해 수학 추론과 코드 생성에서도 최종 성능 및 추론 충실성이 개선되는 경향을 보였다.

### 종합 결론

SAV ER의 가장 중요한 성과는 단순히 정답률을 조금 높인 것이 아니라, **행동이나 메모리에 기록되기 전에 중간 추론의 오류를 찾아내고, 증거와 논리에 맞게 최소한으로 수정했다는 점**이다. 특히 합의 기반 방법인 MAD와 달리, SAV ER은 “여러 모델이 같은 말을 하는가?”가 아니라 **“각 추론 단계가 실제 증거와 논리적 조건을 만족하는가?”**를 확인한다.

다만 여러 차례의 생성·감사·수정 과정 때문에 계산 비용이 증가하며, 논문도 이를 주요 한계로 인정한다.

---

## 



### 1. Competitors and Test Data

The paper evaluates **SAV ER (Self-Audited Verified Reasoning)** against:

* **Vanilla LM**: Direct answer generation without explicit reasoning
* **CoT**: Chain-of-Thought reasoning
* **MAD**: Multi-Agent Debate
* **Self-Refine**: Iterative self-critique and revision
* **B-2**: Best-of-two candidate selection

Six datasets are used:

|Task type|Datasets|Main focus|
|-|-|-|
|Multi-hop QA|HotpotQA, 2WikiMHQA, MuSiQue|Connecting multiple documents and reasoning hops|
|Evidence-sensitive QA|NQ, FEVER|Whether conclusions are sufficiently supported by evidence|
|Local reasoning|Quoref|Resolving references and coreference within one passage|

The experiments use **LLaMA-3.1-8B, LLaMA-3.2-3B, and Qwen-2.5-7B** in a zero-shot setting.

---

### 2. Evaluation Metrics

#### Task performance

* **EM (Exact Match)**: Exact answer-match rate
* **F1**: Token-level overlap between prediction and reference
* **Pass@1**: Whether the first generated program passes the test

#### Reasoning faithfulness

* **Avg Viol.**: Average number of detected violations per trajectory; lower is better
* **VFR**: Violation-Free Rate; higher is better
* **USR**: Unfaithful Step Rate; lower is better
* **Post-Res.**: Remaining violations after audit and repair; lower is better

These metrics evaluate not only whether the final answer is correct, but also whether the intermediate reasoning is logically and evidentially supported.

---

### 3. Final Task Performance

Under **LLaMA-3.1-8B**, SAV ER achieved competitive or superior results on most benchmarks.

|Dataset|SAV ER EM / F1|Main comparison|
|-|-:|-|
|HotpotQA|**43.7 / 52.6**|MAD: 43.1 / 51.2|
|2WikiMHQA|**47.7 / 55.5**|MAD: 47.9 / 55.4|
|MuSiQue|**31.8 / 42.5**|MAD: 30.9 / 40.8|
|NQ|**37.1 / 47.8**|MAD: 36.6 / 46.9|
|Quoref|**37.2 / 45.7**|MAD: 36.3 / 45.2|
|FEVER|**61.1 EM**|MAD: 60.7|

The main findings are:

* SAV ER clearly outperformed MAD on MuSiQue, NQ, Quoref, and FEVER.
* On 2WikiMHQA, its EM was slightly below MAD, but its F1 was higher.
* The improvements were stable across different model families and model sizes.
* Faithfulness verification improved reasoning quality without substantially harming end-task accuracy.

---

### 4. Reasoning Faithfulness

For LLaMA-3.1-8B on multi-hop QA:

|Method|HotpotQA Avg Viol. / VFR / USR|2WikiMHQA|MuSiQue|
|-|-:|-:|-:|
|CoT|1.98 / 24.89% / 27.36%|2.21 / 17.41% / 32.11%|2.91 / 13.26% / 37.58%|
|MAD|1.33 / 36.74% / 23.94%|1.81 / 32.78% / 28.82%|2.16 / 26.17% / 36.51%|
|Self-Refine|1.48 / 31.80% / 21.77%|1.93 / 26.59% / 26.73%|2.03 / 21.96% / 31.45%|
|**SAV ER**|**0.37 / 81.36% / 9.12%**|**0.56 / 72.34% / 13.84%**|**0.83 / 69.38% / 19.73%**|

SAV ER:

* Obtained the lowest average violation count.
* Produced substantially more violation-free trajectories.
* Reduced the proportion of unfaithful reasoning steps.
* Achieved low post-repair residual violations: **0.05, 0.08, and 0.11** on the three datasets.

On HotpotQA, for example:

* Avg Viol. decreased from **1.33 with MAD to 0.37 with SAV ER**
* VFR increased from **36.74% to 81.36%**
* USR decreased from **23.94% to 9.12%**

MAD reduces some errors through debate, but it can still preserve shared unsupported assumptions. SAV ER instead checks whether each reasoning step satisfies explicit evidential and logical conditions.

---

### 5. Ablation Results

Removing any major module degraded faithfulness.

On HotpotQA with LLaMA-3.1-8B:

|Setting|Avg Viol.|VFR|USR|
|-|-:|-:|-:|
|SAV ER|**0.37**|**81.36%**|**9.12%**|
|Without Persona|0.49|74.55%|11.97%|
|Without k-DPP|0.64|71.47%|15.86%|
|Without Auditing|1.37|42.65%|26.74%|
|Without Repair|1.56|33.68%|37.63%|

Interpretation:

* **Persona generation** improves coverage of diverse failure patterns.
* **k-DPP selection** reduces redundant and structurally correlated reasoning candidates.
* **Auditing** is necessary to localize unsupported steps.
* **Repair** is necessary to actually correct those failures.

The largest degradation occurs when auditing or repair is removed.

---

### 6. Computational Cost and Additional Results

SAV ER is more expensive because it generates multiple candidates and performs iterative audit-repair cycles.

For HotpotQA with LLaMA-3.1-8B:

|Method|Avg. tokens|Avg. time|API calls|
|-|-:|-:|-:|
|Vanilla LM|2.3k|2.2 s|1|
|CoT|2.8k|3.1 s|1|
|MAD|11.2k|11.5 s|6|
|Self-Refine|7.6k|8.7 s|4|
|**SAV ER**|**12.4k**|**13.4 s**|**8**|

Thus, SAV ER has higher computational overhead, but it achieves considerably better faithfulness than the baselines at a comparable multi-call inference cost.

Additional experiments also showed improvements beyond QA:

* **GSM8K**: 88.4 EM
* **MATH**: 58.2 EM
* **HumanEval**: 76.5 Pass@1
* **MBPP**: 76.1 Pass@1

### Overall Conclusion

The main contribution of SAV ER is not merely a small improvement in final accuracy. Its key benefit is that it **detects and minimally repairs unsupported intermediate reasoning before the agent commits to an action or writes information into memory**.

Unlike consensus-based methods such as MAD, SAV ER does not treat agreement as evidence of faithfulness. It verifies whether each reasoning step is supported by available evidence and satisfies explicit logical acceptance criteria.

The main drawback is increased inference cost due to multiple candidate generations and iterative audit-repair cycles.



<br/>
# 예제




### 1. 먼저 구분할 점: 학습 데이터와 테스트 데이터

이 논문은 **별도의 태스크별 학습(training/fine-tuning)을 하지 않고**, LLaMA와 Qwen 모델을 **zero-shot 방식**으로 사용합니다. 따라서 논문에 제시된 예시는 전통적인 의미의 “훈련 데이터 입력–정답 출력”이라기보다 다음과 같습니다.

1. **SAV ER의 작동 방식을 보여주는 설명용 예시**
2. **테스트 벤치마크에서 사용한 구체적인 질의 예시**
3. **각 질의에 대해 모델이 생성한 후보 추론, 감사 결과, 수정 결과**

핵심은 최종 답이 맞는지만 보는 것이 아니라, **답을 만들기 전의 중간 추론이 근거와 논리에 의해 정당화되는지 검사**하는 것입니다.

---

### 2. 전체적인 입력–출력 구조

SAV ER은 하나의 답을 바로 내지 않고 다음 절차를 수행합니다.

|단계|입력|출력|
|-|-|-|
|신념 생성|질문, 검색 문서·도구 결과·환경 관찰|여러 개의 후보 답변과 추론 과정|
|신념 선택|후보 추론들의 구조적 특징|서로 다른 유형의 후보 추론 몇 개|
|감사(Audit)|선택된 추론과 이용 가능한 근거|오류가 발생한 단계, 오류 유형, 검증 조건|
|수정(Repair)|원래 추론, 감사 결과, 근거|문제가 있는 부분만 수정한 추론|
|재감사|수정된 추론|통과 또는 추가 수정|
|최종 커밋|검증을 통과한 후보|최종 답변 및 메모리 저장 내용|

감사에서 찾는 대표적인 오류 유형은 다음과 같습니다.

* **Missing Assumption**: 필요한 가정이 빠짐
* **Invalid Precondition**: 전제나 대상이 검증되지 않음
* **Unjustified Inference**: 근거 없이 결론을 도출함
* **Circular Reasoning**: 결론을 사실상 결론 자체로 정당화함
* **Contradiction**: 앞선 정보나 계산과 모순됨
* **Overgeneralization**: 제한된 근거를 지나치게 일반화함

---

## 3. 예시 1: Animorphs 질문과 순환적 추론

### 과제

> “1인칭으로 서술되는 과학 판타지 청소년 시리즈이며, 노예가 된 세계와 외계 종족의 이야기를 다루는 동반 도서가 있는 시리즈는 무엇인가?”

정답은 **Animorphs**입니다.

### 모델의 추론

1. 질문에는 동반 도서와 외계 종족에 관한 단서가 있다.
2. 이 단서는 **The Hork-Bajir Chronicles**를 떠올리게 한다.
3. The Hork-Bajir Chronicles는 Animorphs의 외전이다.
4. 따라서 질문의 시리즈는 Animorphs이다.

### 문제점

최종 답은 맞지만, 2번 단계의 “이 단서가 The Hork-Bajir Chronicles를 떠올리게 한다”는 연결이 **검증된 근거가 아닙니다**. 또한 3~4번 단계는 이미 Animorphs와 연결되는 책을 가정한 뒤, 그 가정을 이용해 Animorphs를 결론으로 도출합니다.

즉, 이 예시는 다음을 보여줍니다.

* **최종 정답이 맞아도 추론은 비충실할 수 있음**
* 모델이 나중에 내린 결론을 중간 추론이 사후적으로 정당화할 수 있음
* 에이전트에서는 이런 잘못된 추론이 메모리에 저장되어 이후 행동에 영향을 줄 수 있음

---

## 4. 예시 2: 경기장 좌석 수 질문

### 입력

> “Lewiston Maineiacs가 홈 경기를 치른 경기장의 좌석 수는 몇 명인가?”

### 후보 추론

#### 후보 1: 추정 기반 추론

> 비슷한 주니어 하키 경기장은 보통 3,500~4,000석이므로 약 3,700석일 것이다.

* 최종 후보 답: **3,700**
* 오류: **Unjustified Inference**
* 이유: 실제 근거 문서에 “좌석 수가 3,700석”이라고 명시되어 있지 않음

#### 후보 2: 근거 중심 추론

> 홈 경기장의 이름을 확인한 뒤, 해당 경기장에 대한 문서에서 정확한 좌석 수를 추출한다.

* 오류 가능성: 경기장 이름과 좌석 수를 연결하는 인용 문장이 없으면 **Invalid Precondition**

### SAV ER의 수정

SAV ER은 첫 번째 추론의 “3,500~4,000 → 3,700”이라는 추정 부분만 삭제합니다. 그리고 다음과 같이 수정합니다.

1. Lewiston Maineiacs의 홈 경기장 이름을 근거 문서에서 확인한다.
2. 경기장 문서에서 좌석 수가 명시된 문장을 찾는다.
3. 해당 문장에서 정확한 정수 값을 추출한다.

근거 문장:

> “Seating capacity: 3,677 (seated).”

### 최종 출력

> **3,677 seated**

메모리에는 “홈 경기장 → 3,677석”이라는 근거 있는 사실만 저장하고, “비슷한 경기장을 보고 3,700석으로 추정한다”는 잘못된 추정 방식은 저장하지 않습니다.

---

## 5. 예시 3: 2WikiMHQA의 수명 비교

### 입력

> “Giuseppe Cesari와 Nicos Poulantzas 중 누가 더 오래 살았는가?”

### 후보 출력

#### 후보 1: 시대 일반화

> Cesari는 더 오래된 역사적 인물이고, 현대 정치 이론가보다 오래 살았을 가능성이 높다. 따라서 Giuseppe Cesari가 더 오래 살았다.

* 오류: **Overgeneralization**
* 문제: 출생일과 사망일을 확인하지 않고 시대적 인상만으로 결론을 냄

#### 후보 2: 날짜와 계산 사용

> Giuseppe Cesari: 1568–1640  
> Nicos Poulantzas: 1936–1979  
> Cesari: 1640−1568 = 72년  
> Poulantzas: 1979−1936 = 73년

여기서 1979−1936은 **73이 아니라 43**입니다.

* 오류: **Contradiction**
* 문제: 인용한 날짜와 계산 결과가 일치하지 않음

### 수정 후

* Cesari: 1640−1568 = 72년
* Poulantzas: 1979−1936 = 43년
* 72 > 43

### 최종 출력

> **Giuseppe Cesari**

이 사례는 근거가 있는 것뿐 아니라, **근거를 사용한 계산 자체도 검증해야 한다**는 점을 보여줍니다.

---

## 6. 예시 4: MuSiQue의 “The Collegian” 질문

### 입력

> “The Collegian을 소유한 기관은 언제 설립되었는가?”

### 잘못된 후보 출력 1: 이름만 보고 대상 추정

> “The Collegian”은 유명한 대학 신문이므로 Hillsdale College를 가리킨다. Hillsdale College는 1844년에 설립되었다.  
> 최종 답: **1844**

* 오류: **Missing Assumption**
* 문제: 여러 기관이 “The Collegian”이라는 이름을 사용할 수 있는데, 어느 기관인지 근거 없이 정함

### 잘못된 후보 출력 2: 신문의 설립연도를 기관의 설립연도로 오인

> The Collegian은 1963년에 창간되었다. 따라서 소유 기관도 1963년에 설립되었다.  
> 최종 답: **1963**

* 오류: **Invalid Precondition**
* 문제: 질문은 신문이 아니라 **신문을 소유한 기관의 설립연도**를 묻고 있음

### SAV ER의 수정

1. 먼저 “The Collegian”의 소유 기관을 명시한 문장을 찾는다.
2. 그 기관이 **Houston Baptist University**임을 확인한다.
3. 신문이 아니라 해당 기관의 설립연도를 찾는다.
4. 기관의 설립연도인 **1960**을 추출한다.

### 최종 출력

> **1960**

이 예시는 다단계 질문에서 **대상(entity)과 속성(attribute)을 끝까지 정확히 연결해야 한다**는 점을 보여줍니다.

---

## 7. 예시 5: NQ의 시민권 선서 질문

### 입력

> “새 시민들이 하는 선서는 무엇인가?”

### 잘못된 추론

> 새 시민들은 충성을 맹세한다. 따라서 그 선서의 이름은 Oath of Allegiance이다.

* 오류: **Circular Reasoning**
* 문제: “충성을 맹세한다”는 의미적 설명을 공식 명칭의 근거처럼 사용함

또한 “new citizens”가 어느 나라의 시민권 취득자인지도 명확하지 않을 수 있습니다.

* 추가 오류 가능성: **Missing Assumption**
* 필요한 정보: 국가 또는 관할권

### SAV ER의 수정 조건

* 미국 귀화 절차를 가리킨다는 맥락을 명시할 것
* 공식 명칭이 실제 근거 문장에 나타날 것

### 최종 출력

> **United States Oath of Allegiance**

이때 모델은 단순히 “allegiance”라는 단어를 보고 명칭을 추측하는 것이 아니라, 미국 귀화 맥락과 공식 명칭을 각각 근거에 연결해야 합니다.

---

## 8. 논문에서 사용한 실제 테스트 태스크

논문은 다음 여섯 가지 벤치마크를 테스트에 사용합니다.

|태스크 유형|데이터셋|구체적인 요구|
|-|-|-|
|다중 홉 질의응답|HotpotQA|여러 문서의 정보를 연결해 답변|
|다중 홉 질의응답|2WikiMHQA|여러 단계의 인물·기관·속성 연결|
|다중 홉 질의응답|MuSiQue|하위 질문들을 순차적으로 조합|
|근거 민감 질의응답|Natural Questions|검색 근거를 바탕으로 자연어 질문에 답변|
|사실 검증|FEVER|주장이 Supported, Refuted, Not Enough Info 중 무엇인지 판단|
|단일 문서 추론|Quoref|문서 내 대명사와 대상의 관계를 해석|

추가 실험에서는 다음 태스크도 평가합니다.

* **GSM8K, MATH**: 수학 문제 풀이
* **HumanEval, MBPP**: 코드 생성

---

## 9. 핵심 결론

이 논문의 예시에서 중요한 점은 다음과 같습니다.

1. **정답이 맞는 것과 추론이 충실한 것은 다르다.**
2. SAV ER은 여러 후보 추론을 만든 뒤, 서로 다른 오류 패턴을 가진 후보를 선택한다.
3. 감사자는 답을 새로 제안하지 않고, 기존 추론의 문제 단계만 찾아낸다.
4. 수정 모듈은 전체 추론을 다시 쓰지 않고, 오류가 있는 부분만 최소한으로 수정한다.
5. 모든 검증 조건을 통과한 경우에만 답변을 행동이나 메모리에 반영한다.
6. 논문은 정확도를 크게 해치지 않으면서, 평균 오류 수와 비충실한 추론 단계 비율을 낮추는 것을 목표로 한다.

---

# 





## 1. Important distinction: training data vs. test examples

The paper does **not** fine-tune the models on task-specific training examples. It uses LLaMA and Qwen models in a **zero-shot setting**.

Therefore, the examples in the paper are mainly:

1. Illustrative examples of unfaithful reasoning
2. Concrete benchmark-style test questions
3. Candidate reasoning traces, audit results, repaired traces, and final answers

The goal is not only to obtain the correct final answer, but also to verify whether the intermediate reasoning is properly supported by evidence and logic.

---

## 2. General input–output structure of SAV ER

|Stage|Input|Output|
|-|-|-|
|Belief generation|Question, documents, tool outputs, observations|Multiple candidate answers and reasoning traces|
|Belief selection|Structural features of the traces|A diverse subset of candidates|
|Auditing|Selected trace and available evidence|Error location, violation type, acceptance criterion|
|Repair|Original trace, audit record, evidence|Minimal correction of the faulty part|
|Re-auditing|Repaired trace|Pass or another violation|
|Commitment|Verified candidate|Final answer and safe memory update|

The main violation types are:

* **Missing Assumption**
* **Invalid Precondition**
* **Unjustified Inference**
* **Circular Reasoning**
* **Contradiction**
* **Overgeneralization**

---

## 3. Example 1: Animorphs and circular reasoning

### Task

> What science-fantasy young adult series, told in first person, has companion books about enslaved worlds and alien species?

The correct answer is **Animorphs**.

### Model reasoning

1. The question mentions companion books and alien species.
2. This reminds me of *The Hork-Bajir Chronicles*.
3. *The Hork-Bajir Chronicles* is a companion book in the Animorphs franchise.
4. Therefore, the answer is Animorphs.

### Problem

The final answer is correct, but the reasoning is not fully faithful. The connection in step 2 is not verified by evidence. The reasoning effectively assumes the Animorphs connection and then uses that assumption to derive the conclusion.

This example shows that:

* A correct answer does not guarantee faithful reasoning.
* A model may use a conclusion to justify itself retrospectively.
* In an agent, such unsupported reasoning may later be stored in memory and affect future actions.

---

## 4. Example 2: Arena seating capacity

### Input

> How many people can the arena where the Lewiston Maineiacs played their home games seat?

### Candidate 1: Heuristic estimation

> Similar junior-hockey arenas seat 3,500–4,000 people, so the capacity is probably about 3,700.

* Candidate answer: **3,700**
* Violation: **Unjustified Inference**
* Reason: No explicit evidence states that the arena has 3,700 seats.

### SAV ER repair

SAV ER removes only the unsupported estimate and replaces it with an evidence-bound procedure:

1. Identify the home arena.
2. Retrieve evidence about that arena.
3. Extract the exact seated capacity from an evidence sentence.

Evidence:

> “Seating capacity: 3,677 (seated).”

### Final output

> **3,677 seated**

Only the evidence-grounded fact is written to memory; the heuristic estimation pattern is not stored.

---

## 5. Example 3: Comparing lifespans

### Input

> Who lived longer, Giuseppe Cesari or Nicos Poulantzas?

### Incorrect candidate 1: Historical generalization

> Cesari was an earlier historical figure, and earlier figures generally lived longer than modern political theorists. Therefore, Cesari lived longer.

* Violation: **Overgeneralization**
* Reason: The model does not extract dates or calculate lifespans.

### Incorrect candidate 2: Arithmetic inconsistency

> Giuseppe Cesari: 1568–1640  
> Nicos Poulantzas: 1936–1979  
> Cesari lived 72 years; Poulantzas lived 73 years.

The subtraction is wrong:

* 1640 − 1568 = 72
* 1979 − 1936 = **43**, not 73
* Violation: **Contradiction**

### Repaired output

> Cesari lived 72 years, while Poulantzas lived 43 years. Therefore, **Giuseppe Cesari** lived longer.

This example demonstrates that both the evidence and the calculations based on that evidence must be verified.

---

## 6. Example 4: The Collegian and entity–attribute confusion

### Input

> When was the institute that owned *The Collegian* founded?

### Incorrect candidate 1: Unsupported entity identification

> “The Collegian” refers to Hillsdale College. Hillsdale College was founded in 1844.  
> Answer: **1844**

* Violation: **Missing Assumption**
* Reason: The name “The Collegian” may refer to publications associated with different institutions.

### Incorrect candidate 2: Wrong attribute

> *The Collegian* was founded in 1963, so the institute that owned it was founded in 1963.  
> Answer: **1963**

* Violation: **Invalid Precondition**
* Reason: The question asks for the founding year of the owning institute, not the newspaper.

### SAV ER repair

1. Identify the owner of *The Collegian* from an explicit ownership statement.
2. Determine that the owner is **Houston Baptist University**.
3. Retrieve the founding year of the university, not the newspaper.
4. Output **1960**.

### Final output

> **1960**

This example illustrates the need to preserve the correct entity–attribute binding throughout multi-hop reasoning.

---

## 7. Example 5: Naturalization oath

### Input

> What is the oath that new citizens take?

### Incorrect reasoning

> New citizens swear allegiance. Therefore, the oath is called the Oath of Allegiance.

* Violation: **Circular Reasoning**
* Reason: The semantic idea of “allegiance” is treated as evidence for the formal name.

There may also be an unstated jurisdictional assumption: “new citizens” could refer to different countries.

### SAV ER repair requirements

* Explicitly establish that the context is U.S. naturalization.
* Extract the formal name from an evidence sentence.

### Final output

> **United States Oath of Allegiance**

The answer must be grounded in both the correct jurisdiction and an explicit naming statement.

---

## 8. Benchmark test tasks

The paper evaluates six main datasets:

|Task type|Dataset|Required reasoning|
|-|-|-|
|Multi-hop QA|HotpotQA|Connect information across multiple documents|
|Multi-hop QA|2WikiMHQA|Track entities, attributes, and multiple reasoning steps|
|Multi-hop QA|MuSiQue|Compose several sub-questions|
|Evidence-sensitive QA|Natural Questions|Answer using appropriate retrieved evidence|
|Fact verification|FEVER|Classify claims as Supported, Refuted, or Not Enough Info|
|Local reasoning|Quoref|Resolve coreference within a single passage|

Additional experiments evaluate:

* **GSM8K and MATH** for mathematical reasoning
* **HumanEval and MBPP** for code generation

---

## 9. Main takeaway

The paper’s examples emphasize that:

1. Correct final answers and faithful reasoning are different.
2. SAV ER generates multiple candidate belief states to expose different failure patterns.
3. The auditor identifies faulty steps rather than proposing a new answer.
4. The repair module changes only the localized faulty portions.
5. A belief is committed to action or memory only after satisfying explicit acceptance criteria.
6. The framework aims to reduce unsupported intermediate reasoning while preserving end-task performance.

<br/>
# 요약

SAV ER는 다양한 페르소나로 후보 추론을 생성·선택한 뒤, 중간 단계의 누락 가정·순환 추론·근거 없는 추론 등을 감사하고 최소한으로 수정한 후 행동이나 메모리에 반영하는 방법이다.  
LLaMA-3.1-8B 기준 HotpotQA에서 평균 위반 수를 0.37, 위반 없는 추론 비율을 81.36%, 수정 후 잔여 위반을 0.05로 낮추며 최종 성능도 경쟁력 있게 유지했다.  
예를 들어 경기장 좌석 수를 추정으로 지어내거나 신문 창간연도를 소유 기관의 설립연도로 착각한 추론을 찾아내고, 명시적 증거에 근거한 답으로 교정했다.  



&#x20; 
SAV ER generates and selects diverse candidate beliefs, audits intermediate steps for missing assumptions, circular reasoning, and unsupported inferences, and minimally repairs them before committing to actions or memory.  
With LLaMA-3.1-8B, it reduced the average violations on HotpotQA to 0.37, achieved an 81.36% violation-free rate, and left only 0.05 residual violations after repair while maintaining competitive task performance.  
For example, it corrected unsupported guesses about arena capacity and the confusion between a newspaper’s founding year and the founding year of its owner institute by requiring explicit evidence.

<br/>
# 기타




아래는 본문 설명을 제외하고 **다이어그램·피규어·테이블·어펜딕스 중심으로 본 핵심 결과와 인사이트**입니다.

---

### 1. Figure 1: 잘못된 추론의 예시

* 최종 답변은 정답인 **Animorphs**였지만, 중간 추론에는 근거가 부족한 단계가 포함되어 있습니다.
* 모델은 “The Hork-Bajir Chronicles가 떠오른다”는 연상에서 출발해, 해당 책이 Animorphs의 companion book이라는 전제를 사실상 검증 없이 사용했습니다.
* 즉, **정답을 맞혔다고 해서 추론 과정이 충실한 것은 아님**을 보여줍니다.

**핵심 인사이트:**  
일반적인 QA에서는 최종 정답만 맞으면 문제가 드러나지 않을 수 있지만, 에이전트에서는 이런 검증되지 않은 중간 믿음이 이후 행동이나 메모리에 저장되어 반복적으로 사용될 수 있습니다.

---

### 2. Figure 2: SAVER 전체 구조

SAVER는 다음의 폐쇄형 흐름으로 구성됩니다.

1. 여러 persona를 이용해 서로 다른 추론 후보 생성
2. 구조적 다양성을 고려해 후보 선택
3. 선택된 추론을 adversarial audit으로 검사
4. 오류가 난 특정 부분만 최소 수정
5. 재검사 후 문제가 없을 때 행동 실행 또는 메모리 저장

감사 대상 오류는 다음과 같이 분류됩니다.

* Missing Assumption
* Invalid Precondition
* Unjustified Inference
* Circular Reasoning
* Contradiction
* Overgeneralization

**핵심 인사이트:**  
SAVER의 핵심은 단순히 여러 답을 생성해 다수결로 고르는 것이 아니라, **중간 추론의 어느 단계가 왜 문제인지 찾아내고, 해당 부분만 검증 가능한 조건에 맞게 수정하는 것**입니다.

---

### 3. Figure 3: Audit–Repair 반복 과정

* SAVER와 MAD의 반복 과정을 비교한 결과입니다.
* 세 개의 multi-hop QA 데이터셋을 포함해 전반적으로 SAVER의 **Unfaithful Step Rate(USR)**가 더 빠르고 안정적으로 감소합니다.
* MAD도 반복 토론을 통해 오류를 줄이지만, 여러 번 토론한 뒤에도 검증되지 않은 추론 단계가 상당 부분 남습니다.
* SAVER는 소수의 반복만으로 낮은 비충실 추론 수준에 도달하는 경향을 보입니다.

**핵심 인사이트:**  
추론의 일관성을 높이는 토론보다, 오류를 특정 단계에 국소화하고 그 단계만 고치는 방식이 **faithfulness 개선에 더 직접적**입니다.

---

### 4. Figure 4: 수치 추정 오류 사례

질문은 Lewiston Maineiacs의 홈 경기장 좌석 수를 묻는 문제입니다.

* 한 후보는 유사한 하키 경기장의 규모를 바탕으로 “3,500–4,000명 정도이므로 3,700명”이라고 추정했습니다.
* 다른 후보는 경기장과 좌석 수의 관계를 명시적으로 검증하지 않았습니다.
* SAVER는 이를 각각:

  * **Unjustified Inference**
  * **Invalid Precondition**
으로 판정했습니다.
* 이후 추정값을 삭제하고, 실제 근거 문장에서 좌석 수를 추출하도록 수정했습니다.
* 최종적으로 **3,677 seated**라는 근거 기반 답변만 실행·메모리에 저장했습니다.

**핵심 인사이트:**  
그럴듯한 수치 추정은 정답과 가까울 수 있어도 faithfulness를 보장하지 않습니다. SAVER는 숫자가 반드시 **검색 결과나 도구 출력에 직접 연결되도록** 강제합니다.

---

## 테이블별 결과와 인사이트

### 5. Table 1: 최종 과제 성능

세 가지 backbone 모델과 여섯 개 데이터셋에서 SAVER의 EM/F1 성능을 비교했습니다.

주요 결과:

* LLaMA-3.1-8B:

  * HotpotQA: **43.7 EM / 52.6 F1**
  * 2WikiMHQA: **47.7 / 55.5**
  * MuSiQue: **31.8 / 42.5**
* Qwen-2.5-7B:

  * HotpotQA: **43.1 / 51.2**
  * 2WikiMHQA: **47.7 / 55.8**
* LLaMA-3.2-3B에서도 대부분의 데이터셋에서 기준선보다 높은 성능을 보였습니다.
* NQ, Quoref, FEVER 같은 evidence-sensitive 또는 local reasoning 과제에서도 경쟁력 있는 결과를 유지했습니다.

**인사이트:**  
SAVER는 추론 검증을 강화하면서도 최종 답변 성능을 크게 희생하지 않았습니다. 즉, faithfulness 개선이 단순한 정확도 저하를 대가로 얻어진 것은 아닙니다.

---

### 6. Table 2: LLaMA-3.1-8B에서의 추론 충실성

Multi-hop QA에서 SAVER는 모든 주요 지표에서 가장 좋은 결과를 보였습니다.

|데이터셋|Avg Viol|VFR|Post-Res|USR|
|-|-:|-:|-:|-:|
|HotpotQA|0.37|81.36%|0.05|9.12%|
|2WikiMHQA|0.56|72.34%|0.08|13.84%|
|MuSiQue|0.83|69.38%|0.11|19.73%|

비교하면 MAD는 HotpotQA에서 Avg Viol 1.33, VFR 36.74%, USR 23.94%였습니다.

**인사이트:**

* 위반 횟수가 크게 감소했습니다.
* 오류가 전혀 없는 trajectory 비율이 높아졌습니다.
* repair 후 남은 오류(Post-Res)도 매우 낮았습니다.
* 최종 정답률보다 **중간 추론의 품질을 직접 개선했다는 점**이 중요합니다.

---

### 7. Table 3: LLaMA-3.1-8B Ablation Study

각 구성요소를 제거하면 최종 EM/F1 변화는 크지 않지만, 추론 충실성은 뚜렷하게 악화됩니다.

|설정|HotpotQA Avg Viol|VFR|USR|
|-|-:|-:|-:|
|SAVER|0.37|81.36%|9.12%|
|w/o Persona|0.49|74.55%|11.97%|
|w/o k-DPP|0.64|71.47%|15.86%|
|w/o Auditing|1.37|42.65%|26.74%|
|w/o Repair|1.56|33.68%|37.63%|

**인사이트:**

* Persona generation은 다양한 오류 유형을 드러내는 데 기여합니다.
* k-DPP는 비슷한 실패 패턴을 반복적으로 선택하지 않도록 합니다.
* Auditing은 오류를 발견하는 핵심 모듈입니다.
* Repair는 발견된 오류를 실제로 제거하는 핵심 모듈입니다.
* 특히 auditing이나 repair를 제거했을 때 성능이 크게 악화되어, 두 단계가 모두 필수적임을 보여줍니다.

---

### 8. Table 4: 데이터셋 구성

총 여섯 개 벤치마크를 세 가지 추론 환경으로 나누었습니다.

* **Multi-hop QA:** HotpotQA, 2WikiMHQA, MuSiQue
* **Evidence-sensitive QA:** NQ, FEVER
* **Local reasoning:** Quoref

데이터셋별 역할은 다음과 같습니다.

* HotpotQA: 여러 문서 간 bridge 및 comparison reasoning
* 2WikiMHQA: entity bridging, 비교, 조합 추론
* MuSiQue: 단계적 compositional reasoning과 entity/attribute binding
* NQ: 자연어 질문의 모호성 및 숨은 가정
* FEVER: 증거가 claim을 실제로 지지하는지 검증
* Quoref: 단일 문맥 내 coreference 및 referential reasoning

**인사이트:**  
SAVER는 단순한 multi-hop QA뿐 아니라, 증거 충분성·모호성·대명사 연결 등 서로 다른 종류의 faithfulness 문제를 평가하도록 설계되었습니다.

---

### 9. Table 5: Qwen-2.5-7B에서의 추가 검증

Qwen에서도 동일한 경향이 나타났습니다.

|데이터셋|Avg Viol|VFR|Post-Res|USR|
|-|-:|-:|-:|-:|
|HotpotQA|0.31|84.20%|0.03|7.78%|
|2WikiMHQA|0.49|75.64%|0.06|11.20%|
|MuSiQue|0.71|72.28%|0.09|15.60%|

**인사이트:**  
SAVER의 효과가 특정 모델 하나에만 의존하지 않고, 다른 backbone에서도 재현됩니다.

---

### 10. Table 6: 계산 비용

SAVER는 faithfulness를 높이는 대신 상당한 계산 비용을 사용합니다.

LLaMA-3.1-8B 기준:

* Vanilla LM: 2.3k tokens, 2.2초, 1회 호출
* CoT: 2.8k tokens, 3.1초, 1회 호출
* MAD: 11.2k tokens, 11.5초, 6회 호출
* SAVER: **12.4k tokens, 13.4초, 8회 호출**

Qwen-2.5-7B에서도 약 12.8k tokens, 12.4초, 8회 호출이 필요합니다.

**인사이트:**

* SAVER는 단일 추론 방식보다 훨씬 비쌉니다.
* 그러나 MAD나 Self-Refine과 비슷한 multi-call 비용 수준에서 더 큰 faithfulness 개선을 보입니다.
* 짧고 단순한 작업에는 과도할 수 있으며, 향후에는 불확실성이나 과제 난이도에 따라 audit 깊이를 조정할 필요가 있습니다.

---

### 11. Table 7: QA 이외의 과제

SAVER는 수학 추론과 코드 생성에서도 좋은 결과를 보였습니다.

최종 성능:

* GSM8K: **88.4 EM**
* MATH: **58.2 EM**
* HumanEval: **76.5 Pass@1**
* MBPP: **76.1 Pass@1**

추론 충실성도 개선되었습니다.

* GSM8K: Avg Viol 0.24, VFR 83.44%, USR 6.53%
* HumanEval: Avg Viol 0.12, VFR 88.46%, Post-Res 0.02, USR 4.88%

**인사이트:**  
SAVER의 audit–repair 구조는 지식형 QA에만 국한되지 않고, 계산 오류나 코드 생성 과정의 잘못된 추론에도 적용될 가능성을 보여줍니다.

---

### 12. Table 8: Qwen-2.5-7B Ablation Study

Qwen에서도 Table 3과 동일한 결과가 확인되었습니다.

* Persona 제거: 오류와 USR 증가
* k-DPP 제거: 유사한 오류의 반복 증가
* Auditing 제거: Avg Viol과 USR 큰 폭 증가
* Repair 제거: 가장 심각한 faithfulness 악화

SAVER는 HotpotQA에서:

* Avg Viol: **0.32**
* VFR: **84.71%**
* Post-Res: **0.03**
* USR: **7.54%**

**인사이트:**  
각 모듈의 역할이 모델 backbone이 달라져도 일관되게 유지됩니다.

---

## 어펜딕스별 핵심 내용

### Appendix A: Violation Types와 Audit Schema

오류를 여섯 가지 유형으로 표준화합니다.

1. **Missing Assumption:** 필요한 가정이 드러나지 않음
2. **Invalid Precondition:** 검증되지 않은 조건이나 잘못된 entity에 의존
3. **Unjustified Inference:** 증거 없이 결론을 도출
4. **Circular Reasoning:** 결론을 다시 결론의 근거로 사용
5. **Contradiction:** 기존 사실이나 증거와 충돌
6. **Overgeneralization:** 제한된 증거를 더 넓은 경우에 부당하게 적용

각 오류는 다음 정보를 포함합니다.

* 오류가 발생한 step index
* 문제가 되는 최소 reasoning slice
* 진단 probe
* 오류 유형
* 해결 여부를 확인하는 acceptance criterion

**핵심 인사이트:**  
자유 형식의 “이 추론은 이상하다”라는 비평을 피하고, 오류를 **위치·유형·검증 조건**으로 구조화합니다.

---

### Appendix B: Acceptance Criteria

수정된 추론이 통과하려면 각 오류별 acceptance criterion을 만족해야 합니다.

예시:

* 필요한 증거를 명시적으로 인용해야 함
* 숨은 가정을 분명하게 밝혀야 함
* entity identity나 factual precondition을 검증해야 함
* 순환성이나 모순을 제거해야 함

어떤 trajectory라도 하나의 조건을 만족하지 못하면 행동이나 메모리에 저장되지 않습니다.

**핵심 인사이트:**  
수정의 기준을 문체나 설득력에 두지 않고, **검사 가능한 조건**에 둔다는 점이 SAVER의 중요한 특징입니다.

---

### Appendix C: 실제 Prompt

세 가지 prompt가 제시됩니다.

* Persona Generation Prompt
* Auditor Prompt
* Repair Prompt

특히 repair prompt는 다음을 요구합니다.

* 오류가 표시된 부분만 수정
* 정상적인 단계는 변경하지 않음
* 새로운 가정은 꼭 필요한 경우에만 추가
* 새 증거나 계산은 사용 가능한 evidence와 연결
* 수정 후 최종 주장이 추론 과정에서 논리적으로 따라와야 함

**인사이트:**  
SAVER는 단순히 “다시 풀어라”라고 지시하지 않습니다. 생성, 감사, 수정의 역할을 분리하여 **오류 진단과 수정 사이의 연결성**을 유지합니다.

---

### Appendix D: 데이터셋 설명

각 데이터셋이 어떤 종류의 faithfulness 문제를 유발하는지 설명합니다.

* multi-hop 데이터셋: hop dependency와 문서 간 evidence binding
* NQ: 질문의 모호성과 숨은 가정
* FEVER: claim을 실제로 뒷받침하는 증거의 충분성
* Quoref: referential dependency와 coreference

**인사이트:**  
평가 대상은 단순한 “답이 맞는가”가 아니라, **답에 도달하는 과정에서 올바른 entity·증거·속성을 계속 유지하는가**입니다.

---

### Appendix E: Faithfulness Metric 계산

주요 지표의 계산 방식은 다음과 같습니다.

* **Avg Viol:** trajectory당 평균 오류 수
* **VFR:** 오류가 하나도 없는 trajectory 비율
* **Post-Res:** repair 후 남은 평균 오류 수
* **USR:** 오류가 표시된 reasoning step의 평균 비율

여러 오류가 한 step에 있어도 USR에서는 그 step을 한 번만 계산합니다. 또한 긴 trajectory가 지나치게 큰 영향을 주지 않도록 trajectory 단위로 평균을 냅니다.

**인사이트:**  
정답률이 아니라 **trajectory 수준의 중간 추론 품질**을 별도로 측정합니다.

---

### Appendix F: 추가 실험

#### F.1 Faithfulness 추가 결과

Qwen에서도 SAVER가 모든 데이터셋에서 낮은 Avg Viol/USR, 높은 VFR을 보였습니다.

#### F.2 계산 비용

높은 faithfulness에는 토큰, 시간, API 호출 증가가 따른다는 점을 확인합니다.

#### F.3 QA 이외의 과제

GSM8K, MATH, HumanEval, MBPP에서 성능과 faithfulness 모두 개선되었습니다.

#### F.4 추가 Ablation

Qwen에서도 persona, k-DPP, auditing, repair 각각이 필요하다는 결과가 재확인되었습니다.

---

### Appendix G: 추가 사례 연구

세 가지 사례가 제시됩니다.

#### 1) 2WikiMHQA

* 시대적 인상에 기반한 일반화
* 생년·사망년을 잘못 계산한 산술 오류
* SAVER는 이를 각각 Overgeneralization과 Contradiction으로 분류하고 수정
* 최종 답: **Giuseppe Cesari**

#### 2) MuSiQue

* “The Collegian”이라는 이름을 잘못된 기관에 연결
* 신문의 창간 연도를 소유 기관의 창립 연도로 잘못 사용
* SAVER는 entity binding과 attribute binding을 다시 검증
* 최종 답: **1960**

#### 3) NQ

* “allegiance”라는 표현을 formal name의 근거로 착각
* 미국이라는 관할권을 명시적으로 검증하지 않음
* SAVER는 순환적 근거와 숨은 가정을 수정
* 최종 답: **United States Oath of Allegiance**

**종합 인사이트:**  
세 사례 모두 최종 답변 자체는 그럴듯할 수 있지만, SAVER는 답변보다 먼저 **근거, entity, 조건, 계산 과정**을 검증합니다.

---

## 전체적으로 얻을 수 있는 결론

1. **정답률만으로 agent reasoning의 품질을 평가할 수 없습니다.**
2. 여러 추론의 합의는 동일한 숨은 가정을 공유할 수 있으므로 faithfulness를 보장하지 않습니다.
3. 구조적 다양성을 가진 후보 생성은 서로 다른 실패 유형을 노출하는 데 도움이 됩니다.
4. 가장 중요한 단계는 auditing과 constraint-guided repair입니다.
5. SAVER는 최종 성능을 유지하면서 중간 추론 오류를 크게 줄였지만, 그 대가로 계산 비용이 증가합니다.
6. 따라서 실제 시스템에서는 모든 요청에 SAVER를 적용하기보다, **고위험·장기 추론·메모리 저장·외부 행동이 필요한 상황에 선택적으로 적용하는 방식**이 적절합니다.

---

# 

## Figures, Tables, and Appendices: Main Results and Insights

### 1. Figure 1: Example of Unfaithful Reasoning

The agent gives the correct answer, **Animorphs**, but part of its reasoning is unsupported.

It moves from the vague association that “The Hork-Bajir Chronicles” comes to mind to the assumption that this book is a companion book of Animorphs, without independently verifying the intermediate claim.

**Key insight:**  
A correct final answer does not necessarily imply faithful reasoning. In an agent, unsupported intermediate beliefs may later influence actions or be stored in memory.

---

### 2. Figure 2: Overall SAVER Framework

SAVER follows a closed-loop process:

1. Generate diverse candidate beliefs using multiple personas
2. Select structurally diverse candidates
3. Audit the selected reasoning trajectories
4. Repair only the localized faulty slices
5. Re-audit before committing to an action or memory

The audit categorizes failures into:

* Missing Assumption
* Invalid Precondition
* Unjustified Inference
* Circular Reasoning
* Contradiction
* Overgeneralization

**Key insight:**  
SAVER does not simply generate multiple answers and choose by majority vote. It identifies **which reasoning step fails and why**, then repairs that step under verifiable conditions.

---

### 3. Figure 3: Audit–Repair Dynamics

Across the evaluated datasets, SAVER reduces the Unfaithful Step Rate more quickly and consistently than MAD.

MAD gradually reduces inconsistencies through repeated debate, but unsupported reasoning steps often remain. SAVER reaches a lower unfaithfulness level with fewer iterations.

**Key insight:**  
Localizing and repairing specific reasoning failures is more effective for faithfulness than relying only on repeated debate or consensus.

---

### 4. Figure 4: Numerical Estimation Case

For the question about the seating capacity of the Lewiston Maineiacs’ home arena:

* One candidate guessed “3,700” from the typical capacity of similar arenas.
* Another candidate identified the relevant arena but did not provide sufficient evidence linking it to a seating capacity.
* SAVER classified these as:

  * **Unjustified Inference**
  * **Invalid Precondition**
* The system replaced the guess with evidence-bound extraction.
* Only the verified answer, **3,677 seated**, was committed and stored.

**Key insight:**  
A plausible numerical estimate is not faithful unless it is explicitly grounded in evidence.

---

## Table-Level Results

### 5. Table 1: End-Task Performance

SAVER remains competitive across three model backbones and six datasets.

For example, with LLaMA-3.1-8B:

* HotpotQA: **43.7 EM / 52.6 F1**
* 2WikiMHQA: **47.7 / 55.5**
* MuSiQue: **31.8 / 42.5**

SAVER also performs competitively on NQ, Quoref, and FEVER.

**Insight:**  
Improving reasoning faithfulness does not require a major sacrifice in final task performance.

---

### 6. Table 2: Faithfulness under LLaMA-3.1-8B

SAVER achieves the best faithfulness results on all three multi-hop QA datasets.

|Dataset|Avg Viol|VFR|Post-Res|USR|
|-|-:|-:|-:|-:|
|HotpotQA|0.37|81.36%|0.05|9.12%|
|2WikiMHQA|0.56|72.34%|0.08|13.84%|
|MuSiQue|0.83|69.38%|0.11|19.73%|

**Insight:**  
SAVER substantially reduces detected violations, increases violation-free trajectories, and leaves very few residual errors after repair.

---

### 7. Table 3: Ablation Study

Removing individual components mainly harms faithfulness rather than final EM/F1.

On HotpotQA:

* Full SAVER: Avg Viol 0.37, VFR 81.36%, USR 9.12%
* Without Persona: Avg Viol 0.49, VFR 74.55%
* Without k-DPP: Avg Viol 0.64, VFR 71.47%
* Without Auditing: Avg Viol 1.37, VFR 42.65%
* Without Repair: Avg Viol 1.56, VFR 33.68%

**Insight:**

* Personas expose diverse failure modes.
* k-DPP reduces redundant, correlated reasoning errors.
* Auditing detects failures.
* Repair removes them.
* Auditing and repair are the most critical components.

---

### 8. Table 4: Dataset Coverage

The six datasets represent three settings:

* **Multi-hop QA:** HotpotQA, 2WikiMHQA, MuSiQue
* **Evidence-sensitive QA:** NQ, FEVER
* **Local reasoning:** Quoref

**Insight:**  
The evaluation covers evidence binding, entity disambiguation, hidden assumptions, claim verification, and coreference—not just final answer accuracy.

---

### 9. Table 5: Additional Results with Qwen-2.5-7B

SAVER shows the same pattern with Qwen:

* HotpotQA: Avg Viol 0.31, VFR 84.20%, USR 7.78%
* 2WikiMHQA: Avg Viol 0.49, VFR 75.64%, USR 11.20%
* MuSiQue: Avg Viol 0.71, VFR 72.28%, USR 15.60%

**Insight:**  
The gains are not limited to a single backbone model.

---

### 10. Table 6: Computational Cost

SAVER is more expensive than single-pass methods.

With LLaMA-3.1-8B:

* Vanilla LM: 2.3k tokens, 2.2 seconds, 1 API call
* CoT: 2.8k tokens, 3.1 seconds, 1 call
* MAD: 11.2k tokens, 11.5 seconds, 6 calls
* SAVER: **12.4k tokens, 13.4 seconds, 8 calls**

**Insight:**  
SAVER provides a favorable faithfulness–cost trade-off compared with other multi-call methods, but it may be excessive for simple tasks. Adaptive auditing depth is an important direction for future work.

---

### 11. Table 7: Non-QA Tasks

SAVER also performs well on mathematics and code generation:

* GSM8K: **88.4 EM**
* MATH: **58.2 EM**
* HumanEval: **76.5 Pass@1**
* MBPP: **76.1 Pass@1**

Faithfulness also improves:

* GSM8K: Avg Viol 0.24, VFR 83.44%, USR 6.53%
* HumanEval: Avg Viol 0.12, VFR 88.46%, Post-Res 0.02, USR 4.88%

**Insight:**  
The audit–repair framework may generalize beyond knowledge-intensive QA to arithmetic reasoning and program synthesis.

---

### 12. Table 8: Qwen Ablation Study

The Qwen results confirm the findings from Table 3:

* Removing personas increases errors.
* Removing k-DPP increases correlated failures.
* Removing auditing or repair causes a much larger faithfulness degradation.

**Insight:**  
The contribution of each component remains stable across different backbone models.

---

## Appendix Highlights

### Appendix A: Violation Types and Audit Schema

The appendix formalizes six violation types and represents every error with:

* Step index
* Failing reasoning slice
* Diagnostic probe
* Violation type
* Acceptance criterion

**Insight:**  
The framework converts vague criticism into structured, auditable diagnostics.

---

### Appendix B: Acceptance Criteria

A repaired trajectory must satisfy explicit conditions, such as:

* Citing supporting evidence
* Stating implicit assumptions
* Verifying entity identity or factual preconditions
* Removing circularity or contradiction

**Insight:**  
Repair is judged by checkable conditions rather than fluency or plausibility.

---

### Appendix C: Prompts

The appendix provides separate prompts for:

* Persona-based belief generation
* Reasoning auditing
* Minimal reasoning repair

The repair prompt specifically prohibits changing unaffected reasoning steps.

**Insight:**  
SAVER preserves the causal connection between diagnosis and correction instead of asking the model to regenerate the entire solution.

---

### Appendix D: Dataset Descriptions

The datasets test different failure sources:

* Multi-hop dependency and evidence binding
* Ambiguous questions and hidden assumptions
* Evidence sufficiency
* Coreference and referential reasoning

**Insight:**  
Faithfulness requires maintaining the correct entities, evidence, attributes, and dependencies throughout the trajectory.

---

### Appendix E: Metric Definitions

The appendix defines:

* Average Violations
* Violation-Free Rate
* Post-Repair Residual
* Unfaithful Step Rate

The metrics are computed at the trajectory level, and multiple violations on the same step count as one unfaithful step for USR.

**Insight:**  
The evaluation directly measures intermediate reasoning quality rather than relying only on final accuracy.

---

### Appendix F: Additional Experiments

The appendix confirms:

* Robustness across model backbones
* Higher computational cost
* Generalization to math and code
* The importance of all SAVER components

---

### Appendix G: Additional Case Studies

Three cases illustrate typical failures:

1. **2WikiMHQA:** era-based generalization and incorrect lifespan arithmetic

   * Final answer: **Giuseppe Cesari**
2. **MuSiQue:** confusing a newspaper’s founding year with the owner institute’s founding year

   * Final answer: **1960**
3. **NQ:** circular paraphrasing and an unsupported jurisdiction assumption

   * Final answer: **United States Oath of Allegiance**

**Overall insight:**  
SAVER verifies the evidence, entity, preconditions, and computations before committing a belief—not merely whether the final answer looks plausible.

---

## Overall Takeaway

SAVER’s main contribution is to treat an agent’s intermediate reasoning as a belief state that must be verified before action or memory update.

Its results suggest that:

1. Final accuracy alone is insufficient to evaluate agent reasoning.
2. Consensus does not guarantee faithfulness.
3. Diverse candidate generation helps expose different failure modes.
4. Auditing and constraint-guided repair are the most important components.
5. The method improves faithfulness while preserving competitive task performance.
6. Its computational cost suggests selective use for high-risk, long-horizon, memory-writing, or externally acting scenarios.

<br/>
# refer format:
### BibTeX



```bibtex
@inproceedings{yuan2026verify,
  author    = {Yuan, Wenhao and Lin, Chenchen and Chen, Jian and Xu, Jinfeng and Wang, Xuehe and Ngai, Edith Cheuk Han},
  title     = {Verify Before You Commit: Towards Faithful Reasoning in {LLM} Agents via Self-Auditing},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {31201--31225},
  year      = {2026},
  month     = jul,
  publisher = {Association for Computational Linguistics},
  note      = {Presented July 2--7, 2026}
}
```

### 시카고 스타일   

Yuan, Wenhao, Chenchen Lin, Jian Chen, Jinfeng Xu, Xuehe Wang, and Edith Cheuk Han Ngai. “Verify Before You Commit: Towards Faithful Reasoning in LLM Agents via Self-Auditing.” In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 31201–31225. Association for Computational Linguistics, 2026.

### 

