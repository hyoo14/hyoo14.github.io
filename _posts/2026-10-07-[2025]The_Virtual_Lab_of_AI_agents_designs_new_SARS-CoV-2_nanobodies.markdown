---
layout: post
title:  "[2025]The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies"
date:   2026-10-07 18:20:36 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 역할을 부여받은(PI, 각 분야 학자들) 에이전트들의 오케스트레이션으로 사스 안티바디 후보 발견? 개발?   


짧은 요약(Abstract) :


이 논문은 여러 분야의 AI 연구자 에이전트가 협력하는 **‘Virtual Lab’**이라는 시스템을 소개합니다. 이 시스템에는 연구책임자(PI) 역할의 AI가 있고, 면역학자·머신러닝 전문가·계산생물학자 역할을 하는 AI 에이전트들이 팀을 이루어 연구 문제를 논의합니다. 인간 연구자는 연구 목표와 방향에 대해 높은 수준의 지침을 제공합니다.

연구진은 Virtual Lab을 이용해 SARS-CoV-2의 최신 변이에 결합할 수 있는 **나노바디(nanobody)**를 설계했습니다. AI 에이전트들은 단백질 언어 모델 **ESM**, 단백질 구조 예측 모델 **AlphaFold-Multimer**, 분자 설계 프로그램 **Rosetta**를 결합한 설계 과정을 만들었습니다. 이를 통해 기존 나노바디를 변형하여 총 **92개의 새로운 후보 나노바디**를 제작했습니다.

실험 결과, 대부분의 후보가 안정적으로 발현되었으며, 일부는 SARS-CoV-2의 기존 우한형 스파이크 단백질에 대한 결합력을 유지하면서 최신 변이인 **JN.1 또는 KP.3**에 더 잘 결합했습니다. 특히 두 후보 나노바디는 기존 변이와 최신 변이를 함께 인식할 가능성을 보여 주어, 앞으로 추가 개발할 가치가 있는 후보로 평가되었습니다.

즉, 이 연구는 AI가 단순히 과학 질문에 답하는 것을 넘어, 인간 연구자와 협력해 **가설 설정부터 계산 설계, 실험 후보 선정까지 포함하는 복잡한 생명과학 연구 과정**을 수행할 수 있음을 보여줍니다.

---



This paper introduces the **Virtual Lab**, a system in which multiple AI agents with different scientific roles work together as an interdisciplinary research team. An AI Principal Investigator coordinates agents acting as immunologists, machine-learning specialists and computational biologists, while a human researcher provides high-level guidance.

The researchers applied the Virtual Lab to design **nanobodies** that could bind to recent SARS-CoV-2 variants. The system developed a computational design pipeline combining the protein language model **ESM**, the protein-structure prediction model **AlphaFold-Multimer**, and the computational biology software **Rosetta**. Using this pipeline, it designed **92 new nanobody candidates** by modifying existing nanobodies.

Experimental testing showed that most candidates were expressed successfully. Two candidates were particularly promising: they maintained strong binding to the ancestral Wuhan spike protein while showing improved binding to the newer **JN.1 or KP.3** variants.

Overall, the study demonstrates that AI agents can do more than answer individual scientific questions. In collaboration with a human researcher, they can help carry out a complex, interdisciplinary research project—from selecting methods and designing a computational workflow to proposing experimentally testable biological candidates.


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


### 1. 전체 방법: 인간–다중 AI 에이전트 협업 구조
이 논문은 **Virtual Lab**이라는 인간–AI 협업 프레임워크를 제안했다. 하나의 LLM이 모든 작업을 수행하는 대신, 여러 역할을 가진 AI 에이전트가 연구팀처럼 상호작용한다.

- **Principal Investigator(PI) 에이전트**: 연구 방향 설정, 에이전트 구성, 논의 종합 및 최종 의사결정
- **Immunologist**: 항체·나노바디 및 바이러스 변이 관련 생물학적 판단
- **Machine Learning Specialist**: 단백질 언어모델과 계산 모델 구현
- **Computational Biologist**: 구조 예측, 결합 분석, Rosetta 계산
- **Scientific Critic**: 다른 에이전트의 오류·한계·근거 부족을 검토
- **Human researcher**: 연구 목표, 제약 조건, 회의 의제 및 최종 실행을 감독

각 에이전트는 다음 네 가지 정보로 정의되었다.

1. **Title**: 역할 이름  
2. **Expertise**: 전문 분야  
3. **Goal**: 연구에서 달성할 목표  
4. **Role**: 구체적인 담당 업무  

PI 에이전트는 연구 주제에 대한 짧은 설명을 바탕으로 필요한 과학자 에이전트를 자동으로 구성했다.

---

### 2. 회의 기반의 특별한 아키텍처
Virtual Lab은 두 종류의 회의를 사용한다.

#### 팀 회의
모든 에이전트가 broad한 연구 문제를 논의한다.

1. 인간 연구자가 의제와 질문을 제시
2. PI가 초기 방향과 질문을 제시
3. 각 과학자 에이전트가 자신의 관점에서 답변
4. Scientific Critic이 오류와 문제점을 지적
5. PI가 논의를 종합하고 추가 질문을 제시
6. 여러 라운드 후 PI가 최종 결론과 연구 방향을 정리

#### 개별 회의
특정 에이전트가 코드 작성이나 계산 방법 설계 같은 세부 작업을 수행한다. 필요하면 Scientific Critic이 피드백을 제공하고, 에이전트가 답변이나 코드를 수정한다.

#### 병렬 회의와 답변 통합
같은 회의를 여러 번 병렬 실행해 다양한 답변을 만든 뒤, 낮은 무작위성 설정의 통합 회의에서 가장 좋은 내용을 합쳤다.

- 병렬 회의의 temperature: **0.8**
- 통합 회의의 temperature: **0.2**

이는 LLM의 답변 변동성을 활용하면서도 최종 결과의 일관성을 높이기 위한 방법이다.

---

### 3. 사용된 모델과 계산 도구

#### LLM
- 연구 에이전트의 기반 모델: **GPT-4o**
- 논문에서는 `gpt-4o-2024-08-06`을 사용했다.
- 별도의 새로운 LLM을 학습한 것이 아니라, 역할별 프롬프트를 사용해 하나의 사전학습 LLM을 여러 전문 에이전트처럼 작동시켰다.

#### 단백질 설계 파이프라인
AI 에이전트가 선택하고 구현한 핵심 도구는 다음 세 가지다.

1. **ESM**
   - 단백질 언어모델
   - 나노바디 서열의 각 단일 아미노산 치환이 얼마나 자연스럽고 안정적인지를 평가
   - 변이 서열과 입력 서열의 log-likelihood 차이인 **ESM LLR**을 계산
   - 높은 LLR은 모델이 해당 변이 서열을 더 선호한다는 의미
   - 항원과의 결합을 직접 평가하지 않고, 나노바디 서열 자체의 전반적인 품질을 평가한다.

2. **AlphaFold-Multimer**
   - 나노바디와 SARS-CoV-2 spike RBD가 함께 있는 복합체 구조를 예측
   - 결합 인터페이스의 신뢰도를 **AF ipLDDT**로 평가
   - 높은 AF ipLDDT는 예측된 결합 인터페이스가 더 신뢰할 만하다는 뜻이다.

3. **Rosetta**
   - 예측된 나노바디–RBD 복합체의 구조를 relaxation으로 보정
   - 결합 에너지인 **RS dG**를 계산
   - 일반적으로 더 낮고 음수인 값이 더 유리한 결합을 의미한다.

논문에서 이 모델들은 새로 학습되지 않았으며, 기존에 학습된 단백질 모델과 계산 생물학 소프트웨어를 조합해 사용했다. 논문 본문에는 ESM이나 AlphaFold-Multimer의 원래 학습 데이터 전체가 새롭게 공개되거나 재학습된 것으로 보고되지 않았다.

---

### 4. 나노바디 설계 절차

기존 Wuhan 균주 spike에 결합하는 네 개의 나노바디를 출발점으로 사용했다.

- Ty1
- H11-D4
- Nb21
- VHH-72

각 나노바디에 대해 다음 과정을 반복했다.

1. 가능한 단일 아미노산 치환을 ESM으로 평가
2. ESM LLR이 높은 상위 **20개 변이 서열** 선택
3. 각 변이체와 KP.3 RBD의 복합체를 AlphaFold-Multimer로 예측
4. AF ipLDDT 계산
5. Rosetta로 구조 relaxation 및 RS dG 계산
6. 세 지표를 하나의 가중 점수로 통합
7. 상위 **5개 서열**을 다음 라운드의 시작점으로 선택
8. 총 **4라운드** 반복하여 최대 4개의 변이를 갖는 서열 생성

사용한 가중 점수는 다음과 같다.

[
WS = 0.2(ESM LLR) + 0.5(AF ipLDDT) - 0.3(RS dG)
]

- ESM LLR: 서열 품질
- AF ipLDDT: 결합 구조 예측의 신뢰도
- RS dG: 계산된 결합 에너지
- Rosetta 값은 더 음수일수록 좋기 때문에 음의 가중치를 사용했다.

최종적으로 각 출발 나노바디에서 23개씩, 총 **92개의 변이 나노바디**를 선정했다.

---

### 5. 실험 검증
계산으로 설계된 92개 변이체와 4개의 야생형 나노바디를 실험적으로 평가했다.

- E. coli에서 발현 및 수용성 확인
- Wuhan, JN.1, KP.3, KP.2.3, BA.2 및 MERS-CoV RBD에 대한 ELISA 결합 측정
- 92개 중 90% 이상이 발현·용해성 측면에서 양호
- 특히 다음 두 변이체가 주목할 만한 결합 특성을 보였다.
  - **Nb21 I77V/L59E/Q87A/R37Q**: JN.1 및 KP.3 결합 획득 또는 향상
  - **Ty1 V32F/G59D/N54S/F32S**: Wuhan 결합이 개선되고 JN.1 결합 획득

즉, 이 방법은 LLM이 직접 실험을 수행한 것이 아니라, **LLM 에이전트가 계산 파이프라인과 후보 서열을 설계하고 인간 연구자가 계산 실행과 실험 검증을 담당한 구조**이다.

---



### 1. Overall method: human–multi-agent collaboration
The study introduces the **Virtual Lab**, a human–AI collaboration framework in which multiple LLM agents act as an interdisciplinary research team.

- **Principal Investigator (PI)**: Defines the research direction, coordinates the agents, and synthesizes the conclusions
- **Immunologist**: Provides biological and antibody-related reasoning
- **Machine Learning Specialist**: Designs and implements protein-modeling components
- **Computational Biologist**: Handles structure prediction and binding-energy calculations
- **Scientific Critic**: Identifies errors, weaknesses, and unsupported assumptions
- **Human researcher**: Sets the objectives, constraints, agendas, and oversees the final execution

Each agent is specified by four attributes:

1. **Title**
2. **Expertise**
3. **Goal**
4. **Role**

The PI agent automatically creates suitable scientist agents from a short description of the research project.

---

### 2. Meeting-based architecture
The Virtual Lab uses two types of meetings.

#### Team meetings
All agents discuss a broad research question.

1. The human researcher provides the agenda.
2. The PI proposes initial ideas and questions.
3. Each scientist agent responds from its own expertise.
4. The Scientific Critic challenges the responses.
5. The PI synthesizes the discussion and asks follow-up questions.
6. After several rounds, the PI produces the final recommendation.

#### Individual meetings
A single agent performs a specific task, such as writing code or designing a computational workflow. The Scientific Critic may provide feedback, after which the agent revises its answer or code.

#### Parallel meetings
The same meeting can be run several times to generate diverse solutions. A separate low-randomness meeting then merges the best elements.

- Parallel-meeting temperature: **0.8**
- Merging-meeting temperature: **0.2**

This combines creativity with improved consistency.

---

### 3. Models and computational tools

#### LLM
- The agents were powered by **GPT-4o**
- The specific model was `gpt-4o-2024-08-06`
- The authors did not train a new LLM. Instead, they used role-specific prompts to make one pretrained LLM behave as different scientific specialists.

#### Protein-design pipeline

1. **ESM**
   - A protein language model
   - Scores single amino-acid substitutions in nanobody sequences
   - Computes an ESM log-likelihood ratio (**ESM LLR**)
   - A higher LLR indicates that the model prefers the mutant sequence
   - ESM evaluates the intrinsic plausibility or quality of the nanobody sequence and does not directly model antigen binding.

2. **AlphaFold-Multimer**
   - Predicts the structure of the nanobody–spike RBD complex
   - Uses interface pLDDT (**AF ipLDDT**) as a confidence measure for the predicted binding interface
   - Higher AF ipLDDT indicates a more confident predicted interface.

3. **Rosetta**
   - Relaxes the predicted complex structure
   - Computes an estimated binding energy, **RS dG**
   - More negative values generally indicate more favorable binding.

These tools were not retrained for this study. The researchers combined pretrained protein models and established computational biology software into a new workflow.

---

### 4. Nanobody design workflow

The starting nanobodies were four known binders of the ancestral Wuhan spike protein:

- Ty1
- H11-D4
- Nb21
- VHH-72

For each starting nanobody:

1. ESM scored all possible single-point mutations.
2. The top 20 mutants by ESM LLR were retained.
3. AlphaFold-Multimer predicted each mutant–KP.3 RBD complex.
4. AF ipLDDT was calculated.
5. Rosetta relaxed the structures and calculated RS dG.
6. The candidates were ranked using a combined weighted score.
7. The top five sequences were carried forward.
8. This process was repeated for four rounds, allowing up to four mutations.

The weighted score was:

[
WS = 0.2(ESM LLR) + 0.5(AF ipLDDT) - 0.3(RS dG)
]

The negative coefficient for RS dG reflects the fact that more negative Rosetta binding energies are considered better.

Finally, 23 mutants were selected from each of the four parental nanobodies, giving **92 designed nanobodies** in total.

---

### 5. Experimental validation
The 92 designed nanobodies and four wild-type controls were experimentally tested.

- Soluble expression was measured in *E. coli*.
- Binding was assessed by ELISA against Wuhan, JN.1, KP.3, KP.2.3, BA.2, and MERS-CoV RBDs.
- More than 90% of the designs were expressed and soluble.
- Two notable candidates were:
  - **Nb21 I77V/L59E/Q87A/R37Q**: gained or improved binding to JN.1 and KP.3
  - **Ty1 V32F/G59D/N54S/F32S**: improved Wuhan binding and gained moderate JN.1 binding

Thus, the LLM agents did not independently perform the laboratory experiments. Rather, they designed the computational strategy and candidate sequences, while human researchers executed the computation and performed the experimental validation.


<br/>
# Results


### 1. 연구 목표와 비교 대상
이 연구는 최신 SARS-CoV-2 변이체, 특히 **KP.3와 JN.1의 스파이크 RBD에 결합하면서 기존 Wuhan 균주에도 결합하는 나노바디**를 설계하는 것이 목표였다.

논문에서 직접적인 성능 비교에 사용한 기준은 다음과 같다.

- **야생형(wild type) 나노바디**: 기존 나노바디의 성능
- **Virtual Lab이 설계한 변이 나노바디**: 기존 나노바디에 1~4개의 점돌연변이를 추가한 후보
- 시작 나노바디:
  - Ty1
  - H11-D4
  - Nb21
  - VHH-72

ChemCrow, Coscientist, AI Scientist 등은 서론에서 관련 AI 연구 프레임워크로 언급되지만, 이 논문에서 나노바디 성능을 직접 비교한 경쟁 모델은 아니다. 실제 비교는 주로 **변이체와 각 야생형 나노바디 간 비교**로 이루어졌다.

---

### 2. 설계 및 테스트 데이터

#### 계산 설계
각 시작 나노바디에 대해 다음 과정을 네 차례 반복했다.

1. 가능한 단일 아미노산 치환을 ESM으로 평가
2. ESM 점수가 높은 상위 20개 후보 선택
3. AlphaFold-Multimer로 나노바디–KP.3 RBD 복합체 구조 예측
4. Rosetta로 구조 완화 및 결합 에너지 계산
5. 종합 점수로 상위 5개 후보를 다음 라운드에 전달

최종적으로 각 시작 나노바디에서 23개씩, 총 **92개의 변이 나노바디**를 선정했다.

#### 실험 검증 데이터
총 96개 샘플을 평가했다.

- 변이 나노바디 92개
- 야생형 나노바디 4개

결합 실험에는 다음 항원이 사용되었다.

- Wuhan RBD
- JN.1 RBD
- KP.3 RBD
- KP.2.3 RBD
- BA.2 RBD
- MERS-CoV RBD
- BSA 대조군

주요 실험은 대장균에서의 발현·용해성 측정과 ELISA 기반 결합 분석이었다.

---

### 3. 사용한 주요 메트릭

#### ① ESM LLR
ESM이 해당 나노바디 서열이 얼마나 자연스럽고 안정적인지를 평가하는 지표다.

- 값이 높을수록 ESM 관점에서 더 선호되는 서열
- 항원과의 결합을 직접 평가하지는 않음
- 따라서 나노바디 자체의 서열 품질 또는 진화적 적합성을 반영

최종 비교에는 야생형 대비 점수인 **ESM LLRWT**를 사용했다.

#### ② AF ipLDDT
AlphaFold-Multimer가 예측한 나노바디–스파이크 RBD 결합면의 신뢰도다.

- 값이 높을수록 결합 인터페이스 구조 예측이 더 신뢰할 만함
- 결합 친화도와 관련될 수 있지만, 직접적인 실험 결합값은 아님
- 일반적으로 80 이상이면 높은 정확도의 항체–항원 구조 모델 수준으로 해석됨

#### ③ Rosetta RS dG
Rosetta가 계산한 나노바디–RBD 결합 에너지다.

- 더 음수일수록 결합이 유리하다는 의미
- 예: −50은 −40보다 더 강한 결합으로 해석

#### ④ 종합 점수 WS
세 지표를 다음과 같이 결합했다.

[
WS = 0.2(ESM LLR) + 0.5(AF ipLDDT) - 0.3(RS dG)
]

Rosetta 결합 에너지는 음수일수록 좋기 때문에 음의 가중치를 적용했다. 최종 후보 선정 시에는 야생형 기준의 ESM LLRWT를 사용한 WSWT를 이용했다.

---

### 4. 계산 결과

총 92개 변이 나노바디에 대해 다음과 같은 결과를 얻었다.

- **92개 모두 ESM LLR이 양수**
  - ESM은 모든 변이 서열을 해당 야생형보다 선호했다.
- **78개(85%)**
  - 야생형보다 높은 AF ipLDDT
- **32개(35%)**
  - AF ipLDDT가 80 이상
- **60개(65%)**
  - 야생형보다 더 낮고 유리한 RS dG
- **23개(25%)**
  - RS dG가 −50 이하

또한 네 차례의 반복 최적화가 진행될수록 ESM LLR, AF ipLDDT, Rosetta 점수를 종합한 WS가 전반적으로 개선되었다. 다만 이는 주로 계산 모델상의 개선이며, 모든 후보가 실제 실험에서 강한 결합을 보인 것은 아니다.

---

### 5. 발현 및 용해성 결과

변이 때문에 나노바디가 잘못 접히거나 응집되는지를 확인했다.

- 92개 설계 중 **35개(38%)**가 배양액 1 L당 25 mg 이상의 가용성 나노바디를 발현
- **6개(6.5%)**만 5 mg/L 미만
- 논문은 전체적으로 90% 이상이 발현·용해 가능했다고 보고

따라서 Virtual Lab이 제안한 돌연변이는 대체로 나노바디의 구조적 안정성을 크게 훼손하지 않았다.

---

### 6. 결합 실험 결과

#### Wuhan RBD 결합 유지
기존 결합 특성을 유지하는지가 중요한 비교 기준이었다.

- H11-D4 계열과 Nb21 계열에서는 **44/46개(96%)**가 Wuhan RBD 결합을 유지
- VHH-72 계열에서는 **13/22개**가 야생형과 유사한 수준의 Wuhan RBD 결합 유지
- Ty1 계열은 **10/23개**만 Wuhan RBD 결합을 유지해 상대적으로 성능 저하가 컸다

H11-D4 일부 변이에서는 R27C 돌연변이와 관련된 비특이적 결합이 관찰되었다.

---

### 7. 가장 중요한 두 후보

#### ① Nb21 I77V/L59E/Q87A/R37Q
- JN.1 RBD에 새롭게 결합
- KP.3 RBD 결합도 다른 Nb21 변이 및 야생형보다 증가
- MERS-CoV RBD와 BSA에는 비특이적 결합이 관찰되지 않음
- Wuhan RBD EC50:
  - 변이체 약 0.2 ng/mL
  - JN.1 RBD 약 2.0 ng/mL
- 야생형 Nb21은 JN.1에 매우 약한 결합만 보였으므로, 이 변이는 JN.1 결합을 실질적으로 개선한 것으로 해석된다.

#### ② Ty1 V32F/G59D/N54S/F32S
- Wuhan RBD 결합이 야생형보다 개선
- 야생형 Ty1에서는 관찰되지 않던 JN.1 RBD 결합을 획득
- 다만 JN.1 결합은 중간 정도 수준이었다.

이 두 후보는 최신 변이체에 대한 결합을 얻거나 개선하면서 기존 Wuhan RBD 결합도 어느 정도 유지했다는 점에서 가장 유망한 결과로 평가되었다.

---

### 8. 결과의 의미와 한계

#### 의미
- LLM 여러 개가 서로 다른 전문 역할을 수행하는 Virtual Lab이 면역학, 단백질 구조 예측, 머신러닝, 계산생물학을 결합했다.
- 단순한 문헌 질의응답을 넘어 실제 설계 파이프라인과 코드를 만들고, 92개 후보를 실험적으로 검증했다.
- 계산 예측만으로 끝나지 않고 JN.1과 KP.3에 대한 새로운 결합 후보를 찾았다.

#### 한계
- 경쟁 나노바디 설계 모델과의 직접적인 벤치마크는 없다.
- 실험은 주로 ELISA 결합 분석이므로 실제 바이러스 중화능을 직접 입증하지는 않는다.
- ESM은 항원을 고려하지 않으므로 결합 특이성을 직접 예측하지 못한다.
- AlphaFold-Multimer와 Rosetta 점수가 좋아도 실제 결합이 항상 개선되지는 않았다.
- 연구에서 확인된 것은 유망한 후보이며, 치료제나 실제 중화항체로 확정된 것은 아니다.

---




### 1. Goal and comparison setting
The study aimed to design nanobodies that bind recent SARS-CoV-2 variants, especially the **KP.3 and JN.1 spike RBDs**, while retaining binding to the ancestral Wuhan RBD.

The main comparison was between:

- Four wild-type parental nanobodies:
  - Ty1
  - H11-D4
  - Nb21
  - VHH-72
- 92 Virtual Lab-designed mutants containing one to four point mutations

Frameworks such as ChemCrow, Coscientist and AI Scientist were discussed as related approaches, but they were not used as direct experimental competitors. The actual evaluation mainly compared mutant nanobodies with their corresponding wild-type sequences.

---

### 2. Design and test data

For each parental nanobody, the workflow was repeated for four rounds:

1. ESM evaluated possible point mutations.
2. The top 20 mutants were selected by ESM score.
3. AlphaFold-Multimer predicted nanobody–KP.3 RBD structures.
4. Rosetta relaxed the structures and calculated binding energies.
5. The top five candidates were carried into the next round.

The final set contained:

- 23 mutants from each parental nanobody
- **92 mutant nanobodies in total**

Experimental testing included:

- 92 mutants and 4 wild-type nanobodies
- Wuhan, JN.1, KP.3, KP.2.3 and BA.2 RBDs
- MERS-CoV RBD and BSA as specificity or negative controls

The experiments measured soluble expression and ELISA-based antigen binding.

---

### 3. Main metrics

#### ESM LLR
ESM LLR measures how favorable or plausible a mutated protein sequence is according to the protein language model.

- Higher values indicate a sequence preferred by ESM.
- It evaluates the nanobody sequence itself, not antigen binding directly.
- Final comparisons used ESM LLRWT, measured relative to the wild-type sequence.

#### AF ipLDDT
AF ipLDDT is the predicted confidence of the nanobody–RBD interface from AlphaFold-Multimer.

- Higher values indicate a more confident predicted interface.
- It is related to structural confidence, not a direct experimental affinity measurement.
- Values above 80 were considered consistent with high-accuracy antibody–antigen structural models.

#### Rosetta RS dG
RS dG estimates the binding energy of the nanobody–RBD complex.

- More negative values indicate more favorable binding.
- For example, −50 is considered better than −40.

#### Weighted score
The three metrics were combined as:

[
WS = 0.2(ESM LLR) + 0.5(AF ipLDDT) - 0.3(RS dG)
]

The negative coefficient for RS dG accounts for the fact that more negative binding energies are better.

---

### 4. Computational results

Among the 92 designed mutants:

- **All 92 had positive ESM LLR values**
  - ESM preferred every mutant over its corresponding wild type.
- **78 mutants (85%)**
  - Had higher AF ipLDDT than their wild-type counterpart.
- **32 mutants (35%)**
  - Had AF ipLDDT values of at least 80.
- **60 mutants (65%)**
  - Had lower and therefore more favorable RS dG values than wild type.
- **23 mutants (25%)**
  - Had RS dG values of −50 or lower.

Across the four optimization rounds, the combined computational score generally improved. However, computational improvement did not guarantee stronger experimental binding for every candidate.

---

### 5. Expression and solubility

The authors also examined whether the mutations caused misfolding or aggregation.

- **35 of 92 mutants (38%)** produced more than 25 mg/L of soluble periplasmic nanobody.
- Only **6 mutants (6.5%)** produced less than 5 mg/L.
- Overall, more than 90% of the designs were reported to be expressed and soluble.

Thus, most Virtual Lab-designed mutations were structurally tolerated.

---

### 6. Binding results

#### Retention of Wuhan RBD binding

- H11-D4 and Nb21 mutants retained Wuhan RBD binding in **44 of 46 cases (96%)**.
- **13 of 22 VHH-72 mutants** retained Wuhan RBD binding at levels comparable to the wild type.
- Only **10 of 23 Ty1 mutants** retained Wuhan RBD binding, indicating a larger loss of parental binding in this series.

Some H11-D4 mutants containing R27C showed substantial nonspecific binding, potentially due to disulfide-mediated crosslinking.

---

### 7. Two leading candidates

#### Nb21 I77V/L59E/Q87A/R37Q
- Acquired measurable binding to the JN.1 RBD.
- Also showed increased KP.3 binding compared with other Nb21 mutants and the wild type.
- Did not show notable nonspecific binding to MERS-CoV RBD or BSA.
- The measured EC50 values were approximately:
  - 0.2 ng/mL for Wuhan RBD
  - 2.0 ng/mL for JN.1 RBD
- Wild-type Nb21 showed only very weak JN.1 binding, suggesting that the mutant improved JN.1 recognition.

#### Ty1 V32F/G59D/N54S/F32S
- Improved binding to the Wuhan RBD.
- Acquired moderate binding to the JN.1 RBD.
- The unmutated Ty1 nanobody showed no clear JN.1 binding.

These two mutants were the strongest examples of gaining or improving binding to newer variants while retaining activity against the ancestral RBD.

---

### 8. Significance and limitations

#### Significance
- The Virtual Lab combined immunology, machine learning, protein structure prediction and computational biology through multiple specialized LLM agents.
- It generated a complete computational design workflow and produced 92 experimentally tested nanobodies.
- The workflow identified candidates with new or improved binding to JN.1 and KP.3.

#### Limitations
- The study did not include a direct experimental benchmark against competing nanobody-design algorithms.
- The validation mainly used ELISA binding assays and did not directly demonstrate viral neutralization.
- ESM does not explicitly model antigen–nanobody interactions.
- Improved computational scores did not always correspond to improved experimental binding.
- The reported candidates are promising research leads, not yet validated therapeutic nanobodies.


<br/>
# 예제



### 1. 먼저 중요한 점: 이 논문에는 일반적인 “학습 데이터–테스트 데이터” 분할이 없다

이 연구는 새로운 예측 모델을 학습시킨 것이 아니라, **이미 학습된 AI 모델과 계산 도구를 연결해 나노바디를 설계**한 연구이다.

- **GPT-4o**: Virtual Lab의 연구자 에이전트를 구동
- **ESM**: 단백질 서열의 안정성·진화적 적합도를 평가
- **AlphaFold-Multimer**: 나노바디와 SARS-CoV-2 RBD의 복합체 구조 예측
- **Rosetta**: 예측된 복합체를 안정화하고 결합 에너지 계산

따라서 논문에서 머신러닝의 의미로 쓰이는 **별도의 training set과 test set은 제시되지 않았다.**  
대신 다음과 같은 방식으로 후보를 만들고 실험적으로 검증했다.

---

### 2. 전체 과제

**과제(task)**  
기존 SARS-CoV-2 Wuhan 균주에 결합하는 나노바디를 변형하여, 최신 변이인 **KP.3 또는 관련 변이 JN.1에도 결합하는 나노바디**를 설계하는 것.

**초기 입력(input)**

- 기존 나노바디 서열 4개
  - Ty1
  - H11-D4
  - Nb21
  - VHH-72
- SARS-CoV-2 spike 단백질의 RBD 서열
  - 주로 KP.3 RBD를 설계 대상으로 사용

**최종 출력(output)**

- 나노바디 후보 92개
  - 기존 나노바디 4개 × 각 23개
- 각 후보에 대한 계산 점수
  - ESM LLR
  - AlphaFold-Multimer interface pLDDT
  - Rosetta binding energy, RS dG
- 실험적으로 측정한 발현량과 RBD 결합능

---

### 3. Virtual Lab 자체의 입력과 출력

#### 입력

사람 연구자가 Virtual Lab에 제공한 내용은 주로 다음과 같다.

- 연구 목표: KP.3에 결합하는 나노바디 설계
- 회의 주제와 질문
- 사용할 수 있는 계산 자원 및 실험적 제약
- 이전 회의의 요약 정보

각 AI 에이전트에는 다음 네 가지 정보가 지정되었다.

1. **Title**: 면역학자, 머신러닝 전문가, 계산생물학자 등  
2. **Expertise**: 해당 전문 분야  
3. **Goal**: 연구 프로젝트에서 달성할 목표  
4. **Role**: 구체적으로 맡을 역할  

#### 출력

Virtual Lab은 회의를 통해 다음을 결정하거나 생성했다.

- 나노바디를 새로 설계할지, 기존 것을 변형할지
- 사용할 계산 도구
- ESM, AlphaFold-Multimer, Rosetta를 연결하는 파이프라인
- 각 도구를 실행하는 Python 및 Rosetta 코드
- 최종 나노바디 후보와 실험 검증 전략

---

### 4. 계산 파이프라인의 구체적인 입력과 출력

#### 단계 1: ESM을 이용한 돌연변이 후보 생성

**입력**

- 현재 나노바디 서열
- 예: 야생형 Nb21 서열

**작업**

- 서열의 각 위치에 가능한 단일 아미노산 치환을 적용
- 각 돌연변이 서열의 ESM log-likelihood ratio, 즉 **ESM LLR** 계산

**출력**

- 돌연변이 서열별 ESM LLR
- LLR이 높은 상위 20개 돌연변이 서열 선택

ESM은 항원인 spike RBD를 직접 고려하지 않고, 해당 나노바디 서열 자체가 얼마나 자연스럽고 안정적인지를 평가한다.

---

#### 단계 2: AlphaFold-Multimer로 복합체 구조 예측

**입력**

- ESM이 고른 돌연변이 나노바디 서열
- KP.3 spike RBD 서열

예를 들어:

```text
Mutant nanobody: Nb21 I77V
Antigen: KP.3 spike RBD
```

**작업**

- 나노바디–RBD 복합체의 3차원 구조 예측
- 두 단백질의 결합 인터페이스 신뢰도 계산

**출력**

- 예측된 복합체 구조
- AF ipLDDT 점수

AF ipLDDT가 높을수록 예측된 결합 인터페이스에 대한 신뢰도가 높다고 해석했다.

---

#### 단계 3: Rosetta로 결합 에너지 계산

**입력**

- AlphaFold-Multimer가 예측한 나노바디–RBD 복합체 구조, 즉 PDB 파일

**작업**

- 복합체 구조를 relax
- 나노바디와 RBD 사이의 결합 에너지 계산

**출력**

- RS dG 값

RS dG는 일반적으로 더 낮고 더 음수일수록 결합이 유리한 것으로 해석했다.

---

#### 단계 4: 세 점수를 결합해 후보 순위 결정

세 가지 점수를 다음 식으로 결합했다.

[
WS = 0.2 times ESM LLR
+ 0.5 times AF ipLDDT
- 0.3 times RS dG
]

- ESM LLR: 서열 품질
- AF ipLDDT: 결합 구조 예측 신뢰도
- RS dG: 계산된 결합 에너지
- WS: 최종 후보 순위 점수

각 단계에서 상위 5개 서열을 다음 돌연변이 라운드의 출발점으로 삼았다.

---

### 5. 반복 설계 과정

각 시작 나노바디에 대해 총 4라운드의 변이를 수행했다.

#### 예시: Nb21

1. 야생형 Nb21에서 모든 가능한 단일 돌연변이 생성
2. ESM LLR 상위 20개 선택
3. AlphaFold-Multimer와 Rosetta로 평가
4. WS 상위 5개 선택
5. 상위 5개 각각에서 다시 새로운 돌연변이 생성
6. 이 과정을 총 4회 반복
7. 모든 라운드의 결과 중 최종 23개 선택

이 과정을 Ty1, H11-D4, Nb21, VHH-72에 각각 적용하여 총 **92개 후보**를 만들었다.

---

### 6. 실험 검증: 사실상의 테스트 단계

이 연구에서 계산 설계 이후의 실험이 새로운 후보에 대한 **실제 검증 단계**에 해당한다. 다만 논문은 이를 머신러닝의 test set이라고 부르지는 않는다.

#### 실험 입력

- 설계된 나노바디 92개
- 야생형 나노바디 4개
- 여러 항원 RBD
  - Wuhan RBD
  - JN.1 RBD
  - KP.3 RBD
  - KP.2.3 RBD
  - BA.2 RBD
  - 대조군인 MERS-CoV RBD와 BSA

#### 실험 과제

1. 나노바디가 대장균에서 제대로 발현되는가?
2. 수용성 단백질로 생산되는가?
3. Wuhan RBD 결합능을 유지하는가?
4. JN.1 또는 KP.3 RBD에 새롭게 결합하는가?
5. 비특이적 결합은 적은가?

#### 실험 출력

- 수용성 나노바디 발현량
- ELISA 결합 강도
- 농도에 따른 결합 곡선
- EC50
- 변이별 결합 특성

---

### 7. 구체적인 결과 예시

#### 예시 1: Nb21 변이체

```text
Nb21 I77V/L59E/Q87A/R37Q
```

- Wuhan RBD에 강하게 결합
- 야생형 Nb21보다 JN.1 RBD 결합이 개선
- KP.3 RBD에도 결합 증가
- JN.1 RBD의 EC50:
  - 변이체: 약 2.0 ng/ml
  - Wuhan RBD: 약 0.2 ng/ml
- MERS-CoV RBD와 BSA에는 비특이적 결합이 관찰되지 않음

즉, Wuhan 결합능을 어느 정도 유지하면서 JN.1 및 KP.3에 대한 결합 특성이 추가된 사례이다.

#### 예시 2: Ty1 변이체

```text
Ty1 V32F/G59D/N54S/F32S
```

- Wuhan RBD 결합능이 야생형보다 향상
- 야생형 Ty1에서는 거의 보이지 않던 JN.1 RBD 결합을 획득
- JN.1에 대한 결합은 강력한 수준이라기보다 중간 정도로 평가됨

---

### 8. 결과를 한 문장으로 정리하면

이 논문은 **단백질 서열 데이터로 새로운 모델을 학습한 연구라기보다**, 기존에 학습된 ESM·AlphaFold-Multimer와 Rosetta를 Virtual Lab의 AI 에이전트가 조합하여 **92개의 나노바디 후보를 계산적으로 설계하고, 그중 실제로 변이 RBD에 결합하는 후보가 있는지 실험으로 검증한 연구**이다.

---




### 1. Important clarification: there is no conventional training–test split

This study did not train a new machine-learning model using a separate training dataset and test dataset. Instead, it used pre-trained models and computational tools:

- **GPT-4o**: powered the Virtual Lab agents
- **ESM**: scored protein sequence plausibility and mutations
- **AlphaFold-Multimer**: predicted nanobody–RBD complex structures
- **Rosetta**: relaxed the structures and estimated binding energies

Therefore, the paper does **not** report a conventional training set and test set for the nanobody-design models.

---

### 2. Overall task

**Task:**  
Modify nanobodies that bind the ancestral Wuhan SARS-CoV-2 strain so that they can also bind recent variants, especially **KP.3 and JN.1**.

**Initial inputs:**

- Four existing nanobody sequences:
  - Ty1
  - H11-D4
  - Nb21
  - VHH-72
- The SARS-CoV-2 spike RBD sequence, mainly the KP.3 RBD

**Final outputs:**

- 92 designed nanobody candidates
- Computational scores for each candidate:
  - ESM LLR
  - AlphaFold-Multimer interface pLDDT
  - Rosetta binding energy, RS dG
- Experimental expression and binding measurements

---

### 3. Inputs and outputs of the Virtual Lab

#### Inputs

The human researcher provided:

- The research objective
- Meeting agendas and questions
- Computational and experimental constraints
- Summaries of previous meetings

Each AI agent was defined by four fields:

1. **Title**
2. **Expertise**
3. **Goal**
4. **Role**

#### Outputs

The Virtual Lab produced:

- The decision to modify existing nanobodies rather than design them de novo
- The choice of Ty1, H11-D4, Nb21 and VHH-72
- The selection of ESM, AlphaFold-Multimer and Rosetta
- Python and Rosetta scripts
- The final computational workflow
- Candidate nanobody sequences for experimental testing

---

### 4. Concrete computational workflow

#### Step 1: ESM mutation scoring

**Input:**

- A current nanobody sequence, for example the wild-type Nb21 sequence

**Task:**

- Generate possible single amino-acid substitutions
- Calculate the ESM log-likelihood ratio, or **ESM LLR**, for each mutant

**Output:**

- A score for each mutated sequence
- The top 20 sequences according to ESM LLR

ESM evaluates the plausibility and quality of the nanobody sequence itself; it does not directly model binding to the spike RBD.

---

#### Step 2: AlphaFold-Multimer structure prediction

**Input:**

- A mutated nanobody sequence
- The KP.3 spike RBD sequence

Example:

```text
Mutant nanobody: Nb21 I77V
Antigen: KP.3 spike RBD
```

**Task:**

- Predict the nanobody–RBD complex structure
- Estimate the confidence of the binding interface

**Output:**

- A predicted 3D complex structure
- An AF ipLDDT score

A higher AF ipLDDT indicates higher predicted confidence in the interface.

---

#### Step 3: Rosetta binding-energy estimation

**Input:**

- The predicted nanobody–RBD complex structure, provided as a PDB file

**Task:**

- Relax the structure
- Estimate the interaction energy between the nanobody and RBD

**Output:**

- An RS dG score

More negative RS dG values were interpreted as more favorable binding energies.

---

#### Step 4: Combining the scores

The three scores were combined as:

[
WS = 0.2 times ESM LLR
+ 0.5 times AF ipLDDT
- 0.3 times RS dG
]

The top five sequences according to this weighted score were selected for the next round.

---

### 5. Iterative design

For each starting nanobody, the workflow was repeated for four mutation rounds:

1. Generate single mutants from the starting sequence
2. Select the top 20 mutants by ESM LLR
3. Evaluate them with AlphaFold-Multimer and Rosetta
4. Select the top five by the weighted score
5. Introduce another round of mutations
6. Repeat for four rounds
7. Select 23 final candidates

The process was applied independently to the four parental nanobodies, producing:

```text
4 parental nanobodies × 23 candidates = 92 designed nanobodies
```

---

### 6. Experimental validation as the practical test stage

The experiments functioned as the practical validation stage, although the paper does not call them a machine-learning “test set.”

#### Experimental inputs

- 92 designed nanobodies
- Four unmodified parental nanobodies
- A panel of RBD antigens:
  - Wuhan
  - JN.1
  - KP.3
  - KP.2.3
  - BA.2
  - MERS-CoV RBD and BSA as controls

#### Experimental tasks

The researchers measured:

1. Whether the nanobodies were expressed in *E. coli*
2. Whether they were soluble
3. Whether Wuhan RBD binding was preserved
4. Whether binding to JN.1 or KP.3 was gained or improved
5. Whether nonspecific binding occurred

#### Experimental outputs

- Soluble expression titre
- ELISA binding intensity
- Binding curves
- EC50 values
- Variant-specific binding profiles

---

### 7. Concrete examples

#### Example 1: Nb21 mutant

```text
Nb21 I77V/L59E/Q87A/R37Q
```

This mutant:

- Retained strong binding to Wuhan RBD
- Showed improved binding to JN.1 compared with wild-type Nb21
- Also showed increased binding to KP.3
- Had an EC50 of approximately:
  - 2.0 ng/ml for JN.1 RBD
  - 0.2 ng/ml for Wuhan RBD
- Did not show nonspecific binding to MERS-CoV RBD or BSA

#### Example 2: Ty1 mutant

```text
Ty1 V32F/G59D/N54S/F32S
```

This mutant:

- Improved binding to Wuhan RBD
- Gained moderate binding to JN.1 RBD
- Showed binding to JN.1 that was not detectable for the unmodified Ty1 nanobody

---

### 8. One-sentence summary

Rather than training a new model on a labeled training dataset, this study used pre-trained models and AI agents to computationally design 92 nanobody candidates, then experimentally tested whether those candidates could bind recent SARS-CoV-2 variants.

<br/>
# 요약



1. 연구진은 인간 연구자가 방향을 제시하고 GPT-4o 기반의 PI·면역학자·머신러닝 전문가·계산생물학자·비평가 에이전트가 협업하는 Virtual Lab을 구축했다.  
2. 이 시스템은 기존 나노바디 Ty1, H11-D4, Nb21, VHH-72에 ESM, AlphaFold-Multimer, Rosetta를 적용해 변이를 반복 선별하고, SARS-CoV-2 KP.3 결합 후보 92개를 설계했다.  
3. 실험 결과 대부분의 나노바디가 발현·용해성을 유지했으며, 특히 Nb21 변이체는 JN.1 및 KP.3에 대한 결합을, Ty1 변이체는 JN.1에 대한 결합을 새롭게 보여 후속 개발 가능성을 입증했다.  




1. The researchers built a Virtual Lab in which a human guides GPT-4o-based agents acting as a PI, immunologist, machine-learning specialist, computational biologist and scientific critic.  
2. The system iteratively applied ESM, AlphaFold-Multimer and Rosetta to four existing nanobodies—Ty1, H11-D4, Nb21 and VHH-72—to design 92 candidates targeting the SARS-CoV-2 KP.3 variant.  
3. Experimental tests showed that most candidates remained soluble and functional, with an Nb21 mutant gaining binding to JN.1 and KP.3 and a Ty1 mutant gaining binding to JN.1, highlighting their potential for further development.

<br/>
# 기타



### 1. Figure 1 — Virtual Lab 아키텍처

**결과**
- 인간 연구자가 연구 의제와 고수준 방향을 제시한다.
- PI(Principal Investigator) 에이전트가 연구 주제에 맞춰 면역학자, 머신러닝 전문가, 계산생물학자 등 과학자 에이전트를 구성한다.
- Scientific Critic 에이전트가 각 답변의 오류와 한계를 지적한다.
- 연구는 **팀 미팅**과 **개별 미팅**으로 진행된다.
  - 팀 미팅: 여러 분야의 에이전트가 광범위한 연구 방향을 논의
  - 개별 미팅: 특정 에이전트가 코드 작성이나 도구 구현 같은 세부 과제를 수행

**인사이트**
- 이 시스템의 핵심은 단일 LLM에게 한 번 질문하는 것이 아니라, 역할이 다른 여러 에이전트가 반복적으로 토론하고 비판하도록 만든 점이다.
- 인간은 모든 세부 작업을 직접 수행하기보다 연구 목표, 제약조건, 최종 판단을 담당한다.

---

### 2. Figure 2 — 나노바디 설계 연구의 5단계

**결과**
Virtual Lab은 다음 다섯 단계로 설계 파이프라인을 구축했다.

1. **팀 구성**: 면역학자, 머신러닝 전문가, 계산생물학자 선정  
2. **프로젝트 구체화**: 새로운 나노바디를 처음부터 만들기보다 기존 나노바디를 수정하기로 결정  
3. **도구 선정**: ESM, AlphaFold-Multimer, Rosetta 선택  
4. **도구 구현**: 각 도구를 실행할 Python 및 Rosetta 스크립트 작성  
5. **워크플로 설계**: 세 도구의 실행 순서와 점수 통합 방식 결정

**인사이트**
- AI가 단순히 기존 도구를 호출한 것이 아니라, 어떤 도구를 선택하고 어떻게 연결할지까지 결정했다.
- 기존 나노바디인 **Ty1, H11-D4, Nb21, VHH-72**를 출발점으로 삼아 시간과 실패 가능성을 줄였다.

---

### 3. Figure 3 및 Extended Data Figures 2–4 — 계산적 나노바디 최적화

#### Figure 3: Nb21 사례

**결과**
- ESM이 가능한 단일 아미노산 변이를 평가하고, 상위 20개를 선택한다.
- AlphaFold-Multimer가 나노바디–스파이크 복합체 구조와 계면 신뢰도(AF ipLDDT)를 예측한다.
- Rosetta가 구조를 완화하고 결합 에너지(RS dG)를 계산한다.
- 세 지표를 다음의 가중 점수로 통합했다.

[
WS = 0.2(ESM LLR) + 0.5(AF ipLDDT) - 0.3(RS dG)
]

- 각 라운드에서 상위 5개를 다음 변이 라운드의 출발점으로 사용했다.
- 총 4라운드의 변이를 거쳐 각 출발 나노바디당 23개, 전체 **92개 후보**를 선정했다.

**인사이트**
- 여러 라운드가 진행될수록 ESM 점수, 구조 계면 신뢰도, Rosetta 결합 에너지 측면에서 전반적인 개선이 관찰됐다.
- ESM은 항원을 직접 고려하지 않고 나노바디 자체의 서열 품질을 평가한다.
- AlphaFold-Multimer와 Rosetta는 KP.3 스파이크와의 결합 가능성을 직접적으로 반영하려는 역할을 한다.
- 따라서 세 도구의 조합은 “잘 접히는 나노바디”와 “표적에 잘 결합할 가능성이 있는 나노바디”를 동시에 찾으려는 전략이다.

#### Extended Data Figures 2–4

- **Extended Data Fig. 2: Ty1**
  - 4라운드의 변이 과정에서 점수 변화가 관찰된다.
  - 최종적으로 JN.1 결합을 획득한 Ty1 변이체가 실험적으로 확인됐다.
- **Extended Data Fig. 3: H11-D4**
  - 계산 지표는 개선됐지만, 일부 변이체는 실험에서 비특이적 결합을 보였다.
  - 계산 점수 향상이 반드시 실험적 특이성 향상으로 이어지지는 않음을 보여준다.
- **Extended Data Fig. 4: VHH-72**
  - 구조 및 계산 점수상 개선 후보들이 만들어졌지만, 가장 두드러진 신규 변이체는 Ty1과 Nb21에서 나타났다.

**전체 계산 결과**
- 92개 변이체 모두 ESM 기준으로 야생형보다 선호됐다.
- 78개(85%)는 야생형보다 높은 AF ipLDDT를 보였다.
- 60개(65%)는 야생형보다 더 좋은 Rosetta 결합 에너지를 보였다.
- 23개(25%)는 RS dG가 −50 이하로 강한 결합 가능성을 보였다.

---

### 4. Figure 4 및 Extended Data Figures 5–6 — 실험적 검증

#### Figure 4: 발현 및 결합 실험

**결과**
- 92개 설계 나노바디와 4개 야생형 나노바디를 실험했다.
- 대부분이 가용성 단백질로 발현됐다.
  - 35/92개(38%)는 배양액 1 L당 25 mg 이상의 가용성 나노바디를 생산
  - 6/92개(6.5%)만 5 mg/L 미만
- Wuhan, JN.1, KP.3, KP.2.3, BA.2 RBD 및 MERS-CoV RBD와 BSA에 대한 ELISA를 수행했다.

#### 주요 후보 1: Nb21 변이체

**Nb21 I77V/L59E/Q87A/R37Q**
- Wuhan RBD 결합을 강하게 유지했다.
- 야생형 Nb21보다 JN.1 RBD 결합이 향상됐다.
- KP.3 RBD에도 다른 Nb21 변이체보다 높은 결합을 보였다.
- JN.1 결합의 EC50은 약 2.0 ng/mL로, Wuhan RBD의 0.2 ng/mL보다 약했다.
- MERS-CoV RBD와 BSA에 대한 비특이적 결합은 관찰되지 않았다.

#### 주요 후보 2: Ty1 변이체

**Ty1 V32F/G59D/N54S/F32S**
- Wuhan RBD 결합이 야생형보다 향상됐다.
- 야생형 Ty1에는 거의 없던 JN.1 RBD 결합을 획득했다.
- 다만 결합 강도는 중간 정도로 평가됐다.

#### 기타 결과

- H11-D4와 Nb21 계열은 대체로 Wuhan 결합을 잘 유지했다.
- Ty1 변이체는 전반적으로 Wuhan 결합을 잃은 경우가 많았다.
- H11-D4 일부 변이체는 R27C 변이와 관련된 것으로 추정되는 비특이적 결합을 보였다.
- VHH-72 변이체의 일부도 Wuhan 결합을 유지했다.

**인사이트**
- 계산 설계의 성공은 단순한 점수 개선이 아니라 실제 발현 가능성과 항원 결합으로 평가됐다.
- 그러나 계산 점수가 좋다고 해서 반드시 특이적 결합이나 원하는 변이체 결합이 보장되지는 않았다.
- 92개 중 특히 2개가 최신 변이체 결합에서 뚜렷한 가능성을 보였다는 점은, 계산적으로 큰 후보군을 줄이는 데 Virtual Lab이 유용했음을 보여준다.
- 다만 이 연구에서 확인한 것은 주로 **ELISA 기반 결합**이며, 실제 바이러스 중화능이나 세포 수준의 방어 효과를 직접 검증한 것은 아니다.

#### Extended Data Fig. 5

- 나노바디 발현, RBD 단백질 제작, 항원 배열, multiplex ELISA로 이어지는 전체 실험 검증 과정을 도식화한다.
- 계산 결과를 실제 단백질 발현과 결합 실험으로 연결한 실험 파이프라인을 보여준다.

#### Extended Data Fig. 6

- 96개 나노바디의 SDS–PAGE 발현 결과를 제시한다.
- 대부분에서 약 15 kDa 크기의 예상 나노바디 밴드가 확인된다.
- 계산 설계에서 도입한 1–4개의 변이가 대규모 misfolding이나 aggregation을 유발하지 않았다는 근거다.

---

### 5. Figure 5 및 Extended Data Figure 7 — 에이전트 상호작용과 인간의 역할

**결과**
- 에이전트들은 서로 다른 전문성을 바탕으로 다른 관점의 의견을 제시했다.
- 전체 연구 과정에서 작성된 단어 수:
  - 인간 연구자: 1,596단어, 약 1.3%
  - LLM 에이전트: 122,462단어, 약 98.7%
- PI는 팀 미팅에서 가장 많은 내용을 작성하며 논의를 종합하고 방향을 정했다.
- Scientific Critic은 다른 에이전트의 답변을 검토하고 문제점을 지적했다.
- 각 전문 분야의 역할이 명확할수록 일반적인 에이전트들만 사용하는 경우보다 논의의 일관성과 품질이 높았다.

**인사이트**
- 인간 연구자는 코드와 세부 논의를 모두 작성하지 않고도 연구 방향을 통제할 수 있었다.
- 하지만 인간이 완전히 배제된 것은 아니다.
  - 연구 목표와 의제 설정
  - 사용 가능한 계산 자원과 도구 선택
  - 코드 실행 및 디버깅 확인
  - 실험 결과 검증
  - 최종 과학적 판단은 인간이 담당했다.
- 즉, 이 시스템은 완전 자율 연구자라기보다 **인간이 감독하는 다중 에이전트 연구팀**에 가깝다.

---

### 6. Extended Data Figure 1 — 병렬 미팅

**결과**
- 동일한 의제와 에이전트를 사용해 여러 회의를 동시에 실행한다.
- 높은 temperature로 다양한 답변을 생성한 뒤, 낮은 temperature의 병합 회의에서 가장 좋은 요소를 통합한다.

**인사이트**
- 한 번의 LLM 응답에 의존하지 않고 여러 답변을 비교함으로써 우연한 오류나 편향을 줄이려는 방식이다.
- 다만 여러 LLM 답변을 병합한다고 해서 과학적 오류가 자동으로 제거되는 것은 아니다.

---

### 7. Extended Data Table 1 — 나노바디 점수표

**내용**
- 야생형 및 일부 변이체에 대해 다음 점수를 제시한다.
  - ESM LLRWT: 야생형 대비 서열 선호도
  - AF ipLDDT: 나노바디–스파이크 계면 구조 예측 신뢰도
  - RS dG: Rosetta 결합 에너지
  - WSWT: 세 지표를 결합한 최종 점수

**인사이트**
- 어떤 후보는 서열 안정성은 높지만 결합 에너지가 상대적으로 약할 수 있고, 반대의 경우도 있다.
- 따라서 단일 지표가 아니라 서로 다른 특성을 반영하는 복합 점수를 사용했다.
- 이 표는 최종 후보 선정이 하나의 기준에 의해 이루어진 것이 아니라, 안정성·구조·결합 에너지 사이의 절충에 기반했음을 보여준다.

---

### 8. Appendix / Methods / Supplementary 내용

논문 본문에서 확인되는 방법론적 핵심은 다음과 같다.

- **Virtual Lab 구현**
  - GPT-4o를 기반으로 PI, 과학자 에이전트, Scientific Critic을 구성
  - 각 에이전트는 Title, Expertise, Goal, Role로 정의
  - 일반적으로 3회의 토론 라운드 사용
- **계산 도구**
  - ESM: 단일 아미노산 변이의 서열 가능성 평가
  - AlphaFold-Multimer: 나노바디–RBD 복합체 구조 예측
  - Rosetta: 구조 완화 및 결합 에너지 계산
- **실험**
  - E. coli에서 나노바디 발현
  - Expi293에서 RBD 발현
  - multiplex ELISA로 변이체별 결합 측정
- **데이터와 코드**
  - 계산 결과와 ELISA 데이터는 Zenodo에 공개
  - Virtual Lab 코드와 에이전트 대화 기록은 GitHub 및 Zenodo에서 제공

**방법론적 한계**
- LLM의 지식 cutoff로 인해 최신 논문이나 도구를 모를 수 있다.
- 적절한 prompt engineering이 없으면 답변이 모호해질 수 있다.
- 에이전트가 잘못된 과학적 사실이나 인용을 만들어낼 가능성이 있다.
- 실험적으로는 결합 여부를 주로 평가했으며, 중화능·생체 내 효능·약동학은 추가 검증이 필요하다.

---

## 핵심 종합 인사이트

1. **Virtual Lab은 연구 아이디어 제안 수준을 넘어 도구 선택, 코드 작성, 파이프라인 설계까지 수행했다.**
2. **92개 후보 중 2개가 JN.1 또는 KP.3에 대한 새로운 결합 특성을 보였다.**
3. **계산 예측은 후보를 효율적으로 좁히는 데 유용했지만, 실험 검증을 대체하지는 못했다.**
4. **역할이 다른 에이전트와 비판 에이전트를 함께 사용한 것이 결과의 다양성과 품질 향상에 기여했다.**
5. **인간 연구자는 적은 텍스트 입력으로도 연구를 이끌었지만, 최종 검증과 과학적 책임은 여전히 인간에게 있었다.**

---




## 1. Figure 1 — Virtual Lab architecture

**Results**
- The human researcher provides the research agenda and high-level guidance.
- The PI agent creates a project-specific team, such as an immunologist, machine-learning specialist and computational biologist.
- A Scientific Critic agent reviews the other agents’ answers for errors and omissions.
- The system uses:
  - **Team meetings** for broad, interdisciplinary decisions
  - **Individual meetings** for specific tasks such as coding and tool implementation

**Insight**
- The main innovation is not simply asking one LLM a question. It is the repeated interaction between agents with different roles, combined with criticism and human oversight.

---

## 2. Figure 2 — Five phases of nanobody design

**Results**
The Virtual Lab developed the workflow in five stages:

1. **Team selection**
2. **Project specification**
3. **Tool selection**
4. **Tool implementation**
5. **Workflow design**

The system decided to modify four existing nanobodies—**Ty1, H11-D4, Nb21 and VHH-72**—rather than design new nanobodies completely de novo. It selected **ESM, AlphaFold-Multimer and Rosetta** as the main computational tools.

**Insight**
- The AI agents did not merely use preselected software. They helped decide which tools to use, how to connect them and how to score candidates.
- Starting from known nanobodies reduced the design space and likely improved the feasibility of the project.

---

## 3. Figure 3 and Extended Data Figures 2–4 — Computational optimization

### Figure 3: Nb21 example

**Results**
- ESM evaluated possible point mutations.
- The top 20 candidates were structurally assessed with AlphaFold-Multimer.
- Rosetta then relaxed the structures and estimated binding energy.
- Candidates were ranked using:

[
WS = 0.2(ESM LLR) + 0.5(AF ipLDDT) - 0.3(RS dG)
]

- The top five candidates were carried into the next mutation round.
- Four rounds of optimization produced 23 selected candidates per starting nanobody, or **92 candidates in total**.

**Insight**
- The three tools serve complementary purposes:
  - ESM evaluates the quality or plausibility of the nanobody sequence.
  - AlphaFold-Multimer evaluates the predicted antibody–RBD interface.
  - Rosetta estimates the energetic favorability of the interaction.
- The optimization improved computational scores across rounds, but these scores were only predictions and still required experimental validation.

### Extended Data Figures 2–4

- **Ty1:** Produced a mutant that gained binding to JN.1.
- **H11-D4:** Some designs had improved computational scores but also showed nonspecific binding experimentally.
- **VHH-72:** Several mutants retained Wuhan binding, but the strongest new variant-binding effects were seen in Ty1 and Nb21.

**Overall computational results**
- All 92 mutants had positive ESM LLR values relative to their wild-type counterparts.
- 78/92 mutants had higher AF ipLDDT than the corresponding wild type.
- 60/92 had improved Rosetta binding energies.
- 23/92 had RS dG values at or below −50.

---

## 4. Figure 4 and Extended Data Figures 5–6 — Experimental validation

### Figure 4: Expression and binding

**Results**
- The study tested 92 designed nanobodies and 4 wild-type controls.
- Most designs were expressed as soluble proteins:
  - 35/92 produced more than 25 mg/L of soluble nanobody.
  - Only 6/92 produced less than 5 mg/L.
- Binding was tested against Wuhan, JN.1, KP.3, KP.2.3 and BA.2 RBDs, as well as MERS-CoV RBD and BSA controls.

### Nb21 mutant

**Nb21 I77V/L59E/Q87A/R37Q**
- Retained strong Wuhan RBD binding.
- Showed improved binding to JN.1 compared with wild-type Nb21.
- Also showed enhanced KP.3 binding compared with other Nb21 mutants.
- It showed no substantial nonspecific binding to MERS-CoV RBD or BSA.
- JN.1 binding was weaker than Wuhan binding, with an EC50 of approximately 2.0 ng/mL versus 0.2 ng/mL.

### Ty1 mutant

**Ty1 V32F/G59D/N54S/F32S**
- Improved Wuhan binding.
- Gained moderate binding to JN.1, which was not detected for wild-type Ty1.

### Other findings

- H11-D4 and Nb21 mutants generally retained Wuhan specificity.
- Many Ty1 mutants lost Wuhan binding.
- Some H11-D4 mutants showed nonspecific binding, possibly associated with the introduced R27C mutation.
- Several VHH-72 mutants retained Wuhan binding.

**Insight**
- The computational pipeline generated candidates that were generally expressible and soluble.
- However, computational improvement did not always translate into improved specificity or experimentally useful binding.
- The two strongest candidates demonstrate the value of using computation to narrow a large sequence space, but the study mainly measured binding by ELISA; neutralization and in vivo efficacy were not established.

### Extended Data Figure 5

- Shows the complete experimental workflow: nanobody expression, RBD production, antigen-array printing and multiplex ELISA.
- It illustrates how computational predictions were connected to laboratory validation.

### Extended Data Figure 6

- Presents SDS–PAGE results for the nanobody panel.
- The expected approximately 15-kDa nanobody bands were detected for most designs.
- This supports the conclusion that the introduced mutations generally did not cause major misfolding or aggregation.

---

## 5. Figure 5 and Extended Data Figure 7 — Agent interactions and the human role

**Results**
- Agents contributed different perspectives according to their assigned expertise.
- Across the project:
  - Human researcher: 1,596 words, about 1.3%
  - LLM agents: 122,462 words, about 98.7%
- The PI synthesized discussions and made high-level decisions.
- The Scientific Critic identified limitations and potential errors.
- Distinct agent identities produced more coherent and comprehensive discussions than generic agents.

**Insight**
- The human researcher did not need to write all of the code or detailed scientific reasoning.
- However, the human still controlled:
  - Research goals and agendas
  - Practical constraints
  - Code execution and debugging
  - Experimental validation
  - Final scientific decisions

Thus, the system is better described as a **human-supervised multi-agent research team** rather than a fully autonomous scientist.

---

## 6. Extended Data Figure 1 — Parallel meetings

**Results**
- The same meeting was run multiple times with different stochastic outputs.
- Higher temperature encouraged diverse solutions.
- A lower-temperature merge meeting combined the strongest elements.

**Insight**
- Parallel meetings reduce dependence on a single LLM response and can improve robustness.
- However, merging multiple answers does not automatically eliminate scientific errors.

---

## 7. Extended Data Table 1 — Nanobody scores

**Content**
The table reports:

- **ESM LLRWT:** sequence preference relative to the wild type
- **AF ipLDDT:** confidence in the predicted nanobody–spike interface
- **RS dG:** Rosetta binding-energy estimate
- **WSWT:** combined weighted score

**Insight**
- A candidate can have a strong sequence score but a weaker predicted interface, or vice versa.
- Combining the three metrics allowed the authors to balance sequence quality, structural confidence and binding energetics rather than relying on a single score.

---

## 8. Methods, supplementary material and data resources

**Methodological highlights**
- GPT-4o powered the PI, scientist and critic agents.
- Each agent was defined by its Title, Expertise, Goal and Role.
- Meetings generally used multiple discussion rounds.
- ESM, AlphaFold-Multimer and Rosetta were implemented through scripts generated by the agents.
- Nanobodies were expressed in *E. coli*, and RBD proteins were produced in Expi293 cells.
- Multiplex ELISA was used to measure binding.

**Limitations**
- LLM knowledge cutoffs may prevent access to the latest literature or software.
- Prompt engineering is important for obtaining useful responses.
- Agents may generate incorrect scientific claims or citations.
- The experiments mainly assessed binding; neutralization, in vivo efficacy and pharmacokinetics remain to be tested.

---

## Overall takeaways

1. **The Virtual Lab went beyond idea generation by selecting tools, writing code and designing a complete computational workflow.**
2. **Two of 92 candidates showed promising new binding profiles against JN.1 or KP.3.**
3. **The system was useful for narrowing a very large sequence space, but experimental testing remained essential.**
4. **Role-specific agents and a dedicated critic improved the breadth and consistency of the discussion.**
5. **Although the human researcher provided relatively little text, human supervision and final scientific judgment remained crucial.**

<br/>
# refer format:
### BibTeX

```bibtex
@article{Swanson2025VirtualLab,
  author  = {Swanson, Kyle and Wu, Wesley and Bulaong, Nash L. and Pak, John E. and Zou, James},
  title   = {The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies},
  journal = {Nature},
  volume  = {646},
  pages   = {716--723},
  year    = {2025},
  date    = {2025-10-16},
  doi     = {10.1038/s41586-025-09442-9},
  url     = {https://doi.org/10.1038/s41586-025-09442-9}
}
```

### 시카고 스타일    

Swanson, Kyle, Wesley Wu, Nash L. Bulaong, John E. Pak, and James Zou. “The Virtual Lab of AI Agents Designs New SARS-CoV-2 Nanobodies.” *Nature* 646 (October 16, 2025): 716–723. https://doi.org/10.1038/s41586-025-09442-9.


