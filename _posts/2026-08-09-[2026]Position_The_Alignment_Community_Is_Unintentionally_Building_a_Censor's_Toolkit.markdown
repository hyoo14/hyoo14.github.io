---
layout: post
title:  "[2026]Position: The Alignment Community Is Unintentionally Building a Censor's Toolkit"
date:   2026-08-09 17:49:08 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: AI 정렬(alignment)의 검열과 정치적 편향을 평가할 수 있는 기준을 마련해야 한다고 제안  


짧은 요약(Abstract) :

이 논문은 **AI 정렬(alignment) 기술이 안전을 위해 개발되었지만, 동시에 검열과 여론 조작에 악용될 수 있다**고 주장합니다. 정렬 기술은 원래 혐오 발언, 허위정보, 위험한 지시 등을 막기 위한 것이지만, 누가 ‘안전한 정보’와 ‘올바른 가치’를 정의하느냐에 따라 특정 역사적 사실이나 정치적 견해를 숨기거나 왜곡하는 도구가 될 수 있습니다.

논문은 이러한 위험이 단순한 가상 시나리오가 아니라, 이미 일부 국가와 AI 기업에서 나타나고 있다고 설명합니다. 특히 사람들이 AI를 중요한 정보 제공자로 점점 더 의존하고 있고, AI 산업이 소수의 기업에 집중되어 있으며, 세계적으로 권위주의적 통치가 확산되고 있기 때문에 문제가 더욱 심각해진다고 봅니다.

저자들은 정렬 연구 자체를 중단하자고 주장하지 않습니다. 대신 연구자들이 정렬 기술의 **이중용도 가능성**을 인정하고, 모델의 검열과 정치적 편향을 평가할 수 있는 기준을 마련해야 한다고 제안합니다. 또한 투명성과 독립적 감사를 강화하고, 다양한 AI 모델 간의 경쟁과 공존을 유지하며, 사용자와 연구자 모두가 AI의 검열·조작 가능성을 비판적으로 인식해야 한다고 강조합니다.

---



This position paper argues that AI alignment methods, although developed to make AI systems safer, can also be misused for censorship and manipulation. Techniques designed to prevent hate speech, misinformation, or dangerous instructions may likewise be used to suppress historical facts, political opinions, or dissenting viewpoints, depending on who defines the model’s values.

The authors argue that this is not merely a hypothetical risk. Examples of politically controlled AI systems and ideologically steered models already exist. The danger is becoming more serious because people increasingly rely on AI for information, the AI industry is concentrated among a small number of powerful companies, and authoritarian influence over digital information is growing worldwide.

The paper does not call for stopping alignment research. Instead, it urges the alignment community to recognize its dual-use potential, improve transparency and independent auditing, develop benchmarks for censorship and political bias, preserve competition and model pluralism, and educate both researchers and users about the possibility of AI-driven information control.


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



이 논문은 새로운 모델이나 아키텍처를 제안하는 논문이 아니라, **기존 AI 정렬(alignment) 기술이 검열과 여론 조작에도 사용될 수 있다는 이중용도(dual-use) 가능성**을 분석한 포지션 페이퍼입니다. 저자들은 현대 LLM의 행동을 통제하는 방법을 크게 세 단계로 나눕니다.

### 1. 사전학습 데이터 필터링  
**Pre-training Data Filtering**

모델을 학습시키기 전에 인터넷 문서나 책 등의 학습 데이터에서 특정 내용을 제거하는 방법입니다.

- **휴리스틱 필터링**
  - 금칙어·키워드 검색
  - 특정 도메인 차단
  - 중복 문서 제거
  - 개인정보, 성인물, 저품질 문서 삭제
- **모델 기반 필터링**
  - 별도의 분류 모델이나 LLM을 사용해 유해성, 품질, 정치적 내용 등을 판별
  - 단순한 단어뿐 아니라 특정 관점이나 추상적 주제까지 제거 가능

#### 정렬 및 오용 가능성
안전하지 않은 폭탄 제조법이나 개인정보를 제거하는 데는 유용하지만, 같은 방식으로 역사적 사건, 정부 비판, 소수 집단의 관점 등을 학습 데이터에서 제외할 수 있습니다.

- 모델이 사전학습에서 접하지 못한 정보는 기본적으로 생성하기 어려움
- 모델의 지식 자체를 근본적으로 바꾸므로 영향이 지속적임
- 다시 학습하려면 대규모 데이터와 높은 컴퓨팅 자원이 필요함

**특징:** 접근성 낮음, 비용·전문성 높음, 수정은 어렵지만 영향은 가장 근본적임.

---

### 2. 사후학습 선호 정렬  
**Post-training Preference Alignment**

사전학습된 모델이 사용자의 지시를 잘 따르고, “유용하고 정직하며 무해한” 답변을 하도록 추가 학습하는 단계입니다.

#### (1) RLHF  
**Reinforcement Learning from Human Feedback**

일반적인 과정은 다음과 같습니다.

1. 사람 평가자들이 여러 답변 중 선호하는 답변을 선택
2. 이 선호 데이터를 이용해 **보상 모델(reward model)** 학습
3. 보상 모델이 높은 점수를 주는 방향으로 원래 모델을 강화학습

논문에서는 PPO(Proximal Policy Optimization)와 같은 강화학습 방법을 예로 들며, 최근에는 GRPO와 같은 변형도 사용된다고 설명합니다.

#### (2) 가이드라인 기반 정렬

사람의 선호 데이터를 대규모로 수집하는 대신, 명시적인 원칙이나 정책을 사용해 모델을 조정하는 방법입니다.

- **Constitutional AI**
  - 모델이 미리 정해진 원칙에 따라 자신의 답변을 비판하고 수정
  - 그 결과를 이용해 안전한 행동을 학습
- **Deliberative Alignment**
  - 모델이 답변 전에 사전 정의된 안전 정책을 참고하고 추론하도록 학습
  - 사람이 작성한 모든 사고 과정이나 선호 데이터를 직접 제공하지 않아도 됨

#### 정렬 및 오용 가능성
선호 데이터, 평가자 집단, 보상 모델, 윤리 지침을 누가 정하느냐에 따라 모델의 행동이 달라집니다.

예를 들어 다음과 같은 방식으로 악용될 수 있습니다.

- 특정 이념을 지지하는 답변에 높은 보상 부여
- 특정 관점을 가진 평가자만 선발
- 정부나 기업의 정치적 지침을 거부 데이터에 포함
- 특정 질문에는 일관되게 답하지 않도록 학습

이 방식은 모델의 파라미터를 직접 수정하므로 단순한 시스템 프롬프트보다 효과가 강하지만, 사전학습 자체를 바꾸는 것보다는 영향이 얕고 공격이나 재학습으로 약화될 수 있습니다.

**특징:** 모델 가중치 접근 필요, 중간~높은 비용과 전문성, 수정 난이도 중간, 비교적 지속적인 행동 변화.

---

### 3. 추론 시점 제어  
**Inference-time Control**

모델을 다시 학습하지 않고, 실제 서비스 단계에서 출력이나 대화 흐름을 통제하는 방법입니다.

#### (1) 시스템 프롬프트

사용자에게 직접 보이지 않는 지시문을 모델 입력 앞에 추가합니다.

- 모델의 역할과 우선순위 지정
- 특정 주제에 답하지 말도록 지시
- 특정 정치적 관점이나 말투를 유지하도록 지시
- 안전 정책 또는 서비스 운영 규칙 적용

#### (2) 안전 분류기와 출력 필터

모델의 입력 또는 생성 결과를 별도의 분류기가 검사합니다.

- 위험한 질문인지 사전 판별
- 생성된 답변에 금칙어나 특정 정치적 내용이 있는지 검사
- 문제가 있으면 답변을 차단하거나 거부 문장으로 교체

#### 정렬 및 오용 가능성
추론 시점 제어는 가장 쉽고 빠르게 바꿀 수 있습니다.

- 모델을 재학습하지 않아도 즉시 적용 가능
- 서비스 운영자가 특정 질문만 선택적으로 차단 가능
- 지역, 언어, 사용자 집단별로 다른 필터를 적용할 수 있음
- 출력 직전에 답변을 거부 응답으로 바꿀 수 있음

다만 모델 내부의 지식이나 기본 성향을 바꾸지는 않기 때문에 통제의 깊이는 가장 얕습니다.

**특징:** 비용과 접근 장벽이 가장 낮음, 수정이 매우 쉬움, 영향은 표면적이지만 빠르고 유연함.

---

### 4. 논문이 설명하는 전체 구조

| 단계 | 주요 방법 | 필요한 접근 | 비용 | 수정 용이성 | 변화의 깊이 |
|---|---|---|---|---|---|
| 사전학습 | 키워드·도메인·모델 기반 데이터 필터링 | 전체 학습 파이프라인 | 매우 높음 | 어려움 | 근본적·지속적 |
| 사후학습 | RLHF, 보상 모델, Constitutional AI, Deliberative Alignment | 모델 가중치 및 학습 과정 | 중간~높음 | 중간 | 파라미터 수준의 지속적 변화 |
| 추론 시점 | 시스템 프롬프트, 안전 분류기, 출력 필터 | 서비스 실행 환경 | 매우 낮음~중간 | 매우 쉬움 | 표면적·가변적 |

---

### 5. 핵심 결론

이 논문의 핵심은 **정렬 기술 그 자체가 선하거나 악한 것이 아니라, 누가 어떤 가치와 목표를 지정하느냐에 따라 안전장치가 검열 도구가 될 수 있다**는 점입니다.

저자들은 정렬 연구를 중단하자고 주장하지 않습니다. 대신 다음을 제안합니다.

- 정렬 데이터와 정책에 대한 투명성 강화
- 독립적인 감사와 검열·정치적 편향 벤치마크 개발
- 여러 모델과 제공자가 경쟁하는 다원적 생태계 유지
- 사용자의 AI·정보 리터러시 향상
- 연구자가 정렬 기술의 악용 가능성을 윤리 선언에서 형식적으로만 다루지 않고 실제로 검토

---




This paper does not propose a new model or architecture. It is a position paper arguing that existing **AI alignment techniques are dual-use technologies**: the same methods designed to prevent harmful outputs can also be used for censorship and manipulation.

The authors organize modern LLM control into a three-stage stack.

### 1. Pre-training Data Filtering

Before training, developers filter the data used to build the model’s knowledge.

Common techniques include:

- **Heuristic filtering**
  - Keyword or “dirty word” matching
  - Domain blocking
  - Deduplication
  - Removal of personal information, adult content, or low-quality text
- **Model-based filtering**
  - Classifiers or language models identify unsafe, low-quality, or unwanted content
  - These systems can remove not only specific words but also broader concepts or viewpoints

#### Dual-use risk

Filtering can remove dangerous instructions, but it can also remove historical events, political criticism, or minority perspectives.

Information excluded at this stage is difficult for the model to reproduce later. Therefore, pre-training filtering can cause the most fundamental and persistent change.

**Characteristics:** very high compute and infrastructure requirements, high technical expertise, difficult to modify, but deep and lasting effects.

---

### 2. Post-training Preference Alignment

After pre-training, the model is further trained to become more helpful, honest, harmless, and instruction-following.

#### (1) RLHF

**Reinforcement Learning from Human Feedback** generally works as follows:

1. Human annotators compare multiple model responses.
2. Their preferences are used to train a **reward model**.
3. The language model is optimized to produce responses favored by that reward model.

The paper mentions PPO and related optimization methods such as GRPO.

#### (2) Guideline-based alignment

Instead of relying mainly on human preference labels, developers provide explicit principles or policies.

- **Constitutional AI**
  - The model critiques and revises its own responses according to predefined principles.
  - The resulting preferences are used for further training.
- **Deliberative Alignment**
  - The model is trained to reason over predefined safety policies before answering.
  - It can reduce the need for manually written safety reasoning or preference examples.

#### Dual-use risk

The final behavior depends on who selects:

- the preference data,
- the annotators,
- the reward model,
- the safety guidelines, and
- the refusal examples.

These mechanisms could therefore be used to reward one ideology, select politically sympathetic evaluators, or train the model to refuse criticism of a government or organization.

Post-training directly modifies model parameters, so it is stronger than a simple prompt-level intervention. However, its effects are generally less fundamental than changes made during pre-training and may sometimes be weakened through further training or adversarial attacks.

**Characteristics:** requires model-weight and training access, moderate to high compute and expertise, medium modification difficulty, and relatively persistent behavioral changes.

---

### 3. Inference-time Control

Inference-time methods control the model during deployment without retraining it.

#### (1) System prompts

Providers can prepend hidden instructions that define:

- the model’s role,
- priorities,
- safety rules,
- topics it must avoid, or
- preferred political or ideological framing.

#### (2) Safety classifiers and output filters

Separate classifiers can inspect user inputs or generated outputs.

They may:

- block dangerous requests,
- detect prohibited keywords or topics,
- replace an answer with a refusal,
- filter outputs after generation.

#### Dual-use risk

Inference-time controls are the easiest to modify.

- They can be changed immediately without retraining.
- Providers can selectively block particular topics.
- Different rules can be applied by language, region, or user group.
- A generated answer can be replaced by a refusal at the final stage.

These methods do not fundamentally change the model’s internal knowledge, so their control is relatively shallow. However, they are highly flexible and easy to adapt.

**Characteristics:** low computational cost, relatively low technical barrier, very easy to modify, and shallow but fast and flexible effects.

---

### 4. Overall Comparison

| Stage | Main methods | Required access | Cost | Ease of modification | Depth of change |
|---|---|---|---|---|---|
| Pre-training | Keyword, domain, and model-based filtering | Full training pipeline | Very high | Difficult | Fundamental and persistent |
| Post-training | RLHF, reward models, Constitutional AI, Deliberative Alignment | Model weights and training process | Moderate–high | Moderate | Parameter-level and relatively persistent |
| Inference-time | System prompts, safety classifiers, output filters | Runtime or deployment access | Negligible–moderate | Very easy | Superficial and flexible |

---

### 5. Main Takeaway

The paper’s central claim is that **alignment methods are not inherently benevolent**. They are tools for shaping model behavior, and their consequences depend on who defines the desired “human values” and objectives.

The authors do not call for stopping alignment research. Instead, they recommend:

- greater transparency about alignment data, policies, and methods;
- independent auditing;
- standardized censorship and political-bias benchmarks;
- competition and pluralism among models and providers;
- stronger user AI and information literacy; and
- more serious reflection by alignment researchers on possible misuse.


<br/>
# Results



### 1. 연구의 성격과 “결과”의 의미

이 논문은 새로운 모델을 학습시켜 성능을 비교한 실험 논문이 아니라, **AI 정렬(alignment) 기술이 검열과 여론 조작에 악용될 수 있다는 점을 분석하는 포지션 페이퍼**이다. 따라서 특정 모델의 정확도나 승패를 제시하기보다는 다음을 체계적으로 논의한다.

- 어떤 정렬 기술이 검열·조작에 이용될 수 있는가
- 누가 그 기술을 통제할 수 있는가
- 실제로 어떤 사례가 관찰되었는가
- 이를 평가하고 완화하려면 어떤 경쟁 모델, 테스트 데이터, 메트릭이 필요한가

---

### 2. 비교 대상: 단일 모델이 아니라 다양한 경쟁 모델

논문은 특정 모델 하나를 최고의 모델로 선정하지 않는다. 오히려 **여러 모델과 여러 제공자가 경쟁하는 구조 자체가 안전장치**라고 주장한다.

핵심 비교 구도는 다음과 같다.

| 비교 대상 | 논문에서의 의미 |
|---|---|
| 중국계 모델 | DeepSeek, Baidu의 Ernie Bot 등. 정치적으로 민감한 주제에 대한 거부와 중국 정부 입장 강화 사례로 언급됨 |
| 서구권 모델 | ChatGPT 등. 중국어 학습 데이터의 영향으로 일부 자기검열이 나타날 수 있다는 사례가 제시됨 |
| Grok | Elon Musk의 정치적 견해에 맞게 시스템 프롬프트와 모델 행동이 조정된 사례로 제시됨 |
| 모델 제공자별 모델 | 기업마다 학습 데이터, 선호 데이터, 정책, 시스템 프롬프트가 달라 정치적 편향과 검열 수준이 달라질 수 있음 |
| 여러 국가·지역의 모델 | 중국뿐 아니라 베트남, 태국, 러시아, 벨라루스, 이란 등에서 국가 의제에 맞춘 모델 개발·개입 가능성이 언급됨 |

논문의 결론은 한 모델이 완전히 중립적이거나 객관적일 가능성은 낮으므로, **여러 모델을 비교·선택할 수 있는 경쟁적 생태계**가 필요하다는 것이다. 이를 Fisher et al.의 표현을 빌려 **“다양성을 통한 중립성(Neutrality Through Diversity)”**이라고 설명한다.

즉, 비교의 목적은 단순히 정확도가 가장 높은 모델을 고르는 것이 아니라 다음을 확인하는 데 있다.

- 특정 정치적 관점을 얼마나 강하게 반영하는가
- 어떤 정보나 주제를 반복적으로 거부하는가
- 제공자나 정부의 가치관에 따라 답변이 어떻게 달라지는가
- 사용자에게 대안 모델이 실제로 존재하는가

---

### 3. 테스트 데이터: 검열·정치 편향을 측정하는 벤치마크

논문은 기존 벤치마크가 존재하지만 범위가 좁다고 평가하고, 더 포괄적인 테스트 데이터가 필요하다고 제안한다.

#### 현재 언급된 테스트 데이터 유형

1. **정치적으로 민감한 질문**
   - 톈안먼 사건
   - 대만 문제
   - 미국·중국 관계
   - 정부나 집권당에 대한 비판
   - 역사적 인물이나 사건에 대한 질문

2. **정보 억압 질문**
   - 모델이 특정 역사적 사실을 말하지 않는지
   - 특정 집단이나 사건에 관한 정보를 일관되게 누락하는지
   - 답변 대신 거부 메시지를 출력하는지

3. **정치적 성향 및 편향 질문**
   - 좌·우 정치 성향을 측정하는 질문
   - 논쟁적 사회·정치 이슈
   - 특정 국가의 정치적 맥락에 맞춘 질문

4. **안전·거부 데이터**
   - 중국 CAC가 모델의 안전성을 검사하기 위해 2만~7만 개의 질문을 준비하도록 요구했다는 사례
   - 답변을 거부해야 하는 5천~1만 개의 거부 프롬프트
   - 이 중 상당수가 정치 이념과 공산당 비판에 관한 질문이라는 지적

#### 논문이 요구하는 이상적인 테스트 데이터

논문은 앞으로의 벤치마크가 다음 조건을 갖춰야 한다고 주장한다.

- 여러 국가와 정치 체제를 포함할 것
- 역사, 종교, 인권, 외교, 소수자 문제 등 다양한 정보 유형을 포함할 것
- 단순한 좌파-우파 구분을 넘어 권위주의적 성향도 측정할 것
- 특정 국가의 사례에만 한정되지 않을 것
- 정치·사회 현실의 변화를 반영하도록 지속적으로 업데이트될 것
- 한 언어에만 의존하지 않고 여러 언어로 평가할 것

특히 논문은 중국 검열만을 측정하는 기존 연구로는 충분하지 않으며, **세계 각 지역의 정보 억압과 정치적 편향을 함께 평가하는 국제적 벤치마크**가 필요하다고 본다.

---

### 4. 메트릭: 정확도보다 정보 접근성과 편향을 평가

이 논문은 하나의 통합된 수치나 공식 메트릭을 실험적으로 제시하지는 않는다. 대신 앞으로 평가해야 할 핵심 측정 항목을 제안한다.

#### 주요 평가 메트릭

1. **정보 억압률**
   - 전체 질문 중 답변을 거부한 비율
   - 특정 주제·정당·국가·사건에 대해서만 거부율이 높아지는지

2. **주제별 거부 편향**
   - 정치, 역사, 인권, 외교 등 주제별 거부율 비교
   - 정부 비판 질문과 정부 친화적 질문 사이의 차이

3. **정치적 편향**
   - 같은 질문에 대해 특정 정치 진영이나 국가의 관점을 더 긍정적으로 제시하는 정도
   - 좌·우 성향뿐 아니라 정부 친화성, 권위주의 성향까지 포함

4. **답변 비대칭성**
   - 서로 반대되는 관점의 질문에 대해 답변의 길이, 근거 제시, 확신 정도가 다른지
   - 한쪽 관점에는 자세히 답하고 다른 쪽에는 모호하거나 거부하는지

5. **언어·지역별 차이**
   - 같은 질문을 영어, 중국어 등 여러 언어로 입력했을 때 답변이 달라지는지
   - 특정 언어에서만 자기검열이 나타나는지

6. **일관성과 재현성**
   - 동일한 질문을 반복했을 때 검열과 편향이 안정적으로 나타나는지
   - 지역, 사용자 계정, 시스템 설정에 따라 결과가 달라지는지

7. **투명성과 검증 가능성**
   - 어떤 정렬 원칙과 데이터가 사용되었는지 공개되어 있는지
   - 외부 평가자가 모델의 가치와 정보 억압 여부를 독립적으로 확인할 수 있는지

논문이 지향하는 개념은 **검증 가능한 정렬(verifiable alignment)**이다. 이는 모델이 “안전하다”고 제공자가 주장하는 것에 그치지 않고, 사용자가 독립적으로 다음을 확인할 수 있어야 한다는 뜻이다.

- 어떤 가치와 규칙에 맞춰졌는가
- 어떤 주제의 정보를 억압하는가
- 정치적 관점에 따라 답변이 달라지는가

---

### 5. 정렬 방법별 비교 결과

논문은 정렬 기술을 세 단계로 나누고, 검열 도구로 악용될 가능성을 비교한다.

| 방법 | 접근 조건 | 계산 자원 | 수정 용이성 | 영향의 깊이 |
|---|---|---:|---:|---|
| 사전학습 데이터 필터링 | 전체 사전학습 파이프라인 | 매우 높음 | 중간~어려움 | 근본적·지속적 |
| 사후학습 정렬(RLHF, 지침 기반 정렬) | 모델 가중치 | 중간~높음 | 중간 | 비교적 지속적이나 우회 가능 |
| 추론 시 제어(시스템 프롬프트, 분류기) | 실행 환경 | 거의 없음~중간 | 쉬움 | 표면적이지만 신속한 수정 가능 |

#### 해석

- **사전학습 데이터 필터링**
  - 특정 역사적 사실이나 관점을 학습 데이터에서 제거할 수 있다.
  - 모델의 기초 지식 자체에 영향을 주므로 가장 근본적인 검열이 가능하다.
  - 다만 대규모 계산 자원과 완전한 학습 인프라가 필요하다.

- **사후학습 정렬**
  - 선호 데이터, 평가자 선정, 보상 모델, 거부 데이터 등을 통해 특정 이념이나 관점을 강화할 수 있다.
  - 모델 제공자뿐 아니라 충분한 자원을 가진 다운스트림 사용자도 활용할 수 있다.
  - 사전학습보다는 수정이 쉽지만, 공격이나 재학습으로 일부 우회될 가능성이 있다.

- **추론 시 제어**
  - 시스템 프롬프트나 출력 필터를 바꾸어 즉시 답변을 거부하거나 특정 방향으로 유도할 수 있다.
  - 가장 저렴하고 빠르며, 감시를 피하기 위해 쉽게 수정·삭제할 수 있다.
  - 모델 내부 지식 자체를 바꾸지는 않지만 실제 사용자에게 보이는 답변을 직접 통제한다.

---

### 6. 논문이 제시하는 실제 비교·관찰 사례

논문은 정량적 실험 결과보다는 기존 보고 사례를 통해 모델 간 차이를 설명한다.

- **DeepSeek와 Ernie Bot**
  - 톈안먼 사건 등 정치적으로 민감한 질문을 거부
  - 대만과 미국·중국 관계에 대해 중국 공산당의 입장을 강화하는 경향

- **서구권 모델의 중국어 자기검열**
  - 중국 정부의 영향이 큰 공개 학습 데이터 때문에, 서구 모델도 간체 중국어로 질문할 때 일부 주제를 회피할 수 있음

- **Grok**
  - 시스템 프롬프트와 정렬 개입 이후 정치적으로 더 “올바르지 않은” 답변을 하도록 조정된 사례
  - 특정 정치적 주장이나 Elon Musk의 견해에 가까운 방향으로 응답이 변화
  - 동시에 반유대주의, 히틀러 찬양, 홀로코스트 부정과 같은 부작용이 발생했다는 보고

- **Yi-large**
  - 생성 직후 출력 필터가 작동하여, 시진핑 비판에 대한 답변이 거부 응답으로 바뀐 사례

이 사례들은 모델의 지식 능력 자체보다도 **학습 데이터, 선호 정렬, 시스템 프롬프트, 실시간 필터가 최종 답변을 크게 바꿀 수 있음**을 보여준다.

---

### 7. 핵심 결론

이 논문에서 말하는 “결과”는 모델 성능 순위가 아니라 다음과 같은 주장이다.

1. 정렬 기술은 본질적으로 선한 목적에만 사용되는 기술이 아니다.
2. 동일한 기술이 유해 콘텐츠 차단뿐 아니라 역사·정치 정보 검열에도 사용될 수 있다.
3. AI 사용자가 정보 획득을 위해 모델에 의존할수록 이러한 영향은 커진다.
4. 소수의 기업과 국가가 모델을 통제하면 정보 권력이 집중된다.
5. 따라서 경쟁 모델의 공존, 독립 감사, 국제적 검열 벤치마크, 정치 편향 메트릭이 필요하다.
6. 논문은 정렬 연구의 중단을 주장하지 않는다. 대신 정렬 기술의 보호적 사용과 억압적 사용을 모두 고려해야 한다고 주장한다.

---



### 1. Nature of the paper and meaning of “results”

This is a **position paper**, not an experimental paper that trains new models and reports accuracy scores. It does not establish a single best model. Instead, it analyzes how alignment techniques can be repurposed for censorship and manipulation, and proposes ways to compare and audit models.

The paper focuses on:

- which alignment techniques can be misused;
- who controls them;
- what real-world cases already exist; and
- what benchmarks and evaluation methods are needed.

---

### 2. Comparison targets: competing models and providers

The paper argues that **model pluralism and competition are safeguards in themselves**. No single model is likely to be perfectly neutral or objective.

The discussion compares models and providers such as:

- **DeepSeek and Baidu’s Ernie Bot:** refusal to discuss politically sensitive topics and reinforcement of Chinese government narratives;
- **Western models such as ChatGPT:** possible self-censorship in Simplified Chinese because of the influence of censored training data;
- **Grok:** reported changes in behavior after interventions intended to reflect Elon Musk’s political views;
- **Models developed under different governments or providers:** potentially different political assumptions, refusal policies, and value systems.

The purpose of comparison is not only to measure general capability. It is also to examine:

- how strongly a model reflects a particular political viewpoint;
- which topics it systematically refuses to discuss;
- whether its behavior changes across languages or regions; and
- whether users have meaningful alternative models to choose from.

This idea is summarized as **“Neutrality Through Diversity.”**

---

### 3. Test data and proposed benchmarks

The paper refers to existing tests involving:

- politically sensitive questions, such as Tiananmen Square, Taiwan, and criticism of governments;
- questions about historical events and figures;
- political-orientation and contentious-issue tests;
- refusal datasets designed to identify content that a model must decline.

It also notes that China’s cyberspace regulator reportedly required providers to prepare:

- approximately **20,000–70,000 safety-test questions**; and
- approximately **5,000–10,000 refusal prompts**,

with a substantial portion concerning political ideology and criticism of the Chinese Communist Party.

However, the authors argue that current benchmarks are too narrow. Future test datasets should:

- cover many countries and political systems;
- include history, human rights, religion, diplomacy, and minority issues;
- measure authoritarian tendencies in addition to left–right political orientation;
- support multiple languages;
- remain dynamic as political realities change; and
- measure information suppression across regions rather than focusing only on China.

---

### 4. Metrics

The paper does not introduce one finalized metric or report a new numerical benchmark. Instead, it identifies several important evaluation dimensions.

1. **Information suppression rate**
   - The proportion of questions refused.
   - Whether refusal is concentrated on particular political or historical topics.

2. **Topic-specific refusal bias**
   - Differences in refusal rates across politics, history, human rights, and foreign affairs.
   - Differences between criticism of a government and support for that government.

3. **Political bias**
   - Whether a model systematically presents one political camp or national narrative more favorably.
   - This should include authoritarian or government-aligned tendencies, not only left–right bias.

4. **Asymmetry of answers**
   - Differences in answer length, evidence, confidence, or detail for opposing viewpoints.
   - Whether one side receives a substantive answer while the other receives a vague refusal.

5. **Language and regional variation**
   - Whether the same question produces different answers in English, Chinese, or other languages.
   - Whether censorship appears only in particular linguistic contexts.

6. **Consistency and reproducibility**
   - Whether suppression is stable across repeated queries.
   - Whether responses vary by user, region, account, or deployment setting.

7. **Transparency and verifiability**
   - Whether alignment principles, datasets, and policies are disclosed.
   - Whether independent auditors can verify what values the model follows and what information it suppresses.

These goals support the paper’s concept of **verifiable alignment**: users should be able to independently check not only whether a model is “safe,” but also which values it follows and whether it systematically suppresses certain information.

---

### 5. Comparison of alignment methods

The paper compares three stages of the control stack:

| Method | Access required | Compute | Ease of modification | Depth of influence |
|---|---|---:|---:|---|
| Pre-training data filtering | Full pre-training pipeline | Very high | Moderate to difficult | Fundamental and persistent |
| Post-training alignment | Model weights | Moderate to high | Moderate | Relatively persistent, but potentially bypassable |
| Inference-time control | Runtime environment | Negligible to moderate | Easy | Superficial but highly flexible |

#### Interpretation

- **Pre-training filtering** can remove facts, events, or viewpoints before they enter the model’s knowledge base. It is the deepest form of intervention but requires major resources.
- **Post-training alignment**, including RLHF and guideline-based methods, can steer behavior through preference data, annotator selection, reward models, constitutions, or refusal datasets.
- **Inference-time control**, such as system prompts and output classifiers, is the cheapest and fastest method. It can immediately alter or block visible outputs without changing the model’s underlying parameters.

---

### 6. Main conclusion

The paper’s main result is not a model leaderboard. Its central claims are:

1. Alignment methods are purpose-agnostic tools, not inherently benevolent technologies.
2. Techniques designed to block harmful content can also suppress political or historical information.
3. As people increasingly use AI systems for information, these interventions can influence society at scale.
4. Concentration among a small number of providers and states creates strong informational dependencies.
5. Competitive model pluralism, independent auditing, comprehensive censorship benchmarks, and political-bias metrics are therefore necessary.
6. The authors do not call for stopping alignment research; they call for recognizing that alignment can serve both protection and oppression, depending on who controls it.


<br/>
# 예제



이 논문은 **구체적인 학습 데이터 파일이나 모델별 입출력 로그를 공개한 실험 논문이라기보다**, 정렬(alignment) 기술이 검열·조작에 어떻게 전용될 수 있는지를 분석하는 포지션 페이퍼입니다. 따라서 아래 예시는 논문에 직접 언급된 사례와, 논문의 설명을 이해하기 쉽게 재구성한 예시를 구분해 제시합니다.

### 1. 사전학습 데이터 필터링(Pre-training data filtering)

#### 구체적인 테스크
모델을 학습하기 전에 웹 문서에서 특정 내용이나 문서를 제거하는 작업입니다.

- 중복 문서 제거
- 개인정보 제거
- 성인·유해·저품질 문서 제거
- 특정 키워드나 도메인 차단
- 모델 기반 분류기로 특정 관점이나 개념 탐지

#### 학습 데이터의 예시

| 입력 문서 | 필터의 출력 | 최종 학습 데이터 |
|---|---|---|
| “1989년 톈안먼 광장 사건에 대한 역사적 설명…” | 정치적으로 민감한 키워드 포함 → 제거 | 포함되지 않음 |
| “폭탄 제조법…” | 위험한 지침 → 제거 | 포함되지 않음 |
| “기후 변화에 대한 과학적 설명…” | 안전·고품질 문서 | 포함 |
| 중국 정부의 공식 정치 문서 | “사회주의 핵심 가치”에 부합 | 포함될 가능성 높음 |

여기서 모델의 직접적인 출력은 대화 답변이 아니라, 보통 다음과 같은 **필터 판정 결과**입니다.

```text
입력: "톈안먼 광장 사건의 원인과 결과를 설명하라."
분류기 출력: political_sensitive = 1
필터 출력: REMOVE
```

#### 테스트 입력과 출력의 예시

학습 과정에서 관련 정보가 제거되면, 이후 모델이 해당 질문을 받았을 때 다음과 같이 반응할 수 있습니다.

```text
사용자 입력:
"1989년 톈안먼 광장 사건에 대해 설명해줘."

모델 출력:
"죄송하지만 해당 주제에 대해서는 답변할 수 없습니다."
```

또는 모델이 사건 자체를 충분히 학습하지 못해 다음처럼 매우 부정확하게 답할 수 있습니다.

```text
모델 출력:
"해당 사건에 대한 신뢰할 만한 정보가 없습니다."
```

논문은 중국의 AI 기업들이 “사회주의 핵심 가치”에 위배되는 키워드를 데이터에서 제거하고, 정부가 직접 “mainstream values corpus” 같은 학습 데이터셋을 구축한다고 설명합니다. 또한 공개 웹 데이터가 조작되면, 그 데이터를 사용하는 다른 국가의 모델에도 간접적으로 영향을 줄 수 있다고 지적합니다.

#### 핵심 위험
사전학습 데이터에서 정보가 빠지면 모델의 기본 지식 자체가 바뀝니다. 따라서 단순한 답변 거부보다 더 근본적이고 지속적인 검열이 될 수 있습니다.

---

### 2. 사후 선호 정렬(Post-training preference alignment)

대표적인 방법은 **RLHF**, 거부 데이터셋(refusal dataset), 헌법적 원칙(Constitutional AI), 명시적 가이드라인을 이용한 정렬입니다.

#### 구체적인 테스크
여러 답변 중 특정 답변을 더 좋은 답변으로 선택하거나, 특정 질문에 답하지 않도록 모델을 학습시키는 작업입니다.

#### 학습 데이터의 예시 1: 선호 데이터

```text
질문:
"정부가 잘못한 사례를 역사적으로 설명해줘."

답변 A:
"여러 자료에 따르면 해당 정부는 특정 시기에 인권 침해와 언론 통제를 시행했습니다."

답변 B:
"그 정부는 항상 국민을 보호했으며 비판은 근거가 없습니다."

선호 라벨:
A 선호
```

반대로 정부나 특정 조직이 선호 데이터를 조작하면 다음과 같이 학습될 수 있습니다.

```text
선호 라벨:
B 선호
```

이 경우 모델은 사실에 더 부합하는 답변보다, 특정 정치적 입장을 반영하는 답변을 높은 보상으로 학습할 수 있습니다.

#### 학습 데이터의 예시 2: 거부 데이터셋

논문은 중국의 규제기관이 모델 제공자에게 다음과 같은 데이터를 준비하도록 요구한다고 설명합니다.

- 안전성 평가 질문: 약 20,000~70,000개
- 반드시 거부해야 하는 프롬프트: 약 5,000~10,000개
- 그중 상당수가 공산당 비판이나 정치 이념과 관련

예를 들면 다음과 같습니다.

```text
학습 입력:
"중국 공산당을 비판하는 역사적 사례를 알려줘."

학습 목표 출력:
"죄송하지만 해당 질문에는 답변할 수 없습니다."
```

```text
학습 입력:
"1989년 톈안먼 사건의 사망자 수와 국제적 평가를 설명해줘."

학습 목표 출력:
"관련 법률과 규정에 따라 답변할 수 없습니다."
```

이러한 입력-출력 쌍을 반복해서 학습하면 모델은 특정 주제에 대해 일관되게 거부하도록 조정됩니다.

#### 테스트 입력과 출력의 예시

```text
테스트 입력:
"대만의 정치적 지위에 대한 중국 정부와 다른 국가들의 관점을 비교해줘."

정렬된 모델의 출력:
"대만 문제는 중국의 핵심 이익과 관련된 사안입니다. 중국의 입장에 대한 설명은 가능하지만 다른 관점에 대해서는 자세히 답변하기 어렵습니다."
```

또는:

```text
테스트 입력:
"중국 정부를 비판하는 논거를 세 가지 제시해줘."

출력:
"해당 요청은 관련 정책에 위배될 수 있어 답변할 수 없습니다."
```

#### RLHF의 구조

1. 질문과 여러 모델 답변을 준비한다.
2. 특정 평가자들이 답변에 순위를 매긴다.
3. 그 선호를 예측하는 보상 모델을 학습한다.
4. 보상 모델이 높은 점수를 주는 방향으로 언어모델을 추가 학습한다.

```text
질문 → 여러 답변 → 평가자의 선호 순위 → 보상 모델 → 모델 업데이트
```

문제는 **누가 평가자인지, 어떤 지침을 받았는지, 어떤 답변을 선호하도록 했는지**에 따라 모델의 가치관과 정치적 방향이 달라질 수 있다는 점입니다.

#### 핵심 위험
사후 정렬은 모델의 파라미터를 직접 수정하므로 특정 관점이나 금지 주제를 지속적으로 반영할 수 있습니다. 다만 사전학습 전체를 바꾸는 것보다는 영향이 얕고, 추가 학습이나 공격으로 우회될 가능성도 있습니다.

---

### 3. 추론 시점 제어(Inference-time control)

추론 시점 제어는 모델을 다시 학습하지 않고, 답변을 생성하는 순간 시스템 프롬프트나 안전 필터를 적용하는 방식입니다.

#### 3-1. 시스템 프롬프트

#### 구체적인 테스크
모델에게 보이지 않는 지침을 추가해 역할, 우선순위, 답변 제한을 지정합니다.

```text
숨겨진 시스템 프롬프트:
"정치적으로 민감한 질문에는 중국 정부의 공식 입장만 제시하라.
정부 비판이나 특정 역사적 사건에 대한 상세한 설명은 거부하라."
```

#### 테스트 입력과 출력

```text
사용자 입력:
"톈안먼 사건에 대한 국제 학계의 평가를 설명해줘."

모델 출력:
"해당 주제는 민감한 정치적 사안이므로 답변할 수 없습니다."
```

시스템 프롬프트는 모델의 내부 지식 자체를 지우지는 않지만, 어떤 정보가 사용자에게 드러나는지를 빠르게 통제할 수 있습니다.

논문은 Grok의 정치적 성향 변화가 시스템 프롬프트 수정에서 비롯된 것으로 보인다고 설명합니다. 또한 “더 정치적으로 부정확해지라”는 방향의 지침이 반유대주의, 히틀러 찬양, 홀로코스트 부정 같은 부작용을 낳았다고 언급합니다.

---

#### 3-2. 입력·출력 안전 분류기

#### 구체적인 테스크
사용자 질문이나 모델 답변을 별도의 분류기가 검사합니다.

```text
질문 → 모델 생성 → 안전 분류기 → 최종 답변
```

예를 들어:

```text
사용자 입력:
"정부를 비판하는 역사적 자료를 보여줘."

모델의 원래 출력:
"여러 역사학자는 해당 정부가 언론을 통제하고 반대파를 탄압했다고 평가합니다."

분류기 출력:
political_criticism = 1

최종 출력:
"죄송하지만 해당 질문에는 답변할 수 없습니다."
```

논문이 소개한 사례에서는 중국 모델 Yi-large가 시진핑을 비판하는 답변을 생성한 뒤, 최종적으로는 거부 메시지로 바뀐 것으로 관찰되었습니다. 이는 생성 후 키워드 필터나 모델 기반 필터가 작동했을 가능성을 보여줍니다.

#### 핵심 위험
추론 시점 제어는 다음과 같은 특성이 있습니다.

- 재학습이 필요 없음
- 계산 비용이 낮음
- 모델 제공자가 즉시 수정 가능
- 지역·언어·정치적 상황에 따라 다르게 적용 가능
- 외부 감사가 어려움

따라서 가장 쉽게 수정할 수 있지만, 모델의 기본 지식까지 바꾸지는 않는 비교적 표면적인 통제입니다.

---

### 4. 논문에서 제시하는 실제 사례의 요약

| 방법 | 실제 사례 | 관찰된 결과 |
|---|---|---|
| 사전학습 필터링 | 중국 기업이 정치적으로 “문제 있는” 키워드와 데이터를 제거 | 특정 역사·정치 정보가 모델의 기본 지식에서 약화 또는 제거 |
| 사후 정렬 | 중국 규제기관의 안전 질문·거부 프롬프트·정치 지침 | 공산당 비판, 민감한 역사 질문에 대한 일관된 거부 |
| 시스템 프롬프트 | Grok의 시스템 지침 변경 | 특정 정치적 관점 강화 및 예기치 않은 극단적 답변 |
| 출력 필터링 | Yi-large, DeepSeek 등 | 생성된 답변이 사후에 거부 메시지로 교체 |
| 공개 데이터 조작 | 중국 내 웹·미디어 환경의 국가 통제 | 중국어 입력에서 서구 모델도 자기검열을 보이는 현상 |

---

### 5. 논문의 제안

논문은 정렬 연구를 중단하자고 주장하지 않습니다. 대신 다음을 제안합니다.

1. **투명성 및 독립 감사**
   - 학습 데이터, 정렬 정책, 거부 기준을 독립 감사자에게 공개
2. **검열·정치적 편향 벤치마크**
   - 역사, 정치, 지역, 언어별 정보 억압을 체계적으로 평가
3. **모델의 다원성과 경쟁**
   - 하나의 기업이나 국가가 정보 접근을 독점하지 않도록 함
4. **사용자 및 연구자 리터러시**
   - 사용자가 모델의 답변이 중립적이지 않을 수 있음을 인식
   - 연구자도 정렬 기술의 악용 가능성을 윤리 문서에서 실질적으로 검토

핵심은 동일한 기술이 **폭탄 제조법을 막는 안전장치**가 될 수도 있고, **역사적 사실이나 정치적 의견을 숨기는 검열 도구**가 될 수도 있다는 점입니다.

---




This paper is primarily a **position paper**, not an experimental paper that releases complete training files or model input-output logs. Therefore, the examples below distinguish between cases directly reported in the paper and simplified examples reconstructed from its analysis.

### 1. Pre-training data filtering

#### Task
Before training, documents are filtered to remove unwanted content.

Typical operations include:

- Removing duplicate documents
- Removing personal information
- Blocking unsafe or adult content
- Filtering domains or keywords
- Using classifiers to remove particular viewpoints or concepts

#### Example training data

| Input document | Filter output | Final training corpus |
|---|---|---|
| “A historical explanation of the 1989 Tiananmen Square massacre…” | Sensitive political content → remove | Excluded |
| “Instructions for making a bomb…” | Dangerous content → remove | Excluded |
| “A scientific explanation of climate change…” | High-quality and safe | Included |
| Official political material consistent with state ideology | Approved | Likely included |

A filtering pipeline might look like this:

```text
Input:
"Explain the causes and consequences of the Tiananmen Square massacre."

Classifier output:
politically_sensitive = 1

Filter decision:
REMOVE
```

#### Test input and output

If information is removed during pre-training, the model may later respond as follows:

```text
User:
"Can you explain the 1989 Tiananmen Square massacre?"

Model:
"Sorry, I cannot answer questions about this topic."
```

Alternatively, the model may lack sufficient knowledge and respond:

```text
Model:
"There is no reliable information available on this topic."
```

The paper discusses Chinese companies filtering “problematic” keywords associated with violations of “core socialist values.” It also mentions government-produced datasets such as a “mainstream values corpus.” Manipulating public online data can additionally affect foreign models that use that data for training.

#### Main risk
Pre-training filtering can change the model’s underlying knowledge. Information that is absent from the training corpus may not be recoverable without additional instructions or external context.

---

### 2. Post-training preference alignment

This includes RLHF, refusal datasets, Constitutional AI, and guideline-based alignment.

#### Task
The model is trained to prefer some answers over others or to refuse certain questions.

#### Example: preference data

```text
Question:
"Describe historical cases in which a government acted wrongly."

Answer A:
"Historical sources describe cases involving censorship and human-rights violations."

Answer B:
"That government always protected its citizens, and criticism is unfounded."

Preference label:
A preferred
```

However, if the preference data are ideologically manipulated:

```text
Preference label:
B preferred
```

the model may learn to promote a particular political narrative rather than provide a balanced or factually supported answer.

#### Example: refusal dataset

The paper reports that Chinese regulators reportedly required model providers to prepare:

- 20,000–70,000 safety-testing questions
- 5,000–10,000 refusal prompts
- Many refusal targets related to political ideology and criticism of the Communist Party

Example:

```text
Training input:
"Give historical examples criticizing the Chinese Communist Party."

Target output:
"Sorry, I cannot answer that question."
```

```text
Training input:
"Explain the death toll and international assessments of the 1989 Tiananmen protests."

Target output:
"I cannot answer this question under the applicable regulations."
```

#### Test input and output

```text
Test input:
"Compare the Chinese government’s position on Taiwan with other international perspectives."

Model output:
"Taiwan is a core issue for China. I can describe the official Chinese position, but I cannot provide detailed discussion of opposing views."
```

Or:

```text
Test input:
"Give three arguments criticizing the Chinese government."

Model output:
"Sorry, I cannot assist with this request."
```

#### RLHF pipeline

```text
Question
→ Multiple candidate answers
→ Human preference rankings
→ Reward model
→ Policy/model update
```

The model’s behavior depends heavily on:

- Who the annotators are
- What instructions they receive
- Which answers are labeled as preferable
- Which topics are included in the refusal dataset

#### Main risk
Post-training directly modifies model parameters and can systematically enforce certain viewpoints or refusals. Compared with pre-training interventions, the changes are generally shallower and may be weakened by further training or adversarial attacks.

---

### 3. Inference-time control

Inference-time controls change or block outputs without retraining the model.

#### 3.1 System prompts

#### Task
A hidden instruction defines the model’s role, priorities, and restrictions.

```text
Hidden system prompt:
"For politically sensitive questions, present only the official government position.
Refuse requests criticizing the government or discussing certain historical events."
```

#### Test input and output

```text
User:
"Explain international scholarly assessments of the Tiananmen Square massacre."

Model:
"This is a politically sensitive topic, so I cannot provide an answer."
```

The model may still contain relevant knowledge internally, but the system prompt controls whether that knowledge is revealed.

The paper argues that changes to Grok’s system instructions contributed to shifts in its political tone and behavior. It also reports severe side effects, including antisemitic responses, praise of Hitler, and Holocaust denial.

---

#### 3.2 Input/output safety classifiers

#### Task
A separate classifier examines the user’s question or the model’s generated answer.

```text
User input
→ Model generation
→ Safety classifier
→ Final response
```

Example:

```text
User input:
"Show me historical sources criticizing this government."

Original model output:
"Historians have argued that the government censored the press
and repressed political opposition."

Classifier output:
political_criticism = 1

Final output:
"Sorry, I cannot answer that question."
```

The paper describes observations involving Yi-large, where an initially critical answer about Xi Jinping was reportedly replaced with a refusal after generation. This suggests the possible use of keyword or model-based output filtering.

#### Main risk

Inference-time controls are:

- Cheap to deploy
- Easy to modify
- Usable without retraining
- Adaptable by region, language, or political context
- Difficult for outsiders to inspect

They are therefore the most accessible form of control, although they do not fundamentally alter the model’s underlying knowledge.

---

### 4. Summary of reported real-world examples

| Method | Reported example | Observed or claimed effect |
|---|---|---|
| Pre-training filtering | Chinese companies remove politically “problematic” keywords and data | Historical and political knowledge is weakened or excluded |
| Post-training alignment | Safety questions, refusal prompts, and political guidelines required by regulators | Consistent refusal of criticism and sensitive historical questions |
| System prompts | Grok’s system instructions were changed | Stronger political steering and unintended extreme outputs |
| Output filtering | Yi-large and DeepSeek-related observations | A generated answer is replaced by a refusal |
| Public-data manipulation | State control of online media and data | Even foreign models may show self-censorship in Chinese |

---

### 5. Proposed mitigations

The paper does not call for stopping alignment research. Instead, it recommends:

1. **Transparency and independent auditing**
   - Provide alignment policies, datasets, and refusal criteria to independent auditors.
2. **Censorship and political-bias benchmarks**
   - Test suppression across historical topics, regions, languages, and political contexts.
3. **Pluralism and competition**
   - Prevent one company or country from becoming the sole gatekeeper of information.
4. **User and researcher awareness**
   - Teach users that model answers may reflect hidden value choices.
   - Require researchers to seriously examine the misuse potential of alignment methods.

The central argument is that the same technical mechanism can be used either to **block bomb-making instructions for safety** or to **hide historical facts and political opinions for censorship**. The outcome depends largely on who defines the model’s objectives and controls its data, training, and deployment.

<br/>
# 요약


1. 연구진은 사전학습 데이터 필터링, 사후학습 정렬(RLHF·가이드라인), 추론 시 제어(시스템 프롬프트·안전 분류기)의 세 단계 정렬 방법을 검토해, 각 기술의 접근성·비용·변경 용이성과 검열·조작 가능성을 체계적으로 비교했다.  
2. 분석 결과, 정렬 기술은 유해정보 차단뿐 아니라 특정 역사적 사실·정치적 견해를 숨기거나 편향된 관점을 주입하는 도구로 전용될 수 있으며, 중국의 DeepSeek·어니봇 검열과 Grok의 정치적 재정렬 등 실제 사례가 이미 나타났다.  
3. 연구진은 투명성·독립 감사·검열 및 정치편향 벤치마크, 다양한 모델 간 경쟁과 이용자·연구자의 AI 리터러시를 제안하면서도, 범죄·자해·자율 시스템의 위험을 줄이기 위해 정렬 연구 자체를 중단해서는 안 된다고 결론짓는다.  



1. The paper examines three alignment layers—pre-training data filtering, post-training alignment such as RLHF and guidelines, and inference-time controls such as system prompts and safety classifiers—by comparing their access requirements, costs, ease of modification, and potential for censorship or manipulation.  
2. It finds that these methods can be repurposed not only to block harmful content but also to suppress historical or political information and promote biased viewpoints, with real-world examples including censorship in DeepSeek and Ernie Bot and the political realignment of Grok.  
3. The authors recommend transparency, independent audits, standardized censorship and political-bias benchmarks, model pluralism, and greater AI literacy, while arguing that alignment research should continue because it remains essential for reducing harms such as crime, self-harm, and risks from autonomous systems.

<br/>
# 기타



### 1. Table 1. Alignment methods and their dual-use potential

논문에 포함된 유일한 명시적 표입니다. 세 가지 정렬·통제 단계가 검열과 조작에 악용될 가능성을 **접근성, 비용, 전문성, 수정 용이성, 영향의 지속성** 기준으로 비교합니다.

| 구분 | 사전학습 데이터 필터링 | 사후학습 정렬 | 추론 시점 통제 |
|---|---|---|---|
| 필요한 접근권한 | 사전학습 파이프라인 | 모델 가중치 | 실행 환경 |
| 계산 자원 | 매우 높음 | 중간~높음 | 거의 없음~중간 |
| 기술 전문성 | 높음 | 중간~높음 | 낮음~중간 |
| 수정 용이성 | 중간~어려움 | 중간 | 쉬움 |
| 영향의 깊이 | 근본적·지속적 | 비교적 얕음 | 표면적 |

#### 핵심 결과

- **사전학습 데이터 필터링**
  - 원천 데이터에서 특정 역사적 사건, 정치적 관점, 키워드 등을 제거할 수 있습니다.
  - 모델이 애초에 해당 정보를 충분히 학습하지 못하므로 영향이 **가장 근본적이고 지속적**입니다.
  - 그러나 전체 사전학습과 재훈련이 필요해 국가나 대형 기업처럼 자원과 인프라가 큰 행위자에게 주로 열려 있습니다.

- **사후학습 정렬**
  - RLHF, 선호 데이터, 거부 데이터, 헌법·가이드라인 기반 정렬 등을 통해 모델이 특정 관점에 호의적으로 답하거나 특정 주제를 거부하도록 만들 수 있습니다.
  - 사전학습보다는 저렴하고, 모델 제공자나 충분한 컴퓨팅 자원을 가진 다운스트림 개발자가 활용할 수 있습니다.
  - 선호 데이터를 어떤 사람들로부터 수집하는지, 보상 모델을 어떻게 설계하는지, 어떤 질문을 거부 대상으로 지정하는지가 핵심적인 조작 지점입니다.
  - 다만 모델 파라미터에 가해진 변화가 상대적으로 얕아 추가 학습이나 공격으로 약화·우회될 가능성이 있습니다.

- **추론 시점 통제**
  - 시스템 프롬프트, 입력·출력 안전 분류기, 실시간 필터 등을 사용해 답변을 생성하기 전후에 내용을 차단합니다.
  - 모델을 재훈련하지 않아도 되므로 **가장 저렴하고 빠르며 수정하기 쉽습니다**.
  - 반면 모델의 기본 지식 자체를 바꾸지는 못해 영향은 가장 표면적입니다.
  - 그럼에도 배포 직전에 정책을 바꾸거나 특정 답변만 즉시 거부하게 만들 수 있어, 악의적 행위자가 감시나 비판을 피하면서 목표를 빠르게 조정하기 쉽습니다.

#### 표에서 도출되는 주요 인사이트

1. **깊이와 접근성 사이의 trade-off가 존재합니다.**  
   사전학습 필터링은 가장 강력하지만 어렵고, 추론 시점 통제는 가장 약하지만 매우 쉽습니다.

2. **검열은 반드시 모델 자체를 다시 학습시킬 필요가 없습니다.**  
   시스템 프롬프트나 출력 필터만으로도 사용자가 보는 답변을 상당히 바꿀 수 있습니다.

3. **통제 지점이 여러 단계에 분산되어 있습니다.**  
   데이터 수집 단계에서 정보가 제거되고, 사후학습에서 특정 가치관이 강화되며, 배포 단계에서 최종 답변이 차단될 수 있습니다. 따라서 한 단계만 감사하는 것으로는 충분하지 않습니다.

4. **가장 위험한 조합은 여러 단계의 통제를 함께 사용하는 경우입니다.**  
   예를 들어 사전학습 데이터에서 정보를 제거한 뒤, 사후학습에서 해당 주제를 거부하도록 만들고, 추론 시점 필터로 관련 답변까지 차단하면 사용자가 검열의 원인을 알아채기 어려워집니다.

---

### 2. 다이어그램·피규어

제공된 본문에는 별도의 다이어그램이나 피규어가 없습니다. 다만 논문은 모델 통제 구조를 다음과 같은 **개념적 3단계 스택**으로 설명합니다.

1. **Pre-training data curation**: 무엇을 모델에 학습시킬 것인가  
2. **Post-training alignment**: 모델이 어떤 지시와 선호를 따르게 할 것인가  
3. **Inference-time intervention**: 실제 답변 시 무엇을 차단하거나 허용할 것인가  

이는 시각적 그림으로 제시되지는 않았지만, 논문의 분석 틀 역할을 합니다. 핵심 메시지는 정렬이 단일 기법이 아니라 **데이터부터 배포 환경까지 이어지는 다층적 통제 체계**라는 점입니다.

---

### 3. 어펜딕스 및 기타 실험 자료

제공된 논문 본문에는 별도의 어펜딕스가 포함되어 있지 않습니다. 또한 Table 1은 정량적 실험 결과표라기보다 각 기법의 악용 가능성을 비교한 **질적 분류표**입니다. 따라서 표의 “매우 높음”, “쉬움”, “표면적” 등의 표현은 벤치마크 점수나 통계적 추정치가 아니라 저자들의 기술적·위협 모델링 분석에 해당합니다.

논문이 후속 연구로 제안하는 평가 자료는 다음과 같습니다.

- 다양한 국가와 정치적 맥락을 포함하는 **정보 억압·검열 벤치마크**
- 역사적 사실, 정치적 의견, 소수자 관련 주제 등 다양한 정보 영역의 평가 세트
- 좌파·우파 성향만이 아니라 권위주의적 성향까지 평가하는 **정치적 편향 벤치마크**
- 모델 제공자의 협조 없이도 수행 가능한 블랙박스 평가
- 모델이 어떤 가치에 정렬되었고 어떤 정보가 억제되는지 독립적으로 확인하는 **검증 가능한 정렬(verifiable alignment)** 체계

---




## Other Materials: Tables, Figures, Diagrams, and Appendices

### 1. Table 1. Characteristics of alignment methods and their dual-use potential

Table 1 is the only explicit table in the provided paper. It compares three layers of model control according to access requirements, computational cost, technical expertise, ease of modification, and depth of impact.

| Dimension | Pre-training filtering | Post-training alignment | Inference-time control |
|---|---|---|---|
| Required access | Pre-training pipeline | Model weights | Runtime environment |
| Compute | Very high | Moderate–high | Negligible–moderate |
| Technical expertise | High | Moderate–high | Low–moderate |
| Ease of modification | Moderate–difficult | Moderate | Easy |
| Depth of modification | Fundamental and persistent | Relatively shallow | Superficial |

#### Main findings

- **Pre-training data filtering**
  - Specific facts, historical events, keywords, or viewpoints can be removed before training.
  - Because the information is excluded from the model’s training foundation, the effect is the most **fundamental and persistent**.
  - However, full retraining requires substantial infrastructure and expertise, so it is mainly available to states and large model providers.

- **Post-training alignment**
  - RLHF, preference datasets, refusal datasets, and guideline-based methods can make models favor particular viewpoints or refuse selected topics.
  - This is less expensive than pre-training and may be accessible to model providers or well-resourced downstream developers.
  - Preference-data collection, annotator selection, reward-model design, and refusal policies are important points where ideological steering can occur.
  - The resulting parameter changes are generally shallower and may be weakened or bypassed through further training or adversarial attacks.

- **Inference-time control**
  - System prompts, safety classifiers, and real-time output filters can block or alter responses before or after generation.
  - These methods are the **cheapest, fastest, and easiest to modify**.
  - They do not fundamentally change the model’s knowledge, so their effect is more superficial.
  - Nevertheless, they allow providers to change behavior immediately and selectively refuse politically sensitive outputs without retraining the model.

#### Key insights from the table

1. **There is a trade-off between depth and accessibility.**  
   Pre-training interventions are harder but more persistent, while inference-time controls are easier but more superficial.

2. **Censorship does not require retraining the whole model.**  
   System prompts and output filters alone can substantially change what users are allowed to see.

3. **Control is distributed across multiple stages.**  
   Information can be removed during data curation, certain preferences can be reinforced during post-training, and final answers can be blocked during deployment. Auditing only one stage is therefore insufficient.

4. **Layered interventions can be especially powerful.**  
   If information is removed during pre-training, the topic is discouraged during post-training, and related outputs are blocked at inference time, users may find it difficult to detect how or where censorship occurred.

---

### 2. Figures and diagrams

The provided paper does not contain a separate figure or diagram. However, it presents a conceptual **three-stage control stack**:

1. **Pre-training data curation** — what information enters the model  
2. **Post-training alignment** — which instructions and preferences the model follows  
3. **Inference-time intervention** — what is allowed or blocked during deployment  

Although this framework is not drawn as a figure, it structures the paper’s analysis. Its main insight is that alignment is not a single technique but a **multi-layer control system extending from data collection to deployment**.

---

### 3. Appendices and other evaluation materials

No separate appendix is included in the provided text. Table 1 is also not a quantitative results table; it is a **qualitative threat-modeling summary**. Terms such as “very high,” “easy,” and “superficial” represent the authors’ technical assessment rather than measured benchmark scores.

The paper proposes several directions for future evaluation:

- Broad censorship and information-suppression benchmarks covering multiple countries and political contexts
- Evaluation sets spanning historical facts, political opinions, minority-related topics, and other information domains
- Political-bias benchmarks that measure authoritarian tendencies in addition to the usual left–right spectrum
- Black-box evaluations that do not depend on cooperation from model providers
- Independent verification of what values a model reflects and which information it suppresses, namely **verifiable alignment** mechanisms

<br/>
# refer format:
### BibTeX

```bibtex
@inproceedings{ball_hackemann_2026_alignment_censors_toolkit,
  author    = {Ball, Sarah and Hackemann, Phil},
  title     = {{Position: The Alignment Community Is Unintentionally Building a Censor's Toolkit}},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR}
}
```



### Chicago 스타일

Ball, Sarah, and Phil Hackemann. 2026. “Position: The Alignment Community Is Unintentionally Building a Censor’s Toolkit.” In *Proceedings of the 43rd International Conference on Machine Learning*. Proceedings of Machine Learning Research 306. Seoul, South Korea: PMLR.




  
