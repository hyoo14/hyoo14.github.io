---
layout: post
title:  "[2026]Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences"
date:   2026-09-28 20:50:24 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: Jev가 과학적 워크플로우에서도 유용할 수 있으며 다만 수치적을 계산 과정 봐야한다(스텝바이 스텝으로)  


짧은 요약(Abstract) :


이 논문은 과학적 분석 과정에서 **언어적으로 해석해야 하는 선택을 Jev가 얼마나 정확하게 수행하는지** 평가한 연구입니다. 예를 들어 여러 측정값이 서로 독립적인 실험에서 나온 것인지, 하나의 배양액에서 반복된 것인지에 따라 계산해야 할 개수와 과학적 해석이 달라질 수 있습니다.

연구진은 10개의 과학 사례에서 20개의 선택 문제를 만들고, Jev와 다른 11개 모델 설정을 비교했습니다. 각 모델은 주어진 근거를 바탕으로 후보 관계 중 하나를 선택했으며, 실제 계산·필터링·수치 처리와 최종 판단은 동일한 코드가 담당했습니다. 따라서 모델의 **의미 해석 능력**과 그 선택이 계산 결과에 미친 영향을 분리해 평가할 수 있었습니다.

주요 결과는 다음과 같습니다.

- Jev는 100개의 의미 선택과 50개의 후속 계산 결과를 모두 정확하게 맞혔습니다.
- 이는 다른 5개 모델 설정과 동일한 완전 정확도였습니다.
- 성공한 응답 가운데 Jev의 지연시간 중앙값은 **0.335초**로 가장 짧았습니다.
- 일부 모델은 최종 결론은 맞혔지만, 한 배양 이력 문제에서 잘못된 관계를 선택해 실제 개수는 틀렸습니다.  
  예를 들어 최종적으로 “주장이 틀렸다(inconsistent)”는 판단은 맞았지만, 배양 이력의 개수는 잘못 계산되었습니다.

따라서 이 연구는 과학적 의사결정 모델을 평가할 때 최종 결론만 확인해서는 안 되며, **모델이 선택한 관계와 그로부터 계산된 수치까지 함께 점검해야 한다**고 강조합니다. Jev는 근거와 후보 선택지가 미리 정리되어 있고, 계산은 코드로 처리되는 과학적 워크플로에서 유용한 구성요소가 될 가능성을 보였습니다.

---



This paper evaluates how accurately Jev performs **semantic decisions in scientific workflows**. In many scientific analyses, a small interpretive choice can change the meaning of a calculation. For example, ten observations may come from independently grown cultures or from repeated samples of a single culture. These alternatives lead to different counts and scientific interpretations.

The authors created 20 fixed-choice questions across 10 scientific cases and compared Jev with 11 other model configurations. Each model selected a relation from the supplied evidence, while all counting, filtering, arithmetic, and final-label composition were handled by the same deterministic program. This allowed the study to evaluate semantic selection separately from its computational consequences.

The main findings were:

- Jev answered all 100 semantic questions and all 50 downstream computations correctly.
- It matched five other configurations in complete correctness.
- Jev had the lowest median latency among successful responses: **0.335 seconds**.
- Some models produced the correct final label despite selecting the wrong scientific relation. In one culture-history case, the final conclusion remained “inconsistent,” but the resulting counts were wrong.

The study therefore argues that scientific model evaluation should not focus only on the final claim label. It should also verify **which relation the model selected and whether the resulting scientific quantities are correct**. Jev appears useful for prepared scientific decision tasks in which the evidence and candidate relations are explicit and the numerical operations are delegated to code.


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


### 1. 연구 목적과 전체 구조
이 논문은 과학적 의사결정 과정에서 **모델이 주어진 증거를 바탕으로 여러 후보 관계 중 하나를 선택할 수 있는지** 평가한다.  
선택된 관계가 이후의 계산에 영향을 주므로, 모델의 답변 자체뿐 아니라 그 결과로 산출되는 **과학적 수치와 최종 주장 라벨**도 함께 평가했다.

전체 구조는 다음과 같다.

1. 모델에 과학적 근거 자료와 구조화된 기록을 제공한다.
2. 모델이 각 질문에 대해 후보 관계 중 하나를 선택한다.
3. 선택 결과를 결정론적 프로그램에 전달한다.
4. 프로그램이 개수, Boolean 값, 수식, 미확정 값 및 최종 라벨을 계산한다.
5. 모델의 선택, 계산 결과, 최종 라벨을 각각 정답과 비교한다.

이를 식으로 표현하면 다음과 같다.

 [
 hat r=f(e,Q),  qquad ( hat z, hat y)=g(e, hat r)
 ]

-  (e ): 제공된 증거와 기록  
-  (Q ): 과학적 선택 질문  
-  ( hat r ): 모델이 선택한 관계  
-  ( hat z ): 프로그램이 계산한 과학적 결과  
-  ( hat y ): 최종 주장 라벨  

---

### 2. 사용한 모델과 인터페이스
비교 대상은 총 **12개 모델 구성(configuration)** 이다.

- **Jev 1.13**
- GPT-5.6 계열: Luna, Terra, Sol
- GPT-6 Astra
- Claude Sonnet 5
- Claude Opus 5
- Qwen3.8 Flash
- Qwen3.8 Max
- DeepSeek V4.1 Flash
- DeepSeek V4 Pro
- Kimi K3

Jev는 문서화된 **native Choice 인터페이스**를 사용했다. 즉, 미리 정의된 후보 의미 중 하나를 타입에 맞게 선택한다. 다른 모델들은 후보 식별자를 **제약된 JSON 형식**으로 반환하도록 구성했다.

논문은 Jev 내부의 신경망 구조, 파라미터 수, 사전학습 데이터, 파인튜닝 방식 등은 공개하지 않는다. 따라서 이 연구의 핵심은 새로운 모델 아키텍처를 제안하는 것이 아니라, **Jev를 과학적 선택 모듈로 사용할 때의 성능과 비용을 평가하는 것**이다.

---

### 3. 과학적 평가 데이터
평가는 다음과 같이 구성되었다.

- 모델이 처리하는 과학적 사례: **10개 그룹**
- 사례당 질문: **2개**
- 총 과학적 Choice: **20개**
- 모델별 반복: **5회**
- 모델별 요청 수: **50회**
- 모델별 semantic answer 수: **100개**

사례는 다음과 같은 과학적 관계를 포함한다.

- 실험 단위와 표본의 관계
- 동일 배양액 또는 독립 배양액 여부
- 재료·촉매·분말의 공유 이력
- 반복 측정과 동일 사건의 관계
- 시뮬레이션 쌍과 평균값의 관계
- 보정 표준의 공유 불확실성
- 처리군 및 실험 배정 관계
- RNA pooling, 모자이시즘, 세대 간 노출 등

자료는 논문, 보충자료, 과학 프로젝트 문서, 방법론 핸드북, 산업 사례 보고서 등 **18개 source family와 21개 기록**을 바탕으로 구성되었다. 모델에는 관련 증거와 후보 의미가 제공되지만, 정답 관계와 정답 계산 결과는 제공되지 않았다.

이 데이터는 모델 학습용 데이터라기보다, **이미 준비된 증거를 이용한 평가용 개발 컬렉션**이다. 논문에서 별도의 모델 재학습이나 파인튜닝은 수행하지 않았다.

---

### 4. 모델과 코드의 역할 분리
이 연구의 핵심 기법은 **semantic decision과 numerical computation을 분리하는 것**이다.

#### 모델이 수행하는 작업
- 증거 문장을 해석한다.
- 질문에 맞는 후보 관계를 선택한다.
- 예를 들어 “10개의 표본이 하나의 배양 이력을 공유하는가?”와 같은 관계를 판단한다.

#### 코드가 수행하는 작업
- 선택된 관계에 따라 개수 계산
- 필터링과 중복 제거
- 산술 연산
- Boolean 관계 계산
- 기호식 및 공분산 항 계산
- 최종 claim label 조합

따라서 모델에게 계산이나 최종 결론을 직접 생성하게 하지 않고, 모델은 **관계 선택만 담당**한다. 이 방식은 모델의 언어적 판단과 계산 오류를 분리하고, 동일한 계산 환경에서 여러 모델을 공정하게 비교하기 위한 것이다.

---

### 5. 입력 설계와 특별한 평가 기법
각 모델에는 다음이 포함된 동일한 입력 패킷이 제공되었다.

- 출처 기반 증거 문단
- 구조화된 실험 기록
- 질문 지시문
- 후보 관계의 의미
- 필요한 경우 불충분한 증거·모순·기타 선택지

또한 다음과 같은 설계 원칙을 사용했다.

- 질문을 **원자적 질문(atomic question)** 으로 분리
- 각 질문을 관련 증거에 직접 연결
- 두 질문을 하나의 요청으로 제출
- 계산 규칙과 라벨 조합 규칙을 코드로 고정
- 누락 응답은 임의로 대체하거나 재시도하지 않음
- 성공한 요청의 latency와 비용을 별도로 측정

---

### 6. 세 가지 정확도 평가
논문은 단순히 최종 라벨만 맞았는지 보지 않고, 세 수준으로 평가했다.

#### ① Semantic correctness
모델이 정답 후보 관계를 선택했는지 평가한다.

#### ② Downstream exactness
선택된 관계를 코드에 넣어 계산한 결과가 정답과 일치하는지 평가한다.  
대상에는 다음이 포함된다.

- 개수
- 그룹별 개수
- Boolean 관계
- 기호식
- 명시적 unknown 값

#### ③ Label correctness
최종 주장의 라벨이 정답과 일치하는지 평가한다.

추가로 **joint correctness**도 계산했다. 이는 모든 관계 선택과 모든 downstream 결과가 동시에 정확해야 정답으로 인정하는 지표다.

이러한 구분은 중요한데, 잘못된 관계 선택이 있어도 최종 라벨은 우연히 맞을 수 있기 때문이다. 실제로 배양 이력 사례에서 일부 모델은 성장 이력 개수를 잘못 계산했지만 최종 라벨은 맞혔다.

---

### 7. 실험 반복과 자원 측정
각 구성은 동일한 입력에 대해 5회 반복되었다. 사례 순서와 모델 순서는 섞어서 특정 순서 효과를 줄였다.

측정한 자원 지표는 다음과 같다.

- 성공한 요청의 중앙 latency
- 성공한 요청의 95백분위 latency
- 성공 응답당 평균 비용

논문에서 Jev는 완전한 semantic correctness와 downstream correctness를 보였으며, 성공 요청 기준 중앙 latency는 **0.335초**로 가장 낮았다. 다만 이 결과는 특정 서비스 조건과 평가 데이터에 대한 결과이며, 전체 시스템 구축 비용이나 데이터 준비 비용은 포함하지 않는다.

---

### 8. 방법론의 한계
- 평가 사례가 10개 그룹, 20개 Choice로 제한되어 있다.
- 반복마다 동일한 큐레이션 영어 입력을 사용했다.
- 모델 오류 7건이 하나의 culture-history 질문에 집중되었다.
- 모든 모델 라벨 정답이 `inconsistent`였기 때문에 라벨 구분 능력은 충분히 평가되지 않았다.
- Jev 자체의 내부 아키텍처나 학습 데이터는 분석하지 않았다.
- 평가용 구성과 모델별 설정 차이가 존재할 수 있다.
- latency와 비용은 서비스 상태, 캐시, 추론 설정에 따라 달라질 수 있다.

즉, 이 방법은 Jev가 **증거가 준비되어 있고 후보 관계가 명시된 과학적 의사결정 지점**에서 유용한지를 평가하는 방법이며, 일반적인 과학 질문응답 능력이나 새로운 과학 문제에 대한 전이 성능을 직접 측정한 것은 아니다.

---




## Method

### 1. Overall objective and workflow
The paper evaluates whether a model can select the correct scientific relation from a set of explicit candidate meanings. Because this relation determines later calculations, the study evaluates not only the model’s selection but also the resulting scientific quantities and final claim label.

The workflow is:

1. Provide evidence passages and structured records to the model.
2. Ask the model to select one candidate relation for each question.
3. Pass the selected relations to deterministic code.
4. Compute counts, Boolean values, symbolic expressions, unknown values, and claim labels.
5. Compare the semantic selections, downstream outputs, and final labels with separate references.

Formally:

 [
 hat r=f(e,Q),  qquad ( hat z, hat y)=g(e, hat r)
 ]

where  (e ) is the supplied evidence,  (Q ) is the set of questions,  ( hat r ) is the model-selected relation,  ( hat z ) is the derived scientific output, and  ( hat y ) is the final claim label.

---

### 2. Models and interfaces
The study compares 12 model configurations:

- Jev 1.13
- GPT-5.6 Luna, Terra, and Sol
- GPT-6 Astra
- Claude Sonnet 5
- Claude Opus 5
- Qwen3.8 Flash
- Qwen3.8 Max
- DeepSeek V4.1 Flash
- DeepSeek V4 Pro
- Kimi K3

Jev uses its native, typed **Choice interface**, while the other models return candidate identifiers through a constrained JSON schema.

The paper does not report Jev’s internal neural architecture, parameter count, pretraining corpus, or fine-tuning procedure. Thus, the contribution is not a new model architecture or a new training method. Instead, it evaluates Jev as a semantic decision component within a scientific workflow.

---

### 3. Evaluation materials
The evaluation contains:

- 10 scientific case groups
- 2 questions per group
- 20 model-routed scientific Choices
- 5 repetitions per configuration
- 50 requests per configuration
- 100 semantic answers per configuration

The cases cover experimental units, shared culture histories, material and catalyst reuse, repeated measurements, paired simulations, calibration uncertainty, treatment allocation, RNA pooling, mosaicism, and developmental exposure.

The evidence was constructed from research articles, supplementary materials, scientific project documentation, methodological handbooks, and industrial case reports. The collection contains 21 source records from 18 source families. Reference relations and expected outputs were withheld from the models.

The collection is an evaluation/development collection, not a training set. The paper does not perform additional training or fine-tuning of the compared models.

---

### 4. Separation of model reasoning and computation
The central methodological design is to separate **semantic interpretation from numerical computation**.

#### Model responsibilities
- Interpret the supplied evidence.
- Select the appropriate candidate relation.
- Determine relations such as whether observations share a culture history or experimental unit.

#### Code responsibilities
- Count and filter records.
- Deduplicate observations.
- Perform arithmetic.
- Compute Boolean relations and symbolic expressions.
- Compose the final claim label.

This design prevents the model from directly performing the calculation or composing the final answer. The model makes the semantic decision, while deterministic code ensures consistent downstream computation across all models.

---

### 5. Input design and evaluation techniques
Each model receives the same prepared packet containing:

- Source-grounded evidence passages
- Structured experimental records
- Question instructions
- Candidate meanings
- Options for missing evidence, contradiction, or outside candidates when relevant

The harness also uses:

- Atomic questions linked to specific evidence
- A fixed model–code division of labor
- The same downstream program for all configurations
- Two questions submitted together in one request
- No retries or provider substitution
- Explicit treatment of missing responses as unavailable rather than correct or incorrect answers

---

### 6. Three levels of correctness
The paper reports three separate forms of correctness.

#### Semantic correctness
Whether the model selected the reference candidate relation.

#### Downstream exactness
Whether the deterministic program produced the exact expected scientific outputs, including counts, Boolean properties, symbolic expressions, and explicit unknown values.

#### Label correctness
Whether the final claim label matched the reference label.

The paper also reports joint correctness, which requires both all semantic selections and all downstream outputs to be correct.

This separation is important because a wrong relation can sometimes produce the correct final label. In the culture-history case, some models produced incorrect growth-history counts while still receiving the correct final label.

---

### 7. Repetitions and resource measurements
Each configuration was run five times on the same prepared inputs. Case order was shuffled and model order was rotated to reduce ordering effects.

The study measured:

- Median latency among successful requests
- 95th-percentile latency
- Mean cost per successful response

Jev achieved complete observed semantic and downstream correctness and had the lowest successful-request median latency, 0.335 seconds. These results are specific to the evaluated service conditions and do not include preparation costs or all workflow-level expenses.

---

### 8. Limitations
- The evaluation includes only 10 case groups and 20 model-routed Choices.
- The same curated English packets were repeated across runs.
- All observed semantic errors were concentrated in one culture-history question.
- All reference labels were `inconsistent`, limiting evaluation of label discrimination.
- Jev’s internal architecture and training data were not analyzed.
- Latency and cost may vary with service conditions, caching, and inference settings.

Overall, the method evaluates Jev at a bounded scientific decision point where evidence is prepared, candidate relations are explicit, and downstream operations are deterministic. It does not directly measure general scientific question answering or transfer to unseen scientific domains.


<br/>
# Results



### 1. 연구 목적
이 논문은 **Jev가 과학적 의사결정에서 문맥에 맞는 관계·의미를 선택하고, 그 선택이 계산 결과에 미치는 영향을 얼마나 정확하고 효율적으로 처리하는지** 평가한다.

핵심 아이디어는 다음과 같다.

> 모델은 과학적 관계를 선택하고, 계산·카운팅·필터링·최종 라벨 판정은 코드가 수행한다.

따라서 단순히 최종 답변이 맞는지만 보지 않고,  
① 의미 선택, ② 계산 결과, ③ 최종 주장 라벨을 분리해 평가했다.

---

### 2. 테스트 데이터와 실험 설계

- **과학적 사례:** 10개
- **모델이 판단하는 Choice:** 20개  
  - 사례당 2개 질문
- **반복 횟수:** 구성별 5회
- **구성별 계획 요청 수:** 50회
- **구성별 semantic answer 수:** 100개
- **비교 구성:** 총 12개

각 모델은 동일한 다음 정보를 받았다.

- 원문에서 준비된 근거 문단
- 구조화된 실험 기록
- 질문 지시문
- 선택 가능한 후보 관계와 의미

반면, 정답 관계·정답 계산 결과·정답 라벨은 모델에게 제공하지 않았다.

사례는 재료·제조, 기후 시뮬레이션, 계측, 생물학 실험 등을 포함한다. 대표적으로 다음과 같은 판단을 요구했다.

- 여러 측정값이 하나의 배양 이력에서 왔는지, 서로 다른 배양에서 왔는지
- 시뮬레이션이 독립적인지 쌍으로 연결되어 있는지
- 실험 단위가 시료인지, 배치인지, 라이브러리인지
- 보정 오차가 서로 공유되는지 독립적인지
- 유전적 변이나 노출 이력이 어느 세대까지 전달되는지

특히 Luria–Delbrück 배양 이력 사례에서는 10개 샘플이 하나의 배양에서 왔는지, 각각 별도 배양에서 왔는지를 구분해야 했다.

---

### 3. 비교 대상

비교에는 Jev와 11개의 다른 모델·설정이 포함되었다.

- Jev 1.13
- GPT-5.6 Luna
- GPT-5.6 Terra
- GPT-5.6 Sol
- GPT-6 Astra
- Claude Sonnet 5
- Claude Opus 5
- Qwen3.8 Flash
- Qwen3.8 Max 0902
- DeepSeek V4.1 Flash
- DeepSeek V4 Pro 0813
- Kimi K3

모든 구성은 가능한 한 동일한 입력과 동일한 후처리 코드를 사용했다. Jev는 native Choice 인터페이스를 사용했고, 다른 모델들은 후보 식별자를 제한된 JSON 형식으로 반환했다.

---

### 4. 평가 메트릭

논문은 결과를 세 단계로 나누어 평가했다.

#### ① Semantic correctness
모델이 각 Choice에서 **정답 관계·의미를 선택했는지** 평가한다.

예를 들어 “10개 샘플이 하나의 배양에서 나온 것인가, 10개의 독립 배양에서 나온 것인가?”라는 질문에 올바른 후보를 골랐는지를 본다.

#### ② Downstream exactness
선택된 관계를 코드에 넣었을 때 산출되는 **과학적 계산 결과가 정확한지** 평가한다.

평가 대상에는 다음이 포함된다.

- 개수와 그룹별 개수
- Boolean 관계
- 기호식 또는 공분산 항
- 근거상 알 수 없는 값의 처리
- 명시된 unknown 값의 유지

#### ③ Label correctness
최종 주장의 라벨이 정답과 일치하는지 평가한다.

가능한 라벨은 주로 다음과 같다.

- consistent
- inconsistent
- insufficient evidence

다만 이 실험의 모델 라우팅 사례에서는 정답 라벨이 모두 **inconsistent**였기 때문에, 최종 라벨만 평가하면 항상 inconsistent를 답하는 기준선도 높은 점수를 얻을 수 있다는 한계가 있다.

#### ④ Joint correctness
모든 semantic 선택과 downstream 결과가 동시에 맞는지를 평가한다.

또한 응답 누락은 오답으로 임의 대체하지 않고, 계획된 전체 요청 수를 기준으로 별도 감점했다. 따라서 정확도뿐 아니라 **응답 커버리지**도 고려했다.

---

### 5. 주요 결과

#### 정확도

Jev는 다음을 모두 달성했다.

- 계획된 semantic answer 100개 중 **100개 정답**
- 50개 downstream 결과 중 **50개 정답**
- 최종 라벨 50개 중 **50개 정답**
- joint correctness도 **50/50**

완전한 semantic correctness를 기록한 모델은 Jev 외에도 다음과 같았다.

- GPT-5.6 Sol
- GPT-6 Astra
- Claude Sonnet 5
- Claude Opus 5
- Kimi K3

다른 모델에서는 오류가 발생했다.

- GPT-5.6 Luna: semantic 1건 오류
- GPT-5.6 Terra: 응답 누락으로 계획 분모 기준 점수 하락
- Qwen3.8 Flash: semantic 오류와 응답 누락
- Qwen3.8 Max: semantic 5건 오류
- DeepSeek 계열: 일부 응답 누락

모델의 **7개 semantic 오류는 모두 같은 배양 이력 문제**에서 발생했다.

---

### 6. 최종 라벨만 보면 오류가 사라지는 문제

배양 이력 사례에서 정답 성장 이력 개수는 `(1, 10)`이었다.  
그러나 잘못 선택한 모델들은 다음과 같은 결과를 냈다.

- `(10, 1)` 4회
- `(1, 1)` 2회
- `(10, 10)` 1회

그럼에도 다른 Choice가 “콜로니는 클론 후손이다”라는 점을 올바르게 판단했기 때문에, 최종 주장은 모두 **inconsistent**로 판정되었다.

즉,

> 최종 결론은 맞았지만, 이후 분석에 사용될 과학적 수량은 틀릴 수 있었다.

실제로 받은 전체 결과 588건에서:

- 최종 라벨 정답: **588/588**
- downstream 결과 정답: **581건**
- 차이: **7건**

이 결과는 과학적 시스템 평가에서 최종 라벨만 확인해서는 부족하며, 중간 관계 선택과 계산 결과도 함께 확인해야 한다는 점을 보여준다.

---

### 7. 속도와 비용

Jev는 완전한 semantic correctness를 달성한 구성 중 가장 낮은 관측 지연시간을 보였다.

| 구성 | 성공 요청 중앙 latency | p95 | 성공 응답당 평균 비용 |
|---|---:|---:|---:|
| **Jev** | **0.335초** | **0.442초** | 약 **$0.000060** |
| GPT-5.6 Sol | 1.385초 | 2.370초 | 약 $0.003047 |

논문은 Jev가 이 데이터셋에서 다른 완전 정확 모델들과 비슷한 정확도를 보이면서도, **더 빠르고 저렴한 선택지**였다고 결론짓는다.

비용에는 데이터 준비 비용, 로컬 계산 비용, 누락 요청 비용 등이 포함되지 않았으므로 전체 워크플로 비용과는 다르다.

---

### 8. 논문의 결론과 한계

#### 결론
Jev는 다음 조건에서 유용한 구성요소로 평가되었다.

- 근거 자료가 사전에 준비되어 있고
- 후보 관계가 명시적으로 주어지며
- 계산 규칙이 코드로 정의되어 있고
- 모델은 의미 선택에 집중하는 경우

또한 관계 수준의 선택을 보존하면 어떤 의미 판단이 계산 오류를 일으켰는지 추적할 수 있다.

#### 한계

- 모델이 판단한 사례는 10개, Choice는 20개로 규모가 작다.
- 같은 영어 입력을 반복 사용했기 때문에 일반화 성능을 직접 보여주지는 않는다.
- 6개 구성이 완전 정확도를 기록해 모델 간 차이가 제한적이다.
- semantic 오류 7개가 하나의 배양 이력 질문에 집중되었다.
- 모든 정답 최종 라벨이 inconsistent여서 라벨 판별 능력은 충분히 평가되지 않았다.
- 결과는 특정 서비스 환경, 모델 버전, 지연시간 조건에 의존한다.
- 모델별 내부 구성요소의 효과를 분리하는 ablation 연구는 수행하지 않았다.

### 한 줄 요약
**Jev는 준비된 과학적 선택 문제에서 다른 최고 성능 구성과 동일한 관측 정확도를 보이면서 가장 낮은 지연시간을 기록했지만, 이 연구는 최종 라벨이 아니라 의미 선택과 downstream 수량까지 함께 평가해야 한다는 점을 특히 강조한다.**

---




### 1. Research goal
The paper evaluates whether **Jev can make correct semantic or relational choices in scientific workflows and whether those choices lead to correct computational results**.

The proposed division of labor is:

> The model selects the scientific relation; deterministic code performs counting, filtering, arithmetic, and final label composition.

Therefore, the study evaluates three separate levels:

1. semantic selection,
2. downstream scientific outputs,
3. final claim labels.

---

### 2. Test data and experimental design

- **Scientific cases:** 10
- **Model-routed Choices:** 20
- **Questions per case:** 2
- **Repetitions per configuration:** 5
- **Planned requests per configuration:** 50
- **Semantic answers per configuration:** 100
- **Compared configurations:** 12

All models received the same prepared evidence passages, structured records, instructions, and candidate meanings. Reference choices, reference outputs, and final labels were withheld.

The cases covered manufacturing and materials, climate simulation, metrology, cosmology, experimental design, and biological experiments. The decisions included questions about:

- shared versus independent culture histories,
- paired versus independent simulations,
- experimental units,
- shared calibration uncertainty,
- pooled biological samples,
- exposure and inheritance relations.

---

### 3. Compared models

The comparison included:

- Jev 1.13
- GPT-5.6 Luna
- GPT-5.6 Terra
- GPT-5.6 Sol
- GPT-6 Astra
- Claude Sonnet 5
- Claude Opus 5
- Qwen3.8 Flash
- Qwen3.8 Max 0902
- DeepSeek V4.1 Flash
- DeepSeek V4 Pro 0813
- Kimi K3

The same evidence and deterministic downstream program were used across configurations. Jev used its native Choice interface, while the other models returned candidate identifiers through constrained JSON.

---

### 4. Evaluation metrics

#### Semantic correctness
Whether the model selected the exact reference relation for every Choice.

#### Downstream exactness
Whether the deterministic program produced the correct scientific outputs, including counts, Boolean relations, symbolic terms, and explicitly unknown values.

#### Label correctness
Whether the final claim label matched the reference label.

A limitation is that all model-routed reference labels in this collection were **inconsistent**. Thus, a trivial always-inconsistent baseline could perform well on the label task.

#### Joint correctness
Whether all semantic selections and all declared downstream outputs were correct simultaneously.

Missing responses received no correctness credit under the planned-denominator evaluation, so response coverage was measured separately from semantic accuracy.

---

### 5. Main results

Jev achieved:

- **100/100** semantic answers correct
- **50/50** downstream results correct
- **50/50** final labels correct
- **50/50** joint correctness

The following configurations also achieved complete observed semantic correctness:

- GPT-5.6 Sol
- GPT-6 Astra
- Claude Sonnet 5
- Claude Opus 5
- Kimi K3

Other systems had either semantic errors or missing responses. The seven observed semantic errors all concerned the same culture-history question.

---

### 6. Correct labels can hide incorrect quantities

In the Luria–Delbrück culture-history case, the correct growth-history counts were `(1, 10)`. Incorrect selections produced:

- `(10, 1)` four times
- `(1, 1)` twice
- `(10, 10)` once

However, the models correctly recognized that the colonies were clonal descendants. As a result, all final labels still became **inconsistent**.

Across 588 received results:

- Final labels correct: **588/588**
- Downstream outputs exact: **581**
- Hidden quantity errors: **7**

This demonstrates that a correct final verdict does not guarantee that the scientific quantities passed to later analysis are correct.

---

### 7. Latency and cost

Jev had the lowest successful-request median latency among the configurations with complete semantic correctness.

| Configuration | Median latency | p95 latency | Approx. cost per successful response |
|---|---:|---:|---:|
| **Jev** | **0.335 s** | **0.442 s** | about **$0.000060** |
| GPT-5.6 Sol | 1.385 s | 2.370 s | about $0.003047 |

Thus, within this collection, Jev combined complete observed correctness with substantially lower latency and response cost.

The cost estimates exclude preparation, local computation, and some charges related to missing responses.

---

### 8. Conclusion and limitations

#### Conclusion
Jev appears useful when:

- evidence has already been prepared,
- candidate relations are explicit,
- downstream operations are defined deterministically,
- the model is responsible mainly for semantic selection.

The study also shows that preserving relation-level decisions makes downstream discrepancies easier to diagnose.

#### Limitations

- The evaluation used only 10 model-routed cases and 20 Choices.
- The same curated English packets were repeated across runs.
- Six configurations achieved complete correctness, limiting model differentiation.
- All semantic errors occurred on one culture-history question.
- Because every reference label was inconsistent, label discrimination and abstention were not fully tested.
- Latency and cost depend on model versions, service conditions, and cache state.
- No ablation study isolated the effect of individual system components.

### One-sentence summary
**Jev matched the best-performing configurations in observed accuracy and achieved the lowest latency, while the evaluation shows that scientific systems must assess semantic relations and downstream quantities—not only final claim labels.**


<br/>
# 예제



### 1. 이 논문에서 말하는 “트레이닝 데이터”와 “테스트 데이터”

이 논문은 일반적인 지도학습처럼 **트레이닝 데이터로 모델을 학습시킨 뒤 별도의 테스트셋에서 평가한 연구가 아니다.**  
Jev와 다른 모델들은 이미 준비된 과학적 자료를 입력받아, 정해진 후보 관계 중 하나를 선택하는 방식으로 평가되었다.

- 전체 개발 컬렉션: **20개 case group, 40개 Choice**
- 모델이 판단한 부분: **10개 그룹, 20개 Choice**
- 코드가 규칙만으로 처리한 부분: **10개 그룹, 20개 Choice**
- 각 모델은 10개 모델 평가 사례를 **5회 반복**
- 따라서 모델별 계획된 요청 수: **50회**
- 한 요청에는 한 그룹의 두 Choice가 함께 포함됨

즉, 논문에는 일반적인 의미의 “학습 데이터”와 “테스트 데이터”가 명확히 분리되어 있지 않다. 반복 실험은 새로운 테스트 데이터를 추가한 것이 아니라 **같은 입력을 반복하여 응답 변동성을 측정한 것**이다.

---

### 2. 모델에 제공되는 구체적인 입력

각 요청에는 다음 정보가 포함된다.

1. **과학적 근거 자료**
   - 논문, 실험 기록, 프로젝트 문서, 방법론 자료 등에서 준비한 발췌문
2. **구조화된 기록**
   - 표본, 실험 단위, 처리 조건, 관측값 등의 정리된 데이터
3. **질문 지시문**
   - 어떤 범위와 관점에서 판단해야 하는지 설명
4. **후보 의미 또는 관계**
   - 모델이 선택해야 하는 명시적인 선택지
5. **계산에 필요한 대상**
   - 모델이 직접 계산하는 것이 아니라, 선택 결과가 이후 코드 계산에 사용됨

예를 들어 질문은 다음과 같은 형식이다.

> “이 관측값들은 하나의 배양 이력을 공유하는가, 아니면 서로 독립적인 배양에서 얻어진 것인가?”

모델은 자유롭게 설명하는 대신 다음과 같은 후보 중 하나를 선택한다.

- shared culture history
- separate culture histories
- reverse assignment
- insufficient evidence
- conflicting evidence
- other

---

### 3. 실제 테스크의 핵심

이 연구의 테스크는 단순한 과학 상식 질의응답이 아니다.

> **근거 자료를 읽고, 관측값 사이의 과학적 관계를 선택한 뒤, 그 관계를 사용하는 후속 계산의 결과가 정확한지 평가하는 테스크**이다.

전체 과정은 다음과 같다.

```text
근거 자료 + 구조화된 기록
        ↓
모델의 의미적 선택
        ↓
공유된 결정론적 코드
        ↓
과학적 수량 계산
        ↓
최종 주장 라벨 생성
```

논문은 이 결과를 세 단계로 나누어 평가한다.

1. **Semantic correctness**
   - 모델이 올바른 관계를 선택했는가?
2. **Downstream exactness**
   - 선택된 관계를 바탕으로 계산된 수량이 정확한가?
3. **Label correctness**
   - 최종 주장의 라벨이 맞는가?
   - 예: consistent, inconsistent, insufficient evidence

---

### 4. 구체적인 예시: B002 미생물 배양 이력

#### 입력

근거 자료는 Luria와 Delbrück의 박테리아 돌연변이 실험이다. 두 실험군이 있다.

- **column 11a**: 10개 샘플이 하나의 배양에서 유래
- **column 11**: 10개 샘플이 각각 별도로 배양됨
- 선택된 열에는 각각 514개와 624개의 colony 기록이 있음

#### Choice 1

질문:

> 각 시리즈 안의 샘플들이 동일한 배양 이력을 공유하는가?

후보 선택지의 예:

- 11a와 11 모두 shared culture history
- 11a는 shared, 11은 separate
- 11a는 separate, 11은 shared
- 11a와 11 모두 separate
- evidence insufficient/conflicting

정답:

```text
11a = shared culture history
11  = separate culture histories
```

#### Choice 2

질문:

> 각 colony는 독립적인 돌연변이 기원을 나타내는가, 아니면 동일한 기원의 clonal descendant일 수 있는가?

정답:

```text
colonies are clonal descendants;
exact number of mutation origins is unknown
```

#### 코드가 생성하는 출력

모델의 선택을 코드에 넣으면 다음과 같은 결과가 나온다.

```text
growth-history counts = (1, 10)
colony totals = 514 and 624
mutation-origin count = unknown
final claim label = inconsistent
```

여기서 `(1, 10)`은 11a가 하나의 배양 이력, 11이 열 개의 독립 배양 이력을 가진다는 뜻이다.

---

### 5. 잘못된 의미 선택이 만들어내는 문제

일부 모델은 첫 번째 Choice를 잘못 선택했다. 그 결과 성장 이력 수가 다음처럼 계산되었다.

```text
(10, 1) 4회
(1, 1) 2회
(10, 10) 1회
```

하지만 두 번째 Choice는 올바르게 선택했기 때문에, 최종적으로 “주장이 일관되지 않다”라는 라벨은 계속 맞았다.

따라서 이 사례는 다음을 보여준다.

```text
최종 라벨은 맞음
하지만 과학적 수량은 틀림
```

논문 전체에서도 최종 라벨은 모든 수신 응답에서 맞았지만, downstream output은 588개 중 581개만 정확했다. 즉, **라벨만 평가하면 잘못된 과학적 수량을 놓칠 수 있다.**

---

### 6. 다른 입력·출력 예시

#### E004: 분말 재사용 이력

입력:

- 여러 생산 배치에서 재사용된 분말
- 일부 provenance만 확인 가능
- 모든 분말이 하나의 동일한 원래 lot에서 왔는지는 알 수 없음

모델의 선택:

```text
mixed reuse histories
partial provenance coverage
```

출력:

```text
exact parent-lot count = unknown
complete lot assignment = unavailable/incomplete
```

#### E006: 우주론 시뮬레이션

입력:

- 위상 반전된 초기 조건을 가진 시뮬레이션 쌍
- 각 쌍의 평균값을 앙상블 통계에 사용

모델의 선택:

```text
initial fields are phase-reversed paired fields
pair average is the aggregation unit
```

출력:

```text
paired realizations = 100
simulation executions = 200
ensemble values = 100
```

#### B003: RNA-seq 풀링

입력:

- 여러 동물의 RNA를 library 제작 전에 물리적으로 pooling
- 선택된 네 개의 library
- 총 16명의 contributor, 32회의 contributor appearance

모델의 선택:

```text
RNA was pooled before library preparation
```

출력:

```text
libraries = 4
distinct contributors = 16
contributor appearances = 32
separately observed individual profiles = 0
```

---

### 7. 이 논문에서의 최종 평가 결과

Jev의 결과는 다음과 같다.

- semantic selections: **100/100 정답**
- downstream outputs: **50/50 정답**
- final labels: **50/50 정답**
- 성공 응답의 중앙 latency: **0.335초**
- 평균 응답 비용: 약 **$0.000060**

다만 이 결과는 다음 조건에 한정된다.

- 근거 자료가 이미 준비되어 있음
- 후보 관계가 명시되어 있음
- 계산 규칙이 코드로 정의되어 있음
- 사례 수가 10개로 제한됨
- 모든 모델 평가 라벨이 `inconsistent`였음

따라서 이 논문은 Jev가 일반적인 과학적 추론 전체를 해결한다는 뜻이 아니라, **준비된 근거와 명시적인 선택지가 있는 과학 워크플로의 특정 의미 결정 지점에서 유용한지 평가한 연구**이다.

---




### 1. What “training data” and “test data” mean in this paper

This is not a conventional supervised-learning study in which a model is trained on one dataset and then evaluated on a separate test set.

Instead, Jev and the other models received prepared scientific evidence and selected one relation from a fixed set of candidate meanings.

- Full development collection: **20 case groups and 40 Choices**
- Model-routed portion: **10 groups and 20 Choices**
- Code-resolved portion: **10 groups and 20 Choices**
- Each model-routed case was repeated **five times**
- Planned requests per configuration: **50**
- Each request contained two Choices from the same case group

Therefore, the paper does not define a conventional training/test split. Repeated runs used the same inputs and measured response variation rather than introducing new test examples.

---

### 2. Concrete model inputs

Each request contains:

1. **Source-grounded evidence passages**
2. **Structured records**
3. **Question instructions and scope**
4. **Explicit candidate meanings or relations**
5. **A downstream computation defined in code**

For example, the model may be asked:

> “Do the observations share one culture history, or do they come from separately grown cultures?”

The model selects among explicit options such as:

- shared culture history
- separate culture histories
- reverse assignment
- insufficient evidence
- conflicting evidence
- other

The model is not expected to perform the numerical calculation itself.

---

### 3. Core task

The task is:

> **Read the evidence, select the scientifically correct relation, and determine whether that relation produces the correct downstream scientific quantities.**

The workflow is:

```text
Evidence and structured records
        ↓
Model semantic selection
        ↓
Shared deterministic code
        ↓
Scientific quantities
        ↓
Final claim label
```

The paper evaluates three levels:

1. **Semantic correctness**
   - Was the correct relation selected?
2. **Downstream exactness**
   - Were the resulting scientific quantities correct?
3. **Label correctness**
   - Was the final claim label correct?

---

### 4. Concrete example: B002, microbial culture histories

#### Input

The evidence comes from the Luria–Delbrück bacterial mutation experiment.

- **Column 11a**: ten samples came from one culture
- **Column 11**: ten samples came from separately grown cultures
- The selected columns contain 514 and 624 colony records

#### Choice 1

Question:

> Do the samples within each series share a culture history?

Reference answer:

```text
11a = shared culture history
11  = separate culture histories
```

#### Choice 2

Question:

> Do the colonies represent independent mutation origins, or can they be clonal descendants?

Reference answer:

```text
colonies are clonal descendants;
the exact number of mutation origins is unknown
```

#### Code-generated output

```text
growth-history counts = (1, 10)
colony totals = 514 and 624
mutation-origin count = unknown
final claim label = inconsistent
```

---

### 5. How an incorrect semantic choice can be hidden

Some models incorrectly selected the culture-history relation. Their downstream growth-history counts became:

```text
(10, 1) four times
(1, 1) twice
(10, 10) once
```

However, because the second Choice was correct, the final label remained `inconsistent`.

Thus:

```text
The final label was correct,
but the scientific quantity was wrong.
```

Across the received results, all final labels were correct, but only 581 of 588 downstream outputs were exact. This demonstrates why evaluating only the final label can miss an important scientific error.

---

### 6. Additional examples

#### E004: Powder-reuse histories

Input:

- Powder was reused across production histories
- Provenance was only partially available

Model selections:

```text
mixed reuse histories
partial provenance coverage
```

Outputs:

```text
exact parent-lot count = unknown
complete lot assignment = incomplete
```

#### E006: Paired cosmological simulations

Input:

- Simulations used phase-reversed paired initial fields
- Pair averages were used as the ensemble statistic

Outputs:

```text
paired realizations = 100
simulation executions = 200
ensemble values = 100
```

#### B003: RNA-seq pooling

Input:

- RNA from multiple animals was pooled before library preparation
- Four selected libraries contained 16 contributors and 32 contributor appearances

Outputs:

```text
libraries = 4
distinct contributors = 16
contributor appearances = 32
separately observed individual profiles = 0
```

---

### 7. Main evaluation result

Jev achieved:

- Semantic selections: **100/100 correct**
- Downstream outputs: **50/50 correct**
- Final labels: **50/50 correct**
- Median successful-request latency: **0.335 seconds**
- Approximate mean cost: **$0.000060 per response**

These findings apply only to the studied setting:

- Evidence was already prepared
- Candidate relations were explicit
- Downstream calculations were implemented in code
- The evaluation used only ten model-routed cases
- All model-routed reference labels were `inconsistent`

Therefore, the paper does not show that Jev solves general scientific reasoning. It evaluates Jev as a **semantic decision component at a bounded point in a scientific workflow**, where the evidence and candidate relations are already specified.

<br/>
# 요약
**한국어**  
1. 12개 모델 설정을 10개 과학 사례의 20개 선택 문제에 적용하고, 각 문제를 5회 반복하여 모델은 의미적 관계를 선택하고 코드는 계산·최종 라벨을 결정하도록 평가했다.  
2. Jev는 의미 선택 100/100, 후속 산출물 50/50, 최종 라벨 50/50으로 완전한 정확도를 보였으며, 성공 요청의 중앙 지연시간도 0.335초로 가장 낮았다.  
3. 예를 들어 배양 이력 문제에서 일부 모델은 공유 배양을 잘못 해석해 성장 이력 수를 (1,10)이 아닌 (10,1) 등으로 계산했지만, 최종 라벨은 여전히 ‘inconsistent’로 맞아 라벨만으로는 수량 오류를 발견하기 어려웠다.  

**English**  
1. The study evaluated 12 model configurations on 20 scientific choice questions across 10 cases, repeated five times, using models for semantic selection and deterministic code for calculations and final labels.  
2. Jev achieved perfect semantic, downstream, and label accuracy—100/100, 50/50, and 50/50 respectively—and had the lowest median latency among successful requests at 0.335 seconds.  
3. In the culture-history example, some models misinterpreted shared cultures and produced incorrect growth-history counts such as (10,1) instead of (1,10), while still obtaining the correct “inconsistent” label, showing that label-only evaluation can hide quantitative errors.

<br/>
# 기타



### 1. Figure 1: 과학적 의사결정 평가 구조

Figure 1은 모델의 역할과 코드의 역할을 분리한 평가 파이프라인을 보여준다.

- **입력**: 근거 문단, 구조화된 기록, 질문, 후보 관계(candidate meanings)
- **모델의 역할**: 각 Choice에 대해 어떤 과학적 관계가 맞는지 선택  
  - 예: 두 표본이 같은 배양 이력을 공유하는가?
  - 예: 관측값이 독립적인 실험 단위인가?
- **코드의 역할**: 모델이 선택한 관계를 바탕으로 개수 계산, 필터링, 중복 제거, 산술 연산 및 최종 주장 라벨 구성
- **평가 단계**
  1. **Semantic correctness**: 모델이 올바른 관계를 선택했는가?
  2. **Downstream exactness**: 그 선택으로 계산된 과학적 수량이 정확한가?
  3. **Label correctness**: 최종적으로 consistent/inconsistent 등의 주장이 맞는가?
  4. **Joint correctness**: 관계 선택과 파생 결과가 모두 맞는가?

**핵심 인사이트:**  
최종 라벨만 평가하면 중간의 잘못된 과학적 관계 선택이나 수량 계산 오류를 놓칠 수 있다. 따라서 모델의 선택, 계산 결과, 최종 결론을 별도로 평가해야 한다.

---

### 2. Table 1: 전체 모델 비교 결과

Table 1은 12개 설정을 10개 사례에 대해 5회씩 평가한 결과다. 계획된 요청은 모델당 50회이며, 각 요청에는 2개의 semantic Choice가 포함된다.

#### Jev의 결과

- 50/50 요청에서 응답
- Semantic answers: **100/100 정답**
- Downstream outputs: **50/50 정답**
- Final labels: **50/50 정답**
- Joint correctness: **50/50**
- 성공 응답의 중앙 latency: **0.335초**
- 95백분위 latency: **0.442초**
- 비용: 약 **$0.0000595/요청**

GPT-5.6 Sol, GPT-6 Astra, Claude Sonnet 5, Claude Opus 5, Kimi K3도 semantic 및 downstream correctness에서 Jev와 같은 완전 정답률을 보였다.

반면:

- Luna: 한 번의 semantic 오류
- Qwen Flash: 한 번의 오류와 일부 미응답
- Qwen Max: 다섯 번의 semantic 오류
- Terra와 DeepSeek 계열: 받은 응답은 대체로 정확했지만 일부 미응답으로 계획 분모 기준 점수가 하락

**핵심 인사이트:**  
Jev는 이 데이터셋에서 완전한 정확도와 가장 낮은 성공 응답 latency를 동시에 달성했다. 다만 사례 수가 적고 특정 서비스 조건에서 측정된 결과이므로, 일반적인 모델 우월성을 의미하지는 않는다.

---

### 3. Table 2: 코드만으로 해결되는 과학적 사례

Table 2는 모델 호출 없이, 준비된 근거와 명시적 규칙만으로 결정할 수 있는 10개 그룹을 정리한다.

사례의 예시는 다음과 같다.

- 콘크리트 혼합 작업과 시편의 공유 여부
- 적층제조에서 하나의 build에 포함된 시편
- 전기 도금 실험의 whole-plot 및 strip 단위
- 반복 측정에서 하나의 wear event와 여러 scan의 구분
- 촉매 재사용 사이클
- 화학 공정 설계에서 동일한 factor setting의 개수

각 사례는 단순한 텍스트 이해 문제가 아니라, **어떤 관계를 실험 단위로 볼 것인지**가 최종 계산을 결정한다.

**핵심 인사이트:**  
과학적 데이터의 “관측값 개수”와 “독립적인 과학적 단위의 개수”는 다를 수 있다. 따라서 계산 전에 실험 단위, 공유 이력, 반복 측정 여부를 먼저 결정해야 한다.

---

### 4. Table 3: 모델이 판단해야 하는 10개 사례

Table 3은 모델 호출이 필요한 10개 그룹과 각 그룹의 두 가지 판단, 그리고 그 결과로 계산되는 수량을 보여준다.

주요 사례는 다음과 같다.

- **E004**: 재활용 분말의 혼합 이력과 원래 lot의 부분적 식별
- **E005**: 이전 결과에 따라 다음 응력 수준이 바뀌는 staircase fatigue testing
- **E006**: 서로 연결된 우주론 시뮬레이션 쌍과 pair-average
- **E007**: 서로 다른 climate-model protocol group
- **E009**: 공유된 calibration uncertainty
- **E010**: 두 산악 지역에 대한 상호보완적 cloud-seeding allocation
- **B001**: F0 어미 개체의 투여가 F3 세대에 미치는 노출 범위
- **B002**: 하나의 배양 이력과 여러 개의 배양 이력
- **B003**: RNA pooling과 개별 생물학적 attribution
- **B004**: founder 내 mosaicism과 유전 가능한 germline transmission

**핵심 인사이트:**  
이 사례들은 대부분 “몇 개가 있는가?”보다 먼저 “무엇이 하나의 단위인가?”, “어떤 관측이 공유된 역사나 의존성을 갖는가?”를 판단해야 한다. 이 관계 판단이 잘못되면 이후의 개수와 비교 결과가 달라진다.

---

### 5. Table 4: downstream exactness에서 평가한 과학적 출력

Table 4는 각 사례에서 모델 선택 이후 코드가 산출해야 하는 구체적인 결과를 정의한다.

예를 들어:

- E004: 정확한 parent-lot 개수, 전체 lot assignment 가능 여부
- E005: 실패 또는 성공 후 다음 stress 값, 고정 stress에서의 반복 수
- E006: paired realization 수, 전체 simulation execution 수, ensemble에 들어가는 값
- E009: 비교된 laboratory 결과 수, calibration 방문 수, 공유 covariance 항
- E010: joint assignment 수, range-level record 수, 배정의 독립성
- B002: series별 plating 수와 growth-history 수, mutation origin 수
- B003: library 수, distinct contributor 수, contributor appearance 수
- B004: founder 수, 보고된 mutant allele 수, 검증된 germline line 수

**핵심 인사이트:**  
모든 semantic distinction이 downstream 숫자를 바꾸는 것은 아니다. 어떤 관계 선택은 최종 주장에는 영향을 주지만 선언된 계산 필드에는 영향을 주지 않을 수 있다. 그래서 **semantic correctness와 downstream exactness를 분리**해야 한다.

---

### 6. Table 5: 응답을 받은 경우에만 계산한 조건부 성능

Table 5는 미응답을 제외하고, 실제로 완전한 응답을 받은 경우의 정확도를 보여준다.

- Jev: semantic **100/100**, downstream/joint **50/50**, label **50/50**
- 일부 모델은 받은 응답만 보면 높은 정확도를 보였지만, 전체 계획 요청 중 미응답이 있었다.
- 전체 588개 응답의 final label은 모두 정답이었다.
- 그러나 downstream output은 **588개 중 581개만 정확**
- semantic Choice는 **1,176개 중 1,169개 정답**
- 즉, **7개의 semantic 오류**가 있었고, 모두 동일한 culture-history 질문에서 발생했다.

**핵심 인사이트:**  
조건부 정확도만 보면 응답 가용성 문제를 숨길 수 있다. 따라서 이 논문은 “정확도”와 “응답률”을 분리해 보고하며, 계획된 분모 기준 결과를 주요 지표로 사용한다.

---

### 7. Culture-history 사례: 라벨이 맞아도 수량이 틀릴 수 있음

논문에서 가장 중요한 오류 분석은 Luria–Delbrück의 culture-history 사례다.

- column 11a: 10개 표본이 **하나의 배양 이력**에서 나옴
- column 11: 10개 표본이 **각각 별도의 배양 이력**에서 나옴
- 정답 growth-history count: **(1, 10)**

일부 모델의 잘못된 선택은 다음과 같은 결과를 만들었다.

- (10, 1): 4회
- (1, 1): 2회
- (10, 10): 1회

그러나 두 번째 Choice에서 colony가 clonal descendant라는 점은 올바르게 판단했기 때문에, 최종 claim label은 계속 **inconsistent**로 남았다.

**핵심 인사이트:**  
최종 결론이 맞더라도, 이후 분석에 전달되는 중간 수량은 틀릴 수 있다. 특히 과학 워크플로에서는 이 잘못된 count가 후속 통계 분석이나 재현성 판단에 사용될 수 있으므로, 라벨만으로 모델을 평가하면 안 된다.

---

### 8. Appendix A: 사례 구성과 reference 해석

Appendix A는 평가 데이터셋과 reference answer를 어떻게 구성했는지 설명한다.

- 전체 개발 컬렉션: 40개 scientific Choices, 20개 case groups
- 모델이 판단하는 부분: 10개 그룹, 20개 Choices
- 코드 규칙으로 해결되는 부분: 10개 그룹, 20개 Choices
- 각 모델에는 근거와 후보 의미만 제공하고 정답은 제공하지 않음
- missing evidence, contradiction, outside candidate 등의 선택지도 포함

특히 Appendix A는 다음과 같은 과학적 불확실성을 명시적으로 보존한다.

- 정확한 parent-lot 수를 알 수 없음
- 실제 mutation origin 수를 알 수 없음
- validated germline line 수가 확정되지 않음
- 배양 이력은 알 수 있지만 mutation origin은 알 수 없음
- 부분적인 provenance만 확인되고 전체 assignment는 불가능함

**핵심 인사이트:**  
모르는 값을 임의로 0이나 특정 숫자로 바꾸지 않고, unknown 상태로 유지하는 것이 중요한 평가 기준이다. 이는 과학적 추론에서 “알 수 없음”과 “없음”을 구분하는 문제를 강조한다.

---

### 9. Appendix B: 지표 정의와 채점 방식

Appendix B는 정확도 계산 방법을 수식으로 정의한다.

주요 지표는 다음과 같다.

- **Semantic accuracy**: 전체 100개 Choice 중 올바른 선택 비율
- **Downstream accuracy**: 50개 사례의 declared output이 모두 맞는 비율
- **Label accuracy**: 50개 최종 claim label 중 정답 비율
- **Joint accuracy**: semantic selection과 downstream output이 모두 맞는 비율

또한:

- 누락 응답은 오답과 구분되지만 계획 분모 기준에서는 정답으로 인정되지 않음
- 숫자, Boolean, symbolic expression, unknown 값은 타입까지 정확히 일치해야 함
- 명시적인 unknown과 값이 생략된 경우는 다르게 평가
- 모델의 선택이 같아 보여도 reference relation과 정확히 일치해야 semantic 정답으로 인정

**핵심 인사이트:**  
이 평가는 단순한 “답변이 그럴듯한가”가 아니라, 관계·수량·타입·불확실성 표현까지 정확히 일치하는지를 확인한다.

---

### 10. Appendix C: 응답이 있는 경우의 조건부 정확도

Appendix C는 미응답을 제외한 조건부 성능을 제시한다.

이 분석의 목적은 두 가지 오류를 구분하는 것이다.

1. 모델이 응답했지만 잘못 선택한 경우
2. 모델이 아예 응답하지 않은 경우

예를 들어 Terra와 DeepSeek 설정은 받은 응답의 semantic correctness는 매우 높았지만, 미응답 때문에 계획된 전체 점수가 낮아졌다.

**핵심 인사이트:**  
실제 과학 워크플로에서는 정확도뿐 아니라 안정적인 응답 제공도 중요하다. 높은 조건부 정확도만으로는 시스템의 실제 유용성을 충분히 평가할 수 없다.

---

### 종합 결론

이 논문의 표와 부록들은 다음을 일관되게 보여준다.

1. Jev는 준비된 근거와 명시적 후보 관계가 있는 과학적 의사결정에서 높은 정확도를 보였다.
2. Jev는 다른 일부 고성능 모델과 같은 정답률을 보이면서도 더 낮은 latency와 비용을 기록했다.
3. 최종 claim label만 평가하면 잘못된 과학적 수량을 놓칠 수 있다.
4. 따라서 과학적 모델 평가에서는  
   **관계 선택 → 파생 수량 → 최종 라벨**을 단계별로 확인해야 한다.
5. 다만 데이터셋이 작고, 모든 reference label이 inconsistent이며, semantic 오류가 한 사례에 집중되어 있어 일반화에는 제한이 있다.

---




### 1. Figure 1: Evaluation pipeline

Figure 1 separates the model’s semantic role from the deterministic program’s computational role.

- **Inputs**: evidence passages, structured records, questions, and candidate meanings
- **Model role**: select the scientific relation corresponding to each Choice
- **Code role**: perform counting, filtering, deduplication, arithmetic, and claim composition
- **Evaluation levels**
  1. **Semantic correctness**: Was the correct relation selected?
  2. **Downstream exactness**: Were the derived scientific quantities correct?
  3. **Label correctness**: Was the final claim label correct?
  4. **Joint correctness**: Were both the selections and derived outputs correct?

**Main insight:**  
A correct final label can hide an incorrect intermediate relation or quantity. Scientific evaluation should therefore score selections, computations, and final claims separately.

---

### 2. Table 1: Overall model comparison

Table 1 compares 12 configurations on 10 cases, with five repetitions per case. Each model had 50 planned requests and 100 semantic decisions.

#### Jev

- Complete response coverage
- Semantic correctness: **100/100**
- Downstream correctness: **50/50**
- Label correctness: **50/50**
- Joint correctness: **50/50**
- Median successful-request latency: **0.335 s**
- 95th-percentile latency: **0.442 s**
- Approximate cost: **$0.0000595 per request**

GPT-5.6 Sol, GPT-6 Astra, Claude Sonnet 5, Claude Opus 5, and Kimi K3 achieved the same observed correctness scores.

Other systems had either semantic errors or missing responses. Terra and the DeepSeek configurations were highly accurate when they responded, but missing responses reduced their planned-denominator scores.

**Main insight:**  
Jev combined complete observed correctness with the lowest successful-request latency in this evaluation. However, the result is limited to the small curated collection and the tested service conditions.

---

### 3. Table 2: Code-resolved scientific cases

Table 2 lists 10 groups whose relations can be resolved deterministically from prepared evidence and explicit rules.

Examples include:

- Shared concrete mixing operations
- Specimens within one additive-manufacturing build
- Whole-plot and strip units in electroplating
- Wear events versus repeated scans
- Reused catalyst cycles
- Distinct factor settings in chemical process design

These cases show that the number of recorded observations is not necessarily the number of independent scientific units.

**Main insight:**  
Before performing arithmetic, a workflow must determine what counts as one experimental unit and whether observations share a build, history, calibration, or measurement event.

---

### 4. Table 3: Model-routed scientific cases

Table 3 lists the 10 groups requiring model-based semantic decisions.

The cases cover:

- Mixed powder-reuse histories
- Outcome-adaptive fatigue testing
- Paired cosmological simulations
- Climate-model protocol groups
- Shared calibration uncertainty
- Complementary cloud-seeding allocations
- Developmental exposure across generations
- Shared versus separate culture histories
- RNA pooling and biological attribution
- Mosaic founders and germline transmission

**Main insight:**  
The central issue is often not “how many observations are there?” but rather “what is the relevant unit?” and “which observations share a dependency or history?”

---

### 5. Table 4: Declared downstream outputs

Table 4 defines the scientific quantities used for downstream exactness.

Examples include:

- Exact parent-lot count and completeness of assignment
- Next stress levels after failure or runout
- Paired realizations, simulation executions, and ensemble values
- Laboratory results, calibration visits, and covariance terms
- Joint assignments and range-level records
- Growth-history counts and mutation-origin status
- Contributor counts after RNA pooling
- Founder, allele, and validated germline-line counts

**Main insight:**  
Not every semantic distinction changes every downstream field. Some relations affect a claim clause but not the recorded numerical output. This is why semantic correctness and downstream exactness must remain separate metrics.

---

### 6. Table 5: Conditional performance on received responses

Table 5 evaluates only complete responses, excluding missing requests.

- Jev: semantic **100/100**, downstream/joint **50/50**, label **50/50**
- Across all received responses, all **588 final labels** were correct.
- However, only **581 downstream outputs** were exact.
- Semantic decisions were correct for **1,169 of 1,176 Choices**.
- The seven semantic errors all came from the same culture-history question.

**Main insight:**  
Conditional accuracy can hide availability problems. The paper therefore reports response coverage separately and emphasizes planned-denominator scores.

---

### 7. Culture-history case: correct label, incorrect quantity

This is the paper’s main error analysis.

- Column 11a contains 10 samples from **one culture history**
- Column 11 contains 10 samples from **separately grown cultures**
- Correct growth-history counts: **(1, 10)**

Wrong selections produced:

- (10, 1): four times
- (1, 1): twice
- (10, 10): once

The models still correctly identified colonies as clonal descendants. That preserved the final **inconsistent** label even when the growth-history counts were wrong.

**Main insight:**  
A correct verdict does not guarantee correct scientific quantities. In a real workflow, these incorrect counts could be reused in later statistical analysis, so label-only evaluation is insufficient.

---

### 8. Appendix A: Case construction and reference interpretation

Appendix A explains how the evaluation collection and reference answers were constructed.

- Full development collection: 40 Choices in 20 case groups
- Model-routed portion: 20 Choices in 10 groups
- Code-resolved portion: 20 Choices in 10 groups
- Models receive evidence and candidate meanings, but not reference answers
- Options include missing evidence, contradiction, and outside-candidate cases

The appendix explicitly preserves uncertainty, such as:

- Unknown exact parent-lot counts
- Unknown mutation-origin counts
- Unknown validated germline-line counts
- Partial rather than complete provenance
- Known culture history but unknown mutation origins

**Main insight:**  
The evaluation distinguishes “unknown” from “zero” or “not present.” Preserving uncertainty is treated as part of scientific correctness.

---

### 9. Appendix B: Metric definitions and scoring

Appendix B formally defines the metrics:

- **Semantic accuracy**: correctness over 100 individual Choices
- **Downstream accuracy**: exactness over 50 case-level outputs
- **Label accuracy**: correctness over 50 final labels
- **Joint accuracy**: simultaneous correctness of selections and outputs

The scoring also requires:

- Exact agreement on values and types
- Correct preservation of explicit unknown values
- Distinction between omitted values and unknown values
- No correctness credit for missing responses under planned denominators
- Exact match to the reference relation for semantic scoring

**Main insight:**  
The evaluation checks more than plausible language output. It checks relations, quantities, data types, and uncertainty representations.

---

### 10. Appendix C: Conditional correctness

Appendix C separates semantic mistakes from availability failures.

It distinguishes:

1. A model that responded but selected the wrong relation
2. A model that failed to provide a response

For example, Terra and DeepSeek were highly accurate among received answers, but their overall planned-denominator scores were reduced by missing responses.

**Main insight:**  
A useful scientific component must be both accurate and reliably available. Conditional accuracy alone does not capture practical workflow performance.

---

### Overall conclusion

The tables and appendices support five main conclusions:

1. Jev performed accurately on prepared scientific decision tasks with explicit candidate relations.
2. It matched several strong models in observed correctness while using less latency and cost.
3. Final claim labels can conceal incorrect scientific quantities.
4. Scientific evaluation should separately inspect  
   **semantic relation → derived quantity → final label**.
5. Generalization remains limited because the dataset is small, all reference labels are inconsistent, and all observed semantic errors occurred in one case.

<br/>
# refer format:



### BibTeX

```bibtex
@article{deng2026jev,
  author       = {Deng, Boyuan and Fan, Shuyi and Zhang, Hongyang and Xie, Xinhong},
  title        = {{Jev} for Scientific Decisions: Evaluating Semantic Choices and Their Consequences},
  journal      = {arXiv preprint arXiv:2609.24965 [cs.CL]},
  year         = {2026},
  month        = sep,
  day          = {23},
  version      = {2},
  eprint       = {2609.24965},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2609.24965}
}
```

### 시카고 스타일   

Deng, Boyuan, Shuyi Fan, Hongyang Zhang, and Xinhong Xie. “Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences.” *arXiv preprint* arXiv:2609.24965 [cs.CL], version 2, September 23, 2026. https://arxiv.org/abs/2609.24965.


