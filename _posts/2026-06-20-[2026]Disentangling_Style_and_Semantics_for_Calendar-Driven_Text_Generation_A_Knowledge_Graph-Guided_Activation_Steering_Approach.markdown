---
layout: post
title:  "[2026]Disentangling Style and Semantics for Calendar-Driven Text Generation: A Knowledge Graph-Guided Activation Steering Approach"
date:   2026-06-20 14:37:12 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 지식 그래프(KG)를 활용한 활성화 조정 방법을 통해 스타일과 의미를 분리하여 개인화된 텍스트 생성을 수행하는 프레임워크를 제안합니다.


짧은 요약(Abstract) :

이 논문의 초록에서는 대규모 언어 모델(LLM)을 개인화하여 이벤트 초대장과 같은 구조화된 커뮤니케이션을 생성하는 데 있어 발생하는 근본적인 문제를 다루고 있습니다. 생성된 텍스트는 날짜, 장소, 주최자와 같은 정확한 사실적 스키마를 유지해야 하며, 동시에 사용자의 독특한 스타일을 반영해야 합니다. 기존의 방법들은 스타일과 내용을 표현 공간에서 혼합하여 '의미 누수'라는 실패를 초래합니다. 본 연구에서는 상징적 지식 그래프(KG) 지침과 대조적 활성화 조정을 결합하여 이러한 누수를 구조적으로 줄이는 프레임워크를 제안합니다. KG는 이메일 쌍의 모든 엔티티를 유형 자리 표시자로 대체하여 사실적 변동을 제거한 후, 생성 과정에서 스타일 벡터를 주입하고, KG는 검증된 엔티티 값을 출력에 다시 삽입합니다. 이 방법은 LLaMA-2-7B 모델에서 ROUGE-L 0.2261 및 METEOR 0.2418을 달성하며, 이는 순수 조정 방식에 비해 각각 11.7% 및 23.3% 향상된 결과입니다. KG 기반의 재구성은 엔티티 환각률(EHR)을 61% 줄이며, BERTScore F1은 +0.193을 기록합니다. 비LLM 오버헤드는 추론 지연의 0.05% 미만이며, 사용자별 스타일 저장은 모델 메모리의 1% 이내에서 10,000명까지 확장 가능합니다.



The abstract of this paper addresses a fundamental challenge in personalizing large language models (LLMs) for structured communications such as event invitations. The generated text must preserve an exact factual schema (dates, venues, hosts) while reflecting the user's distinct stylistic register. Existing methods conflate style and content in representation space, leading to a failure termed "semantic leakage." The authors present a framework that structurally reduces this leakage by pairing Symbolic Knowledge Graph (KG) guidance with Contrastive Activation Steering. Before any activation is extracted, the KG defactualizes email pairs by replacing every entity with a typed placeholder; style vectors derived from these clean pairs are injected during generation, after which the KG refactualizes the output with verified entity values. On the LLaMA-2-7B model, the approach achieves ROUGE-L 0.2261 and METEOR 0.2418, representing improvements of 11.7% and 23.3% over pure steering, respectively. KG-guided refactualization reduces the Entity Hallucination Rate (EHR) by 61%, with a BERTScore F1 of +0.193. The non-LLM overhead is less than 0.05% of inference latency, and per-user style storage scales to 10,000 users within 1% of model memory.


* Useful sentences :  

원본에 없던 새로운 엔티티가 생성되면 환각으로 간주..  
이거 비율 계산이 환각률(EHR)  


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



이 논문에서는 일정 기반 텍스트 생성을 위한 지식 그래프(Knowledge Graph, KG) 기반의 활성화 조정(Activation Steering) 접근 방식을 제안합니다. 이 방법은 스타일과 의미를 분리하여 개인화된 텍스트 생성을 가능하게 합니다. 기존의 방법들은 스타일과 내용을 동일한 표현 공간에서 혼합하여 '의미 누수(semantic leakage)' 문제를 발생시킵니다. 이를 해결하기 위해, 저자들은 다음과 같은 단계로 구성된 프레임워크를 개발했습니다.

1. **스타일 비의존적 및 스타일화된 이메일 쌍 생성**: 먼저, 이벤트 메타데이터를 기반으로 중립적이고 스타일화된 이메일 쌍을 생성합니다.

2. **KG 기반의 비사실화(Defactualization)**: 생성된 이메일 쌍에서 모든 엔티티 값을 유형화된 자리 표시자로 대체하여 사실적 변동을 제거합니다. 이 과정은 KG를 통해 수행됩니다.

3. **대조적 스타일 벡터 추출**: 비사실화된 이메일 쌍을 사용하여 스타일 벡터를 추출합니다. 이 벡터는 사용자의 스타일을 반영합니다.

4. **생성 중 활성화 조정**: 생성 과정에서 추출된 스타일 벡터를 사용하여 텍스트의 스타일을 조정합니다.

5. **KG 재사실화(Refactualization)**: 생성된 텍스트에서 자리 표시자를 KG를 통해 검증된 엔티티 값으로 다시 채웁니다.

이 방법은 LLaMA-2-7B 모델을 사용하여 ROUGE-L 0.2261 및 METEOR 0.2418의 성능을 달성하였으며, 이는 기존의 순수 활성화 조정 방법에 비해 각각 11.7% 및 23.3% 향상된 결과입니다. KG 기반의 재사실화는 엔티티 환각률(Entity Hallucination Rate, EHR)을 61% 감소시켰습니다. 이 접근 방식은 비LLM 오버헤드가 0.05% 미만으로 매우 낮으며, 사용자당 스타일 저장 용량은 10,000명까지 확장 가능하다는 장점이 있습니다.




This paper proposes a Knowledge Graph (KG)-guided Activation Steering approach for calendar-driven text generation, which enables the disentanglement of style and semantics for personalized text generation. Existing methods conflate style and content in the same representation space, leading to a problem known as "semantic leakage." To address this issue, the authors developed a framework consisting of the following steps:

1. **Generation of Style-Agnostic and Styled Email Pairs**: Initially, neutral and styled email pairs are generated based on structured event metadata.

2. **KG-Guided Defactualization**: In this step, all entity values in the generated email pairs are replaced with typed placeholders to eliminate factual variance. This process is facilitated by the KG.

3. **Contrastive Style Vector Extraction**: Using the defactualized email pairs, a style vector is extracted that reflects the user's stylistic preferences.

4. **Activation Steering During Generation**: The extracted style vector is utilized to adjust the style of the text during the generation process.

5. **KG Refactualization**: After generation, the placeholders in the output text are re-filled with verified entity values from the KG.

This approach achieves a performance of ROUGE-L 0.2261 and METEOR 0.2418 on the LLaMA-2-7B model, representing improvements of 11.7% and 23.3% over pure activation steering methods, respectively. The KG-guided refactualization reduces the Entity Hallucination Rate (EHR) by 61%. Additionally, the method boasts a non-LLM overhead of less than 0.05%, and the style storage per user can scale to 10,000 users, making it highly efficient.


<br/>
# Results



이 논문에서는 KG(지식 그래프) 기반의 활성화 조정 방법을 통해 구조적 텍스트 생성에서 의미적 누수를 줄이는 방법을 제안하였습니다. 실험 결과는 다음과 같습니다:

1. **경쟁 모델**: 본 연구에서 사용된 모델은 LLaMA-2-7B로, 기존의 순수 활성화 조정 방법과 비교하였습니다. 순수 활성화 조정 방법은 ROUGE-L 점수 0.2024와 METEOR 점수 0.1961을 기록했습니다.

2. **제안된 방법**: KG 기반 활성화 조정 방법은 ROUGE-L 점수 0.2261과 METEOR 점수 0.2418을 달성하였습니다. 이는 순수 활성화 조정 방법에 비해 각각 11.7%와 23.3%의 향상을 나타냅니다.

3. **테스트 데이터**: 연구에서는 41개의 초대 이메일 쌍을 생성하였으며, 각 쌍은 중립적(형식적)과 캐주얼(친근한) 스타일로 구성되었습니다. 이 데이터는 사실적 일관성, 스타일적 차별성, 엔티티 보존을 만족하도록 설계되었습니다.

4. **메트릭**: ROUGE-L과 METEOR 외에도, 엔티티 환각률(EHR)과 BERTScore F1 점수도 평가되었습니다. 제안된 방법은 EHR을 0.90에서 0.35로 줄여 61%의 개선을 보였으며, BERTScore F1은 +0.193을 기록했습니다.

5. **비교**: 세 가지 변형을 비교한 결과, 변형 A(원시 조정)는 EHR 0.90, 변형 B(KG 재수화)는 EHR 0.65, 변형 C(KG + 주제 주입)는 EHR 0.35로 나타났습니다. 변형 C는 유일하게 긍정적인 BERTScore를 기록했습니다.

6. **레이어 선택**: 활성화 조정은 14개의 레이어를 대상으로 선형 프로브 스캔을 통해 최적의 레이어를 선택하였으며, 레이어 17이 가장 높은 ROUGE-L 점수(0.2261)를 기록했습니다.

7. **샘플 크기 안정성**: 부트스트랩 안정성 분석을 통해 PCA 방법이 최소 8개의 샘플로 안정화되었고, Mean-Difference 방법은 7개로 안정화되었습니다.

이러한 결과들은 KG 기반의 활성화 조정 방법이 텍스트 생성에서 스타일과 사실을 효과적으로 분리할 수 있음을 보여줍니다.

---



In this paper, a Knowledge Graph (KG)-guided activation steering approach is proposed to structurally reduce semantic leakage in controlled text generation. The experimental results are as follows:

1. **Competing Model**: The model used in this study is LLaMA-2-7B, which was compared against existing pure activation steering methods. The pure activation steering method achieved a ROUGE-L score of 0.2024 and a METEOR score of 0.1961.

2. **Proposed Method**: The KG-guided activation steering method achieved a ROUGE-L score of 0.2261 and a METEOR score of 0.2418. This represents an improvement of 11.7% and 23.3%, respectively, over the pure activation steering method.

3. **Test Data**: The study generated 41 pairs of invitation emails, each consisting of neutral (formal) and casual (friendly) styles. This dataset was designed to satisfy factual consistency, stylistic divergence, and entity preservation.

4. **Metrics**: In addition to ROUGE-L and METEOR, the Entity Hallucination Rate (EHR) and BERTScore F1 were also evaluated. The proposed method reduced EHR from 0.90 to 0.35, showing a 61% improvement, and achieved a BERTScore F1 of +0.193.

5. **Comparison**: Three variants were compared: Variant A (raw steering) had an EHR of 0.90, Variant B (KG re-hydration) had an EHR of 0.65, and Variant C (KG + subject injection) had an EHR of 0.35. Variant C was the only one to record a positive BERTScore.

6. **Layer Selection**: Activation steering was performed on 14 layers, and a linear probe scan was conducted to select the optimal layer, with layer 17 achieving the highest ROUGE-L score (0.2261).

7. **Sample Size Stability**: Bootstrap stability analysis showed that the PCA method stabilized at a minimum of 8 samples, while the Mean-Difference method stabilized at 7 samples.

These results demonstrate that the KG-guided activation steering approach effectively separates style and factual content in text generation.


<br/>
# 예제



이 논문에서는 일정 기반 텍스트 생성을 위한 지식 그래프(knowledge graph, KG) 기반의 활성화 조정(activation steering) 접근 방식을 제안합니다. 이 방법은 스타일과 의미를 분리하여 개인화된 텍스트 생성을 가능하게 합니다. 연구의 주요 목표는 이벤트 초대장과 같은 구조화된 커뮤니케이션에서 사용자의 스타일을 반영하면서도 정확한 사실(schema)을 유지하는 것입니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 각 이벤트에 대한 메타데이터(예: 날짜, 시간, 장소, 주최자)와 함께 두 가지 스타일의 이메일 쌍이 생성됩니다. 예를 들어, "2024년 1월 5일 오후 3시, IIT Bhilai에서 열리는 세미나에 초대합니다."라는 중립적인 이메일과 "2024년 1월 5일 오후 3시, IIT Bhilai에서 열리는 세미나에 꼭 오세요!"라는 캐주얼한 이메일이 있습니다.
   - **출력**: 두 스타일의 이메일 쌍이 생성되며, 이들은 사실적 일관성을 유지하면서도 서로 다른 스타일을 반영합니다.

2. **테스트 데이터**:
   - **입력**: 새로운 이벤트 메타데이터가 주어지면, 예를 들어 "2024년 2월 10일 오전 10시, IIT Bhilai에서 열리는 워크숍"과 같은 정보가 입력됩니다.
   - **출력**: 이 메타데이터를 바탕으로 생성된 이메일이 두 가지 스타일(중립적, 캐주얼)로 출력됩니다. 예를 들어, "2024년 2월 10일 오전 10시에 IIT Bhilai에서 열리는 워크숍에 초대합니다."와 "2024년 2월 10일 오전 10시에 IIT Bhilai에서 열리는 워크숍에 꼭 오세요!"와 같은 결과가 생성됩니다.

#### 구체적인 테스크
- **테스크**: 주어진 이벤트 메타데이터에 따라 두 가지 스타일의 이메일을 생성하는 것입니다. 이 과정에서 지식 그래프를 활용하여 사실적 일관성을 유지하고, 스타일 벡터를 통해 사용자의 개별적인 스타일을 반영합니다. 이 방법은 활성화 조정 기법을 통해 스타일과 내용을 분리하여, 생성된 텍스트가 사용자의 스타일을 반영하면서도 정확한 사실을 유지하도록 합니다.

---




This paper proposes a Knowledge Graph (KG)-guided activation steering approach for calendar-driven text generation. The main goal of this method is to enable personalized text generation that reflects the user's style while preserving an exact factual schema (dates, venues, hosts) in structured communications such as event invitations.

#### Training Data and Test Data

1. **Training Data**:
   - **Input**: For each event, metadata (e.g., date, time, venue, host) is provided, and two styles of email pairs are generated. For example, a neutral email might say, "You are invited to a seminar on January 5, 2024, at 3 PM at IIT Bhilai," while a casual email might say, "Don't miss out on the seminar at IIT Bhilai on January 5, 2024, at 3 PM!"
   - **Output**: The output consists of pairs of emails that maintain factual consistency while reflecting different styles.

2. **Test Data**:
   - **Input**: New event metadata is provided, such as "Workshop on February 10, 2024, at 10 AM at IIT Bhilai."
   - **Output**: Based on this metadata, emails are generated in two styles (neutral and casual). For example, the outputs could be "You are invited to a workshop on February 10, 2024, at 10 AM at IIT Bhilai," and "Make sure to join us for the workshop at IIT Bhilai on February 10, 2024, at 10 AM!"

#### Specific Task
- **Task**: The task is to generate emails in two styles based on the given event metadata. In this process, a knowledge graph is utilized to maintain factual consistency, and a style vector is used to reflect the individual style of the user. This method employs activation steering techniques to disentangle style from content, ensuring that the generated text reflects the user's style while preserving accurate facts.

<br/>
# 요약


이 논문에서는 지식 그래프(KG)를 활용한 활성화 조정 방법을 통해 스타일과 의미를 분리하여 개인화된 텍스트 생성을 수행하는 프레임워크를 제안합니다. 실험 결과, 이 방법은 ROUGE-L 0.2261 및 METEOR 0.2418을 달성하며, 엔티티 환각률(EHR)을 61% 감소시켰습니다. 예를 들어, 정중한 초대장과 캐주얼한 초대장 간의 스타일 차이를 유지하면서도 사실적 정확성을 보장하는 이메일 쌍을 생성했습니다.

---

This paper proposes a framework that utilizes a Knowledge Graph (KG) guided activation steering approach to disentangle style and semantics for personalized text generation. The experimental results show that this method achieves ROUGE-L 0.2261 and METEOR 0.2418, while reducing the Entity Hallucination Rate (EHR) by 61%. For instance, it generates email pairs that maintain stylistic differences between formal and casual invitations while ensuring factual accuracy.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Fig. 1**: KG-Guided Disentanglement Activation Steering 파이프라인을 보여줍니다. 이 다이어그램은 스타일 비의존적 및 스타일이 있는 이메일 쌍 생성, KG 기반의 비사실화, 대조적 스타일 벡터 추출, 생성 중 활성화 조정, 출력의 KG 재사실화 단계를 시각적으로 설명합니다. 이 구조는 각 단계가 어떻게 상호작용하여 스타일과 사실을 분리하는지를 명확히 보여줍니다.
   - **Fig. 2**: KG 구조를 나타내며, 이벤트 엔티티와 그들 간의 관계를 시각화합니다. 이 구조는 이벤트 중심의 KG가 어떻게 구성되는지를 보여주며, 각 엔티티가 어떻게 연결되는지를 설명합니다.
   - **Fig. 3**: 다양한 하이퍼파라미터 설정에 따른 성능 변화를 보여줍니다. 이 그래프는 최적의 조정 강도(α=3.0)를 찾는 데 도움을 주며, 조정 강도가 낮을 때는 스타일 변화가 미미하고, 높을 때는 의미적 드리프트가 증가함을 나타냅니다.
   - **Fig. 4**: 부트스트랩 스타일 벡터 안정성을 보여주는 그래프입니다. PCA와 Mean-Diﬀ가 각각 8과 7에서 안정화되는 것을 보여주며, LogReg는 22가 필요하다는 점을 강조합니다.

2. **테이블**
   - **Table 1**: 엔티티 추출 오류 분류 및 정확도를 보여줍니다. 이 표는 각 엔티티 유형의 정확도를 나타내며, DATE와 TIME의 높은 정확도와 VENUE의 상대적으로 낮은 정확도를 강조합니다. 이는 KG 기반의 엔티티 추출이 여전히 개선이 필요함을 시사합니다.
   - **Table 3**: 순수 활성화 조정과 KG 기반 조정의 성능 비교를 보여줍니다. KG 기반 조정이 ROUGE-L 0.2261 및 METEOR 0.2418을 달성하여 순수 조정보다 각각 11.7% 및 23.3% 향상된 것을 나타냅니다.
   - **Table 5**: 환각 비율(EHR) 감소를 보여주는 테이블입니다. KG 재수화가 EHR을 0.90에서 0.35로 줄여 61%의 상대적 감소를 달성한 것을 강조합니다.
   - **Table 8**: 스타일 메트릭을 보여주는 테이블로, KG 조정 후 Flesch Reading Ease가 54.35로 증가하여 목표치에 가까워졌음을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스는 추가적인 데이터 및 실험 결과를 포함할 수 있으며, 연구의 신뢰성을 높이는 데 기여합니다. 예를 들어, 다양한 테스트에서의 성능 결과나 추가적인 실험 설정을 포함할 수 있습니다.




1. **Diagrams and Figures**
   - **Fig. 1**: Illustrates the KG-Guided Disentanglement Activation Steering pipeline. This diagram visually explains the steps of generating style-agnostic and styled email pairs, KG-guided defactualization, contrastive style vector extraction, activation steering during generation, and KG refactualization of outputs. It clearly shows how each step interacts to separate style and facts.
   - **Fig. 2**: Represents the KG structure, visualizing event entities and their relationships. This structure shows how the event-centric KG is constructed and explains how each entity is connected.
   - **Fig. 3**: Displays performance variations across different hyperparameter settings. This graph helps identify the optimal steering strength (α=3.0), indicating that lower steering strengths yield minimal stylistic shifts, while higher strengths increase semantic drift.
   - **Fig. 4**: A graph showing bootstrap style vector stability. It demonstrates that PCA and Mean-Diﬀ stabilize at 8 and 7, respectively, while LogReg requires 22, highlighting the varying stability requirements of different methods.

2. **Tables**
   - **Table 1**: Shows the error taxonomy and per-entity accuracy for entity extraction. This table highlights the accuracy of each entity type, emphasizing the high accuracy of DATE and TIME while indicating the relatively lower accuracy of VENUE, suggesting that KG-based entity extraction still needs improvement.
   - **Table 3**: Compares the performance of pure activation steering and KG-guided steering. It shows that KG-guided steering achieves ROUGE-L 0.2261 and METEOR 0.2418, representing an improvement of 11.7% and 23.3% over pure steering, respectively.
   - **Table 5**: Displays the reduction in the hallucination rate (EHR). It emphasizes that KG re-hydration reduces EHR from 0.90 to 0.35, achieving a 61% relative reduction.
   - **Table 8**: Shows stylometric metrics, indicating that the Flesch Reading Ease increased to 54.35 after KG-guided steering, moving closer to the target.

3. **Appendix**
   - The appendix may include additional data and experimental results, contributing to the reliability of the research. For example, it could contain performance results from various tests or additional experimental settings.

<br/>
# refer format:



```bibtex
@inproceedings{Shrivastava2026,
  author    = {Tanmay Kumar Shrivastava and Aditya Bajpai and Rajesh Kumar Mundotiya},
  title     = {Disentangling Style and Semantics for Calendar-Driven Text Generation: A Knowledge Graph-Guided Activation Steering Approach},
  booktitle = {Proceedings of the PAKDD 2026},
  pages     = {590--602},
  year      = {2026},
  publisher = {Springer Nature Singapore Pte Ltd},
  address   = {Singapore},
  doi       = {10.1007/978-981-92-1468-6_42},
  institution = {MATRA Lab, Department of Computer Science and Engineering, Indian Institute of Technology (IIT) Bhilai, Bhilai, India},
  email     = {tanmayku, adityabajpai, mundotiya@iitbhilai.ac.in}
}
```




Tanmay Kumar Shrivastava, Aditya Bajpai, and Rajesh Kumar Mundotiya. 2026. "Disentangling Style and Semantics for Calendar-Driven Text Generation: A Knowledge Graph-Guided Activation Steering Approach." In *Proceedings of the PAKDD 2026*, 590-602. Singapore: Springer Nature Singapore Pte Ltd. https://doi.org/10.1007/978-981-92-1468-6_42.
