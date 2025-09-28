---
layout: post
title:  "[2025]ixi-GEN: Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining"
date:   2025-09-27 21:49:22 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 도메인 무라벨 중규모 코퍼스와 일반 리플레이(50%)를 혼합해 DACP(continual pretraining)를 수행한 뒤 지시·정렬 튜닝으로 LLaMA, Qwen, EXAONE 등 sLLM을 Telco/Finance 도메인에 적응   


짧은 요약(Abstract) :

- 배경: 오픈소스 LLM 확산으로 기업 활용 기회가 늘었지만, 대규모 모델을 운영할 인프라가 부족한 곳이 많아 소형 LLM(sLLM)이 대안으로 쓰이고 있습니다. 다만 sLLM은 성능 한계가 있습니다.
- 제안: 도메인 적응 지속 사전학습(Domain Adaptive Continual Pretraining, DACP)을 sLLM에 적용하는 실용적 학습 레시피를 제시합니다.
- 방법: 다양한 기반 모델과 서비스 도메인(예: 통신, 금융)에 DACP를 적용하고, 광범위한 실험과 실제 서비스 환경 평가로 검증했습니다.
- 결과: DACP를 적용한 sLLM은 목표 도메인 성능이 크게 향상되면서도 일반적 언어 능력을 잘 유지했습니다.
- 의의: 성능-비용 효율이 높고 확장 가능한 엔터프라이즈 배포 대안으로서, 대형 모델 없이도 산업 현장에서 고성능 서비스를 가능하게 합니다.



- Context: Open-source LLMs broaden enterprise use, but many lack infrastructure to run large models, making small LLMs (sLLMs) a practical yet performance-limited choice.
- Proposal: A practical training recipe using Domain Adaptive Continual Pretraining (DACP) for sLLMs.
- Approach: Apply DACP across diverse base models and service domains, validating with extensive experiments and real-world evaluations.
- Findings: DACP-enhanced sLLMs achieve substantial target-domain gains while preserving general capabilities.
- Implication: Provides a cost-efficient, scalable alternative for enterprise deployment, enabling high-performance services without relying on larger models.


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
아래 내용은 논문에서 제안한 메서드(DACP 레시피)의 핵심 요소를 모델·아키텍처, 학습 데이터 구성, 학습 절차(특별 기법 포함), 평가 및 효과, 실무 가이드라인과 한계로 체계적으로 정리한 것입니다.




1) 개요와 목표
- 메서드 이름: DACP(도메인 적응 지속 사전학습, Domain Adaptive Continual Pretraining) 기반 산업용 sLLM 개발 레시피
- 핵심 목표: 작은 LLM(sLLM)에 도메인 지식을 효율적으로 주입하면서 일반 능력(상식·추론·언어 전반)을 보존하여, 대형 모델 대비 낮은 비용으로 실서비스 수준의 성능을 달성
- 적용 범위: 통신(Telco), 금융(Finance) 등 서로 다른 도메인과 다양한 백본 모델(LLaMA, Qwen, EXAONE) 및 파라미터 규모(약 2.4B–32B+) 전반에 동일한 레시피로 적용

2) 모델/아키텍처
- 특별한 새 아키텍처 없이 공개 오픈 가중치 LLM(예: LLaMA 3.x, Qwen 2.5, EXAONE 3.5)을 그대로 사용
- 아키텍처 수정, 토크나이저/어휘 확장 등의 구조적 변경 없이 “사전학습 데이터 분포”를 도메인 중심으로 재적응시키는 접근
- 따라서 메서드의 핵심은 학습 데이터 구성과 학습 스케줄/절차(지속 사전학습 + 사후(post) 트레이닝)임

3) 학습 데이터 구성(중요)
- DACP 코퍼스 = 도메인 코퍼스 + 일반 리플레이(replay) 코퍼스의 혼합
- 일반 리플레이 코퍼스: FineWeb, Common Crawl, Wikipedia, GitHub Code 등 공개 대용량 영문/코드 데이터 + 한국어 성능 유지를 위한 AIHub, 국립국어원(NIKL) 대규모 한국어 말뭉치
- 도메인 코퍼스
  - Telco: ASR(음성인식) 기반 고객센터 대화 전사 + 통신/네트워크 지식. ASR 전사 오탈자·표기 실수를 줄이기 위해 오류 교정 모델을 별도 구축·적용. 일부 샘플은 “보조 정보(supporting info)” 주석으로 문맥 이해 심화를 유도
  - Finance: 금융 문서·보고서·약관 등에서 구축(도메인 원시 데이터가 상대적으로 제한적이므로 중간 규모로 구성)
- Telco DACP 데이터 예시(총 305GB): Telco 45.90% / 한국어 32.97% / 영어 16.18% / 코드 4.92%
- Finance DACP 데이터 예시(총 53GB): Finance 46.22% / 한국어 37.73% / 영어 10.37% / 코드 5.66%

4) 학습 절차(파이프라인)
- 단계 1: DACP(비지도 지속 사전학습)
  - 목적: 도메인 지식 대량 흡수 + 기존 일반 지식/추론 능력 훼손 최소화
  - 핵심 기법: 리플레이 데이터 혼합으로 ‘망각(catastrophic forgetting)’ 완화
  - 리플레이 비율 선정: EXAONE-3.5 2.4B로 약 30억 토큰 규모의 파일럿 실험을 수행해, 일반 벤치마크 성능 보존과 도메인 성능 향상의 균형점이 리플레이 50%임을 확인(50% 이상에서는 일반 성능 이득이 포화, 도메인 성능은 더 빠르게 하락)
  - 주요 하이퍼파라미터: 코사인 감쇠 스케줄러, 초기 학습률 1e-5, 최대 컨텍스트 길이 32K, 글로벌 배치 2,048
- 단계 2: 사후(post) 트레이닝(지시/정렬 복원)
  - 이유: DACP는 지식 흡수 중심이라 지시 따르기 능력이 약화될 수 있음
  - 데이터: Tulu 3, AIHub 등 공개 지시 데이터 + 인간 시드에서 합성한 데이터
  - 산출: 도메인 지식을 “사용할 줄 아는” 지시형(IT) 도메인 모델
- 단계 3: 서비스 특화 미세튜닝(선택)
  - 예: 고객센터 요약, Telco/Finance RAG QA 등
  - 기법: RAG용 SFT, 필요시 DPO 등 추가 적용(금융 MRC에서 SFT(RAG)/DPO 어블레이션 결과 보고)

5) 평가 프로토콜과 효과(요약)
- 일반 벤치(5종): MMLU, BBH, KMMLU, HAE-RAE, GSM8k-Ko. Few-shot 설정(KMMLU/HAE-RAE/GSM8K-Ko/MMLU 5-shot, BBH 3-shot), vLLM 서빙 환경으로 평가(실서비스 반영)
- Telco 내부 벤치(5태스크): 용어 선택, 고객응대 응답 생성, 단발성 QA, 상담 요약, 상담 유형 분류(로그릿/PoS Rouge-L/BLEU 기반 정확도)
- Finance 벤치(3태스크): 문단 기반 객관식 QA(FinPMCQA), 생성형 QA(FinPQA), 문서 요약(FinSM)
- 핵심 결과
  - Telco: 모든 백본/사이즈에서 도메인 벤치 평균이 크게 향상(+41%~+69%p), 일반 벤치는 소폭 변동(대체로 -7%~+1%p). 동일 크기의 원본 IT 모델 및 더 큰 범용 모델을 도메인 과제에서 능가
  - Finance: EXAONE 2.4B/7.8B에 DACP 후 Finance Avg가 각각 +49%p/+31%p 향상, 일반 Avg는 -6%p/-1%p 수준의 경미한 변동. 금융 RAG/MRC MRR 73.61%로 베이스라인 대비 대폭 개선
  - 실서비스: 네트워크 장비 QA RAG에서 실패율 유의미한 감소, 고객센터 요약 인적 평가에서도 DACP 모델이 베이스라인을 일관되게 상회

6) 왜 작동하는가(메커니즘 관점)
- 도메인 대량 코퍼스로 “지식 분포”를 재정렬하면서, 리플레이 혼합(≈50%)로 기존 일반 지식·추론 회로의 손상을 억제
- DACP를 먼저 수행하고, 이후 지시/정렬을 복원함으로써 “아는 것”과 “쓸 줄 아는 것”을 분리·최적화
- 한국어 대규모 리플레이를 충분히 포함해 다국어(특히 한국어) 성능 저하를 방지
- ASR 전사 오류 교정 등 도메인 데이터 품질 향상 장치로 미지도 학습 효율 제고

7) 실무 가이드라인(레시피로 요약)
- 데이터 혼합: 도메인:리플레이 ≈ 1:1(토큰 기준)이 권장(포가팅-전문화 균형)
- 언어 분포: 타깃 언어(한국어 등)를 리플레이에도 충분 비중으로 포함
- 전처리: 도메인 특이 노이즈(예: ASR 오탈자)는 사전 교정 모델로 정제
- 학습 순서: DACP → 지시/정렬(post-training) → 서비스 SFT/RAG
- 하이퍼파라미터: 초기 lr 1e-5, 코사인 감쇠, 긴 컨텍스트(최대 32K), 대규모 글로벌 배치(≈2k)
- 평가: 일반·도메인 벤치 동시 관찰, 실서비스 PoC에서 사람 평가/실패율 등 운영 지표 확인

8) 한계와 향후 과제
- 오픈 가중치 LLM의 원 사전학습 코퍼스가 비공개이므로, 리플레이 코퍼스가 원 분포와 완벽히 일치하지 않을 수 있음
- 이 분포 불일치가 남는 망각 리스크를 유발 가능. 향후 원 코퍼스 특성의 간접 추정과 더 정밀한 리플레이 구성 전략 필요




1) Overview and Objective
- Method: Domain Adaptive Continual Pretraining (DACP) recipe for industrial sLLMs
- Goal: Inject domain knowledge into small LLMs while preserving general capabilities, enabling service-grade performance at lower cost than larger generic models
- Scope: Validated across Telco and Finance domains, multiple backbones (LLaMA, Qwen, EXAONE), and parameter scales (~2.4B to 32B+), using a single, unified recipe

2) Model/Architecture
- No new architecture; reuse open-weight foundation LLMs as-is
- No tokenizer/vocabulary or structural changes; the core is re-adapting the pretraining data distribution
- The method hinges on data composition and a two-stage training schedule (continual pretraining followed by post-training)

3) Training Data Composition (Key)
- DACP corpus = domain corpus + general replay corpus
- General replay: FineWeb, Common Crawl, Wikipedia, GitHub Code, plus large-scale Korean corpora (AIHub, NIKL) to maintain multilingual (esp. Korean) capabilities
- Domain corpora
  - Telco: ASR-based call-center transcripts + telecom/network knowledge; applied an error-correction model for ASR artifacts; a subset annotated with supporting info for deeper context learning
  - Finance: financial documents/reports/terms; smaller but mid-scale dataset due to limited raw data
- Example sizes
  - Telco total 305GB: Telco 45.90% / Korean 32.97% / English 16.18% / Code 4.92%
  - Finance total 53GB: Finance 46.22% / Korean 37.73% / English 10.37% / Code 5.66%

4) Training Pipeline
- Stage 1: DACP (unsupervised continual pretraining)
  - Aim: absorb domain knowledge at scale while minimizing catastrophic forgetting
  - Core technique: mix replay data to retain general knowledge
  - Replay ratio: selected 50% via a 3B-token pilot on EXAONE-3.5 2.4B—beyond 50%, general gains saturate while domain performance drops faster
  - Key hyperparameters: cosine decay, initial LR 1e-5, max context 32K, global batch 2,048
- Stage 2: Post-training (instruction/alignment restoration)
  - Rationale: DACP may weaken instruction following; post-training restores it
  - Data: public instruction sets (Tulu 3, AIHub) plus synthetic from human seeds
  - Outcome: domain-adapted instruction-tuned models
- Stage 3: Service-specific fine-tuning (optional)
  - Examples: call summarization; Telco/Finance RAG QA
  - Techniques: SFT for RAG; optionally DPO (ablation reported on finance MRC)

5) Evaluation Protocol and Effects
- General benchmarks (5): MMLU, BBH, KMMLU, HAE-RAE, GSM8k-Ko; few-shot (5-shot except BBH 3-shot), evaluated via vLLM to reflect serving constraints
- Telco internal benchmarks (5 tasks): terminology selection, passage-based chat, single-turn QA, chat summarization, chat classification
- Finance benchmarks (3): passage-based MC QA (FinPMCQA), passage-based generation QA (FinPQA), summarization (FinSM)
- Key outcomes
  - Telco: large domain gains across all backbones/sizes (+41% to +69%p), with only minor fluctuations on general benchmarks (about -7% to +1%p); domain-adapted sLLMs outperform larger generic models on Telco tasks
  - Finance: DACP on EXAONE 2.4B/7.8B yields +49%p/+31%p Finance Avg improvements, with small general changes (-6%p/-1%p); finance MRC/RAG achieves 73.61% MRR, substantially surpassing baselines
  - In production: marked failure-rate reduction in network QA RAG; human ratings for call summarization consistently favor DACP models

6) Why It Works (Mechanism)
- Re-centers the pretraining data distribution on domain text while replay mixing (~50%) protects general knowledge and reasoning circuits
- Separates “acquire knowledge” (DACP) from “use knowledge” (post-training), optimizing both
- Sufficient Korean replay coverage mitigates multilingual degradation
- Domain data quality improvements (e.g., ASR error correction) raise unsupervised training efficiency

7) Practical Guidelines (Recipe Summary)
- Data mixture: domain:replay ≈ 1:1 by tokens for the best balance
- Language mix: ensure ample target-language (e.g., Korean) in the replay
- Preprocessing: correct domain-specific noise (e.g., ASR artifacts) with an error-correction model
- Stage order: DACP → instruction/alignment post-training → service SFT/RAG
- Hyperparameters: initial LR 1e-5 with cosine decay; long context (up to 32K); large global batch (~2k)
- Evaluation: track both general and domain benchmarks; verify with production KPIs (human eval, failure rates)

8) Limitations and Future Work
- Original pretraining corpora of open models are undisclosed; replay corpora may not perfectly match their distributions
- This mismatch can leave residual forgetting risk; future work: indirectly infer original data characteristics and craft more distribution-aligned replay sets


<br/>
# Results
다음은 본 논문에서 보고한 결과를 경쟁모델, 테스트 데이터/벤치마크, 평가 지표, 비교 관점에서 체계적으로 정리한 내용입니다. 마지막에 영문 요약을 함께 제공합니다.

[개요]
- 목적: sLLM(small LLM)에 DACP(도메인 적응 지속 사전학습)를 적용해 도메인 성능을 크게 높이면서도 일반 성능 저하를 최소화할 수 있는지 검증.
- 범위: 텔코(Telco)와 금융(Finance) 두 도메인, 여러 백본(LLAMA, Qwen, EXAONE), 다양한 파라미터 규모(약 2.4–3B, 7–8B, 32B 이상)에서 재현.
- 핵심 결론: DACP 적용 sLLM이 도메인 과제에서 대형 범용 모델보다도 우수할 수 있으며, 일반 능력은 소폭 변화(-7%~+1%) 범위로 유지. 실서비스(콜센터 요약, RAG QA)에서도 실패율 감소 및 사용자 평가 향상.

1) 비교 대상(경쟁모델)
- 원본 지시추종(Instruction) 모델(“Vanilla”): 
  - Llama 3.2 3B IT, Llama 3.1 8B IT
  - Qwen 2.5 3B/7B/32B/72B IT
  - EXAONE 3.5 2.4B/7.8B/32B IT
  - Llama 3.3 70B IT(금융 표에서 비교)
- DACP 적용 도메인지향 모델(“Ours”):
  - Telco DACP + Post-training: Llama 3.2/3B, Llama 3.1/8B, Qwen 2.5/3B·7B, EXAONE 3.5/2.4B·7.8B·32B
  - Finance DACP + Post-training: EXAONE 3.5/2.4B·7.8B
- 서비스 적용 비교:
  - 콜센터 요약: EXAONE 3.5 2.4B vs EXAONE 3.5 Telco 2.4B
  - RAG QA(네트워크/금융): EXAONE 3.5 7.8B IT vs RAG 미세조정 vs DACP+RAG 미세조정

2) 테스트 데이터·벤치마크
- 일반 도메인(공개 벤치마크, LM Eval Harness로 평가, vLLM 서빙):
  - MMLU, BBH, KMMLU(국문), HAE-RAE(국문 지식), GSM8k-Ko(국문 수리)
  - 프롬프트: KMMLU/HAE-RAE/GSM8k-Ko/MMLU는 5-shot, BBH는 3-shot
- 텔코 도메인(자체 구축):
  - 과제 구성(총 750 문항): 
    - Vocabulary(200, 용어 의미 선택: Logit Likelihood),
    - Single QA(100, 단발 질문: BLEU 기반 Acc),
    - Passage Chat(150, 대화 응답 생성: 한국어 품사(RoUGE-L) 기반 Acc),
    - Chat SM(100, 상담 요약: 한국어 PoS Rouge-L),
    - Chat CLS(200, 상담 유형 분류: BLEU 기반 Acc).
  - 서비스 RAG QA(네트워크 장비 문서 기반)도 별도 평가(그림 5, 표 20).
- 금융 도메인(자체 구축):
  - FinPMCQA(다지선다 QA: Logit Likelihood), FinPQA(생성형 QA: 한국어 PoS Rouge-L), FinSM(문서 요약: 한국어 PoS Rouge-L)
  - 추가 MRC형태 평가: MRR(Mean Reciprocal Rank)로 정량화(표 5)
  - 외부 RAG 리더보드(Top-3 패시지 고정) 기반 hit@3 및 금융 점수 비교(표 10)

3) 평가 지표 및 절차
- 일반 벤치마크: 정확도(또는 LM Eval Harness 기본 스코어).
- 텔코 벤치마크: 
  - 선택형은 Logit Likelihood/기준 유사도(BLEU)로 후보 선택 정확도,
  - 생성형은 한국어 품사 기반 Rouge-L(정확도 또는 점수),
  - 요약/분류는 PoS Rouge-L 또는 BLEU 기반 Acc.
- 금융 벤치마크: 
  - FinPMCQA(Logit Likelihood Acc), FinPQA/FinSM(한국어 PoS Rouge-L),
  - MRC/도큐먼트 활용 능력: MRR,
  - RAG 리더보드: hit@3 및 금융 점수.
- 인공지능 서비스 평가: 
  - 콜센터 요약(사람 평가, 1–5점 척도; 요청 요약/응답 요약 각각 독립 기준; 표 13),
  - 네트워크 RAG QA 실패율(규칙 기반 판정) 비교(그림 5),
  - 금융 도큐먼트 질의: MRR, hit@3(표 5, 표 10, 표 21).

4) 정량 결과 요약 및 비교
- 일반 능력 유지(표 2):
  - Telco DACP 후 일반 벤치마크 평균은 -7% ~ +1% 내 변동.
  - 예: Qwen 2.5 7B IT 62.97 → 63.91(+1%), Llama 3.2 3B IT 45.43 → 42.16(-7%), EXAONE 3.5 32B IT 68.15 → 67.07(-2%).
  - 결론: 일반 능력은 대체로 소폭 유지되며, 도메인 성능 향상 대비 희생 최소.
- 텔코 도메인 대폭 향상(표 3):
  - 모든 백본/스케일에서 큰 상대 개선(+41%~+69%).
  - 소형(≈3B): 
    - Llama 3.2 3B IT 47.97 → 72.38(+51%),
    - Qwen 2.5 3B IT 47.08 → 70.93(+50%),
    - EXAONE 2.4B IT 42.30 → 70.94(+67%).
  - 중형(≈7–8B):
    - Llama 3.1 8B IT 50.90 → 77.16(+52%),
    - Qwen 2.5 7B IT 49.63 → 70.26(+41%),
    - EXAONE 7.8B IT 42.92 → 71.12(+66%).
  - 대형(32B): EXAONE 32B IT 49.82 → 84.43(+69%).
  - 시사점: 동일 DACP 레시피로 다양한 모델·크기에서 일관된 텔코 성능 향상. sLLM도 도메인 내에서 대형 범용 모델을 능가 가능.
- 금융 도메인(표 12, 표 5, 표 10):
  - 금융 벤치마크 평균(3과제 평균):
    - EXAONE Finance 2.4B IT: 39.52 → 58.88(+49%), 일반 평균은 52.64 → 49.64(-6%).
    - EXAONE Finance 7.8B IT: 45.53 → 59.54(+31%), 일반 평균 60.10 → 59.56(-1%).
  - 대형 범용 모델 초월:
    - 금융 평균에서 DACP 7.8B(59.54)가 Qwen 2.5 32B(50.48), EXAONE 32B(50.70), Llama 3.3 70B(52.23), Qwen 2.5 72B(52.62) 등 대형 범용 모델을 상회.
  - 금융 MRC(MRR, 표 5):
    - EXAONE 7.8B(바닐라) 47.64 → DACP+Post 73.61로 최대 95% 개선(저자 서술 기준). 단순 Post-training만 한 경우(37.64) 대비도 크게 우수.
  - 금융 RAG 리더보드(표 10, top-3 패시지 동일):
    - hit@3는 동일(0.91)이지만, 금융 점수 0.633(바닐라) → 0.666(RAG FT) → 0.683(DACP+RAG FT)로 DACP가 추가 이득.
- 실서비스 평가(콜센터·네트워크 QA):
  - 콜센터 요약(사람 평가, 표 6):
    - EXAONE 2.4B 평균 4.13 → Telco DACP 2.4B 평균 4.50로 유의미한 향상(요청/응답 요약 모두 향상).
  - 네트워크 장비 RAG QA(그림 5, 표 20):
    - 동일 RAG 세팅에서 DACP 모델이 실패율을 크게 감소. 도메인 문서 이해·정답 회수/생성에서 지속적 우위.
  - 시사점: DACP는 지도 미세조정(SFT)만으로 달성하기 어려운 도메인 이해 기반의 안정적 응답 개선을 제공.

5) 아블레이션: 리플레이 비율(그림 3, 표 7–8, 그림 6)
- 문제의식: 도메인 지식 습득 중 일반 지식 망각(파국적 망각) 방지 필요.
- 방법: 공개 코퍼스(FineWeb, CC, Wikipedia, GitHub Code 등)와 한국어 코퍼스(AI Hub, NIKL)로 의사 리플레이 코퍼스 구성.
- 실험: EXAONE 3.5 2.4B, 약 3B 토큰(≈10GB)으로 리플레이 비율 5/10/25/50/75% 비교.
- 결과: 
  - 일반 벤치마크는 리플레이 비율↑에 따라 개선되나(50% 이상에서는 개선 포화), 
  - 텔코 도메인은 25% 이후부터 성능 저하 가속.
  - 균형점: 50% 리플레이 비율이 일반 유지와 도메인 향상 간 최적 균형(그림 3). 본 학습에도 50% 채택(학습셋 구성 표 1 참조).
- 결론: 중간 규모 코퍼스로도 적절한 리플레이 설계가 망각을 효과적으로 억제.

6) 데이터 구성(학습셋, 참고)
- 텔코 DACP 학습셋(총 305GB, 표 1): Telco 45.9%, 한국어 33.0%, 영어 16.2%, 코드 4.9%.
- 금융 DACP 학습셋(총 53GB, 표 9): Finance 46.2%, 한국어 37.7%, 영어 10.4%, 코드 5.7%.
- Post-training: 공개 지시 데이터(Tulu 3, AI Hub 등) 및 합성 데이터 활용.

7) 종합 비교·함의
- 동일 레시피/유사 하이퍼파라미터로 다양한 백본·파라미터 규모에서 도메인 성능을 크게 향상.
- 일반 능력은 -7%~+1%의 제한적 변동. 50% 리플레이로 망각을 실용 수준으로 완화.
- 도메인 내 성능: 소형·중형 sLLM이 대형 범용 모델(32B–72B)을 능가(특히 금융, 텔코 실제 과제).
- 산업 적용성: 콜센터 요약, 네트워크 RAG QA에서 실패율 감소·사용자 경험 개선 → 비용 효율적 대안.




Overview
- Goal: Validate whether DACP (Domain-Adaptive Continual Pretraining) can significantly improve small LLMs (sLLMs) on target domains while largely preserving general capabilities.
- Scope: Two domains (Telco, Finance), multiple backbones (LLaMA, Qwen, EXAONE), and sizes (~2.4–3B, 7–8B, 32B+).
- Key finding: DACP-equipped sLLMs achieve large gains on domain tasks and can outperform larger general models, with only minor shifts in general ability. Also effective in real-world applications (call-center summarization, RAG QA).

1) Competitors
- Baseline instruction (vanilla) models:
  - Llama 3.2 3B IT, Llama 3.1 8B IT
  - Qwen 2.5 3B/7B/32B/72B IT
  - EXAONE 3.5 2.4B/7.8B/32B IT
  - Llama 3.3 70B IT (Finance table)
- DACP models (ours):
  - Telco DACP + post-training for Llama/Qwen/EXAONE (3B, 7–8B, 32B)
  - Finance DACP + post-training for EXAONE (2.4B, 7.8B)
- Real-world comparisons:
  - Call-center summarization: EXAONE 2.4B vs EXAONE Telco 2.4B
  - RAG QA: EXAONE 7.8B vanilla vs RAG FT vs DACP + RAG FT

2) Test sets and benchmarks
- General (public, via LM Eval Harness, served by vLLM):
  - MMLU, BBH, KMMLU (ko), HAE-RAE (ko knowledge), GSM8k-Ko (ko math)
  - Few-shot settings: 5-shot (KMMLU/HAE-RAE/GSM8k-Ko/MMLU), 3-shot (BBH)
- Telco (in-house):
  - 5 subtasks (750 items total): Vocabulary, Single QA, Passage Chat, Chat Summarization, Chat Classification, with PoS Rouge-L/BLEU/Logit-Likelihood-based accuracy.
  - Additional RAG QA evaluation on network manuals (Figure 5, Table 20).
- Finance (in-house):
  - FinPMCQA (MCQ; Logit Likelihood), FinPQA (gen QA; Ko PoS Rouge-L), FinSM (summ.; Ko PoS Rouge-L).
  - MRC/Document use: MRR.
  - External RAG leaderboard with fixed top-3 passages: hit@3 and finance score (Table 10).

3) Metrics and protocol
- General: accuracy (LM Eval Harness defaults).
- Telco: logit-likelihood or BLEU-based accuracy for selection tasks; Ko PoS Rouge-L (or Acc) for generation/summ.; BLEU Acc for classification.
- Finance: MCQ Acc (logit likelihood), Ko PoS Rouge-L for gen/summ.; MRR for MRC; hit@3 and finance score for RAG.
- Human/Service: 
  - Call-center summarization (1–5 scale; separate criteria for customer request vs agent response; Table 13).
  - Network RAG QA failure rate via rule-based judging (Figure 5).

4) Main quantitative results and comparisons
- General ability retention (Table 2):
  - After Telco DACP, general-domain averages shift within -7% to +1%.
  - Examples: Qwen 2.5 7B IT 62.97 → 63.91 (+1%); Llama 3.2 3B IT 45.43 → 42.16 (-7%); EXAONE 32B IT 68.15 → 67.07 (-2%).
- Telco domain gains (Table 3):
  - Large relative improvements across all backbones/sizes (+41% to +69%).
  - 3B class: Llama 47.97 → 72.38 (+51%); Qwen 47.08 → 70.93 (+50%); EXAONE 42.30 → 70.94 (+67%).
  - 7–8B: Llama 50.90 → 77.16 (+52%); Qwen 49.63 → 70.26 (+41%); EXAONE 42.92 → 71.12 (+66%).
  - 32B: EXAONE 49.82 → 84.43 (+69%).
  - Implication: A single DACP recipe transfers well, and sLLMs can surpass larger general models on-domain.
- Finance domain (Tables 12, 5, 10):
  - Finance avg (over three tasks):
    - EXAONE 2.4B: 39.52 → 58.88 (+49%) with general -6%.
    - EXAONE 7.8B: 45.53 → 59.54 (+31%) with general -1%.
  - Beating larger general models:
    - Finance 7.8B DACP (59.54) > Qwen 32B (50.48), EXAONE 32B (50.70), Llama 70B (52.23), Qwen 72B (52.62).
  - MRC (MRR; Table 5): DACP+post reaches 73.61 vs vanilla 47.64 and post-only 37.64 (up to +95% per authors).
  - RAG leaderboard (Table 10, same top-3 passages):
    - hit@3 unchanged (0.91), but finance score 0.633 → 0.666 → 0.683 with DACP+RAG FT.
- Real-world applications:
  - Call-center summarization (Table 6): human rating avg 4.13 → 4.50 with Telco DACP 2.4B.
  - Network RAG QA (Figure 5): failure rate substantially reduced by Telco DACP.

5) Ablation: replay ratio (Figure 3; Tables 7–8; Figure 6)
- Need: mitigate catastrophic forgetting during DACP.
- Method: pseudo-replay built from public corpora (FineWeb, CC, Wikipedia, GitHub Code) + large Korean corpora (AI Hub, NIKL).
- Finding: 
  - General performance improves with higher replay but saturates >50%.
  - Telco performance declines beyond 25% replay, with 50% being the best trade-off overall.
- Adopted: 50% replay in full-scale DACP.

6) Data composition (training, for context)
- Telco DACP training set (305GB): Telco 45.9%, Korean 33.0%, English 16.2%, Code 4.9% (Table 1).
- Finance DACP training set (53GB): Finance 46.2%, Korean 37.7%, English 10.4%, Code 5.7% (Table 9).
- Post-training on public instruction data (Tulu 3, AI Hub) + synthetic.

7) Overall implications
- Robust, reproducible recipe: significant domain gains across backbones/sizes, with small general-domain drift.
- On-domain, DACP sLLMs can outperform 32B–72B general models (esp. Finance and Telco tasks).
- In practice, DACP improves real services (call summarization, RAG QA) with lower cost and smaller compute footprint.


<br/>
# 예제
다음은 논문(“ixi-GEN: Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining”)에서 제시된 데이터와 태스크를 바탕으로, 실제로 어떤 입력과 출력으로 학습·평가했는지, 과업은 무엇이었는지를 체계적으로 정리한 예시입니다. 원문에 포함된 대표 샘플을 그대로 인용·요약하여 구성했습니다.

1) 학습(트레이닝) 데이터와 입·출력 형태
- DACP(도메인 적응 지속 사전학습)
  - 목적: 소규모 LLM(sLLM)에 도메인 지식을 대규모로 주입하면서, 리플레이(Replay) 데이터로 일반 능력의 망각을 방지.
  - 입력: 토큰화된 문서 시퀀스(도메인 코퍼스 + 리플레이 코퍼스). 학습은 일반적인 언어모델 사전학습(다음 토큰 예측) 방식.
  - 출력: 도메인 지식이 반영된 업데이트된 모델 파라미터(도메인-적응 사전학습 완료 모델).
  - 데이터 구성(텔코, Table 1)
    - Telco 코퍼스 140GB (45.90%): 콜센터 대화 음성인식(ASR) 전사, 통신·네트워크 지식 문서. ASR 오탈자 보정을 위한 오류 교정 모델을 사용. 일부 샘플에 보조 정보(근거) 주석 추가.
    - 한국어 코퍼스 100GB (32.97%)
    - 영어 코퍼스 50GB (16.18%)
    - 코드 15GB (4.92%)
    - 총 305GB
  - 리플레이 코퍼스 출처: FineWeb, Common Crawl(CC-Net), Wikipedia, GitHub Code 등 공개 말뭉치. 한국어 성능 유지를 위해 AI Hub, 국립국어원(NIKL) 자료 비중 확대.
  - 리플레이 비율: 사전 실험 결과 50%가 일반 능력 유지와 도메인 적응의 균형에 최적(그림 3, 부록 A).

- 파이낸스 DACP 데이터(부록 E, Table 9)
  - Finance 코퍼스 24.5GB (46.22%), 한국어 20GB (37.73%), 영어 5.5GB (10.37%), 코드 3GB (5.66%), 총 53GB.
  - 재무제표, 금융 보고서, 약관·계약 관련 문서 등으로 구성.

- 포스트 트레이닝(Instruction/Alignment Tuning)
  - 필요성: DACP는 지식 주입 중심이므로 지시 따르기 능력이 약화될 수 있어, 이후 지시 미세조정이 필요.
  - 사용 데이터: Tulu 3, AI Hub 공개 지시 데이터, 인적 시드에서 합성한 지시-응답 쌍 등.
  - 입·출력: (입력) 지시/문맥/질문, (출력) 모델이 생성한 지시-준수 응답.

- 서비스 파인튜닝(SFT) 및 RAG 학습 데이터
  - 콜센터 요약 자동화:
    - 입력: 통화 전사(한 회차 대화 전체).
    - 출력: (1) 고객 요청 요약, (2) 상담사 조치 요약. 이후 사람 평가(1–5점).
  - 네트워크 장비 QA(RAG):
    - 입력: 질의 + 검색된 상위 문서/패시지(top-k).
    - 출력: 근거에 기반한 정답 생성. 실패율을 규칙 기반으로 산정(그림 5, Table 20).
  - 파이낸스 RAG:
    - 입력: 금융 문서(계약/약관/보고서 등)에서 검색된 상위 문서/패시지 + 질의.
    - 출력: 정답(추출·생성). MRR 등으로 평가(표 5, 표 10, Table 21).

2) 테스트(평가) 태스크와 구체 입력·출력
- 일반 도메인 벤치마크
  - 과업: 지식·추론 전반 성능 평가.
  - 데이터셋: MMLU, BBH, KMMLU, HAE-RAE Bench, GSM8k-Ko.
  - 입력: 3–5샷 예시가 포함된 문제 프롬프트.
  - 출력: 정답 선택 또는 수치/문자열 답.
  - 평가: 정확도(Accuracy). 실서비스 반영 위해 vLLM 서빙에서 LM Eval Harness로 측정(부록 C).

- 텔코 도메인 벤치마크(내부 구축, Table 11)
  1) Vocabulary(용어 선택)
     - 입력: “통신기기설치불량”의 의미를 가장 잘 설명하는 보기(객관식).
       예시 보기 발췌(원문 Table 16):
       - ① 고객 1:1 마케팅 기법 설명
       - ② 타임슬롯으로 여러 가입자가 전송로 공유하는 기술
       - ③ 한전 공가 용역 순시 개선요구사항 중 ‘통신기기 고정 불량’ 상태
       - ④ 간편결제 서비스명 설명
     - 출력: 정답 보기(번호 또는 보기 텍스트).
     - 평가: Logit Likelihood 기반 선택 정확도.

  2) Passage Chat(대화 응답 생성·선택)
     - 입력: 요금제 표, 고객의 최근 사용량 요약, 짧은 상담 대화 맥락, 그리고 후보 응답 5개.
       예시(요지, Table 16):
       - 요금제 표: 5G 시그니처/프리미어 플러스/에센셜/스탠다드/데이터 플러스 등, 데이터/공유 데이터/가격.
       - 고객 사용량: 80GB 제공 요금제에서 월 평균 150GB 사용(초과), 공유 데이터 월 5GB 사용.
       - 대화: 고객이 “데이터를 많이 써서 요금제 변경 고민, 더 저렴한 대안?”을 질문.
       - 후보 응답: 5G 스탠다드 추천(단, 초과요금 우려), 5G 라이트+ 등 맥락 불일치 후보 포함 등.
     - 출력: 가장 적절한 후보 응답의 선택.
     - 평가: 한국어 품사 기반 Rouge-L 유사도로 정답 후보를 판정하여 Accuracy 산출.

  3) Single QA(단발 질문·응답 선택)
     - 입력: 고객 질문(예: “프리미엄 혜택에 뭐가 있나요?”)과 보기(객관식, Table 16).
     - 출력: 정답 보기.
     - 평가: BLEU 기반 후보 선택 정확도.

  4) Chat SM(대화 요약)
     - 입력: 상담사–고객 대화 전체(예: 자동이체 카드 변경, VIP 혜택 안내가 포함된 대화; Table 16).
     - 출력: 대화 요약 텍스트(요약 길이·형식은 벤치마크 설정에 따름).
     - 평가: 한국어 품사 기반 Rouge-L.

  5) Chat CLS(대화 유형 분류)
     - 입력: 상담 대화(예: 자동이체 변경 문의; Table 17).
     - 출력: 분류 라벨(보기 중 하나: 예, “잘못 걸린 전화(내부)”, “청구 용어/당월 요금 문의”, “자동이체 방법 변경/문의”, “홈요금제 위약금/약정 문의” 등).
     - 평가: BLEU Acc(후보와의 유사도 기반 분류 정확도).

- 파이낸스 도메인 벤치마크(부록 F, Table 12)
  1) FinPMCQA(지문 기반 객관식 QA)
     - 입력: 긴 금융 약관/계약 조항 등 본문 + 질문 + 보기(객관식).
       예시(요지, Table 18):
       - 본문: “휴대폰 메시지 서비스” 조항(제공 범위, 장애 시 통지, 수수료 청구 방식 등).
       - 질문: “이용 수수료는 어떻게 청구되는가?”
       - 보기: (① 사용자가 선택 납부, ② 모든 이용자 월 300원, ③ 개인회원당 300원, ④ 번호 변경 시에만 부과)
     - 출력: 정답 보기(번호/텍스트).
     - 평가: Logit Likelihood 기반 정확도.

  2) FinPQA(지문 기반 생성형 QA)
     - 입력: 금융 약관/보험 조항 본문 + 질문.
       예시(요지, Table 19):
       - 본문: “청약의 철회” 제20조(철회 가능 조건, 전문/일반 금융소비자 구분, 반환 기한·이자 등).
       - 질문: “청약 철회 시 납입한 보험료 반환 여부?”
     - 출력: 근거에 기반한 자연어 답변(자유 생성).
     - 평가: 한국어 품사 기반 Rouge-L.

  3) FinSM(문서 요약)
     - 입력: 금융 서비스 약관 개정·명시 조항 본문(예: “혁신금융서비스” 지정, 변경 통지 방식 등; Table 19).
     - 출력: 3문장 이하 요약 텍스트.
     - 평가: 한국어 품사 기반 Rouge-L.

- 산업 응용 평가 태스크
  - 콜센터 요약 자동화(사람 평가, Table 13)
    - 입력: 통화 전사.
    - 출력: 고객 요청 요약, 상담사 조치 요약(각 1개).
    - 평가: 1–5점(히스토리 트래킹 관점). 기준:
      - (요청 요약) 고객 의도를 얼마나 간결·정확히 담았는가.
      - (조치 요약) 처리 결과를 얼마나 명확히 담았는가.

  - 네트워크 장비 QA 시스템(RAG, Table 20)
    - 입력: 참조 문서(예: “원격지원 상품 소개자료” 항목별 Q/A 성격의 기술) + 질의(예: “가입 프로세스는? 방문 지원 가능 지역은?”).
    - 출력: 문서에서 근거를 바탕으로 한 정확한 답변(예: 접수 방식, 영업일 기준 처리 기간, 수도권/비수도권 방문 지원 정책 등).
    - 평가: 규칙 기반 실패율(그림 5), hit@3 등(부록 H, 표 10).

  - 파이낸스 RAG MRC 평가(표 5, Table 21)
    - 입력: 상위 검색 패시지 + 질의.
    - 출력: 추출/생성 답변.
    - 평가: MRR 기반 4개 범주 점수(데이터 추출·문서 이해, 최신성·서식 인지/권고, 문서 생성, 내용 기억).

참고: 본 논문은 각 태스크에 대해 정답 자체를 논문 내에 표기하지 않은 경우가 많습니다. 위 예시는 입력 형태와 출력 기대 형식을 구체화하기 위한 것으로, 최종 정답 표시는 생략했습니다.







Below is a structured, example-based description of training and test data, concrete tasks, and input/output formats as used in the paper “ixi-GEN: Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining.” All examples are summarized or directly drawn from the paper.

1) Training data and I/O
- DACP (Domain-Adaptive Continual Pretraining)
  - Goal: Inject domain knowledge at scale into sLLMs while mitigating catastrophic forgetting via replay data.
  - Input: Tokenized document sequences from domain corpora mixed with replay corpora; trained with standard next-token prediction.
  - Output: Updated model parameters (domain-adapted pretrained model).
  - Telco data mix (Table 1)
    - Telco corpus 140GB (45.90%): ASR call transcripts and telecom/network knowledge docs. An error-correction model fixes ASR noise. Some samples include supporting annotations.
    - Korean 100GB (32.97%), English 50GB (16.18%), Code 15GB (4.92%), Total 305GB.
  - Replay sources: FineWeb, Common Crawl(CC-Net), Wikipedia, GitHub Code; enriched with Korean (AI Hub, NIKL).
  - Replay ratio: 50% found optimal for balancing general retention and domain adaptation (Fig. 3, Appx. A).

- Finance DACP data (Appx. E, Table 9)
  - Finance 24.5GB (46.22%), Korean 20GB (37.73%), English 5.5GB (10.37%), Code 3GB (5.66%); total 53GB.
  - Includes financial statements, reports, and contractual/terms documents.

- Post-training (Instruction/Alignment)
  - Need: DACP improves knowledge but may weaken instruction-following; instruction tuning restores it.
  - Data: Tulu 3, AI Hub instruction sets, synthetic instruction–response pairs.
  - I/O: (Input) instruction/context/question, (Output) model-generated compliant response.

- Service SFT and RAG training
  - Call-center summarization:
    - Input: Full call transcript.
    - Output: (1) Customer request summary, (2) Agent action summary. Human evaluation (1–5 scale).
  - Network equipment QA (RAG):
    - Input: Query + retrieved top-k passages/documents.
    - Output: Evidence-based answer. Failure rate measured by rules (Fig. 5, Table 20).
  - Finance RAG:
    - Input: Retrieved top-k passages from financial documents + query.
    - Output: Extractive/generative answer. Evaluated via MRR (Table 5, Table 10, Table 21).

2) Test tasks, with concrete inputs/outputs
- General-domain benchmarks
  - Purpose: Broad knowledge/reasoning.
  - Datasets: MMLU, BBH, KMMLU, HAE-RAE Bench, GSM8k-Ko.
  - Input: 3–5 shot prompts.
  - Output: Single answer (choice or short text/number).
  - Metric: Accuracy. Measured with LM Eval Harness on vLLM serving (Appx. C).

- Telco domain benchmark (in-house, Table 11)
  1) Vocabulary (term selection)
     - Input: A definition-like prompt with multiple choices.
       Example choices (from Table 16):
       - ① One-on-one marketing tailored to customer preference
       - ② Time-slot multiplexing tech sharing a high-speed line
       - ③ A KEPCO inspection category indicating “improperly fixed communications equipment”
       - ④ A brand name of a “simple-pay” service
     - Output: The correct choice (ID or text).
     - Metric: Logit Likelihood Accuracy.

  2) Passage Chat (dialogue response selection)
     - Input: Plan table, recent usage, brief dialogue, and five candidate replies.
       Example (Table 16):
       - Plans: Several 5G plans with price/data/sharing data.
       - Usage: Avg. 150GB/month on an 80GB plan (overuse), sharing data 5GB of 10GB.
       - Dialogue: Customer asks for a cheaper alternative with unlimited data.
       - Candidates: Include contextually proper/ improper recommendations.
     - Output: Best candidate reply.
     - Metric: Korean PoS Rouge-L–based candidate selection accuracy.

  3) Single QA (single-turn QA with choices)
     - Input: A question (e.g., “What are premium benefits?”) with multiple choices (Table 16).
     - Output: Correct choice.
     - Metric: BLEU-based selection accuracy.

  4) Chat SM (chat summarization)
     - Input: Full agent–customer transcript (e.g., auto-payment card change, VVIP benefits; Table 16).
     - Output: Summary text.
     - Metric: Korean PoS Rouge-L.

  5) Chat CLS (chat classification)
     - Input: A customer-service dialogue (e.g., auto-payment change inquiry; Table 17).
     - Output: One label among provided options (e.g., Wrong Number (Internal), Billing Terms/Current Charges, Autopayment Change/Inquiry, Penalty/Contract of Home Plan).
     - Metric: BLEU Acc (selection accuracy via candidate similarity).

- Finance domain benchmarks (Appx. F, Table 12)
  1) FinPMCQA (passage-based MCQ)
     - Input: A long legal/contractual passage + question + choices.
       Example (Table 18):
       - Passage: “Mobile message service” terms including fees and notice requirements.
       - Question: “How is the fee charged?”
       - Choices: e.g., per user, per member, etc.
     - Output: Correct choice.
     - Metric: Logit Likelihood Accuracy.

  2) FinPQA (passage-based generative QA)
     - Input: Insurance “Withdrawal of Application” article + question.
       Example (Table 19):
       - Question: “Is the paid premium refunded upon withdrawal?”
     - Output: Free-form answer grounded in the passage.
     - Metric: Korean PoS Rouge-L.

  3) FinSM (document summarization)
     - Input: Terms on stipulation/revision/notice for an “innovative financial service” (Table 19).
     - Output: ≤3 sentence summary.
     - Metric: Korean PoS Rouge-L.

- Industrial application tasks
  - Call-center summarization (human eval, Table 13)
    - Input: Call transcript.
    - Output: (1) Customer request summary, (2) Agent action summary.
    - Scoring: 1–5 scale for each; criteria emphasize clarity and future traceability.

  - Network QA (RAG, Table 20)
    - Input: Reference doc (e.g., “Remote Support Product Introduction”: subscription steps, on-site support policy) + query (“What is the subscription process? Which areas allow on-site support?”).
    - Output: Evidence-grounded answer (e.g., submission channel, 3 business days, metro/non-metro policy).
    - Evaluation: Rule-based failure rate (Fig. 5) and hit@3 (Appx. H, Table 10).

  - Finance RAG MRC (Table 5, Table 21)
    - Input: Top-k retrieved passages + query.
    - Output: Extractive/generative answer.
    - Metrics: MRR across four categories (data extraction & doc understanding; freshness/format recognition & recommendation; document generation; content memorization).

Note: The paper typically does not print gold answers in the text. The examples above focus on input structure and expected output format; final gold labels are omitted accordingly.

<br/>
# 요약


- 메서드: 도메인 무라벨 중규모 코퍼스와 일반 리플레이(50%)를 혼합해 DACP(continual pretraining)를 수행한 뒤 지시·정렬 튜닝으로 LLaMA, Qwen, EXAONE 등 sLLM을 Telco/Finance 도메인에 적응시킴. 
- 결과: Telco 벤치마크에서 전 백본이 +41~69% 향상(예: EXAONE 32B 49.82→84.43)하며 일반 벤치마크는 대체로 ±0~7% 수준으로 유지, 금융에서도 Finance Avg 59.54%(+31%, 일반 -1%)로 개선되어 도메인 내에서 더 큰 일반 모델을 능가. 
- 예시: 콜센터 요약 인력평가 평균 4.13→4.50로 상승, 네트워크 장비 RAG QA 실패율 크게 감소, 금융 RAG MRC에서 MRR 73.61%로 베이스라인 대비 큰 폭 개선.

- Method: We mix mid-scale unlabeled domain corpora with a 50% general replay set for DACP, then restore instruction following to adapt sLLMs (LLaMA, Qwen, EXAONE) to Telco/Finance. 
- Results: Telco benchmarks improved by +41–69% (e.g., EXAONE 32B 49.82→84.43) while general benchmarks stayed within about ±0–7%; in Finance, the adapted model reached 59.54% Finance Avg (+31%, −1% general), surpassing larger general models in-domain. 
- Examples: Call-center summarization human scores rose from 4.13 to 4.50, network-equipment RAG QA failure rates dropped substantially, and Finance RAG MRC achieved 73.61% MRR, well above baselines.

<br/>
# 기타


다이어그램/피규어 요약 (결과·인사이트)
- Figure 1: Telco 도메인에 DACP를 적용한 sLLM이 상용(서비스) 태스크에서 베이스 모델을 능가하면서 일반 역량은 유지. 인사이트: 소형 모델로도 도메인지표-서비스지표의 동시 개선이 가능해 비용-성능 균형이 좋음.
- Figure 2: 전체 파이프라인(DACP → 포스트트레이닝 → API/서비스 → 피드백 루프). 인사이트: 도메인/모델 상관없이 재사용 가능한 “레시피”로 설계되어 확장·운영이 용이.
- Figure 3: 리플레이 비율 변화에 따른 일반/텔코 성능 트레이드오프. 50% 리플레이가 일반 역량 보존과 도메인 향상을 동시에 달성하는 균형점. 인사이트: 중규모 코퍼스 DACP에서 50% 리플레이가 실무적 기본값으로 적합.
- Figure 4: NW QA 시스템 예시. 도메인 적응 과도 시 일반 대화 품질 저하 가능성 경고. 인사이트: 서비스 현장에서는 도메인 지식과 일반 대화성의 균형이 사용자 경험에 핵심.
- Figure 5: 네트워크 장비 QA RAG에서 DACP 모델이 실패율을 유의미하게 감소. 더 작은 DACP 모델이 더 큰 바닐라 모델보다 우수. 인사이트: 동일 RAG 조건에서 생성기 자체의 도메인 적합도가 성패를 좌우.
- Figure 6: 리플레이 비율×배치 사이즈 영향. 일반 벤치마크는 리플레이 증가에 따라 상승, 텔코는 25% 초과 시 하락 경향. 인사이트: 배치 크기와 무관하게 트렌드 유사하며, 리플레이 과다 시 도메인 성능 저하 가능 → 균형 설계 필요.

테이블 요약 (결과·인사이트)
- Table 1 (Telco DACP 데이터): 총 305GB, Telco 45.9%, 한국어 33.0%, 영어 16.2%, 코드 4.9%. 인사이트: 한국어·도메인 비중을 크게 두되 리플레이(일반)와 코드로 범용성/도구성도 확보.
- Table 2 (일반 벤치): Telco DACP 후 일반 성능 변화는 -7%~+1% 수준의 소폭 변동. 인사이트: 도메인 적응이 일반 능력을 크게 훼손하지 않음(리플레이 전략 유효).
- Table 3 (Telco 벤치): 모든 크기/백본에서 대폭 향상(+41%~+69%). 인사이트: 동일 레시피로도 백본·파라미터 불문 도메인 성능 일관 향상. 작은 모델이 큰 바닐라를 능가.
- Table 4 (Finance 평균지표): EXAONE(2.4B/7.8B)에서 금융 +49%/+31%, 일반 -6%/-1%. 인사이트: 금융 도메인에서도 동일한 패턴(도메인 대폭 향상, 일반 성능 소폭 변화).
- Table 5 (금융 MRC MRR 어블레이션): 바닐라 47.64 → 포스트트레이닝만 37.64(악화) → DACP+포스트 73.61(대폭 개선). 인사이트: 지식 주입(DACP) 없이 SFT/정렬만으로는 성능 한계. DACP와 포스트트레이닝의 결합이 핵심.
- Table 6 (콜센터 요약 인간평가): 2.4B 기준 Telco DACP 4.50 vs 베이스 4.13. 인사이트: 실제 업무 요약 품질 향상(요청/응대 모두).
- Table 7 (리플레이 비율-일반): 50~75% 구간에서 일반 성능 포화·상향, 베이스 수준 근접/상회. 인사이트: 리플레이 50%가 망각 억제의 사실상 최소 기준선.
- Table 8 (리플레이 비율-텔코): 리플레이↑ 시 텔코 평균 하락, 그래도 50%에서 베이스 대비 개선 유지. 인사이트: 리플레이 과다하면 도메인 이득 감소 → 균형 필요(본 논문은 50% 채택).
- Table 9 (Finance DACP 데이터): 총 53GB(도메인 46%). 인사이트: 텔코보다 작은 코퍼스에서도 DACP 효과 입증 → 중규모 데이터로도 산업 적용 가능.
- Table 10 (Finance RAG 리더보드): 같은 hit@3(0.91)에서 DACP 모델의 최종 점수 0.683로 상향. 인사이트: 검색 품질이 같아도 생성기(LLM)의 도메인 적합도가 정답성 향상에 기여.
- Table 11 (Telco 벤치 구성): 어휘/단문QA/대화응답/요약/대화분류 등 5형태, 다양한 측정치. 인사이트: 콜센터 실업무를 반영한 종합 벤치로 도메인 실효성 테스트에 적합.
- Table 12 (Finance 상세): EXAONE Finance 7.8B IT가 Finance Avg 59.54로 대형 일반모델(예: Qwen 32B IT 50.48) 상회, General Avg는 -1% 내 관리. 인사이트: sLLM+DACP가 대형 바닐라 대비 도메인 경쟁우위.
- Table 13 (요약 평가 프레임): 히스토리 추적 가능성 중심의 1–5 척도. 인사이트: 서비스 관점 지표로 현업 적합성 높은 평가 설계.
- Table 14 (이중언어 예시): 도메인 질의→문서 추천→일반 대화까지 수행. 인사이트: 도메인 적응 후에도 일반 대화 능력 유지 필요성 및 가능성 시연.
- Table 15 (금융 MRC 평가 프레임): 1–5 랭킹×MRR 기반 정량화. 인사이트: 문서이해·형식성·생성·암기 등 세부 역량을 일관되게 측정.
- Table 16–17 (Telco 예시): 요금제 추론·대화 분류·요약 등 실제 콜센터 시나리오. 인사이트: 규칙·용어·정책 이해와 컨텍스트 처리 능력 요구 → DACP 적합 과제.
- Table 18–19 (Finance 예시): 약관/법규·보상 조건 등 정밀 독해와 정확한 근거 적용 필요. 인사이트: 일반 SFT만으로는 어려운 영역 → DACP 필요성 강화.
- Table 20 (RAG 평가 샘플): 379문항, 규칙 기반 실패 판정. 인사이트: Figure 5의 실패율 비교의 근거 데이터셋.
- Table 21 (RAG 평가 프레임): 금융 도메인 문서 기반 MRC를 MRR로 측정. 인사이트: 순위형 평가로 미세 성능 차이까지 반영.

부록(Appendix) 요약 (결과·인사이트)

- A (리플레이 비율 실험): 3B 토큰, EXAONE 2.4B로 스크리닝. 일반 성능은 50% 이상에서 포화, 텔코 성능은 25% 이후 급감 추세 → 전체 균형으로 50% 채택. 인사이트: 제한된 토큰 예산에서 가장 무난한 균형점이 50%.
- B (학습 세팅): 코사인 디케이, 초기 LR 1e-5, 컨텍스트 32K, 글로벌 배치 2,048. 인사이트: 스텝·LR 과도 시 망각 위험 → 비교적 보수적 스케줄로 안정화.
- C (일반 벤치 세부): vLLM 기반, CoT 미사용, few-shot(KMMLU/HAERAE/GSM8K-Ko/MMLU 5-shot, BBH 3-shot). 인사이트: 서비스 추론비용 고려한 세팅에서의 일반성 보존 확인.
- D (Telco 벤치 세부): LM Eval Harness 사용, RAG QA 테스트 병행. 인사이트: 통신·네트워크 지식과 콜센터 과업을 함께 커버하도록 설계.
- E (Finance DACP 데이터): 총 53GB(도메인 46%). 인사이트: 데이터가 텔코 대비 적어도 DACP 효과 유지 → 도메인 규모 하한선의 현실적 가이드.
- F (Finance 벤치 세부): FinPMCQA(선다형·LL), FinPQA(생성·Rouge-L), FinSM(요약·Rouge-L). 인사이트: 실무 문서기반 다양한 과제에서 일관된 향상 확인.
- G (콜센터 평가 세부): 인간평가(1–5)로 요청/응대 요약 품질 비교. 인사이트: 모델 개선이 사용자 체감 품질로 연결됨을 검증.
- H (금융 RAG 평가 세부): top-3 동일 조건에서 DACP 모델이 더 높은 점수. 인사이트: 동일 검색 품질에서도 생성기 도메인 적합도가 최종 성능 좌우.

총괄 인사이트
- 동일 DACP 레시피가 다른 백본·모델 크기·도메인(텔코/금융)에서 일관된 이득.
- 리플레이 50%가 일반 보존과 도메인 향상의 실무적 균형점.
- 포스트트레이닝만으로는 부족하며, DACP로 지식을 주입한 뒤 포스트트레이닝을 해야 서비스 성능이 크게 향상.
- RAG에서도 동일Retriever 조건에서 생성기 도메인 적응이 실패율을 크게 낮춤.
- sLLM(+DACP)이 대형 바닐라 모델을 도메인 과업에서 능가 → 비용 효율적 배치에 유리.



Figures
- Figure 1: DACP-applied sLLMs outperform base models on Telco and service tasks while preserving general abilities. Insight: strong cost–performance trade-off with small models.
- Figure 2: End-to-end pipeline (DACP → post-training → API/service → feedback). Insight: reusable “recipe” across domains/backbones for scalable ops.
- Figure 3: Replay ratio trade-off. 50% achieves the best balance: preserves general skills and improves domain performance. Insight: practical default for mid-scale DACP.
- Figure 4: NW QA example. Warns against overfitting to domain at the expense of general chat quality. Insight: balance matters for UX.
- Figure 5: In network QA RAG, DACP slashes failure rates; a smaller DACP model beats a larger vanilla one. Insight: generator’s domain fit matters even with the same retrieval.
- Figure 6: Replay × batch size. General accuracy rises with replay; Telco drops beyond ~25%. Insight: consistent trend across batch sizes; excessive replay can hurt domain gains.

Tables
- Table 1: Telco DACP data (305GB): heavy Telco/Korean with some code. Insight: mixes domain focus with general/code for robustness.
- Table 2: General benchmarks after Telco DACP: minor changes (-7% to +1%). Insight: general capabilities largely retained.
- Table 3: Telco benchmarks: large gains across sizes (+41% to +69%). Insight: small DACP models can surpass larger vanilla ones.
- Table 4: Finance averages: +49%/+31% on Finance, -6%/-1% on General. Insight: mirrors Telco pattern.
- Table 5: Finance MRC ablation (MRR): post-training alone hurts; DACP+post yields big jump (73.61). Insight: knowledge injection via DACP is essential.
- Table 6: Human eval (call summaries): DACP model clearly preferred (4.50 vs 4.13). Insight: tangible UX improvement.
- Table 7: General vs replay: performance saturates from ~50%. Insight: 50% replay effectively mitigates forgetting.
- Table 8: Telco vs replay: domain average declines as replay grows; still above base at 50%. Insight: don’t overdo replay; keep balance.
- Table 9: Finance DACP data 53GB. Insight: mid-scale data suffices for strong gains.
- Table 10: Finance RAG: identical hit@3, higher final score with DACP. Insight: better generator boosts end accuracy.
- Table 11: Telco benchmark design spans vocabulary, QA, dialog, summarization, classification. Insight: realistic call center coverage.
- Table 12: Finance details: EXAONE Finance 7.8B beats larger general models on Finance with near-parity general. Insight: domain-optimized sLLMs > larger vanilla.
- Table 13: Summary eval framework focuses on traceability (1–5). Insight: service-aligned metrics.
- Table 14: Bilingual dialog shows domain QA, doc suggestion, and general chit-chat. Insight: balanced adaptation in action.
- Table 15: Finance MRC framework with MRR. Insight: consistent, reproducible scoring across facets.
- Tables 16–17: Telco task examples (plan reasoning, dialog CLS/SUM). Insight: requires policy/term understanding and context handling.
- Tables 18–19: Finance examples (terms/law/policy). Insight: stresses precise reading and application → DACP helps.
- Table 20: RAG evaluation sample (379 queries). Insight: basis for failure-rate comparisons (Fig. 5).
- Table 21: RAG eval framework with MRR. Insight: ranking-based metrics capture fine-grained differences.

Appendices
- A: Replay-ratio screening identifies 50% as best overall balance under token budget. Insight: practical guideline.
- B: Training settings (cosine LR 1e-5, 32K context, GB 2,048). Insight: conservative schedule to reduce forgetting.
- C: General benchmark details (vLLM, few-shot, no CoT). Insight: reflects production latency constraints.
- D: Telco benchmark details with RAG QA. Insight: covers both knowledge and contact-center tasks.
- E: Finance DACP dataset (53GB, 46% domain). Insight: strong gains with smaller corpora.
- F: Finance benchmark setup (FinPMCQA/FinPQA/FinSM). Insight: broad document-grounded coverage.
- G: Call-center human eval protocol. Insight: links model gains to perceived quality.
- H: Finance RAG details: DACP improves scores with same top-3 passages. Insight: generator-side domain fit is decisive.

Overall insights
- One DACP recipe generalizes across backbones, sizes, and domains.
- 50% replay is a robust starting point to balance retention/specialization.
- Post-training alone is insufficient; DACP+post-training is key.
- In RAG, better generators (via DACP) reduce failure even with identical retrieval.
- DACP-applied sLLMs can outperform larger vanilla models on domain tasks, enabling cost-efficient deployment.

<br/>
# refer format:



BibTeX
@misc{kim2025ixigen,
  title        = {ixi-GEN: Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining},
  author       = {Kim, Seonwu and Na, Yohan and Kim, Kihun and Cho, Hanhee and Lim, Geun and Kim, Mintae and Park, Seongik and Kim, Ki Hyun and Han, Youngsub and Jeon, Byoung-Ki},
  year         = {2025},
  eprint       = {2507.06795},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2507.06795},
  note         = {v2, 10 Jul 2025}
}

시카고 스타일


Kim, Seonwu, Yohan Na, Kihun Kim, Hanhee Cho, Geun Lim, Mintae Kim, Seongik Park, Ki Hyun Kim, Youngsub Han, and Byoung-Ki Jeon. 2025. “ixi-GEN: Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining.” arXiv, July 10, 2025. https://arxiv.org/abs/2507.06795.
