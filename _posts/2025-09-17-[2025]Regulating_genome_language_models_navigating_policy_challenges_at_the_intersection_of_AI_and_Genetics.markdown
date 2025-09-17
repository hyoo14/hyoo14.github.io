---
layout: post
title:  "[2025]Regulating genome language models: navigating policy challenges at the intersection of AI and genetics"
date:   2025-09-17 04:42:52 -0000
categories: study
---

{% highlight ruby %}

한줄 요약:  GLM의 규제 공백을 분석하기 위해 가상의 GenePredictAI 사례 3가지를 제시하고, AI·유전체 규제 비교 검토와 함께 규제 샌드박스·인증·역량 허브·소프트 거버넌스 및 프라이버시 보존(동형암호 등)·임베딩·연합학습·설명가능성(주의집중 시각화, LIME/SHAP, in silico mutagenesis 등) 같은 기술적 수단을 통합한 다층형 거버넌스를 설계했다.

(ai+bio와 법)  



짧은 요약(Abstract) :

이 논문은 유전체 언어 모델(GLM)이 AI와 유전체학의 결합을 통해 생물학적 발견, 의료 혁신, 치료 설계에 큰 잠재력을 가지지만, 기존의 AI 거버넌스와 유전정보 규제가 각각으로는 해결하기 어려운 새로운 규제 공백을 만든다고 지적합니다. 특히 AI는 광범위한 데이터 활용을, 유전체 규제는 엄격한 프라이버시 보호와 동의를 중시해 양자 사이에 긴장이 발생합니다. 저자들은 GLM이 질병위험 예측, 국제 공동연구, 오픈소스 배포 같은 실제 적용에서 기존 규제를 흔드는 양상을 분석하고, 이를 해소하기 위해 정책 혁신(규제 샌드박스, 인증 프레임워크)과 기술적 해법(프라이버시 보존, 모델 해석가능성)을 결합한 다층적 거버넌스 체계를 제안합니다. 이러한 적응형 규제를 통해 개인 권리 보호, 형평성 증진, 생물안보 우려 대응을 병행하면서 GLM의 책임 있는 발전을 가능하게 하자는 것이 핵심 메시지입니다.



This paper argues that genome language models (GLMs)—the fusion of AI and genomics—offer major advances for discovery, healthcare, and therapeutics but also create regulatory gaps that neither AI governance nor genomic privacy regimes can address alone. AI principles favor broad data use, while genomic rules demand strict privacy and informed consent, producing tension. The authors show how GLMs strain existing frameworks in applications like disease risk prediction, cross-border collaboration, and open-source release. They propose a multilayered governance approach that blends policy tools (regulatory sandboxes, certification) with technical measures (privacy-preserving methods, interpretability). The goal is to enable responsible GLM innovation while safeguarding individual rights, promoting equity, and addressing biosecurity concerns.


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


1) 모델 패밀리와 핵심 아키텍처
- 인코더 기반(GLM-Encoder)
  - 대표: DNABERT, Nucleotide Transformer, Google AlphaGenome(맞춤형 인코더)
  - 특징: BERT 계열처럼 입력 서열을 양방향 컨텍스트로 인코딩하여 분류·예측에 사용. 프리트레이닝 목표로 “마스크드 언어모델(MLM)”을 사용.
  - 장점: 서열 전역 컨텍스트를 활용한 분류/예측(프로모터 식별, 변이 효과 예측 등)에 강함.

- 디코더 기반(GLM-Decoder; Genomic GPT류)
  - 대표: EVO(Genome-scale generative model), HyenaDNA(장거리 의존성 효율 처리), 기타 GPT류
  - 특징: 자연어 GPT처럼 “다음 토큰 예측(next-token prediction)”으로 학습. 생성·예측 모두 가능.
  - 장점: 서열 생성, 장거리 상호작용 포착, 멀티샷/제로샷 적용 등에 유리.

- 멀티모달 GLM
  - 예: 텍스트 도구/주석과 DNA 서열처리를 결합(GeneGPT 등). 텍스트 질의+서열 정렬/생성 등 복합 작업을 단일 프레임에서 수행.

- 장거리 의존성 처리 아키텍처 혁신
  - HyenaDNA, Nucleotide Transformer, AlphaGenome에서 언급된 효율적 주의(attention)·상태공간(state space)·확장형 컨텍스트 설계로 긴 서열(수십 kb 이상) 문맥을 다루도록 개선.

2) 입력 표현(토크나이제이션)과 프리트레이닝 목표
- 토큰화(tokenization)
  - k-mer: 일정 길이(k)의 고정 서열 조각(예: 3-mer, 6-mer 등).
  - BPE(Byte-Pair Encoding) 적응: 자연어에서 차용한 가변 길이 토큰화로, 데이터 주도적으로 빈도 높은 서열 조각을 토큰으로 채택.
- 사전학습(프리트레이닝) 목표
  - 마스크드 언어모델(MLM): DNABERT류는 서열 일부 토큰을 무작위 마스킹하고, 주변 문맥으로 복원하도록 학습.
  - 다음 토큰 예측: GPT류 GLM(EVO, 일부 생성 모델)은 순차적으로 다음 염기를 예측.
- 사전학습 샘플 구성
  - 전장 유전체를 10~510 bp 길이로 비중첩 분할 또는 랜덤 샘플링해 무감독 샘플 생성(논문 설명).
  - 대규모 비라벨 유전체 데이터(인간 및 타종)로 학습하여 생물학적 문법(모티프·조절 신호·장거리 상관)을 내재화.

3) 학습 데이터와 다운스트림 태스크
- 학습 데이터
  - 대규모 비라벨 유전체(인간, 다수 종) 서열·변이 데이터. 공개 인구집단 데이터도 활용되나, 샘플 분포 편향(특정 조상집단 과대표집 등) 주의 필요.
- 다운스트림/응용
  - 변이 효과 예측(Variant Effect Prediction)
  - 조절요소·프로모터 식별
  - 유전자 발현 패턴 예측
  - 질병 위험 예측(임상·D2C 맥락 포함)
  - 합성 서열 생성(바이오시큐리티 관점에서 추가 통제 필요)

4) 프라이버시 보존 데이터·학습 기법(Privacy-preserving Methods)
- 연합학습(Federated Learning)
  - 원자료를 기관 외부로 반출하지 않고, 로컬에서 모델 업데이트만 교환. 국경 간 데이터 이전 제한·규제 불일치 속에서 협업 학습을 가능케 함.
- 임베딩/변환(Embeddings and Transformations)
  - 고차원 유전체를 저차원 벡터로 임베딩해 유틸리티를 유지하면서 직접 식별 가능성을 낮춤(특정 희귀 변이·알렐 조합으로 재식별 위험은 여전하므로 주의).
  - 자기지도/지도학습(오토인코더, 트랜스포머 기반)으로 표형 예측·군집 등 목적에 맞춘 표현학습.
- 암호화 기반 계산(Homomorphic Encryption)
  - 복호화 없이 암호문 상태에서 계산 수행. 높은 프라이버시 보장 대가로 연산 비용 증가.
- 사생활보호형 점수계산(예: PRS)
  - D2C 시나리오에서 원시 서열 노출 없이 다유전자 위험점수(PRS)를 비식별 방식으로 계산·전달하는 앱/파이프라인.

5) 해석가능성·설명가능성(Interpretability/Explainability) 기법
- 어텐션 가시화(Transformer Visualizers)
  - 모델이 예측 시 주목한 서열 위치(예: TATA box, CpG island)를 시각화. 직관적이지만, 진짜 의사결정 근거와 불일치할 수 있어 보조적 사용 권장.
- 기여도/속성 기법(Attribution)
  - Integrated Gradients, DeepLIFT, GradCAM, Saliency maps 등으로 염기/모티프 단위 기여도를 정량화.
  - LIME, SHAP 같은 모델 불가지론적(post-hoc) 방법으로 지역적 설명 제공.
  - SQUID(대리모델·서로게이트 기반): DNN을 근사하는 간이모델을 학습해 기여도 신뢰도를 높이는 접근.
- 교란·절제(Perturbation/Ablation)
  - 서열 마스킹/랜덤화, in silico 돌연변이 주입, 특정 어텐션-헤드/뉴런 절제 등으로 인과·기여 검증.
- 권장 사용 원칙
  - 복수 기법 교차검증(일관성 확인)
  - 생물학적 사전지식(모티프·조절 네트워크)과 결합
  - 벤치마크·그라운드트루스 기반 정량 평가
  - 가능하면 실험적 검증(딥 뮤테이셔널 스캐닝, 타깃 변이 실험 등)과 연계

6) 모델 릴리즈·운영 관점에서의 기술적 안전장치(논문 내 제안)
- 공개 배포 시 모델 가중치로부터 개인 유전정보가 추출될 위험(리버스 프로빙)을 줄이기 위한:
  - 프라이버시 보존 임베딩/변환 적용 후 릴리즈
  - 응답 필터링만으로는 불충분하므로(다운스트림 재사용 가능성), 사전 인증·라벨링·감사 기준과 결합
  - 지속 모니터링(파인튜닝·업데이트로 모델이 동적으로 변함)

7) 대표 모델·방법 키워드 요약
- 모델: DNABERT(MLM 인코더), Nucleotide Transformer(확장 문맥 인코더), AlphaGenome(커스텀 인코더), HyenaDNA(장거리 효율), EVO(디코더·생성)
- 토크나이제이션: k-mer, BPE
- 프리트레이닝: MLM, Next-token prediction
- 프라이버시: Federated learning, Embeddings/Transformations, Homomorphic encryption, Privacy-preserving PRS
- 해석: Attention 시각화, IG/DeepLIFT/GradCAM/Saliency, LIME/SHAP, SQUID(서로게이트), Perturbation/Ablation, in silico mutagenesis




1) Model families and core architectures
- Encoder-based GLMs
  - Examples: DNABERT, Nucleotide Transformer, Google AlphaGenome (custom encoder)
  - Traits: BERT-like bidirectional encoders trained with masked language modeling (MLM), well-suited for classification/prediction (e.g., promoter identification, variant effect prediction).
- Decoder-based (Genomic GPT) GLMs
  - Examples: EVO, HyenaDNA, other GPT-style models
  - Traits: Trained with next-token prediction; good for sequence generation and capturing long-range dependencies.
- Multimodal GLMs
  - Combine textual tools/annotations with DNA sequence processing (e.g., question answering plus alignment/generation in one framework).
- Long-context innovations
  - Efficient attention and state-space–style designs (HyenaDNA, Nucleotide Transformer, AlphaGenome) to handle very long genomic contexts.

2) Input representation and pretraining objectives
- Tokenization
  - k-mer tokens: fixed-length subsequences (e.g., 3-mer, 6-mer).
  - Byte-Pair Encoding (BPE): variable-length, data-driven tokenization adapted from NLP.
- Pretraining objectives
  - Masked Language Modeling (MLM): randomly mask tokens and reconstruct them from context (DNABERT-like).
  - Next-token prediction: GPT-style autoregressive modeling (e.g., EVO).
- Pretraining sample construction
  - Split whole genomes into 10–510 bp chunks by non-overlapping segmentation or random sampling; train in an unsupervised fashion (as described in the paper).
  - Large, unlabeled genomic corpora across human and multiple species to internalize biological “grammar.”

3) Training data and downstream tasks
- Training data
  - Massive unlabeled genomic sequences and variants (human and non-human). Beware sampling bias (e.g., ancestral overrepresentation).
- Downstream applications
  - Variant effect prediction
  - Regulatory element/promoter identification
  - Gene expression pattern prediction
  - Disease risk prediction (clinical and D2C use cases)
  - Synthetic sequence generation (with biosecurity caveats)

4) Privacy-preserving data/learning techniques
- Federated learning
  - Train collaboratively without sharing raw data; exchange model updates only—useful under cross-border data-transfer constraints.
- Embeddings/transformations
  - Map high-dimensional genomes to lower-dimensional vectors to retain utility while reducing identifiability; still consider reidentification risk via rare alleles.
  - Learn embeddings via unsupervised (autoencoders, transformer objectives) or supervised objectives (e.g., phenotype prediction).
- Homomorphic encryption
  - Compute directly on encrypted data; strong privacy at higher computational cost.
- Privacy-preserving scoring (e.g., PRS)
  - Compute polygenic risk scores without exposing raw genomes in D2C settings.

5) Interpretability/explainability methods
- Attention visualization
  - Display what sequence regions receive attention (e.g., TATA box, CpG islands). Intuitive but only supportive due to potential misalignment with true decision logic.
- Attribution methods
  - Integrated Gradients, DeepLIFT, GradCAM, Saliency maps quantify base/motif contributions.
  - LIME, SHAP provide model-agnostic, local post-hoc explanations.
  - SQUID (surrogate-based) improves attribution fidelity by approximating the DNN with interpretable surrogates.
- Perturbation/ablation
  - Sequence masking/randomization, in silico mutagenesis, and attention-head/neuron ablation to probe causality/contribution.
- Recommended practice
  - Cross-validate across multiple methods; incorporate biological priors; evaluate on genomic ground-truth benchmarks; couple with experimental validation (e.g., deep mutational scanning, targeted mutagenesis).

6) Technical safeguards for release/operation (as proposed in the paper)
- When releasing models, mitigate extraction risks from weights:
  - Prefer privacy-preserving embeddings/transformations prior to release; response filtering alone is insufficient due to downstream reuse.
  - Pair with certification/labeling/auditing criteria and continuous monitoring, since GLMs evolve via fine-tuning.

7) Quick index of representative methods mentioned
- Models: DNABERT (MLM encoder), Nucleotide Transformer (long-context encoder), AlphaGenome (custom encoder), HyenaDNA (efficient long-range), EVO (decoder/generative)
- Tokenization: k-mer, BPE
- Pretraining: MLM, Next-token prediction
- Privacy: Federated learning, Embeddings/Transformations, Homomorphic encryption, Privacy-preserving PRS
- Interpretability: Attention visualization, IG/DeepLIFT/GradCAM/Saliency, LIME/SHAP, SQUID (surrogates), Perturbation/Ablation, in silico mutagenesis

비고
- 위 내용은 요청하신 논문 본문에 기술된 모델·데이터·기법만을 근거로 요약했습니다. 이 논문은 규제·ガバ넌스 관점을 주로 다루지만, “GLM의 진화·아키텍처” 및 “기술적 접근” 섹션에 핵심 메써드가 체계적으로 정리되어 있습니다.


<br/>
# Results


한글 요약(체계적 정리)
- 논문 성격
  - 유형: 정책/거버넌스 관점의 퍼스펙티브(리뷰) 논문
  - 결과물 성격: 정량 벤치마크가 아닌 개념·사례·프레임워크(도식·표) 제시
  - 실험결과/수치: 본문에 미제시

- 경쟁(비교) 모델 라인업(논문 내 직접 언급·인용)
  - 인코더 계열(GLM/유사 GLM)
    - DNABERT / DNABERT‑2: BERT를 DNA 토큰(k‑mer 등)에 적용, 마스크드LM 사전학습(Ji et al., 2021; Zhou et al., 2023)
    - Nucleotide Transformer: 대규모 인코더 기반 기초모델(Dalla‑Torre et al., 2025)
    - AlphaGenome(구글): 커스텀 인코더형 아키텍처(Avsec et al., 2025 인용 문맥 내 언급)
  - 디코더/생성 계열
    - EVO(“genomic GPT”류): next‑token 예측 기반, 분자~게놈 스케일 시퀀스 모델링(Nguyen et al., 2024)
    - HyenaDNA: 긴 컨텍스트 시퀀스(next‑token) 모델링(Nguyen et al., 2023)
    - GenSLMs: 바이러스(예: SARS‑CoV‑2) 진화 동역학 학습(Zvyagin et al., 2022)
  - 기타 GLM 관련 모델/기법
    - LOGO, GROVER(모티프·비부호 영역 해석 지원)(Yang et al., 2022; Sanabria et al., 2024)
    - 16S rRNA 주의(attention) 시각화 사례(Zhao et al., 2021)

- 학습/테스트 데이터(논문에서 기술된 범주)
  - 사전학습: 대규모 비라벨 게놈 데이터(인간 전장유전체 및 변이, 타 생물종 포함)
  - 응용 맥락: 바이오뱅크/인구집단 연구 데이터(예: 시나리오에서 ‘인구집단 공개 데이터’로 학습)
  - 데이터 공개·접근: 규제 시나리오에서 EU GDPR, HIPAA 등 법·윤리 제약 하의 데이터 이동/공유 이슈 강조
  - 구체적인 벤치마크 데이터셋명/분할: 미보고

- 다운스트림 태스크(논문이 열거·인용)
  - 변이 효과 예측(variant effect prediction)
  - 조절요소(프로모터 등) 식별, 유전자 발현 패턴 예측
  - 병원성 변이 탐지, 질병 위험도 예측
  - 합성 서열 생성(생성형 GLM)
  - 일부 모델에서 멀티모달(텍스트 질의+DNA 시퀀스 처리/정렬/생성) 기능

- 평가 메트릭(논문 내 보고 여부)
  - AUROC/AUPRC/정확도 등 구체 메트릭: 본 논문에서는 미제시
  - 모델 간 정량 비교표/순위: 미제시
  - 대신, “장거리 상호작용 포착”, “기초모델로서 광범위 전이·미세튜닝 가능성” 같은 질적 성능 특성 기술

- 비교·대조 관점(논문이 제시하는 핵심 비교 축)
  1) 분석 패러다임 비교
     - 전통적 유사성 정렬/모티프 기반(배경: BLAST, HMM, MEME 등) vs
     - 언어모델 기반(GLM): 게놈을 “텍스트”로 보고 문맥·장거리 상호작용을 직접 학습
  2) 아키텍처 비교
     - 인코더(BERT류, 마스크드 LM) vs 디코더(GPT류, next‑token), 멀티모달 확장
  3) 개방형 공개 vs 폐쇄형 배포
     - AI 커뮤니티의 개방성(가중치 공개)을 통한 혁신 촉진 vs
     - 유전체 분야의 엄격한 개인정보·동의·재식별 위험 관리
  4) 규제 프레임 비교
     - AI 규제(성능, 영향평가, 투명성, 리스크 등) vs
     - 유전체 규제(개인·가족 레벨 프라이버시, 동의, 차별 금지)

- 논문이 제시하는 “결과물”에 해당하는 도식·표
  - Fig. 1: GLM 개요(토큰화, 인코더/디코더 학습·활용 흐름)
  - Fig. 2: GLM 생애주기 단계별 규제 개입 지점
  - Fig. 3: 다층 거버넌스 프레임워크(기술·제도·정책 레이어)
  - Fig. 4: 프라이버시 보존 기법 스펙트럼(원시서열→임베딩/변환→동형암호 등, 효용–프라이버시 트레이드오프)
  - Table 1: 시나리오별 규제 충돌 포인트(질병위험 예측, 국경 간 협업, 오픈소스 공개)
  - Table 2: AI vs 유전체 규제의 대비와 GLM 고유 과제(프라이버시, 동의, 편향, 해석가능성, 관할·책임)

- 정리(본 논문의 “비교·평가”에 대한 결론적 메시지)
  - 정량 벤치마크는 없으나, GLM은
    - 전통 기법이 놓치기 쉬운 장거리 상호작용 및 비선형 패턴을 포착하는 잠재력
    - 다양한 태스크로 전이 가능한 “기초모델”로서의 범용성
  - 동시에
    - 재식별/가족 노출 위험, 동의범위 초과 추론, 편향·해석가능성 부족, 환각, 생물안보 등 고유 리스크
  - 따라서 “모델 간 성능 경쟁”보다 “거버넌스·안전장치와의 결합”이 핵심 과제로 제시됨
    - 규제 샌드박스, 인증·라벨링, 연합학습·프라이버시 보존 임베딩, 해석기법(주의 시각화/특성 기여/섭동·절제) 등 제안

- 무엇이 보고되지 않았는가(연구 설계 관점에서의 공백)
  - 표준 벤치마크 세트/프로토콜, 데이터 분할·크기, 사전학습 스케일
  - 수치 메트릭(AUROC/AUPRC/정확도 등)과 통계검정
  - 동등 조건에서의 SOTA 간 정량 비교



- Paper type
  - Perspective/review focused on policy and governance; not an empirical benchmarking study.
  - No quantitative results (datasets/splits/metrics/SOTA tables) are reported.

- “Competing models” mentioned (by family; as cited in the paper)
  - Encoder-style GLMs/related
    - DNABERT / DNABERT‑2 (masked‑LM over k‑mers; Ji et al., 2021; Zhou et al., 2023)
    - Nucleotide Transformer (Dalla‑Torre et al., 2025)
    - AlphaGenome (custom encoder; referenced)
  - Decoder/generative
    - EVO (genomic GPT‑like next‑token model; Nguyen et al., 2024)
    - HyenaDNA (long‑context next‑token; Nguyen et al., 2023)
    - GenSLMs (viral genomes, e.g., SARS‑CoV‑2; Zvyagin et al., 2022)
  - Other GLM‑adjacent models/tools
    - LOGO, GROVER (non‑coding interpretation support; Yang et al., 2022; Sanabria et al., 2024)
    - Attention visualization on 16S rRNA (Zhao et al., 2021)

- Data (as described conceptually)
  - Pretraining on large‑scale unlabeled genomic data (human whole genomes and variants; other species).
  - Use contexts include biobank and population studies.
  - No specific benchmark datasets or splits are provided in this paper.

- Downstream tasks (listed/illustrated)
  - Variant effect prediction; regulatory element (e.g., promoter) identification; gene expression pattern prediction.
  - Pathogenic variant detection; disease risk prediction.
  - Synthetic sequence generation; some multimodal capabilities (text + DNA processing/generation).

- Metrics (reporting status)
  - No task metrics (e.g., AUROC/AUPRC/accuracy) or head‑to‑head numbers reported.
  - Performance is discussed qualitatively (e.g., capturing long‑range interactions; foundation‑model transferability).

- Comparative axes emphasized
  1) Methodological: traditional alignment/motif discovery vs GLM treating genome as “text”.
  2) Architecture: encoder (masked‑LM) vs decoder (next‑token) vs multimodal.
  3) Openness: open‑weights culture in AI vs stringent privacy in genomics.
  4) Regulation: AI governance (risk, impact, transparency) vs genomic governance (privacy, consent, non‑discrimination).

- “Results‑like” artifacts provided (conceptual figures/tables)
  - Fig. 1: GLM schematic (tokenization; encoder/decoder pretraining and downstream use).
  - Fig. 2: GLM lifecycle mapped to regulatory intervention points.
  - Fig. 3: Layered governance framework (technical–institutional–policy).
  - Fig. 4: Spectrum of privacy‑preserving techniques (raw → embeddings/transformations → homomorphic encryption).
  - Table 1: Scenario‑based regulatory tensions (risk prediction, cross‑border collaboration, open‑sourcing).
  - Table 2: Contrasts between AI vs genomics regulation and GLM‑specific challenges.

- Take‑home comparative message
  - While no quantitative benchmarks are presented, the paper argues GLMs:
    - Offer potential advantages for modeling long‑range, non‑linear genomic dependencies and broad transfer across tasks as foundation models.
    - Pose distinctive risks (re‑identification, familial privacy, consent overreach, bias, lack of interpretability, hallucinations, biosecurity).
  - Therefore, the central “comparison” is not leaderboard performance but the fit/mismatch with existing AI vs genomic governance, and the need for blended oversight (regulatory sandboxes, certification/labeling, federated and privacy‑preserving representations, interpretability toolkits).

- Not reported (from a benchmarking standpoint)
  - Standardized benchmark suites/protocols, data splits/scale, pretraining compute.
  - Numerical metrics and statistical testing.
  - Controlled, apples‑to‑apples SOTA comparisons.




<br/>
# 예제


예시 1. 사전학습(Pretraining): DNABERT 스타일 마스크드 언어모델
- 목적/태스크
  - 대규모 라벨 없는 게놈 서열로부터 염기서열의 문법(문맥·문법적 패턴)을 학습해 범용 표현을 획득.
- 트레이닝 데이터(입력/정제/토큰화)
  - 입력: 인간 및 다종(멀티-스피시즈) 게놈의 길이가 긴 원시 DNA 서열.
  - 분할: 10–510 bp 길이의 서브시퀀스로 랜덤 샘플 또는 비중복 슬라이싱(논문 본문 설명).
  - 토큰화: k-mer(예: k=6) 또는 byte-pair encoding(BPE) 기반 가변 길이 토큰.
  - 마스킹: 각 서브시퀀스에서 일부 토큰을 무작위로 [MASK] 처리.
- 학습 목표/아웃풋
  - 아웃풋: 마스크된 위치의 원래 토큰 예측(분류).
  - 손실: 마스크된 토큰 위치에 대한 크로스 엔트로피.
- 테스트/검증 셋·평가
  - 홀드아웃: 특정 염색체 구간/종/샘플을 홀드아웃하여 과적합 점검.
  - 평가지표: 마스크 복원 정확도/토큰별 퍼플렉서티·Top-k 정확도.
- 주의/연계
  - 사전학습 표현은 이후 다운스트림(프로모터 식별, 변이효과 예측 등)에 파인튜닝/제로샷으로 전이.

예시 2. 다운스트림 분류: 프로모터(또는 조절요소) 식별
- 목적/태스크
  - 주어진 서열 창(window)이 프로모터(또는 인핸서 등 조절요소)인지 이진 분류.
- 트레이닝 데이터(입력/라벨)
  - 입력: 기준 게놈 상 프로모터로 주석된 영역 주변(예: ±500 bp)과 비프로모터 음성 샘플.
  - 전처리: 길이 정규화(예: 1 kb 윈도우), k-mer/BPE 토큰화.
  - 라벨: {1=프로모터, 0=비프로모터}.
- 모델/아웃풋
  - 사전학습 GLM 인코더를 얕게 미세조정(fine-tune).
  - 아웃풋: 시퀀스별 프로모터 확률(시그모이드).
  - 손실: 이진 크로스 엔트로피.
- 테스트/검증 셋·평가
  - 분할: 염색체 단위/유전자 단위 분리로 누수 방지.
  - 평가지표: AUROC, AUPRC, F1, 캘리브레이션(Brier).
  - 설명가능성: Attention 시각화, Integrated Gradients/SHAP로 TATA box, CpG island 등 기여도 확인(논문 본문: 주의·후향 설명의 한계 유념).
- 주의/연계
  - 데이터 편향(개체군 불균형)과 해석가능성 요구(임상 신뢰) 간균형 필요(논문 ‘Bias/Interpretability’ 논의).

예시 3. 변이 효과(병원성) 예측: 참조/대체 서열 맥락 기반
- 목적/태스크
  - 특정 변이(SNV/indel)가 전사조절/스플라이싱/단백질 기능 등에 미치는 영향(병원성 여부/연속점수) 예측.
- 트레이닝 데이터(입력/라벨)
  - 입력: 변이 좌표 중심의 고정 길이 콘텍스트(예: 201 bp). 두 가지 인코딩 중 하나:
    - (A) 참조(ref) 서열 vs 대체(alt) 서열 쌍을 모두 입력하여 Δ효과를 직접 학습.
    - (B) alt 서열만 입력하고 라벨로 효과 점수/등급을 부여.
  - 라벨: 임상 데이터베이스(예: 병원성/양성) 또는 기능 스크리닝에서 유래한 연속 점수.
- 모델/아웃풋
  - 사전학습 GLM 인코더를 미세조정.
  - 아웃풋: 이진(병원성/양성) 또는 연속 점수(기능저하 확률).
  - 손실: 이진 크로스 엔트로피 또는 MSE/Huber.
- 테스트/검증 셋·평가
  - 유전자/위치 홀드아웃으로 근접 누수 방지.
  - 평가지표: AUROC/AUPRC(분류), R²/Spearman(회귀), 임상 cut-off에서의 민감도/특이도.
  - 설명가능성: In silico mutagenesis, Saliency/IG, LIME/SHAP로 모티프/SNP 기여도 파악(논문 본문 인용 기법).
- 주의/연계
  - 예기치 않은 2차 소견(incidental findings) 발생 가능성 및 통지의무·동의 범위는 규제 샌드박스/인증 프레임으로 다룰 것을 논문이 제안.

예시 4. 질병 위험(PRS 연계) 예측: GLM 임베딩 + 프라이버시-보존 계산
- 목적/태스크
  - 개별 개인의 질병 위험 확률 또는 폴리유전성 위험점수(PRS) 산출.
- 트레이닝 데이터(입력/라벨)
  - 입력: (A) 전장 유전체 변이 벡터(선택 SNP 패널) 또는 (B) 서열 기반 임베딩(사전학습 GLM에서 추출).
  - 라벨: 바이오뱅크 코호트의 이진 표현형(질병 진단) 또는 사건 발생 시점(서바이벌).
  - 개인정보/규제: D2C 데이터의 로컬 계산(논문 인용: PRScalc) 또는 연방학습(데이터 국경 유지)로 프라이버시 보존.
- 모델/아웃풋
  - 아웃풋: 위험 확률/PRS 점수. 로지스틱 회귀·소형 MLP·GLM 헤드 등.
  - 손실: 이진 크로스 엔트로피(또는 Cox 손실).
- 테스트/검증 셋·평가
  - 분할: 계통군(ancestry) 계층화 분할, 국경 간(사이트 간) 외부검증.
  - 평가지표: AUC, 캘리브레이션(O/E, calibration slope), NRI/IDI, 서브그룹 공정성(민족·성별별 성능 차).
- 주의/연계
  - 논문 시사점: (i) 동의 범위 초과 추론 위험, (ii) 집단 편향(유럽계 과대표집) 보정 필요, (iii) 국경 간 규제 충돌 시 연방학습/공통 역량 허브(ARCCH) 등으로 해결.

예시 5. 생성/오픈소스 모델의 프라이버시 점검(레드팀·감사)
- 목적/태스크
  - 공개된(또는 공개 예정) GLM이 학습 데이터(개인 유전체) 정보를 유출/재식별 가능한지 점검.
- 테스트 데이터/입력
  - 멤버십 추론: 학습셋 포함 개인 vs 비포함 개인의 서열을 입력, 출력/로짓 차이를 이용해 멤버십 판별.
  - 모델 역추출: 특정 위치/패턴을 유도하는 프롬프트·프로빙으로 훈련 샘플 유사 서열이 생성되는지 확인.
- 아웃풋/평가
  - 아웃풋: 멤버십 판별 정확도/TPR@FPR, 재식별 성공률, 민감정보 추출 가능성.
  - 통제수단: 임베딩/변환의 비가역성 테스트, 응답 필터링만으로는 불충분(논문 3번 시나리오 지적) → 공개 전 인증 체크리스트/감사 통과 필요.
- 주의/연계
  - 논문 제안: 공개 가중치에는 프라이버시-보존 임베딩/변환, 책임성 있는 인증·라벨링, 사후 모니터링이 필수.

예시 6. 국경 간 연합학습(Federated GLM) 운영 예시
- 목적/태스크
  - EU/미국/중국/인도 등 각 관할의 데이터를 이동 없이 공동 사전학습·파인튜닝.
- 트레이닝 데이터/입력
  - 각국 병원/기관의 로컬 서열/변이 데이터. 중앙 서버에는 모델 업데이트(그래디언트/가중치)만 공유.
- 아웃풋/평가
  - 아웃풋: 글로벌 GLM 가중치. 사이트별 외부검증 성능 비교, 통신·암호화 오버헤드 측정.
  - 보안: 안전집계, 차등프라이버시, 안전한 초기화 등을 조합(논문: 연방학습·국경 규제 충돌 해결책으로 제시).
- 주의/연계
  - 규제 샌드박스 안에서 가설-검증, 종료 후 확장 시 정책 완화의 무비판적 이식 금지(논문 권고).

공통 구현 디테일(요약)
- 토큰화: k-mer 또는 BPE. 컨텍스트 길이는 태스크(프로모터 1 kb, 변이 200 bp 등)에 맞춤.
- 손실/최적화: 사전학습은 마스크 복원/넥스트 토큰, 다운스트림은 BCE/MSE/Cox 등.
- 설명가능성: Attention 시각화, Integrated Gradients/DeepLIFT/GradCAM, LIME/SHAP, in silico mutagenesis/마스킹·어블레이션(논문 본문 다수 인용).
- 프라이버시/보안: 로컬 PRS(예: PRScalc), 임베딩/변환의 비가역성 검증, 동형암호·연합학습(논문 그림·서술 참조).
- 공정성/책임: 서브그룹 성능, 캘리브레이션, 데이터 편향 완화, 인증·라벨링, 책임할당(개발사/의료기관/배포기관).





The examples below translate the paper’s core GLM ideas (masked-LM pretraining, regulatory element/variant tasks, disease risk/PRS with privacy, generation and leakage auditing, and cross-border federated training) into concrete, task-oriented setups with explicit training/test inputs and outputs. Where the paper provides concepts rather than datasets, we follow common practice to specify representative setups.

Example 1. Pretraining (DNABERT-style Masked Language Modeling)
- Goal/Task
  - Learn genome “syntax/semantics” from large unlabeled DNA to obtain transferable representations.
- Training data (inputs/tokenization)
  - Inputs: long raw DNA sequences from human and multi-species genomes.
  - Splitting: sample 10–510 bp subsequences by random sampling or non-overlapping slices (as described in the paper).
  - Tokenization: k-mers (e.g., k=6) or BPE-style variable-length tokens.
  - Masking: randomly replace a fraction of tokens with [MASK].
- Objective/Outputs
  - Output: predict the original token at masked positions (classification).
  - Loss: cross-entropy on masked positions.
- Test/Evaluation
  - Hold out specific chromosomes/species/samples to check overfitting.
  - Metrics: masked-token accuracy, perplexity, top-k accuracy.
- Notes
  - The pretrained encoder is later fine-tuned or used zero-shot for downstream tasks.

Example 2. Downstream classification: Promoter (or regulatory element) identification
- Goal/Task
  - Binary classification: whether a sequence window is a promoter (or enhancer).
- Training data (inputs/labels)
  - Inputs: fixed-length windows (e.g., 1 kb) around annotated promoters and matched negatives.
  - Preprocessing: length normalization; k-mer/BPE tokenization.
  - Labels: {1=promoter, 0=non-promoter}.
- Model/Outputs
  - Fine-tune the pretrained GLM encoder with a classification head.
  - Output: promoter probability (sigmoid).
  - Loss: binary cross-entropy.
- Test/Evaluation
  - Splits: gene/chromosome-level separation to prevent leakage.
  - Metrics: AUROC, AUPRC, F1, calibration (Brier).
  - Explainability: attention maps, Integrated Gradients/SHAP to highlight TATA box, CpG islands (noting limits of post-hoc explanations per the paper).
- Notes
  - Address population/data imbalance and clinical interpretability requirements.

Example 3. Variant effect (pathogenicity) prediction with sequence context
- Goal/Task
  - Predict functional/pathogenic impact of SNVs/indels (binary or continuous).
- Training data (inputs/labels)
  - Inputs: fixed-length context centered at the variant (e.g., 201 bp).
    - Option A: feed ref and alt sequences to learn Δ-effect directly.
    - Option B: feed alt only; labels carry effect magnitude/category.
  - Labels: clinical annotations (pathogenic/benign) or functional assay-derived scores.
- Model/Outputs
  - Fine-tune the pretrained GLM.
  - Output: probability of pathogenicity (binary) or continuous impact score.
  - Loss: BCE or MSE/Huber.
- Test/Evaluation
  - Hold out by gene/region to avoid near-duplicate leakage.
  - Metrics: AUROC/AUPRC (classification), R²/Spearman (regression), sensitivity/specificity at clinical cutoffs.
  - Explainability: in silico mutagenesis, saliency/IG, LIME/SHAP (as referenced in the paper).
- Notes
  - Incidental/secondary findings handling should follow sandbox/certification protocols highlighted by the paper.

Example 4. Disease risk prediction (PRS) with privacy-preserving computation
- Goal/Task
  - Estimate individual disease risk or compute a polygenic risk score (PRS).
- Training data (inputs/labels)
  - Inputs: (A) genotype vectors (selected SNP panels) or (B) sequence-based embeddings extracted from a pretrained GLM.
  - Labels: binary disease status or time-to-event from biobank cohorts.
  - Privacy/regulation: perform local PRS computation (paper cites PRScalc) or federated learning to keep data in-jurisdiction.
- Model/Outputs
  - Output: disease risk probability or PRS value.
  - Loss: binary cross-entropy (or Cox loss for survival).
- Test/Evaluation
  - Splits: stratified by ancestry; external validation across sites/countries.
  - Metrics: AUC, calibration (O/E, slope), NRI/IDI; subgroup fairness (performance gaps by ancestry/sex).
- Notes
  - Paper stresses consent scope, bias from Eurocentric datasets, and cross-border compliance; mitigations include federated learning and shared capacity hubs.

Example 5. Privacy red teaming/auditing for open-source or generative GLMs
- Goal/Task
  - Assess whether a released (or to-be-released) GLM leaks or enables re-identification of training genomes.
- Test inputs
  - Membership inference: submit sequences from known members vs non-members; use output score gaps to classify membership.
  - Model inversion/probing: craft prompts/inputs to elicit near-verbatim training sequences or sensitive attributes.
- Outputs/Evaluation
  - Outputs: membership classification, success rate of sequence/attribute extraction.
  - Metrics: attack AUC/TPR@FPR, re-identification success, sensitive info extractability.
  - Controls: require irreversibility tests for embeddings/transformations; response filtering alone is insufficient (paper’s Scenario 3), so pre-release certification/audits are needed.
- Notes
  - The paper recommends privacy-preserving embeddings, accountability via certification/labels, and post-release monitoring.

Example 6. Cross-border federated GLM training
- Goal/Task
  - Jointly pretrain/fine-tune across EU/US/China/India without moving raw data.
- Training inputs
  - Local genomes/genotypes remain on-prem; only model updates (gradients/weights) are shared to an aggregator.
- Outputs/Evaluation
  - Output: global model weights; compare site-wise external validation; measure comms/crypto overhead.
  - Security: secure aggregation, (optional) differential privacy, careful initialization (as per the paper’s privacy and jurisdiction sections).
- Notes
  - Use regulatory sandboxes to trial such setups; do not “lift-and-shift” sandbox flexibilities to production without safeguards (paper’s guidance).

Common implementation details (summary)
- Tokenization: k-mer or BPE; context length tailored to task (e.g., 1 kb for promoters; ~200 bp for variant effects).
- Objectives: MLM/next-token for pretraining; BCE/MSE/Cox for downstream.
- Explainability: attention, Integrated Gradients/DeepLIFT/GradCAM, LIME/SHAP, in silico mutagenesis/masking/ablation (as cataloged in the paper).
- Privacy/Security: local PRS (e.g., PRScalc), irreversibility checks for embeddings, homomorphic encryption/federated learning where appropriate.
- Fairness/Accountability: subgroup performance and calibration, bias mitigation, certification/labeling, and clear liability allocation among developer/provider/deployer (per the paper).




<br/>
# 요약

- 메써드: 저자들은 GLM의 규제 공백을 분석하기 위해 가상의 GenePredictAI 사례 3가지를 제시하고, AI·유전체 규제 비교 검토와 함께 규제 샌드박스·인증·역량 허브·소프트 거버넌스 및 프라이버시 보존(동형암호 등)·임베딩·연합학습·설명가능성(주의집중 시각화, LIME/SHAP, in silico mutagenesis 등) 같은 기술적 수단을 통합한 다층형 거버넌스를 설계했다.
- 결과: 데이터 프라이버시·동의, 편향·공정성, 해석가능성, 관할·책임 등 축에서 AI 규제와 유전체 규제가 어긋나 GLM 특유 위험(재식별, 비의도적 친족 노출, 국경 간 준수, 오픈소스 유출)이 발생함을 정리하고(Table 2), 모델 생애주기 단계별 개입점과 인증·라벨링·지속 모니터링 로드맵을 제안했다.
- 예시: 사례 1은 질병위험 예측 중 부수소견 통보와 친족 프라이버시 침해, 사례 2는 GDPR·HIPAA·중국 규제 충돌로 인한 데이터/모델 이동 제약, 사례 3은 공개 가중치에서 민감 유전정보 추출 위험을 다루며, 대응책으로 샌드박스 사용제한·연합학습, 동형암호·PRS 프라이버시 계산, GLM 전용 인증 기준을 제시한다.

- Methods: The authors analyze regulatory gaps using three fictional GenePredictAI scenarios, a comparative review of AI and genomic rules, and a multilayered governance design combining regulatory sandboxes, certification, capacity hubs, and soft governance with technical tools (privacy-preserving methods including homomorphic encryption, embeddings, federated learning, and interpretability such as attention visualization, LIME/SHAP, and in silico mutagenesis).
- Results: They map misalignments across data privacy/consent, bias/fairness, interpretability, and jurisdiction/liability that create GLM-specific risks (re-identification, unintended kinship disclosure, cross-border compliance, and open-source leakage) and propose lifecycle intervention points plus certification/labeling and continuous monitoring to mitigate them (Table 2).
- Examples: Scenario 1 covers incidental findings and familial privacy in disease-risk prediction; Scenario 2 shows GDPR–HIPAA–China conflicts constraining data/model flows; Scenario 3 highlights extractability of sensitive information from released weights; countermeasures include sandbox use restrictions and federated learning, homomorphic encryption/PRS privacy computation, and GLM-specific certification standards.

<br/>
# 기타


1) 약어 모음(Abbreviations)
- 결과: AI·유전체 규제·보건의료·표준기구·설명가능성 등 전 분야의 약어(예: EU AI Act, GDPR, GINA, HIPAA, GA4GH, OECD, WHO, LIME/SHAP/PII/LLM/GLM 등)를 한데 정리.
- 인사이트: GLM 거버넌스는 AI와 유전체 규범, 개인정보·보건·국제표준이 중첩되는 광범위한 규제지형을 포괄하며, 공통 용어·약어 정리가 협력과 상호운용성의 전제임을 시사.

2) 표 1: GLM 적용 시 규제 긴장요인(가상의 GenePredictAI 시나리오 요약)
- 결과:
  - 1a(질병위험의 우연 발견): AI 규제는 혁신 허용에 무게, 유전체 규범은 ‘우연소견 통지’ 의무 불명확.
  - 1b(훈련데이터와 유사성 노출): 개인 선택 vs “친족에게 미치는 유전정보의 프라이버시” 간 충돌.
  - 2(국경 간 공동개발): EU의 전송제한·목적제한, 중국의 보안심사, 미국·인도의 상이한 규정이 모델 학습·공유를 저해.
  - 3(오픈소스 공개): 개방은 투명성과 재현성 확보에 유리하나, 모델 가중치에 민감 유전정보가 ‘잠재적으로 새어 나올’ 위험.
- 인사이트: GLM은 AI의 개방성·접근성 원칙과 유전체의 엄격한 동의·프라이버시 원칙 사이의 간극을 실무적으로 드러냄. 이를 해소하려면 규제 샌드박스, 인증·책임 배분, 국경 간 조율이 결합된 특화 프레임이 필요.

3) 표 2: GLM의 핵심 규제 대비(프라이버시, 동의, 편향, 설명가능성, 관할, 책임)
- 결과: 
  - 프라이버시: 익명화에 의존하는 AI 관행 vs 유전정보는 본질적으로 재식별 가능.
  - 동의: 광범위 동의(AI) vs 목적한정·구체적 동의(유전체). GLM의 생성적 추론이 원동의 범위를 초과.
  - 편향: 관측가능 변수 중심(AI) vs 데이터 편중(유럽계 과대표집 등)·생물학적 얽힘(유전체) 증폭.
  - 설명가능성: 일부 영역에서 블랙박스 허용(AI) vs 임상의 신뢰·조치가능성을 위해 높은 설명성(유전체) 요구.
  - 관할/책임: 클라우드·국경 간 처리로 규제 충돌, 개발사·의료제공자·기관 간 책임귀속 불명확.
- 인사이트: GLM은 ‘AI 규제도 유전체 규제도 단독으로는 부족’한 영역을 형성. 추론능력이 익명화와 광범위 동의를 무력화하고, 임상 수용성·책임배분의 새로운 기준을 요구.

4) 그림 1: GLM 개요(토큰화, 인코더/디코더, 멀티모달, 사전학습과 전이)
- 결과: DNA를 k-mer 등 토큰으로 처리, BERT형(마스킹 복원)·GPT형(다음 토큰 예측)·멀티모달 모델이 대규모 비지도 사전학습 후 프로모터/변이효과/질병위험 예측 등 다양한 다운스트림 과제에 적용.
- 인사이트: GLM은 ‘파운데이션 모델’로서 국소·장거리 상호작용을 함께 포착해 기존 정렬/모티프 탐지로 놓치던 복잡 패턴을 학습. 동시에 블랙박스성으로 해석·검증 과제가 커짐.

5) 그림 2: GLM 생애주기별 규제 개입 지점
- 결과: 데이터 수집→사전학습→미세조정→배포→현장사용까지 단계별로 데이터 거버넌스, 프라이버시·보안, 편향감사, 설명성 검증, 모니터링 등 개입 포인트 정렬.
- 인사이트: “언제, 무엇을, 누가” 감독할지의 로드맵을 제시. 현행 규제가 놓치는 구간(예: 사전학습 데이터 투명성, 오픈배포 전 추출 가능성 평가 등)에 맞춤형 점검 필요.

6) 그림 3: 다층 거버넌스 틀(기술–제도–정책의 상호작용)
- 결과: 
  - 기술층: 프라이버시 보존(임베딩/연합학습/암호화/스코어링), 설명가능성(어텐션 시각화·기여도·섭동/절제 실험) 등.
  - 제도층: 인증·감사, 공통역량허브(ARCCH 유형), 사건보고/피드백 루프.
  - 정책층: 규제 샌드박스, 소프트 거버넌스(가이드라인·모범규범), 필요시 공동규제.
- 인사이트: 기술적 안전장치와 제도적 역량·감사, 정책적 유연성(샌드박스)을 결합해야 실효성 확보. 피드백 루프로 지속 개선·국제 정합성 도모.

7) 그림 4: 프라이버시 보존 기법 스펙트럼과 효용-보안 트레이드오프
- 결과: 원시서열(효용↑/프라이버시↓)→임베딩/변환(중간)→프라이버시 보존 스코어링(PRS 등)→연합학습(데이터 국지 보유)→준동형암호(프라이버시↑/계산비용↑)로 갈수록 보안↑·효용/성능·비용의 상충 존재.
- 인사이트: 단일 해법은 없으며, 사용처·위험도·자원에 맞는 ‘혼합 설계’가 현실적. 공개·배포 전 추출가능성 평가와 함께 임베딩/연합학습을 표준 옵션으로 삼고, 고위험 사용은 암호화/격리 강화를 권장.

8) 용어집(Technical terms) — 사실상 부록 성격
- 결과: 어텐션, 마스크드 LM, 포스트혹 설명, 폴리제닉 리스크 스코어, 연합학습, 준동형암호, 규제 샌드박스 등 핵심 개념 정의.
- 인사이트: 기술–정책–임상 간 오해를 줄이고 인증·표준 수립의 공통 기반을 제공. 다기관·국경 간 협업의 용어 정렬에 기여.

요약 인사이트(종합)
- GLM은 AI의 개방성·확장성과 유전체의 프라이버시·동의 중심 원칙 사이의 구조적 간극을 ‘실무 시나리오’에서 선명히 드러낸다.
- 이를 메우려면 생애주기 전반의 개입 포인트 설정(그림 2), 다층 거버넌스(그림 3), 프라이버시 보존 기술의 합리적 조합(그림 4), 인증/감사·샌드박스·역량 허브(표 1·2, 그림 3)가 함께 작동해야 한다.
- 임상 신뢰·책임배분·국경 간 규정 정합성은 향후 채택의 성패를 가를 핵심 축이다.




1) Abbreviations
- Result: Consolidates key acronyms across AI, genomics, health privacy, standards, explainability (e.g., EU AI Act, GDPR, GINA, HIPAA, GA4GH, OECD, WHO, LIME/SHAP/PII/LLM/GLM).
- Insight: Signals that GLM governance spans overlapping AI and genomic regimes; shared vocabulary is foundational for collaboration and interoperability.

2) Table 1: Regulatory tensions (GenePredictAI scenarios)
- Result:
  - 1a (incidental disease-risk findings): AI rules tend to permit novel insights; genomic regimes lack clear duties for returning incidental findings.
  - 1b (similarity to training data): User choice vs genetic privacy extending to relatives—consent boundaries can be breached.
  - 2 (cross-border collaboration): GDPR transfer limits and Chinese security reviews clash with U.S./India practices, hindering training/data sharing.
  - 3 (open-source release): Public weights may leak sensitive genomic info despite openness benefits.
- Insight: GLMs crystallize accountability and consent dilemmas; targeted fixes include regulatory sandboxes, certification/liability allocation, and cross-border coordination.

3) Table 2: Key regulatory contrasts for GLMs
- Result: 
  - Privacy: AI’s anonymization paradigm vs genomics’ re-identifiability.
  - Consent: Broad consent (AI) vs purpose-limited explicit consent (genomics); GLM inference exceeds original scope.
  - Bias: Observable proxies (AI) vs deeply embedded/biologically entangled biases (genomics).
  - Explainability: Some tolerance for black boxes (AI) vs high explainability for clinical trust (genomics).
  - Jurisdiction/Liability: Cloud/global models complicate compliance; responsibility among developers/providers/institutions unclear.
- Insight: GLMs occupy a governance gap where AI-only or genomics-only frameworks are insufficient; inference power undermines anonymization and broad consent, and clinical liability needs rethinking.

4) Figure 1: GLM schematic (tokenization, encoder/decoder, multimodality, pretrain→downstream)
- Result: DNA is tokenized (e.g., k-mers); encoder (masked LM) and decoder (next-token) models, including multimodal, pretrain on large unlabeled genomes and transfer to tasks (promoters, variant effects, disease risk).
- Insight: As foundation models, GLMs capture local and long-range interactions beyond classic alignment/motif tools, but their black-box nature raises interpretation and audit challenges.

5) Figure 2: Lifecycle-aligned regulatory intervention points
- Result: Maps data collection, pretraining, fine-tuning, release, and deployment to oversight checkpoints (data governance, privacy/security, bias audits, explainability tests, monitoring).
- Insight: Provides a “when-what-who” roadmap; highlights neglected stages (e.g., pretraining data transparency, extractability checks prior to release).

6) Figure 3: Multilayered governance (technical–institutional–policy)
- Result:
  - Technical: Privacy-preserving methods (embeddings/federation/encryption/scoring); explainability (attention visualization, attributions, perturbation/ablation).
  - Institutional: Certification/audits, shared capacity hubs (ARCCH-like), incident reporting and feedback loops.
  - Policy: Regulatory sandboxes, soft governance (guidelines/codes), and co-regulation where needed.
- Insight: Effective oversight demands co-evolution of safeguards, institutional capacity, and flexible policy tools, tied together by feedback loops and international alignment.

7) Figure 4: Spectrum of privacy-preserving techniques and trade-offs
- Result: From raw sequences (utility↑/privacy↓) to embeddings/transformations (mid), privacy-preserving scoring (e.g., PRS), federated learning (local data), and homomorphic encryption (privacy↑/compute cost↑)—clear privacy–utility–cost trade-offs.
- Insight: No silver bullet; mix methods by use case and risk. Prioritize extractability assessment pre-release, standardize embeddings/federation for common cases, and apply encryption/segregation for high-risk contexts.

8) Technical terms (glossary) — effectively an appendix
- Result: Defines core concepts (attention, masked LM, post-hoc explanations, PRS, federated learning, homomorphic encryption, regulatory sandbox, etc.).
- Insight: Reduces ambiguity across technical/policy/clinical stakeholders and underpins certification/standards; essential for multi-institutional and cross-border work.

Overall insights
- GLMs expose the practical rift between AI’s openness/scale and genomics’ privacy/consent, especially in real-world scenarios.
- Closing the gap requires lifecycle checkpoints (Fig. 2), multilayered governance (Fig. 3), pragmatic privacy-tech combinations (Fig. 4), and purpose-built certification/sandboxes/capacity hubs (Tables 1–2, Fig. 3).
- Clinical trust, liability allocation, and cross-border coherence are decisive for safe, beneficial adoption.

<br/>
# refer format:


BibTeX
@article{Sokhansanj2025RegulatingGLMs,
  author = {Sokhansanj, Bahrad A. and Rosen, Gail L.},
  title = {Regulating genome language models: navigating policy challenges at the intersection of {AI} and genetics},
  journal = {Human Genetics},
  year = {2025},
  doi = {10.1007/s00439-025-02768-4},
  url = {https://doi.org/10.1007/s00439-025-02768-4},
  note = {Advance online publication}
}

시카고(Author–Date)
Sokhansanj, Bahrad A., and Gail L. Rosen. 2025. “Regulating genome language models: navigating policy challenges at the intersection of AI and genetics.” Human Genetics. https://doi.org/10.1007/s00439-025-02768-4.
