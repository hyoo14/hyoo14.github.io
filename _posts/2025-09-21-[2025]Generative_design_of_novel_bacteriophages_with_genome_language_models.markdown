---
layout: post
title:  "[2025]Generative design of novel bacteriophages with genome language models"
date:   2025-09-21 21:42:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: evo1,2 이용하여 유전체 설계 및 생성, 실제 파지에 도입해서 실험 (Twist Bioscience 라는 회사에서 dna합성해줌, 2억4천만원정도 든다네..300개에.. ㅎㄷㄷ  )   


짧은 요약(Abstract) :


이 연구는 인공지능 기반 ‘유전체 언어 모델’(Evo 1, Evo 2)을 이용해 처음으로 전체 길이의 박테리오파지(세균 감염 바이러스) 유전체를 설계·생성하고, 실제로 기능하는 파지를 얻은 사례를 보고합니다. 연구팀은 E. coli를 숙주로 하는 소형 리틱 파지 ΦX174를 설계 템플릿으로 삼아, 유전자 배열·조절요소·인식서열 등 복잡한 유전체 구조를 유지하면서도 목표 숙주 특이성(트로피즘)을 갖춘 새로운 유전체 서열을 생성했습니다. 실험적으로 약 300개의 설계안을 시험해 16개의 생존 가능(viable) 파지를 확보했으며, 이들은 자연계와 구별될 만큼 진화적 새로움(서열·구조적 차이)을 보였습니다. 크라이오-EM 분석에서는 그중 하나가 캡시드 내부에 진화적으로 먼 DNA 패키징 단백질을 활용함을 확인했습니다. 여러 생성 파지는 ΦX174보다 더 빠른 용균(lysis) 동역학을 보이거나 공동배양 경쟁에서 더 높은 적합도(fitness)를 나타냈습니다. 또한 생성 파지 칵테일은 ΦX174에 내성을 획득한 서로 다른 E. coli 3개 균주에 대해 신속히 내성을 돌파했지만, ΦX174 단독으로는 실패했습니다. 이 결과는 유전체 규모에서 조절 가능한 제약(숙주 범위 등)을 걸고 기능적 생물 시스템을 생성하는 ‘생성 게놈 설계’가 가능함을 보여주며, 합성 파지 및 더 복잡한 생명 시스템 설계를 위한 기반을 제공합니다.


This work demonstrates the first generative design of complete, functional bacteriophage genomes using genome language models (Evo 1 and Evo 2). Using the lytic phage ΦX174 as a template, the authors generated whole-genome sequences that preserve realistic genetic architecture and enforce target host tropism. From roughly 300 AI-designed candidates, 16 viable phages were experimentally validated, exhibiting substantial evolutionary novelty in sequence and structure. Cryo-EM revealed that one designed phage incorporates an evolutionarily distant DNA-packaging protein within its capsid. Several designs outperform ΦX174 in head-to-head growth competitions and show faster lysis kinetics. A cocktail of the designed phages rapidly overcame ΦX174 resistance in three distinct E. coli strains, whereas ΦX174 alone could not. Overall, the study provides a blueprint for steerable, genome-scale generative design of synthetic bacteriophages and lays groundwork for designing more complex living systems.


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





1) 사용한 모델과 아키텍처
- Evo 1, Evo 2: 유전체 언어 모델(generative genome language models)
  - 공통점: 대규모 게놈 서열로 사전학습한 자가회귀(autoregressive) 생성 모델. 코드 베이스는 Transformer의 kv-caching과 Hyena 계열(long-range 서열모델링)의 재귀(recurrent) 구현을 모두 지원하는 샘플러를 사용.
  - 차이(운영 관점): Evo 1(7B 131K), Evo 2(7B 8K) 체크포인트를 사용. Evo 2는 최신(‘Evo 2’) 스택을 기반으로 하며 긴 컨텍스트와 효율적 샘플링을 지원.
- 사전학습(pretraining)
  - 저자들의 이전 작업(OpenGenome 등)에 기반한 대규모 유전체 코퍼스(바이러스 포함, 박테리오파지 수백만 규모 포함)로 사전학습되어, DNA 수준에서 문법·문맥·장거리 제약을 학습.
  - 안전성 관련: 인간·동물 등 진핵 바이러스 데이터 제외(이전 연구에서 검증됨). 본 연구에서는 세균 파지(Microviridae)에 더욱 특화된 파인튜닝을 추가 수행.

2) 미세조정(Supervised Fine-Tuning, SFT)
- 목적: ΦX174(마이크로바이러스과, Microviridae) 계통 유전체를 더 높은 충실도로 생성하도록 전문화.
- 데이터 구축(총 15,507개 확보 후 필터링·중복제거):
  - NCBI Datasets(키워드 ‘Microviridae’, complete), PhageScope(quality: Complete), OpenGenome에서 수집.
  - 길이 >10 kb, 비정규 문자가 포함된 시퀀스 제거 후 14,466개 최종 사용.
  - ΦX174 유사도 조건부 토큰(soft prompt) 부여: 모든 입력 앞에 “+”(Microviridae 표시), 그리고 ΦX174와의 정렬 기반 유사도 구간별로 “∼, ^, #, $, !” 토큰을 추가해 조건화(95–100%, 80–95%, 70–80%, 50–70%, <50% ANI). 실제 최종 생성 시에는 “+∼”만 사용.
  - 학습/검증/테스트 분할: 14,266 / 100 / 100
- 하이퍼파라미터(원문 Methods의 수치)
  - Evo 1 SFT(7B 131K): 16×H100, 5,000 스텝, 배치 64, 컨텍스트 10,240, LR=9.698e−5(5% 워밍업 후 코사인 감쇠, 최저 3e−5)
  - Evo 2 SFT(7B 8K): 32×H100, 12,000 스텝, 배치 32, 컨텍스트 10,240, LR=1e−5(5% 워밍업 후 코사인 감쇠, 최저 1e−6)
- 효과: ΦX174 변이들의 위치별 엔트로피가 크게 개선되어(사전학습 대비) 문맥·문법적 제약을 더 정확히 반영.

3) 생성(프롬프팅·샘플링 전략)
- 제로샷 벤치마크(바이러스 realm 프롬프트)
  - 프롬프트(예): |r__Duplodnaviria;k__, |r__Monodnaviria;k__, |r__Riboviria;k__ (Evo 1), Evo 2는 세미콜론 포맷 사용. 길이 6,000, T=0.7, top-k=4, top-p=1로 대량 샘플링(모델별/realm별 약 1–10k).
  - 외부 판별기(geNomad)로 바이러스성 판정, BLAST로 자연계 유사도 측정 등.
- 표적 설계 프롬프트(ΦX174 계열)
  - ΦX174 변이들의 시작부 컨센서스 핵산을 MSA로 추출, 이 앞부분을 프롬프트로 사용.
  - SFT 모델은 4–9 nt 길이의 컨센서스-프롬프트와 적정 온도(T=0.7–0.9)에서 ΦX174-like이면서도 다양한 시퀀스를 생성. 프롬프트가 너무 길면 기억회상(recall) 위주, 너무 짧으면 조건화 실패.
  - 베이스(사전학습) 모델만으로는 해당 컨센서스 프롬프트 증가에도 ΦX174 recall 실패.

4) 생성물 필터링 및 스크리닝(추론 시 제약·유도)
A. 기본 품질 제약(QC)
  - 뉴클레오타이드(A/C/G/T)만 허용
  - 길이 4–6 kb
  - GC 30–65%
  - 동일염기 연속 길이 ≤10
  - ΦX174 단백질 최소 7개 이상과 유의미한 유사 단백질 히트(아래 자체 주석 파이프라인로 예측)
B. 숙주 특이성(트로피즘) 제약
  - 스파이크 단백질(ΦX174의 G 단백질) 서열 유사도 ≥60%를 요구(MMseqs2 정렬). 수용체 결합 결정성에 착안한 간접적 필터로, E. coli C 타깃 특이성 확보.
C. 다양화(진화적 새로움) 제약
  - 평균 아미노산 서열 동일성(AAI) <95% vs 자연계
  - 유전적 아키텍처 유사도 점수 ≤0.9(아래 설명)
  - ΦX174 대비 시튼티니(synteny) 1개 유전자 단절(단일 유전자 삭감 또는 추가 허용)
  - 총 유전자 수 10 또는 12 허용(ΦX174는 11)

5) 자체 개발 도구와 보조 모델
- 겹침 ORF 예측 문제 해결을 위한 자체 유전자 주석 파이프라인
  - 동기: 기존 6개 도구(Prodigal, Pyrodigal-gv, PHANOTATE, pLannotate, Glimmer, GeneMark) 모두 ΦX174의 11개 유전자를 완전 검출 실패. 겹침 유전자(ORF) 탐지의 난점.
  - 파이프라인:
    1) Pseudo-circularization: 각 리딩프레임에서 첫 스톱코돈들의 가장 하류 위치까지 잘라서 말단에 이어붙여 링형 게놈의 접합부 문제를 완화.
    2) orfipy로 ORF 전수 추출(시작 ATG, 스톱 TAA/TAG/TGA).
    3) PHROGs 단백질 DB에 전대 전탐색(MMseqs2, sensitivity 4.0), 각 ORF의 최상 히트(E-value 최소)로 기능 주석.
  - 성능: ΦX174에서 10/11 유전자를 완전 검출(예외: A*는 부분 검출), 이후 모든 생성물의 단백질 히트 수 산출·필터링에 사용.
- 유전적 아키텍처(ORF 배열) 유사도 점수화
  - 아이디어: ΦX174의 시작·종결 코돈 위치를 원-핫 인코딩한 ‘경계 벡터’를 만들고, 생성물의 전순환(circular shift) 경우의 수 중 최적 일치를 빠르게 점수화.
  - 구현:
    - 참조(ΦX174) 경계 벡터에 가우시안 블러(σ=5)를 적용해 경계 오차에 강건.
    - 쿼리(생성물) 경계 벡터와 참조 간의 내적을 최대화한 값을 가중 정규화하여 점수 산출. ΦX174 자기 점수를 1로 정규화.
  - 기준: ΦX174-like 분류 경계(>0.38)와 다양화 필터 경계(≤0.9)를 설정해 활용.
  - 보조 시각화: 아키텍처 원-핫 벡터의 UMAP 투영으로 Microviridae의 구조적 군집과 ΦX174 스파이크 유사도 연계 확인.
- 외부 보조 모델·도구(생물학적 타당성/새로움 평가)
  - geNomad: 바이러스/모바일 유전요소 판별(바이러스성 스코어)
  - nucleotide BLAST: 자연계 대비 BLAST 히트(정체성·커버리지)
  - Prodigal: 코딩 밀도(통제군 대비)
  - ESMFold: 예측 구조(pLDDT/pTM)로 단백질 접힘 타당성 확인
  - MMseqs2 + PHROGs/ OpenGenome: 단백질 유사도·기능 라벨 분포
  - CheckV: 바이러스 서열 완전도·품질

6) 생성 파라미터 스윕과 다양성–품질의 균형
- 온도(T) × 프롬프트 길이 스윕
  - T: 0.3/0.5/0.7/0.9/1.1, 프롬프트: 1–11 nt(ΦX174 컨센서스 시작부)
  - 결과: 4–9 nt, T=0.7–0.9 구간에서 트로피즘 필터 통과 후에도 Shannon 다양도가 높고, 최종 필터(다양화) 잔존율도 높음.
  - 프롬프트 과도 시 단순 회상, 과소 시 조건화 실패 → 최적대역 필요.
- 유지율(retention rate) 분석
  - 품질→트로피즘→다양화 순 적용 시 잔존율을 파라미터 조합별 비교해, 모델·샘플링 설정을 튜닝.

7) 추가 분석(생성물 새로움 정량)
- 누적 게놈 귀속(cumulative attribution): BLAST 상위 히트들을 누적 매칭해도 자연계에서 모두 설명되지 않는 염기 구간이 다수 존재(새로움 확인).
- 변이 히트맵: 유전자·프로모터·터미네이터별 평균 변이율 산출해 핫스팟 파악(예: promoter A, terminator H, gene J가 많이 변이).

8) 재현성·코드
- SFT 체크포인트(Evo 1·Evo 2 Microviridae), 데이터(정제 Microviridae), 설계·분석 코드(아키텍처 점수/주석 파이프라인/샘플러) 공개(논문 본문 “Data and code availability”).




1) Models and architectures
- Evo 1 and Evo 2: Generative genome language models
  - Both are autoregressive sequence models pretrained on massive genomic corpora (including millions of bacteriophage genomes), implemented with samplers that support Transformer kv-caching and recurrent Hyena-style operators for long-range DNA modeling.
  - We used Evo 1 7B 131K and Evo 2 7B 8K checkpoints; Evo 2 provides efficient long-context generation.

2) Supervised fine-tuning (SFT) on Microviridae
- Goal: Specialize the pretrained models to ΦX174-like Microviridae genomes.
- Data curation:
  - 15,507 Microviridae genomes collected from NCBI Datasets (Complete), PhageScope (Complete), and OpenGenome; sequences >10 kb or with non-ACGT characters removed; MMseqs2 clustering at 99% produced 14,466 final sequences.
  - Soft-conditioning tokens: prepend “+” (Microviridae) and one of “∼, ^, #, $, !” encoding ΦX174 identity bins (95–100%, 80–95%, 70–80%, 50–70%, <50%). For final generation, we only prepended “+∼”.
  - Split: 14,266 train / 100 val / 100 test.
- Hyperparameters (from Methods):
  - Evo 1 SFT: 16×H100, 5,000 steps, batch 64, context 10,240, LR 9.698e−5 with 5% linear warm-up then cosine decay to 3e−5.
  - Evo 2 SFT: 32×H100, 12,000 steps, batch 32, context 10,240, LR 1e−5 with 5% warm-up then cosine decay to 1e−6.
- Effect: markedly improved per-position entropy on ΦX174 variants versus base models, indicating better modeling of genome-scale constraints.

3) Generation: prompting and sampling
- Zero-shot realm prompts:
  - Evo 1 prompts like |r__Duplodnaviria;k__, |r__Monodnaviria;k__, |r__Riboviria;k__; Evo 2 uses analogous formats. We sampled length 6,000 with T=0.7, top-k=4, top-p=1, per realm and model.
  - Validated with geNomad (viral scores), BLAST novelty, coding density, and protein structure proxies.
- ΦX174-guided prompting:
  - Multiple-sequence alignment of ΦX174 variants yielded a consensus start sequence; we prompted with its first 1–11 nt.
  - SFT models recalled and diversified ΦX174-like genomes with prompt lengths 4–9 nt and T=0.7–0.9. Base models failed to recall even with longer prompts.

4) Inference-time constraints and screening
A. Quality control
  - A/C/G/T only, length 4–6 kb, GC 30–65%, homopolymer ≤10 nt
  - At least 7 predicted protein hits to ΦX174 (via our custom annotation pipeline below)
B. Host tropism constraint
  - Spike (G) protein identity ≥60% to ΦX174 spike (MMseqs2), reflecting receptor-binding specificity toward E. coli C.
C. Diversification constraints
  - Mean AAI <95% to natural proteins
  - Genetic architecture similarity score ≤0.9 (method below)
  - Exactly one synteny break relative to ΦX174; total gene count 10 or 12 accepted

5) Custom tools and auxiliary models
- Bespoke gene annotation for ΦX174-like sequences (overlapping ORFs)
  - Motivation: Six mainstream callers (Prodigal, Pyrodigal-gv, PHANOTATE, pLannotate, Glimmer, GeneMark) failed to fully recover all 11 ΦX174 genes.
  - Pipeline:
    1) Pseudo-circularization to handle circular genome boundaries.
    2) Exhaustive ORF discovery with orfipy (start ATG; stop TAA/TAG/TGA).
    3) All-vs-all search against PHROGs using MMseqs2 (sens=4.0), keeping best hit per ORF.
  - Performance: fully recovers 10/11 genes on ΦX174 (A* partially); used to count ΦX174-protein hits in QC.
- Genetic architecture similarity scoring (fast ORF-boundary pattern matching)
  - One-hot encode start/stop codon positions; apply Gaussian blur (σ=5) to ΦX174 boundary vector.
  - Compute the maximum dot-product match across all circular shifts of the query; normalize to ΦX174 self-score (=1).
  - Empirically, >0.38 tags ΦX174-like architectures; ≤0.9 used to enforce diversification.
  - UMAP was used to visualize architectural clusters across Microviridae.
- Auxiliary evaluation tools
  - geNomad (viral classification), BLAST (nucleotide similarity), Prodigal (coding density), ESMFold (pLDDT/pTM), MMseqs2 vs PHROGs/OpenGenome (protein identity/function), CheckV (viral quality).

6) Parameter sweeps and diversity-quality tradeoff
- Temperature × prompt-length sweeps
  - T in {0.3, 0.5, 0.7, 0.9, 1.1}; prompt length 1–11 nt (consensus start).
  - Best region: 4–9 nt with T=0.7–0.9 achieves high Shannon diversity after tropism filtering and high retention after diversification filtering.
  - Too-long prompts cause recall with low novelty; too-short prompts fail to condition.
- Retention rate tracking
  - We measured pass rates after QC → tropism → diversification per parameter setting to tune generation.

7) Novelty quantification and mutation profiling
- Cumulative genome attribution: a greedy BLAST-based assignment of generated nucleotides to natural hits shows many positions remain unassigned, supporting genuine novelty.
- Mutation hotspots: BLAST-short alignments of ΦX174 genes/regulatory elements against generated genomes quantify per-element mutation rates; promoter A, terminator H, and gene J are hotspots.

8) Reproducibility and code
- We release SFT checkpoints (Evo 1/2 Microviridae), processed Microviridae data, and code for generation, filtering, annotation, and architecture scoring (see paper’s Data and code availability).

참고: 위 요약은 논문 본문(Results 섹션의 그림 1–2)과 Methods(특히 B.1.1–B.1.16, B.3 일부) 및 보조 분석에 기초해, 모델·아키텍처·데이터·추론시간 제약·자체 기법에 초점을 맞춰 정리했습니다. 실험적 조립·증식·선발 등 습식 실험은 범위에서 제외했습니다.


<br/>
# Results



- 전체 게놈 규모에서 기능적 박테리오파지(ΦX174 유사)를 생성하는 AI 설계 파이프라인을 확립하고, 생성물의 생물학적 타당성(감염성, 용균력, 서열/구조적 참신성, 적응도, 내성 극복 능력)을 체계적으로 검증.

경쟁모델·비교대상
- 생성 모델: Evo 1, Evo 2(대규모 게놈 언어모델, 사전학습 포함). Supervised Fine-Tuning(SFT)을 통해 Microviridae 전용 모델로 특화(Evo 1 7B 131K SFT, Evo 2 7B 8K SFT).
- 베이스라인/대조:
  - 자연 서열: NCBI/OpenGenome/PHROGs에서 수집된 자연 파지 서열(바이러스 분류 realm별: Duplodnaviria/Monodnaviria/Riboviria).
  - 스크램블(무작위 재배열) 서열: 염기 조성만 보존한 음성 대조.
  - 베이스 모델 대비 SFT 모델: ΦX174 리콜(프롬프트 길이별)과 생성 품질 비교.
  - Microviridae 자연군·스크램블·ΦX174 변이군을 동일 필터에 통과시켜 생성물과 정량 비교.

데이터셋(학습/평가)
- 사전학습: OpenGenome 등 대규모 게놈 코퍼스(>200만 파지 포함).
- SFT 학습: Microviridae 약 15,000개 게놈(길이 4–6 kb 범주 중심), ΦX174 변이군 134개 포함. 학습/검증/테스트 분할(14,266/100/100).
- 제로샷 프롬프트 평가: 바이러스 realm 토큰 프롬프트로 6,000nt 길이 시퀀스 다수 생성(자연·스크램블과 비교).

모델링·프롬프트 조건
- 제로샷(사전학습 모델): realm 프롬프트로 온도 0.7, top-k=4, top-p=1.
- SFT 모델: ΦX174 시작부 컨센서스 1–11nt로 프롬프트, 온도 0.3–1.1 스윕. 최적 대역: 프롬프트 4–9nt, 온도 0.7–0.9에서 다양성과 품질 동시 달성.

유전 서열 제약조건(필터)과 설계 지표
- 3단계 필터
  1) 품질: A/C/G/T만 포함, 길이 4–6 kb, GC 30–65%, 동일 염기 10nt 이하, 맞춤형 CDS 예측으로 ΦX174 단백질 ≥7개 히트.
  2) 표적 기주 범위(트로피즘): ΦX174 스파이크 단백질과 ≥60% 서열 유사성.
  3) 다양화(옵션): 평균 AAI <95%(자연 단백질 대비), 유전 아키텍처 유사도 ≤0.9, ΦX174 대비 유전자 시턴티 1개 붕괴, 총 유전자 수 10/12 선호.
- 보조 품질 진단: CheckV(High Quality/Complete 판정), geNomad(바이러스 분류 점수), BLAST(ANI/커버리지), 유전 아키텍처 점수(원핫 start/stop 기반), 구조예측(ESMFold/AlphaFold3).

유전자 예측 도구 벤치마킹(참고)
- Prodigal, Pyrodigal-gv, PHANOTATE, pLannotate, Glimmer, GeneMark 모두 ΦX174의 11개 중첩 유전자를 완전 주석화하지 못함(특히 중첩 ORF 취약).
- 맞춤형 CDS/아노테이션 파이프라인을 개발해 ΦX174 전 유전자(부분적으로 A* 제외) 회복 → 생성물 필터의 신뢰도 확보.

1) 제로샷 생성 성능(사전학습 모델) – 경쟁모델 비교
- 바이러스 분류(geNomad):
  - 자연 서열: 89–100% 바이러스 판정(평균 점수 >0.96).
  - 스크램블: 1.9–11%(평균 점수 >0.73).
  - Evo 1: 19–33%(평균 >0.80), Evo 2: 34–38%(평균 >0.87).
  → Evo 2가 Evo 1보다 일관되게 우수. 스크램블 대비 뚜렷이 높고, 자연 보다는 낮음.
- BLAST 유사도: 낮은 커버리지/정체성 → 높은 참신성.
- 코딩 밀도: 자연 파지와 유사(스크램블 대비 높음).
- 단백질 구조품질(ESMFold): 생성 단백질 pLDDT 평균이 자연 단백질 수준, 스크램블보다 높음.
- 단백질 유사도/주석: OpenGenome/PHROGs 대비 낮은 서열 정체성 분포(참신성)와 자연 파지와 유사한 기능 범주 프로파일.

2) SFT 후 조건부 생성 성능 – 베이스 vs SFT 비교
- ΦX174 리콜(컨센서스 시작 염기 프롬프트 길이 증가 시):
  - SFT 모델(Evo 1/2)은 프롬프트 길이가 늘수록 ΦX174 회상률 급증.
  - 베이스 모델은 어떤 길이에서도 ΦX174 회상 실패.
- 위치 엔트로피(언어모델 불확실성): SFT 후 ΦX174 변이군에 대한 위치별 엔트로피가 실제 유전 특징에 더 잘 정렬(학습 수렴 정량 지표).

3) 생성 파라미터(온도·프롬프트) 스윕 – 다양성과 품질 균형
- 다양성(Shannon entropy): 온도와 프롬프트 길이 증가에 따라 증가.
- 품질/통과율(설계 필터 retention): 최적 대역(온도 0.7–0.9, 프롬프트 4–9nt)에서 높은 다양성과 높은 필터 통과를 동시 달성.
- 최종 통과율(예시): 트로피즘 필터 후 Evo 1 SFT 최대 100%, Evo 2 SFT 53.8%; 다양화 필터까지 후 Evo 1 10.4%, Evo 2 17.2%.

4) 최종 후보군 합성·실험 검증(테스트 데이터 및 메트릭)
- 설계 후보 302개 선정 → 합성 성공 285개 → E. coli C에서 “게놈 리부팅” 성장억제 스크리닝으로 16개 생존 파지 획득.
- 서열 품질: CheckV로 87%+가 High Quality/Complete.
- 스파이크 유사성: 대부분 ≥85% 유지(트로피즘 통제 타당).
- ΦX174 및 Microviridae 대비 유사도: 대체로 40%+ 뉴클레오타이드 정체성(템플릿 가이드)과 함께 낮은 AAI(최저 63%)로 단백질 수준 참신성.
- 기주 범위: 8개 E. coli 균주 테스트에서 E. coli C, W에서 성장 억제(ΦX174 포함 16/17), K-12 계열 등 6개 균주에선 억제 없음 → 표적 트로피즘 필터의 특이성 실증.
- 타이틀/플라크: 파지별로 티터 가변적(정량 도말 스폿플레이트).

5) 염기서열·계통·구조적 참신성(비교 및 지표)
- 변이 규모: 생성 16개는 자연 최인접군 대비 67–392개의 신규 변이(ANI 93.0–98.8%). Evo-Φ2147은 93.0% ANI(NC51 대비, 392변이)로 자연 종 경계(ANI<95%) 수준의 참신성.
- 유전 아키텍처 변화: 삽입/결실/연장/단축/유전자 교체(예: Evo-Φ36의 J 유전자 G4 호몰로그로 스왑) 포함. 중복·비부호 영역 확장 사례 다수.
- “자연에서 설명 불가” 돌연변이: 16개 중 13개에서 누적 BLAST 귀속으로도 자연변이 조합으로 완전 설명 불가 영역 존재(특히 비부호 확장 구간).
- 구조(크라이오-EM): ΦX174(2.8 Å), Evo-Φ36(2.9 Å) 고해상도 맵. Evo-Φ36은 G4형 짧은 J(25 aa)를 사용하면서도 캡시드 결합 양상/유전자 포장 적합성을 달성(ΦX174 J 38 aa와 상이). N-말단 일부 비정형/미해결 → 새로운 단백질–단백질 공진화 해법.

6) 적응도 경쟁(동일 조건 head-to-head) – 메트릭 및 비교
- 설정: 16개 생성 파지+ΦX174를 동일 MOI로 E. coli C에 동시 감염. 시퀀싱 리드로 파지별 시간 경과 읽힘수 추적.
- 핵심 메트릭:
  - 누적 Fold Change(FC): 각 시간점 리드수 증감의 누적(log2(FC) 시계열).
  - AUC(시간-적분 면적): 전체 실험기간 종합 성과.
- 결과(3회 독립 경쟁):
  - Evo-Φ69가 3회 모두 1위(6h 누적 FC 16×–65×). ΦX174는 동일 시점 1.3×–4.0×.
  - Top performer 반복 등장: Evo-Φ69, Evo-Φ100, Evo-Φ111. ΦX174는 최대 3위(경쟁 1·3) 수준.
  - AUC 기준 통계: 다수 생성 파지가 ΦX174 대비 유의한 우위(일원분산분석+Tukey HSD).

7) 용균 동역학(단일 감염) – 메트릭 및 비교
- 메트릭: 최소 OD600(용균 깊이), 최소 도달 시간(속도), 최대 감소율(분당 OD 감소; 경사).
- 결과(평균 비교, n=3, ANOVA+Tukey):
  - Evo-Φ2483: 최저 OD600 0.07, 최대 감소율 –0.02 OD/min, 최소 도달시간 135분 → ΦX174(0.22, –0.01, 180분) 대비 “더 빠르고 더 강한” 용균.
  - Evo-Φ111, Φ69, Φ108 등도 ΦX174보다 더 낮은 최소 OD600 도달.
  - 단, 빠른 용균이 항상 경쟁 우승으로 직결되진 않음(예: Evo-Φ2483은 경쟁 2개에서 5위).

8) 내성 균주 극복 – 생성 파지 칵테일 vs ΦX174 단독
- 내성화: ΦX174-저항성 E. coli C 3계통(CR1/CR2/CR3) 선발. 전장유전체: LPS 합성 waa 오페론 돌연변이(waaT L259W; waaT 프레임시프트→조기 종결/재개시; waaW A128D).
- 실험: “생성 파지 16종+ΦX174” 칵테일 vs “ΦX174 단독”을 저항성/감수성/혼합 배양에서 계대접종(최대 5 passage).
- 결과:
  - 0 passage: 감수성은 모두 억제, 저항성은 억제 실패(양 군 동일).
  - 칵테일: CR1 1 passage, CR2 2 passage, CR3 5 passage만에 억제 성공.
  - ΦX174 단독: 5 passage 동안 억제 실패.
- 원인 분석: 억제 성공 칵테일에서 단일 우점 파지(Evo-ΦR1, Evo-ΦR2) 분리·시퀀싱 → Evo-Φ111/114/2147 간 재조합 기반 게놈에 추가적 캡시드 변이(SNV) 획득. AlphaFold3로 외부 표면에 집중된 비동의치 변이 확인(스파이크/캡시드 표면—LPS 결합 변화 추정).

핵심 비교 요약
- Evo 2 > Evo 1(제로샷 바이러스성/품질).
- SFT >> 베이스(ΦX174 조건부 생성/리콜, 위치 엔트로피 정합).
- 최적 프롬프트/온도에서 다양성과 필터 통과율 동시 극대화.
- 생성물은 자연 대비 높은 서열·구조 참신성 확보(일부는 종 경계급 ANI).
- 적응도/용균 지표에서 복수 생성 파지가 ΦX174를 능가.
- 내성 극복에서 생성 파지 칵테일이 ΦX174 단독 대비 압도적 우위(계대 1–5회 내 해결).

평가 지표(정의 요약)
- geNomad 바이러스 분류율/점수
- BLAST 기반 유사도(커버리지·정체성, ANI) 및 누적 귀속 비율
- 코딩 밀도, CheckV 품질 등급
- 단백질 구조 신뢰도(ESMFold pLDDT/pTM, AlphaFold3)
- 유전 아키텍처 유사도 점수(시작/종결 경계 기반)
- 스파이크 단백질 ΦX174 대비 %ID
- 평균 AAI(자연 단백질 대비)
- Shannon 다양도(99% ID 클러스터 분포)
- 설계 필터 통과율(retention)
- 경쟁: 누적 Fold Change 시계열, AUC
- 용균: 최소 OD600, 최소 도달시간, 최대 감소율, ANOVA+Tukey

결론(결과 관점)
- 대규모 게놈 LLM(Evo) + SFT + 조건부 프롬프트 + 설계 필터의 결합으로, 전체 게놈 단위의 기능적 파지 설계를 최초로 입증.
- 생성물 16종은 ΦX174 대비 서열·구조·표현형에서 모두 의미 있는 참신성과 성능 향상을 보여주며, 칵테일은 빠른 내성 극복까지 구현.





Study goal
- Establish a genome-scale generative design pipeline that produces functional bacteriophages (ΦX174-like) and validate biological performance (infectivity, lysis, sequence/structural novelty, fitness, and resistance counter-evolution).

Competing models and baselines
- Generative models: Evo 1 and Evo 2 genome language models. Both were specialized via supervised fine-tuning (SFT) on Microviridae (Evo 1 7B 131K SFT; Evo 2 7B 8K SFT).
- Baselines/controls:
  - Natural sequences: curated bacteriophage genomes (Duplodnaviria, Monodnaviria, Riboviria).
  - Scrambled sequences: nucleotide composition preserved, order randomized (negative control).
  - Base vs SFT models: ΦX174 recall and generation quality under prompt length variation.
  - Same design filters applied to Microviridae/natural ΦX174 variants/scrambled controls to benchmark retention.

Datasets (training/evaluation)
- Pretraining: large-scale genomic corpora (including >2M phage genomes).
- SFT: ~15k Microviridae genomes (mostly 4–6 kb), including 134 ΦX174 variants; split into train/val/test (14,266/100/100).
- Zero-shot realm prompting: generate 6,000-nt sequences per realm; compare to natural and scrambled controls.

Modeling and prompts
- Zero-shot: realm prompts (temp 0.7, top-k=4, top-p=1).
- SFT models: 1–11 nt ΦX174 consensus start as prompts; temperature sweep 0.3–1.1; best trade-off at 4–9 nt prompts and temperature 0.7–0.9.

Design constraints (filters) and key metrics
- Three tiers:
  1) Quality: A/C/G/T only; length 4–6 kb; GC 30–65%; homopolymer ≤10 nt; ≥7 predicted protein hits to ΦX174 via bespoke CDS calling.
  2) Tropism: spike protein ≥60% identity to ΦX174 spike.
  3) Diversification (optional): mean AAI <95% vs natural proteins; genetic-architecture similarity ≤0.9; exactly one synteny break; total gene count 10 or 12 preferred.
- Auxiliary diagnostics: CheckV quality, geNomad viral score, BLAST (ANI/coverage), architecture score (one-hot start/stop), structure quality (ESMFold/AlphaFold3).

Gene prediction benchmarking
- Six common tools (Prodigal, Pyrodigal-gv, PHANOTATE, pLannotate, Glimmer, GeneMark) systematically missed overlapping ORFs—none annotated all 11 ΦX174 genes.
- A bespoke ΦX174-like annotation pipeline was developed and used to enforce gene-level constraints.

1) Zero-shot generation (pretrained models) – model comparison
- Viral classification (geNomad):
  - Natural: 89–100% viral (mean >0.96).
  - Scrambled: 1.9–11% (mean >0.73).
  - Evo 1: 19–33% (mean >0.80); Evo 2: 34–38% (mean >0.87).
  → Evo 2 outperforms Evo 1; both far above scrambled, below natural.
- BLAST: low coverage/identity vs natural → high novelty.
- Coding density: high like natural phages (unlike scrambled).
- Protein structure quality (ESMFold): mean pLDDT comparable to natural proteins; higher than scrambled controls.
- Protein identity/annotation: low identities to OpenGenome/PHROGs; functional categories mirror natural phages.

2) Conditional generation after SFT – base vs SFT
- ΦX174 recall vs prompt length:
  - SFT models rapidly gain recall with longer consensus-start prompts.
  - Base models fail to recall ΦX174 at any tested prompt length.
- Positional entropy aligns better with ΦX174 features after SFT.

3) Parameter sweeps – diversity/quality trade-off
- Diversity (Shannon entropy) increases with temperature and prompt length.
- Retention (post-filter passing rate) remains high at optimal region (0.7–0.9 temp; 4–9 nt prompt).
- Example retention: after tropism filter, Evo 1 SFT up to 100%, Evo 2 SFT 53.8%; after diversification filters, Evo 1 10.4%, Evo 2 17.2%.

4) Synthesis and experimental testing (test data and metrics)
- 302 designs selected; 285 assembled; 16 viable phages identified by growth-inhibition “rebooting” assay in E. coli C.
- Sequence quality: >87% rated High Quality/Complete by CheckV.
- Spike identity mostly ≥85% (tropism preserved).
- Against ΦX174/Microviridae, candidates combine template-guided nucleotide similarity (often >40% identity) with low protein-level AAI (down to 63%).
- Host range: ΦX174 and 15/16 generated phages inhibit E. coli C and W; no inhibition on six K-12/B strains—strong specificity corroborating the tropism filter.
- Titers varied across phages.

5) Sequence, phylogenetic, and structural novelty
- Novelty: 67–392 novel mutations vs nearest natural genome (ANI 93.0–98.8%). Evo-Φ2147 at 93.0% ANI (392 mutations) vs NC51—comparable to new-species thresholds (ANI<95%).
- Architectural changes: insertions/deletions/elongations/truncations; gene swaps (e.g., Evo-Φ36 uses G4-like J).
- “Unattributable” mutations: in 13/16, segments cannot be reconstructed by cumulative natural BLAST hits (notably extended noncoding regions).
- Cryo-EM: ΦX174 2.8 Å, Evo-Φ36 2.9 Å; Evo-Φ36 accommodates a shorter G4-type J (25 aa vs 38 aa in ΦX174) with altered N-terminal behavior yet compatible capsid interactions—novel co-evolutionary solution.

6) Fitness competitions (head-to-head) – metrics and comparison
- Setup: 16 generated phages + ΦX174 co-infection at equal MOI in E. coli C; phage-specific read counts tracked over time.
- Metrics:
  - Cumulative fold change (FC) of read counts over time (log2(FC) trajectories).
  - AUC over the 6 h course as overall performance.
- Results (3 replicates):
  - Evo-Φ69 ranked first in all three (cumulative FC 16×–65× at 6 h). ΦX174 reached only 1.3×–4.0×.
  - Evo-Φ69/Φ100/Φ111 repeatedly in top 5; ΦX174 at best 3rd (two runs).
  - AUC: several generated phages significantly outperformed ΦX174 (one-way ANOVA + Tukey HSD).

7) Lysis kinetics (single infections) – metrics and comparison
- Metrics: minimum OD600 (lysis depth), time to minimum (speed), maximum rate of decline (slope; OD/min).
- Results (n=3; ANOVA + Tukey):
  - Evo-Φ2483: min OD600 0.07; max decline −0.02 OD/min; time to min 135 min vs ΦX174 0.22, −0.01, 180 min → faster and stronger lysis.
  - Evo-Φ111/Φ69/Φ108 also achieved significantly lower minima than ΦX174.
  - Fast lysis does not always equal top fitness rank (e.g., Evo-Φ2483 was ~5th in two competitions).

8) Overcoming bacterial resistance – generated cocktail vs ΦX174-only
- Resistant strains (CR1/CR2/CR3): mutations in LPS-synthesis waa operon (waaT L259W; waaT frameshift/premature stop with possible reinitiation; waaW A128D).
- Serial passaging on mixtures of susceptible+resistant cultures:
  - Passage 0: both treatments inhibit susceptible but not resistant.
  - Generated cocktail: overcame CR1 after 1 passage, CR2 after 2, CR3 after 5.
  - ΦX174-only: failed to overcome any resistant strain by passage 5.
- Sequencing: predominant resistant phages (Evo-ΦR1/ΦR2) assembled from recombination of Evo-Φ111/Φ114/Φ2147 with additional capsid SNVs; AlphaFold3 places most nonsynonymous changes on exterior surfaces (consistent with altered LPS interactions).

Key comparative takeaways
- Evo 2 > Evo 1 in zero-shot viral realism.
- SFT >> base models for template-conditional generation and recall.
- Optimal prompt/temperature delivers both high diversity and high filter retention.
- Generated genomes achieve substantial sequence/structural novelty (including near “new-species” ANI).
- Multiple generated phages surpass ΦX174 in fitness and/or lysis metrics.
- Generated phage cocktails rapidly overcome resistance; ΦX174 alone does not.

Core metrics (definitions)
- geNomad viral classification rate/score
- BLAST similarity (coverage/identity; ANI) and cumulative attribution to natural sequences
- Coding density; CheckV quality class
- Protein structure confidence (ESMFold pLDDT/pTM; AlphaFold3)
- Genetic architecture similarity (start/stop boundary vectors)
- Spike % identity to ΦX174; mean AAI vs natural proteins
- Shannon diversity (99% identity clustering)
- Design-filter retention rate
- Competition: cumulative FC over time; AUC
- Lysis: minimum OD600, time to minimum, max decline rate; ANOVA + Tukey

Bottom line (results-centric)
- Combining large genome LMs (Evo), SFT, careful prompting, and rigorous design-time filtering yields the first experimentally verified, genome-scale generative design of viable bacteriophages.
- The 16 generated phages exhibit meaningful sequence/structural novelty and superior phenotypes (fitness, lysis), and a cocktail rapidly overcomes bacterial resistance—substantially outperforming ΦX174 alone.


<br/>
# 예제




1) 사전학습(Pretraining) 맥락에서의 제로샷 생성 벤치마크(바이러스 렐름 프롬프트)
- 목적(테스크)
  - 언어모델(Evo 1, Evo 2)의 “제로샷” 바이러스 유사 게놈 생성 능력 점검.
  - 입력 프롬프트에 바이러스 렐름(Realm) 토큰을 넣고, 길이 6,000 nt의 염기서열을 생성. 생성물이 실제 박테리오파지 유사 특성을 보이는지 자동 분류·구조예측 등으로 평가.
- 입력(Input)
  - 텍스트 프롬프트(렐름 레이블):
    - Evo 1: “|r__Duplodnaviria;k__”, “|r__Monodnaviria;k__”, “|r__Riboviria;k__”
    - Evo 2: “|r__Duplodnaviria;;;;;;;|”, “|r__Monodnaviria;;;;;;;|”, “|r__Riboviria;;;;;;;|”
  - 샘플링 설정(대표): 길이 6,000 nt, temperature 0.7, top-k 4, top-p 1
  - 비교 데이터(평가용):
    - 자연계 양성 컨트롤: 각 렐름에서 10,000개(사전학습 데이터 OpenGenome에서 추출) 자연 게놈
    - 음성 컨트롤: 같은 자연 게놈들을 뉴클레오타이드 순서만 무작위 재배열(“scrambled”)한 서열
- 출력(Output)
  - 모델 생성 DNA 서열(6,000 nt)
  - 평가 결과(대표 지표)
    - geNomad 바이러스 판별률 및 점수(Fig. 1D, Fig. S1A):
      - 자연 게놈: 89–100%가 바이러스로 분류(평균 점수 >0.96)
      - 스크램블 컨트롤: 1.9–11% (평균 점수 >0.73)
      - Evo 1 생성물: 19–33% (평균 점수 >0.80)
      - Evo 2 생성물: 34–38% (평균 점수 >0.87)
    - BLAST(핵산)로 자연계 대비 신기성: 낮은 query cover/identity → 높은 참신성(Fig. 1E,F)
    - 코딩밀도(Prodigal)와 구조예측(ESMFold pLDDT, pTM) 품질은 자연 단백질과 유사(Fig. 1G–H, Fig. S1B)
    - 단백질 동정: OpenGenome/PHROGs에 낮은 유사도 분포, 기능 어노테이션 분포는 자연 파지와 유사(Fig. 1I–J)

2) Microviridae 특화 지도학습(Supervised Fine-Tuning, SFT)
- 목적(테스크)
  - ΦX174와 유사한 Microviridae 전체 게놈을 더 정밀하게 생성하기 위해 Evo 1/2를 Microviridae 데이터로 SFT.
- 데이터셋(트레이닝/검증/테스트)
  - 수집: 총 15,507개 Microviridae 게놈(NCBI Datasets, PhageScope, OpenGenome). 필터링 후 14,466개 최종 사용(Fig. S2B, Methods B.1.4)
  - 스플릿: train 14,266 / val 100 / test 100 (Methods B.1.5)
  - 전처리:
    - 모든 서열에 Microviridae 토큰 “+”를 프리펜드
    - ΦX174(NC_001422.1)와의 %ID 대역을 토큰으로 추가(“∼”(95–100%), “ˆ”(80–95%), “#”(70–80%), “$”(50–70%), “!”(<50%)) (Methods B.1.5)
- 입력(Input)
  - 학습 시: 위 전처리 토큰이 붙은 원시 DNA 서열(최대 10,240 토큰 컨텍스트)
- 출력(Output)
  - 다음 토큰(염기) 예측(언어모델 표준 LMs의 next-token likelihood 학습)
  - 결과(대표):
    - 학습 곡선 수렴(Fig. S3C–F)
    - ΦX174 변이군에 대한 위치별 엔트로피 감소 → ΦX174 유사 게놈을 더 잘 모델링(Fig. S3G–H)

3) ΦX174 합성 지향 프롬프트 및 리콜(Recall) 실험
- 목적(테스크)
  - “ΦX174 합의(consensus) 시작 염기서열”의 앞부분을 프롬프트로 주었을 때, 모델이 ΦX174를 리콜 또는 ΦX174-유사 게놈을 생성할 수 있는지 평가.
- 입력(Input)
  - 프롬프트: “+∼” 토큰 뒤에 ΦX174 시작 합의서열의 앞 1~11 nt (Methods B.1.13)
  - 생성 설정(대표): 길이 6,000 nt, temperature 1.1, top-k 4, top-p 1, 각 조합당 약 1,000개 생성
- 출력(Output)
  - BLAST 또는 MMseqs2 정렬로 ΦX174 리콜 여부(유의미 정렬)
  - 결과:
    - SFT 모델: 4–9 nt 정도의 짧은 합의 프롬프트로도 ΦX174-like 생성/리콜 가능(Fig. 2E, Fig. S6)
    - 베이스(비SFT) 모델: 같은 길이 프롬프트로 ΦX174 리콜 실패(Fig. 2E)

4) 생성–필터–선정 파이프라인(인퍼런스 타임 제약)
- 목적(테스크)
  - “생존 가능한 Microviridae/ΦX174-유사(특정 숙주 트로피즘)”를 갖는 설계 게놈을 자동 생성·평가.
- 입력(Input)
  - SFT 모델 + 프롬프트(합의 시작 nt 4–9, 온도 0.7–0.9 권장; Fig. 2J–K, Fig. S6)
  - 생성물: 전체 게놈(4–6 kb)
- 출력(Output)
  - 3단계 제약 필터 통과 여부(Methods B.1.9; Fig. 2F,H; Fig. S7A)
    1) 품질(QC)
       - 문자 A/C/G/T, 길이 4–6 kb, GC 30–65%, homopolymer ≤10, “예측 단백질 히트 수 ≥7”(ΦX174 단백질 대비)
    2) 트로피즘(숙주 범위) 제약
       - 스파이크 단백질이 ΦX174 스파이크 대비 ≥60% 단백질 서열 일치
    3) 다양화(신규성) 제약(선호·권장)
       - 평균 단백질 AAI ≤95%
       - 유전자 배열(architecture) 유사도 점수 ≤0.9
       - ΦX174 대비 synteny 1개 유전자 단절(또는 총 유전자 수 10 또는 12)
  - 평가 지표
    - 샤논 다양도(클러스터 99% ID 기준) (Fig. 2J)
    - retention rate(각 단계 필터 통과 비율) (Fig. 2I–K)
  - 핵심 결과
    - Tropism 필터까지: Evo 1 SFT 최대 100% 보존, Evo 2 SFT 53.8% 보존(Fig. 2I)
    - Diversification 필터까지: Evo 1 10.4%, Evo 2 17.2% 보존(Fig. 2I)
    - 적정 프롬프트 길이(4–9 nt)+온도(0.7–0.9)에서 다양성과 보존율 균형(Fig. 2J–K, Fig. S6)

5) 유전자 예측/어노테이션 테스크(파이프라인 내부 사용)
- 목적(테스크)
  - Microviridae/ΦX174-유사 게놈에서 겹침 ORF까지 포괄하는 정확한 CDS 예측 필요.
- 입력(Input)
  - 생성 게놈 DNA 서열
- 처리(Task)
  - 기존 6개 도구(Prodigal, Glimmer, PHANOTATE, GeneMark 등) 벤치마크 → ΦX174의 11개 유전자 모두 예측 실패(Fig. 2G, Fig. S4A)
  - 맞춤 파이프라인(Methods B.1.10, Fig. S4C): “pseudo-circularization → orfipy로 ORF 후보 → PHROGs 단백질 DB에 MMseqs2로 all-vs-all”
- 출력(Output)
  - 예측 유전자 목록 및 PHROGs 기반 기능 라벨
  - 파이프라인을 QC 제약(예측 단백질 히트 수 ≥7)에 사용

6) 아키텍처 유사도(Genetic architecture) 점수화 테스크
- 목적(테스크)
  - ΦX174 대비 전체 유전자 경계(ATG 시작/정지코돈) 배열의 유사도 정량화(빠르고 대량 처리 가능).
- 입력(Input)
  - 생성 게놈 DNA 서열
- 처리(Task)
  - 시작/정지코돈 위치를 원-핫 인코딩 → 가우시안 블러(σ=5)로 경계 유연화 → ΦX174 기준 벡터와 도트프로덕트 정규화(1.0=완전일치) (Methods B.1.11; Fig. S5)
- 출력(Output)
  - architecture similarity score (0–1)
  - 필터에서 임계치(≤0.9)로 사용

7) 테스트 세트/지표 예시(설계물 신기성·품질 평가)
- 자연/스크램블 대조군을 통한 필터 성능 확인(Fig. 2I)
- BLAST 최근접 자연 게놈 대비 ANI/돌연변이 수(Fig. 4D), 누적 게놈 귀속(자연 서열로 설명 가능한 비율; Fig. 4F, Fig. S14)
- 단백질 구조 예측(ESMFold/AlphaFold3) 품질 지표(pLDDT, pTM) (Fig. 1H, Fig. S1B)
- PHROGs 기능 분포 유사성(Fig. 1J)

8) 최종 선택·합성 후보(컴퓨팅 관점)
- 입력(Input)
  - 위 제약을 통과한 302개 다양 후보(Fig. 3A–C, Fig. S7B, Fig. S8)
- 출력(Output/특징)
  - 길이 4–6 kb, 스파이크 유사도 대부분 ≥85%, 평균 AAI는 최저 63%까지 관찰(Fig. 3A)
  - CheckV로 High Quality/Complete ≥87% (Fig. 3A)
  - synteny 단절(새 CDS/비코딩 확장 포함) 다수(Fig. 3B–C)

참고: 위 8개 블록 중 1–7은 전산/모델·데이터/분석 중심(안전). 8은 합성 후보의 전산적 최종 상태 요약입니다. 논문에는 이 후보들의 실험 검증(재부팅, 감염, 크라이오-EM, 경쟁·내성 실험)이 있지만, 본 답변은 모델·데이터·평가 테스크에 국한해 정리했습니다.



1) Zero-shot realm prompting during pretraining context
- Goal (task)
  - Test whether Evo 1/2 can generate phage-like genomes “zero-shot” when prompted with viral realm tags.
- Input
  - Text prompts (realm labels):
    - Evo 1: “|r__Duplodnaviria;k__”, “|r__Monodnaviria;k__”, “|r__Riboviria;k__”
    - Evo 2: “|r__Duplodnaviria;;;;;;;|”, “|r__Monodnaviria;;;;;;;|”, “|r__Riboviria;;;;;;;|”
  - Sampling (typical): 6,000 nt length, temperature 0.7, top-k 4, top-p 1
  - Evaluation controls:
    - Natural positives: 10,000 per realm from OpenGenome
    - Scrambled negatives: nucleotide-shuffled versions of the same natural sequences
- Output
  - Generated DNA (6,000 nt)
  - Metrics
    - geNomad viral classification (Fig. 1D; Fig. S1A):
      - Naturals: 89–100% viral (mean >0.96)
      - Scrambled: 1.9–11% (mean >0.73)
      - Evo 1 gens: 19–33% (mean >0.80)
      - Evo 2 gens: 34–38% (mean >0.87)
    - Novelty by BLAST vs natural sequences (Fig. 1E–F)
    - Coding density (Prodigal) and structure quality (ESMFold pLDDT/pTM) comparable to naturals (Fig. 1G–H; Fig. S1B)
    - Low protein sequence identity to OpenGenome/PHROGs with realistic functional annotation distributions (Fig. 1I–J)

2) Supervised fine-tuning (SFT) on Microviridae
- Goal (task)
  - Specialize Evo 1/2 to generate ΦX174-like complete Microviridae genomes.
- Dataset (train/val/test)
  - Collected 15,507 Microviridae genomes; after filtering, 14,466 used (Fig. S2B; Methods B.1.4)
  - Split: train 14,266 / val 100 / test 100 (Methods B.1.5)
  - Preprocessing:
    - Prepend “+” token (Microviridae)
    - Add a ΦX174-identity band token: “∼”(95–100%), “ˆ”(80–95%), “#”(70–80%), “$”(50–70%), “!”(<50%) (Methods B.1.5)
- Input
  - Tokenized DNA per genome (≤10,240 tokens), with the above soft prompts
- Output
  - Next-token DNA modeling (standard LM training)
  - Results:
    - Converged loss (Fig. S3C–F)
    - Reduced per-position entropy on ΦX174 variants → better modeling of ΦX174-like genomes (Fig. S3G–H)

3) ΦX174 consensus-start prompting and recall
- Goal (task)
  - With a short prefix of the ΦX174 consensus start sequence, assess recall of ΦX174 or ΦX174-like generation.
- Input
  - Prompts: “+∼” followed by 1–11 nt of the ΦX174 consensus start (Methods B.1.13)
  - Generation: ~1,000 sequences per prompt; length 6,000 nt; temp 1.1; top-k 4; top-p 1
- Output
  - Recall by significant alignment to ΦX174
  - Results:
    - SFT models: 4–9 nt prompts suffice for ΦX174-like recall/generation (Fig. 2E; Fig. S6)
    - Base models: failed to recall across all prompt lengths (Fig. 2E)

4) Generate–filter–select pipeline with inference-time constraints
- Goal (task)
  - Propose complete, host-tropic Microviridae/ΦX174-like genomes.
- Input
  - SFT model + ΦX174 consensus-start prompt (optimal ~4–9 nt) and sampling temp 0.7–0.9 (Fig. 2J–K; Fig. S6)
- Output
  - Full genomes (4–6 kb) evaluated by 3 constraint tiers (Methods B.1.9; Fig. 2F,H; Fig. S7A)
    1) Quality control:
       - A/C/G/T only; length 4–6 kb; GC 30–65%; homopolymer ≤10; predicted protein hits ≥7 (to ΦX174 proteins)
    2) Tropism:
       - Spike protein identity ≥60% to ΦX174 spike
    3) Diversification (preferred):
       - Mean protein AAI ≤95%
       - Genetic architecture similarity score ≤0.9
       - Synteny break of one gene (or total gene count 10 or 12)
  - Metrics
    - Shannon diversity at 99% clustering (Fig. 2J)
    - Retention after each filter (Fig. 2I–K)
  - Key results
    - After tropism filter: up to 100% retained for Evo 1 SFT, 53.8% for Evo 2 SFT (Fig. 2I)
    - After diversification: 10.4% (Evo 1), 17.2% (Evo 2) (Fig. 2I)
    - Best trade-off at prompt 4–9 nt and temp 0.7–0.9 (Fig. 2J–K; Fig. S6)

5) Gene prediction/annotation task (for filtering)
- Goal (task)
  - Accurate CDS prediction including overlapping ORFs in ΦX174-like genomes.
- Input
  - Generated DNA genomes
- Processing
  - Six tools benchmarked (Prodigal, PHANOTATE, Glimmer, GeneMark, etc.) → none recovered all 11 ΦX174 genes (Fig. 2G; Fig. S4A)
  - Custom method (Methods B.1.10; Fig. S4C): pseudo-circularization → ORF candidates with orfipy → PHROGs DB all-vs-all via MMseqs2
- Output
  - Gene set and functional labels; used to enforce “≥7 predicted protein hits” QC rule

6) Genetic architecture similarity scoring task
- Goal (task)
  - Quantify global gene-boundary arrangement similarity vs ΦX174.
- Input
  - Generated DNA genomes
- Processing
  - One-hot encode start/stop positions; Gaussian blur (σ=5); dot product vs ΦX174 template; normalize to 1.0 (Methods B.1.11; Fig. S5)
- Output
  - Architecture similarity score; used with cutoff ≤0.9

7) Test set/metrics examples for novelty and quality
- Controls to validate filtering (Microviridae, scrambled, ΦX174 variants; Fig. 2I)
- Nearest-natural identity/novel mutations (Fig. 4D) and cumulative genome attribution (Fig. 4F; Fig. S14)
- Structure metrics (ESMFold/AF3 pLDDT/pTM; Fig. 1H; Fig. S1B)
- Functional annotation distributions vs PHROGs (Fig. 1J)

8) Final computationally selected candidates (pre-synthesis snapshot)
- Input
  - 302 diverse candidates passing constraints (Fig. 3A–C; Fig. S7B; Fig. S8)
- Output (properties)
  - 4–6 kb; spike identities mostly ≥85%; mean proteome AAI down to 63% (Fig. 3A)
  - CheckV: ≥87% High Quality/Complete (Fig. 3A)
  - Frequent synteny breaks via novel CDS/noncoding expansions (Fig. 3B–C)

Note: Items 1–7 above cover modeling/data/evaluation tasks; item 8 is a computational summary of the final set prior to any lab work. Experimental protocols are intentionally omitted here.

원문 내 주요 출처: Fig. 1D–J, Fig. 2A–K, Fig. S1–S7, Methods B.1.x 전반, Fig. 3A–C, Fig. S8.

<br/>
# 요약


- 메써드: Evo 1/2 genome LMs를 Microviridae ~1.5만 유전체로 SFT하고 ΦX174 시작부 컨센서스로 프롬프트링, 길이·GC·동일염기 반복·맞춤 ORF/유전자 주석·스파이크 단백질 ≥60% 동일성(표적 숙주 트로피즘)·유전자 배치(아키텍처)·AAI 다변화 필터를 적용해 302개 설계를 만들고(285개 조립), 플라크/성장억제, 증식·타이팅, 숙주범위, 경쟁, cryo-EM, 내성 극복 패시지 실험으로 검증했다. 
- 결과: 16개 AI-설계 유전체가 생존성 파지로 확인되었고(일부 <95% ANI·유전자 삽입/결실·비부호 영역 확장), 여러 개가 ΦX174보다 경쟁 피트니스·용균 속도에서 우수했으며, cryo-EM으로 Evo-Φ36이 ΦX174 유사 캡시드 안에 G4 유래 J 포장단백질을 기능적으로 수용함을 보였다. 
- 예시: Evo-Φ69는 혼합 경쟁에서 누적 16–65배로 최상위 성장을 보였고, Evo-Φ2483는 가장 빠르고 깊은 용균을 유도했으며, 16종 칵테일은 ΦX174-내성 E. coli C 3개 계통의 내성을 수차 패시지 내에 신속히 극복하면서 재조합으로 탄생한 Evo-ΦR1/R2(캡시드/스파이크 변이 보유)가 주도했다.



- Methods: The Evo 1/2 genome LMs were SFT on ~15k Microviridae genomes, prompted with a ΦX174 consensus start, and filtered by length/GC/homopolymers, a bespoke ORF/annotation pipeline, spike identity ≥60% for host tropism, architecture similarity, and proteome AAI to yield 302 designs (285 assembled), which were validated by plaque/growth assays, propagation/titering, host-range, head-to-head competitions, cryo-EM, and resistance-passaging. 
- Results: Sixteen AI-designed genomes produced viable phages with substantial novelty (e.g., <95% ANI, gene insertions/deletions, extended noncoding regions); several surpassed ΦX174 in fitness and lysis kinetics, and cryo-EM showed Evo-Φ36 functionally accommodates the distant G4 J protein within a ΦX174-like capsid. 
- Examples: Evo-Φ69 dominated mixed competitions (16–65× cumulative fold change), Evo-Φ2483 drove the fastest/deepest lysis, and a 16-phage cocktail rapidly overcame ΦX174 resistance in three E. coli C strains via recombination, yielding predominant resistant phages Evo-ΦR1/R2 with capsid/spike mutations.

<br/>
# 기타



메인 피규어(도 1–6)
- Figure 1 | Evo 모델의 제로샷(사전 미세조정 전) 생성능력 평가
  - 결과: Evo 1/2가 바이러스성(genomad 분류)·코딩밀도·단백질 구조 신뢰도(pLDDT/pTM)·기능 주석 분포(파지형) 등에서 자연 파지와 유사한 전장 서열을 생성. BLAST 유사도는 낮아(커버리지/ID↓) 신규성 높음.
  - 인사이트: 대규모 사전학습만으로도 파지 유전체의 문법/제약을 상당히 포착하며, 자연계 범위를 넘어서는 다양성을 창출할 수 있음.

- Figure 2 | 전장 파지 설계 파이프라인과 SFT(미세조정)·프롬프트·필터
  - 결과: Microviridae 데이터로 SFT 후, ΦX174 시작부위 합의서열(4–9nt) 프롬프트와 적정 온도(0.7–0.9)에서 ΦX174-유사 아키텍처/스파이크를 유지하면서도 다양한 후보 생성. 기존 유전자 예측 도구가 ΦX174 겹침 ORF를 놓치는 한계를 보여 새 CDS 예측법 개발. 품질·트로피즘·다양화 필터를 거쳐 높은 유지율과 샤논다양도 확보.
  - 인사이트: 전장 설계를 위해서는 SFT+프롬프트 길이+샘플링 온도 튜닝과, 아키텍처·트로피즘·AAI 기반 다층 필터링이 핵심. 겹침유전자 대응 등 도메인 맞춤 예측기가 필요.

- Figure 3 | 합성·재부팅 및 기능 검증
  - 결과: 302개 설계 중 285개 합성·조립, 그 중 16개가 E. coli C에서 성장 억제(=활성 파지). K-12에선 활성 없음(트로피즘 필터 유효). 일부는 E. coli W에도 활성. 야생형 ΦX174 대조로 플라크/성장곡선/타이팅 검증 완료.
  - 인사이트: AI-설계 유전체가 실제로 감염·용균 가능. 설계 단계의 트로피즘 제어가 실험적으로도 견고함을 입증.

- Figure 4 | 서열·구조적 신규성
  - 결과: 수백 개의 동의·비동의·비코딩 변이, 유전자 삽입/결실/연장, J 유전자(패키징) 교체(예: G4 J)를 포함. 많은 변이가 자연서열 재조합만으로 설명 불가. Evo-Φ36는 G4의 짧은 J를 캡시드와 호환되게 사용. Cryo-EM(ΦX174: 2.8Å, Evo-Φ36: 2.9Å)으로 캡시드–J 결합 모드 차이 확인(N-말단 비정형성 증가 추정).
  - 인사이트: 모델이 자연계에서 비가용하던 모듈 조합(예: J 교체)을 문맥 조화롭게 설계 가능. 생성체는 진화경로 밖의 구조적 해법(단백질-단백질 공진화)을 드러냄.

- Figure 5 | 적응도 경쟁 및 용균 역학
  - 결과: 혼합 감염 경쟁에서 여러 생성 파지가 ΦX174를 상회(Evo-Φ69, Φ-100, Φ-111 등). 용균 속도/깊이 지표에서 Evo-Φ2483가 가장 빠르고 강력한 용균을 보였으나, 경쟁 적응도는 복합 요인.
  - 인사이트: 생성 유전체가 전장 수준의 고적응도 해법을 제시. 용균 역학과 장기 증식성은 분리된 설계 타깃이 될 수 있음.

- Figure 6 | 내성 극복
  - 결과: ΦX174만으로는 내성 균주(waa 오페론 변이) 제압 실패. 반면 16종 생성 파지 칵테일은 1–5회 패시지 내에 3개 내성 균주 모두 억제. 주도 파지는 생성 파지간 재조합+신규 돌연변이로 탄생(Evo-ΦR1/ΦR2), 변이는 캡시드/스파이크 외측에 집중(LPS 결합 개조 시사).
  - 인사이트: 설계 다양성은 빠른 카운터-어댑테이션을 촉발, 치료적 회복탄력성 강화 가능.

보충 피규어(Suppl. Figures S1–S21)
- S1: geNomad가 자연 vs 스크램블 구분 확실. 생성 단백질 pTM도 자연과 유사. 인사이트: 생성 서열이 바이러스적 특징과 접힘 신뢰도를 갖춤.
- S2: Microviridae 데이터 규모·출처 요약. 인사이트: SFT 토대가 충분.
- S3: SFT 학습곡선 안정, 위치 엔트로피 감소(모델이 ΦX174 규칙 내재화). 인사이트: SFT 효과적.
- S4: 겹침 ORF 때문에 범용 도구가 유전자 누락. 새로운 의사-원형화+ORF 호출+PHROGs 주석 파이프라인 제시. 인사이트: 파지 전장 설계엔 전용 주석법 필요.
- S5: 유전자 경계 원-핫+가우시안 블러 기반 ‘유전 아키텍처 유사도’ 지표/임계값 확립. UMAP에서 ΦX174-유사 군집 분리. 인사이트: 전장 아키텍처를 빠르게 정량 필터링 가능.
- S6: 온도·프롬프트 길이 스윗스폿 확인(기억회상 회피·다양성/품질 균형). 인사이트: 생성 하이퍼파라미터가 품질·유사도·다양성에 결정적.
- S7: 최종 필터 기준과 합성 성공률. 인사이트: 공정적으로 재현 가능한 설계군 확보.
- S8: 합성 후보 vs ΦX174 정렬 맵. 인사이트: 전장 수준 다양성 시각화.
- S9: ΦX174 재부팅·플라크·타이팅 검증 확장. 인사이트: 실험 파이프라인 견고성.
- S10: C에서는 활성, K-12에서는 비활성. 인사이트: 트로피즘 설계가 실제 격리됨.
- S11: 활성 생성 파지의 장독 서열검증—일부는 소수 변이 동반. 인사이트: 재부팅·증식 중 미세한 변화 가능.
- S12: 개별 파지 타이팅. 인사이트: 생산성/활성 편차 존재(후속 최적화 타깃).
- S13: 8개 E. coli 균주 감염 프로파일—W에서 예기치 않은 활성. 인사이트: 스파이크 유사성 기반 트로피즘이 주로 유지되나, 숙주 범위 변동성 여지.
- S14: 누적 귀속 분석—다수 변이는 자연서열 재조합으로 설명 불가. 인사이트: 진정한 서열 신규성.
- S15: Evo-Φ36 J의 계통/구조 예측—캡시드 결합 모드 차이. 인사이트: 모듈 교체의 구조적 근거.
- S16: Cryo-EM 준비·품질 지표. 인사이트: 고품질 입자 확보.
- S17: 처리 파이프라인·FSC·해상도 검증. 인사이트: 지도의 신뢰성 확보.
- S18: 국소 해상도 및 모델-지도 적합. 인사이트: 결정적 결합부 해석 가능.
- S19: 내부면 밀도 가시화(J·핵산 밀도 영역). 인사이트: 포장/결합 상호작용 해석 보조.
- S20: 용균 실험의 성장률 곡선. 인사이트: 속도/기울기 지표의 비교 근거.
- S21: 내성 확립·전유전체 차이(waa 변이)와 카운터파지 구조 예측. 인사이트: 내성 기전과 역적응 표면 변이의 연결.

테이블
- Table S1 | Cryo-EM 수집·정제·검증 요약
  - 결과: ΦX174 2.76Å, Evo-Φ36 2.90Å 해상도, 표준 검증(FSC, 모델 검증) 양호.
  - 인사이트: 원자수준에 가까운 품질로 캡시드–J 인터페이스 차이를 신뢰성 있게 비교 가능.

어펜딕스/보충 텍스트
- A. Biosafety and biocontainment discussion
  - 결과/인사이트: 학습데이터에서 사람/진핵 바이러스 배제, Microviridae로만 SFT → 모델 수준의 내재 안전장치. 비병원성 E. coli C–ΦX174 시스템과 엄격한 실험실 절차 채택. 트로피즘 필터로 오프타깃 숙주 억제. 전장 설계의 책임 있는 경로 제시.

- B. Methods(핵심 포인트)
  - 결과/인사이트: 
    - 생성: Evo 1/2 SFT 설정, 프롬프트·온도 스윕, Microviridae 데이터 파이프라인.
    - 평가: 새 CDS 예측법(겹침 ORF 대응), 유전 아키텍처 점수, 샤논 다양도·AAI·트로피즘(스파이크 ID) 지표.
    - 실험: 합성–조립–재부팅–타이팅–감염/경쟁/내성-카운터어댑테이션 일련의 표준화 워크플로.
    - 구조: AF3 예측과 Cryo-EM(정제–수집–정제–모델링)로 분자 상호작용 검증.

- Data and code availability
  - 결과/인사이트: 재현성과 확장 연구를 위한 데이터셋·SFT 모델·코드 전면 공개(생성·분석·미세조정·지도/PDB 제출 예정).

- Supplementary files (File S1)
  - 결과/인사이트: 최종 후보 목록, 합성 단편 설계, 시퀀싱 검증된 활성 유전체 등 실용 자료 제공.

종합 인사이트
- 사전학습+SFT+프롬프트·온도 튜닝+다층 필터 조합으로 “전장”에서도 기능·트로피즘·아키텍처를 동시에 만족하는 설계가 가능.
- 생성체는 자연계 범위를 넘는 서열/구조적 신규성과, 실험적으로 확인된 고적응도·빠른 카운터-어댑테이션 잠재력을 보임.
- 전장 설계를 뒷받침하는 도구(겹침ORF 주석, 아키텍처 점수, 트로피즘 제약)가 일반화 가능한 빌딩블록으로 제시됨.


Below is a results-and-insights–focused digest of diagrams, figures, tables, and appendices (including supplementary material).

Main figures (Figs. 1–6)
- Figure 1 | Zero-shot phage-like genome generation by Evo
  - Results: Evo 1/2 produce sequences classified as viral (geNomad), with high coding density, realistic protein structure confidence (pLDDT/pTM), and phage-like functional annotations. BLAST shows low identity/coverage to known genomes (high novelty).
  - Insight: Large-scale pretraining alone captures genome-scale “grammar” of phages and generates diversity beyond natural evolution.

- Figure 2 | End-to-end design pipeline with SFT, prompts, and filters
  - Results: After Microviridae SFT, prompting with 4–9 nt of the ΦX174 consensus start and temperature 0.7–0.9 yields ΦX174-like, tropism-keeping yet diverse genomes. Standard ORF tools miss overlapping genes; a bespoke CDS predictor rescues annotation. Quality/tropism/diversification filters retain high-quality sets with high Shannon diversity.
  - Insight: Genome-scale design hinges on SFT, prompt length, sampling temperature, and layered filters (architecture, tropism, AAI). Domain-specific gene calling is crucial for overlapping ORFs.

- Figure 3 | Synthesis, rebooting, and functional validation
  - Results: 285/302 designs synthesized; 16 rebooted into active phages in E. coli C. No activity in K-12 (tropism filter holds). Several also act on E. coli W. Plaque/growth/titer validated with ΦX174 controls.
  - Insight: AI-designed genomes function in cells with narrow host range as intended; the screening pipeline is robust.

- Figure 4 | Sequence and structural novelty
  - Results: Hundreds of mutations; gene insertions/deletions/elongations; J gene swap (e.g., from G4) otherwise non-viable in WT ΦX174 context. Many changes not explainable by recombination of known sequences. Cryo-EM (2.8–2.9 Å) reveals distinct capsid–J interactions in Evo-Φ36 (more disordered N-terminus).
  - Insight: Models can compose context-compatible module swaps and reach structural solutions outside known evolutionary paths (protein–protein co-evolution compatibility).

- Figure 5 | Fitness competitions and lysis dynamics
  - Results: Several designed phages outperform ΦX174 in co-infection (e.g., Evo-Φ69/Φ100/Φ111). Fastest and deepest lysis (Evo-Φ2483) doesn’t always equal top fitness in mixed competitions.
  - Insight: Genome-scale design can discover high-fitness phenotypes; lysis kinetics and competitive fitness are separable optimization targets.

- Figure 6 | Rapid overcoming of bacterial resistance
  - Results: ΦX174 alone fails on Waa-pathway mutants; a cocktail of 16 designed phages suppresses all three resistant strains within 1–5 passages. Dominant counter-phages (Evo-ΦR1/ΦR2) arise by recombination among designed phages plus new mutations; most changes map to exterior capsid/spike surfaces.
  - Insight: Designed diversity seeds rapid counter-adaptation, suggesting improved resilience for therapeutic cocktails.

Supplementary figures (S1–S21)
- S1: geNomad sharply separates real vs scrambled; generated proteins show natural-like pTM. Insight: Generated sequences exhibit virome-like features and foldability.
- S2: Sources/scale of Microviridae data. Insight: Solid SFT substrate.
- S3: Stable SFT training; ΦX174 positional entropy drops (better learned constraints). Insight: SFT benefits modeling fidelity.
- S4: Overlapping ORFs defeat generic callers; bespoke pseudo-circularization + ORF + PHROGs pipeline. Insight: Dedicated phage genome annotation is required.
- S5: Start/stop one-hot + Gaussian-blur–based “architecture similarity” metric and threshold; UMAP separates ΦX174-like cluster. Insight: Fast, architecture-aware filtering.
- S6: Prompt/temperature sweeps identify regimes that avoid memorization while keeping quality/diversity. Insight: Generation hyperparameters control similarity–diversity trade-offs.
- S7: Final filter criteria and synthesis pass rates. Insight: Manufacturable, screenable library achieved.
- S8: Genome-wide alignment map vs ΦX174. Insight: Visualizes deep whole-genome diversity.
- S9: Extended ΦX174 rebooting/titers. Insight: Experimental pipeline robustness.
- S10: Activity in C, not K-12. Insight: Tropism targeting holds in practice.
- S11: Long-read verification—some minor acquisitions during rebooting/propagation. Insight: Small evolutionary drift possible.
- S12: Individual plaque titrations. Insight: Productivity differences across designs.
- S13: Eight-strain host profiling—unexpected activity in E. coli W. Insight: Tropism is mostly preserved but can broaden subtly.
- S14: Cumulative attribution—many bases unassignable to natural templates. Insight: Genuine sequence novelty.
- S15: Evo-Φ36 J phylogeny/AF3 modeling—distinct capsid-binding mode. Insight: Structural rationale for viable J swap.
- S16: Cryo-EM sample quality and micrographs. Insight: High-quality particle prep.
- S17: Processing/FSC/resolution validation. Insight: Reliable maps.
- S18: Local resolution and model fit. Insight: Key interfaces resolved.
- S19: Interior density views (J and nucleic acids). Insight: Supports packaging/interface interpretations.
- S20: Lysis experiment growth-rate curves. Insight: Quantitative basis for speed/steepness metrics.
- S21: Resistance establishment; waa mutations; counter-phage structure predictions. Insight: Links resistance mechanism to surface remodeling in counter-phages.

Table
- Table S1 | Cryo-EM collection/refinement/validation
  - Results: 2.76 Å (ΦX174) and 2.90 Å (Evo-Φ36) with strong validation metrics.
  - Insight: Near-atomic quality enables confident comparison of capsid–J interfaces.

Appendices/supplementary text
- A. Biosafety and biocontainment
  - Results/Insight: Eukaryotic viruses excluded from training; SFT restricted to bacteriophages; non-pathogenic host; stringent lab practices; tropism constraints enforced. Provides a responsible path for genome-scale generative design.

- B. Methods (key takeaways)
  - Results/Insight:
    - Generation: Evo 1/2 SFT settings; prompt/temperature sweeps; curated Microviridae pipeline.
    - Evaluation: New CDS predictor for overlapping ORFs; architecture score; Shannon diversity, AAI, spike-identity tropism metric.
    - Experiments: Standardized synthesize–reboot–titer–infect–compete–counter-resistance workflow.
    - Structure: AF3 predictions plus Cryo-EM to verify molecular interactions.

- Data and code availability
  - Results/Insight: Full reproducibility with datasets, SFT models, generation/analysis code, and EMDB/PDB depositions.

- Supplementary files (File S1)
  - Results/Insight: Practical resources—design lists, assembly fragments, sequence-verified genomes.

Overall insight
- Combining pretraining, SFT, prompt/temperature control, and layered filters enables whole-genome designs that satisfy function, tropism, and architecture simultaneously.
- Designed genomes deliver sequence/structural novelty, high fitness, and rapid counter-adaptation in experiments.
- The work contributes reusable tools (overlap-aware annotation, architecture scoring, tropism constraints) toward generalizable genome-scale generative design.

<br/>
# refer format:


@article{King2025GenerativePhages,
  title   = {Generative design of novel bacteriophages with genome language models},
  author  = {King, Samuel H. and Driscoll, Claudia L. and Li, David B. and Guo, Daniel and Merchant, Aditi T. and Brixi, Garyk and Wilkinson, Max E. and Hie, Brian L.},
  journal = {bioRxiv},
  year    = {2025},
  date    = {2025-09-17},
  publisher = {Cold Spring Harbor Laboratory},
  doi     = {10.1101/2025.09.12.675911},
  url     = {https://doi.org/10.1101/2025.09.12.675911},
  note    = {Preprint}
}



King, Samuel H., Claudia L. Driscoll, David B. Li, Daniel Guo, Aditi T. Merchant, Garyk Brixi, Max E. Wilkinson, and Brian L. Hie. 2025. “Generative Design of Novel Bacteriophages with Genome Language Models.” bioRxiv, September 17, 2025. https://doi.org/10.1101/2025.09.12.675911.
