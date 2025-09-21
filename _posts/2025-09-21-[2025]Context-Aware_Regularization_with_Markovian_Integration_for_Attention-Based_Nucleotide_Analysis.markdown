---
layout: post
title:  "[2025]Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis"
date:   2025-09-21 21:34:16 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 메서드: CARMANIA는 슬라이딩 윈도우 어텐션(ROPE, FlashAttention) 기반 LLaMA형 트랜스포머에 각 입력 서열의 경험적 바이그램 전이행렬과 모델 예측을 KL로 정렬하는 TM(Transition Matrix) 보조손실을 NT(next-token) 손실과 함께 학습해, 긴 문맥의 마르코프 구조를 명시적으로 주입합니다.

->프리트레인할때 bigram 확률을 같이 학습  


짧은 요약(Abstract) :


- 문제의식: 트랜스포머가 염기서열 분석에 강하지만, 매우 긴 서열에서의 장거리 의존성 포착은 어렵고, 표준 자가-어텐션은 계산량이 O(n^2)로 비효율적입니다. 또, 다음 토큰 예측만으로는 전역적인 전이(transition) 일관성을 명시적으로 보장하지 못합니다.
- 핵심 아이디어: CARMANIA는 표준 다음 토큰(Next-Token, NT) 예측에 전이행렬(Transition Matrix, TM) 손실을 추가한 사전학습 프레임워크입니다. 각 입력 서열로부터 얻은 경험적 n-그램(주로 바이그램) 통계를 모델의 예측 전이와 정렬시키도록 학습시켜, 지역 문맥을 넘어서는 고차 의존성과 전역 전이 일관성을 학습하게 합니다.
- 모델/효율: LLaMA 스타일의 디코더에 슬라이딩-윈도우 어텐션과 RoPE를 적용해 복잡도를 O(n) 수준으로 낮추고, TM 모듈이 전역 핵심 구조(염기의 공출현/전이 패턴)를 보완적으로 유지합니다. 이를 통해 진화적 제약과 기능적 조직화를 반영하는 종-특이적 서열 구조를 학습합니다.
- 평가: 조절요소 예측, 기능 유전자 분류, 분류군 추론, 항생제 내성(AMR) 검출, 생합성 유전자군(BGC) 분류 등 다양한 작업에서 검증.
- 성능 요약:
  - 가장 최근의 장문맥 모델 대비 최소 7% 우수.
  - 짧은 서열 과제에서는 SOTA와 동급 또는 20/40 과제에서 상회하면서 추론 속도는 약 2.5배 빠름.
  - 특히 엔핸서·하우스키핑 유전자 분류에서 큰 향상(엔핸서 MCC 최대 +34%p).
  - TM 손실은 40개 중 33개 작업에서 정확도 향상, 지역 모티프/조절 패턴이 중요한 과제에서 효과가 큼.
- 의미: 지역 어텐션의 한계를 전이행렬 정규화로 보완해, 긴 비암호화 구간이나 저신호 영역에서도 서열 의존 생물학적 특징을 더 견고하게 모델링합니다.



- Motivation: Transformers excel at nucleotide sequence analysis but struggle to capture long-range dependencies at scale; standard self-attention is O(n^2) and does not enforce global transition consistency.
- Key idea: CARMANIA augments next-token (NT) pretraining with a Transition Matrix (TM) loss that aligns predicted token transitions with empirical n-gram statistics (mainly bigrams) from each input sequence. This regularizes the model with a Markovian prior, encouraging higher-order dependencies and globally consistent transitions.
- Model/efficiency: A LLaMA-style decoder with sliding-window attention and RoPE cuts complexity to O(n). The TM module preserves global nucleotide co-occurrence structure, enabling learning of organism-specific sequence organization shaped by evolution and function.
- Evaluation: Tested across regulatory element prediction, functional gene classification, taxonomic inference, antimicrobial resistance detection, and biosynthetic gene cluster classification.
- Results:
  - ≥7% better than the previous best long-context model.
  - Matches SOTA on shorter sequences and surpasses prior results on 20/40 tasks while running ~2.5× faster.
  - Large gains on enhancer and housekeeping gene tasks (up to +34 percentage points MCC for enhancers).
  - TM loss improves 33/40 tasks, especially where local motifs/regulatory patterns drive prediction.
- Takeaway: By combining local attention with TM-based global regularization, CARMANIA more effectively models sequence-dependent biological features and remains robust in non-coding and low-signal regions.


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





1) 개요
- CARMANIA는 길이-효율적인 자기회귀(autoreg.) 트랜스포머에 “전이행렬(Transition Matrix, TM) 손실”을 추가하여, 다음토큰 예측(NT)만으로는 놓치기 쉬운 전역(n-gram) 전이 통계를 명시적으로 맞추도록 정규화하는 사전학습 프레임워크입니다.
- 핵심 아이디어: 각 입력 서열로부터 관측된(경험적) 1차 전이행렬(4×4, 염기쌍 A/T/C/G의 빅람 통계)을 계산해 “정답”으로 두고, 모델이 출력한 분포들로부터 유도한 예측 전이행렬과의 KL 발산을 최소화합니다. 이로써 지역 창 주의(sliding-window attention)가 놓치기 쉬운 장거리 의존성과 전역 전이 일관성을 보완합니다.

2) 모델/아키텍처
- 기반: LLaMA 계열의 디코더형(단방향, causal) 트랜스포머.
- 구조(기본 설정): 총 5개 트랜스포머 블록, 임베딩 차원 1024, MLP 중간차원 4608, 어텐션 헤드 16개(그 중 K/V 헤드 4개), 활성함수 SiLU, 총 파라미터 약 83M.
- 포지셔널 인코딩: RoPE(Rotary Positional Embeddings)로 긴 컨텍스트 일반화를 개선.
- 주의 메커니즘: Sliding-Window Attention(SWA, 창 크기 128)로 전체 복잡도를 전 길이에 대해 선형에 가깝게(고정 윈도우 가정) 낮춤. FlashAttention-2와 로컬 캐싱을 활용해 메모리·속도를 최적화.
- 설계 선택(“Wide”): 얕지만 폭이 넓은(wide) 설계를 채택(대신 더 깊은 층 수는 줄임). DNA에서 중요한 공발생/모티프 패턴을 한 층의 표현력으로 잘 포착하고, 학습 효율과 수렴 속도를 개선.

3) 토크나이징과 전이행렬 타깃 생성
- 토크나이징: 단일 뉴클레오타이드 수준(A/T/C/G)으로 토큰화. 이는 SNP 등 미세 신호 보존과 ORF 파괴 방지에 유리.
- 경험적 전이행렬(정답 TM): 각 입력 시퀀스별로 빅람 빈도를 계산해 4×4 전이확률 행렬을 정규화. 이 “시퀀스 전용” TM이 자가지도 신호로 사용되어, 모델이 서열 특이적 전이 패턴(생물학적으로 의미 있는 염기 전이)을 학습하도록 유도.

4) 전이 텐서와 예측 전이행렬 계산(모델 쪽)
- 예측 분포: t 시점의 모델 출력 Pt ∈ R^V (V=어휘크기=4)에서 각 염기 확률을 사용.
- 통합 n차 전이 텐서: T(n) ∈ R^{V×…×V}를 정의해 n-그램 연쇄 확률을 근사(논문 식(3)). 실제 구현은 1차(빅람, n=2)에 집중해 효율·안정성을 확보.
- 예측 TM 산출: 연속 시점의 확률 벡터로부터 외적(예: Pt ⋅ P_{t+1}^T)을 구해 서열 전체에 걸쳐 합산·정규화하여 모델의 예측 전이행렬을 얻음(부록 B에 구체식). 이 행렬을 경험적 TM과 정렬시키는 것이 TM 손실의 목적.

5) 학습 목표(손실)
- 기본 NT 손실: 표준 자기회귀 언어모델링(다음토큰 음의 로그우도) L_NT.
- TM 손실: 경험적 전이행렬 p_ij와 예측 전이행렬 q_ij 사이의 KL 발산 L_TM = Σ_{i,j} p_ij log(p_ij/q_ij).
- 전체 손실: L_full = L_NT + β L_TM. 기본값 β=1로 두었을 때 두 손실의 스케일이 자연스럽게 맞고, 40개 다운스트림 중 33개에서 성능 향상. β가 과도하면(예: 5) NT 목표를 저해하므로 주의(부록 A.3).

6) 왜 1차 TM인가(고차 TM에 대한 메모)
- 2차(트라이그램) 이상은 생물서열에서 희소도가 크게 증가하여 손실 신호가 불안정해지고 성능이 저하됨(부록 A.4). 1차 TM은 신뢰도 높은 쌍별(염기-다음 염기) 의존성을 포착하며 일관된 성능 향상을 보임.

7) 효율성·장기 의존 처리
- Sliding-window + RoPE + FlashAttention-2로 긴 서열에서 연산·메모리를 절감하면서, TM 손실이 지역창 바깥의 전이 일관성을 보존하도록 보완.
- 창 크기 128로도 전 주의(Full attention)에 근접한 성능을 달성하고, 추론 속도는 길이에 비례해 더 실용적(선형 스케일링 가정).

8) 사전학습 데이터셋
- GRCh38 인간 유전체(약 30억 bp): 10 kbp 조각과 160 kbp 조각 세트.
- Basic Genome(약 100억 bp, 4,634+ 게놈, 10 kbp 조각).
- Scorpio Gene-Taxa(약 5.8×10^8 bp, 2,046 게놈, 4 kbp 조각).
- 모든 시퀀스는 위 토크나이징과 함께 시퀀스별 4×4 TM을 계산해 TM 손실의 정답으로 사용.

9) 학습 설정
- 프레임워크: PyTorch, NVIDIA A100 80GB.
- 에폭: 2; 배치(메모리 맞춤): 4kb=35, 10kb=19, 160kb=1.
- 최적화: AdamW(β=(0.9, 0.999), ε=1e-6, weight decay 0.2), 러닝레이트 5e-4, 코사인 스케줄, 워밍업 400 스텝, 그레이디언트 클립(예: 0.85/2 범위).
- 효율: 약 7.57e-9 GPU-hours/token.

10) 계산 복잡도와 구현 요약
- SWA로 길이 n에 대한 주의 연산을 선형에 가깝게 축소(고정 창 가정), FlashAttention-2로 커널·메모리 사용 최적화.
- 모델 크기 83M으로, 긴 입력에서도 추론 시간이 경쟁 모델 대비 유리(창 기반 주의 + 경량 파라미터 수).

11) 장점과 한계(방법 관점)
- 장점: 전역 전이 패턴(유기체·기능 특이적 구조)을 보존하면서 지역 모티프 기반 신호(조절요소, 프로모터 등)에 강함. 긴 비부호화·저신호 영역에서도 강건한 표현.
- 한계: β 고정은 데이터셋에 따라 과·소가중이 될 수 있음(희귀 모티프 과소학습 위험). 어휘가 큰 도메인(자연어, 화학 SMILES)에서는 TM 계산 메모리 부담 증가 가능(희소 연산·체크포인팅 등 개선 여지).



1) Overview
- CARMANIA is a self-supervised pretraining framework for long DNA sequences that augments standard next-token (NT) prediction with a Transition Matrix (TM) loss. The TM loss aligns the model’s predicted bigram transition matrix with the empirical 4×4 matrix computed from each input sequence, enforcing global transition consistency beyond the local attention window.

2) Model/Architecture
- Backbone: LLaMA-style causal (decoder-only) Transformer.
- Configuration (default): 5 Transformer blocks; d_model=1024; MLP dim=4608; 16 attention heads (4 key/value heads); SiLU activations; ~83M parameters.
- Positional encoding: RoPE to improve long-context extrapolation.
- Attention: Sliding-Window Attention (window 128) for near-linear scaling in sequence length (with fixed window), plus FlashAttention-2 and local caching for speed/memory efficiency.
- “Wide” design: fewer layers but wider dimensions to capture motif/co-occurrence patterns effectively and train efficiently on genomic data.

3) Tokenization and Target Transition Matrix
- Tokenization: single-nucleotide tokens (A/T/C/G) to preserve fine-grained signals (e.g., SNPs) and avoid ORF disruption.
- Empirical TM (target): per-sequence normalized 4×4 bigram transition matrix computed from counts. This serves as the supervision target for TM loss, guiding the model toward biologically meaningful nucleotide transitions.

4) Transition Tensor and Predicted TM (from model)
- Predicted distributions: Pt ∈ R^V (V=4) at each position t from the model.
- Unified n-th order transition tensor: T(n) ∈ R^{V×…×V} approximates n-gram joint dynamics (paper Eq. (3)); in practice, n=2 (bigrams) for stability/efficiency.
- Predicted TM: aggregate outer products of consecutive distributions (e.g., Pt ⋅ P_{t+1}^T) over the sequence and normalize, yielding the model’s bigram matrix. This is matched to the empirical matrix via KL divergence (Appendix B provides the explicit form).

5) Training Objective
- NT loss: standard autoregressive negative log-likelihood, L_NT.
- TM loss: KL divergence between empirical p_ij and model-predicted q_ij bigram matrices, L_TM = Σ_{i,j} p_ij log(p_ij/q_ij).
- Full loss: L_full = L_NT + β L_TM, with β=1 by default. This balancing improved 33/40 downstream tasks; too large β (e.g., 5) can undermine NT learning (Appendix A.3).

6) Why first-order TM
- Higher-order (≥2) transitions are sparse in biological sequences, making the objective unstable and degrading performance. First-order TM reliably captures pairwise dependencies and consistently improves results (Appendix A.4).

7) Long-context efficiency and dependency capture
- Sliding-window attention with RoPE and FlashAttention-2 provides scalable compute, while TM loss preserves global transition patterns beyond the local window. With window size 128, performance approaches full attention with much better runtime.

8) Pretraining datasets
- GRCh38 human genome (~3B bp): 10 kbp and 160 kbp fragments.
- Basic Genome (~10B bp from 4,634+ genomes): 10 kbp fragments.
- Scorpio Gene-Taxa (~580M bp from 2,046 genomes): 4 kbp fragments.
- For all sequences, the per-sequence 4×4 empirical TM is computed to supervise the TM loss.

9) Training setup
- Framework/Hardware: PyTorch, NVIDIA A100 80GB.
- Epochs: 2; batch sizes by length: 4kb=35, 10kb=19, 160kb=1.
- Optimizer/Schedule: AdamW (betas (0.9, 0.999), ε=1e-6, weight decay 0.2), LR 5e-4, cosine schedule, 400 warmup steps, gradient clipping (e.g., 0.85/2 range).
- Efficiency: ~7.57e-9 GPU-hours per token.

10) Complexity and implementation
- With a fixed window, attention cost scales near-linearly in sequence length, and FlashAttention-2 improves kernel efficiency. The 83M-param model yields competitive long-context inference speed compared to prior long-DNA models.

11) Strengths and limitations (method-level)
- Strengths: preserves organism-specific global transition structure and local regulatory motif signals; robust on long non-coding/low-signal regions; improves domain adaptation and long-range memory.
- Limitations: fixed β may need adaptation for rare but informative motifs; TM computation can become memory-intensive for large vocabularies (e.g., natural language, SMILES), motivating sparse/approximate implementations and checkpointing.

원하시면, 수식과 알고리즘 흐름(의사코드)만 따로 요약한 짧은 버전도 제공해 드리겠습니다.


<br/>
# Results



1) 비교 대상(경쟁모델)
- Attention/Transformer 계열
  - CARMANIA(본 논문, 83M, SWA·RoPE·FlashAttention, TM loss)
  - CARMANIA Baseline w/o TM(동일 구조, TM loss 제거)
  - Caduceus-PH / Caduceus-PS(각 1.9M, 양방향, reverse-complement 대칭 유지)
  - Nucleotide Transformer v2(최대 500M)
  - DNABERT-2(117M)
  - Enformer(252M)
- Convolution/State-space 계열
  - HyenaDNA(1.6M, 긴 컨텍스트 CNN 기반)
  - Mamba(상태공간 모델)
- 기타
  - CNN(고전적 컨볼루션 기반)
  - MetaBERTa(BigBird 기반, 35.2M; 일부 비교에서 사용)

2) 평가 데이터셋(테스트 과업)과 메트릭
- Genomics Benchmark(8개 분류 과업, 인간/마우스 조절요소·엔핸서·프로모터 등)
  - 메트릭: Top-1 Accuracy(5-fold CV)
- Nucleotide Transformer Tasks(18개 과업: 히스톤 마커, 조절요소, 스플라이스 사이트)
  - 메트릭: MCC(히스톤/엔핸서), F1(프로모터·스플라이스), Accuracy(“All” 스플라이스)
  - 10-fold CV, seed 평균·표준편차 보고
- Scorpio-Gene-Taxa(유전자–분류군 연계 일반화)
  - 분할: Test(보던 유전자·분류군), Gene-out(안 보던 유전자), Taxa-out(안 보던 문/phyla)
  - 메트릭: Accuracy(Phylum, Class, Order, Family, Gene 레벨)
  - 평가: FAISS 임베딩 최근접 이웃 검색(파인튜닝 없이 표현력 비교)
- AMR(항생제 내성) 분류(보충자료)
  - 과업: Gene Family, Resistance Mechanism, Drug Class
  - 메트릭: Macro F1(FAISS 임베딩 기반 zero-shot 평가)
- BGC(생합성 유전자 클러스터) 다중분류(긴 문맥)
  - 데이터: MiBiG, 100kb로 트렁케이션
  - 메트릭: Accuracy(5-fold CV)
- 추가 지표(모델링/효율/장기 기억)
  - BLEU: 인간 유전체 재구성 품질(언어모델 지표)
  - Perplexity(언급)
  - 추론시간: 길이별(10k~100k bp) 비교
  - Hamming 유사도: 160kb 영역에서 100bp 윈도우의 내부일관성(장거리 유지력)
  - 학습 곁가지: NT loss, TM loss 곡선; FLOPs 상대량

3) 핵심 결과(과업별 비교)
A. Genomics Benchmark(8과업, Accuracy)
- CARMANIA가 4/8 과업 1위, 2과업 동률 1위를 달성.
  - 예시(±표준편차):
    - Coding vs. Intergenic: 0.935 ± 0.001(경쟁모델 최고 0.915)
    - Human NonTATA Promoters: 0.963 ± 0.002(경쟁모델 최고 0.946)
    - Human Enhancer Ensembl: 0.916 ± 0.002(경쟁모델 최고 0.900)
    - Mouse Enhancers: 0.761 ± 0.019(경쟁모델 최고 0.793; 이 과업은 2위권)
- TM loss의 기여: “Baseline w/o TM” 대비 여러 과업에서 상향(예: Human Enhancer Ensembl 0.892→0.916).

B. Nucleotide Transformer Tasks(18과업)
- CARMANIA가 5/18 과업에서 1위, 평균적으로 기존 모델 대비 3%p 이상 상회.
- 엔핸서 계열 과업에서 큰 폭 개선(MCC 기준):
  - ENHANCER: 0.880 ± 0.013 (vs Caduceus-PH 0.546 ± 0.073 등) → 약 0.33~0.34의 절대 개선
  - ENHANCER TYPES: 0.724 ± 0.013 (vs 최고 0.439 ± 0.054 수준)
- 히스톤 마커:
  - H4ac에서 TM loss가 큰 효과: w/ TM 0.606 ± 0.014 vs w/o TM 0.197 ± 0.011
  - H3K36me3, H3K79me3 등 넓게 퍼진 마크는 개선 폭이 상대적으로 제한적(생물학적 특성 반영).

C. Scorpio-Gene-Taxa(임베딩 일반화)
- Test split(보던 분포):
  - Phylum/Class/Order/Family/Gene에서 CARMANIA가 전반 최상 또는 최상 근접:
    - 예: Phylum 0.861, Class 0.768, Order 0.596, Family 0.419, Gene 0.909
    - Baseline w/o TM 대비 Gene 정확도 대폭 상승(0.845→0.909)
- Gene-out(안 보던 유전자): CARMANIA도 견고하나(예: Phylum 0.596 등), 모델별 편차 존재
- Taxa-out(안 보던 문): 수준별 편차가 큼
  - MetaBERTa가 상위 분류(Phylum)에서 높음(0.640) 반면, Gene 수준 일반화는 매우 낮음(0.074)
  - CARMANIA는 Gene 수준에서 가장 높음(0.728)으로 “유전자-분류군” 양쪽 구조를 임베딩에 반영
- t-SNE(그림 5, 8): CARMANIA는 유전자 군집과 분류군 구조를 동시에 보존하여 검색 성능 우수.

D. AMR(보충자료, zero-shot 임베딩 분류, Macro F1)
- CARMANIA가 3개 과업 모두 최고:
  - GeneFamily: 0.733(다음 최고 0.728)
  - Resistance Mechanism: 0.975(다음 최고 0.974)
  - Drug Class: 0.942(다음 최고 0.931)
- 파인튜닝 없이도 표현력이 우수함을 입증.

E. BGC 분류(긴 문맥, 100kb, Accuracy)
- CARMANIA: 0.484 ± 0.330로 최고
  - HyenaDNA: 0.412 ± 0.130
  - Baseline w/o TM: 0.410 ± 0.240
  - Caduceus-PH: 0.326 ± 0.119
- 긴 서열에서 기능적 패턴 포착 능력 우수.

4) 효율·장기 의존성 및 어블레이션
- 추론 시간(10k~100k bp): CARMANIA가 Caduceus 대비 약 2.5배 빠름. HyenaDNA는 소형이라 빠르지만 길이가 커질수록 CARMANIA가 더 효율적이며 성능 우위.
- 장거리 유지력(160kb, Hamming 유사도):
  - HyenaDNA는 원거리에서 유사도 하락, CARMANIA는 전 구간 안정적 유지.
  - TM loss 도입 시 유지력이 추가 향상.
- TM loss 효과(학습/성능):
  - 40개 다운스트림 중 33개 과업에서 향상.
  - BLEU(인간 유전체): SWA+TM 0.77 vs SWA 0.73; Full attention+TM 0.71(성능은 유사하나 BLEU는 SWA+TM가 더 좋음).
  - β 감도: β=1.0이 최적(F1 0.882, BLEU 0.77); β 과도(5.0) 시 성능 저하.
  - 2차(고차) TM는 희소성으로 성능 악화 → 1차 TM가 안정적.
- 주의창(Window) vs Full attention:
  - 창 크기 128에서 Full attention에 근접한 품질 확보, FLOPs 58% 절감.
- 모델/데이터 스케일링:
  - 4M/0.5B토큰 → 83M/10B토큰으로 스케일 시 AMR F1과 BLEU 큰 폭 개선, 상대 FLOPs 1.0 기준 대비 효율 유지.
- 구조 선택(넓은 모델 vs 깊은 모델):
  - 동일 파라미터대(80–83M)에서 “넓은(wide)” 구조가 Gene-Taxa와 AMR 모두에서 더 높은 성능과 더 빠른 학습.

5) 요약적 결론
- CARMANIA는 긴 문맥과 지역 패턴을 동시에 잡는 “TM loss + 창 주의” 결합으로, 규제요소/엔핸서·긴 BGC·유전자–분류군 일반화·AMR 등 광범위 과업에서 일관된 이득을 보임.
- 특히 엔핸서류(MCC)와 Gene 수준 일반화에서 두드러짐.
- 효율 측면에서도 긴 서열에서 빠르고, 창 주의(128)로 Full attention 대비 58% 계산절감하면서 성능 손실이 거의 없음.




1) Baselines
- Transformer/attention family:
  - CARMANIA (this work, 83M; SWA+RoPE+FlashAttention, with TM loss)
  - CARMANIA Baseline w/o TM (same backbone, no TM loss)
  - Caduceus-PH / Caduceus-PS (1.9M each; bi-directional with reverse-complement symmetry)
  - Nucleotide Transformer v2 (up to 500M)
  - DNABERT-2 (117M)
  - Enformer (252M)
- Convolution/state-space family:
  - HyenaDNA (1.6M; long-range CNN)
  - Mamba
- Others:
  - CNN
  - MetaBERTa (BigBird-based, 35.2M; used in some comparisons)

2) Datasets/Tasks and Metrics
- Genomics Benchmark (8 tasks: regulatory elements, enhancers, promoters)
  - Metric: Top-1 Accuracy (5-fold CV)
- Nucleotide Transformer Tasks (18 tasks: histone marks, regulatory annotation, splice sites)
  - Metrics: MCC (histone/enhancer), F1 (promoter/splice), Accuracy (“All” splice)
  - 10-fold CV; report mean ± std
- Scorpio-Gene-Taxa (gene-to-taxon generalization)
  - Splits: Test (seen), Gene-out (unseen genes), Taxa-out (unseen phyla)
  - Metric: Accuracy at Phylum/Class/Order/Family/Gene
  - Evaluation by FAISS nearest-neighbor on frozen embeddings (no fine-tuning)
- AMR (supplementary; zero-shot)
  - Tasks: Gene Family, Mechanism, Drug Class
  - Metric: Macro F1 via FAISS retrieval on frozen embeddings
- BGC classification (long-context)
  - Data: MiBiG; sequences truncated to 100kb
  - Metric: Accuracy (5-fold CV)
- Additional indicators
  - BLEU on human genome reconstruction; Perplexity
  - Inference time vs sequence length (10k–100k bp)
  - Hamming similarity over 160kb windows (long-range retention)
  - Training curves (NT/TM loss), relative FLOPs

3) Main Results
A. Genomics Benchmark (Accuracy)
- CARMANIA ranks first on 4/8 tasks and ties for first on 2 tasks.
  - Examples (mean ± std):
    - Coding vs Intergenic: 0.935 ± 0.001 (prev best 0.915)
    - Human NonTATA Promoters: 0.963 ± 0.002 (prev best 0.946)
    - Human Enhancer Ensembl: 0.916 ± 0.002 (prev best 0.900)
- TM loss benefit is consistent vs the w/o-TM baseline (e.g., Enhancer Ensembl 0.892→0.916).

B. Nucleotide Transformer Tasks (18 tasks)
- CARMANIA leads on 5/18 tasks; on average >3% improvement over baselines.
- Large gains on enhancer tasks (MCC):
  - ENHANCER: 0.880 ± 0.013 vs ~0.546 best baseline → ~0.33–0.34 absolute gain
  - ENHANCER TYPES: 0.724 ± 0.013 vs ~0.439 best baseline
- Histone markers:
  - H4ac shows strong TM effect: 0.606 ± 0.014 with TM vs 0.197 ± 0.011 without TM
  - Marks like H3K36me3/H3K79me3 show smaller gains, consistent with their diffuse nature.

C. Scorpio-Gene-Taxa (embedding generalization)
- Test split: CARMANIA achieves best or near-best at all taxonomic levels; notably Gene accuracy 0.909 (vs 0.845 w/o TM).
- Gene-out: robust but varies across levels.
- Taxa-out: mixed pattern
  - MetaBERTa excels at Phylum (0.640) but fails at Gene (0.074)
  - CARMANIA achieves the best Gene-level accuracy (0.728), indicating better capture of gene–taxon structure.
- t-SNE shows CARMANIA embeddings align both gene identity and taxonomy, aiding retrieval.

D. AMR (supplementary; zero-shot Macro F1)
- CARMANIA leads on all three tasks:
  - GeneFamily 0.733; Mechanism 0.975; Drug Class 0.942
- Demonstrates strong out-of-the-box embedding quality without fine-tuning.

E. BGC classification (long context, Accuracy)
- CARMANIA: 0.484 ± 0.330 (best)
  - HyenaDNA: 0.412 ± 0.130
  - Baseline w/o TM: 0.410 ± 0.240
  - Caduceus-PH: 0.326 ± 0.119
- Indicates superior ability to capture functional signals over 100kb contexts.

4) Efficiency, Long-range Retention, Ablations
- Inference time (10k–100k bp): CARMANIA is ~2.5× faster than Caduceus; more scalable than HyenaDNA at long lengths while outperforming it.
- Long-range retention (160kb Hamming similarity):
  - HyenaDNA degrades at distal regions; CARMANIA remains stable across the entire span.
  - TM loss further improves retention.
- TM loss impact:
  - Improves 33/40 downstream tasks.
  - BLEU: SWA+TM 0.77 vs SWA 0.73; Full attention+TM 0.71 (SWA+TM attains better BLEU with similar downstream accuracy).
  - β=1.0 is best (F1 0.882; BLEU 0.77); higher β (e.g., 5.0) harms performance.
  - Second-order TM hurts due to sparsity; first-order is robust and biologically meaningful.
- Sliding-window vs Full attention:
  - Window size 128 matches full attention closely while cutting FLOPs by ~58%.
- Scaling and architecture:
  - 83M/10B tokens vs 4M/0.5B tokens: large gains in AMR F1 and BLEU without disproportionate compute.
  - Wide shallow model (83M) outperforms deeper narrow (≈80M) on both in-domain and OOD tasks, and trains faster.

5) Takeaways
- CARMANIA’s TM loss + windowed attention captures both local motif statistics and long-range dependencies efficiently.
- Delivers strong, often state-of-the-art results across diverse genomic tasks, with standout gains on enhancer prediction and gene-level generalization.
- Achieves favorable speed–accuracy trade-offs on very long sequences, making it practical for large-scale genomics.

끝.


<br/>
# 예제





1) 사전학습(Pre-training) 단계: 입력/출력/손실, 예시
- 입력 데이터 소스
  - GRCh38 인간 유전체: 약 3Gbp, 10kbp/160kbp 조각
  - Basic Genome: 약 10Gbp, 4,634개 이상의 다양한 종, 10kbp 조각
  - Scorpio Gene-Taxa: 약 580Mbp, 2,046개 종, 4kbp 조각(패딩 포함)
- 토크나이즈와 전이행렬 타깃 생성
  - 토큰: 단일 뉴클레오타이드 단위 {A, T, C, G}
  - 각 입력 서열마다 빅그램 빈도로 4×4 1차 전이행렬(행별 정규화) p_ij 생성
    - p_ij = count(i→j) / Σ_x count(i→x)
- 모델 입출력(헤드 2개)
  1) Next-Token(오토리그레시브) 헤드
     - 입력: 길이 L의 서열 x = (x1,…,xL)
     - 출력: 각 위치 t의 다음 토큰 확률 Pt ∈ R^4 (A/T/C/G에 대한 분포)
     - 손실: L_NT = −Σ_t log Pθ(xt | x<t)
  2) Transition-Matrix(TM) 정렬 헤드
     - 입력: 동일한 서열 배치의 모델 예측 확률들 {Pt}
     - 출력: 모델이 유도한 전이 분포 q_ij (예측 TM)
     - 손실: L_TM = Σ_{i,j} p_ij log(p_ij / q_ij) (KL 발산)
  - 최종 손실: L_Full = L_NT + β L_TM (논문 기본 설정 β=1)
- 사전학습 입력/출력 예시
  - 입력 서열(10kbp 예): “ATCG…(총 10,000 nt)…GAT”
  - 토큰 단위: [A, T, C, G]
  - 타깃 전이행렬(예, A행): [P(A→A), P(A→T), P(A→C), P(A→G)] = A 다음 글자들의 상대빈도
  - 모델 출력:
    - 모든 t에 대해 P_t = [P(A), P(T), P(C), P(G)]
    - 이들로부터 유도한 q_ij (예측 TM)
  - 손실 계산:
    - NT 손실: 각 위치의 정답 토큰 로그우도 합
    - TM 손실: 입력 서열로 만든 p_ij와 모델 예측 q_ij의 KL

2) 다운스트림 태스크들: 데이터, 입력/출력 형식, 스플릿/메트릭, 예시
A. Genomics Benchmarks(8개 과제, 인간/생쥐 기반 분류)
- 입력 길이: 70–4,776 bp
- 출력 레이블: 과제별 이진(대부분) 혹은 다중 클래스
- 학습/평가: 5-fold 교차검증, 지표는 정확도(Accuracy)
- 예시
  - Human Enhancer Ensembl
    - 입력: 400 bp 내외 유전체 조각
    - 출력: y ∈ {0(비-엔핸서), 1(엔핸서)}
  - Human NonTATA Promoters
    - 입력: 400 bp 내외 조각
    - 출력: y ∈ {0, 1}
  - Human vs. Worm(이진 종 분류)
    - 입력: 1 kb 내외 조각
    - 출력: y ∈ {human, worm}

B. Nucleotide Transformer tasks(18개 과제)
- 입력 길이: 200–500 bp
- 레이블/메트릭
  - 히스톤 마커들(H3, H3K14ac, H3K4me3 등): 이진 분류, MCC 보고
  - ENHANCER, ENHANCER TYPES: 이진/다중, MCC 보고
  - PROMOTER(ALL/NONTATA/TATA): 이진(혹은 ALL은 정확도 보고)
  - Splice Site(ALL/ACCEPTOR/DONOR):
    - ACCEPTOR/DONOR: 이진(F1 보고)
    - ALL: 다중 분류(정확도 보고)
- 학습/평가: 10-fold 교차검증
- 예시
  - H3K4ME3
    - 입력: 300 bp
    - 출력: y ∈ {0(해당 마크 없음), 1(있음)}
  - Splice DONOR
    - 입력: 200–300 bp
    - 출력: y ∈ {0, 1}
  - ENHANCER TYPES
    - 입력: 300–500 bp
    - 출력: y ∈ {다중 클래스 ID}

C. Scorpio-Gene-Taxa(유전자-택사 매핑, 리트리벌 기반)
- 입력 길이: 4 kbp(원본 114–13,227 bp를 4kb로 패딩/절단)
- 출력: 임베딩 최근접 이웃(FAISS)으로부터 예측
  - Gene(497+ 클래스), Taxonomy(Phylum, Class, Order, Family 각각 다중 클래스)
- 스플릿: Test(보던 gene/taxa), Gene-out(미본 gene), Taxa-out(미본 phyla)
- 지표: Accuracy
- 예시
  - 입력: 4,000 bp 서열
  - 출력: Gene = g_k, Phylum = p_i, Class = c_j, Order = o_m, Family = f_n (각 과제별 다중 클래스 예측)

D. AMR(항균내성) 분류 3과제(파인튜닝 없이 임베딩 리트리벌)
- 입력 길이: 211–5,274 bp
- 출력 레이블: Gene Family, Resistance Mechanism, Drug Class(모두 다중 클래스)
- 방법: 사전학습된 임베딩 고정, FAISS 최근접 이웃으로 분류
- 지표: F1 Macro
- 예시
  - 입력: 1,100 bp 서열
  - 출력: GeneFamily = GF_k, Mechanism = RM_m, DrugClass = DC_n

E. BGC(생합성 유전자 군집) 대분류
- 데이터: MiBiG 기반, 서열 길이 다양(평균 377kb), 100kb로 절단해 입력
- 출력: 8개 대사산물 클래스 중 하나(다중 분류)
- 학습/평가: 5-fold 교차검증, 지표는 정확도
- 예시
  - 입력: 100,000 bp
  - 출력: Class ∈ {0,1,…,7}

3) 장거리 서열 유지(Long-range retention) 평가: 입력/출력/절차
- 입력: 인간 유전체 160 kbp 구간 50개(독립)
- 절차:
  - 각 구간 내에서 100 bp 윈도우를 2,000 bp 간격으로 추출
  - 모델 예측 서열과 원본을 해밍 유사도(Hamming similarity)로 비교
- 출력: 위치별 평균 유사도 프로파일(거리 증가에 따른 기억 유지 곡선)

4) 추가 실험 설계에서의 입출력 변화 요약
- TM 손실 유무(β=0 vs 1): 입력/출력 포맷은 동일, 손실 항만 변경
- 주의창 크기(윈도우 128 vs Full Attention): 입력/출력 포맷은 동일, 계산/성능 트레이드오프 비교
- 메트릭:
  - 사전학습 중: NT loss, TM loss, (일부) BLEU, Perplexity
  - 다운스트림: Accuracy, F1 Macro, MCC 등 태스크별 상이

참고 수치/세팅(본문 근거)
- 입력 길이 범위: 70–160,000 bp(과제에 따라 다름)
- 토큰 집합: 4개(A/T/C/G), 전이행렬 4×4
- 스플릿/검증:
  - Genomics Benchmarks: 5-fold CV
  - Nucleotide Transformer tasks: 10-fold CV
  - Scorpio-Gene-Taxa: Test/Gene-out/Taxa-out
  - BGC: 5-fold CV
- 대표 지표: Accuracy, F1 Macro, MCC(히스톤/엔핸서), Splice(ALL은 정확도, ACCEPTOR/DONOR는 F1), 장거리 유지(해밍 유사도), 사전학습 중 BLEU/Perplexity도 보고




Below is a structured, example-driven description (grounded only in the provided paper) of what the model sees as inputs and produces as outputs in both pre-training and downstream evaluations, including tasks, splits, and metrics.

1) Pre-training: inputs/outputs/losses, with examples
- Data sources
  - GRCh38 human genome: ~3 Gbp, 10 kbp and 160 kbp fragments
  - Basic Genome: ~10 Gbp across 4,634+ genomes, 10 kbp fragments
  - Scorpio Gene-Taxa: ~580 Mbp from 2,046 species, 4 kbp fragments (padded as needed)
- Tokenization and TM targets
  - Tokens: single nucleotides {A, T, C, G}
  - For each input sequence, compute a 4×4 first-order transition matrix (row-normalized) by bigram counts:
    - p_ij = count(i→j) / Σ_x count(i→x)
- Model heads and outputs
  1) Next-token (autoregressive) head
     - Input: sequence x = (x1,…,xL)
     - Output: at each position t, a 4-dim probability vector Pt over {A,T,C,G}
     - Loss: L_NT = −Σ_t log Pθ(xt | x<t)
  2) Transition-Matrix (TM) alignment head
     - Input: the model’s per-position distributions {Pt}
     - Output: a predicted transition distribution q_ij (model-derived TM)
     - Loss: L_TM = Σ_{i,j} p_ij log(p_ij / q_ij) (KL divergence)
  - Final objective: L_Full = L_NT + β L_TM (β=1 in the paper)
- Concrete pre-training example
  - Input (10 kbp): “ATCG…(10,000 nt)…GAT”
  - Target TM (e.g., row A): [P(A→A), P(A→T), P(A→C), P(A→G)] from bigram frequencies
  - Model outputs:
    - For each t, P_t = [P(A), P(T), P(C), P(G)]
    - A predicted TM q_ij computed from {P_t}
  - Loss:
    - NT loss over all positions
    - TM loss between empirical p_ij and predicted q_ij

2) Downstream tasks: data, input/output format, splits/metrics, examples
A. Genomics Benchmarks (8 tasks)
- Input length: 70–4,776 bp
- Labels: binary or multi-class depending on task
- Protocol: 5-fold cross-validation, report accuracy
- Examples
  - Human Enhancer Ensembl
    - Input: ~400 bp segment
    - Output: y ∈ {0(non-enhancer), 1(enhancer)}
  - Human NonTATA Promoters
    - Input: ~400 bp segment
    - Output: y ∈ {0,1}
  - Human vs. Worm
    - Input: ~1 kb segment
    - Output: y ∈ {human, worm}

B. Nucleotide Transformer tasks (18 tasks)
- Input length: 200–500 bp
- Labels/metrics
  - Histone marks (e.g., H3, H3K14ac, H3K4me3): binary, report MCC
  - ENHANCER / ENHANCER TYPES: binary/multi-class, report MCC
  - PROMOTER (ALL/NONTATA/TATA): binary (ALL may be reported with accuracy)
  - Splice sites (ALL/ACCEPTOR/DONOR):
    - ACCEPTOR/DONOR: binary (report F1)
    - ALL: multi-class (report accuracy)
- Protocol: 10-fold cross-validation
- Examples
  - H3K4ME3
    - Input: ~300 bp sequence
    - Output: y ∈ {0,1}
  - Splice DONOR
    - Input: ~200–300 bp
    - Output: y ∈ {0,1}
  - ENHANCER TYPES
    - Input: ~300–500 bp
    - Output: y ∈ {multi-class ID}

C. Scorpio-Gene-Taxa (gene-to-taxonomy association via retrieval)
- Input length: 4 kbp (original 114–13,227 bp, padded/trimmed to 4 kbp)
- Outputs via FAISS nearest-neighbor on frozen embeddings:
  - Gene (497+ classes) and Taxonomy (Phylum, Class, Order, Family; each multi-class)
- Splits: Test (seen genes/taxa), Gene-out (unseen genes), Taxa-out (unseen phyla)
- Metric: accuracy
- Example
  - Input: 4,000 bp
  - Outputs: Gene = g_k, Phylum = p_i, Class = c_j, Order = o_m, Family = f_n

D. AMR (antimicrobial resistance) classification: 3 tasks (no fine-tuning; retrieval)
- Input length: 211–5,274 bp
- Outputs: Gene Family, Resistance Mechanism, Drug Class (all multi-class)
- Method: frozen embeddings + FAISS nearest neighbor
- Metric: F1 Macro
- Example
  - Input: 1,100 bp
  - Outputs: GeneFamily = GF_k, Mechanism = RM_m, DrugClass = DC_n

E. BGC (biosynthetic gene cluster) classification (long-context)
- Input: up to 100,000 bp (all sequences truncated to 100 kbp)
- Output: one of 8 metabolite classes (multi-class)
- Protocol: 5-fold cross-validation
- Metric: accuracy
- Example
  - Input: 100,000 bp
  - Output: Class ∈ {0,1,…,7}

3) Long-range sequence retention evaluation: inputs/outputs/procedure
- Inputs: fifty independent 160 kbp human genome segments
- Procedure:
  - Within each segment, extract 100 bp windows every 2,000 bp
  - Compare model-predicted sequences to the original using Hamming similarity
- Output: position-wise average similarity profile across segments (a curve vs genomic distance)

4) Notes on ablations (inputs/outputs unchanged; only setup differs)
- TM loss on/off (β=0 vs 1): same input/output; only loss composition differs
- Attention window size (128 vs full): same input/output; compare efficiency/accuracy trade-offs
- Metrics:
  - During pre-training: NT loss, TM loss, sometimes BLEU/Perplexity
  - Downstream: Accuracy, F1 Macro, MCC (histone/enhancer), Splice (F1 or accuracy), long-range retention (Hamming similarity)

Key ranges/settings (from the paper)
- Input lengths: 70–160,000 bp depending on task
- Token set: 4 nucleotides, TM is 4×4
- Splits/evaluation:
  - Genomics Benchmarks: 5-fold CV
  - NT tasks: 10-fold CV
  - Scorpio-Gene-Taxa: Test/Gene-out/Taxa-out
  - BGC: 5-fold CV
- Representative metrics: Accuracy, F1 Macro, MCC, BLEU/Perplexity (pre-training), Hamming similarity (retention)

원하시면 위 형식에 맞춰 특정 과제(예: “Human Enhancer Ensembl” 또는 “Splice DONOR”)만 따로 골라, 샘플 입력·출력 템플릿(JSON/CSV 등)과 전처리/배치 구성 예시까지 만들어 드리겠습니다.

<br/>


메서드: CARMANIA는 슬라이딩 윈도우 어텐션(ROPE, FlashAttention) 기반 LLaMA형 트랜스포머에 각 입력 서열의 경험적 바이그램 전이행렬과 모델 예측을 KL로 정렬하는 TM(Transition Matrix) 보조손실을 NT(next-token) 손실과 함께 학습해, 긴 문맥의 마르코프 구조를 명시적으로 주입합니다.
결과: 40개 유전체 과제에서 이전 최상 장문맥 모델 대비 ≥7% 향상, 단문맥 과제에서는 20/40개 과제에서 최고 성능(또는 동률)이며 추론 속도는 ~2.5배 빨라지고, 장거리 일관성과 기억 유지도 개선됩니다.
예시: 인핸서·하우스키핑 유전자 분류에서 MCC 최대 +34%p 향상, AMR·Gene–Taxa 임베딩 검색에서 우수, BGC(최대 100kb) 다중분류에서 48.4% 정확도로 HyenaDNA·Caduceus 등을 상회합니다.

Method: CARMANIA augments a LLaMA-style sliding-window Transformer (RoPE, FlashAttention) with a Transition Matrix (TM) auxiliary loss that KL-aligns model bigram transitions to each sequence’s empirical matrix, jointly with next-token loss to inject explicit Markovian structure over long contexts.
Results: Across 40 genomics tasks it improves over the previous best long-context model by ≥7%, tops or matches SOTA on 20/40 short-context tasks while running ~2.5× faster, and shows stronger long-range coherence and memory.
Examples: It yields up to +34% absolute MCC on enhancer/housekeeping gene classification, strong AMR and Gene–Taxa embedding retrieval, and 48.4% accuracy on long-context BGC classification, surpassing HyenaDNA and Caduceus.

<br/>
# 기타

[다이어그램/피규어]
- Figure 1 (프레임워크 개요)
  - 결과: LLaMA형 디코더에 슬라이딩 윈도우 어텐션(SWA)과 전이행렬(TM) 모듈을 결합. O(n^2)→O(n) 수준의 효율과 n-그램 전이 일관성 병행 학습.
  - 인사이트: TM 손실이 지역 어텐션의 한계를 보완해, 긴 문맥에서의 전이 패턴(유기체/조직 특이적)을 안정적으로 포착.

- Figure 2 (TM 손실 유무에 따른 학습 곡선)
  - 결과: NT(다음 토큰) 손실은 두 설정(β=1 vs 0) 모두 유사하게 감소. TM 손실은 β=1에서 초반부터 지속 하락, β=0에서는 부분적/암시적 정렬.
  - 인사이트: TM 손실이 토큰 예측을 방해하지 않으면서 전이 구조를 명시적으로 정렬시켜, 전역 통계 학습을 가속.

- Figure 3 (윈도우 크기 vs 성능, 추론 속도 비교)
  - 결과: 윈도우 128이 풀 어텐션에 근접한 성능. 추론 시간은 CARMANIA가 Caduceus 대비 ~2.5배 빠르고, 길이 증가에도 안정적. HyenaDNA는 빠르지만 장문맥 성능은 열세.
  - 인사이트: TM 손실 덕분에 작은 윈도우로도 복잡한 의존성을 포착. 실사용 관점에서 정확도-효율 트레이드오프가 우수.

- Figure 4 (장거리 시퀀스 유지력)
  - 결과: 160kb 영역에서 CARMANIA는 전 구간 높은 유사도 유지, HyenaDNA는 원거리에서 급락. TM 손실 적용 시 유지력 추가 개선.
  - 인사이트: SWA+TM 조합이 장거리 세부 정보 유지에 유리. 재귀/고정 깊이 구조(예: HyenaDNA)는 장거리 세부 보존에 불리.

- Figure 5 (유전자 t-SNE)
  - 결과: CARMANIA는 유전자별로 콤팩트한 군집을 형성하면서 계통 분류 신호도 잘 정렬. HyenaDNA는 유전자 분리만 강하고, MetaBERTa는 분류학 정렬은 강하나 유전자 분리는 약함.
  - 인사이트: CARMANIA 임베딩이 유전자-계통 양축 구조를 동시에 학습하여 범용 분류/검색에 유리.

- Figure 6 (부록: HyenaDNA의 TM 손실 영향–학습 곡선)
  - 결과: HyenaDNA는 TM 손실을 넣으면 수렴이 악화되고, TM 구조 학습도 불안정.
  - 인사이트: 컨볼루션 기반 모델은 Markovian 전이 정렬을 주 손실과 함께 학습하기 어려움. 어텐션 기반이 TM 통합에 더 적합.

- Figure 7 (부록: TM 손실 성능 영향)
  - 결과: CARMANIA는 TM 손실로 평균 F1 Macro 개선. HyenaDNA는 개선 불가 혹은 악화.
  - 인사이트: TM 손실은 어텐션 모델에서 효과적이며, 컨볼루션 모델에는 비선호적.

- Figure 8 (부록: 계통 t-SNE)
  - 결과: 상위 10개 문(phyla)에서 CARMANIA 임베딩이 계통 구조를 잘 보존하고 상호 근접 관계도 자연스럽게 반영.
  - 인사이트: TM 보강 표현은 유사 계통군 간 전역 구조까지 유지.

[테이블]
- Table 1 (어블레이션: TM 손실, 어텐션 종류)
  - 결과: TM 손실로 F1(0.873→0.882), BLEU(0.73→0.77) 상승. 풀 어텐션은 FLOPs 1.58배 증가 대비 성능 이득 미미.
  - 인사이트: 실용적으론 윈도우 128 + TM 손실이 가장 균형적.

- Table 2 (Genomic Benchmarks)
  - 결과: 8개 태스크 중 4개 1위, 2개 동률. 특히 Coding vs. Intergenic, Human Enhancer Ensembl, NonTATA Promoter 등에서 강세.
  - 인사이트: 단일 뉴클레오타이드 토크나이즈 + TM 손실이 인간 유전체 감독 학습 전이에서 견고.

- Table 3 (Nucleotide Transformer 과업)
  - 결과: 18개 중 5개 1위, 평균적으로 기존 대비 3%+ 향상. Enhancer 계열 MCC 대폭 개선(최대 +34%p). H4ac에서도 상대적 +40% 이상 개선.
  - 인사이트: 지역 모티프/조절 패턴 의존 태스크에 TM 손실 효과 큼. 확산적 히스톤 마크(H3K36me3 등)는 개선 제한적.

- Table 4 (Scorpio-Gene-Taxa)
  - 결과: 메인 테스트/Taxa-out에서 최고. Gene-out도 상위권. 유전자 일반화 vs 계통 일반화 간 트레이드오프 확인.
  - 인사이트: CARMANIA는 유전자-계통 계층 구조를 함께 포착하며, 단일 축에 치우친 모델 대비 균형적 일반화.

- Table 5 (BGC 분류, 100kb)
  - 결과: 정확도 0.484로 최고(HyenaDNA 0.412 대비 +7%p).
  - 인사이트: 단백질 번역/도메인 주석 없이도 뉴클레오타이드 서열만으로 장거리 기능 패턴 포착.

- Table 6 (부록: AMR 임베딩 검색)
  - 결과: 3개 태스크 모두 최상. TM 손실로 추가 소폭 향상.
  - 인사이트: 사전학습 임베딩 자체의 표현력이 높아 비미세조정 전이에도 강함.

- Table 7 (부록: β 감도)
  - 결과: β=1.0에서 F1/BLEU 최고. β 과대(=5.0) 시 성능 저하.
  - 인사이트: NT와 TM의 균형이 중요. 고정 β=1로도 스케일이 자연 정렬됨.

- Table 8–9 (부록: 고차 TM)
  - 결과: 1차 TM는 일관된 개선, 2차 TM는 전반적 성능 저하.
  - 인사이트: 생물 서열에서 고차 전이 희소성이 커 손실 신호가 불안정. 신뢰도 높은 쌍(빅람) 전이가 최적.

- Table 10 (부록: 와이드 vs 딥)
  - 결과: 파라미터 유사(80M대)에서 와이드(5층, 큰 히든/MLP)가 딥(24층)보다 정확도/수렴/시간에서 우세.
  - 인사이트: DNA 모티프/공출현 포착에 폭이 효과적. 학습 효율까지 향상.

- Table 11 (부록: 학습 하이퍼파라미터 범위)
  - 결과/인사이트: 재현성 확보용 세부 설정 공개.

- Table 12 (부록: 모델 스케일링)
  - 결과: 4M/0.5B 토큰→83M/10B 토큰으로 F1 0.809→0.860, BLEU 0.41→0.82, 상대 FLOPs=1로 기준화.
  - 인사이트: 규모 확장으로 표현력 대폭 개선, 계산 증가 대비 효율적 개선.

- Table 13 (부록: 시퀀스 길이 스케일링)
  - 결과: 10kb→160kb 학습으로 F1/BLEU 모두 상승, FLOPs 동일.
  - 인사이트: 장문맥에 노출될수록 더 먼 의존성 학습에 유리.

- Table 14–15 (부록: 데이터셋 통계)
  - 결과: 사전학습/다운스트림 데이터 다양성, 길이/라벨 스펙 제시.
  - 인사이트: 다양한 길이와 분포로 학습하여 광범위 전이를 뒷받침.

[어펜딕스 B: TM 구현 검증]
- 결과: 모델 확률로 구성한 예측 전이행렬이, 원-핫(지시함수)으로 대체 시 실제 빅람 빈도 행렬과 행정규화 수준에서 동치.
- 인사이트: TM 손실이 단순 빈도 기반 직관과 합치되며, 학습 가능한 확률화 일반화로 작동함을 수학적으로 확인.

핵심 종합 인사이트
- TM 손실은 장문맥 모델링을 크게 강화하되, 1차(빅람) 수준이 가장 안정적이고 실용적.
- 작은 윈도우의 지역 어텐션과 결합해 정확도-효율을 동시에 달성.
- 컨볼루션 기반 장문맥 구조는 TM 정렬의 이점을 누리기 어려움.
- 와이드 아키텍처, 장문맥 사전학습, 중간 규모(83M) 모델이 실제 유전체 과업에서 성능/속도 균형이 가장 뛰어남.



[Diagrams/Figures]
- Figure 1 (Framework)
  - Result: LLaMA-style decoder with sliding-window attention and a transition-matrix (TM) head. Achieves O(n) attention cost while enforcing n-gram consistency.
  - Insight: TM loss complements local attention to capture organism-/context-specific long-range transition patterns efficiently.

- Figure 2 (Training curves with/without TM loss)
  - Result: NT loss drops similarly with β=1 or 0; TM loss decreases steadily only when β=1, while β=0 shows partial, implicit alignment.
  - Insight: TM loss explicitly organizes global transition structure without hurting token-level learning.

- Figure 3 (Window size vs performance; inference speed)
  - Result: Window 128 ≈ full attention in quality. CARMANIA is ~2.5× faster than Caduceus at long lengths; HyenaDNA is fast but underperforms on long sequences and is still slower than CARMANIA.
  - Insight: TM enables small windows to capture complex dependencies; best accuracy-efficiency trade-off in practice.

- Figure 4 (Long-range retention)
  - Result: Over 160 kb segments, CARMANIA maintains high sequence similarity; HyenaDNA degrades with distance. TM loss further improves retention.
  - Insight: SWA+TM excels at preserving long-range detail; fixed-depth recurrence struggles.

- Figure 5 (t-SNE of genes)
  - Result: CARMANIA clusters by gene while aligning with taxonomy; HyenaDNA separates genes but weak taxonomy alignment; MetaBERTa shows the opposite tendency.
  - Insight: CARMANIA captures both gene identity and taxonomic structure for robust transfer.

- Figure 6 (Appx: HyenaDNA training with TM)
  - Result: TM loss harms convergence in HyenaDNA; transition learning remains unstable.
  - Insight: Convolutional long-range models are less amenable to Markovian regularization than attention-based ones.

- Figure 7 (Appx: Effect of TM loss)
  - Result: TM loss boosts CARMANIA’s macro F1; no gain for HyenaDNA.
  - Insight: TM regularization is effective with attention, not with convolutions.

- Figure 8 (Appx: t-SNE of phyla)
  - Result: CARMANIA embeddings preserve inter-phyla structure and proximity.
  - Insight: TM-augmented embeddings maintain global taxonomic geometry.

[Tables]
- Table 1 (Ablations: TM loss, attention)
  - Result: TM improves F1 (0.873→0.882) and BLEU (0.73→0.77). Full attention adds 58% FLOPs with minor gains.
  - Insight: Window 128 + TM is the most practical configuration.

- Table 2 (Genomic Benchmarks)
  - Result: Best on 4/8 tasks and tied on 2; strong on coding vs intergenic, human enhancers, non-TATA promoters.
  - Insight: Nucleotide-level tokenization + TM transfers well to human genomic tasks.

- Table 3 (Nucleotide Transformer tasks)
  - Result: Best on 5/18 tasks; >3% average gain. Large MCC jumps on enhancers (up to +34% absolute) and H4ac (40%+ relative).
  - Insight: TM excels where local motifs/regulatory patterns dominate; diffuse marks show limited improvement.

- Table 4 (Scorpio-Gene-Taxa)
  - Result: Top accuracy on main test and Taxa-out; strong on Gene-out. Reveals gene vs taxonomy generalization trade-off.
  - Insight: CARMANIA captures both gene and taxonomic hierarchies, balancing generalization better than baselines.

- Table 5 (BGC classification, 100 kb)
  - Result: CARMANIA reaches 0.484 accuracy, beating HyenaDNA (0.412) by >7%p.
  - Insight: Pure nucleotide models can capture long-range functional patterns without external annotation tools.

- Table 6 (Appx: AMR embedding retrieval)
  - Result: Best F1 macro across all three tasks; TM provides small additional gains.
  - Insight: Strong off-the-shelf embeddings enable zero-finetune transfer.

- Table 7 (Appx: β sensitivity)
  - Result: β=1.0 yields best F1/BLEU; β=5.0 degrades.
  - Insight: Balancing NT and TM is critical; β=1 works well by natural loss scaling.

- Table 8–9 (Appx: Higher-order TM)
  - Result: First-order TM consistently helps; second-order TM hurts.
  - Insight: Higher-order transitions are too sparse in biology, making the loss unstable; bigrams are robust and informative.

- Table 10 (Appx: Wide vs Deep)
  - Result: With similar params, “wide” (few layers, larger dims) outperforms “deep” in accuracy, convergence, and time.
  - Insight: Wider layers are better for motif/co-occurrence and efficiency in DNA tasks.

- Table 11 (Appx: Training hyperparameters)
  - Result/Insight: Full details for reproducibility.

- Table 12 (Appx: Model scaling)
  - Result: 4M/0.5B tokens → 83M/10B tokens improves F1 (0.809→0.860) and BLEU (0.41→0.82) with reasonable compute scaling.
  - Insight: Capacity and data scale translate to markedly better representations.

- Table 13 (Appx: Sequence length scaling)
  - Result: Training on 160 kb > 10 kb in both F1 and BLEU with same FLOPs.
  - Insight: Exposure to longer contexts strengthens long-range dependency learning.

- Table 14–15 (Appx: Dataset stats)
  - Result: Diverse pretraining and downstream datasets, spanning broad lengths and label spaces.
  - Insight: Diversity underpins robust transfer and evaluation.

[Appendix B: TM implementation check]
- Result: The model-derived transition matrix reduces to the empirical bigram frequency matrix under indicator probabilities and row normalization.
- Insight: TM loss is a learnable generalization of frequency matching, ensuring correctness and interpretability.

Key consolidated insights
- TM loss strongly enhances long-context modeling, with first-order (bigram) providing the most stable and useful signal.
- Combining small-window attention with TM yields optimal accuracy-efficiency trade-offs.
- Convolutional long-range models benefit less (or not at all) from TM regularization.
- A “wide” transformer with long-context pretraining at moderate scale (~83M) achieves an excellent balance of speed and accuracy on real genomic tasks.

<br/>
# refer format:



- BibTeX
@article{Refahi2025CARMANIA,
  title = {Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis},
  author = {Refahi, Mohammadsaleh and Abavisani, Mahdi and Sokhansanj, Bahrad A. and Brown, James R. and Rosen, Gail},
  year = {2025},
  month = jul,
  eprint = {2507.09378},
  archivePrefix = {arXiv},
  primaryClass = {q-bio.GN},
  url = {https://arxiv.org/abs/2507.09378},
  note = {Preprint, version 1, posted July 12, 2025}
}

- 시카고 스타일(Author–Date)  
Refahi, Mohammadsaleh, Mahdi Abavisani, Bahrad A. Sokhansanj, James R. Brown, and Gail Rosen. 2025. “Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis.” arXiv, July 12. https://arxiv.org/abs/2507.09378.



- 시카고 스타일(Notes–Bibliography)  
Refahi, Mohammadsaleh, Mahdi Abavisani, Bahrad A. Sokhansanj, James R. Brown, and Gail Rosen. “Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis.” arXiv, July 12, 2025. https://arxiv.org/abs/2507.09378.
