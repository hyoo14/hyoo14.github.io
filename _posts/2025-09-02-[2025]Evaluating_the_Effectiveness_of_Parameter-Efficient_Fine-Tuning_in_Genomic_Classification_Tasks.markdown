---
layout: post
title:  "[2025]Evaluating the Effectiveness of Parameter-Efficient Fine-Tuning in Genomic Classification Tasks"
date:   2025-09-02 16:10:44 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

사전학습 gLM(NT 50M/100M/250M, DNABERT‑S, DNABERT‑2)을 1,500bp 서열 518만 건에 대해 슈퍼킹덤(4), 문(57), 속(1,884) 분류로 전량 미세조정, 무작위 초기화 후 미세조정, LoRA, (IA)^3, 75% 동결을 비교 학습(최대 5에폭)하고, 평균풀링 기반 계층별 헤드로 균형정확도와 훈련 효율(최대 배치·에폭당 시간)을 평가함.


짧은 요약(Abstract) :

- 배경: 생물학 데이터 분석에 대형 사전학습 언어모델이 널리 쓰이면서, 전체 미세조정의 높은 메모리 비용을 줄이기 위해 PEFT(매개변수 효율 미세조정)가 각광받고 있습니다.
- 문제의식: 기존 연구는 주로 이진 분류에서 PEFT 성능 저하가 거의 없다고 보고했지만, 클래스 수가 큰 과제에서의 효과는 체계적으로 검증되지 않았습니다.
- 연구설계: 유전체 언어모델을 분류 백본으로 사용해, 짧은 서열을 초왕계(superkingdom), 문(phylum), 속(genus) 수준에서 분류하는 계통분류 문제에 대해 PEFT, 전체 미세조정, 부분(일부 동결) 미세조정을 비교했습니다.
- 주요 결과: 분류 공간이 커질수록 PEFT로 학습한 모델은 전체/부분 미세조정 대비 성능이 유의하게 낮았습니다. 또한 사전학습된 모델은 무작위 초기화 모델보다 일관되게 더 나은 성능을 보였습니다.
- 함의: 큰 분류 공간을 가진 생물학적 분류 과제에서는 PEFT가 적합하지 않을 수 있으며, 사전학습의 이점이 분명히 존재합니다.



- Background: Foundation models are widely used for biological tasks, and PEFT is popular to reduce the high memory/storage costs of full fine-tuning.
- Gap: Prior work mostly on binary tasks reported little to no performance loss with PEFT, but its impact on tasks with large label spaces has not been systematically assessed.
- Approach: Using pretrained genomic language models as backbones, the authors compare PEFT, full fine-tuning, and partial fine-tuning for taxonomic classification of sequences at superkingdom, phylum, and genus levels.
- Key findings: As the classification space expands, PEFT-trained models significantly underperform compared to full or partially fine-tuned models. Pretrained models also outperform randomly initialized models across settings.
- Implication: For large label-space tasks like taxonomic classification, PEFT may be inappropriate, and pretraining provides clear benefits.


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



연구 개요와 목표
- 목표: 사전학습된 유전체 언어 모델(gLM)에 대해 다양한 미세조정 전략(전체 미세조정, 무작위 초기화 후 미세조정, PEFT[LoRA/(IA)3], 부분 동결)을 적용하여 계통분류(초계, 문, 속) 과제에서의 성능과 효율성을 체계적으로 비교.
- 핵심 질문: 대규모 클래스 공간(계층적 라벨)에서 PEFT가 전체 미세조정 대비 성능·효율(메모리/배치 크기/시간) 이점을 제공하는가?

데이터셋
- 출처: BERTax 논문 부속 데이터(BERTax: Taxonomic Classification of DNA Sequences with Deep Neural Networks)에서 공개된 사전학습용 공개 데이터셋(OSF: https://osf.io/dwkte), 초계(Archaea, Viruses, Eukaryota, Bacteria)별 4개 FASTA 파일.
- 서열 길이: 모든 입력은 고정 길이 1500bp.
- 전처리:
  - 염기서열을 대문자로 표준화.
  - gLM 토크나이저 최대 토큰 길이를 초과하지 않도록 확인(예: NT는 6-mer 토크나이즈, 최대 토큰 길이 1000; 본 데이터는 약 250 토큰 수준).
- 라벨링:
  - ETE3의 ncbi_taxonomy 모듈을 사용해 NCBI Taxonomy(2024년 12월 taxdump 아카이브 기반)로부터 각 시퀀스의 phylum, genus를 조회.
  - phylum 또는 genus 라벨이 없는 시퀀스는 제외.
- 데이터 규모 및 분할:
  - 최종 처리된 전체 데이터: 5,181,880 시퀀스
    - Eukaryota 2,601,890 / Bacteria 1,828,018 / Archaea 524,276 / Viruses 227,696
  - 라벨 분포: 4개 초계, 55개 문, 1,878개 속(총 1,573개 고유 taxonomy ID)
  - 테스트셋: 전체의 2%(103,638), 속 단위로 층화 추출, 모든 실험에서 동일한 홀드아웃으로 평가
  - 검증셋: 전체의 10%, 검증 손실 모니터링에 사용

모델(백본)과 토크나이징
- 사용한 사전학습 gLM(3개 아키텍처, 5개 모델):
  - Nucleotide Transformer (NT): 50M, 100M, 250M 파라미터 규모
  - DNABERT 계열: DNABERT-2-117M, DNABERT-S
- 토크나이징:
  - NT: 6-mer 토크나이저, 최대 토큰 길이 1000(입력 1500bp → 약 250 토큰)
  - DNABERT 계열: 각 모델 기본 토크나이저(사전학습 설정 준수)

분류 아키텍처(계층 라벨용 커스텀 헤드)
- 기본 구조: 사전학습 gLM을 공통 백본으로 사용하고, 초계/문/속 각 레벨에 대해 독립적인 선형(Linear) 분류 헤드 3개를 병렬로 부착.
- 입력 처리:
  - 입력 토큰 ID와 어텐션 마스크를 백본 gLM에 통과시켜 토큰 단위 마지막 히든 상태를 취득.
  - 어텐션 마스크를 가중치로 사용한 평균 풀링(mean pooling)으로 고정 차원 임베딩을 생성(패딩 토큰은 제외).
- 임베딩 차원:
  - NT-50M, NT-100M: 512 차원
  - NT-250M, DNABERT-2, DNABERT-S: 768 차원
- 출력과 손실:
  - 각 헤드가 해당 레벨의 클래스 수로 사상되는 로짓을 산출.
  - 손실 함수는 레벨별 교차엔트로피의 합을 사용(클래스 가중치 미사용; 가중치 사용 시 클래스 간 편차가 증가해 배제).
- 주의: 본 연구의 헤드들은 “독립적”이며, 상위 레벨 예측을 하위 레벨에 조건부로 주입하는 완전 계층형 아키텍처는 적용하지 않음(향후 과제로 남김).

학습 전략(방법론 레퍼토리)
- 전체 미세조정(full fine-tuning): 백본과 분류 헤드를 모두 학습.
- 무작위 초기화 후 미세조정(random init + fine-tune): 사전학습 가중치를 완전히 제거하고, 표준 트랜스포머 초기화 규칙으로 백본의 선형·임베딩 레이어를 N(0, σ^2)로 재설정, LayerNorm 가중치=1/바이어스=0, 선형 바이어스=0으로 초기화한 뒤 전체 미세조정.
- PEFT
  - LoRA: HuggingFace PEFT 라이브러리 사용, 랭크 r를 포함한 하이퍼파라미터를 다양하게 탐색(r=1 및 r≥4 등).
  - (IA)3: HuggingFace PEFT 구현 사용.
- 부분 동결(partial fine-tuning): 모델 파라미터의 75%를 동결하고 나머지 25%만 학습.

최적화, 하이퍼파라미터, 스케줄링
- 공통 설정:
  - 정밀도: FP32
  - 옵티마이저: AdamW
  - 스케줄러: 코사인(cosine) 러닝레이트 스케줄러
  - 가중감쇠(weight decay): 기본 0
  - 워밍업: 미사용(PEFT 탐색에서 워밍업도 시험했으나 유의미한 개선 없음)
  - 에폭: 최대 5 에폭(검증 손실 기준 3 에폭 연속 개선 없으면 조기 종료)
- 초기 학습률(모델별 기본 학습 시):
  - DNABERT-2, DNABERT-S: 2e-5
  - NT-100M, NT-250M: 1e-4
  - NT-50M: 2e-4
- 배치 크기(기본 학습 시, 장비별 per-device):
  - NT-50M: 128
  - NT-100M: 64
  - NT-250M: 64
  - DNABERT-2, DNABERT-S: 16
- PEFT 추가 탐색:
  - 학습률 2e-5 및 1e-4, weight decay 0.1, 워밍업, 최대 20 에폭 등 조합 실험
  - 결과: 손실이 최대 1% 개선되는 정도로, 완전/부분 미세조정 대비 손실 격차 유지(성능/수렴 상 큰 이득 없음)

학습 인프라 및 구현
- 하드웨어:
  - NT 계열: NVIDIA H100 80GB 단일 GPU
  - DNABERT 계열: NVIDIA H100 80GB 4장(데이터 병렬)
- 소프트웨어:
  - HuggingFace Transformers 및 PEFT 라이브러리 사용
  - 코드 공개: https://github.com/jhuapl-bio/microbert

배치 크기 상한 탐색 및 시간 계측(효율 분석용 절차)
- 목적: PEFT로 학습 가능 파라미터 수를 줄였을 때 GPU 메모리 여유가 생겨 배치 크기 상한↑ 및 에폭당 시간↓이 실제로 가능한지 검증.
- 절차:
  - 초기 배치 4로 1에폭 학습 → 배치를 2배씩 증분 → OOM(메모리 한계)까지 반복
  - 각 구성에서 에폭당 평균 학습 시간을 측정
  - 러닝레이트는 배치 크기에 선형 비례하도록 스케일(성능·수렴이 아니라 최대 배치 상한 파악이 목적)
- 요약:
  - 모든 설정에서 배치 상한은 변하지 않음(PEFT가 메모리 상 이득으로 이어지지 않음)
  - (IA)3: 에폭당 시간 1.3%~14.7% 단축
  - LoRA: r=1에서 2.8%~12.2% 단축, r≥4에서는 6.1%~22.4% 증가
  - 75% 동결: 32.4%~48.7%의 일관된 시간 단축

평가
- 검증: 검증 손실로 조기 종료 판단
- 최종 평가: 홀드아웃 테스트셋(2%, 속 단위 층화)에서 성능 평가
- 지표: balanced accuracy(계층 라벨별: 초계, 문, 속)

요약 포인트
- 데이터: 1500bp 고정 길이, 518만 시퀀스, 4/55/1878 클래스(초계/문/속)
- 모델: NT(50M/100M/250M), DNABERT-2, DNABERT-S
- 아키텍처: gLM 백본 + 평균 풀링 + 레벨별 독립 선형 헤드 3개, 손실은 합산
- 전략: 전체 미세조정, 무작위 초기화 후 미세조정, LoRA, (IA)3, 75% 동결
- 최적화: AdamW, FP32, 코사인 스케줄러, 기본 5 에폭(조기 종료)
- 구현/자원: HF Transformers/PEFT, H100(단일 또는 4장), 코드 공개




Study Aim
- Goal: Systematically compare multiple fine-tuning strategies—full fine-tuning, fine-tuning from random initialization, PEFT (LoRA and (IA)3), and partial freezing—for hierarchical taxonomic classification (superkingdom, phylum, genus) using pretrained genomic language models (gLMs).
- Core question: Do PEFT methods retain performance and improve efficiency (memory/batch size/time) for large label spaces compared with full fine-tuning?

Dataset
- Source: Public dataset from the BERTax paper (OSF: https://osf.io/dwkte), provided as four FASTA files by superkingdom (Archaea, Viruses, Eukaryota, Bacteria).
- Sequence length: 1500 bp fixed-length inputs.
- Preprocessing:
  - Uppercase normalization of sequences.
  - Ensured token length does not exceed model maxima (e.g., NT uses 6-mer tokenization with max length 1000; inputs are about 250 tokens).
- Labeling:
  - Used ETE3 ncbi_taxonomy with the December 2024 NCBI taxdump to retrieve phylum and genus labels from taxonomy IDs.
  - Removed sequences missing either phylum or genus labels.
- Scale and splits:
  - Final processed set: 5,181,880 sequences
    - Eukaryota 2,601,890; Bacteria 1,828,018; Archaea 524,276; Viruses 227,696
  - Labels: 4 superkingdoms, 55 phyla, 1,878 genera (1,573 unique taxonomy IDs)
  - Test set: 2% (103,638), stratified by genus; used as a fixed holdout for all runs
  - Validation set: 10% of the full dataset for monitoring validation loss

Backbone Models and Tokenization
- Pretrained gLMs (five models across three architectures):
  - Nucleotide Transformer (NT): 50M, 100M, 250M parameters
  - DNABERT family: DNABERT-2-117M, DNABERT-S
- Tokenization:
  - NT: 6-mer tokenizer, max token length 1000 (about 250 tokens per 1500 bp)
  - DNABERT models: their respective pretrained tokenizers

Classification Architecture (hierarchical labels)
- Base: a shared pretrained gLM backbone with three independent linear classification heads for superkingdom, phylum, and genus.
- Input processing:
  - Feed token IDs and attention masks into the backbone to get the last hidden states.
  - Apply mean pooling using the attention mask to obtain a fixed-size sequence embedding (padding excluded).
- Embedding sizes:
  - NT-50M and NT-100M: 512-dim; NT-250M, DNABERT-2, DNABERT-S: 768-dim
- Outputs and loss:
  - Each head maps to the number of classes at its level to produce logits.
  - Overall loss is the sum of cross-entropy losses from the three levels.
  - No class-weighted loss (it produced uneven per-class performance in preliminary tests).
- Note: Heads are independent; a fully hierarchical model (conditioning lower levels on higher-level outputs) was not used here.

Training Strategies
- Full fine-tuning: update the entire backbone and heads.
- Fine-tuning from random initialization: discard all pretrained weights and reinitialize using standard transformer initialization (linear/embedding layers ~ N(0, σ^2); LayerNorm weights=1, biases=0; linear biases=0), then fine-tune end-to-end.
- PEFT (HuggingFace PEFT library):
  - LoRA with a range of ranks r (including r=1 and r≥4).
  - (IA)3.
- Partial freezing: freeze 75% of the model parameters, train the remaining 25%.

Optimization, Hyperparameters, Scheduling
- Common settings:
  - Precision: FP32
  - Optimizer: AdamW
  - Scheduler: cosine learning-rate scheduler
  - Weight decay: 0 by default
  - Warmup: not used (also tested in PEFT searches; no meaningful benefit)
  - Epochs: up to 5 with early stopping if validation loss fails to improve for 3 consecutive epochs
- Initial learning rates (for base training):
  - DNABERT-2 and DNABERT-S: 2e-5
  - NT-100M and NT-250M: 1e-4
  - NT-50M: 2e-4
- Per-device batch sizes (base training):
  - NT-50M: 128
  - NT-100M: 64
  - NT-250M: 64
  - DNABERT-2 and DNABERT-S: 16
- PEFT hyperparameter search:
  - Explored LRs 2e-5 and 1e-4, weight decay 0.1, warmup, and up to 20 epochs.
  - Findings: at most ~1% loss reduction without closing the gap to full/partial fine-tuning.

Compute and Implementation
- Hardware:
  - NT models: single NVIDIA H100 80GB
  - DNABERT models: 4× NVIDIA H100 80GB
- Software:
  - HuggingFace Transformers and PEFT libraries
  - Code available at https://github.com/jhuapl-bio/microbert

Batch-Size Ceiling Search and Timing (for efficiency analysis)
- Objective: Test whether reducing trainable parameters via PEFT increases batch-size limits and reduces epoch time.
- Procedure:
  - Start with batch size 4, train 1 epoch, then double repeatedly until hitting OOM.
  - Measure average epoch time for each configuration.
  - Scale learning rate linearly with batch size (focus on batch ceiling, not convergence).
- Summary:
  - Batch-size ceilings were unchanged across configurations (no memory headroom benefit from PEFT).
  - (IA)3: 1.3%–14.7% reduction in epoch time.
  - LoRA: 2.8%–12.2% reduction at r=1; 6.1%–22.4% increase for r≥4.
  - 75% freezing: consistent 32.4%–48.7% reduction in epoch time.

Evaluation
- Validation: early stopping based on validation loss.
- Final test: fixed 2% holdout set stratified by genus.
- Metric: balanced accuracy at each taxonomic level (superkingdom, phylum, genus).

Key Takeaways
- Data: 1500 bp, 5.18M sequences, 4/55/1,878 classes (superkingdom/phylum/genus)
- Models: NT (50M/100M/250M), DNABERT-2, DNABERT-S
- Architecture: gLM backbone + mean pooling + 3 independent linear heads; summed cross-entropy loss
- Strategies: full fine-tuning, random-init fine-tuning, LoRA, (IA)3, 75% freezing
- Optimization: AdamW, FP32, cosine LR, up to 5 epochs with early stopping
- Infra: HuggingFace Transformers/PEFT on H100 GPUs; code released


<br/>
# Results


개요
- 과제: 1500bp 고정 길이의 DNA 조각으로 계통분류 상위 3단계(상계, 문, 속)를 예측하는 다중 분류
- 분류 공간: 상계 4클래스, 문 55–57클래스(본문/부록 간 표기 차이), 속 1,878–1,884클래스
- 백본 모델(경쟁모델): Nucleotide Transformer (NT) 50M, 100M, 250M; DNABERT-S; DNABERT-2
- 학습 전략(경쟁 학습법):
  1) 전체 미세튜닝(full fine-tuning)
  2) 랜덤 초기화 후 미세튜닝(사전학습 지식 제거)
  3) LoRA(여러 r 값 실험)
  4) (IA)3
  5) 부분 미세튜닝(모델 가중치 75% 동결)
- 메트릭: 균형 정확도(balanced accuracy)
- 학습 효율성 평가: 에폭당 학습 시간, 가능한 최대 배치 크기

데이터 및 테스트셋
- 원천: BERTax 논문 공개 데이터(각 상계별 FASTA; 전부 1500bp)
- 전처리: 대문자 표준화, NCBI Taxonomy(ETE3)로 문/속 라벨 부여, 라벨 결측 제거
- 총 규모: 5,181,880 서열(진핵 2,601,890; 세균 1,828,018; 고세균 524,276; 바이러스 227,696)
- 라벨 다양성: 4 상계, 55 문, 1,878 속(본문 본문부에는 57/1,884로 표기되기도 함)
- 평가셋: 전체의 2%를 속 레벨로 층화한 홀드아웃 테스트셋 103,638개(모든 실험 공통 사용)
- 검증셋: 10% 분리(검증 손실 모니터링)

분류 성능 결과(테스트셋 기준)
- 전체 미세튜닝(최대 성능 베이스라인)
  - 상계: 모든 모델에서 균형 정확도 >99%
  - 분류 공간 확장에 따른 성능 저하(자연스러움): 상계→문 전환 시 중앙값 기준 1.6%p 하락, 문→속 전환 시 추가로 중앙값 25.8%p 하락
- 사전학습 효과 검증(랜덤 초기화 vs 사전학습 가중치)
  - 모든 백본, 모든 계층에서 랜덤 초기화 모델이 일관되게 열세
  - NT-50M의 속 예측에서 최대 22%p(균형 정확도) 격차
  - 결론: 사전학습은 본 과제(계통분류)에서 실질적 이득을 제공
- PEFT 방법(LoRA, (IA)3)의 분류 성능
  - (IA)3: 상계에서 평균 26%p 낮음, 문에서 평균 83%p 낮음(전체 미세튜닝 대비) — 분류 공간이 커질수록 격차 확대
  - LoRA: 랭크 r을 키우면 점수는 증가하지만 여전히 전체 미세튜닝에 현저히 못 미침(특히 NT-50M 심층 분석에서 확인)
  - 종합: PEFT는 대규모 클래스 공간에서 큰 성능 저하
- 부분 미세튜닝(75% 동결)
  - 전체 미세튜닝에 가장 근접
  - 상계: 0.6%p–2.1%p 하락
  - 속: 5.6%p–12.6%p 하락
  - 결론: 메모리 절감 목적에서 성능-효율 절충안으로 가장 실용적

학습 효율성 결과
- 설정: NT는 H100 80GB 단일 GPU, DNABERT는 H100 4장; 배치 크기 최대화를 위해 배치 4에서 시작해 메모리 한계까지 2배씩 증가, 배치 크기에 비례하여 학습률 선형 스케일링
- 최대 배치 크기: 모든 구성에서 동일 — PEFT가 GPU 메모리 한계를 유의미하게 완화하지 못함
- 에폭당 학습 시간
  - (IA)3: 1.3%–14.7% 단축(소폭)
  - LoRA: r=1에서 2.8%–12.2% 단축 가능하나, r≥4에서는 오히려 6.1%–22.4% 증가
  - 75% 동결: 32.4%–48.7% 일관된 대폭 단축
- 결론: 본 세팅에서 PEFT(LoRA, (IA)3)는 시간/배치 효율 이득이 제한적이거나 불안정했고, 부분 미세튜닝이 가장 확실한 시간 절약을 제공

종합 비교 및 해석
- 성능 우열: 전체 미세튜닝 > 부분 미세튜닝(75% 동결) >> PEFT(LoRA, (IA)3)
- 분류 공간 확대 영향: 모든 방법에서 성능 저하가 있으나, PEFT의 하락 폭이 가장 큼(특히 문/속에서 결정적)
- 사전학습의 가치: 랜덤 초기화 대비 유의미한 성능 향상 — 본 과제에서는 사전학습 이점이 크다는 반증
- 효율성 관점: PEFT는 본 설정에서 배치 한계나 시간에서 뚜렷한 장점 없음. 부분 미세튜닝이 시간 절감과 성능 유지의 균형 측면에서 현실적 대안

실무적 시사점
- 대규모 클래스(예: 문/속) 분류에서는 전체 미세튜닝이 가장 안전하며, 자원 제약 시에는 75% 동결이 차선
- LoRA/(IA)3는 이 과제 유형에서 성능·효율 모두 제한적. 특히 r을 올려 성능을 보완하려 할수록 학습 시간은 악화될 수 있음
- 사전학습을 포기(랜덤 초기화)하는 전략은 권장되지 않음

참고
- 모든 수치는 논문 본문과 그림 1의 서술에서 발췌. 문/속 클래스 수는 본문(57/1,884)과 확장 방법(55/1,878) 간 표기 차이가 존재함(데이터 필터링/스냅샷 차이로 추정).





Scope
- Task: Multi-class taxonomic classification from fixed-length 1500 bp DNA fragments at three ranks: superkingdom, phylum, genus
- Label space: 4 (superkingdom), 55–57 (phylum), 1,878–1,884 (genus)
- Backbone models (competitors): Nucleotide Transformer (NT) 50M, 100M, 250M; DNABERT-S; DNABERT-2
- Training strategies (competitors):
  1) Full fine-tuning
  2) Fine-tuning from randomly initialized weights (no pretraining)
  3) LoRA (multiple r values)
  4) (IA)3
  5) Partial fine-tuning with 75% of weights frozen
- Primary metric: Balanced accuracy
- Efficiency metrics: Time per epoch, maximum achievable batch size

Data and test setup
- Source: BERTax supplemental dataset (four FASTA files by superkingdom; all sequences 1500 bp)
- Preprocessing: Uppercasing; phylum/genus labels via NCBI Taxonomy (ETE3); dropped samples with missing labels
- Size: 5,181,880 sequences total (Eukaryota 2,601,890; Bacteria 1,828,018; Archaea 524,276; Viruses 227,696)
- Label diversity: 4 superkingdoms, 55 phyla, 1,878 genera (note: main text sometimes refers to 57/1,884)
- Evaluation: 2% holdout test set stratified at genus (103,638 sequences), shared across all runs; 10% validation set

Classification performance (on the shared test set)
- Full fine-tuning (best baseline)
  - Superkingdom: >99% balanced accuracy for all models
  - Larger label space reduces performance as expected: median −1.6 percentage points from superkingdom→phylum, and an additional median −25.8 points from phylum→genus
- Effect of pretraining (random init vs pretrained)
  - Randomly initialized models underperform across every backbone and rank
  - Up to 22-point gap at genus for NT-50M
  - Conclusion: Pretraining provides substantial gains for this taxonomic task
- PEFT methods (LoRA, (IA)3)
  - (IA)3: Mean balanced accuracy 26 points lower at superkingdom and 83 points lower at phylum vs full fine-tuning; the gap widens with label space
  - LoRA: Increasing r improves scores but remains far below full fine-tuning (confirmed in-depth on NT-50M)
  - Overall: PEFT struggles severely as the classification space grows
- Partial fine-tuning (75% frozen)
  - Closest to full fine-tuning
  - Superkingdom: −0.6 to −2.1 points; Genus: −5.6 to −12.6 points
  - Practical takeaway: Strong trade-off option when resources are constrained

Training efficiency
- Setup: NT on a single H100 80GB; DNABERTs on 4× H100 80GB; batch size doubled until OOM; learning rate scaled linearly with batch size
- Max batch size: Unchanged across all configurations — PEFT did not relax the memory bound in practice
- Time per epoch
  - (IA)3: 1.3%–14.7% faster (mild)
  - LoRA: 2.8%–12.2% faster at r=1, but 6.1%–22.4% slower when r≥4
  - 75% frozen: 32.4%–48.7% faster consistently
- Conclusion: In this setting, PEFT (LoRA, (IA)3) yields limited or inconsistent efficiency gains; partial freezing delivers the most reliable time savings

Overall comparison and implications
- Performance ranking: Full fine-tuning > 75% frozen >> PEFT (LoRA, (IA)3)
- Sensitivity to label space: All methods degrade with more classes, but PEFT degrades the most—especially at phylum/genus
- Value of pretraining: Strong, consistent benefits vs training from scratch
- Efficiency: PEFT did not improve max batch size and offered limited time gains; partial freezing provided clear, consistent speedups with moderate accuracy cost

Note
- All figures are taken from the paper’s narrative and Figure 1. The discrepancy in phylum/genus class counts (57/1,884 vs 55/1,878) appears to arise from dataset filtering/snapshot differences across sections.


<br/>
# 예제



1) 데이터와 태스크 요약
- 입력(공통): 길이 1,500bp의 DNA 염기서열 문자열(‘A/C/G/T’). 모든 서열은 대문자로 표준화.
- 출력(공통): 동일한 입력에 대해 3개의 계통분류 학습 목표를 동시에 예측
  - 슈퍼킹덤(superkingdom): 4개 클래스 {Archaea, Bacteria, Eukaryota, Viruses}
  - 문(phylum): 다중 클래스(논문 본문에는 57개, Extended Methods의 실제 전처리 결과에는 55개)
  - 속(genus): 다중 클래스(본문 1,884개, Extended Methods 전처리 결과 1,878개)
- 태스크 정의: 하나의 1,500bp 서열을 넣으면, 세 계층(superkingdom, phylum, genus)에 대한 독립적인 다중분류를 동시에 수행하는 멀티-헤드 분류. 각 헤드는 로짓을 출력하고, 각 레벨별 교차엔트로피 손실을 합산해 학습. 헤드 간 조건화는 하지 않음(완전 계층 모델은 향후 과제로 명시).
- 평가 지표: 각 레벨별 Balanced Accuracy(클래스 불균형 보정).

2) 데이터 소스, 규모, 전처리, 분할
- 소스: BERTax 논문 공개 데이터(FASTA) 4개 파일
  - Archaea_db.fa, Bacteria_db.fa, Eukaryota_db.fa, Viruses_db.fa
- 서열 길이: 모두 1,500bp 고정.
- 전처리:
  - 모든 염기를 대문자로 통일.
  - NCBI Taxonomy(로컬 taxdump, 2024-12 버전)로부터 각 서열의 taxonomy ID를 사용해 phylum·genus 이름을 조회(ETE3 ncbi_taxonomy 모듈 이용).
  - phylum 또는 genus 라벨이 없는 서열은 제외.
- 최종 처리된 전체 데이터 규모(Extended Methods 기준):
  - 총 5,181,880 서열
    - Eukaryota 2,601,890 / Bacteria 1,828,018 / Archaea 524,276 / Viruses 227,696
  - 유니크 taxonomy ID: 1,573
  - 클래스 수: superkingdom 4, phylum 55, genus 1,878
- 데이터 분할:
  - 테스트: 전체의 2%(103,638 서열), genus 기준 계층적 층화(stratified by genus)
  - 검증: 전체의 10%
  - 학습: 나머지(약 88%)
- 참고: 본문에서는 분류공간을 4/57/1,884로 기술하고, Extended Methods의 실제 전처리 결과는 4/55/1,878로 기재되어 있습니다. 실험 중 일부 필터링·제외 규칙 차이로 생긴 수치 차이로 보이며, 최종 처리된 데이터셋 기준 수치는 55/1,878입니다.

3) 입력 인코딩과 모델 입출력 형태
- 토크나이저/토큰 길이(예: Nucleotide Transformer, NT):
  - 6-mer 토크나이저 사용, 최대 토큰 길이 1000.
  - 본 작업의 1,500bp 입력은 예상 토큰 수 약 250개(비중첩 6-mer 기준)로 최대 길이 한도보다 충분히 짧음.
- 백본과 임베딩:
  - 마지막 히든 스테이트에 대해 attention mask를 적용한 mean pooling으로 서열 임베딩 생성.
  - 임베딩 차원: NT-50M/NT-100M은 512, NT-250M·DNABERT-2·DNABERT-S는 768.
- 분류 헤드(3개 독립 선형층):
  - superkingdom 헤드: 출력 로짓 차원 4
  - phylum 헤드: 출력 로짓 차원 55(또는 57)
  - genus 헤드: 출력 로짓 차원 1,878(또는 1,884)
- 손실: 각 레벨별 Cross-Entropy Loss를 계산해 합산(클래스 가중치 미사용).

4) 모델·학습 전략(입·출력은 동일, 학습 방식만 다름)
- 공통 하이퍼파라미터(요약): FP32, AdamW, 코사인 러닝레이트 스케줄러, 무 가중감쇠, 최대 5 epoch(early stopping: val loss 3 epoch 연속 미개선 시).
- 전략 종류(출력 태스크는 동일):
  - Full fine-tuning(전체 미세조정)
  - 랜덤 초기화 후 전체 미세조정(Pretraining 지식 제거 검증 목적)
  - LoRA(여러 r 설정)
  - (IA)3
  - Partial fine-tuning(모델 가중치 75% 동결)
- 백본 모델: NT 50M/100M/250M, DNABERT-2, DNABERT-S.

5) 학습/평가 “입력-출력” 구체 예시
주의: 아래 예시는 형식을 보여주기 위한 예시입니다. 라벨 id, 구체 라벨명은 데이터 전처리 시점의 클래스 인덱싱에 따라 달라질 수 있습니다.

- 단일 샘플 예시
  - 입력 x
    - raw_dna: “ATGC…(총 1,500bp)”
    - token_ids: 길이 약 250(예: NT 6-mer 비중첩 토큰화) 또는 모델별 토큰화 방식에 따른 길이
    - attention_mask: token_ids와 동일 길이
  - 정답 y
    - y_superkingdom: 클래스명 “Bacteria”, 클래스 id 예: 1
    - y_phylum: 클래스명 예 “Proteobacteria”(가령), 클래스 id 예: 17
    - y_genus: 클래스명 예 “Escherichia”(가령), 클래스 id 예: 542
  - 모델 출력(로짓)
    - logits_superkingdom: 실수 벡터 길이 4
    - logits_phylum: 실수 벡터 길이 55(또는 57)
    - logits_genus: 실수 벡터 길이 1,878(또는 1,884)
  - 손실
    - loss = CE(logits_superkingdom, y_superkingdom)
            + CE(logits_phylum, y_phylum)
            + CE(logits_genus, y_genus)

- 배치 예시(배치 크기 B)
  - 입력
    - token_ids: 크기 [B, T] (T는 모델별 토큰 길이, NT 예시 T≈250)
    - attention_mask: 동일 크기 [B, T]
  - 정답
    - y_superkingdom: [B] 범주형 인덱스
    - y_phylum: [B] 범주형 인덱스
    - y_genus: [B] 범주형 인덱스
  - 출력
    - logits_superkingdom: [B, 4]
    - logits_phylum: [B, 55] 또는 [B, 57]
    - logits_genus: [B, 1,878] 또는 [B, 1,884]

- 테스트 데이터/평가
  - 입력 포맷: 학습과 동일(1,500bp DNA → 토큰화/마스킹)
  - 출력/지표: 각 레벨에 대한 예측을 집계하여 Balanced Accuracy 산출
  - 테스트 셋: 전체의 2%인 103,638 서열(Genus 기준 계층적 층화)

6) “구체적 태스크”를 한 문장으로 정리
- 동일한 1,500bp DNA 서열 입력에 대해, 세 가지 계통학적 레벨(superkingdom/문/속)의 라벨을 동시에 예측하는 다중-헤드, 대규모 다중분류 문제이며, 각 레벨은 독립적으로 학습되고 평가됩니다.

7) 간단 요약 예시 3건
- 예시 A
  - 입력: 1,500bp 서열
  - 정답: superkingdom=Bacteria, phylum=Firmicutes(또는 Bacillota), genus=Bacillus
  - 모델 출력: 각 레벨 로짓 → 각 레벨 softmax argmax = 위 정답
- 예시 B
  - 입력: 1,500bp 서열
  - 정답: superkingdom=Eukaryota, phylum=Ascomycota, genus=Saccharomyces
- 예시 C
  - 입력: 1,500bp 서열
  - 정답: superkingdom=Viruses, phylum=예: Cressdnaviricota(바이러스 분류), genus=예: Geminivirus
(주의: 위 라벨명 예시는 데이터셋 내 실제 포함 여부와 클래스 id 매핑이 달라질 수 있는 가상의 예시입니다. 실제 학습에서는 전처리된 클래스 사전과 id를 사용합니다.)






1) Data and tasks overview
- Input (common): A DNA sequence of length 1,500 bp (characters A/C/G/T). All sequences are uppercased.
- Output (common): Three independent categorical predictions for the same input
  - Superkingdom: 4 classes {Archaea, Bacteria, Eukaryota, Viruses}
  - Phylum: multi-class (57 in the main text; 55 in the actually processed dataset per Extended Methods)
  - Genus: multi-class (1,884 in the main text; 1,878 in the processed set per Extended Methods)
- Task definition: Given one 1,500 bp sequence, perform three independent multi-class classifications simultaneously via three heads (superkingdom, phylum, genus). Heads are not conditioned on each other; losses are summed across heads.
- Metric: Balanced Accuracy for each rank.

2) Source, scale, preprocessing, and splits
- Source: Four FASTA files from the BERTax paper dataset:
  - Archaea_db.fa, Bacteria_db.fa, Eukaryota_db.fa, Viruses_db.fa
- Sequence length: fixed 1,500 bp.
- Preprocessing:
  - Uppercase normalization.
  - Retrieve phylum and genus from taxonomy IDs using ETE3 (local NCBI taxdump, 2024-12).
  - Exclude sequences missing phylum or genus labels.
- Final processed dataset (Extended Methods):
  - 5,181,880 sequences total
    - Eukaryota 2,601,890 / Bacteria 1,828,018 / Archaea 524,276 / Viruses 227,696
  - 1,573 unique taxonomy IDs
  - Classes: superkingdom 4, phylum 55, genus 1,878
- Splits:
  - Test: 2% (103,638 sequences), stratified by genus
  - Validation: 10%
  - Training: ~88%
- Note on class counts: The main text references 4/57/1,884, while the Extended Methods (final processed data) report 4/55/1,878. This discrepancy likely reflects filtering decisions; experiments are conducted on the processed dataset.

3) Input encoding and model I/O shapes
- Tokenization (example: Nucleotide Transformer, NT):
  - 6-mer tokenizer, max token length 1000.
  - A 1,500 bp input yields roughly 250 tokens if using non-overlapping 6-mers, well below the max length.
- Backbone and embedding:
  - Mean pooling over the last hidden states (masked by attention) to obtain a fixed-size sequence embedding.
  - Embedding dim: 512 for NT-50M/NT-100M; 768 for NT-250M, DNABERT-2, DNABERT-S.
- Classification heads (3 independent linear layers):
  - Superkingdom head: logits size 4
  - Phylum head: logits size 55 (or 57)
  - Genus head: logits size 1,878 (or 1,884)
- Loss: sum of per-rank Cross-Entropy losses (no class weighting).

4) Models and training strategies (same I/O; different optimization)
- Common setup (summary): FP32, AdamW, cosine LR scheduler, no weight decay, up to 5 epochs with early stopping (no val loss improvement for 3 consecutive epochs).
- Strategies:
  - Full fine-tuning
  - Full fine-tuning from random initialization (to ablate pretraining)
  - LoRA (various r)
  - (IA)3
  - Partial fine-tuning (freeze 75% of weights)
- Backbones: NT 50M/100M/250M, DNABERT-2, DNABERT-S.

5) Concrete input-output examples
Note: These examples are illustrative of format only. Class ids and names depend on the processed class dictionary.

- Single-sample example
  - Input x
    - raw_dna: “ATGC…(1,500 bp)”
    - token_ids: length ~250 (e.g., NT with non-overlapping 6-mers), model-dependent otherwise
    - attention_mask: same length as token_ids
  - Target y
    - y_superkingdom: e.g., “Bacteria”, class id e.g., 1
    - y_phylum: e.g., “Proteobacteria”, class id e.g., 17
    - y_genus: e.g., “Escherichia”, class id e.g., 542
  - Model outputs (logits)
    - logits_superkingdom: float vector of length 4
    - logits_phylum: float vector of length 55 (or 57)
    - logits_genus: float vector of length 1,878 (or 1,884)
  - Loss
    - loss = CE(logits_superkingdom, y_superkingdom)
            + CE(logits_phylum, y_phylum)
            + CE(logits_genus, y_genus)

- Batch example (batch size B)
  - Input
    - token_ids: [B, T] (T is token length, e.g., ≈250 for NT)
    - attention_mask: [B, T]
  - Targets
    - y_superkingdom: [B] categorical indices
    - y_phylum: [B] categorical indices
    - y_genus: [B] categorical indices
  - Outputs
    - logits_superkingdom: [B, 4]
    - logits_phylum: [B, 55] or [B, 57]
    - logits_genus: [B, 1,878] or [B, 1,884]

- Test-time/evaluation
  - Inputs: identical to training
  - Outputs/metric: predictions aggregated to compute Balanced Accuracy per rank
  - Test set: 103,638 sequences (2%), stratified by genus

6) One-sentence task definition
- A multi-head, large multi-class classification problem that, for a single 1,500 bp DNA input, predicts three taxonomic ranks (superkingdom, phylum, genus) independently and simultaneously.

7) Three short illustrative cases
- Example A
  - Input: 1,500 bp DNA
  - Targets: superkingdom=Bacteria, phylum=Firmicutes (aka Bacillota), genus=Bacillus
- Example B
  - Input: 1,500 bp DNA
  - Targets: superkingdom=Eukaryota, phylum=Ascomycota, genus=Saccharomyces
- Example C
  - Input: 1,500 bp DNA
  - Targets: superkingdom=Viruses, phylum=e.g., Cressdnaviricota, genus=e.g., Geminivirus
(Again, these labels are illustrative; actual inclusion and id mapping depend on the processed dataset.)

<br/>
# 요약


- 메서드: 사전학습 gLM(NT 50M/100M/250M, DNABERT‑S, DNABERT‑2)을 1,500bp 서열 518만 건에 대해 슈퍼킹덤(4), 문(57), 속(1,884) 분류로 전량 미세조정, 무작위 초기화 후 미세조정, LoRA, (IA)^3, 75% 동결을 비교 학습(최대 5에폭)하고, 평균풀링 기반 계층별 헤드로 균형정확도와 훈련 효율(최대 배치·에폭당 시간)을 평가함. 
- 결과: 전량 미세조정이 항상 최고 성능이었고, 라벨 공간이 커질수록 PEFT 성능 저하가 커졌으며, 사전학습 가중치는 무작위 초기화 대비 속 수준에서 최대 22% 성능 우위를 보였고, PEFT는 배치 한계나 시간 이점을 제공하지 못함. 
- 예시: NT‑50M에서 LoRA는 r을 늘려도 속 분류에서 전량 미세조정보다 크게 뒤졌고, (IA)^3는 평균 균형정확도가 슈퍼킹덤에서 26%·문에서 83% 낮았으며, 부분 동결은 슈퍼킹덤 0.6–2.1%·속 5.6–12.6% 낮지만 훈련시간을 32.4–48.7% 단축함.

- Methods: Five pretrained gLMs (NT 50M/100M/250M, DNABERT‑S, DNABERT‑2) were trained on 5.18M 1,500‑bp sequences for superkingdom (4), phylum (57), and genus (1,884) using full fine‑tuning, random‑init fine‑tuning, LoRA, (IA)^3, and 75% layer freezing; models used mean‑pooled embeddings with independent heads, up to 5 epochs, evaluating balanced accuracy and training efficiency (max batch, time/epoch). 
- Results: Full fine‑tuning consistently performed best; PEFT degraded more as label space grew, pretraining beat random initialization by up to 22% at genus, and PEFT gave no advantage in batch limit or training time overall. 
- Example: On NT‑50M, increasing LoRA r still lagged far behind full fine‑tuning at genus; (IA)^3 averaged 26% lower balanced accuracy at superkingdom and 83% lower at phylum, while partial freezing was closer (−0.6–2.1% at superkingdom, −5.6–12.6% at genus) and reduced training time by 32.4–48.7%.

<br/>
# 기타



피규어(Figure)
- Figure 1a: Phylum 분류 성능(밸런스드 정확도) — 사전학습 가중치로 파인튜닝한 모델이 무작위 초기화 후 파인튜닝한 모델을 모든 아키텍처에서 일관되게 상회
  - 결과: 무작위 초기화 모델은 전 범주에서 성능 열세. 본문 서술에 따르면(랭크 전체 관찰) NT-50M은 genus에서 최대 22%p 낮음.
  - 인사이트: gLM 사전학습의 유효성이 분류 과제(특히 계통분류)에서 뚜렷이 확인됨. “사전학습이 이점이 없다”는 주장에 반례.

- Figure 1b: Superkingdom 분류 성능 — 학습 전략별 비교(PEFT: LoRA, (IA)3; 부분 동결; 풀 파인튜닝)
  - 결과: PEFT는 풀 파인튜닝 대비 큰 폭으로 열세. (IA)3의 평균 밸런스드 정확도는 superkingdom에서 약 26%p, phylum에서 약 83%p 낮음. 75% 레이어 동결은 풀 파인튜닝과 가장 근접하나, 분류공간이 커질수록 격차 확대(예: genus에서 5.6–12.6%p 하락).
  - 인사이트: 분류 클래스 수가 커질수록(상위→하위 계급) PEFT 손실이 급격히 커짐. 성능-효율 타협점은 “부분 동결”이 상대적으로 유리.

- Figure 1c: Genus 분류 성능(NT-50M) — LoRA 하이퍼파라미터 r 스케일링
  - 결과: r 증가에 따라 LoRA 성능은 개선되지만, 여전히 풀 파인튜닝 대비 크게 낮음.
  - 인사이트: LoRA 용량을 늘려도 대규모 클래스 공간(1,800+ 클래스)에서는 한계가 뚜렷. PEFT가 구조적으로 요구 표현력을 충분히 확보하지 못함을 시사.

- Figure 1d: 학습 효율(에폭당 시간, 배치 크기 한계)
  - 결과: PEFT로 학습 가능한 최대 배치 크기는 변하지 않음(H100 80GB 기준). 시간 이득도 제한적 혹은 역효과:
    - (IA)3: 1.3–14.7% 단축
    - LoRA: r=1일 때 2.8–12.2% 단축, r≥4에서는 6.1–22.4% 증가(오히려 느려짐)
    - 부분 동결: 32.4–48.7% 크게 단축
  - 인사이트: “적은 학습 파라미터=훨씬 빠른 학습”이 성립하지 않음. 실무적 관점에서 속도·자원 측면의 이점은 부분 동결이 가장 큼.

부록/확장 방법(Extended Methods)
- Taxonomic Data
  - 결과 요약: BERTax 공개 데이터(1500bp 고정 길이)를 사용·전처리. NCBI Taxonomy(2024-12)에서 phylum/genus 레이블 매핑. 결측 레이블 데이터 제외.
  - 데이터 규모: 총 5,181,880 서열(진핵 2,601,890; 세균 1,828,018; 고세균 524,276; 바이러스 227,696). 라벨: 4 superkingdom, 55 phylum, 1,878 genus. 테스트셋 2%(Genus 기준 층화, 103,638개), 검증 10%.
  - 인사이트: 대규모·불균형·계층형 라벨 구조에서 일반화 성능을 평가할 수 있는 설정.

- Model Architectures
  - 결과 요약: 사전학습 gLM 백본 위에 독립적인 3개 분류 헤드(superkingdom/phylum/genus) 추가. 마지막 히든 스테이트에 mean pooling 후 각 헤드에 전달. 손실은 각 레벨 CE 합. 클래스 가중치 사용은 오히려 성능 불균형 유발하여 미적용.
  - 인사이트: 완전 계층 의존 구조(상위 출력이 하위에 영향)는 미도입. 단순하고 재현 가능한 멀티헤드 셋업으로 비교의 공정성 확보.

- Training and Resources
  - 결과 요약: NT-50M/100M/250M은 H100 80GB 단일 GPU(배치 128/64/64, LR 2e-4/1e-4/1e-4), DNABERT-2/‑S는 H100×4(배치 16, LR 2e-5). 최대 5 에폭, 조기 종료. FP32, weight decay 0, cosine 스케줄. PEFT에 대해 warmup, wd=0.1, 다른 LR, 최대 20에폭 등 탐색했으나 유의미 이득 없음(오히려 15에폭 추가해도 손실 1% 감소에 그치며 풀/부분 학습 대비 여전히 열세).
  - 무작위 초기화 실험: 임베딩·선형 레이어 재초기화, LayerNorm은 표준 설정. 사전학습 지식 완전히 제거.
  - 인사이트: 동일 자원 하에서 PEFT가 배치 한계나 수렴 속도를 실질적으로 개선하지 못함이 반복 확인. 사전학습의 득이 크며, “부분 동결”이 현실적 효율 대안.

표/보충자료
- 본문에 표는 제시되지 않음. “Supplementary dataset(실험 결과 전체)”이 별도 제공(링크만 언급).
- 인사이트: 수치 전모는 보충자료에 있으나, 본문·피규어만으로도 결론(PEFT의 성능·효율 열세, 사전학습의 중요성, 부분 동결의 실용성)이 명확.

추가 참고
- 코드 공개: https://github.com/jhuapl-bio/microbert

핵심 인사이트 총정리
- PEFT(LoRA, (IA)3)는 클래스 공간이 커질수록(특히 genus) 정확도 격차가 크게 벌어짐.
- 학습 효율 측면에서도 배치 크기 상향이나 시간 단축 이득이 제한적(LoRA는 r↑ 시 오히려 느려질 수 있음).
- 사전학습의 이점이 분명하며, 동일 자원에서 성능-속도 균형은 “부분 동결 파인튜닝”이 가장 우수.




Figures
- Figure 1a: Phylum-level balanced accuracy — fine-tuning from pretrained weights vs. fine-tuning from random initialization
  - Result: Randomly initialized models underperform across all architectures. The main text also notes up to a 22 pp drop at genus for NT-50M.
  - Insight: Pretraining clearly benefits taxonomic classification, contradicting claims that it offers no advantage.

- Figure 1b: Superkingdom-level accuracy — training strategies compared (PEFT: LoRA, (IA)3; partial freezing; full fine-tuning)
  - Result: PEFT lags far behind full fine-tuning. (IA)3 shows mean balanced accuracy lower by ~26 pp at superkingdom and ~83 pp at phylum. Freezing 75% of layers is closest to full fine-tuning but degrades as label space grows (e.g., 5.6–12.6 pp lower at genus).
  - Insight: As the label space expands, PEFT performance drops sharply. Partial freezing offers the best accuracy-efficiency trade-off.

- Figure 1c: Genus-level accuracy (NT-50M) — LoRA with varied rank r
  - Result: Increasing r improves LoRA performance but remains well below full fine-tuning.
  - Insight: Even with larger adapter capacity, PEFT struggles on very large label spaces (~1,800+ classes).

- Figure 1d: Training efficiency (time/epoch, batch size limit)
  - Result: PEFT does not increase the max feasible batch size (H100 80GB). Time gains are small or negative:
    - (IA)3: 1.3–14.7% faster
    - LoRA: 2.8–12.2% faster at r=1, but 6.1–22.4% slower at r≥4
    - Partial freezing: 32.4–48.7% faster
  - Insight: Fewer trainable parameters did not translate to higher throughput. In practice, partial freezing delivered the most consistent speedups.

Appendices/Extended Methods
- Taxonomic Data
  - Results: Used BERTax dataset (fixed 1500 bp), mapped phylum/genus via NCBI Taxonomy (Dec 2024). Removed sequences lacking labels.
  - Scale: 5,181,880 sequences total (Eukaryota 2.60M; Bacteria 1.83M; Archaea 0.52M; Viruses 0.23M), covering 4 superkingdoms, 55 phyla, 1,878 genera. Test set: 2% stratified by genus (103,638 seqs); validation: 10%.
  - Insight: Large, imbalanced, hierarchical labels enable robust assessment of generalization in realistic settings.

- Model Architectures
  - Results: Pretrained backbones with three independent heads (superkingdom, phylum, genus). Mean pooling over last hidden states; total loss = sum of cross-entropy losses for each rank. Class weighting was avoided due to uneven per-class performance in trials.
  - Insight: Kept the heads independent (not fully hierarchical) for clear, fair comparisons across training strategies.

- Training and Resources
  - Results: NT models on single H100 80GB (batch 128/64/64, LR 2e-4/1e-4/1e-4); DNABERT models on 4×H100 (batch 16, LR 2e-5); up to 5 epochs with early stopping; FP32, no weight decay, cosine schedule. PEFT hyperparameter sweeps (warmup, wd=0.1, alt LRs, up to 20 epochs) yielded no meaningful gains.
  - Random initialization: Reinitialized embeddings/linear layers; standard LayerNorm/bias settings; eliminated all pretraining knowledge.
  - Insight: Under matched resources, PEFT neither raised batch limits nor materially improved time-to-train; pretraining advantage is robust; partial freezing is the practical efficiency winner.

Tables/Supplementary
- No tables in the text. A supplementary dataset with complete results is referenced but not included here.
- Insight: Even without the supplement, figures and narrative suffice to establish the main conclusions: PEFT underperforms on large label spaces and offers limited training-efficiency gains, while pretraining and partial freezing are beneficial.

Key takeaways
- PEFT methods (LoRA, (IA)3) degrade markedly as the label space grows (especially at genus).
- Training efficiency gains from PEFT are limited; LoRA can even slow training at higher ranks r.
- Pretraining is clearly beneficial; partial freezing provides the best balance of accuracy and speed under fixed hardware.

<br/>
# refer format:



- BibTeX
@article{Berman2025PEFTGenomic,
  title   = {Evaluating the Effectiveness of Parameter-Efficient Fine-Tuning in Genomic Classification Tasks},
  author  = {Berman, Daniel and Jimenez, Daniel and Ta, Stanley and Merritt, Brian and Ratcliff, Jeremy and Narayan, Vijay},
  journal = {bioRxiv},
  year    = {2025},
  month   = aug,
  day     = {26},
  publisher = {Cold Spring Harbor Laboratory},
  doi     = {10.1101/2025.08.21.671544},
  url     = {https://doi.org/10.1101/2025.08.21.671544},
  note    = {Preprint}
}



- 시카고 스타일(Author–Date)   
Berman, Daniel, Daniel Jimenez, Stanley Ta, Brian Merritt, Jeremy Ratcliff, and Vijay Narayan. 2025. “Evaluating the Effectiveness of Parameter-Efficient Fine-Tuning in Genomic Classification Tasks.” bioRxiv, August 26. https://doi.org/10.1101/2025.08.21.671544.
