---
layout: post
title:  "[2025]Overcoming Vocabulary Mismatch: Vocabulary-agnostic Teacher Guided Language Modeling"  
date:   2025-07-27 01:38:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 

teacher student간 vocab mismatch 해결 위해 alignment 함, 이 때 효과적(student의) 학습 위해 teacher guided loss 사용   



짧은 요약(Abstract) :    

최근에는 성능이 좋은 대형 언어 모델(teacher)을 작은 모델(student)의 훈련을 도와주는 방식이 주류가 되고 있습니다. 하지만 teacher와 student 간의 vocabulary mismatch, 즉 서로 다른 토크나이저나 어휘 집합을 사용하는 경우에는 학습이 어려워지는 문제가 발생합니다. 이 논문에서는 이러한 문제를 해결하기 위해 VocAgnoLM이라는 새로운 프레임워크를 제안합니다. 이 방법은 크게 두 가지 핵심 기술을 포함합니다:

Token-level Lexical Alignment: 서로 다른 어휘 체계를 사용하는 모델 간에도 토큰 수준에서 정렬을 가능하게 함.

Teacher Guided Loss: teacher 모델의 손실 값을 이용하여 student 모델이 효과적으로 학습할 수 있도록 유도.

실험 결과, Qwen2.5-Math-Instruct (student 모델인 TinyLlama와 어휘 중복률이 6%에 불과한 모델)를 teacher로 사용했을 때, naive한 연속 사전학습보다 46% 향상된 성능을 보였습니다. 이 방법은 더 강력한 teacher 모델을 사용할수록 student 모델 성능도 함께 향상되는 효과를 보이며, 어휘가 달라도 teacher 지도를 효과적으로 받을 수 있는 강건한 해법을 제시합니다.



Using large teacher models to guide the training of smaller student models has become the prevailing paradigm for efficient and effective learning. However, vocabulary mismatches between teacher and student language models pose significant challenges in language modeling, resulting in divergent token sequences and output distributions. To overcome these limitations, we propose Vocabulary-agnostic Teacher Guided Language Modeling (VocAgnoLM), a novel approach that bridges the gap caused by vocabulary mismatch through two key methods: (1) Token-level Lexical Alignment, which aligns token sequences across mismatched vocabularies, and (2) Teacher Guided Loss, which leverages the loss of teacher model to guide effective student training. We demonstrate its effectiveness in language modeling with 1B student model using various 7B teacher models with different vocabularies. Notably, with Qwen2.5-Math-Instruct, a teacher model sharing only about 6% of its vocabulary with TinyLlama, VocAgnoLM achieves a 46% performance improvement compared to naive continual pretraining. Furthermore, we demonstrate that VocAgnoLM consistently benefits from stronger teacher models, providing a robust solution to vocabulary mismatches in language modeling.




* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()   

<br/>

# 단어정리  

*  agnostic : 독립적인  







 
<br/>
# Methodology    




이 논문에서는 서로 다른 vocabulary를 가진 teacher와 student 모델 사이의 지식 전이 문제를 해결하기 위해 VocAgnoLM이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다음 두 가지 핵심 기법으로 구성됩니다:

Token-level Lexical Alignment (토큰 수준의 어휘 정렬)
서로 다른 토크나이저로 인해 같은 문장이 teacher와 student에서 다르게 나뉘는 문제를 해결하기 위해, 원래의 텍스트 상에서 문자 단위 위치(offset)를 추적하여 student의 각 토큰이 어떤 teacher 토큰 집합과 대응되는지를 정밀하게 계산합니다. 이를 통해 one-to-many mapping을 구성하고, alignment 정보를 효율적으로 계산합니다 (알고리즘 복잡도 O(N log M)).

Teacher Guided Loss (교사 손실 기반 지도)
token alignment 결과를 바탕으로 teacher 모델의 손실값(loss)을 참조하여 student 모델의 학습에 반영합니다. 특히, 학생 토큰의 중요도를 teacher 손실값과의 차이를 기준으로 동적으로 조정하며, 상위 중요 토큰(top-k)만 선택적으로 학습에 반영하는 방식으로 효율성을 높입니다.



이러한 방식은 동일한 vocabulary를 요구하지 않으면서도, teacher 모델의 정밀한 피드백을 student 모델이 받아들일 수 있게 해 주며, 실험에서는 수십 퍼센트의 성능 향상을 달성하였습니다.




The paper introduces VocAgnoLM, a novel framework to address the challenge of vocabulary mismatch between teacher and student language models. The method comprises two main components:

Token-level Lexical Alignment
To resolve the issue of token sequence misalignment due to different tokenizers, this method tracks character-level offsets from the raw input to build precise one-to-many mappings between student and teacher tokens. This alignment is computed efficiently using binary search (O(N log M)) and allows the student to receive fine-grained guidance from the teacher model without requiring vocabulary unification.

Teacher Guided Loss
Based on the token alignment, the method leverages the teacher model’s token-level loss to reweight the student’s training objective. Specifically, it identifies important tokens via the difference in loss between student and mapped teacher tokens, applying a top-k selection strategy to focus on the most informative tokens while still training unmapped tokens such as special tokens.

This vocabulary-agnostic approach allows the student to benefit from powerful teacher models regardless of tokenizer differences and leads to significant performance improvements in practice.




   
 
<br/>
# Results  





논문에서는 TinyLlama 1.1B를 student 모델로 사용하고, vocabulary가 다른 여러 개의 7B teacher 모델(Llemma, Mistral-ProXMath, DeepSeekMath, Qwen2.5-Math 등)을 이용해 수학 문제 해결 능력을 평가했습니다. 실험은 총 **9개 수학 벤치마크(GSM8k, MATH, SVAMP, MAWPS, MMLU-STEM 등)**에서 진행되었고, 메트릭은 정확도(accuracy, %)를 기준으로 했습니다.

주요 비교 대상은 다음과 같습니다:

KLD (KL Divergence): 동일 vocabulary일 때의 대표적 knowledge distillation 기법.

ULD (Universal Logit Distillation): vocabulary가 다를 때의 대안적 확률분포 정렬 기법.

Rho-1: teacher loss 기반의 fine-grained guidance (동일 vocab 필요).

핵심 결과는 다음과 같습니다:

Qwen2.5-Math-Instruct를 teacher로 쓴 VocAgnoLM은 naive continual pretraining 대비 46% 향상, ULD 대비 33% 향상된 성능을 보였습니다.

특히 teacher 모델의 성능이 높을수록 VocAgnoLM의 student 모델 성능도 함께 올라갔으며, 이는 기존 logit 기반 정렬 방식보다 강건하고 확장성 높은 방법임을 보여줍니다.

fine-grained token alignment가 coarse chunking보다 훨씬 효과적이며, unmapped token도 포함해서 학습하는 전략이 가장 좋았습니다.




The experiments used TinyLlama 1.1B as the student model and several 7B-scale teacher models with different vocabularies (Llemma, Mistral-ProXMath, DeepSeekMath, Qwen2.5-Math-Instruct) to evaluate performance across nine mathematical reasoning benchmarks (e.g., GSM8K, MATH, SVAMP, MMLU-STEM, MAWPS). Accuracy was used as the primary metric.

The study compared the proposed method with:

KLD (KL Divergence) for same-vocabulary teacher-student settings.

ULD (Universal Logit Distillation) for cross-vocabulary scenarios.

Rho-1, which also uses token-level loss guidance but requires shared vocabularies.

Key findings include:

When using Qwen2.5-Math-Instruct (only 6% vocab overlap), VocAgnoLM achieved a 46% performance gain over naive continual pretraining and 33% improvement over ULD.

Performance of the student model improved in proportion to the teacher model’s capability, demonstrating robust scalability and effectiveness compared to probabilistic logit alignment methods.

Fine-grained token-level alignment outperformed coarse chunk-level alignment, and including unmapped tokens (such as special tokens) during training yielded the best results.




<br/>
# 예제  





 예시 한글 설명:
이 논문에서의 주요 실험 테스크는 수학적 추론 능력 평가입니다.
모델은 사전학습(pretraining)과 평가(testing) 모두에서 수학 관련 텍스트를 처리합니다.



 트레이닝 데이터:
OpenWebMath: 약 150억 개 토큰 규모의 수학 관련 웹 문서로 구성된 데이터셋.

이 코퍼스는 웹 크롤링 기반으로 수집되었으며, 수학 문제 설명, 풀이 과정, 수식 등이 포함됩니다.

입력(input)은 일반적인 수학 관련 문장들(예: “Find the value of x if...”), 출력(output)은 다음 토큰을 예측하는 언어 모델링 과제입니다.



 테스트 데이터:
총 9개의 수학 벤치마크를 사용하여 다양한 난이도와 문제 유형을 다룸.

예:

GSM8K: 초등 수준의 단계별 산술 문제

MATH: 고등 및 대학 수준의 수학 문제

SVAMP / ASDiv / MAWPS / MMLU-STEM / SAT 등도 포함

문제 유형:

Multiple choice 및 open-ended 형태의 문제

문제 예시: “A train travels at 60 km/h for 2 hours. How far does it go?”

모델은 Chain-of-Thought(CoT) 방식으로 reasoning을 거쳐 답을 도출함.



 테스크 포맷:
입력: 수학 문제 텍스트 (few-shot CoT prompting 사용)

출력: 문제에 대한 정답 (숫자나 식 또는 선택지)



 


The main task in this paper is mathematical reasoning through language modeling, evaluated via both pretraining and downstream testing.



 Training Data:
OpenWebMath: A large-scale corpus of 15B tokens composed of math-related web documents.

These documents include problem statements, derivations, formulas, and explanations.

Task: causal language modeling where the input is a math-related text span (e.g., “Find the value of x if...”) and the output is to predict the next token.



 Evaluation / Test Data:
A total of 9 benchmark datasets were used to evaluate reasoning ability at different levels:

GSM8K: Elementary-level arithmetic reasoning

MATH: High school and university-level math problems

Others: SVAMP, ASDiv, MAWPS, TabMWP, MathQA, MMLU-STEM, and SAT

Problem types:

Both multiple-choice and open-ended formats

Example: “A train travels at 60 km/h for 2 hours. How far does it go?”

Models are prompted with few-shot chain-of-thought (CoT) to encourage step-by-step reasoning.



 Task Format:
Input: math question prompt (with optional few-shot CoT examples)

Output: the final answer (numeric value, formula, or multiple choice)




<br/>  
# 요약   



이 논문은 서로 다른 vocabulary를 가진 teacher와 student 모델 간의 지식 전달을 가능하게 하기 위해 Token-level Lexical Alignment와 Teacher Guided Loss로 구성된 VocAgnoLM을 제안합니다.
수학 텍스트로 구성된 OpenWebMath로 TinyLlama를 사전학습하고, 9개 수학 벤치마크에서 다양한 teacher 모델과의 조합을 비교한 결과, 기존 방법보다 최대 46% 성능 향상을 달성했습니다.
입력은 수학 문제 텍스트이며, 출력은 답을 생성하는 언어 모델링 테스크로 구성되어 있어 다양한 난이도의 수학 문제를 해결할 수 있도록 설계되었습니다.




This paper proposes VocAgnoLM, a method that enables knowledge transfer between teacher and student models with mismatched vocabularies, using Token-level Lexical Alignment and Teacher Guided Loss.
Using OpenWebMath for continual pretraining of TinyLlama and evaluating on nine math reasoning benchmarks, VocAgnoLM achieves up to 46% improvement over naive pretraining and outperforms prior alignment methods.
The task involves math word problems as input and predicted answers as output, designed to assess models' reasoning across various difficulty levels.



<br/>  
# 기타  





 Figure 1: Vocabulary Overlap 문제 시각화
Qwen2.5-Math 모델은 Llemma보다 수학 성능이 훨씬 뛰어나지만, student 모델인 TinyLlama와의 vocabulary 중복률이 6.32%에 불과해 기존 방식으로는 지도 학습이 어렵다는 문제를 강조합니다.
-> 인사이트: 높은 성능의 teacher를 쓰고 싶어도 vocabulary mismatch 때문에 활용이 어려웠음 → VocAgnoLM의 필요성 부각.



 Figure 2: Token-level Lexical Mapping 개요
다양한 teacher 모델이 같은 문장을 다르게 토크나이즈하는 예시를 보여주며, student 모델과의 sequence misalignment 문제를 시각화합니다.
-> 인사이트: 토큰 단위 매핑이 없으면 teacher 지도를 제대로 받을 수 없으며, 이를 alignment로 해결해야 함.



 Figure 3 & 6: Chunk 수 변화에 따른 alignment 성능 및 정확도
chunk 수가 많아질수록(세분화될수록) token overlap(IoU, IoS)이 감소하고, 이로 인해 student 성능도 감소합니다.
-> 인사이트: 무작정 세분화하면 정렬 품질이 떨어지며, VocAgnoLM의 token-level mapping은 항상 100% IoS를 유지함.



 Table 1: 다양한 teacher 모델과의 성능 비교
Qwen2.5-Math-Instruct를 사용한 VocAgnoLM이 ULD 대비 평균 33%, naive continual pretraining 대비 46% 향상.
-> 인사이트: vocabulary mismatch가 있어도 강한 teacher의 성능을 효과적으로 전이할 수 있음.



 Table 2: Chunking 방식 vs. Token-level Alignment
teacher loss를 활용한 chunking alignment는 정렬 품질이 낮고 성능도 떨어지지만, token-level 정렬 방식은 지속적으로 높은 성능 유지.
-> 인사이트: coarse한 mapping은 guidance 효과가 낮으며, fine-grained alignment가 훨씬 강력함.



 Table 3: Unmapped 토큰 처리 및 Teacher 토큰 집계 전략
Unmapped token을 포함(include)하고, teacher token loss는 max로 집계했을 때 가장 성능이 높았음.
-> 인사이트: 특수 토큰도 학습에서 중요한 역할을 하며, 학습 신호가 강한 teacher 토큰에 집중하는 것이 효과적임.



 Appendix A–B: 각 teacher 모델의 vocab 크기 및 tokenization 방식, 실험 설정
예: Qwen2.5-Math는 15만개 vocab, BBPE 기반. 실험은 32개 GPU, 15B 토큰, top-k 40% filtering 적용.
-> 인사이트: 다양한 tokenizer 간의 차이에도 불구하고 VocAgnoLM은 일반화 가능함.







 Figure 1: Visualization of Vocabulary Mismatch
Although Qwen2.5-Math outperforms Llemma in math reasoning, it shares only 6.32% vocabulary with TinyLlama, highlighting how such mismatch impedes knowledge transfer.
-> Insight: Even strong teacher models are underutilized due to tokenizer incompatibility — motivating VocAgnoLM.



 Figure 2: Overview of Token-level Lexical Mapping
Illustrates how different teacher models tokenize the same input phrase differently, leading to sequence misalignment with the student.
-> Insight: Without precise token alignment, teacher supervision becomes noisy or ineffective.



 Figures 3 & 6: Chunk Granularity vs. Alignment Quality & Accuracy
Finer chunking leads to lower token overlap (IoU, IoS), which degrades student model performance.
-> Insight: Token-level alignment (VocAgnoLM) maintains 100% IoS, outperforming chunk-based approaches.



 Table 1: Performance Across Teacher Models
VocAgnoLM with Qwen2.5-Math-Instruct achieves a 46% gain over naive continual pretraining and 33% over ULD, even with only 6% vocabulary overlap.
-> Insight: Strong teacher models can be utilized effectively regardless of vocabulary mismatch.



 Table 2: Comparison of Chunking vs. Token-level Alignment
Chunk-based guidance provides limited improvement, while token-level alignment yields consistently better results.
-> Insight: Fine-grained lexical mapping is crucial for effective teacher supervision.



 Table 3: Strategies for Unmapped Tokens & Aggregation
Including unmapped tokens and using max aggregation for teacher losses led to the best performance.
-> Insight: Special tokens are important for understanding, and emphasizing high-loss tokens is beneficial.



 Appendix A–B: Vocabulary Size, Tokenization, and Implementation Details
E.g., Qwen2.5-Math uses BBPE with 150K tokens; training used 32 H100 GPUs on 15B tokens with top-k 40% thresholding.
-> Insight: Despite tokenizer diversity, VocAgnoLM shows robust applicability across models.


<br/>
# refer format:     



@inproceedings{shin2025vocagnolm,
  title     = {Overcoming Vocabulary Mismatch: Vocabulary-agnostic Teacher Guided Language Modeling},
  author    = {Haebin Shin and Lei Ji and Xiao Liu and Yeyun Gong},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  publisher = {PMLR},
  note      = {PMLR 267, Vancouver, Canada},
  url       = {https://proceedings.mlr.press/v267/}
}



Shin, Haebin, Lei Ji, Xiao Liu, and Yeyun Gong. "Overcoming Vocabulary Mismatch: Vocabulary-agnostic Teacher Guided Language Modeling." In Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267, Vancouver, Canada, 2025.





