---
layout: post
title:  "[2023]Mixtral of Experts"  
date:   2024-03-19 09:03:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    
* 저자들은 Mixtral 8x7B를 소개하는데, 이는 SMoE 언어 모델임

Mixtral은 Mistral 7B와 같은 구조를 가지고 있으나 차이점은 각 층이 8개의 전방향 블록, 즉 전문가로 구성되어 있다는 것임

각 토큰마다 라우터 네트워크가 현재 상태를 처리할 두 전문가를 선택하고 그들의 출력을 결합함

각 토큰이 단 두 개의 전문가만 볼 수 있지만 선택된 전문가는 각 타임스텝마다 다를 수 있음

결과적으로 각 토큰은 47B의 매개변수에 접근할 수 있지만 추론 중에는 13B의 활성 매개변수만 사용함

Mixtral은 32k 토큰의 컨텍스트 크기로 훈련되었으며 평가된 모든 벤치마크에서 Llama 2 70B와 GPT-3.5를 능가하거나 일치함

특히 Mixtral은 수학 코드 생성과 다국어 벤치마크에서 Llama 2 70B를 크게 능가함

저자들은 또한 지시사항을 따르도록 미세조정된 모델인 Mixtral 8x7B - Instruct를 제공하는데, 이는 GPT-3.5 Turbo, Claude-2.1, Gemini Pro, 그리고 Llama 2 70B - 챗 모델을 인간 벤치마크에서 능가함

기본 모델과 지시 모델 모두 Apache 2.0 라이선스 하에 출시됨


Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1LGdCi1N3XPj2lKIqWCwfeHWeEF3FjC18?usp=sharing)  
[~~Lecture link~~]()  

<br/>

# 단어정리  
* 

<br/>
# 1 Introduction    
* 이 논문에서 저자들은 Mixtral 8x7B를 소개하는데, 이는 개방형 가중치를 갖는 희박 전문가 혼합 모델(SMoE)임

Apache 2.0 라이선스에 따라 Mixtral은 대부분의 벤치마크에서 Llama 2 70B와 GPT-3.5를 능가함

각 토큰마다 일부 매개변수만 사용함으로써 Mixtral은 작은 배치 크기에서 더 빠른 추론 속도와 큰 배치 크기에서 더 높은 처리량을 허용함

Mixtral은 희박 전문가 혼합 네트워크로, 디코더 전용 모델임

전방향 블록은 8개의 구별된 매개변수 그룹 중에서 선택하고, 각 층마다 각 토큰에 대해 라우터 네트워크가 이들 그룹 중 두 개를 선택하여 토큰을 처리하고 그들의 출력을 가산적으로 결합함

이 기법은 모델의 매개변수 수를 증가시키면서 비용과 지연을 제어함

저자들은 32k 토큰의 컨텍스트 크기를 사용하여 다국어 데이터로 Mixtral을 사전 훈련시킴

Mixtral은 여러 벤치마크에서 Llama 2 70B와 GPT-3.5의 성능을 매치하거나 초과함

특히 Mixtral은 수학 코드 생성 및 다국어 이해에 필요한 작업에서 Llama 2 70B를 크게 능가함 



<br/>
# 2 Architectural Details  
* Mixtral은 트랜스포머 구조를 기반으로 하며, Mistral 7B와 같은 구조적 변형을 사용하지만 32k 토큰의 완전 밀집 컨텍스트 길이를 지원하고, 전방향 블록을 전문가 혼합 레이어로 대체한 것이 특징임

모델 아키텍처 매개변수는 4096의 차원, 32개의 층, 128의 헤드 차원, 14336의 숨겨진 차원을 가지고 있음

전문가 혼합 레이어에 대한 간략한 개요는 각 입력 x에 대해 전문가 네트워크의 출력의 가중 합으로 결정되며, 가중치는 게이팅 네트워크의 출력에 의해 제공됨

즉, n개의 전문가 네트워크가 주어졌을 때, 전문가 레이어의 출력은 G(x)i · Ei(x)로 주어짐

여기서 G(x)i는 i번째 전문가에 대한 게이팅 네트워크의 n차원 출력을 나타내고, Ei(x)는 i번째 전문가 네트워크의 출력임

게이팅 벡터가 희소한 경우, 게이트가 0인 전문가의 출력을 계산하지 않을 수 있음

G(x) 구현 방법은 여러 가지가 있지만, 선형 레이어의 상위 K 로그에 대한 소프트맥스를 취하는 간단하고 효율적인 방법을 사용함

K 값은 토큰당 사용된 전문가 수를 나타내는 하이퍼파라미터로, 각 토큰을 처리하는 데 사용되는 계산량을 조절함

n을 증가시키면서 K를 고정하면 모델의 매개변수 수를 증가시키면서도 계산 비용을 효과적으로 일정하게 유지할 수 있음

이는 모델의 총 매개변수 수(흔히 희소 매개변수 수라고 함)와 개별 토큰을 처리하는 데 사용되는 매개변수 수(활성 매개변수 수라고 함) 사이의 구분을 동기화함

MoE 레이어는 고성능의 특수 커널을 사용하여 단일 GPU에서 효율적으로 실행할 수 있음

예를 들어, Megablocks는 MoE 레이어의 피드포워드 네트워크(FFN) 연산을 대규모 희소 행렬 곱셈으로 캐스팅하여 실행 속도를 크게 향상시키고, 서로 다른 전문가에게 다양한 수의 토큰이 할당되는 경우를 자연스럽게 처리함 



<br/>
# 3 Results  
* 저자들은 Mixtral을 Llama와 비교하고 모든 벤치마크를 자체 평가 파이프라인으로 다시 실행하여 공정한 비교를 함

다음과 같은 다양한 작업으로 성능을 측정함

- 상식 추론 (0-샷): Hellaswag, Winogrande, PIQA, SIQA, OpenbookQA, ARC-Easy, ARC-Challenge, CommonsenseQA
- 세계 지식 (5-샷): NaturalQuestions, TriviaQA
- 독해력 (0-샷): BoolQ, QuAC
- 수학: GSM8K (8-샷) maj@8과 MATH (4-샷) maj@4로
- 코드: Humaneval (0-샷)과 MBPP (3-샷)
- 인기 집계 결과: MMLU (5-샷), BBH (3-샷), AGI Eval (3-5-샷 영어 객관식 문제만)

모든 모델이 정확한 비교를 위해 모든 메트릭에서 다시 평가되었고 Mixtral은 모든 벤치마크에서 Llama 2 70B를 능가하거나 일치함

특히 수학 및 코드 생성에서 훨씬 우수함

테이블 2에서 Mixtral, Mistral 7B, 그리고 Llama 2 7B/13B/70B 및 Llama 1 34B2의 자세한 결과를 보고함

그림 2에서는 Mixtral이 다양한 카테고리에서 Llama 모델과 비교한 성능을 비교함

Mixtral은 대부분의 메트릭에서 Llama 2 70B를 초과함

특히 코드 및 수학 벤치마크에서 우수한 성능을 보임 .




<br/>
# 4 Instruction Fine-tuning  
* 저자들은 지시 데이터셋에서 감독된 미세조정(SFT)을 사용하여 Mixtral - Instruct를 훈련시키고, 이어서 짝을 이룬 피드백 데이터셋에서 직접 선호 최적화(DPO)를 진행함

Mixtral - Instruct는 MT-Bench에서 8.30의 점수를 달성하여 2023년 12월 기준으로 최고의 오픈-웨이트 모델이 됨

독립적인 인간 평가에서는 Mixtral - Instruct가 GPT-3.5-Turbo, Gemini Pro, Claude-2.1, 그리고 Llama 2 70B 채팅 모델을 능가하는 것으로 나타남   



<br/>
# 5 Routing Analysis   
이 섹션에서는 라우터에 의한 전문가 선택에 대한 간단한 분석을 수행함

특히, 훈련 중에 일부 전문가들이 특정 도메인(예: 수학, 생물학, 철학 등)에 특화되었는지를 알아보고자 함

이를 조사하기 위해 The Pile 검증 데이터셋의 다양한 하위 집합에서 선택된 전문가의 분포를 측정함

0, 15, 31층(각각 모델의 첫 번째와 마지막 층)에 대한 결과가 그림 7에 제시됨

놀랍게도 주제에 기반한 전문가 할당에서 뚜렷한 패턴을 관찰하지 못함

예를 들어, 모든 층에서 ArXiv 논문(Latex로 작성됨), 생물학(PubMed Abstracts), 철학(PhilPapers) 문서에 대한 전문가 할당 분포가 매우 유사함

DM 수학에 대해서만 약간 다른 전문가 분포를 주목함

이러한 차이는 데이터셋의 합성적인 성격과 자연 언어 스펙트럼의 제한된 커버리지 때문일 가능성이 있으며, 특히 입력 및 출력 임베딩과 매우 상관관계가 높은 첫 번째 및 마지막 층에서 특히 두드러짐

이는 라우터가 일정한 구조적 구문적 행동을 보여준다는 것을 시사함

그림 8은 다양한 도메인(Python 코드, 수학, 영어)의 텍스트 예시를 보여주며, 각 토큰은 선택된 전문가에 해당하는 배경색으로 강조됨

그림에서 Python의 'self'와 영어의 'Question'과 같은 단어들이 여러 토큰을 포함하더라도 동일한 전문가를 통해 라우팅되는 것을 보여줌

비슷하게, 코드에서 들여쓰기 토큰은 항상 동일한 전문가에게 할당되며, 특히 모델의 입력 및 출력과 더 상관관계가 높은 첫 번째 및 마지막 층에서 더 그러함

그림 8에서도 연속적인 토큰이 종종 동일한 전문가에게 할당되는 것을 볼 수 있음

실제로 The Pile 데이터셋에서 위치적 국소성의 일정한 정도를 관찰함

표 5는 도메인 및 층별로 연속적인 토큰이 동일한 전문가에게 할당되는 비율을 보여줌

연속적인 할당의 비율은 상위 층에서 무작위 할당보다 현저히 높음 



<br/>
# 6 Conclusion  
이 논문에서 저자들은 Mixtral 8x7B를 소개함

이것은 최초로 최고의 성능을 달성한 전문가의 혼합 네트워크 중 하나임

Mixtral 8x7B Instruct는 인간 평가 벤치마크에서 Claude-2.1, Gemini Pro, 그리고 GPT-3.5 Turbo를 능가함

각 타임스텝에서 단 두 개의 전문가만 사용하기 때문에 Mixtral은 토큰당 13B의 활성 매개변수만 사용하면서 토큰당 70B의 매개변수를 사용하는 이전 최고의 모델(Llama 2 70B)을 능가함

저자들은 훈련되고 미세 조정된 모델들을 Apache 2.0 라이선스 하에 공개함으로써 다양한 산업과 분야에서 새로운 기술과 응용 프로그램의 개발을 촉진하고자 함 [oai_citation:1,Error](data:text/plain;charset=utf-8,Unable%20to%20find%20metadata)





# 요약  

* Mixtral 8x7B는 토큰당 매개변수의 일부만 사용하여 효율성을 향상시키는 Sparse Mixture of Experts 언어 모델을 소개   
** 다수의 전문가 네트워크를 효율적으로 결합하여 각 입력에 가장 적합한 전문가를 동적으로 선택하는 구조
** 이를 통해 모델은 높은 성능과 함께 더 나은 확장성과 맞춤형 처리 능력을 갖추게 함  
* 이를 통해 추론 속도가 빨라지고 처리량이 높임  
* 이 모델은 수학 및 다국어 작업에서 특히 Llama 2 70B와 GPT-3.5와 같은 기존 모델을 여러 벤치마크에서 능가  
* Mixtral은 8개의 전문가 집합에서 토큰당 두 전문가를 선택하는 라우팅 메커니즘을 도입하여 매개변수 사용을 최적화  
** SMoE 모델 내의 라우팅 메커니즘은 각 토큰에 대해 8개의 가능한 전문가 중에서 가장 적합한 두 전문가를 선정  
** 이 과정은 모델이 각 토큰의 특성과 문맥을 고려하여 최적의 처리를 결정하도록 도움  



* Mixtral 8x7B introduces a Sparse Mixture of Experts language model that uses only part of its parameters per token for efficiency
** It combines many expert networks efficiently to dynamically select the best expert for each input
** This gives the model high performance, better scalability, and custom processing abilities
* This leads to faster inference speeds and higher throughput
* The model outperforms existing models like Llama 2 70B and GPT-3.5 in many benchmarks, especially in math and multilingual tasks
* Mixtral uses a routing mechanism that picks two experts from a set of eight for each token to optimize parameter use
** The routing mechanism in the SMoE model selects the two best experts out of eight possible ones for each token
** This process helps the model decide the best processing by considering the characteristics and context of each token

