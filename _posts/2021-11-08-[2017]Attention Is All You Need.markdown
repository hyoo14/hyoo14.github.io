---
layout: post
title:  "[2017]Attention Is All You Need"
date:   2021-11-08 12:55:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

어텐션만으로된 트랜스포머네트워크 제안  
성능이 기존 RNN, CNN 보다 좋음 SOTA  

(QA, LI 등)

{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1ORJIXwL6jF2ja52tEHurFEe_ntETWN0x?usp=sharing)



*RNN, LSTM, GRU가 LM이나 MT에서 이전의 SOTA였음  
**하지만 적용 범위를 넓히는 노력이 있었음에도, 이 적용범위에 한계가 있었음  
***문장이나 텍스트가 너무 길어질 경우 성능이 떨어지는 한계  

**또한 sequence 계산이 병렬적으로 안 되는 문제도 있음  

*여기서 제안하는 어텐션은 이러한 길이제한 문제와 병렬 계산 문제를 모두 풀어줄 해법이 됨  

*트랜스포머도 이전의 sequence용 모델들 처럼 encoder-decoder 구조. autoregressive함  
**self-attention, poitwise FCL로 아키택처 구현  

*어텐션은 query를 입력받으면 key-value 쌍으로 매핑해주는 것  
**이 때, key, query, value 모두 백터임. output도  
**여기서는 scaled dot-product attention 사용함  
***query와 모든 key들을 dot-product(외적)하고 차원크기로 나눠주는 것  
****참고로 dot-product할 때마다 차원 수가 너무 커져서 나눠주는 것임.   

*병렬 가능케하는 것은 다음과 같음  
**어텐션 함수들을 병렬적으로 수행하고 후에 concat해줌  

*멀티헤드는 각각 다른 관점을 대표하는 것.  

*인코더의 self-attention은 q,k,v가 같은 곳에서 옴.  
**output은 이전 layer encoder서 옴  

*인코더-디코더 어텐션은 이전 디코더의 쿼리가 in. 인코더에서 나온 key/value가 나옴.  
**모든 디코더 위치서 input으로 가능. seq2seq의 인코더 디코더 구조 유사하게 구성  
  
*디코더의 self-attention은 인코더의 것과 같으나 masking을 해주는 차이(답 숨기는 masking)  

*FCFN은 potition에 적용하는데 ReLU 2개로 구성됨  
**kernel size 1짜리 두 Convolution이라고 표현할 수도 있음  

*학습된 임베딩을 사용하고 학습도니 linear-transformation과 softmax를 사용함  

*위치정보는 포지션인코딩 사용.  

*왜 self-attention사용하는가 하면,,  
**컴퓨터 복잡도 줄여주니깐.. 병렬성 가능케하고.  
**긴 거리 의존성도 가장 좋음.  

*실제 테스트에서도 결과가 좋았음  




# 최근 정리 2023  

## 1. 논문이 다루는 Task

Natural Language Process with Self-Attention Model

When **n** is length of token id list and **h** is dimension of word embedding vector.

### Encoder

- Input: tokenized text data [n]
- Output: n representation vector [n, h]

### Decoder

- input: generated token vector (in prev step) [n] ( + Encoder output [n, h] as key and value representation), <SOS> in first step
- output: generated token vector [n]

## 2. 기존 연구 한계

- Previous RNN model can’t calculate loss for nth output before calculate (n-1)th output. Because nth output require n-1th hidden state.
- Previous RNN model can’t use parallel processing so that their training speed is slow.
- Due to its sequential processing, it is slow and costly. Additionally, because it processes sequentially, it is difficult to convey information when words are far apart.

## 3. 제안 방법론:

### Main Idea

- By removing the RNN mechanism and using only the Attention mechanism, it is possible to reduce costs and processing time, and process data in parallel to store similarities and information between sentences.
- Normalization improves the stability of learning data information.
- Use Key, Query, Value matrix to calculate attention between each input tokens.

### ✨Contribution

- 

## 4. 실험 및 결과

### Dataset

- Dataset 이름 : WMT 2014 English-German

### Baseline

- GNMT + RL (English to German 24.6 BLEU, English to France 39.92)

### 결과

- 메인 결과
    - English to German 28.4BLEU(SOTA)
    - English to French 41.0BLEU(near SOTA)
- Ablation Study
    - number of head
    - attention key size
    - model size
    - drop out
    - positional encoding
- Analysis
    - better performance and better training speed

---

## 5. 질문 및 생각할거리

- Transformer 모델이 RNN 모델에 비해 학습속도가 빠른 이유를 직관적으로는 이해가 가는데 구체적으로 왜 더 빠른지 잘 모르겠습니다.
- Transfomer에서 Back propagation이 어떻게 진행되나요?

+ 추가 질문) 
- decoder 에서 masking 이후 병렬화 하여 학습을 하는데 어떻게 학습된 결과를 합하는지 궁금합니다.


---

- Encoder Input : I ate Dinner
- 학습 시점

| t | 0 | 1 | 2 | 3 |  |
| --- | --- | --- | --- | --- | --- |
| loss | 0.32 | 0.214 |  |  | 80 |
| label | 나는  | 저녁을 |  |  |  |
| output | P(나는) | 저녁을 |  |  |  |
| input | <SOS> | 나는 | 저녁을 |  |  |

- 번역(Inference) 시점
- Encoder Input : I ate Dinner / I have pencil
- Encoder Output : (3, 768)

| t | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
|  | 나는 |  |  |  |
| input | <SOS> | 나는 | 저녁을 | 먹었다. |
- encoder input : (n)
- encoder output : (n, d)
- decoder intput : (m)
    - Embedding : (m, d)
    - Positional Encoding : (m, d)
    - Masked Self Attention
        - Q → (m, d)*(d, d)
        - K → (m, d)*(d, d)
        - V → (m, d)*(d, d)
    - (Vocab, d)
        - vocab → 33000
        - d → 768
    - Attention Score
        - SoftMax(QK^t)
        - (m, d) (d, n) → (m , n)
        - QK^t
    - Attn Output : Score*V
        - (m, n)*(n, d) → (m, d)
        
        |  | I | ate | dinner |
        | --- | --- | --- | --- |
        | I | 34 | -Inf | -Inf |
        | ate | 45 | 82 | -Inf |
        | dinner | 21 | 4 | 12 |
        |  |  |  |  |
        - SoftMax
            
            
            |  | I | ate | dinner |
            | --- | --- | --- | --- |
            | I | 1 | 0 | 0 |
            | ate | 0.4 | 0.6 | 0 |
            | dinner | 0.2 | 0.4 | 0.4 |
            
            |  | ate | I | dinner |
            | --- | --- | --- | --- |
            | ate | 1 | 0 | 0 |
            | I | 0.4 | 0.6 | 0 |
            | dinner | 0.4 | 0.2 | 0.4 |
            
    - RNN (시간 정보가 자연스럽게 반영)
        - <SOS> → hidden_1 → I
        - I, hidden_1 → hidden_2 → ate
        - ate, hidden_2 → hidden_3 → dinner
        
        - <SOS> → hidden_1 → ate
        - ate, hidden_1 → hidden_2 → I
        
    - Encoder Self Attn 행렬 연산으로 손으로 직접 써보기
    - ENcoder Self Attn Input : (n, d)
    - Q, K, V 어떻게 어떤 행렬이랑 연산이 되는지
    - Q, K, V를 어떻게 연산해서 Attn이 수행이 되는지
    - 최종적인 output의 shape이 어떻게 되는지
    
    +) FFNN