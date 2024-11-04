---
layout: post
title:  "[2021]An Attentive Survey of Attention Models"  
date:   2024-11-04 00:19:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


짧은 요약(Abstract) :    




---

이 논문은 신경망에서 중요한 개념으로 자리 잡은 '어텐션 모델'에 대해 종합적인 개요를 제공합니다. 저자들은 기존의 어텐션 모델 기법을 체계적으로 분류하고, 어텐션이 포함된 신경망 구조 및 주요 응용 사례를 살펴봅니다. 또한, 어텐션이 신경망의 해석 가능성을 향상시키는 방식과 향후 연구 방향을 논의합니다. 이 논문은 어텐션 모델을 처음 접하는 연구자들에게 유익한 입문 자료를 제공하며, 다양한 응용 프로그램을 개발하는 과정에서 실질적인 지침을 제공하고자 합니다.

---

This paper provides a comprehensive overview of the Attention Model, a concept that has become pivotal in neural networks across diverse application domains. The authors propose a structured taxonomy to categorize existing techniques, review key neural architectures incorporating attention, and discuss applications where attention modeling has significantly impacted outcomes. They also explore how attention enhances the interpretability of neural networks and outline potential directions for future research. This survey aims to serve as an introductory resource for newcomers to attention models and to guide practitioners in developing applications across various fields.


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




#### 1. 시퀀스 개수에 따른 어텐션
- **독립적 어텐션**: 입력 시퀀스와 출력 시퀀스가 별도로 존재하며, 번역과 이미지 캡셔닝 등의 작업에 사용됩니다.
- **공동 어텐션**: 여러 시퀀스를 동시에 처리하며 각 시퀀스 간의 상호작용을 학습합니다. 시각 질문 응답과 같은 다중 모드 데이터에서 주로 활용됩니다.
- **자기 어텐션**: 같은 시퀀스 내에서 관련성을 학습하며, 최근 많은 논문에서 사용되는 형태입니다.

#### 2. 추상화 수준에 따른 어텐션
- **단일 수준 어텐션**: 원래의 입력 시퀀스에 대해서만 어텐션을 계산합니다.
- **다중 수준 어텐션**: 단어 수준에서 시작하여 문장 수준으로 점차 높아지는 방식으로 계층적 어텐션을 계산합니다. 이는 문서 분류와 같은 작업에 유용합니다.

#### 3. 위치 기반 어텐션
- **소프트 어텐션**: 모든 위치에 가중치를 할당하는 방식으로 계산 비용이 큽니다.
- **하드 어텐션**: 특정 위치를 샘플링하여 선택하므로 계산 비용이 낮지만 최적화가 어렵습니다.
- **로컬 어텐션**: 특정 위치 주변의 작은 창을 설정해 해당 위치에서의 소프트 어텐션을 수행합니다.

#### 4. 표현 수에 따른 어텐션
- **다중 표현 어텐션**: 동일한 입력의 여러 표현을 다루어, 중복 요소와 노이즈를 필터링합니다.
- **다차원 어텐션**: 입력의 각 차원을 개별적으로 처리하여 문맥적 의미를 추출합니다.

---



#### 1. Attention by Number of Sequences
- **Distinctive Attention**: Involves separate input and output sequences, used in tasks such as translation and image captioning.
- **Co-Attention**: Operates on multiple input sequences simultaneously, learning interactions between these inputs, often used in multi-modal data applications like Visual Question Answering.
- **Self Attention**: Learns relations within the same sequence, commonly used in recent research papers.

#### 2. Attention by Abstraction Levels
- **Single-Level Attention**: Calculates attention only for the original input sequence.
- **Multi-Level Attention**: Gradually computes hierarchical attention, moving from word level to sentence level, useful for tasks like document classification.

#### 3. Attention by Position
- **Soft Attention**: Assigns weights to all positions, though computationally intensive.
- **Hard Attention**: Selects certain positions by sampling, reducing computation costs but challenging to optimize.
- **Local Attention**: Sets a small window around a specific position and performs soft attention locally.

#### 4. Attention by Number of Representations
- **Multi-Representational Attention**: Deals with multiple representations of the same input, filtering out redundancies and noise.
- **Multi-Dimensional Attention**: Processes each dimension of input independently to extract contextual meaning.





이 논문에서는 트랜스포머 모델의 아키텍처와 효율성, 정확도 향상에 대해 설명합니다. 트랜스포머는 순환 연결을 제거하고, 자기 어텐션만을 통해 입력과 출력 간의 전역적 의존성을 학습하는 모델입니다. 주요 내용은 다음과 같습니다.

1. **트랜스포머 아키텍처 개요**: 트랜스포머는 순차 처리를 없애고 병렬 처리가 가능하게 설계되었습니다. 이를 통해 기계 번역과 같은 작업에서 더 빠르고 정확한 결과를 얻을 수 있습니다.

2. **자기 어텐션과 스케일드 닷 프로덕트**: 트랜스포머는 스케일드 닷 프로덕트를 사용해 각 토큰과 해당 위치 간의 관계를 학습합니다. 이를 통해 병렬 처리의 효율성이 크게 향상됩니다.

3. **다중 헤드 어텐션**: 여러 개의 어텐션 헤드를 병렬로 활용하여 입력을 여러 세그먼트로 나누고, 각 세그먼트에 대해 독립적으로 어텐션을 계산합니다. 이를 통해 다차원적 문맥 정보를 학습하며 정보 손실을 줄입니다.

4. **포지셔널 인코딩**: 트랜스포머는 입력 시퀀스의 순서를 반영하기 위해 포지셔널 인코딩을 사용하여 모델이 각 단어의 위치를 이해할 수 있도록 돕습니다.

5. **잔차 연결과 정규화**: 각 층에 잔차 연결과 정규화를 적용하여 훈련 과정의 안정성을 높이고, 깊은 네트워크에서도 정보 손실을 최소화하여 성능을 향상시킵니다.

이와 같이 트랜스포머는 NLP와 컴퓨터 비전 등 다양한 작업에서 활용되며, 여러 변형 모델들이 연구되어 높은 성능을 보입니다.

---



This paper explains the architecture, efficiency, and accuracy improvements of the Transformer model. The Transformer is a model that eliminates recurrent connections, learning global dependencies between input and output through self-attention alone. The main points are as follows:

1. **Overview of the Transformer Architecture**: The Transformer is designed to eliminate sequential processing, enabling parallel computation, resulting in faster and more accurate outcomes in tasks like machine translation.

2. **Self-Attention and Scaled Dot-Product**: The Transformer uses scaled dot-product attention to learn relationships between each token and its position, greatly enhancing processing efficiency.

3. **Multi-Head Attention**: By using multiple attention heads in parallel, the model divides the input into segments and calculates attention independently for each segment, allowing it to capture multi-dimensional contextual information and reducing information loss.

4. **Positional Encoding**: To reflect the order of the input sequence, the Transformer employs positional encoding, helping the model understand the position of each word.

5. **Residual Connections and Normalization**: Residual connections and normalization are applied to each layer to increase training stability and minimize information loss, especially in deeper networks, thereby improving performance.

The Transformer thus finds applications across various tasks in NLP and computer vision, with numerous variants being researched and demonstrating high performance.


<br/>
# Results  





<br/>
# 예제  





#### 1. 독립적 어텐션 (Distinctive Attention)
- **예시**: 번역 작업, 이미지 캡셔닝
- **적용 분야**: 독립적 어텐션은 입력과 출력 시퀀스가 서로 다른 경우에 유용합니다. 주로 기계 번역이나 이미지 캡셔닝과 같은 작업에서 사용되며, 각각의 시퀀스를 독립적으로 처리하여 번역된 문장이나 이미지 설명을 생성하는 데 효과적입니다.

#### 2. 공동 어텐션 (Co-Attention)
- **예시**: 시각 질문 응답
- **적용 분야**: 여러 입력 시퀀스 간의 상호작용을 학습하는 데 효과적입니다. 시각 질문 응답과 같은 다중 모드 데이터에서 사용되며, 이미지와 텍스트 사이의 관련성을 파악해 질문에 대한 답변을 생성할 때 주로 적용됩니다.

#### 3. 자기 어텐션 (Self Attention)
- **예시**: Transformer 모델, 문서 분류
- **적용 분야**: 동일한 시퀀스 내의 관련성을 파악하는 데 적합하며, 최근에는 NLP와 컴퓨터 비전 등 다양한 분야에서 널리 사용됩니다. 예를 들어, Transformer 모델에서는 문장의 모든 단어 간 관계를 파악해 더 효율적이고 정확한 번역을 수행할 수 있습니다.

#### 4. 다중 수준 어텐션 (Multi-Level Attention)
- **예시**: 계층적 문서 분류
- **적용 분야**: 다중 수준 어텐션은 단어 수준에서 문장 수준으로 점차 상위 수준으로 이동하면서 어텐션을 계산하여 문서 내 중요한 단어와 문장을 식별합니다. 이는 문서 분류 작업에서 문서의 계층적 구조를 반영하여 정확도를 높이는 데 유용합니다.

#### 5. 소프트 어텐션 (Soft Attention)
- **예시**: 텍스트 요약, 감정 분석
- **적용 분야**: 전체 입력 시퀀스에서 중요한 정보를 가중치로 반영하여 요약이나 감정 분석과 같은 작업에서 세부 정보를 포착하는 데 적합합니다. 하지만 계산 비용이 높다는 단점이 있습니다.

#### 6. 하드 어텐션 (Hard Attention)
- **예시**: 이미지 처리에서 중요한 위치 선택
- **적용 분야**: 특정 위치를 선택해 계산 자원을 줄이며, 이미지 처리에서 주요 객체나 장면을 포착하는 데 효과적입니다. 다만, 최적화가 어려워 일반적으로 확률적 방법과 함께 사용됩니다.

#### 7. 다중 표현 어텐션 (Multi-Representational Attention)
- **예시**: 문장 표현 향상
- **적용 분야**: 입력의 다양한 표현을 가중치로 조정하여 중요한 의미를 추출하는 데 유리합니다. 여러 임베딩을 조합하여 더 효과적인 문장 표현을 만들어내어 텍스트 분류와 같은 작업에 사용됩니다.

#### 8. 다차원 어텐션 (Multi-Dimensional Attention)
- **예시**: 다의어 해결, 문맥적 의미 파악
- **적용 분야**: 입력의 각 차원에 개별적인 가중치를 부여하여 특정 맥락에서 중요한 의미를 추출합니다. 자연어 처리에서 단어의 다양한 의미를 구분하는 데 도움을 줍니다.

---



#### 1. Distinctive Attention
- **Example**: Machine translation, image captioning
- **Application Area**: Distinctive attention is effective when input and output sequences are separate, making it ideal for tasks like machine translation or image captioning, where each sequence is processed independently to generate translated sentences or image descriptions.

#### 2. Co-Attention
- **Example**: Visual question answering
- **Application Area**: Co-attention is beneficial for learning interactions between multiple input sequences. It is commonly used in multi-modal tasks such as visual question answering, helping to identify the relationship between images and text to generate answers.

#### 3. Self Attention
- **Example**: Transformer model, document classification
- **Application Area**: Self-attention is suited for identifying relationships within the same sequence and is widely used in various fields, including NLP and computer vision. For instance, in the Transformer model, it captures relationships among all words in a sentence, enabling more efficient and accurate translations.

#### 4. Multi-Level Attention
- **Example**: Hierarchical document classification
- **Application Area**: Multi-level attention gradually calculates attention from word to sentence level, helping to identify key words and sentences in a document. This is useful in document classification tasks, enhancing accuracy by reflecting the document’s hierarchical structure.

#### 5. Soft Attention
- **Example**: Text summarization, sentiment analysis
- **Application Area**: Soft attention reflects key information across the entire input sequence through weighted scores, capturing finer details in tasks like summarization and sentiment analysis. However, it requires high computational resources.

#### 6. Hard Attention
- **Example**: Selecting key positions in image processing
- **Application Area**: Hard attention reduces computational costs by selecting specific positions, making it effective for capturing important objects or scenes in image processing. It is often used with probabilistic methods due to optimization challenges.

#### 7. Multi-Representational Attention
- **Example**: Enhancing sentence representation
- **Application Area**: Multi-representational attention is advantageous for extracting significant meanings by adjusting weights across multiple representations of the input. It is used in tasks like text classification, combining various embeddings for more effective sentence representation.

#### 8. Multi-Dimensional Attention
- **Example**: Resolving polysemy, capturing contextual meaning
- **Application Area**: Multi-dimensional attention assigns individual weights to each dimension of input to extract contextually significant meaning. It aids in distinguishing different meanings of words in natural language processing.



<br/>  
# 요약   



이 논문은 다양한 어텐션 모델의 메서드와 각각의 주요 예시 및 적용 분야를 설명합니다. 독립적 어텐션은 번역과 이미지 캡셔닝에서 시퀀스를 개별적으로 처리하는 데 유용하며, 공동 어텐션은 시각 질문 응답과 같은 다중 모드 데이터에 효과적입니다. 자기 어텐션은 동일한 시퀀스 내에서 중요한 관계를 학습하는 데 적합하며, Transformer 모델과 문서 분류 작업에서 많이 활용됩니다. 다중 수준 어텐션은 문서의 계층적 구조를 반영해 문서 분류에서 높은 성능을 발휘하고, 소프트 어텐션과 하드 어텐션은 각각 중요한 정보를 캡처하는 데 중점을 둡니다. 마지막으로 다중 표현 어텐션과 다차원 어텐션은 입력의 다양한 표현과 차원별 맥락적 의미를 추출하는 데 도움을 줍니다.

---



This paper describes various attention model methods, highlighting key examples and application areas for each. Distinctive attention is useful in translation and image captioning tasks, where sequences are processed independently. Co-attention is effective for multi-modal data like visual question answering, capturing interactions across multiple input sequences. Self-attention is suitable for identifying crucial relations within the same sequence, widely applied in Transformer models and document classification tasks. Multi-level attention reflects a document's hierarchical structure and enhances performance in document classification. Soft and hard attention focus on capturing important information differently, while multi-representational and multi-dimensional attention aid in extracting significant meanings from diverse representations and context-specific dimensions.



<br/>  
# 기타  




<br/>
# refer format:     

@article{chaudhari2021attentive,
  title={An Attentive Survey of Attention Models},
  author={Chaudhari, Sneha and Mithal, Varun and Polatkan, Gungor and Ramanath, Rohan},
  journal={ACM Transactions on Intelligent Systems and Technology (TIST)},
  volume={1},
  number={1},
  pages={1--33},
  year={2021},
  publisher={ACM},
  doi={10.1145/3465055}
}



Chaudhari, Sneha, Varun Mithal, Gungor Polatkan, and Rohan Ramanath. "An Attentive Survey of Attention Models." ACM Transactions on Intelligent Systems and Technology (TIST) 1, no. 1 (2021): 1-33. 