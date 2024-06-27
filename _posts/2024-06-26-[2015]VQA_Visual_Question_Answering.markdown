---
layout: post
title:  "[2015]VQA_Visual_Question_Answering"  
date:   2024-06-27 11:25:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    

우리는 자유형식 및 개방형 시각 질문 응답(VQA) 작업을 제안합니다. 이미지와 해당 이미지에 대한 자연어 질문이 주어졌을 때, 이 작업의 목표는 정확한 자연어 답변을 제공하는 것입니다. 실제 시나리오를 반영하여, 시각적 장애인을 돕는 것과 같은 경우에 질문과 답변 모두 개방형으로 설정되어 있습니다. 시각적 질문은 배경 세부 사항과 기본적인 맥락을 포함하여 이미지의 다른 영역을 선택적으로 타겟팅합니다. 그 결과, VQA에서 성공하기 위해서는 일반적인 이미지 캡션을 생성하는 시스템보다 이미지에 대한 더 상세한 이해와 복잡한 추론이 필요합니다. 또한, 많은 개방형 답변이 몇 마디 단어로 이루어지거나 다지선다형 형식에서 제공될 수 있는 닫힌 집합의 답변을 포함하므로 VQA는 자동 평가가 가능합니다. 우리는 약 0.25M개의 이미지, 약 0.76M개의 질문, 약 10M개의 답변을 포함하는 데이터셋을 제공하고, 그것이 제공하는 정보를 논의합니다. VQA를 위한 다양한 기준선을 제공하고 이를 인간의 성능과 비교합니다.


We propose the task of free-form and open-ended Visual Question Answering (VQA). Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. Mirroring real-world scenarios, such as helping the visually impaired, both the questions and answers are open-ended. Visual questions selectively target different areas of an image, including background details and underlying context. As a result, a system that succeeds at VQA typically needs a more detailed understanding of the image and complex reasoning than a system producing generic image captions. Moreover, VQA is amenable to automatic evaluation, since many open-ended answers contain only a few words or a closed set of answers that can be provided in a multiple-choice format. We provide a dataset containing ∼0.25M images, ∼0.76M questions, and ∼10M answers (www.visualqa.org), and discuss the information it provides. Numerous baselines for VQA are provided and compared with human performance.



* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1AREHbGNnNDDvtNJi0Sawir6cMA5BzWQZ?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    




이 논문에서는 자유형식 및 개방형 시각 질문 응답(VQA) 작업을 수행하기 위해 다양한 데이터 수집 및 모델링 접근 방식을 설명합니다. 아래는 VQA 작업을 위한 데이터 수집 과정과 모델링 방법에 대한 자세한 설명입니다.

* 데이터셋 수집

1. **실제 이미지**:
   - MS COCO 데이터셋에서 123,287개의 학습 및 검증 이미지와 81,434개의 테스트 이미지를 사용합니다. 이 데이터셋은 다양한 객체와 풍부한 맥락 정보를 포함하는 이미지를 찾기 위해 수집되었습니다.
   - 이 이미지들은 시각적으로 복잡하여 VQA 작업에 적합합니다.

2. **추상 장면**:
   - 저수준의 시각 작업이 필요 없는 고수준의 추론 연구를 위해 50,000개의 장면이 포함된 새로운 추상 장면 데이터셋을 만듭니다.
   - 이 데이터셋은 다양한 성별, 인종, 연령대의 "종이 인형" 모델과 다양한 포즈의 동물 및 객체를 포함합니다. 이를 통해 더 현실적인 장면을 만들 수 있습니다.

3. **캡션**:
   - MS COCO 데이터셋은 각 이미지에 대해 다섯 개의 단일 문장 캡션을 포함하고 있으며, 추상 장면에도 동일한 방식으로 다섯 개의 단일 캡션을 수집합니다.

4. **질문**:
   - 흥미롭고 다양한 질문을 수집하기 위해 다양한 사용자 인터페이스를 테스트한 결과, "스마트 로봇" 인터페이스가 가장 흥미롭고 다양한 질문을 유도한다는 것을 발견했습니다.
   - 각 이미지나 장면에 대해 세 개의 질문을 수집하고, 질문 작성자는 이전에 작성된 질문을 볼 수 있도록 하여 질문의 다양성을 높였습니다.
   - 총 약 760,000개의 질문을 수집했습니다.

5. **답변**:
   - 각 질문에 대해 10명의 응답자를 통해 답변을 수집합니다. 응답자는 간단한 구 또는 짧은 문장으로 답변해야 하며, 대화체나 의견을 포함하지 않도록 지시받았습니다.
   - 답변의 정확성을 평가하기 위해 응답자는 자신이 답변을 정확하게 했는지에 대해 "아니요", "아마도", "예" 중에서 선택합니다.

* 모델링 접근 방식

1. **기준선 모델**:
   - 다양한 입력 특성을 사용하는 다층 퍼셉트론(MLP) 신경망 분류기와 LSTM 모델을 사용하여 여러 기준선을 제공합니다.
   - 질문: 질문의 상위 1,000개 단어를 사용하여 bag-of-words 표현을 생성하고, 질문의 첫 번째, 두 번째, 세 번째 단어의 상위 10개를 선택하여 30차원의 bag-of-words 표현을 만듭니다.
   - 캡션: 사람이 생성한 캡션을 입력으로 사용하여 상위 1,000개 단어로 bag-of-words 표현을 만듭니다.
   - 이미지: VGGNet의 마지막 은닉층을 사용하여 4096차원의 특징을 추출합니다.

2. **테스트 작업**:
   - 개방형 답변: 모든 가능한 K개의 답변 중에서 가장 높은 활성화를 가진 답변을 선택합니다.
   - 다지선다형: 잠재적인 답변 중에서 가장 높은 활성화를 가진 답변을 선택합니다.

이러한 방법론을 통해 VQA 데이터셋의 난이도를 탐구하고, 다양한 질문 유형에 대한 정확도를 분석하여 인간의 성능과 비교합니다.



This paper details various data collection and modeling approaches for performing the task of free-form and open-ended Visual Question Answering (VQA). Below is a detailed explanation of the data collection process and modeling methods for the VQA task.

* Dataset Collection

1. **Real Images**:
   - The dataset uses 123,287 training and validation images and 81,434 test images from the MS COCO dataset. This dataset was collected to find images containing multiple objects and rich contextual information.
   - These images are visually complex and well-suited for the VQA task.

2. **Abstract Scenes**:
   - To enable research focused on high-level reasoning required for VQA without the need for low-level vision tasks, a new abstract scene dataset containing 50,000 scenes is created.
   - This dataset includes "paper doll" human models spanning various genders, races, and ages with adjustable limbs to allow for continuous pose variations. It also includes various poses of over 100 objects and 31 animals, enabling the creation of more realistic scenes.

3. **Captions**:
   - The MS COCO dataset already contains five single-sentence captions for all images, and similar captions are collected for all abstract scenes using the same user interface.

4. **Questions**:
   - To collect interesting, diverse, and well-posed questions, various user interfaces were tested. The "smart robot" interface was found to elicit the most interesting and diverse questions.
   - Three questions from unique workers were collected for each image/scene, with previous questions shown to increase diversity.
   - In total, approximately 760,000 questions were collected.

5. **Answers**:
   - For each question, answers were collected from 10 unique workers. Respondents were instructed to provide brief phrases and avoid conversational language or opinions.
   - To evaluate the accuracy, respondents were asked if they believed they answered correctly with options "no", "maybe", and "yes".

* Modeling Approaches

1. **Baseline Models**:
   - Several baselines are provided using a multi-layer perceptron (MLP) neural network classifier and an LSTM model with various input features.
   - Question: A bag-of-words representation using the top 1,000 words in the questions, and a 30-dimensional bag-of-words representation for the top 10 first, second, and third words of the questions.
   - Caption: A bag-of-words representation using the top 1,000 words in the captions.
   - Image: The last hidden layer of VGGNet is used as a 4096-dimensional feature.

2. **Testing Tasks**:
   - Open-answer: Selects the answer with the highest activation from all possible K answers.
   - Multiple-choice: Selects the answer with the highest activation from the potential answers.

This methodology explores the difficulty of the VQA dataset, analyzing the accuracy of various question types and comparing them to human performance.

<br/>
# Results  


#### VQA 데이터셋 분석

이 섹션에서는 VQA 학습 데이터셋의 질문과 답변에 대한 분석을 제공합니다. 질문 유형과 제공된 답변의 분포를 시각화하여 질문 유형별로 답변이 어떻게 다른지 탐구합니다. 또한, 이미지를 보지 않고 상식 정보만으로 질문에 답변할 수 있는 빈도를 분석합니다.

##### 질문의 유형
- 질문의 구조를 기반으로 질문을 클러스터링하여 분석합니다. 예를 들어, "What is..."로 시작하는 질문이 많은 경우 특정 정보를 요구하는 경향이 있습니다.
- 실제 이미지와 추상 장면 모두에서 유사한 질문 유형 분포를 보입니다. 이는 추상 장면이 실제 이미지와 유사한 질문을 유도한다는 것을 보여줍니다.
- 다양한 질문 유형이 있으며, 이는 시각적 이해와 상식적 추론이 필요함을 나타냅니다.

##### 질문의 길이
- 질문의 길이는 대개 4~10 단어 사이입니다.
- 질문의 길이 분포는 실제 이미지와 추상 장면 모두에서 유사합니다.

##### 답변의 유형
- 질문 유형별로 다양한 답변이 제공됩니다. 예를 들어, "Is the..."와 같은 질문은 "yes" 또는 "no"로 답변되는 경향이 있습니다.
- 다른 질문 유형, 예를 들어 "What is..."와 같은 질문은 더 다양한 답변을 가집니다.

##### 답변의 길이
- 대부분의 답변은 한 단어로 이루어져 있으며, 이는 질문이 이미지에서 특정 정보를 요구하기 때문입니다.
- 한 단어 답변이 전체 답변의 약 90%를 차지합니다.

##### 상식적 지식
- 이미지를 보지 않고 상식 정보만으로 질문에 답할 수 있는지 여부를 분석합니다. 예를 들어, "What is the color of the fire hydrant?"와 같은 질문은 상식적으로 답변할 수 있습니다.
- "yes/no" 질문의 경우 상식적으로 답변할 확률이 높지만, 기타 질문의 경우 상식 정보만으로는 약 21%만 정답을 맞출 수 있습니다. 이는 시각적 정보가 VQA에서 중요함을 나타냅니다.

#### VQA 베이스라인 및 메소드

VQA 데이터셋의 난이도를 탐구하기 위해 여러 베이스라인과 새로운 메소드를 사용합니다. 베이스라인으로 다층 퍼셉트론(MLP) 신경망 분류기와 LSTM 모델을 사용합니다.

##### 결과
- 질문만 사용하는 경우 정확도는 약 48%로, 질문 유형이 답변을 예측하는 데 중요한 정보를 제공한다는 것을 알 수 있습니다.
- 다지선다형 작업의 경우 개방형 답변 작업보다 더 나은 성능을 보입니다.
- 모든 메소드의 성능이 인간 성능보다 낮습니다.

이러한 결과는 VQA가 시각적 이해와 상식적 추론을 결합해야 하는 복잡한 작업임을 강조합니다.



#### VQA Dataset Analysis

This section provides an analysis of the questions and answers in the VQA training dataset. It visualizes the distribution of question types and the types of answers provided to explore how answers differ by question type. Additionally, it analyzes how often questions can be answered correctly using only commonsense information without seeing the image.

##### Types of Questions
- Questions are clustered based on their structure. For example, questions starting with "What is..." tend to require specific information.
- The distribution of question types is similar for both real images and abstract scenes. This indicates that abstract scenes elicit questions similar to those elicited by real images.
- A wide variety of question types exist, indicating that both visual understanding and commonsense reasoning are needed.

##### Lengths of Questions
- The length of questions typically ranges from 4 to 10 words.
- The length distribution of questions is similar for both real images and abstract scenes.

##### Types of Answers
- Different question types receive a variety of answers. For example, questions like "Is the..." tend to be answered with "yes" or "no".
- Other question types, such as "What is...", have a rich diversity of responses.

##### Lengths of Answers
- Most answers consist of a single word, as the questions tend to elicit specific information from the images.
- Single-word answers account for approximately 90% of all answers.

##### Commonsense Knowledge
- Analyzes whether questions can be answered correctly using only commonsense information without seeing the image. For example, questions like "What is the color of the fire hydrant?" can sometimes be answered using commonsense.
- For "yes/no" questions, humans perform better than chance using commonsense alone. For other questions, humans are correct about 21% of the time using commonsense alone, demonstrating the importance of visual information in VQA.

#### VQA Baselines and Methods

To explore the difficulty of the VQA dataset, several baselines and novel methods are used. Baselines include a multi-layer perceptron (MLP) neural network classifier and an LSTM model with various input features.

##### Results
- Using only the question, the accuracy is around 48%, indicating that the type of question is informative for predicting the answer.
- Multiple-choice tasks perform better than open-answer tasks.
- All methods perform significantly worse than human performance.

These results emphasize that VQA is a complex task requiring a combination of visual understanding and commonsense reasoning.


<br/>
# 예시  



#### 질문과 답변의 유형 (Types of Questions and Answers)

1. **질문 유형**:
   - 예시 1: "What is the man holding?" (남자가 들고 있는 것은 무엇인가요?)
     - 이 질문은 이미지 속 특정 객체를 식별해야 하므로 세밀한 시각적 이해가 필요합니다.
   - 예시 2: "Is the dog running?" (개가 뛰고 있나요?)
     - 이 질문은 객체의 행동을 인식해야 하므로 활동 인식이 필요합니다.
   - 예시 3: "How many apples are on the table?" (테이블 위에 사과가 몇 개 있나요?)
     - 이 질문은 이미지 속 객체의 수를 세야 하므로 객체 탐지가 필요합니다.

2. **답변 유형**:
   - 예시 1에 대한 답변: "A book" (책)
     - 이 답변은 이미지 속 객체를 정확히 식별하고 이를 설명하는 단어로 표현합니다.
   - 예시 2에 대한 답변: "Yes" (네)
     - 이 답변은 객체의 활동을 인식한 후 이에 대한 단순한 확인을 제공합니다.
   - 예시 3에 대한 답변: "Three" (세 개)
     - 이 답변은 이미지 속 객체의 수를 세고 이를 숫자로 표현합니다.

#### 데이터셋 수집 과정 (Data Collection Process)

1. **실제 이미지**:
   - MS COCO 데이터셋에서 다양한 객체와 복잡한 장면을 포함하는 이미지를 선택합니다.
   - 예시: 이미지에는 여러 사람이 공원에서 피크닉을 하고 있는 장면이 포함됩니다.

2. **추상 장면**:
   - 고수준 추론 연구를 위해 추상 장면 데이터셋을 생성합니다.
   - 예시: 추상 장면에는 다양한 포즈의 종이 인형과 동물이 포함된 장면이 포함됩니다.

3. **질문 수집**:
   - Amazon Mechanical Turk을 통해 각 이미지나 장면에 대해 흥미롭고 다양한 질문을 수집합니다.
   - 예시: "What color is the boy's shirt?" (소년의 셔츠는 무슨 색인가요?)

4. **답변 수집**:
   - 각 질문에 대해 10명의 고유 응답자를 통해 답변을 수집합니다.
   - 예시: 질문 "What color is the boy's shirt?"에 대한 답변으로 "Red" (빨간색), "Blue" (파란색) 등이 제공될 수 있습니다.

#### 모델링 접근 방식 (Modeling Approaches)

1. **기준선 모델**:
   - MLP 신경망 분류기와 LSTM 모델을 사용하여 다양한 입력 특성(질문, 캡션, 이미지)으로 예측을 수행합니다.
   - 예시: 질문 "What is the man holding?"에 대해 이미지 특징과 질문의 단어 벡터를 결합하여 답변을 예측합니다.

2. **테스트 작업**:
   - 개방형 답변과 다지선다형 작업으로 결과를 평가합니다.
   - 예시: 개방형 답변 작업에서 모델이 "A book"이라고 예측하고, 다지선다형 작업에서 주어진 옵션 중에서 "A book"을 선택합니다.



#### Types of Questions and Answers

1. **Question Types**:
   - Example 1: "What is the man holding?"
     - This question requires identifying a specific object in the image, necessitating detailed visual understanding.
   - Example 2: "Is the dog running?"
     - This question requires recognizing the activity of an object, necessitating activity recognition.
   - Example 3: "How many apples are on the table?"
     - This question requires counting objects in the image, necessitating object detection.

2. **Answer Types**:
   - Answer to Example 1: "A book"
     - This answer identifies and describes the specific object in the image.
   - Answer to Example 2: "Yes"
     - This answer confirms the recognition of the object's activity.
   - Answer to Example 3: "Three"
     - This answer counts the number of objects in the image and expresses it numerically.

#### Data Collection Process

1. **Real Images**:
   - Select images from the MS COCO dataset that include various objects and complex scenes.
   - Example: An image includes a scene of several people having a picnic in a park.

2. **Abstract Scenes**:
   - Create an abstract scene dataset for high-level reasoning research.
   - Example: An abstract scene includes various poses of paper doll models and animals.

3. **Question Collection**:
   - Collect interesting and diverse questions for each image or scene via Amazon Mechanical Turk.
   - Example: "What color is the boy's shirt?"

4. **Answer Collection**:
   - Collect answers from 10 unique respondents for each question.
   - Example: Answers to "What color is the boy's shirt?" might include "Red" and "Blue".

#### Modeling Approaches

1. **Baseline Models**:
   - Use MLP neural network classifiers and LSTM models to predict answers with various input features (questions, captions, images).
   - Example: For the question "What is the man holding?", combine image features and question word vectors to predict the answer.

2. **Testing Tasks**:
   - Evaluate results with open-answer and multiple-choice tasks.
   - Example: In the open-answer task, the model predicts "A book", and in the multiple-choice task, it selects "A book" from the given options.


<br/>  
# 요약 



이 논문에서는 자유형식 및 개방형 시각 질문 응답(VQA) 작업을 제안합니다. MS COCO 데이터셋을 포함한 다양한 이미지와 추상 장면에서 질문과 답변을 수집하여 약 0.76M개의 질문과 10M개의 답변을 포함하는 대규모 데이터셋을 만듭니다. 질문은 이미지의 세부 사항과 맥락을 이해하고 복잡한 추론을 필요로 하며, 답변은 주로 간단한 단어 또는 짧은 문장으로 구성됩니다. 다양한 기준선 모델과 LSTM 모델을 사용하여 VQA 작업을 수행하고, 인간 성능과 비교하여 평가합니다. 이 연구는 VQA가 시각적 이해와 상식적 추론을 결합한 복잡한 AI 작업임을 강조합니다.



This paper proposes the task of free-form and open-ended Visual Question Answering (VQA). A large dataset is created, comprising approximately 0.76M questions and 10M answers collected from various images and abstract scenes, including the MS COCO dataset. The questions require understanding the details and context of the image, involving complex reasoning, while the answers are mainly simple words or short phrases. Various baseline models and LSTM models are used to perform the VQA task, and their performance is evaluated against human performance. This study emphasizes that VQA is a complex AI task that combines visual understanding and commonsense reasoning.


# 기타  


* LSTM과 FCN(전완전 연결 신경망)을 사용한 접근 방식   

### VQA에서 LSTM과 FCN의 사용 방법

#### 1. 이미지 특징 추출
이미지에서 객체를 감지하거나 특징을 추출하기 위해 CNN(Convolutional Neural Network)을 사용합니다. 이 과정에서 보통 VGGNet이나 ResNet 같은 사전 훈련된 모델의 마지막 은닉층을 사용하여 이미지 특징 벡터를 얻습니다. 

- **예시**: VGGNet의 마지막 은닉층에서 4096차원의 특징 벡터를 추출합니다.

#### 2. 질문 인코딩
자연어 질문을 처리하기 위해 LSTM(Long Short-Term Memory) 네트워크를 사용합니다. LSTM은 질문의 단어 시퀀스를 인코딩하여 고정 길이의 벡터 표현으로 변환합니다.

- **예시**: "What is the man holding?" 질문을 LSTM에 입력하여 1024차원의 벡터로 인코딩합니다.

#### 3. 특징 결합
이미지 특징 벡터와 질문 벡터를 결합하여 최종 답변을 예측하는데 사용됩니다. 이를 결합하는 방법에는 여러 가지가 있습니다. 대표적으로는 두 벡터를 단순히 concatenation(연결) 하거나 element-wise multiplication(원소별 곱셈)을 사용할 수 있습니다.

- **예시**: 이미지 벡터 (4096차원)와 질문 벡터 (1024차원)를 원소별 곱셈으로 결합하여 4096차원의 벡터를 만듭니다.

#### 4. 답변 예측
결합된 벡터를 FCN에 입력하여 최종 답변을 예측합니다. 이 FCN은 여러 개의 은닉층으로 구성될 수 있으며, 각 층에서 비선형 활성화 함수를 사용합니다.

- **예시**: FCN은 결합된 벡터를 입력받아, 1000개의 가장 빈번한 답변 중 하나를 예측하는 소프트맥스 출력층을 사용합니다.

### 요약
- CNN을 사용하여 이미지 특징을 추출합니다.
- LSTM을 사용하여 질문을 인코딩합니다.
- 이미지 특징 벡터와 질문 벡터를 결합합니다.
- FCN을 사용하여 최종 답변을 예측합니다.

* How it works?  

#### 1. Image Feature Extraction
A CNN (Convolutional Neural Network) is used to detect objects or extract features from the image. Typically, a pre-trained model like VGGNet or ResNet is used to obtain a feature vector from the last hidden layer.

- **Example**: Extract a 4096-dimensional feature vector from the last hidden layer of VGGNet.

#### 2. Question Encoding
An LSTM (Long Short-Term Memory) network is used to process the natural language question. The LSTM encodes the sequence of words in the question into a fixed-length vector representation.

- **Example**: Encode the question "What is the man holding?" into a 1024-dimensional vector using an LSTM.

#### 3. Feature Fusion
The image feature vector and the question vector are combined to predict the final answer. This can be done in several ways, such as simple concatenation or element-wise multiplication of the two vectors.

- **Example**: Combine the image vector (4096-dimensional) and the question vector (1024-dimensional) using element-wise multiplication to create a 4096-dimensional vector.

#### 4. Answer Prediction
The combined vector is fed into an FCN to predict the final answer. The FCN can consist of multiple hidden layers, each using nonlinear activation functions.

- **Example**: The FCN takes the combined vector as input and uses a softmax output layer to predict one of the 1000 most frequent answers.

### Summary
- A CNN is used to extract image features.
- An LSTM is used to encode the question.
- The image feature vector and question vector are fused.
- An FCN is used to predict the final answer.


# 기타2  

현재는 대부분 Transformer 기반 모델을 사용하는데 이와 달리 CNN을 이용해서 추출한 이미지 특징 벡터와 LSTM에서 얻은 질문 벡터를 원소별 곱셈방식으로 결합, FCN을 통해 답변을 예측하는 일련의 과정들이 흥미로웠다.
Currently, most models are based on Transformers. 
However, this paper's approach of predicting answers was particularly interesting. 
It combined image feature vectors extracted using CNNs and question vectors obtained from LSTMs through element-wise multiplication. 
This was followed by an FCN to make the final prediction.

사실 지금의 많은 비전과 nlp 융합 연구 결과들이 이 논문과 같은 선행 연구들을 거쳐서 왔다는 것이 새삼 느껴졌다. 
It made me realize that many of today's vision and NLP fusion research results have evolved from pioneering studies like this one.



<br/>
# refer format:     
Antol, Stanislaw, Agrawal, Aishwarya, Lu, Jiasen, Mitchell, Margaret, Batra, Dhruv, Zitnick, C. Lawrence, & Parikh, Devi. (2015). VQA: Visual Question Answering. In 2015 IEEE International Conference on Computer Vision (ICCV) (pp. 2425-2433). IEEE. doi:10.1109/ICCV.2015.279
  
@inproceedings{antol2015vqa,
  title={VQA: Visual Question Answering},
  author={Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
  booktitle={2015 IEEE International Conference on Computer Vision (ICCV)},
  pages={2425--2433},
  year={2015},
  organization={IEEE},
  doi={10.1109/ICCV.2015.279}
}
