---
layout: post
title:  "[2021]Learning Transferable Visual Models From Natural Language Supervision"  
date:   2024-06-26 22:51:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    



컴퓨터 비전 시스템은 최첨단 기술로 고정된 사전 결정된 객체 카테고리를 예측하도록 훈련됩니다. 이러한 제한된 감독 형태는 추가 레이블 데이터가 필요할 때마다 다른 시각적 개념을 지정해야 하기 때문에 시스템의 범용성과 사용성을 제한합니다. 이미지를 설명하는 원시 텍스트로부터 직접 학습하는 것은 훨씬 더 넓은 감독 소스를 활용할 수 있는 유망한 대안입니다. 우리는 인터넷에서 수집한 4억 쌍의 (이미지, 텍스트) 데이터셋에서 캡션과 이미지를 예측하는 간단한 사전 훈련 작업이 최첨단 이미지 표현을 처음부터 학습하는 효율적이고 확장 가능한 방법임을 보여줍니다. 사전 훈련 후에는 자연어를 사용하여 학습된 시각적 개념을 참조하거나 새로운 개념을 설명하여 모델을 다운스트림 작업으로 제로샷 전이할 수 있습니다. 우리는 OCR, 비디오의 동작 인식, 지리적 위치 확인, 다양한 세분화된 객체 분류와 같은 작업을 포함하여 30개 이상의 기존 컴퓨터 비전 데이터셋을 벤치마킹하여 이 접근법의 성능을 연구합니다. 모델은 대부분의 작업에 비트리비얼하게 전이되며, 데이터셋별 훈련 없이도 종종 완전한 감독 방식의 기준과 경쟁합니다. 예를 들어, ImageNet의 원래 ResNet-50의 정확도를 제로샷으로 일치시키면서도 128만 개의 훈련 예제를 사용할 필요가 없습니다. 우리는 우리의 코드와 사전 훈련된 모델 가중치를 https://github.com/OpenAI/CLIP에서 공개합니다.  


State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP.  



* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1PQKSJPuf0uGkqLaUSW9CTkGJVBmaBaFL?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    



1. **데이터 수집**: 인터넷에서 4억 쌍의 이미지와 텍스트 데이터를 수집합니다. 이 데이터는 모델이 다양한 시각적 개념을 학습하는 데 사용됩니다.

2. **모델 아키텍처**:
   - **이미지 인코더**: 이미지를 임베딩 벡터로 변환하는 신경망입니다.
   - **텍스트 인코더**: 텍스트(캡션)를 임베딩 벡터로 변환하는 신경망입니다.

3. **대조 학습**: 이미지와 텍스트 쌍이 맞는지를 예측하는 대조 학습 방법을 사용합니다. 즉, 주어진 이미지에 맞는 텍스트 캡션을 예측하는 대신, 주어진 이미지와 텍스트 쌍이 올바른지 여부를 예측합니다.

4. **훈련 과정**:
   - **임베딩 생성**: 이미지와 텍스트 각각에 대해 임베딩 벡터를 생성합니다.
   - **대조 손실 함수**: 올바른 이미지-텍스트 쌍의 임베딩 벡터 간의 코사인 유사도를 최대화하고, 잘못된 쌍의 유사도를 최소화하는 손실 함수를 최적화합니다.

5. **제로샷 전이**: 모델이 학습된 후, 추가적인 학습 없이 새로운 데이터셋에 대해 바로 적용할 수 있습니다. 이는 모델이 학습 과정에서 이미 다양한 시각적 개념을 이해하고 있기 때문입니다.

이 방법론은 전통적인 이미지 분류 방법보다 더 유연하고 확장 가능하며, 새로운 시각적 개념을 학습하기 위해 추가적인 레이블이 필요 없다는 장점이 있습니다.



1. **Data Collection**: Collect 400 million pairs of images and text from the internet. This data is used for the model to learn various visual concepts.

2. **Model Architecture**:
   - **Image Encoder**: A neural network that converts images into embedding vectors.
   - **Text Encoder**: A neural network that converts text (captions) into embedding vectors.

3. **Contrastive Learning**: Use a contrastive learning method to predict whether an image and text pair match. Instead of predicting the caption for a given image, the model predicts whether a given image and text pair is correct.

4. **Training Process**:
   - **Generating Embeddings**: Generate embedding vectors for both images and text.
   - **Contrastive Loss Function**: Optimize a loss function that maximizes the cosine similarity between the embedding vectors of correct image-text pairs and minimizes the similarity for incorrect pairs.

5. **Zero-Shot Transfer**: After the model is trained, it can be directly applied to new datasets without additional training. This is because the model has already learned various visual concepts during the training process.

This methodology is more flexible and scalable than traditional image classification methods and eliminates the need for additional labels to learn new visual concepts.





<br/>
# Results  



이 연구에서는 다양한 컴퓨터 비전 데이터셋을 통해 모델의 성능을 평가했습니다. 주요 결과는 다음과 같습니다:

1. **다양한 작업에서의 성능**: OCR, 비디오에서의 동작 인식, 지리적 위치 확인, 세분화된 객체 분류 등 30개 이상의 다양한 컴퓨터 비전 작업에서 모델의 성능을 벤치마킹했습니다. 모델은 대부분의 작업에서 비트리비얼하게 전이되었으며, 종종 완전한 감독 방식의 기준 모델과 경쟁력 있는 성능을 보였습니다.

2. **제로샷 전이 성능**: 모델은 추가적인 데이터셋 특화 훈련 없이도 종종 경쟁력 있는 성능을 발휘했습니다. 예를 들어, ImageNet에서 원래의 ResNet-50 모델과 제로샷으로 일치하는 정확도를 달성했습니다. 이는 128만 개의 훈련 예제를 사용하지 않고도 가능했습니다.

3. **모델의 유연성**: 학습된 시각적 개념을 자연어로 참조하거나 새로운 개념을 설명함으로써, 모델은 다양한 다운스트림 작업으로 제로샷 전이될 수 있었습니다.

4. **코드 및 사전 훈련된 모델 가중치 공개**: 연구팀은 그들의 코드와 사전 훈련된 모델 가중치를 공개하여, 다른 연구자들이 이 방법을 활용할 수 있도록 했습니다.


The study evaluated the model's performance across various computer vision datasets. The key results are as follows:

1. **Performance on Various Tasks**: The model's performance was benchmarked on over 30 different computer vision tasks, including OCR, action recognition in videos, geo-localization, and fine-grained object classification. The model transferred non-trivially to most tasks and was often competitive with fully supervised baseline models.

2. **Zero-Shot Transfer Performance**: The model frequently achieved competitive performance without the need for additional dataset-specific training. For instance, it matched the accuracy of the original ResNet-50 on ImageNet zero-shot, without using any of the 1.28 million training examples.

3. **Model Flexibility**: By referencing learned visual concepts with natural language (or describing new ones), the model could be zero-shot transferred to various downstream tasks.

4. **Code and Pre-trained Model Weights Release**: The research team released their code and pre-trained model weights, allowing other researchers to leverage this approach.





<br/>
# 예시  



이 연구에서는 몇 가지 실제 예시를 통해 제안된 방법론의 효과를 보여줍니다:

1. **ImageNet 제로샷 분류**: 모델은 추가적인 훈련 없이 ImageNet 데이터셋에서 높은 성능을 달성했습니다. 예를 들어, '푸들'이라는 텍스트 설명과 일치하는 이미지를 정확하게 식별할 수 있었습니다. 이 예시는 모델이 텍스트와 이미지를 연결하는 능력을 보여줍니다.

2. **OCR (광학 문자 인식)**: 모델은 자연스러운 이미지에서 텍스트를 정확하게 인식했습니다. 예를 들어, 거리 표지판의 이미지를 보고 해당 텍스트를 정확히 추출할 수 있었습니다. 이 예시는 제로샷 설정에서 다양한 텍스트 인식 작업에 모델이 적용될 수 있음을 보여줍니다.

3. **동작 인식**: 모델은 비디오 클립에서 다양한 동작을 인식하는 데 성공했습니다. 예를 들어, 축구 경기의 비디오 클립을 보고 '공을 차다'라는 동작을 정확히 인식할 수 있었습니다. 이는 모델이 비디오 데이터를 효과적으로 처리할 수 있음을 입증합니다.

4. **지리적 위치 확인**: 모델은 이미지의 지리적 위치를 추정하는 데 탁월한 성능을 보였습니다. 예를 들어, 특정 도시의 랜드마크 사진을 보고 해당 위치를 정확히 추측할 수 있었습니다. 이 예시는 모델이 지리적 데이터셋에서도 높은 성능을 보일 수 있음을 보여줍니다.


The study demonstrates the effectiveness of the proposed methodology through several real examples:

1. **ImageNet Zero-Shot Classification**: The model achieved high performance on the ImageNet dataset without additional training. For example, it accurately identified images that match the text description "poodle." This example shows the model's ability to connect text and images.

2. **OCR (Optical Character Recognition)**: The model accurately recognized text from natural images. For instance, it could correctly extract text from an image of a street sign. This example demonstrates that the model can be applied to various text recognition tasks in a zero-shot setting.

3. **Action Recognition**: The model successfully recognized different actions in video clips. For example, it accurately identified the action "kicking a ball" in a soccer game video clip. This demonstrates the model's effectiveness in processing video data.

4. **Geo-Localization**: The model excelled at estimating the geographic location of images. For example, it accurately guessed the location of a landmark in a specific city from a photo. This example shows that the model can also perform well on geographic datasets.



<br/>  
# 요약 


이 연구에서는 이미지 인코더와 텍스트 인코더를 결합하여 4억 쌍의 이미지-텍스트 데이터로 모델을 훈련시켰습니다. 제안된 모델은 다양한 컴퓨터 비전 작업에서 높은 성능을 보였으며, 추가적인 데이터셋 특화 훈련 없이도 경쟁력 있는 성능을 발휘했습니다. 특히, OCR, 비디오 동작 인식, 지리적 위치 확인 등에서 우수한 결과를 나타냈습니다. 이러한 접근 방식은 새로운 시각적 개념을 학습하기 위해 추가적인 레이블이 필요 없다는 장점이 있습니다. 연구팀은 그들의 코드와 사전 훈련된 모델 가중치를 공개하여 다른 연구자들이 활용할 수 있도록 했습니다.  



This study combines image and text encoders to train a model using 400 million image-text pairs. The proposed model achieved high performance across various computer vision tasks and demonstrated competitive results without additional dataset-specific training. It excelled particularly in OCR, action recognition in videos, and geo-localization. This approach eliminates the need for additional labels to learn new visual concepts. The research team has made their code and pre-trained model weights available for other researchers to use.  



# 기타  




<br/>
# refer format:     
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020. https://doi.org/10.48550/arXiv.2103.00020    
  
@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021},
  url={https://doi.org/10.48550/arXiv.2103.00020}
}

