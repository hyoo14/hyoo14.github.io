---
layout: post
title:  "[2021]An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale(ViT:VisionTransformer)"  
date:   2024-10-09 15:16:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    



Transformer 아키텍처는 자연어 처리(NLP)에서 표준으로 자리 잡았지만, 컴퓨터 비전에서는 여전히 제한적으로 사용되었습니다. 기존의 비전 모델은 주로 CNN(Convolutional Neural Network)과 결합된 형태로 사용되거나 CNN의 일부 구성 요소를 대체하는 방식으로 주로 사용되었습니다. 이 연구에서는 CNN에 의존하지 않고 이미지 패치를 순차적으로 처리하는 순수 Transformer 모델인 Vision Transformer(ViT)를 제안했습니다. ViT는 대량의 데이터로 사전 학습한 후, 중형 또는 소형 이미지 인식 데이터셋에 전이 학습을 적용했을 때 우수한 성능을 보였습니다. ImageNet, CIFAR-100, VTAB 등의 데이터셋에서 기존 최첨단 CNN 모델과 비교했을 때 더 적은 계산 자원으로도 뛰어난 성능을 달성했습니다.


While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary, and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train【5†source】.




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



이 연구에서는 Vision Transformer(ViT) 모델의 설계와 훈련 방법론을 제시합니다. ViT는 이미지의 2D 구조를 직접적으로 처리하기 위해 이미지 전체를 고정 크기의 패치로 나누고, 각 패치를 1D 시퀀스 형태로 변환하여 Transformer에 입력합니다. 구체적으로, 이미지를 (H, W, C) 크기의 2D 데이터로 보고, 이를 (P, P) 크기의 작은 패치들로 나누어 각 패치를 벡터로 변환합니다. 그런 다음, 이 패치들에 위치 임베딩을 더해 Transformer에 입력합니다. ViT는 NLP에서의 BERT와 유사하게 "classification token"을 추가하여, 마지막 레이어에서 이 토큰을 이미지의 대표값으로 사용합니다. 

모델은 대규모 데이터셋으로 사전 학습(예: ImageNet-21k, JFT-300M)을 수행한 후, ImageNet이나 CIFAR-100과 같은 작은 데이터셋에 전이 학습을 적용하여 이미지 분류 성능을 향상시킵니다. 훈련 시에는 Adam 최적화를 사용하며, 학습 초기에는 선형적인 학습률 증가를 적용한 후 학습 후반에 학습률을 점차 감소시킵니다. 사전 학습된 ViT 모델은 더 높은 해상도로 미세 조정하여 다양한 이미지 인식 태스크에서 우수한 성능을 보였습니다.


This study introduces the methodology for designing and training the Vision Transformer (ViT) model. ViT is designed to handle the 2D structure of images directly by splitting an entire image into fixed-size patches and converting these patches into a 1D sequence for input into the Transformer. Specifically, an image is treated as a 2D input of size (H, W, C) and is divided into small patches of size (P, P). Each patch is then converted into a vector and combined with position embeddings before being input to the Transformer. Similar to BERT in NLP, ViT includes an additional "classification token" that serves as a representation of the image in the last layer.

The model is pre-trained on large datasets (e.g., ImageNet-21k, JFT-300M) and then fine-tuned on smaller datasets like ImageNet or CIFAR-100 to improve image classification performance. During training, the Adam optimizer is used, with a linear learning rate warm-up at the beginning, followed by a gradual decay of the learning rate. Pre-trained ViT models are fine-tuned at higher resolutions, achieving excellent performance across various image recognition tasks.

 

<br/>
# Results  



이 연구의 결과는 Vision Transformer(ViT)가 대규모 사전 학습 후 다양한 이미지 인식 태스크에서 우수한 성능을 보였음을 보여줍니다. ViT는 기존의 최첨단 모델들인 Big Transfer(BiT)와 Noisy Student(EfficientNet)와 비교되었습니다. 특히 ImageNet, CIFAR-100, ImageNet-ReaL, Oxford-IIIT Pets, Oxford Flowers-102, 그리고 VTAB(19개 과제 포함) 데이터셋에서 성능을 평가했습니다.

ImageNet 데이터셋에서, ViT-H/14 모델은 88.55%의 정확도를 기록하며, 이는 BiT와 Noisy Student 모델보다 더 높은 성능을 보였습니다. ImageNet-ReaL에서는 90.72%의 정확도를 달성했고, CIFAR-100에서는 94.55%의 정확도를 기록하여 다른 모델들보다 성능이 높았습니다. 또한, VTAB의 19개 과제 세트에서 77.63%의 정확도를 보였으며, 이는 다양한 데이터셋에 대한 전이 학습 성능이 뛰어남을 나타냅니다.

이러한 결과는 ViT가 대규모 데이터셋에서 사전 학습할 때, 기존의 ResNet 기반 모델들보다 더 적은 계산 자원을 사용하면서도 더 나은 성능을 보일 수 있음을 보여줍니다. 특히, ViT는 고해상도 데이터와 큰 데이터셋에서 사전 학습되었을 때 성능이 크게 향상되었습니다.


The results of this study demonstrate that the Vision Transformer (ViT) achieves superior performance on various image recognition tasks after extensive pre-training on large datasets. ViT was compared to state-of-the-art methods like Big Transfer (BiT) and Noisy Student (EfficientNet). The evaluation was conducted on datasets such as ImageNet, CIFAR-100, ImageNet-ReaL, Oxford-IIIT Pets, Oxford Flowers-102, and VTAB (comprising 19 tasks).

On the ImageNet dataset, the ViT-H/14 model achieved an accuracy of 88.55%, outperforming both BiT and Noisy Student models. It reached an accuracy of 90.72% on ImageNet-ReaL and recorded 94.55% accuracy on CIFAR-100, which surpassed other models. Furthermore, ViT demonstrated strong transfer learning capabilities with a 77.63% accuracy across the 19 tasks in the VTAB suite.

These results indicate that when pre-trained on large datasets, ViT can surpass ResNet-based models in performance while requiring fewer computational resources. Notably, ViT's performance improves significantly when trained on high-resolution data and large-scale datasets.


<br/>
# 예시  




Vision Transformer(ViT) 논문에서는 ViT 모델의 성능을 평가하기 위해 다양한 이미지 분류 태스크를 수행했습니다. 주요 예시는 다음과 같습니다:

1. **ImageNet**: ImageNet은 가장 널리 사용되는 이미지 분류 데이터셋 중 하나로, 1000개의 서로 다른 클래스(예: 개, 고양이, 자동차 등)로 구성된 약 130만 장의 이미지가 포함되어 있습니다. ViT는 이 데이터셋에서 높은 정확도를 달성하며, 기존의 CNN 모델들과 성능을 비교했습니다.

2. **CIFAR-100**: CIFAR-100은 100개의 클래스가 있는 작은 이미지 데이터셋입니다. 각 클래스는 다양한 동물, 사물 등을 포함하고 있습니다. ViT는 이 데이터셋에서도 높은 분류 성능을 보여주며, 기존의 CNN 기반 모델보다 더 적은 계산 자원으로도 뛰어난 결과를 보였습니다.

3. **VTAB (Visual Task Adaptation Benchmark)**: VTAB은 다양한 시각적 인식 과제를 포함한 19개의 태스크로 구성된 데이터셋입니다. 여기에는 자연 이미지, 의료 이미지, 위성 이미지 등 다양한 데이터가 포함되어 있습니다. ViT는 VTAB에서 다양한 태스크에 대한 전이 학습 성능을 평가받았고, 기존 모델보다 우수한 성능을 보여줬습니다.

4. **Oxford-IIIT Pets**: 이 데이터셋은 고양이와 개의 다양한 품종을 포함한 이미지들로 구성되어 있으며, 각 이미지를 특정 품종으로 분류하는 과제입니다. ViT는 이 데이터셋에서 각 품종을 정확하게 분류하는 데 탁월한 성능을 보였습니다.

5. **Oxford Flowers-102**: 102종의 꽃 이미지로 구성된 이 데이터셋은 각 꽃 이미지를 해당 종으로 분류하는 과제입니다. ViT는 이 데이터셋에서도 높은 정확도를 기록하며, 복잡한 이미지에서도 잘 일반화할 수 있음을 입증했습니다.


The Vision Transformer (ViT) paper evaluated the performance of the ViT model through various image classification tasks. The key examples are as follows:

1. **ImageNet**: ImageNet is one of the most widely used image classification datasets, containing about 1.3 million images across 1,000 different classes (e.g., dog, cat, car). ViT achieved high accuracy on this dataset and compared its performance against traditional CNN models.

2. **CIFAR-100**: CIFAR-100 is a small image dataset with 100 classes, each representing various animals, objects, etc. ViT demonstrated high classification performance on this dataset as well, achieving superior results with fewer computational resources compared to CNN-based models.

3. **VTAB (Visual Task Adaptation Benchmark)**: VTAB consists of 19 tasks, including various visual recognition challenges such as natural images, medical images, and satellite images. ViT's transfer learning performance was evaluated across these diverse tasks, and it showed superior performance over existing models.

4. **Oxford-IIIT Pets**: This dataset consists of images of various breeds of cats and dogs, with the task of classifying each image into a specific breed. ViT showed excellent performance in accurately classifying each breed in this dataset.

5. **Oxford Flowers-102**: Comprising images of 102 types of flowers, this dataset involves classifying each flower image into its corresponding species. ViT achieved high accuracy on this dataset as well, demonstrating its ability to generalize well even with complex images.
 




<br/>  
# 요약 



Vision Transformer(ViT) 논문에서는 기존의 CNN에 의존하지 않고 순수 Transformer 아키텍처를 사용하여 이미지를 처리하는 방법론을 제시합니다. ViT는 이미지를 작은 패치로 나누어 시퀀스 형태로 처리하며, 이를 통해 이미지 분류 태스크에서 뛰어난 성능을 발휘합니다. ViT는 ImageNet, CIFAR-100, VTAB과 같은 대규모 데이터셋에서 사전 학습을 거친 후, 높은 정확도를 기록하며 CNN 기반 모델을 능가하는 성과를 보였습니다. 특히, 다양한 시각적 인식 태스크에서도 뛰어난 전이 학습 능력을 보여주며, Oxford-IIIT Pets와 Oxford Flowers-102 데이터셋에서 우수한 분류 성능을 입증했습니다. ViT는 기존의 모델들보다 적은 계산 자원으로도 높은 성능을 달성할 수 있음을 증명한 연구입니다.


The Vision Transformer (ViT) paper introduces a methodology for processing images using a pure Transformer architecture without relying on traditional CNNs. ViT divides images into small patches and processes them as a sequence, achieving strong performance in image classification tasks. After pre-training on large datasets such as ImageNet, CIFAR-100, and VTAB, ViT recorded high accuracy, surpassing CNN-based models. It demonstrated excellent transfer learning capabilities across various visual recognition tasks, including achieving strong classification performance on the Oxford-IIIT Pets and Oxford Flowers-102 datasets. The study shows that ViT can achieve high performance with fewer computational resources compared to traditional models.


# 기타  


Vision Transformer(ViT)와 같은 모델은 비전 분야에서 프리트레이닝할 때 주로 **이미지 분류**(image classification)를 사용합니다. 이는 언어 모델에서 흔히 사용하는 마스크 예측(masked prediction) 방식과는 다릅니다.


- **프리트레이닝 과정**에서는 대규모 이미지 데이터셋(예: ImageNet-21k, JFT-300M)을 사용하여 모델이 이미지 분류 작업을 수행하도록 학습시킵니다. 모델은 다양한 이미지를 보고 해당 이미지에 맞는 레이블(예: 고양이, 강아지 등)을 예측하도록 학습됩니다.
- 이 과정에서 모델은 이미지의 특징을 추출하고, 이미지 내의 패턴과 객체를 학습하게 됩니다. 이를 통해 ViT는 이미지 분류와 같은 작업에 대한 일반적인 지식을 쌓게 됩니다.
- 이후, 이렇게 학습된 모델을 다른 데이터셋에 **전이 학습**(fine-tuning)하여 특정한 이미지 인식 작업에 맞게 최적화합니다. 이 과정에서 모델은 소규모 데이터셋에 적응하여 성능을 높입니다.

즉, 비전 분야에서의 프리트레이닝은 기본적으로 대규모 이미지 분류를 통해 이루어지며, 이는 모델이 이미지 데이터에서 유용한 특징을 학습하게 하는 방식입니다. 따라서 언어 모델의 마스크 예측과는 달리, ViT는 이미지 자체의 레이블을 예측하는 작업을 통해 프리트레이닝됩니다.






Vision Transformer (ViT) and similar models primarily use **image classification** during pre-training in the vision domain. This differs from the masked prediction method commonly used in language models.

- **Pre-training process**: Large-scale image datasets (e.g., ImageNet-21k, JFT-300M) are used to train the model on image classification tasks. The model learns to predict the correct labels (e.g., cat, dog) for various images it encounters.
- During this process, the model extracts features from the images and learns patterns and objects within the images. Through this, ViT accumulates general knowledge for tasks like image classification.
- Afterward, the pre-trained model undergoes **fine-tuning** on other datasets, optimizing it for specific image recognition tasks. This allows the model to adapt to smaller datasets and improve its performance.

In summary, pre-training in the vision field is primarily done through large-scale image classification, enabling the model to learn useful features from image data. Thus, unlike the masked prediction in language models, ViT is pre-trained by predicting labels of the images themselves.


<br/>
# refer format:     



@inproceedings{Dosovitskiy2021,
  author = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  title = {An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year = {2021},
  url = {https://arxiv.org/abs/2010.11929}
}





Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” In Proceedings of the International Conference on Learning Representations (ICLR) 2021. https://arxiv.org/abs/2010.11929.