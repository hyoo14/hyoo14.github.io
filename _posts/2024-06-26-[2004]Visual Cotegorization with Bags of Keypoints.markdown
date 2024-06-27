---
layout: post
title:  "[2004]Visual Cotegorization with Bags of Keypoints"  
date:   2024-06-27 12:50:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    


우리는 자연 이미지의 객체 내용을 식별하면서 객체 클래스에 내재된 변형을 일반화하는 문제를 해결하기 위한 새로운 방법을 제시합니다. 이 키포인트의 백(bag of keypoints) 방법은 이미지 패치의 아핀 불변 서술자의 벡터 양자화에 기반합니다. 우리는 나이브 베이즈와 SVM이라는 두 가지 다른 분류기를 사용하는 두 가지 대안적 구현 방법을 제안하고 비교합니다. 이 방법의 주요 장점은 간단하고 계산 효율적이며 본질적으로 불변하다는 것입니다. 우리는 일곱 가지 의미적 시각적 범주를 동시에 분류한 결과를 제시합니다. 이 결과들은 이 방법이 배경 잡음에 강하며 기하학적 정보를 활용하지 않고도 좋은 분류 정확도를 나타낸다는 것을 명확히 보여줍니다.



We present a novel method for generic visual categorization: the problem of identifying the object content of natural images while generalizing across variations inherent to the object class. This bag of keypoints method is based on vector quantization of affine invariant descriptors of image patches. We propose and compare two alternative implementations using different classifiers: Naïve Bayes and SVM. The main advantages of the method are that it is simple, computationally efficient, and intrinsically invariant. We present results for simultaneously classifying seven semantic visual categories. These results clearly demonstrate that the method is robust to background clutter and produces good categorization accuracy even without exploiting geometric information.


* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1-Kiar_yb0UN_4lcwKMoxljg_zLcfOblK?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    


방법론은 다음의 주요 단계로 구성됩니다:
1. 이미지 패치의 탐지 및 설명
2. 벡터 양자화 알고리즘을 사용하여 패치 설명자를 사전 결정된 클러스터 집합(어휘)에 할당
3. 각 클러스터에 할당된 패치의 수를 세는 키포인트의 백(bag of keypoints)을 구성
4. 키포인트의 백을 특징 벡터로 취급하여 다중 클래스 분류기를 적용하고, 이를 통해 이미지에 할당할 범주나 범주를 결정

이 단계들은 분류 정확도를 최대화하면서 계산 노력을 최소화하도록 설계되었습니다. 따라서 첫 번째 단계에서 추출된 설명자는 이미지 변환, 조명 변화 및 가리기와 같은 분류 작업과 무관한 변화를 제외하는 동시에 범주 수준에서 구별할 수 있는 충분한 정보를 담아야 합니다. 두 번째 단계에서 사용된 어휘는 이미지 부분의 관련 변화를 구별할 수 있을 만큼 충분히 커야 하지만, 노이즈와 같은 무관한 변화를 구별할 만큼 너무 크지 않아야 합니다.

키포인트의 백 접근법은 텍스트 분류를 위한 단어 백 접근법을 학습하는 방법과 유사한 비유로 동기화될 수 있습니다. 우리의 경우, "단어"는 "눈"이나 "자동차 바퀴"와 같은 반복 가능한 의미를 가지지 않을 수도 있으며 어휘를 선택하는 명확한 최선의 선택이 없을 수 있습니다. 우리의 목표는 주어진 학습 데이터셋에서 좋은 분류 성능을 허용하는 어휘를 사용하는 것입니다. 따라서 시스템을 학습하는 단계에서는 여러 가능한 어휘를 고려할 수 있습니다.


methodology consists of the following main steps:
1. Detection and description of image patches
2. Assigning patch descriptors to a set of predetermined clusters (a vocabulary) with a vector quantization algorithm
3. Constructing a bag of keypoints, which counts the number of patches assigned to each cluster
4. Applying a multi-class classifier, treating the bag of keypoints as the feature vector, and thus determining which category or categories to assign to the image

These steps are designed to maximize classification accuracy while minimizing computational effort. Thus, the descriptors extracted in the first step should be invariant to variations that are irrelevant to the categorization task (image transformations, lighting variations, and occlusions) but rich enough to carry enough information to be discriminative at the category level. The vocabulary used in the second step should be large enough to distinguish relevant changes in image parts, but not so large as to distinguish irrelevant variations such as noise.

The bag of keypoints approach can be motivated by an analogy to learning methods using the bag-of-words representation for text categorization. In our case, "words" do not necessarily have a repeatable meaning such as "eyes" or "car wheels," nor is there an obvious best choice of vocabulary. Rather, our goal is to use a vocabulary that allows good categorization performance on a given training dataset. Therefore, the steps involved in training the system allow consideration of multiple possible vocabularies.

<br/>
# Results  


우리는 세 가지 실험 결과를 제시합니다. 첫 번째 실험에서는 클러스터 수가 분류 정확도에 미치는 영향을 조사하고 나이브 베이즈 분류기의 성능을 평가합니다. 두 번째 실험에서는 같은 문제에 대해 SVM의 성능을 탐구합니다. 이 두 실험은 자체 개발한 7개 클래스 데이터셋을 사용하여 수행되었습니다. 마지막 실험에서는 [16]에서 사용된 네 개의 클래스 데이터셋에 대한 결과를 설명합니다.

우리의 자체 데이터베이스는 얼굴, 건물, 나무, 자동차, 전화기, 자전거, 책 등 7개의 클래스에서 1776개의 이미지를 포함하고 있습니다. 이 데이터셋은 클래스 수가 많을 뿐만 아니라 포즈가 매우 다양하고 배경 잡음이 많으며 때로는 여러 클래스의 객체가 포함된 이미지를 포함하기 때문에 도전적입니다.

나이브 베이즈를 사용한 실험 결과, 클러스터 수가 증가함에 따라 오류율이 약간 감소하였으며, k=1000이 정확도와 속도 사이에서 좋은 균형을 제공한다고 결론지었습니다. 반면, SVM을 사용한 실험에서는 나이브 베이즈보다 더 낮은 오류율을 보였으며, 특히 얼굴 클래스에서 매우 낮은 오류율을 기록했습니다.

SVM의 결과는 나이브 베이즈와 비교했을 때 전반적으로 우수했으며, 우리는 SVM을 사용하여 4개의 클래스 데이터셋에서 좋은 성능을 확인했습니다. 이러한 결과는 백그라운드 잡음에 강하고 기하학적 정보를 활용하지 않고도 좋은 분류 정확도를 나타낸다는 것을 보여줍니다.


We present results from three experiments. In the first experiment, we explore the impact of the number of clusters on classifier accuracy and evaluate the performance of the Naïve Bayes classifier. In the second experiment, we explore the performance of the SVM on the same problem. These first two experiments were conducted using an in-house seven-class dataset. In the last experiment, we describe results on the four-class dataset employed in [16].

Our in-house database contains 1776 images in seven classes: faces, buildings, trees, cars, phones, bikes, and books. This dataset is challenging not only because of the large number of classes but also due to the highly variable poses and significant amounts of background clutter, sometimes containing objects from multiple classes.

In the experiment using Naïve Bayes, we found that the error rate slightly decreased as the number of clusters increased, with k=1000 providing a good trade-off between accuracy and speed. In contrast, the experiment using SVM showed a lower error rate compared to Naïve Bayes, with a very low error rate particularly in the faces class.

The results with SVM were generally superior to those with Naïve Bayes, and we confirmed good performance using SVM on the four-class dataset. These results demonstrate that the method is robust to background clutter and produces good categorization accuracy even without exploiting geometric information.



<br/>
# 예시  


우리는 몇 가지 구체적인 예시를 통해 우리의 방법론이 어떻게 적용되는지 보여줍니다.

1. **여러 객체가 포함된 이미지**: 하나의 이미지에 여러 객체가 포함되어 있을 때도, 우리의 방법은 각 객체를 올바르게 분류할 수 있습니다. 예를 들어, 얼굴, 자전거, 건물이 포함된 이미지를 분류할 때 각각의 객체를 별도로 인식하여 분류할 수 있습니다.

2. **배경 잡음이 많은 이미지**: 배경 잡음이 많아도, 우리의 방법은 주요 객체를 정확하게 분류할 수 있습니다. 예를 들어, 대부분의 관심 지점이 배경에 있는 경우에도, 얼굴이나 자동차와 같은 주요 객체를 올바르게 분류할 수 있습니다.

3. **부분적으로 가려진 객체**: 객체가 부분적으로 가려져 있거나 일부만 보이는 경우에도 우리의 방법은 객체를 정확하게 분류할 수 있습니다. 예를 들어, 얼굴의 일부만 보이거나 자동차의 일부만 보이는 이미지를 분류할 때도 높은 정확도를 유지할 수 있습니다.

4. **여러 클래스의 객체가 포함된 이미지**: 하나의 이미지에 여러 클래스의 객체가 포함된 경우에도, 우리의 방법은 주요 객체를 올바르게 분류할 수 있습니다. 예를 들어, 전화기, 책, 자동차가 포함된 이미지를 분류할 때, 각 클래스의 객체를 올바르게 인식하여 분류할 수 있습니다.


We illustrate how our methodology is applied through several specific examples.

1. **Images with Multiple Objects**: Our method can correctly classify each object in an image containing multiple objects. For example, in an image containing faces, bicycles, and buildings, our method can separately recognize and classify each object.

2. **Images with Significant Background Clutter**: Even with significant background clutter, our method can accurately classify the main objects. For instance, in cases where most of the detected interest points are on the background, our method can still correctly classify main objects like faces or cars.

3. **Partially Occluded Objects**: Our method can accurately classify objects even when they are partially occluded or only partially visible. For example, when only part of a face or a car is visible in an image, our method maintains high accuracy in classification.

4. **Images Containing Objects from Multiple Classes**: Our method can correctly classify the main objects in images containing objects from multiple classes. For instance, in an image containing phones, books, and cars, our method can recognize and classify objects from each class correctly.

<br/>  
# 요약 


이 연구는 이미지 패치의 아핀 불변 서술자를 벡터 양자화하는 방식으로 자연 이미지의 객체를 분류하는 새로운 방법을 제안합니다. 나이브 베이즈와 SVM 두 가지 분류기를 비교하며, 이 방법이 간단하고 계산 효율적이며 본질적으로 불변임을 보여줍니다. 실험 결과, 이 방법은 배경 잡음에 강하며 기하학적 정보를 활용하지 않고도 높은 분류 정확도를 달성합니다. 7개의 시각적 범주를 동시에 분류한 결과가 제시되며, SVM이 나이브 베이즈보다 우수한 성능을 보였습니다. 이 방법은 다양한 객체 유형에 대해 확장 가능하고 새로운 객체 유형에도 적용될 수 있습니다.


This study proposes a novel method for categorizing objects in natural images by vector quantizing affine invariant descriptors of image patches. It compares two classifiers, Naïve Bayes and SVM, demonstrating that the method is simple, computationally efficient, and intrinsically invariant. Experimental results show that the method is robust to background clutter and achieves high classification accuracy without exploiting geometric information. Results for simultaneously classifying seven visual categories are presented, with SVM outperforming Naïve Bayes. This method is extendable to various object types and can be applied to new object types.

# 기타  

텍스트 분류에서 사용되는 단어의 백(bag of words) 접근법을 시각적 데이터에 적용한 키포인트의 백(bag of keypoints)이라는 방법을 사용한 점이 좀 독창적이라고 느꼈다.

I found it a good idea to use the bag of keypoints method, which applies the bag of words approach used in text categorization to visual data. 

지금은 과거보다 적게 쓰이면서 논문에서 SVM과 naive bayesian을 보기 힘들었었는데 이 논문으로 오랜만에 봐서 조금 반가웠다. 

Nowadays, SVM and Naive Bayes are used less frequently, and it has been a while since I last saw them in a paper, so it was somewhat nostalgic to see them used in this study.

사용하는 메서드는 다르지만 태스크 이미지 오브젝트 클래시피케이션으로 유사하다는 생각도 들었다. 

Although the methods used are different, I thought that the task is similar to image object classification.


[refine?]  
I found that it is a good idea to use the bag of keypoints method, which applies the bag of words approach used in text categorization to visual data. Nowadays, SVM and Naive Bayes are used less frequently, and it has been a while since I last saw them in a paper, so it was somewhat refreshing to see them used in this study. Although the methods used are different, I thought that the task is similar to image object classification.



<br/>
# refer format:     
Csurka, Gabriella, Dance, Christopher R., Fan, Lixin, Willamowski, Jutta, & Bray, Cédric. (2004). Visual Categorization with Bags of Keypoints. In Workshop on Statistical Learning in Computer Vision, ECCV, Vol. 1, No. 1-22.
  
@inproceedings{csurka2004visual,
  title={Visual Categorization with Bags of Keypoints},
  author={Csurka, Gabriella and Dance, Christopher R. and Fan, Lixin and Willamowski, Jutta and Bray, C{\'e}dric},
  booktitle={Workshop on Statistical Learning in Computer Vision, ECCV},
  volume={1},
  number={1-22},
  year={2004}
}

