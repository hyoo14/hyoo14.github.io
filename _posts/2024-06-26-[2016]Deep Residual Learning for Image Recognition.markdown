---
layout: post
title:  "[2016]Deep Residual Learning for Image Recognition"  
date:   2024-06-26 23:44:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    



깊은 신경망은 훈련하기 어렵습니다. 우리는 이전에 사용된 네트워크보다 훨씬 깊은 네트워크를 쉽게 훈련할 수 있도록 잔차 학습 프레임워크를 제시합니다. 우리는 층 입력에 대한 참조로 잔차 함수를 학습하는 것으로 층을 명시적으로 재구성하여 참조되지 않은 함수를 학습하는 대신 이를 수행합니다. 우리는 이러한 잔차 네트워크가 최적화하기 쉽고, 깊이가 상당히 증가함에 따라 정확도가 향상될 수 있음을 보여주는 포괄적인 실험적 증거를 제공합니다. 우리는 ImageNet 데이터셋에서 깊이가 최대 152층인 잔차 네트워크를 평가하며, 이는 VGG 네트워크보다 8배 더 깊지만 여전히 복잡도가 낮습니다. 이러한 잔차 네트워크의 앙상블은 ImageNet 테스트 세트에서 3.57%의 오류를 기록했습니다. 이 결과는 ILSVRC 2015 분류 과제에서 1위를 차지했습니다. 우리는 또한 CIFAR-10에서 100층 및 1000층에 대한 분석을 제공합니다.


Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8× deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.




* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1D8b7bZTHO-0tiOcoq-p4KODy88nkQKF3?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    



1. **잔차 학습**:
   잔차 학습은 몇 개의 층을 쌓아 원하는 기본 매핑 \(H(x)\)를 학습하는 대신, 잔차 함수를 학습합니다. 이는 \(H(x)\)를 직접 학습하는 것보다 잔차 함수 \(F(x) := H(x) - x\)를 학습하는 것이 더 쉽다는 가설을 기반으로 합니다. 이를 통해 모델이 \(F(x) + x\) 형태의 매핑을 학습할 수 있게 됩니다.

2. **아이덴티티 매핑을 통한 지름길 연결**:
   지름길 연결(shortcut connections)은 층을 건너뛰는 연결입니다. 이러한 연결은 아이덴티티 매핑을 수행하고, 쌓인 층의 출력에 더해집니다. 이 방법은 추가적인 매개변수나 계산 복잡도를 추가하지 않으며, 잔차 학습을 통해 최적화 문제를 해결하는 데 도움이 됩니다.

3. **네트워크 아키텍처**:
   이미지넷을 위한 두 가지 모델을 제시합니다. 첫 번째는 34층의 "플레인" 네트워크로, 주로 VGG 네트워크에서 영감을 받았습니다. 두 번째는 동일한 네트워크에 지름길 연결을 추가하여 잔차 네트워크로 만든 것입니다. 잔차 네트워크는 아이덴티티 매핑을 사용하며, 차원이 증가할 때는 프로젝션 지름길을 사용합니다.

4. **실험 및 결과**:
   다양한 깊이의 잔차 네트워크를 ImageNet과 CIFAR-10 데이터셋에서 평가했습니다. 잔차 네트워크는 깊이가 증가함에 따라 성능이 향상되었으며, 특히 152층 잔차 네트워크는 ImageNet 테스트 세트에서 3.57%의 오류율을 기록했습니다. 이러한 결과는 잔차 학습이 매우 깊은 네트워크의 최적화를 크게 향상시킨다는 것을 보여줍니다.



1. **Residual Learning**:
   Instead of learning a desired underlying mapping \(H(x)\) by stacking several layers, residual learning proposes to learn the residual function \(F(x) := H(x) - x\). The hypothesis is that it is easier to learn the residual mapping than to learn the original unreferenced mapping. Thus, the model learns the mapping in the form of \(F(x) + x\).

2. **Identity Mapping by Shortcuts**:
   Shortcut connections are those that skip one or more layers. These connections perform identity mapping and are added to the output of the stacked layers. This method adds neither extra parameters nor computational complexity and helps address the optimization problem through residual learning.

3. **Network Architectures**:
   Two models for ImageNet are described. The first is a 34-layer "plain" network inspired by VGG nets. The second model is a residual network created by adding shortcut connections to the same network. The residual network uses identity mappings and projection shortcuts when dimensions increase.

4. **Experiments and Results**:
   Various depths of residual networks were evaluated on the ImageNet and CIFAR-10 datasets. The residual networks improved in performance as depth increased, with the 152-layer residual network achieving a 3.57% error rate on the ImageNet test set. These results demonstrate that residual learning significantly improves the optimization of very deep networks.



<br/>
# Results  



1. **ImageNet 분류 성능**:
   ImageNet 데이터셋에서 18층, 34층, 50층, 101층, 152층 잔차 네트워크를 평가했습니다. 34층 잔차 네트워크는 동일한 깊이의 플레인 네트워크보다 훨씬 낮은 오류율을 기록했습니다. 152층 잔차 네트워크는 ImageNet 테스트 세트에서 3.57%의 오류율을 기록하며, ILSVRC 2015 분류 과제에서 1위를 차지했습니다. 이는 잔차 학습이 깊이가 증가함에 따라 성능을 향상시키는 데 효과적임을 보여줍니다.

2. **CIFAR-10 성능**:
   CIFAR-10 데이터셋에서 20층, 32층, 44층, 56층, 110층 잔차 네트워크를 평가했습니다. 잔차 네트워크는 플레인 네트워크보다 더 깊이 쌓아도 최적화 문제가 발생하지 않았으며, 깊이가 증가할수록 성능이 향상되었습니다. 110층 잔차 네트워크는 CIFAR-10 테스트 세트에서 6.43%의 오류율을 기록했습니다.

3. **객체 검출 성능**:
   PASCAL VOC와 MS COCO 데이터셋에서 Faster R-CNN을 사용하여 객체 검출 성능을 평가했습니다. VGG-16을 사용한 모델과 비교하여 ResNet-101을 사용한 모델이 PASCAL VOC 2007/2012와 COCO 데이터셋 모두에서 더 높은 mAP(Mean Average Precision)를 기록했습니다. 특히 COCO 데이터셋에서 mAP@[.5, .95]에서 6.0% 향상된 성능을 보였습니다.



1. **ImageNet Classification Performance**:
   The 18-layer, 34-layer, 50-layer, 101-layer, and 152-layer residual networks were evaluated on the ImageNet dataset. The 34-layer residual network recorded a much lower error rate compared to its plain counterpart of the same depth. The 152-layer residual network achieved a 3.57% error rate on the ImageNet test set, winning the 1st place in the ILSVRC 2015 classification task. This demonstrates the effectiveness of residual learning in improving performance as depth increases.

2. **CIFAR-10 Performance**:
   On the CIFAR-10 dataset, 20-layer, 32-layer, 44-layer, 56-layer, and 110-layer residual networks were evaluated. Residual networks did not encounter optimization issues even with greater depth and showed improved performance with increasing depth. The 110-layer residual network achieved a 6.43% error rate on the CIFAR-10 test set.

3. **Object Detection Performance**:
   Object detection performance was evaluated using Faster R-CNN on the PASCAL VOC and MS COCO datasets. The ResNet-101 model outperformed the VGG-16 model in terms of mAP (Mean Average Precision) on both the PASCAL VOC 2007/2012 and COCO datasets. Notably, on the COCO dataset, it showed a 6.0% improvement in mAP@[.5, .95].



<br/>
# 예시  


1. **잔차 블록 (Residual Block)**:
   그림 2는 잔차 블록의 구조를 보여줍니다. 이 블록은 두 개의 연속된 레이어 사이에 직접적인 연결을 추가하여 아이덴티티 매핑을 수행합니다. 이러한 구조는 각 블록이 입력 \(x\)를 받으면 이를 \(F(x)\)에 더하는 방식으로 학습을 진행합니다. 이로 인해 모델이 더 깊어지더라도 최적화가 쉬워집니다.

2. **34층 잔차 네트워크와 플레인 네트워크 비교 (Comparison of 34-layer Residual and Plain Networks)**:
   그림 3은 34층 플레인 네트워크와 34층 잔차 네트워크의 구조를 비교합니다. 플레인 네트워크는 단순히 여러 층을 쌓아 놓은 구조인 반면, 잔차 네트워크는 층 사이에 지름길 연결을 추가하여 더 깊은 네트워크에서도 높은 성능을 유지합니다.

3. **CIFAR-10 데이터셋에서의 성능 (Performance on CIFAR-10 Dataset)**:
   그림 6은 CIFAR-10 데이터셋에서의 학습 오류와 테스트 오류를 보여줍니다. 플레인 네트워크는 깊이가 깊어질수록 학습 오류가 증가하는 반면, 잔차 네트워크는 깊이가 깊어지더라도 학습 오류가 낮게 유지됩니다. 특히, 56층과 110층 잔차 네트워크는 학습과 테스트 오류 모두에서 우수한 성능을 보였습니다.



1. **Residual Block**:
   Figure 2 shows the structure of a residual block. This block performs identity mapping by adding a direct connection between two consecutive layers. When each block receives an input \(x\), it learns by adding this input to \(F(x)\). This structure helps in optimizing the model even as it becomes deeper.

2. **Comparison of 34-layer Residual and Plain Networks**:
   Figure 3 compares the structures of a 34-layer plain network and a 34-layer residual network. The plain network simply stacks several layers, while the residual network adds shortcut connections between layers, maintaining high performance even in deeper networks.

3. **Performance on CIFAR-10 Dataset**:
   Figure 6 shows the training and testing errors on the CIFAR-10 dataset. The plain network exhibits increasing training errors as depth increases, whereas the residual network maintains low training errors despite increased depth. Specifically, the 56-layer and 110-layer residual networks demonstrate superior performance in both training and testing errors.





<br/>  
# 요약 



이 연구는 깊은 신경망의 훈련을 용이하게 하기 위해 잔차 학습 프레임워크를 제안합니다. 잔차 네트워크는 기존의 깊은 네트워크보다 최적화가 쉽고, 더 깊은 네트워크에서 높은 정확도를 달성할 수 있습니다. ImageNet 데이터셋에서 152층 잔차 네트워크는 3.57%의 오류율을 기록하며, 이는 ILSVRC 2015에서 1위를 차지했습니다. CIFAR-10 데이터셋에서도 잔차 네트워크는 플레인 네트워크보다 성능이 뛰어났습니다. 또한, MS COCO 데이터셋에서 객체 검출 성능이 크게 향상되었습니다.


This study proposes a residual learning framework to ease the training of deep neural networks. Residual networks are easier to optimize and achieve higher accuracy in deeper networks compared to traditional deep networks. On the ImageNet dataset, the 152-layer residual network recorded a 3.57% error rate, winning the 1st place in the ILSVRC 2015. Residual networks also outperformed plain networks on the CIFAR-10 dataset. Additionally, object detection performance significantly improved on the MS COCO dataset.

# 기타  




<br/>
# refer format:     
He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian. "Deep Residual Learning for Image Recognition." Published in: *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Date of Conference: 27-30 June 2016, Date Added to IEEE Xplore: 12 December 2016, ISBN Information: Electronic ISSN: 1063-6919, DOI: 10.1109/CVPR.2016.90, Publisher: IEEE, Conference Location: Las Vegas, NV, USA.  
  
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={770--778},
  year={2016},
  organization={IEEE},
  doi={10.1109/CVPR.2016.90},
  issn={1063-6919},
  location={Las Vegas, NV, USA},
  month={June},
  dateadded={12 December 2016}
}
