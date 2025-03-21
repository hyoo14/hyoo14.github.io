---
layout: post
title:  "[2025]Distribution-Level Feature Distancing for Machine Unlearning"  
date:   2025-03-21 19:16:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 





짧은 요약(Abstract) :    




딥러닝의 활용이 증가함에 따라 개인 정보 보호에 대한 요구도 높아지고 있습니다. 특히, “잊혀질 권리”에 따라 특정 데이터(예: 얼굴 이미지)를 학습에 사용한 모델에서 제거하는 것이 중요해졌습니다. 하지만 기존의 머신 언러닝 기법은 데이터를 잊게 하는 과정에서 모델의 성능이 전반적으로 저하되는 “상관관계 붕괴(correlation collapse)” 문제가 발생합니다. 이 논문은 이러한 문제를 해결하기 위해 **Distribution-Level Feature Distancing (DLFD)**라는 새로운 방법을 제안합니다. DLFD는 ‘잊어야 하는 데이터’와 ‘유지할 데이터’의 특성 분포를 멀어지게 하여 효율적으로 언러닝을 수행하면서도, 원래의 분류 성능을 보존합니다. 얼굴 인식 데이터셋을 이용한 실험을 통해 DLFD가 기존 최신 기법들보다 더 나은 언러닝 성능과 모델 유지 능력을 보여줍니다.

⸻


With growing concerns over privacy in deep learning, the “right to be forgotten” has become increasingly important—requiring AI systems to remove specific data (e.g., personal face images) upon request. Existing machine unlearning methods often cause correlation collapse, where meaningful connections between features and labels are disrupted, reducing model performance. To address this, the authors propose a novel method called Distribution-Level Feature Distancing (DLFD). DLFD alters the feature distribution of retained data to differ from that of the data to be forgotten, ensuring the model forgets specific instances without degrading its overall performance. Experiments on facial recognition datasets show that DLFD significantly outperforms previous state-of-the-art unlearning methods in both forgetting and utility preservation.



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








       
 
<br/>
# Results  



























<br/>
# 예제  






<br/>  
# 요약   





<br/>  
# 기타  






<br/>
# refer format:     



@inproceedings{choi2025distribution,
  title={Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting},
  author={Choi, Dasol and Na, Dongbin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025},
  organization={Association for the Advancement of Artificial Intelligence}
}


Choi, Dasol, and Dongbin Na. 2025. “Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting.” Proceedings of the AAAI Conference on Artificial Intelligence. Association for the Advancement of Artificial Intelligence.





