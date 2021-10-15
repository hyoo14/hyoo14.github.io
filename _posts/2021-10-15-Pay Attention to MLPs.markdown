---
layout: post
title:  "Pay Attention to MLPs"
date:   2021-10-15 16:10:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

gMLP(gated Multi Layer Perceptron)으로 transformer 만큼의 성능을 확인 (Vision과 NLP 모두에서)

{% endhighlight %}


-transformer는 spatial interaction 지나치게 많음.

-대안으로 gMLP(채널 projection & 공간 projection - static parameterization으로) 제안

-self-attention 대신 SGU(Spatial Gating Unit) 사용

