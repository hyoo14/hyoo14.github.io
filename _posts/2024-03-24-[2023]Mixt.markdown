---
layout: post
title:  "[2023]Mixtral of Experts"  
date:   2024-03-19 09:03:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    
* 
Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()  

<br/>

# 단어정리  
* 

<br/>
# 1. INTRODUCTION  
* 



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

