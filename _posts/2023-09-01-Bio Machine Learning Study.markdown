---
layout: post
title:  "Bio Machine Learning Study"
date:   2023-09-01 14:41:24 -0400
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :   
* Since May 2023    
* 1. Part 1-Basics Concept for Biomedical Machine learning
* 2. Part 3-Deep learning for Biomedical sequence
* 3. Part 4-Deep learning for Biomedical Graph
* target study  

{% endhighlight %}  

<br/>


<br/>

# 1. Part 1-Basics Concept for Biomedical Machine learning
<br/>

[Items link](https://drive.google.com/drive/folders/1n5k0fh6Ts0n50vg1p3qiGbL0J2HQ2Ub4?usp=drive_link)

<br/>
# 2. Part 3-Deep learning for Biomedical sequence
<br/>

[Items link](https://drive.google.com/drive/folders/10tNLY_8ssP-727V_UCXmNg3Q13HjM2wh?usp=drive_link)

<br/>
# 3. Part 4-Deep learning for Biomedical Graph
<br/>

[Items link](https://drive.google.com/drive/folders/1LaM0PPZEhHYNstp78hqtICv8FRjuxY1J?usp=drive_link)

<br/>


# target study  

<br/>

This course will focus on developing the computational, algorithmic, and database navigational skills required to analyze genomic data that have become available with the development of high throughput genomic technologies.   


We will also illustrate statistical signal processing concepts such as dynamic programming, hidden markov models, information theoretic measures, and assessing statistical significance.   


The goals will be achieved through lecture and lab exercises that focus on genomic databases, genome annotation via hidden markov models, sequence alignment through dynamic programming, metagenomic analyses, and phylogenetics with maximum likelihood approaches.    

<br/>

# extracted topics  

* genomic DB/annotation + cs algorithm  
* dynamic programming (sequence alignment)    
* hidden markov models  
* information theoretic measures  
* assessing statistical significance  
* metagenomic analyses  
* maximum likelihodd approaches (for phylogenetics)  


<br/>


# topics extending  

* genomic DB/annotation + cs algorithm  


* dynamic programming (sequence alignment)    
** Forward algorithm  
*** like CKY parsing, uses a table to store intermediate values  
*** compute the likelihood of the observation sequence  
*** By summing over all possible hidden state sequences  
*** But doing this efficiently-> By folding all the sequences into a single trellis(격자)  
** Viterbi algorithm  
*** use a similar trellis to the Forward algirhtm to compute joint probability of the observation squence together with the best state sequence  
*** trace back to find maximum prob sequence  


* hidden markov models  
** Markov Assumption: 바로 이전 상태만 현재 상태에 관여됨.(future state depends only on the current state) P(qi|q1,...qi-1) = P(qi|qi-1)  
** Markov Chain: 상태와 전이 확률로 구성됨, Markov Assumption 따름 (a set of states and the transition probabilities, follow Markov Assumtion)  
** Hidden Markov Model: 관측되지 않는 (숨겨진) 상태들의 마르코프 체인 + 그 상태들로부터 관측값을 생성하는 확률 과정을 결합한 모델(a Markov chain of unobserved (hidden) states + probabilistic process that generates observations from those states)  
*** 상태에서의 관측값 발생 확률과 상태 전이 확률을 모두 포함, 숨겨진 상태들로부터 어떤 관측값 시퀀스가 생성되었는지를 추론  
**** probabilities of observations given a state and the transition probabilities between states, used to infer the sequence of hidden states   


* information theoretic measures  
* assessing statistical significance  
* metagenomic analyses  


* maximum likelihodd approaches (for phylogenetics)  
** likelihood : measure of how well a statistical model explains a set of observations(모델이 관측을 얼마나 잘 설명하나 측정)    


<br/>
