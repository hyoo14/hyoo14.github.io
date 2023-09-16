---
layout: post
title:  "Math and Machine Learning Study"
date:   2023-09-01 14:41:24 -0400
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :   
* target study  

{% endhighlight %}  

<br/>


# target study  

<br/>

This course studies modern statistical machine learning with emphasis on Bayesian modeling and inference.

   
Covered topics include fundamentals of probabilities and decision theory, regression, classification, graphical models, mixture models, clustering, expectation maximization, hidden Markov models, Kalman filtering, and linear dynamical systems.  



<br/>

# extracted topics  

* statistical machine learning -> Bayesian modeling and inference  
* fundamental probabilities and decision theory  
* regression  
* classification  
* graphical madels  
* mixture models  
* clustering  
* expectation maximization  
* hidden markove models  
* kalman filtering  
* linear dynamical systems  


<br/>

# topics extending  

* statistical machine learning -> Bayesian modeling and inference  
* fundamental probabilities and decision theory  
* regression  

* classification  
** decision trees and rule learning  
** naive bayes and bayesian networks  
*** Bayesian Statistics  
**** object: infer posterior probabilities(based on prior probabilities & observed data)   
**** use prior probabilities & observed data to update the uncertainty    
**** posterior probabilities is computed using Bayes' Theorem   
**** P( theta | X ) = P( X | theta ) * P( theta ) / P( X )  
**** Posterior Probability = likelihood * prior probability / total probability of data X  


*** Bayesian Decision Theory  
**** provides a framework for making optimal decision under uncertainity  
**** given data + prior knowledge -- infer --> posterior probabilities ----> making decisions(that minimize a loss function based on posterior probabilities)  
**** (evaluate all expected loss for all possible decisions & choose the one minimize the loss)  
**** alpha(x) = argmin alpha_i Sigma_j L(alpha_i , weight_j) P(W_j | x)  
**** optimal decision given data x(observed) = argmin Sigma loss function(decision, state) Posterior probability(xtate w_j | x)  


*** Naive Bayesian Method(Classificaion)  
**** classification algorithm based on Bayesian theory  
**** assume all features are independent(why called "naivd")  
**** calculating the probability of each class(with given data and priority prob)  


** logistic regression / maximum entropy  
** neural networks(deep)  
** svm  
** nearest-neighbor / instance-based  

* graphical madels  
* mixture models  
* clustering  
* expectation maximization  

* hidden markove models  
** markov chain = first-order observable markov model  
*** current state only depends on previous state  
*** markov assumption: P(qi|q1 ... qi-1) = P(qi|qi-1)  

* kalman filtering  
* linear dynamical systems  

<br/>