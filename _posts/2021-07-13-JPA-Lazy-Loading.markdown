---
layout: post
title: JPA Lazy Loading
date: '2021-07-13 16:45:10 +0900'
categories: study
published: true
---
LAZY Loading, 지연로딩에 대해서는 꼭 한번 서술해보고 싶었습니다.

지연로딩을 설명하려면 프록시를 알아야하는데,
프록시는 최초 호출시 실제 클래스 상속받아 생성하는 것입니다.

지연로딩은 예를 들어 student의 reference인 school을 로딩한다고 할 때
student를 먼저 로딩하고 프록시의 school을 로딩하는 것입니다.

