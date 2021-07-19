---
layout: post
title: JPA how to make DTO
date: '2021-07-09 17:45:10 +0900'
categories: study
published: true
---
아무래도 최근에 하는 일이 백엔드이다 보니 스프링 JPA관련 내용으로 시작을 하게 되었습니다.
특히 api에서는 엔티티를 직접 다루는 것을 주의해야하니 DTO로 래핑하는 것에 대해 작성해보겠습니다.

사실 별건 아니고 아래처럼 별거 아니고 그냥 class 하나 더 정의해서 사용하면 됩니다.
엔티티 대신 api에서 사용할 클래스를 작성하는 것입니다.:

{% highlight ruby %}
class DTO {
 private Long studentId
 private String studentName;
}
{% endhighlight %}

엔티티를 노출시키는 것은 보안에 있어서 바람직하지 않고 확장성 측면에서도 api 스펙이 완전히 바뀌어서 좋지 않으니 대신 dto를 사용하는 것이지요.

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
