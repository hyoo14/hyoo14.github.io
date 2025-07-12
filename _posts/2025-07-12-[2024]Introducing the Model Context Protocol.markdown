---
layout: post
title:  "[2024]Introducing the Model Context Protocol"  
date:   2025-07-12 07:00:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


MCP는 Claude 같은 LLM에게 다양한 **문맥(context)**을 구조화하여 전달하기 위한 표준화된 인터페이스 / 프로토콜입니다.

요약하면, Claude가 더 잘 이해하고 반응하도록 맥락을 구조화해서 전달하는 형식적 규약입니다.





짧은 요약(Abstract) :    



An abstract illustration of critical context connecting to a central hub
Today, we're open-sourcing the Model Context Protocol (MCP), a new standard for connecting AI assistants to the systems where data lives, including content repositories, business tools, and development environments. Its aim is to help frontier models produce better, more relevant responses.

As AI assistants gain mainstream adoption, the industry has invested heavily in model capabilities, achieving rapid advances in reasoning and quality. Yet even the most sophisticated models are constrained by their isolation from data—trapped behind information silos and legacy systems. Every new data source requires its own custom implementation, making truly connected systems difficult to scale.

MCP addresses this challenge. It provides a universal, open standard for connecting AI systems with data sources, replacing fragmented integrations with a single protocol. The result is a simpler, more reliable way to give AI systems access to the data they need.

Model Context Protocol
The Model Context Protocol is an open standard that enables developers to build secure, two-way connections between their data sources and AI-powered tools. The architecture is straightforward: developers can either expose their data through MCP servers or build AI applications (MCP clients) that connect to these servers.

Today, we're introducing three major components of the Model Context Protocol for developers:

The Model Context Protocol specification and SDKs
Local MCP server support in the Claude Desktop apps
An open-source repository of MCP servers
Claude 3.5 Sonnet is adept at quickly building MCP server implementations, making it easy for organizations and individuals to rapidly connect their most important datasets with a range of AI-powered tools. To help developers start exploring, we’re sharing pre-built MCP servers for popular enterprise systems like Google Drive, Slack, GitHub, Git, Postgres, and Puppeteer.

Early adopters like Block and Apollo have integrated MCP into their systems, while development tools companies including Zed, Replit, Codeium, and Sourcegraph are working with MCP to enhance their platforms—enabling AI agents to better retrieve relevant information to further understand the context around a coding task and produce more nuanced and functional code with fewer attempts.

"At Block, open source is more than a development model—it’s the foundation of our work and a commitment to creating technology that drives meaningful change and serves as a public good for all,” said Dhanji R. Prasanna, Chief Technology Officer at Block. “Open technologies like the Model Context Protocol are the bridges that connect AI to real-world applications, ensuring innovation is accessible, transparent, and rooted in collaboration. We are excited to partner on a protocol and use it to build agentic systems, which remove the burden of the mechanical so people can focus on the creative.”

Instead of maintaining separate connectors for each data source, developers can now build against a standard protocol. As the ecosystem matures, AI systems will maintain context as they move between different tools and datasets, replacing today's fragmented integrations with a more sustainable architecture.

Getting started
Developers can start building and testing MCP connectors today. All Claude.ai plans support connecting MCP servers to the Claude Desktop app.

Claude for Work customers can begin testing MCP servers locally, connecting Claude to internal systems and datasets. We'll soon provide developer toolkits for deploying remote production MCP servers that can serve your entire Claude for Work organization.

To start building:

Install pre-built MCP servers through the Claude Desktop app
Follow our quickstart guide to build your first MCP server
Contribute to our open-source repositories of connectors and implementations
An open community
We’re committed to building MCP as a collaborative, open-source project and ecosystem, and we’re eager to hear your feedback. Whether you’re an AI tool developer, an enterprise looking to leverage existing data, or an early adopter exploring the frontier, we invite you to build the future of context-aware AI together.






중앙 허브로 연결된 중요한 컨텍스트들을 추상적으로 묘사한 일러스트

오늘 우리는 **Model Context Protocol (MCP)**을 오픈소스로 공개합니다. MCP는 콘텐츠 저장소, 업무 도구, 개발 환경 등 AI 어시스턴트가 데이터를 보관하는 시스템들과 연결될 수 있도록 하는 새로운 표준입니다. 그 목적은 최첨단 모델들이 더 나은, 더 관련성 높은 응답을 생성할 수 있도록 돕는 것입니다.

AI 어시스턴트가 대중적으로 채택됨에 따라 업계는 모델 성능 개선에 막대한 투자를 하며 추론과 품질 면에서 빠른 진보를 이루고 있습니다. 하지만 아무리 정교한 모델이라도 데이터와 고립되어 있으면 한계가 있습니다. AI는 정보 사일로와 구식 시스템에 갇혀 있으며, 각각의 새로운 데이터 소스마다 별도의 커스텀 구현이 필요하여 진정한 연결형 시스템을 확장하기 어렵습니다.

MCP는 이러한 문제를 해결합니다. 이 프로토콜은 AI 시스템과 데이터 소스를 연결하는 보편적인 오픈 표준을 제공하며, 파편화된 통합 방식을 단일한 프로토콜로 대체합니다. 그 결과, AI가 필요한 데이터에 더 간단하고 신뢰할 수 있게 접근할 수 있는 구조가 됩니다.

Model Context Protocol이란?
Model Context Protocol은 개발자가 AI 기반 도구와 데이터 소스 간에 안전한 양방향 연결을 구축할 수 있도록 하는 오픈 표준입니다. 아키텍처는 간단합니다. 개발자는 자신의 데이터를 MCP 서버를 통해 노출시키거나, 해당 서버에 연결되는 AI 애플리케이션(MCP 클라이언트)을 구축할 수 있습니다.

이번에 공개되는 MCP의 핵심 구성 요소 3가지는 다음과 같습니다:

Model Context Protocol 사양 및 SDK

Claude 데스크탑 앱 내 로컬 MCP 서버 지원

MCP 서버의 오픈소스 저장소

Claude 3.5 Sonnet은 MCP 서버 구현을 신속하게 구축할 수 있어, 조직이나 개인이 중요한 데이터셋을 다양한 AI 기반 도구와 쉽게 연결할 수 있도록 지원합니다. 개발자들이 바로 시작할 수 있도록, Google Drive, Slack, GitHub, Git, Postgres, Puppeteer와 같은 인기 있는 기업 시스템용 사전 구축된 MCP 서버를 공유합니다.

초기 도입 사례
Block, Apollo와 같은 초기 도입 기업들이 이미 MCP를 시스템에 통합하였으며, Zed, Replit, Codeium, Sourcegraph와 같은 개발 도구 업체들도 MCP를 기반으로 플랫폼을 개선 중입니다. 이를 통해 AI 에이전트가 코딩 작업의 문맥을 더 잘 이해하고, 시도 횟수를 줄이면서도 더 정교하고 기능적인 코드를 생성할 수 있습니다.

“Block에서는 오픈소스를 단순한 개발 모델이 아닌, 우리의 기술 철학이자 공공의 이익을 위한 핵심 가치로 여깁니다. MCP와 같은 오픈 기술은 AI를 실제 응용에 연결하는 다리이며, 혁신을 더 투명하고 협업 중심으로 만듭니다. 우리는 이 프로토콜을 통해 반복 작업의 부담을 줄이고, 사람들이 창의적인 일에 더 집중할 수 있는 에이전트 시스템을 만들고자 합니다.”
— Dhanji R. Prasanna, Block 최고기술책임자(CTO)

지금까지는 각 데이터 소스에 대해 개별 커넥터를 유지해야 했지만, 이제는 개발자들이 표준 프로토콜 기반으로 개발할 수 있습니다. 생태계가 성숙해질수록, AI 시스템은 다양한 도구와 데이터셋 사이에서 문맥을 유지하며 이동할 수 있게 되며, 이는 오늘날의 파편화된 통합 구조를 더 지속가능한 아키텍처로 대체할 것입니다.

시작하기
개발자는 지금 바로 MCP 커넥터를 구축하고 테스트할 수 있습니다. 모든 Claude.ai 요금제에서는 Claude 데스크탑 앱을 통해 MCP 서버에 연결할 수 있습니다.

Claude for Work 고객은 로컬 MCP 서버를 테스트하여 Claude를 내부 시스템 및 데이터셋과 연결해볼 수 있습니다. 곧 원격 프로덕션 환경에 MCP 서버를 배포할 수 있도록 도와주는 개발자 도구도 제공할 예정입니다.

시작 방법
Claude 데스크탑 앱에서 사전 구축된 MCP 서버 설치

빠른 시작 가이드를 따라 첫 MCP 서버 구축

커넥터 및 구현 오픈소스 저장소에 기여

오픈 커뮤니티
우리는 MCP를 협업 중심의 오픈소스 프로젝트 및 생태계로 발전시키기 위해 최선을 다하고 있습니다. AI 도구 개발자, 기존 데이터를 활용하려는 기업, 혹은 첨단 기술을 탐색하는 얼리어답터 등 누구나 환영입니다. 함께 문맥 인지 AI의 미래를 만들어가요.





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


주요 특징
📦 구조화된 입력 포맷	사용자의 입력, 시스템 지시문, 이전 대화 내용 등을 명시적으로 구분해 전달
🧠 컨텍스트 이해력 향상	단순한 프롬프트보다 더 정교하게 문맥을 지정해 LLM이 더 적절히 반응하도록 도움
🔐 안전성 및 제어성 ↑	보안, 개인정보 보호, 대화 흐름 제어 등을 포함하기 쉬움
📚 여러 유형의 컨텍스트 지원	예: 사용자 역할, 작업 목적, 대화 기록, 외부 문서 링크 등


{
  "version": "1.0",
  "context": {
    "system": "You are a helpful assistant that provides accurate legal information.",
    "user": "What are my rights if I'm fired without notice?",
    "metadata": {
      "user_id": "abc123",
      "timestamp": "2025-07-12T08:00:00Z"
    },
    "documents": [
      {
        "title": "Employment Law Summary",
        "content": "...",
        "source": "https://example.com/law"
      }
    ]
  }
}




🧩 MCP의 핵심 구성 요소

system	Claude의 역할이나 태도를 지시함 (e.g., 친절한 비서, 과학 논문 리뷰어 등)
user	사용자의 질문 또는 요청
metadata	사용자 ID, 세션 시간, 토픽 태그 등 부가 정보
documents	LLM이 참고할 외부 문서 또는 링크
history	이전 대화 기록을 명시적으로 포함시킬 수 있음


🔧 왜 중요한가?
멀티모달/멀티컨텍스트 작업에 유리
(예: 대화형 검색, 문서 요약, 지식 기반 QA)

RAG(Retrieval-Augmented Generation), tool use, agent orchestration 같은 고급 작업에 사용

Claude 같은 모델이 여러 정보를 한꺼번에 받아야 할 때 명확한 규칙이 필요
→ 그게 바로 MCP   



<br/>  
# 요약   



용어	Model Context Protocol (MCP)
목적	LLM에 다양한 문맥 정보를 구조화하여 제공
구성	system, user, metadata, documents, history 등
장점	문맥 이해 향상, 반응의 정확도 및 제어성 향상
사용처	Claude API, 대화형 에이전트, 툴 사용 통합 등


<br/>  
# 기타  


<br/>
# refer format:     


@misc{anthropic2024mcp,
  author       = {Anthropic},
  title        = {Introducing the Model Context Protocol},
  year         = {2024},
  month        = {November},
  url          = {https://www.anthropic.com/news/model-context-protocol},
  note         = {Accessed: 2025-07-12},
  howpublished = {\url{https://www.anthropic.com/news/model-context-protocol}},
  institution  = {Anthropic PBC}
}



Anthropic. “Introducing the Model Context Protocol.” Anthropic News, November 25, 2024. https://www.anthropic.com/news/model-context-protocol.   




