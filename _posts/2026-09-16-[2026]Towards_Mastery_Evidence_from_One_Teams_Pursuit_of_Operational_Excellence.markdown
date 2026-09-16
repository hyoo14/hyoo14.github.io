---
layout: post
title:  "[2026]Towards Mastery: Evidence from One Team's Pursuit of Operational Excellence"
date:   2026-09-16 17:55:52 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: OpsAcademy(data science) 직원들에게 훈련 후 생산성 유의하게 향상  


짧은 요약(Abstract) :



이 논문은 제약·의료 연구개발 조직의 Data Science Platform Operations 팀에서 운영한 교차교육 프로그램인 “Ops Academy”의 효과를 분석한 연구입니다.

- 목적: 특정 직원에게 지식과 업무가 집중되는 ‘지식 사일로’를 줄이고, 지원 인력 전체가 다양한 데이터 사이언스 시스템과 도구를 다룰 수 있도록 교육하는 것이었습니다.
- 방법: 2021~2023년에 처리된 운영지원 티켓 4,070건을 검토하고, 엄격한 기준에 따라 420건을 표본으로 선정했습니다. 이후 SAS를 이용해 교육 시행 전후의 티켓 해결 시간을 비교했습니다.
- 주요 결과: 2023년에 Ops Academy를 실시한 후 생산성이 통계적으로 유의하게 향상되었습니다. 평균 티켓 해결 시간은  
  - 2022년: 5.99일  
  - 2023년: 2.87일  
  로 감소해 50% 이상 단축되었습니다. 이 차이는 통계적으로 유의했으며(t=5.68, p<0.0001), 교육의 실질적인 효과도 중간 수준으로 확인되었습니다(Cohen’s d=0.554).
- 의미와 활용: Ops Academy는 직원들의 업무 역량을 넓히고, 특정 인력에 대한 의존도를 낮추며, 팀의 유연성과 대응력을 높이는 데 도움이 되었습니다. 논문은 조직마다 사용하는 시스템은 다르더라도, 이러한 구조화된 직무교육과 교차훈련 모델을 다른 조직에서도 활용할 수 있다고 제안합니다.

즉, 이 연구의 핵심은 직원들이 여러 시스템을 함께 다룰 수 있도록 정기적으로 교육하면 업무 사일로가 줄어들고, 지원 업무의 처리 속도와 조직 생산성이 크게 향상될 수 있다는 것입니다.

---




This study evaluated the impact of “Ops Academy,” a formal cross-training program implemented for a Data Science Platform Operations team in a pharmaceutical R&D organization.

- Purpose: The program was designed to reduce knowledge silos and help support staff develop skills across a broad range of data science systems, tools, and infrastructure.
- Methods: The researchers reviewed 4,070 operational support tickets from 2021 to 2023 and selected a sample of 420 tickets using strict inclusion criteria and random sampling. Ticket resolution times before and after the program were then analyzed using SAS.
- Key findings: After Ops Academy was introduced in 2023, productivity improved significantly. The average resolution time decreased from 5.99 days in 2022 to 2.87 days in 2023, representing an efficiency improvement of more than 50%. The difference was statistically significant (t=5.68, p<0.0001), with a moderate practical effect size (Cohen’s d=0.554).
- Implications: The program helped broaden employees’ capabilities, reduce dependence on a small number of specialists, and improve the team’s flexibility and responsiveness. The authors suggest that other organizations could adapt this cross-training framework even if their specific technologies and systems differ.

In short, the study shows that structured cross-training can reduce knowledge silos and substantially improve operational productivity and service-response time.


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



이 연구는 머신러닝 모델이나 새로운 소프트웨어 아키텍처를 개발한 연구가 아니라, 데이터 사이언스 플랫폼 운영팀을 대상으로 한 구조화된 현장훈련(SOJT: Structured On-the-Job Training) 프로그램의 효과를 정량적으로 평가한 연구이다. 프로그램은 “Ops Academy”라고 불렸다.

### 1. 연구 개입: Ops Academy

Ops Academy는 운영지원 담당자들이 특정 시스템에만 의존하지 않고 여러 데이터 사이언스 시스템과 도구를 지원할 수 있도록 설계된 교차훈련(cross-training) 프로그램이다.

- 운영지원 직원들이 매주 참여하는 주 1회, 30분 교육
- 데이터 파이프라인, 애플리케이션, 과학 도구, SaaS 시스템 등 다양한 운영 영역을 학습
- 기존의 L1·L2 중심 지원 구조에서 벗어나 여러 담당자가 다양한 업무를 처리하도록 함
- 지식 사일로(silo)와 특정 담당자 의존성을 줄이는 것이 핵심 목표

교육에는 다음과 같은 실무 중심 기법이 포함되었다.

- 동료 검토(peer review)
- 장애 및 사건 대응 시뮬레이션(incident simulation)
- 섀도잉 워크숍(shadowing workshop)
- 콘텐츠 게임화(content gamification)
- 실제 업무를 기반으로 한 구조화된 현장훈련(SOJT)

즉, 이 연구의 “특별한 기법”은 AI 알고리즘이 아니라 실제 운영업무와 연계된 반복적·협력적 교차훈련 방식이다.

---

### 2. 데이터 수집

운영지원 업무에서 발생한 ServiceNow SAM 티켓 데이터를 후향적으로 수집하였다.

- 대상 기간: 2021~2023년 중심
- 전체 티켓 수: 4,070건
- 데이터 항목: 티켓 생성, 처리, 상태 변경, 해결 시점 등
- 핵심 성과지표: 티켓 해결 소요시간(resolution time)

데이터는 SAM 시스템에서 API를 통해 추출한 뒤 Excel로 정리하였다. 이후 데이터 정제와 통계분석에는 SAS Studio, 시각화에는 Tableau를 사용하였다.

---

### 3. 연구대상 및 표본추출

Ops Academy가 시작된 시점은 2023년 1월 1일이며, 프로그램 효과를 비교하기 위해 2022년과 2023년의 데이터를 비교하였다.

비교의 공정성을 높이기 위해 다음 조건을 적용하였다.

- 2022년과 2023년에 모두 근무한 직원만 포함
- 2023년 Ops Academy에 참여한 직원만 포함
- 직원별 업무 참여기간과 지원업무 비중을 고려
- 해결시간이 평균에서 2표준편차 이상 벗어난 티켓은 검토
- 이상치로 판단될 경우 업무기록과 티켓 설명을 확인

최종적으로 비교 가능한 티켓 모집단은 461건이었으며, Cochran 표본크기 공식을 이용해 연도별 210건씩, 총 420건을 무작위 추출하였다.

- 2022년: 210건
- 2023년: 210건
- 목표 신뢰수준: 95%
- 허용오차: 5%

---

### 4. 분석 방법

연구진은 2022년과 2023년의 티켓 해결시간을 비교하기 위해 다음 분석을 수행하였다.

#### 기술통계

각 연도의 다음 값을 계산하였다.

- 평균(mean)
- 표준편차(standard deviation)
- 최솟값과 최댓값
- 95% 신뢰구간(confidence interval)

#### 분포와 데이터 적합성 확인

- Q-Q plot을 이용해 해결시간 데이터의 정규성을 확인
- 상자수염도와 커널 밀도 곡선을 사용해 두 연도의 분포 차이를 시각화
- 데이터에 심각한 왜곡이나 제거가 필요한 이상치가 있는지 검토

#### 가설검정

2022년과 2023년의 평균 해결시간 차이를 확인하기 위해 독립표본 t-검정을 실시하였다.

- 유의수준: α = 0.01
- t값: 5.68
- p값: p < 0.0001
- 효과크기: Cohen’s d = 0.554

이는 2023년의 해결시간이 2022년보다 통계적으로 유의하게 감소했음을 의미한다.

---

### 5. 주요 결과 측정값

평균 티켓 해결시간은 다음과 같이 변화하였다.

| 연도 | 평균 해결시간 |
|---|---:|
| 2022년 | 5.99일 |
| 2023년 | 2.87일 |

따라서 평균 해결시간은 약 52% 감소하였다. 연구진은 이를 Ops Academy 이후의 생산성 및 운영효율성 향상으로 해석하였다.

또한 표준편차도 다음과 같이 감소하였다.

- 2022년: 6.97일
- 2023년: 3.48일

이는 단순히 평균 처리시간만 줄어든 것이 아니라, 업무 처리시간의 변동성도 감소했음을 보여준다.

---

### 6. 교육효과 평가 프레임워크

교육의 효과를 해석하기 위해 Kirkpatrick의 4단계 평가모형을 참고하였다.

1. 반응(Reaction)  
   직원들이 교육을 어떻게 받아들였는지 평가

2. 학습(Learning)  
   새로운 시스템과 업무지식을 습득했는지 평가

3. 행동(Behavior)  
   습득한 지식을 실제 티켓 처리와 지원업무에 적용했는지 평가

4. 결과(Results)  
   해결시간 감소, 생산성 향상, 사일로 완화와 같은 조직성과를 평가

이 연구에서는 특히 4단계인 운영성과와 비즈니스 결과를 티켓 데이터를 통해 정량적으로 확인하는 데 중점을 두었다.

---

### 7. ROI 평가 방식

연구는 교육프로그램의 투자수익률(ROI)을 단순한 비용절감뿐 아니라 유형·무형의 효과를 포함해 평가하려 했다.

기본 ROI 개념은 다음과 같다.

ROI(%) =
frac(text(프로그램 순편익))(text(프로그램 비용))
times 100


고려한 비용과 편익은 다음과 같다.

- 비용:
  - 교육자료 개발비
  - 교육 진행시간
  - 직원의 교육 참여시간
  - 지식 부족과 사일로로 인한 간접비용

- 편익:
  - 티켓 처리시간 감소
  - 지원인력의 생산성 향상
  - 전문인력의 유지
  - 여러 시스템을 지원할 수 있는 업무 유연성 증가
  - 오류 및 운영위험 감소
  - 데이터 사이언티스트의 업무중단 시간 감소

다만 이 연구의 ROI 계산은 통제집단이 없는 파일럿 연구에 기반한 추정치이므로, 조직 전체의 정확한 금전적 ROI를 확정한 것은 아니다.

---

### 8. 방법론의 핵심 특징과 한계

#### 핵심 특징

- 실제 업무 데이터에 기반한 후향적 연구
- 교육 전후의 연도별 비교
- 무작위 표본추출
- 통계적 유의성뿐 아니라 효과크기도 제시
- 교육, 운영성과, ROI를 연결하는 다층적 평가

#### 주요 한계

- 통제집단이 없어 성과향상이 전적으로 Ops Academy 때문이라고 단정하기 어려움
- 비교 가능한 직원 수가 7명으로 제한됨
- 직원 이직 및 조직변화의 영향을 완전히 제거하기 어려움
- 교육과 성과 간 인과관계를 장기적으로 검증하지 못함
- ROI의 일부는 실제 금액이 아닌 추정값임

---





## Systematic Description of the Method

This study did not develop a machine-learning model or a novel software architecture. Instead, it evaluated a Structured On-the-Job Training program, called the “Ops Academy,” designed to improve the operational effectiveness of a Data Science Platform Operations team.

### 1. Intervention: Ops Academy

Ops Academy was a cross-training program intended to reduce knowledge silos and dependence on individual system experts.

Key characteristics included:

- One 30-minute training session each week
- Training across data pipelines, applications, scientific tools, and SaaS systems
- Practical, work-related training connected to daily support activities
- Peer review
- Incident-response simulation
- Shadowing workshops
- Gamified learning content
- Structured on-the-job training

The main objective was to enable support agents to handle a broader range of systems and tickets rather than specializing in only one domain.

---

### 2. Data Collection

The researchers retrospectively extracted operational ticket data from the ServiceNow Service and Asset Management system.

- Total raw tickets: 4,070
- Observation period: primarily 2021–2023
- Main outcome variable: ticket resolution time
- Data extraction: API and Excel
- Statistical analysis: SAS Studio
- Visualization: Tableau

The ticket records contained timestamps, status changes, work notes, and final resolution information.

---

### 3. Participants and Sampling

Ops Academy was implemented from January through December 2023. Therefore, the main comparison was conducted between 2022 and 2023.

Inclusion criteria required that agents:

- Worked during both 2022 and 2023
- Participated in Ops Academy in 2023
- Had comparable operational involvement across both years

Tickets with resolution times more than two standard deviations from the mean were reviewed for possible outlier status. The final eligible population contained 461 tickets.

Using the Cochran sample-size approach, the researchers randomly selected:

- 210 tickets from 2022
- 210 tickets from 2023
- 420 tickets in total

The target confidence level was 95%, with a 5% margin of error.

---

### 4. Statistical Analysis

The researchers first calculated descriptive statistics for each year:

- Mean
- Standard deviation
- Minimum and maximum
- 95% confidence interval

They also assessed the distribution of the data using:

- Q-Q plots
- Box-and-whisker plots
- Kernel density plots

To test whether average resolution time differed between the two years, the researchers used an independent-samples t-test.

Main statistical results:

- Significance level: α = 0.01
- t-value: 5.68
- p-value: p < 0.0001
- Cohen’s d: 0.554

These results indicated a statistically significant reduction in resolution time after implementation of Ops Academy.

---

### 5. Main Outcome

Mean ticket resolution time changed as follows:

| Year | Mean resolution time |
|---|---:|
| 2022 | 5.99 days |
| 2023 | 2.87 days |

This represented an approximate 52% reduction in mean resolution time.

The standard deviation also decreased:

- 2022: 6.97 days
- 2023: 3.48 days

Thus, the program was associated not only with faster ticket resolution but also with lower variability in operational performance.

---

### 6. Training Evaluation Framework

The study informally applied the Kirkpatrick four-level model:

1. Reaction  
   How employees responded to the training

2. Learning  
   Whether employees acquired new technical and operational knowledge

3. Behavior  
   Whether employees applied that knowledge in real support activities

4. Results  
   Whether the program improved resolution time, productivity, and organizational performance

The study primarily emphasized the fourth level by using ticket-resolution data as an objective performance measure.

---

### 7. ROI Framework

The study also considered the return on investment of the training program.

The basic ROI formula was:


ROI(%) =
frac(text(Net program benefits))(text(Program costs))
times 100


Potential costs included:

- Training-material development
- Training delivery time
- Employee participation time
- Indirect costs associated with knowledge silos

Potential benefits included:

- Reduced ticket-resolution time
- Increased staff productivity
- Improved employee retention
- Broader cross-functional coverage
- Reduced operational risks and errors
- Less downtime for Data Science personnel seeking support

However, the ROI estimate should be interpreted cautiously because the study did not use a formal control group and some organizational benefits were estimated rather than directly measured.

---

### 8. Methodological Strengths and Limitations

Strengths

- Used real-world operational data
- Compared pre- and post-program outcomes
- Applied random ticket sampling
- Reported both statistical significance and effect size
- Connected training outcomes with operational and business impact

Limitations

- No control group was used
- Only seven agents met the year-over-year inclusion criteria
- Staffing changes and other organizational factors may have influenced the results
- Causal effects cannot be established conclusively
- Some ROI components were hypothetical or estimated

In summary, the study’s method was a structured cross-training intervention combined with retrospective ticket-data analysis, random sampling, descriptive statistics, t-testing, Kirkpatrick-based training evaluation, and an extended ROI framework.


<br/>
# Results




### 1. 비교 설계와 데이터

- 연구 설계: 2022년과 2023년의 운영지원 티켓을 비교한 사전–사후(retrospective pre–post) 분석이다.
- 중재 프로그램: Ops Academy는 2023년 1월 1일부터 12월 31일까지 운영되었으며, 지원 직원들이 다양한 데이터 사이언스 시스템과 도구를 교차 학습하도록 했다.
- 비교 대상:
  - 2022년: Ops Academy 시행 전 기준연도
  - 2023년: Ops Academy 시행 후 연도
- 원자료: 2020~2023년 동안 총 4,070건의 티켓이 기록되었다.
- 분석 대상: 두 해 모두 근무하면서 비교가 가능한 7명의 직원이 처리한 티켓을 대상으로 했다.
- 최종 표본: 2022년 210건, 2023년 210건으로 총 420건을 무작위 추출했다.
- 분석 전 해결 시간이 평균에서 2표준편차 이상 벗어난 티켓을 검토했으나, 제거해야 할 이상치는 발견되지 않았다.

### 2. 경쟁 모델 또는 대조군

이 연구에는 여러 교육 모델을 서로 비교하는 경쟁모델 비교나, Ops Academy를 받지 않은 별도의 동시대 대조군(control group)은 없었다.

따라서 분석은 다음과 같은 단순한 연도별 비교에 해당한다.

> 2022년 운영성과 → 2023년 Ops Academy 시행 후 운영성과

저자들은 향후에는 Ops Academy를 받지 않은 직원 또는 팀을 대조군으로 설정해 결과를 검증해야 한다고 제안했다. 그러므로 통계적으로 유의한 개선이 관찰되었지만, 개선이 전적으로 Ops Academy 때문에 발생했다고 단정하기에는 연구 설계상 한계가 있다.

### 3. 평가 데이터와 핵심 메트릭

주요 평가 지표는 티켓이 접수된 후 해결될 때까지 걸린 평균 해결 시간(resolution time, 일수)이었다.

| 지표 | 2022년 | 2023년 |
|---|---:|---:|
| 표본 수 | 210건 | 210건 |
| 평균 해결 시간 | 5.99일 | 2.87일 |
| 표준편차 | 6.97일 | 3.48일 |
| 95% 신뢰구간 | 5.04~6.93일 | 2.35~3.39일 |
| 최대 해결 시간 | 30일 | 18일 |

- 평균 해결 시간은 5.99일에서 2.87일로 3.12일 감소했다.
- 이는 약 52% 감소, 즉 해결 속도가 약 2배 가까이 빨라진 결과이다.
- 표준편차도 6.97일에서 3.48일로 줄어들어, 평균적인 속도뿐 아니라 직원 간 또는 티켓 간 처리 편차도 감소했다.
- 분포와 사분위 범위가 더 짧은 해결 시간 쪽으로 이동했으며, 논문은 이를 운영 효율성과 업무 숙련도 향상의 근거로 해석했다.

### 4. 통계적 검정 결과

두 연도의 해결 시간을 비교하기 위해 t-검정을 실시했다.

- t(418) = 5.68
- p < 0.0001
- 연구에서 설정한 유의수준: α = 0.01
- Cohen’s d = 0.554

해석하면 다음과 같다.

1. 2022년과 2023년의 평균 해결 시간 차이는 통계적으로 유의했다.
2. p값이 0.0001보다 작으므로, 단순한 우연으로 이러한 차이가 발생했을 가능성은 매우 낮다.
3. Cohen’s d가 0.554로 나타나, 효과 크기는 대략 중간 정도(medium effect)로 평가된다.
4. 따라서 결과는 통계적 유의성뿐 아니라 실제 운영상 의미도 가진다고 볼 수 있다.

또한 분산 비교를 위한 Folded F 검정에서도 유의한 차이가 확인되었다.

- F = 3.35
- p < 0.0001

이는 두 연도의 해결 시간 변동성이 동일하지 않으며, 2023년에 업무 처리 편차가 줄어들었음을 뒷받침한다.

### 5. ROI 및 사업적 효과

논문은 Ops Academy의 효과를 단순한 해결 시간 감소뿐 아니라 ROI 관점에서도 설명했다.

고려한 효과는 다음과 같다.

- 유형적 편익: 티켓 처리시간 감소에 따른 인건비 및 시간 절감
- 무형적 편익: 지식 사일로 감소, 업무 대체 가능성 향상, 오류 위험 감소, 협업 개선
- 조직 차원의 편익: 데이터 사이언티스트가 지원을 기다리는 시간이 줄어들어 연구개발 업무에 더 집중 가능
- 비용: 교육자료 개발, 강의 준비 및 참여에 소요된 시간

논문에는 가정에 기반한 ROI 예시로 약 146%의 수치가 제시되었다. 그러나 이는 실제 조직 전체의 재무성과를 직접 측정한 결과가 아니라, 티켓 처리시간 감소와 교육비용을 바탕으로 계산한 가상·추정치이다. 따라서 ROI는 참고용 사업성 분석으로 이해하는 것이 적절하다.

### 6. 종합 결론

Ops Academy 시행 후:

- 평균 티켓 해결 시간이 약 52% 감소했고,
- 처리시간의 변동성도 줄었으며,
- 통계적으로 유의하고 중간 수준의 효과가 확인되었다.

따라서 이 연구는 구조화된 직무교육과 교차훈련이 운영지원 조직의 생산성, 업무 유연성, 지식 사일로 완화에 긍정적인 영향을 줄 가능성을 보여준다.

다만 무작위 대조군이 없고, 분석 대상 직원 수가 7명으로 제한되었으며, 2022년과 2023년 사이의 인력·업무환경 변화가 완전히 통제되지 않았기 때문에, 결과를 인과관계로 확정하기보다는 Ops Academy와 운영성과 개선 사이의 유의한 연관성으로 해석하는 것이 타당하다.

---







### 1. Comparison Design and Data

- Study design: A retrospective pre–post comparison of operational support tickets from 2022 and 2023.
- Intervention: The Ops Academy was implemented from January through December 2023 to cross-train support staff across multiple data science systems, tools, and platforms.
- Comparison periods:
  - 2022: Pre-intervention baseline year
  - 2023: Post-intervention year
- Raw dataset: A total of 4,070 tickets were recorded from 2020 to 2023.
- Eligible personnel: Seven agents who were employed during both 2022 and 2023.
- Final sample: 210 randomly selected tickets from each year, for a total of 420 tickets.
- Potential outliers, defined as tickets with resolution times more than two standard deviations from the mean, were reviewed. No tickets required removal.

### 2. Competitive Models or Control Group

The study did not compare multiple training models and did not include a contemporaneous control group of employees who did not participate in Ops Academy.

The analysis was essentially:

> 2022 operational performance versus 2023 operational performance after Ops Academy implementation

Therefore, although the results showed significant improvement, the study cannot conclusively establish that Ops Academy alone caused the improvement. The authors recommend future studies using a control group or a comparison team without the intervention.

### 3. Evaluation Data and Main Metrics

The primary metric was ticket resolution time, measured in days from ticket submission to resolution.

| Metric | 2022 | 2023 |
|---|---:|---:|
| Sample size | 210 | 210 |
| Mean resolution time | 5.99 days | 2.87 days |
| Standard deviation | 6.97 days | 3.48 days |
| 95% confidence interval | 5.04–6.93 days | 2.35–3.39 days |
| Maximum resolution time | 30 days | 18 days |

Key findings:

- Mean resolution time decreased by 3.12 days, from 5.99 to 2.87 days.
- This represents an approximate 52% reduction in resolution time.
- The standard deviation was also reduced by approximately half, indicating more consistent ticket handling.
- The overall distribution shifted toward shorter resolution times, which the authors interpreted as evidence of improved efficiency and staff mastery.

### 4. Statistical Test Results

A t-test was used to compare resolution times between the two years.

- t(418) = 5.68
- p < 0.0001
- Significance threshold: α = 0.01
- Cohen’s d = 0.554

Interpretation:

1. The difference between 2022 and 2023 was statistically significant.
2. The very small p-value indicates that the observed difference is unlikely to be explained by random sampling variation alone.
3. Cohen’s d of 0.554 represents an approximately medium effect size.
4. Thus, the results demonstrated both statistical significance and practical operational relevance.

A Folded F test also identified a significant difference in variability:

- F = 3.35
- p < 0.0001

This suggests that resolution-time variability changed significantly and became lower in 2023.

### 5. ROI and Business Impact

The article discussed the impact of Ops Academy from an ROI perspective. The proposed benefits included:

- Tangible benefits: Reduced labor time and cost resulting from faster ticket resolution
- Intangible benefits: Reduced knowledge silos, improved backup coverage, lower error risk, and better collaboration
- Organizational benefits: Less waiting time for data scientists and greater focus on R&D activities
- Program costs: Curriculum development, preparation time, and staff participation time

The paper presented a hypothetical ROI estimate of approximately 146%. However, this was not a directly measured organization-wide financial return. It was an illustrative estimate based on assumptions about time savings and training costs. Therefore, it should be interpreted as an approximate business-case calculation rather than definitive financial evidence.

### 6. Overall Conclusion

Following implementation of Ops Academy:

- Mean ticket resolution time decreased by approximately 52%.
- Resolution-time variability also declined.
- The difference was statistically significant, with a medium practical effect.

Overall, the study suggests that structured on-the-job training and cross-training may improve operational productivity, workforce flexibility, and knowledge sharing.

However, because the study lacked a randomized control group, included only seven eligible agents, and could not fully control for staffing or environmental changes between 2022 and 2023, the findings should be interpreted as a strong association between Ops Academy implementation and improved operational outcomes, rather than definitive proof of causality.


<br/>
# 예제



### 1. 먼저 짚을 점: 일반적인 머신러닝 학습·테스트 데이터는 아님

이 논문은 트레이닝데이터와 테스트데이터로 예측모델을 만드는 연구가 아니라,  
2023년에 실시한 Ops Academy 교차훈련 프로그램의 효과를 전후 비교한 정량 연구입니다.

따라서 여기서 말하는 “training”은 머신러닝 학습 데이터가 아니라, 직원 교육(training)을 의미합니다. 논문에는 별도의 ML 모델, train/test split, 예측값 생성 과정은 제시되어 있지 않습니다.

---

### 2. 연구의 구체적인 테스크

연구의 핵심 테스크는 다음과 같습니다.

> Ops Academy 도입 이후 지원 티켓의 평균 해결 시간이 감소했는지 검증한다.

즉, 직원이 데이터 사이언스 플랫폼 관련 문제를 처리하고 해결하는 데 걸리는 시간이 교육 전후에 어떻게 달라졌는지를 분석했습니다.

주요 비교는 다음과 같습니다.

| 구분 | 기간 | 의미 |
|---|---:|---|
| 사전 비교 데이터 | 2022년 | Ops Academy 시행 전 |
| 사후 비교 데이터 | 2023년 | Ops Academy 시행 후 |

2023년에는 지원 직원들이 매주 약 30분씩 참여하는 구조화된 교차훈련을 받았습니다. 교육에는 다음과 같은 활동이 포함되었습니다.

- 동료 검토(peer review)
- 장애·사건 시뮬레이션(incident simulation)
- 섀도잉 워크숍(shadowing workshop)
- 콘텐츠 게임화(content gamification)
- 여러 데이터 사이언스 시스템과 도구에 대한 교차훈련

---

### 3. 데이터의 입력과 출력

#### 입력 데이터

ServiceNow SAM 티켓 시스템에서 다음과 같은 운영 데이터를 추출했습니다.

- 티켓이 접수된 연도
- 티켓 해결까지 걸린 시간
- 티켓의 상태 변화와 처리 이력
- 담당 지원 직원
- 지원 대상 시스템·애플리케이션·파이프라인
- 티켓의 작업 기록과 설명
- 티켓이 속한 업무 영역

논문에서 직접 확인할 수 있는 대표적인 입력 변수는 티켓 해결 시간(`Reso_Time`)입니다.

예를 들어 하나의 티켓이 다음과 같이 표현될 수 있습니다.

```text
티켓 ID: 1001
연도: 2022
담당자: A3
대상 시스템: 데이터 파이프라인
해결 시간: 6일
```

또 다른 예시는 다음과 같습니다.

```text
티켓 ID: 2001
연도: 2023
담당자: A3
대상 시스템: SaaS 애플리케이션
해결 시간: 2일
```

단, 위 티켓 ID와 세부 내용은 논문에 실제로 공개된 개별 기록이 아니라, 논문의 변수 구조를 이해하기 위한 예시입니다.

#### 분석 출력

분석에서 얻고자 한 출력은 다음과 같습니다.

- 연도별 평균 티켓 해결 시간
- 해결 시간의 표준편차
- 95% 신뢰구간
- 2022년과 2023년의 평균 차이
- t-test 통계량과 p-value
- 효과크기(Cohen’s d)
- 업무량 분포 변화
- 교육으로 인한 생산성 및 비용 절감 가능성

---

### 4. 실제 데이터 규모와 데이터 선별 과정

논문에 제시된 데이터 흐름은 다음과 같습니다.

```text
전체 티켓 4,070건
        ↓
2022년과 2023년에 모두 근무한 직원 및 적격 티켓 선별
        ↓
분석 가능한 티켓 461건
        ↓
무작위 표본 추출
        ↓
최종 분석 표본 420건
        ├─ 2022년: 210건
        └─ 2023년: 210건
```

분석 대상 직원은 2022년과 2023년에 모두 근무했고 2023년 Ops Academy에 참여한 7명이었습니다.

이 연구는 극단적인 이상치가 결과를 왜곡하지 않도록 해결 시간이 평균에서 2표준편차 이상 벗어난 티켓을 검토했습니다. 그러나 작업 기록과 티켓 설명을 확인한 결과, 제거해야 할 데이터는 없었다고 보고했습니다.

---

### 5. “트레이닝데이터와 테스트데이터”에 대응시키면

이 논문에는 ML에서 말하는 train/test 데이터셋은 없지만, 연구 설계를 이에 대응시켜 설명하면 다음과 같습니다.

| 머신러닝 용어 | 이 논문에서의 대응 |
|---|---|
| Training data | 해당 없음. 예측모델을 학습하지 않음 |
| Test data | 해당 없음. 새로운 데이터에 대한 예측 성능을 평가하지 않음 |
| 입력값 | 티켓의 연도, 담당자, 시스템, 해결 시간 등 |
| 출력값 또는 결과변수 | 티켓 해결 시간 및 연도별 생산성 차이 |
| 모델 | 통계적 비교 분석 및 t-test |
| 평가 기준 | p-value, 평균 차이, 신뢰구간, Cohen’s d |

따라서 이 연구는 다음과 같은 분석 구조에 가깝습니다.

```text
2022년 티켓 해결 시간
        vs.
2023년 티켓 해결 시간
        ↓
평균·표준편차 비교
        ↓
t-test 실시
        ↓
Ops Academy 이후 차이가 통계적으로 유의한지 판단
```

---

### 6. 실제 분석 결과

주요 결과는 다음과 같습니다.

| 지표 | 2022년 | 2023년 |
|---|---:|---:|
| 표본 수 | 210건 | 210건 |
| 평균 해결 시간 | 5.99일 | 2.87일 |
| 표준편차 | 6.97일 | 3.48일 |
| 95% 신뢰구간 | 5.04~6.93일 | 2.35~3.39일 |

2023년 평균 해결 시간은 2022년보다 약 3.11일 감소했으며, 약 50% 이상 개선된 것으로 해석되었습니다.

통계검정 결과는 다음과 같습니다.

- t = 5.68
- p < 0.0001
- Cohen’s d = 0.554
- 유의수준 α = 0.01

즉, 2022년과 2023년의 해결 시간 차이는 통계적으로 유의했으며, Ops Academy가 생산성 향상에 기여했을 가능성이 있다고 결론 내렸습니다.

다만 이 연구는 무작위 대조군을 둔 실험이 아니므로, 해결 시간 감소가 전적으로 Ops Academy 때문이라고 단정할 수는 없습니다. 논문도 후속 연구에서는 교육을 받지 않은 비교군이 필요하다고 설명합니다.

---

### 7. 예시로 정리한 전체 흐름

#### 예시 입력

```text
[2022년 티켓]
- 문제: 데이터 파이프라인 오류
- 담당자: A3
- 해결 시간: 6일

[2023년 티켓]
- 문제: SaaS 도구 접근 권한 문제
- 담당자: A3
- 해결 시간: 2일
```

#### 연구 테스크

```text
2022년과 2023년에 같은 조건을 충족하는 티켓들을 모아
교육 전후의 평균 해결 시간을 비교한다.
```

#### 예시 출력

```text
2022년 평균: 5.99일
2023년 평균: 2.87일
평균 감소폭: 3.11일
통계적 유의성: p < 0.0001
```

#### 해석

```text
Ops Academy 이후 지원 직원들이 여러 시스템을 처리할 수 있게 되었고,
티켓 해결 시간이 크게 감소했으며,
지원 업무의 생산성과 팀의 업무 유연성이 향상되었다.
```

---




### 1. Important clarification: this is not a conventional ML training/test study

The paper does not build a machine-learning prediction model. The term training refers to employee training through the Ops Academy, not to training data used to fit an algorithm.

There is no separate machine-learning training set, test set, predictive model, or model accuracy score. Instead, the study retrospectively compares support-ticket outcomes before and after the training program.

---

### 2. Main research task

The central task was:

> To determine whether ticket-resolution time decreased after the implementation of Ops Academy.

The researchers compared:

| Period | Meaning |
|---|---|
| 2022 | Pre-Ops Academy period |
| 2023 | Post-Ops Academy period |

The 2023 program provided approximately 30 minutes of weekly structured cross-training. Activities included:

- Peer review
- Incident simulation
- Shadowing workshops
- Gamified learning content
- Cross-training on data-science systems, tools, and infrastructure

---

### 3. Inputs and outputs

#### Input data

Data were extracted from the ServiceNow SAM ticketing system. Relevant variables included:

- Ticket year
- Ticket-resolution time
- Ticket status and milestones
- Assigned support agent
- Supported system, application, or pipeline
- Work notes and ticket descriptions
- Operational domain

The primary quantitative variable was ticket-resolution time, referred to as `Reso_Time`.

An illustrative record might look like this:

```text
Ticket ID: 1001
Year: 2022
Agent: A3
System: Data pipeline
Resolution time: 6 days
```

Another illustrative record might be:

```text
Ticket ID: 2001
Year: 2023
Agent: A3
System: SaaS application
Resolution time: 2 days
```

These individual ticket examples are illustrative; the paper does not publish identifiable ticket-level records.

#### Outputs

The analysis produced:

- Mean resolution time by year
- Standard deviation
- 95% confidence intervals
- Difference between 2022 and 2023 means
- t-test statistic and p-value
- Cohen’s d effect size
- Changes in workload distribution
- Potential productivity and cost benefits

---

### 4. Data-selection process

The data flow was approximately:

```text
4,070 total tickets
        ↓
Eligibility screening
        ↓
461 viable tickets
        ↓
Random sampling
        ↓
420 final observations
        ├─ 210 tickets from 2022
        └─ 210 tickets from 2023
```

Seven agents met the inclusion criteria. They were employed in both years and participated in Ops Academy during 2023.

Potential outliers were reviewed using ticket descriptions and work notes. The authors reported that no data points required removal after this review.

---

### 5. Relationship to training and test data

Although the paper does not use ML train/test terminology, the concepts can be mapped as follows:

| ML concept | Equivalent in this study |
|---|---|
| Training data | Not applicable; no predictive model was trained |
| Test data | Not applicable; no out-of-sample prediction was evaluated |
| Inputs | Ticket year, agent, system, resolution time, and ticket history |
| Outcome variable | Resolution time and year-over-year productivity difference |
| Analytical method | Descriptive statistics and a t-test |
| Evaluation criteria | p-value, confidence interval, mean difference, and Cohen’s d |

The analytical structure was:

```text
2022 ticket-resolution times
        vs.
2023 ticket-resolution times
        ↓
Descriptive comparison
        ↓
t-test
        ↓
Determine whether the difference was statistically significant
```

---

### 6. Main findings

| Measure | 2022 | 2023 |
|---|---:|---:|
| Sample size | 210 | 210 |
| Mean resolution time | 5.99 days | 2.87 days |
| Standard deviation | 6.97 days | 3.48 days |
| 95% confidence interval | 5.04–6.93 days | 2.35–3.39 days |

The average resolution time decreased by approximately 3.11 days, representing an improvement of more than 50%.

The reported statistical results were:

- t = 5.68
- p < 0.0001
- Cohen’s d = 0.554
- Significance level α = 0.01

These findings indicate a statistically significant reduction in resolution time after the introduction of Ops Academy.

However, because the study did not use a randomized control group, the authors did not establish that Ops Academy alone caused the improvement. They recommend future comparisons with staff who did not receive the training.

---

### 7. Simplified end-to-end example

#### Example input

```text
[2022 ticket]
- Issue: Data-pipeline failure
- Agent: A3
- Resolution time: 6 days

[2023 ticket]
- Issue: SaaS access problem
- Agent: A3
- Resolution time: 2 days
```

#### Research task

```text
Compare eligible tickets from 2022 and 2023
to determine whether resolution time improved after training.
```

#### Example output

```text
2022 mean: 5.99 days
2023 mean: 2.87 days
Mean reduction: 3.11 days
Statistical significance: p < 0.0001
```

#### Interpretation

```text
After Ops Academy, support staff appeared able to handle
a broader range of systems and resolve tickets more quickly,
indicating improved operational productivity and cross-functional coverage.
```

<br/>
# 요약



1. 연구진은 ServiceNow 티켓 데이터 4,070건 중 2022년과 2023년에 모두 근무한 7명의 지원 에이전트가 처리한 티켓 420건을 무작위 표본으로 선정해 SAS로 비교·분석했다.  
2. Ops Academy 도입 후 평균 티켓 해결 시간이 2022년 5.99일에서 2023년 2.87일로 50% 이상 단축되었으며, 그 차이는 통계적으로 유의했다(t=5.68, p<0.0001, Cohen’s d=0.554).  
3. 예를 들어 주 1회 30분의 교차훈련에 동료 검토, 장애 상황 시뮬레이션, 섀도잉, 게임화 학습을 결합함으로써 업무 사일로를 줄이고 지원 범위와 생산성을 높였으며, 연구진은 이를 비용 대비 효과적인 교육 모델로 제시했다.  




1. The researchers retrospectively analyzed 4,070 ServiceNow support tickets and randomly selected 420 tickets handled by seven agents who worked in both 2022 and 2023, using SAS for comparison.  
2. After the Ops Academy was introduced, mean ticket resolution time fell by more than 50%, from 5.99 days in 2022 to 2.87 days in 2023, with a statistically significant difference (t=5.68, p<0.0001, Cohen’s d=0.554).  
3. For example, the program combined weekly 30-minute cross-training with peer review, incident simulations, shadowing, and gamification, reducing knowledge silos and improving support coverage and productivity as a cost-effective training model.

<br/>
# 기타



### 1. Table 1. Raw support totals

결과
- 2020~2023년 동안 총 4,070건의 지원 티켓이 처리되었다.
- 전체 티켓 중 1~3일 내 해결된 건수는 2,716건으로 가장 많았다.
- 15일 이상 걸린 티켓은 763건이었다.
- 연도별 총 처리 건수는 2022년이 1,686건으로 가장 많았고, 2023년에는 1,157건으로 감소했다.

인사이트
- 이 표는 Ops Academy의 효과를 분석하기 전, 팀의 전체 업무량과 해결 속도에 대한 기초 현황(baseline) 을 보여준다.
- 2023년의 티켓 수가 감소한 것만으로 생산성 향상을 단정할 수는 없지만, 이후 분석에서 해결시간 감소가 실제로 나타났는지 비교하는 기준이 된다.
- 15일 이상 장기 미해결 티켓도 상당수 존재했기 때문에, 해결시간 단축은 단순한 평균 개선을 넘어 운영 리스크와 업무 적체를 줄이는 의미가 있다.

---

### 2. Fig. 1. 2020–2023 Workload Personas by Agent

결과
- 각 행은 연도, 각 열은 익명화된 지원 담당자(A1~A10)를 나타낸다.
- 2021~2022년에는 담당자별 업무 영역이 비교적 고정되어 있어, 특정 담당자가 특정 시스템이나 도메인에 집중되는 모습이 나타났다.
- 2023년 Ops Academy 이후에는 담당자별 업무 분포가 더 다양해지고 서로 겹치는 영역이 증가했다.

인사이트
- 이 그림의 핵심은 단순한 업무량 변화가 아니라 지식 사일로가 완화되고 교차훈련이 진행되었다는 점이다.
- 특정 시스템을 한두 명만 지원하던 구조에서 여러 담당자가 다양한 영역을 처리할 수 있는 구조로 이동했다.
- 따라서 Ops Academy는 해결시간뿐 아니라 지원 유연성, 대체 가능성, 인력 확장성을 높인 것으로 해석된다.
- 다만 Fig. 1은 업무 분포의 변화와 패턴을 보여주는 자료이며, 그 자체가 통계적 인과관계를 증명하는 것은 아니다.

---

### 3. Table 2. Descriptive statistics

결과
- 2022년 평균 티켓 해결시간: 5.99일
- 2023년 평균 티켓 해결시간: 2.87일
- 평균 해결시간은 약 3.11일 감소, 약 52% 단축되었다.
- 표준편차도 2022년 6.97일에서 2023년 3.48일로 감소했다.
- 95% 신뢰구간은 다음과 같다.
  - 2022년: 5.04~6.93일
  - 2023년: 2.35~3.39일

인사이트
- 2023년에는 평균 해결시간뿐 아니라 변동성도 크게 줄었다.
- 즉, 일부 티켓만 빠르게 처리된 것이 아니라 전반적으로 더 일관되고 예측 가능한 지원 서비스가 제공되었다.
- 최소값은 두 연도 모두 0일이며, 이는 24시간 이내 해결된 티켓을 의미한다.
- Cohen’s d가 0.554로 나타나, 통계적 유의성뿐 아니라 실무적으로도 중간 정도의 효과가 있었음을 보여준다.
- 다만 평균 감소가 전적으로 Ops Academy 때문이라고 단정하기보다는, 인력 구성·업무 난이도·프로세스 변화 등 다른 요인도 함께 고려해야 한다.

---

### 4. Table 3. Final t-test findings

결과
- 두 연도의 해결시간 차이에 대한 t-test 결과:
  - t = 5.68
  - p < 0.0001
  - 자유도(df) = 418
- 연구에서 설정한 유의수준 α = 0.01보다 p값이 훨씬 작았다.
- 따라서 2022년과 2023년의 평균 해결시간 차이는 통계적으로 유의하다.
- Folded F test에서도 분산 차이가 유의하게 나타났다.

인사이트
- 2023년의 해결시간 감소는 우연한 표본 변동만으로 설명되기 어렵다는 통계적 근거를 제공한다.
- t-test는 “두 연도의 평균이 다른가”를 검증하며, 결과는 Ops Academy 이후 운영 효율성이 개선되었다는 연구 질문 1을 지지한다.
- 특히 분산도 감소했기 때문에, 팀이 더 빠르게 처리했을 뿐 아니라 담당자와 티켓 간 성과 편차도 줄어든 것으로 볼 수 있다.
- 그러나 통제집단이 없는 사후적 비교이므로, 엄밀한 인과효과를 확정하기보다는 강한 연관성과 실무적 개선 증거로 해석하는 것이 적절하다.

---

### 5. Fig. 2. SAS Quantile–Quantile Plot

결과
- QQ plot에서 관측값이 기준선에 대체로 가깝게 분포한다.
- 이는 해결시간 데이터가 통계분석에 필요한 정규성에 크게 위배되지 않았음을 시사한다.

인사이트
- t-test를 적용하기 위한 데이터의 분포 가정이 대체로 타당했음을 뒷받침한다.
- 따라서 Table 2와 Table 3의 평균 및 t-test 결과가 극단적인 비정상 분포 때문에 왜곡되었을 가능성이 낮다는 점을 보여준다.
- 다만 해결시간 데이터는 본질적으로 0일 이상의 값이고 일부 장기 티켓이 존재하므로, QQ plot이 완벽한 정규성을 의미하는 것은 아니다.

---

### 6. Fig. 3. Distribution of “Reso_Time”

결과
- 2023년 분포가 2022년에 비해 왼쪽, 즉 0일에 가까운 방향으로 이동했다.
- 박스플롯과 밀도곡선에서도 2023년 해결시간의 중앙 경향과 분포 폭이 모두 감소한 모습을 확인할 수 있다.

인사이트
- 이 그림은 평균값만으로는 알기 어려운 전체 분포 변화를 시각적으로 보여준다.
- 2023년에는 티켓이 전반적으로 더 빨리 해결되었고, 긴 해결시간의 발생도 줄어든 것으로 해석된다.
- 논문에서 말하는 “mastery”는 지원 담당자들이 다양한 시스템을 이해하고, 요구사항을 더 신속하고 안정적으로 충족할 수 있게 되었다는 의미로 연결된다.

---

### 7. ROI 관련 공식 및 예시 계산

결과
논문은 Ops Academy의 효과를 단순한 해결시간 개선뿐 아니라 비용과 조직적 가치까지 확장해 평가하려고 했다.

주요 개념은 다음과 같다.

- 유형 비용: 교육자료, 교육 준비시간, 수업 운영시간
- 유형 편익: 해결시간 단축으로 절약된 시간과 비용, 인력 유지 효과
- 무형 비용: 지식 사일로와 역량 부족으로 인한 운영 비효율
- 무형 편익: 교차지원 확대, 오류 및 운영 리스크 감소
- Data Science 조직의 편익: 연구자가 지원을 기다리는 시간이 줄어들어 R&D 업무에 더 집중할 수 있음

논문은 가정된 비용과 편익을 이용해 예시적으로 약 146.29%의 ROI를 제시했다.

인사이트
- Ops Academy는 교육비만 발생하는 프로그램이 아니라, 지원시간 단축과 R&D 생산성 향상을 통해 조직 전체에 편익을 발생시킬 수 있다.
- 특히 지원팀 내부의 효율성뿐 아니라, 지원을 받는 데이터 과학자들의 업무 중단시간까지 고려해야 실제 조직 ROI를 평가할 수 있다.
- 다만 제시된 ROI는 실제 전사 재무성과를 직접 측정한 결과가 아니라, 가정에 기반한 개념적·예시적 계산이다.
- 저자들도 통제집단과 더 넓은 조직 데이터를 활용한 후속 검증이 필요하다고 설명한다.

---

### 8. Appendix

- 논문 본문에 별도의 Appendix는 제시되어 있지 않다.
- 대신 표, 그림, 통계검정, ROI 공식이 연구 결과와 해석을 뒷받침하는 부속 자료 역할을 한다.

---

### 전체적인 핵심 인사이트

이 논문의 표와 그림은 Ops Academy 이후 다음 세 가지 변화가 함께 나타났음을 보여준다.

1. 속도 향상: 평균 해결시간이 5.99일에서 2.87일로 감소  
2. 일관성 향상: 해결시간의 표준편차와 분포 폭이 감소  
3. 지식 사일로 완화: 담당자들이 더 다양한 시스템과 업무를 처리  

따라서 Ops Academy는 단순한 교육 프로그램이라기보다, 교차훈련을 통해 팀의 운영 효율성, 대응 유연성, 업무 지속가능성을 높인 조직개발 intervention으로 해석할 수 있다. 다만 통제집단이 없으므로, 결과는 인과관계의 확정이라기보다 통계적으로 유의한 개선을 보여주는 후향적 평가 결과로 이해하는 것이 적절하다.

---




### 1. Table 1. Raw support totals

Findings
- A total of 4,070 support tickets were resolved between 2020 and 2023.
- Most tickets, 2,716 cases, were resolved within 1–3 days.
- 763 tickets required more than 15 days to resolve.
- The highest annual ticket volume occurred in 2022, with 1,686 tickets.

Insight
- Table 1 establishes the operational baseline before evaluating Ops Academy.
- Although ticket volume decreased in 2023, reduced volume alone cannot prove higher productivity.
- The considerable number of long-running tickets shows why faster resolution could reduce backlog and operational risk.

---

### 2. Fig. 1. 2020–2023 Workload Personas by Agent

Findings
- Rows represent years, while columns represent anonymized agents and their workload domains.
- Work assignments were relatively concentrated in 2021 and 2022, suggesting that individual agents were closely associated with specific systems or domains.
- After Ops Academy was introduced in 2023, workload distributions became more diverse and overlapping.

Insight
- The figure primarily demonstrates a reduction in knowledge silos.
- More agents became capable of supporting multiple domains, improving coverage, substitution capacity, and scalability.
- However, the figure shows a change in workload patterns; it does not independently establish a causal relationship.

---

### 3. Table 2. Descriptive statistics

Findings
- Mean resolution time in 2022: 5.99 days
- Mean resolution time in 2023: 2.87 days
- Average resolution time decreased by approximately 3.11 days, or about 52%.
- Standard deviation decreased from 6.97 days to 3.48 days.
- The 95% confidence intervals were:
  - 2022: 5.04–6.93 days
  - 2023: 2.35–3.39 days

Insight
- The improvement was not limited to the average; variability also declined substantially.
- This suggests that support became faster and more consistent across tickets.
- Cohen’s d was 0.554, indicating a moderate practical effect in addition to statistical significance.
- Nevertheless, other factors—such as staffing, ticket complexity, or process changes—may also have contributed.

---

### 4. Table 3. Final t-test findings

Findings
- The t-test comparing the two years produced:
  - t = 5.68
  - p < 0.0001
  - Degrees of freedom = 418
- The p-value was much smaller than the study’s significance level of α = 0.01.
- The difference in resolution time was therefore statistically significant.
- The Folded F test also indicated a significant difference in variance.

Insight
- The 2023 reduction in resolution time is unlikely to be explained by random sampling variation alone.
- The findings support the first research question: operational effectiveness differed significantly after Ops Academy implementation.
- The reduction in variance also suggests more stable and predictable team performance.
- Because the study did not include a control group, the results should be interpreted as strong evidence of improvement and association, rather than definitive proof of causality.

---

### 5. Fig. 2. SAS Quantile–Quantile Plot

Findings
- The observations were generally close to the reference line.
- This indicates that the resolution-time data were reasonably consistent with the normality assumption.

Insight
- The plot supports the use of parametric analyses such as the t-test.
- It also suggests that the reported results were not primarily driven by a severely abnormal distribution.
- However, the data are naturally bounded at zero and include some long-duration tickets, so perfect normality should not be assumed.

---

### 6. Fig. 3. Distribution of “Reso_Time”

Findings
- The 2023 distribution shifted toward the left, closer to zero days.
- The boxplots and density curves show both a lower central tendency and a narrower distribution in 2023.

Insight
- This figure illustrates the overall distributional change more clearly than the mean alone.
- Tickets were generally resolved more quickly, and extreme delays appeared to be less common.
- The shift toward zero is interpreted in the paper as evidence of greater operational mastery: staff were able to understand and fulfill support requirements more quickly and consistently.

---

### 7. ROI formulas and illustrative calculation

Findings
The paper evaluates Ops Academy beyond resolution time by considering both financial and organizational value.

Key components include:

- Tangible costs: training materials, preparation time, and class time
- Tangible benefits: time and cost savings from faster resolution and staff retention
- Intangible costs: inefficiencies caused by knowledge gaps and silos
- Intangible benefits: broader cross-functional coverage and reduced operational risk
- Benefits to Data Science: less time waiting for support and fewer interruptions to R&D work

Using assumed costs and benefits, the paper presents an illustrative ROI of approximately 146.29%.

Insight
- Ops Academy may create value not only within Platform Operations but also for the wider R&D organization.
- A complete ROI assessment should include the time saved by data scientists and other users who depend on operational support.
- The reported ROI is illustrative rather than a direct measurement of enterprise-wide financial performance.
- The authors recommend future validation using a control group and broader organizational data.

---

### 8. Appendix

- No separate appendix is provided in the article.
- The tables, figures, statistical tests, and ROI formulas function as the primary supplementary evidence supporting the analysis.

---

### Overall insight

The tables and figures collectively indicate three major changes after Ops Academy:

1. Faster resolution: mean resolution time decreased from 5.99 to 2.87 days  
2. Greater consistency: variation and distribution width decreased  
3. Reduced knowledge silos: agents handled a broader range of systems and domains  

Overall, Ops Academy can be understood not merely as a training course but as an organizational-development intervention that improved operational efficiency, support flexibility, and sustainability through cross-training. However, because the study lacked a control group, the findings should be viewed as evidence of statistically significant post-implementation improvement rather than definitive causal proof.

<br/>
# refer format:
### BibTeX

```bibtex
@article{Duffy2026TowardsMastery,
  author  = {Duffy, Seth R. and Patel, Vishal and Vo-Schneider, Phuong (Clare) and Schultz, Timothy and Chu, Carolyn and Wu, Xiaoying},
  title   = {Towards Mastery: Evidence from One Team's Pursuit of Operational Excellence},
  journal = {Human Factors in Healthcare},
  year    = {2026},
  volume  = {9},
  pages   = {100129},
  doi     = {10.1016/j.hfh.2026.100129},
  url     = {https://doi.org/10.1016/j.hfh.2026.100129},
  publisher = {Elsevier}
}
```

### Chicago Style  

Duffy, Seth R., Vishal Patel, Phuong (Clare) Vo-Schneider, Timothy Schultz, Carolyn Chu, and Xiaoying Wu. “Towards Mastery: Evidence from One Team’s Pursuit of Operational Excellence.” *Human Factors in Healthcare* 9 (2026): 100129. https://doi.org/10.1016/j.hfh.2026.100129.




