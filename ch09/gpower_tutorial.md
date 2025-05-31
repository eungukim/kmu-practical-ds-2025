[![Two independent proportions power analysis | G\*Power Data Analysis Examples](https://tse1.mm.bing.net/th?id=OIP.8ng8rJkK6zwfwENBm50KWwHaHx\&pid=Api)](https://stats.oarc.ucla.edu/other/gpower/two-independent-proportions-power-analysis/)

아래 단계만 따라 하면 **G\*Power**로 A/B-테스트(두 집단 비교)의 **표본 크기 계산‧검정력 분석**을 바로 해볼 수 있어요.
(버전 3.1.9.x 기준‧Windows‧macOS 동일)

---

## 1. 프로그램 준비

| 단계  | 설명                                                                                                                                                                                                                                            |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1-1 | 공식 사이트([https://www.psychologie.hhu.de/arbeitsgruppen/allgemeine-psychologie-und-arbeitspsychologie/gpower](https://www.psychologie.hhu.de/arbeitsgruppen/allgemeine-psychologie-und-arbeitspsychologie/gpower))에서 G\*Power 3.1 최신판 다운로드 & 설치 |
| 1-2 | 실행 후 기본 인터페이스 확인 <br>① **Test family** <br>② **Statistical test** <br>③ **Type of power analysis** <br>④ **Input parameters** <br>⑤ **Output parameters**                                                                                                                                    |

---

## 2. 주로 쓰는 두 가지 A/B-테스트 설정

| 비교 지표               | 메뉴 선택                                                                                                                   | 효과크기 지표                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **전환율·비율** (예: 클릭률) | **Test family** → *Z tests* <br> **Statistical test** → *Proportions - Difference between two independent proportions*  | **Cohen’s *h*** (비율의 차이)      |
| **평균 값** (예: 체류시간)  | **Test family** → *t tests* <br> **Statistical test** → *Means - Difference between two independent means (two groups)* | **Cohen’s *d*** (평균의 차이/표준편차) |

> 같은 화면에서 **Tail(s)**(단측/양측), **Allocation ratio N2/N1**(1:1이 기본)도 함께 지정합니다.

---

## 3. “A priori” 분석으로 **필요 표본 수** 구하기

1. **Type of power analysis** → *A priori: Compute required sample size – given α, power, and effect size*
2. **Input parameters** 입력

   * **Effect size**

     * 비율인 경우

       $$
       h = 2\arcsin\sqrt{p_2}-2\arcsin\sqrt{p_1}
       $$

       (h = 0.2 작음, 0.5 중간, 0.8 큼)
     * 평균인 경우
       $d = \dfrac{\mu_2-\mu_1}{\sigma_{\text{pooled}}}$
   * **α error prob** (보통 0.05)
   * **Power (1-β err prob)** (보통 0.8 또는 0.9)
3. **Calculate** 클릭 → 오른쪽 **Sample size group 1/2** 확인

> **예시**
> 전환율 A=5 %, B=6 %, α=0.05, Power=0.80, 양측 검정, 1:1 할당
> → h ≈ 0.044, 필요 표본 ≈ ≈ **13,750명**씩. (전환율 차이가 작을수록 N이 급격히 증가합니다.)

---

## 4. 다른 분석 모드

| 모드              | 언제 사용?                             | 설정                                                                        |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------- |
| **Post hoc**    | 이미 있는 샘플 크기에서 **실제 검정력** 확인        | *Post hoc: Compute achieved power – given α, sample size and effect size* |
| **Sensitivity** | 고정된 샘플 크기에서 **검출 가능한 최소 효과크기** 구하기 | *Sensitivity: Compute effect size – given α, power, and sample size*      |
| **Compromise**  | α와 β를 동일비율로 조정해 타협점 찾기             | *Compromise: Compute β/α ratio – given sample size and effect size*       |

(메뉴 위치는 모두 **Type of power analysis** 드롭다운)

---

## 5. 결과 해석 팁

* **Actual power**가 0.8 이상이어야 통상 “충분히 강한 테스트”로 간주.
* **Critical z/t 값**은 내부 검정 경계; 리포트 작성 시 인용 가능.
* G\*Power는 **정규 근사 z-test**를 사용합니다. 기대 전환 수가 5 미만이면 **‘Exact test (Fisher)’** 메뉴를 고려하세요.

---

## 6. 자주 묻는 질문(FAQ)

| 질문                           | 답변                                                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **효과크기를 몰라요**                | 파일럿 테스트 또는 과거 로그에서 p₁·p₂, μ₁·μ₂, σ 추정 → 위 공식으로 변환                                                    |
| **한쪽만 90 % 트래픽을 쓰고 싶어요**     | **Allocation ratio N2/N1**을 9 로 입력 (A : B = 10 : 90)                                                 |
| **데이터 수집 후 분석도 G\*Power에서?** | 아니요. G\*Power는 표본 설계·검정력 계산용 도구입니다. 수집한 데이터의 실제 검정은 R/Python(Statsmodels), Excel, 혹은 분석 플랫폼에서 수행하세요. |

---

### 빠른 체크리스트

* [ ] Test family & Statistical test 올바르게 선택했나요?
* [ ] 양측/단측 설정이 가설 방향과 일치하나요?
* [ ] 효과크기 계산 근거가 명확한가요?
* [ ] 목표 검정력(≥0.8) 확보 되었나요?

위 체크리스트만 지켜도 **A/B 실험 설계** 단계의 통계적 오류를 크게 줄일 수 있습니다.
