# Git과 GitHub 기초 및 데이터사이언스 실무 활용 튜토리얼

---

## Git과 GitHub의 기초 개념

### Git이란?

Git은 **버전 관리 시스템(Version Control System, VCS)** 으로, 파일의 변경 이력을 추적하고 관리할 수 있게 해줍니다. 특히 소스 코드나 데이터 분석 프로젝트와 같이 협업이 필요한 작업에서 매우 유용합니다.

### GitHub란?

GitHub는 Git을 기반으로 한 웹 기반의 호스팅 서비스로, 프로젝트의 협업과 공유를 쉽게 만들어줍니다. GitHub를 통해 여러 사람이 동시에 프로젝트에 참여하고, 코드 리뷰, 이슈 관리, 문서화 등을 효율적으로 수행할 수 있습니다.

---

## 데이터사이언티스트가 Git과 GitHub를 사용하는 이유

데이터사이언티스트는 분석 코드, 데이터셋, 결과 보고서 등을 체계적으로 관리하고 협업하기 위해 Git과 GitHub를 사용합니다.

### 주요 활용 사례

- 분석 코드의 버전 관리 및 변경 이력 추적
- 팀원 간의 코드 공유 및 협업
- 분석 결과의 재현성 확보
- 이슈 관리 및 프로젝트 진행 상황 공유

---

## 실전 프로젝트에서의 Git과 GitHub 활용 예시

다음은 데이터사이언스 프로젝트에서 Git과 GitHub를 활용하는 구체적인 사례입니다.

| 단계 | 작업 내용 | Git/GitHub 활용 |
|---|---|---|
| 1 | 데이터 분석 환경 설정 | GitHub에서 프로젝트 레포지토리 생성 및 초기 환경 설정 파일 업로드 |
| 2 | 데이터 전처리 및 분석 코드 작성 | Git을 통해 코드 변경 사항을 커밋(commit)하고 GitHub에 푸시(push) |
| 3 | 팀원 간 코드 리뷰 및 피드백 | GitHub의 Pull Request 기능을 활용하여 코드 리뷰 진행 |
| 4 | 분석 결과 공유 및 문서화 | GitHub의 Wiki 또는 Markdown 문서를 통해 분석 결과 문서화 |

---

## 수업 자료 다운로드 방법 (Pull)

수업 레포지토리: [kmu-practical-ds-2025](https://github.com/dandacompany/kmu-practical-ds-2025)

### 최초 클론(Clone) 방법

처음 레포지토리를 로컬 환경에 복사할 때는 다음 명령어를 사용합니다.

```bash
git clone https://github.com/dandacompany/kmu-practical-ds-2025.git
```

### 매주 수업 자료 업데이트 받기 (Pull)

이미 클론한 레포지토리에서 최신 자료를 받으려면 다음 명령어를 사용합니다.

```bash
cd kmu-practical-ds-2025
git pull origin main
```

---

## 로컬 파일과 충돌 발생 시 대처 방법

Git에서 충돌(conflict)은 로컬에서 수정한 파일과 원격 저장소에서 수정된 파일이 서로 다를 때 발생합니다.

### 충돌 발생 시 나타나는 메시지 예시

```bash
CONFLICT (content): Merge conflict in 파일명
Automatic merge failed; fix conflicts and then commit the result.
```

### 충돌 해결 방법

1. 충돌이 발생한 파일을 열어 충돌 부분을 확인합니다.

```markdown
<<<<<<< HEAD
로컬에서 수정한 내용
=======
원격 저장소에서 수정한 내용
>>>>>>> origin/main
```

2. 충돌 부분을 직접 수정하여 원하는 내용을 선택하거나 병합합니다.

3. 수정 후 파일을 저장하고, 충돌 해결을 완료합니다.

```bash
git add 파일명
git commit -m "충돌 해결: 파일명"
git push origin main
```

---

## 체크리스트로 정리하기

- [ ] Git과 GitHub의 개념 이해하기
- [ ] GitHub 레포지토리 클론하기
- [ ] 매주 수업 자료 업데이트 받기 (`git pull`)
- [ ] 충돌 발생 시 해결 방법 숙지하기

---

## 추가 팁

- GitHub Desktop과 같은 GUI 도구를 사용하면 명령어 없이도 쉽게 Git을 사용할 수 있습니다.
- 주기적으로 커밋(commit)을 하여 변경 사항을 세부적으로 관리하는 습관을 들이는 것이 좋습니다.

---

# 태그: #Git #GitHub #데이터사이언스 #협업 #버전관리 #튜토리얼
