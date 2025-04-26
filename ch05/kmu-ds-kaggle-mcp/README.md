# 국민대학교 AI빅데이터 전공 Kaggle MCP 연동 서버

이 프로젝트는 Kaggle API를 활용한 MCP(Model Control Protocol) 서버를 구현합니다. Cursor 에디터와 연동하여 Kaggle의 데이터를 쉽게 탐색하고 관리할 수 있습니다.

## 설치 방법

프로젝트를 설치하려면 다음 명령어를 실행하세요:

```bash
./install.sh
```

이 스크립트는 다음 작업을 수행합니다:

- Python 가상환경 생성
- 필요한 의존성 패키지 설치
- Cursor의 MCP 설정에 서버 정보 추가

## 사용 방법

설치 후 Cursor 에디터에서 MCP 메뉴를 통해 "KMU-DS-Kaggle" 서버를 선택하여 연결할 수 있습니다.

## 주요 기능

- Kaggle 데이터셋 검색 및 다운로드
- 데이터 분석 작업 지원
- Pandas를 활용한 데이터 처리

## 제거 방법

프로젝트를 제거하려면 다음 명령어를 실행하세요:

```bash
./uninstall.sh
```

이 스크립트는 다음 작업을 수행합니다:

- Cursor의 MCP 설정에서 서버 정보 제거
- 실행 중인 서버 프로세스 종료
- 가상환경 삭제

## 의존성

- fastmcp >= 2.0.0
- kaggle
- pandas
