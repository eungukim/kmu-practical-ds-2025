
## 📌 Docker Compose 개념 및 활용

Docker Compose는 여러 Docker 컨테이너를 정의하고 실행하기 위한 도구로, 복잡한 분석 환경을 쉽게 구성하고 관리할 수 있게 도와줍니다.

### Docker Compose란?

Docker Compose는 YAML 파일을 사용하여 애플리케이션의 서비스, 네트워크, 볼륨 등을 정의하고, 단일 명령어로 모든 서비스를 시작할 수 있는 도구입니다.

- **장점**:
  - 여러 컨테이너를 한 번에 관리
  - 설정 파일을 통한 환경 표준화
  - 개발, 테스트, 배포 과정 간소화

### Docker Compose 설치하기

Docker Desktop을 설치했다면 대부분 Docker Compose가 함께 설치됩니다. 설치 여부를 확인하려면:

```bash
docker-compose --version
또는
docker compose --version
```

별도 설치가 필요한 경우 공식 문서를 참조하세요: [Docker Compose 설치 가이드](https://docs.docker.com/compose/install/)

---

## 📌 데이터 사이언스를 위한 Docker Compose 환경 구성

### docker-compose.yml 파일 작성하기

프로젝트 폴더에 `docker-compose.yml` 파일을 생성하고 다음과 같이 작성합니다:

```yaml
services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./:/workspace
    environment:
      - JUPYTER_ENABLE_LAB=yes
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=kmu2025!
      - POSTGRES_USER=student
      - POSTGRES_DB=kmu_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

위 설정은 Jupyter Lab 환경과 PostgreSQL 데이터베이스를 함께 실행하는 환경을 구성합니다.

### Docker Compose 실행하기

다음 명령어로 모든 서비스를 시작합니다:

```bash
docker-compose up
```

백그라운드에서 실행하려면 `-d` 옵션을 추가합니다:

```bash
docker-compose up -d
```

서비스를 중지하려면:

```bash
docker-compose down
```

---

## 📌 분석 환경 문서화 및 공유하기

### 분석 환경 문서화 방법

분석 환경을 효과적으로 문서화하여 재사용성과 협업을 향상시킬 수 있습니다.

#### 1. README.md 작성하기

프로젝트 폴더에 `README.md` 파일을 생성하고 환경에 대한 정보를 기록합니다:

```markdown
# 데이터 사이언스 분석 환경

## 환경 구성 요소
- Python 3.11
- Jupyter Lab
- PostgreSQL 15
- 필수 패키지: pandas, numpy, scikit-learn, matplotlib 등

## 시작 방법
1. Docker와 Docker Compose 설치
2. 이 저장소 클론: `git clone [저장소 URL]`
3. 프로젝트 폴더로 이동: `cd [프로젝트 폴더]`
4. 환경 시작: `docker-compose up -d`
5. 브라우저에서 `http://localhost:8888` 접속 (Jupyter Lab)

## 주요 파일 설명
- `docker-compose.yml`: 서비스 구성 정의
- `Dockerfile`: Python 환경 설정
- `requirements.txt`: Python 패키지 목록
- `notebooks/`: 분석 노트북 저장 폴더
- `data/`: 데이터 파일 저장 폴더

## 데이터베이스 연결 정보
- 호스트: localhost
- 포트: 5432
- 사용자명: datauser
- 비밀번호: mysecretpassword
- 데이터베이스: datasciencedb
```

#### 2. 환경 설정 문서화

프로젝트의 `docs` 폴더에 환경 설정에 대한 상세 문서를 작성합니다:

```markdown
# 데이터 사이언스 환경 설정 가이드

## PostgreSQL 데이터베이스 사용법

### 데이터베이스 접속
```python
import psycopg2

conn = psycopg2.connect(
    host="postgres",
    database="kmu_db",
    user="student",
    password="kmu2025!"
)
```

### 일반적인 작업 흐름

1. 데이터 수집 및 전처리
2. 데이터베이스에 저장
3. 모델 훈련
4. 결과 시각화 및 평가

## 패키지 의존성 관리

새 패키지 추가 시 `requirements.txt`에 추가한 후 다음 명령어 실행:

```bash
docker-compose build
docker-compose up -d
```


---

## 📌 자신만의 분석 환경 템플릿 만들기

개인 또는 팀에 맞는 분석 환경 템플릿을 만들어 재사용할 수 있습니다.

### 1. 템플릿 저장소 생성하기

GitHub 등의 저장소에 기본 템플릿을 만들어 저장합니다.

### 2. 환경 구성요소 정의하기

```markdown
# 내 데이터 사이언스 환경 템플릿

## 기본 구성
- Jupyter Lab + Python 3.11
- PostgreSQL 데이터베이스
- 데이터 시각화 도구 (Plotly, Matplotlib)
- 머신러닝 라이브러리 (scikit-learn, TensorFlow)

## 확장 구성 (선택적)
- MLflow 실험 관리
- MinIO 객체 저장소
- Streamlit 대시보드

## 폴더 구조
```

project/
├── data/               # 데이터 파일
├── notebooks/          # Jupyter 노트북
├── src/                # Python 소스 코드
├── models/             # 학습된 모델
├── config/             # 설정 파일
├── Dockerfile          # 기본 환경 정의
├── docker-compose.yml  # 서비스 구성
└── requirements.txt    # 패키지 목록


### 3. 커스텀 이미지 만들기

자주 사용하는 환경을 Docker Hub에 공유할 수 있습니다.

```bash
# 이미지 빌드
docker build -t myusername/datascience:latest .

# Docker Hub에 푸시
docker push myusername/datascience:latest
```

### 4. devcontainer.json 설정하기 (VSCode/Cursor IDE용)

`.devcontainer/devcontainer.json` 파일을 생성하여 VSCode/Cursor IDE와의 연동을 강화합니다:

```json
{
  "name": "Data Science Environment",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "jupyter",
  "workspaceFolder": "/workspace",
  "settings": {
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
  },
  "extensions": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "njpwerner.autodocstring"
  ]
}
```

---

## 📌 분석 환경 공유 및 협업하기

### Git을 활용한 환경 공유

1. Docker 및 환경 설정 파일을 Git 저장소에 저장
2. `.gitignore` 파일을 사용하여 데이터 파일 등 불필요한 파일 제외
3. README.md에 환경 설정 방법 상세히 기록

### 환경 재현성 보장하기

1. Docker 이미지 버전 명시 (태그 사용)
2. 패키지 버전 고정 (requirements.txt에 정확한 버전 지정)
3. 설정 파일의 변경 사항 문서화

```markdown
## 버전 변경 내역
- 2025-03-01: TensorFlow 2.15 추가, pandas 2.1.0으로 업데이트
- 2025-02-15: PostgreSQL 15로 업그레이드, scikit-learn 1.4.0 적용
```

---

이 튜토리얼을 통해 Docker와 Docker Compose를 활용한 데이터 사이언스 분석 환경 구축 방법과 환경 문서화 및 공유 방법을 익히셨습니다. 이를 바탕으로 자신만의 효율적인 분석 환경을 구성하여 실무에서 활용해 보세요! #Docker #DockerCompose #데이터사이언스 #분석환경 #템플릿
