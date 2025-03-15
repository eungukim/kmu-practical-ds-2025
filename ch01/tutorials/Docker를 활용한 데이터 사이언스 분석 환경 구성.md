
이 튜토리얼은 데이터 사이언스를 처음 접하는 대학원생들이 Docker를 활용하여 크로스플랫폼 분석 환경을 구축하고, 이를 VSCode 또는 Cursor IDE와 연동하여 실무에서 활용 가능한 고급 분석 환경을 구성하는 방법을 안내합니다.

---

## 📌 Docker란 무엇인가?

Docker는 컨테이너 기반의 가상화 기술로, 애플리케이션과 그 실행 환경을 하나의 패키지로 묶어 어디서든 동일한 환경에서 실행할 수 있도록 도와줍니다.

- **컨테이너(Container)**: 독립된 환경에서 애플리케이션을 실행하는 경량화된 가상 환경
- **이미지(Image)**: 컨테이너를 생성하기 위한 템플릿으로, 필요한 모든 소프트웨어와 설정을 포함

---

## 📌 Docker를 사용하는 이유

- **환경 일관성**: 모든 팀원이 동일한 분석 환경을 사용 가능
- **크로스플랫폼 지원**: Windows, macOS, Linux 등 다양한 운영체제에서 동일한 환경 제공
- **빠른 환경 구축**: 복잡한 설치 과정 없이 간단한 명령어로 환경 구축 가능
- **재현성 보장**: 분석 결과를 언제 어디서나 동일하게 재현 가능

---

## 📌 Docker 설치하기

### 1. Docker Desktop 설치

아래 링크에서 본인의 운영체제에 맞는 Docker Desktop을 설치합니다.

- [Docker Desktop 다운로드](https://www.docker.com/products/docker-desktop/)

설치 후 터미널에서 다음 명령어로 설치 확인:

```bash
docker --version
```

---

## 📌 Docker 기본 개념 이해하기

Docker의 핵심 개념을 간단히 정리하면 다음과 같습니다.

| 개념 | 설명 | 비유 |
|---|---|---|
| 이미지(Image) | 컨테이너를 생성하는 템플릿 | 요리 레시피 |
| 컨테이너(Container) | 이미지를 기반으로 실행된 독립 환경 | 레시피로 만든 요리 |
| Dockerfile | 이미지를 만드는 명령어가 담긴 파일 | 레시피가 적힌 종이 |

---

## 📌 데이터 사이언스 분석 환경 이미지 만들기

### 1. Dockerfile 작성하기

프로젝트 폴더에 `Dockerfile`이라는 파일을 생성하고 다음 내용을 작성합니다.

```dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /workspace

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Jupyter Notebook 포트 설정
EXPOSE 8888

# Jupyter Notebook 실행
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### 2. Python 패키지 목록 작성하기

같은 폴더에 `requirements.txt` 파일을 생성하고 필요한 패키지를 작성합니다.

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

### 3. Docker 이미지 빌드하기

터미널에서 다음 명령어로 이미지를 빌드합니다.

```bash
docker build -t ds-env .
```

---

## 📌 Docker 컨테이너 실행하기

빌드한 이미지를 기반으로 컨테이너를 실행합니다.

```bash
docker run -it -p 8888:8888 -v $(pwd):/workspace ds-env
```

- `-p 8888:8888`: 로컬의 8888 포트를 컨테이너의 8888 포트와 연결
- `-v $(pwd):/workspace`: 현재 폴더를 컨테이너의 `/workspace` 폴더와 연결하여 파일 공유

실행 후 터미널에 출력된 URL을 브라우저에서 열어 Jupyter Notebook을 확인합니다.

---

## 📌 VSCode 또는 Cursor IDE와 Docker 연동하기

### 1. VSCode 연동 방법

- VSCode에서 [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 확장 프로그램 설치
- VSCode에서 명령 팔레트(`Ctrl+Shift+P` 또는 `Cmd+Shift+P`)를 열고 `Remote-Containers: Open Folder in Container` 선택
- 프로젝트 폴더를 선택하면 자동으로 Docker 컨테이너 환경에서 VSCode가 실행됩니다.

### 2. Cursor IDE 연동 방법

- Cursor IDE에서 프로젝트 폴더를 열고 터미널을 실행
- Docker 컨테이너를 실행한 상태에서 Cursor IDE의 터미널을 통해 컨테이너 내부로 접속하여 작업 가능

```bash
docker exec -it [컨테이너 ID 또는 이름] /bin/bash
```

---

## 📌 분석 환경 활용 예시

Docker 환경에서 분석을 수행하는 예시입니다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 생성
data = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.rand(10)
})

# 데이터 시각화
plt.plot(data['x'], data['y'])
plt.title('Docker 환경에서의 데이터 시각화')
plt.xlabel('X축')
plt.ylabel('Y축')
plt.show()
```

---

## 📌 프로젝트에 적용하기 위한 체크리스트

- [ ] Docker Desktop 설치 및 실행 확인
- [ ] Dockerfile 및 requirements.txt 작성
- [ ] Docker 이미지 빌드 및 컨테이너 실행
- [ ] VSCode 또는 Cursor IDE와 Docker 연동
- [ ] 분석 환경에서 데이터 분석 및 시각화 수행

---

## 📌 추가 팁 및 주의사항

- Docker 이미지는 가볍게 유지하는 것이 좋습니다. 불필요한 패키지 설치는 피하세요.
- 분석 환경을 자주 변경할 경우 Dockerfile을 수정하고 이미지를 다시 빌드하면 됩니다.
- 컨테이너 종료 시 데이터가 사라지지 않도록 로컬 폴더와 볼륨을 연결하세요.

---

이 튜토리얼을 통해 Docker를 활용한 데이터 사이언스 분석 환경 구축 방법을 익히고, 실무에서 유용하게 활용할 수 있는 자신만의 고급 분석 환경을 구성해 보세요! #Docker #데이터사이언스 #분석환경 #튜토리얼
