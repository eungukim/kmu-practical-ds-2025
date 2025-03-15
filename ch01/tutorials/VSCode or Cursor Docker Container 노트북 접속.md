# 📌 Docker 컨테이너 환경에서 VSCode 또는 Cursor IDE로 주피터랩 노트북 접속하기

이 튜토리얼은 Docker Compose를 활용하여 데이터 사이언스 분석 환경을 구축하고, VSCode 또는 Cursor IDE의 원격 컨테이너 기능을 통해 주피터랩(JupyterLab) 노트북에 접속하여 분석 작업을 수행하는 방법을 단계별로 안내합니다.

---

## 🚩 사전 준비 사항

- Docker Desktop 설치 및 실행 완료
- Docker Compose 환경 구성 완료 (`docker-compose.yml` 작성 및 실행)
- VSCode 또는 Cursor IDE 설치 완료

## 🚩 Docker Compose 환경 실행하기

터미널에서 Docker Compose 환경을 실행합니다.

```bash
docker-compose up -d
```

## 🚩 VSCode에서 Docker 컨테이너 접속하기

1. VSCode에서 [Remote-Containers 확장 프로그램](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)을 설치합니다.
2. VSCode를 열고 명령 팔레트(`Cmd+Shift+P`)에서 `Remote-Containers: Open Folder in Container`를 선택합니다.
3. 프로젝트 폴더를 선택하면 자동으로 컨테이너 환경이 설정됩니다.

## 🚩 Cursor IDE에서 컨테이너 접속하기

Cursor IDE의 터미널에서 다음 명령어로 컨테이너 내부로 접속합니다.

```bash
docker exec -it [컨테이너 이름 또는 ID] /bin/bash
```

## 🚩 주피터랩 노트북 접속하기

컨테이너가 실행되면 브라우저에서 다음 주소로 접속합니다.

```
http://localhost:8888
```

주피터랩이 실행되며, 분석 작업을 시작할 수 있습니다.

## 🚩 분석 환경 활용 예시

다음은 간단한 데이터 분석 예시입니다.

```python
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 생성
data = {'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40]}
df = pd.DataFrame(data)

# 데이터 시각화
plt.plot(df['x'], df['y'])
plt.title('Docker 환경에서의 데이터 시각화')
plt.xlabel('X축')
plt.ylabel('Y축')
plt.show()
```

## 🚩 환경 종료하기

작업이 끝나면 터미널에서 다음 명령어로 컨테이너를 종료합니다.

```bash
docker-compose down
```

이 튜토리얼을 통해 Docker Compose와 VSCode 또는 Cursor IDE를 활용한 데이터 사이언스 분석 환경을 쉽게 구축하고 활용할 수 있습니다. #Docker #DockerCompose #데이터사이언스 #분석환경
