// 디버그 로그 함수
function debugLog(message) {
    console.log(message);
    const debugElement = document.getElementById('debug');
    if (debugElement) {
        const currentContent = debugElement.innerHTML;
        debugElement.innerHTML = message + '<br>' + currentContent;
    }
}

// 디버그 모드 토글
let debugMode = false;
document.addEventListener('keydown', function (event) {
    // Ctrl+D로 디버그 모드 토글
    if (event.ctrlKey && event.key === 'd') {
        event.preventDefault();
        debugMode = !debugMode;
        document.getElementById('debug').style.display = debugMode ? 'block' : 'none';
        debugLog('디버그 모드: ' + (debugMode ? '켜짐' : '꺼짐'));
    }
});

// 현재 활성화된 단계를 관리하는 함수
function updateActiveStep(stepNumber) {
    // 모든 단계 박스에서 active 클래스 제거
    document.querySelectorAll('.step-info').forEach((box, index) => {
        box.classList.remove('active');
    });

    // 현재 단계에 active 클래스 추가
    if (stepNumber >= 1 && stepNumber <= 4) {
        document.getElementById('step' + stepNumber).classList.add('active');
    }
}

// 전역 설정 변수
let networkData; // 초기화를 위해 먼저 선언
let currentLearningRate = 0.05; // 기본 학습률
let errorHistory = []; // 오차 히스토리 저장 배열
let iterationCount = 0; // 학습 반복 횟수
let initialError = null; // 초기 오차값
let graphInitialized = false; // 그래프 초기화 여부
let cosineGraphInitialized = false; // 코사인 그래프 초기화 여부
let isAutoPlaying = false; // 자동 재생 상태
let autoPlayInterval = null; // 자동 재생 인터벌
let AUTO_PLAY_INTERVAL = 500; // 기본 재생 속도 (0.5초 간격)
let learningRateDisplay; // 학습률 표시 요소
let playSpeedSlider; // 재생 속도 슬라이더
let playSpeedDisplay; // 재생 속도 표시 요소
let svg; // SVG 요소

// 신경망 구조 설정
const networkStructure = {
    input: 1,   // x값 하나만 입력으로 받음
    hidden1: 3,  // 첫 번째 은닉층 뉴런 수 (3개 유지)
    hidden2: 4,  // 두 번째 은닉층 뉴런 수 (5에서 4로 줄임)
    output: 8   // 코사인(x) 곡선의 여러 지점 예측 (10에서 8로 줄임)
};

// SVG 치수 설정
const width = 600; // 900에서 600으로 수정
const height = 600; // 860에서 600으로 수정

// 코사인 함수 데이터 생성 함수
function generateCosineData(count) {
    const data = [];
    for (let i = 0; i < count; i++) {
        // x는 -π에서 π 사이의 값
        const x = (Math.random() * 2 * Math.PI) - Math.PI;
        const y = Math.cos(x);
        data.push({ x, y });
    }
    return data;
}

// 데이터 포인트 범위 생성 (고정된 간격의 x값)
function generateFixedPoints(count) {
    const points = [];
    // -π에서 π까지 균등하게 나눈 지점들
    for (let i = 0; i < count; i++) {
        const x = -Math.PI + (2 * Math.PI * i / (count - 1));
        const y = Math.cos(x);
        points.push({ x, y });
    }
    return points;
}

// 학습 데이터와 출력층 타겟 포인트 (전역 변수로 유지)
const trainingData = generateCosineData(50);
const targetPoints = generateFixedPoints(networkStructure.output);
let currentDataIndex = 0;

// 다음 데이터 포인트 가져오기 (이제 x값만 필요)
function getNextDataPoint() {
    const dataPoint = trainingData[currentDataIndex];
    currentDataIndex = (currentDataIndex + 1) % trainingData.length;
    return dataPoint.x;
}

// 초기화 함수
function initializeNetwork() {
    // 노드 데이터 생성
    const nodes = [];
    let nodeId = 0;

    // 중앙 위치 계산에 사용할 변수
    const centerY = height / 2;
    const verticalMargin = 80; // 상하 여백 (100에서 80으로 수정)

    // 은닉층 노드 간격 계산
    // 각 층의 노드 간격을 더 넓게 설정
    const hidden1Spacing = (height - 2 * verticalMargin) / (networkStructure.hidden1 - 1 || 1);
    const hidden2Spacing = (height - 2 * verticalMargin) / (networkStructure.hidden2 - 1 || 1);
    const outputSpacing = (height - 2 * verticalMargin) / (networkStructure.output - 1 || 1);

    // 입력층 노드 (x값)
    nodes.push({
        id: nodeId++,
        layer: 0,
        type: 'input',
        value: 0,
        x: 100,
        y: centerY, // 중앙 위치에 배치
        label: 'x'
    });

    // 첫 번째 은닉층 노드
    for (let i = 0; i < networkStructure.hidden1; i++) {
        nodes.push({
            id: nodeId++,
            layer: 1,
            type: 'hidden',
            value: 0,
            x: 300,
            // 간격 균등 배치 및 더 넓은 간격 적용
            y: verticalMargin + i * hidden1Spacing,
            label: `h1${i + 1}`
        });
    }

    // 두 번째 은닉층 노드
    for (let i = 0; i < networkStructure.hidden2; i++) {
        nodes.push({
            id: nodeId++,
            layer: 2,
            type: 'hidden',
            value: 0,
            x: 500,
            // 간격 균등 배치 및 더 넓은 간격 적용
            y: verticalMargin + i * hidden2Spacing,
            label: `h2${i + 1}`
        });
    }

    // 출력층 노드 (코사인 예측값, 여러 지점)
    for (let i = 0; i < networkStructure.output; i++) {
        nodes.push({
            id: nodeId++,
            layer: 3,
            type: 'output',
            value: 0,
            x: 700,
            // 간격 균등 배치 및 더 넓은 간격 적용
            y: verticalMargin + i * outputSpacing,
            label: `y${i + 1}`,
            xValue: targetPoints[i].x // 이 노드가 담당하는 x값 저장
        });
    }

    // 타겟 노드들 (실제 코사인값, 여러 지점)
    for (let i = 0; i < networkStructure.output; i++) {
        // 타겟 노드들을 출력 노드 옆에 일렬로 배치
        const outputNodeIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + i;
        const outputNode = nodes[outputNodeIndex];

        nodes.push({
            id: nodeId++,
            layer: 4,
            type: 'target',
            value: 0,
            x: 900,
            y: outputNode.y, // 출력 노드와 같은 y좌표
            label: `t${i + 1}`,
            xValue: targetPoints[i].x // 이 노드가 담당하는 x값 저장
        });
    }

    // 손실 노드 (하단 중앙에 배치)
    nodes.push({
        id: nodeId++,
        layer: 5,
        type: 'loss',
        value: 0,
        x: 800,
        y: height - 50, // 하단에 충분한 공간 확보 (100에서 50으로 변경)
        label: 'loss'
    });

    // 엣지 데이터 생성
    const links = [];
    let weightId = 0;

    // 입력층-첫번째 은닉층 연결
    for (let i = 0; i < networkStructure.input; i++) {
        for (let j = 0; j < networkStructure.hidden1; j++) {
            links.push({
                source: i,
                target: networkStructure.input + j,
                value: 0,
                weight: (Math.random() * 2 - 1).toFixed(4), // -1~1 사이의 가중치
                gradient: 0,
                type: 'forward',
                weightId: 'w' + (weightId++),
                sourceLabel: nodes[i].label,
                targetLabel: nodes[networkStructure.input + j].label
            });
        }
    }

    // 은닉층-은닉층 연결
    for (let i = 0; i < networkStructure.hidden1; i++) {
        for (let j = 0; j < networkStructure.hidden2; j++) {
            links.push({
                source: networkStructure.input + i,
                target: networkStructure.input + networkStructure.hidden1 + j,
                value: 0,
                weight: (Math.random() * 2 - 1).toFixed(4), // -1~1 사이의 가중치
                gradient: 0,
                type: 'forward',
                weightId: 'w' + (weightId++),
                sourceLabel: nodes[networkStructure.input + i].label,
                targetLabel: nodes[networkStructure.input + networkStructure.hidden1 + j].label
            });
        }
    }

    // 은닉층-출력층 연결
    for (let i = 0; i < networkStructure.hidden2; i++) {
        for (let j = 0; j < networkStructure.output; j++) {
            links.push({
                source: networkStructure.input + networkStructure.hidden1 + i,
                target: networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + j,
                value: 0,
                weight: (Math.random() * 2 - 1).toFixed(4), // -1~1 사이의 가중치
                gradient: 0,
                type: 'forward',
                weightId: 'w' + (weightId++),
                sourceLabel: nodes[networkStructure.input + networkStructure.hidden1 + i].label,
                targetLabel: nodes[networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + j].label
            });
        }
    }

    // 출력층-타겟 연결
    for (let i = 0; i < networkStructure.output; i++) {
        const outputNodeIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + i;
        const targetNodeIndex = outputNodeIndex + networkStructure.output;

        links.push({
            source: outputNodeIndex,
            target: targetNodeIndex,
            value: 0,
            weight: 0,
            type: 'comparison',
            sourceLabel: nodes[outputNodeIndex].label,
            targetLabel: nodes[targetNodeIndex].label
        });
    }

    // 모든 출력-손실 및 타겟-손실 연결
    const lossNodeIndex = nodeId - 1;

    // 각 출력 노드와 손실 노드 연결
    for (let i = 0; i < networkStructure.output; i++) {
        const outputNodeIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + i;

        links.push({
            source: outputNodeIndex,
            target: lossNodeIndex,
            value: 0,
            weight: 0,
            type: 'loss',
            sourceLabel: nodes[outputNodeIndex].label,
            targetLabel: nodes[lossNodeIndex].label
        });
    }

    // 각 타겟 노드와 손실 노드 연결
    for (let i = 0; i < networkStructure.output; i++) {
        const targetNodeIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + networkStructure.output + i;

        links.push({
            source: targetNodeIndex,
            target: lossNodeIndex,
            value: 0,
            weight: 0,
            type: 'loss',
            sourceLabel: nodes[targetNodeIndex].label,
            targetLabel: nodes[lossNodeIndex].label
        });
    }

    // 테이블 초기화 부분 제거
    // updateWeightsTable();

    return { nodes, links };
}

// 가중치 테이블 업데이트 함수
function updateWeightsTable() {
    const tableBody = document.querySelector('#weightsTable tbody');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    // 가중치 정보만 필터링
    const weightLinks = networkData.links.filter(link => link.type === 'forward');

    // 가중치를 타입별로 그룹화
    const inputToHidden1 = weightLinks.filter(link =>
        networkData.nodes[link.source].layer === 0 && networkData.nodes[link.target].layer === 1
    );

    const hidden1ToHidden2 = weightLinks.filter(link =>
        networkData.nodes[link.source].layer === 1 && networkData.nodes[link.target].layer === 2
    );

    const hidden2ToOutput = weightLinks.filter(link =>
        networkData.nodes[link.source].layer === 2 && networkData.nodes[link.target].layer === 3
    );

    // 선택적으로 일부만 표시
    let selectedWeights = [];

    // 입력->은닉1 가중치는 모두 표시 (소수이므로)
    selectedWeights = selectedWeights.concat(inputToHidden1);

    // 은닉1->은닉2 가중치는 첫 번째 은닉 노드의 가중치만 표시
    const h1ToH2Sample = hidden1ToHidden2.filter(link =>
        networkData.nodes[link.source].label === 'h11'
    );
    selectedWeights = selectedWeights.concat(h1ToH2Sample);

    // 은닉2->출력 가중치는 첫 번째 은닉 노드의 가중치만 표시
    const h2ToOutSample = hidden2ToOutput.filter(link =>
        networkData.nodes[link.source].label === 'h21'
    );
    selectedWeights = selectedWeights.concat(h2ToOutSample);

    // 행 추가
    selectedWeights.forEach(link => {
        const tr = document.createElement('tr');

        // 가중치 ID
        const idCell = document.createElement('td');
        idCell.textContent = link.weightId;
        idCell.className = 'weight-id';
        tr.appendChild(idCell);

        // 연결 정보
        const connectionCell = document.createElement('td');
        connectionCell.textContent = `${link.sourceLabel} → ${link.targetLabel}`;
        connectionCell.className = 'weight-connection';
        tr.appendChild(connectionCell);

        // 현재 값
        const valueCell = document.createElement('td');
        valueCell.textContent = parseFloat(link.weight).toFixed(4);
        tr.appendChild(valueCell);

        // 이전 값
        const oldValueCell = document.createElement('td');
        oldValueCell.textContent = link.oldWeight ? parseFloat(link.oldWeight).toFixed(4) : '-';
        tr.appendChild(oldValueCell);

        // 변화량
        const changeCell = document.createElement('td');
        if (link.oldWeight) {
            const change = (parseFloat(link.weight) - parseFloat(link.oldWeight)).toFixed(4);
            changeCell.textContent = change;
            if (parseFloat(change) > 0) {
                changeCell.className = 'weight-change-positive';
            } else if (parseFloat(change) < 0) {
                changeCell.className = 'weight-change-negative';
            }
        } else {
            changeCell.textContent = '-';
        }
        tr.appendChild(changeCell);

        // 그래디언트
        const gradientCell = document.createElement('td');
        gradientCell.textContent = link.gradient ? parseFloat(link.gradient).toFixed(4) : '-';
        gradientCell.className = 'weight-gradient';
        tr.appendChild(gradientCell);

        // 가중치가 변경되었으면 행 하이라이트
        if (link.oldWeight && link.oldWeight !== link.weight) {
            tr.classList.add('updated');
        }

        tableBody.appendChild(tr);
    });
}

// 시각화 업데이트 함수
function updateVisualization() {
    // 노드 업데이트
    svg.selectAll('g.node text')
        .filter(function (d) {
            return d3.select(this).attr('dy') === '0.35em';
        })
        .text(d => {
            if (d.type === 'loss') return d.value.toFixed(4);
            if (d.type === 'target') return d.value;
            return d.value;
        });

    // 링크 텍스트 업데이트 - 가중치 ID만 간단히 표시
    svg.selectAll('g text')
        .filter(function () {
            return !d3.select(this.parentNode).classed('node');
        })
        .text(d => {
            if (d.type === 'forward') {
                return d.weightId; // 가중치 ID만 표시
            }
            return '';
        })
        .style('fill', d => d.oldWeight && d.oldWeight !== d.weight ? '#e74c3c' : '#333');

    // 가중치 테이블 업데이트
    updateWeightsTable();
}

// 시그모이드 함수
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// 순전파 함수 (코사인 함수 예측)
function runForwardPass() {
    debugLog("순전파 함수 실행");

    // 현재 단계를 활성화
    updateActiveStep(1);

    // 버튼 상태 업데이트
    document.getElementById('forwardBtn').disabled = true;
    document.getElementById('lossBtn').disabled = false;

    // 입력 데이터 포인트 가져오기 (x값)
    const inputX = getNextDataPoint();

    // 입력값 설정 (x값)
    networkData.nodes[0].value = inputX.toFixed(4);
    debugLog(`입력 x: ${inputX.toFixed(4)}`);

    // 타겟값 설정 (실제 코사인값들)
    const outputStartIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2;
    const targetStartIndex = outputStartIndex + networkStructure.output;

    // 각 타겟 노드에 해당하는 실제 코사인 값 설정
    for (let i = 0; i < networkStructure.output; i++) {
        const targetNode = networkData.nodes[targetStartIndex + i];
        // 이 노드가 담당하는 x값에 대한 코사인 값 계산
        targetNode.value = Math.cos(targetNode.xValue).toFixed(4);
        debugLog(`타겟 ${i + 1} (x=${targetNode.xValue.toFixed(2)}): 코사인 = ${targetNode.value}`);
    }

    // 상태 텍스트 업데이트 제거 - 단계별 박스로 대체

    try {
        // 첫 번째 은닉층 계산
        let hiddenStart = networkStructure.input;
        let hiddenEnd = hiddenStart + networkStructure.hidden1;

        for (let i = hiddenStart; i < hiddenEnd; i++) {
            let sum = 0;

            // 입력층으로부터의 가중합 계산
            for (let j = 0; j < networkStructure.input; j++) {
                const linkIndex = networkData.links.findIndex(
                    link => link.source === j && link.target === i
                );

                if (linkIndex !== -1) {
                    const inputValue = parseFloat(networkData.nodes[j].value);
                    const weight = parseFloat(networkData.links[linkIndex].weight);
                    sum += inputValue * weight;
                    debugLog(`입력 ${j} -> 은닉 ${i}: ${inputValue} * ${weight} = ${inputValue * weight}`);
                }
            }

            // 시그모이드 활성화 함수 적용
            const activationValue = sigmoid(sum);
            networkData.nodes[i].value = activationValue.toFixed(4);
            debugLog(`첫 번째 은닉층 ${i} 노드 값: ${networkData.nodes[i].value}`);
        }

        // 두 번째 은닉층 계산
        hiddenStart = networkStructure.input + networkStructure.hidden1;
        hiddenEnd = hiddenStart + networkStructure.hidden2;

        for (let i = hiddenStart; i < hiddenEnd; i++) {
            let sum = 0;

            // 은닉층으로부터의 가중합 계산
            for (let j = hiddenStart - networkStructure.hidden1; j < hiddenStart; j++) {
                const linkIndex = networkData.links.findIndex(
                    link => link.source === j && link.target === i
                );

                if (linkIndex !== -1) {
                    const hiddenValue = parseFloat(networkData.nodes[j].value);
                    const weight = parseFloat(networkData.links[linkIndex].weight);
                    sum += hiddenValue * weight;
                    debugLog(`은닉 ${j} -> 은닉 ${i}: ${hiddenValue} * ${weight} = ${hiddenValue * weight}`);
                }
            }

            // 시그모이드 활성화 함수 적용
            const activationValue = sigmoid(sum);
            networkData.nodes[i].value = activationValue.toFixed(4);
            debugLog(`두 번째 은닉층 ${i} 노드 값: ${networkData.nodes[i].value}`);
        }

        // 출력층 계산 (여러 노드)
        const outputStart = hiddenEnd;
        const outputEnd = outputStart + networkStructure.output;

        for (let i = outputStart; i < outputEnd; i++) {
            let sum = 0;

            // 은닉층으로부터의 가중합 계산
            for (let j = hiddenStart; j < hiddenEnd; j++) {
                const linkIndex = networkData.links.findIndex(
                    link => link.source === j && link.target === i
                );

                if (linkIndex !== -1) {
                    const hiddenValue = parseFloat(networkData.nodes[j].value);
                    const weight = parseFloat(networkData.links[linkIndex].weight);
                    sum += hiddenValue * weight;
                    debugLog(`은닉 ${j} -> 출력 ${i}: ${hiddenValue} * ${weight} = ${hiddenValue * weight}`);
                }
            }

            // 출력층은 tanh 활성화 함수 사용 (코사인값은 -1~1 범위)
            const outputValue = Math.tanh(sum);
            networkData.nodes[i].value = outputValue.toFixed(4);

            // 이 출력 노드가 담당하는 x값 표시
            const xValue = networkData.nodes[i].xValue;
            debugLog(`출력층 ${i} 노드 값 (x=${xValue.toFixed(2)}): ${outputValue.toFixed(4)}`);
        }

        // 시각화 업데이트
        updateVisualization();

        // 순전파 애니메이션
        animateForwardPass();

        // 코사인 그래프에 현재 예측값 추가
        updateCosineGraphWithCurve();

        debugLog("순전파 완료");
    } catch (error) {
        debugLog("순전파 함수 오류: " + error.message);
        console.error("순전파 함수 오류:", error);
    }
}

// 코사인 그래프에 예측 곡선 추가
function updateCosineGraphWithCurve() {
    const outputStartIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2;
    const targetStartIndex = outputStartIndex + networkStructure.output;

    // 예측값과 실제값 배열 생성
    const predictions = [];
    const targets = [];

    for (let i = 0; i < networkStructure.output; i++) {
        const outputNode = networkData.nodes[outputStartIndex + i];
        const targetNode = networkData.nodes[targetStartIndex + i];
        const x = outputNode.xValue;

        predictions.push({
            x: x,
            y: parseFloat(outputNode.value)
        });

        targets.push({
            x: x,
            y: parseFloat(targetNode.value)
        });
    }

    // 예측값을 x축 기준으로 정렬
    predictions.sort((a, b) => a.x - b.x);
    targets.sort((a, b) => a.x - b.x);

    // 그래프 업데이트
    drawPredictionCurve(predictions, targets);
}

// 손실 계산 함수 (여러 출력 노드의 평균 손실)
function calculateLoss() {
    debugLog("손실 계산 함수 실행");

    // 현재 단계를 활성화
    updateActiveStep(2);

    // 버튼 상태 업데이트
    document.getElementById('lossBtn').disabled = true;
    document.getElementById('backpropBtn').disabled = false;

    // 상태 텍스트 업데이트 제거 - 단계별 박스로 대체

    try {
        let totalLoss = 0;
        const outputStartIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2;
        const targetStartIndex = outputStartIndex + networkStructure.output;

        // 각 출력 노드에 대한 손실 계산
        for (let i = 0; i < networkStructure.output; i++) {
            const outputValue = parseFloat(networkData.nodes[outputStartIndex + i].value);
            const targetValue = parseFloat(networkData.nodes[targetStartIndex + i].value);

            // 제곱 오차 계산
            const nodeLoss = Math.pow(outputValue - targetValue, 2);
            totalLoss += nodeLoss;

            debugLog(`출력 ${i + 1}: 예측=${outputValue}, 목표=${targetValue}, 손실=${nodeLoss.toFixed(4)}`);
        }

        // 평균 손실 계산
        const avgLoss = totalLoss / networkStructure.output;
        debugLog(`평균 손실: ${avgLoss.toFixed(4)}`);

        // 손실 노드 업데이트
        const lossNodeIndex = networkData.nodes.length - 1;
        networkData.nodes[lossNodeIndex].value = avgLoss;

        // 오차 표시 업데이트
        document.getElementById('errorDisplay').textContent = `오차: ${avgLoss.toFixed(4)}`;
        document.getElementById('errorDisplay').style.opacity = 1;

        // 시각화 업데이트
        updateVisualization();

        // 손실 계산 애니메이션
        animateLossCalculation();

        debugLog("손실 계산 완료");
    } catch (error) {
        debugLog("손실 계산 함수 오류: " + error.message);
        console.error("손실 계산 함수 오류:", error);
    }
}

// 순전파 애니메이션
function animateForwardPass() {
    // 입력층 -> 은닉층 애니메이션
    for (let i = 0; i < networkStructure.input; i++) {
        // 입력 노드 강조
        svg.selectAll('circle')
            .filter((d, j) => j === i)
            .classed('pulse', true)
            .transition()
            .duration(500)
            .attr('r', 35)
            .transition()
            .duration(500)
            .attr('r', 30)
            .on('end', function () {
                d3.select(this).classed('pulse', false);
            });

        // 관련 링크 강조
        for (let j = 0; j < networkStructure.hidden1; j++) {
            const linkIndex = networkData.links.findIndex(
                link => link.source === i && link.target === networkStructure.input + j
            );

            if (linkIndex !== -1) {
                svg.selectAll('line')
                    .filter((d, j) => j === linkIndex)
                    .transition()
                    .duration(500)
                    .attr('stroke', '#e74c3c') // 빨간색으로 변경
                    .attr('stroke-width', 3)
                    .attr('marker-end', 'url(#arrowhead-forward)') // 순전파 화살표 추가
                    .transition()
                    .duration(500)
                    .attr('stroke', '#999')
                    .attr('stroke-width', 2)
                    .attr('marker-end', 'url(#arrowhead)'); // 원래 화살표로 복원
            }
        }
    }

    // 은닉층 -> 은닉층 애니메이션
    for (let i = 0; i < networkStructure.hidden1; i++) {
        // 은닉 노드 강조
        svg.selectAll('circle')
            .filter((d, j) => j === networkStructure.input + i)
            .classed('pulse', true)
            .transition()
            .duration(500)
            .attr('r', 35)
            .transition()
            .duration(500)
            .attr('r', 30)
            .on('end', function () {
                d3.select(this).classed('pulse', false);
            });

        // 관련 링크 강조
        for (let j = 0; j < networkStructure.hidden2; j++) {
            const linkIndex = networkData.links.findIndex(
                link => link.source === networkStructure.input + i &&
                    link.target === networkStructure.input + networkStructure.hidden1 + j
            );

            if (linkIndex !== -1) {
                svg.selectAll('line')
                    .filter((d, j) => j === linkIndex)
                    .transition()
                    .duration(500)
                    .attr('stroke', '#e74c3c') // 빨간색으로 변경
                    .attr('stroke-width', 3)
                    .attr('marker-end', 'url(#arrowhead-forward)') // 순전파 화살표 추가
                    .transition()
                    .duration(500)
                    .attr('stroke', '#999')
                    .attr('stroke-width', 2)
                    .attr('marker-end', 'url(#arrowhead)'); // 원래 화살표로 복원
            }
        }
    }

    // 은닉층 -> 출력층 애니메이션 (지연 적용)
    setTimeout(() => {
        for (let i = 0; i < networkStructure.hidden2; i++) {
            // 은닉 노드 강조
            svg.selectAll('circle')
                .filter((d, j) => j === networkStructure.input + networkStructure.hidden1 + i)
                .classed('pulse', true)
                .transition()
                .duration(500)
                .attr('r', 35)
                .transition()
                .duration(500)
                .attr('r', 30)
                .on('end', function () {
                    d3.select(this).classed('pulse', false);
                });

            // 관련 출력층 링크 강조
            for (let j = 0; j < networkStructure.output; j++) {
                const outputIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + j;
                const linkIndex = networkData.links.findIndex(
                    link => link.source === networkStructure.input + networkStructure.hidden1 + i &&
                        link.target === outputIndex
                );

                if (linkIndex !== -1) {
                    svg.selectAll('line')
                        .filter((d, j) => j === linkIndex)
                        .transition()
                        .duration(500)
                        .attr('stroke', '#e74c3c') // 빨간색으로 변경
                        .attr('stroke-width', 3)
                        .attr('marker-end', 'url(#arrowhead-forward)') // 순전파 화살표 추가
                        .transition()
                        .duration(500)
                        .attr('stroke', '#999')
                        .attr('stroke-width', 2)
                        .attr('marker-end', 'url(#arrowhead)'); // 원래 화살표로 복원
                }
            }
        }
    }, 1600);
}

// 가중치 업데이트 함수 수정 - 여러 출력 노드 지원
function updateWeights() {
    debugLog("가중치 업데이트 함수 실행");

    // 현재 단계를 활성화
    updateActiveStep(4);

    // 버튼 상태 업데이트
    document.getElementById('updateBtn').disabled = true;
    document.getElementById('forwardBtn').disabled = false;

    // 상태 텍스트 업데이트 제거 - 단계별 박스로 대체

    try {
        // 사용자가 선택한 학습률 사용
        const learningRate = currentLearningRate;
        debugLog(`학습률: ${learningRate.toFixed(3)}`);

        // 모든 가중치 업데이트
        for (let i = 0; i < networkData.links.length; i++) {
            if (networkData.links[i].type === 'forward' && networkData.links[i].gradient) {
                const oldWeight = parseFloat(networkData.links[i].weight);
                const gradient = parseFloat(networkData.links[i].gradient);

                // 기존 가중치 저장 (시각화 용)
                networkData.links[i].oldWeight = oldWeight.toFixed(4);

                // 가중치 업데이트 (w = w - η * ∇w)
                const newWeight = oldWeight - learningRate * gradient;
                networkData.links[i].weight = newWeight.toFixed(4);

                // 디버그 정보 표시
                const source = networkData.nodes[networkData.links[i].source].type;
                const target = networkData.nodes[networkData.links[i].target].type;
                debugLog(`가중치 업데이트 ${source}→${target}: ${oldWeight.toFixed(4)} → ${newWeight.toFixed(4)} (변화량: ${(newWeight - oldWeight).toFixed(4)})`);

                // 가중치 변화 정보를 링크에 저장
                networkData.links[i].weightChange = (newWeight - oldWeight).toFixed(4);

                // 그래디언트 초기화
                networkData.links[i].gradient = 0;
            }
        }

        // 노드 그래디언트 정보도 초기화
        networkData.nodes.forEach(node => {
            delete node.gradient;
        });

        // 현재 출력값과 목표값 간의 차이 표시
        const outputIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2;
        const targetIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + networkStructure.output;
        const outputValue = parseFloat(networkData.nodes[outputIndex].value);
        const targetValue = parseFloat(networkData.nodes[targetIndex].value);

        // 현재 오차 계산 및 기록
        const lossNodeIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2 + networkStructure.output + 1;
        const currentLoss = parseFloat(networkData.nodes[lossNodeIndex].value);

        // 오차 히스토리 업데이트
        errorHistory.push(currentLoss);
        iterationCount++;

        // 코사인 그래프에 현재 데이터 포인트 추가
        const inputValue = parseFloat(networkData.nodes[0].value);
        updateCosineGraph(inputValue, targetValue, outputValue);

        debugLog(`현재 출력값: ${outputValue.toFixed(4)}, 목표값: ${targetValue.toFixed(4)}, 차이: ${Math.abs(outputValue - targetValue).toFixed(4)}`);
        debugLog(`오차 히스토리 업데이트: ${currentLoss.toFixed(4)} (반복 ${iterationCount})`);

        // 시각화 업데이트
        updateVisualization();

        // 가중치 업데이트 애니메이션
        animateWeightUpdate();

        debugLog("가중치 업데이트 완료");
    } catch (error) {
        debugLog("가중치 업데이트 함수 오류: " + error.message);
        console.error("가중치 업데이트 함수 오류:", error);
    }
}

// 가중치 업데이트 애니메이션 개선
function animateWeightUpdate() {
    // 가중치 업데이트 애니메이션
    svg.selectAll('line')
        .filter(d => d.type === 'forward' && d.oldWeight && d.oldWeight !== d.weight)
        .transition()
        .duration(500)
        .attr('stroke', '#f39c12')
        .attr('stroke-width', 3)
        .transition()
        .duration(500)
        .attr('stroke', '#999')
        .attr('stroke-width', 2);

    // 링크 텍스트 깜빡임
    svg.selectAll('text')
        .filter(d => d.type === 'forward' && d.oldWeight && d.oldWeight !== d.weight)
        .transition()
        .duration(300)
        .style('font-weight', 'bold')
        .style('fill', '#f39c12')
        .transition()
        .duration(300)
        .style('font-weight', 'normal')
        .style('fill', '#e74c3c');

    // 다음 순전파 후에는 변화 기록 초기화
    setTimeout(() => {
        networkData.links.forEach(link => {
            delete link.oldWeight;
            delete link.weightChange;
        });
    }, 2000);
}

// 네트워크 리셋 함수 수정 - 그래프 관련 처리 추가
function resetNetwork() {
    debugLog("네트워크 리셋");

    // 모든 단계 비활성화
    updateActiveStep(0);

    // 버튼 상태 초기화
    document.getElementById('forwardBtn').disabled = false;
    document.getElementById('lossBtn').disabled = true;
    document.getElementById('backpropBtn').disabled = true;
    document.getElementById('updateBtn').disabled = true;

    // 상태 텍스트 초기화 제거 - 단계별 박스로 대체

    // 오차 표시 숨기기
    document.getElementById('errorDisplay').style.opacity = 0;

    // 네트워크 데이터 초기화
    networkData = initializeNetwork();

    // SVG 요소 모두 제거 후 다시 생성
    d3.select('#networkViz svg').remove();

    // SVG 재생성 및 시각화
    const svg = d3.select('#networkViz')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', '0 0 1000 700') // 800에서 700으로 변경
        .attr('preserveAspectRatio', 'xMidYMid meet');

    // 마커 정의 (기본 화살표)
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 15)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#999')
        .style('stroke', 'none');

    // 순전파용 마커 정의 (빨간 화살표, 순방향)
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead-forward')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 15)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#e74c3c')
        .style('stroke', 'none');

    // 역전파용 마커 정의 (빨간 화살표, 역방향)
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead-back')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 15)
        .attr('refY', 0)
        .attr('orient', 'auto-start-reverse') // 역방향 화살표
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#e74c3c')
        .style('stroke', 'none');

    // 링크 생성
    const link = svg.append('g')
        .selectAll('line')
        .data(networkData.links)
        .enter().append('line')
        .attr('class', 'link')
        .attr('x1', d => networkData.nodes[d.source].x)
        .attr('y1', d => networkData.nodes[d.source].y)
        .attr('x2', d => networkData.nodes[d.target].x)
        .attr('y2', d => networkData.nodes[d.target].y)
        .attr('stroke-width', d => d.type === 'forward' ? 2 : 1)
        .attr('stroke', d => {
            if (d.type === 'loss') return '#e74c3c';
            if (d.type === 'comparison') return '#f39c12';
            return '#999';
        })
        .attr('marker-end', d => d.type === 'forward' ? 'url(#arrowhead)' : null);

    // 링크 텍스트 생성
    const linkText = svg.append('g')
        .selectAll('text')
        .data(networkData.links.filter(d => d.type === 'forward'))
        .enter().append('text')
        .attr('x', d => (networkData.nodes[d.source].x + networkData.nodes[d.target].x) / 2)
        .attr('y', d => (networkData.nodes[d.source].y + networkData.nodes[d.target].y) / 2 - 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .text(d => d.weightId); // 가중치 ID만 표시

    // 노드 그룹 생성
    const node = svg.append('g')
        .selectAll('g')
        .data(networkData.nodes)
        .enter().append('g')
        .attr('transform', d => `translate(${d.x}, ${d.y})`)
        .attr('class', 'node');

    // 노드 원 생성
    node.append('circle')
        .attr('r', d => d.type === 'loss' ? 35 : 30)
        .attr('fill', d => {
            if (d.type === 'input') return '#3498db';
            if (d.type === 'hidden') {
                // 두 번째 은닉층인지 확인
                if (d.layer === 2) return '#8e44ad';
                return '#9b59b6';
            }
            if (d.type === 'output') return '#2ecc71';
            if (d.type === 'target') return '#f39c12';
            return '#e74c3c'; // loss
        });

    // 노드 텍스트 생성 (값만 표시)
    node.append('text')
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .attr('font-size', '12px')
        .text(d => {
            if (d.type === 'loss') return d.value.toFixed(4);
            if (d.type === 'target') return d.value;
            return d.value;
        });

    // 노드 라벨 추가 (노드 위에 노드 이름만 표시)
    node.append('text')
        .attr('dy', '-20px')
        .attr('text-anchor', 'middle')
        .attr('class', 'node-label')
        .attr('fill', '#333')
        .text(d => d.label || '');

    // 레이어 그룹 타이틀 추가
    // 입력층 타이틀
    svg.append('text')
        .attr('x', 100)
        .attr('y', 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#000000')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text('입력층');

    // 첫 번째 은닉층 타이틀
    svg.append('text')
        .attr('x', 300)
        .attr('y', 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#000000')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text('은닉층 1');

    // 두 번째 은닉층 타이틀
    svg.append('text')
        .attr('x', 500)
        .attr('y', 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#000000')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text('은닉층 2');

    // 출력층 타이틀
    svg.append('text')
        .attr('x', 700)
        .attr('y', 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#000000')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text('출력층');

    // 타겟 타이틀
    svg.append('text')
        .attr('x', 900)
        .attr('y', 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#000000')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text('실제값');

    // 손실 타이틀은 손실 노드 자체에 표시되므로 별도 표시 안함

    // 툴팁 생성
    const tooltip = d3.select('#tooltip');

    node.on('mouseover', function (event, d) {
        let content = '';

        if (d.type === 'loss') {
            content = `손실값: ${d.value.toFixed(4)}`;
        } else if (d.type === 'target') {
            content = `목표값: ${d.value}`;
        } else {
            content = `노드값: ${d.value}`;
        }

        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .style('opacity', 1);
    })
        .on('mouseout', function () {
            tooltip.style('opacity', 0);
        });

    // 오차 표시 숨기기
    document.getElementById('errorDisplay').style.opacity = 0;

    // 그래프 관련 상태 초기화 (데이터는 유지)
    initialError = null;
    const initialErrorElement = document.getElementById('initialError');
    const currentErrorElement = document.getElementById('currentError');
    if (initialErrorElement) initialErrorElement.textContent = '-';
    if (currentErrorElement) currentErrorElement.textContent = '-';

    debugLog("네트워크 리셋 완료");
}

// 문서 로드 후 실행할 초기화 함수
document.addEventListener('DOMContentLoaded', function () {
    // UI 요소 초기화
    learningRateDisplay = document.getElementById('learningRateValue');
    playSpeedSlider = document.getElementById('playSpeedSlider');
    playSpeedDisplay = document.getElementById('playSpeedValue');

    // 기본 학습률 버튼 활성화
    const defaultRateButton = document.querySelector(`.rate-btn[data-rate="${currentLearningRate}"]`);
    if (defaultRateButton) {
        defaultRateButton.classList.add('active');
    }

    // 학습률 표시 초기화
    if (learningRateDisplay) {
        learningRateDisplay.textContent = currentLearningRate.toFixed(3);
    }

    // 기본 재생 속도 버튼 활성화
    const defaultSpeedButton = document.querySelector(`.speed-btn[data-speed="${AUTO_PLAY_INTERVAL}"]`);
    if (defaultSpeedButton) {
        defaultSpeedButton.classList.add('active');
    }

    // 재생 속도 표시 초기화
    if (playSpeedDisplay) {
        playSpeedDisplay.textContent = AUTO_PLAY_INTERVAL;
    }

    // 코사인 그래프 초기화
    initCosineGraph();

    // 모든 단계 비활성화로 시작
    updateActiveStep(0);

    // 그래프 초기화 버튼 이벤트 설정
    document.getElementById('clearGraphBtn').addEventListener('click', function () {
        // 그래프 초기화 코드
        initCosineGraph();
        cosineGraphInitialized = true;
    });

    // 재생 속도 슬라이더 이벤트 설정
    if (playSpeedSlider) {
        playSpeedSlider.addEventListener('input', function () {
            const speed = parseInt(this.value);
            updatePlaySpeed(speed);
            playSpeedDisplay.textContent = speed;

            // 활성 버튼 스타일 업데이트
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.remove('active');
                if (parseInt(btn.dataset.speed) === speed) {
                    btn.classList.add('active');
                }
            });
        });
    }

    // 재생 속도 프리셋 버튼 이벤트 설정
    document.querySelectorAll('.speed-btn').forEach(button => {
        button.addEventListener('click', function () {
            const speed = parseInt(this.dataset.speed);
            updatePlaySpeed(speed);
            playSpeedSlider.value = speed;
            playSpeedDisplay.textContent = speed;

            // 활성 버튼 스타일 업데이트
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
        });
    });

    // 학습률 슬라이더 이벤트 설정
    const learningRateSlider = document.getElementById('learningRateSlider');

    learningRateSlider.addEventListener('input', function () {
        const value = parseFloat(this.value);
        currentLearningRate = value;
        learningRateDisplay.textContent = value.toFixed(3);
        debugLog(`학습률 변경: ${value.toFixed(3)}`);

        // 활성 버튼 스타일 업데이트
        document.querySelectorAll('.rate-btn').forEach(btn => {
            btn.classList.remove('active');
            if (parseFloat(btn.dataset.rate) === value) {
                btn.classList.add('active');
            }
        });
    });

    // 학습률 프리셋 버튼 이벤트 설정
    document.querySelectorAll('.rate-btn').forEach(button => {
        button.addEventListener('click', function () {
            const rate = parseFloat(this.dataset.rate);
            currentLearningRate = rate;
            learningRateSlider.value = rate;
            learningRateDisplay.textContent = rate.toFixed(3);
            debugLog(`학습률 변경: ${rate.toFixed(3)}`);

            // 활성 버튼 스타일 업데이트
            document.querySelectorAll('.rate-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
        });
    });

    // 재생 및 일시정지 버튼 이벤트 설정
    document.getElementById('playBtn').addEventListener('click', () => toggleAutoPlay(true));
    document.getElementById('pauseBtn').addEventListener('click', () => toggleAutoPlay(false));

    // 리셋 버튼 이벤트 업데이트
    document.getElementById('resetBtn').addEventListener('click', function () {
        resetNetwork();
        stopAutoPlay(); // 자동 재생 중이라면 중지
    });

    // 기존 버튼 이벤트 설정
    document.getElementById('forwardBtn').addEventListener('click', runForwardPass);
    document.getElementById('lossBtn').addEventListener('click', calculateLoss);
    document.getElementById('backpropBtn').addEventListener('click', runBackprop);
    document.getElementById('updateBtn').addEventListener('click', updateWeights);

    debugLog(`페이지 로드 완료. 기본 학습률: ${currentLearningRate}, 기본 재생 속도: ${AUTO_PLAY_INTERVAL}ms`);
});

// 오차 그래프 초기화 함수 - 제거됨
function initErrorGraph() {
    // 오차 그래프가 제거되었으므로 아무 작업도 하지 않음
    graphInitialized = false;
    debugLog('오차 그래프 기능이 비활성화되었습니다');
}

// 오차 그래프 업데이트 함수 - 제거됨
function updateErrorGraph() {
    // 오차 그래프가 제거되었으므로 아무 작업도 하지 않음
}

// 코사인 함수 그래프 초기화
function initCosineGraph() {
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const container = document.getElementById('cosineGraph');

    if (!container) return;

    d3.select('#cosineGraph svg').remove();

    // 컨테이너 크기에 맞게 조정
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    const svg = d3.select('#cosineGraph')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // X축 설정 (-π ~ π)
    const x = d3.scaleLinear()
        .domain([-Math.PI, Math.PI])
        .range([0, width]);

    // Y축 설정 (-1.2 ~ 1.2)
    const y = d3.scaleLinear()
        .domain([-1.2, 1.2])
        .range([height, 0]);

    // X축 추가
    svg.append('g')
        .attr('class', 'axis x-axis')
        .attr('transform', `translate(0,${height / 2})`) // 원점이 중앙에 오도록
        .call(d3.axisBottom(x).ticks(5).tickFormat(d => {
            if (d === 0) return "0";
            if (d === Math.PI) return "π";
            if (d === -Math.PI) return "-π";
            if (d === Math.PI / 2) return "π/2";
            if (d === -Math.PI / 2) return "-π/2";
            return "";
        }));

    // Y축 추가
    svg.append('g')
        .attr('class', 'axis y-axis')
        .attr('transform', `translate(${width / 2},0)`) // 원점이 중앙에 오도록
        .call(d3.axisLeft(y).ticks(5));

    // 코사인 함수 그래프 그리기
    const cosineData = [];
    for (let i = -Math.PI; i <= Math.PI; i += 0.05) {
        cosineData.push({ x: i, y: Math.cos(i) });
    }

    // 코사인 라인 그리기
    const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y))
        .curve(d3.curveNatural);

    svg.append('path')
        .datum(cosineData)
        .attr('class', 'cosine-line')
        .attr('fill', 'none')
        .attr('stroke', '#3498db')
        .attr('stroke-width', 2)
        .attr('d', line);

    // 예측값 그룹 추가
    svg.append('g')
        .attr('class', 'predictions');

    // 예측 곡선 그룹 추가
    svg.append('g')
        .attr('class', 'prediction-curve');

    // X=0 가이드라인
    svg.append('line')
        .attr('class', 'guideline')
        .attr('x1', x(0))
        .attr('y1', 0)
        .attr('x2', x(0))
        .attr('y2', height)
        .attr('stroke', '#ccc')
        .attr('stroke-dasharray', '3,3');

    // Y=0 가이드라인
    svg.append('line')
        .attr('class', 'guideline')
        .attr('x1', 0)
        .attr('y1', y(0))
        .attr('x2', width)
        .attr('y2', y(0))
        .attr('stroke', '#ccc')
        .attr('stroke-dasharray', '3,3');

    cosineGraphInitialized = true;
    debugLog('코사인 그래프 초기화 완료');

    // 창 크기 변경 시 그래프 업데이트
    window.addEventListener('resize', function () {
        if (cosineGraphInitialized) {
            initCosineGraph();
        }
    });
}

// 코사인 그래프에 예측 포인트 추가
function updateCosineGraph(x_val, y_true, y_pred) {
    const container = document.getElementById('cosineGraph');
    if (!container) return;

    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    // 스케일 함수
    const x = d3.scaleLinear()
        .domain([-Math.PI, Math.PI])
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([-1.2, 1.2])
        .range([height, 0]);

    const svg = d3.select('#cosineGraph svg g');

    // 예측 그룹
    const predictions = svg.select('.predictions');

    // 새 예측 포인트 추가
    const predPoint = predictions.append('g')
        .attr('class', 'prediction-pair');

    // 실제값 포인트 (더 이상 표시하지 않음, 실제 코사인 함수로 대체)

    // 예측값 포인트
    predPoint.append('circle')
        .attr('cx', x(x_val))
        .attr('cy', y(y_pred))
        .attr('r', 4)
        .attr('fill', '#e74c3c')
        .attr('fill-opacity', 0.7)
        .attr('stroke', 'white')
        .attr('stroke-width', 1);

    // 오차 선 (실제 코사인 값과 예측값 사이의 거리)
    predPoint.append('line')
        .attr('x1', x(x_val))
        .attr('y1', y(Math.cos(x_val)))
        .attr('x2', x(x_val))
        .attr('y2', y(y_pred))
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '2,2');

    // 포인트가 20개 이상이면 가장 오래된 포인트 제거
    const allPredictions = predictions.selectAll('.prediction-pair');
    if (allPredictions.size() > 20) {
        allPredictions.nodes()[0].remove();
    }

    // 예측 곡선 업데이트
    updatePredictionCurve();
}

// 예측 곡선 업데이트 함수
function updatePredictionCurve() {
    // 모든 예측 포인트 수집
    const predictionPoints = [];
    const predictionPairs = d3.selectAll('.prediction-pair').nodes();

    predictionPairs.forEach(pair => {
        const circle = d3.select(pair).select('circle');
        const cx = parseFloat(circle.attr('cx'));
        const cy = parseFloat(circle.attr('cy'));

        // SVG 좌표를 데이터 좌표로 변환
        const container = document.getElementById('cosineGraph');
        const margin = { top: 20, right: 20, bottom: 30, left: 50 };
        const width = container.clientWidth - margin.left - margin.right;
        const height = container.clientHeight - margin.top - margin.bottom;

        const x = d3.scaleLinear()
            .domain([-Math.PI, Math.PI])
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([-1.2, 1.2])
            .range([height, 0]);

        // 역변환으로 데이터 좌표 계산
        const xValue = x.invert(cx);
        const yValue = y.invert(cy);

        predictionPoints.push({ x: xValue, y: yValue });
    });

    // 포인트가 충분히 있을 때만 곡선 그리기
    if (predictionPoints.length < 2) return;

    // x 값으로 정렬
    predictionPoints.sort((a, b) => a.x - b.x);

    const container = document.getElementById('cosineGraph');
    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    const x = d3.scaleLinear()
        .domain([-Math.PI, Math.PI])
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([-1.2, 1.2])
        .range([height, 0]);

    // 선 생성기
    const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y))
        .curve(d3.curveBasis);  // 부드러운 곡선

    const svg = d3.select('#cosineGraph svg g');

    // 기존 예측 곡선 제거
    svg.select('.prediction-curve path').remove();

    // 새 예측 곡선 추가
    svg.select('.prediction-curve')
        .append('path')
        .datum(predictionPoints)
        .attr('class', 'prediction-line')
        .attr('fill', 'none')
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', 2)
        .attr('d', line);
}

// 코사인 그래프에 예측 곡선 추가 (기존 함수 수정)
function drawPredictionCurve(predictions, targets) {
    // 기존 코드는 유지하되, 여기서는 포인트만 추가하고 
    // 곡선은 updatePredictionCurve()에서 처리
    const container = document.getElementById('cosineGraph');
    if (!container) return;

    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    const x = d3.scaleLinear()
        .domain([-Math.PI, Math.PI])
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([-1.2, 1.2])
        .range([height, 0]);

    const svg = d3.select('#cosineGraph svg g');

    // 예측 포인트 추가
    predictions.forEach(pred => {
        // 기존 updateCosineGraph 함수 호출해서 포인트 추가
        updateCosineGraph(pred.x, Math.cos(pred.x), pred.y);
    });
}

// 자동 재생 토글 함수
function toggleAutoPlay(start) {
    if (start && !isAutoPlaying) {
        startAutoPlay();
    } else if (!start && isAutoPlaying) {
        stopAutoPlay();
    }
}

// 자동 재생 시작
function startAutoPlay() {
    if (isAutoPlaying) return;

    isAutoPlaying = true;
    document.querySelector('.container').classList.add('auto-playing');
    document.getElementById('playBtn').disabled = true;
    document.getElementById('pauseBtn').disabled = false;

    debugLog("자동 재생 시작");

    // 현재 상태에 맞게 다음 단계 실행
    advanceToNextStep();

    // 일정 간격으로 다음 단계 실행
    autoPlayInterval = setInterval(() => {
        advanceToNextStep();
    }, AUTO_PLAY_INTERVAL);
}

// 자동 재생 정지
function stopAutoPlay() {
    if (!isAutoPlaying) return;

    isAutoPlaying = false;
    document.querySelector('.container').classList.remove('auto-playing');
    document.getElementById('playBtn').disabled = false;
    document.getElementById('pauseBtn').disabled = true;

    // 인터벌 정지
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
    }

    debugLog("자동 재생 정지");
}

// 자동 재생 속도 변경
function updatePlaySpeed(speed) {
    AUTO_PLAY_INTERVAL = speed;
    debugLog(`재생 속도 변경: ${speed}ms`);

    // 재생 중이라면 인터벌 재설정
    if (isAutoPlaying) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = setInterval(() => {
            advanceToNextStep();
        }, AUTO_PLAY_INTERVAL);
    }
}

// 다음 단계로 자동 진행
function advanceToNextStep() {
    // 각 버튼의 disabled 상태에 따라 다음 단계 결정
    if (!document.getElementById('forwardBtn').disabled) {
        runForwardPass();
    } else if (!document.getElementById('lossBtn').disabled) {
        calculateLoss();
    } else if (!document.getElementById('backpropBtn').disabled) {
        runBackprop();
    } else if (!document.getElementById('updateBtn').disabled) {
        updateWeights();
    } else {
        // 모든 단계가 완료되었으면 다시 리셋하고 시작
        resetNetwork();
        // 약간의 딜레이 후 순전파 실행
        setTimeout(() => {
            if (isAutoPlaying) {
                runForwardPass();
            }
        }, AUTO_PLAY_INTERVAL);
    }
}

// 역전파 함수 (여러 출력 노드에 대한 역전파)
function runBackprop() {
    debugLog("역전파 함수 실행");

    // 현재 단계를 활성화
    updateActiveStep(3);

    // 버튼 상태 업데이트
    document.getElementById('backpropBtn').disabled = true;
    document.getElementById('updateBtn').disabled = false;

    // 상태 텍스트 업데이트 제거 - 단계별 박스로 대체

    try {
        const lossNodeIndex = networkData.nodes.length - 1;
        const outputStartIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2;
        const targetStartIndex = outputStartIndex + networkStructure.output;

        // 각 출력 노드에 대한 그래디언트 계산
        for (let i = 0; i < networkStructure.output; i++) {
            const outputIndex = outputStartIndex + i;
            const targetIndex = targetStartIndex + i;
            const outputValue = parseFloat(networkData.nodes[outputIndex].value);
            const targetValue = parseFloat(networkData.nodes[targetIndex].value);

            // 출력층 오차 그래디언트 (dL/dy = 2 * (y - t) / outputCount)
            // outputCount로 나누는 이유는 손실이 모든 출력 노드의 평균이기 때문
            const outputGradient = 2 * (outputValue - targetValue) / networkStructure.output;

            // 출력층 활성화 함수의 미분 (tanh 함수의 미분: 1 - tanh²(x))
            const outputActivationGradient = 1 - (outputValue * outputValue);

            // 디버그 시각화를 위해 노드에 그래디언트 정보 저장
            networkData.nodes[outputIndex].gradient = outputGradient.toFixed(4);

            debugLog(`출력 ${i + 1} 그래디언트: ${outputGradient.toFixed(4)}, 활성화 미분: ${outputActivationGradient.toFixed(4)}`);

            // 은닉층-출력층 가중치 그래디언트 계산
            for (let j = 0; j < networkStructure.hidden2; j++) {
                const hiddenIndex = networkStructure.input + networkStructure.hidden1 + j;
                const hiddenValue = parseFloat(networkData.nodes[hiddenIndex].value);

                const linkIndex = networkData.links.findIndex(
                    link => link.source === hiddenIndex && link.target === outputIndex
                );

                if (linkIndex !== -1) {
                    // dL/dw = dL/dy * dy/dnet * dnet/dw = outputGradient * outputActivationGradient * hiddenValue
                    const gradient = outputGradient * outputActivationGradient * hiddenValue;
                    networkData.links[linkIndex].gradient = gradient.toFixed(4);
                    debugLog(`은닉-출력 가중치 그래디언트 ${hiddenIndex}->${outputIndex}: ${gradient.toFixed(4)}`);
                }
            }
        }

        // 두번째 은닉층 노드들의 그래디언트 계산
        for (let i = 0; i < networkStructure.hidden2; i++) {
            const hiddenIndex = networkStructure.input + networkStructure.hidden1 + i;
            const hiddenValue = parseFloat(networkData.nodes[hiddenIndex].value);
            const hiddenActivationGradient = hiddenValue * (1 - hiddenValue); // 시그모이드 미분

            let hiddenGradient = 0;

            // 이 은닉 노드가 모든 출력 노드에 미치는 영향 합산
            for (let j = 0; j < networkStructure.output; j++) {
                const outputIndex = outputStartIndex + j;
                const outputGradient = parseFloat(networkData.nodes[outputIndex].gradient);
                const outputValue = parseFloat(networkData.nodes[outputIndex].value);
                const outputActivationGradient = 1 - (outputValue * outputValue); // tanh 미분

                const linkIndex = networkData.links.findIndex(
                    link => link.source === hiddenIndex && link.target === outputIndex
                );

                if (linkIndex !== -1) {
                    const weight = parseFloat(networkData.links[linkIndex].weight);
                    // 출력 오차가 은닉층 노드에 미치는 영향
                    hiddenGradient += outputGradient * outputActivationGradient * weight;
                }
            }

            // 디버그 시각화를 위해 노드에 그래디언트 정보 저장
            networkData.nodes[hiddenIndex].gradient = hiddenGradient.toFixed(4);
            debugLog(`은닉층2 ${hiddenIndex} 그래디언트: ${hiddenGradient.toFixed(4)}`);

            // 첫번째 은닉층-두번째 은닉층 가중치 그래디언트 계산
            for (let j = 0; j < networkStructure.hidden1; j++) {
                const prevHiddenIndex = networkStructure.input + j;
                const prevHiddenValue = parseFloat(networkData.nodes[prevHiddenIndex].value);

                const linkIndex = networkData.links.findIndex(
                    link => link.source === prevHiddenIndex && link.target === hiddenIndex
                );

                if (linkIndex !== -1) {
                    const gradient = hiddenGradient * hiddenActivationGradient * prevHiddenValue;
                    networkData.links[linkIndex].gradient = gradient.toFixed(4);
                    debugLog(`은닉-은닉 가중치 그래디언트 ${prevHiddenIndex}->${hiddenIndex}: ${gradient.toFixed(4)}`);
                }
            }
        }

        // 첫번째 은닉층 노드들의 그래디언트 계산
        for (let i = 0; i < networkStructure.hidden1; i++) {
            const hiddenIndex = networkStructure.input + i;
            const hiddenValue = parseFloat(networkData.nodes[hiddenIndex].value);
            const hiddenActivationGradient = hiddenValue * (1 - hiddenValue); // 시그모이드 미분

            let hiddenGradient = 0;

            // 이 은닉 노드가 두번째 은닉층의 모든 노드에 미치는 영향 합산
            for (let j = 0; j < networkStructure.hidden2; j++) {
                const nextHiddenIndex = networkStructure.input + networkStructure.hidden1 + j;
                const nextHiddenGradient = parseFloat(networkData.nodes[nextHiddenIndex].gradient);
                const nextHiddenValue = parseFloat(networkData.nodes[nextHiddenIndex].value);
                const nextHiddenActivationGradient = nextHiddenValue * (1 - nextHiddenValue); // 시그모이드 미분

                const linkIndex = networkData.links.findIndex(
                    link => link.source === hiddenIndex && link.target === nextHiddenIndex
                );

                if (linkIndex !== -1) {
                    const weight = parseFloat(networkData.links[linkIndex].weight);
                    // 두번째 은닉층 오차가 첫번째 은닉층 노드에 미치는 영향
                    hiddenGradient += nextHiddenGradient * nextHiddenActivationGradient * weight;
                }
            }

            // 디버그 시각화를 위해 노드에 그래디언트 정보 저장
            networkData.nodes[hiddenIndex].gradient = hiddenGradient.toFixed(4);
            debugLog(`은닉층1 ${hiddenIndex} 그래디언트: ${hiddenGradient.toFixed(4)}`);

            // 입력층-첫번째 은닉층 가중치 그래디언트 계산
            for (let j = 0; j < networkStructure.input; j++) {
                const inputValue = parseFloat(networkData.nodes[j].value);

                const linkIndex = networkData.links.findIndex(
                    link => link.source === j && link.target === hiddenIndex
                );

                if (linkIndex !== -1) {
                    const gradient = hiddenGradient * hiddenActivationGradient * inputValue;
                    networkData.links[linkIndex].gradient = gradient.toFixed(4);
                    debugLog(`입력-은닉 가중치 그래디언트 ${j}->${hiddenIndex}: ${gradient.toFixed(4)}`);
                }
            }
        }

        // 시각화 업데이트
        updateVisualization();

        // 역전파 애니메이션
        animateBackprop();

        debugLog("역전파 완료");
    } catch (error) {
        debugLog("역전파 함수 오류: " + error.message);
        console.error("역전파 함수 오류:", error);
    }
}

// 역전파 애니메이션
function animateBackprop() {
    const lossNodeIndex = networkData.nodes.length - 1;
    const outputStartIndex = networkStructure.input + networkStructure.hidden1 + networkStructure.hidden2;

    // 손실 노드 강조
    svg.selectAll('circle')
        .filter((d, j) => j === lossNodeIndex)
        .transition()
        .duration(500)
        .attr('r', 40)
        .transition()
        .duration(500)
        .attr('r', 35);

    // 출력층 강조 (지연 적용)
    setTimeout(() => {
        // 모든 출력 노드 강조
        for (let i = 0; i < networkStructure.output; i++) {
            const outputIndex = outputStartIndex + i;

            svg.selectAll('circle')
                .filter((d, j) => j === outputIndex)
                .classed('pulse', true)
                .transition()
                .duration(300)
                .attr('r', 35)
                .attr('fill', '#e74c3c')
                .transition()
                .duration(300)
                .attr('r', 30)
                .attr('fill', '#2ecc71')
                .on('end', function () {
                    d3.select(this).classed('pulse', false);
                });

            // 출력층-은닉층 역전파 애니메이션
            for (let j = 0; j < networkStructure.hidden2; j++) {
                const hiddenIndex = networkStructure.input + networkStructure.hidden1 + j;
                const linkIndex = networkData.links.findIndex(
                    link => link.source === hiddenIndex && link.target === outputIndex
                );

                if (linkIndex !== -1) {
                    svg.selectAll('line')
                        .filter((d, k) => k === linkIndex)
                        .transition()
                        .duration(300)
                        .attr('stroke', '#e74c3c')
                        .attr('stroke-width', 3)
                        .attr('marker-start', 'url(#arrowhead-back)') // 역전파 화살표 추가 (시작 부분에)
                        .attr('marker-end', null) // 기존 화살표 제거
                        .transition()
                        .duration(300)
                        .attr('stroke', '#999')
                        .attr('stroke-width', 2)
                        .attr('marker-start', null) // 역전파 화살표 제거
                        .attr('marker-end', 'url(#arrowhead)'); // 원래 화살표 복원
                }
            }
        }
    }, 800);

    // 은닉층2-은닉층1 역전파 애니메이션 (지연 적용)
    setTimeout(() => {
        // 두번째 은닉층 노드 강조
        for (let i = 0; i < networkStructure.hidden2; i++) {
            const hiddenIndex = networkStructure.input + networkStructure.hidden1 + i;

            svg.selectAll('circle')
                .filter((d, j) => j === hiddenIndex)
                .classed('pulse', true)
                .transition()
                .duration(300)
                .attr('r', 35)
                .attr('fill', '#e74c3c')
                .transition()
                .duration(300)
                .attr('r', 30)
                .attr('fill', '#8e44ad')
                .on('end', function () {
                    d3.select(this).classed('pulse', false);
                });

            // 은닉층2-은닉층1 링크 강조
            for (let j = 0; j < networkStructure.hidden1; j++) {
                const prevHiddenIndex = networkStructure.input + j;
                const linkIndex = networkData.links.findIndex(
                    link => link.source === prevHiddenIndex && link.target === hiddenIndex
                );

                if (linkIndex !== -1) {
                    svg.selectAll('line')
                        .filter((d, k) => k === linkIndex)
                        .transition()
                        .duration(300)
                        .attr('stroke', '#e74c3c')
                        .attr('stroke-width', 3)
                        .attr('marker-start', 'url(#arrowhead-back)') // 역전파 화살표 추가 (시작 부분에)
                        .attr('marker-end', null) // 기존 화살표 제거
                        .transition()
                        .duration(300)
                        .attr('stroke', '#999')
                        .attr('stroke-width', 2)
                        .attr('marker-start', null) // 역전파 화살표 제거
                        .attr('marker-end', 'url(#arrowhead)'); // 원래 화살표 복원
                }
            }
        }
    }, 1600);

    // 은닉층1-입력층 역전파 애니메이션 (지연 적용)
    setTimeout(() => {
        // 첫번째 은닉층 노드 강조
        for (let i = 0; i < networkStructure.hidden1; i++) {
            const hiddenIndex = networkStructure.input + i;

            svg.selectAll('circle')
                .filter((d, j) => j === hiddenIndex)
                .classed('pulse', true)
                .transition()
                .duration(300)
                .attr('r', 35)
                .attr('fill', '#e74c3c')
                .transition()
                .duration(300)
                .attr('r', 30)
                .attr('fill', '#9b59b6')
                .on('end', function () {
                    d3.select(this).classed('pulse', false);
                });

            // 은닉층1-입력층 링크 강조
            for (let j = 0; j < networkStructure.input; j++) {
                const linkIndex = networkData.links.findIndex(
                    link => link.source === j && link.target === hiddenIndex
                );

                if (linkIndex !== -1) {
                    svg.selectAll('line')
                        .filter((d, k) => k === linkIndex)
                        .transition()
                        .duration(300)
                        .attr('stroke', '#e74c3c')
                        .attr('stroke-width', 3)
                        .attr('marker-start', 'url(#arrowhead-back)') // 역전파 화살표 추가 (시작 부분에)
                        .attr('marker-end', null) // 기존 화살표 제거
                        .transition()
                        .duration(300)
                        .attr('stroke', '#999')
                        .attr('stroke-width', 2)
                        .attr('marker-start', null) // 역전파 화살표 제거
                        .attr('marker-end', 'url(#arrowhead)'); // 원래 화살표 복원
                }
            }
        }
    }, 2400);
}

// 초기 데이터 생성
networkData = initializeNetwork();

// 이제 networkData가 초기화되었으므로 테이블 업데이트 호출
updateWeightsTable();

debugLog('네트워크 초기화 완료: ' + JSON.stringify(networkData.nodes.map(n => n.value)));

// SVG 설정
svg = d3.select('#networkViz')
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('viewBox', '0 0 1000 700') // 800에서 700으로 변경
    .attr('preserveAspectRatio', 'xMidYMid meet');

// 마커 정의 (기본 화살표)
svg.append('defs').append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 15)
    .attr('refY', 0)
    .attr('orient', 'auto')
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('xoverflow', 'visible')
    .append('svg:path')
    .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
    .attr('fill', '#999')
    .style('stroke', 'none');

// 순전파용 마커 정의 (빨간 화살표, 순방향)
svg.append('defs').append('marker')
    .attr('id', 'arrowhead-forward')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 15)
    .attr('refY', 0)
    .attr('orient', 'auto')
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('xoverflow', 'visible')
    .append('svg:path')
    .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
    .attr('fill', '#e74c3c')
    .style('stroke', 'none');

// 역전파용 마커 정의 (빨간 화살표, 역방향)
svg.append('defs').append('marker')
    .attr('id', 'arrowhead-back')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 15)
    .attr('refY', 0)
    .attr('orient', 'auto-start-reverse') // 역방향 화살표
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('xoverflow', 'visible')
    .append('svg:path')
    .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
    .attr('fill', '#e74c3c')
    .style('stroke', 'none');

// 링크 생성
const link = svg.append('g')
    .selectAll('line')
    .data(networkData.links)
    .enter().append('line')
    .attr('class', 'link')
    .attr('x1', d => networkData.nodes[d.source].x)
    .attr('y1', d => networkData.nodes[d.source].y)
    .attr('x2', d => networkData.nodes[d.target].x)
    .attr('y2', d => networkData.nodes[d.target].y)
    .attr('stroke-width', d => d.type === 'forward' ? 2 : 1)
    .attr('stroke', d => {
        if (d.type === 'loss') return '#e74c3c';
        if (d.type === 'comparison') return '#f39c12';
        return '#999';
    })
    .attr('marker-end', d => d.type === 'forward' ? 'url(#arrowhead)' : null);

// 링크 텍스트 생성
const linkText = svg.append('g')
    .selectAll('text')
    .data(networkData.links.filter(d => d.type === 'forward'))
    .enter().append('text')
    .attr('x', d => (networkData.nodes[d.source].x + networkData.nodes[d.target].x) / 2)
    .attr('y', d => (networkData.nodes[d.source].y + networkData.nodes[d.target].y) / 2 - 5)
    .attr('text-anchor', 'middle')
    .attr('font-size', '10px')
    .text(d => d.weightId); // 가중치 ID만 표시

// 노드 그룹 생성
const node = svg.append('g')
    .selectAll('g')
    .data(networkData.nodes)
    .enter().append('g')
    .attr('transform', d => `translate(${d.x}, ${d.y})`)
    .attr('class', 'node');

// 노드 원 생성
node.append('circle')
    .attr('r', d => d.type === 'loss' ? 35 : 30)
    .attr('fill', d => {
        if (d.type === 'input') return '#3498db';
        if (d.type === 'hidden') {
            // 두 번째 은닉층인지 확인
            if (d.layer === 2) return '#8e44ad';
            return '#9b59b6';
        }
        if (d.type === 'output') return '#2ecc71';
        if (d.type === 'target') return '#f39c12';
        return '#e74c3c'; // loss
    });

// 노드 텍스트 생성 (값만 표시)
node.append('text')
    .attr('dy', '0.35em')
    .attr('text-anchor', 'middle')
    .attr('fill', 'white')
    .attr('font-size', '12px')
    .text(d => {
        if (d.type === 'loss') return d.value.toFixed(4);
        if (d.type === 'target') return d.value;
        return d.value;
    });

// 노드 라벨 추가 (노드 위에 노드 이름만 표시)
node.append('text')
    .attr('dy', '-20px')
    .attr('text-anchor', 'middle')
    .attr('class', 'node-label')
    .attr('fill', '#333')
    .text(d => d.label || '');

// 레이어 그룹 타이틀 추가
// 입력층 타이틀
svg.append('text')
    .attr('x', 100)
    .attr('y', 50)
    .attr('text-anchor', 'middle')
    .attr('fill', '#000000')
    .attr('font-weight', 'bold')
    .attr('font-size', '14px')
    .text('입력층');

// 첫 번째 은닉층 타이틀
svg.append('text')
    .attr('x', 300)
    .attr('y', 50)
    .attr('text-anchor', 'middle')
    .attr('fill', '#000000')
    .attr('font-weight', 'bold')
    .attr('font-size', '14px')
    .text('은닉층 1');

// 두 번째 은닉층 타이틀
svg.append('text')
    .attr('x', 500)
    .attr('y', 50)
    .attr('text-anchor', 'middle')
    .attr('fill', '#000000')
    .attr('font-weight', 'bold')
    .attr('font-size', '14px')
    .text('은닉층 2');

// 출력층 타이틀
svg.append('text')
    .attr('x', 700)
    .attr('y', 50)
    .attr('text-anchor', 'middle')
    .attr('fill', '#000000')
    .attr('font-weight', 'bold')
    .attr('font-size', '14px')
    .text('출력층');

// 타겟 타이틀
svg.append('text')
    .attr('x', 900)
    .attr('y', 50)
    .attr('text-anchor', 'middle')
    .attr('fill', '#000000')
    .attr('font-weight', 'bold')
    .attr('font-size', '14px')
    .text('실제값');

// 손실 타이틀은 손실 노드 자체에 표시되므로 별도 표시 안함

// 툴팁 생성
const tooltip = d3.select('#tooltip');

node.on('mouseover', function (event, d) {
    let content = '';

    if (d.type === 'loss') {
        content = `손실값: ${d.value.toFixed(4)}`;
    } else if (d.type === 'target') {
        content = `목표값: ${d.value}`;
    } else {
        content = `노드값: ${d.value}`;
    }

    tooltip.html(content)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
        .style('opacity', 1);
})
    .on('mouseout', function () {
        tooltip.style('opacity', 0);
    });

// 오차 표시 숨기기
document.getElementById('errorDisplay').style.opacity = 0;

// 그래프 관련 상태 초기화 (데이터는 유지)
initialError = null;
const initialErrorElement = document.getElementById('initialError');
const currentErrorElement = document.getElementById('currentError');
if (initialErrorElement) initialErrorElement.textContent = '-';
if (currentErrorElement) currentErrorElement.textContent = '-';

debugLog("네트워크 리셋 완료");

// 그래프 초기화 버튼 이벤트
document.getElementById('clearGraphBtn').addEventListener('click', function () {
    // 그래프 초기화 코드
    initCosineGraph();
    cosineGraphInitialized = true;
});
