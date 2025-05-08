from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from IPython.display import display, HTML


# 이상치 감지를 위한 재구성 오류 계산
def visualize_reconstruction(x_val, pred, sample_idx=0):
    """
    입력 데이터와 재구성 데이터를 시각화하는 함수
    
    매개변수:
    x_val: 검증 데이터
    pred: 모델 예측 결과
    sample_idx: 시각화할 샘플의 인덱스 (기본값: 0)
    """
    # 시각화를 위한 샘플 선택
    sample_input = x_val[sample_idx]
    sample_reconstruction = pred[sample_idx]
    mse = np.mean(np.power(x_val - pred, 2), axis=1)

    # Plotly를 사용한 시각화
    fig = go.Figure()

    # 입력 데이터 추가
    fig.add_trace(go.Scatter(
        y=sample_input,
        mode='lines',
        name='입력',
        line=dict(color='blue')
    ))

    # 재구성 데이터 추가
    fig.add_trace(go.Scatter(
        y=sample_reconstruction,
        mode='lines',
        name='재구성',
        line=dict(color='red')
    ))

    # 오류 영역 추가
    fig.add_trace(go.Scatter(
        y=sample_input,
        mode='lines',
        name='오류',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        y=sample_reconstruction,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(width=0),
        showlegend=False
    ))

    # 레이아웃 설정
    fig.update_layout(
        title=f'입력 데이터와 재구성 데이터 비교 (샘플 인덱스: {sample_idx})',
        xaxis_title='특성',
        yaxis_title='값',
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor='Black',
            borderwidth=1
        ),
        width=800,
        height=500
    )

    fig.show()
    

# 이상치점수별 각 평가지표 곡선
def anomaly_metrics_curve(y, score, pos = 0) :
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=0)
    f1 = 2 / (1/precision + 1/recall)
    thres_f1_max = thresholds[np.where(f1 == f1.max())][0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=np.delete(precision, -1), mode='lines', name='precision'))
    fig.add_trace(go.Scatter(x=thresholds, y=np.delete(recall, -1), mode='lines', name='recall'))
    fig.add_trace(go.Scatter(x=thresholds, y=np.delete(f1, -1), mode='lines', name='f1'))
    fig.add_vline(x=thres_f1_max, line_color="red", line_dash="dash", line_width=1, annotation_text="{:.2f}".format(thres_f1_max), annotation_position="bottom left", annotation_font_color="red")
    fig.add_hline(y=f1.max(), line_color="red", line_dash="dash", line_width=1)
    fig.add_annotation(x=thres_f1_max, y=f1.max(), text="최대 F1 점수: {:.2f}".format(f1.max()), showarrow=True, arrowhead=1)
    fig.update_layout(
        title='이상치 점수별 평가지표 곡선',
        xaxis_title='이상치 점수',
        yaxis_title='점수',
        legend_title='평가지표',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        width=800,
        font=dict(
            family="Arial",
            size=12,
            color="#2c3e50"
        ),
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    fig.show()

    return precision, recall, f1, thresholds

# 혼동 행렬 레포트
def custom_confusion_matrix(cm):
    cm_html = f"""
    <style>
        .cm-container {{
            margin: 20px 0;
            background-color: #f5f5f5;
            padding: 20px;
            max-width: 800px;
            margin-left: 0;
        }}
        .cm-title {{
            text-align: center;
            color: #2c3e50;
        }}
        .cm-table {{
            border-collapse: collapse;
            width: 100%;
            border: 2px solid #3498db;
        }}
        .cm-header {{
            background-color: #3498db;
            color: white;
            text-align: center;
        }}
        .cm-header th {{
            padding: 10px;
            border: 1px solid #3498db;
            text-align: center;
        }}
        .cm-row td {{
            padding: 10px;
            border: 1px solid #3498db;
            background-color: white;
            color: black;
            text-align: center;
        }}
        .cm-label {{
            font-weight: bold;
        }}
    </style>
    <div class="cm-container">
        <h3 class="cm-title">혼동 행렬 (Confusion Matrix)</h3>
        <table class="cm-table">
            <tr class="cm-header">
                <th></th>
                <th>Predicted Negative</th>
                <th>Predicted Positive</th>
            </tr>
            <tr class="cm-row">
                <td class="cm-label">Actual Negative</td>
                <td>{cm[0][0]}</td>
                <td>{cm[0][1]}</td>
            </tr>
            <tr class="cm-row">
                <td class="cm-label">Actual Positive</td>
                <td>{cm[1][0]}</td>
                <td>{cm[1][1]}</td>
            </tr>
        </table>
    </div>
    """
    return cm_html


def custom_classification_report(cr):
    
    cr_df = pd.DataFrame(cr).transpose()
    
    # 분류 보고서 HTML 테이블 생성
    cr_html = """
    <style>
    .report-container {
        margin: 20px 0;
        background-color: #f5f5f5;
        padding: 20px;
        max-width: 800px;
        margin-left: 0;
    }
    .report-table {
        border-collapse: collapse;
        width: 100%;
        border: 2px solid #2ecc71;
    }
    .report-header {
        background-color: #2ecc71;
        color: white;
        text-align: center;
    }
    .report-header th {
        padding: 10px;
        border: 1px solid #2ecc71;
        text-align: center;
    }
    .report-row td {
        padding: 10px;
        border: 1px solid #2ecc71;
        background-color: white;
        color: black;
        text-align: center;
    }
    .report-label {
        font-weight: bold;
    }
   
    .report-title {
        text-align: center;
        color: #2c3e50;
    }
    </style>
    <div class="report-container">
        <h3 class="report-title">분류 보고서 (Classification Report)</h3>
        <table class="report-table">
            <tr class="report-header">
                <th>클래스</th>
                <th>정밀도<br>(Precision)</th>
                <th>재현율<br>(Recall)</th>
                <th>F1 점수<br>(F1-score)</th>
                <th>지원수<br>(Support)</th>
            </tr>
    """
    
    # 각 클래스별 행 추가
    for idx, row in cr_df.iterrows():
        if idx not in ['accuracy', 'macro avg', 'weighted avg']:
            label = '정상' if idx == '1.0' or idx == '1' else '이상' if idx == '0.0' or idx == '0' else idx
            cr_html += f"""
            <tr class="report-row">
                <td class="report-label">{label}</td>
                <td>{row['precision']:.4f}</td>
                <td>{row['recall']:.4f}</td>
                <td>{row['f1-score']:.4f}</td>
                <td>{int(row['support'])}</td>
            </tr>
            """
    
    # 평균 행 추가
    for idx in ['accuracy', 'macro avg', 'weighted avg']:
        if idx in cr_df.index:
            if idx == 'accuracy':
                row = cr_df.loc[idx]
                cr_html += f"""
                <tr class="report-row">
                    <td class="report-label">정확도</td>
                    <td colspan="3">{row['precision']:.4f}</td>
                    <td>{int(cr_df.loc['macro avg', 'support'])}</td>
                </tr>
                """
            else:
                label = '매크로 평균' if idx == 'macro avg' else '가중 평균'
                row = cr_df.loc[idx]
                cr_html += f"""
                <tr class="report-row">
                    <td class="report-label">{label}</td>
                    <td>{row['precision']:.4f}</td>
                    <td>{row['recall']:.4f}</td>
                    <td>{row['f1-score']:.4f}</td>
                    <td>{int(row['support'])}</td>
                </tr>
                """
    
    cr_html += """
        </table>
    </div>
    """

    return cr_html

# 분류 성능 지표 시각화
def autoencoder_metrics_plot(cr):
    # 정확도, 정밀도, 재현율, F1 점수 시각화 (Plotly 사용)
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # 클래스 레이블 확인 및 추출
    class_labels = [key for key in cr.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # 클래스 1 (정상) 데이터 추출
    class_1_label = class_labels[1] if len(class_labels) > 1 else None  # 두 번째 클래스 (보통 '1.0' 또는 '1')
    class_1_scores = [
        cr['accuracy'],
        cr[class_1_label]['precision'],
        cr[class_1_label]['recall'],
        cr[class_1_label]['f1-score']
    ] if class_1_label else [cr['accuracy'], 0, 0, 0]
    
    # 클래스 0 (이상) 데이터 추출
    class_0_label = class_labels[0]  # 첫 번째 클래스 (보통 '0.0' 또는 '0')
    class_0_scores = [
        cr['accuracy'],
        cr[class_0_label]['precision'],
        cr[class_0_label]['recall'],
        cr[class_0_label]['f1-score']
    ]
    
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        x=metrics,
        y=class_1_scores,
        name='정상 데이터 (클래스 1)',
        marker_color='royalblue'
    ))
    fig_metrics.add_trace(go.Bar(
        x=metrics,
        y=class_0_scores,
        name='이상치 (클래스 0)',
        marker_color='crimson'
    ))
    
    # 최대 F1 점수 표시
    max_f1 = max(class_0_scores[3], class_1_scores[3] if class_1_label else 0)
    fig_metrics.add_hline(y=max_f1, line_color="red", line_dash="dash", line_width=1)
    fig_metrics.add_annotation(x='f1-score', y=max_f1, text="최대 F1 점수: {:.2f}".format(max_f1), 
                              showarrow=True, arrowhead=1, font=dict(color="red"))
    
    fig_metrics.update_layout(
        title='분류 성능 지표',
        xaxis_title='평가 지표',
        yaxis_title='점수',
        yaxis=dict(range=[0, 1]),
        barmode='group',
        width=800,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12
        ),
        plot_bgcolor='white',
        margin=dict(t=80, b=50, l=50, r=50)
    )
    fig_metrics.update_xaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig_metrics.update_yaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig_metrics.show()
    
# 모델 손실 시각화
def autoencoder_loss_plot(loss_history):
    # 훈련 결과 시각화
    fig = go.Figure()
    
    # loss_history가 딕셔너리인 경우 (key로 접근)
    if isinstance(loss_history, dict):
        fig.add_trace(go.Scatter(y=loss_history['loss'], mode='lines', name='Train'))
        if 'val_loss' in loss_history:
            fig.add_trace(go.Scatter(y=loss_history['val_loss'], mode='lines', name='Validation'))
    # loss_history가 리스트인 경우 (직접 사용)
    else:
        fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Train'))

    fig.update_layout(
        title='모델 손실',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        width=800,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12
        ),
        plot_bgcolor='white',
        margin=dict(t=80, b=50, l=50, r=50)
    )
    fig.update_xaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig.show()
    
# 오토인코더 복원오차 라인 그래프
def autoencoder_reconstruction_error_plot1(error_df):   
    # 정상/이상치 오차 분리
    normal_errors = error_df[error_df['실제_클래스'] == 1]['재구성_오차']
    anomaly_errors = error_df[error_df['실제_클래스'] == 0]['재구성_오차']
    
    # 전체 오차와 라벨 생성
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    all_labels = np.concatenate([np.ones(len(normal_errors)), np.zeros(len(anomaly_errors))])

    # precision, recall, threshold 계산
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors, pos_label=0)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    fig = px.scatter(error_df, x='index', y='재구성_오차', color='실제_클래스', 
                     labels={'실제_클래스': '클래스'}, 
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title="클래스별 재구성 오차",
                     width=800,
                     height=500)
    fig.add_hline(y=best_threshold, line_color="red", annotation_text=f"최대 F1 임계값: {best_threshold:.4f}", 
                  annotation_position="bottom right")
    fig.update_layout(
        xaxis_title="데이터 포인트 인덱스", 
        yaxis_title="재구성 오차",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12
        ),
        plot_bgcolor='white',
        margin=dict(t=80, b=50, l=50, r=50)
    )
    fig.update_xaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig.show()

    return best_threshold

# 재구성 오차 분포 시각화
def autoencoder_reconstruction_error_plot2(normal_errors, anomaly_errors):
    # 전체 오차와 라벨 생성
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    all_labels = np.concatenate([np.ones(len(normal_errors)), np.zeros(len(anomaly_errors))])

    # precision, recall, threshold 계산
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors, pos_label=0)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=normal_errors,
        name='정상 데이터',
        opacity=0.7,
        marker_color='royalblue'
    ))
    fig_dist.add_trace(go.Histogram(
        x=anomaly_errors,
        name='이상치',
        opacity=0.7,
        marker_color='crimson'
    ))
    fig_dist.add_vline(
        x=best_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"최대 F1 임계값: {best_threshold:.4f}",
        annotation_position="top right"
    )
    
    fig_dist.update_layout(
        title='재구성 오차 분포',
        xaxis_title='재구성 오차',
        yaxis_title='빈도',
        barmode='overlay',
        width=800,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12
        ),
        plot_bgcolor='white',
        margin=dict(t=80, b=50, l=50, r=50)
    )
    fig_dist.update_xaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig_dist.update_yaxes(
        tickfont=dict(size=12),
        gridcolor='lightgray'
    )
    fig_dist.show()

    return best_threshold


def analysis_autoencoder(normal_errors, anomaly_errors, cm, cr, thresholds, loss_history=None):
    # 혼동 행렬 기반 보고서 생성
    tn, fp, fn, tp = cm.ravel()
    # 클래스 레이블 확인 및 추출
    class_labels = [key for key in cr.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
    # 클래스 0 (이상) 데이터 추출
    class_0_label = class_labels[0]  # 첫 번째 클래스 (보통 '0.0' 또는 '0')
    # 클래스 1 (정상) 데이터 추출
    class_1_label = class_labels[1] if len(class_labels) > 1 else None  # 두 번째 클래스 (보통 '1.0' 또는 '1')
    
    
    # HTML 형식으로 결과 표시
    style = """
    <style>
        .report-container {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 20px 0;
            margin-left: 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background-color: #f9f9f9;
            color: #2c3e50;
        }
        .report-title {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .section-title {
            color: #3498db;
            margin-top: 20px;
        }
        .success {
            color: green;
            font-weight: bold;
        }
        .warning {
            color: orange;
        }
        .error {
            color: red;
        }
        .info {
            color: #2ecc71;
        }
        .highlight {
            color: #3498db;
        }
        .special {
            color: #9b59b6;
            font-weight: bold;
        }
        .list-item {
            margin-bottom: 10px;
        }
    </style>
    """
    
    html_content = f"""
    {style}
    <div class="report-container">
        <h2 class="report-title">오토인코더 모델 해석</h2>
        
        <div>
            <h3 class="section-title">혼동 행렬 분석</h3>
            <p>혼동 행렬을 보면:</p>
            <ul>
                <li>실제 정상 데이터(Actual Positive) {tp+fn}개 중 <span class="success">{tp}개를 정상으로 올바르게 예측</span>하고, <span class="error">{fn}개를 이상치로 잘못 예측</span>했습니다.</li>
                <li>실제 이상치(Actual Negative) {tn+fp}개 중 <span class="success">{tn}개를 이상치로 올바르게 예측</span>하고, <span class="error">{fp}개를 정상으로 잘못 예측</span>했습니다.</li>
            </ul>
        </div>
        
        <div>
            <h3 class="section-title">성능 지표 분석</h3>
            <ul>
                <li>정상 데이터(클래스 1)에 대한 정밀도는 <b>{cr[class_1_label]['precision']*100:.1f}%</b>, 재현율은 <b>{cr[class_1_label]['recall']*100:.1f}%</b>, F1 점수는 <b>{cr[class_1_label]['f1-score']*100:.1f}%</b>입니다.</li>
                <li>이상치(클래스 0)에 대한 정밀도는 <b>{cr[class_0_label]['precision']*100:.1f}%</b>, 재현율은 <b>{cr[class_0_label]['recall']*100:.1f}%</b>, F1 점수는 <b>{cr[class_0_label]['f1-score']*100:.1f}%</b>입니다.</li>
                <li>전체 정확도는 <b>{cr['accuracy']*100:.1f}%</b>입니다.</li>
            </ul>
        </div>
    """
    
    # 손실 곡선 분석 (제공된 경우)
    if loss_history is not None:
        # loss_history가 딕셔너리인 경우 리스트로 변환
        loss_values = loss_history['loss'] if isinstance(loss_history, dict) else loss_history
        
        # 손실 곡선 특성 분석
        initial_loss = loss_values[0] if len(loss_values) > 0 else 0
        final_loss = loss_values[-1] if len(loss_values) > 0 else 0
        loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
        
        # 손실 곡선의 기울기 변화 분석
        loss_changes = np.diff(loss_values) if len(loss_values) > 1 else np.array([0])
        early_changes = loss_changes[:len(loss_changes)//3] if len(loss_changes) >= 3 else loss_changes
        late_changes = loss_changes[2*len(loss_changes)//3:] if len(loss_changes) >= 3 else loss_changes
        
        early_avg_change = np.mean(np.abs(early_changes)) if len(early_changes) > 0 else 0
        late_avg_change = np.mean(np.abs(late_changes)) if len(late_changes) > 0 else 0
        
        convergence_ratio = late_avg_change / early_avg_change if early_avg_change > 0 else 0
        
        html_content += f"""
        <div>
            <h3 class="section-title">학습 손실 곡선 분석</h3>
            <ul>
                <li>초기 손실: <b>{initial_loss:.4f}</b>, 최종 손실: <b>{final_loss:.4f}</b></li>
                <li>손실 감소율: <b>{loss_reduction:.1f}%</b></li>
        """
        
        # 손실 곡선 패턴에 따른 동적 분석
        if loss_reduction > 80:
            html_content += f"""
                <li class="success">모델이 학습 데이터에 매우 잘 적응했습니다. 손실이 {loss_reduction:.1f}% 감소했습니다.</li>
            """
        elif loss_reduction > 50:
            html_content += f"""
                <li class="info">모델이 학습 데이터에 적절히 적응했습니다. 손실이 {loss_reduction:.1f}% 감소했습니다.</li>
            """
        else:
            html_content += f"""
                <li class="warning">모델의 학습이 충분하지 않을 수 있습니다. 손실이 {loss_reduction:.1f}%만 감소했습니다.</li>
            """
        
        if convergence_ratio < 0.1:
            html_content += f"""
                <li class="success">모델이 안정적으로 수렴했습니다. 후반부 손실 변화가 초반부의 {convergence_ratio*100:.1f}%에 불과합니다.</li>
            """
        elif convergence_ratio < 0.3:
            html_content += f"""
                <li class="info">모델이 적절히 수렴했습니다. 후반부 손실 변화가 초반부의 {convergence_ratio*100:.1f}%입니다.</li>
            """
        else:
            html_content += f"""
                <li class="warning">모델이 완전히 수렴하지 않았을 수 있습니다. 후반부 손실 변화가 초반부의 {convergence_ratio*100:.1f}%로 여전히 높습니다.</li>
            """
            
        html_content += """
            </ul>
        </div>
        """
    
    # 재구성 오차 분석 (제공된 경우)
    if not normal_errors.empty and not anomaly_errors.empty:
        
        normal_mean = np.mean(normal_errors)
        anomaly_mean = np.mean(anomaly_errors)
        separation_ratio = anomaly_mean / normal_mean if normal_mean > 0 else 0
        
        html_content += f"""
        <div>
            <h3 class="section-title">재구성 오차 분석</h3>
            <ul>
                <li>정상 데이터 평균 오차: <b>{normal_mean:.4f}</b></li>
                <li>이상치 평균 오차: <b>{anomaly_mean:.4f}</b></li>
                <li>오차 분리율(이상치/정상): <b>{separation_ratio:.2f}배</b></li>
        """
        
        if separation_ratio > 3:
            html_content += f"""
                <li class="success">모델이 정상과 이상치를 매우 잘 구분합니다. 이상치의 재구성 오차가 정상보다 {separation_ratio:.2f}배 높습니다.</li>
            """
        elif separation_ratio > 1.5:
            html_content += f"""
                <li class="info">모델이 정상과 이상치를 적절히 구분합니다. 이상치의 재구성 오차가 정상보다 {separation_ratio:.2f}배 높습니다.</li>
            """
        else:
            html_content += f"""
                <li class="warning">모델이 정상과 이상치를 명확히 구분하지 못합니다. 이상치의 재구성 오차가 정상보다 {separation_ratio:.2f}배로 차이가 작습니다.</li>
            """
            
        html_content += """
            </ul>
        </div>
        """
    
    html_content += f"""
        <div>
            <h3 class="section-title">종합 해석</h3>
            <p>이 오토인코더 모델은:</p>
            <ol>
    """

    # 정상 데이터 식별 능력 평가
    if cr[class_1_label]['f1-score'] > 0.8:
        html_content += f"""
                <li class="list-item"><span class="success">정상 데이터를 식별하는 데는 매우 효과적</span>입니다.</li>
        """
    elif cr[class_1_label]['f1-score'] > 0.6:
        html_content += f"""
                <li class="list-item"><span class="info">정상 데이터를 식별하는 능력이 양호</span>합니다.</li>
        """
    else:
        html_content += f"""
                <li class="list-item"><span class="warning">정상 데이터 식별 능력이 개선이 필요</span>합니다.</li>
        """
    
    # 이상치 탐지 능력 평가
    if cr[class_0_label]['f1-score'] > 0.7:
        html_content += f"""
                <li class="list-item"><span class="success">이상치 탐지 능력이 우수</span>합니다.</li>
        """
    elif cr[class_0_label]['f1-score'] > 0.4:
        html_content += f"""
                <li class="list-item"><span class="info">이상치 탐지 능력이 보통 수준</span>입니다.</li>
        """
    else:
        html_content += f"""
                <li class="list-item"><span class="error">이상치 탐지 능력은 상대적으로 부족</span>합니다(낮은 재현율과 정밀도).</li>
        """
    
    # 손실 곡선 기반 동적 제안
    if loss_history is not None:
        if loss_reduction < 50:
            html_content += f"""
                <li class="list-item"><span class="highlight">더 많은 에포크로 학습</span>하여 모델이 충분히 수렴하도록 해야 합니다.</li>
            """
        elif convergence_ratio > 0.3:
            html_content += f"""
                <li class="list-item"><span class="highlight">학습률을 조정</span>하여 더 안정적인 수렴을 유도할 수 있습니다.</li>
            """
    
    # 재구성 오차 기반 동적 제안
    if not normal_errors.empty and not anomaly_errors.empty:
        if separation_ratio < 1.5:
            html_content += f"""
                <li class="list-item"><span class="highlight">병목층 크기를 줄여</span> 정상과 이상치 간의 재구성 오차 차이를 더 크게 만들 수 있습니다.</li>
            """
    
    # 모델 개선 제안 - 오토인코더 특화
    if cr[class_0_label]['recall'] < 0.5:
        html_content += f"""
                <li class="list-item"><span class="highlight">재구성 오차 임계값을 {thresholds*0.9:.4f}로 낮춰</span> 이상치 탐지 재현율을 높일 수 있습니다.</li>
        """
    elif cr[class_0_label]['precision'] < 0.5:
        html_content += f"""
                <li class="list-item"><span class="highlight">재구성 오차 임계값을 {thresholds*1.1:.4f}로 높여</span> 정밀도를 개선할 수 있습니다.</li>
        """
    
    # 오토인코더 모델 향상을 위한 제안
    html_content += f"""
            <li class="list-item"><span class="special">모델 구조 개선</span>: 현재 신경망 구조에서 더 깊은 층을 추가하거나 드롭아웃을 적용하여 과적합을 방지할 수 있습니다.</li>
            <li class="list-item"><span class="special">손실 함수 최적화</span>: MSE 대신 MAE나 Huber 손실 함수를 사용하여 이상치에 더 강건한 모델을 구축할 수 있습니다.</li>
            <li class="list-item"><span class="special">앙상블 접근법</span>: 여러 오토인코더를 앙상블하여 탐지 성능을 향상시킬 수 있습니다.</li>
    """
    
    # 추가 고급 기법 제안
    html_content += f"""
            <li class="list-item"><span class="highlight">변분 오토인코더(VAE)</span>나 <span class="highlight">적대적 오토인코더(AAE)</span>와 같은 고급 모델을 시도해볼 수 있습니다.</li>
            <li class="list-item"><span class="highlight">특성 중요도 분석</span>을 통해 이상치 탐지에 가장 영향력 있는 특성을 식별하고 모델을 최적화할 수 있습니다.</li>
            <li class="list-item">시계열 특성이 있다면 <span class="highlight">LSTM 오토인코더</span>를 적용하여 시간적 패턴을 더 잘 포착할 수 있습니다.</li>
    """
    
    html_content += """
            </ol>
        </div>
    </div>
    """
    
    display(HTML(html_content))


# 혼동 행렬, 분류 보고서
def autoencoder_report(x, x_pred, y, thresholds, loss_history=None):
    # 재구성 오차를 계산
    mse = np.mean(np.power(x - x_pred, 2), axis=1)
    # 재구성 오차를 기준으로 이상치 여부 예측
    pred_temp = np.where(mse > thresholds, 0, 1)
    # 데이터 타입 일관성 확보
    y = y.astype(int)
    pred_temp = pred_temp.astype(int)

    # y를 numpy 배열로 변환 (pandas Series인 경우)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    # y가 원-핫 인코딩되어 있는 경우 이를 클래스 인덱스로 변환
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    # pred_temp도 원-핫 인코딩되어 있는 경우 처리
    if len(pred_temp.shape) > 1 and pred_temp.shape[1] > 1:
        pred_temp = np.argmax(pred_temp, axis=1)
    
    error_df = pd.DataFrame({'재구성_오차': mse, '실제_클래스': y})
    error_df = error_df.reset_index()
    normal_errors = error_df[error_df['실제_클래스'] == 1]['재구성_오차']
    anomaly_errors = error_df[error_df['실제_클래스'] == 0]['재구성_오차']
    
    # 분류 보고서 생성
    cr = classification_report(y, pred_temp, output_dict=True)
    cm = confusion_matrix(y, pred_temp)
    cm_html = custom_confusion_matrix(cm)
    cr_html = custom_classification_report(cr)
    # 혼동 행렬 보고서 출력
    display(HTML(cm_html))
    # 분류 보고서 출력
    display(HTML(cr_html))
    # 분류 성능 지표 시각화
    autoencoder_metrics_plot(cr)
    # 이상치점수별 각 평가지표 곡선
    anomaly_metrics_curve(y, mse, pos=0)
    # 모델 손실 시각화
    if loss_history is not None:
        autoencoder_loss_plot(loss_history)
    
    autoencoder_reconstruction_error_plot1(error_df)
    # 재구성 오차 분포 시각화
    best_threshold = autoencoder_reconstruction_error_plot2(normal_errors, anomaly_errors)
    
    analysis_autoencoder(normal_errors, anomaly_errors, cm, cr, best_threshold, loss_history)