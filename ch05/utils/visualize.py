import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# t-SNE 시각화 클래스
class TSNEVisualizer:
    
    @staticmethod
    def visualize(X, y, title='t-SNE 결과'):
        
        # 데이터 표준화
        X_scaled = StandardScaler().fit_transform(X)

        # t-SNE를 사용하여 데이터 차원 축소
        tsne = TSNE(n_components=2, random_state=0)
        X_tsne = tsne.fit_transform(X_scaled)

        # 축소된 데이터를 DataFrame으로 변환
        tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE 특성 1', 't-SNE 특성 2'])
        tsne_df['Class'] = y.values

        # Plotly를 사용하여 축소된 데이터 시각화
        fig = px.scatter(tsne_df, x='t-SNE 특성 1', y='t-SNE 특성 2', color='Class', 
                         title=title, labels={'Class': '분류'}, 
                         color_continuous_scale=px.colors.qualitative.Vivid)
        fig.update_layout(width=600, height=400, )
        fig.update_traces(marker=dict(size=12),
                          selector=dict(mode='markers'))
        fig.update_layout(legend_title_text='Class')
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
        fig.show()
    # 2개의 시각화를 2열로 배치하여 비교
    @staticmethod
    def compare_visualizations(X1, y1, X2, y2, title1='t-SNE 결과1', title2='t-SNE 결과2'):
        # 데이터 표준화
        X1_scaled = StandardScaler().fit_transform(X1)
        X2_scaled = StandardScaler().fit_transform(X2)

        # t-SNE를 사용하여 데이터 차원 축소
        tsne = TSNE(n_components=2, random_state=0)
        X1_tsne = tsne.fit_transform(X1_scaled)
        X2_tsne = tsne.fit_transform(X2_scaled)

        # 축소된 데이터를 DataFrame으로 변환
        tsne_df1 = pd.DataFrame(X1_tsne, columns=['t-SNE 특성 1', 't-SNE 특성 2'])
        tsne_df1['Class'] = y1.values
        
        tsne_df2 = pd.DataFrame(X2_tsne, columns=['t-SNE 특성 1', 't-SNE 특성 2'])
        tsne_df2['Class'] = y2.values

        # 서브플롯 생성
        fig = make_subplots(rows=1, cols=2, subplot_titles=[title1, title2])
        
        # 첫 번째 시각화
        for class_val in tsne_df1['Class'].unique():
            df_class = tsne_df1[tsne_df1['Class'] == class_val]
            fig.add_trace(
                go.Scatter(
                    x=df_class['t-SNE 특성 1'], 
                    y=df_class['t-SNE 특성 2'],
                    mode='markers',
                    marker=dict(size=12),
                    name=f'Class {class_val}',
                    showlegend=bool(class_val == tsne_df1['Class'].unique()[0])
                ),
                row=1, col=1
            )
        
        # 두 번째 시각화
        for class_val in tsne_df2['Class'].unique():
            df_class = tsne_df2[tsne_df2['Class'] == class_val]
            fig.add_trace(
                go.Scatter(
                    x=df_class['t-SNE 특성 1'], 
                    y=df_class['t-SNE 특성 2'],
                    mode='markers',
                    marker=dict(size=12),
                    name=f'Class {class_val}',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            width=1200, 
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # 축 레이블 업데이트
        fig.update_xaxes(title_text='t-SNE 특성 1')
        fig.update_yaxes(title_text='t-SNE 특성 2')
        
        return fig
