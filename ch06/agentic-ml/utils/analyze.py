# Logistic Regression 으로 혼동 행렬 분석

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import pandas as pd
from IPython.display import HTML

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC # 범주형 변수가 있을경우 사용


class LogRegBasedConfusionMatrix:
    def __init__(self, X, y, split_ratio=0.3):
        self.X = X
        self.y = y
        self.split_ratio = split_ratio

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.split_ratio, random_state=42)
        return X_train, X_test, y_train, y_test

    def fit(self):
        X_train, X_test, y_train, y_test = self.split_data()
        self.X_test = X_test
        self.y_test = y_test
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

    def predict(self):
        return self.model.predict(self.X_test)
    
    def get_confusion_matrix(self):
        y_pred = self.predict()
        return confusion_matrix(self.y_test, y_pred)
    
    def get_classification_report(self):
        y_pred = self.predict()
        return classification_report(self.y_test, y_pred)
    
    def get_classification_report_html(self):
        y_pred = self.predict()
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(report).T
        
        # 인덱스 이름 변경
        df = df.rename(index={'accuracy': '정확도'})
        
        # HTML 스타일 정의
        styles = [
            dict(selector="caption", 
                 props=[("text-align", "center"), 
                        ("font-size", "16px"), 
                        ("font-weight", "bold"),
                        ("color", "#ffffff"),
                        ("margin-bottom", "10px")])
        ]
        
        # 데이터프레임을 HTML로 변환하고 스타일 적용
        html = (df.style
                .set_caption("분류 보고서")
                .set_table_styles(styles)
                .format(precision=3)
                .background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                .set_properties(**{'text-align': 'center', 'font-size': '12px'})
                .to_html())
        
        return HTML(html)

    def plot_confusion_matrix(self, class_names=None):
        cm = self.get_confusion_matrix()
        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]
        
        # 혼동 행렬 시각화
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title='혼동 행렬',
            xaxis=dict(title='예측 클래스'),
            yaxis=dict(title='실제 클래스'),
            width=600,
            height=600,
        )
        
        return fig
    def execute(self):
        self.fit()
        confusion_matrix = self.plot_confusion_matrix()
        report_html = self.get_classification_report_html()
        
        # 결과 반환만 하고 출력하지 않음
        return confusion_matrix, report_html

class ImbalancedDataAnalyzer:
    def __init__(self, X, y, split_ratio=0.3):
        self.X = X
        self.y = y
        self.split_ratio = split_ratio
        
    def random_undersample(self):
        """무작위 언더샘플링 수행"""
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(self.X, self.y)
        return X_resampled, y_resampled
    
    def random_oversample(self):
        """무작위 오버샘플링 수행"""
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(self.X, self.y)
        return X_resampled, y_resampled
    
    def smote(self):
        """SMOTE 오버샘플링 수행"""
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X, self.y)
        return X_resampled, y_resampled
    
    def adasyn(self):
        """ADASYN 오버샘플링 수행"""
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(self.X, self.y)
        return X_resampled, y_resampled
    
    def smote_tomek(self):
        """SMOTE + Tomek 링크 결합 샘플링 수행"""
        from imblearn.combine import SMOTETomek
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(self.X, self.y)
        return X_resampled, y_resampled
    
    def smote_enn(self):
        """SMOTE + ENN 결합 샘플링 수행"""
        from imblearn.combine import SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(self.X, self.y)
   