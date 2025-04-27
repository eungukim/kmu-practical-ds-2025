from typing import List, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field

class DescribeArgs(BaseModel):
    csv_url: Optional[str] = Field(None, description="HTTP-accessible CSV file URL")
    df_key: Optional[str] = Field(None, description="세션에 저장된 DataFrame의 키")

class CorrelationArgs(BaseModel):
    csv_url: Optional[str] = Field(None)
    df_key: Optional[str] = Field(None, description="세션에 저장된 DataFrame의 키")
    columns: List[str]

def describe_dataset(csv_url: Optional[str] = None, df_key: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> Dict:
    """데이터셋의 기술 통계를 반환합니다"""
    if df is not None:
        # DataFrame이 직접 제공된 경우
        return df.describe().to_dict()
    elif csv_url:
        # URL이 제공된 경우
        df = pd.read_csv(csv_url)
        return df.describe().to_dict()
    elif df_key:
        # 세션 상태에서 DataFrame을 가져와야 하는 경우 (스트림릿에서 사용)
        import streamlit as st
        if df_key in st.session_state:
            df = st.session_state[df_key]
            return df.describe().to_dict()
    
    # 데이터를 찾을 수 없는 경우
    return {"error": "데이터를 찾을 수 없습니다"}

def correlation_matrix(csv_url: Optional[str] = None, df_key: Optional[str] = None, 
                        columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> Dict:
    """선택한 컬럼들 간의 상관계수를 계산합니다"""
    if df is not None:
        # DataFrame이 직접 제공된 경우
        return df[columns].corr(method="pearson").round(3).to_dict()
    elif csv_url:
        # URL이 제공된 경우
        df = pd.read_csv(csv_url)
        return df[columns].corr(method="pearson").round(3).to_dict()
    elif df_key:
        # 세션 상태에서 DataFrame을 가져와야 하는 경우 (스트림릿에서 사용)
        import streamlit as st
        if df_key in st.session_state:
            df = st.session_state[df_key]
            return df[columns].corr(method="pearson").round(3).to_dict()
    
    # 데이터를 찾을 수 없는 경우
    return {"error": "데이터를 찾을 수 없습니다"}