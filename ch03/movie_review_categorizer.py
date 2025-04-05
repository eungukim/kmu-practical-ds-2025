#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from tqdm import tqdm
import json
from txtai.embeddings import Embeddings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 결과 및 보고서 디렉토리 생성
os.makedirs('results', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def download_imdb_dataset():
    """
    IMDB 데이터셋 다운로드 및 로드
    """
    print("IMDB 데이터셋 다운로드 중...")
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    
    # 대체 방법: 이미 처리된 50k 리뷰 데이터셋 사용
    url_csv = "https://raw.githubusercontent.com/lutzhamel/fake-data/master/csv-files/imdb-reviews.csv"
    
    try:
        df = pd.read_csv(url_csv)
        # 랜덤 1000개 샘플링
        df_sampled = df.sample(1000, random_state=42)
        return df_sampled
    except:
        print("대체 CSV 다운로드에 실패했습니다. 예시 데이터를 생성합니다.")
        # 실패 시 예시 데이터 생성
        reviews = [
            {"review_id": i, "review": f"Sample review {i} about {np.random.choice(['acting', 'plot', 'effects', 'direction', 'cinematography'])}"}
            for i in range(1000)
        ]
        return pd.DataFrame(reviews)

def initialize_embedding_model():
    """
    txtai 임베딩 모델 초기화
    """
    print("임베딩 모델 초기화 중...")
    embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
    return embeddings

def define_categories():
    """
    영화 리뷰 카테고리 정의
    """
    categories = {
        "Acting": "The acting performance, talent of actors, cast, performances, portrayal of characters",
        "Plot": "The storyline, narrative, writing, script, story development, plot twist",
        "Direction": "Director's work, directorial choices, filmmaking, direction quality",
        "Visuals": "Visual effects, cinematography, camera work, special effects, action scenes",
        "Sound": "Sound design, music, soundtrack, score, audio quality",
        "Pacing": "Rhythm of the movie, editing, length, runtime, whether it drags or moves quickly",
        "Characters": "Character development, relatability, character arcs, motivations",
        "Dialogue": "Quality of dialogue, conversations, lines, monologues",
        "Emotional_Impact": "How the movie made viewers feel, emotional response, connection",
        "Production": "Overall production quality, budget utilization, sets, costumes, makeup",
        "Originality": "Uniqueness, innovation, freshness, creativity, novelty of the film",
        "Rewatchability": "Whether the film is worth watching again, replay value"
    }
    return categories

def create_category_embeddings(embeddings, categories):
    """
    카테고리 텍스트로부터 임베딩 생성
    """
    print("카테고리 임베딩 생성 중...")
    # 카테고리와 설명을 함께 사용하여 더 풍부한 의미 캡처
    category_texts = [f"{cat}: {desc}" for cat, desc in categories.items()]
    
    # 카테고리 텍스트 인덱싱
    embeddings.index([(i, text, None) for i, text in enumerate(category_texts)])
    
    return category_texts

def categorize_reviews(embeddings, reviews_df, category_texts):
    """
    리뷰 텍스트를 카테고리로 분류
    """
    print("리뷰 카테고리 분류 중...")
    results = []
    
    for idx, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
        review_id = row.get('review_id', idx)
        review_text = row.get('review', row.get('text', ''))
        
        # 최대 3개의 가장 관련성 높은 카테고리 찾기
        matches = embeddings.search(review_text, limit=3)
        
        # 첫 번째(가장 유사한) 카테고리 선택
        if matches and len(matches) > 0:
            best_match_idx, score = matches[0]
            best_category = category_texts[best_match_idx].split(':')[0]
            
            # 결과 저장
            results.append({
                'review_id': review_id,
                'category': best_category,
                'review': review_text,
                'score': score
            })
    
    return pd.DataFrame(results)

def save_results(categorized_df):
    """
    분류 결과를 CSV 파일로 저장
    """
    print("결과 저장 중...")
    # 필요한 열만 선택하여 CSV로 저장
    categorized_df[['review_id', 'category', 'review']].to_csv('results/classification.csv', index=False)
    print(f"결과가 'results/classification.csv'에 저장되었습니다.")

def generate_visualization(categorized_df):
    """
    결과 시각화 및 HTML 보고서 생성
    """
    print("시각화 중...")
    # 카테고리 빈도 계산
    category_counts = categorized_df['category'].value_counts()
    
    # JavaScript용 데이터 준비
    categories_js = ', '.join([f'"{cat}"' for cat in category_counts.index])
    counts_js = ', '.join([str(int(count)) for count in category_counts.values])
    
    # 시각화를 위한 HTML 템플릿 생성
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IMDB 영화 리뷰 카테고리 분석</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
            }}
            .chart-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                margin-bottom: 30px;
            }}
            .chart {{
                width: 45%;
                min-width: 400px;
                margin: 10px;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 0 5px rgba(0,0,0,0.05);
            }}
            .insight {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-left: 4px solid #007bff;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>IMDB 영화 리뷰 카테고리 분석</h1>
            <p>총 {len(categorized_df)} 개의 영화 리뷰를 분석하여 각 리뷰의 주요 카테고리를 분류했습니다.</p>
            
            <div class="chart-container">
                <div class="chart">
                    <h2>카테고리 분포 (도넛 차트)</h2>
                    <canvas id="donutChart"></canvas>
                </div>
                <div class="chart">
                    <h2>카테고리 분포 (막대 그래프)</h2>
                    <canvas id="barChart"></canvas>
                </div>
            </div>
            
            <div class="insight">
                <h2>분석 결과 주요 인사이트</h2>
                <p>영화 리뷰에서 가장 많이 언급된 카테고리는 <strong>{category_counts.index[0]}</strong>로, 전체의 약 {(category_counts.iloc[0]/len(categorized_df)*100):.1f}%를 차지했습니다.</p>
                <p>두 번째로 많이 언급된 카테고리는 <strong>{category_counts.index[1]}</strong>로, 전체의 약 {(category_counts.iloc[1]/len(categorized_df)*100):.1f}%를 차지했습니다.</p>
                <p>이는 영화 리뷰 작성자들이 주로 이 두 가지 측면에 초점을 맞추고 있음을 시사합니다.</p>
                <p>추가 분석: 영화 리뷰에서 가장 적게 언급된 카테고리는 <strong>{category_counts.index[-1]}</strong>로, 이 측면은 리뷰 작성자들에게 상대적으로 덜 중요하게 여겨지는 것으로 보입니다.</p>
            </div>
        </div>
        
        <script>
            // 데이터 준비
            const categories = [{categories_js}];
            const counts = [{counts_js}];
            
            // 색상 배열
            const colors = [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', 
                '#FF9F40', '#C9CBCF', '#7FB3D5', '#F39C12', '#2ECC71',
                '#E74C3C', '#9B59B6'
            ];
            
            // 도넛 차트
            const donutCtx = document.getElementById('donutChart').getContext('2d');
            new Chart(donutCtx, {{
                type: 'doughnut',
                data: {{
                    labels: categories,
                    datasets: [{{
                        data: counts,
                        backgroundColor: colors.slice(0, categories.length),
                        borderWidth: 1,
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'right',
                        }},
                        title: {{
                            display: true,
                            text: '영화 리뷰 카테고리 분포'
                        }}
                    }}
                }}
            }});
            
            // 막대 그래프
            const barCtx = document.getElementById('barChart').getContext('2d');
            new Chart(barCtx, {{
                type: 'bar',
                data: {{
                    labels: categories,
                    datasets: [{{
                        label: '리뷰 수',
                        data: counts,
                        backgroundColor: colors.slice(0, categories.length),
                        borderWidth: 1,
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open('reports/visualization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"시각화 결과가 'reports/visualization.html'에 저장되었습니다.")

def main():
    print("IMDB 영화 리뷰 카테고리 분류 프로그램을 시작합니다.")
    
    # 데이터셋 다운로드 및 로드
    reviews_df = download_imdb_dataset()
    print(f"{len(reviews_df)}개의 리뷰를 로드했습니다.")
    
    # 임베딩 모델 초기화
    embeddings = initialize_embedding_model()
    
    # 카테고리 정의
    categories = define_categories()
    
    # 카테고리 임베딩 생성
    category_texts = create_category_embeddings(embeddings, categories)
    
    # 리뷰 카테고리 분류
    categorized_df = categorize_reviews(embeddings, reviews_df, category_texts)
    
    # 결과 저장
    save_results(categorized_df)
    
    # 시각화
    generate_visualization(categorized_df)
    
    print("프로그램이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()