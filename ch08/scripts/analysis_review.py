from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import requests
import json
import gradio as gr

# 엘라스틱서치 토크나이저 함수
def extract_words_from_text(text):
    # Elasticsearch의 endpoint 설정
    url = 'http://elasticsearch:9200/_analyze'
    
    # 요청할 데이터 준비: 사용자 정의 분석기 사용 시 'analyzer' 설정을 변경
    headers = {'Content-Type': 'application/json'}
    payload = {
        "analyzer": "nori",  # 'standard', 'nori' 등의 분석기 지정 가능
        "text": text
    }
    
    # POST 요청 수행
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    # 응답 데이터 처리
    if response.status_code == 200:
        tokens = []
        for token in response.json()['tokens']:
            if token['token'].isdigit() :
                continue
            if len(token['token']) < 2 :
                continue
            if isinstance(token['token'], int):
                continue
            tokens.append(token['token'])
        return tokens

    else:
        print("Error:", response.status_code, response.text)
        return []
    
def recommend_products(antecedent, rules_df, metric='confidence', top_n=10):
    """
    주어진 상품(antecedent)에 대해 연관 규칙을 기반으로 추천 상품 리스트를 반환하는 함수.
    
    Parameters:
    antecedent (str): 추천의 기준이 되는 상품.
    rules_df (DataFrame): 연관 규칙이 담긴 데이터프레임.
    metric (str): 정렬 기준 
    top_n (int): 반환할 추천 상품의 최대 개수.
    
    Returns:
    list의 개별 요소 n개: 추천 상품 리스트.
    """
    # 주어진 상품에 대한 연관 규칙 필터링
    filtered_rules = rules_df[rules_df['antecedents'].apply(lambda x: antecedent in x)]
    
    # 신뢰도(confidence)가 높은 순으로 정렬
    sorted_rules = filtered_rules.sort_values(by=metric, ascending=False)
    
    # 상위 N개의 결과에서 추천 상품(consequents) 추출
    recommendations = [""] * top_n
    cal_recommendations = sorted_rules['consequents'].head(top_n).apply(lambda x: list(x)[0]).tolist()
    for i, rec in enumerate(cal_recommendations):
        recommendations[i] = rec
    
    return recommendations[0], recommendations[1], recommendations[2], recommendations[3], recommendations[4], recommendations[5], recommendations[6], recommendations[7], recommendations[8], recommendations[9]

def start_server(nouns_list, rules) :
    # Gradio 인터페이스 설정
    interface = gr.Interface(
        fn=lambda product_name: recommend_products(product_name, rules, metric='lift', top_n=10),
        inputs=gr.Dropdown(choices=nouns_list, label="단어 선택"),
        outputs=[gr.Textbox(label="연관 단어 리스트" + str(i)) for i in range(1, 11)],
        title="리뷰 기반 연관 단어 추출",
        description="태깅용으로 활용 가능"
    )

    # 인터페이스 실행, 화면 높이 조정
    interface.launch(height=1200, server_name="0.0.0.0")



if __name__ == "__main__":
    print('리뷰 데이터 로드 중...')
    review_df = pd.read_csv('datasets/review/review.csv')
    print('리뷰 데이터 로드 완료')
    print('토크나이저 적용 중...')
    review_df['nouns'] = review_df.content.apply(extract_words_from_text)
    print('토크나이저 적용 완료')
    nouns_list = []
    for i in range(len(review_df)):
        nouns_list.extend(review_df.loc[i, 'nouns'])
    nouns_list = list(set(nouns_list))
    print('불필요한 단어 제거 중...')
    review_df.nouns = review_df.nouns.apply(lambda x : [noun for noun in x if not noun.isdigit()])
    nouns_list = [noun for noun in nouns_list if not noun.isdigit()]
    print('불필요한 단어 제거 완료')

    dataset = review_df.nouns.tolist()
    print('트랜잭션 인코딩 중...')
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    as_df = pd.DataFrame(te_ary, columns=te.columns_)
    print('트랜잭션 인코딩 완료')
    
    # 연관 규칙 생성
    print('연관 규칙 생성 중...')
    frequent_itemsets = apriori(
        as_df,
        min_support=0.003,      
        use_colnames=True,
        low_memory=True,
        max_len=3              
    )
    rules = association_rules(frequent_itemsets, metric="conviction", min_threshold=0.001)
    
    
    print('연관 규칙 생성 완료')
    print('서버 실행 중...')
    start_server(nouns_list, rules)
    print('서버 실행 완료')
