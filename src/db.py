import FinanceDataReader as fdr
from datetime import datetime
import numpy as np
import pandas as pd
from yahooquery import Ticker
import sqlite3
import os
from langchain_community.utilities import SQLDatabase


def make_db(market: str="", date1: str="",date2:str="",db_path=None):
    """
    market: 'KOSPI', 'KOSDAQ', 'ALL'
    date1: 시작 날짜 (예: '2024-01-01', 기본값: '2024-01-01')
    date2: 종료 날짜 (예: '2024-12-27', 기본값: 오늘 날짜)
    db_path: DB 파일 경로 (기본값: data/stock_db.db)
    """
    # 기본 경로 설정
    if db_path is None:
        # 현재 파일(db.py)이 있는 폴더의 상위 폴더에서 data/stock_db.db 경로 생성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Agent-yj 폴더
        db_path = os.path.join(parent_dir, 'data', 'stock_db.db')
    
    # 기본 날짜 설정
    if not date1:
        date1 = '2024-01-01'
    if not date2:
        date2 = datetime.today().strftime('%Y-%m-%d')
    
    print(f"📅 데이터 조회 기간: {date1} ~ {date2}")
    
    # 기존 DB 파일이 있으면 삭제
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ 기존 DB 파일을 삭제했습니다: {db_path}")
    
    # Finance datareader를 통해 ticker 정보 추출 
    kr_stocks = fdr.StockListing('KRX')
    kr_stocks = kr_stocks.loc[kr_stocks['Market'].isin(['KOSPI', 'KOSDAQ'])]
    
    # 타겟 시장 (코스피 or 코스닥 or 둘다)
    market = market.upper()
    if market == 'KOSPI' or market=='코스피':
        suffix = '.KS'
        df = kr_stocks[kr_stocks['Market'] == 'KOSPI'].copy()
        df['Ticker'] = df['Code'] + suffix
    elif market == 'KOSDAQ' or market=='코스닥':
        suffix = '.KQ'
        df = kr_stocks[kr_stocks['Market'] == 'KOSDAQ'].copy()
        df['Ticker'] = df['Code'] + suffix
    else:  # 전체
        kospi = kr_stocks[kr_stocks['Market'] == 'KOSPI'].copy()
        kosdaq = kr_stocks[kr_stocks['Market'] == 'KOSDAQ'].copy()
        kospi['Ticker'] = kospi['Code'] + '.KS'
        kosdaq['Ticker'] = kosdaq['Code'] + '.KQ'
        df = pd.concat([kospi, kosdaq], ignore_index=True)

    # Name(종목명)과 Ticker(코드+suffix)만 남기기
    stockCode_df = df[['Name', 'Ticker','Market']].reset_index(drop=True)

    # ticker list
    tickers = stockCode_df['Ticker'].tolist()

    # 날짜 범위 설정 (항상 기간 조회)
    d1 = pd.to_datetime(date1)
    d2 = pd.to_datetime(date2)
    
    if d1 <= d2:
        start_date = d1.strftime('%Y-%m-%d')
        end_date = (d2 + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        # date1이 date2보다 크면 순서 바꾸기
        start_date = d2.strftime('%Y-%m-%d')
        end_date = (d1 + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Yahoo Query를 이용해 tickers 목록의 종목들의 데이터를 호출 - yfinance보다 대용량 처리, 데이터는 동일 
    tickers_str=' '.join(tickers)
    t=Ticker(tickers_str)
    stock_hist = t.history(start=start_date,
                     end=end_date,
                     interval='1d',
                     adj_ohlc=True)

    stock_df=stock_hist.reset_index()
    stock_df=stock_df.rename(columns={'symbol':'Ticker'})
    db_df=pd.merge(stockCode_df,stock_df,how='inner',on='Ticker')
    
    db_df=db_df.rename(columns={'Name':'Stock_Name',
                                'Ticker':'Stock_Ticker'})
    
    # DB Table 별로 데이터 분리 
    # Table Stock_info
    cols=['date','open','high','low','volume','close','dividends','splits']
    df_stocks=db_df.drop(columns=cols)
    df_stocks2=df_stocks.drop_duplicates()
    
    # Table Stock_price
    cols2=['Market','Stock_Ticker']
    df_prices=db_df.drop(columns=cols2)
    
    # DB에 저장하기
    # 1. DB 연결 
    conn=sqlite3.connect(db_path)
    cur=conn.cursor()

    # 2. Stock_Info 테이블 생성
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Stock_Info (
        Stock_ticker TEXT,
        Stock_Name TEXT PRIMARY KEY,
        Market TEXT
    );
    """)

    # 3. Stock_Prices 테이블 생성
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Stock_Prices (
        Stock_Name TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        dividends REAL,
        splits REAL,
        PRIMARY KEY (Stock_Name, date),
        FOREIGN KEY (Stock_Name) REFERENCES Stocks(Stock_Name)
    );
    """)
    
    # 4. 테이블에 데이터 전달 
    df_stocks2.to_sql('Stocks', conn, if_exists='append', index=False)
    df_prices.to_sql('Stock_Prices', conn, if_exists='append', index=False)

    # 5. 커밋 및 종료
    conn.commit()
    conn.close()
        
    print(f"db가 {db_path}에 성공적으로 생성 되었습니다")
    
    
def set_db(db_path): 
    db=SQLDatabase.from_uri(f'sqlite:///{db_path}')
    return db