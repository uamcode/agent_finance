import FinanceDataReader as fdr
from datetime import datetime
import pandas as pd
from yahooquery import Ticker
import sqlite3
from langchain_community.utilities import SQLDatabase


def make_db(market: str="", date1: str="",date2:str="",db_path='stock_db.db'):
    """
    market: 'KOSPI', 'KOSDAQ', 'ALL'
    date1: 조회할 날짜 (예: '2024-12-04')
    date2: 조회할 날짜 (예: '2024-12-04')
    """
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

    # 날짜 설정
    if not date1 and not date2:
      date =datetime.today().strftime('%Y-%m-%d')
    elif date1 and not date2:
      date=date1
    elif not date1 and date2:
      date=date2
    else :
        d1 = pd.to_datetime(date1)
        d2 = pd.to_datetime(date2)
        if d1 <= d2:
          start_date = d1.strftime('%Y-%m-%d')
          end_str=d2.strftime('%Y-%m-%d')
          end_date = (d2 + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
          date=""
        else:
          start_date = d2.strftime('%Y-%m-%d')
          end_str=d1.strftime('%Y-%m-%d')
          end_date = (d1 + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
          date=""

    if date :
      start_date=date
      end_date=(pd.to_datetime(date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')

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
    cols2=['Market','Stock_ticker']
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