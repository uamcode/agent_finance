네이버 서버 연결에 실패 했습니다. 
정말 아쉽지만 설계했던 파일이라도 올리겠습니다.

에이전트는 ChatClovX HCX-005 모델을 사용했습니다.
처음에는 function calling과 langchain을 이용해서 구현을 했는데 속도가 매우 느리거나, rate limit에 걸렸습니다

이를 해결하기 위해서 yfinance와 같은 데이터를 공유하지만 대용량 데이터를 다루는 YahooQuery 라이브러리를 사용했습니다. YahooQuery를 이용해서 stock_db를 구축하였습니다. 

이후 DB와 상호작용을 하는 Agent를 설계하고자 하였고, 이를 langraph를 통해서 구현했습니다. 
![alt text](image.png)

DB의 구조를 인식하고 query를 생성하고 스스로 피드백하여 다시 query를 점검하여 사용자의 요청을 해결하는 agent를 설계했습니다. 


