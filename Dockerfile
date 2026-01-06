FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN pip install uv 

# 마이크로 서비스가 아니니까 그냥 app에 다 넣는다 
WORKDIR /app 
COPY requirements.txt uv.lock ./ 

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    uv pip install --system -r requirements.txt

# 아 일단 이미지가 빌드되는 구조를 먼저 생각해봐, 에이전트가 동작하려면 일단 src 다 있어야함 일단 다복사 
COPY . .
# 컴파일로 문법 검사 
RUN python -m compileall -q .

#포트 노출 - 개발용 Fast API 
EXPOSE 8000
# 포트 노출 - 서비스용 Streamlit 
EXPOSE 8501

# CMD["python","main.py"] - 개발용 
# Dockerfile 24번째 줄
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.headless", "true"]
