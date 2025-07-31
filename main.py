# app.py
from fastapi import FastAPI, Header, HTTPException, Query
from dotenv import load_dotenv
import pandas as pd
import os
from agent import agent
from pydantic import BaseModel
from langchain.schema import HumanMessage


app=FastAPI()


class AnswerResponse(BaseModel):
    answer: str

@app.get("/agent", response_model=AnswerResponse)
async def agent_get(
    question: str = Query(..., description="질문 텍스트"),
    authorization: str = Header(..., alias="Authorization"),
    x_request_id: str = Header(..., alias="X-NCP-CLOVASTUDIO-REQUEST-ID"),
):

    # 2) 에이전트 실행
    init_state = {"messages": [HumanMessage(content=question)]}
    final_state = agent.invoke(init_state)

    # 3) 최종 답변 추출
    last_msg = final_state["messages"][-1]
    if not hasattr(last_msg, "content"):
        raise HTTPException(status_code=500, detail="Agent did not return a valid response")

    return AnswerResponse(answer=last_msg.content)
