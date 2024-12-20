from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


def load_api_key():
    """.env 파일에서 API 키를 로드하여 환경 변수에 설정합니다."""
    load_dotenv(".env.local")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        raise ValueError("OPENAI_API_KEY not found in the environment variables.")


def load_json(file_path):
    """주어진 파일 경로에서 JSON 데이터를 로드합니다."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_text_into_chunks(text):
    """텍스트를 지정된 크기와 중첩으로 분리합니다."""
    document = Document(page_content=text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents([document])


# RAG 체인 초기화
def initialize_rag_chain():
    # 환경 변수 로드
    load_api_key()

    # JSON 데이터 로드 및 변환
    file_path = "./data_init.json"
    json_data = load_json(file_path)
    json_string = json.dumps(json_data, ensure_ascii=False, indent=4)

    # 텍스트를 청크로 분리
    splits = split_text_into_chunks(json_string)

    # OpenAI 임베딩 생성
    embedding_function = OpenAIEmbeddings()

    # ChromaDB에 데이터 저장
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

    # LLM 설정 (환경변수 MODEL_NAME 변경)
    llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"))

    # Retriever (검색기) 설정
    retriever = vectorstore.as_retriever()

    # RAG 체인 구성
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# 전역 rag_chain 변수
rag_chain = initialize_rag_chain()


@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        response = rag_chain.invoke({"query": request.query})
        return {"result": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
