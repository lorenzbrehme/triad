from fastapi import FastAPI
from pydantic import BaseModel
from qa_chain import get_rag_graph

app = FastAPI()
rag_graph = get_rag_graph()

class QuestionRequest(BaseModel):
    question: str

@app.post("/rag")
def ask_question(request: QuestionRequest):
    


    result = rag_graph.invoke(
        {"question": request.question},
    )
    
    return {"answer": result["answer"], "context":result["context"]}


