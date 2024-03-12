from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Fast API Demo"
)

class DummyResult(BaseModel):
    
    message: str = Field(
        description="A simple greeting message."
    )

@app.get("/")
def index() -> DummyResult:
    
    return DummyResult(message="Hello, Fast API!")

class User(BaseModel):
    
    name: str = Field(description="User name.")
    age: int = Field(ge=0, le=150)

@app.post("/user")
def echo_user(user: User) -> User:
    
    return user
