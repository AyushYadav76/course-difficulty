from pydantic import BaseModel
from typing import List

class Interaction(BaseModel):
    user_id: int
    content_id: int
    answered_correctly: int

class RecommendationRequest(BaseModel):
    user_id: int
    history: List[Interaction]

class RecommendationResponse(BaseModel):
    question_id: int
    difficulty_level: str
