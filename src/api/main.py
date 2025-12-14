import pandas as pd
import random
from fastapi import FastAPI, HTTPException
from src.api.schemas import RecommendationRequest, RecommendationResponse

app = FastAPI(title="Adaptive Learning API - Advanced")

# ➕ ADD THIS HEALTH CHECK ENDPOINT
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Adaptive Learning API is running!"}

# Load model artifacts at startup
try:
    difficulty_df = pd.read_csv("output/final_question_difficulty.csv")
    difficulty_pool = difficulty_df.groupby('difficulty_level')['content_id'].apply(list).to_dict()
except FileNotFoundError:
    difficulty_pool = {'Easy': [], 'Medium': [], 'Hard': []}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    if not request.history:
        # New user: recommend Easy
        if difficulty_pool['Easy']:
            q_id = random.choice(difficulty_pool['Easy'])
            return RecommendationResponse(question_id=q_id, difficulty_level='Easy')
        else:
            raise HTTPException(status_code=500, detail="No questions available")
    
    # Simple rule: if >70% recent correct → harder
    recent = request.history[-5:]
    accuracy = sum(i.answered_correctly for i in recent) / len(recent)
    
    if accuracy >= 0.7:
        target = 'Hard'
    elif accuracy >= 0.5:
        target = 'Medium'
    else:
        target = 'Easy'
    
    # Fallback logic
    candidates = difficulty_pool.get(target, [])
    if not candidates:
        all_questions = [q for sublist in difficulty_pool.values() for q in sublist]
        if all_questions:
            q_id = random.choice(all_questions)
            # Infer level (simplified)
            level = 'Medium'
        else:
            raise HTTPException(status_code=404, detail="No questions found")
    else:
        q_id = random.choice(candidates)
        level = target
    
    return RecommendationResponse(question_id=q_id, difficulty_level=level)
