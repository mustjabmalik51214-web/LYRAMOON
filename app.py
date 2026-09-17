import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoTokenizer

app = FastAPI()

# CORS enabled for local/web connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template Engine Setup (looks in templates/ folder)
templates = Jinja2Templates(directory="templates")

# Initialize Model
print("⏳ Loading Ira AI Core Engine...")
model_id = 'Qwen/Qwen1.5-0.5B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_id)
ai_pipeline = pipeline('text-generation', model=model_id, tokenizer=tokenizer)
print("✅ Ira AI Engine Active!")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """HTML frontend serve karne ke liye root route"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(data: dict):
    # Extracting parameters sent from Ira AI frontend
    user_message = data.get("message", "")
    custom_rules = data.get("custom_rules", "You are a helpful AI assistant.")
    
    if not user_message:
        return JSONResponse({"error": "Message is missing"}, status_code=400)

    # Constructing prompt using incoming custom rules
    full_prompt = f"<|im_start|>system\n{custom_rules}\n<|im_end|>\n<|im_start|>user\n{user_message}\n<|im_end|>\n<|im_start|>assistant\n"
    
    # Model Generation
    output = ai_pipeline(
        full_prompt, 
        max_new_tokens=150, 
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract response
    raw_text = output[0]['generated_text']
    response_text = raw_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    
    return {
        "status": "success",
        "response": response_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
