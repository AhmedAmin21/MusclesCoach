# MusclesCoach
## 🏋️‍♂️ AI-powered Fitness & Nutrition Chatbot  

An intelligent chatbot that acts as both a *Personal Fitness Coach* and *Nutritionist*, built using the power of FastAPI, LangChain, Groq LLM, NVIDIA AI Endpoints, and CalorieNinja API.

This chatbot helps users with personalized fitness guidance and accurate nutrition analysis of any food item — just like having a coach & nutritionist available 24/7!

---

## 🚀 Features

- 🤖 AI Fitness Coach — Gym advice, workout plans, muscle targeting tips.
- 🥗 AI Nutritionist — Analyze any food's nutrition with real API data.
- 💡 Smart Recommendation — Suggest healthier alternatives.
- 🧠 Memory Powered Conversations — Remembers last interactions.
- ⚡ FastAPI Backend — Lightweight & scalable.
- 🌐 LangChain Agents — Seamless tool-calling for nutrition analysis.

---

## 🛠️ Tech Stack

| Tech | Purpose |
|------|---------|
| FastAPI | API Framework |
| LangChain | LLM Orchestration & Memory |
| ChatGroq (GROQ LLM) | Nutrition Expert Responses |
| ChatNVIDIA (LLM) | Fitness Coach Chat |
| Calorie Ninja API | Real-time Nutrition Data |
| Python | Core Programming Language |

---

## 📊 Architecture Diagram

```plaintext
User Query
     |
     v
FastAPI Endpoint
     |
     v
Check if Query is Nutrition Related?
     |                         |
Yes (Nutrition Agent)      No (Fitness Coach)
     |                         |
Fetch Data from API        Generate Response using LLM
     |                         |
Return AI Response to User
```
---

## How to Run Locally 💻

1. Clone the repo  
```bash
git clone https://github.com/your-username/fitness-nutrition-chatbot.git
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Add your .env file
```bash
GROQ_API_KEY=your_key  
NVIDIA_API_KEY=your_key  
NINGA_API_KEY=your_key
```
4. Run the API
```bash
uvicorn main:app --reload
```
---

## API Usage 📲
POST → /fitness-advice/

Request:
```json
{
  "query": "How much protein in 100g chicken?"
}
```
Response:
```json
{
  "response": "100g chicken breast contains approx 31g protein..."
}
```
---

## Demo: Ask anything like:

- "How much protein in 100g of chicken breast?"
- "Suggest alternatives to rice with lower calories"
- "What's the best workout to target shoulders?"
- "Plan me a high-protein vegetarian meal"

---

![1](images/q1.png)
![2](images/q2.png)
![3](images/q3.png)
![4](images/q4.png)
![5](images/q5.png)
