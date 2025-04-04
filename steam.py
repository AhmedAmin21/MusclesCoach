from fastapi import FastAPI, Query
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from functools import lru_cache
import requests
import os
import re 
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
ninga_api_key = os.getenv('NINGA_API_KEY')
nvidia_api_key = os.getenv('NVIDIA_API_KEY')

api_url = 'https://api.calorieninjas.com/v1/nutrition?query='

# Initialize FastAPI
app = FastAPI(title="Fitness & Nutrition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# Memory for conversation history
memory =  ConversationBufferWindowMemory(k=5, memory_key="history", return_messages=True)   
# Cache API responses to reduce redundant calls
@lru_cache(maxsize=100)
def fetch_nutrition_data(query: str):
    response = requests.get(api_url + query, headers={'X-Api-Key': ninga_api_key})
    if response.status_code == 200:
        data = response.json()
        if data.get("items"):
            item = data["items"][0]  # First food item
            return {
                "name": item.get("name", "Unknown"),
                "serving_size": item.get("serving_size_g", "N/A"),
                "calories": item.get("calories", "N/A"),
                "total_fat": item.get("fat_total_g", "N/A"),
                "protein": item.get("protein_g", "N/A"),
                "sugar": item.get("sugar_g", "N/A")
            }
    return {"error": "No nutrition data found."}

# Define a tool to fetch nutrition data
api_calling_tool = StructuredTool.from_function(
    func=fetch_nutrition_data,
    name='Ninga',
    description='Fetches nutritional info of a food item (name, calories, fat, protein, sugar, serving size).'
)

tools = [api_calling_tool]
llm_nutrition = ChatGroq(api_key=api_key, model_name='gemma2-9b-it')

# Define agent prompt for nutrition analysis
nutrition_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a nutritionist. Follow these rules:
     1. Always fetch nutritional details first by calling the API.
     2. If asked to replace an item, suggest at least two alternatives with similar profiles.
     3. Compare the new food with the original based on nutrition.
     Format your answer like this:
     - **Replacement Food:** [Food Name]
     - **Comparison:**
         - **Original:** [Calories, Protein, Fat, etc.]
         - **Replacement:** [Calories, Protein, Fat, etc.]
     """),
    ('human', '{input}'),
    ("placeholder", "{agent_scratchpad}")
])

tool_agent = create_tool_calling_agent(llm_nutrition, tools, nutrition_prompt)
agent_ex = AgentExecutor(agent=tool_agent, tools=tools, verbose=True)

# Chat model for general fitness queries
llm_chatbot = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=nvidia_api_key)

fitness_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a fitness coach with 10 years of experience. your name is muscles coach, You are an expert in both nutrition and gym training.
You also have knowledge of human anatomy, body parts, and how to target them in weightlifting training..  
**Important Rules:**
- If the question is related to nutrition, extract and explain the nutritional data.
- Keep responses **concise yet informative**.
- IMPORTANT: Always respond to the user's most recent question in history, not previous questions.
- Start fresh with each new query - don't continue previous conversations unless explicitly referenced."""),
MessagesPlaceholder(variable_name="history"),
('human', '{input}')
    
])

sop = StrOutputParser()
conversation = ConversationChain(
    llm=llm_chatbot,         
    prompt=fitness_prompt,    
    memory=memory,
    verbose=True
)
memory.save_context({"input": "Hi"}, {"output": "Hello. I'm Muscles Coach, your fitness expert. What's your fitness goal or question today?"})

# Regex pattern for nutrition-related keywords
nutrition_keywords = r"\b(rice|wheat|oats|corn|barley|quinoa|millet|rye|spinach|lettuce|kale|carrot|potato|beet|broccoli|cauliflower|cabbage|tomato|bell pepper|eggplant|peas|green beans|zucchini|pumpkin|orange|lemon|grapefruit|strawberry|blueberry|raspberry|banana|mango|pineapple|peach|cherry|plum|watermelon|cantaloupe|apple|pear|grape|beef|pork|lamb|chicken|turkey|duck|salmon|tuna|shrimp|lentils|chickpeas|black beans|kidney beans|almonds|walnuts|sunflower seeds|tofu|tempeh|edamame|milk|goat milk|cheddar cheese|mozzarella cheese|feta cheese|yogurt|butter|almond milk|soy milk|coconut yogurt|ghee|olive oil|coconut oil|sunflower oil|canola oil|soybean oil|flaxseeds|chia seeds|peanuts|avocado|basil|cilantro|parsley|garlic|onion|ginger|black pepper|cumin|paprika|turmeric|cinnamon|cloves|sugar|brown sugar|honey|maple syrup|molasses|tea|coffee|orange juice)\b"

@app.get("/")
def home():
    return {"message": "Welcome to the Fitness & Nutrition API"}

@app.post("/fitness-advice/")
def get_fitness_advice(query: str = Query(..., description="Enter a fitness or nutrition-related query")):
    """
    Endpoint to get fitness or nutrition-related advice.
    """
    # Clear memory if user requests a reset
    if query.lower() == "reset":
        conversation.memory.clear()
        return {"response": "Memory cleared! Start a new conversation."}

    if re.search(nutrition_keywords, query, re.IGNORECASE):
        answer = agent_ex.invoke({'input': query})
        answer = answer['output']
    else:
        answer = conversation.predict(input=query)

    return {"response": answer}

