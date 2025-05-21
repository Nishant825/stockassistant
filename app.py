import streamlit as st
import os
import requests
from dotenv import load_dotenv
import openai
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI as LangOpenAI  # avoid conflict

openai.api_key = os.getenv("OPENAI_API_KEY")

# Get Alpha Vantage API Key
alpha_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "9YVV7B4N0AH7L891")

# Function to fetch stock price
def fetch_stock_price(stock_symbol: str):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={alpha_api_key}'
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" in data:
        latest_day = list(data["Time Series (Daily)"].keys())[0]
        close_price = data["Time Series (Daily)"][latest_day]['4. close']
        return f"The latest closing price of {stock_symbol.upper()} is ${close_price}"
    else:
        return "Stock data not available or invalid symbol."

# Function to generate investment suggestions using OpenAI
def generate_investment_suggestions(query: str):
    prompt = f"""
    Based on the following query, suggest some investment options and strategies for stocks:

    Query: {query}
    
    Please provide a detailed plan with investment suggestions, including stock names, market trends, and tips.
    """

    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # GPT-4 not supported via `.Completion`, switch to Chat API for that
        prompt=prompt,
        max_tokens=300
    )

    return response.choices[0].text.strip()

# Tool-compatible function
def stock_investment_function(query: str):
    if "stock" in query.lower() or "investment" in query.lower():
        return generate_investment_suggestions(query)
    return "Please enter a query related to stock investments."

# LangChain LLM setup
llm = LangOpenAI(openai_api_key=openai.api_key)

# LangChain tools
tools = [
    Tool(
        name="Stock Investment Advisor",
        func=stock_investment_function,
        description="Provides investment suggestions and plans for stocks based on input queries"
    ),
    Tool(
        name="Fetch Stock Price",
        func=fetch_stock_price,
        description="Fetches real-time stock prices using the Alpha Vantage API"
    )
]

# Agent setup
agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.title("📈 Stock Investment Assistant")

user_query = st.text_area("Enter your investment goals or queries")

if user_query:
    st.write("🤖 Processing your query...")
    result = agent.run(user_query)
    st.subheader("💡 Suggestions")
    st.write(result)
