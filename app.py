import streamlit as st
import yfinance as yf
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Load API key from Streamlit secrets
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("Google API Key missing! Please set GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

@tool(description="Fetch recent stock data and company info for a given ticker.")
def get_stock_data(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="5d")
        info = stock.info

        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        market_cap = info.get("marketCap", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        summary = info.get("longBusinessSummary", "No business summary available.")

        if history.empty:
            price_data = f"No recent price history available for {ticker}."
        else:
            price_data = history[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()

        return (
            f"Price History (last 5 days):\n{price_data}\n\n"
            f"Company Info for {ticker.upper()}:\n"
            f"Sector: {sector}\n"
            f"Industry: {industry}\n"
            f"Market Cap: {market_cap}\n"
            f"Trailing PE Ratio: {trailing_pe}\n"
            f"Forward PE Ratio: {forward_pe}\n"
            f"Business Summary: {summary}"
        )
    except Exception as e:
        return f"Error fetching stock data for '{ticker}': {e}. Please check the ticker symbol."

# Initialize Gemini model with API key
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# Bind tools to model
llm_with_tools = llm.bind_tools([get_stock_data])

# Prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledgeable financial advisor. Use 'get_stock_data' tool for stock queries. "
        "If no data, explain why. For other questions, use general knowledge."
    )),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Create agent and executor
agent = create_tool_calling_agent(llm_with_tools, [get_stock_data], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_stock_data], verbose=True)

# Streamlit UI setup
st.set_page_config(page_title="Financial Stock Advisor", page_icon="📈")
st.title("📈 Financial Stock Advisor")
st.markdown("Ask me about stocks and I'll fetch latest data and advice!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.chat_message("tool_output"):
            st.markdown(f"**Tool Output:**\n```\n{msg.content}\n```")

# User input
user_input = st.chat_input("Ask about a stock (e.g., 'Analyze AAPL', 'Market cap TSLA', or anything else)")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Analyzing..."):
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.messages
            })
            answer = response["output"]
            st.session_state.messages.append(AIMessage(content=answer))
            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as err:
            error_msg = f"Oops! Something went wrong: {err}"
            st.session_state.messages.append(AIMessage(content=error_msg))
            with st.chat_message("assistant"):
                st.error(error_msg)
