import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_ollama import OllamaLLM

from tools import ALL_TOOLS

google_api_key = os.getenv("GEMINI_API_KEY")

if not google_api_key:
    print("GOOGLE_API_KEY environment variable not set. Please set it in the .env file or your environment.")

# Tools
tools = ALL_TOOLS

# Model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
model = OllamaLLM(
        model="llama3.1",  # Change this to your preferred model
        temperature=0.1,
        base_url="http://localhost:11434"
    )

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return FileChatMessageHistory(file_path=f"agent/memory/{session_id}.json")

def chat():
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant'),
        ('placeholder', "{history}"),
        ('human', "{input}"),
        ('placeholder', "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(model, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key='input',
        history_messages_key='history'
    )

    query = input('Enter a prompt: ')
    print(agent_executor_with_memory.invoke(
        {"input": query},
        config={'configurable': {'session_id': 'abc123'}}
    )['output'])
    

if __name__ == "__main__":
    chat()