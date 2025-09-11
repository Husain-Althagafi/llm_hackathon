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

google_api_key = os.getenv("GEMINI_API_KEY")

if not google_api_key:
    print("GOOGLE_API_KEY environment variable not set. Please set it in the .env file or your environment.")


# TOOLS
@tool
def multiplication(a: float, b: float) -> float:
    """Multiply two numbers."""
    print('using tool: multiplication')
    return a * b  

tools = [multiplication]

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
# model = model.bind_tools(tools)

# MEMORY_FILE = 'memory.txt'

# def load_memory():
#     if os.path.exists(MEMORY_FILE):
#         with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
#             return f.read()
#     return ""

# def save_memory(history):
#     with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
#         f.write(history)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return FileChatMessageHistory(file_path=f"agent/memory/{session_id}.json")

def chat():

    # memory = load_memory().split("<>")[4:]
    # print(memory)

    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant'),
        ('placeholder', "{history}"),
        ('human', "{input}"),
        ('placeholder', "{agent_scratchpad}")
    ])

    # prompt = ChatPromptTemplate.from_messages([
    #     ('system', 'You are a helpful assistant'),
    #     ('human', "{input}"),
    #     ('placeholder', "{agent_scratchpad}")
    # ])


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
    


    # messages = []
    # query = HumanMessage(content=input("Enter a prompt: "))
    # messages.append(query)
    
    # msg = model.invoke([query])
    # tool_calls = getattr(msg, "tool_calls", None) or msg.additional_kwargs.get("tool_calls", [])

    # if not tool_calls: #handles when no tool calls were made
    #     print(f'msg content: {msg.content}')

    # else:
    #     tool_msgs = []
    #     for tc in tool_calls:
    #         name = tc["name"]
    #         args = tc.get("args") or tc.get("arguments") or {}
    #         call_id = tc.get("id") or tc.get("tool_call_id")  

    #         # Execute the right tool
    #         if name == "multiplication":
    #             result = multiplication.invoke(args)  #calls the multiplication tool with the same args
    #         else:
    #             result = f"Unknown tool: {name}"

    #         # 3) Return tool result with the same tool_call_id
    #         tool_msgs.append(
    #             ToolMessage(content=str(result), tool_call_id=call_id)
    #         )

    #     # 4) Final model turn to compose the answer
    #     final = model.invoke([msg] + tool_msgs)
    #     messages.append(final)
    #     print(f'messages: {messages}')
    #     print(f'final: {final.content}')



    
    








    # messages = []

    # query = input('Enter a prompt: ')
    # msg = model.invoke(query)
    # print(msg.content)
    # messages.append((
    #      ("system", 'You are a helpful assistant',),
    #     ("human", query),
    # ))
    # messages.append({"role": "assistant", "content": msg.content})
    # msg = model.invoke('what did i tell u last')


if __name__ == "__main__":
    print('starting chat')
    chat()
    