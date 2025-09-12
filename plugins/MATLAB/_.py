from langchain_ollama.llms import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate, ChatPromptTemplate
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

import subprocess
import sys
from time import sleep

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return FileChatMessageHistory(file_path=f"agent/memory/{session_id}.json")

def create_chemistry_agent():
    # Initialize Ollama LLM
    llm = ChatOllama(
        model="gpt-oss"
    )

    # llm = llm.bind_tools(tools)
    
    # Create a custom prompt template for the ReAct agent

    template = """You have these tools:
    {tools}

    With their names:
    {tool_names}
    
    and these are the descriptions of the tools:
    {rendered_tools}

    You will solve the user's task by choosing the correct sequence of tools with their arguments.
    Return ONLY JSON with keys:
    - "name": tool name to call next, or "final" to finish
    - "arguments": dict of arguments for that tool (omit if name == "final")
    - "final_answer": present ONLY if name == "final"

    Use prior tool results (provided in the scratchpad) to decide the next step.
    stop using tools if the result is what the user needs."""

    template = """You are a helpful chemistry assistant with access to molecular analysis tools.

TOOLS:
------
You have access to the following tools:


{tools}
When you use a tool, you MUST format exactly:
Action Input: <VALID JSON object matching the tool schema>
Do NOT use key=value. Do NOT add backticks or extra text.
Example:
Action: descriptor_calculation
Action Input: {{\"smiles\": \"c1ccccc1\", \"descriptor\": \"TPSA\"}}
To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

    #prompt = PromptTemplate.from_template(template)

    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. You may use tools."), 
                                           MessagesPlaceholder("chat_history"), ("human", "{input}"), 
                                           MessagesPlaceholder("agent_scratchpad"), MessagesPlaceholder("tools"), 
                                           MessagesPlaceholder("tool_names")])
    
    # Create the ReAct agent
    #agent = create_react_agent(llm, tools, prompt)
    agent = create_tool_calling_agent(llm, tools, prompt=ChatPromptTemplate.from_messages([
         ('system', 'You are a helpful assistant with access to molecular analysis tools.'),
         ('placeholder', "{chat_history}"),
         ('human', "{input}"),
         ('placeholder', "{agent_scratchpad}")
     ]))
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    return agent_executor

try:
    # Create the agent
    agent_executor = create_chemistry_agent()
    _output = ""

    chat_history = [""]
    
    try:
        user_input = sys.argv[1]
        
        # Run the agent
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # Update chat history
        chat_history += f"Human: {user_input}\nAssistant: {response['output']}\n"

        c = open('./o.io','w')
        c.write(response['output'])
        c.close()
        
    except Exception as e:
        _output = "null"
            
except Exception as e:
    _output = "null"