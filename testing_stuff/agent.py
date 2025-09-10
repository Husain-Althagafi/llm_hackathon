import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


import getpass

#config
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in the .env file or your environment.")

@tool
def multiplication(a: float, b: float) -> float:
    """Multiply two numbers."""
    print('using tool: multiplication')
    return a * b   

tools = [multiplication]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

query = input('Enter a prompt: ')
messages = [HumanMessage(content=query)]

msg = llm_with_tools.invoke(query)

tool_calls = getattr(msg, "tool_calls", None) or msg.additional_kwargs.get("tool_calls", [])

if not tool_calls:
    # Model chose not to call a tool; just print the reply
    print(msg.content)


else:
    tool_msgs = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args") or tc.get("arguments") or {}
        call_id = tc.get("id") or tc.get("tool_call_id")  # Gemini typically sets "id"

        # Execute the right tool
        if name == "multiplication":
            # With @tool objects, prefer .invoke(args) (dict) or .func(**args)
            result = multiplication.invoke(args)  # same as multiplication.func(**args)
        else:
            result = f"Unknown tool: {name}"

        # 3) Return tool result with the SAME tool_call_id
        tool_msgs.append(
            ToolMessage(content=str(result), tool_call_id=call_id)
        )

    # 4) Final model turn to compose the answer
    final = llm_with_tools.invoke(messages + [msg] + tool_msgs)
    print(final.content)