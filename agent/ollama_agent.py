#!/usr/bin/env python3
"""
LangChain Ollama Agent with RDKit Molecular Structure Analysis Tool
Usage: python langchain_agent.py
"""

from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

from tools import ALL_TOOLS

def create_chemistry_agent():
    """Create a chemistry agent with molecular analysis capabilities"""
    
    # Initialize Ollama LLM
    llm = OllamaLLM(
        model="llama3.1",  # Change this to your preferred model
        temperature=0.1,
        base_url="http://localhost:11434"
    )
    
    # Create tools list
    tools = ALL_TOOLS
    
    # Create a custom prompt template for the ReAct agent
    template = """You are a helpful chemistry assistant with access to molecular analysis tools.

TOOLS:
------
You have access to the following tools:

{tools}

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

    prompt = PromptTemplate.from_template(template)
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

def main():
    print("🧪 LangChain Chemistry Agent with RDKit Tools")
    print("=" * 60)
    print("Prerequisites:")
    print("1. Ollama running locally (ollama serve)")
    print("2. A model downloaded (e.g., ollama pull llama3.2)")
    print("3. Dependencies installed:")
    print("   pip install langchain langchain-community rdkit")
    print("=" * 60)
    
    try:
        # Create the agent
        print("Initializing agent...")
        agent_executor = create_chemistry_agent()
        print("Agent initialized successfully!")
        
        print("\nTry these examples:")
        print("- 'Analyze the structure of C1OC1' (epoxide)")
        print("- 'What atoms and bonds are in c1ccccc1?' (benzene)")
        print("- 'Show me the molecular structure of CCO' (ethanol)")
        print("- 'Analyze CC(=O)O' (acetic acid)")
        print("- 'quit' to exit")
        print()
        
        chat_history = ""
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                if not user_input:
                    continue
                
                print("\n" + "="*50)
                
                # Run the agent
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                print("="*50)
                print(f"\nFinal Answer: {response['output']}")
                print("-" * 50)
                
                # Update chat history
                chat_history += f"Human: {user_input}\nAssistant: {response['output']}\n"
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during execution: {e}")
                print("Continuing...")
                
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check if model is available: ollama list")
        print("3. Verify dependencies are installed")

if __name__ == "__main__":
    main()