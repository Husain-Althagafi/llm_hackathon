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

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps

from io import BytesIO
import matplotlib
import math
import matplotlib.pyplot as plt


google_api_key = os.getenv("GEMINI_API_KEY")

if not google_api_key:
    print("GOOGLE_API_KEY environment variable not set. Please set it in the .env file or your environment.")

# TOOLS 
@tool
def descriptor_calculation(smiles: str, descriptor: str) -> str:
    """Calculate molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"

    desc = {desc_name: func(mol) for desc_name, func in Descriptors.descList}
    return desc[descriptor]


@tool
def partial_charge_calculation(smiles: str) -> str:
    """Calculate Gasteiger partial charges from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"

    AllChem.Mol.ComputeGasteigerCharges(mol)
    charges = [atom.GetProp('_GasteigerCharge') for atom in mol.GetAtoms()]
    return str(charges)


@tool
def generate_png_descriptors(smiles: str) -> str:
    """Generate a 2D depiction of the molecule with descriptors annotated."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    
    AllChem.Mol.ComputeGasteigerCharges(mol)
    contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
    d2d = Draw.MolDraw2DCairo(400, 400)
    sim = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, d2d, colorMap='jet', contourLines=10)
    sim.FinishDrawing()
    sim.WriteDrawingText('molecule.png')

    return "molecule.png"

# @tool
# def save_image(image) -> str:
#     """Save the image to a file and return the file path."""
#     file_path = "molecule.png"
#     with open(file_path, "wb") as f:
#         f.write(image)
#     return file_path

@tool
def display_image(file_path: str) -> str:
    """Display the image from the file path."""
    from PIL import Image
    img = Image.open(file_path)
    img.show()
    return "Image displayed"



tools =  [descriptor_calculation, partial_charge_calculation, generate_png_descriptors, display_image]

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

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
    print('starting chat')
    chat()
    