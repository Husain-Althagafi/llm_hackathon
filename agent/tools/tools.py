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

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Annotated

# Pydantic stuff i dont like this >.<
DESC_MAP = {name: fn for name, fn in Descriptors.descList}
DESC_NAMES = sorted(DESC_MAP.keys())

class DescriptorArgs(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule.")

    # Pylance sees 'str', but the LLM will see an enum in the schema
    descriptor: Annotated[
        str,
        Field(
            description=f"RDKit descriptor name. One of: {', '.join(DESC_NAMES[:50])} … (total {len(DESC_NAMES)})",
            json_schema_extra={"enum": DESC_NAMES},
        ),
    ]

    # Runtime validation (and optional case-insensitive normalization)
    @field_validator("descriptor")
    @classmethod
    def validate_descriptor(cls, v: str) -> str:
        v2 = v.strip()
        if v2 in DESC_MAP:
            return v2
        # case-insensitive fallback
        matches = [n for n in DESC_MAP if n.lower() == v2.lower()]
        if matches:
            return matches[0]
        raise ValueError(f"Invalid descriptor '{v}'. Choose one of: {', '.join(DESC_NAMES[:20])} …")

# TOOLS 
@tool(args_schema=DescriptorArgs)
def descriptor_calculation(smiles: str, descriptor: str) -> str:
    """Calculate molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    
    desc = {desc_name: func(mol) for desc_name, func in Descriptors.descList}
    return desc[descriptor]


@tool
def MolToSmiles(molecule: str) -> str:
    """Format the molecule into SMILES string."""
    candidate = molecule.strip()
    mol = Chem.MolFromSmiles(candidate)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES: {candidate}")
    return Chem.MolToSmiles(mol, canonical=True)


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

#@tool
#def display_image(file_path: str) -> str:
#    """Display the image from the file path."""
#    from PIL import Image
#    img = Image.open(file_path)
#    img.show()
#    return "Image displayed"



