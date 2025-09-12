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
from rdkit.Chem import Descriptors, AllChem, Draw, rdMolDescriptors, rdFMCS
from rdkit.Chem.Draw import SimilarityMaps

from pydantic import BaseModel, Field, field_validator
from typing import Annotated

# Pydantic stuff i dont like this >.<
# DESC_MAP = {name: fn for name, fn in Descriptors.descList}
# DESC_NAMES = sorted(DESC_MAP.keys())

# class DescriptorArgs(BaseModel):
#     smiles: str = Field(..., description="SMILES string of the molecule.")

#     # Pylance sees 'str', but the LLM will see an enum in the schema
#     descriptor: Annotated[
#         str,
#         Field(
#             description=f"RDKit descriptor name. One of: {', '.join(DESC_NAMES[:50])} … (total {len(DESC_NAMES)})",
#             json_schema_extra={"enum": DESC_NAMES},
#         ),
#     ]

#     # Runtime validation (and optional case-insensitive normalization)
#     @field_validator("descriptor")
#     @classmethod
#     def validate_descriptor(cls, v: str) -> str:
#         v2 = v.strip()
#         if v2 in DESC_MAP:
#             return v2
#         # case-insensitive fallback
#         matches = [n for n in DESC_MAP if n.lower() == v2.lower()]
#         if matches:
#             return matches[0]
#         raise ValueError(f"Invalid descriptor '{v}'. Choose one of: {', '.join(DESC_NAMES[:20])} …")

# TOOLS 
@tool#(args_schema=DescriptorArgs)
def descriptor_calculation(smiles: str, descriptor: str) -> str:
    """Calculate molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    
    desc = {desc_name: func(mol) for desc_name, func in Descriptors.descList}
    return desc[descriptor]


@tool
def MolToSmiles(molecule: str) -> str:
    """Convert the molecule into SMILES format"""
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


@tool
def chemical_transformation_remove(smiles: str, smarts: str):
    """Remove SMARTS from SMILES
    
    Args:
        smiles: molecule string to remove molecule from.
        smarts: The molecule pattern to remove. 

    Returns:
        The modified molecules in SMILES format.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"Invalid molecule SMILES: {smiles}"
        
        # Try SMARTS first, then SMILES
        patt = Chem.MolFromSmarts(smarts) or Chem.MolFromSmiles(smarts)
        if patt is None:
            return f"Invalid pattern: {smarts}"
        
        rm = Chem.DeleteSubstructs(mol,patt)
        return "Modified Molecule: "+ Chem.MolToSmiles(rm)
    except Exception as e:
        return f"Error: {e}"
    

@tool
def chemical_transformation_replace(smiles: str, smarts: str, replace: str):
    """Remove SMARTS from SMILES
    
    Args:
        smiles: molecule string to remove molecule from.
        smarts: The molecule pattern to remove.
        replace: The molecule pattern to replace with. 

    Returns:
        The modified molecules in SMILES format.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"Invalid molecule SMILES: {smiles}"
        
        replace = Chem.MolFromSmiles(replace)
        if mol is None:
            return f"Invalid molecule SMILES: {replace}"
        
        # Try SMARTS first, then SMILES
        patt = Chem.MolFromSmarts(smarts) or Chem.MolFromSmiles(smarts)
        if patt is None:
            return f"Invalid pattern: {smarts}"
        
        mod_m = Chem.ReplaceSubstructs(mol,patt,replace)
        return "Modified Molecule: "+ Chem.MolToSmiles(mod_m)
    except Exception as e:
        return f"Error: {e}"


@tool
def molecule_side_chain_removal(smiles: str, smiles_sc: str):
    """
    Replaces sidechains with the provided side chain
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"Invalid molecule SMILES: {smiles}"
        
        sc = Chem.MolFromSmiles(smiles_sc)
        if sc is None:
            return f"Invalid molecule SMILES: {smiles_sc}"

        
        mod_m = Chem.ReplaceSidechains(smiles, smiles_sc)
        return "Modified Molecule: "+ Chem.MolToSmiles(mod_m)
    except Exception as e:
        return f"Error: {e}"

@tool
def molecule_core_removal(smiles: str, core: str):
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"Invalid molecule SMILES: {smiles}"
    
        core_ = Chem.MolFromSmiles(core)
        if core_ is None:
            return f"Invalid molecule SMILES: {core}"
        
        mod_m = ReplaceCore(mol, core_)
        return "Modified Molecule: " + Chem.MolToSmiles(mod_m)
    except Exception as e:
        return f"Error: {e}"
def display_image(file_path: str) -> str:
    """Display the image from the file path."""
    from PIL import Image
    img = Image.open(file_path)
    img.show()
    return "Image displayed"


@tool
def molecular_structure_analysis(smiles: str) -> str:
    """Analyze molecular structure by looping over atoms and bonds from a SMILES string.
    
    Args:
        smiles: A SMILES string representing a molecule (e.g., 'C1OC1', 'c1ccccc1', 'CCO')
    
    Returns:
        Detailed information about atoms, bonds, and their relationships including
        atomic numbers, symbols, bond types, connectivity, neighbor relationships,
        atom indices and valences.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string. Please provide a valid SMILES representation."
        
        result = []
        result.append(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(mol)}")
        result.append(f"Number of atoms: {mol.GetNumAtoms()}")
        result.append(f"Number of bonds: {mol.GetNumBonds()}")
        result.append("")
        
        # Loop over atoms
        result.append("ATOMS:")
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            valence = atom.GetExplicitValence()
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
            neighbor_indices = [n.GetIdx() for n in atom.GetNeighbors()]
            
            result.append(f"Atom {i}: {symbol} (atomic_num={atomic_num}, valence={valence})")
            result.append(f"  Neighbors: {neighbors} at indices {neighbor_indices}")
        
        result.append("")
        
        # Loop over bonds
        result.append("BONDS:")
        for i, bond in enumerate(mol.GetBonds()):
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            begin_symbol = mol.GetAtomWithIdx(begin_idx).GetSymbol()
            end_symbol = mol.GetAtomWithIdx(end_idx).GetSymbol()
            
            result.append(f"Bond {i}: {begin_symbol}({begin_idx})-{end_symbol}({end_idx}) [{bond_type}]")
        
        result.append("")
        
        # Additional connectivity information
        result.append("CONNECTIVITY MATRIX:")
        for i in range(mol.GetNumAtoms()):
            connections = []
            for j in range(mol.GetNumAtoms()):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond:
                    connections.append(f"{j}({bond.GetBondType()})")
            if connections:
                symbol = mol.GetAtomWithIdx(i).GetSymbol()
                result.append(f"Atom {i}({symbol}) connected to: {', '.join(connections)}")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error analyzing molecule: {str(e)}"



@tool
def substructure_search(molecule_smiles: str, pattern: str, use_chirality: bool = False) -> str:
    """Search for substructures in a molecule using SMILES or SMARTS patterns.
    
    Args:
        molecule_smiles: SMILES string of the target molecule to search in
        pattern: SMILES or SMARTS pattern to search for (e.g., 'ccO', 'CO', 'c[NH1]')
        use_chirality: Whether to consider stereochemistry in matching (default: False)
    
    Returns:
        Information about substructure matches including whether pattern is found,
        atom indices of matches, and total number of matches found.
    """
    try:
        mol = Chem.MolFromSmiles(molecule_smiles)
        if mol is None:
            return f"Invalid molecule SMILES: {molecule_smiles}"
        
        # Try SMARTS first, then SMILES
        patt = Chem.MolFromSmarts(pattern) or Chem.MolFromSmiles(pattern)
        if patt is None:
            return f"Invalid pattern: {pattern}"
        
        # Check for matches
        has_match = mol.HasSubstructMatch(patt, useChirality=use_chirality)
        
        if not has_match:
            return f"Pattern '{pattern}' not found in {molecule_smiles}"
        
        # Get all matches
        matches = mol.GetSubstructMatches(patt, useChirality=use_chirality)
        
        result = [f"Pattern '{pattern}' found {len(matches)} time(s) in {molecule_smiles}"]
        result.append(f"Match indices: {matches}")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error: {str(e)}"
    


