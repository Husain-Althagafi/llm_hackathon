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

@tool
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
    


@tool
def mass_spec_frag_tree(non_bin_tree, ndigits:int, addHs:bool=False, verbose:bool=False, **kwargs):
    """
    Draw a mass spectrometry fragmentation tree

    :returns: RDKit grid image, and (if verbose=True) fragment tree as 2D (nested) list
    :rtype: RDKit grid image, and (if verbose=True) molecules as list[list[mol]], labels as list[list[str]], mass (Daltons) as list[list[float]], m/z (mass-to-charge ratio) as list[list[float]]
    :param non_bin_tree: The NonBinTree corresponding to the parent compound
    :param ndigits: The number of digits to round mass and m/z (mass-to-charge ratio) values to. Use None to round to integer values, for example 92.
    :param addHs: Whether to add (show) hydrogen atoms in the molecular structures; default is False.
    :param verbose: Whether to return verbose output; default is False so calling this function will present a grid image automatically
    """

    # Do all processing on grids (2D lists)

    smiles_grid, labels_grid = non_bin_tree.get_grids()
    row_length = longest_row(smiles_grid)

    # Convert SMILES into molecules, adding Hs (hydrogen atoms) if requested
    mols_raw_grid = [[Chem.MolFromSmiles(smile) for smile in row] for row in smiles_grid]
    if addHs:
        mols_grid = [[Chem.AddHs(mol_raw) for mol_raw in row] for row in mols_raw_grid]
    else:
        mols_grid = mols_raw_grid

    # Determine masses, charges, and m/z
    masses_grid = [[Descriptors.ExactMolWt(mol) for mol in row] for row in mols_grid]
    masses_rounded_grid = [[round(mass, ndigits) for mass in row] for row in masses_grid]
    charges_grid = [[Chem.GetFormalCharge(mol) for mol in row] for row in mols_grid]

    mzs_grid = []
    for row_index in range(len(masses_grid)):
        mzs_row = []
        for col_index in range(len(masses_grid[0])):
            # print(f"{smiles_grid[row_index][col_index]}")
            mzs = get_fraction(masses_grid[row_index][col_index], charges_grid[row_index][col_index])
            mzs_row += [mzs]
        mzs_grid += [mzs_row]

    mzs_rounded_grid = [[round(mz, ndigits) for mz in row] for row in mzs_grid]

    # Create legend for each species:
    # <SMILES>, <mass> Da OR m/z=<m/z>
    # where m/z is used if the species is charged, and the mass if used if the species is neutral
    descrs_grid = [[]]
    descrs_row = []
    for row_index in range(len(mols_grid)):
        descrs_row = []
        for col_index in range(len(mols_grid[0])):
            descr = "  " + labels_grid[row_index][col_index] + ", " #+ ": "
            if charges_grid[row_index][col_index] != 0:
                descr += "m/z=" + str(mzs_rounded_grid[row_index][col_index]) + "  "
            else:
                descr += str(masses_rounded_grid[row_index][col_index]) + " Da"
            descrs_row += [descr]
        descrs_grid += [descrs_row]

    # Use MolsMatrixToGridImage if available in the installed version of RDKit
    try:
        drawing = Draw.MolsMatrixToGridImage(molsMatrix=mols_grid, legendsMatrix=descrs_grid, **kwargs)
    except AttributeError:
        # Flatten grids (2D lists) into 1D lists for MolsToGridImage
        mols = flatten_twoD_list(mols_grid)
        descrs = flatten_twoD_list(descrs_grid)
        drawing = Draw.MolsToGridImage(mols, legends=descrs, molsPerRow=row_length, **kwargs)
    if verbose:
          return drawing, smiles_grid, labels_grid, mols_grid, masses_grid, charges_grid, mzs_grid
    else:
          return drawing




# Utilities
def flatten_twoD_list(twoD_list: list[list]) -> list:
    return [item for sublist in twoD_list for item in sublist]

def longest_row(twoD_list: list[list]) -> int:
    return max(len(row) for row in twoD_list)

def get_fraction(num, denom):
    if denom:
        return num / denom
    else:
        return 0

def concat_grids_horizontally(grid1:list[list[str]], grid2:list[list[str]]) -> list[list[str]]:
    """Concatenate two nested lists horizontally, for example
    inputs [['a'],['b'],['c']] and [['d'], ['e'], ['f']] 
    produce [['a', 'd'], ['b', 'e'], ['c', 'f']]

    :returns: The combined grid, a two-deep nested list of strings
    :param grid1: The first grid, a two-deep nested list of strings
    :param grid2: The second grid, a two-deep nested list of strings
    """
    if grid1 == [[]]:
        combined = grid2
    elif grid2 == [[]]:
        combined = grid1
    else:
        combined = []
        for row_counter in range(len(grid1)):
            combined += [grid1[row_counter] + grid2[row_counter]]
    return combined



class NonBinTree:
    """
    Nonbinary tree class
    Note that this class is not designed to sort nodes as they are added to the tree;
    the assumption is that they should be ordered in the order added
    Adapted from https://stackoverflow.com/questions/60579330/non-binary-tree-data-structure-in-python#60579464
    """

    def __init__(self, val:str, label=None):
        """Create a NonBinTree instance"""
        self.val = val
        self.label = label
        self.nodes = []

    def add_node(self, val:str, label=None):
        """Add a node to the tree and return the new node"""
        self.nodes.append(NonBinTree(val, label))
        return self.nodes[-1]

    def __repr__(self) -> str:
        """Print out the tree as a nested list"""
        return f"NonBinTree({self.val}): {self.nodes}"

    def get_ncols(self) -> int:
        """Get the number of columns in the tree"""
        self.ncols = 0
        if len(self.nodes) > 0:
            # If there are nodes under this one, call get_ncols on them recursively
            for node in self.nodes:
                self.ncols += node.get_ncols()
        else:
            # If there are no nodes under this one, add 1 for this node
            self.ncols += 1
        return self.ncols

    def get_max_depth(self) -> int:
        """Get the maximum depth of the tree"""
        max_depth = 0
        if len(self.nodes) > 0:
            for node in self.nodes:
                this_depth = node.get_max_depth()
                max_depth = max(this_depth + 1, max_depth)
        else:
            max_depth = max(1, max_depth)
        self.max_depth = max_depth
        return self.max_depth

    def get_grids(self) -> list[list[str]]:
        """
        Get 2x two-dimensional grids for molecules' SMILES and labels where
        each row is a level in the fragment hierarchy, and
        the columns serve to arrange the fragments horizontally
        """
        # Call methods to calculate self.ncols and self.max_depth
        self.get_ncols()
        self.get_max_depth()

        # Create top row: Node value, then the rest of columns are blank (empty strings)
        grid = [[self.val] + [""] * (self.ncols - 1)]
        grid_label = [[self.label] + [""] * (self.ncols - 1)]

        n_nodes = len(self.nodes)

        if n_nodes > 0:
            nodes_grid = [[]]
            nodes_grid_label = [[]]

            # Iterate through the child nodes
            for node_counter, node in enumerate(self.nodes):
                # Recursively call this function to get the grid for children
                node_grid, node_grid_label = node.get_grids()

                # Add spacer rows if needed
                node_grid_rows = len(node_grid)
                rows_padding = self.max_depth - node_grid_rows - 1
                for padding in range(rows_padding):
                    padding_to_add = [[""] * len(node_grid[0])]
                    node_grid += padding_to_add
                    node_grid_label += padding_to_add

                nodes_grid = concat_grids_horizontally(nodes_grid, node_grid)
                nodes_grid_label = concat_grids_horizontally(nodes_grid_label, node_grid_label)

            grid += nodes_grid
            grid_label += nodes_grid_label

        return grid, grid_label
