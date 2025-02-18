"""
borrow from instructor library tutorial: https://python.useinstructor.com/tutorials/5-knowledge-graphs/

"""

from pydantic import BaseModel, Field
from typing import Optional
from graphviz import Digraph
from IPython.display import display

import instructor 
from openai import OpenAI

client = instructor.patch(OpenAI())

class Node(BaseModel):
    id: int
    label: str
    color: str

    def __hash__(self) -> int:
        return hash((id, self.label))
    
class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.label))

class KnowledgeGraph(BaseModel):
    # Optional list of nodes and edges in the knowledge graph
    nodes: Optional[list[Node]] = Field(..., default_factory=list)
    edges: Optional[list[Edge]] = Field(..., default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        # This method updates the current graph with the other graph, deduplicating nodes and edges.
        return KnowledgeGraph(
            nodes=list(set(self.nodes + other.nodes)),  # Combine and deduplicate nodes
            edges=list(set(self.edges + other.edges)),  # Combine and deduplicate edges
        )
    

    def visualize_knowledge_graph(self):
        dot = Digraph(comment="Knowledge Graph")

        for node in self.nodes:
            dot.node(str(node.id), node.label, color=node.color)
        for edge in self.edges:
            dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)
        
        return display(dot)
    

def generate_graph(input: list[str]) -> KnowledgeGraph:
    # Initialize an empty KnowledgeGraph
    cur_state = KnowledgeGraph()

    # Iterate over the input list
    for i, inp in enumerate(input):
        new_updates = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are an iterative knowledge graph builder.
                    You are given the current state of the graph, and you must append the nodes and edges 
                    to it Do not procide any duplcates and try to reuse nodes as much as possible.""",
                },
                {
                    "role": "user",
                    "content": f"""Extract any new nodes and edges from the following:
                    # Part {i}/{len(input)} of the input:

                    {inp}""",
                },
                {
                    "role": "user",
                    "content": f"""Here is the current state of the graph:
                    {cur_state.model_dump_json(indent=2)}""",
                },
            ],
            response_model=KnowledgeGraph,
        )  # type: ignore

        # Update the current state with the new updates
        cur_state = cur_state.update(new_updates)

        # Draw the current state of the graph
        cur_state.visualize_knowledge_graph() 
        
    # Return the final state of the KnowledgeGraph
    return cur_state