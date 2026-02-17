from IPython.display import Image, display

__all__ = ["display_graph_png"]


def display_graph_png(graph):
    """Display a graph using IPython's display capabilities."""
    display(Image(graph.get_graph().draw_mermaid_png()))