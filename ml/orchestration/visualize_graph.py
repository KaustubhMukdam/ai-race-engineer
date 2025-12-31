"""
Visualize the multi-agent workflow graph
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.orchestration.race_strategy_agent import RaceStrategyGraph

def visualize_workflow():
    """Generate workflow visualization"""
    print("🎨 Generating multi-agent workflow visualization...")
    
    # Create graph
    graph = RaceStrategyGraph()
    
    # Generate visualization
    try:
        from IPython.display import Image, display
        
        # Get the graph visualization
        png_data = graph.graph.get_graph().draw_mermaid_png()
        
        # Save to file
        output_path = Path(__file__).parent.parent.parent / "docs" / "multi_agent_workflow.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(png_data)
        
        print(f"✅ Visualization saved to: {output_path}")
        
        # Also save Mermaid diagram
        mermaid_output = output_path.with_suffix('.mmd')
        mermaid_code = graph.graph.get_graph().draw_mermaid()
        
        with open(mermaid_output, "w") as f:
            f.write(mermaid_code)
        
        print(f"✅ Mermaid diagram saved to: {mermaid_output}")
        
    except Exception as e:
        print(f"⚠️  Could not generate PNG (install graphviz): {e}")
        print("Generating Mermaid diagram instead...")
        
        # Fallback: Just save Mermaid
        output_path = Path(__file__).parent.parent.parent / "docs" / "multi_agent_workflow.mmd"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mermaid_code = graph.graph.get_graph().draw_mermaid()
        
        with open(output_path, "w") as f:
            f.write(mermaid_code)
        
        print(f"✅ Mermaid diagram saved to: {output_path}")
        print("💡 View it at: https://mermaid.live/")

if __name__ == "__main__":
    visualize_workflow()
