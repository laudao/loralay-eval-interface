import os
import streamlit.components.v1 as components

from typing import List, Tuple



# Now the React interface only accepts an array of 1 or 2 elements.
_component_func = None

def init_components(dev: bool):
    global _component_func
    if dev:
        _component_func = components.declare_component(
            "highlightable_text",
            url="http://localhost:3000",
        )
    else:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(parent_dir, "frontend/build")
        _component_func = components.declare_component(
            "highlightable_text",
            path=build_dir
        )


# Edit arguments sent and result received from React component, so the initial input is converted to an array and returned value extracted from the component
def st_highlightable_text(text: str, highlighted: List[str], keywords: List[str], key=None) -> int:
    component_value = _component_func(text=text, highlighted=highlighted, keywords=keywords, key=key)
    return component_value
