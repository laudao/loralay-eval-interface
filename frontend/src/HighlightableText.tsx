import React, { useEffect, useMemo, useState } from "react"
import { ComponentProps, Streamlit, withStreamlitConnection } from "streamlit-component-lib"
import { useStyletron } from "styletron-react"

/**
 * We can use a Typescript interface to destructure the arguments from Python
 * and validate the types of the input
 */
interface PythonArgs {
  text: string[]
  keywords: string[]
  highlighted: boolean[]
}

/**
 * No more props manipulation in the code.
 * We store props in state and pass value directly to underlying Slider
 * and then back to Streamlit.
 */
const HighlightableText = (props: ComponentProps) => {
  // Destructure using Typescript interface
  // This ensures typing validation for received props from Python
  const { text, keywords: _keywords, highlighted: _highlighted }: PythonArgs = props.args

  const keywords = useMemo(() => new Set(_keywords), [_keywords])
  console.log("Keywords:", keywords)
  const [highlighted, setHighlighted] = useState(_highlighted)
  const [css] = useStyletron();
  

  useEffect(() => Streamlit.setFrameHeight(), [])

  

  const getAnchorSpan = (node: Node, offset: number): HTMLElement|null => {
    if (node.nodeType == Node.TEXT_NODE) {
      const element = node.parentElement
      if (element?.tagName == "div") return node.previousSibling as HTMLElement
      return element
    }
    if (node.nodeType == Node.ELEMENT_NODE) {
      const element = (node as HTMLElement)
      console.log("Got an element", element)
      if (element.tagName == "span") return element
      
    }

    throw "Not handled"
  }
  const onMouseUp = () => {
    const selection = window.getSelection()
    if (selection && selection.anchorNode && selection.focusNode) {
      // Get the selection
      const start = getAnchorSpan(selection.anchorNode, selection.anchorOffset)?.getAttribute("data-ix")
      const end = getAnchorSpan(selection.focusNode, selection.focusOffset)?.getAttribute("data-ix")
      if (start && end) {
        let start_ix = parseInt(start)
        let end_ix =  parseInt(end)
        if (start_ix > end_ix) {
          const ix = end_ix
          end_ix = start_ix
          start_ix = ix
        }
        const new_highlighted = highlighted.map((h, ix) => 
          (ix >= start_ix && ix <= end_ix) ? !h : h
        )

        setHighlighted(new_highlighted)
        Streamlit.setComponentValue(new_highlighted)
        selection.empty()
      }
    }
  }


  return (
    <>
      <div onMouseUp={onMouseUp}>{text.map((word, word_ix) => 
        <span key={word_ix} data-ix={word_ix} className={css({
          background: highlighted[word_ix] ? "#ffffbf" :  "",
          color: keywords.has(word.toLowerCase()) ? "red" : "",
          fontWeight: keywords.has(word.toLowerCase()) ? "bold" : "normal"
        })}>{word} </span>
      )}</div>
    </>
  )
}

export default withStreamlitConnection(HighlightableText)
