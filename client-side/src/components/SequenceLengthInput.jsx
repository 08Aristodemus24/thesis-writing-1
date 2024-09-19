import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function SequenceLengthInput({ children }){
    // initialize and define theme of component by using
    // context
    let style;
    const designs = useContext(DesignsContext);
    const themes = useContext(ThemeContext);
    const { design, theme } = themes;
    
    // sometimes themes context will contain only the design 
    // and not the theme key so check if theme key is in themes
    if('theme' in themes){
        style = designs[design][theme];
    }else{
        style = designs[design];
    }

    // based on the context provider of wrapped Form containing
    // all its states we use the state appropriate to the SequenceLengthInput
    // component and its setter to set from this component the state of
    // the form
    let { seqLen, setSeqLen } = useContext(FormInputsContext);

    return (
        <div className={`seq-len-container ${design}`} style={style}>
            <label htmlFor="seq-len" className="seq-len-label">Sequence Length</label>
            <input 
                value={seqLen} 
                type="number" 
                id="seq-len" 
                className={`seq-len-field ${design}`} 
                min={0.0} 
                placeholder="250" 
                onChange={(event) => setSeqLen(event.target.value)}
            />
        </div>
    );
}