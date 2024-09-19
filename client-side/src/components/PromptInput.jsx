import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function PromptInput({ children }){
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
    // all its states we use the state appropriate to the PromptInput
    // component and its setter to set from this component the state of
    // the form
    let { prompt, setPrompt } = useContext(FormInputsContext);

    return (
        <div className={`prompt-container ${design}`} style={style}
        >
            <label htmlFor="prompt" className="prompt-label">Prompt</label>
            <input
                value={prompt} 
                id="prompt" 
                className={`prompt-field ${design}`} 
                type="text" 
                placeholder="Type a prompt e.g. Dostoevsky"
                onChange={(event) => setPrompt(event.target.value)}
            />
        </div>
    );
}