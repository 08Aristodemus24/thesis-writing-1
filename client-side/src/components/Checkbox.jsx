import { useContext, useEffect } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function Checkbox(props){
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

    // console.log(props);

    // based on the context provider of wrapped Form containing
    // all its states we use the state appropriate to the ImageInput
    // component and its setter to set from this component the state of
    // the form
    const { showRaw, setShowRaw, showCorrect, setShowCorrect, showArt, setShowArt, showStressLevels, setShowStressLevels } = useContext(FormInputsContext);
    const check_setter = {
        show_raw: [showRaw, setShowRaw],
        show_correct: [showCorrect, setShowCorrect],
        show_artifact: [showArt, setShowArt],
        show_stress_levels: [showStressLevels, setShowStressLevels]
    };
    const [check, setter] = check_setter[props["name"]] ;

    const handleCheck = (event) => {
        // set boolean of check input to the
        // opposite of it
        setter((curr_state) => !curr_state);
    }

    const toggle = design.includes('neomorphic') ? (event) => {
        console.log(event.target.classList);
        if(event.target.classList.contains('clicked')){
            event.target.classList.remove('clicked');
        }else{
            event.target.classList.add('clicked');
        }
    } : null;

    useEffect(() => {
        // console.log(`show artifact: ${showArt}`);
        // console.log(`show correct: ${showCorrect}`);
        // console.log(`show raw: ${showRaw}`);
    }, [showArt, showCorrect, showRaw]);

    return (
        <div className={`checkbox-container ${design}`} style={style}>
                
            <input 
                type="checkbox"  
                id="checkbox"
                // if for example user wants checkbox to show artifacts
                // then name of input must be show_artifacts, so when 
                // processed by server dictioonary can be easily accessed 
                // through show_artifacts key
                name={`${props["name"]}`}
                className={`checkbox-field ${design}`} 
                checked={check}
                value={check}
                onChange={handleCheck}
                onMouseDown={toggle} 
                onMouseUp={toggle}
            />
            <label htmlFor="checkbox" className="checkbox-label">{props["label"]}</label>
        </div>
    );
}