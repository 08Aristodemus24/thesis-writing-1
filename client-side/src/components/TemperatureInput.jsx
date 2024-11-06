import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function TemperatureInput({ children }){
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
    // all its states we use the state appropriate to the TemperatureInput
    // component and its setter to set from this component the state of
    // the form
    let { temperature, setTemperature } = useContext(FormInputsContext);

    return (
        <div 
            className={`temp-container ${design}`} 
            style={style}>
            <label 
                htmlFor="temp" 
                className="temp-label">Temperature</label>
            <input 
                value={temperature} 
                type="range" 
                min={0.0} 
                max={2.0} 
                step={0.01} 
                id="field" 
                className={`temp-field ${design}`} 
                onChange={(event) => setTemperature(event.target.value)}
            />
        </div>
    );
}