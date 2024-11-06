import { useContext, useEffect, useRef, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function ModelNameInput({ children }){
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
    // all its states we use the state appropriate to the ModelNameInput
    // component and its setter to set from this component the state of
    // the form
    let { modelNames, setModelNames, modelName, setModelName } = useContext(FormInputsContext);

    
    const get_model_names = async () => {
        try{
            const url = 'http://127.0.0.1:5000/model-names';
            const response = await fetch(url);

            if(response.status === 200){
                console.log("retrieval successful");
                const data = await response.json();

                // returned data consists of key value pairs 
                // particularly the model_names and the list of names
                setModelNames([...data['model_names']]);

                // on mount set state of model_name to 
                // first model_name in model_names list
                setModelName(data['model_names'][0]);
            }else{
                console.log(`retrieval unsuccessful. Response status ${response.status} occured`)
            }
        }catch(error){
            console.log(`Server access denied. Error '${error}' occured`);
        }
    };

    useEffect(() => {
        get_model_names();
    }, []);

    return (
        <div 
            className={`model-name-container ${design}`} 
            style={style}
        >
            <label 
                htmlFor="model-name" 
                className="model-name-label"
            >Model Name</label>
            <select 
                name="model_name" 
                id="model-name" 
                className={`model-name-field ${design}`} 
                value={modelName} 
                onChange={(event) => setModelName(event.target.value)}
            >
                {modelNames.map((value, index) => {
                    return (<option
                        key={value}
                        value={value} 
                        label={value}>
                    </option>);
                })}
            </select>
        </div>
    );
}