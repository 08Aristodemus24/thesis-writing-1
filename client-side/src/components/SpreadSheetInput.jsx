import { useContext, useEffect, useRef } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

import Button from "./Button";

import * as d3 from 'd3';

export default function SpreadSheetInput({ children }){

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
    // all its states we use the state appropriate to the ImageInput
    // component and its setter to set from this component the state of
    // the form
    let { initSprSheet, setInitSprSheet, sprSheetFile, setSprSheetFile } = useContext(FormInputsContext);

    const handleUpload = (event) => {
        event.preventDefault();

        // read .csv file
        let reader = new FileReader();
        let file = event.target.files[0];

        setSprSheetFile(file);
        reader.onload = (event) => {
            let csvToText = event.target.result;
            let output = csvToJSON(csvToText, ";");
            setInitSprSheet(output);
        };
        reader.readAsText(file);
    }

    const csvToJSON = (csv, delimiter) => {
        let lines = csv.split("\n");
        let result = [];
        let headers = ['time', 'raw_signal', 'clean_signal', 'label', 'auto_signal', 'pred_art', 'post_proc_pred_art'];
        
        for (let i = 1; i < lines.length; i++) {
            let obj = {};
    
            if(lines[i] == undefined || lines[i].trim() == "") {
                continue;
            }
    
            let words = lines[i].split(delimiter);
            for(let j = 0; j < words.length; j++) {
                obj[headers[j].trim()] = words[j];
            }
    
            result.push(obj);
        }
        return result;
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
        // console.log("spreadsheet uploaded");
        // console.log(initSprSheet);
    }, [initSprSheet])

    return (
        
        <div className={`spreadsheet-upload-container ${design}`} style={style}>
            <label htmlFor="spreadsheet-upload" className="spreadsheet-upload-label">File</label>    
            <input 
                type="file" 
                accept="file/*" 
                id="spreadsheet-upload" 
                className={`spreadsheet-upload-field ${design}`} 
                onChange={handleUpload}
                onMouseDown={toggle} 
                onMouseUp={toggle}
            />
        </div>
    );
}