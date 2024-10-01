import { useContext, useEffect, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

import Papa from "papaparse";
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
    let { sprSheet, setSprSheet } = useContext(FormInputsContext);
    let [sprSheetObj, setSprSheetObj] = useState(null);

    const handleUpload = (event) => {
        event.preventDefault();
        let reader = new FileReader();
        let file = event.target.files[0];
        reader.onload = (event) => {
            let csvToText = event.target.result;
            let output = csvToJSON(csvToText);
            setSprSheet(output);
        };
        reader.readAsText(file);
    }

    const csvToJSON = (csv) => {
        let lines = csv.split("\n");
        let result = [];
        let headers;
        headers = lines[0].split(";");
    
        for (let i = 1; i < lines.length; i++) {
            let obj = {};
    
            if(lines[i] == undefined || lines[i].trim() == "") {
                continue;
            }
    
            let words = lines[i].split(";");
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
        console.log("state updated");
        console.log(sprSheet);
    });

    // const createGraph = async () => {
    //     console.log('test')
    //     let data = await d3.csv(sprSheet);

    //     data.forEach((row) => {
    //         console.log(row.rawdata);
    //     });
    // };
    

    return (
        <div className={`file-upload-container ${design}`} style={style}>
            {/* when file is uploaded files becomes are not null anymore
            but when another upload occurs and is cancelled files becomes
            a list of length 0 
            
            when the .csv file is uploaded we need someway to visualize
            the signals inside it. So we need to parse the uploaded .csv
            file into some kind of dataframe
            */}
            <div className="file-upload-field-wrapper">
                <label htmlFor="file-upload" className="file-upload-label">File</label>    
                <input 
                    type="file" 
                    accept="file/*" 
                    id="file-upload" 
                    className={`file-upload-field ${design}`} 
                    onChange={handleUpload}
                    onMouseDown={toggle} 
                    onMouseUp={toggle}
                />
            </div>
        </div>
    );
}