import { useContext, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

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
    let src = sprSheetObj != null ? sprSheetObj.length != 0 ? URL.createObjectURL(sprSheetObj[0]) : null : null;

    const handleUpload = (event) => {
        setSprSheetObj(event.target.files);
        setSprSheet(event.target.files[0]);
        console.log('image uploaded');
    }

    const toggle = design.includes('neomorphic') ? (event) => {
        console.log(event.target.classList);
        if(event.target.classList.contains('clicked')){
            event.target.classList.remove('clicked');
        }else{
            event.target.classList.add('clicked');
        }
    } : null;

    return (
        <div className={`file-upload-container ${design}`} style={style}>
            {/* when file is uploaded files becomes are not null anymore
            but when another upload occurs and is cancelled files becomes
            a list of length 0 */}
            <img className={`uploaded-file ${design}`} src={src} alt=" "/>
            <div className="file-upload-field-wrapper">
                <label htmlFor="file-upload" className="file-upload-label">file</label>    
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