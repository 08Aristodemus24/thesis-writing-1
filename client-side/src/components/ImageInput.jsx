import { useContext, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function ImageInput({ children }){
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
    let { image, setImage } = useContext(FormInputsContext);
    let [imageObj, setImageObj] = useState(null);
    let src = imageObj != null ? imageObj.length != 0 ? URL.createObjectURL(imageObj[0]) : null : null;

    const handleUpload = (event) => {
        setImageObj(event.target.files);
        setImage(event.target.files[0]);
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
        <div className={`image-upload-container ${design}`} style={style}>
            {/* when image is uploaded images becomes are not null anymore
            but when another upload occurs and is cancelled images becomes
            a list of length 0 */}
            <img className={`uploaded-image ${design}`} src={src} alt=" "/>
            <div className="image-upload-field-wrapper">
                <label htmlFor="image-upload" className="image-upload-label">Image</label>    
                <input 
                    type="file" 
                    accept="image/*" 
                    id="image-upload" 
                    className={`image-upload-field ${design}`} 
                    onChange={handleUpload} 
                    onMouseDown={toggle} 
                    onMouseUp={toggle}
                />
            </div>
        </div>
    );
}