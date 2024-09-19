import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function Message({ children }){
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
    // all its states we use the state appropriate to the MessageInput
    // component and its setter to set from this component the state of
    // the form
    let { message, setMessage } = useContext(FormInputsContext);

    return (
        <div 
            className={`message-container ${design}`}
            style={style}
        >
            <label 
                htmlFor="message" 
                className="message-label"
            >Message</label>
            <textarea 
                id="message" 
                rows="5" 
                name="message" 
                className={`message-field ${design}`} 
                placeholder="Your message here" 
                onChange={(event) => setMessage(event.target.value)} 
                value={message}
                required
            />
        </div>
    );
}