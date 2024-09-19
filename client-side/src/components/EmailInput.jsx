import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function EmailInput({ children }){
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
    // all its states we use the state appropriate to the EmailInput
    // component and its setter to set from this component the state of
    // the form
    let { email, setEmail } = useContext(FormInputsContext);

    return (
        <div 
            className={`email-container ${design}`} 
            style={style}
        >
            <label 
                htmlFor="email-address" 
                className="email-label"
            >Email</label>
            <input 
                type="email" 
                name="email_address" 
                id="email-address" 
                className={`email-field ${design}`} 
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="johnmeyer87@gmail.com" 
                required
            />
        </div>
    );
}