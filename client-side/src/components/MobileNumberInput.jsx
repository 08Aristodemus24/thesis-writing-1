import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function MobileNumberInput({ children }){
    // regex of mobile number field to follow
    const phone_reg = `[0-9]{'{'}3{'}'}-[0-9]{'{'}3{'}'}-[0-9]{'{'}4{'}'}`;

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
    // all its states we use the state appropriate to the MobileNumberInput
    // component and its setter to set from this component the state of
    // the form
    let { mobileNum, setMobileNum } = useContext(FormInputsContext);

    return (
        <div className={`mobile-num-container ${design}`} 
            style={style}
        >
            <label htmlFor="mobile-number" className="mobile-num-label">Phone</label>
            <input type="tel" pattern={phone_reg} name="mobile_number" id="mobile-number" className={`mobile-num-field ${design}`} placeholder="XXX-XXX-XXXX" value={mobileNum} onChange={(event) => setMobileNum(event.target.value)}/>
        </div>
    );
}