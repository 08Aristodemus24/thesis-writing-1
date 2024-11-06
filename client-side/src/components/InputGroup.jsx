import { useContext } from "react";
import { DesignsContext } from "../contexts/DesignsContext";
import { ThemeContext } from "../contexts/ThemeContext";

export default function InputGroup(props){
    // console.log(props);

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

    return (
        <div className={`input-group-container ${design}`} style={style}>
            {props["children"]}
        </div>
    );
}