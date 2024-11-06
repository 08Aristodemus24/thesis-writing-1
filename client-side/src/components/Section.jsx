import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";

export default function Section({ 'section-name': name, 'children': children }){
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
        <section id={`${name}-section`} className={`${name} ${design}`} style={style}>
            <div className={`${name}-content`}>
                {children}
            </div>
        </section>
    );
}