import { useContext, useEffect, useRef, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function CountryCodeInput({ children }){
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
    // all its states we use the state appropriate to the CountryCodeInput
    // component and its setter to set from this component the state of
    // the form
    let { countryCode, setCountryCode } = useContext(FormInputsContext);

    let [countries, setCountries] = useState([]);
    const get_country_codes = async () => {
        try{
            const url = 'https://gist.githubusercontent.com/anubhavshrimal/75f6183458db8c453306f93521e93d37/raw/f77e7598a8503f1f70528ae1cbf9f66755698a16/CountryCodes.json';
            const response = await fetch(url);

            if(response.status === 200){
                console.log("retrieval successful");
                const data = await response.json();

                // returned data list consists of dictionaries containing
                // keys name, dial_code, and code e.g. 'Afghanistan', '+93', 'AF'
                setCountries([...data]);
                setCountryCode(data[0]['dial_code']);
            }else{
                console.log(`retrieval unsuccessful. Response status ${response.status} occured`)
            }
        }catch(error){
            console.log(`Server access denied. Error '${error}' occured`);
        }
    };

    useEffect(() => {
        get_country_codes();
    }, []);


    return (
        <div 
            className={`country-code-container ${design}`}             
            style={style}
        >
            <label 
            htmlFor="country-code" 
            className="country-code-label">Country Code</label>
            <select 
                name="country_code" 
                id="country-code" 
                className={`country-code-field ${design}`} 
                onChange={(event) => setCountryCode(event.target.value)}
                value={countryCode}
            >
                {countries.map((value, index) => {
                    return (<option
                        key={value['name']}
                        value={value['dial_code']} 
                        label={`${value['name']} (${value['dial_code']})`} 
                        data-country-name={value['name']} 
                        data-country-code={value['code']} 
                        data-dial-code={value['dial_code']}>
                    </option>);
                })}
            </select>
        </div>
    );
}