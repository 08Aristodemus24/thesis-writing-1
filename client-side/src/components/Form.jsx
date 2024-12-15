import ModelNameInput from './ModelNameInput';
import SpreadSheetInput from './SpreadSheetInput';
import InputGroup from './InputGroup';
import Checkbox from './Checkbox';
import Button from "./Button";

import { useContext, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";
import Visualizer from './Visualizer';
import Alert from './Alert';


export default function Form(){
    let [modelName, setModelName] = useState("");
    let [modelNames, setModelNames] = useState([]);

    let [sprSheetFile, setSprSheetFile] = useState(null);
    let [initSprSheet, setInitSprSheet] = useState([]);

    let [showRaw, setShowRaw] = useState(true);
    let [showCorrect, setShowCorrect] = useState(false);
    let [showArt, setShowArt] = useState(true);
    let [showStressLevels, setShowStressLevels] = useState(false);

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

    // send a post request and retrieve the response and set 
    // the state of the following states for the alert component
    let [response, setResponse] = useState(null);
    let [msgStatus, setMsgStatus] = useState();
    let [errorType, setErrorType] = useState(null);
    const handleSubmit = async (event) => {
        try {
            event.preventDefault();
            const form_data = new FormData();
            form_data.append('model_name', modelName);
            // form_data.append('spreadsheet', sprSheet);
            form_data.append('spreadsheet_file', sprSheetFile)
            form_data.append('show_raw', showRaw);
            form_data.append('show_correct', showCorrect);
            form_data.append('show_art', showArt);
            form_data.append('show_stress_levels', showStressLevels);

            // once data is validated submitted and then extracted
            // reset form components form element
            setModelName(modelNames.length != 0 ? modelNames[0] : "");
            // setInitSprSheet([]);
            setShowRaw(true);
            setShowCorrect(false);
            setShowArt(true);
            setShowStressLevels(false);

            // send here the data from the contact component to 
            // the backend proxy server
            // // for development
            const url = 'http://127.0.0.1:5000/send-data';
            // for production
            // const url = 'https://project-alexander.vercel.app/send-data';

            const resp = await fetch(url, {
                'method': 'POST',
                'body': form_data,
            });
            const data = await resp.json();
            setResponse(resp);
            setInitSprSheet(data['corrected_df']);

            // if response.status is 200 then that means contact information
            // has been successfully sent to the email.js api
            if(resp.status === 200){
                setMsgStatus("success");
                console.log(`message has been sent with code ${resp.status}`);

            }else{
                setMsgStatus("failure");
                console.log(`message submission unsucessful. Response status '${resp.status}' occured`);
            }

        }catch(error){
            setMsgStatus("denied");
            setErrorType(error);
            console.log(`Submission denied. Error '${error}' occured`);
        }
    };

    // console.log(`response: ${response}`);
    // console.log(`message status: ${msgStatus}`);
    // console.log(`error type: ${errorType}`)

    return (
        <FormInputsContext.Provider value={{
            modelNames, setModelNames,
            modelName, setModelName,
            sprSheetFile, setSprSheetFile,
            initSprSheet, setInitSprSheet,
            // finalSprSheet, setFinalSprSheet,
            showRaw, setShowRaw,
            showCorrect, setShowCorrect,
            showArt, setShowArt,
            showStressLevels, setShowStressLevels,
            
            response,
            msgStatus, setMsgStatus,
            errorType, 
            handleSubmit,
        }}>
            <div className="form-container">
                <form
                    className={`form ${design}`}
                    style={style}
                    method="POST"
                >
                    <ModelNameInput/>
                    <SpreadSheetInput/>
                    <InputGroup>
                        <Checkbox label="show raw" name="show_raw"/>
                        <Checkbox label="show correct" name="show_correct"/>
                        <Checkbox label="show artifact" name="show_artifact"/>
                        <Checkbox label="show stress levels" name="show_stress_levels"/>
                    </InputGroup>
                    <Button/>
                </form>
                <Alert/>
                <Visualizer/>
            </div>
        </FormInputsContext.Provider>
    );
}

const palette = {
    dark: {
        primary_color: "rgb(38,39,43)",
        secondary_color: "white",
        tertiary_color: "rgba(0, 0, 0, 0.267)"    
    },
    light: {
        primary_color: "rgb(231, 238, 246)",
        secondary_color: "black",
        tertiary_color: "rgba(255, 255, 255, 0.267)"    
    }
};