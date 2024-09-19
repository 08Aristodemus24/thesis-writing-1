import NameInput from "./NameInput";
import EmailInput from './EmailInput';
import MobileNumberInput from './MobileNumberInput';
import CountryCodeInput from './CountryCodeInput';
import MessageInput from './MessageInput';
import ModelNameInput from './ModelNameInput';
import PromptInput from './PromptInput';
import SequenceLengthInput from './SequenceLengthInput';
import TemperatureInput from './TemperatureInput';
import ImageInput from './ImageInput';
import SpreadsheetInput from './SpreadsheetInput';
import Button from "./Button";

import { useContext, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";


export default function Form(){
    let [fname, setFname] = useState("");
    let [lname, setLname] = useState("");
    let [email, setEmail] = useState("");
    let [mobileNum, setMobileNum] = useState("");
    let [countryCode, setCountryCode] = useState("");
    let [message, setMessage] = useState("");
    let [modelName, setModelName] = useState("");
    let [prompt, setPrompt] = useState("");
    let [seqLen, setSeqLen] = useState("250");
    let [temperature, setTemperature] = useState("1.0");
    let [image, setImage] = useState(null);

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
            form_data.append('first_name', fname);
            form_data.append('last_name', lname);
            form_data.append('email_address', email);
            form_data.append('country_code', mobileNum);
            form_data.append('mobile_num', countryCode);
            form_data.append('message', message);
            form_data.append('model_name', modelName);
            form_data.append('prompt', prompt);
            form_data.append('seq_len', seqLen);
            form_data.append('temperature', temperature);
            form_data.append('image', image);

            // once data is validated submitted and then extracted
            // reset form components form element
            setFname("");
            setLname("");
            setEmail("");
            setMobileNum("");
            setCountryCode("");
            setMessage("");
            setModelName("");
            setPrompt("");
            setSeqLen("250");
            setTemperature("1.0");
            setImage(null);

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
            setResponse(resp);

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

    console.log(`response: ${response}`);
    console.log(`message status: ${msgStatus}`);
    console.log(`error type: ${errorType}`)

    return (
        <FormInputsContext.Provider value={{
            fname, setFname, 
            lname, setLname, 
            email, setEmail, 
            mobileNum, setMobileNum, 
            countryCode, setCountryCode,
            message, setMessage,
            modelName, setModelName,
            prompt, setPrompt,
            seqLen, setSeqLen,
            temperature, setTemperature,
            image, setImage,
            handleSubmit,
        }}>
            <div className="form-container">
                <form
                    className={`form ${design}`}
                    style={style}
                    method="POST"
                >
                    <NameInput name-type="first"/>
                    <NameInput name-type="last"/>
                    <EmailInput/>
                    <MobileNumberInput/>
                    <CountryCodeInput/>
                    <MessageInput/>
                    <ModelNameInput/>
                    <PromptInput/>
                    <SequenceLengthInput/>
                    <TemperatureInput/>
                    <ImageInput/>
                    <SpreadsheetInput/>
                    <Button/>
                </form>
                <div className={`alert ${msgStatus !== undefined ? 'show' : ''}`} onClick={(event) => {
                    // remove class from alert container to hide it again
                    event.target.classList.remove('show');
                
                    // reset msg_status to undefined in case of another submission
                    setMsgStatus(undefined);
                }}>
                    <div className="alert-wrapper">
                        {msgStatus === "success" || msgStatus === "failed" ? 
                        <span className="alert-message">Message has been sent with code {response?.status}</span> : 
                        <span className="alert-message">Submission denied. Error {errorType?.message} occured</span>}
                    </div>
                </div>
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