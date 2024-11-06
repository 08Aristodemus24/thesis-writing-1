import { useContext } from "react";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function Alert({ children }){
    let { response, msgStatus, setMsgStatus, errorType } = useContext(FormInputsContext);
    return (
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
    );
}