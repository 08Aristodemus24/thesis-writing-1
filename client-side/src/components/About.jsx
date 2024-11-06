import Section from './Section';  

export default function Correspondence(){

    return (
        <Section section-name={"about"}>
            <h1>
                This study exclusively aimed to address the challenges present 
                in studies revolving around artifact recognition/detection by
                proposing a hybridized LSTM-SVM model, where the LSTM aims to 
                improve the shortcomings of traditional ML methods 
                of feature extraction from time series data, specifically 
                those involving EDA signals, by integrating higher order
                features in addition to lower order hand crafted features 
                used by the aforementioned studies, and wherein the integrated
                mechanism of traditional ML-based methods to the LSTM namely the
                Support Vector Machine (SVM) aims to equalize if not improve 
                further the difficult to interpret architectures of CNNs by 
                using instead much more simpler methods namely ML based ones 
                like the SVM. This system moreover not only classifies artifacts
                present in electrodermal activity data but also corrects and cleans
                the segments of the signals that do have artifacts which can 
                subsequently be used to classify arousal in the signals i.e. stress
                detection.
            </h1>
        </Section>
    )
}