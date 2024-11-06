import Section from './Section';  

export default function Correspondence(){

    return (
        <Section section-name={"faq"}>
            <h2 className="faq-header">Frequently asked questions</h2>
            <div className="grid-container">
                <div className="grid-col">
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            What is Electrodermal Activity (EDA)?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">EDA, often referred to as skin conductance, measures changes in the electrical conductivity of the skin. It's a physiological response linked to emotional arousal and stress.</p>
                    </div>
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            What are EDA artifacts?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">EDA signals can be contaminated by various artifacts, such as movement artifacts, sensor noise, and temperature fluctuations.</p>
                        <p className="text-gray-500 dark:text-gray-400">These artifacts can significantly impact the accuracy of stress detection and other analyses.</p>
                    </div>
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            How does your tool address EDA artifact recognition and correction?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">Our tool employs a hybridized LSTM-SVM model to effectively detect and correct EDA artifacts.</p>
                        <p className="text-gray-500 dark:text-gray-400">The LSTM component captures complex temporal dependencies in the EDA signal, while the SVM provides a robust classification mechanism.</p>                
                    </div>
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            Why combine LSTM and SVM?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">LSTM: Captures long-term dependencies in the time series data, making it suitable for complex patterns in EDA signals.</p>
                        <p className="text-gray-500 dark:text-gray-400">SVM: Provides a strong classification performance, especially in high-dimensional spaces, and is less prone to overfitting.</p>                
                    </div>
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            What are the limitations of your tool?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">Like any machine learning model, our tool's performance depends on the quality of the input data. Noisy data or artifacts that are significantly different from the training data may impact the accuracy of the model. Additionally, the tool's performance might be influenced by individual physiological differences.</p>
                    </div>
                    
                </div>
                <div className="grid-col">
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            How does your tool handle imbalanced datasets, which are common in EDA research?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">We address the issue of imbalanced datasets by employing techniques such as:</p>
                        <p>Class weighting: Assigning higher weights to the minority class during training.</p>
                        <p>Oversampling: Increasing the number of samples in the minority class through techniques like SMOTE.</p>
                        <p>Undersampling: Reducing the number of samples in the majority class.</p>
                    </div>
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            How does your tool handle different sampling rates?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">Our tool is designed to handle various sampling rates. The windowing technique used in the preprocessing stage is flexible and can adapt to different sampling rates.</p>
                    </div>
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            Can your tool be used for real-time applications?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">While our current implementation is focused on offline analysis, the underlying techniques can be adapted for real-time applications. However, real-time implementation would require careful consideration of computational efficiency and latency constraints.</p>
                    </div>
                    <div className="mb-10">
                        <h3 className="flex items-center mb-4 text-lg font-medium text-gray-900 dark:text-white">
                            <div className="bullet"></div>
                            How can researchers use your tool?
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400">Researchers can use our tool to:

                            Preprocess EDA data to remove artifacts.
                            Extract relevant features from the cleaned data.
                            Train and evaluate machine learning models for stress detection and other applications.
                            Gain insights into the underlying physiological mechanisms of stress and emotion.
                        </p>
                    </div>
                </div>
            </div>
            
            

            
            

            
            

            
            
            
            
            

            
            
            

            
            

            
            

            
            
        </Section>
    )
}