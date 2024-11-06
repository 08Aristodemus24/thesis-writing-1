import { createContext } from "react";

// how can I make other components use this by providing only the style and theme value in the provider and then
// accessing this object of styles by other components without having to place this object in each file
// e.g. in name component because style and theme is provided I will have to access the styles dictionary values
// through styles[style][theme]
export const DesignsContext = createContext();

export const DesignsProvider = ({ children }) => {
    return (
        <DesignsContext.Provider value={{
            'sharp-minimal': {
                dark: {
                    '--primary-color': "white",
                    '--secondary-color': "black",
                    '--tertiary-color': "rgba(255, 255, 255, 0.267)",
                    '--primary-background': "black",
                },
                light: {
                    '--primary-color': "black",
                    '--secondary-color': "white",
                    '--tertiary-color': "rgba(0, 0, 0, 0.267)",
                    '--primary-background': "white",
                }
            },
            'light-neomorphic': {
                '--primary-color': "black",
                '--secondary-color': "rgb(231, 238, 246)",
                '--tertiary-color': "rgba(0, 0, 0, 0.267)",
                '--primary-shadow': "rgba(0, 0, 0, 0.25)",
                '--secondary-shadow': "rgba(255, 255, 255, 0.5)",
                '--primary-background': "rgb(231, 238, 246)",
            },
            'dark-neomorphic': {
                '--primary-color': "rgb(231, 238, 246)",
                '--secondary-color': "rgb(38, 39, 43)",
                '--tertiary-color': "rgba(255, 255, 255, 0.267)",
                '--primary-shadow': "rgba(0, 0, 0, 0.25)",
                '--secondary-shadow': "rgba(210, 210, 210, 0.5)",
                '--primary-background': "rgb(38, 39, 43)",
            }
          }}>
            {children}
        </DesignsContext.Provider>
    );
}

