import { ThemeContext } from './contexts/ThemeContext';
import { DesignsProvider } from './contexts/DesignsContext';

import Correspondence from './components/Correspondence';
import './App.css';
import './navbar-862-and-up.css';
import './navbar-862-down.css';

function App() {

  return (
    <DesignsProvider>
      <ThemeContext.Provider value={{design: "sharp-minimal", theme: 'light'}}>
        <Navbar/>
        <Correspondence/>
      </ThemeContext.Provider>  
    </DesignsProvider> 
  );
}

export default App