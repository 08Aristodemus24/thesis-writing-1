import { ThemeContext } from './contexts/ThemeContext';
import { DesignsProvider } from './contexts/DesignsContext';

import Correspondence from './components/Correspondence';
import Navbar from './components/Navbar';
import About from './components/About';
import Researchers from './components/Researchers';
import FAQ from './components/FAQ';
import './App.css';
import './navbar-862-and-up.css';
import './navbar-862-down.css';

function App() {

  return (
    <DesignsProvider>
      {/* values can be:
      design: 'sharp-minimal' theme: 'light', 
      design: 'light-neomorphic',
      design: 'dark-neomorphic' */}
      <ThemeContext.Provider value={{design: "sharp-minimal", theme: 'light'}}>
        <Navbar/>
        <Correspondence/>
        <About/>
        <Researchers/>
        <FAQ/>
      </ThemeContext.Provider>  
    </DesignsProvider> 
  );
}

export default App