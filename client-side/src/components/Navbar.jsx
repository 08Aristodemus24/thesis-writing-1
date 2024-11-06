import { useEffect, useState } from "react";

export default function Navbar({ children }){

  let [isOpened, setIsOpened] = useState(false)
  const body = document.body;
  
  // if div is closed then its class is .closed if opened then .opened
  const toggle_menu = (event) => {
    event.preventDefault();
    if(isOpened === false){
      setIsOpened(!isOpened);
      body.style.overflow = "hidden";

    }else{
      setIsOpened(!isOpened);
      body.style.overflow = "auto";
    }
  };

  // I don't want links in desktop mode to have access to the modal
  // if navbar is opened then only then can it be closed
  // but what if user opens modal and sets the dims to desktop
  // then when a tag is clicked modal will be closed
  const close_and_go = (event) => {
    event.preventDefault();
    if(isOpened === true){
      setIsOpened(!isOpened);
      body.style.overflow = "auto";
    }

    const section_id = event.target.classList[1];
    const section = document.querySelector(`#${section_id}`);
    section.scrollIntoView({
      block: 'start',
    });
  }

  return (
    <header className={`navbar-container ${isOpened === true ? "opened" : ""}`}>
        <nav className="navbar">
          <div className="nav-brand-container">
            <a className="navbar-brand" href="/" onClick={(event) => {
              event.preventDefault();
              document.body.scrollIntoView();
            }}>~</a>
            
            <div onClick={toggle_menu} className={`button-container ${isOpened === true ? "opened" : ""}`}>
                <a href="#" className="top-left-corner"></a>
                <a href="#" className="top-edge"></a>
                <a href="#" className="top-right-corner"></a>
                
                <a href="#" className="left-edge"></a>
                <a href="#" className="center"></a>
                <a href="#" className="right-edge"></a>
                
                <a href="#" className="bottom-left-corner"></a>
                <a href="#" className="bottom-edge"></a>
                <a href="#" className="bottom-right-corner"></a>
            </div>
          </div>
          
          <div className="nav-menu-container">
            <div className="nav-menu">
              <a className="nav-item about-section" aria-current="page" onClick={close_and_go}>ABOUT</a>
              <a className="nav-item researchers-section" onClick={close_and_go}>RESEARCHERS</a>
              <a className="nav-item faq-section" onClick={close_and_go}>FAQ</a>
            </div>
          </div>
        </nav>
    </header>
  );
}