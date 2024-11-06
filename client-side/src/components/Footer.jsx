import { useEffect, useState } from "react";
import dost_asti from '../assets/mediafiles/dost asti larger.png';
import dost from '../assets/mediafiles/dost seal larger.png';
import coare from '../assets/mediafiles/coare larger.png';

export default function Footer({ children }){

  return (
    <footer className="footer-container">
      <div className="footer-content">
          <div className="wrapper">
              <p className="footer-credits">Powered by DOST-ASTI COARE <a href="https://ph.linkedin.com/in/lara-marie-cueva-423925248">LinkedIn</a>.</p>
              <div className="footer-sponsors">
                <img src={dost_asti} alt="" className="dost-asti-logo"/>
                <img src={dost} alt="" className="dost-logo"/>
                <img src={coare} alt="" className="coare-logo"/>
              </div>
              <p className="footer-end">
                  2023 © by Michael Cueva. All rights reserved.
              </p>
          </div>
      </div>
    </footer>
  );
}