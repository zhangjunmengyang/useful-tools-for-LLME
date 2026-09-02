import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { ColorSchemeProvider } from "./components/ColorSchemeProvider";
import { I18nProvider } from "./components/I18nProvider";
import "./styles/globals.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ColorSchemeProvider>
      <I18nProvider>
        <App />
      </I18nProvider>
    </ColorSchemeProvider>
  </React.StrictMode>,
);
