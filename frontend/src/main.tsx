import { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import Routes from "./navigation/routes";
import { StoreProvider } from "./store";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <StoreProvider>
      <Routes />
    </StoreProvider>
  </StrictMode>
);
