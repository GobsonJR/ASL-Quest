import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { LandmarkDebugView } from "./LandmarkDebugView";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LandmarkDebugView />
  </StrictMode>
);
