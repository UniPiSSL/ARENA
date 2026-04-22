import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function DetAnnPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="detector"
      modelKey="ann"
      title="ANN Attack Detector"
      subtitle=""
    />
  );
}
