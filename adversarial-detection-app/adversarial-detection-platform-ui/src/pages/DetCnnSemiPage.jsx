import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function DetCnnSemiPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="detector"
      modelKey="cnn_semi"
      title="CNN (Semi) Attack Detector"
      subtitle=""
    />
  );
}
