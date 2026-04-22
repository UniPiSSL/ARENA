import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function AnnPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="classifier"
      modelKey="ann"
      title="ANN Classifier "
      subtitle=""
    />
  );
}
