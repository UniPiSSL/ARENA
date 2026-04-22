import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function CnnPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="classifier"
      modelKey="cnn"
      title="CNN Classifier"
      subtitle=""
    />
  );
}
