import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function DetCnnPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="detector"
      modelKey="cnn"
      title="CNN Attack Detector"
      subtitle=""
    />
  );
}
