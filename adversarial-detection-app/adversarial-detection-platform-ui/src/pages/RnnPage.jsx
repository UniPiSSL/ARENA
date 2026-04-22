import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function RnnPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="classifier"
      modelKey="rnn"
      title="RNN Classifier "
      subtitle=""
    />
  );
}
