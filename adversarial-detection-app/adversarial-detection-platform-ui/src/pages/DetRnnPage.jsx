import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function DetRnnPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="detector"
      modelKey="rnn"
      title="RNN Attack Detector"
      subtitle=""
    />
  );
}
