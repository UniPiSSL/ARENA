import React from "react";
import ModelOrDetectorCard from "../components/ModelOrDetectorCard";

export default function DetCnnTeacherPage({ apiBase }) {
  return (
    <ModelOrDetectorCard
      apiBase={apiBase}
      variant="detector"
      modelKey="cnn_teacher"
      title="CNN (Teacher) Attack Detector"
      subtitle=""
    />
  );
}
