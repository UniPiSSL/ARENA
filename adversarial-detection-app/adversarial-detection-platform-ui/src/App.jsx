import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";

import AnnPage from "./pages/AnnPage";
import CnnPage from "./pages/CnnPage";
import RnnPage from "./pages/RnnPage";
import YoloPage from "./pages/YoloPage";
import YoloSweep from "./pages/YoloSweep";
import BatchEval from "./pages/BatchEval";
import DetAnnPage from "./pages/DetAnnPage";
import DetCnnPage from "./pages/DetCnnPage";
import DetRnnPage from "./pages/DetRnnPage";
import DetCnnSemiPage from "./pages/DetCnnSemiPage";
import DetCnnTeacherPage from "./pages/DetCnnTeacherPage";
import DetEval from "./pages/DetEval";

const API_BASE = (
  import.meta.env.VITE_API_BASE || "http://localhost:5000"
).replace(/\/+$/, "");

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/yolo" replace />} />
        {/* Models */}
        <Route path="/ann" element={<AnnPage apiBase={API_BASE} />} />
        <Route path="/cnn" element={<CnnPage apiBase={API_BASE} />} />
        <Route path="/rnn" element={<RnnPage apiBase={API_BASE} />} />
        <Route path="/yolo" element={<YoloPage apiBase={API_BASE} />} />

        <Route path="/sweep-yolo" element={<YoloSweep apiBase={API_BASE} />} /> 
        <Route path="/batch-eval" element={<BatchEval apiBase={API_BASE} />} />
        <Route path="/det-eval" element={<DetEval apiBase={API_BASE} />} />
        {/* Detectors */}
        <Route path="/det-ann" element={<DetAnnPage apiBase={API_BASE} />} />
        <Route path="/det-cnn" element={<DetCnnPage apiBase={API_BASE} />} />
        <Route path="/det-rnn" element={<DetRnnPage apiBase={API_BASE} />} />
        <Route path="/det-semi-cnn" element={<DetCnnSemiPage apiBase={API_BASE} />} />
        <Route path="/det-teacher-cnn" element={<DetCnnTeacherPage apiBase={API_BASE} />} />
      </Routes>
    </Layout>
  );
}
