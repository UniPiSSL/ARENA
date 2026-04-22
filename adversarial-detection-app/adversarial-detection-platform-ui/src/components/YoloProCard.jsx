import React, { useState } from "react";
import axios from "axios";
import {
  Box, Card, CardHeader, CardContent, CardActions, Divider, Typography, Stack,
  Slider, Chip, Tooltip, IconButton, Button, Input, CircularProgress,
  FormControl, FormLabel, RadioGroup, FormControlLabel, Radio, Switch,
  Checkbox, FormGroup, Select, MenuItem, TextField, Grid
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import PhotoFilterIcon from "@mui/icons-material/PhotoFilter";
import TimelineIcon from "@mui/icons-material/Timeline";
import MemoryIcon from "@mui/icons-material/Memory";
import ScatterPlotIcon from "@mui/icons-material/ScatterPlot";

const ICONS = { ann: <TimelineIcon />, cnn: <MemoryIcon />, rnn: <ScatterPlotIcon /> };
const CLASSIFIERS = [
  { key: "ann", label: "ANN" },
  { key: "cnn", label: "CNN" },
  { key: "rnn", label: "RNN" },
  { key: "cnn_semi", label: "CNN (Semi)" },
  { key: "cnn_teacher", label: "CNN (Teacher)" },
];
const DETECTORS = CLASSIFIERS;

function b64ToBlob(base64, mime = "image/png") {
  const byteString = atob(base64);
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
  return new Blob([ab], { type: mime });
}

function DetectorChip({ score = 0, thr = 0.5 }) {
  const adv = score >= thr;
  const label = adv ? `Adversarial • ${(score * 100).toFixed(1)}%` : `Clean • ${(score * 100).toFixed(1)}%`;
  return <Chip size="small" label={label} color={adv ? "error" : "success"} variant={adv ? "filled" : "outlined"} />;
}

function ensembleDecision(votes, mode) {
  const n = votes.length;
  const sum = votes.reduce((a, b) => a + (b ? 1 : 0), 0);
  if (mode === "or") return sum >= 1;
  if (mode === "and") return sum === n;
  return sum >= Math.ceil(n / 2);
}

export default function YoloProCard({ apiBase }) {
  // Inputs
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  // YOLO params
  const [conf, setConf] = useState(0.10);
  const [iou, setIou] = useState(0.45);
  const [imgsz, setImgSz] = useState(640);
  const [expand, setExpand] = useState(0.10);
  const [square, setSquare] = useState(true);
  const [cropSize, setCropSize] = useState(50);
  const [extras, setExtras] = useState(false);

  // classifiers/detectors (crops)
  const [isNight, setIsNight] = useState("0");
  const [useClf, setUseClf] = useState(false);
  const [selectedClf, setSelectedClf] = useState([]); // 1..3
  const [useDet, setUseDet] = useState(false);
  const [selectedDet, setSelectedDet] = useState([]); // 1..3
  const [aggMode, setAggMode] = useState("or");
  const [detThr, setDetThr] = useState(0.5);

  // Results
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [yoloImage, setYoloImage] = useState(null);
  const [detections, setDetections] = useState([]); // [{...yolo, crop:{resized_base64}}, ...]
  const [perCropEnriched, setPerCropEnriched] = useState([]); // [{clf:[...], det:{...}} per detection]

  const FIELD_W = 160;

  const onFile = (e) => {
    const f = e.target.files?.[0];
    setError(null);
    setYoloImage(null);
    setDetections([]);
    setPerCropEnriched([]);
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const toggleSel = (arr, setArr, key) => {
    setArr((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  };

  const endpoints = {
    clf: {
      ann: `${apiBase}/predict-ann`,
      cnn: `${apiBase}/predict-cnn`,
      rnn: `${apiBase}/predict-rnn`,
    },
    det: {
      ann: `${apiBase}/detect-ann-attack`,
      cnn: `${apiBase}/detect-cnn-attack`,
      rnn: `${apiBase}/detect-rnn-attack`,
      cnn_semi:    `${apiBase}/detect-cnn-semi-attack`,
      cnn_teacher: `${apiBase}/detect-cnn-teacher-attack`,
    },
  };

  const runYolo = async () => {
    if (!file) { setError("❌ Select an image."); return; }
    setLoading(true); setError(null);
    setPerCropEnriched([]);
    try {
      const form = new FormData();
      form.append("image", file);
      const params = {
        conf, iou, imgsz,
        return_crops: 1,
        expand, square: square ? 1 : 0,
        crop_size: cropSize,
        extras: extras ? 1 : 0,
      };
      const yres = await axios.post(`${apiBase}/yolo-predict`, form, { params, timeout: 45000 });
      const yimg = yres?.data?.yolo_image || null;
      const dets = yres?.data?.detections || [];

      setYoloImage(yimg);
      setDetections(dets);

      const out = [];
      for (const d of dets) {
        const cropB64 = d?.crop?.resized_base64 || d?.crop?.base64;
        let clfRes = [];
        let detRes = null;

        if (cropB64 && (useClf || useDet) && (selectedClf.length || selectedDet.length)) {
          const blob = b64ToBlob(cropB64, "image/png");
          const form2 = new FormData();
          form2.append("image", blob, "crop.png");
          form2.append("isNight", isNight);

          // classifiers
          if (useClf && selectedClf.length) {
            const promises = selectedClf.map(async (key) => {
              try {
                const r = await axios.post(endpoints.clf[key], form2, { timeout: 20000 });
                return { key, ok: true, data: r.data };
              } catch (e) {
                return { key, ok: false, data: { predicted_class: "error", confidence: 0 } };
              }
            });
            clfRes = await Promise.all(promises);
          }

          // detectors 
          if (useDet && selectedDet.length) {
            const promises = selectedDet.map(async (key) => {
              try {
                const r = await axios.post(endpoints.det[key], form2, { timeout: 20000 });
                const score = r?.data?.score ?? 0;
                const byThr = score >= detThr;
                return { key, score, byThr };
              } catch (_) {
                return { key, score: 0, byThr: false, error: true };
              }
            });
            const per = await Promise.all(promises);
            const votes = per.map((x) => x.byThr);
            const positive = votes.length ? ensembleDecision(votes, aggMode) : false;
            detRes = {
              threshold: detThr,
              mode: aggMode,
              selected: selectedDet.slice(),
              perDetector: per,
              ensemble: { positive, votes: votes.filter((v) => v).length, total: votes.length },
            };
          }
        }

        out.push({ clf: clfRes, det: detRes });
      }
      setPerCropEnriched(out);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader
        avatar={<PhotoFilterIcon />}
        title={<Typography variant="h6">YOLO Detection</Typography>}
        subheader={<Typography variant="body2" color="text.secondary">Traffic light detection + optional classification & attack pre-filter on crops.</Typography>}
        action={
          <Tooltip title="Υποστηριζόμενα: PNG, JPG, WEBP, BMP">
            <IconButton><InfoOutlinedIcon /></IconButton>
          </Tooltip>
        }
      />
      <Divider />

      <CardContent>
        <Stack spacing={3}>
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 360px" }, gap: 2 }}>
            {/* Inputs */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>Input</Typography>
              <Stack spacing={1.5}>
                <Input type="file" accept="image/*" onChange={onFile} />
                {preview && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">Preview:</Typography>
                    <Box component="img" src={preview} alt="preview"
                      sx={{ width: 200, height: "auto", borderRadius: 1, border: "1px solid", borderColor: "divider", mt: 0.5 }} />
                  </Box>
                )}
                <Box>
                  <Button
                    variant="contained"
                    onClick={runYolo}
                    disabled={loading || !file}
                    startIcon={loading ? <CircularProgress size={16} /> : null}
                  >
                    {loading ? "Execution..." : "Run YOLO"}
                  </Button>
                </Box>
                {error && <Typography color="error" variant="body2">{error}</Typography>}
              </Stack>
            </Box>

            {/* Params & crop options */}
            <Box>
              <Box sx={{ border: "1px dashed", borderColor: "divider", borderRadius: 2, p: 2 }}>
                <Typography variant="subtitle2">YOLO Parameters</Typography>
                <Box sx={{ mt: 1 }}>
                  <FormLabel>conf: {conf.toFixed(2)}</FormLabel>
                  <Slider min={0} max={1} step={0.01} value={conf} onChange={(_, v) => setConf(Array.isArray(v) ? v[0] : v)} />
                </Box>
                <Box sx={{ mt: 1 }}>
                  <FormLabel>iou: {iou.toFixed(2)}</FormLabel>
                  <Slider min={0} max={1} step={0.01} value={iou} onChange={(_, v) => setIou(Array.isArray(v) ? v[0] : v)} />
                </Box>

                <Stack direction="row" spacing={2} sx={{ mt: 1 }} flexWrap="wrap">

                  <Grid container spacing={2} alignItems="flex-start">
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        size="small"
                        label="imgsz"
                        type="number"
                        InputLabelProps={{ shrink: true }}
                        value={imgsz}
                        onChange={(e) => setImgSz(Math.max(64, Number(e.target.value || 640)))}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        size="small"
                        label="expand"
                        type="number"
                        inputProps={{ step: "0.05" }}
                        InputLabelProps={{ shrink: true }}
                        value={expand}
                        onChange={(e) => setExpand(Math.max(0, Number(e.target.value || 0)))}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        size="small"
                        label="crop_size"
                        type="number"
                        InputLabelProps={{ shrink: true }}
                        value={cropSize}
                        onChange={(e) => setCropSize(Math.max(8, Number(e.target.value || 50)))}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        select
                        fullWidth
                        size="small"
                        label="Square"
                        InputLabelProps={{ shrink: true }}
                        value={square ? 1 : 0}
                        onChange={(e) => setSquare(Number(e.target.value) === 1)}
                      >
                        <MenuItem value={1}>On</MenuItem>
                        <MenuItem value={0}>Off</MenuItem>
                      </TextField>
                    </Grid>
                  </Grid>
                </Stack>
              </Box>

              {/* Crop post-processing options */}
              <Box sx={{ border: "1px dashed", borderColor: "divider", borderRadius: 2, p: 2, mt: 2 }}>
                <Typography variant="subtitle2">Edit Crops</Typography>

                <FormControl fullWidth sx={{ mt: 1 }}>
                  <FormLabel>Brightness (for classifiers/detectors)</FormLabel>
                  <RadioGroup row value={isNight} onChange={(e) => setIsNight(e.target.value)}>
                    <FormControlLabel value="0" control={<Radio size="small" />} label="☀️ Ημέρα" />
                    <FormControlLabel value="1" control={<Radio size="small" />} label="🌙 Νύχτα" />
                  </RadioGroup>
                </FormControl>

                {/* Classifiers */}
                <Stack direction="row" alignItems="center" spacing={1} justifyContent="space-between" sx={{ mt: 1 }}>
                  <Typography variant="body2">Classify crops</Typography>
                  <Switch checked={useClf} onChange={(e) => setUseClf(e.target.checked)} />
                </Stack>
                {useClf && (
                  <>
                    <Typography variant="body2" sx={{ mt: 0.5 }}>Select classifiers (1–3):</Typography>
                    <FormGroup row>
                      {CLASSIFIERS.map((c) => (
                        <FormControlLabel
                          key={c.key}
                          control={
                            <Checkbox
                              size="small"
                              checked={selectedClf.includes(c.key)}
                              onChange={() => toggleSel(selectedClf, setSelectedClf, c.key)}
                            />
                          }
                          label={c.label}
                        />
                      ))}
                    </FormGroup>
                  </>
                )}

                {/* Detectors */}
                <Stack direction="row" alignItems="center" spacing={1} justifyContent="space-between" sx={{ mt: 1 }}>
                  <Typography variant="body2">Attack pre-filter for crops</Typography>
                  <Switch checked={useDet} onChange={(e) => setUseDet(e.target.checked)} />
                </Stack>
                {useDet && (
                  <>
                    <Typography variant="body2" sx={{ mt: 0.5 }}>Select detectors (1–3):</Typography>
                    <FormGroup row>
                      {DETECTORS.map((d) => (
                        <FormControlLabel
                          key={d.key}
                          control={
                            <Checkbox
                              size="small"
                              checked={selectedDet.includes(d.key)}
                              onChange={() => toggleSel(selectedDet, setSelectedDet, d.key)}
                            />
                          }
                          label={d.label}
                        />
                      ))}
                    </FormGroup>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mt: 1 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          Threshold: {(detThr * 100).toFixed(0)}%
                        </Typography>
                        <Slider
                          size="small"
                          min={0}
                          max={1}
                          step={0.01}
                          value={detThr}
                          onChange={(_, v) => setDetThr(Array.isArray(v) ? v[0] : v)}
                        />
                      </Box>
                      <FormControl size="small" sx={{ minWidth: 140 }}>
                        <FormLabel>Aggregation</FormLabel>
                        <Select value={aggMode} onChange={(e) => setAggMode(e.target.value)}>
                          <MenuItem value="or">OR (At least 1)</MenuItem>
                          <MenuItem value="and">AND (All)</MenuItem>
                          <MenuItem value="maj">Majority</MenuItem>
                        </Select>
                      </FormControl>
                    </Stack>

                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Rule: each detector ⇒ <strong>Adversarial if score ≥ Threshold</strong>. Ensemble: <strong>{aggMode.toUpperCase()}</strong>.
                    </Typography>
                  </>
                )}
              </Box>
            </Box>
          </Box>

          {/* Results */}
          {(yoloImage || detections.length > 0) && (
            <Box>
              {yoloImage && (
                <Box mb={2}>
                  <Typography variant="subtitle2">YOLO Annotated</Typography>
                  <Box
                    component="img"
                    src={`data:image/png;base64,${yoloImage}`}
                    alt="annotated"
                    sx={{ width: "100%", maxHeight: 460, objectFit: "contain", borderRadius: 1, border: "1px solid", borderColor: "divider", mt: 0.5 }}
                  />
                </Box>
              )}

              <Grid container spacing={2}>
                {detections.map((det, idx) => {
                  const crop = det?.crop || {};
                  const enrich = perCropEnriched[idx] || {};
                  const clfList = enrich.clf || [];
                  const detInfo = enrich.det || null;

                  return (
                    <Grid key={idx} item xs={12} md={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle2">Detection #{idx + 1}</Typography>
                          <Typography variant="body2" color="text.secondary">
                            {det.class_name} • {(det.confidence * 100).toFixed(1)}% • box: [
                            {(det.box || []).map((v) => (typeof v === "number" ? v.toFixed(1) : v)).join(", ")}]
                          </Typography>

                          {/* Crop */}
                          {(crop.resized_base64 || crop.base64) && (
                            <Box mt={1} display="flex" alignItems="center" gap={2}>
                              <Box>
                                <Typography variant="caption" color="text.secondary">
                                  Crop {cropSize}×{cropSize}
                                </Typography>
                                <Box
                                  component="img"
                                  src={`data:image/png;base64,${crop.resized_base64 || crop.base64}`}
                                  alt="crop"
                                  sx={{ width: cropSize, height: cropSize, borderRadius: 1, border: "1px solid", borderColor: "divider", mt: 0.5 }}
                                />
                              </Box>

                              <Box sx={{ flex: 1 }}>
                                {/* Classification results */}
                                {useClf && clfList.length > 0 && (
                                  <Box sx={{ mb: 1 }}>
                                    <Typography variant="caption" color="text.secondary">Classification</Typography>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 0.5 }}>
                                      {clfList.map((c) => {
                                        const cls = c?.data?.predicted_class ?? "—";
                                        const conf = c?.data?.confidence ?? 0;
                                        return (
                                          <Chip
                                            key={c.key}
                                            icon={ICONS[c.key]}
                                            label={`${c.key.toUpperCase()}: ${cls} (${(conf * 100).toFixed(1)}%)`}
                                            variant="outlined"
                                          />
                                        );
                                      })}
                                    </Stack>
                                  </Box>
                                )}

                                {/* Detector prefilter results */}
                                {useDet && detInfo && (
                                  <Box>
                                    <Typography variant="caption" color="text.secondary">Adversarial pre-filter</Typography>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 0.5 }}>
                                      {detInfo.perDetector.map((d) => (
                                        <Stack key={d.key} direction="row" spacing={0.5} alignItems="center">
                                          <Chip size="small" variant="outlined" label={d.key.toUpperCase()} />
                                          <DetectorChip score={d.score} thr={detInfo.threshold} />
                                        </Stack>
                                      ))}
                                      <Chip
                                        size="small"
                                        label={`${detInfo.ensemble.positive ? "Adversarial" : "Clean"} (votes ${detInfo.ensemble.votes}/${detInfo.ensemble.total}, ${detInfo.mode.toUpperCase()})`}
                                        color={detInfo.ensemble.positive ? "error" : "success"}
                                      />
                                    </Stack>
                                  </Box>
                                )}
                              </Box>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
            </Box>
          )}
        </Stack>
      </CardContent>

      <CardActions sx={{ px: 2, pb: 2, pt: 0 }}>
        <Typography variant="caption" color="text.secondary">
          "Tip: Open “Edit Crops” to label each crop with multiple classifiers or to filter attacks using ensemble detectors.
        </Typography>
      </CardActions>
    </Card>
  );
}
