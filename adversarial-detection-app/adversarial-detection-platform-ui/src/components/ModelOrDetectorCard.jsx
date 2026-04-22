import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Divider,
  Typography,
  Stack,
  Slider,
  Chip,
  Tooltip,
  IconButton,
  Button,
  Input,
  CircularProgress,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Switch,
  Checkbox,
  FormGroup,
  Select,
  MenuItem,
  TextField,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import TimelineIcon from "@mui/icons-material/Timeline"; // ANN
import MemoryIcon from "@mui/icons-material/Memory"; // CNN
import ScatterPlotIcon from "@mui/icons-material/ScatterPlot"; // RNN

const ICONS = {
  ann: <TimelineIcon />,
  cnn: <MemoryIcon />,
  rnn: <ScatterPlotIcon />,
};
const DETECTORS = [
  { key: "ann", label: "ANN" },
  { key: "cnn", label: "CNN" },
  { key: "rnn", label: "RNN" },
  { key: "cnn_semi", label: "CNN (Semi)" },
  { key: "cnn_teacher", label: "CNN (Teacher)" },
];

function DetectorDecisionChip({ value = 0, thr = 0.5 }) {
  const adv = value >= thr;
  const label = adv
    ? `Adversarial • ${(value * 100).toFixed(1)}%`
    : `Clean • ${(value * 100).toFixed(1)}%`;
  const color = adv ? "error" : "success";
  return (
    <Chip
      size="small"
      label={label}
      color={color}
      variant={adv ? "filled" : "outlined"}
    />
  );
}

function ensembleDecision(votes, mode) {
  const n = votes.length;
  const sum = votes.reduce((a, b) => a + (b ? 1 : 0), 0);
  if (mode === "or") return sum >= 1;
  if (mode === "and") return sum === n;
  // majority
  return sum >= Math.ceil(n / 2);
}

/**
 * Unified card:
 *  - variant="classifier": classification + (optional) pre-filter with 1..3 detectors & OR/AND/Majority
 *  - variant="detector":   single detector (as before) with Threshold
 *
 * results (classifier + prefilter):
 *  { previewUrl, predicted_class, confidence,
 *    prefilter: {
 *      threshold, mode, selected: ["ann","cnn",...],
 *      perDetector: [{key, score, is_adversarial, byThr}],
 *      ensemble: {positive: boolean, votes: number, total: number}
 *    }
 *  }
 */
export default function ModelOrDetectorCard({
  apiBase,
  variant = "classifier", // "classifier" | "detector"
  modelKey = "ann", // "ann" | "cnn" | "rnn"
  title = "Model",
  subtitle = "Ομοιόμορφη κάρτα για predict/detect.",
}) {
  const isClassifier = variant === "classifier";

  const [detThreshold, setDetThreshold] = useState(0.5);

  const [prefilter, setPrefilter] = useState(false);
  const [selectedDets, setSelectedDets] = useState([]); // array of "ann"|"cnn"|"rnn"
  const [aggMode, setAggMode] = useState("or"); // "or" | "and" | "maj"

  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [isNight, setIsNight] = useState("0");
  const [loading, setLoading] = useState(false);
  const [inputSize, setInputSize] = useState(50);
  const [clfInputSize, setClfInputSize] = useState(128);
  const [detInputSize, setDetInputSize] = useState(100);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);

  // Endpoints
  const clfEndpoint = {
    ann: `${apiBase}/predict-ann`,
    cnn: `${apiBase}/predict-cnn`,
    rnn: `${apiBase}/predict-rnn`,
  }[modelKey];

  const detMap = {
    ann: `${apiBase}/detect-ann-attack`,
    cnn: `${apiBase}/detect-cnn-attack`,
    rnn: `${apiBase}/detect-rnn-attack`,
    cnn_semi: `${apiBase}/detect-cnn-semi-attack`,
    cnn_teacher: `${apiBase}/detect-cnn-teacher-attack`,
  };

  const detEndpoint = detMap[modelKey];

  const onFile = (e) => {
    const fs = Array.from(e.target.files || []);
    setFiles(fs);
    setPreviews(fs.map((f) => URL.createObjectURL(f)));
    setResults([]);
    setError(null);
  };

  const toggleSelected = (key) => {
    setSelectedDets((prev) => {
      if (prev.includes(key)) return prev.filter((k) => k !== key);
      return [...prev, key];
    });
  };

  const runAction = async () => {
    if (files.length === 0) {
      setError("❌ Select at least one image.");
      return;
    }
    if (isClassifier && prefilter && selectedDets.length === 0) {
      setError("❌ Select at least one detector for the pre-filter.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const out = [];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const form = new FormData();
        form.append("image", file);
        form.append("isNight", isNight);
        form.append("clf_size", clfInputSize);
        form.append("det_size", detInputSize);

        if (isClassifier) {
          let pref = null;
          if (prefilter) {
            const perDetector = [];
            for (const detKey of selectedDets) {
              try {
                const res = await axios.post(detMap[detKey], form, {
                  timeout: 30000,
                });
                const score = res?.data?.score ?? 0;
                const is_adv = !!res?.data?.is_adversarial;
                const byThr = score >= detThreshold;
                perDetector.push({
                  key: detKey,
                  score,
                  is_adversarial: is_adv,
                  byThr,
                });
              } catch (e) {
                perDetector.push({
                  key: detKey,
                  score: 0,
                  is_adversarial: false,
                  byThr: false,
                  error: true,
                });
              }
            }
            const votes = perDetector.map((d) => d.byThr);
            const positive = votes.length
              ? ensembleDecision(votes, aggMode)
              : false;
            pref = {
              threshold: detThreshold,
              mode: aggMode,
              selected: selectedDets.slice(),
              perDetector,
              ensemble: {
                positive,
                votes: votes.filter((v) => v).length,
                total: votes.length,
              },
            };
          }

          const clfRes = await axios.post(clfEndpoint, form, {
            timeout: 30000,
          });
          out.push({
            ...(clfRes.data || {}),
            prefilter: pref,
            previewUrl: previews[i],
          });
        } else {
          const detRes = await axios.post(detEndpoint, form, {
            timeout: 30000,
          });
          out.push({ ...(detRes.data || {}), previewUrl: previews[i] });
        }
      }

      setResults(out);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader
        avatar={ICONS[modelKey]}
        title={<Typography variant="h6">{title}</Typography>}
        subheader={
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        }
        action={
          <Tooltip title="Supported: PNG, JPG, WEBP, BMP">
            <IconButton>
              <InfoOutlinedIcon />
            </IconButton>
          </Tooltip>
        }
      />
      <Divider />

      <CardContent>
        <Stack spacing={3}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "1fr 360px" },
              gap: 2,
            }}
          >
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Inputs
              </Typography>
              <Stack spacing={1.5}>
                <Input
                  type="file"
                  inputProps={{ multiple: true, accept: "image/*" }}
                  onChange={onFile}
                />
                <Box display="flex" gap={1.5} flexWrap="wrap">
                  {previews.map((src, i) => (
                    <Box
                      key={i}
                      component="img"
                      src={src}
                      alt={`preview ${i + 1}`}
                      sx={{
                        width: 96,
                        height: 96,
                        objectFit: "cover",
                        borderRadius: 1,
                        border: "1px solid",
                        borderColor: "divider",
                      }}
                    />
                  ))}
                </Box>

                <Box>
                  <Button
                    variant="contained"
                    onClick={runAction}
                    disabled={loading || files.length === 0}
                    startIcon={loading ? <CircularProgress size={16} /> : null}
                  >
                    {loading
                      ? isClassifier
                        ? "Analysis..."
                        : "Detection..."
                      : isClassifier
                      ? `Analysis (${files.length || 0})`
                      : `Detection (${files.length || 0})`}
                  </Button>
                </Box>
                {error && (
                  <Typography color="error" variant="body2">
                    {error}
                  </Typography>
                )}
              </Stack>
            </Box>

            <Box>
              <FormControl fullWidth>
                <FormLabel>Brightness</FormLabel>
                <RadioGroup
                  row
                  value={isNight}
                  onChange={(e) => setIsNight(e.target.value)}
                >
                  <FormControlLabel
                    value="0"
                    control={<Radio size="small" />}
                    label="☀️ Day"
                  />
                  <FormControlLabel
                    value="1"
                    control={<Radio size="small" />}
                    label="🌙 Night"
                  />
                </RadioGroup>
              </FormControl>
              {isClassifier && (
                <TextField
                  label="Classifier input size"
                  type="number"
                  size="small"
                  value={clfInputSize}
                  onChange={(e) => setClfInputSize(Number(e.target.value))}
                  sx={{ width: 180, mt: 1 }}
                />
              )}
              {prefilter && (
                <TextField
                  label="Detector input size"
                  type="number"
                  size="small"
                  value={detInputSize}
                  onChange={(e) => setDetInputSize(Number(e.target.value))}
                  sx={{ width: 180, mt: 1 }}
                />
              )}
              :
              {!isClassifier && (
                <TextField
                  label="Detector input size"
                  type="number"
                  size="small"
                  value={detInputSize}
                  onChange={(e) => setDetInputSize(Number(e.target.value))}
                  sx={{ width: 180, mt: 1 }}
                />
              )}
              {isClassifier ? (
                <Box
                  sx={{
                    border: "1px dashed",
                    borderColor: "divider",
                    borderRadius: 2,
                    p: 2,
                    mt: 2,
                  }}
                >
                  <Stack
                    direction="row"
                    alignItems="center"
                    spacing={1}
                    justifyContent="space-between"
                  >
                    <Typography variant="subtitle2">
                      Adversarial Pre-filter
                    </Typography>
                    <Switch
                      checked={prefilter}
                      onChange={(e) => setPrefilter(e.target.checked)}
                    />
                  </Stack>

                  {prefilter && (
                    <>
                      <Typography variant="body2" sx={{ mt: 1, mb: 0.5 }}>
                        Select detectors (1–3):
                      </Typography>
                      <FormGroup row>
                        {DETECTORS.map((d) => (
                          <FormControlLabel
                            key={d.key}
                            control={
                              <Checkbox
                                size="small"
                                checked={selectedDets.includes(d.key)}
                                onChange={() => toggleSelected(d.key)}
                              />
                            }
                            label={d.label}
                          />
                        ))}
                      </FormGroup>

                      <Stack
                        direction={{ xs: "column", sm: "row" }}
                        spacing={2}
                        sx={{ mt: 1 }}
                      >
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Threshold (applies to all):{" "}
                            {(detThreshold * 100).toFixed(0)}%
                          </Typography>
                          <Slider
                            size="small"
                            min={0}
                            max={1}
                            step={0.01}
                            value={detThreshold}
                            onChange={(_, v) =>
                              setDetThreshold(Array.isArray(v) ? v[0] : v)
                            }
                          />
                        </Box>

                        <FormControl size="small" sx={{ minWidth: 140 }}>
                          <FormLabel>Aggregation</FormLabel>
                          <Select
                            value={aggMode}
                            onChange={(e) => setAggMode(e.target.value)}
                          >
                            <MenuItem value="or">OR (at least 1)</MenuItem>
                            <MenuItem value="and">AND (all)</MenuItem>
                            <MenuItem value="maj">Majority</MenuItem>
                          </Select>
                        </FormControl>
                      </Stack>

                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ mt: 1 }}
                      >
                        Rule: each detector outputs{" "}
                        <strong>Adversarial if score ≥ Threshold</strong>.
                        Ensemble decision with{" "}
                        <strong>{aggMode.toUpperCase()}</strong>.
                      </Typography>
                    </>
                  )}
                </Box>
              ) : (
                <Box
                  sx={{
                    border: "1px dashed",
                    borderColor: "divider",
                    borderRadius: 2,
                    p: 2,
                    mt: 2,
                  }}
                >
                  <Typography variant="subtitle2">
                    Decision Parameters
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Threshold: {(detThreshold * 100).toFixed(0)}%
                    </Typography>
                    <Slider
                      size="small"
                      min={0}
                      max={1}
                      step={0.01}
                      value={detThreshold}
                      onChange={(_, v) =>
                        setDetThreshold(Array.isArray(v) ? v[0] : v)
                      }
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Rule: <strong>score ≥ Threshold</strong> ⇒ Adversarial
                    (otherwise Clean).
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>

          {results.length > 0 && (
            <Stack spacing={2}>
              <Typography variant="subtitle1">Results</Typography>

              {results.map((r, i) => {
                const preview = r.previewUrl;

                // Detector mode (single)
                if (!isClassifier) {
                  const score = r?.score ?? 0;
                  const byThr = score >= detThreshold;
                  return (
                    <Box
                      key={i}
                      sx={{
                        border: "1px solid",
                        borderColor: "divider",
                        borderRadius: 2,
                        p: 2,
                      }}
                    >
                      <Stack
                        direction={{ xs: "column", md: "row" }}
                        spacing={2}
                        alignItems="flex-start"
                      >
                        <Box sx={{ width: { xs: "100%", md: 220 } }}>
                          <Typography variant="caption" color="text.secondary">
                            Preview
                          </Typography>
                          {preview && (
                            <Box
                              component="img"
                              src={preview}
                              alt={`preview ${i + 1}`}
                              sx={{
                                width: "100%",
                                height: "auto",
                                borderRadius: 1,
                                border: "1px solid",
                                borderColor: "divider",
                                mt: 0.5,
                              }}
                            />
                          )}
                        </Box>
                        <Box sx={{ flex: 1 }}>
                          <Stack
                            direction="row"
                            alignItems="center"
                            spacing={1}
                            flexWrap="wrap"
                          >
                            <Typography variant="h6" sx={{ mr: 1 }}>
                              {byThr
                                ? "Adversarial (by threshold)"
                                : "Clean (by threshold)"}
                            </Typography>
                            <DetectorDecisionChip
                              value={score}
                              thr={detThreshold}
                            />
                            {typeof r?.is_adversarial === "boolean" && (
                              <Chip
                                size="small"
                                label={
                                  r.is_adversarial
                                    ? "Model says: Adversarial"
                                    : "Model says: Clean"
                                }
                                color={r.is_adversarial ? "error" : "success"}
                                variant="outlined"
                              />
                            )}
                          </Stack>
                          <Typography variant="body2" color="text.secondary">
                            Score: {(score * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Stack>
                    </Box>
                  );
                }

                // Classifier mode
                const conf = r?.confidence ?? 0;
                const cls = r?.predicted_class ?? "—";
                const pref = r?.prefilter;

                return (
                  <Box
                    key={i}
                    sx={{
                      border: "1px solid",
                      borderColor: "divider",
                      borderRadius: 2,
                      p: 2,
                    }}
                  >
                    <Stack
                      direction={{ xs: "column", md: "row" }}
                      spacing={2}
                      alignItems="flex-start"
                    >
                      <Box sx={{ width: { xs: "100%", md: 220 } }}>
                        <Typography variant="caption" color="text.secondary">
                          Preview
                        </Typography>
                        {preview && (
                          <Box
                            component="img"
                            src={preview}
                            alt={`preview ${i + 1}`}
                            sx={{
                              width: "100%",
                              height: "auto",
                              borderRadius: 1,
                              border: "1px solid",
                              borderColor: "divider",
                              mt: 0.5,
                            }}
                          />
                        )}
                      </Box>

                      <Box sx={{ flex: 1 }}>
                        {pref && (
                          <Box sx={{ mb: 1.5 }}>
                            <Stack
                              direction="row"
                              alignItems="center"
                              spacing={1}
                              flexWrap="wrap"
                            >
                              <Typography
                                variant="overline"
                                color="text.secondary"
                              >
                                Adversarial pre-filter
                              </Typography>
                              <Chip
                                size="small"
                                label={`${
                                  pref.ensemble.positive
                                    ? "Adversarial"
                                    : "Clean"
                                } (votes ${pref.ensemble.votes}/${
                                  pref.ensemble.total
                                }, ${pref.mode.toUpperCase()})`}
                                color={
                                  pref.ensemble.positive ? "error" : "success"
                                }
                                variant="filled"
                              />
                            </Stack>
                            <Stack
                              direction="row"
                              spacing={1}
                              flexWrap="wrap"
                              sx={{ mt: 0.5 }}
                            >
                              {pref.perDetector.map((d) => (
                                <Stack
                                  key={d.key}
                                  direction="row"
                                  spacing={0.5}
                                  alignItems="center"
                                >
                                  <Chip
                                    size="small"
                                    variant="outlined"
                                    label={d.key.toUpperCase()}
                                  />
                                  <DetectorDecisionChip
                                    value={d.score}
                                    thr={pref.threshold}
                                  />
                                </Stack>
                              ))}
                            </Stack>
                          </Box>
                        )}

                        {/* Classification */}
                        <Stack
                          direction="row"
                          alignItems="center"
                          spacing={1}
                          flexWrap="wrap"
                        >
                          <Typography variant="h6" sx={{ mr: 1 }}>
                            {cls}
                          </Typography>
                          <Chip
                            size="small"
                            label={`Confidence: ${(conf * 100).toFixed(1)}%`}
                            variant="outlined"
                          />
                        </Stack>
                      </Box>
                    </Stack>
                  </Box>
                );
              })}
            </Stack>
          )}
        </Stack>
      </CardContent>

      <CardActions sx={{ px: 2, pb: 2, pt: 0 }}>
        <Typography variant="caption" color="text.secondary">
          Tip: With OR you catch more attacks (but also more false alarms). With
          AND you are stricter. Majority provides balance.
        </Typography>
      </CardActions>
    </Card>
  );
}
