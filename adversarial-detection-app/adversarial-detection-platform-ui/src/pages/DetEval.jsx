// src/pages/DetEval.jsx
import React, { useState } from "react";
import {
  Box, Card, CardContent, Typography, Stack, Button, Select, MenuItem,
  FormControl, FormLabel, TextField, CircularProgress,
  Table, TableHead, TableBody, TableRow, TableCell, Paper, Chip
} from "@mui/material";
import axios from "axios";

const DETS = [
  { key: "ann", label: "ANN" },
  { key: "cnn", label: "CNN" },
  { key: "rnn", label: "RNN" },
  { key: "cnn_semi",    label: "CNN (Semi)" },
  { key: "cnn_teacher", label: "CNN (Teacher)" },
];

export default function DetEval({ apiBase }) {
  const [selected, setSelected] = useState(["ann"]);
  const [aggregation, setAggregation] = useState("or");
  const [threshold, setThreshold] = useState(0.5);

  const [zipFile, setZipFile] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const toggle = (k) =>
    setSelected((prev) => (prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k]));

  const runEval = async () => {
    if (!zipFile || !csvFile) { setError("Select an images ZIP and a labels CSV."); return; }
    if (selected.length === 0) { setError("Select at least one detector."); return; }
    setError(null); setLoading(true); setResult(null);
    try {
      const form = new FormData();
      form.append("images_zip", zipFile);
      form.append("labels_csv", csvFile);
      form.append("detectors", selected.join(","));
      form.append("split", "test");

      // from CSV:
      form.append("use_csv_isNight", "1");

      form.append("threshold", String(threshold));
      form.append("aggregation", aggregation);
      const resp = await axios.post(`${apiBase}/batch-eval-detect`, form, { timeout: 0 });
      setResult(resp.data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  };

  const ens = result?.summary?.ensemble;
  const labels = result?.summary?.labels || ["clean", "adversarial"];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Detector Evaluation (ANN/CNN/RNN)</Typography>
        <Stack spacing={2}>
          <Stack direction="row" spacing={2} flexWrap="wrap" alignItems="center">
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <FormLabel>Detectors</FormLabel>
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mt: 0.5 }}>
                {DETS.map((d) => (
                  <Chip
                    key={d.key}
                    label={d.label}
                    variant={selected.includes(d.key) ? "filled" : "outlined"}
                    color={selected.includes(d.key) ? "primary" : "default"}
                    onClick={() => toggle(d.key)}
                  />
                ))}
              </Box>
            </FormControl>

            <TextField
              select
              label="Aggregation"
              size="small"
              value={aggregation}
              onChange={(e) => setAggregation(e.target.value)}
              sx={{ width: 160 }}          
            >
              <MenuItem value="or">OR (≥1)</MenuItem>
              <MenuItem value="and">AND (all)</MenuItem>
              <MenuItem value="maj">Majority</MenuItem>
            </TextField>

            <TextField
              label="Threshold"
              type="number"
              size="small"
              value={threshold}
              onChange={(e) => setThreshold(Math.max(0, Math.min(1, parseFloat(e.target.value || 0))))}
              inputProps={{ step: "0.01", min: "0", max: "1" }}
              sx={{ width: 130 }}
            />

          </Stack>

          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
            <Box>
              <FormLabel>Images ZIP</FormLabel>
              <input type="file" accept=".zip" onChange={(e) => setZipFile(e.target.files?.[0] || null)} />
            </Box>
            <Box>
              <FormLabel>CSV (image_id,label,isNight,split)</FormLabel>
              <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files?.[0] || null)} />
            </Box>
            <Button variant="contained" onClick={runEval} disabled={loading || !zipFile || !csvFile}>
              {loading ? <CircularProgress size={18} /> : "Run"}
            </Button>
          </Stack>

          {error && <Typography color="error">{error}</Typography>}

          {result && (
            <>
              <Typography variant="subtitle1">Summary</Typography>
              <Typography variant="body2">
                N={result.summary.n} • Aggregation: {result.summary.aggregation.toUpperCase()} • Threshold={(result.summary.threshold * 100).toFixed(0)}% • Split: {result.summary.split}
              </Typography>

              {/* ENSEMBLE */}
              <Typography variant="subtitle2" sx={{ mt: 1 }}>Ensemble</Typography>
              <Typography variant="body2">Accuracy={(ens.accuracy * 100).toFixed(2)}%</Typography>
              <Paper sx={{ overflow: "auto", display: "inline-block", mt: 0.5 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>True \ Pred</TableCell>
                      {labels.map((l) => <TableCell key={l} align="right">{l}</TableCell>)}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {ens.confusion.map((row, i) => (
                      <TableRow key={i}>
                        <TableCell>{labels[i]}</TableCell>
                        {row.map((v, j) => <TableCell key={j} align="right">{v}</TableCell>)}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>

              <Typography variant="subtitle2" sx={{ mt: 1 }}>Per-class</Typography>
              <Table size="small" component={Paper}>
                <TableHead>
                  <TableRow>
                    <TableCell>Class</TableCell>
                    <TableCell align="right">Precision</TableCell>
                    <TableCell align="right">Recall</TableCell>
                    <TableCell align="right">F1</TableCell>
                    <TableCell align="right">Support</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {labels.map((l, i) => (
                    <TableRow key={l}>
                      <TableCell>{l}</TableCell>
                      <TableCell align="right">{(ens.per_class.precision[i] * 100).toFixed(1)}%</TableCell>
                      <TableCell align="right">{(ens.per_class.recall[i] * 100).toFixed(1)}%</TableCell>
                      <TableCell align="right">{(ens.per_class.f1[i] * 100).toFixed(1)}%</TableCell>
                      <TableCell align="right">{ens.per_class.support[i]}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {/* Samples */}
              <Typography variant="subtitle2" sx={{ mt: 2 }}>Samples</Typography>
              <Paper sx={{ maxHeight: 460, overflow: "auto" }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Filename</TableCell>
                      <TableCell>True</TableCell>
                      {result.summary.selected.map((k) => <TableCell key={k} align="right">{k.toUpperCase()} score</TableCell>)}
                      {result.summary.selected.map((k) => <TableCell key={k + "_v"} align="right">{k.toUpperCase()} vote</TableCell>)}
                      <TableCell align="right">Ensemble</TableCell>
                      <TableCell>isNight</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.items.map((it, i) => (
                      <TableRow key={i}>
                        <TableCell>{it.filename}</TableCell>
                        <TableCell>{it.true}</TableCell>
                        {result.summary.selected.map((k) => (
                          <TableCell key={k} align="right">
                            {it.scores?.[k] != null ? (it.scores[k] * 100).toFixed(1) + '%' : '—'}
                          </TableCell>
                        ))}
                        {result.summary.selected.map((k) => (
                          <TableCell key={k + "_v"} align="right">
                            {it.votes?.[k] ? '✅' : '❌'}
                          </TableCell>
                        ))}
                        <TableCell align="right">{it.ensemble_pred ? '✅ Adv' : '✔︎ Clean'}</TableCell>
                        <TableCell>{it.isNight}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>
            </>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}
