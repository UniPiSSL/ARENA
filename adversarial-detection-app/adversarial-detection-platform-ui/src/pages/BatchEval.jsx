// src/pages/BatchEval.jsx
import React, { useState } from 'react';
import {
  Box, Card, CardContent, Typography, Stack, Button, Select, MenuItem,
  FormControl, FormLabel, CircularProgress,
  Table, TableHead, TableBody, TableRow, TableCell, Paper
} from '@mui/material';
import axios from 'axios';

export default function BatchEval({ apiBase }) {
  const [model, setModel] = useState('ann');

  const [zipFile, setZipFile] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runEval = async () => {
    if (!zipFile || !csvFile) {
      setError('Select an images ZIP and a labels CSV.');
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const form = new FormData();
      form.append('images_zip', zipFile);
      form.append('labels_csv', csvFile);
      form.append('model', model);
      form.append('split', 'test');

      // isNight ONLY from CSV
      form.append('use_csv_isNight', '1');

      const resp = await axios.post(`${apiBase}/batch-eval-classify`, form, { timeout: 0 }); // 0 = no timeout
      setResult(resp.data);
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  };

  const cm = result?.summary?.confusion || [];
  const labels = result?.summary?.labels || [];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Classifiers Evaluation (ANN/CNN/RNN)</Typography>
        <Stack spacing={2}>
          <Stack direction="row" spacing={2} flexWrap="wrap" alignItems="center">
            <FormControl size="small" sx={{ minWidth: 140 }}>
              <FormLabel>Model</FormLabel>
              <Select value={model} onChange={(e) => setModel(e.target.value)}>
                <MenuItem value="ann">ANN</MenuItem>
                <MenuItem value="cnn">CNN</MenuItem>
                <MenuItem value="rnn">RNN</MenuItem>
              </Select>
            </FormControl>
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
              {loading ? <CircularProgress size={18} /> : 'Run'}
            </Button>
          </Stack>

          {error && <Typography color="error">{error}</Typography>}

          {result && (
            <>
              <Typography variant="subtitle1">Summary</Typography>
              <Typography variant="body2">
                N={result.summary.n} • Accuracy={(result.summary.accuracy*100).toFixed(2)}% • Split: {result.summary.split}
              </Typography>

              <Typography variant="subtitle2" sx={{ mt: 1 }}>Confusion Matrix</Typography>
              <Paper sx={{ overflow: 'auto', display: 'inline-block' }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>True \ Pred</TableCell>
                      {labels.map((l) => <TableCell key={l} align="right">{l}</TableCell>)}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {cm.map((row, i) => (
                      <TableRow key={i}>
                        <TableCell>{labels[i]}</TableCell>
                        {row.map((v, j) => <TableCell key={j} align="right">{v}</TableCell>)}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>

              <Typography variant="subtitle2" sx={{ mt: 2 }}>Per-class</Typography>
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
                      <TableCell align="right">{(result.summary.per_class.precision[i]*100).toFixed(1)}%</TableCell>
                      <TableCell align="right">{(result.summary.per_class.recall[i]*100).toFixed(1)}%</TableCell>
                      <TableCell align="right">{(result.summary.per_class.f1[i]*100).toFixed(1)}%</TableCell>
                      <TableCell align="right">{result.summary.per_class.support[i]}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              <Typography variant="subtitle2" sx={{ mt: 2 }}>Samples</Typography>
              <Paper sx={{ maxHeight: 460, overflow: 'auto' }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Filename</TableCell>
                      <TableCell>True</TableCell>
                      <TableCell>Pred</TableCell>
                      <TableCell align="right">Conf.</TableCell>
                      <TableCell>isNight</TableCell>
                      <TableCell>OK</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.items.map((it, i) => (
                      <TableRow key={i}>
                        <TableCell>{it.filename}</TableCell>
                        <TableCell>{it.true}</TableCell>
                        <TableCell>{it.pred}</TableCell>
                        <TableCell align="right">{it.confidence != null ? (it.confidence*100).toFixed(1)+'%' : '—'}</TableCell>
                        <TableCell>{it.isNight}</TableCell>
                        <TableCell>{it.correct ? '✅' : '❌'}</TableCell>
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
