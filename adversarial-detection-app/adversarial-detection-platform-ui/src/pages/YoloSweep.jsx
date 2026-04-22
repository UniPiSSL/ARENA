import React, { useState, useCallback } from 'react';
import axios from 'axios';
import {
  Box, Button, TextField, Typography, Stack, Card, CardContent,
  CircularProgress, Table, TableBody, TableCell, TableHead, TableRow,
  Paper, Divider
} from '@mui/material';

export default function YoloSweep({ apiBase = '' }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [confs, setConfs] = useState('0.01,0.05,0.1');
  const [ious, setIous] = useState('0.3,0.45,0.6');
  const [sweepResult, setSweepResult] = useState(null);
  const [loadingSweep, setLoadingSweep] = useState(false);
  const [selectedCombo, setSelectedCombo] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [loadingAnnot, setLoadingAnnot] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setSweepResult(null);
    setSelectedCombo(null);
    setAnnotatedImage(null);
    setError(null);
  };

  const runSweep = useCallback(async () => {
    if (!file) {
      setError('Please select an image first.');
      return;
    }
    setError(null);
    setLoadingSweep(true);
    setSweepResult(null);
    setSelectedCombo(null);
    setAnnotatedImage(null);
    try {
      const form = new FormData();
      form.append('image', file);
      const resp = await axios.post(
        `${apiBase || ''}/yolo-sweep?confs=${encodeURIComponent(confs)}&ious=${encodeURIComponent(ious)}`,
        form,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 30000 }
      );
      setSweepResult(resp.data);
    } catch (e) {
      console.error(e);
      setError('Sweep failed. Check the console for details.');
    } finally {
      setLoadingSweep(false);
    }
  }, [file, confs, ious, apiBase]);

  const fetchAnnotated = useCallback(
    async ({ conf, iou }) => {
      if (!file) return;
      setAnnotatedImage(null);
      setLoadingAnnot(true);
      try {
        const form = new FormData();
        form.append('image', file);
        const resp = await axios.post(`${apiBase || ''}/yolo-predict?conf=${conf}&iou=${iou}`, form, {
          headers: { 'Content-Type': 'multipart/form-data' }, timeout: 30000
        });
        if (resp.data?.yolo_image) {
          setAnnotatedImage(`data:image/png;base64,${resp.data.yolo_image}`);
        }
      } catch (e) {
        console.error(e);
        setError('Annotated inference failed.');
      } finally {
        setLoadingAnnot(false);
      }
    },
    [file, apiBase]
  );

  const handleSelectCombo = (combo) => {
    setSelectedCombo(combo);
    fetchAnnotated({ conf: combo.conf, iou: combo.iou });
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>YOLO Sweep & Preview</Typography>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems="flex-start">
          <Box flex="1" minWidth={250}>
            <Typography variant="subtitle2">1. Image selection</Typography>
            <input type="file" accept="image/*" onChange={handleFileChange} />
            {previewUrl && (
              <Box mt={1}>
                <Typography variant="caption">Preview:</Typography>
                <Box component="img" src={previewUrl} alt="preview"
                  sx={{ width: 150, height: 'auto', borderRadius: 1, border: '1px solid #ccc', mt: 0.5 }} />
              </Box>
            )}
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2">2. Sweep configuration</Typography>
            <TextField label="Confs (comma-separated)" value={confs} size="small" fullWidth
              onChange={(e) => setConfs(e.target.value)} helperText="e.g. 0.01,0.05,0.1" sx={{ mb: 1 }} />
            <TextField label="IoUs (comma-separated)" value={ious} size="small" fullWidth
              onChange={(e) => setIous(e.target.value)} helperText="e.g. 0.3,0.45,0.6" sx={{ mb: 1 }} />
            <Button variant="contained" onClick={runSweep} disabled={loadingSweep || !file}>
              {loadingSweep ? <CircularProgress size={18} /> : 'Τρέξε Sweep'}
            </Button>
            {error && <Typography variant="body2" color="error" mt={1}>{error}</Typography>}
          </Box>

          <Box flex="2" minWidth={300}>
            <Typography variant="subtitle2">3. Sweep results</Typography>
            {loadingSweep && <Typography>Load results...</Typography>}
            {!loadingSweep && sweepResult && (
              <>
                <Typography variant="body2" sx={{ mt: 1 }}>Best combinations:</Typography>
                <Table size="small" component={Paper} sx={{ mb: 2, mt: 1 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>conf</TableCell>
                      <TableCell>iou</TableCell>
                      <TableCell align="right">detections</TableCell>
                      <TableCell>Preview</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {(sweepResult.best || []).map((c, i) => (
                      <TableRow key={i} hover selected={selectedCombo && selectedCombo.conf === c.conf && selectedCombo.iou === c.iou}
                        onClick={() => handleSelectCombo(c)} sx={{ cursor: 'pointer' }}>
                        <TableCell>{c.conf}</TableCell>
                        <TableCell>{c.iou}</TableCell>
                        <TableCell align="right">{c.detections}</TableCell>
                        <TableCell><Button size="small" onClick={() => handleSelectCombo(c)}>Select</Button></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <Divider />
                <Typography variant="body2" sx={{ mt: 1 }}>All combinations (top 10):</Typography>
                <Box sx={{ maxHeight: 240, overflow: 'auto' }}>
                  <Table size="small" component={Paper}>
                    <TableHead>
                      <TableRow>
                        <TableCell>conf</TableCell>
                        <TableCell>iou</TableCell>
                        <TableCell align="right">detections</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(sweepResult.all || []).slice(0, 10).map((c, i) => (
                        <TableRow key={i} hover onClick={() => handleSelectCombo(c)} sx={{ cursor: 'pointer' }}>
                          <TableCell>{c.conf}</TableCell>
                          <TableCell>{c.iou}</TableCell>
                          <TableCell align="right">{c.detections}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Box>
              </>
            )}
          </Box>

          <Box flex="2" minWidth={300}>
            <Typography variant="subtitle2">4. Annotated preview</Typography>
            {loadingAnnot && <CircularProgress size={24} />}
            {!loadingAnnot && annotatedImage && (
              <Box>
                <Typography variant="caption">
                  Select: conf={selectedCombo?.conf} iou={selectedCombo?.iou}
                </Typography>
                <Box component="img" src={annotatedImage} alt="annotated"
                  sx={{ width: '100%', maxHeight: 400, objectFit: 'contain', borderRadius: 1, mt: 1, border: '1px solid #ccc' }} />
              </Box>
            )}
            {!annotatedImage && selectedCombo && !loadingAnnot && (
              <Typography variant="body2" color="text.secondary">No annotated image received.</Typography>
            )}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}
