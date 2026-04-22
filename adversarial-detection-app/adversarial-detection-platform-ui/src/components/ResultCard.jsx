import React from 'react';
import { Box, Typography, Grid } from '@mui/material';
import ImageWithBoxes from './ImageWithBoxes';

export default function ResultCard({ result, index }) {
  const isDetection = 'yolo_detections' in result;
  const hasAdversarial = result.adversarial && result.adversarial_image;
  const formatConfidence = (c) => `${(c * 100).toFixed(2)}%`;

  return (
    <Box mt={3} p={2} border="1px solid #ddd" borderRadius={2} key={index}>
      <Typography>
        <strong>📷 Image {index + 1}</strong>
      </Typography>

      {!isDetection && (
        <>
          {result.predicted_class && (
            <>
              <Typography>
                ➤ <strong>Category:</strong> {result.predicted_class} – <strong>Confidence:</strong> {formatConfidence(result.confidence || 0)}
              </Typography>
              <Typography>
                🔍 Input: {result.adversarial ? 'Adversarial (FGSM)' : 'Normal'}
              </Typography>
            </>
          )}
        </>
      )}

      {isDetection && (
        <Box mt={2}>
          <Typography variant="body2">📦 YOLO Detections:</Typography>
          {Array.isArray(result.yolo_detections) && result.yolo_detections.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              ❌ Nothing detected.
            </Typography>
          ) : (
            result.yolo_detections.map((det, j) => (
              <Typography key={j}>
                ➤ {det.class_name} ({((det.confidence || 0) * 100).toFixed(1)}%) – box: [{(det.box || []).map((v) => v.toFixed?.(1) ?? v).join(', ')}]
              </Typography>
            ))
          )}
        </Box>
      )}

      <Grid container spacing={2} mt={1}>
        {!isDetection && result.previewUrl && (
          <Grid item xs={6}>
            <Typography variant="body2">🖼️ Original:</Typography>
            <img
              src={result.previewUrl}
              alt={`original image preview ${index + 1}`}
              style={{ width: '100%', borderRadius: 8, border: '1px solid #ccc' }}
            />
          </Grid>
        )}
        {hasAdversarial && (
          <Grid item xs={6}>
            <Typography variant="body2">🧪 Adversarial:</Typography>
            <img
              src={`data:image/png;base64,${result.adversarial_image}`}
              alt={`adversarial preview ${index + 1}`}
              style={{ width: '100%', borderRadius: 8, border: '1px solid #ccc' }}
            />
          </Grid>
        )}
        {isDetection && (
          <Grid item xs={6}>
            <Typography variant="body2">🖼️ YOLO Overlay:</Typography>
            <ImageWithBoxes src={result.previewUrl} detections={result.yolo_detections} />
          </Grid>
        )}
      </Grid>
    </Box>
  );
}
