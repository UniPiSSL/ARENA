import React, { useRef, useState, useEffect } from 'react';
import { Box } from '@mui/material';

export default function ImageWithBoxes({ src, detections }) {
  const imgRef = useRef(null);
  const containerRef = useRef(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });

  const handleLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setNatural({ w: img.naturalWidth, h: img.naturalHeight });
    setDisplaySize({ w: img.width, h: img.height });
  };

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(() => {
      const img = imgRef.current;
      if (img) setDisplaySize({ w: img.width, h: img.height });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  if (!detections || !Array.isArray(detections)) return null;

  return (
    <Box ref={containerRef} sx={{ position: 'relative', width: '100%' }}>
      <img
        ref={imgRef}
        src={src}
        alt="preview"
        style={{ width: '100%', display: 'block' }}
        onLoad={handleLoad}
      />
      {detections.map((det, i) => {
        const [x1, y1, x2, y2] = det.box || [];
        if (natural.w === 0 || displaySize.w === 0 || det.box?.length !== 4) return null;

        const scaleX = displaySize.w / natural.w;
        const scaleY = displaySize.h / natural.h;
        const dispX1 = x1 * scaleX;
        const dispY1 = y1 * scaleY;
        const dispX2 = x2 * scaleX;
        const dispY2 = y2 * scaleY;
        const dispW = dispX2 - dispX1;
        const dispH = dispY2 - dispY1;

        return (
          <Box
            key={i}
            sx={{
              position: 'absolute',
              left: dispX1,
              top: dispY1,
              width: dispW,
              height: dispH,
              border: '2px solid red',
              boxSizing: 'border-box',
              pointerEvents: 'none',
            }}
            aria-label={`Detection box for ${det.class_name}`}
            title={`${det.class_name} ${(det.confidence*100).toFixed(1)}%`}
          />
        );
      })}
    </Box>
  );
}
