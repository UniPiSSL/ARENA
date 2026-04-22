import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  Button,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
  Stack,
  Input,
  Select,
  MenuItem,
  Box,
  Switch,
  CircularProgress,
  Alert,
  Typography,
  Slider,
  TextField,
} from "@mui/material";
import { promisePool } from "../utils/promisePool";

const API_FALLBACK = import.meta.env.VITE_API_BASE || "http://localhost:5000";

/**
 * Props:
 * - onResults(results[])
 * - apiBase (default: env/localhost)
 * - allowedModels: string[] (default: ['ann','cnn','rnn','yolo'])
 * - defaultModel: string (default: 'ann')
 * - hideModelSelect: boolean (default: false)
 * - yoloDefaults: object -> initial values for YOLO params
 * - yoloMode: "full" | "minimal" | "hybrid"
 */

export default function PredictionForm({
  onResults,
  apiBase = API_FALLBACK,
  allowedModels = ["ann", "cnn", "rnn", "yolo"],
  defaultModel = "ann",
  hideModelSelect = false,
  yoloDefaults = { conf: 0.01, iou: 0.45, imgsz: 640 },
  yoloMode = "full",
}) {
  const [images, setImages] = useState([]);
  const [isNight, setIsNight] = useState("0");
  const [model, setModel] = useState(defaultModel);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [useAdversarial, setUseAdversarial] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [globalError, setGlobalError] = useState(null);

  const [yoloParams, setYoloParams] = useState(yoloDefaults);

  useEffect(() => {
    if (hideModelSelect) setModel(defaultModel);
  }, [hideModelSelect, defaultModel]);

  useEffect(() => {
    if (!allowedModels.includes(model)) setModel(defaultModel);
    setYoloParams(yoloDefaults);
  }, [allowedModels, defaultModel, JSON.stringify(yoloDefaults)]);

  useEffect(() => {
    return () => previewUrls.forEach((url) => URL.revokeObjectURL(url));
  }, [previewUrls]);

  const endpointMap = {
    ann: {
      clean: `${apiBase}/predict-ann`,
      adv: `${apiBase}/predict-ann-adversarial`,
    },
    cnn: {
      clean: `${apiBase}/predict-cnn`,
      adv: `${apiBase}/predict-cnn-adversarial`,
    },
    rnn: {
      clean: `${apiBase}/predict-rnn`,
      adv: `${apiBase}/predict-rnn-adversarial`,
    },
    yolo: {
      clean: `${apiBase}/yolo-predict`,
    },
  };

  const handleImageChange = (e) => {
    const files = Array.from(e.target.files || []);
    previewUrls.forEach((url) => URL.revokeObjectURL(url));
    setImages(files);
    setPreviewUrls(files.map((file) => URL.createObjectURL(file)));
  };

  const handleSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      setGlobalError(null);

      if (images.length === 0) {
        setGlobalError("❌ Select at least one image");
        return;
      }

      setIsLoading(true);
      try {
        const endpoint =
          model === "yolo"
            ? endpointMap.yolo.clean
            : useAdversarial
            ? endpointMap[model].adv
            : endpointMap[model].clean;

        const tasks = images.map((img, index) => async () => {
          const formData = new FormData();
          formData.append("image", img);
          if (model !== "yolo") formData.append("isNight", isNight);
          const previewUrl = previewUrls[index];

          try {
            let finalParams = { ...yoloParams };

            if (yoloMode === "hybrid") {
              finalParams = { ...finalParams, return_crops: 1, square: 1 };
            }

            const axiosCfg =
              model === "yolo"
                ? { params: finalParams, timeout: 30000 }
                : { timeout: 30000 };

            const response = await axios.post(endpoint, formData, axiosCfg);
            if (model === "yolo") {
              return {
                previewUrl,
                yolo_detections: response.data?.detections || [],
              };
            } else {
              return { ...response.data, previewUrl };
            }
          } catch (error) {
            console.error(
              "⚠️ Prediction error for image:",
              img.name,
              error?.message || error
            );
            return model === "yolo"
              ? { previewUrl, yolo_detections: [], error: true }
              : { predicted_class: "error", confidence: 0, previewUrl };
          }
        });

        const results = await promisePool(tasks, 4);
        onResults(results);
      } catch (err) {
        console.error("Unexpected error during prediction:", err);
        setGlobalError("Something went wrong while sending the predictions.");
      } finally {
        setIsLoading(false);
      }
    },
    [
      images,
      isNight,
      model,
      useAdversarial,
      previewUrls,
      onResults,
      apiBase,
      yoloParams,
      yoloMode,
    ]
  );

  const renderYoloControls = () => {
    if (model !== "yolo") return null;

    return (
      <Box sx={{ border: "1px dashed #ccc", borderRadius: 2, p: 2 }}>
        <FormLabel>YOLO Parameters</FormLabel>

        {/* conf */}
        <Box sx={{ mt: 2 }}>
          <FormLabel>conf: {Number(yoloParams.conf).toFixed(2)}</FormLabel>
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={Number(yoloParams.conf)}
            onChange={(_, v) =>
              setYoloParams((p) => ({
                ...p,
                conf: Array.isArray(v) ? v[0] : v,
              }))
            }
          />
        </Box>

        {/* iou */}
        <Box sx={{ mt: 1 }}>
          <FormLabel>iou: {Number(yoloParams.iou).toFixed(2)}</FormLabel>
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={Number(yoloParams.iou)}
            onChange={(_, v) =>
              setYoloParams((p) => ({ ...p, iou: Array.isArray(v) ? v[0] : v }))
            }
          />
        </Box>

        <Box
          sx={{
            display: "flex",
            gap: 2,
            mt: 1,
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <TextField
            label="imgsz"
            type="number"
            size="small"
            sx={{ width: 120 }}
            value={yoloParams.imgsz}
            onChange={(e) =>
              setYoloParams((p) => ({
                ...p,
                imgsz: Math.max(64, Number(e.target.value || 640)),
              }))
            }
          />

          {yoloMode !== "minimal" && (
            <>
              <TextField
                label="expand"
                type="text" 
                size="small"
                sx={{ width: 120 }}
                inputProps={{
                  inputMode: "decimal",
                  pattern: "[0-9]*[.,]?[0-9]*",
                }}
                value={String(yoloParams.expand ?? 0.1)}
                onChange={(e) => {
                  const v = (e.target.value || "").replace(",", ".");
                  const num = Math.max(0, Number(v));
                  setYoloParams((p) => ({
                    ...p,
                    expand: isNaN(num) ? p.expand : num,
                  }));
                }}
              />

              <TextField
                label="crop_size"
                type="number"
                size="small"
                sx={{ width: 120 }}
                value={yoloParams.crop_size ?? 50}
                onChange={(e) =>
                  setYoloParams((p) => ({
                    ...p,
                    crop_size: Math.max(8, Number(e.target.value || 50)),
                  }))
                }
              />
            </>
          )}

          {yoloMode === "full" && (
            <>
              <Box
                sx={{
                  flex: 1,
                  minWidth: 100,
                  display: "flex",
                  flexDirection: "column",
                  gap: 0.5,
                }}
              >
                <FormLabel>Square</FormLabel>
                <Switch
                  checked={!!(yoloParams.square ?? 1)}
                  onChange={(e) =>
                    setYoloParams((p) => ({
                      ...p,
                      square: e.target.checked ? 1 : 0,
                    }))
                  }
                  size="small"
                />
              </Box>

              <Box
                sx={{
                  flex: 1,
                  minWidth: 100,
                  display: "flex",
                  flexDirection: "column",
                  gap: 0.5,
                }}
              >
                <FormLabel>Extras</FormLabel>
                <Switch
                  checked={!!(yoloParams.extras ?? 0)}
                  onChange={(e) =>
                    setYoloParams((p) => ({
                      ...p,
                      extras: e.target.checked ? 1 : 0,
                    }))
                  }
                  size="small"
                />
              </Box>
            </>
          )}
        </Box>
      </Box>
    );
  };

  return (
    <form onSubmit={handleSubmit} aria-label="Prediction form">
      <Stack spacing={3}>
        {globalError && <Alert severity="error">{globalError}</Alert>}

        {!hideModelSelect ? (
          <FormControl>
            <FormLabel>Model</FormLabel>
            <Select value={model} onChange={(e) => setModel(e.target.value)}>
              {allowedModels.map((m) => (
                <MenuItem key={m} value={m}>
                  {m.toUpperCase()}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        ) : (
          <Box>
            <FormLabel>Model</FormLabel>
            <Typography sx={{ fontWeight: 600 }}>
              {defaultModel.toUpperCase()}
            </Typography>
          </Box>
        )}

        <Input
          type="file"
          inputProps={{ multiple: true, accept: "image/*" }}
          onChange={handleImageChange}
        />
        <Box display="flex" gap={2} flexWrap="wrap">
          {previewUrls.map((url, i) => (
            <img
              key={i}
              src={url}
              alt={`preview ${i + 1}`}
              width="100"
              style={{ borderRadius: 8 }}
            />
          ))}
        </Box>

        {model !== "yolo" && (
          <>
            <FormControl component="fieldset">
              <FormLabel component="legend">Brightness</FormLabel>
              <RadioGroup
                row
                value={isNight}
                onChange={(e) => setIsNight(e.target.value)}
              >
                <FormControlLabel
                  value="0"
                  control={<Radio />}
                  label="☀️ Day"
                />
                <FormControlLabel
                  value="1"
                  control={<Radio />}
                  label="🌙 Night"
                />
              </RadioGroup>
            </FormControl>
          </>
        )}

        {renderYoloControls()}

        <Button
          type="submit"
          variant="contained"
          disabled={isLoading}
          startIcon={isLoading ? <CircularProgress size={16} /> : null}
        >
          {isLoading
            ? "Analysis in progress..."
            : `Analyzing ${
                images.length > 1 ? `(${images.length} images)` : ""
              }`}
        </Button>
      </Stack>
    </form>
  );
}
