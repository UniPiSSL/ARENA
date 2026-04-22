import React from "react";
import { Card, CardContent, Typography } from "@mui/material";
import YoloProCard from "../components/YoloProCard";

export default function YoloPage({ apiBase }) {
  return (
    <Card>
        <YoloProCard apiBase={apiBase} />
    </Card>
  );
}
