// src/components/Layout.jsx
import * as React from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  AppBar, Box, CssBaseline, Divider, Drawer, IconButton, List, ListItem, ListItemButton,
  ListItemIcon, ListItemText, Toolbar, Typography, useMediaQuery, Tooltip, Container
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import SecurityIcon from "@mui/icons-material/Security";
import TimelineIcon from "@mui/icons-material/Timeline";
import MemoryIcon from "@mui/icons-material/Memory";
import ScatterPlotIcon from "@mui/icons-material/ScatterPlot";
import LayersIcon from "@mui/icons-material/Layers";
import PhotoFilterIcon from "@mui/icons-material/PhotoFilter";
import AssessmentIcon from "@mui/icons-material/Assessment";
import FactCheckIcon from "@mui/icons-material/FactCheck";
import TuneIcon from "@mui/icons-material/Tune";
import GppBadIcon from "@mui/icons-material/GppBad";
import AppFooter, { FOOTER_HEIGHT } from "./AppFooter";

const drawerWidth = 300;

const SECTIONS = [
  {
    header: "Detectors",
    icon: <SecurityIcon />,
    items: [
      { to: "/det-ann", label: "ANN Attack Detector", icon: <TimelineIcon /> },
      { to: "/det-cnn", label: "CNN Attack Detector", icon: <MemoryIcon /> },
      { to: "/det-rnn", label: "RNN Attack Detector", icon: <ScatterPlotIcon /> },
      { to: "/det-semi-cnn", label: "CNN (Semi) Attack Detector", icon: <MemoryIcon /> },
      { to: "/det-teacher-cnn", label: "CNN (Teacher) Attack Detector", icon: <MemoryIcon /> },
    ],
  },
  {
    header: "Models",
    icon: <LayersIcon />,
    items: [
      { to: "/ann", label: "ANN", icon: <TimelineIcon /> },
      { to: "/cnn", label: "CNN", icon: <MemoryIcon /> },
      { to: "/rnn", label: "RNN", icon: <ScatterPlotIcon /> },
      { to: "/yolo", label: "YOLO", icon: <PhotoFilterIcon /> },
      { to: "/sweep-yolo", label: "YOLO Sweep", icon: <TuneIcon /> },
    ],
  },
  {
    header: "Evaluation",
    icon: <AssessmentIcon />,
    items: [
      { to: "/batch-eval", label: "Classifiers Eval", icon: <FactCheckIcon /> },
      { to: "/det-eval", label: "Detectors Eval", icon: <GppBadIcon /> },
    ],
  },
];

export default function Layout({
  title = "Adversarial AI Attack Detection Suite",
  children,
}) {
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const isDesktop = useMediaQuery("(min-width:900px)");
  const location = useLocation();

  React.useEffect(() => {
    const section = location.pathname.replace("/", "") || "home";
    document.title = `${title} • ${section.toUpperCase()}`;
  }, [location.pathname, title]);

  const drawer = (
    <div>
      <Toolbar />
      <Divider />
      {SECTIONS.map((sec, i) => (
        <Box key={sec.header}>
          <List
            subheader={
              <ListItem disablePadding>
                <ListItemButton disabled>
                  <ListItemIcon>{sec.icon}</ListItemIcon>
                  <ListItemText
                    primary={sec.header}
                    primaryTypographyProps={{ fontWeight: 700 }}
                  />
                </ListItemButton>
              </ListItem>
            }
          >
            {sec.items.map((item) => {
              const isActive = location.pathname.startsWith(item.to);
              return (
                <ListItem key={item.to} disablePadding>
                  <Tooltip
                    title={item.label}
                    placement="right"
                    disableHoverListener={isDesktop}
                  >
                    <ListItemButton
                      component={NavLink}
                      to={item.to}
                      onClick={() => !isDesktop && setMobileOpen(false)}
                      selected={isActive}
                      sx={{
                        mx: 1,
                        my: 0.5,
                        borderRadius: 1.5,
                        "&.Mui-selected": {
                          bgcolor: (t) => t.palette.action.selected,
                          "&:hover": { bgcolor: (t) => t.palette.action.selected },
                        },
                      }}
                    >
                      <ListItemIcon
                        sx={{
                          minWidth: 40,
                          color: isActive ? "primary.main" : "inherit",
                        }}
                      >
                        {item.icon}
                      </ListItemIcon>
                      <ListItemText
                        primary={item.label}
                        primaryTypographyProps={{
                          fontWeight: isActive ? 700 : 500,
                          color: isActive ? "primary.main" : "inherit",
                        }}
                      />
                    </ListItemButton>
                  </Tooltip>
                </ListItem>
              );
            })}
          </List>
          {i < SECTIONS.length - 1 && <Divider />}
        </Box>
      ))}
    </div>
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "background.default" }}>
      <CssBaseline />

      {/* AppBar */}
      <AppBar position="fixed" sx={{ zIndex: (t) => t.zIndex.drawer + 1 }}>
        <Toolbar sx={{ justifyContent: "space-between", gap: 1.25 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.25 }}>
            {!isDesktop && (
              <IconButton
                color="inherit"
                edge="start"
                onClick={() => setMobileOpen(true)}
                sx={{ mr: 0.5 }}
              >
                <MenuIcon />
              </IconButton>
            )}

            <Box
              component="img"
              src="/icon.svg"   
              alt="App logo"
              sx={{ width: 28, height: 28, display: "block" }}
            />
            <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
              {title}
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Drawer (temporary on mobile, permanent on desktop) */}
      <Box component="nav" sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }} aria-label="sidebar">
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: "block", md: "none" },
            "& .MuiDrawer-paper": { boxSizing: "border-box", width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          open
          sx={{
            display: { xs: "none", md: "block" },
            "& .MuiDrawer-paper": { boxSizing: "border-box", width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content + sticky footer spacing */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { xs: "100%", md: `calc(100% - ${drawerWidth}px)` },
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Toolbar />
        <Container maxWidth="lg" sx={{ flex: 1, py: 3, pb: `${FOOTER_HEIGHT + 16}px` }}>
          {children}
        </Container>
        <AppFooter />
      </Box>
    </Box>
  );
}
