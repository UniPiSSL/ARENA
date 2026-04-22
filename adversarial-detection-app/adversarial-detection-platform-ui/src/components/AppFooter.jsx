import { Box, Container, Divider, IconButton, Stack, Typography } from "@mui/material";
import GitHubIcon from "@mui/icons-material/GitHub";
import MailOutlineIcon from "@mui/icons-material/MailOutline";

export const FOOTER_HEIGHT = 64;  

export default function AppFooter() {
  return (
    <Box
      component="footer"
      sx={{
        position: "fixed",
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: (t) => t.zIndex.appBar,
        borderTop: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
        height: FOOTER_HEIGHT,
      }}
    >
      <Container maxWidth="lg" sx={{ height: "100%" }}>
        <Divider />
        <Stack
          direction={{ xs: "column", sm: "row" }}
          alignItems="center"
          justifyContent="space-between"
          spacing={1.5}
          sx={{ height: "100%", py: 1 }}
        >
          <Typography variant="body2" color="text.secondary">
            © {new Date().getFullYear()} Adversarial AI Detection • Developed by Evripidis Katsianos
          </Typography>
          <Stack direction="row" spacing={1}>
            <IconButton size="small" color="inherit" href="https://github.com/eyripidiska/Adversarial-AI-Attack-Detection" target="_blank" aria-label="GitHub">
              <GitHubIcon fontSize="small" />
            </IconButton>
            <IconButton size="small" color="inherit" href="mailto:euripidiskatsianos@gmail.com" aria-label="Email">
              <MailOutlineIcon fontSize="small" />
            </IconButton>
          </Stack>
        </Stack>
      </Container>
    </Box>
  );
}
