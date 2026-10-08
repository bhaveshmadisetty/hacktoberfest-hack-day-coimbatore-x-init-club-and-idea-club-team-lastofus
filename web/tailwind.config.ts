import type { Config } from "tailwindcss";

// Vercel-style monochrome scale. Colour is reserved for verdicts only, so the
// one thing on screen that is coloured is the thing that carries judgment.
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#ffffff",
        surface: "#fafafa",
        border: "#ebebeb",
        borderStrong: "#d4d4d4",
        fg: "#000000",
        muted: "#666666",
        subtle: "#8f8f8f",
        retrain: "#dc2626",
        rollback: "#ea580c",
        investigate: "#ca8a04",
        benign: "#64748b",
      },
      fontFamily: {
        sans: ["var(--font-geist)", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,0.04)",
        lift: "0 4px 12px rgba(0,0,0,0.06)",
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-out",
        shimmer: "shimmer 1.6s linear infinite",
      },
      keyframes: {
        fadeIn: { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
        shimmer: { "0%": { backgroundPosition: "-500px 0" }, "100%": { backgroundPosition: "500px 0" } },
      },
    },
  },
  plugins: [],
};
export default config;
