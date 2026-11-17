"use client";

import { useEffect } from "react";

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log non-sensitive error telemetry
    console.error("[RoomAI Application Error]", error.digest ?? error.message);
  }, [error]);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#0A0908",
        color: "#F2F0EA",
        padding: "24px",
        fontFamily: 'var(--sans, "DM Sans", Arial, sans-serif)',
        textAlign: "center",
      }}
    >
      <div
        style={{
          maxWidth: "480px",
          border: "1px solid rgba(242, 240, 234, 0.14)",
          backgroundColor: "#151412",
          padding: "40px 32px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "20px",
        }}
      >
        <span
          style={{
            fontFamily: 'var(--mono, "DM Mono", monospace)',
            fontSize: "10px",
            letterSpacing: "0.18em",
            color: "#8B4937",
            textTransform: "uppercase",
          }}
        >
          System Disruption // 0x500
        </span>

        <h1
          style={{
            fontFamily: 'var(--serif, "Instrument Serif", Georgia, serif)',
            fontSize: "32px",
            fontWeight: 400,
            margin: "0",
            letterSpacing: "-0.02em",
          }}
        >
          Spatial Viewport Interrupted
        </h1>

        <p
          style={{
            fontSize: "13px",
            lineHeight: 1.6,
            color: "#B9B4AA",
            margin: "0",
          }}
        >
          The graphics or narrative runtime encountered an unexpected state.
          Re-initializing the viewport may restore session continuity.
        </p>

        <button
          onClick={() => reset()}
          style={{
            marginTop: "12px",
            padding: "10px 24px",
            backgroundColor: "transparent",
            color: "#F2F0EA",
            border: "1px solid rgba(242, 240, 234, 0.28)",
            fontFamily: 'var(--mono, "DM Mono", monospace)',
            fontSize: "10px",
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = "#8B4937";
            e.currentTarget.style.borderColor = "#8B4937";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "transparent";
            e.currentTarget.style.borderColor = "rgba(242, 240, 234, 0.28)";
          }}
        >
          Re-initialize Session
        </button>
      </div>
    </div>
  );
}
