"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiClient } from "@/lib/api/client";
import { ReadinessResponse } from "@/lib/api/types";

interface LiveStatusBadgeProps {
  className?: string;
  showDetails?: boolean;
}

export function LiveStatusBadge({ className = "", showDetails = false }: LiveStatusBadgeProps) {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [isChecking, setIsChecking] = useState<boolean>(true);
  const [isOnline, setIsOnline] = useState<boolean>(false);

  const checkStatus = async () => {
    try {
      const data = await apiClient.checkReadiness();
      setReadiness(data);
      setIsOnline(data.status === "ready" || data.cuda_available);
    } catch {
      setIsOnline(false);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Link
      href="/model"
      className={`live-status-pill ${isOnline ? "is-online" : "is-standby"} ${className}`}
      title={
        isOnline
          ? `Connected to Inference Engine: ${readiness?.device_name || "CPU / GPU"} ${readiness?.free_vram_gb ? `(${readiness.free_vram_gb.toFixed(1)} GB Free VRAM)` : ""}`
          : "Backend API offline or connecting. Click to launch Model Studio."
      }
    >
      <span className="status-dot-pulse" aria-hidden="true" />
      <span className="status-label">
        {isChecking
          ? "CHECKING ENGINE..."
          : isOnline
          ? `MODEL A3 // ${readiness?.device_name?.includes("4060") ? "RTX 4060 GPU" : readiness?.device_name === "CPU" ? "CPU ONLINE" : "ENGINE ONLINE"}`
          : "ENGINE STANDBY"}
      </span>
      {showDetails && readiness && (
        <span className="status-vram-tag">
          {readiness.free_vram_gb > 0 ? `${readiness.free_vram_gb.toFixed(1)}GB FREE` : "CPU INFERENCE"}
        </span>
      )}
    </Link>
  );
}
