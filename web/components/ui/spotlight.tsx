"use client";

import React, { useRef, useState, useCallback, useEffect } from "react";
import { motion, useSpring, useMotionValue } from "framer-motion";
import { cn } from "@/lib/utils";

type SpotlightProps = {
  className?: string;
  fill?: string;
  size?: number;
};

export function Spotlight({
  className = "",
  fill = "rgba(180, 119, 98, 0.28)", // Default to RoomAI oxide tint
  size = 500,
}: SpotlightProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isHovered, setIsHovered] = useState(false);
  const parentElementRef = useRef<HTMLElement | null>(null);

  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const springConfig = { damping: 25, stiffness: 200, mass: 0.2 };
  const smoothX = useSpring(mouseX, springConfig);
  const smoothY = useSpring(mouseY, springConfig);

  const handleMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      const parent = parentElementRef.current;
      if (!parent) return;
      const { left, top } = parent.getBoundingClientRect();
      mouseX.set(event.clientX - left);
      mouseY.set(event.clientY - top);
    },
    [mouseX, mouseY]
  );

  useEffect(() => {
    if (!containerRef.current) return;
    const parent = containerRef.current.parentElement;
    if (!parent) return;

    parentElementRef.current = parent;

    // Check parent positioning without destroying existing class-based layout
    const computedPosition = window.getComputedStyle(parent).position;
    if (computedPosition === "static") {
      parent.style.position = "relative";
    }

    parent.addEventListener("mouseenter", handleMouseEnter);
    parent.addEventListener("mouseleave", handleMouseLeave);
    parent.addEventListener("mousemove", handleMouseMove);

    return () => {
      parent.removeEventListener("mouseenter", handleMouseEnter);
      parent.removeEventListener("mouseleave", handleMouseLeave);
      parent.removeEventListener("mousemove", handleMouseMove);
    };
  }, [handleMouseEnter, handleMouseLeave, handleMouseMove]);

  return (
    <div
      ref={containerRef}
      className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)}
      aria-hidden="true"
    >
      <motion.div
        className="pointer-events-none absolute rounded-full blur-[120px]"
        style={{
          width: size,
          height: size,
          top: smoothY,
          left: smoothX,
          x: "-50%",
          y: "-50%",
          background: `radial-gradient(circle, ${fill} 0%, transparent 70%)`,
          opacity: isHovered ? 1 : 0,
          transition: "opacity 300ms ease",
        }}
      />
    </div>
  );
}
