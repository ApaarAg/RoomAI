/** Temporary scene fixtures for interaction design only; never model output. */
export const USE_DEVELOPMENT_FIXTURES = process.env.NODE_ENV !== "production";

export const developmentRoom = {
  width: 3.2,
  height: 2.4,
  depth: 3.6,
  floorY: -1.2,
  backZ: -3.6,
} as const;
