"use client";

import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useRef,
  useSyncExternalStore,
} from "react";

/* -------------------------------------------------------------------------- */
/* Representation states                                                      */
/* -------------------------------------------------------------------------- */

export const REPRESENTATIONS = ["REALITY", "DEPTH", "POINTS", "GEOMETRY", "WORLD", "HYBRID"] as const;

export type Representation = (typeof REPRESENTATIONS)[number];

export function isRepresentation(value: unknown): value is Representation {
  return typeof value === "string" && (REPRESENTATIONS as readonly string[]).includes(value);
}

/* -------------------------------------------------------------------------- */
/* Store (plain object, no React state)                                       */
/* -------------------------------------------------------------------------- */

export interface RepresentationStore {
  /** Current representation. */
  getRepresentation(): Representation;
  /** The representation that was current before the last change (equals current until one happens). */
  getPrevious(): Representation;
  /**
   * Transition progress 0 → 1 from `previous` to the current representation.
   * Read it imperatively (useFrame, GSAP, etc.); it never triggers a React render.
   */
  getProgress(): number;
  /** Select a representation. `progress` defaults to 1 (an instant switch). */
  setRepresentation(next: Representation, progress?: number): void;
  /** Set transition progress (clamped to 0–1). */
  setProgress(progress: number): void;
  /** Subscribe to any change. Returns an unsubscribe function. */
  subscribe(listener: () => void): () => void;
}

const clamp01 = (v: number) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0);

function assertRepresentation(value: unknown): asserts value is Representation {
  if (!isRepresentation(value)) {
    throw new Error(
      `RepresentationController: invalid representation "${String(value)}". Expected one of ${REPRESENTATIONS.join(", ")}.`,
    );
  }
}

export function createRepresentationStore(
  initial: Representation = "REALITY",
  initialProgress = 1,
): RepresentationStore {
  assertRepresentation(initial);
  let representation = initial;
  let previous = initial;
  let progress = clamp01(initialProgress);
  const listeners = new Set<() => void>();
  const emit = () => listeners.forEach((l) => l());

  return {
    getRepresentation: () => representation,
    getPrevious: () => previous,
    getProgress: () => progress,
    setRepresentation(next, nextProgress = 1) {
      assertRepresentation(next);
      const p = clamp01(nextProgress);
      if (next === representation && p === progress) return;
      if (next !== representation) {
        previous = representation;
        representation = next;
      }
      progress = p;
      emit();
    },
    setProgress(next) {
      const p = clamp01(next);
      if (p === progress) return;
      progress = p;
      emit();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

/* -------------------------------------------------------------------------- */
/* React binding                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Exported so the store can be re-provided inside an R3F <Canvas>, because React
 * context does not cross the Canvas boundary:
 *   <Canvas><RepresentationContext.Provider value={store}>…</…></Canvas>
 */
export const RepresentationContext = createContext<RepresentationStore | null>(null);

export interface RepresentationControllerProps {
  children?: ReactNode;
  /**
   * Optional external store (from createRepresentationStore) so code outside
   * React, such as a future master timeline, can drive the same state.
   * If omitted, the controller creates its own.
   */
  store?: RepresentationStore;
  /** Controlled representation. When it changes, it is written to the store. */
  representation?: Representation;
  /** Controlled transition progress (0–1). When it changes, it is written to the store. */
  progress?: number;
  /** Initial representation for an internally created store. Default "REALITY". */
  initialRepresentation?: Representation;
}

export function RepresentationController({
  children,
  store: externalStore,
  representation,
  progress,
  initialRepresentation = "REALITY",
}: RepresentationControllerProps) {
  const internalRef = useRef<RepresentationStore | null>(null);
  if (!externalStore && !internalRef.current) {
    internalRef.current = createRepresentationStore(representation ?? initialRepresentation, progress ?? 1);
  }
  const store = externalStore ?? (internalRef.current as RepresentationStore);

  // Sync controlled props into the store (the store's no-op guards prevent loops).
  useEffect(() => {
    if (representation !== undefined) store.setRepresentation(representation, progress ?? 1);
  }, [store, representation]); // eslint-disable-line react-hooks/exhaustive-deps -- progress handled below
  useEffect(() => {
    if (progress !== undefined) store.setProgress(progress);
  }, [store, progress]);

  return <RepresentationContext.Provider value={store}>{children}</RepresentationContext.Provider>;
}

/** The store itself, for imperative reads/writes. Throws outside a provider. */
export function useRepresentationStore(): RepresentationStore {
  const store = useContext(RepresentationContext);
  if (!store) {
    throw new Error("useRepresentationStore must be used inside <RepresentationController>.");
  }
  return store;
}

/**
 * The current representation. Re-renders only when the representation itself
 * changes, not on progress updates.
 */
export function useRepresentation(): Representation {
  const store = useRepresentationStore();
  return useSyncExternalStore(store.subscribe, store.getRepresentation, store.getRepresentation);
}
