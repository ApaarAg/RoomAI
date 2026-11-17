import type { ReactNode } from "react";

export function TechnicalLabel({ children }: { children: ReactNode }) {
  return <span className="technical-label">{children}</span>;
}
