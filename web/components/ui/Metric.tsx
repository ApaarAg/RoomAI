export function Metric({ label, value, suffix, description, unit }: { label: string; value: string; suffix?: string; description: string; unit: string }) {
  return <article className="metric">
    <p className="metric-label">{label}<span>{unit}</span></p>
    <p className="metric-value">{value}<small>{suffix}</small></p>
    <p className="metric-description">{description}</p>
  </article>;
}
