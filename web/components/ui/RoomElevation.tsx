export function RoomElevation({ variant = "large" }: { variant?: "large" | "small" }) {
  return <svg className={`room-elevation room-elevation-${variant}`} viewBox="0 0 960 520" role="img" aria-label="A restrained architectural elevation drawing used as a static room image placeholder" preserveAspectRatio="xMidYMid meet">
    <g fill="none" stroke="currentColor" strokeWidth="1">
      <path d="M90 410V126h780v284M90 410h780M90 126h780" />
      <path d="M90 160h780M90 375h780M118 126v284M842 126v284" opacity=".4" />
      <path d="M230 410V220h178v190M552 410V220h178v190" />
      <path d="M250 240h138M572 240h138M320 220v190M642 220v190" opacity=".5" />
      <path d="M90 438h780M90 448v-20M870 448v-20M90 465h780M90 475v-20M870 475v-20" opacity=".55" />
      <path d="M170 126v-22h90v22M690 126v-22h90v22" opacity=".5" />
      <path d="M422 284h116M422 300h116M422 316h116" opacity=".35" />
    </g>
    <g fill="currentColor" fontFamily="monospace" fontSize="11" letterSpacing="2">
      <text x="90" y="76">SPATIAL STUDY / 001</text>
      <text x="870" y="76" textAnchor="end">ELEVATION / N.T.S.</text>
      <text x="90" y="502">SOURCE / SINGLE RGB VIEW</text>
      <text x="870" y="502" textAnchor="end">ROOMAI / STATIC PLATE</text>
      <text x="318" y="430" textAnchor="middle">01</text>
      <text x="640" y="430" textAnchor="middle">02</text>
    </g>
  </svg>;
}
