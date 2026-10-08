/**
 * Arbiter logo — a balance scale that has tipped to a verdict.
 *
 * Concept: every drift tool shows you a balanced scale and leaves you to decide.
 * Arbiter's beam is deliberately tilted — the decision is already made. The low
 * pan is filled (the evidence that won), the high pan is empty (dismissed noise).
 *
 * The notch in the fulcrum reads as a stylised "A" at small sizes.
 */

type LogoProps = {
  /** Rendered size in px (square). Default 32. */
  size?: number;
  /** Show the "Arbiter" wordmark beside the glyph. */
  withWordmark?: boolean;
  className?: string;
};

export function LogoMark({ size = 32, className }: Omit<LogoProps, "withWordmark">) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      role="img"
      aria-label="Arbiter"
      className={className}
    >
      <defs>
        {/* Decided side: the verdict. Warm, confident. */}
        <linearGradient id="arb-decided" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#dc2626" />
        </linearGradient>
        {/* Dismissed side: suppressed noise. Cool, receded. */}
        <linearGradient id="arb-dismissed" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#64748b" />
          <stop offset="100%" stopColor="#475569" />
        </linearGradient>
      </defs>

      {/* Fulcrum — the notch forms an implied "A" */}
      <path
        d="M24 7 L33 41 L28.5 41 L24 24 L19.5 41 L15 41 Z"
        fill="currentColor"
        opacity="0.92"
      />

      {/* Beam, tilted ~14° — the scale has ruled */}
      <g transform="rotate(-14 24 16)">
        <rect x="7" y="14.6" width="34" height="2.8" rx="1.4" fill="currentColor" />

        {/* Low pan — filled: the evidence that carried the decision */}
        <g transform="translate(9.5 16)">
          <line x1="0" y1="0" x2="0" y2="6.5" stroke="currentColor" strokeWidth="1.3" />
          <path d="M-6.5 6.5 A 6.5 6.5 0 0 0 6.5 6.5 Z" fill="url(#arb-decided)" />
        </g>

        {/* High pan — empty outline: the noise that was dismissed */}
        <g transform="translate(38.5 16)">
          <line x1="0" y1="0" x2="0" y2="6.5" stroke="currentColor" strokeWidth="1.3" />
          <path
            d="M-6.5 6.5 A 6.5 6.5 0 0 0 6.5 6.5 Z"
            fill="none"
            stroke="url(#arb-dismissed)"
            strokeWidth="1.6"
          />
        </g>
      </g>
    </svg>
  );
}

export default function Logo({ size = 32, withWordmark = true, className }: LogoProps) {
  if (!withWordmark) return <LogoMark size={size} className={className} />;

  return (
    <div className={`flex items-center gap-2.5 ${className ?? ""}`}>
      <LogoMark size={size} />
      <div className="flex flex-col leading-none">
        <span
          className="font-semibold tracking-tight text-[1.05rem]"
          style={{ letterSpacing: "-0.02em" }}
        >
          Arbiter
        </span>
        <span className="text-[0.6rem] uppercase tracking-[0.14em] text-slate-500">
          AI judgment for ML
        </span>
      </div>
    </div>
  );
}
