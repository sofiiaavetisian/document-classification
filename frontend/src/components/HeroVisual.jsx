const seedParticles = [
  { x: 374, y: 164, size: 4, delay: 0.08, driftX: 132, driftY: -18, duration: 3.4 },
  { x: 390, y: 170, size: 4, delay: 0.16, driftX: 148, driftY: -6, duration: 3.3 },
  { x: 406, y: 178, size: 3.4, delay: 0.24, driftX: 168, driftY: 10, duration: 3.6 },
  { x: 420, y: 188, size: 3, delay: 0.32, driftX: 160, driftY: -12, duration: 3.2 },
  { x: 434, y: 196, size: 3.8, delay: 0.4, driftX: 180, driftY: 6, duration: 3.5 },
  { x: 448, y: 204, size: 4, delay: 0.5, driftX: 190, driftY: -10, duration: 3.7 },
  { x: 426, y: 214, size: 3.2, delay: 0.58, driftX: 174, driftY: 14, duration: 3.1 },
  { x: 442, y: 222, size: 3.4, delay: 0.66, driftX: 165, driftY: -8, duration: 3.4 },
  { x: 458, y: 232, size: 3, delay: 0.74, driftX: 182, driftY: 12, duration: 3.3 },
  { x: 474, y: 240, size: 3.6, delay: 0.82, driftX: 194, driftY: -6, duration: 3.6 },
  { x: 428, y: 248, size: 3.2, delay: 0.9, driftX: 170, driftY: 20, duration: 3.2 },
  { x: 446, y: 258, size: 3.8, delay: 1, driftX: 184, driftY: -14, duration: 3.5 },
  { x: 462, y: 268, size: 3.2, delay: 1.08, driftX: 176, driftY: 16, duration: 3.2 },
  { x: 478, y: 276, size: 3, delay: 1.16, driftX: 196, driftY: -9, duration: 3.6 },
  { x: 494, y: 286, size: 2.8, delay: 1.24, driftX: 206, driftY: 8, duration: 3.3 },
  { x: 380, y: 284, size: 3, delay: 1.32, driftX: 162, driftY: -12, duration: 3.4 },
  { x: 398, y: 294, size: 2.8, delay: 1.4, driftX: 174, driftY: 10, duration: 3.5 },
  { x: 416, y: 302, size: 2.6, delay: 1.48, driftX: 182, driftY: -8, duration: 3.2 },
  { x: 456, y: 168, size: 2.8, delay: 1.56, driftX: 194, driftY: -16, duration: 3.6 },
  { x: 476, y: 178, size: 3, delay: 1.64, driftX: 204, driftY: -4, duration: 3.3 },
  { x: 498, y: 190, size: 2.8, delay: 1.72, driftX: 214, driftY: 14, duration: 3.7 },
  { x: 520, y: 202, size: 2.8, delay: 1.8, driftX: 220, driftY: -10, duration: 3.4 },
  { x: 538, y: 214, size: 2.6, delay: 1.88, driftX: 228, driftY: 8, duration: 3.3 },
  { x: 512, y: 226, size: 2.8, delay: 1.96, driftX: 218, driftY: -6, duration: 3.5 },
  { x: 530, y: 236, size: 2.6, delay: 2.04, driftX: 226, driftY: 12, duration: 3.2 },
  { x: 548, y: 248, size: 2.6, delay: 2.12, driftX: 236, driftY: -7, duration: 3.4 },
  { x: 518, y: 258, size: 2.4, delay: 2.2, driftX: 222, driftY: 16, duration: 3.6 },
  { x: 536, y: 270, size: 2.4, delay: 2.28, driftX: 232, driftY: -12, duration: 3.3 },
  { x: 554, y: 282, size: 2.4, delay: 2.36, driftX: 242, driftY: 10, duration: 3.5 },
  { x: 492, y: 302, size: 2.2, delay: 2.44, driftX: 218, driftY: -8, duration: 3.3 },
]

const particles = [
  ...seedParticles,
  ...seedParticles.map((p, i) => ({
    x: p.x + 10 + (i % 4) * 5,
    y: p.y - 10 + (i % 5) * 4,
    size: p.size,
    delay: p.delay + 0.12,
    driftX: p.driftX + 24 + (i % 6) * 5,
    driftY: p.driftY + (-8 + (i % 7) * 3),
    duration: p.duration + 0.12,
  })),
  ...seedParticles.map((p, i) => ({
    x: p.x + 18 + (i % 5) * 4,
    y: p.y + 8 + (i % 4) * 3,
    size: p.size,
    delay: p.delay + 0.24,
    driftX: p.driftX + 40 + (i % 7) * 4,
    driftY: p.driftY + (-10 + (i % 8) * 3),
    duration: p.duration + 0.22,
  })),
  ...seedParticles.map((p, i) => ({
    x: p.x + 26 + (i % 6) * 3,
    y: p.y - 4 + (i % 6) * 3,
    size: p.size,
    delay: p.delay + 0.36,
    driftX: p.driftX + 52 + (i % 5) * 6,
    driftY: p.driftY + (-6 + (i % 6) * 3),
    duration: p.duration + 0.32,
  })),
]

function HeroVisual() {
  return (
    <div className="hero-visual reveal-hero delay-2" aria-hidden="true">
      <svg viewBox="0 0 920 420" role="presentation">
        <defs>
          <linearGradient id="docStroke" x1="156" y1="78" x2="402" y2="334">
            <stop offset="0%" stopColor="#c0d0ca" stopOpacity="0.92" />
            <stop offset="65%" stopColor="#a5b9b1" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#a5b9b1" stopOpacity="0.06" />
          </linearGradient>
          <linearGradient id="docFill" x1="156" y1="80" x2="414" y2="80">
            <stop offset="0%" stopColor="#1c2926" stopOpacity="0.8" />
            <stop offset="58%" stopColor="#1a2724" stopOpacity="0.62" />
            <stop offset="100%" stopColor="#1a2724" stopOpacity="0.03" />
          </linearGradient>
          <linearGradient id="docLine" x1="180" y1="0" x2="408" y2="0">
            <stop offset="0%" stopColor="#9db2aa" stopOpacity="0.56" />
            <stop offset="100%" stopColor="#9db2aa" stopOpacity="0.04" />
          </linearGradient>
          <radialGradient id="particleGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#dee9e5" stopOpacity="0.38" />
            <stop offset="100%" stopColor="#dee9e5" stopOpacity="0" />
          </radialGradient>
        </defs>

        <ellipse cx="350" cy="210" rx="258" ry="178" fill="url(#particleGlow)" />

        <g className="document-shape">
          <rect x="156" y="80" width="258" height="260" rx="24" fill="url(#docFill)" stroke="url(#docStroke)" strokeWidth="1.5" />
          <path d="M346 80 L414 80 L414 148" fill="none" stroke="url(#docStroke)" strokeWidth="1.5" />
          <path d="M346 80 C380 80 414 112 414 148" fill="rgba(136, 157, 148, 0.16)" />

          <rect x="188" y="136" width="48" height="30" rx="7" fill="#9ab5ab" opacity="0.9" />
          <rect x="254" y="142" width="122" height="8" rx="4" fill="url(#docLine)" />
          <rect x="254" y="158" width="92" height="8" rx="4" fill="url(#docLine)" />
          <rect x="188" y="196" width="178" height="8" rx="4" fill="url(#docLine)" />
          <rect x="188" y="220" width="172" height="8" rx="4" fill="url(#docLine)" />
          <rect x="188" y="244" width="162" height="8" rx="4" fill="url(#docLine)" />
          <rect x="188" y="268" width="132" height="8" rx="4" fill="url(#docLine)" />
        </g>

        <g className="flow-particles">
          {particles.map((particle, idx) => (
            <rect
              key={`${particle.x}-${particle.y}-${idx}`}
              className="pixel"
              x={particle.x}
              y={particle.y}
              width={particle.size}
              height={particle.size}
              rx={0.5}
              style={{
                '--delay': `${particle.delay}s`,
                '--drift-x': `${particle.driftX}px`,
                '--drift-y': `${particle.driftY}px`,
                '--duration': `${particle.duration}s`,
              }}
            />
          ))}
        </g>
      </svg>
    </div>
  )
}

export default HeroVisual
