import { motion } from "framer-motion";
import { useEffect, useState } from "react";

interface ScoreGaugeProps {
  score: number;
  label: string;
  size?: "sm" | "lg";
}

const ScoreGauge = ({ score, label, size = "sm" }: ScoreGaugeProps) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  const isLarge = size === "lg";
  const radius = isLarge ? 80 : 40;
  const stroke = isLarge ? 8 : 5;
  const circumference = 2 * Math.PI * radius;
  const svgSize = (radius + stroke) * 2;

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 200);
    return () => clearTimeout(timer);
  }, [score]);

  const getColor = (s: number) => {
    if (s < 30) return "hsl(142, 71%, 45%)";
    if (s < 60) return "hsl(50, 100%, 50%)";
    return "hsl(0, 84%, 60%)";
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: svgSize, height: svgSize }}>
        <svg width={svgSize} height={svgSize} className="-rotate-90">
          <circle
            cx={radius + stroke}
            cy={radius + stroke}
            r={radius}
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth={stroke}
          />
          <motion.circle
            cx={radius + stroke}
            cy={radius + stroke}
            r={radius}
            fill="none"
            stroke={getColor(score)}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference - (circumference * animatedScore) / 100 }}
            transition={{ duration: 1.5, ease: "easeOut", delay: 0.3 }}
            style={{
              filter: isLarge ? `drop-shadow(0 0 8px ${getColor(score)})` : undefined,
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className={`font-mono font-bold ${isLarge ? "text-3xl" : "text-lg"}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {animatedScore}%
          </motion.span>
        </div>
      </div>
      <span className={`text-center font-medium text-muted-foreground ${isLarge ? "text-sm" : "text-xs"}`}>
        {label}
      </span>
    </div>
  );
};

export default ScoreGauge;
