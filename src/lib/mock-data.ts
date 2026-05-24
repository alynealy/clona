export const agents = [
  {
    id: "statistic",
    name: "Statistic Agent",
    icon: "BarChart3",
    description: "Detects lexical repetition, sentence patterns, and entropy signals.",
    details:
      "This agent analyzes the repetition of connective words, evaluates average sentence length, and measures word predictability through entropy analysis. AI-generated text often exhibits unusually uniform sentence structures and greater lexical predictability, making these patterns useful indicators of synthetic authorship.",
    color: "from-yellow-400 to-amber-500",
  },
  {
    id: "grammatical",
    name: "Grammatical Agent",
    icon: "SpellCheck",
    description: "Reviews grammar, punctuation, formatting, and writing style.",
    details:
      "This agent checks for spelling and punctuation errors, analyzes text formatting patterns, and evaluates the overall writing style. AI text tends to be overly polished with fewer natural errors and a more uniform style compared to human writing.",
    color: "from-cyan-400 to-blue-500",
  },
  {
    id: "factcheck",
    name: "Fact-Checking Agent",
    icon: "SearchCheck",
    description: "Cross-checks claims against trusted factual sources.",
    details:
      "This agent cross-references factual claims made in the text against known databases. AI-generated content sometimes contains plausible-sounding but incorrect or fabricated facts, known as 'hallucinations'.",
    color: "from-emerald-400 to-green-500",
  },
  {
    id: "orchestrator",
    name: "Master Agent",
    icon: "Brain",
    description: "Combines all agent findings into one confidence verdict.",
    details:
      "The Master Agent receives the Statistic, Grammatical, and Fact-Checking results, converts factual trust into AI suspicion, and averages the available scores into one overall AI-written percentage.",
    color: "from-yellow-400 to-yellow-600",
  },
];
