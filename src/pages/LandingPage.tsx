import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowRight, Shield, Zap, Brain } from "lucide-react";
import AnimatedBackground from "@/components/AnimatedBackground";
import AgentCard from "@/components/AgentCard";
import { agents } from "@/lib/mock-data";
import { useState } from "react";

const pageVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.5, staggerChildren: 0.1 } },
  exit: { opacity: 0 },
};

const LandingPage = () => {
  const [openAgent, setOpenAgent] = useState<string | null>(null);

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="relative min-h-screen bg-slate-50 text-slate-900 dark:bg-background dark:text-foreground"
    >
      <AnimatedBackground />

      {/* Hero */}
      <section className="relative flex min-h-screen flex-col items-center justify-center px-4 pt-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="max-w-3xl"
        >
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-yellow-300 bg-white px-4 py-1.5 text-xs font-medium text-yellow-700 shadow-sm dark:border-primary/30 dark:bg-primary/10 dark:text-primary dark:shadow-none">
            <Zap size={12} />
            Multi-Agent AI Detection
          </div>
          <h1 className="mb-6 text-4xl font-black tracking-tight text-slate-900 dark:text-foreground sm:text-6xl lg:text-7xl">
            AI Content Detection with{" "}
            <span className="text-yellow-600 dark:text-yellow-400">CheckWise</span>
          </h1>
          <p className="mx-auto mb-8 max-w-xl text-base text-slate-600 dark:text-muted-foreground sm:text-lg">
            Powerful AI content detection built for accuracy, speed, and clarity.
          </p>
          <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
            <Link
              to="/checker"
              className="group inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-bold text-primary-foreground transition-all cyber-glow hover:cyber-glow-strong"
            >
              Start Checking
              <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
            </Link>
            <Link
              to="/register"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-6 py-3 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-100 dark:border-border dark:bg-transparent dark:text-foreground dark:shadow-none dark:hover:bg-secondary"
            >
              Create Account
            </Link>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 1 }}
          className="absolute bottom-8 flex flex-col items-center text-xs text-slate-500 dark:text-muted-foreground"
        >
          <span>Scroll to learn more</span>
          <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 1.5, repeat: Infinity }} className="mt-2 h-6 w-4 rounded-full border border-slate-400/40 p-0.5 dark:border-muted-foreground/30">
            <div className="h-1.5 w-full rounded-full bg-slate-400/50 dark:bg-muted-foreground/40" />
          </motion.div>
        </motion.div>
      </section>

      {/* Features */}
      <section className="relative px-4 py-24">
        <div className="container mx-auto max-w-5xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center"
          >
            <h2 className="mb-3 text-2xl font-bold text-slate-900 dark:text-foreground sm:text-3xl">How AI Text Detection Works</h2>
            <p className="mx-auto max-w-2xl text-slate-600 dark:text-muted-foreground">
              Four specialized agents, one final verdict. Each contributes a different text detection approach.
            </p>
          </motion.div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {agents.map((agent, i) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                index={i}
                isOpen={openAgent === agent.id}
                onToggle={() => setOpenAgent(openAgent === agent.id ? null : agent.id)}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="relative border-t border-slate-200 px-4 py-24 dark:border-border">
        <div className="container mx-auto grid max-w-4xl gap-8 sm:grid-cols-3">
          {[
            { icon: Shield, label: "Accuracy Rate", value: "96.4%" },
            { icon: Zap, label: "Avg Response Time", value: "<2s" },
            { icon: Brain, label: "Agents Working", value: "4" },
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15 }}
              className="flex flex-col items-center gap-2 text-center"
            >
              <stat.icon size={20} className="text-yellow-600 dark:text-primary" />
              <span className="text-3xl font-black text-slate-900 dark:text-foreground">{stat.value}</span>
              <span className="text-sm text-slate-600 dark:text-muted-foreground">{stat.label}</span>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="relative border-t border-slate-200 px-4 py-24 text-center dark:border-border">
        <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
          <h2 className="mb-4 text-2xl font-bold text-slate-900 dark:text-foreground">Ready to verify your text?</h2>
          <Link
            to="/checker"
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-8 py-3.5 text-sm font-bold text-primary-foreground cyber-glow hover:cyber-glow-strong"
          >
            Launch Checker <ArrowRight size={16} />
          </Link>
        </motion.div>
      </section>
    </motion.div>
  );
};

export default LandingPage;
