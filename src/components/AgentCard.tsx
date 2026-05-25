import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { BarChart3, SpellCheck, SearchCheck, Brain } from "lucide-react";
import { motion } from "framer-motion";

const iconMap: Record<string, React.ReactNode> = {
  BarChart3: <BarChart3 size={24} />,
  SpellCheck: <SpellCheck size={24} />,
  SearchCheck: <SearchCheck size={24} />,
  Brain: <Brain size={24} />,
};

interface AgentCardProps {
  agent: {
    id: string;
    name: string;
    icon: string;
    description: string;
    details: string;
    color: string;
  };
  index: number;
  isOpen: boolean;
  onToggle: () => void;
  modalContent?: React.ReactNode;
}

const AgentCard = ({ agent, index, isOpen, onToggle, modalContent }: AgentCardProps) => {
  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1, duration: 0.4 }}
        onClick={onToggle}
        className="group cursor-pointer rounded-lg border border-slate-200 bg-white p-5 shadow-sm transition-all hover:border-yellow-500/40 hover:shadow-md dark:border-border dark:bg-card dark:shadow-none dark:hover:border-primary/40 dark:hover:cyber-glow"
      >
        <div className={`mb-3 inline-flex rounded-md bg-gradient-to-br ${agent.color} p-2.5 text-primary-foreground`}>
          {iconMap[agent.icon]}
        </div>
        <h3 className="mb-2 text-sm font-semibold leading-snug text-slate-900 dark:text-foreground">{agent.name}</h3>
        <p className="text-xs leading-relaxed text-slate-600 dark:text-muted-foreground">{agent.description}</p>
      </motion.div>

      <Dialog open={isOpen} onOpenChange={(open) => !open && onToggle()}>
        <DialogContent className="max-h-[85vh] overflow-y-auto border-slate-200 bg-white sm:max-w-xl dark:border-border dark:bg-card">
          <DialogHeader>
            <div className={`mb-3 inline-flex w-fit rounded-md bg-gradient-to-br ${agent.color} p-3 text-primary-foreground`}>
              {iconMap[agent.icon]}
            </div>
            <DialogTitle className="text-slate-900 dark:text-foreground">{agent.name}</DialogTitle>
            <DialogDescription className="max-w-none text-sm leading-7 text-slate-600 dark:text-muted-foreground">
              {agent.details}
            </DialogDescription>
          </DialogHeader>
          {modalContent}
        </DialogContent>
      </Dialog>
    </>
  );
};

export default AgentCard;
