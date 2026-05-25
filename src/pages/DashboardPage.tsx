import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Calendar, Percent, FileText } from "lucide-react";
import AnalysisSummaryModal from "@/components/AnalysisSummaryModal";
import { fetchHistory, type HistoryEntry } from "@/lib/statistical-agent";

interface DashboardPageProps {
  userEmail?: string;
}

const clampPercentage = (score: number) => Math.max(0, Math.min(100, Math.round(score)));

const formatGrammaticalHistoryScore = (item: HistoryEntry) => {
  const grammaticalResult = item.structured_result?.grammatical_result;
  if (!grammaticalResult) {
    return "Not available";
  }

  const isAiVerdict = item.structured_result?.verdict === "likely AI-generated";
  const displayedScore = isAiVerdict
    ? grammaticalResult.score
    : 100 - grammaticalResult.score;
  const displayedLabel = isAiVerdict ? "AI-written" : "human-written";

  return `${clampPercentage(displayedScore)}% ${displayedLabel} (${grammaticalResult.confidence})`;
};

const formatFactCheckingHistoryScore = (item: HistoryEntry) => {
  const factCheckingResult = item.structured_result?.fact_checking_result;
  if (!factCheckingResult) {
    return "-";
  }

  return `${clampPercentage(factCheckingResult.overall_trust_score)}% factual trust`;
};

const DashboardPage = ({ userEmail }: DashboardPageProps) => {
  const [selectedEntry, setSelectedEntry] = useState<HistoryEntry | null>(null);
  const historyQuery = useQuery({
    queryKey: ["history", userEmail],
    queryFn: () => fetchHistory(userEmail ?? ""),
    enabled: Boolean(userEmail),
  });

  const history = historyQuery.data ?? [];
  const averageScore = history.length
    ? `${Math.round(history.reduce((total, item) => total + item.statistical_percentage, 0) / history.length)}%`
    : "0%";
  const lastCheck = history.length ? new Date(history[0].created_at).toLocaleString() : "No checks yet";
  const closeSummaryModal = (open: boolean) => {
    if (!open) {
      setSelectedEntry(null);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="min-h-screen px-4 pt-24 pb-12">
      <div className="container mx-auto max-w-6xl">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="mb-2 text-3xl font-bold">History</h1>
          <p className="mb-7 text-sm text-muted-foreground">Your analysis history at a glance.</p>
        </motion.div>

        <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[
            { icon: FileText, label: "Total Checks", value: history.length },
            { icon: Percent, label: "Avg AI Likelihood", value: averageScore },
            { icon: Calendar, label: "Last Check", value: lastCheck },
          ].map((item, index) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="min-h-[118px] rounded-lg border border-border bg-card p-5 shadow-sm"
            >
              <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-md border border-border bg-background/50">
                <item.icon size={18} className="text-primary" />
              </div>
              <div className="text-xs font-medium text-muted-foreground">{item.label}</div>
              <div className="mt-1 break-words text-lg font-semibold leading-snug text-foreground sm:text-xl">{item.value}</div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="overflow-hidden rounded-lg border border-border bg-card shadow-sm"
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1040px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/40">
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">User</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">Input Type</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">Preview</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">Rating</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">Statistical Agent</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">Grammatical Agent</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">Fact-Checking Agent</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold text-muted-foreground">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {!userEmail && (
                  <tr>
                    <td colSpan={8} className="px-4 py-6 text-center text-muted-foreground">
                      Sign in to view your search history.
                    </td>
                  </tr>
                )}
                {userEmail && historyQuery.isLoading && (
                  <tr>
                    <td colSpan={8} className="px-4 py-6 text-center text-muted-foreground">
                      Loading history...
                    </td>
                  </tr>
                )}
                {userEmail && historyQuery.isError && (
                  <tr>
                    <td colSpan={8} className="px-4 py-6 text-center text-destructive">
                      {historyQuery.error.message}
                    </td>
                  </tr>
                )}
                {userEmail && !historyQuery.isLoading && !historyQuery.isError && history.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-6 text-center text-muted-foreground">
                      No text verification searches have been stored yet.
                    </td>
                  </tr>
                )}
                {history.map((item) => (
                  <tr
                    key={item.id}
                    role="button"
                    tabIndex={0}
                    aria-label={`Open analysis summary for ${item.text_preview}`}
                    onClick={() => setSelectedEntry(item)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setSelectedEntry(item);
                      }
                    }}
                    className="cursor-pointer border-b border-border transition-colors last:border-0 hover:bg-muted/40 focus-visible:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary/50"
                  >
                    <td className="px-5 py-4 text-muted-foreground">{item.user_email}</td>
                    <td className="px-5 py-4">{item.input_type}</td>
                    <td className="max-w-[300px] truncate px-5 py-4">{item.text_preview}</td>
                    <td className="px-5 py-4 font-medium">{item.verification_rating}</td>
                    <td className="px-5 py-4">{item.statistical_percentage}% ({item.confidence})</td>
                    <td className="px-5 py-4">{formatGrammaticalHistoryScore(item)}</td>
                    <td className="px-5 py-4">{formatFactCheckingHistoryScore(item)}</td>
                    <td className="px-5 py-4 text-muted-foreground">{new Date(item.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
      <AnalysisSummaryModal entry={selectedEntry} open={Boolean(selectedEntry)} onOpenChange={closeSummaryModal} />
    </motion.div>
  );
};

export default DashboardPage;
