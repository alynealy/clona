import { useEffect, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { BarChart3, FileText, Percent, ShieldCheck, SpellCheck, User } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import type { HistoryEntry, MasterScoreDetail } from "@/lib/statistical-agent";

interface AnalysisSummaryModalProps {
  entry: HistoryEntry | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface InfoItemProps {
  icon: LucideIcon;
  label: string;
  value: string;
}

interface ScorePanelProps {
  icon: LucideIcon;
  title: string;
  score: string;
  caption: string;
  children: ReactNode;
}

const clampPercentage = (score: number) => Math.max(0, Math.min(100, Math.round(score)));

const formatPercentage = (value: number | null | undefined, scaleFraction = false) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "Not available";
  }

  const normalizedValue = scaleFraction && value >= 0 && value <= 1 ? value * 100 : value;
  return `${clampPercentage(normalizedValue)}%`;
};

const formatRating = (value: string | number | null | undefined) => {
  if (typeof value === "number") {
    return `${clampPercentage(value)}%`;
  }

  if (typeof value === "string" && value.trim()) {
    return value.includes("%") ? value : `${value}%`;
  }

  return "Not available";
};

const formatInputType = (value: string) => value.replace(/_/g, " ");

const formatDate = (value: string) => {
  const parsedDate = new Date(value);
  return Number.isNaN(parsedDate.getTime()) ? value : parsedDate.toLocaleString();
};

const getReadableText = (entry: HistoryEntry) =>
  entry.submitted_text?.trim() || entry.text_preview?.trim() || "No submitted text was stored for this analysis.";

const getGrammaticalScoreLabel = (entry: HistoryEntry) => {
  const grammaticalResult = entry.structured_result?.grammatical_result;
  if (!grammaticalResult) {
    return "Not available";
  }

  const isAiVerdict = entry.structured_result?.verdict === "likely AI-generated";
  const displayedScore = isAiVerdict
    ? grammaticalResult.score
    : 100 - grammaticalResult.score;
  const displayedLabel = isAiVerdict ? "likely AI-written" : "likely human-written";

  return `${formatPercentage(displayedScore)} ${displayedLabel}`;
};

const getFactCheckingAiSuspicion = (trustScore: number | null | undefined) => {
  if (typeof trustScore !== "number" || Number.isNaN(trustScore)) {
    return "Not available";
  }

  return `${formatPercentage(100 - trustScore)} AI suspicion conversion`;
};

const findMasterScore = (scores: MasterScoreDetail[] | undefined, agent: string) =>
  scores?.find((scoreDetail) => scoreDetail.agent === agent);

const InfoItem = ({ icon: Icon, label, value }: InfoItemProps) => (
  <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-border dark:bg-background/40">
    <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-md border border-amber-100 bg-amber-50 dark:border-amber-700/40 dark:bg-amber-900/20">
      <Icon size={17} className="text-amber-700 dark:text-amber-400" />
    </div>
    <p className="text-xs font-medium text-slate-600 dark:text-muted-foreground">{label}</p>
    <p className="mt-1 break-words text-sm font-semibold leading-6 text-slate-900 dark:text-foreground">{value}</p>
  </div>
);

const ScorePanel = ({ icon: Icon, title, score, caption, children }: ScorePanelProps) => (
  <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-border dark:bg-background/40">
    <div className="mb-4 flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0">
        <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-md border border-amber-100 bg-amber-50 dark:border-amber-700/40 dark:bg-amber-900/20">
          <Icon size={17} className="text-amber-700 dark:text-amber-400" />
        </div>
        <h3 className="text-sm font-semibold text-slate-900 dark:text-foreground">{title}</h3>
      </div>
      <Badge
        variant="outline"
        className="max-w-full shrink whitespace-normal break-words border-transparent bg-amber-100 px-2 py-1 text-center text-[11px] leading-snug text-amber-800 sm:max-w-[9rem] dark:border-amber-700/60 dark:bg-amber-900/30 dark:text-amber-400"
      >
        {score}
      </Badge>
    </div>
    <p className="mb-4 text-xs text-slate-600 dark:text-muted-foreground">{caption}</p>
    <div className="space-y-3 text-sm text-slate-600 dark:text-muted-foreground">{children}</div>
  </div>
);

const DetailRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-start justify-between gap-4 border-t border-slate-200 pt-3 first:border-t-0 first:pt-0 dark:border-border">
    <span className="text-xs text-slate-600 dark:text-muted-foreground">{label}</span>
    <span className="max-w-[60%] text-right text-xs font-semibold text-slate-900 dark:text-foreground">{value}</span>
  </div>
);

const ReasonList = ({ title, items }: { title: string; items: string[] }) => (
  <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-border dark:bg-background/40">
    <h3 className="mb-3 text-sm font-semibold text-slate-900 dark:text-foreground">{title}</h3>
    {items.length > 0 ? (
      <div className="space-y-2 text-sm leading-7 text-slate-600 dark:text-muted-foreground">
        {items.map((item, index) => (
          <p key={`${item}-${index}`}>- {item}</p>
        ))}
      </div>
    ) : (
      <p className="text-sm text-slate-600 dark:text-muted-foreground">No additional details were reported.</p>
    )}
  </div>
);

const AnalysisSummaryModal = ({ entry, open, onOpenChange }: AnalysisSummaryModalProps) => {
  useEffect(() => {
    if (!open || typeof document === "undefined") {
      return undefined;
    }

    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [open]);

  const structuredResult = entry?.structured_result;
  const signalBreakdown = structuredResult?.signal_breakdown;
  const grammaticalResult = structuredResult?.grammatical_result;
  const factCheckingResult = structuredResult?.fact_checking_result;
  const masterScores = structuredResult?.master_result?.details?.scores;
  const statisticalMasterScore = findMasterScore(masterScores, "statistical");
  const grammaticalMasterScore = findMasterScore(masterScores, "grammatical");
  const factCheckingMasterScore = findMasterScore(masterScores, "fact_checking");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[92vh] max-w-4xl overflow-hidden border-slate-200 bg-white p-0 text-slate-900 shadow-2xl dark:border-border dark:bg-card dark:text-foreground [&>button]:text-slate-500 [&>button:hover]:bg-slate-100 [&>button:hover]:text-slate-800 dark:[&>button]:text-muted-foreground dark:[&>button:hover]:bg-muted/50 dark:[&>button:hover]:text-foreground">
        {entry && (
          <>
            <div className="border-b border-slate-200 bg-white px-5 py-5 dark:border-border dark:bg-muted/20 sm:px-6">
              <DialogHeader className="pr-8">
                <DialogTitle className="text-xl font-semibold text-slate-900 dark:text-foreground">Analysis Summary</DialogTitle>
                <DialogDescription className="pt-2 text-slate-600 dark:text-muted-foreground">
                  Detailed agent results for this saved analysis.
                </DialogDescription>
                <div className="flex flex-wrap items-center gap-2 pt-1 text-sm text-slate-600 dark:text-muted-foreground">
                  <Badge variant="secondary" className="border-transparent bg-amber-100 text-amber-800 dark:border-amber-700/60 dark:bg-amber-900/30 dark:text-amber-400">#{entry.id}</Badge>
                  <Badge variant="outline" className="border-transparent bg-slate-100 text-slate-700 dark:border-border dark:bg-muted/30 dark:text-muted-foreground">{formatInputType(entry.input_type)}</Badge>
                  <span>{formatDate(entry.created_at)}</span>
                </div>
              </DialogHeader>
            </div>

            <div className="max-h-[calc(92vh-112px)] overflow-y-auto px-5 py-5 sm:px-6">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <InfoItem icon={User} label="User" value={entry.user_email} />
                <InfoItem icon={FileText} label="Input Type" value={formatInputType(entry.input_type)} />
                <InfoItem icon={ShieldCheck} label="Rating" value={formatRating(entry.verification_rating)} />
                <InfoItem icon={BarChart3} label="Confidence" value={entry.confidence || "Not available"} />
              </div>

              <section className="mt-5">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <h2 className="text-base font-semibold text-slate-900 dark:text-foreground">Full Text / Preview</h2>
                  <Badge variant="outline" className="border-transparent bg-slate-100 text-slate-700 dark:border-border dark:bg-muted/30 dark:text-muted-foreground">{getReadableText(entry).length.toLocaleString()} chars</Badge>
                </div>
                <div className="max-h-56 overflow-y-auto whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm leading-7 text-slate-900 dark:border-border dark:bg-background/40 dark:text-foreground">
                  {getReadableText(entry)}
                </div>
              </section>

              <Separator className="my-5 bg-slate-200 dark:bg-border" />

              <section>
                <h2 className="mb-3 text-base font-semibold text-slate-900 dark:text-foreground">Agent Scores</h2>
                <div className="grid gap-3 lg:grid-cols-3">
                  <ScorePanel
                    icon={Percent}
                    title="Statistical Agent"
                    score={`${entry.statistical_percentage}%`}
                    caption={`${entry.confidence || "Unknown"} confidence from the statistical result`}
                  >
                    <DetailRow
                      label="Document AI likelihood"
                      value={formatPercentage(structuredResult?.document_assessment?.ai_likelihood_score, true)}
                    />
                    <DetailRow label="Verdict" value={structuredResult?.final_label || entry.verification_rating?.toString() || "Not available"} />
                    <DetailRow label="Semantic model" value={formatPercentage(signalBreakdown?.semantic_model_score, true)} />
                    <DetailRow label="Stylometric score" value={formatPercentage(signalBreakdown?.stylometric_score, true)} />
                    <DetailRow label="Robustness score" value={formatPercentage(signalBreakdown?.robustness_score, true)} />
                    <DetailRow label="Master input" value={formatPercentage(statisticalMasterScore?.score)} />
                  </ScorePanel>

                  <ScorePanel
                    icon={SpellCheck}
                    title="Grammatical Agent"
                    score={getGrammaticalScoreLabel(entry)}
                    caption={`${grammaticalResult?.confidence || "Unknown"} confidence from grammar-based signals`}
                  >
                    <DetailRow label="Raw grammar score" value={formatPercentage(grammaticalResult?.score)} />
                    <DetailRow label="Displayed result" value={getGrammaticalScoreLabel(entry)} />
                    <DetailRow label="Reasons reported" value={`${grammaticalResult?.reasons_for_rating?.length ?? 0}`} />
                    <DetailRow
                      label="Confidence reducers"
                      value={`${grammaticalResult?.lowered_confidence_reasons?.length ?? 0}`}
                    />
                    <DetailRow label="Master input" value={formatPercentage(grammaticalMasterScore?.score)} />
                  </ScorePanel>

                  <ScorePanel
                    icon={ShieldCheck}
                    title="Fact-Checking Agent"
                    score={formatPercentage(factCheckingResult?.overall_trust_score)}
                    caption="Factual trust score from extracted claims"
                  >
                    <DetailRow label="Factual trust" value={formatPercentage(factCheckingResult?.overall_trust_score)} />
                    <DetailRow label="Confidence" value={formatPercentage(factCheckingResult?.overall_confidence_score)} />
                    <DetailRow label="Claims checked" value={`${factCheckingResult?.total_claims ?? 0}`} />
                    <DetailRow label="Converted score" value={getFactCheckingAiSuspicion(factCheckingResult?.overall_trust_score)} />
                    <DetailRow label="Master input" value={formatPercentage(factCheckingMasterScore?.score)} />
                  </ScorePanel>
                </div>
              </section>

              <section className="mt-5 grid gap-3 lg:grid-cols-2">
                <ReasonList title="Statistical Summary" items={structuredResult?.summary ?? structuredResult?.why ?? []} />
                <ReasonList title="Grammatical Notes" items={grammaticalResult?.reasons_for_rating ?? []} />
              </section>

              <section className="mt-3 grid gap-3 lg:grid-cols-2">
                <ReasonList
                  title="Confidence Notes"
                  items={structuredResult?.limitations ?? structuredResult?.what_weakens_the_conclusion ?? []}
                />
                <ReasonList title="Grammar Confidence Notes" items={grammaticalResult?.lowered_confidence_reasons ?? []} />
              </section>

              <section className="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-border dark:bg-background/40">
                <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <h2 className="text-base font-semibold text-slate-900 dark:text-foreground">Fact-Checking Details</h2>
                  <Badge variant="outline" className="border-transparent bg-slate-100 text-slate-700 dark:border-border dark:bg-muted/30 dark:text-muted-foreground">{factCheckingResult?.total_claims ?? 0} claims</Badge>
                </div>
                {factCheckingResult?.claims?.length ? (
                  <div className="space-y-3">
                    {factCheckingResult.claims.map((claim, index) => (
                      <div key={`${claim.claim}-${index}`} className="rounded-md border border-slate-200 bg-white p-3 dark:border-border dark:bg-card">
                        <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                          <div>
                            <p className="break-words text-sm font-semibold leading-6 text-slate-900 dark:text-foreground">{claim.claim}</p>
                            <p className="mt-1 text-xs text-slate-600 dark:text-muted-foreground">{claim.type.replace(/_/g, " ")}</p>
                          </div>
                          <Badge variant="secondary" className="border-transparent bg-amber-100 text-amber-800 dark:border-amber-700/60 dark:bg-amber-900/30 dark:text-amber-400">{claim.verdict.replace(/_/g, " ")}</Badge>
                        </div>
                        <div className="grid gap-2 text-xs text-slate-600 dark:text-muted-foreground sm:grid-cols-2">
                          <span>Claim score: {formatPercentage(claim.claim_score)}</span>
                          <span>Confidence: {formatPercentage(claim.confidence_score)}</span>
                        </div>
                        <p className="mt-3 break-words text-sm leading-7 text-slate-600 dark:text-muted-foreground">{claim.explanation}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-slate-600 dark:text-muted-foreground">No factual claims were available for this analysis.</p>
                )}
              </section>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default AnalysisSummaryModal;
