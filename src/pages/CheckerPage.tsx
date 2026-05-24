import { useMemo, useState, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useMutation } from "@tanstack/react-query";
import { Send, Loader2 } from "lucide-react";
import AgentCard from "@/components/AgentCard";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { agents } from "@/lib/mock-data";
import { verifyText, type TextVerificationResponse } from "@/lib/statistical-agent";

const MAX_CHARS = 10000;
const MIN_TEXT_CHARS = 10;

const formatVisibleVerdictLabel = (result: TextVerificationResponse) =>
  result.verdict === "likely AI-generated"
    ? `${result.percentage}% likely AI-written`
    : `${result.percentage}% likely human-written`;

const clampPercentage = (score: number) => Math.max(0, Math.min(100, Math.round(score)));

const formatGrammaticalVerdictLabel = (result: TextVerificationResponse) =>
  result.verdict === "likely AI-generated"
    ? `${clampPercentage(result.grammatical_result.score)}% likely AI-written`
    : `${clampPercentage(100 - result.grammatical_result.score)}% likely human-written`;

const formatFactCheckingScore = (score: number) => `${clampPercentage(score)}%`;

const formatFactCheckingVerdictLabel = (score: number) =>
  `${clampPercentage(score)}% factual trust`;

const formatMasterVerdictLabel = (score: number | null | undefined) =>
  typeof score === "number"
    ? `${clampPercentage(score)}% overall likely AI-written`
    : "Not enough agent results to calculate an overall AI-written score";

interface GrammaticalSignalSpan {
  start: number;
  end: number;
  reason: string;
}

const buildGrammaticalSignalSpans = (value: string): GrammaticalSignalSpan[] => {
  const spans: GrammaticalSignalSpan[] = [];

  const addMatches = (pattern: RegExp, reason: string) => {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(value)) !== null && spans.length < 8) {
      const start = match.index;
      const end = start + match[0].length;
      if (end > start) {
        spans.push({ start, end, reason });
      }
    }
  };

  addMatches(/\s+[,.!?;:]/g, "Spacing appears before punctuation.");
  addMatches(/[,.!?;:](?=\S)/g, "Punctuation is followed by no space.");
  addMatches(/\s{2,}/g, "Repeated spacing affects formatting consistency.");
  addMatches(/\n{3,}/g, "Large blank gaps affect formatting consistency.");
  addMatches(/\b\w*(?:aaa|eee|iii|ooo|uuu)\w*\b/gi, "Repeated letters may indicate informal or typo-like wording.");

  const filteredSpans: GrammaticalSignalSpan[] = [];
  let cursor = 0;
  [...spans]
    .sort((a, b) => a.start - b.start)
    .forEach((span) => {
      if (span.start >= cursor) {
        filteredSpans.push(span);
        cursor = span.end;
      }
    });

  return filteredSpans;
};

interface CheckerPageProps {
  userEmail?: string;
}

const CheckerPage = ({ userEmail }: CheckerPageProps) => {
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [openAgent, setOpenAgent] = useState<string | null>(null);
  const [urlLoading, setUrlLoading] = useState(false);
  const [urlStatus, setUrlStatus] = useState<string | null>(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textValidationError, setTextValidationError] = useState<string | null>(null);
  const resolvedUserEmail =
    userEmail ??
    (typeof window !== "undefined"
      ? (() => {
          const storedUser = window.localStorage.getItem("checkwise-user");
          if (!storedUser) return undefined;

          try {
            const parsedUser = JSON.parse(storedUser) as { email?: string };
            return parsedUser.email;
          } catch {
            return undefined;
          }
        })()
      : undefined);
  const trimmedUserEmail = resolvedUserEmail?.trim();

  const textVerificationMutation = useMutation<
    TextVerificationResponse,
    Error,
    { text: string }
  >({
    mutationFn: ({ text }) => {
      console.log("handleVerifyText: mutationFn started", {
        hasUserEmail: Boolean(trimmedUserEmail),
        textLength: text.length,
      });

      if (!trimmedUserEmail || trimmedUserEmail.length < 3) {
        throw new Error("Please sign in before verifying text.");
      }

      console.log("handleVerifyText: calling verifyText");
      return verifyText({ userEmail: trimmedUserEmail, text });
    },
    onError: (error) => {
      console.error("Text verification mutation failed", error);
    },
  });

  const handleTextSubmit = async () => {
    const trimmedText = text.trim();
    setTextValidationError(null);
    console.log("handleVerifyText: button clicked", {
      rawLength: text.length,
      trimmedLength: trimmedText.length,
      isPending: textVerificationMutation.isPending,
      textLoading,
    });

    if (!trimmedText) {
      console.warn("handleVerifyText: aborting because text is empty");
      setTextValidationError("Please paste or type text before verifying.");
      return;
    }

    if (textLoading || textVerificationMutation.isPending) {
      console.warn("handleVerifyText: aborting because a text verification request is already running");
      return;
    }

    if (trimmedText.length < MIN_TEXT_CHARS) {
      console.warn("handleVerifyText: aborting because text is too short");
      setTextValidationError(`Please enter at least ${MIN_TEXT_CHARS} characters of text.`);
      return;
    }

    if (!trimmedUserEmail || trimmedUserEmail.length < 3) {
      console.warn("handleVerifyText: aborting because the user email is invalid");
      setTextValidationError("Please sign in with a valid email before verifying text.");
      return;
    }

    try {
      setTextLoading(true);
      console.log("handleVerifyText: invoking mutateAsync");
      await textVerificationMutation.mutateAsync({ text: trimmedText });
      console.log("handleVerifyText: mutation resolved successfully");
    } catch (error) {
      console.error("handleVerifyText: mutation rejected", error);
    } finally {
      setTextLoading(false);
      console.log("handleVerifyText: loading state reset");
    }
  };

const handleUrlSubmit = async () => {
  const trimmedUrl = url.trim();
  if (!trimmedUrl || urlLoading) return;
  if (!trimmedUserEmail) {
    setUrlStatus("Please sign in before verifying a URL.");
    return;
  }
  try {
    setUrlLoading(true);
    setUrlStatus("Fetching and analyzing content from URL...");
    await textVerificationMutation.mutateAsync({ text: trimmedUrl });
    setUrlStatus("Analysis complete!");
  } catch (error: any) {
    setUrlStatus(`Error: ${error.message}`);
    console.error("URL Submit Error:", error);
  } finally {
    setUrlLoading(false);
  }
};

  const highlightedText = useMemo(() => {
    const result = textVerificationMutation.data;
    const spans = result?.highlights?.length ? result.highlights : [];

    if (!result || spans.length === 0) {
      return textVerificationMutation.data ? [text] : null;
    }

    const sortedSpans = [...spans].sort((a, b) => a.start - b.start);
    const nodes: ReactNode[] = [];
    let cursor = 0;

    sortedSpans.forEach((span, index) => {
      const safeStart = Math.max(cursor, span.start);
      const safeEnd = Math.min(text.length, span.end);
      if (safeStart > cursor) {
        nodes.push(<span key={`plain-${index}-${cursor}`}>{text.slice(cursor, safeStart)}</span>);
      }
      if (safeEnd > safeStart) {
        nodes.push(
          <mark
            key={`highlight-${index}-${safeStart}`}
            className="rounded bg-blue-500/20 px-0.5 text-foreground"
            title={"reason" in span ? span.reason : "highlighted signal"}
          >
            {text.slice(safeStart, safeEnd)}
          </mark>,
        );
        cursor = safeEnd;
      }
    });

    if (cursor < text.length) {
      nodes.push(<span key={`plain-final-${cursor}`}>{text.slice(cursor)}</span>);
    }

    return nodes;
  }, [text, textVerificationMutation.data]);

  const scorePercentage = textVerificationMutation.data
    ? textVerificationMutation.data.percentage
    : null;
  const visibleVerdictLabel = textVerificationMutation.data
    ? formatVisibleVerdictLabel(textVerificationMutation.data)
    : null;
  const grammaticalVerdictLabel = textVerificationMutation.data
    ? formatGrammaticalVerdictLabel(textVerificationMutation.data)
    : null;
  const grammaticalResult = textVerificationMutation.data?.grammatical_result ?? null;
  const factCheckingResult = textVerificationMutation.data?.fact_checking_result ?? null;
  const masterResult = textVerificationMutation.data?.master_result ?? null;
  const masterVerdictLabel = masterResult ? formatMasterVerdictLabel(masterResult.score) : null;
  const grammaticalHighlightedText = useMemo(() => {
    if (!grammaticalResult) {
      return null;
    }

    const spans = buildGrammaticalSignalSpans(text);
    if (spans.length === 0) {
      return [text];
    }

    const nodes: ReactNode[] = [];
    let cursor = 0;

    spans.forEach((span, index) => {
      if (span.start > cursor) {
        nodes.push(<span key={`grammar-plain-${index}-${cursor}`}>{text.slice(cursor, span.start)}</span>);
      }

      nodes.push(
        <mark
          key={`grammar-highlight-${index}-${span.start}`}
          className="rounded bg-cyan-500/20 px-0.5 text-foreground"
          title={span.reason}
        >
          {text.slice(span.start, span.end)}
        </mark>,
      );
      cursor = span.end;
    });

    if (cursor < text.length) {
      nodes.push(<span key={`grammar-plain-final-${cursor}`}>{text.slice(cursor)}</span>);
    }

    return nodes;
  }, [text, grammaticalResult]);

  const statisticalAgentModalContent = textVerificationMutation.data ? (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h2 className="mb-2 text-base font-semibold text-foreground">{textVerificationMutation.data.verification_title}</h2>
        <p className="text-sm text-muted-foreground">
          Result: <span className="font-semibold text-foreground">{visibleVerdictLabel}</span>
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Confidence: <span className="font-semibold text-foreground">{textVerificationMutation.data.document_assessment.confidence}</span>
        </p>
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">Why this text received this rating</h3>
        <div className="space-y-2 text-sm leading-7 text-muted-foreground">
          {textVerificationMutation.data.summary.map((reason) => (
            <p key={reason}>- {reason}</p>
          ))}
        </div>
      </div>

      {textVerificationMutation.data.limitations.length > 0 && (
        <div className="rounded-lg border border-border bg-background/40 p-4">
          <h3 className="mb-3 text-sm font-semibold text-foreground">What lowered confidence</h3>
          <div className="space-y-2 text-sm leading-7 text-muted-foreground">
            {textVerificationMutation.data.limitations.map((item) => (
              <p key={item}>- {item}</p>
            ))}
          </div>
        </div>
      )}

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">Detector details</h3>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            Detector status: <span className="font-semibold text-foreground">{textVerificationMutation.data.detector_details.status.replaceAll("_", " ")}</span>
          </p>
          {textVerificationMutation.data.detector_details.raw_score !== null && (
            <p>
              Raw score extracted: <span className="font-semibold text-foreground">{textVerificationMutation.data.detector_details.raw_score}</span>
            </p>
          )}
          {textVerificationMutation.data.detector_details.observations.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-semibold text-foreground">Observations</h4>
              <div className="space-y-2 leading-7">
                {textVerificationMutation.data.detector_details.observations.map((item) => (
                  <p key={item}>- {item}</p>
                ))}
              </div>
            </div>
          )}
          {textVerificationMutation.data.detector_details.influential_phrases.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-semibold text-foreground">Influential phrases</h4>
              <div className="space-y-2 leading-7">
                {textVerificationMutation.data.detector_details.influential_phrases.map((item) => (
                  <p key={item}>- {item}</p>
                ))}
              </div>
            </div>
          )}
          {textVerificationMutation.data.detector_details.technical_note && (
            <div className="border-t border-border pt-3">
              <h4 className="mb-2 text-sm font-semibold text-foreground">Technical note</h4>
              <p>{textVerificationMutation.data.detector_details.technical_note}</p>
            </div>
          )}
          {(textVerificationMutation.data.detector_details.invoke_error_type ||
            textVerificationMutation.data.detector_details.invoke_error_status_code ||
            textVerificationMutation.data.detector_details.diagnostic_timestamp) && (
            <details className="border-t border-border pt-3">
              <summary className="cursor-pointer text-sm font-semibold text-foreground">
                Developer diagnostics
              </summary>
              <div className="mt-3 space-y-2">
                {textVerificationMutation.data.detector_details.invoke_error_type && (
                  <p>Exception type: {textVerificationMutation.data.detector_details.invoke_error_type}</p>
                )}
                {textVerificationMutation.data.detector_details.invoke_error_message && (
                  <p>Error message: {textVerificationMutation.data.detector_details.invoke_error_message}</p>
                )}
                {textVerificationMutation.data.detector_details.invoke_error_status_code !== null &&
                  textVerificationMutation.data.detector_details.invoke_error_status_code !== undefined && (
                    <p>Status code: {textVerificationMutation.data.detector_details.invoke_error_status_code}</p>
                )}
                {textVerificationMutation.data.detector_details.invoke_error_provider && (
                  <p>Provider: {textVerificationMutation.data.detector_details.invoke_error_provider}</p>
                )}
                {textVerificationMutation.data.detector_details.invoke_error_model && (
                  <p>Model: {textVerificationMutation.data.detector_details.invoke_error_model}</p>
                )}
                {textVerificationMutation.data.detector_details.invoke_error_base_url && (
                  <p>Base URL: {textVerificationMutation.data.detector_details.invoke_error_base_url}</p>
                )}
                {textVerificationMutation.data.detector_details.schema_present_keys &&
                  textVerificationMutation.data.detector_details.schema_present_keys.length > 0 && (
                    <p>Returned keys: {textVerificationMutation.data.detector_details.schema_present_keys.join(", ")}</p>
                )}
                {textVerificationMutation.data.detector_details.invoke_error_timeout_seconds !== null &&
                  textVerificationMutation.data.detector_details.invoke_error_timeout_seconds !== undefined && (
                    <p>Timeout (read): {textVerificationMutation.data.detector_details.invoke_error_timeout_seconds}</p>
                )}
                {textVerificationMutation.data.detector_details.raw_output_excerpt && (
                  <p>Raw output excerpt: {textVerificationMutation.data.detector_details.raw_output_excerpt}</p>
                )}
                {textVerificationMutation.data.detector_details.diagnostic_timestamp && (
                  <p>Captured at: {textVerificationMutation.data.detector_details.diagnostic_timestamp}</p>
                )}
              </div>
            </details>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">Highlighted Signals In The Original Text</h3>
        <div className="whitespace-pre-wrap text-sm leading-7 text-foreground">{highlightedText}</div>
      </div>
    </div>
  ) : null;

  const grammaticalAgentModalContent = grammaticalResult ? (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h2 className="mb-2 text-base font-semibold text-foreground">Grammatical Verification Result</h2>
        <p className="text-sm text-muted-foreground">
          Result: <span className="font-semibold text-foreground">{grammaticalVerdictLabel}</span>
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Confidence: <span className="font-semibold text-foreground">{grammaticalResult.confidence}</span>
        </p>
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">Why this text received this rating</h3>
        <div className="space-y-2 text-sm leading-7 text-muted-foreground">
          {grammaticalResult.reasons_for_rating.map((reason, index) => (
            <p key={`${reason}-${index}`}>- {reason}</p>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">What lowered confidence</h3>
        <div className="space-y-2 text-sm leading-7 text-muted-foreground">
          {grammaticalResult.lowered_confidence_reasons.length > 0 ? (
            grammaticalResult.lowered_confidence_reasons.map((item, index) => (
              <p key={`${item}-${index}`}>- {item}</p>
            ))
          ) : (
            <p>- No major confidence reducers were reported.</p>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">Highlighted Signals In The Original Text</h3>
        <div className="whitespace-pre-wrap text-sm leading-7 text-foreground">{grammaticalHighlightedText}</div>
      </div>
    </div>
  ) : null;

  const factCheckingAgentModalContent = factCheckingResult ? (
    <div className="max-w-full space-y-4 overflow-x-hidden whitespace-normal break-words [overflow-wrap:anywhere] [word-break:break-word]">
      <div className="max-w-full overflow-x-hidden rounded-lg border border-border bg-background/40 p-4">
        <h2 className="mb-2 text-base font-semibold text-foreground">Fact-Checking Verification Result</h2>
        <p className="text-sm text-muted-foreground">
          Result: <span className="font-semibold text-foreground">{formatFactCheckingVerdictLabel(factCheckingResult.overall_trust_score)}</span>
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Lower factual trust can increase AI suspicion.
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Confidence: <span className="font-semibold text-foreground">{formatFactCheckingScore(factCheckingResult.overall_confidence_score)}</span>
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Claims checked: <span className="font-semibold text-foreground">{factCheckingResult.total_claims}</span>
        </p>
      </div>

      {factCheckingResult.claims.length > 0 ? (
        <div className="space-y-3">
          {factCheckingResult.claims.map((claim, index) => (
            <div key={`${claim.claim}-${index}`} className="max-w-full overflow-x-hidden rounded-lg border border-border bg-background/40 p-4">
              <div className="mb-3 flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0">
                  <h3 className="whitespace-normal break-words text-sm font-semibold text-foreground [overflow-wrap:anywhere] [word-break:break-word]">{claim.claim}</h3>
                  <p className="mt-1 whitespace-normal break-words text-xs text-muted-foreground [overflow-wrap:anywhere] [word-break:break-word]">{claim.type.replaceAll("_", " ")}</p>
                </div>
                <span className="w-fit rounded-md border border-border px-2 py-1 text-xs font-semibold text-foreground">
                  {claim.verdict.replaceAll("_", " ")}
                </span>
              </div>

              <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                <p>
                  Claim score: <span className="font-semibold text-foreground">{formatFactCheckingScore(claim.claim_score)}</span>
                </p>
                <p>
                  Confidence: <span className="font-semibold text-foreground">{formatFactCheckingScore(claim.confidence_score)}</span>
                </p>
              </div>

              <p className="mt-3 whitespace-normal break-words text-sm leading-7 text-muted-foreground [overflow-wrap:anywhere] [word-break:break-word]">{claim.explanation}</p>

              {claim.sources.length > 0 && (
                <div className="mt-4 max-w-full space-y-2 overflow-x-hidden">
                  <h4 className="text-sm font-semibold text-foreground">Sources</h4>
                  {claim.sources.map((source, sourceIndex) => (
                    <div key={`${source.url}-${sourceIndex}`} className="max-w-full overflow-x-hidden rounded-md border border-border bg-background/40 p-3">
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noreferrer"
                        className="block max-w-full whitespace-normal break-words text-sm font-semibold text-foreground [overflow-wrap:anywhere] [word-break:break-word] hover:text-primary"
                      >
                        {source.title}
                      </a>
                      <p className="mt-1 max-w-full whitespace-normal break-words text-xs text-muted-foreground [overflow-wrap:anywhere] [word-break:break-word]">
                        Credibility: {source.credibility_score.toFixed(2)}
                      </p>
                      <p className="mt-2 max-w-full whitespace-normal break-words text-sm leading-6 text-muted-foreground [overflow-wrap:anywhere] [word-break:break-word]">{source.snippet}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-border bg-background/40 p-4">
          <p className="text-sm text-muted-foreground">No factual claims were found for fact-checking.</p>
        </div>
      )}
    </div>
  ) : null;

  const masterAgentModalContent = masterResult ? (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h2 className="mb-2 text-base font-semibold text-foreground">{masterResult.title}</h2>
        <p className="text-sm text-muted-foreground">
          Result: <span className="font-semibold text-foreground">{masterVerdictLabel}</span>
        </p>
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">Scores used by the Master Agent</h3>
        <div className="space-y-3 text-sm text-muted-foreground">
          {masterResult.details.scores.map((scoreDetail) => (
            <div key={scoreDetail.agent} className="rounded-md border border-border bg-background/40 p-3">
              <p className="font-semibold text-foreground">{scoreDetail.label}</p>
              <p className="mt-1">
                Score: <span className="font-semibold text-foreground">
                  {scoreDetail.score === null ? "Missing" : `${clampPercentage(scoreDetail.score)}%`}
                </span>
              </p>
              {scoreDetail.agent === "fact_checking" && scoreDetail.original_score !== null && (
                <p className="mt-1">
                  Factual trust conversion: <span className="font-semibold text-foreground">
                    100 - {clampPercentage(scoreDetail.original_score)} = {scoreDetail.score === null ? "Missing" : clampPercentage(scoreDetail.score)}
                  </span>
                </p>
              )}
              <p className="mt-1">{scoreDetail.explanation}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">Calculation</h3>
        <div className="space-y-2 text-sm leading-7 text-muted-foreground">
          <p>Fact-check conversion: {masterResult.details.fact_check_conversion}</p>
          <p>Averaging formula: {masterResult.details.average_formula}</p>
          <p>
            Final rounded result: <span className="font-semibold text-foreground">
              {masterResult.details.final_rounded_result === null
                ? "Not available"
                : `${masterResult.details.final_rounded_result}% overall likely AI-written`}
            </span>
          </p>
        </div>
      </div>

      {masterResult.missing_agents.length > 0 && (
        <div className="rounded-lg border border-border bg-background/40 p-4">
          <h3 className="mb-3 text-sm font-semibold text-foreground">Missing agent results</h3>
          <p className="text-sm text-muted-foreground">
            {masterResult.missing_agents.join(", ")}
          </p>
        </div>
      )}
    </div>
  ) : null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="min-h-screen px-4 pb-12 pt-24"
    >
      <div className="container mx-auto max-w-5xl">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="mb-2 text-2xl font-bold">AI Text Checker</h1>
          <p className="mb-6 text-sm text-muted-foreground">
            Paste your text or submit a URL below and let our 4 agents analyze it.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-4 rounded-lg border border-border bg-card p-1"
        >
          <div className="px-4 pt-4">
            <Label htmlFor="checker-text" className="text-sm font-medium">
              Check text
            </Label>
          </div>
          <div className="relative">
            {textVerificationMutation.data && (
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 overflow-hidden whitespace-pre-wrap break-words p-4 text-sm leading-normal text-foreground"
              >
                {highlightedText}
              </div>
            )}
            <textarea
              id="checker-text"
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
              placeholder="Paste or type your text here..."
              className={`min-h-[200px] w-full resize-none rounded-md bg-transparent p-4 text-sm placeholder:text-muted-foreground focus:outline-none ${
                textVerificationMutation.data ? "relative z-10 text-transparent caret-foreground" : "text-foreground"
              }`}
            />
          </div>
          <div className="flex items-center justify-between border-t border-border px-4 py-2">
            <span className="font-mono text-xs text-muted-foreground">
              {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
            <button
              type="button"
              onClick={handleTextSubmit}
              disabled={!text.trim() || textLoading}
              className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2 text-sm font-bold text-primary-foreground transition-all cyber-glow hover:cyber-glow-strong disabled:opacity-40"
            >
              {textLoading ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Verifying...
                </>
              ) : (
                <>
                  <Send size={14} />
                  Verify Text
                </>
              )}
            </button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mb-6 rounded-lg border border-border bg-card p-4"
        >
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
            <div className="flex-1 space-y-2">
              <Label htmlFor="checker-url" className="text-sm font-medium">
                Check a URL
              </Label>
              <Input
                id="checker-url"
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Paste a URL to analyze its extracted text"
                className="h-11 bg-background"
                aria-describedby="checker-url-help"
              />
              <p id="checker-url-help" className="text-xs text-muted-foreground">
                Submit a webpage URL so the agents can fetch and analyze its extracted text.
              </p>
            </div>
            <button
              onClick={handleUrlSubmit}
              disabled={!url.trim() || urlLoading}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-primary px-5 text-sm font-bold text-primary-foreground transition-all cyber-glow hover:cyber-glow-strong disabled:opacity-40 sm:min-w-[140px]"
            >
              {urlLoading ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Verifying...
                </>
              ) : (
                <>
                  <Send size={14} />
                  Verify URL
                </>
              )}
            </button>
          </div>
          {urlStatus && <p className="mt-3 text-xs text-muted-foreground">{urlStatus}</p>}
        </motion.div>

        <div className="mb-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {agents.map((agent, i) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              index={i}
              isOpen={openAgent === agent.id}
              onToggle={() => setOpenAgent(openAgent === agent.id ? null : agent.id)}
              modalContent={
                agent.id === "statistic"
                  ? statisticalAgentModalContent
                  : agent.id === "grammatical"
                    ? grammaticalAgentModalContent
                    : agent.id === "factcheck"
                      ? factCheckingAgentModalContent
                      : agent.id === "orchestrator"
                        ? masterAgentModalContent
                        : undefined
              }
            />
          ))}
        </div>

        <AnimatePresence mode="wait">
          {(textValidationError || textVerificationMutation.isError) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive"
            >
              {textValidationError ?? textVerificationMutation.error.message}
            </motion.div>
          )}

          {textVerificationMutation.data && scorePercentage !== null && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <div className="rounded-lg border border-border bg-card p-5">
                <h2 className="mb-2 text-lg font-semibold">{textVerificationMutation.data.verification_title}</h2>
                <p className="text-sm text-muted-foreground">
                  Result: <span className="font-semibold text-foreground">{visibleVerdictLabel}</span>
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Open the <span className="font-semibold text-foreground">Statistic Agent</span> card above to view the full analysis details.
                </p>
              </div>

              {grammaticalResult && (
                <div className="rounded-lg border border-border bg-card p-5">
                  <h2 className="mb-2 text-lg font-semibold">Grammatical Verification Result</h2>
                  <p className="text-sm text-muted-foreground">
                    Result: <span className="font-semibold text-foreground">{grammaticalVerdictLabel}</span>
                  </p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Open the <span className="font-semibold text-foreground">Grammatical Agent</span> card above to view the full analysis details.
                  </p>
                </div>
              )}

              {factCheckingResult && (
                <div className="rounded-lg border border-border bg-card p-5">
                  <h2 className="mb-2 text-lg font-semibold">Fact-Checking Verification Result</h2>
                  <p className="text-sm text-muted-foreground">
                    Result: <span className="font-semibold text-foreground">{formatFactCheckingVerdictLabel(factCheckingResult.overall_trust_score)}</span>
                  </p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Lower factual trust can increase AI suspicion.
                  </p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Open the <span className="font-semibold text-foreground">Fact-Checking Agent</span> card above to view checked claims and sources.
                  </p>
                </div>
              )}

              {masterResult && (
                <div className="rounded-lg border border-border bg-card p-5">
                  <h2 className="mb-2 text-lg font-semibold">{masterResult.title}</h2>
                  <p className="text-sm text-muted-foreground">
                    Result: <span className="font-semibold text-foreground">{masterVerdictLabel}</span>
                  </p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Open the <span className="font-semibold text-foreground">Master Agent</span> card above to view how the result was calculated.
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default CheckerPage;
