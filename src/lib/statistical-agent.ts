export interface EligibilityAssessment {
  strong_verdict_allowed: boolean;
  reasons: string[];
}

export interface DocumentAssessment {
  ai_likelihood_score: number;
  ai_likelihood_label: "low" | "moderate" | "high";
  confidence: "low" | "medium" | "high";
}

export interface SignalBreakdown {
  semantic_model_score: number;
  stylometric_score: number;
  robustness_score: number;
}

export interface SegmentSpan {
  start: number;
  end: number;
  score: number;
}

export interface SegmentAssessment {
  available: boolean;
  spans: SegmentSpan[];
}

export interface Highlight {
  text: string;
  start: number;
  end: number;
  reason: string;
}

export interface TextAnalysisMetrics {
  sentence_count: number;
  sentence_lengths: number[];
  length_variation_score: number;
  repeated_linking_words: Record<string, number>;
  expressive_repetition_score: number;
  expressive_repeated_phrases: string[];
  linguistic_style_score: number;
  sentence_length_range: number;
  sentence_length_std_dev: number;
  consecutive_diff_over_10: boolean;
  linking_word_ai_score: number;
  linguistic_ai_score: number;
}

export interface DetectorDetails {
  status: string;
  score_semantics: string;
  raw_score: number | null;
  observations: string[];
  influential_phrases: string[];
  technical_note: string | null;
  schema_present_keys?: string[] | null;
  invoke_error_type?: string | null;
  invoke_error_message?: string | null;
  invoke_error_status_code?: number | null;
  invoke_error_body?: string | null;
  invoke_error_provider?: string | null;
  invoke_error_model?: string | null;
  invoke_error_base_url?: string | null;
  invoke_error_timeout_seconds?: number | null;
  raw_output_excerpt?: string | null;
  diagnostic_timestamp?: string | null;
}

export interface GrammaticalResult {
  score: number;
  confidence: "low" | "medium" | "high";
  reasons_for_rating: string[];
  lowered_confidence_reasons: string[];
}

export interface SourceEvidence {
  title: string;
  url: string;
  credibility_score: number;
  snippet: string;
}

export interface FactCheckedClaim {
  claim: string;
  type: string;
  queries: string[];
  verdict: "VERIFIED" | "LIKELY_TRUE" | "CONTRADICTED" | "OUTDATED" | "UNVERIFIABLE";
  claim_score: number;
  confidence_score: number;
  sources: SourceEvidence[];
  explanation: string;
}

export interface FactCheckingResult {
  agent: "fact_checking";
  overall_trust_score: number;
  overall_confidence_score: number;
  total_claims: number;
  claims: FactCheckedClaim[];
}

export interface MasterScoreDetail {
  agent: string;
  label: string;
  score: number | null;
  original_score: number | null;
  used: boolean;
  source_field: string;
  explanation: string;
}

export interface MasterResult {
  agent: "master";
  title: string;
  model: string;
  score: number | null;
  raw_score: number | null;
  label: string;
  available: boolean;
  used_agents: string[];
  missing_agents: string[];
  formula: string;
  details: {
    scores: MasterScoreDetail[];
    fact_check_conversion: string;
    average_formula: string;
    final_rounded_result: number | null;
    missing_agents: string[];
    note: string;
  };
}

export interface TextVerificationResponse {
  title: string;
  verification_title: string;
  language: string;
  eligibility: EligibilityAssessment;
  document_assessment: DocumentAssessment;
  signal_breakdown: SignalBreakdown;
  why: string[];
  what_weakens_the_conclusion: string[];
  segment_assessment: SegmentAssessment;
  final_user_message: string;
  verdict: string;
  percentage: number;
  final_label: string;
  bullet_points: string[];
  summary: string[];
  limitations: string[];
  detector_details: DetectorDetails;
  highlights: Highlight[];
  metrics: TextAnalysisMetrics;
  grammatical_result: GrammaticalResult;
  fact_checking_result?: FactCheckingResult | null;
  master_result?: MasterResult | null;
}

export interface HistoryEntry {
  id: number;
  user_email: string;
  input_type: string;
  submitted_text: string;
  text_preview: string;
  verification_rating: string;
  statistical_percentage: number;
  confidence: string;
  structured_result: TextVerificationResponse;
  created_at: string;
}

const normalizeApiBaseUrl = (value: string) => value.replace(/\/+$/, "");

const getApiBaseUrl = () => {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return normalizeApiBaseUrl(configuredBaseUrl);
  }

  return "";
};

const buildApiUrl = (path: string) => {
  const baseUrl = getApiBaseUrl();
  return baseUrl ? `${baseUrl}${path}` : path;
};

const parseApiErrorMessage = (errorPayload: unknown, fallbackMessage: string) => {
  if (
    errorPayload &&
    typeof errorPayload === "object" &&
    "detail" in errorPayload
  ) {
    const detail = (errorPayload as { detail?: unknown }).detail;

    if (typeof detail === "string" && detail) {
      return detail;
    }

    if (Array.isArray(detail)) {
      const firstIssue = detail.find(
        (item): item is { msg?: string; loc?: Array<string | number> } =>
          Boolean(item) && typeof item === "object",
      );

      if (firstIssue?.msg) {
        const location = firstIssue.loc?.join(" -> ");
        return location ? `${location}: ${firstIssue.msg}` : firstIssue.msg;
      }
    }
  }

  return fallbackMessage;
};

export const verifyText = async (payload: {
  userEmail: string;
  text: string;
}): Promise<TextVerificationResponse> => {
  console.log("verifyText: preparing payload", {
    userEmail: payload.userEmail,
    textLength: payload.text.length,
  });

  const requestUrl = buildApiUrl("/api/text/verify");
  const requestBody = {
    user_email: payload.userEmail,
    text: payload.text,
    input_type: "text",
  };

  console.log("verifyText: sending fetch", {
    url: requestUrl,
    payload: {
      ...requestBody,
      text: `[length=${payload.text.length}]`,
    },
  });

  try {
    const response = await fetch(requestUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    console.log("verifyText: fetch completed", {
      status: response.status,
      ok: response.ok,
      url: requestUrl,
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => null);
      const message = parseApiErrorMessage(errorPayload, "The text verification request failed.");
      throw new Error(message);
    }

    return (await response.json()) as TextVerificationResponse;
  } catch (error) {
    console.error("verifyText: request failed before completion", error);
    throw error;
  } finally {
    console.log("verifyText: request cleanup finished");
  }
};

export const fetchHistory = async (userEmail: string): Promise<HistoryEntry[]> => {
  const response = await fetch(buildApiUrl(`/api/history?user_email=${encodeURIComponent(userEmail)}`));

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => null);
    const message = parseApiErrorMessage(errorPayload, "The history request failed.");
    throw new Error(message);
  }

  return response.json();
};
