"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import { CornerDownLeft, Layers, Scale, Sparkles } from "lucide-react";

import AnswerPanel from "@/components/AnswerPanel";
import AskBar from "@/components/AskBar";
import DataRoomPanel from "@/components/DataRoomPanel";
import ErrorPanel from "@/components/ErrorPanel";
import Masthead from "@/components/Masthead";
import NotFoundPanel from "@/components/NotFoundPanel";
import PipelineTimeline from "@/components/PipelineTimeline";
import RefusalPanel from "@/components/RefusalPanel";
import SourcesRail from "@/components/SourcesRail";
import TieOutPanel from "@/components/TieOutPanel";
import UnavailablePanel from "@/components/UnavailablePanel";
import UploadPanel from "@/components/UploadPanel";
import { BRAND } from "@/lib/brand";
import {
  API_BASE,
  getBudget,
  listDeals,
  listDocuments,
  listRiskSignals,
  streamQuery,
} from "@/lib/api";
import { parseAnswer } from "@/lib/citations";
import { ApiError, toApiError } from "@/lib/errors";
import { applyStage, initialStages, settleStages } from "@/lib/pipeline";
import { useSandbox } from "@/lib/useSandbox";
import { answerVerdict, resultView } from "@/lib/verification";
import type {
  BudgetStatus,
  Deal,
  DocumentRecord,
  PipelineStage,
  QueryResponse,
  RiskSignal,
} from "@/lib/types";

type Phase = "idle" | "running" | "done";

interface AskedQuery {
  query: string;
  includePii: boolean;
}

export default function Home() {
  const [deals, setDeals] = useState<Deal[]>([]);
  const [activeDealId, setActiveDealId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [risks, setRisks] = useState<RiskSignal[]>([]);
  const [budget, setBudget] = useState<BudgetStatus>({});

  const [phase, setPhase] = useState<Phase>("idle");
  const [question, setQuestion] = useState("");
  const [stages, setStages] = useState<PipelineStage[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  // Why the deal list could not be loaded; without a deal nothing can be asked,
  // and an unreachable engine must not look like an empty data room.
  const [dealsError, setDealsError] = useState<ApiError | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [highlighted, setHighlighted] = useState<number | null>(null);
  // Answer text streamed while synthesis runs — an unverified draft, dropped
  // when the validated `result` arrives or the server sends `answer_reset`.
  const [draft, setDraft] = useState("");
  // The landing page's figure tie-out, opened on request: it reads the whole deal.
  const [showTieOut, setShowTieOut] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const lastAskedRef = useRef<AskedQuery | null>(null);

  const refreshDeals = useCallback(async (preferDealId?: string) => {
    let fetched: Deal[];
    try {
      fetched = await listDeals();
      setDealsError(null);
    } catch (e) {
      setDealsError(toApiError(e, API_BASE));
      // A sandbox the visitor just created is still usable without the list.
      if (preferDealId) setActiveDealId(preferDealId);
      return;
    }
    setDeals(fetched);
    setActiveDealId((current) => {
      if (preferDealId) return preferDealId;
      if (current && fetched.some((d) => d.deal_id === current)) return current;
      // Default to whichever deal has the most indexed documents — on a public
      // demo that is the pre-loaded data room, which is what a first-time
      // visitor should land on.
      const best = [...fetched].sort(
        (a, b) => b.document_count - a.document_count,
      )[0];
      return best?.deal_id ?? null;
    });
  }, []);

  const sandbox = useSandbox(
    useCallback(
      (deal: Deal) => {
        void refreshDeals(deal.deal_id);
      },
      [refreshDeals],
    ),
    useCallback(() => {
      // Clear the selection before refetching so `refreshDeals` falls back to
      // the pre-indexed demo deal rather than keeping the purged id.
      setActiveDealId(null);
      void refreshDeals();
    }, [refreshDeals]),
  );

  // A failed budget read keeps the last known figures rather than blanking them.
  const refreshBudget = useCallback(() => {
    void getBudget().then((b) => {
      if (b) setBudget(b);
    });
  }, []);

  useEffect(() => {
    void refreshDeals();
    refreshBudget();
  }, [refreshDeals, refreshBudget]);

  // Deal-scoped panels. Refetched whenever the deal changes, and after an
  // upload lands (the sandbox hook bumps `deals`, which changes the id).
  useEffect(() => {
    if (!activeDealId) {
      setDocuments([]);
      setRisks([]);
      return;
    }
    // Responses for a deal the visitor has already switched away from are
    // dropped, so a slow request cannot paint another deal's documents.
    let current = true;
    void listDocuments(activeDealId).then((d) => current && setDocuments(d ?? []));
    void listRiskSignals(activeDealId).then((r) => current && setRisks(r ?? []));
    return () => {
      current = false;
    };
  }, [activeDealId, sandbox.uploading]);

  // Wall clock for the running query. Driven by an interval rather than the
  // stage events, so the counter keeps moving during a long retrieval pass.
  useEffect(() => {
    if (phase !== "running") return;
    const started = performance.now();
    const id = setInterval(() => setElapsed(performance.now() - started), 100);
    return () => clearInterval(id);
  }, [phase]);

  const handleSubmit = useCallback(
    async (query: string, includePii: boolean) => {
      if (!activeDealId) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      lastAskedRef.current = { query, includePii };

      setQuestion(query);
      setPhase("running");
      setResult(null);
      setError(null);
      setStages([]);
      setElapsed(0);
      setHighlighted(null);
      setDraft("");

      try {
        await streamQuery({
          query,
          dealId: activeDealId,
          includePii,
          signal: controller.signal,
          onEvent: (evt) => {
            if (evt.event === "start") {
              setStages(initialStages(evt.data.planned_stages));
            } else if (evt.event === "stage") {
              setStages((prev) => applyStage(prev, evt.data));
            } else if (evt.event === "token") {
              setDraft((prev) => prev + evt.data.text);
            } else if (evt.event === "answer_reset") {
              setDraft("");
            } else if (evt.event === "result") {
              setDraft("");
              setResult(evt.data);
              setStages((prev) => settleStages(prev, "result"));
              setPhase("done");
            }
          },
        });
      } catch (e) {
        // An abort is the user pressing stop (or asking something else), not
        // a failure worth reporting; whoever aborted owns the screen now.
        if (controller.signal.aborted) return;
        // The draft never reached verification. Leaving it on screen under an
        // error would let an unchecked answer pass for the answer, so it goes.
        setDraft("");
        setError(toApiError(e, API_BASE));
        setStages((prev) => settleStages(prev, "failed"));
        setPhase("done");
      } finally {
        if (!controller.signal.aborted) refreshBudget();
      }
    },
    [activeDealId, refreshBudget],
  );

  const handleRetry = useCallback(() => {
    const last = lastAskedRef.current;
    if (last) void handleSubmit(last.query, last.includePii);
  }, [handleSubmit]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setPhase("idle");
    setStages([]);
    setDraft("");
  }, []);

  const handleSelectCitation = useCallback((index: number) => {
    setHighlighted(index);
    document
      .getElementById(`cite-${index}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
    // Clear the flash so re-clicking the same pill highlights it again.
    setTimeout(() => setHighlighted(null), 1600);
  }, []);

  // Memoised: `parseAnswer` walks the whole answer body, and this component
  // re-renders on every tick of the elapsed-time interval.
  const parsedCitations = useMemo(
    () => (result ? parseAnswer(result.answer, result.citations).citations : []),
    [result],
  );

  const isLanding = phase === "idle" && !result;
  const view = result ? resultView(result) : null;
  // The public deal list no longer includes sandboxes, so this tab's own
  // sandbox has to be matched separately or its landing hint never shows.
  const activeDeal =
    deals.find((d) => d.deal_id === activeDealId) ??
    (sandbox.sandboxDeal?.deal_id === activeDealId ? sandbox.sandboxDeal : null);

  // One polite sentence for assistive technology: which step is running, then
  // how the run ended. Errors announce themselves (role="alert"); the token
  // draft is deliberately not announced — it is unverified and would be read
  // out a fragment at a time.
  const runningStage = stages.find((s) => s.status === "running");
  const announcement =
    phase === "running"
      ? runningStage
        ? `Working: ${runningStage.label}.`
        : "Working."
      : result
        ? view === "answer"
          ? `Answer ready. ${answerVerdict(result).label}.`
          : view === "refusal"
            ? "The engine declined: the data room does not support an answer."
            : view === "not_found"
              ? "Not found: the documents searched do not contain the answer."
            : "No answer: the language models did not respond."
        : "";

  return (
    <MotionConfig reducedMotion="user">
    <div className="min-h-screen">
      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>
      <Masthead
        deals={deals}
        activeDealId={activeDealId}
        onSelectDeal={setActiveDealId}
        budget={budget}
      />

      <main className="mx-auto max-w-[1400px] px-5 pb-24">
        <AnimatePresence mode="wait">
          {isLanding ? (
            /* ---------------- Landing ---------------- */
            <motion.div
              key="landing"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
              className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-3xl flex-col justify-center py-16"
            >
              <div className="mb-9 text-center">
                <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-ink-850/70 px-3 py-1.5">
                  <Sparkles size={11} className="text-gold-400" />
                  <span className="text-[0.7rem] font-medium tracking-wide text-ash-300">
                    8 agents · hybrid retrieval · grounded or refused
                  </span>
                </div>

                <h1 className="text-balance bg-gradient-to-br from-ash-50 via-ash-50 to-gold-400 bg-clip-text text-[2.6rem] font-semibold leading-[1.08] tracking-tight text-transparent sm:text-[3.1rem]">
                  Ask the data room
                </h1>

                <p className="mx-auto mt-4 max-w-[54ch] text-pretty text-[0.95rem] leading-relaxed text-ash-300">
                  {BRAND.promise}
                </p>
              </div>

              <AskBar
                variant="hero"
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                running={false}
                disabled={!activeDealId}
              />

              {dealsError && !activeDealId && (
                <div
                  role="alert"
                  className="mx-auto mt-6 flex max-w-xl flex-wrap items-center justify-center gap-x-4 gap-y-2 text-center text-[0.8rem] leading-relaxed text-ash-300"
                >
                  <span>{dealsError.message}</span>
                  <button
                    type="button"
                    onClick={() => void refreshDeals()}
                    className="font-medium text-gold-400 underline-offset-2 hover:underline"
                  >
                    Try again
                  </button>
                </div>
              )}

              {activeDeal && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.25 }}
                  className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[0.75rem] text-ash-500"
                >
                  <span className="inline-flex items-center gap-1.5">
                    <Layers size={11} />
                    {activeDeal.document_count} document
                    {activeDeal.document_count === 1 ? "" : "s"} indexed
                  </span>
                  {risks.length > 0 && (
                    <span>
                      {risks.length} risk signal{risks.length === 1 ? "" : "s"}{" "}
                      detected
                    </span>
                  )}
                  <span className="inline-flex items-center gap-1.5">
                    <CornerDownLeft size={11} />
                    Enter to ask
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowTieOut((v) => !v)}
                    aria-expanded={showTieOut}
                    aria-controls="landing-tieout"
                    className="inline-flex items-center gap-1.5 font-medium text-gold-400 underline-offset-2 hover:underline"
                  >
                    <Scale size={11} />
                    {showTieOut ? "Hide figure tie-out" : "Tie out the figures"}
                  </button>
                </motion.div>
              )}

              {showTieOut && activeDealId && (
                // The deal-level tie-out (Decision 92), on request only: it
                // reads the whole data room.
                <div id="landing-tieout" className="panel mt-6 overflow-hidden text-left">
                  <TieOutPanel dealId={activeDealId} />
                </div>
              )}
            </motion.div>
          ) : (
            /* ---------------- Result ---------------- */
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="pt-8"
            >
              <h1 className="mb-6 max-w-[46ch] text-balance text-[1.55rem] font-semibold leading-snug tracking-tight text-ash-50">
                {question}
              </h1>

              <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
                <div className="min-w-0 space-y-5">
                  {error && <ErrorPanel error={error} onRetry={handleRetry} />}

                  {/* The draft exists only while the run is live. Every way a
                      run ends — result, error, cancel — leaves this phase. */}
                  {!result && draft && phase === "running" && (
                    <AnswerPanel result={null} draft={draft} />
                  )}

                  {/* Chosen by `outcome`. `is_refusal` is also true when no
                      model answered, which is an outage, not a refusal. */}
                  {view === "refusal" && result && <RefusalPanel result={result} />}
                  {view === "unavailable" && result && (
                    <UnavailablePanel result={result} onRetry={handleRetry} />
                  )}
                  {/* Not an answer, but the model's explanation and what it
                      did find are worth reading: the banner, then the text. */}
                  {view === "not_found" && result && <NotFoundPanel result={result} />}
                  {(view === "answer" || view === "not_found") && result && (
                    <AnswerPanel result={result} onSelectCitation={handleSelectCitation} />
                  )}

                  {/* While running, the timeline is the main event and belongs
                      in the primary column. Once the answer lands it moves to
                      the rail, where it becomes supporting evidence. */}
                  {phase === "running" && (
                    <PipelineTimeline
                      stages={stages}
                      running
                      elapsedMs={elapsed}
                    />
                  )}
                </div>

                <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
                  {(view === "answer" || view === "not_found") && (
                    <SourcesRail
                      citations={parsedCitations}
                      highlighted={highlighted}
                    />
                  )}

                  {phase === "done" && stages.length > 0 && (
                    <PipelineTimeline
                      stages={stages}
                      running={false}
                      elapsedMs={result?.total_latency_ms ?? elapsed}
                    />
                  )}

                  <DataRoomPanel documents={documents} risks={risks} dealId={activeDealId} />
                </aside>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Compact ask bar, docked once the conversation has started. */}
      {!isLanding && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-x-0 bottom-0 z-20 border-t border-line/70 bg-ink-950/78 backdrop-blur-xl"
        >
          <div className="mx-auto max-w-[1400px] px-5 py-3">
            <div className="lg:max-w-[calc(100%-21.25rem)]">
              <AskBar
                variant="compact"
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                running={phase === "running"}
                disabled={!activeDealId}
              />
            </div>
          </div>
        </motion.div>
      )}

      {/* Upload lives on the landing state only — it is an invitation, not a
          control the reader needs while reading an answer. */}
      {isLanding && (
        <div className="pointer-events-none fixed bottom-5 right-5 z-20 hidden w-72 xl:block">
          <div className="pointer-events-auto">
            <UploadPanel
              sandboxDeal={sandbox.sandboxDeal}
              uploading={sandbox.uploading}
              error={sandbox.error}
              lastUpload={sandbox.lastUpload}
              onUpload={sandbox.upload}
              onDiscard={sandbox.discard}
            />
          </div>
        </div>
      )}
    </div>
    </MotionConfig>
  );
}
