"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { FileWarning, GitCompare, Globe, Loader2, ShieldCheck, Trash2, Upload } from "lucide-react";

import type { DataHandling, Deal, VersionCandidate } from "@/lib/types";
import type { LastUpload } from "@/lib/useSandbox";
import { uploadScreenNote } from "@/lib/verification";

interface UploadPanelProps {
  sandboxDeal: Deal | null;
  uploading: boolean;
  error: string | null;
  lastUpload?: LastUpload | null;
  /** Where uploaded text goes: undefined while loading, null if it could not be read. */
  dataHandling?: DataHandling | null;
  onUpload: (file: File) => void;
  onDiscard: () => void;
}

/** Shown when the server could not say: assume the text leaves (Decision 116). */
export const DATA_HANDLING_UNKNOWN =
  "Could not confirm whether this deployment sends document text to an outside AI " +
  "service. Assume it does: do not upload confidential, client or personal documents.";

/**
 * The notice before an upload (Decision 116): the server's wording when a hosted
 * model is configured, a cautious one when the server could not be asked, and
 * nothing while loading or when text stays on the server.
 */
function DataHandlingNotice({ dataHandling }: { dataHandling: DataHandling | null | undefined }) {
  if (dataHandling === undefined) return null;
  const text = dataHandling === null ? DATA_HANDLING_UNKNOWN : dataHandling.upload_notice;
  if (!text) return null;
  return (
    <div className="border-t border-warn/30 bg-warn/[0.05] px-4 py-3" role="note" aria-label="Where uploaded text goes">
      <p className="flex items-start gap-2 text-[0.7rem] leading-relaxed text-warn">
        <Globe size={12} className="mt-0.5 shrink-0" aria-hidden />
        <span>{text}</span>
      </p>
    </div>
  );
}

const ACCEPTED = ".pdf,.docx,.xlsx,.pptx,.txt";

const RELATION: Record<VersionCandidate["relation"], string> = {
  newer: "a newer version of",
  older: "an older version of",
  same_document: "the same document as",
};

/**
 * What the last upload produced. Version matches are reported as what they
 * are — the server's guess, with nothing superseded — so a reader does not
 * assume the older document has dropped out of the answers.
 */
function UploadOutcome({ upload }: { upload: LastUpload }) {
  const { filename, result } = upload;
  const partial = result.status === "partial";
  const screen = uploadScreenNote(result.injection_screen);

  return (
    <div className="space-y-2 border-t border-line px-4 py-3 text-left" role="status">
      <p className={`text-[0.74rem] leading-snug ${partial ? "text-warn" : "text-ok"}`}>
        {partial ? "Partly indexed: " : "Indexed: "}
        <span className="font-medium text-ash-50">{filename}</span> ·{" "}
        {result.chunks_created} passage{result.chunks_created === 1 ? "" : "s"}
      </p>

      {result.notices.length > 0 && (
        <ul className="space-y-1">
          {result.notices.map((n, i) => (
            <li key={i} className="text-[0.7rem] leading-snug text-ash-300">
              {n}
            </li>
          ))}
        </ul>
      )}

      {screen && (
        // The uploader is the one best placed to open a seller-supplied file
        // and look (Decision 82). A label, not a failure: the file is indexed.
        <div className="rounded-md border border-warn/30 bg-warn/[0.05] px-2.5 py-2">
          <p className="flex items-start gap-1.5 text-[0.7rem] leading-snug text-warn">
            <FileWarning size={11} className="mt-0.5 shrink-0" aria-hidden />
            <span>{screen.finding}</span>
          </p>
          <p className="mt-1.5 text-[0.66rem] leading-snug text-ash-500">{screen.consequence}</p>
        </div>
      )}

      {result.version_candidates.length > 0 && (
        <div className="rounded-md border border-line bg-ink-850/70 px-2.5 py-2">
          {result.version_candidates.map((c) => (
            <p
              key={`${c.doc_id}-${c.relation}`}
              className="flex items-start gap-1.5 text-[0.7rem] leading-snug text-ash-300"
            >
              <GitCompare size={11} className="mt-0.5 shrink-0 text-info" />
              <span>
                This looks like {RELATION[c.relation]}{" "}
                <span className="font-medium text-ash-50">{c.source_file || c.doc_id}</span>.
              </span>
            </p>
          ))}
          <p className="mt-1.5 text-[0.66rem] leading-snug text-ash-500">
            Nothing was replaced — both are searchable and either may be cited.
          </p>
        </div>
      )}
    </div>
  );
}

export default function UploadPanel({
  sandboxDeal,
  uploading,
  error,
  lastUpload,
  dataHandling,
  onUpload,
  onDiscard,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className="panel overflow-hidden">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`px-4 py-5 text-center transition-colors ${
          dragging ? "bg-gold-500/[0.07]" : ""
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          // Opened by the button below; as its own tab stop it was an
          // unlabelled control that did nothing visible when focused.
          aria-label="Choose a document to upload"
          tabIndex={-1}
          className="sr-only"
          onChange={(e) => {
            handleFiles(e.target.files);
            // Reset so re-uploading the same filename fires a change event.
            e.target.value = "";
          }}
        />

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="mx-auto flex flex-col items-center gap-2 disabled:opacity-60"
        >
          <span
            className={`flex h-9 w-9 items-center justify-center rounded-xl border transition-colors ${
              dragging
                ? "border-gold-500/60 bg-gold-500/12 text-gold-400"
                : "border-line bg-ink-800 text-ash-500"
            }`}
          >
            {uploading ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <Upload size={15} />
            )}
          </span>
          <span className="text-[0.8rem] font-medium text-ash-50">
            {uploading ? "Indexing…" : "Test it on your own document"}
          </span>
          <span className="max-w-[24ch] text-[0.7rem] leading-snug text-ash-500">
            PDF, DOCX, XLSX, PPTX or TXT. Drop it here or click to browse.
          </span>
        </button>

        {uploading && (
          <p className="mt-3 text-[0.7rem] leading-snug text-ash-500">
            Chunking, embedding and indexing runs on CPU — expect roughly a
            minute for a long document.
          </p>
        )}
      </div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-line bg-bad/[0.06]"
          >
            <p className="px-4 py-2.5 text-[0.75rem] leading-snug text-bad">
              {error}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {lastUpload && !uploading && <UploadOutcome upload={lastUpload} />}

      <DataHandlingNotice dataHandling={dataHandling} />

      {/* The privacy promise, stated where the decision to upload is made
          rather than buried in a footer. */}
      <div className="border-t border-line bg-ink-850/60 px-4 py-3">
        <div className="flex items-start gap-2">
          <ShieldCheck size={12} className="mt-0.5 shrink-0 text-ok" />
          <p className="text-[0.7rem] leading-relaxed text-ash-500">
            Uploads go to a temporary deal that is deleted when you close this
            tab, and swept from the server on a timer if that never arrives.
          </p>
        </div>

        {sandboxDeal && (
          <button
            type="button"
            onClick={onDiscard}
            className="mt-2.5 inline-flex items-center gap-1.5 text-[0.72rem] font-medium text-ash-500 transition-colors hover:text-bad"
          >
            <Trash2 size={11} />
            Delete my documents now
          </button>
        )}
      </div>
    </div>
  );
}
