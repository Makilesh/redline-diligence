import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import AnswerPanel from "@/components/AnswerPanel";
import { response } from "./fixtures";

/*
 * The answer is model output written from documents the counterparty
 * supplied, so prompt injection can put anything into it. None of it may
 * become markup, script or a navigable non-web link.
 */
const HOSTILE = [
  "Revenue was $452.8 million [📄 10-K.pdf | p.4 | Income Statement].",
  "",
  "<script>window.__pwned = 1</script>",
  "",
  '<img src=x onerror="window.__pwned = 2">',
  "",
  "[Open the data room](javascript:window.__pwned=3) and [the filing](https://www.sec.gov/) " +
    "and [data](data:text/html,<script>alert(1)</script>) and [mail](mailto:ir@example.com).",
  "",
  '<a href="javascript:window.__pwned=5">raw anchor</a>',
].join("\n");

describe("AnswerPanel rendering of untrusted markdown", () => {
  it("renders no raw HTML, script or event handler", () => {
    const { container } = render(<AnswerPanel result={response({ answer: HOSTILE })} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("[onerror]")).toBeNull();
    expect((window as unknown as { __pwned?: number }).__pwned).toBeUndefined();
  });

  it("links only http(s) and mailto, opening them in a new tab without referrer", () => {
    const { container } = render(<AnswerPanel result={response({ answer: HOSTILE })} />);
    const hrefs = [...container.querySelectorAll("a[href]")].map((a) => a.getAttribute("href"));
    // Citation pills link within the page; every other link is web or mail.
    for (const href of hrefs) expect(href).toMatch(/^(#cite-\d+|https:\/\/|mailto:)/);
    const external = screen.getByRole("link", { name: "the filing" });
    expect(external).toHaveAttribute("target", "_blank");
    expect(external.getAttribute("rel")).toContain("noopener");
    // The unsafe links survive as text, so nothing the model wrote is hidden.
    expect(screen.getByText("Open the data room").closest("a")).toBeNull();
  });

  it("renders the draft and the error-free answer as text nodes", () => {
    render(<AnswerPanel result={null} draft={"<b>bold?</b> Revenue"} />);
    expect(document.querySelector("b")).toBeNull();
  });
});
