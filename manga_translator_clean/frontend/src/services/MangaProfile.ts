/**
 * Frontend MangaProfile — mirrors the backend src/translation/manga_profile.py.
 * Stores glossary, character names, and rolling page context in memory.
 * Used to build prompt injections for client-side display and future AI integration.
 */

export interface GlossaryEntry {
  term: string;
  translation: string;
  category?: string;
  notes?: string;
}

export interface PageContext {
  pageNumber: number;
  summary: string;
  detectedCharacters: string[];
}

export class MangaProfile {
  private glossary: Map<string, GlossaryEntry> = new Map();
  private memory: PageContext[] = [];
  private rollingContext = "";
  readonly seriesName: string;

  constructor(seriesName: string, initialGlossary: GlossaryEntry[] = []) {
    this.seriesName = seriesName;
    initialGlossary.forEach((e) => this.glossary.set(e.term, e));
  }

  addGlossaryEntry(entry: GlossaryEntry) {
    this.glossary.set(entry.term, entry);
  }

  getGlossary(): GlossaryEntry[] {
    return Array.from(this.glossary.values());
  }

  addPageContext(ctx: PageContext) {
    this.memory.push(ctx);
    const recent = this.memory.slice(-3);
    this.rollingContext = recent
      .map((p) => `Page ${p.pageNumber}: ${p.summary}`)
      .join("\n");
  }

  getRollingContext(): string {
    return this.rollingContext;
  }

  /** Returns a formatted block for LLM prompt injection. */
  getPromptInjection(): string {
    const lines: string[] = ["GLOSSARY:"];
    this.glossary.forEach((e) => {
      lines.push(
        `- ${e.term}: ${e.translation}${e.notes ? ` (${e.notes})` : ""}`
      );
    });
    if (this.rollingContext) {
      lines.push("\nRECENT CONTEXT:", this.rollingContext);
    }
    return lines.join("\n");
  }

  /** Human-readable summary for the inspector panel. */
  getContextSummary(): string {
    const terms = this.glossary.size;
    const pages = this.memory.length;
    if (terms === 0 && pages === 0) return "No profile data yet.";
    const parts: string[] = [];
    if (terms > 0) parts.push(`${terms} glossary term${terms !== 1 ? "s" : ""}`);
    if (pages > 0) parts.push(`${pages} page${pages !== 1 ? "s" : ""} of context`);
    return parts.join(" · ") + " loaded from series profile.";
  }
}
