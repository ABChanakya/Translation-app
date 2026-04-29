import { Search, Bell, Command, Sparkles } from "lucide-react";

export default function TopNav() {
  return (
    <header className="fixed top-0 right-0 left-64 z-40 flex justify-between items-center px-6 bg-surface/80 backdrop-blur-xl h-14 border-b border-outline-variant/10 font-sans">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 gradient-cta flex items-center justify-center rounded-sm shadow-lg">
            <Sparkles size={18} className="text-on-primary" strokeWidth={2.5} />
          </div>
          <span className="text-xl font-black tracking-tighter text-on-surface">
            MangaReview <span className="text-primary">Pro</span>
          </span>
        </div>
        <div className="hidden md:flex items-center gap-6">
          <span className="text-[10px] font-black uppercase tracking-widest text-primary border-b-2 border-primary pb-1 cursor-pointer">
            Workflow
          </span>
          <span className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant/40 hover:text-primary transition-colors cursor-pointer">
            History
          </span>
          <span className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant/40 hover:text-primary transition-colors cursor-pointer">
            Assets
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="relative group">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant/40 group-focus-within:text-primary transition-colors" size={14} />
          <input
            className="bg-surface-container-low border border-outline-variant/10 text-[11px] rounded-sm pl-10 pr-4 py-1.5 w-72 focus:ring-1 focus:ring-primary/30 focus:border-primary/30 text-on-surface placeholder:text-on-surface-variant/20 transition-all outline-none font-medium"
            placeholder="Search projects, chapters, or bubbles..."
            type="text"
          />
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-20">
            <Command size={10} />
            <span className="text-[9px] font-bold">K</span>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button aria-label="Notifications" className="p-2 hover:bg-surface-container transition-colors rounded-sm text-on-surface-variant/60 hover:text-primary relative">
            <Bell size={18} />
            <span className="absolute top-2 right-2 w-1.5 h-1.5 bg-primary rounded-full border border-surface" />
          </button>
          <button aria-label="Shortcuts" className="p-2 hover:bg-surface-container transition-colors rounded-sm text-on-surface-variant/60 hover:text-primary">
            <Command size={18} />
          </button>
        </div>

        <div className="w-px h-6 bg-outline-variant/20" />

        <button className="gradient-cta text-on-primary px-5 py-1.5 rounded-sm font-black text-[10px] uppercase tracking-widest shadow-lg hover:scale-105 active:scale-95 transition-all">
          Commit Changes
        </button>

        <div className="flex items-center gap-3 pl-2">
          <div className="flex flex-col items-end">
            <span className="text-[10px] font-black text-on-surface leading-none">Chanakya B.</span>
            <span className="text-[8px] font-bold text-primary uppercase tracking-widest">Lead Editor</span>
          </div>
          <div className="w-9 h-9 rounded-sm bg-surface-container-highest overflow-hidden border border-outline-variant/20 shadow-inner flex items-center justify-center text-on-surface-variant">
            <span className="text-xs font-black">CB</span>
          </div>
        </div>
      </div>
    </header>
  );
}
