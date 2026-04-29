import { Link, useLocation } from "react-router-dom";
import { Home, FolderOpen, Edit3, Share2, Settings, HelpCircle, User } from "lucide-react";

const navItems = [
  { path: "/", label: "Home", icon: Home },
  { path: "/projects", label: "Projects", icon: FolderOpen },
  { path: "/review", label: "Review", icon: Edit3 },
  { path: "/export", label: "Export", icon: Share2 },
] as const;

export default function Sidebar() {
  const location = useLocation();

  const isActive = (path: string) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  return (
    <aside className="fixed left-0 top-0 h-full flex flex-col py-6 bg-surface w-64 border-r border-outline-variant/10 z-50">
      <div className="px-6 mb-10 flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary-container flex items-center justify-center rounded-sm">
          <div className="w-3 h-3 bg-on-primary-container rounded-full" />
        </div>
        <div>
          <h1 className="text-lg font-black tracking-tighter text-on-surface">The Obsidian Lens</h1>
          <p className="text-[10px] uppercase tracking-widest text-on-surface-variant/60 font-bold">Precision Editorial</p>
        </div>
      </div>

      <nav className="flex-1 px-3 space-y-1">
        {navItems.map((item) => {
          const active = isActive(item.path);
          return (
            <Link
              key={item.path}
              to={item.path}
              aria-current={active ? "page" : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2 transition-all duration-150 rounded-sm active:scale-[0.98] ${
                active
                  ? "text-primary font-bold bg-surface-container"
                  : "text-on-surface-variant/60 hover:text-on-surface hover:bg-surface-container"
              }`}
            >
              <item.icon size={20} />
              <span className="text-sm">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto px-3 space-y-1 border-t border-outline-variant/10 pt-4">
        <button className="w-full flex items-center gap-3 px-3 py-2 text-on-surface-variant/60 hover:text-on-surface hover:bg-surface-container transition-colors duration-150 rounded-sm">
          <Settings size={20} />
          <span className="text-sm">Settings</span>
        </button>
        <button className="w-full flex items-center gap-3 px-3 py-2 text-on-surface-variant/60 hover:text-on-surface hover:bg-surface-container transition-colors duration-150 rounded-sm">
          <HelpCircle size={20} />
          <span className="text-sm">Support</span>
        </button>

        <div className="mt-6 px-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-surface-container-highest flex items-center justify-center overflow-hidden grayscale brightness-90 border border-outline-variant/20">
            <User size={16} />
          </div>
          <div className="overflow-hidden">
            <p className="text-xs font-bold truncate">Chanakya B.</p>
            <p className="text-[10px] text-on-surface-variant/50">Lead Editor</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
