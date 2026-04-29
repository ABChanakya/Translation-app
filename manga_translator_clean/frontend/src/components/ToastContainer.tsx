import React from "react";
import { motion, AnimatePresence } from "motion/react";
import { useStore } from "../store/useStore";
import type { ToastType } from "../store/useStore";
import { CheckCircle2, AlertCircle, Info, X } from "lucide-react";
import { cn } from "../lib/utils";

const icons: Record<ToastType, React.ElementType> = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
};

const colors: Record<ToastType, string> = {
  success: "border-emerald-500 text-emerald-400",
  error: "border-error text-error",
  info: "border-primary text-primary",
};

export const ToastContainer: React.FC = () => {
  const { toasts, removeToast } = useStore();

  return (
    <div className="fixed bottom-8 right-8 z-[100] flex flex-col gap-3 pointer-events-none">
      <AnimatePresence>
        {toasts.map((toast) => {
          const Icon = icons[toast.type];
          return (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, x: 50, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 20, scale: 0.9 }}
              className={cn(
                "pointer-events-auto w-80 bg-surface-container-highest border-l-4 p-4 shadow-2xl flex gap-3 items-start",
                colors[toast.type]
              )}
            >
              <Icon size={18} className="shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="text-xs font-black uppercase tracking-widest text-on-surface">{toast.message}</h4>
                {toast.description && (
                  <p className="text-[10px] text-on-surface-variant mt-1 leading-tight">{toast.description}</p>
                )}
              </div>
              <button
                onClick={() => removeToast(toast.id)}
                className="text-on-surface-variant/40 hover:text-on-surface transition-colors"
              >
                <X size={14} />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
};
