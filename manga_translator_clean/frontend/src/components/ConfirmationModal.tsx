import React from "react";
import { motion, AnimatePresence } from "motion/react";
import { useStore } from "../store/useStore";
import { cn } from "../lib/utils";

export const ConfirmationModal: React.FC = () => {
  const { modal, hideModal } = useStore();

  if (!modal) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={hideModal}
          className="absolute inset-0 bg-black/80 backdrop-blur-sm"
        />
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="relative w-full max-w-md bg-surface-container border border-outline-variant/20 p-8 shadow-2xl"
        >
          <h2 className="text-2xl font-black tracking-tighter text-on-surface mb-2 uppercase">{modal.title}</h2>
          <p className="text-sm text-on-surface-variant mb-8 leading-relaxed">{modal.message}</p>

          <div className="flex gap-3 justify-end">
            <button
              onClick={() => {
                modal.onCancel?.();
                hideModal();
              }}
              className="px-6 py-2 text-xs font-bold uppercase tracking-widest text-on-surface-variant hover:bg-surface-container-high transition-colors"
            >
              {modal.cancelLabel || "CANCEL"}
            </button>
            <button
              onClick={() => {
                modal.onConfirm();
                hideModal();
              }}
              className={cn(
                "px-8 py-2 text-xs font-black uppercase tracking-widest text-on-primary transition-all active:scale-95",
                modal.variant === "danger" ? "bg-error" : "gradient-cta"
              )}
            >
              {modal.confirmLabel}
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
