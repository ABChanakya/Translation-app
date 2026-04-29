import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "motion/react";
import Home from "./pages/Home";
import Sidebar from "./components/Sidebar";
import TopNav from "./components/TopNav";
import { ToastContainer } from "./components/ToastContainer";
import { ConfirmationModal } from "./components/ConfirmationModal";

const ProjectView = lazy(() => import("./pages/ProjectView"));
const Review = lazy(() => import("./pages/Review"));
const Export = lazy(() => import("./pages/Export"));

function Loading() {
  return (
    <div className="flex-1 pt-20 p-8 text-on-surface-variant animate-pulse">
      Loading...
    </div>
  );
}

function PageWrapper({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="flex-1 flex flex-col min-h-screen"
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}

function Layout() {
  const location = useLocation();
  const isReview = location.pathname.startsWith("/review/");

  return (
    <div className="min-h-screen bg-surface-container-lowest text-on-surface selection:bg-primary/30 selection:text-primary flex">
      <Sidebar />
      <div className="flex-1 flex flex-col min-h-screen ml-64">
        {!isReview && <TopNav />}
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<PageWrapper><Home /></PageWrapper>} />
            <Route path="/projects/:series" element={<PageWrapper><ProjectView /></PageWrapper>} />
            <Route path="/review/:chapterId" element={<Review />} />
            <Route path="/export/:chapterId" element={<PageWrapper><Export /></PageWrapper>} />
          </Routes>
        </Suspense>
      </div>
      <ToastContainer />
      <ConfirmationModal />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  );
}
