import { useEffect, useState } from "react";
import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AnimatePresence } from "framer-motion";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Navbar from "@/components/Navbar";
import LandingPage from "@/pages/LandingPage";
import CheckerPage from "@/pages/CheckerPage";
import DashboardPage from "@/pages/DashboardPage";
import AuthPage from "@/pages/AuthPage";
import SettingsPage from "@/pages/SettingsPage";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

export interface AuthUser {
  email: string;
  name?: string;
}

const AppRoutes = () => {
  const location = useLocation();
  const [user, setUser] = useState<AuthUser | null>(() => {
    if (typeof window === "undefined") return null;
    const storedUser = window.localStorage.getItem("checkwise-user");
    if (!storedUser) return null;

    try {
      return JSON.parse(storedUser) as AuthUser;
    } catch {
      return null;
    }
  });
  const isLoggedIn = Boolean(user);

  useEffect(() => {
    if (typeof window === "undefined") return;

    if (user) {
      window.localStorage.setItem("checkwise-user", JSON.stringify(user));
      return;
    }

    window.localStorage.removeItem("checkwise-user");
  }, [user]);

  return (
    <>
      <Navbar
        isLoggedIn={isLoggedIn}
        userEmail={user?.email}
        onLogout={() => setUser(null)}
      />
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/checker" element={<CheckerPage userEmail={user?.email} />} />
          <Route path="/dashboard" element={<DashboardPage userEmail={user?.email} />} />
          <Route
            path="/login"
            element={<AuthPage mode="login" onLogin={(nextUser) => setUser(nextUser)} />}
          />
          <Route
            path="/register"
            element={<AuthPage mode="register" onLogin={(nextUser) => setUser(nextUser)} />}
          />
          <Route path="/settings" element={<SettingsPage user={user} />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AnimatePresence>
    </>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
