import { Link, useLocation } from "react-router-dom";
import { Sun, Moon, Menu, X, Mail, LogOut } from "lucide-react";
import { useTheme } from "@/hooks/useTheme";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface NavbarProps {
  isLoggedIn: boolean;
  userEmail?: string;
  onLogout: () => void;
}

const Navbar = ({ isLoggedIn, userEmail, onLogout }: NavbarProps) => {
  const { isDark, toggle } = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const links = [
    { to: "/", label: "Home" },
    { to: "/checker", label: "Checker" },
    ...(isLoggedIn ? [{ to: "/dashboard", label: "History" }, { to: "/settings", label: "Settings" }] : []),
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-slate-200 bg-white backdrop-blur-xl dark:border-border dark:bg-background/80">
      <div className="container mx-auto grid h-16 grid-cols-[1fr_auto] items-center px-4 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
        <Link to="/" className="flex items-center gap-2 justify-self-start">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary">
            <span className="text-sm font-bold text-primary-foreground">✓</span>
          </div>
          <span className="text-lg font-bold tracking-tight text-slate-900 dark:text-foreground">
            Check<span className="text-yellow-600 dark:text-primary">Wise</span>
          </span>
        </Link>

        {/* Desktop links */}
        <div className="hidden items-center justify-center gap-1 md:flex">
          {links.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                isActive(l.to)
                  ? "bg-yellow-100 text-yellow-700 dark:bg-primary/10 dark:text-primary"
                  : "text-slate-600 hover:text-slate-900 dark:text-muted-foreground dark:hover:text-foreground"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </div>

        <div className="flex items-center gap-2 justify-self-end">
          <button
            onClick={toggle}
            className="flex h-9 w-9 items-center justify-center rounded-md border border-slate-200 text-slate-600 transition-colors hover:text-slate-900 dark:border-border dark:text-muted-foreground dark:hover:text-foreground"
          >
            {isDark ? <Sun size={16} /> : <Moon size={16} />}
          </button>

          {isLoggedIn ? (
            <>
              <div className="hidden items-center gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 lg:flex dark:border-border dark:bg-card dark:text-muted-foreground">
                <Mail size={14} className="text-yellow-600 dark:text-primary" />
                <span className="max-w-[220px] truncate">{userEmail ?? "Signed in"}</span>
              </div>
              <button
                onClick={onLogout}
                className="hidden items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 transition-colors hover:text-slate-900 md:flex dark:border-border dark:text-muted-foreground dark:hover:text-foreground"
              >
                <LogOut size={14} />
                Logout
              </button>
            </>
          ) : (
            <Link
              to="/login"
              className="hidden rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-all hover:cyber-glow-strong md:block"
            >
              Login
            </Link>
          )}

          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="flex h-9 w-9 items-center justify-center rounded-md border border-slate-200 text-slate-600 md:hidden dark:border-border dark:text-muted-foreground"
          >
            {mobileOpen ? <X size={16} /> : <Menu size={16} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-slate-200 bg-white md:hidden dark:border-border dark:bg-background"
          >
            <div className="flex flex-col gap-1 p-4">
              {links.map((l) => (
                <Link
                  key={l.to}
                  to={l.to}
                  onClick={() => setMobileOpen(false)}
                  className={`rounded-md px-3 py-2 text-sm font-medium ${
                    isActive(l.to)
                      ? "bg-yellow-100 text-yellow-700 dark:bg-primary/10 dark:text-primary"
                      : "text-slate-600 dark:text-muted-foreground"
                  }`}
                >
                  {l.label}
                </Link>
              ))}
              {isLoggedIn ? (
                <>
                  <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 dark:border-border dark:bg-card dark:text-muted-foreground">
                    {userEmail ?? "Signed in"}
                  </div>
                  <button onClick={() => { onLogout(); setMobileOpen(false); }} className="rounded-md px-3 py-2 text-left text-sm font-medium text-slate-600 dark:text-muted-foreground">
                    Logout
                  </button>
                </>
              ) : (
                <Link to="/login" onClick={() => setMobileOpen(false)} className="rounded-md bg-primary px-3 py-2 text-center text-sm font-semibold text-primary-foreground">
                  Login
                </Link>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

export default Navbar;
