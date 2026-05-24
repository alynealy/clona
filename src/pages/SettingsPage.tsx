import { motion } from "framer-motion";
import { useTheme } from "@/hooks/useTheme";
import { Sun, Moon, User } from "lucide-react";
import type { AuthUser } from "@/App";

interface SettingsPageProps {
  user: AuthUser | null;
}

const SettingsPage = ({ user }: SettingsPageProps) => {
  const { isDark, toggle } = useTheme();

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="min-h-screen px-4 pt-24 pb-12">
      <div className="container mx-auto max-w-2xl">
        <h1 className="mb-6 text-2xl font-bold">Settings</h1>

        {/* Profile */}
        <div className="mb-4 rounded-lg border border-border bg-card p-5">
          <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold">
            <User size={16} className="text-primary" /> Profile
          </h2>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">Name</label>
              <input
                defaultValue={user?.name ?? "Alex Student"}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">Email</label>
              <input
                defaultValue={user?.email ?? "alex@checkwise.io"}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
          </div>
        </div>

        {/* Theme */}
        <div className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-4 text-sm font-semibold">Appearance</h2>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">{isDark ? "Dark Mode" : "Light Mode"}</p>
              <p className="text-xs text-muted-foreground">Toggle between dark and light themes</p>
            </div>
            <button
              onClick={toggle}
              className="flex h-10 w-10 items-center justify-center rounded-md border border-border transition-colors hover:bg-muted"
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default SettingsPage;
