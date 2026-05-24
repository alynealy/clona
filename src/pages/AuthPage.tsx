import { useState } from "react";
import { motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff } from "lucide-react";
import AnimatedBackground from "@/components/AnimatedBackground";
import type { AuthUser } from "@/App";

interface AuthPageProps {
  mode: "login" | "register";
  onLogin: (user: AuthUser) => void;
}

const AuthPage = ({ mode, onLogin }: AuthPageProps) => {
  const navigate = useNavigate();
  const [showPass, setShowPass] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  const isLogin = mode === "login";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onLogin({
      email: email.trim(),
      name: name.trim() || undefined,
    });
    navigate("/checker");
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative flex min-h-screen items-center justify-center overflow-hidden px-4 pt-16">
      <AnimatedBackground />
      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4 }}
        className="relative z-10 w-full max-w-sm rounded-lg border border-border bg-card/95 p-6 backdrop-blur-sm"
      >
        <div className="mb-6 text-center">
          <h1 className="text-xl font-bold">{isLogin ? "Welcome Back" : "Create Account"}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {isLogin ? "Sign in to access your dashboard" : "Join CheckWise to save your history"}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          {!isLogin && (
            <input
              type="text"
              placeholder="Full Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="rounded-md border border-border bg-background px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          )}
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="rounded-md border border-border bg-background px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
          <div className="relative">
            <input
              type={showPass ? "text" : "password"}
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2.5 pr-10 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <button type="button" onClick={() => setShowPass(!showPass)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground">
              {showPass ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
          <button
            type="submit"
            className="mt-2 rounded-md bg-primary py-2.5 text-sm font-bold text-primary-foreground transition-all cyber-glow hover:cyber-glow-strong"
          >
            {isLogin ? "Sign In" : "Create Account"}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-muted-foreground">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <Link to={isLogin ? "/register" : "/login"} className="font-medium text-primary hover:underline">
            {isLogin ? "Register" : "Sign In"}
          </Link>
        </p>
      </motion.div>
    </motion.div>
  );
};

export default AuthPage;
