
import { Moon, Sun } from "lucide-react"

import { useTheme } from "@/components/ThemeProvider"
import { Switch } from "@/components/ui/switch"

export function ThemeToggle() {
  const { setTheme, theme } = useTheme()

  const isDark = theme === "dark"

  const toggleTheme = (checked: boolean) => {
    setTheme(checked ? "dark" : "light")
  }

  return (
    <div className="flex items-center space-x-2">
      <Sun className={`h-[1.2rem] w-[1.2rem] transition-all ${isDark ? 'text-muted-foreground' : 'text-foreground'}`} />
      <Switch
        checked={isDark}
        onCheckedChange={toggleTheme}
        aria-label="Toggle between light and dark mode"
      />
      <Moon className={`h-[1.2rem] w-[1.2rem] transition-all ${isDark ? 'text-foreground' : 'text-muted-foreground'}`} />
    </div>
  )
}
