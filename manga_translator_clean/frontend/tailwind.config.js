/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', "ui-sans-serif", "system-ui", "sans-serif"],
        manga: ['"Bangers"', "cursive"],
        hand: ['"Kalam"', "cursive"],
      },
      colors: {
        primary: {
          DEFAULT: "#adc6ff",
          container: "#4d8eff",
        },
        "on-primary": {
          DEFAULT: "#002e6a",
          container: "#00285d",
        },
        secondary: {
          DEFAULT: "#b1c6f9",
          container: "#304671",
        },
        "on-secondary": "#182f59",
        tertiary: {
          DEFAULT: "#ffb786",
          container: "#df7412",
        },
        "on-tertiary": "#502400",
        surface: {
          DEFAULT: "#131313",
          dim: "#131313",
          bright: "#3a3939",
          "container-lowest": "#0e0e0e",
          "container-low": "#1c1b1b",
          container: "#201f1f",
          "container-high": "#2a2a2a",
          "container-highest": "#353534",
        },
        "on-surface": {
          DEFAULT: "#e5e2e1",
          variant: "#c2c6d6",
        },
        outline: {
          DEFAULT: "#8c909f",
          variant: "#424754",
        },
        error: {
          DEFAULT: "#ffb4ab",
          container: "#93000a",
        },
        "on-error": "#690005",
      },
    },
  },
  plugins: [],
};
