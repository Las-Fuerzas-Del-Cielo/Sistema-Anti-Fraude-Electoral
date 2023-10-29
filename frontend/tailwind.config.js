const defaultTheme = require('tailwindcss/defaultTheme');

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {

      },
      fontFamily: {
        'sans': ['Poppins', ...defaultTheme.fontFamily.sans],
      }
    },
  },
  plugins: [],
}

