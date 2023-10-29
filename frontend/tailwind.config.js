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
        'violet': '#61439D',
        'light-violet': '#8B6ACE',
        'gray': '#64748B',
        'gray-inactive': '#CACACB',
        'light-gray': '#CBD5E1',
        'red': '#AD3459',
        'black': '#000000',
        'white': '#FFFFFF',
      },
      fontFamily: {
        'sans': ['Poppins', ...defaultTheme.fontFamily.sans],
      }
    },
  },
  plugins: [],
}

