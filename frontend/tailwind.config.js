// eslint-disable-next-line @typescript-eslint/no-var-requires, no-undef
const defaultTheme = require('tailwindcss/defaultTheme');

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'violet-brand': '#61439D',
        'violet-dark': '#3C2865',
        'violet-light': '#8B6ACE',
        'gray-dark': '#64748B',
        'gray-inactive': '#CACACB',
        'gray-light': '#CBD5E1',
        'text-off': '#363F45',
        red: '#AD3459',
        black: '#000000',
        white: '#FFFFFF',
      },
      fontFamily: {
        sans: ['Poppins', ...defaultTheme.fontFamily.sans],
      },
    },
  },
  plugins: [],
};
