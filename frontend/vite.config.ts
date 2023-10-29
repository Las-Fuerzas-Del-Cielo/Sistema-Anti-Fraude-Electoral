/// <reference types="Vite/client"/>

import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";
import autoAlias from "vite-plugin-auto-alias";
import dynamicImport from "vite-plugin-dynamic-import";
import { ViteImageOptimizer } from "vite-plugin-image-optimizer";
import { ViteMinifyPlugin } from "vite-plugin-minify";
import { viteObfuscateFile } from "vite-plugin-obfuscator";

const obfuscator_options = {
  compact: true,
  controlFlowFlattening: false,
  deadCodeInjection: false,
  debugProtection: false,
  debugProtectionInterval: 0,
  disableConsoleOutput: false,
  identifierNamesGenerator: "hexadecimal",
  log: false,
  numbersToExpressions: false,
  renameGlobals: false,
  selfDefending: false,
  simplify: true,
  splitStrings: false,
  stringArray: true,
  stringArrayCallsTransform: false,
  stringArrayCallsTransformThreshold: 0.5,
  stringArrayEncoding: [],
  stringArrayIndexShift: true,
  stringArrayRotate: true,
  stringArrayShuffle: true,
  stringArrayWrappersCount: 1,
  stringArrayWrappersChainedCalls: true,
  stringArrayWrappersParametersMaxCount: 2,
  stringArrayWrappersType: "variable",
  stringArrayThreshold: 0.01,
  unicodeEscapeSequence: false,
};

export default defineConfig({
  plugins: [
    react(),
    autoAlias({
      mode: "sync",
      prefix: "#",
    }),

    ViteImageOptimizer(),
    ViteMinifyPlugin({}),
    viteObfuscateFile(obfuscator_options),
    dynamicImport(),
  ],
  build: {
    target: "esnext",
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          if (id.includes("node_modules"))
            return id
              .toString()
              .split("node_modules/")[1]
              .split("/")[0]
              .toString();
        },
      },
    },
  },
});
