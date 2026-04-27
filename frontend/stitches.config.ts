// frontend/stitches.config.ts
import { createStitches } from '@stitches/react';

export const { styled, css, globalCss, theme, config } = createStitches({
  theme: {
    colors: {
      primaryBg: '#FCFDFB',
      textMain: '#1A1C1E',
      palePeach: '#FFF5EB',
      softMint: '#F0FFF4',
      mistyLavender: '#F5F3FF',
      sage: '#879185',
      charcoal: '#36393E',
    },
    shadows: {
      softDepth: '0 20px 40px -5px rgba(0, 0, 0, 0.03)',
    },
    radii: {
      porcelain: '32px',
      card: '12px',
    },
    fonts: {
      heading: 'Lora, serif',
      body: 'DM Sans, sans-serif',
    },
  },
});