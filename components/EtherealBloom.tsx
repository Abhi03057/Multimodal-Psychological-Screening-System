import { styled, breathing } from '../stitches.config';

/**
 * Zen Mode Questionnaire Components
 */
export const ZenContainer = styled('div', {
  minHeight: 'calc(100vh - 84px)',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '$5 $4',
  maxWidth: '800px',
  margin: '0 auto',
});

export const ZenProgressLine = styled('div', {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  height: '2px',
  background: '$outline',
  zIndex: 100,
  
  '&::after': {
    content: '""',
    position: 'absolute',
    left: 0,
    top: 0,
    height: '100%',
    width: 'var(--progress, 0%)',
    background: '$sage',
    transition: 'width 0.6s cubic-bezier(0.2, 0.8, 0.2, 1)',
  }
});

export const ZenQuestion = styled('h2', {
  fontFamily: '$display',
  fontSize: 'clamp(2rem, 5vw, 3.5rem)',
  fontWeight: 400,
  lineHeight: 1.2,
  color: '$text',
  textAlign: 'center',
  marginBottom: '$6',
  maxWidth: '720px',
});

/**
 * Live Session Interface Components
 */
export const WebcamContainer = styled('div', {
  width: '100%',
  aspectRatio: '16/9',
  borderRadius: '$xl',
  border: '32px solid #ffffff',
  boxShadow: '$softDepth',
  position: 'relative',
  overflow: 'hidden',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: '#EAEAEA',
  transition: '$smooth',
  
  variants: {
    recording: {
      true: {
        animation: `${breathing} 3s infinite ease-in-out`,
      }
    }
  }
});

/**
 * Editorial Report Card Components
 */
export const ReportCard = styled('div', {
  background: '$surface',
  borderRadius: '$xl',
  padding: '$5',
  boxShadow: '$softDepth',
  border: '1px solid $outline',
  display: 'flex',
  flexDirection: 'column',
  gap: '$3',
  transition: '$smooth',
  activeTap: '', // Using the utility
  
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 20px 50px -10px rgba(0, 0, 0, 0.05)',
  }
});

export const DataVizLine = styled('div', {
  width: '100%',
  height: '2px',
  background: '$outline',
  position: 'relative',
  margin: '$3 0',
  
  '&::after': {
    content: '""',
    position: 'absolute',
    left: 0,
    top: 0,
    height: '100%',
    width: 'var(--value, 0%)',
    background: 'var(--viz-color, $sage)',
  }
});

export const CardLabel = styled('span', {
  fontFamily: '$body',
  fontSize: '$labelSm',
  fontWeight: 600,
  letterSpacing: '$wide',
  color: '$sage',
  textTransform: 'uppercase',
});

export const CardValue = styled('div', {
  fontFamily: '$display',
  fontSize: '2rem',
  color: '$text',
});
