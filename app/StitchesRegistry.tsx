'use client';

import React, { useState } from 'react';
import { useServerInsertedHTML } from 'next/navigation';
import { getCssText, globalStyles } from '../stitches.config';

export function StitchesRegistry({ children }: { children: React.ReactNode }) {
  const [isRendered, setIsRendered] = useState(false);
  
  if (!isRendered) {
    globalStyles();
    setIsRendered(true);
  }

  useServerInsertedHTML(() => {
    return <style id="stitches" dangerouslySetInnerHTML={{ __html: getCssText() }} />;
  });

  return <>{children}</>;
}
