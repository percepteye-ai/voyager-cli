/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useCallback } from 'react';
import type { LoadedSettings } from '../../config/settings.js';
import { isModelSupported } from '@google/gemini-cli-core';

export function validateModelWithSettings(
  model: string,
  settings: LoadedSettings,
): string | null {
  if (!isModelSupported(model)) {
    return `Model '${model}' is not supported. Please select a supported model.`;
  }

  return null;
}

export const useModelsCommand = (settings: LoadedSettings) => {
  const [modelError, setModelError] = useState<string | null>(null);

  const onModelError = useCallback(
    (error: string) => {
      setModelError(error);
    },
    [setModelError],
  );

  const clearModelError = useCallback(() => {
    setModelError(null);
  }, []);

  return {
    modelError,
    onModelError,
    clearModelError,
  };
};
