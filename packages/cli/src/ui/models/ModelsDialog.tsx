/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useCallback } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { RadioButtonSelect } from '../components/shared/RadioButtonSelect.js';
import type { LoadedSettings } from '../../config/settings.js';
import { SettingScope } from '../../config/settings.js';
import type { Config } from '@google/gemini-cli-core';
import { useKeypress } from '../hooks/useKeypress.js';
import { API_MODEL_MAPPINGS } from '@google/gemini-cli-core';
import type { ModelMapping } from '@google/gemini-cli-core';

interface ModelsDialogProps {
  config: Config;
  settings: LoadedSettings;
  onModelError: (error: string) => void;
  onModelSelect: (model: string | undefined, scope: SettingScope) => void;
}

export function ModelsDialog({
  config,
  settings,
  onModelError,
  onModelSelect,
}: ModelsDialogProps): React.JSX.Element {
  // Group models by provider for better organization
  const openaiModels = API_MODEL_MAPPINGS.filter(
    (m: ModelMapping) => m.provider === 'openai',
  );
  const anthropicModels = API_MODEL_MAPPINGS.filter(
    (m: ModelMapping) => m.provider === 'anthropic',
  );
  const googleModels = API_MODEL_MAPPINGS.filter(
    (m: ModelMapping) => m.provider === 'google',
  );

  // Create items array with provider groupings
  const items = [
    // OpenAI models
    ...openaiModels.map((model: ModelMapping) => ({
      label: `OpenAI: ${model.internal}`,
      value: model.internal,
    })),
    // Anthropic models
    ...anthropicModels.map((model: ModelMapping) => ({
      label: `Anthropic: ${model.internal}`,
      value: model.internal,
    })),
    // Google models
    ...googleModels.map((model: ModelMapping) => ({
      label: `Google: ${model.internal}`,
      value: model.internal,
    })),
  ];

  // Find initial selection index
  let initialModelIndex = 0;
  const selectedModel = (settings.merged as any).model?.selectedModel;
  if (selectedModel) {
    const index = items.findIndex((item) => item.value === selectedModel);
    if (index !== -1) {
      initialModelIndex = index;
    }
  } else {
    // Default to a popular model if none selected
    const defaultModel = 'gpt-4o';
    const index = items.findIndex((item) => item.value === defaultModel);
    if (index !== -1) {
      initialModelIndex = index;
    }
  }

  const onSelect = useCallback(
    async (model: string | undefined, scope: SettingScope) => {
      if (model) {
        onModelSelect(model, scope);
      }
    },
    [onModelSelect],
  );

  const handleModelSelect = (model: string) => {
    onSelect(model, SettingScope.User);
  };

  useKeypress(
    (key) => {
      if (key.name === 'escape') {
        onSelect(undefined, SettingScope.User);
      }
    },
    { isActive: true },
  );

  return (
    <Box
      borderStyle="round"
      borderColor={theme.border.default}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold color={theme.text.primary}>
        Model Selection
      </Text>
      <Box marginTop={1}>
        <Text color={theme.text.primary}>
          Which model would you like to use for content generation?
        </Text>
      </Box>
      <Box marginTop={1}>
        <RadioButtonSelect
          items={items}
          initialIndex={initialModelIndex}
          onSelect={handleModelSelect}
        />
      </Box>
      <Box marginTop={1}>
        <Text color={theme.text.secondary}>(Use Enter to select)</Text>
      </Box>
    </Box>
  );
}
