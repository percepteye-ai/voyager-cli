/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Model mapping utilities for API-based content generation
 * Maps between internal model names and API model names
 */

export interface ModelMapping {
  internal: string;
  api: string;
  provider: 'openai' | 'anthropic' | 'google';
}

export const API_MODEL_MAPPINGS: ModelMapping[] = [
  // OpenAI models
  { internal: 'gpt-4o', api: 'gpt-4o', provider: 'openai' },
  { internal: 'gpt-4o-mini', api: 'gpt-4o-mini', provider: 'openai' },
  { internal: 'gpt-4-turbo', api: 'gpt-4-turbo', provider: 'openai' },
  { internal: 'gpt-3.5-turbo', api: 'gpt-3.5-turbo', provider: 'openai' },

  // Anthropic models
  {
    internal: 'claude-opus-4-1',
    api: 'claude-opus-4-1-20250805',
    provider: 'anthropic',
  },
  {
    internal: 'claude-opus-4-0',
    api: 'claude-opus-4-20250514',
    provider: 'anthropic',
  },
  {
    internal: 'claude-sonnet-4-0',
    api: 'claude-sonnet-4-20250514',
    provider: 'anthropic',
  },
  {
    internal: 'claude-3-7-sonnet-latest',
    api: 'claude-3-7-sonnet-20250219',
    provider: 'anthropic',
  },
  {
    internal: 'claude-3-5-haiku-latest',
    api: 'claude-3-5-haiku-20241022',
    provider: 'anthropic',
  },

  // Google models
  { internal: 'gemini-2.5-pro', api: 'gemini-2.5-pro', provider: 'google' },
  { internal: 'gemini-1.5-pro', api: 'gemini-1.5-pro', provider: 'google' },
  { internal: 'gemini-1.5-flash', api: 'gemini-1.5-flash', provider: 'google' },
  { internal: 'gemini-pro', api: 'gemini-pro', provider: 'google' },

  // Legacy Gemini models
  { internal: 'gemini-2.5-flash', api: 'gemini-1.5-flash', provider: 'google' },
  {
    internal: 'gemini-2.5-flash-lite',
    api: 'gemini-1.5-flash',
    provider: 'google',
  },
];

/**
 * Maps an internal model name to the corresponding API model name
 * @param internalModel The internal model name
 * @returns The API model name, or the original name if no mapping found
 */
export function mapToApiModel(internalModel: string): string {
  const mapping = API_MODEL_MAPPINGS.find((m) => m.internal === internalModel);
  return mapping ? mapping.api : internalModel;
}

/**
 * Maps an API model name back to the internal model name
 * @param apiModel The API model name
 * @returns The internal model name, or the original name if no mapping found
 */
export function mapFromApiModel(apiModel: string): string {
  const mapping = API_MODEL_MAPPINGS.find((m) => m.api === apiModel);
  return mapping ? mapping.internal : apiModel;
}

/**
 * Gets the provider for a given model name
 * @param modelName The model name (internal or API)
 * @returns The provider name, or null if not found
 */
export function getModelProvider(modelName: string): string | null {
  // First try to find by internal name
  let mapping = API_MODEL_MAPPINGS.find((m) => m.internal === modelName);

  // If not found, try to find by API name
  if (!mapping) {
    mapping = API_MODEL_MAPPINGS.find((m) => m.api === modelName);
  }

  return mapping ? mapping.provider : null;
}

/**
 * Gets all available models for a specific provider
 * @param provider The provider name
 * @returns Array of model mappings for the provider
 */
export function getModelsForProvider(
  provider: 'openai' | 'anthropic' | 'google',
): ModelMapping[] {
  return API_MODEL_MAPPINGS.filter((m) => m.provider === provider);
}

/**
 * Checks if a model is supported by the API
 * @param modelName The model name to check
 * @returns True if the model is supported
 */
export function isModelSupported(modelName: string): boolean {
  return API_MODEL_MAPPINGS.some(
    (m) => m.internal === modelName || m.api === modelName,
  );
}
