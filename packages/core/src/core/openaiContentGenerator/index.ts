/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  ContentGenerator,
  ContentGeneratorConfig,
} from '../contentGenerator.js';
import type { Config } from '../../config/config.js';
import { OpenAIContentGenerator } from './openaiContentGenerator.js';
import { DefaultOpenAICompatibleProvider } from './provider/index.js';

export { OpenAIContentGenerator } from './openaiContentGenerator.js';
export { ContentGenerationPipeline, type PipelineConfig } from './pipeline.js';

export { DefaultOpenAICompatibleProvider } from './provider/index.js';

export { OpenAIContentConverter } from './converter.js';

/**
 * Create an OpenAI-compatible content generator with the appropriate provider
 */
export function createOpenAIContentGenerator(
  contentGeneratorConfig: ContentGeneratorConfig,
  cliConfig: Config,
): ContentGenerator {
  const provider = new DefaultOpenAICompatibleProvider(
    contentGeneratorConfig,
    cliConfig,
  );
  return new OpenAIContentGenerator(
    contentGeneratorConfig,
    cliConfig,
    provider,
  );
}
