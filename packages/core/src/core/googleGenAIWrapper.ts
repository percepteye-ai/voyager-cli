/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';

/**
 * Wrapper for GoogleGenAI models that adds the getModel() method
 */
export class GoogleGenAIWrapper implements ContentGenerator {
  private models: any;
  private modelName: string;

  constructor(models: any, modelName: string) {
    this.models = models;
    this.modelName = modelName;
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    return this.models.generateContent(request, userPromptId);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    return this.models.generateContentStream(request, userPromptId);
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    return this.models.countTokens(request);
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    return this.models.embedContent(request);
  }

  getModel(): string {
    return this.modelName;
  }
}
