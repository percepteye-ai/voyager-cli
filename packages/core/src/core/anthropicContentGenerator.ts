/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import Anthropic from '@anthropic-ai/sdk';
import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import {
  FinishReason,
  GenerateContentResponse as GenerateContentResponseClass,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { Config } from '../config/config.js';

/**
 * Anthropic implementation of ContentGenerator that adapts Anthropic's API to match Gemini's interface
 */
export class AnthropicContentGenerator implements ContentGenerator {
  private anthropic: Anthropic;
  private model: string;

  constructor(
    apiKey: string,
    model: string = 'claude-3-5-sonnet-20241022',
    config?: Config,
  ) {
    this.model = model;
    this.anthropic = new Anthropic({
      apiKey,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    try {
      const messages = this.convertGeminiMessagesToAnthropic(request.contents);

      const response = await this.anthropic.messages.create({
        model: this.model,
        messages,
        temperature: request.config?.temperature,
        max_tokens: request.config?.maxOutputTokens || 4096,
        top_p: request.config?.topP,
        stop_sequences: request.config?.stopSequences,
      });

      return this.convertAnthropicResponseToGemini(response);
    } catch (error) {
      throw new Error(
        `Anthropic API error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    try {
      const messages = this.convertGeminiMessagesToAnthropic(request.contents);

      const stream = await this.anthropic.messages.create({
        model: this.model,
        messages,
        temperature: request.config?.temperature,
        max_tokens: request.config?.maxOutputTokens || 4096,
        top_p: request.config?.topP,
        stop_sequences: request.config?.stopSequences,
        stream: true,
      });

      return this.convertAnthropicStreamToGemini(stream);
    } catch (error) {
      throw new Error(
        `Anthropic API error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    try {
      const messages = this.convertGeminiMessagesToAnthropic(request.contents);

      // Anthropic doesn't have a direct token counting API, so we'll estimate
      // This is a rough estimation - in production you might want to use a more accurate method
      const text = messages.map((msg) => msg.content).join(' ');
      const estimatedTokens = Math.ceil(text.length / 4); // Rough estimation: 1 token ≈ 4 characters

      return {
        totalTokens: estimatedTokens,
      };
    } catch (error) {
      throw new Error(
        `Anthropic token counting error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    try {
      const text = this.extractTextFromContents(request.contents);

      // Anthropic doesn't have a dedicated embedding API, so we'll return empty embeddings
      // In a real implementation, you might want to use a different service for embeddings
      // For now, we'll just validate the text is not empty
      if (!text.trim()) {
        throw new Error('No content provided for embedding');
      }

      return {
        embeddings: [
          {
            values: [],
          },
        ],
      };
    } catch (error) {
      throw new Error(
        `Anthropic embedding error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  getModel(): string {
    return this.model;
  }

  private convertGeminiMessagesToAnthropic(
    contents: any,
  ): Array<{ role: 'user' | 'assistant'; content: string }> {
    if (typeof contents === 'string') {
      return [{ role: 'user', content: contents }];
    }

    if (Array.isArray(contents)) {
      return contents.map((content) => {
        if (typeof content === 'string') {
          return { role: 'user' as const, content };
        }
        const role =
          content.role === 'user' ? ('user' as const) : ('assistant' as const);
        const text =
          content.parts?.map((part: any) => part.text).join(' ') || '';
        return { role, content: text };
      });
    }

    return [];
  }

  private extractTextFromContents(contents: any): string {
    if (typeof contents === 'string') {
      return contents;
    }

    if (Array.isArray(contents)) {
      return contents
        .map((content) => {
          if (typeof content === 'string') {
            return content;
          }
          return content.parts?.map((part: any) => part.text).join(' ') || '';
        })
        .join(' ');
    }

    return '';
  }

  private convertAnthropicResponseToGemini(
    response: any,
  ): GenerateContentResponse {
    const content = response.content?.[0]?.text || '';

    const geminiResponse = new GenerateContentResponseClass();
    geminiResponse.candidates = [
      {
        content: {
          parts: [{ text: content }],
          role: 'model',
        },
        finishReason: this.convertFinishReason(response.stop_reason),
      },
    ];
    geminiResponse.usageMetadata = {
      promptTokenCount: response.usage?.input_tokens || 0,
      candidatesTokenCount: response.usage?.output_tokens || 0,
      totalTokenCount:
        response.usage?.input_tokens + response.usage?.output_tokens || 0,
    };

    return geminiResponse;
  }

  private async *convertAnthropicStreamToGemini(
    stream: any,
  ): AsyncGenerator<GenerateContentResponse> {
    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta') {
        const content = chunk.delta?.text || '';
        if (content) {
          const geminiResponse = new GenerateContentResponseClass();
          geminiResponse.candidates = [
            {
              content: {
                parts: [{ text: content }],
                role: 'model',
              },
              finishReason: this.convertFinishReason(chunk.stop_reason),
            },
          ];
          yield geminiResponse;
        }
      }
    }
  }

  private convertFinishReason(reason: string): FinishReason {
    switch (reason) {
      case 'end_turn':
        return FinishReason.STOP;
      case 'max_tokens':
        return FinishReason.MAX_TOKENS;
      case 'stop_sequence':
        return FinishReason.STOP;
      default:
        return FinishReason.FINISH_REASON_UNSPECIFIED;
    }
  }
}
