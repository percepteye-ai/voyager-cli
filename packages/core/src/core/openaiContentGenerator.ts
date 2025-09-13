/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
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
 * OpenAI implementation of ContentGenerator that adapts OpenAI's API to match Gemini's interface
 */
export class OpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI;
  private model: string;

  constructor(apiKey: string, model: string = 'gpt-4o', config?: Config) {
    this.model = model;
    this.openai = new OpenAI({
      apiKey,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    try {
      const messages = this.convertGeminiMessagesToOpenAI(request.contents);

      const response = await this.openai.chat.completions.create({
        model: this.model,
        messages,
        temperature: request.config?.temperature,
        max_tokens: request.config?.maxOutputTokens,
        top_p: request.config?.topP,
        stop: request.config?.stopSequences,
      });

      return this.convertOpenAIResponseToGemini(response);
    } catch (error) {
      throw new Error(
        `OpenAI API error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    try {
      const messages = this.convertGeminiMessagesToOpenAI(request.contents);

      const stream = await this.openai.chat.completions.create({
        model: this.model,
        messages,
        temperature: request.config?.temperature,
        max_tokens: request.config?.maxOutputTokens,
        top_p: request.config?.topP,
        stop: request.config?.stopSequences,
        stream: true,
      });

      return this.convertOpenAIStreamToGemini(stream);
    } catch (error) {
      throw new Error(
        `OpenAI API error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    try {
      const messages = this.convertGeminiMessagesToOpenAI(request.contents);

      // OpenAI doesn't have a direct token counting API, so we'll estimate
      // This is a rough estimation - in production you might want to use a more accurate method
      const text = messages.map((msg) => msg.content).join(' ');
      const estimatedTokens = Math.ceil(text.length / 4); // Rough estimation: 1 token ≈ 4 characters

      return {
        totalTokens: estimatedTokens,
      };
    } catch (error) {
      throw new Error(
        `OpenAI token counting error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    try {
      const text = this.extractTextFromContents(request.contents);

      const response = await this.openai.embeddings.create({
        model: 'text-embedding-3-small', // Using a default embedding model
        input: text,
      });

      return {
        embeddings: [
          {
            values: response.data[0]?.embedding || [],
          },
        ],
      };
    } catch (error) {
      throw new Error(
        `OpenAI embedding error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  getModel(): string {
    return this.model;
  }

  private convertGeminiMessagesToOpenAI(
    contents: any,
  ): Array<{ role: 'user' | 'assistant' | 'system'; content: string }> {
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

  private convertOpenAIResponseToGemini(
    response: any,
  ): GenerateContentResponse {
    const content = response.choices?.[0]?.message?.content || '';

    const geminiResponse = new GenerateContentResponseClass();
    geminiResponse.candidates = [
      {
        content: {
          parts: [{ text: content }],
          role: 'model',
        },
        finishReason: this.convertFinishReason(
          response.choices?.[0]?.finish_reason,
        ),
      },
    ];
    geminiResponse.usageMetadata = {
      promptTokenCount: response.usage?.prompt_tokens || 0,
      candidatesTokenCount: response.usage?.completion_tokens || 0,
      totalTokenCount: response.usage?.total_tokens || 0,
    };

    return geminiResponse;
  }

  private async *convertOpenAIStreamToGemini(
    stream: any,
  ): AsyncGenerator<GenerateContentResponse> {
    for await (const chunk of stream) {
      const content = chunk.choices?.[0]?.delta?.content || '';
      if (content) {
        const geminiResponse = new GenerateContentResponseClass();
        geminiResponse.candidates = [
          {
            content: {
              parts: [{ text: content }],
              role: 'model',
            },
            finishReason: this.convertFinishReason(
              chunk.choices?.[0]?.finish_reason,
            ),
          },
        ];
        yield geminiResponse;
      }
    }
  }

  private convertFinishReason(reason: string): FinishReason {
    switch (reason) {
      case 'stop':
        return FinishReason.STOP;
      case 'length':
        return FinishReason.MAX_TOKENS;
      case 'content_filter':
        return FinishReason.SAFETY;
      default:
        return FinishReason.FINISH_REASON_UNSPECIFIED;
    }
  }
}
