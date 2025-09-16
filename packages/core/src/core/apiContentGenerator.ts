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
import {
  FinishReason,
  GenerateContentResponse as GenerateContentResponseClass,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { Config } from '../config/config.js';
import { mapToApiModel } from './apiModelMapping.js';

export interface ChatApiRequest {
  message: string;
  model_name: string;
  // mode removed - now fetched from Supabase in backend
  stream?: boolean;
}

export interface ChatApiResponse {
  message: string;
  model_name: string;
  mode: 'gateway' | 'byok' | 'byog'; // Still returned by backend for reference
  metadata?: {
    response_time_ms?: number;
    tokens_used?: number;
  };
}

export interface ChatStreamChunk {
  chunk: string;
  finished: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * API-based implementation of ContentGenerator that forwards requests to the chat API endpoints
 */
export class ApiContentGenerator implements ContentGenerator {
  private apiEndpoint: string;
  private authToken: string;
  private model: string;
  // mode removed - backend fetches it from Supabase

  constructor(
    apiEndpoint: string,
    authToken: string,
    model: string,
    config?: Config,
  ) {
    this.apiEndpoint = apiEndpoint;
    this.authToken = authToken;
    this.model = model;
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    try {
      const message = this.convertContentsToMessage(request.contents);

      const apiRequest: ChatApiRequest = {
        message,
        model_name: mapToApiModel(this.model),
        stream: false,
      };

      const response = await this.makeApiRequest('/api/chat/send', apiRequest);

      return this.convertApiResponseToGemini(response);
    } catch (error) {
      throw new Error(
        `API content generation error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    try {
      const message = this.convertContentsToMessage(request.contents);

      const apiRequest: ChatApiRequest = {
        message,
        model_name: mapToApiModel(this.model),
        // mode removed - now fetched from Supabase in backend
        stream: true,
      };

      return this.convertApiStreamToGemini(apiRequest);
    } catch (error) {
      throw new Error(
        `API content streaming error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    try {
      const message = this.convertContentsToMessage(request.contents);

      // For now, we'll estimate tokens since the API doesn't provide a direct token counting endpoint
      // In a real implementation, you might want to add a separate token counting endpoint
      const estimatedTokens = Math.ceil(message.length / 4); // Rough estimation: 1 token ≈ 4 characters

      return {
        totalTokens: estimatedTokens,
      };
    } catch (error) {
      throw new Error(
        `API token counting error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    try {
      const text = this.extractTextFromContents(request.contents);

      // The API doesn't support embeddings, so we'll return empty embeddings
      // In a real implementation, you might want to add a separate embeddings endpoint
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
        `API embedding error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  getModel(): string {
    return this.model;
  }

  private async makeApiRequest(
    endpoint: string,
    data: ChatApiRequest,
  ): Promise<ChatApiResponse> {
    const response = await fetch(`${this.apiEndpoint}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.authToken}`,
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `API request failed: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    return response.json();
  }

  private async *convertApiStreamToGemini(
    apiRequest: ChatApiRequest,
  ): AsyncGenerator<GenerateContentResponse> {
    const response = await fetch(`${this.apiEndpoint}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.authToken}`,
      },
      body: JSON.stringify(apiRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `API stream request failed: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body reader available');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;

          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6)) as ChatStreamChunk;
              if (data.chunk) {
                const geminiResponse = new GenerateContentResponseClass();
                geminiResponse.candidates = [
                  {
                    content: {
                      parts: [{ text: data.chunk }],
                      role: 'model',
                    },
                    finishReason: data.finished
                      ? FinishReason.STOP
                      : FinishReason.FINISH_REASON_UNSPECIFIED,
                  },
                ];
                yield geminiResponse;
              }
            } catch (parseError) {
              // Skip malformed JSON lines
              continue;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private convertContentsToMessage(contents: any): string {
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

  private extractTextFromContents(contents: any): string {
    return this.convertContentsToMessage(contents);
  }

  private convertApiResponseToGemini(
    response: ChatApiResponse,
  ): GenerateContentResponse {
    const geminiResponse = new GenerateContentResponseClass();
    geminiResponse.candidates = [
      {
        content: {
          parts: [{ text: response.message }],
          role: 'model',
        },
        finishReason: FinishReason.STOP,
      },
    ];

    if (response.metadata?.tokens_used) {
      geminiResponse.usageMetadata = {
        promptTokenCount: 0, // API doesn't provide prompt token count
        candidatesTokenCount: response.metadata.tokens_used,
        totalTokenCount: response.metadata.tokens_used,
      };
    }

    return geminiResponse;
  }
}
