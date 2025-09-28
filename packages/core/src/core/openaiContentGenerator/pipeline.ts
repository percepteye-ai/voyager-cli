/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type OpenAI from 'openai';
import {
  type GenerateContentParameters,
  GenerateContentResponse,
} from '@google/genai';
import type { Config } from '../../config/config.js';
import type { ContentGeneratorConfig } from '../contentGenerator.js';
import type { OpenAICompatibleProvider } from './provider/index.js';
import { OpenAIContentConverter } from './converter.js';

export interface PipelineConfig {
  cliConfig: Config;
  provider: OpenAICompatibleProvider;
  contentGeneratorConfig: ContentGeneratorConfig;
}

export class ContentGenerationPipeline {
  client: OpenAI;
  private converter: OpenAIContentConverter;
  private contentGeneratorConfig: ContentGeneratorConfig;

  constructor(private config: PipelineConfig) {
    this.contentGeneratorConfig = config.contentGeneratorConfig;
    this.client = this.config.provider.buildClient();
    this.converter = new OpenAIContentConverter(
      this.contentGeneratorConfig.openaiModel || 'gpt-4o',
    );
  }

  async execute(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const startTime = Date.now();

    try {
      const openaiRequest = await this.buildRequest(
        request,
        userPromptId,
        false,
      );
      const openaiResponse = (await this.client.chat.completions.create(
        openaiRequest,
      )) as OpenAI.Chat.ChatCompletion;

      const geminiResponse =
        this.converter.convertOpenAIResponseToGemini(openaiResponse);

      return geminiResponse;
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage =
        error instanceof Error ? error.message : String(error);

      // Handle specific tool call flow errors
      if (
        errorMessage.includes('tool_calls') &&
        errorMessage.includes('tool_call_id')
      ) {
        console.error('OpenAI Tool Call Flow Error:', {
          error: errorMessage,
          duration,
          userPromptId,
          suggestion:
            'This error indicates that tool calls were made but not properly followed by tool response messages. Check conversation history.',
        });
      } else {
        console.error('OpenAI API Error:', {
          error: errorMessage,
          duration,
          userPromptId,
        });
      }
      throw error;
    }
  }

  async executeStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const startTime = Date.now();

    try {
      const openaiRequest = await this.buildRequest(
        request,
        userPromptId,
        true,
      );

      // Create OpenAI stream
      const stream = (await this.client.chat.completions.create(
        openaiRequest,
      )) as AsyncIterable<OpenAI.Chat.ChatCompletionChunk>;

      // Process stream with conversion
      return this.processStream(stream, startTime, userPromptId);
    } catch (error) {
      const duration = Date.now() - startTime;
      console.error('OpenAI API Streaming Error:', {
        error: error instanceof Error ? error.message : String(error),
        duration,
        userPromptId,
      });
      throw error;
    }
  }

  private async processStream(
    stream: AsyncIterable<OpenAI.Chat.ChatCompletionChunk>,
    startTime: number,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const self = this;
    return (async function* () {
      let hasReceivedAnyChunk = false;
      let lastChunkWithFinishReason: GenerateContentResponse | null = null;
      let hasYieldedLastChunk = false;

      try {
        for await (const chunk of stream) {
          hasReceivedAnyChunk = true;
          const geminiResponse =
            self.converter.convertOpenAIChunkToGemini(chunk);

          // Check if this chunk has a finish reason
          const hasFinishReason =
            geminiResponse.candidates &&
            geminiResponse.candidates.length > 0 &&
            geminiResponse.candidates[0].finishReason;

          if (hasFinishReason) {
            lastChunkWithFinishReason = geminiResponse;
          }

          // Yield responses that have content OR finish reason
          if (
            geminiResponse.candidates &&
            geminiResponse.candidates.length > 0
          ) {
            const candidate = geminiResponse.candidates[0];

            // Yield if there's content OR if this is a finish chunk
            if (
              (candidate.content &&
                candidate.content.parts &&
                candidate.content.parts.length > 0) ||
              candidate.finishReason
            ) {
              yield geminiResponse;

              // Mark if we've yielded the last chunk with finish reason
              if (hasFinishReason) {
                hasYieldedLastChunk = true;
              }
            }
          }
        }

        // Only yield the final chunk if we haven't already yielded it
        if (lastChunkWithFinishReason && !hasYieldedLastChunk) {
          if (process.env['DEBUG_OPENAI_STREAMING']) {
            console.log('Yielding final chunk with finish reason:', {
              finishReason:
                lastChunkWithFinishReason.candidates?.[0]?.finishReason,
              hasContent:
                !!lastChunkWithFinishReason.candidates?.[0]?.content?.parts
                  ?.length,
            });
          }
          yield lastChunkWithFinishReason;
        } else if (process.env['DEBUG_OPENAI_STREAMING']) {
          if (!lastChunkWithFinishReason) {
            console.warn(
              'No chunk with finish reason found - this may cause stream validation to fail',
            );
          } else {
            console.log('Final chunk with finish reason already yielded');
          }
        }

        // If we haven't received any chunks at all, this is an error
        if (!hasReceivedAnyChunk) {
          throw new Error('OpenAI stream completed without any chunks');
        }
      } catch (error) {
        const duration = Date.now() - startTime;
        console.error('OpenAI Stream Processing Error:', {
          error: error instanceof Error ? error.message : String(error),
          duration,
          userPromptId,
          hasReceivedAnyChunk,
          hasLastChunkWithFinishReason: !!lastChunkWithFinishReason,
          hasYieldedLastChunk,
        });
        throw error;
      }
    })();
  }

  private async buildRequest(
    request: GenerateContentParameters,
    userPromptId: string,
    streaming: boolean = false,
  ): Promise<OpenAI.Chat.ChatCompletionCreateParams> {
    const messages = this.converter.convertGeminiRequestToOpenAI(request);

    // Debug logging for conversation flow
    if (process.env['DEBUG_OPENAI_CONVERSATION']) {
      console.log(
        'OpenAI Request Messages:',
        JSON.stringify(messages, null, 2),
      );
    }

    // Apply provider-specific enhancements
    const baseRequest: OpenAI.Chat.ChatCompletionCreateParams = {
      model: this.contentGeneratorConfig.openaiModel || 'gpt-4o',
      messages,
      ...this.buildSamplingParameters(request),
    };

    // Debug logging for streaming
    if (process.env['DEBUG_OPENAI_STREAMING']) {
      console.log('OpenAI Streaming Request:', {
        model: baseRequest.model,
        streaming,
        messageCount: messages.length,
        hasTools: !!request.config?.tools,
      });
    }

    // Let provider enhance the request
    const enhancedRequest = this.config.provider.buildRequest(
      baseRequest,
      userPromptId,
    );

    // Add tools if present
    if (request.config?.tools) {
      enhancedRequest.tools = await this.converter.convertGeminiToolsToOpenAI(
        request.config.tools,
      );

      if (process.env['DEBUG_OPENAI_CONVERSATION']) {
        console.log(
          'OpenAI Tools:',
          JSON.stringify(enhancedRequest.tools, null, 2),
        );
      }
    }

    // Add streaming options if needed
    if (streaming) {
      enhancedRequest.stream = true;
      enhancedRequest.stream_options = { include_usage: true };
    }

    return enhancedRequest;
  }

  private buildSamplingParameters(
    request: GenerateContentParameters,
  ): Record<string, unknown> {
    const params: Record<string, unknown> = {};

    if (request.config?.temperature !== undefined) {
      params['temperature'] = request.config.temperature;
    }

    if (request.config?.maxOutputTokens !== undefined) {
      params['max_tokens'] = request.config.maxOutputTokens;
    }

    if (request.config?.topP !== undefined) {
      params['top_p'] = request.config.topP;
    }

    if (
      request.config?.stopSequences &&
      request.config.stopSequences.length > 0
    ) {
      params['stop'] = request.config.stopSequences;
    }

    return params;
  }
}
