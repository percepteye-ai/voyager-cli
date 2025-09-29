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
  Tool,
  ToolListUnion,
  CallableTool,
  Part,
} from '@google/genai';
import {
  FinishReason,
  GenerateContentResponse as GenerateContentResponseClass,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import { mapToApiModel } from './apiModelMapping.js';
import { safeJsonParse } from '../utils/safeJsonParse.js';

export interface ChatApiRequest {
  message: string;
  model_name: string;
  // mode removed - now fetched from Supabase in backend
  stream?: boolean;
  // Tool calling support
  tools?: ToolApiDefinition[];
  conversation_history?: ChatApiMessage[];
}

export interface ChatApiResponse {
  message: string;
  model_name: string;
  mode: 'gateway' | 'byok' | 'byog'; // Still returned by backend for reference
  metadata?: {
    response_time_ms?: number;
    tokens_used?: number;
  };
  // Tool calling support
  tool_calls?: ToolApiCall[];
}

export interface ChatStreamChunk {
  chunk: string;
  finished: boolean;
  metadata?: Record<string, unknown>;
  // Tool calling support
  tool_calls?: ToolApiCall[];
}

export interface ChatApiMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_call_id?: string; // For tool messages
  tool_calls?: ToolApiCall[]; // For assistant messages
}

export interface ToolApiDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ToolApiCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

/**
 * API-based implementation of ContentGenerator that forwards requests to the chat API endpoints
 */
export class ApiContentGenerator implements ContentGenerator {
  private apiEndpoint: string;
  private authToken: string;
  private model: string;
  // mode removed - backend fetches it from Supabase

  constructor(apiEndpoint: string, authToken: string, model: string) {
    this.apiEndpoint = apiEndpoint;
    this.authToken = authToken;
    this.model = model;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<GenerateContentResponse> {
    try {
      const { conversationHistory, latestUserMessage } =
        this.convertRequestToApiFormat(request);

      const apiRequest: ChatApiRequest = {
        message: latestUserMessage || '',
        model_name: mapToApiModel(this.model),
        stream: false,
        conversation_history: conversationHistory,
      };

      // Add tools if present
      if (request.config?.tools) {
        apiRequest.tools = await this.convertGeminiToolsToApi(
          request.config.tools,
        );
      }

      const response = await this.makeApiRequest(
        '/api/chat/tool-call',
        apiRequest,
      );

      return this.convertApiResponseToGemini(response);
    } catch (error) {
      throw new Error(
        `API content generation error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    try {
      const { conversationHistory, latestUserMessage } =
        this.convertRequestToApiFormat(request);

      const apiRequest: ChatApiRequest = {
        message: latestUserMessage || '',
        model_name: mapToApiModel(this.model),
        stream: true,
        conversation_history: conversationHistory,
      };

      // Add tools if present
      if (request.config?.tools) {
        apiRequest.tools = await this.convertGeminiToolsToApi(
          request.config.tools,
        );
      }

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
    const response = await fetch(
      `${this.apiEndpoint}/api/chat/tool-call/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.authToken}`,
        },
        body: JSON.stringify(apiRequest),
      },
    );

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
    let lastChunkWithFinishReason: GenerateContentResponse | null = null;
    let hasYieldedLastChunk = false;

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
              const geminiResponse = this.convertApiChunkToGemini(data);

              // Check if this chunk has a finish reason
              const hasFinishReason =
                geminiResponse.candidates &&
                geminiResponse.candidates.length > 0 &&
                geminiResponse.candidates[0].finishReason;

              if (hasFinishReason) {
                lastChunkWithFinishReason = geminiResponse;
              }

              // Yield if there's content OR if this is a finish chunk
              if (
                geminiResponse.candidates &&
                geminiResponse.candidates.length > 0
              ) {
                const candidate = geminiResponse.candidates[0];

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
            } catch (_parseError) {
              // Skip malformed JSON lines
              continue;
            }
          }
        }
      }

      // Only yield the final chunk if we haven't already yielded it
      if (lastChunkWithFinishReason && !hasYieldedLastChunk) {
        yield lastChunkWithFinishReason;
      }
    } finally {
      reader.releaseLock();
    }
  }

  private convertContentsToMessage(contents: unknown): string {
    if (typeof contents === 'string') {
      return contents;
    }

    if (Array.isArray(contents)) {
      return contents
        .map((content) => {
          if (typeof content === 'string') {
            return content;
          }
          return (
            (content as any).parts?.map((part: any) => part.text).join(' ') ||
            ''
          );
        })
        .join(' ');
    }

    return '';
  }

  private extractTextFromContents(contents: unknown): string {
    return this.convertContentsToMessage(contents);
  }

  private convertApiResponseToGemini(
    response: ChatApiResponse,
  ): GenerateContentResponse {
    const geminiResponse = new GenerateContentResponseClass();
    const parts: Part[] = [];

    // Add text content
    if (response.message) {
      parts.push({ text: response.message });
    }

    // Add tool calls if present
    if (response.tool_calls) {
      for (const toolCall of response.tool_calls) {
        let args: Record<string, unknown> = {};
        if (toolCall.function.arguments) {
          args = safeJsonParse(toolCall.function.arguments, {});
        }

        parts.push({
          functionCall: {
            id: toolCall.id,
            name: toolCall.function.name,
            args,
          },
        });
      }
    }

    geminiResponse.candidates = [
      {
        content: {
          parts,
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

  private convertApiChunkToGemini(
    chunk: ChatStreamChunk,
  ): GenerateContentResponse {
    const geminiResponse = new GenerateContentResponseClass();
    const parts: Part[] = [];

    // Add text content
    if (chunk.chunk) {
      parts.push({ text: chunk.chunk });
    }

    // Add tool calls if present
    if (chunk.tool_calls) {
      for (const toolCall of chunk.tool_calls) {
        let args: Record<string, unknown> = {};
        if (toolCall.function.arguments) {
          args = safeJsonParse(toolCall.function.arguments, {});
        }

        parts.push({
          functionCall: {
            id: toolCall.id,
            name: toolCall.function.name,
            args,
          },
        });
      }
    }

    geminiResponse.candidates = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason: chunk.finished
          ? FinishReason.STOP
          : FinishReason.FINISH_REASON_UNSPECIFIED,
      },
    ];

    return geminiResponse;
  }

  private convertRequestToApiFormat(request: GenerateContentParameters): {
    conversationHistory: ChatApiMessage[];
    latestUserMessage: string;
  } {
    const conversationHistory: ChatApiMessage[] = [];
    let latestUserMessage = '';
    const pendingToolCalls = new Set<string>();

    // Process conversation history
    if (Array.isArray(request.contents)) {
      for (let i = 0; i < request.contents.length; i++) {
        const content = request.contents[i];

        if (
          (content as { role?: string }).role === 'user' ||
          (content as { role?: string }).role === 'system'
        ) {
          // Check if this user message contains function responses (tool results)
          const functionResponses =
            this.extractFunctionResponsesFromContent(content);

          if (functionResponses.length > 0) {
            // This user message contains tool responses, add them as tool messages
            // But only if we have pending tool calls to match them with
            for (const functionResponse of functionResponses) {
              if (pendingToolCalls.has(functionResponse.id)) {
                conversationHistory.push({
                  role: 'tool',
                  content: functionResponse.response,
                  tool_call_id: functionResponse.id,
                });
                pendingToolCalls.delete(functionResponse.id);
              }
            }
          }

          // Always check for text content and add as user/system message
          const textContent = this.extractTextFromContent(content);
          if (textContent) {
            // If this is a user message, it's the latest user message
            if ((content as { role?: string }).role === 'user') {
              latestUserMessage = textContent;
            }
            conversationHistory.push({
              role: (content as { role?: string }).role as 'user' | 'system',
              content: textContent,
            });
          }
        } else if ((content as { role?: string }).role === 'model') {
          const textContent = this.extractTextFromContent(content);
          const toolCalls = this.extractToolCallsFromContent(content);

          // Track tool calls for future tool responses
          for (const toolCall of toolCalls) {
            pendingToolCalls.add(toolCall.id);
          }

          conversationHistory.push({
            role: 'assistant',
            content: textContent,
            tool_calls: toolCalls,
          });
        } else if ((content as { role?: string }).role === 'function') {
          // Handle function responses (legacy format)
          const functionResponse =
            this.extractFunctionResponseFromContent(content);
          if (functionResponse) {
            if (pendingToolCalls.has(functionResponse.id)) {
              conversationHistory.push({
                role: 'tool',
                content: functionResponse.response,
                tool_call_id: functionResponse.id,
              });
              pendingToolCalls.delete(functionResponse.id);
            }
          }
        }
      }
    }

    return {
      conversationHistory,
      latestUserMessage,
    };
  }

  private extractTextFromContent(content: unknown): string {
    if (typeof content === 'string') {
      return content;
    }

    if ((content as any).parts) {
      return (content as any).parts
        .map((part: any) => {
          if (typeof part === 'string') return part;
          if (part.text) return part.text;
          return '';
        })
        .join(' ');
    }

    return '';
  }

  private extractToolCallsFromContent(content: unknown): ToolApiCall[] {
    if (!(content as any).parts) return [];

    const toolCalls: ToolApiCall[] = [];
    for (const part of (content as any).parts) {
      if (part.functionCall) {
        toolCalls.push({
          id:
            part.functionCall.id ||
            `call_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
          type: 'function',
          function: {
            name: part.functionCall.name || '',
            arguments: JSON.stringify(part.functionCall.args || {}),
          },
        });
      }
    }

    return toolCalls;
  }

  private extractFunctionResponseFromContent(
    content: unknown,
  ): { id: string; response: string } | null {
    if (!(content as any).parts) return null;

    for (const part of (content as any).parts) {
      if (part.functionResponse) {
        return {
          id: part.functionResponse.id || '',
          response:
            typeof part.functionResponse.response === 'string'
              ? part.functionResponse.response
              : JSON.stringify(part.functionResponse.response),
        };
      }
    }

    return null;
  }

  private extractFunctionResponsesFromContent(
    content: unknown,
  ): Array<{ id: string; response: string }> {
    const functionResponses: Array<{ id: string; response: string }> = [];

    if (!(content as any).parts) return functionResponses;

    for (const part of (content as any).parts) {
      if (part.functionResponse) {
        functionResponses.push({
          id: part.functionResponse.id || '',
          response:
            typeof part.functionResponse.response === 'string'
              ? part.functionResponse.response
              : JSON.stringify(part.functionResponse.response),
        });
      }
    }

    return functionResponses;
  }

  private async convertGeminiToolsToApi(
    tools: ToolListUnion,
  ): Promise<ToolApiDefinition[]> {
    const apiTools: ToolApiDefinition[] = [];

    for (const tool of tools) {
      let actualTool: Tool;

      // Handle CallableTool vs Tool
      if ('tool' in tool) {
        // This is a CallableTool
        actualTool = await (tool as CallableTool).tool();
      } else {
        // This is already a Tool
        actualTool = tool as Tool;
      }

      if (actualTool.functionDeclarations) {
        for (const func of actualTool.functionDeclarations) {
          if (func.name && func.description) {
            let parameters: Record<string, unknown> | undefined;

            // Handle both Gemini tools (parameters) and MCP tools (parametersJsonSchema)
            if (func.parametersJsonSchema) {
              // MCP tool format - use parametersJsonSchema directly
              parameters = func.parametersJsonSchema as Record<string, unknown>;
            } else if (func.parameters) {
              // Gemini tool format - convert parameters to API format
              parameters = this.convertGeminiToolParametersToApi(
                func.parameters as Record<string, unknown>,
              );
            }

            apiTools.push({
              type: 'function',
              function: {
                name: func.name,
                description: func.description,
                parameters: parameters || {},
              },
            });
          }
        }
      }
    }

    return apiTools;
  }

  private convertGeminiToolParametersToApi(
    parameters: Record<string, unknown>,
  ): Record<string, unknown> | undefined {
    if (!parameters || typeof parameters !== 'object') {
      return parameters;
    }

    const converted = JSON.parse(JSON.stringify(parameters));

    const convertTypes = (obj: unknown): unknown => {
      if (typeof obj !== 'object' || obj === null) {
        return obj;
      }

      if (Array.isArray(obj)) {
        return obj.map(convertTypes);
      }

      const result: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key === 'type' && typeof value === 'string') {
          // Convert Gemini types to API types
          const lowerValue = value.toLowerCase();
          if (lowerValue === 'integer') {
            result[key] = 'integer';
          } else if (lowerValue === 'number') {
            result[key] = 'number';
          } else {
            result[key] = lowerValue;
          }
        } else if (
          key === 'minimum' ||
          key === 'maximum' ||
          key === 'multipleOf'
        ) {
          // Ensure numeric constraints are actual numbers, not strings
          if (typeof value === 'string' && !isNaN(Number(value))) {
            result[key] = Number(value);
          } else {
            result[key] = value;
          }
        } else if (
          key === 'minLength' ||
          key === 'maxLength' ||
          key === 'minItems' ||
          key === 'maxItems'
        ) {
          // Ensure length constraints are integers, not strings
          if (typeof value === 'string' && !isNaN(Number(value))) {
            result[key] = parseInt(value, 10);
          } else {
            result[key] = value;
          }
        } else if (typeof value === 'object') {
          result[key] = convertTypes(value);
        } else {
          result[key] = value;
        }
      }
      return result;
    };

    return convertTypes(converted) as Record<string, unknown> | undefined;
  }
}
