/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  GenerateContentParameters,
  Part,
  Content,
  Tool,
  ToolListUnion,
  CallableTool,
  FunctionCall,
  FunctionResponse,
  ContentListUnion,
  ContentUnion,
  PartUnion,
  Candidate,
} from '@google/genai';
import { GenerateContentResponse, FinishReason } from '@google/genai';
import type OpenAI from 'openai';
import { safeJsonParse } from '../../utils/safeJsonParse.js';
import { StreamingToolCallParser } from './streamingToolCallParser.js';

/**
 * Parsed parts from Gemini content, categorized by type
 */
interface ParsedParts {
  textParts: string[];
  functionCalls: FunctionCall[];
  functionResponses: FunctionResponse[];
}

/**
 * Converter class for transforming data between Gemini and OpenAI formats
 */
export class OpenAIContentConverter {
  private model: string;
  private streamingToolCallParser: StreamingToolCallParser =
    new StreamingToolCallParser();

  constructor(model: string) {
    this.model = model;
  }

  /**
   * Reset streaming tool calls parser for new stream processing
   */
  resetStreamingToolCalls(): void {
    this.streamingToolCallParser.reset();
  }

  /**
   * Convert Gemini tool parameters to OpenAI JSON Schema format
   */
  convertGeminiToolParametersToOpenAI(
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
          // Convert Gemini types to OpenAI JSON Schema types
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

  /**
   * Convert Gemini tools to OpenAI format for API compatibility.
   */
  async convertGeminiToolsToOpenAI(
    geminiTools: ToolListUnion,
  ): Promise<OpenAI.Chat.ChatCompletionTool[]> {
    const openAITools: OpenAI.Chat.ChatCompletionTool[] = [];

    for (const tool of geminiTools) {
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
              if (func.parametersJsonSchema) {
                // Create a shallow copy to avoid mutating the original object
                const paramsCopy = {
                  ...(func.parametersJsonSchema as Record<string, unknown>),
                };
                parameters = paramsCopy;
              }
            } else if (func.parameters) {
              // Gemini tool format - convert parameters to OpenAI format
              parameters = this.convertGeminiToolParametersToOpenAI(
                func.parameters as Record<string, unknown>,
              );
            }

            openAITools.push({
              type: 'function',
              function: {
                name: func.name,
                description: func.description,
                parameters,
              },
            });
          }
        }
      }
    }

    return openAITools;
  }

  /**
   * Convert Gemini request to OpenAI message format
   */
  convertGeminiRequestToOpenAI(
    request: GenerateContentParameters,
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    // Handle system instruction from config
    this.addSystemInstructionMessage(request, messages);

    // Handle contents
    this.processContents(request.contents, messages);

    // Clean up orphaned tool calls and merge consecutive assistant messages
    const cleanedMessages = this.cleanOrphanedToolCalls(messages);
    const mergedMessages =
      this.mergeConsecutiveAssistantMessages(cleanedMessages);

    // Validate the conversation flow
    this.validateConversationFlow(mergedMessages);

    return mergedMessages;
  }

  /**
   * Extract and add system instruction message from request config
   */
  private addSystemInstructionMessage(
    request: GenerateContentParameters,
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): void {
    if (!request.config?.systemInstruction) return;

    const systemText = this.extractTextFromContentUnion(
      request.config.systemInstruction,
    );

    if (systemText) {
      messages.push({
        role: 'system' as const,
        content: systemText,
      });
    }
  }

  /**
   * Process contents and convert to OpenAI messages
   */
  private processContents(
    contents: ContentListUnion,
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): void {
    if (Array.isArray(contents)) {
      for (const content of contents) {
        this.processContent(content, messages);
      }
    } else if (contents) {
      this.processContent(contents, messages);
    }
  }

  /**
   * Process a single content item and convert to OpenAI message(s)
   */
  private processContent(
    content: ContentUnion | PartUnion,
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): void {
    if (typeof content === 'string') {
      messages.push({ role: 'user' as const, content });
      return;
    }

    if (!this.isContentObject(content)) return;

    const parsedParts = this.parseParts(content.parts || []);

    // Handle function responses (tool results) first
    if (parsedParts.functionResponses.length > 0) {
      for (const funcResponse of parsedParts.functionResponses) {
        messages.push({
          role: 'tool' as const,
          tool_call_id: funcResponse.id || '',
          content:
            typeof funcResponse.response === 'string'
              ? funcResponse.response
              : JSON.stringify(funcResponse.response),
        });
      }
      return;
    }

    // Handle model messages with function calls
    if (content.role === 'model' && parsedParts.functionCalls.length > 0) {
      const toolCalls = parsedParts.functionCalls.map((fc, index) => ({
        id: fc.id || `call_${Date.now()}_${index}`,
        type: 'function' as const,
        function: {
          name: fc.name || '',
          arguments: JSON.stringify(fc.args || {}),
        },
      }));

      const textContent = parsedParts.textParts.join('');

      messages.push({
        role: 'assistant' as const,
        content: textContent || null,
        tool_calls: toolCalls,
      });
      return;
    }

    // Handle regular messages with text content
    const role = content.role === 'model' ? 'assistant' : 'user';
    const textContent = parsedParts.textParts.join('');

    if (textContent) {
      messages.push({
        role: role as 'user' | 'assistant',
        content: textContent,
      });
    }
  }

  /**
   * Parse Gemini parts into categorized components
   */
  private parseParts(parts: Part[]): ParsedParts {
    const textParts: string[] = [];
    const functionCalls: FunctionCall[] = [];
    const functionResponses: FunctionResponse[] = [];

    for (const part of parts) {
      if (typeof part === 'string') {
        textParts.push(part);
      } else if ('text' in part && part.text) {
        textParts.push(part.text);
      } else if ('functionCall' in part && part.functionCall) {
        functionCalls.push(part.functionCall);
      } else if ('functionResponse' in part && part.functionResponse) {
        functionResponses.push(part.functionResponse);
      }
    }

    return { textParts, functionCalls, functionResponses };
  }

  /**
   * Type guard to check if content is a valid Content object
   */
  private isContentObject(
    content: unknown,
  ): content is { role: string; parts: Part[] } {
    return (
      typeof content === 'object' &&
      content !== null &&
      'role' in content &&
      'parts' in content &&
      Array.isArray((content as Record<string, unknown>)['parts'])
    );
  }

  /**
   * Extract text content from various Gemini content union types
   */
  private extractTextFromContentUnion(contentUnion: unknown): string {
    if (typeof contentUnion === 'string') {
      return contentUnion;
    }

    if (Array.isArray(contentUnion)) {
      return contentUnion
        .map((item) => this.extractTextFromContentUnion(item))
        .filter(Boolean)
        .join('\n');
    }

    if (typeof contentUnion === 'object' && contentUnion !== null) {
      if ('parts' in contentUnion) {
        const content = contentUnion as Content;
        return (
          content.parts
            ?.map((part: Part) => {
              if (typeof part === 'string') return part;
              if ('text' in part) return part.text || '';
              return '';
            })
            .filter(Boolean)
            .join('\n') || ''
        );
      }
    }

    return '';
  }

  /**
   * Clean up orphaned tool calls (tool calls without corresponding tool responses)
   */
  private cleanOrphanedToolCalls(
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const cleaned: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    const pendingToolCalls = new Set<string>();

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];

      if (message.role === 'assistant' && message.tool_calls) {
        // Track pending tool calls
        for (const toolCall of message.tool_calls) {
          pendingToolCalls.add(toolCall.id);
        }
        cleaned.push(message);
      } else if (message.role === 'tool') {
        // Remove from pending when we get a tool response
        if (message.tool_call_id) {
          pendingToolCalls.delete(message.tool_call_id);
        }
        cleaned.push(message);
      } else if (message.role === 'user' || message.role === 'system') {
        // For now, don't automatically add tool responses
        // This can cause issues if the conversation flow is already correct
        cleaned.push(message);
      } else {
        // Other message types
        cleaned.push(message);
      }
    }

    // For now, don't automatically add tool responses at the end
    // This can cause issues if the conversation flow is already correct
    if (pendingToolCalls.size > 0 && process.env['DEBUG_OPENAI_CONVERSATION']) {
      console.warn(
        `Found ${pendingToolCalls.size} pending tool calls at end of conversation: ${Array.from(pendingToolCalls).join(', ')}`,
      );
    }

    return cleaned;
  }

  /**
   * Merge consecutive assistant messages
   */
  private mergeConsecutiveAssistantMessages(
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const merged: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    for (const message of messages) {
      const lastMessage = merged[merged.length - 1];

      if (
        message.role === 'assistant' &&
        lastMessage?.role === 'assistant' &&
        !message.tool_calls &&
        !lastMessage.tool_calls
      ) {
        // Merge text content
        const lastContent = lastMessage.content;
        const currentContent = message.content;

        if (
          typeof lastContent === 'string' &&
          typeof currentContent === 'string'
        ) {
          lastMessage.content = lastContent + currentContent;
        } else if (
          Array.isArray(lastContent) &&
          Array.isArray(currentContent)
        ) {
          lastMessage.content = [...lastContent, ...currentContent];
        } else {
          merged.push(message);
        }
      } else {
        merged.push(message);
      }
    }

    return merged;
  }

  /**
   * Validate conversation flow to ensure tool calls are properly followed by tool responses
   */
  private validateConversationFlow(
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): void {
    const pendingToolCalls = new Set<string>();

    if (process.env['DEBUG_OPENAI_CONVERSATION']) {
      console.log(
        'Validating conversation flow with',
        messages.length,
        'messages',
      );
    }

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];

      if (process.env['DEBUG_OPENAI_CONVERSATION']) {
        console.log(`Message ${i}:`, {
          role: message.role,
          hasToolCalls: !!(message as any).tool_calls?.length,
          toolCallIds:
            (message as any).tool_calls?.map((tc: any) => tc.id) || [],
          toolCallId: (message as any).tool_call_id,
          pendingToolCalls: Array.from(pendingToolCalls),
        });
      }

      if (message.role === 'assistant' && (message as any).tool_calls) {
        // Track pending tool calls
        for (const toolCall of (message as any).tool_calls) {
          pendingToolCalls.add(toolCall.id);
        }
      } else if (message.role === 'tool') {
        // Remove from pending when we get a tool response
        if ((message as any).tool_call_id) {
          pendingToolCalls.delete((message as any).tool_call_id);
        }
      } else if (message.role === 'user' || message.role === 'system') {
        // If we encounter a user/system message while there are pending tool calls,
        // this indicates a conversation flow issue
        if (pendingToolCalls.size > 0) {
          console.warn(
            `Conversation flow issue: Found ${pendingToolCalls.size} pending tool calls when encountering ${message.role} message. Tool call IDs: ${Array.from(pendingToolCalls).join(', ')}`,
          );
        }
      }
    }

    // Check for any remaining pending tool calls at the end
    if (pendingToolCalls.size > 0) {
      console.warn(
        `Conversation flow issue: Found ${pendingToolCalls.size} pending tool calls at the end of conversation. Tool call IDs: ${Array.from(pendingToolCalls).join(', ')}`,
      );
    }
  }

  /**
   * Convert OpenAI response to Gemini format
   */
  convertOpenAIResponseToGemini(
    openaiResponse: OpenAI.Chat.ChatCompletion,
  ): GenerateContentResponse {
    const choice = openaiResponse.choices[0];
    const response = new GenerateContentResponse();

    const parts: Part[] = [];

    // Handle text content
    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    // Handle tool calls
    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        if (toolCall.function) {
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
    }

    response.responseId = openaiResponse.id;
    response.createTime = openaiResponse.created
      ? openaiResponse.created.toString()
      : new Date().getTime().toString();

    response.candidates = [
      {
        content: {
          parts,
          role: 'model' as const,
        },
        finishReason: this.mapOpenAIFinishReasonToGemini(
          choice.finish_reason || 'stop',
        ),
        index: 0,
        safetyRatings: [],
      },
    ];

    response.modelVersion = this.model;
    response.promptFeedback = { safetyRatings: [] };

    // Add usage metadata if available
    if (openaiResponse.usage) {
      const usage = openaiResponse.usage;

      const promptTokens = usage.prompt_tokens || 0;
      const completionTokens = usage.completion_tokens || 0;
      const totalTokens = usage.total_tokens || 0;

      // If we only have total tokens but no breakdown, estimate the split
      let finalPromptTokens = promptTokens;
      let finalCompletionTokens = completionTokens;

      if (totalTokens > 0 && promptTokens === 0 && completionTokens === 0) {
        // Estimate: assume 70% input, 30% output
        finalPromptTokens = Math.round(totalTokens * 0.7);
        finalCompletionTokens = Math.round(totalTokens * 0.3);
      }

      response.usageMetadata = {
        promptTokenCount: finalPromptTokens,
        candidatesTokenCount: finalCompletionTokens,
        totalTokenCount: totalTokens,
      };
    }

    return response;
  }

  /**
   * Convert OpenAI stream chunk to Gemini format
   */
  convertOpenAIChunkToGemini(
    chunk: OpenAI.Chat.ChatCompletionChunk,
  ): GenerateContentResponse {
    const choice = chunk.choices?.[0];
    const response = new GenerateContentResponse();

    if (choice) {
      const parts: Part[] = [];

      // Handle text content
      if (choice.delta?.content) {
        if (typeof choice.delta.content === 'string') {
          parts.push({ text: choice.delta.content });
        }
      }

      // Handle tool calls using the streaming parser
      if (choice.delta?.tool_calls) {
        for (const toolCall of choice.delta.tool_calls) {
          const index = toolCall.index ?? 0;

          // Process the tool call chunk through the streaming parser
          if (toolCall.function?.arguments) {
            this.streamingToolCallParser.addChunk(
              index,
              toolCall.function.arguments,
              toolCall.id,
              toolCall.function.name,
            );
          } else {
            // Handle metadata-only chunks (id and/or name without arguments)
            this.streamingToolCallParser.addChunk(
              index,
              '', // Empty chunk for metadata-only updates
              toolCall.id,
              toolCall.function?.name,
            );
          }
        }
      }

      // Only emit function calls when streaming is complete (finish_reason is present)
      if (choice.finish_reason) {
        const completedToolCalls =
          this.streamingToolCallParser.getCompletedToolCalls();

        for (const toolCall of completedToolCalls) {
          if (toolCall.name) {
            parts.push({
              functionCall: {
                id:
                  toolCall.id ||
                  `call_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
                name: toolCall.name,
                args: toolCall.args,
              },
            });
          }
        }

        // Clear the parser for the next stream
        this.streamingToolCallParser.reset();
      }

      // Only include finishReason key if finish_reason is present
      const candidate: Candidate = {
        content: {
          parts,
          role: 'model' as const,
        },
        index: 0,
        safetyRatings: [],
      };
      if (choice.finish_reason) {
        candidate.finishReason = this.mapOpenAIFinishReasonToGemini(
          choice.finish_reason,
        );
      }
      response.candidates = [candidate];
    } else {
      response.candidates = [];
    }

    response.responseId = chunk.id;
    response.createTime = chunk.created
      ? chunk.created.toString()
      : new Date().getTime().toString();

    response.modelVersion = this.model;
    response.promptFeedback = { safetyRatings: [] };

    // Add usage metadata if available in the chunk
    if (chunk.usage) {
      const usage = chunk.usage;

      const promptTokens = usage.prompt_tokens || 0;
      const completionTokens = usage.completion_tokens || 0;
      const totalTokens = usage.total_tokens || 0;

      // If we only have total tokens but no breakdown, estimate the split
      let finalPromptTokens = promptTokens;
      let finalCompletionTokens = completionTokens;

      if (totalTokens > 0 && promptTokens === 0 && completionTokens === 0) {
        // Estimate: assume 70% input, 30% output
        finalPromptTokens = Math.round(totalTokens * 0.7);
        finalCompletionTokens = Math.round(totalTokens * 0.3);
      }

      response.usageMetadata = {
        promptTokenCount: finalPromptTokens,
        candidatesTokenCount: finalCompletionTokens,
        totalTokenCount: totalTokens,
      };
    }

    return response;
  }

  /**
   * Map OpenAI finish reason to Gemini finish reason
   */
  private mapOpenAIFinishReasonToGemini(
    openaiReason: string | null,
  ): FinishReason {
    switch (openaiReason) {
      case 'stop':
        return FinishReason.STOP;
      case 'length':
        return FinishReason.MAX_TOKENS;
      case 'content_filter':
        return FinishReason.SAFETY;
      case 'tool_calls':
        return FinishReason.STOP; // Tool calls are considered a normal stop
      default:
        return FinishReason.FINISH_REASON_UNSPECIFIED;
    }
  }
}
