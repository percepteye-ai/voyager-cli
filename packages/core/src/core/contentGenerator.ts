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
import { GoogleGenAI } from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import type { Config } from '../config/config.js';

import type { UserTierId } from '../code_assist/types.js';
import { LoggingContentGenerator } from './loggingContentGenerator.js';
import { InstallationManager } from '../utils/installationManager.js';
import { OpenAIContentGenerator } from './openaiContentGenerator.js';
import { AnthropicContentGenerator } from './anthropicContentGenerator.js';
import { GoogleGenAIWrapper } from './googleGenAIWrapper.js';
import { ApiContentGenerator } from './apiContentGenerator.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  getModel(): string;

  userTier?: UserTierId;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  CLOUD_SHELL = 'cloud-shell',
  USE_OPENAI = 'openai-api-key',
  USE_ANTHROPIC = 'anthropic-api-key',
  USE_API = 'api',
}

export type ContentGeneratorConfig = {
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType;
  proxy?: string;
  openaiApiKey?: string;
  openaiModel?: string;
  anthropicApiKey?: string;
  anthropicModel?: string;
  // API-based configuration
  apiEndpoint?: string;
  apiAuthToken?: string;
  apiModel?: string;
  // configMode is now fetched from Supabase, no longer needed in config
};

export function createContentGeneratorConfig(
  config: Config,
  authType: AuthType | undefined,
  settings?: { model?: { selectedModel?: string } },
): ContentGeneratorConfig {
  const geminiApiKey = process.env['GEMINI_API_KEY'] || undefined;
  const googleApiKey = process.env['GOOGLE_API_KEY'] || undefined;
  const googleCloudProject = process.env['GOOGLE_CLOUD_PROJECT'] || undefined;
  const googleCloudLocation = process.env['GOOGLE_CLOUD_LOCATION'] || undefined;
  const openaiApiKey = process.env['OPENAI_API_KEY'] || undefined;
  const openaiModel =
    settings?.model?.selectedModel || process.env['OPENAI_MODEL'] || 'gpt-4o';
  const anthropicApiKey = process.env['ANTHROPIC_API_KEY'] || undefined;
  const anthropicModel =
    settings?.model?.selectedModel ||
    process.env['ANTHROPIC_MODEL'] ||
    'claude-3-5-sonnet-20241022';

  // API-based configuration
  const apiEndpoint = process.env['API_ENDPOINT'] || undefined;
  const apiAuthToken = process.env['API_AUTH_TOKEN'] || undefined;
  const apiModel =
    settings?.model?.selectedModel || process.env['API_MODEL'] || 'gpt-4o';

  const contentGeneratorConfig: ContentGeneratorConfig = {
    authType,
    proxy: config?.getProxy(),
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.CLOUD_SHELL
  ) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;

    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_OPENAI && openaiApiKey) {
    contentGeneratorConfig.openaiApiKey = openaiApiKey;
    contentGeneratorConfig.openaiModel = openaiModel;

    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_ANTHROPIC && anthropicApiKey) {
    contentGeneratorConfig.anthropicApiKey = anthropicApiKey;
    contentGeneratorConfig.anthropicModel = anthropicModel;

    return contentGeneratorConfig;
  }

  // Handle unified API authentication type
  if (authType === AuthType.USE_API && apiEndpoint && apiAuthToken) {
    contentGeneratorConfig.apiEndpoint = apiEndpoint;
    contentGeneratorConfig.apiAuthToken = apiAuthToken;
    contentGeneratorConfig.apiModel = apiModel;
    // configMode will be fetched from Supabase at runtime

    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const version = process.env['CLI_VERSION'] || process.version;
  const userAgent = `GeminiCLI/${version} (${process.platform}; ${process.arch})`;
  const baseHeaders: Record<string, string> = {
    'User-Agent': userAgent,
  };

  if (
    config.authType === AuthType.LOGIN_WITH_GOOGLE ||
    config.authType === AuthType.CLOUD_SHELL
  ) {
    const httpOptions = { headers: baseHeaders };
    return new LoggingContentGenerator(
      await createCodeAssistContentGenerator(
        httpOptions,
        config.authType,
        gcConfig,
        sessionId,
      ),
      gcConfig,
    );
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    let headers: Record<string, string> = { ...baseHeaders };
    if (gcConfig?.getUsageStatisticsEnabled()) {
      const installationManager = new InstallationManager();
      const installationId = installationManager.getInstallationId();
      headers = {
        ...headers,
        'x-gemini-api-privileged-user-id': `${installationId}`,
      };
    }
    const httpOptions = { headers };

    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
    });

    // Determine the model name based on auth type and config
    const modelName = config.vertexai ? 'vertex-ai' : 'gemini-2.5-pro';
    const wrapper = new GoogleGenAIWrapper(googleGenAI.models, modelName);
    return new LoggingContentGenerator(wrapper, gcConfig);
  }

  if (config.authType === AuthType.USE_OPENAI) {
    if (!config.openaiApiKey) {
      throw new Error('OpenAI API key is required for OpenAI authentication');
    }
    const openaiGenerator = new OpenAIContentGenerator(
      config.openaiApiKey,
      config.openaiModel || 'gpt-4o',
      gcConfig,
    );
    return new LoggingContentGenerator(openaiGenerator, gcConfig);
  }

  if (config.authType === AuthType.USE_ANTHROPIC) {
    if (!config.anthropicApiKey) {
      throw new Error(
        'Anthropic API key is required for Anthropic authentication',
      );
    }
    const anthropicGenerator = new AnthropicContentGenerator(
      config.anthropicApiKey,
      config.anthropicModel || 'claude-3-5-sonnet-20241022',
      gcConfig,
    );
    return new LoggingContentGenerator(anthropicGenerator, gcConfig);
  }

  // Handle unified API-based content generator
  if (config.authType === AuthType.USE_API) {
    if (!config.apiEndpoint || !config.apiAuthToken) {
      throw new Error(
        'API endpoint and auth token are required for API-based authentication',
      );
    }

    // Backend fetches configMode from Supabase, no need to fetch it here
    const apiGenerator = new ApiContentGenerator(
      config.apiEndpoint,
      config.apiAuthToken,
      config.apiModel || 'gpt-4o',
      gcConfig,
    );
    return new LoggingContentGenerator(apiGenerator, gcConfig);
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
  );
}
