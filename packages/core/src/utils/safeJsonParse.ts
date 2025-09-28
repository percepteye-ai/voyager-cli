/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Safely parse JSON string with fallback for malformed JSON.
 * This function attempts to parse JSON normally first, and if that fails,
 * it returns the fallback value.
 *
 * @param jsonString - The JSON string to parse
 * @param fallbackValue - The value to return if parsing fails completely
 * @returns The parsed object or the fallback value
 */
export function safeJsonParse<T = Record<string, unknown>>(
  jsonString: string,
  fallbackValue: T = {} as T,
): T {
  if (!jsonString || typeof jsonString !== 'string') {
    return fallbackValue;
  }

  try {
    // First attempt: try normal JSON.parse
    return JSON.parse(jsonString) as T;
  } catch (error) {
    console.error('Failed to parse JSON:', {
      originalError: error,
      jsonString,
    });
    return fallbackValue;
  }
}
