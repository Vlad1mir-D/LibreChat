const axios = require('axios');
const { Providers } = require('@librechat/agents');
const { HttpsProxyAgent } = require('https-proxy-agent');
const { EModelEndpoint, defaultModels, CacheKeys } = require('librechat-data-provider');
const { inputSchema, logAxiosError, extractBaseURL, processModelData } = require('~/utils');
const { OllamaClient } = require('~/app/clients/OllamaClient');
const { isUserProvided } = require('~/server/utils');
const getLogStores = require('~/cache/getLogStores');
const { logger } = require('~/config');

/**
 * Splits a string by commas and trims each resulting value.
 * @param {string} input - The input string to split.
 * @returns {string[]} An array of trimmed values.
 */
const splitAndTrim = (input) => {
  if (!input || typeof input !== 'string') {
    return [];
  }
  return input
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
};

const { openAIApiKey, userProvidedOpenAI } = require('./Config/EndpointService').config;

/**
 * Fetches OpenAI models from the specified base API path or Azure, based on the provided configuration.
 *
 * @param {Object} params - The parameters for fetching the models.
 * @param {Object} params.user - The user ID to send to the API.
 * @param {string} params.apiKey - The API key for authentication with the API.
 * @param {string} params.baseURL - The base path URL for the API.
 * @param {string} [params.name='OpenAI'] - The name of the API; defaults to 'OpenAI'.
 * @param {boolean} [params.azure=false] - Whether to fetch models from Azure.
 * @param {boolean} [params.userIdQuery=false] - Whether to send the user ID as a query parameter.
 * @param {boolean} [params.createTokenConfig=true] - Whether to create a token configuration from the API response.
 * @param {string} [params.tokenKey] - The cache key to save the token configuration. Uses `name` if omitted.
 * @returns {Promise<string[]>} A promise that resolves to an array of model identifiers.
 * @async
 */
const fetchModels = async ({
  user,
  apiKey,
  baseURL,
  name = EModelEndpoint.openAI,
  azure = false,
  userIdQuery = false,
  createTokenConfig = true,
  tokenKey,
}) => {
  let models = [];

  if (!baseURL && !azure) {
    return models;
  }

  if (!apiKey) {
    return models;
  }

  if (name && name.toLowerCase().startsWith(Providers.OLLAMA)) {
    return await OllamaClient.fetchModels(baseURL);
  }

  try {
    const options = {
      headers: {},
      timeout: 5000,
    };

    if (name === EModelEndpoint.anthropic) {
      options.headers = {
        'x-api-key': apiKey,
        'anthropic-version': process.env.ANTHROPIC_VERSION || '2023-06-01',
      };
    } else {
      options.headers.Authorization = `Bearer ${apiKey}`;
    }

    if (process.env.PROXY) {
      options.httpsAgent = new HttpsProxyAgent(process.env.PROXY);
    }

    if (process.env.OPENAI_ORGANIZATION && baseURL.includes('openai')) {
      options.headers['OpenAI-Organization'] = process.env.OPENAI_ORGANIZATION;
    }

    const url = new URL(`${baseURL}${azure ? '' : '/models'}`);
    if (user && userIdQuery) {
      url.searchParams.append('user', user);
    }
    const res = await axios.get(url.toString(), options);

    /** @type {z.infer<typeof inputSchema>} */
    const input = res.data;

    const validationResult = inputSchema.safeParse(input);
    if (validationResult.success && createTokenConfig) {
      const endpointTokenConfig = processModelData(input);
      const cache = getLogStores(CacheKeys.TOKEN_CONFIG);
      await cache.set(tokenKey ?? name, endpointTokenConfig);
    }
    models = input.data.map((item) => item.id);
  } catch (error) {
    const logMessage = `Failed to fetch models from ${azure ? 'Azure ' : ''}${name} API`;
    logAxiosError({ message: logMessage, error });
  }

  return models;
};

/**
 * Fetches models from the specified API path or Azure, based on the provided options.
 * @async
 * @function
 * @param {object} opts - The options for fetching the models.
 * @param {string} opts.user - The user ID to send to the API.
 * @param {boolean} [opts.azure=false] - Whether to fetch models from Azure.
 * @param {boolean} [opts.assistants=false] - Whether to fetch models from Azure.
 * @param {boolean} [opts.plugins=false] - Whether to fetch models from the plugins.
 * @param {string[]} [_models=[]] - The models to use as a fallback.
 */
const fetchOpenAIModels = async (opts, _models = []) => {
  let models = _models.slice() ?? [];
  let apiKey = openAIApiKey;
  const openaiBaseURL = 'https://api.openai.com/v1';
  let baseURL = openaiBaseURL;
  let reverseProxyUrl = process.env.OPENAI_REVERSE_PROXY;

  if (opts.assistants && process.env.ASSISTANTS_BASE_URL) {
    reverseProxyUrl = process.env.ASSISTANTS_BASE_URL;
  } else if (opts.azure) {
    return models;
    // const azure = getAzureCredentials();
    // baseURL = (genAzureChatCompletion(azure))
    //   .split('/deployments')[0]
    //   .concat(`/models?api-version=${azure.azureOpenAIApiVersion}`);
    // apiKey = azureOpenAIApiKey;
  }

  if (reverseProxyUrl) {
    baseURL = extractBaseURL(reverseProxyUrl);
  }

  const modelsCache = getLogStores(CacheKeys.MODEL_QUERIES);

  const cachedModels = await modelsCache.get(baseURL);
  if (cachedModels) {
    return cachedModels;
  }

  if (baseURL || opts.azure) {
    models = await fetchModels({
      apiKey,
      baseURL,
      azure: opts.azure,
      user: opts.user,
      name: EModelEndpoint.openAI,
    });
  }

  if (models.length === 0) {
    return _models;
  }

  if (baseURL === openaiBaseURL) {
    const regex = /(text-davinci-003|gpt-|o\d+)/;
    const excludeRegex = /audio|realtime/;
    models = models.filter((model) => regex.test(model) && !excludeRegex.test(model));
    const instructModels = models.filter((model) => model.includes('instruct'));
    const otherModels = models.filter((model) => !model.includes('instruct'));
    models = otherModels.concat(instructModels);
  }

  await modelsCache.set(baseURL, models);
  return models;
};

/**
 * Loads the default models for the application.
 * @async
 * @function
 * @param {object} opts - The options for fetching the models.
 * @param {string} opts.user - The user ID to send to the API.
 * @param {boolean} [opts.azure=false] - Whether to fetch models from Azure.
 * @param {boolean} [opts.plugins=false] - Whether to fetch models for the plugins endpoint.
 * @param {boolean} [opts.assistants=false] - Whether to fetch models for the Assistants endpoint.
 */
const getOpenAIModels = async (opts) => {
  let models = defaultModels[EModelEndpoint.openAI];

  if (opts.assistants) {
    models = defaultModels[EModelEndpoint.assistants];
  } else if (opts.azure) {
    models = defaultModels[EModelEndpoint.azureAssistants];
  }

  if (opts.plugins) {
    models = models.filter(
      (model) =>
        !model.includes('text-davinci') &&
        !model.includes('instruct') &&
        !model.includes('0613') &&
        !model.includes('0314') &&
        !model.includes('0301'),
    );
  }

  let key;
  if (opts.assistants) {
    key = 'ASSISTANTS_MODELS';
  } else if (opts.azure) {
    key = 'AZURE_OPENAI_MODELS';
  } else if (opts.plugins) {
    key = 'PLUGIN_MODELS';
  } else {
    key = 'OPENAI_MODELS';
  }

  if (process.env[key]) {
    models = splitAndTrim(process.env[key]);
    return models;
  }

  // If using a sentinel "user_provided" API key, attempt to fetch models with the user-supplied key
  if (userProvidedOpenAI) {
    try {
      const { getUserKeyValues } = require('~/server/services/UserService');
      const userValues = await getUserKeyValues({ userId: opts.user, name: EModelEndpoint.openAI });
      const userApiKey = userValues?.apiKey;
      if (userApiKey) {
        // Determine effective base URL
        const openaiBaseURL = 'https://api.openai.com/v1';
        let baseURL = openaiBaseURL;
        let reverseProxyUrl = process.env.OPENAI_REVERSE_PROXY;
        if (opts.assistants && process.env.ASSISTANTS_BASE_URL) {
          reverseProxyUrl = process.env.ASSISTANTS_BASE_URL;
        } else if (opts.azure) {
          return models;
        }
        if (reverseProxyUrl) {
          baseURL = extractBaseURL(reverseProxyUrl);
        }
        // Check cache first
        const modelsCache = getLogStores(CacheKeys.MODEL_QUERIES);
        const cached = await modelsCache.get(baseURL);
        if (cached) {
          return cached;
        }
        // Fetch models using the user-provided key
        let fetched = await fetchModels({
          user: opts.user,
          apiKey: userApiKey,
          baseURL,
          name: EModelEndpoint.openAI,
        });
        // Fallback to defaults if none fetched
        if (!fetched?.length) {
          return models;
        }
        // Apply OpenAI model filters when using the official API
        if (baseURL === openaiBaseURL) {
          const regex = /(text-davinci-003|gpt-|o\d+)/;
          const exclude = /audio|realtime/;
          fetched = fetched.filter((m) => regex.test(m) && !exclude.test(m));
          const instruct = fetched.filter((m) => m.includes('instruct'));
          const other = fetched.filter((m) => !m.includes('instruct'));
          fetched = other.concat(instruct);
        }
        // Apply plugin filters if requested
        if (opts.plugins) {
          fetched = fetched.filter(
            (m) =>
              !m.includes('text-davinci') &&
              !m.includes('instruct') &&
              !m.includes('0613') &&
              !m.includes('0314') &&
              !m.includes('0301'),
          );
        }
        // Cache and return
        await modelsCache.set(baseURL, fetched);
        return fetched;
      }
    } catch (err) {
      // ignore and fallback to defaults
    }
    return models;
  }

  return await fetchOpenAIModels(opts, models);
};

const getChatGPTBrowserModels = () => {
  let models = ['text-davinci-002-render-sha', 'gpt-4'];
  if (process.env.CHATGPT_MODELS) {
    models = splitAndTrim(process.env.CHATGPT_MODELS);
  }

  return models;
};

/**
 * Fetches models from the Anthropic API.
 * @async
 * @function
 * @param {object} opts - The options for fetching the models.
 * @param {string} opts.user - The user ID to send to the API.
 * @param {string[]} [_models=[]] - The models to use as a fallback.
 */
const fetchAnthropicModels = async (opts, _models = []) => {
  let models = _models.slice() ?? [];
  let apiKey = process.env.ANTHROPIC_API_KEY;
  const anthropicBaseURL = 'https://api.anthropic.com/v1';
  let baseURL = anthropicBaseURL;
  let reverseProxyUrl = process.env.ANTHROPIC_REVERSE_PROXY;

  if (reverseProxyUrl) {
    baseURL = extractBaseURL(reverseProxyUrl);
  }

  if (!apiKey) {
    return models;
  }

  const modelsCache = getLogStores(CacheKeys.MODEL_QUERIES);

  const cachedModels = await modelsCache.get(baseURL);
  if (cachedModels) {
    return cachedModels;
  }

  if (baseURL) {
    models = await fetchModels({
      apiKey,
      baseURL,
      user: opts.user,
      name: EModelEndpoint.anthropic,
      tokenKey: EModelEndpoint.anthropic,
    });
  }

  if (models.length === 0) {
    return _models;
  }

  await modelsCache.set(baseURL, models);
  return models;
};

const getAnthropicModels = async (opts = {}) => {
  let models = defaultModels[EModelEndpoint.anthropic];
  if (process.env.ANTHROPIC_MODELS) {
    models = splitAndTrim(process.env.ANTHROPIC_MODELS);
    return models;
  }

  // If using a sentinel "user_provided" Anthropic key, attempt to fetch models with the user-supplied key
  if (isUserProvided(process.env.ANTHROPIC_API_KEY)) {
    // Attempt to use the user-supplied Anthropic key from UserService
    let userApiKey;
    const userSvc = require('~/server/services/UserService');
    try {
      const userValues = await userSvc.getUserKeyValues({ userId: opts.user, name: EModelEndpoint.anthropic });
      userApiKey = userValues?.apiKey;
    } catch {
      try {
        userApiKey = await userSvc.getUserKey({ userId: opts.user, name: EModelEndpoint.anthropic });
      } catch {
        // No valid user key available
        return models;
      }
    }
    if (userApiKey) {
      const anthroBase = 'https://api.anthropic.com/v1';
      let baseURL = anthroBase;
      const reverseProxy = process.env.ANTHROPIC_REVERSE_PROXY;
      if (reverseProxy) {
        baseURL = extractBaseURL(reverseProxy);
      }
      const modelsCache = getLogStores(CacheKeys.MODEL_QUERIES);
      const cached = await modelsCache.get(baseURL);
      if (cached) {
        return cached;
      }
      const fetched = await fetchModels({
        user: opts.user,
        apiKey: userApiKey,
        baseURL,
        name: EModelEndpoint.anthropic,
        tokenKey: EModelEndpoint.anthropic,
      });
      const result = fetched && fetched.length ? fetched : models;
      await modelsCache.set(baseURL, result);
      return result;
    }
    return models;
  }

  try {
    return await fetchAnthropicModels(opts, models);
  } catch (error) {
    logger.error('Error fetching Anthropic models:', error);
    return models;
  }
};

/**
 * Retrieves Google/Gemini models, supporting direct or user_provided API keys.
 * @param {object} opts - Options including user ID.
 * @returns {string[]|Promise<string[]>} List of model identifiers or a promise resolving to the list.
 */
const getGoogleModels = (opts = {}) => {
  const models = defaultModels[EModelEndpoint.google];
  // ENV override for explicit model list
  if (process.env.GOOGLE_MODELS) {
    return splitAndTrim(process.env.GOOGLE_MODELS);
  }
  const envKey = process.env.GOOGLE_KEY;
  const reverseProxy = process.env.GOOGLE_REVERSE_PROXY;
  // If an API key is provided (directly or via sentinel), fetch dynamically
  if (envKey) {
    return (async () => {
      try {
        let apiKey = envKey;
        if (isUserProvided(envKey)) {
          const { getUserKeyValues } = require('~/server/services/UserService');
          const userValues = await getUserKeyValues({ userId: opts.user, name: EModelEndpoint.google });
          apiKey = userValues?.apiKey;
        }
        if (!apiKey) return models;
        // Build models URL
        let modelsUrl = 'https://generativelanguage.googleapis.com/v1/models';
        if (reverseProxy) {
          modelsUrl = extractBaseURL(reverseProxy).replace(/\/$/, '') + '/models';
        }
        const cache = getLogStores(CacheKeys.MODEL_QUERIES);
        const cached = await cache.get(modelsUrl);
        if (cached) return cached;
        const urlObj = new URL(modelsUrl);
        urlObj.searchParams.set('key', apiKey);
        const res = await axios.get(urlObj.toString(), { timeout: 5000 });
        const data = res.data;
        const fetched = Array.isArray(data.models)
          ? data.models.map((m) => {
              const nm = m.name || m.model || '';
              return nm.includes('/') ? nm.split('/').pop() : nm;
            })
          : [];
        await cache.set(modelsUrl, fetched);
        return fetched.length ? fetched : models;
      } catch {
        return models;
      }
    })();
  }
  return models;
};

const getBedrockModels = () => {
  let models = defaultModels[EModelEndpoint.bedrock];
  if (process.env.BEDROCK_AWS_MODELS) {
    models = splitAndTrim(process.env.BEDROCK_AWS_MODELS);
  }

  return models;
};

module.exports = {
  fetchModels,
  splitAndTrim,
  getOpenAIModels,
  getBedrockModels,
  getChatGPTBrowserModels,
  getAnthropicModels,
  getGoogleModels,
};
