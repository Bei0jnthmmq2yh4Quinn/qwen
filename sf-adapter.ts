import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const ARK_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations";
const SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/images/generations";

const JSON_HEADERS = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS, HEAD",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const AVAILABLE_MODELS = [
  {
    id: "Qwen/Qwen-Image",
    object: "model",
    created: Date.UTC(2024, 0, 1) / 1000,
    owned_by: "siliconflow",
  },
  {
    id: "Kwai-Kolors/Kolors",
    object: "model",
    created: Date.UTC(2024, 0, 1) / 1000,
    owned_by: "siliconflow",
  },
  {
    id: "doubao-seedream-4-0-250828",
    object: "model",
    created: Date.UTC(2024, 0, 1) / 1000,
    owned_by: "ark",
  },
];

type ChatMessage = {
  role: string;
  content: string | Array<Record<string, unknown>>;
};

interface OpenAIRequest {
  model?: string;
  messages: ChatMessage[];
  n?: number;
  size?: string;
  seed?: number;
  stream?: boolean;
  image?: string[];
  [extra: string]: unknown;
}

interface ProviderPayload {
  prompt: string;
  dataUrls: string[];
  httpImages: string[];
}

function detectProvider(model?: string): "ark" | "siliconflow" {
  if (!model) return "ark";
  if (
    model.startsWith("Qwen/") ||
    model.startsWith("Kwai-Kolors/") ||
    model.includes("siliconflow")
  ) {
    return "siliconflow";
  }
  return "ark";
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function coerceMessagePayload(messages: ChatMessage[]): ProviderPayload {
  const payload: ProviderPayload = { prompt: "", dataUrls: [], httpImages: [] };

  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== "user") continue;
    const content = messages[i].content;

    if (typeof content === "string") {
      payload.prompt = content.trim();
    } else if (Array.isArray(content)) {
      for (const chunk of content) {
        const type = (chunk as { type?: unknown }).type;
        if (type === "text" && typeof (chunk as { text?: unknown }).text === "string") {
          payload.prompt = ((chunk as { text: string }).text).trim();
        }
        if (
          type === "image_url" &&
          typeof (chunk as { image_url?: unknown }).image_url === "object" &&
          (chunk as { image_url?: Record<string, unknown> }).image_url &&
          typeof ((chunk as { image_url: { url?: unknown } }).image_url.url) === "string"
        ) {
          const url = (chunk as { image_url: { url: string } }).image_url.url;
          if (url.startsWith("data:")) {
            payload.dataUrls.push(url);
          } else {
            payload.httpImages.push(url);
          }
        }
      }
    }
    if (payload.prompt) break;
  }

  if (!payload.prompt) {
    throw new Error("无法从用户消息中解析出 prompt 文本");
  }

  return payload;
}

function readHeaderApiKey(req: Request): string | undefined {
  const raw = req.headers.get("Authorization");
  if (!raw) return undefined;
  return raw.replace(/^Bearer\s+/i, "").trim();
}

async function callArk(apiKey: string, requestBody: OpenAIRequest, payload: ProviderPayload) {
  const arkModel = requestBody.model ?? "doubao-seedream-4-0-250828";
  const size = typeof requestBody.size === "string" ? requestBody.size : "1024x1024";
  const n = requestBody.n ? clamp(requestBody.n, 1, 4) : undefined;

  const arkRequest: Record<string, unknown> = {
    model: arkModel,
    prompt: payload.prompt,
    image: payload.httpImages,
    sequential_image_generation: n ? "auto" : "disabled",
    response_format: "b64_json",
    size,
    seed: typeof requestBody.seed === "number" ? requestBody.seed : -1,
    stream: false,
    watermark: false,
  };

  if (n) {
    arkRequest.sequential_image_generation_options = { max_images: n };
  }

  const res = await fetch(ARK_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(arkRequest),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Response(JSON.stringify({ error: errorText }), {
      status: res.status,
      headers: JSON_HEADERS,
    });
  }

  const arkData = await res.json();
  const images = Array.isArray(arkData?.data)
    ? arkData.data
        .map((img: Record<string, unknown>) => {
          const url = typeof img?.url === "string"
            ? img.url
            : typeof img?.b64_json === "string"
              ? `data:image/png;base64,${img.b64_json}`
              : undefined;
          if (!url) return undefined;
          return {
            image_url: { url },
            size: typeof img?.size === "string" ? img.size : undefined,
          };
        })
        .filter(Boolean)
    : [];

  return {
    data: images,
    usage: arkData?.usage,
    created: typeof arkData?.created === "number"
      ? arkData.created
      : Math.floor(Date.now() / 1000),
    model: arkModel,
  };
}

async function callSiliconFlow(
  apiKey: string,
  requestBody: OpenAIRequest,
  payload: ProviderPayload,
) {
  const model = requestBody.model ?? "Qwen/Qwen-Image";
  const n = clamp(typeof requestBody.n === "number" ? requestBody.n : 1, 1, 4);
  const body: Record<string, unknown> = {
    model,
    prompt: payload.prompt,
    batch_size: n,
  };

  const overrideSize = typeof requestBody.size === "string"
    ? requestBody.size
    : typeof (requestBody as { image_size?: unknown }).image_size === "string"
      ? (requestBody as { image_size: string }).image_size
      : undefined;
  body.image_size = overrideSize ?? (model.startsWith("Kwai-Kolors/") ? "1024x1024" : "1328x1328");

  const negative = typeof (requestBody as { negative_prompt?: unknown }).negative_prompt === "string"
    ? (requestBody as { negative_prompt: string }).negative_prompt
    : undefined;
  if (negative) body.negative_prompt = negative;

  if (typeof requestBody.seed === "number") {
    body.seed = clamp(requestBody.seed, 0, 9_999_999_999);
  }
  if (typeof (requestBody as { num_inference_steps?: unknown }).num_inference_steps === "number") {
    body.num_inference_steps = clamp(
      (requestBody as { num_inference_steps: number }).num_inference_steps,
      1,
      100,
    );
  }
  if (typeof (requestBody as { guidance_scale?: unknown }).guidance_scale === "number") {
    body.guidance_scale = clamp(
      (requestBody as { guidance_scale: number }).guidance_scale,
      0,
      20,
    );
  }
  if (typeof (requestBody as { cfg?: unknown }).cfg === "number") {
    body.cfg = clamp((requestBody as { cfg: number }).cfg, 0.1, 20);
  }

  if (payload.dataUrls.length > 0) {
    body.image = payload.dataUrls[0];
  }

  const res = await fetch(SILICONFLOW_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Response(JSON.stringify({ error: text }), {
      status: res.status,
      headers: JSON_HEADERS,
    });
  }

  const data = await res.json();
  const images = Array.isArray(data?.images)
    ? data.images
        .map((img: Record<string, unknown>) => {
          const url = typeof img?.url === "string"
            ? img.url
            : typeof img?.b64_json === "string"
              ? `data:image/png;base64,${img.b64_json}`
              : undefined;
          if (!url) return undefined;
          return { image_url: { url } };
        })
        .filter(Boolean)
    : [];

  return {
    data: images,
    usage: undefined,
    created: Math.floor(Date.now() / 1000),
    model,
    timings: data?.timings,
    seed: data?.seed,
  };
}

async function handleChatCompletions(req: Request): Promise<Response> {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: JSON_HEADERS,
    });
  }

  try {
    const requestBody: OpenAIRequest = await req.json();
    if (!Array.isArray(requestBody.messages) || requestBody.messages.length === 0) {
      return new Response(JSON.stringify({ error: "messages 字段不能为空" }), {
        status: 400,
        headers: JSON_HEADERS,
      });
    }

    const payload = coerceMessagePayload(requestBody.messages);
    const provider = detectProvider(requestBody.model);

    const envKey = provider === "ark"
      ? Deno.env.get("ARK_API_KEY")
      : Deno.env.get("SILICONFLOW_API_KEY");
    const headerKey = readHeaderApiKey(req);
    const apiKey = headerKey || envKey;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "缺少 API Key" }), {
        status: 401,
        headers: JSON_HEADERS,
      });
    }

    const result = provider === "ark"
      ? await callArk(apiKey, requestBody, payload)
      : await callSiliconFlow(apiKey, requestBody, payload);

    const openAIResponse = {
      id: `chatcmpl-${crypto.randomUUID()}`,
      object: "chat.completion",
      created: result.created,
      model: result.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "图像生成完成",
            images: result.data,
            metadata: result.timings ? { timings: result.timings, seed: result.seed } : undefined,
          },
          finish_reason: "stop",
        },
      ],
      usage: result.usage,
    };

    return new Response(JSON.stringify(openAIResponse), {
      headers: JSON_HEADERS,
    });
  } catch (err) {
    if (err instanceof Response) return err;
    console.error("Error:", err);
    return new Response(JSON.stringify({
      error: {
        message: err instanceof Error ? err.message : "Unknown error",
        type: "server_error",
      },
    }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
}

async function handleOptions(): Promise<Response> {
  return new Response(null, {
    headers: JSON_HEADERS,
  });
}

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);

  if (req.method === "OPTIONS") {
    return handleOptions();
  }

  if (req.method === "GET" || req.method === "HEAD") {
    if (url.pathname === "/" || url.pathname === "/healthz") {
      return req.method === "HEAD"
        ? new Response(null, { headers: JSON_HEADERS })
        : new Response(JSON.stringify({ status: "ok" }), {
          headers: JSON_HEADERS,
        });
    }

    if (url.pathname === "/v1/chat/completions") {
      return req.method === "HEAD"
        ? new Response(null, { headers: JSON_HEADERS })
        : new Response(JSON.stringify({
          status: "ok",
          message: "Use POST with OpenAI Chat schema to generate images.",
        }), {
          headers: JSON_HEADERS,
        });
    }

    if (url.pathname === "/v1/models") {
      return req.method === "HEAD"
        ? new Response(null, { headers: JSON_HEADERS })
        : new Response(JSON.stringify({
          object: "list",
          data: AVAILABLE_MODELS,
        }), {
          headers: JSON_HEADERS,
        });
    }
  }

  if (url.pathname === "/v1/models") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: JSON_HEADERS,
    });
  }

  if (url.pathname === "/v1/chat/completions") {
    return handleChatCompletions(req);
  }

  return new Response(JSON.stringify({ error: "Not found" }), {
    status: 404,
    headers: JSON_HEADERS,
  });
}

const port = Number(Deno.env.get("PORT") ?? 8000);
console.log(`Server running on http://localhost:${port}`);
serve(handler, { port });
