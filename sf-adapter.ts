import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

// 1. 目标 API URL 修改为 SiliconFlow
const SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/images/generations";

// OpenAI 客户端请求的接口定义 (保持不变)
interface OpenAIRequest {
  model: string;
  messages: Array<{
    role: string;
    content: string | Array<{ type: 'text', text: string }>; // 兼容不同格式的 content
  }>;
  n?: number;
  size?: string;
  seed?: number;
  negative_prompt?: string;
  num_inference_steps?: number;
  guidance_scale?: number;
}

interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      tool_calls?: Array<{
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        }
      }>
    };
    finish_reason: string;
  }>;
}

async function handleRequest(req: Request): Promise<Response> {
  const url = new URL(req.url);

  if (url.pathname !== "/v1/chat/completions" && url.pathname !== "/v1/images/generations") {
    return new Response(JSON.stringify({ error: "Not found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    const apiKey = req.headers.get("Authorization");
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "Authorization header is required" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }
    const openAIRequest: OpenAIRequest = await req.json();
    let promptText = "";
    const lastMessage = openAIRequest.messages[openAIRequest.messages.length - 1];
    if (lastMessage && lastMessage.role === "user") {
        if (typeof lastMessage.content === 'string') {
            promptText = lastMessage.content;
        } else if (Array.isArray(lastMessage.content)) {
            const textContent = lastMessage.content.find(item => item.type === 'text');
            if (textContent) {
                promptText = textContent.text;
            }
        }
    }
    if (!promptText) {
        return new Response(JSON.stringify({ error: "No user prompt found in messages" }), {
            status: 400,
            headers: { "Content-Type": "application/json" },
        });
    }
    const siliconflowRequest = {
      model: openAIRequest.model || "Qwen/Qwen-Image",
      prompt: promptText,
      image_size: openAIRequest.size || "1328x1328",
      batch_size: openAIRequest.n || 1,
      seed: openAIRequest.seed,
      negative_prompt: openAIRequest.negative_prompt,
      num_inference_steps: openAIRequest.num_inference_steps,
      guidance_scale: openAIRequest.guidance_scale,
    };
    const sfResponse = await fetch(SILICONFLOW_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": apiKey,
      },
      body: JSON.stringify(siliconflowRequest),
    });
    if (!sfResponse.ok) {
      const errorText = await sfResponse.text();
      console.error("SiliconFlow API Error:", errorText);
      return new Response(JSON.stringify({ error: `SiliconFlow API Error: ${errorText}` }), {
        status: sfResponse.status,
        headers: { "Content-Type": "application/json" },
      });
    }
    const sfData = await sfResponse.json();
    const imageUrl = sfData.images[0].url;
    const openAIResponse: OpenAIResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: openAIRequest.model || "Qwen/Qwen-Image",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: `✅ 图像已生成！\n\n[点击查看图片](${imageUrl})`,
            tool_calls: [{
                id: `call_${Date.now()}`,
                type: 'function',
                function: {
                  name: 'dalle3',
                  arguments: JSON.stringify({
                    prompts: [promptText],
                    urls: sfData.images.map((img: any) => img.url)
                  }),
                }
            }]
          },
          finish_reason: "tool_calls"
        }
      ],
    };
    return new Response(JSON.stringify(openAIResponse), {
      headers: { 
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
      },
    });
  } catch (error) {
    console.error("Server Error:", error);
    return new Response(JSON.stringify({ error: { message: error.message, type: "server_error" } }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

async function handleOptions(req: Request): Promise<Response> {
  if (req.headers.get("Access-Control-Request-Method") !== null) {
      return new Response(null, {
          status: 204,
          headers: {
              "Access-Control-Allow-Origin": "*",
              "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
              "Access-Control-Allow-Headers": "Content-Type, Authorization",
          },
      });
  }
  return new Response(null, { status: 400 });
}

async function handler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") {
    return handleOptions(req);
  }
  return handleRequest(req);
}

// 读取环境变量中的 PORT，如果不存在则默认为 8000 (方便本地测试)
const port = parseInt(Deno.env.get("PORT") || "8000");

console.log(`🚀 SiliconFlow API Adapter is running on http://localhost:${port}`);
console.log("👉 Point your client (e.g., NextChat) to this address.");

serve(handler, { port });
