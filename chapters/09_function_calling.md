# 第九章：工具的力量 — Function Calling与Agent

## 故事开场：无法行动的AI

Bob让GPT-4帮他查一下北京的天气。模型返回：

```
"北京今天天气晴朗，气温25°C，适合外出活动。"
```

"不错！"Bob很高兴。但他随后问：

```
"帮我查一下北京到上海的机票，并预订最便宜的"
```

模型回答：

```
"抱歉，我无法执行这个操作。我只能生成文本，不能帮你预订机票。"
```

**问题：LLM能说话，但不能行动。**

Function Calling（函数调用）让LLM能够**调用外部工具**，执行真实世界的动作。

## 9.1 Function Calling概述

### 核心思想

```
传统LLM:
User → "查一下天气" → LLM → "北京今天晴天" (只说话)

Function Calling:
User → "查一下天气" → LLM → 调用 get_weather("北京")
                                      ↓
                               外部API返回结果
                                      ↓
LLM → 结合结果 → "北京今天晴天，气温25°C" (真的查了)
```

### 使用场景

1. **API调用**：查询数据库、调用第三方API
2. **实时信息**：天气、股票、新闻
3. **执行动作**：发送邮件、创建日程、控制IoT设备
4. **多步骤任务**：需要多个工具配合

## 9.2 函数定义格式

Function Calling的核心是**清晰的函数定义**（JSON Schema格式）：

```json
{
  "name": "get_weather",
  "description": "获取指定城市的天气信息",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名称，例如：北京、上海"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "温度单位，默认celsius"
      }
    },
    "required": ["city"]
  }
}
```

### 函数调用流程

```
User: "北京天气怎么样？"
         ↓
┌────────────────────────────────────────┐
│ LLM识别需要调用函数                      │
│                                         │
│ 函数名: get_weather                     │
│ 参数:   {"city": "北京"}                 │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 执行函数（外部）                         │
│ get_weather(city="北京")                │
│ 返回: {"temperature": 25, "condition": "晴天"} │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ LLM结合结果生成最终回答                  │
│ "北京今天天气晴朗，气温25°C。"          │
└────────────────────────────────────────┘
```

## 9.3 从零实现Function Calling

```python
# code/inference/function_calling.py
"""
Function Calling 实现
"""

import json
from typing import List, Dict, Callable, Any


class Tool:
    """工具定义"""
    def __init__(self, name: str, description: str, parameters: dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def call(self, arguments: dict) -> Any:
        """调用工具"""
        return self.handler(**arguments)


class FunctionCallingSystem:
    """函数调用系统"""
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool

    def parse_function_call(self, model_output: str) -> Dict:
        """解析模型输出中的函数调用"""
        # 简化版：实际应该用JSON解析
        # 模型应该输出特定格式的JSON
        try:
            # 假设模型输出格式
            # ```json
            # {"name": "get_weather", "arguments": {"city": "北京"}}
            # ```
            start = model_output.find('```json')
            if start != -1:
                end = model_output.find('```', start + 7)
                json_str = model_output[start+7:end]
                return json.loads(json_str)
        except:
            return None

    def execute(self, function_call: Dict) -> Any:
        """执行函数调用"""
        name = function_call.get("name")
        arguments = function_call.get("arguments", {})

        if name not in self.tools:
            return {"error": f"Tool {name} not found"}

        tool = self.tools[name]
        return tool.call(arguments)

    def chat(self, user_message: str, model) -> str:
        """完整的函数调用对话流程"""
        # 1. 让模型决定是否调用函数
        functions_prompt = self._build_functions_prompt()
        response = model.generate(
            user_message,
            system=f"你是一个助手。可以调用以下函数: {functions_prompt}"
        )

        # 2. 检查是否需要函数调用
        function_call = self.parse_function_call(response)

        if function_call:
            # 3. 执行函数
            result = self.execute(function_call)

            # 4. 返回结果给模型生成最终回答
            final_response = model.generate(
                f"用户问: {user_message}\n函数返回: {result}\n请根据函数结果回答用户。"
            )
            return final_response
        else:
            return response

    def _build_functions_prompt(self) -> str:
        """构建函数描述"""
        tools_desc = []
        for name, tool in self.tools.items():
            tools_desc.append(f"- {name}: {tool.description}")
        return "\n".join(tools_desc)


# 示例工具
def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟获取天气"""
    # 实际应该调用天气API
    return {
        "city": city,
        "temperature": 25,
        "unit": unit,
        "condition": "晴天"
    }

def calculate(expression: str) -> dict:
    """计算器"""
    try:
        result = eval(expression)  # 注意：实际使用应该用安全的方式
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

# 测试
system = FunctionCallingSystem()
system.register_tool(Tool("get_weather", "获取城市天气", {}, get_weather))
system.register_tool(Tool("calculate", "计算数学表达式", {}, calculate))

# 模拟对话
user_message = "北京天气怎么样？"
print(f"用户: {user_message}")

# 模型输出（实际应该由LLM生成）
model_output = '''
好的，我需要查询北京的天气。
```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```
'''

# 解析和执行
fc = system.parse_function_call(model_output)
if fc:
    result = system.execute(fc)
    print(f"函数执行结果: {result}")
    print(f"最终回答: 北京今天天气晴朗，气温25°C。")
```

## 9.4 ReAct: 推理+行动

ReAct（Reasoning + Acting）让模型能够**推理下一步行动**：

```
Thought: 我需要知道今天北京的天气
Action: call_weather(city="北京")
Observation: 天气晴朗，25°C
Thought: 根据天气信息，我可以回答用户问题了
Action: response(message="北京今天天气晴朗，25°C")
```

### ReAct循环

```python
def react_loop(query, max_iterations=5):
    """
    ReAct 推理循环

    状态:
    - thought: 思考下一步
    - action: 决定执行什么
    - observation: 观察行动结果
    """
    history = []
    current_query = query

    for i in range(max_iterations):
        # 1. 思考
        thought = model.think(
            f"用户问题: {current_query}\n历史: {history}\n思考下一步该怎么做?"
        )

        # 2. 决定行动
        if "final" in thought.lower():
            # 模型认为已经可以回答了
            return extract_final_answer(thought)

        action = model.decide_action(thought, available_tools)

        # 3. 执行
        result = execute_action(action)

        # 4. 记录
        history.append({
            "thought": thought,
            "action": action,
            "result": result
        })

        # 5. 如果行动是查询信息，模型会继续思考
        current_query = f"根据历史信息继续: {result}"

    return "无法完成"
```

## 9.5 完整的Agent架构

```
                    ┌─────────────────────┐
                    │       Agent         │
                    │   (LLM + 工具)      │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ↓                     ↓                     ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Planner    │    │    Tools     │    │   Memory     │
│  (分解任务)  │    │ (执行动作)   │    │  (存储历史)  │
└──────────────┘    └──────────────┘    └──────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │     Executor        │
                    │   (执行计划)        │
                    └─────────────────────┘
```

### 代码示例

```python
# code/agent/agent.py
"""
简单的Agent实现
"""

from dataclasses import dataclass
from typing import List, Dict, Callable


@dataclass
class Message:
    role: str
    content: str


class SimpleAgent:
    """简单Agent"""
    def __init__(self, llm, tools: List[Callable], max_steps=10):
        self.llm = llm
        self.tools = {t.__name__: t for t in tools}
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        """运行Agent完成任务"""
        messages = [
            Message("system", f"你可以调用这些工具: {list(self.tools.keys())}"),
            Message("user", task)
        ]

        for step in range(self.max_steps):
            # 模型决定下一步
            response = self.llm.chat(messages)

            # 检查是否包含函数调用
            if response.tool_calls:
                # 执行工具
                for call in response.tool_calls:
                    tool_name = call.function.name
                    tool_args = call.function.arguments

                    if tool_name in self.tools:
                        result = self.tools[tool_name](**tool_args)
                        messages.append(Message("assistant", response.content))
                        messages.append(Message("tool", str(result)))
                    else:
                        messages.append(Message("assistant", f"工具 {tool_name} 不存在"))
            else:
                # 模型直接回答
                return response.content

        return "任务未完成"


# 工具定义
def search_web(query: str) -> str:
    """搜索网页"""
    return f"搜索结果: {query}的相关信息..."

def send_email(to: str, content: str) -> str:
    """发送邮件"""
    return f"邮件已发送至 {to}"

def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 使用
agent = SimpleAgent(
    llm=your_llm,
    tools=[search_web, send_email, get_current_time]
)

result = agent.run("帮我查一下今天北京的天气，然后发邮件给alice@example.com告诉她")
```

## 9.6 Function Calling的实际应用

### 1. 数据库查询

```python
# 结构化数据查询
def query_database(sql: str) -> list:
    """执行SQL查询"""
    return db.execute(sql)

# 定义工具
tools = [
    {
        "name": "query_database",
        "description": "执行SQL查询，返回数据",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL查询语句"}
            },
            "required": ["sql"]
        }
    }
]

# 使用
user = "有多少用户注册了？"
# LLM自动生成: query_database(sql="SELECT COUNT(*) FROM users")
```

### 2. 代码执行

```python
def execute_python(code: str) -> str:
    """执行Python代码"""
    try:
        result = eval(code)
        return str(result)
    except Exception as e:
        return f"错误: {e}"

# 用户: "计算 2^10"
# LLM生成: execute_python(code="2**10")
# 返回: "1024"
```

## 9.7 本章小结

1. **Function Calling**：让LLM能调用外部工具执行动作
2. **函数定义**：使用JSON Schema清晰描述工具的输入输出
3. **ReAct**：推理+行动，循环直到完成任务
4. **Agent**：LLM + 工具 + 记忆的完整系统
5. **应用场景**：API调用、实时信息、多步骤任务

### 思考题

1. Function Calling的安全性如何保证？如果模型调用了危险操作怎么办？
2. 如何让模型知道何时应该调用工具，何时应该直接回答？
3. Agent如何处理长时间任务中的状态管理和错误恢复？

### 延伸阅读

- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)