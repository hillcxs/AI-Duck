from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Iterator

from openai import OpenAI

from robotduck_voice_assistant.state import ChatState, Emotion

EMOTIONS = ["neutral", "fearful", "angry", "sad", "surprised", "happy", "disgusted"]

DIALECTS = [
    "广东话", "东北话", "甘肃话", "贵州话", "河南话", "湖北话", "江西话", "闽南话", "宁夏话",
    "山西话", "陕西话", "山东话", "上海话", "四川话", "天津话", "云南话"
]

SCENES = ["闲聊对话", "比赛解说", "深夜电台广播", "剧情解说", "诗歌朗诵", "科普知识推广", "产品推广", "脱口秀表演"]
ROLES = ["温和客服"]

'''
意图分发器：根据用户输入的文本和当前状态，判断用户的意图（比如是普通问答、模仿音色、视觉问答、方言模式还是角色场景模式），并提取相关参数（情感、方言类型、角色/场景信息等）。
'''

@dataclass
class RouteDecision:
    intent: str  # default|clone|vision|dialect|role_scene|reset|wheel_move|wheel_follow_face|wheel_stop
    emotion: Emotion = "neutral"
    query: str = ""
    dialect: Optional[str] = None
    scene: Optional[str] = None
    role: Optional[str] = None
    style_hint: Optional[str] = None
    move_action: Optional[str] = None
    move_amount: Optional[str] = None
    face_lock: bool = True


def _safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


class IntentDispatcher:
    """根据用户输入和当前状态，调用 LLM 来判断用户意图并提取参数。"""
    # 参数：API 密钥、基础 URL、路由模型、文本模型
    def __init__(self, api_key: str, base_url: str, router_model: str, text_model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.router_model = router_model
        self.text_model = text_model
        self.tools = self._build_tools()

    def _build_tools(self) -> List[Dict[str, Any]]:
        """定义工具（工作流），每个工具对应一种用户意图，包含参数说明。"""
        return [
            {"type": "function", "function": {"name": "workflow_default",
                "description": "默认模式：常规问答/闲聊，没有触发其它模式时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion", "query"]}}},

            {"type": "function", "function": {"name": "workflow_clone",
                "description": "模仿/克隆音色：当用户说‘模仿/学一下xx的音色/克隆音色/用xx的音色说’等时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "query": {"type": "string"},
                        "seconds": {"type": "integer", "default": 7}
                    },
                    "required": ["emotion"]}}},

            {"type": "function", "function": {"name": "workflow_vision",
                "description": "视觉问答：当用户说‘看一下/帮我看一下…’等时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion", "query"]}}},

            {"type": "function", "function": {"name": "workflow_dialect",
                "description": "方言模式：当用户说‘用<方言>和我聊/用<方言>说’等时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "dialect": {"type": "string", "enum": DIALECTS},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion", "dialect"]}}},

            {"type": "function", "function": {"name": "workflow_role_scene",
                "description": "角色/场景模式：‘你是xx/在xx场景/唱rap/脱口秀…’等。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "role": {"type": "string", "enum": ROLES},
                        "scene": {"type": "string", "enum": SCENES},
                        "style_hint": {"type": "string"},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion"]}}},

            {"type": "function", "function": {"name": "workflow_reset",
                "description": "恢复默认：‘换回默认模式/恢复默认音色/别模仿了’等。",
                "parameters": {"type": "object",
                    "properties": {"emotion": {"type": "string", "enum": EMOTIONS}},
                    "required": ["emotion"]}}},

            {"type": "function", "function": {"name": "workflow_wheel_move",
                "description": "轮子移动：当前进、后退、左转、右转、停下等需要控制两驱底盘时使用。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "action": {"type": "string", "enum": ["forward", "backward", "turn_left", "turn_right", "stop"]},
                        "amount": {"type": "string", "enum": ["small", "normal", "large"]},
                        "face_lock": {"type": "boolean", "default": True},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion", "action"]}}},

            {"type": "function", "function": {"name": "workflow_wheel_follow_face",
                "description": "开启底盘人脸跟随，让机器人尽量正面朝向用户并保持合适距离。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion"]}}},

            {"type": "function", "function": {"name": "workflow_wheel_stop",
                "description": "停止底盘移动，并退出自动跟随模式。",
                "parameters": {"type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": EMOTIONS},
                        "query": {"type": "string"}
                    },
                    "required": ["emotion"]}}},
        ]

    def route(self, user_text: str, state: ChatState) -> RouteDecision:
        system = (
            "你是语音助手项目的意图分发器。"
            "从工具中选择最合适的工作流，并给出emotion、query等参数。"
        )
        state_hint = {
            "is_cloned_voice": state.is_cloned_voice,
            "dialect": state.dialect,
            "role": state.role,
            "scene": state.scene,
        }

        resp = self.client.chat.completions.create(
            model=self.router_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"当前状态: {json.dumps(state_hint, ensure_ascii=False)}\n用户输入: {user_text}"},
            ],
            tools=self.tools,
            tool_choice="auto",
            temperature=0.1,
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return RouteDecision(intent="default", emotion="neutral", query=user_text)

        tc = tool_calls[0] # 目前 router 模型输出一个工具调用，取第一个（如果有多个工具调用，可以改进这里的逻辑）
        fn_name = tc.function.name
        args = _safe_json_load(tc.function.arguments or "{}")

        emotion = str(args.get("emotion", "neutral")).strip()
        if emotion not in EMOTIONS:
            emotion = "neutral"

        if fn_name == "workflow_default":
            return RouteDecision(intent="default", emotion=emotion, query=str(args.get("query", "")).strip())

        if fn_name == "workflow_clone":
            q = str(args.get("query", "")).strip()
            seconds = args.get("seconds", 7)
            return RouteDecision(intent="clone", emotion=emotion, query=q, style_hint=str(seconds))

        if fn_name == "workflow_vision":
            return RouteDecision(intent="vision", emotion=emotion, query=str(args.get("query", "")).strip())

        if fn_name == "workflow_dialect":
            return RouteDecision(
                intent="dialect",
                emotion=emotion,
                query=str(args.get("query", "")).strip(),
                dialect=str(args.get("dialect", "")).strip(),
            )

        if fn_name == "workflow_role_scene":
            role = args.get("role")
            scene = args.get("scene")
            style_hint = args.get("style_hint")
            q = str(args.get("query", "")).strip()
            return RouteDecision(
                intent="role_scene",
                emotion=emotion,
                query=q,
                role=str(role).strip() if isinstance(role, str) and role.strip() else None,
                scene=str(scene).strip() if isinstance(scene, str) and scene.strip() else None,
                style_hint=str(style_hint).strip() if isinstance(style_hint, str) and style_hint.strip() else None,
            )

        if fn_name == "workflow_reset":
            return RouteDecision(intent="reset", emotion=emotion)

        if fn_name == "workflow_wheel_move":
            return RouteDecision(
                intent="wheel_move",
                emotion=emotion,
                query=str(args.get("query", "")).strip(),
                move_action=str(args.get("action", "")).strip() or None,
                move_amount=str(args.get("amount", "normal")).strip() or "normal",
                face_lock=bool(args.get("face_lock", True)),
            )

        if fn_name == "workflow_wheel_follow_face":
            return RouteDecision(
                intent="wheel_follow_face",
                emotion=emotion,
                query=str(args.get("query", "")).strip(),
            )

        if fn_name == "workflow_wheel_stop":
            return RouteDecision(
                intent="wheel_stop",
                emotion=emotion,
                query=str(args.get("query", "")).strip(),
            )

        return RouteDecision(intent="default", emotion=emotion, query=user_text)

    def _build_chat_system_prompt(self, state: ChatState, emotion: Emotion) -> str:
        """构建对话提示词，指导后续的对话生成。根据当前状态（如是否克隆音色、方言/角色/场景信息）和情感，生成适当的提示语。"""
        system = (
            "你是语音对话助手。输出要口语化、简洁、适合直接朗读。\n"
            f"本轮情感类型：{emotion}。请用对应语气表达（用词/语气/标点体现）。\n"
            "如果问题不清楚，只问1个最关键的澄清问题。\n"
        )

        # 未克隆阶段：用 LLM 文本实现方言口吻（克隆后方言由 TTS instruction 实现）
        if state.dialect and not state.is_cloned_voice:
            system += f"\n接下来请用的口吻表达，可夹带少量典型方言词汇。\n"

        if state.role:
            system += f"\n你正在扮演：{state.role}。\n"
        if state.scene:
            system += f"\n当前场景：{state.scene}。\n"

        if state.style_hint:
            hint = state.style_hint.lower()
            if "rap" in hint or "押韵" in hint or "唱" in hint:
                system += "\n用户想听中文rap：输出 8~16 行，每行尽量押韵，不要写括号舞台说明。\n"
            if "脱口秀" in hint:
                system += "\n用户想听脱口秀：输出 120~200 字，包含一个包袱或反转。\n"

        return system

    def chat_answer(self, user_text: str, state: ChatState, emotion: Emotion) -> str:
        """根据用户输入和当前状态，生成助手的回答文本。这个函数会把用户输入和系统提示词加入对话历史，然后调用 LLM 生成回答，并把回答加入对话历史。"""
        system = self._build_chat_system_prompt(state, emotion)
        state.ensure_system(system)
        state.add_user(user_text)

        resp = self.client.chat.completions.create(
            model=self.text_model,
            messages=state.messages,
            temperature=0.7,
        )
        ans = (resp.choices[0].message.content or "").strip()
        if ans:
            state.add_assistant(ans)
        return ans

    # ✅关键：流式文本输出（给 cosyvoice.speak_stream 用）
    def chat_answer_stream(self, user_text: str, state: ChatState, emotion: Emotion) -> Iterator[str]:
        """
        流式输出：边生成边 yield token，同时在终端边打印。
        最终把完整答案写入 state.messages（和 chat_answer 一样有记忆）。
        """
        system = self._build_chat_system_prompt(state, emotion)
        state.ensure_system(system)
        state.add_user(user_text)

        stream = self.client.chat.completions.create(
            model=self.text_model,
            messages=state.messages,
            temperature=0.7,
            stream=True,
        )

        parts: List[str] = []
        print("[ASSIST] ", end="", flush=True)

        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if not delta:
                continue
            parts.append(delta)
            print(delta, end="", flush=True)
            yield delta

        print()
        final = "".join(parts).strip()
        if final:
            state.add_assistant(final)
