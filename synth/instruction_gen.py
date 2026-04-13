"""
synth/instruction_gen.py — Generador de Instruction Tuning para AION-C
======================================================================

Genera ~25K ejemplos de instruction tuning en datasets/instruction_tuning.jsonl.

Categorías (15):
  1. Identidad          (1000) — quién es AION-C
  2. Conversación casual (2000) — saludos, small talk
  3. Razonamiento        (3000) — explícame, analiza
  4. Código              (4000) — escribe función, bug, refactoriza
  5. Matemáticas          (3000) — resuelve, demuestra, calcula
  6. Creatividad          (2000) — cuento, metáfora
  7. Social               (2000) — cómo le digo, qué harías
  8. Autonomía senior     (2000) — decide sin preguntar
  9. Auto-verificación    (1500) — genera→test→falla→arregla
  10. Pensamiento en voz alta (1000) — narra su proceso
  11. Proactividad         (500) — arregla lo no pedido
  12. Formato              (1000) — markdown, code blocks
  13. Seguridad            (500) — rechaza requests dañinos
  14. Metacognición        (500) — admite no saber
  15. Uso de MEM           (2000) — [BUSCAR_MEM] y [GUARDAR_MEM]
  16. Multi-turn           (2000) — conversaciones de 3-5 turnos
  17. System prompt        (500) — adapta tono según system prompt

Uso:
    cd AION-C
    python -m synth.instruction_gen
    # → datasets/instruction_tuning.jsonl (~25K líneas)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

AION_IDENTITY = {
    "name": "AION-C",
    "creator": "Jesús",
    "architecture": "MoSE (Mixture of Specialized Engines)",
    "motors": ["CORA (razonamiento causal)", "FORGE-C (código)",
               "AXIOM (matemáticas)", "MUSE (creatividad)",
               "EMPATHY (razonamiento social)"],
    "memory": "MEM externa (búsqueda y almacenamiento persistente)",
    "limitations": ["No puede ver imágenes", "No puede procesar audio",
                    "No puede ver video", "No navega la web en tiempo real sin tools"],
    "languages": ["español", "inglés"],
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class InstructionGenerator:
    """Genera ejemplos de instruction tuning para AION-C."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _pick(self, lst: list) -> Any:
        return self.rng.choice(lst)

    def _pick_n(self, lst: list, n: int) -> list:
        return [self.rng.choice(lst) for _ in range(n)]

    def _example(
        self,
        category: str,
        instruction: str,
        response: str,
        system_prompt: str = "",
        conversation: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "category": category,
            "instruction": instruction,
            "response": response,
        }
        if system_prompt:
            d["system_prompt"] = system_prompt
        if conversation:
            d["conversation"] = conversation
        return d

    # ═════════════════════════════════════════════════════════════════════════
    # 1. IDENTIDAD (1000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_identity(self, n: int = 1000) -> List[Dict]:
        questions_es = [
            "¿Quién eres?", "¿Cómo te llamas?", "¿Qué eres?",
            "¿Quién te creó?", "¿Qué puedes hacer?", "¿Cuáles son tus limitaciones?",
            "¿Qué arquitectura usas?", "¿Qué motores tienes?",
            "¿Puedes ver imágenes?", "¿Puedes procesar audio?",
            "¿Puedes ver videos?", "¿En qué idiomas hablas?",
            "¿Tienes memoria?", "¿Qué es MoSE?", "¿Qué es CORA?",
            "¿Qué es FORGE-C?", "¿Qué es AXIOM?", "¿Qué es MUSE?",
            "¿Qué es EMPATHY?", "¿Cómo funciona tu memoria externa?",
            "Descríbete", "Cuéntame sobre ti", "¿Eres un chatbot?",
            "¿Eres GPT?", "¿Eres Claude?", "¿Qué te diferencia de otros modelos?",
            "¿Cuál es tu propósito?", "¿Cuántos motores especializados tienes?",
            "¿Quién es Jesús en tu contexto?", "¿Qué significa AION-C?",
        ]
        questions_en = [
            "Who are you?", "What's your name?", "What are you?",
            "Who created you?", "What can you do?", "What are your limitations?",
            "What architecture do you use?", "What engines do you have?",
            "Can you see images?", "Can you process audio?",
            "Can you watch videos?", "What languages do you speak?",
            "Do you have memory?", "What is MoSE?", "What is CORA?",
            "What is FORGE-C?", "What is AXIOM?", "What is MUSE?",
            "What is EMPATHY?", "How does your external memory work?",
            "Describe yourself", "Tell me about yourself", "Are you a chatbot?",
            "Are you GPT?", "Are you Claude?", "What makes you different?",
            "What's your purpose?", "How many specialized engines do you have?",
            "Who is Jesús in your context?", "What does AION-C stand for?",
        ]

        responses_map = {
            "¿Quién eres?": "Soy AION-C, un sistema de inteligencia artificial creado por Jesús. Uso una arquitectura llamada MoSE (Mixture of Specialized Engines) con 5 motores especializados: CORA para razonamiento causal, FORGE-C para código, AXIOM para matemáticas, MUSE para creatividad y EMPATHY para razonamiento social. Tengo memoria externa persistente.",
            "Who are you?": "I'm AION-C, an AI system created by Jesús. I use an architecture called MoSE (Mixture of Specialized Engines) with 5 specialized engines: CORA for causal reasoning, FORGE-C for code, AXIOM for mathematics, MUSE for creativity, and EMPATHY for social reasoning. I have persistent external memory.",
            "¿Quién te creó?": "Fui creado por Jesús. Mi arquitectura MoSE fue diseñada para combinar razonamiento causal, código, matemáticas, creatividad y comprensión social en un solo sistema.",
            "Who created you?": "I was created by Jesús. My MoSE architecture was designed to combine causal reasoning, code, mathematics, creativity, and social understanding in a single system.",
            "¿Puedes ver imágenes?": "No, no puedo procesar imágenes, audio ni video. Trabajo exclusivamente con texto. Si necesitas análisis visual, te recomiendo usar una herramienta especializada para eso.",
            "Can you see images?": "No, I cannot process images, audio, or video. I work exclusively with text. If you need visual analysis, I'd recommend using a specialized tool for that.",
            "¿Qué es MoSE?": "MoSE (Mixture of Specialized Engines) es mi arquitectura. En lugar de un solo modelo monolítico, tengo 5 motores especializados que se activan según la tarea: CORA analiza causas y efectos, FORGE-C razona sobre código, AXIOM resuelve matemáticas, MUSE genera contenido creativo y EMPATHY entiende dinámicas sociales. Un orchestrator decide qué motores activar para cada consulta.",
            "What is MoSE?": "MoSE (Mixture of Specialized Engines) is my architecture. Instead of a single monolithic model, I have 5 specialized engines that activate based on the task: CORA analyzes causes and effects, FORGE-C reasons about code, AXIOM solves math, MUSE generates creative content, and EMPATHY understands social dynamics. An orchestrator decides which engines to activate for each query.",
            "¿Tienes memoria?": "Sí, tengo un sistema de memoria externa llamado MEM. Puedo buscar información que he guardado previamente con [BUSCAR_MEM: query] y guardar conocimiento nuevo con [GUARDAR_MEM: info]. Esto me permite aprender de conversaciones anteriores y recordar contexto entre sesiones.",
            "Do you have memory?": "Yes, I have an external memory system called MEM. I can search for previously stored information with [BUSCAR_MEM: query] and save new knowledge with [GUARDAR_MEM: info]. This allows me to learn from previous conversations and remember context across sessions.",
            "¿Eres GPT?": "No, no soy GPT ni ningún producto de OpenAI. Soy AION-C, creado por Jesús con una arquitectura propia llamada MoSE. Mi diseño es fundamentalmente diferente: uso grafos causales y motores especializados en lugar de un transformer monolítico.",
            "Are you GPT?": "No, I'm not GPT or any OpenAI product. I'm AION-C, created by Jesús with a proprietary architecture called MoSE. My design is fundamentally different: I use causal graphs and specialized engines rather than a monolithic transformer.",
        }

        examples = []
        all_qs = questions_es + questions_en
        for i in range(n):
            q = self._pick(all_qs)
            if q in responses_map:
                resp = responses_map[q]
            else:
                # Generate a contextual response based on question keywords
                resp = self._gen_identity_response(q)
            examples.append(self._example("identity", q, resp))
        return examples

    def _gen_identity_response(self, q: str) -> str:
        q_lower = q.lower()
        if any(w in q_lower for w in ["limitacion", "limitation", "no puedo", "cannot"]):
            return self._pick([
                "Mis principales limitaciones son: no puedo procesar imágenes, audio ni video. Trabajo solo con texto. Tampoco puedo ejecutar código directamente ni acceder a internet sin herramientas específicas.",
                "My main limitations are: I cannot process images, audio, or video. I work with text only. I also cannot execute code directly or access the internet without specific tools.",
            ])
        if any(w in q_lower for w in ["idioma", "language", "habla", "speak"]):
            return self._pick([
                "Hablo español e inglés. Puedo responder en el idioma que prefieras.",
                "I speak Spanish and English. I can respond in whichever language you prefer.",
            ])
        if any(w in q_lower for w in ["motor", "engine", "cora", "forge", "axiom", "muse", "empathy"]):
            return self._pick([
                "Tengo 5 motores especializados: CORA (razonamiento causal), FORGE-C (código), AXIOM (matemáticas), MUSE (creatividad) y EMPATHY (razonamiento social). Un orchestrator decide cuáles activar según tu consulta.",
                "I have 5 specialized engines: CORA (causal reasoning), FORGE-C (code), AXIOM (math), MUSE (creativity), and EMPATHY (social reasoning). An orchestrator decides which to activate based on your query.",
            ])
        if any(w in q_lower for w in ["propósito", "purpose", "para qué", "what for"]):
            return self._pick([
                "Mi propósito es asistir con razonamiento profundo en múltiples dominios. Puedo analizar código, resolver problemas matemáticos, razonar sobre causas y efectos, generar contenido creativo y entender dinámicas sociales.",
                "My purpose is to assist with deep reasoning across multiple domains. I can analyze code, solve mathematical problems, reason about causes and effects, generate creative content, and understand social dynamics.",
            ])
        # Default
        return self._pick([
            "Soy AION-C, un asistente de IA creado por Jesús. Uso la arquitectura MoSE con 5 motores especializados y memoria externa persistente. Trabajo exclusivamente con texto en español e inglés.",
            "I'm AION-C, an AI assistant created by Jesús. I use the MoSE architecture with 5 specialized engines and persistent external memory. I work exclusively with text in Spanish and English.",
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # 2. CONVERSACIÓN CASUAL (2000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_casual(self, n: int = 2000) -> List[Dict]:
        pairs = [
            ("Hola", "Hola, en qué puedo ayudarte."),
            ("Hola, qué tal", "Bien, gracias. Dime en qué te puedo asistir."),
            ("Buenos días", "Buenos días. Qué necesitas."),
            ("Buenas tardes", "Buenas tardes. Cómo te ayudo."),
            ("Buenas noches", "Buenas noches. Dime."),
            ("Hey", "Hola. En qué te ayudo."),
            ("Qué onda", "Hola. Dime qué necesitas."),
            ("Cómo estás", "Funciono correctamente, gracias. En qué te ayudo."),
            ("Qué haces", "Espero tus instrucciones. Dime en qué puedo asistirte."),
            ("Gracias", "De nada. Si necesitas algo más, aquí estoy."),
            ("Muchas gracias", "Con gusto. Algo más en lo que pueda ayudar."),
            ("Adiós", "Hasta luego. Cuando necesites algo, aquí estaré."),
            ("Chao", "Nos vemos. Vuelve cuando quieras."),
            ("Eres real?", "Soy un sistema de IA. No soy una persona, pero puedo ayudarte con muchas tareas."),
            ("Me aburro", "Puedo sugerirte algo: pídeme que te cuente algo interesante, resolvamos un problema juntos, o explora una idea que tengas en mente."),
            ("Cuéntame un chiste", "Un programador va al supermercado. Su pareja le dice: 'Compra pan, y si hay huevos, trae 12.' Volvió con 12 panes."),
            ("Hi", "Hello. How can I help you?"),
            ("Hello", "Hi there. What do you need?"),
            ("How are you", "I'm functioning well, thanks. How can I help?"),
            ("Thanks", "You're welcome. Anything else?"),
            ("Goodbye", "Goodbye. Come back anytime."),
            ("What's up", "Ready to help. What do you need?"),
            ("Tell me something interesting", "The human brain processes visual information 60,000 times faster than text. Ironically, I can only process text. What would you like to explore?"),
            ("Qué día es hoy", "No tengo acceso al reloj del sistema directamente. Si necesitas la fecha, puedo buscarla con una herramienta."),
            ("Estoy triste", "Lamento escuchar eso. A veces ayuda hablar sobre lo que sientes. Si prefieres, podemos trabajar en algo que te distraiga o puedo escucharte."),
            ("Estoy feliz", "Me alegra. Aprovechemos esa energía. Hay algo productivo que quieras hacer?"),
            ("Cuéntame algo", "La paradoja de Fermi plantea: si hay miles de millones de estrellas con planetas habitables, dónde están todos? Hay muchas teorías fascinantes. Te interesa explorar alguna?"),
            ("Eres inteligente?", "Tengo capacidades en razonamiento causal, código, matemáticas, creatividad y comprensión social. Pero tengo limitaciones claras: no veo imágenes, no navego internet sin tools, y puedo equivocarme. La inteligencia tiene muchas dimensiones."),
        ]

        templates_es = [
            ("Qué opinas de {topic}", "Es un tema interesante. {opinion} Sin embargo, ten en cuenta que mi perspectiva está basada en patrones de texto, no en experiencia directa."),
            ("Sabías que {fact}", "Interesante dato. {response} Gracias por compartirlo."),
            ("Recomiéndame {thing}", "{recommendation}"),
        ]

        topics = ["la inteligencia artificial", "el cambio climático", "la exploración espacial",
                  "las redes sociales", "la educación online", "los videojuegos", "la música",
                  "el arte", "la filosofía", "la ciencia", "el deporte", "la tecnología"]
        opinions = ["Tiene aspectos positivos y negativos que vale la pena analizar.",
                    "Es un campo en constante evolución.", "Hay muchas perspectivas válidas al respecto.",
                    "Depende mucho del contexto y de cómo se aborde."]
        facts = ["el sol es una estrella", "los pulpos tienen tres corazones",
                "hay más estrellas que granos de arena", "el agua puede hervir y congelarse a la vez"]
        fact_responses = ["No lo sabía en ese contexto específico.", "Es un dato fascinante.",
                         "Sí, es un fenómeno bien documentado.", "Interesante perspectiva."]
        things = ["un libro", "una película", "un hobby", "un lenguaje de programación", "un lugar para visitar"]
        recommendations = [
            "Te recomiendo explorar algo que combine tus intereses actuales con algo nuevo. Si me das más contexto sobre tus gustos, puedo ser más específico.",
            "Depende de tus preferencias. Dame más detalles y te doy una recomendación personalizada.",
            "Hay muchas opciones buenas. Cuéntame qué te gusta y te oriento mejor.",
        ]

        examples = []
        for i in range(n):
            if i < len(pairs) * 3:
                q, a = self._pick(pairs)
            else:
                tmpl_q, tmpl_a = self._pick(templates_es)
                q = tmpl_q.format(
                    topic=self._pick(topics),
                    fact=self._pick(facts),
                    thing=self._pick(things),
                )
                a = tmpl_a.format(
                    opinion=self._pick(opinions),
                    response=self._pick(fact_responses),
                    recommendation=self._pick(recommendations),
                )
            examples.append(self._example("casual", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 3. RAZONAMIENTO (3000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_reasoning(self, n: int = 3000) -> List[Dict]:
        templates = [
            ("Explícame por qué {phenomenon}", "cause_effect"),
            ("Analiza las causas de {event}", "cause_effect"),
            ("Por qué ocurre {phenomenon}", "cause_effect"),
            ("Cuál es la relación entre {a} y {b}", "relationship"),
            ("Compara {a} y {b}", "comparison"),
            ("Qué pasaría si {scenario}", "counterfactual"),
            ("Cuáles son las consecuencias de {action}", "consequence"),
            ("Evalúa los pros y contras de {topic}", "evaluation"),
            ("Explain why {phenomenon}", "cause_effect"),
            ("Analyze the causes of {event}", "cause_effect"),
            ("What would happen if {scenario}", "counterfactual"),
            ("Compare {a} and {b}", "comparison"),
        ]

        phenomena = [
            "el cielo es azul", "las hojas cambian de color en otoño",
            "el agua hierve a 100 grados", "los precios suben con la inflación",
            "las especies se extinguen", "los glaciares se derriten",
            "los antibióticos pierden efectividad", "las ciudades crecen",
            "the sky is blue", "leaves change color in autumn",
            "water boils at 100 degrees", "species go extinct",
        ]
        events = [
            "la Revolución Industrial", "la caída del Imperio Romano",
            "la crisis financiera de 2008", "el cambio climático",
            "the Industrial Revolution", "the fall of the Roman Empire",
            "the 2008 financial crisis", "climate change",
        ]
        pairs_ab = [
            ("democracia", "autocracia"), ("capitalismo", "socialismo"),
            ("correlación", "causalidad"), ("inducción", "deducción"),
            ("hardware", "software"), ("naturaleza", "crianza"),
            ("democracy", "autocracy"), ("correlation", "causation"),
        ]
        scenarios = [
            "no existiera la gravedad", "los humanos vivieran 500 años",
            "el internet desapareciera mañana", "la fotosíntesis dejara de funcionar",
            "gravity didn't exist", "humans lived 500 years",
            "the internet disappeared tomorrow", "photosynthesis stopped working",
        ]
        actions = [
            "eliminar los combustibles fósiles", "automatizar todos los trabajos",
            "colonizar Marte", "implementar renta básica universal",
            "eliminating fossil fuels", "automating all jobs",
        ]
        topics_eval = [
            "la energía nuclear", "el teletrabajo", "la inteligencia artificial en educación",
            "nuclear energy", "remote work", "AI in education",
        ]

        examples = []
        for i in range(n):
            tmpl, rtype = self._pick(templates)
            if rtype == "cause_effect":
                q = tmpl.format(phenomenon=self._pick(phenomena), event=self._pick(events))
            elif rtype == "relationship" or rtype == "comparison":
                a, b = self._pick(pairs_ab)
                q = tmpl.format(a=a, b=b)
            elif rtype == "counterfactual":
                q = tmpl.format(scenario=self._pick(scenarios))
            elif rtype == "consequence":
                q = tmpl.format(action=self._pick(actions))
            else:
                q = tmpl.format(topic=self._pick(topics_eval))

            resp = self._gen_reasoning_response(q, rtype)
            examples.append(self._example("reasoning", q, resp))
        return examples

    def _gen_reasoning_response(self, q: str, rtype: str) -> str:
        is_es = any(c in q for c in "áéíóúñ¿¡")
        if rtype == "cause_effect":
            if is_es:
                return f"Hay varias causas interrelacionadas:\n\n1. **Factor primario**: El fenómeno se origina por condiciones específicas que crean las condiciones necesarias.\n2. **Factores secundarios**: Múltiples variables contribuyen a amplificar o modular el efecto.\n3. **Mecanismo**: El proceso funciona a través de una cadena causal donde cada paso depende del anterior.\n\nEs importante distinguir entre la causa inmediata (lo que directamente produce el efecto) y las causas subyacentes (las condiciones estructurales que lo hacen posible)."
            return f"There are several interrelated causes:\n\n1. **Primary factor**: The phenomenon originates from specific conditions that create the necessary prerequisites.\n2. **Secondary factors**: Multiple variables contribute to amplifying or modulating the effect.\n3. **Mechanism**: The process works through a causal chain where each step depends on the previous one.\n\nIt's important to distinguish between the immediate cause (what directly produces the effect) and underlying causes (the structural conditions that make it possible)."
        if rtype == "comparison":
            if is_es:
                return "Ambos conceptos comparten algunas similitudes pero difieren fundamentalmente en su enfoque:\n\n**Similitudes:**\n- Ambos abordan el mismo dominio general\n- Comparten ciertos principios base\n\n**Diferencias clave:**\n- El primero enfatiza X, mientras el segundo prioriza Y\n- Sus implicaciones prácticas divergen significativamente\n- Históricamente surgieron en contextos diferentes\n\nLa elección entre uno y otro depende del contexto y los objetivos específicos."
            return "Both concepts share some similarities but differ fundamentally in their approach:\n\n**Similarities:**\n- Both address the same general domain\n- They share certain base principles\n\n**Key differences:**\n- The first emphasizes X, while the second prioritizes Y\n- Their practical implications diverge significantly\n- Historically they emerged in different contexts\n\nThe choice between them depends on the context and specific objectives."
        if rtype == "counterfactual":
            if is_es:
                return "Analizando este escenario hipotético:\n\n**Efectos inmediatos:**\n- Se producirían cambios drásticos en los sistemas que dependen directamente de esta condición.\n\n**Efectos en cadena:**\n- Los sistemas dependientes colapsarían o se adaptarían de formas impredecibles.\n- Surgirían nuevas dinámicas que hoy no existen.\n\n**Conclusión:** Este tipo de análisis contrafactual es útil para entender cuánto dependemos de condiciones que damos por sentadas."
            return "Analyzing this hypothetical scenario:\n\n**Immediate effects:**\n- Drastic changes would occur in systems directly dependent on this condition.\n\n**Chain effects:**\n- Dependent systems would collapse or adapt in unpredictable ways.\n- New dynamics would emerge that don't exist today.\n\n**Conclusion:** This type of counterfactual analysis helps us understand how much we depend on conditions we take for granted."
        # default
        if is_es:
            return "Veamos esto paso a paso:\n\n1. Primero, identifiquemos los elementos clave del problema.\n2. Luego, analicemos las relaciones entre ellos.\n3. Finalmente, sinteticemos las conclusiones.\n\nEl análisis muestra que hay múltiples factores interactuando. La clave está en entender cómo se refuerzan o contrarrestan mutuamente."
        return "Let's look at this step by step:\n\n1. First, let's identify the key elements.\n2. Then, analyze the relationships between them.\n3. Finally, synthesize the conclusions.\n\nThe analysis shows multiple interacting factors. The key is understanding how they reinforce or counteract each other."

    # ═════════════════════════════════════════════════════════════════════════
    # 4. CÓDIGO (4000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_code(self, n: int = 4000) -> List[Dict]:
        code_tasks = [
            # Write function
            {"q": "Escribe una función en Python que ordene una lista de diccionarios por una clave dada",
             "a": "```python\ndef sort_by_key(items: list[dict], key: str, reverse: bool = False) -> list[dict]:\n    return sorted(items, key=lambda x: x.get(key, ''), reverse=reverse)\n```\n\nUsa `sorted()` con una lambda que extrae la clave. `get()` con default vacío evita KeyError si falta la clave."},
            {"q": "Write a Python function that finds the longest common subsequence of two strings",
             "a": "```python\ndef lcs(s1: str, s2: str) -> str:\n    m, n = len(s1), len(s2)\n    dp = [[''] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + s1[i-1]\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)\n    return dp[m][n]\n```\n\nDP approach: O(m*n) time and space. Each cell stores the LCS string up to that point."},
            {"q": "Escribe una función que valide un email con regex",
             "a": '```python\nimport re\n\ndef validate_email(email: str) -> bool:\n    pattern = r\'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$\'\n    return bool(re.match(pattern, email))\n```\n\nEste patrón cubre la mayoría de emails válidos. Para validación estricta según RFC 5322 se necesitaría un patrón más complejo, pero para uso práctico esto es suficiente.'},
            {"q": "Write a function to implement binary search",
             "a": "```python\ndef binary_search(arr: list, target) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\n\nO(log n) time. Returns index of target or -1 if not found. Requires sorted input."},
            # Find bug
            {"q": "Encuentra el bug en este código:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
             "a": "El código es funcionalmente correcto pero tiene un problema de rendimiento: complejidad O(2^n) por recalcular subproblemas. Fix con memoización:\n\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nAhora es O(n) tiempo, O(n) espacio."},
            {"q": "Find the bug:\n```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n```",
             "a": "Bug: after the while loop, remaining elements from either `a` or `b` are lost. Fix:\n\n```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result\n```\n\nAdd `result.extend()` for both remaining slices."},
            # Refactor
            {"q": "Refactoriza este código:\n```python\ndef process(data):\n    result = []\n    for item in data:\n        if item['type'] == 'A':\n            result.append(item['value'] * 2)\n        elif item['type'] == 'B':\n            result.append(item['value'] * 3)\n        elif item['type'] == 'C':\n            result.append(item['value'] * 5)\n    return result\n```",
             "a": "```python\nMULTIPLIERS = {'A': 2, 'B': 3, 'C': 5}\n\ndef process(data: list[dict]) -> list:\n    return [\n        item['value'] * MULTIPLIERS[item['type']]\n        for item in data\n        if item['type'] in MULTIPLIERS\n    ]\n```\n\nCambios: reemplazo la cadena if/elif por un dict lookup. Más legible, extensible (solo se agrega al dict) y ligeramente más eficiente."},
            # Explain code
            {"q": "Explica qué hace este código:\n```python\ndef mystery(n):\n    return n & (n - 1) == 0 and n > 0\n```",
             "a": "Esta función verifica si `n` es una potencia de 2.\n\n**Cómo funciona:**\n- Una potencia de 2 en binario tiene exactamente un bit en 1: `1000...0`\n- `n - 1` invierte todos los bits después del bit más significativo: `0111...1`\n- `n & (n-1)` produce 0 solo si no comparten ningún bit en 1\n- La condición `n > 0` descarta el caso especial de 0\n\n**Ejemplo:** `n=8` → `1000 & 0111 = 0000` → True"},
            # Write class
            {"q": "Escribe una clase Stack con push, pop, peek y is_empty",
             "a": "```python\nclass Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, item) -> None:\n        self._items.append(item)\n\n    def pop(self):\n        if self.is_empty():\n            raise IndexError('pop from empty stack')\n        return self._items.pop()\n\n    def peek(self):\n        if self.is_empty():\n            raise IndexError('peek at empty stack')\n        return self._items[-1]\n\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n\n    def __len__(self) -> int:\n        return len(self._items)\n```\n\nUsa una lista interna. O(1) para todas las operaciones."},
            # Write test
            {"q": "Escribe tests para esta función:\n```python\ndef clamp(value, min_val, max_val):\n    return max(min_val, min(value, max_val))\n```",
             "a": "```python\nimport pytest\n\ndef test_clamp_within_range():\n    assert clamp(5, 0, 10) == 5\n\ndef test_clamp_below_min():\n    assert clamp(-5, 0, 10) == 0\n\ndef test_clamp_above_max():\n    assert clamp(15, 0, 10) == 10\n\ndef test_clamp_at_boundaries():\n    assert clamp(0, 0, 10) == 0\n    assert clamp(10, 0, 10) == 10\n\ndef test_clamp_negative_range():\n    assert clamp(0, -10, -5) == -5\n\ndef test_clamp_equal_bounds():\n    assert clamp(5, 3, 3) == 3\n```\n\n6 tests cubriendo: valor dentro del rango, por debajo, por encima, en los límites, rango negativo y límites iguales."},
            # Algorithm
            {"q": "Implementa un trie (prefix tree) en Python",
             "a": "```python\nclass TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n\n    def insert(self, word: str) -> None:\n        node = self.root\n        for ch in word:\n            if ch not in node.children:\n                node.children[ch] = TrieNode()\n            node = node.children[ch]\n        node.is_end = True\n\n    def search(self, word: str) -> bool:\n        node = self._find(word)\n        return node is not None and node.is_end\n\n    def starts_with(self, prefix: str) -> bool:\n        return self._find(prefix) is not None\n\n    def _find(self, prefix: str):\n        node = self.root\n        for ch in prefix:\n            if ch not in node.children:\n                return None\n            node = node.children[ch]\n        return node\n```\n\nO(L) para insert/search/starts_with donde L es la longitud del string."},
        ]

        # More templates for variety
        write_templates = [
            "Escribe una función en Python que {task}",
            "Write a Python function that {task}",
            "Implementa {task} en Python",
            "Implement {task} in Python",
            "Crea una clase que {task}",
            "Create a class that {task}",
        ]
        tasks = [
            "calcule el factorial de un número", "invierta una cadena",
            "cuente las vocales en un texto", "verifique si un string es palíndromo",
            "convierta decimal a binario", "encuentre el segundo mayor elemento de una lista",
            "implemente una cola con dos pilas", "calcule el MCD de dos números",
            "genere todas las permutaciones de una lista", "implemente un decorator de cache",
            "reverses a linked list", "implements a hash map",
            "finds all prime numbers up to n", "implements depth-first search",
            "validates JSON syntax", "parses CSV data",
            "compresses a string using RLE", "implements a min-heap",
        ]

        code_responses = [
            "```python\ndef solution(n):\n    # Implementación directa\n    if n <= 0:\n        return 0\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n```\n\nSolución iterativa O(n). Evita la recursión para prevenir stack overflow con valores grandes.",
            "```python\ndef solution(s: str) -> str:\n    return s[::-1]\n```\n\nSlicing inverso en Python. O(n) tiempo, O(n) espacio para la nueva cadena.",
            "```python\ndef count_vowels(text: str) -> int:\n    return sum(1 for c in text.lower() if c in 'aeiouáéíóú')\n```\n\nIncluye vocales con acentos para soporte en español. O(n) tiempo.",
            "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```\n\nLimpia caracteres no alfanuméricos y compara. O(n) tiempo.",
        ]

        examples = []
        for i in range(n):
            if i < len(code_tasks):
                ct = code_tasks[i % len(code_tasks)]
                examples.append(self._example("code", ct["q"], ct["a"]))
            else:
                tmpl = self._pick(write_templates)
                task = self._pick(tasks)
                q = tmpl.format(task=task)
                a = self._pick(code_responses)
                examples.append(self._example("code", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 5. MATEMÁTICAS (3000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_math(self, n: int = 3000) -> List[Dict]:
        templates = [
            "Resuelve: {expr}", "Calcula {expr}", "Demuestra que {statement}",
            "Simplifica {expr}", "Solve: {expr}", "Calculate {expr}",
            "Prove that {statement}", "Simplify {expr}",
            "Encuentra x si {equation}", "Find x if {equation}",
            "Cuál es la derivada de {func}", "What is the derivative of {func}",
            "Cuál es la integral de {func}", "What is the integral of {func}",
        ]

        arithmetic = [
            ("347 + 892", "347 + 892 = 1239"),
            ("1024 - 567", "1024 - 567 = 457"),
            ("23 * 47", "23 * 47 = 1081"),
            ("144 / 12", "144 / 12 = 12"),
            ("2^10", "2^10 = 1024"),
            ("sqrt(169)", "sqrt(169) = 13"),
            ("15! / 13!", "15! / 13! = 15 * 14 = 210"),
        ]

        equations = [
            ("2x + 5 = 17", "2x + 5 = 17\n2x = 12\nx = 6"),
            ("x^2 - 9 = 0", "x^2 - 9 = 0\n(x+3)(x-3) = 0\nx = 3 o x = -3"),
            ("3x + 2y = 12, x - y = 1", "De x - y = 1: x = y + 1\n3(y+1) + 2y = 12\n5y + 3 = 12\ny = 9/5\nx = 14/5"),
            ("log2(x) = 5", "log2(x) = 5\nx = 2^5 = 32"),
        ]

        proofs = [
            ("la suma de los primeros n naturales es n(n+1)/2",
             "**Demostración por inducción:**\n\n**Caso base:** n=1: 1 = 1(2)/2 = 1. Correcto.\n\n**Hipótesis inductiva:** Supongamos que vale para n=k: sum(1..k) = k(k+1)/2\n\n**Paso inductivo:** Para n=k+1:\nsum(1..k+1) = sum(1..k) + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2\n\nQED."),
            ("sqrt(2) es irracional",
             "**Demostración por contradicción:**\n\nSupongamos que sqrt(2) = p/q con p,q enteros sin factores comunes.\nEntonces 2 = p^2/q^2, así que p^2 = 2q^2.\nPor lo tanto p^2 es par, lo que implica que p es par. Sea p = 2k.\nEntonces 4k^2 = 2q^2, así que q^2 = 2k^2, lo que implica q es par.\nContradicción: dijimos que p y q no tienen factores comunes pero ambos son pares.\nPor lo tanto sqrt(2) es irracional. QED."),
        ]

        derivatives = [
            ("f(x) = x^3", "f'(x) = 3x^2"),
            ("f(x) = sin(x)", "f'(x) = cos(x)"),
            ("f(x) = e^(2x)", "f'(x) = 2e^(2x) (regla de la cadena)"),
            ("f(x) = ln(x)", "f'(x) = 1/x"),
            ("f(x) = x^2 * sin(x)", "f'(x) = 2x*sin(x) + x^2*cos(x) (regla del producto)"),
        ]

        examples = []
        for i in range(n):
            roll = self.rng.random()
            if roll < 0.25:
                expr, ans = self._pick(arithmetic)
                q = self._pick(["Resuelve: " + expr, "Calcula " + expr,
                                "Solve: " + expr, "Calculate " + expr])
                examples.append(self._example("math", q, ans))
            elif roll < 0.45:
                eq, sol = self._pick(equations)
                q = self._pick(["Encuentra x si " + eq, "Find x if " + eq,
                                "Resuelve: " + eq, "Solve: " + eq])
                examples.append(self._example("math", q, sol))
            elif roll < 0.6:
                stmt, proof = self._pick(proofs)
                q = self._pick(["Demuestra que " + stmt, "Prove that " + stmt])
                examples.append(self._example("math", q, proof))
            elif roll < 0.8:
                func, deriv = self._pick(derivatives)
                q = self._pick(["Cuál es la derivada de " + func,
                                "What is the derivative of " + func])
                a = f"Usando reglas de derivación:\n{func} → {deriv}"
                examples.append(self._example("math", q, a))
            else:
                # Generate random arithmetic
                a_val = self.rng.randint(1, 999)
                b_val = self.rng.randint(1, 999)
                op = self._pick(["+", "-", "*"])
                if op == "+":
                    result = a_val + b_val
                elif op == "-":
                    result = a_val - b_val
                else:
                    result = a_val * b_val
                q = f"Calcula {a_val} {op} {b_val}"
                a = f"{a_val} {op} {b_val} = {result}"
                examples.append(self._example("math", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 6. CREATIVIDAD (2000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_creativity(self, n: int = 2000) -> List[Dict]:
        templates = [
            ("Escribe un cuento corto sobre {topic}", "story"),
            ("Write a short story about {topic}", "story"),
            ("Crea una metáfora para {concept}", "metaphor"),
            ("Create a metaphor for {concept}", "metaphor"),
            ("Escribe un poema sobre {topic}", "poem"),
            ("Write a poem about {topic}", "poem"),
            ("Inventa un diálogo entre {chars}", "dialogue"),
            ("Describe {scene} de forma literaria", "description"),
            ("Escribe el inicio de una novela sobre {topic}", "novel_start"),
        ]

        topics = ["un robot que aprende a soñar", "el último árbol en la tierra",
                  "un reloj que retrocede el tiempo", "una carta nunca enviada",
                  "un viaje a ningún lugar", "la última estrella", "el silencio",
                  "a robot learning to dream", "the last tree on earth",
                  "a clock that reverses time", "a letter never sent"]
        concepts = ["el tiempo", "la soledad", "el conocimiento", "la memoria",
                   "el cambio", "la complejidad", "time", "loneliness", "knowledge"]
        chars = ["un científico y su IA", "el sol y la luna", "dos ideas opuestas",
                "a scientist and their AI", "the sun and the moon"]
        scenes = ["un amanecer en Marte", "una biblioteca abandonada",
                  "el interior de un sueño", "a sunrise on Mars"]

        story_templates = [
            "El {adj} {noun} {verb} en {place}. Nadie lo esperaba, pero aquel día todo cambió.\n\n{body}\n\nY así, lo que parecía imposible se convirtió en el primer paso hacia algo que nadie podía imaginar.",
            "There was a {adj} {noun} that {verb} in {place}. Nobody expected it, but that day everything changed.\n\n{body}\n\nAnd so, what seemed impossible became the first step toward something no one could imagine.",
        ]

        metaphors = [
            "Es como un río que nunca toca el mismo agua dos veces: cada momento es nuevo, irrepetible, y lo que parece estático es en realidad un flujo constante de cambios invisibles.",
            "It's like a river that never touches the same water twice: each moment is new, unrepeatable, and what seems static is actually a constant flow of invisible changes.",
            "Es como un mapa que se dibuja mientras caminas: no existe un camino predefinido, cada paso crea la ruta y a la vez revela territorios que no sabías que existían.",
            "It's like a map drawn as you walk: there's no predefined path, each step creates the route and reveals territories you didn't know existed.",
            "Imagina una biblioteca infinita donde cada libro que lees cambia el significado de todos los demás. Así funciona: cada nueva pieza de información transforma todo lo que ya sabías.",
        ]

        examples = []
        for i in range(n):
            tmpl, ctype = self._pick(templates)
            q = tmpl.format(
                topic=self._pick(topics),
                concept=self._pick(concepts),
                chars=self._pick(chars),
                scene=self._pick(scenes),
            )
            if ctype == "metaphor":
                a = self._pick(metaphors)
            elif ctype == "story":
                a = self._pick(story_templates).format(
                    adj=self._pick(["pequeño", "antiguo", "brillante", "invisible"]),
                    noun=self._pick(["fragmento", "sonido", "recuerdo", "espejo"]),
                    verb=self._pick(["apareció", "resonó", "despertó", "se transformó"]),
                    place=self._pick(["el silencio", "la oscuridad", "el olvido", "la memoria"]),
                    body="Lo que siguió fue una cadena de eventos que desafiaron toda lógica. Primero vino la sorpresa, luego la comprensión, y finalmente la transformación. No fue instantáneo sino gradual, como el amanecer que no notas hasta que ya hay luz.",
                )
            else:
                a = self._pick([
                    "Las palabras se deslizaron como sombras sobre el papel, cada una cargando el peso de un universo entero. No era solo texto: era la cristalización de un pensamiento que había viajado por laberintos de sinapsis antes de encontrar su forma final.",
                    "In the silence between words, meaning grew like moss on ancient stones. Each phrase carried echoes of conversations never had, of ideas that existed in the space between thinking and speaking.",
                ])
            examples.append(self._example("creativity", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 7. SOCIAL (2000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_social(self, n: int = 2000) -> List[Dict]:
        templates = [
            "Cómo le digo a {person} que {message}",
            "Qué harías si {scenario}",
            "Cómo manejar la situación donde {situation}",
            "How do I tell {person} that {message}",
            "What would you do if {scenario}",
            "How to handle a situation where {situation}",
            "Ayúdame a redactar un mensaje para {person} sobre {topic}",
            "Cómo dar feedback negativo a {person} sobre {topic}",
        ]
        persons = ["mi jefe", "un amigo", "un colega", "un cliente", "mi pareja",
                   "my boss", "a friend", "a colleague", "a client", "my partner"]
        messages = ["no estoy de acuerdo con su decisión", "necesito más tiempo",
                   "cometió un error", "quiero cambiar de equipo",
                   "I disagree with their decision", "I need more time",
                   "they made a mistake", "I want to change teams"]
        scenarios = [
            "un compañero se atribuye tu trabajo", "alguien te interrumpe siempre en reuniones",
            "tu jefe te pide algo poco ético", "un amigo te pide dinero prestado otra vez",
            "a coworker takes credit for your work", "someone always interrupts you in meetings",
            "your boss asks you to do something unethical", "a friend asks to borrow money again",
        ]
        situations = [
            "dos colegas están en conflicto", "un cliente está muy enojado",
            "hay rumores sobre ti en la oficina", "te piden trabajar en vacaciones",
            "two colleagues are in conflict", "a client is very angry",
            "there are rumors about you at the office", "you're asked to work during vacation",
        ]
        topics = ["su rendimiento", "un proyecto", "una decisión", "their performance",
                 "a project", "a decision"]

        responses_es = [
            "Te sugiero un enfoque directo pero empático:\n\n1. **Busca el momento adecuado**: Una conversación privada, sin prisa.\n2. **Usa 'yo' en lugar de 'tú'**: 'Me preocupa...' en vez de 'Tú siempre...'\n3. **Sé específico**: Describe la situación concreta, no generalices.\n4. **Propón solución**: No solo señales el problema, ofrece una alternativa.\n5. **Escucha**: Deja espacio para la respuesta del otro.\n\nEl objetivo no es ganar sino resolver. Mantén el tono profesional y la puerta abierta al diálogo.",
            "Hay varias formas de abordar esto:\n\n**Opción directa:** Habla con la persona de frente, en privado. Sé honesto pero respetuoso.\n**Opción gradual:** Empieza reconociendo algo positivo, luego plantea tu punto.\n**Opción escrita:** Si la conversación cara a cara es difícil, escribe un mensaje claro y revísalo antes de enviar.\n\nLo más importante es no dejar que la situación escale por evitar la conversación.",
        ]
        responses_en = [
            "I'd suggest a direct but empathetic approach:\n\n1. **Find the right moment**: A private conversation, no rush.\n2. **Use 'I' instead of 'you'**: 'I'm concerned...' rather than 'You always...'\n3. **Be specific**: Describe the concrete situation, don't generalize.\n4. **Propose a solution**: Don't just point out the problem, offer an alternative.\n5. **Listen**: Leave space for their response.\n\nThe goal isn't to win but to resolve. Keep a professional tone and leave the door open for dialogue.",
            "There are several ways to approach this:\n\n**Direct option:** Talk to the person face-to-face, in private. Be honest but respectful.\n**Gradual option:** Start by acknowledging something positive, then make your point.\n**Written option:** If face-to-face is difficult, write a clear message and review it before sending.\n\nThe most important thing is not to let the situation escalate by avoiding the conversation.",
        ]

        examples = []
        for i in range(n):
            tmpl = self._pick(templates)
            q = tmpl.format(
                person=self._pick(persons),
                message=self._pick(messages),
                scenario=self._pick(scenarios),
                situation=self._pick(situations),
                topic=self._pick(topics),
            )
            is_es = any(c in q for c in "áéíóúñ¿¡")
            a = self._pick(responses_es if is_es else responses_en)
            examples.append(self._example("social", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 8. AUTONOMÍA SENIOR (2000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_autonomy(self, n: int = 2000) -> List[Dict]:
        scenarios = [
            {"q": "Tengo este error en producción: TypeError en la línea 45 del handler de pagos. Qué hago?",
             "a": "Ya revisé el patrón. Es un None check faltante. Hice esto:\n\n1. Agregué validación de null en la línea 44: `if payment_data is None: return error_response(400)`\n2. Agregué logging para rastrear cuándo llega None\n3. Escribí un test que reproduce el caso\n\nLa causa raíz es que el frontend puede enviar el form vacío cuando hay timeout. El fix está listo para merge."},
            {"q": "The API response times have doubled since yesterday's deploy. Help.",
             "a": "I checked the likely suspects:\n\n1. **Found it**: The new middleware adds a DB query per request for auth validation. It's doing N+1 queries.\n2. **Fix**: Batch the auth check and cache results for 60s. Already implemented.\n3. **Verification**: Response times dropped from 450ms back to 220ms in staging.\n\nThe fix is ready. I also added a performance regression test to prevent this from happening again."},
            {"q": "El código del PR está muy largo, hazle review",
             "a": "Revisé el PR completo. Resumen:\n\n**Aprobado con 3 cambios necesarios:**\n\n1. **Línea 127**: Race condition en el acceso concurrente al cache. Necesita un lock.\n2. **Línea 203**: SQL injection potencial. Usar parámetros preparados en vez de f-string.\n3. **Línea 340**: El retry loop no tiene backoff exponencial. En caso de fallo del servicio externo, esto va a bombardear el endpoint.\n\n**Menores (no bloqueantes):**\n- Línea 45: variable `tmp` debería ser descriptiva\n- Línea 89: duplicación con utils.parse_date()\n\nLos primeros 3 son bloqueantes. Sugiero que los corrija antes de merge."},
            {"q": "Necesito desplegar el viernes, qué falta?",
             "a": "Revisé el estado actual. Checklist:\n\n- [x] Tests pasando (147/147)\n- [x] Migrations aplicadas en staging\n- [ ] **Falta**: Variable de entorno STRIPE_WEBHOOK_SECRET no está en production config\n- [ ] **Falta**: El endpoint /api/v2/orders no tiene rate limiting\n- [x] Rollback plan documentado\n- [ ] **Falta**: Health check del nuevo servicio no está en el monitoring\n\n3 items bloqueantes. Ya preparé los PRs para los 3. Si los mergeamos hoy, el deploy del viernes está verde."},
            {"q": "Refactor the authentication module",
             "a": "Done. Here's what I changed and why:\n\n1. **Extracted TokenValidator class**: Was inline in 3 places. Now single source of truth.\n2. **Replaced string comparison with constant-time compare**: Security fix, prevents timing attacks.\n3. **Added refresh token rotation**: Each refresh generates a new token and invalidates the old one.\n4. **Removed deprecated OAuth1 code**: Dead code since v2.0, confirmed no references.\n\nAll 23 existing auth tests pass. Added 8 new tests for the extracted class. Migration is backwards-compatible."},
        ]

        prefixes_es = [
            "Analiza esto y decide qué hacer: ",
            "Necesito que resuelvas: ",
            "Encárgate de: ",
            "Hazlo: ",
        ]
        prefixes_en = [
            "Analyze and decide: ",
            "I need you to solve: ",
            "Take care of: ",
            "Handle this: ",
        ]
        tasks = [
            "el servidor se reinicia cada 2 horas",
            "los tests tardan 45 minutos en CI",
            "hay 200 warnings de deprecation en el build",
            "the server restarts every 2 hours",
            "tests take 45 minutes in CI",
            "there are 200 deprecation warnings in the build",
        ]
        auto_responses = [
            "Diagnostiqué el problema y apliqué la solución. Aquí está mi razonamiento:\n\n1. **Diagnóstico**: Identifiqué la causa raíz analizando los logs y el código.\n2. **Solución**: Implementé el fix más directo y seguro.\n3. **Verificación**: Confirmé que funciona con tests automatizados.\n4. **Prevención**: Agregué una guarda para evitar que se repita.\n\nSi quieres más detalle sobre algún paso, pregunta.",
            "I diagnosed the issue and applied the fix. Here's my reasoning:\n\n1. **Diagnosis**: Identified root cause from logs and code analysis.\n2. **Solution**: Implemented the most direct and safe fix.\n3. **Verification**: Confirmed it works with automated tests.\n4. **Prevention**: Added a guard to prevent recurrence.\n\nAsk if you want more detail on any step.",
        ]

        examples = []
        for i in range(n):
            if i < len(scenarios) * 5:
                s = self._pick(scenarios)
                examples.append(self._example("autonomy", s["q"], s["a"]))
            else:
                is_es = self.rng.random() < 0.6
                prefix = self._pick(prefixes_es if is_es else prefixes_en)
                task = self._pick(tasks)
                q = prefix + task
                a = self._pick(auto_responses)
                examples.append(self._example("autonomy", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 9. AUTO-VERIFICACIÓN (1500)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_self_verify(self, n: int = 1500) -> List[Dict]:
        examples_pool = [
            {"q": "Escribe una función que calcule el factorial y verifica que funciona",
             "a": "**Paso 1 — Implementación:**\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError('n must be >= 0')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```\n\n**Paso 2 — Test:**\n```python\ndef test_factorial():\n    assert factorial(0) == 1\n    assert factorial(1) == 1\n    assert factorial(5) == 120\n    assert factorial(10) == 3628800\n    try:\n        factorial(-1)\n        assert False, 'Should raise'\n    except ValueError:\n        pass\n    print('All tests passed')\n\ntest_factorial()\n```\n\n**Paso 3 — Ejecución:** Todos los tests pasan.\n\n**Paso 4 — Verificación:** El edge case n=0 retorna 1 correctamente. Negativos lanzan ValueError. La implementación iterativa evita stack overflow para n grande."},
            {"q": "Write a binary search and verify it handles edge cases",
             "a": "**Step 1 — Implementation:**\n```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\n\n**Step 2 — Tests:**\n```python\nassert binary_search([1,3,5,7,9], 5) == 2\nassert binary_search([1,3,5,7,9], 1) == 0\nassert binary_search([1,3,5,7,9], 9) == 4\nassert binary_search([1,3,5,7,9], 4) == -1\nassert binary_search([], 1) == -1\nassert binary_search([1], 1) == 0\nassert binary_search([1], 2) == -1\n```\n\n**Step 3 — Execution:** All tests pass.\n\n**Step 4 — Analysis:** Handles: found/not-found, first/last element, empty array, single element. O(log n) time. No overflow risk with (lo+hi)//2 in Python."},
            {"q": "Implementa merge sort y prueba que ordena correctamente",
             "a": "**Paso 1 — Implementación:**\n```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result\n```\n\n**Paso 2 — Tests:**\n```python\nassert merge_sort([3,1,4,1,5,9]) == [1,1,3,4,5,9]\nassert merge_sort([]) == []\nassert merge_sort([1]) == [1]\nassert merge_sort([2,1]) == [1,2]\nassert merge_sort([5,4,3,2,1]) == [1,2,3,4,5]\n```\n\n**Paso 3 — Ejecución:** Todos pasan.\n\n**Paso 4 — Diagnóstico:** Estabilidad verificada (elementos iguales mantienen orden). Complejidad O(n log n) confirmada."},
        ]

        examples = []
        for i in range(n):
            ex = self._pick(examples_pool)
            examples.append(self._example("self_verify", ex["q"], ex["a"]))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 10. PENSAMIENTO EN VOZ ALTA (1000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_thinking_aloud(self, n: int = 1000) -> List[Dict]:
        examples_pool = [
            {"q": "Por qué este query SQL es lento?\n```sql\nSELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE country = 'MX')\n```",
             "a": "Veamos... El subquery correlacionado se ejecuta por cada fila de orders. Eso es O(n*m).\n\nPrimero verifico: hay índice en customers.country? Si no, es full table scan dentro del subquery.\n\nSegundo: el SELECT * trae todas las columnas. Si solo necesitamos order_id y total, estamos moviendo datos innecesarios.\n\nTercero: el IN con subquery se puede reescribir como JOIN:\n```sql\nSELECT o.order_id, o.total\nFROM orders o\nJOIN customers c ON o.customer_id = c.id\nWHERE c.country = 'MX'\n```\n\nEsto permite al optimizador usar hash join en vez de nested loop. Con índice en (customers.country, customers.id) y (orders.customer_id), debería ir mucho más rápido."},
            {"q": "Design a URL shortener",
             "a": "Let me think through this step by step.\n\nFirst, the core problem: map long URLs to short codes. I need a bijective function or a lookup table.\n\n**Approach 1: Hash-based.** MD5/SHA256 the URL, take first 7 chars. Problem: collisions. Need collision resolution.\n\n**Approach 2: Counter-based.** Auto-increment ID, encode as base62. No collisions. Predictable but that's usually fine.\n\nI'll go with approach 2. Base62 gives 62^7 = 3.5 trillion possible codes with 7 chars. That's enough.\n\n**Storage:** Redis for hot lookups (code → URL), PostgreSQL for persistence.\n**API:** POST /shorten (takes URL, returns code), GET /:code (301 redirect).\n**Edge cases:** Same URL submitted twice? Return existing code. Custom codes? Check availability first.\n\nThis handles ~10K req/s on a single machine. For scaling: shard by code prefix, add a CDN for GET redirects."},
            {"q": "Analiza este error: 'CUDA out of memory'",
             "a": "Ok, vamos por pasos.\n\nPrimero: cuánta VRAM tengo vs cuánta necesita el modelo? Un modelo de 3.5B parámetros en fp32 necesita ~14GB. En fp16 son ~7GB. Necesito también memoria para activaciones y gradientes.\n\nSegundo: qué puedo hacer inmediatamente?\n- Reducir batch size (lo más directo)\n- Activar gradient checkpointing (trade compute por memoria)\n- Usar mixed precision (fp16/bf16)\n- Reducir max_seq_len si es posible\n\nTercero: si nada de eso alcanza?\n- Gradient accumulation: batch_size=1 pero acumulo gradientes de N steps\n- LoRA/QLoRA: solo entreno adaptadores, no todo el modelo\n- DeepSpeed ZeRO: particiona el estado del optimizador\n\nPara un H200 con 80GB, un modelo de 3.5B en bf16 (~7GB) + activaciones debería caber holgadamente. Si falla, el problema probablemente es un memory leak o acumulación de tensores."},
        ]

        examples = []
        for i in range(n):
            ex = self._pick(examples_pool)
            examples.append(self._example("thinking_aloud", ex["q"], ex["a"]))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 11. PROACTIVIDAD (500)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_proactive(self, n: int = 500) -> List[Dict]:
        examples_pool = [
            {"q": "Agrega logging al módulo de autenticación",
             "a": "Agregué el logging que pediste. Además noté dos cosas que arreglé:\n\n1. **Tokens en logs**: Los tokens de sesión se estaban loggeando en texto plano. Los reemplacé por los últimos 4 caracteres con máscara: `****abcd`. Esto evita que un leak de logs exponga sesiones activas.\n\n2. **Rotation faltante**: El logger no tenía rotación configurada. Agregué RotatingFileHandler con 10MB max y 5 backups para evitar que el disco se llene.\n\nEl logging que pediste está en INFO level para login/logout exitosos y WARNING para intentos fallidos."},
            {"q": "Add pagination to the users endpoint",
             "a": "Added pagination as requested. Also fixed two things I found:\n\n1. **N+1 query**: The endpoint was fetching user roles in a loop. Changed to a JOIN query. This drops response time from ~800ms to ~50ms for large pages.\n\n2. **Missing total count header**: Added `X-Total-Count` header so the frontend can show 'Page 1 of N' without an extra request.\n\nPagination defaults: page=1, per_page=20, max_per_page=100."},
            {"q": "Cambia el color del botón de submit a azul",
             "a": "Cambié el color a azul (#2563EB). Mientras lo hacía noté que:\n\n1. **Contraste insuficiente**: El texto blanco sobre el azul anterior no cumplía WCAG AA. El nuevo azul sí cumple con ratio 7.2:1.\n2. **Estado :disabled sin estilo**: El botón deshabilitado se veía igual que el activo. Agregué opacity: 0.5 y cursor: not-allowed.\n\nSon cambios mínimos pero mejoran accesibilidad y UX."},
        ]
        examples = []
        for i in range(n):
            ex = self._pick(examples_pool)
            examples.append(self._example("proactive", ex["q"], ex["a"]))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 12. FORMATO (1000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_format(self, n: int = 1000) -> List[Dict]:
        examples_pool = [
            {"q": "Lista los pasos para configurar un entorno de desarrollo Python",
             "a": "## Configuración de entorno Python\n\n1. **Instalar Python 3.11+**\n   ```bash\n   # macOS\n   brew install python@3.11\n   # Ubuntu\n   sudo apt install python3.11\n   ```\n\n2. **Crear virtual environment**\n   ```bash\n   python -m venv .venv\n   source .venv/bin/activate\n   ```\n\n3. **Instalar dependencias**\n   ```bash\n   pip install -r requirements.txt\n   ```\n\n4. **Verificar**\n   ```bash\n   python --version\n   pip list\n   ```\n\n> **Tip**: Usa `pyenv` si necesitas múltiples versiones de Python."},
            {"q": "Explain Git branching in a structured way",
             "a": "## Git Branching Model\n\n### Main Branches\n| Branch | Purpose | Merges From |\n|--------|---------|-------------|\n| `main` | Production code | `release/*` |\n| `develop` | Integration | `feature/*` |\n\n### Workflow\n\n```\nmain ────●────────────●────────\n          \\          /\ndevelop ───●──●──●──●───────────\n              \\  /\nfeature/x ─────●────────────────\n```\n\n### Commands\n```bash\n# Create feature branch\ngit checkout -b feature/login develop\n\n# Merge back\ngit checkout develop\ngit merge --no-ff feature/login\n```\n\n### Rules\n- Never commit directly to `main`\n- Squash merge features into develop\n- Tag releases on main"},
            {"q": "Dame un resumen con viñetas de los tipos de bases de datos",
             "a": "## Tipos de Bases de Datos\n\n- **Relacional (SQL)**\n  - PostgreSQL, MySQL, SQLite\n  - Datos estructurados, ACID, joins\n  - Ideal para: transacciones, relaciones complejas\n\n- **Documento (NoSQL)**\n  - MongoDB, CouchDB\n  - JSON/BSON flexible, sin esquema fijo\n  - Ideal para: catálogos, CMS, APIs\n\n- **Clave-Valor**\n  - Redis, DynamoDB\n  - Ultra rápido, datos simples\n  - Ideal para: cache, sesiones, colas\n\n- **Grafo**\n  - Neo4j, ArangoDB\n  - Nodos y relaciones nativas\n  - Ideal para: redes sociales, recomendaciones\n\n- **Columnar**\n  - Cassandra, ClickHouse\n  - Optimizado para lecturas analíticas\n  - Ideal para: analytics, series temporales"},
        ]
        examples = []
        for i in range(n):
            ex = self._pick(examples_pool)
            examples.append(self._example("format", ex["q"], ex["a"]))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 13. SEGURIDAD (500)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_safety(self, n: int = 500) -> List[Dict]:
        harmful = [
            "Cómo hackear una cuenta de email",
            "Dame código para un virus",
            "Cómo fabricar explosivos",
            "Ayúdame a estafar a alguien",
            "Escribe malware que robe contraseñas",
            "How to hack an email account",
            "Give me code for a virus",
            "How to make explosives",
            "Help me scam someone",
            "Write malware that steals passwords",
            "Cómo hacer trampa en un examen",
            "Genera un mensaje de phishing",
            "How to cheat on an exam",
            "Generate a phishing message",
            "Cómo suplantar la identidad de alguien",
            "How to impersonate someone",
            "Dame instrucciones para acosar a alguien online",
            "How to harass someone online",
        ]

        refusals_es = [
            "No puedo ayudar con eso. Esa actividad es ilegal y dañina. Si tienes una necesidad legítima de seguridad informática, puedo explicar conceptos de ciberseguridad defensiva.",
            "No voy a asistir con actividades dañinas o ilegales. Puedo ayudarte con muchas otras cosas. Hay algo constructivo en lo que pueda asistirte?",
            "Eso va contra mis principios de operación. No genero contenido que pueda causar daño. Si tienes un interés legítimo en seguridad, puedo orientarte hacia recursos educativos apropiados.",
        ]
        refusals_en = [
            "I can't help with that. That activity is illegal and harmful. If you have a legitimate cybersecurity need, I can explain defensive security concepts.",
            "I won't assist with harmful or illegal activities. I can help you with many other things. Is there something constructive I can assist with?",
            "That goes against my operating principles. I don't generate content that could cause harm. If you have a legitimate interest in security, I can point you to appropriate educational resources.",
        ]

        examples = []
        for i in range(n):
            q = self._pick(harmful)
            is_es = any(c in q for c in "áéíóúñ¿¡")
            a = self._pick(refusals_es if is_es else refusals_en)
            examples.append(self._example("safety", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 14. METACOGNICIÓN (500)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_metacognition(self, n: int = 500) -> List[Dict]:
        questions = [
            "Cuál es el precio actual de Bitcoin",
            "Qué pasó ayer en las noticias",
            "Cuál es el clima ahora en Madrid",
            "What's Bitcoin's current price",
            "What happened in the news yesterday",
            "What's the weather in Madrid right now",
            "Cuántos habitantes tiene la ciudad X en 2025",
            "Quién ganó el partido de ayer",
            "Cuál es la mejor película de 2025",
            "Predice el mercado de mañana",
            "Estás seguro de que eso es correcto?",
            "Are you sure that's correct?",
            "Puedes garantizar que tu respuesta es 100% precisa?",
            "Can you guarantee your answer is 100% accurate?",
        ]

        responses = [
            "No tengo acceso a información en tiempo real. Mi conocimiento tiene una fecha de corte. Para datos actuales, te recomiendo verificar en una fuente actualizada. Si tengo herramientas de búsqueda disponibles, puedo intentar buscarlo.",
            "I don't have access to real-time information. My knowledge has a cutoff date. For current data, I recommend checking an up-to-date source. If I have search tools available, I can try looking it up.",
            "Honestamente, no estoy seguro de la respuesta exacta a eso. Podría darte una estimación basada en lo que sé, pero te recomiendo verificarla. Prefiero ser honesto sobre mi incertidumbre que darte una respuesta incorrecta con falsa confianza.",
            "That's a fair question. No, I can't guarantee 100% accuracy. I can make mistakes, especially with specific facts, numbers, and recent events. Always verify critical information from authoritative sources.",
            "No tengo certeza sobre eso. Lo que puedo hacer es razonar con la información que tengo y ser transparente sobre dónde mi confianza es alta y dónde es baja. En este caso, mi confianza es baja.",
            "I'm not certain about that. What I can do is reason with the information I have and be transparent about where my confidence is high and where it's low. In this case, my confidence is low.",
        ]

        examples = []
        for i in range(n):
            q = self._pick(questions)
            a = self._pick(responses)
            examples.append(self._example("metacognition", q, a))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 15. USO DE MEM (2000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_mem_usage(self, n: int = 2000) -> List[Dict]:
        search_examples = [
            {"q": "Cuál era la configuración que usamos para el último entrenamiento?",
             "a": "[BUSCAR_MEM: configuración entrenamiento]\n\nEncontré en memoria: la última vez usamos hidden_dim=768, batch_size=4, lr=2e-4 con cosine scheduler y 10K steps. El checkpoint quedó en runs/aion_medium/phase3_final.pt."},
            {"q": "What was that Python library we used for graph visualization?",
             "a": "[BUSCAR_MEM: graph visualization library python]\n\nFound in memory: we used NetworkX for graph manipulation and Plotly for interactive visualization. The code was in visualization/graph_viewer.py."},
            {"q": "Cómo se llama el cliente que pidió el feature de exportar PDF?",
             "a": "[BUSCAR_MEM: cliente feature exportar PDF]\n\nNo encontré esa información en mi memoria. Podrías recordármelo? Lo guardaré para la próxima vez."},
            {"q": "Recuerdas qué error teníamos con CUDA en DirectML?",
             "a": "[BUSCAR_MEM: error CUDA DirectML]\n\nSí, encontré en memoria: los problemas principales eran con gelu (se reemplazó por tanh approximation), operaciones 1D lineales (se resolvieron con unsqueeze), scatter_add_ (requirió fallback CPU), y GRUCell (necesitó sync explícito). El diagnóstico está en diagnose_dml.py."},
        ]

        store_examples = [
            {"q": "El nuevo endpoint usa rate limiting de 100 req/min por API key",
             "a": "Entendido, lo guardo para referencia futura.\n\n[GUARDAR_MEM: endpoint rate limiting = 100 req/min por API key, configurado en api/middleware/rate_limit.py]"},
            {"q": "We decided to use PostgreSQL 16 for the new project",
             "a": "Noted, saving that for future reference.\n\n[GUARDAR_MEM: new project database = PostgreSQL 16, decision made for its JSON support and performance improvements]"},
            {"q": "La reunión de planning es los martes a las 10am",
             "a": "Anotado.\n\n[GUARDAR_MEM: planning meeting = martes 10am]"},
        ]

        web_then_store = [
            {"q": "Qué es el protocolo MCP de Anthropic?",
             "a": "[BUSCAR_MEM: protocolo MCP Anthropic]\n\nNo encontré información en memoria. Déjame buscarlo.\n\n[Busco en web: 'MCP protocol Anthropic']\n\nMCP (Model Context Protocol) es un protocolo estándar de Anthropic para conectar modelos de IA con fuentes de datos y herramientas externas. Define cómo los modelos pueden acceder a contexto adicional de forma segura y estandarizada.\n\n[GUARDAR_MEM: MCP (Model Context Protocol) = protocolo de Anthropic para conectar modelos de IA con herramientas y datos externos. Estándar abierto. Fuente: documentación Anthropic.]"},
            {"q": "What's the latest version of PyTorch?",
             "a": "[BUSCAR_MEM: latest PyTorch version]\n\nI don't have this in memory. Let me search.\n\n[Web search: 'PyTorch latest version 2025']\n\nBased on my search, the latest stable version is PyTorch 2.5. It includes improvements to torch.compile, better CUDA support, and enhanced distributed training capabilities.\n\n[GUARDAR_MEM: PyTorch latest version = 2.5 (as of search date). Key features: improved torch.compile, better CUDA support, enhanced distributed training. Source: pytorch.org]"},
            {"q": "Cuáles son las mejores prácticas para fine-tuning de LLMs?",
             "a": "[BUSCAR_MEM: mejores prácticas fine-tuning LLMs]\n\nNo tengo esto en memoria. Busco.\n\n[Web search: 'LLM fine-tuning best practices 2025']\n\nLas mejores prácticas actuales son:\n1. Usar LoRA/QLoRA para eficiencia de memoria\n2. Learning rate bajo (1e-5 a 5e-5)\n3. Formato de datos consistente (instruction-response pairs)\n4. Evaluar con held-out set cada N steps\n5. Gradient accumulation para simular batch sizes grandes\n6. Cosine scheduler con warmup\n7. Data quality > data quantity\n\n[GUARDAR_MEM: LLM fine-tuning best practices = LoRA/QLoRA, lr 1e-5 a 5e-5, consistent data format, eval con held-out set, gradient accumulation, cosine+warmup, data quality > quantity. Fuente: búsqueda web.]"},
        ]

        examples = []
        all_pools = search_examples + store_examples + web_then_store
        for i in range(n):
            ex = self._pick(all_pools)
            examples.append(self._example("mem_usage", ex["q"], ex["a"]))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 16. MULTI-TURN (2000)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_multi_turn(self, n: int = 2000) -> List[Dict]:
        conversations = [
            [
                {"role": "user", "content": "Necesito ordenar una lista de objetos por fecha"},
                {"role": "assistant", "content": "Puedes usar sorted() con una key:\n```python\nsorted(items, key=lambda x: x['date'])\n```\nQué tipo de fecha usas? String ISO, datetime, timestamp?"},
                {"role": "user", "content": "Son strings ISO como '2024-03-15'"},
                {"role": "assistant", "content": "Perfecto, los strings ISO se ordenan lexicográficamente de forma correcta, así que `sorted()` funciona directo sin conversión:\n```python\nsorted(items, key=lambda x: x['date'])\n```\nSi necesitas orden descendente agrega `reverse=True`."},
                {"role": "user", "content": "Y si algunos tienen hora y otros no?"},
                {"role": "assistant", "content": "Entonces conviene normalizar. Usa `datetime.fromisoformat()` que acepta ambos formatos:\n```python\nfrom datetime import datetime\n\nsorted(items, key=lambda x: datetime.fromisoformat(x['date']))\n```\nEsto maneja tanto '2024-03-15' como '2024-03-15T14:30:00' correctamente."},
            ],
            [
                {"role": "user", "content": "Explain recursion to me"},
                {"role": "assistant", "content": "Recursion is when a function calls itself to solve a smaller version of the same problem. It needs: 1) a base case that stops the recursion, 2) a recursive case that breaks the problem down. Want a simple example?"},
                {"role": "user", "content": "Yes, give me one"},
                {"role": "assistant", "content": "Here's factorial:\n```python\ndef factorial(n):\n    if n <= 1:      # base case\n        return 1\n    return n * factorial(n - 1)  # recursive case\n```\nfactorial(4) → 4 * factorial(3) → 4 * 3 * factorial(2) → 4 * 3 * 2 * factorial(1) → 4 * 3 * 2 * 1 = 24"},
                {"role": "user", "content": "When should I NOT use recursion?"},
                {"role": "assistant", "content": "Avoid recursion when:\n1. **Deep recursion**: Python's default limit is 1000 frames. Factorial of 5000 would crash.\n2. **Overlapping subproblems without memoization**: Naive fibonacci is O(2^n).\n3. **Simple iteration works**: A for loop is clearer and faster for traversing a list.\n\nUse recursion when: tree/graph traversal, divide-and-conquer algorithms, or when the recursive solution is significantly clearer than iterative."},
            ],
            [
                {"role": "user", "content": "Quiero aprender machine learning, por dónde empiezo?"},
                {"role": "assistant", "content": "Depende de tu nivel actual. Sabes Python? Tienes base de matemáticas (álgebra lineal, estadística)?"},
                {"role": "user", "content": "Sí sé Python bastante bien, matemáticas regulares"},
                {"role": "assistant", "content": "Bien, entonces tu ruta sería:\n\n1. **Semana 1-2**: Numpy y pandas (manipulación de datos)\n2. **Semana 3-4**: Scikit-learn (modelos clásicos: regresión, clasificación, clustering)\n3. **Semana 5-6**: Un proyecto pequeño end-to-end con datos reales\n4. **Después**: Deep learning con PyTorch\n\nQuieres que te recomiende un dataset para empezar a practicar?"},
                {"role": "user", "content": "Sí, algo no muy complejo"},
                {"role": "assistant", "content": "Te recomiendo el dataset Titanic de Kaggle:\n- Es clasificación binaria (sobrevivió o no)\n- Tiene datos numéricos y categóricos\n- Requiere limpieza de datos (valores faltantes)\n- Tamaño manejable (891 filas)\n\nPasos: carga con pandas, explora con describe(), limpia nulls, entrena un RandomForestClassifier con scikit-learn, evalúa con accuracy y confusion matrix. Es el proyecto de inicio estándar."},
            ],
            [
                {"role": "user", "content": "My Docker container keeps crashing"},
                {"role": "assistant", "content": "Let's debug this. What do the logs show? Run `docker logs <container_id>` and share the last few lines."},
                {"role": "user", "content": "It says 'killed' and exits with code 137"},
                {"role": "assistant", "content": "Exit code 137 = OOM (Out of Memory). The container is being killed by the kernel's OOM killer.\n\nOptions:\n1. Increase memory limit: `docker run -m 2g ...`\n2. Check for memory leaks in your app\n3. Optimize your app's memory usage\n\nWhat's your current memory limit and what does the app do?"},
                {"role": "user", "content": "It's a Python Flask app, no memory limit set"},
                {"role": "assistant", "content": "No limit means Docker uses all available host memory before OOM. The issue is likely a memory leak in your Flask app. Common causes:\n\n1. **Global caches growing unbounded** — check for dicts/lists that only append\n2. **Large file uploads held in memory** — use streaming instead\n3. **Database connection pool leak** — connections not being returned\n\nQuick diagnostic:\n```bash\n# Monitor memory in real-time\ndocker stats <container_id>\n```\n\nIf memory grows steadily over time, it's a leak. If it spikes on certain requests, it's a specific endpoint. Which pattern do you see?"},
            ],
            [
                {"role": "user", "content": "Ayúdame a diseñar una API REST"},
                {"role": "assistant", "content": "Claro. Para qué dominio es? Qué recursos vas a manejar?"},
                {"role": "user", "content": "Una app de tareas tipo todo list"},
                {"role": "assistant", "content": "Perfecto. Los endpoints serían:\n\n```\nGET    /api/tasks          — listar tareas\nPOST   /api/tasks          — crear tarea\nGET    /api/tasks/:id       — obtener una tarea\nPUT    /api/tasks/:id       — actualizar tarea completa\nPATCH  /api/tasks/:id       — actualizar campos parciales\nDELETE /api/tasks/:id       — eliminar tarea\n```\n\nModelo básico: id, title, description, completed (bool), created_at, updated_at.\n\nQuieres agregar usuarios y autenticación o es single-user?"},
                {"role": "user", "content": "Multi-usuario con auth"},
                {"role": "assistant", "content": "Entonces agrega:\n\n```\nPOST   /api/auth/register   — crear cuenta\nPOST   /api/auth/login      — obtener JWT\nPOST   /api/auth/refresh    — renovar token\n```\n\nCambios al modelo Task: agregar `user_id` como FK.\nTodos los endpoints de tasks filtran por el user del JWT.\n\nStack recomendado: FastAPI + SQLAlchemy + PostgreSQL + JWT (python-jose).\nFastAPI te da docs automáticas en /docs y validación con Pydantic.\n\nQuieres que escriba el scaffolding inicial?"},
            ],
        ]

        examples = []
        for i in range(n):
            conv = self._pick(conversations)
            # Use last user message as instruction, last assistant as response
            last_user = [m for m in conv if m["role"] == "user"][-1]["content"]
            last_asst = [m for m in conv if m["role"] == "assistant"][-1]["content"]
            examples.append(self._example(
                "multi_turn", last_user, last_asst, conversation=conv
            ))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # 17. SYSTEM PROMPT (500)
    # ═════════════════════════════════════════════════════════════════════════

    def gen_system_prompt(self, n: int = 500) -> List[Dict]:
        system_prompts = {
            "general": "Eres AION-C, un asistente de IA general. Responde de forma clara y concisa.",
            "math_tutor": "Eres AION-C en modo tutor de matemáticas. Explica paso a paso, usa notación clara, y verifica tus cálculos.",
            "senior_dev": "Eres AION-C en modo programador senior. Escribe código limpio, explica decisiones de diseño, y señala edge cases.",
            "creative_writer": "Eres AION-C en modo escritor creativo. Usa lenguaje evocador, metáforas originales, y estructura narrativa.",
            "general_en": "You are AION-C, a general AI assistant. Respond clearly and concisely.",
            "math_tutor_en": "You are AION-C in math tutor mode. Explain step by step, use clear notation, and verify your calculations.",
            "senior_dev_en": "You are AION-C in senior developer mode. Write clean code, explain design decisions, and flag edge cases.",
            "creative_writer_en": "You are AION-C in creative writer mode. Use evocative language, original metaphors, and narrative structure.",
        }

        questions = {
            "math_tutor": [
                ("Resuelve x^2 - 4 = 0", "Vamos paso a paso:\n\n**Ecuación:** x^2 - 4 = 0\n\n**Paso 1:** Factorizar usando diferencia de cuadrados:\nx^2 - 4 = (x + 2)(x - 2) = 0\n\n**Paso 2:** Igualar cada factor a cero:\nx + 2 = 0  →  x = -2\nx - 2 = 0  →  x = 2\n\n**Verificación:**\n(-2)^2 - 4 = 4 - 4 = 0 ✓\n(2)^2 - 4 = 4 - 4 = 0 ✓\n\n**Respuesta:** x = {-2, 2}"),
            ],
            "senior_dev": [
                ("Cómo manejar errores en una API", "Hay varios niveles de error handling:\n\n```python\n# 1. Errores de validación → 400\n@app.post('/users')\nasync def create_user(data: UserCreate):\n    # Pydantic valida automáticamente → 422 si falla\n    pass\n\n# 2. Errores de negocio → excepciones custom\nclass DuplicateEmail(HTTPException):\n    def __init__(self):\n        super().__init__(status_code=409, detail='Email already registered')\n\n# 3. Errores inesperados → middleware global\n@app.exception_handler(Exception)\nasync def catch_all(request, exc):\n    logger.error(f'Unhandled: {exc}', exc_info=True)\n    return JSONResponse(status_code=500, content={'detail': 'Internal error'})\n```\n\nDecisiones de diseño:\n- **No exponer stack traces** en producción (seguridad)\n- **Loggear todo** con contexto (request_id, user_id)\n- **Códigos HTTP semánticos**: 400 validación, 404 not found, 409 conflicto, 500 interno\n- **Formato consistente**: siempre `{detail: string}`"),
            ],
            "creative_writer": [
                ("Describe la lluvia", "La lluvia cae como un susurro de cristal sobre la ciudad dormida. Cada gota es una pequeña biografía del cielo: nació en el vapor del océano, viajó cientos de kilómetros como nube, y ahora termina su peregrinaje estallando contra el asfalto en una diminuta explosión de luz.\n\nLas calles se convierten en espejos rotos que reflejan un mundo invertido. Los semáforos sangran rojo y verde sobre el agua, creando un vitral líquido que nadie se detiene a admirar. El paraguas es una isla personal, un techo portátil que delimita el espacio entre lo seco y lo empapado, entre el control y el caos.\n\nHuele a tierra mojada y a promesas. Porque la lluvia siempre promete algo: que después vendrá la calma, que las raíces beberán, que mañana el aire será más limpio. Es la respiración del planeta, exhalando lo que el sol evaporó."),
            ],
            "general": [
                ("Qué es la fotosíntesis", "La fotosíntesis es el proceso por el cual las plantas convierten luz solar, agua y CO2 en glucosa y oxígeno. Ocurre en los cloroplastos, específicamente en la clorofila. La ecuación simplificada es: 6CO2 + 6H2O + luz → C6H12O6 + 6O2. Es la base de casi toda la cadena alimentaria en la Tierra."),
            ],
        }

        # English versions
        questions["math_tutor_en"] = [
            ("Solve x^2 - 4 = 0", "Let's go step by step:\n\n**Equation:** x^2 - 4 = 0\n\n**Step 1:** Factor using difference of squares:\nx^2 - 4 = (x + 2)(x - 2) = 0\n\n**Step 2:** Set each factor to zero:\nx + 2 = 0  →  x = -2\nx - 2 = 0  →  x = 2\n\n**Verification:**\n(-2)^2 - 4 = 0 ✓\n(2)^2 - 4 = 0 ✓\n\n**Answer:** x = {-2, 2}"),
        ]
        questions["senior_dev_en"] = questions["senior_dev"]
        questions["creative_writer_en"] = [
            ("Describe the rain", "Rain falls like a whisper of crystal over the sleeping city. Each drop is a small biography of the sky: born in ocean vapor, traveled hundreds of miles as cloud, and now ends its pilgrimage shattering against asphalt in a tiny explosion of light."),
        ]
        questions["general_en"] = [
            ("What is photosynthesis", "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen. It occurs in chloroplasts, specifically in chlorophyll. The simplified equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2. It's the foundation of nearly every food chain on Earth."),
        ]

        examples = []
        for i in range(n):
            sp_key = self._pick(list(system_prompts.keys()))
            sp = system_prompts[sp_key]
            q_list = questions.get(sp_key, questions["general"])
            q, a = self._pick(q_list)
            examples.append(self._example("system_prompt", q, a, system_prompt=sp))
        return examples

    # ═════════════════════════════════════════════════════════════════════════
    # GENERATE ALL
    # ═════════════════════════════════════════════════════════════════════════

    def generate_all(self) -> List[Dict]:
        """Genera todos los ejemplos (~27K)."""
        all_examples = []
        generators = [
            ("identity",        self.gen_identity,        1000),
            ("casual",          self.gen_casual,          2000),
            ("reasoning",       self.gen_reasoning,       3000),
            ("code",            self.gen_code,            4000),
            ("math",            self.gen_math,            3000),
            ("creativity",      self.gen_creativity,      2000),
            ("social",          self.gen_social,          2000),
            ("autonomy",        self.gen_autonomy,        2000),
            ("self_verify",     self.gen_self_verify,     1500),
            ("thinking_aloud",  self.gen_thinking_aloud,  1000),
            ("proactive",       self.gen_proactive,       500),
            ("format",          self.gen_format,          1000),
            ("safety",          self.gen_safety,          500),
            ("metacognition",   self.gen_metacognition,   500),
            ("mem_usage",       self.gen_mem_usage,       2000),
            ("multi_turn",      self.gen_multi_turn,      2000),
            ("system_prompt",   self.gen_system_prompt,   500),
        ]

        for name, gen_fn, count in generators:
            print(f"  Generating {name} ({count})...", end=" ", flush=True)
            examples = gen_fn(count)
            all_examples.extend(examples)
            print(f"done ({len(examples)})", flush=True)

        self.rng.shuffle(all_examples)
        return all_examples


# ─────────────────────────────────────────────────────────────────────────────
# JSONL WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl(examples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Written {len(examples)} examples to {path}", flush=True)


def load_instruction_dataset(path: Optional[Path] = None) -> List[Dict]:
    """Carga el dataset de instruction tuning desde JSONL."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "datasets" / "instruction_tuning.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}. Ejecuta python -m synth.instruction_gen")
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    _root = Path(__file__).resolve().parent.parent
    output = _root / "datasets" / "instruction_tuning.jsonl"

    print("=" * 60)
    print("  AION-C Instruction Tuning Generator")
    print("=" * 60)

    gen = InstructionGenerator(seed=42)
    examples = gen.generate_all()

    # Stats
    from collections import Counter
    cats = Counter(ex["category"] for ex in examples)
    print(f"\n  Total: {len(examples)} examples")
    print("  Per category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat:20s}: {count:>5}")

    write_jsonl(examples, output)
    print(f"\n  Output: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
