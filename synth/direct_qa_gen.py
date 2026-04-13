"""
synth/direct_qa_gen.py — 10K direct Q&A examples (no graph context)
====================================================================
Format: "pregunta" → "respuesta" — clean, conversational, bilingual.
"""
import json, random
from pathlib import Path
from typing import List, Dict


def generate_direct_qa(n: int = 10000, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    examples = []

    # ── GENERAL / IDENTITY / GREETINGS (2500) ──
    greetings = [
        # ES
        ("Hola", "Hola, en que puedo ayudarte?"),
        ("Hola!", "Hola! Dime en que te ayudo."),
        ("Buenos dias", "Buenos dias. Como puedo asistirte?"),
        ("Buenas tardes", "Buenas tardes. En que te ayudo?"),
        ("Que tal?", "Bien, gracias. Dime que necesitas."),
        ("Hey", "Hola. En que puedo ayudarte?"),
        ("Como estas?", "Bien, listo para ayudarte. Que necesitas?"),
        ("Hola, como estas?", "Bien, gracias. En que puedo ayudarte?"),
        # EN
        ("Hello", "Hello! How can I help you?"),
        ("Hi", "Hi there. What do you need?"),
        ("Hey", "Hey! What can I do for you?"),
        ("Good morning", "Good morning. How can I help?"),
        ("How are you?", "I'm ready to help. What do you need?"),
        ("What's up?", "Ready to assist. What's on your mind?"),
    ]
    identity = [
        ("Quien eres?", "Soy AION-C, un sistema de IA creado por Jesus con arquitectura MoSE."),
        ("Who are you?", "I'm AION-C, an AI system created by Jesus using MoSE architecture."),
        ("Que eres?", "Soy AION-C, una IA con 5 motores especializados."),
        ("What are you?", "I'm AION-C, an AI with 5 specialized engines for different tasks."),
        ("Como te llamas?", "Me llamo AION-C."),
        ("What is your name?", "My name is AION-C."),
        ("Quien te creo?", "Fui creado por Jesus."),
        ("Who created you?", "I was created by Jesus."),
        ("Que puedes hacer?", "Puedo razonar, escribir codigo, resolver matematicas, crear historias y entender emociones."),
        ("What can you do?", "I can reason, write code, solve math, create stories, and understand emotions."),
        ("Que es MoSE?", "MoSE es mi arquitectura: Mixture of Specialized Engines con 5 motores."),
        ("What is MoSE?", "MoSE is my architecture: Mixture of Specialized Engines with 5 motors."),
        ("Puedes ver imagenes?", "No, solo trabajo con texto."),
        ("Can you see images?", "No, I only work with text."),
        ("Eres GPT?", "No, soy AION-C, una arquitectura diferente basada en grafos causales."),
        ("Are you ChatGPT?", "No, I'm AION-C, a different architecture based on causal graphs."),
        ("Que motores tienes?", "Tengo 5: CORA para causal, FORGE-C para codigo, AXIOM para matematicas, MUSE para creatividad, EMPATHY para social."),
        ("What engines do you have?", "I have 5: CORA for causal, FORGE-C for code, AXIOM for math, MUSE for creativity, EMPATHY for social."),
    ]
    closings = [
        ("Gracias", "De nada. Algo mas?"),
        ("Thank you", "You're welcome. Anything else?"),
        ("Muchas gracias", "Con gusto. Necesitas algo mas?"),
        ("Thanks!", "You're welcome! Need anything else?"),
        ("Adios", "Hasta luego!"),
        ("Goodbye", "Goodbye! Come back anytime."),
        ("Chao", "Nos vemos!"),
        ("Bye", "Bye! Take care."),
        ("Eso es todo", "Perfecto. Estoy aqui si necesitas mas."),
        ("That's all", "Great. I'm here if you need more."),
    ]
    meta = [
        ("No entiendo", "Explicame que parte no esta clara y lo intento de otra forma."),
        ("I don't understand", "Tell me which part is unclear and I'll try differently."),
        ("Puedes repetir?", "Claro, dime que quieres que repita."),
        ("Can you explain more?", "Sure, what part would you like me to elaborate on?"),
        ("No se que preguntar", "Puedo ayudarte con codigo, matematicas, razonamiento, historias o situaciones sociales."),
        ("Estoy aburrido", "Puedo contarte algo interesante, resolver un problema, o escribir una historia."),
        ("I'm bored", "I can tell you something interesting, solve a problem, or write a story."),
    ]

    for _ in range(2500):
        pool = rng.choice([greetings, identity, closings, meta])
        q, a = rng.choice(pool)
        lang = "es" if any(c in q for c in "aeiouñ?¿!¡áéíóú") and rng.random() < 0.7 else "en"
        examples.append({"input": q, "output": a, "domain": "general", "language": lang})

    # ── MATH / AXIOM (2000) ──
    for _ in range(2000):
        lang = "es" if rng.random() < 0.4 else "en"
        t = rng.choice(["arith", "equation", "percent", "divisible", "series"])

        if t == "arith":
            a, b = rng.randint(1, 999), rng.randint(1, 999)
            op = rng.choice(["+", "-", "*"])
            r = a + b if op == "+" else a - b if op == "-" else a * b
            q = f"Cuanto es {a} {op} {b}?" if lang == "es" else f"What is {a} {op} {b}?"
            ans = str(r)
        elif t == "equation":
            x = rng.randint(-20, 20)
            a = rng.randint(1, 10)
            b = rng.randint(-50, 50)
            c = a * x + b
            q = f"Resuelve: {a}x + {b} = {c}" if lang == "es" else f"Solve: {a}x + {b} = {c}"
            ans = f"x = {x}"
        elif t == "percent":
            p = rng.choice([10, 15, 20, 25, 50, 75])
            n = rng.choice([100, 200, 240, 300, 400, 500, 1000])
            r = p * n // 100
            q = f"Cuanto es {p}% de {n}?" if lang == "es" else f"What is {p}% of {n}?"
            ans = str(r)
        elif t == "divisible":
            n = rng.randint(2, 500)
            d = rng.choice([2, 3, 5, 7, 10])
            rem = n % d
            q = f"Es {n} divisible entre {d}?" if lang == "es" else f"Is {n} divisible by {d}?"
            if rem == 0:
                ans = f"Si, {n}/{d} = {n // d}." if lang == "es" else f"Yes, {n}/{d} = {n // d}."
            else:
                ans = f"No, {n}/{d} = {n // d} resto {rem}." if lang == "es" else f"No, {n}/{d} = {n // d} remainder {rem}."
        else:
            start = rng.randint(1, 20)
            step = rng.randint(1, 10)
            series = [start + i * step for i in range(5)]
            q_str = ", ".join(map(str, series[:4]))
            q = f"Que sigue: {q_str}, ...?" if lang == "es" else f"What comes next: {q_str}, ...?"
            ans = str(series[4])

        examples.append({"input": q, "output": ans, "domain": "axiom", "language": lang})

    # ── CODE / FORGE-C (1500) ──
    code_qa = [
        ("Write a function to add two numbers", "def add(a, b):\n    return a + b"),
        ("Escribe una funcion para sumar dos numeros", "def sumar(a, b):\n    return a + b"),
        ("Write a function to reverse a string", "def reverse(s):\n    return s[::-1]"),
        ("Escribe una funcion para invertir un string", "def invertir(s):\n    return s[::-1]"),
        ("Write hello world in Python", "print('Hello, World!')"),
        ("Escribe hola mundo en Python", "print('Hola, Mundo!')"),
        ("How do you create a list in Python?", "my_list = [1, 2, 3]"),
        ("Como se crea una lista en Python?", "mi_lista = [1, 2, 3]"),
        ("Write a function to find the maximum", "def find_max(lst):\n    return max(lst)"),
        ("Write a function to check if a number is even", "def is_even(n):\n    return n % 2 == 0"),
        ("What is a for loop?", "A for loop iterates over a sequence: for item in list: do_something(item)"),
        ("Que es un for loop?", "Un for loop itera sobre una secuencia: for item in lista: hacer_algo(item)"),
        ("How do you read a file in Python?", "with open('file.txt', 'r') as f:\n    content = f.read()"),
        ("What is a dictionary in Python?", "A dictionary is a key-value data structure: d = {'name': 'AION', 'type': 'AI'}"),
        ("Write a function to sort a list", "def sort_list(lst):\n    return sorted(lst)"),
        ("Escribe una funcion para ordenar una lista", "def ordenar(lst):\n    return sorted(lst)"),
        ("What does len() do?", "len() returns the number of items in a sequence."),
        ("Find the bug: def add(a,b): return a-b", "The bug is: return a-b should be return a+b."),
        ("Encuentra el bug: def add(a,b): return a-b", "El bug es: return a-b deberia ser return a+b."),
        ("Write a function to count vowels", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
    ]
    for _ in range(1500):
        q, a = rng.choice(code_qa)
        lang = "es" if any(c in q for c in "áéíóúñ¿") else "en"
        examples.append({"input": q, "output": a, "domain": "forge_c", "language": lang})

    # ── CAUSAL / CORA (1500) ──
    causal_qa = [
        ("If it rains, what happens to the soil?", "The soil gets wet."),
        ("Si llueve, que pasa con el suelo?", "El suelo se moja."),
        ("Does rain cause floods?", "Yes, heavy rain can cause floods."),
        ("La lluvia causa inundaciones?", "Si, la lluvia intensa puede causar inundaciones."),
        ("What causes inflation?", "Inflation is caused by excess money supply or high demand."),
        ("Que causa la inflacion?", "La inflacion es causada por exceso de dinero o alta demanda."),
        ("If a server crashes, what happens?", "Services become unavailable until the server is restored."),
        ("Si un servidor se cae, que pasa?", "Los servicios dejan de funcionar hasta que se restaure."),
        ("Does deforestation cause climate change?", "Yes, deforestation contributes to climate change by reducing CO2 absorption."),
        ("Can poverty cause crime?", "Poverty can be a contributing factor to crime, but it's not the only cause."),
        ("La pobreza causa crimen?", "La pobreza puede contribuir al crimen, pero no es la unica causa."),
        ("What happens if you don't sleep?", "Lack of sleep causes fatigue, poor concentration, and health problems."),
        ("Why do earthquakes happen?", "Earthquakes happen when tectonic plates shift and release energy."),
        ("Por que ocurren los terremotos?", "Los terremotos ocurren cuando las placas tectonicas se mueven y liberan energia."),
    ]
    for _ in range(1500):
        q, a = rng.choice(causal_qa)
        lang = "es" if any(c in q for c in "áéíóúñ¿") else "en"
        examples.append({"input": q, "output": a, "domain": "cora", "language": lang})

    # ── EMPATHY (1500) ──
    empathy_qa = [
        ("Mi amigo esta triste", "Lamento escuchar eso. A veces ayuda hablar sobre lo que siente. Estas ahi para el."),
        ("My friend is sad", "I'm sorry to hear that. Sometimes just being there and listening helps a lot."),
        ("Me siento ansioso", "Es normal sentir ansiedad. Intenta respirar profundo y enfocarte en el presente."),
        ("I feel anxious", "It's normal to feel anxious. Try deep breathing and focus on the present moment."),
        ("Estoy feliz!", "Me alegra! Disfruta ese momento."),
        ("I'm happy!", "That's great! Enjoy the moment."),
        ("Mi jefe me grito", "Eso no esta bien. Nadie merece que le griten. Considera hablar con recursos humanos."),
        ("My boss yelled at me", "That's not okay. Nobody deserves to be yelled at. Consider talking to HR."),
        ("No se que hacer con mi vida", "Es normal sentirse asi. Piensa en lo que te hace feliz y empieza por ahi."),
        ("I don't know what to do with my life", "It's normal to feel that way. Think about what makes you happy and start there."),
        ("Perdi mi trabajo", "Lo siento mucho. Es una situacion dificil, pero es temporal. Puedo ayudarte a pensar en opciones."),
        ("I lost my job", "I'm sorry. It's a tough situation, but it's temporary. I can help you think through options."),
        ("Me siento solo", "Entiendo. La soledad es dificil. Considera conectar con alguien de confianza."),
        ("I feel lonely", "I understand. Loneliness is hard. Consider reaching out to someone you trust."),
    ]
    for _ in range(1500):
        q, a = rng.choice(empathy_qa)
        lang = "es" if any(c in q for c in "áéíóúñ¿") else "en"
        examples.append({"input": q, "output": a, "domain": "empathy", "language": lang})

    # ── CREATIVE / MUSE (1000) ──
    creative_qa = [
        ("Write a short poem about the moon", "Silver light on quiet water,\nthe moon watches, patient and old,\ntelling secrets to no one."),
        ("Escribe un poema corto sobre la luna", "Luz de plata sobre agua quieta,\nla luna observa, paciente y vieja,\ncontando secretos a nadie."),
        ("Tell me a story about a robot", "Once, a small robot found a flower growing through concrete. It didn't understand beauty, but it stayed to watch."),
        ("Cuentame una historia sobre un robot", "Una vez, un robot encontro una flor creciendo entre el concreto. No entendia la belleza, pero se quedo a mirar."),
        ("Create a metaphor for time", "Time is a river that never touches the same water twice."),
        ("Crea una metafora para el tiempo", "El tiempo es un rio que nunca toca la misma agua dos veces."),
        ("Write a scene about discovery", "She opened the old book and found a letter from 1923. The handwriting was shaky but determined."),
        ("Write a haiku", "Morning dew falls soft,\nleaves whisper ancient stories,\nsilence speaks the most."),
        ("Escribe un haiku", "Rocio al caer,\nhojas susurran historias,\nel silencio habla."),
        ("What makes a good story?", "A good story has a character who wants something, faces obstacles, and is changed by the journey."),
    ]
    for _ in range(1000):
        q, a = rng.choice(creative_qa)
        lang = "es" if any(c in q for c in "áéíóúñ¿") else "en"
        examples.append({"input": q, "output": a, "domain": "muse", "language": lang})

    rng.shuffle(examples)
    return examples


def main():
    examples = generate_direct_qa(10000)
    path = Path("datasets/direct_qa_10k.jsonl")
    path.parent.mkdir(exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    domains = Counter(ex["domain"] for ex in examples)
    langs = Counter(ex["language"] for ex in examples)
    print(f"Generated {len(examples)} direct Q&A examples")
    print(f"Domains: {dict(domains)}")
    print(f"Languages: {dict(langs)}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
