"""
synth/knowledge/language_patterns.py — Patrones reales de español e inglés.

Gramática básica bilingüe: tiempos verbales, diferencias clave entre
los dos idiomas, modismos, "false friends". Verificable en libros de
texto estándar.
"""

from __future__ import annotations

from typing import List, Dict


_EN_GRAMMAR = [
    ("present simple", "describes habits, facts, and general truths",
     "describe hábitos, hechos y verdades generales",
     "I work, she works, they work. Use 's' for third person singular. Example: 'The sun rises in the east'."),
    ("present continuous", "describes actions happening right now or around now",
     "describe acciones que suceden ahora o alrededor de ahora",
     "am/is/are + -ing. 'I am working on a project this week'. Used for temporary situations."),
    ("past simple", "describes completed actions at a definite time in the past",
     "describe acciones completadas en un tiempo definido en el pasado",
     "Regular verbs add -ed: walked, worked. Irregular verbs memorized: went, saw, ate. 'I visited Paris in 2019'."),
    ("present perfect", "connects a past action to the present",
     "conecta una acción pasada con el presente",
     "have/has + past participle. 'I have lived here for 5 years'. Used for experiences and recent changes with present relevance."),
    ("past perfect", "describes an action completed before another past action",
     "describe una acción completada antes que otra pasada",
     "had + past participle. 'When I arrived, she had already left'. Used to show sequence in the past."),
    ("future simple", "describes future actions or predictions",
     "describe acciones futuras o predicciones",
     "will + base form. 'It will rain tomorrow'. Also 'going to' for plans: 'I'm going to start exercising'."),
    ("modal verbs", "auxiliary verbs expressing ability, possibility, permission, obligation",
     "verbos auxiliares que expresan habilidad, posibilidad, permiso, obligación",
     "can, could, may, might, must, should, would, will. Followed by base form of main verb. 'You should sleep more'."),
    ("conditional", "hypothetical situations with if-clauses",
     "situaciones hipotéticas con cláusulas if",
     "Zero: 'If you heat water, it boils'. First: 'If it rains, I will stay'. Second: 'If I had time, I would read'. Third: 'If I had known, I would have helped'."),
    ("passive voice", "subject receives the action instead of performing it",
     "el sujeto recibe la acción en lugar de realizarla",
     "be + past participle. 'The book was written by her'. Used when the doer is unknown, unimportant, or the focus is on the action."),
    ("gerund", "a verb form used as a noun",
     "una forma verbal usada como sustantivo",
     "-ing form. 'Swimming is fun'. Used after certain verbs: enjoy, avoid, finish. Different from present participle (same form)."),
    ("infinitive", "the base form of a verb, usually with 'to'",
     "la forma base de un verbo, usualmente con 'to'",
     "to + base form. 'I want to learn'. Used after some verbs: want, hope, decide. Some verbs take gerund, some take infinitive."),
    ("article", "'a', 'an', or 'the' that precedes a noun",
     "'a', 'an', o 'the' que precede a un sustantivo",
     "'the' is definite (specific). 'a/an' is indefinite (any one). 'an' before vowel sounds. Uncountables often have no article."),
    ("countable vs uncountable nouns", "nouns that can or cannot be counted individually",
     "sustantivos que pueden o no contarse individualmente",
     "Countable: one apple, two apples. Uncountable: water, information. Use 'much' with uncountables, 'many' with countables."),
    ("subject-verb agreement", "the verb must agree in number with the subject",
     "el verbo debe concordar en número con el sujeto",
     "Singular subject takes singular verb: 'She runs'. Plural: 'They run'. Collective nouns can be tricky: 'the team is' or 'the team are'."),
    ("relative clause", "a clause that gives more information about a noun",
     "una cláusula que da más información sobre un sustantivo",
     "Uses who, whom, whose, which, that. 'The book that I read was great'. Restrictive (no commas) vs non-restrictive (with commas)."),
    ("phrasal verb", "a verb combined with a preposition or adverb to create a new meaning",
     "un verbo combinado con preposición o adverbio para crear un nuevo significado",
     "pick up, look after, give up. Often the meaning cannot be guessed from the parts. Very common in everyday English."),
    ("question tag", "a short question added to the end of a statement",
     "una pequeña pregunta añadida al final de una afirmación",
     "'You're coming, aren't you?' Positive statement + negative tag; negative statement + positive tag. Used to seek confirmation."),
    ("idioms", "fixed expressions with meanings that differ from the literal",
     "expresiones fijas con significados que difieren del literal",
     "'Kick the bucket' means to die. 'Break a leg' means good luck. Idioms must be memorized; they rarely translate directly."),
]


_ES_GRAMMAR = [
    ("presente de indicativo", "expresses habitual actions, facts, or current situations",
     "expresa acciones habituales, hechos o situaciones actuales",
     "Conjugations for -ar, -er, -ir verbs. Hablo, comes, viven. 'Ella vive en Madrid'. Used for generalizations and habits."),
    ("pretérito perfecto simple", "the simple past tense for completed actions",
     "el pretérito simple para acciones completadas",
     "Hablé, comiste, vivieron. Used for specific moments in the past: 'Ayer fui al cine'. Different from pretérito perfecto compuesto."),
    ("pretérito imperfecto", "the past tense for ongoing or habitual past actions",
     "el pasado para acciones continuas o habituales",
     "Hablaba, comías, vivían. 'Cuando era niño, jugaba en el parque'. Used for descriptions and continuous past actions."),
    ("pretérito perfecto compuesto", "the present perfect tense, auxiliary haber plus past participle",
     "el pretérito perfecto, auxiliar haber más participio",
     "He hablado, has comido, han vivido. Used for recent events with present relevance: 'Hoy he visto a Juan'."),
    ("subjuntivo", "mood used for doubt, desire, hypotheticals",
     "modo usado para duda, deseo, hipótesis",
     "Triggered by ojalá, quiero que, es posible que. 'Quiero que vengas'. Different tenses: presente, imperfecto, perfecto, pluscuamperfecto."),
    ("ser vs estar", "two verbs that both mean 'to be' but are not interchangeable",
     "dos verbos que significan 'to be' pero no son intercambiables",
     "Ser for permanent characteristics, identity: 'Soy médico'. Estar for states, location, temporary conditions: 'Estoy cansado'."),
    ("por vs para", "two prepositions that are often translated as 'for'",
     "dos preposiciones que a menudo se traducen como 'for'",
     "Por: cause, duration, exchange. 'Gracias por tu ayuda'. Para: purpose, destination, recipient. 'Este regalo es para ti'."),
    ("género de sustantivos", "every Spanish noun is masculine or feminine",
     "cada sustantivo español es masculino o femenino",
     "Mostly -o is masculine and -a is feminine, but many exceptions: el día, la mano, el problema. Articles and adjectives agree in gender."),
    ("concordancia de adjetivos", "adjectives agree in gender and number with the noun",
     "los adjetivos concuerdan en género y número con el sustantivo",
     "Niño alto, niña alta, niños altos, niñas altas. Adjectives usually follow the noun in Spanish: 'una casa grande'."),
    ("pronombres personales", "I, you, he, she, we, they in Spanish",
     "yo, tú, él, ella, nosotros, ellos en español",
     "Yo, tú, él/ella/usted, nosotros/as, vosotros/as, ellos/ellas/ustedes. Subject pronouns are often omitted (pro-drop language)."),
    ("gustar", "the verb 'to please' used for expressing likes",
     "el verbo gustar usado para expresar gustos",
     "Structurally inverted from English: 'Me gusta el café' literally means 'coffee pleases me'. Verb agrees with the thing liked, not the liker."),
    ("reflexivos", "verbs where the subject and object are the same",
     "verbos donde sujeto y objeto son el mismo",
     "Me lavo, te levantas, se despierta. Reflexive pronouns precede conjugated verbs or attach to infinitives/gerunds: 'quiero levantarme'."),
    ("acentos", "written accents mark stress on certain syllables",
     "los acentos escritos marcan el énfasis en ciertas sílabas",
     "Rules: words ending in vowel, n, or s stress the penultimate; others the last. Accent marks breaks of the rule: canción, árbol."),
    ("imperativo", "the command form of verbs",
     "la forma imperativa de los verbos",
     "¡Come!, ¡Habla!, ¡Vengan! Affirmative uses a special form; negative uses the present subjunctive: 'No hables', 'No comas'."),
    ("voseo", "use of vos instead of tú in parts of Latin America",
     "uso de vos en lugar de tú en partes de Latinoamérica",
     "Common in Argentina, Uruguay, parts of Central America. Has distinct verb forms: 'vos tenés' vs 'tú tienes'. Regional standard."),
]


_FALSE_FRIENDS = [
    ("embarrassed vs embarazada", "in English 'embarrassed' means ashamed, in Spanish 'embarazada' means pregnant",
     "'embarrassed' en inglés significa avergonzado, 'embarazada' en español significa pregnant",
     "A classic false cognate. 'I am embarrassed' translates to 'estoy avergonzado/a', not 'estoy embarazada'. Causes famous translation mistakes."),
    ("actual vs actual", "in Spanish 'actual' means current, in English it means real",
     "en español 'actual' significa current, en inglés significa real",
     "Spanish: 'el actual presidente' = 'the current president'. English: 'actual cost' = 'el coste real'. Beware the mismatch."),
    ("library vs librería", "'library' is a place to borrow books, 'librería' is a bookstore",
     "'library' es un lugar para pedir prestado libros, 'librería' es una tienda de libros",
     "The Spanish for 'library' is 'biblioteca'. The English for 'librería' is 'bookstore'. They swap meanings."),
    ("success vs suceso", "'success' means a favorable outcome, 'suceso' means an event",
     "'success' significa un resultado favorable, 'suceso' significa un evento",
     "Spanish: 'un suceso trágico' = 'a tragic event'. English: 'her success' = 'su éxito'. False friend."),
    ("assist vs asistir", "'assist' means to help, 'asistir' can mean to attend",
     "'assist' significa ayudar, 'asistir' puede significar atender",
     "Spanish: 'asistir a la reunión' = 'to attend the meeting'. English 'assist' is more limited to the helping sense."),
    ("exit vs éxito", "'exit' means a way out, 'éxito' means success",
     "'exit' significa una salida, 'éxito' significa success",
     "False cognate. English 'exit' → Spanish 'salida'. Spanish 'éxito' → English 'success'. The letters overlap but the meanings diverge."),
    ("constipated vs constipado", "'constipated' refers to bowel issues, 'constipado' means having a cold",
     "'constipated' se refiere a problemas intestinales, 'constipado' significa resfriado",
     "Very embarrassing mistake. Spanish 'estoy constipado' means 'I have a cold', not what English speakers think."),
    ("sensible vs sensible", "'sensible' in English means practical, in Spanish it means sensitive",
     "'sensible' en inglés significa práctico, en español significa sensitive",
     "A practical person in Spanish is 'sensato/a'. An emotionally sensitive person in English is 'sensitive'."),
    ("argument vs argumento", "'argument' can be a dispute, 'argumento' is a line of reasoning or a plot",
     "'argument' puede ser una disputa, 'argumento' es una línea de razonamiento o un argumento de obra",
     "Spanish 'argumento de la película' = 'plot of the movie'. English 'we had an argument' = 'tuvimos una discusión'."),
    ("realize vs realizar", "'realize' means to become aware, 'realizar' means to carry out",
     "'realize' significa darse cuenta, 'realizar' significa llevar a cabo",
     "Spanish 'realizar una tarea' = 'to perform a task'. English 'I realize' = 'me doy cuenta'. Don't translate literally."),
]


def facts() -> List[Dict]:
    out = []
    for name, en_short, es_short, detail in _EN_GRAMMAR:
        out.append({
            "topic": "language",
            "subtopic": "english_grammar",
            "q_en": f"What is the {name} in English grammar?",
            "a_en": f"The {name} {en_short}. {detail}",
            "q_es": f"¿Qué es el {name} en gramática inglesa?",
            "a_es": f"El {name} {es_short}. {detail}",
            "difficulty": "medium",
        })
    for name, en_short, es_short, detail in _ES_GRAMMAR:
        out.append({
            "topic": "language",
            "subtopic": "spanish_grammar",
            "q_en": f"What is the {name} in Spanish grammar?",
            "a_en": f"The {name} {en_short}. {detail}",
            "q_es": f"¿Qué es el {name} en gramática española?",
            "a_es": f"El {name} {es_short}. {detail}",
            "difficulty": "medium",
        })
    for name, en_short, es_short, detail in _FALSE_FRIENDS:
        out.append({
            "topic": "language",
            "subtopic": "false_friends",
            "q_en": f"Explain the false cognate {name}.",
            "a_en": f"{name}: {en_short}. {detail}",
            "q_es": f"Explica el falso cognado {name}.",
            "a_es": f"{name}: {es_short}. {detail}",
            "difficulty": "medium",
        })
    return out
