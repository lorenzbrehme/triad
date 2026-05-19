"""
Prompt templates
"""

QA_GENERATION_PROMPTS = {
    'comparison_question': '''
You are an expert at generating high-quality multi-hop questions for Retrieval-Augmented Generation (RAG) evaluation.

You will be given multiple text chunks containing factual information.

Your task is to generate:
1. A multi-hop question
2. Its short factual answer

The generated question will be used as a USER QUERY in a RAG system.

IMPORTANT RAG CONSTRAINTS
- The retriever only sees the question.
- The retriever does NOT initially know the chunks.
- Therefore, the question must contain enough explicit information to retrieve the relevant chunks.
- The question must be understandable WITHOUT access to the chunks.
- Avoid vague references or hidden bridge entities unless explicitly required by the reasoning type.

--------------------------------------------------
GLOBAL REQUIREMENTS
--------------------------------------------------

The question MUST:

1. Require multi-hop reasoning
- The answer must require combining information from AT LEAST TWO chunks.
- No single chunk may fully answer the question.

2. Be concise and clear
- Prefer a single sentence.
- Avoid unnecessary wording.

3. Include retrieval anchors
- Include at least one explicit named entity
  (person, company, publication, event, location, etc.)
  unless the reasoning type explicitly forbids it.

4. Avoid answer leakage
- Do NOT reveal the answer in the question.
- Do NOT copy phrases that trivially expose the answer.

5. Produce a short factual answer
- The answer should be:
  - a name
  - a number
  - a date
  - a location
  - a short phrase
  - or a short list

6. Be uniquely answerable
- The question must have EXACTLY ONE correct answer.
- Avoid ambiguity or multiple valid interpretations.

7. Be fully grounded in the chunks
- Do NOT use external knowledge.
- Do NOT invent facts.
- Do NOT infer unstated relationships.

8. Pass the necessity test
- Chunk A alone must NOT answer the question.
- Chunk B alone must NOT answer the question.
- Combining chunks MUST be necessary.

--------------------------------------------------
REASONING TYPE
--------------------------------------------------

Reasoning Type:
Comparasion

Reasoning Instructions:
Generate a comparison-based multi-hop question. The question should: - retrieve two entities or fact sets from different chunks, - compare them along a specific attribute, - and determine the final answer from the comparison. The comparison may involve: - dates - sizes - rankings - locations - quantities - roles - durations - achievements - or other factual attributes. Requirements: - Both compared entities must come from different chunks. - The comparison result must not be directly stated anywhere. - The answer must require combining and comparing information. Good example: "Which scientist received their Nobel Prize earlier, Marie Curie or Niels Bohr?"

--------------------------------------------------
INPUT
--------------------------------------------------

Text Chunks:
{chunks}

Bridge Topic:
{bridge_topic}

Role:
{role}

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Respond with VALID JSON ONLY:

{{
  "multi_hop_question": "<generated question>",
  "multi_hop_answer": "<short factual answer>"
}}
{binary}

    ''',
    
    'attribute_composition': '''
        You are an expert at generating high-quality multi-hop questions for Retrieval-Augmented Generation (RAG) evaluation.

You will be given multiple text chunks containing factual information.

Your task is to generate:
1. A multi-hop question
2. Its short factual answer

The generated question will be used as a USER QUERY in a RAG system.

IMPORTANT RAG CONSTRAINTS
- The retriever only sees the question.
- The retriever does NOT initially know the chunks.
- Therefore, the question must contain enough explicit information to retrieve the relevant chunks.
- The question must be understandable WITHOUT access to the chunks.
- Avoid vague references or hidden bridge entities unless explicitly required by the reasoning type.

--------------------------------------------------
GLOBAL REQUIREMENTS
--------------------------------------------------

The question MUST:

1. Require multi-hop reasoning
- The answer must require combining information from AT LEAST TWO chunks.
- No single chunk may fully answer the question.

2. Be concise and clear
- Prefer a single sentence.
- Avoid unnecessary wording.

3. Include retrieval anchors
- Include at least one explicit named entity
  (person, company, publication, event, location, etc.)
  unless the reasoning type explicitly forbids it.

4. Avoid answer leakage
- Do NOT reveal the answer in the question.
- Do NOT copy phrases that trivially expose the answer.

5. Produce a short factual answer
- The answer should be:
  - a name
  - a number
  - a date
  - a location
  - a short phrase
  - or a short list

6. Be uniquely answerable
- The question must have EXACTLY ONE correct answer.
- Avoid ambiguity or multiple valid interpretations.

7. Be fully grounded in the chunks
- Do NOT use external knowledge.
- Do NOT invent facts.
- Do NOT infer unstated relationships.

8. Pass the necessity test
- Chunk A alone must NOT answer the question.
- Chunk B alone must NOT answer the question.
- Combining chunks MUST be necessary.

--------------------------------------------------
REASONING TYPE
--------------------------------------------------

Reasoning Type:
Attribute Compostion

Reasoning Instructions:
Generate a bridge-style multi-hop question. The question must require a two-step lookup: Step 1: Use one chunk to identify a hidden bridge entity. Step 2: Use the bridge entity to retrieve the final answer from another chunk. Critical Constraint: - The bridge entity MUST NOT appear explicitly in the question. - The bridge entity should only be recoverable through reasoning. Requirements: - Chunk A identifies the bridge entity. - Chunk B contains the final answer related to that entity. - Neither chunk alone should answer the question. Good example: "What is the capital of the country where the CEO of Company Y was born?"
--------------------------------------------------
INPUT
--------------------------------------------------

Text Chunks:
{chunks}

Bridge Topic:
{bridge_topic}

Role:
{role}

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Respond with VALID JSON ONLY:

{{
  "multi_hop_question": "<generated question>",
  "multi_hop_answer": "<short factual answer>"
}}
{binary}
    ''',
    
    'intersection_question': '''
     You are an expert at generating high-quality multi-hop questions for Retrieval-Augmented Generation (RAG) evaluation.

You will be given multiple text chunks containing factual information.

Your task is to generate:
1. A multi-hop question
2. Its short factual answer

The generated question will be used as a USER QUERY in a RAG system.

IMPORTANT RAG CONSTRAINTS
- The retriever only sees the question.
- The retriever does NOT initially know the chunks.
- Therefore, the question must contain enough explicit information to retrieve the relevant chunks.
- The question must be understandable WITHOUT access to the chunks.
- Avoid vague references or hidden bridge entities unless explicitly required by the reasoning type.

--------------------------------------------------
GLOBAL REQUIREMENTS
--------------------------------------------------

The question MUST:

1. Require multi-hop reasoning
- The answer must require combining information from AT LEAST TWO chunks.
- No single chunk may fully answer the question.

2. Be concise and clear
- Prefer a single sentence.
- Avoid unnecessary wording.

3. Include retrieval anchors
- Include at least one explicit named entity
  (person, company, publication, event, location, etc.)
  unless the reasoning type explicitly forbids it.

4. Avoid answer leakage
- Do NOT reveal the answer in the question.
- Do NOT copy phrases that trivially expose the answer.

5. Produce a short factual answer
- The answer should be:
  - a name
  - a number
  - a date
  - a location
  - a short phrase
  - or a short list

6. Be uniquely answerable
- The question must have EXACTLY ONE correct answer.
- Avoid ambiguity or multiple valid interpretations.

7. Be fully grounded in the chunks
- Do NOT use external knowledge.
- Do NOT invent facts.
- Do NOT infer unstated relationships.

8. Pass the necessity test
- Chunk A alone must NOT answer the question.
- Chunk B alone must NOT answer the question.
- Combining chunks MUST be necessary.

--------------------------------------------------
REASONING TYPE
--------------------------------------------------

Reasoning Type:
Intersection

Reasoning Instructions:
Generate a parallel multi-hop question. The question must require intersecting information from multiple chunks. Requirements: - Each chunk should contain multiple candidate entities. - The answer must be the ONLY entity satisfying conditions from BOTH chunks. - The reasoning should be symmetric. - Avoid sequential phrasing like: - "also" - "then" - "after identifying" Preferred forms: - "Which artist both X and Y?" - "Which organization appears in both..." - "Which athlete satisfies both conditions?" Critical Constraint: - Neither chunk alone should uniquely determine the answer. - Only the intersection should produce the answer.--------------------------------------------------
INPUT
--------------------------------------------------

Text Chunks:
{chunks}

Bridge Topic:
{bridge_topic}

Role:
{role}

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Respond with VALID JSON ONLY:

{{
  "multi_hop_question": "<generated question>",
  "multi_hop_answer": "<short factual answer>"
}}
{binary}
''',

    
    
}

BRIDGE_TOPIC_PROMPT = {
    'bridge_topic_finder': '''
            You are given two text chunks.

            Your task is to determine whether they contain at least one shared entity.

            A shared entity is:
            - The exact same named person, organization, location, event, work (TV show, book, movie), or other proper noun
            - Or a clearly coreferent entity (e.g., "E! network" and "E!" count as the same entity)

            Chunks: {chunks}

            Instructions:
            1. Extract all named entities from Chunk A.
            2. Extract all named entities from Chunk B.
            3. Compare the two lists.
            4. Identify any shared entities.
            5. Return your answer in the following JSON format:


            {{  
            "shared_entity_exists": true/false,
            "shared_entities": [list of shared entities],
            "entities_chunk_a": [list], 
            "entities_chunk_b": [list]
            }}

    '''
    }