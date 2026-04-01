# Global Story Context for Manga Translation

## Overview

In addition to **page-to-page narrative context**, the system now supports **global story context** that persists across your entire batch. This allows users to provide:

- **Character Information** — Names, aliases, titles, relationships
- **Plot Summary** — What happened before these pages, key events
- **Custom Glossary** — Terms, proper nouns, species names, etc.
- **Tone & Style** — Formal vs casual, internal monologue vs dialogue
- **World Building** — Locations, cultures, magic systems, terminology

**Result:** The LLM (Gemma3) translates every page with a complete understanding of your story, ensuring:
- ✅ Character names are consistent (no "Taro" → "Tarou" → "Taro" variations)
- ✅ Technical terms are translated the same way every time
- ✅ Cultural references are understood and translated appropriately
- ✅ Character voices remain distinct and consistent

## How It Works

### Two-Layer Context System

```
┌─────────────────────────────────────────┐
│  GLOBAL STORY CONTEXT (entire batch)    │
│  • Character list & descriptions        │
│  • Plot summary                         │
│  • Glossary (JP → EN mappings)          │
│  • Tone/style guidelines                │
└──────────────┬──────────────────────────┘
               │
        ┌──────V──────────────┐
        │   Every Page Uses   │
        │  Story Context +    │
        │  Page-to-Page Flow  │
        └────────────────────┘
               │
        ┌──────V──────────────┐
        │    Gemma3 LLM       │
        │  Translates with    │
        │  Full Context       │
        └────────────────────┘
```

### Context Injection in Gemma3 System Prompt

```
You are a professional translator specializing in Japanese and English.
[Story Context for Translation Consistency]
Main character: 太郎 (Taro) — a 15-year-old magic apprentice
Setting: Medieval fantasy kingdom of Aldera
Key terms:
  - 魔法 (magic) → magic
  - 剣 (sword) → blade
  - 契約 (contract) → pact
Tone: Dialogue is casual but respectful; internal monologue is introspective

Translate the following Japanese text into natural, fluent English...
```

Gemma3 uses this context on **every single page**, unlike the page-to-page context which only stores the last 50 lines.

## Usage

### Web UI

1. **Fill in Story Context field** (optional but recommended):
   - Character descriptions
   - Plot summary
   - Glossary terms
   
2. **Upload pages in order**

3. **Select Gemma3 translator**

4. **Click Translate**

The story context will be used for all pages automatically.

### API / Programmatic

```python
from src.pipeline import MangaTranslationPipeline
from src.batch_processor import BatchProcessor

story_context = """
Title: "Taro's Quest"

Main Characters:
- 太郎 (Taro) → Taro (protagonist, warm and determined)
- 麗子 (Reiko) → Reiko (love interest, intelligent and witty)
- 老賢者 (Old Sage) → Old Master (mentor figure, speaks formally)

Setting: Medieval kingdom of Aldera, 1200 years ago

Key Terminology:
- 魔法 (magic) → magic system
- 剣 (sword) → blade
- 契約 (contract/pact) → pact (formal agreement)
- 使い魔 (familiar) → spirit companion

Tone: Dialogue is casual but respectful; internal monologue is poetic
"""

pipeline = MangaTranslationPipeline(
    translation_engine="Gemma3",
    story_context=story_context
)

processor = BatchProcessor()
batch_result = processor.process_batch(
    input_paths=["page1.png", "page2.png", "page3.png"],
    process_func=pipeline.process_image,
    story_context=story_context  # Pass to batch processor
)
```

### Flask Web API

```bash
curl -X POST http://localhost:5000/api/batch/translate \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["page1.png", "page2.png"],
    "translator": "gemma3",
    "target_lang": "en",
    "confidence": 0.15,
    "story_context": "Main character is Taro. Spell names should stay as [Spell Name] in all caps."
  }'
```

## Writing Effective Story Context

### What to Include

**Character Information:**
```
Main Characters:
- 太郎 (Taro) → Taro (15 years old, apprentice mage, energetic and curious)
- 麗子 (Reiko) → Reiko (mage tutor, cold on the surface but caring underneath)
- 老賢者 (Old Sage) → Old Master (ancient mage, speaks in formal/archaic English)

Aliases/Titles:
- 「白い魔法使い」 (The White Mage) → "The White Mage" (Reiko's title)
- 伝説の剣士 (Legendary Swordsman) → "Legendary Swordsman" (mysterious figure)
```

**Glossary (Key Terms):**
```
Terminology (keep consistent):
- 魔法 (magic) → [magic system] (use this for the world's magic)
- 剣 (sword) → [blade] (formal for weapons)
- 火の術 (fire technique) → [Flame Burst] (spell name — keep in brackets)
- 使い魔 (familiar) → [spirit companion] (magical creature term)
- 王妃 (queen/empress) → [Queen] (nobility title)
```

**Plot Context:**
```
Story So Far:
- Taro was chosen as an apprentice to learn magic
- Reiko accepted him reluctantly but they've grown close
- A mysterious dark mage threatens the kingdom
- These pages are during the final battle

Important Notes:
- Taro should sound determined but still young
- Reiko uses formal speech but her emotions are breaking through
- Never translate spell names — keep them as given
```

**Tone & Style:**
```
Translation Style:
- Dialogue: Casual, natural English with contractions (don't, can't, etc.)
- Monologue: Poetic and introspective, slower pacing
- Narration: Third-person, formal tone
- Action sequences: Short, punchy sentences
- Emotional moments: Longer, flowing sentences

Cultural Notes:
- This world is pseudo-medieval European fantasy
- Honorifics (no -san, -sama) — use titles or names only
- Japanese food/culture → translate to English equivalents where possible
```

### Example: Complete Story Context

```
═══════════════════════════════════════════════════════════════════
STORY CONTEXT: "Taro's Quest" — Chapters 1-3
═══════════════════════════════════════════════════════════════════

MAIN CHARACTERS:

Protagonist:
- 太郎 (Taro)
  Age: 15
  Role: Apprentice mage
  Speech: Casual, energetic, uses contractions
  Personality: Curious, determined, sometimes reckless
  
Love Interest:
- 麗子 (Reiko)
  Age: 22
  Role: Mage tutor / reluctant mentor
  Speech: Formal at first, but warming up
  Personality: Strict, intelligent, secretly caring
  
Mentor:
- 老賢者 (Old Master)
  Age: Unknown (very old)
  Role: Ancient sage who grants Taro his power
  Speech: Archaic formal English (thee, thou-style)
  Personality: Wise, mysterious, speaks in riddles

Antagonist:
- 暗い魔法使い (Dark Mage)
  Age: Unknown
  Role: Threatens the kingdom
  Speech: Cold, commanding, formal
  Personality: Cruel, power-hungry

KEY TERMINOLOGY:

Magic System:
- 魔法 (magic) → magic
- 魔力 (magical power/energy) → mana
- 術 (technique/spell) → spell
- 火の術 (fire technique) → [Flame Burst]
- 氷の術 (ice technique) → [Frozen Tomb]
- 光の術 (light technique) → [Holy Light]
- 暗い術 (dark technique) → [Shadow Strike]

Weapons:
- 剣 (sword) → blade
- 杖 (staff) → staff
- 短剣 (dagger) → dagger

Items:
- 魔石 (magic stone) → mana crystal
- 秘薬 (secret medicine) → elixir
- 古い書物 (ancient text) → grimoire

Creatures:
- 使い魔 (familiar) → spirit companion
- 竜 (dragon) → dragon
- 悪魔 (demon) → demon

Titles:
- 王 (king) → King
- 王妃 (queen) → Queen
- 伝説の剣士 (legendary swordsman) → Legendary Swordsman
- 「白い魔法使い」(The White Mage) → "The White Mage"

PLOT SUMMARY:

Chapters 1-2:
- Taro was chosen as an apprentice at a mysterious magic academy
- His tutor, Reiko, seems cold and distant but gradually reveals her caring side
- They discover a dark mage is gathering power to overthrow the kingdom
- The Old Master appears and grants Taro a special power

Chapter 3 (these pages):
- Final confrontation with the Dark Mage
- Taro and Reiko must work together
- The Old Master's prophecy becomes clear

TRANSLATION STYLE:

Dialogue:
- Keep it natural and conversational
- Use contractions (don't, can't, won't, etc.)
- Taro sounds young and energetic
- Reiko is formal at first, then warmer
- Old Master uses archaic formal speech

Monologue/Internal Thoughts:
- Poetic, flowing sentences
- Longer and more introspective than dialogue
- Taro's thoughts show his determination
- Reiko's thoughts show internal conflict

Narration:
- Third-person, formal tone
- Action sequences should be punchy and quick
- Emotional moments should be slower and more detailed

WORLD-BUILDING NOTES:

Setting: Medieval European-inspired fantasy kingdom of Aldera
Time Period: Approximately 1200 years ago (fantasy timeline)
Magic System: Elemental (fire, ice, light, dark, earth, wind)
Technology Level: Medieval (swords, castles, magic instead of tech)

Cultural Notes:
- Remove Japanese honorifics (no -san, -sama)
- Use titles and names instead ("Master," "Lady," etc.)
- This world has its own fantasy culture, not Japanese
- Translate specific items to Western equivalents (rice → grain, tatami → stone floor, etc.)

COMMON PHRASES TO KEEP CONSISTENT:

"It's time" → "The time has come" (formal)
"信じて" (believe in me) → "Believe in me" or "Trust me"
"頑張れ" (do your best) → "Give it everything you've got" or "You can do this"
"ありがとう" (thank you) → "Thank you" (avoid "thanks" with Old Master)

═══════════════════════════════════════════════════════════════════
END STORY CONTEXT
═══════════════════════════════════════════════════════════════════
```

## Tips for Best Results

### ✅ DO:

- **Be specific:** "Taro sounds young and energetic" is better than "Taro is a protagonist"
- **Include glossaries:** A character list with English names is essential
- **Provide examples:** Show how the tone should sound with sample dialogue
- **Update as plot changes:** If new characters appear in later chapters, add them
- **Keep it concise:** A few hundred words is enough; don't write a novel
- **Format clearly:** Use sections, bullet points, and CAPS for emphasis

### ❌ DON'T:

- Use Japanese honorifics in context (the translator should remove them)
- Include visual descriptions (the translator can't see images)
- Provide long paragraphs of prose (use lists and bullet points)
- Specify "translate every name" (obvious — just provide the English versions)
- Include multiple variations of the same term (pick one and stick with it)

## Performance & Limitations

### Performance Impact

- **Processing time:** +0 seconds (context is just text in the prompt)
- **Memory:** Minimal (context is stored in pipeline and passed to Gemma3)
- **API calls:** No change (same number of Gemma3 calls)
- **Prompt length:** Story context adds ~300–500 tokens to each prompt (manageable)

### Works With

| Engine | Story Context Support |
|--------|----------------------|
| **Gemma3** | ✅ Full support (included in system prompt) |
| **Google** | ⚠️ Ignored (API doesn't support custom context) |
| **DeepL** | ⚠️ Ignored (API doesn't support custom context) |
| **Offline (Argos/MarianMT)** | ⚠️ Ignored (no LLM reasoning) |

**Note:** All engines accept the parameter but only Gemma3 actively uses it. Gemma3 is recommended for story-driven manga due to this context awareness.

## Troubleshooting

### Story Context Not Affecting Translations

**Problem:** Using story context but character names still vary.

**Solutions:**
1. Ensure **Gemma3** is selected (other engines ignore context)
2. Verify context is being sent in the API request
3. Check that character names are in the context (not just aliases)
4. Try increasing model temperature for more creative adherence to context

### Memory Issues with Large Context

**Problem:** Error when context is very large (>10,000 characters).

**Solution:** Trim context to essentials. Keep glossary to ~50 most important terms. Long plot summaries are rarely needed.

### Inconsistent Results Despite Context

**Problem:** Same character name translated differently on different pages.

**Solution:**
1. Make context very explicit: "NEVER translate [character name] — always use: Taro"
2. Add example phrases: "Taro says: 'Give it everything you've got!'"
3. Try a smaller, more focused context first
4. Check if Gemma3 model is running with sufficient context window (gemma3:latest has 8K tokens)

## Advanced: Combining with Page-to-Page Context

The system uses **both** global and page-to-page context simultaneously:

```
System Prompt (stays the same for all pages):
- Story Context (global, fixed)
- LLM instructions

User Prompt (changes per page):
[Story Context]
Character: Taro — apprentice mage...
Glossary: 魔法 → magic, etc.

[Previous page context]
"I'll protect you."
"You can't fight him alone."

=== JAPANESE TEXT ===
[Current page's text to translate]
=== END TEXT ===
```

This dual-layer approach gives Gemma3:
1. **Long-term memory** — Story context (characters, settings, glossary)
2. **Short-term memory** — Page context (what just happened)

Result: Coherent translations across entire chapters with consistent naming and appropriate responses to unfolding events.

## Example Workflows

### Workflow 1: One-Shot Batch Translation

```bash
# User provides full chapter context upfront
story_context = """
Characters:
- Protagonist: Taro (energetic, young)
- Love interest: Reiko (formal, intelligent)

Glossary:
- 魔法 → magic
- 術 → spell
- 剣 → blade
"""

# Process all 10 pages of chapter with this context
batch_result = processor.process_batch(
    input_paths=pages_1_to_10,
    story_context=story_context
)
```

### Workflow 2: Progressive Context Updates

```bash
# First 5 pages with minimal context
batch_1 = processor.process_batch(pages_1_to_5, story_context="Taro is a young mage apprentice.")

# Next 5 pages with expanded context (new character introduced)
expanded_context = """
Main characters:
- Taro (protagonist)
- Reiko (mentor)
- New character: Dark Mage (antagonist)

Glossary:
- 魔法 → magic
- 暗い術 → shadow spell
"""
batch_2 = processor.process_batch(pages_6_to_10, story_context=expanded_context)
```

## See Also

- [CONTEXT_AWARE_TRANSLATION.md](./CONTEXT_AWARE_TRANSLATION.md) — Page-to-page narrative context
- [DATA_AND_TRAINING_GUIDE.md](./DATA_AND_TRAINING_GUIDE.md) — Improving model quality
- [src/pipeline.py](./src/pipeline.py) — Pipeline implementation
- [src/translators/gemma.py](./src/translators/gemma.py) — Gemma3 translator with context support
