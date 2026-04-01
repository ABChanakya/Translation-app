"""
Honorifics Preservation Module
Detects and preserves Japanese honorifics in translations
"""
import re
from typing import List, Tuple, Optional, Dict, Set


class HonorificPreserver:
    """
    Detects and preserves Japanese honorifics in translated text
    """
    
    # Common Japanese honorifics
    DEFAULT_HONORIFICS = {
        # Name suffixes
        'san': '～さん',
        'kun': '～くん',
        'chan': '～ちゃん',
        'sama': '～様',
        'dono': '～殿',
        'senpai': '先輩',
        'kohai': '後輩',
        'sensei': '先生',
        'shi': '～氏',
        
        # Family/relationship terms
        'onii': 'お兄',
        'onee': 'お姉',
        'oniisan': 'お兄さん',
        'oneesan': 'お姉さん',
        'oniichan': 'お兄ちゃん',
        'oneechan': 'お姉ちゃん',
        'oniisama': 'お兄様',
        'oneesama': 'お姉様',
        
        'otouto': '弟',
        'imouto': '妹',
        'otoutokun': '弟くん',
        'imoutochan': '妹ちゃん',
        
        'okaasan': 'お母さん',
        'oka': 'お母',
        'kaasan': '母さん',
        'okaasama': 'お母様',
        'haha': '母',
        'hahaue': '母上',
        
        'otousan': 'お父さん',
        'otou': 'お父',
        'tousan': '父さん',
        'otousama': 'お父様',
        'chichi': '父',
        'chichiue': '父上',
        
        'ojiisan': 'おじいさん',
        'obaasan': 'おばあさん',
        'ojisan': 'おじさん',
        'obasan': 'おばさん',
        
        # Other terms
        'danna': '旦那',
        'dannasama': '旦那様',
        'goshujin': 'ご主人',
        'goshujinsama': 'ご主人様',
        'ojou': 'お嬢',
        'ojousama': 'お嬢様',
        'bocchama': '坊ちゃま',
        'ouji': '王子',
        'ojisama': '王子様',
        'hime': '姫',
        'himesama': '姫様',
    }
    
    def __init__(
        self,
        custom_honorifics: Optional[Dict[str, str]] = None,
        preserve_all: bool = True
    ):
        """
        Initialize honorific preserver
        
        Args:
            custom_honorifics: Additional honorifics dict (romanji -> Japanese)
            preserve_all: If True, preserve all default honorifics
        """
        self.honorifics = self.DEFAULT_HONORIFICS.copy()
        if custom_honorifics:
            self.honorifics.update(custom_honorifics)
        
        self.preserve_all = preserve_all
        
        # Create regex patterns for detection
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        # Sort by length (longest first) to match longer honorifics first
        sorted_honorifics = sorted(
            self.honorifics.keys(),
            key=len,
            reverse=True
        )
        
        # Pattern to detect honorifics (case-insensitive)
        # Matches: name-san, name san, namesan
        self.honorific_pattern = re.compile(
            r'\b(\w+?)[\s\-]?(' + '|'.join(re.escape(h) for h in sorted_honorifics) + r')\b',
            re.IGNORECASE
        )
        
        # Pattern to detect standalone honorifics
        self.standalone_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(h) for h in sorted_honorifics) + r')\b',
            re.IGNORECASE
        )
    
    def detect_honorifics(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Detect honorifics in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples: (full_match, honorific, start_pos, end_pos)
        """
        matches = []
        
        # Find name + honorific patterns
        for match in self.honorific_pattern.finditer(text):
            full_match = match.group(0)
            name = match.group(1)
            honorific = match.group(2).lower()
            
            matches.append((
                full_match,
                honorific,
                match.start(),
                match.end()
            ))
        
        return matches
    
    def preserve_in_translation(
        self,
        original_text: str,
        translated_text: str,
        keep_romanji: bool = True
    ) -> str:
        """
        Preserve honorifics from original text in translation
        
        Args:
            original_text: Original Japanese text
            translated_text: Translated English text
            keep_romanji: If True, keep romanji honorifics; if False, use Japanese
            
        Returns:
            Translated text with honorifics preserved
        """
        # Detect honorifics in original
        detected = self.detect_honorifics(original_text)
        
        if not detected:
            return translated_text
        
        # Build mapping of names to honorifics
        name_honorifics = {}
        for full_match, honorific, _, _ in detected:
            # Extract name (everything before honorific)
            name_match = re.match(r'(\w+?)[\s\-]?' + re.escape(honorific), full_match, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
                name_honorifics[name.lower()] = honorific
        
        # Apply honorifics to translated text
        result = translated_text
        
        for name, honorific in name_honorifics.items():
            # Find name in translation (case-insensitive)
            name_pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
            
            # Determine suffix format
            if keep_romanji:
                suffix = f"-{honorific}"
            else:
                suffix = self.honorifics.get(honorific, f"-{honorific}")
            
            # Replace name with name+honorific
            def replace_with_honorific(match):
                matched_name = match.group(0)
                # Preserve original capitalization
                if matched_name[0].isupper():
                    return matched_name + suffix
                return matched_name + suffix
            
            result = name_pattern.sub(replace_with_honorific, result)
        
        return result
    
    def add_honorifics_post_translation(
        self,
        translated_text: str,
        character_honorifics: Dict[str, str]
    ) -> str:
        """
        Add honorifics to character names in translation based on mapping
        
        Args:
            translated_text: Translated text
            character_honorifics: Dict mapping character names to their honorifics
                                 Example: {"Naruto": "kun", "Sakura": "chan"}
            
        Returns:
            Text with honorifics added
        """
        result = translated_text
        
        for character, honorific in character_honorifics.items():
            # Find character name (case-insensitive, word boundary)
            pattern = re.compile(r'\b' + re.escape(character) + r'\b', re.IGNORECASE)
            
            # Check if honorific already present
            if not re.search(pattern.pattern + r'[\s\-]?' + re.escape(honorific), result, re.IGNORECASE):
                # Add honorific
                result = pattern.sub(lambda m: m.group(0) + f"-{honorific}", result)
        
        return result
    
    def extract_character_honorifics(
        self,
        dialogue_log: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Extract character-honorific mappings from dialogue log
        
        Args:
            dialogue_log: List of dialogue entries with 'src_text' field
            
        Returns:
            Dict mapping character names (romanji) to their most common honorific
        """
        character_honorifics = {}
        honorific_counts = {}
        
        for entry in dialogue_log:
            src_text = entry.get('src_text', '')
            detected = self.detect_honorifics(src_text)
            
            for full_match, honorific, _, _ in detected:
                # Extract name
                name_match = re.match(r'(\w+?)[\s\-]?' + re.escape(honorific), full_match, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1).capitalize()
                    
                    # Count occurrences
                    key = (name, honorific)
                    honorific_counts[key] = honorific_counts.get(key, 0) + 1
        
        # Select most common honorific for each character
        for (name, honorific), count in honorific_counts.items():
            if name not in character_honorifics:
                character_honorifics[name] = honorific
            else:
                # Keep most frequent
                current_key = (name, character_honorifics[name])
                if honorific_counts.get(current_key, 0) < count:
                    character_honorifics[name] = honorific
        
        return character_honorifics
    
    def suggest_honorific(
        self,
        character_name: str,
        context: str,
        gender_hint: Optional[str] = None
    ) -> Optional[str]:
        """
        Suggest appropriate honorific based on context
        
        Args:
            character_name: Character name
            context: Surrounding dialogue/context
            gender_hint: Optional gender hint ('male', 'female', 'neutral')
            
        Returns:
            Suggested honorific or None
        """
        context_lower = context.lower()
        
        # Detect relationship terms
        if any(term in context_lower for term in ['teacher', 'sensei', 'instructor']):
            return 'sensei'
        
        if any(term in context_lower for term in ['master', 'lord', 'lady']):
            return 'sama'
        
        if any(term in context_lower for term in ['senior', 'upperclassman']):
            return 'senpai'
        
        if any(term in context_lower for term in ['junior', 'underclassman']):
            return 'kohai'
        
        # Default based on gender
        if gender_hint == 'male':
            return 'kun'
        elif gender_hint == 'female':
            return 'chan'
        else:
            return 'san'  # Neutral default
    
    def validate_honorific_consistency(
        self,
        dialogue_log: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """
        Check for inconsistent honorific usage
        
        Args:
            dialogue_log: List of dialogue entries
            
        Returns:
            Dict mapping character names to list of different honorifics used
        """
        character_usage = {}
        
        for entry in dialogue_log:
            src_text = entry.get('src_text', '')
            detected = self.detect_honorifics(src_text)
            
            for full_match, honorific, _, _ in detected:
                name_match = re.match(r'(\w+?)[\s\-]?' + re.escape(honorific), full_match, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1).capitalize()
                    
                    if name not in character_usage:
                        character_usage[name] = set()
                    character_usage[name].add(honorific)
        
        # Return only characters with multiple honorifics
        inconsistent = {
            name: list(honorifics)
            for name, honorifics in character_usage.items()
            if len(honorifics) > 1
        }
        
        return inconsistent
