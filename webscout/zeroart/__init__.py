"""
ZeroArt: A zero-dependency ASCII art text generator

Create awesome ASCII art text without external dependencies!
"""

from typing import Dict, List, Literal, Optional, Union
from .base import ZeroArtFont
from .fonts import BlockFont, SlantFont, NeonFont, CyberFont
from .effects import AsciiArtEffects

FontType = Literal['block', 'slant', 'neon', 'cyber']

def figlet_format(text: str, font: Union[str, ZeroArtFont] = 'block') -> str:
    """
    Generate ASCII art text
    
    :param text: Text to convert
    :param font: Font style (default: 'block')
    :return: ASCII art representation of text
    """
    font_map: Dict[str, ZeroArtFont] = {
        'block': BlockFont(),
        'slant': SlantFont(),
        'neon': NeonFont(),
        'cyber': CyberFont()
    }
    
    if isinstance(font, str):
        selected_font: ZeroArtFont = font_map.get(font.lower(), BlockFont())
    else:
        selected_font = font
    return selected_font.render(text)

def print_figlet(text: str, font: Union[str, ZeroArtFont] = 'block') -> None:
    """
    Print ASCII art text directly
    
    :param text: Text to convert and print
    :param font: Font style (default: 'block')
    """
    print(figlet_format(text, font))

# Expose additional effects
rainbow = AsciiArtEffects.rainbow_effect
glitch = AsciiArtEffects.glitch_effect
wrap_text = AsciiArtEffects.wrap_text
outline = AsciiArtEffects.outline_effect

__all__ = [
    'figlet_format', 
    'print_figlet', 
    'rainbow', 
    'glitch', 
    'wrap_text', 
    'outline',
    'BlockFont', 
    'SlantFont', 
    'NeonFont', 
    'CyberFont',
    'ZeroArtFont',
    'FontType'
]